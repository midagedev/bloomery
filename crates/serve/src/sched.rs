//! Who gets a slot: the server's N slots (`--parallel N`), the requests
//! waiting for one, and the bound on how many may wait (`--queue-depth Q`).
//!
//! One queue in arrival order holds every waiting request; a free slot takes
//! the oldest of them. When more requests wait than slots are free, a
//! [`SlotPicker`] chooses which one goes next ([`FifoPicker`], the oldest, by
//! default). The request then takes, of the free slots, the one whose held ids
//! share the longest prefix with its own, the least recently used among
//! equals (llama-server's choice, without its similarity floor): its cache
//! keeps the most. When the slots take one engine in turns, the slot whose
//! state the engine holds goes before the least recently used: taking
//! another moves that state aside.
//!
//! A slot action (save, restore, erase) reserves its slot, and a residency
//! reset every slot, only while the slot is free and no request waits, so
//! neither ever goes ahead of a waiting request. A request that would wait
//! past the queue's depth is refused at once, never queued.
//!
//! The board holds no engine: the HTTP threads read and change it under its
//! lock while the engine thread is inside a call, so `/health`, `/slots`,
//! `/metrics` and a refusal answer at once. Nor does it keep the ids a slot
//! holds: the engine thread's slots are their one owner, and it hands them to
//! [`Board::admit`] with the slot the engine is on, so a move of the engine's
//! states leaves no copy behind to steer a request to a slot that lost them.

use std::cmp::Reverse;
use std::collections::VecDeque;

use serde_json::Value;

use crate::api::SlotQueue;
use crate::promptcache::common_prefix;

/// What a [`SlotPicker`] sees of a waiting request.
#[derive(Clone, Copy, Debug)]
pub struct WaitingRequest<'a> {
    /// The prompt's ids.
    pub ids: &'a [u32],
}

/// What a [`SlotPicker`] sees of a slot.
#[derive(Clone, Copy, Debug)]
pub struct SlotSummary<'a> {
    pub id: usize,
    /// A request or an action holds it.
    pub busy: bool,
    /// The ids its cache holds.
    pub held: &'a [u32],
    /// The routed-expert counts of the sequence the slot runs, layer by layer
    /// and expert by expert, for a policy that groups requests by routing.
    /// Empty: no engine reports them yet.
    pub routing: &'a [f32],
}

/// Chooses which waiting request takes the next free slot when more requests
/// wait than slots are free.
pub trait SlotPicker: Send {
    /// An index into `waiting` (oldest first, never empty); `slots` is every
    /// slot, free and busy. An index past `waiting` stops the server by name.
    fn pick(&mut self, waiting: &[WaitingRequest<'_>], slots: &[SlotSummary<'_>]) -> usize;
}

/// The default picker: the oldest waiting request, arrival order.
#[derive(Clone, Copy, Debug, Default)]
pub struct FifoPicker;

impl SlotPicker for FifoPicker {
    fn pick(&mut self, _waiting: &[WaitingRequest<'_>], _slots: &[SlotSummary<'_>]) -> usize {
        0
    }
}

/// How many slots the server runs and how many requests may wait for one
/// ([`crate::Server::bind_with`]).
pub struct SlotConfig {
    /// `--parallel N`: slots, each with the engine's whole context. The
    /// engine must declare at least this many ([`crate::Engine::slots`]).
    pub parallel: usize,
    /// `--queue-depth Q`: requests that may wait past the free slots. `None`
    /// is every request that can wait at once under
    /// [`crate::MAX_CONNECTIONS`]: `MAX_CONNECTIONS - parallel`.
    pub queue_depth: Option<usize>,
    pub picker: Box<dyn SlotPicker>,
}

impl Default for SlotConfig {
    fn default() -> Self {
        SlotConfig {
            parallel: 1,
            queue_depth: None,
            picker: Box::new(FifoPicker),
        }
    }
}

/// The queue's depth when none is given: every request that can wait at once.
/// A waiting request holds a connection, and `parallel` running ones hold one
/// each, so no more than `max_connections - parallel` can ever wait: a deeper
/// queue never fills, and this one refuses nothing the connection limit admits.
#[must_use]
pub(crate) fn default_depth(max_connections: usize, parallel: usize) -> usize {
    max_connections.saturating_sub(parallel)
}

/// What a slot is doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Use {
    Free,
    /// A request runs on it.
    Running,
    /// An action (or a residency reset) holds it.
    Held,
}

/// What `/slots` shows of a slot's last or running request.
#[derive(Default)]
pub(crate) struct SlotView {
    pub id_task: u64,
    pub prompt: Value,
    pub settings: Value,
    /// The running request's `n_predict` (`-1` unbounded).
    pub n_predict: i64,
    pub n_past: usize,
    pub n_decoded: usize,
    pub stopped_eos: bool,
    pub stopped_word: bool,
    pub stopped_limit: bool,
    pub stopping_word: String,
}

pub(crate) struct SlotState {
    pub state: Use,
    /// The board's clock at the slot's last release; 0 never used.
    pub last_used: u64,
    pub routing: Vec<f32>,
    pub view: SlotView,
}

/// The slots an action reserves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reserve {
    One(usize),
    All,
}

/// Why a request or an action was not taken.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// The queue holds its depth past the free slots.
    Full { depth: usize },
    /// The slot is running a request or an action, or a request waits.
    Busy,
    /// The engine failed; the reason.
    Dead(String),
}

struct Waiter<R> {
    ids: Vec<u32>,
    payload: R,
}

/// The slots and the queue. `R` is a waiting request's payload, `A` an
/// action's.
pub(crate) struct Board<R, A> {
    /// Arrival tickets: one drawn per request queued, its turn passed when the
    /// request takes a slot, so `serving == next` exactly when none waits.
    tickets: SlotQueue,
    waiting: VecDeque<Waiter<R>>,
    actions: VecDeque<(Reserve, A)>,
    slots: Vec<SlotState>,
    depth: usize,
    clock: u64,
    picker: Box<dyn SlotPicker>,
    dead: Option<String>,
}

impl<R, A> Board<R, A> {
    pub(crate) fn new(parallel: usize, depth: usize, picker: Box<dyn SlotPicker>) -> Self {
        Board {
            tickets: SlotQueue::default(),
            waiting: VecDeque::new(),
            actions: VecDeque::new(),
            slots: (0..parallel)
                .map(|_| SlotState {
                    state: Use::Free,
                    last_used: 0,
                    routing: Vec::new(),
                    view: SlotView::default(),
                })
                .collect(),
            depth,
            clock: 0,
            picker,
            dead: None,
        }
    }

    pub(crate) fn slots(&self) -> &[SlotState] {
        &self.slots
    }

    pub(crate) fn view_mut(&mut self, slot: usize) -> &mut SlotView {
        &mut self.slots[slot].view
    }

    /// Requests waiting for a slot.
    pub(crate) fn waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Slots no request and no action holds.
    pub(crate) fn free(&self) -> usize {
        self.slots.iter().filter(|s| s.state == Use::Free).count()
    }

    /// Queues a request, unless the requests that would then wait past the
    /// free slots exceed the depth.
    pub(crate) fn enqueue(&mut self, ids: Vec<u32>, payload: R) -> Result<(), Refusal> {
        if let Some(reason) = &self.dead {
            return Err(Refusal::Dead(reason.clone()));
        }
        if self.waiting.len() >= self.depth + self.free() {
            return Err(Refusal::Full { depth: self.depth });
        }
        self.tickets.draw();
        self.waiting.push_back(Waiter { ids, payload });
        Ok(())
    }

    /// Reserves the slots of an action: free, and no request waiting. `slot`
    /// in `Reserve::One` is below the slot count; the caller checked it.
    pub(crate) fn reserve(&mut self, what: Reserve, action: A) -> Result<(), Refusal> {
        if let Some(reason) = &self.dead {
            return Err(Refusal::Dead(reason.clone()));
        }
        let free = match what {
            Reserve::One(i) => self.slots[i].state == Use::Free,
            Reserve::All => self.free() == self.slots.len(),
        };
        if !free {
            return Err(Refusal::Busy);
        }
        let Some(ticket) = self.tickets.try_draw() else {
            return Err(Refusal::Busy);
        };
        // The action takes its slots at once: the turn passes on.
        self.tickets.pass(ticket);
        for (i, s) in self.slots.iter_mut().enumerate() {
            if what == Reserve::All || what == Reserve::One(i) {
                s.state = Use::Held;
            }
        }
        self.actions.push_back((what, action));
        Ok(())
    }

    /// The actions reserved since the last call, in order.
    pub(crate) fn take_actions(&mut self) -> Vec<(Reserve, A)> {
        self.actions.drain(..).collect()
    }

    /// Gives every waiting request it can a free slot, oldest first unless the
    /// picker chooses another while more wait than slots are free. `held` is
    /// the ids each slot's cache holds; `on` the slot whose state the engine
    /// holds when the slots take it in turns, `None` when each slot keeps its
    /// own. Returns each request's slot, its ids and its payload, in the order
    /// they took them.
    ///
    /// # Panics
    ///
    /// When the picker answers an index past the waiting requests.
    pub(crate) fn admit<'h>(
        &mut self,
        held: impl Fn(usize) -> &'h [u32],
        on: Option<usize>,
    ) -> Vec<(usize, Vec<u32>, R)> {
        let mut out = Vec::new();
        if self.dead.is_some() {
            return out;
        }
        loop {
            let free: Vec<usize> = (0..self.slots.len())
                .filter(|&i| self.slots[i].state == Use::Free)
                .collect();
            if free.is_empty() || self.waiting.is_empty() {
                return out;
            }
            let at = if self.waiting.len() > free.len() {
                let waiting: Vec<WaitingRequest<'_>> = self
                    .waiting
                    .iter()
                    .map(|w| WaitingRequest { ids: &w.ids })
                    .collect();
                let slots: Vec<SlotSummary<'_>> = self
                    .slots
                    .iter()
                    .enumerate()
                    .map(|(id, s)| SlotSummary {
                        id,
                        busy: s.state != Use::Free,
                        held: held(id),
                        routing: &s.routing,
                    })
                    .collect();
                let at = self.picker.pick(&waiting, &slots);
                assert!(
                    at < waiting.len(),
                    "the slot picker chose waiting request {at} of {}",
                    waiting.len()
                );
                at
            } else {
                0
            };
            let Some(w) = self.waiting.remove(at) else {
                unreachable!("{at} is below the waiting requests");
            };
            let slot = best_slot(&self.slots, &free, &held, on, &w.ids);
            self.slots[slot].state = Use::Running;
            let serving = self.tickets.serving();
            self.tickets.pass(serving);
            out.push((slot, w.ids, w.payload));
        }
    }

    /// Frees `slot`.
    pub(crate) fn release(&mut self, slot: usize) {
        self.clock += 1;
        let s = &mut self.slots[slot];
        s.state = Use::Free;
        s.last_used = self.clock;
    }

    /// Refuses every later request and action with `reason`, and hands back
    /// what waits: the requests' payloads and the actions.
    pub(crate) fn kill(&mut self, reason: &str) -> (Vec<R>, Vec<A>) {
        self.dead = Some(reason.to_owned());
        let waiting = self.waiting.drain(..).map(|w| w.payload).collect();
        let actions = self.actions.drain(..).map(|(_, a)| a).collect();
        (waiting, actions)
    }
}

/// Of the free slots, the one whose held ids share the longest prefix with
/// `ids`; among equals the one the engine is on, then the least recently
/// used, then the lowest id.
fn best_slot<'h>(
    slots: &[SlotState],
    free: &[usize],
    held: &impl Fn(usize) -> &'h [u32],
    on: Option<usize>,
    ids: &[u32],
) -> usize {
    let key = |i: usize| {
        (
            common_prefix(held(i), ids),
            on == Some(i),
            Reverse(slots[i].last_used),
        )
    };
    let mut best = free[0];
    let mut top = key(best);
    for &i in &free[1..] {
        let k = key(i);
        if k > top {
            best = i;
            top = k;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::{
        Board, FifoPicker, Refusal, Reserve, SlotPicker, SlotSummary, Use, WaitingRequest,
    };

    fn board(n: usize, depth: usize) -> Board<u32, &'static str> {
        Board::new(n, depth, Box::new(FifoPicker))
    }

    /// No slot holds an id.
    fn nothing(_: usize) -> &'static [u32] {
        &[]
    }

    /// At N = 2 a free slot takes the oldest waiting request: four queued
    /// behind two busy slots take them in arrival order as the slots free.
    #[test]
    fn a_free_slot_takes_the_oldest_waiting_request() {
        let mut b = board(2, 8);
        for r in 0..6 {
            b.enqueue(vec![r], r).expect("room");
        }
        let first: Vec<u32> = b
            .admit(nothing, None)
            .into_iter()
            .map(|(_, _, r)| r)
            .collect();
        assert_eq!(first, [0, 1]);
        assert_eq!(b.waiting(), 4);
        let mut order = Vec::new();
        for slot in [1, 0, 0, 1] {
            b.release(slot);
            let got = b.admit(nothing, None);
            assert_eq!(got.len(), 1, "one slot freed");
            assert_eq!(got[0].0, slot, "the freed slot");
            order.push(got[0].2);
        }
        assert_eq!(order, [2, 3, 4, 5], "arrival order");
    }

    /// Of the free slots a request takes the one whose held ids share the
    /// longest prefix with it, not the least recently used one nor the one the
    /// engine is on; with no prefix shared it takes the one the engine is on
    /// when the slots take turns, else the least recently used.
    #[test]
    fn a_request_takes_the_slot_that_shares_its_prefix() {
        let mut b = board(3, 8);
        b.enqueue(vec![1, 2, 3], 0).expect("room");
        b.enqueue(vec![7, 8], 1).expect("room");
        b.enqueue(vec![4, 4], 2).expect("room");
        let slots: Vec<usize> = b
            .admit(nothing, None)
            .into_iter()
            .map(|(s, _, _)| s)
            .collect();
        assert_eq!(slots, [0, 1, 2], "nothing held: the lowest id among equals");
        // Slot 0 released first, so it is the least recently used.
        for slot in 0..3 {
            b.release(slot);
        }
        let ids = [vec![1, 2, 3, 9], vec![7, 8, 9], vec![4, 4, 9]];
        let held = |i: usize| ids[i].as_slice();
        b.enqueue(vec![7, 8, 5], 3).expect("room");
        assert_eq!(
            b.admit(held, Some(2))[0].0,
            1,
            "slot 1 holds 7 8; the engine is on slot 2"
        );
        b.enqueue(vec![6], 4).expect("room");
        assert_eq!(
            b.admit(held, None)[0].0,
            0,
            "no prefix shared: least recently used"
        );
        b.release(0);
        b.enqueue(vec![6], 5).expect("room");
        assert_eq!(
            b.admit(held, Some(0))[0].0,
            0,
            "no prefix shared: the slot the engine is on, not slot 2, the least recently used"
        );
    }

    /// Past the depth a request is refused, never queued: the depth counts the
    /// requests that wait past the free slots.
    #[test]
    fn a_request_past_the_depth_is_refused() {
        let mut b = board(2, 1);
        b.enqueue(vec![0], 0).expect("a free slot");
        b.enqueue(vec![1], 1).expect("a free slot");
        b.enqueue(vec![2], 2).expect("the one place in the queue");
        assert_eq!(
            b.enqueue(vec![3], 3),
            Err(Refusal::Full { depth: 1 }),
            "two free slots and one place: a fourth waits past the depth"
        );
        assert_eq!(b.admit(nothing, None).len(), 2);
        assert_eq!(b.waiting(), 1);
        assert_eq!(b.enqueue(vec![3], 3), Err(Refusal::Full { depth: 1 }));
        b.release(0);
        assert_eq!(b.admit(nothing, None).len(), 1);
        b.enqueue(vec![3], 3).expect("the place freed");
        let mut none = board(1, 0);
        none.enqueue(vec![0], 0).expect("the free slot");
        none.admit(nothing, None);
        assert_eq!(none.enqueue(vec![1], 1), Err(Refusal::Full { depth: 0 }));
    }

    /// An action takes a free slot only while no request waits, and the slot
    /// it holds takes no request until it is released.
    #[test]
    fn an_action_never_goes_ahead_of_a_waiting_request() {
        let mut b = board(2, 8);
        b.enqueue(vec![0], 0).expect("room");
        assert_eq!(b.admit(nothing, None)[0].0, 0);
        b.reserve(Reserve::One(1), "erase").expect("slot 1 is free");
        assert_eq!(b.reserve(Reserve::One(0), "save"), Err(Refusal::Busy));
        assert_eq!(b.reserve(Reserve::All, "reset"), Err(Refusal::Busy));
        b.enqueue(vec![1], 1).expect("room");
        assert!(
            b.admit(nothing, None).is_empty(),
            "slot 1 is held by the action"
        );
        assert_eq!(b.take_actions(), [(Reserve::One(1), "erase")]);
        b.release(1);
        b.release(0);
        assert_eq!(
            b.reserve(Reserve::One(0), "save"),
            Err(Refusal::Busy),
            "a request waits for a slot"
        );
        assert_eq!(b.admit(nothing, None)[0].2, 1);
        b.reserve(Reserve::One(0), "save")
            .expect("free, nothing waits");
        assert_eq!(b.slots()[0].state, Use::Held);
    }

    /// A picker that answers past the waiting requests stops the server by
    /// name; the picker is asked only while more wait than slots are free.
    #[test]
    #[should_panic(expected = "the slot picker chose waiting request 5 of 3")]
    fn a_picker_index_past_the_waiting_requests_is_named() {
        struct Past;
        impl SlotPicker for Past {
            fn pick(&mut self, _: &[WaitingRequest<'_>], _: &[SlotSummary<'_>]) -> usize {
                5
            }
        }
        let mut b: Board<u32, ()> = Board::new(2, 8, Box::new(Past));
        for r in 0..2 {
            b.enqueue(vec![r], r).expect("room");
        }
        assert_eq!(
            b.admit(nothing, None).len(),
            2,
            "two wait, two free: no pick"
        );
        for r in 2..5 {
            b.enqueue(vec![r], r).expect("room");
        }
        b.release(0);
        b.admit(nothing, None);
    }

    /// A picker chooses which waiting request takes a freed slot; the others
    /// keep their order.
    #[test]
    fn a_picker_chooses_the_request_a_freed_slot_takes() {
        struct Newest;
        impl SlotPicker for Newest {
            fn pick(&mut self, waiting: &[WaitingRequest<'_>], slots: &[SlotSummary<'_>]) -> usize {
                assert!(slots.iter().all(|s| s.routing.is_empty()));
                waiting.len() - 1
            }
        }
        let mut b: Board<u32, ()> = Board::new(1, 8, Box::new(Newest));
        for r in 0..4 {
            b.enqueue(vec![r], r).expect("room");
        }
        let mut order = Vec::new();
        for _ in 0..4 {
            let got = b.admit(nothing, None);
            order.push(got[0].2);
            b.release(0);
        }
        assert_eq!(order, [3, 2, 1, 0]);
        b.reserve(Reserve::All, ()).expect("nothing waits");
    }
}
