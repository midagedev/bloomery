//! Who gets a slot: the server's N slots (`--parallel N`), the requests
//! waiting for one, and the bound on how many may wait (`--queue-depth Q`).
//!
//! One queue in arrival order holds every waiting request; a free slot takes
//! the oldest of them. When more requests wait than slots are free, a
//! [`SlotPicker`] chooses which one goes next ([`FifoPicker`], the oldest, by
//! default). The request then takes, of the free slots, the one whose held
//! sequence shares the longest prefix with its own: its cache keeps the
//! most. A shared prefix counts only past llama-server's similarity floor
//! (strictly more than 0.1 of the request's ids,
//! [`PROMPT_SIMILARITY_FLOOR`]; a slot at or under it counts as sharing
//! nothing), so a request that barely matches an idle conversation takes an
//! empty slot instead of cutting it. A shared prefix counts an image span
//! only when both sides hold the same image ([`common_prefix`]), so a
//! request takes the slot that holds its image, not one that holds another
//! of the same grid. Among equals a slot that holds no ids comes before one
//! that holds a conversation: seating the request there cuts nothing, so
//! nothing is saved either. When the slots take one engine in turns, the
//! slot whose state the engine holds goes before the least recently used:
//! taking another moves that state aside. Those last two orders are
//! bloomery's; llama-server floors the prefix and then takes the least
//! recently used free slot alone.
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
use crate::media::{Held, common_prefix};

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
    held: Held,
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
    pub(crate) fn enqueue(&mut self, held: Held, payload: R) -> Result<(), Refusal> {
        if let Some(reason) = &self.dead {
            return Err(Refusal::Dead(reason.clone()));
        }
        if self.waiting.len() >= self.depth + self.free() {
            return Err(Refusal::Full { depth: self.depth });
        }
        self.tickets.draw();
        self.waiting.push_back(Waiter { held, payload });
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
    /// what each slot's cache holds; `on` the slot whose state the engine
    /// holds when the slots take it in turns, `None` when each slot keeps its
    /// own. Returns each request's slot, what its prompt holds and its
    /// payload, in the order they took them.
    ///
    /// # Panics
    ///
    /// When the picker answers an index past the waiting requests.
    pub(crate) fn admit<'h>(
        &mut self,
        held: impl Fn(usize) -> &'h Held,
        on: Option<usize>,
    ) -> Vec<(usize, Held, R)> {
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
                    .map(|w| WaitingRequest { ids: &w.held.ids })
                    .collect();
                let slots: Vec<SlotSummary<'_>> = self
                    .slots
                    .iter()
                    .enumerate()
                    .map(|(id, s)| SlotSummary {
                        id,
                        busy: s.state != Use::Free,
                        held: &held(id).ids,
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
            let slot = best_slot(&self.slots, &free, &held, on, &w.held);
            self.slots[slot].state = Use::Running;
            let serving = self.tickets.serving();
            self.tickets.pass(serving);
            out.push((slot, w.held, w.payload));
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

/// llama-server's prompt-similarity floor, 0.1, held as the exact fraction
/// (1, 10): its `slot_prompt_similarity` default, set by
/// `-sps/--slot-prompt-similarity` (`0` there turns the floor off; ours is
/// no flag). A free slot's shared prefix counts only when it holds strictly
/// more than this fraction of the request's ids.
const PROMPT_SIMILARITY_FLOOR: (usize, usize) = (1, 10);

/// The prefix a free slot ranks by: itself when it clears
/// [`PROMPT_SIMILARITY_FLOOR`] against a request of `len` ids, else none —
/// the strict `>` of llama-server, so a prefix exactly at the floor shares
/// nothing. In integers (`prefix * 10 > len`), so no float division rounds
/// at the floor's edge; a request of no ids, which the HTTP layer refuses
/// before one queues, divides nothing and shares nothing here.
fn floored_prefix(prefix: usize, len: usize) -> usize {
    if prefix * PROMPT_SIMILARITY_FLOOR.1 > len * PROMPT_SIMILARITY_FLOOR.0 {
        prefix
    } else {
        0
    }
}

/// Of the free slots, the one whose held sequence shares the longest prefix
/// with `req` — the same images in the spans ([`common_prefix`]), so the slot
/// that holds the request's image beats one that holds another of the same
/// grid. A prefix at or under [`PROMPT_SIMILARITY_FLOOR`], 0.1 of `req`'s
/// ids, counts as none, so a request that clears it nowhere falls to the
/// orders below the prefix: one that holds no ids (nothing is cut or
/// saved), then the one the engine is on, then the least recently used, then
/// the lowest id.
fn best_slot<'h>(
    slots: &[SlotState],
    free: &[usize],
    held: &impl Fn(usize) -> &'h Held,
    on: Option<usize>,
    req: &Held,
) -> usize {
    let key = |i: usize| {
        let held = held(i);
        (
            floored_prefix(common_prefix(held, req), req.ids.len()),
            held.is_empty(),
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
    use crate::media::{Held, ImageKey, MediaSpan};

    fn board(n: usize, depth: usize) -> Board<u32, &'static str> {
        Board::new(n, depth, Box::new(FifoPicker))
    }

    /// No slot holds an id.
    static NOTHING: Held = Held {
        ids: Vec::new(),
        media: Vec::new(),
    };

    fn nothing(_: usize) -> &'static Held {
        &NOTHING
    }

    /// A held sequence of one image span at `at`: the ids of the span carry
    /// the image token, as an expanded prompt's do.
    fn held_image(mut ids: Vec<u32>, at: usize, len: usize, key: u8) -> Held {
        for id in &mut ids[at..at + len] {
            *id = 7;
        }
        Held {
            media: vec![MediaSpan {
                at,
                len,
                key: ImageKey([key; 32]),
            }],
            ids,
        }
    }

    /// A sequence of 20 ids whose first `n` are 0 to `n − 1` and whose rest
    /// starts at `tail`: two of these share exactly the ids their heads
    /// share, so a request of `seq_sharing(20, _)` shares `n` of its 20 with
    /// `seq_sharing(n, other_tail)`.
    fn seq_sharing(n: u32, tail: u32) -> Held {
        Held::from((0..n).chain(tail..tail + 20 - n).collect::<Vec<u32>>())
    }

    /// At N = 2 a free slot takes the oldest waiting request: four queued
    /// behind two busy slots take them in arrival order as the slots free.
    #[test]
    fn a_free_slot_takes_the_oldest_waiting_request() {
        let mut b = board(2, 8);
        for r in 0..6 {
            b.enqueue(Held::from(vec![r]), r).expect("room");
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
        b.enqueue(Held::from(vec![1, 2, 3]), 0).expect("room");
        b.enqueue(Held::from(vec![7, 8]), 1).expect("room");
        b.enqueue(Held::from(vec![4, 4]), 2).expect("room");
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
        let ids = [
            Held::from(vec![1, 2, 3, 9]),
            Held::from(vec![7, 8, 9]),
            Held::from(vec![4, 4, 9]),
        ];
        let held = |i: usize| &ids[i];
        b.enqueue(Held::from(vec![7, 8, 5]), 3).expect("room");
        assert_eq!(
            b.admit(held, Some(2))[0].0,
            1,
            "slot 1 holds 7 8; the engine is on slot 2"
        );
        b.enqueue(Held::from(vec![6]), 4).expect("room");
        assert_eq!(
            b.admit(held, None)[0].0,
            0,
            "no prefix shared: least recently used"
        );
        b.release(0);
        b.enqueue(Held::from(vec![6]), 5).expect("room");
        assert_eq!(
            b.admit(held, Some(0))[0].0,
            0,
            "no prefix shared: the slot the engine is on, not slot 2, the least recently used"
        );
    }

    /// A request whose prompt carries an image takes the slot that holds the
    /// same image. Two slots can hold the same ids with different images (a
    /// span's every position carries the image token), and the key tells them
    /// apart: the ids alone would leave the request on the slot the engine is
    /// on, which holds the other image.
    #[test]
    fn a_request_takes_the_slot_that_holds_its_image() {
        let mut b = board(2, 8);
        b.enqueue(Held::from(vec![1]), 0).expect("room");
        b.enqueue(Held::from(vec![2]), 1).expect("room");
        b.admit(nothing, None);
        b.release(0);
        b.release(1);
        let ids = vec![1, 2, 7, 7, 7, 4];
        let slots = [held_image(ids.clone(), 2, 3, 1), held_image(ids, 2, 3, 2)];
        let held = |i: usize| &slots[i];
        b.enqueue(held_image(vec![1, 2, 7, 7, 7, 4, 9], 2, 3, 1), 3)
            .expect("room");
        assert_eq!(
            b.admit(held, Some(1))[0].0,
            0,
            "slot 0 holds the request's image; slot 1, where the engine is, holds another"
        );
    }

    /// Of the free slots whose held ids share the request's prefix equally,
    /// one that holds no ids comes before one that holds a conversation:
    /// seating the request there cuts nothing, so nothing is saved either. A
    /// longer prefix still wins, and among empty slots the order is as it
    /// was: the slot the engine is on, then the least recently used, then the
    /// lowest id.
    #[test]
    fn a_request_takes_an_empty_slot_before_one_that_holds_ids() {
        let mut b = board(2, 8);
        for r in 0..2 {
            b.enqueue(Held::from(vec![r]), r).expect("room");
        }
        b.admit(nothing, None);
        // Slot 0, released first, holds a conversation and is the least
        // recently used; slot 1, released after, holds nothing.
        b.release(0);
        b.release(1);
        let a = Held::from(vec![7, 8, 9]);
        let empty = Held::default();
        let held = |i: usize| if i == 0 { &a } else { &empty };
        b.enqueue(Held::from(vec![1, 2, 3]), 2).expect("room");
        assert_eq!(
            b.admit(held, None)[0].0,
            1,
            "nothing shared with A: the empty slot, not the idle conversation"
        );
        b.release(1);
        b.enqueue(Held::from(vec![7, 8, 9, 4]), 3).expect("room");
        assert_eq!(
            b.admit(held, None)[0].0,
            0,
            "the request shares A's prefix: the prefix beats the empty slot"
        );
        let mut b = board(2, 8);
        for r in 0..2 {
            b.enqueue(Held::from(vec![r]), r).expect("room");
        }
        b.admit(nothing, None);
        b.release(1);
        b.release(0);
        b.enqueue(Held::from(vec![5]), 0).expect("room");
        assert_eq!(
            b.admit(nothing, None)[0].0,
            1,
            "both empty: the least recently used, not the lowest id"
        );
        b.release(1);
        b.enqueue(Held::from(vec![5]), 1).expect("room");
        assert_eq!(
            b.admit(nothing, Some(1))[0].0,
            1,
            "both empty: the slot the engine is on, not the least recently used"
        );
    }

    /// A shared prefix counts only past llama-server's similarity floor, 0.1
    /// of the request's ids: a request that shares 1 of its 20 ids with an
    /// idle conversation (5 %) takes the empty slot, one that shares 3 of 20
    /// (15 %) takes the conversation's slot over the empty one, and one
    /// exactly at the floor (2 of 20, 10 %) takes the empty slot — the floor
    /// is a strict `>`.
    #[test]
    fn a_request_under_the_similarity_floor_takes_the_empty_slot() {
        let mut b = board(2, 8);
        for r in 0..2 {
            b.enqueue(Held::from(vec![r]), r).expect("room");
        }
        b.admit(nothing, None);
        // Slot 0, released first, holds a conversation and is the least
        // recently used; slot 1, released after, holds nothing.
        b.release(0);
        b.release(1);
        let (a, empty) = (seq_sharing(20, 100), Held::default());
        let held = |i: usize| if i == 0 { &a } else { &empty };
        b.enqueue(seq_sharing(1, 200), 2).expect("room");
        assert_eq!(
            b.admit(held, None)[0].0,
            1,
            "1 of 20 shared (5 %, under the floor): the empty slot, not the idle conversation"
        );
        b.release(1);
        b.enqueue(seq_sharing(3, 200), 3).expect("room");
        assert_eq!(
            b.admit(held, None)[0].0,
            0,
            "3 of 20 shared (15 %, past the floor): the conversation's slot"
        );
        b.release(0);
        b.enqueue(seq_sharing(2, 200), 4).expect("room");
        assert_eq!(
            b.admit(held, None)[0].0,
            1,
            "2 of 20 shared (10 %, exactly the floor): not a candidate"
        );
    }

    /// Of the held slots past the floor the longest shared prefix wins,
    /// wherever the slot sits in the orders below the prefix; two that share
    /// equally fall to them: the slot the engine is on, then the least
    /// recently used.
    #[test]
    fn held_slots_past_the_floor_rank_by_shared_prefix() {
        let mut b = board(3, 8);
        for r in 0..3 {
            b.enqueue(Held::from(vec![r]), r).expect("room");
        }
        b.admit(nothing, None);
        for slot in 0..3 {
            b.release(slot);
        }
        // Released in id order: slot 0 is the least recently used, slot 2
        // the most.
        let one = [
            seq_sharing(3, 100),
            seq_sharing(0, 140),
            seq_sharing(4, 180),
        ];
        let held = |i: usize| &one[i];
        b.enqueue(seq_sharing(20, 0), 3).expect("room");
        assert_eq!(
            b.admit(held, None)[0].0,
            2,
            "4 of 20 on slot 2 beats 3 on slot 0, the least recently used"
        );
        b.release(2);
        let two = [
            seq_sharing(4, 100),
            seq_sharing(0, 140),
            seq_sharing(4, 180),
        ];
        let held = |i: usize| &two[i];
        b.enqueue(seq_sharing(20, 0), 4).expect("room");
        assert_eq!(
            b.admit(held, Some(2))[0].0,
            2,
            "equal 4 of 20: the slot the engine is on, not slot 0, the least recently used"
        );
        b.release(2);
        b.enqueue(seq_sharing(20, 0), 5).expect("room");
        assert_eq!(
            b.admit(held, None)[0].0,
            0,
            "equal 4 of 20, no engine: the least recently used"
        );
    }

    /// With every slot held and no prefix past the floor — one exactly at
    /// it, one under — the orders below the prefix decide as they did before
    /// it: the slot the engine is on, then the least recently used. The floor
    /// turns a small prefix into none; it never leaves a request unseated.
    #[test]
    fn with_no_slot_past_the_floor_the_orders_below_decide() {
        let mut b = board(3, 8);
        for r in 0..3 {
            b.enqueue(Held::from(vec![r]), r).expect("room");
        }
        b.admit(nothing, None);
        for slot in 0..3 {
            b.release(slot);
        }
        // Slot 0, the least recently used, would win on prefix with no floor
        // (2 of 20, exactly at it); the engine's slot 2 shares nothing.
        let all = [
            seq_sharing(2, 100),
            seq_sharing(1, 140),
            seq_sharing(0, 180),
        ];
        let held = |i: usize| &all[i];
        b.enqueue(seq_sharing(20, 0), 3).expect("room");
        assert_eq!(
            b.admit(held, Some(2))[0].0,
            2,
            "no slot past the floor: the slot the engine is on, not slot 0, whose 2 of 20 \
             sits exactly at it"
        );
        b.release(2);
        b.enqueue(seq_sharing(20, 0), 4).expect("room");
        assert_eq!(
            b.admit(held, None)[0].0,
            0,
            "no engine: the least recently used"
        );
    }

    /// Past the depth a request is refused, never queued: the depth counts the
    /// requests that wait past the free slots.
    #[test]
    fn a_request_past_the_depth_is_refused() {
        let mut b = board(2, 1);
        b.enqueue(Held::from(vec![0]), 0).expect("a free slot");
        b.enqueue(Held::from(vec![1]), 1).expect("a free slot");
        b.enqueue(Held::from(vec![2]), 2)
            .expect("the one place in the queue");
        assert_eq!(
            b.enqueue(Held::from(vec![3]), 3),
            Err(Refusal::Full { depth: 1 }),
            "two free slots and one place: a fourth waits past the depth"
        );
        assert_eq!(b.admit(nothing, None).len(), 2);
        assert_eq!(b.waiting(), 1);
        assert_eq!(
            b.enqueue(Held::from(vec![3]), 3),
            Err(Refusal::Full { depth: 1 })
        );
        b.release(0);
        assert_eq!(b.admit(nothing, None).len(), 1);
        b.enqueue(Held::from(vec![3]), 3).expect("the place freed");
        let mut none = board(1, 0);
        none.enqueue(Held::from(vec![0]), 0).expect("the free slot");
        none.admit(nothing, None);
        assert_eq!(
            none.enqueue(Held::from(vec![1]), 1),
            Err(Refusal::Full { depth: 0 })
        );
    }

    /// An action takes a free slot only while no request waits, and the slot
    /// it holds takes no request until it is released.
    #[test]
    fn an_action_never_goes_ahead_of_a_waiting_request() {
        let mut b = board(2, 8);
        b.enqueue(Held::from(vec![0]), 0).expect("room");
        assert_eq!(b.admit(nothing, None)[0].0, 0);
        b.reserve(Reserve::One(1), "erase").expect("slot 1 is free");
        assert_eq!(b.reserve(Reserve::One(0), "save"), Err(Refusal::Busy));
        assert_eq!(b.reserve(Reserve::All, "reset"), Err(Refusal::Busy));
        b.enqueue(Held::from(vec![1]), 1).expect("room");
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
            b.enqueue(Held::from(vec![r]), r).expect("room");
        }
        assert_eq!(
            b.admit(nothing, None).len(),
            2,
            "two wait, two free: no pick"
        );
        for r in 2..5 {
            b.enqueue(Held::from(vec![r]), r).expect("room");
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
            b.enqueue(Held::from(vec![r]), r).expect("room");
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
