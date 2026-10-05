//! Slots that take one engine in turns: [`SwapEngine`] serves N logical slots
//! over an engine that holds one sequence, and the engine thread
//! ([`crate::worker`]) moves a slot's state aside when another slot takes the
//! engine.
//!
//! The rules the engine thread runs for such an engine:
//!
//! - Requests that take slots together start shortest prompt first, each
//!   after the one before at its turn boundary (or its end).
//! - A request that takes a slot while one other request decodes preempts it
//!   at the next step: the running request's state goes into the parked table
//!   ([`ParkTable`], never the prompt cache, whose eviction is silent), and the
//!   newcomer's prompt runs, then it decodes.
//! - With two requests or more live, they take the engine in turns of
//!   [`QUANTUM`] tokens, the one that waited longest next; a new prompt starts
//!   only at the running request's turn boundary, or when it ends.
//! - One request uses the engine at a time, so an engine that drafts keeps
//!   drafting.
//! - [`Park::States`]: a slot's state is the engine's snapshot, and the parked
//!   states' bytes stay under the budget the seat states. A live request's
//!   snapshot that fails or does not fit refuses the newcomer by name (a 503
//!   with `Retry-After`) and the running request goes on; at a turn boundary
//!   the running request keeps the engine until a request ends or starts. A
//!   state that does not resume ends its request with a named error.
//! - [`Park::Ids`], for an engine that cannot snapshot: a parked request keeps
//!   its ids, and its next turn feeds them again (re-prefill) before it steps.
//! - The state of an idle slot (its request finished) is parked when it fits
//!   and is the first to leave when a live one needs room. A state that
//!   leaves goes into the prompt cache as it is (no copy), as does one that
//!   found no room to park (its snapshot already taken); the slot then holds
//!   nothing, and a request on it starts from what the prompt cache holds.
//!   A request that starts on a slot whose idle state is parked takes that
//!   state as its slot's own, without a copy: back into the engine only if it
//!   keeps some of it, into the prompt cache as it is where the cache's rule
//!   saves it. Under [`Park::Ids`] an idle slot's state is not kept.
//! - A slot action waits until no request is live: it would move a running
//!   request's state aside.

use std::io::{Read, Write};
use std::sync::Arc;

use crate::api::ServeError;
use crate::engine::{
    CacheNote, Drafted, Engine, EngineError, EngineProps, ResidencyReset, Saved, SavedState,
    SlotRow, StateError, Tokenizer,
};

/// Tokens a request takes on the engine before the next live request's turn:
/// long enough that a turn's steps outweigh the two state copies that end it
/// while the states are a few thousand positions deep, short enough that a
/// parked stream waits at most a turn per live request for its next token.
pub const QUANTUM: usize = 64;

/// How a [`SwapEngine`] keeps the state of a slot the engine leaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Park {
    /// The engine's snapshot ([`Engine::snapshot`]), the parked snapshots'
    /// bytes ([`Saved::n_bytes`]) at most `budget`.
    States { budget: u64 },
    /// The slot's ids alone: its next turn feeds them again.
    Ids,
}

/// N logical slots over an engine of one: every call goes to the inner
/// engine, and [`Engine::select_slot`] only names the slot the server's next
/// per-slot calls act on — the engine thread moves the states
/// ([`Engine::turns`]). It never steps two slots in one call.
pub struct SwapEngine {
    inner: Box<dyn Engine>,
    slots: usize,
    park: Park,
}

impl SwapEngine {
    /// `inner`, which serves one slot, as `slots` slots that take it in turns.
    /// Refused by name: no slot, an inner engine of several slots or of its
    /// own turns, and a budget of no bytes.
    pub fn new(inner: Box<dyn Engine>, slots: usize, park: Park) -> Result<SwapEngine, ServeError> {
        let refuse = |why: String| Err(ServeError::Slots(why));
        if slots == 0 {
            return refuse("a swap engine of no slots".to_owned());
        }
        if inner.slots() != 1 || inner.turns().is_some() {
            return refuse(format!(
                "a swap engine takes an engine of one slot; this one serves {} slot(s){}",
                inner.slots(),
                if inner.turns().is_some() {
                    " in turns"
                } else {
                    ""
                }
            ));
        }
        if park == (Park::States { budget: 0 }) {
            return refuse("a park budget of 0 bytes parks no state".to_owned());
        }
        Ok(SwapEngine { inner, slots, park })
    }
}

impl Engine for SwapEngine {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.inner.tokenizer()
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        self.inner.prefill(ids)
    }
    fn next(&mut self, last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.inner.next(last, logits_out)
    }
    fn advance(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
        self.inner.advance(last, out)
    }
    fn advance_rows(&self) -> usize {
        self.inner.advance_rows()
    }
    fn slots(&self) -> usize {
        self.slots
    }
    fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
        if slot < self.slots {
            Ok(())
        } else {
            Err(EngineError(format!(
                "slot {slot}: this swap engine serves {} slots",
                self.slots
            )))
        }
    }
    fn step_slots(&mut self, rows: &mut [SlotRow<'_>]) -> Result<(), EngineError> {
        Err(EngineError(format!(
            "a step of {} slots: the slots of a swap engine take it in turns, one a call",
            rows.len()
        )))
    }
    fn turns(&self) -> Option<Park> {
        Some(self.park)
    }
    fn reset(&mut self) -> Result<(), EngineError> {
        self.inner.reset()
    }
    fn will_reply(&mut self, tokens: Option<usize>) {
        self.inner.will_reply(tokens);
    }
    fn keepable(&self, n: usize) -> usize {
        self.inner.keepable(n)
    }
    fn cut(&mut self, n: usize) -> Result<(), EngineError> {
        self.inner.cut(n)
    }
    fn ctx_max(&self) -> usize {
        self.inner.ctx_max()
    }
    fn describe(&self) -> String {
        self.inner.describe()
    }
    fn props_engine(&self) -> EngineProps {
        self.inner.props_engine()
    }
    fn save_state(&self, out: &mut dyn Write) -> Result<SavedState, StateError> {
        self.inner.save_state(out)
    }
    fn restore_state(&mut self, input: &mut dyn Read) -> Result<SavedState, StateError> {
        self.inner.restore_state(input)
    }
    fn keep_limit(&self, n: usize) -> Option<String> {
        self.inner.keep_limit(n)
    }
    fn cache_ram(&self) -> u64 {
        self.inner.cache_ram()
    }
    fn residency_reset(&mut self) -> Result<Option<ResidencyReset>, EngineError> {
        self.inner.residency_reset()
    }
    fn snapshot(&self) -> Result<Arc<dyn Saved>, StateError> {
        self.inner.snapshot()
    }
    fn resume(&mut self, state: &Arc<dyn Saved>) -> Result<(), StateError> {
        self.inner.resume(state)
    }
    fn prefill_splits(&self, first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
        self.inner.prefill_splits(first, end, marks)
    }
    fn note(&self, note: &CacheNote) {
        self.inner.note(note);
    }
}

/// What `/slots` shows of a slot of an engine whose slots take turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Turn {
    /// No request holds it.
    Idle,
    /// Its request is on the engine.
    Running,
    /// Its request is live and its state is parked.
    Parked,
    /// Its request took it and waits for its prompt's turn.
    Queued,
}

impl Turn {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Turn::Idle => "idle",
            Turn::Running => "running",
            Turn::Parked => "parked",
            Turn::Queued => "queued",
        }
    }
}

/// A parked slot.
pub(crate) enum Entry {
    /// The engine's snapshot; `live` while its request runs, else the first
    /// to leave for room. `at` orders the idle ones, oldest first.
    State {
        state: Arc<dyn Saved>,
        live: bool,
        at: u64,
    },
    /// The slot's ids are fed again on its next turn ([`Park::Ids`]).
    Ids,
}

/// The states of the slots the engine left, one entry a slot, and their
/// bytes against the budget.
pub(crate) struct ParkTable {
    park: Park,
    entries: Vec<Option<Entry>>,
    bytes: u64,
    clock: u64,
}

/// An idle state that left the table to make room, and its slot
/// ([`ParkTable::room`]).
pub(crate) type LetGo = (usize, Arc<dyn Saved>);

/// Why a state does not fit the table: its bytes, and those the table holds
/// that no eviction can free.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct NoRoom {
    pub need: u64,
    pub pinned: u64,
    pub budget: u64,
}

impl ParkTable {
    pub(crate) fn new(park: Park, slots: usize) -> ParkTable {
        ParkTable {
            park,
            entries: (0..slots).map(|_| None).collect(),
            bytes: 0,
            clock: 0,
        }
    }

    pub(crate) fn park(&self) -> Park {
        self.park
    }

    /// The parked states' bytes.
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Room for a state of `need` bytes: the idle states not in `keep` leave,
    /// oldest first, until it fits; returns them with their slots, for the
    /// prompt cache to take as they are. When even all of them leaving leaves
    /// no room, none leaves.
    ///
    /// # Panics
    ///
    /// Under [`Park::Ids`], which keeps no state.
    pub(crate) fn room(&mut self, need: u64, keep: &[usize]) -> Result<Vec<LetGo>, NoRoom> {
        let Park::States { budget } = self.park else {
            panic!("room for a state in a table of ids");
        };
        let mut idle: Vec<(u64, usize, u64)> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(slot, e)| match e {
                Some(Entry::State {
                    state,
                    live: false,
                    at,
                }) if !keep.contains(&slot) => Some((*at, slot, state.n_bytes())),
                _ => None,
            })
            .collect();
        let freeable: u64 = idle.iter().map(|&(_, _, b)| b).sum();
        let pinned = self.bytes - freeable;
        if pinned.saturating_add(need) > budget {
            return Err(NoRoom {
                need,
                pinned,
                budget,
            });
        }
        idle.sort_unstable();
        let mut gone = Vec::new();
        for (_, slot, _) in idle {
            if self.bytes.saturating_add(need) <= budget {
                break;
            }
            let Some(Entry::State { state, .. }) = self.take(slot) else {
                unreachable!("slot {slot}'s idle state left the table while it made room");
            };
            gone.push((slot, state));
        }
        Ok(gone)
    }

    /// Parks `entry` for `slot`, which holds none; a state's bytes are counted.
    ///
    /// # Panics
    ///
    /// When `slot` holds an entry already: the engine left a slot twice.
    pub(crate) fn put(&mut self, slot: usize, entry: Entry) {
        assert!(
            self.entries[slot].is_none(),
            "slot {slot} parked twice: the engine left it again before it came back"
        );
        let entry = match entry {
            Entry::State { state, live, .. } => {
                self.bytes += state.n_bytes();
                self.clock += 1;
                Entry::State {
                    state,
                    live,
                    at: self.clock,
                }
            }
            Entry::Ids => Entry::Ids,
        };
        self.entries[slot] = Some(entry);
    }

    /// `slot`'s entry, out of the table.
    pub(crate) fn take(&mut self, slot: usize) -> Option<Entry> {
        let e = self.entries[slot].take();
        if let Some(Entry::State { state, .. }) = &e {
            self.bytes -= state.n_bytes();
        }
        e
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;

    use super::*;

    /// A saved state of no positions, its bytes the value.
    struct Bytes(u64);

    impl Saved for Bytes {
        fn n_tokens(&self) -> usize {
            0
        }
        fn n_bytes(&self) -> u64 {
            self.0
        }
        fn keepable(&self, _n: usize) -> usize {
            0
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn state(bytes: u64, live: bool) -> Entry {
        Entry::State {
            state: Arc::new(Bytes(bytes)),
            live,
            at: 0,
        }
    }

    /// [`ParkTable::room`] with each state that left read as its bytes.
    fn room(t: &mut ParkTable, need: u64, keep: &[usize]) -> Result<Vec<(usize, u64)>, NoRoom> {
        t.room(need, keep)
            .map(|gone| gone.iter().map(|(slot, s)| (*slot, s.n_bytes())).collect())
    }

    #[test]
    fn a_state_past_what_no_eviction_frees_is_refused_and_nothing_leaves() {
        let mut t = ParkTable::new(Park::States { budget: 100 }, 4);
        t.put(0, state(60, true));
        t.put(1, state(30, false));
        assert_eq!(
            room(&mut t, 50, &[]),
            Err(NoRoom {
                need: 50,
                pinned: 60,
                budget: 100
            })
        );
        assert_eq!(t.bytes(), 90, "nothing left for a refusal");
    }

    #[test]
    fn idle_states_leave_oldest_first_and_only_as_needed() {
        let mut t = ParkTable::new(Park::States { budget: 100 }, 4);
        t.put(0, state(30, false));
        t.put(1, state(30, false));
        t.put(2, state(30, true));
        assert_eq!(room(&mut t, 10, &[]), Ok(vec![]));
        assert_eq!(
            room(&mut t, 40, &[]),
            Ok(vec![(0, 30)]),
            "slot 0's state, handed out"
        );
        assert_eq!(t.bytes(), 60);
        assert!(t.take(0).is_none(), "slot 0 left the table");
    }

    #[test]
    fn a_kept_idle_state_does_not_leave() {
        let mut t = ParkTable::new(Park::States { budget: 100 }, 4);
        t.put(0, state(30, false));
        t.put(1, state(30, false));
        t.put(2, state(30, false));
        assert_eq!(room(&mut t, 40, &[0]), Ok(vec![(1, 30)]));
        assert!(t.room(80, &[0]).is_err(), "slot 0 is kept: 30 + 80 > 100");
    }
}
