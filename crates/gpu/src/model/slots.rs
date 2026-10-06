//! Resident sequence slots: the common owner of N sequences inside one
//! [`GpuModel`] ([`GpuModel::add_slots`]). A body that implements [`Slots`]
//! parks its per-sequence state with the model, and the model switches
//! slots by pointer exchange — no device copy, no allocation, no capture, no
//! synchronization.
//!
//! Why exchange: a CUDA graph replays the buffer addresses it recorded, so a
//! slot's captured chains must stay with that slot's buffers. The model
//! keeps one graph cache ([`Graphs`]) a slot, and every capture the body
//! records over its sequence's stores travels inside its [`Slots::Seq`] —
//! exchanged by the body's own [`Slots::swap_seq`], so a select costs no
//! capture and never replays one slot's chain over another's buffers.
//!
//! Where each slot's state lives: slot 0's home is the live fields (the
//! load's own sequence, where every enqueue already reads it); each further
//! slot's home is its [`ParkedSlot`] entry, and the entry of the slot that
//! is live holds slot 0's state. A switch to or from slot 0 is one
//! exchange; a switch between two parked slots is two (the live slot's
//! state home first, the target's out) — pointer swaps either way.
//!
//! One pass of several slots ([`GpuModel::step_slots`], a body that is
//! [`SlotRows`]) runs every busy slot's rows at once: the row-wise launches
//! over all of them, each slot's sequence-bound launches over its own rows
//! and stores. The homes are made canonical first — slot 0 selected, so
//! every slot's state sits in its home — and the body binds each slot's
//! stores by the slot alone. Those captures live in one model-wide cache
//! keyed by the busy slots and their row counts ([`SlotGraphs`]).

use super::{
    ChainBody, GpuModel, Graphs, MAX_PASS_ROWS, RowHeads, StepMode, launch_served, no_head,
    pass_slice,
};
use crate::fault::Fault;
use crate::head::Head;
use crate::host::PassKind;
use crate::host::swap::BoundaryAt;
use crate::hybrid::{Chain, Refusal};
use crate::weights::Weights;
use crate::{Gpu, GpuError, Graph, NodeInfo};
use cuda_core::CudaStream;

use runtime::swaprule::KeptRows;
use std::fmt;
use std::ops::Range;

/// A [`ChainBody`] that can hold several sequences at once: N resident
/// sequences inside one model over one set of weights, switched by pointer
/// exchange. The body keeps the live sequence where its own enqueues read
/// it; the model parks the others ([`GpuModel::add_slots`]) and every call
/// acts on the selected one ([`GpuModel::select_slot`]). Model-wide state
/// stays one: the fault word, the residency machine and host tier, the
/// heads and the weights.
pub trait Slots: ChainBody {
    /// Everything one sequence owns on the card and host: the stores a step
    /// or a prompt writes and a later call reads, and any body-internal
    /// capture that records those stores' addresses.
    type Seq: Send;

    /// A sequence in the state the load leaves ([`ChainBody::reset`]'s
    /// contract), of the same shape (`ctx_max`) as the live one. Load-time
    /// allocation.
    fn new_seq(&mut self, gpu: &Gpu) -> Result<Self::Seq, GpuError>;

    /// Exchange the live sequence with `seq`: pointer moves only for a body
    /// whose sequences are device buffers. Fallible and handed the card for
    /// a body that must drain work in flight before its exchange.
    fn swap_seq(&mut self, gpu: &Gpu, seq: &mut Self::Seq) -> Result<(), GpuError>;

    /// Device bytes one sequence holds: what [`GpuModel::resident_bytes`]
    /// grows by per added slot.
    fn seq_bytes(&self) -> usize;
}

/// A [`Slots`] body that runs several resident sequences' rows as one pass
/// ([`GpuModel::step_slots`]): its row-wise launches once over all `R` rows,
/// the launches bound to one sequence's stores once per busy slot over that
/// slot's rows and stores, each the launch that slot's rows would run
/// alone. Each row ends in the heads [`SlotRows::HEADS`] lays. A body
/// without it serves several slots by a select and a step a slot.
pub trait SlotRows: Slots {
    /// The most rows one pass of several slots takes, 1 to
    /// [`MAX_PASS_ROWS`]: a fact of the body's walk, which every body
    /// states. A pass of more rows is refused by name
    /// ([`GpuModel::step_slots`]); a bound outside 1 to [`MAX_PASS_ROWS`]
    /// does not compile. Not [`super::Rows::MAX_ROWS`], the bound of a
    /// verify of one sequence's positions.
    const MAX_ROWS: usize;

    /// The chain a replay of the pass of `rows` is served as
    /// ([`super::HostServed::serve_captured`]), as
    /// [`super::Rows::chain_of`] is a verify's. Read only on a load whose
    /// chain holds host work.
    fn chain_of(rows: &[SlotRange]) -> Chain;

    /// How the pass lays its rows over output heads, as [`super::Rows::HEADS`]
    /// lays a verify's: [`RowHeads::PerRow`] row `r` into `heads[r]`,
    /// [`RowHeads::One`] every row into one head of the pass's rows.
    const HEADS: RowHeads = RowHeads::PerRow;

    /// The pass's host half: each busy slot's input record — its first
    /// position `rows[i].pos0` and its ids `ids[rows[i].rows]` — written into
    /// the buffers the pass reads. `parked` is every busy slot's sequence but
    /// the live one, in `rows` order, as [`SlotRows::enqueue_slots`] gets it:
    /// a body whose sequences carry host state plans each slot from its own.
    /// Never inside a capture.
    fn plan_slots(
        &mut self,
        stream: &CudaStream,
        parked: &mut [&mut Self::Seq],
        rows: &[SlotRange],
        ids: &[u32],
    ) -> Result<(), GpuError>;

    /// Enqueue the pass of `rows` the last [`SlotRows::plan_slots`] planned
    /// into `heads`, as [`SlotRows::HEADS`] lays them: `heads[r]` row `r`'s
    /// for [`RowHeads::PerRow`], `heads[0]` a head of every row for
    /// [`RowHeads::One`]. The homes are canonical: slot 0's sequence
    /// is the live one, and `parked` holds every other busy slot's, in
    /// `rows` order. Reads no per-call value from `rows` but the slots and
    /// their row ranges, so one capture serves every call of the same
    /// slots and counts. Asynchronous as [`ChainBody::enqueue_chain`] is.
    fn enqueue_slots(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        heads: &mut [Head],
        parked: &mut [&mut Self::Seq],
        rows: &[SlotRange],
    ) -> Result<(), GpuError>;

    /// Whether this body settles a partial keep of a pass's rows at
    /// [`GpuModel::commit_slots`]: a body that does overrides
    /// [`SlotRows::keep_slot`] with the settling. A body without it keeps
    /// every row as it runs (the default keep), so
    /// [`GpuModel::verify_slots`] — whose commit keeps each slot's accepted
    /// rows alone — refuses by name before anything runs.
    const SETTLES_PARTIAL_KEEP: bool = false;

    /// The slot of `r` keeps the first `kept` rows (1 to its rows) of the
    /// pass that just ran, after its readback, on `seq` — its parked
    /// sequence, or the live one (slot 0's) when `None`: every row after
    /// [`GpuModel::step_slots`], the rows its caller accepted at
    /// [`GpuModel::commit_slots`]. A body that holds a pass's rows waiting
    /// settles them here. By default a body keeps every row as it runs
    /// and takes none back: any other count is refused by name.
    fn keep_slot(
        &mut self,
        gpu: &Gpu,
        seq: Option<&mut Self::Seq>,
        r: &SlotRange,
        kept: usize,
    ) -> Result<(), GpuError> {
        let _ = (gpu, seq);
        if kept == r.rows.len() {
            return Ok(());
        }
        Err(GpuError::shape(
            "SlotRows::keep_slot",
            format!(
                "slot {} keeping {kept} of its {} rows: this body keeps a pass's rows whole",
                r.slot,
                r.rows.len()
            ),
        ))
    }

    /// A `step_slots` pass (every row kept) of these ranges refused by name when
    /// this load gives several rows of one slot another meaning (a verify's rows
    /// on a drafted load). Default: nothing refused.
    fn refuse_kept(&self, ranges: &[SlotRange]) -> Result<(), GpuError> {
        let _ = ranges;
        Ok(())
    }
}

/// One busy slot's rows in a pass of several slots: the slot, the range of
/// the pass's rows it holds, and the position of its first row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotRange {
    pub slot: usize,
    pub rows: Range<usize>,
    pub pos0: u32,
}

/// What [`GpuModel::step_slots`] returns: every row's greedy next token in
/// the pass's row order — the order of the rows it was handed, each slot's
/// tokens in turn. Row `r`'s logits are its own head's, as a
/// [`RowHeads::PerRow`] pass of [`GpuModel::step_rows`] leaves them, and
/// [`GpuModel::slots_logits`] reads them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotsOut {
    pub ids: Vec<u32>,
}

/// The slots a faulted call ran on, ascending: the live slot of a step,
/// every busy slot of a pass — at most [`MAX_PASS_ROWS`], a slot holding a
/// row at least.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SlotSet {
    ids: [usize; MAX_PASS_ROWS],
    len: usize,
}

impl SlotSet {
    /// The one slot `slot`.
    pub(super) fn one(slot: usize) -> SlotSet {
        let mut ids = [0; MAX_PASS_ROWS];
        ids[0] = slot;
        SlotSet { ids, len: 1 }
    }

    /// The slots of a pass's ranges (distinct, at most [`MAX_PASS_ROWS`]:
    /// [`GpuModel::step_slots`] refuses any other set before it runs).
    fn of(rows: &[SlotRange]) -> SlotSet {
        let mut ids = [0; MAX_PASS_ROWS];
        let len = rows.len().min(MAX_PASS_ROWS);
        for (id, r) in ids.iter_mut().zip(rows) {
            *id = r.slot;
        }
        ids[..len].sort_unstable();
        SlotSet { ids, len }
    }

    /// Every slot of a model that serves `n`, 1 to [`MAX_PASS_ROWS`]: the
    /// set an unrecorded tier refusal poisons — the slot it ran on is
    /// unknown, so every slot owes its reset. A larger lot is refused by
    /// name: a poison's set holds what a pass can, and no load builds a
    /// lot a pass cannot serve.
    pub(super) fn every(n: usize, what: &'static str) -> Result<SlotSet, GpuError> {
        if n == 0 || n > MAX_PASS_ROWS {
            return Err(GpuError::shape(
                what,
                format!("{n} slots; a poison's slot set holds 1 to {MAX_PASS_ROWS}"),
            ));
        }
        let mut ids = [0; MAX_PASS_ROWS];
        for (i, id) in ids.iter_mut().enumerate().take(n) {
            *id = i;
        }
        Ok(SlotSet { ids, len: n })
    }

    pub(super) fn as_slice(&self) -> &[usize] {
        &self.ids[..self.len]
    }

    /// Take `slot` out of the set; true when that left it empty.
    pub(super) fn lift(&mut self, slot: usize) -> bool {
        if let Some(i) = self.as_slice().iter().position(|&s| s == slot) {
            self.ids.copy_within(i + 1..self.len, i);
            self.len -= 1;
        }
        self.len == 0
    }
}

/// "slot 1", "slots 0 and 1", "slots 0, 1 and 2".
impl fmt::Display for SlotSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.as_slice() {
            [] => write!(f, "no slot"),
            [one] => write!(f, "slot {one}"),
            [head @ .., last] => {
                let head: Vec<String> = head.iter().map(usize::to_string).collect();
                write!(f, "slots {} and {last}", head.join(", "))
            }
        }
    }
}

/// The fault that poisons the model and the slots whose state it condemned
/// ([`SlotSet`]): what every later call refuses by, and what
/// [`GpuModel::reset`] lifts once each of those slots has been reset.
#[derive(Clone)]
pub(super) struct SlotFault {
    pub(super) slots: SlotSet,
    pub(super) why: PoisonWhy,
}

/// Why the model is poisoned ([`SlotFault`]): the card's fault word a call
/// read back, a host service's refusal of the call's input, or a commit of
/// several slots' rows that settled some slots and failed on one.
#[derive(Clone)]
pub(super) enum PoisonWhy {
    /// The fault word, raised by a kernel and read by the call that carried
    /// it.
    Fault(Fault),
    /// The host tier refused the call's input ([`Refusal`]): the layers
    /// before the refused one had already written the slots' in-place state,
    /// so each of them owes its own reset before the tier serves again
    /// ([`GpuModel::reset`] lifts the tier once the set is empty).
    Refused(Refusal),
    /// [`GpuModel::commit_slots`] failed at `slot`'s keep, after the keeps
    /// before it had run: `error`, the failure's text.
    SlotKeep { slot: usize, error: String },
}

/// The captured passes of several slots ([`GpuModel::step_slots`]), one per
/// key: the busy slots in pass order and each one's row count. Model-wide,
/// never exchanged on a select.
///
/// Every capture here addresses each busy slot's own stores, and the
/// model-wide buffers (the pass arena, the input records, the heads, the
/// weights). Nothing it addresses moves while the model lives: a select
/// exchanges the handles of the slots' stores, never their memory; a key
/// names a busy set of slots that exist, and [`GpuModel::add_slots`] only
/// adds slots; [`GpuModel::reset`] moves a slot's position, not its
/// buffers. So a capture replays over the same slots' stores after any
/// later select, reset or added slot, and only a body change that moves
/// what a capture recorded drops the cache ([`GpuModel::set_mode`]). It is
/// declared before the parked slots and the body in [`GpuModel`]: a capture
/// is destroyed while every buffer it addresses is alive.
pub(super) struct SlotGraphs(Vec<(Vec<(usize, usize)>, Graph)>);

impl SlotGraphs {
    pub(super) fn new() -> SlotGraphs {
        SlotGraphs(Vec::new())
    }

    /// The capture of the pass whose slots and row counts, in pass order,
    /// are `key`, if one is held.
    fn get(&self, key: impl Iterator<Item = (usize, usize)> + Clone) -> Option<&Graph> {
        self.0
            .iter()
            .find(|(k, _)| k.iter().copied().eq(key.clone()))
            .map(|(_, g)| g)
    }

    pub(super) fn clear(&mut self) {
        self.0.clear();
    }
}

/// The key of a pass of `rows`: each busy slot and its row count, in pass
/// order.
fn key_of(rows: &[SlotRange]) -> impl Iterator<Item = (usize, usize)> + Clone + '_ {
    rows.iter().map(|r| (r.slot, r.rows.len()))
}

/// A parked sequence as the model holds it: type-erased only as far as the
/// parking needs, and given back to the body's exchange whole — the blanket
/// impl below is the only `ParkedSeq` a `B::Seq` ever becomes, so a box
/// holds exactly the sequence [`Slots::new_seq`] made.
pub(super) trait ParkedSeq<B: ChainBody>: Send {
    /// Exchange this parked sequence with `body`'s live one
    /// ([`Slots::swap_seq`]).
    fn exchange(&mut self, gpu: &Gpu, body: &mut B) -> Result<(), GpuError>;

    /// The parked sequence itself, typed: what a pass of several slots
    /// hands the body ([`SlotRows::enqueue_slots`]).
    fn seq(&mut self) -> &mut B::Seq
    where
        B: Slots;
}

impl<B: Slots> ParkedSeq<B> for B::Seq {
    fn exchange(&mut self, gpu: &Gpu, body: &mut B) -> Result<(), GpuError> {
        body.swap_seq(gpu, self)
    }

    fn seq(&mut self) -> &mut B::Seq {
        self
    }
}

/// One parked slot's state in the model's parking lot
/// ([`GpuModel::parked`]): the slot's captured chains, the position it stood
/// at, its device bytes and its sequence. The captures are declared first —
/// fields drop in declaration order, and a capture must be destroyed while
/// every buffer it addresses is alive.
pub(super) struct ParkedSlot<B: ChainBody> {
    pub(super) graphs: Graphs,
    pub(super) pos: u32,
    pub(super) bytes: usize,
    pub(super) seq: Box<dyn ParkedSeq<B>>,
}

impl<B: Slots> GpuModel<B>
where
    // A parked sequence outlives the call that parked it: it sits in the
    // model's lot until the model drops, so it can borrow nothing.
    B::Seq: 'static,
{
    /// The model serves `n` slots from now on: `n − slots()` new sequences
    /// allocated through [`Slots::new_seq`], each parked empty at position 0
    /// with no capture. The live slot keeps its state. `n == 0` and
    /// `n < slots()` are refused by name (slots are never removed);
    /// callable any time.
    pub fn add_slots(&mut self, n: usize) -> Result<(), GpuError> {
        const WHAT: &str = "GpuModel::add_slots";
        let held = self.parked.len() + 1;
        if n == 0 {
            return Err(GpuError::shape(WHAT, "a slot count of at least 1"));
        }
        if n < held {
            return Err(GpuError::shape(
                WHAT,
                format!("{n} slots of a model that already serves {held}; slots are never removed"),
            ));
        }
        let bytes = self.body.seq_bytes();
        for _ in held..n {
            let seq = self.body.new_seq(&self.gpu)?;
            self.parked.push(ParkedSlot {
                graphs: Graphs::new(),
                pos: 0,
                bytes,
                seq: Box::new(seq),
            });
        }
        Ok(())
    }

    /// The slots the model serves: 1 until [`GpuModel::add_slots`] grows it.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.parked.len() + 1
    }

    /// The slot every later call acts on: 0 until a
    /// [`GpuModel::select_slot`] moves it.
    #[must_use]
    pub fn selected(&self) -> usize {
        self.selected
    }

    /// Make `slot` the one every later call acts on, exchanging the live
    /// slot's sequence ([`Slots::swap_seq`]), position and capture cache
    /// with its parked state. Selecting the live slot is a no-op; out of
    /// range is refused by name. Pointer moves only: nothing allocated,
    /// captured, copied or synchronized.
    pub fn select_slot(&mut self, slot: usize) -> Result<(), GpuError> {
        const WHAT: &str = "GpuModel::select_slot";
        self.refuse_if_slots_wait(WHAT)?;
        let n = self.parked.len() + 1;
        if slot >= n {
            return Err(GpuError::shape(
                WHAT,
                format!("slot {slot} of a model that serves 0..{n}"),
            ));
        }
        if slot == self.selected {
            return Ok(());
        }
        // The homes rule (module doc): bring the live slot's state home —
        // slot 0's into the live fields — then park it in the target's
        // entry. One of the two exchanges runs on every switch.
        // A failed exchange leaves the live fields whole: `selected` names
        // the slot they hold after each step, so a refusal mid-switch leaves
        // slot 0 (or the old slot) live and named, never a mix.
        self.one_pass = None;
        self.slot_rows = None;
        if self.selected != 0 {
            self.swap_parked(self.selected - 1)?;
            self.selected = 0;
        }
        if slot != 0 {
            self.swap_parked(slot - 1)?;
            self.selected = slot;
        }
        Ok(())
    }

    /// Exchange the live slot's state with parked entry `i`'s: the sequence
    /// first (the one fallible part, [`Slots::swap_seq`]), then the capture
    /// cache and the position — so a refused exchange moves nothing.
    fn swap_parked(&mut self, i: usize) -> Result<(), GpuError> {
        let GpuModel {
            graphs,
            parked,
            body,
            gpu,
            pos,
            ..
        } = self;
        let p = &mut parked[i];
        // Called by name: method syntax probes the blanket impl's `B::Seq`
        // against the box itself and fails there instead of dereferencing.
        ParkedSeq::exchange(p.seq.as_mut(), gpu, body.as_mut())?;
        std::mem::swap(graphs, &mut p.graphs);
        std::mem::swap(pos, &mut p.pos);
        Ok(())
    }

    /// Where slot `slot`'s state lives (the homes rule, module doc): `None`
    /// for the live fields, else the index of its parked entry. `slot` is
    /// below [`GpuModel::slots`].
    fn parked_home(&self, slot: usize) -> Option<usize> {
        if slot == self.selected {
            None
        } else if slot == 0 {
            Some(self.selected - 1)
        } else {
            Some(slot - 1)
        }
    }

    /// The cache row slot `slot`'s next token lands in, wherever its state
    /// lives.
    fn slot_pos(&self, slot: usize, what: &'static str) -> Result<u32, GpuError> {
        match self.parked_home(slot) {
            None => Ok(self.pos),
            Some(i) => self
                .parked
                .get(i)
                .map(|p| p.pos)
                .ok_or_else(|| GpuError::shape(what, format!("slot {slot}'s parked entry {i}"))),
        }
    }

    /// Stand slot `slot` at `pos`, wherever its state lives.
    fn stand_slot(&mut self, slot: usize, pos: u32, what: &'static str) -> Result<(), GpuError> {
        match self.parked_home(slot) {
            None => self.stand_at(pos),
            Some(i) => {
                self.parked
                    .get_mut(i)
                    .ok_or_else(|| {
                        GpuError::shape(what, format!("slot {slot}'s parked entry {i}"))
                    })?
                    .pos = pos;
            }
        }
        Ok(())
    }

    /// The ranges of a pass of `rows` — each `(slot, tokens)` in pass order —
    /// with each slot's first position, or the named refusal of a pass of no
    /// slot, a slot out of range or twice, a slot of no token, more than
    /// `max_rows` rows (the body's [`SlotRows::MAX_ROWS`]), or a slot's rows
    /// past the resident cache.
    fn slot_ranges(
        &self,
        rows: impl Iterator<Item = (usize, usize)>,
        max_rows: usize,
        what: &'static str,
    ) -> Result<Vec<SlotRange>, GpuError> {
        let n = self.slots();
        let mut out: Vec<SlotRange> = Vec::new();
        let mut at = 0;
        for (slot, m) in rows {
            if slot >= n {
                return Err(GpuError::shape(
                    what,
                    format!("slot {slot} of a model that serves 0..{n}"),
                ));
            }
            if out.iter().any(|r| r.slot == slot) {
                return Err(GpuError::shape(
                    what,
                    format!("slot {slot} twice in one pass"),
                ));
            }
            if m == 0 {
                return Err(GpuError::shape(what, format!("slot {slot} with no token")));
            }
            let pos0 = self.slot_pos(slot, what)?;
            out.push(SlotRange {
                slot,
                rows: at..at + m,
                pos0,
            });
            at += m;
        }
        if out.is_empty() {
            return Err(GpuError::shape(what, "a pass of no slot"));
        }
        if at > max_rows {
            return Err(GpuError::shape(
                what,
                format!(
                    "{at} rows in one pass; this body's pass of several slots holds at most \
                     {max_rows} (SlotRows::MAX_ROWS)"
                ),
            ));
        }
        for r in &out {
            if r.pos0 as usize + r.rows.len() > self.ctx_max {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "slot {}'s {} rows from position {} pass the resident cache's {} rows",
                        r.slot,
                        r.rows.len(),
                        r.pos0,
                        self.ctx_max
                    ),
                ));
            }
        }
        Ok(out)
    }
}

impl<B: SlotRows> GpuModel<B>
where
    B::Seq: 'static,
{
    /// One pass of several slots' tokens: `rows[i]` is slot `rows[i].0`'s
    /// tokens at that slot's next positions, laid into the pass in this
    /// order ([`SlotRows`]). Returns each row's greedy next token
    /// ([`SlotsOut`]); each row's ids and logits are bit for bit those of
    /// the slot's own tokens stepped alone, and each slot's position moves
    /// by its tokens, no other's.
    ///
    /// Refused by name before anything runs: a pass of no slot, a slot out
    /// of range or twice, a slot of no token, more than the body's
    /// [`SlotRows::MAX_ROWS`] rows, a slot's rows past the resident cache
    /// and a poisoned model. The homes are made canonical first — slot 0
    /// selected, so every slot's state sits in its home and the body binds
    /// each slot's stores by the slot alone — and the call leaves slot 0
    /// selected. What the body's own walk cannot run it refuses by name in
    /// [`SlotRows::plan_slots`], after the homes are canonical and the
    /// pass's heads made. The first pass of a key (the busy slots in pass
    /// order and their row counts) makes the heads it needs and, in graph
    /// mode, captures it into the model-wide cache ([`SlotGraphs`]); eager
    /// mode runs the same enqueues uncaptured. A load whose chain holds
    /// host work runs the pass as it runs a step: the residency boundary
    /// before it, a replay served as [`SlotRows::chain_of`], a host refusal
    /// named with its row's slot, every row kept ([`PassKind::Slots`]) and
    /// the next pass's boundary made ahead before the readback. A fault the
    /// pass raised is its error and poisons every slot of the pass: only a
    /// [`GpuModel::reset`] of each of them lifts it. So does a host
    /// service's refusal of the pass's input, whose slots' state the layers
    /// before the refused one had already written. A body whose
    /// [`SlotRows::MAX_ROWS`] is not 1 to [`MAX_PASS_ROWS`] does not
    /// compile.
    pub fn step_slots(&mut self, rows: &[(usize, &[u32])]) -> Result<SlotsOut, GpuError> {
        const WHAT: &str = "GpuModel::step_slots";
        let _busy = crate::watchdog::busy(&self.watch, crate::watchdog::STEP_SLOTS, self.reads);
        let ranges = self.plan_pass(rows, SlotPass::Kept, WHAT)?;
        let out = self.run_slots(&ranges, SlotPass::Kept, WHAT);
        self.note_fault_in(WHAT, out, SlotSet::of(&ranges))
    }

    /// One pass of several slots' verify rows: `rows` as
    /// [`GpuModel::step_slots`] takes them, each slot's rows bit for bit its
    /// own verify of them, each row's greedy next token returned. Each slot
    /// stands past its rows with them waiting for
    /// [`GpuModel::commit_slots`], which keeps each slot's accepted rows;
    /// until then every call but that commit and [`GpuModel::reset`] is
    /// refused by name. No rows are kept for the residency machine and no
    /// boundary is made ahead: the kept counts are known after the
    /// readback. Refused by name before anything runs on a body that keeps
    /// a pass's rows whole ([`SlotRows::SETTLES_PARTIAL_KEEP`]): its commit
    /// could not keep a part of them. Refused, run and poisoned as
    /// `step_slots` is.
    pub fn verify_slots(&mut self, rows: &[(usize, &[u32])]) -> Result<SlotsOut, GpuError> {
        const WHAT: &str = "GpuModel::verify_slots";
        let _busy = crate::watchdog::busy(&self.watch, crate::watchdog::VERIFY_SLOTS, self.reads);
        if !B::SETTLES_PARTIAL_KEEP {
            return Err(GpuError::shape(
                WHAT,
                "this body keeps a pass's rows whole (SlotRows::keep_slot's default), so a \
                 commit cannot keep a slot's accepted rows of them \
                 (SlotRows::SETTLES_PARTIAL_KEEP)",
            ));
        }
        let ranges = self.plan_pass(rows, SlotPass::Verify, WHAT)?;
        let out = self.run_slots(&ranges, SlotPass::Verify, WHAT);
        let out = self.note_fault_in(WHAT, out, SlotSet::of(&ranges))?;
        self.slots_waiting = Some(ranges);
        Ok(out)
    }

    /// Keep the first `kept[i]` rows of the `i`-th slot of the pass waiting
    /// since [`GpuModel::verify_slots`], in its order, and take the rest
    /// back: each slot's body state ([`SlotRows::keep_slot`]) and its
    /// position, then the pass's rows for the residency machine. A count
    /// list of another length, or a count outside 1 to its slot's rows, is
    /// refused by name with the pass left waiting — every count is checked
    /// before any slot moves; with no pass waiting the call is refused. A
    /// keep that fails inside the body leaves the slots where the keeps
    /// that ran left them — a state no caller chose — so the model is
    /// poisoned until each slot of the pass is reset
    /// ([`GpuModel::reset`]) and the error names the slot.
    ///
    /// The residency machine folds each slot's accepted rows
    /// ([`PassKind::SlotsDrafted`]): the kept rows of the pass are the union
    /// of each slot's prefix, no prefix of the pass, so a rejected row
    /// leaves no trace.
    pub fn commit_slots(&mut self, kept: &[usize]) -> Result<(), GpuError> {
        const WHAT: &str = "GpuModel::commit_slots";
        let _busy = crate::watchdog::busy(&self.watch, crate::watchdog::COMMIT_SLOTS, self.reads);
        let ranges = self.slots_waiting.as_deref().ok_or(GpuError::state(
            WHAT,
            "a pass of several slots' verify rows waiting (verify_slots)",
        ))?;
        if kept.len() != ranges.len() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} kept counts for a pass of {} slots",
                    kept.len(),
                    ranges.len()
                ),
            ));
        }
        if let Some((r, k)) = ranges
            .iter()
            .zip(kept)
            .find(|(r, k)| !(1..=r.rows.len()).contains(k))
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "slot {} keeping {k} of its {} rows (row 0 always)",
                    r.slot,
                    r.rows.len()
                ),
            ));
        }
        let ranges = self.slots_waiting.take().unwrap_or_default();
        let accepted = KeptRows::of(
            ranges
                .iter()
                .zip(kept)
                .flat_map(|(r, &k)| r.rows.start..r.rows.start + k),
        );
        let mut settled = Ok(());
        for (r, &k) in ranges.iter().zip(kept) {
            if let Err(e) = self.keep_one(r, k, WHAT) {
                settled = Err(self.commit_failed(&ranges, r.slot, e));
                break;
            }
        }
        let out = settled.and_then(|()| self.keep_rows(accepted, PassKind::SlotsDrafted));
        self.note_fault_in(WHAT, out, SlotSet::of(&ranges))
    }

    /// A keep of slot `slot` that failed inside the body, after the pass's
    /// waiting was taken: the slots of `ranges` stand where the keeps that
    /// ran left them — the settled ones at their kept rows, the rest past
    /// their rejected rows — a state no caller chose, so the model is
    /// poisoned until each of them is reset ([`GpuModel::reset`], a select
    /// and a reset a slot) and the error names the slot.
    fn commit_failed(&mut self, ranges: &[SlotRange], slot: usize, e: GpuError) -> GpuError {
        self.poisoned = Some(SlotFault {
            slots: SlotSet::of(ranges),
            why: PoisonWhy::SlotKeep {
                slot,
                error: e.to_string(),
            },
        });
        GpuError::shape(
            "GpuModel::commit_slots",
            format!(
                "slot {slot}'s keep failed, the model poisoned until each slot of the pass is \
                 reset: {e}"
            ),
        )
    }

    /// Capture the pass whose slots and row counts, in pass order, are
    /// `key` (each `(slot, rows)`) into the model-wide cache without running
    /// it — the homes made canonical and the heads made first, as
    /// [`GpuModel::step_slots`] does — and return its node count. Refused
    /// as `step_slots` refuses the same rows.
    pub fn capture_slots(&mut self, key: &[(usize, usize)]) -> Result<usize, GpuError> {
        const WHAT: &str = "GpuModel::capture_slots";
        const { slots_fit::<B>() };
        self.refuse_if_slots_wait(WHAT)?;
        let ranges = self.slot_ranges(key.iter().copied(), B::MAX_ROWS, WHAT)?;
        self.canonical_homes(&ranges, WHAT)?;
        self.capture_ranges(&ranges, WHAT)
    }

    /// Every node of the captured pass of `key` ([`GpuModel::capture_slots`]),
    /// as the driver lists them.
    pub fn slots_graph_nodes(&self, key: &[(usize, usize)]) -> Result<Vec<NodeInfo>, GpuError> {
        self.slot_graphs
            .get(key.iter().copied())
            .ok_or(GpuError::state(
                "GpuModel::slots_graph_nodes",
                "no captured pass of these slots and rows",
            ))?
            .nodes()
    }

    /// Each row's logits of the last call, in its row order, when it was a
    /// [`GpuModel::step_slots`] or a [`GpuModel::verify_slots`] (`n_vocab`
    /// f32 each): row `r`'s head's, or row `r` of the pass's one head.
    /// Blocking read; gate/debug use.
    pub fn slots_logits(&self) -> Result<Vec<Vec<f32>>, GpuError> {
        const WHAT: &str = "GpuModel::slots_logits";
        let rows = self.slot_rows.ok_or(GpuError::state(
            WHAT,
            "a pass of several slots as the last call",
        ))?;
        match B::HEADS {
            RowHeads::PerRow => self
                .heads
                .get(..rows)
                .ok_or(no_head(WHAT))?
                .iter()
                .map(|h| h.logits_to_host(&self.gpu))
                .collect(),
            RowHeads::One => {
                // The head's layout is `[v·m + r]`: row r is every m-th value.
                let all = self.pass_head(rows, WHAT)?.logits_to_host(&self.gpu)?;
                Ok((0..rows)
                    .map(|r| all.iter().skip(r).step_by(rows).copied().collect())
                    .collect())
            }
        }
    }

    /// The pass of `rows` refused, planned and, in graph mode, captured:
    /// [`GpuModel::step_slots`]' and [`GpuModel::verify_slots`]' first
    /// half, which returns its ranges. A `pass` that keeps every row the
    /// body refuses ([`SlotRows::refuse_kept`]) is refused before anything
    /// is selected, made or planned.
    fn plan_pass(
        &mut self,
        rows: &[(usize, &[u32])],
        pass: SlotPass,
        what: &'static str,
    ) -> Result<Vec<SlotRange>, GpuError> {
        const { slots_fit::<B>() };
        self.refuse_if_poisoned(what)?;
        self.refuse_if_slots_wait(what)?;
        let ranges = self.slot_ranges(
            rows.iter().map(|&(slot, ids)| (slot, ids.len())),
            B::MAX_ROWS,
            what,
        )?;
        if pass == SlotPass::Kept {
            self.body.refuse_kept(&ranges)?;
        }
        let ids: Vec<u32> = rows
            .iter()
            .flat_map(|&(_, ids)| ids.iter().copied())
            .collect();
        self.canonical_homes(&ranges, what)?;
        {
            let GpuModel {
                parked, body, gpu, ..
            } = self;
            let mut seqs = parked_seqs(parked, &ranges, what)?;
            body.plan_slots(gpu.stream(), &mut seqs, &ranges, &ids)?;
        }
        if self.mode == StepMode::Graph && self.slot_graphs.get(key_of(&ranges)).is_none() {
            self.capture_ranges(&ranges, what)?;
        }
        Ok(ranges)
    }

    /// Each slot of `ranges` keeps its first `kept` rows: its body state
    /// ([`SlotRows::keep_slot`]) on its home, and its position.
    fn keep_ranges(
        &mut self,
        ranges: &[SlotRange],
        kept: &[usize],
        what: &'static str,
    ) -> Result<(), GpuError> {
        for (r, &k) in ranges.iter().zip(kept) {
            self.keep_one(r, k, what)?;
        }
        Ok(())
    }

    /// One slot of a pass's keeps: its body state settled
    /// ([`SlotRows::keep_slot`]) on its home, and its position stood at its
    /// kept rows.
    fn keep_one(&mut self, r: &SlotRange, k: usize, what: &'static str) -> Result<(), GpuError> {
        let GpuModel {
            parked, body, gpu, ..
        } = self;
        let seq = match r.slot {
            0 => None,
            s => Some(parked_seq(parked, s, what)?),
        };
        body.keep_slot(gpu, seq, r, k)?;
        let k = crate::launch_u32(what, "rows", k)?;
        self.stand_slot(r.slot, r.pos0 + k, what)
    }

    /// Before a pass of `ranges`: the homes canonical (slot 0 selected:
    /// pointer moves only, nothing when it is already) and the pass's heads
    /// made.
    fn canonical_homes(
        &mut self,
        ranges: &[SlotRange],
        what: &'static str,
    ) -> Result<(), GpuError> {
        self.select_slot(0)?;
        self.make_heads(rows_of(ranges), B::HEADS, what)
    }

    /// Capture the pass of `ranges` into the model-wide cache (replacing a
    /// capture of the same key) and return its node count.
    fn capture_ranges(
        &mut self,
        ranges: &[SlotRange],
        what: &'static str,
    ) -> Result<usize, GpuError> {
        let GpuModel {
            slot_graphs,
            parked,
            heads,
            pass_heads,
            body,
            weights,
            gpu,
            ..
        } = self;
        let heads = pass_slice(heads, pass_heads, rows_of(ranges), B::HEADS, what)?;
        let graph = gpu.capture(|_| {
            enqueue_ranges(gpu, weights, body.as_mut(), heads, parked, ranges, what)
        })?;
        let nodes = graph.node_count();
        slot_graphs
            .0
            .retain(|(k, _)| !k.iter().copied().eq(key_of(ranges)));
        slot_graphs.0.push((key_of(ranges).collect(), graph));
        Ok(nodes)
    }

    /// A pass of several slots once planned, in the step's order
    /// (`GpuModel::run_tokens`): the eager enqueue behind the residency
    /// boundary, or the replay served as [`SlotRows::chain_of`]
    /// ([`launch_served`]); a host refusal named with its row's slot; for
    /// [`SlotPass::Kept`] every row kept and the next pass's boundary made
    /// ahead; each slot's position moved past its rows, then every row's
    /// readback in row order; for [`SlotPass::Kept`] each slot's rows then
    /// kept on its home ([`SlotRows::keep_slot`]). Each host step is a no-op
    /// on a load with no host work. An error here can follow launches, so
    /// the caller passes it through [`GpuModel::note_fault_in`].
    fn run_slots(
        &mut self,
        ranges: &[SlotRange],
        pass: SlotPass,
        what: &'static str,
    ) -> Result<SlotsOut, GpuError> {
        let total = rows_of(ranges);
        self.one_pass = None;
        self.slot_rows = None;
        let r = match self.mode {
            StepMode::Eager => self.pass_boundary().and_then(|()| {
                let GpuModel {
                    parked,
                    heads,
                    pass_heads,
                    body,
                    weights,
                    gpu,
                    ..
                } = self;
                let heads = pass_slice(heads, pass_heads, total, B::HEADS, what)?;
                enqueue_ranges(gpu, weights, body.as_mut(), heads, parked, ranges, what)
            }),
            // The capture addresses each busy slot's own stores, which no
            // select since has moved ([`SlotGraphs`]).
            StepMode::Graph => {
                let GpuModel {
                    slot_graphs,
                    body,
                    gpu,
                    reads,
                    ..
                } = self;
                let graph = slot_graphs
                    .get(key_of(ranges))
                    .ok_or(GpuError::state(what, "no captured pass of these slots"))?;
                launch_served(graph, body.as_mut(), gpu, *reads, B::chain_of(ranges))
            }
        };
        self.name_host_refusal_by(r, SlotSet::of(ranges), |refusal| {
            name_slots(refusal, ranges)
        })?;
        if pass == SlotPass::Kept {
            self.keep_rows(KeptRows::prefix(total), PassKind::Slots)?;
            self.boundary_at(BoundaryAt::Ahead { reads: self.reads })?;
        }
        for r in ranges {
            let m = crate::launch_u32(what, "rows", r.rows.len())?;
            self.stand_slot(r.slot, r.pos0 + m, what)?;
        }
        self.slot_rows = Some(total);
        self.one_pass = (B::HEADS == RowHeads::One).then_some(total);
        self.reads += 1;
        let ids = match B::HEADS {
            RowHeads::PerRow => self
                .heads
                .get(..total)
                .ok_or(no_head(what))?
                .iter()
                .map(|h| h.token(&self.gpu))
                .collect::<Result<Vec<u32>, GpuError>>()?,
            RowHeads::One => {
                let ids = self.pass_head(total, what)?.tokens(&self.gpu)?;
                if ids.len() != total {
                    return Err(GpuError::shape(
                        what,
                        format!("{} tokens from a head of {total} rows", ids.len()),
                    ));
                }
                ids
            }
        };
        if pass == SlotPass::Kept {
            let all: Vec<usize> = ranges.iter().map(|r| r.rows.len()).collect();
            self.keep_ranges(ranges, &all, what)?;
        }
        Ok(SlotsOut { ids })
    }
}

/// What a pass of several slots keeps: every row as it runs
/// ([`GpuModel::step_slots`]), or each slot's accepted rows at its commit
/// ([`GpuModel::verify_slots`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotPass {
    Kept,
    Verify,
}

/// Holds when `B`'s [`SlotRows::MAX_ROWS`] is 1 to [`MAX_PASS_ROWS`].
/// Evaluated at compile time by [`GpuModel::step_slots`] and
/// [`GpuModel::capture_slots`].
const fn slots_fit<B: SlotRows>() {
    assert!(
        B::MAX_ROWS >= 1 && B::MAX_ROWS <= MAX_PASS_ROWS,
        "a body's SlotRows::MAX_ROWS is 1 to MAX_PASS_ROWS"
    );
}

/// The rows of a pass of `ranges`: where its last range ends.
fn rows_of(ranges: &[SlotRange]) -> usize {
    ranges.last().map_or(0, |r| r.rows.end)
}

/// A host refusal of a pass of `ranges` with the pass's rows named by slot:
/// the refusal's row is a row of the pass, not a position of one sequence.
fn name_slots(refusal: &mut Refusal, ranges: &[SlotRange]) {
    let rows: Vec<String> = ranges
        .iter()
        .map(|r| match r.rows.len() {
            1 => format!("row {} is slot {}'s", r.rows.start, r.slot),
            _ => format!("rows {:?} are slot {}'s", r.rows, r.slot),
        })
        .collect();
    refusal.detail = format!(
        "{} (a pass of several slots: {})",
        refusal.detail,
        rows.join(", ")
    );
}

/// Enqueue the pass of `rows` through the body, handing it each busy slot's
/// parked sequence ([`parked_seqs`]).
fn enqueue_ranges<B: SlotRows>(
    gpu: &Gpu,
    w: &Weights,
    body: &mut B,
    heads: &mut [Head],
    parked: &mut [ParkedSlot<B>],
    rows: &[SlotRange],
    what: &'static str,
) -> Result<(), GpuError>
where
    B::Seq: 'static,
{
    let mut seqs = parked_seqs(parked, rows, what)?;
    body.enqueue_slots(gpu, w, heads, &mut seqs, rows)
}

/// Each busy slot's parked sequence but slot 0's, typed, in `rows` order:
/// with the homes canonical, slot `s > 0` sits in parked entry `s − 1`,
/// and slot 0 is the live one.
fn parked_seqs<'p, B: Slots>(
    parked: &'p mut [ParkedSlot<B>],
    rows: &[SlotRange],
    what: &'static str,
) -> Result<Vec<&'p mut B::Seq>, GpuError>
where
    B::Seq: 'static,
{
    let mut homes: Vec<Option<&mut ParkedSlot<B>>> = parked.iter_mut().map(Some).collect();
    let mut seqs = Vec::with_capacity(rows.len());
    for r in rows.iter().filter(|r| r.slot != 0) {
        let p = r
            .slot
            .checked_sub(1)
            .and_then(|i| homes.get_mut(i))
            .and_then(Option::take)
            .ok_or_else(|| no_home(r.slot, what))?;
        // Called by name, as `swap_parked` calls `exchange`.
        seqs.push(ParkedSeq::<B>::seq(p.seq.as_mut()));
    }
    Ok(seqs)
}

/// Slot `slot`'s parked sequence (`slot > 0`, the homes canonical), typed.
fn parked_seq<'p, B: Slots>(
    parked: &'p mut [ParkedSlot<B>],
    slot: usize,
    what: &'static str,
) -> Result<&'p mut B::Seq, GpuError>
where
    B::Seq: 'static,
{
    let p = slot
        .checked_sub(1)
        .and_then(|i| parked.get_mut(i))
        .ok_or_else(|| no_home(slot, what))?;
    Ok(ParkedSeq::<B>::seq(p.seq.as_mut()))
}

/// Slot `slot` with no parked entry `slot − 1` to hold it, refused.
fn no_home(slot: usize, what: &'static str) -> GpuError {
    GpuError::shape(
        what,
        format!(
            "slot {slot}'s parked sequence, at entry {}",
            slot.wrapping_sub(1)
        ),
    )
}
