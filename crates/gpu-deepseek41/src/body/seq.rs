//! The body's sequence state as a value: the live sequence's state as one
//! value ([`Seq`]), which the model's resident slots exchange by pointer
//! moves ([`Slots`]); which sequence each row of a pass binds
//! ([`RowSeqs`]); what a cut keeps ([`keep_rule`], with the reason it
//! keeps less, [`KeepLimit`]); and where a prompt call is cut so a position
//! inside it stays keepable ([`Body::prefill_splits`]). The whole state saved
//! to the host and put back is [`super::snap`]'s.
//!
//! The state a later step reads, as [`ChainBody::reset`] empties it:
//!
//! - one sequence's own ([`Seq`]), saved and exchanged: per layer the window
//!   ring, the compressed rows and index keys of
//!   the positions held (`⌈n / ratio⌉` rows; a row past them is written by the
//!   step that completes it before any step reads it), the compressor state;
//!   the ring shadows' rows a cut may restore (from `shadow_from`, outside the
//!   prompt calls' holes: a cut never reads another); the token history (the
//!   engram steps read it), the ring and state slots' positions ([`Holds`]),
//!   the holes and a pending ring restore;
//! - written by every step before it is read: the streams, folds, lists and
//!   image copies of the rows, the pieces' scratch — the source compressor's
//!   pooled rows among them, of which a step reads only the groups it
//!   completes — and a prompt call's needs, which the call's start sets;
//! - a failed step's refusal and the fault word: [`super::resume`] resets
//!   first, and a snapshot of a poisoned model or of a step whose rows failed
//!   is refused. Whether a step's rows failed belongs to the sequence it ran on
//!   ([`Seq`]); the fault word and the host tier's poison are the model's.
//!
//! Nothing the body captures records a sequence's buffers: the step and the
//! pair pass are the model's captures, one cache a slot, and a pass of
//! several slots is the model's, in its one cache ([`super::slots`]), so
//! [`Seq`] carries none ([`Slots::Seq`]'s contract).
//!
//! [`ChainBody::reset`]: bloomery_gpu::model::ChainBody::reset

use std::fmt;
use std::ops::Range;

use bloomery_gpu::model::Slots;
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::CudaStream;
use model::arch::deepseek41::hparams::Hparams;

use super::prefill::batches;
use super::{Body, Holds, LayerKv, PAIR_ROWS, Shadows, layer_kv};

/// The tokens a sequence holds, one per position: what each step's engram
/// n-grams read back from (`Planner::plan_into`'s `before`), cut, cleared and
/// saved with the sequence. The one owner of the sequence's per-position
/// record: the media positions — the spans of a vision prompt, whose engram
/// hash is DEAD and whose lookback the blocked chain pads
/// ([`Planner::plan_dead_into`]) — join it here as a second per-position
/// field, cut and cleared with the ids.
#[derive(Clone, Debug, Default, Hash)]
pub(super) struct History {
    ids: Vec<u32>,
    /// The media positions, ascending disjoint ranges. Positions are `u32`,
    /// the plan's own type, so a decode step hands them to
    /// [`Planner::plan_dead_into`] as they are.
    media: Vec<Range<u32>>,
}

impl History {
    /// An empty history with room for `positions` tokens.
    fn with_capacity(positions: usize) -> History {
        History {
            ids: Vec::with_capacity(positions),
            media: Vec::new(),
        }
    }

    /// The tokens, in position order.
    pub(super) fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// The media positions, as [`Planner::plan_dead_into`]'s `dead`.
    pub(super) fn media(&self) -> &[Range<u32>] {
        &self.media
    }

    pub(super) fn len(&self) -> usize {
        self.ids.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The token at the next position.
    pub(super) fn push(&mut self, id: u32) {
        self.ids.push(id);
    }

    /// `ids` at the next positions, in order.
    pub(super) fn extend(&mut self, ids: &[u32]) {
        self.ids.extend_from_slice(ids);
    }

    /// A prompt call's media spans, absolute and past every range held: the
    /// call adds them before its ids, so each of its chunks plans them dead.
    pub(super) fn extend_media(&mut self, media: &[Range<u32>]) {
        self.media.extend_from_slice(media);
    }

    /// The positions from `n` on taken back, and every media range that does
    /// not end by `n` — [`Body::rollback`] refuses a cut inside one first, so
    /// such a range lies wholly past the cut.
    pub(super) fn truncate(&mut self, n: usize) {
        self.ids.truncate(n);
        if let Ok(n) = u32::try_from(n) {
            self.media.retain(|r| r.end <= n);
        }
    }

    /// No position.
    pub(super) fn clear(&mut self) {
        self.ids.clear();
        self.media.clear();
    }

    /// Exactly `ids` from position 0, and the media positions `media`,
    /// refused by name unless each range is non-empty, inside the ids, and
    /// the ranges ascend disjoint.
    pub(super) fn set_media(&mut self, ids: &[u32], media: &[Range<u32>]) -> Result<(), GpuError> {
        let len = u32::try_from(ids.len()).map_err(|_| GpuError::Shape {
            what: "deepseek41 Body::set_history",
            detail: format!("{} tokens pass u32", ids.len()),
        })?;
        let outside = |r: &Range<u32>| r.start >= r.end || r.end > len;
        if media.iter().any(outside) || media.windows(2).any(|p| p[0].end > p[1].start) {
            return Err(GpuError::Shape {
                what: "deepseek41 Body::set_history",
                detail: format!(
                    "the media positions {media:?} of a history of {len} tokens: each range \
                     inside it, non-empty, ascending and disjoint"
                ),
            });
        }
        self.ids.clear();
        self.ids.extend_from_slice(ids);
        self.media.clear();
        self.media.extend_from_slice(media);
        Ok(())
    }
}

/// One sequence's state (the module comment's own parts): every layer's
/// cache and its ring shadow, the token history, which position each ring
/// and state slot holds, where the shadow is written from and its holes, a
/// pending ring restore, and whether a step's rows failed on it. The body
/// holds the live one; each resident slot past it holds its own, exchanged
/// with the live one by pointer moves ([`Slots::swap_seq`]).
pub struct Seq {
    /// Per layer of the body's layers, in order.
    pub(super) kv: Vec<LayerKv>,
    /// Every layer's ring shadow, in page-locked host memory.
    pub(super) shadows: Shadows,
    /// The tokens decoded so far: `ctx_max` reserved.
    pub(super) history: History,
    /// Which position's row each ring slot and each compressor state slot
    /// holds, as the steps refreshed since the last known state left them.
    pub(super) holds: Holds,
    /// The first position whose shadow row a step of the current history
    /// wrote: rows below it hold nothing a cut may restore (a caller wrote
    /// the caches itself, [`Body::set_history`]).
    pub(super) shadow_from: usize,
    /// Positions of the history whose shadow rows some layer never wrote:
    /// each prompt call's [`super::Need::hole`], in position order. A cut
    /// whose restore reads one is not granted.
    pub(super) holes: Vec<Range<usize>>,
    /// A cut left ring slots the next step reads holding other positions'
    /// rows: the next refresh restores them before the step runs.
    pub(super) restore: bool,
    /// A step's rows failed after its launch: the card ran it on the rows
    /// the staging held before, and every step on this sequence is refused
    /// until its reset.
    pub(super) rows_failed: bool,
}

impl Seq {
    /// A sequence of `layers` of the model `hp` describes at `ctx_max`
    /// positions in the state the load leaves: zeroed caches and shadows, no
    /// history, the slots' record over the streams of `ratios`. Allocates on
    /// `gpu`'s card and pins host memory: load-time only.
    pub(super) fn new(
        gpu: &Gpu,
        hp: &Hparams,
        layers: Range<usize>,
        ctx_max: usize,
        ratios: &[u32],
    ) -> Result<Seq, GpuError> {
        gpu.context().bind_to_thread()?;
        let stream = gpu.stream();
        let kv = layers
            .clone()
            .map(|l| layer_kv(stream, hp, l, ctx_max))
            .collect::<Result<Vec<_>, _>>()?;
        let shadows = Shadows::new(gpu, layers.len(), ctx_max, hp.head_dim)?;
        let ring_rows = kv.first().map_or(0, |k| k.ring.rows());
        Ok(Seq {
            kv,
            shadows,
            history: History::with_capacity(ctx_max),
            holds: Holds::new(ring_rows, ratios),
            shadow_from: 0,
            holes: Vec::new(),
            restore: false,
            rows_failed: false,
        })
    }

    /// Rows of each layer's window ring.
    pub(super) fn ring_rows(&self) -> usize {
        self.holds.ring.len()
    }

    /// Device bytes: every layer's cache and compressor state.
    pub(super) fn device_bytes(&self) -> usize {
        self.kv
            .iter()
            .map(|l| l.buffers().iter().map(|&(_, n)| n).sum::<usize>())
            .sum()
    }

    /// Page-locked host bytes: every layer's ring shadow.
    pub(super) fn shadow_bytes(&self) -> usize {
        self.shadows.host.num_bytes()
    }

    /// Back to the state the load leaves, on `stream`: every cache zeroed in
    /// place (a captured chain keeps their addresses), the history emptied,
    /// the record of a known empty state. The shadows keep their rows: a cut
    /// reads only rows a step since wrote.
    pub(super) fn clear(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        for layer in &mut self.kv {
            layer.zero(stream)?;
        }
        self.history.clear();
        self.holds.known(0);
        self.shadow_from = 0;
        self.holes.clear();
        self.restore = false;
        self.rows_failed = false;
        Ok(())
    }

    /// Before the step or batch whose first position is `at`: the ring slots
    /// a cut left holding other positions' rows put back from the shadows,
    /// host to device on `stream`, when a restore is pending.
    pub(super) fn restore_before(
        &mut self,
        stream: &CudaStream,
        at: usize,
    ) -> Result<(), GpuError> {
        if !self.restore {
            return Ok(());
        }
        for run in self.holds.stale_runs(at) {
            for (i, layer) in self.kv.iter_mut().enumerate() {
                self.shadows
                    .restore(i, &mut layer.ring, stream, run.clone())?;
            }
            run.for_each(|q| self.holds.ring_wrote(q));
        }
        self.restore = false;
        Ok(())
    }
}

/// The sequences a pass's rows bind ([`super::Parts`]): one for every row
/// — the live one for the step, the verify pair and the prompt call, or the
/// one slot's of a pass of one slot's two rows — or one a row for a pass of
/// two slots of one row each ([`super::slots`]).
pub(super) enum RowSeqs<'a> {
    One(&'a mut Seq),
    Two([&'a mut Seq; PAIR_ROWS]),
}

impl RowSeqs<'_> {
    /// Row `row`'s sequence; a row past the pass's is refused by name.
    pub(super) fn of(&mut self, row: usize) -> Result<&mut Seq, GpuError> {
        match self {
            RowSeqs::One(seq) => Ok(seq),
            RowSeqs::Two(seqs) => seqs.get_mut(row).map(|s| &mut **s).ok_or(GpuError::State {
                what: "deepseek41 Body::enqueue_chain",
                missing: "the row's sequence",
            }),
        }
    }
}

impl Slots for Body {
    type Seq = Seq;

    /// A sequence of the live one's shape at the load's `ctx_max`, in the
    /// state the load leaves ([`Seq::new`]); refused by name past the
    /// sequences the load's plan counts ([`Body::open_placed_slots`]), the
    /// live one included.
    fn new_seq(&mut self, gpu: &Gpu) -> Result<Seq, GpuError> {
        if self.slots_made >= self.slots_planned {
            return Err(GpuError::Shape {
                what: "deepseek41 Body::new_seq",
                detail: format!(
                    "sequence {} of a plan that counts {} slots; plan the load for as many \
                     resident sequences as it serves (body::open_slots)",
                    self.slots_made + 1,
                    self.slots_planned
                ),
            });
        }
        let seq = Seq::new(
            gpu,
            &self.hp,
            self.layers.clone(),
            self.positions(),
            self.planner.stream_ratios(),
        )?;
        self.slots_made += 1;
        Ok(seq)
    }

    /// The pointer form of a save and a load: the rows in flight delivered
    /// first, as a save takes them — a failure there is returned with
    /// nothing exchanged, and marks the live sequence's rows failed — then
    /// the live sequence and `seq` exchanged, and what a load clears
    /// cleared: the last call's needs and the feature rows' positions, which
    /// were the other sequence's. No device work.
    fn swap_seq(&mut self, _gpu: &Gpu, seq: &mut Seq) -> Result<(), GpuError> {
        self.arrive()?;
        self.rows.finish()?;
        std::mem::swap(&mut self.seq, seq);
        self.need = None;
        if let Some(tap) = self.tap.as_mut() {
            tap.pos = [None; PAIR_ROWS];
        }
        Ok(())
    }

    /// The live sequence's device bytes ([`Seq`]'s caches and compressor
    /// states); its ring shadows are page-locked host memory
    /// ([`Body::seq_shadow_bytes`]).
    fn seq_bytes(&self) -> usize {
        self.seq.device_bytes()
    }
}

impl Body {
    /// Page-locked host bytes one sequence holds: its ring shadows, which
    /// the plan counts on the host (`KvBytes::shadow_bytes`) and
    /// [`Slots::seq_bytes`] leaves out.
    #[must_use]
    pub fn seq_shadow_bytes(&self) -> usize {
        self.seq.shadow_bytes()
    }

    /// The resident sequences the load's plan counts
    /// ([`Body::open_placed_slots`]): the most the model makes.
    #[must_use]
    pub fn slots_planned(&self) -> usize {
        self.slots_planned
    }
}

/// Why [`Body::keep_point`] keeps less than it was asked: the last rule that
/// moved the cut down.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeepLimit {
    /// A compressor state ring no longer holds the positions of the group
    /// the cut falls in: the cut moved from `from` down to `to`.
    State { from: usize, to: usize },
    /// The window row of position `row`, which the step at the cut reads, is
    /// in a prompt call's hole: the cut moved to the hole's start.
    Hole { row: usize, hole: Range<usize> },
    /// The window row of position `row` lies below the first shadow row a
    /// step wrote: nothing is kept.
    NoShadow { row: usize, shadow_from: usize },
}

impl fmt::Display for KeepLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeepLimit::State { from, to } => write!(
                f,
                "compressor state: the ring no longer holds the group of position {from}, cut \
                 to {to}"
            ),
            KeepLimit::Hole { row, hole } => write!(
                f,
                "window row {row} lies in the CED hole {hole:?} of a prompt call: cut to the \
                 call's start {}",
                hole.start
            ),
            KeepLimit::NoShadow { row, shadow_from } => write!(
                f,
                "window row {row} lies below the first shadow row {shadow_from}: nothing kept"
            ),
        }
    }
}

/// The longest prefix of at most `n` positions a cut keeps, of a state of
/// `len` positions whose slots `holds` records, whose shadow holds rows from
/// `shadow_from` on outside `holes` — [`Body::keep_point`]'s rule, of the
/// live body and of a [`super::SeqSnapshot`] alike — and the rule that moved it
/// below `n`, if one did.
pub(super) fn keep_rule(
    len: usize,
    holds: &Holds,
    shadow_from: usize,
    holes: &[Range<usize>],
    n: usize,
) -> (usize, Option<KeepLimit>) {
    if n >= len {
        return (len, None);
    }
    let mut k = n;
    let mut why = None;
    loop {
        let from = k;
        while k > 0 && !holds.state_keeps(k) {
            k -= 1;
        }
        if k < from {
            why = Some(KeepLimit::State { from, to: k });
        }
        let unwritten = holds.stale(k).find_map(|q| {
            if q < shadow_from {
                Some((q, None))
            } else {
                holes
                    .iter()
                    .find(|h| h.contains(&q))
                    .map(|h| (q, Some(h.clone())))
            }
        });
        match unwritten {
            None => return (k, why),
            Some((row, None)) => return (0, Some(KeepLimit::NoShadow { row, shadow_from })),
            Some((row, Some(hole))) => {
                k = hole.start;
                why = Some(KeepLimit::Hole { row, hole });
            }
        }
    }
}

impl Body {
    /// [`Body::keep_point`] and the rule that kept less than `n`, if one did.
    #[must_use]
    pub fn keep_why(&self, n: usize) -> (usize, Option<KeepLimit>) {
        keep_rule(
            self.seq.history.len(),
            &self.seq.holds,
            self.seq.shadow_from,
            &self.seq.holes,
            n,
        )
    }

    /// The positions of a prompt call of `first .. end` whose shadow rows
    /// some layer leaves unwritten: [`super::Need::hole`] of the needs the
    /// call's start would set, fed as [`batches`] with no feature tap.
    #[must_use]
    pub fn call_hole(&self, first: usize, end: usize) -> Range<usize> {
        if end <= first {
            return first..first;
        }
        let starts: Vec<usize> = batches(first, end - first)
            .iter()
            .map(|r| r.start)
            .collect();
        self.ced.need(first, end, &starts, None).hole()
    }

    /// Where to cut a prompt call of `first .. end` into calls so that each
    /// of `marks` a call would leave in its hole stays keepable: walking the
    /// marks in order, a mark at least `min` past the current call's start
    /// whose window rows — and, for the state ring's parity, the row before
    /// them — the rest of the call would leave in its hole starts a call.
    #[must_use]
    pub fn prefill_splits(
        &self,
        first: usize,
        end: usize,
        marks: &[usize],
        min: usize,
    ) -> Vec<usize> {
        let window = self.seq.holds.ring.len();
        let mut at = Vec::new();
        let mut from = first;
        for &u in marks {
            if u < from.saturating_add(min) || u >= end {
                continue;
            }
            let hole = self.call_hole(from, end);
            let reads = u.saturating_sub(window)..u;
            if hole.start < reads.end && reads.start < hole.end {
                at.push(u);
                from = u;
            }
        }
        at
    }
}
