//! The body's sequence state as a value: the live sequence's state as one
//! value ([`Seq`]), which the model's resident slots exchange by pointer
//! moves ([`Slots`]); what a cut keeps ([`keep_rule`], with the reason it
//! keeps less, [`KeepLimit`]); the whole state saved to the host
//! ([`snapshot`]) and put back ([`resume`]); and where a prompt call is cut
//! so a position inside it stays keepable ([`Body::prefill_splits`]).
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
//! - a failed step's refusal and the fault word: [`resume`] resets first,
//!   and a snapshot of a poisoned model or of a step whose rows failed is
//!   refused. Whether a step's rows failed belongs to the sequence it ran on
//!   ([`Seq`]); the fault word and the host tier's poison are the model's.
//!
//! Nothing the body captures records a sequence's buffers: the step and the
//! pair pass are the model's captures, one cache a slot, so [`Seq`] carries
//! none ([`Slots::Seq`]'s contract).
//!
//! [`ChainBody::reset`]: bloomery_gpu::model::ChainBody::reset

use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Range;

use bloomery_gpu::model::Slots;
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer, DeviceCopy};
use model::arch::deepseek41::hparams::Hparams;

use super::prefill::batches;
use super::{Body, Deepseek41Model, Holds, LayerKv, PAIR_ROWS, Shadows, layer_kv};
use crate::span::{span, span_mut};

const WHAT: &str = "deepseek41 sequence state";

/// The tokens a sequence holds, one per position: what each step's engram
/// n-grams read back from (`Planner::plan_into`'s `before`), cut, cleared and
/// saved with the sequence. The one owner of the sequence's per-position
/// record: vision's engram mask (the image positions whose engram step is
/// off) joins it here as a second per-position field, cut and cleared with
/// the ids.
#[derive(Clone, Debug, Default, Hash)]
pub(super) struct History {
    ids: Vec<u32>,
}

impl History {
    /// An empty history with room for `positions` tokens.
    fn with_capacity(positions: usize) -> History {
        History {
            ids: Vec::with_capacity(positions),
        }
    }

    /// The tokens, in position order.
    pub(super) fn ids(&self) -> &[u32] {
        &self.ids
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

    /// The positions from `n` on taken back.
    pub(super) fn truncate(&mut self, n: usize) {
        self.ids.truncate(n);
    }

    /// No position.
    pub(super) fn clear(&mut self) {
        self.ids.clear();
    }

    /// Exactly `ids`, from position 0.
    pub(super) fn set(&mut self, ids: &[u32]) {
        self.ids.clear();
        self.ids.extend_from_slice(ids);
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
/// live body and of a [`SeqSnapshot`] alike — and the rule that moved it
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

/// One layer's cache as saved: the ring whole, the compressed rows and index
/// keys of the positions held, the compressor state whole.
struct LayerSaved {
    ring: Vec<u16>,
    rows: Vec<u16>,
    keys: Vec<u16>,
    values: Vec<f32>,
    scores: Vec<f32>,
}

/// A body's sequence state on the host ([`snapshot`]); [`resume`] puts it
/// back into the body it came from.
pub struct SeqSnapshot {
    layers: Range<usize>,
    kv: Vec<LayerSaved>,
    /// The shadow's width and its positions per layer, which the saved rows
    /// are laid out by.
    width: usize,
    /// The shadow positions saved, as runs; per run, per layer, its rows.
    runs: Vec<Range<usize>>,
    shadow: Vec<u16>,
    history: History,
    holds: Holds,
    shadow_from: usize,
    holes: Vec<Range<usize>>,
    restore: bool,
}

/// Every part a save copies, in the order [`snapshot`] lays it out, the f32
/// states by their bits: two snapshots hash alike when a [`resume`] of either
/// puts back the same sequence.
impl Hash for SeqSnapshot {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.layers.hash(h);
        for l in &self.kv {
            l.ring.hash(h);
            l.rows.hash(h);
            l.keys.hash(h);
            for v in l.values.iter().chain(&l.scores) {
                v.to_bits().hash(h);
            }
        }
        self.width.hash(h);
        self.runs.hash(h);
        self.shadow.hash(h);
        self.history.hash(h);
        self.holds.hash(h);
        self.shadow_from.hash(h);
        self.holes.hash(h);
        self.restore.hash(h);
    }
}

impl SeqSnapshot {
    /// The positions it holds.
    #[must_use]
    pub fn positions(&self) -> usize {
        self.history.len()
    }

    /// Its host bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let kv: usize = self
            .kv
            .iter()
            .map(|l| {
                2 * (l.ring.len() + l.rows.len() + l.keys.len())
                    + 4 * (l.values.len() + l.scores.len())
            })
            .sum();
        kv + 2 * self.shadow.len() + 4 * self.history.len()
    }

    /// [`Body::keep_point`] of the body right after a [`resume`] of this
    /// state.
    #[must_use]
    pub fn keep_point(&self, n: usize) -> usize {
        keep_rule(
            self.history.len(),
            &self.holds,
            self.shadow_from,
            &self.holes,
            n,
        )
        .0
    }
}

/// `len` values of host memory, refused by name when the allocation fails.
fn host_vec<T: Clone + Default>(len: usize) -> Result<Vec<T>, GpuError> {
    let mut v = Vec::new();
    v.try_reserve_exact(len).map_err(|e| GpuError::Shape {
        what: WHAT,
        detail: format!("{len} values of host memory for a snapshot: {e}"),
    })?;
    v.resize(len, T::default());
    Ok(v)
}

/// The first `len` values of `buf`, device to host (a blocking copy).
fn read<T: DeviceCopy + Clone + Default>(
    gpu: &Gpu,
    buf: &DeviceBuffer<T>,
    len: usize,
) -> Result<Vec<T>, GpuError> {
    let mut v = host_vec(len)?;
    if len > 0 {
        span(WHAT, buf, 0, len)?.copy_to_host(gpu.stream(), &mut v)?;
    }
    Ok(v)
}

/// `vals` into the first values of `buf`, host to device (a blocking copy).
fn write<T: DeviceCopy>(gpu: &Gpu, buf: &mut DeviceBuffer<T>, vals: &[T]) -> Result<(), GpuError> {
    if vals.is_empty() {
        return Ok(());
    }
    span_mut(WHAT, buf, 0, vals.len())?.copy_from_host(gpu.stream(), vals)?;
    Ok(())
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

    /// The sequence state to the host: the rows in flight delivered, the
    /// stream finished, then every saved part copied (module comment).
    fn save(&mut self, gpu: &Gpu) -> Result<SeqSnapshot, GpuError> {
        self.arrive()?;
        self.rows.finish()?;
        if self.seq.rows_failed {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a reset: an earlier step's engram rows failed after its launch",
            });
        }
        gpu.stream().synchronize()?;
        let n = self.seq.history.len();
        let mut kv = Vec::with_capacity(self.seq.kv.len());
        for (i, l) in self.seq.kv.iter().enumerate() {
            let layer = self.layers.start + i;
            let held = |rows: usize| -> Result<usize, GpuError> {
                let ratio = self.hp.layers.get(layer).map_or(0, |k| k.ratio() as usize);
                if ratio == 0 {
                    return Err(GpuError::Shape {
                        what: WHAT,
                        detail: format!("layer {layer} holds compressed rows and has no ratio"),
                    });
                }
                Ok(n.div_ceil(ratio).min(rows))
            };
            let rows = match l.rows.as_ref() {
                Some(t) => read(gpu, t.buf(), held(t.rows())? * t.cols())?,
                None => Vec::new(),
            };
            let keys = match l.keys.as_ref() {
                Some(t) => read(gpu, t.buf(), held(t.rows())? * t.cols())?,
                None => Vec::new(),
            };
            let whole = |t: Option<&bloomery_gpu::DeviceTensor<f32>>| match t {
                Some(t) => read(gpu, t.buf(), t.buf().len()),
                None => Ok(Vec::new()),
            };
            kv.push(LayerSaved {
                ring: read(gpu, l.ring.buf(), l.ring.buf().len())?,
                rows,
                keys,
                values: whole(l.values.as_ref())?,
                scores: whole(l.scores.as_ref())?,
            });
        }
        let runs = kept_runs(self.seq.shadow_from, n, &self.seq.holes);
        let (width, per) = (self.seq.shadows.width, self.seq.shadows.rows);
        let layers = self.seq.kv.len();
        let total = runs.iter().map(|r| r.len()).sum::<usize>() * layers * width;
        let mut shadow: Vec<u16> = Vec::new();
        shadow
            .try_reserve_exact(total)
            .map_err(|e| GpuError::Shape {
                what: WHAT,
                detail: format!("{total} shadow values of host memory for a snapshot: {e}"),
            })?;
        let host = self.seq.shadows.host.as_slice();
        for r in &runs {
            for i in 0..layers {
                let at = (i * per + r.start) * width;
                shadow.extend_from_slice(&host[at..at + r.len() * width]);
            }
        }
        Ok(SeqSnapshot {
            layers: self.layers.clone(),
            kv,
            width,
            runs,
            shadow,
            history: self.seq.history.clone(),
            holds: self.seq.holds.clone(),
            shadow_from: self.seq.shadow_from,
            holes: self.seq.holes.clone(),
            restore: self.seq.restore,
        })
    }

    /// `s` back into a body that was just reset: every saved part copied in,
    /// then the host record set to the state's. Refused by name when `s` came
    /// from a body of other layers or buffers.
    fn load(&mut self, gpu: &Gpu, s: &SeqSnapshot) -> Result<(), GpuError> {
        let n = s.history.len();
        let fits = s.layers == self.layers
            && s.kv.len() == self.seq.kv.len()
            && s.width == self.seq.shadows.width
            && s.runs.last().is_none_or(|r| r.end <= self.seq.shadows.rows)
            && s.kv.iter().zip(&self.seq.kv).all(|(a, b)| {
                a.ring.len() == b.ring.buf().len()
                    && a.values.len() == b.values.as_ref().map_or(0, |t| t.buf().len())
                    && a.scores.len() == b.scores.as_ref().map_or(0, |t| t.buf().len())
            })
            && n <= self.positions();
        if !fits {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a state of {n} positions over layers {:?} ({} caches, shadow width {}) into \
                     a body of layers {:?} ({} caches, shadow width {}, {} positions)",
                    s.layers,
                    s.kv.len(),
                    s.width,
                    self.layers,
                    self.seq.kv.len(),
                    self.seq.shadows.width,
                    self.positions()
                ),
            });
        }
        for (l, saved) in self.seq.kv.iter_mut().zip(&s.kv) {
            write(gpu, l.ring.buf_mut(), &saved.ring)?;
            if let Some(t) = l.rows.as_mut() {
                write(gpu, t.buf_mut(), &saved.rows)?;
            }
            if let Some(t) = l.keys.as_mut() {
                write(gpu, t.buf_mut(), &saved.keys)?;
            }
            if let Some(t) = l.values.as_mut() {
                write(gpu, t.buf_mut(), &saved.values)?;
            }
            if let Some(t) = l.scores.as_mut() {
                write(gpu, t.buf_mut(), &saved.scores)?;
            }
        }
        // No step in flight writes a shadow row while the host does.
        gpu.stream().synchronize()?;
        let (width, per, layers) = (
            self.seq.shadows.width,
            self.seq.shadows.rows,
            self.seq.kv.len(),
        );
        let host = self.seq.shadows.host.as_mut_slice();
        let mut from = 0;
        for r in &s.runs {
            for i in 0..layers {
                let at = (i * per + r.start) * width;
                let len = r.len() * width;
                host[at..at + len].copy_from_slice(&s.shadow[from..from + len]);
                from += len;
            }
        }
        self.seq.history.clone_from(&s.history);
        self.seq.holds.clone_from(&s.holds);
        self.seq.shadow_from = s.shadow_from;
        self.seq.holes.clone_from(&s.holes);
        self.seq.restore = s.restore;
        self.need = None;
        if let Some(tap) = self.tap.as_mut() {
            tap.pos = [None; PAIR_ROWS];
        }
        Ok(())
    }
}

/// The positions `shadow_from .. n` outside `holes` (ascending, disjoint), as
/// runs.
fn kept_runs(shadow_from: usize, n: usize, holes: &[Range<usize>]) -> Vec<Range<usize>> {
    let mut runs = Vec::new();
    let mut p = shadow_from;
    for h in holes {
        if h.start > p {
            runs.push(p..h.start.min(n));
        }
        p = p.max(h.end);
    }
    if p < n {
        runs.push(p..n);
    }
    runs.retain(|r| !r.is_empty());
    runs
}

/// `m`'s sequence state on the host. Refused on a poisoned model: its caches
/// hold what a fault condemned.
pub fn snapshot(m: &mut Deepseek41Model) -> Result<SeqSnapshot, GpuError> {
    if let Some(fault) = m.poisoned() {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!("a snapshot of a model a fault poisoned ({fault:?})"),
        });
    }
    let (gpu, _, body) = m.body_parts(WHAT)?;
    body.save(gpu)
}

/// Replace `m`'s sequence state with `s`, which [`snapshot`] took of this
/// model: a reset, then — as a pass of `s.positions()` positions whose work is
/// the copies — the state put back; the model stands at `s.positions()`, and
/// the steps after it are bit for bit those after the state was taken.
pub fn resume(m: &mut Deepseek41Model, s: &SeqSnapshot) -> Result<(), GpuError> {
    m.reset()?;
    let n = s.positions();
    if n == 0 {
        return Ok(());
    }
    m.run_rows(n, WHAT, |gpu, _, body, _, _| {
        body.load(gpu, s)?;
        Ok(false)
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::kept_runs;

    #[test]
    fn kept_runs_skip_the_holes() {
        let first = 0..40;
        let one = std::slice::from_ref(&first);
        assert_eq!(kept_runs(0, 100, &[]), vec![0..100]);
        assert_eq!(kept_runs(0, 100, one), vec![40..100]);
        assert_eq!(kept_runs(10, 100, &[0..40, 60..70]), vec![40..60, 70..100]);
        assert_eq!(kept_runs(50, 100, &[0..40, 60..70]), vec![50..60, 70..100]);
        assert_eq!(kept_runs(0, 30, one), Vec::<std::ops::Range<usize>>::new());
    }
}
