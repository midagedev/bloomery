//! The body's sequence state as a value: what a cut keeps ([`keep_rule`],
//! with the reason it keeps less, [`KeepLimit`]), the whole state saved to the
//! host ([`snapshot`]) and put back ([`resume`]), and where a prompt call is
//! cut so a position inside it stays keepable ([`Body::prefill_splits`]).
//!
//! The state a later step reads, as [`ChainBody::reset`] empties it:
//!
//! - saved: per layer the window ring, the compressed rows and index keys of
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
//!   refused.
//!
//! [`ChainBody::reset`]: bloomery_gpu::model::ChainBody::reset

use std::fmt;
use std::ops::Range;

use bloomery_gpu::{Gpu, GpuError};
use cuda_core::{DeviceBuffer, DeviceCopy};

use super::prefill::batches;
use super::{Body, Deepseek41Model, Holds, PAIR_ROWS};
use crate::span::{span, span_mut};

const WHAT: &str = "deepseek41 sequence state";

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
    history: Vec<u32>,
    holds: Holds,
    shadow_from: usize,
    holes: Vec<Range<usize>>,
    restore: bool,
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
            self.history.len(),
            &self.holds,
            self.shadow_from,
            &self.holes,
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
        let window = self.holds.ring.len();
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
        if self.rows_failed {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a reset: an earlier step's engram rows failed after its launch",
            });
        }
        gpu.stream().synchronize()?;
        let n = self.history.len();
        let mut kv = Vec::with_capacity(self.kv.len());
        for (i, l) in self.kv.iter().enumerate() {
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
        let runs = kept_runs(self.shadow_from, n, &self.holes);
        let (width, per) = (self.shadows.width, self.shadows.rows);
        let layers = self.kv.len();
        let total = runs.iter().map(|r| r.len()).sum::<usize>() * layers * width;
        let mut shadow: Vec<u16> = Vec::new();
        shadow
            .try_reserve_exact(total)
            .map_err(|e| GpuError::Shape {
                what: WHAT,
                detail: format!("{total} shadow values of host memory for a snapshot: {e}"),
            })?;
        let host = self.shadows.host.as_slice();
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
            history: self.history.clone(),
            holds: self.holds.clone(),
            shadow_from: self.shadow_from,
            holes: self.holes.clone(),
            restore: self.restore,
        })
    }

    /// `s` back into a body that was just reset: every saved part copied in,
    /// then the host record set to the state's. Refused by name when `s` came
    /// from a body of other layers or buffers.
    fn load(&mut self, gpu: &Gpu, s: &SeqSnapshot) -> Result<(), GpuError> {
        let n = s.history.len();
        let fits = s.layers == self.layers
            && s.kv.len() == self.kv.len()
            && s.width == self.shadows.width
            && s.runs.last().is_none_or(|r| r.end <= self.shadows.rows)
            && s.kv.iter().zip(&self.kv).all(|(a, b)| {
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
                    self.kv.len(),
                    self.shadows.width,
                    self.positions()
                ),
            });
        }
        for (l, saved) in self.kv.iter_mut().zip(&s.kv) {
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
        let (width, per, layers) = (self.shadows.width, self.shadows.rows, self.kv.len());
        let host = self.shadows.host.as_mut_slice();
        let mut from = 0;
        for r in &s.runs {
            for i in 0..layers {
                let at = (i * per + r.start) * width;
                let len = r.len() * width;
                host[at..at + len].copy_from_slice(&s.shadow[from..from + len]);
                from += len;
            }
        }
        self.history.clear();
        self.history.extend_from_slice(&s.history);
        self.holds.clone_from(&s.holds);
        self.shadow_from = s.shadow_from;
        self.holes.clone_from(&s.holes);
        self.restore = s.restore;
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
