//! The body's sequence state saved to the host and put back: the live
//! sequence's every saved part as one value ([`SeqSnapshot`], [`snapshot`])
//! and back into the model ([`resume`]) — what the server's prompt cache
//! holds — and the same state as bytes ([`save_state`], [`restore_state`]),
//! what a slot file carries.
//!
//! A snapshot copies the sequence's own parts ([`super::Seq`]): per layer the
//! window ring whole, the compressed rows and index keys of the positions
//! held, the compressor state whole; the ring shadows' rows a cut may restore
//! (from `shadow_from`, outside the prompt calls' holes); the token history,
//! the slots' record, the holes and a pending ring restore.

use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::ops::Range;

use bloomery_gpu::{Gpu, GpuError};
use cuda_core::{DeviceBuffer, DeviceCopy};

use super::seq::{History, keep_rule};
use super::{Body, Deepseek41Model, Holds, PAIR_ROWS};
use crate::span::{span, span_mut};

const WHAT: &str = "deepseek41 sequence state";

/// Why [`save_state`] or [`restore_state`] failed.
#[derive(Debug)]
pub enum StateFail {
    /// Refused by name: a state of another model file, build, layout or
    /// context, a stream that is not one, or a sequence that cannot be saved
    /// as it stands. A save leaves the sequence as it was; a restore may have
    /// read part of the stream, and the caller resets the sequence.
    Refused(GpuError),
    /// The card failed mid-copy: the model's error.
    Card(GpuError),
}

/// The selected sequence's state ([`snapshot`]) written to `out` in the form
/// [`restore_state`] reads, the sequence unchanged; the bytes written.
/// Refused by name: the byte form is not built.
pub fn save_state(m: &mut Deepseek41Model, out: &mut dyn Write) -> Result<u64, StateFail> {
    let _ = (m, out);
    Err(StateFail::Refused(bytes_not_built("deepseek41 save_state")))
}

/// The state `input` carries, which [`save_state`] wrote of a model of the
/// same file, build and shape, put back into the selected sequence as
/// [`resume`] puts a snapshot back; the positions it holds. Refused by name:
/// the byte form is not built.
pub fn restore_state(m: &mut Deepseek41Model, input: &mut dyn Read) -> Result<usize, StateFail> {
    let _ = (m, input);
    Err(StateFail::Refused(bytes_not_built(
        "deepseek41 restore_state",
    )))
}

/// The refusal of the byte form.
fn bytes_not_built(what: &'static str) -> GpuError {
    GpuError::State {
        what,
        missing: "the byte form of a V4.1 sequence state (round v41snap builds it)",
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
