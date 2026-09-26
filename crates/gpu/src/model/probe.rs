//! Instrumentation of the step, shared by every architecture: the per-launch
//! byte accounting, the per-op profiling observer, and the node-price probe.
//! The tap snapshots an architecture reads back live beside its chain.

use crate::GpuError;
use crate::q5::Q8Blocks32;
use crate::tensor::Q8Act;
use crate::weights::{DevWeight, resident_size};
use cuda_core::CudaStream;
use gguf::quant::GgmlType;

/// Device bytes one launch touches, each distinct byte counted once: the
/// weight rows it addresses, the activation planes it reads, the spans it
/// writes. Allocation padding no thread addresses (a `Q8Act` group tail, a
/// q5 `q_stride` window) is not counted, and a byte several blocks read
/// counts once — the number is the op's traffic, the divisor of its
/// effective GB/s. `None` is an op whose count is not derivable from the
/// shapes this file holds; the profile prints it as `?`.
pub(crate) type Bytes = Option<u64>;

/// Sum of byte parts, `None` if any part is not derivable.
pub(crate) fn bsum(parts: &[Option<usize>]) -> Bytes {
    parts
        .iter()
        .try_fold(0usize, |acc, p| Some(acc + (*p)?))
        .map(|v| v as u64)
}

/// Bytes of one K-quant row: the type's super-block size times the row's
/// super-block count. `None` for a type with no block geometry or a `k`
/// that is not a whole number of blocks.
fn kq_row_bytes(ty: GgmlType, k: usize) -> Option<usize> {
    let blck = ty.blck_size()? as usize;
    if blck == 0 || !k.is_multiple_of(blck) {
        return None;
    }
    Some(ty.type_size()? as usize * (k / blck))
}

/// Bytes of `rows` rows of a resident weight in the layout its kernel
/// addresses. The q5 rows are eight code words plus the block's scale
/// (Q5_0) or scale and min (Q5_1) per 32 values — the `q_stride` window
/// padding past the last block is allocated but never addressed. The q8_0
/// planes are addressed whole, so theirs are the card format's own bytes.
pub(crate) fn weight_bytes(w: &DevWeight, rows: usize) -> Option<usize> {
    let k = w.k();
    Some(match w {
        DevWeight::KQuant { ty, .. } => rows * kq_row_bytes(*ty, k)?,
        DevWeight::Q5_0 { .. } => rows * 36 * (k / 32),
        DevWeight::Q5_1 { .. } => rows * 40 * (k / 32),
        DevWeight::Q8_0 { .. } | DevWeight::Q8_0Derived { .. } => {
            resident_size(GgmlType::Q8_0, k, rows)?
        }
        DevWeight::F32 { .. } => rows * k * 4,
    })
}

/// Bytes of one q8_1 activation column's five planes at `k` values, in the
/// order `(q3, q4, q6, s8, d8)` — the geometry `Q8Act::with_k` allocates,
/// minus the group tails it never writes.
fn act_planes(k: usize) -> (usize, usize, usize, usize, usize) {
    let n_sb = k / 256;
    (
        8 * 64 * n_sb.div_ceil(2),
        4 * 256 * n_sb.div_ceil(4),
        4 * 128 * n_sb.div_ceil(2),
        4 * 8 * n_sb,
        4 * 2 * n_sb,
    )
}

/// Bytes `cols` columns of the q8_1 quantizer write: every plane, since one
/// quantize serves the Q3_K, Q4_K and Q6_K gemvs alike.
pub(crate) fn act_write_bytes(a: &Q8Act, cols: usize) -> usize {
    let (q3, q4, q6, s8, d8) = act_planes(a.k());
    cols * (q3 + q4 + q6 + s8 + d8)
}

/// Bytes `cols` activation columns cost the gemv of `w`: Q3_K loads the u64
/// code plane and the block scales, Q4_K the 32-bit codes, the group sums
/// and the scales, Q6_K its own code plane and the scales. `None` for a
/// weight that is not a K-quant.
pub(crate) fn gemv_act_bytes(w: &DevWeight, a: &Q8Act, cols: usize) -> Option<usize> {
    let (q3, q4, q6, s8, d8) = act_planes(a.k());
    let DevWeight::KQuant { ty, .. } = w else {
        return None;
    };
    Some(
        cols * match ty {
            GgmlType::Q3_K => q3 + d8,
            GgmlType::Q4_K => q4 + s8 + d8,
            GgmlType::Q6_K => q6 + d8,
            _ => return None,
        },
    )
}

/// Bytes of `cols` columns of the 32-value q8 blocks the q5 gemvs read and
/// their quantizer writes: eight code words, one scale and one sum per 32
/// values (the `q_stride` padding is never addressed).
pub(crate) fn blocks32_bytes(b: &Q8Blocks32, cols: usize) -> usize {
    cols * 40 * (b.k() / 32)
}

/// Per-op timing of one layer chain run from [`GpuModel::profile_layer`](crate::GpuModel::profile_layer):
/// `us_mean`/`us_min` over the measured reps of that op's eager launch +
/// body + the one stream synchronize the profiling observer issues after
/// it. Every op carries the same sync overhead; subtract a touch-launch+sync
/// constant (the gate's `sync_floor_us`) to compare op bodies. `bytes` is
/// the op's [`Bytes`] count, recorded by the same `tick` that names it.
pub struct OpTime {
    pub index: usize,
    pub name: &'static str,
    pub us_mean: f64,
    pub us_min: f64,
    pub bytes: Bytes,
}

/// The profiling observer's state: per-op sample lists with the op's byte
/// count, and the wall clock of the previous synchronize. Owned by
/// [`GpuModel::profile_layer`](crate::GpuModel::profile_layer)'s rep loop, borrowed by the observer closure
/// for one chain run at a time.
pub(crate) struct ProfRec {
    pub(crate) ops: Vec<(&'static str, Bytes, Vec<f64>)>,
    pub(crate) last: std::time::Instant,
}

impl ProfRec {
    /// Synchronize after op `i`'s enqueue and record the wall time since the
    /// previous sync — the op's eager launch + body + one synchronize. The
    /// name and the byte count must be the same on every rep: both are
    /// functions of the shapes, so a rep that changes either would be
    /// timing a different chain.
    pub(crate) fn observe(
        &mut self,
        i: usize,
        name: &'static str,
        bytes: Bytes,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        stream.synchronize()?;
        let now = std::time::Instant::now();
        let us = now.duration_since(self.last).as_secs_f64() * 1e6;
        self.last = now;
        if i == self.ops.len() {
            self.ops.push((name, bytes, Vec::new()));
        }
        let slot = &mut self.ops[i];
        if slot.0 != name {
            return Err(GpuError::shape(
                "profile_layer",
                format!("op {i} was {} on an earlier rep, now {name}", slot.0),
            ));
        }
        if slot.1 != bytes {
            return Err(GpuError::shape(
                "profile_layer",
                format!(
                    "op {i} ({name}) touched {:?} bytes on an earlier rep, now {bytes:?}",
                    slot.1
                ),
            ));
        }
        slot.2.push(us);
        Ok(())
    }
}

/// What the step's node-price probe does to the chain. Both levers are off
/// in every value-carrying step; a chain either lever armed is a TIMING
/// INSTRUMENT and its logits are not the model's answer.
///
/// The question it exists to answer is what one graph node costs in the
/// assembled step, which no per-op table can say: a per-op row is an eager
/// launch plus a synchronize, and the step runs as graph nodes.
/// `pad_per_layer` adds empty nodes and prices the slope; `skip_quant` drops
/// the small quantize launches and prices their removal, work included.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepProbe {
    /// Empty `probe::touch` launches enqueued once per layer. The captured
    /// step is a linear chain, so the site does not change a node's marginal
    /// cost; these sit at the end of the attention half.
    pub pad_per_layer: usize,
    /// Skip the layer's small quantize launches — `act_ao`'s, the routed
    /// experts' 32-value quantize and the shared expert's q8_1 — and the
    /// `kqvc` quantization that rides inside the attention launch, which
    /// takes that launch back to its plain twin. Their consumers then read
    /// whatever the activation buffers already hold (zeros from load), which
    /// addresses the same bytes and runs the same launches.
    pub skip_quant: bool,
}

/// Tick the observer for op `*i` and advance. `obs` fires on the host after
/// every enqueue, carrying the op's index (one per launch, in chain order),
/// the name of the numbered step it belongs to and the [`Bytes`] that
/// launch touches; returning `Err` aborts the chain at that op. The byte
/// count is computed at the tick, from the same shapes and buffers the
/// launch above it was given, so it cannot name a different op's traffic.
/// The normal and captured paths pass a no-op, which issues byte-for-byte
/// the same launches in the same order as an uninstrumented chain — the
/// observer is host-side only and never touches the stream. The op index is
/// the graph node index of the same launch.
pub(crate) fn tick(
    i: &mut usize,
    obs: &mut Observer<'_>,
    name: &'static str,
    bytes: Bytes,
) -> Result<(), GpuError> {
    obs(*i, name, bytes)?;
    *i += 1;
    Ok(())
}

/// The host-side observer a chain enqueue ticks: `(op index, op name, the
/// bytes that launch touches)`.
pub(crate) type Observer<'a> = dyn FnMut(usize, &'static str, Bytes) -> Result<(), GpuError> + 'a;

/// The rep loop a per-op profile runs: 20 warm-up runs whose samples are
/// discarded, then `reps` measured runs of `run` under the profiling
/// observer.
pub(crate) fn profile_reps(
    stream: &CudaStream,
    reps: u32,
    run: &mut impl FnMut(&mut Observer<'_>) -> Result<(), GpuError>,
) -> Result<ProfRec, GpuError> {
    let mut rec = ProfRec {
        // Slot i is created by op i's first firing (indices arrive in
        // chain order); the name and the byte count are cross-checked on
        // every later rep.
        ops: Vec::new(),
        last: std::time::Instant::now(),
    };
    const WARMUP: u32 = 20;
    for rep in 0..(WARMUP + reps) {
        rec.last = std::time::Instant::now();
        let mut obs = |i: usize, name: &'static str, b: Bytes| rec.observe(i, name, b, stream);
        run(&mut obs)?;
        if rep < WARMUP {
            for (_, _, s) in rec.ops.iter_mut() {
                s.clear();
            }
        }
    }
    Ok(rec)
}

/// Err unless a capture of `run` holds exactly `observed` nodes.
pub(crate) fn check_one_node_per_tick(
    gpu: &crate::Gpu,
    observed: usize,
    run: &mut impl FnMut(&mut Observer<'_>) -> Result<(), GpuError>,
) -> Result<(), GpuError> {
    // The observer bills the window between two ticks to one op, so a
    // tick that covered two launches would fold them into one row with
    // no sign of it. Capture the same chain into a throwaway graph (the
    // stage's own capture is untouched) and require one node per tick:
    // the node count is where a folded pair shows.
    let probe = gpu.capture(|_| run(&mut |_, _, _| Ok(())))?;
    if observed != probe.node_count() {
        return Err(GpuError::shape(
            "profile_layer",
            format!(
                "{} ops observed but the same chain captures {} nodes — an op \
             issued more than one launch before its tick, so its neighbours' times are \
             mis-attributed",
                observed,
                probe.node_count()
            ),
        ));
    }
    Ok(())
}
