//! Instrumentation of the step: the tap snapshots, the per-launch byte
//! accounting, the per-op profiling observer, and the node-price probe.

use crate::GpuError;
use crate::q5::Q8Blocks32;
use crate::tensor::Q8Act;
use crate::weights::DevWeight;
use cuda_core::CudaStream;
use gguf::quant::GgmlType;

/// Host copies of block 0's tap tensors for one position, in the dump's
/// logical order at that position. The fused FFN exposes neither
/// `ffn_norm-0` nor the down-projection `ffn_out-0` (the norm feeds the
/// quantizer in registers; the down store folds the residual) — `l_out-0`
/// carries that span.
pub struct Block0Taps {
    pub attn_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub kv_rope_compressed: Vec<f32>,
    /// Head-major `q_rope(h)` per head, the dump's `(d, h, t)` at one t.
    pub q_rope: Vec<f32>,
    pub k_rope: Vec<f32>,
    pub kv_compressed: Vec<f32>,
    /// Head-major `kqv(h)` per head.
    pub kqv_compressed: Vec<f32>,
    pub kqv_out: Vec<f32>,
    pub ffn_inp: Vec<f32>,
    pub l_out: Vec<f32>,
}

impl Block0Taps {
    /// First tap holding a non-finite value, if any — the gate's finiteness
    /// check over every span, including the ones not compared.
    pub fn non_finite(&self) -> Option<&'static str> {
        let all: [&str; 10] = [
            "attn_norm",
            "q",
            "kv_rope_compressed",
            "q_rope",
            "k_rope",
            "kv_compressed",
            "kqv_compressed",
            "kqv_out",
            "ffn_inp",
            "l_out",
        ];
        let vals: [&[f32]; 10] = [
            &self.attn_norm,
            &self.q,
            &self.kv_rope_compressed,
            &self.q_rope,
            &self.k_rope,
            &self.kv_compressed,
            &self.kqv_compressed,
            &self.kqv_out,
            &self.ffn_inp,
            &self.l_out,
        ];
        all.iter()
            .zip(vals)
            .find(|(_, v)| v.iter().any(|x| !x.is_finite()))
            .map(|(n, _)| *n)
    }

    /// Bit equality of every tap — the gate's rerun/replay checks.
    pub fn bits_equal(&self, other: &Block0Taps) -> bool {
        let eq = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        eq(&self.attn_norm, &other.attn_norm)
            && eq(&self.q, &other.q)
            && eq(&self.kv_rope_compressed, &other.kv_rope_compressed)
            && eq(&self.q_rope, &other.q_rope)
            && eq(&self.k_rope, &other.k_rope)
            && eq(&self.kv_compressed, &other.kv_compressed)
            && eq(&self.kqv_compressed, &other.kqv_compressed)
            && eq(&self.kqv_out, &other.kqv_out)
            && eq(&self.ffn_inp, &other.ffn_inp)
            && eq(&self.l_out, &other.l_out)
    }
}

/// Host copies of one layer's tap tensors for one position, in the dump's
/// logical order at that position. The MoE spans are empty for a layer
/// without a router, and so is `ffn_norm` (the fused dense FFN keeps its
/// normed vector in registers). The routed
/// half's `ffn_moe_out` and `ffn_out` are not here: `moe_combine` folds the
/// weighted sum, the shared expert and the residual into one store, so
/// `l_out` carries that span — `expert_down` and `moe_weights` are the
/// operands a caller can recombine.
pub struct LayerTaps {
    pub layer: usize,
    pub attn_norm: Vec<f32>,
    pub q: Vec<f32>,
    pub kv_rope_compressed: Vec<f32>,
    /// Head-major `q_rope(h)` per head, the dump's `(d, h, t)` at one t.
    pub q_rope: Vec<f32>,
    pub k_rope: Vec<f32>,
    pub kv_compressed: Vec<f32>,
    /// Head-major `kqv(h)` per head.
    pub kqv_compressed: Vec<f32>,
    pub kqv_out: Vec<f32>,
    pub ffn_inp: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub moe_logits: Vec<f32>,
    /// The router's chosen expert ids, rank order — the `sel` the expert
    /// kernels read.
    pub moe_ids: Vec<u32>,
    /// The router weight of each chosen expert, same order.
    pub moe_weights: Vec<f32>,
    /// Each slot's down projection, slot-major (`n_used * hidden`).
    pub expert_down: Vec<f32>,
    pub ffn_shexp: Vec<f32>,
    pub l_out: Vec<f32>,
}

impl LayerTaps {
    /// Every f32 span with its name, in forward order — the finiteness and
    /// bit-equality checks walk this one list so a new tap cannot be added
    /// to the struct and forgotten by the checks.
    fn spans(&self) -> [(&'static str, &[f32]); 14] {
        [
            ("attn_norm", &self.attn_norm),
            ("q", &self.q),
            ("kv_rope_compressed", &self.kv_rope_compressed),
            ("q_rope", &self.q_rope),
            ("k_rope", &self.k_rope),
            ("kv_compressed", &self.kv_compressed),
            ("kqv_compressed", &self.kqv_compressed),
            ("kqv_out", &self.kqv_out),
            ("ffn_inp", &self.ffn_inp),
            ("ffn_norm", &self.ffn_norm),
            ("moe_logits", &self.moe_logits),
            ("moe_weights", &self.moe_weights),
            ("expert_down", &self.expert_down),
            ("ffn_shexp", &self.ffn_shexp),
        ]
    }

    /// First tap holding a non-finite value, if any — the gate's finiteness
    /// check over every span, including the ones not compared.
    pub fn non_finite(&self) -> Option<&'static str> {
        self.spans()
            .into_iter()
            .chain([("l_out", self.l_out.as_slice())])
            .find(|(_, v)| v.iter().any(|x| !x.is_finite()))
            .map(|(n, _)| n)
    }

    /// Bit equality of every tap, the routed ids included — the gate's
    /// rerun/replay checks.
    pub fn bits_equal(&self, other: &LayerTaps) -> bool {
        let eq = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        self.layer == other.layer
            && self.moe_ids == other.moe_ids
            && eq(&self.l_out, &other.l_out)
            && self
                .spans()
                .into_iter()
                .zip(other.spans())
                .all(|((_, a), (_, b))| eq(a, b))
    }
}

/// Device bytes one launch touches, each distinct byte counted once: the
/// weight rows it addresses, the activation planes it reads, the spans it
/// writes. Allocation padding no thread addresses (a `Q8Act` group tail, a
/// q5 `q_stride` window) is not counted, and a byte several blocks read
/// counts once — the number is the op's traffic, the divisor of its
/// effective GB/s. `None` is an op whose count is not derivable from the
/// shapes this file holds; the profile prints it as `?`.
pub(crate) type Bytes = Option<u64>;

/// Sum of byte parts, `None` if any part is not derivable.
pub(super) fn bsum(parts: &[Option<usize>]) -> Bytes {
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
/// padding past the last block is allocated but never addressed.
pub(super) fn weight_bytes(w: &DevWeight, rows: usize) -> Option<usize> {
    let k = w.k();
    Some(match w {
        DevWeight::KQuant { ty, .. } => rows * kq_row_bytes(*ty, k)?,
        DevWeight::Q5_0 { .. } => rows * 36 * (k / 32),
        DevWeight::Q5_1 { .. } => rows * 40 * (k / 32),
        DevWeight::Q8_0 { .. } | DevWeight::Q8_0Derived { .. } => rows * (k + 4 * (k / 32)),
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
pub(super) fn act_write_bytes(a: &Q8Act, cols: usize) -> usize {
    let (q3, q4, q6, s8, d8) = act_planes(a.k());
    cols * (q3 + q4 + q6 + s8 + d8)
}

/// Bytes `cols` activation columns cost the gemv of `w`: Q3_K loads the u64
/// code plane and the block scales, Q4_K the 32-bit codes, the group sums
/// and the scales, Q6_K its own code plane and the scales. `None` for a
/// weight that is not a K-quant.
pub(super) fn gemv_act_bytes(w: &DevWeight, a: &Q8Act, cols: usize) -> Option<usize> {
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
pub(super) fn blocks32_bytes(b: &Q8Blocks32, cols: usize) -> usize {
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
pub(super) struct ProfRec {
    pub(super) ops: Vec<(&'static str, Bytes, Vec<f64>)>,
    pub(super) last: std::time::Instant,
}

impl ProfRec {
    /// Synchronize after op `i`'s enqueue and record the wall time since the
    /// previous sync — the op's eager launch + body + one synchronize. The
    /// name and the byte count must be the same on every rep: both are
    /// functions of the shapes, so a rep that changes either would be
    /// timing a different chain.
    pub(super) fn observe(
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
/// the five small quantize launches and prices their removal, work included.
///
/// The `flash_*` levers ask the other question — what one STAGE of the
/// attention walk costs — and they answer it by doing that stage a second
/// time (`crate::flash::TWICE_QK` and its siblings). They are launch-shape
/// neutral and value neutral: the same node count, and the same tokens as
/// the shipped path, which `gate_e2e` pins. The slowdown of one against
/// `base` is a lower bound on that stage's price, the second pass running
/// against a warm cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepProbe {
    /// Empty `probe::touch` launches enqueued once per layer. The captured
    /// step is a linear chain, so the site does not change a node's marginal
    /// cost; these sit at the end of the attention half.
    pub pad_per_layer: usize,
    /// Skip the layer's small quantize launches — `kqvc`'s (whether it is
    /// riding inside the attention launch or standing alone), `act_ao`, the
    /// routed experts' 32-value quantize and the shared expert's q8_1. Their
    /// consumers then read whatever the activation buffers already hold
    /// (zeros from load), which addresses the same bytes and runs the same
    /// launches. `kqvc`'s is the one that has moved into a producer, so this
    /// also takes the attention launch back to its plain twin.
    pub skip_quant: bool,
    /// Run the head gemv as the two half-launches instead of the merged
    /// one. Value-neutral — the merged launch is bit-identical to the pair,
    /// so this is the same-binary arm a sub-1 % claim about the merge is
    /// judged against, and the rollback if one is ever needed.
    pub split_heads: bool,
    /// The same, for the `kqvc` q8_1 quantization: two launches instead of
    /// the one that covers both halves. Only reachable with
    /// `split_flash_quant`, which is what puts that quantization back on a
    /// launch of its own.
    pub split_kqvc: bool,
    /// Take the `kqvc` q8_1 quantization back out of the attention launch
    /// and give it its own, the shape before the side output was folded in.
    /// Value-neutral — the folded twin writes the bytes the standalone
    /// quantizer writes — so this is the same-binary arm the fold's claim is
    /// judged against, and the rollback if one is ever needed.
    pub split_flash_quant: bool,
    /// The MoE half's two quantizations as two launches instead of the one
    /// that carries both geometries. The launch order around them does not
    /// change, so this arm isolates the merge itself.
    pub split_moe_quant: bool,
    /// Run the flash segment pass's QK dot twice, over the same key rows.
    pub flash_qk2: bool,
    /// The same, over rows one key tile along inside the segment — the same
    /// work on rows the tile did not just read, so the pair brackets what a
    /// cache hit is worth in that loop.
    pub flash_qk2c: bool,
    /// Run the segment pass's V accumulation twice, over the same rows.
    pub flash_v2: bool,
    /// The same, over the neighbouring tile's rows.
    pub flash_v2c: bool,
    /// Run the key butterfly and the two warp reductions twice.
    pub flash_coll2: bool,
    /// Give every tile a second pair of block barriers.
    pub flash_sync2: bool,
    /// Run warp 0's softmax arithmetic twice, both exponentials included.
    pub flash_sm2: bool,
    /// Run the merge pass's fold over the segment partials twice.
    pub flash_merge2: bool,
}

impl StepProbe {
    /// The flash stage-doubling arms, in the order
    /// `generate --ab-set keyaxis` rotates them: the shipped path and each
    /// lever alone. One list, so the gate that pins their tokens and the
    /// runner that times them cannot disagree about what an arm is.
    pub fn keyaxis_arms() -> [(&'static str, StepProbe); 9] {
        let arm = |set: fn(&mut StepProbe)| {
            let mut p = StepProbe::default();
            set(&mut p);
            p
        };
        [
            ("base", StepProbe::default()),
            ("flash_qk2", arm(|p| p.flash_qk2 = true)),
            ("flash_qk2c", arm(|p| p.flash_qk2c = true)),
            ("flash_v2", arm(|p| p.flash_v2 = true)),
            ("flash_v2c", arm(|p| p.flash_v2c = true)),
            ("flash_coll2", arm(|p| p.flash_coll2 = true)),
            ("flash_sync2", arm(|p| p.flash_sync2 = true)),
            ("flash_sm2", arm(|p| p.flash_sm2 = true)),
            ("flash_merge2", arm(|p| p.flash_merge2 = true)),
        ]
    }

    /// The segment-pass probe entry this probe selects, as the stage bit and
    /// the second pass's row offset; `None` is the shipped entry.
    pub(super) fn flash_seg_twice(&self) -> Option<(u32, usize)> {
        [
            (self.flash_qk2, crate::flash::TWICE_QK, 0),
            (
                self.flash_qk2c,
                crate::flash::TWICE_QK,
                crate::flash::KEY_TILE,
            ),
            (self.flash_v2, crate::flash::TWICE_V, 0),
            (
                self.flash_v2c,
                crate::flash::TWICE_V,
                crate::flash::KEY_TILE,
            ),
            (self.flash_coll2, crate::flash::TWICE_COLL, 0),
            (self.flash_sync2, crate::flash::TWICE_SYNC, 0),
            (self.flash_sm2, crate::flash::TWICE_SM, 0),
        ]
        .into_iter()
        .find(|&(on, _, _)| on)
        .map(|(_, bit, shift)| (bit, shift))
    }

    /// Refuse a flash lever that would silently do nothing — the shape a
    /// later round reads as a broken lever. Two segment-pass levers would
    /// each claim the one segment launch; a cache short enough to hold one
    /// segment runs a kernel with no probe twin at all; and the merge lever
    /// probes the q8 merge, which `skip_quant` and `split_flash_quant`
    /// replace with the plain one.
    pub(super) fn check(&self, cache_rows: usize) -> Result<(), GpuError> {
        let seg = [
            self.flash_qk2,
            self.flash_qk2c,
            self.flash_v2,
            self.flash_v2c,
            self.flash_coll2,
            self.flash_sync2,
            self.flash_sm2,
        ]
        .iter()
        .filter(|on| **on)
        .count();
        if seg > 1 {
            return Err(GpuError::shape(
                "StepProbe",
                "one flash segment-pass lever at a time — each names the probe \
                 entry the segment launch runs, and they are one launch",
            ));
        }
        if (seg == 1 || self.flash_merge2) && crate::flash::segments_for(cache_rows) == 1 {
            return Err(GpuError::shape(
                "StepProbe",
                format!(
                    "the flash levers probe the split launch, and a {cache_rows}-row \
                 cache takes the single-block kernel — raise ctx or drop the lever"
                ),
            ));
        }
        if self.flash_merge2 && (self.skip_quant || self.split_flash_quant) {
            return Err(GpuError::shape(
                "StepProbe",
                "flash_merge2 probes the q8 merge, and skip_quant / \
                 split_flash_quant put the plain merge back in its place",
            ));
        }
        Ok(())
    }
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
pub(super) fn tick(
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
pub(super) type Observer<'a> = dyn FnMut(usize, &'static str, Bytes) -> Result<(), GpuError> + 'a;
