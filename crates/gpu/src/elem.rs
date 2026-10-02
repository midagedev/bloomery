//! P4: every kernel of a decode step that is not a matmul, attention or the
//! router — the embedding row dequant, rms_norm, rope (the YaRN cos/sin cache
//! is computed on the host and uploaded; the kernel applies it), swiglu, the
//! residual add, the routed-expert weighted sum and argmax. One
//! `#[cuda_module]` in its own file (docs/gpu-design.md decision 6); the
//! arithmetic bodies are cores above the module so a later fused block kernel
//! can call the same bodies.
//!
//! Buffer layout, every op: ggml's token-major order — token `t`'s values are
//! contiguous, the token axis slowest. `rms_norm`/`swiglu`/`add` work on flat
//! `width * m` spans; `rope` on `[n_dims, n_vec, m]` (column `c = t*n_vec +
//! v`, the layout of the reference dump's `q_rope`/`k_rope` views);
//! `weighted_sum` on `[rows, n_exp, m]` down-projections with `[n_exp, m]`
//! weights; the embedding table is Q3_K rows of 2048 values (880 bytes = 220
//! u32 words each) or Q4_K, Q5_K or Q6_K rows of whole super-blocks.
//! Extents are launch arguments, never buffer lengths: scratch buffers may
//! be larger than the shape in flight.

use crate::GpuError;
use crate::cores::{funnel16, half_to_f32, q3k_aux_scales, q3k_sub_scale, q4k_scale_min};
use crate::fault::{FaultSink, FaultSite, LAYER_NONE};
use crate::flash::{f32_to_f16_bits, f32x2_to_f16x2_bits};
use crate::hybrid::HOST;
use crate::launch_u32;
use crate::tensor::DeviceTensor;
use crate::view::{Apply, Elem, Elem2, RopePair};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads a norm gives one token. The row's serial memory depth is
/// `k / RMS_THREADS` loads per thread, not `k / 32`: one warp per token
/// leaves a single warp resident for the whole launch, so nothing is in
/// flight to cover a load's latency. `rms_norm` and the fused `norm_quant`
/// share this geometry — their sum of squares must agree bit for bit.
pub const RMS_THREADS: usize = 256;
/// [`RMS_THREADS`] as the `u32` block width a launch takes.
pub(crate) const RMS_THREADS_U32: u32 = RMS_THREADS as u32;
const _: () = assert!(RMS_THREADS_U32 as usize == RMS_THREADS);
/// Warps in that block, and so the width of the second-stage combine.
pub const RMS_WARPS: usize = RMS_THREADS / 32;

/// f32 patterns `half_encode_check` also runs through the device build of
/// `f32_to_f16_bits` alone, for the host to hold against its own: the
/// patterns `x` with `x · HALF_ENCODE_MUL < HALF_ENCODE_SAMPLES` (mod 2^32),
/// slot `x · HALF_ENCODE_MUL` — an odd multiplier, so a bijection that
/// scatters the sample over the whole space.
pub const HALF_ENCODE_SAMPLES: usize = 1 << 20;
/// The sample's multiplier (odd).
pub const HALF_ENCODE_MUL: u32 = 0x9e37_79b1;
const _: () = assert!(HALF_ENCODE_MUL % 2 == 1);
/// `half_encode_check`'s report words: mismatches on a non-NaN input, on a
/// NaN input, and the smallest mismatching input (`u32::MAX` for none).
pub const HALF_ENCODE_REPORT: usize = 3;
/// Threads a `half_encode_check` block runs; the grid is `2^32 /` this.
const HALF_ENCODE_THREADS: u32 = 256;
// The kernel's launch contract states these three as literals.
const _: () = assert!(
    HALF_ENCODE_SAMPLES == 1_048_576 && HALF_ENCODE_REPORT == 3 && HALF_ENCODE_THREADS == 256
);

/// Threads the argmax gives one vector: a whole block's worth, since the
/// scan is a latency chain — each thread's strided loads (the head's row is
/// the vocabulary: V2-Lite's 102,400, V4.1's 129,280, Qwen3's 151,936
/// logits) wait on one another, and more threads mean fewer loads each. The
/// scan is strided by this width, the merge is a per-warp butterfly then a
/// fixed ascending walk of the warp slots. Every argmax kernel here declares
/// the same width in its launch contract.
pub const ARGMAX_THREADS: usize = 1024;
/// [`ARGMAX_THREADS`] as the `u32` block width a launch takes.
const ARGMAX_THREADS_U32: u32 = ARGMAX_THREADS as u32;
const _: () = assert!(ARGMAX_THREADS_U32 as usize == ARGMAX_THREADS);
// The kernels' `launch_bounds` and `launch_contract` spell the width out.
const _: () = assert!(ARGMAX_THREADS == 1024);
/// Warps in that block, and so the width of the argmax's final combine.
pub const ARGMAX_WARPS: usize = ARGMAX_THREADS / 32;

// ------------------------------------------------------------------ cores
//
// Same standing as `crate::cores`: ordinary `#[inline(always)]` functions a
// per-op wrapper and a later fused kernel can both call. Slice arguments are
// the whole device buffer plus indices — the verified core shape
// (`cores::q4k_a_chain` takes a buffer and a base).

/// The f32 value of Q3_K weight `v16` (0..256) of the super-block at byte
/// `base` of `w`, in the reference dequantizer's op order: `d_all *
/// (scale - 32)`, then that times `(qv - hv)` — plain multiplies, no fused
/// op — so a correct caller is bit-identical to `gguf::quant::dequant_row`
/// on the same row bytes. The 2-bit code sits in qs byte `32·half +
/// 16·half16 + l` at field `2·field`; the high bit is hmask byte `v16 % 32`
/// bit `4·half + field` (clear subtracts 4); the sub-block scale is the
/// aux-shuffle byte `v16/16` minus 32; `d` is the f16 at super-block bytes
/// 108..109. A super-block starts 0 or 2 mod 4 (a word plus an even byte
/// count): the scale window funnels the 2-mod-4 case, single bytes load from
/// their covering words.
///
/// # Safety
///
/// `base + 110 <= 4 * w.len()` and `v16 < 256`. The qs byte is at most
/// `base + 95`, the hmask byte below `base + 32`, and the scale window's four
/// words end at or before the first word boundary at or past `base + 110`;
/// `4 * w.len()` is a word boundary at or past `base + 110`, so every word
/// read is inside `w`. A kernel caller discharges both from its launch facts
/// (`crate::view`'s module doc): the `requires` clause bounding `w` at the
/// super-block its thread's guarded index names, and `v16` taken `% 256`.
#[inline(always)]
pub unsafe fn q3k_embed_value(w: &[u32], base: usize, v16: usize) -> f32 {
    let field = (v16 >> 5) & 3;
    let qs_byte = 32 * (v16 >> 7) + 16 * ((v16 >> 4) & 1) + (v16 & 15);
    // Single bytes load directly from their covering word — the value's qs
    // and hmask bytes sit at arbitrary byte offsets (the gemv reads whole
    // aligned quads and funnels; a lone byte needs no funnel).
    let qx = base + 32 + qs_byte;
    // SAFETY: qx <= base + 95 < base + 110 <= 4 * w.len() by this fn's
    // `# Safety`, so the covering word is inside w.
    let qsw = unsafe { *w.get_unchecked(qx >> 2) };
    let qv = (qsw >> (8 * (qx & 3) + 2 * field)) & 3;

    let hx = base + (v16 & 31);
    // SAFETY: hx < base + 32, inside the super-block by this fn's `# Safety`.
    let hmw = unsafe { *w.get_unchecked(hx >> 2) };
    let hv = if (hmw >> (8 * (hx & 3) + 4 * (v16 >> 7) + field)) & 1 != 0 {
        0
    } else {
        4
    };

    let par = (base >> 1) & 1;

    let ak = (base + 96) >> 2;
    // SAFETY: the 12 scale bytes end at 108 and d at 110, inside the
    // super-block by this fn's `# Safety`.
    let (aw0, aw1, aw2, aw3) = unsafe {
        (
            *w.get_unchecked(ak),
            *w.get_unchecked(ak + 1),
            *w.get_unchecked(ak + 2),
            *w.get_unchecked(ak + 3),
        )
    };
    let (a0w, a1w, a2w) = if par == 0 {
        (aw0, aw1, aw2)
    } else {
        (funnel16(aw0, aw1), funnel16(aw1, aw2), funnel16(aw2, aw3))
    };
    let ts = q3k_aux_scales(a0w, a1w, a2w);
    let sc = q3k_sub_scale(&ts, v16 >> 4);
    let d_bits = if par == 0 {
        (aw3 & 0xffff) as u16
    } else {
        (aw3 >> 16) as u16
    };
    (half_to_f32(d_bits) * sc as f32) * ((qv as i32 - hv) as f32)
}

/// The f32 value of Q4_K weight `v` (0..256) of the super-block at word
/// `wk` of `w`, in the reference dequantizer's op order: sub-block `2j + h`
/// (`j = v / 64`, `h` the nibble half) takes `d1 = d · sc`, `m1 = dmin · m`
/// from `get_scale_min_k4`, and the value is `q · d1 − m1` for its 4-bit code
/// `q` (low nibble of qs byte `32j + v % 32` for `h = 0`, high for `h = 1`).
/// `q · d1` holds at most 21 significant bits, so it is exact in f32 and a
/// fused or unfused subtraction rounds alike: a correct caller is
/// bit-identical to `gguf::quant::dequant_row` on the same row bytes.
///
/// # Safety
///
/// `wk + 36 <= w.len()` (the super-block's 36 words) and `v < 256`: every
/// word read is `wk + 0..=35`. A kernel caller discharges both from its
/// launch facts (`crate::view`'s module doc): the `requires` clause bounding
/// `w` at the super-block its thread's guarded index names, and `v` taken
/// `% 256`.
#[inline(always)]
pub(crate) unsafe fn q4k_embed_value(w: &[u32], wk: usize, v: usize) -> f32 {
    let j = v >> 6;
    let h = (v >> 5) & 1;
    let l = v & 31;
    let qb = 32 * j + l;
    // SAFETY: every index is wk + 0..=35 (qb < 128, so 4 + qb/4 <= 35),
    // inside `w` by this fn's `# Safety`.
    let (w0, w1, w2, w3, qw) = unsafe {
        (
            *w.get_unchecked(wk),
            *w.get_unchecked(wk + 1),
            *w.get_unchecked(wk + 2),
            *w.get_unchecked(wk + 3),
            *w.get_unchecked(wk + 4 + (qb >> 2)),
        )
    };
    let byte = (qw >> (8 * (qb & 3) as u32)) & 0xff;
    let q = if h == 0 { byte & 0x0f } else { byte >> 4 };
    let (sc, mi) = q4k_scale_min(2 * j + h, w1, w2, w3);
    let d = half_to_f32((w0 & 0xffff) as u16);
    let dmin = half_to_f32((w0 >> 16) as u16);
    let d1 = d * sc as f32;
    let m1 = dmin * mi as f32;
    q as f32 * d1 - m1
}

/// The byte `x` of the byte stream `w` holds.
///
/// # Safety
///
/// `x < 4 * w.len()`.
#[inline(always)]
unsafe fn stream_byte(w: &[u32], x: usize) -> u32 {
    // SAFETY: x >> 2 < w.len() by this fn's `# Safety`.
    (unsafe { *w.get_unchecked(x >> 2) } >> (8 * (x & 3))) & 0xff
}

/// A K-quant super-block as an embedding row reads it: its bytes in the
/// file, and the f32 value of weight `v` (0..256) of the super-block at byte
/// `base` of the row stream `w`, bit-identical to `gguf::quant::dequant_row`.
pub(crate) trait EmbedSb {
    /// Bytes of one super-block (ggml's `block_q*_K`).
    const BYTES: usize;

    /// # Safety
    ///
    /// `base + BYTES <= 4 * w.len()`, `base` a multiple of 4 where the type's
    /// super-block is whole words, and `v < 256`.
    unsafe fn value(w: &[u32], base: usize, v: usize) -> f32;
}

/// Q5_K (176 bytes: `d`, `dmin`, 12 scale bytes, the 32 `qh` bytes, 128
/// nibble bytes; whole words): sub-block `2j + h` (`j = v / 64`, `h` the
/// nibble half) takes `d1 = d · sc`, `m1 = dmin · m` from
/// `get_scale_min_k4`, its code is the nibble of qs byte `32j + v % 32` plus
/// 16 when bit `2j + h` of qh byte `v % 32` is set, and the value is
/// `q · d1 − m1`. `q · d1` holds at most 22 significant bits, exact in f32,
/// so a fused or unfused subtraction rounds alike.
pub(crate) struct Q5kRows;

impl EmbedSb for Q5kRows {
    const BYTES: usize = 176;

    #[inline(always)]
    unsafe fn value(w: &[u32], base: usize, v: usize) -> f32 {
        let wk = base >> 2;
        let j = v >> 6;
        let h = (v >> 5) & 1;
        let l = v & 31;
        let qb = 32 * j + l;
        // SAFETY: every index is wk + 0..=43 (the qh words 4..=11, the qs
        // words 12..=43), inside `w` by this fn's `# Safety`.
        let (w0, w1, w2, w3, hw, qw) = unsafe {
            (
                *w.get_unchecked(wk),
                *w.get_unchecked(wk + 1),
                *w.get_unchecked(wk + 2),
                *w.get_unchecked(wk + 3),
                *w.get_unchecked(wk + 4 + (l >> 2)),
                *w.get_unchecked(wk + 12 + (qb >> 2)),
            )
        };
        let byte = (qw >> (8 * (qb & 3) as u32)) & 0xff;
        let nib = if h == 0 { byte & 0x0f } else { byte >> 4 };
        let hb = (hw >> (8 * (l & 3) as u32)) & 0xff;
        let q = nib + if (hb >> (2 * j + h)) & 1 != 0 { 16 } else { 0 };
        let (sc, mi) = q4k_scale_min(2 * j + h, w1, w2, w3);
        let d = half_to_f32((w0 & 0xffff) as u16);
        let dmin = half_to_f32((w0 >> 16) as u16);
        let d1 = d * sc as f32;
        let m1 = dmin * mi as f32;
        q as f32 * d1 - m1
    }
}

/// Q6_K (210 bytes: 128 `ql` bytes, 64 `qh` bytes, 16 int8 scales, the f16
/// `d`; a super-block may start 2 mod 4, so every field is read by byte):
/// value `128c + 32a + l` takes the nibble `a / 2` of ql byte `64c + 32(a %
/// 2) + l`, bits `2a..2a+1` of qh byte `32c + l` as its high two bits, and
/// scale `8c + l / 16 + 2a`; the value is `(d · sc) · (q − 32)`, two
/// roundings in the reference's order.
pub(crate) struct Q6kRows;

impl EmbedSb for Q6kRows {
    const BYTES: usize = 210;

    #[inline(always)]
    unsafe fn value(w: &[u32], base: usize, v: usize) -> f32 {
        let c = v >> 7;
        let a = (v >> 5) & 3;
        let l = v & 31;
        // SAFETY: every byte is base + 0..=209 < 4 * w.len() by this fn's
        // `# Safety`.
        let (ql, qh, sc, d0, d1) = unsafe {
            (
                stream_byte(w, base + 64 * c + 32 * (a & 1) + l),
                stream_byte(w, base + 128 + 32 * c + l),
                stream_byte(w, base + 192 + 8 * c + (l >> 4) + 2 * a),
                stream_byte(w, base + 208),
                stream_byte(w, base + 209),
            )
        };
        let nib = if a >= 2 { ql >> 4 } else { ql & 0x0f };
        let q = (nib | (((qh >> (2 * a)) & 3) << 4)) as i32;
        let d = half_to_f32((d0 | (d1 << 8)) as u16);
        (d * (sc as u8 as i8) as f32) * (q - 32) as f32
    }
}

/// The body of an embedding entry over super-blocks of type `S`, for thread
/// `i` of the grid: row `t = i / width` of `y` (`width = 256 · n_sb`) takes
/// value `i % width` of table row `ids[t]`, an id past the table's `n_rows`
/// rows raising [`FaultSite::TokenId`] (thread of value 0) and writing NaN;
/// threads `i < ids.len()` also write row `i`'s position `pos0[0] + first +
/// i` and live key count one more through [`position_word`] — the Q4_K
/// entry's work for any `S`.
///
/// # Safety
///
/// The entry's launch contract: `4 · w.len() >= S::BYTES · n_sb · n_rows`,
/// `y.len() >= 256 · n_sb · ids.len()`, `pos0.len() >= 1`, `pos.len()` and
/// `n_keys.len() >= ids.len()`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub(crate) unsafe fn embed_rows_sb<S: EmbedSb>(
    i: usize,
    (w, ids, pos0): (&[u32], &[u32], &[u32]),
    (first, n_rows, n_sb): (u32, u32, u32),
    fault: FaultSink,
    y: &mut DisjointSlice<f32>,
    pos: &mut DisjointSlice<u32>,
    n_keys: &mut DisjointSlice<u32>,
) {
    if i < ids.len() {
        // SAFETY: pos0 holds a word by this fn's `# Safety`.
        let p = unsafe { *pos0.get_unchecked(0) } as usize + first as usize + i;
        // SAFETY: i < ids.len() <= pos.len(), n_keys.len(); thread i owns
        // entry i of both.
        unsafe {
            *pos.get_unchecked_mut(i) = position_word(p);
            *n_keys.get_unchecked_mut(i) = position_word(p + 1);
        }
    }
    let width = 256 * n_sb as usize;
    if i >= ids.len() * width {
        return;
    }
    let t = i / width;
    let k = i % width;
    // SAFETY: t < ids.len() by the guard.
    let id = unsafe { *ids.get_unchecked(t) };
    let v = if id < n_rows {
        // SAFETY: id < n_rows and k >> 8 < n_sb, so the super-block ends at
        // byte <= S::BYTES·n_sb·n_rows <= 4·w.len(); the base is a multiple
        // of S::BYTES, so of 4 for a whole-word type; k & 255 < 256.
        unsafe {
            S::value(
                w,
                (id as usize * n_sb as usize + (k >> 8)) * S::BYTES,
                k & 255,
            )
        }
    } else {
        if k == 0 {
            fault.raise(FaultSite::TokenId);
        }
        f32::NAN
    };
    // SAFETY: i < ids.len() * width <= y.len() by this fn's `# Safety`.
    unsafe {
        *y.get_unchecked_mut(i) = v;
    }
}

/// `v` as a position or live key count word: `v` while it fits a u32, else
/// `u32::MAX` — a value past every cache, which the rope and the flash
/// refuse, where a wrapped one would name a plausible row.
#[inline(always)]
fn position_word(v: usize) -> u32 {
    if v > u32::MAX as usize {
        u32::MAX
    } else {
        v as u32
    }
}

/// The partial sum of squares thread `tid` of an [`RMS_THREADS`] block owns
/// for the row of `k` values at `base`: values `tid, tid + RMS_THREADS, …`
/// ascending, each square added with one fused multiply-add (the device
/// build contracts `acc + v·v`; a host transcription of this order uses
/// `mul_add`). The norm's fixed per-thread order, shared by `rms_norm` and
/// the fused norms. While `RMS_BATCH` strides remain, they are read as one
/// batch — every load issued before the batch is folded, one round trip for
/// the lot — and the rest one stride at a time. Only the loads move; the
/// fold is the same squares in the same order.
///
/// # Safety
///
/// `base + k <= x.len()`: every load is `x[base + it]` with `it < k`, read
/// unchecked. `tid < RMS_THREADS` keeps `it + (RMS_BATCH − 1)·RMS_THREADS`,
/// the batch loop's condition, from overflowing. A kernel caller discharges
/// both from its launch facts (`crate::view`'s module doc): the `requires`
/// clause bounding `x`, a `base` that is 0 or a row its block-uniform row
/// guard has bounded, and a block of `RMS_THREADS` threads (or a `tid <
/// RMS_THREADS` branch).
#[inline(always)]
pub unsafe fn rms_partial_sq(x: &[f32], base: usize, k: usize, tid: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut it = tid;
    while it + (RMS_BATCH - 1) * RMS_THREADS < k {
        // Value `it + j·RMS_THREADS` of the batch.
        macro_rules! load {
            ($j:literal) => {
                // SAFETY: it + j·RMS_THREADS <= it + (RMS_BATCH − 1)·RMS_THREADS
                // < k (the loop condition), and base + k <= x.len() by this
                // fn's `# Safety`.
                unsafe { *x.get_unchecked(base + it + $j * RMS_THREADS) }
            };
        }
        let (v0, v1, v2, v3, v4) = (load!(0), load!(1), load!(2), load!(3), load!(4));
        let (v5, v6, v7, v8, v9) = (load!(5), load!(6), load!(7), load!(8), load!(9));
        let (v10, v11, v12, v13, v14) = (load!(10), load!(11), load!(12), load!(13), load!(14));
        let (v15, v16, v17, v18, v19) = (load!(15), load!(16), load!(17), load!(18), load!(19));
        acc += v0 * v0;
        acc += v1 * v1;
        acc += v2 * v2;
        acc += v3 * v3;
        acc += v4 * v4;
        acc += v5 * v5;
        acc += v6 * v6;
        acc += v7 * v7;
        acc += v8 * v8;
        acc += v9 * v9;
        acc += v10 * v10;
        acc += v11 * v11;
        acc += v12 * v12;
        acc += v13 * v13;
        acc += v14 * v14;
        acc += v15 * v15;
        acc += v16 * v16;
        acc += v17 * v17;
        acc += v18 * v18;
        acc += v19 * v19;
        it += RMS_BATCH * RMS_THREADS;
    }
    while it < k {
        // SAFETY: it < k and base + k <= x.len() by this fn's `# Safety`.
        let v = unsafe { *x.get_unchecked(base + it) };
        acc += v * v;
        it += RMS_THREADS;
    }
    acc
}

/// Strides of a row [`rms_partial_sq`] reads as one batch: a V4.1 hidden
/// row's whole share per thread (5120 values over 256 threads). The batch's
/// values are named one by one (named scalars, not an array indexed in a
/// loop, which is placed in a local depot).
const RMS_BATCH: usize = 20;
// `rms_partial_sq` names the batch's loads one by one.
const _: () = assert!(RMS_BATCH == 20);

/// The row's sum of squares from its [`RMS_WARPS`] warp sums, in the fixed
/// tree `((w0+w1)+(w2+w3)) + ((w4+w5)+(w6+w7))`. This order is the gate: it
/// is what makes the fused norm's scale equal the op path's.
#[inline(always)]
pub fn rms_warp_tree(w: [f32; RMS_WARPS]) -> f32 {
    ((w[0] + w[1]) + (w[2] + w[3])) + ((w[4] + w[5]) + (w[6] + w[7]))
}

/// The norm's scale from the summed squares: the mean divided in the sum's
/// own width, `eps` added inside the sqrt — the reference's form. The
/// reference sums the squares in f64 serially; the device's fixed f32
/// lane/butterfly tree moves last ulps only, which the gate's band owns.
#[inline(always)]
pub fn rms_scale(sum_sq: f32, k: u32, eps: f32) -> f32 {
    let mean = sum_sq / k as f32;
    1.0 / (mean + eps).sqrt()
}

/// Adjacent-pair rotation of one rope pair, the reference's op order (the
/// pairs are (2i, 2i+1), not NeoX split halves): `y0 = x0·cos − x1·sin`,
/// `y1 = x0·sin + x1·cos`. The device build contracts each line into one
/// multiply and one fused multiply-add, so the result is not the host's
/// op-by-op rounding; the gate's band owns that difference. A core that must
/// round every op on its own uses the `_rn` intrinsics instead.
#[inline(always)]
pub(crate) fn rope_pair_core(x0: f32, x1: f32, c: f32, s: f32) -> (f32, f32) {
    (x0 * c - x1 * s, x0 * s + x1 * c)
}

/// `silu(gate) * up` in the reference's scalar op order: `g / (1 + e^(−g))`
/// then one multiply. The device `expf` and the host's differ in the last
/// ulps; the gate bands that distance and prints the measured max.
#[inline(always)]
pub(crate) fn silu_mul(g: f32, u: f32) -> f32 {
    g / (1.0 + (-g).exp()) * u
}

/// Token `t`'s output value `d` of the routed-expert combine:
/// `Σ_e w[t·n_exp + e] · down[(t·n_exp + e)·rows + d]`, experts ascending in
/// the buffer's expert axis, from 0. The device build contracts each
/// `acc + w·d` into one fused multiply-add, so a host transcription is
/// `acc = w.mul_add(d, acc)` per term, not a multiply then an add; the
/// sequence of terms is fixed.
///
/// # Safety
///
/// For some `m`: `down.len() >= rows * n_exp * m`, `w.len() >= n_exp * m`,
/// `t < m` and `d < rows`, which bound every weight index `t·n_exp + e` and
/// down index `(t·n_exp + e)·rows + d` with `e < n_exp`. A kernel caller
/// discharges them from its launch facts (`crate::view`'s module doc): the
/// `requires` clauses bounding `down` and `w`, and `(t, d)` split from its
/// thread's index past its `i < rows·m` guard.
#[inline(always)]
pub(crate) unsafe fn weighted_expert_sum(
    down: &[f32],
    w: &[f32],
    rows: u32,
    n_exp: u32,
    t: usize,
    d: usize,
) -> f32 {
    let rows = rows as usize;
    let n_exp = n_exp as usize;
    let mut acc = 0.0f32;
    let mut e = 0usize;
    while e < n_exp {
        // SAFETY: e < n_exp and t < m bound the weight index t*n_exp + e
        // inside `w` by this fn's `# Safety`.
        let wv = unsafe { *w.get_unchecked(t * n_exp + e) };
        // SAFETY: the same e < n_exp and t < m, with d < rows, bound the down
        // index (t*n_exp + e)*rows + d inside `down` by this fn's `# Safety`.
        let dv = unsafe { *down.get_unchecked((t * n_exp + e) * rows + d) };
        acc += wv * dv;
        e += 1;
    }
    acc
}

/// Whether candidate `(v, i)` beats the running best: strictly greater
/// value, or an equal value at a lower index — the greedy sampler's tie
/// rule. A total order on (value, index), so any fixed reduction tree over
/// it is deterministic.
#[inline(always)]
pub(crate) fn argmax_take(v: f32, i: u32, best_v: f32, best_i: u32) -> bool {
    v > best_v || (v == best_v && i < best_i)
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod elem_kernels {
    use super::*;

    /// Dequantize `ids.len()` rows of the Q3_K embedding table `w` (220 u32
    /// words per row, 2048 values) into `y`, token-major. Ids live on the
    /// device, so their validity cannot be host-checked: an id past the
    /// table's `n_rows` rows raises [`FaultSite::TokenId`] on `fault`, reads
    /// no table word, and writes NaN to every value of its row — no value a
    /// later kernel could take for an embedding; the step's readback turns
    /// the fault into an error.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (4 * w.len() >= 880 * n_rows, y.len() >= 2048 * ids.len())
    )]
    pub fn embed_rows(
        w: &[u32],
        ids: &[u32],
        n_rows: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= ids.len() * 2048 {
            return;
        }
        let t = i >> 11;
        let k = i & 2047;
        // SAFETY: t < ids.len() by the guard.
        let id = unsafe { *ids.get_unchecked(t) };
        let v = if id < n_rows {
            // Row spans are whole 880-byte blocks, so only the super-block
            // offset can sit 2 mod 4; the core funnels both alignments.
            // SAFETY: id < n_rows and k >> 8 < 8, so the super-block ends at
            // byte <= 880·(id + 1) <= 880·n_rows <= 4·w.len() (the launch
            // contract); k & 255 < 256.
            unsafe { q3k_embed_value(w, id as usize * 880 + ((k >> 8) * 110), k & 255) }
        } else {
            if k == 0 {
                fault.raise(FaultSite::TokenId);
            }
            f32::NAN
        };
        // SAFETY: i < ids.len() * 2048 <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = v;
        }
    }

    /// Dequantize `ids.len()` rows of the Q4_K embedding table `w` (`36 ·
    /// n_sb` u32 words per row, `256 · n_sb` values) into `y`, token-major,
    /// one thread per value through [`q4k_embed_value`], and give each row
    /// its position and live key count: row `t` is position
    /// `pos0[0] + first + t` (`first` the ids' offset in the input `pos0`
    /// heads), its count one more, written by thread `t` into `pos[t]` and
    /// `n_keys[t]` through [`position_word`]. Q4_K rows are whole words, so
    /// no funnel is needed.
    /// An id past the table's `n_rows` rows raises [`FaultSite::TokenId`]
    /// and writes a NaN row, as `embed_rows` does; the positions do not
    /// depend on the ids.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= 36 * n_sb * n_rows,
            y.len() >= 256 * n_sb * ids.len(),
            pos0.len() >= 1,
            pos.len() >= ids.len(),
            n_keys.len() >= ids.len()
        )
    )]
    pub fn embed_rows_q4k(
        w: &[u32],
        ids: &[u32],
        pos0: &[u32],
        first: u32,
        n_rows: u32,
        n_sb: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut pos: DisjointSlice<u32>,
        mut n_keys: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        if i < ids.len() {
            // SAFETY: pos0 holds a word by the launch contract.
            let p = unsafe { *pos0.get_unchecked(0) } as usize + first as usize + i;
            // SAFETY: i < ids.len() <= pos.len(), n_keys.len() by the launch
            // contract; thread i owns entry i of both.
            unsafe {
                *pos.get_unchecked_mut(i) = position_word(p);
                *n_keys.get_unchecked_mut(i) = position_word(p + 1);
            }
        }
        let width = 256 * n_sb as usize;
        if i >= ids.len() * width {
            return;
        }
        let t = i / width;
        let k = i % width;
        // SAFETY: t < ids.len() by the guard.
        let id = unsafe { *ids.get_unchecked(t) };
        let v = if id < n_rows {
            // SAFETY: id < n_rows and k >> 8 < n_sb (k < width), so the
            // super-block's 36 words end at <= 36·n_sb·n_rows <= w.len() (the
            // launch contract); k & 255 < 256.
            unsafe { q4k_embed_value(w, (id as usize * n_sb as usize + (k >> 8)) * 36, k & 255) }
        } else {
            if k == 0 {
                fault.raise(FaultSite::TokenId);
            }
            f32::NAN
        };
        // SAFETY: i < ids.len() * width <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = v;
        }
    }

    /// Dequantize `ids.len()` rows of the Q5_K embedding table `w` (`44 ·
    /// n_sb` u32 words per row) into `y` with their positions and live key
    /// counts ([`embed_rows_sb`] over [`Q5kRows`]), as `embed_rows_q4k`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= 176 * n_sb * n_rows,
            y.len() >= 256 * n_sb * ids.len(),
            pos0.len() >= 1,
            pos.len() >= ids.len(),
            n_keys.len() >= ids.len()
        )
    )]
    pub fn embed_rows_q5k(
        w: &[u32],
        ids: &[u32],
        pos0: &[u32],
        first: u32,
        n_rows: u32,
        n_sb: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut pos: DisjointSlice<u32>,
        mut n_keys: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        // SAFETY: the launch contract is the body's (176 = Q5kRows::BYTES,
        // whole words).
        unsafe {
            embed_rows_sb::<Q5kRows>(
                i,
                (w, ids, pos0),
                (first, n_rows, n_sb),
                fault,
                &mut y,
                &mut pos,
                &mut n_keys,
            );
        }
    }

    /// Dequantize `ids.len()` rows of the Q6_K embedding table `w` (`210 ·
    /// n_sb` bytes per row, a whole number of words) into `y` with their
    /// positions and live key counts ([`embed_rows_sb`] over [`Q6kRows`]),
    /// as `embed_rows_q4k`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= 210 * n_sb * n_rows,
            y.len() >= 256 * n_sb * ids.len(),
            pos0.len() >= 1,
            pos.len() >= ids.len(),
            n_keys.len() >= ids.len()
        )
    )]
    pub fn embed_rows_q6k(
        w: &[u32],
        ids: &[u32],
        pos0: &[u32],
        first: u32,
        n_rows: u32,
        n_sb: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut pos: DisjointSlice<u32>,
        mut n_keys: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        // SAFETY: the launch contract is the body's (210 = Q6kRows::BYTES).
        unsafe {
            embed_rows_sb::<Q6kRows>(
                i,
                (w, ids, pos0),
                (first, n_rows, n_sb),
                fault,
                &mut y,
                &mut pos,
                &mut n_keys,
            );
        }
    }

    /// RMS norm, one [`RMS_THREADS`] block per token: `rms_partial_sq` per
    /// thread, the fixed five-step butterfly per warp, the warp sums combined
    /// by `rms_warp_tree`, then `(scale · gain) · x` per value in the
    /// reference's order. `k` a positive multiple of 32 (host-checked); the
    /// token guard is block-uniform, so no barrier and no warp collective is
    /// skipped, and a thread past `k` contributes an exact zero.
    ///
    /// The apply pass runs over [`crate::view`]'s block-strided view: one
    /// `unsafe` at entry carries the whole pass — the launcher evaluated the
    /// contract's `requires` clauses before enqueueing, the shape values are
    /// this kernel's own inputs (one value per grid), and the block is the
    /// contract's 1-D `(RMS_THREADS, 1, 1)`. The same entry facts discharge
    /// `rms_partial_sq`'s `# Safety` (`base + k <= x.len()`). The
    /// warp-slot reduce keeps its raw shared reach: no per-thread type can
    /// prove every slot was written before the barrier that publishes it (a
    /// warp that diverges around the write leaves its slot uninitialized,
    /// and reading an unwritten slot is undefined for any element type), so
    /// that access stays `unsafe` with its `// SAFETY:`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= k * m, gain.len() >= k, y.len() >= k * m)
    )]
    pub fn rms_norm(x: &[f32], gain: &[f32], eps: f32, k: u32, m: u32, mut y: DisjointSlice<f32>) {
        static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let k = k as usize;
        let base = t * k;
        // SAFETY: the contract above was launcher-checked (`requires` on the
        // host before enqueue), `base = t*k` past the `t < m` guard and `k`,
        // `tid`, `RMS_THREADS` are this kernel's own inputs, and the launch's
        // block is the contract's exact 1-D `(RMS_THREADS, 1, 1)`.
        let apply = unsafe { Apply::new(gain, x, &mut y, base, k, tid, RMS_THREADS) };
        // SAFETY: WSUM is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Every access is below RMS_WARPS and ordered by `sync_threads`.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        // SAFETY: base + k = (t + 1)·k <= m·k <= x.len() (the `t < m` guard
        // and the launcher-checked `requires`); tid < RMS_THREADS, the
        // contract's exact block width.
        let part = warp::reduce_sum_f32(unsafe { rms_partial_sq(x, base, k, tid) });
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < RMS_WARPS; one lane per warp writes its slot.
            unsafe {
                *ws.add(tid / 32) = part;
            }
        }
        thread::sync_threads();
        // SAFETY: every slot was written above and is visible past the
        // barrier.
        let sums = unsafe {
            [
                *ws.add(0),
                *ws.add(1),
                *ws.add(2),
                *ws.add(3),
                *ws.add(4),
                *ws.add(5),
                *ws.add(6),
                *ws.add(7),
            ]
        };
        let scale = rms_scale(rms_warp_tree(sums), k as u32, eps);
        apply.map(move |g, v| (scale * g) * v);
    }

    /// Apply the host-computed rope cos/sin cache: one thread per (column,
    /// pair). Column `c = t*n_vec + v` covers `nd` values; token `t`'s cache
    /// is `nd` f32 at `cs[t*nd ..]`, interleaved `[cos0, sin0, …]`. `nd`
    /// even (host-checked).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            src.len() >= m * n_vec * nd,
            cs.len() >= m * nd,
            dst.len() >= m * n_vec * nd
        )
    )]
    pub fn rope(src: &[f32], cs: &[f32], nd: u32, n_vec: u32, m: u32, mut dst: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        let npairs = (nd >> 1) as usize;
        let total = m as usize * n_vec as usize * npairs;
        if i >= total {
            return;
        }
        let nd = nd as usize;
        // SAFETY: the `requires` above were launcher-checked, `nd`, `n_vec`
        // and `npairs` are this kernel's own parameters and `i` this thread's
        // `index_1d` past the `i < total` guard, and the launch is `domain =
        // 1` with the contract's exact 1-D block.
        let pair = unsafe { RopePair::new(src, cs, &mut dst, i, npairs, nd, n_vec as usize) };
        pair.map(rope_pair_core);
    }

    /// The card's slot list of a hybrid layer whose card holds the id
    /// prefix `[0, n_card)` of a stack of `n_expert`: `sel[i] = ids[i]` for
    /// an id below `n_card`, [`HOST`] otherwise — the one value the `_sel`
    /// kernels skip without a fault. An id at or past `n_expert` names no
    /// expert at all: it raises [`FaultSite::ExpertId`] on `fault` and is
    /// written as [`HOST`] too, so no card kernel reads a row for it. One
    /// thread per slot.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (ids.len() >= n, sel.len() >= n)
    )]
    pub fn card_sel(
        ids: &[u32],
        n: u32,
        n_card: u32,
        n_expert: u32,
        fault: FaultSink,
        mut sel: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= ids.len() by the launch contract.
        let id = unsafe { *ids.get_unchecked(i) };
        if id >= n_expert {
            fault.raise(FaultSite::ExpertId);
        }
        // SAFETY: i < n <= sel.len() by the launch contract; thread i alone
        // writes it.
        unsafe {
            *sel.get_unchecked_mut(i) = if id < n_card { id } else { HOST };
        }
    }

    /// `y[i] = silu(gate[i]) * up[i]`, elementwise over `n` values.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (gate.len() >= n, up.len() >= n, y.len() >= n)
    )]
    pub fn swiglu(gate: &[f32], up: &[f32], n: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: the `requires` above were launcher-checked, `n` is this
        // kernel's parameter and `i` this thread's `index_1d` past the `i < n`
        // guard, and the launch is `domain = 1` with the contract's exact 1-D
        // block.
        let e = unsafe { Elem2::new(gate, up, &mut y, i) };
        e.map(silu_mul);
    }

    /// `y[i] = a[i] + b[i]`, elementwise over `n` values — exact; the body
    /// is one add, so nothing is extracted into a core.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (a.len() >= n, b.len() >= n, y.len() >= n)
    )]
    pub fn add(a: &[f32], b: &[f32], n: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: the `requires` above were launcher-checked, `n` is this
        // kernel's parameter and `i` this thread's `index_1d` past the `i < n`
        // guard, and the launch is `domain = 1` with the contract's exact 1-D
        // block.
        let e = unsafe { Elem2::new(a, b, &mut y, i) };
        e.map(|av, bv| av + bv);
    }

    /// The routed-expert combine, one thread per (token, output value):
    /// `y[t*rows + d] = Σ_e w[t*n_exp + e] · down[(t*n_exp + e)*rows + d]`,
    /// token-major `[rows, n_exp, m]` down-projections with `[n_exp, m]`
    /// weights.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            down.len() >= rows * n_exp * m,
            w.len() >= n_exp * m,
            y.len() >= rows * m
        )
    )]
    pub fn weighted_sum(
        down: &[f32],
        w: &[f32],
        rows: u32,
        n_exp: u32,
        m: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= rows as usize * m as usize {
            return;
        }
        let t = i / rows as usize;
        let d = i % rows as usize;
        // SAFETY: i < rows·m (the guard above) gives t < m and d < rows; the
        // launch contract gives down.len() >= rows·n_exp·m and w.len() >=
        // n_exp·m.
        let v = unsafe { weighted_expert_sum(down, w, rows, n_exp, t, d) };
        // SAFETY: i < rows*m <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = v;
        }
    }

    /// Widen `n` f16 bit patterns (one per word of `bits`, the low 16 bits)
    /// to f32 through [`half_to_f32`], the decode every weight scale in this
    /// package goes through. It exists so the gate can hold that decode
    /// against the host's transcription over the whole 16-bit input space —
    /// the only exhaustive statement available about a conversion whose
    /// hardware and software forms need not agree on NaN payloads.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (bits.len() >= n, y.len() >= n)
    )]
    pub fn half_decode(bits: &[u32], n: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: the `requires` above were launcher-checked, `n` is this
        // kernel's parameter and `i` this thread's `index_1d` past the `i < n`
        // guard, and the launch is `domain = 1` with the contract's exact 1-D
        // block.
        let e = unsafe { Elem::new(bits, &mut y, i) };
        e.map(|b| half_to_f32(b as u16));
    }

    /// Every f32 bit pattern `x`, one per thread (the launch is exactly 2^32
    /// threads), through [`f32x2_to_f16x2_bits`] as the pair `(x, !x)` — so
    /// each pattern is tested in both halves — against the device build of
    /// [`f32_to_f16_bits`] of each. A mismatching half counts into `report[0]`
    /// (its input is not a NaN) or `report[1]` (it is), and `report[2]` keeps
    /// the smallest mismatching input by atomic minimum; the host zeroes the
    /// counts and sets `report[2]` to `u32::MAX` first. The sample patterns
    /// ([`HALF_ENCODE_SAMPLES`]) also write `f32_to_f16_bits(x)` to their slot
    /// of `sample`. The gate's instrument, not a step op.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (report.len() >= 3, sample.len() >= 1048576)
    )]
    pub fn half_encode_check(mut report: DisjointSlice<u32>, mut sample: DisjointSlice<u32>) {
        let x = thread::index_1d().get() as u32;
        let nx = !x;
        let got = f32x2_to_f16x2_bits(f32::from_bits(x), f32::from_bits(nx));
        let rep = report.as_mut_ptr();
        let mut h = 0;
        while h < 2 {
            let (input, got16) = if h == 0 {
                (x, got & 0xffff)
            } else {
                (nx, got >> 16)
            };
            let v = f32::from_bits(input);
            if got16 != f32_to_f16_bits(v) as u32 {
                let class = if v.is_nan() { 1 } else { 0 };
                // SAFETY: class < 2 < 3 <= report.len() by the launch
                // contract; these words are only reached atomically while
                // the launch runs.
                unsafe {
                    DeviceAtomicU32::from_ptr(rep.add(class)).fetch_add(1, AtomicOrdering::Relaxed);
                    DeviceAtomicU32::from_ptr(rep.add(2)).fetch_min(input, AtomicOrdering::Relaxed);
                }
            }
            h += 1;
        }
        let slot = x.wrapping_mul(HALF_ENCODE_MUL) as usize;
        if slot < HALF_ENCODE_SAMPLES {
            // SAFETY: slot < HALF_ENCODE_SAMPLES <= sample.len() by the launch
            // contract; the multiplier is odd, so one thread owns each slot.
            unsafe {
                *sample.get_unchecked_mut(slot) = f32_to_f16_bits(f32::from_bits(x)) as u32;
            }
        }
    }

    /// Index of the maximum of `n` f32, ties to the lower index — the greedy
    /// sampler's rule — and the fault next to the token. One
    /// [`ARGMAX_THREADS`] block over the whole vector: each thread's best over
    /// its strided share (indices ascending within a thread), the fixed xor
    /// butterfly merging by (value desc, index asc) inside each warp, then
    /// thread 0 walking the [`ARGMAX_WARPS`] warp slots in ascending order.
    /// Every stage is the same total order ([`argmax_take`]), so the result
    /// is a function of the input alone and the tie rule survives every
    /// regrouping. A thread whose share is empty keeps the `(-inf, 0)`
    /// sentinel and can never win against a real value, which is also what an
    /// all-`-inf` vector answers: index 0, the host reference's answer.
    ///
    /// Thread 0 writes the index to `out[0]`, the first-layer word as it
    /// stands to `out[1]` and that layer's site mask to `out[2]`, so the
    /// head's one readback carries all three. Every launch before this one on
    /// the stream has finished raising, so the copy is the step's whole fault.
    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(domain = 1, block = (1024, 1, 1), requires = (x.len() >= n, out.len() >= 3))]
    pub fn argmax_fault(x: &[f32], n: u32, fault: FaultSink, mut out: DisjointSlice<u32>) {
        static mut BEST_V: SharedArray<f32, ARGMAX_WARPS> = SharedArray::UNINIT;
        static mut BEST_I: SharedArray<u32, ARGMAX_WARPS> = SharedArray::UNINIT;

        // One block: the launcher's grid is exactly 1, so this fires only on a
        // larger grid, whose extra blocks it keeps off `out`. Block-uniform,
        // before any access or barrier.
        if thread::blockIdx_x() != 0 {
            return;
        }

        // SAFETY: both arrays are this block's own shared allocations; the
        // raw form is the only way to reach them without a reference to a
        // `static mut`.
        let (bv, bi) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut BEST_V),
                SharedArray::as_raw_mut_ptr(&raw mut BEST_I),
            )
        };
        // SAFETY: bv and bi are this block's ARGMAX_WARPS-slot shared arrays,
        // and x.len() >= n by the launch contract, so element i*1 + 0 of
        // every i < n is inside x.
        let fi = unsafe { argmax_block(x, n, 1, 0, bv, bi) };
        if thread::threadIdx_x() == 0 {
            let word = fault.read();
            let sites = fault.read_sites(word);
            // SAFETY: out.len() >= 3 by the launch contract; thread 0 of the
            // grid's one block (the guard above) alone writes.
            unsafe {
                *out.get_unchecked_mut(0) = fi;
                *out.get_unchecked_mut(1) = word;
                *out.get_unchecked_mut(2) = sites;
            }
        }
    }

    /// [`argmax_fault`] for a head whose logits no quantizer checked before
    /// them (a Q8_0 lm_head over f32 rows): every thread first walks its
    /// share for a value that is not finite and raises
    /// [`FaultSite::Logit`] on `fault`, then the same walk, butterfly and
    /// tie rule pick the index, and thread 0 copies the word as it stands
    /// past the block barrier, this launch's raise in it. A NaN never wins
    /// the walk, so without the raise a row of NaNs would answer index 0.
    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(domain = 1, block = (1024, 1, 1), requires = (x.len() >= n, out.len() >= 3))]
    pub fn argmax_finite_fault(x: &[f32], n: u32, fault: FaultSink, mut out: DisjointSlice<u32>) {
        static mut BEST_V: SharedArray<f32, ARGMAX_WARPS> = SharedArray::UNINIT;
        static mut BEST_I: SharedArray<u32, ARGMAX_WARPS> = SharedArray::UNINIT;

        // One block: the launcher's grid is exactly 1, so this fires only on a
        // larger grid, whose extra blocks it keeps off `out`. Block-uniform,
        // before any access or barrier.
        if thread::blockIdx_x() != 0 {
            return;
        }

        let mut finite = true;
        let mut i = thread::threadIdx_x();
        while i < n {
            // SAFETY: i < n <= x.len() by the launch contract.
            let v = unsafe { *x.get_unchecked(i as usize) };
            finite &= v.is_finite();
            i += ARGMAX_THREADS as u32;
        }
        if !finite {
            fault.raise(FaultSite::Logit);
        }
        // SAFETY: both arrays are this block's own shared allocations; the
        // raw form is the only way to reach them without a reference to a
        // `static mut`.
        let (bv, bi) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut BEST_V),
                SharedArray::as_raw_mut_ptr(&raw mut BEST_I),
            )
        };
        // SAFETY: bv and bi are this block's ARGMAX_WARPS-slot shared arrays,
        // and x.len() >= n by the launch contract, so element i*1 + 0 of
        // every i < n is inside x.
        let fi = unsafe { argmax_block(x, n, 1, 0, bv, bi) };
        if thread::threadIdx_x() == 0 {
            let word = fault.read();
            let sites = fault.read_sites(word);
            // SAFETY: out.len() >= 3 by the launch contract; thread 0 of the
            // grid's one block (the guard above) alone writes.
            unsafe {
                *out.get_unchecked_mut(0) = fi;
                *out.get_unchecked_mut(1) = word;
                *out.get_unchecked_mut(2) = sites;
            }
        }
    }

    /// [`argmax_fault`]'s walk over each of `m` interleaved rows: block c
    /// takes row c's `n` values `x[i·m + c]` and writes their argmax to
    /// `out[c]`, with the same walk, butterfly, slot order and tie rule — each
    /// row's answer is the argmax of that row alone. The layout is the gemvs'
    /// `y[r·m + c]` output, so a head's m logit rows go straight in. Block 0's
    /// thread 0 also writes the first-layer word to `out[m]` and its layer's
    /// site mask to `out[m + 1]`, as [`argmax_fault`] does.
    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(domain = 1, block = (1024, 1, 1), requires = (x.len() >= n * m, out.len() >= m + 2))]
    pub fn argmax_rows_fault(
        x: &[f32],
        n: u32,
        m: u32,
        fault: FaultSink,
        mut out: DisjointSlice<u32>,
    ) {
        static mut BEST_V: SharedArray<f32, ARGMAX_WARPS> = SharedArray::UNINIT;
        static mut BEST_I: SharedArray<u32, ARGMAX_WARPS> = SharedArray::UNINIT;

        let c = thread::blockIdx_x();
        if c >= m {
            return;
        }
        // SAFETY: both arrays are this block's own shared allocations; the
        // raw form is the only way to reach them without a reference to a
        // `static mut`.
        let (bv, bi) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut BEST_V),
                SharedArray::as_raw_mut_ptr(&raw mut BEST_I),
            )
        };
        // SAFETY: bv and bi are this block's shared arrays, and element
        // i*m + c of every i < n is below n*m <= x.len() because c < m.
        let fi = unsafe { argmax_block(x, n, m, c, bv, bi) };
        if thread::threadIdx_x() == 0 {
            // SAFETY: c < m < out.len(); block c's thread 0 alone writes it.
            unsafe {
                *out.get_unchecked_mut(c as usize) = fi;
            }
            if c == 0 {
                let word = fault.read();
                let sites = fault.read_sites(word);
                // SAFETY: m + 1 < out.len() by the launch contract; block 0's
                // thread 0 alone writes both.
                unsafe {
                    *out.get_unchecked_mut(m as usize) = word;
                    *out.get_unchecked_mut(m as usize + 1) = sites;
                }
            }
        }
    }

    /// [`argmax_rows_fault`] for a head whose logits no quantizer checked
    /// ([`argmax_finite_fault`]'s case at `m` rows): block c first walks row
    /// c for a value that is not finite and its thread 0 raises
    /// [`FaultSite::Logit`], then the same walk, butterfly and tie rule write
    /// the row's argmax to `out[c]`. Each block then draws one ticket from
    /// `done[0]`; the block that draws the `m`-th copies the word and its
    /// site mask to `out[m]` and `out[m + 1]` past every block's raise, and
    /// puts the count back to zero for the next launch or graph replay.
    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(
        domain = 1,
        block = (1024, 1, 1),
        requires = (x.len() >= n * m, out.len() >= m + 2, done.len() >= 1)
    )]
    pub fn argmax_rows_finite_fault(
        x: &[f32],
        n: u32,
        m: u32,
        fault: FaultSink,
        mut out: DisjointSlice<u32>,
        mut done: DisjointSlice<u32>,
    ) {
        static mut BEST_V: SharedArray<f32, ARGMAX_WARPS> = SharedArray::UNINIT;
        static mut BEST_I: SharedArray<u32, ARGMAX_WARPS> = SharedArray::UNINIT;
        static mut BAD: SharedArray<u32, ARGMAX_WARPS> = SharedArray::UNINIT;

        let c = thread::blockIdx_x();
        if c >= m {
            return;
        }
        let tid = thread::threadIdx_x();
        let mut finite = true;
        let mut i = tid;
        while i < n {
            // SAFETY: i < n and c < m, so i*m + c < n*m <= x.len() by the
            // launch contract.
            let v = unsafe { *x.get_unchecked((i * m + c) as usize) };
            finite &= v.is_finite();
            i += ARGMAX_THREADS as u32;
        }
        let bad = warp::ballot(!finite);
        // SAFETY: all three arrays are this block's own shared allocations;
        // the raw form is the only way to reach them without a reference to
        // a `static mut`.
        let (bv, bi, bw) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut BEST_V),
                SharedArray::as_raw_mut_ptr(&raw mut BEST_I),
                SharedArray::as_raw_mut_ptr(&raw mut BAD),
            )
        };
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < ARGMAX_WARPS; one lane per warp writes its
            // own slot, which thread 0 reads past argmax_block's barrier.
            unsafe { *bw.add(tid as usize / 32) = bad };
        }
        // SAFETY: bv and bi are this block's shared arrays, and element
        // i*m + c of every i < n is below n*m <= x.len() because c < m.
        let fi = unsafe { argmax_block(x, n, m, c, bv, bi) };
        if tid != 0 {
            return;
        }
        let mut any = 0u32;
        let mut w = 0usize;
        while w < ARGMAX_WARPS {
            // SAFETY: w < ARGMAX_WARPS, written above, past the barrier.
            any |= unsafe { *bw.add(w) };
            w += 1;
        }
        if any != 0 {
            fault.raise(FaultSite::Logit);
        }
        // SAFETY: c < m < out.len(); block c's thread 0 alone writes it.
        unsafe { *out.get_unchecked_mut(c as usize) = fi };
        // SAFETY: done.len() >= 1 by the launch contract; every access to
        // done[0] on the card is atomic.
        let count = unsafe { DeviceAtomicU32::from_ptr(done.as_mut_ptr()) };
        // The ticket's release orders this block's raise before it; the last
        // block's acquire sees every block's.
        if count.fetch_add(1, AtomicOrdering::AcqRel) + 1 != m {
            return;
        }
        count.store(0, AtomicOrdering::Relaxed);
        let word = fault.read();
        let sites = fault.read_sites(word);
        // SAFETY: m + 1 < out.len() by the launch contract; the last block's
        // thread 0 alone writes both.
        unsafe {
            *out.get_unchecked_mut(m as usize) = word;
            *out.get_unchecked_mut(m as usize + 1) = sites;
        }
    }

    /// The argmax of `x[i·stride + col]` for `i < n`, as [`argmax_fault`]
    /// documents its walk, meaningful on thread 0 of the block; every thread of the
    /// block must call it (it holds a block barrier).
    ///
    /// # Safety
    ///
    /// `bv` and `bi` are the calling block's own shared arrays of
    /// [`ARGMAX_WARPS`] slots, and `(n - 1)·stride + col < x.len()` when `n > 0`.
    #[inline(always)]
    unsafe fn argmax_block(
        x: &[f32],
        n: u32,
        stride: u32,
        col: u32,
        bv: *mut f32,
        bi: *mut u32,
    ) -> u32 {
        let tid = thread::threadIdx_x();
        let mut best_v = f32::NEG_INFINITY;
        let mut best_i = 0u32;
        let mut i = tid;
        while i < n {
            // SAFETY: i < n, so i*stride + col < x.len() by the contract.
            let v = unsafe { *x.get_unchecked((i * stride + col) as usize) };
            if argmax_take(v, i, best_v, best_i) {
                best_v = v;
                best_i = i;
            }
            i += ARGMAX_THREADS as u32;
        }
        let mut off = 16u32;
        while off > 0 {
            let (ov, oi) = (
                warp::shuffle_xor_f32(best_v, off),
                warp::shuffle_xor(best_i, off),
            );
            if argmax_take(ov, oi, best_v, best_i) {
                best_v = ov;
                best_i = oi;
            }
            off >>= 1;
        }
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < ARGMAX_WARPS; one lane per warp writes its
            // own slot, and the pair is written together.
            unsafe {
                *bv.add(tid as usize / 32) = best_v;
                *bi.add(tid as usize / 32) = best_i;
            }
        }
        thread::sync_threads();
        let mut fi = 0u32;
        if tid == 0 {
            // SAFETY: every slot was written above and is visible past the
            // barrier; the walk stays below ARGMAX_WARPS.
            let (mut fv, f0) = unsafe { (*bv.add(0), *bi.add(0)) };
            fi = f0;
            let mut w = 1usize;
            while w < ARGMAX_WARPS {
                // SAFETY: w < ARGMAX_WARPS, written above, past the barrier.
                let (cv, ci) = unsafe { (*bv.add(w), *bi.add(w)) };
                if argmax_take(cv, ci, fv, fi) {
                    fv = cv;
                    fi = ci;
                }
                w += 1;
            }
        }
        fi
    }
}

/// [`ElemKernels::enqueue_embed_rows_q4k`]'s arguments: the Q4_K table, one
/// unit's ids, the word holding the first position of the input the ids are
/// a window of and the window's offset in that input (`first`), and the
/// three outputs — the rows, each row's position and its live key count.
pub struct EmbedRowsArgs<'a> {
    pub w: &'a DeviceTensor<u32>,
    pub ids: &'a DeviceBuffer<u32>,
    pub pos0: &'a DeviceBuffer<u32>,
    pub first: usize,
    pub y: &'a mut DeviceBuffer<f32>,
    pub pos: &'a mut DeviceBuffer<u32>,
    pub n_keys: &'a mut DeviceBuffer<u32>,
}

/// The loaded P4 device module. Owns no context and no stream — the caller
/// passes the engine stream (`Gpu::stream()`) per enqueue, so launches order
/// with the rest of the step and are capturable.
pub struct ElemKernels {
    module: elem_kernels::LoadedModule,
    /// The fault word of the `Gpu` that owns the context: what an embedding
    /// id past the table raises into.
    fault: Arc<DeviceBuffer<u32>>,
}

impl ElemKernels {
    /// Load this file's device bundle into `ctx`, raising into `word`, the
    /// fault word of the `Gpu` that owns `ctx` ([`crate::Gpu::fault_word`]);
    /// a word of another context is refused. Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        word: &Arc<DeviceBuffer<u32>>,
    ) -> Result<ElemKernels, GpuError> {
        let fault = crate::module_fault_word(ctx, word, "ElemKernels::load")?;
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(elem_kernels, ctx)? };
        Ok(ElemKernels { module, fault })
    }

    /// Enqueue the embedding lookup: `ids` (device-resident token ids, every
    /// id below the table's row count) each select one Q3_K row of `w` (220
    /// u32 words per row), dequantized to 2048 f32. `y` holds `2048 *
    /// ids.len()` f32, token-major. Bit-identical to
    /// `gguf::quant::dequant_row` on the same row bytes. An id past the
    /// table raises [`FaultSite::TokenId`] on the owning `Gpu`'s fault word
    /// as an unlabelled launch ([`LAYER_NONE`]) and its row of `y` is NaN.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_embed_rows(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        ids: &DeviceBuffer<u32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if w.cols() != 220 || w.rows() == 0 {
            return Err(GpuError::shape(
                "enqueue_embed_rows",
                format!(
                    "Q3_K table is 220 words (880 bytes) per row, got {}x{}",
                    w.rows(),
                    w.cols()
                ),
            ));
        }
        if ids.is_empty() {
            return Err(GpuError::shape("enqueue_embed_rows", "empty ids"));
        }
        if y.len() < 2048 * ids.len() {
            return Err(GpuError::shape(
                "enqueue_embed_rows",
                format!("y.len() {} < 2048*{}", y.len(), ids.len()),
            ));
        }
        let what = "enqueue_embed_rows";
        let grid = launch_u32(what, "grid", (ids.len() * 2048).div_ceil(256))?;
        let n_rows = launch_u32(what, "w.rows()", w.rows())?;
        let prep = self
            .module
            .prepare_embed_rows(LaunchConfig1D::new(grid, 256, 0))?;
        let fault = crate::sink_over(&self.fault, LAYER_NONE);
        self.module
            .embed_rows(stream, &prep, w.buf(), ids, n_rows, fault, y)?;
        Ok(())
    }

    /// Enqueue the Q4_K embedding lookup of one unit of rows: `args.ids`
    /// (device-resident token ids) each select one row of `args.w` (`36 ·
    /// n_sb` u32 words per row, `256 · n_sb` values, `n_sb = w.cols() / 36`),
    /// dequantized into `args.y` token-major, and row `t`'s position `pos0 +
    /// first + t` and live key count one more go into `args.pos[t]` and
    /// `args.n_keys[t]` ([`EmbedRowsArgs`]). The rows are bit-identical to
    /// `gguf::quant::dequant_row` on the same row bytes. An id past the table
    /// raises [`FaultSite::TokenId`] as [`Self::enqueue_embed_rows`] does.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_embed_rows_q4k(
        &self,
        stream: &CudaStream,
        args: EmbedRowsArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_embed_rows_q4k";
        let EmbedRowsArgs {
            w,
            ids,
            pos0,
            first,
            y,
            pos,
            n_keys,
        } = args;
        if w.rows() == 0 || w.cols() == 0 || !w.cols().is_multiple_of(36) {
            return Err(GpuError::shape(
                what,
                format!(
                    "a Q4_K table is 36 words per super-block per row, got {}x{}",
                    w.rows(),
                    w.cols()
                ),
            ));
        }
        let n_sb = w.cols() / 36;
        if ids.is_empty() {
            return Err(GpuError::shape(what, "empty ids"));
        }
        if y.len() < 256 * n_sb * ids.len() {
            return Err(GpuError::shape(
                what,
                format!("y.len() {} < {}*{}", y.len(), 256 * n_sb, ids.len()),
            ));
        }
        if pos0.is_empty() || pos.len() < ids.len() || n_keys.len() < ids.len() {
            return Err(GpuError::shape(
                what,
                format!(
                    "pos0 holds {} words (want 1), pos {} and n_keys {} (want {} each)",
                    pos0.len(),
                    pos.len(),
                    n_keys.len(),
                    ids.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", (ids.len() * 256 * n_sb).div_ceil(256))?;
        let first = launch_u32(what, "first", first)?;
        let n_rows = launch_u32(what, "w.rows()", w.rows())?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_embed_rows_q4k(LaunchConfig1D::new(grid, 256, 0))?;
        let fault = crate::sink_over(&self.fault, LAYER_NONE);
        self.module.embed_rows_q4k(
            stream,
            &prep,
            w.buf(),
            ids,
            pos0,
            first,
            n_rows,
            n_sb,
            fault,
            y,
            pos,
            n_keys,
        )?;
        Ok(())
    }

    /// Enqueue the embedding lookup of one unit of rows from a K-quant table
    /// of type `ty`, as [`Self::enqueue_embed_rows_q4k`] does for Q4_K: the
    /// rows dequantized token-major into `args.y`, bit-identical to
    /// `gguf::quant::dequant_row`, with each row's position and live key
    /// count. Q4_K runs `embed_rows_q4k`, Q5_K `embed_rows_q5k` (44 words a
    /// super-block), Q6_K `embed_rows_q6k` (210 bytes a super-block, an even
    /// count of them a row so a row is whole words); any other type, or a
    /// table whose rows are not whole super-blocks of it, is refused by name.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_embed_rows_kquant(
        &self,
        stream: &CudaStream,
        ty: gguf::quant::GgmlType,
        args: EmbedRowsArgs<'_>,
    ) -> Result<(), GpuError> {
        use gguf::quant::GgmlType;
        let what = "enqueue_embed_rows_kquant";
        let sb_bytes = match ty {
            GgmlType::Q4_K => return self.enqueue_embed_rows_q4k(stream, args),
            GgmlType::Q5_K => Q5kRows::BYTES,
            GgmlType::Q6_K => Q6kRows::BYTES,
            other => {
                return Err(GpuError::shape(
                    what,
                    format!("a {other} table: the embedding lookups read Q4_K, Q5_K and Q6_K rows"),
                ));
            }
        };
        let EmbedRowsArgs {
            w,
            ids,
            pos0,
            first,
            y,
            pos,
            n_keys,
        } = args;
        let row_bytes = 4 * w.cols();
        if w.rows() == 0 || row_bytes == 0 || !row_bytes.is_multiple_of(sb_bytes) {
            return Err(GpuError::shape(
                what,
                format!(
                    "a {ty} table's rows are whole {sb_bytes}-byte super-blocks in whole words, \
                     got {}x{} words",
                    w.rows(),
                    w.cols()
                ),
            ));
        }
        let n_sb = row_bytes / sb_bytes;
        if ids.is_empty() {
            return Err(GpuError::shape(what, "empty ids"));
        }
        if y.len() < 256 * n_sb * ids.len() {
            return Err(GpuError::shape(
                what,
                format!("y.len() {} < {}*{}", y.len(), 256 * n_sb, ids.len()),
            ));
        }
        if pos0.is_empty() || pos.len() < ids.len() || n_keys.len() < ids.len() {
            return Err(GpuError::shape(
                what,
                format!(
                    "pos0 holds {} words (want 1), pos {} and n_keys {} (want {} each)",
                    pos0.len(),
                    pos.len(),
                    n_keys.len(),
                    ids.len()
                ),
            ));
        }
        let cfg = LaunchConfig1D::new(
            launch_u32(what, "grid", (ids.len() * 256 * n_sb).div_ceil(256))?,
            256,
            0,
        );
        let first = launch_u32(what, "first", first)?;
        let n_rows = launch_u32(what, "w.rows()", w.rows())?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let fault = crate::sink_over(&self.fault, LAYER_NONE);
        if ty == GgmlType::Q5_K {
            let prep = self.module.prepare_embed_rows_q5k(cfg)?;
            self.module.embed_rows_q5k(
                stream,
                &prep,
                w.buf(),
                ids,
                pos0,
                first,
                n_rows,
                n_sb,
                fault,
                y,
                pos,
                n_keys,
            )?;
        } else {
            let prep = self.module.prepare_embed_rows_q6k(cfg)?;
            self.module.embed_rows_q6k(
                stream,
                &prep,
                w.buf(),
                ids,
                pos0,
                first,
                n_rows,
                n_sb,
                fault,
                y,
                pos,
                n_keys,
            )?;
        }
        Ok(())
    }

    /// Enqueue `y = rms_norm(x, gain, eps)` over `m` tokens of `k` values
    /// (token-major). `k` a positive multiple of 32; `x`, `y` hold `k * m`
    /// f32, `gain` holds `k`. Asynchronous, allocation-free, capturable.
    pub fn enqueue_rms_norm(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        gain: &DeviceBuffer<f32>,
        eps: f32,
        k: usize,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if k == 0 || !k.is_multiple_of(32) {
            return Err(GpuError::shape(
                "enqueue_rms_norm",
                format!("k must be a positive multiple of 32, got {k}"),
            ));
        }
        if m == 0 || x.len() < k * m || gain.len() < k || y.len() < k * m {
            return Err(GpuError::shape(
                "enqueue_rms_norm",
                format!(
                    "m={m}, x.len() {} (need {}), gain.len() {} (need {k}), y.len() {} (need {})",
                    x.len(),
                    k * m,
                    gain.len(),
                    y.len(),
                    k * m
                ),
            ));
        }
        let what = "enqueue_rms_norm";
        let k = launch_u32(what, "k", k)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_rms_norm(LaunchConfig1D::new(m, RMS_THREADS_U32, 0))?;
        self.module.rms_norm(stream, &prep, x, gain, eps, k, m, y)?;
        Ok(())
    }

    /// Enqueue the rope rotation of `src` (`m * n_vec * nd` f32, column
    /// `c = t*n_vec + v` covering `nd` values) by the cos/sin caches in `cs`
    /// (`m * nd` f32, token `t`'s interleaved `[cos0, sin0, …]` at
    /// `t*nd` — the host YaRN cache, uploaded per step). `nd` even and at
    /// least 2. `dst` holds `m * n_vec * nd` f32 in `src`'s layout.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_rope(
        &self,
        stream: &CudaStream,
        src: &DeviceBuffer<f32>,
        cs: &DeviceBuffer<f32>,
        n_dims: usize,
        n_vec: u32,
        m: usize,
        dst: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if n_dims < 2 || !n_dims.is_multiple_of(2) {
            return Err(GpuError::shape(
                "enqueue_rope",
                format!("n_dims must be even and >= 2, got {n_dims}"),
            ));
        }
        if n_vec == 0 || m == 0 {
            return Err(GpuError::shape(
                "enqueue_rope",
                format!("need n_vec >= 1 and m >= 1, got n_vec={n_vec} m={m}"),
            ));
        }
        let span = m * n_vec as usize * n_dims;
        if src.len() < span || cs.len() < m * n_dims || dst.len() < span {
            return Err(GpuError::shape(
                "enqueue_rope",
                format!(
                    "src.len() {} / cs.len() {} / dst.len() {} vs span {span}, cache {}",
                    src.len(),
                    cs.len(),
                    dst.len(),
                    m * n_dims
                ),
            ));
        }
        let threads = m * n_vec as usize * (n_dims / 2);
        let what = "enqueue_rope";
        let grid = launch_u32(what, "grid", threads.div_ceil(256))?;
        let n_dims = launch_u32(what, "n_dims", n_dims)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_rope(LaunchConfig1D::new(grid, 256, 0))?;
        self.module
            .rope(stream, &prep, src, cs, n_dims, n_vec, m, dst)?;
        Ok(())
    }

    /// Enqueue the card's slot list (`card_sel`) of the first `n` ids over a
    /// stack of `n_expert` whose card holds `[0, n_card)`: an id below
    /// `n_card` as it is, any other as [`HOST`], and an id at or past
    /// `n_expert` raised on `fault` as [`FaultSite::ExpertId`].
    /// Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher over the kernel's arguments; a fault sink and the stack's size joined them"
    )]
    pub fn enqueue_card_sel(
        &self,
        stream: &CudaStream,
        ids: &DeviceBuffer<u32>,
        n: usize,
        n_card: usize,
        n_expert: usize,
        fault: FaultSink,
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_card_sel";
        if n == 0 || ids.len() < n || sel.len() < n {
            return Err(GpuError::shape(
                what,
                format!("n={n}, ids.len() {}, sel.len() {}", ids.len(), sel.len()),
            ));
        }
        if n_card > n_expert {
            return Err(GpuError::shape(
                what,
                format!("the card's {n_card} experts are more than the stack's {n_expert}"),
            ));
        }
        let n = launch_u32(what, "n", n)?;
        let n_card = launch_u32(what, "n_card", n_card)?;
        let n_expert = launch_u32(what, "n_expert", n_expert)?;
        let prep = self
            .module
            .prepare_card_sel(LaunchConfig1D::new(n.div_ceil(32), 32, 0))?;
        self.module
            .card_sel(stream, &prep, ids, n, n_card, n_expert, fault, sel)?;
        Ok(())
    }

    /// Enqueue `y = silu(gate) * up` over `n` values (flat; token-major
    /// spans of any width). Asynchronous, allocation-free, capturable.
    pub fn enqueue_swiglu(
        &self,
        stream: &CudaStream,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if n == 0 || gate.len() < n || up.len() < n || y.len() < n {
            return Err(GpuError::shape(
                "enqueue_swiglu",
                format!(
                    "n={n}, gate.len() {}, up.len() {}, y.len() {}",
                    gate.len(),
                    up.len(),
                    y.len()
                ),
            ));
        }
        let n = launch_u32("enqueue_swiglu", "n", n)?;
        let prep = self
            .module
            .prepare_swiglu(LaunchConfig1D::new(n.div_ceil(256), 256, 0))?;
        self.module.swiglu(stream, &prep, gate, up, n, y)?;
        Ok(())
    }

    /// Enqueue `y = a + b` over `n` values (flat). Exact. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_add(
        &self,
        stream: &CudaStream,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if n == 0 || a.len() < n || b.len() < n || y.len() < n {
            return Err(GpuError::shape(
                "enqueue_add",
                format!(
                    "n={n}, a.len() {}, b.len() {}, y.len() {}",
                    a.len(),
                    b.len(),
                    y.len()
                ),
            ));
        }
        let n = launch_u32("enqueue_add", "n", n)?;
        let prep = self
            .module
            .prepare_add(LaunchConfig1D::new(n.div_ceil(256), 256, 0))?;
        self.module.add(stream, &prep, a, b, n, y)?;
        Ok(())
    }

    /// Enqueue the routed-expert combine: `down` holds `m` tokens' stacks of
    /// `n_exp` expert outputs of `rows` values (token-major `[rows, n_exp,
    /// m]`), `w` the router weights (`n_exp * m` f32, `[n_exp, m]`); `y`
    /// holds `rows * m` f32, token-major. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_weighted_sum(
        &self,
        stream: &CudaStream,
        down: &DeviceBuffer<f32>,
        w: &DeviceBuffer<f32>,
        rows: usize,
        n_exp: u32,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if rows == 0 || n_exp == 0 || m == 0 {
            return Err(GpuError::shape(
                "enqueue_weighted_sum",
                format!("need rows/n_exp/m >= 1, got {rows}/{n_exp}/{m}"),
            ));
        }
        if down.len() < rows * n_exp as usize * m
            || w.len() < n_exp as usize * m
            || y.len() < rows * m
        {
            return Err(GpuError::shape(
                "enqueue_weighted_sum",
                format!(
                    "down.len() {} (need {}), w.len() {} (need {}), y.len() {} (need {})",
                    down.len(),
                    rows * n_exp as usize * m,
                    w.len(),
                    n_exp as usize * m,
                    y.len(),
                    rows * m
                ),
            ));
        }
        let what = "enqueue_weighted_sum";
        let grid = launch_u32(what, "grid", (rows * m).div_ceil(256))?;
        let rows = launch_u32(what, "rows", rows)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_weighted_sum(LaunchConfig1D::new(grid, 256, 0))?;
        self.module
            .weighted_sum(stream, &prep, down, w, rows, n_exp, m, y)?;
        Ok(())
    }

    /// Enqueue the f16 widening of `bits` (the low 16 bits of each word)
    /// into `y`, through the same `cores::half_to_f32` the gemvs and the
    /// embedding dequant call. The gate's instrument, not a step op.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_half_decode(
        &self,
        stream: &CudaStream,
        bits: &DeviceBuffer<u32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let n = bits.len();
        if n == 0 || y.len() < n {
            return Err(GpuError::shape(
                "enqueue_half_decode",
                format!("n={n}, y.len() {} (need {n})", y.len()),
            ));
        }
        let n = launch_u32("enqueue_half_decode", "n", n)?;
        let prep = self
            .module
            .prepare_half_decode(LaunchConfig1D::new(n.div_ceil(256), 256, 0))?;
        self.module.half_decode(stream, &prep, bits, n, y)?;
        Ok(())
    }

    /// Enqueue the exhaustive check of `flash::f32x2_to_f16x2_bits` against
    /// the device build of `f32_to_f16_bits` over every f32 pattern: the
    /// counts and the smallest mismatching input into `report` (the caller
    /// sets it to `[0, 0, u32::MAX]`), the sample of the reference into
    /// `sample` ([`HALF_ENCODE_SAMPLES`] slots). The gate's instrument, not a
    /// step op. Asynchronous, allocation-free.
    pub fn enqueue_half_encode_check(
        &self,
        stream: &CudaStream,
        report: &mut DeviceBuffer<u32>,
        sample: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        if report.len() < HALF_ENCODE_REPORT || sample.len() < HALF_ENCODE_SAMPLES {
            return Err(GpuError::shape(
                "enqueue_half_encode_check",
                format!(
                    "report.len() {} (need {HALF_ENCODE_REPORT}), sample.len() {} (need \
                     {HALF_ENCODE_SAMPLES})",
                    report.len(),
                    sample.len()
                ),
            ));
        }
        let blocks = (1u64 << 32) / u64::from(HALF_ENCODE_THREADS);
        let blocks = launch_u32("enqueue_half_encode_check", "blocks", blocks as usize)?;
        let prep = self.module.prepare_half_encode_check(LaunchConfig1D::new(
            blocks,
            HALF_ENCODE_THREADS,
            0,
        ))?;
        self.module
            .half_encode_check(stream, &prep, report, sample)?;
        Ok(())
    }

    /// Enqueue the argmax of `n` f32 (ties to the lower index) into `out[0]`
    /// (u32, device-resident) with the first-layer word `fault` addresses
    /// copied to `out[1]` and its layer's site mask to `out[2]` — the head's
    /// readback of token and fault in one copy. One [`ARGMAX_THREADS`] block
    /// walks the whole vector. Asynchronous, allocation-free, capturable.
    pub fn enqueue_argmax_fault(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        n: usize,
        fault: FaultSink,
        out: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        if n == 0 || x.len() < n || out.len() < 3 {
            return Err(GpuError::shape(
                "enqueue_argmax_fault",
                format!(
                    "n={n}, x.len() {}, out.len() {} (need 3)",
                    x.len(),
                    out.len()
                ),
            ));
        }
        let n = launch_u32("enqueue_argmax_fault", "n", n)?;
        let prep =
            self.module
                .prepare_argmax_fault(LaunchConfig1D::new(1, ARGMAX_THREADS_U32, 0))?;
        self.module.argmax_fault(stream, &prep, x, n, fault, out)?;
        Ok(())
    }

    /// [`Self::enqueue_argmax_fault`] that also raises
    /// [`FaultSite::Logit`] on `fault` when a value of the `n` is not finite,
    /// its readback carrying the raise ([`argmax_finite_fault`]): the head of
    /// a projection no quantizer checks the input of. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_argmax_finite_fault(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        n: usize,
        fault: FaultSink,
        out: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "enqueue_argmax_finite_fault";
        if n == 0 || x.len() < n || out.len() < 3 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "n={n}, x.len() {}, out.len() {} (need 3)",
                    x.len(),
                    out.len()
                ),
            ));
        }
        let n = launch_u32(WHAT, "n", n)?;
        let prep = self
            .module
            .prepare_argmax_finite_fault(LaunchConfig1D::new(1, ARGMAX_THREADS_U32, 0))?;
        self.module
            .argmax_finite_fault(stream, &prep, x, n, fault, out)?;
        Ok(())
    }

    /// Enqueue the argmax of each of `m` interleaved rows of `n` f32 — row c
    /// is `x[i*m + c]`, the gemvs' `m`-column output layout — into `out[c]`,
    /// each row's answer the one [`Self::enqueue_argmax_fault`] gives that row
    /// alone, with the first-layer word copied to `out[m]` and its layer's
    /// site mask to `out[m + 1]`. One [`ARGMAX_THREADS`] block per row.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_argmax_rows_fault(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        n: usize,
        m: usize,
        fault: FaultSink,
        out: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_argmax_rows_fault";
        if n == 0 || m == 0 || x.len() < n * m || out.len() < m + 2 {
            return Err(GpuError::shape(
                what,
                format!(
                    "n={n} m={m}, x.len() {}, out.len() {} (need m+2)",
                    x.len(),
                    out.len()
                ),
            ));
        }
        // The walk indexes i*m + c in u32.
        launch_u32(what, "n*m", n * m)?;
        let n = launch_u32(what, "n", n)?;
        let m = launch_u32(what, "m", m)?;
        let prep =
            self.module
                .prepare_argmax_rows_fault(LaunchConfig1D::new(m, ARGMAX_THREADS_U32, 0))?;
        self.module
            .argmax_rows_fault(stream, &prep, x, n, m, fault, out)?;
        Ok(())
    }

    /// [`Self::enqueue_argmax_rows_fault`] that also raises
    /// [`FaultSite::Logit`] on `fault` when a value of any row is not finite,
    /// its readback carrying every row's raise ([`argmax_rows_finite_fault`]):
    /// the head of `m` rows of a projection no quantizer checks the input of.
    /// `done` is the launch's ticket count, one u32 at zero before the launch
    /// and back at zero after it; launches sharing it must be ordered on one
    /// stream. Asynchronous, allocation-free, capturable.
    pub fn enqueue_argmax_rows_finite_fault(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        (n, m): (usize, usize),
        fault: FaultSink,
        out: &mut DeviceBuffer<u32>,
        done: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "enqueue_argmax_rows_finite_fault";
        if n == 0 || m == 0 || x.len() < n * m || out.len() < m + 2 || done.is_empty() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "n={n} m={m}, x.len() {}, out.len() {} (need m+2), done.len() {} (need 1)",
                    x.len(),
                    out.len(),
                    done.len()
                ),
            ));
        }
        // The walk indexes i*m + c in u32.
        launch_u32(WHAT, "n*m", n * m)?;
        let n = launch_u32(WHAT, "n", n)?;
        let m = launch_u32(WHAT, "m", m)?;
        let prep = self
            .module
            .prepare_argmax_rows_finite_fault(LaunchConfig1D::new(m, ARGMAX_THREADS_U32, 0))?;
        self.module
            .argmax_rows_finite_fault(stream, &prep, x, n, m, fault, out, done)?;
        Ok(())
    }
}
