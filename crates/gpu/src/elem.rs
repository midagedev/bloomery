//! P4: every kernel of a decode step that is not a matmul, attention or the
//! router — the embedding row dequant, rms_norm, the gain-scaled RMS norm
//! (`gain_rms_norm`), rope (the YaRN cos/sin cache
//! is computed on the host and uploaded; the kernel applies it), swiglu and
//! the tanh GELU-gated unit (`geglu_tanh`, [`crate::gelu`]'s rule), the
//! residual add and the two-branch gain mix (`mix_rows`), the elementwise
//! product of two vectors (`mul_rows`), the in-place scalar `scale_rows`, the
//! per-head SSMax query scale (`ssmax_rows`), the routed-expert weighted sum
//! and argmax, and the head's
//! logit softcap (`softcap_rows`). One
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
//! u32 words each), Q3_K, Q4_K, Q5_K or Q6_K rows of whole super-blocks, or
//! Q5_0 rows in the `pack_q5_0` layout.
//! Extents are launch arguments, never buffer lengths: scratch buffers may
//! be larger than the shape in flight.

use crate::GpuError;
use crate::cores::{funnel16, half_to_f32, q3k_aux_scales, q3k_sub_scale, q4k_scale_min};
use crate::fault::{FaultSink, FaultSite, LAYER_NONE};
use crate::flash::{f32_to_f16_bits, f32x2_to_f16x2_bits};
use crate::gelu::gelu_tanh_dev;
use crate::hybrid::HOST;
use crate::launch_u32;
use crate::tensor::DeviceTensor;
use crate::view::{Apply, Elem, Elem2, RopePair};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
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

/// The f32 value of IQ4_NL weight `v` (0..32) of the 18-byte block at byte
/// `base` of the row stream `w`, bit-identical to `gguf::quant::dequant_row`:
/// `d · kvalues_iq4nl[code]`, `d` the f16 in the block's first two bytes and
/// `code` the low nibble of qs byte `v` for `v < 16`, the high nibble of qs
/// byte `v − 16` above. The product of an f16 and an int8 is exact in f32. The
/// code word is decoded by the shared nibble decode ([`crate::iq::iq4_word`]),
/// and the value's lane picked from it. A block starts 0 or 2 mod 4 (18 bytes
/// in a row of whole words), so `d` is one half of a word and the code byte is
/// loaded alone from its covering word.
///
/// # Safety
///
/// `base` even, `base + 18 <= 4 * w.len()` and `v < 32`: the code byte is at
/// most `base + 17` and `d` sits in the word of byte `base`.
#[inline(always)]
pub unsafe fn iq4nl_embed_value(w: &[u32], base: usize, v: usize) -> f32 {
    let x = base + 2 + (v & 15);
    // SAFETY: x <= base + 17 < 4 * w.len() and base < 4 * w.len() by this fn's
    // `# Safety`, so both covering words are inside w.
    let (cw, dw) = unsafe { (*w.get_unchecked(x >> 2), *w.get_unchecked(base >> 2)) };
    let (lo, hi) = crate::iq::iq4_word(cw);
    let kv = ((if v < 16 { lo } else { hi }) >> (8 * (x & 3))) as u8 as i8;
    let d = half_to_f32((dw >> (16 * ((base >> 1) & 1))) as u16);
    d * f32::from(kv)
}

/// A K-quant super-block as an embedding row reads it: its bytes in the
/// file, and the f32 value of weight `v` (0..256) of the super-block at byte
/// `base` of the row stream `w`, bit-identical to `gguf::quant::dequant_row`.
pub trait EmbedSb {
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
pub struct Q5kRows;

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
pub struct Q6kRows;

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

/// Q3_K (110 bytes: 32 `hmask` bytes, 64 `qs` bytes, 12 scale bytes, the f16
/// `d`; a super-block starts 0 or 2 mod 4 in a row of an even count of them,
/// so [`q3k_embed_value`] funnels both): value `128c + 32a + 16h + l` takes
/// the 2-bit code at field `2a` of qs byte `32c + 16h + l`, the high bit
/// `hmask[16h + l]` bit `4c + a` (clear subtracts 4), and sub-block scale
/// `8c + 2a + h` of the aux shuffle minus 32; the value is `(d · (sc − 32)) ·
/// (q − hv)`, two roundings in the reference's order.
pub struct Q3kRows;

impl EmbedSb for Q3kRows {
    const BYTES: usize = 110;

    #[inline(always)]
    unsafe fn value(w: &[u32], base: usize, v: usize) -> f32 {
        // SAFETY: `base + 110 <= 4 * w.len()` and `v < 256` by this fn's
        // `# Safety`, the core's.
        unsafe { q3k_embed_value(w, base, v) }
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
///
/// A scale that is not finite and positive has no norm behind it — a sum of
/// squares that overflowed (finite values whose squares sum past f32's
/// range) or was never finite gives 0 or NaN — and comes back NaN: every
/// value it scales is NaN, never the finite zero `1/∞` would write.
#[inline(always)]
pub fn rms_scale(sum_sq: f32, k: u32, eps: f32) -> f32 {
    rms_scale_of_mean(sum_sq / k as f32, eps)
}

/// [`rms_scale`] from a mean square the caller rounded itself: the norms that
/// sum their squares in f64 and round `(sum / n) as f32` once (the head norms
/// of `rope_neox`, `qsa`). The one owner of the refusal: `1 / sqrt(mean +
/// eps)` where it is finite and positive, NaN everywhere else.
#[inline(always)]
pub(crate) fn rms_scale_of_mean(mean: f32, eps: f32) -> f32 {
    let scale = 1.0 / (mean + eps).sqrt();
    if (scale > 0.0) & (scale < f32::INFINITY) {
        scale
    } else {
        f32::NAN
    }
}

/// [`rms_scale`] for a kernel with a fault sink: a refused scale (NaN) raises
/// [`FaultSite::F32Product`] from the lane `lead` names — one lane of the
/// block, so one atomic rather than one a thread — and is returned, so the
/// values it scales are NaN. `rms_norm` and `kv_norm_rope_append`'s latent
/// norm both take their scale here, so a row raises the same site on either
/// path.
#[inline(always)]
pub fn rms_scale_checked(sum_sq: f32, k: u32, eps: f32, fault: FaultSink, lead: bool) -> f32 {
    rms_scale_of_mean_checked(sum_sq / k as f32, eps, fault, lead)
}

/// [`rms_scale_of_mean`] for a kernel with a fault sink, as
/// [`rms_scale_checked`] is to [`rms_scale`].
#[inline(always)]
pub(crate) fn rms_scale_of_mean_checked(mean: f32, eps: f32, fault: FaultSink, lead: bool) -> f32 {
    let scale = rms_scale_of_mean(mean, eps);
    if lead & scale.is_nan() {
        fault.raise(FaultSite::F32Product);
    }
    scale
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

/// The position basis of SSMax: `ln(pos + 1)`, as the reference's graph forms
/// it. The input row holds `ln(pos + 1) + 1` — the logarithm in f64 of the
/// position (held as an f32) plus one, rounded once to f32 — and the graph
/// subtracts the one in f32, which is exact. The reference's log scale of
/// 1.0 multiplies the logarithm before the sum and changes no bit.
#[inline(always)]
pub fn ssmax_basis(pos: u32) -> f32 {
    let at = f64::from(pos as f32) + 1.0;
    (at.ln() + 1.0) as f32 - 1.0
}

/// One thread's f64 partial of the mean of `x[base .. base + k]`: the sum of
/// its strided values, each widened before the sum.
///
/// # Safety
/// `base + k <= x.len()` and `tid < RMS_THREADS`.
#[inline(always)]
unsafe fn ln_mean(x: &[f32], base: usize, k: usize, tid: usize) -> f64 {
    let mut acc = 0.0f64;
    let mut i = tid;
    while i < k {
        // SAFETY: i < k, so base + i < base + k <= x.len() (this fn's
        // contract).
        acc += f64::from(unsafe { *x.get_unchecked(base + i) });
        i += RMS_THREADS;
    }
    acc
}

/// One thread's f64 partial of the variance of `x[base .. base + k]` around
/// `mean`: the sum of the f32-rounded squares of its strided f32 deviations.
///
/// # Safety
/// `base + k <= x.len()` and `tid < RMS_THREADS`.
#[inline(always)]
unsafe fn ln_var(x: &[f32], base: usize, k: usize, tid: usize, mean: f32) -> f64 {
    let mut acc = 0.0f64;
    let mut i = tid;
    while i < k {
        // SAFETY: i < k, so base + i < base + k <= x.len() (this fn's
        // contract).
        let dev = unsafe { *x.get_unchecked(base + i) } - mean;
        acc += f64::from(mul_rn_f32(dev, dev));
        i += RMS_THREADS;
    }
    acc
}

/// The block's f64 sum of the warps' partials `v` (each lane's widened by the
/// xor butterfly first): thread `tid` holds its warp's sum at lane 0, the
/// first warp reduces the per-warp sums through the caller's shared `ws`
/// ([`RMS_WARPS`] slots), and every thread reads the result — the value all
/// threads need before the apply pass. The caller's barriers order the slots
/// against both reads.
///
/// # Safety
/// `tid < RMS_THREADS` and `ws` points to this block's own [`RMS_WARPS`]
/// shared slots that no concurrent call shares; all threads of the block call
/// it converged.
#[inline(always)]
unsafe fn ln_block_tree(v: f64, tid: usize, ws: *mut f64) -> f64 {
    let mut acc = v;
    acc += warp::shuffle_xor_f64(acc, 16);
    acc += warp::shuffle_xor_f64(acc, 8);
    acc += warp::shuffle_xor_f64(acc, 4);
    acc += warp::shuffle_xor_f64(acc, 2);
    acc += warp::shuffle_xor_f64(acc, 1);
    // SAFETY: the caller's contract: the block's own slots, the whole block
    // converged. Lane 0 of each warp writes its slot before the barrier, warp
    // 0 alone reduces the slots, and the whole block reads slot 0 after the
    // second barrier.
    unsafe {
        let (lane, wid) = (warp::lane_id() as usize, tid / 32);
        if lane == 0 {
            *ws.add(wid) = acc;
        }
        thread::sync_threads();
        if wid == 0 {
            let mut all = 0.0f64;
            let mut w = 0usize;
            while w < RMS_WARPS {
                all += *ws.add(w);
                w += 1;
            }
            if lane == 0 {
                *ws.add(0) = all;
            }
        }
        thread::sync_threads();
        *ws.add(0)
    }
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

    /// Dequantize `ids.len()` rows of a Q5_0 table in the `pack_q5_0` layout
    /// (`q5::q5_0_row_words(k_blocks)` u32 words a row: block `b`'s nibble
    /// byte `j < 16` is byte `j % 4` of word `(j/4)·k_blocks + b`, its high
    /// bits `qh` word `4·k_blocks + b`, its f16 scale half `b & 1` of word
    /// `5·k_blocks + b/2`) into `y`, with each row's position and live key
    /// count as `embed_rows_q4k`. Value `v` of block `b = v/32` with `t =
    /// v % 32` is the low (`t < 16`) or high nibble of byte `t % 16`, with
    /// `qh` bit `t` as its fifth bit, minus 16, times the scale — ggml's
    /// `dequantize_row_q5_0`. Q5_0's rows are 32-value blocks split into
    /// planes, not 256-value super-blocks, so this does not share
    /// [`embed_rows_sb`]. An id past `n_rows` raises [`FaultSite::TokenId`]
    /// and writes a NaN row, as `embed_rows` does. Rows are `5·k_blocks +
    /// s_words` words apart: the launcher passes `s_words = ⌈k_blocks/2⌉`.
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
            2 * s_words >= k_blocks,
            w.len() >= (5 * k_blocks + s_words) * n_rows,
            y.len() >= k * ids.len(),
            pos0.len() >= 1,
            pos.len() >= ids.len(),
            n_keys.len() >= ids.len()
        )
    )]
    pub fn embed_rows_q5_0(
        w: &[u32],
        ids: &[u32],
        pos0: &[u32],
        first: u32,
        n_rows: u32,
        k: u32,
        s_words: u32,
        k_blocks: u32,
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
        let width = k as usize;
        if i >= ids.len() * width {
            return;
        }
        let t = i / width;
        let v = i % width;
        // SAFETY: t < ids.len() by the guard.
        let id = unsafe { *ids.get_unchecked(t) };
        let out = if id < n_rows {
            let kb = k_blocks as usize;
            let b = v / 32;
            let within = v % 32;
            let j = within % 16;
            let row = id as usize * (5 * kb + s_words as usize);
            // SAFETY: id < n_rows and b < k/32 = k_blocks (v < k and the
            // launcher passes k = 32·k_blocks), so the qs word (j/4 < 4), the
            // qh word and the scale word 5·kb + b/2 < 5·kb + s_words
            // (2·s_words >= kb) lie inside the row, which ends at
            // (id+1)·(5·kb + s_words) <= w.len() by the launch contract.
            let (qs, qh, dw) = unsafe {
                (
                    *w.get_unchecked(row + (j / 4) * kb + b),
                    *w.get_unchecked(row + 4 * kb + b),
                    *w.get_unchecked(row + 5 * kb + b / 2),
                )
            };
            let byte = (qs >> (8 * (j % 4))) & 0xff;
            let nib = if within < 16 { byte & 0x0f } else { byte >> 4 };
            let code = (nib | (((qh >> within) & 1) << 4)) as i32 - 16;
            let scale = half_to_f32((dw >> (16 * (b & 1))) as u16);
            (code as f32) * scale
        } else {
            if v == 0 {
                fault.raise(FaultSite::TokenId);
            }
            f32::NAN
        };
        // SAFETY: i < ids.len() * k <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = out;
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

    /// Dequantize `ids.len()` rows of the Q3_K embedding table `w` (`110 ·
    /// n_sb` bytes per row, an even count of super-blocks so a row is whole
    /// words) into `y` with their positions and live key counts
    /// ([`embed_rows_sb`] over [`Q3kRows`]), as `embed_rows_q4k`.
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
            4 * w.len() >= 110 * n_sb * n_rows,
            y.len() >= 256 * n_sb * ids.len(),
            pos0.len() >= 1,
            pos.len() >= ids.len(),
            n_keys.len() >= ids.len()
        )
    )]
    pub fn embed_rows_q3k(
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
        // SAFETY: the launch contract is the body's (110 = Q3kRows::BYTES).
        unsafe {
            embed_rows_sb::<Q3kRows>(
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
    pub fn rms_norm(
        x: &[f32],
        gain: &[f32],
        eps: f32,
        k: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
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
        let scale = rms_scale_checked(rms_warp_tree(sums), k as u32, eps, fault, tid == 0);
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

    /// `y[i] = gelu_tanh(gate[i]) · up[i]`, elementwise over `n` values: the
    /// rule of [`crate::gelu`] on the gate value, then one rounded multiply by
    /// the up value (llama.cpp's `unary_gated_op_kernel` at `op_gelu`).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (gate.len() >= n, up.len() >= n, y.len() >= n)
    )]
    pub fn geglu_tanh(gate: &[f32], up: &[f32], n: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: the `requires` above were launcher-checked, `n` is this
        // kernel's parameter and `i` this thread's `index_1d` past the `i < n`
        // guard, and the launch is `domain = 1` with the contract's exact 1-D
        // block.
        let e = unsafe { Elem2::new(gate, up, &mut y, i) };
        e.map(|g, u| mul_rn_f32(gelu_tanh_dev(g), u));
    }

    /// `y[i] = situ(gate[i], up[i])`, elementwise over `n` values: the
    /// gate·up rule [`crate::kquant::act::situ`] at `beta` and
    /// [`crate::kquant::act::SITU_LINEAR_BETA`], the rows the `_sel`
    /// gate·up writes under it for the same sums.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (gate.len() >= n, up.len() >= n, y.len() >= n)
    )]
    pub fn situ_rows(gate: &[f32], up: &[f32], n: u32, beta: f32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: the `requires` above were launcher-checked, `n` is this
        // kernel's parameter and `i` this thread's `index_1d` past the `i < n`
        // guard, and the launch is `domain = 1` with the contract's exact 1-D
        // block.
        let e = unsafe { Elem2::new(gate, up, &mut y, i) };
        e.map(|g, u| crate::kquant::act::situ(g, u, beta, crate::kquant::act::SITU_LINEAR_BETA));
    }

    /// `y[i] = rule(gate[i], up[i])`, elementwise over `n` values, under a
    /// gate·up rule of the K-quant family (`kquant::act::apply_checked`):
    /// ik's clamped SwiGLU at `limit` (`ACT_SWIGLU_CLAMP`), the OAI SwiGLU at
    /// `limit` (`ACT_SWIGLU_OAI`), SiTU at beta `limit` (`ACT_SITU`) or
    /// `silu_mul` (`ACT_SILU_MUL`); the contract admits those four codes
    /// alone. The f32 rows an f32-form down reads, each value the one the
    /// rule's quantizing entries quantize. A non-finite gate or up value
    /// writes NaN and raises [`FaultSite::F32Product`]: a clamp would carry it
    /// to a finite value.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (act <= 3, gate.len() >= n, up.len() >= n, y.len() >= n)
    )]
    pub fn swiglu_rule_rows(
        gate: &[f32],
        up: &[f32],
        n: u32,
        act: u32,
        limit: f32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: the `requires` above were launcher-checked, `n` is this
        // kernel's parameter and `i` this thread's `index_1d` past the `i < n`
        // guard, and the launch is `domain = 1` with the contract's exact 1-D
        // block.
        let e = unsafe { Elem2::new(gate, up, &mut y, i) };
        e.map(|g, u| crate::kquant::act::apply_checked(act, limit, g, u, fault));
    }

    /// `y[i] = max(x[i], 0)^2` over `n` values. A non-finite input is copied
    /// as it is: `max` of a non-finite is not this rule, and the square of
    /// `−inf` is not 0.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= n, y.len() >= n)
    )]
    pub fn relu_sqr(x: &[f32], n: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= x.len() and y.len() by the launch contract; thread
        // i owns entry i.
        let v = unsafe { *x.get_unchecked(i) };
        let out = if !v.is_finite() {
            v
        } else {
            let r = if v > 0.0 { v } else { 0.0 };
            r * r
        };
        // SAFETY: i < n <= y.len() by the launch contract; thread i owns
        // entry i.
        unsafe {
            *y.get_unchecked_mut(i) = out;
        }
    }

    /// `y[i] = sigmoid(y[i])` over `n` values, in place — the router score
    /// ([`crate::route_core::router_score`] of
    /// [`Sigmoid`](crate::route_core::Sigmoid)) over a tile GEMM's logits: a
    /// logit that is not finite becomes NaN.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (y.len() >= n)
    )]
    pub fn sigmoid_rows(mut y: DisjointSlice<f32>, n: u32) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= y.len() by the launch contract; thread i owns
        // entry i.
        unsafe {
            let v = *y.get_unchecked_mut(i);
            *y.get_unchecked_mut(i) =
                crate::route_core::router_score::<crate::route_core::Sigmoid>(v);
        }
    }

    /// `y[i] = sqrt_softplus(y[i])` over `n` values, in place — the router
    /// score ([`crate::route_core::router_score`] of
    /// [`SqrtSoftplus`](crate::route_core::SqrtSoftplus)) over a tile GEMM's
    /// logits: a logit that is not finite becomes NaN.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (y.len() >= n)
    )]
    pub fn softplus_rows(mut y: DisjointSlice<f32>, n: u32) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= y.len() by the launch contract; thread i owns
        // entry i.
        unsafe {
            let v = *y.get_unchecked_mut(i);
            *y.get_unchecked_mut(i) =
                crate::route_core::router_score::<crate::route_core::SqrtSoftplus>(v);
        }
    }

    /// `y[i] = cap·tanh(y[i]/cap)` over `n` values, in place — the head's
    /// logit softcap epilogue (`x/cap → tanh → ·cap`). A NaN logit stays NaN
    /// and ±inf maps to ±cap, tanh's own limits: the rule's values, not a
    /// clamp, and no fault (the reference computes the same).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (y.len() >= n)
    )]
    pub fn softcap_rows(mut y: DisjointSlice<f32>, n: u32, cap: f32) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= y.len() by the launch contract; thread i owns
        // entry i.
        unsafe {
            let v = *y.get_unchecked_mut(i);
            *y.get_unchecked_mut(i) = mul_rn_f32((v / cap).tanh(), cap);
        }
    }

    /// `y[i] = a[i] · b[i]`, elementwise over `n` values — one rounded
    /// multiply, the product of two vectors (a gate's rows and the values it
    /// gates). Exact in the one op; nothing is extracted into a core.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (a.len() >= n, b.len() >= n, y.len() >= n)
    )]
    pub fn mul_rows(a: &[f32], b: &[f32], n: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= a.len(), b.len() and y.len() by the launch
        // contract; thread i owns entry i of each.
        let e = unsafe { Elem2::new(a, b, &mut y, i) };
        e.map(mul_rn_f32);
    }

    /// `y[i] = a[i] · ga[i] + b[i] · gb[i]`, elementwise over `n` values —
    /// the residual mix of two branches, each scaled by its own gain vector:
    /// three roundings (two products, one sum), as the graph's two MUL nodes
    /// and one ADD node produce.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            a.len() >= n,
            ga.len() >= n,
            b.len() >= n,
            gb.len() >= n,
            y.len() >= n
        )
    )]
    pub fn mix_rows(
        a: &[f32],
        ga: &[f32],
        b: &[f32],
        gb: &[f32],
        n: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= every input's and y's length by the launch
        // contract; thread i owns entry i of each.
        unsafe {
            let (av, gav) = (*a.get_unchecked(i), *ga.get_unchecked(i));
            let (bv, gbv) = (*b.get_unchecked(i), *gb.get_unchecked(i));
            *y.get_unchecked_mut(i) = add_rn_f32(mul_rn_f32(av, gav), mul_rn_f32(bv, gbv));
        }
    }

    /// `y[i] = s · y[i]` over the first `n` values, in place — a scalar on
    /// the head's logits before their softcap. `s` finite and nonzero: any
    /// other is refused by name on the host, for it would read every value as
    /// ±inf or NaN.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (y.len() >= n))]
    pub fn scale_rows(mut y: DisjointSlice<f32>, n: u32, s: f32) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= y.len() by the launch contract; thread i owns
        // entry i.
        unsafe {
            let v = *y.get_unchecked_mut(i);
            *y.get_unchecked_mut(i) = mul_rn_f32(v, s);
        }
    }

    /// SSMax over `m` tokens of `h` heads of `d` values, in place: `x[t][head][i]
    /// *= f`, the factor `f = (basis·s[head]) + b[head]` of token `t`'s
    /// [`ssmax_basis`] at `pos[t]`, the product and the sum each rounded on
    /// their own as the graph's MUL and ADD nodes round them, then one rounded
    /// multiply of the value by `f`. The tokens' rows are contiguous
    /// (`[m][h·d]`); a row takes `ceil(h·d / 256)` blocks, and block `b`
    /// serves token `b / ceil(h·d / 256)`: its first thread forms the basis
    /// once and the block reads it past the barrier. A value or a weight that
    /// is not finite makes the values it scales not finite, the rule's own
    /// result (nothing is clamped, and no fault is raised).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= m * h * d, pos.len() >= m, s.len() >= h, b.len() >= h)
    )]
    pub fn ssmax_rows(
        mut x: DisjointSlice<f32>,
        pos: &[u32],
        s: &[f32],
        b: &[f32],
        d: u32,
        h: u32,
        m: u32,
    ) {
        static mut BASIS: SharedArray<f32, 1> = SharedArray::UNINIT;

        let row = h as usize * d as usize;
        let per_row = row.div_ceil(256);
        let blk = thread::blockIdx_x() as usize;
        let t = blk / per_row;
        // The guard reads only the block's index, so every thread of the
        // block leaves before the barrier together.
        if t >= m as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        // SAFETY: BASIS is this block's own shared allocation; the raw form
        // is the only way to reach it without a reference to a `static mut`.
        // Thread 0 alone writes it, before the barrier.
        let basis_at = unsafe { SharedArray::as_raw_mut_ptr(&raw mut BASIS) };
        if tid == 0 {
            // SAFETY: t < m <= pos.len() by the launch contract; slot 0 of the
            // one-slot array.
            unsafe {
                *basis_at = ssmax_basis(*pos.get_unchecked(t));
            }
        }
        thread::sync_threads();
        let i = (blk - t * per_row) * 256 + tid;
        if i >= row {
            return;
        }
        // SAFETY: the slot was written by thread 0 before the barrier.
        let basis = unsafe { *basis_at };
        let head = i / d as usize;
        // SAFETY: head < h <= s.len(), b.len() (i < h·d), and t·h·d + i <
        // (t + 1)·h·d <= m·h·d <= x.len() by the launch contract; thread i of
        // token t owns entry i of its row.
        unsafe {
            let f = add_rn_f32(
                mul_rn_f32(basis, *s.get_unchecked(head)),
                *b.get_unchecked(head),
            );
            let v = x.get_unchecked_mut(t * row + i);
            *v = mul_rn_f32(*v, f);
        }
    }

    /// The LayerNorm at `eps` of each row of `k` values of `x`, the row then
    /// scaled by the gain `gain` and shifted by the bias `bias`
    /// (`ggml_add(ggml_mul(ggml_norm(x, eps), gain), bias)`, one block a
    /// row): the mean and the variance each summed in f64 over the row's
    /// plain values and its f32-rounded deviations ([`ln_mean`] and
    /// [`ln_var`], the lightning index key's own order), the scale
    /// `1/√(var + eps)`, then `(dev·scale)·gain + bias` a value with the
    /// product and the sum each rounded on their own. A variance that is not
    /// finite leaves the scale NaN (every value NaN) and raises
    /// [`FaultSite::F32Product`] from lane 0.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= k * m, gain.len() >= k, bias.len() >= k, y.len() >= k * m)
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn ln_rows(
        x: &[f32],
        gain: &[f32],
        bias: &[f32],
        eps: f32,
        k: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        static mut WSUM: SharedArray<f64, RMS_WARPS> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return; // block-uniform: no barrier is skipped
        }
        let tid = thread::threadIdx_x() as usize;
        let k = k as usize;
        let base = t * k;
        // SAFETY: WSUM is this block's own shared allocation.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        // SAFETY: base + k <= m·k <= x.len() (the launch contract) and
        // tid < 256 = RMS_THREADS; `ws` is this block's own slots and the
        // whole block reaches the call converged.
        let mean = (unsafe { ln_block_tree(ln_mean(x, base, k, tid), tid, ws) } / k as f64) as f32;
        // SAFETY: as for the mean: the same row, slots and converged block.
        let var =
            (unsafe { ln_block_tree(ln_var(x, base, k, tid, mean), tid, ws) } / k as f64) as f32;
        let scale = if var.is_finite() {
            1.0f32 / (var + eps).sqrt()
        } else {
            f32::NAN
        };
        if tid == 0 && !var.is_finite() {
            fault.raise(FaultSite::F32Product);
        }
        let mut i = tid;
        while i < k {
            // SAFETY: i < k <= gain.len(), bias.len() by the launch contract,
            // and base + i < (t + 1)·k <= m·k <= x.len(), y.len() (the same
            // contract); thread i owns entry i of its row.
            unsafe {
                let dev = *x.get_unchecked(base + i) - mean;
                let v = add_rn_f32(
                    mul_rn_f32(mul_rn_f32(dev, scale), *gain.get_unchecked(i)),
                    *bias.get_unchecked(i),
                );
                *y.get_unchecked_mut(base + i) = v;
            }
            i += RMS_THREADS;
        }
    }

    /// The RMS norm at `eps` of each row of `k` values of `x` after the row
    /// is scaled by the gain `gain` (`ggml_rms_norm(ggml_mul(x, gain),
    /// eps)`, one [`RMS_THREADS`] block a row): each value's product
    /// `gv = gain·x` rounded once ([`mul_rn_f32`], the separate MUL node's
    /// bit), the squares of those products summed in [`rms_partial_sq`]'s
    /// order, the scale from [`rms_scale_checked`], then `scale · gv` a value
    /// — the product recomputed, the same two roundings the graph's MUL and
    /// NORM nodes make. A scale that is not finite and positive raises
    /// [`FaultSite::F32Product`] from lane 0 and comes back NaN.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= k * m, gain.len() >= k, y.len() >= k * m)
    )]
    pub fn gain_rms_norm(
        x: &[f32],
        gain: &[f32],
        eps: f32,
        k: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let k = k as usize;
        let base = t * k;
        // The row's gained values, one product a value in the caller's fixed
        // thread order, squared and summed as `rms_partial_sq` sums a plain
        // row's.
        let mut acc = 0.0f32;
        let mut it = tid;
        while it < k {
            // SAFETY: base + it < base + k = (t + 1)·k <= m·k <= x.len() (the
            // `t < m` guard and the launch contract); it < k <= gain.len().
            let gv = unsafe { mul_rn_f32(*gain.get_unchecked(it), *x.get_unchecked(base + it)) };
            // `gv·gv + acc`, the same square fold as `rms_partial_sq`.
            acc = gv.mul_add(gv, acc);
            it += RMS_THREADS;
        }
        // SAFETY: WSUM is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Every access is below RMS_WARPS and ordered by `sync_threads`.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        let part = warp::reduce_sum_f32(acc);
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
        let scale = rms_scale_checked(rms_warp_tree(sums), k as u32, eps, fault, tid == 0);
        let mut it = tid;
        while it < k {
            // SAFETY: the bounds of the summing loop's, and base + it <
            // m·k <= y.len() by the launch contract; thread `tid` owns values
            // tid, tid + RMS_THREADS, … of the row.
            unsafe {
                let gv = mul_rn_f32(*gain.get_unchecked(it), *x.get_unchecked(base + it));
                *y.get_unchecked_mut(base + it) = mul_rn_f32(scale, gv);
            }
            it += RMS_THREADS;
        }
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

    /// Enqueue the Q5_0 embedding lookup of one unit of rows: `args.ids` each
    /// select one row of `args.w` in the `pack_q5_0` layout (`k` values, `k`
    /// a positive multiple of 32, `q5::q5_0_row_words(k/32)` words a row),
    /// dequantized into `args.y`
    /// token-major, bit-identical to `gguf::quant::dequant_row` on the same
    /// row bytes, with each row's position and live key count as
    /// [`Self::enqueue_embed_rows_q4k`]. An id past the table raises
    /// [`FaultSite::TokenId`]. A table whose word count is not that layout's
    /// is refused by name. Asynchronous, allocation-free, capturable.
    pub fn enqueue_embed_rows_q5_0(
        &self,
        stream: &CudaStream,
        args: EmbedRowsArgs<'_>,
        k: usize,
    ) -> Result<(), GpuError> {
        let what = "enqueue_embed_rows_q5_0";
        let EmbedRowsArgs {
            w,
            ids,
            pos0,
            first,
            y,
            pos,
            n_keys,
        } = args;
        if k == 0 || !k.is_multiple_of(32) {
            return Err(GpuError::shape(
                what,
                format!("k must be a positive multiple of 32, got {k}"),
            ));
        }
        let k_blocks = k / 32;
        let row_words = crate::q5::q5_0_row_words(k_blocks);
        if w.rows() == 0 || w.cols() != row_words {
            return Err(GpuError::shape(
                what,
                format!(
                    "a Q5_0 table is {row_words} words a row, got {}x{}",
                    w.rows(),
                    w.cols()
                ),
            ));
        }
        if ids.is_empty() {
            return Err(GpuError::shape(what, "empty ids"));
        }
        if y.len() < k * ids.len() {
            return Err(GpuError::shape(
                what,
                format!("y.len() {} < {k}*{}", y.len(), ids.len()),
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
        let grid = launch_u32(what, "grid", (ids.len() * k).div_ceil(256))?;
        let first = launch_u32(what, "first", first)?;
        let n_rows = launch_u32(what, "w.rows()", w.rows())?;
        let k = launch_u32(what, "k", k)?;
        let s_words = launch_u32(what, "s_words", k_blocks.div_ceil(2))?;
        let k_blocks = launch_u32(what, "k_blocks", k_blocks)?;
        let prep = self
            .module
            .prepare_embed_rows_q5_0(LaunchConfig1D::new(grid, 256, 0))?;
        let fault = crate::sink_over(&self.fault, LAYER_NONE);
        self.module.embed_rows_q5_0(
            stream,
            &prep,
            w.buf(),
            ids,
            pos0,
            first,
            n_rows,
            k,
            s_words,
            k_blocks,
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
    /// super-block), Q3_K `embed_rows_q3k` (110 bytes a super-block) and Q6_K
    /// `embed_rows_q6k` (210 bytes a super-block), each of those two with an
    /// even count of them a row so a row is whole words; any other type, or a
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
            GgmlType::Q3_K => Q3kRows::BYTES,
            GgmlType::Q5_K => Q5kRows::BYTES,
            GgmlType::Q6_K => Q6kRows::BYTES,
            other => {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "a {other} table: the embedding lookups read Q3_K, Q4_K, Q5_K and Q6_K rows"
                    ),
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
        match ty {
            GgmlType::Q3_K => {
                let prep = self.module.prepare_embed_rows_q3k(cfg)?;
                self.module.embed_rows_q3k(
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
            GgmlType::Q5_K => {
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
            }
            GgmlType::Q6_K => {
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
            other => {
                return Err(GpuError::shape(
                    what,
                    format!("a {other} table: no embedding entry launches it"),
                ));
            }
        }
        Ok(())
    }

    /// Enqueue `y = rms_norm(x, gain, eps)` over `m` tokens of `k` values
    /// (token-major). `k` a positive multiple of 32; `x`, `y` hold `k * m`
    /// f32, `gain` holds `k`. Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher: one argument per kernel argument it binds (rust-quality R8)"
    )]
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
        let fault = crate::sink_over(&self.fault, LAYER_NONE);
        self.module
            .rms_norm(stream, &prep, x, gain, eps, k, m, fault, y)?;
        Ok(())
    }

    /// Enqueue the rope rotation of `src` (`m * n_vec * nd` f32, column
    /// `c = t*n_vec + v` covering `nd` values) by the cos/sin caches in `cs`
    /// (`m * nd` f32, token `t`'s interleaved `[cos0, sin0, …]` at
    /// `t*nd` — the host YaRN cache, uploaded per step). `nd` even and at
    /// least 2. `dst` holds `m * n_vec * nd` f32 in `src`'s layout.
    /// Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher: one argument per kernel argument it binds (rust-quality R8)"
    )]
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

    /// Enqueue `y = gelu_tanh(gate) · up` over `n` values (flat; token-major
    /// spans of any width). Asynchronous, allocation-free, capturable.
    pub fn enqueue_geglu(
        &self,
        stream: &CudaStream,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if n == 0 || gate.len() < n || up.len() < n || y.len() < n {
            return Err(GpuError::shape(
                "enqueue_geglu",
                format!(
                    "n={n}, gate.len() {}, up.len() {}, y.len() {}",
                    gate.len(),
                    up.len(),
                    y.len()
                ),
            ));
        }
        let n = launch_u32("enqueue_geglu", "n", n)?;
        let prep = self
            .module
            .prepare_geglu_tanh(LaunchConfig1D::new(n.div_ceil(256), 256, 0))?;
        self.module.geglu_tanh(stream, &prep, gate, up, n, y)?;
        Ok(())
    }

    /// Enqueue `y = situ(gate, up)` at `rule` over `n` values (flat;
    /// token-major spans of any width): [`crate::kquant::Act::Situ`] alone,
    /// any other rule or one [`crate::kquant::Act::code`] refuses is refused
    /// by name. Asynchronous, allocation-free, capturable.
    pub fn enqueue_situ(
        &self,
        stream: &CudaStream,
        rule: crate::kquant::Act,
        (gate, up): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_situ";
        let crate::kquant::Act::Situ { .. } = rule else {
            return Err(GpuError::shape(
                what,
                format!("{rule:?} is not a situ rule"),
            ));
        };
        let (_, beta) = rule.code(what)?;
        if n == 0 || gate.len() < n || up.len() < n || y.len() < n {
            return Err(GpuError::shape(
                what,
                format!(
                    "n={n}, gate.len() {}, up.len() {}, y.len() {}",
                    gate.len(),
                    up.len(),
                    y.len()
                ),
            ));
        }
        let n = launch_u32(what, "n", n)?;
        let prep = self
            .module
            .prepare_situ_rows(LaunchConfig1D::new(n.div_ceil(256), 256, 0))?;
        self.module.situ_rows(stream, &prep, gate, up, n, beta, y)?;
        Ok(())
    }

    /// Enqueue `y = rule(gate, up)` over `n` values (flat; token-major spans
    /// of any width) into f32 rows ([`ElemKernels`]' `swiglu_rule_rows`): any
    /// gate·up rule [`crate::kquant::Act::code`] admits, one it refuses
    /// refused by name. A non-finite gate or up value is NaN in `y` and
    /// raised on `fault`. Asynchronous, allocation-free, capturable.
    pub fn enqueue_swiglu_rule(
        &self,
        stream: &CudaStream,
        rule: crate::kquant::Act,
        (gate, up): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        n: usize,
        (fault, y): (FaultSink, &mut DeviceBuffer<f32>),
    ) -> Result<(), GpuError> {
        let what = "enqueue_swiglu_rule";
        let (act, limit) = rule.code(what)?;
        if n == 0 || gate.len() < n || up.len() < n || y.len() < n {
            return Err(GpuError::shape(
                what,
                format!(
                    "n={n}, gate.len() {}, up.len() {}, y.len() {}",
                    gate.len(),
                    up.len(),
                    y.len()
                ),
            ));
        }
        let n = launch_u32(what, "n", n)?;
        let prep =
            self.module
                .prepare_swiglu_rule_rows(LaunchConfig1D::new(n.div_ceil(256), 256, 0))?;
        self.module
            .swiglu_rule_rows(stream, &prep, gate, up, n, act, limit, fault, y)?;
        Ok(())
    }

    /// Enqueue `y = max(x, 0)^2` over `n` values (flat). A non-finite input
    /// stays that value. Asynchronous, allocation-free, capturable.
    pub fn enqueue_relu_sqr(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if n == 0 || x.len() < n || y.len() < n {
            return Err(GpuError::shape(
                "enqueue_relu_sqr",
                format!("n={n}, x.len() {}, y.len() {}", x.len(), y.len()),
            ));
        }
        let n = launch_u32("enqueue_relu_sqr", "n", n)?;
        let prep = self
            .module
            .prepare_relu_sqr(LaunchConfig1D::new(n.div_ceil(256), 256, 0))?;
        self.module.relu_sqr(stream, &prep, x, n, y)?;
        Ok(())
    }

    /// Enqueue `y[i] = sigmoid(y[i])` over the first `n` values of `y`, in
    /// place. Asynchronous, allocation-free, capturable.
    pub fn enqueue_sigmoid_rows(
        &self,
        stream: &CudaStream,
        y: &mut DeviceBuffer<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        if n == 0 || y.len() < n {
            return Err(GpuError::shape(
                "enqueue_sigmoid_rows",
                format!("n={n}, y.len() {}", y.len()),
            ));
        }
        let n = launch_u32("enqueue_sigmoid_rows", "n", n)?;
        let prep =
            self.module
                .prepare_sigmoid_rows(LaunchConfig1D::new(n.div_ceil(256), 256, 0))?;
        self.module.sigmoid_rows(stream, &prep, y, n)?;
        Ok(())
    }

    /// Enqueue `y[i] = cap·tanh(y[i]/cap)` over the first `n` values of `y`,
    /// in place (the head's logit softcap epilogue). `cap` is finite and
    /// nonzero: any other is refused by name, for it would read every logit
    /// as ±inf or NaN and write a constant. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_softcap_rows(
        &self,
        stream: &CudaStream,
        y: &mut DeviceBuffer<f32>,
        n: usize,
        cap: f32,
    ) -> Result<(), GpuError> {
        let what = "enqueue_softcap_rows";
        if n == 0 || y.len() < n {
            return Err(GpuError::shape(what, format!("n={n}, y.len() {}", y.len())));
        }
        if !cap.is_finite() || cap == 0.0 {
            return Err(GpuError::shape(
                what,
                format!("the softcap {cap} must be finite and nonzero"),
            ));
        }
        let n = launch_u32(what, "n", n)?;
        let prep =
            self.module
                .prepare_softcap_rows(LaunchConfig1D::new(n.div_ceil(256), 256, 0))?;
        self.module.softcap_rows(stream, &prep, y, n, cap)?;
        Ok(())
    }

    /// Enqueue `y[i] = a[i] · b[i]` over `n` values (flat): one rounded
    /// multiply. Asynchronous, allocation-free, capturable.
    pub fn enqueue_mul_rows(
        &self,
        stream: &CudaStream,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        if n == 0 || a.len() < n || b.len() < n || y.len() < n {
            return Err(GpuError::shape(
                "enqueue_mul_rows",
                format!(
                    "n={n}, a.len() {}, b.len() {}, y.len() {}",
                    a.len(),
                    b.len(),
                    y.len()
                ),
            ));
        }
        let n = launch_u32("enqueue_mul_rows", "n", n)?;
        let prep = self
            .module
            .prepare_mul_rows(LaunchConfig1D::new(n.div_ceil(256), 256, 0))?;
        self.module.mul_rows(stream, &prep, a, b, n, y)?;
        Ok(())
    }

    /// Enqueue `y[i] = a[i]·ga[i] + b[i]·gb[i]` over `n` values (flat): the
    /// residual mix of two branches by their own gain vectors, three roundings.
    /// Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher: one argument per kernel argument it binds (rust-quality R8)"
    )]
    pub fn enqueue_mix_rows(
        &self,
        stream: &CudaStream,
        a: &DeviceBuffer<f32>,
        ga: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        gb: &DeviceBuffer<f32>,
        n: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_mix_rows";
        if n == 0 || a.len() < n || ga.len() < n || b.len() < n || gb.len() < n || y.len() < n {
            return Err(GpuError::shape(
                what,
                format!(
                    "n={n}, a.len() {}, ga.len() {}, b.len() {}, gb.len() {}, y.len() {}",
                    a.len(),
                    ga.len(),
                    b.len(),
                    gb.len(),
                    y.len()
                ),
            ));
        }
        let n = launch_u32(what, "n", n)?;
        let prep = self
            .module
            .prepare_mix_rows(LaunchConfig1D::new(n.div_ceil(256), 256, 0))?;
        self.module.mix_rows(stream, &prep, a, ga, b, gb, n, y)?;
        Ok(())
    }

    /// Enqueue `y[i] = s·y[i]` over the first `n` values of `y`, in place
    /// (the head's logit scale before its softcap). `s` finite and nonzero:
    /// any other is refused by name. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_scale_rows(
        &self,
        stream: &CudaStream,
        y: &mut DeviceBuffer<f32>,
        n: usize,
        s: f32,
    ) -> Result<(), GpuError> {
        let what = "enqueue_scale_rows";
        if n == 0 || y.len() < n {
            return Err(GpuError::shape(what, format!("n={n}, y.len() {}", y.len())));
        }
        if !s.is_finite() || s == 0.0 {
            return Err(GpuError::shape(
                what,
                format!("the scale {s} must be finite and nonzero"),
            ));
        }
        let n = launch_u32(what, "n", n)?;
        let prep = self
            .module
            .prepare_scale_rows(LaunchConfig1D::new(n.div_ceil(256), 256, 0))?;
        self.module.scale_rows(stream, &prep, y, n, s)?;
        Ok(())
    }

    /// Enqueue SSMax over `m` tokens of `h` heads of `d` values of `x`, in
    /// place (`ssmax_rows`): token `t`'s value `i` of head `head` is scaled by
    /// `(ssmax_basis(pos[t])·s[head]) + b[head]`. `pos` holds `m` positions
    /// and `s` and `b` the `h` heads' scale and bias, each a device buffer
    /// of f32 (positions u32); the counts and the one `m·h·d` span fit a
    /// launch's u32. Refused before the launch: an empty token, head or
    /// width, short `x`, `pos`, `s` or `b`. Asynchronous, allocation-free,
    /// capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher: one argument per kernel argument it binds (rust-quality R8)"
    )]
    pub fn enqueue_ssmax_rows(
        &self,
        stream: &CudaStream,
        x: &mut DeviceBuffer<f32>,
        pos: &DeviceBuffer<u32>,
        s: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        h: usize,
        d: usize,
        m: usize,
    ) -> Result<(), GpuError> {
        let what = "enqueue_ssmax_rows";
        if m == 0 || h == 0 || d == 0 {
            return Err(GpuError::shape(
                what,
                format!("need m, h and d >= 1, got m={m} h={h} d={d}"),
            ));
        }
        let row = launch_u32(what, "h*d", h.saturating_mul(d))?;
        let span = launch_u32(what, "m*h*d", m.saturating_mul(row as usize))?;
        if x.len() < span as usize || pos.len() < m || s.len() < h || b.len() < h {
            return Err(GpuError::shape(
                what,
                format!(
                    "x.len() {} (need {span}), pos.len() {} (need {m}), s.len() {} and b.len() \
                     {} (need {h})",
                    x.len(),
                    pos.len(),
                    s.len(),
                    b.len()
                ),
            ));
        }
        let per_row = (row as usize).div_ceil(256);
        let grid = launch_u32(what, "grid", m * per_row)?;
        let d = launch_u32(what, "d", d)?;
        let h = launch_u32(what, "h", h)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_ssmax_rows(LaunchConfig1D::new(grid, 256, 0))?;
        self.module
            .ssmax_rows(stream, &prep, x, pos, s, b, d, h, m)?;
        Ok(())
    }

    /// Enqueue `y = gain·LayerNorm(x, eps) + bias` over `m` rows of `k`
    /// values (`ln_rows`): the LayerNorm's mean and variance in f64, the row's
    /// deviations in f32, each output's two products and sum rounded on their
    /// own. `gain` and `bias` hold `k` values each. Refused before the launch:
    /// an empty row or count, short `x`, `gain`, `bias` or `y`. Asynchronous,
    /// allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher: one argument per kernel argument it binds (rust-quality R8)"
    )]
    pub fn enqueue_ln_rows(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        gain: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        eps: f32,
        k: usize,
        m: usize,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_ln_rows";
        if k == 0 || m == 0 {
            return Err(GpuError::shape(
                what,
                format!("need k and m >= 1, got k={k} m={m}"),
            ));
        }
        let span = launch_u32(what, "m*k", m.saturating_mul(k))?;
        if x.len() < span as usize || gain.len() < k || bias.len() < k || y.len() < span as usize {
            return Err(GpuError::shape(
                what,
                format!(
                    "x.len() {} and y.len() {} (need {span}), gain.len() {} and bias.len() {} \
                     (need {k})",
                    x.len(),
                    y.len(),
                    gain.len(),
                    bias.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", m)?;
        let k = launch_u32(what, "k", k)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_ln_rows(LaunchConfig1D::new(grid, RMS_THREADS_U32, 0))?;
        self.module
            .ln_rows(stream, &prep, x, gain, bias, eps, k, m, fault, y)?;
        Ok(())
    }

    /// Enqueue `y = rms_norm(x·gain, eps)` over `m` rows of `k` values, the
    /// norm weightless after the row is scaled by `gain` (`k` a positive
    /// multiple of 32, host-checked; the scaled-and-normed post-sublayer norm
    /// of a model that scales its sublayer output by a gain vector before the
    /// weightless norm). Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher: one argument per kernel argument it binds (rust-quality R8)"
    )]
    pub fn enqueue_gain_rms_norm(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        gain: &DeviceBuffer<f32>,
        eps: f32,
        k: usize,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_gain_rms_norm";
        if k == 0 || !k.is_multiple_of(32) {
            return Err(GpuError::shape(
                what,
                format!("k must be a positive multiple of 32, got {k}"),
            ));
        }
        if m == 0 || x.len() < k * m || gain.len() < k || y.len() < k * m {
            return Err(GpuError::shape(
                what,
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
        let k = launch_u32(what, "k", k)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_gain_rms_norm(LaunchConfig1D::new(m, RMS_THREADS_U32, 0))?;
        let fault = crate::sink_over(&self.fault, LAYER_NONE);
        self.module
            .gain_rms_norm(stream, &prep, x, gain, eps, k, m, fault, y)?;
        Ok(())
    }

    /// Enqueue `y[i] = sqrt_softplus(y[i])` over the first `n` values of
    /// `y`, in place. Asynchronous, allocation-free, capturable.
    pub fn enqueue_softplus_rows(
        &self,
        stream: &CudaStream,
        y: &mut DeviceBuffer<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        if n == 0 || y.len() < n {
            return Err(GpuError::shape(
                "enqueue_softplus_rows",
                format!("n={n}, y.len() {}", y.len()),
            ));
        }
        let n = launch_u32("enqueue_softplus_rows", "n", n)?;
        let prep =
            self.module
                .prepare_softplus_rows(LaunchConfig1D::new(n.div_ceil(256), 256, 0))?;
        self.module.softplus_rows(stream, &prep, y, n)?;
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
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher: one argument per kernel argument it binds (rust-quality R8)"
    )]
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The scale a norm with no row behind it would write if `1/∞ = 0` stood:
    /// an overflowed mean, a non-finite one, and a zero mean with no `eps` are
    /// refused (NaN) at the one owner, whichever width the mean was rounded in.
    #[test]
    fn a_scale_with_no_norm_behind_it_is_nan() {
        let eps = 1e-6f32;
        for mean in [f32::INFINITY, f32::NAN, f32::NEG_INFINITY, -1.0] {
            assert!(rms_scale_of_mean(mean, eps).is_nan(), "mean {mean}");
            assert!(rms_scale(mean, 1, eps).is_nan(), "sum {mean}");
        }
        // No `eps` and an all-zero row: `1/sqrt(0)` is infinite.
        assert!(rms_scale_of_mean(0.0, 0.0).is_nan());
        assert!(rms_scale(0.0, 128, 0.0).is_nan());
    }

    /// Every finite, positive scale is the formula's own bits: the refusal adds
    /// nothing to a row that has a norm.
    #[test]
    fn a_finite_scale_is_the_formula_bit_for_bit() {
        let eps = 1e-6f32;
        for mean in [0.0f32, 1e-30, 0.25, 1.0, 4.0, 1e19, 1e38] {
            let want = 1.0 / (mean + eps).sqrt();
            assert_eq!(rms_scale_of_mean(mean, eps).to_bits(), want.to_bits());
        }
        for (sum, k) in [(512.0f32, 128u32), (2.0e7, 5120), (0.0, 7168)] {
            let want = 1.0 / (sum / k as f32 + eps).sqrt();
            assert_eq!(rms_scale(sum, k, eps).to_bits(), want.to_bits());
            assert_eq!(
                rms_scale(sum, k, eps).to_bits(),
                rms_scale_of_mean(sum / k as f32, eps).to_bits()
            );
        }
    }
}
