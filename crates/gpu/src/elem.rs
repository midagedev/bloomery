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
//! u32 words each) or Q4_K rows of whole super-blocks (36 words each).
//! Extents are launch arguments, never buffer lengths: scratch buffers may
//! be larger than the shape in flight.

use crate::GpuError;
use crate::cores::{funnel16, half_to_f32, q3k_aux_scales, q3k_sub_scale, q4k_scale_min};
use crate::launch_u32;
use crate::tensor::DeviceTensor;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
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

/// Threads the argmax gives one vector. Same reason as [`RMS_THREADS`] and
/// the same shape: the head's row is 102,400 logits, so one warp over the
/// whole vector is 3,200 dependent loads per lane with a single warp
/// resident. The scan is strided by this width, the merge is a per-warp
/// butterfly then a fixed ascending walk of the warp slots.
pub const ARGMAX_THREADS: usize = 256;
/// [`ARGMAX_THREADS`] as the `u32` block width a launch takes.
const ARGMAX_THREADS_U32: u32 = ARGMAX_THREADS as u32;
const _: () = assert!(ARGMAX_THREADS_U32 as usize == ARGMAX_THREADS);
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
/// 108..109.
///
/// Caller contract: `base + 110 <= 4 * w.len()` and `base` inside one row's
/// span (rows are 880 bytes, so `base` sits 0 or 2 mod 4 with the
/// super-block; the scale window funnels the 2-mod-4 case, single bytes load
/// from their covering words), `v16 < 256`.
#[inline(always)]
pub(crate) fn q3k_embed_value(w: &[u32], base: usize, v16: usize) -> f32 {
    let field = (v16 >> 5) & 3;
    let qs_byte = 32 * (v16 >> 7) + 16 * ((v16 >> 4) & 1) + (v16 & 15);
    // Single bytes load directly from their covering word — the value's qs
    // and hmask bytes sit at arbitrary byte offsets (the gemv reads whole
    // aligned quads and funnels; a lone byte needs no funnel).
    let qx = base + 32 + qs_byte;
    // SAFETY: qx < base + 95 < base + 110 <= 4 * w.len() by the caller
    // contract, so the covering word is inside w.
    let qsw = unsafe { *w.get_unchecked(qx >> 2) };
    let qv = (qsw >> (8 * (qx & 3) + 2 * field)) & 3;

    let hx = base + (v16 & 31);
    // SAFETY: hx < base + 32, inside the super-block by the caller contract.
    let hmw = unsafe { *w.get_unchecked(hx >> 2) };
    let hv = if (hmw >> (8 * (hx & 3) + 4 * (v16 >> 7) + field)) & 1 != 0 {
        0
    } else {
        4
    };

    let par = (base >> 1) & 1;

    let ak = (base + 96) >> 2;
    // SAFETY: the 12 scale bytes end at 108 and d at 110, inside the
    // super-block by the caller contract.
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
/// Caller contract: `wk + 36 <= w.len()` (the super-block's 36 words), `v < 256`.
#[inline(always)]
pub(crate) fn q4k_embed_value(w: &[u32], wk: usize, v: usize) -> f32 {
    let j = v >> 6;
    let h = (v >> 5) & 1;
    let l = v & 31;
    let qb = 32 * j + l;
    // SAFETY: every index is wk + 0..=35 (qb < 128, so 4 + qb/4 <= 35),
    // inside `w` by the caller contract.
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

/// The partial sum of squares thread `tid` of an [`RMS_THREADS`] block owns
/// for the row of `k` values at `base`: values `tid, tid + RMS_THREADS, …`
/// ascending, each square added with one fused multiply-add (the device
/// build contracts `acc + v·v`; a host transcription of this order uses
/// `mul_add`). The norm's fixed per-thread order, shared by `rms_norm` and
/// the fused `norm_quant`.
///
/// Caller contract: `base + k <= x.len()`, `tid < RMS_THREADS`.
#[inline(always)]
pub fn rms_partial_sq(x: &[f32], base: usize, k: usize, tid: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut it = tid;
    while it < k {
        // SAFETY: it < k and base + k <= x.len() by the caller contract.
        let v = unsafe { *x.get_unchecked(base + it) };
        acc += v * v;
        it += RMS_THREADS;
    }
    acc
}

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
/// the buffer's expert axis, one plain multiply then add per term (no fused
/// multiply-add), so the sequence of adds is fixed.
///
/// Caller contract: `down.len() >= rows * n_exp * m`, `w.len() >= n_exp * m`,
/// `t < m`, `d < rows`.
#[inline(always)]
pub(crate) fn weighted_expert_sum(
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
        // SAFETY: e < n_exp and t < m bound the weight index t*n_exp + e and
        // the down index (t*n_exp + e)*rows + d inside their buffers by the
        // caller contract.
        let wv = unsafe { *w.get_unchecked(t * n_exp + e) };
        // SAFETY: the same e < n_exp and t < m bound the down index
        // (t*n_exp + e)*rows + d inside `down` by the caller contract.
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
    /// table's `n_rows` rows reads row 0 — deterministic and in-bounds, never
    /// an out-of-bounds read — and the step's id source is the argmax, below
    /// the vocabulary by construction.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (4 * w.len() >= 880 * n_rows, y.len() >= 2048 * ids.len())
    )]
    pub fn embed_rows(w: &[u32], ids: &[u32], n_rows: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= ids.len() * 2048 {
            return;
        }
        let t = i >> 11;
        let k = i & 2047;
        // SAFETY: t < ids.len() by the guard.
        let id = unsafe { *ids.get_unchecked(t) };
        let id = (if id < n_rows { id } else { 0 }) as usize;
        // Row spans are whole 880-byte blocks, so only the super-block offset
        // can sit 2 mod 4; the core funnels both alignments.
        let v = q3k_embed_value(w, id * 880 + ((k >> 8) * 110), k & 255);
        // SAFETY: i < ids.len() * 2048 <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = v;
        }
    }

    /// Dequantize `ids.len()` rows of the Q4_K embedding table `w` (`36 ·
    /// n_sb` u32 words per row, `256 · n_sb` values) into `y`, token-major,
    /// one thread per value through [`q4k_embed_value`]. Q4_K rows are whole
    /// words, so no funnel is needed. An id past the table's `n_rows` rows
    /// reads row 0, as `embed_rows` does.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= 36 * n_sb * n_rows,
            y.len() >= 256 * n_sb * ids.len()
        )
    )]
    pub fn embed_rows_q4k(
        w: &[u32],
        ids: &[u32],
        n_rows: u32,
        n_sb: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let width = 256 * n_sb as usize;
        if i >= ids.len() * width {
            return;
        }
        let t = i / width;
        let k = i % width;
        // SAFETY: t < ids.len() by the guard.
        let id = unsafe { *ids.get_unchecked(t) };
        let id = (if id < n_rows { id } else { 0 }) as usize;
        let wk = (id * n_sb as usize + (k >> 8)) * 36;
        let v = q4k_embed_value(w, wk, k & 255);
        // SAFETY: i < ids.len() * width <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = v;
        }
    }

    /// RMS norm, one [`RMS_THREADS`] block per token: `rms_partial_sq` per
    /// thread, the fixed five-step butterfly per warp, the warp sums combined
    /// by `rms_warp_tree`, then `(scale · gain) · x` per value in the
    /// reference's order. `k` a positive multiple of 32 (host-checked); the
    /// token guard is block-uniform, so no barrier and no warp collective is
    /// skipped, and a thread past `k` contributes an exact zero.
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
        // SAFETY: WSUM is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Every access is below RMS_WARPS and ordered by `sync_threads`.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        let part = warp::reduce_sum_f32(rms_partial_sq(x, base, k, tid));
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
        let mut it = tid;
        while it < k {
            // SAFETY: it < k bounds the gain read by the contract and the x
            // read as base + it < k*m; base + it < k*m <= y.len() too.
            unsafe {
                let g = *gain.get_unchecked(it);
                let v = *x.get_unchecked(base + it);
                *y.get_unchecked_mut(base + it) = (scale * g) * v;
            }
            it += RMS_THREADS;
        }
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
        let col = i / npairs;
        let d = (i % npairs) * 2;
        let nd = nd as usize;
        let t = col / n_vec as usize;
        // SAFETY: d + 1 <= nd - 1, so both src indices stay below
        // m*n_vec*nd <= src.len() and both cache indices below m*nd <=
        // cs.len() by the launch contract.
        let (x0, x1, c, s) = unsafe {
            (
                *src.get_unchecked(col * nd + d),
                *src.get_unchecked(col * nd + d + 1),
                *cs.get_unchecked(t * nd + d),
                *cs.get_unchecked(t * nd + d + 1),
            )
        };
        let (y0, y1) = rope_pair_core(x0, x1, c, s);
        // SAFETY: the same indices as the loads, inside dst by the contract.
        unsafe {
            *dst.get_unchecked_mut(col * nd + d) = y0;
            *dst.get_unchecked_mut(col * nd + d + 1) = y1;
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
        // SAFETY: i < n bounds both loads by the launch contract.
        let (g, u) = unsafe { (*gate.get_unchecked(i), *up.get_unchecked(i)) };
        // SAFETY: i < n <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = silu_mul(g, u);
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
        // SAFETY: i < n bounds both loads by the launch contract.
        let (av, bv) = unsafe { (*a.get_unchecked(i), *b.get_unchecked(i)) };
        // SAFETY: i < n <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = av + bv;
        }
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
        let v = weighted_expert_sum(down, w, rows, n_exp, t, d);
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
        // SAFETY: i < n <= bits.len() by the launch contract.
        let b = unsafe { *bits.get_unchecked(i) } as u16;
        // SAFETY: i < n <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = half_to_f32(b);
        }
    }

    /// Index of the maximum of `n` f32, ties to the lower index — the greedy
    /// sampler's rule. One [`ARGMAX_THREADS`] block over the whole vector:
    /// each thread's best over its strided share (indices ascending within a
    /// thread), the fixed xor butterfly merging by (value desc, index asc)
    /// inside each warp, then thread 0 walking the [`ARGMAX_WARPS`] warp
    /// slots in ascending order. Every stage is the same total order
    /// ([`argmax_take`]), so the result is a function of the input alone
    /// and the tie rule survives every regrouping. A thread whose share is
    /// empty keeps the `(-inf, 0)` sentinel and can never win against a real
    /// value, which is also what an all-`-inf` vector answers: index 0, the
    /// host reference's answer. The result stays on the device (`out[0]`,
    /// u32); the step reads it back once. Inputs are finite — the gate's
    /// loader rejects anything else.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= n, out.len() >= 1))]
    pub fn argmax(x: &[f32], n: u32, mut out: DisjointSlice<u32>) {
        static mut BEST_V: SharedArray<f32, ARGMAX_WARPS> = SharedArray::UNINIT;
        static mut BEST_I: SharedArray<u32, ARGMAX_WARPS> = SharedArray::UNINIT;

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
            // SAFETY: out.len() >= 1 by the launch contract; only thread 0
            // writes.
            unsafe {
                *out.get_unchecked_mut(0) = fi;
            }
        }
    }

    /// [`argmax`] over each of `m` interleaved rows: block c takes row c's
    /// `n` values `x[i·m + c]` and writes their argmax to `out[c]`, with
    /// `argmax`'s walk, butterfly, slot order and tie rule — each row's
    /// answer is `argmax` on that row alone. The layout is the gemvs'
    /// `y[r·m + c]` output, so a head's m logit rows go straight in.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= n * m, out.len() >= m))]
    pub fn argmax_rows(x: &[f32], n: u32, m: u32, mut out: DisjointSlice<u32>) {
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
            // SAFETY: c < m <= out.len(); block c's thread 0 alone writes it.
            unsafe {
                *out.get_unchecked_mut(c as usize) = fi;
            }
        }
    }

    /// The argmax of `x[i·stride + col]` for `i < n`, as [`argmax`] documents
    /// its walk, meaningful on thread 0 of the block; every thread of the
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

/// The loaded P4 device module. Owns no context and no stream — the caller
/// passes the engine stream (`Gpu::stream()`) per enqueue, so launches order
/// with the rest of the step and are capturable.
pub struct ElemKernels {
    module: elem_kernels::LoadedModule,
}

impl ElemKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<ElemKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { elem_kernels::load(ctx)? };
        Ok(ElemKernels { module })
    }

    /// Enqueue the embedding lookup: `ids` (device-resident token ids, every
    /// id below the table's row count) each select one Q3_K row of `w` (220
    /// u32 words per row), dequantized to 2048 f32. `y` holds `2048 *
    /// ids.len()` f32, token-major. Bit-identical to
    /// `gguf::quant::dequant_row` on the same row bytes. Asynchronous,
    /// allocation-free, capturable.
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
        self.module
            .embed_rows(stream, &prep, w.buf(), ids, n_rows, y)?;
        Ok(())
    }

    /// Enqueue the Q4_K embedding lookup: `ids` (device-resident token ids)
    /// each select one row of `w` (`36 · n_sb` u32 words per row, `256 ·
    /// n_sb` values, `n_sb = w.cols() / 36`), dequantized into `y`
    /// token-major. Bit-identical to `gguf::quant::dequant_row` on the same
    /// row bytes. Asynchronous, allocation-free, capturable.
    pub fn enqueue_embed_rows_q4k(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        ids: &DeviceBuffer<u32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_embed_rows_q4k";
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
        let grid = launch_u32(what, "grid", (ids.len() * 256 * n_sb).div_ceil(256))?;
        let n_rows = launch_u32(what, "w.rows()", w.rows())?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_embed_rows_q4k(LaunchConfig1D::new(grid, 256, 0))?;
        self.module
            .embed_rows_q4k(stream, &prep, w.buf(), ids, n_rows, n_sb, y)?;
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

    /// Enqueue the argmax of `n` f32 into `out[0]` (u32, device-resident;
    /// ties to the lower index). One [`ARGMAX_THREADS`] block walks the
    /// whole vector. Asynchronous, allocation-free, capturable.
    pub fn enqueue_argmax(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        n: usize,
        out: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        if n == 0 || x.len() < n || out.is_empty() {
            return Err(GpuError::shape(
                "enqueue_argmax",
                format!("n={n}, x.len() {}, out.len() {}", x.len(), out.len()),
            ));
        }
        let n = launch_u32("enqueue_argmax", "n", n)?;
        let prep = self
            .module
            .prepare_argmax(LaunchConfig1D::new(1, ARGMAX_THREADS_U32, 0))?;
        self.module.argmax(stream, &prep, x, n, out)?;
        Ok(())
    }

    /// Enqueue the argmax of each of `m` interleaved rows of `n` f32 — row c
    /// is `x[i*m + c]`, the gemvs' `m`-column output layout — into `out[c]`,
    /// each row's answer the one [`Self::enqueue_argmax`] gives that row
    /// alone. One [`ARGMAX_THREADS`] block per row. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_argmax_rows(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        n: usize,
        m: usize,
        out: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        if n == 0 || m == 0 || x.len() < n * m || out.len() < m {
            return Err(GpuError::shape(
                "enqueue_argmax_rows",
                format!("n={n} m={m}, x.len() {}, out.len() {}", x.len(), out.len()),
            ));
        }
        let what = "enqueue_argmax_rows";
        // The walk indexes i*m + c in u32.
        launch_u32(what, "n*m", n * m)?;
        let n = launch_u32(what, "n", n)?;
        let m = launch_u32(what, "m", m)?;
        let prep =
            self.module
                .prepare_argmax_rows(LaunchConfig1D::new(m, ARGMAX_THREADS_U32, 0))?;
        self.module.argmax_rows(stream, &prep, x, n, m, out)?;
        Ok(())
    }
}
