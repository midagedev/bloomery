//! Q8_0-weight and F32-weight gemv with f32 activations (package P3): the
//! two matmul sites whose weights are not K-quants. The attention `q_nope2`
//! site consumes the load-time Q8_0 requant of `wk_b` (32-value blocks), and
//! the MoE router `ffn_gate_inp` is F32. Both are precision-sensitive — a
//! flipped near-tie in the router's top-6 changes the token — so activations
//! stay f32 here: the kernel dequantizes inline and accumulates in f32, and
//! the gate band is accordingly tighter than the q8_1-activation kernels'.
//!
//! Q8_0 device layout, fixed at load time (decision 4: format conversion is
//! load-time work):
//! - `qs`: row-major `u32` words, 8 words per 32-value block, code `j` in
//!   word `j/4`, byte `j%4` (little-endian), `k/4` words per row;
//! - `d`: row-major block scales, `k/32` per row, each the Q8_0 block's f16
//!   bits as the file stores them. A lane widens the scale it loads with the
//!   hardware convert (`flash::half_bits_to_f32`); widening f16 to f32 is
//!   exact, so the kernels multiply by the value the reference dequantizes
//!   with and do no f16 arithmetic.
//!
//! Numeric contract, both kernels: one warp owns one output row, each lane
//! accumulates its share of the row sequentially, one f32 multiply-add per
//! term, and the 32 lane sums are then combined by the fixed five-step
//! butterfly (xor 16, 8, 4, 2, 1). A lane's share:
//! - F32, and Q8_0 at `m > 1`: the row's 32-value chunks — value index
//!   `32·it + L` for lane L, chunks in increasing `it`, columns in increasing
//!   order;
//! - Q8_0 at `m = 1`, the decode shape: whole code words — words `L, L + 32,
//!   L + 64, …` of the row in increasing order, each word's four values in
//!   byte order.
//!
//! The combination tree is a function of (k, m) only — never of the data,
//! the row index, or the grid geometry. Output layout matches the K-quant
//! gemvs: `y[r·m + c]`.

use crate::GpuError;
use crate::flash::half_bits_to_f32;
use crate::launch_u32;
use crate::tensor::DeviceTensor;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::vector::{F32x4, as_vectors};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

// --------------------------------------------------------------- cores
//
// Same standing as `crate::cores`: ordinary `#[inline(always)]` functions a
// per-op wrapper and a later fused kernel can both call. Slice arguments are
// the whole device buffer plus the row index — the verified core shape
// (`cores::q4k_a_chain` takes a buffer and a base), not a `DisjointSlice`.

/// Lane `lane`'s partial sums for one F32 row: `Σ_it fma(w[row·k + 32·it +
/// lane], x[c·k + 32·it + lane])`, accumulated sequentially in `it`.
/// Columns past `m_cols` stay 0.0.
///
/// Caller contract: `w.len() >= (row + 1) * k`, `x.len() >= m_cols * k`,
/// `k` a positive multiple of 32, `m_cols` in 1..=8, `lane < 32`.
#[inline(always)]
pub(crate) fn f32_lane_partials(
    w: &[f32],
    x: &[f32],
    k: u32,
    row: usize,
    m_cols: u32,
    lane: usize,
) -> [f32; 8] {
    // One column is the decode shape, and it gets a body of its own: the
    // guards below are runtime tests, so in the general loop each chunk's
    // multiply-add sits behind its own branch and the lane can have only
    // that chunk's load in flight. See `f32_lane_partial_1col`.
    if m_cols == 1 {
        // SAFETY: with m_cols 1 this function's caller contract is the
        // callee's safety contract.
        let f0 = unsafe { f32_lane_partial_1col(w, x, k, row, lane) };
        return [f0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    }
    let k = k as usize;
    let w_row = row * k;
    let mut f0 = 0.0f32;
    let mut f1 = 0.0f32;
    let mut f2 = 0.0f32;
    let mut f3 = 0.0f32;
    let mut f4 = 0.0f32;
    let mut f5 = 0.0f32;
    let mut f6 = 0.0f32;
    let mut f7 = 0.0f32;
    let mut it = 0usize;
    while it < k >> 5 {
        let kk = it * 32 + lane;
        // SAFETY: kk < k <= w.len() - w_row by the caller contract.
        let wv = unsafe { *w.get_unchecked(w_row + kk) };
        // Column 0 (always active).
        // SAFETY: kk < k <= x.len() by the caller contract (m_cols >= 1).
        f0 = f32::mul_add(wv, unsafe { *x.get_unchecked(kk) }, f0);
        // Columns 1..7: one launch-uniform guard per column so the work
        // scales with m_cols; each guard makes the column's span live.
        if m_cols > 1 {
            // SAFETY: m_cols > 1 => x.len() >= 2*k > k + kk.
            f1 = f32::mul_add(wv, unsafe { *x.get_unchecked(k + kk) }, f1);
        }
        if m_cols > 2 {
            // SAFETY: m_cols > 2 => x.len() >= 3*k > 2*k + kk.
            f2 = f32::mul_add(wv, unsafe { *x.get_unchecked(2 * k + kk) }, f2);
        }
        if m_cols > 3 {
            // SAFETY: m_cols > 3 => x.len() >= 4*k > 3*k + kk.
            f3 = f32::mul_add(wv, unsafe { *x.get_unchecked(3 * k + kk) }, f3);
        }
        if m_cols > 4 {
            // SAFETY: m_cols > 4 => x.len() >= 5*k > 4*k + kk.
            f4 = f32::mul_add(wv, unsafe { *x.get_unchecked(4 * k + kk) }, f4);
        }
        if m_cols > 5 {
            // SAFETY: m_cols > 5 => x.len() >= 6*k > 5*k + kk.
            f5 = f32::mul_add(wv, unsafe { *x.get_unchecked(5 * k + kk) }, f5);
        }
        if m_cols > 6 {
            // SAFETY: m_cols > 6 => x.len() >= 7*k > 6*k + kk.
            f6 = f32::mul_add(wv, unsafe { *x.get_unchecked(6 * k + kk) }, f6);
        }
        if m_cols > 7 {
            // SAFETY: m_cols > 7 => x.len() >= 8*k > 7*k + kk.
            f7 = f32::mul_add(wv, unsafe { *x.get_unchecked(7 * k + kk) }, f7);
        }
        it += 1;
    }
    [f0, f1, f2, f3, f4, f5, f6, f7]
}

/// Chunks whose loads the F32 single-column body issues before its first
/// multiply-add. The row walk is a dependent chain — each chunk's address is
/// known in advance but its multiply-add feeds the next — so with one chunk
/// in flight a lane pays one memory round trip per chunk and the row's cost
/// is the chunk count times that latency, whatever the grid geometry is.
/// Issuing this many loads first is what turns the walk into a bandwidth
/// problem; it does not touch the accumulation order. The body spells the
/// chunks out as named scalars, so the value is pinned to the width it is
/// written for. The Q8_0 single-column body walks whole words instead and
/// carries its own width, [`Q8_STEP_UNROLL`].
pub const LANE_UNROLL: usize = 8;
const _: () = assert!(LANE_UNROLL == 8);

/// Lane `lane`'s partial sum for one F32 row against a single activation
/// column: `Σ_it fma(w[row·k + 32·it + lane], x[32·it + lane])`, accumulated
/// sequentially in `it` — [`f32_lane_partials`]'s column 0, in the same order,
/// with [`LANE_UNROLL`] chunks' loads hoisted above the multiply-adds that
/// consume them.
///
/// # Safety
///
/// The caller contract of [`f32_lane_partials`] with `m_cols` 1: `w.len() >=
/// (row + 1) * k`, `x.len() >= k`, `k` a positive multiple of 32, `lane <
/// 32`. The body reads `w` and `x` unchecked within those bounds.
#[inline(always)]
pub unsafe fn f32_lane_partial_1col(w: &[f32], x: &[f32], k: u32, row: usize, lane: usize) -> f32 {
    let k = k as usize;
    let w_row = row * k;
    let iters = k >> 5;
    let mut f = 0.0f32;
    let mut it = 0usize;
    // Named scalars, not an array: an array whose element a loop selects is
    // served from a local depot whatever the loop unrolls to, and the round
    // trip that costs is the thing this body exists to avoid.
    while it + LANE_UNROLL <= iters {
        let kk = it * 32 + lane;
        // SAFETY: kk + 224 < k by the loop guard (LANE_UNROLL = 8), so every
        // read is inside w[w_row .. w_row + k] and x[0 .. k], both covered by
        // the caller contract (w.len() >= (row+1)*k, x.len() >= k).
        let (w0, w1, w2, w3, w4, w5, w6, w7, x0, x1, x2, x3, x4, x5, x6, x7) = unsafe {
            (
                *w.get_unchecked(w_row + kk),
                *w.get_unchecked(w_row + kk + 32),
                *w.get_unchecked(w_row + kk + 64),
                *w.get_unchecked(w_row + kk + 96),
                *w.get_unchecked(w_row + kk + 128),
                *w.get_unchecked(w_row + kk + 160),
                *w.get_unchecked(w_row + kk + 192),
                *w.get_unchecked(w_row + kk + 224),
                *x.get_unchecked(kk),
                *x.get_unchecked(kk + 32),
                *x.get_unchecked(kk + 64),
                *x.get_unchecked(kk + 96),
                *x.get_unchecked(kk + 128),
                *x.get_unchecked(kk + 160),
                *x.get_unchecked(kk + 192),
                *x.get_unchecked(kk + 224),
            )
        };
        f = f32::mul_add(w0, x0, f);
        f = f32::mul_add(w1, x1, f);
        f = f32::mul_add(w2, x2, f);
        f = f32::mul_add(w3, x3, f);
        f = f32::mul_add(w4, x4, f);
        f = f32::mul_add(w5, x5, f);
        f = f32::mul_add(w6, x6, f);
        f = f32::mul_add(w7, x7, f);
        it += LANE_UNROLL;
    }
    while it < iters {
        let kk = it * 32 + lane;
        // SAFETY: kk < k by the loop guard, so both reads are inside
        // w[w_row .. w_row + k] and x[0 .. k] by the caller contract.
        let (wv, xv) = unsafe { (*w.get_unchecked(w_row + kk), *x.get_unchecked(kk)) };
        f = f32::mul_add(wv, xv, f);
        it += 1;
    }
    f
}

/// Lane `lane`'s partial sums for one Q8_0 row: the weight at value `kk` of
/// `row` is `q·d` with `q` the signed code in word `qs[row·k/4 + kk/4]`,
/// byte `kk%4`, and `d` the block scale `d[row·k/32 + kk/32]` widened from
/// its f16 bits — the same bits the reference's dequantizer produces. `x`
/// is read from base `x0` (column c's values at `x0 + c*k .. +k`), so a
/// caller can dot against a slice of a wider buffer without subslicing it.
/// Accumulation order: for `m_cols > 1` as `f32_lane_partials`; for
/// `m_cols == 1` as [`q8_0_lane_partial_1col`].
///
/// Caller contract: `qs.len() >= (row + 1) * k/4`, `d.len() >= (row + 1) *
/// k/32`, `x.len() >= x0 + m_cols * k`, `k` a positive multiple of 32,
/// `m_cols` in 1..=8, `lane < 32`.
#[inline(always)]
pub(crate) fn q8_0_lane_partials(
    qs: &[u32],
    d: &[u16],
    x: &[f32],
    k: u32,
    row: usize,
    x0: usize,
    m_cols: u32,
    lane: usize,
) -> [f32; 8] {
    // One column is the decode shape and gets a body of its own: whole code
    // words per lane, loads hoisted. See `q8_0_lane_partial_1col`.
    if m_cols == 1 {
        // SAFETY: with m_cols 1 this function's caller contract is the
        // callee's safety contract.
        let f0 = unsafe { q8_0_lane_partial_1col(qs, d, x, k, row, x0, lane) };
        return [f0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    }
    let k = k as usize;
    let qs_row = row * (k >> 2);
    let d_row = row * (k >> 5);
    let mut f0 = 0.0f32;
    let mut f1 = 0.0f32;
    let mut f2 = 0.0f32;
    let mut f3 = 0.0f32;
    let mut f4 = 0.0f32;
    let mut f5 = 0.0f32;
    let mut f6 = 0.0f32;
    let mut f7 = 0.0f32;
    let mut it = 0usize;
    while it < k >> 5 {
        let kk = it * 32 + lane;
        // SAFETY: kk < k, so word kk/4 < k/4 <= qs.len() - qs_row by the
        // caller contract.
        let q = unsafe { (*qs.get_unchecked(qs_row + (kk >> 2)) >> (8 * (kk & 3))) as u8 as i8 };
        // SAFETY: it = kk/32 < k/32 <= d.len() - d_row by the caller
        // contract.
        let wv = q as f32 * half_bits_to_f32(unsafe { *d.get_unchecked(d_row + it) });
        // Column 0 (always active).
        // SAFETY: kk < k, and x0 + kk < x0 + k <= x.len() by the caller
        // contract (m_cols >= 1).
        f0 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + kk) }, f0);
        // Columns 1..7, one launch-uniform guard per column.
        if m_cols > 1 {
            // SAFETY: m_cols > 1 => x.len() >= x0 + 2*k > x0 + k + kk.
            f1 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + k + kk) }, f1);
        }
        if m_cols > 2 {
            // SAFETY: m_cols > 2 => x.len() >= x0 + 3*k > x0 + 2*k + kk.
            f2 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + 2 * k + kk) }, f2);
        }
        if m_cols > 3 {
            // SAFETY: m_cols > 3 => x.len() >= x0 + 4*k > x0 + 3*k + kk.
            f3 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + 3 * k + kk) }, f3);
        }
        if m_cols > 4 {
            // SAFETY: m_cols > 4 => x.len() >= x0 + 5*k > x0 + 4*k + kk.
            f4 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + 4 * k + kk) }, f4);
        }
        if m_cols > 5 {
            // SAFETY: m_cols > 5 => x.len() >= x0 + 6*k > x0 + 5*k + kk.
            f5 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + 5 * k + kk) }, f5);
        }
        if m_cols > 6 {
            // SAFETY: m_cols > 6 => x.len() >= x0 + 7*k > x0 + 6*k + kk.
            f6 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + 6 * k + kk) }, f6);
        }
        if m_cols > 7 {
            // SAFETY: m_cols > 7 => x.len() >= x0 + 8*k > x0 + 7*k + kk.
            f7 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + 7 * k + kk) }, f7);
        }
        it += 1;
    }
    [f0, f1, f2, f3, f4, f5, f6, f7]
}

/// Values one step of the single-column Q8_0 body covers: each of the 32
/// lanes takes one whole code word — four consecutive values of one block —
/// so a step is 32 consecutive words, 128 values, four blocks, and the warp's
/// code load in a step is one contiguous 128-byte line.
const Q8_STEP: usize = 128;

/// Steps whose code, scale and activation loads the single-column Q8_0 body
/// issues before its first multiply-add: the Q8_0 counterpart of
/// [`LANE_UNROLL`], counted in 128-value steps. A step is six registers of
/// loaded data per lane (a code word, a scale, four activations), so four
/// steps are 24, and the entry keeps four 256-thread blocks per SM. The body
/// spells the steps out as named scalars, so the value is pinned to the
/// width it is written for.
pub const Q8_STEP_UNROLL: usize = 4;
const _: () = assert!(Q8_STEP_UNROLL == 4);

/// Step `s` of lane `lane`'s walk: the lane's code word `qs[wq + 32s]`, the
/// scale `d[wd + 4s]` of the block that word sits in, widened from its f16
/// bits, and the word's four activations, quad `lane + 32s` of `xq`.
///
/// # Safety
///
/// Word `w = 32s + lane` of the row must exist, `w < k/4`, with `wq`, `wd`
/// and `xq` built as [`q8_0_lane_partial_1col`] builds them under its caller
/// contract.
#[inline(always)]
unsafe fn q8_0_step(
    qs: &[u32],
    d: &[u16],
    xq: &[F32x4],
    wq: usize,
    wd: usize,
    lane: usize,
    s: usize,
) -> (u32, f32, F32x4) {
    // SAFETY: w = 32s + lane < k/4 by this function's contract, so the code
    // word wq + 32s is below row*k/4 + k/4 <= qs.len(), its block wd + 4s =
    // row*k/32 + w/8 below row*k/32 + k/32 <= d.len(), and quad w below k/4
    // = xq.len().
    let (q, bits, v) = unsafe {
        (
            *qs.get_unchecked(wq + 32 * s),
            *d.get_unchecked(wd + 4 * s),
            *xq.get_unchecked(lane + 32 * s),
        )
    };
    (q, half_bits_to_f32(bits), v)
}

/// `f` plus one word's four terms, in byte order: `fma(q_j·d, x_j, f)` with
/// `q_j` the signed code in byte `j` of `q` and `x_j` lane `j` of `x`. The
/// product `q_j·d` is exact in f32, so each term rounds once.
#[inline(always)]
fn q8_0_word_dot(f: f32, q: u32, d: f32, x: F32x4) -> f32 {
    let [x0, x1, x2, x3] = x.to_array();
    let f = f32::mul_add(q as u8 as i8 as f32 * d, x0, f);
    let f = f32::mul_add((q >> 8) as u8 as i8 as f32 * d, x1, f);
    let f = f32::mul_add((q >> 16) as u8 as i8 as f32 * d, x2, f);
    f32::mul_add((q >> 24) as u8 as i8 as f32 * d, x3, f)
}

/// Lane `lane`'s partial sum for one Q8_0 row against a single activation
/// column. The weight decode is [`q8_0_lane_partials`]'s; the lane's share of
/// the row is not: lane L owns whole code words — words `L, L + 32, L + 64,
/// …` below `k/4`, word `w` holding values `4w .. 4w + 3` — and accumulates
/// them in increasing order, each word's four values in byte order, one f32
/// multiply-add per term. [`Q8_STEP_UNROLL`] steps' loads are hoisted above
/// the multiply-adds that consume them; up to three leftover steps run as a
/// hoisted pair and a single, and a `k` that is not a multiple of 128 ends
/// with one word on each of the first `(k % 128) / 4` lanes. A row of at
/// most 128 values — one word per lane at most — skips that ladder: it is
/// the same single word, reached by one test instead of four. The
/// activations are read as one 16-byte quad per word when `x0` leaves them
/// 16-byte aligned, and as four scalars otherwise — the same values in the
/// same order, so the sum does not depend on which.
///
/// # Safety
///
/// The caller contract of [`q8_0_lane_partials`] with `m_cols` 1: `qs.len()
/// >= (row + 1) * k/4`, `d.len() >= (row + 1) * k/32`, `x.len() >= x0 + k`,
/// `k` a positive multiple of 32, `lane < 32`. The body reads `qs`, `d` and
/// `x` unchecked within those bounds.
#[inline(always)]
pub unsafe fn q8_0_lane_partial_1col(
    qs: &[u32],
    d: &[u16],
    x: &[f32],
    k: u32,
    row: usize,
    x0: usize,
    lane: usize,
) -> f32 {
    let k = k as usize;
    let words = k >> 2;
    let steps = k / Q8_STEP;
    // Step 0 of the lane's walk: its word and that word's scale; a step
    // moves both on by 32 words and 4 scales.
    let wq = row * words + lane;
    let wd = row * (k >> 5) + (lane >> 3);
    // SAFETY: x0 + k <= x.len() by the caller contract.
    let xr = unsafe { x.get_unchecked(x0..x0 + k) };
    let mut f = 0.0f32;
    let Some(xq) = as_vectors::<F32x4>(xr) else {
        // Activations not 16-byte aligned at x0: the same walk, one word at a
        // time, through scalar reads.
        let mut w = lane;
        while w < words {
            // SAFETY: w < k/4 by the loop guard, so word row*k/4 + w, its
            // block row*k/32 + w/8 and values 4w .. 4w + 3 of xr are inside
            // qs, d and xr by the caller contract.
            let (q, bits, xv) = unsafe {
                (
                    *qs.get_unchecked(wq - lane + w),
                    *d.get_unchecked(wd - (lane >> 3) + (w >> 3)),
                    F32x4::new([
                        *xr.get_unchecked(4 * w),
                        *xr.get_unchecked(4 * w + 1),
                        *xr.get_unchecked(4 * w + 2),
                        *xr.get_unchecked(4 * w + 3),
                    ]),
                )
            };
            f = q8_0_word_dot(f, q, half_bits_to_f32(bits), xv);
            w += 32;
        }
        return f;
    };
    // At most 128 values: at most one word per lane, step 0's — the word the
    // ladder below reaches through four tests, each its own branch region,
    // reached here through one.
    if words <= 32 {
        if lane < words {
            // SAFETY: lane < k/4 is the step helper's bound for step 0.
            let (q0, d0, v0) = unsafe { q8_0_step(qs, d, xq, wq, wd, lane, 0) };
            f = q8_0_word_dot(f, q0, d0, v0);
        }
        return f;
    }
    let mut s = 0usize;
    while s + Q8_STEP_UNROLL <= steps {
        // SAFETY: s + 4 <= steps by the loop guard, so steps s .. s + 3 are
        // whole: each lane's word in them is below 32 * steps <= k/4.
        let ((q0, d0, v0), (q1, d1, v1), (q2, d2, v2), (q3, d3, v3)) = unsafe {
            (
                q8_0_step(qs, d, xq, wq, wd, lane, s),
                q8_0_step(qs, d, xq, wq, wd, lane, s + 1),
                q8_0_step(qs, d, xq, wq, wd, lane, s + 2),
                q8_0_step(qs, d, xq, wq, wd, lane, s + 3),
            )
        };
        f = q8_0_word_dot(f, q0, d0, v0);
        f = q8_0_word_dot(f, q1, d1, v1);
        f = q8_0_word_dot(f, q2, d2, v2);
        f = q8_0_word_dot(f, q3, d3, v3);
        s += Q8_STEP_UNROLL;
    }
    if s + 2 <= steps {
        // SAFETY: s + 2 <= steps by the guard: both steps are whole.
        let ((q0, d0, v0), (q1, d1, v1)) = unsafe {
            (
                q8_0_step(qs, d, xq, wq, wd, lane, s),
                q8_0_step(qs, d, xq, wq, wd, lane, s + 1),
            )
        };
        f = q8_0_word_dot(f, q0, d0, v0);
        f = q8_0_word_dot(f, q1, d1, v1);
        s += 2;
    }
    if s < steps {
        // SAFETY: s < steps by the guard: the step is whole.
        let (q0, d0, v0) = unsafe { q8_0_step(qs, d, xq, wq, wd, lane, s) };
        f = q8_0_word_dot(f, q0, d0, v0);
    }
    if 32 * steps + lane < words {
        // SAFETY: the guard is the step helper's bound for step `steps`.
        let (q0, d0, v0) = unsafe { q8_0_step(qs, d, xq, wq, wd, lane, steps) };
        f = q8_0_word_dot(f, q0, d0, v0);
    }
    f
}

/// The row's m sums from the per-lane partials: the fixed five-step
/// butterfly per column (`warp::reduce_sum_f32`), columns past `m_cols`
/// left at 0.0. `m_cols` must be warp-uniform — a launch-wide constant in
/// every caller.
#[inline(always)]
pub(crate) fn gemv_lane_sums(f: [f32; 8], m_cols: u32) -> [f32; 8] {
    let s0 = warp::reduce_sum_f32(f[0]);
    if m_cols == 1 {
        return [s0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    }
    let s1 = warp::reduce_sum_f32(f[1]);
    let s2 = if m_cols > 2 {
        warp::reduce_sum_f32(f[2])
    } else {
        0.0
    };
    let s3 = if m_cols > 3 {
        warp::reduce_sum_f32(f[3])
    } else {
        0.0
    };
    let s4 = if m_cols > 4 {
        warp::reduce_sum_f32(f[4])
    } else {
        0.0
    };
    let s5 = if m_cols > 5 {
        warp::reduce_sum_f32(f[5])
    } else {
        0.0
    };
    let s6 = if m_cols > 6 {
        warp::reduce_sum_f32(f[6])
    } else {
        0.0
    };
    let s7 = if m_cols > 7 {
        warp::reduce_sum_f32(f[7])
    } else {
        0.0
    };
    [s0, s1, s2, s3, s4, s5, s6, s7]
}

// -------------------------------------------------------------- kernels

#[cuda_module]
mod q8f32_kernels {
    use super::*;

    /// F32 gemv, `y[r·m + c] = Σ_k w[r·k + k'] · x[c·k + k']`, M <= 8: one
    /// warp per row, 8 rows per 256-thread block, the K-quant gemvs'
    /// skeleton. The row guard is warp-uniform, so the cores' butterfly
    /// always sees a full warp. Summation order: the module doc's contract.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_rows * k,
            x.len() >= m_cols * k,
            y.len() >= n_rows * m_cols
        )
    )]
    pub fn f32_gemv(
        w: &[f32],
        x: &[f32],
        n_rows: u32,
        k: u32,
        m_cols: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let sums = gemv_lane_sums(f32_lane_partials(w, x, k, row, m_cols, lane), m_cols);
        let m = m_cols as usize;
        if lane == 0 {
            // SAFETY: only lane 0 of the warp owning `row` writes, exactly
            // the m live slots of y[row*m .. row*m + m]; y.len() >=
            // n_rows * m_cols by the launch contract and the guards bound
            // every store by m.
            unsafe {
                let b = row * m;
                *y.get_unchecked_mut(b) = sums[0];
                if m > 1 {
                    *y.get_unchecked_mut(b + 1) = sums[1];
                }
                if m > 2 {
                    *y.get_unchecked_mut(b + 2) = sums[2];
                }
                if m > 3 {
                    *y.get_unchecked_mut(b + 3) = sums[3];
                }
                if m > 4 {
                    *y.get_unchecked_mut(b + 4) = sums[4];
                }
                if m > 5 {
                    *y.get_unchecked_mut(b + 5) = sums[5];
                }
                if m > 6 {
                    *y.get_unchecked_mut(b + 6) = sums[6];
                }
                if m > 7 {
                    *y.get_unchecked_mut(b + 7) = sums[7];
                }
            }
        }
    }

    /// Q8_0 gemv against f32 activations, M <= 8: same skeleton and output
    /// layout as `f32_gemv`, weights decoded per the module doc's device
    /// layout. `k` a multiple of 32 (host-validated; the contract binds the
    /// word and scale buffers through `k`: 4·qs.len() and 32·d.len() cover
    /// `n_rows * k` values exactly when 32 | k).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * qs.len() >= n_rows * k,
            32 * d.len() >= n_rows * k,
            x.len() >= m_cols * k,
            y.len() >= n_rows * m_cols
        )
    )]
    pub fn q8_0_gemv(
        qs: &[u32],
        d: &[u16],
        x: &[f32],
        n_rows: u32,
        k: u32,
        m_cols: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let sums = gemv_lane_sums(
            q8_0_lane_partials(qs, d, x, k, row, 0, m_cols, lane),
            m_cols,
        );
        let m = m_cols as usize;
        if lane == 0 {
            // SAFETY: row < n_rows, so the m = m_cols slots written below lie
            // in row*m_cols .. (row+1)*m_cols <= n_rows*m_cols <= y.len(),
            // the launch contract's bound; only lane 0 of the row's warp
            // writes them.
            unsafe {
                let b = row * m;
                *y.get_unchecked_mut(b) = sums[0];
                if m > 1 {
                    *y.get_unchecked_mut(b + 1) = sums[1];
                }
                if m > 2 {
                    *y.get_unchecked_mut(b + 2) = sums[2];
                }
                if m > 3 {
                    *y.get_unchecked_mut(b + 3) = sums[3];
                }
                if m > 4 {
                    *y.get_unchecked_mut(b + 4) = sums[4];
                }
                if m > 5 {
                    *y.get_unchecked_mut(b + 5) = sums[5];
                }
                if m > 6 {
                    *y.get_unchecked_mut(b + 6) = sums[6];
                }
                if m > 7 {
                    *y.get_unchecked_mut(b + 7) = sums[7];
                }
            }
        }
    }
}

/// The loaded P3 device module: `f32_gemv` and `q8_0_gemv`. Owns no context
/// and no stream — the caller passes the engine stream (`Gpu::stream()`) per
/// enqueue, so launches order with the rest of the step and are capturable.
pub struct Q8F32Kernels {
    module: q8f32_kernels::LoadedModule,
}

impl Q8F32Kernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Q8F32Kernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { q8f32_kernels::load(ctx)? };
        Ok(Q8F32Kernels { module })
    }

    /// Enqueue `y = W · x` for an F32 weight of `w.rows()` rows × `w.cols()`
    /// (= k) values, against `m` f32 activation columns of k values each.
    /// `y` holds `rows * m` f32, row-major with m outputs per row.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_f32_gemv(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (n_rows, k) = (w.rows(), w.cols());
        check_gemv_geometry("enqueue_f32_gemv", n_rows, k, x.len(), m, y.len())?;
        let what = "enqueue_f32_gemv";
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let k = launch_u32(what, "k", k)?;
        let m = launch_u32(what, "m", m)?;
        let prep = self
            .module
            .prepare_f32_gemv(LaunchConfig1D::new(n_rows.div_ceil(8), 256, 0))?;
        self.module
            .f32_gemv(stream, &prep, w.buf(), x, n_rows, k, m, y)?;
        Ok(())
    }

    /// Enqueue `y = W · x` for a Q8_0 weight in the module doc's device
    /// layout: `qs` `rows × k/4` u32 words and `d` `rows × k/32` f16 scale
    /// bits — against `m` f32 activation columns of `k = d.cols() * 32` values
    /// each (the scale buffer's width fixes k; `qs.cols()` must equal
    /// `d.cols() * 8`). Output layout as `enqueue_f32_gemv`. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_q8_0_gemv(
        &self,
        stream: &CudaStream,
        qs: &DeviceTensor<u32>,
        d: &DeviceTensor<u16>,
        x: &DeviceBuffer<f32>,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let n_rows = d.rows();
        let k = d.cols() * 32;
        check_gemv_geometry("enqueue_q8_0_gemv", n_rows, k, x.len(), m, y.len())?;
        if qs.rows() != n_rows || qs.cols() != d.cols() * 8 {
            return Err(GpuError::shape(
                "enqueue_q8_0_gemv",
                format!(
                    "qs is {}x{}, want {}x{} (k/4 words per row, k = d.cols()*32 = {k})",
                    qs.rows(),
                    qs.cols(),
                    n_rows,
                    d.cols() * 8
                ),
            ));
        }
        let what = "enqueue_q8_0_gemv";
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let k = launch_u32(what, "k", k)?;
        let m = launch_u32(what, "m", m)?;
        let prep =
            self.module
                .prepare_q8_0_gemv(LaunchConfig1D::new(n_rows.div_ceil(8), 256, 0))?;
        self.module
            .q8_0_gemv(stream, &prep, qs.buf(), d.buf(), x, n_rows, k, m, y)?;
        Ok(())
    }
}

/// Reject geometry the two gemvs' launch contracts do not cover: both walk
/// the row in 32-value chunks, so k must be a positive multiple of 32, rows
/// >= 1, and both support 1..=8 columns.
fn check_gemv_geometry(
    what: &'static str,
    n_rows: usize,
    k: usize,
    x_len: usize,
    m: usize,
    y_len: usize,
) -> Result<(), GpuError> {
    if n_rows == 0 || k == 0 || !k.is_multiple_of(32) {
        return Err(GpuError::shape(
            what,
            format!("need n_rows >= 1 and k a positive multiple of 32, got n_rows={n_rows} k={k}"),
        ));
    }
    if !(1..=8).contains(&m) {
        return Err(GpuError::shape(
            what,
            format!("need 1 <= m <= 8, got m={m}"),
        ));
    }
    if x_len < m * k {
        return Err(GpuError::shape(
            what,
            format!("x.len() {x_len} < m*k = {}", m * k),
        ));
    }
    if y_len < n_rows * m {
        return Err(GpuError::shape(
            what,
            format!("y.len() {y_len} < n_rows*m = {}", n_rows * m),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::q8_0_lane_partial_1col;

    /// The single-column Q8_0 body sums every value of the row exactly once,
    /// for every k the launchers accept (a positive multiple of 32) up to two
    /// hoisted trips — every combination of trips, pair, single and partial
    /// word, and the one-word rows of at most 128 values — on the quad path
    /// and on the unaligned scalar path. The codes, scales (f16 bits of 1, 2
    /// and 3) and activations are small integers, so every lane sum and the
    /// total are exact in f32 whatever the order: a skipped or doubled value
    /// shows as a different integer.
    #[test]
    fn q8_0_1col_sums_each_value_once() {
        let code = |r: usize, v: usize| ((r * 131 + v * 37) % 255) as i32 - 127;
        // Block b's scale, and its f16 bits: 1.0, 2.0 and 3.0.
        let scale = |b: usize| (b % 3 + 1) as i64;
        const F16_1_2_3: [u16; 3] = [0x3c00, 0x4000, 0x4200];
        for k in (32..=1024).step_by(32) {
            let rows = 3;
            let words = k / 4;
            let mut qs = vec![0u32; rows * words];
            for r in 0..rows {
                for v in 0..k {
                    qs[r * words + v / 4] |= u32::from(code(r, v) as u8) << (8 * (v % 4));
                }
            }
            let d: Vec<u16> = (0..rows * k / 32).map(|b| F16_1_2_3[b % 3]).collect();
            let x: Vec<f32> = (0..k + 8).map(|i| ((i * 7) % 17) as f32 - 8.0).collect();
            // The first element offset at which x is 16-byte aligned takes the
            // quad path; one past it, the scalar path.
            let aligned = (16 - x.as_ptr() as usize % 16) % 16 / 4;
            for x0 in [aligned, aligned + 1] {
                for r in 0..rows {
                    let want: i64 = (0..k)
                        .map(|v| {
                            i64::from(code(r, v)) * scale(r * k / 32 + v / 32) * x[x0 + v] as i64
                        })
                        .sum();
                    let got: f32 = (0..32)
                        .map(|lane| {
                            // SAFETY: qs and d hold `rows` whole rows of k
                            // values, x holds k + 8 >= x0 + k (x0 <= 4), k is
                            // a multiple of 32 and lane < 32: the body's
                            // contract for r < rows.
                            unsafe { q8_0_lane_partial_1col(&qs, &d, &x, k as u32, r, x0, lane) }
                        })
                        .sum();
                    assert_eq!(got, want as f32, "k={k} row={r} x0={x0}");
                }
            }
        }
    }
}
