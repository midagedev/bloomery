//! Hyper-connections: the four residual streams around every sub-layer.
//! HC_PRE mixes them into the sub-layer's input and derives `pre`, `post` and
//! the Sinkhorn-normalized `comb` from a 24-value head gemv; HC_POST adds the
//! output back into the streams; the stream fold collapses them before the head.
//!
//! Layouts, every entry: token-major. A token's streams are `4 * n` f32,
//! stream `i` at `i * n` (ggml's `[n, 4, T]`); read flat they are the `K = 4n`
//! values the head gemv takes. A token's HC_PRE result is [`HC_MIX`] f32 —
//! `pre[4]`, `post[4]`, then `comb[16]` row-major (`comb[r * 4 + c]`, ik's
//! `m[r*S + c]`).
//!
//! The numeric rule each entry holds, and the gate transcribes on the host:
//!
//! `ds41_hc_pre` — RMS + split-K gemv + HC_PRE in one launch, one block per
//! [`HC_PIECE`] values of K (one q3_K row-walk iteration: super-blocks `2p`,
//! `2p + 1` of every weight row).
//! 1. The raw stream values are quantized to q8_1 per 128-value block by the
//!    rule of `bloomery_gpu`'s `q3k_quantize_q8_1` (d = amax/127, bytes by
//!    `cores::q8_quad`), in the lanes of `cores::q3k_row_dot` that consume
//!    them. The RMS scale multiplies the gemv result instead of the input:
//!    a q8 block's relative error does not depend on a common factor, and
//!    the gemv then waits for no reduction.
//! 2. Per (row, piece, token), [`piece_sum`]: each lane's `cores::q3k_chain`
//!    summed in integers over the eight lanes of its q8 block, the group sum
//!    `g` scaled by `dd = d8 * d_sb`, then across the four groups `s =
//!    fma(g, dd, g' * dd')` with the lane-8 partner and `s + s''` with the
//!    lane-16 partner. The fused multiply-add is written out because the
//!    device compiler fuses a product that feeds an add on its own (cuda-oxide
//!    defaults to nvcc's `--fmad=true`); written out, the order is pinned.
//! 3. Per (piece, token): each lane's 16 values squared by `v.mul_add(v, acc)`
//!    from 0 in (field, byte) order, then `warp::reduce_sum_f32` (lane `l`
//!    adds lane `l ^ s` for s = 16, 8, 4, 2, 1).
//! 4. The last block to finish sums, per (row, token), the piece partials in
//!    ascending piece order starting from piece 0's value, and the squares the
//!    same way; `scale = 1 / sqrt(squares / K + rms_eps)`, `mix = raw * scale`.
//! 5. HC_PRE per token on one warp ([`hc_pre_lane`]).
//!
//! `ds41_hc_post` — HC_POST and the next sub-layer's input fold per value
//! `d` of token `t`, with the fused multiply-adds where ik's build puts them:
//! `o_i = fma(x, post_i, comb[0][i] * r_0)`, then `o_i = fma(comb[j][i], r_j,
//! o_i)` for j = 1..3; the fold `y = o_0 * pre_0`, then `y = fma(o_j, pre_j,
//! y)`. The fold's weights are the same HC_PRE's `pre` (the lag: a sub-layer's
//! input is folded by the previous sub-layer's HC_PRE). After the last layer
//! the fold of the last token is the head's input.
//!
//! `ds41_hc_fold` — the fold alone.

use bloomery_gpu::cores::{q3k_chain, q3k_sb_decode, q8_quad};
use bloomery_gpu::{DeviceTensor, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, threadfence, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Residual streams (the file's `hyper_connection.count`).
pub const HC_STREAMS: usize = 4;
/// HC_PRE values per token: `pre`, `post`, `comb`.
pub const HC_MIX: usize = 24;
const _: () = assert!(HC_MIX == 2 * HC_STREAMS + HC_STREAMS * HC_STREAMS);
/// Values of K one `ds41_hc_pre` block owns: one q3_K row-walk iteration.
pub const HC_PIECE: usize = 512;
/// Threads of a `ds41_hc_pre` block.
pub(crate) const HC_PRE_THREADS: usize = 256;
const HC_PRE_THREADS_U32: u32 = HC_PRE_THREADS as u32;
const _: () = assert!(HC_PRE_THREADS_U32 as usize == HC_PRE_THREADS);
const HC_PRE_WARPS: usize = HC_PRE_THREADS / 32;
/// Weight rows each warp of a piece block dots: warp `w` owns rows `w`,
/// `w + 8`, `w + 16`.
const HC_ROWS_PER_WARP: usize = HC_MIX / HC_PRE_WARPS;
const _: () = assert!(HC_ROWS_PER_WARP * HC_PRE_WARPS == HC_MIX && HC_ROWS_PER_WARP == 3);
/// Tokens one `ds41_hc_pre` launch takes: its last block gives each token a
/// warp for HC_PRE and a thread per partial sum.
pub const HC_MAX_TOKENS: usize = 8;
const _: () = assert!(HC_MAX_TOKENS <= HC_PRE_WARPS);
const _: () = assert!((HC_MIX + 1) * HC_MAX_TOKENS <= HC_PRE_THREADS);
/// The finishing block's shared mix slots: one per (token, row).
const HC_MIX_SLOTS: usize = HC_MIX * HC_MAX_TOKENS;
/// Threads of the element-wise entries.
pub(crate) const HC_ELEM_THREADS: usize = 256;
const HC_ELEM_THREADS_U32: u32 = HC_ELEM_THREADS as u32;
const _: () = assert!(HC_ELEM_THREADS_U32 as usize == HC_ELEM_THREADS);

// ------------------------------------------------------------------ cores

/// The sum `start + v[a] + v[b] + v[c] + v[d]` in that order, the four
/// values read from lanes `a..d` of the warp. Every lane must call it.
#[inline(always)]
fn lane_sum4(v: f32, lanes: [u32; 4], start: f32) -> f32 {
    let x0 = warp::shuffle_f32(v, lanes[0]);
    let x1 = warp::shuffle_f32(v, lanes[1]);
    let x2 = warp::shuffle_f32(v, lanes[2]);
    let x3 = warp::shuffle_f32(v, lanes[3]);
    (((start + x0) + x1) + x2) + x3
}

/// HC_PRE of one token on one warp, ik's `ggml_compute_forward_hc_pre_f32`
/// op for op with `exp` computed in f64 and rounded to f32 (bit-identical to
/// glibc's `expf` wherever that is correctly rounded). Lane `l < 24` enters
/// with mix value `l` and its `base[l]` and leaves with output value `l` of
/// the [`HC_MIX`] layout; lanes 24..31 enter with zeros and leave with an
/// unspecified value. Lanes 8..23 own comb entries 0..15; every other lane
/// reads a real row and column and carries a placeholder through the
/// normalizations, so no lane divides by a value outside the rule's range.
/// Every lane of the warp must call it: the sums are warp shuffles.
#[inline(always)]
pub(crate) fn hc_pre_lane(
    mix: f32,
    lane: u32,
    sc: [f32; 3],
    base: f32,
    eps: f32,
    iters: u32,
) -> f32 {
    let s = if lane < 4 {
        sc[0]
    } else if lane < 8 {
        sc[1]
    } else {
        sc[2]
    };
    let v = mix.mul_add(s, base);
    let comb = (8..24).contains(&lane);
    let k = lane.wrapping_sub(8) & 15;
    let (r, c) = (k >> 2, k & 3);
    let row = [8 + 4 * r, 9 + 4 * r, 10 + 4 * r, 11 + 4 * r];
    let col = [8 + c, 12 + c, 16 + c, 20 + c];

    // The row max as ik's MAX fold: `a > b ? a : b`, starting at column 0.
    let x0 = warp::shuffle_f32(v, row[0]);
    let x1 = warp::shuffle_f32(v, row[1]);
    let x2 = warp::shuffle_f32(v, row[2]);
    let x3 = warp::shuffle_f32(v, row[3]);
    let mut mx = x0;
    mx = if mx > x1 { mx } else { x1 };
    mx = if mx > x2 { mx } else { x2 };
    mx = if mx > x3 { mx } else { x3 };

    // One exp per lane: the sigmoid argument on lanes 0..7, the softmax one
    // on the comb lanes.
    let arg = if lane < 8 {
        -v
    } else if comb {
        v - mx
    } else {
        0.0
    };
    let e = f64::from(arg).exp() as f32;
    let sig = 1.0 / (1.0 + e);
    let head = if lane < 4 { sig + eps } else { 2.0 * sig };

    // Softmax over the row (the sum from 0 in column order), then eps.
    let ec = if comb { e } else { 1.0 };
    let mut m = ec / lane_sum4(ec, row, 0.0) + eps;
    // One column normalization, then (iters - 1) row-column pairs; every
    // sum starts at eps and adds in index order.
    m /= lane_sum4(m, col, eps);
    let mut it = 1u32;
    while it < iters {
        m /= lane_sum4(m, row, eps);
        m /= lane_sum4(m, col, eps);
        it += 1;
    }
    if lane < 8 { head } else { m }
}

/// HC_POST of one value `d` of one token: the four new streams at `d` from
/// the sub-layer output `x`, the four old streams `res`, and the token's
/// `post` and `comb`, with the multiply-adds fused where ik's build fuses
/// them.
#[inline(always)]
pub(crate) fn hc_post_elem(x: f32, res: [f32; 4], post: [f32; 4], comb: &[f32; 16]) -> [f32; 4] {
    [
        hc_post_stream(x, res, post[0], [comb[0], comb[4], comb[8], comb[12]]),
        hc_post_stream(x, res, post[1], [comb[1], comb[5], comb[9], comb[13]]),
        hc_post_stream(x, res, post[2], [comb[2], comb[6], comb[10], comb[14]]),
        hc_post_stream(x, res, post[3], [comb[3], comb[7], comb[11], comb[15]]),
    ]
}

/// New stream `i` at one value: `x * post_i` plus `comb[j][i] * res_j` over
/// the old streams `j`, `cj` holding `comb[0..4][i]`.
#[inline(always)]
fn hc_post_stream(x: f32, res: [f32; 4], post_i: f32, cj: [f32; 4]) -> f32 {
    let mut s = x.mul_add(post_i, cj[0] * res[0]);
    s = cj[1].mul_add(res[1], s);
    s = cj[2].mul_add(res[2], s);
    cj[3].mul_add(res[3], s)
}

/// One weight row's sum over a piece (the module doc's step 2) from each
/// lane's integer term `a`: exact within the eight lanes that share a q8
/// block and a super-block (lane bits 0..2), then in f32 across the four
/// groups. Lane 0 holds the sum. Every lane of the warp must call it.
#[inline(always)]
fn piece_sum(a: i32, dd: f32) -> f32 {
    let mut g = a;
    g += warp::shuffle_xor(g as u32, 1) as i32;
    g += warp::shuffle_xor(g as u32, 2) as i32;
    g += warp::shuffle_xor(g as u32, 4) as i32;
    // |g| <= 8 lanes * 16 values * 4 * 127 * 32 < 2^24: exact in f32.
    let own = g as f32;
    let partner = warp::shuffle_xor_f32(own * dd, 8);
    let s = own.mul_add(dd, partner);
    s + warp::shuffle_xor_f32(s, 16)
}

/// The largest magnitude of 16 values (`f32::max`, as the q8_1 quantizer's
/// amax: order-free for finite values).
#[inline(always)]
fn abs_max16(v: &[f32; 16]) -> f32 {
    let a = v[0].abs().max(v[1].abs()).max(v[2].abs()).max(v[3].abs());
    let b = v[4].abs().max(v[5].abs()).max(v[6].abs()).max(v[7].abs());
    let c = v[8].abs().max(v[9].abs()).max(v[10].abs()).max(v[11].abs());
    let d = v[12]
        .abs()
        .max(v[13].abs())
        .max(v[14].abs())
        .max(v[15].abs());
    a.max(b).max(c).max(d)
}

/// The sum of squares of 16 values in index order, one fused multiply-add
/// per value from 0.
#[inline(always)]
fn squares16(v: &[f32; 16]) -> f32 {
    let mut a = v[0].mul_add(v[0], 0.0);
    a = v[1].mul_add(v[1], a);
    a = v[2].mul_add(v[2], a);
    a = v[3].mul_add(v[3], a);
    a = v[4].mul_add(v[4], a);
    a = v[5].mul_add(v[5], a);
    a = v[6].mul_add(v[6], a);
    a = v[7].mul_add(v[7], a);
    a = v[8].mul_add(v[8], a);
    a = v[9].mul_add(v[9], a);
    a = v[10].mul_add(v[10], a);
    a = v[11].mul_add(v[11], a);
    a = v[12].mul_add(v[12], a);
    a = v[13].mul_add(v[13], a);
    a = v[14].mul_add(v[14], a);
    v[15].mul_add(v[15], a)
}

/// The fold of four stream values by `pre`: `o_0 * pre_0`, then one fused
/// multiply-add per further stream.
#[inline(always)]
pub(crate) fn hc_fold_elem(o: [f32; 4], pre: [f32; 4]) -> f32 {
    let mut y = o[0] * pre[0];
    y = o[1].mul_add(pre[1], y);
    y = o[2].mul_add(pre[2], y);
    o[3].mul_add(pre[3], y)
}

/// HC_POST at value `i = t * n + d`: the four new stream values at `d` of
/// token `t` from the sub-layer output `x`, the old streams `res` and the
/// HC_PRE results `hc`, and token `t`'s `pre`.
///
/// SAFETY: `i < n * m` for some `m` with `x.len() >= n * m`, `res.len() >=
/// 4 * n * m` and `hc.len() >= 24 * m`.
#[inline(always)]
unsafe fn hc_post_at(
    x: &[f32],
    res: &[f32],
    hc: &[f32],
    n: usize,
    i: usize,
) -> ([f32; 4], [f32; 4]) {
    let (t, d) = (i / n, i % n);
    let (h, s) = (t * HC_MIX, 4 * t * n + d);
    // SAFETY: h + 23 < 24(t + 1) <= 24m <= hc.len() by this fn's contract.
    let (pre, post, comb) = unsafe {
        (
            [
                *hc.get_unchecked(h),
                *hc.get_unchecked(h + 1),
                *hc.get_unchecked(h + 2),
                *hc.get_unchecked(h + 3),
            ],
            [
                *hc.get_unchecked(h + 4),
                *hc.get_unchecked(h + 5),
                *hc.get_unchecked(h + 6),
                *hc.get_unchecked(h + 7),
            ],
            [
                *hc.get_unchecked(h + 8),
                *hc.get_unchecked(h + 9),
                *hc.get_unchecked(h + 10),
                *hc.get_unchecked(h + 11),
                *hc.get_unchecked(h + 12),
                *hc.get_unchecked(h + 13),
                *hc.get_unchecked(h + 14),
                *hc.get_unchecked(h + 15),
                *hc.get_unchecked(h + 16),
                *hc.get_unchecked(h + 17),
                *hc.get_unchecked(h + 18),
                *hc.get_unchecked(h + 19),
                *hc.get_unchecked(h + 20),
                *hc.get_unchecked(h + 21),
                *hc.get_unchecked(h + 22),
                *hc.get_unchecked(h + 23),
            ],
        )
    };
    // SAFETY: i < n*m <= x.len(), and s + 3n < 4n(t + 1) <= res.len() by
    // this fn's contract.
    let (xv, r) = unsafe {
        (
            *x.get_unchecked(i),
            [
                *res.get_unchecked(s),
                *res.get_unchecked(s + n),
                *res.get_unchecked(s + 2 * n),
                *res.get_unchecked(s + 3 * n),
            ],
        )
    };
    (hc_post_elem(xv, r, post, &comb), pre)
}

/// Store the four stream values `o` of value `i = t * n + d` at `d` of
/// token `t`'s four streams.
///
/// SAFETY: `i < n * m` with `out.len() >= 4 * n * m`, and no other thread
/// stores value `i`.
#[inline(always)]
unsafe fn store_streams(out: &mut DisjointSlice<f32>, n: usize, i: usize, o: [f32; 4]) {
    let s = 4 * (i / n) * n + i % n;
    // SAFETY: s + 3n < 4n(t + 1) <= 4n*m <= out.len(); the four slots are
    // value i's alone by this fn's contract.
    unsafe {
        *out.get_unchecked_mut(s) = o[0];
        *out.get_unchecked_mut(s + n) = o[1];
        *out.get_unchecked_mut(s + 2 * n) = o[2];
        *out.get_unchecked_mut(s + 3 * n) = o[3];
    }
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod hc_kernels {
    use super::*;

    /// RMS + split-K q3_K gemv + HC_PRE for `m` tokens (the module doc's
    /// rule, steps 1-5). Block `p` owns values `512p .. 512p + 512` of every
    /// token; warp `w` dots weight rows `w`, `w + 8`, `w + 16` against them
    /// and warp 0 also sums their squares. The block that takes the last
    /// ticket of `ctr` finishes: it sums the partials of every block, scales,
    /// runs HC_PRE with one warp per token, and puts the ticket back to 0 for
    /// the next launch. `ctr` must hold 0 when the launch starts (zeroed at
    /// allocation, and every launch that completes leaves it 0). The launch
    /// has exactly `n_pieces` blocks.
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
            4 * w.len() >= 5280 * n_pieces,
            x.len() >= 512 * n_pieces * m,
            scale.len() >= 3,
            base.len() >= 24,
            part.len() >= 24 * n_pieces * m,
            ss.len() >= n_pieces * m,
            ctr.len() >= 1,
            mixes.len() >= 24 * m,
            hc.len() >= 24 * m,
            m >= 1,
            m <= 8,
            n_pieces >= 1
        )
    )]
    pub fn ds41_hc_pre(
        w: &[u32],
        x: &[f32],
        scale: &[f32],
        base: &[f32],
        n_pieces: u32,
        m: u32,
        rms_eps: f32,
        hc_eps: f32,
        iters: u32,
        mut part: DisjointSlice<f32>,
        mut ss: DisjointSlice<f32>,
        mut ctr: DisjointSlice<u32>,
        mut mixes: DisjointSlice<f32>,
        mut hc: DisjointSlice<f32>,
    ) {
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut MIX: SharedArray<f32, HC_MIX_SLOTS> = SharedArray::UNINIT;
        static mut SCL: SharedArray<f32, HC_MAX_TOKENS> = SharedArray::UNINIT;

        let p = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let wp = tid / 32;
        let (np, mm) = (n_pieces as usize, m as usize);
        let k = HC_PIECE * np;
        let n_sb = 2 * np;

        // The lane geometry of `cores::q3k_row_dot`: lanes 0..15 read
        // super-block 2p, 16..31 super-block 2p + 1; the lane's qs word
        // `w16` covers 16 values of one 128-value q8 block, four quads 32
        // apart, with sub-block scales s0, s0 + 2, s0 + 4, s0 + 6.
        let w16 = lane & 15;
        let sb = 2 * p + (lane >> 4);
        let s0 = 8 * (w16 >> 3) + ((w16 & 7) >> 2);
        let vb = 256 * sb + 128 * (w16 >> 3) + 4 * (w16 & 7);
        let row_bytes = 110 * n_sb;
        // The warp's three rows, decoded once for every token.
        // SAFETY: wp + 16 < 24 and sb < n_sb; `w` holds 24 rows of 110 * n_sb bytes by contract.
        let (d0, d1, d2) = unsafe {
            (
                q3k_sb_decode(w, wp * row_bytes + 110 * sb, w16, s0),
                q3k_sb_decode(w, (wp + 8) * row_bytes + 110 * sb, w16, s0),
                q3k_sb_decode(w, (wp + 16) * row_bytes + 110 * sb, w16, s0),
            )
        };

        let mut t = 0usize;
        while t < mm {
            let xb = t * k + vb;
            // SAFETY: xb + 99 < t*k + 256*(sb + 1) <= (t + 1)*k <= x.len()
            // by the launch contract (sb < 2*n_pieces, t < m); the sixteen
            // reads are this lane's four quads.
            let v = unsafe {
                [
                    *x.get_unchecked(xb),
                    *x.get_unchecked(xb + 1),
                    *x.get_unchecked(xb + 2),
                    *x.get_unchecked(xb + 3),
                    *x.get_unchecked(xb + 32),
                    *x.get_unchecked(xb + 33),
                    *x.get_unchecked(xb + 34),
                    *x.get_unchecked(xb + 35),
                    *x.get_unchecked(xb + 64),
                    *x.get_unchecked(xb + 65),
                    *x.get_unchecked(xb + 66),
                    *x.get_unchecked(xb + 67),
                    *x.get_unchecked(xb + 96),
                    *x.get_unchecked(xb + 97),
                    *x.get_unchecked(xb + 98),
                    *x.get_unchecked(xb + 99),
                ]
            };
            // The block amax: the eight lanes that differ in bits 0..2 hold
            // the block's 128 values.
            let mut am = abs_max16(&v);
            am = am.max(warp::shuffle_xor_f32(am, 1));
            am = am.max(warp::shuffle_xor_f32(am, 2));
            am = am.max(warp::shuffle_xor_f32(am, 4));
            // `q3k_quantize_q8_1`'s block scale.
            let d8 = if am > 0.0 { am / 127.0 } else { 1.0 };
            let (q0, _) = q8_quad([v[0], v[1], v[2], v[3]], d8);
            let (q1, _) = q8_quad([v[4], v[5], v[6], v[7]], d8);
            let (q2, _) = q8_quad([v[8], v[9], v[10], v[11]], d8);
            let (q3, _) = q8_quad([v[12], v[13], v[14], v[15]], d8);
            // The q3 permutation's u64 pairs: fields 0/1, then 2/3.
            let w01 = u64::from(q0) | (u64::from(q1) << 32);
            let w23 = u64::from(q2) | (u64::from(q3) << 32);

            let s_a = piece_sum(q3k_chain(&d0.0, w01, w23, &d0.1), d8 * d0.2);
            let s_b = piece_sum(q3k_chain(&d1.0, w01, w23, &d1.1), d8 * d1.2);
            let s_c = piece_sum(q3k_chain(&d2.0, w01, w23, &d2.1), d8 * d2.2);
            if lane == 0 {
                let pb = (p * mm + t) * HC_MIX + wp;
                // SAFETY: pb + 16 < (p*m + t + 1)*24 <= 24*n_pieces*m <=
                // part.len(); lane 0 of warp wp alone writes rows wp, wp + 8,
                // wp + 16 of (piece p, token t).
                unsafe {
                    *part.get_unchecked_mut(pb) = s_a;
                    *part.get_unchecked_mut(pb + 8) = s_b;
                    *part.get_unchecked_mut(pb + 16) = s_c;
                }
            }
            if wp == 0 {
                let sq = warp::reduce_sum_f32(squares16(&v));
                if lane == 0 {
                    // SAFETY: p*m + t < n_pieces*m <= ss.len(); lane 0 of
                    // warp 0 alone writes (piece p, token t).
                    unsafe {
                        *ss.get_unchecked_mut(p * mm + t) = sq;
                    }
                }
            }
            t += 1;
        }

        // Every thread's stores reach the device before the block takes its
        // ticket; the ticket that completes the count is the last block's.
        threadfence();
        thread::sync_threads();
        // SAFETY: LAST is this block's own shared allocation; the raw form
        // is the only way to reach it without a reference to a `static mut`.
        // Thread 0 writes slot 0 before the barrier that publishes it.
        let last = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LAST) };
        if tid == 0 {
            // SAFETY: `ctr` is a live u32 device buffer of at least one
            // element (launch contract), aligned, and every access to it in
            // this launch is atomic.
            let ticket = unsafe { DeviceAtomicU32::from_ptr(ctr.as_mut_ptr()) };
            let done = ticket.fetch_add(1, AtomicOrdering::AcqRel) + 1 == n_pieces;
            if done {
                ticket.store(0, AtomicOrdering::Relaxed);
            }
            // SAFETY: slot 0 of LAST, written by thread 0 alone.
            unsafe {
                *last = u32::from(done);
            }
        }
        thread::sync_threads();
        // SAFETY: slot 0 was written before the barrier above.
        if unsafe { *last } == 0 {
            return;
        }

        // The finishing block. Threads 0..24m sum one (token, row) partial
        // each, threads 24m..25m one token's squares.
        // SAFETY: MIX and SCL are this block's own shared allocations; each
        // slot below is written by one thread before the barrier that
        // publishes it (24m <= 192, m <= 8 by the contract).
        let (mix_s, scl_s) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut MIX),
                SharedArray::as_raw_mut_ptr(&raw mut SCL),
            )
        };
        if tid < HC_MIX * mm {
            let (tt, r) = (tid / HC_MIX, tid % HC_MIX);
            // SAFETY: (q*m + tt)*24 + r < 24*n_pieces*m <= part.len() for
            // every q < n_pieces; the ticket ordered every block's stores
            // before these reads, and no thread writes `part` again.
            let mut acc = unsafe { *part.get_unchecked_mut(tt * HC_MIX + r) };
            let mut q = 1;
            while q < np {
                // SAFETY: as the read above.
                acc += unsafe { *part.get_unchecked_mut((q * mm + tt) * HC_MIX + r) };
                q += 1;
            }
            // SAFETY: tid < 24m <= 192, this thread's own slot.
            unsafe {
                *mix_s.add(tid) = acc;
            }
        } else if tid < (HC_MIX + 1) * mm {
            let tt = tid - HC_MIX * mm;
            // SAFETY: q*m + tt < n_pieces*m <= ss.len(); ordered as above.
            let mut acc = unsafe { *ss.get_unchecked_mut(tt) };
            let mut q = 1;
            while q < np {
                // SAFETY: as the read above.
                acc += unsafe { *ss.get_unchecked_mut(q * mm + tt) };
                q += 1;
            }
            let mean = acc / k as f32;
            // SAFETY: tt < m <= 8, this thread's own slot.
            unsafe {
                *scl_s.add(tt) = 1.0 / (mean + rms_eps).sqrt();
            }
        }
        thread::sync_threads();
        if wp < mm {
            // SAFETY: scale.len() >= 3 by the launch contract.
            let sc = unsafe {
                [
                    *scale.get_unchecked(0),
                    *scale.get_unchecked(1),
                    *scale.get_unchecked(2),
                ]
            };
            let (mv, bv) = if lane < HC_MIX {
                // SAFETY: wp*24 + lane < 24m, written before the barrier;
                // lane < 24 <= base.len().
                unsafe {
                    (
                        *mix_s.add(wp * HC_MIX + lane) * *scl_s.add(wp),
                        *base.get_unchecked(lane),
                    )
                }
            } else {
                (0.0, 0.0)
            };
            let y = hc_pre_lane(mv, lane as u32, sc, bv, hc_eps, iters);
            if lane < HC_MIX {
                // SAFETY: wp*24 + lane < 24m <= mixes.len(), hc.len(); one
                // lane per slot.
                unsafe {
                    *mixes.get_unchecked_mut(wp * HC_MIX + lane) = mv;
                    *hc.get_unchecked_mut(wp * HC_MIX + lane) = y;
                }
            }
        }
    }

    /// HC_POST and the next input fold, one thread per value `d` of token
    /// `t`: `x` the sub-layer output (`n` per token), `res` the streams it
    /// read (`4n` per token), `hc` the sub-layer's HC_PRE result; writes the
    /// new streams to `out` and their fold by the same `pre` to `fold`.
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
            x.len() >= n * m,
            res.len() >= 4 * n * m,
            hc.len() >= 24 * m,
            out.len() >= 4 * n * m,
            fold.len() >= n * m
        )
    )]
    pub fn ds41_hc_post(
        x: &[f32],
        res: &[f32],
        hc: &[f32],
        n: u32,
        m: u32,
        mut out: DisjointSlice<f32>,
        mut fold: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        // SAFETY: i < n*m, and the launch contract gives the lengths.
        let (o, pre) = unsafe { hc_post_at(x, res, hc, n, i) };
        // SAFETY: i < n*m, out.len() >= 4nm by the launch contract, and the
        // thread owns value i.
        unsafe { store_streams(&mut out, n, i, o) };
        // SAFETY: i < n*m <= fold.len(); one thread per value.
        unsafe {
            *fold.get_unchecked_mut(i) = hc_fold_elem(o, pre);
        }
    }

    /// The fold alone: `y[t][d]` from the four streams of `s` at `d` and the
    /// `pre` of token `t`'s HC_PRE result in `hc`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (s.len() >= 4 * n * m, hc.len() >= 24 * m, y.len() >= n * m)
    )]
    pub fn ds41_hc_fold(s: &[f32], hc: &[f32], n: u32, m: u32, mut y: DisjointSlice<f32>) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        let (t, d) = (i / n, i % n);
        let (h, b) = (t * HC_MIX, 4 * t * n + d);
        // SAFETY: h + 3 < 24m <= hc.len(); b + 3n < 4n(t + 1) <= s.len().
        let (pre, o) = unsafe {
            (
                [
                    *hc.get_unchecked(h),
                    *hc.get_unchecked(h + 1),
                    *hc.get_unchecked(h + 2),
                    *hc.get_unchecked(h + 3),
                ],
                [
                    *s.get_unchecked(b),
                    *s.get_unchecked(b + n),
                    *s.get_unchecked(b + 2 * n),
                    *s.get_unchecked(b + 3 * n),
                ],
            )
        };
        // SAFETY: i < n*m <= y.len(); one thread per value.
        unsafe {
            *y.get_unchecked_mut(i) = hc_fold_elem(o, pre);
        }
    }
}

// ---------------------------------------------------------------- host

/// Scratch of one `ds41_hc_pre` site: the per-block partial sums and the
/// ticket counter. Launches on one stream never overlap, so one scratch
/// serves every site of a step; the counter is zero at allocation and every
/// completed launch leaves it zero.
pub struct HcPreScratch {
    part: DeviceBuffer<f32>,
    ss: DeviceBuffer<f32>,
    ctr: DeviceBuffer<u32>,
    n_pieces: usize,
}

impl HcPreScratch {
    /// Scratch for inputs of `k` values per token (a positive multiple of
    /// [`HC_PIECE`]) and up to [`HC_MAX_TOKENS`] tokens a launch. Load-time
    /// only.
    pub fn new(stream: &CudaStream, k: usize) -> Result<HcPreScratch, GpuError> {
        if k == 0 || !k.is_multiple_of(HC_PIECE) {
            return Err(GpuError::Shape {
                what: "HcPreScratch::new",
                detail: format!("k must be a positive multiple of {HC_PIECE}, got {k}"),
            });
        }
        let n_pieces = k / HC_PIECE;
        Ok(HcPreScratch {
            part: DeviceBuffer::zeroed(stream, HC_MIX * n_pieces * HC_MAX_TOKENS)?,
            ss: DeviceBuffer::zeroed(stream, n_pieces * HC_MAX_TOKENS)?,
            ctr: DeviceBuffer::zeroed(stream, 1)?,
            n_pieces,
        })
    }

    /// Values per token this scratch serves.
    #[must_use]
    pub fn k(&self) -> usize {
        self.n_pieces * HC_PIECE
    }
}

/// One sub-layer's HC_PRE parameters as the file holds them.
pub struct HcParams<'a> {
    /// `hc_{attn,ffn}_fn`: [`HC_MIX`] q3_K rows of K values, `110 * K / 1024`
    /// u32 words each.
    pub w: &'a DeviceTensor<u32>,
    /// `hc_{attn,ffn}_scale`: the three affine scales (pre, post, comb).
    pub scale: &'a DeviceBuffer<f32>,
    /// `hc_{attn,ffn}_base`: the [`HC_MIX`] affine offsets.
    pub base: &'a DeviceBuffer<f32>,
    /// `hyper_connection.epsilon`.
    pub eps: f32,
    /// `hyper_connection.sinkhorn_iterations`.
    pub iters: u32,
}

/// The inputs of one `ds41_hc_pre` launch.
pub struct HcPreArgs<'a> {
    /// The sub-layer's HC_PRE parameters.
    pub params: &'a HcParams<'a>,
    /// The streams the sub-layer reads: `m` tokens of the scratch's K values.
    pub x: &'a DeviceBuffer<f32>,
    /// Tokens, `1..=`[`HC_MAX_TOKENS`].
    pub tokens: usize,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
}

/// The inputs of one `ds41_hc_post` launch.
pub struct HcPostArgs<'a> {
    /// The sub-layer output, `n_embd` per token.
    pub x: &'a DeviceBuffer<f32>,
    /// The streams the sub-layer read, `4 * n_embd` per token.
    pub res: &'a DeviceBuffer<f32>,
    /// The sub-layer's HC_PRE result, [`HC_MIX`] per token.
    pub hc: &'a DeviceBuffer<f32>,
    /// Values per stream.
    pub n_embd: usize,
    /// Tokens.
    pub tokens: usize,
}

/// The loaded hyper-connection module. Owns no stream: every enqueue takes
/// the caller's, so launches order with the rest of the step and are
/// capturable.
pub struct HcKernels {
    module: hc_kernels::LoadedModule,
}

impl HcKernels {
    /// Load this module's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<HcKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; each launcher checks its launch contract.
        let module = unsafe { hc_kernels::load(ctx)? };
        Ok(HcKernels { module })
    }

    /// Enqueue RMS + split-K gemv + HC_PRE (`ds41_hc_pre`): `mixes` takes
    /// the scaled mixes and `hc` the HC_PRE result, [`HC_MIX`] per token.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_pre(
        &self,
        stream: &CudaStream,
        a: &HcPreArgs<'_>,
        scratch: &mut HcPreScratch,
        mixes: &mut DeviceBuffer<f32>,
        hc: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "HcKernels::enqueue_pre";
        let (k, m, p) = (scratch.k(), a.tokens, a.params);
        check_params(what, p, k)?;
        if !(1..=HC_MAX_TOKENS).contains(&m)
            || a.x.len() < k * m
            || mixes.len() < HC_MIX * m
            || hc.len() < HC_MIX * m
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "tokens = {m} (1..={HC_MAX_TOKENS}), x.len() {} (need {}), mixes.len() {} / hc.len() {} (need {})",
                    a.x.len(),
                    k * m,
                    mixes.len(),
                    hc.len(),
                    HC_MIX * m
                ),
            });
        }
        let n_pieces = launch_u32(what, "n_pieces", scratch.n_pieces)?;
        let m = launch_u32(what, "tokens", m)?;
        let prep = self.module.prepare_ds41_hc_pre(LaunchConfig1D::new(
            n_pieces,
            HC_PRE_THREADS_U32,
            0,
        ))?;
        self.module.ds41_hc_pre(
            stream,
            &prep,
            p.w.buf(),
            a.x,
            p.scale,
            p.base,
            n_pieces,
            m,
            a.rms_eps,
            p.eps,
            p.iters,
            &mut scratch.part,
            &mut scratch.ss,
            &mut scratch.ctr,
            mixes,
            hc,
        )?;
        Ok(())
    }

    /// Enqueue HC_POST and the next input fold (`ds41_hc_post`): `out` takes
    /// the new streams (`4 * n_embd` per token), `fold` their fold by the
    /// same HC_PRE's `pre` (`n_embd` per token). Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_post(
        &self,
        stream: &CudaStream,
        a: &HcPostArgs<'_>,
        out: &mut DeviceBuffer<f32>,
        fold: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "HcKernels::enqueue_post";
        let (grid, n, m) = post_launch(what, a, out.len(), fold.len())?;
        let prep =
            self.module
                .prepare_ds41_hc_post(LaunchConfig1D::new(grid, HC_ELEM_THREADS_U32, 0))?;
        self.module
            .ds41_hc_post(stream, &prep, a.x, a.res, a.hc, n, m, out, fold)?;
        Ok(())
    }

    /// Enqueue the fold alone (`ds41_hc_fold`): `y` (`n_embd` per token)
    /// from the streams `s` (`4 * n_embd` per token) and the `pre` of `hc`
    /// ([`HC_MIX`] per token). Asynchronous, allocation-free, capturable.
    pub fn enqueue_fold(
        &self,
        stream: &CudaStream,
        s: &DeviceBuffer<f32>,
        hc: &DeviceBuffer<f32>,
        n_embd: usize,
        tokens: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "HcKernels::enqueue_fold";
        let (n, m) = (n_embd, tokens);
        if n == 0
            || m == 0
            || s.len() < HC_STREAMS * n * m
            || hc.len() < HC_MIX * m
            || y.len() < n * m
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "n_embd = {n}, tokens = {m}: s {} hc {} y {}",
                    s.len(),
                    hc.len(),
                    y.len()
                ),
            });
        }
        let grid = launch_u32(what, "grid", (n * m).div_ceil(HC_ELEM_THREADS))?;
        let (n, m) = (
            launch_u32(what, "n_embd", n)?,
            launch_u32(what, "tokens", m)?,
        );
        let prep =
            self.module
                .prepare_ds41_hc_fold(LaunchConfig1D::new(grid, HC_ELEM_THREADS_U32, 0))?;
        self.module.ds41_hc_fold(stream, &prep, s, hc, n, m, y)?;
        Ok(())
    }
}

/// An HC_POST launch's grid, `n_embd` and tokens, after checking `a`, the
/// output streams' length and the fold's.
fn post_launch(
    what: &'static str,
    a: &HcPostArgs<'_>,
    out_len: usize,
    fold_len: usize,
) -> Result<(u32, u32, u32), GpuError> {
    let (n, m) = (a.n_embd, a.tokens);
    if n == 0
        || m == 0
        || a.x.len() < n * m
        || a.res.len() < HC_STREAMS * n * m
        || a.hc.len() < HC_MIX * m
        || out_len < HC_STREAMS * n * m
        || fold_len < n * m
    {
        return Err(GpuError::Shape {
            what,
            detail: format!(
                "n_embd = {n}, tokens = {m}: x {} res {} hc {} out {out_len} fold {fold_len}",
                a.x.len(),
                a.res.len(),
                a.hc.len()
            ),
        });
    }
    Ok((
        launch_u32(what, "grid", (n * m).div_ceil(HC_ELEM_THREADS))?,
        launch_u32(what, "n_embd", n)?,
        launch_u32(what, "tokens", m)?,
    ))
}

/// The site's parameters fit a chain over `k` values: [`HC_MIX`] q3_K rows of
/// `110 * k / 1024` words, three scales, [`HC_MIX`] offsets, at least one
/// Sinkhorn iteration.
fn check_params(what: &'static str, p: &HcParams<'_>, k: usize) -> Result<(), GpuError> {
    let words = 110 * (k / 256) / 4;
    if p.w.rows() != HC_MIX
        || p.w.cols() != words
        || p.scale.len() < 3
        || p.base.len() < HC_MIX
        || p.iters == 0
    {
        return Err(GpuError::Shape {
            what,
            detail: format!(
                "w {}x{} (need {HC_MIX}x{words}), scale.len() {} (need 3), base.len() {} (need {HC_MIX}), iters {}",
                p.w.rows(),
                p.w.cols(),
                p.scale.len(),
                p.base.len(),
                p.iters
            ),
        });
    }
    Ok(())
}
