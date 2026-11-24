//! HC over F32 rows in pieces: the split-K geometry of `hc.rs`'s `ds41_hc_pre`
//! and `hc_pre_q8_0` — a block a [`HC_PIECE`]-value piece of K, the last block of a
//! token group finishing the group's tokens — for the F32 `hc_fn` a one-block
//! launch walks too slowly: a target whose every site's `mapping_proj` is F32
//! `[K, 24]` at K of four streams of a few thousand values. The layouts are
//! [`crate::hc`]'s: token-major streams, `K = 4n` f32 a token read flat; a token's
//! result [`HC_MIX`] f32 (`pre[4]`, `post[4]`, `comb[16]`), its scaled mixes as many.
//!
//! `hc_pre_f32_rmsadd` — [`HcF32PiecesKernels::enqueue_pre`]: the F32 gemv over
//! pieces, then HC_PRE, with the RMS factor the ERNIE graph puts on it (llama.cpp
//! `build_hc_pre_ernie`): the mixes are the raw dots divided by
//! `sqrt(mean(x²)) + rms_eps`, not multiplied by `rsqrt(mean(x²) + rms_eps)` as every
//! entry of `hc.rs` and `hc_f32.rs` has it. So this is a second instance beside
//! `ds41_hc_pre_f32`, never a flag inside it.
//! 1. The activations stay f32: no quantization. Lane `l` of a piece block owns the
//!    values `512p + 4l + 128s .. + 3` (`s` = 0..3) of every weight row and of the
//!    token, read flat.
//! 2. Per (row, piece, token): the lane's `f = fma(w_j, x_j, f)` from 0 in (s, j)
//!    order ([`piece_dot`]), then `warp::reduce_sum_f32` (lane `l` adds lane `l ^ s`
//!    for s = 16, 8, 4, 2, 1). Warp `w` dots rows `w`, `w + 8`, `w + 16`.
//! 3. Per (piece, token): warp 0 squares its sixteen values by `v.mul_add(v, acc)`
//!    from 0 in the same order, then `warp::reduce_sum_f32`.
//! 4. The group's last block sums, per (row, token), the piece partials in ascending
//!    piece order from piece 0's value, the squares the same way;
//!    `denom = sqrt(squares / K) + rms_eps`, `mix = raw / denom`. A `denom` that is
//!    not finite and positive is NaN and raises [`FaultSite::F32Product`] (a token
//!    whose squares overflowed would otherwise read a zero mix, a plausible value).
//! 5. HC_PRE per token on one warp ([`hc_pre_lane`]: `pre = σ(mix·s₀ + base) + eps`,
//!    `post = 2σ(mix·s₁ + base)`, the row softmax and the Sinkhorn rounds) — the
//!    split rule every entry of `hc.rs` holds.
//!
//! `hc_head_f32` — [`HcF32PiecesKernels::enqueue_head`]: the head's four-row mix
//! (`output_hc_fn` `[K, 4]`, `output_hc_scale` `[1]`, `output_hc_base` `[4]`) under
//! the V4 rule (`ggml_rms_norm`'s; llama.cpp `build_hc_head`): steps 1-4 with one row a warp
//! (`w < 4`) and `scale = 1 / sqrt(squares / K + rms_eps)` ([`rms_scale_checked`]),
//! `mix = raw * scale`; then `pre_i = σ(mix_i·s + base_i) + eps` for the four rows,
//! [`hc_pre_lane`]'s first four lanes, written as lanes 0..3 of a [`HC_MIX`]-wide row
//! with lanes 4..23 zero, so the common `ds41_hc_fold` folds the streams by it.
//!
//! A non-finite stream value reaches every output of its token as NaN and raises
//! `F32Product`, its token's squares leaving the denominator no scale; every other
//! token keeps its bits. One launch serves any count of tokens: token groups as
//! `ds41_hc_pre_groups` cuts them, each with its own partials, squares and ticket.

use crate::elem::rms_scale_checked;
use crate::hc::{
    HC_MAX_TOKENS, HC_MIX, HC_MIX_SLOTS, HC_PIECE, HC_PRE_THREADS, HC_SS_THREAD, HC_STREAMS,
    HcPreAt, HcPreScratch, HcPreShared, column_sum, hc_pre_lane, squares16,
};
use crate::hc_f32::HcF32Args;
use crate::{ColGroups, FaultSink, FaultSite, GpuError, col_group, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, threadfence, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

// ------------------------------------------------------------------ cores

/// Lane `lane`'s sixteen activations of piece `p` of token `t`: the values
/// `512p + 4·lane + 128s .. + 3` (`s` = 0..3), in that order.
///
/// # Safety
///
/// `(t + 1) * k <= x.len()`, `512 * (p + 1) <= k` and `lane < 32`.
#[inline(always)]
unsafe fn piece_x(x: &[f32], k: usize, t: usize, p: usize, lane: usize) -> [f32; 16] {
    let at = t * k + HC_PIECE * p + 4 * lane;
    // SAFETY: at + 387 <= t·k + 512p + 511 < (t + 1)·k <= x.len() by this fn's
    // contract (p < k / 512, lane < 32); the sixteen reads are the lane's four quads.
    unsafe {
        [
            *x.get_unchecked(at),
            *x.get_unchecked(at + 1),
            *x.get_unchecked(at + 2),
            *x.get_unchecked(at + 3),
            *x.get_unchecked(at + 128),
            *x.get_unchecked(at + 129),
            *x.get_unchecked(at + 130),
            *x.get_unchecked(at + 131),
            *x.get_unchecked(at + 256),
            *x.get_unchecked(at + 257),
            *x.get_unchecked(at + 258),
            *x.get_unchecked(at + 259),
            *x.get_unchecked(at + 384),
            *x.get_unchecked(at + 385),
            *x.get_unchecked(at + 386),
            *x.get_unchecked(at + 387),
        ]
    }
}

/// Lane `lane`'s sixteen values of piece `p` of weight row `row`, the values
/// [`piece_x`] reads of the token, in that order.
///
/// # Safety
///
/// `(row + 1) * k <= w.len()`, `512 * (p + 1) <= k` and `lane < 32`.
#[inline(always)]
unsafe fn piece_w(w: &[f32], k: usize, row: usize, p: usize, lane: usize) -> [f32; 16] {
    let at = row * k + HC_PIECE * p + 4 * lane;
    // SAFETY: at + 387 <= row·k + 512p + 511 < (row + 1)·k <= w.len() by this fn's
    // contract; the sixteen reads are the lane's four quads.
    unsafe {
        [
            *w.get_unchecked(at),
            *w.get_unchecked(at + 1),
            *w.get_unchecked(at + 2),
            *w.get_unchecked(at + 3),
            *w.get_unchecked(at + 128),
            *w.get_unchecked(at + 129),
            *w.get_unchecked(at + 130),
            *w.get_unchecked(at + 131),
            *w.get_unchecked(at + 256),
            *w.get_unchecked(at + 257),
            *w.get_unchecked(at + 258),
            *w.get_unchecked(at + 259),
            *w.get_unchecked(at + 384),
            *w.get_unchecked(at + 385),
            *w.get_unchecked(at + 386),
            *w.get_unchecked(at + 387),
        ]
    }
}

/// One lane's partial dot of a row's piece with the token's: `f = fma(w_j, x_j, f)`
/// from 0, `j` ascending — one fused multiply-add a value, the order pinned.
#[inline(always)]
fn piece_dot(w: &[f32; 16], x: &[f32; 16]) -> f32 {
    let mut f = w[0].mul_add(x[0], 0.0);
    f = w[1].mul_add(x[1], f);
    f = w[2].mul_add(x[2], f);
    f = w[3].mul_add(x[3], f);
    f = w[4].mul_add(x[4], f);
    f = w[5].mul_add(x[5], f);
    f = w[6].mul_add(x[6], f);
    f = w[7].mul_add(x[7], f);
    f = w[8].mul_add(x[8], f);
    f = w[9].mul_add(x[9], f);
    f = w[10].mul_add(x[10], f);
    f = w[11].mul_add(x[11], f);
    f = w[12].mul_add(x[12], f);
    f = w[13].mul_add(x[13], f);
    f = w[14].mul_add(x[14], f);
    w[15].mul_add(x[15], f)
}

/// The RMS factor's divisor of a token, the ERNIE rule: `sqrt(squares / K) + eps`.
/// NaN where that is not finite and positive, and then raised as
/// [`FaultSite::F32Product`] — one thread holds a token, so one raise.
#[inline(always)]
fn rmsadd_denominator(squares: f32, k: u32, eps: f32, fault: FaultSink) -> f32 {
    let denom = (squares / k as f32).sqrt() + eps;
    if denom.is_finite() & (denom > 0.0) {
        denom
    } else {
        fault.raise(FaultSite::F32Product);
        f32::NAN
    }
}

/// The group's ticket: every thread's stores ordered device-wide, thread 0 takes the
/// block's ticket and the block that completes the group's block count puts it back to
/// 0; whether this is that block. Called by every thread of the block.
///
/// # Safety
///
/// `ctr` holds the group's ticket slot at `ticket`, a u32 that is 0 when the launch
/// starts and that every access of this launch makes through this function; `blocks`
/// is the group's block count; `last` addresses one u32 of the block's own shared
/// memory; and this block's stores to the buffers the finishing half reads are made.
#[inline(always)]
unsafe fn last_ticket(
    ctr: &mut DisjointSlice<u32>,
    ticket: usize,
    blocks: u32,
    last: *mut u32,
) -> bool {
    threadfence();
    thread::sync_threads();
    if thread::threadIdx_x() == 0 {
        // SAFETY: `ctr` is a live u32 device buffer holding the group's ticket slot
        // at `ticket` (this fn's contract), aligned, and every access to it in this
        // launch is this function's atomics.
        let ticket = unsafe { DeviceAtomicU32::from_ptr(ctr.as_mut_ptr().add(ticket)) };
        let done = ticket.fetch_add(1, AtomicOrdering::AcqRel) + 1 == blocks;
        if done {
            ticket.store(0, AtomicOrdering::Relaxed);
        }
        // SAFETY: slot 0 of the block's own shared flag, written by thread 0 alone
        // before the barrier that publishes it.
        unsafe {
            *last = u32::from(done);
        }
    }
    thread::sync_threads();
    // SAFETY: slot 0 of the shared flag was written before the barrier above.
    let last = unsafe { *last };
    last != 0
}

/// One piece block of `hc_pre_f32_rmsadd` (the module doc's steps 1-5): piece
/// `at.p` of the `m` tokens from `at.t0`, then — for the block that takes the group's
/// last ticket — the finish of those tokens.
///
/// # Safety
///
/// Called by every thread of a 256-thread block under `hc_pre_f32_rmsadd`'s launch
/// contract, for `m` (1..=8) tokens read from token `at.t0` of `x` and written from
/// `at.t0` of `mixes` and `hc`; the group's partials, squares and ticket at
/// `at.part0`, `at.ss0` and `at.ticket` of `part`, `ss` and `ctr` (`24 * n_pieces * 8`,
/// `n_pieces * 8` and one slot); the group has exactly `n_pieces` blocks, no other
/// group's block touches its scratch or its tokens; and `sh` holds the block's own
/// shared slots.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn pre_block(
    w: &[f32],
    x: &[f32],
    scale: &[f32],
    base: &[f32],
    n_pieces: u32,
    m: u32,
    rms_eps: f32,
    hc_eps: f32,
    iters: u32,
    part: &mut DisjointSlice<f32>,
    ss: &mut DisjointSlice<f32>,
    ctr: &mut DisjointSlice<u32>,
    mixes: &mut DisjointSlice<f32>,
    hc: &mut DisjointSlice<f32>,
    fault: FaultSink,
    at: HcPreAt,
    sh: HcPreShared,
) {
    let p = at.p;
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let wp = tid / 32;
    let (np, mm) = (n_pieces as usize, m as usize);
    let k = HC_PIECE * np;

    // The warp's three rows, dotted against each token's piece once read.
    // SAFETY: wp + 16 < 24 and 512 * (p + 1) <= k; `w` holds 24 rows of k values by
    // this fn's contract; lane < 32.
    let (w0, w1, w2) = unsafe {
        (
            piece_w(w, k, wp, p, lane),
            piece_w(w, k, wp + 8, p, lane),
            piece_w(w, k, wp + 16, p, lane),
        )
    };
    let mut t = 0usize;
    while t < mm {
        // SAFETY: t0 + t < t0 + m <= tokens, so (t0 + t + 1)·k <= tokens·k <= x.len()
        // by this fn's contract; 512 * (p + 1) <= k and lane < 32.
        let v = unsafe { piece_x(x, k, at.t0 + t, p, lane) };
        let s_a = warp::reduce_sum_f32(piece_dot(&w0, &v));
        let s_b = warp::reduce_sum_f32(piece_dot(&w1, &v));
        let s_c = warp::reduce_sum_f32(piece_dot(&w2, &v));
        if lane == 0 {
            let pb = at.part0 + (p * mm + t) * HC_MIX + wp;
            // SAFETY: pb + 16 < (p·m + t + 1)·24 <= 24·n_pieces·m past the group's
            // part0, inside `part`; lane 0 of warp wp alone writes rows wp, wp + 8,
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
                // SAFETY: ss0 + p·m + t < ss0 + n_pieces·m, inside `ss`; lane 0 of
                // warp 0 alone writes (piece p, token t).
                unsafe {
                    *ss.get_unchecked_mut(at.ss0 + p * mm + t) = sq;
                }
            }
        }
        t += 1;
    }

    // SAFETY: this fn's contract is the callee's, and every store above to `part`
    // and `ss` is made.
    if !unsafe { last_ticket(ctr, at.ticket, n_pieces, sh.last) } {
        return;
    }

    // The finishing block. Threads 0..24m sum one (token, row) partial each,
    // threads HC_SS_THREAD..+m one token's squares, each over the pieces in
    // ascending order.
    let (mix_s, den_s) = (sh.mix, sh.scl);
    if tid < HC_MIX * mm {
        // SAFETY: piece q's slot (q·m + t)·24 + r = q·24m + tid < 24·n_pieces·m
        // past the group's part0, inside `part` for every q < n_pieces; the ticket
        // ordered every block's stores before these reads, and no thread writes
        // `part` again.
        let acc = unsafe { column_sum(part, at.part0 + tid, HC_MIX * mm, np) };
        // SAFETY: tid < 24m <= 192, this thread's own slot of the block's shared mixes.
        unsafe {
            *mix_s.add(tid) = acc;
        }
    } else if (HC_SS_THREAD..HC_SS_THREAD + mm).contains(&tid) {
        let tt = tid - HC_SS_THREAD;
        // SAFETY: q·m + tt < n_pieces·m past the group's ss0, inside `ss` for every
        // q < n_pieces; the ticket ordered every block's stores first.
        let acc = unsafe { column_sum(ss, at.ss0 + tt, mm, np) };
        let den = rmsadd_denominator(acc, k as u32, rms_eps, fault);
        // SAFETY: tt < m <= 8, this thread's own slot of the block's shared divisors.
        unsafe {
            *den_s.add(tt) = den;
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
            // SAFETY: wp·24 + lane < 24m, a slot written before the barrier, and
            // wp < m one the divisors hold; lane < 24 <= base.len().
            unsafe {
                (
                    *mix_s.add(wp * HC_MIX + lane) / *den_s.add(wp),
                    *base.get_unchecked(lane),
                )
            }
        } else {
            (0.0, 0.0)
        };
        let y = hc_pre_lane(mv, lane as u32, sc, bv, hc_eps, iters);
        if lane < HC_MIX {
            let o = (at.t0 + wp) * HC_MIX + lane;
            // SAFETY: o < 24·tokens <= mixes.len(), hc.len(); one lane per slot.
            unsafe {
                *mixes.get_unchecked_mut(o) = mv;
                *hc.get_unchecked_mut(o) = y;
            }
        }
    }
}

/// One piece block of `hc_head_f32` (the module doc): the four rows' piece `at.p`
/// of the `m` tokens from `at.t0`, then the finish of those tokens for the block
/// that takes the group's last ticket.
///
/// # Safety
///
/// [`pre_block`]'s contract under `hc_head_f32`'s launch contract, `w` holding four
/// rows of k values and `scale` one.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn head_block(
    w: &[f32],
    x: &[f32],
    scale: &[f32],
    base: &[f32],
    n_pieces: u32,
    m: u32,
    rms_eps: f32,
    hc_eps: f32,
    iters: u32,
    part: &mut DisjointSlice<f32>,
    ss: &mut DisjointSlice<f32>,
    ctr: &mut DisjointSlice<u32>,
    mixes: &mut DisjointSlice<f32>,
    hc: &mut DisjointSlice<f32>,
    fault: FaultSink,
    at: HcPreAt,
    sh: HcPreShared,
) {
    let p = at.p;
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let wp = tid / 32;
    let (np, mm) = (n_pieces as usize, m as usize);
    let k = HC_PIECE * np;

    // Warps 0..3 own a row each, dotted against each token's piece once read; the
    // other four dot nothing.
    let w0 = if wp < HC_STREAMS {
        // SAFETY: under the guard wp < 4, so row wp is one of the four rows `w` holds
        // by this fn's contract; 512 * (p + 1) <= k and lane < 32.
        unsafe { piece_w(w, k, wp, p, lane) }
    } else {
        [0.0; 16]
    };
    let mut t = 0usize;
    while t < mm {
        // SAFETY: t0 + t < t0 + m <= tokens, so (t0 + t + 1)·k <= tokens·k <=
        // x.len() by this fn's contract; 512 * (p + 1) <= k and lane < 32.
        let v = unsafe { piece_x(x, k, at.t0 + t, p, lane) };
        if wp < HC_STREAMS {
            let s = warp::reduce_sum_f32(piece_dot(&w0, &v));
            if lane == 0 {
                let pb = at.part0 + (p * mm + t) * HC_MIX + wp;
                // SAFETY: pb < (p·m + t + 1)·24 <= 24·n_pieces·m past the group's
                // part0, inside `part`; lane 0 of warp wp alone writes row wp of
                // (piece p, token t).
                unsafe {
                    *part.get_unchecked_mut(pb) = s;
                }
            }
        }
        if wp == 0 {
            let sq = warp::reduce_sum_f32(squares16(&v));
            if lane == 0 {
                // SAFETY: ss0 + p·m + t < ss0 + n_pieces·m, inside `ss`; lane 0 of
                // warp 0 alone writes (piece p, token t).
                unsafe {
                    *ss.get_unchecked_mut(at.ss0 + p * mm + t) = sq;
                }
            }
        }
        t += 1;
    }

    // SAFETY: this fn's contract is the callee's, and every store above to `part`
    // and `ss` is made.
    if !unsafe { last_ticket(ctr, at.ticket, n_pieces, sh.last) } {
        return;
    }

    // The finishing block. Thread `4t + r` sums (token t, row r)'s partials — a
    // (piece, token)'s rows 24 apart, the pre's layout, of which four are ever
    // written — threads HC_SS_THREAD..+m one token's squares, each over the pieces
    // in ascending order.
    let (mix_s, scl_s) = (sh.mix, sh.scl);
    if tid < HC_STREAMS * mm {
        let off = at.part0 + (tid / HC_STREAMS) * HC_MIX + tid % HC_STREAMS;
        // SAFETY: piece q's slot off + q·24m = q·24m + 24t + r, below 24·n_pieces·m
        // past the group's part0 for every q < n_pieces, inside `part`; the ticket
        // ordered every block's stores before these reads.
        let acc = unsafe { column_sum(part, off, HC_MIX * mm, np) };
        // SAFETY: tid < 4m <= 32, this thread's own slot of the block's shared mixes.
        unsafe {
            *mix_s.add(tid) = acc;
        }
    } else if (HC_SS_THREAD..HC_SS_THREAD + mm).contains(&tid) {
        let tt = tid - HC_SS_THREAD;
        // SAFETY: q·m + tt < n_pieces·m past the group's ss0, inside `ss` for every
        // q < n_pieces; the ticket ordered every block's stores first.
        let acc = unsafe { column_sum(ss, at.ss0 + tt, mm, np) };
        // One thread a token: a scale with no norm behind it (squares that
        // overflowed, or never finite) is NaN and raises F32Product.
        let scl = rms_scale_checked(acc, k as u32, rms_eps, fault, true);
        // SAFETY: tt < m <= 8, this thread's own slot of the block's shared scales.
        unsafe {
            *scl_s.add(tt) = scl;
        }
    }
    thread::sync_threads();
    if wp < mm {
        // SAFETY: scale.len() >= 1 by the launch contract; lanes past the pre read
        // no scale.
        let sc = unsafe { [*scale.get_unchecked(0), 0.0, 0.0] };
        let (mv, bv) = if lane < HC_STREAMS {
            // SAFETY: wp·4 + lane < 4m, a slot written before the barrier, and wp < m
            // one the scales hold; lane < 4 <= base.len().
            unsafe {
                (
                    *mix_s.add(wp * HC_STREAMS + lane) * *scl_s.add(wp),
                    *base.get_unchecked(lane),
                )
            }
        } else {
            (0.0, 0.0)
        };
        let y = hc_pre_lane(mv, lane as u32, sc, bv, hc_eps, iters);
        if lane < HC_MIX {
            let o = (at.t0 + wp) * HC_MIX + lane;
            // The four pre gates on lanes 0..3; the rest of the row zero, so the
            // common fold reads four weights and nothing stale.
            let pre = lane < HC_STREAMS;
            // SAFETY: o < 24·tokens <= mixes.len(), hc.len(); one lane per slot.
            unsafe {
                *mixes.get_unchecked_mut(o) = if pre { mv } else { 0.0 };
                *hc.get_unchecked_mut(o) = if pre { y } else { 0.0 };
            }
        }
    }
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod hc_f32_pieces_kernels {
    use super::*;

    /// [`hc_pre_f32_rmsadd`]: RMS + split-K F32 gemv + HC_PRE with the ERNIE RMS
    /// factor (the module doc's rule) over `tokens` tokens in token groups — the
    /// first `lead`, then groups of [`HC_MAX_TOKENS`] ([`col_group`]): block `b` is
    /// piece `b % n_pieces` of group `b / n_pieces`, which keeps its own partials,
    /// squares and ticket `ctr[g]`, and whose last block finishes its tokens and puts
    /// the ticket back to 0. `w` holds [`HC_MIX`] rows of `512 · n_pieces` values,
    /// `x` the tokens' streams flat. `n_groups` is [`crate::col_group_count`]`(lead,
    /// tokens)`, the launch has exactly `n_pieces · n_groups` blocks, and every
    /// `ctr[g]` holds 0 when it starts. A token whose squares leave its denominator
    /// neither finite nor positive raises [`FaultSite::F32Product`] on `fault`, its
    /// outputs NaN.
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
            w.len() >= 12288 * n_pieces,
            x.len() >= 512 * n_pieces * tokens,
            scale.len() >= 3,
            base.len() >= 24,
            part.len() >= 192 * n_pieces * n_groups,
            ss.len() >= 8 * n_pieces * n_groups,
            ctr.len() >= n_groups,
            mixes.len() >= 24 * tokens,
            hc.len() >= 24 * tokens,
            lead >= 1,
            lead <= 8,
            lead <= tokens,
            n_pieces >= 1
        )
    )]
    pub fn hc_pre_f32_rmsadd(
        w: &[f32],
        x: &[f32],
        scale: &[f32],
        base: &[f32],
        n_pieces: u32,
        lead: u32,
        tokens: u32,
        n_groups: u32,
        rms_eps: f32,
        hc_eps: f32,
        iters: u32,
        mut part: DisjointSlice<f32>,
        mut ss: DisjointSlice<f32>,
        mut ctr: DisjointSlice<u32>,
        mut mixes: DisjointSlice<f32>,
        mut hc: DisjointSlice<f32>,
        fault: FaultSink,
    ) {
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut MIX: SharedArray<f32, HC_MIX_SLOTS> = SharedArray::UNINIT;
        static mut DEN: SharedArray<f32, HC_MAX_TOKENS> = SharedArray::UNINIT;

        let np = n_pieces as usize;
        let b = thread::blockIdx_x() as usize;
        let g = b / np;
        let (t0, m) = col_group(g, lead as usize, tokens as usize);
        // Block-uniform: a block past the last group, none by the launch's
        // geometry, touches nothing.
        if g >= n_groups as usize || m == 0 {
            return;
        }
        // SAFETY: LAST, MIX and DEN are this block's own shared allocations; the raw
        // form is the only way to reach them without a reference to a `static mut`.
        let sh = unsafe {
            HcPreShared {
                last: SharedArray::as_raw_mut_ptr(&raw mut LAST),
                mix: SharedArray::as_raw_mut_ptr(&raw mut MIX),
                scl: SharedArray::as_raw_mut_ptr(&raw mut DEN),
            }
        };
        let at = HcPreAt {
            p: b % np,
            t0,
            part0: g * HC_MIX * np * HC_MAX_TOKENS,
            ss0: g * np * HC_MAX_TOKENS,
            ticket: g,
        };
        // SAFETY: group g is m <= 8 tokens from t0 (t0 + m <= tokens), its
        // n_pieces blocks the consecutive b of g; its scratch regions lie inside
        // part, ss and ctr (g < n_groups) and no other group's block touches them, by
        // the launch contract.
        unsafe {
            pre_block(
                w, x, scale, base, n_pieces, m as u32, rms_eps, hc_eps, iters, &mut part, &mut ss,
                &mut ctr, &mut mixes, &mut hc, fault, at, sh,
            );
        }
    }

    /// [`hc_head_f32`]: the head's four-row mix over `tokens` tokens in the token
    /// groups of [`hc_pre_f32_rmsadd`], with its grid, per-group scratch and
    /// tickets: block `b` is piece `b % n_pieces` of group `b / n_pieces`, whose last
    /// block finishes its tokens and puts `ctr[g]` back to 0. `w` holds
    /// [`HC_STREAMS`] rows of `512 · n_pieces` values, `scale` one value and `base`
    /// four; `mixes` and `hc` take [`HC_MIX`] values a token, the four mixes and pre
    /// gates on values 0..4 and zero on the rest. The launch has exactly `n_pieces ·
    /// n_groups` blocks, and every `ctr[g]` holds 0 when it starts.
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
            w.len() >= 2048 * n_pieces,
            x.len() >= 512 * n_pieces * tokens,
            scale.len() >= 1,
            base.len() >= 4,
            part.len() >= 192 * n_pieces * n_groups,
            ss.len() >= 8 * n_pieces * n_groups,
            ctr.len() >= n_groups,
            mixes.len() >= 24 * tokens,
            hc.len() >= 24 * tokens,
            lead >= 1,
            lead <= 8,
            lead <= tokens,
            n_pieces >= 1
        )
    )]
    pub fn hc_head_f32(
        w: &[f32],
        x: &[f32],
        scale: &[f32],
        base: &[f32],
        n_pieces: u32,
        lead: u32,
        tokens: u32,
        n_groups: u32,
        rms_eps: f32,
        hc_eps: f32,
        iters: u32,
        mut part: DisjointSlice<f32>,
        mut ss: DisjointSlice<f32>,
        mut ctr: DisjointSlice<u32>,
        mut mixes: DisjointSlice<f32>,
        mut hc: DisjointSlice<f32>,
        fault: FaultSink,
    ) {
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut MIX: SharedArray<f32, HC_MIX_SLOTS> = SharedArray::UNINIT;
        static mut SCL: SharedArray<f32, HC_MAX_TOKENS> = SharedArray::UNINIT;

        let np = n_pieces as usize;
        let b = thread::blockIdx_x() as usize;
        let g = b / np;
        let (t0, m) = col_group(g, lead as usize, tokens as usize);
        // Block-uniform: a block past the last group, none by the launch's
        // geometry, touches nothing.
        if g >= n_groups as usize || m == 0 {
            return;
        }
        // SAFETY: LAST, MIX and SCL are this block's own shared allocations; the raw
        // form is the only way to reach them without a reference to a `static mut`.
        let sh = unsafe {
            HcPreShared {
                last: SharedArray::as_raw_mut_ptr(&raw mut LAST),
                mix: SharedArray::as_raw_mut_ptr(&raw mut MIX),
                scl: SharedArray::as_raw_mut_ptr(&raw mut SCL),
            }
        };
        let at = HcPreAt {
            p: b % np,
            t0,
            part0: g * HC_MIX * np * HC_MAX_TOKENS,
            ss0: g * np * HC_MAX_TOKENS,
            ticket: g,
        };
        // SAFETY: group g is m <= 8 tokens from t0 (t0 + m <= tokens), its
        // n_pieces blocks the consecutive b of g; its scratch regions lie inside
        // part, ss and ctr (g < n_groups) and no other group's block touches them, by
        // the launch contract.
        unsafe {
            head_block(
                w, x, scale, base, n_pieces, m as u32, rms_eps, hc_eps, iters, &mut part, &mut ss,
                &mut ctr, &mut mixes, &mut hc, fault, at, sh,
            );
        }
    }
}

// ------------------------------------------------------------------ host

/// The piece entry a launch runs: the [`HC_MIX`]-row pre with the ERNIE RMS factor,
/// or the [`HC_STREAMS`]-row head under the V4 rule.
#[derive(Clone, Copy)]
enum Entry {
    Pre,
    Head,
}

impl Entry {
    /// The weight rows the entry dots.
    fn rows(self) -> usize {
        match self {
            Entry::Pre => HC_MIX,
            Entry::Head => HC_STREAMS,
        }
    }

    /// The scale values the entry reads.
    fn scales(self) -> usize {
        match self {
            Entry::Pre => 3,
            Entry::Head => 1,
        }
    }

    /// What its refusals name.
    fn what(self) -> &'static str {
        match self {
            Entry::Pre => "HcF32PiecesKernels::enqueue_pre",
            Entry::Head => "HcF32PiecesKernels::enqueue_head",
        }
    }
}

/// The loaded piece module. Owns no stream: every enqueue takes the caller's, so
/// launches order with the rest of the step and are capturable.
pub struct HcF32PiecesKernels {
    module: hc_f32_pieces_kernels::LoadedModule,
}

/// The first token group of every launch here: one token, then groups of
/// [`HC_MAX_TOKENS`]; the scratch is cut for the same groups ([`crate::col_group_count`]).
pub const LEAD: usize = 1;

impl HcF32PiecesKernels {
    /// Load this module's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<HcF32PiecesKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; each launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(hc_f32_pieces_kernels, ctx)? };
        Ok(HcF32PiecesKernels { module })
    }

    /// Enqueue RMS + split-K F32 gemv + HC_PRE with the ERNIE RMS factor
    /// (`hc_pre_f32_rmsadd`, the module doc's rule) over `a.tokens` tokens of `a.x`
    /// in token groups — the first `lead` (`1 ..= HC_MAX_TOKENS`), then groups of
    /// [`HC_MAX_TOKENS`]: `mixes` takes the scaled mixes and `hc` the HC_PRE result,
    /// [`HC_MIX`] per token. `a.params.w` holds [`HC_MIX`] rows of the scratch's K
    /// values. The scratch holds a group's partials and ticket for each group
    /// ([`HcPreScratch::with_groups`]). Asynchronous, allocation-free, capturable.
    pub fn enqueue_pre(
        &self,
        stream: &CudaStream,
        a: &HcF32Args<'_>,
        lead: usize,
        scratch: &mut HcPreScratch,
        mixes: &mut DeviceBuffer<f32>,
        hc: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.enqueue(stream, a, lead, scratch, (mixes, hc), Entry::Pre)
    }

    /// Enqueue the head's four-row mix (`hc_head_f32`, the module doc) over
    /// `a.tokens` tokens of `a.x` in the token groups of [`Self::enqueue_pre`]:
    /// `mixes` and `hc` take [`HC_MIX`] values a token, the four scaled mixes and
    /// the four pre gates on values 0..4 and zero on the rest, so
    /// `HcKernels::enqueue_fold` folds the streams by `hc`. `a.params.w` holds
    /// [`HC_STREAMS`] rows of the scratch's K values, `scale` one value and `base`
    /// four. Asynchronous, allocation-free, capturable.
    pub fn enqueue_head(
        &self,
        stream: &CudaStream,
        a: &HcF32Args<'_>,
        lead: usize,
        scratch: &mut HcPreScratch,
        mixes: &mut DeviceBuffer<f32>,
        hc: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.enqueue(stream, a, lead, scratch, (mixes, hc), Entry::Head)
    }

    /// The one launcher of both entries.
    fn enqueue(
        &self,
        stream: &CudaStream,
        a: &HcF32Args<'_>,
        lead: usize,
        scratch: &mut HcPreScratch,
        (mixes, hc): (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
        entry: Entry,
    ) -> Result<(), GpuError> {
        let what = entry.what();
        let (rows, scales) = (entry.rows(), entry.scales());
        let (p, m) = (a.params, a.tokens);
        let k = scratch.k();
        let groups = ColGroups::new(0, lead, m)?.count();
        if p.w.rows() != rows
            || p.w.cols() != k
            || p.scale.len() < scales
            || p.base.len() < rows
            || p.iters == 0
            || groups > scratch.groups()
            || a.x.len() < k * m
            || mixes.len() < HC_MIX * m
            || hc.len() < HC_MIX * m
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "w {}x{} (need {rows}x{k}), scale.len() {} (need {scales}), base.len() {} (need {rows}), \
                     iters {}, tokens {m} in {groups} groups (the scratch holds {}), x.len() {} (need {}), \
                     mixes.len() {} / hc.len() {} (need {})",
                    p.w.rows(),
                    p.w.cols(),
                    p.scale.len(),
                    p.base.len(),
                    p.iters,
                    scratch.groups(),
                    a.x.len(),
                    k * m,
                    mixes.len(),
                    hc.len(),
                    HC_MIX * m
                ),
            });
        }
        let grid = launch_u32(what, "grid", scratch.n_pieces * groups)?;
        let n_pieces = launch_u32(what, "n_pieces", scratch.n_pieces)?;
        let lead = launch_u32(what, "lead", lead)?;
        let tokens = launch_u32(what, "tokens", m)?;
        let n_groups = launch_u32(what, "groups", groups)?;
        let cfg = LaunchConfig1D::new(grid, HC_PRE_THREADS as u32, 0);
        let (part, ss, ctr) = (&mut scratch.part, &mut scratch.ss, &mut scratch.ctr);
        match entry {
            Entry::Pre => {
                let prep = self.module.prepare_hc_pre_f32_rmsadd(cfg)?;
                self.module.hc_pre_f32_rmsadd(
                    stream,
                    &prep,
                    p.w.buf(),
                    a.x,
                    p.scale,
                    p.base,
                    n_pieces,
                    lead,
                    tokens,
                    n_groups,
                    a.rms_eps,
                    p.eps,
                    p.iters,
                    part,
                    ss,
                    ctr,
                    mixes,
                    hc,
                    a.fault,
                )?;
            }
            Entry::Head => {
                let prep = self.module.prepare_hc_head_f32(cfg)?;
                self.module.hc_head_f32(
                    stream,
                    &prep,
                    p.w.buf(),
                    a.x,
                    p.scale,
                    p.base,
                    n_pieces,
                    lead,
                    tokens,
                    n_groups,
                    a.rms_eps,
                    p.eps,
                    p.iters,
                    part,
                    ss,
                    ctr,
                    mixes,
                    hc,
                    a.fault,
                )?;
            }
        }
        Ok(())
    }
}
