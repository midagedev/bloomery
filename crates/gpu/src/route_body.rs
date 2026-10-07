//! The MoE router bodies the sigmoid and √softplus instances share: the three
//! kernel bodies of a router over `E` experts, `U` slots and a [`Score`], and
//! the two stages they are built from. A device crate's `#[kernel]` entry is a
//! thin instance of one: it declares the shared arrays at its size (a `static`
//! inside a generic function cannot take its size from the function's
//! parameters), hands them in, and keeps its own launch contract.
//!
//! - [`router_body`]: the logit gemv, the score, the block ticket and, in the
//!   block that draws the last one, the selection; one token a launch.
//! - [`scores_body`] and [`pick_body`]: a batch in two launches, the scores of
//!   every (token, expert), then one block a token selecting on them; bit for
//!   bit what [`router_body`] writes a token.
//!
//! The bodies are `#[inline(always)]` and each is whole: on this
//! toolchain a function boundary around the selection loop changes the
//! instruction stream (predicated selects become branches), so the loop, its
//! butterfly and the weight stage stay inside their bodies. The score and the
//! tie rule come from [`crate::route_core`]. Every shape an instance asserts
//! holds when `E` is a multiple of 32 and of [`SCORE_ROWS`] and `E / 32 <= 32`.

use crate::q8f32::f32_lane_partial_1col_w32;
use crate::route_core::{Score, renorm_divisor, take};
use crate::{FaultSink, FaultSite};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{DisjointSlice, thread, threadfence, warp};

/// Threads per router block, four warps, one expert row per warp: 96 blocks
/// at 384 experts, so every SM of the card holds a row's walk.
pub const ROUTER_THREADS: usize = 128;
/// [`ROUTER_THREADS`] as a launch argument.
pub const ROUTER_THREADS_U32: u32 = ROUTER_THREADS as u32;
/// Expert rows a block walks, one a warp.
pub const ROWS_PER_BLOCK: usize = ROUTER_THREADS / 32;

/// The batch score kernel's tile: a block of [`ROUTER_THREADS`] threads
/// owns `SCORE_ROWS` expert rows (two a warp) for `SCORE_COLS` consecutive
/// tokens; every weight value a lane loads serves `SCORE_COLS` tokens and
/// every activation value two rows.
pub const SCORE_ROWS: usize = 8;
/// Tokens a score block covers.
pub const SCORE_COLS: usize = 8;

const _: () = assert!(SCORE_ROWS == 2 * ROWS_PER_BLOCK);
// The entry spells its eight columns out as named accumulators.
const _: () = assert!(SCORE_COLS == 8);
const _: () = assert!(ROUTER_THREADS_U32 as usize == ROUTER_THREADS);
// The entries' `launch_bounds` and `launch_contract` block are literals.
const _: () = assert!(ROUTER_THREADS == 128);

/// V4.1's `publish` over `E` experts: thread `tid` parks `score[e] = probs[at +
/// e]` (a volatile load) and `sel_v[e] = score[e] + bias[e]` for `e = tid +
/// ROUTER_THREADS · i`, raising [`FaultSite::Router`] on a selection value
/// that is not finite.
///
/// # Safety
///
/// `probs` addresses at least `at + E` values, `bias` holds `E`,
/// `score` and `sel_v` are the block's shared arrays of `E`, and the caller's
/// barrier follows before any read of them.
#[inline(always)]
pub unsafe fn publish_n<const E: usize>(
    probs: *const f32,
    bias: &[f32],
    at: usize,
    score: *mut f32,
    sel_v: *mut f32,
    tid: usize,
    fault: FaultSink,
) {
    let mut e = tid;
    while e < E {
        // SAFETY: at + e < at + E, inside probs by this fn's contract.
        let p = unsafe { core::ptr::read_volatile(probs.add(at + e)) };
        // SAFETY: e < E <= bias.len() by this fn's contract.
        let v = p + unsafe { *bias.get_unchecked(e) };
        if !v.is_finite() {
            fault.raise(FaultSite::Router);
        }
        // SAFETY: e < E; thread `tid` writes entries tid + ROUTER_THREADS·i
        // only, before the caller's barrier.
        unsafe {
            *score.add(e) = p;
            *sel_v.add(e) = v;
        }
        e += ROUTER_THREADS;
    }
}

/// V4.1's `select` over `E` experts and `U` slots: `U` rounds of the butterfly
/// argmax under [`take`]`::<true>`, lane `L` scanning experts `L + 32 j`
/// (`j < E / 32`) no earlier round took; lane 0 writes each round's id to
/// `ids[at + s]`, then sums the `U` scores in f64 in slot order, takes
/// [`renorm_divisor`] of the sum narrowed to f32, and writes each weight
/// `score / sum · scale` to `weights[at + s]`, raising [`FaultSite::Router`]
/// on one that is not finite.
///
/// # Safety
///
/// all 32 lanes of warp 0 call it; `E / 32 <= 32`; `score` and
/// `sel_v` are shared arrays of `E` values published by a barrier, `slot_p`
/// one of `U` that lane 0 alone uses; `ids` and `weights` hold `at + U`
/// values, of which the calling block is the only writer of `at ..`.
#[allow(
    clippy::too_many_arguments,
    reason = "the selecting block's shared arrays, its lane and scale, the fault and the outputs"
)]
#[inline(always)]
pub unsafe fn select_n<const E: usize, const U: usize>(
    score: *const f32,
    sel_v: *const f32,
    slot_p: *mut f32,
    lane: usize,
    scale: f32,
    fault: FaultSink,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    at: usize,
) {
    let mut taken = 0u32;
    let mut s = 0usize;
    while s < U {
        let mut bv = f32::NEG_INFINITY;
        let mut bi = 0u32;
        let mut j = 0usize;
        while j < E / 32 {
            if (taken >> j) & 1 == 0 {
                let ej = lane + 32 * j;
                // SAFETY: ej < 32·(E / 32) = E, published by the caller's
                // barrier.
                let v = unsafe { *sel_v.add(ej) };
                if take::<true>(v, ej as u32, bv, bi) {
                    bv = v;
                    bi = ej as u32;
                }
            }
            j += 1;
        }
        let mut off = 16u32;
        while off > 0 {
            let (ov, oi) = (warp::shuffle_xor_f32(bv, off), warp::shuffle_xor(bi, off));
            if take::<true>(ov, oi, bv, bi) {
                bv = ov;
                bi = oi;
            }
            off >>= 1;
        }
        // Every lane leaves the butterfly with the same winner; its owner
        // marks it taken.
        if bi as usize % 32 == lane {
            taken |= 1 << (bi as usize / 32);
        }
        if lane == 0 {
            // SAFETY: bi < E (every candidate is one of a lane's own
            // entries); s < U; lane 0 alone writes and reads slot_p; at + s <
            // at + U is inside ids by this fn's contract.
            unsafe {
                *slot_p.add(s) = *score.add(bi as usize);
                *ids.get_unchecked_mut(at + s) = bi;
            }
        }
        s += 1;
    }
    if lane != 0 {
        return;
    }
    let mut sum = 0.0f64;
    let mut s = 0usize;
    while s < U {
        // SAFETY: s < U, this lane's own write.
        sum += f64::from(unsafe { *slot_p.add(s) });
        s += 1;
    }
    let sum = renorm_divisor(sum as f32);
    let mut s = 0usize;
    while s < U {
        // SAFETY: s < U, this lane's own write.
        let wt = unsafe { *slot_p.add(s) } / sum * scale;
        if !wt.is_finite() {
            fault.raise(FaultSite::Router);
        }
        // SAFETY: at + s < at + U is inside weights by this fn's contract;
        // lane 0 is the only writer.
        unsafe { *weights.get_unchecked_mut(at + s) = wt };
        s += 1;
    }
}

/// `ds41_router`'s body (`gpu-deepseek41`) over `E` experts, `U` slots and the score `S`: the
/// row walk (`f32_gemv`'s m = 1 row, the logit into `logits`, `S::score` of
/// it into `probs`), the fence and the block's ticket, and in the block that
/// draws the last ticket [`publish_n`] and [`select_n`] into `ids` and
/// `weights` from 0, then the ticket count back to zero.
///
/// # Safety
///
/// the grid is `E / ROWS_PER_BLOCK` blocks of [`ROUTER_THREADS`]
/// (the launcher's), every thread arrives converged; `n_expert == E`, `w`
/// holds `n_expert · k` values, `x` `k`, `bias`, `logits` and `probs`
/// `n_expert`, `ids` and `weights` `U`, `done` one, zero before the launch;
/// `k` is a positive multiple of 32; `last`, `score`, `sel_v` and `slot_p`
/// are this block's own shared arrays of 1, `E`, `E` and `U` entries.
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's arguments and the shared arrays it declares (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn router_body<S: Score, const E: usize, const U: usize>(
    w: &[f32],
    x: &[f32],
    bias: &[f32],
    n_expert: u32,
    k: u32,
    scale: f32,
    logits: &mut DisjointSlice<f32>,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    done: &mut DisjointSlice<u32>,
    fault: FaultSink,
    last: *mut u32,
    score: *mut f32,
    sel_v: *mut f32,
    slot_p: *mut f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let row = thread::blockIdx_x() as usize * ROWS_PER_BLOCK + tid / 32;
    if row < n_expert as usize {
        // `f32_gemv`'s m = 1 row: the same sum order and the same tree.
        // SAFETY: row < n_expert, so w.len() >= n_expert * k >= (row + 1) *
        // k, and x.len() >= k, by this fn's contract; k is a positive
        // multiple of 32; lane < 32.
        let partial = unsafe { f32_lane_partial_1col_w32(w, x, k, row, lane) };
        let logit = warp::reduce_sum_f32(partial);
        if lane == 0 {
            // SAFETY: row < n_expert <= logits.len() by this fn's contract;
            // lane 0 of the row's warp is the slot's only writer.
            unsafe { *logits.get_unchecked_mut(row) = logit };
            // SAFETY: row < n_expert <= probs.len() by this fn's contract;
            // lane 0 of the row's warp is the slot's only writer.
            unsafe { *probs.get_unchecked_mut(row) = S::score(logit) };
        }
    }
    threadfence();
    thread::sync_threads();

    let ticket = done.as_mut_ptr();
    if tid == 0 {
        // SAFETY: `ticket` is done[0], inside `done` by this fn's contract;
        // every access to it is atomic.
        let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
        let is_last = count.fetch_add(1, AtomicOrdering::AcqRel) + 1 == thread::gridDim_x();
        // SAFETY: block-shared, one element, written before the barrier that
        // publishes it.
        unsafe { *last = u32::from(is_last) };
    }
    thread::sync_threads();
    // SAFETY: block-shared, one element, written by thread 0 before the
    // barrier above.
    if unsafe { *last } == 0 {
        return;
    }

    // The last block: every block's rows are published.
    // SAFETY: probs holds n_expert = E scores by this fn's contract; score,
    // sel_v and slot_p are this block's shared arrays of E, E and U entries.
    unsafe { publish_n::<E>(probs.as_mut_ptr(), bias, 0, score, sel_v, tid, fault) };
    thread::sync_threads();
    if tid < 32 {
        // SAFETY: the barrier above published score and sel_v; ids and
        // weights hold U entries from 0 by this fn's contract.
        unsafe { select_n::<E, U>(score, sel_v, slot_p, lane, scale, fault, ids, weights, 0) };
        if lane == 0 {
            // SAFETY: `ticket` is done[0], inside `done` by this fn's
            // contract; every block has drawn its ticket by now.
            let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
            count.store(0, AtomicOrdering::Relaxed);
        }
    }
}

/// `ds41_router_scores`' body over `E` experts and the score `S`: block `b`
/// owns rows `SCORE_ROWS · (b % (E / SCORE_ROWS)) ..` and tokens `SCORE_COLS
/// · (b / (E / SCORE_ROWS)) ..`, warp `j` rows `2j` and `2j + 1` of the tile;
/// each lane folds `fma(w[row · k + 32 i + lane], x[token · k + 32 i +
/// lane])` into one partial per (row, token) in increasing `i`, and the warp
/// tree and `S::score` give `probs[token · E + row]` — [`router_body`]'s
/// score of that token's row, bit for bit.
///
/// # Safety
///
/// the block is [`ROUTER_THREADS`] threads, converged; `n_expert ==
/// E` (else the block returns), `w` holds `n_expert · k` values, `x` `k · t`,
/// `probs` `n_expert · t`; `k` is a multiple of 32.
#[inline(always)]
pub unsafe fn scores_body<S: Score, const E: usize>(
    w: &[f32],
    x: &[f32],
    n_expert: u32,
    k: u32,
    t: u32,
    probs: &mut DisjointSlice<f32>,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let b = thread::blockIdx_x() as usize;
    let (k, t) = (k as usize, t as usize);
    let tiles = E / SCORE_ROWS;
    let r0 = (b % tiles) * SCORE_ROWS + 2 * (tid / 32);
    let t0 = (b / tiles) * SCORE_COLS;
    if t0 >= t || n_expert as usize != E {
        return;
    }
    // Warp-uniform: every lane of the block shares t0.
    let cols = (t - t0).min(SCORE_COLS);
    let (w0, w1) = (r0 * k, (r0 + 1) * k);
    // Named accumulators, row 0 then row 1 of the warp, one a token: an
    // array indexed in a loop would live in local memory.
    let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7) = (
        0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32,
    );
    let (mut b0, mut b1, mut b2, mut b3, mut b4, mut b5, mut b6, mut b7) = (
        0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32,
    );
    let mut kk = lane;
    while kk < k {
        // SAFETY: r0 + 1 < n_expert and kk < k, so both rows' values are
        // inside w (w.len() >= n_expert * k by this fn's contract).
        let (wa, wb) = unsafe { (*w.get_unchecked(w0 + kk), *w.get_unchecked(w1 + kk)) };
        macro_rules! fold {
            ($c:expr, $a:ident, $b:ident) => {
                if $c < cols {
                    // SAFETY: t0 + c < t and kk < k, so the value is inside
                    // x (x.len() >= k * t by this fn's contract).
                    let xv = unsafe { *x.get_unchecked((t0 + $c) * k + kk) };
                    $a = f32::mul_add(wa, xv, $a);
                    $b = f32::mul_add(wb, xv, $b);
                }
            };
        }
        fold!(0, a0, b0);
        fold!(1, a1, b1);
        fold!(2, a2, b2);
        fold!(3, a3, b3);
        fold!(4, a4, b4);
        fold!(5, a5, b5);
        fold!(6, a6, b6);
        fold!(7, a7, b7);
        kk += 32;
    }
    macro_rules! store {
        ($c:expr, $r:expr, $v:ident) => {
            let logit = warp::reduce_sum_f32($v);
            if lane == 0 && $c < cols {
                let at = (t0 + $c) * E + r0 + $r;
                // SAFETY: t0 + c < t and r0 + r < n_expert, so at <
                // n_expert * t <= probs.len(); lane 0 of the row's warp is
                // the slot's only writer.
                unsafe { *probs.get_unchecked_mut(at) = S::score(logit) };
            }
        };
    }
    store!(0, 0, a0);
    store!(1, 0, a1);
    store!(2, 0, a2);
    store!(3, 0, a3);
    store!(4, 0, a4);
    store!(5, 0, a5);
    store!(6, 0, a6);
    store!(7, 0, a7);
    store!(0, 1, b0);
    store!(1, 1, b1);
    store!(2, 1, b2);
    store!(3, 1, b3);
    store!(4, 1, b4);
    store!(5, 1, b5);
    store!(6, 1, b6);
    store!(7, 1, b7);
}

/// `ds41_router_pick`'s body over `E` experts and `U` slots: token
/// `blockIdx.x`'s scores (`probs[token · E ..]`) through [`publish_n`] and
/// [`select_n`] into `ids` and `weights` at `U · token` — [`router_body`]'s
/// last block on those scores.
///
/// # Safety
///
/// the block is [`ROUTER_THREADS`] threads, converged; `n_expert ==
/// E` (else the block returns), `probs` holds `n_expert · t` values, `bias`
/// `n_expert`, `ids` and `weights` `U · t`; `score`, `sel_v` and `slot_p` are
/// this block's own shared arrays of `E`, `E` and `U` entries.
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's arguments and the shared arrays it declares (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn pick_body<const E: usize, const U: usize>(
    probs: &[f32],
    bias: &[f32],
    n_expert: u32,
    t: u32,
    scale: f32,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    fault: FaultSink,
    score: *mut f32,
    sel_v: *mut f32,
    slot_p: *mut f32,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let token = thread::blockIdx_x() as usize;
    if token >= t as usize || n_expert as usize != E {
        return;
    }
    // SAFETY: token < t, so the token's E scores are inside probs
    // (probs.len() >= n_expert * t by this fn's contract).
    unsafe { publish_n::<E>(probs.as_ptr(), bias, token * E, score, sel_v, tid, fault) };
    thread::sync_threads();
    if tid < 32 {
        // SAFETY: the barrier above published score and sel_v; token < t, so
        // ids and weights hold U entries from U · token by this fn's
        // contract.
        unsafe {
            select_n::<E, U>(
                score,
                sel_v,
                slot_p,
                lane,
                scale,
                fault,
                ids,
                weights,
                token * U,
            )
        };
    }
}
