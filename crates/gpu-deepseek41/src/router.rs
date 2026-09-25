//! Expert routing: each expert scores √softplus(logit); a per-expert bias is
//! added for the selection only; the top 6 of 384 experts are kept, and their
//! unbiased scores are renormalized to sum 1 and scaled by 1.5.
//!
//! Two shapes. The decode step's is one launch per token ([`RouterKernels::enqueue_router`]):
//! the router's logit gemv with the rest of the chain fused into it. A
//! prompt batch's is two launches for all its tokens
//! ([`RouterKernels::enqueue_router_rows`]): the scores of every (token,
//! expert), each the step's number, then one block per token that runs the
//! step's selection on them. The gemv is `q8f32::f32_gemv`'s m = 1 row
//! in its accumulation order — one warp per expert row, the lane walk
//! `f32_lane_partial_1col_w32` (`f32_lane_partial_1col`'s sum with 32
//! chunks' loads in flight) and the fixed butterfly — so the logits are bit
//! for bit that kernel's. The warp that owns a row also scores it
//! ([`sqrt_softplus`]). The selection needs every expert's score, so it runs
//! in the block that finishes last: each block publishes its rows (a fence,
//! then one atomic ticket per block), and the block that draws the last
//! ticket reads all scores back, adds the selection bias, picks the top six,
//! writes the weights, and puts the ticket count back to zero for the next
//! launch or graph replay.
//!
//! Numeric contract — ik's CPU rule for this architecture, node for node
//! (`SQRT_SOFTPLUS`, `ADD`, `ARGSORT`, `GET_ROWS`, `SUM_ROWS`, `DIV`,
//! `SCALE`), except where named:
//! - score `sqrt(x > 20 ? x : ln(1 + e^x))`. The device's `expf`/`logf` are
//!   CUDA's, a few ulp from the host libm's, and the compiler fuses
//!   `1 + expf(x)` into `expf`'s last step, so that sum rounds once;
//! - selection value `score + bias`; top six by (value descending, index
//!   descending) — ik sorts `(value, index)` pairs with `std::greater`, so an
//!   equal value goes to the LARGER expert id;
//! - weights: the six unbiased scores in slot order summed in f64, the sum
//!   narrowed to f32 and guarded ([`renorm_divisor`]: ik divides by the bare
//!   sum, which is 0 when every selected score underflowed, and the guard
//!   leaves every other sum's bits alone), each score divided by it in f32,
//!   then multiplied by `expert_weights_scale` in f32.
//!
//! No silent selection: every expert's selection value must be finite — a
//! NaN or infinite score or bias has no rank, and dropping it while six
//! finite candidates remain would route the token as if the expert did not
//! exist — and a weight that is not finite has no meaning either. Both raise
//! [`FaultSite::Router`] on the launch's fault sink; the ids and weights
//! written stay what the rounds left, and the step that reads the fault back
//! is refused before anything uses them.

use bloomery_gpu::q8f32::f32_lane_partial_1col_w32;
/// ik's expert score in f32; the gates' host side simulates the device with it.
pub use bloomery_gpu::route_core::sqrt_softplus;
use bloomery_gpu::route_core::{renorm_divisor, take};
use bloomery_gpu::{DeviceTensor, FaultSink, FaultSite, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, threadfence, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Experts the router scores (this model's `expert_count`): the width of the
/// selecting block's shared arrays, so a compile-time shape.
pub const N_EXPERT: usize = 384;

/// Experts each token routes to (`expert_used_count`).
pub const N_USED: usize = 6;

/// Threads per router block, four warps, one expert row per warp: 96 blocks,
/// so every SM of the card holds a row's walk.
const ROUTER_THREADS: usize = 128;
const ROUTER_THREADS_U32: u32 = ROUTER_THREADS as u32;
const ROWS_PER_BLOCK: usize = ROUTER_THREADS / 32;

/// Scores each lane of the selecting warp owns: experts `lane + 32 j`.
const PER_LANE: usize = N_EXPERT / 32;

/// The batch score kernel's tile: a block of [`ROUTER_THREADS`] threads
/// owns `SCORE_ROWS` expert rows (two a warp) for `SCORE_COLS` consecutive
/// tokens; every weight value a lane loads serves `SCORE_COLS` tokens and
/// every activation value two rows.
const SCORE_ROWS: usize = 8;
const SCORE_COLS: usize = 8;
const SCORE_TILES: usize = N_EXPERT / SCORE_ROWS;
const _: () = assert!(SCORE_ROWS == 2 * ROWS_PER_BLOCK && N_EXPERT.is_multiple_of(SCORE_ROWS));
// The entry spells its eight columns out as named accumulators.
const _: () = assert!(SCORE_COLS == 8);

const _: () = assert!(ROUTER_THREADS_U32 as usize == ROUTER_THREADS);
// The entry's `launch_bounds` and `launch_contract` block are literals.
const _: () = assert!(ROUTER_THREADS == 128);
const _: () = assert!(N_EXPERT.is_multiple_of(32) && N_EXPERT.is_multiple_of(ROWS_PER_BLOCK));
// The selecting lane keeps its taken entries as bits of one u32.
const _: () = assert!(PER_LANE <= 32);

#[cuda_module]
mod router_kernels {
    use super::*;

    /// The router for one token: `n_expert` (= [`N_EXPERT`]) rows of `k`
    /// f32 in `w`, the token's `k` activations in `x`, the selection bias in
    /// `bias`. Writes `logits[e]` and `probs[e]` (the score) for every
    /// expert, and `ids[s]`/`weights[s]` for the [`N_USED`] slots in rank
    /// order. `done[0]` is the block ticket count: zero before the launch,
    /// zero again after it.
    ///
    /// Grid `n_expert / 4` blocks of [`ROUTER_THREADS`]; warp `r % 4` of
    /// block `r / 4` owns row `r`. After its rows every thread fences and the
    /// block meets at a barrier, so the rows are visible device-wide before
    /// thread 0 draws the block's ticket. The block that draws the last
    /// ticket copies every score into shared memory (coherent loads — its L1
    /// never held those lines, and a volatile load does not ask it), adds the
    /// bias, and its warp 0 runs six rounds of a butterfly argmax under
    /// [`take`]`::<true>` (ties to the larger id): lane `L` scans experts
    /// `L + 32 j` that no earlier round took, the butterfly merges the lanes,
    /// and the winning lane marks its entry taken. Lane 0 writes the ids,
    /// then the weights from the six scores, and puts `done[0]` back to zero.
    /// A selection value that is not finite, or a weight that is not, raises
    /// [`FaultSite::Router`] on `fault` ([`publish`], [`select`]).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n_expert == 384,
            w.len() >= n_expert * k,
            x.len() >= k,
            bias.len() >= n_expert,
            logits.len() >= n_expert,
            probs.len() >= n_expert,
            ids.len() >= 6,
            weights.len() >= 6,
            done.len() >= 1
        )
    )]
    pub fn ds41_router(
        w: &[f32],
        x: &[f32],
        bias: &[f32],
        n_expert: u32,
        k: u32,
        scale: f32,
        mut logits: DisjointSlice<f32>,
        mut probs: DisjointSlice<f32>,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
        mut done: DisjointSlice<u32>,
        fault: FaultSink,
    ) {
        static mut SCORE: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
        static mut SEL_V: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut SLOT_P: SharedArray<f32, N_USED> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let row = thread::blockIdx_x() as usize * ROWS_PER_BLOCK + tid / 32;
        if row < n_expert as usize {
            // `f32_gemv`'s m = 1 row: the same sum order and the same tree.
            // SAFETY: row < n_expert, so w.len() >= n_expert * k >= (row + 1) *
            // k, and x.len() >= k, by the launch contract; the launcher
            // passes k a positive multiple of 32; lane < 32.
            let partial = unsafe { f32_lane_partial_1col_w32(w, x, k, row, lane) };
            let logit = warp::reduce_sum_f32(partial);
            if lane == 0 {
                // SAFETY: row < n_expert <= logits.len() by the launch
                // contract; lane 0 of the row's warp is the slot's only writer.
                unsafe { *logits.get_unchecked_mut(row) = logit };
                let score = sqrt_softplus(logit);
                // SAFETY: row < n_expert <= probs.len() by the launch
                // contract; lane 0 of the row's warp is the slot's only writer.
                unsafe { *probs.get_unchecked_mut(row) = score };
            }
        }
        threadfence();
        thread::sync_threads();

        // SAFETY: block-shared, one element; the raw form reaches the
        // `static mut` without a reference.
        let last = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LAST) };
        let ticket = done.as_mut_ptr();
        if tid == 0 {
            // SAFETY: `ticket` is done[0], inside `done` by the launch
            // contract; every access to it is atomic.
            let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
            let is_last = count.fetch_add(1, AtomicOrdering::AcqRel) + 1 == thread::gridDim_x();
            // SAFETY: block-shared, one element, written before the barrier
            // that publishes it.
            unsafe { *last = u32::from(is_last) };
        }
        thread::sync_threads();
        // SAFETY: block-shared, one element, written by thread 0 before the
        // barrier above.
        if unsafe { *last } == 0 {
            return;
        }

        // The last block: every block's rows are published.
        // SAFETY: block-shared, N_EXPERT entries; the raw form reaches the
        // `static mut` without a reference.
        let score = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORE) };
        // SAFETY: block-shared, N_EXPERT entries; the raw form reaches the
        // `static mut` without a reference.
        let sel_v = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SEL_V) };
        // SAFETY: block-shared, N_USED entries; the raw form reaches the
        // `static mut` without a reference.
        let slot_p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SLOT_P) };
        let probs_ptr = probs.as_mut_ptr();
        // SAFETY: probs holds n_expert = N_EXPERT scores by the launch
        // contract; score, sel_v and slot_p are this block's shared arrays of
        // N_EXPERT, N_EXPERT and N_USED entries.
        unsafe { publish(probs_ptr, bias, 0, score, sel_v, tid, fault) };
        thread::sync_threads();
        if tid < 32 {
            // SAFETY: the barrier above published score and sel_v; ids and
            // weights hold N_USED entries from 0 by the launch contract.
            unsafe {
                select(
                    score,
                    sel_v,
                    slot_p,
                    lane,
                    scale,
                    fault,
                    &mut ids,
                    &mut weights,
                    0,
                )
            };
            if lane == 0 {
                // SAFETY: `ticket` is done[0], inside `done` by the launch
                // contract; every block has drawn its ticket by now.
                let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
                count.store(0, AtomicOrdering::Relaxed);
            }
        }
    }

    /// The scores of `t` tokens' rows of `x` (`k` a token, token-major)
    /// against the router's `n_expert` (= [`N_EXPERT`]) rows of `k` f32 in
    /// `w`: `probs[token · n_expert + e]` is the score [`ds41_router`] writes
    /// for that token's expert `e`, bit for bit. Block `b` owns rows
    /// `SCORE_ROWS · (b % SCORE_TILES) ..` and tokens `SCORE_COLS · (b /
    /// SCORE_TILES) ..` (fewer past the last token); warp `j` of it rows `2j`
    /// and `2j + 1` of the tile. Each lane keeps one partial per (row,
    /// token) and folds `fma(w[row · k + 32 i + lane], x[token · k + 32 i +
    /// lane])` into it in increasing `i` — `f32_lane_partial_1col_w32`'s sum —
    /// and the warp tree is the one-token launch's.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n_expert == 384,
            w.len() >= n_expert * k,
            x.len() >= k * t,
            probs.len() >= n_expert * t
        )
    )]
    pub fn ds41_router_scores(
        w: &[f32],
        x: &[f32],
        n_expert: u32,
        k: u32,
        t: u32,
        mut probs: DisjointSlice<f32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let b = thread::blockIdx_x() as usize;
        let (k, t) = (k as usize, t as usize);
        let r0 = (b % SCORE_TILES) * SCORE_ROWS + 2 * (tid / 32);
        let t0 = (b / SCORE_TILES) * SCORE_COLS;
        if t0 >= t || n_expert as usize != N_EXPERT {
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
            // inside w (w.len() >= n_expert * k by the launch contract).
            let (wa, wb) = unsafe { (*w.get_unchecked(w0 + kk), *w.get_unchecked(w1 + kk)) };
            macro_rules! fold {
                ($c:expr, $a:ident, $b:ident) => {
                    if $c < cols {
                        // SAFETY: t0 + c < t and kk < k, so the value is
                        // inside x (x.len() >= k * t by the launch contract).
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
                    let at = (t0 + $c) * N_EXPERT + r0 + $r;
                    // SAFETY: t0 + c < t and r0 + r < n_expert, so at <
                    // n_expert * t <= probs.len(); lane 0 of the row's warp is
                    // the slot's only writer.
                    unsafe { *probs.get_unchecked_mut(at) = sqrt_softplus(logit) };
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

    /// The selection of token `blockIdx.x` of a batch from its scores
    /// (`probs`, [`N_EXPERT`] a token, as [`ds41_router_scores`] writes them)
    /// and `bias`: [`ds41_router`]'s last block on those scores — the same
    /// shared arrays, the same six rounds and weights — into `ids` and
    /// `weights` at `N_USED · token`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n_expert == 384,
            probs.len() >= n_expert * t,
            bias.len() >= n_expert,
            ids.len() >= 6 * t,
            weights.len() >= 6 * t
        )
    )]
    pub fn ds41_router_pick(
        probs: &[f32],
        bias: &[f32],
        n_expert: u32,
        t: u32,
        scale: f32,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
        fault: FaultSink,
    ) {
        static mut SCORE: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
        static mut SEL_V: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
        static mut SLOT_P: SharedArray<f32, N_USED> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let token = thread::blockIdx_x() as usize;
        if token >= t as usize || n_expert as usize != N_EXPERT {
            return;
        }
        // SAFETY: block-shared arrays; the raw forms reach the `static mut`s
        // without a reference.
        let (score, sel_v, slot_p) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut SCORE),
                SharedArray::as_raw_mut_ptr(&raw mut SEL_V),
                SharedArray::as_raw_mut_ptr(&raw mut SLOT_P),
            )
        };
        // SAFETY: token < t, so the token's N_EXPERT scores are inside probs
        // (probs.len() >= n_expert * t by the launch contract).
        unsafe {
            publish(
                probs.as_ptr(),
                bias,
                token * N_EXPERT,
                score,
                sel_v,
                tid,
                fault,
            );
        }
        thread::sync_threads();
        if tid < 32 {
            // SAFETY: the barrier above published score and sel_v; token < t,
            // so ids and weights hold N_USED entries from N_USED · token by
            // the launch contract.
            unsafe {
                select(
                    score,
                    sel_v,
                    slot_p,
                    lane,
                    scale,
                    fault,
                    &mut ids,
                    &mut weights,
                    token * N_USED,
                );
            }
        }
    }
}

/// Thread `tid` of a [`ROUTER_THREADS`] block parks experts `tid +
/// ROUTER_THREADS · i`: the score `probs[at + e]` (a volatile load, so the
/// row is read where the block that wrote it left it) into `score[e]` and
/// its selection value `score + bias[e]` into `sel_v[e]`. A selection value
/// that is not finite raises [`FaultSite::Router`] — the finite ballot every
/// expert passes before the rounds.
///
/// SAFETY: `probs` addresses at least `at + N_EXPERT` values, `bias` holds
/// `N_EXPERT`, `score` and `sel_v` are the block's shared arrays of
/// `N_EXPERT`, and the caller's barrier follows before any read of them.
#[inline(always)]
unsafe fn publish(
    probs: *const f32,
    bias: &[f32],
    at: usize,
    score: *mut f32,
    sel_v: *mut f32,
    tid: usize,
    fault: FaultSink,
) {
    let mut e = tid;
    while e < N_EXPERT {
        // SAFETY: at + e < at + N_EXPERT, inside probs by this fn's contract.
        let p = unsafe { core::ptr::read_volatile(probs.add(at + e)) };
        // SAFETY: e < N_EXPERT <= bias.len() by this fn's contract.
        let v = p + unsafe { *bias.get_unchecked(e) };
        if !v.is_finite() {
            fault.raise(FaultSite::Router);
        }
        // SAFETY: e < N_EXPERT; thread `tid` writes entries tid +
        // ROUTER_THREADS·i only, before the caller's barrier.
        unsafe {
            *score.add(e) = p;
            *sel_v.add(e) = v;
        }
        e += ROUTER_THREADS;
    }
}

/// Warp 0's selection over the published `score` and `sel_v`: six rounds of
/// a butterfly argmax under [`take`]`::<true>` (ties to the larger id), lane
/// `L` scanning experts `L + 32 j` that no earlier round took, the winning
/// lane marking its entry; lane 0 writes each round's id to `ids[at + s]`,
/// then sums the six scores in f64, takes [`renorm_divisor`], and writes each
/// weight `score / sum · scale` to `weights[at + s]`. A weight that is not
/// finite raises [`FaultSite::Router`].
///
/// SAFETY: all 32 lanes of warp 0 call it; `score` and `sel_v` are shared
/// arrays of `N_EXPERT` values published by a barrier, `slot_p` one of
/// `N_USED` that lane 0 alone uses; `ids` and `weights` hold `at + N_USED`
/// values, of which the calling block is the only writer of `at ..`.
#[allow(
    clippy::too_many_arguments,
    reason = "the selecting block's shared arrays, its lane and scale, the fault and the outputs"
)]
#[inline(always)]
unsafe fn select(
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
    while s < N_USED {
        let mut bv = f32::NEG_INFINITY;
        let mut bi = 0u32;
        let mut j = 0usize;
        while j < PER_LANE {
            if (taken >> j) & 1 == 0 {
                let ej = lane + 32 * j;
                // SAFETY: ej < 32·PER_LANE = N_EXPERT, published by the
                // caller's barrier.
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
            // SAFETY: bi < N_EXPERT (every candidate is one of a lane's own
            // entries); s < N_USED; lane 0 alone writes and reads slot_p; at
            // + s < at + N_USED is inside ids by this fn's contract.
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
    while s < N_USED {
        // SAFETY: s < N_USED, this lane's own write.
        sum += f64::from(unsafe { *slot_p.add(s) });
        s += 1;
    }
    let sum = renorm_divisor(sum as f32);
    let mut s = 0usize;
    while s < N_USED {
        // SAFETY: s < N_USED, this lane's own write.
        let wt = unsafe { *slot_p.add(s) } / sum * scale;
        if !wt.is_finite() {
            fault.raise(FaultSite::Router);
        }
        // SAFETY: at + s < at + N_USED is inside weights by this fn's
        // contract; lane 0 is the only writer.
        unsafe { *weights.get_unchecked_mut(at + s) = wt };
        s += 1;
    }
}

/// Where one router launch leaves its results, allocated once and reused by
/// every launch (and by a captured graph's replays): per expert the logit and
/// the score, per slot the expert id and the weight, and the block ticket
/// count the launch returns to zero. The count serves one launch at a time,
/// so launches that share a `RouterOut` must be ordered on one stream.
pub struct RouterOut {
    pub logits: DeviceBuffer<f32>,
    pub probs: DeviceBuffer<f32>,
    pub ids: DeviceBuffer<u32>,
    pub weights: DeviceBuffer<f32>,
    done: DeviceBuffer<u32>,
}

impl RouterOut {
    /// Allocate the buffers, the ticket count zeroed. Load-time only.
    pub fn new(stream: &CudaStream) -> Result<RouterOut, GpuError> {
        Ok(RouterOut {
            logits: DeviceBuffer::zeroed(stream, N_EXPERT)?,
            probs: DeviceBuffer::zeroed(stream, N_EXPERT)?,
            ids: DeviceBuffer::zeroed(stream, N_USED)?,
            weights: DeviceBuffer::zeroed(stream, N_USED)?,
            done: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    /// The ticket count as it stands: zero between launches.
    pub fn tickets(&self, stream: &CudaStream) -> Result<u32, GpuError> {
        Ok(self.done.to_host_vec(stream)?[0])
    }
}

/// A launch's per-expert rows and ticket count: [`RouterOut`]'s, lent apart
/// from its routing.
struct Rows<'a> {
    logits: &'a mut DeviceBuffer<f32>,
    probs: &'a mut DeviceBuffer<f32>,
    done: &'a mut DeviceBuffer<u32>,
}

/// The loaded router module. Owns no context and no stream — every enqueue
/// takes the engine stream, so launches order with the rest of the step and
/// are capturable.
pub struct RouterKernels {
    module: router_kernels::LoadedModule,
}

impl RouterKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RouterKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { router_kernels::load(ctx)? };
        Ok(RouterKernels { module })
    }

    /// Enqueue the router for one token: `w` the router weight as f32
    /// ([`N_EXPERT`] rows of `k`, `k` a positive multiple of 32), `x` the
    /// token's `k` activations, `bias` the [`N_EXPERT`] selection biases,
    /// `scale` the file's `expert_weights_scale`; results into `out`. A
    /// selection it cannot make raises `fault` (the launch's layer).
    /// Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "the launcher's inputs are the kernel's; a fault sink joined the six"
    )]
    pub fn enqueue_router(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        scale: f32,
        out: &mut RouterOut,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let RouterOut {
            logits,
            probs,
            ids,
            weights,
            done,
        } = out;
        let rows = Rows {
            logits,
            probs,
            done,
        };
        self.launch(stream, w, x, bias, scale, rows, ids, weights, fault)
    }

    /// [`RouterKernels::enqueue_router`] with the routing — the six ids and
    /// weights — written to `ids` and `weights` instead of `out`'s: a prompt
    /// batch keeps each token's routing in its own run of one buffer. The
    /// per-expert rows and the ticket are `out`'s.
    #[allow(
        clippy::too_many_arguments,
        reason = "enqueue_router's inputs and the two routing outputs (rust-quality R8)"
    )]
    pub fn enqueue_router_into(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        scale: f32,
        out: &mut RouterOut,
        ids: &mut DeviceBuffer<u32>,
        weights: &mut DeviceBuffer<f32>,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let rows = Rows {
            logits: &mut out.logits,
            probs: &mut out.probs,
            done: &mut out.done,
        };
        self.launch(stream, w, x, bias, scale, rows, ids, weights, fault)
    }

    /// Enqueue the router for `t` tokens of a prompt batch in two launches:
    /// `x` their activations, `k` a token and token-major, the per-token
    /// scores into `probs` ([`N_EXPERT`] a token), then each token's six ids
    /// and weights into `ids` and `weights` ([`N_USED`] a token) — what `t`
    /// [`RouterKernels::enqueue_router_into`] launches write, bit for bit. A
    /// selection value or a weight that is not finite raises `fault`.
    /// Asynchronous, allocation-free.
    #[allow(
        clippy::too_many_arguments,
        reason = "enqueue_router's inputs, the token count and the batch's buffers (rust-quality R8)"
    )]
    pub fn enqueue_router_rows(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        scale: f32,
        t: usize,
        probs: &mut DeviceBuffer<f32>,
        ids: &mut DeviceBuffer<u32>,
        weights: &mut DeviceBuffer<f32>,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "enqueue_router_rows";
        let shape = |detail: String| GpuError::Shape { what, detail };
        let k = w.cols();
        if w.rows() != N_EXPERT || k == 0 || !k.is_multiple_of(32) || t == 0 {
            return Err(shape(format!(
                "router weight is {} x {k} for {t} tokens, want {N_EXPERT} rows of a positive \
                 multiple of 32 and at least one token",
                w.rows()
            )));
        }
        if x.len() < k * t
            || bias.len() < N_EXPERT
            || probs.len() < N_EXPERT * t
            || ids.len() < N_USED * t
            || weights.len() < N_USED * t
        {
            return Err(shape(format!(
                "{t} tokens of {k}: x {}, bias {}, probs {}, ids {}, weights {}",
                x.len(),
                bias.len(),
                probs.len(),
                ids.len(),
                weights.len()
            )));
        }
        let (k, n_expert) = (
            launch_u32(what, "k", k)?,
            launch_u32(what, "n_expert", N_EXPERT)?,
        );
        let tt = launch_u32(what, "t", t)?;
        let grid = launch_u32(what, "grid", SCORE_TILES * t.div_ceil(SCORE_COLS))?;
        let prep = self.module.prepare_ds41_router_scores(LaunchConfig1D::new(
            grid,
            ROUTER_THREADS_U32,
            0,
        ))?;
        self.module
            .ds41_router_scores(stream, &prep, w.buf(), x, n_expert, k, tt, &mut *probs)?;
        let prep =
            self.module
                .prepare_ds41_router_pick(LaunchConfig1D::new(tt, ROUTER_THREADS_U32, 0))?;
        self.module.ds41_router_pick(
            stream, &prep, probs, bias, n_expert, tt, scale, ids, weights, fault,
        )?;
        Ok(())
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the launcher's inputs are the kernel's (rust-quality R8)"
    )]
    fn launch(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        scale: f32,
        rows: Rows<'_>,
        ids: &mut DeviceBuffer<u32>,
        weights: &mut DeviceBuffer<f32>,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "enqueue_router";
        let shape = |detail: String| GpuError::Shape { what, detail };
        let k = w.cols();
        if w.rows() != N_EXPERT || k == 0 || !k.is_multiple_of(32) {
            return Err(shape(format!(
                "router weight is {} x {k}, want {N_EXPERT} rows of a positive multiple of 32",
                w.rows()
            )));
        }
        if x.len() < k || bias.len() < N_EXPERT {
            return Err(shape(format!(
                "x.len() {} < k {k} or bias.len() {} < {N_EXPERT}",
                x.len(),
                bias.len()
            )));
        }
        if ids.len() < N_USED || weights.len() < N_USED {
            return Err(shape(format!(
                "ids.len() {} and weights.len() {}: the routing takes {N_USED} each",
                ids.len(),
                weights.len()
            )));
        }
        let k = launch_u32(what, "k", k)?;
        let n_expert = launch_u32(what, "n_expert", N_EXPERT)?;
        let grid = launch_u32(what, "grid", N_EXPERT / ROWS_PER_BLOCK)?;
        let prep =
            self.module
                .prepare_ds41_router(LaunchConfig1D::new(grid, ROUTER_THREADS_U32, 0))?;
        self.module.ds41_router(
            stream,
            &prep,
            w.buf(),
            x,
            bias,
            n_expert,
            k,
            scale,
            rows.logits,
            rows.probs,
            ids,
            weights,
            rows.done,
            fault,
        )?;
        Ok(())
    }
}
