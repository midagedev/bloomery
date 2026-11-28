//! The MoE router bodies the sigmoid and √softplus instances share: the three
//! kernel bodies of a router over `E` experts, `U` slots and a [`Score`], and
//! the two stages they are built from. A device crate's `#[kernel]` entry is a
//! thin instance of one: it declares the shared arrays at its size (a `static`
//! inside a generic function cannot take its size from the function's
//! parameters), hands them in, and keeps its own launch contract.
//!
//! - [`router_body`]: the logit gemv, the score, the block ticket and, in the
//!   block that draws the last one, the selection; one token a launch.
//! - [`pick_body`]: a batch's selection, one block a token over scores the
//!   pass's F32 product (the tile GEMM, or a few rows' warp-a-row gemv) and its
//!   sigmoid wrote; bit for bit what [`router_body`] writes a token.
//!
//! [`sigmoid_router_instance!`](crate::sigmoid_router_instance) is the one
//! form of a sigmoid instance of the three: its entries, its output buffers
//! and its launchers at one expert and slot count.
//!
//! The bodies are `#[inline(always)]` and each is whole: on this
//! toolchain a function boundary around the selection loop changes the
//! instruction stream (predicated selects become branches), so the loop, its
//! butterfly and the weight stage stay inside their bodies. The score and the
//! tie rule come from [`crate::route_core`]. Every shape an instance asserts
//! holds when `E` is a multiple of [`ROWS_PER_BLOCK`] and `⌈E / 32⌉ <= 32`. Where
//! `E` is not a multiple of 32 the selecting lanes own `⌈E / 32⌉` or one fewer
//! experts; the guard that skips the missing ones is a compile-time `true`
//! at a multiple of 32, so those instances' instructions are the guard-free
//! ones.

use crate::q8f32::f32_lane_partial_1col_w32;
use crate::route_core::{Score, floor_divisor, renorm_divisor, router_score, take};
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

const _: () = assert!(ROUTER_THREADS_U32 as usize == ROUTER_THREADS);
// The entries' `launch_bounds` and `launch_contract` block are literals.
const _: () = assert!(ROUTER_THREADS == 128);

/// Publish `E` scores: thread `tid` parks `score[e] = probs[at + e]`
/// (a volatile load) and the same value plus any supplied selection bias in
/// `sel_v[e]`, for `e = tid + ROUTER_THREADS · i`. A nonfinite selection
/// value raises [`FaultSite::Router`].
///
/// # Safety
///
/// `probs` addresses at least `at + E` values, any `bias` holds `E`,
/// `score` and `sel_v` are the block's shared arrays of `E`, and the caller's
/// barrier follows before any read of them.
#[inline(always)]
pub unsafe fn publish_n<const E: usize>(
    probs: *const f32,
    bias: Option<&[f32]>,
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
        let v = match bias {
            Some(bias) => {
                // SAFETY: e < E and any supplied bias holds E entries.
                p + unsafe { *bias.get_unchecked(e) }
            }
            None => p,
        };
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
/// argmax under [`take`]`::<true>`, lane `L` scanning experts `L + 32 j < E`
/// (`j < ⌈E / 32⌉`) no earlier round took; lane 0 writes each round's id to
/// `ids[at + s]`, then sums the `U` scores in f64 in slot order. The divisor
/// is [`floor_divisor`] with `FLOOR`, [`renorm_divisor`] otherwise, of the
/// sum narrowed to f32. Each weight is `score / divisor · scale`, raising
/// [`FaultSite::Router`] on one that is not finite. With `REFUSE_NAN`, any
/// nonfinite selection value writes NaN weights instead, so the ids are
/// not a defined pick.
///
/// # Safety
///
/// all 32 lanes of warp 0 call it; `⌈E / 32⌉ <= 32`; `score` and
/// `sel_v` are shared arrays of `E` values published by a barrier, `slot_p`
/// one of `U` that lane 0 alone uses; `ids` and `weights` hold `at + U`
/// values, of which the calling block is the only writer of `at ..`.
#[allow(
    clippy::too_many_arguments,
    reason = "the selecting block's shared arrays, its lane and scale, the fault and the outputs"
)]
#[inline(always)]
pub unsafe fn select_n<
    const E: usize,
    const U: usize,
    const REFUSE_NAN: bool,
    const FLOOR: bool,
>(
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
        while j < const { E.div_ceil(32) } {
            if (taken >> j) & 1 == 0 {
                let ej = lane + 32 * j;
                // At a multiple of 32 every `ej` is below `E`: the guard is a
                // compile-time `true` and leaves no instruction.
                if const { E.is_multiple_of(32) } || ej < E {
                    // SAFETY: ej < E (lane < 32 and j < E / 32 at a multiple
                    // of 32, the guard otherwise), published by the caller's
                    // barrier.
                    let v = unsafe { *sel_v.add(ej) };
                    if take::<true>(v, ej as u32, bv, bi) {
                        bv = v;
                        bi = ej as u32;
                    }
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
    // A non-finite lane is not a winner. Refuse it by NaN weights so the
    // finite rank order in `ids` is not a defined pick.
    let mut poisoned = false;
    if const { REFUSE_NAN } {
        let mut e = 0usize;
        while e < E {
            // SAFETY: e < E; `sel_v` was published by the caller's barrier.
            if !unsafe { *sel_v.add(e) }.is_finite() {
                poisoned = true;
            }
            e += 1;
        }
    }
    if poisoned {
        let mut s = 0usize;
        while s < U {
            fault.raise(FaultSite::Router);
            // SAFETY: at + s < at + U; lane 0 is the only writer.
            unsafe { *weights.get_unchecked_mut(at + s) = f32::NAN };
            s += 1;
        }
    } else {
        let mut sum = 0.0f64;
        let mut s = 0usize;
        while s < U {
            // SAFETY: s < U, this lane's own write.
            sum += f64::from(unsafe { *slot_p.add(s) });
            s += 1;
        }
        let sum = if const { FLOOR } {
            floor_divisor(sum as f32)
        } else {
            renorm_divisor(sum as f32)
        };
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
}

/// `ds41_router`'s body (`gpu-deepseek41`) over `E` experts, `U` slots and the score `S`: the
/// row walk (`f32_gemv`'s m = 1 row, the logit into `logits`, its
/// [`router_score`] into `probs`), the fence and the block's ticket, and in
/// the block that draws the last ticket [`publish_n`] and [`select_n`] into
/// `ids` and `weights` from 0, then the ticket count back to zero.
///
/// # Safety
///
/// the grid is `E / ROWS_PER_BLOCK` blocks of [`ROUTER_THREADS`]
/// (the launcher's), every thread arrives converged; `n_expert == E`, `w`
/// holds `n_expert · k` values, `x` `k`, any `bias`, `logits` and `probs`
/// `n_expert`, `ids` and `weights` `U`, `done` one, zero before the launch;
/// `k` is a positive multiple of 32; `last`, `score`, `sel_v` and `slot_p`
/// are this block's own shared arrays of 1, `E`, `E` and `U` entries.
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's arguments and the shared arrays it declares (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn router_body<
    S: Score,
    const E: usize,
    const U: usize,
    const REFUSE_NAN: bool,
    const FLOOR: bool,
>(
    w: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
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
            unsafe { *probs.get_unchecked_mut(row) = router_score::<S>(logit) };
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
        unsafe {
            select_n::<E, U, REFUSE_NAN, FLOOR>(
                score, sel_v, slot_p, lane, scale, fault, ids, weights, 0,
            )
        };
        if lane == 0 {
            // SAFETY: `ticket` is done[0], inside `done` by this fn's
            // contract; every block has drawn its ticket by now.
            let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
            count.store(0, AtomicOrdering::Relaxed);
        }
    }
}

/// `ds41_router_pick`'s body over `E` experts and `U` slots: token
/// `blockIdx.x`'s scores (`probs[token · E ..]`) through [`publish_n`] and
/// [`select_n`] into `ids` and `weights` at `U · token` — [`router_body`]'s
/// last block on those scores.
///
/// # Safety
///
/// the block is [`ROUTER_THREADS`] threads, converged; `n_expert ==
/// E` (else the block returns), `probs` holds `n_expert · t` values, any `bias`
/// `n_expert`, `ids` and `weights` `U · t`; `score`, `sel_v` and `slot_p` are
/// this block's own shared arrays of `E`, `E` and `U` entries.
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's arguments and the shared arrays it declares (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn pick_body<
    const E: usize,
    const U: usize,
    const REFUSE_NAN: bool,
    const FLOOR: bool,
>(
    probs: &[f32],
    bias: Option<&[f32]>,
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
            select_n::<E, U, REFUSE_NAN, FLOOR>(
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

/// The selection of a router whose kept weights take a learned scale an
/// expert: [`select_n`] at a unit scale — `score / divisor · 1`, the one
/// product exact — then, by lane 0, each kept weight multiplied by `escale`
/// of its expert id. This order is the gate: the references divide a kept
/// score by the divisor and scale the quotient, never the score first. A
/// scaled weight that is not finite (a scale that is not) raises
/// [`FaultSite::Router`]; a poisoned selection's NaN weights stay NaN.
///
/// # Safety
///
/// [`select_n`]'s contract, and `escale` holds `E` values.
#[allow(
    clippy::too_many_arguments,
    reason = "the selecting block's shared arrays, its lane, the scale vector, the fault and the outputs"
)]
#[inline(always)]
pub unsafe fn select_scaled_n<
    const E: usize,
    const U: usize,
    const REFUSE_NAN: bool,
    const FLOOR: bool,
>(
    score: *const f32,
    sel_v: *const f32,
    slot_p: *mut f32,
    escale: &[f32],
    lane: usize,
    fault: FaultSink,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    at: usize,
) {
    // SAFETY: this fn's contract is select_n's.
    unsafe {
        select_n::<E, U, REFUSE_NAN, FLOOR>(
            score, sel_v, slot_p, lane, 1.0, fault, ids, weights, at,
        )
    };
    if lane != 0 {
        return;
    }
    let mut s = 0usize;
    while s < U {
        // SAFETY: at + s < at + U is inside ids and weights by select_n's
        // contract, and lane 0 is their only writer; every id select_n wrote
        // is below E (each round's winner is one of the selecting lanes' own
        // experts, or 0 where no candidate is), so escale's entry is inside
        // its E values.
        unsafe {
            let id = *ids.get_unchecked_mut(at + s);
            let wt = *weights.get_unchecked_mut(at + s) * escale.get_unchecked(id as usize);
            if !wt.is_finite() {
                fault.raise(FaultSite::Router);
            }
            *weights.get_unchecked_mut(at + s) = wt;
        }
        s += 1;
    }
}

/// A split-feature router's body over `E` experts and `U` slots: each expert
/// row walked in `w0` and in `w1` (`f32_gemv`'s m = 1 row in each, the same
/// sum order and tree), its two logits into `logits0` and `logits1`, its
/// score [`router_score`]`::<S>` of each summed into `probs`; then the
/// fence and the block's ticket, and in the block that draws the last one
/// [`publish_n`] and [`select_scaled_n`] into `ids` and `weights` from 0, the
/// ticket count back to zero.
///
/// # Safety
///
/// the grid is `E / ROWS_PER_BLOCK` blocks of [`ROUTER_THREADS`]
/// (the launcher's), every thread arrives converged; `n_expert == E`; `w0`
/// and `w1` hold `n_expert · k` values each, `x` `k`, `bias` and `escale`
/// `n_expert`, `logits0`, `logits1` and `probs` `n_expert`, `ids` and
/// `weights` `U`, `done` one, zero before the launch; `k` is a positive
/// multiple of 32; `last`, `score`, `sel_v` and `slot_p` are this block's own
/// shared arrays of 1, `E`, `E` and `U` entries.
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's arguments and the shared arrays it declares (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn router_body_split<
    S: Score,
    const E: usize,
    const U: usize,
    const REFUSE_NAN: bool,
    const FLOOR: bool,
>(
    w0: &[f32],
    w1: &[f32],
    x: &[f32],
    bias: &[f32],
    escale: &[f32],
    n_expert: u32,
    k: u32,
    logits0: &mut DisjointSlice<f32>,
    logits1: &mut DisjointSlice<f32>,
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
        // `f32_gemv`'s m = 1 row, once a router: the same sum order and the
        // same tree.
        // SAFETY: row < n_expert, so w0.len() and w1.len() >= n_expert * k
        // >= (row + 1) * k, and x.len() >= k, by this fn's contract; k is a
        // positive multiple of 32; lane < 32.
        let (p0, p1) = unsafe {
            (
                f32_lane_partial_1col_w32(w0, x, k, row, lane),
                f32_lane_partial_1col_w32(w1, x, k, row, lane),
            )
        };
        let l0 = warp::reduce_sum_f32(p0);
        let l1 = warp::reduce_sum_f32(p1);
        if lane == 0 {
            // SAFETY: row < n_expert <= logits0.len(), logits1.len() and
            // probs.len() by this fn's contract; lane 0 of the row's warp is
            // each slot's only writer.
            unsafe {
                *logits0.get_unchecked_mut(row) = l0;
                *logits1.get_unchecked_mut(row) = l1;
                *probs.get_unchecked_mut(row) = router_score::<S>(l0) + router_score::<S>(l1);
            }
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
    unsafe { publish_n::<E>(probs.as_mut_ptr(), Some(bias), 0, score, sel_v, tid, fault) };
    thread::sync_threads();
    if tid < 32 {
        // SAFETY: the barrier above published score and sel_v; ids and
        // weights hold U entries from 0 by this fn's contract; escale holds
        // n_expert = E values.
        unsafe {
            select_scaled_n::<E, U, REFUSE_NAN, FLOOR>(
                score, sel_v, slot_p, escale, lane, fault, ids, weights, 0,
            )
        };
        if lane == 0 {
            // SAFETY: `ticket` is done[0], inside `done` by this fn's
            // contract; every block has drawn its ticket by now.
            let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
            count.store(0, AtomicOrdering::Relaxed);
        }
    }
}

/// A batch's selection of a split-feature router's scores: token
/// `blockIdx.x`'s scores (`probs[token · E ..]`) through [`publish_n`] and
/// [`select_scaled_n`] into `ids` and `weights` at `U · token` —
/// [`router_body_split`]'s last block on those scores.
///
/// # Safety
///
/// the block is [`ROUTER_THREADS`] threads, converged; `n_expert == E`
/// (else the block returns), `probs` holds `n_expert · t` values, `bias` and
/// `escale` `n_expert`, `ids` and `weights` `U · t`; `score`, `sel_v` and
/// `slot_p` are this block's own shared arrays of `E`, `E` and `U` entries.
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel entry's arguments and the shared arrays it declares (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn pick_body_scaled<
    const E: usize,
    const U: usize,
    const REFUSE_NAN: bool,
    const FLOOR: bool,
>(
    probs: &[f32],
    bias: &[f32],
    escale: &[f32],
    n_expert: u32,
    t: u32,
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
    unsafe {
        publish_n::<E>(
            probs.as_ptr(),
            Some(bias),
            token * E,
            score,
            sel_v,
            tid,
            fault,
        )
    };
    thread::sync_threads();
    if tid < 32 {
        // SAFETY: the barrier above published score and sel_v; token < t, so
        // ids and weights hold U entries from U · token by this fn's
        // contract; escale holds n_expert = E values.
        unsafe {
            select_scaled_n::<E, U, REFUSE_NAN, FLOOR>(
                score,
                sel_v,
                slot_p,
                escale,
                lane,
                fault,
                ids,
                weights,
                token * U,
            )
        };
    }
}

/// One sigmoid router instance, from the generic bodies above: a module
/// holding `N_EXPERT` and `N_USED`, the three `#[cuda_module]` entries
/// (`$router`, `$scores`, `$pick`), `RouterOut` and `RouterKernels` at those
/// counts, compiled into the crate that invokes it (`gpu-deepseek41`'s
/// `glm5next` and `mimo2`; Bolt's five, `arch/nemotron_h_moe/router.rs`; the
/// 128/6 and 256/8 ones below). The expert and slot counts are tokens, not constants, so the
/// entries' `launch_contract` literals and the module's constants are one
/// source. The entry names are the PTX entry names, and `$p_router`,
/// `$p_scores` and `$p_pick` the `prepare_*` launchers `#[cuda_module]`
/// derives from them: a macro cannot join identifiers. `refuse_nan` (a
/// non-finite selection value writes NaN weights) and `floor` (the kept scores'
/// sum is floored at `2^-14` before the division, the [`floor_divisor`] rule,
/// instead of `1e-20` added to it) are optional and false when left out.
#[macro_export]
macro_rules! sigmoid_router_instance {
    (
        $(#[$meta:meta])*
        pub mod $m:ident {
            experts: $e:tt,
            used: $u:tt,
            kernels: $kernels:ident,
            entries: $router:ident, $pick:ident,
            prepare: $p_router:ident, $p_pick:ident $(,)?
        }
    ) => {
        $crate::sigmoid_router_instance! {
            @instance
            $(#[$meta])*
            pub mod $m {
                experts: $e,
                used: $u,
                kernels: $kernels,
                entries: $router, $pick,
                prepare: $p_router, $p_pick,
                refuse_nan: false,
                floor: false,
            }
        }
    };
    (
        $(#[$meta:meta])*
        pub mod $m:ident {
            experts: $e:tt,
            used: $u:tt,
            kernels: $kernels:ident,
            entries: $router:ident, $pick:ident,
            prepare: $p_router:ident, $p_pick:ident,
            refuse_nan: $refuse:tt $(,)?
        }
    ) => {
        $crate::sigmoid_router_instance! {
            @instance
            $(#[$meta])*
            pub mod $m {
                experts: $e,
                used: $u,
                kernels: $kernels,
                entries: $router, $pick,
                prepare: $p_router, $p_pick,
                refuse_nan: $refuse,
                floor: false,
            }
        }
    };
    (
        $(#[$meta:meta])*
        pub mod $m:ident {
            experts: $e:tt,
            used: $u:tt,
            kernels: $kernels:ident,
            entries: $router:ident, $pick:ident,
            prepare: $p_router:ident, $p_pick:ident,
            refuse_nan: $refuse:tt,
            floor: $floor:tt $(,)?
        }
    ) => {
        $crate::sigmoid_router_instance! {
            @instance
            $(#[$meta])*
            pub mod $m {
                experts: $e,
                used: $u,
                kernels: $kernels,
                entries: $router, $pick,
                prepare: $p_router, $p_pick,
                refuse_nan: $refuse,
                floor: $floor,
            }
        }
    };
    (
        @instance
        $(#[$meta:meta])*
        pub mod $m:ident {
            experts: $e:tt,
            used: $u:tt,
            kernels: $kernels:ident,
            entries: $router:ident, $pick:ident,
            prepare: $p_router:ident, $p_pick:ident,
            refuse_nan: $refuse:tt,
            floor: $floor:tt $(,)?
        }
    ) => {
        $(#[$meta])*
        pub mod $m {

        use $crate::gemm::Gemm32Kernels;
        use $crate::route_body::{
            ROUTER_THREADS, ROUTER_THREADS_U32, ROWS_PER_BLOCK, pick_body, router_body,
        };
        use $crate::route_core::Sigmoid;
        use $crate::{DeviceTensor, FaultSink, Gpu, GpuError, launch_u32};
        use model::arch::models::shape::{self, RouterBody, RouterInst, rules};
        use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
        use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract};
        use cuda_host::cuda_module;
        use std::sync::Arc;

        /// Experts the router scores (`expert_count`).
        pub const N_EXPERT: usize = $e;
        /// Experts each token routes to (`expert_used_count`).
        pub const N_USED: usize = $u;

        /// The instance's row in `models::shape::ROUTERS`: a shape read from
        /// a file selects it through `select_router`. With `floor` the kept
        /// scores' sum is floored at `2^-14` before the division, else
        /// `1e-20` is added to it.
        pub const ROW: RouterInst = if $floor {
            shape::router_row_rule(
                RouterBody::SigmoidFloor,
                rules::BIASED_SIGMOID_FLOOR,
                N_EXPERT as u32,
                N_USED as u32,
            )
        } else {
            shape::router_row_k(RouterBody::BiasedSigmoid, N_EXPERT as u32, N_USED as u32)
        };
        const _: () = assert!(shape::router_row_is(
            ROW,
            if $floor { rules::BIASED_SIGMOID_FLOOR } else { rules::BIASED_SIGMOID },
            false,
            (N_USED as u32, N_USED as u32)
        ));

        const _: () = assert!(N_EXPERT.is_multiple_of(ROWS_PER_BLOCK));
        // The selecting lane keeps its taken entries as bits of one u32.
        const _: () = assert!(N_EXPERT.div_ceil(32) <= 32);
        // The entries' `launch_bounds` and `launch_contract` block are literals; the expert and slot counts in the contracts are the instance's own tokens.
        const _: () = assert!(ROUTER_THREADS == 128);

        #[allow(
            dead_code,
            reason = "#[cuda_module] derives an `_unchecked` launcher per entry that no caller uses"
        )]
        #[cuda_module]
        mod $kernels {
            use super::*;

            /// The router for one token (`router_body` at [`N_EXPERT`],
            /// [`N_USED`], sigmoid): `n_expert` rows of `k` f32 in `w`, the
            /// token's `k` activations in `x`, the selection bias in `bias`.
            /// Writes `logits[e]` and `probs[e]` (the score) for every expert and
            /// `ids[s]`/`weights[s]` for the slots in rank order. `done[0]` is
            /// the block ticket count: zero before the launch, zero again after
            /// it. Grid `n_expert / 4` blocks of 128, warp `r % 4` of block `r /
            /// 4` owning row `r`.
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
                    n_expert == $e,
                    w.len() >= n_expert * k,
                    x.len() >= k,
                    bias.len() >= n_expert,
                    logits.len() >= n_expert,
                    probs.len() >= n_expert,
                    ids.len() >= $u,
                    weights.len() >= $u,
                    done.len() >= 1
                )
            )]
            pub fn $router(
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
                // SAFETY: SCORE, SEL_V, LAST and SLOT_P are this block's own
                // shared arrays of N_EXPERT, N_EXPERT, 1 and N_USED entries (the
                // raw form reaches each `static mut` without a reference); the
                // grid is N_EXPERT / 4 blocks of 128 (the launcher's), every
                // thread arrives converged; the lengths are the launch
                // contract's, the launcher passes k a positive multiple of 32,
                // and `done[0]` is zero before the launch.
                unsafe {
                    router_body::<Sigmoid, N_EXPERT, N_USED, $refuse, $floor>(
                        w,
                        x,
                        Some(bias),
                        n_expert,
                        k,
                        scale,
                        &mut logits,
                        &mut probs,
                        &mut ids,
                        &mut weights,
                        &mut done,
                        fault,
                        SharedArray::as_raw_mut_ptr(&raw mut LAST),
                        SharedArray::as_raw_mut_ptr(&raw mut SCORE),
                        SharedArray::as_raw_mut_ptr(&raw mut SEL_V),
                        SharedArray::as_raw_mut_ptr(&raw mut SLOT_P),
                    );
                }
            }

            /// The selection of token `blockIdx.x` of a batch from its scores
            /// (`probs`, [`N_EXPERT`] a token, as the scores
            /// entry writes them) and `bias` into `ids` and `weights` at `N_USED ·
            /// token` (`pick_body`): the router entry's last block on those
            /// scores.
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
                    n_expert == $e,
                    probs.len() >= n_expert * t,
                    bias.len() >= n_expert,
                    ids.len() >= $u * t,
                    weights.len() >= $u * t
                )
            )]
            pub fn $pick(
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
                // SAFETY: SCORE, SEL_V and SLOT_P are this block's own shared
                // arrays of N_EXPERT, N_EXPERT and N_USED entries (the raw form
                // reaches each `static mut` without a reference); the block is
                // 128 threads, converged; the lengths are the launch contract's.
                unsafe {
                    pick_body::<N_EXPERT, N_USED, $refuse, $floor>(
                        probs,
                        Some(bias),
                        n_expert,
                        t,
                        scale,
                        &mut ids,
                        &mut weights,
                        fault,
                        SharedArray::as_raw_mut_ptr(&raw mut SCORE),
                        SharedArray::as_raw_mut_ptr(&raw mut SEL_V),
                        SharedArray::as_raw_mut_ptr(&raw mut SLOT_P),
                    );
                }
            }
        }

        /// Where one [`RouterKernels::enqueue_router`] launch leaves its results:
        /// per expert the logit and the score, per slot the expert id and the
        /// weight, and the block ticket count the launch returns to zero. The
        /// count serves one launch at a time, so launches that share a
        /// `RouterOut` must be ordered on one stream.
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

        /// The loaded router module. Owns no context and no stream — every
        /// enqueue takes the engine stream, so launches order with the rest of
        /// the step and are capturable.
        pub struct RouterKernels {
            module: $kernels::LoadedModule,
        }

        /// The router weight's shape: [`N_EXPERT`] rows of a positive multiple
        /// of 32.
        fn weight_k(what: &'static str, w: &DeviceTensor<f32>) -> Result<usize, GpuError> {
            let k = w.cols();
            if w.rows() != N_EXPERT || k == 0 || !k.is_multiple_of(32) {
                return Err(GpuError::Shape {
                    what,
                    detail: format!(
                        "router weight is {} x {k}, want {N_EXPERT} rows of a positive multiple of 32",
                        w.rows()
                    ),
                });
            }
            Ok(k)
        }

        impl RouterKernels {
            /// Load this module's device bundle into `ctx`. Load-time only.
            pub fn load(ctx: &Arc<CudaContext>) -> Result<RouterKernels, GpuError> {
                // SAFETY: this package owns the embedded device bundle produced
                // for the module above; the launcher checks its launch contract.
                let module = unsafe { $crate::shared_module!($kernels, ctx)? };
                Ok(RouterKernels { module })
            }

            /// Enqueue the router for one token: `w` the router weight as f32
            /// ([`N_EXPERT`] rows of `k`, `k` a positive multiple of 32), `x` the
            /// token's `k` activations, `bias` the [`N_EXPERT`] selection biases,
            /// `scale` the file's `expert_weights_scale`; results into `out`. A
            /// selection it cannot make raises `fault`. Asynchronous,
            /// allocation-free, capturable.
            #[allow(
                clippy::too_many_arguments,
                reason = "the launcher's inputs are the kernel's (rust-quality R8)"
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
                let what = concat!(stringify!($m), "::enqueue_router");
                let k = weight_k(what, w)?;
                if x.len() < k || bias.len() < N_EXPERT {
                    return Err(GpuError::Shape {
                        what,
                        detail: format!(
                            "x.len() {} < k {k} or bias.len() {} < {N_EXPERT}",
                            x.len(),
                            bias.len()
                        ),
                    });
                }
                let k = launch_u32(what, "k", k)?;
                let n_expert = launch_u32(what, "n_expert", N_EXPERT)?;
                let grid = launch_u32(what, "grid", N_EXPERT / ROWS_PER_BLOCK)?;
                let prep = self.module.$p_router(LaunchConfig1D::new(
                    grid,
                    ROUTER_THREADS_U32,
                    0,
                ))?;
                let RouterOut {
                    logits,
                    probs,
                    ids,
                    weights,
                    done,
                } = out;
                self.module.$router(
                    stream,
                    &prep,
                    w.buf(),
                    x,
                    bias,
                    n_expert,
                    k,
                    scale,
                    logits,
                    probs,
                    ids,
                    weights,
                    done,
                    fault,
                )?;
                Ok(())
            }

            /// Enqueue the router for `t` tokens of a batch in two launches: `x`
            /// their activations, `k` a token and token-major, the per-token
            /// scores into `probs` ([`N_EXPERT`] a token), then each token's ids
            /// and weights into `ids` and `weights` ([`N_USED`] a token) — what
            /// `t` [`RouterKernels::enqueue_router`] launches write, bit for bit.
            /// The scores are the common F32 product
            /// (`Gemm32Kernels::enqueue_f32_cols`): up to eight tokens `k` is a
            /// positive multiple of 32; past eight the F32 tile's, a multiple
            /// of 64 with `w` and `x` 16-byte aligned, and any other is a named
            /// `Shape` refusal. A selection value or a weight that is not
            /// finite raises `fault`. Asynchronous, allocation-free.
            #[allow(
                clippy::too_many_arguments,
                reason = "enqueue_router's inputs, the token count and the batch's buffers (rust-quality R8)"
            )]
            pub fn enqueue_router_rows(
                &self,
                stream: &CudaStream,
                gpu: &Gpu,
                wide: &Gemm32Kernels,
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
                let what = concat!(stringify!($m), "::enqueue_router_rows");
                let k = weight_k(what, w)?;
                if t == 0
                    || x.len() < k * t
                    || bias.len() < N_EXPERT
                    || probs.len() < N_EXPERT * t
                    || ids.len() < N_USED * t
                    || weights.len() < N_USED * t
                {
                    return Err(GpuError::Shape {
                        what,
                        detail: format!(
                            "{t} tokens of {k} (at least one): x {}, bias {}, probs {}, ids {}, weights {}",
                            x.len(),
                            bias.len(),
                            probs.len(),
                            ids.len(),
                            weights.len()
                        ),
                    });
                }
                let n_expert = launch_u32(what, "n_expert", N_EXPERT)?;
                let tt = launch_u32(what, "t", t)?;
                // The scores over every token — the common F32 product
                // (`enqueue_f32_cols`: a verify's few tokens a warp a row, a
                // pass's the tile GEMM), bit for bit this body's
                // per-(token, expert) gemv (the same lane walk,
                // `f32_lane_partials`, and the same warp tree,
                // `gemv_lane_sums`) — then the score's scalar over its
                // logits, in place. The pick launch is unchanged.
                Gemm32Kernels::enqueue_f32_cols(Some(wide), gpu.q8f32(), stream, w, x, t, probs)?;
                gpu.elem()
                    .enqueue_sigmoid_rows(stream, probs, N_EXPERT * t)?;
                let prep = self
                    .module
                    .$p_pick(LaunchConfig1D::new(tt, ROUTER_THREADS_U32, 0))?;
                self.module.$pick(
                    stream, &prep, probs, bias, n_expert, tt, scale, ids, weights, fault,
                )?;
                Ok(())
            }
        }

        /// The instance as the common routed block (`routed_block`) takes it:
        /// its counts, its results' buffers and the two launches.
        impl $crate::routed_block::RouterEntry for RouterKernels {
            const N_EXPERT: usize = N_EXPERT;
            const N_USED: usize = N_USED;
            type Out = RouterOut;

            fn load(ctx: &Arc<CudaContext>) -> Result<RouterKernels, GpuError> {
                RouterKernels::load(ctx)
            }

            fn new_out(stream: &CudaStream) -> Result<RouterOut, GpuError> {
                RouterOut::new(stream)
            }

            fn out_ids(out: &RouterOut) -> &DeviceBuffer<u32> {
                &out.ids
            }

            fn out_weights(out: &RouterOut) -> &DeviceBuffer<f32> {
                &out.weights
            }

            fn out_bytes(out: &RouterOut) -> usize {
                out.logits.num_bytes()
                    + out.probs.num_bytes()
                    + out.ids.num_bytes()
                    + out.weights.num_bytes()
                    + out.done.num_bytes()
            }

            fn route(
                &self,
                stream: &CudaStream,
                x: &DeviceBuffer<f32>,
                r: &$crate::routed_block::RouteIn<'_>,
                out: &mut RouterOut,
            ) -> Result<(), GpuError> {
                let bias = r.bias.ok_or($crate::GpuError::State {
                    what: concat!(stringify!($m), "::route"),
                    missing: "the required selection bias",
                })?;
                self.enqueue_router(stream, r.w, x, bias, r.scale, out, r.fault)
            }

            fn route_rows(
                &self,
                (stream, gpu, wide): (&CudaStream, &Gpu, &Gemm32Kernels),
                x: &DeviceBuffer<f32>,
                r: &$crate::routed_block::RouteIn<'_>,
                t: usize,
                rows: $crate::routed_block::RoutedRows<'_>,
            ) -> Result<(), GpuError> {
                let bias = r.bias.ok_or($crate::GpuError::State {
                    what: concat!(stringify!($m), "::route_rows"),
                    missing: "the required selection bias",
                })?;
                self.enqueue_router_rows(
                    stream,
                    gpu,
                    wide,
                    r.w,
                    x,
                    bias,
                    r.scale,
                    t,
                    rows.probs,
                    rows.ids,
                    rows.weights,
                    r.fault,
                )
            }
        }
        }
    };
}

crate::sigmoid_router_instance! {
    /// The sigmoid router at 128 experts and the top 6 (Raon-1's and
    /// Nemotron-3.5-Lightning's): the rule of `sigmoid_128_4` (MiniMax-M3)
    /// at another pick count, from the generic bodies. Named by its shape:
    /// any family at 128/6 imports it.
    pub mod sigmoid_128_6 {
        experts: 128,
        used: 6,
        kernels: sigmoid_128_6_router_kernels,
        entries: sigmoid_128_6_router, sigmoid_128_6_router_pick,
        prepare: prepare_sigmoid_128_6_router, prepare_sigmoid_128_6_router_pick,
    }
}

crate::sigmoid_router_instance! {
    /// The sigmoid router at 128 experts and the top 8 (Inkling 2's): the
    /// rule of `sigmoid_128_6` at the other pick count — `exp_probs_b`
    /// steering the selection only, the kept unbiased scores renormalized
    /// over their `1e-20`-guarded sum — with the launch's `scale` the file's
    /// `expert_weights_scale` (the row this instance was built for takes 8).
    /// A non-finite selection value writes NaN weights, so the finite rank
    /// order is not a defined pick, as at `sigmoid_256_8`.
    pub mod sigmoid_128_8 {
        experts: 128,
        used: 8,
        kernels: sigmoid_128_8_router_kernels,
        entries: sigmoid_128_8_router, sigmoid_128_8_router_pick,
        prepare: prepare_sigmoid_128_8_router, prepare_sigmoid_128_8_router_pick,
        refuse_nan: true,
    }
}

crate::sigmoid_router_instance! {
    /// The sigmoid router at 384 experts and the top 8 (Inkling 2 Small's):
    /// the rule of `sigmoid_128_8` at three times the experts, twelve scored
    /// a lane. The launch's `scale` is the file's `expert_weights_scale`
    /// (the row this instance was built for takes 8). A non-finite selection
    /// value writes NaN weights, as at `sigmoid_128_8`.
    pub mod sigmoid_384_8 {
        experts: 384,
        used: 8,
        kernels: sigmoid_384_8_router_kernels,
        entries: sigmoid_384_8_router, sigmoid_384_8_router_pick,
        prepare: prepare_sigmoid_384_8_router, prepare_sigmoid_384_8_router_pick,
        refuse_nan: true,
    }
}

crate::sigmoid_router_instance! {
    /// The sigmoid router at 256 experts and the top 6: the same rule as
    /// `sigmoid_128_6` (biased sigmoid, the kept unbiased scores renormalized)
    /// at another expert count. The launch's `scale` is the file's
    /// `expert_weights_scale` (1 for the row this instance was built for).
    /// MiMo's 256/top-8 instance is a different row and is not this one.
    pub mod sigmoid_256_6 {
        experts: 256,
        used: 6,
        kernels: sigmoid_256_6_router_kernels,
        entries: sigmoid_256_6_router, sigmoid_256_6_router_pick,
        prepare: prepare_sigmoid_256_6_router, prepare_sigmoid_256_6_router_pick,
    }
}

crate::sigmoid_router_instance! {
    /// The sigmoid router at 256 experts and the top 8: the same rule as
    /// `sigmoid_256_6` at the other pick count, the common instance Muse
    /// Spark's router runs (`exp_probs_b` steering the selection only, the
    /// kept unbiased scores renormalized, scale 1). The shape's row in
    /// `models::shape::ROUTERS` is the one MiMo's own compiled instance
    /// holds (`mimo2_router`): the pick ranges coincide, so `router_row` at
    /// 256 stays that row and this is the second compiled instance of it. A
    /// non-finite selection value writes NaN weights, so the finite rank
    /// order is not a defined pick, as at the fixture's `sigmoid_16_8`.
    pub mod sigmoid_256_8 {
        experts: 256,
        used: 8,
        kernels: sigmoid_256_8_router_kernels,
        entries: sigmoid_256_8_router, sigmoid_256_8_router_pick,
        prepare: prepare_sigmoid_256_8_router, prepare_sigmoid_256_8_router_pick,
        refuse_nan: true,
    }
}

crate::sigmoid_router_instance! {
    /// The sigmoid router at 16 experts and the top 8: the same rule as
    /// `sigmoid_256_8` at the muse-spark fixture's expert count (the mini
    /// keeps the real file's top 8 over its 16 rivals). The real file's own
    /// count is `sigmoid_256_8`'s. A non-finite selection value writes NaN
    /// weights, so the finite rank order is not a defined pick.
    pub mod sigmoid_16_8 {
        experts: 16,
        used: 8,
        kernels: sigmoid_16_8_router_kernels,
        entries: sigmoid_16_8_router, sigmoid_16_8_router_pick,
        prepare: prepare_sigmoid_16_8_router, prepare_sigmoid_16_8_router_pick,
        refuse_nan: true,
    }
}

crate::sigmoid_router_instance! {
    /// The sigmoid router at 896 experts and the top 16 (Kimi K3's): the rule
    /// of `sigmoid_128_6` at another expert and pick count (`exp_probs_b`
    /// steering the selection only, the kept unbiased scores renormalized,
    /// scale 1). Its 28 experts a selecting lane and 16 rounds are inside the
    /// generic bodies' limits (`⌈E / 32⌉ <= 32`, one `u32` of taken bits). A
    /// non-finite selection value writes NaN weights, so the finite rank order
    /// is not a defined pick.
    pub mod sigmoid_896_16 {
        experts: 896,
        used: 16,
        kernels: sigmoid_896_16_router_kernels,
        entries: sigmoid_896_16_router, sigmoid_896_16_router_pick,
        prepare: prepare_sigmoid_896_16_router, prepare_sigmoid_896_16_router_pick,
        refuse_nan: true,
    }
}

crate::sigmoid_router_instance! {
    /// The sigmoid router at 512 experts and the top 16 (Hy4-flash's): the rule
    /// of `sigmoid_896_16` (`exp_probs_b` steering the selection only, the kept
    /// unbiased scores renormalized) with the kept scores' sum floored at
    /// `2^-14` before the division, as llama.cpp's `build_moe_ffn` clamps it for
    /// every architecture whose file carries no `expert_weights_norm_eps`. The
    /// launch's `scale` is the file's `expert_weights_scale`. Its 16 experts a
    /// selecting lane and 16 rounds are inside the generic bodies' limits
    /// (`⌈E / 32⌉ <= 32`, one `u32` of taken bits). A non-finite selection value
    /// writes NaN weights, so the finite rank order is not a defined pick.
    pub mod sigmoid_512_16 {
        experts: 512,
        used: 16,
        kernels: sigmoid_512_16_router_kernels,
        entries: sigmoid_512_16_router, sigmoid_512_16_router_pick,
        prepare: prepare_sigmoid_512_16_router, prepare_sigmoid_512_16_router_pick,
        refuse_nan: true,
        floor: true,
    }
}

crate::sigmoid_router_instance! {
    /// The sigmoid router at 8 experts and the top 2: `sigmoid_896_16`'s rule
    /// at the expert and pick counts of Kimi K3's gate fixture. The real file's
    /// own count is `sigmoid_896_16`'s.
    pub mod sigmoid_8_2 {
        experts: 8,
        used: 2,
        kernels: sigmoid_8_2_router_kernels,
        entries: sigmoid_8_2_router, sigmoid_8_2_router_pick,
        prepare: prepare_sigmoid_8_2_router, prepare_sigmoid_8_2_router_pick,
        refuse_nan: true,
    }
}

/// A fixed selected-sum-floor sigmoid binding over 256 experts and six
/// slots, with an actual optional selection-only bias. Shares the score,
/// publication, selection and sum order with the legacy instances above.
pub mod sigmoid_floor_256_6 {
    use super::{ROUTER_THREADS_U32, ROWS_PER_BLOCK, pick_body, router_body};
    use crate::gemm::Gemm32Kernels;
    use crate::route_core::Sigmoid;
    use crate::routed_block::{RouteIn, RoutedRows, RouterEntry};
    use crate::{DeviceTensor, FaultSink, Gpu, GpuError, launch_u32};
    use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
    use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract};
    use cuda_host::cuda_module;
    use model::arch::models::shape::{self, RouterBody, RouterInst, rules};
    use std::sync::Arc;

    pub const N_EXPERT: usize = 256;
    pub const N_USED: usize = 6;
    /// Both metadata bindings name the same optional-bias device entries.
    pub const ROW: RouterInst =
        shape::router_row_rule(RouterBody::SigmoidFloor, rules::SIGMOID_FLOOR, 256, 6);
    pub const BIASED_ROW: RouterInst = shape::router_row_rule(
        RouterBody::SigmoidFloor,
        rules::BIASED_SIGMOID_FLOOR,
        256,
        6,
    );
    const _: () = assert!(shape::router_row_is(
        ROW,
        rules::SIGMOID_FLOOR,
        false,
        (6, 6)
    ));
    const _: () = assert!(shape::router_row_is(
        BIASED_ROW,
        rules::BIASED_SIGMOID_FLOOR,
        false,
        (6, 6)
    ));
    const _: () = assert!(N_EXPERT.is_multiple_of(ROWS_PER_BLOCK));
    const _: () = assert!(N_EXPERT.div_ceil(32) <= 32);

    #[allow(
        dead_code,
        reason = "#[cuda_module] derives unchecked launchers that no caller uses"
    )]
    #[cuda_module]
    mod sigmoid_floor_256_6_router_kernels {
        use super::*;

        /// One token, with `has_bias` distinguishing an unused borrowed
        /// input slice from a real selection bias. No zero-bias storage.
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
                w.len() >= 256 * k,
                x.len() >= k,
                bias.len() >= 256,
                logits.len() >= 256,
                probs.len() >= 256,
                ids.len() >= 6,
                weights.len() >= 6,
                done.len() >= 1
            )
        )]
        pub fn sigmoid_floor_256_6_router(
            w: &[f32],
            x: &[f32],
            bias: &[f32],
            has_bias: u32,
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
            // SAFETY: the launch contract supplies every input/output length;
            // the launcher uses k a positive multiple of 32, 64 blocks of 128
            // converged threads, and done is zero before each ordered launch.
            // Shared arrays belong to this block and have the body's sizes.
            unsafe {
                router_body::<Sigmoid, N_EXPERT, N_USED, true, true>(
                    w,
                    x,
                    if has_bias != 0 { Some(bias) } else { None },
                    N_EXPERT as u32,
                    k,
                    scale,
                    &mut logits,
                    &mut probs,
                    &mut ids,
                    &mut weights,
                    &mut done,
                    fault,
                    SharedArray::as_raw_mut_ptr(&raw mut LAST),
                    SharedArray::as_raw_mut_ptr(&raw mut SCORE),
                    SharedArray::as_raw_mut_ptr(&raw mut SEL_V),
                    SharedArray::as_raw_mut_ptr(&raw mut SLOT_P),
                );
            }
        }

        /// Each block selects one token's scores, with the same optional
        /// bias and floor as the one-token entry's final block.
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
                probs.len() >= 256 * t,
                bias.len() >= 256,
                ids.len() >= 6 * t,
                weights.len() >= 6 * t
            )
        )]
        pub fn sigmoid_floor_256_6_router_pick(
            probs: &[f32],
            bias: &[f32],
            has_bias: u32,
            t: u32,
            scale: f32,
            mut ids: DisjointSlice<u32>,
            mut weights: DisjointSlice<f32>,
            fault: FaultSink,
        ) {
            static mut SCORE: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
            static mut SEL_V: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
            static mut SLOT_P: SharedArray<f32, N_USED> = SharedArray::UNINIT;
            // SAFETY: launch-contract lengths and 128 converged threads per
            // block; shared arrays belong to this block at the body's sizes.
            unsafe {
                pick_body::<N_EXPERT, N_USED, true, true>(
                    probs,
                    if has_bias != 0 { Some(bias) } else { None },
                    N_EXPERT as u32,
                    t,
                    scale,
                    &mut ids,
                    &mut weights,
                    fault,
                    SharedArray::as_raw_mut_ptr(&raw mut SCORE),
                    SharedArray::as_raw_mut_ptr(&raw mut SEL_V),
                    SharedArray::as_raw_mut_ptr(&raw mut SLOT_P),
                );
            }
        }
    }

    /// One token's expert logits/scores and six rank-ordered ids/weights.
    /// Launches sharing this output must be ordered on one stream.
    pub struct RouterOut {
        pub logits: DeviceBuffer<f32>,
        pub probs: DeviceBuffer<f32>,
        pub ids: DeviceBuffer<u32>,
        pub weights: DeviceBuffer<f32>,
        done: DeviceBuffer<u32>,
    }

    impl RouterOut {
        /// Allocate result storage at load time, including the zero ticket.
        pub fn new(stream: &CudaStream) -> Result<Self, GpuError> {
            Ok(Self {
                logits: DeviceBuffer::zeroed(stream, N_EXPERT)?,
                probs: DeviceBuffer::zeroed(stream, N_EXPERT)?,
                ids: DeviceBuffer::zeroed(stream, N_USED)?,
                weights: DeviceBuffer::zeroed(stream, N_USED)?,
                done: DeviceBuffer::zeroed(stream, 1)?,
            })
        }

        /// The ticket is zero between ordered launches, including replay.
        pub fn tickets(&self, stream: &CudaStream) -> Result<u32, GpuError> {
            Ok(self.done.to_host_vec(stream)?[0])
        }
    }

    /// Loaded floor router entries; all enqueue paths are allocation-free
    /// and capturable, and borrow the optional bias rather than materialize it.
    pub struct RouterKernels {
        module: sigmoid_floor_256_6_router_kernels::LoadedModule,
    }

    fn weight_k(what: &'static str, w: &DeviceTensor<f32>) -> Result<usize, GpuError> {
        let k = w.cols();
        if w.rows() != N_EXPERT || k == 0 || !k.is_multiple_of(32) {
            return Err(GpuError::shape(
                what,
                format!(
                    "router weight is {} x {k}, want 256 rows of a positive multiple of 32",
                    w.rows()
                ),
            ));
        }
        Ok(k)
    }

    fn check_bias(what: &'static str, bias: Option<&DeviceBuffer<f32>>) -> Result<(), GpuError> {
        if let Some(bias) = bias
            && bias.len() < N_EXPERT
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "selection bias has {} entries, want at least 256",
                    bias.len()
                ),
            ));
        }
        Ok(())
    }

    impl RouterKernels {
        /// Load the embedded module once, before capture.
        pub fn load(ctx: &Arc<CudaContext>) -> Result<Self, GpuError> {
            // SAFETY: this crate owns the module's embedded device bundle.
            let module = unsafe { crate::shared_module!(sigmoid_floor_256_6_router_kernels, ctx)? };
            Ok(Self { module })
        }

        /// Six largest sigmoid scores, larger id on ties. Bias changes
        /// selection only; unbiased selected scores are summed in f64 slot
        /// order, narrowed to f32, divided by max(sum, 2^-14), then scaled.
        /// Nonfinite logits, bias-derived keys or weights raise Router and
        /// an invalid key poisons every slot weight instead of routing it.
        #[allow(
            clippy::too_many_arguments,
            reason = "the launcher's inputs are the kernel's (rust-quality R8)"
        )]
        pub fn enqueue_router(
            &self,
            stream: &CudaStream,
            w: &DeviceTensor<f32>,
            x: &DeviceBuffer<f32>,
            bias: Option<&DeviceBuffer<f32>>,
            scale: f32,
            out: &mut RouterOut,
            fault: FaultSink,
        ) -> Result<(), GpuError> {
            let what = "sigmoid_floor_256_6::enqueue_router";
            let k = weight_k(what, w)?;
            check_bias(what, bias)?;
            if x.len() < k {
                return Err(GpuError::shape(
                    what,
                    format!("x.len() {} < k {k}", x.len()),
                ));
            }
            let k = launch_u32(what, "k", k)?;
            let prep = self
                .module
                .prepare_sigmoid_floor_256_6_router(LaunchConfig1D::new(
                    (N_EXPERT / ROWS_PER_BLOCK) as u32,
                    ROUTER_THREADS_U32,
                    0,
                ))?;
            let RouterOut {
                logits,
                probs,
                ids,
                weights,
                done,
            } = out;
            // With no bias the weight slice is only an ABI borrow: the
            // kernel takes None and never loads it as a selection bias.
            self.module.sigmoid_floor_256_6_router(
                stream,
                &prep,
                w.buf(),
                x,
                bias.unwrap_or(w.buf()),
                u32::from(bias.is_some()),
                k,
                scale,
                logits,
                probs,
                ids,
                weights,
                done,
                fault,
            )?;
            Ok(())
        }

        /// Token-major batch equivalent to ordered single-token launches.
        /// The scores are the common F32 product
        /// (`Gemm32Kernels::enqueue_f32_cols`): up to eight tokens k is a
        /// positive multiple of 32; past eight the F32 tile's, a multiple of
        /// 64 with aligned weight/input buffers, and any other is a named
        /// `Shape` refusal. All buffers are caller-provided; no optional-bias
        /// scratch is needed.
        #[allow(
            clippy::too_many_arguments,
            reason = "router inputs, token count and the batch's output buffers (rust-quality R8)"
        )]
        pub fn enqueue_router_rows(
            &self,
            stream: &CudaStream,
            gpu: &Gpu,
            wide: &Gemm32Kernels,
            w: &DeviceTensor<f32>,
            x: &DeviceBuffer<f32>,
            bias: Option<&DeviceBuffer<f32>>,
            scale: f32,
            t: usize,
            probs: &mut DeviceBuffer<f32>,
            ids: &mut DeviceBuffer<u32>,
            weights: &mut DeviceBuffer<f32>,
            fault: FaultSink,
        ) -> Result<(), GpuError> {
            let what = "sigmoid_floor_256_6::enqueue_router_rows";
            let k = weight_k(what, w)?;
            check_bias(what, bias)?;
            let tt = launch_u32(what, "t", t)?;
            let input = k
                .checked_mul(t)
                .ok_or_else(|| GpuError::shape(what, "input size overflow"))?;
            let scores = N_EXPERT
                .checked_mul(t)
                .ok_or_else(|| GpuError::shape(what, "score size overflow"))?;
            let slots = N_USED
                .checked_mul(t)
                .ok_or_else(|| GpuError::shape(what, "slot size overflow"))?;
            if t == 0
                || x.len() < input
                || probs.len() < scores
                || ids.len() < slots
                || weights.len() < slots
            {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "{t} tokens of {k} (at least one): x {}, probs {}, ids {}, weights {}",
                        x.len(),
                        probs.len(),
                        ids.len(),
                        weights.len()
                    ),
                ));
            }
            // The common F32 product uses the one-token row walk and warp sum tree.
            Gemm32Kernels::enqueue_f32_cols(Some(wide), gpu.q8f32(), stream, w, x, t, probs)?;
            gpu.elem().enqueue_sigmoid_rows(stream, probs, scores)?;
            let prep = self
                .module
                .prepare_sigmoid_floor_256_6_router_pick(LaunchConfig1D::new(
                    tt,
                    ROUTER_THREADS_U32,
                    0,
                ))?;
            self.module.sigmoid_floor_256_6_router_pick(
                stream,
                &prep,
                probs,
                bias.unwrap_or(w.buf()),
                u32::from(bias.is_some()),
                tt,
                scale,
                ids,
                weights,
                fault,
            )?;
            Ok(())
        }
    }

    impl RouterEntry for RouterKernels {
        const N_EXPERT: usize = N_EXPERT;
        const N_USED: usize = N_USED;
        const OPTIONAL_BIAS: bool = true;
        const NORM: model::arch::models::RouterNorm =
            model::arch::models::RouterNorm::SelectedSumFloor;
        type Out = RouterOut;

        fn load(ctx: &Arc<CudaContext>) -> Result<Self, GpuError> {
            Self::load(ctx)
        }
        fn new_out(stream: &CudaStream) -> Result<RouterOut, GpuError> {
            RouterOut::new(stream)
        }
        fn out_ids(out: &RouterOut) -> &DeviceBuffer<u32> {
            &out.ids
        }
        fn out_weights(out: &RouterOut) -> &DeviceBuffer<f32> {
            &out.weights
        }
        fn out_bytes(out: &RouterOut) -> usize {
            out.logits.num_bytes()
                + out.probs.num_bytes()
                + out.ids.num_bytes()
                + out.weights.num_bytes()
                + out.done.num_bytes()
        }
        fn route(
            &self,
            stream: &CudaStream,
            x: &DeviceBuffer<f32>,
            r: &RouteIn<'_>,
            out: &mut RouterOut,
        ) -> Result<(), GpuError> {
            self.enqueue_router(stream, r.w, x, r.bias, r.scale, out, r.fault)
        }
        fn route_rows(
            &self,
            (stream, gpu, wide): (&CudaStream, &Gpu, &Gemm32Kernels),
            x: &DeviceBuffer<f32>,
            r: &RouteIn<'_>,
            t: usize,
            rows: RoutedRows<'_>,
        ) -> Result<(), GpuError> {
            self.enqueue_router_rows(
                stream,
                gpu,
                wide,
                r.w,
                x,
                r.bias,
                r.scale,
                t,
                rows.probs,
                rows.ids,
                rows.weights,
                r.fault,
            )
        }
    }
}

/// The split-feature sigmoid router at 512 experts and the top 10 (ERNIE
/// 5.5's): an expert's score is the sum of two routers' sigmoid logits —
/// `sigmoid(w0 · x) + sigmoid(w1 · x)`, the checkpoint's
/// `moe_split_feature_routing` — the selection bias steers the pick only,
/// the kept unbiased scores renormalize over their `1e-20`-guarded sum
/// ([`renorm_divisor`]) and each kept weight then takes a learned per-expert
/// scale ([`RouterKernels::enqueue_router`]'s `escale`). Two routers and a
/// scale vector are inputs [`routed_block::RouteIn`] does not carry (one
/// weight, one scalar), so the instance is its own launches and a block
/// calls them; its selection is the common one ([`select_n`]) and its scores
/// are published by the common [`publish_n`]. A non-finite selection value
/// writes NaN weights, so the finite rank order is not a defined pick.
///
/// The rows launch is three: the two routers' logits over `t` tokens as two
/// F32 tile GEMMs (the first into `probs`, the second into
/// [`RoutedRows::logits`], which it refuses to miss by name), the scores
/// `sigmoid(a) + sigmoid(b)` over them in one element-wise launch into
/// `probs`, and the pick — what `t` one-token launches write, bit for bit.
pub mod sigmoid_split_512_10 {
    use super::{
        ROUTER_THREADS, ROUTER_THREADS_U32, ROWS_PER_BLOCK, pick_body_scaled, router_body_split,
    };
    use crate::gemm::Gemm32Kernels;
    use crate::route_core::{Sigmoid, router_score};
    use crate::routed_block::RoutedRows;
    use crate::{DeviceTensor, FaultSink, GpuError, launch_u32};
    use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
    use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread};
    use cuda_host::cuda_module;
    use std::sync::Arc;

    /// Experts the router scores (`expert_count`).
    pub const N_EXPERT: usize = 512;
    /// Experts each token routes to (`expert_used_count`).
    pub const N_USED: usize = 10;

    const _: () = assert!(N_EXPERT.is_multiple_of(ROWS_PER_BLOCK));
    // The selecting lane keeps its taken entries as bits of one u32.
    const _: () = assert!(N_EXPERT.div_ceil(32) <= 32);
    // The entries' `launch_bounds` and `launch_contract` block are literals.
    const _: () = assert!(ROUTER_THREADS == 128);

    #[allow(
        dead_code,
        reason = "#[cuda_module] derives an `_unchecked` launcher per entry that no caller uses"
    )]
    #[cuda_module]
    mod sigmoid_split_512_10_kernels {
        use super::*;

        /// The router for one token (`router_body_split` at [`N_EXPERT`],
        /// [`N_USED`], sigmoid): `n_expert` rows of `k` f32 in `w0` and in
        /// `w1`, the token's `k` activations in `x`, the selection bias in
        /// `bias`, the learned per-expert scale in `escale`. Writes each
        /// router's `logits[e]`, the summed score `probs[e]` for every expert
        /// and `ids[s]`/`weights[s]` for the slots in rank order. `done[0]`
        /// is the block ticket count: zero before the launch, zero again after
        /// it. Grid `n_expert / 4` blocks of 128, warp `r % 4` of block `r / 4`
        /// owning row `r`.
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
                n_expert == 512,
                w0.len() >= n_expert * k,
                w1.len() >= n_expert * k,
                x.len() >= k,
                bias.len() >= n_expert,
                escale.len() >= n_expert,
                logits0.len() >= n_expert,
                logits1.len() >= n_expert,
                probs.len() >= n_expert,
                ids.len() >= 10,
                weights.len() >= 10,
                done.len() >= 1
            )
        )]
        pub fn sigmoid_split_512_10_router(
            w0: &[f32],
            w1: &[f32],
            x: &[f32],
            bias: &[f32],
            escale: &[f32],
            n_expert: u32,
            k: u32,
            mut logits0: DisjointSlice<f32>,
            mut logits1: DisjointSlice<f32>,
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
            // SAFETY: SCORE, SEL_V, LAST and SLOT_P are this block's own
            // shared arrays of N_EXPERT, N_EXPERT, 1 and N_USED entries (the
            // raw form reaches each `static mut` without a reference); the
            // grid is N_EXPERT / 4 blocks of 128 (the launcher's), every
            // thread arrives converged; the lengths are the launch
            // contract's, the launcher passes k a positive multiple of 32,
            // and `done[0]` is zero before the launch.
            unsafe {
                router_body_split::<Sigmoid, N_EXPERT, N_USED, true, false>(
                    w0,
                    w1,
                    x,
                    bias,
                    escale,
                    n_expert,
                    k,
                    &mut logits0,
                    &mut logits1,
                    &mut probs,
                    &mut ids,
                    &mut weights,
                    &mut done,
                    fault,
                    SharedArray::as_raw_mut_ptr(&raw mut LAST),
                    SharedArray::as_raw_mut_ptr(&raw mut SCORE),
                    SharedArray::as_raw_mut_ptr(&raw mut SEL_V),
                    SharedArray::as_raw_mut_ptr(&raw mut SLOT_P),
                );
            }
        }

        /// The scores of a batch from its two routers' logits, in place:
        /// `a[i] = sigmoid(a[i]) + sigmoid(b[i])` over the first `n` values,
        /// the one-token entry's score of a pair bit for bit. Grid `⌈n / 128⌉`
        /// blocks of 128, thread `i` entry `i`.
        #[kernel]
        #[launch_bounds(128)]
        #[launch_contract(
            domain = 1,
            block = (128, 1, 1),
            requires = (a.len() >= n, b.len() >= n)
        )]
        pub fn sigmoid_split_512_10_scores(mut a: DisjointSlice<f32>, b: &[f32], n: u32) {
            let i = thread::index_1d().get();
            if i >= n as usize {
                return;
            }
            // SAFETY: i < n <= a.len() and b.len() by the launch contract;
            // thread i owns entry i of a.
            unsafe {
                let p = router_score::<Sigmoid>(*a.get_unchecked_mut(i))
                    + router_score::<Sigmoid>(*b.get_unchecked(i));
                *a.get_unchecked_mut(i) = p;
            }
        }

        /// The selection of token `blockIdx.x` of a batch from its scores
        /// (`probs`, [`N_EXPERT`] a token, as the scores entry writes them),
        /// `bias` and `escale` into `ids` and `weights` at [`N_USED`] ·
        /// token (`pick_body_scaled`): the router entry's last block on those
        /// scores.
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
                n_expert == 512,
                probs.len() >= n_expert * t,
                bias.len() >= n_expert,
                escale.len() >= n_expert,
                ids.len() >= 10 * t,
                weights.len() >= 10 * t
            )
        )]
        pub fn sigmoid_split_512_10_pick(
            probs: &[f32],
            bias: &[f32],
            escale: &[f32],
            n_expert: u32,
            t: u32,
            mut ids: DisjointSlice<u32>,
            mut weights: DisjointSlice<f32>,
            fault: FaultSink,
        ) {
            static mut SCORE: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
            static mut SEL_V: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
            static mut SLOT_P: SharedArray<f32, N_USED> = SharedArray::UNINIT;
            // SAFETY: SCORE, SEL_V and SLOT_P are this block's own shared
            // arrays of N_EXPERT, N_EXPERT and N_USED entries (the raw form
            // reaches each `static mut` without a reference); the block is
            // 128 threads, converged; the lengths are the launch contract's.
            unsafe {
                pick_body_scaled::<N_EXPERT, N_USED, true, false>(
                    probs,
                    bias,
                    escale,
                    n_expert,
                    t,
                    &mut ids,
                    &mut weights,
                    fault,
                    SharedArray::as_raw_mut_ptr(&raw mut SCORE),
                    SharedArray::as_raw_mut_ptr(&raw mut SEL_V),
                    SharedArray::as_raw_mut_ptr(&raw mut SLOT_P),
                );
            }
        }
    }

    /// Where one [`RouterKernels::enqueue_router`] launch leaves its results:
    /// per expert each router's logit and the summed score, per slot the
    /// expert id and the weight, and the block ticket count the launch returns
    /// to zero. The count serves one launch at a time, so launches that share
    /// a `RouterOut` must be ordered on one stream.
    pub struct RouterOut {
        /// Router 0's logit of each expert.
        pub logits_0: DeviceBuffer<f32>,
        /// Router 1's logit of each expert.
        pub logits_1: DeviceBuffer<f32>,
        /// The score of each expert: the sum of the two sigmoids.
        pub probs: DeviceBuffer<f32>,
        pub ids: DeviceBuffer<u32>,
        pub weights: DeviceBuffer<f32>,
        done: DeviceBuffer<u32>,
    }

    impl RouterOut {
        /// Allocate the buffers, the ticket count zeroed. Load-time only.
        pub fn new(stream: &CudaStream) -> Result<RouterOut, GpuError> {
            Ok(RouterOut {
                logits_0: DeviceBuffer::zeroed(stream, N_EXPERT)?,
                logits_1: DeviceBuffer::zeroed(stream, N_EXPERT)?,
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

        /// Device bytes of the six buffers, as allocated.
        #[must_use]
        pub fn bytes(&self) -> usize {
            self.logits_0.num_bytes()
                + self.logits_1.num_bytes()
                + self.probs.num_bytes()
                + self.ids.num_bytes()
                + self.weights.num_bytes()
                + self.done.num_bytes()
        }
    }

    /// The loaded router module. Owns no context and no stream — every
    /// enqueue takes the engine stream, so launches order with the rest of the
    /// step and are capturable.
    pub struct RouterKernels {
        module: sigmoid_split_512_10_kernels::LoadedModule,
    }

    /// The two routers' weight shape: [`N_EXPERT`] rows of one positive
    /// multiple of 32 (`k`), `None` at another, as a refusal of `what`.
    fn weight_k(
        what: &'static str,
        (w0, w1): (&DeviceTensor<f32>, &DeviceTensor<f32>),
    ) -> Result<usize, GpuError> {
        let (k0, k1) = (w0.cols(), w1.cols());
        if w0.rows() != N_EXPERT
            || w1.rows() != N_EXPERT
            || k0 == 0
            || !k0.is_multiple_of(32)
            || k1 != k0
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "router weights are {} x {k0} and {} x {k1}, want {N_EXPERT} rows of one positive multiple of 32",
                    w0.rows(),
                    w1.rows()
                ),
            });
        }
        Ok(k0)
    }

    impl RouterKernels {
        /// Load this module's device bundle into `ctx`. Load-time only.
        pub fn load(ctx: &Arc<CudaContext>) -> Result<RouterKernels, GpuError> {
            // SAFETY: this package owns the embedded device bundle produced
            // for the module above; the launcher checks its launch contract.
            let module = unsafe { crate::shared_module!(sigmoid_split_512_10_kernels, ctx)? };
            Ok(RouterKernels { module })
        }

        /// Enqueue the router for one token: `w0` and `w1` the two router
        /// weights as f32 ([`N_EXPERT`] rows of one `k`, a positive multiple
        /// of 32), `x` the token's `k` activations, `bias` the [`N_EXPERT`]
        /// selection biases, `escale` the [`N_EXPERT`] learned scales; the
        /// results into `out`. A selection it cannot make, or a scale that is
        /// not finite, raises `fault`. Asynchronous, allocation-free,
        /// capturable.
        #[allow(
            clippy::too_many_arguments,
            reason = "the launcher's inputs are the kernel's (rust-quality R8)"
        )]
        pub fn enqueue_router(
            &self,
            stream: &CudaStream,
            (w0, w1): (&DeviceTensor<f32>, &DeviceTensor<f32>),
            x: &DeviceBuffer<f32>,
            bias: &DeviceBuffer<f32>,
            escale: &DeviceBuffer<f32>,
            out: &mut RouterOut,
            fault: FaultSink,
        ) -> Result<(), GpuError> {
            let what = "sigmoid_split_512_10::enqueue_router";
            let k = weight_k(what, (w0, w1))?;
            if x.len() < k || bias.len() < N_EXPERT || escale.len() < N_EXPERT {
                return Err(GpuError::Shape {
                    what,
                    detail: format!(
                        "x.len() {} < k {k}, bias.len() {} or escale.len() {} < {N_EXPERT}",
                        x.len(),
                        bias.len(),
                        escale.len()
                    ),
                });
            }
            let k = launch_u32(what, "k", k)?;
            let n_expert = launch_u32(what, "n_expert", N_EXPERT)?;
            let grid = launch_u32(what, "grid", N_EXPERT / ROWS_PER_BLOCK)?;
            let prep = self
                .module
                .prepare_sigmoid_split_512_10_router(LaunchConfig1D::new(
                    grid,
                    ROUTER_THREADS_U32,
                    0,
                ))?;
            let RouterOut {
                logits_0,
                logits_1,
                probs,
                ids,
                weights,
                done,
            } = out;
            self.module.sigmoid_split_512_10_router(
                stream,
                &prep,
                w0.buf(),
                w1.buf(),
                x,
                bias,
                escale,
                n_expert,
                k,
                logits_0,
                logits_1,
                probs,
                ids,
                weights,
                done,
                fault,
            )?;
            Ok(())
        }

        /// Enqueue the router for `t` tokens of `x` in three launches (module
        /// doc) into `rows`: what `t` [`RouterKernels::enqueue_router`]
        /// launches write, bit for bit. `rows.logits` holds the second
        /// router's logits and is refused by name when absent. Asynchronous,
        /// allocation-free.
        #[allow(
            clippy::too_many_arguments,
            reason = "enqueue_router's inputs, the token count and the batch's buffers (rust-quality R8)"
        )]
        pub fn enqueue_router_rows(
            &self,
            (stream, wide): (&CudaStream, &Gemm32Kernels),
            (w0, w1): (&DeviceTensor<f32>, &DeviceTensor<f32>),
            x: &DeviceBuffer<f32>,
            bias: &DeviceBuffer<f32>,
            escale: &DeviceBuffer<f32>,
            t: usize,
            rows: RoutedRows<'_>,
            fault: FaultSink,
        ) -> Result<(), GpuError> {
            let what = "sigmoid_split_512_10::enqueue_router_rows";
            let k = weight_k(what, (w0, w1))?;
            let RoutedRows {
                probs,
                logits,
                ids,
                weights,
            } = rows;
            let logits = logits.ok_or(GpuError::State {
                what,
                missing: "the rows' logits buffer: the split router's scores are not the scalar of one router's logits",
            })?;
            let n = N_EXPERT * t;
            if t == 0 || x.len() < k * t || bias.len() < N_EXPERT || escale.len() < N_EXPERT {
                return Err(GpuError::Shape {
                    what,
                    detail: format!(
                        "{t} tokens of {k} (at least one): x {}, bias {}, escale {}",
                        x.len(),
                        bias.len(),
                        escale.len()
                    ),
                });
            }
            if probs.len() < n
                || logits.len() < n
                || ids.len() < N_USED * t
                || weights.len() < N_USED * t
            {
                return Err(GpuError::Shape {
                    what,
                    detail: format!(
                        "{t} tokens of {N_EXPERT} (at least one): probs {}, logits {}, ids {}, weights {}",
                        probs.len(),
                        logits.len(),
                        ids.len(),
                        weights.len()
                    ),
                });
            }
            let tt = launch_u32(what, "t", t)?;
            let n = launch_u32(what, "n", n)?;
            let n_expert = launch_u32(what, "n_expert", N_EXPERT)?;
            // Each router's logits as one F32 tile GEMM over every token, bit
            // for bit the one-token entry's row walk; then the summed score
            // over each pair, into the first router's logits.
            wide.enqueue_f32_tile(stream, w0, x, t, probs)?;
            wide.enqueue_f32_tile(stream, w1, x, t, logits)?;
            let scores = self
                .module
                .prepare_sigmoid_split_512_10_scores(LaunchConfig1D::new(
                    launch_u32(what, "grid", (N_EXPERT * t).div_ceil(ROUTER_THREADS))?,
                    ROUTER_THREADS_U32,
                    0,
                ))?;
            self.module
                .sigmoid_split_512_10_scores(stream, &scores, probs, logits, n)?;
            let pick = self
                .module
                .prepare_sigmoid_split_512_10_pick(LaunchConfig1D::new(
                    tt,
                    ROUTER_THREADS_U32,
                    0,
                ))?;
            self.module.sigmoid_split_512_10_pick(
                stream, &pick, probs, bias, escale, n_expert, tt, ids, weights, fault,
            )?;
            Ok(())
        }
    }
}
