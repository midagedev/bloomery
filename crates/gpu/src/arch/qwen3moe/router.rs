//! The qwen3moe router: softmax over the 128 experts' logits, the top 8 by
//! probability, and their weights renormalized to sum to one
//! (`norm_topk_prob`).
//!
//! Four entries share one warp-level routing body, [`route_warp`]:
//! `qwen3moe_router_fused` — the router's logit gemv with the routing fused
//! into it, one launch for up to [`MAX_TOKENS`] tokens; `qwen3moe_router_norm`,
//! the chain's at one token — the same launch with `fused::norm_quant` run
//! in every block first (its q8_1 bytes out, its normed row kept in shared
//! memory for the gemv); `qwen3moe_router`, the routing alone over one
//! token's given logits; and `qwen3moe_router_route`, the routing alone over
//! a ubatch's logits, one warp per token. `qwen3moe_router_logits` writes
//! those: the fused entry's row body over a ubatch of up to [`UBATCH`]
//! tokens, eight tokens and eight expert rows a block, so the two ubatch
//! launches leave each token's logits, ids and weights bit for bit what the
//! fused launch leaves for it. The
//! fused gemv is `q8f32::f32_gemv`'s row body — one warp per expert row,
//! `f32_lane_partials`' sum (at one token the same sum through
//! `f32_lane_partial_1col_w32`, which keeps 32 chunks' loads in flight) and
//! one fixed warp tree per column — so every logit is bit for bit that
//! kernel's. Each block computes one row, with its first warp, so the rows
//! spread over as many SMs as there are experts. The routing needs all of a
//! token's logits, so it runs in the block that finishes last: each block
//! publishes its row (a fence, then one atomic ticket per block), and the
//! block that draws the last ticket puts the count back to zero for the next
//! launch or graph replay and routes token `t` with its warp `t`.
//!
//! Numeric contract, op for op:
//! - the max of the 128 logits by `f32::max` (exact, so its order is free);
//! - `exp(logit − max)` in f32; the sum of the exps in f64, each lane's four
//!   (experts `lane + 32 j`, ascending `j`) and then a five-step xor
//!   butterfly over the lanes; each probability the f32 divide by `sum as
//!   f32`;
//! - the top 8 by descending probability with ties toward the smaller
//!   expert id;
//! - the chosen probabilities summed in f64 in slot order, the sum rounded
//!   once to f32, each weight the f32 divide of its probability by it.

use super::ubatch::UBATCH;
use crate::elem::{RMS_THREADS, RMS_WARPS, rms_partial_sq, rms_scale, rms_warp_tree};
use crate::fault::{FaultSink, FaultSite, quad_finite};
use crate::q8_1_quant_vals;
use crate::q8f32::{f32_lane_partial_1col_w32, f32_lane_partials, gemv_lane_sums};
use crate::route_core::take;
use crate::tensor::{DeviceTensor, Q8Act};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, threadfence, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Experts the router softmaxes over (`expert_count`).
pub const N_EXPERT: usize = 128;

/// Experts each token routes to (`expert_used_count`).
pub const N_USED: usize = 8;

/// Tokens one fused launch routes: the f32 gemv's column bound, and one
/// warp of the routing block per token.
pub const MAX_TOKENS: usize = 8;

/// Threads per routing-only launch: one warp.
const ROUTER_THREADS: u32 = 32;

/// Blocks per eight-token chunk of `qwen3moe_router_logits`: one expert row
/// per warp, so the 128 rows take sixteen blocks.
const ROW_GROUPS: usize = N_EXPERT / FUSED_WARPS;

/// Threads per fused block: eight warps — the first computes the block's
/// expert row, and in the routing block warp `t` routes token `t`.
const FUSED_THREADS: usize = 256;
const FUSED_THREADS_U32: u32 = FUSED_THREADS as u32;
const FUSED_WARPS: usize = FUSED_THREADS / 32;

/// Probabilities each lane of the routing warp owns: experts `lane + 32 j`.
const PER_LANE: usize = N_EXPERT / 32;

/// The fused block's shared scratch: one routing warp's entries per token.
const P_LEN: usize = MAX_TOKENS * N_EXPERT;
const SLOT_LEN: usize = MAX_TOKENS * N_USED;

const _: () = assert!(FUSED_THREADS_U32 as usize == FUSED_THREADS);
// The norm-fused entry runs `norm_quant`'s sum of squares on its whole
// block: the same threads, loads and tree as the norm's first RMS_THREADS.
const _: () = assert!(FUSED_THREADS == RMS_THREADS && FUSED_WARPS == RMS_WARPS);

/// The widest row the norm-fused entry normalizes into shared memory: the
/// qwen3moe hidden width.
pub const NORM_K: usize = 2048;
// `qwen3moe_router_norm`'s launch contract spells the bound out.
const _: () = assert!(NORM_K == 2048);
// Block `g` quantizes the row's 128-value group `g`, so every group needs a
// block.
const _: () = assert!(NORM_K / 128 <= N_EXPERT);
// The routing warp's lanes hold four logits each, as named scalars.
const _: () = assert!(PER_LANE == 4);
// The routing block has a warp for every token.
const _: () = assert!(MAX_TOKENS <= FUSED_WARPS);
// A ubatch logits block covers MAX_TOKENS tokens and FUSED_WARPS rows, the
// rows split evenly over a chunk's blocks, and a ubatch routing block routes
// one token per warp into the MAX_TOKENS entries of P and SLOT_P.
const _: () = assert!(MAX_TOKENS == FUSED_WARPS && N_EXPERT.is_multiple_of(FUSED_WARPS));

/// One token's routing by one warp, the module doc's contract: lane `L`
/// brings the token's logits of experts `L + 32 j` in `v`; `p` and `slot_p`
/// are the token's [`N_EXPERT`] and [`N_USED`] shared entries. Writes
/// `probs[t·128 + e]` for every expert, then `ids[t·8 + s]` and
/// `weights[t·8 + s]` in rank order. The selection runs eight rounds of a
/// butterfly argmax under [`take`]`::<false>` (the order the serial scan's
/// strict `>` over ascending ids realizes, seed `(−inf, 0)`), the winning
/// lane marking its entry taken; lane 0 writes the ids, then the weights.
///
/// # Safety
///
/// All 32 lanes of the warp call it, converged, with the same `p`, `slot_p`
/// and `t`; `p` and `slot_p` point to [`N_EXPERT`] and [`N_USED`] entries of
/// the block's shared memory that no other warp touches; `probs.len() >= (t +
/// 1)·128`, and `ids.len()`, `weights.len() >= (t + 1)·8`, those slots
/// written by this warp alone.
#[inline(always)]
unsafe fn route_warp(
    v: (f32, f32, f32, f32),
    p: *mut f32,
    slot_p: *mut f32,
    t: usize,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
) {
    let lane = warp::lane_id() as usize;
    let (v0, v1, v2, v3) = v;
    let mut mx = f32::NEG_INFINITY.max(v0).max(v1).max(v2).max(v3);
    let mut off = 16u32;
    while off > 0 {
        mx = mx.max(warp::shuffle_xor_f32(mx, off));
        off >>= 1;
    }
    let (e0, e1, e2, e3) = (
        (v0 - mx).exp(),
        (v1 - mx).exp(),
        (v2 - mx).exp(),
        (v3 - mx).exp(),
    );
    let mut sum = f64::from(e0) + f64::from(e1) + f64::from(e2) + f64::from(e3);
    let mut off = 16u32;
    while off > 0 {
        // Both lanes of a pair add the same two values, so every lane leaves
        // with the same sum.
        sum += warp::shuffle_xor_f64(sum, off);
        off >>= 1;
    }
    let inv = sum as f32;
    let (p0, p1, p2, p3) = (e0 / inv, e1 / inv, e2 / inv, e3 / inv);
    let pb = t * N_EXPERT + lane;
    // SAFETY: lane + 96 < N_EXPERT — inside the token's shared entries and
    // its probs slots (this fn's contract); entries `lane + 32 j` are this
    // lane's own.
    unsafe {
        *p.add(lane) = p0;
        *p.add(lane + 32) = p1;
        *p.add(lane + 64) = p2;
        *p.add(lane + 96) = p3;
        *probs.get_unchecked_mut(pb) = p0;
        *probs.get_unchecked_mut(pb + 32) = p1;
        *probs.get_unchecked_mut(pb + 64) = p2;
        *probs.get_unchecked_mut(pb + 96) = p3;
    }
    // Lane 0 reads each round's winner, whichever lane wrote it.
    warp::sync_mask(u32::MAX);

    let mut taken = 0u32;
    let mut s = 0usize;
    while s < N_USED {
        let mut bv = f32::NEG_INFINITY;
        let mut bi = 0u32;
        let mut j = 0usize;
        while j < PER_LANE {
            if (taken >> j) & 1 == 0 {
                let ej = lane + 32 * j;
                // SAFETY: ej < 32·PER_LANE = N_EXPERT, this lane's own entry.
                let v = unsafe { *p.add(ej) };
                if take::<false>(v, ej as u32, bv, bi) {
                    bv = v;
                    bi = ej as u32;
                }
            }
            j += 1;
        }
        let mut off = 16u32;
        while off > 0 {
            let (ov, oi) = (warp::shuffle_xor_f32(bv, off), warp::shuffle_xor(bi, off));
            if take::<false>(ov, oi, bv, bi) {
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
            // SAFETY: bi < N_EXPERT — inside the token's entries, published
            // by the warp sync above; s < N_USED, inside `slot_p` and the
            // token's ids slots (this fn's contract). The probability is read
            // from the winner's entry rather than from the comparison, so one
            // no comparison won is carried through unchanged.
            unsafe {
                *slot_p.add(s) = *p.add(bi as usize);
                *ids.get_unchecked_mut(t * N_USED + s) = bi;
            }
        }
        s += 1;
    }
    if lane == 0 {
        let mut sum = 0.0f64;
        let mut s = 0usize;
        while s < N_USED {
            // SAFETY: s < N_USED, this lane's own write.
            sum += f64::from(unsafe { *slot_p.add(s) });
            s += 1;
        }
        let sum = sum as f32;
        let mut s = 0usize;
        while s < N_USED {
            // SAFETY: s < N_USED, inside `slot_p` and the token's weights
            // slots (this fn's contract); lane 0 is the only writer.
            unsafe { *weights.get_unchecked_mut(t * N_USED + s) = *slot_p.add(s) / sum };
            s += 1;
        }
    }
}

/// The fused entries' tail, after each block's rows are stored: every thread
/// fences and the block meets at a barrier, so its rows are visible
/// device-wide before thread 0 draws the block's ticket; the block that draws
/// the last ticket returns the count to zero, and its warp `t < m` reads
/// token `t`'s logits back (volatile loads: this block's L1 never held other
/// blocks' rows, and a volatile load does not ask it) and runs
/// [`route_warp`].
///
/// # Safety
///
/// Every thread of the block calls it, converged, with the same arguments;
/// `last`, `p` and `slot_p` point to 1, `MAX_TOKENS · N_EXPERT` and
/// `MAX_TOKENS · N_USED` entries of the block's shared memory that nothing
/// else in the block touches from here on; `1 <= m <= MAX_TOKENS <=
/// FUSED_WARPS`; `logits`, `probs` hold `128 · m` and `ids`, `weights` `8 ·
/// m` entries, and `done[0]` is this launch's ticket count, zero before it.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn publish_route(
    m: usize,
    last: *mut u32,
    p: *mut f32,
    slot_p: *mut f32,
    logits: &mut DisjointSlice<f32>,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    done: &mut DisjointSlice<u32>,
) {
    threadfence();
    thread::sync_threads();

    let tid = thread::threadIdx_x() as usize;
    let wi = tid / 32;
    let lane = warp::lane_id() as usize;
    let ticket = done.as_mut_ptr();
    if tid == 0 {
        // SAFETY: `ticket` is done[0] (this fn's contract); every access to
        // it is atomic.
        let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
        let is_last = count.fetch_add(1, AtomicOrdering::AcqRel) + 1 == thread::gridDim_x();
        if is_last {
            // Every block has drawn its ticket.
            count.store(0, AtomicOrdering::Relaxed);
        }
        // SAFETY: block-shared, one element, written before the barrier
        // that publishes it.
        unsafe { *last = u32::from(is_last) };
    }
    thread::sync_threads();
    // SAFETY: block-shared, one element, written by thread 0 before the
    // barrier above.
    if unsafe { *last } == 0 || wi >= m {
        return;
    }

    // The last block, warp `wi` routing token `wi`: every block's rows are
    // published.
    let base = wi * N_EXPERT;
    let lp = logits.as_mut_ptr();
    // SAFETY: base + lane + 96 < m·128 <= logits.len() (this fn's contract).
    let v = unsafe {
        (
            core::ptr::read_volatile(lp.add(base + lane)),
            core::ptr::read_volatile(lp.add(base + lane + 32)),
            core::ptr::read_volatile(lp.add(base + lane + 64)),
            core::ptr::read_volatile(lp.add(base + lane + 96)),
        )
    };
    // SAFETY: wi < m <= MAX_TOKENS, so token wi's N_EXPERT and N_USED
    // entries are inside `p` and `slot_p` and no other warp's; the whole warp
    // is here (`wi` is warp-uniform), converged past the barrier; its slots
    // are inside probs, ids and weights (this fn's contract).
    unsafe {
        route_warp(
            v,
            p.add(base),
            slot_p.add(wi * N_USED),
            wi,
            probs,
            ids,
            weights,
        );
    }
}

/// Lane 0's stores of one expert row's `m` column logits, token-major:
/// `y[c · rows + r] = sums[c]`. One guarded store per column with a constant
/// index: a loop over `c` would index `sums` at run time and put it in a
/// local depot. `r` may carry a token offset (`t0 · rows + row`), which
/// moves every store by whole rows.
///
/// # Safety
///
/// `1 <= m <= 8`, `(m − 1) · rows + r < y.len()`, and no other thread
/// writes those slots.
#[inline(always)]
unsafe fn store_cols(y: &mut DisjointSlice<f32>, r: usize, rows: usize, m: usize, sums: &[f32; 8]) {
    let [s0, s1, s2, s3, s4, s5, s6, s7] = *sums;
    // SAFETY: each store is guarded by c < m, so its slot c·rows + r is one
    // this fn's contract puts inside y and gives to this thread alone.
    unsafe {
        *y.get_unchecked_mut(r) = s0;
        if m > 1 {
            *y.get_unchecked_mut(rows + r) = s1;
        }
        if m > 2 {
            *y.get_unchecked_mut(2 * rows + r) = s2;
        }
        if m > 3 {
            *y.get_unchecked_mut(3 * rows + r) = s3;
        }
        if m > 4 {
            *y.get_unchecked_mut(4 * rows + r) = s4;
        }
        if m > 5 {
            *y.get_unchecked_mut(5 * rows + r) = s5;
        }
        if m > 6 {
            *y.get_unchecked_mut(6 * rows + r) = s6;
        }
        if m > 7 {
            *y.get_unchecked_mut(7 * rows + r) = s7;
        }
    }
}

#[cuda_module]
mod qwen3moe_router_kernels {
    use super::*;

    /// The routing alone for one token: its [`N_EXPERT`] logits in `x` →
    /// every probability in `probs`, the ids in rank order in `ids` and
    /// their renormalized weights in `weights`. One block of one warp,
    /// [`route_warp`].
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            x.len() >= 128,
            probs.len() >= 128,
            ids.len() >= 8,
            weights.len() >= 8
        )
    )]
    pub fn qwen3moe_router(
        x: &[f32],
        mut probs: DisjointSlice<f32>,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
    ) {
        static mut P: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
        static mut SLOT_P: SharedArray<f32, N_USED> = SharedArray::UNINIT;

        let lane = warp::lane_id() as usize;
        // SAFETY: block-shared, N_EXPERT entries; the raw form reaches the
        // `static mut` without a reference.
        let p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut P) };
        // SAFETY: block-shared, N_USED entries; the raw form reaches the
        // `static mut` without a reference.
        let slot_p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SLOT_P) };
        // SAFETY: lane + 96 < 128 <= x.len() by the launch contract.
        let v = unsafe {
            (
                *x.get_unchecked(lane),
                *x.get_unchecked(lane + 32),
                *x.get_unchecked(lane + 64),
                *x.get_unchecked(lane + 96),
            )
        };
        // SAFETY: the block is this one warp, converged here; P and SLOT_P
        // are its own; token 0's slots are inside probs, ids and weights by
        // the launch contract.
        unsafe { route_warp(v, p, slot_p, 0, &mut probs, &mut ids, &mut weights) };
    }

    /// The router of `m_cols` tokens: `w` the router weight ([`N_EXPERT`]
    /// rows of `k` f32), `x` the tokens' normed activations (`m_cols`
    /// columns of `k`). Writes `logits[t·128 + e]`, `probs[t·128 + e]`,
    /// `ids[t·8 + s]` and `weights[t·8 + s]`. `done[0]` is the block ticket
    /// count: zero before the launch, zero again after it.
    ///
    /// Grid [`N_EXPERT`] blocks of 256; warp 0 of block `r` owns row `r`
    /// with `f32_gemv`'s row body, and the other warps meet the barriers
    /// only. After its row every thread fences and the block meets at a
    /// barrier, so the row is visible device-wide before thread 0 draws the
    /// block's ticket. The
    /// block that draws the last ticket returns the count to zero, and its
    /// warp `t < m_cols` reads token `t`'s logits back (volatile loads: this
    /// block's L1 never held other blocks' rows, and a volatile load does not
    /// ask it) and runs [`route_warp`].
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
            m_cols >= 1,
            m_cols <= 8,
            w.len() >= 128 * k,
            x.len() >= m_cols * k,
            logits.len() >= 128 * m_cols,
            probs.len() >= 128 * m_cols,
            ids.len() >= 8 * m_cols,
            weights.len() >= 8 * m_cols,
            done.len() >= 1
        )
    )]
    pub fn qwen3moe_router_fused(
        w: &[f32],
        x: &[f32],
        k: u32,
        m_cols: u32,
        mut logits: DisjointSlice<f32>,
        mut probs: DisjointSlice<f32>,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
        mut done: DisjointSlice<u32>,
    ) {
        static mut P: SharedArray<f32, P_LEN> = SharedArray::UNINIT;
        static mut SLOT_P: SharedArray<f32, SLOT_LEN> = SharedArray::UNINIT;
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let wi = tid / 32;
        let m = m_cols as usize;
        // The row guard is warp-uniform, so the warp tree sees a full warp;
        // the other warps, and a block past the rows, still meet the barrier
        // and the ticket.
        let row = thread::blockIdx_x() as usize;
        if wi == 0 && row < N_EXPERT {
            let partials = if m == 1 {
                // SAFETY: row < N_EXPERT, so w.len() >= 128·k >= (row + 1)·k,
                // and x.len() >= k, by the launch contract; the launcher
                // passes k a positive multiple of 32; lane < 32.
                let f0 = unsafe { f32_lane_partial_1col_w32(w, x, k, row, lane) };
                [f0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            } else {
                f32_lane_partials(w, x, k, row, m_cols, lane)
            };
            let sums = gemv_lane_sums(partials, m_cols);
            if lane == 0 {
                // SAFETY: row < N_EXPERT and 1 <= m <= 8, so the slots c·128 +
                // row (c < m) lie inside logits (launch contract); lane 0 of
                // the row's warp is their only writer.
                unsafe { store_cols(&mut logits, row, N_EXPERT, m, &sums) };
            }
        }
        // SAFETY: LAST, P and SLOT_P are this block's own shared
        // allocations of 1, P_LEN and SLOT_LEN entries (the raw form reaches
        // each `static mut` without a reference), untouched until here; every
        // thread arrives converged with the launch's arguments; 1 <= m <= 8
        // and the output lengths are the launch contract's; `done[0]` is zero
        // before the launch.
        unsafe {
            publish_route(
                m,
                SharedArray::as_raw_mut_ptr(&raw mut LAST),
                SharedArray::as_raw_mut_ptr(&raw mut P),
                SharedArray::as_raw_mut_ptr(&raw mut SLOT_P),
                &mut logits,
                &mut probs,
                &mut ids,
                &mut weights,
                &mut done,
            );
        }
    }

    /// The FFN norm and the router of one token in one launch: `x` the
    /// token's FFN input residual (`k` values), `gain` the norm's gain, `w`
    /// the router weight ([`N_EXPERT`] rows of `k` f32). Writes the q8_1 of
    /// the normed row into `q3 … d8` (the experts' input, one column) and the
    /// router's outputs as `qwen3moe_router_fused` at one token.
    ///
    /// Every block (256 threads) first runs `fused::norm_quant` on the row
    /// with the same cores in the same order: phase A is its sum of squares
    /// on the block's 256 threads (`rms_partial_sq`, the warp butterfly,
    /// `rms_warp_tree`, `rms_scale`), phase B its per-128-value geometry,
    /// warp `w` taking groups `w, w + 8, …`, each lane the normalized
    /// `(scale · gain) · x` of four consecutive values — kept in shared
    /// memory, and in block `g` for group `g` also quantized with
    /// `q8_1_quant_vals` (`norm_quant`'s tail, factored) and checked as
    /// `norm_quant` checks it: a non-finite normalized value or a scale that
    /// is not positive raises [`FaultSite::NormQuant`] on `fault`. So the
    /// q8_1 bytes are `norm_quant`'s, each group written by one block, and
    /// every block holds the normed row `norm_quant` stores in `y`. Then warp
    /// 0 of block `r` computes row `r`'s logit from that shared row with the
    /// fused entry's one-token body, and the ticket and the routing follow
    /// ([`publish_route`]). `k` a multiple of 128 and at most [`NORM_K`]
    /// (host-checked).
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
            k <= 2048,
            w.len() >= 128 * k,
            x.len() >= k,
            gain.len() >= k,
            q3.len() >= 64 * half_it,
            q4.len() >= 256 * quad_it,
            q6.len() >= 128 * half_it,
            s8.len() >= 8 * n_sb,
            d8.len() >= 2 * n_sb,
            logits.len() >= 128,
            probs.len() >= 128,
            ids.len() >= 8,
            weights.len() >= 8,
            done.len() >= 1
        )
    )]
    pub fn qwen3moe_router_norm(
        w: &[f32],
        x: &[f32],
        gain: &[f32],
        eps: f32,
        k: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut q3: DisjointSlice<u64>,
        mut q4: DisjointSlice<u32>,
        mut q6: DisjointSlice<u32>,
        mut s8: DisjointSlice<i32>,
        mut d8: DisjointSlice<f32>,
        mut logits: DisjointSlice<f32>,
        mut probs: DisjointSlice<f32>,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
        mut done: DisjointSlice<u32>,
        fault: FaultSink,
    ) {
        static mut WSUM: SharedArray<f32, RMS_WARPS> = SharedArray::UNINIT;
        static mut NORMED: SharedArray<f32, NORM_K> = SharedArray::UNINIT;
        static mut P: SharedArray<f32, P_LEN> = SharedArray::UNINIT;
        static mut SLOT_P: SharedArray<f32, SLOT_LEN> = SharedArray::UNINIT;
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let wi = tid / 32;
        let row = thread::blockIdx_x() as usize;
        let k = k as usize;
        let n_sb = n_sb as usize;

        // Phase A: `norm_quant`'s sum of squares on this block's threads.
        // SAFETY: WSUM is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        // Every access is below RMS_WARPS and ordered by `sync_threads`.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        // tid < RMS_THREADS: the block is RMS_THREADS wide (the const
        // assert beside FUSED_THREADS); x.len() >= k by the launch contract.
        let part = warp::reduce_sum_f32(rms_partial_sq(x, 0, k, tid));
        if lane == 0 {
            // SAFETY: wi < RMS_WARPS; one lane per warp writes its slot.
            unsafe { *ws.add(wi) = part };
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

        // Phase B: `norm_quant`'s per-group values into the shared row, and
        // group `row`'s q8_1 in this block. The group index is warp-uniform,
        // so the quantizer's collectives see a full warp.
        // SAFETY: NORMED is this block's own shared allocation of NORM_K
        // entries, the raw form as above; each entry below k <= NORM_K is
        // written by exactly one lane before the barrier that publishes it.
        let nr = unsafe { SharedArray::as_raw_mut_ptr(&raw mut NORMED) };
        let mut b = wi;
        while b < k / 128 {
            let vb = 128 * b + 4 * lane;
            // SAFETY: vb + 3 < k <= x.len(), gain.len() by the launch
            // contract.
            let (v0, v1, v2, v3, gn0, gn1, gn2, gn3) = unsafe {
                (
                    *x.get_unchecked(vb),
                    *x.get_unchecked(vb + 1),
                    *x.get_unchecked(vb + 2),
                    *x.get_unchecked(vb + 3),
                    *gain.get_unchecked(vb),
                    *gain.get_unchecked(vb + 1),
                    *gain.get_unchecked(vb + 2),
                    *gain.get_unchecked(vb + 3),
                )
            };
            // `norm_quant`'s (and `elem::rms_norm`'s) store expression.
            let nv = [
                (scale * gn0) * v0,
                (scale * gn1) * v1,
                (scale * gn2) * v2,
                (scale * gn3) * v3,
            ];
            // SAFETY: vb + 3 < k <= NORM_K; this lane owns the four entries.
            unsafe {
                *nr.add(vb) = nv[0];
                *nr.add(vb + 1) = nv[1];
                *nr.add(vb + 2) = nv[2];
                *nr.add(vb + 3) = nv[3];
            }
            if b == row {
                if !(quad_finite(nv) & (scale > 0.0)) {
                    fault.raise(FaultSite::NormQuant);
                }
                // SAFETY: one column (col 0 < 1), b < k/128 = 2·n_sb (the
                // host passes n_sb = k/256), the output bounds are the launch
                // contract's, the whole warp is here with the same `b`, and
                // `nv` holds values 128·b + 4·lane .. +3 of the column.
                unsafe {
                    q8_1_quant_vals(
                        nv, 0, b, n_sb, half_it, quad_it, lane, &mut q3, &mut q4, &mut q6, &mut s8,
                        &mut d8,
                    );
                }
            }
            b += FUSED_WARPS;
        }
        thread::sync_threads();

        if wi == 0 && row < N_EXPERT {
            // SAFETY: the shared row's k entries were written before the
            // barrier above and nothing writes them from here on.
            let xs = unsafe { core::slice::from_raw_parts(nr.cast_const(), k) };
            // SAFETY: row < N_EXPERT, so w.len() >= 128·k >= (row + 1)·k,
            // and xs holds k values; the launcher passes k a positive
            // multiple of 128; lane < 32.
            let f0 = unsafe { f32_lane_partial_1col_w32(w, xs, k as u32, row, lane) };
            let sums = gemv_lane_sums([f0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1);
            if lane == 0 {
                // SAFETY: row < N_EXPERT <= logits.len() (launch contract);
                // lane 0 of the row's warp is the slot's only writer.
                unsafe { store_cols(&mut logits, row, N_EXPERT, 1, &sums) };
            }
        }
        // SAFETY: as in `qwen3moe_router_fused`, at one token: LAST, P and
        // SLOT_P are this block's own and untouched until here, every thread
        // arrives converged, the output lengths are the launch contract's and
        // `done[0]` is zero before the launch.
        unsafe {
            publish_route(
                1,
                SharedArray::as_raw_mut_ptr(&raw mut LAST),
                SharedArray::as_raw_mut_ptr(&raw mut P),
                SharedArray::as_raw_mut_ptr(&raw mut SLOT_P),
                &mut logits,
                &mut probs,
                &mut ids,
                &mut weights,
                &mut done,
            );
        }
    }

    /// The router logits of `n_tok` tokens, a ubatch's first router launch:
    /// `w` the router weight ([`N_EXPERT`] rows of `k` f32), `x` the tokens'
    /// normed activations (`n_tok` columns of `k`). Block `b` takes tokens
    /// `8·c .. min(8·c + 8, n_tok)` for `c = b / ROW_GROUPS` and warp `w` of
    /// it expert row `8·(b % ROW_GROUPS) + w` of those tokens, with
    /// `qwen3moe_router_fused`'s row body at the chunk's width (the
    /// one-column walk for a chunk of one token, `f32_lane_partials`
    /// otherwise, then `gemv_lane_sums`), so each logit is bit for bit the
    /// fused launch's for its column. Writes `logits[t·128 + e]`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= 128 * k,
            x.len() >= n_tok * k,
            logits.len() >= 128 * n_tok
        )
    )]
    pub fn qwen3moe_router_logits(
        w: &[f32],
        x: &[f32],
        k: u32,
        n_tok: u32,
        mut logits: DisjointSlice<f32>,
    ) {
        let b = thread::blockIdx_x() as usize;
        let t0 = (b / ROW_GROUPS) * MAX_TOKENS;
        let n = n_tok as usize;
        if t0 >= n {
            return; // block-uniform
        }
        let m = (n - t0).min(MAX_TOKENS);
        let lane = warp::lane_id() as usize;
        let row = (b % ROW_GROUPS) * FUSED_WARPS + thread::threadIdx_x() as usize / 32;
        let kk = k as usize;
        // SAFETY: (t0 + m)·k <= n_tok·k <= x.len() by the launch contract.
        let xc = unsafe { x.get_unchecked(t0 * kk..(t0 + m) * kk) };
        let partials = if m == 1 {
            // SAFETY: row < ROW_GROUPS·FUSED_WARPS = N_EXPERT, so w.len() >=
            // 128·k >= (row + 1)·k; xc holds k values; the launcher passes k
            // a positive multiple of 32; lane < 32.
            let f0 = unsafe { f32_lane_partial_1col_w32(w, xc, k, row, lane) };
            [f0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        } else {
            f32_lane_partials(w, xc, k, row, m as u32, lane)
        };
        let sums = gemv_lane_sums(partials, m as u32);
        if lane == 0 {
            // SAFETY: 1 <= m <= 8 and (m − 1)·128 + t0·128 + row < (t0 +
            // m)·128 <= n_tok·128 <= logits.len() (launch contract); lane 0 of
            // the row's warp is the only writer of the chunk's slots of row.
            unsafe { store_cols(&mut logits, t0 * N_EXPERT + row, N_EXPERT, m, &sums) };
        }
    }

    /// The routing of `n_tok` tokens from their logits (`n_tok · 128`, as
    /// `qwen3moe_router_logits` writes them): warp `w` of block `b` routes
    /// token `8·b + w` with [`route_warp`] and writes its probabilities, ids
    /// and weights where the fused launch writes a token's.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            logits.len() >= 128 * n_tok,
            probs.len() >= 128 * n_tok,
            ids.len() >= 8 * n_tok,
            weights.len() >= 8 * n_tok
        )
    )]
    pub fn qwen3moe_router_route(
        logits: &[f32],
        n_tok: u32,
        mut probs: DisjointSlice<f32>,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
    ) {
        static mut P: SharedArray<f32, P_LEN> = SharedArray::UNINIT;
        static mut SLOT_P: SharedArray<f32, SLOT_LEN> = SharedArray::UNINIT;

        let wi = thread::threadIdx_x() as usize / 32;
        let t = thread::blockIdx_x() as usize * FUSED_WARPS + wi;
        if t >= n_tok as usize {
            return; // warp-uniform
        }
        let lane = warp::lane_id() as usize;
        let base = t * N_EXPERT;
        // SAFETY: base + lane + 96 < (t + 1)·128 <= n_tok·128 <= logits.len()
        // (launch contract).
        let v = unsafe {
            (
                *logits.get_unchecked(base + lane),
                *logits.get_unchecked(base + lane + 32),
                *logits.get_unchecked(base + lane + 64),
                *logits.get_unchecked(base + lane + 96),
            )
        };
        // SAFETY: P and SLOT_P are this block's own shared allocations (the
        // raw form reaches each `static mut` without a reference); warp wi <
        // FUSED_WARPS = MAX_TOKENS takes entries wi·128 .. and wi·8 .. of
        // them, which no other warp touches; the whole warp is here (the
        // guard is warp-uniform); token t's slots are inside probs, ids and
        // weights by the launch contract, written by this warp alone.
        unsafe {
            route_warp(
                v,
                SharedArray::as_raw_mut_ptr(&raw mut P).add(wi * N_EXPERT),
                SharedArray::as_raw_mut_ptr(&raw mut SLOT_P).add(wi * N_USED),
                t,
                &mut probs,
                &mut ids,
                &mut weights,
            );
        }
    }
}

/// Where one router launch leaves its results, allocated once and reused by
/// every launch and replay: per token the logits (the fused launch's) and
/// the probabilities, per slot the expert id and the weight, and the fused
/// launch's block ticket count, which it returns to zero. The count serves
/// one launch at a time, so launches that share a `RouterOut` must be
/// ordered on one stream.
pub struct RouterOut {
    pub logits: DeviceBuffer<f32>,
    pub probs: DeviceBuffer<f32>,
    pub ids: DeviceBuffer<u32>,
    pub weights: DeviceBuffer<f32>,
    done: DeviceBuffer<u32>,
    tokens: usize,
}

impl RouterOut {
    /// Allocate the buffers for one token, the ticket count zeroed.
    /// Load-time only.
    pub fn new(stream: &CudaStream) -> Result<RouterOut, GpuError> {
        RouterOut::with_tokens(stream, 1)
    }

    /// Allocate the buffers for up to `tokens` (1..=[`MAX_TOKENS`]) tokens
    /// per launch. Load-time only.
    pub fn with_tokens(stream: &CudaStream, tokens: usize) -> Result<RouterOut, GpuError> {
        if !(1..=MAX_TOKENS).contains(&tokens) {
            return Err(GpuError::shape(
                "qwen3moe::router::RouterOut::with_tokens",
                format!("{tokens} tokens, want 1..={MAX_TOKENS}"),
            ));
        }
        Ok(RouterOut {
            logits: DeviceBuffer::zeroed(stream, tokens * N_EXPERT)?,
            probs: DeviceBuffer::zeroed(stream, tokens * N_EXPERT)?,
            ids: DeviceBuffer::zeroed(stream, tokens * N_USED)?,
            weights: DeviceBuffer::zeroed(stream, tokens * N_USED)?,
            done: DeviceBuffer::zeroed(stream, 1)?,
            tokens,
        })
    }

    /// Allocate the buffers for a ubatch of up to `tokens` (1..=[`UBATCH`])
    /// tokens, the output of [`RouterKernels::enqueue_ubatch`]. The fused
    /// launch still refuses more than [`MAX_TOKENS`] tokens into them by its
    /// own contract. Load-time only.
    pub fn for_ubatch(stream: &CudaStream, tokens: usize) -> Result<RouterOut, GpuError> {
        if !(1..=UBATCH).contains(&tokens) {
            return Err(GpuError::shape(
                "qwen3moe::router::RouterOut::for_ubatch",
                format!("{tokens} tokens, want 1..={UBATCH}"),
            ));
        }
        Ok(RouterOut {
            logits: DeviceBuffer::zeroed(stream, tokens * N_EXPERT)?,
            probs: DeviceBuffer::zeroed(stream, tokens * N_EXPERT)?,
            ids: DeviceBuffer::zeroed(stream, tokens * N_USED)?,
            weights: DeviceBuffer::zeroed(stream, tokens * N_USED)?,
            done: DeviceBuffer::zeroed(stream, 1)?,
            tokens,
        })
    }

    /// The tokens one launch into these buffers may route.
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// The ticket count as it stands: zero between launches.
    pub fn tickets(&self, stream: &CudaStream) -> Result<u32, GpuError> {
        Ok(self.done.to_host_vec(stream)?[0])
    }

    /// Device bytes of the buffers.
    pub fn bytes(&self) -> usize {
        self.logits.num_bytes()
            + self.probs.num_bytes()
            + self.ids.num_bytes()
            + self.weights.num_bytes()
            + self.done.num_bytes()
    }
}

/// The loaded router module. Owns no stream: each enqueue takes the engine
/// stream, so its launch orders with the rest of the step and capture.
pub struct RouterKernels {
    module: qwen3moe_router_kernels::LoadedModule,
}

impl RouterKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RouterKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { qwen3moe_router_kernels::load(ctx)? };
        Ok(RouterKernels { module })
    }

    /// Enqueue the routing alone of one token over its `x` ([`N_EXPERT`]
    /// logits) into `out`'s first token. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        out: &mut RouterOut,
    ) -> Result<(), GpuError> {
        if x.len() < N_EXPERT {
            return Err(GpuError::shape(
                "qwen3moe::router::enqueue",
                format!("x.len() {} needs {N_EXPERT}", x.len()),
            ));
        }
        let prep =
            self.module
                .prepare_qwen3moe_router(LaunchConfig1D::new(1, ROUTER_THREADS, 0))?;
        self.module.qwen3moe_router(
            stream,
            &prep,
            x,
            &mut out.probs,
            &mut out.ids,
            &mut out.weights,
        )?;
        Ok(())
    }

    /// Enqueue the router of `m` (1..=`out.tokens()`) tokens: `w` the
    /// router weight as f32 ([`N_EXPERT`] rows of `k`, `k` a positive
    /// multiple of 32), `x` the tokens' `m` columns of `k` activations;
    /// results into `out`. Asynchronous, allocation-free, capturable.
    pub fn enqueue_fused(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        m: usize,
        out: &mut RouterOut,
    ) -> Result<(), GpuError> {
        let what = "qwen3moe::router::enqueue_fused";
        let k = w.cols();
        if w.rows() != N_EXPERT || k == 0 || !k.is_multiple_of(32) {
            return Err(GpuError::shape(
                what,
                format!(
                    "router weight is {} x {k}, want {N_EXPERT} rows of a positive multiple of 32",
                    w.rows()
                ),
            ));
        }
        if !(1..=out.tokens).contains(&m) || x.len() < m * k {
            return Err(GpuError::shape(
                what,
                format!(
                    "{m} tokens into buffers for {}, x.len() {} for {m} x {k}",
                    out.tokens,
                    x.len()
                ),
            ));
        }
        let k = launch_u32(what, "k", k)?;
        let m = launch_u32(what, "m", m)?;
        let grid = launch_u32(what, "grid", N_EXPERT)?;
        let prep = self
            .module
            .prepare_qwen3moe_router_fused(LaunchConfig1D::new(grid, FUSED_THREADS_U32, 0))?;
        self.module.qwen3moe_router_fused(
            stream,
            &prep,
            w.buf(),
            x,
            k,
            m,
            &mut out.logits,
            &mut out.probs,
            &mut out.ids,
            &mut out.weights,
            &mut out.done,
        )?;
        Ok(())
    }

    /// Enqueue the router of a ubatch of `n` (1..=`out.tokens()`) tokens in
    /// two launches, `qwen3moe_router_logits` then `qwen3moe_router_route`:
    /// `w` the router weight as f32 ([`N_EXPERT`] rows of `k`, `k` a
    /// positive multiple of 32), `x` the tokens' `n` columns of `k` normed
    /// activations; results into `out`, each token's the bits
    /// [`RouterKernels::enqueue_fused`] leaves for it. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_ubatch(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        n: usize,
        out: &mut RouterOut,
    ) -> Result<(), GpuError> {
        let what = "qwen3moe::router::enqueue_ubatch";
        let k = w.cols();
        if w.rows() != N_EXPERT || k == 0 || !k.is_multiple_of(32) {
            return Err(GpuError::shape(
                what,
                format!(
                    "router weight is {} x {k}, want {N_EXPERT} rows of a positive multiple of 32",
                    w.rows()
                ),
            ));
        }
        if !(1..=out.tokens).contains(&n) || x.len() < n * k {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n} tokens into buffers for {}, x.len() {} for {n} x {k}",
                    out.tokens,
                    x.len()
                ),
            ));
        }
        let chunks = n.div_ceil(MAX_TOKENS);
        let k = launch_u32(what, "k", k)?;
        let n_tok = launch_u32(what, "n", n)?;
        let grid = launch_u32(what, "logits grid", chunks * ROW_GROUPS)?;
        let prep = self
            .module
            .prepare_qwen3moe_router_logits(LaunchConfig1D::new(grid, FUSED_THREADS_U32, 0))?;
        self.module
            .qwen3moe_router_logits(stream, &prep, w.buf(), x, k, n_tok, &mut out.logits)?;
        let grid = launch_u32(what, "route grid", n.div_ceil(FUSED_WARPS))?;
        let prep = self
            .module
            .prepare_qwen3moe_router_route(LaunchConfig1D::new(grid, FUSED_THREADS_U32, 0))?;
        self.module.qwen3moe_router_route(
            stream,
            &prep,
            &out.logits,
            n_tok,
            &mut out.probs,
            &mut out.ids,
            &mut out.weights,
        )?;
        Ok(())
    }

    /// Enqueue the FFN norm and the router of one token in one launch
    /// (`qwen3moe_router_norm`): `x` the token's FFN input residual, `gain`
    /// the norm's gain (`w.cols()` values each), `eps` its epsilon; the q8_1
    /// of the normed row into `act` (one column of `w.cols()`) — the bytes
    /// `fused::norm_quant` writes — and the routing into `out`'s first token,
    /// as [`RouterKernels::enqueue_fused`] at one token. `fault` takes the
    /// norm's raise. Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher over the norm's and the router's buffers, the shape of the kernel's arguments"
    )]
    pub fn enqueue_norm_fused(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        gain: &DeviceBuffer<f32>,
        eps: f32,
        act: &mut Q8Act,
        fault: FaultSink,
        out: &mut RouterOut,
    ) -> Result<(), GpuError> {
        let what = "qwen3moe::router::enqueue_norm_fused";
        let k = w.cols();
        if w.rows() != N_EXPERT || k == 0 || !k.is_multiple_of(128) || k > NORM_K {
            return Err(GpuError::shape(
                what,
                format!(
                    "router weight is {} x {k}, want {N_EXPERT} rows of a positive multiple of 128 up to {NORM_K}",
                    w.rows()
                ),
            ));
        }
        if act.m() != 1 || act.k() != k || x.len() < k || gain.len() < k {
            return Err(GpuError::shape(
                what,
                format!(
                    "one column of {k}: act {} x {}, x.len() {}, gain.len() {}",
                    act.m(),
                    act.k(),
                    x.len(),
                    gain.len()
                ),
            ));
        }
        let n_sb = act.n_sb();
        let k = launch_u32(what, "k", k)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let grid = launch_u32(what, "grid", N_EXPERT)?;
        let prep = self
            .module
            .prepare_qwen3moe_router_norm(LaunchConfig1D::new(grid, FUSED_THREADS_U32, 0))?;
        self.module.qwen3moe_router_norm(
            stream,
            &prep,
            w.buf(),
            x,
            gain,
            eps,
            k,
            n_sb,
            n_sb.div_ceil(2),
            n_sb.div_ceil(4),
            &mut act.q3,
            &mut act.q4,
            &mut act.q6,
            &mut act.s8,
            &mut act.d8,
            &mut out.logits,
            &mut out.probs,
            &mut out.ids,
            &mut out.weights,
            &mut out.done,
            fault,
        )?;
        Ok(())
    }
}
