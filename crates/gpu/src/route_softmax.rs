//! The softmax router: softmax over a token's 128 expert logits, the top `k`
//! by probability (`k` the file's `expert_used_count`, a launch argument of
//! one to 32), and their weights renormalized to sum to one
//! (`norm_topk_prob`). Every model whose router has this body loads these
//! entries through [`RouterKernels`], over [`RouterBufs`] cut for its
//! [`RouterDims`]; a model's own code holds none of it. [`RouterBufs`] and the
//! norm-fused row body ([`norm_row_body`]) also serve the gated router that
//! adds a shared expert's slot (`arch::qwen3moe::router::gated`).
//!
//! Four entries share one warp-level routing body, [`route_warp`]:
//! `qwen3moe_router_fused` — the router's logit gemv with the routing fused
//! into it, one launch for up to [`MAX_TOKENS`] tokens; `qwen3moe_router_norm`,
//! the decode chain's at one token — the same launch with `fused::norm_quant`
//! run in every block first (its q8_1 bytes out, its normed row kept in shared
//! memory for the gemv); `qwen3moe_router_norm_m`, the chain's at two to
//! [`MAX_TOKENS`] tokens — the same fold over the tokens' rows one after
//! another ([`norm_row_body`], the one-token entry's body); and
//! `qwen3moe_router_route`, the routing alone over given logits, one warp per
//! token. A ubatch's logits come from
//! `qwen3moe_router_logits`: the fused entry's lane sums over a ubatch of up
//! to [`RouterDims::ubatch`] tokens as register tiles, the wide
//! F32 product's body (`crate::gemm`'s `f32_tile_body`, which keeps every
//! lane sum and the butterfly of the row walk) over the router's rows, so
//! the two ubatch launches leave each token's logits, ids and weights bit
//! for bit what the fused launch leaves for it. The
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
//! - the top `k` by descending probability with ties toward the smaller
//!   expert id;
//! - the chosen probabilities summed in f64 in slot order, the sum rounded
//!   once to f32, each weight the f32 divide of its probability by it.
//!
//! No silent routing. The softmax is defined on finite logits: then the max
//! is finite, the winning expert's exp is 1, the sum lies in [1, 128], and
//! every probability, every round's winner and every weight is finite. A
//! token with any non-finite logit — NaN, an overflow to ±inf from a finite
//! row, a non-finite router weight — has no defined routing (a NaN would
//! turn every probability NaN and the top `k` into copies of expert 0; a
//! −inf would read as a probability of 0). Every entry raises
//! [`FaultSite::Router`] on its fault sink for such a token, writes its
//! probabilities and weights NaN, and leaves its ids as they stand, which are
//! in range: this module writes only ids below [`WIDTH`] and allocates
//! them zeroed.

use crate::cores::l2_prefetch;
use crate::elem::{RMS_THREADS, RMS_WARPS, rms_partial_sq, rms_scale, rms_warp_tree};
use crate::fault::{FaultSink, FaultSite, quad_finite};
use crate::gemm::{
    F32Rows, F32T_THREADS_U32, GEMM_MAX_SLOTS, STAGE_FLOATS, STAGES, f32_tile_body, tile_launch,
};
use crate::q8_1_quant_vals;
use crate::q8f32::{f32_lane_partial_1col_w32, f32_lane_partials, gemv_lane_sums};
use crate::route_core::take;
use crate::tensor::{DeviceTensor, Q8Act};
use crate::{GpuError, launch_u32, store_cols};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, threadfence, warp,
};
use cuda_host::cuda_module;
use model::arch::models::shape::{self, RouterBody, RouterInst, rules};
use std::sync::Arc;

pub use chain::router::RouterDims;

/// Tokens one fused launch routes: the f32 gemv's column bound, and one
/// warp of the routing block per token.
pub const MAX_TOKENS: usize = 8;

/// Threads per fused block: eight warps — the first computes the block's
/// expert row, and in the routing block warp `t` routes token `t`.
pub(crate) const FUSED_THREADS: usize = 256;
pub(crate) const FUSED_THREADS_U32: u32 = FUSED_THREADS as u32;
pub(crate) const FUSED_WARPS: usize = FUSED_THREADS / 32;

/// Probabilities each lane of the routing warp owns: experts `lane + 32 j`,
/// four named scalars.
const PER_LANE: usize = 4;

/// Experts the body routes over: the router weight's rows, and logits and
/// probabilities a token.
const WIDTH: usize = 32 * PER_LANE;

/// This body's instance row: its width, the softmax renormalized over the
/// top k, `k` a launch argument of one to 32 (one pick a lane). A file's
/// shape selects it through [`RouterDims::of`].
pub const ROW: RouterInst = shape::router_row(RouterBody::Qwen3moe, 32 * PER_LANE as u32);
const _: () = assert!(shape::router_row_is(
    ROW,
    rules::SOFTMAX_NORM,
    false,
    (1, shape::LANE_PICKS)
));

/// The fused block's shared scratch: one routing warp's entries per token.
const P_LEN: usize = MAX_TOKENS * WIDTH;

const _: () = assert!(FUSED_THREADS_U32 as usize == FUSED_THREADS);
// The norm-fused entry runs `norm_quant`'s sum of squares on its whole
// block: the same threads, loads and tree as the norm's first RMS_THREADS.
const _: () = assert!(FUSED_THREADS == RMS_THREADS && FUSED_WARPS == RMS_WARPS);

/// The widest row the norm-fused entry normalizes into shared memory: the
/// qwen3moe hidden width.
pub const NORM_K: usize = 2048;
// `qwen3moe_router_norm`'s launch contract spells the bound out.
const _: () = assert!(NORM_K == 2048);
// Block `t · (k / 128) + g` quantizes row `t`'s 128-value group `g`, so
// every group of every token a launch routes needs a block.
const _: () = assert!(NORM_K / 128 <= WIDTH && MAX_TOKENS * (NORM_K / 128) <= WIDTH);
// The m-row entry's routing scratch is its shared row once the last gemv
// has read it.
const _: () = assert!(P_LEN <= NORM_K);
// The routing warp's lanes hold four logits each, as named scalars.
const _: () = assert!(PER_LANE == 4);
// The routing block has a warp for every token.
const _: () = assert!(MAX_TOKENS <= FUSED_WARPS);
// A ubatch routing block routes one token per warp into the MAX_TOKENS
// entries of P.
const _: () = assert!(MAX_TOKENS == FUSED_WARPS);
// The entries' launch contracts spell WIDTH and MAX_TOKENS out as 128 and 8,
// and bound `used` by the lane picks.
const _: () = assert!(WIDTH == 128 && MAX_TOKENS == 8 && shape::LANE_PICKS == 32);

/// One token's routing by one warp, the module doc's contract: lane `L`
/// brings the token's logits of experts `L + 32 j` in `v`; `p` is the
/// token's [`WIDTH`] shared entries. Writes `probs[t·128 + e]` for every
/// expert, then `ids[t·used + s]` and `weights[t·used + s]` in rank order.
/// The selection runs `used` rounds of a butterfly argmax under
/// [`take`]`::<false>` (the order the serial scan's strict `>` over
/// ascending ids realizes, seed `(−inf, 0)`), the winning lane marking its
/// entry taken; lane 0 writes each round's id. Every lane reads each round's
/// winning probability and sums them in f64 in round order, and lane `s`
/// keeps round `s`'s, so lane `s` writes slot `s`'s weight: the picks live
/// one a lane, and nothing is sized by `used`.
///
/// A token with a non-finite logit (the module doc) raises
/// [`FaultSite::Router`] on `fault`, gets NaN probabilities and weights, and
/// keeps its ids; the warp returns before the softmax.
///
/// # Safety
///
/// All 32 lanes of the warp call it, converged, with the same `p`, `used`
/// and `t`; `p` points to [`WIDTH`] entries of the block's shared memory
/// that no other warp touches; `1 <= used <= 32`; `probs.len() >= (t +
/// 1)·128`, and `ids.len()`, `weights.len() >= (t + 1)·used`, those slots
/// written by this warp alone.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn route_warp(
    v: (f32, f32, f32, f32),
    p: *mut f32,
    used: usize,
    t: usize,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    fault: FaultSink,
) {
    let lane = warp::lane_id() as usize;
    let (v0, v1, v2, v3) = v;
    let pb = t * WIDTH + lane;
    // Warp-uniform: every lane leaves the ballot with the same mask.
    if warp::ballot(!quad_finite([v0, v1, v2, v3])) != 0 {
        // SAFETY: lane + 96 < WIDTH — the token's probs slots (this fn's
        // contract); entries `lane + 32 j` are this lane's own.
        unsafe {
            *probs.get_unchecked_mut(pb) = f32::NAN;
            *probs.get_unchecked_mut(pb + 32) = f32::NAN;
            *probs.get_unchecked_mut(pb + 64) = f32::NAN;
            *probs.get_unchecked_mut(pb + 96) = f32::NAN;
        }
        if lane == 0 {
            fault.raise(FaultSite::Router);
            let mut s = 0usize;
            while s < used {
                // SAFETY: s < used, inside the token's weights slots (this
                // fn's contract); lane 0 is the only writer.
                unsafe { *weights.get_unchecked_mut(t * used + s) = f32::NAN };
                s += 1;
            }
        }
        return;
    }
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
    // SAFETY: lane + 96 < WIDTH — inside the token's shared entries and its
    // probs slots (this fn's contract); entries `lane + 32 j` are this
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
    // Every lane reads each round's winner, whichever lane wrote it.
    warp::sync_mask(u32::MAX);

    let mut taken = 0u32;
    // The chosen probabilities' f64 sum in slot order (the same on every
    // lane), and this lane's own slot's probability.
    let mut chosen = 0.0f64;
    let mut mine = 0.0f32;
    let mut s = 0usize;
    while s < used {
        let mut bv = f32::NEG_INFINITY;
        let mut bi = 0u32;
        let mut j = 0usize;
        while j < PER_LANE {
            if (taken >> j) & 1 == 0 {
                let ej = lane + 32 * j;
                // SAFETY: ej < 32·PER_LANE = WIDTH, this lane's own entry.
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
        // SAFETY: bi < WIDTH — inside the token's entries, published by the
        // warp sync above and not written since. The probability is read
        // from the winner's entry rather than from the comparison, so one no
        // comparison won is carried through unchanged.
        let pw = unsafe { *p.add(bi as usize) };
        chosen += f64::from(pw);
        if lane == s {
            mine = pw;
        }
        if lane == 0 {
            // SAFETY: s < used, inside the token's ids slots (this fn's
            // contract); lane 0 is the only writer.
            unsafe { *ids.get_unchecked_mut(t * used + s) = bi };
        }
        s += 1;
    }
    let chosen = chosen as f32;
    if lane < used {
        // SAFETY: lane < used, inside the token's weights slots (this fn's
        // contract); lane `lane` is slot `lane`'s only writer.
        unsafe { *weights.get_unchecked_mut(t * used + lane) = mine / chosen };
    }
}

/// The fused entries' tail, after each block's rows are stored: every thread
/// fences and the block meets at a barrier, so its rows are visible
/// device-wide before thread 0 draws the block's ticket; the block that draws
/// the last ticket returns the count to zero, and its warp `t < m` reads
/// token `t`'s logits back (volatile loads: this block's L1 never held other
/// blocks' rows, and a volatile load does not ask it) and runs
/// [`route_warp`], which raises a refused token on `fault`.
///
/// # Safety
///
/// Every thread of the block calls it, converged, with the same arguments;
/// `last` and `p` point to 1 and `MAX_TOKENS · WIDTH` entries of the
/// block's shared memory that nothing else in the block touches from here
/// on; `1 <= m <= MAX_TOKENS <= FUSED_WARPS`; `1 <= used <= 32`; `logits`,
/// `probs` hold `128 · m` and `ids`, `weights` `used · m` entries, and
/// `done[0]` is this launch's ticket count, zero before it.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn publish_route(
    m: usize,
    used: usize,
    last: *mut u32,
    p: *mut f32,
    logits: &mut DisjointSlice<f32>,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    done: &mut DisjointSlice<u32>,
    fault: FaultSink,
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
    let base = wi * WIDTH;
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
    // SAFETY: wi < m <= MAX_TOKENS, so token wi's WIDTH entries are inside
    // `p` and no other warp's; the whole warp is here (`wi` is
    // warp-uniform), converged past the barrier; 1 <= used <= 32 and its
    // slots are inside probs, ids and weights (this fn's contract).
    unsafe {
        route_warp(v, p.add(base), used, wi, probs, ids, weights, fault);
    }
}

/// The norm-fused entries' body for token `t`, up to the routing, at `rows`
/// router rows, one block per router row: `fused::norm_quant` on row `t` of
/// `x` with the same cores in the same order, in every block — phase A its
/// sum of squares on the block's threads (`rms_partial_sq`, the warp
/// butterfly, `rms_warp_tree`, `rms_scale`), phase B its per-128-value
/// geometry, warp `w` taking groups `w, w + 8, …`, each lane the normalized
/// `(scale · gain) · x` of four consecutive values — kept in the shared row
/// `nr`, and group `g` also quantized into column `t` of the q8_1 planes by
/// block `t · (k / 128) + g` with `q8_1_quant_vals` (`norm_quant`'s tail,
/// factored) and checked as `norm_quant` checks it: a non-finite normalized
/// value or a scale that is not positive raises [`FaultSite::NormQuant`] on
/// `fault`, and its group is refused as `norm_quant` refuses it — NaN values
/// in the shared row, a NaN scale in `d8`. So the q8_1 bytes are
/// `norm_quant`'s, each group written by one block, and every block holds
/// the normed row `norm_quant` stores in `y`. Then warp 0 of block `r`
/// computes row `r`'s logit of token `t` from that shared row with the fused
/// entry's one-token sum and stores it at `logits[t·rows + r]`.
///
/// A later token's call may follow in the same launch: its phase A writes
/// `ws` only past this call's second barrier, which every warp reaches after
/// reading `ws`, and its phase B writes `nr` only past its own first
/// barrier, which warp 0 reaches after this call's gemv.
///
/// # Safety
///
/// Every thread of the block calls it, converged, with the same `t`; the
/// block is [`FUSED_THREADS`] wide and the grid at least `(t + 1) · (k /
/// 128)` blocks; `ws` and `nr` are [`RMS_WARPS`] and [`NORM_K`] entries of the
/// block's shared memory that nothing else touches; `k` a positive multiple
/// of 128 up to [`NORM_K`], `n_sb = k / 256`, `half_it` and `quad_it` its
/// halves and quarters rounded up; `w.len() >= rows·k`, `x.len() >= (t +
/// 1)·k`, `gain.len() >= k`, the q8_1 planes at least `t + 1` columns of
/// `k`, and `logits.len() >= (t + 1)·rows`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub(crate) unsafe fn norm_row_body<const PREFETCH: bool>(
    w: &[f32],
    x: &[f32],
    gain: &[f32],
    eps: f32,
    k: u32,
    n_sb: u32,
    half_it: u32,
    quad_it: u32,
    t: usize,
    rows: usize,
    ws: *mut f32,
    nr: *mut f32,
    q3: &mut DisjointSlice<u64>,
    q4: &mut DisjointSlice<u32>,
    q6: &mut DisjointSlice<u32>,
    s8: &mut DisjointSlice<i32>,
    d8: &mut DisjointSlice<f32>,
    logits: &mut DisjointSlice<f32>,
    fault: FaultSink,
) {
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let wi = tid / 32;
    let row = thread::blockIdx_x() as usize;
    let k = k as usize;
    let n_sb = n_sb as usize;
    let base = t * k;
    // The row's weights stream into L2 while the norm runs.
    if PREFETCH && t == 0 && wi == 0 && row < rows {
        l2_prefetch(w, 4 * row * k, 4 * k, lane);
    }

    // Phase A: `norm_quant`'s sum of squares on this block's threads.
    // SAFETY: base + k = (t + 1)·k <= x.len() by this fn's contract; tid <
    // RMS_THREADS, the block being FUSED_THREADS = RMS_THREADS wide (the
    // const assert beside FUSED_THREADS).
    let part = warp::reduce_sum_f32(unsafe { rms_partial_sq(x, base, k, tid) });
    if lane == 0 {
        // SAFETY: wi < RMS_WARPS; one lane per warp writes its slot.
        unsafe { *ws.add(wi) = part };
    }
    thread::sync_threads();
    // SAFETY: every slot was written above and is visible past the barrier.
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

    // Phase B: `norm_quant`'s per-group values into the shared row, and group
    // `g` of column `t` quantized in block `t · (k / 128) + g`. The group
    // index is warp-uniform, so the quantizer's collectives see a full warp.
    let groups = k / 128;
    let mut b = wi;
    while b < groups {
        let vb = 128 * b + 4 * lane;
        let xb = base + vb;
        // SAFETY: xb + 3 < base + k <= x.len() and vb + 3 < k <= gain.len()
        // by this fn's contract.
        let (v0, v1, v2, v3, gn0, gn1, gn2, gn3) = unsafe {
            (
                *x.get_unchecked(xb),
                *x.get_unchecked(xb + 1),
                *x.get_unchecked(xb + 2),
                *x.get_unchecked(xb + 3),
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
        // `norm_quant`'s refusal: one warp owns the group, so the ballot
        // refuses all of it.
        let bad = !(quad_finite(nv) & (scale > 0.0));
        let refused = warp::ballot(bad) != 0;
        let nv = if refused { [f32::NAN; 4] } else { nv };
        // SAFETY: vb + 3 < k <= NORM_K; this lane owns the four entries.
        unsafe {
            *nr.add(vb) = nv[0];
            *nr.add(vb + 1) = nv[1];
            *nr.add(vb + 2) = nv[2];
            *nr.add(vb + 3) = nv[3];
        }
        if t * groups + b == row {
            if bad {
                fault.raise(FaultSite::NormQuant);
            }
            // A refused group reaches the quantizer as NaN, which refuses it
            // in turn: a NaN scale, zero codes.
            // SAFETY: column t is inside the planes and b < k/128 = 2·n_sb
            // (this fn's contract), the whole warp is here with the same `b`,
            // and `nv` holds values 128·b + 4·lane .. +3 of column t.
            unsafe {
                q8_1_quant_vals(nv, t, b, n_sb, half_it, quad_it, lane, q3, q4, q6, s8, d8);
            }
        }
        b += FUSED_WARPS;
    }
    thread::sync_threads();

    if wi == 0 && row < rows {
        // SAFETY: the shared row's k entries were written before the barrier
        // above and nothing writes them before the next call's first barrier,
        // which this warp reaches after this gemv.
        let xs = unsafe { core::slice::from_raw_parts(nr.cast_const(), k) };
        // SAFETY: row < rows, so w.len() >= rows·k >= (row + 1)·k, and xs
        // holds k values; k a positive multiple of 128; lane < 32.
        let f0 = unsafe { f32_lane_partial_1col_w32(w, xs, k as u32, row, lane) };
        let sums = gemv_lane_sums([f0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1);
        if lane == 0 {
            // SAFETY: t·rows + row < (t + 1)·rows <= logits.len() (this fn's
            // contract); lane 0 of the row's warp is the slot's only writer.
            unsafe { store_cols(logits, t * rows + row, rows, 1, sums) };
        }
    }
}

#[cuda_module]
mod qwen3moe_router_kernels {
    use super::*;

    /// The router of `m_cols` tokens: `w` the router weight ([`WIDTH`] rows
    /// of `k` f32), `x` the tokens' normed activations (`m_cols` columns of
    /// `k`), `used` the experts a token keeps. Writes `logits[t·128 + e]`,
    /// `probs[t·128 + e]`, `ids[t·used + s]` and `weights[t·used + s]`.
    /// `done[0]` is the block ticket count: zero before the launch, zero
    /// again after it.
    ///
    /// Grid [`WIDTH`] blocks of 256; warp 0 of block `r` owns row `r`
    /// with `f32_gemv`'s row body, and the other warps meet the barriers
    /// only. After its row every thread fences and the block meets at a
    /// barrier, so the row is visible device-wide before thread 0 draws the
    /// block's ticket. The
    /// block that draws the last ticket returns the count to zero, and its
    /// warp `t < m_cols` reads token `t`'s logits back (volatile loads: this
    /// block's L1 never held other blocks' rows, and a volatile load does not
    /// ask it) and runs [`route_warp`]; a token with a non-finite logit
    /// raises [`FaultSite::Router`] on `fault`.
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
            used >= 1,
            used <= 32,
            w.len() >= 128 * k,
            x.len() >= m_cols * k,
            logits.len() >= 128 * m_cols,
            probs.len() >= 128 * m_cols,
            ids.len() >= used * m_cols,
            weights.len() >= used * m_cols,
            done.len() >= 1
        )
    )]
    pub fn qwen3moe_router_fused(
        w: &[f32],
        x: &[f32],
        k: u32,
        m_cols: u32,
        used: u32,
        mut logits: DisjointSlice<f32>,
        mut probs: DisjointSlice<f32>,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
        mut done: DisjointSlice<u32>,
        fault: FaultSink,
    ) {
        static mut P: SharedArray<f32, P_LEN> = SharedArray::UNINIT;
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let wi = tid / 32;
        let m = m_cols as usize;
        // The row guard is warp-uniform, so the warp tree sees a full warp;
        // the other warps, and a block past the rows, still meet the barrier
        // and the ticket.
        let row = thread::blockIdx_x() as usize;
        if wi == 0 && row < WIDTH {
            let partials = if m == 1 {
                // SAFETY: row < WIDTH, so w.len() >= 128·k >= (row + 1)·k,
                // and x.len() >= k, by the launch contract; the launcher
                // passes k a positive multiple of 32; lane < 32.
                let f0 = unsafe { f32_lane_partial_1col_w32(w, x, k, row, lane) };
                [f0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            } else {
                // SAFETY: as above, with x.len() >= m_cols·k and m_cols in
                // 1..=8 by the launch contract.
                unsafe { f32_lane_partials(w, x, k, row, m_cols, lane) }
            };
            let sums = gemv_lane_sums(partials, m_cols);
            if lane == 0 {
                // SAFETY: row < WIDTH and 1 <= m <= 8, so the slots c·128 +
                // row (c < m) lie inside logits (launch contract); lane 0 of
                // the row's warp is their only writer.
                unsafe { store_cols(&mut logits, row, WIDTH, m, sums) };
            }
        }
        // SAFETY: LAST and P are this block's own shared allocations of 1
        // and P_LEN entries (the raw form reaches each `static mut` without a
        // reference), untouched until here; every thread arrives converged
        // with the launch's arguments; 1 <= m <= 8, 1 <= used <= 32 and the
        // output lengths are the launch contract's; `done[0]` is zero before
        // the launch.
        unsafe {
            publish_route(
                m,
                used as usize,
                SharedArray::as_raw_mut_ptr(&raw mut LAST),
                SharedArray::as_raw_mut_ptr(&raw mut P),
                &mut logits,
                &mut probs,
                &mut ids,
                &mut weights,
                &mut done,
                fault,
            );
        }
    }

    /// The FFN norm and the router of one token in one launch: `x` the
    /// token's FFN input residual (`k` values), `gain` the norm's gain, `w`
    /// the router weight ([`WIDTH`] rows of `k` f32), `used` the experts a
    /// token keeps. Writes the q8_1 of
    /// the normed row into `q3 … d8` (the experts' input, one column) and the
    /// router's outputs as `qwen3moe_router_fused` at one token.
    ///
    /// Every block (256 threads) first runs `fused::norm_quant` on the row
    /// with the same cores in the same order, keeps the normed row in shared
    /// memory, quantizes its group `g` in block `g` and computes row `r`'s
    /// logit in block `r` ([`norm_row_body`] at token 0: the q8_1 bytes are
    /// `norm_quant`'s, a refused group raises [`FaultSite::NormQuant`] on
    /// `fault`); then the ticket and the routing follow ([`publish_route`]),
    /// a non-finite logit raising [`FaultSite::Router`] on the same `fault`.
    /// `k` a multiple of 128 and at most [`NORM_K`] (host-checked).
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
            used >= 1,
            used <= 32,
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
            ids.len() >= used,
            weights.len() >= used,
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
        used: u32,
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
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;

        // SAFETY: WSUM and NORMED are this block's own shared allocations (the
        // raw form reaches each `static mut` without a reference), touched by
        // nothing else; every thread arrives converged with token 0, and the
        // grid is WIDTH >= k / 128 blocks of 256 (the launcher's, and the
        // const assert beside NORM_K); the launcher passes k a positive
        // multiple of 128 up to NORM_K, n_sb = k/256 and its halves and
        // quarters rounded up; the lengths are the launch contract's.
        unsafe {
            norm_row_body::<false>(
                w,
                x,
                gain,
                eps,
                k,
                n_sb,
                half_it,
                quad_it,
                0,
                WIDTH,
                SharedArray::as_raw_mut_ptr(&raw mut WSUM),
                SharedArray::as_raw_mut_ptr(&raw mut NORMED),
                &mut q3,
                &mut q4,
                &mut q6,
                &mut s8,
                &mut d8,
                &mut logits,
                fault,
            );
        }
        // SAFETY: as in `qwen3moe_router_fused`, at one token: LAST and P
        // are this block's own and untouched until here, every thread arrives
        // converged, 1 <= used <= 32, the output lengths are the launch
        // contract's and `done[0]` is zero before the launch.
        unsafe {
            publish_route(
                1,
                used as usize,
                SharedArray::as_raw_mut_ptr(&raw mut LAST),
                SharedArray::as_raw_mut_ptr(&raw mut P),
                &mut logits,
                &mut probs,
                &mut ids,
                &mut weights,
                &mut done,
                fault,
            );
        }
    }

    /// `qwen3moe_router_norm` with warp 0 of each block asking L2 for its router row
    /// first ([`l2_prefetch`]): the entry an integrated device launches.
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
            used >= 1,
            used <= 32,
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
            ids.len() >= used,
            weights.len() >= used,
            done.len() >= 1
        )
    )]
    pub fn qwen3moe_router_norm_prefetch(
        w: &[f32],
        x: &[f32],
        gain: &[f32],
        eps: f32,
        k: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        used: u32,
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
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;

        // SAFETY: WSUM and NORMED are this block's own shared allocations (the
        // raw form reaches each `static mut` without a reference), touched by
        // nothing else; every thread arrives converged with token 0, and the
        // grid is WIDTH >= k / 128 blocks of 256 (the launcher's, and the
        // const assert beside NORM_K); the launcher passes k a positive
        // multiple of 128 up to NORM_K, n_sb = k/256 and its halves and
        // quarters rounded up; the lengths are the launch contract's.
        unsafe {
            norm_row_body::<true>(
                w,
                x,
                gain,
                eps,
                k,
                n_sb,
                half_it,
                quad_it,
                0,
                WIDTH,
                SharedArray::as_raw_mut_ptr(&raw mut WSUM),
                SharedArray::as_raw_mut_ptr(&raw mut NORMED),
                &mut q3,
                &mut q4,
                &mut q6,
                &mut s8,
                &mut d8,
                &mut logits,
                fault,
            );
        }
        // SAFETY: as in `qwen3moe_router_fused`, at one token: LAST and P
        // are this block's own and untouched until here, every thread arrives
        // converged, 1 <= used <= 32, the output lengths are the launch
        // contract's and `done[0]` is zero before the launch.
        unsafe {
            publish_route(
                1,
                used as usize,
                SharedArray::as_raw_mut_ptr(&raw mut LAST),
                SharedArray::as_raw_mut_ptr(&raw mut P),
                &mut logits,
                &mut probs,
                &mut ids,
                &mut weights,
                &mut done,
                fault,
            );
        }
    }

    /// The FFN norm and the router of `m_cols` tokens in one launch:
    /// `fused::norm_quant` then `qwen3moe_router_fused` at `m_cols` tokens,
    /// bit for bit, each token's row as those two launches leave it. `x` the
    /// tokens' FFN input residuals (`m_cols` rows of `k`), `gain` the norm's
    /// gain, `w` the router weight ([`WIDTH`] rows of `k` f32), `used` the
    /// experts a token keeps. Writes the q8_1 of the normed rows into `q3 …
    /// d8` (`m_cols` columns) and the router's outputs as
    /// `qwen3moe_router_fused` at `m_cols` tokens; `fault` takes the norm's
    /// raise and the router's.
    ///
    /// Grid [`WIDTH`] blocks of 256: `qwen3moe_router_norm`'s body
    /// ([`norm_row_body`]) for each token in turn — its row normed into
    /// shared memory, its groups quantized by blocks `t · (k / 128) ..`,
    /// warp 0 of block `r` its logit of row `r` — then the ticket and the
    /// routing ([`publish_route`]), whose scratch is the shared row, read by
    /// then. `k` a multiple of 128 and at most [`NORM_K`] (host-checked).
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
            k <= 2048,
            used >= 1,
            used <= 32,
            w.len() >= 128 * k,
            x.len() >= m_cols * k,
            gain.len() >= k,
            q3.len() >= m_cols * 64 * half_it,
            q4.len() >= m_cols * 256 * quad_it,
            q6.len() >= m_cols * 128 * half_it,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb,
            logits.len() >= 128 * m_cols,
            probs.len() >= 128 * m_cols,
            ids.len() >= used * m_cols,
            weights.len() >= used * m_cols,
            done.len() >= 1
        )
    )]
    pub fn qwen3moe_router_norm_m(
        w: &[f32],
        x: &[f32],
        gain: &[f32],
        eps: f32,
        k: u32,
        m_cols: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        used: u32,
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
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;

        let mut t = 0usize;
        while t < m_cols as usize {
            // SAFETY: WSUM and NORMED are this block's own shared allocations
            // (the raw form reaches each `static mut` without a reference),
            // touched by nothing else; every thread arrives converged with
            // the same t < m_cols <= 8, and the grid is WIDTH >= m_cols ·
            // (k / 128) blocks of 256 (the launcher's, and the const assert
            // beside NORM_K); the launcher passes k a positive multiple of 128
            // up to NORM_K, n_sb = k/256 and its halves and quarters rounded
            // up; the lengths are the launch contract's at m_cols tokens.
            unsafe {
                norm_row_body::<false>(
                    w,
                    x,
                    gain,
                    eps,
                    k,
                    n_sb,
                    half_it,
                    quad_it,
                    t,
                    WIDTH,
                    SharedArray::as_raw_mut_ptr(&raw mut WSUM),
                    SharedArray::as_raw_mut_ptr(&raw mut NORMED),
                    &mut q3,
                    &mut q4,
                    &mut q6,
                    &mut s8,
                    &mut d8,
                    &mut logits,
                    fault,
                );
            }
            t += 1;
        }
        // SAFETY: as in `qwen3moe_router_fused` at m_cols tokens: LAST is
        // this block's own and untouched until here, and the routing's
        // scratch is NORMED's first P_LEN <= NORM_K entries (const assert),
        // which warp 0's last gemv has read before publish_route's first
        // barrier and nothing touches after it; every thread arrives
        // converged, 1 <= m_cols <= 8, 1 <= used <= 32, the output lengths
        // are the launch contract's and `done[0]` is zero before the launch.
        unsafe {
            publish_route(
                m_cols as usize,
                used as usize,
                SharedArray::as_raw_mut_ptr(&raw mut LAST),
                SharedArray::as_raw_mut_ptr(&raw mut NORMED),
                &mut logits,
                &mut probs,
                &mut ids,
                &mut weights,
                &mut done,
                fault,
            );
        }
    }

    /// `qwen3moe_router_norm_m` with warp 0 of each block asking L2 for its router row
    /// first ([`l2_prefetch`]): the entry an integrated device launches.
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
            k <= 2048,
            used >= 1,
            used <= 32,
            w.len() >= 128 * k,
            x.len() >= m_cols * k,
            gain.len() >= k,
            q3.len() >= m_cols * 64 * half_it,
            q4.len() >= m_cols * 256 * quad_it,
            q6.len() >= m_cols * 128 * half_it,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb,
            logits.len() >= 128 * m_cols,
            probs.len() >= 128 * m_cols,
            ids.len() >= used * m_cols,
            weights.len() >= used * m_cols,
            done.len() >= 1
        )
    )]
    pub fn qwen3moe_router_norm_m_prefetch(
        w: &[f32],
        x: &[f32],
        gain: &[f32],
        eps: f32,
        k: u32,
        m_cols: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        used: u32,
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
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;

        let mut t = 0usize;
        while t < m_cols as usize {
            // SAFETY: WSUM and NORMED are this block's own shared allocations
            // (the raw form reaches each `static mut` without a reference),
            // touched by nothing else; every thread arrives converged with
            // the same t < m_cols <= 8, and the grid is WIDTH >= m_cols ·
            // (k / 128) blocks of 256 (the launcher's, and the const assert
            // beside NORM_K); the launcher passes k a positive multiple of 128
            // up to NORM_K, n_sb = k/256 and its halves and quarters rounded
            // up; the lengths are the launch contract's at m_cols tokens.
            unsafe {
                norm_row_body::<true>(
                    w,
                    x,
                    gain,
                    eps,
                    k,
                    n_sb,
                    half_it,
                    quad_it,
                    t,
                    WIDTH,
                    SharedArray::as_raw_mut_ptr(&raw mut WSUM),
                    SharedArray::as_raw_mut_ptr(&raw mut NORMED),
                    &mut q3,
                    &mut q4,
                    &mut q6,
                    &mut s8,
                    &mut d8,
                    &mut logits,
                    fault,
                );
            }
            t += 1;
        }
        // SAFETY: as in `qwen3moe_router_fused` at m_cols tokens: LAST is
        // this block's own and untouched until here, and the routing's
        // scratch is NORMED's first P_LEN <= NORM_K entries (const assert),
        // which warp 0's last gemv has read before publish_route's first
        // barrier and nothing touches after it; every thread arrives
        // converged, 1 <= m_cols <= 8, 1 <= used <= 32, the output lengths
        // are the launch contract's and `done[0]` is zero before the launch.
        unsafe {
            publish_route(
                m_cols as usize,
                used as usize,
                SharedArray::as_raw_mut_ptr(&raw mut LAST),
                SharedArray::as_raw_mut_ptr(&raw mut NORMED),
                &mut logits,
                &mut probs,
                &mut ids,
                &mut weights,
                &mut done,
                fault,
            );
        }
    }

    /// The router logits of `n_tok` tokens, a ubatch's first router launch:
    /// `w` the router weight ([`WIDTH`] rows of `k` f32), `x` the tokens'
    /// normed activations (`n_tok` columns of `k`): the wide F32 product's
    /// tiles ([`f32_tile_body`]) over the router's rows, `⌈n_tok / 32⌉ ·
    /// WIDTH / 32` blocks. Its lane sums take the fused entry's terms `w[row·k
    /// + 32·it + l] · x[token·k + 32·it + l]` in the fused entry's order and
    /// its butterfly is the fused entry's, so each logit is bit for bit the
    /// fused launch's for its column. Writes `logits[t·128 + e]`. `k` a
    /// positive multiple of 64, `w` and `x` 16-byte aligned (host-checked).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            n_tok >= 1,
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
        static mut STAGE: SharedArray<f32, { STAGES * STAGE_FLOATS }, 16> = SharedArray::UNINIT;
        // SAFETY: STAGE is this block's own shared allocation of STAGES ·
        // STAGE_FLOATS values, 16-byte aligned; the raw form reaches the
        // `static mut` without a reference.
        let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE) };
        // SAFETY: every thread of the 256-wide block calls it converged; the
        // host sized the grid (`tile_launch`), checked k and both alignments;
        // the lengths are the launch contract's.
        unsafe { f32_tile_body(F32Rows(w), x, WIDTH as u32, k, n_tok, &mut logits, sh) };
    }

    /// The routing of `n_tok` tokens from their logits (`n_tok · 128`, as
    /// `qwen3moe_router_logits` writes them): warp `w` of block `b` routes
    /// token `8·b + w` with [`route_warp`] at `used` experts a token and
    /// writes its probabilities, ids and weights where the fused launch
    /// writes a token's; a token with a non-finite logit raises
    /// [`FaultSite::Router`] on `fault`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            used >= 1,
            used <= 32,
            logits.len() >= 128 * n_tok,
            probs.len() >= 128 * n_tok,
            ids.len() >= used * n_tok,
            weights.len() >= used * n_tok
        )
    )]
    pub fn qwen3moe_router_route(
        logits: &[f32],
        n_tok: u32,
        used: u32,
        mut probs: DisjointSlice<f32>,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
        fault: FaultSink,
    ) {
        static mut P: SharedArray<f32, P_LEN> = SharedArray::UNINIT;

        let wi = thread::threadIdx_x() as usize / 32;
        let t = thread::blockIdx_x() as usize * FUSED_WARPS + wi;
        if t >= n_tok as usize {
            return; // warp-uniform
        }
        let lane = warp::lane_id() as usize;
        let base = t * WIDTH;
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
        // SAFETY: P is this block's own shared allocation (the raw form
        // reaches the `static mut` without a reference); warp wi <
        // FUSED_WARPS = MAX_TOKENS takes entries wi·128 .. of it, which no
        // other warp touches; the whole warp is here (the guard is
        // warp-uniform); 1 <= used <= 32, and token t's slots are inside
        // probs, ids and weights by the launch contract, written by this warp
        // alone.
        unsafe {
            route_warp(
                v,
                SharedArray::as_raw_mut_ptr(&raw mut P).add(wi * WIDTH),
                used as usize,
                t,
                &mut probs,
                &mut ids,
                &mut weights,
                fault,
            );
        }
    }
}

/// Where one router launch leaves its results, allocated once and reused by
/// every launch and replay: per token the logits (the fused launches') and
/// the probabilities, per slot the expert id and the weight, and the fused
/// launches' block ticket count, which they return to zero. The count serves
/// one launch at a time, so launches that share one must be ordered on one
/// stream. `tokens` is how many tokens the buffers hold, and only that: each
/// launcher checks its own launch's bound beside it. The dims set the widths
/// and the instance; a router's launchers take only buffers of their own
/// body and pass `used` from them.
pub struct RouterBufs {
    pub logits: DeviceBuffer<f32>,
    pub probs: DeviceBuffer<f32>,
    pub ids: DeviceBuffer<u32>,
    pub weights: DeviceBuffer<f32>,
    pub(crate) done: DeviceBuffer<u32>,
    tokens: usize,
    dims: RouterDims,
}

/// The router's buffers, the plain body's and the gated body's alike.
pub type RouterOut = RouterBufs;

impl RouterBufs {
    /// Allocate the buffers for up to `tokens` (1..=[`MAX_TOKENS`]) tokens
    /// per launch. Load-time only.
    pub fn with_tokens(
        stream: &CudaStream,
        dims: RouterDims,
        tokens: usize,
    ) -> Result<Self, GpuError> {
        let what = "route_softmax::RouterBufs::with_tokens";
        Self::alloc(stream, dims, tokens, MAX_TOKENS, what)
    }

    /// Allocate the buffers for a ubatch of up to `tokens`
    /// (1..=[`RouterDims::ubatch`]) tokens, the output of the body's
    /// `enqueue_ubatch`. A fused launch into them still routes at most
    /// [`MAX_TOKENS`] tokens and is refused by name past that. Load-time
    /// only.
    pub fn for_ubatch(
        stream: &CudaStream,
        dims: RouterDims,
        tokens: usize,
    ) -> Result<Self, GpuError> {
        let what = "route_softmax::RouterBufs::for_ubatch";
        Self::alloc(stream, dims, tokens, dims.ubatch(GEMM_MAX_SLOTS), what)
    }

    /// The buffers for `tokens` (1..=`most`) tokens, the ticket count zeroed;
    /// a count outside that is refused as `what`.
    fn alloc(
        stream: &CudaStream,
        dims: RouterDims,
        tokens: usize,
        most: usize,
        what: &'static str,
    ) -> Result<Self, GpuError> {
        if !(1..=most).contains(&tokens) {
            return Err(GpuError::shape(
                what,
                format!("{tokens} tokens, want 1..={most}"),
            ));
        }
        Ok(RouterBufs {
            logits: DeviceBuffer::zeroed(stream, tokens * dims.logits())?,
            probs: DeviceBuffer::zeroed(stream, tokens * dims.experts())?,
            ids: DeviceBuffer::zeroed(stream, tokens * dims.slots())?,
            weights: DeviceBuffer::zeroed(stream, tokens * dims.slots())?,
            done: DeviceBuffer::zeroed(stream, 1)?,
            tokens,
            dims,
        })
    }

    /// The tokens these buffers hold: a ubatch or routing launch may take all
    /// of them, a fused launch at most [`MAX_TOKENS`].
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// The shape the buffers were cut for.
    pub fn dims(&self) -> RouterDims {
        self.dims
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

    /// `used` as the launch argument, after the buffers' body is `body`
    /// (and, when `per_lane` is given, its width): a launcher's refusal of
    /// buffers cut for another router, named `what`.
    pub(crate) fn used_for(
        &self,
        what: &'static str,
        body: RouterBody,
        per_lane: Option<u32>,
    ) -> Result<u32, GpuError> {
        let inst = self.dims.inst();
        if inst.body != body || per_lane.is_some_and(|w| w != inst.per_lane()) {
            return Err(GpuError::shape(
                what,
                format!(
                    "buffers cut for {}; this launch runs another router",
                    inst.at
                ),
            ));
        }
        launch_u32(what, "used", self.dims.used())
    }
}

/// The loaded router module. Owns no stream: each enqueue takes the engine
/// stream, so its launch orders with the rest of the step and capture.
pub struct RouterKernels {
    module: qwen3moe_router_kernels::LoadedModule,
    /// [`crate::row_prefetch`] of the context's device.
    row_prefetch: bool,
}

impl RouterKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RouterKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(qwen3moe_router_kernels, ctx)? };
        Ok(RouterKernels {
            module,
            row_prefetch: crate::row_prefetch(ctx)?,
        })
    }

    /// Enqueue the router of `m` tokens, 1..=[`MAX_TOKENS`] and at most
    /// `out.tokens()`: `w` the router weight as f32 ([`WIDTH`] rows of
    /// `k`, `k` a positive multiple of 32), `x` the tokens' `m` columns of
    /// `k` activations; results into `out`, a token with a non-finite logit
    /// raising [`FaultSite::Router`] on `fault`. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_fused(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        m: usize,
        fault: FaultSink,
        out: &mut RouterOut,
    ) -> Result<(), GpuError> {
        let what = "route_softmax::enqueue_fused";
        let used = out.used_for(what, RouterBody::Qwen3moe, None)?;
        let k = w.cols();
        if w.rows() != WIDTH || k == 0 || !k.is_multiple_of(32) {
            return Err(GpuError::shape(
                what,
                format!(
                    "router weight is {} x {k}, want {WIDTH} rows of a positive multiple of 32",
                    w.rows()
                ),
            ));
        }
        if !(1..=MAX_TOKENS.min(out.tokens)).contains(&m) || x.len() < m * k {
            return Err(GpuError::shape(
                what,
                format!(
                    "{m} tokens: one launch routes 1..={MAX_TOKENS}, into buffers for {}; x.len() \
                     {} for {m} x {k}",
                    out.tokens,
                    x.len()
                ),
            ));
        }
        let k = launch_u32(what, "k", k)?;
        let m = launch_u32(what, "m", m)?;
        let grid = launch_u32(what, "grid", WIDTH)?;
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
            used,
            &mut out.logits,
            &mut out.probs,
            &mut out.ids,
            &mut out.weights,
            &mut out.done,
            fault,
        )?;
        Ok(())
    }

    /// Enqueue the router of a ubatch of `n` (1..=`out.tokens()`) tokens in
    /// two launches, `qwen3moe_router_logits` then `qwen3moe_router_route`
    /// ([`RouterKernels::enqueue_route`]):
    /// `w` the router weight as f32 ([`WIDTH`] rows of `k`, `k` a
    /// positive multiple of 64), `x` the tokens' `n` columns of `k` normed
    /// activations, both 16-byte aligned (the logits launch copies them in
    /// 16-byte pieces); results into `out`, each token's the bits
    /// [`RouterKernels::enqueue_fused`] leaves for it, its refusal raised on
    /// `fault` as there. Asynchronous, allocation-free, capturable.
    pub fn enqueue_ubatch(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        n: usize,
        fault: FaultSink,
        out: &mut RouterOut,
    ) -> Result<(), GpuError> {
        let what = "route_softmax::enqueue_ubatch";
        out.used_for(what, RouterBody::Qwen3moe, None)?;
        let k = w.cols();
        if w.rows() != WIDTH || n > out.tokens {
            return Err(GpuError::shape(
                what,
                format!(
                    "router weight is {} x {k}, want {WIDTH} rows; {n} tokens into buffers for {}",
                    w.rows(),
                    out.tokens
                ),
            ));
        }
        let w_at = w.buf().cu_deviceptr();
        if !w_at.is_multiple_of(16) {
            return Err(GpuError::shape(
                what,
                format!("router weight at {w_at:#x} must be 16-byte aligned"),
            ));
        }
        let t = tile_launch(what, WIDTH, k, x, n, &out.logits)?;
        let prep = self
            .module
            .prepare_qwen3moe_router_logits(LaunchConfig1D::new(t.grid, F32T_THREADS_U32, 0))?;
        self.module
            .qwen3moe_router_logits(stream, &prep, w.buf(), x, t.k, t.n, &mut out.logits)?;
        self.enqueue_route(stream, n, fault, out)
    }

    /// Enqueue the routing alone of the first `n` (1..=`out.tokens()`)
    /// tokens' logits in `out.logits` (`qwen3moe_router_route`, warp `w` of
    /// block `b` routing token `8·b + w`): each token's probabilities, ids
    /// and weights into `out` where the fused launch writes a token's, a
    /// token with a non-finite logit raising [`FaultSite::Router`] on
    /// `fault`. The second launch of [`RouterKernels::enqueue_ubatch`].
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_route(
        &self,
        stream: &CudaStream,
        n: usize,
        fault: FaultSink,
        out: &mut RouterOut,
    ) -> Result<(), GpuError> {
        let what = "route_softmax::enqueue_route";
        let used = out.used_for(what, RouterBody::Qwen3moe, None)?;
        if !(1..=out.tokens).contains(&n) {
            return Err(GpuError::shape(
                what,
                format!("{n} tokens into buffers for {}", out.tokens),
            ));
        }
        let n_tok = launch_u32(what, "n", n)?;
        let grid = launch_u32(what, "grid", n.div_ceil(FUSED_WARPS))?;
        let prep = self
            .module
            .prepare_qwen3moe_router_route(LaunchConfig1D::new(grid, FUSED_THREADS_U32, 0))?;
        self.module.qwen3moe_router_route(
            stream,
            &prep,
            &out.logits,
            n_tok,
            used,
            &mut out.probs,
            &mut out.ids,
            &mut out.weights,
            fault,
        )?;
        Ok(())
    }

    /// Enqueue the FFN norm and the router of one token in one launch
    /// (`qwen3moe_router_norm`): `x` the token's FFN input residual, `gain`
    /// the norm's gain (`w.cols()` values each), `eps` its epsilon; the q8_1
    /// of the normed row into `act` (one column of `w.cols()`) — the bytes
    /// `fused::norm_quant` writes — and the routing into `out`'s first token,
    /// as [`RouterKernels::enqueue_fused`] at one token. `fault` takes the
    /// norm's raise and the router's. Asynchronous, allocation-free,
    /// capturable.
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
        let what = "route_softmax::enqueue_norm_fused";
        let used = out.used_for(what, RouterBody::Qwen3moe, None)?;
        let k = w.cols();
        if w.rows() != WIDTH || k == 0 || !k.is_multiple_of(128) || k > NORM_K {
            return Err(GpuError::shape(
                what,
                format!(
                    "router weight is {} x {k}, want {WIDTH} rows of a positive multiple of 128 up to {NORM_K}",
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
        let grid = launch_u32(what, "grid", WIDTH)?;
        if self.row_prefetch {
            let prep = self
                .module
                .prepare_qwen3moe_router_norm_prefetch(LaunchConfig1D::new(
                    grid,
                    FUSED_THREADS_U32,
                    0,
                ))?;
            self.module.qwen3moe_router_norm_prefetch(
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
                used,
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
        } else {
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
                used,
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
        }
        Ok(())
    }

    /// Enqueue the FFN norm and the router of `act.m()` tokens
    /// (1..=[`MAX_TOKENS`], at most `out.tokens()`) in one launch
    /// (`qwen3moe_router_norm_m`): `x` the tokens' FFN input residuals
    /// (`act.m()` rows of `w.cols()`), `gain` the norm's gain, `eps` its
    /// epsilon; the q8_1 of the normed rows into `act` — the bytes
    /// `fused::norm_quant` writes — and the routing into `out`, as
    /// [`RouterKernels::enqueue_fused`] at `act.m()` tokens. `fault` takes
    /// the norm's raise and the router's. Asynchronous, allocation-free,
    /// capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher over the norm's and the router's buffers, the shape of the kernel's arguments"
    )]
    pub fn enqueue_norm_fused_m(
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
        let what = "route_softmax::enqueue_norm_fused_m";
        let used = out.used_for(what, RouterBody::Qwen3moe, None)?;
        let k = w.cols();
        if w.rows() != WIDTH || k == 0 || !k.is_multiple_of(128) || k > NORM_K {
            return Err(GpuError::shape(
                what,
                format!(
                    "router weight is {} x {k}, want {WIDTH} rows of a positive multiple of 128 up to {NORM_K}",
                    w.rows()
                ),
            ));
        }
        let m = act.m();
        if !(1..=MAX_TOKENS.min(out.tokens)).contains(&m)
            || act.k() != k
            || x.len() < m * k
            || gain.len() < k
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "{m} rows of {k}: one launch routes 1..={MAX_TOKENS}, into buffers for {}; act \
                     {m} x {}, x.len() {}, gain.len() {}",
                    out.tokens,
                    act.k(),
                    x.len(),
                    gain.len()
                ),
            ));
        }
        let n_sb = act.n_sb();
        let k = launch_u32(what, "k", k)?;
        let m = launch_u32(what, "m", m)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let grid = launch_u32(what, "grid", WIDTH)?;
        if self.row_prefetch {
            let prep = self
                .module
                .prepare_qwen3moe_router_norm_m_prefetch(LaunchConfig1D::new(
                    grid,
                    FUSED_THREADS_U32,
                    0,
                ))?;
            self.module.qwen3moe_router_norm_m_prefetch(
                stream,
                &prep,
                w.buf(),
                x,
                gain,
                eps,
                k,
                m,
                n_sb,
                n_sb.div_ceil(2),
                n_sb.div_ceil(4),
                used,
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
        } else {
            let prep = self
                .module
                .prepare_qwen3moe_router_norm_m(LaunchConfig1D::new(grid, FUSED_THREADS_U32, 0))?;
            self.module.qwen3moe_router_norm_m(
                stream,
                &prep,
                w.buf(),
                x,
                gain,
                eps,
                k,
                m,
                n_sb,
                n_sb.div_ceil(2),
                n_sb.div_ceil(4),
                used,
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
        }
        Ok(())
    }
}
