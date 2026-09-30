//! The qwen3moe router: softmax over the 128 experts' logits, the top `k`
//! by probability (`k` the file's `expert_used_count`, a launch argument of
//! one to 32), and their weights renormalized to sum to one
//! (`norm_topk_prob`).
//!
//! Three entries share one warp-level routing body, [`route_warp`]:
//! `qwen3moe_router_fused` — the router's logit gemv with the routing fused
//! into it, one launch for up to [`MAX_TOKENS`] tokens; `qwen3moe_router_norm`,
//! the chain's at one token — the same launch with `fused::norm_quant` run
//! in every block first (its q8_1 bytes out, its normed row kept in shared
//! memory for the gemv); and `qwen3moe_router_route`, the routing alone over
//! given logits, one warp per token. A ubatch's logits come from
//! `qwen3moe_router_logits`: the fused entry's lane sums over a ubatch of up
//! to [`UBATCH`] tokens as register tiles — a block of sixteen warps covers
//! 32 tokens and 32 expert rows, each warp eight rows by eight tokens, the
//! rows' and the tokens' values staged in shared memory — so the two ubatch
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

use super::ubatch::UBATCH;
use crate::elem::{RMS_THREADS, RMS_WARPS, rms_partial_sq, rms_scale, rms_warp_tree};
use crate::fault::{FaultSink, FaultSite, quad_finite};
use crate::gemm::GEMM_MAX_SLOTS;
use crate::q8_1_quant_vals;
use crate::q8f32::{
    TILE, f32_lane_partial_1col_w32, f32_lane_partials, f32_tile_chunk, gemv_lane_sums,
};
use crate::route_core::{sigmoid, take};
use crate::tensor::{DeviceTensor, Q8Act};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, threadfence, warp,
};
use cuda_host::cuda_module;
use model::arch::models::shape::{self, MoeShape, RouterBody, RouterInst, rules};
use std::sync::Arc;

/// Tokens one fused launch routes: the f32 gemv's column bound, and one
/// warp of the routing block per token.
pub const MAX_TOKENS: usize = 8;

/// Tokens and expert rows one `qwen3moe_router_logits` block covers.
const LOGITS_TOKENS: usize = 32;
const LOGITS_ROWS: usize = 32;

/// Threads per logits block: sixteen warps, [`ROW_TILES`] row groups by
/// [`TOKEN_TILES`] token groups, each warp a [`TILE`]-row × [`TILE`]-token
/// tile of the block.
const LOGITS_THREADS: usize = 512;
const LOGITS_THREADS_U32: u32 = LOGITS_THREADS as u32;
const LOGITS_WARPS: usize = LOGITS_THREADS / 32;
const ROW_TILES: usize = LOGITS_ROWS / TILE;
const TOKEN_TILES: usize = LOGITS_TOKENS / TILE;

/// Logits blocks per token span: the expert rows cut into [`LOGITS_ROWS`].
const ROW_BLOCKS: usize = WIDTH / LOGITS_ROWS;

/// 32-value chunks one logits stage holds of every row and token, and the
/// values of one such line.
const STAGE_CHUNKS: usize = 2;
const LINE: usize = 32 * STAGE_CHUNKS;

/// Logits stages in shared memory: the one the warps read and two being
/// copied in.
const STAGES: usize = 3;

/// One logits stage: every row's line, then every token's.
const STAGE_FLOATS: usize = (LOGITS_ROWS + LOGITS_TOKENS) * LINE;

/// 16-byte copies per line.
const LINE_COPIES: usize = LINE / 4;

/// Threads per fused block: eight warps — the first computes the block's
/// expert row, and in the routing block warp `t` routes token `t`.
const FUSED_THREADS: usize = 256;
const FUSED_THREADS_U32: u32 = FUSED_THREADS as u32;
const FUSED_WARPS: usize = FUSED_THREADS / 32;

/// Probabilities each lane of the routing warp owns: experts `lane + 32 j`,
/// four named scalars.
const PER_LANE: usize = 4;

/// Experts the body routes over: the router weight's rows, and logits and
/// probabilities a token.
const WIDTH: usize = 32 * PER_LANE;

/// This body's instance row: its width, the softmax renormalized over the
/// top k, `k` a launch argument of one to 32 (one pick a lane). A file's
/// shape selects it through [`RouterDims::of`].
pub const ROW: RouterInst = shape::router_row(RouterBody::Qwen3moe, PER_LANE as u32);
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
// Block `g` quantizes the row's 128-value group `g`, so every group needs a
// block.
const _: () = assert!(NORM_K / 128 <= WIDTH);
// The routing warp's lanes hold four logits each, as named scalars.
const _: () = assert!(PER_LANE == 4);
// The routing block has a warp for every token.
const _: () = assert!(MAX_TOKENS <= FUSED_WARPS);
// A ubatch routing block routes one token per warp into the MAX_TOKENS
// entries of P.
const _: () = assert!(MAX_TOKENS == FUSED_WARPS);
// The warp tiles cover a logits block, and a token span's blocks the rows.
const _: () = assert!(ROW_TILES * TILE == LOGITS_ROWS && TOKEN_TILES * TILE == LOGITS_TOKENS);
const _: () =
    assert!(ROW_TILES * TOKEN_TILES == LOGITS_WARPS && LOGITS_WARPS * 32 == LOGITS_THREADS);
const _: () = assert!(ROW_BLOCKS * LOGITS_ROWS == WIDTH);
// Each logits thread copies one piece of a row line and the same piece of a
// token line of every stage.
const _: () = assert!(LOGITS_ROWS * LINE_COPIES == LOGITS_THREADS && LOGITS_TOKENS == LOGITS_ROWS);
// The kernel reads a stage's chunks as two spelled-out tile steps, and its
// wait leaves STAGES − 2 = 1 group in flight.
const _: () = assert!(STAGE_CHUNKS == 2 && STAGES == 3);
// The stages fit the 48 KiB of static shared memory a block may declare.
const _: () = assert!(STAGES * STAGE_FLOATS * size_of::<f32>() <= 48 * 1024);
// `qwen3moe_router_logits`' launch contract spells the block out.
const _: () = assert!(LOGITS_THREADS == 512);
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

/// Lane 0's stores of one token's [`TILE`] row sums of a logits tile:
/// `y[base + i] = sums[i]`, one store per slot with a constant index.
///
/// # Safety
///
/// `base + TILE <= y.len()`, and no other thread writes those slots.
#[inline(always)]
unsafe fn store_rows(y: &mut DisjointSlice<f32>, base: usize, sums: &[f32; TILE]) {
    let [s0, s1, s2, s3, s4, s5, s6, s7] = *sums;
    // SAFETY: each slot base + i, i < TILE, lies inside y and belongs to this
    // thread alone by this fn's contract.
    unsafe {
        *y.get_unchecked_mut(base) = s0;
        *y.get_unchecked_mut(base + 1) = s1;
        *y.get_unchecked_mut(base + 2) = s2;
        *y.get_unchecked_mut(base + 3) = s3;
        *y.get_unchecked_mut(base + 4) = s4;
        *y.get_unchecked_mut(base + 5) = s5;
        *y.get_unchecked_mut(base + 6) = s6;
        *y.get_unchecked_mut(base + 7) = s7;
    }
}

/// Lane 0's stores of the first `live` of one token's [`TILE`] row sums:
/// [`store_rows`] with every slot from `live` on left alone, each store
/// guarded by a constant index.
///
/// # Safety
///
/// `live <= TILE`, `base + live <= y.len()`, and no other thread writes
/// those slots.
#[inline(always)]
unsafe fn store_rows_live(
    y: &mut DisjointSlice<f32>,
    base: usize,
    sums: &[f32; TILE],
    live: usize,
) {
    let [s0, s1, s2, s3, s4, s5, s6, s7] = *sums;
    // SAFETY: each store is guarded by i < live, so its slot base + i lies
    // inside y and belongs to this thread alone by this fn's contract.
    unsafe {
        if live > 0 {
            *y.get_unchecked_mut(base) = s0;
        }
        if live > 1 {
            *y.get_unchecked_mut(base + 1) = s1;
        }
        if live > 2 {
            *y.get_unchecked_mut(base + 2) = s2;
        }
        if live > 3 {
            *y.get_unchecked_mut(base + 3) = s3;
        }
        if live > 4 {
            *y.get_unchecked_mut(base + 4) = s4;
        }
        if live > 5 {
            *y.get_unchecked_mut(base + 5) = s5;
        }
        if live > 6 {
            *y.get_unchecked_mut(base + 6) = s6;
        }
        if live > 7 {
            *y.get_unchecked_mut(base + 7) = s7;
        }
    }
}

/// One token's routing by one warp for a router whose last row is a shared
/// expert's gate ([`gated`]'s contract): `32 · PER_LANE` experts, lane `L`
/// owning experts `L + 32 j`. On entry `p` holds the token's expert logits,
/// entry `e` expert `e`'s, each written by the lane that owns it, and every
/// lane brings the gate's logit in `g`. The warp turns `p` into the
/// probabilities in place and writes `probs[t·n + e]` for every expert
/// `e < n`, then per slot `ids[t·(used + 1) + s]` and
/// `weights[t·(used + 1) + s]`: slots `s < used` in rank order as
/// [`route_warp`] writes them (one pick a lane), slot `used` the shared
/// expert, `(n, sigmoid(g))`. The
/// logits live in shared memory rather than in a register array: each lane
/// walks its `PER_LANE` entries in loops that a register array would put in
/// a local depot.
///
/// A token with a non-finite logit among the `n + 1` raises
/// [`FaultSite::Router`] on `fault`, gets NaN probabilities and weights in
/// every slot, and keeps its ids; the warp returns before the softmax.
///
/// # Safety
///
/// All 32 lanes of the warp call it, converged, with the same `g`, `p`,
/// `used` and `t`; `p` points to `32 · PER_LANE` entries of the block's
/// shared memory that no other warp touches, written as above; `1 <= used
/// <= 32`; `probs.len() >= (t + 1)·n`, and `ids.len()`,
/// `weights.len() >= (t + 1)·(used + 1)`, those slots written by this warp
/// alone; `PER_LANE <= 32`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn route_warp_gated<const PER_LANE: usize>(
    g: f32,
    p: *mut f32,
    used: usize,
    t: usize,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    fault: FaultSink,
) {
    let n = 32 * PER_LANE;
    let slots = used + 1;
    let lane = warp::lane_id() as usize;
    let pb = t * n + lane;
    let mut finite = g.is_finite();
    let mut j = 0usize;
    while j < PER_LANE {
        // SAFETY: lane + 32 j < n, this lane's own entry (this fn's
        // contract).
        let x = unsafe { *p.add(lane + 32 * j) };
        // `&`, not `&&`: every test runs.
        finite &= x.is_finite();
        j += 1;
    }
    // Warp-uniform: every lane leaves the ballot with the same mask.
    if warp::ballot(!finite) != 0 {
        let mut j = 0usize;
        while j < PER_LANE {
            // SAFETY: lane + 32 j < n — the token's probs slots (this fn's
            // contract); entries `lane + 32 j` are this lane's own.
            unsafe { *probs.get_unchecked_mut(pb + 32 * j) = f32::NAN };
            j += 1;
        }
        if lane == 0 {
            fault.raise(FaultSite::Router);
            let mut s = 0usize;
            while s < slots {
                // SAFETY: s < used + 1, inside the token's weights slots
                // (this fn's contract); lane 0 is the only writer.
                unsafe { *weights.get_unchecked_mut(t * slots + s) = f32::NAN };
                s += 1;
            }
        }
        return;
    }
    let mut mx = f32::NEG_INFINITY;
    let mut j = 0usize;
    while j < PER_LANE {
        // SAFETY: as in the finiteness walk.
        mx = mx.max(unsafe { *p.add(lane + 32 * j) });
        j += 1;
    }
    let mut off = 16u32;
    while off > 0 {
        mx = mx.max(warp::shuffle_xor_f32(mx, off));
        off >>= 1;
    }
    // Each entry becomes its exp, summed in f64 in ascending `j`.
    let mut sum = 0.0f64;
    let mut j = 0usize;
    while j < PER_LANE {
        // SAFETY: as in the finiteness walk; this lane is the entry's only
        // reader and writer.
        unsafe {
            let at = p.add(lane + 32 * j);
            let e = (*at - mx).exp();
            *at = e;
            sum += f64::from(e);
        }
        j += 1;
    }
    let mut off = 16u32;
    while off > 0 {
        // Both lanes of a pair add the same two values, so every lane leaves
        // with the same sum.
        sum += warp::shuffle_xor_f64(sum, off);
        off >>= 1;
    }
    let inv = sum as f32;
    let mut j = 0usize;
    while j < PER_LANE {
        // SAFETY: lane + 32 j < n — inside the token's shared entries and
        // its probs slots (this fn's contract); entries `lane + 32 j` are
        // this lane's own.
        unsafe {
            let at = p.add(lane + 32 * j);
            let pj = *at / inv;
            *at = pj;
            *probs.get_unchecked_mut(pb + 32 * j) = pj;
        }
        j += 1;
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
                // SAFETY: ej < 32·PER_LANE = n, this lane's own entry.
                let pv = unsafe { *p.add(ej) };
                if take::<false>(pv, ej as u32, bv, bi) {
                    bv = pv;
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
        // SAFETY: bi < n — inside the token's entries, published by the warp
        // sync above and not written since. The probability is read from the
        // winner's entry, as `route_warp` reads it.
        let pw = unsafe { *p.add(bi as usize) };
        chosen += f64::from(pw);
        if lane == s {
            mine = pw;
        }
        if lane == 0 {
            // SAFETY: s < used, inside the token's ids slots (this fn's
            // contract); lane 0 is the only writer.
            unsafe { *ids.get_unchecked_mut(t * slots + s) = bi };
        }
        s += 1;
    }
    let chosen = chosen as f32;
    if lane < used {
        // SAFETY: lane < used, inside the token's weights slots (this fn's
        // contract); lane `lane` is slot `lane`'s only writer.
        unsafe { *weights.get_unchecked_mut(t * slots + lane) = mine / chosen };
    }
    if lane == 0 {
        // SAFETY: slot `used` is the token's last ids and weights slot (this
        // fn's contract); lane 0 is its only writer.
        unsafe {
            *ids.get_unchecked_mut(t * slots + used) = n as u32;
            *weights.get_unchecked_mut(t * slots + used) = sigmoid(g);
        }
    }
}

/// [`publish_route`] for a gated router of `32 · PER_LANE` experts and
/// `used` routed slots: token `t`'s `n + 1` logits at `logits[t·(n + 1)
/// ..]`, the last the gate's, and [`route_warp_gated`] in place of
/// [`route_warp`].
///
/// # Safety
///
/// [`publish_route`]'s, with `p` holding `m · n` entries, `logits` `(n + 1)
/// · m`, `probs` `n · m`, and `ids`, `weights` `(used + 1) · m`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn publish_route_gated<const PER_LANE: usize>(
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

    let n = 32 * PER_LANE;
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
    // published. Volatile loads, as `publish_route` reads them, each lane's
    // logits into its own entries of token wi's `p`.
    let base = wi * (n + 1);
    let lp = logits.as_mut_ptr();
    let pt = p.wrapping_add(wi * n);
    let mut j = 0usize;
    while j < PER_LANE {
        // SAFETY: base + lane + 32 j < base + n < m·(n + 1) <= logits.len(),
        // and wi·n + lane + 32 j < m·n, inside `p` (this fn's contract); the
        // entry is this lane's own.
        unsafe { *pt.add(lane + 32 * j) = core::ptr::read_volatile(lp.add(base + lane + 32 * j)) };
        j += 1;
    }
    // SAFETY: base + n < m·(n + 1) <= logits.len(); every lane reads the
    // gate's logit.
    let g = unsafe { core::ptr::read_volatile(lp.add(base + n)) };
    // SAFETY: wi < m, so token wi's n entries are inside `p` and no other
    // warp's, its expert logits written above by their lanes; the whole warp
    // is here (`wi` is warp-uniform), converged past the barrier; 1 <= used
    // <= 32 and its slots are inside probs, ids and weights (this fn's
    // contract).
    unsafe {
        route_warp_gated::<PER_LANE>(g, pt, used, wi, probs, ids, weights, fault);
    }
}

/// The gated router's fused launch body, `qwen3moe_router_fused`'s at `n + 1
/// = 32 · PER_LANE + 1` rows: warp 0 of block `r` computes row `r`'s logit of
/// each of the `m_cols` tokens with `f32_gemv`'s row body and stores it at
/// `logits[c·(n + 1) + r]`, then [`publish_route_gated`].
///
/// # Safety
///
/// Every thread of the block calls it, converged; the grid is `n + 1`
/// blocks of [`FUSED_THREADS`]; `last` and `p` are 1 and `MAX_TOKENS · n`
/// entries of the block's shared memory that nothing else touches; `1 <=
/// m_cols <= MAX_TOKENS`, `1 <= used <= 32`, `k` a positive multiple of 32,
/// `w.len() >= (n + 1)·k`, `x.len() >= m_cols·k`, the outputs sized as
/// [`publish_route_gated`] takes them for `m_cols` tokens, and `done[0]`
/// zero before the launch.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn gated_fused_body<const PER_LANE: usize>(
    w: &[f32],
    x: &[f32],
    k: u32,
    m_cols: u32,
    used: u32,
    last: *mut u32,
    p: *mut f32,
    logits: &mut DisjointSlice<f32>,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    done: &mut DisjointSlice<u32>,
    fault: FaultSink,
) {
    let rows = 32 * PER_LANE + 1;
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let wi = tid / 32;
    let m = m_cols as usize;
    // The row guard is warp-uniform, so the warp tree sees a full warp; the
    // other warps, and a block past the rows, still meet the barrier and the
    // ticket.
    let row = thread::blockIdx_x() as usize;
    if wi == 0 && row < rows {
        let partials = if m == 1 {
            // SAFETY: row < rows, so w.len() >= rows·k >= (row + 1)·k, and
            // x.len() >= k (this fn's contract); k a positive multiple of 32;
            // lane < 32.
            let f0 = unsafe { f32_lane_partial_1col_w32(w, x, k, row, lane) };
            [f0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        } else {
            // SAFETY: as above, with x.len() >= m_cols·k and m_cols in
            // 1..=8.
            unsafe { f32_lane_partials(w, x, k, row, m_cols, lane) }
        };
        let sums = gemv_lane_sums(partials, m_cols);
        if lane == 0 {
            // SAFETY: row < rows and 1 <= m <= 8, so the slots c·rows + row
            // (c < m) lie inside logits (this fn's contract); lane 0 of the
            // row's warp is their only writer.
            unsafe { store_cols(logits, row, rows, m, &sums) };
        }
    }
    // SAFETY: this fn's contract is publish_route_gated's for m tokens.
    unsafe {
        publish_route_gated::<PER_LANE>(
            m,
            used as usize,
            last,
            p,
            logits,
            probs,
            ids,
            weights,
            done,
            fault,
        );
    }
}

/// The gated router's norm-fused launch body, `qwen3moe_router_norm`'s at
/// `n + 1 = 32 · PER_LANE + 1` rows and one token: phases A and B of
/// `norm_quant` in every block (the q8_1 of group `g` written by block `g`,
/// the normed row kept in `nr`), then warp 0 of block `r` computes row `r`'s
/// logit from that row and [`publish_route_gated`] routes the token.
///
/// # Safety
///
/// Every thread of the block calls it, converged; the grid is `n + 1`
/// blocks of [`FUSED_THREADS`]; `ws`, `nr`, `last`, `p` are [`RMS_WARPS`],
/// [`NORM_K`], 1 and `n` entries of the block's shared memory that nothing
/// else touches; `1 <= used <= 32`; `k` a positive multiple of 128
/// up to [`NORM_K`] with `k / 128 <= n + 1`, `n_sb = k / 256`, `half_it` and
/// `quad_it` its halves and quarters rounded up; `w.len() >= (n + 1)·k`,
/// `x.len()`, `gain.len() >= k`, the q8_1 planes one column of `k`; the
/// router outputs sized for one token as [`publish_route_gated`] takes them,
/// and `done[0]` zero before the launch.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn gated_norm_body<const PER_LANE: usize>(
    w: &[f32],
    x: &[f32],
    gain: &[f32],
    eps: f32,
    k: u32,
    n_sb: u32,
    half_it: u32,
    quad_it: u32,
    used: u32,
    ws: *mut f32,
    nr: *mut f32,
    last: *mut u32,
    p: *mut f32,
    q3: &mut DisjointSlice<u64>,
    q4: &mut DisjointSlice<u32>,
    q6: &mut DisjointSlice<u32>,
    s8: &mut DisjointSlice<i32>,
    d8: &mut DisjointSlice<f32>,
    logits: &mut DisjointSlice<f32>,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    done: &mut DisjointSlice<u32>,
    fault: FaultSink,
) {
    let rows = 32 * PER_LANE + 1;
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let wi = tid / 32;
    let row = thread::blockIdx_x() as usize;
    let k = k as usize;
    let n_sb = n_sb as usize;

    // Phase A: `norm_quant`'s sum of squares on this block's threads.
    // SAFETY: 0 + k <= x.len() by this fn's contract; tid < RMS_THREADS, the
    // block being FUSED_THREADS = RMS_THREADS wide (the const assert beside
    // FUSED_THREADS).
    let part = warp::reduce_sum_f32(unsafe { rms_partial_sq(x, 0, k, tid) });
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
    // `row`'s q8_1 in this block. The group index is warp-uniform, so the
    // quantizer's collectives see a full warp.
    let mut b = wi;
    while b < k / 128 {
        let vb = 128 * b + 4 * lane;
        // SAFETY: vb + 3 < k <= x.len(), gain.len() by this fn's contract.
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
        if b == row {
            if bad {
                fault.raise(FaultSite::NormQuant);
            }
            // A refused group reaches the quantizer as NaN, which refuses it
            // in turn: a NaN scale, zero codes.
            // SAFETY: one column (col 0 < 1), b < k/128 = 2·n_sb, the output
            // bounds are this fn's contract, the whole warp is here with the
            // same `b`, and `nv` holds values 128·b + 4·lane .. +3 of the
            // column.
            unsafe {
                q8_1_quant_vals(nv, 0, b, n_sb, half_it, quad_it, lane, q3, q4, q6, s8, d8);
            }
        }
        b += FUSED_WARPS;
    }
    thread::sync_threads();

    if wi == 0 && row < rows {
        // SAFETY: the shared row's k entries were written before the barrier
        // above and nothing writes them from here on.
        let xs = unsafe { core::slice::from_raw_parts(nr.cast_const(), k) };
        // SAFETY: row < rows, so w.len() >= rows·k >= (row + 1)·k, and xs
        // holds k values; k a positive multiple of 128; lane < 32.
        let f0 = unsafe { f32_lane_partial_1col_w32(w, xs, k as u32, row, lane) };
        let sums = gemv_lane_sums([f0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1);
        if lane == 0 {
            // SAFETY: row < rows <= logits.len() (this fn's contract); lane 0
            // of the row's warp is the slot's only writer.
            unsafe { store_cols(logits, row, rows, 1, &sums) };
        }
    }
    // SAFETY: this fn's contract is publish_route_gated's at one token.
    unsafe {
        publish_route_gated::<PER_LANE>(
            1,
            used as usize,
            last,
            p,
            logits,
            probs,
            ids,
            weights,
            done,
            fault,
        );
    }
}

/// The logits of `n_tok` tokens over `ROWS` router rows,
/// `qwen3moe_router_logits`' body with the row count a parameter: block `b`
/// covers tokens `32·(b / B) ..` and rows `32·(b % B) ..`, `B = ⌈ROWS /
/// 32⌉`. A row at or past `ROWS` is staged as row `ROWS − 1` and its sums
/// are not stored, as a token past `n_tok` is staged as the last token.
/// Writes `logits[t·ROWS + r]`, each the bits the fused launch leaves for
/// token `t`.
///
/// # Safety
///
/// Every thread of the block calls it, converged; the block is
/// [`LOGITS_THREADS`] wide and the grid `⌈n_tok / 32⌉ · B` blocks; `sh` is
/// the block's own `STAGES · STAGE_FLOATS` shared values, 16-byte aligned;
/// `n_tok >= 1`, `k` a positive multiple of [`LINE`], `w.len() >= ROWS·k`,
/// `x.len() >= n_tok·k`, both 16-byte aligned, `logits.len() >= ROWS·n_tok`.
#[inline(always)]
unsafe fn logits_body<const ROWS: usize>(
    w: &[f32],
    x: &[f32],
    k: u32,
    n_tok: u32,
    logits: &mut DisjointSlice<f32>,
    sh: *mut f32,
) {
    use cuda_device::async_copy::{cp_async_cg_16, cp_async_commit_group, cp_async_wait_group};

    let row_blocks = ROWS.div_ceil(LOGITS_ROWS);
    let b = thread::blockIdx_x() as usize;
    let t0 = (b / row_blocks) * LOGITS_TOKENS;
    let row0 = (b % row_blocks) * LOGITS_ROWS;
    let n = n_tok as usize;
    if t0 >= n {
        return; // block-uniform, before the first barrier
    }
    let tid = thread::threadIdx_x() as usize;
    let lane = warp::lane_id() as usize;
    let (rt, tt) = ((tid / 32) / TOKEN_TILES, (tid / 32) % TOKEN_TILES);
    let kk = k as usize;
    let stages = kk / LINE;

    // This thread's copies of every stage: piece `piece` of row line `line`
    // and of token line `line`, a row past the last staged as the last and a
    // token past the last as the last.
    let line = tid / LINE_COPIES;
    let piece = 4 * (tid % LINE_COPIES);
    let w_off = (row0 + line).min(ROWS - 1) * kk + piece;
    let x_off = (t0 + line).min(n - 1) * kk + piece;
    let w_dst = line * LINE + piece;
    let x_dst = (LOGITS_ROWS + line) * LINE + piece;
    macro_rules! stage_copy {
        ($s:expr, $buf:expr) => {{
            let (s, buf): (usize, usize) = ($s, $buf);
            // SAFETY: s < stages, so s·LINE + piece + 4 <= k: the 16 source
            // bytes lie in a row below ROWS of w (w.len() >= ROWS·k) and in
            // a column below n_tok of x (x.len() >= n_tok·k), 16-byte aligned
            // (k a multiple of 64, both bases 16-byte aligned: this fn's
            // contract). The destination is this thread's own 16 bytes of
            // stage buf < STAGES, 16-byte aligned (LINE and piece are
            // multiples of four values), read by no warp until the barrier
            // after the wait that completes this copy.
            unsafe {
                let dst = sh.add(buf * STAGE_FLOATS);
                cp_async_cg_16(
                    dst.add(w_dst).cast::<u32>(),
                    w.as_ptr().add(w_off + s * LINE).cast::<u32>(),
                );
                cp_async_cg_16(
                    dst.add(x_dst).cast::<u32>(),
                    x.as_ptr().add(x_off + s * LINE).cast::<u32>(),
                );
            }
        }};
    }

    stage_copy!(0, 0);
    // SAFETY: commits this thread's copies above as one group.
    unsafe { cp_async_commit_group() };
    if stages > 1 {
        stage_copy!(1, 1);
    }
    // SAFETY: commits this thread's copies of stage 1 as the second group,
    // empty when `stages` is 1; the loop's wait counts it as stage 1's either
    // way.
    unsafe { cp_async_commit_group() };

    let mut acc = [[0.0f32; TILE]; TILE];
    let mut s = 0usize;
    let mut buf = 0usize;
    while s < stages {
        // SAFETY: one group per stage was committed before this wait (two
        // ahead of the loop, one per earlier pass), so leaving the newest
        // pending completes stage s's. The barrier then publishes every
        // thread's copies of it, and tells that every warp is done reading
        // stage s − 1, whose buffer the copy below refills.
        unsafe { cp_async_wait_group(1) };
        thread::sync_threads();
        if s + 2 < stages {
            stage_copy!(s + 2, if buf == 0 { STAGES - 1 } else { buf - 1 });
        }
        // SAFETY: one group per pass, empty near the end.
        unsafe { cp_async_commit_group() };
        // SAFETY: stage buf's row lines rt·TILE + i and token lines
        // LOGITS_ROWS + tt·TILE + c (i, c < TILE) hold values 32·j + lane of
        // chunks j < STAGE_CHUNKS, published by the barrier above and not
        // refilled before the next pass's barrier.
        unsafe {
            let base = sh.add(buf * STAGE_FLOATS).cast_const();
            let wp = base.add(rt * TILE * LINE + lane);
            let xp = base.add((LOGITS_ROWS + tt * TILE) * LINE + lane);
            acc = f32_tile_chunk(acc, wp, xp, LINE);
            acc = f32_tile_chunk(acc, wp.add(32), xp.add(32), LINE);
        }
        s += 1;
        buf = if buf == STAGES - 1 { 0 } else { buf + 1 };
    }

    // The warp's first token, how many of its tokens exist, its first row and
    // how many of its rows exist.
    let tw = t0 + tt * TILE;
    let live = if tw < n { (n - tw).min(TILE) } else { 0 };
    let r0 = row0 + rt * TILE;
    let live_rows = if r0 < ROWS { (ROWS - r0).min(TILE) } else { 0 };
    // Every lane runs each column's butterfly; lane 0 stores a live one's
    // live rows.
    macro_rules! column {
        ($c:literal) => {{
            let sums = gemv_lane_sums(acc[$c], TILE as u32);
            if lane == 0 && $c < live {
                // SAFETY: token tw + $c < n_tok, so its slots (tw + $c)·ROWS
                // + r0 + i, i < live_rows, lie inside logits (this fn's
                // contract); lane 0 of the warp owning rows r0 .. of that
                // token is their only writer.
                unsafe {
                    store_rows_live(logits, (tw + $c) * ROWS + r0, &sums, live_rows);
                }
            }
        }};
    }
    column!(0);
    column!(1);
    column!(2);
    column!(3);
    column!(4);
    column!(5);
    column!(6);
    column!(7);
}

/// The routing of `n_tok` tokens from their logits (`n_tok · (n + 1)`, as
/// [`logits_body`] writes them at `n + 1` rows): warp `w` of block `b`
/// routes token `FUSED_WARPS·b + w` with [`route_warp_gated`].
///
/// # Safety
///
/// Every thread of the block calls it; the block is [`FUSED_THREADS`] wide;
/// `p` is the block's own `FUSED_WARPS · n` shared entries; `1 <= used <=
/// 32`; `logits.len() >= (n + 1)·n_tok`, `probs.len() >= n·n_tok`,
/// `ids.len()`, `weights.len() >= (used + 1)·n_tok`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
unsafe fn gated_route_body<const PER_LANE: usize>(
    logits: &[f32],
    n_tok: u32,
    used: u32,
    p: *mut f32,
    probs: &mut DisjointSlice<f32>,
    ids: &mut DisjointSlice<u32>,
    weights: &mut DisjointSlice<f32>,
    fault: FaultSink,
) {
    let n = 32 * PER_LANE;
    let wi = thread::threadIdx_x() as usize / 32;
    let t = thread::blockIdx_x() as usize * FUSED_WARPS + wi;
    if t >= n_tok as usize {
        return; // warp-uniform
    }
    let lane = warp::lane_id() as usize;
    let base = t * (n + 1);
    let pt = p.wrapping_add(wi * n);
    let mut j = 0usize;
    while j < PER_LANE {
        // SAFETY: base + lane + 32 j < base + n < (t + 1)·(n + 1) <=
        // logits.len(), and wi·n + lane + 32 j < FUSED_WARPS·n, inside `p`
        // (this fn's contract); the entry is this lane's own.
        unsafe { *pt.add(lane + 32 * j) = *logits.get_unchecked(base + lane + 32 * j) };
        j += 1;
    }
    // SAFETY: base + n < (t + 1)·(n + 1) <= logits.len().
    let g = unsafe { *logits.get_unchecked(base + n) };
    // SAFETY: warp wi < FUSED_WARPS takes entries wi·n .. of the block's
    // shared scratch, which no other warp touches, its expert logits written
    // above by their lanes; the whole warp is here (the guard is
    // warp-uniform); 1 <= used <= 32, and token t's slots are inside probs,
    // ids and weights (this fn's contract), written by this warp alone.
    unsafe {
        route_warp_gated::<PER_LANE>(g, pt, used as usize, t, probs, ids, weights, fault);
    }
}

#[cuda_module]
mod qwen3moe_router_kernels {
    use super::*;
    use cuda_device::async_copy::{cp_async_cg_16, cp_async_commit_group, cp_async_wait_group};

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
                unsafe { store_cols(&mut logits, row, WIDTH, m, &sums) };
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
    /// with the same cores in the same order: phase A is its sum of squares
    /// on the block's 256 threads (`rms_partial_sq`, the warp butterfly,
    /// `rms_warp_tree`, `rms_scale`), phase B its per-128-value geometry,
    /// warp `w` taking groups `w, w + 8, …`, each lane the normalized
    /// `(scale · gain) · x` of four consecutive values — kept in shared
    /// memory, and in block `g` for group `g` also quantized with
    /// `q8_1_quant_vals` (`norm_quant`'s tail, factored) and checked as
    /// `norm_quant` checks it: a non-finite normalized value or a scale that
    /// is not positive raises [`FaultSite::NormQuant`] on `fault`, and its
    /// group is refused as `norm_quant` refuses it — NaN values in the shared
    /// row, a NaN scale in `d8`. So the q8_1 bytes are `norm_quant`'s, each
    /// group written by one block, and every block holds the normed row
    /// `norm_quant` stores in `y`. Then warp
    /// 0 of block `r` computes row `r`'s logit from that shared row with the
    /// fused entry's one-token body, and the ticket and the routing follow
    /// ([`publish_route`]), a non-finite logit raising [`FaultSite::Router`]
    /// on the same `fault`. `k` a multiple of 128 and at most [`NORM_K`]
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
        // SAFETY: 0 + k <= x.len() (the launcher-checked `requires`); tid <
        // RMS_THREADS, the contract's exact block width (the const assert
        // beside FUSED_THREADS).
        let part = warp::reduce_sum_f32(unsafe { rms_partial_sq(x, 0, k, tid) });
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
            if b == row {
                if bad {
                    fault.raise(FaultSite::NormQuant);
                }
                // A refused group reaches the quantizer as NaN, which
                // refuses it in turn: a NaN scale, zero codes.
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

        if wi == 0 && row < WIDTH {
            // SAFETY: the shared row's k entries were written before the
            // barrier above and nothing writes them from here on.
            let xs = unsafe { core::slice::from_raw_parts(nr.cast_const(), k) };
            // SAFETY: row < WIDTH, so w.len() >= 128·k >= (row + 1)·k,
            // and xs holds k values; the launcher passes k a positive
            // multiple of 128; lane < 32.
            let f0 = unsafe { f32_lane_partial_1col_w32(w, xs, k as u32, row, lane) };
            let sums = gemv_lane_sums([f0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1);
            if lane == 0 {
                // SAFETY: row < WIDTH <= logits.len() (launch contract);
                // lane 0 of the row's warp is the slot's only writer.
                unsafe { store_cols(&mut logits, row, WIDTH, 1, &sums) };
            }
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

    /// The router logits of `n_tok` tokens, a ubatch's first router launch:
    /// `w` the router weight ([`WIDTH`] rows of `k` f32), `x` the tokens'
    /// normed activations (`n_tok` columns of `k`). Block `b` covers tokens
    /// `32·(b / 4) ..` and expert rows `32·(b % 4) ..`; its warp `v` owns
    /// eight rows from `8·(v / 4)` and eight tokens from `8·(v % 4)` of
    /// those. The block stages each of its rows' and tokens' next 64 values
    /// in shared memory with `cp.async`, three stages deep, and each lane
    /// advances its 64 sums over the staged 32-value chunks in increasing
    /// order ([`f32_tile_chunk`]): the sum of lane `l` for a (row, token)
    /// pair takes the fused entry's terms `w[row·k + 32·it + l] · x[token·k +
    /// 32·it + l]` in the fused entry's order, and each token's eight sums go
    /// through [`gemv_lane_sums`], so each logit is bit for bit the fused
    /// launch's for its column. A token past `n_tok` is staged as token
    /// `n_tok − 1` and its sums are not stored. Writes `logits[t·128 + e]`.
    /// `k` a positive multiple of 64, `w` and `x` 16-byte aligned
    /// (host-checked).
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
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

        let b = thread::blockIdx_x() as usize;
        let t0 = (b / ROW_BLOCKS) * LOGITS_TOKENS;
        let row0 = (b % ROW_BLOCKS) * LOGITS_ROWS;
        let n = n_tok as usize;
        if t0 >= n {
            return; // block-uniform, before the first barrier
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let (rt, tt) = ((tid / 32) / TOKEN_TILES, (tid / 32) % TOKEN_TILES);
        let kk = k as usize;
        let stages = kk / LINE;
        // SAFETY: STAGE is this block's own shared allocation of STAGES ·
        // STAGE_FLOATS values, 16-byte aligned; the raw form reaches the
        // `static mut` without a reference.
        let sh = unsafe { SharedArray::as_raw_mut_ptr(&raw mut STAGE) };

        // This thread's copies of every stage: piece `piece` of row line
        // `line` and of token line `line`, a token past the last staged as
        // the last.
        let line = tid / LINE_COPIES;
        let piece = 4 * (tid % LINE_COPIES);
        let w_off = (row0 + line) * kk + piece;
        let x_off = (t0 + line).min(n - 1) * kk + piece;
        let w_dst = line * LINE + piece;
        let x_dst = (LOGITS_ROWS + line) * LINE + piece;
        macro_rules! stage_copy {
            ($s:expr, $buf:expr) => {{
                let (s, buf): (usize, usize) = ($s, $buf);
                // SAFETY: s < stages, so s·LINE + piece + 4 <= k: the 16
                // source bytes lie in row row0 + line < WIDTH of w
                // (w.len() >= 128·k) and in a column below n_tok of x
                // (x.len() >= n_tok·k), 16-byte aligned (k a multiple of 64,
                // both bases 16-byte aligned: host-checked). The destination
                // is this thread's own 16 bytes of stage buf < STAGES, 16-byte
                // aligned (LINE and piece are multiples of four values), read
                // by no warp until the barrier after the wait that completes
                // this copy.
                unsafe {
                    let dst = sh.add(buf * STAGE_FLOATS);
                    cp_async_cg_16(
                        dst.add(w_dst).cast::<u32>(),
                        w.as_ptr().add(w_off + s * LINE).cast::<u32>(),
                    );
                    cp_async_cg_16(
                        dst.add(x_dst).cast::<u32>(),
                        x.as_ptr().add(x_off + s * LINE).cast::<u32>(),
                    );
                }
            }};
        }

        stage_copy!(0, 0);
        // SAFETY: commits this thread's copies above as one group.
        unsafe { cp_async_commit_group() };
        if stages > 1 {
            stage_copy!(1, 1);
        }
        // SAFETY: commits this thread's copies of stage 1 as the second
        // group, empty when `stages` is 1 (no copy was issued); the loop's
        // wait counts it as stage 1's either way.
        unsafe { cp_async_commit_group() };

        let mut acc = [[0.0f32; TILE]; TILE];
        let mut s = 0usize;
        let mut buf = 0usize;
        while s < stages {
            // SAFETY: one group per stage was committed before this wait (two
            // ahead of the loop, one per earlier pass), so leaving the newest
            // pending completes stage s's. The barrier then publishes every
            // thread's copies of it, and tells that every warp is done
            // reading stage s − 1, whose buffer the copy below refills.
            unsafe { cp_async_wait_group(1) };
            thread::sync_threads();
            if s + 2 < stages {
                stage_copy!(s + 2, if buf == 0 { STAGES - 1 } else { buf - 1 });
            }
            // SAFETY: one group per pass, empty near the end.
            unsafe { cp_async_commit_group() };
            // SAFETY: stage buf's row lines rt·TILE + i and token lines
            // LOGITS_ROWS + tt·TILE + c (i, c < TILE) hold values 32·j +
            // lane of chunks j < STAGE_CHUNKS, published by the barrier above
            // and not refilled before the next pass's barrier.
            unsafe {
                let base = sh.add(buf * STAGE_FLOATS).cast_const();
                let wp = base.add(rt * TILE * LINE + lane);
                let xp = base.add((LOGITS_ROWS + tt * TILE) * LINE + lane);
                acc = f32_tile_chunk(acc, wp, xp, LINE);
                acc = f32_tile_chunk(acc, wp.add(32), xp.add(32), LINE);
            }
            s += 1;
            buf = if buf == STAGES - 1 { 0 } else { buf + 1 };
        }

        // The warp's first token, how many of its tokens exist, its first row.
        let tw = t0 + tt * TILE;
        let live = if tw < n { (n - tw).min(TILE) } else { 0 };
        let r0 = row0 + rt * TILE;
        // Every lane runs each column's butterfly; lane 0 stores a live one.
        macro_rules! column {
            ($c:literal) => {{
                let sums = gemv_lane_sums(acc[$c], TILE as u32);
                if lane == 0 && $c < live {
                    // SAFETY: token tw + $c < n_tok, so its slots (tw +
                    // $c)·128 + r0 + i, i < TILE, lie inside logits (launch
                    // contract); lane 0 of the warp owning rows r0 .. of that
                    // token is their only writer.
                    unsafe { store_rows(&mut logits, (tw + $c) * WIDTH + r0, &sums) };
                }
            }};
        }
        column!(0);
        column!(1);
        column!(2);
        column!(3);
        column!(4);
        column!(5);
        column!(6);
        column!(7);
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

/// A router's shape as its buffers and launches take it: the instance row a
/// file's shape selected, and the experts a token keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouterDims {
    inst: RouterInst,
    used: usize,
}

impl RouterDims {
    /// The dims of `shape`: its instance row, which must be one of this
    /// module's bodies (`qwen3moe_router_*` or `gated`'s entries), and its
    /// `top_k`. A shape no row serves, or a row another module owns, is
    /// refused by name.
    pub fn of(shape: MoeShape) -> Result<RouterDims, GpuError> {
        const WHAT: &str = "qwen3moe::router::RouterDims::of";
        let inst = shape::select_router(shape).map_err(|e| GpuError::shape(WHAT, e.to_string()))?;
        if !matches!(inst.body, RouterBody::Qwen3moe | RouterBody::Qwen35moe) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{shape:?} selects {}, which this module does not run",
                    inst.at
                ),
            ));
        }
        Ok(RouterDims {
            inst,
            used: shape.top_k as usize,
        })
    }

    /// The instance row.
    #[must_use]
    pub fn inst(&self) -> RouterInst {
        self.inst
    }

    /// Experts the router softmaxes over.
    #[must_use]
    pub fn experts(&self) -> usize {
        self.inst.experts() as usize
    }

    /// Routed experts a token keeps.
    #[must_use]
    pub fn used(&self) -> usize {
        self.used
    }

    /// Whether a shared expert's gate is the router's last row and its slot
    /// the token's last.
    #[must_use]
    pub fn gated(&self) -> bool {
        self.inst.rule.gated
    }

    /// Logits per token: one per router row.
    #[must_use]
    pub fn logits(&self) -> usize {
        self.experts() + usize::from(self.gated())
    }

    /// Slots (an id and a weight each) per token: the routed ones, then a
    /// gated shared expert's.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.used + usize::from(self.gated())
    }

    /// The most tokens a ubatch routing may hold: at most [`UBATCH`], and
    /// every slot of them in one GEMM route table.
    #[must_use]
    pub fn ubatch(&self) -> usize {
        UBATCH.min(GEMM_MAX_SLOTS / self.slots())
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
    done: DeviceBuffer<u32>,
    tokens: usize,
    dims: RouterDims,
}

/// The router's buffers, the plain body's and [`gated`]'s alike.
pub type RouterOut = RouterBufs;

impl RouterBufs {
    /// Allocate the buffers for up to `tokens` (1..=[`MAX_TOKENS`]) tokens
    /// per launch. Load-time only.
    pub fn with_tokens(
        stream: &CudaStream,
        dims: RouterDims,
        tokens: usize,
    ) -> Result<Self, GpuError> {
        let what = "qwen3moe::router::RouterBufs::with_tokens";
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
        let what = "qwen3moe::router::RouterBufs::for_ubatch";
        Self::alloc(stream, dims, tokens, dims.ubatch(), what)
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
    fn used_for(
        &self,
        what: &'static str,
        body: RouterBody,
        per_lane: Option<u32>,
    ) -> Result<u32, GpuError> {
        let inst = self.dims.inst;
        if inst.body != body || per_lane.is_some_and(|w| w != inst.per_lane) {
            return Err(GpuError::shape(
                what,
                format!(
                    "buffers cut for {}; this launch runs another router",
                    inst.at
                ),
            ));
        }
        launch_u32(what, "used", self.dims.used)
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
        let what = "qwen3moe::router::enqueue_fused";
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
        let what = "qwen3moe::router::enqueue_ubatch";
        out.used_for(what, RouterBody::Qwen3moe, None)?;
        let k = w.cols();
        if w.rows() != WIDTH || k == 0 || !k.is_multiple_of(LINE) {
            return Err(GpuError::shape(
                what,
                format!(
                    "router weight is {} x {k}, want {WIDTH} rows of a positive multiple of {LINE}",
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
        let (w_at, x_at) = (w.buf().cu_deviceptr(), x.cu_deviceptr());
        if !w_at.is_multiple_of(16) || !x_at.is_multiple_of(16) {
            return Err(GpuError::shape(
                what,
                format!(
                    "router weight at {w_at:#x}, input at {x_at:#x}: both must be 16-byte aligned"
                ),
            ));
        }
        let k = launch_u32(what, "k", k)?;
        let n_tok = launch_u32(what, "n", n)?;
        let grid = launch_u32(what, "logits grid", n.div_ceil(LOGITS_TOKENS) * ROW_BLOCKS)?;
        let prep = self
            .module
            .prepare_qwen3moe_router_logits(LaunchConfig1D::new(grid, LOGITS_THREADS_U32, 0))?;
        self.module
            .qwen3moe_router_logits(stream, &prep, w.buf(), x, k, n_tok, &mut out.logits)?;
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
        let what = "qwen3moe::router::enqueue_route";
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
        let what = "qwen3moe::router::enqueue_norm_fused";
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
        Ok(())
    }
}

/// The router of an MoE block whose shared expert runs as one more routed
/// slot (Qwen3.6, Qwen3.8: `n` experts, the top `k`, and a shared expert
/// behind a sigmoid gate).
///
/// The router weight is the file's `ffn_gate_inp` with the shared expert's
/// gate `ffn_gate_inp_shexp` joined below it as row `n` (`n + 1` rows of the
/// hidden width), and the expert stacks carry the shared expert as expert
/// `n`. Each token gets `k + 1` slots: slots `0 .. k` are the module doc's
/// numeric contract over the `n` experts' logits, op for op (softmax, the
/// top `k`, the weights renormalized over them); slot `k` is `(n,
/// sigmoid(logit_n))` — `route_core::sigmoid`, not renormalized. So the
/// combine's slot-ascending sum over a token's slots is `((Σ_{s < k}
/// w_s·d_s) + w_sh·d_sh) + resid`. `k` is a launch argument of one to 32,
/// read from the buffers' [`RouterDims`]; `n` is the instance's.
///
/// Two instances of the const-generic bodies: [`ROW`] (256 experts) with
/// four entries — `qwen35moe_router_fused` (up to `MAX_TOKENS` tokens, one
/// block per row), `qwen35moe_router_norm` (one token, the FFN norm folded
/// in), `qwen35moe_router_logits` (a ubatch's logits, `⌈(n + 1) / 32⌉` row
/// blocks per 32 tokens, the last one's rows past `n + 1` staged as the last
/// row and not stored) and `qwen35moe_router_route` (the routing alone) —
/// and [`ROW_WIDE`] (512 experts) with the same three but the norm-fused
/// one, `qwen35moe_router_*_512`, whose router reads the hyper-connection
/// mix with no norm in front of it. The launchers pick the entry from the
/// buffers' instance. A token with a non-finite value among its `n + 1`
/// logits raises [`FaultSite::Router`], gets NaN probabilities and NaN
/// weights in all `k + 1` slots, and keeps its ids.
pub mod gated {
    use super::{
        FUSED_THREADS_U32, FUSED_WARPS, LINE, LOGITS_THREADS_U32, LOGITS_TOKENS, MAX_TOKENS,
        NORM_K, RMS_WARPS, RouterBufs, STAGE_FLOATS, STAGES, gated_fused_body, gated_norm_body,
        gated_route_body, logits_body,
    };
    use crate::fault::FaultSink;
    use crate::tensor::{DeviceTensor, Q8Act};
    use crate::{GpuError, launch_u32};
    use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
    use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract};
    use cuda_host::cuda_module;
    use model::arch::models::shape::{self, RouterBody, RouterInst, rules};
    use std::sync::Arc;

    /// Experts each lane of a routing warp owns (`lane + 32 j`) at the
    /// narrow and the wide instance.
    const PER_LANE: usize = 8;
    const PER_LANE_WIDE: usize = 16;
    /// Experts each instance softmaxes over, and the rows of its joined
    /// router weight (the logits a token): the experts, then the shared gate.
    const EXPERTS: usize = 32 * PER_LANE;
    const ROWS: usize = EXPERTS + 1;
    const EXPERTS_WIDE: usize = 32 * PER_LANE_WIDE;
    const ROWS_WIDE: usize = EXPERTS_WIDE + 1;

    /// The narrow instance's row: 256 experts, `k` one to 32.
    pub const ROW: RouterInst = shape::router_row(RouterBody::Qwen35moe, PER_LANE as u32);
    /// The wide instance's row: 512 experts, `k` one to 32.
    pub const ROW_WIDE: RouterInst = shape::router_row(RouterBody::Qwen35moe, PER_LANE_WIDE as u32);
    const _: () = assert!(
        shape::router_row_is(
            ROW,
            rules::SOFTMAX_NORM_GATED,
            false,
            (1, shape::LANE_PICKS)
        ) && shape::router_row_is(
            ROW_WIDE,
            rules::SOFTMAX_NORM_GATED,
            false,
            (1, shape::LANE_PICKS)
        )
    );
    // Each lane's taken experts are the bits of one u32, one per `j`.
    const _: () = assert!(PER_LANE <= 32 && PER_LANE_WIDE <= 32);
    // The fused and norm-fused grids are one block per row, and the norm
    // quantizes group `g` in block `g`.
    const _: () = assert!(NORM_K / 128 <= ROWS);
    // A slot id is a u32.
    const _: () = assert!(ROWS_WIDE <= u32::MAX as usize);
    // The fused and routing entries' static shared memory fits the 48 KiB a
    // block may declare: `P` for eight tokens, and the ticket.
    const _: () = assert!((MAX_TOKENS * EXPERTS_WIDE + 1) * 4 <= 48 * 1024);
    const _: () = assert!(FUSED_WARPS * EXPERTS_WIDE * 4 <= 48 * 1024);

    #[cuda_module]
    mod qwen35moe_router_kernels {
        use super::*;

        /// The router of `m_cols` tokens (the module doc of [`super`]) at
        /// the narrow instance: `w` the joined router weight ([`ROWS`]
        /// rows of `k` f32), `x` the tokens' normed activations (`m_cols`
        /// columns of `k`), `used` the routed experts a token keeps. Writes
        /// `logits[t·ROWS + r]`, `probs[t·EXPERTS + e]`, `ids[t·(used + 1) +
        /// s]` and `weights[t·(used + 1) + s]`. `done[0]` is the block ticket
        /// count: zero before the launch, zero again after it. Grid
        /// [`ROWS`] blocks of 256; warp 0 of block `r` owns row `r` with
        /// `f32_gemv`'s row body, and the block that draws the last ticket
        /// routes every token, warp `t` token `t`.
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
                w.len() >= ROWS * k,
                x.len() >= m_cols * k,
                logits.len() >= ROWS * m_cols,
                probs.len() >= EXPERTS * m_cols,
                ids.len() >= (used + 1) * m_cols,
                weights.len() >= (used + 1) * m_cols,
                done.len() >= 1
            )
        )]
        pub fn qwen35moe_router_fused(
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
            static mut P: SharedArray<f32, { MAX_TOKENS * EXPERTS }> = SharedArray::UNINIT;
            static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
            // SAFETY: LAST and P are this block's own shared allocations (the
            // raw form reaches each `static mut` without a reference), touched
            // by nothing else; the grid is ROWS blocks of 256 (the
            // launcher's), every thread arrives converged; 1 <= m_cols <= 8,
            // 1 <= used <= 32, the input and output lengths are the launch
            // contract's, the launcher passes k a positive multiple of 32, and
            // `done[0]` is zero before the launch.
            unsafe {
                gated_fused_body::<PER_LANE>(
                    w,
                    x,
                    k,
                    m_cols,
                    used,
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

        /// The FFN norm and the router of one token in one launch
        /// (`qwen3moe_router_norm`'s contract at [`ROWS`] rows): the q8_1 of
        /// the normed row into `q3 … d8`, `norm_quant`'s bytes, and the
        /// router's outputs as `qwen35moe_router_fused` at one token. `k` a
        /// multiple of 128 up to [`NORM_K`] (host-checked).
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
                k <= NORM_K,
                used >= 1,
                used <= 32,
                w.len() >= ROWS * k,
                x.len() >= k,
                gain.len() >= k,
                q3.len() >= 64 * half_it,
                q4.len() >= 256 * quad_it,
                q6.len() >= 128 * half_it,
                s8.len() >= 8 * n_sb,
                d8.len() >= 2 * n_sb,
                logits.len() >= ROWS,
                probs.len() >= EXPERTS,
                ids.len() >= used + 1,
                weights.len() >= used + 1,
                done.len() >= 1
            )
        )]
        pub fn qwen35moe_router_norm(
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
            static mut P: SharedArray<f32, EXPERTS> = SharedArray::UNINIT;
            static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
            // SAFETY: WSUM, NORMED, LAST and P are this block's own shared
            // allocations (the raw form reaches each `static mut` without a
            // reference), touched by nothing else; the grid is ROWS blocks of
            // 256, every thread arrives converged; the launcher passes k a
            // positive multiple of 128 up to NORM_K, n_sb = k/256 and its
            // halves and quarters rounded up; 1 <= used <= 32; the lengths
            // are the launch contract's, and `done[0]` is zero before the
            // launch.
            unsafe {
                gated_norm_body::<PER_LANE>(
                    w,
                    x,
                    gain,
                    eps,
                    k,
                    n_sb,
                    half_it,
                    quad_it,
                    used,
                    SharedArray::as_raw_mut_ptr(&raw mut WSUM),
                    SharedArray::as_raw_mut_ptr(&raw mut NORMED),
                    SharedArray::as_raw_mut_ptr(&raw mut LAST),
                    SharedArray::as_raw_mut_ptr(&raw mut P),
                    &mut q3,
                    &mut q4,
                    &mut q6,
                    &mut s8,
                    &mut d8,
                    &mut logits,
                    &mut probs,
                    &mut ids,
                    &mut weights,
                    &mut done,
                    fault,
                );
            }
        }

        /// The router logits of `n_tok` tokens at the narrow instance, a
        /// ubatch's first router launch: `qwen3moe_router_logits`' tiles over
        /// [`ROWS`] rows, `⌈ROWS / 32⌉` row blocks per 32 tokens; each
        /// logit the fused launch's bits for its token. Writes
        /// `logits[t·ROWS + r]`. `k` a positive multiple of 64, `w` and `x`
        /// 16-byte aligned (host-checked).
        #[kernel]
        #[launch_bounds(512)]
        #[launch_contract(
            domain = 1,
            block = (512, 1, 1),
            requires = (
                n_tok >= 1,
                w.len() >= ROWS * k,
                x.len() >= n_tok * k,
                logits.len() >= ROWS * n_tok
            )
        )]
        pub fn qwen35moe_router_logits(
            w: &[f32],
            x: &[f32],
            k: u32,
            n_tok: u32,
            mut logits: DisjointSlice<f32>,
        ) {
            static mut STAGE: SharedArray<f32, { STAGES * STAGE_FLOATS }, 16> = SharedArray::UNINIT;
            // SAFETY: STAGE is this block's own 16-byte aligned shared
            // allocation of STAGES · STAGE_FLOATS values (the raw form
            // reaches the `static mut` without a reference); the block is 512
            // wide and the grid the launcher's ⌈n_tok / 32⌉ · ⌈ROWS / 32⌉;
            // the launcher checks k and both alignments, and the lengths are
            // the launch contract's.
            unsafe {
                logits_body::<ROWS>(
                    w,
                    x,
                    k,
                    n_tok,
                    &mut logits,
                    SharedArray::as_raw_mut_ptr(&raw mut STAGE),
                );
            }
        }

        /// The routing of `n_tok` tokens at the narrow instance from their
        /// logits (`n_tok · ROWS`, as `qwen35moe_router_logits` writes
        /// them): warp `w` of block `b` routes token `8·b + w` at `used`
        /// routed experts and writes its probabilities, ids and weights where
        /// the fused launch writes a token's.
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(
            domain = 1,
            block = (256, 1, 1),
            requires = (
                used >= 1,
                used <= 32,
                logits.len() >= ROWS * n_tok,
                probs.len() >= EXPERTS * n_tok,
                ids.len() >= (used + 1) * n_tok,
                weights.len() >= (used + 1) * n_tok
            )
        )]
        pub fn qwen35moe_router_route(
            logits: &[f32],
            n_tok: u32,
            used: u32,
            mut probs: DisjointSlice<f32>,
            mut ids: DisjointSlice<u32>,
            mut weights: DisjointSlice<f32>,
            fault: FaultSink,
        ) {
            static mut P: SharedArray<f32, { FUSED_WARPS * EXPERTS }> = SharedArray::UNINIT;
            // SAFETY: P is this block's own shared allocation of FUSED_WARPS
            // tokens' entries (the raw form reaches the `static mut` without a
            // reference); the block is 256 wide; 1 <= used <= 32 and the
            // lengths are the launch contract's.
            unsafe {
                gated_route_body::<PER_LANE>(
                    logits,
                    n_tok,
                    used,
                    SharedArray::as_raw_mut_ptr(&raw mut P),
                    &mut probs,
                    &mut ids,
                    &mut weights,
                    fault,
                );
            }
        }

        /// The router of `m_cols` tokens (the module doc of [`super`]) at
        /// the wide instance: `w` the joined router weight ([`ROWS_WIDE`]
        /// rows of `k` f32), `x` the tokens' normed activations (`m_cols`
        /// columns of `k`), `used` the routed experts a token keeps. Writes
        /// `logits[t·ROWS_WIDE + r]`, `probs[t·EXPERTS_WIDE + e]`, `ids[t·(used + 1) +
        /// s]` and `weights[t·(used + 1) + s]`. `done[0]` is the block ticket
        /// count: zero before the launch, zero again after it. Grid
        /// [`ROWS_WIDE`] blocks of 256; warp 0 of block `r` owns row `r` with
        /// `f32_gemv`'s row body, and the block that draws the last ticket
        /// routes every token, warp `t` token `t`.
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
                w.len() >= ROWS_WIDE * k,
                x.len() >= m_cols * k,
                logits.len() >= ROWS_WIDE * m_cols,
                probs.len() >= EXPERTS_WIDE * m_cols,
                ids.len() >= (used + 1) * m_cols,
                weights.len() >= (used + 1) * m_cols,
                done.len() >= 1
            )
        )]
        pub fn qwen35moe_router_fused_512(
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
            static mut P: SharedArray<f32, { MAX_TOKENS * EXPERTS_WIDE }> = SharedArray::UNINIT;
            static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
            // SAFETY: LAST and P are this block's own shared allocations (the
            // raw form reaches each `static mut` without a reference), touched
            // by nothing else; the grid is ROWS_WIDE blocks of 256 (the
            // launcher's), every thread arrives converged; 1 <= m_cols <= 8,
            // 1 <= used <= 32, the input and output lengths are the launch
            // contract's, the launcher passes k a positive multiple of 32, and
            // `done[0]` is zero before the launch.
            unsafe {
                gated_fused_body::<PER_LANE_WIDE>(
                    w,
                    x,
                    k,
                    m_cols,
                    used,
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

        /// The router logits of `n_tok` tokens at the wide instance, a
        /// ubatch's first router launch: `qwen3moe_router_logits`' tiles over
        /// [`ROWS_WIDE`] rows, `⌈ROWS_WIDE / 32⌉` row blocks per 32 tokens; each
        /// logit the fused launch's bits for its token. Writes
        /// `logits[t·ROWS_WIDE + r]`. `k` a positive multiple of 64, `w` and `x`
        /// 16-byte aligned (host-checked).
        #[kernel]
        #[launch_bounds(512)]
        #[launch_contract(
            domain = 1,
            block = (512, 1, 1),
            requires = (
                n_tok >= 1,
                w.len() >= ROWS_WIDE * k,
                x.len() >= n_tok * k,
                logits.len() >= ROWS_WIDE * n_tok
            )
        )]
        pub fn qwen35moe_router_logits_512(
            w: &[f32],
            x: &[f32],
            k: u32,
            n_tok: u32,
            mut logits: DisjointSlice<f32>,
        ) {
            static mut STAGE: SharedArray<f32, { STAGES * STAGE_FLOATS }, 16> = SharedArray::UNINIT;
            // SAFETY: STAGE is this block's own 16-byte aligned shared
            // allocation of STAGES · STAGE_FLOATS values (the raw form
            // reaches the `static mut` without a reference); the block is 512
            // wide and the grid the launcher's ⌈n_tok / 32⌉ · ⌈ROWS_WIDE / 32⌉;
            // the launcher checks k and both alignments, and the lengths are
            // the launch contract's.
            unsafe {
                logits_body::<ROWS_WIDE>(
                    w,
                    x,
                    k,
                    n_tok,
                    &mut logits,
                    SharedArray::as_raw_mut_ptr(&raw mut STAGE),
                );
            }
        }

        /// The routing of `n_tok` tokens at the wide instance from their
        /// logits (`n_tok · ROWS_WIDE`, as `qwen35moe_router_logits_512` writes
        /// them): warp `w` of block `b` routes token `8·b + w` at `used`
        /// routed experts and writes its probabilities, ids and weights where
        /// the fused launch writes a token's.
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(
            domain = 1,
            block = (256, 1, 1),
            requires = (
                used >= 1,
                used <= 32,
                logits.len() >= ROWS_WIDE * n_tok,
                probs.len() >= EXPERTS_WIDE * n_tok,
                ids.len() >= (used + 1) * n_tok,
                weights.len() >= (used + 1) * n_tok
            )
        )]
        pub fn qwen35moe_router_route_512(
            logits: &[f32],
            n_tok: u32,
            used: u32,
            mut probs: DisjointSlice<f32>,
            mut ids: DisjointSlice<u32>,
            mut weights: DisjointSlice<f32>,
            fault: FaultSink,
        ) {
            static mut P: SharedArray<f32, { FUSED_WARPS * EXPERTS_WIDE }> = SharedArray::UNINIT;
            // SAFETY: P is this block's own shared allocation of FUSED_WARPS
            // tokens' entries (the raw form reaches the `static mut` without a
            // reference); the block is 256 wide; 1 <= used <= 32 and the
            // lengths are the launch contract's.
            unsafe {
                gated_route_body::<PER_LANE_WIDE>(
                    logits,
                    n_tok,
                    used,
                    SharedArray::as_raw_mut_ptr(&raw mut P),
                    &mut probs,
                    &mut ids,
                    &mut weights,
                    fault,
                );
            }
        }
    }

    /// The gated router's buffers ([`RouterKernels`]).
    pub type RouterOut = RouterBufs;

    /// Which instance's entries buffers cut for `out`'s row take.
    #[derive(Clone, Copy)]
    enum Width {
        Narrow,
        Wide,
    }

    /// The instance of `out`'s buffers, and `used` as the launch argument;
    /// buffers cut for another router are refused as `what`.
    fn width(what: &'static str, out: &RouterBufs) -> Result<(Width, u32), GpuError> {
        let used = out.used_for(what, RouterBody::Qwen35moe, None)?;
        let inst = out.dims().inst();
        if inst == ROW {
            Ok((Width::Narrow, used))
        } else if inst == ROW_WIDE {
            Ok((Width::Wide, used))
        } else {
            Err(GpuError::shape(
                what,
                format!("buffers cut for {}; no entry here runs it", inst.at),
            ))
        }
    }

    /// Err unless `w` is a joined router weight of `rows` rows whose width
    /// is a positive multiple of `unit` (and at most `most`): its width.
    fn router_width(
        what: &'static str,
        w: &DeviceTensor<f32>,
        rows: usize,
        unit: usize,
        most: usize,
    ) -> Result<usize, GpuError> {
        let k = w.cols();
        if w.rows() != rows || k == 0 || !k.is_multiple_of(unit) || k > most {
            return Err(GpuError::shape(
                what,
                format!(
                    "router weight is {} x {k}, want {rows} rows of a positive multiple of {unit} \
                     up to {most}",
                    w.rows()
                ),
            ));
        }
        Ok(k)
    }

    /// A fused launch's `(k, m, grid)` for `m` tokens of `x` through `w`
    /// into `out`; its refusal named `what`.
    fn fused_dims(
        what: &'static str,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        m: usize,
        out: &RouterBufs,
    ) -> Result<(u32, u32, u32), GpuError> {
        let rows = out.dims().logits();
        let k = router_width(what, w, rows, 32, usize::MAX)?;
        if !(1..=MAX_TOKENS.min(out.tokens())).contains(&m) || x.len() < m * k {
            return Err(GpuError::shape(
                what,
                format!(
                    "{m} tokens: one launch routes 1..={MAX_TOKENS}, into buffers for {}; \
                     x.len() {} for {m} x {k}",
                    out.tokens(),
                    x.len()
                ),
            ));
        }
        let k = launch_u32(what, "k", k)?;
        let m = launch_u32(what, "m", m)?;
        let grid = launch_u32(what, "grid", rows)?;
        Ok((k, m, grid))
    }

    /// A ubatch logits launch's `(k, n_tok, grid)` for `n` tokens of `x`
    /// through `w` into `out`; its refusal named `what`.
    fn ubatch_dims(
        what: &'static str,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        n: usize,
        out: &RouterBufs,
    ) -> Result<(u32, u32, u32), GpuError> {
        let rows = out.dims().logits();
        let k = router_width(what, w, rows, LINE, usize::MAX)?;
        if !(1..=out.tokens()).contains(&n) || x.len() < n * k {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n} tokens into buffers for {}, x.len() {} for {n} x {k}",
                    out.tokens(),
                    x.len()
                ),
            ));
        }
        let (w_at, x_at) = (w.buf().cu_deviceptr(), x.cu_deviceptr());
        if !w_at.is_multiple_of(16) || !x_at.is_multiple_of(16) {
            return Err(GpuError::shape(
                what,
                format!(
                    "router weight at {w_at:#x}, input at {x_at:#x}: both must be 16-byte \
                     aligned"
                ),
            ));
        }
        let k = launch_u32(what, "k", k)?;
        let n_tok = launch_u32(what, "n", n)?;
        let grid = launch_u32(
            what,
            "logits grid",
            n.div_ceil(LOGITS_TOKENS) * rows.div_ceil(32),
        )?;
        Ok((k, n_tok, grid))
    }

    /// A routing launch's `(n_tok, grid)` for the first `n` tokens of `out`;
    /// its refusal named `what`.
    fn route_dims(what: &'static str, n: usize, out: &RouterBufs) -> Result<(u32, u32), GpuError> {
        if !(1..=out.tokens()).contains(&n) {
            return Err(GpuError::shape(
                what,
                format!("{n} tokens into buffers for {}", out.tokens()),
            ));
        }
        let n_tok = launch_u32(what, "n", n)?;
        let grid = launch_u32(what, "grid", n.div_ceil(FUSED_WARPS))?;
        Ok((n_tok, grid))
    }

    /// The loaded gated router module, both instances. Owns no stream: each
    /// enqueue takes the engine stream.
    pub struct RouterKernels {
        module: qwen35moe_router_kernels::LoadedModule,
    }

    impl RouterKernels {
        /// Load this file's device bundle into `ctx`. Load-time only.
        pub fn load(ctx: &Arc<CudaContext>) -> Result<RouterKernels, GpuError> {
            // SAFETY: this package owns the embedded device bundle produced
            // for the module above; each launcher checks its launch contract.
            let module = unsafe { qwen35moe_router_kernels::load(ctx)? };
            Ok(RouterKernels { module })
        }

        /// Enqueue the router of `m` tokens, 1..=`MAX_TOKENS` and at most
        /// `out.tokens()`, at `out`'s instance and `used`: `w` the joined
        /// router weight (`n + 1` rows of `k`, `k` a positive multiple of
        /// 32), `x` the tokens' `m` columns of `k` activations; results into
        /// `out`, a token with a non-finite logit raising
        /// [`FaultSite::Router`](crate::FaultSite::Router) on `fault`.
        /// Asynchronous, allocation-free, capturable.
        pub fn enqueue_fused(
            &self,
            stream: &CudaStream,
            w: &DeviceTensor<f32>,
            x: &DeviceBuffer<f32>,
            m: usize,
            fault: FaultSink,
            out: &mut RouterOut,
        ) -> Result<(), GpuError> {
            let what = "qwen3moe::router::gated::enqueue_fused";
            let (width, used) = width(what, out)?;
            let (k, m, grid) = fused_dims(what, w, x, m, out)?;
            let cfg = LaunchConfig1D::new(grid, FUSED_THREADS_U32, 0);
            match width {
                Width::Narrow => {
                    let prep = self.module.prepare_qwen35moe_router_fused(cfg)?;
                    self.module.qwen35moe_router_fused(
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
                }
                Width::Wide => {
                    let prep = self.module.prepare_qwen35moe_router_fused_512(cfg)?;
                    self.module.qwen35moe_router_fused_512(
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
                }
            }
            Ok(())
        }

        /// Enqueue the router of a ubatch of `n` (1..=`out.tokens()`)
        /// tokens in two launches at `out`'s instance,
        /// `qwen35moe_router_logits` then `qwen35moe_router_route`
        /// ([`RouterKernels::enqueue_route`]): `w` the joined router weight
        /// (`n + 1` rows of `k`, `k` a positive multiple of 64), `x` the
        /// tokens' `n` columns of `k` normed activations, both 16-byte
        /// aligned; each token's results the bits
        /// [`RouterKernels::enqueue_fused`] leaves for it, its refusal raised
        /// on `fault` as there. Asynchronous, allocation-free, capturable.
        pub fn enqueue_ubatch(
            &self,
            stream: &CudaStream,
            w: &DeviceTensor<f32>,
            x: &DeviceBuffer<f32>,
            n: usize,
            fault: FaultSink,
            out: &mut RouterOut,
        ) -> Result<(), GpuError> {
            let what = "qwen3moe::router::gated::enqueue_ubatch";
            let (width, _) = width(what, out)?;
            let (k, n_tok, grid) = ubatch_dims(what, w, x, n, out)?;
            let cfg = LaunchConfig1D::new(grid, LOGITS_THREADS_U32, 0);
            match width {
                Width::Narrow => {
                    let prep = self.module.prepare_qwen35moe_router_logits(cfg)?;
                    self.module.qwen35moe_router_logits(
                        stream,
                        &prep,
                        w.buf(),
                        x,
                        k,
                        n_tok,
                        &mut out.logits,
                    )?;
                }
                Width::Wide => {
                    let prep = self.module.prepare_qwen35moe_router_logits_512(cfg)?;
                    self.module.qwen35moe_router_logits_512(
                        stream,
                        &prep,
                        w.buf(),
                        x,
                        k,
                        n_tok,
                        &mut out.logits,
                    )?;
                }
            }
            self.enqueue_route(stream, n, fault, out)
        }

        /// Enqueue the routing alone of the first `n` (1..=`out.tokens()`)
        /// tokens' logits in `out.logits` at `out`'s instance
        /// (`qwen35moe_router_route`): each token's probabilities, ids and
        /// weights into `out` where the fused launch writes a token's, a
        /// token with a non-finite logit raising
        /// [`FaultSite::Router`](crate::FaultSite::Router) on `fault`. The
        /// second launch of [`RouterKernels::enqueue_ubatch`].
        /// Asynchronous, allocation-free, capturable.
        pub fn enqueue_route(
            &self,
            stream: &CudaStream,
            n: usize,
            fault: FaultSink,
            out: &mut RouterOut,
        ) -> Result<(), GpuError> {
            let what = "qwen3moe::router::gated::enqueue_route";
            let (width, used) = width(what, out)?;
            let (n_tok, grid) = route_dims(what, n, out)?;
            let cfg = LaunchConfig1D::new(grid, FUSED_THREADS_U32, 0);
            match width {
                Width::Narrow => {
                    let prep = self.module.prepare_qwen35moe_router_route(cfg)?;
                    self.module.qwen35moe_router_route(
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
                }
                Width::Wide => {
                    let prep = self.module.prepare_qwen35moe_router_route_512(cfg)?;
                    self.module.qwen35moe_router_route_512(
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
                }
            }
            Ok(())
        }

        /// Enqueue the FFN norm and the router of one token in one launch
        /// (`qwen35moe_router_norm`), at the narrow instance only (the wide
        /// one has no norm-fused entry, and its buffers are refused by name):
        /// `x` the token's FFN input residual, `gain` the norm's gain
        /// (`w.cols()` values each), `eps` its epsilon; the q8_1 of the
        /// normed row into `act` (one column of `w.cols()`) — the bytes
        /// `fused::norm_quant` writes — and the routing into `out`'s first
        /// token, as [`RouterKernels::enqueue_fused`] at one token. `fault`
        /// takes the norm's raise and the router's. Asynchronous,
        /// allocation-free, capturable.
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
            let what = "qwen3moe::router::gated::enqueue_norm_fused";
            let used = out.used_for(what, RouterBody::Qwen35moe, Some(ROW.per_lane))?;
            let k = router_width(what, w, ROWS, 128, NORM_K)?;
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
            let grid = launch_u32(what, "grid", ROWS)?;
            let prep = self
                .module
                .prepare_qwen35moe_router_norm(LaunchConfig1D::new(grid, FUSED_THREADS_U32, 0))?;
            self.module.qwen35moe_router_norm(
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
            Ok(())
        }
    }
}
