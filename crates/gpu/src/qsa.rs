//! Qwen3.8's token-pool selector (QSA) on the card: the pooled keys, each
//! query row's scores over them, and the exact top-k that turns the scores
//! into the token list the selected flash reads
//! ([`crate::flash_gqa::FlashGqaKernels::enqueue_pass_256_p12_sel`]).
//!
//! The rule is `runtime::qsa`'s, mirrored here: pool `j` is tokens `[4j, 4j +
//! 4)`; a row of live count `c` (its position plus one) sees the `c / 4`
//! complete pools below it and a tail of `c % 4`; when it sees more than
//! `kept` pools it keeps the `kept` best by score (the lower pool on a tie)
//! and the tail, else every token — the modeling code's cut, whole pools.
//! (ik and mainline cut `top_k + 3` cells instead, which past position 2,050
//! also takes cells of the next pool when the tail is short.) [`scored`] is
//! the one predicate for "this row selects": the score pass writes a row's
//! scores only then and the top-k pass reads them only then. The two passes share no other state: the top-k
//! counts its histograms from the scores itself, so no count is carried from
//! one launch to the next.
//!
//! GLM-5.3-Flash's k-pool selector takes the same shape, whole pools and
//! the tail, from its own pool and score passes (`latent::index_pool`,
//! `kpool`) through [`qsa_kernels::qsa_topk_high`]: the higher pools of a
//! tie at the cut (`runtime::qsa::Tie::Higher`, ik's CPU `top_k`), and each
//! row's list length written as its attention's visible count.
//!
//! Three launches:
//! - [`qsa_kernels::qsa_pool`] — the key of every pool some row completes
//!   (`c % 4 == 0` completes pool `c/4 − 1`) from the raw indexer keys: the
//!   four rows' mean, the RMS gain norm, the NEOX turn of the first [`ROT`]
//!   values at the pool's first position, rounded once to f16 into the pooled
//!   plane. Every position passes through some row of some launch, so every
//!   pool is written exactly once, before any row can see it complete.
//! - [`qsa_kernels::qsa_score`] — per row: its [`HEADS`] indexer query heads
//!   normed and turned at its position (every block builds them; block 0
//!   writes them to `q_out`), then per pool the score `Σ_h relu(q_h · k)`.
//! - [`qsa_kernels::qsa_topk`] — one block per row: the exact `kept`-th
//!   order key by three histogram passes over the row's scores (10, 11, 11
//!   bits of the order key), then every pool
//!   above it and the first pools equal to it in pool order, written as their
//!   tokens in pool order, then the tail. A row that does not select gets the
//!   identity list `0 .. c`; a refused count (0, or past the cache) gets an
//!   empty list, which the flash's merge refuses by name.
//!
//! Numeric contract (reruns are bit-identical):
//! - a head's norm: lane `l` owns values `l, l + 32` (NEOX pair `l`) and `l +
//!   64, l + 96`; each squared in f32, the pairs' squares added in f64 and the
//!   two sums added, the lanes by the xor butterfly (16, 8, 4, 2, 1) in f64;
//!   the mean `(sum / 128) as f32`, the scale `1 / sqrt(mean + eps)` in f32,
//!   the normed value `(scale · gain) · x`; the turn through
//!   [`crate::rope_neox::neox_pair`] by table pair `l` of the position's row;
//! - a pool's mean: `((r0 + r1) + r2) + r3` of its rows' f16 values in f32,
//!   times 0.25;
//! - a score: per head, lane `j`'s dot of the query head with pool `j`'s key
//!   over [`ILP`] rotating f32 partials (partial `p` takes the value pairs
//!   `p, p + ILP, …`, each value one fused multiply-add), combined `(a0 + a1)
//!   + (a2 + a3)`; `relu` (`+0` for every value not above zero); the heads
//!   added `((h0 + h1) + h2) + h3`.
//!
//! No silent failure: a pooled value that is not finite after its f16
//! rounding (a non-finite raw key, or one past f16's range) and a head dot
//! that is not finite raise [`FaultSite::PoolSelect`]; the value is written
//! as computed, never replaced by a plausible one, and the list is still a
//! defined list of tokens below the row's count.

use crate::fault::{FaultSink, FaultSite};
use crate::flash::{f32_to_f16_bits, half_bits_to_f32};
use crate::rope_neox::neox_pair;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Indexer query heads (`attention.indexer.head_count`).
pub const HEADS: usize = 4;
/// Values of an indexer head and of a pooled key (`attention.indexer.key_length`).
pub const DIM: usize = 128;
/// Values at the head of a key or query that turn: NEOX pairs `(i, i + 32)`.
/// The rope table's row is `ROT` f32, the main attention's.
pub const ROT: usize = 64;
/// Tokens per pool (`attention.compress_ratios` of a QSA layer).
pub const POOL: usize = 4;
/// Rows one score launch serves: a decode step or a draft's verify.
pub const MAX_ROWS: usize = 8;
/// Pools one score block takes, one per lane.
pub const POOL_TILE: usize = 32;
/// Bins of the top-k's first histogram: the top ten bits of an order key.
const HIST_BINS: usize = 1024;
/// The longest list for `kept` pools: their tokens and a tail of `POOL − 1`.
#[must_use]
pub const fn list_width(kept: usize) -> usize {
    kept * POOL + POOL - 1
}

const _: () = assert!(HEADS == 4 && DIM == 128 && ROT == 64 && POOL == 4);
const _: () = assert!(MAX_ROWS == 8 && HIST_BINS == 1024 && HIST_BINS <= FINE_BINS);

/// Rotating partials of the score dot, and u32 words of a key row.
const ILP: usize = 4;
const ROW_WORDS: usize = DIM / 2;
/// The staged key tile's stride in u32 words: lane `j` reads word `w` of key
/// `j` at `j·65 + w`, a distinct bank per lane.
const K_STRIDE: usize = ROW_WORDS + 1;
/// u64 words of a key row.
const ROW_QWORDS: usize = DIM / 4;
const _: () = assert!(ROW_WORDS.is_multiple_of(ILP) && ROW_QWORDS == 32);

/// Threads of a pool block and of a score block: four warps.
const THREADS: u32 = 128;
const WARPS: usize = THREADS as usize / 32;
const _: () = assert!(WARPS == HEADS);
/// u64 words of the key tile one score thread stages.
const STAGE_PER: usize = POOL_TILE * ROW_QWORDS / THREADS as usize;
const _: () = assert!(STAGE_PER * THREADS as usize == POOL_TILE * ROW_QWORDS);

/// Threads of a top-k block, and its warps.
const TOPK_THREADS: u32 = 512;
const TOPK_WARPS: usize = TOPK_THREADS as usize / 32;
/// Pools one top-k thread reads per batch of a refining pass.
const LANE_ROWS: usize = 16;
const ROUND_ROWS: usize = TOPK_THREADS as usize * LANE_ROWS;
/// Bins of the two refining passes: eleven bits each.
const FINE_BINS: usize = 2048;
const COARSE_PER: usize = HIST_BINS / TOPK_THREADS as usize;
const FINE_PER: usize = FINE_BINS / TOPK_THREADS as usize;
const _: () = assert!(COARSE_PER * TOPK_THREADS as usize == HIST_BINS);
const _: () = assert!(FINE_PER * TOPK_THREADS as usize == FINE_BINS);

// ------------------------------------------------------------------ cores

/// Whether a row of live count `count` over a `ctx`-row cache selects: an
/// accepted count (`1..=ctx`) that sees more than `kept` complete pools —
/// `runtime::qsa::Qsa::scored`. The score pass and the top-k pass both
/// decide through this.
#[inline(always)]
#[must_use]
pub fn scored(count: u32, ctx: u32, kept: u32) -> bool {
    count >= 1 && count <= ctx && count / POOL as u32 > kept
}

/// The order-preserving key of a score — `runtime::qsa::order_key`.
#[inline(always)]
#[must_use]
pub fn order_key(v: f32) -> u32 {
    let b = v.to_bits();
    let b = if b == 0x8000_0000 { 0 } else { b };
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

/// `x` where it is above zero, `+0` elsewhere.
#[inline(always)]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

/// Lane `lane`'s four values of a normed and turned head (module doc): `x`
/// the head's values `lane, lane + 32, lane + 64, lane + 96`, `g` the gain's
/// at those dims, `(c, s)` table pair `lane` of the position's row. All 32
/// lanes of the warp call it together.
#[inline(always)]
fn norm_turn(x: [f32; 4], g: [f32; 4], c: f32, s: f32, eps: f32) -> [f32; 4] {
    let mut acc = (f64::from(x[0] * x[0]) + f64::from(x[1] * x[1]))
        + (f64::from(x[2] * x[2]) + f64::from(x[3] * x[3]));
    acc += warp::shuffle_xor_f64(acc, 16);
    acc += warp::shuffle_xor_f64(acc, 8);
    acc += warp::shuffle_xor_f64(acc, 4);
    acc += warp::shuffle_xor_f64(acc, 2);
    acc += warp::shuffle_xor_f64(acc, 1);
    let mean = (acc / DIM as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    let n0 = (scale * g[0]) * x[0];
    let n1 = (scale * g[1]) * x[1];
    let (y0, y1) = neox_pair(n0, n1, c, s);
    [y0, y1, (scale * g[2]) * x[2], (scale * g[3]) * x[3]]
}

/// The dims lane `lane` owns in a head: its NEOX pair, then two passed
/// through.
#[inline(always)]
const fn lane_dims(lane: usize) -> [usize; 4] {
    [lane, lane + ROT / 2, lane + ROT, lane + ROT + ROT / 2]
}

// The four dims of the 32 lanes cover the head once, the first pair of each
// lane a NEOX pair of the turned values.
const _: () = assert!(lane_dims_hold());
const fn lane_dims_hold() -> bool {
    let mut seen = [false; DIM];
    let mut l = 0;
    while l < 32 {
        let d = lane_dims(l);
        if d[1] != d[0] + ROT / 2 || d[0] >= ROT / 2 {
            return false;
        }
        let mut i = 0;
        while i < 4 {
            if d[i] >= DIM || seen[d[i]] {
                return false;
            }
            seen[d[i]] = true;
            i += 1;
        }
        l += 1;
    }
    true
}

/// A block-wide exclusive prefix sum of `v` over thread order in a
/// [`TOPK_THREADS`] block, and the block's total. Every thread calls it (two
/// barriers).
///
/// # Safety
/// `wsum` points at [`TOPK_WARPS`] words of this block's shared memory that
/// nothing else uses across the call.
#[inline(always)]
unsafe fn block_scan(v: u32, lane: u32, wid: usize, wsum: *mut u32) -> (u32, u32) {
    let mut x = v;
    let mut off = 1u32;
    while off < 32 {
        let y = warp::shuffle_up(x, off);
        if lane >= off {
            x += y;
        }
        off <<= 1;
    }
    if lane == 31 {
        // SAFETY: wid < TOPK_WARPS words of `wsum`; lane 31 of warp wid is
        // the slot's only writer.
        unsafe { *wsum.add(wid) = x };
    }
    thread::sync_threads();
    let (mut before, mut total) = (0u32, 0u32);
    for w in 0..TOPK_WARPS {
        thread::__unroll_config::<0>();
        // SAFETY: w < TOPK_WARPS; the barrier above published every slot.
        let s = unsafe { *wsum.add(w) };
        if w < wid {
            before += s;
        }
        total += s;
    }
    thread::sync_threads();
    (before + x - v, total)
}

/// The bin that holds the `need`-th largest key, counting down from bin
/// `top`, and how many keys lie in the bins above it. Thread `tid` owns the
/// `PER` bins `top − PER·tid − j`, counts `c[j]`; they total at least
/// `need >= 1`. Every thread calls it and gets the same pair. This module's
/// top-k passes and V4.1's indexer (`bloomery_gpu_deepseek41::indexer`) pick
/// their bins through it.
///
/// # Safety
/// As [`block_scan`]; `pick` points at two words of this block's shared
/// memory that nothing else uses across the call.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
pub unsafe fn pick_bin<const PER: usize>(
    c: [u32; PER],
    top: u32,
    need: u32,
    lane: u32,
    tid: usize,
    wsum: *mut u32,
    pick: *mut u32,
) -> (u32, u32) {
    let mut sum = 0u32;
    for j in 0..PER {
        thread::__unroll_config::<0>();
        sum += c[j];
    }
    // SAFETY: forwarded from this fn's contract.
    let (mut before, _) = unsafe { block_scan(sum, lane, tid / 32, wsum) };
    for j in 0..PER {
        thread::__unroll_config::<0>();
        if before < need && need <= before + c[j] {
            // SAFETY: two words of `pick`; exactly one (thread, bin) holds
            // the need-th key, so one thread writes.
            unsafe {
                *pick = top - (PER * tid + j) as u32;
                *pick.add(1) = before;
            }
        }
        before += c[j];
    }
    thread::sync_threads();
    // SAFETY: written before the barrier above.
    let out = unsafe { (*pick, *pick.add(1)) };
    thread::sync_threads();
    out
}

/// The order keys of entries `base + tid + 512·m`, `m < LANE_ROWS`, of the
/// scores at `scores[sbase ..]`; an entry at or past `end` gives 0.
///
/// # Safety
/// `sbase + end <= scores.len()`.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn round_keys(
    scores: &[f32],
    sbase: usize,
    base: usize,
    end: usize,
    tid: usize,
) -> [u32; LANE_ROWS] {
    let mut kk = [0u32; LANE_ROWS];
    for m in 0..LANE_ROWS {
        thread::__unroll_config::<0>();
        let i = base + tid + TOPK_THREADS as usize * m;
        if i < end {
            // SAFETY: i < end, so sbase + i < scores.len() (this fn's
            // contract).
            kk[m] = order_key(unsafe { *scores.get_unchecked(sbase + i) });
        }
    }
    kk
}

/// The order keys of pools `base + 32·m + lane`, `m < LANE_ROWS`, of one
/// row's scores; a pool at or past `end` gives 0.
///
/// # Safety
/// `sbase + end <= scores.len()`.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn lane_keys(
    scores: &[f32],
    sbase: usize,
    base: usize,
    end: usize,
    lane: usize,
) -> [u32; LANE_ROWS] {
    let mut kk = [0u32; LANE_ROWS];
    for m in 0..LANE_ROWS {
        thread::__unroll_config::<0>();
        let i = base + 32 * m + lane;
        if i < end {
            // SAFETY: i < end, so sbase + i < scores.len() (this fn's
            // contract).
            kk[m] = order_key(unsafe { *scores.get_unchecked(sbase + i) });
        }
    }
    kk
}

/// Zero the fine histogram and publish it.
///
/// # Safety
/// `fine` points at [`FINE_BINS`] words of this block's shared memory; every
/// thread of the block calls it.
#[inline(always)]
unsafe fn fine_zero(fine: *mut u32, tid: usize) {
    for j in 0..FINE_PER {
        thread::__unroll_config::<0>();
        // SAFETY: FINE_PER·tid + j < FINE_BINS; thread tid alone writes these
        // bins before the barrier.
        unsafe { *fine.add(FINE_PER * tid + j) = 0 };
    }
    thread::sync_threads();
}

/// This thread's [`FINE_PER`] bins of the fine histogram, from the top bin
/// down: bins `FINE_BINS − 1 − (FINE_PER·tid + j)`.
///
/// # Safety
/// `fine` points at [`FINE_BINS`] published words of this block's shared
/// memory.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn fine_bins(fine: *const u32, tid: usize) -> [u32; FINE_PER] {
    let mut c = [0u32; FINE_PER];
    for j in 0..FINE_PER {
        thread::__unroll_config::<0>();
        // SAFETY: the index is below FINE_BINS.
        c[j] = unsafe { *fine.add(FINE_BINS - 1 - (FINE_PER * tid + j)) };
    }
    c
}

/// The first pass: count into `fine` the top ten bits of the order keys of
/// the row's `n` pools; then this thread's [`COARSE_PER`] bins, from the top
/// bin down.
///
/// # Safety
/// As [`refine`].
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn coarse(
    scores: &[f32],
    sbase: usize,
    n: usize,
    fine: *mut u32,
    tid: usize,
) -> [u32; COARSE_PER] {
    // SAFETY: forwarded from this fn's contract.
    unsafe { fine_zero(fine, tid) };
    let mut base = 0usize;
    while base < n {
        // SAFETY: sbase + n <= scores.len().
        let kk = unsafe { round_keys(scores, sbase, base, n, tid) };
        for m in 0..LANE_ROWS {
            thread::__unroll_config::<0>();
            if base + tid + TOPK_THREADS as usize * m < n {
                // SAFETY: the bin `kk >> 22` is below HIST_BINS <= FINE_BINS;
                // every access to FINE between the barriers is atomic.
                unsafe {
                    BlockAtomicU32::from_ptr(fine.add((kk[m] >> 22) as usize))
                        .fetch_add(1, AtomicOrdering::Relaxed)
                };
            }
        }
        base += ROUND_ROWS;
    }
    thread::sync_threads();
    let mut c = [0u32; COARSE_PER];
    for j in 0..COARSE_PER {
        thread::__unroll_config::<0>();
        // SAFETY: the index is below HIST_BINS <= FINE_BINS, published by the
        // barrier above.
        c[j] = unsafe { *fine.add(HIST_BINS - 1 - (COARSE_PER * tid + j)) };
    }
    c
}

/// One refining pass: count into `fine`, by the bits `(key >> shift) &
/// 0x7ff`, the keys of the `n` scores at `scores[sbase ..]` whose bits above
/// `shift + 11` equal `prefix`; then this thread's bins. This module's top-k
/// passes and V4.1's indexer (`bloomery_gpu_deepseek41::indexer`) refine through it.
///
/// # Safety
/// `fine` points at [`FINE_BINS`] words of this block's shared memory used by
/// nothing else across the call; `sbase + n <= scores.len()`; every thread
/// calls it.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
pub unsafe fn refine(
    scores: &[f32],
    sbase: usize,
    n: usize,
    prefix: u32,
    shift: u32,
    fine: *mut u32,
    tid: usize,
) -> [u32; FINE_PER] {
    // SAFETY: forwarded from this fn's contract.
    unsafe { fine_zero(fine, tid) };
    let mut base = 0usize;
    while base < n {
        // SAFETY: sbase + n <= scores.len().
        let kk = unsafe { round_keys(scores, sbase, base, n, tid) };
        for m in 0..LANE_ROWS {
            thread::__unroll_config::<0>();
            let i = base + tid + TOPK_THREADS as usize * m;
            if i < n && kk[m] >> (shift + 11) == prefix {
                // SAFETY: the bin is below FINE_BINS; every access to FINE
                // between the barriers is atomic.
                unsafe {
                    BlockAtomicU32::from_ptr(fine.add(((kk[m] >> shift) & 0x7ff) as usize))
                        .fetch_add(1, AtomicOrdering::Relaxed)
                };
            }
        }
        base += ROUND_ROWS;
    }
    thread::sync_threads();
    // SAFETY: FINE was published by the barrier above.
    unsafe { fine_bins(fine, tid) }
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod qsa_kernels {
    use super::*;

    /// The pool pass: warp `w` of block `b` serves row `t = 4b + w`. A row
    /// whose live count `c` is accepted (`1..=ctx`) and a multiple of
    /// [`POOL`] completes pool `j = c/4 − 1`: lane `l` reads values
    /// [`lane_dims`] of the raw rows `4j .. 4j + 3` (`raw`, `[ctx][DIM]` f16),
    /// takes their mean, norms it by `gain`, turns it by the table's row `4j`
    /// (`table[4j·ROT ..]`), and rounds the four values into
    /// `pooled[j·DIM ..]`. Any other row writes nothing. The warp's rows are
    /// its own; no barrier.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n_keys.len() >= m,
            raw.len() >= ctx * 128,
            gain.len() >= 128,
            table.len() >= ctx * 64,
            pools * 4 >= ctx,
            pooled.len() >= pools * 128
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn qsa_pool(
        raw: &[u16],
        gain: &[f32],
        table: &[f32],
        n_keys: &[u32],
        eps: f32,
        ctx: u32,
        pools: u32,
        m: u32,
        fault: FaultSink,
        mut pooled: DisjointSlice<u16>,
    ) {
        let _ = pools;
        let tid = thread::threadIdx_x() as usize;
        let t = thread::blockIdx_x() as usize * WARPS + tid / 32;
        if t >= m as usize {
            return; // warp-uniform: the row is the warp's, and no barrier follows
        }
        // SAFETY: t < m <= n_keys.len() by the launch contract.
        let c = unsafe { *n_keys.get_unchecked(t) };
        if c == 0 || c > ctx || !c.is_multiple_of(POOL as u32) {
            return; // warp-uniform
        }
        let j = (c / POOL as u32 - 1) as usize;
        let lane = warp::lane_id() as usize;
        let d = lane_dims(lane);
        let mut x = [0.0f32; 4];
        let mut g = [0.0f32; 4];
        for i in 0..4 {
            thread::__unroll_config::<0>();
            let at = POOL * j * DIM + d[i];
            // SAFETY: the rows 4j .. 4j + 3 are below c <= ctx, so each index
            // is below ctx·DIM <= raw.len(); d[i] < DIM <= gain.len().
            let (r0, r1, r2, r3, gi) = unsafe {
                (
                    half_bits_to_f32(*raw.get_unchecked(at)),
                    half_bits_to_f32(*raw.get_unchecked(at + DIM)),
                    half_bits_to_f32(*raw.get_unchecked(at + 2 * DIM)),
                    half_bits_to_f32(*raw.get_unchecked(at + 3 * DIM)),
                    *gain.get_unchecked(d[i]),
                )
            };
            x[i] = (((r0 + r1) + r2) + r3) * 0.25;
            g[i] = gi;
        }
        let p = POOL * j;
        // SAFETY: p < c <= ctx and 2·lane + 1 < ROT: inside row p of the
        // table's ctx rows of ROT.
        let (cs, sn) = unsafe {
            (
                *table.get_unchecked(p * ROT + 2 * lane),
                *table.get_unchecked(p * ROT + 2 * lane + 1),
            )
        };
        let y = norm_turn(x, g, cs, sn, eps);
        let mut finite = true;
        for i in 0..4 {
            thread::__unroll_config::<0>();
            let h = f32_to_f16_bits(y[i]);
            finite &= half_bits_to_f32(h).is_finite();
            // SAFETY: j < c/4 <= ctx/4 <= pools, so j·DIM + d[i] < pools·DIM
            // <= pooled.len(); the rows of one launch hold distinct counts, so
            // one lane writes each value.
            unsafe { *pooled.get_unchecked_mut(j * DIM + d[i]) = h };
        }
        if !finite {
            fault.raise(FaultSite::PoolSelect);
        }
    }

    /// The score pass: block `b` takes pools `[32b, 32b + 32)` for every row
    /// `t < m` that is [`scored`] and sees that tile complete. Warp `w` is
    /// indexer head `w`. First every row's query heads are normed by `gain`
    /// and turned by the table's row `c − 1` into shared memory (block 0
    /// writes them to `q_out`, `[m][HEADS][DIM]`; a refused count's heads are
    /// NaN there); then the tile's keys are staged, and per row lane `j`
    /// scores pool `32b + j` (module doc) and writes `scores[t·pools + 32b +
    /// j]`.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            m <= 8,
            n_keys.len() >= m,
            q.len() >= m * 512,
            gain.len() >= 128,
            table.len() >= ctx * 64,
            pools * 4 >= ctx,
            pooled.len() >= pools * 128,
            q_out.len() >= m * 512,
            scores.len() >= m * pools
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    pub fn qsa_score(
        q: &[f32],
        gain: &[f32],
        table: &[f32],
        n_keys: &[u32],
        pooled: &[u16],
        eps: f32,
        ctx: u32,
        pools: u32,
        kept: u32,
        m: u32,
        fault: FaultSink,
        mut q_out: DisjointSlice<f32>,
        mut scores: DisjointSlice<f32>,
    ) {
        static mut QS: SharedArray<f32, { MAX_ROWS * HEADS * DIM }> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { POOL_TILE * K_STRIDE }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { HEADS * POOL_TILE }> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let w = tid / 32;
        let lane = warp::lane_id() as usize;
        let rows = m as usize;
        let p0 = b * POOL_TILE;
        // The largest complete pool count among the rows this tile serves;
        // the same in every thread.
        let mut live_max = 0usize;
        let mut t = 0usize;
        while t < rows {
            // SAFETY: t < m <= n_keys.len() by the launch contract.
            let c = unsafe { *n_keys.get_unchecked(t) };
            if scored(c, ctx, kept) {
                live_max = live_max.max(c as usize / POOL);
            }
            t += 1;
        }
        if p0 >= live_max && b != 0 {
            return; // block-uniform
        }

        // SAFETY: each `static mut` above is this block's own shared
        // allocation; the raw form reaches it without a reference. Every
        // index below is inside its array, and every write is ordered before
        // its reads by a block barrier.
        let (qs, ks, ws) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut QS),
                SharedArray::as_raw_mut_ptr(&raw mut KS),
                SharedArray::as_raw_mut_ptr(&raw mut WS),
            )
        };

        // ---- the query heads: warp w builds head w of every row.
        let d = lane_dims(lane);
        let mut g = [0.0f32; 4];
        for i in 0..4 {
            thread::__unroll_config::<0>();
            // SAFETY: d[i] < DIM <= gain.len() by the launch contract.
            g[i] = unsafe { *gain.get_unchecked(d[i]) };
        }
        let mut t = 0usize;
        while t < rows {
            // SAFETY: t < m <= n_keys.len().
            let c = unsafe { *n_keys.get_unchecked(t) };
            let base = (t * HEADS + w) * DIM;
            let y = if c >= 1 && c <= ctx {
                let mut x = [0.0f32; 4];
                for i in 0..4 {
                    thread::__unroll_config::<0>();
                    // SAFETY: base + d[i] < m·HEADS·DIM <= q.len().
                    x[i] = unsafe { *q.get_unchecked(base + d[i]) };
                }
                let p = c as usize - 1;
                // SAFETY: p < ctx and 2·lane + 1 < ROT: inside the table.
                let (cs, sn) = unsafe {
                    (
                        *table.get_unchecked(p * ROT + 2 * lane),
                        *table.get_unchecked(p * ROT + 2 * lane + 1),
                    )
                };
                norm_turn(x, g, cs, sn, eps)
            } else {
                [f32::NAN; 4]
            };
            for i in 0..4 {
                thread::__unroll_config::<0>();
                // SAFETY: base + d[i] < MAX_ROWS·HEADS·DIM; this lane's own.
                unsafe { *qs.add(base + d[i]) = y[i] };
                if b == 0 {
                    // SAFETY: base + d[i] < m·HEADS·DIM <= q_out.len();
                    // block 0 alone writes it, each value by one lane.
                    unsafe { *q_out.get_unchecked_mut(base + d[i]) = y[i] };
                }
            }
            t += 1;
        }
        if p0 >= live_max {
            return; // block-uniform: block 0 with no tile to score
        }

        // ---- the tile: pools p0 .. p0 + 32 below live_max, zeros past it.
        for i in 0..STAGE_PER {
            thread::__unroll_config::<0>();
            let e = tid + THREADS as usize * i;
            let key = e / ROW_QWORDS;
            let word = e - key * ROW_QWORDS;
            let kw = if p0 + key < live_max {
                // SAFETY: p0 + key < live_max <= ctx/4 <= pools, so the row
                // is inside pooled's pools·DIM u16 = pools·32 u64; the buffer
                // starts 8-byte aligned (a device allocation).
                unsafe { *(pooled.as_ptr() as *const u64).add((p0 + key) * ROW_QWORDS + word) }
            } else {
                0u64
            };
            // SAFETY: key < POOL_TILE and 2·word + 1 < K_STRIDE.
            unsafe {
                *ks.add(key * K_STRIDE + 2 * word) = kw as u32;
                *ks.add(key * K_STRIDE + 2 * word + 1) = (kw >> 32) as u32;
            }
        }
        thread::sync_threads();

        let mut t = 0usize;
        while t < rows {
            // SAFETY: t < m <= n_keys.len().
            let c = unsafe { *n_keys.get_unchecked(t) };
            let nb = c as usize / POOL;
            // Block-uniform: c is the row's.
            if scored(c, ctx, kept) && p0 < nb {
                let live = p0 + lane < nb;
                let qh = (t * HEADS + w) * DIM;
                let mut a = [0.0f32; ILP];
                let mut wd = 0usize;
                while wd < ROW_WORDS {
                    for p in 0..ILP {
                        thread::__unroll_config::<0>();
                        // SAFETY: lane < POOL_TILE and wd + p < ROW_WORDS:
                        // inside KS; qh + 2(wd + p) + 1 < MAX_ROWS·HEADS·DIM.
                        let (kw, q0, q1) = unsafe {
                            (
                                *ks.add(lane * K_STRIDE + wd + p),
                                *qs.add(qh + 2 * (wd + p)),
                                *qs.add(qh + 2 * (wd + p) + 1),
                            )
                        };
                        a[p] = f32::mul_add(q0, half_bits_to_f32(kw as u16), a[p]);
                        a[p] = f32::mul_add(q1, half_bits_to_f32((kw >> 16) as u16), a[p]);
                    }
                    wd += ILP;
                }
                let dot = (a[0] + a[1]) + (a[2] + a[3]);
                if live && !dot.is_finite() {
                    fault.raise(FaultSite::PoolSelect);
                }
                // SAFETY: w < HEADS and lane < POOL_TILE: this lane's slot.
                unsafe { *ws.add(w * POOL_TILE + lane) = relu(dot) };
                thread::sync_threads();
                if w == 0 && live {
                    // SAFETY: the four slots of lane `lane`, written
                    // before the barrier above.
                    let s = unsafe {
                        ((*ws.add(lane) + *ws.add(POOL_TILE + lane))
                            + *ws.add(2 * POOL_TILE + lane))
                            + *ws.add(3 * POOL_TILE + lane)
                    };
                    let pool = p0 + lane;
                    // SAFETY: pool < nb <= ctx/4 <= pools and t < m, so
                    // t·pools + pool < m·pools <= scores.len(); one lane
                    // writes each score.
                    unsafe { *scores.get_unchecked_mut(t * pools as usize + pool) = s };
                }
                thread::sync_threads();
            }
            t += 1;
        }
    }

    /// The top-k pass: block `t` writes row `t`'s list into `list[t·width
    /// ..]` and its length into `n_sel[t]` (module doc). A [`scored`] row
    /// reads its scores; a row that sees at most `kept` pools gets `0 .. c`;
    /// a refused count gets length 0.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys.len() >= m,
            pools * 4 >= ctx,
            scores.len() >= m * pools,
            width >= kept * 4 + 3,
            list.len() >= m * width,
            n_sel.len() >= m
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    pub fn qsa_topk(
        n_keys: &[u32],
        scores: &[f32],
        ctx: u32,
        pools: u32,
        kept: u32,
        width: u32,
        m: u32,
        mut list: DisjointSlice<u32>,
        mut n_sel: DisjointSlice<u32>,
    ) {
        static mut FINE: SharedArray<u32, FINE_BINS> = SharedArray::UNINIT;
        static mut WSUM: SharedArray<u32, TOPK_WARPS> = SharedArray::UNINIT;
        static mut WTOT: SharedArray<u32, { 2 * TOPK_WARPS }> = SharedArray::UNINIT;
        static mut PICK: SharedArray<u32, 2> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return; // block-uniform
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id();
        let wid = tid / 32;
        let lbase = t * width as usize;
        // SAFETY: t < m <= n_keys.len() by the launch contract.
        let c = unsafe { *n_keys.get_unchecked(t) };
        if !scored(c, ctx, kept) {
            let len = if c >= 1 && c <= ctx { c as usize } else { 0 };
            let mut i = tid;
            while i < len {
                // SAFETY: a row that does not select has c/4 <= kept, so c <=
                // 4·kept + 3 <= width and lbase + i < m·width <= list.len();
                // thread tid alone writes entries tid + 512·r.
                unsafe { *list.get_unchecked_mut(lbase + i) = i as u32 };
                i += TOPK_THREADS as usize;
            }
            if tid == 0 {
                // SAFETY: t < m <= n_sel.len().
                unsafe { *n_sel.get_unchecked_mut(t) = len as u32 };
            }
            return; // block-uniform
        }
        // SAFETY: block-shared statics; the raw forms reach them without a
        // reference, and every access below is bounded and barrier-ordered.
        let (fine, wsum, wtot, pick) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut FINE),
                SharedArray::as_raw_mut_ptr(&raw mut WSUM),
                SharedArray::as_raw_mut_ptr(&raw mut WTOT),
                SharedArray::as_raw_mut_ptr(&raw mut PICK),
            )
        };
        let n = c as usize / POOL;
        let sbase = t * pools as usize;
        let need = kept;

        // ---- the top ten bits. Every count below is of the row's own `n`
        // scores, read in this launch: they total `n > kept`, so every pick
        // finds its bin whatever the scores hold.
        // SAFETY: n <= ctx/4 <= pools, so sbase + n <= m·pools <= scores.len();
        // FINE is this pass's alone.
        let hc = unsafe { coarse(scores, sbase, n, fine, tid) };
        // SAFETY: WSUM and PICK are used by nothing else across the call.
        let (b1, above1) = unsafe {
            pick_bin::<COARSE_PER>(hc, HIST_BINS as u32 - 1, need, lane, tid, wsum, pick)
        };
        let need2 = need - above1;

        // ---- the next eleven bits, then the last eleven.
        // SAFETY: n <= ctx/4 <= pools, so sbase + n <= m·pools <= scores.len();
        // FINE is this pass's alone.
        let fc = unsafe { refine(scores, sbase, n, b1, 11, fine, tid) };
        // SAFETY: WSUM and PICK are used by nothing else across the call.
        let (b2, above2) =
            unsafe { pick_bin::<FINE_PER>(fc, FINE_BINS as u32 - 1, need2, lane, tid, wsum, pick) };
        let need3 = need2 - above2;
        let prefix = (b1 << 11) | b2;
        // SAFETY: as the pass above.
        let fc = unsafe { refine(scores, sbase, n, prefix, 0, fine, tid) };
        // SAFETY: WSUM and PICK are used by nothing else across the call.
        let (b3, above3) =
            unsafe { pick_bin::<FINE_PER>(fc, FINE_BINS as u32 - 1, need3, lane, tid, wsum, pick) };
        let thr = (prefix << 11) | b3;
        let take_eq = need - (above1 + above2 + above3);

        // ---- the list, in pool order: warp `wid` owns pools `c0 .. c1`,
        // counts the pools it keeps, the block scans the counts, and the warp
        // walks its pools again, writing each kept pool's tokens at its place.
        let chunk = n.div_ceil(32 * TOPK_WARPS) * 32;
        let c0 = (wid * chunk).min(n);
        let c1 = (c0 + chunk).min(n);
        let l = lane as usize;
        let (mut tg, mut te) = (0u32, 0u32);
        let mut base = c0;
        while base < c1 {
            // SAFETY: c1 <= n, so sbase + c1 <= scores.len().
            let kk = unsafe { lane_keys(scores, sbase, base, c1, l) };
            for r in 0..LANE_ROWS {
                thread::__unroll_config::<0>();
                let live = base + 32 * r + l < c1;
                tg += warp::ballot(live && kk[r] > thr).count_ones();
                te += warp::ballot(live && kk[r] == thr).count_ones();
            }
            base += 32 * LANE_ROWS;
        }
        if lane == 0 {
            // SAFETY: wid < TOPK_WARPS bounds both slots; lane 0 of warp wid is
            // their only writer, before the barrier.
            unsafe {
                *wtot.add(wid) = tg;
                *wtot.add(TOPK_WARPS + wid) = te;
            }
        }
        thread::sync_threads();
        let (mut og, mut oe) = (0u32, 0u32);
        for w in 0..TOPK_WARPS {
            thread::__unroll_config::<0>();
            if w < wid {
                // SAFETY: w < TOPK_WARPS; published by the barrier above.
                unsafe {
                    og += *wtot.add(w);
                    oe += *wtot.add(TOPK_WARPS + w);
                }
            }
        }
        let lt = warp::lanemask_lt();
        let mut base = c0;
        while base < c1 {
            // SAFETY: c1 <= n, so sbase + c1 <= scores.len().
            let kk = unsafe { lane_keys(scores, sbase, base, c1, l) };
            for r in 0..LANE_ROWS {
                thread::__unroll_config::<0>();
                let i = base + 32 * r + l;
                let live = i < c1;
                let mg = warp::ballot(live && kk[r] > thr);
                let me = warp::ballot(live && kk[r] == thr);
                let gt_before = og + (mg & lt).count_ones();
                let eq_before = oe + (me & lt).count_ones();
                let is_gt = (mg >> lane) & 1 != 0;
                let is_eq = (me >> lane) & 1 != 0;
                if is_gt || (is_eq && eq_before < take_eq) {
                    let at = lbase + POOL * (gt_before + eq_before.min(take_eq)) as usize;
                    // SAFETY: the pools kept number `kept`, and this one's
                    // place counts those before it, so it is below kept and
                    // at + 3 < lbase + 4·kept <= t·width + width <=
                    // list.len(); each place has one pool.
                    unsafe {
                        for q in 0..POOL {
                            thread::__unroll_config::<0>();
                            *list.get_unchecked_mut(at + q) = (POOL * i + q) as u32;
                        }
                    }
                }
                og += mg.count_ones();
                oe += me.count_ones();
            }
            base += 32 * LANE_ROWS;
        }
        // ---- the tail and the length.
        let tail = c as usize - POOL * n;
        let kept_tokens = POOL * kept as usize;
        if tid < tail {
            // SAFETY: tid < tail < POOL, so the entry is below 4·kept + 3 <=
            // width.
            unsafe { *list.get_unchecked_mut(lbase + kept_tokens + tid) = (POOL * n + tid) as u32 };
        }
        if tid == 0 {
            // SAFETY: t < m <= n_sel.len().
            unsafe { *n_sel.get_unchecked_mut(t) = (kept_tokens + tail) as u32 };
        }
    }

    /// [`qsa_topk`] with the higher pools of a tie at the cut: block `t`
    /// writes row `t`'s list into `list[t·width ..]` and its length into
    /// `vis[2t + 1]`, the visible count the selected attention reads
    /// (`vis[2t]` is not written). A [`scored`] row keeps every pool above
    /// the `kept`-th order key and the last pools equal to it in pool order,
    /// then its tail; a row that sees at most `kept` pools gets `0 .. c`; a
    /// refused count (0, or past the cache) gets length 0 and raises
    /// [`FaultSite::PoolSelect`].
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            n_keys.len() >= m,
            pools * 4 >= ctx,
            scores.len() >= m * pools,
            width >= kept * 4 + 3,
            list.len() >= m * width,
            vis.len() >= 2 * m
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    pub fn qsa_topk_high(
        n_keys: &[u32],
        scores: &[f32],
        ctx: u32,
        pools: u32,
        kept: u32,
        width: u32,
        m: u32,
        fault: FaultSink,
        mut list: DisjointSlice<u32>,
        mut vis: DisjointSlice<u32>,
    ) {
        static mut FINE: SharedArray<u32, FINE_BINS> = SharedArray::UNINIT;
        static mut WSUM: SharedArray<u32, TOPK_WARPS> = SharedArray::UNINIT;
        static mut WTOT: SharedArray<u32, { 2 * TOPK_WARPS }> = SharedArray::UNINIT;
        static mut PICK: SharedArray<u32, 2> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        if t >= m as usize {
            return; // block-uniform
        }
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id();
        let wid = tid / 32;
        let lbase = t * width as usize;
        // SAFETY: t < m <= n_keys.len() by the launch contract.
        let c = unsafe { *n_keys.get_unchecked(t) };
        if !scored(c, ctx, kept) {
            let accepted = c >= 1 && c <= ctx;
            let len = if accepted { c as usize } else { 0 };
            let mut i = tid;
            while i < len {
                // SAFETY: a row that does not select has c/4 <= kept, so c <=
                // 4·kept + 3 <= width and lbase + i < m·width <= list.len();
                // thread tid alone writes entries tid + 512·r.
                unsafe { *list.get_unchecked_mut(lbase + i) = i as u32 };
                i += TOPK_THREADS as usize;
            }
            if tid == 0 {
                if !accepted {
                    fault.raise(FaultSite::PoolSelect);
                }
                // SAFETY: 2t + 1 < 2m <= vis.len() (launch contract).
                unsafe { *vis.get_unchecked_mut(2 * t + 1) = len as u32 };
            }
            return; // block-uniform
        }
        // SAFETY: block-shared statics; the raw forms reach them without a
        // reference, and every access below is bounded and barrier-ordered.
        let (fine, wsum, wtot, pick) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut FINE),
                SharedArray::as_raw_mut_ptr(&raw mut WSUM),
                SharedArray::as_raw_mut_ptr(&raw mut WTOT),
                SharedArray::as_raw_mut_ptr(&raw mut PICK),
            )
        };
        let n = c as usize / POOL;
        let sbase = t * pools as usize;
        let need = kept;

        // ---- the exact `kept`-th order key, as `qsa_topk` finds it.
        // SAFETY: n <= ctx/4 <= pools, so sbase + n <= m·pools <= scores.len();
        // FINE is this pass's alone.
        let hc = unsafe { coarse(scores, sbase, n, fine, tid) };
        // SAFETY: WSUM and PICK are used by nothing else across the call.
        let (b1, above1) = unsafe {
            pick_bin::<COARSE_PER>(hc, HIST_BINS as u32 - 1, need, lane, tid, wsum, pick)
        };
        let need2 = need - above1;
        // SAFETY: as the pass above.
        let fc = unsafe { refine(scores, sbase, n, b1, 11, fine, tid) };
        // SAFETY: WSUM and PICK are used by nothing else across the call.
        let (b2, above2) =
            unsafe { pick_bin::<FINE_PER>(fc, FINE_BINS as u32 - 1, need2, lane, tid, wsum, pick) };
        let need3 = need2 - above2;
        let prefix = (b1 << 11) | b2;
        // SAFETY: as the pass above.
        let fc = unsafe { refine(scores, sbase, n, prefix, 0, fine, tid) };
        // SAFETY: WSUM and PICK are used by nothing else across the call.
        let (b3, above3) =
            unsafe { pick_bin::<FINE_PER>(fc, FINE_BINS as u32 - 1, need3, lane, tid, wsum, pick) };
        let thr = (prefix << 11) | b3;
        let take_eq = need - (above1 + above2 + above3);

        // ---- the list, in pool order: warp `wid` owns pools `c0 .. c1` and
        // counts the pools above the key and equal to it; the block's total
        // of equal pools says how many of the first ones in pool order are
        // left out, and the warp walks its pools again, writing each kept
        // pool's tokens at its place.
        let chunk = n.div_ceil(32 * TOPK_WARPS) * 32;
        let c0 = (wid * chunk).min(n);
        let c1 = (c0 + chunk).min(n);
        let l = lane as usize;
        let (mut tg, mut te) = (0u32, 0u32);
        let mut base = c0;
        while base < c1 {
            // SAFETY: c1 <= n, so sbase + c1 <= scores.len().
            let kk = unsafe { lane_keys(scores, sbase, base, c1, l) };
            for r in 0..LANE_ROWS {
                thread::__unroll_config::<0>();
                let live = base + 32 * r + l < c1;
                tg += warp::ballot(live && kk[r] > thr).count_ones();
                te += warp::ballot(live && kk[r] == thr).count_ones();
            }
            base += 32 * LANE_ROWS;
        }
        if lane == 0 {
            // SAFETY: wid < TOPK_WARPS bounds both slots; lane 0 of warp wid is
            // their only writer, before the barrier.
            unsafe {
                *wtot.add(wid) = tg;
                *wtot.add(TOPK_WARPS + wid) = te;
            }
        }
        thread::sync_threads();
        let (mut og, mut oe, mut te_all) = (0u32, 0u32, 0u32);
        for w in 0..TOPK_WARPS {
            thread::__unroll_config::<0>();
            // SAFETY: w < TOPK_WARPS; published by the barrier above.
            let (g, e) = unsafe { (*wtot.add(w), *wtot.add(TOPK_WARPS + w)) };
            if w < wid {
                og += g;
                oe += e;
            }
            te_all += e;
        }
        // The equal pools left out: the first `skip` of them in pool order.
        let skip = te_all - take_eq;
        let lt = warp::lanemask_lt();
        let mut base = c0;
        while base < c1 {
            // SAFETY: c1 <= n, so sbase + c1 <= scores.len().
            let kk = unsafe { lane_keys(scores, sbase, base, c1, l) };
            for r in 0..LANE_ROWS {
                thread::__unroll_config::<0>();
                let i = base + 32 * r + l;
                let live = i < c1;
                let mg = warp::ballot(live && kk[r] > thr);
                let me = warp::ballot(live && kk[r] == thr);
                let gt_before = og + (mg & lt).count_ones();
                let eq_before = oe + (me & lt).count_ones();
                let is_gt = (mg >> lane) & 1 != 0;
                let is_eq = (me >> lane) & 1 != 0;
                if is_gt || (is_eq && eq_before >= skip) {
                    let at = lbase + POOL * (gt_before + eq_before.saturating_sub(skip)) as usize;
                    // SAFETY: the pools kept number `kept`, and this one's
                    // place counts the kept ones before it, so it is below
                    // kept and at + 3 < lbase + 4·kept <= t·width + width <=
                    // list.len(); each place has one pool.
                    unsafe {
                        for q in 0..POOL {
                            thread::__unroll_config::<0>();
                            *list.get_unchecked_mut(at + q) = (POOL * i + q) as u32;
                        }
                    }
                }
                og += mg.count_ones();
                oe += me.count_ones();
            }
            base += 32 * LANE_ROWS;
        }
        // ---- the tail and the length.
        let tail = c as usize - POOL * n;
        let kept_tokens = POOL * kept as usize;
        if tid < tail {
            // SAFETY: tid < tail < POOL, so the entry is below 4·kept + 3 <=
            // width.
            unsafe { *list.get_unchecked_mut(lbase + kept_tokens + tid) = (POOL * n + tid) as u32 };
        }
        if tid == 0 {
            // SAFETY: 2t + 1 < 2m <= vis.len() (launch contract).
            unsafe { *vis.get_unchecked_mut(2 * t + 1) = (kept_tokens + tail) as u32 };
        }
    }
}

// -------------------------------------------------------------- launchers

/// What one selection leaves besides the list, for up to [`MAX_ROWS`] rows
/// over a cache of `ctx` rows, allocated once and reused by every launch: the
/// normed and turned query heads and the scores; then the lists and their
/// lengths the selected flash reads.
pub struct QsaScratch {
    /// `[rows][HEADS][DIM]`: the query heads the scores used.
    pub q_out: DeviceBuffer<f32>,
    /// `[rows][pools]`: each scored row's pool scores.
    pub scores: DeviceBuffer<f32>,
    /// `[rows][width]`: each row's tokens, ascending.
    pub list: DeviceBuffer<u32>,
    /// `[rows]`: each list's length.
    pub n_sel: DeviceBuffer<u32>,
    rows: usize,
    pools: usize,
    width: usize,
}

impl QsaScratch {
    /// Allocate for `rows` rows over a `ctx`-row cache keeping `kept` pools.
    /// Load-time only.
    ///
    /// # Errors
    /// [`GpuError::Shape`] for `rows` outside `1..=MAX_ROWS` or `kept` zero;
    /// an allocation error.
    pub fn new(
        stream: &CudaStream,
        rows: usize,
        ctx: usize,
        kept: usize,
    ) -> Result<QsaScratch, GpuError> {
        if rows == 0 || rows > MAX_ROWS || kept == 0 {
            return Err(GpuError::shape(
                "qsa::scratch",
                format!("need 1..={MAX_ROWS} rows and kept >= 1, got rows={rows} kept={kept}"),
            ));
        }
        let pools = pools_for(ctx);
        let width = list_width(kept);
        Ok(QsaScratch {
            q_out: DeviceBuffer::zeroed(stream, rows * HEADS * DIM)?,
            scores: DeviceBuffer::zeroed(stream, rows * pools)?,
            list: DeviceBuffer::zeroed(stream, rows * width)?,
            n_sel: DeviceBuffer::zeroed(stream, rows)?,
            rows,
            pools,
            width,
        })
    }

    /// The list stride: [`list_width`] of the scratch's `kept`.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }
}

/// Pooled rows a `ctx`-row cache needs: one per pool, the last one
/// incomplete when `ctx` is not a multiple of [`POOL`].
#[must_use]
pub fn pools_for(ctx: usize) -> usize {
    ctx.div_ceil(POOL)
}

/// [`QsaKernels::enqueue_pool`]'s arguments: the layer's raw indexer keys
/// (`[ctx][DIM]` f16, the rows the call wrote included), the key norm's gain
/// (`DIM`), the rope table (`ctx` rows of [`ROT`]), the rows' live counts on
/// the device (`m`), the norm's epsilon, and the pooled plane
/// (`[pools_for(ctx)][DIM]` f16).
pub struct PoolArgs<'a> {
    pub raw: &'a DeviceBuffer<u16>,
    pub gain: &'a DeviceBuffer<f32>,
    pub table: &'a DeviceBuffer<f32>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub eps: f32,
    pub ctx: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub pooled: &'a mut DeviceBuffer<u16>,
}

/// [`QsaKernels::enqueue_select`]'s arguments: the rows' indexer query
/// projections (`[m][HEADS][DIM]` f32, raw), the query norm's gain, the rope
/// table, the live counts, the pooled plane [`QsaKernels::enqueue_pool`]
/// filled, the epsilon, the cache height, the pools kept, and the scratch the
/// lists land in.
pub struct SelectArgs<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub gain: &'a DeviceBuffer<f32>,
    pub table: &'a DeviceBuffer<f32>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub pooled: &'a DeviceBuffer<u16>,
    pub eps: f32,
    pub ctx: usize,
    pub kept: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub scratch: &'a mut QsaScratch,
}

/// [`QsaKernels::enqueue_topk_high`]'s arguments: the `m` rows' live counts,
/// their scores (`[m][pools_for(ctx)]`, a scored row's first `count / 4`
/// written by its score pass), the cache height, the pools kept, and where
/// the lists (`[m][list_width(kept)]`) and the visible counts (`[m][2]`, the
/// second word of each) land.
pub struct TopkHighArgs<'a> {
    pub n_keys: &'a DeviceBuffer<u32>,
    pub scores: &'a DeviceBuffer<f32>,
    pub ctx: usize,
    pub kept: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub list: &'a mut DeviceBuffer<u32>,
    pub vis: &'a mut DeviceBuffer<u32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct QsaKernels {
    module: qsa_kernels::LoadedModule,
}

impl QsaKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<QsaKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { crate::shared_module!(qsa_kernels, ctx)? };
        Ok(QsaKernels { module })
    }

    /// Enqueue the pool pass for `m` rows: every row whose count completes a
    /// pool writes that pool's key. One launch of `⌈m/4⌉` blocks; it must
    /// follow the call's raw-key writes and precede its selection, on every
    /// call from position 0 on — a pool no launch wrote is never written.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_pool(&self, stream: &CudaStream, a: PoolArgs<'_>) -> Result<(), GpuError> {
        let what = "qsa::enqueue_pool";
        if a.ctx == 0 || a.m == 0 {
            return Err(GpuError::shape(
                what,
                format!("need ctx and m >= 1, got ctx={} m={}", a.ctx, a.m),
            ));
        }
        let pools = pools_for(a.ctx);
        let lens = [
            ("raw", a.raw.len(), a.ctx * DIM),
            ("gain", a.gain.len(), DIM),
            ("table", a.table.len(), a.ctx * ROT),
            ("n_keys", a.n_keys.len(), a.m),
            ("pooled", a.pooled.len(), pools * DIM),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let grid = launch_u32(what, "grid", a.m.div_ceil(WARPS))?;
        let cfg = LaunchConfig1D::new(grid, THREADS, 0);
        let prep = self.module.prepare_qsa_pool(cfg)?;
        self.module.qsa_pool(
            stream,
            &prep,
            a.raw,
            a.gain,
            a.table,
            a.n_keys,
            a.eps,
            launch_u32(what, "ctx", a.ctx)?,
            launch_u32(what, "pools", pools)?,
            launch_u32(what, "m", a.m)?,
            a.fault,
            a.pooled,
        )?;
        Ok(())
    }

    /// Enqueue the selection of `m` rows: the score pass (`⌈pools/32⌉` blocks
    /// of 128) and the top-k pass (`m` blocks of 512), each row's list and
    /// length left in `a.scratch`. `m` is 1 to the scratch's rows (at most
    /// [`MAX_ROWS`]) and the cache and `kept` the scratch's; any other shape
    /// is refused by name. Two launches, in order, on one stream: the top-k
    /// reads the scores the score pass wrote. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_select(&self, stream: &CudaStream, a: SelectArgs<'_>) -> Result<(), GpuError> {
        let what = "qsa::enqueue_select";
        let s = a.scratch;
        let pools = pools_for(a.ctx);
        if a.m == 0 || a.m > s.rows || a.kept == 0 || s.pools != pools {
            return Err(GpuError::shape(
                what,
                format!(
                    "need 1..={} rows, kept >= 1 and the scratch's {} pools; got m={} kept={} \
                     pools={pools}",
                    s.rows, s.pools, a.m, a.kept
                ),
            ));
        }
        if s.width != list_width(a.kept) {
            return Err(GpuError::shape(
                what,
                format!(
                    "the scratch's lists are {} wide, kept {} needs {}",
                    s.width,
                    a.kept,
                    list_width(a.kept)
                ),
            ));
        }
        let lens = [
            ("q", a.q.len(), a.m * HEADS * DIM),
            ("gain", a.gain.len(), DIM),
            ("table", a.table.len(), a.ctx * ROT),
            ("n_keys", a.n_keys.len(), a.m),
            ("pooled", a.pooled.len(), pools * DIM),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let ctx = launch_u32(what, "ctx", a.ctx)?;
        let pools_u = launch_u32(what, "pools", pools)?;
        let kept = launch_u32(what, "kept", a.kept)?;
        let width = launch_u32(what, "width", s.width)?;
        let m = launch_u32(what, "m", a.m)?;
        let grid = launch_u32(what, "score grid", pools.div_ceil(POOL_TILE))?;
        let prep = self
            .module
            .prepare_qsa_score(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.qsa_score(
            stream,
            &prep,
            a.q,
            a.gain,
            a.table,
            a.n_keys,
            a.pooled,
            a.eps,
            ctx,
            pools_u,
            kept,
            m,
            a.fault,
            &mut s.q_out,
            &mut s.scores,
        )?;
        let prep = self
            .module
            .prepare_qsa_topk(LaunchConfig1D::new(m, TOPK_THREADS, 0))?;
        self.module.qsa_topk(
            stream,
            &prep,
            a.n_keys,
            &s.scores,
            ctx,
            pools_u,
            kept,
            width,
            m,
            &mut s.list,
            &mut s.n_sel,
        )?;
        Ok(())
    }

    /// Enqueue the top-k pass with the higher pools of a tie
    /// ([`qsa_kernels::qsa_topk_high`]): `m` blocks of 512, each row's list
    /// and its length as the attention's visible count. Refused by name: no
    /// row, `kept` zero, scores, lists or counts shorter than `m` rows need.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_topk_high(
        &self,
        stream: &CudaStream,
        a: TopkHighArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "qsa::enqueue_topk_high";
        let pools = pools_for(a.ctx);
        let width = list_width(a.kept);
        if a.m == 0 || a.kept == 0 || a.ctx == 0 {
            return Err(GpuError::shape(
                what,
                format!(
                    "need m, kept and ctx >= 1, got m={} kept={} ctx={}",
                    a.m, a.kept, a.ctx
                ),
            ));
        }
        let lens = [
            ("n_keys", a.n_keys.len(), a.m),
            ("scores", a.scores.len(), a.m * pools),
            ("list", a.list.len(), a.m * width),
            ("vis", a.vis.len(), 2 * a.m),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("{name}.len() {got} < {need}"),
            ));
        }
        let m = launch_u32(what, "m", a.m)?;
        let prep = self
            .module
            .prepare_qsa_topk_high(LaunchConfig1D::new(m, TOPK_THREADS, 0))?;
        self.module.qsa_topk_high(
            stream,
            &prep,
            a.n_keys,
            a.scores,
            launch_u32(what, "ctx", a.ctx)?,
            launch_u32(what, "pools", pools)?,
            launch_u32(what, "kept", a.kept)?,
            launch_u32(what, "width", width)?,
            m,
            a.fault,
            a.list,
            a.vis,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scored_is_the_selecting_rows_with_an_accepted_count() {
        let (ctx, kept) = (4096u32, 512u32);
        assert!(!scored(0, ctx, kept));
        assert!(!scored(2051, ctx, kept));
        assert!(scored(2052, ctx, kept));
        assert!(scored(ctx, ctx, kept));
        assert!(!scored(ctx + 1, ctx, kept));
        assert_eq!(list_width(512), 2051);
        assert_eq!((pools_for(4096), pools_for(4097)), (1024, 1025));
    }

    #[test]
    fn order_key_orders_and_ties_the_zeros() {
        let v = [-1.0f32, -0.0, 0.0, 1.0e-30, 2.0, f32::INFINITY];
        for w in v.windows(2) {
            assert!(order_key(w[0]) <= order_key(w[1]));
        }
        assert_eq!(order_key(-0.0), order_key(0.0));
    }
}
