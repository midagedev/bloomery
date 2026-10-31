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
//! Four launches:
//! - [`qsa_kernels::qsa_pool`] — the key of every pool some row completes
//!   (`c % 4 == 0` completes pool `c/4 − 1`) from the raw indexer keys: the
//!   four rows' mean, the RMS gain norm, the NEOX turn of the first [`ROT`]
//!   values at the pool's first position, rounded once to f16 into the pooled
//!   plane. Every position passes through some row of some launch, so every
//!   pool is written exactly once, before any row can see it complete. A
//!   raw key is the f16 rounding of its projection: a row of the launch reads
//!   its own from the projection's output, a row before the launch from the
//!   raw key ring (`[RING_ROWS][DIM]` f16, position `p` in slot `p mod
//!   RING_ROWS`), which a pool reaches at most `POOL − 1` rows back. The
//!   launch's rows are consecutive positions (`n_keys[t] = n_keys[0] + t`);
//!   it runs before the launch's key append writes the ring.
//! - [`qsa_kernels::qsa_key_append`] — one thread per value: the raw key,
//!   `DIM` values a row of the key projection (token-major `[m][DIM]`),
//!   rounded once to f16; the call's last `min(m, RING_ROWS)` rows go into
//!   the raw key ring, after the call's pool pass read the slots of the
//!   positions before it.
//! - [`qsa_kernels::qsa_score`] — per row: its [`HEADS`] indexer query heads
//!   normed and turned at its position (every block of its group of
//!   [`MAX_ROWS`] rows builds them; the group's first block writes them to
//!   `q_out`), then per pool the score `Σ_h relu(q_h · k)`.
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
//!   the mean `(sum / 128) as f32`, the scale `1 / sqrt(mean + eps)` in f32
//!   (`elem::rms_scale_of_mean`: NaN where it is not finite and positive, so a
//!   head whose squares overflow is NaN and the checks below name it),
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
//! rounding (a non-finite raw key, or one past f16's range), a raw key that
//! is not finite after its rounding, a key row whose position is not `pos[0]`
//! plus its index and a head dot that is not finite raise
//! [`FaultSite::PoolSelect`], and a key row at or past the cache's `ctx` rows
//! raises [`FaultSite::CachePos`] and writes nothing; every other value is
//! written as computed, never replaced by a plausible one, and the list is
//! still a defined list of tokens below the row's count.
//!
//! The pool, key append and score passes each have a `_seqs` entry over the
//! rows of several sequences in one launch (`crate::seq_rows`), each
//! sequence's rows over its own ring, pooled plane and rope table; the top-k
//! pass reads only its rows' counts and scores, so one launch of it serves
//! every sequence.

use crate::elem::rms_scale_of_mean;
use crate::fault::{FaultSink, FaultSite};
use crate::flash::{f32_to_f16_bits, half_bits_to_f32};
use crate::rope_neox::neox_pair;
use crate::seq_rows::{SEQ_WORDS, SeqRows};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::ops::Range;
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
/// Rows one score block serves: a decode step's, a draft's verify, or one
/// group of a longer launch's rows (`⌈m / MAX_ROWS⌉` groups, each its own
/// blocks a pool tile).
pub const MAX_ROWS: usize = 8;
/// Pools one score block takes, one per lane.
pub const POOL_TILE: usize = 32;
/// Bins of the top-k's first histogram: the top ten bits of an order key.
const HIST_BINS: usize = 1024;
/// Slots of the raw key ring: the `POOL − 1` keys of a pool not yet complete
/// and a pass ([`crate::linear::PASS_ROWS`], `runtime::stores::index_ring_rows`),
/// so a launch from any of a pass's positions finds the keys its pools read
/// before it.
pub const RING_ROWS: usize = POOL - 1 + crate::linear::PASS_ROWS;
/// The longest list for `kept` pools: their tokens and a tail of `POOL − 1`.
#[must_use]
pub const fn list_width(kept: usize) -> usize {
    kept * POOL + POOL - 1
}

const _: () = assert!(HEADS == 4 && DIM == 128 && ROT == 64 && POOL == 4);
// The pool pass's launch contract names the ring's length as a literal.
const _: () = assert!(RING_ROWS * DIM == 1408);
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

/// Threads of a key append block, one a value.
const APPEND_THREADS: u32 = 256;
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
    let scale = rms_scale_of_mean(mean, eps);
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

/// [`pool_row`]'s inputs: the call's raw keys, the ring, the norm's gain,
/// the rotation table and the rows' key counts.
type PoolIn<'a> = (&'a [f32], &'a [u16], &'a [f32], &'a [f32], &'a [u32]);

/// Row `t` of [`qsa_kernels::qsa_pool`]'s launch (that entry's doc), the
/// warp's work: the pool its count completes, from the call's rows `kr`
/// before it and the ring, written through `pooled`.
///
/// # Safety
///
/// `qsa_pool`'s launch contract over these inputs and the plane `pooled`
/// addresses, `t` below its rows, and all 32 lanes of the warp here; no
/// other warp of the launch writes the pool row `t` completes.
#[inline(always)]
unsafe fn pool_row(
    t: usize,
    (kr, ring, gain, table, n_keys): PoolIn<'_>,
    eps: f32,
    ctx: u32,
    fault: FaultSink,
    pooled: *mut u16,
) {
    // SAFETY: t < m <= n_keys.len() by the launch contract.
    let c = unsafe { *n_keys.get_unchecked(t) };
    // SAFETY: t < m makes m nonzero, and m <= n_keys.len().
    let c0 = unsafe { *n_keys.get_unchecked(0) };
    if c0.checked_add(t as u32) != Some(c) {
        fault.raise(FaultSite::PoolSelect);
        return; // warp-uniform
    }
    if c == 0 || c > ctx || !c.is_multiple_of(POOL as u32) {
        return; // warp-uniform
    }
    for back in 1..POOL {
        if back <= t {
            // SAFETY: t − back < t < m <= n_keys.len().
            let before = unsafe { *n_keys.get_unchecked(t - back) };
            if c.checked_sub(back as u32) != Some(before) {
                fault.raise(FaultSite::PoolSelect);
                return; // warp-uniform
            }
        }
    }
    let j = (c / POOL as u32 - 1) as usize;
    let lane = warp::lane_id() as usize;
    let d = lane_dims(lane);
    let mut x = [0.0f32; 4];
    let mut g = [0.0f32; 4];
    for i in 0..4 {
        thread::__unroll_config::<0>();
        // Row k of the pool lies POOL − 1 − k rows before row t: the
        // call's row when it holds one, else the ring's slot.
        let row = |k: usize| {
            let back = POOL - 1 - k;
            let bits = if back <= t {
                // SAFETY: back <= t and t < m; d[i] < DIM, so
                // (t - back) * DIM + d[i] < m * DIM <= kr.len().
                let value = unsafe { *kr.get_unchecked((t - back) * DIM + d[i]) };
                f32_to_f16_bits(value)
            } else {
                // SAFETY: the slot is below RING_ROWS and d[i] < DIM,
                // so the index is below RING_ROWS * DIM <= ring.len().
                unsafe { *ring.get_unchecked((POOL * j + k) % RING_ROWS * DIM + d[i]) }
            };
            half_bits_to_f32(bits)
        };
        x[i] = (((row(0) + row(1)) + row(2)) + row(3)) * 0.25;
        // SAFETY: lane_dims gives d[i] < DIM <= gain.len().
        g[i] = unsafe { *gain.get_unchecked(d[i]) };
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
        // SAFETY: j < c/4 <= ctx/4 <= pools, so j·DIM + d[i] < pools·DIM,
        // inside the plane `pooled` addresses; the rows of one launch hold
        // distinct counts, so one lane writes each value.
        unsafe { *pooled.add(j * DIM + d[i]) = h };
    }
    if !finite {
        fault.raise(FaultSite::PoolSelect);
    }
}

/// [`score_block`]'s inputs: the indexer queries, the norm's gain, the
/// rotation table, the rows' key counts and the pooled keys.
type ScoreIn<'a> = (&'a [f32], &'a [f32], &'a [f32], &'a [u32], &'a [u16]);

/// Block `(group, tile)` of [`qsa_kernels::qsa_score`]'s launch of `m`
/// rows (that entry's doc), its query heads written through `q_out` and its
/// scores through `scores`, over the block's shared tiles `qs`, `ks`, `ws`.
/// Every index into a tile is inside it, and every write is ordered before
/// its reads by a block barrier.
///
/// # Safety
///
/// `qsa_score`'s launch contract over these inputs and the rows `q_out` and
/// `scores` address, a block of [`THREADS`], and `qs`, `ks`, `ws` the
/// block's own shared tiles of that entry's sizes.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn score_block(
    (group, tile): (usize, usize),
    (q, gain, table, n_keys, pooled): ScoreIn<'_>,
    eps: f32,
    (ctx, pools, kept, m): (u32, u32, u32, u32),
    fault: FaultSink,
    (q_out, scores): (*mut f32, *mut f32),
    (qs, ks, ws): (*mut f32, *mut u32, *mut f32),
) {
    let r0 = group * MAX_ROWS;
    if r0 >= m as usize {
        return; // block-uniform: past the launch's groups
    }
    let tid = thread::threadIdx_x() as usize;
    let w = tid / 32;
    let lane = warp::lane_id() as usize;
    let rows = (m as usize - r0).min(MAX_ROWS);
    let p0 = tile * POOL_TILE;
    // The largest complete pool count among the rows this tile serves;
    // the same in every thread.
    let mut live_max = 0usize;
    let mut t = 0usize;
    while t < rows {
        // SAFETY: r0 + t < m <= n_keys.len() by the launch contract.
        let c = unsafe { *n_keys.get_unchecked(r0 + t) };
        if scored(c, ctx, kept) {
            live_max = live_max.max(c as usize / POOL);
        }
        t += 1;
    }
    if p0 >= live_max && tile != 0 {
        return; // block-uniform
    }

    // ---- the query heads: warp w builds head w of every row of the group.
    let d = lane_dims(lane);
    let mut g = [0.0f32; 4];
    for i in 0..4 {
        thread::__unroll_config::<0>();
        // SAFETY: d[i] < DIM <= gain.len() by the launch contract.
        g[i] = unsafe { *gain.get_unchecked(d[i]) };
    }
    let mut t = 0usize;
    while t < rows {
        // SAFETY: r0 + t < m <= n_keys.len().
        let c = unsafe { *n_keys.get_unchecked(r0 + t) };
        let base = (t * HEADS + w) * DIM;
        let at = r0 * HEADS * DIM + base;
        let y = if c >= 1 && c <= ctx {
            let mut x = [0.0f32; 4];
            for i in 0..4 {
                thread::__unroll_config::<0>();
                // SAFETY: at + d[i] < m·HEADS·DIM <= q.len().
                x[i] = unsafe { *q.get_unchecked(at + d[i]) };
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
            if tile == 0 {
                // SAFETY: at + d[i] < m·HEADS·DIM, inside the rows
                // `q_out` addresses; the group's tile-0 block alone
                // writes it, each value by one lane.
                unsafe { *q_out.add(at + d[i]) = y[i] };
            }
        }
        t += 1;
    }
    if p0 >= live_max {
        return; // block-uniform: a tile-0 block with no tile to score
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
        // SAFETY: r0 + t < m <= n_keys.len().
        let c = unsafe { *n_keys.get_unchecked(r0 + t) };
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
                    ((*ws.add(lane) + *ws.add(POOL_TILE + lane)) + *ws.add(2 * POOL_TILE + lane))
                        + *ws.add(3 * POOL_TILE + lane)
                };
                if !s.is_finite() {
                    fault.raise(FaultSite::PoolSelect);
                }
                let pool = p0 + lane;
                // SAFETY: pool < nb <= ctx/4 <= pools and r0 + t < m, so
                // (r0 + t)·pools + pool < m·pools, inside the rows
                // `scores` addresses; one lane writes each score.
                unsafe { *scores.add((r0 + t) * pools as usize + pool) = s };
            }
            thread::sync_threads();
        }
        t += 1;
    }
}

/// Value `i` of [`qsa_kernels::qsa_key_append`]'s launch of `m` rows (that
/// entry's doc), the ring written through `ring`.
///
/// # Safety
///
/// `qsa_key_append`'s launch contract over `kr`, `pos` and the ring `ring`
/// addresses, and `i < DIM·m`.
#[inline(always)]
unsafe fn key_append_value(
    i: usize,
    (kr, pos): (&[f32], &[u32]),
    m: u32,
    ctx: u32,
    fault: FaultSink,
    ring: *mut u16,
) {
    let (c, r) = (i / DIM, i % DIM);
    // SAFETY: c = i / DIM < m <= pos.len() by the launch contract.
    let p = unsafe { *pos.get_unchecked(c) };
    // SAFETY: i < DIM·m makes m nonzero, and m <= pos.len().
    let p0 = unsafe { *pos.get_unchecked(0) };
    // SAFETY: i < DIM·m <= kr.len() by the launch contract.
    let x = unsafe { *kr.get_unchecked(i) };
    if p >= ctx {
        if r == 0 {
            fault.raise(FaultSite::CachePos);
        }
        return;
    }
    let bits = f32_to_f16_bits(x);
    if !half_bits_to_f32(bits).is_finite() {
        fault.raise(FaultSite::PoolSelect);
    }
    if p0.checked_add(c as u32) != Some(p) {
        if r == 0 {
            fault.raise(FaultSite::PoolSelect);
        }
        return;
    }
    if c + RING_ROWS < m as usize {
        return;
    }
    // SAFETY: the slot is below RING_ROWS, so slot·DIM + r < RING_ROWS·DIM,
    // inside the ring `ring` addresses; the rows written are consecutive
    // (`pos[0] + c`, checked above) and at most RING_ROWS, so their slots are
    // distinct and thread (c, r) is the value's only writer.
    unsafe {
        *ring.add(p as usize % RING_ROWS * DIM + r) = bits;
    }
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod qsa_kernels {
    use super::*;

    /// The pool pass: warp `w` of block `b` serves row `t = 4b + w`. A row
    /// whose live count `c` is accepted (`1..=ctx`) and a multiple of
    /// [`POOL`] completes pool `j = c/4 − 1`: lane `l` reads values
    /// [`lane_dims`] of the raw rows `q = 4j .. 4j + 3` (the f16 rounding of
    /// the call's row `t − (4j + 3 − q)` of `kr` when there is one, else ring
    /// slot `q mod RING_ROWS`, which holds that rounding's bits), takes their mean, norms it by `gain`, turns it by the
    /// table's row `4j` (`table[4j·ROT ..]`), and rounds the four values into
    /// `pooled[j·DIM ..]`. Any other row writes nothing. A row whose count is
    /// not `n_keys[0] + t`, or a completing row whose call rows before it do
    /// not count down by one each, raises [`FaultSite::PoolSelect`] and writes
    /// nothing: its rows of the call are not where the rule reads them. The
    /// warp's rows are its own; no barrier.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n_keys.len() >= m,
            kr.len() >= m * 128,
            ring.len() >= 1408,
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
        kr: &[f32],
        ring: &[u16],
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
        // SAFETY: the launch contract's lengths, t < m, and the whole warp
        // here; the pooled plane is this launch's own output.
        unsafe {
            pool_row(
                t,
                (kr, ring, gain, table, n_keys),
                eps,
                ctx,
                fault,
                pooled.as_mut_ptr(),
            );
        }
    }

    /// [`qsa_pool`] over several sequences' rows (`crate::seq_rows`): row `t`
    /// of sequence `s` (rows `lead .. lead + m_s`) is that sequence's launch's
    /// row `t − lead` over its rows of `kr` and `n_keys` and its own words,
    /// the raw key ring, the pooled plane and the rope table, in that order.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            n_keys.len() >= rows,
            kr.len() >= rows * 128,
            gain.len() >= 128
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn qsa_pool_seqs(
        kr: &[f32],
        gain: &[f32],
        n_keys: &[u32],
        eps: f32,
        ctx: u32,
        rows: u32,
        fault: FaultSink,
        #[grid_constant] seqs: &SeqRows,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let t = thread::blockIdx_x() as usize * WARPS + tid / 32;
        if t >= rows as usize {
            return; // warp-uniform: the row is the warp's, and no barrier follows
        }
        let s = seqs.seq_of(t);
        // SAFETY: s < the table's sequences (seq_of), whose rows tile the
        // launch's: row t is its row t − lead, the windows from its first row
        // hold its rows of kr and n_keys, and the launcher put its own ring
        // (RING_ROWS·DIM f16), pooled plane (pools_for(ctx)·DIM f16,
        // borrowed by this launch alone) and rope table (ctx·ROT f32) at its
        // words, so the call is that sequence's launch's warp over them.
        unsafe {
            let (lead, _) = seqs.span(s);
            let ring = seqs.word(s, 0) as usize as *const u16;
            let table = seqs.word(s, 2) as usize as *const f32;
            pool_row(
                t - lead,
                (
                    kr.get_unchecked(lead * DIM..),
                    core::slice::from_raw_parts(ring, RING_ROWS * DIM),
                    gain,
                    core::slice::from_raw_parts(table, ctx as usize * ROT),
                    n_keys.get_unchecked(lead..),
                ),
                eps,
                ctx,
                fault,
                seqs.word(s, 1) as usize as *mut u16,
            );
        }
    }

    /// Row `c` of `m` indexer keys — value `r` at `kr[c·DIM + r]`, the key
    /// projection's token-major output — rounded once to f16; a row of the
    /// last [`RING_ROWS`] writes it into the ring's slot
    /// `pos[c] mod RING_ROWS`, [`DIM`] values a row. A position at or past
    /// `ctx` raises [`FaultSite::CachePos`] and writes nothing; a row whose
    /// position is not `pos[0] + c` raises [`FaultSite::PoolSelect`] and
    /// writes nothing, so no two rows share a slot; a value that is not
    /// finite after its rounding raises [`FaultSite::PoolSelect`], whichever
    /// row it is, and is written when its row is.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (kr.len() >= 128 * m, pos.len() >= m, ring.len() >= 1408)
    )]
    pub fn qsa_key_append(
        kr: &[f32],
        pos: &[u32],
        m: u32,
        ctx: u32,
        fault: FaultSink,
        mut ring: DisjointSlice<u16>,
    ) {
        let i = thread::index_1d().get();
        if i >= DIM * m as usize {
            return;
        }
        // SAFETY: the launch contract's lengths and i < DIM·m; the ring is
        // this launch's own output.
        unsafe { key_append_value(i, (kr, pos), m, ctx, fault, ring.as_mut_ptr()) };
    }

    /// [`qsa_key_append`] over several sequences' rows (`crate::seq_rows`):
    /// value `i` of row `c` of sequence `s` (rows `lead .. lead + m_s`) is that
    /// sequence's launch's value `i − lead·DIM` over its rows of `kr` and
    /// `pos` and its own raw key ring (word 0).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (kr.len() >= 128 * rows, pos.len() >= rows)
    )]
    pub fn qsa_key_append_seqs(
        kr: &[f32],
        pos: &[u32],
        rows: u32,
        ctx: u32,
        fault: FaultSink,
        #[grid_constant] seqs: &SeqRows,
    ) {
        let i = thread::index_1d().get();
        if i >= DIM * rows as usize {
            return;
        }
        let s = seqs.seq_of(i / DIM);
        // SAFETY: s < the table's sequences (seq_of), whose rows tile the
        // launch's: row i / DIM is its row i / DIM − lead, the windows from
        // its first row hold its m rows of kr and pos, and the launcher put
        // its own ring (RING_ROWS·DIM f16, borrowed by this launch alone) at
        // word 0, so the thread is that sequence's launch's over them.
        unsafe {
            let (lead, m) = seqs.span(s);
            key_append_value(
                i - lead * DIM,
                (kr.get_unchecked(lead * DIM..), pos.get_unchecked(lead..)),
                m as u32,
                ctx,
                fault,
                seqs.word(s, 0) as usize as *mut u16,
            );
        }
    }

    /// The score pass: block `b = g·tiles + i` (`tiles = ⌈pools/32⌉`) takes
    /// pools `[32i, 32i + 32)` for every row `t` of group `g` — rows `8g ..
    /// min(8g + 8, m)` — that is [`scored`] and sees that tile complete. Warp
    /// `w` is indexer head `w`. First the group's query heads are normed by
    /// `gain` and turned by the table's row `c − 1` into shared memory (the
    /// group's block `i = 0` writes them to `q_out`, `[m][HEADS][DIM]`; a
    /// refused count's heads are NaN there); then the tile's keys are staged,
    /// and per row lane `j` scores pool `32i + j` (module doc) and writes
    /// `scores[t·pools + 32i + j]`. A row's heads and scores are the same
    /// whatever group or launch it is in.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            pools >= 1,
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

        let tiles = (pools as usize).div_ceil(POOL_TILE);
        let b = thread::blockIdx_x() as usize;
        // SAFETY: each `static mut` above is this block's own shared
        // allocation; the raw form reaches it without a reference. The
        // launch contract's lengths and outputs, and block b of its grid.
        unsafe {
            score_block(
                (b / tiles, b % tiles),
                (q, gain, table, n_keys, pooled),
                eps,
                (ctx, pools, kept, m),
                fault,
                (q_out.as_mut_ptr(), scores.as_mut_ptr()),
                (
                    SharedArray::as_raw_mut_ptr(&raw mut QS),
                    SharedArray::as_raw_mut_ptr(&raw mut KS),
                    SharedArray::as_raw_mut_ptr(&raw mut WS),
                ),
            );
        }
    }

    /// [`qsa_score`] over several sequences' rows (`crate::seq_rows`): blocks
    /// `s·per .. (s + 1)·per`, `per = groups·tiles`, are sequence `s`'s launch,
    /// block `b − s·per` of it, over its rows `lead .. lead + m_s` of `q`,
    /// `n_keys`, `q_out` and `scores` and its own pooled plane and rope
    /// table (words 1 and 2);
    /// `groups` is `⌈most / MAX_ROWS⌉` for the longest sequence's `most`
    /// rows, and a block past its sequence's groups returns.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            pools >= 1,
            n_keys.len() >= rows,
            q.len() >= rows * 512,
            gain.len() >= 128,
            pools * 4 >= ctx,
            q_out.len() >= rows * 512,
            scores.len() >= rows * pools
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn qsa_score_seqs(
        q: &[f32],
        gain: &[f32],
        n_keys: &[u32],
        eps: f32,
        ctx: u32,
        pools: u32,
        kept: u32,
        rows: u32,
        groups: u32,
        fault: FaultSink,
        mut q_out: DisjointSlice<f32>,
        mut scores: DisjointSlice<f32>,
        #[grid_constant] seqs: &SeqRows,
    ) {
        static mut QS: SharedArray<f32, { MAX_ROWS * HEADS * DIM }> = SharedArray::UNINIT;
        static mut KS: SharedArray<u32, { POOL_TILE * K_STRIDE }> = SharedArray::UNINIT;
        static mut WS: SharedArray<f32, { HEADS * POOL_TILE }> = SharedArray::UNINIT;

        // rows only bounds the lengths in the launch contract.
        let _ = rows;
        let tiles = (pools as usize).div_ceil(POOL_TILE);
        let per = groups as usize * tiles;
        let b = thread::blockIdx_x() as usize;
        let s = b / per;
        if s >= seqs.len() {
            return; // block-uniform
        }
        let r = b - s * per;
        let (q_at, p_at) = (HEADS * DIM, pools as usize);
        // SAFETY: s < the table's sequences, whose rows tile the launch's:
        // the windows from its first row hold its m rows of q, n_keys, q_out
        // and scores, and the launcher put its own pooled plane
        // (pools·DIM f16) and rope table (ctx·ROT f32) at words 1 and 2, so
        // the block is that sequence's
        // launch's block r over them (a group past its rows returns). Each
        // `static mut` above is this block's own shared allocation; the raw
        // form reaches it without a reference.
        unsafe {
            let (lead, m) = seqs.span(s);
            let pooled = seqs.word(s, 1) as usize as *const u16;
            let table = seqs.word(s, 2) as usize as *const f32;
            score_block(
                (r / tiles, r % tiles),
                (
                    q.get_unchecked(lead * q_at..),
                    gain,
                    core::slice::from_raw_parts(table, ctx as usize * ROT),
                    n_keys.get_unchecked(lead..),
                    core::slice::from_raw_parts(pooled, pools as usize * DIM),
                ),
                eps,
                (ctx, pools, kept, m as u32),
                fault,
                (
                    q_out.as_mut_ptr().add(lead * q_at),
                    scores.as_mut_ptr().add(lead * p_at),
                ),
                (
                    SharedArray::as_raw_mut_ptr(&raw mut QS),
                    SharedArray::as_raw_mut_ptr(&raw mut KS),
                    SharedArray::as_raw_mut_ptr(&raw mut WS),
                ),
            );
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

/// What one selection leaves besides the list, for up to `rows` rows over a
/// cache of `ctx` rows, allocated once and reused by every launch: the normed
/// and turned query heads and the scores; then the lists and their lengths the
/// selected flash reads.
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
    /// [`GpuError::Shape`] for `rows` or `kept` zero, or rows whose bytes
    /// ([`QsaScratch::row_bytes`] each) overflow; an allocation error.
    pub fn new(
        stream: &CudaStream,
        rows: usize,
        ctx: usize,
        kept: usize,
    ) -> Result<QsaScratch, GpuError> {
        let what = "qsa::scratch";
        if rows == 0 || kept == 0 {
            return Err(GpuError::shape(
                what,
                format!("need rows and kept >= 1, got rows={rows} kept={kept}"),
            ));
        }
        // Each buffer's length is at most the scratch's bytes.
        let row = QsaScratch::row_bytes(ctx, kept);
        if rows.checked_mul(row).is_none() {
            return Err(GpuError::shape(
                what,
                format!("{rows} rows of {row} B overflow"),
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

    /// The most rows one selection into this scratch takes.
    #[must_use]
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// Device bytes a scratch over a cache of `ctx` rows keeping `kept`
    /// pools holds a row: its query heads, scores, list and length, 4 B each.
    #[must_use]
    pub(crate) fn row_bytes(ctx: usize, kept: usize) -> usize {
        4 * (HEADS * DIM + pools_for(ctx) + list_width(kept) + 1)
    }
}

/// Pooled rows a `ctx`-row cache needs: one per pool, the last one
/// incomplete when `ctx` is not a multiple of [`POOL`].
#[must_use]
pub fn pools_for(ctx: usize) -> usize {
    ctx.div_ceil(POOL)
}

/// [`QsaKernels::enqueue_pool`]'s arguments: the call's raw indexer key
/// projections (`[m][DIM]` f32, token-major), the layer's raw key ring
/// (`[RING_ROWS][DIM]` f16, the keys of the positions before the call), the
/// key norm's gain (`DIM`), the rope table (`ctx` rows of [`ROT`]), the
/// rows' live counts on the device (`m`, consecutive), the norm's epsilon,
/// and the pooled plane (`[pools_for(ctx)][DIM]` f16).
pub struct PoolArgs<'a> {
    pub kr: &'a DeviceBuffer<f32>,
    pub ring: &'a DeviceBuffer<u16>,
    pub gain: &'a DeviceBuffer<f32>,
    pub table: &'a DeviceBuffer<f32>,
    pub n_keys: &'a DeviceBuffer<u32>,
    pub eps: f32,
    pub ctx: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub pooled: &'a mut DeviceBuffer<u16>,
}

/// One sequence of the indexer launches over several sequences' rows
/// ([`QsaKernels::enqueue_pool_seqs`], [`QsaKernels::enqueue_select_seqs`],
/// [`QsaKernels::enqueue_key_append_seqs`]): its rows of the launch, its raw
/// key ring (`[RING_ROWS][DIM]` f16), its pooled plane
/// (`[pools_for(ctx)][DIM]` f16) and its rope table (`ctx` rows of [`ROT`]).
pub(crate) struct QsaSeq<'a> {
    pub(crate) rows: Range<usize>,
    pub(crate) ring: &'a mut DeviceBuffer<u16>,
    pub(crate) pooled: &'a mut DeviceBuffer<u16>,
    pub(crate) rope: &'a DeviceBuffer<f32>,
}

/// The table of `seqs` for a launch of `what` over a `ctx`-row cache, each
/// sequence's words its ring, its pooled plane and its rope table; one
/// shorter than [`QsaSeq`] names, or rows that do not tile the launch, are
/// refused by name.
pub(crate) fn qsa_seqs(
    what: &'static str,
    seqs: &[QsaSeq<'_>],
    ctx: usize,
) -> Result<SeqRows, GpuError> {
    let pooled = pools_for(ctx) * DIM;
    let mut words = Vec::with_capacity(seqs.len());
    for (s, q) in seqs.iter().enumerate() {
        let lens = [
            ("ring", q.ring.len(), RING_ROWS * DIM),
            ("pooled", q.pooled.len(), pooled),
            ("rope", q.rope.len(), ctx * ROT),
        ];
        if let Some((name, got, need)) = lens.iter().find(|(_, got, need)| got < need) {
            return Err(GpuError::shape(
                what,
                format!("sequence {s}'s {name}.len() {got} < {need}"),
            ));
        }
        let mut word = [0; SEQ_WORDS];
        word[0] = q.ring.cu_deviceptr();
        word[1] = q.pooled.cu_deviceptr();
        word[2] = q.rope.cu_deviceptr();
        words.push((q.rows.clone(), word));
    }
    SeqRows::new(what, &words)
}

/// [`QsaKernels::enqueue_key_append`]'s arguments: the projection's `[m][DIM]`
/// raw keys and the `m` rows' positions, the cache's `ctx` rows, and the
/// layer's raw key ring (`[RING_ROWS][DIM]` f16).
pub struct KeyAppendArgs<'a> {
    pub kr: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub m: usize,
    pub ctx: usize,
    pub fault: FaultSink,
    pub ring: &'a mut DeviceBuffer<u16>,
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

/// [`QsaKernels::enqueue_pool_seqs`]'s arguments: [`PoolArgs`] over every
/// sequence's rows, each sequence's ring, pooled plane and rope table its
/// [`QsaSeq`]'s.
pub(crate) struct PoolSeqsArgs<'a> {
    pub(crate) kr: &'a DeviceBuffer<f32>,
    pub(crate) gain: &'a DeviceBuffer<f32>,
    pub(crate) n_keys: &'a DeviceBuffer<u32>,
    pub(crate) eps: f32,
    pub(crate) ctx: usize,
    pub(crate) fault: FaultSink,
}

/// [`QsaKernels::enqueue_select_seqs`]'s arguments: [`SelectArgs`] over
/// every sequence's rows, each sequence's pooled plane and rope table its
/// [`QsaSeq`]'s.
pub(crate) struct SelectSeqsArgs<'a> {
    pub(crate) q: &'a DeviceBuffer<f32>,
    pub(crate) gain: &'a DeviceBuffer<f32>,
    pub(crate) n_keys: &'a DeviceBuffer<u32>,
    pub(crate) eps: f32,
    pub(crate) ctx: usize,
    pub(crate) kept: usize,
    pub(crate) fault: FaultSink,
    pub(crate) scratch: &'a mut QsaScratch,
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

/// The refusals the key appends share over `m` rows: no row, no cache, and
/// keys or positions shorter than the rows.
fn check_key_append(
    what: &'static str,
    (kr, pos): (&DeviceBuffer<f32>, &DeviceBuffer<u32>),
    m: usize,
    ctx: usize,
) -> Result<(), GpuError> {
    if m == 0 || ctx == 0 {
        return Err(GpuError::shape(
            what,
            format!("need m and ctx >= 1, got m={m} ctx={ctx}"),
        ));
    }
    let lens = [("kr", kr.len(), DIM * m), ("pos", pos.len(), m)];
    crate::short(what, &lens)
}

/// A key append's launch over `m` rows: one thread a value, in blocks of
/// [`APPEND_THREADS`].
fn append_grid(what: &'static str, m: usize) -> Result<LaunchConfig1D, GpuError> {
    let blocks = launch_u32(what, "grid", (DIM * m).div_ceil(APPEND_THREADS as usize))?;
    Ok(LaunchConfig1D::new(blocks, APPEND_THREADS, 0))
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
    /// follow the call's key projection, precede the call's ring append and
    /// its selection, on every call from position 0 on — a pool no launch
    /// wrote is never written.
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
            ("kr", a.kr.len(), a.m * DIM),
            ("ring", a.ring.len(), RING_ROWS * DIM),
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
            a.kr,
            a.ring,
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

    /// Enqueue [`QsaKernels::enqueue_pool`]'s pass over every sequence of
    /// `seqs` in one launch (`crate::seq_rows`): `a`'s rows of every
    /// sequence, each over its own ring, pooled plane and rope table, bit
    /// for bit the
    /// launch of its rows alone.
    /// Refused by name: what that launch refuses over the rows, and the
    /// sequences [`qsa_seqs`] refuses. One launch of `⌈rows/4⌉` blocks,
    /// before the rows' key append. Asynchronous and capturable; builds its
    /// sequence table on the host.
    pub(crate) fn enqueue_pool_seqs(
        &self,
        stream: &CudaStream,
        a: PoolSeqsArgs<'_>,
        seqs: &[QsaSeq<'_>],
    ) -> Result<(), GpuError> {
        let what = "qsa::enqueue_pool_seqs";
        let table = qsa_seqs(what, seqs, a.ctx)?;
        let m = table.rows();
        if a.ctx == 0 {
            return Err(GpuError::shape(what, "need ctx >= 1, got ctx=0"));
        }
        let lens = [
            ("kr", a.kr.len(), m * DIM),
            ("gain", a.gain.len(), DIM),
            ("n_keys", a.n_keys.len(), m),
        ];
        crate::short(what, &lens)?;
        let grid = launch_u32(what, "grid", m.div_ceil(WARPS))?;
        let prep = self
            .module
            .prepare_qsa_pool_seqs(LaunchConfig1D::new(grid, THREADS, 0))?;
        // SAFETY: each sequence's words are its own ring, pooled plane and
        // rope table of the lengths `qsa_seqs` checked, borrowed by `seqs`
        // while this
        // enqueue runs; a graph that captures the launch replays it over the
        // same buffers, which the caller keeps alive and in place while the
        // graph lives, as for any buffer it captured.
        unsafe {
            self.module.qsa_pool_seqs(
                stream,
                &prep,
                a.kr,
                a.gain,
                a.n_keys,
                a.eps,
                launch_u32(what, "ctx", a.ctx)?,
                launch_u32(what, "rows", m)?,
                a.fault,
                table,
            )?;
        }
        Ok(())
    }

    /// Enqueue the raw key append of `m` rows ([`KeyAppendArgs`], module
    /// doc): `kr` the projection's `[m][DIM]` output, `pos` the rows'
    /// positions (consecutive, `pos[0] + c`; any other row raises
    /// `PoolSelect` on the device and writes nothing), `ring` the layer's
    /// `[RING_ROWS][DIM]` f16 ring. One launch, after the call's pool pass,
    /// which reads the ring's slots of the positions before the call.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_key_append(
        &self,
        stream: &CudaStream,
        a: KeyAppendArgs<'_>,
    ) -> Result<(), GpuError> {
        let KeyAppendArgs {
            kr,
            pos,
            m,
            ctx,
            fault,
            ring,
        } = a;
        let what = "qsa::enqueue_key_append";
        check_key_append(what, (kr, pos), m, ctx)?;
        if ring.len() < DIM * RING_ROWS {
            return Err(GpuError::shape(
                what,
                format!("ring.len() {} < {}", ring.len(), DIM * RING_ROWS),
            ));
        }
        let prep = self.module.prepare_qsa_key_append(append_grid(what, m)?)?;
        self.module.qsa_key_append(
            stream,
            &prep,
            kr,
            pos,
            launch_u32(what, "m", m)?,
            launch_u32(what, "ctx", ctx)?,
            fault,
            ring,
        )?;
        Ok(())
    }

    /// Enqueue [`QsaKernels::enqueue_key_append`] over every sequence of
    /// `seqs` in one launch (`crate::seq_rows`): `kr`'s and `pos`' rows of
    /// every sequence, each into its own ring, bit for bit the launch of its
    /// rows alone. Refused by name: what that launch refuses over the rows,
    /// and the sequences [`qsa_seqs`] refuses. One launch, after the rows'
    /// pool pass. Asynchronous and capturable; builds its sequence table on the host.
    pub(crate) fn enqueue_key_append_seqs(
        &self,
        stream: &CudaStream,
        (kr, pos): (&DeviceBuffer<f32>, &DeviceBuffer<u32>),
        (ctx, fault): (usize, FaultSink),
        seqs: &[QsaSeq<'_>],
    ) -> Result<(), GpuError> {
        let what = "qsa::enqueue_key_append_seqs";
        let table = qsa_seqs(what, seqs, ctx)?;
        let m = table.rows();
        check_key_append(what, (kr, pos), m, ctx)?;
        let prep = self
            .module
            .prepare_qsa_key_append_seqs(append_grid(what, m)?)?;
        // SAFETY: each sequence's word 0 is its own ring of the length
        // `qsa_seqs` checked, borrowed by `seqs` while this enqueue runs; a
        // graph that captures the launch replays it over the same rings,
        // which the caller keeps alive and in place while the graph lives,
        // as for any buffer it captured.
        unsafe {
            self.module.qsa_key_append_seqs(
                stream,
                &prep,
                kr,
                pos,
                launch_u32(what, "rows", m)?,
                launch_u32(what, "ctx", ctx)?,
                fault,
                table,
            )?;
        }
        Ok(())
    }

    /// Enqueue the selection of `m` rows: the score pass (`⌈pools/32⌉` blocks
    /// of 128 for each group of [`MAX_ROWS`] rows) and the top-k pass (`m`
    /// blocks of 512), each row's list and length left in `a.scratch`. `m` is
    /// 1 to the scratch's rows and the cache and `kept` the scratch's; any
    /// other shape is refused by name. Two launches, in order, on one stream: the top-k
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
        let tiles = pools.div_ceil(POOL_TILE);
        let grid =
            a.m.div_ceil(MAX_ROWS)
                .checked_mul(tiles)
                .ok_or_else(|| GpuError::shape(what, format!("{} rows of {tiles} tiles", a.m)))?;
        let grid = launch_u32(what, "score grid", grid)?;
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

    /// Enqueue [`QsaKernels::enqueue_select`]'s selection of every sequence
    /// of `seqs`: the score pass of each sequence's rows over its own pooled
    /// plane in one launch (`crate::seq_rows`), then the top-k pass over every
    /// row, each row's list and length left in its row of `a.scratch`, bit
    /// for bit the launches of its sequence alone. Refused by name: what
    /// `enqueue_select` refuses over the sequences' rows, and the sequences
    /// [`qsa_seqs`] refuses. Two launches. Asynchronous and capturable;
    /// builds its sequence table on the host.
    pub(crate) fn enqueue_select_seqs(
        &self,
        stream: &CudaStream,
        a: SelectSeqsArgs<'_>,
        seqs: &[QsaSeq<'_>],
    ) -> Result<(), GpuError> {
        let what = "qsa::enqueue_select_seqs";
        let table = qsa_seqs(what, seqs, a.ctx)?;
        let (s, m) = (a.scratch, table.rows());
        let pools = pools_for(a.ctx);
        if m > s.rows || a.kept == 0 || s.pools != pools || s.width != list_width(a.kept) {
            return Err(GpuError::shape(
                what,
                format!(
                    "need 1..={} rows and the scratch's {} pools of lists {} wide; got m={m} \
                     kept={} pools={pools}",
                    s.rows, s.pools, s.width, a.kept
                ),
            ));
        }
        let lens = [
            ("q", a.q.len(), m * HEADS * DIM),
            ("gain", a.gain.len(), DIM),
            ("n_keys", a.n_keys.len(), m),
        ];
        crate::short(what, &lens)?;
        let ctx = launch_u32(what, "ctx", a.ctx)?;
        let pools_u = launch_u32(what, "pools", pools)?;
        let kept = launch_u32(what, "kept", a.kept)?;
        let rows = launch_u32(what, "rows", m)?;
        let groups = table.most().div_ceil(MAX_ROWS);
        let grid = groups * pools.div_ceil(POOL_TILE) * table.len();
        let prep = self.module.prepare_qsa_score_seqs(LaunchConfig1D::new(
            launch_u32(what, "score grid", grid)?,
            THREADS,
            0,
        ))?;
        // SAFETY: each sequence's words 1 and 2 are its own pooled plane and
        // rope table of the lengths `qsa_seqs` checked, borrowed by `seqs`
        // while this enqueue runs; a graph that captures the launch replays it over the same
        // planes, which the caller keeps alive and in place while the graph
        // lives, as for any buffer it captured.
        unsafe {
            self.module.qsa_score_seqs(
                stream,
                &prep,
                a.q,
                a.gain,
                a.n_keys,
                a.eps,
                ctx,
                pools_u,
                kept,
                rows,
                launch_u32(what, "groups", groups)?,
                a.fault,
                &mut s.q_out,
                &mut s.scores,
                table,
            )?;
        }
        let prep = self
            .module
            .prepare_qsa_topk(LaunchConfig1D::new(rows, TOPK_THREADS, 0))?;
        self.module.qsa_topk(
            stream,
            &prep,
            a.n_keys,
            &s.scores,
            ctx,
            pools_u,
            kept,
            launch_u32(what, "width", s.width)?,
            rows,
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
