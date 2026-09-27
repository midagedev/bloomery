//! Two-level candidate selection over blocks of rows: a source row of scores
//! ranks blocks of `block` consecutive rows by their best score and keeps
//! `blocks` of them; a consumer row of scores over the same rows is then
//! ranked only inside the kept blocks. The rule (`select_candidate_blocks`
//! of the reference this family serves) per row of `n` visible rows:
//!
//! - the rows form `nb = ⌈n / block⌉` blocks, the last one short when `block`
//!   does not divide `n`; a block's key is the largest score of its visible
//!   rows (rows at or past `n` are never read);
//! - the row selects ([`selects`]) when `nb > blocks`; a row that does not
//!   keeps every block, and every entry leaves it as it found it;
//! - a selecting row keeps its last block (it holds the newest row, and the
//!   reference pins it) and the `blocks − 1` best of the other `nb − 1` by
//!   their keys' order ([`order_key`]: `−0` and `+0` tie), ties going to the
//!   lower block — the order the row top-k this feeds breaks its ties in.
//!
//! The kept list is the kept block ids in ascending order, `blocks` of them,
//! the pinned `nb − 1` last. Its slots are the candidate rows in row order:
//! slot `s` is row `kept[s / block]·block + s % block`, and a selecting row has
//! `n_c = (blocks − 1)·block + (n − (nb − 1)·block)` of them. The map is
//! strictly increasing, so a top-k over the slots that breaks ties to the
//! lower slot breaks them to the lower row.
//!
//! Four launches, each a no-op on a row that does not select:
//!
//! - [`cand_kernels::cand_block_max`] (source) — every block's key into the
//!   block-maximum buffer;
//! - [`cand_kernels::cand_select`] (source, one block per row) — the exact
//!   `(blocks − 1)`-th key by three histogram passes over the keys (10, 11,
//!   11 bits of the order key), then the kept list;
//! - [`cand_kernels::cand_compact`] (consumer, one block per row, between the
//!   consumer's score pass and its row top-k) — in place, slot `s` of the
//!   consumer's scores takes row `slot → row`'s score, the row's top-k
//!   histogram is rebuilt over the `n_c` slots, and the top-k's count view
//!   (`n_c` per row, then `top_k`) is written;
//! - [`cand_kernels::cand_remap`] (consumer, after its row top-k) — each list
//!   entry, a slot, becomes its row.
//!
//! The histogram is the row top-k's: [`HIST_BINS`] bins of an order key's top
//! ten bits per row, zero between launches, added into by the score pass only
//! when the top-k will read it (`k > 0` and `n > k`, `k = min(top_k,
//! stride)`) and zeroed by the top-k as it reads it. [`cand_kernels::cand_compact`] keeps that
//! protocol over the slots: it leaves the row holding the counts of the `n_c`
//! moved scores when `k > 0` and `n_c > k`, and zero otherwise.
//!
//! Counts are device words, read per launch (`n` at `ints[n_at + t]`, `top_k`
//! at `ints[top_k_at]`), so a captured step replays at any depth; `blocks` and
//! `block` are load-time constants. Grids depend on the row count and the
//! buffer shapes alone.
//!
//! No silent failure, one site ([`FaultSite::CandMask`]): a visible count past
//! the score rows; a source score that is not finite (a NaN, or an infinity —
//! so the reference's `+inf` pin can never tie a score, and its dropping of
//! `−inf` picks is the rule's never reading rows at or past `n`); a kept list
//! that is not strictly ascending or whose last entry is not the pin; a
//! consumer score that is not finite; a top-k list entry at or past `n_c`.
//! A refused row's compaction writes `n_c = 0`, a refused list entry
//! `u32::MAX`; a value is written as computed, never replaced by a plausible
//! one.

use crate::fault::{FaultSink, FaultSite};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, BlockAtomicU32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Bins of the row top-k's first histogram: the top ten bits of an order key.
pub const HIST_BINS: usize = 1024;
/// The largest block size the kernels take.
pub const MAX_BLOCK: usize = 32;

/// Threads of a block-maximum block: one block key per thread.
const MAX_THREADS: u32 = 256;
/// Threads of a select or compact block, and its warps.
const SEL_THREADS: u32 = 512;
const SEL_WARPS: usize = SEL_THREADS as usize / 32;
/// Threads of a remap block.
const MAP_THREADS: u32 = 256;
/// Keys one select thread reads per batch: sixteen loads in flight.
const LANE_ROWS: usize = 16;
/// Keys the block reads per batch of a refining pass, `tid + 512·m`.
const ROUND_ROWS: usize = SEL_THREADS as usize * LANE_ROWS;
/// Bins of the two refining passes: eleven bits each.
const FINE_BINS: usize = 2048;
const COARSE_PER: usize = HIST_BINS / SEL_THREADS as usize;
const FINE_PER: usize = FINE_BINS / SEL_THREADS as usize;
const _: () = assert!(COARSE_PER * SEL_THREADS as usize == HIST_BINS);
const _: () = assert!(FINE_PER * SEL_THREADS as usize == FINE_BINS);
/// Slots one compact thread moves per chunk.
const MOVE_PER: usize = 8;
/// Slots a compact chunk moves: read all, barrier, write all.
const MOVE_CHUNK: usize = SEL_THREADS as usize * MOVE_PER;

// ------------------------------------------------------------------ cores

/// Whether a row of `n` visible rows selects: more than `blocks` blocks of
/// `block` rows, i.e. `n > blocks·block`. Every entry decides through this.
#[inline(always)]
#[must_use]
pub fn selects(n: u32, blocks: u32, block: u32) -> bool {
    u64::from(n) > u64::from(blocks) * u64::from(block)
}

/// The candidate rows of a selecting row of `n` visible rows: the `blocks − 1`
/// full blocks it keeps besides the pin, and the pin's visible rows.
#[inline(always)]
#[must_use]
pub fn candidate_rows(n: u32, blocks: u32, block: u32) -> u32 {
    let nb = n.div_ceil(block);
    (blocks - 1) * block + (n - (nb - 1) * block)
}

/// The order-preserving key of a score: an unsigned integer that orders as
/// the score does, `-0.0` first made `+0.0` so the two tie.
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

/// A block-wide exclusive prefix sum of `v` over thread order in a
/// [`SEL_THREADS`] block, and the block's total. Every thread calls it (two
/// barriers).
///
/// # Safety
/// `wsum` points at [`SEL_WARPS`] words of this block's shared memory that
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
        // SAFETY: wid < SEL_WARPS words of `wsum`; lane 31 of warp wid is the
        // slot's only writer.
        unsafe { *wsum.add(wid) = x };
    }
    thread::sync_threads();
    let (mut before, mut total) = (0u32, 0u32);
    for w in 0..SEL_WARPS {
        thread::__unroll_config::<0>();
        // SAFETY: w < SEL_WARPS; the barrier above published every slot.
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
/// `PER` bins `top − PER·tid − j`, counts `c[j]`; they total at least `need
/// >= 1`. Every thread calls it and gets the same pair.
///
/// # Safety
/// As [`block_scan`]; `pick` points at two words of this block's shared
/// memory that nothing else uses across the call.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn pick_bin<const PER: usize>(
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
            // SAFETY: two words of `pick`; exactly one (thread, bin) holds the
            // need-th key, so one thread writes.
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

/// The order keys of block keys `base + tid + 512·m`, `m < LANE_ROWS`, of one
/// row at `keys[kbase ..]`; a block at or past `end` gives 0.
///
/// # Safety
/// `kbase + end <= keys.len()`.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn round_keys(
    keys: &[f32],
    kbase: usize,
    base: usize,
    end: usize,
    tid: usize,
) -> [u32; LANE_ROWS] {
    let mut kk = [0u32; LANE_ROWS];
    for m in 0..LANE_ROWS {
        thread::__unroll_config::<0>();
        let i = base + tid + SEL_THREADS as usize * m;
        if i < end {
            // SAFETY: i < end, so kbase + i < keys.len() (this fn's contract).
            kk[m] = order_key(unsafe { *keys.get_unchecked(kbase + i) });
        }
    }
    kk
}

/// The order keys of block keys `base + 32·m + lane`, `m < LANE_ROWS`, of one
/// row; a block at or past `end` gives 0.
///
/// # Safety
/// `kbase + end <= keys.len()`.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn lane_keys(
    keys: &[f32],
    kbase: usize,
    base: usize,
    end: usize,
    lane: usize,
) -> [u32; LANE_ROWS] {
    let mut kk = [0u32; LANE_ROWS];
    for m in 0..LANE_ROWS {
        thread::__unroll_config::<0>();
        let i = base + 32 * m + lane;
        if i < end {
            // SAFETY: i < end, so kbase + i < keys.len() (this fn's contract).
            kk[m] = order_key(unsafe { *keys.get_unchecked(kbase + i) });
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
/// the row's first `n` block keys; then this thread's [`COARSE_PER`] bins,
/// from the top bin down.
///
/// # Safety
/// As [`refine`].
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn coarse(
    keys: &[f32],
    kbase: usize,
    n: usize,
    fine: *mut u32,
    tid: usize,
) -> [u32; COARSE_PER] {
    // SAFETY: forwarded from this fn's contract.
    unsafe { fine_zero(fine, tid) };
    let mut base = 0usize;
    while base < n {
        // SAFETY: kbase + n <= keys.len().
        let kk = unsafe { round_keys(keys, kbase, base, n, tid) };
        for m in 0..LANE_ROWS {
            thread::__unroll_config::<0>();
            if base + tid + SEL_THREADS as usize * m < n {
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
/// 0x7ff`, the keys among the row's first `n` whose bits above `shift + 11`
/// equal `prefix`; then this thread's bins.
///
/// # Safety
/// `fine` points at [`FINE_BINS`] words of this block's shared memory used by
/// nothing else across the call; `kbase + n <= keys.len()`; every thread
/// calls it.
#[inline(always)]
#[allow(
    clippy::needless_range_loop,
    reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
)]
unsafe fn refine(
    keys: &[f32],
    kbase: usize,
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
        // SAFETY: kbase + n <= keys.len().
        let kk = unsafe { round_keys(keys, kbase, base, n, tid) };
        for m in 0..LANE_ROWS {
            thread::__unroll_config::<0>();
            let i = base + tid + SEL_THREADS as usize * m;
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
mod cand_kernels {
    use super::*;

    /// The block keys: block `b` serves row `t = b / grid_per_row` and its
    /// blocks `j = (b % grid_per_row)·256 + tid`. A selecting row's block `j <
    /// nb` takes the largest of its visible rows' scores (`scores[t·rows ..]`),
    /// folded in row order with `>` from its first row, into `bmax[t·nb_cap +
    /// j]`; a score that is not finite raises the site. Any other row writes
    /// nothing; a count past `rows` raises the site.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            ints.len() >= n_at + tokens,
            scores.len() >= tokens * rows,
            nb_cap * block >= rows,
            grid_per_row * 256 >= nb_cap,
            bmax.len() >= tokens * nb_cap
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn cand_block_max(
        ints: &[u32],
        scores: &[f32],
        tokens: u32,
        grid_per_row: u32,
        n_at: u32,
        rows: u32,
        nb_cap: u32,
        blocks: u32,
        block: u32,
        fault: FaultSink,
        mut bmax: DisjointSlice<f32>,
    ) {
        let _ = tokens;
        let bid = thread::blockIdx_x();
        let t = (bid / grid_per_row) as usize;
        let gb = (bid % grid_per_row) as usize;
        let tid = thread::threadIdx_x() as usize;
        // SAFETY: t < tokens (the grid is grid_per_row · tokens), so n_at + t <
        // ints.len() by the launch contract.
        let n = unsafe { *ints.get_unchecked(n_at as usize + t) };
        if n > rows {
            if gb == 0 && tid == 0 {
                fault.raise(FaultSite::CandMask);
            }
            return; // block-uniform
        }
        if !selects(n, blocks, block) {
            return; // block-uniform
        }
        let nb = n.div_ceil(block) as usize;
        let j = gb * MAX_THREADS as usize + tid;
        if j >= nb {
            return; // no barrier follows
        }
        let n = n as usize;
        let b = block as usize;
        let sbase = t * rows as usize;
        let r0 = j * b;
        let r1 = (r0 + b).min(n);
        // SAFETY: r0 < n <= rows, so sbase + r0 < tokens·rows <= scores.len()
        // (launch contract).
        let mut m = unsafe { *scores.get_unchecked(sbase + r0) };
        let mut finite = m.is_finite();
        let mut r = r0 + 1;
        while r < r1 {
            // SAFETY: r < r1 <= n <= rows bounds the load as above.
            let v = unsafe { *scores.get_unchecked(sbase + r) };
            finite &= v.is_finite();
            if v > m {
                m = v;
            }
            r += 1;
        }
        if !finite {
            fault.raise(FaultSite::CandMask);
        }
        // SAFETY: j < nb = ⌈n / block⌉ <= ⌈rows / block⌉ <= nb_cap (launch
        // contract), so t·nb_cap + j < tokens·nb_cap <= bmax.len(); one thread
        // writes each block's key.
        unsafe { *bmax.get_unchecked_mut(t * nb_cap as usize + j) = m };
    }

    /// The selection: block `t` writes row `t`'s kept list into `kept[t·kstride
    /// ..]` — the `blocks − 1` best of blocks `0 .. nb − 1` by their keys in
    /// `bmax[t·nb_cap ..]` (ties to the lower block) in ascending order, then
    /// the pin `nb − 1` at entry `blocks − 1` (module doc). A row that does
    /// not select writes nothing; a count past `rows` raises the site.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            ints.len() >= n_at + tokens,
            nb_cap * block >= rows,
            bmax.len() >= tokens * nb_cap,
            kstride >= blocks,
            kept.len() >= tokens * kstride
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
    pub fn cand_select(
        ints: &[u32],
        bmax: &[f32],
        tokens: u32,
        n_at: u32,
        rows: u32,
        nb_cap: u32,
        blocks: u32,
        block: u32,
        kstride: u32,
        fault: FaultSink,
        mut kept: DisjointSlice<u32>,
    ) {
        static mut FINE: SharedArray<u32, FINE_BINS> = SharedArray::UNINIT;
        static mut WSUM: SharedArray<u32, SEL_WARPS> = SharedArray::UNINIT;
        static mut WTOT: SharedArray<u32, { 2 * SEL_WARPS }> = SharedArray::UNINIT;
        static mut PICK: SharedArray<u32, 2> = SharedArray::UNINIT;

        let _ = tokens;
        let t = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id();
        let wid = tid / 32;
        // SAFETY: t < tokens (one block per row), so n_at + t < ints.len() by
        // the launch contract.
        let n = unsafe { *ints.get_unchecked(n_at as usize + t) };
        if n > rows {
            if tid == 0 {
                fault.raise(FaultSite::CandMask);
            }
            return; // block-uniform
        }
        if !selects(n, blocks, block) {
            return; // block-uniform
        }
        let nb = n.div_ceil(block);
        let lbase = t * kstride as usize;
        if tid == 0 {
            // SAFETY: blocks − 1 < blocks <= kstride, so the entry is inside
            // row t of `kept`, below tokens·kstride <= kept.len() (launch
            // contract).
            unsafe { *kept.get_unchecked_mut(lbase + blocks as usize - 1) = nb - 1 };
        }
        let need = blocks - 1;
        if need == 0 {
            return; // block-uniform: the pin is the whole list
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
        // The ranked blocks: every one but the pin. They number nb − 1 >
        // blocks − 1 = need, so every pick finds its bin.
        let n_r = nb as usize - 1;
        let kbase = t * nb_cap as usize;

        // ---- the top ten bits, then eleven, then the last eleven.
        // SAFETY: n_r < nb <= nb_cap, so kbase + n_r <= tokens·nb_cap <=
        // bmax.len(); FINE is this pass's alone.
        let hc = unsafe { coarse(bmax, kbase, n_r, fine, tid) };
        // SAFETY: WSUM and PICK are used by nothing else across the call.
        let (b1, above1) = unsafe {
            pick_bin::<COARSE_PER>(hc, HIST_BINS as u32 - 1, need, lane, tid, wsum, pick)
        };
        let need2 = need - above1;
        // SAFETY: as the pass above.
        let fc = unsafe { refine(bmax, kbase, n_r, b1, 11, fine, tid) };
        // SAFETY: WSUM and PICK are used by nothing else across the call.
        let (b2, above2) =
            unsafe { pick_bin::<FINE_PER>(fc, FINE_BINS as u32 - 1, need2, lane, tid, wsum, pick) };
        let need3 = need2 - above2;
        let prefix = (b1 << 11) | b2;
        // SAFETY: as the pass above.
        let fc = unsafe { refine(bmax, kbase, n_r, prefix, 0, fine, tid) };
        // SAFETY: WSUM and PICK are used by nothing else across the call.
        let (b3, above3) =
            unsafe { pick_bin::<FINE_PER>(fc, FINE_BINS as u32 - 1, need3, lane, tid, wsum, pick) };
        let thr = (prefix << 11) | b3;
        let take_eq = need - (above1 + above2 + above3);

        // ---- the list, in block order: warp `wid` owns blocks `c0 .. c1`,
        // counts the blocks it keeps, the block scans the counts, and the warp
        // walks its blocks again, writing each kept block at its place — every
        // block above `thr`, and the first `take_eq` equal to it.
        let chunk = n_r.div_ceil(32 * SEL_WARPS) * 32;
        let c0 = (wid * chunk).min(n_r);
        let c1 = (c0 + chunk).min(n_r);
        let l = lane as usize;
        let (mut tg, mut te) = (0u32, 0u32);
        let mut base = c0;
        while base < c1 {
            // SAFETY: c1 <= n_r, so kbase + c1 <= bmax.len().
            let kk = unsafe { lane_keys(bmax, kbase, base, c1, l) };
            for r in 0..LANE_ROWS {
                thread::__unroll_config::<0>();
                let live = base + 32 * r + l < c1;
                tg += warp::ballot(live && kk[r] > thr).count_ones();
                te += warp::ballot(live && kk[r] == thr).count_ones();
            }
            base += 32 * LANE_ROWS;
        }
        if lane == 0 {
            // SAFETY: wid < SEL_WARPS bounds both slots; lane 0 of warp wid is
            // their only writer, before the barrier.
            unsafe {
                *wtot.add(wid) = tg;
                *wtot.add(SEL_WARPS + wid) = te;
            }
        }
        thread::sync_threads();
        let (mut og, mut oe) = (0u32, 0u32);
        for w in 0..SEL_WARPS {
            thread::__unroll_config::<0>();
            if w < wid {
                // SAFETY: w < SEL_WARPS; published by the barrier above.
                unsafe {
                    og += *wtot.add(w);
                    oe += *wtot.add(SEL_WARPS + w);
                }
            }
        }
        let lt = warp::lanemask_lt();
        let mut base = c0;
        while base < c1 {
            // SAFETY: c1 <= n_r, so kbase + c1 <= bmax.len().
            let kk = unsafe { lane_keys(bmax, kbase, base, c1, l) };
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
                    let pos = (gt_before + eq_before.min(take_eq)) as usize;
                    // SAFETY: the blocks taken number need = blocks − 1 and pos
                    // counts those before this one, so pos < blocks − 1 <
                    // kstride and lbase + pos < tokens·kstride <= kept.len();
                    // each place has one block.
                    unsafe { *kept.get_unchecked_mut(lbase + pos) = i as u32 };
                }
                og += mg.count_ones();
                oe += me.count_ones();
            }
            base += 32 * LANE_ROWS;
        }
    }

    /// The compaction, in place: block `t` serves row `t` (module doc). A
    /// selecting row checks its kept list, then — when its top-k will read
    /// the scores (`k > 0` and `n_c > k`) — moves slot `s`'s row score into
    /// `scores[t·rows + s]` for every `s < n_c` in chunks of [`MOVE_CHUNK`]
    /// (each chunk read whole before it is written; a slot's row is never
    /// below it, so no chunk reads what an earlier one wrote) and sets the
    /// histogram row `hist[t·HIST_BINS ..]` to the moved scores' counts;
    /// otherwise it zeroes that row. `counts[counts_at + t]` takes `n_c` (a
    /// row that does not select: `n`; a refused row: 0), and block 0 writes
    /// `top_k` to `counts[counts_at + tokens]`, so the counts are the view
    /// the row top-k and [`cand_remap`] read.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            ints.len() >= n_at + tokens,
            ints.len() > top_k_at,
            kstride >= blocks,
            kept.len() >= tokens * kstride,
            scores.len() >= tokens * rows,
            hist.len() >= tokens * 1024,
            counts.len() > counts_at + tokens
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
    pub fn cand_compact(
        ints: &[u32],
        kept: &[u32],
        tokens: u32,
        n_at: u32,
        top_k_at: u32,
        rows: u32,
        blocks: u32,
        block: u32,
        kstride: u32,
        stride: u32,
        counts_at: u32,
        fault: FaultSink,
        mut scores: DisjointSlice<f32>,
        mut hist: DisjointSlice<u32>,
        mut counts: DisjointSlice<u32>,
    ) {
        static mut HS: SharedArray<u32, HIST_BINS> = SharedArray::UNINIT;
        static mut BAD: SharedArray<u32, 1> = SharedArray::UNINIT;

        let t = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        // SAFETY: top_k_at < ints.len() by the launch contract.
        let top_k = unsafe { *ints.get_unchecked(top_k_at as usize) };
        let cbase = counts_at as usize;
        if t == 0 && tid == 0 {
            // SAFETY: counts_at + tokens < counts.len() (launch contract).
            unsafe { *counts.get_unchecked_mut(cbase + tokens as usize) = top_k };
        }
        // SAFETY: t < tokens (one block per row), so n_at + t < ints.len() by
        // the launch contract.
        let n = unsafe { *ints.get_unchecked(n_at as usize + t) };
        if n > rows {
            if tid == 0 {
                fault.raise(FaultSite::CandMask);
                // SAFETY: t < tokens, so counts_at + t < counts.len().
                unsafe { *counts.get_unchecked_mut(cbase + t) = 0 };
            }
            return; // block-uniform
        }
        if !selects(n, blocks, block) {
            if tid == 0 {
                // SAFETY: t < tokens, so counts_at + t < counts.len().
                unsafe { *counts.get_unchecked_mut(cbase + t) = n };
            }
            return; // block-uniform
        }
        // SAFETY: block-shared statics; the raw forms reach them without a
        // reference, and every access below is bounded and barrier-ordered.
        let (hs, bad) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut HS),
                SharedArray::as_raw_mut_ptr(&raw mut BAD),
            )
        };
        let nb = n.div_ceil(block);
        let lbase = t * kstride as usize;
        let k_blocks = blocks as usize;

        // ---- the kept list: strictly ascending, the pin last.
        if tid == 0 {
            // SAFETY: one word of BAD; thread 0 alone writes it before the
            // barrier.
            unsafe { *bad = 0 };
        }
        let mut wrong = false;
        let mut j = tid;
        while j < k_blocks {
            // SAFETY: j < blocks <= kstride, so lbase + j < tokens·kstride <=
            // kept.len() (launch contract); so is j + 1 below blocks.
            let v = unsafe { *kept.get_unchecked(lbase + j) };
            if j + 1 < k_blocks {
                // SAFETY: as above, j + 1 < blocks.
                wrong |= v >= unsafe { *kept.get_unchecked(lbase + j + 1) };
            } else {
                wrong |= v != nb - 1;
            }
            j += SEL_THREADS as usize;
        }
        thread::sync_threads();
        if warp::any(wrong) && warp::lane_id() == 0 {
            // SAFETY: one word of BAD, every access to it between the barriers
            // atomic.
            unsafe { BlockAtomicU32::from_ptr(bad).fetch_or(1, AtomicOrdering::Relaxed) };
        }
        thread::sync_threads();
        // SAFETY: published by the barrier above.
        if unsafe { *bad } != 0 {
            if tid == 0 {
                fault.raise(FaultSite::CandMask);
                // SAFETY: t < tokens, so counts_at + t < counts.len().
                unsafe { *counts.get_unchecked_mut(cbase + t) = 0 };
            }
            return; // block-uniform
        }

        let n_c = candidate_rows(n, blocks, block) as usize;
        let k = top_k.min(stride) as usize;
        let hbase = t * HIST_BINS;
        // The row top-k reads the histogram and the scores exactly when it
        // selects: `k > 0` and more rows than `k`.
        let read = k > 0 && n_c > k;
        if read {
            let mut b = tid;
            while b < HIST_BINS {
                // SAFETY: b < HIST_BINS; thread tid alone writes bins tid +
                // 512·i before the barrier below.
                unsafe { *hs.add(b) = 0 };
                b += SEL_THREADS as usize;
            }
            thread::sync_threads();
            let lb = block.trailing_zeros();
            let low = block as usize - 1;
            let sbase = t * rows as usize;
            let sp = scores.as_mut_ptr();
            let mut finite = true;
            let mut s0 = 0usize;
            while s0 < n_c {
                let mut v = [0.0f32; MOVE_PER];
                for m in 0..MOVE_PER {
                    thread::__unroll_config::<0>();
                    let s = s0 + tid + SEL_THREADS as usize * m;
                    if s < n_c {
                        // SAFETY: s < n_c <= blocks·block, so s >> lb < blocks
                        // <= kstride and the entry is inside row t of `kept`.
                        let kb = unsafe { *kept.get_unchecked(lbase + (s >> lb)) } as usize;
                        let r = (kb << lb) | (s & low);
                        // SAFETY: the list is checked (strictly ascending, the
                        // pin nb − 1 last), so a kept block below the pin is
                        // below nb − 1 and its rows below (nb − 1)·block < n,
                        // and the pin's slots map below n by n_c's definition:
                        // r < n <= rows, inside row t of `scores`
                        // (tokens·rows <= scores.len()).
                        v[m] = unsafe { *sp.add(sbase + r) };
                    }
                }
                // Every read of this chunk lands before any write of it.
                thread::sync_threads();
                for m in 0..MOVE_PER {
                    thread::__unroll_config::<0>();
                    let s = s0 + tid + SEL_THREADS as usize * m;
                    if s < n_c {
                        finite &= v[m].is_finite();
                        // SAFETY: s < n_c <= n <= rows, inside row t of
                        // `scores`; one thread writes each slot, after the
                        // chunk's barrier; a later chunk reads only rows at
                        // or past its own first slot, above this one.
                        unsafe { *sp.add(sbase + s) = v[m] };
                        // SAFETY: the bin is below HIST_BINS; every access
                        // to HS between the barriers is atomic.
                        unsafe {
                            BlockAtomicU32::from_ptr(hs.add((order_key(v[m]) >> 22) as usize))
                                .fetch_add(1, AtomicOrdering::Relaxed)
                        };
                    }
                }
                s0 += MOVE_CHUNK;
            }
            if !finite {
                fault.raise(FaultSite::CandMask);
            }
            thread::sync_threads();
        }
        let mut b = tid;
        while b < HIST_BINS {
            // SAFETY: b < HIST_BINS was published by the barrier above when
            // `read`; hbase + b < tokens·HIST_BINS <= hist.len() (launch
            // contract); this block alone writes row t.
            unsafe {
                *hist.get_unchecked_mut(hbase + b) = if read { *hs.add(b) } else { 0 };
            }
            b += SEL_THREADS as usize;
        }
        if tid == 0 {
            // SAFETY: t < tokens, so counts_at + t < counts.len().
            unsafe { *counts.get_unchecked_mut(cbase + t) = n_c as u32 };
        }
    }

    /// The remap: block `t` turns the first `min(top_k, stride, n_c)` entries
    /// of row `t`'s list (`list[t·stride ..]`), slots, into their rows through
    /// the kept list, `n_c` and `top_k` read from the counts
    /// [`cand_compact`] wrote. A row that does not select, or whose count
    /// is past `rows`, is left as it is; an entry at or past `n_c` raises the
    /// site and becomes `u32::MAX`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            ints.len() >= n_at + tokens,
            kstride >= blocks,
            kept.len() >= tokens * kstride,
            counts.len() > counts_at + tokens,
            list.len() >= tokens * stride
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    pub fn cand_remap(
        ints: &[u32],
        kept: &[u32],
        counts: &[u32],
        tokens: u32,
        n_at: u32,
        rows: u32,
        blocks: u32,
        block: u32,
        kstride: u32,
        counts_at: u32,
        stride: u32,
        fault: FaultSink,
        mut list: DisjointSlice<u32>,
    ) {
        let t = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        // SAFETY: t < tokens (one block per row), so n_at + t < ints.len() by
        // the launch contract.
        let n = unsafe { *ints.get_unchecked(n_at as usize + t) };
        if n > rows || !selects(n, blocks, block) {
            return; // block-uniform
        }
        let cbase = counts_at as usize;
        // SAFETY: counts_at + tokens < counts.len() (launch contract), and t <
        // tokens.
        let (n_c, top_k) = unsafe {
            (
                *counts.get_unchecked(cbase + t),
                *counts.get_unchecked(cbase + tokens as usize),
            )
        };
        let len = top_k.min(stride).min(n_c) as usize;
        let lb = block.trailing_zeros();
        let low = block - 1;
        let lbase = t * stride as usize;
        let kbase = t * kstride as usize;
        let mut i = tid;
        while i < len {
            // SAFETY: i < len <= stride, so lbase + i < tokens·stride <=
            // list.len() (launch contract); thread tid alone touches entries
            // tid + 256·m.
            let s = unsafe { *list.get_unchecked_mut(lbase + i) };
            let row = if s < n_c {
                // SAFETY: s < n_c <= blocks·block, so s >> lb < blocks <=
                // kstride: inside row t of `kept` (launch contract).
                let kb = unsafe { *kept.get_unchecked(kbase + (s >> lb) as usize) };
                (kb << lb) | (s & low)
            } else {
                fault.raise(FaultSite::CandMask);
                u32::MAX
            };
            // SAFETY: as the read above.
            unsafe { *list.get_unchecked_mut(lbase + i) = row };
            i += MAP_THREADS as usize;
        }
    }
}

// -------------------------------------------------------------- launchers

/// The shape of a candidate selection: `blocks` kept blocks of `block` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandShape {
    blocks: usize,
    block: usize,
}

impl CandShape {
    /// `blocks >= 1` kept blocks of `block` rows, a power of two up to
    /// [`MAX_BLOCK`]; any other shape is refused by name.
    pub fn new(blocks: usize, block: usize) -> Result<CandShape, GpuError> {
        if blocks == 0 || !block.is_power_of_two() || block > MAX_BLOCK {
            return Err(GpuError::shape(
                "cand::CandShape",
                format!(
                    "need blocks >= 1 and a power-of-two block up to {MAX_BLOCK}, got blocks={blocks} \
                     block={block}"
                ),
            ));
        }
        Ok(CandShape { blocks, block })
    }

    /// Blocks kept.
    #[must_use]
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    /// Rows per block.
    #[must_use]
    pub fn block(&self) -> usize {
        self.block
    }

    /// Block keys a row of `rows` score rows needs: `⌈rows / block⌉`.
    #[must_use]
    pub fn block_rows(&self, rows: usize) -> usize {
        rows.div_ceil(self.block)
    }
}

/// The source side's own scratch, allocated once for `tokens` rows over
/// `rows` score rows: each row's block keys.
pub struct CandScratch {
    /// `[tokens × ⌈rows / block⌉]`: a selecting row's block keys.
    pub bmax: DeviceBuffer<f32>,
    tokens: usize,
    rows: usize,
    nb_cap: usize,
}

impl CandScratch {
    /// Allocate for `tokens` rows over `rows` score rows under `shape`.
    /// Load-time only.
    pub fn new(
        stream: &CudaStream,
        shape: CandShape,
        tokens: usize,
        rows: usize,
    ) -> Result<CandScratch, GpuError> {
        if tokens == 0 || rows == 0 {
            return Err(GpuError::shape(
                "cand::CandScratch",
                format!("need tokens and rows >= 1, got tokens={tokens} rows={rows}"),
            ));
        }
        let nb_cap = shape.block_rows(rows);
        Ok(CandScratch {
            bmax: DeviceBuffer::zeroed(stream, tokens * nb_cap)?,
            tokens,
            rows,
            nb_cap,
        })
    }

    /// Bytes of device memory the scratch holds.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        4 * self.bmax.len()
    }
}

/// [`CandKernels::enqueue_select`]'s arguments: the source's visible counts
/// (`ints[n_at + t]`), its scores (`[tokens × rows]`), the `k = min(top_k,
/// stride)` of the source's own row top-k (`score_k`: its score pass writes
/// a row's scores only when `k > 0` and the count is above `k`), and the kept
/// lists (`[tokens × kstride]`, `kstride >= blocks`).
pub struct SelectArgs<'a> {
    pub ints: &'a DeviceBuffer<u32>,
    pub n_at: usize,
    pub scores: &'a DeviceBuffer<f32>,
    pub score_k: usize,
    pub rows: usize,
    pub tokens: usize,
    pub shape: CandShape,
    pub fault: FaultSink,
    pub scratch: &'a mut CandScratch,
    pub kept: &'a mut DeviceBuffer<u32>,
    pub kstride: usize,
}

/// [`CandKernels::enqueue_compact`]'s arguments: the source's counts
/// (`ints[n_at + t]`) and the consumer's `top_k` (`ints[top_k_at]`), the
/// source's kept lists, the consumer top-k's list stride, and what the
/// compaction rewrites — the consumer's scores (`[tokens × rows]`) and
/// histogram (`[tokens × HIST_BINS]`) — and writes: the counts view
/// (`counts[counts_at + t]` = `n_c`, `counts[counts_at + tokens]` = `top_k`).
pub struct CompactArgs<'a> {
    pub ints: &'a DeviceBuffer<u32>,
    pub n_at: usize,
    pub top_k_at: usize,
    pub kept: &'a DeviceBuffer<u32>,
    pub kstride: usize,
    pub rows: usize,
    pub tokens: usize,
    pub stride: usize,
    pub shape: CandShape,
    pub fault: FaultSink,
    pub scores: &'a mut DeviceBuffer<f32>,
    pub hist: &'a mut DeviceBuffer<u32>,
    pub counts: &'a mut DeviceBuffer<u32>,
    pub counts_at: usize,
}

/// [`CandKernels::enqueue_remap`]'s arguments: the source's counts, the kept
/// lists, the counts view [`CandKernels::enqueue_compact`] wrote, and the
/// consumer's lists (`[tokens × stride]`), rewritten in place.
pub struct RemapArgs<'a> {
    pub ints: &'a DeviceBuffer<u32>,
    pub n_at: usize,
    pub kept: &'a DeviceBuffer<u32>,
    pub kstride: usize,
    pub counts: &'a DeviceBuffer<u32>,
    pub counts_at: usize,
    pub rows: usize,
    pub tokens: usize,
    pub shape: CandShape,
    pub fault: FaultSink,
    pub list: &'a mut DeviceBuffer<u32>,
    pub stride: usize,
}

/// The loaded module. Owns no stream: each enqueue takes the caller's.
pub struct CandKernels {
    module: cand_kernels::LoadedModule,
}

/// The first of `lens` shorter than it needs, as the error `what` names.
fn short(what: &'static str, lens: &[(&str, usize, usize)]) -> Result<(), GpuError> {
    match lens.iter().find(|(_, have, want)| have < want) {
        Some((name, have, want)) => Err(GpuError::shape(
            what,
            format!("{name} holds {have}, want >= {want}"),
        )),
        None => Ok(()),
    }
}

/// `tokens`, `rows` and `kstride` checked against `shape`.
fn dims(
    what: &'static str,
    shape: CandShape,
    tokens: usize,
    rows: usize,
    kstride: usize,
) -> Result<(), GpuError> {
    if tokens == 0 || rows == 0 || kstride < shape.blocks {
        return Err(GpuError::shape(
            what,
            format!(
                "need tokens and rows >= 1 and a kept stride of at least {} blocks, got \
                 tokens={tokens} rows={rows} kstride={kstride}",
                shape.blocks
            ),
        ));
    }
    Ok(())
}

impl CandKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<CandKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { cand_kernels::load(ctx)? };
        Ok(CandKernels { module })
    }

    /// Enqueue the source side: the block keys (`⌈⌈rows/block⌉ / 256⌉` blocks
    /// per row), then the selection (one block per row), each selecting row's
    /// kept list left in `a.kept`. Two launches, in order, on one stream,
    /// after the source's score pass. A selecting row must have scores: a
    /// `score_k` of 0 or above `blocks·block` is refused by name. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_select(&self, stream: &CudaStream, a: SelectArgs<'_>) -> Result<(), GpuError> {
        let what = "cand::enqueue_select";
        let s = a.scratch;
        dims(what, a.shape, a.tokens, a.rows, a.kstride)?;
        let kept_rows = a.shape.blocks * a.shape.block;
        if a.score_k == 0 || kept_rows < a.score_k {
            return Err(GpuError::shape(
                what,
                format!(
                    "a row selects past {kept_rows} rows, and the source's score pass writes a \
                     row's scores only past its k = {}: need 1 <= k <= {kept_rows}",
                    a.score_k
                ),
            ));
        }
        if s.tokens != a.tokens || s.rows != a.rows || s.nb_cap != a.shape.block_rows(a.rows) {
            return Err(GpuError::shape(
                what,
                format!(
                    "the scratch serves {} rows over {} score rows in {} blocks, the launch {} over \
                     {} in {}",
                    s.tokens,
                    s.rows,
                    s.nb_cap,
                    a.tokens,
                    a.rows,
                    a.shape.block_rows(a.rows)
                ),
            ));
        }
        short(
            what,
            &[
                ("ints", a.ints.len(), a.n_at + a.tokens),
                ("scores", a.scores.len(), a.tokens * a.rows),
                ("kept", a.kept.len(), a.tokens * a.kstride),
            ],
        )?;
        let grid_per_row = s.nb_cap.div_ceil(MAX_THREADS as usize);
        let tokens = launch_u32(what, "tokens", a.tokens)?;
        let n_at = launch_u32(what, "n_at", a.n_at)?;
        let rows = launch_u32(what, "rows", a.rows)?;
        let nb_cap = launch_u32(what, "nb_cap", s.nb_cap)?;
        let blocks = launch_u32(what, "blocks", a.shape.blocks)?;
        let block = launch_u32(what, "block", a.shape.block)?;
        let grid = launch_u32(what, "block-key grid", grid_per_row * a.tokens)?;
        let prep = self
            .module
            .prepare_cand_block_max(LaunchConfig1D::new(grid, MAX_THREADS, 0))?;
        self.module.cand_block_max(
            stream,
            &prep,
            a.ints,
            a.scores,
            tokens,
            launch_u32(what, "grid per row", grid_per_row)?,
            n_at,
            rows,
            nb_cap,
            blocks,
            block,
            a.fault,
            &mut s.bmax,
        )?;
        let prep = self
            .module
            .prepare_cand_select(LaunchConfig1D::new(tokens, SEL_THREADS, 0))?;
        self.module.cand_select(
            stream,
            &prep,
            a.ints,
            &s.bmax,
            tokens,
            n_at,
            rows,
            nb_cap,
            blocks,
            block,
            launch_u32(what, "kstride", a.kstride)?,
            a.fault,
            a.kept,
        )?;
        Ok(())
    }

    /// Enqueue the consumer's compaction (one block per row), between its
    /// score pass and its row top-k: the scores and histogram rewritten over
    /// the candidate slots, the counts view written. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_compact(&self, stream: &CudaStream, a: CompactArgs<'_>) -> Result<(), GpuError> {
        let what = "cand::enqueue_compact";
        dims(what, a.shape, a.tokens, a.rows, a.kstride)?;
        short(
            what,
            &[
                ("ints (n)", a.ints.len(), a.n_at + a.tokens),
                ("ints (top_k)", a.ints.len(), a.top_k_at + 1),
                ("kept", a.kept.len(), a.tokens * a.kstride),
                ("scores", a.scores.len(), a.tokens * a.rows),
                ("hist", a.hist.len(), a.tokens * HIST_BINS),
                ("counts", a.counts.len(), a.counts_at + a.tokens + 1),
            ],
        )?;
        let tokens = launch_u32(what, "tokens", a.tokens)?;
        let prep = self
            .module
            .prepare_cand_compact(LaunchConfig1D::new(tokens, SEL_THREADS, 0))?;
        self.module.cand_compact(
            stream,
            &prep,
            a.ints,
            a.kept,
            tokens,
            launch_u32(what, "n_at", a.n_at)?,
            launch_u32(what, "top_k_at", a.top_k_at)?,
            launch_u32(what, "rows", a.rows)?,
            launch_u32(what, "blocks", a.shape.blocks)?,
            launch_u32(what, "block", a.shape.block)?,
            launch_u32(what, "kstride", a.kstride)?,
            launch_u32(what, "stride", a.stride)?,
            launch_u32(what, "counts_at", a.counts_at)?,
            a.fault,
            a.scores,
            a.hist,
            a.counts,
        )?;
        Ok(())
    }

    /// Enqueue the consumer's remap (one block per row), after its row top-k:
    /// every list entry, a slot, becomes its row. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_remap(&self, stream: &CudaStream, a: RemapArgs<'_>) -> Result<(), GpuError> {
        let what = "cand::enqueue_remap";
        dims(what, a.shape, a.tokens, a.rows, a.kstride)?;
        short(
            what,
            &[
                ("ints", a.ints.len(), a.n_at + a.tokens),
                ("kept", a.kept.len(), a.tokens * a.kstride),
                ("counts", a.counts.len(), a.counts_at + a.tokens + 1),
                ("list", a.list.len(), a.tokens * a.stride),
            ],
        )?;
        let tokens = launch_u32(what, "tokens", a.tokens)?;
        let prep = self
            .module
            .prepare_cand_remap(LaunchConfig1D::new(tokens, MAP_THREADS, 0))?;
        self.module.cand_remap(
            stream,
            &prep,
            a.ints,
            a.kept,
            a.counts,
            tokens,
            launch_u32(what, "n_at", a.n_at)?,
            launch_u32(what, "rows", a.rows)?,
            launch_u32(what, "blocks", a.shape.blocks)?,
            launch_u32(what, "block", a.shape.block)?,
            launch_u32(what, "kstride", a.kstride)?,
            launch_u32(what, "counts_at", a.counts_at)?,
            launch_u32(what, "stride", a.stride)?,
            a.fault,
            a.list,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_past_the_kept_rows_and_counts_the_candidates() {
        let (k, b) = (2048u32, 8u32);
        assert!(!selects(16_384, k, b));
        assert!(selects(16_385, k, b));
        assert_eq!(candidate_rows(16_385, k, b), 2047 * 8 + 1);
        assert_eq!(candidate_rows(16_392, k, b), 2048 * 8);
        assert_eq!(candidate_rows(32_767, k, b), 2047 * 8 + 7);
        assert!(!selects(u32::MAX, 1 << 20, 1 << 12));
        assert_eq!(order_key(-0.0), order_key(0.0));
        assert!(CandShape::new(2048, 6).is_err() && CandShape::new(0, 8).is_err());
        assert_eq!(
            CandShape::new(16, 8).map(|s| s.block_rows(301)).ok(),
            Some(38)
        );
    }
}
