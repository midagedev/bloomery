//! Q4_K gemv over expert slots selected on the device: the MoE down
//! projection of a Q4_K expert stack in the decode shape, all `n_slots`
//! routed experts in one launch. The ids are read from a device buffer, so
//! the launch lives inside a captured graph whose replay consumes whatever
//! the router last wrote there. Down is the per-slot-column shape: slot `s`
//! dots the rows of expert `sel[s]` with activation column `s` (that
//! expert's own swiglu output), as `q5_0_gemv_sel` does for Q5_0 — where
//! `q3k_gemv_sel` (gate/up) shares one column across slots.
//!
//! The kernel owns no arithmetic: its per-row body is `q4k_gemv`'s m = 1
//! path verbatim (`cores::q4k_row_dot_1col`, the warp reduction, the lane-0
//! store), with `col0 = slot`. So slot `s` is bit-identical to `q4k_gemv`
//! run on an upload of expert `sel[s]`'s rows alone against a one-column
//! activation quantized from input column `s`; `gate_q4k_sel` pins that.
//! Q4_K rows are `36 * n_sb` u32 words, whole words for any `n_sb`, so an
//! odd super-block count needs no load-time repack (unlike Q3_K): the core's
//! guarded tail runs the partial last four-super-block iteration.
//!
//! [`q4k_gemv_grouped`] reads each expert's rows once for all its slots,
//! walking a table that groups the slots by expert; [`grouped_run`] is the
//! one rule of that table's runs, shared with the other grouped kernels.
//! Their input, the q8_1 form of each slot's own column, comes from
//! [`Q4kSelKernels::enqueue_quantize_sel`]: it quantizes the columns of the
//! slots whose place is on the card — the columns a routed gate·up wrote —
//! and no other, so a column the host serves is never read, even by the
//! quantizer.
//!
//! [`q4k_gemv_tiles`] cuts the same table into tiles — up to [`TILE_COLS`]
//! consecutive entries of one expert's run, listed by [`grouped_tiles`] —
//! and runs a block per tile and eight weight rows with the m-column core,
//! so a weight row is read once for up to eight slots at a time. Its columns
//! are the table's entries (column `j` is the slot `order[j]`), which
//! [`Q4kSelKernels::enqueue_quantize_ord`] fills, and it scatters each
//! column's value to its slot. [`tile_at`] is the one reading of a tile,
//! shared with the other tiled kernels.

use crate::cores::{q4k_row_dot, q4k_row_dot_1col};
use crate::fault::{FaultSink, FaultSite, LAYER_NONE};
use crate::hybrid::HOST;
use crate::tensor::{DeviceTensor, Q8Act};
use crate::{GpuError, launch_u32};
use crate::{col_sums, q8_1_quant_block};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::ops::Range;
use std::sync::Arc;

/// Threads per block of the gemv kernels here, one warp per output row.
const THREADS: u32 = 256;
// The kernels' launch attributes spell the block as a literal.
const _: () = assert!(THREADS == 256);

/// Output rows a gemv block computes: one per warp.
const ROWS_PER_BLOCK: usize = THREADS as usize / 32;

/// The run of expert `e` in a grouped slot table: entries `j0 .. j1` of
/// `order`, `(j0, j1) = (start[e], start[e + 1])` — the layout a bucket pass
/// writes, one run per expert, each inside `order[.. n_slots]`. A run that
/// starts after its end or ends past `n_slots` names entries the table does
/// not hold: lane 0 raises [`FaultSite::ExpertId`] on `fault` and the answer
/// is `None`. A caller returns on `None` before it reads `order`, rather than
/// walk an empty run: joined into the walk's entry, the refusal costs the
/// kernel registers.
///
/// # Safety
///
/// `e + 1 < start.len()`, and the 32 lanes of the warp call it with the same
/// `e`: the run, like the fault, is the warp's.
#[inline(always)]
pub unsafe fn grouped_run(
    start: &[u32],
    e: usize,
    n_slots: usize,
    lane: usize,
    fault: FaultSink,
) -> Option<(usize, usize)> {
    // SAFETY: e + 1 < start.len() by this fn's contract.
    let (j0, j1) = unsafe {
        (
            *start.get_unchecked(e) as usize,
            *start.get_unchecked(e + 1) as usize,
        )
    };
    if j0 <= j1 && j1 <= n_slots {
        Some((j0, j1))
    } else {
        if lane == 0 {
            fault.raise(FaultSite::ExpertId);
        }
        None
    }
}

/// Slots a column tile covers at most: the m-column cores' width.
pub const TILE_COLS: usize = 8;

/// The shift of a tile word's expert: word `(e << TILE_E_SHIFT) | j` is
/// expert `e`'s tile from table entry `j`.
const TILE_E_SHIFT: u32 = 22;

/// The most experts, and the most table entries, a tile word can name.
pub const TILE_MAX_EXPERTS: usize = 1 << (32 - TILE_E_SHIFT);
pub const TILE_MAX_SLOTS: usize = 1 << TILE_E_SHIFT;

/// The most column tiles a grouped slot table of `n_slots` slots over
/// `n_experts` runs cuts into ([`grouped_tiles`]): a run of `r` slots makes
/// `⌈r / 8⌉ <= (r + 7) / 8` tiles and each tile holds a slot, so the count is
/// at most `min(n_slots, ⌊(n_slots + 7 · n_experts) / 8⌋)`. A tile table for
/// them is one count word and a word a tile.
#[must_use]
pub fn tile_cap(n_slots: usize, n_experts: usize) -> usize {
    n_slots.min((n_slots + (TILE_COLS - 1) * n_experts) / TILE_COLS)
}

/// Tile `g` of a tile table ([`grouped_tiles`]): its expert `e`, first table
/// entry `j` and column count `m` (`1 <= m <= TILE_COLS`, the run of `e` in
/// `start` ending at or past `j + m`). `None` for a tile past the table's
/// count `tiles[0]`, and for a word that names an expert past `n_experts` or
/// columns outside its run or past `n_slots` — for which lane 0 raises
/// [`FaultSite::ExpertId`] on `fault`.
///
/// # Safety
///
/// `1 + g < tiles.len()`, `n_experts < start.len()`, and the 32 lanes of the
/// warp call it with the same `g`.
#[allow(
    clippy::too_many_arguments,
    reason = "device helper: the tile table, the runs and their bounds (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn tile_at(
    tiles: &[u32],
    start: &[u32],
    n_experts: u32,
    n_slots: u32,
    g: u32,
    lane: usize,
    fault: FaultSink,
) -> Option<(usize, usize, usize)> {
    // SAFETY: 1 + g < tiles.len() by this fn's contract.
    let (count, word) = unsafe {
        (
            *tiles.get_unchecked(0),
            *tiles.get_unchecked(1 + g as usize),
        )
    };
    if g >= count {
        return None;
    }
    let (e, j) = (word >> TILE_E_SHIFT, word & ((1 << TILE_E_SHIFT) - 1));
    // An expert past the stack has no run: j1 = 0 fails the test below.
    let j1 = if e < n_experts {
        // SAFETY: e + 1 <= n_experts < start.len() by this fn's contract.
        unsafe { *start.get_unchecked(e as usize + 1) }
    } else {
        0
    };
    if j < j1 && j1 <= n_slots {
        let m = ((j1 - j) as usize).min(TILE_COLS);
        Some((e as usize, j as usize, m))
    } else {
        if lane == 0 {
            fault.raise(FaultSite::ExpertId);
        }
        None
    }
}

/// Lane 0's store of one tile column: the value `v` of table entry `at`
/// into row `r` of its slot `order[at]`'s rows of `y`, or
/// [`FaultSite::ExpertId`] on `fault` for an entry that names no slot.
///
/// # Safety
///
/// `at < order.len()`, `r < rpe`, `y.len() >= n_slots * rpe`, and no other
/// thread writes row `r` of that slot.
#[allow(
    clippy::too_many_arguments,
    reason = "device helper: a kernel's output, table and bounds (rust-quality R8)"
)]
#[inline(always)]
unsafe fn scatter_col(
    y: &mut DisjointSlice<f32>,
    order: &[u32],
    at: usize,
    n_slots: usize,
    rpe: usize,
    r: usize,
    v: f32,
    fault: FaultSink,
) {
    // SAFETY: at < order.len() by this fn's contract.
    let slot = unsafe { *order.get_unchecked(at) } as usize;
    if slot < n_slots {
        // SAFETY: slot < n_slots and r < rpe, so the index is below
        // n_slots * rpe <= y.len(); this thread is its only writer.
        unsafe { *y.get_unchecked_mut(slot * rpe + r) = v };
    } else {
        fault.raise(FaultSite::ExpertId);
    }
}

#[cuda_module]
mod q4k_sel_kernels {
    use super::*;

    /// Q4_K gemv over expert slots selected on the device (the MoE down
    /// projection, decode shape): one launch computes `n_slots` experts of
    /// the resident flat stack — `n_experts * rows_per_expert` rows of
    /// `36 * n_sb` words — where slot s reads activation column `s` of the
    /// q8_1 scratch (each expert's down input differs) and writes
    /// `y[s*rows_per_expert + r]`. Thread geometry as `q4k_gemv` (one warp
    /// per output row, eight rows per 256-thread block); thread row
    /// `n = slot * rows_per_expert + r` stores `y[n]` and reads weight row
    /// `sel[slot] * rows_per_expert + r`. The per-row body is
    /// `cores::q4k_row_dot_1col` with col0 = slot — the m = 1 path of
    /// `q4k_gemv`, same loads, same accumulation order, same reduction.
    ///
    /// An id >= n_experts cannot be rejected by the host contract (it
    /// lives in device memory): the slot's warps return before their first
    /// weight load — warp-uniform, no divergent branch — leaving that slot
    /// of `y` untouched and every other slot unaffected. [`HOST`] is a slot
    /// the host tier serves, skipped by contract; any other id past the
    /// stack raises [`FaultSite::ExpertId`] on `fault` first.
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
            w.len() >= n_experts * rows_per_expert * 36 * n_sb,
            q.len() >= n_slots * 256 * iters,
            s8.len() >= n_slots * 8 * n_sb,
            d8.len() >= n_slots * 2 * n_sb,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn q4k_gemv_sel(
        w: &[u32],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        n_sb: u32,
        iters: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % THREADS as usize;
        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        let slot = row / rows_per_expert as usize;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the
        // load is warp-uniform (all 32 lanes of the warp share `row`, hence
        // `slot`), so the out-of-range return below never diverges a warp.
        let id = unsafe { *sel.get_unchecked(slot) };
        let lane = warp::lane_id() as usize;
        if id >= n_experts {
            if id != HOST && lane == 0 {
                fault.raise(FaultSite::ExpertId);
            }
            return;
        }
        let row_abs = id as usize * rows_per_expert as usize + row % rows_per_expert as usize;
        // The core's caller contract, from the launch contract: row_abs <
        // n_experts * rows_per_expert rows of `w`, column slot < n_slots of
        // `q`/`s8`/`d8`, iters = ceil(n_sb/4) from the host, and all 32
        // lanes of the warp are here (both returns above are warp-uniform).
        let f0 = q4k_row_dot_1col(w, q, s8, d8, n_sb as usize, iters, row_abs, slot, lane);
        let s0 = warp::reduce_sum_f32(f0);
        if lane == 0 {
            // SAFETY: row < n_slots*rows_per_expert <= y.len() by the launch
            // contract; only lane 0 of the warp writes y[row].
            unsafe {
                *y.get_unchecked_mut(row) = s0;
            }
        }
    }

    /// [`q4k_gemv_sel`] with each expert's rows read once for all its
    /// slots: block `b` is expert `e = b / ⌈rows_per_expert / 8⌉` and warp
    /// `j` of it row `r = 8 (b mod ⌈rows_per_expert / 8⌉) + j`; the warp
    /// walks the expert's slots `order[start[e] .. start[e + 1]]` (a run per
    /// expert, as a bucket pass lays them out) and for slot `s` dots column
    /// `col0 + s` of the q8_1 scratch (`cols` columns) with
    /// `cores::q4k_row_dot_1col`, the warp reduction and the lane-0 store
    /// into `y[s * rows_per_expert + r]` — `q4k_gemv_sel`'s value for that
    /// slot, bit for bit. The row comes from memory once and from the cache
    /// for the expert's other slots. A run [`grouped_run`] refuses, and an
    /// `order` entry not below `n_slots`, name no slot: each raises
    /// [`FaultSite::ExpertId`] and is skipped.
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
            w.len() >= n_experts * rows_per_expert * 36 * n_sb,
            q.len() >= cols * 256 * iters,
            s8.len() >= cols * 8 * n_sb,
            d8.len() >= cols * 2 * n_sb,
            n_slots + col0 <= cols,
            order.len() >= n_slots,
            start.len() >= n_experts + 1,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn q4k_gemv_grouped(
        w: &[u32],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
        order: &[u32],
        start: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        col0: u32,
        cols: u32,
        n_sb: u32,
        iters: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::threadIdx_x() as usize;
        let rpe = rows_per_expert as usize;
        let tiles = rpe.div_ceil(ROWS_PER_BLOCK);
        let b = thread::blockIdx_x() as usize;
        let (e, r) = (b / tiles, (b % tiles) * ROWS_PER_BLOCK + t / 32);
        // Warp-uniform: all 32 lanes share e and r.
        if e >= n_experts as usize || r >= rpe {
            return;
        }
        let row_abs = e * rpe + r;
        let lane = warp::lane_id() as usize;
        // SAFETY: e + 1 <= n_experts < start.len() by the launch contract, and
        // the warp's lanes share e.
        let run = unsafe { grouped_run(start, e, n_slots as usize, lane, fault) };
        // Block-uniform: the block's warps share e, hence the run.
        let Some((j0, j1)) = run else {
            return;
        };
        let mut j = j0;
        while j < j1 {
            // SAFETY: j < j1 <= n_slots <= order.len() by `grouped_run` and the
            // launch contract; the load is warp-uniform.
            let slot = unsafe { *order.get_unchecked(j) } as usize;
            if slot >= n_slots as usize || col0 as usize + slot >= cols as usize {
                if lane == 0 {
                    fault.raise(FaultSite::ExpertId);
                }
            } else {
                // The core's caller contract, from the launch contract:
                // row_abs < n_experts * rows_per_expert rows of `w`, column
                // col0 + slot < cols of `q`/`s8`/`d8`, iters = ceil(n_sb/4)
                // from the host, and all 32 lanes of the warp are here.
                let f0 = q4k_row_dot_1col(
                    w,
                    q,
                    s8,
                    d8,
                    n_sb as usize,
                    iters,
                    row_abs,
                    col0 as usize + slot,
                    lane,
                );
                let s0 = warp::reduce_sum_f32(f0);
                if lane == 0 {
                    // SAFETY: slot < n_slots and r < rows_per_expert, so the
                    // value is below n_slots * rows_per_expert <= y.len();
                    // each slot sits in one expert's run, so lane 0 of this
                    // warp is its only writer.
                    unsafe {
                        *y.get_unchecked_mut(slot * rpe + r) = s0;
                    }
                }
            }
            j += 1;
        }
    }

    /// The tile table of a grouped slot table ([`q4k_gemv_grouped`]'s layout,
    /// runs `start` of `n_experts` experts inside `n_slots` slots), in one
    /// block: expert `e`'s run `start[e] .. start[e + 1]` is cut into
    /// `⌈run / 8⌉` column tiles of up to [`TILE_COLS`] consecutive entries,
    /// the tiles of the experts in turn, tile `g` written as the word `(e <<
    /// 22) | j` — its expert and first entry — at `tiles[1 + g]`, and their
    /// count at `tiles[0]`. A run that does not lie inside `0 ..
    /// start[n_experts] <= n_slots`, or tiles past `cap`, raise
    /// [`FaultSite::ExpertId`] on `fault` and leave a count of 0: no tile
    /// kernel then reads the table.
    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(
        domain = 1,
        block = (1024, 1, 1),
        requires = (
            start.len() >= n_experts + 1,
            tiles.len() >= cap + 1,
            n_experts <= 1024,
            n_slots <= 4194304
        )
    )]
    pub fn grouped_tiles(
        start: &[u32],
        n_experts: u32,
        n_slots: u32,
        cap: u32,
        fault: FaultSink,
        mut tiles: DisjointSlice<u32>,
    ) {
        static mut AT: SharedArray<u32, TILE_MAX_EXPERTS> = SharedArray::UNINIT;
        static mut COUNT: SharedArray<u32, 1> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let n = n_experts as usize;
        // SAFETY: block-shared; the raw forms reach the `static mut`s without
        // a reference.
        let (at, total) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut AT),
                SharedArray::as_raw_mut_ptr(&raw mut COUNT),
            )
        };
        // SAFETY: n < start.len() by the launch contract.
        let end = unsafe { *start.get_unchecked(n) };
        if tid < n {
            // SAFETY: tid + 1 <= n < start.len() by the launch contract.
            let (j0, j1) = unsafe { (*start.get_unchecked(tid), *start.get_unchecked(tid + 1)) };
            let c = if j0 <= j1 && j1 <= end && end <= n_slots {
                (j1 - j0).div_ceil(TILE_COLS as u32)
            } else {
                u32::MAX
            };
            // SAFETY: tid < n <= TILE_MAX_EXPERTS; thread tid is the entry's
            // only writer before the barrier.
            unsafe { *at.add(tid) = c };
        }
        thread::sync_threads();
        if tid == 0 {
            let mut sum = 0u32;
            let mut e = 0usize;
            while e < n {
                // SAFETY: e < n <= TILE_MAX_EXPERTS, published by the barrier;
                // thread 0 alone rewrites it now, before the next.
                let c = unsafe { *at.add(e) };
                if c == u32::MAX || c > cap - sum {
                    sum = u32::MAX;
                    break;
                }
                // SAFETY: as above.
                unsafe { *at.add(e) = sum };
                sum += c;
                e += 1;
            }
            let count = if sum == u32::MAX {
                fault.raise(FaultSite::ExpertId);
                0
            } else {
                sum
            };
            // SAFETY: the one shared word; thread 0 is its only writer, and
            // tiles[0] is its only writer too.
            unsafe {
                *total = count;
                *tiles.get_unchecked_mut(0) = count;
            }
        }
        thread::sync_threads();
        // SAFETY: the word thread 0 wrote before the barrier.
        if tid < n && unsafe { *total } != 0 {
            // SAFETY: tid < n, the entry thread 0 left before the barrier;
            // start as above.
            let (g0, j0, j1) = unsafe {
                (
                    *at.add(tid) as usize,
                    *start.get_unchecked(tid),
                    *start.get_unchecked(tid + 1),
                )
            };
            let mut j = j0;
            let mut g = g0;
            while j < j1 {
                // SAFETY: g < the count <= cap < tiles.len() - 1; the experts'
                // tiles are disjoint ranges of it, so thread tid is the only
                // writer of its tiles.
                unsafe { *tiles.get_unchecked_mut(1 + g) = (tid as u32) << TILE_E_SHIFT | j };
                j += TILE_COLS as u32;
                g += 1;
            }
        }
    }

    /// [`q4k_gemv_grouped`] by tile items: block `g + tile_cap · ρ` (of
    /// `tile_cap · row_tiles`, [`tile_cap`] of the table) is tile `g` of the
    /// table `tiles` ([`grouped_tiles`], [`tile_at`]) — expert `e`, the `m`
    /// table entries from `j` — on the eight weight rows `8ρ ..`, warp `w`
    /// dotting row `r = 8ρ + w` with the `m` columns `j .. j + m` of the q8_1
    /// scratch (column `j` holding slot `order[j]`'s activation) through the
    /// m-column core `cores::q4k_row_dot`, whose column c is
    /// `q4k_row_dot_1col` on that column bit for bit; lane 0 stores column c's
    /// sum into `y[order[j + c] · rows_per_expert + r]`, `q4k_gemv_sel`'s
    /// value for that slot. The blocks of one row tile are consecutive, so an
    /// expert's tiles run side by side on the same weight rows, each row read
    /// once for up to [`TILE_COLS`] slots. A block past the table's count
    /// returns; a tile word [`tile_at`] refuses and an entry that names no slot
    /// raise [`FaultSite::ExpertId`].
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    // Five blocks an SM: 48 registers a thread, what the m-column core
    // takes in `q4k_gemv_mcol`.
    #[launch_bounds(256, 5)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_experts * rows_per_expert * 36 * n_sb,
            q.len() >= n_slots * 256 * iters,
            s8.len() >= n_slots * 8 * n_sb,
            d8.len() >= n_slots * 2 * n_sb,
            order.len() >= n_slots,
            start.len() >= n_experts + 1,
            tiles.len() >= tile_cap + 1,
            tile_cap >= 1,
            y.len() >= n_slots * rows_per_expert,
            rows_per_expert >= 8 * row_tiles
        )
    )]
    pub fn q4k_gemv_tiles(
        w: &[u32],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
        order: &[u32],
        start: &[u32],
        tiles: &[u32],
        tile_cap: u32,
        row_tiles: u32,
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        n_sb: u32,
        iters: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        // Block b is tile b mod tile_cap (tile_cap >= 1 by the launch
        // contract) of row tile b / tile_cap: the tiles of one row tile are
        // consecutive blocks.
        let b = thread::blockIdx_x();
        let (g, rho) = (b % tile_cap, b / tile_cap);
        // Block-uniform: a block past the grid the tables were sized for
        // reads nothing.
        if rho >= row_tiles {
            return;
        }
        let lane = warp::lane_id() as usize;
        // SAFETY: 1 + g <= tile_cap < tiles.len() and n_experts < start.len()
        // by the launch contract; the block's warps share g.
        let tile = unsafe { tile_at(tiles, start, n_experts, n_slots, g, lane, fault) };
        // Block-uniform: every warp reads the same word.
        let Some((e, j, m)) = tile else {
            return;
        };
        let rpe = rows_per_expert as usize;
        let r = rho as usize * ROWS_PER_BLOCK + thread::threadIdx_x() as usize / 32;
        // The core's caller contract: row e·rpe + r < n_experts·rpe rows of
        // `w` (r < 8 · row_tiles <= rows_per_expert); columns j + m <= n_slots
        // of `q`/`s8`/`d8` by `tile_at` and the launch contract; iters =
        // ceil(n_sb/4) from the host; the warp's 32 lanes are here and m is
        // block-uniform.
        let f = q4k_row_dot(w, q, s8, d8, n_sb as usize, iters, e * rpe + r, j, m, lane);
        let v = col_sums(f, m);
        if lane == 0 {
            let n_slots = n_slots as usize;
            // SAFETY: each at = j + c < j + m <= n_slots <= order.len(), r <
            // rpe, y.len() >= n_slots·rpe by the launch contract; a table
            // entry sits in one tile, so lane 0 of this warp is the only
            // writer of its slot's row r.
            unsafe {
                scatter_col(&mut y, order, j, n_slots, rpe, r, v[0], fault);
                if m > 1 {
                    scatter_col(&mut y, order, j + 1, n_slots, rpe, r, v[1], fault);
                }
                if m > 2 {
                    scatter_col(&mut y, order, j + 2, n_slots, rpe, r, v[2], fault);
                }
                if m > 3 {
                    scatter_col(&mut y, order, j + 3, n_slots, rpe, r, v[3], fault);
                }
                if m > 4 {
                    scatter_col(&mut y, order, j + 4, n_slots, rpe, r, v[4], fault);
                }
                if m > 5 {
                    scatter_col(&mut y, order, j + 5, n_slots, rpe, r, v[5], fault);
                }
                if m > 6 {
                    scatter_col(&mut y, order, j + 6, n_slots, rpe, r, v[6], fault);
                }
                if m > 7 {
                    scatter_col(&mut y, order, j + 7, n_slots, rpe, r, v[7], fault);
                }
            }
        }
    }

    /// The q8_1 form of the first `start[n_experts]` columns of `x` — the
    /// entries of a grouped slot table ([`grouped_run`]'s layout), column `j` the
    /// activation of slot `order[j]` — each into the same column of the five
    /// planes: one 32-thread block per 128-value block of column `j <
    /// m_cols`, `q8_1_quant_block` as in [`q8_1_quantize_sel`], which does
    /// not depend on the column's position, so column `j` holds the bytes the
    /// plain quantizer writes for it. Columns from the count on return before
    /// any load and keep what they held. A non-finite value raises
    /// [`FaultSite::QuantColumn`] on `fault`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            x.len() >= m_cols * 256 * n_sb,
            start.len() >= n_experts + 1,
            q3.len() >= m_cols * 64 * half_it,
            q4.len() >= m_cols * 256 * quad_it,
            q6.len() >= m_cols * 128 * half_it,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb
        )
    )]
    pub fn q8_1_quantize_ord(
        x: &[f32],
        start: &[u32],
        n_experts: u32,
        m_cols: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut q3: DisjointSlice<u64>,
        mut q4: DisjointSlice<u32>,
        mut q6: DisjointSlice<u32>,
        mut s8: DisjointSlice<i32>,
        mut d8: DisjointSlice<f32>,
        fault: FaultSink,
    ) {
        let blk = thread::index_1d().get() / 32;
        let n_sb = n_sb as usize;
        let blocks_per_col = 2 * n_sb;
        if blk >= m_cols as usize * blocks_per_col {
            return;
        }
        let (j, b) = (blk / blocks_per_col, blk % blocks_per_col);
        // SAFETY: n_experts < start.len() by the launch contract. The 32 lanes
        // of the block share `j`: the return is uniform.
        if j >= unsafe { *start.get_unchecked(n_experts as usize) } as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        // The block body's contract: column j < m_cols and b < 2 n_sb by the
        // lines above, the launch contract bounds `x` (at base 0) and the five
        // planes at m_cols columns, and the block index is warp-uniform.
        q8_1_quant_block(
            x,
            0,
            j,
            b,
            n_sb,
            half_it,
            quad_it,
            lane,
            &mut q3,
            &mut q4,
            &mut q6,
            &mut s8,
            &mut d8,
            fault,
            FaultSite::QuantColumn,
        );
    }

    /// The q8_1 form of the activation columns the card's expert slots read:
    /// one 32-thread block per 128-value block of column `c0 + j` (`j <
    /// m_cols`), which quantizes it from the same column of `x` into the five
    /// planes — `q8_1_quant_block`, the body of every q8_1 quantizer, so a
    /// column's bytes are the plain quantizer's — when its slot's place
    /// `sel[j]` is below `n_card`, and returns before any load otherwise: a
    /// slot the host serves has no activation here, and its column keeps
    /// what it held. A non-finite value raises [`FaultSite::QuantColumn`] on
    /// `fault`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        requires = (
            x.len() >= (c0 + m_cols) * 256 * n_sb,
            sel.len() >= m_cols,
            q3.len() >= (c0 + m_cols) * 64 * half_it,
            q4.len() >= (c0 + m_cols) * 256 * quad_it,
            q6.len() >= (c0 + m_cols) * 128 * half_it,
            s8.len() >= (c0 + m_cols) * 8 * n_sb,
            d8.len() >= (c0 + m_cols) * 2 * n_sb
        )
    )]
    pub fn q8_1_quantize_sel(
        x: &[f32],
        sel: &[u32],
        c0: u32,
        m_cols: u32,
        n_card: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut q3: DisjointSlice<u64>,
        mut q4: DisjointSlice<u32>,
        mut q6: DisjointSlice<u32>,
        mut s8: DisjointSlice<i32>,
        mut d8: DisjointSlice<f32>,
        fault: FaultSink,
    ) {
        let blk = thread::index_1d().get() / 32;
        let n_sb = n_sb as usize;
        let blocks_per_col = 2 * n_sb;
        if blk >= m_cols as usize * blocks_per_col {
            return;
        }
        let (j, b) = (blk / blocks_per_col, blk % blocks_per_col);
        // SAFETY: j < m_cols <= sel.len() by the launch contract. The 32
        // lanes of the block share `blk`, hence `j`: the return is uniform.
        if unsafe { *sel.get_unchecked(j) } >= n_card {
            return;
        }
        let lane = warp::lane_id() as usize;
        // The block body's contract: column c0 + j < c0 + m_cols and b < 2
        // n_sb by the lines above, the launch contract bounds `x` (at base 0)
        // and the five planes at c0 + m_cols columns, and the block index is
        // warp-uniform.
        q8_1_quant_block(
            x,
            0,
            c0 as usize + j,
            b,
            n_sb,
            half_it,
            quad_it,
            lane,
            &mut q3,
            &mut q4,
            &mut q6,
            &mut s8,
            &mut d8,
            fault,
            FaultSite::QuantColumn,
        );
    }
}

/// What [`Q4kSelKernels::enqueue_quantize_sel`] quantizes: the columns
/// `cols` of `x` (the activation's `k` values a column) whose slot — column
/// `c` is slot `c − cols.start` — has its place `sel[c − cols.start]` below
/// `n_card`, the count of the card's experts.
pub struct QuantSel<'a> {
    pub x: &'a DeviceBuffer<f32>,
    pub cols: Range<usize>,
    pub sel: &'a DeviceBuffer<u32>,
    pub n_card: usize,
}

/// The loaded Q4_K expert-select module and its enqueue API. Owns no
/// context and no stream — every enqueue takes the engine stream
/// (`Gpu::stream()`), so launches order with the rest of the step and are
/// capturable.
pub struct Q4kSelKernels {
    module: q4k_sel_kernels::LoadedModule,
    /// The fault word of the `Gpu` that owns the context.
    fault: Arc<DeviceBuffer<u32>>,
}

impl Q4kSelKernels {
    /// Load this file's device bundle into `ctx`, raising into `word`, the
    /// fault word of the `Gpu` that owns `ctx` ([`crate::Gpu::fault_word`]);
    /// a word of another context is refused. Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        word: &Arc<DeviceBuffer<u32>>,
    ) -> Result<Q4kSelKernels, GpuError> {
        let fault = crate::module_fault_word(ctx, word, "Q4kSelKernels::load")?;
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { q4k_sel_kernels::load(ctx)? };
        Ok(Q4kSelKernels { module, fault })
    }

    /// Enqueue the device-indirect MoE down projection for a Q4_K expert
    /// stack: one launch computes `n_slots` experts, slot s writing
    /// `y[s*rows_per_expert .. (s+1)*rows_per_expert]` as
    /// `w[sel[s]*rows_per_expert .. +rows_per_expert] · act` column `s`.
    /// `act` holds exactly `n_slots` quantized columns, one per slot — a
    /// shared-input (m = 1) `act` is a misuse here. `w` is the full resident
    /// stack in `Gpu::enqueue_gemv_q4k`'s row format (`36 * n_sb` u32 words
    /// per row), `w.rows()` a positive multiple of `rows_per_expert`
    /// (`n_experts = w.rows() / rows_per_expert`). `sel` is a device buffer
    /// of at least `n_slots` ids read by the kernel per launch, so a
    /// captured graph replay picks up new ids written between replays; an
    /// id >= n_experts leaves that slot of `y` untouched, and one that is not
    /// [`HOST`] raises [`FaultSite::ExpertId`] on the owning `Gpu`'s fault
    /// word as an unlabelled launch ([`LAYER_NONE`]: the launcher knows no
    /// layer). Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
    pub fn enqueue_gemv_q4k_sel(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        sel: &DeviceBuffer<u32>,
        n_slots: usize,
        rows_per_expert: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_gemv_q4k_sel";
        let n_sb = act.n_sb();
        if act.m() != n_slots {
            return Err(GpuError::shape(
                what,
                format!(
                    "one activation column per slot: act.m() = {} must equal \
                 n_slots = {n_slots}",
                    act.m()
                ),
            ));
        }
        if w.cols() != 36 * n_sb {
            return Err(GpuError::shape(
                what,
                format!(
                    "Q4_K rows are 36*{n_sb} = {} words at K={}, got {}",
                    36 * n_sb,
                    act.k(),
                    w.cols()
                ),
            ));
        }
        if rows_per_expert == 0 || !w.rows().is_multiple_of(rows_per_expert) {
            return Err(GpuError::shape(
                what,
                format!(
                    "w.rows() {} must be a positive multiple of \
                 rows_per_expert {rows_per_expert}",
                    w.rows()
                ),
            ));
        }
        if n_slots == 0 || sel.len() < n_slots {
            return Err(GpuError::shape(
                what,
                format!(
                    "need n_slots >= 1 and sel.len() >= n_slots, got \
                 n_slots {n_slots} sel.len() {}",
                    sel.len()
                ),
            ));
        }
        if y.len() < n_slots * rows_per_expert {
            return Err(GpuError::shape(
                what,
                format!(
                    "y.len() {} < n_slots*rows_per_expert = {}",
                    y.len(),
                    n_slots * rows_per_expert
                ),
            ));
        }
        let grid = launch_u32(
            what,
            "grid",
            (n_slots * rows_per_expert).div_ceil(ROWS_PER_BLOCK),
        )?;
        let n_experts = launch_u32(what, "n_experts", w.rows() / rows_per_expert)?;
        let rows_per_expert = launch_u32(what, "rows_per_expert", rows_per_expert)?;
        let n_slots = launch_u32(what, "n_slots", n_slots)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_q4k_gemv_sel(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.q4k_gemv_sel(
            stream,
            &prep,
            w.buf(),
            &act.q4,
            &act.s8,
            &act.d8,
            sel,
            n_experts,
            rows_per_expert,
            n_slots,
            n_sb,
            n_sb.div_ceil(4),
            crate::sink_over(&self.fault, LAYER_NONE),
            y,
        )?;
        Ok(())
    }

    /// Enqueue [`q4k_gemv_grouped`]: the down of `n_slots` slots, each
    /// expert's rows read once, slot `s` reading column `col0 + s` of `act`
    /// and writing `y[s * rows_per_expert ..]`; `order` and `start` group the
    /// slots by expert, one run per expert of `w` (`w.rows() /
    /// rows_per_expert` of them). Each slot's value is
    /// [`Q4kSelKernels::enqueue_gemv_q4k_sel`]'s. A run that does not fit
    /// the table ([`grouped_run`]) and an `order` entry that names no slot
    /// raise [`FaultSite::ExpertId`] on `fault`. Asynchronous,
    /// allocation-free.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher over the kernel's flat inputs (rust-quality R8)"
    )]
    pub fn enqueue_gemv_q4k_grouped(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        order: &DeviceBuffer<u32>,
        start: &DeviceBuffer<u32>,
        n_slots: usize,
        col0: usize,
        rows_per_expert: usize,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_gemv_q4k_grouped";
        let n_sb = act.n_sb();
        let shape = |detail: String| GpuError::shape(what, detail);
        if w.cols() != 36 * n_sb
            || rows_per_expert == 0
            || !w.rows().is_multiple_of(rows_per_expert)
        {
            return Err(shape(format!(
                "Q4_K rows of 36*{n_sb} words in experts of {rows_per_expert} rows, got {} x {}",
                w.rows(),
                w.cols()
            )));
        }
        let n_experts = w.rows() / rows_per_expert;
        if n_slots == 0
            || col0 + n_slots > act.m()
            || order.len() < n_slots
            || start.len() < n_experts + 1
            || y.len() < n_slots * rows_per_expert
        {
            return Err(shape(format!(
                "{n_slots} slots from column {col0} of {}: order {}, start {} for {n_experts} \
                 experts, y {}",
                act.m(),
                order.len(),
                start.len(),
                y.len()
            )));
        }
        let grid = launch_u32(
            what,
            "grid",
            n_experts * rows_per_expert.div_ceil(ROWS_PER_BLOCK),
        )?;
        let prep = self
            .module
            .prepare_q4k_gemv_grouped(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.q4k_gemv_grouped(
            stream,
            &prep,
            w.buf(),
            &act.q4,
            &act.s8,
            &act.d8,
            order,
            start,
            launch_u32(what, "n_experts", n_experts)?,
            launch_u32(what, "rows_per_expert", rows_per_expert)?,
            launch_u32(what, "n_slots", n_slots)?,
            launch_u32(what, "col0", col0)?,
            launch_u32(what, "cols", act.m())?,
            launch_u32(what, "n_sb", n_sb)?,
            launch_u32(what, "iters", n_sb.div_ceil(4))?,
            fault,
            y,
        )?;
        Ok(())
    }

    /// Enqueue [`grouped_tiles`]: the tile table of the grouped slot table
    /// `start` (runs of `n_experts` experts inside `n_slots` slots) into
    /// `tiles`, which holds [`tile_cap`]`(n_slots, n_experts) + 1` words. A run
    /// outside the table raises [`FaultSite::ExpertId`] on `fault` and leaves
    /// no tiles. One launch. Asynchronous, allocation-free.
    pub fn enqueue_grouped_tiles(
        &self,
        stream: &CudaStream,
        start: &DeviceBuffer<u32>,
        n_experts: usize,
        n_slots: usize,
        fault: FaultSink,
        tiles: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_grouped_tiles";
        let cap = tile_cap(n_slots, n_experts);
        if n_experts == 0
            || n_experts > TILE_MAX_EXPERTS
            || n_slots > TILE_MAX_SLOTS
            || start.len() < n_experts + 1
            || tiles.len() < cap + 1
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_slots} slots over {n_experts} experts (1..={TILE_MAX_EXPERTS}, at most \
                     {TILE_MAX_SLOTS} slots): start {}, tiles {} for {} tiles",
                    start.len(),
                    tiles.len(),
                    cap
                ),
            ));
        }
        let prep = self.module.prepare_grouped_tiles(LaunchConfig1D::new(
            1,
            TILE_MAX_EXPERTS as u32,
            0,
        ))?;
        self.module.grouped_tiles(
            stream,
            &prep,
            start,
            launch_u32(what, "n_experts", n_experts)?,
            launch_u32(what, "n_slots", n_slots)?,
            launch_u32(what, "cap", cap)?,
            fault,
            tiles,
        )?;
        Ok(())
    }

    /// Enqueue [`q4k_gemv_tiles`]: the down of the `n_slots` slots of the
    /// grouped slot table `order`/`start` (one run per expert of `w`,
    /// `w.rows() / rows_per_expert` of them) by the tiles of `tiles`
    /// ([`Q4kSelKernels::enqueue_grouped_tiles`] of the same table), table
    /// entry `j` reading column `j` of `act` and writing its slot `order[j]`'s
    /// rows `y[order[j] · rows_per_expert ..]`. Each slot's value is
    /// [`Q4kSelKernels::enqueue_gemv_q4k_sel`]'s. A tile word that names no
    /// run and an entry that names no slot raise [`FaultSite::ExpertId`] on
    /// `fault`. The tile count is the table's, a device value: the grid is
    /// sized for [`tile_cap`] of them and a block past the count returns.
    /// Asynchronous, allocation-free.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher over the kernel's flat inputs (rust-quality R8)"
    )]
    pub fn enqueue_gemv_q4k_tiles(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        order: &DeviceBuffer<u32>,
        start: &DeviceBuffer<u32>,
        tiles: &DeviceBuffer<u32>,
        n_slots: usize,
        rows_per_expert: usize,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_gemv_q4k_tiles";
        let n_sb = act.n_sb();
        let shape = |detail: String| GpuError::shape(what, detail);
        if w.cols() != 36 * n_sb
            || rows_per_expert == 0
            || !rows_per_expert.is_multiple_of(ROWS_PER_BLOCK)
            || !w.rows().is_multiple_of(rows_per_expert)
        {
            return Err(shape(format!(
                "Q4_K rows of 36*{n_sb} words in experts of {rows_per_expert} rows (a multiple of \
                 {ROWS_PER_BLOCK}), got {} x {}",
                w.rows(),
                w.cols()
            )));
        }
        let n_experts = w.rows() / rows_per_expert;
        let cap = tile_cap(n_slots, n_experts);
        if cap == 0
            || n_slots > act.m()
            || order.len() < n_slots
            || start.len() < n_experts + 1
            || tiles.len() < cap + 1
            || y.len() < n_slots * rows_per_expert
        {
            return Err(shape(format!(
                "{n_slots} slots over {} columns: order {}, start {} for {n_experts} experts, \
                 tiles {} for {cap} tiles, y {}",
                act.m(),
                order.len(),
                start.len(),
                tiles.len(),
                y.len()
            )));
        }
        let row_tiles = rows_per_expert / ROWS_PER_BLOCK;
        let blocks = launch_u32(what, "grid", cap * row_tiles)?;
        let grid = (
            launch_u32(what, "tile_cap", cap)?,
            launch_u32(what, "row_tiles", row_tiles)?,
        );
        let prep = self
            .module
            .prepare_q4k_gemv_tiles(LaunchConfig1D::new(blocks, THREADS, 0))?;
        self.module.q4k_gemv_tiles(
            stream,
            &prep,
            w.buf(),
            &act.q4,
            &act.s8,
            &act.d8,
            order,
            start,
            tiles,
            grid.0,
            grid.1,
            launch_u32(what, "n_experts", n_experts)?,
            launch_u32(what, "rows_per_expert", rows_per_expert)?,
            launch_u32(what, "n_slots", n_slots)?,
            launch_u32(what, "n_sb", n_sb)?,
            launch_u32(what, "iters", n_sb.div_ceil(4))?,
            fault,
            y,
        )?;
        Ok(())
    }

    /// Enqueue [`q8_1_quantize_ord`]: the q8_1 form of the first
    /// `start[n_experts]` columns of `x` (`act.k()` values a column; at most
    /// `m_cols`, the columns launched), each into the same column of `act`
    /// with the bytes [`crate::Gpu::enqueue_quantize_q8_1`] writes for it;
    /// every other column keeps what it held. A non-finite value raises
    /// [`FaultSite::QuantColumn`] on `fault`. Asynchronous, allocation-free.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher over the kernel's flat inputs (rust-quality R8)"
    )]
    pub fn enqueue_quantize_ord(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        start: &DeviceBuffer<u32>,
        n_experts: usize,
        m_cols: usize,
        fault: FaultSink,
        act: &mut Q8Act,
    ) -> Result<(), GpuError> {
        let what = "enqueue_quantize_ord";
        let (k, n_sb) = (act.k(), act.n_sb());
        if m_cols == 0 || m_cols > act.m() || x.len() < m_cols * k || start.len() < n_experts + 1 {
            return Err(GpuError::shape(
                what,
                format!(
                    "{m_cols} columns of {k} values from {} into {} columns, start {} for \
                     {n_experts} experts",
                    x.len(),
                    act.m(),
                    start.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", m_cols * 2 * n_sb)?;
        let prep = self
            .module
            .prepare_q8_1_quantize_ord(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.q8_1_quantize_ord(
            stream,
            &prep,
            x,
            start,
            launch_u32(what, "n_experts", n_experts)?,
            launch_u32(what, "m_cols", m_cols)?,
            launch_u32(what, "n_sb", n_sb)?,
            launch_u32(what, "half_it", n_sb.div_ceil(2))?,
            launch_u32(what, "quad_it", n_sb.div_ceil(4))?,
            &mut act.q3,
            &mut act.q4,
            &mut act.q6,
            &mut act.s8,
            &mut act.d8,
            fault,
        )?;
        Ok(())
    }

    /// Enqueue [`q8_1_quantize_sel`]: the q8_1 form of the columns `q.cols`
    /// of `q.x` whose slot's place in `q.sel` is on the card, each into the
    /// same column of `act` with the bytes [`crate::Gpu::enqueue_quantize_q8_1`]
    /// writes for it; every other column of `act` keeps what it held. These
    /// are the columns a routed gate·up wrote and the only ones the `_sel`
    /// and grouped downs read. A non-finite value raises
    /// [`FaultSite::QuantColumn`] on `fault`. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_quantize_sel(
        &self,
        stream: &CudaStream,
        q: &QuantSel<'_>,
        fault: FaultSink,
        act: &mut Q8Act,
    ) -> Result<(), GpuError> {
        let what = "enqueue_quantize_sel";
        let (k, n_sb, m) = (act.k(), act.n_sb(), q.cols.len());
        if m == 0
            || q.cols.end > act.m()
            || q.x.len() < q.cols.end * k
            || q.sel.len() < m
            || q.n_card == 0
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "columns {:?} of {} values from {} into {} columns, {} places, a card of {} \
                     experts: the columns non-empty and inside both, a place a column, at least \
                     one expert",
                    q.cols,
                    k,
                    q.x.len(),
                    act.m(),
                    q.sel.len(),
                    q.n_card
                ),
            ));
        }
        let grid = launch_u32(what, "grid", m * 2 * n_sb)?;
        let prep = self
            .module
            .prepare_q8_1_quantize_sel(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.q8_1_quantize_sel(
            stream,
            &prep,
            q.x,
            q.sel,
            launch_u32(what, "c0", q.cols.start)?,
            launch_u32(what, "m_cols", m)?,
            launch_u32(what, "n_card", q.n_card)?,
            launch_u32(what, "n_sb", n_sb)?,
            launch_u32(what, "half_it", n_sb.div_ceil(2))?,
            launch_u32(what, "quad_it", n_sb.div_ceil(4))?,
            &mut act.q3,
            &mut act.q4,
            &mut act.q6,
            &mut act.s8,
            &mut act.d8,
            fault,
        )?;
        Ok(())
    }
}
