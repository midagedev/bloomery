//! The family's grouped entries: a block's card slots by tile items — an
//! expert, a tile of eight of its weight rows, up to [`TILE_COLS`] of its
//! slots — over a grouped slot table (`q4k_sel`'s module doc: `order`, runs
//! `start`, tiles from `q4k_sel::grouped_tiles`), each weight row read once
//! for up to eight slots. Every slot's value is its `_sel` entry's bit for
//! bit ([`super::sel`]): column `c` of Walk A's m-column core
//! ([`walk::row_dot`]) is the one-column walk on that column, and the warp
//! tree, the rule and the store are the `_sel` bodies'.
//!
//! - [`kq_card_gather`]: the q8_1 planes Walk A reads (q4, s8, d8) of a
//!   block's token columns by table entry, column `j` the token of slot
//!   `order[j]`, `slots_per_col` slots a token.
//! - `kq_gate_up_act_*_tiles`: the gate·up with its rule as a launch
//!   argument over the gathered entries, into the rule's outputs by entry —
//!   [`super::sel::gate_up_act_body`]'s value for the entry's slot.
//! - [`q5k_gemv_tiles`]: the Q5_K down over the entries' q8_1 columns
//!   (`q4k_sel::q8_1_quantize_ord`), each value scattered to its slot's rows
//!   — [`super::sel::gemv_sel_body`]'s value for that slot. The Q4_K down is
//!   `q4k_sel::q4k_gemv_tiles`.
//!
//! A tile word that names no run of the table raises [`FaultSite::ExpertId`]
//! (`q4k_sel::tile_at`), as does an entry that names no slot; a block past
//! the table's count returns before any load.

use super::act::Act;
use super::q5k::Q5k;
use super::walk::{Q4k, SbDecode, row_dot};
use crate::fault::{FaultSink, FaultSite};
use crate::q4k_sel::{TILE_COLS, scatter_col, tile_at, tile_cap};
use crate::tensor::{DeviceTensor, Q8Act};
use crate::{GpuError, col_sums, launch_u32, store_cols};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads per block of every entry here, one warp per weight row.
const THREADS: u32 = 256;
// The entries' launch attributes spell the block as a literal.
const _: () = assert!(THREADS == 256);

/// Weight rows a tile block computes: one per warp.
const ROWS_PER_BLOCK: usize = THREADS as usize / 32;

// The m-column core takes the tile's columns at most.
const _: () = assert!(TILE_COLS == 8);

/// The body of a grouped gate·up entry over format `D` (module doc): block
/// `g + tile_cap · ρ` is tile `g` of `tiles` — expert `e`, the `m` table
/// entries from `j` — on the eight weight rows `8ρ ..`; warp `w` dots row `r
/// = 8ρ + w` of both stacks with columns `j .. j + m` of the gathered planes
/// through [`row_dot`], reduces each column with the warp tree and stores
/// `h[(j + c) · rows_per_expert + r] = act::apply(act, limit, g, u)` from
/// lane 0.
///
/// # Safety
///
/// The entry's launch contract: both stacks hold `n_experts ·
/// rows_per_expert` rows of `D::WORDS · n_sb` words, the planes hold
/// `n_slots` columns, `start.len() >= n_experts + 1`, `tiles.len() >=
/// tile_cap + 1 >= 2`, `h.len() >= n_slots · rows_per_expert`,
/// `rows_per_expert >= 8 · row_tiles`; `iters = ceil(n_sb / 4)`; all 256
/// threads of the block here.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn gate_up_tiles_body<D: SbDecode>(
    wg: &[u32],
    wu: &[u32],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    start: &[u32],
    tiles: &[u32],
    tile_cap: u32,
    row_tiles: u32,
    n_experts: u32,
    rows_per_expert: u32,
    n_slots: u32,
    n_sb: u32,
    iters: u32,
    act: u32,
    limit: f32,
    fault: FaultSink,
    h: &mut DisjointSlice<f32>,
) {
    // Block b is tile b mod tile_cap of row tile b / tile_cap: the tiles of
    // one row tile are consecutive blocks.
    let b = thread::blockIdx_x();
    let (g, rho) = (b % tile_cap, b / tile_cap);
    // Block-uniform: a block past the grid the tables were sized for reads
    // nothing.
    if rho >= row_tiles {
        return;
    }
    let lane = warp::lane_id() as usize;
    // SAFETY: 1 + g <= tile_cap < tiles.len() and n_experts < start.len() by
    // this fn's contract; the block's warps share g.
    let tile = unsafe { tile_at(tiles, start, n_experts, n_slots, g, lane, fault) };
    // Block-uniform: every warp reads the same word.
    let Some((e, j, m)) = tile else {
        return;
    };
    let rpe = rows_per_expert as usize;
    let r = rho as usize * ROWS_PER_BLOCK + thread::threadIdx_x() as usize / 32;
    let row_abs = e * rpe + r;
    // SAFETY: row_abs < n_experts · rpe rows of both stacks (e < n_experts
    // by `tile_at`, r < 8 · row_tiles <= rpe); columns j + m <= n_slots of
    // the planes by `tile_at`; 1 <= m <= 8 and block-uniform; iters =
    // ceil(n_sb/4); the warp's 32 lanes are here.
    let (fg, fu) = unsafe {
        (
            row_dot::<D>(wg, q, s8, d8, n_sb as usize, iters, row_abs, j, m, lane),
            row_dot::<D>(wu, q, s8, d8, n_sb as usize, iters, row_abs, j, m, lane),
        )
    };
    let g = col_sums(fg, m);
    let u = col_sums(fu, m);
    if lane == 0 {
        let v = [
            super::act::apply(act, limit, g[0], u[0]),
            super::act::apply(act, limit, g[1], u[1]),
            super::act::apply(act, limit, g[2], u[2]),
            super::act::apply(act, limit, g[3], u[3]),
            super::act::apply(act, limit, g[4], u[4]),
            super::act::apply(act, limit, g[5], u[5]),
            super::act::apply(act, limit, g[6], u[6]),
            super::act::apply(act, limit, g[7], u[7]),
        ];
        // SAFETY: values (j + c)·rpe + r for c < m are below n_slots·rpe <=
        // h.len(); a table entry sits in one tile, so lane 0 of this row's
        // warp is their only writer.
        unsafe { store_cols(h, j * rpe + r, rpe, m, v) };
    }
}

#[cuda_module]
mod kquant_tile_kernels {
    use super::*;

    /// The Walk A planes of a grouped table's entries: block `j` copies the
    /// q4, s8 and d8 words of token column `order[j] / slots_per_col` of
    /// `q_in`, `s8_in` and `d8_in` (`cols` columns) to column `j` of the
    /// outputs, for each entry `j` below the count `start[n_experts]` — a
    /// word copy, so column `j` is its slot's activation as the slot's
    /// gate·up `_sel` reads it. Entries from the count on are left as they
    /// were; an entry that names no slot of the block or a token past `cols`
    /// raises [`FaultSite::ExpertId`] on `fault` and is skipped.
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
            q_in.len() >= 256 * iters * cols,
            s8_in.len() >= 8 * n_sb * cols,
            d8_in.len() >= 2 * n_sb * cols,
            order.len() >= n_slots,
            start.len() >= n_experts + 1,
            slots_per_col >= 1,
            q_out.len() >= 256 * iters * n_slots,
            s8_out.len() >= 8 * n_sb * n_slots,
            d8_out.len() >= 2 * n_sb * n_slots
        )
    )]
    pub fn kq_card_gather(
        q_in: &[u32],
        s8_in: &[i32],
        d8_in: &[f32],
        order: &[u32],
        start: &[u32],
        n_experts: u32,
        n_slots: u32,
        cols: u32,
        slots_per_col: u32,
        n_sb: u32,
        iters: u32,
        fault: FaultSink,
        mut q_out: DisjointSlice<u32>,
        mut s8_out: DisjointSlice<i32>,
        mut d8_out: DisjointSlice<f32>,
    ) {
        let j = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        // SAFETY: n_experts < start.len() by the launch contract.
        let count = unsafe { *start.get_unchecked(n_experts as usize) } as usize;
        // Block-uniform: j, the count and the entry are the block's.
        if j >= n_slots as usize || j >= count {
            return;
        }
        // SAFETY: j < n_slots <= order.len() by the launch contract.
        let slot = unsafe { *order.get_unchecked(j) } as usize;
        let col = slot / slots_per_col as usize;
        if slot >= n_slots as usize || col >= cols as usize {
            if tid == 0 {
                fault.raise(FaultSite::ExpertId);
            }
            return;
        }
        let (qc, sc, dc) = (256 * iters as usize, 8 * n_sb as usize, 2 * n_sb as usize);
        let mut k = tid;
        while k < qc {
            // SAFETY: col < cols and j < n_slots with k < qc, inside both
            // buffers by the launch contract; thread tid of block j is the
            // only writer of word k of column j.
            unsafe { *q_out.get_unchecked_mut(j * qc + k) = *q_in.get_unchecked(col * qc + k) };
            k += THREADS as usize;
        }
        let mut k = tid;
        while k < sc {
            // SAFETY: as above, with sc words a column.
            unsafe { *s8_out.get_unchecked_mut(j * sc + k) = *s8_in.get_unchecked(col * sc + k) };
            k += THREADS as usize;
        }
        let mut k = tid;
        while k < dc {
            // SAFETY: as above, with dc words a column.
            unsafe { *d8_out.get_unchecked_mut(j * dc + k) = *d8_in.get_unchecked(col * dc + k) };
            k += THREADS as usize;
        }
    }

    /// The Q4_K grouped gate·up ([`gate_up_tiles_body`] over [`Q4k`]): every
    /// entry's rows are `kq_gate_up_act_q4k`'s for its slot.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 4)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            wg.len() >= n_experts * rows_per_expert * 36 * n_sb,
            wu.len() >= n_experts * rows_per_expert * 36 * n_sb,
            q.len() >= n_slots * 256 * iters,
            s8.len() >= n_slots * 8 * n_sb,
            d8.len() >= n_slots * 2 * n_sb,
            start.len() >= n_experts + 1,
            tiles.len() >= tile_cap + 1,
            tile_cap >= 1,
            h.len() >= n_slots * rows_per_expert,
            rows_per_expert >= 8 * row_tiles
        )
    )]
    pub fn kq_gate_up_act_q4k_tiles(
        wg: &[u32],
        wu: &[u32],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
        start: &[u32],
        tiles: &[u32],
        tile_cap: u32,
        row_tiles: u32,
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        n_sb: u32,
        iters: u32,
        act: u32,
        limit: f32,
        fault: FaultSink,
        mut h: DisjointSlice<f32>,
    ) {
        // SAFETY: the launch contract is the body's (36 = Q4k::WORDS); the
        // host passes iters = ceil(n_sb/4) and one of Act::code's codes.
        unsafe {
            gate_up_tiles_body::<Q4k>(
                wg,
                wu,
                q,
                s8,
                d8,
                start,
                tiles,
                tile_cap,
                row_tiles,
                n_experts,
                rows_per_expert,
                n_slots,
                n_sb,
                iters,
                act,
                limit,
                fault,
                &mut h,
            );
        }
    }

    /// The Q5_K grouped gate·up ([`gate_up_tiles_body`] over [`Q5k`]): every
    /// entry's rows are `kq_gate_up_act_q5k`'s for its slot.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 4)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            wg.len() >= n_experts * rows_per_expert * 44 * n_sb,
            wu.len() >= n_experts * rows_per_expert * 44 * n_sb,
            q.len() >= n_slots * 256 * iters,
            s8.len() >= n_slots * 8 * n_sb,
            d8.len() >= n_slots * 2 * n_sb,
            start.len() >= n_experts + 1,
            tiles.len() >= tile_cap + 1,
            tile_cap >= 1,
            h.len() >= n_slots * rows_per_expert,
            rows_per_expert >= 8 * row_tiles
        )
    )]
    pub fn kq_gate_up_act_q5k_tiles(
        wg: &[u32],
        wu: &[u32],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
        start: &[u32],
        tiles: &[u32],
        tile_cap: u32,
        row_tiles: u32,
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        n_sb: u32,
        iters: u32,
        act: u32,
        limit: f32,
        fault: FaultSink,
        mut h: DisjointSlice<f32>,
    ) {
        // SAFETY: the launch contract is the body's (44 = Q5k::WORDS); the
        // host passes iters = ceil(n_sb/4) and one of Act::code's codes.
        unsafe {
            gate_up_tiles_body::<Q5k>(
                wg,
                wu,
                q,
                s8,
                d8,
                start,
                tiles,
                tile_cap,
                row_tiles,
                n_experts,
                rows_per_expert,
                n_slots,
                n_sb,
                iters,
                act,
                limit,
                fault,
                &mut h,
            );
        }
    }

    /// The Q5_K grouped down: block `g + tile_cap · ρ` is tile `g` of `tiles`
    /// — expert `e`, the `m` table entries from `j` — on the eight weight rows
    /// `8ρ ..`; warp `w` dots row `r = 8ρ + w` with columns `j .. j + m` of
    /// the entries' q8_1 planes through [`row_dot`] over [`Q5k`], reduces
    /// each column with the warp tree, and lane 0 stores column `c` into
    /// `y[order[j + c] · rows_per_expert + r]`: `q5k_gemv_sel`'s value for
    /// that slot and row.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 4)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_experts * rows_per_expert * 44 * n_sb,
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
    pub fn q5k_gemv_tiles(
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
        // SAFETY: row e·rpe + r < n_experts·rpe rows of `w` (r < 8 · row_tiles
        // <= rpe); columns j + m <= n_slots of the planes by `tile_at` and the
        // launch contract; 1 <= m <= 8 block-uniform; iters = ceil(n_sb/4)
        // from the host; the warp's 32 lanes are here.
        let f =
            unsafe { row_dot::<Q5k>(w, q, s8, d8, n_sb as usize, iters, e * rpe + r, j, m, lane) };
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
}

// The entries spell Q4_K's super-block words as 36 and Q5_K's as 44.
const _: () = assert!(Q4k::WORDS == 36 && Q5k::WORDS == 44);

/// A grouped slot table on the card: `order` (the entries, slot ids), the
/// runs `start` of `n_experts` experts and their tiles
/// (`q4k_sel::grouped_tiles`), over `n_slots` slots.
pub struct TileTable<'a> {
    pub order: &'a DeviceBuffer<u32>,
    pub start: &'a DeviceBuffer<u32>,
    pub tiles: &'a DeviceBuffer<u32>,
    pub n_experts: usize,
    pub n_slots: usize,
}

/// What [`KquantTileKernels::enqueue_card_gather`] reads: the block's q8_1
/// activation by token (`cols` columns) and its table, `slots_per_col`
/// slots a token.
pub struct TileGather<'a> {
    pub x: &'a Q8Act,
    pub cols: usize,
    pub slots_per_col: usize,
    pub table: TileTable<'a>,
}

/// The Walk A planes of a table's entries by entry, one column an entry:
/// the q4 codes, the 32-value sums and the 128-value scales, the layout a
/// [`Q8Act`]'s three Walk A planes take.
pub struct EntryPlanes {
    q4: DeviceBuffer<u32>,
    s8: DeviceBuffer<i32>,
    d8: DeviceBuffer<f32>,
    cols: usize,
    k: usize,
}

impl EntryPlanes {
    /// Planes for `cols` entries of `k` values (a positive multiple of 256).
    /// Load-time only.
    pub fn new(stream: &CudaStream, cols: usize, k: usize) -> Result<EntryPlanes, GpuError> {
        if cols == 0 || k == 0 || !k.is_multiple_of(256) {
            return Err(GpuError::shape(
                "EntryPlanes::new",
                format!(
                    "at least one entry of a positive multiple of 256 values, got {cols} of {k}"
                ),
            ));
        }
        let n_sb = k / 256;
        Ok(EntryPlanes {
            q4: DeviceBuffer::zeroed(stream, cols * 256 * n_sb.div_ceil(4))?,
            s8: DeviceBuffer::zeroed(stream, cols * 8 * n_sb)?,
            d8: DeviceBuffer::zeroed(stream, cols * 2 * n_sb)?,
            cols,
            k,
        })
    }

    /// The planes' device bytes: `4 · cols · (256 · ⌈n_sb / 4⌉ + 10 · n_sb)`.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.q4.num_bytes() + self.s8.num_bytes() + self.d8.num_bytes()
    }

    /// Entries the planes hold.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Values an entry.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }
}

/// What [`KquantTileKernels::enqueue_gate_up_tiles`] reads: both
/// stacks of the table's `n_experts` experts of `rows_per_expert` rows, the
/// entries' gathered planes and the table; `rule` the activation.
pub struct TiledGateUpAct<'a> {
    pub wg: &'a DeviceTensor<u32>,
    pub wu: &'a DeviceTensor<u32>,
    pub act: &'a EntryPlanes,
    pub table: TileTable<'a>,
    pub rows_per_expert: usize,
    pub rule: Act,
}

/// What [`KquantTileKernels::enqueue_gemv_q5k_tiles`] reads: the Q5_K stack
/// of the table's experts, the entries' q8_1 columns (column `j` the
/// activation of entry `j`, `q4k_sel::q8_1_quantize_ord`) and the table.
pub struct TiledDown<'a> {
    pub w: &'a DeviceTensor<u32>,
    pub act: &'a Q8Act,
    pub table: TileTable<'a>,
    pub rows_per_expert: usize,
}

/// The loaded grouped module and its launchers. Owns no stream: every
/// enqueue takes the engine stream, so launches order with the step and are
/// capturable.
pub struct KquantTileKernels {
    module: kquant_tile_kernels::LoadedModule,
    /// The fault word of the `Gpu` that owns the context: the launches'
    /// sinks point into it, and the module keeps it alive.
    _fault: Arc<DeviceBuffer<u32>>,
}

/// A tile launch's checked grid: the table's [`tile_cap`] and the stack's
/// row tiles, and their product in blocks.
struct TileGrid {
    blocks: u32,
    tile_cap: u32,
    row_tiles: u32,
    n_experts: u32,
    n_slots: u32,
}

/// `table` checked for a stack `w` of `words · n_sb`-word rows in experts of
/// `rows_per_expert` rows (a multiple of eight), one expert a run of the
/// table, or the shape error naming `what`.
fn tile_grid(
    what: &'static str,
    words: usize,
    w: &DeviceTensor<u32>,
    n_sb: usize,
    rows_per_expert: usize,
    table: &TileTable<'_>,
) -> Result<TileGrid, GpuError> {
    if w.cols() != words * n_sb
        || rows_per_expert == 0
        || !rows_per_expert.is_multiple_of(ROWS_PER_BLOCK)
        || !w.rows().is_multiple_of(rows_per_expert)
        || w.rows() / rows_per_expert != table.n_experts
    {
        return Err(GpuError::shape(
            what,
            format!(
                "{} experts of rows of {words}*{n_sb} words in experts of {rows_per_expert} rows \
                 (a multiple of {ROWS_PER_BLOCK}), got {} x {}",
                table.n_experts,
                w.rows(),
                w.cols()
            ),
        ));
    }
    let cap = tile_cap(table.n_slots, table.n_experts);
    if cap == 0
        || table.order.len() < table.n_slots
        || table.start.len() < table.n_experts + 1
        || table.tiles.len() < cap + 1
    {
        return Err(GpuError::shape(
            what,
            format!(
                "a table of {} slots over {} experts: order {}, start {}, tiles {} for {cap} tiles",
                table.n_slots,
                table.n_experts,
                table.order.len(),
                table.start.len(),
                table.tiles.len()
            ),
        ));
    }
    let row_tiles = rows_per_expert / ROWS_PER_BLOCK;
    Ok(TileGrid {
        blocks: launch_u32(what, "grid", cap * row_tiles)?,
        tile_cap: launch_u32(what, "tile_cap", cap)?,
        row_tiles: launch_u32(what, "row_tiles", row_tiles)?,
        n_experts: launch_u32(what, "n_experts", table.n_experts)?,
        n_slots: launch_u32(what, "n_slots", table.n_slots)?,
    })
}

impl KquantTileKernels {
    /// Load this module's device bundle into `ctx`, whose launches raise into
    /// `word`, the fault word of the `Gpu` that owns `ctx`
    /// ([`crate::Gpu::fault_word`]); a word of another context is refused.
    /// Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        word: &Arc<DeviceBuffer<u32>>,
    ) -> Result<KquantTileKernels, GpuError> {
        let fault = crate::module_fault_word(ctx, word, "KquantTileKernels::load")?;
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { kquant_tile_kernels::load(ctx)? };
        Ok(KquantTileKernels {
            module,
            _fault: fault,
        })
    }

    /// Enqueue [`kq_card_gather`]: the Walk A planes of the table's entries,
    /// entry `j` the token column `order[j] / slots_per_col` of `g.x`, into
    /// `out`'s columns `0 .. n_slots`; a launch of one block an entry.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_card_gather(
        &self,
        stream: &CudaStream,
        g: &TileGather<'_>,
        fault: FaultSink,
        out: &mut EntryPlanes,
    ) -> Result<(), GpuError> {
        let what = "enqueue_card_gather";
        let (k, n_sb) = (g.x.k(), g.x.n_sb());
        let t = &g.table;
        if out.k != k
            || out.cols < t.n_slots
            || t.n_slots == 0
            || g.slots_per_col == 0
            || g.cols == 0
            || g.cols > g.x.m()
            || t.n_slots > g.cols * g.slots_per_col
            || t.order.len() < t.n_slots
            || t.start.len() < t.n_experts + 1
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} entries at {} slots a token over {} of {} columns of {k} values into {} \
                     entries of {}: order {}, start {} for {} experts",
                    t.n_slots,
                    g.slots_per_col,
                    g.cols,
                    g.x.m(),
                    out.cols,
                    out.k,
                    t.order.len(),
                    t.start.len(),
                    t.n_experts
                ),
            ));
        }
        let cfg = LaunchConfig1D::new(launch_u32(what, "grid", t.n_slots)?, THREADS, 0);
        let prep = self.module.prepare_kq_card_gather(cfg)?;
        let (q, s8, d8) = super::walk_a_planes(g.x);
        self.module.kq_card_gather(
            stream,
            &prep,
            q,
            s8,
            d8,
            t.order,
            t.start,
            launch_u32(what, "n_experts", t.n_experts)?,
            launch_u32(what, "n_slots", t.n_slots)?,
            launch_u32(what, "cols", g.cols)?,
            launch_u32(what, "slots_per_col", g.slots_per_col)?,
            launch_u32(what, "n_sb", n_sb)?,
            launch_u32(what, "iters", n_sb.div_ceil(4))?,
            fault,
            &mut out.q4,
            &mut out.s8,
            &mut out.d8,
        )?;
        Ok(())
    }

    /// Enqueue the grouped gate·up of `a.wg`'s format — Q4_K
    /// (`kq_gate_up_act_q4k_tiles`) or Q5_K (`kq_gate_up_act_q5k_tiles`),
    /// any other refused by name: entry `j` writes `h[j · rows_per_expert
    /// ..][..rows_per_expert]`, the `_sel` gate·up's rows for its slot. The
    /// tile count is the table's, a device value: the grid is sized for
    /// [`tile_cap`] of them and a block past the count returns.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_gate_up_tiles(
        &self,
        stream: &CudaStream,
        ty: gguf::quant::GgmlType,
        a: &TiledGateUpAct<'_>,
        fault: FaultSink,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_gate_up_tiles";
        let words = match ty {
            gguf::quant::GgmlType::Q4_K => Q4k::WORDS,
            gguf::quant::GgmlType::Q5_K => Q5k::WORDS,
            other => {
                return Err(GpuError::shape(
                    what,
                    format!("a {other:?} gate·up stack; the grouped gate·up runs Q4_K and Q5_K"),
                ));
            }
        };
        let n_sb = a.act.k / 256;
        let gr = tile_grid(what, words, a.wg, n_sb, a.rows_per_expert, &a.table)?;
        let rpe = a.rows_per_expert;
        if a.wu.rows() != a.wg.rows()
            || a.wu.cols() != a.wg.cols()
            || a.act.cols < a.table.n_slots
            || h.len() < a.table.n_slots * rpe
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "the up stack {} x {} beside the gate's {} x {}; {} entries of {} planes; \
                     h.len() {} >= n_slots*rows_per_expert = {}",
                    a.wu.rows(),
                    a.wu.cols(),
                    a.wg.rows(),
                    a.wg.cols(),
                    a.table.n_slots,
                    a.act.cols,
                    h.len(),
                    a.table.n_slots * rpe
                ),
            ));
        }
        if let Act::SwigluClamp { limit } = a.rule
            && !limit.is_finite()
        {
            return Err(GpuError::shape(
                what,
                format!("the clamp limit must be finite, got {limit}"),
            ));
        }
        let (act, limit) = a.rule.code();
        let cfg = LaunchConfig1D::new(gr.blocks, THREADS, 0);
        let p = &a.act;
        macro_rules! launch {
            ($prepare:ident, $entry:ident) => {{
                let prep = self.module.$prepare(cfg)?;
                self.module.$entry(
                    stream,
                    &prep,
                    a.wg.buf(),
                    a.wu.buf(),
                    &p.q4,
                    &p.s8,
                    &p.d8,
                    a.table.start,
                    a.table.tiles,
                    gr.tile_cap,
                    gr.row_tiles,
                    gr.n_experts,
                    launch_u32(what, "rows_per_expert", rpe)?,
                    gr.n_slots,
                    launch_u32(what, "n_sb", n_sb)?,
                    launch_u32(what, "iters", n_sb.div_ceil(4))?,
                    act,
                    limit,
                    fault,
                    h,
                )?;
            }};
        }
        if words == Q4k::WORDS {
            launch!(prepare_kq_gate_up_act_q4k_tiles, kq_gate_up_act_q4k_tiles);
        } else {
            launch!(prepare_kq_gate_up_act_q5k_tiles, kq_gate_up_act_q5k_tiles);
        }
        Ok(())
    }

    /// Enqueue [`q5k_gemv_tiles`]: entry `j` reads column `j` of `a.act` and
    /// writes its slot `order[j]`'s rows `y[order[j] · rows_per_expert ..]`,
    /// `q5k_gemv_sel`'s value for that slot. The tile count is the table's,
    /// a device value: the grid is sized for [`tile_cap`] of them and a block
    /// past the count returns. Asynchronous, allocation-free, capturable.
    pub fn enqueue_gemv_q5k_tiles(
        &self,
        stream: &CudaStream,
        a: &TiledDown<'_>,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_gemv_q5k_tiles";
        let n_sb = a.act.n_sb();
        let gr = tile_grid(what, Q5k::WORDS, a.w, n_sb, a.rows_per_expert, &a.table)?;
        let rpe = a.rows_per_expert;
        if a.table.n_slots > a.act.m() || y.len() < a.table.n_slots * rpe {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} entries over {} columns, y.len() {} >= n_slots*rows_per_expert = {}",
                    a.table.n_slots,
                    a.act.m(),
                    y.len(),
                    a.table.n_slots * rpe
                ),
            ));
        }
        let prep = self
            .module
            .prepare_q5k_gemv_tiles(LaunchConfig1D::new(gr.blocks, THREADS, 0))?;
        let (q, s8, d8) = super::walk_a_planes(a.act);
        self.module.q5k_gemv_tiles(
            stream,
            &prep,
            a.w.buf(),
            q,
            s8,
            d8,
            a.table.order,
            a.table.start,
            a.table.tiles,
            gr.tile_cap,
            gr.row_tiles,
            gr.n_experts,
            launch_u32(what, "rows_per_expert", rpe)?,
            gr.n_slots,
            launch_u32(what, "n_sb", n_sb)?,
            launch_u32(what, "iters", n_sb.div_ceil(4))?,
            fault,
            y,
        )?;
        Ok(())
    }
}
