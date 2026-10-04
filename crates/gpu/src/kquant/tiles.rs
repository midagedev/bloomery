//! The family's grouped entries: a block's card slots by tile items — an
//! expert, a tile of eight of its weight rows, up to [`TILE_COLS`] of its
//! slots — over a grouped slot table (`q4k_sel`'s module doc: `order`, runs
//! `start`, tiles from `q4k_sel::grouped_tiles`), each weight row read once
//! for up to eight slots. Every slot's value is its `_sel` entry's bit for
//! bit ([`super::sel`]): column `c` of Walk A's m-column core
//! ([`walk::row_dot`]) is the one-column walk on that column, and the warp
//! tree, the rule and the store are the `_sel` bodies'.
//!
//! - [`gather`] over [`GatherPlanes`] ([`gather_plane`] a plane): the
//!   gather of a table's entries, one block an entry, whatever planes a walk
//!   reads; every grouped path's gather entry is this body.
//! - [`kq_card_gather`]: the q8_1 planes Walk A reads (q4, s8, d8) of a
//!   block's token columns by table entry, column `j` the token of slot
//!   `order[j]`, `slots_per_col` slots a token.
//! - `kq_gate_up_act_*_tiles`: the gate·up with its rule as a launch
//!   argument over the gathered entries, into the rule's outputs by entry —
//!   [`super::sel::gate_up_act_body`]'s value for the entry's slot.
//! - [`GemvTiles`]: the down over the entries' q8_1 columns
//!   (`q4k_sel::q8_1_quantize_ord`), each value scattered to its slot's rows
//!   — [`super::sel::gemv_sel_body`]'s value for that slot; its entries are
//!   [`q5k_gemv_tiles`] here and `q4k_sel::q4k_gemv_tiles`, the Q4_K down.
//!
//! Every tile entry runs its work ([`TileWork`]) under `q4k_sel::tile_block`,
//! the one prologue that reads a block's tile.
//!
//! A tile word that names no run of the table raises [`FaultSite::ExpertId`]
//! (`q4k_sel::tile_at`), as does an entry that names no slot; a block past
//! the table's count returns before any load.

use super::act::Act;
use super::q5k::Q5k;
use super::walk::{Q4k, SbDecode, row_dot};
use crate::fault::{FaultSink, FaultSite};
use crate::q4k_sel::{
    BUCKET_EXPERTS, TILE_COLS, TileBlock, TileWork, scatter_cols, tile_block, tile_cap,
};
use crate::tensor::{DeviceTensor, Q8Act};
use crate::{Gpu, GpuError, col_sums, launch_u32, store_cols};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::marker::PhantomData;
use std::sync::Arc;

/// Threads per block of every entry here, one warp per weight row.
const THREADS: u32 = 256;
// The entries' launch attributes spell the block as a literal.
const _: () = assert!(THREADS == 256);

/// Weight rows a tile block computes: one per warp.
const ROWS_PER_BLOCK: usize = THREADS as usize / 32;

// The m-column core takes the tile's columns at most.
const _: () = assert!(TILE_COLS == 8);

/// A grouped gate·up entry's work over format `D` (module doc,
/// [`tile_block`]): warp `w`'s weight row `r` of the tile's expert in both
/// stacks, dotted with columns `j .. j + m` of the gathered planes through
/// [`row_dot`] and reduced with the warp tree; lane 0 stores `h[(j + c) ·
/// rows_per_expert + r] = act::apply(act, limit, g, u)`. Its launch
/// contract: both stacks hold `n_experts · rows_per_expert` rows of
/// `D::WORDS · n_sb` words, the planes hold `n_slots` columns, `h.len() >=
/// n_slots · rows_per_expert`; `iters = ceil(n_sb / 4)`.
struct GateUpTiles<'a, D> {
    wg: &'a [u32],
    wu: &'a [u32],
    q: &'a [u32],
    s8: &'a [i32],
    d8: &'a [f32],
    rows_per_expert: u32,
    n_sb: u32,
    iters: u32,
    act: u32,
    limit: f32,
    h: DisjointSlice<'a, f32>,
    format: PhantomData<D>,
}

impl<D: SbDecode> TileWork for GateUpTiles<'_, D> {
    #[inline(always)]
    unsafe fn run(&mut self, t: TileBlock) {
        let TileBlock { e, j, m, r, lane } = t;
        let rpe = self.rows_per_expert as usize;
        let row_abs = e * rpe + r;
        let (wg, wu, q, s8, d8) = (self.wg, self.wu, self.q, self.s8, self.d8);
        let (n_sb, iters) = (self.n_sb as usize, self.iters);
        // SAFETY: row_abs < n_experts · rpe rows of both stacks (e <
        // n_experts, r < rpe); columns j + m <= n_slots of the planes; 1 <= m
        // <= 8 and block-uniform; iters = ceil(n_sb/4); the warp's 32 lanes
        // are here: this fn's contract and the work's.
        let (fg, fu) = unsafe {
            (
                row_dot::<D>(wg, q, s8, d8, n_sb, iters, row_abs, j, m, lane),
                row_dot::<D>(wu, q, s8, d8, n_sb, iters, row_abs, j, m, lane),
            )
        };
        let g = col_sums(fg, m);
        let u = col_sums(fu, m);
        if lane == 0 {
            let (act, limit) = (self.act, self.limit);
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
            // SAFETY: values (j + c)·rpe + r for c < m are below n_slots·rpe
            // <= h.len(); a table entry sits in one tile, so lane 0 of this
            // row's warp is their only writer.
            unsafe { store_cols(&mut self.h, j * rpe + r, rpe, m, v) };
        }
    }
}

/// A grouped down entry's work over format `D` (module doc, [`tile_block`]):
/// warp `w`'s weight row `r` of the tile's expert, dotted with columns `j ..
/// j + m` of the entries' q8_1 planes through [`row_dot`] and reduced with
/// the warp tree; lane 0 stores column `c` into `y[order[j + c] ·
/// rows_per_expert + r]` ([`scatter_cols`]), the `_sel` down's value for
/// that slot and row. Its launch contract: `w` holds `n_experts ·
/// rows_per_expert` rows of `D::WORDS · n_sb` words, the planes hold
/// `n_slots` columns, `order.len() >= n_slots`, `y.len() >= n_slots ·
/// rows_per_expert`; `iters = ceil(n_sb / 4)`.
pub(crate) struct GemvTiles<'a, D> {
    pub(crate) w: &'a [u32],
    pub(crate) q: &'a [u32],
    pub(crate) s8: &'a [i32],
    pub(crate) d8: &'a [f32],
    pub(crate) order: &'a [u32],
    pub(crate) rows_per_expert: u32,
    pub(crate) n_slots: u32,
    pub(crate) n_sb: u32,
    pub(crate) iters: u32,
    pub(crate) fault: FaultSink,
    pub(crate) y: DisjointSlice<'a, f32>,
    pub(crate) format: PhantomData<D>,
}

impl<D: SbDecode> TileWork for GemvTiles<'_, D> {
    #[inline(always)]
    unsafe fn run(&mut self, t: TileBlock) {
        let TileBlock { e, j, m, r, lane } = t;
        let rpe = self.rows_per_expert as usize;
        let (w, q, s8, d8) = (self.w, self.q, self.s8, self.d8);
        // SAFETY: row e·rpe + r < n_experts·rpe rows of `w` (e < n_experts, r
        // < rpe); columns j + m <= n_slots of the planes; 1 <= m <= 8
        // block-uniform; iters = ceil(n_sb/4); the warp's 32 lanes are here:
        // this fn's contract and the work's.
        let f = unsafe {
            row_dot::<D>(
                w,
                q,
                s8,
                d8,
                self.n_sb as usize,
                self.iters,
                e * rpe + r,
                j,
                m,
                lane,
            )
        };
        let v = col_sums(f, m);
        if lane == 0 {
            // SAFETY: j + m <= n_slots <= order.len(), r < rpe, y.len() >=
            // n_slots·rpe; a table entry sits in one tile, so lane 0 of this
            // warp is the only writer of its slot's row r.
            unsafe {
                scatter_cols(
                    &mut self.y,
                    self.order,
                    j,
                    m,
                    self.n_slots as usize,
                    rpe,
                    r,
                    v,
                    self.fault,
                );
            }
        }
    }
}

/// The planes a gather copies ([`gather`]), each a column a table entry.
pub trait GatherPlanes {
    /// Thread `tid`'s words of token column `col` of every input plane,
    /// copied to column `j` of its output ([`gather_plane`] a plane).
    ///
    /// # Safety
    ///
    /// Column `col` of every input and column `j` of every output are inside
    /// their buffers; the block is 256 threads, `tid` this thread's index,
    /// and block `j` the only writer of column `j`.
    unsafe fn copy(&mut self, col: usize, j: usize, tid: usize);
}

/// The gather of a grouped table's entries (`order`, runs `start` of
/// `n_experts` experts), one block an entry, whatever planes a walk reads:
/// block `j` copies token column `col0 + order[j] / slots_per_col` of the
/// block's columns `.. cols` to column `j` of `planes`' outputs, for each
/// entry `j` below the count `start[n_experts]` and `n_slots`. Entries from
/// the count on are left as they were; an entry that names no slot of the
/// block or a token past `cols` raises [`FaultSite::ExpertId`] on `fault`
/// and is skipped.
///
/// # Safety
///
/// `order.len() >= n_slots`, `start.len() >= n_experts + 1`,
/// `slots_per_col >= 1`; `planes` holds `cols` input columns and `n_slots`
/// output columns; one block of 256 threads an entry, every thread here.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: the table, the block's columns, their bounds and the planes (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn gather<P: GatherPlanes>(
    order: &[u32],
    start: &[u32],
    n_experts: u32,
    n_slots: u32,
    col0: u32,
    cols: u32,
    slots_per_col: usize,
    fault: FaultSink,
    planes: &mut P,
) {
    let j = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    // SAFETY: n_experts < start.len() by this fn's contract.
    let count = unsafe { *start.get_unchecked(n_experts as usize) } as usize;
    // Block-uniform: j, the count and the entry are the block's.
    if j >= n_slots as usize || j >= count {
        return;
    }
    // SAFETY: j < n_slots <= order.len() by this fn's contract.
    let slot = unsafe { *order.get_unchecked(j) } as usize;
    let col = col0 as usize + slot / slots_per_col;
    if slot >= n_slots as usize || col >= cols as usize {
        if tid == 0 {
            fault.raise(FaultSite::ExpertId);
        }
        return;
    }
    // SAFETY: col < cols and j < n_slots, inside the planes by this fn's
    // contract; block j writes column j alone.
    unsafe { planes.copy(col, j, tid) };
}

/// One plane of a gather's copy ([`GatherPlanes::copy`]): thread `tid`
/// copies words `tid, tid + 256, ..` below `words` of column `col` of `src`
/// (`words` a column) to column `j` of `dst`.
///
/// # Safety
///
/// `src.len() >= (col + 1) · words`, `dst.len() >= (j + 1) · words`, the
/// block is 256 threads and `tid` this thread's index, and block `j` is the
/// only writer of column `j`.
#[inline(always)]
pub unsafe fn gather_plane<T: Copy>(
    src: &[T],
    dst: &mut DisjointSlice<T>,
    words: usize,
    col: usize,
    j: usize,
    tid: usize,
) {
    let mut k = tid;
    while k < words {
        // SAFETY: k < words, so word k of column col of `src` and of column
        // j of `dst` are inside both by this fn's contract; thread tid of
        // block j is the only writer of word k of column j.
        unsafe { *dst.get_unchecked_mut(j * words + k) = *src.get_unchecked(col * words + k) };
        k += THREADS as usize;
    }
}

/// Walk A's planes of a gather ([`kq_card_gather`]): the q4, s8 and d8
/// planes of the block's tokens and of the table's entries, `iters` and
/// `n_sb` their columns' extents.
struct WalkAPlanes<'a> {
    q_in: &'a [u32],
    s8_in: &'a [i32],
    d8_in: &'a [f32],
    iters: u32,
    n_sb: u32,
    q_out: DisjointSlice<'a, u32>,
    s8_out: DisjointSlice<'a, i32>,
    d8_out: DisjointSlice<'a, f32>,
}

impl GatherPlanes for WalkAPlanes<'_> {
    #[inline(always)]
    unsafe fn copy(&mut self, col: usize, j: usize, tid: usize) {
        let (qc, sc, dc) = (
            256 * self.iters as usize,
            8 * self.n_sb as usize,
            2 * self.n_sb as usize,
        );
        // SAFETY: the three planes' columns are this fn's contract.
        unsafe {
            gather_plane(self.q_in, &mut self.q_out, qc, col, j, tid);
            gather_plane(self.s8_in, &mut self.s8_out, sc, col, j, tid);
            gather_plane(self.d8_in, &mut self.d8_out, dc, col, j, tid);
        }
    }
}

#[cuda_module]
mod kquant_tile_kernels {
    use super::*;

    /// The Walk A planes of a grouped table's entries ([`gather`] over
    /// [`WalkAPlanes`]): block `j` copies the q4, s8 and d8 words of token
    /// column `order[j] / slots_per_col` of `q_in`, `s8_in` and `d8_in`
    /// (`cols` columns) to column `j` of the outputs, for each entry `j`
    /// below the count `start[n_experts]` — a word copy, so column `j` is its
    /// slot's activation as the slot's gate·up `_sel` reads it. Entries from
    /// the count on are left as they were; an entry that names no slot of the
    /// block or a token past `cols` raises [`FaultSite::ExpertId`] on `fault`
    /// and is skipped.
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
        q_out: DisjointSlice<u32>,
        s8_out: DisjointSlice<i32>,
        d8_out: DisjointSlice<f32>,
    ) {
        let mut planes = WalkAPlanes {
            q_in,
            s8_in,
            d8_in,
            iters,
            n_sb,
            q_out,
            s8_out,
            d8_out,
        };
        // SAFETY: order.len() >= n_slots, start.len() >= n_experts + 1,
        // slots_per_col >= 1 and the planes' columns by the launch contract;
        // the block's threads are all here.
        unsafe {
            gather(
                order,
                start,
                n_experts,
                n_slots,
                0,
                cols,
                slots_per_col as usize,
                fault,
                &mut planes,
            );
        }
    }

    /// The Q4_K grouped gate·up ([`GateUpTiles`] over [`Q4k`]): every
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
        h: DisjointSlice<f32>,
    ) {
        let mut work = GateUpTiles::<Q4k> {
            wg,
            wu,
            q,
            s8,
            d8,
            rows_per_expert,
            n_sb,
            iters,
            act,
            limit,
            h,
            format: PhantomData,
        };
        // SAFETY: the launch contract is the work's (36 = Q4k::WORDS) and
        // `tile_block`'s; the host passes iters = ceil(n_sb/4) and one of
        // Act::code's codes.
        unsafe {
            tile_block(
                tiles, start, n_experts, n_slots, tile_cap, row_tiles, fault, &mut work,
            );
        }
    }

    /// The Q5_K grouped gate·up ([`GateUpTiles`] over [`Q5k`]): every
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
        h: DisjointSlice<f32>,
    ) {
        let mut work = GateUpTiles::<Q5k> {
            wg,
            wu,
            q,
            s8,
            d8,
            rows_per_expert,
            n_sb,
            iters,
            act,
            limit,
            h,
            format: PhantomData,
        };
        // SAFETY: the launch contract is the work's (44 = Q5k::WORDS) and
        // `tile_block`'s; the host passes iters = ceil(n_sb/4) and one of
        // Act::code's codes.
        unsafe {
            tile_block(
                tiles, start, n_experts, n_slots, tile_cap, row_tiles, fault, &mut work,
            );
        }
    }

    /// The Q5_K grouped down ([`GemvTiles`] over [`Q5k`]): every
    /// card slot's rows are `q5k_gemv_sel`'s for it.
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
        y: DisjointSlice<f32>,
    ) {
        let mut work = GemvTiles::<Q5k> {
            w,
            q,
            s8,
            d8,
            order,
            rows_per_expert,
            n_slots,
            n_sb,
            iters,
            fault,
            y,
            format: PhantomData,
        };
        // SAFETY: the launch contract is the work's (44 = Q5k::WORDS) and
        // `tile_block`'s; the host passes iters = ceil(n_sb/4).
        unsafe {
            tile_block(
                tiles, start, n_experts, n_slots, tile_cap, row_tiles, fault, &mut work,
            );
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

/// The tile path's scratch for blocks of up to `slots` slots through experts
/// of `ff` rows, but the planes its format's gather writes: the grouped slot
/// table — the slots by expert (`order`, runs `start` of up to
/// [`BUCKET_EXPERTS`] experts) and its tiles — the gate·up rows by entry
/// (`h`, `ff` an entry) and their q8_1 form (`act_h`), column `j` slot
/// `order[j]`'s. Every model's tile path holds one beside the planes its
/// gate·up reads; a block runs [`TileScratch::enqueue_table`], its format's
/// gather and gate·up into [`TileScratch::table_and_h`],
/// [`TileScratch::enqueue_quantize_h`] and its format's down over
/// [`TileScratch::down`].
pub struct TileScratch {
    order: DeviceBuffer<u32>,
    start: DeviceBuffer<u32>,
    tiles: DeviceBuffer<u32>,
    h: DeviceBuffer<f32>,
    act_h: Q8Act,
    slots: usize,
}

impl TileScratch {
    /// The scratch for blocks of up to `slots` slots through experts of `ff`
    /// rows, on `stream`'s card. Load-time only.
    pub fn new(stream: &CudaStream, slots: usize, ff: usize) -> Result<TileScratch, GpuError> {
        Ok(TileScratch {
            order: DeviceBuffer::zeroed(stream, slots)?,
            start: DeviceBuffer::zeroed(stream, BUCKET_EXPERTS + 1)?,
            tiles: DeviceBuffer::zeroed(stream, tile_cap(slots, BUCKET_EXPERTS) + 1)?,
            h: DeviceBuffer::zeroed(stream, slots * ff)?,
            act_h: Q8Act::with_slots(stream, slots, ff)?,
            slots,
        })
    }

    /// Device bytes of the scratch.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.order.num_bytes()
            + self.start.num_bytes()
            + self.tiles.num_bytes()
            + self.h.num_bytes()
            + self.act_h.device_bytes()
    }

    /// Slots a block takes at most.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Enqueue a block's table on `gpu`: the `n_slots` places `sel` (a slot
    /// of the `n_experts` card experts, or [`crate::hybrid::HOST`]) grouped
    /// by expert ([`crate::q4k_sel::Q4kSelKernels::enqueue_buckets`]), then
    /// its tiles ([`crate::q4k_sel::Q4kSelKernels::enqueue_grouped_tiles`]).
    /// A block of no slot or of more than the scratch takes is refused by
    /// name, as the two launchers refuse what they do not take. Two launches.
    /// Asynchronous, allocation-free.
    pub fn enqueue_table(
        &mut self,
        gpu: &Gpu,
        sel: &DeviceBuffer<u32>,
        n_slots: usize,
        n_experts: usize,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        if n_slots == 0 || n_slots > self.slots {
            return Err(GpuError::shape(
                "TileScratch::enqueue_table",
                format!(
                    "a block of {n_slots} slots; the scratch takes 1..={}",
                    self.slots
                ),
            ));
        }
        let (stream, k) = (gpu.stream(), gpu.q4k_sel());
        k.enqueue_buckets(
            stream,
            sel,
            n_slots,
            n_experts,
            fault,
            &mut self.order,
            &mut self.start,
        )?;
        k.enqueue_grouped_tiles(
            stream,
            &self.start,
            n_experts,
            n_slots,
            fault,
            &mut self.tiles,
        )
    }

    /// The table of `n_slots` slots over `n_experts` experts, as the last
    /// [`TileScratch::enqueue_table`] of those counts wrote it.
    #[must_use]
    pub fn table(&self, n_slots: usize, n_experts: usize) -> TileTable<'_> {
        TileTable {
            order: &self.order,
            start: &self.start,
            tiles: &self.tiles,
            n_experts,
            n_slots,
        }
    }

    /// [`TileScratch::table`] and the gate·up rows by entry, for the
    /// format's gate·up to write.
    pub fn table_and_h(
        &mut self,
        n_slots: usize,
        n_experts: usize,
    ) -> (TileTable<'_>, &mut DeviceBuffer<f32>) {
        let table = TileTable {
            order: &self.order,
            start: &self.start,
            tiles: &self.tiles,
            n_experts,
            n_slots,
        };
        (table, &mut self.h)
    }

    /// Enqueue the q8_1 form of the gate·up rows of the table's entries
    /// (`q4k_sel::q8_1_quantize_ord`, column `j` entry `j`'s), the down's
    /// input. One launch. Asynchronous, allocation-free.
    pub fn enqueue_quantize_h(
        &mut self,
        gpu: &Gpu,
        n_slots: usize,
        n_experts: usize,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        gpu.q4k_sel().enqueue_quantize_ord(
            gpu.stream(),
            &self.h,
            &self.start,
            n_experts,
            n_slots,
            fault,
            &mut self.act_h,
        )
    }

    /// The down of the table's entries through the stack `w` of experts of
    /// `rows_per_expert` rows, over the q8_1 form
    /// [`TileScratch::enqueue_quantize_h`] wrote: a format's down launcher
    /// takes it.
    #[must_use]
    pub fn down<'a>(
        &'a self,
        w: &'a DeviceTensor<u32>,
        rows_per_expert: usize,
        n_slots: usize,
        n_experts: usize,
    ) -> TiledDown<'a> {
        TiledDown {
            w,
            act: &self.act_h,
            table: self.table(n_slots, n_experts),
            rows_per_expert,
        }
    }
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
pub(crate) struct TileGrid {
    pub(crate) blocks: u32,
    pub(crate) tile_cap: u32,
    pub(crate) row_tiles: u32,
    pub(crate) n_experts: u32,
    pub(crate) n_slots: u32,
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

/// A grouped down's grid ([`tile_grid`] over `a`'s stack of `words` a
/// super-block) with its activation and output checked: a column an entry,
/// `y_len >= n_slots · rows_per_expert`; or the shape error naming `what`.
/// Every format's down launcher checks through it.
pub(crate) fn down_grid(
    what: &'static str,
    words: usize,
    a: &TiledDown<'_>,
    y_len: usize,
) -> Result<TileGrid, GpuError> {
    let gr = tile_grid(what, words, a.w, a.act.n_sb(), a.rows_per_expert, &a.table)?;
    if a.table.n_slots > a.act.m() || y_len < a.table.n_slots * a.rows_per_expert {
        return Err(GpuError::shape(
            what,
            format!(
                "{} entries over {} columns, y.len() {} >= n_slots*rows_per_expert = {}",
                a.table.n_slots,
                a.act.m(),
                y_len,
                a.table.n_slots * a.rows_per_expert
            ),
        ));
    }
    Ok(gr)
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
        let module = unsafe { crate::shared_module!(kquant_tile_kernels, ctx)? };
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
        let gr = down_grid(what, Q5k::WORDS, a, y.len())?;
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
            launch_u32(what, "rows_per_expert", a.rows_per_expert)?,
            gr.n_slots,
            launch_u32(what, "n_sb", n_sb)?,
            launch_u32(what, "iters", n_sb.div_ceil(4))?,
            fault,
            y,
        )?;
        Ok(())
    }
}
