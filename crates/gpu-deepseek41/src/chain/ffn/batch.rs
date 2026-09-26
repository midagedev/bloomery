//! The MoE sub-layer over a prompt batch: [`FfnBatch`], the buffers a batch
//! of up to its column count keeps between a layer's phases, and the piece's
//! batch enqueues. A layer runs, over the chunks of at most [`HC_MAX_TOKENS`]
//! consecutive tokens that hold its block — a suffix of the batch's tokens,
//! `at .. u`:
//!
//! 1. the route ([`FfnPiece::enqueue_batch_route`]): per chunk the norm with
//!    its q8_1 form — the step's `norm_quant`, whose f32 output is the
//!    router's input and the host's activation — then over the block the
//!    router (`RouterKernels::enqueue_router_rows`, each token's scores and
//!    selection the one-token launch's) into the batch's ids and weights, and
//!    each slot's place in the card's routed stacks (`ds41_ffn_places`, the
//!    handoff's rule, an id past the expert count raising its fault);
//!    the layer's tensors are resolved once for the route and the shadow
//!    ([`FfnPiece::resolve_batch`]);
//! 2. the exchange ([`FfnBatch::enqueue_download`], [`FfnBatch::serve`],
//!    [`FfnBatch::enqueue_upload`]): the block's activations and routing to
//!    the host, one union call over its tokens for the layer's host experts
//!    ([`Hybrid::serve_batch`]), the sums back;
//! 3. the shadow ([`FfnPiece::enqueue_batch_shadow`]), enqueued before the
//!    host computes: per chunk HC_PRE, the norm's q8_1 form again, the card's
//!    routed experts over the chunk's slots (the gate·up, the step's dot on
//!    column `slot / 6`; the q8_1 of the card slots' columns of `h`, the only
//!    columns the gate·up wrote; `q4k_sel` over the slots) and their sum
//!    (`ds41_ffn_card_acc`), the shared expert. By default the card's
//!    routed experts run over the whole block by tile items — an expert, a
//!    tile of its rows, up to eight of its slots — each weight row read
//!    once for the tile's slots (`ds41_card_buckets`, `ds41_card_gather`,
//!    `ds41_expert_gate_up_tiles`, `q4k_gemv_tiles`, between the chunks'
//!    parts before and after them, [`CardExperts::Tile`]); or each card
//!    expert's row walks its slots one at a time over the block
//!    (`ds41_expert_gate_up_grouped`, [`CardExperts::Expert`]); or per slot
//!    chunk by chunk (`ds41_expert_gate_up_tok`, [`CardExperts::Slot`]);
//! 4. the join ([`FfnPiece::enqueue_batch_join`]), one launch over the block:
//!    the card sum, the host sum and the shared expert combined, then HC_POST
//!    with the next fold where the layer folds.
//!
//! The feature tap of a batch's kept tokens is one launch here too
//! ([`FfnBatch::enqueue_tap_means`], `ds41_tap_means`).
//!
//! The launches that raise on input a batch's own route cannot produce — a
//! place for an id past the stack, a run table that does not fit its slots —
//! are [`FfnBatchKernels`]'s, which a gate loads alone and drives with it.
//!
//! Every token's values are the step's bit for bit: each launch writes, per
//! token, what its one-token launch writes — the m-column kernels carry that
//! contract, the routed dot reads its token's column through the step's
//! `q3k_row_dot`, and the combine is [`combine_elem`] cut at its one seam
//! ([`card_sum_elem`], then [`join_elem`]).

use std::mem::size_of;

use bloomery_gpu::cores::q3k_row_dot;
use bloomery_gpu::q4k_sel::{QuantSel, TILE_MAX_EXPERTS, grouped_run, tile_at, tile_cap};
use bloomery_gpu::{FaultSink, FaultSite, col_sums, store_cols};
use cuda_core::{CudaContext, CudaEvent, IntoResult, PinnedHostBuffer, sys};
use cuda_device::{SharedArray, warp};

use super::*;
use crate::chain::nanos;
use crate::experts::swiglu_clamp;
use crate::hc::hc_mean_elem;
use crate::span::{span, span_mut};

const WHAT: &str = "FfnBatch";

/// Threads per block of every kernel here but the bucket kernel.
const THREADS: u32 = 256;
// The kernels' launch attributes spell the block as a literal.
const _: () = assert!(THREADS == 256);

/// Weight rows a gate·up block computes: one per warp.
const ROWS_PER_BLOCK: usize = THREADS as usize / 32;

/// Threads of the bucket kernel's one block, and the most card experts a
/// layer's stack holds for it (its shared start table's width).
const BUCKET_THREADS: u32 = 1024;
const BUCKET_EXPERTS: usize = 1024;
// The count pass gives each expert a thread of its own, and the kernel's
// launch attributes and contract spell both as a literal.
const _: () = assert!(BUCKET_THREADS as usize >= BUCKET_EXPERTS);
const _: () = assert!(BUCKET_THREADS == 1024 && BUCKET_EXPERTS == 1024);
// A tile word names any expert the bucket kernel takes.
const _: () = assert!(BUCKET_EXPERTS <= TILE_MAX_EXPERTS);

#[cuda_module]
mod ffn_batch_kernels {
    use super::*;

    /// Each slot's place, the handoff's rule: thread `i < n` writes `sel[i] =
    /// map[row_off + ids[i]]` — the id's slot in the card's routed stacks, or
    /// [`HOST`]. An id not below `n_expert` has no place: it raises
    /// [`FaultSite::ExpertId`] on `fault` and its place is [`HOST`], so no
    /// card kernel reads a row for it.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (ids.len() >= n, map.len() >= row_off + n_expert, sel.len() >= n)
    )]
    pub fn ds41_ffn_places(
        ids: &[u32],
        map: &[u32],
        row_off: u32,
        n_expert: u32,
        n: u32,
        fault: FaultSink,
        mut sel: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= ids.len() by the launch contract.
        let id = unsafe { *ids.get_unchecked(i) };
        let place = if id < n_expert {
            // SAFETY: id < n_expert, so row_off + id < map.len() by the launch
            // contract.
            unsafe { *map.get_unchecked(row_off as usize + id as usize) }
        } else {
            fault.raise(FaultSite::ExpertId);
            HOST
        };
        // SAFETY: i < n <= sel.len(); thread i is sel[i]'s only writer.
        unsafe { *sel.get_unchecked_mut(i) = place };
    }

    /// The routed experts' gate·up·SwiGLU over the slots of `cols` tokens,
    /// six a token: `ds41_expert_gate_up` with slot `s` dotting column `s /
    /// 6` of the q8_1 activation — thread row `n = s · rows_per_expert + r`
    /// (a warp per row, 8 rows per block) reads weight row `sel[s] ·
    /// rows_per_expert + r` of both stacks, `cores::q3k_row_dot` on that one
    /// column, the warp tree, then `swiglu_clamp` into `h[n]`. A slot whose
    /// place is not below `n_experts` returns before any load and leaves its
    /// rows of `h` as they were: [`HOST`] is the host's slot, any other such
    /// place raises [`FaultSite::ExpertId`] on `fault`.
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
            4 * wg.len() >= n_experts * rows_per_expert * 110 * n_sb,
            4 * wu.len() >= n_experts * rows_per_expert * 110 * n_sb,
            q.len() >= 64 * iters * cols,
            d8.len() >= 2 * n_sb * cols,
            n_slots <= 6 * cols,
            sel.len() >= n_slots,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn ds41_expert_gate_up_tok(
        wg: &[u32],
        wu: &[u32],
        q: &[u64],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        cols: u32,
        n_sb: u32,
        iters: u32,
        limit: f32,
        fault: FaultSink,
        mut h: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * ROWS_PER_BLOCK + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        let slot = row / rows_per_expert as usize;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract. All 32
        // lanes of the warp share `row`, hence `slot`, `id` and `col`: the
        // returns below are warp-uniform.
        let id = unsafe { *sel.get_unchecked(slot) };
        if id >= n_experts {
            if id != HOST && t % 32 == 0 {
                fault.raise(FaultSite::ExpertId);
            }
            return;
        }
        let id = id as usize;
        let col = slot / N_USED;
        if col >= cols as usize {
            return;
        }
        let row_abs = id * rows_per_expert as usize + row % rows_per_expert as usize;
        let lane = warp::lane_id() as usize;
        let fg = q3k_row_dot(wg, q, d8, n_sb as usize, iters, row_abs, col, 1, lane);
        let fu = q3k_row_dot(wu, q, d8, n_sb as usize, iters, row_abs, col, 1, lane);
        let g = warp::reduce_sum_f32(fg[0]);
        let u = warp::reduce_sum_f32(fu[0]);
        if lane == 0 {
            let v = swiglu_clamp(g, u, limit);
            // SAFETY: row < n_slots * rows_per_expert <= h.len() by the
            // launch contract; lane 0 of the row's warp is its only writer.
            unsafe { *h.get_unchecked_mut(row) = v };
        }
    }

    /// The card slots of a block of `n_slots` slots grouped by expert, in one
    /// block of [`BUCKET_THREADS`]: `start[e] .. start[e + 1]` of `order` are
    /// the slots whose place `sel[s]` is `e`, in increasing `s`, for every
    /// place `e < n_experts`; `start[n_experts]` is the count of card slots.
    /// A place that is neither below `n_experts` nor [`HOST`] raises
    /// [`FaultSite::ExpertId`] on `fault` and is left out.
    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(
        domain = 1,
        block = (1024, 1, 1),
        requires = (
            sel.len() >= n_slots,
            order.len() >= n_slots,
            start.len() >= n_experts + 1,
            n_experts <= 1024
        )
    )]
    pub fn ds41_card_buckets(
        sel: &[u32],
        n_slots: u32,
        n_experts: u32,
        fault: FaultSink,
        mut order: DisjointSlice<u32>,
        mut start: DisjointSlice<u32>,
    ) {
        static mut COUNT: SharedArray<u32, BUCKET_EXPERTS> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let (n_slots, n_experts) = (n_slots as usize, n_experts as usize);
        // SAFETY: block-shared; the raw form reaches the `static mut`
        // without a reference.
        let count = unsafe { SharedArray::as_raw_mut_ptr(&raw mut COUNT) };
        let mut s = tid;
        while s < n_slots {
            // SAFETY: s < n_slots <= sel.len() by the launch contract.
            let p = unsafe { *sel.get_unchecked(s) };
            if p as usize >= n_experts && p != HOST {
                fault.raise(FaultSite::ExpertId);
            }
            s += BUCKET_THREADS as usize;
        }
        if tid < n_experts {
            let mut c = 0u32;
            let mut s = 0usize;
            while s < n_slots {
                // SAFETY: s < n_slots <= sel.len() by the launch contract.
                c += u32::from(unsafe { *sel.get_unchecked(s) } as usize == tid);
                s += 1;
            }
            // SAFETY: tid < n_experts <= BUCKET_EXPERTS; thread tid is the
            // entry's only writer before the barrier.
            unsafe { *count.add(tid) = c };
        }
        thread::sync_threads();
        if tid == 0 {
            let mut at = 0u32;
            let mut e = 0usize;
            while e < n_experts {
                // SAFETY: e < n_experts <= BUCKET_EXPERTS, published by the
                // barrier; thread 0 alone rewrites it now, before the next.
                let c = unsafe { *count.add(e) };
                // SAFETY: e < n_experts + 1 <= start.len(); thread 0 is the
                // only writer.
                unsafe {
                    *start.get_unchecked_mut(e) = at;
                    *count.add(e) = at;
                }
                at += c;
                e += 1;
            }
            // SAFETY: n_experts < start.len() by the launch contract.
            unsafe { *start.get_unchecked_mut(n_experts) = at };
        }
        thread::sync_threads();
        if tid < n_experts {
            // SAFETY: tid < n_experts, the entry thread 0 left before the
            // barrier.
            let mut at = unsafe { *count.add(tid) } as usize;
            let mut s = 0usize;
            while s < n_slots {
                // SAFETY: s < n_slots <= sel.len() by the launch contract.
                if unsafe { *sel.get_unchecked(s) } as usize == tid {
                    // SAFETY: at < start[tid + 1] <= n_slots <= order.len(); the
                    // runs of distinct places are disjoint, so thread tid is
                    // the only writer of its run.
                    unsafe { *order.get_unchecked_mut(at) = s as u32 };
                    at += 1;
                }
                s += 1;
            }
        }
    }

    /// [`ds41_expert_gate_up_tok`] with each card expert read once: block
    /// `b` is expert `e = b / ⌈rows_per_expert / 8⌉`, and warp `j` of it
    /// weight row `r = 8 (b mod ⌈rows_per_expert / 8⌉) + j` of that expert;
    /// the warp walks the expert's slots `order[start[e] .. start[e + 1]]`
    /// ([`ds41_card_buckets`]) and for slot `s` dots column `col0 + s / 6`
    /// of the q8_1 activation (`cols` columns, the block's from `col0`)
    /// through `cores::q3k_row_dot`, the warp tree and
    /// `swiglu_clamp` into `h[s · rows_per_expert + r]` — the per-slot
    /// kernel's value for that slot and row, bit for bit. A slot's weight row
    /// comes from memory once and from the cache for the expert's other slots.
    /// A run that does not fit the block's slots (`q4k_sel::grouped_run`, the
    /// grouped down's rule too) and an `order` entry that names no slot of the
    /// block raise [`FaultSite::ExpertId`] and are skipped.
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
            4 * wg.len() >= n_experts * rows_per_expert * 110 * n_sb,
            4 * wu.len() >= n_experts * rows_per_expert * 110 * n_sb,
            q.len() >= 64 * iters * cols,
            d8.len() >= 2 * n_sb * cols,
            n_slots + 6 * col0 <= 6 * cols,
            order.len() >= n_slots,
            start.len() >= n_experts + 1,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn ds41_expert_gate_up_grouped(
        wg: &[u32],
        wu: &[u32],
        q: &[u64],
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
        limit: f32,
        fault: FaultSink,
        mut h: DisjointSlice<f32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let rpe = rows_per_expert as usize;
        let tiles = rpe.div_ceil(ROWS_PER_BLOCK);
        let b = thread::blockIdx_x() as usize;
        let (e, r) = (b / tiles, (b % tiles) * ROWS_PER_BLOCK + tid / 32);
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
            // launch contract.
            let slot = unsafe { *order.get_unchecked(j) } as usize;
            let col = col0 as usize + slot / N_USED;
            if slot >= n_slots as usize || col >= cols as usize {
                if lane == 0 {
                    fault.raise(FaultSite::ExpertId);
                }
            } else {
                let fg = q3k_row_dot(wg, q, d8, n_sb as usize, iters, row_abs, col, 1, lane);
                let fu = q3k_row_dot(wu, q, d8, n_sb as usize, iters, row_abs, col, 1, lane);
                let g = warp::reduce_sum_f32(fg[0]);
                let u = warp::reduce_sum_f32(fu[0]);
                if lane == 0 {
                    // SAFETY: slot < n_slots and r < rows_per_expert, so the
                    // value is below n_slots * rows_per_expert <= h.len();
                    // each slot sits in one expert's run, so lane 0 of this
                    // warp is its only writer.
                    unsafe { *h.get_unchecked_mut(slot * rpe + r) = swiglu_clamp(g, u, limit) };
                }
            }
            j += 1;
        }
    }

    /// The q8_1 planes of a grouped table's entries (`order`, runs `start`,
    /// [`ds41_card_buckets`]): block `j` copies token column `col0 + order[j]
    /// / 6` of `q_in` and `d8_in` (`cols` columns, the block's from `col0`)
    /// to column `j` of `q_out` and `d8_out`, for each entry `j` below the
    /// count `start[n_experts]` — a byte copy, so column `j` is its slot's
    /// activation as the per-slot gate·up reads it. Entries from the count on
    /// are left as they were; an entry that names no slot of the block raises
    /// [`FaultSite::ExpertId`] on `fault` and is skipped.
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
            q_in.len() >= 64 * iters * cols,
            d8_in.len() >= 2 * n_sb * cols,
            order.len() >= n_slots,
            start.len() >= n_experts + 1,
            q_out.len() >= 64 * iters * n_slots,
            d8_out.len() >= 2 * n_sb * n_slots
        )
    )]
    pub fn ds41_card_gather(
        q_in: &[u64],
        d8_in: &[f32],
        order: &[u32],
        start: &[u32],
        n_experts: u32,
        n_slots: u32,
        col0: u32,
        cols: u32,
        n_sb: u32,
        iters: u32,
        fault: FaultSink,
        mut q_out: DisjointSlice<u64>,
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
        let col = col0 as usize + slot / N_USED;
        if slot >= n_slots as usize || col >= cols as usize {
            if tid == 0 {
                fault.raise(FaultSite::ExpertId);
            }
            return;
        }
        let (qc, dc) = (64 * iters as usize, 2 * n_sb as usize);
        let mut k = tid;
        while k < qc {
            // SAFETY: col < cols and j < n_slots with k < qc, inside both
            // buffers by the launch contract; thread tid of block j is the
            // only writer of value k of column j.
            unsafe { *q_out.get_unchecked_mut(j * qc + k) = *q_in.get_unchecked(col * qc + k) };
            k += THREADS as usize;
        }
        let mut k = tid;
        while k < dc {
            // SAFETY: as above, with dc values a column.
            unsafe { *d8_out.get_unchecked_mut(j * dc + k) = *d8_in.get_unchecked(col * dc + k) };
            k += THREADS as usize;
        }
    }

    /// [`ds41_expert_gate_up_grouped`] by tile items: block `g + tile_cap · ρ` (of
    /// `tile_cap · row_tiles`) is tile `g` of the table `tiles`
    /// (`q4k_sel::grouped_tiles`, `q4k_sel::tile_at`) — expert `e`, the `m`
    /// table entries from `j` — on the eight weight rows `8ρ ..`: warp `w`
    /// takes row `r = 8ρ + w` of both stacks and dots it with the `m` columns
    /// `j .. j + m` of the q8_1 planes — the table's entries
    /// ([`ds41_card_gather`]) — through the m-column core
    /// `cores::q3k_row_dot`, whose column c is the one-column call on that
    /// column bit for bit, then the warp tree and `swiglu_clamp` into `h[(j +
    /// c) · rows_per_expert + r]`: entry `j + c`'s value, the per-slot
    /// kernel's for its slot and row. The blocks of one row tile are
    /// consecutive, so an expert's tiles run side by side on the same weight
    /// rows, each read once for up to eight slots. A block past the table's
    /// count returns; a tile word `tile_at` refuses raises
    /// [`FaultSite::ExpertId`].
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    // Five blocks an SM: 48 registers a thread, the budget of the m-column
    // core over two stacks.
    #[launch_bounds(256, 5)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * wg.len() >= n_experts * rows_per_expert * 110 * n_sb,
            4 * wu.len() >= n_experts * rows_per_expert * 110 * n_sb,
            q.len() >= 64 * iters * n_slots,
            d8.len() >= 2 * n_sb * n_slots,
            start.len() >= n_experts + 1,
            tiles.len() >= tile_cap + 1,
            tile_cap >= 1,
            h.len() >= n_slots * rows_per_expert,
            rows_per_expert >= 8 * row_tiles
        )
    )]
    pub fn ds41_expert_gate_up_tiles(
        wg: &[u32],
        wu: &[u32],
        q: &[u64],
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
        limit: f32,
        fault: FaultSink,
        mut h: DisjointSlice<f32>,
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
        // both stacks (r < 8 · row_tiles <= rows_per_expert); columns j + m <=
        // n_slots of `q`/`d8` by `tile_at` and the launch contract; iters =
        // ceil(n_sb/2) from the host; the warp's 32 lanes are here and m is
        // block-uniform.
        let row_abs = e * rpe + r;
        let fg = q3k_row_dot(wg, q, d8, n_sb as usize, iters, row_abs, j, m, lane);
        let fu = q3k_row_dot(wu, q, d8, n_sb as usize, iters, row_abs, j, m, lane);
        let g = col_sums(fg, m);
        let u = col_sums(fu, m);
        if lane == 0 {
            let v = [
                swiglu_clamp(g[0], u[0], limit),
                swiglu_clamp(g[1], u[1], limit),
                swiglu_clamp(g[2], u[2], limit),
                swiglu_clamp(g[3], u[3], limit),
                swiglu_clamp(g[4], u[4], limit),
                swiglu_clamp(g[5], u[5], limit),
                swiglu_clamp(g[6], u[6], limit),
                swiglu_clamp(g[7], u[7], limit),
            ];
            // SAFETY: values (j + c)·rpe + r for c < m are below n_slots·rpe
            // <= h.len(); a table entry sits in one tile, so lane 0 of this
            // row's warp is their only writer.
            unsafe { store_cols(&mut h, j * rpe + r, rpe, m, v) };
        }
    }

    /// The card slots' sum of `m` tokens: thread `i < m·n` is token `t = i /
    /// n`, value `d = i % n`, and writes [`card_sum_elem`] of its six slots'
    /// down outputs `down[(6t + j)·n + d]`, weights `w[6t + j]` and places
    /// `sel[6t + j]` (the card's below `n_card`; no other slot's rows are
    /// read) to `acc[i]`.
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
            down.len() >= 6 * n * m,
            w.len() >= 6 * m,
            sel.len() >= 6 * m,
            acc.len() >= n * m
        )
    )]
    pub fn ds41_ffn_card_acc(
        down: &[f32],
        w: &[f32],
        sel: &[u32],
        n: u32,
        m: u32,
        n_card: u32,
        mut acc: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        let (t, d) = (i / n, i % n);
        let mut dv = [0.0f32; N_USED];
        let mut wv = [0.0f32; N_USED];
        let mut card = [false; N_USED];
        let mut j = 0usize;
        while j < N_USED {
            let s = t * N_USED + j;
            // SAFETY: s < 6m <= sel.len() and w.len() by the launch contract.
            let (place, ws) = unsafe { (*sel.get_unchecked(s), *w.get_unchecked(s)) };
            if place < n_card {
                card[j] = true;
                wv[j] = ws;
                // SAFETY: s < 6m and d < n, so s·n + d < 6nm <= down.len().
                dv[j] = unsafe { *down.get_unchecked(s * n + d) };
            }
            j += 1;
        }
        // SAFETY: i < n·m <= acc.len(); thread i is acc[i]'s only writer.
        unsafe { *acc.get_unchecked_mut(i) = card_sum_elem(dv, wv, card) };
    }

    /// The join of `m` tokens: thread `i < m·n` (token `t = i / n`, value `d
    /// = i % n`) combines `y = join_elem(acc[i], hsum[i], shexp[i])`, then
    /// HC_POST of it (`hc_post_elem`) by token `t`'s HC_PRE result
    /// `hc[24t ..]` and its four streams `res[4nt + kn + d]`, into `out` in
    /// the same layout, and their fold by the result's `pre` into `fold[i]`.
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
            acc.len() >= n * m,
            hsum.len() >= n * m,
            shexp.len() >= n * m,
            res.len() >= 4 * n * m,
            hc.len() >= 24 * m,
            out.len() >= 4 * n * m,
            fold.len() >= n * m
        )
    )]
    pub fn ds41_ffn_post_batch(
        acc: &[f32],
        hsum: &[f32],
        shexp: &[f32],
        res: &[f32],
        hc: &[f32],
        n: u32,
        m: u32,
        mut out: DisjointSlice<f32>,
        mut fold: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        let a = JoinIn {
            acc,
            hsum,
            shexp,
            res,
            hc,
        };
        // SAFETY: i < n·m, and the launch contract gives every length the
        // helper's contract asks for.
        let (o, pre) = unsafe { join_post_at(&a, n, i) };
        let b = 4 * (i / n) * n + i % n;
        // SAFETY: b + 3n < 4n(t + 1) <= 4nm <= out.len(), i < nm <= fold.len(),
        // by the launch contract; thread i is the only writer of the four
        // stream values at b and of fold[i].
        unsafe {
            store4(&mut out, n, b, o);
            *fold.get_unchecked_mut(i) = hc_fold_elem(o, pre);
        }
    }

    /// [`ds41_ffn_post_batch`] without the fold: into an engram layer and
    /// after the last layer.
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
            acc.len() >= n * m,
            hsum.len() >= n * m,
            shexp.len() >= n * m,
            res.len() >= 4 * n * m,
            hc.len() >= 24 * m,
            out.len() >= 4 * n * m
        )
    )]
    pub fn ds41_ffn_post_batch_streams(
        acc: &[f32],
        hsum: &[f32],
        shexp: &[f32],
        res: &[f32],
        hc: &[f32],
        n: u32,
        m: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        let a = JoinIn {
            acc,
            hsum,
            shexp,
            res,
            hc,
        };
        // SAFETY: i < n·m, and the launch contract gives every length the
        // helper's contract asks for.
        let (o, _) = unsafe { join_post_at(&a, n, i) };
        let b = 4 * (i / n) * n + i % n;
        // SAFETY: b + 3n < 4nm <= out.len() by the launch contract; thread i
        // is the only writer of the four stream values at b.
        unsafe { store4(&mut out, n, b, o) };
    }

    /// The feature tap over `m` tokens: thread `i < m·n` is token `t = i /
    /// n`, value `d = i % n`, and writes the mean of token `t`'s four streams
    /// at `d` ([`hc_mean_elem`], the one-token tap's value) to `y[t · width +
    /// off + d]`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (s.len() >= 4 * n * m, width >= off + n, y.len() >= width * m)
    )]
    pub fn ds41_tap_means(
        s: &[f32],
        n: u32,
        m: u32,
        width: u32,
        off: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let (n, m) = (n as usize, m as usize);
        let i = thread::index_1d().get();
        if i >= n * m {
            return;
        }
        let (t, d) = (i / n, i % n);
        let b = 4 * t * n + d;
        // SAFETY: b + 3n < 4n(t + 1) <= 4nm <= s.len() by the launch contract.
        let o = unsafe {
            [
                *s.get_unchecked(b),
                *s.get_unchecked(b + n),
                *s.get_unchecked(b + 2 * n),
                *s.get_unchecked(b + 3 * n),
            ]
        };
        let at = t * width as usize + off as usize + d;
        // SAFETY: at < t·width + width <= m·width <= y.len() (off + n <=
        // width, launch contract); thread i is at's only writer.
        unsafe { *y.get_unchecked_mut(at) = hc_mean_elem(o) };
    }
}

/// The batch's kernels: loaded once by [`FfnBatch::new`], or alone by a gate
/// that drives the launches with inputs a batch's route cannot produce.
pub struct FfnBatchKernels {
    module: ffn_batch_kernels::LoadedModule,
}

/// What [`FfnBatchKernels::enqueue_places`] reads: `n` routed ids, and the
/// slot map's card copy with the layer's row at `row_off` (`n_expert` places
/// a row).
pub struct Places<'a> {
    pub ids: &'a DeviceBuffer<u32>,
    pub n: usize,
    pub map: &'a DeviceBuffer<u32>,
    pub row_off: usize,
    pub n_expert: usize,
}

/// What [`FfnBatchKernels::enqueue_gate_up_grouped`] reads: both Q3_K stacks
/// of `n_experts` card experts of `rows_per_expert` rows, the q8_1 planes of
/// `cols` token columns of `n_sb` super-blocks (the block's from `col0`), and
/// the `n_slots` slots of the block grouped by expert (`order`, runs
/// `start`); `limit` is the SwiGLU clamp.
pub struct GroupedGateUp<'a> {
    pub wg: &'a DeviceBuffer<u32>,
    pub wu: &'a DeviceBuffer<u32>,
    pub q3: &'a DeviceBuffer<u64>,
    pub d8: &'a DeviceBuffer<f32>,
    pub order: &'a DeviceBuffer<u32>,
    pub start: &'a DeviceBuffer<u32>,
    pub n_experts: usize,
    pub rows_per_expert: usize,
    pub n_slots: usize,
    pub col0: usize,
    pub cols: usize,
    pub n_sb: usize,
    pub limit: f32,
}

/// What [`FfnBatchKernels::enqueue_gate_up_tiles`] reads: both Q3_K stacks
/// of `n_experts` card experts of `rows_per_expert` rows, the q8_1 planes of
/// the table's `n_slots` entries of `n_sb` super-blocks (entry `j` the
/// activation of slot `order[j]`, [`ds41_card_gather`]), the table's runs
/// `start` and its tiles (`q4k_sel::grouped_tiles`); `limit` is the SwiGLU
/// clamp.
struct TiledGateUp<'a> {
    wg: &'a DeviceBuffer<u32>,
    wu: &'a DeviceBuffer<u32>,
    q3: &'a DeviceBuffer<u64>,
    d8: &'a DeviceBuffer<f32>,
    start: &'a DeviceBuffer<u32>,
    tiles: &'a DeviceBuffer<u32>,
    n_experts: usize,
    rows_per_expert: usize,
    n_slots: usize,
    n_sb: usize,
    limit: f32,
}

/// What [`FfnBatchKernels::enqueue_card_gather`] reads: the block's q8_1
/// planes by token (`q3`, `d8`, `cols` columns of `n_sb` super-blocks, the
/// block's tokens from `col0`) and the table of its `n_slots` slots
/// (`order`, runs `start` of `n_experts` experts).
struct CardGather<'a> {
    q3: &'a DeviceBuffer<u64>,
    d8: &'a DeviceBuffer<f32>,
    order: &'a DeviceBuffer<u32>,
    start: &'a DeviceBuffer<u32>,
    n_experts: usize,
    n_slots: usize,
    col0: usize,
    cols: usize,
    n_sb: usize,
}

impl FfnBatchKernels {
    /// Load the batch's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<FfnBatchKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; each launcher checks its launch contract.
        let module = unsafe { ffn_batch_kernels::load(ctx)? };
        Ok(FfnBatchKernels { module })
    }

    /// Enqueue `ds41_ffn_places`: `sel[i]` the place of `p.ids[i]` for `i <
    /// p.n`, [`HOST`] and [`FaultSite::ExpertId`] on `fault` for an id past
    /// the stack. One launch. Asynchronous, allocation-free.
    pub fn enqueue_places(
        &self,
        stream: &CudaStream,
        p: &Places<'_>,
        fault: FaultSink,
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let what = "ds41_ffn_places";
        let grid = launch_u32(what, "grid", p.n.div_ceil(THREADS as usize))?;
        let prep = self
            .module
            .prepare_ds41_ffn_places(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.ds41_ffn_places(
            stream,
            &prep,
            p.ids,
            p.map,
            launch_u32(what, "row_off", p.row_off)?,
            launch_u32(what, "n_expert", p.n_expert)?,
            launch_u32(what, "n", p.n)?,
            fault,
            sel,
        )?;
        Ok(())
    }

    /// Enqueue `ds41_expert_gate_up_grouped`: the SwiGLU outputs of the
    /// card's slots of `g`, each card expert read once, into `h` (`g.n_slots
    /// · g.rows_per_expert` values). One launch. Asynchronous,
    /// allocation-free.
    pub fn enqueue_gate_up_grouped(
        &self,
        stream: &CudaStream,
        g: &GroupedGateUp<'_>,
        fault: FaultSink,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "ds41_expert_gate_up_grouped";
        let grid = launch_u32(
            what,
            "grid",
            g.n_experts * g.rows_per_expert.div_ceil(ROWS_PER_BLOCK),
        )?;
        let prep = self
            .module
            .prepare_ds41_expert_gate_up_grouped(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.ds41_expert_gate_up_grouped(
            stream,
            &prep,
            g.wg,
            g.wu,
            g.q3,
            g.d8,
            g.order,
            g.start,
            launch_u32(what, "n_experts", g.n_experts)?,
            launch_u32(what, "rows_per_expert", g.rows_per_expert)?,
            launch_u32(what, "n_slots", g.n_slots)?,
            launch_u32(what, "col0", g.col0)?,
            launch_u32(what, "cols", g.cols)?,
            launch_u32(what, "n_sb", g.n_sb)?,
            launch_u32(what, "iters", g.n_sb.div_ceil(2))?,
            g.limit,
            fault,
            h,
        )?;
        Ok(())
    }

    /// Enqueue `ds41_card_buckets`: the table of the `n_slots` places `sel`
    /// grouped by their `n_experts` card experts, into `order` and `start`.
    /// One launch. Asynchronous, allocation-free.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher over the kernel's flat inputs (rust-quality R8)"
    )]
    fn enqueue_buckets(
        &self,
        stream: &CudaStream,
        sel: &DeviceBuffer<u32>,
        n_slots: usize,
        n_experts: usize,
        fault: FaultSink,
        order: &mut DeviceBuffer<u32>,
        start: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let what = "ds41_card_buckets";
        let prep =
            self.module
                .prepare_ds41_card_buckets(LaunchConfig1D::new(1, BUCKET_THREADS, 0))?;
        self.module.ds41_card_buckets(
            stream,
            &prep,
            sel,
            launch_u32(what, "n_slots", n_slots)?,
            launch_u32(what, "n_experts", n_experts)?,
            fault,
            order,
            start,
        )?;
        Ok(())
    }

    /// Enqueue `ds41_card_gather`: the q8_1 planes of `g`'s table entries
    /// into columns `0 .. g.n_slots` of `q3` and `d8`. One launch.
    /// Asynchronous, allocation-free.
    fn enqueue_card_gather(
        &self,
        stream: &CudaStream,
        g: &CardGather<'_>,
        fault: FaultSink,
        q3: &mut DeviceBuffer<u64>,
        d8: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "ds41_card_gather";
        let prep = self.module.prepare_ds41_card_gather(LaunchConfig1D::new(
            launch_u32(what, "grid", g.n_slots)?,
            THREADS,
            0,
        ))?;
        self.module.ds41_card_gather(
            stream,
            &prep,
            g.q3,
            g.d8,
            g.order,
            g.start,
            launch_u32(what, "n_experts", g.n_experts)?,
            launch_u32(what, "n_slots", g.n_slots)?,
            launch_u32(what, "col0", g.col0)?,
            launch_u32(what, "cols", g.cols)?,
            launch_u32(what, "n_sb", g.n_sb)?,
            launch_u32(what, "iters", g.n_sb.div_ceil(2))?,
            fault,
            q3,
            d8,
        )?;
        Ok(())
    }

    /// Enqueue `ds41_expert_gate_up_tiles`: the SwiGLU outputs of `g`'s
    /// table entries by tiles, entry `j`'s rows into `h[j ·
    /// g.rows_per_expert ..]`. One launch of `tile_cap · row_tiles`
    /// blocks; the tile count is the table's, a device value, and a block
    /// past it returns. Asynchronous, allocation-free.
    fn enqueue_gate_up_tiles(
        &self,
        stream: &CudaStream,
        g: &TiledGateUp<'_>,
        fault: FaultSink,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "ds41_expert_gate_up_tiles";
        if g.rows_per_expert == 0
            || !g.rows_per_expert.is_multiple_of(ROWS_PER_BLOCK)
            || tile_cap(g.n_slots, g.n_experts) == 0
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "{what} of {} slots over {} experts of {} rows: at least one tile, rows a \
                     positive multiple of {ROWS_PER_BLOCK}",
                    g.n_slots, g.n_experts, g.rows_per_expert
                ),
            });
        }
        let (cap, row_tiles) = (
            tile_cap(g.n_slots, g.n_experts),
            g.rows_per_expert / ROWS_PER_BLOCK,
        );
        let blocks = launch_u32(what, "grid", cap * row_tiles)?;
        let grid = (
            launch_u32(what, "tile_cap", cap)?,
            launch_u32(what, "row_tiles", row_tiles)?,
        );
        let prep = self
            .module
            .prepare_ds41_expert_gate_up_tiles(LaunchConfig1D::new(blocks, THREADS, 0))?;
        self.module.ds41_expert_gate_up_tiles(
            stream,
            &prep,
            g.wg,
            g.wu,
            g.q3,
            g.d8,
            g.start,
            g.tiles,
            grid.0,
            grid.1,
            launch_u32(what, "n_experts", g.n_experts)?,
            launch_u32(what, "rows_per_expert", g.rows_per_expert)?,
            launch_u32(what, "n_slots", g.n_slots)?,
            launch_u32(what, "n_sb", g.n_sb)?,
            launch_u32(what, "iters", g.n_sb.div_ceil(2))?,
            g.limit,
            fault,
            h,
        )?;
        Ok(())
    }
}

/// A layer's tensors, resolved once for a batch's route and shadow of it
/// ([`FfnPiece::resolve_batch`]).
pub struct BatchLayer<'w> {
    layer: usize,
    lw: LayerWeights<'w>,
}

/// What the batch join reads, `m` tokens token-major: `acc` the card sums,
/// `hsum` the host sums and `shexp` the shared expert's outputs (`n` a
/// token), `res` the streams the sub-layer read (`4n` a token) and `hc` its
/// HC_PRE results (24 a token).
struct JoinIn<'a> {
    acc: &'a [f32],
    hsum: &'a [f32],
    shexp: &'a [f32],
    res: &'a [f32],
    hc: &'a [f32],
}

/// Value `i` of the batch (token `i / n`): its combine ([`join_elem`]), then
/// HC_POST of it (`hc_post_elem`): the four new stream values and `pre`.
///
/// SAFETY: every buffer of `a` holds the token `i / n`: `acc`, `hsum`,
/// `shexp` more than `i` values, `res` at least `4n(i / n + 1)`, `hc` at
/// least `24(i / n + 1)`.
#[inline(always)]
unsafe fn join_post_at(a: &JoinIn<'_>, n: usize, i: usize) -> ([f32; 4], [f32; 4]) {
    let (t, d) = (i / n, i % n);
    let (b, h) = (4 * t * n + d, t * HC_MIX);
    // SAFETY: i < acc/hsum/shexp lengths; b + 3n < 4n(t + 1) <= res.len();
    // h + 23 < 24(t + 1) <= hc.len() — all by this fn's contract.
    let (ac, hs, sh, r, hc) = unsafe {
        let mut hc = [0.0f32; HC_MIX];
        let mut k = 0usize;
        while k < HC_MIX {
            hc[k] = *a.hc.get_unchecked(h + k);
            k += 1;
        }
        (
            *a.acc.get_unchecked(i),
            *a.hsum.get_unchecked(i),
            *a.shexp.get_unchecked(i),
            [
                *a.res.get_unchecked(b),
                *a.res.get_unchecked(b + n),
                *a.res.get_unchecked(b + 2 * n),
                *a.res.get_unchecked(b + 3 * n),
            ],
            hc,
        )
    };
    let pre = [hc[0], hc[1], hc[2], hc[3]];
    let post = [hc[4], hc[5], hc[6], hc[7]];
    let comb = [
        hc[8], hc[9], hc[10], hc[11], hc[12], hc[13], hc[14], hc[15], hc[16], hc[17], hc[18],
        hc[19], hc[20], hc[21], hc[22], hc[23],
    ];
    let y = join_elem(ac, hs, sh);
    (hc_post_elem(y, r, post, &comb), pre)
}

/// The buffers a prompt batch keeps between a layer's phases, for up to
/// `cap` tokens a batch and chunks of up to [`HC_MAX_TOKENS`]: allocated
/// once, when the first batch runs — a decode that never prefills a batch
/// never holds them.
///
/// A group of batches runs its layers in turn over each of its batches, one
/// batch's route enqueued while the host serves the batch before it. What a
/// layer of one batch leaves for the next layer of the same batch is kept
/// per batch of the group (`hc`, by `set`); what a layer's phases consume
/// before the next batch's launches write it is one buffer (the stream runs
/// them in order); and the host copies come in two sets ([`exchange`]),
/// since the host reads them outside the stream's order.
pub struct FfnBatch {
    kernels: FfnBatchKernels,
    cap: usize,
    n_embd: usize,
    ff: usize,
    /// Per token: the norm's f32 output (the router's input and the host's
    /// activation, `n_embd`), the routing (six ids and weights), each slot's
    /// place, the card sum, the shared expert's output and the host sum
    /// (`n_embd` each); per batch of a group, per token, the HC_PRE result,
    /// which the join, the next layer's engram step and the head read.
    x: DeviceBuffer<f32>,
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
    sel: DeviceBuffer<u32>,
    hc: Vec<DeviceBuffer<f32>>,
    acc: DeviceBuffer<f32>,
    shexp: DeviceBuffer<f32>,
    hsum: DeviceBuffer<f32>,
    /// Per token, the router's score of every expert.
    probs: DeviceBuffer<f32>,
    /// A chunk's scratch: the norm's
    /// f32 output a second time (discarded) and, per token count `m` (index
    /// `m − 1`), its q8_1 form; the HC_PRE mixes; the routed SwiGLU outputs
    /// (six slots a token), their q8_1 form per token count, the routed down
    /// outputs; the shared expert's SwiGLU output (token-major), its q8_1 form
    /// per token count and its down output row-major.
    normed: DeviceBuffer<f32>,
    act_x: Vec<Q8Act>,
    mixes: DeviceBuffer<f32>,
    h: DeviceBuffer<f32>,
    act_h: Vec<Q8Act>,
    down: DeviceBuffer<f32>,
    sh_h: DeviceBuffer<f32>,
    act_sh: Vec<Q8Act>,
    sh_raw: DeviceBuffer<f32>,
    /// The block-wide arms' buffers ([`CardExperts::Tile`],
    /// [`CardExperts::Expert`]): per token the norm's q8_1 planes the gate·up
    /// reads (q3 and d8, as the chunk's scratch lays out a column), per slot
    /// the SwiGLU output, its q8_1 form and the down output, and the block's
    /// card slots by expert (`order`, runs `start`). The tile arm keeps the
    /// SwiGLU output and its q8_1 form by table entry — column `j` is slot
    /// `order[j]`'s — gathers the norm's planes by entry into `q3_ord` and
    /// `d8_ord`, and lists the table's tiles in `tiles`.
    experts: CardExperts,
    q3_all: DeviceBuffer<u64>,
    d8_all: DeviceBuffer<f32>,
    h_all: DeviceBuffer<f32>,
    act_h_all: Q8Act,
    down_all: DeviceBuffer<f32>,
    order: DeviceBuffer<u32>,
    start: DeviceBuffer<u32>,
    q3_ord: DeviceBuffer<u64>,
    d8_ord: DeviceBuffer<f32>,
    tiles: DeviceBuffer<u32>,
    /// The exchange: page-locked copies of the activations, the routing and
    /// the host sums, the union reading the activations in place, in two
    /// sets, each with the event its route's copies complete at.
    exchange: exchange::Exchange,
}

/// The host exchange ([`FfnBatch::enqueue_download`], [`FfnBatch::serve`],
/// [`FfnBatch::enqueue_upload`]) in two sets, taken in turn: a layer's
/// route copies into one set while the union still reads the other, and the
/// union writes its sums into its own set while the other's upload may not
/// have run yet.
///
/// Each set owns the event its download records, and a serve waits on the
/// event of the set whose buffers it reads: a serve that waited on a shared
/// event would wait for the next batch's route too — the same bits with the
/// overlap gone. Nothing outside this module names a set's event.
mod exchange {
    use std::sync::Arc;
    use std::time::Instant;

    use super::{
        CudaContext, CudaEvent, CudaStream, DeviceBuffer, ExchangeKey, GpuError, HostExperts,
        Hybrid, N_USED, PinnedHostBuffer, ServeTimes, Tensor2View, WHAT, dtoh, htod, nanos,
    };

    /// Where a set stands: free, holding a layer-batch's route copies, or
    /// holding its host sums before their upload.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Stage {
        Free,
        Routed(ExchangeKey),
        Served(ExchangeKey),
    }

    /// One set: page-locked copies of a block's activations, routing and host
    /// sums, the event its route's copies complete at, and its stage.
    struct Set {
        x: PinnedHostBuffer<f32>,
        ids: PinnedHostBuffer<u32>,
        w: PinnedHostBuffer<f32>,
        sum: PinnedHostBuffer<f32>,
        routed: CudaEvent,
        stage: Stage,
    }

    /// The two sets, and the set the next download, serve and upload take:
    /// each of the three goes through the sets in turn, so a serve takes the
    /// set of the oldest download not served yet.
    pub(super) struct Exchange {
        sets: [Set; 2],
        n_embd: usize,
        down: usize,
        serve: usize,
        up: usize,
    }

    impl Exchange {
        /// Two sets for blocks of up to `cap` tokens of rows of `n_embd`.
        pub(super) fn new(
            ctx: &Arc<CudaContext>,
            n_embd: usize,
            cap: usize,
        ) -> Result<Exchange, GpuError> {
            let pinned = |what: &'static str, n: usize| {
                PinnedHostBuffer::<f32>::zeroed(ctx, n).map_err(|source| GpuError::Driver {
                    op: Some(what),
                    source,
                })
            };
            let set = || -> Result<Set, GpuError> {
                Ok(Set {
                    x: pinned("cuMemAllocHost (the batch's activations)", cap * n_embd)?,
                    ids: PinnedHostBuffer::<u32>::zeroed(ctx, cap * N_USED).map_err(|source| {
                        GpuError::Driver {
                            op: Some("cuMemAllocHost (the batch's routing)"),
                            source,
                        }
                    })?,
                    w: pinned("cuMemAllocHost (the batch's routing)", cap * N_USED)?,
                    sum: pinned("cuMemAllocHost (the batch's host sums)", cap * n_embd)?,
                    routed: ctx.new_event(None)?,
                    stage: Stage::Free,
                })
            };
            Ok(Exchange {
                sets: [set()?, set()?],
                n_embd,
                down: 0,
                serve: 0,
                up: 0,
            })
        }

        /// Both sets free, the next download into the first: at a group's
        /// start, when the stream holds no copy of an earlier group (the
        /// group's prologue waits for the stream).
        pub(super) fn begin(&mut self) {
            for s in &mut self.sets {
                s.stage = Stage::Free;
            }
            (self.down, self.serve, self.up) = (0, 0, 0);
        }

        /// Enqueue the copies of `key`'s tokens `at .. u` of `x` (`n_embd` a
        /// token), `ids` and `w` (six a token) into the next set, and its
        /// event. Refused while that set holds a layer not uploaded yet.
        pub(super) fn download(
            &mut self,
            stream: &CudaStream,
            [x, w]: [&DeviceBuffer<f32>; 2],
            ids: &DeviceBuffer<u32>,
            key: ExchangeKey,
        ) -> Result<(), GpuError> {
            let (n, s) = (self.n_embd, N_USED);
            let ExchangeKey { at, u, .. } = key;
            let set = &mut self.sets[self.down];
            if set.stage != Stage::Free {
                return Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!(
                        "a route of {key:?} into an exchange set that holds {:?}: both sets \
                         hold a layer not uploaded yet",
                        set.stage
                    ),
                });
            }
            // SAFETY: each copy reads the first values of a device buffer
            // (u ≤ cap, checked by the caller) and writes as many into this
            // set's page-locked buffers. The set is free: its last serve
            // returned, so the union no longer reads them, and its upload is
            // enqueued before these copies. The host reads them again only
            // in this set's next serve, after its wait on the event recorded
            // below.
            unsafe {
                dtoh(stream, &mut set.x, x, at * n..u * n)?;
                dtoh(stream, &mut set.ids, ids, at * s..u * s)?;
                dtoh(stream, &mut set.w, w, at * s..u * s)?;
            }
            set.routed.record(stream)?;
            set.stage = Stage::Routed(key);
            self.down ^= 1;
            Ok(())
        }

        /// Wait for the oldest unserved set's copies, which must be `key`'s,
        /// then serve its layer's host experts for its tokens in one union
        /// call: the sums into the set's copy the upload sends.
        pub(super) fn serve<H: HostExperts>(
            &mut self,
            hybrid: &mut Hybrid<H>,
            key: ExchangeKey,
        ) -> Result<ServeTimes, GpuError> {
            let n = self.n_embd;
            let ExchangeKey { layer, at, u, .. } = key;
            let set = &mut self.sets[self.serve];
            if set.stage != Stage::Routed(key) {
                return Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!(
                        "the serve of {key:?}; the oldest unserved exchange set holds {:?}",
                        set.stage
                    ),
                });
            }
            let t0 = Instant::now();
            set.routed.synchronize()?;
            let t1 = Instant::now();
            let x = Tensor2View::new(&set.x[at * n..u * n], n, u - at)?;
            let times = ServeTimes {
                wait_ns: nanos(t1 - t0),
                copy_ns: nanos(t1.elapsed()),
            };
            hybrid.serve_batch(
                layer,
                x,
                &set.ids[at * N_USED..u * N_USED],
                &set.w[at * N_USED..u * N_USED],
                &[],
                &mut set.sum[at * n..u * n],
            )?;
            set.stage = Stage::Served(key);
            self.serve ^= 1;
            Ok(times)
        }

        /// Enqueue the copy of the oldest served set's sums, which must be
        /// `key`'s, to `hsum`; the set is free again.
        pub(super) fn upload(
            &mut self,
            stream: &CudaStream,
            hsum: &mut DeviceBuffer<f32>,
            key: ExchangeKey,
        ) -> Result<(), GpuError> {
            let n = self.n_embd;
            let ExchangeKey { at, u, .. } = key;
            let set = &mut self.sets[self.up];
            if set.stage != Stage::Served(key) {
                return Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!(
                        "the upload of {key:?}; the oldest served exchange set holds {:?}",
                        set.stage
                    ),
                });
            }
            // SAFETY: the copy writes values at·n .. u·n of `hsum` from this
            // set's page-locked sums, which the host writes again only in
            // this set's next serve, after a wait on the event of its next
            // download — enqueued after this copy, since the set is free
            // from here on.
            unsafe { htod(stream, hsum, &set.sum, at * n..u * n)? };
            set.stage = Stage::Free;
            self.up ^= 1;
            Ok(())
        }
    }
}

/// How a batch's shadow reads the card's routed experts
/// ([`CardExperts::from_env`]): over the layer's block by tile items, each
/// weight row once for up to eight slots (`ds41_card_buckets`, then
/// `ds41_card_gather`, `ds41_expert_gate_up_tiles` and `q4k_gemv_tiles`);
/// over the block with each expert's row walking its slots one at a time
/// (`ds41_expert_gate_up_grouped`, `q4k_gemv_grouped`); or once per slot
/// chunk by chunk (`ds41_expert_gate_up_tok`). The last two are the
/// same-binary arms; all three write the same bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CardExperts {
    Tile,
    Expert,
    Slot,
}

impl CardExperts {
    /// `BLOOMERY_CARD_EXPERTS`: unset or `tile` walks tile items, `expert`
    /// walks each expert's slots, `slot` reads per slot; any other value is
    /// refused by name.
    pub fn from_env() -> Result<CardExperts, GpuError> {
        match std::env::var("BLOOMERY_CARD_EXPERTS").as_deref() {
            Err(_) | Ok("tile") => Ok(CardExperts::Tile),
            Ok("expert") => Ok(CardExperts::Expert),
            Ok("slot") => Ok(CardExperts::Slot),
            Ok(_) => Err(GpuError::State {
                what: "BLOOMERY_CARD_EXPERTS",
                missing: "tile, expert or slot",
            }),
        }
    }

    /// The name a `load` line prints.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            CardExperts::Tile => "tile",
            CardExperts::Expert => "expert",
            CardExperts::Slot => "slot",
        }
    }
}

/// Which part of a chunk's shadow a call enqueues: all of it (the per-slot
/// arm), or the part before the block's grouped experts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Whole,
    /// HC_PRE, the norm's q8_1 form (and its copy into the block's), the
    /// shared expert; the routed experts and the card sum run over the block.
    Before,
}

/// [`FfnBatch::serve`]'s host time outside the union call, in nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ServeTimes {
    /// Waiting on the route's copies to the host.
    pub wait_ns: u64,
    /// Copying the activations into the union's view.
    pub copy_ns: u64,
}

impl FfnBatch {
    /// The buffers for groups of up to `sets` batches of up to `cap` tokens
    /// (at most [`UNION_MAX_COLS`], the host union's columns) of rows of
    /// `n_embd` through experts of `ff`, on `gpu`.
    pub fn new(
        gpu: &Gpu,
        n_embd: usize,
        ff: usize,
        [cap, sets]: [usize; 2],
        experts: CardExperts,
    ) -> Result<FfnBatch, GpuError> {
        if cap == 0 || cap > UNION_MAX_COLS || sets == 0 {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "groups of {sets} batches of {cap} tokens: at least one batch, of \
                     1..={UNION_MAX_COLS} tokens, the host union's"
                ),
            });
        }
        let (ctx, stream) = (gpu.context(), gpu.stream());
        let c = HC_MAX_TOKENS;
        let z = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let counts = 1..=c;
        let n_sb = n_embd / 256;
        let kernels = FfnBatchKernels::load(ctx)?;
        let slot_cols = cap * N_USED;
        Ok(FfnBatch {
            kernels,
            cap,
            n_embd,
            ff,
            x: z(cap * n_embd)?,
            ids: DeviceBuffer::zeroed(stream, cap * N_USED)?,
            weights: z(cap * N_USED)?,
            sel: DeviceBuffer::zeroed(stream, cap * N_USED)?,
            hc: (0..sets)
                .map(|_| z(cap * HC_MIX))
                .collect::<Result<_, _>>()?,
            acc: z(cap * n_embd)?,
            shexp: z(cap * n_embd)?,
            hsum: z(cap * n_embd)?,
            probs: z(cap * N_EXPERT)?,
            normed: z(c * n_embd)?,
            act_x: counts
                .clone()
                .map(|m| Q8Act::with_k(stream, m, n_embd))
                .collect::<Result<_, _>>()?,
            mixes: z(c * HC_MIX)?,
            h: z(c * N_USED * ff)?,
            act_h: counts
                .clone()
                .map(|m| {
                    let slots = m * N_USED;
                    if slots <= 8 {
                        Q8Act::with_k(stream, slots, ff)
                    } else {
                        Q8Act::with_slots(stream, slots, ff)
                    }
                })
                .collect::<Result<_, _>>()?,
            down: z(c * N_USED * n_embd)?,
            sh_h: z(c * ff)?,
            act_sh: counts
                .map(|m| Q8Act::with_k(stream, m, ff))
                .collect::<Result<_, _>>()?,
            sh_raw: z(c * n_embd)?,
            experts,
            q3_all: DeviceBuffer::zeroed(stream, cap * 64 * n_sb.div_ceil(2))?,
            d8_all: z(cap * 2 * n_sb)?,
            h_all: z(cap * N_USED * ff)?,
            act_h_all: Q8Act::with_slots(stream, cap * N_USED, ff)?,
            down_all: z(cap * N_USED * n_embd)?,
            order: DeviceBuffer::zeroed(stream, cap * N_USED)?,
            start: DeviceBuffer::zeroed(stream, BUCKET_EXPERTS + 1)?,
            q3_ord: DeviceBuffer::zeroed(stream, slot_cols * 64 * n_sb.div_ceil(2))?,
            d8_ord: z(slot_cols * 2 * n_sb)?,
            tiles: DeviceBuffer::zeroed(stream, tile_cap(slot_cols, BUCKET_EXPERTS) + 1)?,
            exchange: exchange::Exchange::new(ctx, n_embd, cap)?,
        })
    }

    /// How the shadow reads the card's routed experts.
    #[must_use]
    pub fn experts(&self) -> CardExperts {
        self.experts
    }

    /// Tokens a batch takes at most.
    #[must_use]
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Batches a group holds at most: the sets of HC_PRE results.
    #[must_use]
    pub fn sets(&self) -> usize {
        self.hc.len()
    }

    /// The HC_PRE results of the last layer's shadow of the group's batch
    /// `set`, [`HC_MIX`] a token: the `pre` that folds the streams into an
    /// engram layer's attention and into the head. Refused past the sets.
    pub fn hc(&self, set: usize) -> Result<&DeviceBuffer<f32>, GpuError> {
        hc_set(&self.hc, set)
    }

    /// Of [`FfnBatch::device_bytes`], the HC_PRE results of every batch of a
    /// group.
    #[must_use]
    pub fn hc_bytes(&self) -> usize {
        self.hc.iter().map(DeviceBuffer::num_bytes).sum()
    }

    /// Both exchange sets free: at a group's start, before its first launch,
    /// once the stream holds nothing an earlier group enqueued.
    pub fn begin_group(&mut self) {
        self.exchange.begin();
    }

    /// Device bytes of the buffers, the chunk scratch's included; the
    /// page-locked host copies are not counted.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        let f32s = [
            &self.x,
            &self.weights,
            &self.acc,
            &self.shexp,
            &self.hsum,
            &self.probs,
            &self.normed,
            &self.mixes,
            &self.h,
            &self.down,
            &self.sh_h,
            &self.sh_raw,
            &self.d8_all,
            &self.h_all,
            &self.down_all,
            &self.d8_ord,
        ];
        let acts = self
            .act_x
            .iter()
            .chain(&self.act_h)
            .chain(&self.act_sh)
            .chain(std::iter::once(&self.act_h_all));
        f32s.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.hc.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.ids.num_bytes()
            + self.sel.num_bytes()
            + self.q3_all.num_bytes()
            + self.q3_ord.num_bytes()
            + self.tiles.num_bytes()
            + self.order.num_bytes()
            + self.start.num_bytes()
            + acts.map(q8act_bytes).sum::<usize>()
    }

    /// Enqueue the feature tap of `m` tokens: the mean of each one's four
    /// streams (`streams`, `4 · n_embd` a token) into its row of `rows`
    /// (`width` values a row, the mean at `off`), the value the step's tap
    /// writes. One launch. Asynchronous, allocation-free.
    pub fn enqueue_tap_means(
        &self,
        gpu: &Gpu,
        streams: &DeviceBuffer<f32>,
        m: usize,
        width: usize,
        off: usize,
        rows: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (n, what) = (self.n_embd, "ds41_tap_means");
        if m == 0 || off + n > width || streams.len() < HC_STREAMS * n * m || rows.len() < width * m
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a tap of {m} tokens at {off} of rows of {width}: streams {}, rows {}",
                    streams.len(),
                    rows.len()
                ),
            });
        }
        let grid = launch_u32(what, "grid", (m * n).div_ceil(THREADS as usize))?;
        let prep = self
            .kernels
            .module
            .prepare_ds41_tap_means(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.kernels.module.ds41_tap_means(
            gpu.stream(),
            &prep,
            streams,
            launch_u32(what, "n", n)?,
            launch_u32(what, "m", m)?,
            launch_u32(what, "width", width)?,
            launch_u32(what, "off", off)?,
            rows,
        )?;
        Ok(())
    }

    /// Enqueue the copies of `key`'s tokens' activations and routing to the
    /// host into the next exchange set, and the event they complete at.
    /// Asynchronous. Refused while both sets hold a layer not uploaded yet.
    pub fn enqueue_download(&mut self, gpu: &Gpu, key: ExchangeKey) -> Result<(), GpuError> {
        self.check_key(key)?;
        self.exchange
            .download(gpu.stream(), [&self.x, &self.weights], &self.ids, key)
    }

    /// Wait for the oldest download not served yet — `key`'s, else refused
    /// by name — then serve its layer's host experts for its tokens in one
    /// union call: the sums into the set's host copy
    /// [`FfnBatch::enqueue_upload`] sends back. Returns the host time
    /// outside the union call.
    pub fn serve<H: HostExperts>(
        &mut self,
        hybrid: &mut Hybrid<H>,
        key: ExchangeKey,
    ) -> Result<ServeTimes, GpuError> {
        self.check_key(key)?;
        self.exchange.serve(hybrid, key)
    }

    /// Enqueue the copy of the oldest served set's host sums — `key`'s, else
    /// refused by name — to the card. Asynchronous.
    pub fn enqueue_upload(&mut self, gpu: &Gpu, key: ExchangeKey) -> Result<(), GpuError> {
        self.check_key(key)?;
        self.exchange.upload(gpu.stream(), &mut self.hsum, key)
    }

    /// A key's tokens `at .. u` of a batch: at least one, at most the cap;
    /// its batch within the group's sets.
    fn check_key(&self, key: ExchangeKey) -> Result<(), GpuError> {
        let ExchangeKey { set, at, u, .. } = key;
        if at >= u || u > self.cap || set >= self.hc.len() {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "{key:?}: tokens of a batch of at most {}, a batch of a group of at most {}",
                    self.cap,
                    self.hc.len()
                ),
            });
        }
        Ok(())
    }
}

/// Which layer-batch a host exchange set holds ([`FfnBatch::enqueue_download`],
/// [`FfnBatch::serve`], [`FfnBatch::enqueue_upload`]): model layer `layer` of
/// the group's batch `set`, the batch's tokens `at .. u` of its block. Two
/// batches' blocks can cover the same tokens of their batches; their keys
/// still differ, so a serve or an upload out of the route order is refused
/// by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExchangeKey {
    pub layer: usize,
    pub set: usize,
    pub at: usize,
    pub u: usize,
}

/// The group's batch `set`'s HC_PRE results of `hc`, refused past the sets.
fn hc_set(hc: &[DeviceBuffer<f32>], set: usize) -> Result<&DeviceBuffer<f32>, GpuError> {
    hc.get(set).ok_or_else(|| GpuError::Shape {
        what: WHAT,
        detail: format!("batch {set} of a group of at most {}", hc.len()),
    })
}

/// [`hc_set`], to write.
fn hc_set_mut(
    hc: &mut [DeviceBuffer<f32>],
    set: usize,
) -> Result<&mut DeviceBuffer<f32>, GpuError> {
    let n = hc.len();
    hc.get_mut(set).ok_or_else(|| GpuError::Shape {
        what: WHAT,
        detail: format!("batch {set} of a group of at most {n}"),
    })
}

/// Enqueue the copy of `len` values of `src` from its start to `dst` from
/// value `at`.
///
/// SAFETY: neither buffer is written or freed by anything but this stream
/// until the copy completes.
unsafe fn dtod<T: cuda_core::DeviceCopy>(
    stream: &CudaStream,
    dst: &mut DeviceBuffer<T>,
    at: usize,
    src: &DeviceBuffer<T>,
    len: usize,
) -> Result<(), GpuError> {
    if len == 0 || len > src.len() || at + len > dst.len() {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!("{len} values from {} into {} at {at}", src.len(), dst.len()),
        });
    }
    // SAFETY: both ranges are inside their buffers (checked above); the
    // buffers stay untouched off this stream until the copy completes by
    // this fn's contract.
    let rc = unsafe {
        sys::cuMemcpyDtoDAsync_v2(
            dst.cu_deviceptr() + (at * size_of::<T>()) as u64,
            src.cu_deviceptr(),
            len * size_of::<T>(),
            stream.cu_stream(),
        )
    };
    rc.result().map_err(|source| GpuError::Driver {
        op: Some("cuMemcpyDtoDAsync_v2 (the block's q8_1 planes)"),
        source,
    })
}

/// Enqueue the copy of values `at` of `src` to the same values of `dst`.
///
/// SAFETY: `dst` is not read, written or freed until the copy completes.
unsafe fn dtoh<T: cuda_core::DeviceCopy>(
    stream: &CudaStream,
    dst: &mut PinnedHostBuffer<T>,
    src: &DeviceBuffer<T>,
    at: Range<usize>,
) -> Result<(), GpuError> {
    if at.is_empty() || at.end > src.len() || at.end > dst.len() {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!("values {at:?} from {} into {}", src.len(), dst.len()),
        });
    }
    // SAFETY: both buffers hold the values of `at` (checked above); `dst`
    // stays untouched until the copy completes by this fn's contract.
    let rc = unsafe {
        sys::cuMemcpyDtoHAsync_v2(
            dst.as_mut_ptr().add(at.start).cast(),
            src.cu_deviceptr() + (at.start * size_of::<T>()) as u64,
            at.len() * size_of::<T>(),
            stream.cu_stream(),
        )
    };
    rc.result().map_err(|source| GpuError::Driver {
        op: Some("cuMemcpyDtoHAsync_v2 (the batch's handoffs)"),
        source,
    })
}

/// Enqueue the copy of values `at` of `src` to the same values of `dst`.
///
/// SAFETY: `src` is not written or freed until the copy completes.
unsafe fn htod<T: cuda_core::DeviceCopy>(
    stream: &CudaStream,
    dst: &mut DeviceBuffer<T>,
    src: &PinnedHostBuffer<T>,
    at: Range<usize>,
) -> Result<(), GpuError> {
    if at.is_empty() || at.end > src.len() || at.end > dst.len() {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!("values {at:?} from {} into {}", src.len(), dst.len()),
        });
    }
    // SAFETY: both buffers hold the values of `at` (checked above); `src`
    // stays unwritten until the copy completes by this fn's contract.
    let rc = unsafe {
        sys::cuMemcpyHtoDAsync_v2(
            dst.cu_deviceptr() + (at.start * size_of::<T>()) as u64,
            src.as_ptr().add(at.start).cast(),
            at.len() * size_of::<T>(),
            stream.cu_stream(),
        )
    };
    rc.result().map_err(|source| GpuError::Driver {
        op: Some("cuMemcpyHtoDAsync_v2 (the batch's host sums)"),
        source,
    })
}

/// The buffers one chunk of a batch shares with the rest of it: tokens `at
/// .. at + m` of the batch, their streams and folds.
pub struct ChunkIo<'a> {
    /// The batch's place in its group ([`FfnBatch::hc`]'s set).
    pub set: usize,
    /// The chunk's first token in the batch, and its tokens.
    pub at: usize,
    pub m: usize,
    /// The streams the sub-layer reads, `4 · n_embd` a token.
    pub streams: &'a DeviceBuffer<f32>,
    /// The norm's input, `n_embd` a token.
    pub fold_in: &'a DeviceBuffer<f32>,
}

/// A layer's block over a batch: its chunks, consecutive runs of at most
/// [`HC_MAX_TOKENS`] positions (position `p` is the batch's token `p −
/// base`), and the batch's streams and folds from its token 0 on.
pub struct BlockIo<'a> {
    /// The batch's place in its group ([`FfnBatch::hc`]'s set).
    pub set: usize,
    pub chunks: &'a [Range<usize>],
    /// The batch's first position.
    pub base: usize,
    /// The streams the sub-layer reads, `4 · n_embd` a token.
    pub streams: &'a DeviceBuffer<f32>,
    /// The norm's input, `n_embd` a token.
    pub fold_in: &'a DeviceBuffer<f32>,
}

/// The buffers a batch join shares with the rest of the batch: its streams
/// and folds, the batch's tokens from 0 on — the join reads and writes the
/// tokens it is given.
pub struct JoinIo<'a> {
    /// The batch's place in its group ([`FfnBatch::hc`]'s set).
    pub set: usize,
    /// The streams the sub-layer read, `4 · n_embd` a token.
    pub streams: &'a DeviceBuffer<f32>,
    /// The new streams.
    pub streams_out: &'a mut DeviceBuffer<f32>,
    /// The next fold, `n_embd` a token: `Some` exactly where the layer folds.
    pub fold_out: Option<&'a mut DeviceBuffer<f32>>,
}

impl FfnPiece {
    /// Layer `layer`'s tensors in `w` for a batch's route and shadow of it,
    /// resolved once: each in the format its launch reads, the shared
    /// expert's down in the format the file's header names for it.
    pub fn resolve_batch<'w>(
        &self,
        w: &'w Weights,
        layer: usize,
    ) -> Result<BatchLayer<'w>, GpuError> {
        let i = self.layer_index(layer, WHAT)?;
        let c = &self.cfg[i];
        let lw = LayerWeights::resolve(c, w, [self.n_embd, self.ff], self.hc_eps, self.hc_iters)?;
        if lw.sh_down.reads_q8_1() != c.sh_down_q8_1 {
            return Err(tensor_err(
                &c.sh_down,
                "of the format the file's header names for it",
            ));
        }
        Ok(BatchLayer { layer, lw })
    }

    /// The route of `bl`'s layer for the block `io` of a batch: per chunk the
    /// norm with its q8_1 form; then, over the block's tokens, the router
    /// ([`RouterKernels::enqueue_router_rows`], two launches) and each slot's
    /// place from `slots`, the map's card copy (one launch). Asynchronous,
    /// allocation-free.
    pub fn enqueue_batch_route(
        &mut self,
        gpu: &Gpu,
        bl: &BatchLayer<'_>,
        b: &mut FfnBatch,
        io: &BlockIo<'_>,
        slots: &DeviceTensor<u32>,
    ) -> Result<(), GpuError> {
        let (layer, lw) = (bl.layer, &bl.lw);
        let (i, at, u) = self.batch_block(layer, b, io)?;
        let (stream, n) = (gpu.stream(), self.n_embd);
        let c = &self.cfg[i];
        let fault = gpu.layer_sink(layer)?;
        for r in io.chunks {
            let (at, m) = (r.start - io.base, r.len());
            let streams = span(WHAT, io.streams, at * HC_STREAMS * n, m * HC_STREAMS * n)?;
            let fold_in = span(WHAT, io.fold_in, at * n, m * n)?;
            let chunk = ChunkIo {
                set: io.set,
                at,
                m,
                streams: &streams,
                fold_in: &fold_in,
            };
            self.batch_layer(layer, b, &chunk)?;
            let mut x = span_mut(WHAT, &mut b.x, chunk.at * n, chunk.m * n)?;
            self.fused.enqueue_norm_quant(
                stream,
                chunk.fold_in,
                lw.gain,
                self.rms_eps,
                count_of(&mut b.act_x, chunk.m)?,
                &mut x,
                fault,
            )?;
        }
        let t = u - at;
        {
            let x = span(WHAT, &b.x, at * n, t * n)?;
            let mut probs = span_mut(WHAT, &mut b.probs, 0, t * N_EXPERT)?;
            let mut ids = span_mut(WHAT, &mut b.ids, at * N_USED, t * N_USED)?;
            let mut wts = span_mut(WHAT, &mut b.weights, at * N_USED, t * N_USED)?;
            self.router.enqueue_router_rows(
                stream, lw.router, &x, lw.bias, self.scale, t, &mut probs, &mut ids, &mut wts,
                fault,
            )?;
        }
        let slots_n = t * N_USED;
        let ids = span(WHAT, &b.ids, at * N_USED, slots_n)?;
        let mut sel = span_mut(WHAT, &mut b.sel, at * N_USED, slots_n)?;
        let places = Places {
            ids: &ids,
            n: slots_n,
            map: slots.buf(),
            row_off: c.row_off,
            n_expert: self.n_expert,
        };
        b.kernels.enqueue_places(stream, &places, fault, &mut sel)
    }

    /// The shadow work of `bl`'s layer for the block `io`, after the batch's
    /// route of it. Per slot ([`CardExperts::Slot`]): each chunk's whole
    /// shadow in turn. Per expert: each chunk's HC_PRE, norm and shared
    /// expert; then over the block the card slots grouped by expert and one
    /// gate·up launch that reads each card expert once; then each chunk's down
    /// and card sum. Asynchronous, allocation-free.
    pub fn enqueue_batch_shadow(
        &mut self,
        gpu: &Gpu,
        bl: &BatchLayer<'_>,
        card: Option<CardStacks<'_>>,
        b: &mut FfnBatch,
        io: &BlockIo<'_>,
    ) -> Result<(), GpuError> {
        let (layer, lw) = (bl.layer, &bl.lw);
        let (i, at, u) = self.batch_block(layer, b, io)?;
        let card = self.check_card(layer, i, card)?;
        let phase = match b.experts {
            CardExperts::Slot => Phase::Whole,
            CardExperts::Tile | CardExperts::Expert => Phase::Before,
        };
        let n = self.n_embd;
        {
            for r in io.chunks {
                let (at, m) = (r.start - io.base, r.len());
                let streams = span(WHAT, io.streams, at * HC_STREAMS * n, m * HC_STREAMS * n)?;
                let fold_in = span(WHAT, io.fold_in, at * n, m * n)?;
                let chunk = ChunkIo {
                    set: io.set,
                    at,
                    m,
                    streams: &streams,
                    fold_in: &fold_in,
                };
                self.batch_layer(layer, b, &chunk)?;
                self.enqueue_batch_shadow_chunk(gpu, lw, i, card, b, layer, &chunk, phase)?;
            }
        }
        if phase == Phase::Before {
            self.enqueue_grouped_block(gpu, i, card, b, layer, at, u)?;
        }
        Ok(())
    }

    /// The block-wide arms' launches for tokens `at .. u`, after every
    /// chunk's part before them: with card experts, the card slots by expert
    /// ([`ds41_card_buckets`]) and the gate·up, its q8_1 form and the down of
    /// every card slot — by tile items ([`FfnPiece::enqueue_tiled_experts`])
    /// or each expert's slots in turn ([`FfnPiece::enqueue_grouped_experts`])
    /// — into the block's down outputs by slot; then the card sum of every
    /// token of the block.
    #[allow(
        clippy::too_many_arguments,
        reason = "the block's layer, stacks, buffers and tokens (rust-quality R8)"
    )]
    fn enqueue_grouped_block(
        &self,
        gpu: &Gpu,
        i: usize,
        card: Option<CardStacks<'_>>,
        b: &mut FfnBatch,
        layer: usize,
        at: usize,
        u: usize,
    ) -> Result<(), GpuError> {
        let (stream, n) = (gpu.stream(), self.n_embd);
        let c = &self.cfg[i];
        let slots_n = (u - at) * N_USED;
        if let Some(s) = card {
            if b.experts == CardExperts::Tile {
                self.enqueue_tiled_experts(gpu, i, s, b, layer, at, u)?;
            } else {
                self.enqueue_grouped_experts(gpu, i, s, b, layer, at, u)?;
            }
        }
        let what = "ds41_ffn_card_acc";
        let m = u - at;
        let down = span(WHAT, &b.down_all, at * N_USED * n, slots_n * n)?;
        let sel = span(WHAT, &b.sel, at * N_USED, slots_n)?;
        let wts = span(WHAT, &b.weights, at * N_USED, slots_n)?;
        let mut acc = span_mut(WHAT, &mut b.acc, at * n, m * n)?;
        let grid = launch_u32(what, "grid", (m * n).div_ceil(THREADS as usize))?;
        let prep = b
            .kernels
            .module
            .prepare_ds41_ffn_card_acc(LaunchConfig1D::new(grid, THREADS, 0))?;
        b.kernels.module.ds41_ffn_card_acc(
            stream,
            &prep,
            &down,
            &wts,
            &sel,
            launch_u32(what, "n", n)?,
            launch_u32(what, "m", m)?,
            launch_u32(what, "n_card", c.n_card)?,
            &mut acc,
        )?;
        Ok(())
    }

    /// The card's stack count for layer `layer`'s block, refused past the
    /// bucket kernel's width, and the block's table `b.order`/`b.start` of
    /// its `slots_n` slots from `at`. One launch.
    #[allow(
        clippy::too_many_arguments,
        reason = "the block's layer, stack, buffers and tokens (rust-quality R8)"
    )]
    fn enqueue_block_buckets(
        &self,
        gpu: &Gpu,
        s: CardStacks<'_>,
        b: &mut FfnBatch,
        layer: usize,
        at: usize,
        slots_n: usize,
        fault: FaultSink,
    ) -> Result<usize, GpuError> {
        let n_experts = s.gate.rows() / self.ff;
        if n_experts > BUCKET_EXPERTS {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "layer {layer}'s card stack of {n_experts} experts; the block-wide arms \
                     take at most {BUCKET_EXPERTS} (BLOOMERY_CARD_EXPERTS=slot runs any)"
                ),
            });
        }
        let sel = span(WHAT, &b.sel, at * N_USED, slots_n)?;
        let mut order = span_mut(WHAT, &mut b.order, 0, slots_n)?;
        b.kernels.enqueue_buckets(
            gpu.stream(),
            &sel,
            slots_n,
            n_experts,
            fault,
            &mut order,
            &mut b.start,
        )?;
        Ok(n_experts)
    }

    /// The card experts of the block `at .. u` ([`FfnPiece::enqueue_grouped_block`]).
    #[allow(
        clippy::too_many_arguments,
        reason = "the block's layer, stacks, buffers and tokens (rust-quality R8)"
    )]
    fn enqueue_grouped_experts(
        &self,
        gpu: &Gpu,
        i: usize,
        s: CardStacks<'_>,
        b: &mut FfnBatch,
        layer: usize,
        at: usize,
        u: usize,
    ) -> Result<(), GpuError> {
        let (stream, ff, n) = (gpu.stream(), self.ff, self.n_embd);
        let c = &self.cfg[i];
        let fault = gpu.layer_sink(layer)?;
        let slots_n = (u - at) * N_USED;
        let n_experts = self.enqueue_block_buckets(gpu, s, b, layer, at, slots_n, fault)?;
        let sel = span(WHAT, &b.sel, at * N_USED, slots_n)?;
        let order = span(WHAT, &b.order, 0, slots_n)?;
        let mut h = span_mut(WHAT, &mut b.h_all, at * N_USED * ff, slots_n * ff)?;
        let g = GroupedGateUp {
            wg: s.gate.buf(),
            wu: s.up.buf(),
            q3: &b.q3_all,
            d8: &b.d8_all,
            order: &order,
            start: &b.start,
            n_experts,
            rows_per_expert: ff,
            n_slots: slots_n,
            col0: at,
            cols: u,
            n_sb: self.n_embd / 256,
            limit: c.limit,
        };
        b.kernels
            .enqueue_gate_up_grouped(stream, &g, fault, &mut h)?;
        drop(h);
        let q = QuantSel {
            x: &b.h_all,
            cols: at * N_USED..u * N_USED,
            sel: &sel,
            n_card: n_experts,
        };
        gpu.q4k_sel()
            .enqueue_quantize_sel(stream, &q, fault, &mut b.act_h_all)?;
        let mut down = span_mut(WHAT, &mut b.down_all, at * N_USED * n, slots_n * n)?;
        gpu.q4k_sel().enqueue_gemv_q4k_grouped(
            stream,
            s.down,
            &b.act_h_all,
            &order,
            &b.start,
            slots_n,
            at * N_USED,
            n,
            fault,
            &mut down,
        )?;
        Ok(())
    }

    /// The card experts of the block `at .. u` by tile items
    /// ([`FfnPiece::enqueue_grouped_block`]): the table and its tiles
    /// (`q4k_sel::grouped_tiles`), the norm's q8_1 planes gathered by table
    /// entry (`ds41_card_gather`), the gate·up
    /// (`ds41_expert_gate_up_tiles`) into the SwiGLU outputs by entry, their
    /// q8_1 form by entry, and the down (`q4k_gemv_tiles`), which scatters
    /// each entry's rows to its slot's down outputs. The table's size is a
    /// device value: no launch here reads it on the host.
    #[allow(
        clippy::too_many_arguments,
        reason = "the block's layer, stacks, buffers and tokens (rust-quality R8)"
    )]
    fn enqueue_tiled_experts(
        &self,
        gpu: &Gpu,
        i: usize,
        s: CardStacks<'_>,
        b: &mut FfnBatch,
        layer: usize,
        at: usize,
        u: usize,
    ) -> Result<(), GpuError> {
        let (stream, ff, n) = (gpu.stream(), self.ff, self.n_embd);
        let c = &self.cfg[i];
        let fault = gpu.layer_sink(layer)?;
        let n_sb = self.n_embd / 256;
        let slots_n = (u - at) * N_USED;
        let n_experts = self.enqueue_block_buckets(gpu, s, b, layer, at, slots_n, fault)?;
        gpu.q4k_sel().enqueue_grouped_tiles(
            stream,
            &b.start,
            n_experts,
            slots_n,
            fault,
            &mut b.tiles,
        )?;
        let order = span(WHAT, &b.order, 0, slots_n)?;
        {
            let g = CardGather {
                q3: &b.q3_all,
                d8: &b.d8_all,
                order: &order,
                start: &b.start,
                n_experts,
                n_slots: slots_n,
                col0: at,
                cols: u,
                n_sb,
            };
            let mut q3 = span_mut(WHAT, &mut b.q3_ord, 0, slots_n * 64 * n_sb.div_ceil(2))?;
            let mut d8 = span_mut(WHAT, &mut b.d8_ord, 0, slots_n * 2 * n_sb)?;
            b.kernels
                .enqueue_card_gather(stream, &g, fault, &mut q3, &mut d8)?;
        }
        let q3 = span(WHAT, &b.q3_ord, 0, slots_n * 64 * n_sb.div_ceil(2))?;
        let d8 = span(WHAT, &b.d8_ord, 0, slots_n * 2 * n_sb)?;
        let mut h = span_mut(WHAT, &mut b.h_all, 0, slots_n * ff)?;
        let g = TiledGateUp {
            wg: s.gate.buf(),
            wu: s.up.buf(),
            q3: &q3,
            d8: &d8,
            start: &b.start,
            tiles: &b.tiles,
            n_experts,
            rows_per_expert: ff,
            n_slots: slots_n,
            n_sb,
            limit: c.limit,
        };
        b.kernels.enqueue_gate_up_tiles(stream, &g, fault, &mut h)?;
        drop(h);
        gpu.q4k_sel().enqueue_quantize_ord(
            stream,
            &b.h_all,
            &b.start,
            n_experts,
            slots_n,
            fault,
            &mut b.act_h_all,
        )?;
        let mut down = span_mut(WHAT, &mut b.down_all, at * N_USED * n, slots_n * n)?;
        gpu.q4k_sel().enqueue_gemv_q4k_tiles(
            stream,
            s.down,
            &b.act_h_all,
            &order,
            &b.start,
            &b.tiles,
            slots_n,
            n,
            fault,
            &mut down,
        )?;
        Ok(())
    }

    /// Layer `layer`'s shadow work for the chunk `io`, after the batch's
    /// route — the parts `phase` names: HC_PRE into the batch's results,
    /// the norm's q8_1 form again, the card's routed experts over the
    /// chunk's slots (the per-slot gate·up, or the block's grouped outputs)
    /// and their sum, the shared expert into the batch's outputs. `lw` are
    /// layer `i`'s tensors.
    #[allow(
        clippy::too_many_arguments,
        reason = "the shadow's inputs, the layer's resolved tensors and the phase (rust-quality R8)"
    )]
    fn enqueue_batch_shadow_chunk(
        &mut self,
        gpu: &Gpu,
        lw: &LayerWeights<'_>,
        i: usize,
        card: Option<CardStacks<'_>>,
        b: &mut FfnBatch,
        layer: usize,
        io: &ChunkIo<'_>,
        phase: Phase,
    ) -> Result<(), GpuError> {
        let (stream, n, ff, m, at) = (gpu.stream(), self.n_embd, self.ff, io.m, io.at);
        let c = &self.cfg[i];
        let fault = gpu.layer_sink(layer)?;
        {
            let pre = HcPreArgs {
                params: &lw.hc,
                x: io.streams,
                tokens: m,
                rms_eps: self.rms_eps,
                fault,
            };
            let mut hc = span_mut(
                WHAT,
                hc_set_mut(&mut b.hc, io.set)?,
                at * HC_MIX,
                m * HC_MIX,
            )?;
            self.hc
                .enqueue_pre(stream, &pre, &mut self.hc_scratch, &mut b.mixes, &mut hc)?;
            drop(hc);
            let act_x = count_of(&mut b.act_x, m)?;
            self.fused.enqueue_norm_quant(
                stream,
                io.fold_in,
                lw.gain,
                self.rms_eps,
                act_x,
                &mut b.normed,
                fault,
            )?;
            if phase == Phase::Before && card.is_some() {
                let act_x = &b.act_x[m - 1];
                let (q, d) = (act_x.q3().len() / m, act_x.d8().len() / m);
                // SAFETY: the copies read the chunk's own scratch and write
                // its tokens' columns of the block's planes; both are this
                // piece's buffers on the engine stream, which orders them
                // before the grouped gate·up that reads them and after any
                // earlier launch that touched those columns.
                unsafe {
                    dtod(stream, &mut b.q3_all, at * q, act_x.q3(), m * q)?;
                    dtod(stream, &mut b.d8_all, at * d, act_x.d8(), m * d)?;
                }
            }
        }
        if phase == Phase::Whole {
            let slots_n = m * N_USED;
            let sel = span(WHAT, &b.sel, at * N_USED, slots_n)?;
            if let Some(s) = card {
                let act_x = &b.act_x[m - 1];
                let what = "ds41_expert_gate_up_tok";
                let n_sb = act_x.n_sb();
                let grid = launch_u32(what, "grid", (slots_n * ff).div_ceil(ROWS_PER_BLOCK))?;
                let prep = b
                    .kernels
                    .module
                    .prepare_ds41_expert_gate_up_tok(LaunchConfig1D::new(grid, THREADS, 0))?;
                b.kernels.module.ds41_expert_gate_up_tok(
                    stream,
                    &prep,
                    s.gate.buf(),
                    s.up.buf(),
                    act_x.q3(),
                    act_x.d8(),
                    &sel,
                    launch_u32(what, "n_experts", s.gate.rows() / ff)?,
                    launch_u32(what, "rows_per_expert", ff)?,
                    launch_u32(what, "n_slots", slots_n)?,
                    launch_u32(what, "cols", m)?,
                    launch_u32(what, "n_sb", n_sb)?,
                    launch_u32(what, "iters", n_sb.div_ceil(2))?,
                    c.limit,
                    fault,
                    &mut b.h,
                )?;
                let act_h = count_of(&mut b.act_h, m)?;
                let q = QuantSel {
                    x: &b.h,
                    cols: 0..slots_n,
                    sel: &sel,
                    n_card: c.n_card,
                };
                gpu.q4k_sel()
                    .enqueue_quantize_sel(stream, &q, fault, act_h)?;
                gpu.q4k_sel().enqueue_gemv_q4k_sel(
                    stream,
                    s.down,
                    act_h,
                    &sel,
                    slots_n,
                    n,
                    &mut b.down,
                )?;
            }
            let what = "ds41_ffn_card_acc";
            let wts = span(WHAT, &b.weights, at * N_USED, slots_n)?;
            let mut acc = span_mut(WHAT, &mut b.acc, at * n, m * n)?;
            let grid = launch_u32(what, "grid", (m * n).div_ceil(THREADS as usize))?;
            let prep = b
                .kernels
                .module
                .prepare_ds41_ffn_card_acc(LaunchConfig1D::new(grid, THREADS, 0))?;
            b.kernels.module.ds41_ffn_card_acc(
                stream,
                &prep,
                &b.down,
                &wts,
                &sel,
                launch_u32(what, "n", n)?,
                launch_u32(what, "m", m)?,
                launch_u32(what, "n_card", c.n_card)?,
                &mut acc,
            )?;
        }
        match (lw.sh_gate, lw.sh_up) {
            (
                DevWeight::KQuant {
                    ty: GgmlType::Q3_K,
                    w: g,
                    ..
                },
                DevWeight::KQuant {
                    ty: GgmlType::Q3_K,
                    w: u,
                    ..
                },
            ) => self.dense.enqueue_shexp_gate_up_q3k(
                stream,
                g,
                u,
                &b.act_x[m - 1],
                c.limit_shared,
                &mut b.sh_h,
            )?,
            (gate, up) => {
                for t in 0..m {
                    let x = span(WHAT, &b.x, (at + t) * n, n)?;
                    let mut h = span_mut(WHAT, &mut b.sh_h, t * ff, ff)?;
                    self.experts.enqueue_shexp_gate_up(
                        stream,
                        gate,
                        up,
                        &x,
                        c.limit_shared,
                        &mut h,
                    )?;
                }
            }
        }
        let act = if lw.sh_down.reads_q8_1() {
            let act_sh = count_of(&mut b.act_sh, m)?;
            gpu.enqueue_quantize_q8_1_layer(&b.sh_h, act_sh, layer)?;
            Some(&b.act_sh[m - 1])
        } else {
            None
        };
        let mut shexp = span_mut(WHAT, &mut b.shexp, at * n, m * n)?;
        if m == 1 {
            self.dense
                .enqueue_m(gpu, lw.sh_down, &b.sh_h, act, 1, &mut shexp)?;
        } else {
            self.dense
                .enqueue_m(gpu, lw.sh_down, &b.sh_h, act, m, &mut b.sh_raw)?;
            self.transpose
                .enqueue(stream, &b.sh_raw, n, m, &mut shexp)?;
        }
        Ok(())
    }

    /// Layer `layer`'s join over the batch's tokens `at .. u`, after the host
    /// sums' upload ([`JoinIo`]). One launch. Asynchronous, allocation-free.
    pub fn enqueue_batch_join(
        &mut self,
        gpu: &Gpu,
        b: &FfnBatch,
        layer: usize,
        at: usize,
        u: usize,
        io: JoinIo<'_>,
    ) -> Result<(), GpuError> {
        let JoinIo {
            set,
            streams,
            streams_out,
            fold_out,
        } = io;
        let i = self.layer_index(layer, WHAT)?;
        let n = self.n_embd;
        if at >= u
            || u > b.cap
            || fold_out.is_some() != self.cfg[i].fold
            || streams.len() < HC_STREAMS * n * u
            || streams_out.len() < HC_STREAMS * n * u
            || fold_out.as_ref().is_some_and(|f| f.len() < n * u)
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "layer {layer}'s join of tokens {at}..{u} (batches of {}): streams {} and {}, \
                     a fold out {:?} (the layer folds: {})",
                    b.cap,
                    streams.len(),
                    streams_out.len(),
                    fold_out.as_ref().map(|f| f.len()),
                    self.cfg[i].fold
                ),
            });
        }
        let what = "ds41_ffn_post_batch";
        let m = u - at;
        let s4 = HC_STREAMS * n;
        let grid = launch_u32(what, "grid", (m * n).div_ceil(THREADS as usize))?;
        let cfg = LaunchConfig1D::new(grid, THREADS, 0);
        let (nn, mm) = (launch_u32(what, "n", n)?, launch_u32(what, "m", m)?);
        let stream = gpu.stream();
        let acc = span(WHAT, &b.acc, at * n, m * n)?;
        let hsum = span(WHAT, &b.hsum, at * n, m * n)?;
        let shexp = span(WHAT, &b.shexp, at * n, m * n)?;
        let hc = span(WHAT, hc_set(&b.hc, set)?, at * HC_MIX, m * HC_MIX)?;
        let res = span(WHAT, streams, at * s4, m * s4)?;
        let mut out = span_mut(WHAT, streams_out, at * s4, m * s4)?;
        match fold_out {
            Some(fold) => {
                let mut fold = span_mut(WHAT, fold, at * n, m * n)?;
                let prep = b.kernels.module.prepare_ds41_ffn_post_batch(cfg)?;
                b.kernels.module.ds41_ffn_post_batch(
                    stream, &prep, &acc, &hsum, &shexp, &res, &hc, nn, mm, &mut out, &mut fold,
                )?;
            }
            None => {
                let prep = b.kernels.module.prepare_ds41_ffn_post_batch_streams(cfg)?;
                b.kernels.module.ds41_ffn_post_batch_streams(
                    stream, &prep, &acc, &hsum, &shexp, &res, &hc, nn, mm, &mut out,
                )?;
            }
        }
        Ok(())
    }

    /// Layer `layer`'s index and the block's tokens `at .. u`, once the block
    /// `io` is checked: at least one chunk, each non-empty and starting where
    /// the one before it ends, from the batch's first position on, within
    /// the batch's cap.
    fn batch_block(
        &self,
        layer: usize,
        b: &FfnBatch,
        io: &BlockIo<'_>,
    ) -> Result<(usize, usize, usize), GpuError> {
        let i = self.layer_index(layer, WHAT)?;
        let (Some(first), Some(last)) = (io.chunks.first(), io.chunks.last()) else {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("layer {layer}'s block of no chunks"),
            });
        };
        let joined = io
            .chunks
            .windows(2)
            .all(|p| p[0].end == p[1].start && !p[1].is_empty());
        if first.is_empty() || !joined || first.start < io.base || last.end - io.base > b.cap {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "layer {layer}'s block {:?} of a batch from {} of at most {}: chunks must be \
                     non-empty and consecutive",
                    first.start..last.end,
                    io.base,
                    b.cap
                ),
            });
        }
        Ok((i, first.start - io.base, last.end - io.base))
    }

    /// Layer `layer`'s index, once the chunk `io` is checked against the
    /// batch's buffers.
    fn batch_layer(&self, layer: usize, b: &FfnBatch, io: &ChunkIo<'_>) -> Result<usize, GpuError> {
        let n = self.n_embd;
        let i = self.layer_index(layer, WHAT)?;
        if !(1..=HC_MAX_TOKENS).contains(&io.m)
            || io.at + io.m > b.cap
            || b.n_embd != n
            || b.ff != self.ff
            || io.streams.len() < HC_STREAMS * n * io.m
            || io.fold_in.len() < n * io.m
        {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "a chunk of {} tokens from {} in a batch of {} (rows of {} through {}): streams \
                     {}, fold {}; the piece's rows are {n} through {}",
                    io.m,
                    io.at,
                    b.cap,
                    b.n_embd,
                    b.ff,
                    io.streams.len(),
                    io.fold_in.len(),
                    self.ff
                ),
            });
        }
        Ok(i)
    }
}

/// The scratch of token count `m` in `by_count` (index `m − 1`).
fn count_of<T>(by_count: &mut [T], m: usize) -> Result<&mut T, GpuError> {
    let n = by_count.len();
    m.checked_sub(1)
        .and_then(|i| by_count.get_mut(i))
        .ok_or_else(|| GpuError::Shape {
            what: WHAT,
            detail: format!("a chunk of {m} tokens; the batch holds scratch for 1..={n}"),
        })
}
