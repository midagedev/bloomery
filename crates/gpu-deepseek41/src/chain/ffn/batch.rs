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
//!    [`FfnBatch::enqueue_upload`], through the host tier's batch port):
//!    the block's activations and routing to the host, one union call over
//!    its tokens for the layer's host experts ([`Hybrid::serve_key`]), the
//!    sums back;
//! 3. the shadow ([`FfnPiece::enqueue_batch_shadow`]), enqueued before the
//!    host computes: per chunk HC_PRE, the norm's q8_1 form again and the
//!    shared expert; then over the whole block the card's routed experts by
//!    tile items — an expert, a tile of its rows, up to eight of its slots —
//!    each weight row read once for the tile's slots (`ds41_card_buckets`,
//!    `ds41_card_gather`, `ds41_expert_gate_up_tiles`, the q8_1 of the
//!    table's entries, `q4k_gemv_tiles`), and their sum
//!    (`ds41_ffn_card_acc`);
//! 4. the join ([`FfnPiece::enqueue_batch_join`]), one launch over the block:
//!    the card sum, the host sum and the shared expert combined, then HC_POST
//!    with the next fold where the layer folds.
//!
//! The feature tap of a batch's kept tokens is one launch here too
//! ([`FfnBatch::enqueue_tap_means`], `ds41_tap_means`).
//!
//! The gather and the card sum have an 8-slot instance each (`_8`), which
//! their launchers ([`FfnBatchKernels::enqueue_card_gather`],
//! [`FfnBatchKernels::enqueue_card_acc`]) pick from the slot count; the
//! piece's batch runs six.
//!
//! The launches that raise on input a batch's own route cannot produce — a
//! place for an id past the stack, a tile that names no run of its table —
//! are [`FfnBatchKernels`]'s, which a gate loads alone and drives with it.
//!
//! Every token's values are the step's bit for bit: each launch writes, per
//! token, what its one-token launch writes — the m-column kernels carry that
//! contract, the routed dot reads its token's column through the step's
//! `q3k_row_dot`, and the combine is [`combine_elem`] cut at its one seam
//! ([`card_sum_elem`], then [`join_elem`]).

use std::mem::size_of;

use bloomery_gpu::cores::q3k_row_dot;
use bloomery_gpu::hybrid::{BatchKey, ServeTimes};
use bloomery_gpu::q4k_sel::{TILE_MAX_EXPERTS, tile_at, tile_cap};
use bloomery_gpu::{FaultSink, FaultSite, col_sums, store_cols};
use cuda_core::{CudaContext, IntoResult, sys};
use cuda_device::{SharedArray, warp};

use super::*;
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

    /// A block's places again after a prompt call's pick moved the layer's
    /// map, for the streamed pass: thread `i < n` reads slot `i`'s place
    /// under the map before the pick, `sel[i]`, and under the map now,
    /// `map[row_off + ids[i]]`. An id the pick admitted — its place was
    /// [`HOST`] and is a card slot now — goes to `ssel[i]` with `sel[i] =`
    /// [`HOST`] (its slot is landing); every other id's place now goes to
    /// `sel[i]` with `ssel[i] =` [`HOST`] (a victim's is [`HOST`]). An id not
    /// below `n_expert` raises [`FaultSite::ExpertId`] on `fault` and both
    /// places are [`HOST`].
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
            ids.len() >= n,
            map.len() >= row_off + n_expert,
            sel.len() >= n,
            ssel.len() >= n
        )
    )]
    pub fn ds41_ffn_replace(
        ids: &[u32],
        map: &[u32],
        row_off: u32,
        n_expert: u32,
        n: u32,
        fault: FaultSink,
        mut sel: DisjointSlice<u32>,
        mut ssel: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= ids.len() by the launch contract.
        let id = unsafe { *ids.get_unchecked(i) };
        // SAFETY: i < n <= sel.len() by the launch contract; thread i is
        // sel[i]'s only reader and writer.
        let was = unsafe { *sel.get_unchecked_mut(i) };
        let now = if id < n_expert {
            // SAFETY: id < n_expert, so row_off + id < map.len() by the launch
            // contract.
            unsafe { *map.get_unchecked(row_off as usize + id as usize) }
        } else {
            fault.raise(FaultSite::ExpertId);
            HOST
        };
        let fresh = was == HOST && now != HOST;
        // SAFETY: i < n <= sel.len(), ssel.len(); thread i is the only writer
        // of both entries.
        unsafe {
            *sel.get_unchecked_mut(i) = if fresh { HOST } else { now };
            *ssel.get_unchecked_mut(i) = if fresh { now } else { HOST };
        }
    }

    /// The streamed places back into the block's places once the streamed
    /// pass has run: thread `i < n` writes `ssel[i]` into `sel[i]` when it is
    /// not [`HOST`], so the card sum reads every card slot of the block.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (ssel.len() >= n, sel.len() >= n)
    )]
    pub fn ds41_ffn_sel_merge(ssel: &[u32], n: u32, mut sel: DisjointSlice<u32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= ssel.len() by the launch contract.
        let s = unsafe { *ssel.get_unchecked(i) };
        if s != HOST {
            // SAFETY: i < n <= sel.len(); thread i is sel[i]'s only writer.
            unsafe { *sel.get_unchecked_mut(i) = s };
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

    /// The q8_1 planes of a grouped table's entries (`order`, runs `start`,
    /// [`ds41_card_buckets`]): block `j` copies token column `col0 + order[j]
    /// / 6` of `q_in` and `d8_in` (`cols` columns, the block's from `col0`)
    /// to column `j` of `q_out` and `d8_out`, for each entry `j` below the
    /// count `start[n_experts]` — a byte copy, so column `j` is its slot's
    /// activation as the step's gate·up reads it. Entries from the count on
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

    /// The routed experts' gate·up·SwiGLU of a grouped table's entries by
    /// tile items: block `g + tile_cap · ρ` (of `tile_cap · row_tiles`) is
    /// tile `g` of the table `tiles` (`q4k_sel::grouped_tiles`,
    /// `q4k_sel::tile_at`) — expert `e`, the `m` table entries from `j` — on
    /// the eight weight rows `8ρ ..`: warp `w` takes row `r = 8ρ + w` of both
    /// stacks and dots it with the `m` columns `j .. j + m` of the q8_1
    /// planes — the table's entries ([`ds41_card_gather`]) — through the
    /// m-column core `cores::q3k_row_dot`, whose column c is the one-column
    /// call on that column bit for bit, then the warp tree and
    /// `swiglu_clamp` into `h[(j + c) · rows_per_expert + r]`: entry `j +
    /// c`'s value, the step's gate·up value (`ds41_expert_gate_up`) for its
    /// slot and row. The blocks of one row tile are consecutive, so an
    /// expert's tiles run side by side on the same weight rows, each read
    /// once for up to eight slots. A block past the table's count returns; a
    /// tile word `tile_at` refuses raises [`FaultSite::ExpertId`].
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

    /// The card sum of `m` tokens over the stage card's and the expert
    /// tier's slots: thread `i < m·n` (token `t = i / n`, value `d = i % n`)
    /// writes [`card_sum_elem`] of its six slots to `acc[i]` — a slot whose
    /// place `sel[6t + j]` is below `n_card` read from `down`, else one whose
    /// tier place `tsel[6t + j]` is below `n_tier` read from `trows` (the
    /// tier's down outputs, slot-major as `down`), each at its slot's turn in
    /// slot order: the step's tier combine (`combine_post_tier_at`, the same
    /// mask), the one-card combine over the union of the two cards' slots.
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
            trows.len() >= 6 * n * m,
            w.len() >= 6 * m,
            sel.len() >= 6 * m,
            tsel.len() >= 6 * m,
            acc.len() >= n * m
        )
    )]
    pub fn ds41_ffn_card_acc_tier(
        down: &[f32],
        trows: &[f32],
        w: &[f32],
        sel: &[u32],
        tsel: &[u32],
        n: u32,
        m: u32,
        n_card: u32,
        n_tier: u32,
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
            // SAFETY: s < 6m <= sel.len(), tsel.len() and w.len() by the
            // launch contract.
            let (place, tplace, ws) = unsafe {
                (
                    *sel.get_unchecked(s),
                    *tsel.get_unchecked(s),
                    *w.get_unchecked(s),
                )
            };
            if place < n_card {
                card[j] = true;
                wv[j] = ws;
                // SAFETY: s < 6m and d < n, so s·n + d < 6nm <= down.len().
                dv[j] = unsafe { *down.get_unchecked(s * n + d) };
            } else if tplace < n_tier {
                card[j] = true;
                wv[j] = ws;
                // SAFETY: s < 6m and d < n, so s·n + d < 6nm <= trows.len().
                dv[j] = unsafe { *trows.get_unchecked(s * n + d) };
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

    /// [`ds41_card_gather`] of eight slots a token ([`gather_at`]): entry
    /// `j`'s token column is `col0 + order[j] / 8`.
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
    pub fn ds41_card_gather_8(
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
        let a = GatherIn {
            q_in,
            d8_in,
            order,
            start,
            n_experts,
            n_slots,
            col0,
            cols,
            n_sb,
            iters,
            fault,
        };
        // SAFETY: the launch contract is `gather_at`'s.
        unsafe { gather_at::<SLOTS_8>(&a, &mut q_out, &mut d8_out) };
    }

    /// [`ds41_ffn_card_acc`] of eight slots a token ([`card_acc_at`]).
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
            down.len() >= 8 * n * m,
            w.len() >= 8 * m,
            sel.len() >= 8 * m,
            acc.len() >= n * m
        )
    )]
    pub fn ds41_ffn_card_acc_8(
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
        // SAFETY: i < n·m, and the launch contract gives every length the
        // helper's contract asks for at SLOTS_8 slots.
        let v = unsafe { card_acc_at::<SLOTS_8>(down, w, sel, n, n_card, i) };
        // SAFETY: i < n·m <= acc.len(); thread i is acc[i]'s only writer.
        unsafe { *acc.get_unchecked_mut(i) = v };
    }
}

/// What a gather entry of `N` slots a token reads ([`gather_at`]): its
/// arguments, as [`ffn_batch_kernels::ds41_card_gather`] names them.
struct GatherIn<'a> {
    q_in: &'a [u64],
    d8_in: &'a [f32],
    order: &'a [u32],
    start: &'a [u32],
    n_experts: u32,
    n_slots: u32,
    col0: u32,
    cols: u32,
    n_sb: u32,
    iters: u32,
    fault: FaultSink,
}

/// The gather of `N` slots a token, block `j` per table entry:
/// `ds41_card_gather`'s rule with `N` in place of six — entry `j` below the
/// count `start[n_experts]` copies token column `col0 + order[j] / N` of the
/// q8_1 planes to column `j`; an entry that names no slot of the block raises
/// [`FaultSite::ExpertId`] on `fault` and is skipped.
///
/// SAFETY: `q_in.len() >= 64 · iters · cols`, `d8_in.len() >= 2 · n_sb ·
/// cols`, `order.len() >= n_slots`, `start.len() >= n_experts + 1`,
/// `q_out.len() >= 64 · iters · n_slots`, `d8_out.len() >= 2 · n_sb ·
/// n_slots`; one block per entry, [`THREADS`] threads a block, as the
/// launch's index.
#[inline(always)]
unsafe fn gather_at<const N: usize>(
    a: &GatherIn<'_>,
    q_out: &mut DisjointSlice<u64>,
    d8_out: &mut DisjointSlice<f32>,
) {
    let j = thread::blockIdx_x() as usize;
    let tid = thread::threadIdx_x() as usize;
    // SAFETY: n_experts < start.len() by this fn's contract.
    let count = unsafe { *a.start.get_unchecked(a.n_experts as usize) } as usize;
    // Block-uniform: j, the count and the entry are the block's.
    if j >= a.n_slots as usize || j >= count {
        return;
    }
    // SAFETY: j < n_slots <= order.len() by this fn's contract.
    let slot = unsafe { *a.order.get_unchecked(j) } as usize;
    let col = a.col0 as usize + slot / N;
    if slot >= a.n_slots as usize || col >= a.cols as usize {
        if tid == 0 {
            a.fault.raise(FaultSite::ExpertId);
        }
        return;
    }
    let (qc, dc) = (64 * a.iters as usize, 2 * a.n_sb as usize);
    let mut k = tid;
    while k < qc {
        // SAFETY: col < cols and j < n_slots with k < qc, inside both
        // buffers by this fn's contract; thread tid of block j is the only
        // writer of value k of column j.
        unsafe { *q_out.get_unchecked_mut(j * qc + k) = *a.q_in.get_unchecked(col * qc + k) };
        k += THREADS as usize;
    }
    let mut k = tid;
    while k < dc {
        // SAFETY: as above, with dc values a column.
        unsafe { *d8_out.get_unchecked_mut(j * dc + k) = *a.d8_in.get_unchecked(col * dc + k) };
        k += THREADS as usize;
    }
}

/// The card slots' sum of value `i` (token `t = i / n`, value `d = i % n`)
/// of `N` slots a token: `acc = fma(down[(Nt + j)·n + d], w[Nt + j], acc)`
/// from zero for each slot `j` below `N` whose place `sel[Nt + j]` is below
/// `n_card`, in ascending `j` — the order `runtime::combine::card_sum` pins (this
/// order is the gate); no other slot's rows are read.
///
/// SAFETY: `i < n · m` for an `m` with `down.len() >= N · n · m`, `w.len()`
/// and `sel.len() >= N · m`.
#[inline(always)]
unsafe fn card_acc_at<const N: usize>(
    down: &[f32],
    w: &[f32],
    sel: &[u32],
    n: usize,
    n_card: u32,
    i: usize,
) -> f32 {
    let (t, d) = (i / n, i % n);
    let mut acc = 0.0f32;
    for j in 0..N {
        cuda_device::thread::__unroll_config::<0>();
        let s = t * N + j;
        // SAFETY: s < N·m <= sel.len() and w.len() by this fn's contract.
        let (place, ws) = unsafe { (*sel.get_unchecked(s), *w.get_unchecked(s)) };
        if place < n_card {
            // SAFETY: s < N·m and d < n, so s·n + d < N·n·m <= down.len().
            let ds = unsafe { *down.get_unchecked(s * n + d) };
            acc = ds.mul_add(ws, acc);
        }
    }
    acc
}

/// What the tile path reads for one block ([`enqueue_tiled_experts`]): the
/// layer's routed stacks on the card, the q8_1 planes of the token columns
/// `col0 .. cols` of `q3` and `d8` (a column as a [`Q8Act`] lays one out),
/// each of the block's `N_USED · (cols − col0)` slots' place in the stacks
/// (`sel`: a stack slot, or [`HOST`] for a slot another device computes),
/// the layer and its SwiGLU clamp.
pub struct TiledBlock<'a> {
    pub stacks: CardStacks<'a>,
    pub q3: &'a DeviceBuffer<u64>,
    pub d8: &'a DeviceBuffer<f32>,
    pub col0: usize,
    pub cols: usize,
    pub sel: &'a DeviceBuffer<u32>,
    pub layer: usize,
    pub limit: f32,
}

/// A block's routed experts on one card by tile items: the slots of `t.sel`
/// grouped by expert (`ds41_card_buckets`), the table's tiles
/// (`q4k_sel::grouped_tiles`), the q8_1 planes gathered by table entry
/// (`ds41_card_gather`), the gate·up (`ds41_expert_gate_up_tiles`) into the
/// SwiGLU outputs by entry, their q8_1 form (`quantize_ord`) and the down
/// (`q4k_gemv_tiles`), which scatters each entry's rows to its slot's rows
/// of `down` (slot-major, `n_embd` a slot); a slot whose place is [`HOST`]
/// is left as it was. Each weight row is read once for up to eight slots.
/// The table's size is a device value: no launch here reads it on the host.
/// The stage card's batch shadow and the expert tier's batch service both
/// run it, each over its own scratch. Asynchronous, allocation-free.
pub fn enqueue_tiled_experts(
    gpu: &Gpu,
    kernels: &FfnBatchKernels,
    t: &TiledBlock<'_>,
    s: &mut TileScratch,
    down: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let (stream, ff, n, layer) = (gpu.stream(), s.ff, s.n_embd, t.layer);
    let fault = gpu.layer_sink(layer)?;
    let n_sb = n / 256;
    let slots_n = t.cols.saturating_sub(t.col0) * N_USED;
    let n_experts = t.stacks.gate.rows() / ff;
    if slots_n == 0 || slots_n > s.slots || t.sel.len() < slots_n || down.len() < slots_n * n {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "layer {layer}'s tiled block of columns {}..{}: {} places, {} down values; the \
                 scratch takes 1..={} slots",
                t.col0,
                t.cols,
                t.sel.len(),
                down.len(),
                s.slots
            ),
        });
    }
    if n_experts > BUCKET_EXPERTS {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "layer {layer}'s card stack of {n_experts} experts; the bucket kernel takes at \
                 most {BUCKET_EXPERTS}"
            ),
        });
    }
    {
        let mut order = span_mut(WHAT, &mut s.order, 0, slots_n)?;
        kernels.enqueue_buckets(
            stream,
            t.sel,
            slots_n,
            n_experts,
            fault,
            &mut order,
            &mut s.start,
        )?;
    }
    gpu.q4k_sel().enqueue_grouped_tiles(
        stream,
        &s.start,
        n_experts,
        slots_n,
        fault,
        &mut s.tiles,
    )?;
    let order = span(WHAT, &s.order, 0, slots_n)?;
    {
        let g = CardGather {
            q3: t.q3,
            d8: t.d8,
            order: &order,
            start: &s.start,
            n_experts,
            n_slots: slots_n,
            col0: t.col0,
            cols: t.cols,
            n_sb,
            n_used: N_USED,
        };
        let mut q3 = span_mut(WHAT, &mut s.q3_ord, 0, slots_n * 64 * n_sb.div_ceil(2))?;
        let mut d8 = span_mut(WHAT, &mut s.d8_ord, 0, slots_n * 2 * n_sb)?;
        kernels.enqueue_card_gather(stream, &g, fault, &mut q3, &mut d8)?;
    }
    let q3 = span(WHAT, &s.q3_ord, 0, slots_n * 64 * n_sb.div_ceil(2))?;
    let d8 = span(WHAT, &s.d8_ord, 0, slots_n * 2 * n_sb)?;
    let mut h = span_mut(WHAT, &mut s.h, 0, slots_n * ff)?;
    let g = TiledGateUp {
        wg: t.stacks.gate.buf(),
        wu: t.stacks.up.buf(),
        q3: &q3,
        d8: &d8,
        start: &s.start,
        tiles: &s.tiles,
        n_experts,
        rows_per_expert: ff,
        n_slots: slots_n,
        n_sb,
        limit: t.limit,
    };
    kernels.enqueue_gate_up_tiles(stream, &g, fault, &mut h)?;
    drop(h);
    gpu.q4k_sel().enqueue_quantize_ord(
        stream,
        &s.h,
        &s.start,
        n_experts,
        slots_n,
        fault,
        &mut s.act_h,
    )?;
    gpu.q4k_sel().enqueue_gemv_q4k_tiles(
        stream,
        t.stacks.down,
        &s.act_h,
        &order,
        &s.start,
        &s.tiles,
        slots_n,
        n,
        fault,
        down,
    )
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

/// What [`FfnBatchKernels::enqueue_gate_up_tiles`] reads: both Q3_K stacks
/// of `n_experts` card experts of `rows_per_expert` rows, the q8_1 planes of
/// the table's `n_slots` entries of `n_sb` super-blocks (entry `j` the
/// activation of slot `order[j]`, [`ds41_card_gather`]), the table's runs
/// `start` and its tiles (`q4k_sel::grouped_tiles`); `limit` is the SwiGLU
/// clamp.
pub struct TiledGateUp<'a> {
    pub wg: &'a DeviceBuffer<u32>,
    pub wu: &'a DeviceBuffer<u32>,
    pub q3: &'a DeviceBuffer<u64>,
    pub d8: &'a DeviceBuffer<f32>,
    pub start: &'a DeviceBuffer<u32>,
    pub tiles: &'a DeviceBuffer<u32>,
    pub n_experts: usize,
    pub rows_per_expert: usize,
    pub n_slots: usize,
    pub n_sb: usize,
    pub limit: f32,
}

/// What [`FfnBatchKernels::enqueue_card_gather`] reads: the block's q8_1
/// planes by token (`q3`, `d8`, `cols` columns of `n_sb` super-blocks, the
/// block's tokens from `col0`) and the table of its `n_slots` slots
/// (`order`, runs `start` of `n_experts` experts), `n_used` slots a token.
pub struct CardGather<'a> {
    pub q3: &'a DeviceBuffer<u64>,
    pub d8: &'a DeviceBuffer<f32>,
    pub order: &'a DeviceBuffer<u32>,
    pub start: &'a DeviceBuffer<u32>,
    pub n_experts: usize,
    pub n_slots: usize,
    pub col0: usize,
    pub cols: usize,
    pub n_sb: usize,
    pub n_used: usize,
}

/// What [`FfnBatchKernels::enqueue_card_acc`] reads, `m` tokens of `n_used`
/// slots: `down` the slots' down outputs slot-major (`n` each, token `t`'s
/// slots `n_used · t ..`), `w` their routing weights and `sel` their places
/// (the card's below `n_card`; no other slot's rows are read).
pub struct CardAcc<'a> {
    pub down: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub sel: &'a DeviceBuffer<u32>,
    pub n: usize,
    pub m: usize,
    pub n_card: usize,
    pub n_used: usize,
}

/// What [`FfnBatchKernels::enqueue_card_acc_tier`] reads, `m` tokens of
/// `n_used` slots: [`CardAcc`]'s, and the expert tier's down outputs `trows`
/// (slot-major as `down`) and each slot's tier place `tsel` (the tier's
/// below `n_tier`, the layer's tier experts; no other slot's rows are read).
pub struct CardAccTier<'a> {
    pub down: &'a DeviceBuffer<f32>,
    pub trows: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub sel: &'a DeviceBuffer<u32>,
    pub tsel: &'a DeviceBuffer<u32>,
    pub n: usize,
    pub m: usize,
    pub n_card: usize,
    pub n_tier: usize,
    pub n_used: usize,
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

    /// Enqueue `ds41_ffn_replace`: the `p.n` places `sel` the route wrote
    /// under the map before a call's pick, split under the map now into
    /// `sel` (every slot but the admitted ids') and `ssel` (the admitted
    /// ids'). One launch. Asynchronous, allocation-free.
    pub fn enqueue_replace(
        &self,
        stream: &CudaStream,
        p: &Places<'_>,
        fault: FaultSink,
        sel: &mut DeviceBuffer<u32>,
        ssel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let what = "ds41_ffn_replace";
        let grid = launch_u32(what, "grid", p.n.div_ceil(THREADS as usize))?;
        let prep = self
            .module
            .prepare_ds41_ffn_replace(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.ds41_ffn_replace(
            stream,
            &prep,
            p.ids,
            p.map,
            launch_u32(what, "row_off", p.row_off)?,
            launch_u32(what, "n_expert", p.n_expert)?,
            launch_u32(what, "n", p.n)?,
            fault,
            sel,
            ssel,
        )?;
        Ok(())
    }

    /// Enqueue `ds41_ffn_sel_merge`: the `n` streamed places `ssel` that are
    /// not [`HOST`] into `sel`. One launch. Asynchronous, allocation-free.
    pub fn enqueue_sel_merge(
        &self,
        stream: &CudaStream,
        ssel: &DeviceBuffer<u32>,
        n: usize,
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        let what = "ds41_ffn_sel_merge";
        let grid = launch_u32(what, "grid", n.div_ceil(THREADS as usize))?;
        let prep = self
            .module
            .prepare_ds41_ffn_sel_merge(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module
            .ds41_ffn_sel_merge(stream, &prep, ssel, launch_u32(what, "n", n)?, sel)?;
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

    /// Enqueue the gather of `g.n_used` slots a token (`ds41_card_gather` for
    /// 6, `ds41_card_gather_8` for 8; any other count is refused by name):
    /// the q8_1 planes of `g`'s table entries into columns `0 .. g.n_slots`
    /// of `q3` and `d8`. One launch. Asynchronous, allocation-free.
    pub fn enqueue_card_gather(
        &self,
        stream: &CudaStream,
        g: &CardGather<'_>,
        fault: FaultSink,
        q3: &mut DeviceBuffer<u64>,
        d8: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let slots = slots_of("ds41_card_gather", g.n_used)?;
        let what = match slots {
            Slots::Six => "ds41_card_gather",
            Slots::Eight => "ds41_card_gather_8",
        };
        let cfg = LaunchConfig1D::new(launch_u32(what, "grid", g.n_slots)?, THREADS, 0);
        let v = [
            launch_u32(what, "n_experts", g.n_experts)?,
            launch_u32(what, "n_slots", g.n_slots)?,
            launch_u32(what, "col0", g.col0)?,
            launch_u32(what, "cols", g.cols)?,
            launch_u32(what, "n_sb", g.n_sb)?,
            launch_u32(what, "iters", g.n_sb.div_ceil(2))?,
        ];
        let m = &self.module;
        match slots {
            Slots::Six => {
                let prep = m.prepare_ds41_card_gather(cfg)?;
                m.ds41_card_gather(
                    stream, &prep, g.q3, g.d8, g.order, g.start, v[0], v[1], v[2], v[3], v[4],
                    v[5], fault, q3, d8,
                )?;
            }
            Slots::Eight => {
                let prep = m.prepare_ds41_card_gather_8(cfg)?;
                m.ds41_card_gather_8(
                    stream, &prep, g.q3, g.d8, g.order, g.start, v[0], v[1], v[2], v[3], v[4],
                    v[5], fault, q3, d8,
                )?;
            }
        }
        Ok(())
    }

    /// Enqueue the card sums of `a.m` tokens of `a.n_used` slots
    /// (`ds41_ffn_card_acc` for 6, `ds41_ffn_card_acc_8` for 8; any other
    /// count is refused by name): value `i` of token `t = i / a.n` into
    /// `acc[i]`, the card half of the combine. One launch. Asynchronous,
    /// allocation-free.
    pub fn enqueue_card_acc(
        &self,
        stream: &CudaStream,
        a: &CardAcc<'_>,
        acc: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let slots = slots_of("ds41_ffn_card_acc", a.n_used)?;
        let what = match slots {
            Slots::Six => "ds41_ffn_card_acc",
            Slots::Eight => "ds41_ffn_card_acc_8",
        };
        let grid = launch_u32(what, "grid", (a.m * a.n).div_ceil(THREADS as usize))?;
        let cfg = LaunchConfig1D::new(grid, THREADS, 0);
        let (n, m, n_card) = (
            launch_u32(what, "n", a.n)?,
            launch_u32(what, "m", a.m)?,
            launch_u32(what, "n_card", a.n_card)?,
        );
        match slots {
            Slots::Six => {
                let prep = self.module.prepare_ds41_ffn_card_acc(cfg)?;
                self.module
                    .ds41_ffn_card_acc(stream, &prep, a.down, a.w, a.sel, n, m, n_card, acc)?;
            }
            Slots::Eight => {
                let prep = self.module.prepare_ds41_ffn_card_acc_8(cfg)?;
                self.module
                    .ds41_ffn_card_acc_8(stream, &prep, a.down, a.w, a.sel, n, m, n_card, acc)?;
            }
        }
        Ok(())
    }

    /// Enqueue `ds41_ffn_card_acc_tier`: the card sums of `a.m` tokens over
    /// the stage card's and the tier's slots, value `i` of token `t = i / a.n`
    /// into `acc[i]`. Six slots a token; any other count is refused by name.
    /// One launch. Asynchronous, allocation-free.
    pub fn enqueue_card_acc_tier(
        &self,
        stream: &CudaStream,
        a: &CardAccTier<'_>,
        acc: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "ds41_ffn_card_acc_tier";
        if a.n_used != N_USED {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("{what} of {} slots a token; it takes {N_USED}", a.n_used),
            });
        }
        let grid = launch_u32(what, "grid", (a.m * a.n).div_ceil(THREADS as usize))?;
        let prep = self
            .module
            .prepare_ds41_ffn_card_acc_tier(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.ds41_ffn_card_acc_tier(
            stream,
            &prep,
            a.down,
            a.trows,
            a.w,
            a.sel,
            a.tsel,
            launch_u32(what, "n", a.n)?,
            launch_u32(what, "m", a.m)?,
            launch_u32(what, "n_card", a.n_card)?,
            launch_u32(what, "n_tier", a.n_tier)?,
            acc,
        )?;
        Ok(())
    }

    /// Enqueue `ds41_expert_gate_up_tiles`: the SwiGLU outputs of `g`'s
    /// table entries by tiles, entry `j`'s rows into `h[j ·
    /// g.rows_per_expert ..]`. One launch of `tile_cap · row_tiles`
    /// blocks; the tile count is the table's, a device value, and a block
    /// past it returns. Asynchronous, allocation-free.
    pub fn enqueue_gate_up_tiles(
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
/// them in order); and the host copies are the host tier's batch port's
/// ([`Hybrid::prepare_batch`]), in two sets, since the host reads them outside
/// the stream's order.
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
    /// `m − 1`), its q8_1 form; the HC_PRE mixes; the shared expert's SwiGLU
    /// output (token-major), its q8_1 form per token count and its down
    /// output row-major.
    normed: DeviceBuffer<f32>,
    act_x: Vec<Q8Act>,
    mixes: DeviceBuffer<f32>,
    sh_h: DeviceBuffer<f32>,
    act_sh: Vec<Q8Act>,
    sh_raw: DeviceBuffer<f32>,
    /// The block's card experts: per token the norm's q8_1 planes (q3 and
    /// d8, as the chunk's scratch lays out a column), the tile path's scratch
    /// ([`TileScratch`]) and the down output by slot.
    q3_all: DeviceBuffer<u64>,
    d8_all: DeviceBuffer<f32>,
    down_all: DeviceBuffer<f32>,
    tile: TileScratch,
    /// With an expert tier, per token each slot's place in the tier's routed
    /// stacks, which the route writes and the tier computes by; and per
    /// batch of a group the shadow's copy of a tiered block's routing — the
    /// weights, the card places and the tier places — which the card sum in
    /// the post reads: in a group the next layer-batch's route rewrites the
    /// batch-wide ones before this one's post.
    tsel: Option<DeviceBuffer<u32>>,
    held: Vec<HeldRouting>,
    /// Under host streaming, per token each slot's streamed place
    /// ([`FfnPiece::enqueue_batch_replace`]); made by
    /// [`FfnBatch::enable_stream`].
    ssel: Option<DeviceBuffer<u32>>,
}

/// A tiered block's routing as the shadow left it for the post's card sum
/// ([`FfnPiece::enqueue_batch_acc_tier`]), a batch's own, token `t`'s six
/// slots at `6t ..` as in the batch-wide buffers.
struct HeldRouting {
    w: DeviceBuffer<f32>,
    sel: DeviceBuffer<u32>,
    tsel: DeviceBuffer<u32>,
}

/// The tile path's scratch for blocks of up to `slots` routed slots
/// ([`enqueue_tiled_experts`]): the slots grouped by expert (`order`, runs
/// `start`) and the table's tiles, the q8_1 planes gathered by table entry
/// (`q3_ord`, `d8_ord`), the SwiGLU output and its q8_1 form by entry —
/// column `j` is slot `order[j]`'s. The stage card's batch and the expert
/// tier each own one; the path is the same.
pub struct TileScratch {
    order: DeviceBuffer<u32>,
    start: DeviceBuffer<u32>,
    tiles: DeviceBuffer<u32>,
    q3_ord: DeviceBuffer<u64>,
    d8_ord: DeviceBuffer<f32>,
    h: DeviceBuffer<f32>,
    act_h: Q8Act,
    slots: usize,
    n_embd: usize,
    ff: usize,
}

impl TileScratch {
    /// The scratch for blocks of up to `slots` slots (1..=3072, six a token of
    /// the host union's columns) over rows of `n_embd` through experts of
    /// `ff`, on `stream`'s card. Load-time only.
    pub fn new(
        stream: &CudaStream,
        slots: usize,
        n_embd: usize,
        ff: usize,
    ) -> Result<TileScratch, GpuError> {
        let n_sb = n_embd / 256;
        Ok(TileScratch {
            order: DeviceBuffer::zeroed(stream, slots)?,
            start: DeviceBuffer::zeroed(stream, BUCKET_EXPERTS + 1)?,
            tiles: DeviceBuffer::zeroed(stream, tile_cap(slots, BUCKET_EXPERTS) + 1)?,
            q3_ord: DeviceBuffer::zeroed(stream, slots * 64 * n_sb.div_ceil(2))?,
            d8_ord: DeviceBuffer::zeroed(stream, slots * 2 * n_sb)?,
            h: DeviceBuffer::zeroed(stream, slots * ff)?,
            act_h: Q8Act::with_slots(stream, slots, ff)?,
            slots,
            n_embd,
            ff,
        })
    }

    /// Device bytes of the scratch.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.order.num_bytes()
            + self.start.num_bytes()
            + self.tiles.num_bytes()
            + self.q3_ord.num_bytes()
            + self.d8_ord.num_bytes()
            + self.h.num_bytes()
            + q8act_bytes(&self.act_h)
    }
}

impl FfnBatch {
    /// The buffers for groups of up to `sets` batches of up to `cap` tokens
    /// (at most [`UNION_MAX_COLS`], the host union's columns) of rows of
    /// `n_embd` through experts of `ff`, on `gpu`; with `tier`, the tier
    /// places of a model with an expert tier.
    pub fn new(
        gpu: &Gpu,
        n_embd: usize,
        ff: usize,
        [cap, sets]: [usize; 2],
        tier: bool,
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
            sh_h: z(c * ff)?,
            act_sh: counts
                .map(|m| Q8Act::with_k(stream, m, ff))
                .collect::<Result<_, _>>()?,
            sh_raw: z(c * n_embd)?,
            q3_all: DeviceBuffer::zeroed(stream, cap * 64 * n_sb.div_ceil(2))?,
            d8_all: z(cap * 2 * n_sb)?,
            down_all: z(slot_cols * n_embd)?,
            tile: TileScratch::new(stream, slot_cols, n_embd, ff)?,
            tsel: if tier {
                Some(DeviceBuffer::zeroed(stream, slot_cols)?)
            } else {
                None
            },
            held: if tier {
                (0..sets)
                    .map(|_| -> Result<HeldRouting, GpuError> {
                        Ok(HeldRouting {
                            w: z(slot_cols)?,
                            sel: DeviceBuffer::zeroed(stream, slot_cols)?,
                            tsel: DeviceBuffer::zeroed(stream, slot_cols)?,
                        })
                    })
                    .collect::<Result<_, _>>()?
            } else {
                Vec::new()
            },
            ssel: None,
        })
    }

    /// The streamed places a prompt call under host streaming writes, one a
    /// slot of the batch; a second call is refused by name. Load-time only.
    pub fn enable_stream(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        if self.ssel.is_some() {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a batch without its streamed places (FfnBatch::enable_stream once)",
            });
        }
        self.ssel = Some(DeviceBuffer::zeroed(stream, self.cap * N_USED)?);
        Ok(())
    }

    /// Whether the batch holds the streamed places.
    #[must_use]
    pub fn streams(&self) -> bool {
        self.ssel.is_some()
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
            &self.sh_h,
            &self.sh_raw,
            &self.d8_all,
            &self.down_all,
        ];
        let acts = self.act_x.iter().chain(&self.act_sh);
        f32s.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.hc.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.ids.num_bytes()
            + self.sel.num_bytes()
            + self.q3_all.num_bytes()
            + self.tile.device_bytes()
            + self.tsel.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.ssel.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self
                .held
                .iter()
                .map(|h| h.w.num_bytes() + h.sel.num_bytes() + h.tsel.num_bytes())
                .sum::<usize>()
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
    /// host into the host tier's next batch set, and the event they complete
    /// at ([`Hybrid::enqueue_download`]); for a `tiered` layer the slots'
    /// tier places too ([`Hybrid::enqueue_download_tiered`]), which a batch
    /// made without them refuses by name. Asynchronous. Refused while both
    /// sets hold a layer not uploaded yet.
    pub fn enqueue_download<H: HostExperts>(
        &self,
        gpu: &Gpu,
        tier: &mut Hybrid<H>,
        key: BatchKey,
        tiered: bool,
    ) -> Result<(), GpuError> {
        self.check_key(key)?;
        let xw = [&self.x, &self.weights];
        if !tiered {
            return tier.enqueue_download(gpu.stream(), xw, &self.ids, key);
        }
        let tsel = self.tsel()?;
        tier.enqueue_download_tiered(gpu.stream(), xw, &self.ids, &[tsel], key)
    }

    /// The tier places, refused by name on a batch made without a tier.
    fn tsel(&self) -> Result<&DeviceBuffer<u32>, GpuError> {
        self.tsel.as_ref().ok_or(GpuError::State {
            what: WHAT,
            missing: "the tier places of a batch made with a tier (FfnBatch::new)",
        })
    }

    /// Wait for the oldest download not served yet — `key`'s, else refused
    /// by name — then serve its layer's host experts for its tokens in one
    /// union call ([`Hybrid::serve_key`]): the sums into the set's host copy
    /// [`FfnBatch::enqueue_upload`] sends back. Returns the host time
    /// outside the union call.
    pub fn serve<H: HostExperts>(
        &self,
        tier: &mut Hybrid<H>,
        key: BatchKey,
    ) -> Result<ServeTimes, GpuError> {
        self.check_key(key)?;
        tier.serve_key(key)
    }

    /// Enqueue the copy of the oldest served set's host sums — `key`'s, else
    /// refused by name — to the card ([`Hybrid::enqueue_upload`]).
    /// Asynchronous.
    pub fn enqueue_upload<H: HostExperts>(
        &mut self,
        gpu: &Gpu,
        tier: &mut Hybrid<H>,
        key: BatchKey,
    ) -> Result<(), GpuError> {
        self.check_key(key)?;
        tier.enqueue_upload(gpu.stream(), &mut self.hsum, key)
    }

    /// A key's tokens `at .. u` of a batch: at least one, at most the cap;
    /// its batch within the group's sets.
    fn check_key(&self, key: BatchKey) -> Result<(), GpuError> {
        let BatchKey { set, at, u, .. } = key;
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

/// The group's batch `set`'s held routing, refused past the sets and on a
/// batch made without a tier.
fn held_of(held: &[HeldRouting], set: usize) -> Result<&HeldRouting, GpuError> {
    held.get(set).ok_or_else(|| GpuError::Shape {
        what: WHAT,
        detail: format!(
            "the held routing of batch {set} of a group of {} (none on a batch made without a tier)",
            held.len()
        ),
    })
}

/// Enqueue the copies of slots `from .. from + len` of the batch-wide
/// weights, card places and tier places into the group's batch `set`'s held
/// routing, at the same slots: three copies.
fn hold_routing(
    stream: &CudaStream,
    b: &mut FfnBatch,
    set: usize,
    from: usize,
    len: usize,
) -> Result<(), GpuError> {
    let FfnBatch {
        weights,
        sel,
        tsel,
        held,
        ..
    } = b;
    let n = held.len();
    let h = held.get_mut(set).ok_or_else(|| GpuError::Shape {
        what: WHAT,
        detail: format!("the held routing of batch {set} of a group of {n}"),
    })?;
    let tsel = tsel.as_ref().ok_or(GpuError::State {
        what: WHAT,
        missing: "the tier places of a batch made with a tier (FfnBatch::new)",
    })?;
    let (w, s, t) = (
        span(WHAT, weights, from, len)?,
        span(WHAT, sel, from, len)?,
        span(WHAT, tsel, from, len)?,
    );
    // SAFETY: each copy reads the batch-wide buffers and writes this batch's
    // held copies, all this piece's buffers on the engine stream: the route
    // that wrote the slots comes before, the next route that rewrites them
    // and the post's card sum that reads the copies after, in stream order.
    unsafe {
        dtod(stream, &mut h.w, from, &w, len)?;
        dtod(stream, &mut h.sel, from, &s, len)?;
        dtod(stream, &mut h.tsel, from, &t, len)?;
    }
    Ok(())
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

    /// The expert tier's places of the block `io`'s slots, after the batch's
    /// route of it: `ds41_ffn_places` over the route's ids with `tier_map`,
    /// the slot map's tier copy (`SlotMap::tier_view`, same row offsets as
    /// the card copy), into the batch's tier places. One launch.
    /// Asynchronous, allocation-free.
    pub fn enqueue_batch_tier_places(
        &self,
        gpu: &Gpu,
        bl: &BatchLayer<'_>,
        b: &mut FfnBatch,
        io: &BlockIo<'_>,
        tier_map: &DeviceTensor<u32>,
    ) -> Result<(), GpuError> {
        let layer = bl.layer;
        let (i, at, u) = self.batch_block(layer, b, io)?;
        let c = &self.cfg[i];
        let fault = gpu.layer_sink(layer)?;
        let slots_n = (u - at) * N_USED;
        let ids = span(WHAT, &b.ids, at * N_USED, slots_n)?;
        let tsel = b.tsel.as_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the tier places of a batch made with a tier (FfnBatch::new)",
        })?;
        let mut tsel = span_mut(WHAT, tsel, at * N_USED, slots_n)?;
        let places = Places {
            ids: &ids,
            n: slots_n,
            map: tier_map.buf(),
            row_off: c.row_off,
            n_expert: self.n_expert,
        };
        b.kernels
            .enqueue_places(gpu.stream(), &places, fault, &mut tsel)
    }

    /// The card sum of layer `layer`'s tiered block, tokens `at .. u` of the
    /// group's batch `set`, once the tier's rows have landed (`trows`: the
    /// tier's down outputs of the block's slots, slot-major from the block's
    /// first slot, as the host tier hands them over after the tier's
    /// service): each token's six slots in slot order, from the card's down
    /// outputs or the tier's rows by their places — the tier's below `n_tier`,
    /// the layer's tier experts — with the weights and places the shadow held
    /// for the batch. Before the join, in place of the
    /// shadow's card sum. One launch. Asynchronous, allocation-free.
    pub fn enqueue_batch_acc_tier(
        &self,
        gpu: &Gpu,
        b: &mut FfnBatch,
        [layer, set, n_tier]: [usize; 3],
        [at, u]: [usize; 2],
        trows: &DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let i = self.layer_index(layer, WHAT)?;
        let n = self.n_embd;
        if at >= u || u > b.cap {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!(
                    "layer {layer}'s tier card sum of tokens {at}..{u} (batches of {})",
                    b.cap
                ),
            });
        }
        let (m, slots_n) = (u - at, (u - at) * N_USED);
        let held = held_of(&b.held, set)?;
        let tsel = span(WHAT, &held.tsel, at * N_USED, slots_n)?;
        let down = span(WHAT, &b.down_all, at * N_USED * n, slots_n * n)?;
        let trows = span(WHAT, trows, 0, slots_n * n)?;
        let sel = span(WHAT, &held.sel, at * N_USED, slots_n)?;
        let wts = span(WHAT, &held.w, at * N_USED, slots_n)?;
        let mut acc = span_mut(WHAT, &mut b.acc, at * n, m * n)?;
        let a = CardAccTier {
            down: &down,
            trows: &trows,
            w: &wts,
            sel: &sel,
            tsel: &tsel,
            n,
            m,
            n_card: self.cfg[i].n_card,
            n_tier,
            n_used: N_USED,
        };
        b.kernels.enqueue_card_acc_tier(gpu.stream(), &a, &mut acc)
    }

    /// The shadow work of `bl`'s layer for the block `io`, after the batch's
    /// route of it: each chunk's HC_PRE, norm and shared expert; then over the
    /// block the card slots grouped by expert, the gate·up, its q8_1 form and
    /// the down by tile items, each reading a weight row once for up to eight
    /// slots, and — unless the layer is `tiered` — the card sum.
    /// Asynchronous, allocation-free.
    pub fn enqueue_batch_shadow(
        &mut self,
        gpu: &Gpu,
        bl: &BatchLayer<'_>,
        card: Option<CardStacks<'_>>,
        b: &mut FfnBatch,
        io: &BlockIo<'_>,
        tiered: bool,
    ) -> Result<(), GpuError> {
        self.enqueue_batch_shadow_chunks(gpu, bl, card, b, io)?;
        self.enqueue_batch_block(gpu, bl, card, b, io, tiered)
    }

    /// The first part of [`FfnPiece::enqueue_batch_shadow`]: each chunk's
    /// HC_PRE, norm and shared expert, which read no place. Asynchronous,
    /// allocation-free.
    pub fn enqueue_batch_shadow_chunks(
        &mut self,
        gpu: &Gpu,
        bl: &BatchLayer<'_>,
        card: Option<CardStacks<'_>>,
        b: &mut FfnBatch,
        io: &BlockIo<'_>,
    ) -> Result<(), GpuError> {
        let (layer, lw) = (bl.layer, &bl.lw);
        let (i, _, _) = self.batch_block(layer, b, io)?;
        let card = self.check_card(layer, i, card)?;
        let n = self.n_embd;
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
            self.enqueue_batch_shadow_chunk(gpu, lw, i, card, b, layer, &chunk)?;
        }
        Ok(())
    }

    /// The second part of [`FfnPiece::enqueue_batch_shadow`]: the block's
    /// card experts by tile items and the card sum, or for a `tiered` layer
    /// the held routing ([`FfnPiece::enqueue_grouped_block`]). Asynchronous,
    /// allocation-free.
    pub fn enqueue_batch_block(
        &self,
        gpu: &Gpu,
        bl: &BatchLayer<'_>,
        card: Option<CardStacks<'_>>,
        b: &mut FfnBatch,
        io: &BlockIo<'_>,
        tiered: bool,
    ) -> Result<(), GpuError> {
        let layer = bl.layer;
        let (i, at, u) = self.batch_block(layer, b, io)?;
        let card = self.check_card(layer, i, card)?;
        self.enqueue_grouped_block(gpu, i, card, b, [layer, io.set], [at, u], tiered)
    }

    /// Under host streaming, the block's places again after the call's pick
    /// moved `bl`'s layer in `slots`, the map's card copy
    /// (`ds41_ffn_replace`): the ids the pick admitted into the streamed
    /// places, the rest into the block's places under the map now. One
    /// launch. Refused by name on a batch without streamed places.
    /// Asynchronous, allocation-free.
    pub fn enqueue_batch_replace(
        &self,
        gpu: &Gpu,
        bl: &BatchLayer<'_>,
        b: &mut FfnBatch,
        io: &BlockIo<'_>,
        slots: &DeviceTensor<u32>,
    ) -> Result<(), GpuError> {
        let layer = bl.layer;
        let (i, at, u) = self.batch_block(layer, b, io)?;
        let c = &self.cfg[i];
        let fault = gpu.layer_sink(layer)?;
        let slots_n = (u - at) * N_USED;
        let FfnBatch {
            kernels,
            ids,
            sel,
            ssel,
            ..
        } = b;
        let ssel = ssel.as_mut().ok_or(GpuError::State {
            what: WHAT,
            missing: "the streamed places of a batch under host streaming (FfnBatch::enable_stream)",
        })?;
        let ids = span(WHAT, ids, at * N_USED, slots_n)?;
        let mut sel = span_mut(WHAT, sel, at * N_USED, slots_n)?;
        let mut ssel = span_mut(WHAT, ssel, at * N_USED, slots_n)?;
        let places = Places {
            ids: &ids,
            n: slots_n,
            map: slots.buf(),
            row_off: c.row_off,
            n_expert: self.n_expert,
        };
        kernels.enqueue_replace(gpu.stream(), &places, fault, &mut sel, &mut ssel)
    }

    /// Under host streaming, the block's card experts but those the call's
    /// pick admitted, by tile items (the tile part of
    /// [`FfnPiece::enqueue_batch_block`] over the replaced places), after
    /// [`FfnPiece::enqueue_batch_replace`]; the streamed pass follows.
    /// Asynchronous, allocation-free.
    pub fn enqueue_batch_block_kept(
        &self,
        gpu: &Gpu,
        bl: &BatchLayer<'_>,
        card: Option<CardStacks<'_>>,
        b: &mut FfnBatch,
        io: &BlockIo<'_>,
    ) -> Result<(), GpuError> {
        let layer = bl.layer;
        let (i, at, u) = self.batch_block(layer, b, io)?;
        let card = self.check_card(layer, i, card)?;
        self.enqueue_block_tiles(gpu, i, card, b, layer, [at, u], false)
    }

    /// Under host streaming, the streamed pass of the block, once the engine
    /// stream waits for the pick's copies: the admitted ids' experts by tile
    /// items over the streamed places, those places merged into the block's
    /// (`ds41_ffn_sel_merge`), then the card sum in slot order over every
    /// card slot, or for a `tiered` layer the held routing. Asynchronous,
    /// allocation-free.
    pub fn enqueue_batch_stream(
        &self,
        gpu: &Gpu,
        bl: &BatchLayer<'_>,
        card: Option<CardStacks<'_>>,
        b: &mut FfnBatch,
        io: &BlockIo<'_>,
        tiered: bool,
    ) -> Result<(), GpuError> {
        let layer = bl.layer;
        let (i, at, u) = self.batch_block(layer, b, io)?;
        let card = self.check_card(layer, i, card)?;
        self.enqueue_block_tiles(gpu, i, card, b, layer, [at, u], true)?;
        let slots_n = (u - at) * N_USED;
        {
            let FfnBatch {
                kernels, sel, ssel, ..
            } = &mut *b;
            let ssel = ssel.as_ref().ok_or(GpuError::State {
                what: WHAT,
                missing: "the streamed places of a batch under host streaming \
                          (FfnBatch::enable_stream)",
            })?;
            let ssel = span(WHAT, ssel, at * N_USED, slots_n)?;
            let mut sel = span_mut(WHAT, sel, at * N_USED, slots_n)?;
            kernels.enqueue_sel_merge(gpu.stream(), &ssel, slots_n, &mut sel)?;
        }
        self.enqueue_block_finish(gpu, i, b, io.set, [at, u], tiered)
    }

    /// The block's launches for tokens `at .. u` of the group's batch `set`,
    /// after every chunk's part before them: with card experts, the tile path
    /// over the block's card slots ([`enqueue_tiled_experts`]) into the
    /// block's down outputs by slot; then the card sum of every token of the
    /// block — or, for a `tiered` layer, whose card sum waits for the tier's
    /// rows ([`FfnPiece::enqueue_batch_acc_tier`]), the copy of the block's
    /// weights, card places and tier places into the batch's held routing,
    /// three copies.
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
        [layer, set]: [usize; 2],
        [at, u]: [usize; 2],
        tiered: bool,
    ) -> Result<(), GpuError> {
        self.enqueue_block_tiles(gpu, i, card, b, layer, [at, u], false)?;
        self.enqueue_block_finish(gpu, i, b, set, [at, u], tiered)
    }

    /// With card experts, the tile path over the block's card slots
    /// ([`enqueue_tiled_experts`]) into the block's down outputs by slot:
    /// over the block's places, or with `streamed` over its streamed places.
    #[allow(
        clippy::too_many_arguments,
        reason = "the block's layer, stacks, buffers, tokens and places (rust-quality R8)"
    )]
    fn enqueue_block_tiles(
        &self,
        gpu: &Gpu,
        i: usize,
        card: Option<CardStacks<'_>>,
        b: &mut FfnBatch,
        layer: usize,
        [at, u]: [usize; 2],
        streamed: bool,
    ) -> Result<(), GpuError> {
        let n = self.n_embd;
        let c = &self.cfg[i];
        let slots_n = (u - at) * N_USED;
        if let Some(s) = card {
            let places = if streamed {
                b.ssel.as_ref().ok_or(GpuError::State {
                    what: WHAT,
                    missing: "the streamed places of a batch under host streaming \
                              (FfnBatch::enable_stream)",
                })?
            } else {
                &b.sel
            };
            let sel = span(WHAT, places, at * N_USED, slots_n)?;
            let mut down = span_mut(WHAT, &mut b.down_all, at * N_USED * n, slots_n * n)?;
            let t = TiledBlock {
                stacks: s,
                q3: &b.q3_all,
                d8: &b.d8_all,
                col0: at,
                cols: u,
                sel: &sel,
                layer,
                limit: c.limit,
            };
            enqueue_tiled_experts(gpu, &b.kernels, &t, &mut b.tile, &mut down)?;
        }
        Ok(())
    }

    /// The block's card sum of every token, or for a `tiered` layer, whose
    /// card sum waits for the tier's rows, the copy of the block's weights,
    /// card places and tier places into the batch's held routing.
    fn enqueue_block_finish(
        &self,
        gpu: &Gpu,
        i: usize,
        b: &mut FfnBatch,
        set: usize,
        [at, u]: [usize; 2],
        tiered: bool,
    ) -> Result<(), GpuError> {
        let (stream, n) = (gpu.stream(), self.n_embd);
        let c = &self.cfg[i];
        let slots_n = (u - at) * N_USED;
        if tiered {
            return hold_routing(stream, b, set, at * N_USED, slots_n);
        }
        let m = u - at;
        let down = span(WHAT, &b.down_all, at * N_USED * n, slots_n * n)?;
        let sel = span(WHAT, &b.sel, at * N_USED, slots_n)?;
        let wts = span(WHAT, &b.weights, at * N_USED, slots_n)?;
        let mut acc = span_mut(WHAT, &mut b.acc, at * n, m * n)?;
        let a = CardAcc {
            down: &down,
            w: &wts,
            sel: &sel,
            n,
            m,
            n_card: c.n_card,
            n_used: N_USED,
        };
        b.kernels.enqueue_card_acc(stream, &a, &mut acc)
    }

    /// Layer `layer`'s shadow work for the chunk `io`, after the batch's
    /// route: HC_PRE into the batch's results, the norm's q8_1 form again —
    /// with card experts copied into the block's planes, which the block's
    /// gather reads — and the shared expert into the batch's outputs. `lw`
    /// are layer `i`'s tensors.
    #[allow(
        clippy::too_many_arguments,
        reason = "the shadow's inputs and the layer's resolved tensors (rust-quality R8)"
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
            if card.is_some() {
                let act_x = &b.act_x[m - 1];
                let (q, d) = (act_x.q3().len() / m, act_x.d8().len() / m);
                // SAFETY: the copies read the chunk's own scratch and write
                // its tokens' columns of the block's planes; both are this
                // piece's buffers on the engine stream, which orders them
                // before the block's gather that reads them and after any
                // earlier launch that touched those columns.
                unsafe {
                    dtod(stream, &mut b.q3_all, at * q, act_x.q3(), m * q)?;
                    dtod(stream, &mut b.d8_all, at * d, act_x.d8(), m * d)?;
                }
            }
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
