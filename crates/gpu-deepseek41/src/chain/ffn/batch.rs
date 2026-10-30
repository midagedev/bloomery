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
//!    handoff's rule, an id past the expert count raising its fault); on a
//!    layer with card experts the layer's map row beside them into the
//!    batch's GEMM copy; the layer's tensors are resolved once for the route
//!    and the shadow ([`FfnPiece::resolve_batch`]);
//! 2. the exchange ([`FfnBatch::enqueue_download`], [`FfnBatch::serve`],
//!    [`FfnBatch::enqueue_upload`], through the host tier's batch port):
//!    the block's activations and routing to the host, one union call over
//!    its tokens for the layer's host experts ([`Hybrid::serve_key`]), the
//!    sums back;
//! 3. the shadow ([`FfnPiece::enqueue_batch_shadow`]), enqueued before the
//!    host computes: per chunk HC_PRE, the norm's q8_1 form again and the
//!    shared expert; then over the whole block the card's routed experts on
//!    the grouped GEMM ([`bloomery_gpu::gemm::enqueue_gemm_batch`]: the route
//!    table over the block's ids and the layer's map row, the gate and up
//!    GEMMs, the SwiGLU of the card slots' places, the down GEMM into the
//!    down rows by slot) and their sum (`ds41_ffn_card_acc`); the streamed
//!    passes under host streaming keep the tile items
//!    ([`enqueue_tiled_experts`]), an expert a tile of its rows at most eight
//!    slots of its own;
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
//! Every token's route values are the step's bit for bit: each launch of
//! the route writes, per token, what its one-token launch writes — the
//! m-column kernels carry that contract — and the combine is
//! [`combine_elem`] cut at its one seam ([`card_sum_elem`], then
//! [`join_elem`]). The streamed passes' tile path stays the step's bits (its
//! gate·up reads its token's column through the step's `q3k_row_dot`); the
//! default block's card experts run the grouped GEMM, which folds each
//! 128-value activation block one fma where the step keeps per-lane partials
//! a warp tree reduces: their rows are a reassociated sum of the same exact
//! integer block sums, a band the prefill gate pins, not the steps' bits.

use std::mem::size_of;

use bloomery_gpu::card_leg::LegRun;
use bloomery_gpu::card_slots::CardSlotKernels;
use bloomery_gpu::cores::q3k_row_dot;
use bloomery_gpu::gemm::{
    DownW, GateUp, Gemm32Kernels, GemmAct, GemmKernels, GemmWeight, enqueue_gemm_batch,
};
use bloomery_gpu::hybrid::{BatchKey, ServeTimes};
use bloomery_gpu::kquant::TileScratch;
use bloomery_gpu::kquant::tiles::{GatherPlanes, gather, gather_plane};
use bloomery_gpu::q4k_sel::{TileBlock, TileWork, tile_block, tile_cap};
use bloomery_gpu::{FaultSink, FaultSite, col_sums, store_cols};
use cuda_core::{CudaContext, IntoResult, sys};

use super::*;
use crate::hc::hc_mean_elem;
use crate::span::{span, span_mut};
use bloomery_gpu::kquant::act::{self, ACT_SWIGLU_CLAMP};

const WHAT: &str = "FfnBatch";

/// Threads per block of every kernel here.
const THREADS: u32 = 256;
// The kernels' launch attributes spell the block as a literal.
const _: () = assert!(THREADS == 256);

/// Weight rows a gate·up block computes: one per warp.
const ROWS_PER_BLOCK: usize = THREADS as usize / 32;

#[cuda_module]
mod ffn_batch_kernels {
    use super::*;

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

    /// The q8_1 planes of a grouped table's entries (`order`, runs `start`,
    /// `q4k_sel::grouped_buckets`), the K-quant family's gather ([`gather`]) over
    /// the planes the step's Q3_K walk reads ([`Q3Planes`]): block `j`
    /// copies token column `col0 + order[j] / 6` of `q_in` and `d8_in`
    /// (`cols` columns, the block's from `col0`) to column `j` of `q_out` and
    /// `d8_out`, for each entry `j` below the count `start[n_experts]` — a
    /// byte copy, so column `j` is its slot's activation as the step's
    /// gate·up reads it. Entries from the count on are left as they were; an
    /// entry that names no slot of the block raises [`FaultSite::ExpertId`]
    /// on `fault` and is skipped.
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
        q_out: DisjointSlice<u64>,
        d8_out: DisjointSlice<f32>,
    ) {
        let mut planes = Q3Planes {
            q_in,
            d8_in,
            iters,
            n_sb,
            q_out,
            d8_out,
        };
        // SAFETY: order.len() >= n_slots, start.len() >= n_experts + 1 and the
        // planes' columns by the launch contract; the block's threads are all
        // here.
        unsafe {
            gather(
                order,
                start,
                n_experts,
                n_slots,
                col0,
                cols,
                N_USED,
                fault,
                &mut planes,
            );
        }
    }

    /// The routed experts' gate·up·SwiGLU of a grouped table's entries by
    /// tile items: block `g + tile_cap · ρ` (of `tile_cap · row_tiles`) is
    /// tile `g` of the table `tiles` (`q4k_sel::grouped_tiles`,
    /// `q4k_sel::tile_block`) — expert `e`, the `m` table entries from `j` — on
    /// the eight weight rows `8ρ ..`: warp `w` takes row `r = 8ρ + w` of both
    /// stacks and dots it with the `m` columns `j .. j + m` of the q8_1
    /// planes — the table's entries ([`ds41_card_gather`]) — through the
    /// m-column core `cores::q3k_row_dot`, whose column c is the one-column
    /// call on that column bit for bit, then the warp tree and
    /// the checked SwiGLU into `h[(j + c) * rows_per_expert + r]`: entry `j +
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
        h: DisjointSlice<f32>,
    ) {
        let mut work = Q3GateUpTiles {
            wg,
            wu,
            q,
            d8,
            rows_per_expert,
            n_sb,
            iters,
            limit,
            fault,
            h,
        };
        // SAFETY: the launch contract is the work's and `tile_block`'s; the
        // host passes iters = ceil(n_sb/2).
        unsafe {
            tile_block(
                tiles, start, n_experts, n_slots, tile_cap, row_tiles, fault, &mut work,
            );
        }
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

    /// [`ds41_card_gather`] of eight slots a token: entry `j`'s token column
    /// is `col0 + order[j] / 8`.
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
        q_out: DisjointSlice<u64>,
        d8_out: DisjointSlice<f32>,
    ) {
        let mut planes = Q3Planes {
            q_in,
            d8_in,
            iters,
            n_sb,
            q_out,
            d8_out,
        };
        // SAFETY: order.len() >= n_slots, start.len() >= n_experts + 1 and the
        // planes' columns by the launch contract; the block's threads are all
        // here.
        unsafe {
            gather(
                order,
                start,
                n_experts,
                n_slots,
                col0,
                cols,
                SLOTS_8,
                fault,
                &mut planes,
            );
        }
    }

    /// [`ds41_ffn_card_acc_8`] over the stage card's and the expert tier's
    /// slots ([`card_acc_tier_at`]): a slot on the stage card read from
    /// `down`, one on the tier from `trows`, each at its turn in slot order —
    /// the one-card sum over the union of the two cards' slots.
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
            trows.len() >= 8 * n * m,
            w.len() >= 8 * m,
            sel.len() >= 8 * m,
            tsel.len() >= 8 * m,
            acc.len() >= n * m
        )
    )]
    pub fn ds41_ffn_card_acc_8_tier(
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
        let a = AccTierIn {
            down,
            trows,
            w,
            sel,
            tsel,
            n_card,
            n_tier,
        };
        // SAFETY: i < n·m, and the launch contract gives every length the
        // helper's contract asks for at SLOTS_8 slots.
        let v = unsafe { card_acc_tier_at::<SLOTS_8>(&a, n, i) };
        // SAFETY: i < n·m <= acc.len(); thread i is acc[i]'s only writer.
        unsafe { *acc.get_unchecked_mut(i) = v };
    }

    /// [`ds41_ffn_card_acc_8_tier`] at four slots a token
    /// ([`card_acc_tier_at`]): the one-card sum of four slots over the
    /// union of the stage card's and the expert tier's.
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
            down.len() >= 4 * n * m,
            trows.len() >= 4 * n * m,
            w.len() >= 4 * m,
            sel.len() >= 4 * m,
            tsel.len() >= 4 * m,
            acc.len() >= n * m
        )
    )]
    pub fn ds41_ffn_card_acc_4_tier(
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
        let a = AccTierIn {
            down,
            trows,
            w,
            sel,
            tsel,
            n_card,
            n_tier,
        };
        // SAFETY: i < n·m, and the launch contract gives every length the
        // helper's contract asks for at SLOTS_4 slots.
        let v = unsafe { card_acc_tier_at::<SLOTS_4>(&a, n, i) };
        // SAFETY: i < n·m <= acc.len(); thread i is acc[i]'s only writer.
        unsafe { *acc.get_unchecked_mut(i) = v };
    }
}

/// The step's Q3_K gate·up·SwiGLU as a tile entry's work
/// ([`ffn_batch_kernels::ds41_expert_gate_up_tiles`] under `tile_block`):
/// warp `w`'s weight row `r` of the tile's expert in both stacks, dotted with
/// columns `j .. j + m` of the gathered planes through the m-column core
/// `cores::q3k_row_dot` and reduced with the warp tree; lane 0 stores the
/// checked SwiGLU ([`act::apply_checked`]) in `h[(j + c) * rows_per_expert + r]`.
struct Q3GateUpTiles<'a> {
    wg: &'a [u32],
    wu: &'a [u32],
    q: &'a [u64],
    d8: &'a [f32],
    rows_per_expert: u32,
    n_sb: u32,
    iters: u32,
    limit: f32,
    fault: FaultSink,
    h: DisjointSlice<'a, f32>,
}

impl TileWork for Q3GateUpTiles<'_> {
    #[inline(always)]
    unsafe fn run(&mut self, t: TileBlock) {
        let TileBlock { e, j, m, r, lane } = t;
        let rpe = self.rows_per_expert as usize;
        let (wg, wu, q, d8) = (self.wg, self.wu, self.q, self.d8);
        let (n_sb, iters) = (self.n_sb as usize, self.iters);
        // The core's caller contract: row e·rpe + r < n_experts·rpe rows of
        // both stacks (r < 8 · row_tiles <= rows_per_expert); columns j + m <=
        // n_slots of `q`/`d8` by `tile_at` and the launch contract; iters =
        // ceil(n_sb/2) from the host; the warp's 32 lanes are here and m is
        // block-uniform.
        let row_abs = e * rpe + r;
        let fg = q3k_row_dot(wg, q, d8, n_sb, iters, row_abs, j, m, lane);
        let fu = q3k_row_dot(wu, q, d8, n_sb, iters, row_abs, j, m, lane);
        let g = col_sums(fg, m);
        let u = col_sums(fu, m);
        if lane == 0 {
            let limit = self.limit;
            let v = [
                act::apply_checked(ACT_SWIGLU_CLAMP, limit, g[0], u[0], self.fault),
                act::apply_checked(ACT_SWIGLU_CLAMP, limit, g[1], u[1], self.fault),
                act::apply_checked(ACT_SWIGLU_CLAMP, limit, g[2], u[2], self.fault),
                act::apply_checked(ACT_SWIGLU_CLAMP, limit, g[3], u[3], self.fault),
                act::apply_checked(ACT_SWIGLU_CLAMP, limit, g[4], u[4], self.fault),
                act::apply_checked(ACT_SWIGLU_CLAMP, limit, g[5], u[5], self.fault),
                act::apply_checked(ACT_SWIGLU_CLAMP, limit, g[6], u[6], self.fault),
                act::apply_checked(ACT_SWIGLU_CLAMP, limit, g[7], u[7], self.fault),
            ];
            // SAFETY: values (j + c)·rpe + r for c < m are below n_slots·rpe
            // <= h.len(); a table entry sits in one tile, so lane 0 of this
            // row's warp is their only writer.
            unsafe { store_cols(&mut self.h, j * rpe + r, rpe, m, v) };
        }
    }
}

/// The planes of a gather the step's Q3_K walk reads ([`gather`] in
/// [`ffn_batch_kernels::ds41_card_gather`] and `_8`): the q3 codes and the
/// block scales of the block's tokens and of the table's entries, `iters`
/// and `n_sb` their columns' extents.
struct Q3Planes<'a> {
    q_in: &'a [u64],
    d8_in: &'a [f32],
    iters: u32,
    n_sb: u32,
    q_out: DisjointSlice<'a, u64>,
    d8_out: DisjointSlice<'a, f32>,
}

impl GatherPlanes for Q3Planes<'_> {
    #[inline(always)]
    unsafe fn copy(&mut self, col: usize, j: usize, tid: usize) {
        let (qc, dc) = (64 * self.iters as usize, 2 * self.n_sb as usize);
        // SAFETY: the two planes' columns are this fn's contract.
        unsafe {
            gather_plane(self.q_in, &mut self.q_out, qc, col, j, tid);
            gather_plane(self.d8_in, &mut self.d8_out, dc, col, j, tid);
        }
    }
}

/// What a tier card sum of `N` slots a token reads ([`card_acc_tier_at`]):
/// the stage card's down outputs and the tier's (slot-major), the weights,
/// each slot's stage and tier place, and the two cards' expert counts.
struct AccTierIn<'a> {
    down: &'a [f32],
    trows: &'a [f32],
    w: &'a [f32],
    sel: &'a [u32],
    tsel: &'a [u32],
    n_card: u32,
    n_tier: u32,
}

/// The card slots' sum with the expert tier's slots in it: `acc =
/// fma(row, w[Nt + j], acc)` from zero for each slot `j` below `N` in
/// ascending `j`, `row` the stage card's `down[(Nt + j)·n + d]` when its
/// place `sel[Nt + j]` is below `n_card`, else the tier's `trows[(Nt +
/// j)·n + d]` when its tier place `tsel[Nt + j]` is below `n_tier`, the slot
/// left out otherwise — over the union of the two cards' slots, the order
/// the common card sum (`bloomery_gpu::card_slots`) sums one card's in (this
/// order is the gate).
///
/// SAFETY: `i < n · m` for an `m` with `down.len()` and `trows.len() >= N ·
/// n · m`, `w.len()`, `sel.len()` and `tsel.len() >= N · m`.
#[inline(always)]
unsafe fn card_acc_tier_at<const N: usize>(a: &AccTierIn<'_>, n: usize, i: usize) -> f32 {
    let (t, d) = (i / n, i % n);
    let mut acc = 0.0f32;
    for j in 0..N {
        cuda_device::thread::__unroll_config::<0>();
        let s = t * N + j;
        // SAFETY: s < N·m <= sel.len(), tsel.len() and w.len() by this fn's
        // contract.
        let (place, tplace, ws) = unsafe {
            (
                *a.sel.get_unchecked(s),
                *a.tsel.get_unchecked(s),
                *a.w.get_unchecked(s),
            )
        };
        if place < a.n_card {
            // SAFETY: s < N·m and d < n, so s·n + d < N·n·m <= down.len().
            let ds = unsafe { *a.down.get_unchecked(s * n + d) };
            acc = ds.mul_add(ws, acc);
        } else if tplace < a.n_tier {
            // SAFETY: as above, for trows.
            let ds = unsafe { *a.trows.get_unchecked(s * n + d) };
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

/// A block's routed experts on one card by tile items, the K-quant family's
/// tile path ([`TileScratch`]) with the step's Q3_K gate·up: the slots of
/// `t.sel` grouped by expert and the table's tiles
/// ([`TileScratch::enqueue_table`]), the q8_1 planes gathered by table entry
/// (`ds41_card_gather`), the gate·up (`ds41_expert_gate_up_tiles`) into the
/// SwiGLU outputs by entry, their q8_1 form
/// ([`TileScratch::enqueue_quantize_h`]) and the Q4_K down
/// (`q4k_gemv_tiles`), which scatters each entry's rows to its slot's rows of
/// `down` (slot-major, `n_embd` a slot); a slot whose place is [`HOST`] is
/// left as it was. Each weight row is read once for up to eight slots. The
/// table's size is a device value: no launch here reads it on the host. The
/// stage card's batch shadow and the expert tier's batch service both run
/// it, each over its own scratch. Asynchronous, allocation-free.
pub fn enqueue_tiled_experts(
    gpu: &Gpu,
    kernels: &FfnBatchKernels,
    t: &TiledBlock<'_>,
    s: &mut Q3TileScratch,
    down: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let (stream, ff, n, layer) = (gpu.stream(), s.ff, s.n_embd, t.layer);
    let fault = gpu.layer_sink(layer)?;
    let n_sb = n / 256;
    let slots_n = t.cols.saturating_sub(t.col0) * N_USED;
    let n_experts = t.stacks.gate.rows() / ff;
    if slots_n == 0 || slots_n > s.tile.slots() || t.sel.len() < slots_n || down.len() < slots_n * n
    {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "layer {layer}'s tiled block of columns {}..{}: {} places, {} down values; the \
                 scratch takes 1..={} slots",
                t.col0,
                t.cols,
                t.sel.len(),
                down.len(),
                s.tile.slots()
            ),
        });
    }
    s.tile
        .enqueue_table(gpu, t.sel, slots_n, n_experts, fault)?;
    {
        let table = s.tile.table(slots_n, n_experts);
        let g = CardGather {
            q3: t.q3,
            d8: t.d8,
            order: table.order,
            start: table.start,
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
    {
        let q3 = span(WHAT, &s.q3_ord, 0, slots_n * 64 * n_sb.div_ceil(2))?;
        let d8 = span(WHAT, &s.d8_ord, 0, slots_n * 2 * n_sb)?;
        let (table, h) = s.tile.table_and_h(slots_n, n_experts);
        let mut h = span_mut(WHAT, h, 0, slots_n * ff)?;
        let g = TiledGateUp {
            wg: t.stacks.gate.buf(),
            wu: t.stacks.up.buf(),
            q3: &q3,
            d8: &d8,
            start: table.start,
            tiles: table.tiles,
            n_experts,
            rows_per_expert: ff,
            n_slots: slots_n,
            n_sb,
            limit: t.limit,
        };
        kernels.enqueue_gate_up_tiles(stream, &g, fault, &mut h)?;
    }
    s.tile.enqueue_quantize_h(gpu, slots_n, n_experts, fault)?;
    gpu.q4k_sel().enqueue_gemv_q4k_tiles(
        stream,
        &s.tile.down(t.stacks.down, n, slots_n, n_experts),
        fault,
        down,
    )
}

/// Tokens one run of the common card leg covers in a block of a layer on it
/// ([`enqueue_leg_block`]): the leg's buffers are sized for this many.
pub const LEG_PIECE: usize = 64;

/// What the leg path reads for one block ([`enqueue_leg_block`]): the layer's
/// stacks as the common card leg reads them, the f32 normed rows of the
/// block's `cols` tokens (`n_embd` values a token), each of its `N_USED ·
/// cols` slots' place in the stacks (`sel`: a stack slot, or [`HOST`] for a
/// slot another device computes), the layer, its SwiGLU clamp and the experts
/// the stacks hold.
pub struct LegBlock<'a> {
    pub stacks: &'a LegStacks<'a>,
    pub x: &'a DeviceBuffer<f32>,
    pub sel: &'a DeviceBuffer<u32>,
    pub cols: usize,
    pub layer: usize,
    pub limit: f32,
    pub n_card: usize,
}

/// A block's routed experts on one card through the common card leg, for a
/// layer whose types the tile path does not read: the block cut in pieces of
/// at most [`LEG_PIECE`] tokens, each piece the leg's four launches over its
/// tokens' slots (`CardLeg::enqueue_rows`), the down rows into `down`
/// (slot-major, `n_embd` a slot); a slot whose place is [`HOST`] is left as
/// it was. The launches are slot-local, so a slot's rows are the step's for
/// the same slot. Asynchronous, allocation-free.
pub fn enqueue_leg_block(
    gpu: &Gpu,
    leg: &mut (CardLeg, LegBufs),
    t: &LegBlock<'_>,
    down: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    let (card, bufs) = leg;
    let n = bufs.shape().n;
    let rule = Act::SwigluClamp { limit: t.limit };
    let mut t0 = 0;
    while t0 < t.cols {
        let m = (t.cols - t0).min(LEG_PIECE);
        let x = span(WHAT, t.x, t0 * n, m * n)?;
        let sel = span(WHAT, t.sel, t0 * N_USED, m * N_USED)?;
        let mut rows = span_mut(WHAT, down, t0 * N_USED * n, m * N_USED * n)?;
        let run = LegRun {
            layer: t.layer,
            stacks: t.stacks,
            normed: &x,
            m,
            sel: &sel,
            n_card: t.n_card,
            rule,
        };
        card.enqueue_rows(gpu, bufs, &run, &mut rows)?;
        t0 += m;
    }
    Ok(())
}

/// Which path a block's card experts run on
/// ([`FfnPiece::enqueue_block_experts`]): the default pass's grouped GEMM,
/// or the K-quant tile path, which the passes under host streaming keep (the
/// kept and the streamed pass, over the block's own or its streamed places).
#[derive(Clone, Copy)]
enum CardPath {
    Gemm,
    Tiles,
}

/// The batch's kernels: loaded once by [`FfnBatch::new`], or alone by a gate
/// that drives the launches with inputs a batch's route cannot produce.
pub struct FfnBatchKernels {
    module: ffn_batch_kernels::LoadedModule,
    /// The places and the card sum, which the common card slots own
    /// ([`CardSlotKernels`]).
    slots: CardSlotKernels,
    /// The grouped GEMM and the 32-value GEMM's route fill, for the default
    /// block's card experts ([`enqueue_block_experts`]).
    gemm: GemmKernels,
    g32: Gemm32Kernels,
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
        let module = unsafe { bloomery_gpu::shared_module!(ffn_batch_kernels, ctx)? };
        Ok(FfnBatchKernels {
            module,
            slots: CardSlotKernels::load(ctx)?,
            gemm: GemmKernels::load(ctx)?,
            g32: Gemm32Kernels::load(ctx)?,
        })
    }

    /// Enqueue `ds41_ffn_places` ([`CardSlotKernels::enqueue_places`]): `sel[i]`
    /// the place of `p.ids[i]` for `i < p.n`, [`HOST`] and
    /// [`FaultSite::ExpertId`] on `fault` for an id past the stack. One launch.
    /// Asynchronous, allocation-free.
    pub fn enqueue_places(
        &self,
        stream: &CudaStream,
        p: &Places<'_>,
        fault: FaultSink,
        sel: &mut DeviceBuffer<u32>,
    ) -> Result<(), GpuError> {
        self.slots.enqueue_places(stream, p, fault, sel)
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

    /// Enqueue the gather of `g.n_used` slots a token (`ds41_card_gather` for
    /// 6, `ds41_card_gather_8` for 8; any other count, four included, is
    /// refused by name):
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
            Slots::Four => return Err(no_entry("ds41_card_gather", slots)),
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
            Slots::Four => return Err(no_entry(what, slots)),
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
    /// ([`CardSlotKernels::enqueue_card_acc`]: `ds41_ffn_card_acc` for 6,
    /// `ds41_ffn_card_acc_8` for 8; any other count is refused by name): value
    /// `i` of token `t = i / a.n` into `acc[i]`, the card half of the combine.
    /// One launch. Asynchronous, allocation-free.
    pub fn enqueue_card_acc(
        &self,
        stream: &CudaStream,
        a: &CardAcc<'_>,
        acc: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.slots.enqueue_card_acc(stream, a, acc)
    }

    /// Enqueue `ds41_ffn_card_acc_tier`: the card sums of `a.m` tokens over
    /// the stage card's and the tier's slots, value `i` of token `t = i / a.n`
    /// into `acc[i]`: `ds41_ffn_card_acc_tier` for six slots a token,
    /// `ds41_ffn_card_acc_4_tier` and `ds41_ffn_card_acc_8_tier` (the fma
    /// chains of `ds41_ffn_card_acc_4` and `_8`) for four and eight; any
    /// other count is refused by name. One launch. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_card_acc_tier(
        &self,
        stream: &CudaStream,
        a: &CardAccTier<'_>,
        acc: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let slots = slots_of("ds41_ffn_card_acc_tier", a.n_used)?;
        let what = match slots {
            Slots::Four => "ds41_ffn_card_acc_4_tier",
            Slots::Six => "ds41_ffn_card_acc_tier",
            Slots::Eight => "ds41_ffn_card_acc_8_tier",
        };
        let grid = launch_u32(what, "grid", (a.m * a.n).div_ceil(THREADS as usize))?;
        let cfg = LaunchConfig1D::new(grid, THREADS, 0);
        let (n, m, n_card, n_tier) = (
            launch_u32(what, "n", a.n)?,
            launch_u32(what, "m", a.m)?,
            launch_u32(what, "n_card", a.n_card)?,
            launch_u32(what, "n_tier", a.n_tier)?,
        );
        match slots {
            Slots::Four => {
                let prep = self.module.prepare_ds41_ffn_card_acc_4_tier(cfg)?;
                self.module.ds41_ffn_card_acc_4_tier(
                    stream, &prep, a.down, a.trows, a.w, a.sel, a.tsel, n, m, n_card, n_tier, acc,
                )?;
            }
            Slots::Six => {
                let prep = self.module.prepare_ds41_ffn_card_acc_tier(cfg)?;
                self.module.ds41_ffn_card_acc_tier(
                    stream, &prep, a.down, a.trows, a.w, a.sel, a.tsel, n, m, n_card, n_tier, acc,
                )?;
            }
            Slots::Eight => {
                let prep = self.module.prepare_ds41_ffn_card_acc_8_tier(cfg)?;
                self.module.ds41_ffn_card_acc_8_tier(
                    stream, &prep, a.down, a.trows, a.w, a.sel, a.tsel, n, m, n_card, n_tier, acc,
                )?;
            }
        }
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
    /// d8, as the chunk's scratch lays out a column) and the tile path's
    /// scratch ([`Q3TileScratch`]) — both the streamed passes' — the default
    /// block's GEMM scratch ([`GemmScratch`]) and the down output by slot,
    /// which every path writes the same rows of.
    q3_all: DeviceBuffer<u64>,
    d8_all: DeviceBuffer<f32>,
    down_all: DeviceBuffer<f32>,
    tile: Q3TileScratch,
    gemm: GemmScratch,
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
    /// The common card leg's launches and buffers, for a block of a layer on
    /// it ([`FfnBatch::prepare_leg`]).
    leg: Option<(CardLeg, LegBufs)>,
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
/// ([`enqueue_tiled_experts`]): the K-quant family's ([`TileScratch`]: the
/// table, the SwiGLU output and its q8_1 form by entry) and the planes the
/// step's Q3_K walk reads gathered by table entry (`q3_ord`, `d8_ord`) —
/// column `j` is slot `order[j]`'s. The stage card's batch and the expert
/// tier each own one; the path is the same.
pub struct Q3TileScratch {
    tile: TileScratch,
    q3_ord: DeviceBuffer<u64>,
    d8_ord: DeviceBuffer<f32>,
    n_embd: usize,
    ff: usize,
}

impl Q3TileScratch {
    /// The scratch for blocks of up to `slots` slots (1..=3072, six a token of
    /// the host union's columns) over rows of `n_embd` through experts of
    /// `ff`, on `stream`'s card. Load-time only.
    pub fn new(
        stream: &CudaStream,
        slots: usize,
        n_embd: usize,
        ff: usize,
    ) -> Result<Q3TileScratch, GpuError> {
        let n_sb = n_embd / 256;
        Ok(Q3TileScratch {
            tile: TileScratch::new(stream, slots, ff)?,
            q3_ord: DeviceBuffer::zeroed(stream, slots * 64 * n_sb.div_ceil(2))?,
            d8_ord: DeviceBuffer::zeroed(stream, slots * 2 * n_sb)?,
            n_embd,
            ff,
        })
    }

    /// Device bytes of the scratch.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.tile.device_bytes() + self.q3_ord.num_bytes() + self.d8_ord.num_bytes()
    }
}

/// The default block's card experts on the grouped GEMM
/// ([`enqueue_block_experts`]): the block's normed rows in the GEMM's q8_1
/// form (a column a token — the route's norm quantized again, whose bytes
/// the norm's fused launch defines equal to its own, bit for bit), the gate
/// and up products (slot-major, `ff` a slot), the SwiGLU's q8_1 form (a
/// column a slot) and the layer's map row as the block's route fill reads
/// it — the slot map lives outside the batch, so the route, which holds it,
/// copies the row (the row a pick moves is copied again by the route that
/// runs under it).
struct GemmScratch {
    act_x: GemmAct,
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
    act_down: GemmAct,
    map_row: DeviceBuffer<u32>,
}

impl GemmScratch {
    /// The scratch for blocks of up to `cap` tokens of `n_embd` through
    /// experts of `ff` over `n_expert` experts, on `stream`'s card. Load-time
    /// only; the shapes the GEMM family does not take are its named refusal.
    fn new(
        stream: &CudaStream,
        (cap, n_embd, ff, n_expert): (usize, usize, usize, usize),
    ) -> Result<GemmScratch, GpuError> {
        let slots = cap * N_USED;
        Ok(GemmScratch {
            act_x: GemmAct::new(stream, cap, n_embd)?,
            g: DeviceBuffer::zeroed(stream, slots * ff)?,
            u: DeviceBuffer::zeroed(stream, slots * ff)?,
            act_down: GemmAct::new(stream, slots, ff)?,
            map_row: DeviceBuffer::zeroed(stream, n_expert)?,
        })
    }

    /// Device bytes of the scratch.
    fn bytes(&self) -> usize {
        self.act_x.bytes()
            + self.g.num_bytes()
            + self.u.num_bytes()
            + self.act_down.bytes()
            + self.map_row.num_bytes()
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
            tile: Q3TileScratch::new(stream, slot_cols, n_embd, ff)?,
            gemm: GemmScratch::new(stream, (cap, n_embd, ff, N_EXPERT))?,
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
            leg: None,
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

    /// The last route's buffers, for a reader that runs the selection again
    /// on the host: the router's scores ([`N_EXPERT`] a token from the
    /// block's first) and the routed ids ([`N_USED`] a token from the
    /// batch's token 0), which the next route rewrites.
    #[must_use]
    pub fn route_rows(&self) -> (&DeviceBuffer<f32>, &DeviceBuffer<u32>) {
        (&self.probs, &self.ids)
    }

    /// Of [`FfnBatch::device_bytes`], the HC_PRE results of every batch of a
    /// group.
    #[must_use]
    pub fn hc_bytes(&self) -> usize {
        self.hc.iter().map(DeviceBuffer::num_bytes).sum()
    }

    /// The common card leg for the blocks of layers whose types it runs
    /// ([`CardStacks::fused`] false): its launches and buffers for pieces of
    /// [`LEG_PIECE`] tokens. A batch over a file whose card layers are all
    /// fused needs none, and a block of a layer that does refuses by name
    /// until this has run. Load-time only.
    pub fn prepare_leg(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        if self.leg.is_some() {
            return Ok(());
        }
        let shape = LegShape {
            cap: LEG_PIECE,
            slots_per_col: N_USED,
            n: self.n_embd,
            ff: self.ff,
        };
        self.leg = Some((CardLeg::load(gpu)?, LegBufs::new(gpu.stream(), shape)?));
        Ok(())
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
            + self.gemm.bytes()
            + self.tsel.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.ssel.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.leg.as_ref().map_or(0, |(_, b)| b.bytes())
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

/// A run of one kind's tokens of a block, with the bias its tokens pick
/// with (`vl` on media tokens, `text` on the rest).
type PickRun<'a> = (Range<usize>, &'a DeviceBuffer<f32>);

/// The block `at .. u` of a batch whose tokens are media positions where
/// `media` says so, as block-relative runs of one kind, each with the bias it
/// picks with: `vl` on media tokens, `text` on the rest.
fn pick_runs<'a>(
    media: &[bool],
    at: usize,
    u: usize,
    [text, vl]: [&'a DeviceBuffer<f32>; 2],
) -> Result<Vec<PickRun<'a>>, GpuError> {
    let flags = media.get(at..u).ok_or_else(|| GpuError::Shape {
        what: WHAT,
        detail: format!(
            "the media flags of a block {at}..{u} of a batch of {} tokens",
            media.len()
        ),
    })?;
    let mut runs = Vec::new();
    let mut from = 0;
    for k in 1..=flags.len() {
        if k == flags.len() || flags[k] != flags[from] {
            runs.push((from..k, if flags[from] { vl } else { text }));
            from = k;
        }
    }
    Ok(runs)
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
    /// place from `slots`, the map's card copy (one launch); on a layer with
    /// card experts, the layer's map row beside it into the batch's GEMM copy
    /// (one copy, which the default block's route fill reads). With `vl`, a
    /// media call's — the layer's `exp_probs_b_vl` row, and per token of the
    /// batch whether it is a media position — the block's media tokens pick
    /// with that bias: the router's scores once over the block, then a pick
    /// launch per run of tokens of one kind
    /// ([`RouterKernels::enqueue_router_rows_runs`]). Returns the router's
    /// pick launches. Asynchronous, allocation-free without `vl`.
    pub fn enqueue_batch_route(
        &mut self,
        gpu: &Gpu,
        bl: &BatchLayer<'_>,
        b: &mut FfnBatch,
        io: &BlockIo<'_>,
        slots: &DeviceTensor<u32>,
        vl: Option<(&DeviceBuffer<f32>, &[bool])>,
    ) -> Result<usize, GpuError> {
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
        let picks = {
            let x = span(WHAT, &b.x, at * n, t * n)?;
            let mut probs = span_mut(WHAT, &mut b.probs, 0, t * N_EXPERT)?;
            let mut ids = span_mut(WHAT, &mut b.ids, at * N_USED, t * N_USED)?;
            let mut wts = span_mut(WHAT, &mut b.weights, at * N_USED, t * N_USED)?;
            match vl {
                None => {
                    self.router.enqueue_router_rows(
                        stream,
                        gpu,
                        &self.kernels.g32,
                        lw.router,
                        &x,
                        lw.bias,
                        self.scale,
                        t,
                        &mut probs,
                        &mut ids,
                        &mut wts,
                        fault,
                    )?;
                    1
                }
                Some((vl, media)) => {
                    let runs = pick_runs(media, at, u, [lw.bias, vl])?;
                    self.router.enqueue_router_rows_runs(
                        stream,
                        gpu,
                        &self.kernels.g32,
                        lw.router,
                        &x,
                        self.scale,
                        t,
                        &runs,
                        &mut probs,
                        &mut ids,
                        &mut wts,
                        fault,
                    )?;
                    runs.len()
                }
            }
        };
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
        b.kernels.enqueue_places(stream, &places, fault, &mut sel)?;
        if c.n_card > 0 {
            // The default block's GEMM route fill reads the layer's map row
            // from the batch's own copy: the shadow holds no slot map, and a
            // row a pick moves is copied again by the route that runs under
            // it.
            let row = span(WHAT, slots.buf(), c.row_off, self.n_expert)?;
            // SAFETY: the copy reads the slot map's card copy beside the
            // places launch that read the same row before it on this stream
            // (so it sees the row that launch saw), and writes the batch's
            // row, which nothing but the blocks' route fills read, every one
            // of them behind this route in stream order.
            unsafe {
                dtod(stream, &mut b.gemm.map_row, 0, &row, self.n_expert)?;
            }
        }
        Ok(picks)
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
    /// block the card's routed experts on the grouped GEMM ([`enqueue_block_experts`]:
    /// the route table over the block's ids and the layer's map row, the gate
    /// and up GEMMs, the SwiGLU of the card slots' places, the down GEMM) and
    /// — unless the layer is `tiered` — the card sum. Asynchronous,
    /// allocation-free.
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
    /// card experts on the grouped GEMM and the card sum, or for a `tiered`
    /// layer the held routing ([`FfnPiece::enqueue_grouped_block`]).
    /// Asynchronous, allocation-free.
    pub fn enqueue_batch_block(
        &mut self,
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
    /// pick admitted, on the tile path the streamed passes keep (over the
    /// replaced places), after [`FfnPiece::enqueue_batch_replace`]; the
    /// streamed pass follows. Asynchronous, allocation-free.
    pub fn enqueue_batch_block_kept(
        &mut self,
        gpu: &Gpu,
        bl: &BatchLayer<'_>,
        card: Option<CardStacks<'_>>,
        b: &mut FfnBatch,
        io: &BlockIo<'_>,
    ) -> Result<(), GpuError> {
        let layer = bl.layer;
        let (i, at, u) = self.batch_block(layer, b, io)?;
        let card = self.check_card(layer, i, card)?;
        self.enqueue_block_experts(gpu, i, card, b, layer, [at, u], CardPath::Tiles, false)
    }

    /// Under host streaming, the streamed pass of the block, once the engine
    /// stream waits for the pick's copies: the admitted ids' experts on the
    /// tile path the streamed passes keep, over the streamed places, those
    /// places merged into the block's (`ds41_ffn_sel_merge`), then the card
    /// sum in slot order over every card slot, or for a `tiered` layer the
    /// held routing. Asynchronous, allocation-free.
    pub fn enqueue_batch_stream(
        &mut self,
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
        self.enqueue_block_experts(gpu, i, card, b, layer, [at, u], CardPath::Tiles, true)?;
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
        self.enqueue_block_finish(gpu, i, b, [layer, io.set], [at, u], tiered)
    }

    /// The block's launches for tokens `at .. u` of the group's batch `set`,
    /// after every chunk's part before them: with card experts, the default
    /// pass's GEMM over the block's card slots
    /// ([`enqueue_block_experts`]) into the block's down outputs by slot;
    /// then the card sum of every token of the block — or, for a `tiered`
    /// layer, whose card sum waits for the tier's rows
    /// ([`FfnPiece::enqueue_batch_acc_tier`]), the copy of the block's
    /// weights, card places and tier places into the batch's held routing,
    /// three copies.
    #[allow(
        clippy::too_many_arguments,
        reason = "the block's layer, stacks, buffers and tokens (rust-quality R8)"
    )]
    fn enqueue_grouped_block(
        &mut self,
        gpu: &Gpu,
        i: usize,
        card: Option<CardStacks<'_>>,
        b: &mut FfnBatch,
        [layer, set]: [usize; 2],
        [at, u]: [usize; 2],
        tiered: bool,
    ) -> Result<(), GpuError> {
        self.enqueue_block_experts(gpu, i, card, b, layer, [at, u], CardPath::Gemm, false)?;
        self.enqueue_block_finish(gpu, i, b, [layer, set], [at, u], tiered)
    }

    /// With card experts, the block's card slots into the block's down
    /// outputs by slot, on `path`: a layer the common card leg runs goes
    /// through it whatever the path; the default pass's GEMM quantizes the
    /// block's normed rows ([`Gpu::enqueue_quantize_gemm`], the bytes the
    /// route's fused norm leaves, bit for bit) and runs the common batch
    /// sequence ([`enqueue_gemm_batch`]) — the route table over the block's
    /// ids and the layer's map row the route copied, the gate and up GEMMs,
    /// the SwiGLU of the card slots' places, the down GEMM — each stack's
    /// type read from the stacks and refused by name where the grouped GEMM
    /// decodes it not; the tile path ([`enqueue_tiled_experts`]) reads the
    /// block's q8_1 planes over the block's places, or with `streamed` over
    /// its streamed places.
    #[allow(
        clippy::too_many_arguments,
        reason = "the block's layer, stacks, buffers, tokens, path and places (rust-quality R8)"
    )]
    fn enqueue_block_experts(
        &mut self,
        gpu: &Gpu,
        i: usize,
        card: Option<CardStacks<'_>>,
        b: &mut FfnBatch,
        layer: usize,
        [at, u]: [usize; 2],
        path: CardPath,
        streamed: bool,
    ) -> Result<(), GpuError> {
        let (n, ff) = (self.n_embd, self.ff);
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
            if let Some(legs) = s.legs() {
                let leg = b.leg.as_mut().ok_or(GpuError::State {
                    what: WHAT,
                    missing: "the common card leg of a block of a layer of types it runs \
                              (FfnBatch::prepare_leg)",
                })?;
                let x = span(WHAT, &b.x, at * n, (u - at) * n)?;
                let t = LegBlock {
                    stacks: legs,
                    x: &x,
                    sel: &sel,
                    cols: u - at,
                    layer,
                    limit: c.limit,
                    n_card: c.n_card,
                };
                return enqueue_leg_block(gpu, leg, &t, &mut down);
            }
            match path {
                CardPath::Gemm => {
                    let t = u - at;
                    let (gty, dty) = (
                        GemmWeight::from_ggml(s.gate_up_ty())?,
                        GemmWeight::from_ggml(s.down_ty())?,
                    );
                    let table = self.gemm_route_at(layer, c.n_card)?;
                    let sink = gpu.layer_sink(layer)?;
                    let x = span(WHAT, &b.x, at * n, t * n)?;
                    gpu.enqueue_quantize_gemm(&x, t, &mut b.gemm.act_x, sink)?;
                    let ids = span(WHAT, &b.ids, at * N_USED, slots_n)?;
                    enqueue_gemm_batch(
                        (&b.kernels.gemm, &b.kernels.g32),
                        gpu.stream(),
                        &ids,
                        &b.gemm.map_row,
                        GateUp::Grouped(gty, [s.gate, s.up], &b.gemm.act_x),
                        DownW::Grouped(
                            dty,
                            s.down,
                            Act::SwigluClamp { limit: c.limit },
                            &mut b.gemm.act_down,
                            &mut down,
                        ),
                        (t, c.n_card),
                        (N_USED, ff, n),
                        sink,
                        (&mut b.gemm.g, &mut b.gemm.u),
                        (&*sel, &mut self.routes[table]),
                    )
                }
                CardPath::Tiles => {
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
                    enqueue_tiled_experts(gpu, &b.kernels, &t, &mut b.tile, &mut down)
                }
            }
        } else {
            Ok(())
        }
    }

    /// The block's card sum of every token, or for a `tiered` layer, whose
    /// card sum waits for the tier's rows, the copy of the block's weights,
    /// card places and tier places into the batch's held routing.
    fn enqueue_block_finish(
        &self,
        gpu: &Gpu,
        i: usize,
        b: &mut FfnBatch,
        [layer, set]: [usize; 2],
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
            w_pitch: N_USED,
            fault: gpu.layer_sink(layer)?,
        };
        b.kernels.enqueue_card_acc(stream, &a, &mut acc)
    }

    /// Layer `layer`'s shadow work for the chunk `io`, after the batch's
    /// route: HC_PRE into the batch's results, the norm's q8_1 form again —
    /// with card experts copied into the block's planes, which a streamed
    /// block's tile path reads — and the shared expert into the batch's
    /// outputs. `lw` are layer `i`'s tensors.
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
            let pre = HcFnArgs {
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
                .enqueue_pre_fn(stream, &pre, &mut self.hc_scratch, &mut b.mixes, &mut hc)?;
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
                gpu.layer_sink(layer)?,
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
                        gpu.layer_sink(layer)?,
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

#[cfg(test)]
mod tests {
    use super::{CardAccTier, FfnBatchKernels};
    use bloomery_gpu::host::handoff::{Handoff, HandoffKernels};
    use bloomery_gpu::host::page::HandoffLayout;
    use bloomery_gpu::host::step::HandoffTarget;
    use bloomery_gpu::host::tier::{TierImageLayout, TierTarget};
    use bloomery_gpu::hybrid::HOST;
    use bloomery_gpu::{FaultSite, Gpu, GpuError, Window};
    use cuda_core::DeviceBuffer;

    /// The handoff's places feed the production tier sum, not a readback
    /// alone: every routed row belongs to exactly one of host, card and
    /// tier, and all other copies are poisoned. Both slot entries must
    /// conserve that union, including the final routed lane and partial
    /// blocks, eagerly and on replay. Invalid ids must still raise.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-ds41-lib` runs it on the box"]
    fn hw_f32_tier_handoffs_conserve_consumed_routes() {
        const EXPERTS: usize = 16;
        const ROW_OFF: usize = 3;
        const LAYER: usize = 7;
        let gpu = Gpu::new().expect("a GPU");
        let stream = gpu.stream();
        let handoff = HandoffKernels::load(gpu.context()).expect("handoff kernels");
        let sums = FfnBatchKernels::load(gpu.context()).expect("tier sum kernels");
        let mut map = vec![HOST - 1; ROW_OFF + EXPERTS];
        let mut tmap = map.clone();
        for id in 0..EXPERTS {
            map[ROW_OFF + id] = if id % 3 == 0 { (id / 3) as u32 } else { HOST };
            tmap[ROW_OFF + id] = if id % 3 == 1 { (id / 3) as u32 } else { HOST };
        }
        let map = DeviceBuffer::from_host(stream, &map).expect("card map");
        let tmap = DeviceBuffer::from_host(stream, &tmap).expect("tier map");
        for n_used in [4, 6, 8] {
            for hidden in [n_used, 256, 257] {
                for bad in [None, Some(EXPERTS as u32), Some(u32::MAX)] {
                    let mut ids = vec![15, 0, 10, 5, 8, 1, 2, 7];
                    ids.truncate(n_used);
                    if let Some(id) = bad {
                        ids[n_used - 1] = id;
                    }
                    // No handoff may read the router's trailing lane.
                    ids.push(u32::MAX);
                    let weights: Vec<f32> = (0..n_used)
                        .map(|s| (s + 1) as f32 / 16.0)
                        .chain([f32::NAN])
                        .collect();
                    let x: Vec<f32> = (0..hidden).map(|d| (d % 7 + 1) as f32 / 8.0).collect();
                    let row = |s: usize, d: usize| (ids[s] + 1) as f32 * x[d];
                    let mut down = vec![f32::NAN; n_used * hidden];
                    let mut trows = down.clone();
                    let mut host = vec![0.0f32; hidden];
                    let mut want = host.clone();
                    for d in 0..hidden {
                        let mut card_tier = 0.0f32;
                        for s in 0..n_used {
                            if ids[s] >= EXPERTS as u32 {
                                continue;
                            }
                            let value = row(s, d);
                            match ids[s] % 3 {
                                0 => down[s * hidden + d] = value,
                                1 => trows[s * hidden + d] = value,
                                _ => host[d] = value.mul_add(weights[s], host[d]),
                            }
                            if ids[s] % 3 != 2 {
                                card_tier = value.mul_add(weights[s], card_tier);
                            }
                        }
                        want[d] = card_tier + host[d];
                    }
                    let ids_d = DeviceBuffer::from_host(stream, &ids).expect("ids");
                    let mut weights_d = DeviceBuffer::from_host(stream, &weights).expect("weights");
                    let x_d = DeviceBuffer::from_host(stream, &x).expect("activation");
                    let seq = DeviceBuffer::from_host(stream, &[19u32]).expect("sequence");
                    let down = DeviceBuffer::from_host(stream, &down).expect("card rows");
                    let trows = DeviceBuffer::from_host(stream, &trows).expect("tier rows");
                    let doubled_host: Vec<f32> = host.iter().map(|h| 2.0 * h).collect();
                    let mut host = DeviceBuffer::from_host(stream, &host).expect("host sum");
                    let lay = HandoffLayout {
                        seq: 0,
                        ids: 1,
                        weights: 1 + n_used,
                        x: 1 + 2 * n_used,
                        n_used,
                        hidden,
                    };
                    let tl = TierImageLayout {
                        sel: 0,
                        q3: 0,
                        d8: 0,
                        x: n_used,
                        n_used,
                        q3_words: 0,
                        d8_len: 0,
                        x_len: hidden,
                        hidden,
                        cols: 1,
                    };
                    for captured in [false, true] {
                        let mut image =
                            DeviceBuffer::zeroed(stream, lay.x + hidden).expect("image");
                        let mut timage =
                            DeviceBuffer::zeroed(stream, tl.x + hidden).expect("tier image");
                        let poison = vec![HOST - 1; n_used + 1];
                        let mut sel = DeviceBuffer::from_host(stream, &poison).expect("places");
                        let mut tsel =
                            DeviceBuffer::from_host(stream, &poison).expect("tier places");
                        let mut acc = DeviceBuffer::zeroed(stream, hidden).expect("sum");
                        let mut result = DeviceBuffer::zeroed(stream, hidden).expect("result");
                        let mut enqueue = || -> Result<(), GpuError> {
                            handoff.enqueue_handoff_tier(
                                stream,
                                &Handoff {
                                    ids: &ids_d,
                                    weights: &weights_d,
                                    map: &map,
                                    row_off: ROW_OFF,
                                    n_expert: EXPERTS,
                                },
                                &tmap,
                                (
                                    HandoffTarget {
                                        x: &x_d,
                                        seq: &seq,
                                        image: &mut image,
                                        layout: lay,
                                    },
                                    TierTarget {
                                        image: &mut timage,
                                        rows: &trows,
                                        layout: tl,
                                    },
                                ),
                                gpu.layer_sink(LAYER)?,
                                (&mut sel, &mut tsel),
                            )?;
                            let routed_weights =
                                Window::<f32>::of(&image, 4 * lay.weights, n_used)?;
                            sums.enqueue_card_acc_tier(
                                stream,
                                &CardAccTier {
                                    down: &down,
                                    trows: &trows,
                                    w: &routed_weights,
                                    sel: &sel,
                                    tsel: &tsel,
                                    n: hidden,
                                    m: 1,
                                    n_card: EXPERTS.div_ceil(3),
                                    n_tier: EXPERTS.div_ceil(3),
                                    n_used,
                                },
                                &mut acc,
                            )?;
                            gpu.elem()
                                .enqueue_add(stream, &acc, &host, hidden, &mut result)
                        };
                        if captured {
                            let graph = gpu.capture(|_| enqueue()).expect("capture");
                            graph.launch(stream).expect("replay");
                            let doubled_weights: Vec<f32> =
                                weights.iter().map(|w| 2.0 * w).collect();
                            weights_d
                                .copy_from_host(stream, &doubled_weights)
                                .expect("new route weights");
                            host.copy_from_host(stream, &doubled_host)
                                .expect("new host sum");
                            graph.launch(stream).expect("second replay");
                            stream.synchronize().expect("replays settled");
                        } else {
                            enqueue().expect("eager handoff and consumption");
                        }
                        let scale = if captured { 2.0 } else { 1.0 };
                        let got = result.to_host_vec(stream).expect("result");
                        let places = sel.to_host_vec(stream).expect("places");
                        let tier_places = tsel.to_host_vec(stream).expect("tier places");
                        let image = image.to_host_vec(stream).expect("host image");
                        let timage = timage.to_host_vec(stream).expect("tier image");
                        stream.synchronize().expect("settled");
                        let fault = gpu.take_fault().expect("fault word");
                        if bad.is_some() {
                            let fault = fault.expect("an invalid id must raise");
                            assert_eq!(fault.layer, LAYER as u32);
                            assert_eq!(fault.code, FaultSite::ExpertId as u32);
                            assert_eq!(fault.sites, 1 << FaultSite::ExpertId as u32);
                        } else {
                            assert!(fault.is_none(), "{fault:?}");
                        }
                        assert_eq!(places[n_used], HOST - 1, "card places tail");
                        assert_eq!(tier_places[n_used], HOST - 1, "tier places tail");
                        assert_eq!(image[lay.seq], 19, "sequence");
                        assert_eq!(&image[lay.ids..lay.ids + n_used], &ids[..n_used]);
                        for s in 0..n_used {
                            assert_eq!(image[lay.weights + s], (weights[s] * scale).to_bits());
                            assert_eq!(timage[tl.sel + s], tier_places[s]);
                        }
                        for d in 0..hidden {
                            assert_eq!(image[lay.x + d], x[d].to_bits());
                            assert_eq!(timage[tl.x + d], x[d].to_bits());
                        }
                        assert!(
                            got.iter()
                                .zip(&want)
                                .all(|(g, w)| g.to_bits() == (w * scale).to_bits()),
                            "{n_used} slots, width {hidden}, bad {bad:?}, capture {captured}: result conservation"
                        );
                    }
                }
            }
        }
    }
}
