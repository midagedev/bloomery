//! The DSpark draft's routed MoE: its router (128 experts, 3 used) and its
//! MXFP4 experts, for up to eight tokens per launch.
//!
//! Kernels, in the order a layer runs them:
//! - `dflash_router`: `ds41_router`'s body with this model's constants — the
//!   `f32_gemv` m = 1 row per expert, [`sqrt_softplus`], the selection bias,
//!   three rounds of the butterfly argmax under `take::<true>` (an equal
//!   value goes to the larger id), the weights from the unbiased scores. One
//!   token per launch; token `tok` reads `x[tok·k ..]` and writes its logits,
//!   scores, ids and weights at `tok`'s rows of a [`DraftRouterOut`], so `m`
//!   launches fill one plan.
//! - `dflash_quantize_q8_1`: the engine's q8_1 activation rule
//!   (`bloomery_gpu::q8_1_quant_block`) over any number of columns — the down
//!   projection reads `n_slots · m` of them, past `Q8Act`'s eight.
//! - `dflash_expert_gate_up`: gate·up·SwiGLU over the MXFP4 stacks. A warp
//!   per (slot, row) reads weight row `sel[slot] · rows_per_expert + r` of
//!   both stacks once and dots it with every token column that routes to
//!   the slot (`bloomery_gpu::mxfp4::lane_partials`), then [`swiglu_clamp`].
//! - `dflash_expert_down`: the down projection and the weighted combine in
//!   one pass. A warp per output row `d` walks the slots in order; slot `j`'s
//!   row of expert `sel[j]` is read once and dotted with column `(j, c)` of
//!   the q8_1 of `h` for every token `c` routed to it, and each dot enters
//!   token `c`'s sum by one fused multiply-add with its weight.
//!
//! The plan. `sel` lists `n_slots` expert ids (the stacks' own expert order),
//! and `route[c · N_USED + u]` names the slot that holds token `c`'s `u`-th
//! expert, with its weight at `wts[c · N_USED + u]`. A token names a slot at
//! most once. A (slot, token) pair no route names costs no dot: its `h`
//! column is written as 0 and the down pass skips it. One token is `sel` =
//! the router's ids and `route` = `[0, 1, 2]`; `m` tokens can list every
//! token's ids in turn ([`concat_route`], `n_slots = 3m`), or list each
//! expert once and route the tokens that share it to one slot, which reads
//! its rows once for all of them.
//!
//! Numeric contract (the gate holds each kernel to its host transcription
//! bit for bit):
//! - a dot: the MXFP4 × q8_1 rule of `bloomery_gpu::mxfp4`, then the warp
//!   butterfly;
//! - `h[(slot · m + c) · rows_per_expert + r] = swiglu_clamp(g, u, limit)`,
//!   `crate::experts`' rule with the layer's `swiglu_clamp_exp`;
//! - `out[c · rows + d]`: from 0, for each slot `j` in order that token `c`
//!   routes to, `acc = fma(dot_j, w, acc)`.

use crate::experts::swiglu_clamp;
use bloomery_gpu::mxfp4::{self, Planes, lane_partials};
use bloomery_gpu::q8f32::f32_lane_partial_1col;
/// ik's expert score in f32; the gate's host side simulates the device with it.
pub use bloomery_gpu::route_core::sqrt_softplus;
use bloomery_gpu::route_core::{renorm_divisor, take};
use bloomery_gpu::{DeviceTensor, GpuError, launch_u32, q8_1_quant_block};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::vector::{U32x4, as_vectors};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, threadfence, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Experts the draft's router scores (`expert_count`): the width of the
/// selecting block's shared arrays.
pub const N_EXPERT: usize = 128;

/// Experts each token routes to (`expert_used_count`).
pub const N_USED: usize = 3;

/// Token columns one expert launch carries.
pub const MAX_TOKENS: usize = 8;

/// Threads per block of the expert kernels: eight warps, a row each.
const BLOCK: u32 = 256;

/// Threads per router block, eight warps, one expert row per warp.
const ROUTER_THREADS: usize = 256;
const ROUTER_THREADS_U32: u32 = ROUTER_THREADS as u32;
const ROWS_PER_BLOCK: usize = ROUTER_THREADS / 32;

/// Scores each lane of the selecting warp owns: experts `lane + 32 j`.
const PER_LANE: usize = N_EXPERT / 32;

const _: () = assert!(ROUTER_THREADS_U32 as usize == ROUTER_THREADS);
const _: () = assert!(N_EXPERT.is_multiple_of(32) && N_EXPERT.is_multiple_of(ROWS_PER_BLOCK));
// The selecting lane keeps its taken entries as bits of one u32.
const _: () = assert!(PER_LANE <= 32);
// The kernels' route masks and column macros stop at eight tokens.
const _: () = assert!(MAX_TOKENS == 8);

/// The slots `0 .. 3m` in token order: the plan that lists every token's
/// router ids in turn (`sel` = a [`DraftRouterOut`]'s `ids` over `m`
/// tokens).
#[must_use]
pub fn concat_route(m: usize) -> Vec<u32> {
    (0..(N_USED * m) as u32).collect()
}

/// The token columns that route to slot `j`, one bit per token `c < m`.
///
/// # Safety
///
/// `route.len() >= N_USED · m` and `m <= 8`.
#[inline(always)]
unsafe fn slot_mask(route: &[u32], j: u32, m: usize) -> u32 {
    let mut mask = 0u32;
    let mut i = 0usize;
    while i < N_USED * m {
        // SAFETY: i < N_USED·m <= route.len() by this fn's contract.
        if unsafe { *route.get_unchecked(i) } == j {
            mask |= 1 << (i / N_USED);
        }
        i += 1;
    }
    mask
}

/// Token `c`'s weight for slot `j`, 0.0 where it does not route there.
///
/// # Safety
///
/// `route.len()` and `wts.len()` are at least `N_USED · (c + 1)`.
#[inline(always)]
unsafe fn slot_weight(route: &[u32], wts: &[f32], j: u32, c: usize) -> f32 {
    let mut w = 0.0f32;
    let mut u = 0usize;
    while u < N_USED {
        let at = c * N_USED + u;
        // SAFETY: at < N_USED·(c+1) <= route.len() and wts.len() by this
        // fn's contract.
        if unsafe { *route.get_unchecked(at) } == j {
            // SAFETY: as above.
            w = unsafe { *wts.get_unchecked(at) };
        }
        u += 1;
    }
    w
}

#[cuda_module]
mod dflash_kernels {
    use super::*;

    /// q8_1 activations for `m_cols` columns of `256 · n_sb` values at base
    /// `x0`: `q3k_quantize_q8_1`'s body — one warp per 128-value block, the
    /// engine's rule and its five buffers — without `Q8Act`'s column cap.
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
            x.len() >= x0 + m_cols * 256 * n_sb,
            q3.len() >= m_cols * 64 * half_it,
            q4.len() >= m_cols * 256 * quad_it,
            q6.len() >= m_cols * 128 * half_it,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb
        )
    )]
    pub fn dflash_quantize_q8_1(
        x: &[f32],
        x0: u32,
        m_cols: u32,
        n_sb: u32,
        half_it: u32,
        quad_it: u32,
        mut q3: DisjointSlice<u64>,
        mut q4: DisjointSlice<u32>,
        mut q6: DisjointSlice<u32>,
        mut s8: DisjointSlice<i32>,
        mut d8: DisjointSlice<f32>,
    ) {
        let blk = thread::index_1d().get() / 32;
        let n_sb = n_sb as usize;
        let blocks_per_col = 2 * n_sb;
        if blk >= m_cols as usize * blocks_per_col {
            return;
        }
        let lane = warp::lane_id() as usize;
        // SAFETY: col < m_cols and b < 2·n_sb by the division; the launch
        // contract carries the rest of `q8_1_quant_block`'s preconditions,
        // and the block index is warp-uniform.
        q8_1_quant_block(
            x,
            x0 as usize,
            blk / blocks_per_col,
            blk % blocks_per_col,
            n_sb,
            half_it,
            quad_it,
            lane,
            &mut q3,
            &mut q4,
            &mut q6,
            &mut s8,
            &mut d8,
        );
    }

    /// Gate·up·SwiGLU over MXFP4 stacks for `m_cols` token columns: thread
    /// row `n = slot · rows_per_expert + r` (one warp per row, eight rows per
    /// block) reads row `sel[slot] · rows_per_expert + r` of both stacks and
    /// dots it with every token column the slot's route mask holds; lane 0
    /// stores `h[(slot · m_cols + c) · rows_per_expert + r]` for every `c <
    /// m_cols` (0.0 for a column outside the mask). An id `>= n_experts`
    /// returns before any load — warp-uniform — and leaves its slot of `h`
    /// as it was.
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
            gqs.len() >= n_experts * rows_per_expert * 4 * n_blk,
            4 * ge.len() >= n_experts * rows_per_expert * n_blk,
            uqs.len() >= n_experts * rows_per_expert * 4 * n_blk,
            4 * ue.len() >= n_experts * rows_per_expert * n_blk,
            q.len() >= m_cols * 256 * n_grp,
            4 * d8.len() >= m_cols * n_blk,
            32 * n_grp >= n_blk,
            sel.len() >= n_slots,
            route.len() >= 3 * m_cols,
            m_cols >= 1,
            m_cols <= 8,
            h.len() >= n_slots * m_cols * rows_per_expert
        )
    )]
    pub fn dflash_expert_gate_up(
        gqs: &[u32],
        ge: &[u32],
        uqs: &[u32],
        ue: &[u32],
        q: &[u32],
        d8: &[f32],
        sel: &[u32],
        route: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        m_cols: u32,
        n_blk: u32,
        n_grp: u32,
        limit: f32,
        mut h: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        let rpe = rows_per_expert as usize;
        if row >= n_slots as usize * rpe {
            return;
        }
        let slot = row / rpe;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract. All 32
        // lanes share `row`, hence `slot` and `id`: the return is warp-uniform.
        let id = unsafe { *sel.get_unchecked(slot) } as usize;
        if id >= n_experts as usize {
            return;
        }
        let (Some(gv), Some(uv)) = (as_vectors::<U32x4>(gqs), as_vectors::<U32x4>(uqs)) else {
            return;
        };
        let m = m_cols as usize;
        // SAFETY: route.len() >= 3·m_cols and m_cols <= 8 by the launch contract.
        let mask = unsafe { slot_mask(route, slot as u32, m) };
        let row_abs = id * rpe + row % rpe;
        let lane = warp::lane_id() as usize;
        let (n_blk, q_col, d_col) = (n_blk as usize, 256 * n_grp as usize, n_blk as usize >> 2);
        // SAFETY: row_abs < n_experts·rows_per_expert rows of 4·n_blk words
        // and n_blk/4 scale words (the contract's first four bounds); column
        // c < m_cols has 256·n_grp >= 256·ceil(n_blk/32) q words at c·q_col
        // and n_blk/4 scales at c·d_col (the q, d8 and n_grp bounds); the
        // host passes n_blk a multiple of 4 and scales repacked in range.
        let fg = unsafe {
            lane_partials::<8>(
                gv, ge, row_abs, n_blk, q, d8, 0, q_col, 0, d_col, mask, lane,
            )
        };
        // SAFETY: the same bounds for the up stack.
        let fu = unsafe {
            lane_partials::<8>(
                uv, ue, row_abs, n_blk, q, d8, 0, q_col, 0, d_col, mask, lane,
            )
        };
        let base = slot * m * rpe + row % rpe;
        macro_rules! col {
            ($c:literal) => {
                if $c < m {
                    let (g, u) = if (mask >> $c) & 1 != 0 {
                        (warp::reduce_sum_f32(fg[$c]), warp::reduce_sum_f32(fu[$c]))
                    } else {
                        (0.0, 0.0)
                    };
                    if lane == 0 {
                        let v = swiglu_clamp(g, u, limit);
                        // SAFETY: (slot·m + c)·rpe + r < n_slots·m_cols·rpe
                        // <= h.len() by the launch contract; lane 0 of the
                        // row's warp is the slot's only writer.
                        unsafe { *h.get_unchecked_mut(base + $c * rpe) = v };
                    }
                }
            };
        }
        col!(0);
        col!(1);
        col!(2);
        col!(3);
        col!(4);
        col!(5);
        col!(6);
        col!(7);
    }

    /// The down projection and the weighted combine for `m_cols` tokens:
    /// output row `d` (one warp per row, eight rows per block) walks the
    /// slots in order; for slot `j` with an id below `n_experts` and a
    /// non-empty route mask, row `sel[j] · rows_per_expert + d` of the stack
    /// is dotted with q8_1 column `j · m_cols + c` for each token `c` in the
    /// mask, and token `c`'s sum takes `fma(dot, w_jc, acc)`. Lane 0 stores
    /// `out[c · rows_per_expert + d]` for every `c < m_cols`.
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
            wqs.len() >= n_experts * rows_per_expert * 4 * n_blk,
            4 * we.len() >= n_experts * rows_per_expert * n_blk,
            q.len() >= n_slots * m_cols * 256 * n_grp,
            4 * d8.len() >= n_slots * m_cols * n_blk,
            32 * n_grp >= n_blk,
            sel.len() >= n_slots,
            route.len() >= 3 * m_cols,
            wts.len() >= 3 * m_cols,
            m_cols >= 1,
            m_cols <= 8,
            out.len() >= m_cols * rows_per_expert
        )
    )]
    pub fn dflash_expert_down(
        wqs: &[u32],
        we: &[u32],
        q: &[u32],
        d8: &[f32],
        sel: &[u32],
        route: &[u32],
        wts: &[f32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        m_cols: u32,
        n_blk: u32,
        n_grp: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let d = (thread::index_1d().get() / 256) * 8 + t / 32;
        let rpe = rows_per_expert as usize;
        if d >= rpe {
            return;
        }
        let Some(wv) = as_vectors::<U32x4>(wqs) else {
            return;
        };
        let m = m_cols as usize;
        let lane = warp::lane_id() as usize;
        let (n_blk, q_col, d_col) = (n_blk as usize, 256 * n_grp as usize, n_blk as usize >> 2);
        let mut acc = [0.0f32; 8];
        let mut j = 0usize;
        while j < n_slots as usize {
            // SAFETY: j < n_slots <= sel.len() by the launch contract; every
            // lane reads the same id, so the skip below is warp-uniform.
            let id = unsafe { *sel.get_unchecked(j) } as usize;
            // SAFETY: route.len() >= 3·m_cols and m_cols <= 8 by the contract.
            let mask = unsafe { slot_mask(route, j as u32, m) };
            if id < n_experts as usize && mask != 0 {
                // SAFETY: row id·rpe + d < n_experts·rows_per_expert (the
                // stack bounds); columns j·m + c < n_slots·m_cols start at
                // (j·m + c)·q_col and (j·m + c)·d_col inside q and d8 (the
                // q, d8 and n_grp bounds); n_blk a multiple of 4 and scales
                // in range by the host.
                let f = unsafe {
                    lane_partials::<8>(
                        wv,
                        we,
                        id * rpe + d,
                        n_blk,
                        q,
                        d8,
                        j * m * q_col,
                        q_col,
                        j * m * d_col,
                        d_col,
                        mask,
                        lane,
                    )
                };
                macro_rules! col {
                    ($c:literal) => {
                        if $c < m && (mask >> $c) & 1 != 0 {
                            let s = warp::reduce_sum_f32(f[$c]);
                            // SAFETY: route and wts hold 3·m_cols >= 3·($c+1)
                            // entries by the launch contract.
                            let w = unsafe { slot_weight(route, wts, j as u32, $c) };
                            acc[$c] = s.mul_add(w, acc[$c]);
                        }
                    };
                }
                col!(0);
                col!(1);
                col!(2);
                col!(3);
                col!(4);
                col!(5);
                col!(6);
                col!(7);
            }
            j += 1;
        }
        if lane == 0 {
            macro_rules! store {
                ($c:literal) => {
                    if $c < m {
                        // SAFETY: c·rpe + d < m_cols·rows_per_expert <=
                        // out.len() by the launch contract; lane 0 of row
                        // d's warp is the only writer.
                        unsafe { *out.get_unchecked_mut($c * rpe + d) = acc[$c] };
                    }
                };
            }
            store!(0);
            store!(1);
            store!(2);
            store!(3);
            store!(4);
            store!(5);
            store!(6);
            store!(7);
        }
    }

    /// The draft's router for token `tok`: `n_expert` (=
    /// [`N_EXPERT`]) rows of `k` f32 in `w`, the token's `k` activations at
    /// `x[tok·k ..]`, the selection bias in `bias`. Writes `logits` and
    /// `probs` (the score) at `tok · N_EXPERT + e` for every expert, and
    /// `ids`/`weights` at `tok · N_USED + s` for the slots in rank order;
    /// the weights are `score / sum · scale` with `norm != 0`, the sum
    /// guarded by [`renorm_divisor`], and `score · scale` without.
    /// `done[0]` is the block ticket count: zero before the
    /// launch, zero again after it. The body is `ds41_router`'s (its doc
    /// states the ticket protocol and the selection).
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
            n_expert == 128,
            w.len() >= n_expert * k,
            x.len() >= tok * k + k,
            bias.len() >= n_expert,
            logits.len() >= tok * n_expert + n_expert,
            probs.len() >= tok * n_expert + n_expert,
            ids.len() >= 3 * tok + 3,
            weights.len() >= 3 * tok + 3,
            done.len() >= 1
        )
    )]
    pub fn dflash_router(
        w: &[f32],
        x: &[f32],
        bias: &[f32],
        n_expert: u32,
        k: u32,
        tok: u32,
        scale: f32,
        norm: u32,
        mut logits: DisjointSlice<f32>,
        mut probs: DisjointSlice<f32>,
        mut ids: DisjointSlice<u32>,
        mut weights: DisjointSlice<f32>,
        mut done: DisjointSlice<u32>,
    ) {
        static mut SCORE: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
        static mut SEL_V: SharedArray<f32, N_EXPERT> = SharedArray::UNINIT;
        static mut LAST: SharedArray<u32, 1> = SharedArray::UNINIT;
        static mut SLOT_P: SharedArray<f32, N_USED> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = warp::lane_id() as usize;
        let row = thread::blockIdx_x() as usize * ROWS_PER_BLOCK + tid / 32;
        let e0 = tok as usize * N_EXPERT;
        let s0 = tok as usize * N_USED;
        if row < n_expert as usize {
            let x0 = tok as usize * k as usize;
            // SAFETY: x.len() >= tok·k + k = x0 + k by the launch contract.
            let xt = unsafe { x.get_unchecked(x0..x0 + k as usize) };
            // `f32_gemv`'s m = 1 row: the same body and the same tree.
            // SAFETY: row < n_expert, so w.len() >= n_expert * k >= (row + 1) *
            // k, and xt.len() = k, by the launch contract; the launcher
            // passes k a positive multiple of 32; lane < 32.
            let partial = unsafe { f32_lane_partial_1col(w, xt, k, row, lane) };
            let logit = warp::reduce_sum_f32(partial);
            if lane == 0 {
                // SAFETY: e0 + row < tok·n_expert + n_expert <= logits.len()
                // by the launch contract; lane 0 of the row's warp is the only writer.
                unsafe { *logits.get_unchecked_mut(e0 + row) = logit };
                let score = sqrt_softplus(logit);
                // SAFETY: e0 + row < tok·n_expert + n_expert <= probs.len()
                // by the launch contract; lane 0 of the row's warp is the only writer.
                unsafe { *probs.get_unchecked_mut(e0 + row) = score };
            }
        }
        threadfence();
        thread::sync_threads();

        // SAFETY: block-shared, one element; the raw form reaches the
        // `static mut` without a reference.
        let last = unsafe { SharedArray::as_raw_mut_ptr(&raw mut LAST) };
        let ticket = done.as_mut_ptr();
        if tid == 0 {
            // SAFETY: `ticket` is done[0], inside `done` by the launch
            // contract; every access to it is atomic.
            let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
            let is_last = count.fetch_add(1, AtomicOrdering::AcqRel) + 1 == thread::gridDim_x();
            // SAFETY: block-shared, one element, written before the barrier
            // that publishes it.
            unsafe { *last = u32::from(is_last) };
        }
        thread::sync_threads();
        // SAFETY: block-shared, one element, written by thread 0 before the
        // barrier above.
        if unsafe { *last } == 0 {
            return;
        }

        // The last block: every block's rows are published.
        // SAFETY: block-shared, N_EXPERT entries; the raw form reaches the
        // `static mut` without a reference.
        let score = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SCORE) };
        // SAFETY: block-shared, N_EXPERT entries; the raw form reaches the
        // `static mut` without a reference.
        let sel_v = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SEL_V) };
        // SAFETY: block-shared, N_USED entries; the raw form reaches the
        // `static mut` without a reference.
        let slot_p = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SLOT_P) };
        let probs_ptr = probs.as_mut_ptr();
        let mut e = tid;
        while e < N_EXPERT {
            // SAFETY: e0 + e < tok·N_EXPERT + N_EXPERT <= probs.len() by the
            // launch contract; the volatile load reads the row where its block left it.
            let p = unsafe { core::ptr::read_volatile(probs_ptr.add(e0 + e)) };
            // SAFETY: e < N_EXPERT = n_expert <= bias.len() by the launch
            // contract.
            let b = unsafe { *bias.get_unchecked(e) };
            // SAFETY: block-shared, e < N_EXPERT; thread `tid` writes entries
            // tid + 256·i only, before the barrier that publishes them.
            unsafe { *score.add(e) = p };
            // SAFETY: block-shared, e < N_EXPERT; thread `tid` writes entries
            // tid + 256·i only, before the barrier that publishes them.
            unsafe { *sel_v.add(e) = p + b };
            e += ROUTER_THREADS;
        }
        thread::sync_threads();

        if tid < 32 {
            let mut taken = 0u32;
            let mut s = 0usize;
            while s < N_USED {
                let mut bv = f32::NEG_INFINITY;
                let mut bi = 0u32;
                let mut j = 0usize;
                while j < PER_LANE {
                    if (taken >> j) & 1 == 0 {
                        let ej = lane + 32 * j;
                        // SAFETY: block-shared, ej < 32·PER_LANE = N_EXPERT,
                        // written before the barrier above.
                        let v = unsafe { *sel_v.add(ej) };
                        if take::<true>(v, ej as u32, bv, bi) {
                            bv = v;
                            bi = ej as u32;
                        }
                    }
                    j += 1;
                }
                let mut off = 16u32;
                while off > 0 {
                    let (ov, oi) = (warp::shuffle_xor_f32(bv, off), warp::shuffle_xor(bi, off));
                    if take::<true>(ov, oi, bv, bi) {
                        bv = ov;
                        bi = oi;
                    }
                    off >>= 1;
                }
                // Every lane leaves the butterfly with the same winner; its
                // owner marks it taken.
                if bi as usize % 32 == lane {
                    taken |= 1 << (bi as usize / 32);
                }
                if lane == 0 {
                    // SAFETY: block-shared, bi < N_EXPERT (every candidate is
                    // one of a lane's own entries), written before the barrier.
                    let p = unsafe { *score.add(bi as usize) };
                    // SAFETY: block-shared, s < N_USED; lane 0 alone writes and
                    // reads these entries.
                    unsafe { *slot_p.add(s) = p };
                    // SAFETY: s0 + s < 3·tok + 3 <= ids.len() by the launch
                    // contract; lane 0 is the only writer.
                    unsafe { *ids.get_unchecked_mut(s0 + s) = bi };
                }
                s += 1;
            }
            if lane == 0 {
                let mut sum = 0.0f64;
                let mut s = 0usize;
                while s < N_USED {
                    // SAFETY: block-shared, s < N_USED, this lane's own write.
                    sum += f64::from(unsafe { *slot_p.add(s) });
                    s += 1;
                }
                let sum = renorm_divisor(sum as f32);
                let mut s = 0usize;
                while s < N_USED {
                    // SAFETY: block-shared, s < N_USED, this lane's own write.
                    let p = unsafe { *slot_p.add(s) };
                    let v = if norm != 0 { p / sum } else { p };
                    // SAFETY: s0 + s < 3·tok + 3 <= weights.len() by the launch
                    // contract; lane 0 is the only writer.
                    unsafe { *weights.get_unchecked_mut(s0 + s) = v * scale };
                    s += 1;
                }
                // SAFETY: `ticket` is done[0], inside `done` by the launch
                // contract; every block has drawn its ticket by now.
                let count = unsafe { DeviceAtomicU32::from_ptr(ticket) };
                count.store(0, AtomicOrdering::Relaxed);
            }
        }
    }
}

/// An MXFP4 expert stack on the card, in `bloomery_gpu::mxfp4`'s card
/// layout: `n_experts · rows_per_expert` rows of `k` values.
pub struct MxStack {
    qs: DeviceTensor<u32>,
    e: DeviceTensor<u32>,
    rows_per_expert: usize,
    k: usize,
}

impl MxStack {
    /// Repack the file bytes of a `[k, rows_per_expert, n_experts]` MXFP4
    /// tensor (`mxfp4::repack`, which refuses a scale byte out of range) and
    /// upload both planes. `k` a multiple of 256 (the q8_1 activation's
    /// block). Load-time only.
    pub fn upload(
        stream: &CudaStream,
        bytes: &[u8],
        n_experts: usize,
        rows_per_expert: usize,
        k: usize,
    ) -> Result<MxStack, GpuError> {
        if !k.is_multiple_of(256) || n_experts == 0 || rows_per_expert == 0 {
            return Err(GpuError::Shape {
                what: "MxStack::upload",
                detail: format!(
                    "k {k} must be a multiple of 256, with {n_experts} experts of \
                     {rows_per_expert} rows both positive"
                ),
            });
        }
        let rows = n_experts * rows_per_expert;
        let Planes { qs, e } = mxfp4::repack(bytes, rows, k)?;
        let n_blk = k / mxfp4::BLOCK_VALUES;
        Ok(MxStack {
            qs: DeviceTensor::upload(stream, &qs, rows, mxfp4::QS_WORDS * n_blk)?,
            e: DeviceTensor::upload(stream, &e, rows, n_blk / 4)?,
            rows_per_expert,
            k,
        })
    }

    /// Experts in the stack.
    pub fn n_experts(&self) -> usize {
        self.qs.rows() / self.rows_per_expert
    }

    /// Rows per expert.
    pub fn rows_per_expert(&self) -> usize {
        self.rows_per_expert
    }

    /// Values per row.
    pub fn k(&self) -> usize {
        self.k
    }
}

/// q8_1 activation scratch for `n_cols` columns of `k` values, the buffers
/// `bloomery_gpu::q8_1_quant_block` writes; the expert kernels read `q4`
/// and `d8`. Allocated once.
pub struct MxAct {
    q3: DeviceBuffer<u64>,
    q4: DeviceBuffer<u32>,
    q6: DeviceBuffer<u32>,
    s8: DeviceBuffer<i32>,
    d8: DeviceBuffer<f32>,
    n_cols: usize,
    k: usize,
}

impl MxAct {
    /// Scratch for `n_cols >= 1` columns of `k` values, `k` a positive
    /// multiple of 256. Load-time only.
    pub fn new(stream: &CudaStream, n_cols: usize, k: usize) -> Result<MxAct, GpuError> {
        if n_cols == 0 || k == 0 || !k.is_multiple_of(256) {
            return Err(GpuError::Shape {
                what: "MxAct::new",
                detail: format!("{n_cols} columns of {k}: want >= 1 of a multiple of 256"),
            });
        }
        let n_sb = k / 256;
        Ok(MxAct {
            q3: DeviceBuffer::zeroed(stream, n_cols * 64 * n_sb.div_ceil(2))?,
            q4: DeviceBuffer::zeroed(stream, n_cols * 256 * n_sb.div_ceil(4))?,
            q6: DeviceBuffer::zeroed(stream, n_cols * 128 * n_sb.div_ceil(2))?,
            s8: DeviceBuffer::zeroed(stream, n_cols * 8 * n_sb)?,
            d8: DeviceBuffer::zeroed(stream, n_cols * 2 * n_sb)?,
            n_cols,
            k,
        })
    }

    /// Columns.
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Values per column.
    pub fn k(&self) -> usize {
        self.k
    }

    /// The int8 codes in the Q4_K gemv's permutation: `256 · ceil(k / 1024)`
    /// words per column.
    pub fn q4(&self) -> &DeviceBuffer<u32> {
        &self.q4
    }

    /// The 128-value block scales: `k / 128` per column.
    pub fn d8(&self) -> &DeviceBuffer<f32> {
        &self.d8
    }
}

/// Where router launches leave their results for up to `n_tok` tokens,
/// allocated once: per token the logits and scores of every expert (`tok ·
/// N_EXPERT + e`), the ids and weights of its slots (`tok · N_USED + s`),
/// and the block ticket count each launch returns to zero. Launches that
/// share one are ordered on one stream.
pub struct DraftRouterOut {
    pub logits: DeviceBuffer<f32>,
    pub probs: DeviceBuffer<f32>,
    pub ids: DeviceBuffer<u32>,
    pub weights: DeviceBuffer<f32>,
    done: DeviceBuffer<u32>,
    n_tok: usize,
}

impl DraftRouterOut {
    /// Allocate for `n_tok` tokens, the ticket count zeroed. Load-time only.
    pub fn new(stream: &CudaStream, n_tok: usize) -> Result<DraftRouterOut, GpuError> {
        if n_tok == 0 {
            return Err(GpuError::Shape {
                what: "DraftRouterOut::new",
                detail: "zero tokens".into(),
            });
        }
        Ok(DraftRouterOut {
            logits: DeviceBuffer::zeroed(stream, n_tok * N_EXPERT)?,
            probs: DeviceBuffer::zeroed(stream, n_tok * N_EXPERT)?,
            ids: DeviceBuffer::zeroed(stream, n_tok * N_USED)?,
            weights: DeviceBuffer::zeroed(stream, n_tok * N_USED)?,
            done: DeviceBuffer::zeroed(stream, 1)?,
            n_tok,
        })
    }

    /// The ticket count as it stands: zero between launches.
    pub fn tickets(&self, stream: &CudaStream) -> Result<u32, GpuError> {
        Ok(self.done.to_host_vec(stream)?[0])
    }
}

/// One router launch ([`DraftExpertKernels::enqueue_router`]).
pub struct RouterArgs<'a> {
    /// `ffn_gate_inp` as f32: [`N_EXPERT`] rows of `k`, `k` a positive
    /// multiple of 32.
    pub w: &'a DeviceTensor<f32>,
    /// At least `(tok + 1) · k` activations; the token's are `x[tok · k ..]`.
    pub x: &'a DeviceBuffer<f32>,
    /// `exp_probs_b`: [`N_EXPERT`] selection biases.
    pub bias: &'a DeviceBuffer<f32>,
    /// `expert_weights_scale`.
    pub scale: f32,
    /// `expert_weights_norm`.
    pub norm: bool,
    /// The token's row in the output.
    pub tok: usize,
}

/// One gate·up·SwiGLU launch ([`DraftExpertKernels::enqueue_gate_up`]).
pub struct GateUpArgs<'a> {
    pub gate: &'a MxStack,
    pub up: &'a MxStack,
    /// `m` columns of the stacks' `k`, one per token.
    pub act: &'a MxAct,
    /// `n_slots` expert ids, read on the device at each launch.
    pub sel: &'a DeviceBuffer<u32>,
    /// `N_USED · m` slot indices (the module doc's plan).
    pub route: &'a DeviceBuffer<u32>,
    pub n_slots: usize,
    /// The layer's `swiglu_clamp_exp` value.
    pub limit: f32,
}

/// One down-and-combine launch ([`DraftExpertKernels::enqueue_down`]).
pub struct DownArgs<'a> {
    pub down: &'a MxStack,
    /// `n_slots · m` columns of the stack's `k`: the q8_1 of `h`.
    pub act: &'a MxAct,
    pub sel: &'a DeviceBuffer<u32>,
    pub route: &'a DeviceBuffer<u32>,
    /// `N_USED · m` weights, token `c`'s at `c · N_USED + u`.
    pub wts: &'a DeviceBuffer<f32>,
    pub n_slots: usize,
    /// Tokens.
    pub m: usize,
}

/// The loaded draft MoE module. Owns no context and no stream — every
/// enqueue takes the engine stream, so launches order with the rest of the
/// step and are capturable.
pub struct DraftExpertKernels {
    module: dflash_kernels::LoadedModule,
}

impl DraftExpertKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<DraftExpertKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { dflash_kernels::load(ctx)? };
        Ok(DraftExpertKernels { module })
    }

    /// Enqueue the q8_1 quantization of `act.n_cols()` columns of
    /// `act.k()` values, `x` column-major (`x[c · k + i]`). Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_quantize(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        act: &mut MxAct,
    ) -> Result<(), GpuError> {
        let what = "DraftExpertKernels::enqueue_quantize";
        let (n_cols, k) = (act.n_cols, act.k);
        if x.len() < n_cols * k {
            return Err(GpuError::Shape {
                what,
                detail: format!("x.len() {} < {n_cols} columns of {k}", x.len()),
            });
        }
        let n_sb = k / 256;
        let grid = launch_u32(what, "grid", n_cols * 2 * n_sb)?;
        let prep = self
            .module
            .prepare_dflash_quantize_q8_1(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.dflash_quantize_q8_1(
            stream,
            &prep,
            x,
            0,
            launch_u32(what, "m_cols", n_cols)?,
            launch_u32(what, "n_sb", n_sb)?,
            launch_u32(what, "half_it", n_sb.div_ceil(2))?,
            launch_u32(what, "quad_it", n_sb.div_ceil(4))?,
            &mut act.q3,
            &mut act.q4,
            &mut act.q6,
            &mut act.s8,
            &mut act.d8,
        )?;
        Ok(())
    }

    /// Enqueue gate·up·SwiGLU for `a.act.n_cols()` tokens over `a.n_slots`
    /// slots: `h` takes `n_slots · m · rows_per_expert` f32, slot-major then
    /// token-major. Asynchronous, allocation-free, capturable.
    pub fn enqueue_gate_up(
        &self,
        stream: &CudaStream,
        a: &GateUpArgs<'_>,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "DraftExpertKernels::enqueue_gate_up";
        let shape = |detail: String| GpuError::Shape { what, detail };
        let (g, u, m, n_slots) = (a.gate, a.up, a.act.n_cols, a.n_slots);
        let rpe = g.rows_per_expert;
        if u.rows_per_expert != rpe
            || u.k != g.k
            || u.n_experts() != g.n_experts()
            || a.act.k != g.k
        {
            return Err(shape(format!(
                "gate {}x{rpe}x{}, up {}x{}x{} and act k {} must match",
                g.n_experts(),
                g.k,
                u.n_experts(),
                u.rows_per_expert,
                u.k,
                a.act.k
            )));
        }
        if !(1..=MAX_TOKENS).contains(&m)
            || n_slots == 0
            || a.sel.len() < n_slots
            || a.route.len() < N_USED * m
            || h.len() < n_slots * m * rpe
        {
            return Err(shape(format!(
                "m {m} (1..={MAX_TOKENS}), n_slots {n_slots}: sel {} route {} h {} (need {})",
                a.sel.len(),
                a.route.len(),
                h.len(),
                n_slots * m * rpe
            )));
        }
        let n_blk = g.k / mxfp4::BLOCK_VALUES;
        let grid = launch_u32(what, "grid", (n_slots * rpe).div_ceil(8))?;
        let prep = self
            .module
            .prepare_dflash_expert_gate_up(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.dflash_expert_gate_up(
            stream,
            &prep,
            g.qs.buf(),
            g.e.buf(),
            u.qs.buf(),
            u.e.buf(),
            &a.act.q4,
            &a.act.d8,
            a.sel,
            a.route,
            launch_u32(what, "n_experts", g.n_experts())?,
            launch_u32(what, "rows_per_expert", rpe)?,
            launch_u32(what, "n_slots", n_slots)?,
            launch_u32(what, "m_cols", m)?,
            launch_u32(what, "n_blk", n_blk)?,
            launch_u32(what, "n_grp", n_blk.div_ceil(32))?,
            a.limit,
            h,
        )?;
        Ok(())
    }

    /// Enqueue the down projection and the weighted combine for `a.m`
    /// tokens: `out` takes `m · rows_per_expert` f32, token-major.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_down(
        &self,
        stream: &CudaStream,
        a: &DownArgs<'_>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "DraftExpertKernels::enqueue_down";
        let (w, m, n_slots) = (a.down, a.m, a.n_slots);
        let rpe = w.rows_per_expert;
        if !(1..=MAX_TOKENS).contains(&m)
            || n_slots == 0
            || a.act.k != w.k
            || a.act.n_cols != n_slots * m
            || a.sel.len() < n_slots
            || a.route.len() < N_USED * m
            || a.wts.len() < N_USED * m
            || out.len() < m * rpe
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "m {m} (1..={MAX_TOKENS}), n_slots {n_slots}: act {}x{} (want {} of {}), \
                     sel {} route {} wts {} out {} (need {})",
                    a.act.n_cols,
                    a.act.k,
                    n_slots * m,
                    w.k,
                    a.sel.len(),
                    a.route.len(),
                    a.wts.len(),
                    out.len(),
                    m * rpe
                ),
            });
        }
        let n_blk = w.k / mxfp4::BLOCK_VALUES;
        let grid = launch_u32(what, "grid", rpe.div_ceil(8))?;
        let prep = self
            .module
            .prepare_dflash_expert_down(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.dflash_expert_down(
            stream,
            &prep,
            w.qs.buf(),
            w.e.buf(),
            &a.act.q4,
            &a.act.d8,
            a.sel,
            a.route,
            a.wts,
            launch_u32(what, "n_experts", w.n_experts())?,
            launch_u32(what, "rows_per_expert", rpe)?,
            launch_u32(what, "n_slots", n_slots)?,
            launch_u32(what, "m_cols", m)?,
            launch_u32(what, "n_blk", n_blk)?,
            launch_u32(what, "n_grp", n_blk.div_ceil(32))?,
            out,
        )?;
        Ok(())
    }

    /// Enqueue the router for token `a.tok`; results into `out` at that
    /// token's rows. Asynchronous, allocation-free, capturable.
    pub fn enqueue_router(
        &self,
        stream: &CudaStream,
        a: &RouterArgs<'_>,
        out: &mut DraftRouterOut,
    ) -> Result<(), GpuError> {
        let what = "DraftExpertKernels::enqueue_router";
        let shape = |detail: String| GpuError::Shape { what, detail };
        let k = a.w.cols();
        if a.w.rows() != N_EXPERT || k == 0 || !k.is_multiple_of(32) {
            return Err(shape(format!(
                "router weight is {} x {k}, want {N_EXPERT} rows of a positive multiple of 32",
                a.w.rows()
            )));
        }
        if a.tok >= out.n_tok || a.x.len() < (a.tok + 1) * k || a.bias.len() < N_EXPERT {
            return Err(shape(format!(
                "tok {} of {} tokens: x.len() {} (need {}), bias.len() {} (need {N_EXPERT})",
                a.tok,
                out.n_tok,
                a.x.len(),
                (a.tok + 1) * k,
                a.bias.len()
            )));
        }
        let grid = launch_u32(what, "grid", N_EXPERT / ROWS_PER_BLOCK)?;
        let prep =
            self.module
                .prepare_dflash_router(LaunchConfig1D::new(grid, ROUTER_THREADS_U32, 0))?;
        self.module.dflash_router(
            stream,
            &prep,
            a.w.buf(),
            a.x,
            a.bias,
            launch_u32(what, "n_expert", N_EXPERT)?,
            launch_u32(what, "k", k)?,
            launch_u32(what, "tok", a.tok)?,
            a.scale,
            u32::from(a.norm),
            &mut out.logits,
            &mut out.probs,
            &mut out.ids,
            &mut out.weights,
            &mut out.done,
        )?;
        Ok(())
    }
}
