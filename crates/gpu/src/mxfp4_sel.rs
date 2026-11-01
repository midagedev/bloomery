//! The FP4 routed-expert kernels: MXFP4's, two forms of one rule
//! ([`crate::mxfp4`]), and NVFP4's pair.
//!
//! The draft's route-plan kernels ([`MxExpertKernels`]), for up to eight
//! tokens per launch: the experts of a model whose stacks hold MXFP4 in the
//! two-plane card layout. The V4.1 DSpark draft is their first caller.
//!
//! The common table's per-slot pair ([`Mxfp4SelKernels`]), shaped like every
//! other `_sel` pair ([`crate::kquant::sel`]): the stacks are the file's
//! 17-byte blocks (`CardFormat::KQuant`, read through five-word windows,
//! [`crate::mxfp4::file_block`]), the activation a [`crate::Q8Act`], and
//! `mxfp4_gate_up_sel` / `mxfp4_gemv_sel` run all slots in one launch each.
//! Their ids, activation rule, fault sink and geometry are the family's, and
//! the block rule is the route-plan kernels', so a (slot, column) of the pair
//! equals the route kernel's value for that (slot, token) bit for bit while
//! both sums are finite; the scale byte follows ggml's half rule for every
//! value. A gate or up sum that is not finite raises
//! [`FaultSite::F32Product`] in both gate-up paths and writes NaN into its
//! row of `h`: the SwiGLU clamp would map a NaN to a finite value.
//!
//! The route-plan kernels, in the order a layer runs them:
//! - `dflash_quantize_q8_1`: the engine's q8_1 activation rule
//!   ([`crate::q8_1_quant_block`]) over any number of columns — the down
//!   projection reads `n_slots · m` of them, past `Q8Act`'s eight.
//! - `dflash_expert_gate_up`: gate·up·SwiGLU over the MXFP4 stacks. A warp
//!   per (slot, row) reads weight row `sel[slot] · rows_per_expert + r` of
//!   both stacks once and dots it with every token column that routes to
//!   the slot ([`crate::mxfp4::lane_partials`]), then the checked SwiGLU
//!   ([`act::apply_checked`]).
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
//! Numeric contract for finite sums (each kernel matches its host
//! transcription bit for bit):
//! - a dot: the MXFP4 × q8_1 rule of [`crate::mxfp4`], then the warp
//!   butterfly;
//! - `h[(slot · m + c) · rows_per_expert + r] = swiglu_clamp(g, u, limit)`,
//!   [`crate::kquant::act::swiglu_clamp`]'s rule with the layer's
//!   `swiglu_clamp_exp`;
//! - `out[c · rows + d]`: from 0, for each slot `j` in order that token `c`
//!   routes to, `acc = fma(dot_j, w, acc)`.
//!
//! The NVFP4 routed pair ([`Nvfp4SelKernels`]): a gate·up and a down
//! `_sel` over stacks in the two-plane card layout
//! (`CardFormat::Nvfp4Planes`, [`crate::mxfp4`]'s
//! [`repack`](crate::mxfp4::repack) under [`Nvfp4`]) with one tensor scale an
//! expert, shaped like every other `_sel` pair ([`crate::kquant::sel`]).
//!
//! - `nvfp4_gate_up_sel`: slot `s` dots the rows of expert `sel[s]` of the
//!   gate and up stacks with column `s / slots_per_col` of a [`Q8Act`]
//!   ([`lane_partials`]), multiplies each reduced dot by its stack's scale of
//!   expert `sel[s]` and writes `act::apply_checked` of the two.
//! - `nvfp4_gemv_sel32`: slot `s` dots the rows of expert `sel[s]` of the
//!   down stack (or, in the relu² form, the up stack) with column `s` of a
//!   [`Q8Blocks32`] ([`lane_partials`],
//!   one scale a 32-value unit), times the stack's scale of expert `sel[s]`.
//!   The 32-value form takes any row width the format does (a multiple of
//!   64), where a `Q8Act` takes whole 256-value super-blocks only.
//! - `nvfp4_gate_up_sel_cols` and `nvfp4_gemv_sel32_cols`: the same two
//!   for a run of more than one token (`n_slots > slots_per_col`), where the
//!   tokens' slots can name one expert (a verify's rows of one sequence). The
//!   first slot that names an expert reads its rows for every slot that
//!   names it, one slot after the other, and the other slots' warps load no
//!   weight; a down warp takes four rows, so it reads each column unit once
//!   for them. Every slot's row is the one-token entry's bit for bit.
//!
//! The rule of a dot is the FP4 owner's under [`Nvfp4`], then the warp
//! butterfly, then one multiply by the expert's scale: the dense gemv's
//! (`nvfp4_gemv`) with the scale read by id. Ids: [`HOST`] returns before any
//! load and writes nothing; any other id at or past the stack's experts
//! raises [`FaultSite::ExpertId`] on the launch's sink, and the gate·up
//! writes NaN into its row of `h` (warp-uniform, before the first weight
//! load); the down leaves its rows of `y` as they were. A scaled gate or up
//! sum that is not finite raises [`FaultSite::F32Product`] and writes NaN
//! into its row of `h`: the SwiGLU clamp would map a NaN to a finite value.

use crate::hybrid::HOST;
use crate::kquant::act::{self, Act};
use crate::kquant::{GateUpAct, SelDown};
use crate::mxfp4::{
    self, Fp4, Mxfp4, NV_BLOCK_VALUES, Nvfp4, Planes, lane_partial_file, lane_partials,
};
use crate::q5::Q8Blocks32;
use crate::{DeviceTensor, FaultSink, FaultSite, GpuError, Q8Act, launch_u32, q8_1_quant_block};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::vector::{U32x4, as_vectors};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Experts each token routes to: compiled into the expert kernels' route
/// layout (the DSpark draft's three).
pub const N_USED: usize = 3;

/// Token columns one expert launch carries.
pub const MAX_TOKENS: usize = 8;

/// Threads per block of the expert kernels: eight warps, a row each.
const BLOCK: u32 = 256;

// The kernels' route masks and column macros stop at eight tokens.
const _: () = assert!(MAX_TOKENS == 8);

/// Output rows a block computes: one per warp.
const ROWS_PER_BLOCK: usize = BLOCK as usize / 32;

/// Rows of one slot a warp of the multi-token NVFP4 down
/// (`nvfp4_gemv_sel32_cols`) takes, so a column unit is read once for them.
const DOWN_COLS_ROWS: usize = 4;

// The entries' launch attributes spell the block as a literal.
const _: () = assert!(BLOCK == 256);

/// Words of a file-layout MXFP4 row a 256-value super-block (eight 17-byte
/// blocks): the `34 · n_sb` of the pair's launch contracts.
const SB_WORDS: usize = 34;
const _: () = assert!(4 * SB_WORDS == 8 * mxfp4::BLOCK_BYTES);

/// The slots `0 .. 3m` in token order: the plan that lists every token's
/// router ids in turn (`sel` = a router's `ids` over `m` tokens).
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

/// Whether a slot before `slot` names `id`. Warp-collective: every lane
/// passes the same `sel`, `id` and `slot` and its own `lane`, and all get the
/// answer.
///
/// # Safety
///
/// `slot <= sel.len()` and `lane < 32`.
#[inline(always)]
unsafe fn names_before(sel: &[u32], id: u32, slot: usize, lane: usize) -> bool {
    let mut i0 = 0;
    while i0 < slot {
        let i = i0 + lane;
        // SAFETY: i < slot <= sel.len() by this fn's contract.
        if warp::any(i < slot && unsafe { *sel.get_unchecked(i) } == id) {
            return true;
        }
        i0 += 32;
    }
    false
}

#[cuda_module]
mod mxfp4_sel_kernels {
    use super::*;

    /// q8_1 activations for `m_cols` columns of `256 · n_sb` values at base
    /// `x0`: `q3k_quantize_q8_1`'s body — one warp per 128-value block, the
    /// engine's rule and its five buffers, a non-finite value raising
    /// [`FaultSite::QuantColumn`] on `fault` — without `Q8Act`'s column cap.
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
        fault: FaultSink,
    ) {
        let blk = thread::index_1d().get() / 32;
        let n_sb = n_sb as usize;
        let blocks_per_col = 2 * n_sb;
        if blk >= m_cols as usize * blocks_per_col {
            return;
        }
        let lane = warp::lane_id() as usize;
        // SAFETY: col < m_cols, b < 2·n_sb (the division), so `requires`
        // bounds x and the planes; `enqueue_quantize` passes n_sb.div_ceil(2)
        // and div_ceil(4); one warp per block index, `lane` its lane id.
        unsafe {
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
                fault,
                FaultSite::QuantColumn,
            )
        };
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
        fault: FaultSink,
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
            lane_partials::<8, Mxfp4, 2>(
                gv, ge, row_abs, n_blk, q, d8, 0, q_col, 0, d_col, mask, lane,
            )
        };
        // SAFETY: the same bounds for the up stack.
        let fu = unsafe {
            lane_partials::<8, Mxfp4, 2>(
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
                        let v = act::apply_checked(act::ACT_SWIGLU_CLAMP, limit, g, u, fault);
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
                    lane_partials::<8, Mxfp4, 2>(
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
}

#[cuda_module]
mod mxfp4_pair_kernels {
    use super::*;

    /// The MXFP4 gate·up with its rule as an argument: thread row `n = slot ·
    /// rows_per_expert + r` (one warp per row, eight rows per block) dots row
    /// `sel[slot] · rows_per_expert + r` of the gate stack and of the up stack
    /// ([`lane_partial_file`], the file's blocks) with activation column `slot
    /// / slots_per_col`, reduces each with the warp tree and stores `h[n] =
    /// act::apply_checked(act, limit, g, u, fault)` from lane 0.
    /// [`HOST`] returns before any
    /// load, writing and raising nothing; any other id at or past
    /// `n_experts` raises [`FaultSite::ExpertId`] and writes NaN into its row
    /// of `h` (warp-uniform, before the first weight load). A gate or up sum
    /// that is not finite raises [`FaultSite::F32Product`] and writes NaN
    /// into its row of `h`: the SwiGLU clamp would map a NaN to a finite
    /// value. `m_cols` only bounds the activation in the launch contract.
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
            wg.len() >= n_experts * rows_per_expert * 34 * n_sb,
            wu.len() >= n_experts * rows_per_expert * 34 * n_sb,
            slots_per_col >= 1,
            n_slots <= m_cols * slots_per_col,
            q.len() >= m_cols * 256 * iters,
            d8.len() >= m_cols * 2 * n_sb,
            sel.len() >= n_slots,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn mxfp4_gate_up_sel(
        wg: &[u32],
        wu: &[u32],
        q: &[u32],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        m_cols: u32,
        slots_per_col: u32,
        n_sb: u32,
        iters: u32,
        act: u32,
        limit: f32,
        fault: FaultSink,
        mut h: DisjointSlice<f32>,
    ) {
        let _ = m_cols;
        let t = thread::index_1d().get() % BLOCK as usize;
        let row = (thread::index_1d().get() / BLOCK as usize) * ROWS_PER_BLOCK + t / 32;
        let rpe = rows_per_expert as usize;
        if row >= n_slots as usize * rpe {
            return;
        }
        let slot = row / rpe;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the
        // load is warp-uniform (the warp's lanes share `row`, hence `slot`),
        // so the returns below never diverge a warp.
        let id = unsafe { *sel.get_unchecked(slot) };
        let lane = warp::lane_id() as usize;
        if id >= n_experts {
            if id != HOST && lane == 0 {
                fault.raise(FaultSite::ExpertId);
                // SAFETY: row < n_slots · rows_per_expert <= h.len() by the
                // launch contract; only lane 0 of the warp writes h[row].
                unsafe { *h.get_unchecked_mut(row) = f32::NAN };
            }
            return;
        }
        let row_abs = id as usize * rpe + row % rpe;
        let col = slot / slots_per_col as usize;
        let n_sb = n_sb as usize;
        let (n_blk, q0, d0) = (8 * n_sb, col * 256 * iters as usize, col * 2 * n_sb);
        // The launch contract keeps lane_partial_file's boundary: row_abs <
        // n_experts · rows_per_expert rows of 34 · n_sb words in both stacks,
        // so each holds row_abs's last block; n_blk = 8 · n_sb is a multiple
        // of 4; column col < m_cols because slot < n_slots <= m_cols ·
        // slots_per_col, its q words end at q0 + 256 · iters and its scales
        // at d0 + 2 · n_sb inside q and d8 (the host passes iters =
        // ceil(n_sb/4) = ceil(n_blk/32)); all 32 lanes of the warp are here
        // (both returns above are warp-uniform).
        let fg = lane_partial_file(wg, row_abs, n_blk, q, d8, q0, d0, lane);
        let fu = lane_partial_file(wu, row_abs, n_blk, q, d8, q0, d0, lane);
        let g = warp::reduce_sum_f32(fg);
        let u = warp::reduce_sum_f32(fu);
        if lane == 0 {
            let v = act::apply_checked(act, limit, g, u, fault);
            // SAFETY: row < n_slots · rows_per_expert <= h.len() by the
            // launch contract; only lane 0 of the warp writes h[row].
            unsafe { *h.get_unchecked_mut(row) = v };
        }
    }

    /// The MXFP4 down `_sel`: thread row `n = slot · rows_per_expert + r`
    /// stores `y[n]`, the dot of row `sel[slot] · rows_per_expert + r` of the
    /// stack ([`lane_partial_file`]) with activation column `slot`, reduced by
    /// the warp tree. [`HOST`] and ids at or past `n_experts` as the gate·up,
    /// but the rows of `y` stay as they were.
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
            w.len() >= n_experts * rows_per_expert * 34 * n_sb,
            q.len() >= n_slots * 256 * iters,
            d8.len() >= n_slots * 2 * n_sb,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn mxfp4_gemv_sel(
        w: &[u32],
        q: &[u32],
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
        let t = thread::index_1d().get() % BLOCK as usize;
        let row = (thread::index_1d().get() / BLOCK as usize) * ROWS_PER_BLOCK + t / 32;
        let rpe = rows_per_expert as usize;
        if row >= n_slots as usize * rpe {
            return;
        }
        let slot = row / rpe;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the
        // load is warp-uniform, so the returns below never diverge a warp.
        let id = unsafe { *sel.get_unchecked(slot) };
        let lane = warp::lane_id() as usize;
        if id >= n_experts {
            if id != HOST && lane == 0 {
                fault.raise(FaultSite::ExpertId);
            }
            return;
        }
        let row_abs = id as usize * rpe + row % rpe;
        let n_sb = n_sb as usize;
        let (n_blk, q0, d0) = (8 * n_sb, slot * 256 * iters as usize, slot * 2 * n_sb);
        // The launch contract keeps lane_partial_file's boundary: row_abs <
        // n_experts · rows_per_expert rows of 34 · n_sb words, so the stack
        // holds its last block; n_blk a multiple of 4; column slot < n_slots
        // of the planes (q0 + 256 · iters <= q.len(), d0 + 2 · n_sb <=
        // d8.len(), iters = ceil(n_sb/4)); all 32 lanes of the warp are here.
        let f = lane_partial_file(w, row_abs, n_blk, q, d8, q0, d0, lane);
        let s = warp::reduce_sum_f32(f);
        if lane == 0 {
            // SAFETY: row < n_slots · rows_per_expert <= y.len() by the
            // launch contract; only lane 0 of the warp writes y[row].
            unsafe { *y.get_unchecked_mut(row) = s };
        }
    }
}

/// An MXFP4 expert stack on the card, in [`crate::mxfp4`]'s card
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
        let Planes { qs, e } = mxfp4::repack::<Mxfp4>(bytes, rows, k)?;
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

    /// Device bytes of the two planes, in allocation order: the codes, then
    /// the scales.
    #[must_use]
    pub fn buffer_bytes(&self) -> [usize; 2] {
        [self.qs.buf().num_bytes(), self.e.buf().num_bytes()]
    }
}

/// q8_1 activation scratch for `n_cols` columns of `k` values, the buffers
/// [`crate::q8_1_quant_block`] writes; the expert kernels read `q4`
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

    /// Device bytes of the five planes, as allocated.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.q3.num_bytes()
            + self.q4.num_bytes()
            + self.q6.num_bytes()
            + self.s8.num_bytes()
            + self.d8.num_bytes()
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

/// One gate·up·SwiGLU launch ([`MxExpertKernels::enqueue_gate_up`]).
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
    pub fault: FaultSink,
}

/// One down-and-combine launch ([`MxExpertKernels::enqueue_down`]).
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

/// The loaded MXFP4 expert module. Owns no context and no stream — every
/// enqueue takes the engine stream, so launches order with the rest of the
/// step and are capturable.
pub struct MxExpertKernels {
    module: mxfp4_sel_kernels::LoadedModule,
}

impl MxExpertKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<MxExpertKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(mxfp4_sel_kernels, ctx)? };
        Ok(MxExpertKernels { module })
    }

    /// Enqueue the q8_1 quantization of `act.n_cols()` columns of
    /// `act.k()` values, `x` column-major (`x[c · k + i]`); a non-finite
    /// value raises `fault`. Asynchronous, allocation-free, capturable.
    pub fn enqueue_quantize(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        act: &mut MxAct,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "MxExpertKernels::enqueue_quantize";
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
            fault,
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
        let what = "MxExpertKernels::enqueue_gate_up";
        crate::tensor::check_alignment::<U32x4, _>(what, "gate.qs", a.gate.qs.buf())?;
        crate::tensor::check_alignment::<U32x4, _>(what, "up.qs", a.up.qs.buf())?;
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
            a.fault,
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
        let what = "MxExpertKernels::enqueue_down";
        crate::tensor::check_alignment::<U32x4, _>(what, "down.qs", a.down.qs.buf())?;
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
}

/// The loaded MXFP4 per-slot pair and its launchers. Owns no stream: every
/// enqueue takes the engine stream, so launches order with the step and are
/// capturable.
pub struct Mxfp4SelKernels {
    module: mxfp4_pair_kernels::LoadedModule,
    /// The fault word of the `Gpu` that owns the context: the launches'
    /// sinks point into it, and the module keeps it alive.
    _fault: Arc<DeviceBuffer<u32>>,
}

/// The experts of a stack of `SB_WORDS · n_sb`-word rows, or the shape error
/// naming `what`.
fn stack_experts(
    what: &'static str,
    w: &DeviceTensor<u32>,
    n_sb: usize,
    rows_per_expert: usize,
) -> Result<usize, GpuError> {
    if w.cols() != SB_WORDS * n_sb {
        return Err(GpuError::shape(
            what,
            format!(
                "MXFP4 rows are {SB_WORDS}*{n_sb} = {} words (the file's 17-byte blocks), got {}",
                SB_WORDS * n_sb,
                w.cols()
            ),
        ));
    }
    if rows_per_expert == 0 || w.rows() == 0 || !w.rows().is_multiple_of(rows_per_expert) {
        return Err(GpuError::shape(
            what,
            format!(
                "w.rows() {} must be a positive multiple of rows_per_expert {rows_per_expert}",
                w.rows()
            ),
        ));
    }
    Ok(w.rows() / rows_per_expert)
}

impl Mxfp4SelKernels {
    /// Load this module's device bundle into `ctx`, whose launches raise into
    /// `word`, the fault word of the `Gpu` that owns `ctx`
    /// ([`crate::Gpu::fault_word`]); a word of another context is refused.
    /// Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        word: &Arc<DeviceBuffer<u32>>,
    ) -> Result<Mxfp4SelKernels, GpuError> {
        let fault = crate::module_fault_word(ctx, word, "Mxfp4SelKernels::load")?;
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(mxfp4_pair_kernels, ctx)? };
        Ok(Mxfp4SelKernels {
            module,
            _fault: fault,
        })
    }

    /// Enqueue the MXFP4 gate·up: slot `s` writes `h[s · rows_per_expert
    /// ..][..rows_per_expert]` as `a.rule` over the rows of expert `sel[s]` of
    /// both stacks (the file's 17-byte blocks as the KQuant word stream, `34 ·
    /// n_sb` words a row) dotted with column `s / slots_per_col` of `a.act`
    /// (module doc for [`HOST`], ids past the stack and sums that are not
    /// finite, which raise on `fault` and write NaN). Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_gate_up(
        &self,
        stream: &CudaStream,
        a: &GateUpAct<'_>,
        fault: FaultSink,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "Mxfp4SelKernels::enqueue_gate_up";
        let n_sb = a.act.n_sb();
        let n_experts = stack_experts(what, a.wg, n_sb, a.rows_per_expert)?;
        let (n_slots, rpe, spc, m) = (a.n_slots, a.rows_per_expert, a.slots_per_col, a.act.m());
        if a.wu.rows() != a.wg.rows() || a.wu.cols() != a.wg.cols() {
            return Err(GpuError::shape(
                what,
                format!(
                    "the up stack {} x {} must be the gate stack's {} x {}",
                    a.wu.rows(),
                    a.wu.cols(),
                    a.wg.rows(),
                    a.wg.cols()
                ),
            ));
        }
        let slot_rows = crate::kquant::sel::check_gate_up_slots(
            what,
            n_slots,
            rpe,
            m,
            spc,
            a.sel.len(),
            h.len(),
        )?;
        let grid = launch_u32(what, "grid", slot_rows.div_ceil(ROWS_PER_BLOCK))?;
        let prep = self
            .module
            .prepare_mxfp4_gate_up_sel(LaunchConfig1D::new(grid, BLOCK, 0))?;
        let (act, limit) = a.rule.code(what)?;
        self.module.mxfp4_gate_up_sel(
            stream,
            &prep,
            a.wg.buf(),
            a.wu.buf(),
            &a.act.q4,
            &a.act.d8,
            a.sel,
            launch_u32(what, "n_experts", n_experts)?,
            launch_u32(what, "rows_per_expert", rpe)?,
            launch_u32(what, "n_slots", n_slots)?,
            launch_u32(what, "m_cols", m)?,
            launch_u32(what, "slots_per_col", spc)?,
            launch_u32(what, "n_sb", n_sb)?,
            launch_u32(what, "iters", n_sb.div_ceil(4))?,
            act,
            limit,
            fault,
            h,
        )?;
        Ok(())
    }

    /// Enqueue the MXFP4 down `_sel`: slot `s` writes `y[s · rows_per_expert
    /// ..][..rows_per_expert]` as the rows of expert `sel[s]` dotted with
    /// column `s` of `a.act` (module doc for [`HOST`] and ids past the stack,
    /// which raise on `fault`). `a.act` holds at least `n_slots` columns;
    /// column `slot` is read for `slot < n_slots` and the rest are not
    /// touched, so a cap-sized activation serves a smaller batch. The family
    /// sums the slots' rows. Asynchronous, allocation-free, capturable.
    pub fn enqueue_down(
        &self,
        stream: &CudaStream,
        a: &SelDown<'_>,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "Mxfp4SelKernels::enqueue_down";
        let n_sb = a.act.n_sb();
        let n_experts = stack_experts(what, a.w, n_sb, a.rows_per_expert)?;
        let (n_slots, rpe) = (a.n_slots, a.rows_per_expert);
        let slot_rows = crate::kquant::sel::check_down_slots(
            what,
            n_slots,
            rpe,
            a.act.m(),
            a.sel.len(),
            y.len(),
        )?;
        let grid = launch_u32(what, "grid", slot_rows.div_ceil(ROWS_PER_BLOCK))?;
        let prep = self
            .module
            .prepare_mxfp4_gemv_sel(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.mxfp4_gemv_sel(
            stream,
            &prep,
            a.w.buf(),
            &a.act.q4,
            &a.act.d8,
            a.sel,
            launch_u32(what, "n_experts", n_experts)?,
            launch_u32(what, "rows_per_expert", rpe)?,
            launch_u32(what, "n_slots", n_slots)?,
            launch_u32(what, "n_sb", n_sb)?,
            launch_u32(what, "iters", n_sb.div_ceil(4))?,
            fault,
            y,
        )?;
        Ok(())
    }
}

#[cuda_module]
mod nvfp4_sel_kernels {
    use super::*;

    /// The NVFP4 gate·up: thread row `n = slot · rows_per_expert + r` (one
    /// warp a row) dots row `sel[slot] · rows_per_expert + r` of both stacks
    /// with activation column `slot / slots_per_col`, reduces each by the warp
    /// tree, scales each by its stack's `sel[slot]` scale and stores `h[n] =
    /// act::apply_checked(act, limit, g, u, fault)` from lane 0. Ids: module doc.
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
            gq.len() >= n_experts * rows_per_expert * 32 * n_sb,
            ge.len() >= n_experts * rows_per_expert * 4 * n_sb,
            gs.len() >= n_experts,
            uq.len() >= n_experts * rows_per_expert * 32 * n_sb,
            ue.len() >= n_experts * rows_per_expert * 4 * n_sb,
            us.len() >= n_experts,
            slots_per_col >= 1,
            n_slots <= m_cols * slots_per_col,
            4 * iters >= n_sb,
            q.len() >= m_cols * 256 * iters,
            d8.len() >= m_cols * 2 * n_sb,
            sel.len() >= n_slots,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn nvfp4_gate_up_sel(
        gq: &[u32],
        ge: &[u32],
        gs: &[f32],
        uq: &[u32],
        ue: &[u32],
        us: &[f32],
        q: &[u32],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        m_cols: u32,
        slots_per_col: u32,
        n_sb: u32,
        iters: u32,
        act: u32,
        limit: f32,
        fault: FaultSink,
        mut h: DisjointSlice<f32>,
    ) {
        let _ = m_cols;
        let t = thread::index_1d().get() % BLOCK as usize;
        let row = (thread::index_1d().get() / BLOCK as usize) * ROWS_PER_BLOCK + t / 32;
        let rpe = rows_per_expert as usize;
        if row >= n_slots as usize * rpe {
            return;
        }
        let slot = row / rpe;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the
        // load is warp-uniform (the warp's lanes share `row`, hence `slot`),
        // so the returns below never diverge a warp.
        let id = unsafe { *sel.get_unchecked(slot) };
        let lane = warp::lane_id() as usize;
        if id >= n_experts {
            if id != HOST && lane == 0 {
                fault.raise(FaultSite::ExpertId);
                // SAFETY: row < n_slots · rows_per_expert <= h.len() by the
                // launch contract; only lane 0 of the warp writes h[row].
                unsafe { *h.get_unchecked_mut(row) = f32::NAN };
            }
            return;
        }
        // The launcher checks 16-byte bases and whole vectors (`k / 8`
        // words a row, a multiple of 32), so the views exist.
        let (Some(gv), Some(uv)) = (as_vectors::<U32x4>(gq), as_vectors::<U32x4>(uq)) else {
            return;
        };
        let row_abs = id as usize * rpe + row % rpe;
        let col = slot / slots_per_col as usize;
        let n_sb = n_sb as usize;
        let (q0, d0) = (col * 256 * iters as usize, col * 2 * n_sb);
        // SAFETY: n_units = 8·n_sb is a multiple of 4; row_abs < n_experts ·
        // rows_per_expert, so the row's 8·n_sb vectors and 4·n_sb scale words
        // are inside both stacks' planes (the stack bounds); column col <
        // m_cols because slot < n_slots <= m_cols · slots_per_col, its
        // 256·iters >= 256·ceil(n_units/32) q words at q0 and 2·n_sb scales
        // at d0 inside q and d8 (the q, d8 and iters bounds); lane < 32; the
        // repack took every scale byte.
        let fg = unsafe {
            lane_partials::<1, Nvfp4, 2>(gv, ge, row_abs, 8 * n_sb, q, d8, q0, 0, d0, 0, 1, lane)
        }[0];
        // SAFETY: row_abs < n_experts*rpe bounds the up row's 8*n_sb vectors and 4*n_sb scale words by the launch contract.
        // col < m_cols and 4*iters >= n_sb bound q/d8; lane < 32; 8*n_sb is divisible by 4; repack checked every scale byte.
        let fu = unsafe {
            lane_partials::<1, Nvfp4, 2>(uv, ue, row_abs, 8 * n_sb, q, d8, q0, 0, d0, 0, 1, lane)
        }[0];
        let g = warp::reduce_sum_f32(fg);
        let u = warp::reduce_sum_f32(fu);
        if lane == 0 {
            // SAFETY: id < n_experts <= gs.len() by the launch contract.
            let sg = unsafe { *gs.get_unchecked(id as usize) };
            // SAFETY: id < n_experts <= us.len() by the launch contract.
            let su = unsafe { *us.get_unchecked(id as usize) };
            let (g, u) = (g * sg, u * su);
            let v = act::apply_checked(act, limit, g, u, fault);
            // SAFETY: row < n_slots · rows_per_expert <= h.len() by the
            // launch contract; only lane 0 of the warp writes h[row].
            unsafe { *h.get_unchecked_mut(row) = v };
        }
    }

    /// `nvfp4_gate_up_sel` over slots of more than one column: thread row
    /// `n = slot · rows_per_expert + r`. The warp of the first slot that names
    /// an expert dots row `r` of the expert's gate and up stacks with the
    /// activation column of each slot that names it, one slot after the
    /// other, and writes each one's row of `h`; the other slots' warps return
    /// before any weight load. A slot's row is `nvfp4_gate_up_sel`'s bit for
    /// bit. Ids: module doc.
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
            gq.len() >= n_experts * rows_per_expert * 32 * n_sb,
            ge.len() >= n_experts * rows_per_expert * 4 * n_sb,
            gs.len() >= n_experts,
            uq.len() >= n_experts * rows_per_expert * 32 * n_sb,
            ue.len() >= n_experts * rows_per_expert * 4 * n_sb,
            us.len() >= n_experts,
            slots_per_col >= 1,
            n_slots <= m_cols * slots_per_col,
            4 * iters >= n_sb,
            q.len() >= m_cols * 256 * iters,
            d8.len() >= m_cols * 2 * n_sb,
            sel.len() >= n_slots,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn nvfp4_gate_up_sel_cols(
        gq: &[u32],
        ge: &[u32],
        gs: &[f32],
        uq: &[u32],
        ue: &[u32],
        us: &[f32],
        q: &[u32],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        m_cols: u32,
        slots_per_col: u32,
        n_sb: u32,
        iters: u32,
        act: u32,
        limit: f32,
        fault: FaultSink,
        mut h: DisjointSlice<f32>,
    ) {
        let _ = m_cols;
        let t = thread::index_1d().get() % BLOCK as usize;
        let row = (thread::index_1d().get() / BLOCK as usize) * ROWS_PER_BLOCK + t / 32;
        let rpe = rows_per_expert as usize;
        if row >= n_slots as usize * rpe {
            return;
        }
        let slot = row / rpe;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the
        // load is warp-uniform, so the returns below never diverge a warp.
        let id = unsafe { *sel.get_unchecked(slot) };
        let lane = warp::lane_id() as usize;
        if id >= n_experts {
            if id != HOST && lane == 0 {
                fault.raise(FaultSite::ExpertId);
                // SAFETY: row < n_slots · rows_per_expert <= h.len() by the
                // launch contract; only lane 0 of the warp writes h[row].
                unsafe { *h.get_unchecked_mut(row) = f32::NAN };
            }
            return;
        }
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; every
        // lane reads the same `id` and `slot`.
        if unsafe { names_before(sel, id, slot, lane) } {
            return;
        }
        // The launcher checks 16-byte bases and whole vectors (`k / 8`
        // words a row, a multiple of 32), so the views exist.
        let (Some(gv), Some(uv)) = (as_vectors::<U32x4>(gq), as_vectors::<U32x4>(uq)) else {
            return;
        };
        let row_abs = id as usize * rpe + row % rpe;
        let (n_slots, spc, n_sb) = (n_slots as usize, slots_per_col as usize, n_sb as usize);
        // SAFETY: id < n_experts <= gs.len() by the launch contract.
        let sg = unsafe { *gs.get_unchecked(id as usize) };
        // SAFETY: id < n_experts <= us.len() by the launch contract.
        let su = unsafe { *us.get_unchecked(id as usize) };
        let mut i0 = slot;
        while i0 < n_slots {
            let i = i0 + lane;
            // SAFETY: i < n_slots <= sel.len() by the launch contract.
            let mut hits = warp::ballot(i < n_slots && unsafe { *sel.get_unchecked(i) } == id);
            while hits != 0 {
                let at = i0 + hits.trailing_zeros() as usize;
                hits &= hits - 1;
                let col = at / spc;
                let (q0, d0) = (col * 256 * iters as usize, col * 2 * n_sb);
                // SAFETY: nvfp4_gate_up_sel's bounds for slot `at` < n_slots:
                // the row's 8·n_sb vectors and 4·n_sb scale words inside both
                // stacks, column col < m_cols with its q words at q0 and its
                // scales at d0 inside q and d8; lane < 32.
                let fg = unsafe {
                    lane_partials::<1, Nvfp4, 2>(
                        gv,
                        ge,
                        row_abs,
                        8 * n_sb,
                        q,
                        d8,
                        q0,
                        0,
                        d0,
                        0,
                        1,
                        lane,
                    )
                }[0];
                // SAFETY: the same bounds for the up stack's row.
                let fu = unsafe {
                    lane_partials::<1, Nvfp4, 2>(
                        uv,
                        ue,
                        row_abs,
                        8 * n_sb,
                        q,
                        d8,
                        q0,
                        0,
                        d0,
                        0,
                        1,
                        lane,
                    )
                }[0];
                let g = warp::reduce_sum_f32(fg);
                let u = warp::reduce_sum_f32(fu);
                if lane == 0 {
                    let v = act::apply_checked(act, limit, g * sg, u * su, fault);
                    // SAFETY: at · rpe + r < n_slots · rows_per_expert <=
                    // h.len() by the launch contract; slot `at` names this
                    // warp's expert and this warp is its first slot, so lane
                    // 0 here is the row's only writer.
                    unsafe { *h.get_unchecked_mut(at * rpe + row % rpe) = v };
                }
            }
            i0 += 32;
        }
    }

    /// The NVFP4 down `_sel` over the 32-value q8_1 form: thread row `n =
    /// slot · rows_per_expert + r` stores `y[n]`, the dot of row `sel[slot] ·
    /// rows_per_expert + r` of the stack ([`lane_partials`], a scale a unit) with
    /// activation column `slot`, reduced by the warp tree, times the stack's
    /// `sel[slot]` scale. `n_blk` is a row's 64-value blocks. Ids: module doc.
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
            qs.len() >= n_experts * rows_per_expert * 8 * n_blk,
            e.len() >= n_experts * rows_per_expert * n_blk,
            s.len() >= n_experts,
            16 * iters >= n_blk,
            q_stride >= 256 * iters,
            q.len() >= n_slots * q_stride,
            d8.len() >= n_slots * 2 * n_blk,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn nvfp4_gemv_sel32(
        qs: &[u32],
        e: &[u32],
        s: &[f32],
        q: &[u32],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        n_blk: u32,
        iters: u32,
        q_stride: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let _ = iters;
        let t = thread::index_1d().get() % BLOCK as usize;
        let row = (thread::index_1d().get() / BLOCK as usize) * ROWS_PER_BLOCK + t / 32;
        let rpe = rows_per_expert as usize;
        if row >= n_slots as usize * rpe {
            return;
        }
        let slot = row / rpe;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the
        // load is warp-uniform, so the returns below never diverge a warp.
        let id = unsafe { *sel.get_unchecked(slot) };
        let lane = warp::lane_id() as usize;
        if id >= n_experts {
            if id != HOST && lane == 0 {
                fault.raise(FaultSite::ExpertId);
            }
            return;
        }
        // The launcher checks a 16-byte base and whole vectors.
        let Some(qv) = as_vectors::<U32x4>(qs) else {
            return;
        };
        let row_abs = id as usize * rpe + row % rpe;
        let n_units = 2 * n_blk as usize;
        // SAFETY: n_units is even, the units an NVFP4 scale word covers;
        // row_abs < n_experts · rows_per_expert, so the row's n_units vectors
        // and n_blk scale words are inside qs and e; column slot < n_slots
        // starts at slot · q_stride with q_stride >= 256 · iters >= 256 ·
        // ceil(n_units/32) words inside q, and its n_units scales at slot ·
        // n_units inside d8 (the q, d8, iters and q_stride bounds); lane < 32;
        // the repack took every scale byte.
        let f = unsafe {
            lane_partials::<1, Nvfp4, 0>(
                qv,
                e,
                row_abs,
                n_units,
                q,
                d8,
                slot * q_stride as usize,
                0,
                slot * n_units,
                0,
                1,
                lane,
            )
        }[0];
        let sum = warp::reduce_sum_f32(f);
        if lane == 0 {
            // SAFETY: id < n_experts <= s.len() by the launch contract.
            let scale = unsafe { *s.get_unchecked(id as usize) };
            // SAFETY: row < n_slots · rows_per_expert <= y.len() by the
            // launch contract; lane 0 of the row's warp is its only writer.
            unsafe { *y.get_unchecked_mut(row) = sum * scale };
        }
    }

    /// `nvfp4_gemv_sel32` over slots of more than one token: warp `w` takes
    /// rows `r0 .. r0 + DOWN_COLS_ROWS` (those below `rows_per_expert`) of
    /// slot `w / ceil(rows_per_expert / DOWN_COLS_ROWS)`. The warp of the
    /// first slot that names an expert dots those rows of the expert with the
    /// activation column of each slot that names it, one slot after the
    /// other, reading each unit of the column once for its rows, and writes
    /// each one's `y`; the other slots' warps return before any weight load.
    /// A slot's row is `nvfp4_gemv_sel32`'s bit for bit: the same units in
    /// the same lanes and order, then the same butterfly. Ids: module doc.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[allow(
        clippy::needless_range_loop,
        reason = "a range `for` is the loop the unroller recognizes; an iterator loop keeps its array in a local depot"
    )]
    #[kernel]
    #[launch_bounds(256, 4)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            qs.len() >= n_experts * rows_per_expert * 8 * n_blk,
            e.len() >= n_experts * rows_per_expert * n_blk,
            s.len() >= n_experts,
            16 * iters >= n_blk,
            q_stride >= 256 * iters,
            q.len() >= n_slots * q_stride,
            d8.len() >= n_slots * 2 * n_blk,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn nvfp4_gemv_sel32_cols(
        qs: &[u32],
        e: &[u32],
        s: &[f32],
        q: &[u32],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        n_blk: u32,
        iters: u32,
        q_stride: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let _ = iters;
        let w = thread::index_1d().get() / 32;
        let rpe = rows_per_expert as usize;
        let n_slots = n_slots as usize;
        let per_slot = rpe.div_ceil(DOWN_COLS_ROWS);
        if w >= n_slots * per_slot {
            return;
        }
        let slot = w / per_slot;
        let r0 = (w % per_slot) * DOWN_COLS_ROWS;
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the
        // load is warp-uniform, so the returns below never diverge a warp.
        let id = unsafe { *sel.get_unchecked(slot) };
        let lane = warp::lane_id() as usize;
        if id >= n_experts {
            if id != HOST && lane == 0 {
                fault.raise(FaultSite::ExpertId);
            }
            return;
        }
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; every
        // lane reads the same `id` and `slot`.
        if unsafe { names_before(sel, id, slot, lane) } {
            return;
        }
        // The launcher checks a 16-byte base and whole vectors.
        let Some(qv) = as_vectors::<U32x4>(qs) else {
            return;
        };
        let n_units = 2 * n_blk as usize;
        let q_stride = q_stride as usize;
        // The warp's first row of the stack: rows row0 + j, j below `live`.
        let row0 = id as usize * rpe + r0;
        let live = (rpe - r0).min(DOWN_COLS_ROWS);
        // SAFETY: id < n_experts <= s.len() by the launch contract.
        let scale = unsafe { *s.get_unchecked(id as usize) };
        let mut i0 = slot;
        while i0 < n_slots {
            let i = i0 + lane;
            // SAFETY: i < n_slots <= sel.len() by the launch contract.
            let mut hits = warp::ballot(i < n_slots && unsafe { *sel.get_unchecked(i) } == id);
            while hits != 0 {
                let at = i0 + hits.trailing_zeros() as usize;
                hits &= hits - 1;
                // lane_partials::<1, Nvfp4, 0> of each live row against
                // column `at`, the column's unit read once for the rows.
                let mut f = [0.0f32; DOWN_COLS_ROWS];
                let mut b = lane;
                while b < n_units {
                    let qb = at * q_stride + 256 * (b >> 5) + (b & 31);
                    // SAFETY: column `at` < n_slots starts at at · q_stride
                    // with q_stride >= 256 · iters >= 256 · ceil(n_units/32)
                    // words (the q, iters and q_stride bounds), so qb + 224
                    // is inside it, inside q; b < n_units.
                    let x = unsafe {
                        [
                            *q.get_unchecked(qb),
                            *q.get_unchecked(qb + 32),
                            *q.get_unchecked(qb + 64),
                            *q.get_unchecked(qb + 96),
                            *q.get_unchecked(qb + 128),
                            *q.get_unchecked(qb + 160),
                            *q.get_unchecked(qb + 192),
                            *q.get_unchecked(qb + 224),
                        ]
                    };
                    // SAFETY: column `at`'s n_units scales start at at ·
                    // n_units inside d8 (the d8 bound) and b < n_units.
                    let dx = unsafe { *d8.get_unchecked(at * n_units + b) };
                    for j in 0..DOWN_COLS_ROWS {
                        thread::__unroll_config::<0>();
                        if j < live {
                            let g = (row0 + j) * n_units + b;
                            // SAFETY: row0 + j < (id + 1) · rows_per_expert
                            // <= n_experts · rows_per_expert and b < n_units,
                            // so vector g is inside qs (the stack bound).
                            let v = unsafe { *qv.get_unchecked(g) };
                            // SAFETY: vector g's scale word is inside e (the
                            // stack bound; n_units is even, the units a scale
                            // word covers); the repack took every scale byte.
                            let ew = unsafe { *e.get_unchecked(Nvfp4::scale_word(g)) };
                            f[j] = Nvfp4::fma(&Nvfp4::decode(v.0, ew, b), &x, dx, f[j]);
                        }
                    }
                    b += 32;
                }
                for j in 0..DOWN_COLS_ROWS {
                    thread::__unroll_config::<0>();
                    if j < live {
                        let sum = warp::reduce_sum_f32(f[j]);
                        if lane == 0 {
                            // SAFETY: at · rpe + r0 + j < n_slots ·
                            // rows_per_expert <= y.len() by the launch
                            // contract; slot `at` names this warp's expert,
                            // this warp is its first slot and alone holds rows
                            // r0 + j, so lane 0 here is the row's only writer.
                            unsafe { *y.get_unchecked_mut(at * rpe + r0 + j) = sum * scale };
                        }
                    }
                }
            }
            i0 += 32;
        }
    }
}

/// A resident routed NVFP4 stack: the planes of its `n_experts ·
/// rows_per_expert` rows (`qs` `k / 8` words a row, `e` `k / 64`) and its
/// experts' tensor scales, one a row of `scales` in the stack's expert order.
#[derive(Clone, Copy)]
pub struct Nvfp4Stack<'a> {
    pub qs: &'a DeviceTensor<u32>,
    pub e: &'a DeviceTensor<u32>,
    pub scales: &'a DeviceTensor<f32>,
}

impl Nvfp4Stack<'_> {
    /// The stack's experts, its planes holding `k` values a row in whole
    /// experts of `rows_per_expert` rows with one scale each, or the shape
    /// error naming `what`.
    pub(crate) fn experts(
        &self,
        what: &'static str,
        k: usize,
        rows_per_expert: usize,
    ) -> Result<usize, GpuError> {
        crate::tensor::check_alignment::<U32x4, _>(what, "NVFP4 code plane", self.qs.buf())?;
        let rows = self.qs.rows();
        let n = rows.checked_div(rows_per_expert).unwrap_or(0);
        if k == 0
            || !k.is_multiple_of(NV_BLOCK_VALUES)
            || self.qs.cols() != k / 8
            || self.e.cols() != k / NV_BLOCK_VALUES
            || self.e.rows() != rows
            || n == 0
            || n * rows_per_expert != rows
            || self.scales.buf().len() != n
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "NVFP4 stack at K={k}: planes of {} and {} words a row in whole experts of \
                     {rows_per_expert} rows, one scale an expert; got {rows} x {}, {} x {} and \
                     {} scales",
                    k / 8,
                    k / NV_BLOCK_VALUES,
                    self.qs.cols(),
                    self.e.rows(),
                    self.e.cols(),
                    self.scales.buf().len()
                ),
            ));
        }
        Ok(n)
    }
}

/// An NVFP4 gate·up launch ([`Nvfp4SelKernels::enqueue_gate_up`]).
pub struct Nvfp4GateUp<'a> {
    pub gate: Nvfp4Stack<'a>,
    /// The up stack, the gate's shape.
    pub up: Nvfp4Stack<'a>,
    /// `act.m()` q8_1 columns; slot `s` reads column `s / slots_per_col`.
    pub act: &'a Q8Act,
    /// At least `n_slots` ids, read on the device at each launch.
    pub sel: &'a DeviceBuffer<u32>,
    pub n_slots: usize,
    pub rows_per_expert: usize,
    /// Slots a column: a token's run of slots.
    pub slots_per_col: usize,
    pub rule: Act,
}

/// An NVFP4 `_sel` launch over the 32-value activation ([`Nvfp4SelKernels::enqueue_sel32`]): a
/// routed down, the up of the relu² form (an up is a down-shaped `_sel`, slot `s` over column
/// `s`) or a dense site's one expert.
pub struct Nvfp4Down<'a> {
    pub w: Nvfp4Stack<'a>,
    /// At least `n_slots` 32-value q8_1 columns: column `s` is read for `s <
    /// n_slots` and the rest are not touched, so a cap-sized activation
    /// serves a smaller batch.
    pub act: &'a Q8Blocks32,
    /// At least `n_slots` ids, read on the device at each launch.
    pub sel: &'a DeviceBuffer<u32>,
    pub n_slots: usize,
    pub rows_per_expert: usize,
    /// Slots per token. Multi-token runs compute matching slots' dots
    /// serially in the first matching slot's warp for each of its expert rows.
    pub slots_per_col: usize,
}

/// The loaded NVFP4 pair and its launchers. Owns no stream: every enqueue
/// takes the engine stream, so launches order with the step and are
/// capturable.
pub struct Nvfp4SelKernels {
    module: nvfp4_sel_kernels::LoadedModule,
    /// The fault word of the `Gpu` that owns the context: the launches'
    /// sinks point into it, and the module keeps it alive.
    _fault: Arc<DeviceBuffer<u32>>,
}

impl Nvfp4SelKernels {
    /// Load this module's device bundle into `ctx`, whose launches raise into
    /// `word`, the fault word of the `Gpu` that owns `ctx`
    /// ([`crate::Gpu::fault_word`]); a word of another context is refused.
    /// Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        word: &Arc<DeviceBuffer<u32>>,
    ) -> Result<Nvfp4SelKernels, GpuError> {
        let fault = crate::module_fault_word(ctx, word, "Nvfp4SelKernels::load")?;
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(nvfp4_sel_kernels, ctx)? };
        Ok(Nvfp4SelKernels {
            module,
            _fault: fault,
        })
    }

    /// Enqueue the NVFP4 gate·up: slot `s` writes `h[s · rows_per_expert
    /// ..][..rows_per_expert]` (module doc). Stacks of another shape than
    /// the activation's width, an up unlike the gate, or slots the buffers
    /// cannot hold are named errors. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_gate_up(
        &self,
        stream: &CudaStream,
        a: &Nvfp4GateUp<'_>,
        fault: FaultSink,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        use crate::kquant::sel::{Fmt, GateUpAct, gate_up_dims};

        let what = "Nvfp4SelKernels::enqueue_gate_up";
        a.gate.experts(what, a.act.k(), a.rows_per_expert)?;
        a.up.experts(what, a.act.k(), a.rows_per_expert)?;
        let common = GateUpAct {
            wg: a.gate.qs,
            wu: a.up.qs,
            act: a.act,
            sel: a.sel,
            n_slots: a.n_slots,
            rows_per_expert: a.rows_per_expert,
            slots_per_col: a.slots_per_col,
            rule: a.rule,
        };
        let d = gate_up_dims(
            what,
            Fmt {
                name: "NVFP4 code planes",
                words: 256 / 8,
            },
            &common,
            h.len(),
        )?;
        macro_rules! launch {
            ($prepare:ident, $entry:ident) => {{
                let prep = self
                    .module
                    .$prepare(LaunchConfig1D::new(d.grid, BLOCK, 0))?;
                self.module.$entry(
                    stream,
                    &prep,
                    a.gate.qs.buf(),
                    a.gate.e.buf(),
                    a.gate.scales.buf(),
                    a.up.qs.buf(),
                    a.up.e.buf(),
                    a.up.scales.buf(),
                    &a.act.q4,
                    &a.act.d8,
                    a.sel,
                    d.n_experts,
                    d.rows_per_expert,
                    d.n_slots,
                    d.m_cols,
                    d.slots_per_col,
                    d.n_sb,
                    d.iters,
                    d.act,
                    d.limit,
                    fault,
                    h,
                )?
            }};
        }
        if a.n_slots > a.slots_per_col {
            launch!(prepare_nvfp4_gate_up_sel_cols, nvfp4_gate_up_sel_cols);
        } else {
            launch!(prepare_nvfp4_gate_up_sel, nvfp4_gate_up_sel);
        }
        Ok(())
    }

    /// Enqueue the NVFP4 `_sel` over the 32-value activation: slot `s` writes
    /// `y[s · rows_per_expert ..][..rows_per_expert]` (module doc). The launch of
    /// a routed down, of the relu² form's up and of a dense site read as a stack of
    /// one expert; the rows are the stack's, whatever the role. A stack of another
    /// width than the activation's, fewer activation columns or ids than slots, or
    /// a short `y` are named errors. Asynchronous, allocation-free, capturable.
    pub fn enqueue_sel32(
        &self,
        stream: &CudaStream,
        a: &Nvfp4Down<'_>,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "Nvfp4SelKernels::enqueue_sel32";
        let (n_slots, rpe, k) = (a.n_slots, a.rows_per_expert, a.act.k());
        if a.slots_per_col == 0 {
            return Err(GpuError::shape(
                what,
                "slots_per_col 0: a token has at least one slot",
            ));
        }
        let n_experts = a.w.experts(what, k, rpe)?;
        let slot_rows = crate::kquant::sel::check_down_slots(
            what,
            n_slots,
            rpe,
            a.act.m(),
            a.sel.len(),
            y.len(),
        )?;
        let n_blk = k / NV_BLOCK_VALUES;
        let cols = n_slots > a.slots_per_col;
        let warps = if cols {
            n_slots * rpe.div_ceil(DOWN_COLS_ROWS)
        } else {
            slot_rows
        };
        let grid = launch_u32(what, "grid", warps.div_ceil(ROWS_PER_BLOCK))?;
        macro_rules! launch {
            ($prepare:ident, $entry:ident) => {{
                let prep = self.module.$prepare(LaunchConfig1D::new(grid, BLOCK, 0))?;
                self.module.$entry(
                    stream,
                    &prep,
                    a.w.qs.buf(),
                    a.w.e.buf(),
                    a.w.scales.buf(),
                    &a.act.q,
                    &a.act.d8,
                    a.sel,
                    launch_u32(what, "n_experts", n_experts)?,
                    launch_u32(what, "rows_per_expert", rpe)?,
                    launch_u32(what, "n_slots", n_slots)?,
                    launch_u32(what, "n_blk", n_blk)?,
                    launch_u32(what, "iters", n_blk.div_ceil(16))?,
                    launch_u32(what, "q_stride", a.act.q_stride())?,
                    fault,
                    y,
                )?
            }};
        }
        if cols {
            launch!(prepare_nvfp4_gemv_sel32_cols, nvfp4_gemv_sel32_cols);
        } else {
            launch!(prepare_nvfp4_gemv_sel32, nvfp4_gemv_sel32);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gpu;
    use crate::fault::{Fault, LAYER_NONE};
    use crate::kquant::Act;

    #[test]
    #[ignore = "needs a CUDA device"]
    fn hw_mxfp4_refuses_unaligned_vector_planes() {
        let gpu = Gpu::new().expect("card");
        let ctx = gpu.context();
        let stream = gpu.stream();
        let kernels = MxExpertKernels::load(ctx).expect("MXFP4 kernels");
        let act = MxAct::new(stream, 1, 256).expect("activation");
        let sel = DeviceBuffer::from_host(stream, &[0u32]).expect("selection");
        let route = DeviceBuffer::from_host(stream, &[0u32; N_USED]).expect("route");
        let wts = DeviceBuffer::from_host(stream, &[1.0f32; N_USED]).expect("weights");
        let parent = DeviceBuffer::<u32>::zeroed(stream, 33).expect("codes");
        let e = DeviceTensor::zeroed(stream, 1, 2).expect("scales");
        // SAFETY: the window covers 32 u32-aligned words of parent, which
        // lives through the launches; release below prevents its free.
        let qs = unsafe { DeviceTensor::window(parent.cu_deviceptr() + 4, 1, 32, ctx) };
        let w = std::mem::ManuallyDrop::new(MxStack {
            qs: std::mem::ManuallyDrop::into_inner(qs),
            e,
            rows_per_expert: 1,
            k: 256,
        });
        let mut bytes = [0; 8 * mxfp4::BLOCK_BYTES];
        for block in bytes.as_chunks_mut::<{ mxfp4::BLOCK_BYTES }>().0 {
            block[0] = 127;
        }
        let aligned = MxStack::upload(stream, &bytes, 1, 1, 256).expect("aligned stack");
        let mut y = DeviceBuffer::zeroed(stream, 1).expect("output");
        let gate = kernels.enqueue_gate_up(
            stream,
            &GateUpArgs {
                gate: &w,
                up: &aligned,
                act: &act,
                sel: &sel,
                route: &route,
                n_slots: 1,
                limit: 0.0,
                fault: gpu.unlabelled_sink(),
            },
            &mut y,
        );
        let up = kernels.enqueue_gate_up(
            stream,
            &GateUpArgs {
                gate: &aligned,
                up: &w,
                act: &act,
                sel: &sel,
                route: &route,
                n_slots: 1,
                limit: 0.0,
                fault: gpu.unlabelled_sink(),
            },
            &mut y,
        );
        let down = kernels.enqueue_down(
            stream,
            &DownArgs {
                down: &w,
                act: &act,
                sel: &sel,
                route: &route,
                wts: &wts,
                n_slots: 1,
                m: 1,
            },
            &mut y,
        );
        stream.synchronize().expect("launches finished");
        let w = std::mem::ManuallyDrop::into_inner(w);
        DeviceTensor::release(std::mem::ManuallyDrop::new(w.qs));
        drop(w.e);
        let mut failures = Vec::new();
        for (name, result) in [("gate.qs", gate), ("up.qs", up), ("down.qs", down)] {
            println!("MXFP4 {name}: {result:?}");
            if !matches!(&result, Err(err) if err.to_string().contains(name)
                && err.to_string().contains("16"))
            {
                failures.push(format!("{name}: {result:?}"));
            }
        }
        kernels
            .enqueue_gate_up(
                stream,
                &GateUpArgs {
                    gate: &aligned,
                    up: &aligned,
                    act: &act,
                    sel: &sel,
                    route: &route,
                    n_slots: 1,
                    limit: 0.0,
                    fault: gpu.unlabelled_sink(),
                },
                &mut y,
            )
            .expect("aligned gate/up");
        assert_eq!(y.to_host_vec(stream).expect("gate/up output"), [0.0]);
        kernels
            .enqueue_down(
                stream,
                &DownArgs {
                    down: &aligned,
                    act: &act,
                    sel: &sel,
                    route: &route,
                    wts: &wts,
                    n_slots: 1,
                    m: 1,
                },
                &mut y,
            )
            .expect("aligned down");
        assert_eq!(y.to_host_vec(stream).expect("down output"), [0.0]);
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// An MXFP4 gate·up whose up sum overflows on a finite column of 1e38:
    /// K = 256, scale byte 128 in both stacks, gate codes 0, up codes 0x77
    /// (+12) over the first four blocks and 0xff (−12) over the last four.
    /// The up blocks reach +inf and −inf and the warp sum NaN, which the
    /// clamp at 7 alone maps to a finite 0.
    #[test]
    #[ignore = "needs a CUDA card"]
    fn hw_mxfp4_gate_up_sel_faults_a_non_finite_sum() {
        const K: usize = 256;
        const ROWS: usize = 256;
        const EXPERTS: usize = 3;
        let gpu = Gpu::new().expect("card");
        let stream = gpu.stream();
        let pair = Mxfp4SelKernels::load(gpu.context(), gpu.fault_word()).expect("pair");
        let route = MxExpertKernels::load(gpu.context()).expect("route");
        let bytes = |head: u8, tail: u8| {
            let block = |codes| {
                let mut b = vec![128u8];
                b.resize(mxfp4::BLOCK_BYTES, codes);
                b.repeat(4)
            };
            [block(head), block(tail)].concat().repeat(EXPERTS * ROWS)
        };
        let (gb, ub) = (bytes(0, 0), bytes(0x77, 0xff));
        let words = |b: &[u8]| {
            let w = crate::card_leg::tests::words_of(b, EXPERTS * ROWS);
            DeviceTensor::upload(stream, &w, EXPERTS * ROWS, w.len() / (EXPERTS * ROWS))
                .expect("words")
        };
        let (wg, wu) = (words(&gb), words(&ub));
        let gate = MxStack::upload(stream, &gb, EXPERTS, ROWS, K).expect("gate");
        let up = MxStack::upload(stream, &ub, EXPERTS, ROWS, K).expect("up");
        let x = DeviceBuffer::from_host(stream, &[1.0e38f32; K]).expect("column");
        let mut act = Q8Act::with_k(stream, 1, K).expect("activation");
        let mut mx = MxAct::new(stream, 1, K).expect("route activation");
        gpu.enqueue_quantize_q8_1(&x, &mut act).expect("quantize");
        route
            .enqueue_quantize(stream, &x, &mut mx, gpu.unlabelled_sink())
            .expect("route quantize");
        assert_eq!(gpu.take_fault().expect("quantizer fault"), None);
        let sel = DeviceBuffer::from_host(stream, &[0u32, 1, 2]).expect("route");
        let mut failures = Vec::new();
        for routed in [false, true] {
            let mut h = DeviceBuffer::zeroed(stream, EXPERTS * ROWS).expect("rows");
            if routed {
                route
                    .enqueue_gate_up(
                        stream,
                        &GateUpArgs {
                            gate: &gate,
                            up: &up,
                            act: &mx,
                            sel: &sel,
                            route: &sel,
                            n_slots: EXPERTS,
                            limit: 7.0,
                            fault: gpu.unlabelled_sink(),
                        },
                        &mut h,
                    )
                    .expect("route launch");
            } else {
                pair.enqueue_gate_up(
                    stream,
                    &GateUpAct {
                        wg: &wg,
                        wu: &wu,
                        act: &act,
                        sel: &sel,
                        n_slots: EXPERTS,
                        rows_per_expert: ROWS,
                        slots_per_col: EXPERTS,
                        rule: Act::SwigluClamp { limit: 7.0 },
                    },
                    gpu.unlabelled_sink(),
                    &mut h,
                )
                .expect("pair launch");
            }
            let fault = gpu.take_fault().expect("fault word");
            let rows = h.to_host_vec(stream).expect("rows");
            if fault != Some(Fault::at(LAYER_NONE, FaultSite::F32Product))
                || !rows.iter().all(|v| v.is_nan())
            {
                failures.push(format!(
                    "routed={routed}: fault {fault:?}, first row {}",
                    rows[0]
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    #[ignore = "needs a CUDA card"]
    fn hw_nvfp4_sel_refuses_unaligned_code_planes() {
        let gpu = Gpu::new().expect("card");
        let stream = gpu.stream();
        let kernels = Nvfp4SelKernels::load(gpu.context(), gpu.fault_word()).expect("kernels");
        let parent = DeviceBuffer::from_host(stream, &[0u32; 33]).expect("parent");
        let bad = DeviceTensor::<u32>::window_of(&parent, 4, 1, 32).expect("offset window");
        let qs = DeviceTensor::zeroed(stream, 1, 32).expect("codes");
        let e = DeviceTensor::zeroed(stream, 1, 4).expect("block scales");
        let scales = DeviceTensor::upload(stream, &[1.0f32], 1, 1).expect("expert scale");
        let stack = |qs| Nvfp4Stack {
            qs,
            e: &e,
            scales: &scales,
        };
        let act = Q8Act::with_slots(stream, 1, 256).expect("gate activation");
        let down = Q8Blocks32::new(stream, 256, 1).expect("down activation");
        let sel = DeviceBuffer::from_host(stream, &[0u32]).expect("id");
        let mut out = DeviceBuffer::from_host(stream, &[f32::NAN]).expect("output");
        let fault = gpu.layer_sink(0).expect("fault sink");
        let mut refusals = Vec::new();
        for (gate, up) in [(&*bad, &qs), (&qs, &*bad)] {
            refusals.push(kernels.enqueue_gate_up(
                stream,
                &Nvfp4GateUp {
                    gate: stack(gate),
                    up: stack(up),
                    act: &act,
                    sel: &sel,
                    n_slots: 1,
                    rows_per_expert: 1,
                    slots_per_col: 1,
                    rule: Act::SiluMul,
                },
                fault,
                &mut out,
            ));
        }
        refusals.push(kernels.enqueue_sel32(
            stream,
            &Nvfp4Down {
                w: stack(&bad),
                act: &down,
                sel: &sel,
                n_slots: 1,
                rows_per_expert: 1,
                slots_per_col: 1,
            },
            fault,
            &mut out,
        ));
        stream.synchronize().expect("launches");
        assert!(
            refusals.iter().all(|r| matches!(r,
                Err(GpuError::Shape { detail, .. }) if detail.contains("16-byte aligned")
            )),
            "gate, up and down refusals: {refusals:?}"
        );
        let zero = kernels.enqueue_sel32(
            stream,
            &Nvfp4Down {
                w: stack(&qs),
                act: &down,
                sel: &sel,
                n_slots: 1,
                rows_per_expert: 1,
                slots_per_col: 0,
            },
            fault,
            &mut out,
        );
        assert!(
            matches!(&zero, Err(GpuError::Shape { detail, .. }) if detail.contains("slots_per_col 0")),
            "a down of no slots a token: {zero:?}"
        );
    }

    /// The K units a tensor-parallel plan keeps a rank's part whole in
    /// (`kernels::card_k_unit`) are the card layouts' own: an MXFP4 stack
    /// and MXFP8 planes load at that K and refuse half of it.
    #[test]
    #[ignore = "needs a CUDA card"]
    fn hw_card_k_units_are_the_layouts_own() {
        use gguf::GgmlType;
        use model::placement::Role;
        use model::placement::kernels::card_k_unit;
        let gpu = Gpu::new().expect("card");
        let mx = card_k_unit(GgmlType::MXFP4, Role::RoutedExperts) as usize;
        let blocks = |k: usize| {
            let block = [vec![127u8], vec![0u8; 16]].concat();
            block.repeat(k / 32)
        };
        assert!(MxStack::upload(gpu.stream(), &blocks(mx), 1, 1, mx).is_ok());
        assert!(MxStack::upload(gpu.stream(), &blocks(mx / 2), 1, 1, mx / 2).is_err());
        let m8 = card_k_unit(GgmlType::MXFP8, Role::DenseFfn) as usize;
        let planes = |k: usize| {
            crate::q8f32::e4m3_planes(
                crate::q8f32::E4m3Scales::Mxfp8,
                &vec![0u8; k / 32 * 33],
                1,
                k,
            )
        };
        assert!(planes(m8).is_ok() && planes(m8 / 2).is_err());
    }
}
