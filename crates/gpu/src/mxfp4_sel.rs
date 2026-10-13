//! The MXFP4 routed-expert kernels, two forms of one rule ([`crate::mxfp4`]).
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
//! equals the route kernel's value for that (slot, token) bit for bit; the
//! scale byte follows ggml's half rule for every value.
//!
//! The slot tiles (`mxfp4_gate_up_tiles`, `mxfp4_gemv_tiles`), the pair's
//! grouped form for a prompt batch's card slots: over a grouped slot table
//! (`crate::q4k_sel`'s module doc: the slots by expert, cut into tiles of up to
//! eight consecutive slots of one expert) a block takes a tile and eight
//! weight rows, so each file block is read and decoded once for up to eight
//! slots. Entry `c` of a tile reads the activation column of its own slot —
//! the gate·up's token column `slot / slots_per_col` of the normed rows' q8_1,
//! the down's column `slot` of the slots' — and writes row `r` of its own
//! slot, so nothing is gathered and the rows are the pair's, bit for bit: the
//! dot of a (row, column) is the pair's [`lane_partial_file`] rule whatever the
//! tile holds, and the slot-order sum of the leg that reads the rows is
//! untouched. A slot the table does not hold (a place that is [`HOST`]) is
//! left as it was, as the pair leaves it. [`Mxfp4SelKernels::enqueue_gate_up_tiles`]
//! and [`Mxfp4SelKernels::enqueue_down_tiles`] launch them.
//!
//! The route-plan kernels, in the order a layer runs them:
//! - `dflash_quantize_q8_1`: the engine's q8_1 activation rule
//!   ([`crate::q8_1_quant_block`]) over any number of columns — the down
//!   projection reads `n_slots · m` of them, past `Q8Act`'s eight.
//! - `dflash_expert_gate_up`: gate·up·SwiGLU over the MXFP4 stacks. A warp
//!   per (slot, row) reads weight row `sel[slot] · rows_per_expert + r` of
//!   both stacks once and dots it with every token column that routes to
//!   the slot ([`crate::mxfp4::lane_partials`]), then [`swiglu_clamp`].
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
//! - a dot: the MXFP4 × q8_1 rule of [`crate::mxfp4`], then the warp
//!   butterfly;
//! - `h[(slot · m + c) · rows_per_expert + r] = swiglu_clamp(g, u, limit)`,
//!   [`crate::kquant::act::swiglu_clamp`]'s rule with the layer's
//!   `swiglu_clamp_exp`;
//! - `out[c · rows + d]`: from 0, for each slot `j` in order that token `c`
//!   routes to, `acc = fma(dot_j, w, acc)`.

use crate::hybrid::HOST;
use crate::kquant::act::{self, swiglu_clamp};
use crate::kquant::tiles::{TileTable, tile_grid};
use crate::kquant::{Act, GateUpAct, SelDown};
use crate::mxfp4::{self, Planes, lane_partial_file, lane_partials, lane_partials_file_cols};
use crate::q4k_sel::{TileBlock, TileWork, tile_block};
use crate::{
    DeviceTensor, FaultSink, FaultSite, GpuError, Q8Act, col_sums, launch_u32, q8_1_quant_block,
};
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
}

/// The q8_1 column each of a tile's `m` entries `j .. j + m` of `order` reads,
/// `slot / per_col`: the tile's [`TileWork`] resolves its columns through the
/// table, so the activation stays where its quantizer wrote it. A table entry
/// past `n_slots` raises [`FaultSite::ExpertId`] on `fault` (lane 0) and the
/// whole tile is skipped: the same entries, so the same answer, in every lane
/// of the warp.
///
/// # Safety
///
/// `j + m <= order.len()`, `1 <= m <= 8`, `per_col >= 1`, and the 32 lanes of
/// the warp call it with the same arguments.
#[allow(
    clippy::too_many_arguments,
    reason = "device helper: the table, the tile, its bounds and the fault sink (rust-quality R8)"
)]
#[inline(always)]
unsafe fn tile_cols(
    order: &[u32],
    j: usize,
    m: usize,
    n_slots: u32,
    per_col: u32,
    fault: FaultSink,
    lane: usize,
) -> Option<[u32; 8]> {
    let mut col = [0u32; 8];
    let mut bad = false;
    macro_rules! entry {
        ($c:literal) => {
            if m > $c {
                // SAFETY: j + $c < j + m <= order.len() by this fn's contract.
                let s = unsafe { *order.get_unchecked(j + $c) };
                bad |= s >= n_slots;
                col[$c] = s / per_col;
            }
        };
    }
    entry!(0);
    entry!(1);
    entry!(2);
    entry!(3);
    entry!(4);
    entry!(5);
    entry!(6);
    entry!(7);
    if bad {
        if lane == 0 {
            fault.raise(FaultSite::ExpertId);
        }
        return None;
    }
    Some(col)
}

/// Lane 0's stores of a tile's columns: value `v[c]` into row `r` of the slot
/// `order[j + c]`'s rows of `y`, for each `c < m` — the entries
/// [`tile_cols`] checked, read again here so no slot stays live through the
/// dot loops.
///
/// # Safety
///
/// `j + m <= order.len()`, `1 <= m <= 8`, every `order[j + c]` below `n_slots`
/// for `c < m` ([`tile_cols`] returned `Some` for the tile), `r < rpe`,
/// `y.len() >= n_slots · rpe`, and no other thread writes row `r` of those
/// slots.
#[allow(
    clippy::too_many_arguments,
    reason = "device helper: a kernel's output, table, tile and bounds (rust-quality R8)"
)]
#[inline(always)]
unsafe fn store_slots(
    y: &mut DisjointSlice<f32>,
    order: &[u32],
    j: usize,
    m: usize,
    rpe: usize,
    r: usize,
    v: [f32; 8],
) {
    macro_rules! store {
        ($c:literal) => {
            if m > $c {
                // SAFETY: j + $c < j + m <= order.len(); the slot is below
                // n_slots and r < rpe, so the index is below n_slots · rpe <=
                // y.len(); this thread is its only writer.
                unsafe {
                    let slot = *order.get_unchecked(j + $c) as usize;
                    *y.get_unchecked_mut(slot * rpe + r) = v[$c];
                }
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

/// A grouped MXFP4 gate·up's work (module doc, [`tile_block`]): warp `w`'s
/// weight row `r` of the tile's expert in both stacks, dotted with the tile's
/// token columns ([`tile_cols`], `slot / slots_per_col`) through
/// [`lane_partials_file_cols`] and reduced with the warp tree; lane 0 stores
/// `h[slot · rows_per_expert + r] = act::apply(act, limit, g, u)` for each
/// entry's slot. Its launch contract: both stacks hold `n_experts ·
/// rows_per_expert` rows of `34 · n_sb` words, `q` and `d8` hold `m_cols`
/// columns with `n_slots <= m_cols · slots_per_col`, `order.len() >= n_slots`,
/// `h.len() >= n_slots · rows_per_expert`; `iters = ceil(n_sb / 4)`.
struct MxGateUpTiles<'a> {
    wg: &'a [u32],
    wu: &'a [u32],
    q: &'a [u32],
    d8: &'a [f32],
    order: &'a [u32],
    rows_per_expert: u32,
    n_slots: u32,
    slots_per_col: u32,
    n_sb: u32,
    iters: u32,
    act: u32,
    limit: f32,
    fault: FaultSink,
    h: DisjointSlice<'a, f32>,
}

impl TileWork for MxGateUpTiles<'_> {
    #[inline(always)]
    unsafe fn run(&mut self, t: TileBlock) {
        let TileBlock { e, j, m, r, lane } = t;
        let rpe = self.rows_per_expert as usize;
        // SAFETY: j + m <= n_slots <= order.len() and 1 <= m <= 8 (the tile
        // is `tile_at`'s), slots_per_col >= 1 by the launch contract, the
        // warp's lanes share the tile.
        let tile = unsafe {
            tile_cols(
                self.order,
                j,
                m,
                self.n_slots,
                self.slots_per_col,
                self.fault,
                lane,
            )
        };
        let Some(col) = tile else {
            return;
        };
        let n_sb = self.n_sb as usize;
        let (q_col, d_col) = (256 * self.iters, 2 * self.n_sb);
        let row_abs = e * rpe + r;
        // SAFETY: row_abs < n_experts · rpe rows of both stacks (e <
        // n_experts, r < rpe); each column is slot / slots_per_col <
        // m_cols (slot < n_slots <= m_cols · slots_per_col), and the host
        // keeps m_cols · 256 · iters below 2^32; 1 <= m <= 8 block-uniform;
        // iters = ceil(n_sb/4) so q and d8 hold each column's words; the
        // warp's 32 lanes are here: this fn's contract and the work's. The
        // gate's sums are reduced before the up's loop starts, so only its
        // eight values stay live through it.
        let g = col_sums(
            unsafe {
                lane_partials_file_cols(
                    self.wg,
                    row_abs,
                    8 * n_sb,
                    self.q,
                    self.d8,
                    col,
                    q_col,
                    d_col,
                    m,
                    lane,
                )
            },
            m,
        );
        let u = col_sums(
            unsafe {
                lane_partials_file_cols(
                    self.wu,
                    row_abs,
                    8 * n_sb,
                    self.q,
                    self.d8,
                    col,
                    q_col,
                    d_col,
                    m,
                    lane,
                )
            },
            m,
        );
        if lane == 0 {
            let (rule, limit) = (self.act, self.limit);
            let mut v = [0.0f32; 8];
            macro_rules! apply {
                ($c:literal) => {
                    if m > $c {
                        v[$c] = act::apply(rule, limit, g[$c], u[$c]);
                    }
                };
            }
            apply!(0);
            apply!(1);
            apply!(2);
            apply!(3);
            apply!(4);
            apply!(5);
            apply!(6);
            apply!(7);
            // SAFETY: the tile's entries are below n_slots ([`tile_cols`]
            // returned Some), r < rpe, h.len() >= n_slots · rpe; a slot sits
            // in one table entry, so lane 0 of this row's warp is the only
            // writer of its row r.
            unsafe { store_slots(&mut self.h, self.order, j, m, rpe, r, v) };
        }
    }
}

/// A grouped MXFP4 down's work (module doc, [`tile_block`]): warp `w`'s weight
/// row `r` of the tile's expert, dotted with the tile's columns — column
/// `slot` of the slots' q8_1 ([`tile_cols`]) — through
/// [`lane_partials_file_cols`] and reduced with the warp tree; lane 0 stores
/// `y[slot · rows_per_expert + r]` for each entry's slot. Its launch contract:
/// `w` holds `n_experts · rows_per_expert` rows of `34 · n_sb` words, `q` and
/// `d8` hold `n_slots` columns, `order.len() >= n_slots`, `y.len() >= n_slots
/// · rows_per_expert`; `iters = ceil(n_sb / 4)`.
struct MxGemvTiles<'a> {
    w: &'a [u32],
    q: &'a [u32],
    d8: &'a [f32],
    order: &'a [u32],
    rows_per_expert: u32,
    n_slots: u32,
    n_sb: u32,
    iters: u32,
    fault: FaultSink,
    y: DisjointSlice<'a, f32>,
}

impl TileWork for MxGemvTiles<'_> {
    #[inline(always)]
    unsafe fn run(&mut self, t: TileBlock) {
        let TileBlock { e, j, m, r, lane } = t;
        let rpe = self.rows_per_expert as usize;
        // SAFETY: j + m <= n_slots <= order.len() and 1 <= m <= 8 (the tile
        // is `tile_at`'s), the per-column divisor is 1, the warp's lanes share
        // the tile.
        let tile = unsafe { tile_cols(self.order, j, m, self.n_slots, 1, self.fault, lane) };
        let Some(col) = tile else {
            return;
        };
        let n_sb = self.n_sb as usize;
        // SAFETY: row e·rpe + r < n_experts·rpe rows of `w` (e < n_experts,
        // r < rpe); each column is its slot < n_slots of the planes, and the
        // host keeps n_slots · 256 · iters below 2^32; 1 <= m <= 8
        // block-uniform; iters = ceil(n_sb/4); the warp's 32 lanes are here:
        // this fn's contract and the work's.
        let f = unsafe {
            lane_partials_file_cols(
                self.w,
                e * rpe + r,
                8 * n_sb,
                self.q,
                self.d8,
                col,
                256 * self.iters,
                2 * self.n_sb,
                m,
                lane,
            )
        };
        let v = col_sums(f, m);
        if lane == 0 {
            // SAFETY: the tile's entries are below n_slots ([`tile_cols`]
            // returned Some), r < rpe, y.len() >= n_slots · rpe; a slot sits
            // in one table entry, so lane 0 of this row's warp is the only
            // writer of its row r.
            unsafe { store_slots(&mut self.y, self.order, j, m, rpe, r, v) };
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
    /// act::apply(act, limit, g, u)` from lane 0. [`HOST`] returns before any
    /// load, writing and raising nothing; any other id at or past
    /// `n_experts` raises [`FaultSite::ExpertId`] and writes NaN into its row
    /// of `h` (warp-uniform, before the first weight load). `m_cols` only
    /// bounds the activation in the launch contract.
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
            // SAFETY: row < n_slots · rows_per_expert <= h.len() by the
            // launch contract; only lane 0 of the warp writes h[row].
            unsafe { *h.get_unchecked_mut(row) = act::apply(act, limit, g, u) };
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

    /// The MXFP4 gate·up of a grouped slot table's slots by tile items
    /// ([`MxGateUpTiles`]): block `g + tile_cap · ρ` (of `tile_cap ·
    /// row_tiles`, [`crate::q4k_sel::tile_cap`] of the table) is tile `g` of
    /// the table `tiles` ([`crate::q4k_sel::grouped_tiles`], [`tile_block`]) on
    /// the eight weight rows `8ρ ..` of both stacks, its entries' slots read
    /// from `order`: entry `c` dots the rows with column `slot / slots_per_col`
    /// of the q8_1 planes and writes `h[slot · rows_per_expert + r] =
    /// act::apply(act, limit, g, u)`, [`mxfp4_gate_up_sel`]'s value for that
    /// slot bit for bit; each file block is read once for up to eight slots. A
    /// block past the table's count returns; a tile word [`crate::q4k_sel::
    /// tile_at`] refuses and an entry that names no slot raise
    /// [`FaultSite::ExpertId`] on `fault`.
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
            order.len() >= n_slots,
            start.len() >= n_experts + 1,
            tiles.len() >= tile_cap + 1,
            tile_cap >= 1,
            h.len() >= n_slots * rows_per_expert,
            rows_per_expert >= 8 * row_tiles
        )
    )]
    pub fn mxfp4_gate_up_tiles(
        wg: &[u32],
        wu: &[u32],
        q: &[u32],
        d8: &[f32],
        order: &[u32],
        start: &[u32],
        tiles: &[u32],
        tile_cap: u32,
        row_tiles: u32,
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
        h: DisjointSlice<f32>,
    ) {
        let _ = m_cols;
        let mut work = MxGateUpTiles {
            wg,
            wu,
            q,
            d8,
            order,
            rows_per_expert,
            n_slots,
            slots_per_col,
            n_sb,
            iters,
            act,
            limit,
            fault,
            h,
        };
        // SAFETY: the launch contract is the work's (34 = SB_WORDS) and
        // `tile_block`'s; the host passes iters = ceil(n_sb/4).
        unsafe {
            tile_block(
                tiles, start, n_experts, n_slots, tile_cap, row_tiles, fault, &mut work,
            );
        }
    }

    /// The MXFP4 down of a grouped slot table's slots by tile items
    /// ([`MxGemvTiles`]): block `g + tile_cap · ρ` is tile `g` of the table
    /// `tiles` on the eight weight rows `8ρ ..`, entry `c` dotting them with
    /// column `slot` of the q8_1 planes (the slots' own activation, as
    /// [`mxfp4_gemv_sel`] reads it) and writing `y[slot · rows_per_expert +
    /// r]`, [`mxfp4_gemv_sel`]'s value for that slot bit for bit; each file
    /// block is read once for up to eight slots. A block past the table's
    /// count returns; a tile word [`crate::q4k_sel::tile_at`] refuses and an
    /// entry that names no slot raise [`FaultSite::ExpertId`] on `fault`.
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
            order.len() >= n_slots,
            start.len() >= n_experts + 1,
            tiles.len() >= tile_cap + 1,
            tile_cap >= 1,
            y.len() >= n_slots * rows_per_expert,
            rows_per_expert >= 8 * row_tiles
        )
    )]
    pub fn mxfp4_gemv_tiles(
        w: &[u32],
        q: &[u32],
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
        let mut work = MxGemvTiles {
            w,
            q,
            d8,
            order,
            rows_per_expert,
            n_slots,
            n_sb,
            iters,
            fault,
            y,
        };
        // SAFETY: the launch contract is the work's (34 = SB_WORDS) and
        // `tile_block`'s; the host passes iters = ceil(n_sb/4).
        unsafe {
            tile_block(
                tiles, start, n_experts, n_slots, tile_cap, row_tiles, fault, &mut work,
            );
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
        let what = "MxExpertKernels::enqueue_down";
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

/// What [`Mxfp4SelKernels::enqueue_gate_up_tiles`] reads: both stacks of the
/// table's `n_experts` experts of `rows_per_expert` rows (the file's blocks, as
/// [`Mxfp4SelKernels::enqueue_gate_up`] reads them), the block's q8_1
/// activation by token (`slots_per_col` slots a token: slot `s` reads column
/// `s / slots_per_col`) and the grouped slot table of the block's `n_slots`
/// slots; `rule` the activation. A slot's row `h` is written where the
/// per-slot gate·up writes it.
pub struct SlotTilesGateUp<'a> {
    pub wg: &'a DeviceTensor<u32>,
    pub wu: &'a DeviceTensor<u32>,
    pub act: &'a Q8Act,
    pub table: TileTable<'a>,
    pub rows_per_expert: usize,
    pub slots_per_col: usize,
    pub rule: Act,
}

/// What [`Mxfp4SelKernels::enqueue_down_tiles`] reads: the down stack of the
/// table's `n_experts` experts of `rows_per_expert` rows, the slots' q8_1
/// activation (column `s` the activation of slot `s`, as
/// [`Mxfp4SelKernels::enqueue_down`] reads it) and the grouped slot table.
pub struct SlotTilesDown<'a> {
    pub w: &'a DeviceTensor<u32>,
    pub act: &'a Q8Act,
    pub table: TileTable<'a>,
    pub rows_per_expert: usize,
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
    /// (module doc for [`HOST`] and ids past the stack, which raise on
    /// `fault` and write NaN). Asynchronous, allocation-free, capturable.
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
        if n_slots == 0
            || spc == 0
            || n_slots > m * spc
            || a.sel.len() < n_slots
            || h.len() < n_slots * rpe
        {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_slots} slots (at least one) at {spc} a column (at least one) over {m} \
                     columns, sel.len() {}, h.len() {} >= n_slots*rows_per_expert = {}",
                    a.sel.len(),
                    h.len(),
                    n_slots * rpe
                ),
            ));
        }
        let grid = launch_u32(what, "grid", (n_slots * rpe).div_ceil(ROWS_PER_BLOCK))?;
        let prep = self
            .module
            .prepare_mxfp4_gate_up_sel(LaunchConfig1D::new(grid, BLOCK, 0))?;
        let (act, limit) = a.rule.code();
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
        if n_slots == 0 || a.act.m() < n_slots || a.sel.len() < n_slots || y.len() < n_slots * rpe {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_slots} slots (at least one): an activation column a slot (act.m() {} >= \
                     n_slots), an id a slot (sel.len() {}), y.len() {} >= n_slots*rows_per_expert = {}",
                    a.act.m(),
                    a.sel.len(),
                    y.len(),
                    n_slots * rpe
                ),
            ));
        }
        let grid = launch_u32(what, "grid", (n_slots * rpe).div_ceil(ROWS_PER_BLOCK))?;
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

    /// Enqueue [`mxfp4_gate_up_tiles`]: the gate·up of the table's slots
    /// (one run per expert of the stacks, `a.wg.rows() / rows_per_expert` of
    /// them) by its tiles ([`crate::kquant::TileTables::enqueue`] of the same
    /// table): slot `s` writes `h[s · rows_per_expert ..][..rows_per_expert]`
    /// as `a.rule` over the rows of its expert dotted with column `s /
    /// slots_per_col` of `a.act` — [`Mxfp4SelKernels::enqueue_gate_up`]'s value
    /// for it. A slot the table does not hold keeps its rows. A tile word that
    /// names no run and an entry that names no slot raise
    /// [`FaultSite::ExpertId`] on `fault`. The tile count is the table's, a
    /// device value: the grid is sized for [`crate::q4k_sel::tile_cap`] of them
    /// and a block past the count returns. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_gate_up_tiles(
        &self,
        stream: &CudaStream,
        a: &SlotTilesGateUp<'_>,
        fault: FaultSink,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "Mxfp4SelKernels::enqueue_gate_up_tiles";
        let n_sb = a.act.n_sb();
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
        let gr = tile_grid(what, SB_WORDS, a.wg, n_sb, a.rows_per_expert, &a.table)?;
        let (n_slots, rpe, spc, m) = (
            a.table.n_slots,
            a.rows_per_expert,
            a.slots_per_col,
            a.act.m(),
        );
        if spc == 0 || n_slots > m * spc || h.len() < n_slots * rpe {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_slots} slots at {spc} a column (at least one) over {m} columns, h.len() \
                     {} >= n_slots*rows_per_expert = {}",
                    h.len(),
                    n_slots * rpe
                ),
            ));
        }
        // The kernel's column offsets are 32-bit.
        launch_u32(what, "activation words", m * 256 * n_sb.div_ceil(4))?;
        let prep = self
            .module
            .prepare_mxfp4_gate_up_tiles(LaunchConfig1D::new(gr.blocks, BLOCK, 0))?;
        let (act, limit) = a.rule.code();
        self.module.mxfp4_gate_up_tiles(
            stream,
            &prep,
            a.wg.buf(),
            a.wu.buf(),
            &a.act.q4,
            &a.act.d8,
            a.table.order,
            a.table.start,
            a.table.tiles,
            gr.tile_cap,
            gr.row_tiles,
            gr.n_experts,
            launch_u32(what, "rows_per_expert", rpe)?,
            gr.n_slots,
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

    /// Enqueue [`mxfp4_gemv_tiles`]: the down of the table's slots by its tiles
    /// ([`crate::kquant::TileTables::enqueue`] of the same table): slot `s`
    /// writes `y[s · rows_per_expert ..][..rows_per_expert]` as the rows of its
    /// expert dotted with column `s` of `a.act` —
    /// [`Mxfp4SelKernels::enqueue_down`]'s value for it. A slot the table does
    /// not hold keeps its rows. Refusals and the grid as
    /// [`Mxfp4SelKernels::enqueue_gate_up_tiles`]. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_down_tiles(
        &self,
        stream: &CudaStream,
        a: &SlotTilesDown<'_>,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "Mxfp4SelKernels::enqueue_down_tiles";
        let n_sb = a.act.n_sb();
        let gr = tile_grid(what, SB_WORDS, a.w, n_sb, a.rows_per_expert, &a.table)?;
        let (n_slots, rpe) = (a.table.n_slots, a.rows_per_expert);
        if n_slots > a.act.m() || y.len() < n_slots * rpe {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_slots} slots over {} activation columns (a column a slot), y.len() {} >= \
                     n_slots*rows_per_expert = {}",
                    a.act.m(),
                    y.len(),
                    n_slots * rpe
                ),
            ));
        }
        // The kernel's column offsets are 32-bit.
        launch_u32(what, "activation words", n_slots * 256 * n_sb.div_ceil(4))?;
        let prep = self
            .module
            .prepare_mxfp4_gemv_tiles(LaunchConfig1D::new(gr.blocks, BLOCK, 0))?;
        self.module.mxfp4_gemv_tiles(
            stream,
            &prep,
            a.w.buf(),
            &a.act.q4,
            &a.act.d8,
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
