//! The family's expert-select entries: the down `_sel` (slot `s` dots the
//! rows of expert `sel[s]` with activation column `s`) and the gate·up with
//! its activation rule as a launch argument (slot `s` against column `s /
//! slots_per_col`), each one launch for all slots, the ids read from a
//! device buffer so a captured graph's replay follows whatever the router
//! last wrote. One warp a row, eight rows a 256-thread block; thread row
//! `n = slot · rows_per_expert + r` reads weight row `sel[slot] ·
//! rows_per_expert + r` and runs Walk A ([`super::walk::row_dot_1col`]) with
//! its format's decoder, reduced by the fixed warp tree.
//!
//! The bodies are generic over the decoder ([`gemv_sel_body`],
//! [`gate_up_act_body`]); this module instantiates them for Q5_K and Q8_0
//! and the gate·up for Q4_K (whose down `_sel` is `q4k_sel`'s
//! `q4k_gemv_sel`), and a caller outside the crate may instantiate them for another format
//! in its own module (its entries then land in its own binary alone).
//!
//! A non-routed FFN (a dense layer, a shared expert) is the one-expert
//! case: a stack of one expert, every id 0, one slot per token.
//!
//! Ids. [`HOST`] is a slot the host tier serves: both entries skip it,
//! write nothing and raise nothing. Any other id at or past the stack's
//! expert count raises [`FaultSite::ExpertId`] before the slot's first
//! weight load (warp-uniform); the down leaves the slot's rows of `y` as
//! they were, the gate·up writes NaN into its rows of `h`, so no stale row of
//! an earlier launch passes for its output.
//!
//! Activations. A q8_1 block the quantizer refused (a NaN or infinite value)
//! holds a NaN scale; the walk multiplies every term by its block scale, so
//! every row that reads the block is NaN.

use super::act::{self, Act};
use super::q5k::Q5k;
use super::q8_0::Q8_0;
use super::walk::{Q4k, SbDecode, row_dot_1col};
use crate::fault::{FaultSink, FaultSite};
use crate::hybrid::HOST;
use crate::tensor::{DeviceTensor, Q8Act};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads per block of the entries here, one warp per output row.
pub const THREADS: u32 = 256;
// The entries' launch attributes spell the block as a literal.
const _: () = assert!(THREADS == 256);

/// Output rows a block computes: one per warp.
pub const ROWS_PER_BLOCK: usize = THREADS as usize / 32;

/// The planes Walk A reads from a q8_1 activation: the q4 plane (`256 ·
/// ceil(n_sb/4)` words a column), the 32-value sums (`8 · n_sb` a column)
/// and the block scales (`2 · n_sb` a column).
#[must_use]
pub fn walk_a_planes(act: &Q8Act) -> (&DeviceBuffer<u32>, &DeviceBuffer<i32>, &DeviceBuffer<f32>) {
    (&act.q4, &act.s8, &act.d8)
}

/// The body of a down `_sel` entry over format `D` (module doc): thread row
/// `n = slot · rows_per_expert + r` stores `y[n]`, the dot of weight row
/// `sel[slot] · rows_per_expert + r` with activation column `slot`.
///
/// # Safety
///
/// The entry's launch contract over its `n_slots` slots: `w.len() >=
/// n_experts · rows_per_expert · D::WORDS · n_sb`, `q.len() >= n_slots · 256
/// · iters`, `s8.len() >= n_slots · 8 · n_sb`, `d8.len() >= n_slots · 2 ·
/// n_sb`, `sel.len() >= n_slots`, `y.len() >= n_slots · rows_per_expert`;
/// `iters = ceil(n_sb / 4)`; `row < n_slots · rows_per_expert` is the
/// calling warp's row (one warp a row), all 32 lanes of the warp here.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn gemv_sel_body<D: SbDecode>(
    w: &[u32],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    sel: &[u32],
    n_experts: u32,
    rows_per_expert: u32,
    n_sb: u32,
    iters: u32,
    fault: FaultSink,
    row: usize,
    y: &mut DisjointSlice<f32>,
) {
    let slot = row / rows_per_expert as usize;
    // SAFETY: slot < n_slots <= sel.len() by this fn's contract; the load is
    // warp-uniform (the warp's 32 lanes share `row`, hence `slot`), so the
    // returns below never diverge a warp.
    let id = unsafe { *sel.get_unchecked(slot) };
    let lane = warp::lane_id() as usize;
    if id >= n_experts {
        if id != HOST && lane == 0 {
            fault.raise(FaultSite::ExpertId);
        }
        return;
    }
    let row_abs = id as usize * rows_per_expert as usize + row % rows_per_expert as usize;
    // SAFETY: row_abs < n_experts · rows_per_expert rows of `w`, column slot
    // < n_slots of the planes, iters = ceil(n_sb/4), and all 32 lanes of the
    // warp are here (both returns above are warp-uniform).
    let f0 = unsafe { row_dot_1col::<D>(w, q, s8, d8, n_sb as usize, iters, row_abs, slot, lane) };
    let s0 = warp::reduce_sum_f32(f0);
    if lane == 0 {
        // SAFETY: row < n_slots · rows_per_expert <= y.len() by this fn's
        // contract; only lane 0 of the warp writes y[row].
        unsafe {
            *y.get_unchecked_mut(row) = s0;
        }
    }
}

/// The body of a gate·up entry over format `D` (module doc): thread row
/// `n = slot · rows_per_expert + r` dots weight row `sel[slot] ·
/// rows_per_expert + r` of `wg` and of `wu` with activation column `slot /
/// slots_per_col`, reduces each with the warp tree and stores `h[n] =
/// act::apply(act, limit, g, u)` from lane 0.
///
/// # Safety
///
/// The entry's launch contract over its `n_slots` slots: both stacks hold
/// `n_experts · rows_per_expert` rows of `D::WORDS · n_sb` words,
/// `slots_per_col >= 1`,
/// `n_slots <= m_cols · slots_per_col`, the planes hold `m_cols` columns,
/// `sel.len() >= n_slots`, `h.len() >= n_slots · rows_per_expert`; `iters =
/// ceil(n_sb / 4)`; `row < n_slots · rows_per_expert` is thread row `row`'s
/// (one warp a row), all 32 lanes of the warp here.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn gate_up_act_body<D: SbDecode>(
    wg: &[u32],
    wu: &[u32],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    sel: &[u32],
    n_experts: u32,
    rows_per_expert: u32,
    slots_per_col: u32,
    n_sb: u32,
    iters: u32,
    act: u32,
    limit: f32,
    fault: FaultSink,
    row: usize,
    h: &mut DisjointSlice<f32>,
) {
    let slot = row / rows_per_expert as usize;
    // SAFETY: slot < n_slots <= sel.len() by this fn's contract; the load is
    // warp-uniform, so the returns below never diverge a warp.
    let id = unsafe { *sel.get_unchecked(slot) };
    let lane = warp::lane_id() as usize;
    if id >= n_experts {
        if id != HOST && lane == 0 {
            fault.raise(FaultSite::ExpertId);
            // SAFETY: row < n_slots · rows_per_expert <= h.len() by this fn's
            // contract; only lane 0 of the warp writes h[row].
            unsafe { *h.get_unchecked_mut(row) = f32::NAN };
        }
        return;
    }
    let row_abs = id as usize * rows_per_expert as usize + row % rows_per_expert as usize;
    let col = slot / slots_per_col as usize;
    // SAFETY: row_abs < n_experts · rows_per_expert rows of both stacks,
    // column col < m_cols because slot < n_slots <= m_cols · slots_per_col,
    // iters = ceil(n_sb/4), and all 32 lanes of the warp are here.
    let (fg, fu) = unsafe {
        (
            row_dot_1col::<D>(wg, q, s8, d8, n_sb as usize, iters, row_abs, col, lane),
            row_dot_1col::<D>(wu, q, s8, d8, n_sb as usize, iters, row_abs, col, lane),
        )
    };
    let g = warp::reduce_sum_f32(fg);
    let u = warp::reduce_sum_f32(fu);
    if lane == 0 {
        // SAFETY: row < n_slots · rows_per_expert <= h.len() by this fn's
        // contract; only lane 0 of the warp writes h[row].
        unsafe { *h.get_unchecked_mut(row) = act::apply(act, limit, g, u) };
    }
}

/// An arena entry's per-slot table ([`GateUpAt`], [`SelDownAt`]): slot `s`'s
/// part `p` (gate 0, up 1, down 2) lies at word `offs[s · ARENA_PARTS + p]`
/// of the arena, the expert's rows of that stack as its file holds them.
pub const ARENA_PARTS: usize = 3;
/// The arena table's gate, up and down part of a slot.
pub const PART_GATE: usize = 0;
pub const PART_UP: usize = 1;
pub const PART_DOWN: usize = 2;
/// An arena table offset marking a slot the arena entries skip: they write
/// nothing and raise nothing for it.
pub const ARENA_SKIP: u64 = u64::MAX;

/// The `words` words of `arena` from word `off`, or `None` when they do not
/// all lie inside it.
#[inline(always)]
pub fn arena_part(arena: &[u32], off: u64, words: usize) -> Option<&[u32]> {
    let end = off.checked_add(words as u64)?;
    if end > arena.len() as u64 {
        return None;
    }
    // SAFETY: off <= end <= arena.len(), checked above.
    Some(unsafe { arena.get_unchecked(off as usize..end as usize) })
}

/// The body of a down arena entry over format `D`: [`gemv_sel_body`] with
/// slot `slot`'s expert read from its down part in `arena` (the slot's
/// [`PART_DOWN`] offset) instead of from a stack by id. Row `r` of the part
/// is the expert's row `r`, so the same bytes give the same bits. A slot
/// marked [`ARENA_SKIP`] is left alone; a part that does not lie inside the
/// arena raises [`FaultSite::ExpertId`] before any weight load and leaves
/// the slot's rows of `y` as they were.
///
/// # Safety
///
/// As [`gemv_sel_body`] with `offs.len() >= n_slots · ARENA_PARTS` in place
/// of the stack and id bounds.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn gemv_at_body<D: SbDecode>(
    arena: &[u32],
    offs: &[u64],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    rows_per_expert: u32,
    n_sb: u32,
    iters: u32,
    fault: FaultSink,
    row: usize,
    y: &mut DisjointSlice<f32>,
) {
    let rpe = rows_per_expert as usize;
    let slot = row / rpe;
    // SAFETY: slot < n_slots, so the index is inside offs by this fn's
    // contract; the load is warp-uniform, so the returns below never
    // diverge a warp.
    let off = unsafe { *offs.get_unchecked(slot * ARENA_PARTS + PART_DOWN) };
    let lane = warp::lane_id() as usize;
    if off == ARENA_SKIP {
        return;
    }
    let Some(w) = arena_part(arena, off, rpe * D::WORDS * n_sb as usize) else {
        if lane == 0 {
            fault.raise(FaultSite::ExpertId);
        }
        return;
    };
    // SAFETY: `w` is the part's rpe rows of D::WORDS · n_sb words and row %
    // rpe one of them; column slot < n_slots of the planes, iters =
    // ceil(n_sb/4), and all 32 lanes of the warp are here.
    let f0 =
        unsafe { row_dot_1col::<D>(w, q, s8, d8, n_sb as usize, iters, row % rpe, slot, lane) };
    let s0 = warp::reduce_sum_f32(f0);
    if lane == 0 {
        // SAFETY: row < n_slots · rows_per_expert <= y.len() by this fn's
        // contract; only lane 0 of the warp writes y[row].
        unsafe {
            *y.get_unchecked_mut(row) = s0;
        }
    }
}

/// The body of a gate·up arena entry over format `D`: [`gate_up_act_body`]
/// with slot `slot`'s expert read from its gate and up parts in `arena`.
/// A slot whose gate offset is [`ARENA_SKIP`] is left alone; a gate or up
/// part that does not lie inside the arena raises [`FaultSite::ExpertId`]
/// and writes NaN into the slot's rows of `h`.
///
/// # Safety
///
/// As [`gate_up_act_body`] with `offs.len() >= n_slots · ARENA_PARTS` in
/// place of the stacks' and ids' bounds.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn gate_up_act_at_body<D: SbDecode>(
    arena: &[u32],
    offs: &[u64],
    q: &[u32],
    s8: &[i32],
    d8: &[f32],
    rows_per_expert: u32,
    slots_per_col: u32,
    n_sb: u32,
    iters: u32,
    act: u32,
    limit: f32,
    fault: FaultSink,
    row: usize,
    h: &mut DisjointSlice<f32>,
) {
    let rpe = rows_per_expert as usize;
    let slot = row / rpe;
    // SAFETY: slot < n_slots, so both indices are inside offs by this fn's
    // contract; the loads are warp-uniform.
    let (og, ou) = unsafe {
        (
            *offs.get_unchecked(slot * ARENA_PARTS + PART_GATE),
            *offs.get_unchecked(slot * ARENA_PARTS + PART_UP),
        )
    };
    let lane = warp::lane_id() as usize;
    if og == ARENA_SKIP {
        return;
    }
    let words = rpe * D::WORDS * n_sb as usize;
    let (Some(wg), Some(wu)) = (arena_part(arena, og, words), arena_part(arena, ou, words)) else {
        if lane == 0 {
            fault.raise(FaultSite::ExpertId);
            // SAFETY: row < n_slots · rows_per_expert <= h.len() by this
            // fn's contract; only lane 0 of the warp writes h[row].
            unsafe { *h.get_unchecked_mut(row) = f32::NAN };
        }
        return;
    };
    let (r, col) = (row % rpe, slot / slots_per_col as usize);
    // SAFETY: both parts hold rpe rows of D::WORDS · n_sb words and r is one
    // of them; column col < m_cols because slot < n_slots <= m_cols ·
    // slots_per_col; iters = ceil(n_sb/4); all 32 lanes of the warp are here.
    let (fg, fu) = unsafe {
        (
            row_dot_1col::<D>(wg, q, s8, d8, n_sb as usize, iters, r, col, lane),
            row_dot_1col::<D>(wu, q, s8, d8, n_sb as usize, iters, r, col, lane),
        )
    };
    let g = warp::reduce_sum_f32(fg);
    let u = warp::reduce_sum_f32(fu);
    if lane == 0 {
        // SAFETY: row < n_slots · rows_per_expert <= h.len() by this fn's
        // contract; only lane 0 of the warp writes h[row].
        unsafe { *h.get_unchecked_mut(row) = act::apply(act, limit, g, u) };
    }
}

#[cuda_module]
mod kquant_kernels {
    use super::*;

    /// The Q5_K down `_sel` ([`gemv_sel_body`] over [`Q5k`]). Four blocks an
    /// SM are pinned: a register count past 64 spills (the spill ratchet's
    /// red) instead of dropping a resident block.
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
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn q5k_gemv_sel(
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
        // SAFETY: the launch contract is the body's (44 = Q5k::WORDS); the
        // host passes iters = ceil(n_sb/4); the return above is warp-uniform
        // (a warp's lanes share `row`) and keeps row in range.
        unsafe {
            gemv_sel_body::<Q5k>(
                w,
                q,
                s8,
                d8,
                sel,
                n_experts,
                rows_per_expert,
                n_sb,
                iters,
                fault,
                row,
                &mut y,
            );
        }
    }

    /// The Q5_K gate·up with its rule as an argument ([`gate_up_act_body`]
    /// over [`Q5k`]), four blocks an SM pinned as [`q5k_gemv_sel`].
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
            slots_per_col >= 1,
            n_slots <= m_cols * slots_per_col,
            q.len() >= m_cols * 256 * iters,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb,
            sel.len() >= n_slots,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn kq_gate_up_act_q5k(
        wg: &[u32],
        wu: &[u32],
        q: &[u32],
        s8: &[i32],
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
        // m_cols only bounds the activation in the launch contract.
        let _ = m_cols;
        let t = thread::index_1d().get() % THREADS as usize;
        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        // SAFETY: the launch contract is the body's (44 = Q5k::WORDS); the
        // host passes iters = ceil(n_sb/4) and one of Act::code's codes; the
        // return above is warp-uniform and keeps row in range.
        unsafe {
            gate_up_act_body::<Q5k>(
                wg,
                wu,
                q,
                s8,
                d8,
                sel,
                n_experts,
                rows_per_expert,
                slots_per_col,
                n_sb,
                iters,
                act,
                limit,
                fault,
                row,
                &mut h,
            );
        }
    }

    /// The Q4_K gate·up with its rule as an argument ([`gate_up_act_body`]
    /// over [`Q4k`]), four blocks an SM pinned as [`q5k_gemv_sel`].
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
            slots_per_col >= 1,
            n_slots <= m_cols * slots_per_col,
            q.len() >= m_cols * 256 * iters,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb,
            sel.len() >= n_slots,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn kq_gate_up_act_q4k(
        wg: &[u32],
        wu: &[u32],
        q: &[u32],
        s8: &[i32],
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
        // m_cols only bounds the activation in the launch contract.
        let _ = m_cols;
        let t = thread::index_1d().get() % THREADS as usize;
        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        // SAFETY: the launch contract is the body's (36 = Q4k::WORDS); the
        // host passes iters = ceil(n_sb/4) and one of Act::code's codes; the
        // return above is warp-uniform and keeps row in range.
        unsafe {
            gate_up_act_body::<Q4k>(
                wg,
                wu,
                q,
                s8,
                d8,
                sel,
                n_experts,
                rows_per_expert,
                slots_per_col,
                n_sb,
                iters,
                act,
                limit,
                fault,
                row,
                &mut h,
            );
        }
    }

    /// The Q8_0 down `_sel` ([`gemv_sel_body`] over [`Q8_0`]), four blocks
    /// an SM pinned as [`q5k_gemv_sel`].
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
            w.len() >= n_experts * rows_per_expert * 68 * n_sb,
            q.len() >= n_slots * 256 * iters,
            s8.len() >= n_slots * 8 * n_sb,
            d8.len() >= n_slots * 2 * n_sb,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn q8_0_gemv_sel(
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
        // SAFETY: the launch contract is the body's (68 = Q8_0::WORDS); the
        // host passes iters = ceil(n_sb/4); the return above is warp-uniform
        // (a warp's lanes share `row`) and keeps row in range.
        unsafe {
            gemv_sel_body::<Q8_0>(
                w,
                q,
                s8,
                d8,
                sel,
                n_experts,
                rows_per_expert,
                n_sb,
                iters,
                fault,
                row,
                &mut y,
            );
        }
    }

    /// The Q8_0 gate·up with its rule as an argument ([`gate_up_act_body`]
    /// over [`Q8_0`]), four blocks an SM pinned as [`q5k_gemv_sel`].
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
            wg.len() >= n_experts * rows_per_expert * 68 * n_sb,
            wu.len() >= n_experts * rows_per_expert * 68 * n_sb,
            slots_per_col >= 1,
            n_slots <= m_cols * slots_per_col,
            q.len() >= m_cols * 256 * iters,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb,
            sel.len() >= n_slots,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn kq_gate_up_act_q8_0(
        wg: &[u32],
        wu: &[u32],
        q: &[u32],
        s8: &[i32],
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
        // m_cols only bounds the activation in the launch contract.
        let _ = m_cols;
        let t = thread::index_1d().get() % THREADS as usize;
        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        // SAFETY: the launch contract is the body's (68 = Q8_0::WORDS); the
        // host passes iters = ceil(n_sb/4) and one of Act::code's codes; the
        // return above is warp-uniform and keeps row in range.
        unsafe {
            gate_up_act_body::<Q8_0>(
                wg,
                wu,
                q,
                s8,
                d8,
                sel,
                n_experts,
                rows_per_expert,
                slots_per_col,
                n_sb,
                iters,
                act,
                limit,
                fault,
                row,
                &mut h,
            );
        }
    }

    /// The Q5_K down arena entry ([`gemv_at_body`] over [`Q5k`]), four
    /// blocks an SM pinned as [`q5k_gemv_sel`].
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
            offs.len() >= n_slots * 3,
            q.len() >= n_slots * 256 * iters,
            s8.len() >= n_slots * 8 * n_sb,
            d8.len() >= n_slots * 2 * n_sb,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn q5k_gemv_sel_at(
        arena: &[u32],
        offs: &[u64],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
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
        // SAFETY: the launch contract is the body's (3 = ARENA_PARTS); the
        // host passes iters = ceil(n_sb/4); the return above is warp-uniform
        // and keeps row in range.
        unsafe {
            gemv_at_body::<Q5k>(
                arena,
                offs,
                q,
                s8,
                d8,
                rows_per_expert,
                n_sb,
                iters,
                fault,
                row,
                &mut y,
            );
        }
    }

    /// The Q5_K gate·up arena entry ([`gate_up_act_at_body`] over [`Q5k`]),
    /// four blocks an SM pinned as [`q5k_gemv_sel`].
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
            offs.len() >= n_slots * 3,
            slots_per_col >= 1,
            n_slots <= m_cols * slots_per_col,
            q.len() >= m_cols * 256 * iters,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn kq_gate_up_act_q5k_at(
        arena: &[u32],
        offs: &[u64],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
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
        // m_cols only bounds the activation in the launch contract.
        let _ = m_cols;
        let t = thread::index_1d().get() % THREADS as usize;
        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        // SAFETY: the launch contract is the body's (3 = ARENA_PARTS); the
        // host passes iters = ceil(n_sb/4) and one of Act::code's codes; the
        // return above is warp-uniform and keeps row in range.
        unsafe {
            gate_up_act_at_body::<Q5k>(
                arena,
                offs,
                q,
                s8,
                d8,
                rows_per_expert,
                slots_per_col,
                n_sb,
                iters,
                act,
                limit,
                fault,
                row,
                &mut h,
            );
        }
    }

    /// The Q4_K gate·up arena entry ([`gate_up_act_at_body`] over [`Q4k`]),
    /// four blocks an SM pinned as [`q5k_gemv_sel`].
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
            offs.len() >= n_slots * 3,
            slots_per_col >= 1,
            n_slots <= m_cols * slots_per_col,
            q.len() >= m_cols * 256 * iters,
            s8.len() >= m_cols * 8 * n_sb,
            d8.len() >= m_cols * 2 * n_sb,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn kq_gate_up_act_q4k_at(
        arena: &[u32],
        offs: &[u64],
        q: &[u32],
        s8: &[i32],
        d8: &[f32],
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
        // m_cols only bounds the activation in the launch contract.
        let _ = m_cols;
        let t = thread::index_1d().get() % THREADS as usize;
        let row = (thread::index_1d().get() / THREADS as usize) * ROWS_PER_BLOCK + t / 32;
        if row >= n_slots as usize * rows_per_expert as usize {
            return;
        }
        // SAFETY: the launch contract is the body's (3 = ARENA_PARTS); the
        // host passes iters = ceil(n_sb/4) and one of Act::code's codes; the
        // return above is warp-uniform and keeps row in range.
        unsafe {
            gate_up_act_at_body::<Q4k>(
                arena,
                offs,
                q,
                s8,
                d8,
                rows_per_expert,
                slots_per_col,
                n_sb,
                iters,
                act,
                limit,
                fault,
                row,
                &mut h,
            );
        }
    }
}

// The arena entries' contracts spell the table's parts a slot as 3.
const _: () = assert!(ARENA_PARTS == 3);

// The entries spell Q4_K's super-block words as 36, Q5_K's as 44 and Q8_0's as 68.
const _: () = assert!(Q4k::WORDS == 36 && Q5k::WORDS == 44 && Q8_0::WORDS == 68);

/// A down `_sel` launch ([`KquantKernels::enqueue_gemv_q5k_sel`],
/// [`KquantKernels::enqueue_gemv_q8_0_sel`]).
pub struct SelDown<'a> {
    /// The resident stack: `n_experts · rows_per_expert` rows of `W · n_sb`
    /// words, `W` the entry's format's super-block words (Q5_K 44, Q8_0 68;
    /// `w.rows()` a positive multiple of `rows_per_expert`).
    pub w: &'a DeviceTensor<u32>,
    /// At least `n_slots` q8_1 columns: column `slot` is read for `slot <
    /// n_slots` and the rest are not touched, so a cap-sized activation serves
    /// a smaller batch.
    pub act: &'a Q8Act,
    /// At least `n_slots` ids, read on the device at each launch.
    pub sel: &'a DeviceBuffer<u32>,
    pub n_slots: usize,
    pub rows_per_expert: usize,
}

/// A gate·up launch ([`KquantKernels::enqueue_gate_up_q4k`],
/// [`KquantKernels::enqueue_gate_up_q5k`],
/// [`KquantKernels::enqueue_gate_up_q8_0`]).
pub struct GateUpAct<'a> {
    /// The gate stack, as [`SelDown::w`] (Q4_K 36 words a super-block).
    pub wg: &'a DeviceTensor<u32>,
    /// The up stack, the same shape as `wg`.
    pub wu: &'a DeviceTensor<u32>,
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

/// A down arena launch ([`KquantKernels::enqueue_gemv_q5k_sel_at`],
/// `Q6kSelKernels::enqueue_gemv_q6k_sel_at`): [`SelDown`] with each slot's
/// expert read from its down part in `arena` at the slot's [`PART_DOWN`]
/// offset of `offs` instead of from a stack by id.
pub struct SelDownAt<'a> {
    /// The arena as the card addresses it, in words.
    pub arena: &'a DeviceBuffer<u32>,
    /// [`ARENA_PARTS`] word offsets a slot, at least `n_slots` slots' worth,
    /// read on the device at each launch; [`ARENA_SKIP`] skips the slot.
    pub offs: &'a DeviceBuffer<u64>,
    /// As [`SelDown::act`].
    pub act: &'a Q8Act,
    pub n_slots: usize,
    pub rows_per_expert: usize,
}

/// A gate·up arena launch ([`KquantKernels::enqueue_gate_up_q4k_at`],
/// [`KquantKernels::enqueue_gate_up_q5k_at`]): [`GateUpAct`] with each
/// slot's expert read from its gate and up parts in `arena`.
pub struct GateUpAt<'a> {
    /// The arena as the card addresses it, in words.
    pub arena: &'a DeviceBuffer<u32>,
    /// As [`SelDownAt::offs`]; a slot's [`PART_GATE`] offset marks the skip.
    pub offs: &'a DeviceBuffer<u64>,
    /// As [`GateUpAct::act`].
    pub act: &'a Q8Act,
    pub n_slots: usize,
    pub rows_per_expert: usize,
    pub slots_per_col: usize,
    pub rule: Act,
}

/// The loaded family module and its launchers. Owns no stream: every enqueue
/// takes the engine stream, so launches order with the step and are
/// capturable.
pub struct KquantKernels {
    module: kquant_kernels::LoadedModule,
    /// The fault word of the `Gpu` that owns the context: the launches'
    /// sinks point into it, and the module keeps it alive.
    _fault: Arc<DeviceBuffer<u32>>,
}

/// A format's name and super-block words, as its launchers check a stack.
#[derive(Clone, Copy)]
struct Fmt {
    name: &'static str,
    words: usize,
}

const FMT_Q4K: Fmt = Fmt {
    name: "Q4_K",
    words: Q4k::WORDS,
};
const FMT_Q5K: Fmt = Fmt {
    name: "Q5_K",
    words: Q5k::WORDS,
};
const FMT_Q8_0: Fmt = Fmt {
    name: "Q8_0",
    words: Q8_0::WORDS,
};

/// The experts of a stack of `fmt.words · n_sb`-word rows, or the shape
/// error naming `what`.
fn stack_experts(
    what: &'static str,
    fmt: Fmt,
    w: &DeviceTensor<u32>,
    n_sb: usize,
    rows_per_expert: usize,
) -> Result<usize, GpuError> {
    if w.cols() != fmt.words * n_sb {
        return Err(GpuError::shape(
            what,
            format!(
                "{} rows are {}*{n_sb} = {} words, got {}",
                fmt.name,
                fmt.words,
                fmt.words * n_sb,
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

/// A checked down `_sel` launch's grid and scalars, the entry's argument
/// order.
struct DownDims {
    grid: u32,
    n_experts: u32,
    rows_per_expert: u32,
    n_slots: u32,
    n_sb: u32,
    iters: u32,
}

/// [`SelDown`] checked against `fmt` and an output of `y_len` values.
fn down_dims(
    what: &'static str,
    fmt: Fmt,
    a: &SelDown<'_>,
    y_len: usize,
) -> Result<DownDims, GpuError> {
    let n_sb = a.act.n_sb();
    let n_experts = stack_experts(what, fmt, a.w, n_sb, a.rows_per_expert)?;
    let (n_slots, rpe) = (a.n_slots, a.rows_per_expert);
    if n_slots == 0 || a.act.m() < n_slots || a.sel.len() < n_slots || y_len < n_slots * rpe {
        return Err(GpuError::shape(
            what,
            format!(
                "{n_slots} slots (at least one): an activation column a slot (act.m() {} >= \
                 n_slots), an id a slot (sel.len() {}), y.len() {y_len} >= n_slots*rows_per_expert = {}",
                a.act.m(),
                a.sel.len(),
                n_slots * rpe
            ),
        ));
    }
    Ok(DownDims {
        grid: launch_u32(what, "grid", (n_slots * rpe).div_ceil(ROWS_PER_BLOCK))?,
        n_experts: launch_u32(what, "n_experts", n_experts)?,
        rows_per_expert: launch_u32(what, "rows_per_expert", rpe)?,
        n_slots: launch_u32(what, "n_slots", n_slots)?,
        n_sb: launch_u32(what, "n_sb", n_sb)?,
        iters: launch_u32(what, "iters", n_sb.div_ceil(4))?,
    })
}

/// A checked gate·up launch's grid and scalars, the entry's argument order.
struct GateUpDims {
    grid: u32,
    n_experts: u32,
    rows_per_expert: u32,
    n_slots: u32,
    m_cols: u32,
    slots_per_col: u32,
    n_sb: u32,
    iters: u32,
    act: u32,
    limit: f32,
}

/// [`GateUpAct`] checked against `fmt` and an output of `h_len` values.
fn gate_up_dims(
    what: &'static str,
    fmt: Fmt,
    a: &GateUpAct<'_>,
    h_len: usize,
) -> Result<GateUpDims, GpuError> {
    let n_sb = a.act.n_sb();
    let n_experts = stack_experts(what, fmt, a.wg, n_sb, a.rows_per_expert)?;
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
        || h_len < n_slots * rpe
    {
        return Err(GpuError::shape(
            what,
            format!(
                "{n_slots} slots (at least one) at {spc} a column (at least one) over {m} \
                 columns, sel.len() {}, h.len() {h_len} >= n_slots*rows_per_expert = {}",
                a.sel.len(),
                n_slots * rpe
            ),
        ));
    }
    let (act, limit) = a.rule.code();
    Ok(GateUpDims {
        grid: launch_u32(what, "grid", (n_slots * rpe).div_ceil(ROWS_PER_BLOCK))?,
        n_experts: launch_u32(what, "n_experts", n_experts)?,
        rows_per_expert: launch_u32(what, "rows_per_expert", rpe)?,
        n_slots: launch_u32(what, "n_slots", n_slots)?,
        m_cols: launch_u32(what, "m_cols", m)?,
        slots_per_col: launch_u32(what, "slots_per_col", spc)?,
        n_sb: launch_u32(what, "n_sb", n_sb)?,
        iters: launch_u32(what, "iters", n_sb.div_ceil(4))?,
        act,
        limit,
    })
}

/// A checked down arena launch's grid and scalars, the entry's argument
/// order.
pub(crate) struct DownAtDims {
    pub(crate) grid: u32,
    pub(crate) rows_per_expert: u32,
    pub(crate) n_slots: u32,
    pub(crate) n_sb: u32,
    iters: u32,
}

/// A checked gate·up arena launch's grid and scalars, the entry's argument
/// order.
struct GateUpAtDims {
    grid: u32,
    rows_per_expert: u32,
    n_slots: u32,
    m_cols: u32,
    slots_per_col: u32,
    n_sb: u32,
    iters: u32,
    act: u32,
    limit: f32,
}

/// The slot checks the arena launches share: at least one slot, an
/// offset table of `n_slots` slots, a nonempty arena and positive rows a
/// part.
pub(crate) fn arena_slots(
    what: &'static str,
    arena: &DeviceBuffer<u32>,
    offs: &DeviceBuffer<u64>,
    n_slots: usize,
    rows_per_expert: usize,
) -> Result<(), GpuError> {
    if n_slots == 0
        || offs.len() < n_slots * ARENA_PARTS
        || arena.is_empty()
        || rows_per_expert == 0
    {
        return Err(GpuError::shape(
            what,
            format!(
                "{n_slots} slots (at least one): offs.len() {} >= n_slots*{ARENA_PARTS}, an \
                 arena of {} words (at least one), rows_per_expert {rows_per_expert} (at least \
                 one)",
                offs.len(),
                arena.len()
            ),
        ));
    }
    Ok(())
}

/// [`SelDownAt`] checked as [`down_dims`] checks a stack's launch, for an
/// output of `y_len` values.
pub(crate) fn down_at_dims(
    what: &'static str,
    a: &SelDownAt<'_>,
    y_len: usize,
) -> Result<DownAtDims, GpuError> {
    let (n_slots, rpe, n_sb) = (a.n_slots, a.rows_per_expert, a.act.n_sb());
    arena_slots(what, a.arena, a.offs, n_slots, rpe)?;
    if a.act.m() < n_slots || y_len < n_slots * rpe {
        return Err(GpuError::shape(
            what,
            format!(
                "{n_slots} slots: an activation column a slot (act.m() {}), y.len() {y_len} >= \
                 n_slots*rows_per_expert = {}",
                a.act.m(),
                n_slots * rpe
            ),
        ));
    }
    Ok(DownAtDims {
        grid: launch_u32(what, "grid", (n_slots * rpe).div_ceil(ROWS_PER_BLOCK))?,
        rows_per_expert: launch_u32(what, "rows_per_expert", rpe)?,
        n_slots: launch_u32(what, "n_slots", n_slots)?,
        n_sb: launch_u32(what, "n_sb", n_sb)?,
        iters: launch_u32(what, "iters", n_sb.div_ceil(4))?,
    })
}

/// [`GateUpAt`] checked as [`gate_up_dims`] checks a stack's launch, for an
/// output of `h_len` values.
fn gate_up_at_dims(
    what: &'static str,
    a: &GateUpAt<'_>,
    h_len: usize,
) -> Result<GateUpAtDims, GpuError> {
    let (n_slots, rpe, spc, m, n_sb) = (
        a.n_slots,
        a.rows_per_expert,
        a.slots_per_col,
        a.act.m(),
        a.act.n_sb(),
    );
    arena_slots(what, a.arena, a.offs, n_slots, rpe)?;
    if spc == 0 || n_slots > m * spc || h_len < n_slots * rpe {
        return Err(GpuError::shape(
            what,
            format!(
                "{n_slots} slots at {spc} a column (at least one) over {m} columns, h.len() \
                 {h_len} >= n_slots*rows_per_expert = {}",
                n_slots * rpe
            ),
        ));
    }
    let (act, limit) = a.rule.code();
    Ok(GateUpAtDims {
        grid: launch_u32(what, "grid", (n_slots * rpe).div_ceil(ROWS_PER_BLOCK))?,
        rows_per_expert: launch_u32(what, "rows_per_expert", rpe)?,
        n_slots: launch_u32(what, "n_slots", n_slots)?,
        m_cols: launch_u32(what, "m_cols", m)?,
        slots_per_col: launch_u32(what, "slots_per_col", spc)?,
        n_sb: launch_u32(what, "n_sb", n_sb)?,
        iters: launch_u32(what, "iters", n_sb.div_ceil(4))?,
        act,
        limit,
    })
}

impl KquantKernels {
    /// Load this module's device bundle into `ctx`, whose launches raise into
    /// `word`, the fault word of the `Gpu` that owns `ctx`
    /// ([`crate::Gpu::fault_word`]); a word of another context is refused.
    /// Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        word: &Arc<DeviceBuffer<u32>>,
    ) -> Result<KquantKernels, GpuError> {
        let fault = crate::module_fault_word(ctx, word, "KquantKernels::load")?;
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(kquant_kernels, ctx)? };
        Ok(KquantKernels {
            module,
            _fault: fault,
        })
    }

    /// Enqueue the Q5_K down `_sel`: slot `s` writes `y[s · rows_per_expert
    /// ..][..rows_per_expert]` as the rows of expert `sel[s]` dotted with
    /// column `s` of `a.act` (module doc for [`HOST`] and ids past the
    /// stack, which raise on `fault`). Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_gemv_q5k_sel(
        &self,
        stream: &CudaStream,
        a: &SelDown<'_>,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let d = down_dims("enqueue_gemv_q5k_sel", FMT_Q5K, a, y.len())?;
        let prep = self
            .module
            .prepare_q5k_gemv_sel(LaunchConfig1D::new(d.grid, THREADS, 0))?;
        let (q, s8, d8) = walk_a_planes(a.act);
        self.module.q5k_gemv_sel(
            stream,
            &prep,
            a.w.buf(),
            q,
            s8,
            d8,
            a.sel,
            d.n_experts,
            d.rows_per_expert,
            d.n_slots,
            d.n_sb,
            d.iters,
            fault,
            y,
        )?;
        Ok(())
    }

    /// Enqueue the Q5_K gate·up: slot `s` writes `h[s · rows_per_expert
    /// ..][..rows_per_expert]` as `a.rule` over the rows of expert `sel[s]`
    /// of both stacks dotted with column `s / slots_per_col` of `a.act`
    /// (module doc for [`HOST`] and ids past the stack, which raise on
    /// `fault` and write NaN). Asynchronous, allocation-free, capturable.
    pub fn enqueue_gate_up_q5k(
        &self,
        stream: &CudaStream,
        a: &GateUpAct<'_>,
        fault: FaultSink,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let d = gate_up_dims("enqueue_gate_up_q5k", FMT_Q5K, a, h.len())?;
        let prep = self
            .module
            .prepare_kq_gate_up_act_q5k(LaunchConfig1D::new(d.grid, THREADS, 0))?;
        let (q, s8, d8) = walk_a_planes(a.act);
        self.module.kq_gate_up_act_q5k(
            stream,
            &prep,
            a.wg.buf(),
            a.wu.buf(),
            q,
            s8,
            d8,
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
        )?;
        Ok(())
    }

    /// Enqueue the Q4_K gate·up: as [`Self::enqueue_gate_up_q5k`], `36 ·
    /// n_sb` words a row, the Walk A arithmetic of `q4k_gemv_sel`.
    pub fn enqueue_gate_up_q4k(
        &self,
        stream: &CudaStream,
        a: &GateUpAct<'_>,
        fault: FaultSink,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let d = gate_up_dims("enqueue_gate_up_q4k", FMT_Q4K, a, h.len())?;
        let prep = self
            .module
            .prepare_kq_gate_up_act_q4k(LaunchConfig1D::new(d.grid, THREADS, 0))?;
        let (q, s8, d8) = walk_a_planes(a.act);
        self.module.kq_gate_up_act_q4k(
            stream,
            &prep,
            a.wg.buf(),
            a.wu.buf(),
            q,
            s8,
            d8,
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
        )?;
        Ok(())
    }

    /// Enqueue the Q8_0 down `_sel` over rows in the file's block layout
    /// ([`super::q8_0`]): as [`Self::enqueue_gemv_q5k_sel`], `68 · n_sb`
    /// words a row. A non-routed down is a one-expert stack with every id 0
    /// (module doc).
    pub fn enqueue_gemv_q8_0_sel(
        &self,
        stream: &CudaStream,
        a: &SelDown<'_>,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let d = down_dims("enqueue_gemv_q8_0_sel", FMT_Q8_0, a, y.len())?;
        let prep = self
            .module
            .prepare_q8_0_gemv_sel(LaunchConfig1D::new(d.grid, THREADS, 0))?;
        let (q, s8, d8) = walk_a_planes(a.act);
        self.module.q8_0_gemv_sel(
            stream,
            &prep,
            a.w.buf(),
            q,
            s8,
            d8,
            a.sel,
            d.n_experts,
            d.rows_per_expert,
            d.n_slots,
            d.n_sb,
            d.iters,
            fault,
            y,
        )?;
        Ok(())
    }

    /// Enqueue the Q8_0 gate·up over rows in the file's block layout
    /// ([`super::q8_0`]): as [`Self::enqueue_gate_up_q5k`], `68 · n_sb`
    /// words a row. A non-routed gate·up is a one-expert stack with every
    /// id 0 (module doc).
    pub fn enqueue_gate_up_q8_0(
        &self,
        stream: &CudaStream,
        a: &GateUpAct<'_>,
        fault: FaultSink,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let d = gate_up_dims("enqueue_gate_up_q8_0", FMT_Q8_0, a, h.len())?;
        let prep = self
            .module
            .prepare_kq_gate_up_act_q8_0(LaunchConfig1D::new(d.grid, THREADS, 0))?;
        let (q, s8, d8) = walk_a_planes(a.act);
        self.module.kq_gate_up_act_q8_0(
            stream,
            &prep,
            a.wg.buf(),
            a.wu.buf(),
            q,
            s8,
            d8,
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
        )?;
        Ok(())
    }

    /// Enqueue the Q5_K down arena entry: as [`Self::enqueue_gemv_q5k_sel`],
    /// slot `s`'s expert the down part at its offset in `a.arena` (module
    /// consts for the table, [`ARENA_SKIP`] and a part past the arena,
    /// which raises on `fault`). Asynchronous, allocation-free, capturable.
    pub fn enqueue_gemv_q5k_sel_at(
        &self,
        stream: &CudaStream,
        a: &SelDownAt<'_>,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let d = down_at_dims("enqueue_gemv_q5k_sel_at", a, y.len())?;
        let prep = self
            .module
            .prepare_q5k_gemv_sel_at(LaunchConfig1D::new(d.grid, THREADS, 0))?;
        let (q, s8, d8) = walk_a_planes(a.act);
        self.module.q5k_gemv_sel_at(
            stream,
            &prep,
            a.arena,
            a.offs,
            q,
            s8,
            d8,
            d.rows_per_expert,
            d.n_slots,
            d.n_sb,
            d.iters,
            fault,
            y,
        )?;
        Ok(())
    }

    /// Enqueue the Q5_K gate·up arena entry: as [`Self::enqueue_gate_up_q5k`],
    /// slot `s`'s expert the gate and up parts at its offsets in `a.arena`
    /// (a part past the arena raises on `fault` and writes NaN). Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_gate_up_q5k_at(
        &self,
        stream: &CudaStream,
        a: &GateUpAt<'_>,
        fault: FaultSink,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let d = gate_up_at_dims("enqueue_gate_up_q5k_at", a, h.len())?;
        let prep = self
            .module
            .prepare_kq_gate_up_act_q5k_at(LaunchConfig1D::new(d.grid, THREADS, 0))?;
        let (q, s8, d8) = walk_a_planes(a.act);
        self.module.kq_gate_up_act_q5k_at(
            stream,
            &prep,
            a.arena,
            a.offs,
            q,
            s8,
            d8,
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
        )?;
        Ok(())
    }

    /// Enqueue the Q4_K gate·up arena entry: as [`Self::enqueue_gate_up_q5k_at`]
    /// over Q4_K rows, the arithmetic of [`Self::enqueue_gate_up_q4k`].
    pub fn enqueue_gate_up_q4k_at(
        &self,
        stream: &CudaStream,
        a: &GateUpAt<'_>,
        fault: FaultSink,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let d = gate_up_at_dims("enqueue_gate_up_q4k_at", a, h.len())?;
        let prep = self
            .module
            .prepare_kq_gate_up_act_q4k_at(LaunchConfig1D::new(d.grid, THREADS, 0))?;
        let (q, s8, d8) = walk_a_planes(a.act);
        self.module.kq_gate_up_act_q4k_at(
            stream,
            &prep,
            a.arena,
            a.offs,
            q,
            s8,
            d8,
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
        )?;
        Ok(())
    }
}
