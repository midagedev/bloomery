//! MXFP8 routed experts over expert slots selected on the device: a gate·up
//! and a down `_sel` over stacks in the two-plane card layout
//! (`CardFormat::Mxfp8Planes`, [`crate::q8f32::e4m3_planes`]: `qs` a row's
//! value bytes as `k / 4` words, `e` its E8M0 scale bytes four to a word,
//! `ceil(k / 128)` words a row), shaped like every other `_sel` pair
//! ([`crate::kquant::sel`]).
//!
//! - `mxfp8_gate_up_sel`: slot `s` dots the rows of expert `sel[s]` of the
//!   gate and up stacks with column `s / slots_per_col` of a [`Q8Act`] and
//!   writes `act::apply_checked` of the two.
//! - `mxfp8_gemv_sel32`: slot `s` dots the rows of expert `sel[s]` of the
//!   down stack with column `s` of a [`Q8Blocks32`].
//! - `mxfp8_gate_up_sel_cols` and `mxfp8_gemv_sel32_cols`: the same two for
//!   a run of more than one token (`n_slots > slots_per_col`), where the
//!   tokens' slots can name one expert (a verify's rows of one sequence). The
//!   warp of the first slot that names an expert reads its row for every slot
//!   that names it, one slot after the other, and the other slots' warps load
//!   no weight. Every slot's row is the one-token entry's bit for bit.
//!
//! Arithmetic ([`lane_partial`]). A value byte is its F8_E4M3 (OCP E4M3FN)
//! value times `2^9`, an integer of at most 18 bits ([`e4m3_int`]); a 32-value
//! block's `isum` is the exact i32 dot of those integers with the column's
//! int8 codes (`|isum| < 2^30`, [`block_isum`]), and the block enters the
//! lane's partial as `f32(isum) · 2^(E − 136)` ([`block_scale`]: the block's
//! E8M0 byte `E` as `2^(E − 127)` times the integers' `2^-9`) times the
//! column's block scale, `acc + v · d`, each operation rounded once. Lane `ℓ`
//! takes blocks `ℓ, ℓ + 32, ..` of the row and the warp tree reduces the
//! partials. Both activations hold a block's 32 codes as eight words at a
//! stride of 32 (block `b`'s word `i` at `256 · (b / 32) + b % 32 + 32 i` of
//! its column), so a warp's load of a word is 32 consecutive words; a
//! `Q8Act` holds one scale a 128 values (block `b`'s is its `b / 4`th), a
//! `Q8Blocks32` one a block.
//!
//! The planes hold no NaN byte (0x7F, 0xFF) and no E8M0 255: the upload's
//! repack refuses them by name (`gguf::quant::check_mxfp8`). Ids: [`HOST`]
//! returns before any load and writes nothing; any other id at or past the
//! stack's experts raises [`FaultSite::ExpertId`] on the launch's sink, and
//! the gate·up writes NaN into its row of `h` (warp-uniform, before the first
//! weight load); the down leaves its rows of `y` as they were. A gate or up
//! sum that is not finite raises [`FaultSite::F32Product`] and writes NaN into
//! its row of `h`: the SwiGLU clamp would map a NaN to a finite value.

use crate::hybrid::HOST;
use crate::kquant::act::{self, Act};
use crate::q5::Q8Blocks32;
use crate::{DeviceTensor, FaultSink, FaultSite, GpuError, Q8Act, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::vector::{U32x4, as_vectors};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads per block of the entries: eight warps, a row each.
const BLOCK: u32 = 256;
/// Output rows a block computes: one per warp.
const ROWS_PER_BLOCK: usize = BLOCK as usize / 32;

// The entries' launch attributes spell the block as a literal.
const _: () = assert!(BLOCK == 256);

/// The F8_E4M3 byte in the low eight bits of `b` (OCP E4M3FN) times `2^9`:
/// `m` for a subnormal (exponent field 0), `(8 + m) · 2^(e − 1)` for a
/// normal, negated by bit 7. Exact. The NaN bytes are the planes' refusal,
/// not values here.
#[inline(always)]
#[must_use]
pub fn e4m3_int(b: u32) -> i32 {
    let (m, e) = ((b & 7) as i32, (b >> 3) & 15);
    let mag = if e == 0 { m } else { (8 | m) << (e - 1) };
    if b & 0x80 == 0 { mag } else { -mag }
}

/// `2^(E − 136)` for the E8M0 byte `E` (0..=254): its `2^(E − 127)` times the
/// `2^-9` of [`e4m3_int`]'s integers, exact in f32 (a subnormal for `E < 10`).
#[inline(always)]
#[must_use]
pub fn block_scale(e: u32) -> f32 {
    f32::from_bits(if e >= 10 {
        (e - 9) << 23
    } else {
        1 << (e + 13)
    })
}

/// The exact dot of a block's 32 value bytes `w` (value `4i + j` byte `j` of
/// word `i`, as [`e4m3_int`] reads it) with the column's 32 int8 codes `a`
/// (the same order): at most `32 · 229376 · 128 < 2^30` in magnitude.
///
/// Each word by its constant index, so the arrays stay in registers.
#[inline(always)]
#[must_use]
pub fn block_isum(w: [u32; 8], a: [u32; 8]) -> i32 {
    word_isum(w[0], a[0])
        + word_isum(w[1], a[1])
        + word_isum(w[2], a[2])
        + word_isum(w[3], a[3])
        + word_isum(w[4], a[4])
        + word_isum(w[5], a[5])
        + word_isum(w[6], a[6])
        + word_isum(w[7], a[7])
}

/// The exact dot of one word's four value bytes with its four int8 codes.
#[inline(always)]
fn word_isum(wi: u32, ai: u32) -> i32 {
    e4m3_int(wi & 0xFF) * i32::from(ai as u8 as i8)
        + e4m3_int((wi >> 8) & 0xFF) * i32::from((ai >> 8) as u8 as i8)
        + e4m3_int((wi >> 16) & 0xFF) * i32::from((ai >> 16) as u8 as i8)
        + e4m3_int(wi >> 24) * i32::from((ai >> 24) as u8 as i8)
}

/// Lane `lane`'s partial of one row's dot with one activation column (the
/// caller reduces over the warp): lane `ℓ` takes blocks `ℓ, ℓ + 32, ..` of row
/// `row_abs` and adds each block's `f32(isum) · 2^(E − 136)` times the
/// column's scale of the block, `d[d0 + (b >> D_SHIFT)]` (module doc).
///
/// # Safety
///
/// `qv` (the code plane as 16-byte vectors) holds the row's `2 · n_blk`
/// vectors from vector `2 · n_blk · row_abs`; `e` its `e_cols` words from word
/// `e_cols · row_abs`, `4 · e_cols >= n_blk`; `q` the column's words up to
/// `q0 + 256 · ceil(n_blk / 32)`; `d` its scales `d0 + (b >> D_SHIFT)` for
/// every `b < n_blk`; `lane < 32`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub unsafe fn lane_partial<const D_SHIFT: usize>(
    qv: &[U32x4],
    e: &[u32],
    (row_abs, n_blk, e_cols): (usize, usize, usize),
    q: &[u32],
    d: &[f32],
    (q0, d0): (usize, usize),
    lane: usize,
) -> f32 {
    let (v0, e0) = (2 * n_blk * row_abs, e_cols * row_abs);
    let mut f = 0.0f32;
    let mut b = lane;
    while b < n_blk {
        // SAFETY: b < n_blk, so the block's two vectors, its scale word and
        // its column words and scale are inside the planes by this fn's
        // contract.
        let (lo, hi, ew, a, da) = unsafe {
            let qb = q0 + 256 * (b >> 5) + (b & 31);
            (
                qv.get_unchecked(v0 + 2 * b).0,
                qv.get_unchecked(v0 + 2 * b + 1).0,
                *e.get_unchecked(e0 + (b >> 2)),
                [
                    *q.get_unchecked(qb),
                    *q.get_unchecked(qb + 32),
                    *q.get_unchecked(qb + 64),
                    *q.get_unchecked(qb + 96),
                    *q.get_unchecked(qb + 128),
                    *q.get_unchecked(qb + 160),
                    *q.get_unchecked(qb + 192),
                    *q.get_unchecked(qb + 224),
                ],
                *d.get_unchecked(d0 + (b >> D_SHIFT)),
            )
        };
        let w = [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
        let v = mul_rn_f32(
            block_isum(w, a) as f32,
            block_scale((ew >> (8 * (b & 3))) & 0xFF),
        );
        f = add_rn_f32(f, mul_rn_f32(v, da));
        b += 32;
    }
    f
}

/// Whether a slot before `slot` names `id`. Warp-collective: every lane passes
/// the same `sel`, `id` and `slot` and its own `lane`, and all get the answer.
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
mod mxfp8_sel_kernels {
    use super::*;

    /// The MXFP8 gate·up: thread row `n = slot · rows_per_expert + r` (one
    /// warp a row) dots row `sel[slot] · rows_per_expert + r` of both stacks
    /// with `Q8Act` column `slot / slots_per_col` ([`lane_partial`]), reduces
    /// each by the warp tree and stores `h[n] = act::apply_checked(act,
    /// limit, g, u, fault)` from lane 0. `n_blk` is a row's 32-value blocks,
    /// `e_cols` its scale words, `iters` the column's 1024-value groups. Ids:
    /// module doc.
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
            gq.len() >= n_experts * rows_per_expert * 8 * n_blk,
            ge.len() >= n_experts * rows_per_expert * e_cols,
            uq.len() >= n_experts * rows_per_expert * 8 * n_blk,
            ue.len() >= n_experts * rows_per_expert * e_cols,
            4 * e_cols >= n_blk,
            slots_per_col >= 1,
            n_slots <= m_cols * slots_per_col,
            32 * iters >= n_blk,
            q.len() >= m_cols * 256 * iters,
            4 * d8.len() >= m_cols * n_blk,
            sel.len() >= n_slots,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn mxfp8_gate_up_sel(
        gq: &[u32],
        ge: &[u32],
        uq: &[u32],
        ue: &[u32],
        q: &[u32],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        m_cols: u32,
        slots_per_col: u32,
        n_blk: u32,
        e_cols: u32,
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
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the load
        // is warp-uniform (the warp's lanes share `row`, hence `slot`), so the
        // returns below never diverge a warp.
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
        // The launcher checks 16-byte bases; a row is `2 · n_blk` vectors.
        let (Some(gv), Some(uv)) = (as_vectors::<U32x4>(gq), as_vectors::<U32x4>(uq)) else {
            return;
        };
        let row_abs = id as usize * rpe + row % rpe;
        let col = slot / slots_per_col as usize;
        let (n_blk, e_cols) = (n_blk as usize, e_cols as usize);
        let at = (col * 256 * iters as usize, col * (n_blk >> 2));
        // SAFETY: row_abs < n_experts · rows_per_expert, so the row's vectors
        // and scale words are inside both stacks' planes (the stack bounds);
        // col < m_cols because slot < n_slots <= m_cols · slots_per_col, its
        // 256 · iters >= 256 · ceil(n_blk / 32) words and n_blk / 4 scales
        // inside q and d8 (n_blk a multiple of 8: a Q8Act's whole
        // super-blocks); lane < 32.
        let (fg, fu) = unsafe {
            (
                lane_partial::<2>(gv, ge, (row_abs, n_blk, e_cols), q, d8, at, lane),
                lane_partial::<2>(uv, ue, (row_abs, n_blk, e_cols), q, d8, at, lane),
            )
        };
        let g = warp::reduce_sum_f32(fg);
        let u = warp::reduce_sum_f32(fu);
        if lane == 0 {
            let v = act::apply_checked(act, limit, g, u, fault);
            // SAFETY: row < n_slots · rows_per_expert <= h.len() by the launch
            // contract; only lane 0 of the warp writes h[row].
            unsafe { *h.get_unchecked_mut(row) = v };
        }
    }

    /// `mxfp8_gate_up_sel` over slots of more than one column: thread row `n
    /// = slot · rows_per_expert + r`. The warp of the first slot that names an
    /// expert dots row `r` of the expert's gate and up stacks with the
    /// activation column of each slot that names it, one slot after the
    /// other, and writes each one's row of `h`; the other slots' warps return
    /// before any weight load. A slot's row is `mxfp8_gate_up_sel`'s bit for
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
            gq.len() >= n_experts * rows_per_expert * 8 * n_blk,
            ge.len() >= n_experts * rows_per_expert * e_cols,
            uq.len() >= n_experts * rows_per_expert * 8 * n_blk,
            ue.len() >= n_experts * rows_per_expert * e_cols,
            4 * e_cols >= n_blk,
            slots_per_col >= 1,
            n_slots <= m_cols * slots_per_col,
            32 * iters >= n_blk,
            q.len() >= m_cols * 256 * iters,
            4 * d8.len() >= m_cols * n_blk,
            sel.len() >= n_slots,
            h.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn mxfp8_gate_up_sel_cols(
        gq: &[u32],
        ge: &[u32],
        uq: &[u32],
        ue: &[u32],
        q: &[u32],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        m_cols: u32,
        slots_per_col: u32,
        n_blk: u32,
        e_cols: u32,
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
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the load
        // is warp-uniform, so the returns below never diverge a warp.
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
        // The launcher checks 16-byte bases; a row is `2 · n_blk` vectors.
        let (Some(gv), Some(uv)) = (as_vectors::<U32x4>(gq), as_vectors::<U32x4>(uq)) else {
            return;
        };
        let (r, row_abs) = (row % rpe, id as usize * rpe + row % rpe);
        let (n_slots, spc) = (n_slots as usize, slots_per_col as usize);
        let (n_blk, e_cols) = (n_blk as usize, e_cols as usize);
        let mut i0 = slot;
        while i0 < n_slots {
            let i = i0 + lane;
            // SAFETY: i < n_slots <= sel.len() by the launch contract.
            let mut hits = warp::ballot(i < n_slots && unsafe { *sel.get_unchecked(i) } == id);
            while hits != 0 {
                let s = i0 + hits.trailing_zeros() as usize;
                hits &= hits - 1;
                let col = s / spc;
                let at = (col * 256 * iters as usize, col * (n_blk >> 2));
                // SAFETY: mxfp8_gate_up_sel's bounds for slot `s` < n_slots:
                // the row's vectors and scale words inside both stacks, column
                // col < m_cols with its words and scales inside q and d8; lane
                // < 32.
                let (fg, fu) = unsafe {
                    (
                        lane_partial::<2>(gv, ge, (row_abs, n_blk, e_cols), q, d8, at, lane),
                        lane_partial::<2>(uv, ue, (row_abs, n_blk, e_cols), q, d8, at, lane),
                    )
                };
                let g = warp::reduce_sum_f32(fg);
                let u = warp::reduce_sum_f32(fu);
                if lane == 0 {
                    let v = act::apply_checked(act, limit, g, u, fault);
                    // SAFETY: s · rpe + r < n_slots · rows_per_expert <= h.len()
                    // by the launch contract; slot `s` names this warp's expert
                    // and this warp is its first slot, so lane 0 here is the
                    // row's only writer.
                    unsafe { *h.get_unchecked_mut(s * rpe + r) = v };
                }
            }
            i0 += 32;
        }
    }

    /// The MXFP8 down `_sel` over the 32-value q8_1 form: thread row `n = slot
    /// · rows_per_expert + r` stores `y[n]`, the dot of row `sel[slot] ·
    /// rows_per_expert + r` of the stack with activation column `slot`
    /// ([`lane_partial`], a scale a block), reduced by the warp tree. Ids:
    /// module doc.
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
            e.len() >= n_experts * rows_per_expert * e_cols,
            4 * e_cols >= n_blk,
            32 * iters >= n_blk,
            q_stride >= 256 * iters,
            q.len() >= n_slots * q_stride,
            d8.len() >= n_slots * n_blk,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn mxfp8_gemv_sel32(
        qs: &[u32],
        e: &[u32],
        q: &[u32],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        n_blk: u32,
        e_cols: u32,
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
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the load
        // is warp-uniform, so the returns below never diverge a warp.
        let id = unsafe { *sel.get_unchecked(slot) };
        let lane = warp::lane_id() as usize;
        if id >= n_experts {
            if id != HOST && lane == 0 {
                fault.raise(FaultSite::ExpertId);
            }
            return;
        }
        // The launcher checks a 16-byte base; a row is `2 · n_blk` vectors.
        let Some(qv) = as_vectors::<U32x4>(qs) else {
            return;
        };
        let row_abs = id as usize * rpe + row % rpe;
        let n_blk = n_blk as usize;
        let at = (slot * q_stride as usize, slot * n_blk);
        // SAFETY: row_abs < n_experts · rows_per_expert bounds the row's
        // vectors and scale words (the stack bounds); column slot < n_slots
        // holds q_stride >= 256 · ceil(n_blk / 32) words and n_blk scales;
        // lane < 32.
        let f =
            unsafe { lane_partial::<0>(qv, e, (row_abs, n_blk, e_cols as usize), q, d8, at, lane) };
        let s = warp::reduce_sum_f32(f);
        if lane == 0 {
            // SAFETY: row < n_slots · rows_per_expert <= y.len() by the launch
            // contract; only lane 0 of the warp writes y[row].
            unsafe { *y.get_unchecked_mut(row) = s };
        }
    }

    /// `mxfp8_gemv_sel32` over slots of more than one token: the warp of the
    /// first slot that names an expert dots row `r` of the expert with the
    /// column of each slot that names it, one after the other, and writes each
    /// one's row of `y`; the other slots' warps return before any weight load.
    /// A slot's row is `mxfp8_gemv_sel32`'s bit for bit. Ids: module doc.
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
            e.len() >= n_experts * rows_per_expert * e_cols,
            4 * e_cols >= n_blk,
            32 * iters >= n_blk,
            q_stride >= 256 * iters,
            q.len() >= n_slots * q_stride,
            d8.len() >= n_slots * n_blk,
            sel.len() >= n_slots,
            y.len() >= n_slots * rows_per_expert
        )
    )]
    pub fn mxfp8_gemv_sel32_cols(
        qs: &[u32],
        e: &[u32],
        q: &[u32],
        d8: &[f32],
        sel: &[u32],
        n_experts: u32,
        rows_per_expert: u32,
        n_slots: u32,
        n_blk: u32,
        e_cols: u32,
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
        // SAFETY: slot < n_slots <= sel.len() by the launch contract; the load
        // is warp-uniform, so the returns below never diverge a warp.
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
        // The launcher checks a 16-byte base; a row is `2 · n_blk` vectors.
        let Some(qv) = as_vectors::<U32x4>(qs) else {
            return;
        };
        let (r, row_abs) = (row % rpe, id as usize * rpe + row % rpe);
        let (n_slots, n_blk, e_cols) = (n_slots as usize, n_blk as usize, e_cols as usize);
        let mut i0 = slot;
        while i0 < n_slots {
            let i = i0 + lane;
            // SAFETY: i < n_slots <= sel.len() by the launch contract.
            let mut hits = warp::ballot(i < n_slots && unsafe { *sel.get_unchecked(i) } == id);
            while hits != 0 {
                let s = i0 + hits.trailing_zeros() as usize;
                hits &= hits - 1;
                let at = (s * q_stride as usize, s * n_blk);
                // SAFETY: mxfp8_gemv_sel32's bounds for slot `s` < n_slots.
                let f =
                    unsafe { lane_partial::<0>(qv, e, (row_abs, n_blk, e_cols), q, d8, at, lane) };
                let v = warp::reduce_sum_f32(f);
                if lane == 0 {
                    // SAFETY: s · rpe + r < n_slots · rows_per_expert <= y.len()
                    // by the launch contract; this warp is the only writer of
                    // the rows of the slots that name its expert.
                    unsafe { *y.get_unchecked_mut(s * rpe + r) = v };
                }
            }
            i0 += 32;
        }
    }
}

/// An MXFP8 routed stack in its two planes (`DevWeight::Mxfp8`,
/// `CardFormat::Mxfp8Planes`): `qs` the experts' rows of `k / 4` value words,
/// `e` the same rows of `ceil(k / 128)` scale words.
#[derive(Clone, Copy)]
pub struct Mxfp8Stack<'a> {
    pub qs: &'a DeviceTensor<u32>,
    pub e: &'a DeviceTensor<u32>,
}

impl Mxfp8Stack<'_> {
    /// The experts the planes hold at `rows_per_expert` rows of `k` values
    /// each (`k` a positive multiple of 32), or the shape refusal naming
    /// `what`.
    pub fn experts(&self, what: &'static str, k: usize, rpe: usize) -> Result<usize, GpuError> {
        let rows = self.qs.rows();
        let fits = k > 0
            && k.is_multiple_of(32)
            && self.qs.cols() == k / 4
            && self.e.cols() == k.div_ceil(128)
            && self.e.rows() == rows
            && rpe > 0
            && rows > 0
            && rows.is_multiple_of(rpe);
        if !fits {
            return Err(GpuError::shape(
                what,
                format!(
                    "MXFP8 planes of {rows} x {} code and {} x {} scale words are not experts of \
                     {rpe} rows of k = {k} (k / 4 and ceil(k / 128) words a row)",
                    self.qs.cols(),
                    self.e.rows(),
                    self.e.cols()
                ),
            ));
        }
        Ok(rows / rpe)
    }
}

/// An MXFP8 gate·up launch ([`Mxfp8SelKernels::enqueue_gate_up`]).
pub struct Mxfp8GateUp<'a> {
    pub gate: Mxfp8Stack<'a>,
    pub up: Mxfp8Stack<'a>,
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

/// An MXFP8 down launch ([`Mxfp8SelKernels::enqueue_sel32`]).
pub struct Mxfp8Down<'a> {
    pub w: Mxfp8Stack<'a>,
    /// At least `n_slots` 32-value q8_1 columns, column `s` read for slot `s`.
    pub act: &'a Q8Blocks32,
    /// At least `n_slots` ids, read on the device at each launch.
    pub sel: &'a DeviceBuffer<u32>,
    pub n_slots: usize,
    pub rows_per_expert: usize,
    /// Slots a token: a run of more than one token takes the `_cols` entry.
    pub slots_per_col: usize,
}

/// The loaded MXFP8 expert-select module. Owns no stream: every enqueue takes
/// the engine stream, so launches order with the step and are capturable.
pub struct Mxfp8SelKernels {
    module: mxfp8_sel_kernels::LoadedModule,
    /// The fault word of the `Gpu` that owns the context: the launches' sinks
    /// point into it, and the module keeps it alive.
    _fault: Arc<DeviceBuffer<u32>>,
}

impl Mxfp8SelKernels {
    /// Load this file's device bundle into `ctx`, whose launches raise into
    /// `word`, the fault word of the `Gpu` that owns `ctx`
    /// ([`crate::Gpu::fault_word`]); a word of another context is refused.
    /// Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        word: &Arc<DeviceBuffer<u32>>,
    ) -> Result<Mxfp8SelKernels, GpuError> {
        let fault = crate::module_fault_word(ctx, word, "Mxfp8SelKernels::load")?;
        // SAFETY: this package owns the embedded device bundle produced for the
        // module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(mxfp8_sel_kernels, ctx)? };
        Ok(Mxfp8SelKernels {
            module,
            _fault: fault,
        })
    }

    /// Enqueue the MXFP8 gate·up: slot `s` writes `h[s · rows_per_expert
    /// ..][..rows_per_expert]` (module doc); a run of more than one column
    /// takes the `_cols` entry. Stacks of another shape than the activation's
    /// width, an up unlike the gate, unaligned code planes, or slots the
    /// buffers cannot hold are named errors. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_gate_up(
        &self,
        stream: &CudaStream,
        a: &Mxfp8GateUp<'_>,
        fault: FaultSink,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.gate_up_entry(stream, a, a.n_slots > a.slots_per_col, fault, h)
    }

    /// [`Mxfp8SelKernels::enqueue_gate_up`] on the entry `cols` names.
    pub(crate) fn gate_up_entry(
        &self,
        stream: &CudaStream,
        a: &Mxfp8GateUp<'_>,
        cols: bool,
        fault: FaultSink,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        use crate::kquant::sel::{Fmt, GateUpAct, gate_up_dims};

        let what = "Mxfp8SelKernels::enqueue_gate_up";
        let k = a.act.k();
        a.gate.experts(what, k, a.rows_per_expert)?;
        a.up.experts(what, k, a.rows_per_expert)?;
        if a.up.e.cols() != a.gate.e.cols() || a.up.e.rows() != a.gate.e.rows() {
            return Err(GpuError::shape(
                what,
                "the up stack's scale plane must be the gate stack's shape",
            ));
        }
        for p in [a.gate.qs, a.up.qs] {
            crate::tensor::check_alignment::<U32x4, _>(what, "MXFP8 code plane", p.buf())?;
        }
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
                name: "MXFP8 code planes",
                words: 256 / 4,
            },
            &common,
            h.len(),
        )?;
        let n_blk = launch_u32(what, "n_blk", k / 32)?;
        let e_cols = launch_u32(what, "e_cols", a.gate.e.cols())?;
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
                    a.up.qs.buf(),
                    a.up.e.buf(),
                    &a.act.q4,
                    &a.act.d8,
                    a.sel,
                    d.n_experts,
                    d.rows_per_expert,
                    d.n_slots,
                    d.m_cols,
                    d.slots_per_col,
                    n_blk,
                    e_cols,
                    d.iters,
                    d.act,
                    d.limit,
                    fault,
                    h,
                )?
            }};
        }
        if cols {
            launch!(prepare_mxfp8_gate_up_sel_cols, mxfp8_gate_up_sel_cols);
        } else {
            launch!(prepare_mxfp8_gate_up_sel, mxfp8_gate_up_sel);
        }
        Ok(())
    }

    /// Enqueue the MXFP8 down `_sel` over the 32-value activation: slot `s`
    /// writes `y[s · rows_per_expert ..][..rows_per_expert]` (module doc); a
    /// run of more than one token takes the `_cols` entry. A stack of another
    /// width than the activation's, an unaligned code plane, fewer activation
    /// columns or ids than slots, or a short `y` are named errors.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_sel32(
        &self,
        stream: &CudaStream,
        a: &Mxfp8Down<'_>,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.sel32_entry(stream, a, a.n_slots > a.slots_per_col, fault, y)
    }

    /// [`Mxfp8SelKernels::enqueue_sel32`] on the entry `cols` names.
    pub(crate) fn sel32_entry(
        &self,
        stream: &CudaStream,
        a: &Mxfp8Down<'_>,
        cols: bool,
        fault: FaultSink,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "Mxfp8SelKernels::enqueue_sel32";
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
        crate::tensor::check_alignment::<U32x4, _>(what, "MXFP8 code plane", a.w.qs.buf())?;
        let n_blk = k / 32;
        let grid = launch_u32(what, "grid", slot_rows.div_ceil(ROWS_PER_BLOCK))?;
        macro_rules! launch {
            ($prepare:ident, $entry:ident) => {{
                let prep = self.module.$prepare(LaunchConfig1D::new(grid, BLOCK, 0))?;
                self.module.$entry(
                    stream,
                    &prep,
                    a.w.qs.buf(),
                    a.w.e.buf(),
                    &a.act.q,
                    &a.act.d8,
                    a.sel,
                    launch_u32(what, "n_experts", n_experts)?,
                    launch_u32(what, "rows_per_expert", rpe)?,
                    launch_u32(what, "n_slots", n_slots)?,
                    launch_u32(what, "n_blk", n_blk)?,
                    launch_u32(what, "e_cols", a.w.e.cols())?,
                    launch_u32(what, "iters", n_blk.div_ceil(32))?,
                    launch_u32(what, "q_stride", a.act.q_stride())?,
                    fault,
                    y,
                )?
            }};
        }
        if cols {
            launch!(prepare_mxfp8_gemv_sel32_cols, mxfp8_gemv_sel32_cols);
        } else {
            launch!(prepare_mxfp8_gemv_sel32, mxfp8_gemv_sel32);
        }
        Ok(())
    }
}

/// The host's twin of the device rule over one row and one column: the row's
/// value bytes `codes` (value order) under its blocks' E8M0 bytes `scales`,
/// the column's int8 codes `a` and its scales `d`, block `b` reading
/// `d[b >> d_shift]`; lane `ℓ`'s partial over blocks `ℓ, ℓ + 32, ..` as
/// [`lane_partial`] adds them, then the warp tree's butterfly
/// (`warp::reduce_sum_f32`). The bits the entries store.
#[cfg(test)]
pub(crate) fn rule_dot(codes: &[u8], scales: &[u8], a: &[i8], d: &[f32], d_shift: usize) -> f32 {
    let mut lanes = [0.0f32; 32];
    let (codes, a) = (codes.as_chunks::<32>().0, a.as_chunks::<32>().0);
    for (b, (w, x)) in codes.iter().zip(a).enumerate() {
        let isum: i32 = w
            .iter()
            .zip(x)
            .map(|(&w, &x)| e4m3_int(u32::from(w)) * i32::from(x))
            .sum();
        let v = isum as f32 * block_scale(u32::from(scales[b]));
        lanes[b % 32] += v * d[b >> d_shift];
    }
    for off in [16, 8, 4, 2, 1] {
        let prev = lanes;
        for (l, v) in lanes.iter_mut().enumerate() {
            *v = prev[l] + prev[l ^ off];
        }
    }
    lanes[0]
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::Gpu;
    use crate::fault::{Fault, LAYER_NONE};
    use crate::q8f32::{E4m3Scales, e4m3_planes};
    use gguf::GgmlType;

    /// Experts a test stack holds.
    pub(crate) const EXPERTS: usize = 5;

    /// Next value of a xorshift64 stream.
    pub(crate) fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// `rows` rows of `k` values of MXFP8 file blocks: E8M0 bytes 118..=133,
    /// every value byte but the two NaNs.
    pub(crate) fn file_bytes(rows: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut out = vec![0u8; rows * k / 32 * 33];
        for blk in out.as_chunks_mut::<33>().0 {
            blk[0] = 118 + (next(&mut state) % 16) as u8;
            for v in &mut blk[1..] {
                let b = (next(&mut state) & 0xFF) as u8;
                *v = if b & 0x7F == 0x7F { b ^ 1 } else { b };
            }
        }
        out
    }

    /// A row's value bytes in value order and its blocks' E8M0 bytes.
    pub(crate) fn row_of(bytes: &[u8], row: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
        let blocks = &bytes[row * k / 32 * 33..][..k / 32 * 33];
        let blocks = blocks.as_chunks::<33>().0;
        let codes = blocks.iter().flat_map(|b| b[1..].to_vec()).collect();
        let scales = blocks.iter().map(|b| b[0]).collect();
        (codes, scales)
    }

    /// The f64 dot of row `row` (dequantized by `gguf`, not by this file)
    /// with `x`, and the band the column's q8 rounding leaves it in: each
    /// value's `|w|` times half its block's scale `dq(i)`, plus `2^-20` of the
    /// dot's magnitude for the f32 walk.
    pub(crate) fn f64_band(
        bytes: &[u8],
        (row, k): (usize, usize),
        x: &[f32],
        dq: impl Fn(usize) -> f32,
    ) -> (f64, f64) {
        let mut w = vec![0.0f32; k];
        let src = &bytes[row * k / 32 * 33..][..k / 32 * 33];
        gguf::quant::dequant_row(GgmlType::MXFP8, src, &mut w).expect("MXFP8 blocks");
        let (mut dot, mut mag, mut q8) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..k {
            let (wi, xi) = (f64::from(w[i]), f64::from(x[i]));
            dot += wi * xi;
            mag += (wi * xi).abs();
            q8 += wi.abs() * f64::from(dq(i)) / 2.0;
        }
        (dot, q8 + mag * f64::from(2.0f32.powi(-20)))
    }

    /// The planes of `rows` rows of `k` values on the card.
    pub(crate) fn planes(
        stream: &CudaStream,
        bytes: &[u8],
        rows: usize,
        k: usize,
    ) -> (DeviceTensor<u32>, DeviceTensor<u32>) {
        let (qs, e) = e4m3_planes(E4m3Scales::Mxfp8, bytes, rows, k).expect("the planes");
        (
            DeviceTensor::upload(stream, &qs, rows, k / 4).expect("qs"),
            DeviceTensor::upload(stream, &e, rows, k.div_ceil(128)).expect("e"),
        )
    }

    /// A Q8Act column's codes in value order and its 128-value scales.
    fn q8act_column(q4: &[u32], d8: &[f32], c: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
        let (n_blk, iters) = (k / 32, (k / 32).div_ceil(32));
        let q0 = c * 256 * iters;
        let mut a = vec![0i8; k];
        for b in 0..n_blk {
            for i in 0..8 {
                let w = q4[q0 + 256 * (b >> 5) + (b & 31) + 32 * i];
                for j in 0..4 {
                    a[32 * b + 4 * i + j] = (w >> (8 * j)) as u8 as i8;
                }
            }
        }
        (a, d8[c * k / 128..][..k / 128].to_vec())
    }

    /// The MXFP8 `_sel` pair against the host: three tokens of four slots
    /// over five experts (duplicates within and across tokens, a host slot),
    /// gate·up over a `Q8Act` (K 2304) and down over a `Q8Blocks32` (K 1120,
    /// a last scale word of three blocks). Every gate and up sum and every
    /// down row is [`rule_dot`] over the activations' own codes bit for bit
    /// (`h` its activation within the device exp's ulps); each lies within
    /// the q8 band of the f64 dot of `gguf`'s dequantized row with the
    /// unquantized column; the `_cols` entries' rows are the `_sel` entries'
    /// bit for bit; a host slot's rows keep what they held; an id past the
    /// stack raises `ExpertId` and NaNs its gate·up rows.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it"]
    fn hw_mxfp8_sel_rows_are_the_integer_rule() {
        const K_GU: usize = 2304;
        const FF: usize = 48;
        const K_D: usize = 1120;
        const N: usize = 64;
        const SPC: usize = 4;
        const M: usize = 3;
        const SENT: f32 = 1.0e30;
        let sel_host: [u32; M * SPC] = [3, 0, HOST, 4, 0, 3, 1, 1, 4, HOST, 3, 0];
        let gpu = Gpu::new().expect("card");
        let stream = gpu.stream();
        let k8 = Mxfp8SelKernels::load(gpu.context(), gpu.fault_word()).expect("the module");
        let (gb, ub, db) = (
            file_bytes(EXPERTS * FF, K_GU, 0x51),
            file_bytes(EXPERTS * FF, K_GU, 0x52),
            file_bytes(EXPERTS * N, K_D, 0x53),
        );
        let (gq, ge) = planes(stream, &gb, EXPERTS * FF, K_GU);
        let (uq, ue) = planes(stream, &ub, EXPERTS * FF, K_GU);
        let (dq, de) = planes(stream, &db, EXPERTS * N, K_D);
        let mut state = 0x77u64;
        let mut val = |_| ((next(&mut state) >> 33) % 2001) as f32 / 1000.0 - 1.0;
        let x_gu: Vec<f32> = (0..M * K_GU).map(&mut val).collect();
        let x_d: Vec<f32> = (0..M * SPC * K_D).map(&mut val).collect();
        let xg = DeviceBuffer::from_host(stream, &x_gu).expect("x");
        let xd = DeviceBuffer::from_host(stream, &x_d).expect("x");
        let mut act = Q8Act::with_k(stream, M, K_GU).expect("Q8Act");
        gpu.enqueue_quantize_q8_1_cols(&xg, &mut act, M, 0)
            .expect("quantize");
        let mut act32 = Q8Blocks32::with_slots(stream, K_D, M * SPC).expect("Q8Blocks32");
        gpu.q5()
            .enqueue_quantize_q8(stream, &xd, &mut act32, gpu.unlabelled_sink())
            .expect("quantize32");
        let sel = DeviceBuffer::from_host(stream, &sel_host).expect("sel");
        let (q4, d8) = (
            act.q4.to_host_vec(stream).expect("q4"),
            act.d8.to_host_vec(stream).expect("d8"),
        );
        let a32 = act32.readback(stream).expect("the 32-value form");
        let (code, limit) = Act::SiluMul.code("test").expect("the rule");
        let mut failures = Vec::new();

        for (slots, spc) in [(SPC, SPC), (M * SPC, SPC)] {
            let mut rows = [Vec::new(), Vec::new()];
            let mut downs = [Vec::new(), Vec::new()];
            for (i, cols) in [false, true].into_iter().enumerate() {
                let mut h = DeviceBuffer::from_host(stream, &vec![SENT; slots * FF]).expect("h");
                let a = Mxfp8GateUp {
                    gate: Mxfp8Stack { qs: &gq, e: &ge },
                    up: Mxfp8Stack { qs: &uq, e: &ue },
                    act: &act,
                    sel: &sel,
                    n_slots: slots,
                    rows_per_expert: FF,
                    slots_per_col: spc,
                    rule: Act::SiluMul,
                };
                k8.gate_up_entry(stream, &a, cols, gpu.unlabelled_sink(), &mut h)
                    .expect("gate·up");
                let mut y = DeviceBuffer::from_host(stream, &vec![SENT; slots * N]).expect("y");
                let d = Mxfp8Down {
                    w: Mxfp8Stack { qs: &dq, e: &de },
                    act: &act32,
                    sel: &sel,
                    n_slots: slots,
                    rows_per_expert: N,
                    slots_per_col: spc,
                };
                k8.sel32_entry(stream, &d, cols, gpu.unlabelled_sink(), &mut y)
                    .expect("down");
                assert_eq!(gpu.take_fault().expect("fault word"), None);
                rows[i] = h.to_host_vec(stream).expect("h");
                downs[i] = y.to_host_vec(stream).expect("y");
            }
            if rows[0]
                .iter()
                .zip(&rows[1])
                .any(|(a, b)| a.to_bits() != b.to_bits())
                || downs[0]
                    .iter()
                    .zip(&downs[1])
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                failures.push(format!(
                    "{slots} slots: the _cols rows are not the _sel rows"
                ));
            }
            for (s, &id) in sel_host[..slots].iter().enumerate() {
                let (hs, ys) = (&rows[0][s * FF..][..FF], &downs[0][s * N..][..N]);
                if id == HOST {
                    if hs.iter().chain(ys).any(|&v| v != SENT) {
                        failures.push(format!("{slots} slots: host slot {s} was written"));
                    }
                    continue;
                }
                let col = s / spc;
                let (a, d) = q8act_column(&q4, &d8, col, K_GU);
                let xs = &x_gu[col * K_GU..][..K_GU];
                for (r, &hr) in hs.iter().enumerate() {
                    let row = id as usize * FF + r;
                    let mut gu = [0.0f32; 2];
                    for (j, bytes) in [&gb, &ub].into_iter().enumerate() {
                        let (codes, scales) = row_of(bytes, row, K_GU);
                        gu[j] = rule_dot(&codes, &scales, &a, &d, 2);
                        let (dot, band) = f64_band(bytes, (row, K_GU), xs, |i| d[i / 128]);
                        if (f64::from(gu[j]) - dot).abs() > band {
                            failures.push(format!(
                                "{slots} slots: slot {s} row {r} {}: {} against f64 {dot} ± {band}",
                                ["gate", "up"][j],
                                gu[j]
                            ));
                        }
                    }
                    let want = act::apply(code, limit, gu[0], gu[1]);
                    if (hr - want).abs() > 4.0 * f32::EPSILON * want.abs().max(1.0) {
                        failures.push(format!(
                            "{slots} slots: slot {s} row {r}: h {hr} against the rule's {want}"
                        ));
                    }
                }
                let (a, d) = a32.column(s, K_D, M * SPC);
                let xs = &x_d[s * K_D..][..K_D];
                for (r, &yr) in ys.iter().enumerate() {
                    let row = id as usize * N + r;
                    let (codes, scales) = row_of(&db, row, K_D);
                    let want = rule_dot(&codes, &scales, &a, &d, 0);
                    let (dot, band) = f64_band(&db, (row, K_D), xs, |i| d[i / 32]);
                    if yr.to_bits() != want.to_bits() || (f64::from(want) - dot).abs() > band {
                        failures.push(format!(
                            "{slots} slots: slot {s} down row {r}: {yr} against the rule's {want} \
                             and f64 {dot} ± {band}"
                        ));
                    }
                }
            }
        }

        let bad = DeviceBuffer::from_host(stream, &[EXPERTS as u32 + 2]).expect("bad id");
        let mut h = DeviceBuffer::zeroed(stream, FF).expect("h");
        let a = Mxfp8GateUp {
            gate: Mxfp8Stack { qs: &gq, e: &ge },
            up: Mxfp8Stack { qs: &uq, e: &ue },
            act: &act,
            sel: &bad,
            n_slots: 1,
            rows_per_expert: FF,
            slots_per_col: 1,
            rule: Act::SiluMul,
        };
        k8.enqueue_gate_up(stream, &a, gpu.unlabelled_sink(), &mut h)
            .expect("gate·up");
        let fault = gpu.take_fault().expect("fault word");
        let h = h.to_host_vec(stream).expect("h");
        if fault != Some(Fault::at(LAYER_NONE, FaultSite::ExpertId))
            || !h.iter().all(|v| v.is_nan())
        {
            failures.push(format!("id past the stack: fault {fault:?}, h[0] {}", h[0]));
        }
        println!(
            "mxfp8_sel: {} failures; first: {:?}",
            failures.len(),
            failures.first()
        );
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }
}
