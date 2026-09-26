//! The dense projections of the V4.1 step — attention, the shared expert,
//! `engram_wkv` — in the format the file carries them in. The kernel of a
//! projection follows from its resident weight's format ([`Dense::of`]),
//! resolved when the step is enqueued (at capture, once per graph); a step
//! never branches on it.
//!
//! - Q8_0: `q8f32`'s gemv, f32 activations.
//! - Q3_K and Q4_K: `bloomery_gpu`'s gemvs against the q8_1 form of the
//!   activation (`Gpu::enqueue_gemv_q3k`, `Gpu::enqueue_gemv_q4k`), the
//!   rule every K-quant site of the card runs — `cores::q3k_row_dot`,
//!   `cores::q4k_row_dot`, ik's CUDA `mmvq` shape. The caller quantizes the
//!   activation once for every projection that reads it; a Q3_K row may have
//!   an odd super-block count.
//! - Q5_K: [`DenseKernels`]'s f32-activation gemv, the one Q5_K dense site
//!   (two layers' shared down projection): each weight dequantized as
//!   `gguf::quant::dequant_row` does it, then one fused multiply-add.
//!
//! Two shapes of their own, where the file's format is a K-quant:
//! - `attn_output_a`'s block diagonal ([`DenseKernels::enqueue_q3k_heads`]):
//!   group `g`'s rows dot column `g` of a q8_1 activation of `groups`
//!   columns — each row bit for bit what `q3k_gemv` computes for it on that
//!   column alone;
//! - the shared expert's gate·up·SwiGLU
//!   ([`DenseKernels::enqueue_shexp_gate_up_q3k`]): both Q3_K dots against the
//!   one q8_1 column, then `experts::swiglu_clamp` — the routed experts'
//!   `ds41_expert_gate_up` on one expert of its own.
//!
//! Numeric contract of the Q5_K gemv: one warp per row; lane `L` walks the
//! row's super-blocks in order, and in each its eight values `64j + L` and
//! `64j + 32 + L` for `j` ascending, the pair in that order; a value is
//! `fma(q, d·sc, −(dmin·m))` with the two products rounded first
//! (`dequant_q5_k`'s op order), and `acc = fma(value, x, acc)` from 0. The
//! 32 lane sums then go through `warp::reduce_sum_f32`.
//!
//! m columns (the k-token pass): every kernel here has an m-column twin
//! (`_mcol`, m in 1..=8) whose column c is its one-column launch on column c
//! alone, bit for bit — the weights decoded once per row step for all m
//! columns, each column folded in the one-column order (`cores::q3k_row_dot`'s
//! contract; the Q5_K accumulators one per column in the order above). The
//! launchers send m = 1 to the one-column kernel. Output layouts: the plain
//! projections `y[r·m + c]` (the K-quant gemvs' layout, [`DenseKernels::enqueue_m`]);
//! the block diagonal and the shared expert token-major, `y[c·rows + r]`, the
//! m columns the next projection quantizes.

use bloomery_gpu::cores::{q3k_row_dot, q3k_row_dot_cols};
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{
    ColGroups, DeviceTensor, Gpu, GpuError, Q8Act, col_group, col_group_count, col_group_of,
    col_sums, launch_u32, store_cols,
};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::{cvt_f32_f16x2_hi, cvt_f32_f16x2_lo};
use cuda_device::float::{fma_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use gguf::quant::GgmlType;
use std::sync::Arc;

use crate::experts::swiglu_clamp;

const WHAT: &str = "deepseek41 dense";

/// Threads per block of every kernel here: eight warps, a warp per row.
const BLOCK: u32 = 256;

/// Bytes, and u32 words, of one Q5_K super-block of 256 values: `d` and
/// `dmin` (f16), 12 scale bytes, 32 high-bit bytes, 128 low-nibble bytes.
const Q5K_BYTES: usize = 176;
const Q5K_WORDS: usize = Q5K_BYTES / 4;

/// Byte `i` of the words from `base`.
///
/// # Safety
///
/// `base + i / 4 < w.len()`.
#[inline(always)]
unsafe fn byte(w: &[u32], base: usize, i: usize) -> u32 {
    // SAFETY: the covering word is inside `w` by this function's contract.
    (unsafe { *w.get_unchecked(base + i / 4) } >> (8 * (i % 4))) & 0xff
}

/// `get_scale_min_k4(j, scales)` of the super-block whose scale bytes start
/// at word `sc` (byte 4 of the super-block): the 6-bit scale and min of
/// sub-block `j` (0..8).
///
/// # Safety
///
/// `sc + 3 <= w.len()`.
#[inline(always)]
unsafe fn scale_min(w: &[u32], sc: usize, j: usize) -> (u32, u32) {
    // SAFETY: every byte index below is under 12, inside the three words
    // from `sc` by this function's contract.
    unsafe {
        if j < 4 {
            (byte(w, sc, j) & 63, byte(w, sc, j + 4) & 63)
        } else {
            (
                (byte(w, sc, j + 4) & 0x0f) | ((byte(w, sc, j - 4) >> 6) << 4),
                (byte(w, sc, j + 4) >> 4) | ((byte(w, sc, j) >> 6) << 4),
            )
        }
    }
}

/// Lane `lane`'s two values of sub-block pair `j` (0..4) of the Q5_K
/// super-block at word `base` — values `64j + lane` and `64j + 32 + lane`,
/// dequantized in the module's op order from the super-block scales `d`,
/// `dmin` and the lane's high-bit byte `qh`.
///
/// # Safety
///
/// `base + 44 <= w.len()`, `lane < 32`, `j < 4`.
#[inline(always)]
unsafe fn q5k_values(
    w: &[u32],
    base: usize,
    j: usize,
    lane: usize,
    d: f32,
    dmin: f32,
    qh: u32,
) -> (f32, f32) {
    // SAFETY: the scale words base+1 .. base+4 and the low-nibble byte 32j +
    // lane of the 128 from word base + 12 are inside the super-block.
    let ((sc1, m1), (sc2, m2), ql) = unsafe {
        (
            scale_min(w, base + 1, 2 * j),
            scale_min(w, base + 1, 2 * j + 1),
            byte(w, base + 12, 32 * j + lane),
        )
    };
    let d1 = mul_rn_f32(d, sc1 as f32);
    let n1 = mul_rn_f32(dmin, m1 as f32);
    let d2 = mul_rn_f32(d, sc2 as f32);
    let n2 = mul_rn_f32(dmin, m2 as f32);
    let q1 = (ql & 0x0f) + 16 * ((qh >> (2 * j)) & 1);
    let q2 = (ql >> 4) + 16 * ((qh >> (2 * j + 1)) & 1);
    (
        fma_rn_f32(q1 as f32, d1, -n1),
        fma_rn_f32(q2 as f32, d2, -n2),
    )
}

/// Lane `lane`'s partial sum of Q5_K row `row` (`n_sb` super-blocks, word
/// aligned: 176 bytes each) against the f32 column `x`, in the module's
/// contract order.
///
/// # Safety
///
/// `w.len() >= (row + 1) · 44 · n_sb`, `x.len() >= 256 · n_sb`, `lane < 32`.
#[inline(always)]
unsafe fn q5k_lane_partial(w: &[u32], x: &[f32], n_sb: usize, row: usize, lane: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut sb = 0usize;
    while sb < n_sb {
        let base = (row * n_sb + sb) * Q5K_WORDS;
        // SAFETY: base + 44 <= w.len() for sb < n_sb by this function's
        // contract; every word read below is one of those 44.
        let (dm, qh) = unsafe { (*w.get_unchecked(base), byte(w, base + 4, lane)) };
        let (d, dmin) = (cvt_f32_f16x2_lo(dm), cvt_f32_f16x2_hi(dm));
        let mut j = 0usize;
        while j < 4 {
            // SAFETY: the super-block is inside `w` (above), lane < 32, j < 4.
            let (v1, v2) = unsafe { q5k_values(w, base, j, lane, d, dmin, qh) };
            let at = 256 * sb + 64 * j + lane;
            // SAFETY: at + 32 < 256 · n_sb <= x.len() by this function's
            // contract.
            let (x1, x2) = unsafe { (*x.get_unchecked(at), *x.get_unchecked(at + 32)) };
            acc = fma_rn_f32(v1, x1, acc);
            acc = fma_rn_f32(v2, x2, acc);
            j += 1;
        }
        sb += 1;
    }
    acc
}

/// One column's step of the m-column Q5_K walk: `acc` folds the pair `v1`,
/// `v2` against that column's values `at` and `at + 32`, the order
/// [`q5k_lane_partial`] folds them in.
///
/// # Safety
///
/// `at + 32 < x.len()`.
#[inline(always)]
unsafe fn q5k_col_step(acc: f32, v1: f32, v2: f32, x: &[f32], at: usize) -> f32 {
    // SAFETY: both indices are inside `x` by this function's contract.
    let (x1, x2) = unsafe { (*x.get_unchecked(at), *x.get_unchecked(at + 32)) };
    fma_rn_f32(v2, x2, fma_rn_f32(v1, x1, acc))
}

/// Lane `lane`'s partial sums of Q5_K row `row` against `m` f32 columns of
/// `256 · n_sb` values (column c at `x[c·k ..]`): each value decoded once
/// and folded into every column's accumulator in [`q5k_lane_partial`]'s
/// order, so column c is that function on column c alone. Columns past `m`
/// stay 0.0.
///
/// # Safety
///
/// `w.len() >= (row + 1) · 44 · n_sb`, `x.len() >= m · 256 · n_sb`, `m` in
/// 1..=8, `lane < 32`.
#[inline(always)]
unsafe fn q5k_lane_partials(
    w: &[u32],
    x: &[f32],
    n_sb: usize,
    row: usize,
    m: usize,
    lane: usize,
) -> [f32; 8] {
    let k = 256 * n_sb;
    let [
        mut a0,
        mut a1,
        mut a2,
        mut a3,
        mut a4,
        mut a5,
        mut a6,
        mut a7,
    ] = [0.0f32; 8];
    let mut sb = 0usize;
    while sb < n_sb {
        let base = (row * n_sb + sb) * Q5K_WORDS;
        // SAFETY: base + 44 <= w.len() for sb < n_sb by this function's
        // contract; every word read below is one of those 44.
        let (dm, qh) = unsafe { (*w.get_unchecked(base), byte(w, base + 4, lane)) };
        let (d, dmin) = (cvt_f32_f16x2_lo(dm), cvt_f32_f16x2_hi(dm));
        let mut j = 0usize;
        while j < 4 {
            // SAFETY: the super-block is inside `w` (above), lane < 32, j < 4.
            let (v1, v2) = unsafe { q5k_values(w, base, j, lane, d, dmin, qh) };
            let at = 256 * sb + 64 * j + lane;
            // SAFETY: every column c < m reads c·k + at + 32 < (c + 1)·k <=
            // m·k <= x.len() by this function's contract; the guards are
            // launch-uniform, so no warp diverges on them.
            unsafe {
                a0 = q5k_col_step(a0, v1, v2, x, at);
                if m > 1 {
                    a1 = q5k_col_step(a1, v1, v2, x, k + at);
                }
                if m > 2 {
                    a2 = q5k_col_step(a2, v1, v2, x, 2 * k + at);
                }
                if m > 3 {
                    a3 = q5k_col_step(a3, v1, v2, x, 3 * k + at);
                }
                if m > 4 {
                    a4 = q5k_col_step(a4, v1, v2, x, 4 * k + at);
                }
                if m > 5 {
                    a5 = q5k_col_step(a5, v1, v2, x, 5 * k + at);
                }
                if m > 6 {
                    a6 = q5k_col_step(a6, v1, v2, x, 6 * k + at);
                }
                if m > 7 {
                    a7 = q5k_col_step(a7, v1, v2, x, 7 * k + at);
                }
            }
            j += 1;
        }
        sb += 1;
    }
    [a0, a1, a2, a3, a4, a5, a6, a7]
}

#[cuda_module]
mod dense_kernels {
    use super::*;

    /// `y[r] = W[r] · x` for a Q5_K weight of `n_rows` rows of `n_sb`
    /// super-blocks and one f32 column `x`: a warp per row, eight rows per
    /// block, the module's contract order.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_rows * 44 * n_sb,
            x.len() >= 256 * n_sb,
            y.len() >= n_rows
        )
    )]
    pub fn ds41_q5k_gemv_f32(
        w: &[u32],
        x: &[f32],
        n_rows: u32,
        n_sb: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        // SAFETY: row < n_rows puts the row's words inside w, and x holds
        // 256·n_sb values, by the launch contract; lane < 32.
        let f = unsafe { q5k_lane_partial(w, x, n_sb as usize, row, lane) };
        let s = warp::reduce_sum_f32(f);
        if lane == 0 {
            // SAFETY: row < n_rows <= y.len(); lane 0 of the row's warp is
            // its only writer.
            unsafe { *y.get_unchecked_mut(row) = s };
        }
    }

    /// `ds41_q5k_gemv_f32` over `m_cols` (1..=8) f32 columns of `256·n_sb`
    /// values, `x[c·k ..]`: `y[r·m + c]`, column c bit for bit the
    /// one-column launch on that column.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_rows * 44 * n_sb,
            x.len() >= m_cols * 256 * n_sb,
            y.len() >= n_rows * m_cols,
            m_cols >= 1,
            m_cols <= 8
        )
    )]
    pub fn ds41_q5k_gemv_f32_mcol(
        w: &[u32],
        x: &[f32],
        n_rows: u32,
        n_sb: u32,
        m_cols: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let m = m_cols as usize;
        // SAFETY: row < n_rows puts the row's words inside w, x holds m
        // columns of 256·n_sb values and 1 <= m <= 8, by the launch
        // contract; lane < 32.
        let f = unsafe { q5k_lane_partials(w, x, n_sb as usize, row, m, lane) };
        let s = col_sums(f, m);
        if lane == 0 {
            // SAFETY: slots row·m + c for c < m are inside y (y.len() >=
            // n_rows·m) and belong to this row's warp alone.
            unsafe { store_cols(&mut y, row * m, 1, m, s) };
        }
    }

    /// The block diagonal of a Q3_K weight: row `r` of `n_rows` belongs to
    /// group `r / rows_per_head` and dots that column of the q8_1 activation
    /// (`groups` columns of `n_sb` super-blocks), `cores::q3k_row_dot` at
    /// one column; `y[r]` from lane 0.
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
            4 * w.len() >= n_rows * 110 * n_sb,
            q.len() >= groups * 64 * iters,
            d8.len() >= groups * 2 * n_sb,
            groups * rows_per_head >= n_rows,
            y.len() >= n_rows
        )
    )]
    pub fn ds41_q3k_gemv_heads(
        w: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        rows_per_head: u32,
        groups: u32,
        n_sb: u32,
        iters: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        let col = row / rows_per_head as usize;
        // The second test is the contract's, warp-uniform like the first.
        if row >= n_rows as usize || col >= groups as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        // q3k_row_dot's contract: the row inside w, and column col (< groups,
        // as row < n_rows <= groups·rows_per_head) inside q and d8, all by the
        // launch contract; every lane of the warp calls it.
        let f = q3k_row_dot(w, q, d8, n_sb as usize, iters, row, col, 1, lane);
        let s = warp::reduce_sum_f32(f[0]);
        if lane == 0 {
            // SAFETY: row < n_rows <= y.len(); lane 0 of the row's warp is
            // its only writer.
            unsafe { *y.get_unchecked_mut(row) = s };
        }
    }

    /// `ds41_q3k_gemv_heads` for `m_cols` (1..=8) tokens: the activation
    /// holds `m_cols · groups` q8_1 columns, token t's group g at column
    /// `t·groups + g`, and row r of group g dots its group's column of every
    /// token (`cores::q3k_row_dot_cols`, stride `groups`). Token-major
    /// output, `y[t·n_rows + r]`; token t bit for bit the one-token launch
    /// on that token's `groups` columns.
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
            4 * w.len() >= n_rows * 110 * n_sb,
            q.len() >= m_cols * groups * 64 * iters,
            d8.len() >= m_cols * groups * 2 * n_sb,
            groups * rows_per_head >= n_rows,
            y.len() >= m_cols * n_rows,
            m_cols >= 1,
            m_cols <= 8
        )
    )]
    pub fn ds41_q3k_gemv_heads_mcol(
        w: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        rows_per_head: u32,
        groups: u32,
        m_cols: u32,
        n_sb: u32,
        iters: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        let g = row / rows_per_head as usize;
        // The second test is the contract's, warp-uniform like the first.
        if row >= n_rows as usize || g >= groups as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let m = m_cols as usize;
        // q3k_row_dot_cols's contract: the row inside w, and column g +
        // (m-1)·groups (< m·groups) inside q and d8, all by the launch
        // contract; every lane of the warp calls it.
        let f = q3k_row_dot_cols(
            w,
            q,
            d8,
            n_sb as usize,
            iters,
            row,
            g,
            groups as usize,
            m,
            lane,
        );
        let s = col_sums(f, m);
        if lane == 0 {
            // SAFETY: slots t·n_rows + row for t < m are inside y (y.len()
            // >= m·n_rows) and belong to this row's warp alone.
            unsafe { store_cols(&mut y, row, n_rows as usize, m, s) };
        }
    }

    /// [`ds41_q3k_gemv_heads_mcol`] over the token groups of a
    /// [`ColGroups`] in one grid: block `b` runs token group `b % n_groups`
    /// ([`col_group_count`]) on the eight rows from `8 · (b / n_groups)` —
    /// row `r` of head group `h = r / rows_per_head` against columns
    /// `(col0 + t0 + c)·groups + h` of the group's `m` tokens from `t0`, the
    /// core `ds41_q3k_gemv_heads_mcol` runs (its one-column body at m = 1,
    /// `ds41_q3k_gemv_heads`'s) — into the token-major `y[(t0 + c)·n_rows +
    /// r]`: token t bit for bit the one-token launch on its `groups` columns,
    /// the groups' outputs end to end.
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
            4 * w.len() >= n_rows * 110 * n_sb,
            q.len() >= col0 * groups * 64 * iters + tokens * groups * 64 * iters,
            d8.len() >= col0 * groups * 2 * n_sb + tokens * groups * 2 * n_sb,
            groups * rows_per_head >= n_rows,
            y.len() >= tokens * n_rows,
            lead >= 1,
            lead <= 8,
            lead <= tokens
        )
    )]
    pub fn ds41_q3k_gemv_heads_groups(
        w: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        rows_per_head: u32,
        groups: u32,
        col0: u32,
        lead: u32,
        tokens: u32,
        n_sb: u32,
        iters: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let (lead, tokens) = (lead as usize, tokens as usize);
        let n_groups = col_group_count(lead, tokens);
        let b = thread::blockIdx_x() as usize;
        let t = thread::threadIdx_x() as usize;
        let row = (b / n_groups) * 8 + t / 32;
        let (t0, m) = col_group(b % n_groups, lead, tokens);
        let h = row / rows_per_head as usize;
        // Block-uniform tests; the second is the contract's.
        if row >= n_rows as usize || h >= groups as usize || m == 0 {
            return;
        }
        let lane = warp::lane_id() as usize;
        let groups = groups as usize;
        // q3k_row_dot_cols's contract: the row inside w, and column (col0 +
        // t0 + m − 1)·groups + h < (col0 + tokens)·groups inside q and d8, by
        // the launch contract; every lane of the warp calls it.
        let f = q3k_row_dot_cols(
            w,
            q,
            d8,
            n_sb as usize,
            iters,
            row,
            (col0 as usize + t0) * groups + h,
            groups,
            m,
            lane,
        );
        let s = col_sums(f, m);
        if lane == 0 {
            // SAFETY: slots (t0 + c)·n_rows + row for c < m are inside y
            // (y.len() >= tokens·n_rows, t0 + m <= tokens) and belong to this
            // row's warp alone.
            unsafe { store_cols(&mut y, t0 * n_rows as usize + row, n_rows as usize, m, s) };
        }
    }

    /// The token-major copy of rows `r0 .. r0 + rows` of a grouped launch's
    /// output over `tokens` tokens ([`ColGroups`]: group `g`'s `total_rows ×
    /// m` row-major block at `total_rows · c0`): `dst[t·rows + r] =
    /// src[total_rows·c0 + (r0 + r)·m + (t − c0)]`, token `t` in group `g` —
    /// each group's block what `transpose::ds41_rows_to_tokens` copies of
    /// its own launch. One thread per value.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            src.len() >= total_rows * tokens,
            dst.len() >= rows * tokens,
            total_rows >= r0 + rows,
            lead >= 1,
            lead <= 8,
            lead <= tokens
        )
    )]
    pub fn ds41_groups_to_tokens(
        src: &[f32],
        total_rows: u32,
        r0: u32,
        rows: u32,
        lead: u32,
        tokens: u32,
        mut dst: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let (rows, lead, tokens) = (rows as usize, lead as usize, tokens as usize);
        if i >= rows * tokens {
            return;
        }
        let (t, r) = (i / rows, i % rows);
        let (c0, m) = col_group(col_group_of(t, lead), lead, tokens);
        let at = total_rows as usize * c0 + (r0 as usize + r) * m + (t - c0);
        // SAFETY: t < tokens puts t in a group with c0 <= t < c0 + m, so at <
        // total_rows·(c0 + m) <= total_rows·tokens <= src.len() (r0 + r <
        // total_rows); i < rows·tokens <= dst.len(), and thread i is dst[i]'s
        // only writer.
        unsafe {
            *dst.get_unchecked_mut(i) = *src.get_unchecked(at);
        }
    }

    /// The shared expert's gate·up·SwiGLU for one token, Q3_K weights: row
    /// `r` (a warp per row) dots gate row `r` and up row `r` against the one
    /// q8_1 column, reduces each with the warp tree and stores
    /// `h[r] = swiglu_clamp(g, u, limit)` from lane 0.
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
            4 * wg.len() >= n_rows * 110 * n_sb,
            4 * wu.len() >= n_rows * 110 * n_sb,
            q.len() >= 64 * iters,
            d8.len() >= 2 * n_sb,
            h.len() >= n_rows
        )
    )]
    pub fn ds41_shexp_gate_up_q3k(
        wg: &[u32],
        wu: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        n_sb: u32,
        iters: u32,
        limit: f32,
        mut h: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let fg = q3k_row_dot(wg, q, d8, n_sb as usize, iters, row, 0, 1, lane);
        let fu = q3k_row_dot(wu, q, d8, n_sb as usize, iters, row, 0, 1, lane);
        let g = warp::reduce_sum_f32(fg[0]);
        let u = warp::reduce_sum_f32(fu[0]);
        if lane == 0 {
            let v = swiglu_clamp(g, u, limit);
            // SAFETY: row < n_rows <= h.len(); lane 0 of the row's warp is
            // its only writer.
            unsafe { *h.get_unchecked_mut(row) = v };
        }
    }

    /// `ds41_shexp_gate_up_q3k` for `m_cols` (1..=8) tokens: both Q3_K dots
    /// against each of the m q8_1 columns, token-major output
    /// `h[t·n_rows + r]`; token t bit for bit the one-token launch on its
    /// column.
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
            4 * wg.len() >= n_rows * 110 * n_sb,
            4 * wu.len() >= n_rows * 110 * n_sb,
            q.len() >= m_cols * 64 * iters,
            d8.len() >= m_cols * 2 * n_sb,
            h.len() >= m_cols * n_rows,
            m_cols >= 1,
            m_cols <= 8
        )
    )]
    pub fn ds41_shexp_gate_up_q3k_mcol(
        wg: &[u32],
        wu: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        m_cols: u32,
        n_sb: u32,
        iters: u32,
        limit: f32,
        mut h: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let m = m_cols as usize;
        let fg = q3k_row_dot(wg, q, d8, n_sb as usize, iters, row, 0, m, lane);
        let fu = q3k_row_dot(wu, q, d8, n_sb as usize, iters, row, 0, m, lane);
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
            // SAFETY: slots t·n_rows + row for t < m are inside h (h.len()
            // >= m·n_rows) and belong to this row's warp alone.
            unsafe { store_cols(&mut h, row, n_rows as usize, m, v) };
        }
    }
}

/// A dense projection's resident weight, by its file format.
#[derive(Clone, Copy)]
pub enum Dense<'w> {
    Q8_0 {
        qs: &'w DeviceTensor<u32>,
        d: &'w DeviceTensor<u16>,
    },
    Q3K(&'w DeviceTensor<u32>),
    Q4K(&'w DeviceTensor<u32>),
    Q5K(&'w DeviceTensor<u32>),
}

impl<'w> Dense<'w> {
    /// `name`'s resident weight, refused unless it projects `k` values onto
    /// `rows` rows in a format this module runs.
    pub fn of(w: &'w Weights, name: &str, k: usize, rows: usize) -> Result<Dense<'w>, GpuError> {
        let shape = |got_k: usize, got_rows: usize| GpuError::Shape {
            what: WHAT,
            detail: format!("{name} is {got_rows} rows of {got_k} values, want {rows} of {k}"),
        };
        match w.get(name) {
            Some(DevWeight::Q8_0 { qs, d, k: wk }) => {
                if *wk != k || d.rows() != rows {
                    return Err(shape(*wk, d.rows()));
                }
                Ok(Dense::Q8_0 { qs, d })
            }
            Some(DevWeight::KQuant { ty, w: t, k: wk }) => {
                if *wk != k || t.rows() != rows {
                    return Err(shape(*wk, t.rows()));
                }
                match ty {
                    GgmlType::Q3_K => Ok(Dense::Q3K(t)),
                    GgmlType::Q4_K => Ok(Dense::Q4K(t)),
                    GgmlType::Q5_K => Ok(Dense::Q5K(t)),
                    _ => Err(GpuError::Tensor {
                        what: WHAT,
                        name: name.to_string(),
                        need: "Q8_0, Q3_K, Q4_K or Q5_K",
                    }),
                }
            }
            found => Err(GpuError::Tensor {
                what: WHAT,
                name: name.to_string(),
                need: if found.is_some() {
                    "Q8_0, Q3_K, Q4_K or Q5_K"
                } else {
                    "resident"
                },
            }),
        }
    }

    /// Whether the projection reads its input in q8_1.
    #[must_use]
    pub fn reads_q8_1(&self) -> bool {
        matches!(self, Dense::Q3K(_) | Dense::Q4K(_))
    }
}

/// The loaded module. Owns no context and no stream — every enqueue takes
/// the engine stream, so the launches order with the rest of the step and
/// are capturable.
pub struct DenseKernels {
    module: dense_kernels::LoadedModule,
}

impl DenseKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<DenseKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { dense_kernels::load(ctx)? };
        Ok(DenseKernels { module })
    }

    /// Enqueue `y = W · x` for one token: `x` its f32 input, `act` that
    /// input's q8_1 form when `d` reads one ([`Dense::reads_q8_1`]; the
    /// caller quantized it), `y` one f32 per row. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue(
        &self,
        gpu: &Gpu,
        d: Dense<'_>,
        x: &DeviceBuffer<f32>,
        act: Option<&Q8Act>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        self.enqueue_m(gpu, d, x, act, 1, y)
    }

    /// Enqueue `y = W · x` for `m` (1..=8) tokens: `x` their f32 inputs, `m`
    /// columns of the projection's K values, `act` those inputs' q8_1 form
    /// (`m` columns) when `d` reads one, `y` row-major, `y[r·m + c]`.
    /// Column c is the one-token enqueue on token c bit for bit, whatever
    /// the format. Asynchronous, allocation-free, capturable.
    pub fn enqueue_m(
        &self,
        gpu: &Gpu,
        d: Dense<'_>,
        x: &DeviceBuffer<f32>,
        act: Option<&Q8Act>,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let act = || match act {
            Some(a) if a.m() == m => Ok(a),
            Some(a) => Err(GpuError::Shape {
                what: WHAT,
                detail: format!("a q8_1 input of {} columns for {m} tokens", a.m()),
            }),
            None => Err(GpuError::Shape {
                what: WHAT,
                detail: "a K-quant projection without its input's q8_1 form".to_string(),
            }),
        };
        match d {
            Dense::Q8_0 { qs, d } => gpu.q8f32().enqueue_q8_0_gemv(stream, qs, d, x, m, y),
            Dense::Q3K(w) => gpu.enqueue_gemv_q3k(w, act()?, y),
            Dense::Q4K(w) => gpu.enqueue_gemv_q4k(w, act()?, y),
            Dense::Q5K(w) => self.enqueue_q5k(stream, w, x, m, y),
        }
    }

    /// Enqueue the Q5_K f32-activation gemv of `w` (rows of whole
    /// super-blocks, word aligned) over the `m` columns of `x`, `y[r·m + c]`:
    /// the one-column kernel at `m` 1, its m-column twin otherwise.
    fn enqueue_q5k(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        x: &DeviceBuffer<f32>,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "DenseKernels::enqueue_q5k";
        let n_rows = w.rows();
        if w.cols() == 0 || !w.cols().is_multiple_of(Q5K_WORDS) {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "Q5_K rows of {} words: not whole super-blocks of {Q5K_WORDS}",
                    w.cols()
                ),
            });
        }
        let n_sb = w.cols() / Q5K_WORDS;
        if !(1..=8).contains(&m) || x.len() < m * 256 * n_sb || y.len() < n_rows * m {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "m {m} (1..=8), x.len() {} < {} or y.len() {} < {}",
                    x.len(),
                    m * 256 * n_sb,
                    y.len(),
                    n_rows * m
                ),
            });
        }
        let grid = launch_u32(what, "grid", n_rows.div_ceil(8))?;
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let cfg = LaunchConfig1D::new(grid, BLOCK, 0);
        if m == 1 {
            let prep = self.module.prepare_ds41_q5k_gemv_f32(cfg)?;
            self.module
                .ds41_q5k_gemv_f32(stream, &prep, w.buf(), x, n_rows, n_sb, y)?;
        } else {
            let m = launch_u32(what, "m", m)?;
            let prep = self.module.prepare_ds41_q5k_gemv_f32_mcol(cfg)?;
            self.module
                .ds41_q5k_gemv_f32_mcol(stream, &prep, w.buf(), x, n_rows, n_sb, m, y)?;
        }
        Ok(())
    }

    /// Enqueue the block diagonal of Q3_K weight `w` (`w.rows()` rows of
    /// `act.n_sb()` super-blocks): row `r` against column `r /
    /// rows_per_head` of `act`, which holds one column per group, the groups
    /// covering every row. `y` takes one f32 per row. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_q3k_heads(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        rows_per_head: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "DenseKernels::enqueue_q3k_heads";
        let (n_rows, groups, n_sb) = (w.rows(), act.m(), act.n_sb());
        if rows_per_head == 0 || n_rows != groups * rows_per_head {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "{n_rows} rows are not {groups} groups of {rows_per_head}: one q8_1 column \
                     per group"
                ),
            });
        }
        kquant_rows(what, w, 110, n_sb)?;
        if y.len() < n_rows {
            return Err(GpuError::Shape {
                what,
                detail: format!("y.len() {} < {n_rows}", y.len()),
            });
        }
        let grid = launch_u32(what, "grid", n_rows.div_ceil(8))?;
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let rows_per_head = launch_u32(what, "rows_per_head", rows_per_head)?;
        let groups = launch_u32(what, "groups", groups)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_ds41_q3k_gemv_heads(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.ds41_q3k_gemv_heads(
            stream,
            &prep,
            w.buf(),
            act.q3(),
            act.d8(),
            n_rows,
            rows_per_head,
            groups,
            n_sb,
            n_sb.div_ceil(2),
            y,
        )?;
        Ok(())
    }

    /// Enqueue the block diagonal of Q3_K weight `a.w` for `a.m` (1..=8)
    /// tokens ([`Q3kHeadsMcolArgs`]): row `r` of group `g = r /
    /// rows_per_head` against column `t·groups + g` of the activation for
    /// every token t, token-major into `a.y`. Token t is
    /// [`DenseKernels::enqueue_q3k_heads`] on that token's `groups` columns,
    /// bit for bit; one token is that kernel. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_q3k_heads_mcol(
        &self,
        stream: &CudaStream,
        a: Q3kHeadsMcolArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "DenseKernels::enqueue_q3k_heads_mcol";
        let (n_rows, groups, n_sb, m) = (a.w.rows(), a.groups, a.n_sb, a.m);
        if a.rows_per_head == 0 || n_rows != groups * a.rows_per_head || !(1..=8).contains(&m) {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "{n_rows} rows are not {groups} groups of {}, or m {m} is not in 1..=8",
                    a.rows_per_head
                ),
            });
        }
        kquant_rows(what, a.w, 110, n_sb)?;
        let cols = m * groups;
        if a.q3.len() < cols * 64 * n_sb.div_ceil(2)
            || a.d8.len() < cols * 2 * n_sb
            || a.y.len() < m * n_rows
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "{cols} q8_1 columns of {n_sb} super-blocks want {} code and {} scale \
                     slots, got {} and {}; y wants {}, got {}",
                    cols * 64 * n_sb.div_ceil(2),
                    cols * 2 * n_sb,
                    a.q3.len(),
                    a.d8.len(),
                    m * n_rows,
                    a.y.len()
                ),
            });
        }
        let grid = launch_u32(what, "grid", n_rows.div_ceil(8))?;
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let rows_per_head = launch_u32(what, "rows_per_head", a.rows_per_head)?;
        let groups = launch_u32(what, "groups", groups)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let cfg = LaunchConfig1D::new(grid, BLOCK, 0);
        if m == 1 {
            let prep = self.module.prepare_ds41_q3k_gemv_heads(cfg)?;
            self.module.ds41_q3k_gemv_heads(
                stream,
                &prep,
                a.w.buf(),
                a.q3,
                a.d8,
                n_rows,
                rows_per_head,
                groups,
                n_sb,
                n_sb.div_ceil(2),
                a.y,
            )?;
        } else {
            let m = launch_u32(what, "m", m)?;
            let prep = self.module.prepare_ds41_q3k_gemv_heads_mcol(cfg)?;
            self.module.ds41_q3k_gemv_heads_mcol(
                stream,
                &prep,
                a.w.buf(),
                a.q3,
                a.d8,
                n_rows,
                rows_per_head,
                groups,
                m,
                n_sb,
                n_sb.div_ceil(2),
                a.y,
            )?;
        }
        Ok(())
    }

    /// Enqueue the block diagonal of Q3_K weight `a.w` over a prompt pass's
    /// token groups in one launch ([`Q3kHeadsGroupsArgs`]): each group's
    /// tokens [`DenseKernels::enqueue_q3k_heads_mcol`] on that group's
    /// columns alone, bit for bit, token-major into `a.y` from the groups'
    /// first token. Asynchronous, allocation-free, capturable.
    pub fn enqueue_q3k_heads_groups(
        &self,
        stream: &CudaStream,
        a: Q3kHeadsGroupsArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "DenseKernels::enqueue_q3k_heads_groups";
        let (n_rows, groups, n_sb) = (a.w.rows(), a.groups, a.act.n_sb());
        let (col0, lead, tokens) = (a.tokens.col0(), a.tokens.lead(), a.tokens.cols());
        if a.rows_per_head == 0 || n_rows != groups * a.rows_per_head {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "{n_rows} rows are not {groups} groups of {}",
                    a.rows_per_head
                ),
            });
        }
        kquant_rows(what, a.w, 110, n_sb)?;
        if (col0 + tokens) * groups > a.act.m() || a.y.len() < tokens * n_rows {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "tokens {col0}..{} of {groups} columns each in an activation of {}; y wants \
                     {}, got {}",
                    col0 + tokens,
                    a.act.m(),
                    tokens * n_rows,
                    a.y.len()
                ),
            });
        }
        let grid = launch_u32(what, "grid", a.tokens.count() * n_rows.div_ceil(8))?;
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let rows_per_head = launch_u32(what, "rows_per_head", a.rows_per_head)?;
        let groups = launch_u32(what, "groups", groups)?;
        let col0 = launch_u32(what, "col0", col0)?;
        let lead = launch_u32(what, "lead", lead)?;
        let tokens = launch_u32(what, "tokens", tokens)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_ds41_q3k_gemv_heads_groups(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.ds41_q3k_gemv_heads_groups(
            stream,
            &prep,
            a.w.buf(),
            a.act.q3(),
            a.act.d8(),
            n_rows,
            rows_per_head,
            groups,
            col0,
            lead,
            tokens,
            n_sb,
            n_sb.div_ceil(2),
            a.y,
        )?;
        Ok(())
    }

    /// Enqueue the token-major copy of `part`'s rows of a grouped launch's
    /// output `src` over the tokens of `groups` (laid out as
    /// `Gpu::enqueue_gemv_q3k_groups` lays it) into `dst`, `part.rows`
    /// values a token: each group's tokens what the transpose of its own
    /// launch's output copies. Refused before the launch when the rows pass
    /// the output's or a buffer is short. Asynchronous, allocation-free.
    pub fn enqueue_groups_to_tokens(
        &self,
        stream: &CudaStream,
        src: &DeviceBuffer<f32>,
        part: RowsPart,
        groups: ColGroups,
        dst: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "DenseKernels::enqueue_groups_to_tokens";
        let RowsPart {
            total_rows,
            r0,
            rows,
        } = part;
        let tokens = groups.cols();
        if rows == 0
            || r0 + rows > total_rows
            || src.len() < total_rows * tokens
            || dst.len() < rows * tokens
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "rows {r0}..{} of {total_rows} over {tokens} tokens from {} values into {}",
                    r0 + rows,
                    src.len(),
                    dst.len()
                ),
            });
        }
        let grid = launch_u32(what, "grid", (rows * tokens).div_ceil(BLOCK as usize))?;
        let total_rows = launch_u32(what, "total_rows", total_rows)?;
        let r0 = launch_u32(what, "r0", r0)?;
        let rows = launch_u32(what, "rows", rows)?;
        let lead = launch_u32(what, "lead", groups.lead())?;
        let tokens = launch_u32(what, "tokens", tokens)?;
        let prep = self
            .module
            .prepare_ds41_groups_to_tokens(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module
            .ds41_groups_to_tokens(stream, &prep, src, total_rows, r0, rows, lead, tokens, dst)?;
        Ok(())
    }

    /// Enqueue the shared expert's gate·up·SwiGLU for `act.m()` (1..=8)
    /// tokens from Q3_K `gate` and `up` (the same rows of `act.n_sb()`
    /// super-blocks) and the tokens' q8_1 columns `act`; `h` takes one f32
    /// per row and token, token-major (`h[t·rows + r]`). At one token this
    /// is the one-token kernel; at more, its m-column twin, token t bit for
    /// bit the one-token launch on its column. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_shexp_gate_up_q3k(
        &self,
        stream: &CudaStream,
        gate: &DeviceTensor<u32>,
        up: &DeviceTensor<u32>,
        act: &Q8Act,
        limit: f32,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "DenseKernels::enqueue_shexp_gate_up_q3k";
        let (n_rows, n_sb, m) = (gate.rows(), act.n_sb(), act.m());
        if up.rows() != n_rows || h.len() < m * n_rows {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "gate and up of one height and h of m of it: m {m}, gate {n_rows} rows, \
                     up {}, h.len() {}",
                    up.rows(),
                    h.len()
                ),
            });
        }
        kquant_rows(what, gate, 110, n_sb)?;
        kquant_rows(what, up, 110, n_sb)?;
        let grid = launch_u32(what, "grid", n_rows.div_ceil(8))?;
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let cfg = LaunchConfig1D::new(grid, BLOCK, 0);
        if m == 1 {
            let prep = self.module.prepare_ds41_shexp_gate_up_q3k(cfg)?;
            self.module.ds41_shexp_gate_up_q3k(
                stream,
                &prep,
                gate.buf(),
                up.buf(),
                act.q3(),
                act.d8(),
                n_rows,
                n_sb,
                n_sb.div_ceil(2),
                limit,
                h,
            )?;
        } else {
            let m = launch_u32(what, "m", m)?;
            let prep = self.module.prepare_ds41_shexp_gate_up_q3k_mcol(cfg)?;
            self.module.ds41_shexp_gate_up_q3k_mcol(
                stream,
                &prep,
                gate.buf(),
                up.buf(),
                act.q3(),
                act.d8(),
                n_rows,
                m,
                n_sb,
                n_sb.div_ceil(2),
                limit,
                h,
            )?;
        }
        Ok(())
    }
}

/// Arguments of [`DenseKernels::enqueue_q3k_heads_mcol`].
pub struct Q3kHeadsMcolArgs<'a> {
    /// The Q3_K block-diagonal weight, `groups · rows_per_head` rows.
    pub w: &'a DeviceTensor<u32>,
    /// The activation's codes, `m · groups` q8_1 columns of `n_sb`
    /// super-blocks in [`Q8Act::q3`]'s per-column layout, token t's group g
    /// at column `t·groups + g`. The quantizer is column-local, so these are
    /// the bytes of the m one-token activations laid end to end.
    pub q3: &'a DeviceBuffer<u64>,
    /// The activation's block scales, in [`Q8Act::d8`]'s per-column layout.
    pub d8: &'a DeviceBuffer<f32>,
    pub n_sb: usize,
    pub groups: usize,
    pub rows_per_head: usize,
    /// Tokens, 1..=8.
    pub m: usize,
    /// `m · rows` outputs, token-major.
    pub y: &'a mut DeviceBuffer<f32>,
}

/// Arguments of [`DenseKernels::enqueue_q3k_heads_groups`].
pub struct Q3kHeadsGroupsArgs<'a> {
    /// The Q3_K block-diagonal weight, `groups · rows_per_head` rows.
    pub w: &'a DeviceTensor<u32>,
    /// The heads in q8_1, `groups` columns a token — token t's group g at
    /// column `t·groups + g`, [`Q3kHeadsMcolArgs::q3`]'s layout.
    pub act: &'a Q8Act,
    pub groups: usize,
    pub rows_per_head: usize,
    /// The tokens, counted in the activation's tokens, and their groups.
    pub tokens: ColGroups,
    /// `tokens.cols() · rows` outputs, token-major from the groups' first
    /// token.
    pub y: &'a mut DeviceBuffer<f32>,
}

/// Rows `r0 .. r0 + rows` of a projection of `total_rows` rows: a part of a
/// joined launch's output.
#[derive(Clone, Copy, Debug)]
pub struct RowsPart {
    pub total_rows: usize,
    pub r0: usize,
    pub rows: usize,
}

/// Refuse a K-quant tensor whose rows are not `n_sb` super-blocks of
/// `sb_bytes`: its flat word stream is the upload's (`CardFormat::KQuant`,
/// zero-padded at its end to whole words per row).
fn kquant_rows(
    what: &'static str,
    w: &DeviceTensor<u32>,
    sb_bytes: usize,
    n_sb: usize,
) -> Result<(), GpuError> {
    let rows = w.rows();
    let words = (rows * sb_bytes * n_sb).div_ceil(4).div_ceil(rows.max(1));
    if rows == 0 || n_sb == 0 || w.cols() != words {
        return Err(GpuError::Shape {
            what,
            detail: format!(
                "{rows} rows of {} words: not rows of {n_sb} super-blocks of {sb_bytes} bytes \
                 ({words} words a row)",
                w.cols()
            ),
        });
    }
    Ok(())
}
