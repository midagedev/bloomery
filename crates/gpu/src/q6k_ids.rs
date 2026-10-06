//! Q6_K gemv over a row map of one matrix: the MTP draft's borrowed head
//! over the target's Q6_K `output.weight` (`arch::qwen3moe::mtp38`), the
//! head-form sibling of `q8f32`'s `q8_0_gemv_ids` entries. Launch row `r`
//! dots the matrix's row `ids[r]` — or row `r` itself when no map is given
//! (the full head) — against every q8_1 activation column `c`, writing
//! `y[r·m + c]`, the layout `enqueue_argmax_p_rows_fault` reads.
//!
//! The per-row body is `q6k_gemv_sel`'s verbatim — the same byte-window
//! loads with the 16-bit funnel for a super-block at 2 mod 4, the same dp4a
//! chain, the same `f0 += a · (e8 · d · sc)` accumulation and the same warp
//! reduction — so row `r` equals `enqueue_gemv_q6k` over the gathered rows
//! bit for bit, every column of an `m`-column launch the `m = 1` launch of
//! that column. K must be a multiple of 512 (an even super-block count), so
//! every row starts on a word boundary; the launcher refuses an odd count
//! as `enqueue_gemv_q6k` refuses it.
//!
//! An id at or past the matrix's rows raises [`FaultSite::TokenId`] before
//! the row's first weight load (warp-uniform) and writes the row's outputs
//! NaN. Never clamped.

use crate::cores::{funnel16, half_to_f32, q6k_chain, q6k_dequant, q6k_sub_scale};
use crate::fault::{FaultSink, FaultSite};
use crate::tensor::{DeviceTensor, Q8Act};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

#[cuda_module]
mod q6k_ids_kernels {
    use super::*;

    /// The mapped Q6_K gemv at one column: one warp per output row, eight
    /// rows per 256-thread block; row `r` stores `y[r]` from weight row
    /// `map(r)` (`ids[r]` when `mapped` is 1, else `r`) — `210 · n_sb` bytes
    /// at byte `map(r) · 210 · n_sb` of the matrix — against q8_1 column 0
    /// in the q6 permutation. An id at or past `n_matrix` raises
    /// [`FaultSite::TokenId`] and stores NaN, before the row's first weight
    /// load; the raise and the store are warp-uniform in `lane == 0`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the flat arguments (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_matrix * 210 * n_sb,
            q.len() >= 128 * iters,
            d8.len() >= 2 * n_sb,
            ids.len() >= mapped * n_rows,
            y.len() >= n_rows
        )
    )]
    pub fn q6k_gemv_ids(
        w: &[u32],
        q: &[u32],
        d8: &[f32],
        ids: &[u32],
        n_rows: u32,
        n_matrix: u32,
        n_sb: u32,
        iters: u32,
        mapped: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let id = if mapped != 0 {
            // SAFETY: row < n_rows and mapped = 1 puts n_rows <= ids.len()
            // by the launch contract; the load is warp-uniform (all 32
            // lanes share row).
            unsafe { *ids.get_unchecked(row) }
        } else {
            row as u32
        };
        let lane = warp::lane_id() as usize;
        if id >= n_matrix {
            if lane == 0 {
                fault.raise(FaultSite::TokenId);
                // SAFETY: row < n_rows <= y.len() by the launch contract; lane
                // 0 of the row's warp alone writes y[row].
                unsafe { *y.get_unchecked_mut(row) = f32::NAN };
            }
            return;
        }
        let row_abs = id as usize;
        let n_sb = n_sb as usize;
        let row_bytes = 210 * n_sb;

        let w16 = lane & 15;
        let half = lane >> 4;
        let qlok = 16 * (w16 & 3);
        let qhok = 16 * (w16 & 1);
        let nib_sh = ((w16 >> 2) & 1) as u32 * 4;
        let hib_sh = 2 * ((w16 >> 1) as u32 & 3);

        let mut f0 = 0.0f32;
        let mut it: u32 = 0;
        while it < iters {
            let sbp = ((it << 1) | half as u32) as usize;
            if sbp < n_sb {
                let base = row_abs * row_bytes + sbp * 210;
                let par = (base >> 1) & 1;

                let lk = (base + 64 * (w16 >> 3) + qlok) >> 2;
                // SAFETY: id < n_matrix, so the window ends inside the
                // matrix's words (launch contract: 4·w.len() covers every
                // row's 210·n_sb bytes; the floored word index of a window
                // stays inside the row's words, as in q6k_gemv).
                let (l0, l1, l2, l3, l4) = unsafe {
                    (
                        *w.get_unchecked(lk),
                        *w.get_unchecked(lk + 1),
                        *w.get_unchecked(lk + 2),
                        *w.get_unchecked(lk + 3),
                        *w.get_unchecked(lk + 4),
                    )
                };
                let ql0 = if par == 0 { l0 } else { funnel16(l0, l1) };
                let ql1 = if par == 0 { l1 } else { funnel16(l1, l2) };
                let ql2 = if par == 0 { l2 } else { funnel16(l2, l3) };
                let ql3 = if par == 0 { l3 } else { funnel16(l3, l4) };

                let hk = (base + 128 + 32 * (w16 >> 3) + qhok) >> 2;
                // SAFETY: the same row bounds as the ql window.
                let (h0, h1, h2, h3, h4) = unsafe {
                    (
                        *w.get_unchecked(hk),
                        *w.get_unchecked(hk + 1),
                        *w.get_unchecked(hk + 2),
                        *w.get_unchecked(hk + 3),
                        *w.get_unchecked(hk + 4),
                    )
                };
                let qh0 = if par == 0 { h0 } else { funnel16(h0, h1) };
                let qh1 = if par == 0 { h1 } else { funnel16(h1, h2) };
                let qh2 = if par == 0 { h2 } else { funnel16(h2, h3) };
                let qh3 = if par == 0 { h3 } else { funnel16(h3, h4) };

                let ak = (base + 192) >> 2;
                // SAFETY: ak + 4 is at most the matrix's final (possibly
                // zero-padded) word for the last super-block of the last row;
                // the launch contract bounds 4·w.len().
                let (a0, a1, a2, a3, a4) = unsafe {
                    (
                        *w.get_unchecked(ak),
                        *w.get_unchecked(ak + 1),
                        *w.get_unchecked(ak + 2),
                        *w.get_unchecked(ak + 3),
                        *w.get_unchecked(ak + 4),
                    )
                };
                let (sw0, sw1, sw2, sw3, d_bits) = if par == 0 {
                    (a0, a1, a2, a3, (a4 & 0xffff) as u16)
                } else {
                    (
                        funnel16(a0, a1),
                        funnel16(a1, a2),
                        funnel16(a2, a3),
                        funnel16(a3, a4),
                        (a4 >> 16) as u16,
                    )
                };
                let sc = q6k_sub_scale(&[sw0, sw1, sw2, sw3], w16);
                let drow = half_to_f32(d_bits);
                let vi = [
                    q6k_dequant(ql0, qh0, nib_sh, hib_sh),
                    q6k_dequant(ql1, qh1, nib_sh, hib_sh),
                    q6k_dequant(ql2, qh2, nib_sh, hib_sh),
                    q6k_dequant(ql3, qh3, nib_sh, hib_sh),
                ];
                let qb = 128 * it as usize + lane;
                let d8b = 2 * sbp + (w16 >> 3);
                // SAFETY: qb + 99 < 128·iters — this lane's four q8 words are
                // inside column 0's 128·iters words (the permutation's group
                // bound); d8b < 2·n_sb by the sbp guard.
                let (q0, q1, q2, q3, e0) = unsafe {
                    (
                        *q.get_unchecked(qb),
                        *q.get_unchecked(qb + 32),
                        *q.get_unchecked(qb + 64),
                        *q.get_unchecked(qb + 96),
                        *d8.get_unchecked(d8b),
                    )
                };
                let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                f0 += (a as f32) * (e0 * drow * sc as f32);
            }
            it += 1;
        }
        let s0 = warp::reduce_sum_f32(f0);
        if lane == 0 {
            // SAFETY: row < n_rows <= y.len() by the launch contract; only
            // lane 0 of the warp writes y[row].
            unsafe {
                *y.get_unchecked_mut(row) = s0;
            }
        }
    }

    /// The mapped Q6_K gemv at `m_cols` columns (1 < m_cols <= 8): `q6k_gemv`
    /// over the mapped rows — every column's chain, accumulation and warp
    /// reduction that kernel's, so column c of the launch is the `m = 1`
    /// entry of column c bit for bit — with lane 0 storing column c at
    /// `y[r·m_cols + c]`. The id load and its refusal are as `q6k_gemv_ids`;
    /// a refused row stores NaN in its columns.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the flat arguments (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_matrix * 210 * n_sb,
            q.len() >= m_cols * 128 * iters,
            d8.len() >= m_cols * 2 * n_sb,
            ids.len() >= mapped * n_rows,
            y.len() >= n_rows * m_cols
        )
    )]
    pub fn q6k_gemv_ids_mcol(
        w: &[u32],
        q: &[u32],
        d8: &[f32],
        ids: &[u32],
        n_rows: u32,
        n_matrix: u32,
        m_cols: u32,
        n_sb: u32,
        iters: u32,
        mapped: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let id = if mapped != 0 {
            // SAFETY: row < n_rows and mapped = 1 puts n_rows <= ids.len()
            // by the launch contract; the load is warp-uniform (all 32
            // lanes share row).
            unsafe { *ids.get_unchecked(row) }
        } else {
            row as u32
        };
        let lane = warp::lane_id() as usize;
        if id >= n_matrix {
            if lane == 0 {
                fault.raise(FaultSite::TokenId);
                // SAFETY: the slots row*m_cols + c, c < m_cols <= 8, lie in
                // row*m_cols .. (row+1)*m_cols <= n_rows*m_cols <= y.len(),
                // the launch contract's bound; only lane 0 of the row's warp
                // writes them.
                unsafe {
                    let b = row * m_cols as usize;
                    *y.get_unchecked_mut(b) = f32::NAN;
                    if m_cols > 1 {
                        *y.get_unchecked_mut(b + 1) = f32::NAN;
                    }
                    if m_cols > 2 {
                        *y.get_unchecked_mut(b + 2) = f32::NAN;
                    }
                    if m_cols > 3 {
                        *y.get_unchecked_mut(b + 3) = f32::NAN;
                    }
                    if m_cols > 4 {
                        *y.get_unchecked_mut(b + 4) = f32::NAN;
                    }
                    if m_cols > 5 {
                        *y.get_unchecked_mut(b + 5) = f32::NAN;
                    }
                    if m_cols > 6 {
                        *y.get_unchecked_mut(b + 6) = f32::NAN;
                    }
                    if m_cols > 7 {
                        *y.get_unchecked_mut(b + 7) = f32::NAN;
                    }
                }
            }
            return;
        }
        let row_abs = id as usize;
        let m = m_cols as usize;
        let n_sb = n_sb as usize;
        let row_bytes = 210 * n_sb;
        let q_col = 128 * iters as usize;
        let d8_col = 2 * n_sb;

        let w16 = lane & 15;
        let half = lane >> 4;
        let qlok = 16 * (w16 & 3);
        let qhok = 16 * (w16 & 1);
        let nib_sh = ((w16 >> 2) & 1) as u32 * 4;
        let hib_sh = 2 * ((w16 >> 1) as u32 & 3);

        let mut f0 = 0.0f32;
        let mut f1 = 0.0f32;
        let mut f2 = 0.0f32;
        let mut f3 = 0.0f32;
        let mut f4 = 0.0f32;
        let mut f5 = 0.0f32;
        let mut f6 = 0.0f32;
        let mut f7 = 0.0f32;

        let mut it: u32 = 0;
        while it < iters {
            let sbp = ((it << 1) | half as u32) as usize;
            if sbp < n_sb {
                let base = row_abs * row_bytes + sbp * 210;
                let par = (base >> 1) & 1;

                let lk = (base + 64 * (w16 >> 3) + qlok) >> 2;
                // SAFETY: id < n_matrix, so the window ends inside the
                // matrix's words (launch contract: 4·w.len() covers every
                // row's 210·n_sb bytes; the floored word index of a window
                // stays inside the row's words, as in q6k_gemv).
                let (l0, l1, l2, l3, l4) = unsafe {
                    (
                        *w.get_unchecked(lk),
                        *w.get_unchecked(lk + 1),
                        *w.get_unchecked(lk + 2),
                        *w.get_unchecked(lk + 3),
                        *w.get_unchecked(lk + 4),
                    )
                };
                let ql0 = if par == 0 { l0 } else { funnel16(l0, l1) };
                let ql1 = if par == 0 { l1 } else { funnel16(l1, l2) };
                let ql2 = if par == 0 { l2 } else { funnel16(l2, l3) };
                let ql3 = if par == 0 { l3 } else { funnel16(l3, l4) };

                let hk = (base + 128 + 32 * (w16 >> 3) + qhok) >> 2;
                // SAFETY: the same row bounds as the ql window.
                let (h0, h1, h2, h3, h4) = unsafe {
                    (
                        *w.get_unchecked(hk),
                        *w.get_unchecked(hk + 1),
                        *w.get_unchecked(hk + 2),
                        *w.get_unchecked(hk + 3),
                        *w.get_unchecked(hk + 4),
                    )
                };
                let qh0 = if par == 0 { h0 } else { funnel16(h0, h1) };
                let qh1 = if par == 0 { h1 } else { funnel16(h1, h2) };
                let qh2 = if par == 0 { h2 } else { funnel16(h2, h3) };
                let qh3 = if par == 0 { h3 } else { funnel16(h3, h4) };

                let ak = (base + 192) >> 2;
                // SAFETY: ak + 4 is at most the matrix's final (possibly
                // zero-padded) word for the last super-block of the last row;
                // the launch contract bounds 4·w.len().
                let (a0, a1, a2, a3, a4) = unsafe {
                    (
                        *w.get_unchecked(ak),
                        *w.get_unchecked(ak + 1),
                        *w.get_unchecked(ak + 2),
                        *w.get_unchecked(ak + 3),
                        *w.get_unchecked(ak + 4),
                    )
                };
                let (sw0, sw1, sw2, sw3, d_bits) = if par == 0 {
                    (a0, a1, a2, a3, (a4 & 0xffff) as u16)
                } else {
                    (
                        funnel16(a0, a1),
                        funnel16(a1, a2),
                        funnel16(a2, a3),
                        funnel16(a3, a4),
                        (a4 >> 16) as u16,
                    )
                };
                let sc = q6k_sub_scale(&[sw0, sw1, sw2, sw3], w16);
                let drow = half_to_f32(d_bits);
                let vi = [
                    q6k_dequant(ql0, qh0, nib_sh, hib_sh),
                    q6k_dequant(ql1, qh1, nib_sh, hib_sh),
                    q6k_dequant(ql2, qh2, nib_sh, hib_sh),
                    q6k_dequant(ql3, qh3, nib_sh, hib_sh),
                ];
                let qb = 128 * it as usize + lane;
                let d8b = 2 * sbp + (w16 >> 3);

                // Column 0 (always active): one dp4a chain, one FMA.
                {
                    // SAFETY: qb + 96 + 3 <= q_col − 1 — this lane's four q8
                    // words are inside column 0's q_col words (the
                    // permutation's group bound); d8b < d8_col by the sbp
                    // guard.
                    let (q0, q1, q2, q3, e0) = unsafe {
                        (
                            *q.get_unchecked(qb),
                            *q.get_unchecked(qb + 32),
                            *q.get_unchecked(qb + 64),
                            *q.get_unchecked(qb + 96),
                            *d8.get_unchecked(d8b),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f0 += (a as f32) * (e0 * drow * sc as f32);
                }
                // Columns 1..7, one launch-uniform guard per column so the
                // work scales with m. Column c reads q8 words at q_col·c +
                // qb (+32 per word) and block d8b + d8_col·c.
                // SAFETY: guard m > c means q.len() >= (c+1)·q_col >
                // q_col·c + qb + 99 and d8.len() >= (c+1)·d8_col >
                // d8b + d8_col·c, launch-uniform.
                if m > 1 {
                    let cb = q_col + qb;
                    let d8c = d8b + d8_col;
                    // SAFETY: m > 1, so q.len() >= (1+1)*q_col >
                    // 1*q_col + qb + 99 and d8.len() >= (1+1)*d8_col >
                    // d8b + 1*d8_col.
                    let (q0, q1, q2, q3, e1) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f1 += (a as f32) * (e1 * drow * sc as f32);
                }
                if m > 2 {
                    let cb = 2 * q_col + qb;
                    let d8c = d8b + 2 * d8_col;
                    // SAFETY: m > 2, so q.len() >= (2+1)*q_col >
                    // 2*q_col + qb + 99 and d8.len() >= (2+1)*d8_col >
                    // d8b + 2*d8_col.
                    let (q0, q1, q2, q3, e2) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f2 += (a as f32) * (e2 * drow * sc as f32);
                }
                if m > 3 {
                    let cb = 3 * q_col + qb;
                    let d8c = d8b + 3 * d8_col;
                    // SAFETY: m > 3, so q.len() >= (3+1)*q_col >
                    // 3*q_col + qb + 99 and d8.len() >= (3+1)*d8_col >
                    // d8b + 3*d8_col.
                    let (q0, q1, q2, q3, e3) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f3 += (a as f32) * (e3 * drow * sc as f32);
                }
                if m > 4 {
                    let cb = 4 * q_col + qb;
                    let d8c = d8b + 4 * d8_col;
                    // SAFETY: m > 4, so q.len() >= (4+1)*q_col >
                    // 4*q_col + qb + 99 and d8.len() >= (4+1)*d8_col >
                    // d8b + 4*d8_col.
                    let (q0, q1, q2, q3, e4) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f4 += (a as f32) * (e4 * drow * sc as f32);
                }
                if m > 5 {
                    let cb = 5 * q_col + qb;
                    let d8c = d8b + 5 * d8_col;
                    // SAFETY: m > 5, so q.len() >= (5+1)*q_col >
                    // 5*q_col + qb + 99 and d8.len() >= (5+1)*d8_col >
                    // d8b + 5*d8_col.
                    let (q0, q1, q2, q3, e5) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f5 += (a as f32) * (e5 * drow * sc as f32);
                }
                if m > 6 {
                    let cb = 6 * q_col + qb;
                    let d8c = d8b + 6 * d8_col;
                    // SAFETY: m > 6, so q.len() >= (6+1)*q_col >
                    // 6*q_col + qb + 99 and d8.len() >= (6+1)*d8_col >
                    // d8b + 6*d8_col.
                    let (q0, q1, q2, q3, e6) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f6 += (a as f32) * (e6 * drow * sc as f32);
                }
                if m > 7 {
                    let cb = 7 * q_col + qb;
                    let d8c = d8b + 7 * d8_col;
                    // SAFETY: m > 7, so q.len() >= (7+1)*q_col >
                    // 7*q_col + qb + 99 and d8.len() >= (7+1)*d8_col >
                    // d8b + 7*d8_col.
                    let (q0, q1, q2, q3, e7) = unsafe {
                        (
                            *q.get_unchecked(cb),
                            *q.get_unchecked(cb + 32),
                            *q.get_unchecked(cb + 64),
                            *q.get_unchecked(cb + 96),
                            *d8.get_unchecked(d8c),
                        )
                    };
                    let a = q6k_chain(&vi, &[q0, q1, q2, q3]);
                    f7 += (a as f32) * (e7 * drow * sc as f32);
                }
            }
            it += 1;
        }
        let s0 = warp::reduce_sum_f32(f0);
        let s1 = if m > 1 { warp::reduce_sum_f32(f1) } else { 0.0 };
        let s2 = if m > 2 { warp::reduce_sum_f32(f2) } else { 0.0 };
        let s3 = if m > 3 { warp::reduce_sum_f32(f3) } else { 0.0 };
        let s4 = if m > 4 { warp::reduce_sum_f32(f4) } else { 0.0 };
        let s5 = if m > 5 { warp::reduce_sum_f32(f5) } else { 0.0 };
        let s6 = if m > 6 { warp::reduce_sum_f32(f6) } else { 0.0 };
        let s7 = if m > 7 { warp::reduce_sum_f32(f7) } else { 0.0 };
        if lane == 0 {
            // SAFETY: only lane 0 of each warp writes the disjoint segment
            // y[row*m .. row*m+m]: store c is guarded by m > c, so exactly
            // the first m slots of the row are touched.
            unsafe {
                let b = row * m;
                *y.get_unchecked_mut(b) = s0;
                if m > 1 {
                    *y.get_unchecked_mut(b + 1) = s1;
                }
                if m > 2 {
                    *y.get_unchecked_mut(b + 2) = s2;
                }
                if m > 3 {
                    *y.get_unchecked_mut(b + 3) = s3;
                }
                if m > 4 {
                    *y.get_unchecked_mut(b + 4) = s4;
                }
                if m > 5 {
                    *y.get_unchecked_mut(b + 5) = s5;
                }
                if m > 6 {
                    *y.get_unchecked_mut(b + 6) = s6;
                }
                if m > 7 {
                    *y.get_unchecked_mut(b + 7) = s7;
                }
            }
        }
    }
}

/// [`Q6kIdsKernels::enqueue_gemv_q6k_ids`]'s arguments: the matrix's Q6_K
/// word plane, the q8_1 activation (the launch's `m` its columns), the row →
/// matrix-row map (`None`: row `r` reads row `r` — the full head) with a
/// stand-in word the entry takes in its place, the output rows, the logits
/// buffer (`rows · m` f32, row-major with `m` outputs a row) and the sink.
pub struct Q6kIdsArgs<'a> {
    pub w: &'a DeviceTensor<u32>,
    pub act: &'a Q8Act,
    pub map: Option<&'a DeviceBuffer<u32>>,
    pub no_map: &'a DeviceBuffer<u32>,
    pub rows: usize,
    pub m: usize,
    pub y: &'a mut DeviceBuffer<f32>,
    pub fault: FaultSink,
}

/// The loaded Q6_K row-map module. Owns no stream and no fault word: each
/// enqueue takes the engine stream and the caller's sink.
pub struct Q6kIdsKernels {
    module: q6k_ids_kernels::LoadedModule,
}

impl Q6kIdsKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Q6kIdsKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(q6k_ids_kernels, ctx)? };
        Ok(Q6kIdsKernels { module })
    }

    /// Enqueue the mapped Q6_K gemv (module doc): `y[r·m + c]` is the
    /// matrix's row `map(r)` dotted with activation column `c`. `m = 1` runs
    /// `q6k_gemv_ids`, `1 < m <= 8` `q6k_gemv_ids_mcol`, so column c of an
    /// m-column launch is the `m = 1` launch of column c bit for bit.
    /// Refused by name: `m` outside 1..=8 or past the activation's columns,
    /// `rows` zero or past the matrix's rows, a K not a multiple of 512
    /// (rows must start word-aligned), a weight of other than `210·n_sb/4`
    /// words a row or fewer words than its rows, a map shorter than `rows`
    /// and an empty stand-in, and `y` shorter than `rows·m`. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_gemv_q6k_ids(
        &self,
        stream: &CudaStream,
        a: Q6kIdsArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "enqueue_gemv_q6k_ids";
        let (n_matrix, m, rows) = (a.w.rows(), a.m, a.rows);
        let n_sb = a.act.n_sb();
        if !(1..=8).contains(&m) || m > a.act.m() {
            return Err(GpuError::shape(
                what,
                format!(
                    "need 1 <= m <= {} (the activation's columns), got m={m}",
                    a.act.m()
                ),
            ));
        }
        if rows == 0 || rows > n_matrix {
            return Err(GpuError::shape(
                what,
                format!("{rows} output rows of a matrix of {n_matrix} rows"),
            ));
        }
        if !a.act.k().is_multiple_of(512) {
            return Err(GpuError::shape(
                what,
                format!(
                    "K={} is not a multiple of 512: Q6_K rows must start word-aligned; repack \
                     rows at load time",
                    a.act.k()
                ),
            ));
        }
        if a.w.cols() != 210 * n_sb / 4 {
            return Err(GpuError::shape(
                what,
                format!(
                    "Q6_K rows are 210*{n_sb}/4 = {} words at K={}, got {}",
                    210 * n_sb / 4,
                    a.act.k(),
                    a.w.cols()
                ),
            ));
        }
        // The widest reach is the scales window of the last row's last
        // super-block: its fifth word holds the super-block's `d`, the
        // stream's last bytes, so the stream's own words suffice.
        let need = (n_matrix * 210 * n_sb).div_ceil(4);
        if a.w.buf().len() < need {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n_matrix} rows of 210*{n_sb} bytes need {need} words, got {}",
                    a.w.buf().len()
                ),
            ));
        }
        if a.map.is_some_and(|map| map.len() < rows) || a.no_map.is_empty() {
            return Err(GpuError::shape(
                what,
                format!(
                    "a map of {:?} words (at least rows = {rows}) and a stand-in of {}",
                    a.map.map(DeviceBuffer::len),
                    a.no_map.len()
                ),
            ));
        }
        if a.y.len() < rows * m {
            return Err(GpuError::shape(
                what,
                format!("y.len() {} < rows*m = {}", a.y.len(), rows * m),
            ));
        }
        let grid = launch_u32(what, "grid", rows.div_ceil(8))?;
        let n_matrix = launch_u32(what, "n_matrix", n_matrix)?;
        let rows = launch_u32(what, "rows", rows)?;
        let m = launch_u32(what, "m", m)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let iters = n_sb.div_ceil(2);
        let mapped = u32::from(a.map.is_some());
        let map = a.map.unwrap_or(a.no_map);
        if a.m == 1 {
            let prep = self
                .module
                .prepare_q6k_gemv_ids(LaunchConfig1D::new(grid, 256, 0))?;
            self.module.q6k_gemv_ids(
                stream,
                &prep,
                a.w.buf(),
                &a.act.q6,
                &a.act.d8,
                map,
                rows,
                n_matrix,
                n_sb,
                iters,
                mapped,
                a.fault,
                a.y,
            )?;
        } else {
            let prep = self
                .module
                .prepare_q6k_gemv_ids_mcol(LaunchConfig1D::new(grid, 256, 0))?;
            self.module.q6k_gemv_ids_mcol(
                stream,
                &prep,
                a.w.buf(),
                &a.act.q6,
                &a.act.d8,
                map,
                rows,
                n_matrix,
                m,
                n_sb,
                iters,
                mapped,
                a.fault,
                a.y,
            )?;
        }
        Ok(())
    }
}
