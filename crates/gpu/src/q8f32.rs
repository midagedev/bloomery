//! Q8_0-weight and F32-weight gemv with f32 activations (package P3): the
//! two matmul sites whose weights are not K-quants. The attention `q_nope2`
//! site consumes the load-time Q8_0 requant of `wk_b` (32-value blocks), and
//! the MoE router `ffn_gate_inp` is F32. Both are precision-sensitive — a
//! flipped near-tie in the router's top-6 changes the token — so activations
//! stay f32 here: the kernel dequantizes inline and accumulates in f32, and
//! the gate band is accordingly tighter than the q8_1-activation kernels'.
//!
//! Q8_0 device layout, fixed at load time (decision 4: format conversion is
//! load-time work):
//! - `qs`: row-major `u32` words, 8 words per 32-value block, code `j` in
//!   word `j/4`, byte `j%4` (little-endian), `k/4` words per row;
//! - `d`: row-major `f32` block scales, `k/32` per row, converted from the
//!   Q8_0 block's f16 scale at load — the exact value the reference
//!   dequantizes with, so the device side never does f16 arithmetic.
//!
//! Numeric contract, both kernels: one warp owns one output row; lane L
//! accumulates sequentially over the row's 32-value chunks (value index
//! `32·it + L`, chunks in increasing `it`, columns in increasing order, one
//! f32 multiply-add per term), and the 32 lane sums are then combined by the
//! fixed five-step butterfly (xor 16, 8, 4, 2, 1). The combination tree is a
//! function of (k, m) only — never of the data, the row index, or the grid
//! geometry. Output layout matches the K-quant gemvs: `y[r·m + c]`.

use crate::GpuError;
use crate::tensor::DeviceTensor;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

// --------------------------------------------------------------- cores
//
// Same standing as `crate::cores`: ordinary `#[inline(always)]` functions a
// per-op wrapper and a later fused kernel can both call. Slice arguments are
// the whole device buffer plus the row index — the verified core shape
// (`cores::q4k_a_chain` takes a buffer and a base), not a `DisjointSlice`.

/// Lane `lane`'s partial sums for one F32 row: `Σ_it fma(w[row·k + 32·it +
/// lane], x[c·k + 32·it + lane])`, accumulated sequentially in `it`.
/// Columns past `m_cols` stay 0.0.
///
/// Caller contract: `w.len() >= (row + 1) * k`, `x.len() >= m_cols * k`,
/// `k` a positive multiple of 32, `m_cols` in 1..=8, `lane < 32`.
#[inline(always)]
pub fn f32_lane_partials(
    w: &[f32],
    x: &[f32],
    k: u32,
    row: usize,
    m_cols: u32,
    lane: usize,
) -> [f32; 8] {
    let k = k as usize;
    let w_row = row * k;
    let mut f0 = 0.0f32;
    let mut f1 = 0.0f32;
    let mut f2 = 0.0f32;
    let mut f3 = 0.0f32;
    let mut f4 = 0.0f32;
    let mut f5 = 0.0f32;
    let mut f6 = 0.0f32;
    let mut f7 = 0.0f32;
    let mut it = 0usize;
    while it < k >> 5 {
        let kk = it * 32 + lane;
        // SAFETY: kk < k <= w.len() - w_row by the caller contract.
        let wv = unsafe { *w.get_unchecked(w_row + kk) };
        // Column 0 (always active).
        // SAFETY: kk < k <= x.len() by the caller contract (m_cols >= 1).
        f0 = f32::mul_add(wv, unsafe { *x.get_unchecked(kk) }, f0);
        // Columns 1..7: one launch-uniform guard per column so the work
        // scales with m_cols; each guard makes the column's span live.
        if m_cols > 1 {
            // SAFETY: m_cols > 1 => x.len() >= 2*k > k + kk.
            f1 = f32::mul_add(wv, unsafe { *x.get_unchecked(k + kk) }, f1);
        }
        if m_cols > 2 {
            // SAFETY: m_cols > 2 => x.len() >= 3*k > 2*k + kk.
            f2 = f32::mul_add(wv, unsafe { *x.get_unchecked(2 * k + kk) }, f2);
        }
        if m_cols > 3 {
            // SAFETY: m_cols > 3 => x.len() >= 4*k > 3*k + kk.
            f3 = f32::mul_add(wv, unsafe { *x.get_unchecked(3 * k + kk) }, f3);
        }
        if m_cols > 4 {
            // SAFETY: m_cols > 4 => x.len() >= 5*k > 4*k + kk.
            f4 = f32::mul_add(wv, unsafe { *x.get_unchecked(4 * k + kk) }, f4);
        }
        if m_cols > 5 {
            // SAFETY: m_cols > 5 => x.len() >= 6*k > 5*k + kk.
            f5 = f32::mul_add(wv, unsafe { *x.get_unchecked(5 * k + kk) }, f5);
        }
        if m_cols > 6 {
            // SAFETY: m_cols > 6 => x.len() >= 7*k > 6*k + kk.
            f6 = f32::mul_add(wv, unsafe { *x.get_unchecked(6 * k + kk) }, f6);
        }
        if m_cols > 7 {
            // SAFETY: m_cols > 7 => x.len() >= 8*k > 7*k + kk.
            f7 = f32::mul_add(wv, unsafe { *x.get_unchecked(7 * k + kk) }, f7);
        }
        it += 1;
    }
    [f0, f1, f2, f3, f4, f5, f6, f7]
}

/// Lane `lane`'s partial sums for one Q8_0 row: the weight at value `kk` of
/// `row` is `q·d` with `q` the signed code in word `qs[row·k/4 + kk/4]`,
/// byte `kk%4`, and `d` the block scale `d[row·k/32 + kk/32]` — the same
/// bits the reference's dequantizer produces. `x` is read from base `x0`
/// (column c's values at `x0 + c*k .. +k`), so a caller can dot against a
/// slice of a wider buffer without subslicing it. Accumulation order as
/// `f32_lane_partials`.
///
/// Caller contract: `qs.len() >= (row + 1) * k/4`, `d.len() >= (row + 1) *
/// k/32`, `x.len() >= x0 + m_cols * k`, `k` a positive multiple of 32,
/// `m_cols` in 1..=8, `lane < 32`.
#[inline(always)]
pub fn q8_0_lane_partials(
    qs: &[u32],
    d: &[f32],
    x: &[f32],
    k: u32,
    row: usize,
    x0: usize,
    m_cols: u32,
    lane: usize,
) -> [f32; 8] {
    let k = k as usize;
    let qs_row = row * (k >> 2);
    let d_row = row * (k >> 5);
    let mut f0 = 0.0f32;
    let mut f1 = 0.0f32;
    let mut f2 = 0.0f32;
    let mut f3 = 0.0f32;
    let mut f4 = 0.0f32;
    let mut f5 = 0.0f32;
    let mut f6 = 0.0f32;
    let mut f7 = 0.0f32;
    let mut it = 0usize;
    while it < k >> 5 {
        let kk = it * 32 + lane;
        // SAFETY: kk < k, so word kk/4 < k/4 <= qs.len() - qs_row by the
        // caller contract.
        let q = unsafe { (*qs.get_unchecked(qs_row + (kk >> 2)) >> (8 * (kk & 3))) as u8 as i8 };
        // SAFETY: it = kk/32 < k/32 <= d.len() - d_row by the caller
        // contract.
        let wv = q as f32 * unsafe { *d.get_unchecked(d_row + it) };
        // Column 0 (always active).
        // SAFETY: kk < k, and x0 + kk < x0 + k <= x.len() by the caller
        // contract (m_cols >= 1).
        f0 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + kk) }, f0);
        // Columns 1..7, one launch-uniform guard per column.
        if m_cols > 1 {
            // SAFETY: m_cols > 1 => x.len() >= x0 + 2*k > x0 + k + kk.
            f1 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + k + kk) }, f1);
        }
        if m_cols > 2 {
            // SAFETY: m_cols > 2 => x.len() >= x0 + 3*k > x0 + 2*k + kk.
            f2 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + 2 * k + kk) }, f2);
        }
        if m_cols > 3 {
            // SAFETY: m_cols > 3 => x.len() >= x0 + 4*k > x0 + 3*k + kk.
            f3 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + 3 * k + kk) }, f3);
        }
        if m_cols > 4 {
            // SAFETY: m_cols > 4 => x.len() >= x0 + 5*k > x0 + 4*k + kk.
            f4 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + 4 * k + kk) }, f4);
        }
        if m_cols > 5 {
            // SAFETY: m_cols > 5 => x.len() >= x0 + 6*k > x0 + 5*k + kk.
            f5 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + 5 * k + kk) }, f5);
        }
        if m_cols > 6 {
            // SAFETY: m_cols > 6 => x.len() >= x0 + 7*k > x0 + 6*k + kk.
            f6 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + 6 * k + kk) }, f6);
        }
        if m_cols > 7 {
            // SAFETY: m_cols > 7 => x.len() >= x0 + 8*k > x0 + 7*k + kk.
            f7 = f32::mul_add(wv, unsafe { *x.get_unchecked(x0 + 7 * k + kk) }, f7);
        }
        it += 1;
    }
    [f0, f1, f2, f3, f4, f5, f6, f7]
}

/// The row's m sums from the per-lane partials: the fixed five-step
/// butterfly per column (`warp::reduce_sum_f32`), columns past `m_cols`
/// left at 0.0. `m_cols` must be warp-uniform — a launch-wide constant in
/// every caller.
#[inline(always)]
pub fn gemv_lane_sums(f: [f32; 8], m_cols: u32) -> [f32; 8] {
    let s0 = warp::reduce_sum_f32(f[0]);
    if m_cols == 1 {
        return [s0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    }
    let s1 = warp::reduce_sum_f32(f[1]);
    let s2 = if m_cols > 2 {
        warp::reduce_sum_f32(f[2])
    } else {
        0.0
    };
    let s3 = if m_cols > 3 {
        warp::reduce_sum_f32(f[3])
    } else {
        0.0
    };
    let s4 = if m_cols > 4 {
        warp::reduce_sum_f32(f[4])
    } else {
        0.0
    };
    let s5 = if m_cols > 5 {
        warp::reduce_sum_f32(f[5])
    } else {
        0.0
    };
    let s6 = if m_cols > 6 {
        warp::reduce_sum_f32(f[6])
    } else {
        0.0
    };
    let s7 = if m_cols > 7 {
        warp::reduce_sum_f32(f[7])
    } else {
        0.0
    };
    [s0, s1, s2, s3, s4, s5, s6, s7]
}

// -------------------------------------------------------------- kernels

#[cuda_module]
mod q8f32_kernels {
    use super::*;

    /// F32 gemv, `y[r·m + c] = Σ_k w[r·k + k'] · x[c·k + k']`, M <= 8: one
    /// warp per row, 8 rows per 256-thread block, the K-quant gemvs'
    /// skeleton. The row guard is warp-uniform, so the cores' butterfly
    /// always sees a full warp. Summation order: the module doc's contract.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_rows * k,
            x.len() >= m_cols * k,
            y.len() >= n_rows * m_cols
        )
    )]
    pub fn f32_gemv(
        w: &[f32],
        x: &[f32],
        n_rows: u32,
        k: u32,
        m_cols: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let sums = gemv_lane_sums(f32_lane_partials(w, x, k, row, m_cols, lane), m_cols);
        let m = m_cols as usize;
        if lane == 0 {
            // SAFETY: only lane 0 of the warp owning `row` writes, exactly
            // the m live slots of y[row*m .. row*m + m]; y.len() >=
            // n_rows * m_cols by the launch contract and the guards bound
            // every store by m.
            unsafe {
                let b = row * m;
                *y.get_unchecked_mut(b) = sums[0];
                if m > 1 {
                    *y.get_unchecked_mut(b + 1) = sums[1];
                }
                if m > 2 {
                    *y.get_unchecked_mut(b + 2) = sums[2];
                }
                if m > 3 {
                    *y.get_unchecked_mut(b + 3) = sums[3];
                }
                if m > 4 {
                    *y.get_unchecked_mut(b + 4) = sums[4];
                }
                if m > 5 {
                    *y.get_unchecked_mut(b + 5) = sums[5];
                }
                if m > 6 {
                    *y.get_unchecked_mut(b + 6) = sums[6];
                }
                if m > 7 {
                    *y.get_unchecked_mut(b + 7) = sums[7];
                }
            }
        }
    }

    /// Q8_0 gemv against f32 activations, M <= 8: same skeleton and output
    /// layout as `f32_gemv`, weights decoded per the module doc's device
    /// layout. `k` a multiple of 32 (host-validated; the contract binds the
    /// word and scale buffers through `k`: 4·qs.len() and 32·d.len() cover
    /// `n_rows * k` values exactly when 32 | k).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * qs.len() >= n_rows * k,
            32 * d.len() >= n_rows * k,
            x.len() >= m_cols * k,
            y.len() >= n_rows * m_cols
        )
    )]
    pub fn q8_0_gemv(
        qs: &[u32],
        d: &[f32],
        x: &[f32],
        n_rows: u32,
        k: u32,
        m_cols: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let sums = gemv_lane_sums(
            q8_0_lane_partials(qs, d, x, k, row, 0, m_cols, lane),
            m_cols,
        );
        let m = m_cols as usize;
        if lane == 0 {
            // SAFETY: as in f32_gemv — lane 0 of the row's warp writes only
            // the m live slots of its row segment, inside y by the launch
            // contract.
            unsafe {
                let b = row * m;
                *y.get_unchecked_mut(b) = sums[0];
                if m > 1 {
                    *y.get_unchecked_mut(b + 1) = sums[1];
                }
                if m > 2 {
                    *y.get_unchecked_mut(b + 2) = sums[2];
                }
                if m > 3 {
                    *y.get_unchecked_mut(b + 3) = sums[3];
                }
                if m > 4 {
                    *y.get_unchecked_mut(b + 4) = sums[4];
                }
                if m > 5 {
                    *y.get_unchecked_mut(b + 5) = sums[5];
                }
                if m > 6 {
                    *y.get_unchecked_mut(b + 6) = sums[6];
                }
                if m > 7 {
                    *y.get_unchecked_mut(b + 7) = sums[7];
                }
            }
        }
    }
}

/// The loaded P3 device module: `f32_gemv` and `q8_0_gemv`. Owns no context
/// and no stream — the caller passes the engine stream (`Gpu::stream()`) per
/// enqueue, so launches order with the rest of the step and are capturable.
pub struct Q8F32Kernels {
    module: q8f32_kernels::LoadedModule,
}

impl Q8F32Kernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Q8F32Kernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { q8f32_kernels::load(ctx)? };
        Ok(Q8F32Kernels { module })
    }

    /// Enqueue `y = W · x` for an F32 weight of `w.rows()` rows × `w.cols()`
    /// (= k) values, against `m` f32 activation columns of k values each.
    /// `y` holds `rows * m` f32, row-major with m outputs per row.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_f32_gemv(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let (n_rows, k) = (w.rows(), w.cols());
        check_gemv_geometry("enqueue_f32_gemv", n_rows, k, x.len(), m, y.len())?;
        let prep =
            self.module
                .prepare_f32_gemv(LaunchConfig1D::new(n_rows.div_ceil(8) as u32, 256, 0))?;
        self.module.f32_gemv(
            stream,
            &prep,
            w.buf(),
            x,
            n_rows as u32,
            k as u32,
            m as u32,
            y,
        )?;
        Ok(())
    }

    /// Enqueue `y = W · x` for a Q8_0 weight in the module doc's device
    /// layout: `qs` `rows × k/4` u32 words and `d` `rows × k/32` f32 scales —
    /// against `m` f32 activation columns of `k = d.cols() * 32` values each
    /// (the scale buffer's width fixes k; `qs.cols()` must equal
    /// `d.cols() * 8`). Output layout as `enqueue_f32_gemv`. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_q8_0_gemv(
        &self,
        stream: &CudaStream,
        qs: &DeviceTensor<u32>,
        d: &DeviceTensor<f32>,
        x: &DeviceBuffer<f32>,
        m: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let n_rows = d.rows();
        let k = d.cols() * 32;
        check_gemv_geometry("enqueue_q8_0_gemv", n_rows, k, x.len(), m, y.len())?;
        if qs.rows() != n_rows || qs.cols() != d.cols() * 8 {
            return Err(format!(
                "enqueue_q8_0_gemv: qs is {}x{}, want {}x{} (k/4 words per row, k = d.cols()*32 = {k})",
                qs.rows(),
                qs.cols(),
                n_rows,
                d.cols() * 8
            )
            .into());
        }
        let prep = self.module.prepare_q8_0_gemv(LaunchConfig1D::new(
            n_rows.div_ceil(8) as u32,
            256,
            0,
        ))?;
        self.module.q8_0_gemv(
            stream,
            &prep,
            qs.buf(),
            d.buf(),
            x,
            n_rows as u32,
            k as u32,
            m as u32,
            y,
        )?;
        Ok(())
    }
}

/// Reject geometry the two gemvs' launch contracts do not cover: both walk
/// the row in 32-value chunks, so k must be a positive multiple of 32, rows
/// >= 1, and both support 1..=8 columns.
fn check_gemv_geometry(
    what: &str,
    n_rows: usize,
    k: usize,
    x_len: usize,
    m: usize,
    y_len: usize,
) -> Result<(), GpuError> {
    if n_rows == 0 || k == 0 || !k.is_multiple_of(32) {
        return Err(format!(
            "{what}: need n_rows >= 1 and k a positive multiple of 32, got n_rows={n_rows} k={k}"
        )
        .into());
    }
    if !(1..=8).contains(&m) {
        return Err(format!("{what}: need 1 <= m <= 8, got m={m}").into());
    }
    if x_len < m * k {
        return Err(format!("{what}: x.len() {x_len} < m*k = {}", m * k).into());
    }
    if y_len < n_rows * m {
        return Err(format!("{what}: y.len() {y_len} < n_rows*m = {}", n_rows * m).into());
    }
    Ok(())
}
