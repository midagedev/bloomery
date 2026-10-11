//! RMSNorm over bf16 rows with f32 gains, `vision.py`'s `RMSNorm`: `x.float()`, then
//! `x · rsqrt(mean(x²) + eps)`, then `weight · x`, rounded to bf16.
//!
//! Not `bloomery_gpu::elem::rms_norm`: that kernel reads and writes f32 rows, and the rule here
//! rounds its input from bf16 and its output to bf16 — the same arithmetic in between on other
//! types, so a copy of the shape, not of the code.
//!
//! Numeric rule, per row of `dim` values: thread `t` of the [`THREADS`]-thread block sums the
//! squares of the value pairs at words `t, t + THREADS, …` of the row, each square added with one
//! fused multiply-add, low value first; each warp combines its 32 partials by the xor butterfly
//! 16, 8, 4, 2, 1; the eight warp sums combine as `((w0+w1)+(w2+w3))+((w4+w5)+(w6+w7))`. The
//! scale is `1 / sqrt(sum / dim + eps)` — every op correctly rounded, where the reference's
//! `rsqrt` is the card's approximation — then `y = bf16(g · (x · scale))`, each multiply rounded
//! on its own. [`rms_norm_ref`] is that order on the host, bit for bit.
//!
//! The per-head norm ([`NormKernels::enqueue_heads`]) is the same rule on rows of [`HEAD`] values:
//! one warp takes one head, lane `t` the value pair at word `t`, so the sum is the warp's butterfly
//! alone — what the eight-warp tree of a 64-value row reduces to, the other seven warps adding
//! exact zeros — and [`head_rms_norm_ref`] is [`rms_norm_ref`] on each head. The gain is one row
//! of [`HEAD`] values every head of the segment shares.

use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne};
use cuda_device::float::{add_rn_f32, div_rn_f32, fma_rn_f32, mul_rn_f32, sqrt_rn_f32};
use cuda_device::{
    DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads one row takes.
pub const THREADS: usize = 256;
const THREADS_U32: u32 = THREADS as u32;
const WARPS: usize = THREADS / 32;
const _: () = assert!(WARPS == 8);
/// Values in one attention head: the width the per-head norm normalizes and its gain has.
pub const HEAD: usize = 64;
const _: () = assert!(HEAD == 2 * 32);

/// The row's sum from its eight warp sums, in the rule's fixed tree.
#[inline(always)]
fn warp_tree(w: [f32; WARPS]) -> f32 {
    ((w[0] + w[1]) + (w[2] + w[3])) + ((w[4] + w[5]) + (w[6] + w[7]))
}

/// The host rule of one row: `x` the row's bf16, `g` the gains.
#[must_use]
pub fn rms_norm_ref(x: &[u16], g: &[f32], eps: f32) -> Vec<u16> {
    let dim = x.len();
    let words = dim / 2;
    let mut part = [0.0f32; THREADS];
    for (t, p) in part.iter_mut().enumerate() {
        let mut w = t;
        while w < words {
            let (a, b) = (crate::bf16_f32(x[2 * w]), crate::bf16_f32(x[2 * w + 1]));
            *p = a.mul_add(a, *p);
            *p = b.mul_add(b, *p);
            w += THREADS;
        }
    }
    let mut warp_sums = [0.0f32; WARPS];
    for (wi, ws) in warp_sums.iter_mut().enumerate() {
        let mut lanes: [f32; 32] = part[wi * 32..wi * 32 + 32].try_into().expect("32 lanes");
        for m in [16usize, 8, 4, 2, 1] {
            let prev = lanes;
            for (l, v) in lanes.iter_mut().enumerate() {
                *v = prev[l] + prev[l ^ m];
            }
        }
        *ws = lanes[0];
    }
    let mean = warp_tree(warp_sums) / dim as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    x.iter()
        .zip(g)
        .map(|(&v, &gi)| crate::f32_bf16(gi * (crate::bf16_f32(v) * scale)))
        .collect()
}

/// The host rule of the per-head norm: every head of `n_heads` at column `col0 + h·`[`HEAD`] of
/// every `row_width`-value row of `x`, normalized in place by [`rms_norm_ref`] with the gain `g`
/// of [`HEAD`] values.
pub fn head_rms_norm_ref(
    x: &mut [u16],
    row_width: usize,
    col0: usize,
    n_heads: usize,
    g: &[f32],
    eps: f32,
) {
    for row in x.chunks_exact_mut(row_width) {
        for h in 0..n_heads {
            let seg = &mut row[col0 + h * HEAD..col0 + (h + 1) * HEAD];
            seg.copy_from_slice(&rms_norm_ref(seg, g, eps));
        }
    }
}

#[cuda_module]
mod norm_kernels {
    use super::*;

    /// `y = RMSNorm(x)` over `rows` rows of `dim` bf16, one block per row, the module rule. `dim`
    /// even and non-zero (host-checked). The row guard is block-uniform.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= rows * dim, gain.len() >= dim, y.len() >= rows * dim)
    )]
    pub fn vis_rms_norm(
        x: &[u16],
        gain: &[f32],
        eps: f32,
        dim: u32,
        rows: u32,
        mut y: DisjointSlice<u16>,
    ) {
        static mut WSUM: SharedArray<f32, WARPS> = SharedArray::UNINIT;

        let r = thread::blockIdx_x() as usize;
        if r >= rows as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let d = dim as usize;
        let words = d / 2;
        let base = r * d;
        let xw = x.as_ptr().cast::<u32>();
        let mut acc = 0.0f32;
        let mut w = tid;
        while w < words {
            // SAFETY: w < dim / 2 and r < rows keep the word inside row r of x (contract); dim
            // is even, so the word is aligned.
            let pair = unsafe { *xw.add(base / 2 + w) };
            let a = bf16_to_f32(pair as u16);
            let b = bf16_to_f32((pair >> 16) as u16);
            acc = fma_rn_f32(a, a, acc);
            acc = fma_rn_f32(b, b, acc);
            w += THREADS;
        }
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 16));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 8));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 4));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 2));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 1));
        // SAFETY: block-shared, WARPS == blockDim.x / 32, written before the barrier that
        // publishes it.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM) };
        if warp::lane_id() == 0 {
            // SAFETY: tid / 32 < WARPS bounds the slot; lane 0 is its one writer.
            unsafe {
                *ws.add(tid / 32) = acc;
            }
        }
        thread::sync_threads();
        // SAFETY: the eight slots, published by the barrier above.
        let sums = unsafe {
            [
                *ws,
                *ws.add(1),
                *ws.add(2),
                *ws.add(3),
                *ws.add(4),
                *ws.add(5),
                *ws.add(6),
                *ws.add(7),
            ]
        };
        let mean = div_rn_f32(warp_tree(sums), d as f32);
        let scale = div_rn_f32(1.0, sqrt_rn_f32(add_rn_f32(mean, eps)));
        let mut w = tid;
        while w < words {
            // SAFETY: as the first walk.
            let pair = unsafe { *xw.add(base / 2 + w) };
            // SAFETY: 2w + 1 < dim <= gain.len() (contract).
            let (g0, g1) = unsafe { (*gain.get_unchecked(2 * w), *gain.get_unchecked(2 * w + 1)) };
            let y0 = mul_rn_f32(g0, mul_rn_f32(bf16_to_f32(pair as u16), scale));
            let y1 = mul_rn_f32(g1, mul_rn_f32(bf16_to_f32((pair >> 16) as u16), scale));
            // SAFETY: base + 2w + 1 < rows·dim <= y.len(); this thread owns both positions.
            unsafe {
                *y.get_unchecked_mut(base + 2 * w) = f32_to_bf16_rne(y0);
                *y.get_unchecked_mut(base + 2 * w + 1) = f32_to_bf16_rne(y1);
            }
            w += THREADS;
        }
    }

    /// `x = RMSNorm(x)` in place over the `n_heads` heads of [`HEAD`] bf16 values at columns
    /// `col0 + h·64` of each of `rows` rows of `row_width`, every head with the gain of 64: the
    /// module rule per head. One warp per (row, head), eight to a block; lane `t` owns the pair at
    /// word `t` of its head, reads it before it writes it, and no other lane touches it. The head
    /// guard is warp-uniform.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= rows * row_width, gain.len() >= 64)
    )]
    pub fn vis_head_rms_norm(
        gain: &[f32],
        eps: f32,
        row_width: u32,
        col0: u32,
        n_heads: u32,
        rows: u32,
        mut x: DisjointSlice<u16>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let unit = thread::blockIdx_x() as usize * WARPS + tid / 32;
        let heads = n_heads as usize;
        if unit >= rows as usize * heads {
            return;
        }
        let (r, h) = (unit / heads, unit % heads);
        let at = r * row_width as usize + col0 as usize + h * HEAD + 2 * lane;
        // SAFETY: r < rows and col0 + n_heads·64 <= row_width (host-checked) put at + 1 below
        // rows·row_width <= x.len() (contract); 2·lane + 1 < 64 <= gain.len() (contract). This
        // lane is the only one touching its two positions.
        let (a, b, g0, g1) = unsafe {
            (
                bf16_to_f32(*x.get_unchecked_mut(at)),
                bf16_to_f32(*x.get_unchecked_mut(at + 1)),
                *gain.get_unchecked(2 * lane),
                *gain.get_unchecked(2 * lane + 1),
            )
        };
        let mut acc = fma_rn_f32(a, a, 0.0);
        acc = fma_rn_f32(b, b, acc);
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 16));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 8));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 4));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 2));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 1));
        let mean = div_rn_f32(acc, HEAD as f32);
        let scale = div_rn_f32(1.0, sqrt_rn_f32(add_rn_f32(mean, eps)));
        let y0 = mul_rn_f32(g0, mul_rn_f32(a, scale));
        let y1 = mul_rn_f32(g1, mul_rn_f32(b, scale));
        // SAFETY: the two positions read above.
        unsafe {
            *x.get_unchecked_mut(at) = f32_to_bf16_rne(y0);
            *x.get_unchecked_mut(at + 1) = f32_to_bf16_rne(y1);
        }
    }
}

/// [`NormKernels::enqueue_heads`]'s arguments: `x` holds `rows` rows of `row_width` bf16, and the
/// `n_heads` heads of [`HEAD`] values from column `col0` are normalized in place with `gain`.
pub struct HeadNormArgs<'a> {
    pub x: &'a mut DeviceBuffer<u16>,
    pub gain: &'a DeviceBuffer<f32>,
    pub eps: f32,
    pub rows: usize,
    pub row_width: usize,
    pub col0: usize,
    pub n_heads: usize,
}

/// The loaded norm module.
pub struct NormKernels {
    module: norm_kernels::LoadedModule,
}

impl NormKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<NormKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { bloomery_gpu::shared_module!(norm_kernels, ctx)? };
        Ok(NormKernels { module })
    }

    /// Enqueue `y = RMSNorm(x)` over `rows` rows of `gain.len()` values. Asynchronous.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<u16>,
        gain: &DeviceBuffer<f32>,
        eps: f32,
        rows: usize,
        y: &mut DeviceBuffer<u16>,
    ) -> Result<(), GpuError> {
        let what = "NormKernels::enqueue";
        let dim = gain.len();
        if dim == 0
            || !dim.is_multiple_of(2)
            || rows == 0
            || x.len() < rows * dim
            || y.len() < rows * dim
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "rows {rows} of dim {dim} (even, non-zero): x.len() {}, y.len() {}",
                    x.len(),
                    y.len()
                ),
            });
        }
        let prep = self.module.prepare_vis_rms_norm(LaunchConfig1D::new(
            launch_u32(what, "rows", rows)?,
            THREADS_U32,
            0,
        ))?;
        self.module.vis_rms_norm(
            stream,
            &prep,
            x,
            gain,
            eps,
            launch_u32(what, "dim", dim)?,
            launch_u32(what, "rows", rows)?,
            y,
        )?;
        Ok(())
    }
    /// Enqueue the per-head norm ([`HeadNormArgs`]): in place, one launch per segment. The gain is
    /// [`HEAD`] values; the heads must lie inside the row. Asynchronous.
    pub fn enqueue_heads(
        &self,
        stream: &CudaStream,
        args: HeadNormArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "NormKernels::enqueue_heads";
        let HeadNormArgs {
            x,
            gain,
            eps,
            rows,
            row_width,
            col0,
            n_heads,
        } = args;
        if rows == 0
            || n_heads == 0
            || gain.len() != HEAD
            || col0 + n_heads * HEAD > row_width
            || x.len() < rows * row_width
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "{n_heads} heads of {HEAD} at column {col0} of {rows} rows of {row_width}: gain.len() {} (need {HEAD}), x.len() {}",
                    gain.len(),
                    x.len()
                ),
            });
        }
        let grid = launch_u32(what, "grid", (rows * n_heads).div_ceil(WARPS))?;
        let prep =
            self.module
                .prepare_vis_head_rms_norm(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.vis_head_rms_norm(
            stream,
            &prep,
            gain,
            eps,
            launch_u32(what, "row_width", row_width)?,
            launch_u32(what, "col0", col0)?,
            launch_u32(what, "n_heads", n_heads)?,
            launch_u32(what, "rows", rows)?,
            x,
        )?;
        Ok(())
    }
}
