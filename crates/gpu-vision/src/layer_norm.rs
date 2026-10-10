//! LayerNorm with a gain and a bias over bf16 rows, `nn.LayerNorm` of the Qwen3-VL tower
//! (llama.cpp's `ggml_norm`, then the gain and the bias): `x` widened exactly from bf16, the mean
//! first, then the centred sum of squares, then `y = ((x − μ) · rstd) · g + b`, rounded to bf16.
//!
//! Not [`crate::norm`]'s RMSNorm: that rule has no mean and no bias.
//!
//! Numeric rule, per row of `dim` values: thread `t` of the [`THREADS`]-thread block adds the
//! value pairs at words `t, t + THREADS, …` of the row, low value first; each warp combines its 32
//! partials by the xor butterfly 16, 8, 4, 2, 1; the eight warp sums combine as
//! `((w0+w1)+(w2+w3))+((w4+w5)+(w6+w7))`; `μ = sum / dim`. The second walk is the same on the
//! centred values: each square added with one fused multiply-add, low value first, `var = sum /
//! dim`. `rstd = 1 / sqrt(var + eps)`, every op correctly rounded, then each of the three
//! multiply-adds of `y` is rounded on its own. [`layer_norm_ref`] is that order on the host, bit
//! for bit.

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

/// The row's sum from its eight warp sums, in the rule's fixed tree.
#[inline(always)]
fn warp_tree(w: [f32; WARPS]) -> f32 {
    ((w[0] + w[1]) + (w[2] + w[3])) + ((w[4] + w[5]) + (w[6] + w[7]))
}

/// The per-thread partials of one walk over the row's word pairs, as the kernel's threads hold
/// them: `step(acc, a, b)` folds one value pair into a partial.
fn partials(x: &[u16], step: impl Fn(f32, f32, f32) -> f32) -> [f32; THREADS] {
    let words = x.len() / 2;
    let mut part = [0.0f32; THREADS];
    for (t, p) in part.iter_mut().enumerate() {
        let mut w = t;
        while w < words {
            *p = step(*p, crate::bf16_f32(x[2 * w]), crate::bf16_f32(x[2 * w + 1]));
            w += THREADS;
        }
    }
    part
}

/// The row's sum from the per-thread partials: the xor butterfly inside each warp, then the tree.
fn combine(part: [f32; THREADS]) -> f32 {
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
    warp_tree(warp_sums)
}

/// The host rule of one row: `x` the row's bf16, `g` the gains, `b` the biases.
#[must_use]
pub fn layer_norm_ref(x: &[u16], g: &[f32], b: &[f32], eps: f32) -> Vec<u16> {
    let dim = x.len() as f32;
    let mean = combine(partials(x, |acc, v0, v1| (acc + v0) + v1)) / dim;
    let var = combine(partials(x, |acc, v0, v1| {
        let (d0, d1) = (v0 - mean, v1 - mean);
        d1.mul_add(d1, d0.mul_add(d0, acc))
    })) / dim;
    let rstd = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(g.iter().zip(b))
        .map(|(&v, (&gi, &bi))| {
            let d = crate::bf16_f32(v) - mean;
            crate::f32_bf16(((d * rstd) * gi) + bi)
        })
        .collect()
}

#[cuda_module]
mod layer_norm_kernels {
    use super::*;

    /// `y = LayerNorm(x)` over `rows` rows of `dim` bf16, one block per row, the module rule.
    /// `dim` even and non-zero (host-checked). The row guard is block-uniform.
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
            x.len() >= rows * dim,
            gain.len() >= dim,
            bias.len() >= dim,
            y.len() >= rows * dim
        )
    )]
    pub fn vis_layer_norm(
        x: &[u16],
        gain: &[f32],
        bias: &[f32],
        eps: f32,
        dim: u32,
        rows: u32,
        mut y: DisjointSlice<u16>,
    ) {
        static mut WSUM_MEAN: SharedArray<f32, WARPS> = SharedArray::UNINIT;
        static mut WSUM_VAR: SharedArray<f32, WARPS> = SharedArray::UNINIT;

        let r = thread::blockIdx_x() as usize;
        if r >= rows as usize {
            return;
        }
        let tid = thread::threadIdx_x() as usize;
        let d = dim as usize;
        let words = d / 2;
        let base = r * d;
        let xw = x.as_ptr().cast::<u32>();

        // The mean: the thread's pairs, the warp's butterfly, the eight warp sums.
        let mut acc = 0.0f32;
        let mut w = tid;
        while w < words {
            // SAFETY: w < dim / 2 and r < rows keep the word inside row r of x (contract); dim
            // is even, so the word is aligned.
            let pair = unsafe { *xw.add(base / 2 + w) };
            acc = add_rn_f32(acc, bf16_to_f32(pair as u16));
            acc = add_rn_f32(acc, bf16_to_f32((pair >> 16) as u16));
            w += THREADS;
        }
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 16));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 8));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 4));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 2));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 1));
        // SAFETY: block-shared, WARPS == blockDim.x / 32, written before the barrier that
        // publishes it.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM_MEAN) };
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

        // The variance: the same walk over the centred values.
        let mut acc = 0.0f32;
        let mut w = tid;
        while w < words {
            // SAFETY: as the first walk.
            let pair = unsafe { *xw.add(base / 2 + w) };
            let d0 = add_rn_f32(bf16_to_f32(pair as u16), -mean);
            let d1 = add_rn_f32(bf16_to_f32((pair >> 16) as u16), -mean);
            acc = fma_rn_f32(d0, d0, acc);
            acc = fma_rn_f32(d1, d1, acc);
            w += THREADS;
        }
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 16));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 8));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 4));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 2));
        acc = add_rn_f32(acc, warp::shuffle_xor_f32(acc, 1));
        // SAFETY: as the first array; a second one, so no barrier separates the two phases' reads.
        let ws = unsafe { SharedArray::as_raw_mut_ptr(&raw mut WSUM_VAR) };
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
        let var = div_rn_f32(warp_tree(sums), d as f32);
        let rstd = div_rn_f32(1.0, sqrt_rn_f32(add_rn_f32(var, eps)));

        let mut w = tid;
        while w < words {
            // SAFETY: as the first walk.
            let pair = unsafe { *xw.add(base / 2 + w) };
            // SAFETY: 2w + 1 < dim <= gain.len(), bias.len() (contract).
            let (g0, g1, b0, b1) = unsafe {
                (
                    *gain.get_unchecked(2 * w),
                    *gain.get_unchecked(2 * w + 1),
                    *bias.get_unchecked(2 * w),
                    *bias.get_unchecked(2 * w + 1),
                )
            };
            let d0 = add_rn_f32(bf16_to_f32(pair as u16), -mean);
            let d1 = add_rn_f32(bf16_to_f32((pair >> 16) as u16), -mean);
            let y0 = add_rn_f32(mul_rn_f32(mul_rn_f32(d0, rstd), g0), b0);
            let y1 = add_rn_f32(mul_rn_f32(mul_rn_f32(d1, rstd), g1), b1);
            // SAFETY: base + 2w + 1 < rows·dim <= y.len(); this thread owns both positions.
            unsafe {
                *y.get_unchecked_mut(base + 2 * w) = f32_to_bf16_rne(y0);
                *y.get_unchecked_mut(base + 2 * w + 1) = f32_to_bf16_rne(y1);
            }
            w += THREADS;
        }
    }
}

/// The loaded LayerNorm module.
pub struct LayerNormKernels {
    module: layer_norm_kernels::LoadedModule,
}

impl LayerNormKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<LayerNormKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { bloomery_gpu::shared_module!(layer_norm_kernels, ctx)? };
        Ok(LayerNormKernels { module })
    }

    /// Enqueue `y = LayerNorm(x)` over `rows` rows of `gain.len()` values. Asynchronous.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<u16>,
        (gain, bias): (&DeviceBuffer<f32>, &DeviceBuffer<f32>),
        eps: f32,
        rows: usize,
        y: &mut DeviceBuffer<u16>,
    ) -> Result<(), GpuError> {
        let what = "LayerNormKernels::enqueue";
        let dim = gain.len();
        if dim == 0
            || !dim.is_multiple_of(2)
            || bias.len() != dim
            || rows == 0
            || x.len() < rows * dim
            || y.len() < rows * dim
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "rows {rows} of dim {dim} (even, non-zero): bias.len() {}, x.len() {}, y.len() {}",
                    bias.len(),
                    x.len(),
                    y.len()
                ),
            });
        }
        let prep = self.module.prepare_vis_layer_norm(LaunchConfig1D::new(
            launch_u32(what, "rows", rows)?,
            THREADS_U32,
            0,
        ))?;
        self.module.vis_layer_norm(
            stream,
            &prep,
            x,
            gain,
            bias,
            eps,
            launch_u32(what, "dim", dim)?,
            launch_u32(what, "rows", rows)?,
            y,
        )?;
        Ok(())
    }
}
