//! F32 convolution, channel normalization and clipped element operations.

use bloomery_gpu::GpuError;
use cuda_core::LaunchConfig1D;
use cuda_device::convert::{bf16_to_f32, cvt_f16x2_f32, cvt_f32x2_f16x2, f32_to_bf16_rne};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

#[cfg(test)]
mod tests;

#[cuda_module]
pub(crate) mod kernels {
    use super::*;

    /// Clamp a live prefix into a separate buffer.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= n, y.len() >= n))]
    pub fn pointwise(x: &[f32], n: u32, lo: f32, hi: f32, bf16: u32, mut y: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        let value = if x[i].is_finite() {
            x[i].max(lo).min(hi)
        } else {
            x[i]
        };
        let value = if bf16 != 0 {
            bf16_to_f32(f32_to_bf16_rne(value))
        } else {
            value
        };
        // SAFETY: the live prefix is checked by the contract; each thread owns i.
        unsafe {
            *y.get_unchecked_mut(i) = value;
        }
    }

    /// Clamp or SiLU in place over a live prefix.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= n))]
    pub fn unary(n: u32, lo: f32, hi: f32, op: u32, mut x: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: each thread owns i inside the live prefix.
        let p = unsafe { x.get_unchecked_mut(i) };
        *p = if !p.is_finite() {
            *p
        } else if op == 0 {
            (*p).max(lo).min(hi)
        } else {
            *p / (1.0 + (-*p).exp())
        };
    }

    /// Add a repeated channel bias or a scaled residual in place.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= n, other.len() >= stride))]
    pub fn accumulate(other: &[f32], n: u32, stride: u32, scale: f32, mut x: DisjointSlice<f32>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: each thread owns i; stride is positive and bounded by other.len().
        unsafe {
            *x.get_unchecked_mut(i) += scale * other[i % stride as usize];
        }
    }

    /// Stride-2 3x3 convolution, F16 im2col operands and F32 accumulation, channel-fast.
    #[allow(clippy::too_many_arguments, reason = "kernel entry: flat device ABI")]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= time * freq * ci, w.len() >= 9 * ci * co, bias.len() >= co, y.len() >= ot * of * co))]
    pub fn conv2d(
        x: &[f32],
        w: &[f32],
        bias: &[f32],
        time: u32,
        freq: u32,
        ci: u32,
        co: u32,
        ot: u32,
        of: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= (ot * of * co) as usize {
            return;
        }
        let c = i % co as usize;
        let f = i / co as usize % of as usize;
        let t = i / (co * of) as usize;
        let mut sum = 0.0f32;
        for ch in 0..ci as usize {
            for kt in 0..3usize {
                for kf in 0..3usize {
                    let it = 2 * t + kt;
                    let jf = 2 * f + kf;
                    if it > 0 && it <= time as usize && jf > 0 && jf <= freq as usize {
                        let a = x[((it - 1) * freq as usize + jf - 1) * ci as usize + ch];
                        let b = w[kf + 3 * (kt + 3 * (ch + ci as usize * c))];
                        let (a, b) = cvt_f32x2_f16x2(cvt_f16x2_f32(a, b));
                        sum = a.mul_add(b, sum);
                    }
                }
            }
        }
        // SAFETY: i lies in the checked output; each thread owns i.
        unsafe {
            *y.get_unchecked_mut(i) = sum + bias[c];
        }
    }

    /// LayerNorm over channels, followed by ReLU, one thread per spatial row.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= rows * dim, gain.len() >= dim, y.len() >= rows * dim))]
    pub fn channel_norm(
        x: &[f32],
        gain: &[f32],
        rows: u32,
        dim: u32,
        eps: f32,
        fault: bloomery_gpu::FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let r = thread::index_1d().get();
        if r >= rows as usize {
            return;
        }
        let d = dim as usize;
        let mut mean = 0.0f32;
        for j in 0..d {
            mean += x[r * d + j];
        }
        mean /= dim as f32;
        let mut var = 0.0f32;
        for j in 0..d {
            let v = x[r * d + j] - mean;
            var = v.mul_add(v, var);
        }
        let inv = if mean.is_finite() && var.is_finite() {
            1.0 / (var / dim as f32 + eps).sqrt()
        } else {
            fault.raise(bloomery_gpu::FaultSite::F32Product);
            f32::NAN
        };
        for j in 0..d {
            // SAFETY: thread r owns this entire row, inside rows * dim.
            unsafe {
                let value = (x[r * d + j] - mean) * inv * gain[j];
                *y.get_unchecked_mut(r * d + j) = if value.is_finite() {
                    value.max(0.0)
                } else {
                    value
                };
            }
        }
    }

    /// Split a channel row in half, multiplying its first half by sigmoid(second).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= 2 * rows * dim, y.len() >= rows * dim))]
    pub fn glu(
        x: &[f32],
        rows: u32,
        dim: u32,
        fault: bloomery_gpu::FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= (rows * dim) as usize {
            return;
        }
        let j = (i / dim as usize) * 2 * dim as usize + i % dim as usize;
        // SAFETY: the contract covers both halves; each thread writes one output.
        unsafe {
            *y.get_unchecked_mut(i) = if x[j].is_finite() && x[j + dim as usize].is_finite() {
                x[j] / (1.0 + (-x[j + dim as usize]).exp())
            } else {
                fault.raise(bloomery_gpu::FaultSite::F32Product);
                f32::NAN
            };
        }
    }

    /// Causal depthwise convolution with left zero padding and a channel bias.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= rows * dim, w.len() >= taps * dim, bias.len() >= dim, y.len() >= rows * dim))]
    pub fn depthwise(
        x: &[f32],
        w: &[f32],
        bias: &[f32],
        rows: u32,
        dim: u32,
        taps: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= (rows * dim) as usize {
            return;
        }
        let t = i / dim as usize;
        let c = i % dim as usize;
        let mut sum = 0.0f32;
        for k in 0..taps as usize {
            if t + k + 1 >= taps as usize {
                let v = x[(t + k + 1 - taps as usize) * dim as usize + c];
                sum = v.mul_add(w[c * taps as usize + k], sum);
            }
        }
        // SAFETY: each thread owns i inside the checked output.
        unsafe {
            *y.get_unchecked_mut(i) = sum + bias[c];
        }
    }
}

pub(crate) struct Ops {
    pub(crate) module: kernels::LoadedModule,
    pub(crate) fault: Arc<cuda_core::DeviceBuffer<u32>>,
    pub(crate) sink: bloomery_gpu::FaultSink,
}

impl Ops {
    pub(crate) fn load(gpu: &bloomery_gpu::Gpu) -> Result<Self, GpuError> {
        let ctx = gpu.context();
        // SAFETY: this crate owns the embedded bundle and checks launch spans.
        let module = unsafe { bloomery_gpu::shared_module!(kernels, ctx)? };
        Ok(Self {
            module,
            fault: Arc::clone(gpu.fault_word()),
            sink: gpu.unlabelled_sink(),
        })
    }
}

pub(crate) fn config(n: usize) -> LaunchConfig1D {
    LaunchConfig1D::new(
        u32::try_from(n.div_ceil(256)).expect("bounded tower grid"),
        256,
        0,
    )
}
