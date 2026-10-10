//! GELU with the tanh approximation, in place over bf16 values (`ggml_gelu`, the activation of
//! the Qwen3-VL tower's MLP and of its merger): `0.5·x·(1 + tanh(√(2/π)·x·(1 + 0.044715·x²)))`.
//!
//! Not [`crate::gemm_bf16::Epilogue::Gelu`]: that epilogue is the exact (erf) GELU, a different
//! function; the two differ by up to 4.7e-4 in the range the network's rows take.
//!
//! Numeric rule, per value: `x` widened exactly, the expression of llama.cpp's CUDA `op_gelu` in
//! f32 in its op order — the polynomial `1 + (0.044715·x)·x` contracted to one fused multiply-add
//! as nvcc contracts it, every other op rounded on its own — with libdevice's `tanhf`, and the
//! result rounded once to bf16. [`gelu_tanh_ref`] is the rule on the host with the host's `tanh`,
//! which differs from the device's by ulps; the gate pins the count of values that round apart.

use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne};
use cuda_device::float::{add_rn_f32, fma_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

const BLOCK: u32 = 256;
/// `√(2/π)` as the f32 llama.cpp's `SQRT_2_OVER_PI` rounds to.
const SQRT_2_OVER_PI: f32 = 0.797_884_6;
/// llama.cpp's `GELU_COEF_A`.
const COEF_A: f32 = 0.044_715;

/// The rule on one f32 on the host.
#[must_use]
pub fn gelu_tanh_f32(x: f32) -> f32 {
    let poly = (COEF_A * x).mul_add(x, 1.0);
    let t = (SQRT_2_OVER_PI * x * poly).tanh();
    (0.5 * x) * (1.0 + t)
}

/// The host rule of one value, bf16 bits in and out.
#[must_use]
pub fn gelu_tanh_ref(x: u16) -> u16 {
    crate::f32_bf16(gelu_tanh_f32(crate::bf16_f32(x)))
}

#[cuda_module]
mod gelu_tanh_kernels {
    use super::*;

    /// `x[i] = gelu_tanh(x[i])` for `i < n`, the module rule, in place. One thread per value.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= n))]
    pub fn vis_gelu_tanh(n: u32, mut x: DisjointSlice<u16>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= x.len() (contract); one thread per value.
        let p = unsafe { x.get_unchecked_mut(i) };
        let v = bf16_to_f32(*p);
        let poly = fma_rn_f32(mul_rn_f32(COEF_A, v), v, 1.0);
        let t = mul_rn_f32(mul_rn_f32(SQRT_2_OVER_PI, v), poly).tanh();
        *p = f32_to_bf16_rne(mul_rn_f32(mul_rn_f32(0.5, v), add_rn_f32(1.0, t)));
    }
}

/// The loaded GELU module.
pub struct GeluTanhKernels {
    module: gelu_tanh_kernels::LoadedModule,
}

impl GeluTanhKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<GeluTanhKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { bloomery_gpu::shared_module!(gelu_tanh_kernels, ctx)? };
        Ok(GeluTanhKernels { module })
    }

    /// Enqueue the GELU over the first `n` values of `x`, in place. Asynchronous.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        n: usize,
        x: &mut DeviceBuffer<u16>,
    ) -> Result<(), GpuError> {
        let what = "GeluTanhKernels::enqueue";
        if n == 0 || x.len() < n {
            return Err(GpuError::Shape {
                what,
                detail: format!("n {n} values: x.len() {}", x.len()),
            });
        }
        let grid = launch_u32(what, "grid", n.div_ceil(BLOCK as usize))?;
        let prep = self
            .module
            .prepare_vis_gelu_tanh(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module
            .vis_gelu_tanh(stream, &prep, launch_u32(what, "n", n)?, x)?;
        Ok(())
    }
}
