//! ReLU squared in place over bf16 values (`ggml_relu_sqr`, the activation of the RADIO
//! projector's MLP): `x` widened exactly, `relu(x) · relu(x)` in f32 (one rounded product), and
//! the result rounded once to bf16. A NaN stays a NaN (`x < 0` is false for it), where ggml's
//! `relu` would turn it into 0: a NaN row is the fault, not a value. [`relu_sq_ref`] is the rule on
//! the host.

use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne};
use cuda_device::float::mul_rn_f32;
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

const BLOCK: u32 = 256;

/// The host rule of one value, bf16 bits in and out.
#[must_use]
pub fn relu_sq_ref(x: u16) -> u16 {
    let x = crate::bf16_f32(x);
    let v = if x < 0.0 { 0.0 } else { x };
    crate::f32_bf16(v * v)
}

#[cuda_module]
mod relu_sq_kernels {
    use super::*;

    /// `x[i] = relu(x[i])²` for `i < n`, the module rule, in place. One thread per value.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), requires = (x.len() >= n))]
    pub fn vis_relu_sq(n: u32, mut x: DisjointSlice<u16>) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        // SAFETY: i < n <= x.len() (contract); one thread per value.
        let p = unsafe { x.get_unchecked_mut(i) };
        let x = bf16_to_f32(*p);
        let v = if x < 0.0 { 0.0 } else { x };
        *p = f32_to_bf16_rne(mul_rn_f32(v, v));
    }
}

/// The loaded module.
pub struct ReluSqKernels {
    module: relu_sq_kernels::LoadedModule,
}

impl ReluSqKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<ReluSqKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { bloomery_gpu::shared_module!(relu_sq_kernels, ctx)? };
        Ok(ReluSqKernels { module })
    }

    /// Enqueue ReLU² over the first `n` values of `x`, in place. Asynchronous.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        n: usize,
        x: &mut DeviceBuffer<u16>,
    ) -> Result<(), GpuError> {
        let what = "ReluSqKernels::enqueue";
        if n == 0 || x.len() < n {
            return Err(GpuError::Shape {
                what,
                detail: format!("n {n} values: x.len() {}", x.len()),
            });
        }
        let grid = launch_u32(what, "grid", n.div_ceil(BLOCK as usize))?;
        let prep = self
            .module
            .prepare_vis_relu_sq(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module
            .vis_relu_sq(stream, &prep, launch_u32(what, "n", n)?, x)?;
        Ok(())
    }
}
