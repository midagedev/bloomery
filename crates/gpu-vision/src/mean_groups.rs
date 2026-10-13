//! The mean of consecutive groups of `g` bf16 rows (`g` = 4 for MiniCPM-V's vit_merger down-sample,
//! where the rows lie in merge order and a group is a 2×2 window): one bf16 output row per group,
//! `out[r] = bf16((((p0 + p1) + p2) + p3) · (1/g))` for the group's rows `p0 .. p3` of `src`.
//!
//! Numeric rule, per value: the bf16 inputs widen exactly to f32; the sum runs in the group's row
//! order with every add rounded on its own (mainline's chain of `ggml_add`); then one multiply by
//! the f32 `1/g` (`0.25` for `g` = 4, exact); then one rounding to bf16, to nearest even. Nothing
//! is contracted: this order is the gate. The rows are the chain's dtype, so the output is the
//! GEMM's residual operand as it stands. [`mean_groups_ref`] is the same expression on the host,
//! and the kernel equals it bit for bit.
//!
//! Geometry: one thread per output value, [`BLOCK`] threads per block, `rows/g · width` threads in
//! all. Adjacent threads read adjacent values of every row of the group, so a warp's loads are
//! whole 64-byte segments; there is no shared state and no barrier. The kernel moves `(g + 1)/g`
//! of the input in bytes, so its time is the card's memory time for that.

use crate::{bf16_f32, f32_bf16};
use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads per block.
pub const BLOCK: u32 = 256;

/// The rule on the host: the means of the consecutive groups of `group` rows of `rows` (bf16 rows
/// of `width` values), in the module's order. `group` divides the row count.
#[must_use]
pub fn mean_groups_ref(rows: &[u16], width: usize, group: usize) -> Vec<u16> {
    let scale = 1.0f32 / group as f32;
    rows.chunks_exact(group * width)
        .flat_map(|g| {
            (0..width).map(move |d| {
                let mut acc = bf16_f32(g[d]);
                for j in 1..group {
                    acc += bf16_f32(g[j * width + d]);
                }
                f32_bf16(acc * scale)
            })
        })
        .collect()
}

#[cuda_module]
mod mean_groups_kernels {
    use super::*;

    /// `out[r·width + d]` = the module rule over `src[(r·group + j)·width + d]`, `j < group`, for
    /// `r < n_out`, `d < width`. One thread per output value.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (src.len() >= n_out * group * width, out.len() >= n_out * width)
    )]
    pub fn vis_mean_groups(
        src: &[u16],
        n_out: u32,
        group: u32,
        width: u32,
        scale: f32,
        mut out: DisjointSlice<u16>,
    ) {
        let idx = thread::index_1d();
        let e = idx.get();
        let w = width as usize;
        if e >= n_out as usize * w {
            return;
        }
        let r = e / w;
        let base = r * group as usize * w + (e - r * w);
        let mut acc = bf16_to_f32(src[base]);
        let mut j = 1usize;
        while j < group as usize {
            acc = add_rn_f32(acc, bf16_to_f32(src[base + j * w]));
            j += 1;
        }
        if let Some(o) = out.get_mut(idx) {
            *o = f32_to_bf16_rne(mul_rn_f32(acc, scale));
        }
    }
}

/// The loaded group-mean module.
pub struct MeanGroupsKernels {
    module: mean_groups_kernels::LoadedModule,
}

impl MeanGroupsKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<MeanGroupsKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { bloomery_gpu::shared_module!(mean_groups_kernels, ctx)? };
        Ok(MeanGroupsKernels { module })
    }

    /// Enqueue the means of the consecutive groups of `group` of the first `n` rows of `src`
    /// (`width` bf16 each) into the first `n / group` rows of `out`. A `group` that does not
    /// divide `n` is refused by name. Asynchronous.
    pub fn enqueue(
        &self,
        stream: &CudaStream,
        n: usize,
        group: usize,
        width: usize,
        src: &DeviceBuffer<u16>,
        out: &mut DeviceBuffer<u16>,
    ) -> Result<(), GpuError> {
        let what = "MeanGroupsKernels::enqueue";
        if group == 0 || n == 0 || !n.is_multiple_of(group) {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "n {n} is not a whole number of groups of {group} rows (n % group must be 0)"
                ),
            });
        }
        let n_out = n / group;
        if width == 0 || src.len() < n * width || out.len() < n_out * width {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "n {n}, group {group}, width {width}: src.len() {}, out.len() {}",
                    src.len(),
                    out.len()
                ),
            });
        }
        // The launch contract multiplies in u32: the input's value count bounds every product.
        launch_u32(what, "n * width", n * width)?;
        let grid = launch_u32(what, "grid", (n_out * width).div_ceil(BLOCK as usize))?;
        let prep = self
            .module
            .prepare_vis_mean_groups(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.vis_mean_groups(
            stream,
            &prep,
            src,
            launch_u32(what, "n_out", n_out)?,
            launch_u32(what, "group", group)?,
            launch_u32(what, "width", width)?,
            1.0f32 / group as f32,
            out,
        )?;
        Ok(())
    }
}
