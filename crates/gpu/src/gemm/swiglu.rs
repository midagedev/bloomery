//! The SwiGLU quantizer between a gate·up pair and its down: the launcher of
//! `gemm_swiglu_quant` (declared in `kernels.rs`).

use super::{GemmAct, GemmKernels};
use crate::fault::FaultSink;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

impl GemmKernels {
    /// Enqueue `act = q8_1(silu(g) · u)` over the first `n_cols` slot
    /// columns (`act.k()` values each, slot-major as the gate and up GEMMs
    /// write them): one launch, the bytes `ElemKernels::enqueue_swiglu` then
    /// [`Gpu::enqueue_quantize_gemm`] leave: a 128-value block holding a
    /// non-finite value is stored refused (NaN scale, zero codes) and raised
    /// on `fault` as [`FaultSite::QuantColumn`]. Asynchronous,
    /// allocation-free, capturable.
    ///
    /// [`Gpu::enqueue_quantize_gemm`]: crate::Gpu::enqueue_quantize_gemm
    /// [`FaultSite::QuantColumn`]: crate::FaultSite::QuantColumn
    pub fn enqueue_swiglu_quant(
        &self,
        stream: &CudaStream,
        g: &DeviceBuffer<f32>,
        u: &DeviceBuffer<f32>,
        n_cols: usize,
        act: &mut GemmAct,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "GemmKernels::enqueue_swiglu_quant";
        if n_cols == 0 || n_cols > act.cols {
            return Err(GpuError::shape(
                what,
                format!("1 <= n_cols <= act.cols() = {}, got {n_cols}", act.cols),
            ));
        }
        if g.len() < n_cols * act.k || u.len() < n_cols * act.k {
            return Err(GpuError::shape(
                what,
                format!(
                    "g.len() {} and u.len() {} need n_cols*k = {n_cols}*{}",
                    g.len(),
                    u.len(),
                    act.k
                ),
            ));
        }
        let n_sb = act.n_sb();
        let grid = launch_u32(what, "grid", n_cols * n_sb * 2)?;
        let n_cols = launch_u32(what, "n_cols", n_cols)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_gemm_swiglu_quant(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.gemm_swiglu_quant(
            stream,
            &prep,
            g,
            u,
            n_cols,
            n_sb,
            n_sb.div_ceil(2),
            n_sb.div_ceil(4),
            &mut act.q3,
            &mut act.q4,
            &mut act.q6,
            &mut act.s8,
            &mut act.d8,
            fault,
        )?;
        Ok(())
    }
}
