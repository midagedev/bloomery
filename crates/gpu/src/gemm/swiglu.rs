//! The SwiGLU quantizer between a gate·up pair and its down: the launcher of
//! `gemm_swiglu_quant` (declared in `kernels.rs`), into the GEMM's
//! [`GemmAct`] or a gemv's [`Q8Act`] — the two hold the same q8_1 planes, so
//! one kernel writes either.

use super::act::GemmActPlanesMut;
use super::{GemmAct, GemmKernels};
use crate::fault::FaultSink;
use crate::kquant::Act;
use crate::tensor::Q8Act;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

/// `act`'s five planes as the launch writes them: a `Q8Act` holds the
/// layout [`GemmAct`] documents (the q3/q4/q6 permutations, the 32-value
/// code sums, the 128-value scales), per-column sizes equal.
fn q8_planes(act: &mut Q8Act) -> GemmActPlanesMut<'_> {
    GemmActPlanesMut {
        q3: &mut act.q3,
        q4: &mut act.q4,
        q6: &mut act.q6,
        s8: &mut act.s8,
        d8: &mut act.d8,
    }
}

/// The columns, values per column and planes of the activation one launch
/// fills.
struct Target<'a> {
    cols: usize,
    k: usize,
    planes: GemmActPlanesMut<'a>,
}

impl GemmKernels {
    /// Enqueue `act = q8_1(silu(g) · u)` over the first `n_cols` slot
    /// columns (`act.k()` values each, slot-major as the gate and up GEMMs
    /// write them): [`GemmKernels::enqueue_swiglu_act_quant`] under
    /// [`Act::SiluMul`], the bytes `ElemKernels::enqueue_swiglu` then
    /// [`Gpu::enqueue_quantize_gemm`] leave.
    ///
    /// [`Gpu::enqueue_quantize_gemm`]: crate::Gpu::enqueue_quantize_gemm
    pub fn enqueue_swiglu_quant(
        &self,
        stream: &CudaStream,
        g: &DeviceBuffer<f32>,
        u: &DeviceBuffer<f32>,
        n_cols: usize,
        act: &mut GemmAct,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        self.enqueue_swiglu_act_quant(stream, g, u, Act::SiluMul, n_cols, act, fault)
    }

    /// Enqueue `act = q8_1(rule(g, u))` over the first `n_cols` slot columns
    /// of `act`: one launch of `gemm_swiglu_quant`, `rule` the gate·up rule
    /// of the K-quant family (`kquant::act::apply`: `silu(g) · u` or ik's
    /// clamped SwiGLU, the values of a fused `_sel` gate·up under the same
    /// rule). A 128-value block holding a non-finite `g` or `u` value is
    /// stored refused (NaN scale, zero codes) and raised on `fault` as
    /// [`FaultSite::QuantColumn`] under either rule; a clamp limit that is
    /// not finite is refused by name. Asynchronous, allocation-free,
    /// capturable.
    ///
    /// [`FaultSite::QuantColumn`]: crate::FaultSite::QuantColumn
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
    pub fn enqueue_swiglu_act_quant(
        &self,
        stream: &CudaStream,
        g: &DeviceBuffer<f32>,
        u: &DeviceBuffer<f32>,
        rule: Act,
        n_cols: usize,
        act: &mut GemmAct,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let target = Target {
            cols: act.cols,
            k: act.k,
            planes: act.planes_mut(),
        };
        self.launch(
            "GemmKernels::enqueue_swiglu_act_quant",
            stream,
            (g, u, rule),
            n_cols,
            target,
            fault,
        )
    }

    /// [`GemmKernels::enqueue_swiglu_act_quant`] into the gemv activation
    /// `act` (`act.m()` columns of `act.k()` values): the input of a K-quant
    /// down's `site::kgemv`, whose column `c` holds the bytes a `Q8Act`
    /// quantized from the same values holds.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
    pub fn enqueue_swiglu_act_quant_q8(
        &self,
        stream: &CudaStream,
        g: &DeviceBuffer<f32>,
        u: &DeviceBuffer<f32>,
        rule: Act,
        n_cols: usize,
        act: &mut Q8Act,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let target = Target {
            cols: act.m(),
            k: act.k(),
            planes: q8_planes(act),
        };
        self.launch(
            "GemmKernels::enqueue_swiglu_act_quant_q8",
            stream,
            (g, u, rule),
            n_cols,
            target,
            fault,
        )
    }

    /// The launch both enqueues share, over `n_cols` of `target`'s columns.
    fn launch(
        &self,
        what: &'static str,
        stream: &CudaStream,
        (g, u, rule): (&DeviceBuffer<f32>, &DeviceBuffer<f32>, Act),
        n_cols: usize,
        target: Target<'_>,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let Target { cols, k, planes } = target;
        if n_cols == 0 || n_cols > cols {
            return Err(GpuError::shape(
                what,
                format!("1 <= n_cols <= act cols = {cols}, got {n_cols}"),
            ));
        }
        if let Act::SwigluClamp { limit } = rule
            && !limit.is_finite()
        {
            return Err(GpuError::shape(
                what,
                format!("the clamp limit must be finite, got {limit}"),
            ));
        }
        if g.len() < n_cols * k || u.len() < n_cols * k {
            return Err(GpuError::shape(
                what,
                format!(
                    "g.len() {} and u.len() {} need n_cols*k = {n_cols}*{k}",
                    g.len(),
                    u.len(),
                ),
            ));
        }
        let (act_code, limit) = rule.code();
        let n_sb = k / 256;
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
            act_code,
            limit,
            n_cols,
            n_sb,
            n_sb.div_ceil(2),
            n_sb.div_ceil(4),
            planes.q3,
            planes.q4,
            planes.q6,
            planes.s8,
            planes.d8,
            fault,
        )?;
        Ok(())
    }
}
