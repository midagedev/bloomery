//! The GEMM's q8_1 activations and the quantizer launch that fills them.

use super::GEMM_MAX_SLOTS;
use crate::fault::FaultSink;
use crate::tensor::Q8ACT_MAX_K;
use crate::{Gpu, GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

/// q8_1 activations for up to `cols` columns of `k` values each, in the
/// layout the quantizer writes for a `Q8Act` (the q3/q4/q6 permutations, the
/// 32-value code sums and the 128-value scales), for more columns than a
/// `Q8Act` holds. The GEMM reads the q6 permutation, `s8` and `d8`.
pub struct GemmAct {
    pub(super) q3: DeviceBuffer<u64>,
    pub(super) q4: DeviceBuffer<u32>,
    pub(super) q6: DeviceBuffer<u32>,
    pub(super) s8: DeviceBuffer<i32>,
    pub(super) d8: DeviceBuffer<f32>,
    pub(super) cols: usize,
    pub(super) k: usize,
}

/// A [`GemmAct`]'s five planes borrowed for writing, for a quantizer outside
/// this module that writes the layout [`GemmAct`] documents.
pub(crate) struct GemmActPlanesMut<'a> {
    /// The codes in the q3 pair permutation: `64 * ceil(n_sb/2)` u64 per column.
    pub(crate) q3: &'a mut DeviceBuffer<u64>,
    /// The codes in the q4 permutation: `256 * ceil(n_sb/4)` u32 per column.
    pub(crate) q4: &'a mut DeviceBuffer<u32>,
    /// The codes in the q6 permutation the GEMM stages: `128 * ceil(n_sb/2)` u32 per column.
    pub(crate) q6: &'a mut DeviceBuffer<u32>,
    /// The 32-value code sums: `8 * n_sb` i32 per column.
    pub(crate) s8: &'a mut DeviceBuffer<i32>,
    /// The 128-value block scales: `2 * n_sb` f32 per column.
    pub(crate) d8: &'a mut DeviceBuffer<f32>,
}

impl GemmAct {
    /// Scratch for `cols` (1..=[`GEMM_MAX_SLOTS`]) columns of `k` values, `k`
    /// a multiple of 256 up to the quantizer's cap. Load-time only.
    pub fn new(stream: &CudaStream, cols: usize, k: usize) -> Result<GemmAct, GpuError> {
        let what = "GemmAct::new";
        if !(1..=GEMM_MAX_SLOTS).contains(&cols) {
            return Err(GpuError::shape(
                what,
                format!("1 <= cols <= {GEMM_MAX_SLOTS}, got {cols}"),
            ));
        }
        if !k.is_multiple_of(256) || !(256..=Q8ACT_MAX_K).contains(&k) {
            return Err(GpuError::shape(
                what,
                format!("k must be a multiple of 256 in 256..={Q8ACT_MAX_K}, got {k}"),
            ));
        }
        let n_sb = k / 256;
        Ok(GemmAct {
            q3: DeviceBuffer::zeroed(stream, cols * 64 * n_sb.div_ceil(2))?,
            q4: DeviceBuffer::zeroed(stream, cols * 256 * n_sb.div_ceil(4))?,
            q6: DeviceBuffer::zeroed(stream, cols * 128 * n_sb.div_ceil(2))?,
            s8: DeviceBuffer::zeroed(stream, cols * 8 * n_sb)?,
            d8: DeviceBuffer::zeroed(stream, cols * 2 * n_sb)?,
            cols,
            k,
        })
    }

    /// Columns this scratch holds.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Values per column.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Super-blocks per column (`k / 256`).
    #[must_use]
    pub fn n_sb(&self) -> usize {
        self.k / 256
    }

    /// The 128-value block scales: `2 * n_sb` f32 per column.
    #[must_use]
    pub fn d8(&self) -> &DeviceBuffer<f32> {
        &self.d8
    }

    /// The 32-value code sums: `8 * n_sb` i32 per column.
    #[must_use]
    pub fn s8(&self) -> &DeviceBuffer<i32> {
        &self.s8
    }

    /// The codes in the q3 pair permutation: `64 * ceil(n_sb/2)` u64 per
    /// column.
    #[must_use]
    pub fn q3(&self) -> &DeviceBuffer<u64> {
        &self.q3
    }

    /// The codes in the q4 permutation: `256 * ceil(n_sb/4)` u32 per column.
    #[must_use]
    pub fn q4(&self) -> &DeviceBuffer<u32> {
        &self.q4
    }

    /// The codes in the q6 permutation the GEMM stages: `128 * ceil(n_sb/2)`
    /// u32 per column.
    #[must_use]
    pub fn q6(&self) -> &DeviceBuffer<u32> {
        &self.q6
    }

    /// The five planes, writable, for a quantizer outside this module that
    /// writes the layout above (`gated_quant`).
    pub(crate) fn planes_mut(&mut self) -> GemmActPlanesMut<'_> {
        GemmActPlanesMut {
            q3: &mut self.q3,
            q4: &mut self.q4,
            q6: &mut self.q6,
            s8: &mut self.s8,
            d8: &mut self.d8,
        }
    }

    /// Device bytes of the five planes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.q3.num_bytes()
            + self.q4.num_bytes()
            + self.q6.num_bytes()
            + self.s8.num_bytes()
            + self.d8.num_bytes()
    }
}

impl Gpu {
    /// Enqueue the q8_1 quantization of the first `n_cols` columns of `x`
    /// (`act.k()` f32 each) into `act`, a block holding a non-finite value
    /// stored refused and raised into `fault` — `q3k_quantize_q8_1`, the kernel every gemv's
    /// activations come from, so column `c` holds the bytes a `Q8Act`
    /// quantized from the same values holds. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue_quantize_gemm(
        &self,
        x: &DeviceBuffer<f32>,
        n_cols: usize,
        act: &mut GemmAct,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "enqueue_quantize_gemm";
        if n_cols == 0 || n_cols > act.cols {
            return Err(GpuError::shape(
                what,
                format!("1 <= n_cols <= act.cols() = {}, got {n_cols}", act.cols),
            ));
        }
        if x.len() < n_cols * act.k {
            return Err(GpuError::shape(
                what,
                format!("x.len() {} < n_cols*k = {n_cols}*{}", x.len(), act.k),
            ));
        }
        let n_sb = act.n_sb();
        let grid = launch_u32(what, "grid", n_cols * n_sb * 2)?;
        let m = launch_u32(what, "n_cols", n_cols)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_q3k_quantize_q8_1(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.q3k_quantize_q8_1(
            &self.stream,
            &prep,
            x,
            0,
            m,
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
