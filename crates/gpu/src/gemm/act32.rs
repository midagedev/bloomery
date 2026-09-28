//! The 32-value-block activations of the Q8_0 and Q5_1 GEMMs, and the two
//! launches that fill them: the quantizer (`quantize_gemm32`) and the
//! SwiGLU quantizer between a gate·up pair and its down (`swiglu_quant32`).
//! Both entries are declared in `kernels32.rs`.

use super::kernels32::Gemm32Kernels;
use super::{GEMM_MAX_SLOTS, GEMM32_STEP};
use crate::fault::FaultSink;
use crate::tensor::Q8ACT_MAX_K;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};

/// q8_1 activations with one scale and one code sum per 32 values, for up to
/// `cols` columns of `k` values each — `q5::Q8Blocks32`'s convention (per
/// 32-value block an f32 scale `d`, an i32 code sum `s` and 32 int8 codes)
/// for as many columns as a GEMM takes, K any multiple of 32.
///
/// Layout, column `c` of `n_steps = ceil(k / 64)` 64-value steps:
/// - `q`: `16 · n_steps` u32 words, the codes in value order (block `b`'s
///   eight words at `8b`, value `4i + j` of the block in byte `j` of word
///   `i`); a column's words past `8 · k/32` are never written;
/// - `d`: `2 · n_steps` f32, block `b`'s scale at `b`;
/// - `s`: `2 · n_steps` i32, block `b`'s code sum at `b`.
///
/// The per-column pad to whole 64-value steps keeps every step's codes one
/// 64-byte run and every step's two scales one 8-byte pair, which the GEMM
/// stages with aligned copies.
///
/// The host remembers how many columns the last quantizer launch wrote
/// ([`GemmAct32::filled`]): a GEMM that would read a column past them is
/// refused by name rather than reading another call's bytes.
pub struct GemmAct32 {
    pub(super) q: DeviceBuffer<u32>,
    pub(super) d: DeviceBuffer<f32>,
    pub(super) s: DeviceBuffer<i32>,
    pub(super) cols: usize,
    pub(super) k: usize,
    /// The column count of the last enqueued quantizer launch; 0 before one.
    pub(super) filled: usize,
}

impl GemmAct32 {
    /// Scratch for `cols` (1..=[`GEMM_MAX_SLOTS`]) columns of `k` values, `k`
    /// a positive multiple of 32 up to the quantizer's cap. Load-time only.
    pub fn new(stream: &CudaStream, cols: usize, k: usize) -> Result<GemmAct32, GpuError> {
        let what = "GemmAct32::new";
        if !(1..=GEMM_MAX_SLOTS).contains(&cols) {
            return Err(GpuError::shape(
                what,
                format!("1 <= cols <= {GEMM_MAX_SLOTS}, got {cols}"),
            ));
        }
        if !k.is_multiple_of(32) || !(32..=Q8ACT_MAX_K).contains(&k) {
            return Err(GpuError::shape(
                what,
                format!("k must be a multiple of 32 in 32..={Q8ACT_MAX_K}, got {k}"),
            ));
        }
        let steps = k.div_ceil(GEMM32_STEP);
        Ok(GemmAct32 {
            q: DeviceBuffer::zeroed(stream, cols * 16 * steps)?,
            d: DeviceBuffer::zeroed(stream, cols * 2 * steps)?,
            s: DeviceBuffer::zeroed(stream, cols * 2 * steps)?,
            cols,
            k,
            filled: 0,
        })
    }

    /// Columns this scratch holds.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Columns the last quantizer launch wrote (0 before one): the most a
    /// GEMM over this scratch may read.
    #[must_use]
    pub fn filled(&self) -> usize {
        self.filled
    }

    /// Values per column.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// 32-value blocks per column (`k / 32`).
    #[must_use]
    pub fn blocks(&self) -> usize {
        self.k / 32
    }

    /// 64-value steps per column (`ceil(k / 64)`): the column strides of the
    /// three planes are `16 · steps` words and `2 · steps` scales and sums.
    #[must_use]
    pub fn steps(&self) -> usize {
        self.k.div_ceil(GEMM32_STEP)
    }

    /// The codes, `16 · steps` words per column (struct doc).
    #[must_use]
    pub fn q(&self) -> &DeviceBuffer<u32> {
        &self.q
    }

    /// The block scales, `2 · steps` per column.
    #[must_use]
    pub fn d(&self) -> &DeviceBuffer<f32> {
        &self.d
    }

    /// The block code sums, `2 · steps` per column.
    #[must_use]
    pub fn s(&self) -> &DeviceBuffer<i32> {
        &self.s
    }

    /// Device bytes of the three planes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.q.num_bytes() + self.d.num_bytes() + self.s.num_bytes()
    }

    /// The launch shape both quantizers take for `n_cols` columns: the grid
    /// (one 32-thread block per column and group of four 32-value blocks),
    /// then `n_cols`, the blocks, the groups and the steps per column, each
    /// as the kernel's u32.
    fn quant_launch(
        &self,
        what: &'static str,
        n_cols: usize,
    ) -> Result<(u32, u32, u32, u32, u32), GpuError> {
        if n_cols == 0 || n_cols > self.cols {
            return Err(GpuError::shape(
                what,
                format!("1 <= n_cols <= act.cols() = {}, got {n_cols}", self.cols),
            ));
        }
        let groups = self.blocks().div_ceil(4);
        Ok((
            launch_u32(what, "grid", n_cols * groups)?,
            launch_u32(what, "n_cols", n_cols)?,
            launch_u32(what, "blocks", self.blocks())?,
            launch_u32(what, "groups", groups)?,
            launch_u32(what, "steps", self.steps())?,
        ))
    }
}

impl Gemm32Kernels {
    /// Enqueue the 32-value quantization of the first `n_cols` columns of `x`
    /// (`act.k()` f32 each, column after column) into `act`: per block
    /// `d = amax/127` (1.0 for an all-zero block), codes `round(x/d)`
    /// clamped to ±127 and their sum — `q5::Q8Blocks32`'s rule. A block
    /// holding a non-finite value is stored refused (NaN scale, zero codes,
    /// zero sum) and raised on `fault` as [`FaultSite::QuantColumn`]. Column
    /// `c`'s bytes are a function of column `c`'s values alone.
    /// Asynchronous, allocation-free, capturable.
    ///
    /// [`FaultSite::QuantColumn`]: crate::FaultSite::QuantColumn
    pub fn enqueue_quantize_gemm32(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        n_cols: usize,
        act: &mut GemmAct32,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_quantize_gemm32";
        let (grid, m, blocks, groups, steps) = act.quant_launch(what, n_cols)?;
        if x.len() < n_cols * act.k {
            return Err(GpuError::shape(
                what,
                format!("x.len() {} < n_cols*k = {n_cols}*{}", x.len(), act.k),
            ));
        }
        let prep = self
            .module
            .prepare_quantize_gemm32(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.quantize_gemm32(
            stream, &prep, x, m, blocks, groups, steps, &mut act.q, &mut act.d, &mut act.s, fault,
        )?;
        act.filled = n_cols;
        Ok(())
    }

    /// Enqueue `act = q8_1_32(silu(g) · u)` over the first `n_cols` slot
    /// columns (`act.k()` values each, slot-major as the gate and up GEMMs
    /// write them): one launch, the bytes `ElemKernels::enqueue_swiglu` then
    /// [`Gemm32Kernels::enqueue_quantize_gemm32`] leave, a non-finite block
    /// refused and raised as that launch refuses and raises it. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_swiglu_quant32(
        &self,
        stream: &CudaStream,
        g: &DeviceBuffer<f32>,
        u: &DeviceBuffer<f32>,
        n_cols: usize,
        act: &mut GemmAct32,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_swiglu_quant32";
        let (grid, m, blocks, groups, steps) = act.quant_launch(what, n_cols)?;
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
        let prep = self
            .module
            .prepare_swiglu_quant32(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.swiglu_quant32(
            stream, &prep, g, u, m, blocks, groups, steps, &mut act.q, &mut act.d, &mut act.s,
            fault,
        )?;
        act.filled = n_cols;
        Ok(())
    }
}

/// One warp's group of four 32-value blocks of column `col` (values
/// `128·g ..`), the body both quantizer entries run: lane ℓ holds the four
/// values `v` of word ℓ, its octet `ℓ >> 3` is block `b = 4g + (ℓ >> 3)`, and
/// the block's scale and sum are 1-2-4 xor butterflies inside the octet
/// (`q5::q5_quant_group`'s geometry). A lane of a block at or past `blocks`
/// holds the last block's values, joins every shuffle and stores nothing.
/// Per block: `d = amax/127` (1.0 for an all-zero block), codes and their sum
/// by `cores::q8_quad`; a block holding a non-finite value is stored refused
/// — NaN scale, zero codes, zero sum — and the return is `true` on its lanes.
///
/// # Safety
///
/// All 32 lanes of the warp call it with one `(col, g)`; `col` is below the
/// launch's columns and `g` below `ceil(blocks / 4)`; `q`, `d`, `s` hold
/// `16 · steps`, `2 · steps`, `2 · steps` entries per column for every
/// column of the launch, with `2 · steps >= blocks`.
#[allow(
    clippy::too_many_arguments,
    reason = "device core: it is handed a kernel entry's flat arguments (rust-quality R8)"
)]
#[inline(always)]
pub(super) unsafe fn quant32_group(
    v: [f32; 4],
    col: usize,
    g: usize,
    blocks: usize,
    steps: usize,
    lane: usize,
    q: &mut cuda_device::DisjointSlice<u32>,
    d: &mut cuda_device::DisjointSlice<f32>,
    s: &mut cuda_device::DisjointSlice<i32>,
) -> bool {
    use cuda_device::warp;
    let oct = lane >> 3;
    let i = lane & 7;
    let b = 4 * g + oct;
    let bad = warp::ballot(!crate::fault::quad_finite(v));
    let refused = (bad >> (8 * oct)) & 0xff != 0;
    let mut amax = v[0].abs().max(v[1].abs()).max(v[2].abs()).max(v[3].abs());
    amax = amax.max(warp::shuffle_xor_f32(amax, 1));
    amax = amax.max(warp::shuffle_xor_f32(amax, 2));
    amax = amax.max(warp::shuffle_xor_f32(amax, 4));
    let dv = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let (word, quad) = crate::cores::q8_quad(v, dv);
    let mut sum = quad;
    sum += warp::shuffle_xor(sum as u32, 1) as i32;
    sum += warp::shuffle_xor(sum as u32, 2) as i32;
    sum += warp::shuffle_xor(sum as u32, 4) as i32;
    let (word, sum, dv) = if refused {
        (0, 0, f32::NAN)
    } else {
        (word, sum, dv)
    };
    if b < blocks {
        // SAFETY: b < blocks <= 2·steps, so word 8b + i < 16·steps and entry
        // b < 2·steps of column col, inside the planes by this fn's contract;
        // one lane writes each word, lane 0 of the octet each scale and sum.
        unsafe {
            *q.get_unchecked_mut(col * 16 * steps + 8 * b + i) = word;
            if i == 0 {
                *d.get_unchecked_mut(col * 2 * steps + b) = dv;
                *s.get_unchecked_mut(col * 2 * steps + b) = sum;
            }
        }
    }
    refused && b < blocks
}
