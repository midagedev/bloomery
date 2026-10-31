//! The 32-value-block activations of the Q8_0 and Q5_1 GEMMs, and the five
//! launches that fill them: the quantizer (`quantize_gemm32`), the SwiGLU
//! quantizer between a gate·up pair and its down (`swiglu_quant32`), its
//! card-slots-only form over a remapped route (`swiglu_quant32_sel`), its
//! form under a gate·up rule of `kquant::act`, ik's clamped SwiGLU among
//! them (`swiglu_act_quant32`), and the squared-ReLU quantizer between a
//! relu² expert's up and its down (`relu_sqr_quant32`); and the forms of
//! the same columns a site on the dense MMA tile reads ([`MmaForm`]: BF16,
//! `bf16_cols`; MXFP8, `mxfp8_cols`). The entries are declared in
//! `kernels32.rs`, the native MXFP8 one in `nvfp4mma.rs`.

use super::f32tile::FORM_THREADS;
#[cfg(not(bloomery_nvfp4_mma))]
use super::f32tile::NATIVE_BUILD;
use super::kernels32::Gemm32Kernels;
use super::{GEMM_MAX_SLOTS, GEMM32_STEP};
use crate::fault::FaultSink;
use crate::kquant::Act;
use crate::tensor::Q8ACT_MAX_K;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::vector::U32x4;
use gguf::GgmlType;

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
///
/// A scratch made [`GemmAct32::with_form`] also holds room for its columns
/// in a [`MmaForm`], which a site on the dense MMA tile copies instead of
/// converting the f32 rows in each tile.
pub struct GemmAct32 {
    #[cfg(bloomery_nvfp4_mma)]
    pub(super) native: Option<super::nvfp4mma::Act>,
    pub(super) q: DeviceBuffer<u32>,
    pub(super) d: DeviceBuffer<f32>,
    pub(super) s: DeviceBuffer<i32>,
    /// Whether the scratch holds the Q8 planes: false for a native input
    /// ([`GemmAct32::new_native_input`]) and a fused gate and up output
    /// ([`GemmAct32::new_gate_up_output`]), whose planes are empty.
    pub(super) q8_required: bool,
    pub(super) cols: usize,
    pub(super) k: usize,
    /// The column count of the last enqueued quantizer launch; 0 before one.
    filled: usize,
    /// Room for each [`MmaForm`], at its index.
    forms: [Option<ColsForm>; 2],
}

/// A form of a [`GemmAct32`]'s columns that a site on the dense MMA tile
/// copies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmaForm {
    /// BF16 words, two values a word in increasing order, each value rounded
    /// as the tile rounds it (`f32tile::Bf16Cols`): a BF16 site's.
    Bf16,
    /// E4M3 codes, four values a word in increasing order, with one UE8M0
    /// scale a 32-value block, four a word (`nvfp4mma::Mxfp8Cols`): an MXFP8
    /// site's on the native W8A8 product.
    Mxfp8,
}

/// A [`MmaForm`] of a [`GemmAct32`]'s columns: `words`, `k / 2` (BF16) or
/// `k / 4` (MXFP8) a column; `scales`, MXFP8's `k / 128` a column and empty
/// for BF16; and a word a column that is nonzero when one of its f32 values
/// falls outside the tensor cores' range.
pub(super) struct ColsForm {
    pub(super) words: DeviceBuffer<u32>,
    pub(super) scales: DeviceBuffer<u32>,
    pub(super) refused: DeviceBuffer<u32>,
    /// The f32 rows' address and the columns the last launch of this form
    /// read; `None` before one and after a quantizer launch.
    pub(super) of: Option<(u64, usize)>,
}

impl GemmAct32 {
    /// Scratch for `cols` (1..=[`GEMM_MAX_SLOTS`]) columns of `k` values, `k`
    /// a positive multiple of 32 up to the quantizer's cap. Load-time only.
    pub fn new(stream: &CudaStream, cols: usize, k: usize) -> Result<GemmAct32, GpuError> {
        Self::with_planes(stream, cols, k, true)
    }

    /// [`GemmAct32::new`]'s scratch, its Q8 planes empty unless `q8` (a Q8
    /// reader and a launch that stores them are then refused by name).
    pub(crate) fn with_planes(
        stream: &CudaStream,
        cols: usize,
        k: usize,
        q8: bool,
    ) -> Result<GemmAct32, GpuError> {
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
        // The 64-value steps of every column the planes hold.
        let steps = if q8 {
            cols * k.div_ceil(GEMM32_STEP)
        } else {
            0
        };
        Ok(GemmAct32 {
            #[cfg(bloomery_nvfp4_mma)]
            native: None,
            q: DeviceBuffer::zeroed(stream, 16 * steps)?,
            d: DeviceBuffer::zeroed(stream, 2 * steps)?,
            s: DeviceBuffer::zeroed(stream, 2 * steps)?,
            q8_required: q8,
            cols,
            k,
            filled: 0,
            forms: [None, None],
        })
    }

    /// This scratch with room for its columns in `form` as well
    /// ([`Gemm32Kernels::enqueue_bf16_cols`], `enqueue_mxfp8_cols`). MXFP8
    /// takes `k` a multiple of 128 (its scale words); another is a named
    /// `Shape` error. Load-time only.
    pub fn with_form(mut self, stream: &CudaStream, form: MmaForm) -> Result<GemmAct32, GpuError> {
        let (cols, k) = (self.cols, self.k);
        let (words, scales) = match form {
            MmaForm::Bf16 => (cols * k / 2, 0),
            MmaForm::Mxfp8 if k.is_multiple_of(128) => (cols * k / 4, cols * k / 128),
            MmaForm::Mxfp8 => {
                return Err(GpuError::shape(
                    "GemmAct32::with_form",
                    format!("an MXFP8 form takes k a multiple of 128, got {k}"),
                ));
            }
        };
        self.forms[form as usize] = Some(ColsForm {
            words: DeviceBuffer::zeroed(stream, words)?,
            scales: DeviceBuffer::zeroed(stream, scales)?,
            refused: DeviceBuffer::zeroed(stream, cols)?,
            of: None,
        });
        Ok(self)
    }

    /// `form` when its last launch read the first `n` columns or more of the
    /// f32 rows `x` and no quantizer launch came after it.
    pub(super) fn form_of(
        &self,
        form: MmaForm,
        x: &DeviceBuffer<f32>,
        n: usize,
    ) -> Option<&ColsForm> {
        self.forms[form as usize]
            .as_ref()
            .filter(|f| f.of.is_some_and(|(at, m)| at == x.cu_deviceptr() && n <= m))
    }

    /// `form`'s room for the first `n` columns of `k` values, as `what`'s: a
    /// scratch of another K or fewer columns is a named `Shape` error, one
    /// made without room for `form` a named `State` error.
    pub(super) fn room(
        &mut self,
        what: &'static str,
        form: MmaForm,
        k: usize,
        n: usize,
    ) -> Result<&mut ColsForm, GpuError> {
        if k != self.k {
            return Err(GpuError::shape(
                what,
                format!("the scratch holds columns of {} values, not {k}", self.k),
            ));
        }
        if n == 0 || n > self.cols {
            return Err(GpuError::shape(
                what,
                format!("1 <= n_cols <= act.cols() = {}, got {n}", self.cols),
            ));
        }
        self.forms[form as usize]
            .as_mut()
            .ok_or_else(|| GpuError::state(what, "a scratch made with_form for this form"))
    }

    /// `form`'s words, scale words and refusal words for an epilogue that
    /// writes it of the first `n` columns of `k` values ([`Self::room`]'s
    /// checks, as `what`'s), the refusal words zeroed on `stream` for the
    /// epilogue's pieces to OR. The form holds no rows' until
    /// [`Self::wrote`] marks it after the launch.
    pub(crate) fn out_planes(
        &mut self,
        stream: &CudaStream,
        what: &'static str,
        form: MmaForm,
        (k, n): (usize, usize),
    ) -> Result<[&mut DeviceBuffer<u32>; 3], GpuError> {
        let f = self.room(what, form, k, n)?;
        f.of = None;
        f.refused.zero_async(stream)?;
        Ok([&mut f.words, &mut f.scales, &mut f.refused])
    }

    /// Marks `form` as an epilogue's of the first `n` columns of the f32 rows
    /// at device address `at`, and the other form as another rows'.
    pub(crate) fn wrote(&mut self, form: MmaForm, at: u64, n: usize) {
        if let Some(f) = &mut self.forms[form as usize] {
            f.of = Some((at, n));
        }
        self.forget(match form {
            MmaForm::Bf16 => MmaForm::Mxfp8,
            MmaForm::Mxfp8 => MmaForm::Bf16,
        });
    }

    /// Whether a site can read `form` of the first `n` columns of `x`
    /// ([`GemmAct32::form_of`]).
    #[must_use]
    pub(crate) fn holds(&self, form: MmaForm, x: &DeviceBuffer<f32>, n: usize) -> bool {
        self.form_of(form, x, n).is_some()
    }

    /// Whether this scratch holds room for `form` ([`GemmAct32::with_form`]).
    #[must_use]
    pub(crate) fn has(&self, form: MmaForm) -> bool {
        self.forms[form as usize].is_some()
    }

    /// Marks `form` as no rows' (its columns stay as they are), so no site
    /// reads it until its next launch.
    pub(crate) fn forget(&mut self, form: MmaForm) {
        if let Some(f) = &mut self.forms[form as usize] {
            f.of = None;
        }
    }

    /// Records a quantizer launch of `n_cols` columns, which leaves every
    /// form of other rows.
    pub(super) fn fill(&mut self, n_cols: usize) {
        self.filled = n_cols;
        self.forget(MmaForm::Bf16);
        self.forget(MmaForm::Mxfp8);
    }

    /// Routed columns, including the native FP4 representation when the build supports it.
    pub fn new_routed(stream: &CudaStream, cols: usize, k: usize) -> Result<Self, GpuError> {
        let act = Self::new(stream, cols, k)?;
        #[cfg(bloomery_nvfp4_mma)]
        let act = {
            let mut act = act;
            if k.is_multiple_of(64) {
                act.native = Some(super::nvfp4mma::Act::new(stream, cols, k)?);
            }
            act
        };
        Ok(act)
    }

    /// Scratch for the resident routed formats. Other formats keep only their Q8 representation.
    pub fn new_routed_for(
        stream: &CudaStream,
        cols: usize,
        k: usize,
        types: impl IntoIterator<Item = GgmlType>,
    ) -> Result<Self, GpuError> {
        if types.into_iter().any(|ty| ty == GgmlType::NVFP4) {
            Self::new_routed(stream, cols, k)
        } else {
            Self::new(stream, cols, k)
        }
    }

    /// Routed input columns every reader takes in the native FP4 form: that
    /// form alone on a native FP4 build at `k` a multiple of 64, its Q8
    /// planes empty (a Q8 reader and a launch that stores them are refused
    /// by name); [`GemmAct32::new`]'s scratch otherwise. Load-time only.
    pub(crate) fn new_native_input(
        stream: &CudaStream,
        cols: usize,
        k: usize,
    ) -> Result<Self, GpuError> {
        #[cfg(bloomery_nvfp4_mma)]
        if k.is_multiple_of(64) {
            let mut act = Self::with_planes(stream, cols, k, false)?;
            act.native = Some(super::nvfp4mma::Act::new(stream, cols, k)?);
            return Ok(act);
        }
        Self::new(stream, cols, k)
    }

    /// The down's input columns the fused NVFP4 gate and up launch writes
    /// (`Gemm32Kernels::enqueue_gate_up_sel32`): [`GemmAct32::new_native_input`]'s
    /// scratch, with the frames the launch keeps beside the native FP4 form
    /// where it has one. Load-time only.
    pub fn new_gate_up_output(
        stream: &CudaStream,
        cols: usize,
        k: usize,
    ) -> Result<Self, GpuError> {
        let act = Self::new_native_input(stream, cols, k)?;
        #[cfg(bloomery_nvfp4_mma)]
        let act = Self {
            native: act.native.map(|n| n.for_gate_up(stream)).transpose()?,
            ..act
        };
        Ok(act)
    }

    /// The native FP4 form's codes (`k / 8` words a column), block scale
    /// words (`k / 64` a column) and column scales; `None` on a scratch
    /// without the form.
    #[must_use]
    pub fn fp4_planes(
        &self,
    ) -> Option<(&DeviceBuffer<u32>, &DeviceBuffer<u32>, &DeviceBuffer<f32>)> {
        #[cfg(bloomery_nvfp4_mma)]
        {
            self.native.as_ref().map(super::nvfp4mma::Act::planes)
        }
        #[cfg(not(bloomery_nvfp4_mma))]
        None
    }

    /// Whether the scratch holds the native FP4 form beside the 32-value one
    /// ([`GemmAct32::new_routed`] on a native FP4 build).
    #[must_use]
    pub fn has_native_form(&self) -> bool {
        #[cfg(bloomery_nvfp4_mma)]
        {
            self.native.is_some()
        }
        #[cfg(not(bloomery_nvfp4_mma))]
        false
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

    /// The codes, `16 · steps` words per column (struct doc); empty on a
    /// native input ([`GemmAct32::new_native_input`]).
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

    /// Device bytes of the stored activation representations.
    #[must_use]
    pub fn bytes(&self) -> usize {
        let bytes = self.q.num_bytes()
            + self.d.num_bytes()
            + self.s.num_bytes()
            + self
                .forms
                .iter()
                .flatten()
                .map(|f| f.words.num_bytes() + f.scales.num_bytes() + f.refused.num_bytes())
                .sum::<usize>();
        #[cfg(bloomery_nvfp4_mma)]
        let bytes = bytes + self.native.as_ref().map_or(0, super::nvfp4mma::Act::bytes);
        bytes
    }

    /// The launch shape a quantizer that stores the Q8 planes takes for
    /// `n_cols` columns ([`GemmAct32::input_launch`]); a scratch that holds
    /// none ([`GemmAct32::new_native_input`],
    /// [`GemmAct32::new_gate_up_output`]) is refused by name.
    pub(super) fn quant_launch(
        &self,
        what: &'static str,
        n_cols: usize,
    ) -> Result<(u32, u32, u32, u32, u32), GpuError> {
        if !self.q8_required {
            return Err(GpuError::state(
                what,
                "a scratch with Q8 planes; a native FP4 scratch holds none",
            ));
        }
        self.input_launch(what, n_cols)
    }

    /// The launch shape the quantizers take for `n_cols` columns: the grid
    /// (one 32-thread block per column and group of four 32-value blocks),
    /// then `n_cols`, the blocks, the groups and the steps per column, each
    /// as the kernel's u32. The input quantizer's
    /// ([`Gemm32Kernels::enqueue_quantize_gemm32`]), which stores the Q8
    /// planes only where the scratch holds them.
    fn input_launch(
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

/// The checks a form launch over the first `n_cols` columns of `x` into
/// `act`'s `form` shares, as `what`'s: the columns, `x`'s length and
/// 16-byte alignment, and room for the form; then the launch (a warp a
/// column) and its `k` and `n_cols` as the kernel's u32.
fn form_launch<'a>(
    what: &'static str,
    x: &DeviceBuffer<f32>,
    n_cols: usize,
    act: &'a mut GemmAct32,
    form: MmaForm,
) -> Result<(LaunchConfig1D, u32, u32, &'a mut ColsForm), GpuError> {
    let k_values = act.k;
    let f = act.room(what, form, k_values, n_cols)?;
    if x.len() < n_cols * k_values {
        return Err(GpuError::shape(
            what,
            format!("x.len() {} < n_cols*k = {n_cols}*{k_values}", x.len()),
        ));
    }
    crate::tensor::check_alignment::<U32x4, _>(what, "x", x)?;
    let k = launch_u32(what, "k", k_values)?;
    let n = launch_u32(what, "n_cols", n_cols)?;
    let warps = (FORM_THREADS / 32) as usize;
    let grid = launch_u32(what, "grid", n_cols.div_ceil(warps))?;
    Ok((LaunchConfig1D::new(grid, FORM_THREADS, 0), k, n, f))
}

impl Gemm32Kernels {
    /// Enqueue the BF16 form of the first `n_cols` columns of `x` (`act.k()`
    /// f32 each, column after column) into `act`, made with room for it
    /// ([`GemmAct32::with_form`]): each value rounded to nearest even as the
    /// dense MMA tile rounds it, and each column's refusal word. The q8
    /// blocks stay as they are. Asynchronous, allocation-free, capturable.
    pub fn enqueue_bf16_cols(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        n_cols: usize,
        act: &mut GemmAct32,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_bf16_cols";
        let (cfg, k, n, form) = form_launch(what, x, n_cols, act, MmaForm::Bf16)?;
        let prep = self.module.prepare_bf16_cols(cfg)?;
        self.module
            .bf16_cols(stream, &prep, x, k, n, &mut form.words, &mut form.refused)?;
        form.of = Some((x.cu_deviceptr(), n_cols));
        Ok(())
    }

    /// Enqueue the MXFP8 form of the first `n_cols` columns of `x` (`act.k()`
    /// f32 each, column after column) into `act`, made with room for it
    /// ([`GemmAct32::with_form`]): each 32-value block's scale
    /// `2^ceil(log2(amax / 448))` as a UE8M0 byte, its values over that scale
    /// as E4M3 rounded to nearest even and saturated, and each column's
    /// refusal word. The q8 blocks stay as they are. A build without the
    /// native block-scaled kernels refuses it by name. Asynchronous,
    /// allocation-free, capturable.
    #[cfg_attr(
        not(bloomery_nvfp4_mma),
        allow(unused_variables, reason = "the native build's launch reads them")
    )]
    pub fn enqueue_mxfp8_cols(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        n_cols: usize,
        act: &mut GemmAct32,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_mxfp8_cols";
        let (cfg, k, n, form) = form_launch(what, x, n_cols, act, MmaForm::Mxfp8)?;
        #[cfg(bloomery_nvfp4_mma)]
        {
            self.native.mxfp8_cols(stream, cfg, x, (k, n), form)?;
            form.of = Some((x.cu_deviceptr(), n_cols));
            Ok(())
        }
        #[cfg(not(bloomery_nvfp4_mma))]
        Err(GpuError::state(what, NATIVE_BUILD))
    }

    /// Enqueue `form` of the first `n_cols` columns of `x` into `act`:
    /// [`Self::enqueue_bf16_cols`] or [`Self::enqueue_mxfp8_cols`].
    pub fn enqueue_form_cols(
        &self,
        stream: &CudaStream,
        form: MmaForm,
        x: &DeviceBuffer<f32>,
        n_cols: usize,
        act: &mut GemmAct32,
    ) -> Result<(), GpuError> {
        match form {
            MmaForm::Bf16 => self.enqueue_bf16_cols(stream, x, n_cols, act),
            MmaForm::Mxfp8 => self.enqueue_mxfp8_cols(stream, x, n_cols, act),
        }
    }

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
        let (grid, m, blocks, groups, steps) = act.input_launch(what, n_cols)?;
        if x.len() < n_cols * act.k {
            return Err(GpuError::shape(
                what,
                format!("x.len() {} < n_cols*k = {n_cols}*{}", x.len(), act.k),
            ));
        }
        #[cfg(bloomery_nvfp4_mma)]
        if act.native.is_some() {
            self.native.quantize_input(stream, x, m, act, fault)?;
            act.fill(n_cols);
            return Ok(());
        }
        let prep = self
            .module
            .prepare_quantize_gemm32(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.quantize_gemm32(
            stream, &prep, x, m, blocks, groups, steps, &mut act.q, &mut act.d, &mut act.s, fault,
        )?;
        act.fill(n_cols);
        Ok(())
    }

    /// Enqueue the 32-value quantization of `max(v, 0)²` of the first
    /// `n_cols` columns of `x` (`act.k()` f32 each, column after column) into
    /// `act`: one launch, the bytes `ElemKernels::enqueue_relu_sqr` then
    /// [`Gemm32Kernels::enqueue_quantize_gemm32`] leave, a non-finite block
    /// refused and raised as that launch refuses and raises it (a NaN or a
    /// negative infinity is not turned into 0 first). The input of a relu²
    /// expert's down GEMM. Asynchronous, allocation-free, capturable.
    pub fn enqueue_quantize_gemm32_relu_sqr(
        &self,
        stream: &CudaStream,
        x: &DeviceBuffer<f32>,
        n_cols: usize,
        act: &mut GemmAct32,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_quantize_gemm32_relu_sqr";
        let (grid, m, blocks, groups, steps) = act.quant_launch(what, n_cols)?;
        if x.len() < n_cols * act.k {
            return Err(GpuError::shape(
                what,
                format!("x.len() {} < n_cols*k = {n_cols}*{}", x.len(), act.k),
            ));
        }
        let prep = self
            .module
            .prepare_relu_sqr_quant32(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.relu_sqr_quant32(
            stream, &prep, x, m, blocks, groups, steps, &mut act.q, &mut act.d, &mut act.s, fault,
        )?;
        act.fill(n_cols);
        #[cfg(bloomery_nvfp4_mma)]
        self.native.quantize(stream, act)?;
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
        #[cfg(bloomery_nvfp4_mma)]
        if act.native.is_some() {
            self.native.swiglu(
                stream,
                super::nvfp4mma::SwigluArgs {
                    g,
                    u,
                    selected: None,
                    rule: Act::SiluMul,
                    cols: m,
                    act,
                },
                fault,
            )?;
            act.fill(n_cols);
            return Ok(());
        }
        let prep = self
            .module
            .prepare_swiglu_quant32(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.swiglu_quant32(
            stream, &prep, g, u, m, blocks, groups, steps, &mut act.q, &mut act.d, &mut act.s,
            fault,
        )?;
        act.fill(n_cols);
        Ok(())
    }

    /// Enqueue `act = q8_1_32(rule(g, u))` over the first `n_cols` slot
    /// columns (`swiglu_act_quant32`, declared in `kernels32.rs`):
    /// [`Gemm32Kernels::enqueue_swiglu_quant32`] with the gate·up rule
    /// `rule` (`kquant::act::apply`) in place of `silu(g) · u` — ik's
    /// clamped SwiGLU or the OAI SwiGLU for a feed-forward block its model
    /// clamps. A non-finite `g` or `u` value refuses its block and raises
    /// [`FaultSite::QuantColumn`] under any rule (a clamp would carry it to a
    /// finite value); [`Act::SiluMul`] writes `enqueue_swiglu_quant32`'s
    /// bytes. A rule [`Act::code`] refuses is refused by name. Asynchronous,
    /// allocation-free, capturable.
    ///
    /// [`FaultSite::QuantColumn`]: crate::FaultSite::QuantColumn
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
    pub fn enqueue_swiglu_act_quant32(
        &self,
        stream: &CudaStream,
        g: &DeviceBuffer<f32>,
        u: &DeviceBuffer<f32>,
        rule: Act,
        n_cols: usize,
        act: &mut GemmAct32,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_swiglu_act_quant32";
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
        let (code, limit) = rule.code(what)?;
        #[cfg(bloomery_nvfp4_mma)]
        if act.native.is_some() {
            self.native.swiglu(
                stream,
                super::nvfp4mma::SwigluArgs {
                    g,
                    u,
                    selected: None,
                    rule,
                    cols: m,
                    act,
                },
                fault,
            )?;
            act.fill(n_cols);
            return Ok(());
        }
        let prep = self
            .module
            .prepare_swiglu_act_quant32(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.swiglu_act_quant32(
            stream, &prep, g, u, code, limit, m, blocks, groups, steps, &mut act.q, &mut act.d,
            &mut act.s, fault,
        )?;
        act.fill(n_cols);
        Ok(())
    }

    /// Enqueue `act = q8_1_32(silu(g) · u)` over the first `n_cols` slot
    /// columns a remapped route left the card — the slots whose place
    /// `sel[0..n_cols]` is below `n_card` (`swiglu_quant32_sel`, declared in
    /// `kernels32.rs`): a host slot's column (`crate::hybrid::HOST`) is
    /// left exactly as it stands — a GEMM over the route reads only the
    /// listed (card) columns — and a place in `[n_card, HOST)` raises
    /// [`FaultSite::ExpertId`] on `fault` with nothing stored. The written
    /// columns' bytes are [`Gemm32Kernels::enqueue_swiglu_quant32`]'s.
    /// Asynchronous, allocation-free, capturable; the scratch's filled count
    /// is `n_cols`, the most a GEMM over the same route may read.
    ///
    /// [`FaultSite::ExpertId`]: crate::FaultSite::ExpertId
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher; folding these into a *Args struct is the R8 round"
    )]
    pub fn enqueue_swiglu_quant32_sel(
        &self,
        stream: &CudaStream,
        g: &DeviceBuffer<f32>,
        u: &DeviceBuffer<f32>,
        sel: &DeviceBuffer<u32>,
        n_card: usize,
        n_cols: usize,
        act: &mut GemmAct32,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_swiglu_quant32_sel";
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
        if sel.len() < n_cols {
            return Err(GpuError::shape(
                what,
                format!("sel.len() {} < n_cols {n_cols}", sel.len()),
            ));
        }
        let n_card = launch_u32(what, "n_card", n_card)?;
        #[cfg(bloomery_nvfp4_mma)]
        if act.native.is_some() {
            self.native.swiglu(
                stream,
                super::nvfp4mma::SwigluArgs {
                    g,
                    u,
                    selected: Some((sel, n_card)),
                    rule: Act::SiluMul,
                    cols: m,
                    act,
                },
                fault,
            )?;
            act.fill(n_cols);
            return Ok(());
        }
        let prep = self
            .module
            .prepare_swiglu_quant32_sel(LaunchConfig1D::new(grid, 32, 0))?;
        self.module.swiglu_quant32_sel(
            stream, &prep, g, u, sel, n_card, m, blocks, groups, steps, &mut act.q, &mut act.d,
            &mut act.s, fault,
        )?;
        act.fill(n_cols);
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
    let i = lane & 7;
    let b = 4 * g + (lane >> 3);
    let (word, sum, dv, refused) = quant32_lane(v, lane);
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

/// Lane ℓ's share of its octet's 32-value block, the four values `v` of
/// word `ℓ & 7`: the block's rule as [`quant32_group`] states it, returned
/// as `(word, sum, d, refused)` — the lane's code word, the block's code sum
/// and scale (the same on every lane of the octet) and whether the block
/// holds a non-finite value (then `(0, 0, NaN, true)`). All 32 lanes of the
/// warp call it; each octet is one block.
#[inline(always)]
pub(super) fn quant32_lane(v: [f32; 4], lane: usize) -> (u32, i32, f32, bool) {
    use cuda_device::warp;
    let bad = warp::ballot(!crate::fault::quad_finite(v));
    let refused = (bad >> (8 * (lane >> 3))) & 0xff != 0;
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
    if refused {
        (0, 0, f32::NAN, true)
    } else {
        (word, sum, dv, false)
    }
}
