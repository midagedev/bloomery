//! The launch of the Q4_K, Q6_K, A4, A8, A4H, A8H, Q8_0 and Q5_K mat-vecs, and
//! the f32 one's segmented form ([`Quant::F32`]).

use bloomery_device::{DeviceError, LAYER_NONE, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;

use super::{Kernels, MatvecPipelines, args_bytes, check_window_len, pack, word};

/// The values of the f32 block the kernels read: a quarter of a SIMD group's
/// lanes take four of them each (`quant.metal`'s `block_f32_16`).
const F32_BLOCK_VALUES: usize = 16;

/// The weight type a mat-vec reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    /// GGUF `Q4_K`: 144-byte blocks of 256 values.
    Q4K,
    /// GGUF `Q6_K`: 210-byte blocks of 256 values.
    Q6K,
    /// MLX's affine 4-bit: 36-byte groups of 64 values (`MLX_A4`).
    A4,
    /// MLX's affine 8-bit: 68-byte groups of 64 values (`MLX_A8`).
    A8,
    /// MLX's affine 4-bit with F16 scales and biases: 36-byte groups of 64
    /// values (`MLX_A4H`).
    A4H,
    /// MLX's affine 8-bit with F16 scales and biases: 68-byte groups of 64
    /// values (`MLX_A8H`).
    A8H,
    /// GGUF `Q8_0`: 34-byte blocks of 32 values.
    Q8_0,
    /// GGUF `Q5_K`: 176-byte blocks of 256 values.
    Q5K,
    /// GGUF `F32` rows, read in blocks of 16 values: 64 bytes. Its one launch
    /// is the segmented mat-vec (and, in a pass, the segmented GEMM that
    /// runs each segment as the router's f32 GEMM): the delta net's `ssm_alpha`
    /// and `ssm_beta`. Every other launch refuses it by name
    /// ([`Quant::dense`]).
    F32,
}

impl Quant {
    /// The bytes of one block: gguf's own row for the type, the one owner
    /// of the number. F32's block is the kernel's, sixteen values (gguf's row
    /// is one).
    #[must_use]
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => F32_BLOCK_VALUES * 4,
            _ => usize::try_from(
                self.ggml()
                    .type_size()
                    .expect("every kernel type has a GGUF row"),
            )
            .expect("a block size fits usize"),
        }
    }

    /// Every type a session's arena opens: all the mat-vec families, in the
    /// order a layer's P1 dispatches them.
    pub const ALL: [Quant; 9] = [
        Self::Q4K,
        Self::Q6K,
        Self::A4,
        Self::A8,
        Self::A4H,
        Self::A8H,
        Self::Q8_0,
        Self::Q5K,
        Self::F32,
    ];

    /// The kernel type that reads file type `ty`, or `None` for a type a
    /// session does not open: [`Quant::ggml`] read backwards over
    /// [`Quant::ALL`], the one map.
    #[must_use]
    pub fn of_ggml(ty: gguf::quant::GgmlType) -> Option<Quant> {
        Self::ALL.into_iter().find(|q| q.ggml() == ty)
    }

    /// The GGUF type this kernel reads.
    #[must_use]
    pub const fn ggml(self) -> gguf::quant::GgmlType {
        match self {
            Self::Q4K => gguf::quant::GgmlType::Q4_K,
            Self::Q6K => gguf::quant::GgmlType::Q6_K,
            Self::A4 => gguf::quant::GgmlType::MLX_A4,
            Self::A8 => gguf::quant::GgmlType::MLX_A8,
            Self::A4H => gguf::quant::GgmlType::MLX_A4H,
            Self::A8H => gguf::quant::GgmlType::MLX_A8H,
            Self::Q8_0 => gguf::quant::GgmlType::Q8_0,
            Self::Q5K => gguf::quant::GgmlType::Q5_K,
            Self::F32 => gguf::quant::GgmlType::F32,
        }
    }

    /// The values one block holds, gguf's row for the type: the columns of a
    /// launch are a multiple of it, and its block count a division by it.
    /// F32's block is the kernel's, sixteen values (gguf's row is one).
    #[must_use]
    pub fn values_per_block(self) -> usize {
        match self {
            Self::F32 => F32_BLOCK_VALUES,
            _ => usize::try_from(
                self.ggml()
                    .blck_size()
                    .expect("every kernel type has a GGUF row"),
            )
            .expect("a block's value count fits usize"),
        }
    }

    /// Whether the type has every launch but the segmented mat-vec: the
    /// plain, norm-on-load and residual mat-vecs, the dense GEMMs, the
    /// experts' pair, fold and grouped GEMM. F32 has the segmented mat-vec
    /// only, so a site that is not a layer's P1 projection refuses it by name.
    #[must_use]
    pub const fn dense(self) -> bool {
        match self {
            Self::Q4K
            | Self::Q6K
            | Self::A4
            | Self::A8
            | Self::A4H
            | Self::A8H
            | Self::Q8_0
            | Self::Q5K => true,
            Self::F32 => false,
        }
    }

    /// The alignment the row loops' weight reads need: the affine types' 32-bit
    /// words, the 16-bit words of the other quantized blocks, or F32's
    /// `float4` loads.
    pub(crate) const fn weight_align(self) -> super::WeightAlign {
        match self {
            Self::F32 => super::WeightAlign::Float4,
            Self::A4 | Self::A8 | Self::A4H | Self::A8H => super::WeightAlign::Word,
            Self::Q4K | Self::Q6K | Self::Q8_0 | Self::Q5K => super::WeightAlign::Half,
        }
    }

    /// Whether the embedding launch has an entry for the type: the
    /// `Kernels::embed` pipeline match, which refuses the others by name.
    #[must_use]
    pub const fn embeds(self) -> bool {
        match self {
            Self::Q4K | Self::A4 | Self::A8 | Self::A4H | Self::A8H | Self::Q8_0 | Self::Q5K => {
                true
            }
            Self::Q6K | Self::F32 => false,
        }
    }

    /// Whether the head launch has an entry for the type: the
    /// `Kernels::head` logits pipeline match, which refuses the others by
    /// name.
    #[must_use]
    pub const fn heads(self) -> bool {
        match self {
            Self::Q6K | Self::A4 | Self::A8 | Self::A4H | Self::A8H | Self::Q8_0 | Self::Q5K => {
                true
            }
            Self::Q4K | Self::F32 => false,
        }
    }

    /// Whether the head-normed residual mat-vec has an entry for the type:
    /// the `Kernels::headnorm_resid` pipeline match, which refuses the
    /// others by name.
    #[must_use]
    pub const fn headnorms(self) -> bool {
        match self {
            Self::Q4K | Self::A4 | Self::A8 | Self::A4H | Self::A8H | Self::Q8_0 | Self::Q5K => {
                true
            }
            Self::Q6K | Self::F32 => false,
        }
    }

    /// The type's name as an error prints it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Q4K => "q4k",
            Self::Q6K => "q6k",
            Self::A4 => "a4",
            Self::A8 => "a8",
            Self::A4H => "a4h",
            Self::A8H => "a8h",
            Self::Q8_0 => "q8_0",
            Self::Q5K => "q5k",
            Self::F32 => "f32",
        }
    }
}

/// The RMS norm a mat-vec applies as it loads its input: `x * gain / sqrt(mean(x^2) + eps)`.
/// A norm whose input or scale it cannot take, or whose gain leaves a row
/// not finite, raises the norm site of `layer` on `fault` and writes NaN to
/// its rows.
pub struct NormOnLoad<'w> {
    /// The norm's gain, one value per input column, starting on a 16-byte
    /// boundary of its buffer.
    pub gain: Window<'w, f32, MetalDevice>,
    /// The epsilon added to the mean of the squares.
    pub eps: f32,
    /// The fault word a raise lands in. Metal's own type: how a word binds to
    /// a kernel is backend code, and the contract's `FaultWord` does not say.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in: [`LAYER_NONE`] for a direct launch.
    /// The caller's: the chain walk's `LayerCtx.layer` feeds it.
    pub layer: u32,
}

/// Where a launch that norms its input finds the norm. `Gain`: the input is
/// the raw `x`, and the launch sums its squares as it loads it and multiplies
/// it by the gain (the norm on load). `Scaled`: the input is already `x · gain`,
/// written by the sandwich join before the launch ([`Kernels::norm_add_next`]),
/// and `scale` is the one word of the norm's scale that join computed in the
/// launch's own lane order; the rows are the `Gain` launch's bits.
#[derive(Clone, Copy)]
pub enum LoadNorm<'w> {
    Gain {
        /// One value per input column, starting on a 16-byte boundary of its
        /// buffer.
        gain: Window<'w, f32, MetalDevice>,
        /// The epsilon added to the mean of the squares.
        eps: f32,
    },
    Scaled {
        /// The scale word: a NaN scale (its join found the norm unusable) makes
        /// every row NaN.
        scale: Window<'w, f32, MetalDevice>,
    },
}

impl<'w> LoadNorm<'w> {
    /// Whether the launch takes the scaled entry.
    #[must_use]
    pub fn scaled(&self) -> bool {
        matches!(self, Self::Scaled { .. })
    }

    /// The epsilon the argument block carries: 0 for a scaled launch, which
    /// reads none.
    #[must_use]
    pub fn eps(&self) -> f32 {
        match self {
            Self::Gain { eps, .. } => *eps,
            Self::Scaled { .. } => 0.0,
        }
    }

    /// The window bound in the gain's slot: the gain, or the scale word.
    #[must_use]
    pub fn window(&self) -> &Window<'w, f32, MetalDevice> {
        match self {
            Self::Gain { gain, .. } => gain,
            Self::Scaled { scale } => scale,
        }
    }

    /// The norm's windows against an input of `cols` values: a gain of `cols`
    /// values on a `float4` boundary, or one scale word.
    pub(crate) fn check(&self, what: &'static str, cols: usize) -> Result<(), DeviceError> {
        match self {
            Self::Gain { gain, .. } => {
                check_window_len(what, "gain", gain.len(), cols)?;
                super::check_weights_align(
                    what,
                    "gain",
                    gain.byte_range()?.start.get(),
                    super::WeightAlign::Float4,
                )
            }
            Self::Scaled { scale } => check_window_len(what, "scale", scale.len(), 1),
        }
    }
}

/// What a mat-vec does with its input and its output.
pub enum Mode<'w> {
    /// `y = W . x`.
    Store,
    /// `y = W . rms_norm(x)`, the norm applied as the input is loaded.
    NormOnLoad(NormOnLoad<'w>),
    /// `y = y + W . x`, in place over `y`: the residual epilogue.
    Residual,
}

/// One mat-vec: `rows` outputs over `cols` input columns of one weight type.
pub struct MatvecParams<'w> {
    /// The weight type.
    pub quant: Quant,
    /// The output rows.
    pub rows: usize,
    /// The input columns, a multiple of the type's block values.
    pub cols: usize,
    /// The weights' bytes: `rows * cols / values_per_block()` blocks.
    pub weights: Window<'w, u8, MetalDevice>,
    /// The input vector, `cols` values, starting on a 16-byte boundary of its
    /// buffer.
    pub x: Window<'w, f32, MetalDevice>,
    /// The output, `rows` values.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// What the mat-vec does with its input and its output.
    pub mode: Mode<'w>,
}

/// The shape of a launch, once its windows were checked against it.
struct Shape {
    nblocks: u32,
    rows: u32,
    groups: usize,
}

const WHAT: &str = "matvec";

fn refuse(detail: impl Into<String>) -> DeviceError {
    super::range(WHAT, detail)
}

/// The shape of `p`, refused by name when a window is not the shape's, the
/// columns are not whole blocks, or a count passes the kernel's 32-bit words.
fn check(
    p: &MatvecParams<'_>,
    simdgroups: usize,
    rows_per_simd_group: usize,
) -> Result<Shape, DeviceError> {
    if p.rows == 0 || p.cols == 0 {
        return Err(refuse(format!(
            "{} rows by {} columns is empty",
            p.rows, p.cols
        )));
    }
    // The kernels index the input with signed 32-bit arithmetic (a block
    // index times its values) and divide the norm's sum by the column count
    // in 32 bits.
    if i32::try_from(p.cols).is_err() {
        return Err(refuse(format!(
            "{} columns pass the kernels' signed 32-bit index",
            p.cols
        )));
    }
    if !p.cols.is_multiple_of(p.quant.values_per_block()) {
        return Err(refuse(format!(
            "{} columns are not whole blocks of {}",
            p.cols,
            p.quant.values_per_block()
        )));
    }
    let nblocks = p.cols / p.quant.values_per_block();
    let weight_bytes = super::weight_bytes(p.quant, p.rows, nblocks)
        .ok_or_else(|| refuse(format!("{} rows of {} columns wrap", p.rows, p.cols)))?;
    check_window_len(WHAT, "weights", p.weights.len(), weight_bytes)?;
    check_window_len(WHAT, "x", p.x.len(), p.cols)?;
    check_window_len(WHAT, "y", p.y.len(), p.rows)?;
    if let Mode::NormOnLoad(norm) = &p.mode {
        check_window_len(WHAT, "gain", norm.gain.len(), p.cols)?;
    }
    // The kernels read the weights as words of the type's width, from the
    // window's byte offset in its buffer, and the input and the gain as
    // `float4`s.
    let start = p.weights.byte_range()?.start.get();
    super::check_weights_align(WHAT, "weights", start, p.quant.weight_align())?;
    let start = p.x.byte_range()?.start.get();
    super::check_weights_align(WHAT, "x", start, super::WeightAlign::Float4)?;
    if let Mode::NormOnLoad(norm) = &p.mode {
        let start = norm.gain.byte_range()?.start.get();
        super::check_weights_align(WHAT, "gain", start, super::WeightAlign::Float4)?;
    }
    Ok(Shape {
        nblocks: word(WHAT, "the block count", nblocks)?,
        rows: word(WHAT, "the row count", p.rows)?,
        groups: p.rows.div_ceil(simdgroups * rows_per_simd_group),
    })
}

impl MatvecPipelines {
    fn of(&self, mode: &Mode<'_>) -> &crate::library::Pipeline {
        match mode {
            Mode::Store => &self.store,
            Mode::NormOnLoad(_) => &self.norm,
            Mode::Residual => &self.residual,
        }
    }
}

impl Kernels {
    /// The pipelines of the mat-vec family of `quant`, or `None` for a type
    /// whose one entry is the segmented mat-vec ([`Quant::dense`]).
    pub(super) fn dense_family(&self, quant: Quant) -> Option<&MatvecPipelines> {
        match quant {
            Quant::Q4K => Some(&self.q4k),
            Quant::Q6K => Some(&self.q6k),
            Quant::A4 => Some(&self.a4),
            Quant::A8 => Some(&self.a8),
            Quant::A4H => Some(&self.a4h),
            Quant::A8H => Some(&self.a8h),
            Quant::Q8_0 => Some(&self.q8_0),
            Quant::Q5K => Some(&self.q5k),
            Quant::F32 => None,
        }
    }

    /// Encode one mat-vec dispatch on `encoder`. Every window is checked
    /// against the shape first; nothing is encoded for a launch that is
    /// refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn matvec(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: MatvecParams<'_>,
    ) -> Result<(), DeviceError> {
        let shape = check(&p, self.simdgroups, self.rows_per_simd_group)?;
        let (layer, eps) = match &p.mode {
            Mode::NormOnLoad(norm) => (norm.layer, norm.eps),
            Mode::Store | Mode::Residual => (LAYER_NONE, 0.0),
        };
        let args = args_bytes([shape.rows, shape.nblocks, layer, eps.to_bits()]);
        let family = self
            .dense_family(p.quant)
            .ok_or_else(|| refuse(format!("{} has no plain mat-vec entry", p.quant.name())))?;
        encoder.set_pipeline(family.of(&p.mode));
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.weights)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        if let Mode::NormOnLoad(norm) = &p.mode {
            slot.set(Binding::of(&norm.gain)?)?;
            slot.set(norm.fault.binding())?;
        }
        let threads = self.threads_per_group();
        slot.dispatch_groups(shape.groups, 1, threads)
    }
}
