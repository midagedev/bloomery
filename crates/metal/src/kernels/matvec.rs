//! The launch of the Q4_K and Q6_K mat-vecs.

use bloomery_device::{DeviceError, LAYER_NONE, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, ComputeEncoder};
use crate::fault::MetalFaultWord;

use super::geometry::{BLOCK_VALUES, ROWS_PER_SIMD_GROUP};
use super::{Kernels, MatvecPipelines, args_bytes, check_window_len, word};

/// The weight type a mat-vec reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    /// GGUF `Q4_K`: 144-byte blocks of 256 values.
    Q4K,
    /// GGUF `Q6_K`: 210-byte blocks of 256 values.
    Q6K,
}

impl Quant {
    /// The bytes of one block of 256 values.
    #[must_use]
    pub const fn block_bytes(self) -> usize {
        match self {
            Self::Q4K => 144,
            Self::Q6K => 210,
        }
    }

    /// The type's name as an error prints it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Q4K => "q4k",
            Self::Q6K => "q6k",
        }
    }
}

/// The RMS norm a mat-vec applies as it loads its input: `x * gain / sqrt(mean(x^2) + eps)`.
/// A norm that finds a value that is not finite, or a zero scale, raises the
/// norm site of `layer` on `fault` and writes NaN to its rows.
pub struct NormOnLoad<'w> {
    /// The norm's gain, one value per input column.
    pub gain: Window<'w, f32, MetalDevice>,
    /// The epsilon added to the mean of the squares.
    pub eps: f32,
    /// The fault word a raise lands in. Metal's own type: how a word binds to
    /// a kernel is backend code, and the contract's `FaultWord` does not say.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in: [`LAYER_NONE`] for a direct launch.
    /// The caller's: when Metal implements `ChainOps`, the walk's
    /// `LayerCtx.layer` feeds it.
    pub layer: u32,
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
    /// The input columns, a multiple of 256.
    pub cols: usize,
    /// The weights' bytes: `rows * cols / 256` blocks.
    pub weights: Window<'w, u8, MetalDevice>,
    /// The input vector, `cols` values.
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

fn refuse(detail: String) -> DeviceError {
    DeviceError::Range {
        what: "matvec",
        detail,
    }
}

/// The shape of `p`, refused by name when a window is not the shape's, the
/// columns are not whole blocks, or a count passes the kernel's 32-bit words.
fn check(p: &MatvecParams<'_>, simdgroups: usize) -> Result<Shape, DeviceError> {
    if p.rows == 0 || p.cols == 0 {
        return Err(refuse(format!(
            "{} rows by {} columns is empty",
            p.rows, p.cols
        )));
    }
    if !p.cols.is_multiple_of(BLOCK_VALUES) {
        return Err(refuse(format!(
            "{} columns are not whole blocks of {BLOCK_VALUES}",
            p.cols
        )));
    }
    let nblocks = p.cols / BLOCK_VALUES;
    let weight_bytes = p
        .rows
        .checked_mul(nblocks)
        .and_then(|blocks| blocks.checked_mul(p.quant.block_bytes()))
        .ok_or_else(|| refuse(format!("{} rows of {} columns wrap", p.rows, p.cols)))?;
    check_window_len("matvec", "weights", p.weights.len(), weight_bytes)?;
    check_window_len("matvec", "x", p.x.len(), p.cols)?;
    check_window_len("matvec", "y", p.y.len(), p.rows)?;
    if let Mode::NormOnLoad(norm) = &p.mode {
        check_window_len("matvec", "gain", norm.gain.len(), p.cols)?;
    }
    // The kernels read the weights as 16-bit words, from the window's byte
    // offset in its buffer.
    let start = p.weights.byte_range()?.start.get();
    if !start.is_multiple_of(2) {
        return Err(refuse(format!(
            "the weights start {start} bytes into their buffer: the kernels read 16-bit words"
        )));
    }
    Ok(Shape {
        nblocks: word("matvec", "the block count", nblocks)?,
        rows: word("matvec", "the row count", p.rows)?,
        groups: p.rows.div_ceil(simdgroups * ROWS_PER_SIMD_GROUP),
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
    /// Encode one mat-vec dispatch on `encoder`. Every window is checked
    /// against the shape first; nothing is encoded for a launch that is
    /// refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn matvec(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: MatvecParams<'_>,
    ) -> Result<(), DeviceError> {
        let shape = check(&p, self.simdgroups)?;
        let (layer, eps) = match &p.mode {
            Mode::NormOnLoad(norm) => (norm.layer, norm.eps),
            Mode::Store | Mode::Residual => (LAYER_NONE, 0.0),
        };
        let args = args_bytes([shape.rows, shape.nblocks, layer, eps.to_bits()]);
        let family = match p.quant {
            Quant::Q4K => &self.q4k,
            Quant::Q6K => &self.q6k,
        };
        encoder.set_pipeline(family.of(&p.mode));
        encoder.set_binding(0, Binding::of(&p.weights)?)?;
        encoder.set_binding(1, Binding::of(&p.x)?)?;
        encoder.set_binding(2, Binding::of_mut(&p.y)?)?;
        encoder.set_binding(3, Binding::bytes(&args)?)?;
        if let Mode::NormOnLoad(norm) = &p.mode {
            encoder.set_binding(4, Binding::of(&norm.gain)?)?;
            encoder.set_binding(5, norm.fault.binding())?;
        }
        let threads = self.threads_per_group();
        encoder.dispatch(shape.groups * threads, threads)
    }
}
