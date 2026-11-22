//! The launch of the few-row mat-vec: one weight read multiplied into 3 to 16
//! activation rows on the matrix units, the kernel `kernels/matvec_mrows.metal`:
//! one activation fragment up to 8 rows, two (TM = 2) for 9 to 16.
//!
//! Where the M-row family ([`super::matvec_rows`]) runs the 1-row kernel's own
//! order on exactly [`super::MROWS`] rows and equals it in bits, this launch
//! holds the rows past it: its order is the fragment multiply's, so a row
//! equals the 1-row kernel's within the f32 accumulation band
//! (([`few_path_ops`])), not in bits. A4 only; every other quant, an M outside
//! [`FEW_MIN`]..=[`FEW_MAX`] and a column count that is not a whole number of
//! 128-value blocks are refused by name, with nothing encoded.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::library::{Library, Pipeline};

use super::matvec::Quant;
use super::{Kernels, args_bytes, check_weights_align, check_window_len, pack, range, word};

const WHAT: &str = "matvec few rows";

/// The fewest activation rows the launch takes: the M-row family owns two.
pub const FEW_MIN: usize = 3;

/// The most activation rows the launch takes: two fragments' width.
pub const FEW_MAX: usize = 16;

/// The activation rows one fragment holds: past it the launch takes a second.
pub const FEW_FRAGMENT: usize = 8;

/// The weight rows a threadgroup owns: four 8-row fragments.
const ROWS_PER_GROUP: usize = 32;

/// The SIMD groups in a threadgroup, splitting K between them.
const SIMD_GROUPS: usize = 4;

/// The values one K step of a SIMD group covers: two A4 groups.
const BLOCK_VALUES: usize = 128;

/// The rounded operations on the longest path of one output over `nblocks`
/// A4 groups, the count the band `gamma(n)` is made of: the weight value's
/// fused multiply-add (1), the fragment multiply's product (1), the sums of a
/// SIMD group's K share — one add per product, worst case, of the 128 products
/// a 128-value block holds on each of its `ceil(nblocks / 8)` blocks —, the
/// three adds that join the four SIMD groups in SIMD group order, and the
/// residual's add (1). A row equals the 1-row kernel's within the sum of this
/// band and the 1-row kernel's own (`oracle::band`), both measured on the
/// terms' magnitudes `sum |w x|`.
#[must_use]
pub const fn few_path_ops(nblocks: usize) -> usize {
    1 + 1 + BLOCK_VALUES * nblocks.div_ceil(2 * SIMD_GROUPS) + (SIMD_GROUPS - 1) + 1
}

/// The few-row residual mat-vec: `y[m, r] += W[r, :] . x[m, :]`, `m` rows of
/// `rows` outputs over `cols` columns each.
pub struct MatvecFewParams<'w> {
    /// The weight type. A4 today; the launch refuses the others by name.
    pub quant: Quant,
    /// The activation rows, [`FEW_MIN`]..=[`FEW_MAX`] (one activation fragment
    /// up to [`FEW_FRAGMENT`], two above it).
    pub m: usize,
    /// The output rows an activation row takes.
    pub rows: usize,
    /// The input columns, a multiple of 128.
    pub cols: usize,
    /// The weights' bytes: `rows * cols / 64` A4 groups.
    pub weights: Window<'w, u8, MetalDevice>,
    /// The inputs, `m` rows of `cols` values `x_stride` apart (the last row
    /// ends where its values do), starting on a 16-byte boundary of their
    /// buffer.
    pub x: Window<'w, f32, MetalDevice>,
    /// Floats from one input row to the next: at least `cols`, a multiple of
    /// four.
    pub x_stride: usize,
    /// The outputs, `m` rows of `rows` values, which the launch adds into.
    pub y: WindowMut<'w, f32, MetalDevice>,
}

/// The pipeline of the few-row family.
pub(crate) struct Pipelines {
    /// One activation fragment: 3 to [`FEW_FRAGMENT`] rows.
    resid: Pipeline,
    /// Two: [`FEW_FRAGMENT`] + 1 to [`FEW_MAX`] rows.
    resid2: Pipeline,
}

impl Pipelines {
    pub(crate) fn new(library: &Library) -> Result<Self, DeviceError> {
        Ok(Self {
            resid: library.pipeline("a4_matvec_mrows_resid")?,
            resid2: library.pipeline("a4_matvec_mrows2_resid")?,
        })
    }
}

struct Shape {
    nblocks: u32,
    rows: u32,
    m: u32,
    x_stride: u32,
    groups: usize,
}

fn check(p: &MatvecFewParams<'_>) -> Result<Shape, DeviceError> {
    if p.quant != Quant::A4 {
        return Err(range(
            WHAT,
            format!(
                "{} has no few-row entry; the launch holds A4",
                p.quant.name()
            ),
        ));
    }
    if !(FEW_MIN..=FEW_MAX).contains(&p.m) {
        return Err(range(
            WHAT,
            format!("M is {}; the launch runs {FEW_MIN} to {FEW_MAX}", p.m),
        ));
    }
    if p.rows == 0 || p.cols == 0 {
        return Err(range(
            WHAT,
            format!("{} rows by {} columns is empty", p.rows, p.cols),
        ));
    }
    if !p.cols.is_multiple_of(BLOCK_VALUES) {
        return Err(range(
            WHAT,
            format!("{} columns are not whole blocks of {BLOCK_VALUES}", p.cols),
        ));
    }
    if i32::try_from(p.cols).is_err() {
        return Err(range(
            WHAT,
            format!("{} columns pass the kernel's signed 32-bit index", p.cols),
        ));
    }
    let nblocks = p.cols / p.quant.values_per_block();
    let weight_bytes = super::weight_bytes(p.quant, p.rows, nblocks)
        .ok_or_else(|| range(WHAT, format!("{} rows of {} columns wrap", p.rows, p.cols)))?;
    check_window_len(WHAT, "weights", p.weights.len(), weight_bytes)?;
    if p.x_stride < p.cols || !p.x_stride.is_multiple_of(4) {
        return Err(range(
            WHAT,
            format!(
                "the x stride {} is narrower than its row of {} columns, or not a whole number of \
                 float4s",
                p.x_stride, p.cols
            ),
        ));
    }
    let x_len = (p.m - 1)
        .checked_mul(p.x_stride)
        .and_then(|at| at.checked_add(p.cols))
        .ok_or_else(|| range(WHAT, format!("{} rows of stride {} wrap", p.m, p.x_stride)))?;
    check_window_len(WHAT, "x", p.x.len(), x_len)?;
    check_window_len(WHAT, "y", p.y.len(), p.m * p.rows)?;
    let weights_start = p.weights.byte_range()?.start.get();
    check_weights_align(WHAT, "weights", weights_start, p.quant.weight_align())?;
    let x_start = p.x.byte_range()?.start.get();
    check_weights_align(WHAT, "x", x_start, super::WeightAlign::Float4)?;
    Ok(Shape {
        nblocks: word(WHAT, "the block count", nblocks)?,
        rows: word(WHAT, "the row count", p.rows)?,
        m: word(WHAT, "M", p.m)?,
        x_stride: word(WHAT, "the x stride", p.x_stride)?,
        groups: p.rows.div_ceil(ROWS_PER_GROUP),
    })
}

impl Kernels {
    /// Encode one few-row residual mat-vec dispatch on `encoder`. Nothing is
    /// encoded for a launch that is refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn matvec_few(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: MatvecFewParams<'_>,
    ) -> Result<(), DeviceError> {
        let shape = check(&p)?;
        let args = args_bytes([
            shape.rows,
            shape.nblocks,
            shape.m,
            shape.rows,
            shape.x_stride,
        ]);
        encoder.set_pipeline(if p.m <= FEW_FRAGMENT {
            &self.matvec_mrows.resid
        } else {
            &self.matvec_mrows.resid2
        });
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.weights)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.dispatch_groups(shape.groups, 1, SIMD_GROUPS * super::geometry::SIMD_WIDTH)
    }
}
