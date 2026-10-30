//! The head-normed residual mat-vec: a gated delta net's out projection.
//!
//! The input of the projection is the gated numerator `g` (the delta step's
//! output, the norm gain and the gate activation, per value head) times the
//! norm scale of the head each column belongs to. The scale comes from the
//! head's partial sums of the squares of its values, which the delta kernel
//! leaves in `partials[head][block]`: the head's sum is the sum of its eight
//! partials, and the scale is `1 / sqrt(sum / head_cols + eps)`.
//!
//! The layout of `g` and `partials` is the contract with the delta kernel: the
//! value heads' count and width are the instance's (`DeltaGeom`), and each
//! head's partials are one per column block (`geometry::PARTIALS_PER_HEAD`).

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::geom::DeltaGeom;

use super::matvec::Quant;
use super::{Kernels, args_bytes, check_window_len, pack, word};

/// The head-normed residual mat-vec's parameters: the out projection's weights,
/// the gated numerator, the heads' partial sums, and the output the residual
/// is added into.
pub struct HeadNormParams<'w> {
    /// The weight type. Q4_K, A4 and A8 have a head-normed entry; Q6_K has
    /// none and is refused by name.
    pub quant: Quant,
    /// The output rows: the model's hidden width.
    pub rows: usize,
    /// The input columns: one value head's gated output after another.
    pub cols: usize,
    /// The out projection's weights.
    pub weights: Window<'w, u8, MetalDevice>,
    /// The gated numerator, one value per input column, starting on a 16-byte
    /// boundary of its buffer.
    pub g: Window<'w, f32, MetalDevice>,
    /// The partial sums of the squares of the values: every head's row of
    /// one partial per column block.
    pub partials: Window<'w, f32, MetalDevice>,
    /// The output, the residual added into it.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// The epsilon of the head norm.
    pub eps: f32,
    /// The fault word a raise lands in.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

/// The shape of a head-normed launch, once its windows were checked against it.
struct Shape {
    nblocks: u32,
    rows: u32,
    groups: usize,
}

const WHAT: &str = "headnorm matvec";

fn refuse(detail: impl Into<String>) -> DeviceError {
    super::range(WHAT, detail)
}

/// The shape of `p`, refused by name when the width is not the heads' own, when
/// a window is not the shape's, or when a count passes the kernel's 32-bit
/// words.
fn check(
    d: &DeltaGeom,
    p: &HeadNormParams<'_>,
    simdgroups: usize,
    rows_per_simd_group: usize,
) -> Result<Shape, DeviceError> {
    if p.rows == 0 {
        return Err(refuse("no rows".to_string()));
    }
    if p.cols != d.value_heads * d.head_cols {
        return Err(refuse(format!(
            "{} columns, the kernel's heads hold {} of {}",
            p.cols, d.value_heads, d.head_cols
        )));
    }
    let nblocks = p.cols / p.quant.values_per_block();
    let weight_bytes = super::weight_bytes(p.quant, p.rows, nblocks)
        .ok_or_else(|| refuse(format!("{} rows of {} columns wrap", p.rows, p.cols)))?;
    check_window_len(WHAT, "weights", p.weights.len(), weight_bytes)?;
    check_window_len(WHAT, "g", p.g.len(), p.cols)?;
    check_window_len(WHAT, "partials", p.partials.len(), d.partials_stride)?;
    check_window_len(WHAT, "y", p.y.len(), p.rows)?;
    let start = p.weights.byte_range()?.start.get();
    super::check_weights_align(WHAT, "weights", start, p.quant.weight_align())?;
    // The gated numerator is read as `float4`s.
    let start = p.g.byte_range()?.start.get();
    super::check_weights_align(WHAT, "g", start, super::WeightAlign::Float4)?;
    Ok(Shape {
        nblocks: word(WHAT, "the block count", nblocks)?,
        rows: word(WHAT, "the row count", p.rows)?,
        groups: p.rows.div_ceil(simdgroups * rows_per_simd_group),
    })
}

impl Kernels {
    /// Encode one head-normed residual mat-vec dispatch on `encoder`. Every
    /// window is checked against the shape first; nothing is encoded for a
    /// launch that is refused.
    #[must_use = "a refused launch encoded nothing"]
    pub fn headnorm_resid(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: HeadNormParams<'_>,
    ) -> Result<(), DeviceError> {
        let shape = check(
            &self.geom.delta,
            &p,
            self.simdgroups,
            self.rows_per_simd_group,
        )?;
        let args = args_bytes([shape.rows, shape.nblocks, p.layer, p.eps.to_bits()]);
        let pipeline = match p.quant {
            Quant::Q4K => &self.headnorm,
            Quant::A4 => &self.headnorm_a4,
            Quant::A8 => &self.headnorm_a8,
            Quant::Q8_0 => &self.headnorm_q8_0,
            Quant::Q5K => &self.headnorm_q5k,
            // A quant with no head-normed entry is refused by name: the mode
            // is absent, not defaulted.
            Quant::Q6K => {
                return Err(refuse("Q6_K has no head-normed mat-vec".to_string()));
            }
            Quant::F32 => {
                return Err(refuse("F32 has no head-normed mat-vec".to_string()));
            }
        };
        encoder.set_pipeline(pipeline);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.weights)?)?;
        slot.set(Binding::of(&p.g)?)?;
        slot.set(Binding::of(&p.partials)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.fault.binding())?;
        let threads = self.threads_per_group();
        slot.dispatch_groups(shape.groups, 1, threads)
    }
}
