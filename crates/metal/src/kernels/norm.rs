//! The RMS norm of a chunk of rows: `y = (x · gain) · scale` for each row, in
//! the order of the router's norm. The kernel is `kernels/norm.metal`'s
//! `norm_rows`: one threadgroup a row, the router's thread count.
//!
//! A row whose scale is unusable raises the norm site and is NaN, and the
//! other rows are unaffected.
//!
//! The sandwich residual is the same norm with another store
//! ([`Kernels::norm_add_rows`], the kernel's `norm_add_rows`): `y += add ·
//! ((x · gain) · scale)` in place, `x` the sub-layer's output and `y` the
//! residual stream. The norm's sums are `norm_rows`' own, so the value added
//! is the bits `norm_rows` writes. A row the norm cannot take is NaN in the
//! stream.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::library::Pipeline;

use super::rows::check_strided;
use super::{Kernels, WeightAlign, check_weights_align, check_window_len, word};

const WHAT: &str = "norm rows";
const WHAT_ADD: &str = "norm add rows";

/// The norm of `rows` rows of `cols` values.
pub struct NormRowsParams<'w> {
    /// Rows in the chunk. Zero is refused.
    pub rows: usize,
    /// Values in a row. A multiple of four floats for every thread of the
    /// router's threadgroup.
    pub cols: usize,
    /// The input, `x_stride` elements between rows.
    pub x: Window<'w, f32, MetalDevice>,
    /// Elements between rows of `x`: a multiple of four, at least `cols`.
    pub x_stride: usize,
    /// The norm's gain, `cols` values.
    pub gain: Window<'w, f32, MetalDevice>,
    /// The output, `y_stride` elements between rows.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// Elements between rows of `y`: a multiple of four, at least `cols`.
    pub y_stride: usize,
    /// The epsilon added to the mean of the squares.
    pub eps: f32,
    /// The fault word a raise lands in.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

/// The sandwich residual of `rows` rows: `y += add · norm(x)` for each row.
/// `norm.x` is the sub-layer's output, `norm.y` the residual stream, read and
/// written in place.
pub struct NormAddRowsParams<'w> {
    /// The norm and the windows it reads and writes.
    pub norm: NormRowsParams<'w>,
    /// The scale of the normed value added into the stream. Finite.
    pub add: f32,
}

fn refuse(what: &'static str, detail: impl Into<String>) -> DeviceError {
    super::range(what, detail)
}

impl Kernels {
    /// Encode the norm of a chunk of rows: one dispatch, a threadgroup a row.
    /// A shape the kernel does not take or a window that is not that shape is
    /// refused by name and nothing is encoded.
    #[must_use = "a refused launch encoded nothing"]
    pub fn norm_rows(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: NormRowsParams<'_>,
    ) -> Result<(), DeviceError> {
        self.encode_norm(encoder, WHAT, &self.norm_rows, &p, None)
    }

    /// Encode the sandwich residual of a chunk of rows: one dispatch, a
    /// threadgroup a row, the shapes and refusals of [`Kernels::norm_rows`].
    /// A scale that is not finite is refused by name too.
    #[must_use = "a refused launch encoded nothing"]
    pub fn norm_add_rows(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: NormAddRowsParams<'_>,
    ) -> Result<(), DeviceError> {
        if !p.add.is_finite() {
            return Err(refuse(
                WHAT_ADD,
                format!("the add scale {} is not finite", p.add),
            ));
        }
        self.encode_norm(encoder, WHAT_ADD, &self.norm_add_rows, &p.norm, Some(p.add))
    }

    fn encode_norm(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        what: &'static str,
        pipeline: &Pipeline,
        p: &NormRowsParams<'_>,
        add: Option<f32>,
    ) -> Result<(), DeviceError> {
        if p.rows == 0 {
            return Err(refuse(what, "rows of 0"));
        }
        let lanes = self.router_threads() * 4;
        if p.cols == 0 || !p.cols.is_multiple_of(lanes) {
            return Err(refuse(
                what,
                format!(
                    "cols {} is not a multiple of {lanes}, the float4s the norm's threads cover",
                    p.cols
                ),
            ));
        }
        for (name, stride) in [("x", p.x_stride), ("y", p.y_stride)] {
            if !stride.is_multiple_of(4) {
                return Err(refuse(
                    what,
                    format!("the {name} stride {stride} is not a multiple of four floats"),
                ));
            }
        }
        check_strided(what, "x", p.x.len(), p.rows, p.x_stride, p.cols)?;
        check_strided(what, "y", p.y.len(), p.rows, p.y_stride, p.cols)?;
        check_window_len(what, "gain", p.gain.len(), p.cols)?;
        for (name, start) in [
            ("x", p.x.byte_range()?.start.get()),
            ("gain", p.gain.byte_range()?.start.get()),
            ("y", p.y.byte_range()?.start.get()),
        ] {
            check_weights_align(what, name, start, WeightAlign::Float4)?;
        }
        let mut words = vec![
            word(what, "cols", p.cols)?,
            p.layer,
            p.eps.to_bits(),
            word(what, "the x stride", p.x_stride)?,
            word(what, "the y stride", p.y_stride)?,
        ];
        if let Some(add) = add {
            words.push(add.to_bits());
        }
        let args: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        encoder.set_pipeline(pipeline);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&p.gain)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(&args)?)?;
        slot.set(p.fault.binding())?;
        slot.dispatch_groups(p.rows, 1, self.router_threads())
    }
}
