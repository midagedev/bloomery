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
//!
//! A one-row join whose next launch norms the stream on load
//! ([`Kernels::norm_add_next`], each type's `<q>_norm_add_next`) also writes
//! that launch's input for it: the new row times the next norm's gain, and
//! after it one word, the next norm's scale, summed in the next launch's own
//! lane order over its weight type. The next launch takes them as
//! [`LoadNorm::Scaled`](super::LoadNorm) and its rows are the bits of the
//! norm on load.
//!
//! A plain residual hands on the same input with no join
//! ([`Kernels::norm_next`], each type's `<q>_norm_next`): the stream, which
//! the out projection already added into, times the next norm's gain, and the
//! next norm's scale summed in the same walk.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::library::Pipeline;

use super::rows::check_strided;
use super::{Kernels, Quant, WeightAlign, args_bytes, check_weights_align, check_window_len, word};

/// The next launch's gain, the output it reads and the two argument words
/// (its epsilon and its layer), bound after the join's own.
type Next<'a, 'w> = Option<(
    &'a Window<'w, f32, MetalDevice>,
    &'a WindowMut<'w, f32, MetalDevice>,
    &'a [[u8; 4]; 2],
)>;

const WHAT: &str = "norm rows";
const WHAT_ADD: &str = "norm add rows";
const WHAT_NEXT: &str = "norm add next";
const WHAT_PLAIN: &str = "norm next";

/// The argument words both norm kernels take: cols, layer, eps and the two
/// strides.
const SHARED_WORDS: usize = 5;

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

/// A one-row sandwich join that also hands the next launch its normed input.
/// `join.norm.rows` is 1.
pub struct NormAddNextParams<'w> {
    /// The join: `y += add · norm(x)` on one row.
    pub join: NormAddRowsParams<'w>,
    /// The next launch's weight type, whose lane order the scale is summed
    /// in. F32 has no entry and is refused by name.
    pub quant: Quant,
    /// The next norm's gain, `cols` values on a 16-byte boundary.
    pub gain: Window<'w, f32, MetalDevice>,
    /// The next norm's epsilon.
    pub eps: f32,
    /// The layer the next norm's raise is recorded in.
    pub layer: u32,
    /// `cols` values of the new row times `gain`, then the scale word, on a
    /// 16-byte boundary.
    pub out: WindowMut<'w, f32, MetalDevice>,
}

/// A plain residual's hand-on: the next launch's input from the stream as it
/// stands, `cols` values (the stream's length).
pub struct NormNextParams<'w> {
    /// The next launch's weight type, whose lane order the scale is summed
    /// in. F32 has no entry and is refused by name.
    pub quant: Quant,
    /// The stream, whole blocks of `quant`, on a 16-byte boundary.
    pub x: Window<'w, f32, MetalDevice>,
    /// The next norm's gain, the stream's length, on a 16-byte boundary.
    pub gain: Window<'w, f32, MetalDevice>,
    /// The next norm's epsilon.
    pub eps: f32,
    /// The layer the next norm's raise is recorded in.
    pub layer: u32,
    /// The stream times `gain`, then the scale word, on a 16-byte boundary.
    pub out: WindowMut<'w, f32, MetalDevice>,
    /// The fault word a raise lands in.
    pub fault: &'w MetalFaultWord,
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
        self.encode_rows(encoder, WHAT, &self.norm_rows, &p, None, None)
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
        self.encode_join(encoder, WHAT_ADD, &self.norm_add_rows, &p, None)
    }

    /// Encode a one-row sandwich join and the next launch's input: one
    /// dispatch, one threadgroup. The join's shapes and refusals are
    /// [`Kernels::norm_add_rows`]'; a row count other than 1, columns that
    /// are not whole blocks of `quant`, a next epsilon that is not finite and
    /// an output shorter than `cols + 1` are refused by name too.
    #[must_use = "a refused launch encoded nothing"]
    pub fn norm_add_next(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: NormAddNextParams<'_>,
    ) -> Result<(), DeviceError> {
        let n = &p.join.norm;
        if n.rows != 1 {
            return Err(refuse(
                WHAT_NEXT,
                format!("{} rows, the join hands one row on", n.rows),
            ));
        }
        let Some(family) = self.dense_family(p.quant) else {
            return Err(refuse(
                WHAT_NEXT,
                format!("{} has no entry", p.quant.name()),
            ));
        };
        if !n.cols.is_multiple_of(p.quant.values_per_block()) {
            return Err(refuse(
                WHAT_NEXT,
                format!("cols {} are not whole blocks of {}", n.cols, p.quant.name()),
            ));
        }
        if !p.eps.is_finite() {
            return Err(refuse(
                WHAT_NEXT,
                format!("the next eps {} is not finite", p.eps),
            ));
        }
        check_window_len(WHAT_NEXT, "gain", p.gain.len(), n.cols)?;
        check_window_len(WHAT_NEXT, "out", p.out.len(), n.cols + 1)?;
        for (name, start) in [
            ("gain", p.gain.byte_range()?.start.get()),
            ("out", p.out.byte_range()?.start.get()),
        ] {
            check_weights_align(WHAT_NEXT, name, start, WeightAlign::Float4)?;
        }
        let next = args_bytes([p.eps.to_bits(), p.layer]);
        self.encode_join(
            encoder,
            WHAT_NEXT,
            family.norm_add_next(),
            &p.join,
            Some((&p.gain, &p.out, &next)),
        )
    }

    /// Encode a plain residual's hand-on: one dispatch, one threadgroup of
    /// the router's thread count. A type with no entry, a stream that is not
    /// whole blocks of `quant`, a next epsilon that is not finite, a gain
    /// other than the stream's length, an output other than `cols + 1` and a
    /// window off a 16-byte boundary are refused by name, and nothing is
    /// encoded.
    #[must_use = "a refused launch encoded nothing"]
    pub fn norm_next(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: NormNextParams<'_>,
    ) -> Result<(), DeviceError> {
        let Some(family) = self.dense_family(p.quant) else {
            return Err(refuse(
                WHAT_PLAIN,
                format!("{} has no entry", p.quant.name()),
            ));
        };
        let cols = p.x.len().0;
        if cols == 0 || !cols.is_multiple_of(p.quant.values_per_block()) {
            return Err(refuse(
                WHAT_PLAIN,
                format!("cols {cols} are not whole blocks of {}", p.quant.name()),
            ));
        }
        if !p.eps.is_finite() {
            return Err(refuse(
                WHAT_PLAIN,
                format!("the next eps {} is not finite", p.eps),
            ));
        }
        check_window_len(WHAT_PLAIN, "gain", p.gain.len(), cols)?;
        check_window_len(WHAT_PLAIN, "out", p.out.len(), cols + 1)?;
        for (name, start) in [
            ("x", p.x.byte_range()?.start.get()),
            ("gain", p.gain.byte_range()?.start.get()),
            ("out", p.out.byte_range()?.start.get()),
        ] {
            check_weights_align(WHAT_PLAIN, name, start, WeightAlign::Float4)?;
        }
        let args = args_bytes([word(WHAT_PLAIN, "cols", cols)?, p.eps.to_bits(), p.layer]);
        encoder.set_pipeline(family.norm_next());
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&p.gain)?)?;
        slot.set(Binding::of_mut(&p.out)?)?;
        slot.set(Binding::bytes(args.as_flattened())?)?;
        slot.set(p.fault.binding())?;
        slot.dispatch_groups(1, 1, self.router_threads())
    }

    /// A join with its next launch's input windows and argument words.
    fn encode_join(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        what: &'static str,
        pipeline: &Pipeline,
        p: &NormAddRowsParams<'_>,
        next: Next<'_, '_>,
    ) -> Result<(), DeviceError> {
        if !p.add.is_finite() {
            return Err(refuse(
                what,
                format!("the add scale {} is not finite", p.add),
            ));
        }
        self.encode_rows(encoder, what, pipeline, &p.norm, Some(p.add), next)
    }

    fn encode_rows(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        what: &'static str,
        pipeline: &Pipeline,
        p: &NormRowsParams<'_>,
        add: Option<f32>,
        next: Next<'_, '_>,
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
        // The add scale is the sixth word, sent only by the sandwich residual.
        let args = args_bytes([
            word(what, "cols", p.cols)?,
            p.layer,
            p.eps.to_bits(),
            word(what, "the x stride", p.x_stride)?,
            word(what, "the y stride", p.y_stride)?,
            add.map_or(0, f32::to_bits),
        ]);
        let sent = &args[..SHARED_WORDS + usize::from(add.is_some())];
        encoder.set_pipeline(pipeline);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&p.gain)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(sent.as_flattened())?)?;
        slot.set(p.fault.binding())?;
        if let Some((gain, out, words)) = next {
            slot.set(Binding::of(gain)?)?;
            slot.set(Binding::of_mut(out)?)?;
            slot.set(Binding::bytes(words.as_flattened())?)?;
        }
        slot.dispatch_groups(p.rows, 1, self.router_threads())
    }
}
