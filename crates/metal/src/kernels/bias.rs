//! The bias of an attention layer's P1 output: `y[r][i] += bias[i]` for each
//! row. The kernel is `kernels/bias.metal`'s `bias_rows`, one thread a float4
//! of a row. The query, key and value biases are one vector, laid in the
//! P1 output's own order, so one launch adds all three.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};

use super::rows::check_strided;
use super::{Kernels, WeightAlign, args_bytes, check_weights_align, check_window_len, word};

/// The threads in a bias launch's threadgroup.
const GROUP_THREADS: usize = 64;

const WHAT: &str = "bias rows";

/// The bias of `rows` rows of `cols` values.
pub struct BiasRowsParams<'w> {
    /// Rows in the chunk. Zero is refused.
    pub rows: usize,
    /// Values in a row: a multiple of four.
    pub cols: usize,
    /// The rows, `y_stride` elements apart, added in place.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// Elements between rows of `y`: a multiple of four, at least `cols`.
    pub y_stride: usize,
    /// The bias, `cols` values on a 16-byte boundary.
    pub bias: Window<'w, f32, MetalDevice>,
}

impl Kernels {
    /// Encode the bias add of a chunk of rows: one dispatch, a thread a float4.
    /// A shape the kernel does not take or a window that is not that shape is
    /// refused by name and nothing is encoded.
    #[must_use = "a refused launch encoded nothing"]
    pub fn bias_rows(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: BiasRowsParams<'_>,
    ) -> Result<(), DeviceError> {
        if p.rows == 0 || p.cols == 0 {
            return Err(super::range(
                WHAT,
                format!("{} rows of {} values is empty", p.rows, p.cols),
            ));
        }
        if !p.cols.is_multiple_of(4) {
            return Err(super::range(
                WHAT,
                format!("{} values a row are not whole float4s", p.cols),
            ));
        }
        check_strided(WHAT, "y", p.y.len(), p.rows, p.y_stride, p.cols)?;
        if !p.y_stride.is_multiple_of(4) {
            return Err(super::range(
                WHAT,
                format!("a row stride of {} is not whole float4s", p.y_stride),
            ));
        }
        check_window_len(WHAT, "bias", p.bias.len(), p.cols)?;
        for (name, start) in [
            ("y", p.y.byte_range()?.start.get()),
            ("bias", p.bias.byte_range()?.start.get()),
        ] {
            check_weights_align(WHAT, name, start, WeightAlign::Float4)?;
        }
        let cols4 = p.cols / 4;
        let args = args_bytes([
            word(WHAT, "the float4 count", cols4)?,
            word(WHAT, "the row stride", p.y_stride / 4)?,
        ]);
        encoder.set_pipeline(&self.bias_rows);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::of(&p.bias)?)?;
        slot.set(Binding::bytes(args.as_flattened())?)?;
        slot.dispatch_groups(cols4.div_ceil(GROUP_THREADS), p.rows, GROUP_THREADS)
    }
}
