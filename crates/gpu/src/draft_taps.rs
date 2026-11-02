//! The taps a block draft reads: the rows entering the layers it names (HF's
//! `hidden_states[L]`), laid by a body's walks into one device buffer and
//! read back by its session. A body holds one [`DraftTaps`] a kind of walk,
//! copies each tapped layer's rows in with [`DraftTaps::enter`] (launched in
//! a capture as in an eager walk), and plans what a read may ask for where
//! the host plans the walk ([`DraftTaps::plan`]). `n_layer` is the count of
//! hidden states the body can lay in: the layers' when the last layer's
//! output is not one, one more when it is.
//!
//! The buffer is `[tap][row][hidden]` f32, `rows_cap` rows a tap, so one
//! tapped layer's rows of a walk are one contiguous device copy; the
//! readback interleaves them into a position's row of the taps' layers in
//! the order they were named.

use crate::GpuError;
use crate::model::fit::FIT_RESERVE;
use crate::tensor::{Window, WindowMut};
use cuda_core::{CudaStream, DeviceBuffer};

const WHAT: &str = "draft_taps";

/// Where a tapped prompt call hands its rows: the first position they hold,
/// then one tapped row a position.
pub type TapRowsSink<'a> = dyn FnMut(u32, &[f32]) -> Result<(), GpuError> + 'a;

/// The floats of the taps' buffer for `layers` (`rows_cap` rows of `hidden`
/// values a layer), once the layers are checked: strictly rising, each below
/// `n_layer`; and the bytes with [`FIT_RESERVE`] more within `free`. Each
/// refusal is by name.
fn buffer_len(
    layers: &[usize],
    n_layer: usize,
    (hidden, rows_cap): (usize, usize),
    free: usize,
) -> Result<usize, GpuError> {
    if layers.windows(2).any(|w| w[0] >= w[1]) {
        return Err(GpuError::shape(
            WHAT,
            format!("tapped layers {layers:?} are not strictly rising"),
        ));
    }
    if let Some(&l) = layers.iter().find(|&&l| l >= n_layer) {
        return Err(GpuError::shape(
            WHAT,
            format!("tapped layer {l} of a model of {n_layer} layers"),
        ));
    }
    let len = rows_cap
        .checked_mul(layers.len())
        .and_then(|n| n.checked_mul(hidden))
        .ok_or_else(|| GpuError::shape(WHAT, "the taps' buffer overflows usize"))?;
    let need = len * size_of::<f32>();
    if need + FIT_RESERVE > free {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "the taps at {layers:?} ({rows_cap} rows of {hidden} values a layer) need {need} \
                 bytes and {FIT_RESERVE} more stay free for what the load allocates after them; \
                 the card has {free} free"
            ),
        ));
    }
    Ok(len)
}

/// The tapped layers and their buffer.
pub struct DraftTaps {
    layers: Vec<usize>,
    hidden: usize,
    rows_cap: usize,
    buf: DeviceBuffer<f32>,
    /// The rows the last walk laid in, set where the host plans the walk (a
    /// replayed graph runs no host code): what a read may ask for.
    last: usize,
}

impl DraftTaps {
    /// The buffer for taps at `layers` (strictly rising, each below
    /// `n_layer`) of a body of `hidden` values a row and walks of up to
    /// `rows_cap` rows, against `free` device bytes with [`FIT_RESERVE`]
    /// left ([`buffer_len`]).
    pub fn new(
        stream: &CudaStream,
        layers: &[usize],
        n_layer: usize,
        (hidden, rows_cap): (usize, usize),
        free: usize,
    ) -> Result<DraftTaps, GpuError> {
        let len = buffer_len(layers, n_layer, (hidden, rows_cap), free)?;
        Ok(DraftTaps {
            layers: layers.to_vec(),
            hidden,
            rows_cap,
            buf: DeviceBuffer::<f32>::zeroed(stream, len)?,
            last: 0,
        })
    }

    /// The tapped layers, in the order they were named.
    #[must_use]
    pub fn layers(&self) -> &[usize] {
        &self.layers
    }

    /// Floats one position's tapped row holds.
    #[must_use]
    pub fn width(&self) -> usize {
        self.layers.len() * self.hidden
    }

    /// Device bytes of the taps' buffer.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.buf.num_bytes()
    }

    /// The tap slot of `layer`, if it is tapped.
    #[must_use]
    pub fn slot(&self, layer: usize) -> Option<usize> {
        self.layers.iter().position(|&l| l == layer)
    }

    /// The walk about to run lays in `rows` rows: what [`DraftTaps::read`]
    /// may ask for until the next walk is planned.
    pub fn plan(&mut self, rows: usize) {
        self.last = rows;
    }

    /// Byte offset of row `row` of tap slot `slot`.
    fn offset(&self, slot: usize, row: usize) -> usize {
        (slot * self.rows_cap + row) * self.hidden * size_of::<f32>()
    }

    /// Enqueue the copy of the first `m` rows of `x`, the rows entering the
    /// layer of `slot`, into the slot's rows. One device copy, launched in a
    /// capture as in an eager walk.
    pub fn enter(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        x: &DeviceBuffer<f32>,
        m: usize,
    ) -> Result<(), GpuError> {
        self.enter_at(stream, slot, x, (0, m))
    }

    /// [`DraftTaps::enter`] into the slot's rows from `at`: the chunk of a
    /// walk planned over several, whose rows are `at ..` of the plan.
    pub fn enter_at(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        x: &DeviceBuffer<f32>,
        (at, m): (usize, usize),
    ) -> Result<(), GpuError> {
        if m == 0 || at + m > self.rows_cap {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a walk of {m} rows from row {at} over taps of {} rows (set the taps again \
                     after a larger walk)",
                    self.rows_cap
                ),
            ));
        }
        let len = m * self.hidden;
        let src = Window::<f32>::of(x, 0, len)?;
        let off = self.offset(slot, at);
        let mut dst = WindowMut::<f32>::of_mut(&mut self.buf, off, len)?;
        dst.copy_from_device_async(&src, stream)?;
        Ok(())
    }

    /// Rows `from .. from + rows` of the last walk, one [`DraftTaps::width`]
    /// row each, a position's taps in the order the layers were named.
    /// Blocking. Refused by name past the rows the last walk laid in.
    pub fn read(
        &self,
        stream: &CudaStream,
        from: usize,
        rows: usize,
    ) -> Result<Vec<f32>, GpuError> {
        if rows == 0 || from + rows > self.last {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "rows {from}..{} of the last walk, which laid in {} (a walk before the read)",
                    from + rows,
                    self.last
                ),
            ));
        }
        let width = self.width();
        let mut out = vec![0f32; rows * width];
        for slot in 0..self.layers.len() {
            let tap = Window::<f32>::of(&self.buf, self.offset(slot, from), rows * self.hidden)?
                .to_host_vec(stream)?;
            for (r, row) in tap.chunks_exact(self.hidden).enumerate() {
                let at = r * width + slot * self.hidden;
                out[at..at + self.hidden].copy_from_slice(row);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::{FIT_RESERVE, buffer_len};

    /// Qwen3.6-35B-A3B's shape: 40 layers of 2,048 values, a 4,096-row
    /// ubatch arena.
    const SHAPE: (usize, usize) = (2048, 4096);

    /// The buffer is `rows_cap · taps · hidden` floats, and a card with
    /// exactly its bytes and the reserve holds it.
    #[test]
    fn the_buffer_holds_a_row_of_every_tap_and_fits_to_the_reserve() {
        let layers = [0, 9, 19, 29, 39];
        let len = buffer_len(&layers, 40, SHAPE, usize::MAX / 2).unwrap();
        assert_eq!(len, 4096 * 5 * 2048);
        let need = len * 4;
        assert!(buffer_len(&layers, 40, SHAPE, need + FIT_RESERVE).is_ok());
        let short = buffer_len(&layers, 40, SHAPE, need + FIT_RESERVE - 1).unwrap_err();
        assert!(short.to_string().contains("stay free"), "{short}");
    }

    /// Layers out of order, repeated or past the model are refused by name.
    #[test]
    fn layers_out_of_order_or_past_the_model_are_refused_by_name() {
        let free = usize::MAX / 2;
        for bad in [&[3usize, 3][..], &[5, 2][..]] {
            let e = buffer_len(bad, 40, SHAPE, free).unwrap_err();
            assert!(e.to_string().contains("strictly rising"), "{e}");
        }
        let e = buffer_len(&[0, 40], 40, SHAPE, free).unwrap_err();
        assert!(e.to_string().contains("layer 40 of a model of 40"), "{e}");
        assert!(buffer_len(&[0, 39], 40, SHAPE, free).is_ok());
    }
}
