//! The production taps of a Qwen3.6 body: the residual rows entering the
//! layers a block draft reads (HF's `hidden_states[L]`; entering layer 0 is
//! the embedding row), laid into one device buffer by every walk of the
//! whole-card load — the decode step, a pass of rows, each prompt unit —
//! and read back by the session ([`GpuModel::draft_taps`]) or, for a prompt
//! call, after each unit that holds a position of its window
//! ([`GpuModel::prefill_tapped`]).
//!
//! They are not the gate instrument ([`GpuModel::set_layer_taps`], every
//! layer's output of the decode chain, eager): the two are independent, a
//! step runs with both set, the instrument keeps refusing a prompt unit by
//! name, and these taps refuse a placed load and a pass of several slots.
//!
//! The buffer is `[tap][row][hidden]` f32, `rows_cap` rows a tap, so one
//! tapped layer's rows of a walk are one contiguous device copy; the
//! readback interleaves them into a position's row of the taps' layers in
//! the order they were named.

use super::body35::{Body35, FIT_RESERVE};
use super::prefill::PrefillPath;
use super::program::{NoLeg, Program, Stores};
use crate::model::GpuModel;
use crate::tensor::{Window, WindowMut};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaStream, DeviceBuffer};
use runtime::sched::{At, LayerProgram};

const WHAT: &str = "qwen35moe::draft_taps";

/// Where [`GpuModel::prefill_tapped`] hands its rows: the first position
/// they hold, then one tapped row a position.
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
pub(super) struct DraftTaps {
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
    pub(super) fn new(
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

    /// Floats one position's tapped row holds.
    pub(super) fn width(&self) -> usize {
        self.layers.len() * self.hidden
    }

    pub(super) fn bytes(&self) -> usize {
        self.buf.num_bytes()
    }

    /// The tap slot of `layer`, if it is tapped.
    fn slot(&self, layer: usize) -> Option<usize> {
        self.layers.iter().position(|&l| l == layer)
    }

    /// The walk about to run lays in `rows` rows: what [`DraftTaps::read`]
    /// may ask for until the next walk is planned.
    pub(super) fn plan(&mut self, rows: usize) {
        self.last = rows;
    }

    /// Byte offset of row `row` of tap slot `slot`.
    fn offset(&self, slot: usize, row: usize) -> usize {
        (slot * self.rows_cap + row) * self.hidden * size_of::<f32>()
    }

    /// Enqueue the copy of the first `m` rows of `x`, the rows entering the
    /// layer of `slot`, into the slot's rows. One device copy, launched in a
    /// capture as in an eager walk.
    pub(super) fn enter(
        &mut self,
        stream: &CudaStream,
        slot: usize,
        x: &DeviceBuffer<f32>,
        m: usize,
    ) -> Result<(), GpuError> {
        if m == 0 || m > self.rows_cap {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a walk of {m} rows over taps of {} rows (set the taps again after a larger \
                     ubatch)",
                    self.rows_cap
                ),
            ));
        }
        let len = m * self.hidden;
        let src = Window::<f32>::of(x, 0, len)?;
        let at = self.offset(slot, 0);
        let mut dst = WindowMut::<f32>::of_mut(&mut self.buf, at, len)?;
        dst.copy_from_device_async(&src, stream)?;
        Ok(())
    }

    /// Rows `from .. from + rows` of the last walk, one [`DraftTaps::width`]
    /// row each, a position's taps in the order the layers were named.
    /// Blocking. Refused by name past the rows the last walk laid in.
    pub(super) fn read(
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

/// A [`Program`] that also lays in the rows entering the layers a
/// [`DraftTaps`] names: its tapped layers run with the copy between mixer
/// and FFN ([`Program::layer`]), the rest as the program does.
pub(super) struct Tapping<'t, 'p, S: Stores + ?Sized> {
    program: &'t mut Program<'p, S>,
    taps: &'t mut DraftTaps,
}

impl<'t, 'p, S: Stores + ?Sized> Tapping<'t, 'p, S> {
    pub(super) fn new(program: &'t mut Program<'p, S>, taps: &'t mut DraftTaps) -> Self {
        Tapping { program, taps }
    }
}

impl<S: Stores + ?Sized> LayerProgram for Tapping<'_, '_, S> {
    type Port = NoLeg;

    fn front(&mut self, _: &mut NoLeg, at: At) -> Result<(), GpuError> {
        let entering = self.taps.slot(at.layer).map(|slot| (&mut *self.taps, slot));
        self.program.layer(at.layer, entering)
    }

    fn end(&mut self, unit: usize) -> Result<(), GpuError> {
        self.program.end(unit)
    }
}

impl GpuModel<Body35> {
    /// Tap the rows entering `layers` (strictly rising, each below the
    /// layer count; empty stops tapping) from the next walk on: every
    /// walk — a step, a pass of rows, each prompt unit — lays them into a
    /// buffer of the ubatch arena's rows a layer, allocated here against the
    /// card's free bytes with [`FIT_RESERVE`] left. The captured chains are
    /// dropped first (a capture records the copies into the buffer of its
    /// time), and the next graph step or pass recaptures with the copies. A
    /// placed load, whose walks run through the host tier's legs, is
    /// refused by name; so is a pass of several slots while taps are set. A
    /// refused call leaves taps off. Load-time allocation.
    pub fn set_draft_taps(&mut self, layers: &[usize]) -> Result<(), GpuError> {
        const WHAT_S: &str = "qwen35moe::set_draft_taps";
        self.drop_captures();
        let (gpu, _, body) = self.body_parts(WHAT_S)?;
        body.draft = None;
        if layers.is_empty() {
            return Ok(());
        }
        if body.placed.is_some() {
            return Err(GpuError::shape(
                WHAT_S,
                "taps on a placed load: its walks run through the host tier's legs, which lay no \
                 rows in",
            ));
        }
        let (hidden, rows_cap) = (body.s.dims.hidden, body.u.rows.max(body.a.rows));
        let free = gpu.alloc_room()?;
        body.draft = Some(DraftTaps::new(
            gpu.stream(),
            layers,
            body.plans.len(),
            (hidden, rows_cap),
            free,
        )?);
        Ok(())
    }

    /// Floats one position's tapped row holds: the tapped layers times the
    /// body's width; 0 with no taps set.
    #[must_use]
    pub fn draft_tap_width(&self) -> usize {
        self.body(WHAT)
            .map_or(0, |b| b.draft.as_ref().map_or(0, DraftTaps::width))
    }

    /// The first `rows` rows of the last walk — a step's one, a verify's
    /// `m`, a prompt unit's — one [`GpuModel::draft_tap_width`] row each.
    /// Blocking. Refused by name with no taps set, and for more rows than
    /// the last walk laid in.
    pub fn draft_taps(&mut self, rows: usize) -> Result<Vec<f32>, GpuError> {
        self.draft_tap_rows(0, rows)
    }

    fn draft_tap_rows(&mut self, from: usize, rows: usize) -> Result<Vec<f32>, GpuError> {
        const WHAT_T: &str = "qwen35moe::draft_taps";
        let (gpu, _, body) = self.body_parts(WHAT_T)?;
        body.draft
            .as_ref()
            .ok_or(GpuError::state(WHAT_T, "taps set (set_draft_taps)"))?
            .read(gpu.stream(), from, rows)
    }

    /// [`GpuModel::prefill_with`] with the tapped rows of the call's last
    /// `window` positions (all of them when the call is shorter) handed to
    /// `sink` as `(first position, rows)`, one [`GpuModel::draft_tap_width`]
    /// row a position, after each unit that holds any of them — in order,
    /// no position twice or missed. The units are the plain call's, so its
    /// cache, state and argmax are bit for bit. Refused by name with no taps
    /// set; a sink that fails ends the call with its error.
    pub fn prefill_tapped(
        &mut self,
        tokens: &[u32],
        path: PrefillPath,
        window: usize,
        sink: &mut TapRowsSink<'_>,
    ) -> Result<u32, GpuError> {
        const WHAT_P: &str = "qwen35moe::prefill_tapped";
        if self.draft_tap_width() == 0 {
            return Err(GpuError::state(WHAT_P, "taps set (set_draft_taps)"));
        }
        let end = self.pos() + launch_u32(WHAT_P, "tokens", tokens.len())?;
        let held = end - launch_u32(WHAT_P, "window", window.min(tokens.len()))?;
        let mut unit = |m: &mut GpuModel<Body35>, first: u32, rows: usize| {
            let past = first + launch_u32(WHAT_P, "rows", rows)?;
            if past <= held {
                return Ok(());
            }
            let from = first.max(held);
            let skip = (from - first) as usize;
            let rows = m.draft_tap_rows(skip, rows - skip)?;
            sink(from, &rows)
        };
        self.prefill_hooked(tokens, path, Some(&mut unit))
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
