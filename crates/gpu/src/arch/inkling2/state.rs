//! The state book of Inkling 2's step: what a decode leaves on the card for the next one, one
//! store a visit (the reference gives every visit its own KV cache and its own convolution state;
//! a looped layer's visits share its weights alone).
//!
//! - A visit's K and V: head-major `[n_kv][rows][128]` f16 planes as `qk_clip` appends them. A
//!   windowed visit holds a ring of the plan's rows ([`runtime::stores::ring_rows`]: the window
//!   and a prompt pass's rows, padded) when that is under the context (position `p` at row
//!   `p mod rows`, the rel-bias ring entries' indexing), so a pass that appends its rows before
//!   it reads keeps every key its rows read; every other visit holds a plane of `ctx` rows. Only
//!   rows below the live count are read, and each is written by its own step or pass first, so a
//!   reset needs nothing of them.
//! - A visit's conv ring: [`Ring::values`] over the visit's whole channel width (the k and v
//!   paths, then the two post convs, `conv_res`'s one ring a visit). A position before the
//!   sequence start reads zero whatever the ring holds (`conv_res`'s rule); the ring is zeroed at
//!   load and at every reset all the same.
//! - The step's words ([`StepIn`]): the token, the position and the ring row of a windowed
//!   visit's append, one copy a refresh; the global visits' log-scaled position factor, one copy;
//!   the embedding writes the row's position and live key count from them.
//!
//! Every store's size is [`shapes`]'s, the one place a load sizes them.

use crate::DeviceBufferRaw;
use crate::GpuError;
use chain::linear::Ring;
use cuda_core::{CudaStream, DeviceBuffer};

/// Word slots of [`StepIn::words`].
pub(super) const WORD_TOKEN: usize = 0;
pub(super) const WORD_POS: usize = 1;
pub(super) const WORD_RING: usize = 2;
const WORDS: usize = 3;

/// One visit's stores as a load sizes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct StoreShape {
    /// Key heads and their values.
    pub(super) n_kv: usize,
    pub(super) head: usize,
    /// The planes' rows a key head: the ring's ([`runtime::stores::ring_rows`]), or `ctx`.
    pub(super) rows: usize,
    /// Whether the planes are a ring of `rows` rows (the append at the ring row).
    pub(super) ring: bool,
    /// The conv ring's f32 values: [`Ring::values`] of the visit's channels.
    pub(super) conv: usize,
}

/// The stores of every visit, in run order: `windowed[v]` whether visit `v` attends a window of
/// `window` positions, every visit over `ctx` positions with `n_kv` key heads of `head` values
/// and a conv ring of `ring` over `ch` channels. A windowed visit rings at
/// [`runtime::stores::ring_rows`] rows where those are under `ctx`.
pub(super) fn shapes(
    windowed: impl Iterator<Item = bool>,
    (n_kv, head): (usize, usize),
    (window, ctx): (usize, usize),
    (ring, ch): (Ring, usize),
) -> Vec<StoreShape> {
    let ring_rows = usize::try_from(runtime::stores::ring_rows(window as u64, ctx as u64))
        .unwrap_or(ctx)
        .min(ctx);
    windowed
        .map(|w| {
            let rings = w && ring_rows < ctx;
            StoreShape {
                n_kv,
                head,
                rows: if rings { ring_rows } else { ctx },
                ring: rings,
                conv: ring.values(ch),
            }
        })
        .collect()
}

/// One visit's stores read back: the K and V planes and the conv ring as the card holds them.
pub struct StoreHost {
    pub k: Vec<u16>,
    pub v: Vec<u16>,
    pub conv: Vec<f32>,
    /// The planes' rows a key head, and whether they ring (position `p` at row `p mod rows`).
    pub rows: usize,
    pub ring: bool,
}

/// One visit's stores.
pub(super) struct VisitStore {
    pub(super) kc: DeviceBuffer<u16>,
    pub(super) vc: DeviceBuffer<u16>,
    pub(super) shape: StoreShape,
    pub(super) conv: DeviceBuffer<f32>,
}

impl VisitStore {
    /// A visit's stores of `shape`, zeroed. Load-time only.
    pub(super) fn new(stream: &CudaStream, shape: StoreShape) -> Result<VisitStore, GpuError> {
        let plane = shape.n_kv * shape.rows * shape.head;
        Ok(VisitStore {
            kc: DeviceBuffer::zeroed(stream, plane)?,
            vc: DeviceBuffer::zeroed(stream, plane)?,
            shape,
            conv: DeviceBuffer::zeroed(stream, shape.conv)?,
        })
    }

    /// The conv ring back to zero, ordered on `stream`. Never inside a capture.
    pub(super) fn clear(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.conv.zero_raw(stream)?;
        Ok(())
    }

    pub(super) fn bytes(&self) -> usize {
        self.kc.num_bytes() + self.vc.num_bytes() + self.conv.num_bytes()
    }

    /// The stores as the card holds them, after `stream`'s work so far. A blocking read.
    pub(super) fn to_host(&self, stream: &CudaStream) -> Result<StoreHost, GpuError> {
        stream.synchronize()?;
        Ok(StoreHost {
            k: self.kc.to_host_vec(stream)?,
            v: self.vc.to_host_vec(stream)?,
            conv: self.conv.to_host_vec(stream)?,
            rows: self.shape.rows,
            ring: self.shape.ring,
        })
    }
}

/// The step's input, rewritten by every refresh: the words and the global visits' factor; the
/// embedding's outputs for the one row, its position and its live key count; `one` is the
/// windowed visits' factor, 1.0 for the whole load.
pub(super) struct StepIn {
    pub(super) words: DeviceBuffer<u32>,
    pub(super) tau: DeviceBuffer<f32>,
    pub(super) one: DeviceBuffer<f32>,
    pub(super) pos: DeviceBuffer<u32>,
    pub(super) n_keys: DeviceBuffer<u32>,
}

impl StepIn {
    pub(super) fn new(stream: &CudaStream) -> Result<StepIn, GpuError> {
        Ok(StepIn {
            words: DeviceBuffer::zeroed(stream, WORDS)?,
            tau: DeviceBuffer::from_host(stream, &[1.0f32])?,
            one: DeviceBuffer::from_host(stream, &[1.0f32])?,
            pos: DeviceBuffer::zeroed(stream, 1)?,
            n_keys: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    /// The words and the factor of the step at `pos` of `token` under rings of `window` rows (0:
    /// no visit rings): two copies.
    pub(super) fn refresh(
        &mut self,
        stream: &CudaStream,
        (token, pos): (u32, u32),
        (window, tau): (u32, f32),
    ) -> Result<(), GpuError> {
        let ring = if window == 0 { 0 } else { pos % window };
        self.words.copy_from_host(stream, &[token, pos, ring])?;
        self.tau.copy_from_host(stream, &[tau])?;
        Ok(())
    }

    pub(super) fn bytes(&self) -> usize {
        self.words.num_bytes()
            + self.tau.num_bytes()
            + self.one.num_bytes()
            + self.pos.num_bytes()
            + self.n_keys.num_bytes()
    }
}

/// The global visits' log-scaled position factor at `pos`: `1 + alpha · ln(max(1, (pos + 1) /
/// floor))` in the reference's f32 order (`inkling2.cpp`'s `tau`: the position plus one, times
/// `1 / floor`, clamped at 1, the log, times alpha, plus one).
#[must_use]
pub(super) fn tau_of(pos: u32, floor: u32, alpha: f32) -> f32 {
    let x = (pos as f32 + 1.0) * (1.0 / floor as f32);
    x.max(1.0).ln() * alpha + 1.0
}

#[cfg(test)]
mod tests {
    use super::{Ring, shapes, tau_of};

    /// A windowed visit rings at the window plus a prompt pass's rows, padded (1,024 + 511 up to
    /// 1,536), only under a longer context, and every visit's conv ring holds the ring's rows over
    /// its channels. Mutants: the ring at the window alone (1,024) fails the first; the ring sized
    /// at `reach()` (the rows a token reads) fails the conv count.
    #[test]
    fn the_shapes_ring_a_window_and_a_pass_under_the_context_alone() {
        let ring = Ring::new(4);
        let s = shapes(
            [true, false].into_iter(),
            (8, 128),
            (1024, 4096),
            (ring, 10),
        );
        assert_eq!((s[0].rows, s[0].ring), (1536, true));
        assert_eq!((s[1].rows, s[1].ring), (4096, false));
        assert_eq!(s[0].conv, ring.rows() * 10);
        for ctx in [512, 1536] {
            let short = shapes([true].into_iter(), (8, 128), (1024, ctx), (ring, 10));
            assert_eq!((short[0].rows, short[0].ring), (ctx, false));
        }
    }

    /// The factor is 1 through the floor and grows by alpha's log past it.
    #[test]
    fn tau_is_one_through_the_floor() {
        assert_eq!(tau_of(0, 8192, 0.1).to_bits(), 1.0f32.to_bits());
        assert_eq!(tau_of(8191, 8192, 0.1).to_bits(), 1.0f32.to_bits());
        let far = tau_of(16383, 8192, 0.1);
        assert!((far - (1.0 + 0.1 * 2.0f32.ln())).abs() < 1e-6, "{far}");
    }
}
