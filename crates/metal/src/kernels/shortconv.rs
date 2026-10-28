//! The short conv of a gated-conv mixer: `y_t[c] = C_t[c] · Σ_k w[c][k] ·
//! u_{t-(taps-1)+k}[c]` with `u = B ⊙ x`, causal and depthwise, oldest tap
//! first (`kernels/shortconv.metal`). The input is the in_proj output, a row of
//! `B | C | x` of `hidden` values each; the weight is `[hidden][taps]`.
//!
//! The conv's width (`taps`), its ring's rows and the channels are launch
//! arguments, so any conv width runs. The ring keeps `u` of the last positions
//! of a sequence, position `p` in slot `p mod ring_rows`; a predecessor before
//! the sequence's first position is 0 whatever the ring holds. The ring has at
//! least `taps` rows, so the slot a token writes is not one it reads.
//!
//! [`Kernels::shortconv_step`] is one row per slot, each slot at its own
//! position over its own ring (the multi-stream decode step).
//! [`Kernels::shortconv_rows`] is the rows of one sequence's chunk and the
//! ring commit after it, which a later chunk or step reads: a chunk's rows are
//! the bits of stepped launches at their positions.
//!
//! A value that is not finite raises the linear-conv site and the output is
//! NaN at that element.

use bloomery_device::{DeviceError, Elems, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::library::{Library, Pipeline};

use super::rows::check_strided;
use super::shortconv_geom::{MAX_SLOTS, THREADS, groups, kept_rows, ring_elems};
use super::{Kernels, check_window_len, pack, range, word};

const WHAT: &str = "short conv";

/// One decode step of `pos.len()` slots: row `s` of the input is slot `s`'s
/// token, at position `pos[s]`, over slot `s`'s ring.
pub struct ShortConvStepParams<'w> {
    /// The in_proj output, one row a slot, `bcx_stride` elements apart.
    pub bcx: Window<'w, f32, MetalDevice>,
    /// Elements between rows of `bcx`: at least `3 · hidden`.
    pub bcx_stride: usize,
    /// The conv weights, `hidden · taps`, a channel's taps together, oldest first.
    pub conv_w: Window<'w, f32, MetalDevice>,
    /// The rings, `slots · ring_rows · hidden`: slot `s`'s ring is the `s`-th
    /// `ring_rows` rows. Read and written in place.
    pub ring: WindowMut<'w, f32, MetalDevice>,
    /// The output, one row a slot, `y_stride` elements apart.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// Elements between rows of `y`: at least `hidden`.
    pub y_stride: usize,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// Channels. Zero is refused.
    pub hidden: usize,
    /// Conv taps. Zero is refused.
    pub taps: usize,
    /// Rows of one slot's ring: at least `taps`.
    pub ring_rows: usize,
    /// The layer a raise is recorded in.
    pub layer: u32,
    /// The position of each slot's token: one to [`MAX_SLOTS`] words.
    pub pos: &'w [u32],
}

/// One chunk of `rows` rows of one sequence, from position `pos`.
pub struct ShortConvRowsParams<'w> {
    /// Rows in the chunk. Zero is refused.
    pub rows: usize,
    /// The in_proj output, `bcx_stride` elements between rows.
    pub bcx: Window<'w, f32, MetalDevice>,
    /// Elements between rows of `bcx`: at least `3 · hidden`.
    pub bcx_stride: usize,
    /// The conv weights, `hidden · taps`, a channel's taps together, oldest first.
    pub conv_w: Window<'w, f32, MetalDevice>,
    /// The sequence's ring, `ring_rows · hidden`: the pre-chunk slots are
    /// read, then the last `min(rows, ring_rows)` positions' slots are written.
    pub ring: WindowMut<'w, f32, MetalDevice>,
    /// The output, `y_stride` elements between rows.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// Elements between rows of `y`: at least `hidden`.
    pub y_stride: usize,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// Channels. Zero is refused.
    pub hidden: usize,
    /// Conv taps. Zero is refused.
    pub taps: usize,
    /// Rows of the ring: at least `taps`.
    pub ring_rows: usize,
    /// The layer a raise is recorded in.
    pub layer: u32,
    /// The position of the chunk's first row.
    pub pos: u32,
}

/// The pipelines of the short conv: the step, the chunk's rows, and the ring
/// commit after a chunk.
pub(crate) struct Pipelines {
    step: Pipeline,
    rows: Pipeline,
    ring: Pipeline,
}

impl Pipelines {
    pub(crate) fn new(library: &Library) -> Result<Self, DeviceError> {
        Ok(Self {
            step: library.pipeline("shortconv_step")?,
            rows: library.pipeline("shortconv_rows")?,
            ring: library.pipeline("shortconv_ring_rows")?,
        })
    }
}

fn refuse(detail: impl Into<String>) -> DeviceError {
    range(WHAT, detail)
}

/// What both launches share: the shape, and the window lengths against it.
struct Shape {
    hidden: usize,
    taps: usize,
    ring_rows: usize,
    /// The slots of a step, the rows of a chunk.
    rows: usize,
    /// Rings the ring window holds: the slots of a step, one for a chunk.
    rings: usize,
    bcx_stride: usize,
    y_stride: usize,
}

/// The windows' lengths.
struct Held {
    bcx: Elems,
    conv_w: Elems,
    ring: Elems,
    y: Elems,
}

/// The shape and the windows against it, refused by name before anything is
/// encoded.
fn check(s: &Shape, held: &Held, unit: &str) -> Result<(), DeviceError> {
    if s.rows == 0 {
        return Err(refuse(format!("{unit} of 0")));
    }
    if s.hidden == 0 {
        return Err(refuse("hidden of 0"));
    }
    if s.taps == 0 {
        return Err(refuse("taps of 0"));
    }
    if s.ring_rows < s.taps {
        return Err(refuse(format!(
            "ring rows {} are fewer than the {} taps: the slot a token writes would be one it reads",
            s.ring_rows, s.taps
        )));
    }
    let wide = s
        .hidden
        .checked_mul(3)
        .ok_or_else(|| refuse(format!("hidden {} does not fit three times", s.hidden)))?;
    check_strided(WHAT, "bcx", held.bcx, s.rows, s.bcx_stride, wide)?;
    check_strided(WHAT, "y", held.y, s.rows, s.y_stride, s.hidden)?;
    let weights = s
        .hidden
        .checked_mul(s.taps)
        .ok_or_else(|| refuse(format!("conv weights of {} channels do not fit", s.hidden)))?;
    check_window_len(WHAT, "conv weight", held.conv_w, weights)?;
    let ring = ring_elems(s.rings, s.ring_rows, s.hidden)
        .ok_or_else(|| refuse(format!("the ring of {} rows does not fit", s.ring_rows)))?;
    word(WHAT, "the ring length", ring)?;
    check_window_len(WHAT, "ring", held.ring, ring)
}

/// The argument block's eight words, in the kernel's `ShortConvArgs` order.
fn arg_words(s: &Shape, layer: u32, pos: u32) -> Result<[[u8; 4]; 8], DeviceError> {
    Ok([
        word(WHAT, "taps", s.taps)?,
        word(WHAT, "ring rows", s.ring_rows)?,
        word(WHAT, "hidden", s.hidden)?,
        word(WHAT, "rows", s.rows)?,
        word(WHAT, "the bcx stride", s.bcx_stride)?,
        word(WHAT, "the y stride", s.y_stride)?,
        layer,
        pos,
    ]
    .map(u32::to_ne_bytes))
}

impl Kernels {
    /// Encode one decode step of `p.pos.len()` slots: one dispatch, a thread
    /// a (channel, slot). Each thread reads its predecessors and then writes
    /// its token's `u` into its slot's ring. The shape and every window are
    /// checked before any dispatch, and a refused launch encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn shortconv_step(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: ShortConvStepParams<'_>,
    ) -> Result<(), DeviceError> {
        let slots = p.pos.len();
        if slots > MAX_SLOTS {
            return Err(refuse(format!(
                "{slots} slots, the step takes at most {MAX_SLOTS}"
            )));
        }
        let shape = Shape {
            hidden: p.hidden,
            taps: p.taps,
            ring_rows: p.ring_rows,
            rows: slots,
            rings: slots,
            bcx_stride: p.bcx_stride,
            y_stride: p.y_stride,
        };
        let held = Held {
            bcx: p.bcx.len(),
            conv_w: p.conv_w.len(),
            ring: p.ring.len(),
            y: p.y.len(),
        };
        check(&shape, &held, "slots")?;
        let args = arg_words(&shape, p.layer, 0)?;
        // A count past MAX_SLOTS is refused above, so every position has a lane.
        let mut pos = [[0u8; 4]; MAX_SLOTS];
        for (lane, q) in pos.iter_mut().zip(p.pos) {
            *lane = q.to_ne_bytes();
        }
        encoder.set_pipeline(&self.shortconv.step);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.bcx)?)?;
        slot.set(Binding::of(&p.conv_w)?)?;
        slot.set(Binding::of_mut(&p.ring)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(Binding::bytes(pos[..slots].as_flattened())?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(groups(p.hidden), slots, THREADS)
    }

    /// Encode a chunk of `p.rows` rows of one sequence: the rows dispatch, a
    /// memory barrier, and the ring commit. The ring commit is the last
    /// reader's successor: the rows read the pre-chunk slots first. The shape
    /// and every window are checked before any dispatch, and a refused launch
    /// encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn shortconv_rows(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: ShortConvRowsParams<'_>,
    ) -> Result<(), DeviceError> {
        let shape = Shape {
            hidden: p.hidden,
            taps: p.taps,
            ring_rows: p.ring_rows,
            rows: p.rows,
            rings: 1,
            bcx_stride: p.bcx_stride,
            y_stride: p.y_stride,
        };
        let held = Held {
            bcx: p.bcx.len(),
            conv_w: p.conv_w.len(),
            ring: p.ring.len(),
            y: p.y.len(),
        };
        check(&shape, &held, "rows")?;
        let rows = word(WHAT, "rows", p.rows)?;
        if p.pos.checked_add(rows).is_none() {
            return Err(refuse(format!(
                "positions {}..+{rows} pass the 32-bit position word",
                p.pos
            )));
        }
        let args = arg_words(&shape, p.layer, p.pos)?;

        encoder.set_pipeline(&self.shortconv.rows);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.bcx)?)?;
        slot.set(Binding::of(&p.conv_w)?)?;
        slot.set(Binding::of_mut(&p.ring)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(groups(p.hidden), p.rows, THREADS)?;

        encoder.memory_barrier();
        encoder.set_pipeline(&self.shortconv.ring);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.bcx)?)?;
        slot.set(Binding::of_mut(&p.ring)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.dispatch_groups(groups(p.hidden), kept_rows(p.rows, p.ring_rows), THREADS)
    }
}
