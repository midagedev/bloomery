//! The Mamba-2 launchers: the decode step and the grouped RMS norm over its
//! output (`kernels/mamba2.metal`, whose header states the rule).
//!
//! The step is one dispatch of `n_head · chunks` threadgroups of 128 threads:
//! a threadgroup takes 16 channels of one head. It reads the state lane the
//! word at `lane_at` names and writes the lane the word at `lane_to` names, as
//! the delta step does, and keeps the conv ring in place. It writes each
//! channel's gated `y` and each threadgroup's sum of squares. The norm is a
//! second dispatch of one threadgroup a group over those sums.
//!
//! Heads, groups and the head width are launch arguments; the state length
//! (128) and the conv taps (4) are the kernel's, and any other value is
//! refused by name before anything is encoded.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::library::{Library, Pipeline};

use super::delta_geom::RING_ROWS;
use super::mamba2_geom::{STATE, TAPS, THREADS, chunks as chunks_of};
use super::{Kernels, check_window_len, pack, range, word};

const STEP: &str = "mamba2 step";
const NORM: &str = "mamba2 norm";

/// The widths of one Mamba-2 layer, with the sizes they imply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mamba2Shape {
    /// Heads.
    pub n_head: usize,
    /// Groups that share one `B` and one `C`; `n_head` is a multiple of it.
    pub n_group: usize,
    /// Channels of a head.
    pub head_dim: usize,
    /// Heads of a group, `n_head / n_group`.
    pub per: usize,
    /// Channels, `n_head · head_dim`.
    pub inner: usize,
    /// Conv channels, `inner + 2 · n_group · 128`.
    pub conv_dim: usize,
    /// Threadgroups of a head: `ceil(head_dim / 16)`.
    pub chunks: usize,
}

impl Mamba2Shape {
    /// The shape of `n_head` heads of `head_dim` channels in `n_group` groups,
    /// or the refusal of `what` that names the width that does not hold.
    pub fn of(
        what: &'static str,
        n_head: usize,
        n_group: usize,
        head_dim: usize,
    ) -> Result<Self, DeviceError> {
        if n_head == 0 {
            return Err(range(what, "n_head of 0"));
        }
        if n_group == 0 {
            return Err(range(what, "n_group of 0"));
        }
        if !n_head.is_multiple_of(n_group) {
            return Err(range(
                what,
                format!("n_head {n_head} is not a multiple of n_group {n_group}"),
            ));
        }
        if head_dim == 0 {
            return Err(range(what, "head_dim of 0"));
        }
        let inner = n_head
            .checked_mul(head_dim)
            .ok_or_else(|| range(what, format!("{n_head} heads of {head_dim} do not fit")))?;
        let conv_dim = n_group
            .checked_mul(2 * STATE)
            .and_then(|bc| bc.checked_add(inner))
            .ok_or_else(|| {
                range(
                    what,
                    format!("the conv channels of {n_group} groups do not fit"),
                )
            })?;
        let chunks = chunks_of(head_dim);
        // The kernel's indices are 32-bit words.
        for (name, n) in [
            ("the conv channels", conv_dim.checked_mul(RING_ROWS)),
            ("the blocks", n_head.checked_mul(chunks)),
        ] {
            let n = n.ok_or_else(|| range(what, format!("{name} do not fit")))?;
            word(what, name, n)?;
        }
        Ok(Self {
            n_head,
            n_group,
            head_dim,
            per: n_head / n_group,
            inner,
            conv_dim,
            chunks,
        })
    }

    /// Threadgroups of the step: one a chunk of each head.
    #[must_use]
    pub fn blocks(&self) -> usize {
        self.n_head * self.chunks
    }

    /// f32s of the scan state of `lanes` lanes: `lanes · inner · 128`.
    #[must_use]
    pub fn state_elems(&self, lanes: usize) -> Option<usize> {
        lanes
            .checked_mul(self.inner)
            .and_then(|n| n.checked_mul(STATE))
    }

    /// f32s of the conv ring: `RING_ROWS` rows of `conv_dim`.
    #[must_use]
    pub fn ring_elems(&self) -> usize {
        RING_ROWS * self.conv_dim
    }

    /// f32s of the sums of squares: one a threadgroup.
    #[must_use]
    pub fn partials(&self) -> usize {
        self.blocks()
    }
}

/// One Mamba-2 decode step.
pub struct Mamba2Params<'w> {
    /// The conv input: `x`, then `B` per group, then `C` per group.
    pub xbc: Window<'w, f32, MetalDevice>,
    /// The raw `dt`, one value per head, before its bias.
    pub dt: Window<'w, f32, MetalDevice>,
    /// The `dt` bias, one value per head.
    pub dt_bias: Window<'w, f32, MetalDevice>,
    /// The gate's input, one value per channel.
    pub z: Window<'w, f32, MetalDevice>,
    /// `A`, one value per head.
    pub a: Window<'w, f32, MetalDevice>,
    /// `D`, one value per head.
    pub d: Window<'w, f32, MetalDevice>,
    /// The conv weights, tap innermost, tap 0 the oldest input:
    /// `4 · conv_dim`.
    pub conv_w: Window<'w, f32, MetalDevice>,
    /// The conv bias, one value per conv channel.
    pub conv_b: Window<'w, f32, MetalDevice>,
    /// Lane words. The kernel reads the words at `lane_at` and `lane_to`.
    pub lane: Window<'w, u32, MetalDevice>,
    /// The scan state: the lane the word at `lane_at` names is read, the lane
    /// the word at `lane_to` names is written.
    pub state: WindowMut<'w, f32, MetalDevice>,
    /// The conv ring, read and written in place.
    pub ring: WindowMut<'w, f32, MetalDevice>,
    /// The gated `y`, one value per channel, before the norm.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// The sums of squares of `y`, one per threadgroup, in the order of
    /// [`Mamba2Shape::blocks`]; the norm's input.
    pub part: WindowMut<'w, f32, MetalDevice>,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// Heads of this layer.
    pub n_head: usize,
    /// Groups of this layer; `n_head` is a multiple of it.
    pub n_group: usize,
    /// Channels of a head.
    pub head_dim: usize,
    /// The scan's state length. The kernel runs [`STATE`].
    pub d_state: usize,
    /// Conv taps. The kernel runs [`TAPS`].
    pub taps: usize,
    /// Lanes of the state. Zero is refused.
    pub lanes: usize,
    /// The word of the lane this step reads.
    pub lane_at: usize,
    /// The word of the lane this step writes. A word naming the lane
    /// `lane_at`'s names updates that lane in place.
    pub lane_to: usize,
    /// The layer a raise is recorded in.
    pub layer: u32,
    /// The position window. A step reads one position.
    pub pos: &'w [u32],
}

/// The grouped RMS norm over a step's `y`: `gain · y / sqrt(mean + eps)` over
/// each group of `d_inner / n_group` channels, the mean taken from the sums
/// of squares the step wrote.
pub struct Mamba2NormParams<'w> {
    /// The gated `y` the step wrote.
    pub y: Window<'w, f32, MetalDevice>,
    /// The sums of squares the step wrote.
    pub part: Window<'w, f32, MetalDevice>,
    /// The norm gain, one value per channel.
    pub gain: Window<'w, f32, MetalDevice>,
    /// The normalized `y`, one value per channel.
    pub out: WindowMut<'w, f32, MetalDevice>,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// Heads of this layer.
    pub n_head: usize,
    /// Groups of this layer; `n_head` is a multiple of it.
    pub n_group: usize,
    /// Channels of a head.
    pub head_dim: usize,
    /// The layer a raise is recorded in.
    pub layer: u32,
    /// The norm's epsilon. It must be finite.
    pub eps: f32,
}

/// The pipelines of the step and the norm.
pub(crate) struct Pipelines {
    step: Pipeline,
    norm: Pipeline,
}

impl Pipelines {
    pub(crate) fn new(library: &Library) -> Result<Self, DeviceError> {
        Ok(Self {
            step: library.pipeline("mamba2_step")?,
            norm: library.pipeline("mamba2_norm")?,
        })
    }
}

/// The lane window holds the words the read and the write index name.
fn check_lane_words(held: usize, lane_at: usize, lane_to: usize) -> Result<(), DeviceError> {
    for (what, at) in [("read", lane_at), ("write", lane_to)] {
        let need = at
            .checked_add(1)
            .ok_or_else(|| range(STEP, format!("the {what} lane index {at} does not fit")))?;
        if held < need {
            return Err(range(
                STEP,
                format!("the lane window holds {held}, the {what} lane index needs {need}"),
            ));
        }
    }
    Ok(())
}

/// The shape the kernel takes, refused by name before any window is read.
fn check_step(p: &Mamba2Params<'_>) -> Result<Mamba2Shape, DeviceError> {
    if p.d_state != STATE {
        return Err(range(
            STEP,
            format!(
                "state {} is refused: the scan runs state {STATE}",
                p.d_state
            ),
        ));
    }
    if p.taps != TAPS {
        return Err(range(
            STEP,
            format!("conv taps {} are not the kernel's taps {TAPS}", p.taps),
        ));
    }
    let shape = Mamba2Shape::of(STEP, p.n_head, p.n_group, p.head_dim)?;
    if p.lanes == 0 {
        return Err(range(STEP, "lanes of 0"));
    }
    if p.pos.len() != 1 {
        return Err(range(
            STEP,
            format!(
                "the position window holds {}, the step takes one position",
                p.pos.len()
            ),
        ));
    }
    let state = shape
        .state_elems(p.lanes)
        .ok_or_else(|| range(STEP, format!("the state of {} lanes does not fit", p.lanes)))?;
    word(STEP, "the state length", state)?;
    check_window_len(STEP, "xbc", p.xbc.len(), shape.conv_dim)?;
    check_window_len(STEP, "dt", p.dt.len(), shape.n_head)?;
    check_window_len(STEP, "dt bias", p.dt_bias.len(), shape.n_head)?;
    check_window_len(STEP, "z", p.z.len(), shape.inner)?;
    check_window_len(STEP, "a", p.a.len(), shape.n_head)?;
    check_window_len(STEP, "d", p.d.len(), shape.n_head)?;
    check_window_len(STEP, "conv weight", p.conv_w.len(), TAPS * shape.conv_dim)?;
    check_window_len(STEP, "conv bias", p.conv_b.len(), shape.conv_dim)?;
    check_lane_words(p.lane.len().get(), p.lane_at, p.lane_to)?;
    check_window_len(STEP, "state", p.state.len(), state)?;
    check_window_len(STEP, "ring", p.ring.len(), shape.ring_elems())?;
    check_window_len(STEP, "y", p.y.len(), shape.inner)?;
    check_window_len(STEP, "partials", p.part.len(), shape.partials())?;
    Ok(shape)
}

/// The shape and windows the norm takes, refused by name before anything is
/// encoded.
fn check_norm(p: &Mamba2NormParams<'_>) -> Result<Mamba2Shape, DeviceError> {
    let shape = Mamba2Shape::of(NORM, p.n_head, p.n_group, p.head_dim)?;
    if !p.eps.is_finite() {
        return Err(range(
            NORM,
            format!("the norm epsilon {} is not finite", p.eps),
        ));
    }
    check_window_len(NORM, "y", p.y.len(), shape.inner)?;
    check_window_len(NORM, "partials", p.part.len(), shape.partials())?;
    check_window_len(NORM, "gain", p.gain.len(), shape.inner)?;
    check_window_len(NORM, "out", p.out.len(), shape.inner)?;
    Ok(shape)
}

impl Kernels {
    /// Encode one Mamba-2 decode step on `encoder`. The shape is checked
    /// before any window length, and a refused launch encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn mamba2_step(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: Mamba2Params<'_>,
    ) -> Result<(), DeviceError> {
        let shape = check_step(&p)?;
        let words = [
            word(STEP, "n_head", shape.n_head)?,
            word(STEP, "n_group", shape.n_group)?,
            word(STEP, "head_dim", shape.head_dim)?,
            word(STEP, "the chunk count", shape.chunks)?,
            word(STEP, "the lane count", p.lanes)?,
            word(STEP, "the read lane index", p.lane_at)?,
            word(STEP, "the write lane index", p.lane_to)?,
            p.layer,
            p.pos[0],
        ]
        .map(u32::to_ne_bytes);
        encoder.set_pipeline(&self.mamba2.step);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.xbc)?)?;
        slot.set(Binding::of(&p.dt)?)?;
        slot.set(Binding::of(&p.dt_bias)?)?;
        slot.set(Binding::of(&p.z)?)?;
        slot.set(Binding::of(&p.a)?)?;
        slot.set(Binding::of(&p.d)?)?;
        slot.set(Binding::of(&p.conv_w)?)?;
        slot.set(Binding::of(&p.conv_b)?)?;
        slot.set(Binding::of(&p.lane)?)?;
        slot.set(Binding::of_mut(&p.state)?)?;
        slot.set(Binding::of_mut(&p.ring)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::of_mut(&p.part)?)?;
        slot.set(Binding::bytes(pack(&words))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(shape.blocks(), 1, THREADS)
    }

    /// Encode the grouped RMS norm over a step's `y` on `encoder`: one
    /// threadgroup a group. The shape is checked before any window length, and
    /// a refused launch encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn mamba2_norm(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: Mamba2NormParams<'_>,
    ) -> Result<(), DeviceError> {
        let shape = check_norm(&p)?;
        let words = [
            word(NORM, "n_head", shape.n_head)?,
            word(NORM, "n_group", shape.n_group)?,
            word(NORM, "head_dim", shape.head_dim)?,
            word(NORM, "the chunk count", shape.chunks)?,
            p.layer,
            p.eps.to_bits(),
        ]
        .map(u32::to_ne_bytes);
        encoder.set_pipeline(&self.mamba2.norm);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.y)?)?;
        slot.set(Binding::of(&p.part)?)?;
        slot.set(Binding::of(&p.gain)?)?;
        slot.set(Binding::of_mut(&p.out)?)?;
        slot.set(Binding::bytes(pack(&words))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(shape.n_group, 1, THREADS)
    }
}
