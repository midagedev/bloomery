//! The gated delta step: one decode token's conv, decay and delta update.
//!
//! The kernel writes the gated numerator and the partial sums of `o²`. The
//! head-normed mat-vec multiplies the norm scale on load; this launch does
//! not. The grid is fixed: one threadgroup per value-head column block, 128
//! threads in the group.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, ComputeEncoder};
use crate::fault::MetalFaultWord;

use super::geometry::{HEAD_COLS, PARTIALS_PER_HEAD, VALUE_HEADS};
use super::{Kernels, check_window_len, check_word, word};

/// Taps of the causal conv.
pub const CONV_TAPS: usize = 4;

/// Predecessors the conv reads: the taps except the current token.
pub const CONV_ROWS: usize = CONV_TAPS - 1;

/// Positions one step's position window may name.
pub const PASS_ROWS: usize = 8;

/// Slots of the conv ring: the predecessors plus the positions of one pass.
pub const RING_ROWS: usize = CONV_ROWS + PASS_ROWS;

/// `1 / sqrt(HEAD_COLS)` rounded to f32. The kernel multiplies the query by
/// this value; an oracle promotes these bits, rather than a fresh reciprocal.
pub const Q_SCALE: f32 = f32::from_bits(0x3db5_04f3);

/// Threads in the step's threadgroup: one per query and key channel of a head.
const GROUP_THREADS: usize = HEAD_COLS;

/// Threadgroups: one per value head, per column block.
const GROUPS: usize = VALUE_HEADS * PARTIALS_PER_HEAD;

const _: () = assert!(RING_ROWS == 11 && GROUPS == 256 && GROUP_THREADS == 128);
const _: () = assert!(HEAD_COLS / PARTIALS_PER_HEAD == 16);

const WHAT: &str = "delta step";

/// Key index of lane `j`'s value `i`: `32·(i/4) + 4·j + (i%4)`.
#[must_use]
pub fn key_of(j: usize, i: usize) -> usize {
    32 * (i / 4) + 4 * j + (i % 4)
}

/// How a value head finds its key head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadMap {
    /// `h % n_k`.
    Tiled,
    /// `h / (n_v / n_k)`.
    Grouped,
}

impl HeadMap {
    /// The key head of value head `h`.
    #[must_use]
    pub fn k_head(self, h: usize, n_k: usize) -> usize {
        match self {
            HeadMap::Tiled => h % n_k,
            HeadMap::Grouped => h / (VALUE_HEADS / n_k),
        }
    }

    fn word(self) -> u32 {
        match self {
            HeadMap::Tiled => 0,
            HeadMap::Grouped => 1,
        }
    }
}

/// Conv channels: `(2·n_k + n_v)·HEAD_COLS`.
#[must_use]
pub fn channels(n_k: usize) -> Option<usize> {
    n_k.checked_mul(2)
        .and_then(|k| k.checked_add(VALUE_HEADS))
        .and_then(|h| h.checked_mul(HEAD_COLS))
}

/// f32s of the recurrent state: `lanes·n_v·HEAD_COLS·HEAD_COLS`.
#[must_use]
pub fn state_elems(lanes: usize) -> Option<usize> {
    lanes
        .checked_mul(VALUE_HEADS)
        .and_then(|n| n.checked_mul(HEAD_COLS))
        .and_then(|n| n.checked_mul(HEAD_COLS))
}

/// f32s of the conv ring: `RING_ROWS` rows of [`channels`].
#[must_use]
pub fn ring_elems(n_k: usize) -> Option<usize> {
    channels(n_k).and_then(|ch| RING_ROWS.checked_mul(ch))
}

/// Index of one state value: lane, value head, column, key.
#[must_use]
pub fn state_index(lane: usize, h: usize, col: usize, key: usize) -> usize {
    ((lane * VALUE_HEADS + h) * HEAD_COLS + col) * HEAD_COLS + key
}

/// One gated delta step.
pub struct DeltaParams<'w> {
    /// The conv input, one value per channel.
    pub x: Window<'w, f32, MetalDevice>,
    /// The conv weights, tap innermost: `4·channels`.
    pub conv_w: Window<'w, f32, MetalDevice>,
    /// `β`'s pre-sigmoid input, one value per value head.
    pub b_raw: Window<'w, f32, MetalDevice>,
    /// The decay's pre-softplus input, one value per value head.
    pub a_raw: Window<'w, f32, MetalDevice>,
    /// The decay bias, one value per value head.
    pub dt_bias: Window<'w, f32, MetalDevice>,
    /// The decay slope, one value per value head.
    pub ssm_a: Window<'w, f32, MetalDevice>,
    /// The gate activation's input, one value per value-head column.
    pub z: Window<'w, f32, MetalDevice>,
    /// The norm gain, one value per column. The norm scale is not applied.
    pub norm_w: Window<'w, f32, MetalDevice>,
    /// Lane words. The kernel reads the word at `lane_at`.
    pub lane: Window<'w, u32, MetalDevice>,
    /// The recurrent state, read and written in place.
    pub state: WindowMut<'w, f32, MetalDevice>,
    /// The conv ring, read and written in place.
    pub ring: WindowMut<'w, f32, MetalDevice>,
    /// The partial sums of `o²`: [`VALUE_HEADS`] rows of [`PARTIALS_PER_HEAD`].
    pub partials: WindowMut<'w, f32, MetalDevice>,
    /// The gated numerator, one value per value-head column.
    pub g: WindowMut<'w, f32, MetalDevice>,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// Value heads. This layer has [`VALUE_HEADS`].
    pub n_v: usize,
    /// Columns of a head. This layer has [`HEAD_COLS`].
    pub head: usize,
    /// Conv taps. This layer has [`CONV_TAPS`].
    pub taps: usize,
    /// Key heads. `n_v` is a multiple of this, and it is not zero.
    pub n_k: usize,
    /// How a value head finds its key head.
    pub map: HeadMap,
    /// Lanes of the state. Zero is refused.
    pub lanes: usize,
    /// The lane word this step reads.
    pub lane_at: usize,
    /// The layer a raise is recorded in.
    pub layer: u32,
    /// The position window. A step reads one position; more than
    /// [`PASS_ROWS`] is past the ring.
    pub pos: &'w [u32],
    /// The L2 norm's epsilon.
    pub eps: f32,
}

fn refuse(detail: impl Into<String>) -> DeviceError {
    DeviceError::Range {
        what: WHAT,
        detail: detail.into(),
    }
}

fn pack(words: [u32; 8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, word) in words.into_iter().enumerate() {
        out[i * 4..][..4].copy_from_slice(&word.to_ne_bytes());
    }
    out
}

/// The shape, refused by name before any window is read. The position window
/// is checked here too: a window past the ring is a different refusal from a
/// window of the wrong length.
fn check_shape(p: &DeltaParams<'_>) -> Result<(), DeviceError> {
    if p.n_v != VALUE_HEADS {
        return Err(refuse(format!(
            "n_v {} is not this layer's n_v {VALUE_HEADS}",
            p.n_v
        )));
    }
    if p.head != HEAD_COLS {
        return Err(refuse(format!(
            "head {} is not this layer's head {HEAD_COLS}",
            p.head
        )));
    }
    if p.taps != CONV_TAPS {
        return Err(refuse(format!(
            "conv taps {} are not this layer's taps {CONV_TAPS}",
            p.taps
        )));
    }
    if p.n_k == 0 || !VALUE_HEADS.is_multiple_of(p.n_k) {
        return Err(refuse(format!(
            "n_k {} does not divide this layer's n_v {VALUE_HEADS}",
            p.n_k
        )));
    }
    if p.lanes == 0 {
        return Err(refuse("lanes of 0"));
    }
    if p.pos.len() > PASS_ROWS {
        return Err(refuse(format!(
            "a position window of {} is past the ring",
            p.pos.len()
        )));
    }
    if p.pos.len() != 1 {
        return Err(refuse(format!(
            "the position window holds {}, the step takes one position",
            p.pos.len()
        )));
    }
    Ok(())
}

fn check_windows(p: &DeltaParams<'_>) -> Result<(), DeviceError> {
    let ch = channels(p.n_k).ok_or_else(|| refuse(format!("n_k {} channels do not fit", p.n_k)))?;
    let conv = ch
        .checked_mul(CONV_TAPS)
        .ok_or_else(|| refuse(format!("conv weights of {ch} channels do not fit")))?;
    let z_len = VALUE_HEADS
        .checked_mul(HEAD_COLS)
        .ok_or_else(|| refuse("the gated numerator does not fit"))?;
    let state = state_elems(p.lanes)
        .ok_or_else(|| refuse(format!("the state of {} lanes does not fit", p.lanes)))?;
    let ring = ring_elems(p.n_k)
        .ok_or_else(|| refuse(format!("the ring of n_k {} does not fit", p.n_k)))?;
    check_window_len(WHAT, "x", p.x.len(), ch)?;
    check_window_len(WHAT, "conv weight", p.conv_w.len(), conv)?;
    check_window_len(WHAT, "b", p.b_raw.len(), VALUE_HEADS)?;
    check_window_len(WHAT, "a", p.a_raw.len(), VALUE_HEADS)?;
    check_window_len(WHAT, "dt bias", p.dt_bias.len(), VALUE_HEADS)?;
    check_window_len(WHAT, "ssm a", p.ssm_a.len(), VALUE_HEADS)?;
    check_window_len(WHAT, "z", p.z.len(), z_len)?;
    check_window_len(WHAT, "norm weight", p.norm_w.len(), HEAD_COLS)?;
    let lane_need = p
        .lane_at
        .checked_add(1)
        .ok_or_else(|| refuse(format!("the lane index {} does not fit", p.lane_at)))?;
    if p.lane.len().get() < lane_need {
        return Err(refuse(format!(
            "the lane window holds {}, the lane index needs {lane_need}",
            p.lane.len().get()
        )));
    }
    check_word(WHAT, "the state length", state)?;
    check_word(WHAT, "the ring length", ring)?;
    check_window_len(WHAT, "state", p.state.len(), state)?;
    check_window_len(WHAT, "ring", p.ring.len(), ring)?;
    check_window_len(WHAT, "partials", p.partials.len(), GROUPS)?;
    check_window_len(WHAT, "g", p.g.len(), z_len)?;
    Ok(())
}

impl Kernels {
    /// Encode one gated delta step on `encoder`. The shape is checked before
    /// any window length, and a refused launch encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn delta(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: DeltaParams<'_>,
    ) -> Result<(), DeviceError> {
        check_shape(&p)?;
        check_windows(&p)?;
        let n_k = word(WHAT, "n_k", p.n_k)?;
        let lanes = word(WHAT, "the lane count", p.lanes)?;
        let lane_at = word(WHAT, "the lane index", p.lane_at)?;
        let args = pack([
            n_k,
            p.map.word(),
            lanes,
            lane_at,
            p.layer,
            p.pos[0],
            p.eps.to_bits(),
            0,
        ]);
        encoder.set_pipeline(&self.delta);
        encoder.set_binding(0, Binding::of(&p.x)?)?;
        encoder.set_binding(1, Binding::of(&p.conv_w)?)?;
        encoder.set_binding(2, Binding::of(&p.b_raw)?)?;
        encoder.set_binding(3, Binding::of(&p.a_raw)?)?;
        encoder.set_binding(4, Binding::of(&p.dt_bias)?)?;
        encoder.set_binding(5, Binding::of(&p.ssm_a)?)?;
        encoder.set_binding(6, Binding::of(&p.z)?)?;
        encoder.set_binding(7, Binding::of(&p.norm_w)?)?;
        encoder.set_binding(8, Binding::of(&p.lane)?)?;
        encoder.set_binding(9, Binding::of_mut(&p.state)?)?;
        encoder.set_binding(10, Binding::of_mut(&p.ring)?)?;
        encoder.set_binding(11, Binding::of_mut(&p.partials)?)?;
        encoder.set_binding(12, Binding::of_mut(&p.g)?)?;
        encoder.set_binding(13, Binding::bytes(&args)?)?;
        encoder.set_binding(14, p.faults.binding())?;
        encoder.dispatch(GROUPS * GROUP_THREADS, GROUP_THREADS)
    }
}
