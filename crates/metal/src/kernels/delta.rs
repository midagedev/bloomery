//! The gated delta step: one decode token's conv, decay and delta update.
//!
//! The kernel writes the gated numerator and the partial sums of `o²`. The
//! head-normed mat-vec multiplies the norm scale on load; this launch does
//! not. The grid is the instance's (`DeltaGeom::groups` threadgroups, one per
//! value-head column block, `DeltaGeom::group_threads` threads in the group).
//!
//! [`Kernels::delta_rows`] is the same step over a chunk of rows: the rows
//! are the bits of the stepped launches. A row has a state-free half (the
//! conv, the q/k norm, β, the decay, the gate's SiLU) and a recurrence half
//! (the delta update of the state). The launch runs the state-free half of
//! every row in parallel into a prepared row (the caller's scratch window), the
//! recurrence over the rows in order, and then the numerators and partial sums
//! in parallel. The ring commit follows the state-free half, which is the last
//! reader of the pre-chunk slots. A launch may also leave copies after rows
//! inside its chunk ([`DeltaSnaps`]): each the state lane and the ring a chunk
//! ending at that row leaves, the chunk's own outputs unmoved.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;

use crate::geom::{DeltaGeom, HOST_GEOMETRY};
use crate::library::{Library, Pipeline};

use super::delta_geom::SNAPS;
use super::geometry::VALUE_HEADS;
use super::rows::check_strided;
use super::{Kernels, check_window_len, pack, word};

/// `1 / sqrt(head_cols)` rounded to f32 (`DeltaGeom::q_scale`). The kernel
/// multiplies the query by this value; an oracle promotes these bits, rather
/// than a fresh reciprocal. A shim: the build's host instance's. Callers left:
/// the kernel gate's oracles.
pub const Q_SCALE: f32 = f32::from_bits(0x3db5_04f3);

const WHAT: &str = "delta step";

/// Where the parts of a prepared row sit, in floats from the row's start. The
/// scaled queries, then the scaled keys (`n_k` key heads of `head_cols`
/// values each), the value conv outputs and the gate's SiLU (`value_heads`
/// heads of `head_cols`), β and the decay (`value_heads` each), and one
/// status word for each recurrence threadgroup. Every part starts on a
/// 16-byte boundary: the recurrence reads the keys four at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrepLayout {
    stride: usize,
    k: usize,
    v: usize,
    gate: usize,
    decay: usize,
    beta: usize,
    bad: usize,
}

impl PrepLayout {
    fn of(g: &DeltaGeom, n_k: usize) -> Option<Self> {
        let k = n_k.checked_mul(g.head_cols)?;
        let v = k.checked_mul(2)?;
        let gate = v.checked_add(g.value_heads * g.head_cols)?;
        let decay = gate.checked_add(g.value_heads * g.head_cols)?;
        let beta = decay.checked_add(g.value_heads)?;
        let bad = beta.checked_add(g.value_heads)?;
        let stride = bad.checked_add(g.groups)?;
        Some(Self {
            stride,
            k,
            v,
            gate,
            decay,
            beta,
            bad,
        })
    }

    /// The seven words the kernel's `GdnPrepLayout` holds, in its order.
    fn words(self) -> Result<[u32; 7], DeviceError> {
        let w = |name: &str, n: usize| word(WHAT, name, n);
        Ok([
            w("the prepared row stride", self.stride)?,
            w("the prepared keys offset", self.k)?,
            w("the prepared values offset", self.v)?,
            w("the prepared gates offset", self.gate)?,
            w("the prepared decays offset", self.decay)?,
            w("the prepared betas offset", self.beta)?,
            w("the prepared status offset", self.bad)?,
        ])
    }
}

impl DeltaGeom {
    /// f32s of one prepared row for `n_k` key heads: the scratch window of
    /// [`DeltaRowsParams::prep`] holds that many a row.
    #[must_use]
    pub fn prepared_elems(&self, n_k: usize) -> Option<usize> {
        PrepLayout::of(self, n_k).map(|l| l.stride)
    }

    /// Conv channels: `(2·n_k + n_v)·head_cols`.
    #[must_use]
    pub fn channels(&self, n_k: usize) -> Option<usize> {
        n_k.checked_mul(2)
            .and_then(|k| k.checked_add(self.value_heads))
            .and_then(|h| h.checked_mul(self.head_cols))
    }

    /// f32s of the recurrent state: `lanes·n_v·head_cols·head_cols`.
    #[must_use]
    pub fn state_elems(&self, lanes: usize) -> Option<usize> {
        lanes
            .checked_mul(self.value_heads)
            .and_then(|n| n.checked_mul(self.head_cols))
            .and_then(|n| n.checked_mul(self.head_cols))
    }

    /// f32s of the conv ring: `ring_rows` rows of [`DeltaGeom::channels`].
    #[must_use]
    pub fn ring_elems(&self, n_k: usize) -> Option<usize> {
        self.channels(n_k)
            .and_then(|ch| self.ring_rows.checked_mul(ch))
    }

    /// Index of one state value: lane, value head, column, key.
    #[must_use]
    pub fn state_index(&self, lane: usize, h: usize, col: usize, key: usize) -> usize {
        ((lane * self.value_heads + h) * self.head_cols + col) * self.head_cols + key
    }
}

/// f32s of one prepared row for `n_k` key heads. A shim: the build's host
/// instance's (`DeltaGeom::prepared_elems`). Callers left: the arch, the
/// kernel gate and the probes.
#[must_use]
pub fn prepared_elems(n_k: usize) -> Option<usize> {
    HOST_GEOMETRY.delta.prepared_elems(n_k)
}

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
    /// The key head of value head `h` of a layer of `n_v` value heads.
    #[must_use]
    pub fn k_head_of(self, h: usize, n_v: usize, n_k: usize) -> usize {
        match self {
            HeadMap::Tiled => h % n_k,
            HeadMap::Grouped => h / (n_v / n_k),
        }
    }

    /// The key head of value head `h`. A shim: the build's host instance's
    /// value heads (`HeadMap::k_head_of`). Callers left: the kernel gate's
    /// oracles.
    #[must_use]
    pub fn k_head(self, h: usize, n_k: usize) -> usize {
        self.k_head_of(h, VALUE_HEADS, n_k)
    }

    fn word(self) -> u32 {
        match self {
            HeadMap::Tiled => 0,
            HeadMap::Grouped => 1,
        }
    }
}

/// Conv channels: `(2·n_k + n_v)·head_cols`. A shim: the build's host
/// instance's (`DeltaGeom::channels`). Callers left: the arch, the e2e and
/// kernel gates, their oracles and the probes.
#[must_use]
pub fn channels(n_k: usize) -> Option<usize> {
    HOST_GEOMETRY.delta.channels(n_k)
}

/// f32s of the recurrent state: `lanes·n_v·head_cols·head_cols`. A shim: the
/// build's host instance's (`DeltaGeom::state_elems`). Callers left: the
/// arch, the kernel gate, its oracles and the probes.
#[must_use]
pub fn state_elems(lanes: usize) -> Option<usize> {
    HOST_GEOMETRY.delta.state_elems(lanes)
}

/// f32s of the conv ring: `ring_rows` rows of [`channels`]. A shim: the
/// build's host instance's (`DeltaGeom::ring_elems`). Callers left: the
/// arch, the kernel gate, its oracles and the probes.
#[must_use]
pub fn ring_elems(n_k: usize) -> Option<usize> {
    HOST_GEOMETRY.delta.ring_elems(n_k)
}

/// Index of one state value: lane, value head, column, key. A shim: the
/// build's host instance's (`DeltaGeom::state_index`). Callers left: the
/// kernel gate and its oracles.
#[must_use]
pub fn state_index(lane: usize, h: usize, col: usize, key: usize) -> usize {
    HOST_GEOMETRY.delta.state_index(lane, h, col, key)
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
    /// Lane words. The kernel reads the words at `lane_at` and `lane_to`.
    pub lane: Window<'w, u32, MetalDevice>,
    /// The recurrent state: the lane the word at `lane_at` names is read, the
    /// lane the word at `lane_to` names is written.
    pub state: WindowMut<'w, f32, MetalDevice>,
    /// The conv ring, read and written in place.
    pub ring: WindowMut<'w, f32, MetalDevice>,
    /// The partial sums of `o²`: `DeltaGeom::partials_stride` values, a row of
    /// the column blocks of each value head.
    pub partials: WindowMut<'w, f32, MetalDevice>,
    /// The gated numerator, one value per value-head column.
    pub g: WindowMut<'w, f32, MetalDevice>,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// Value heads. This layer has the instance's (`DeltaGeom::value_heads`).
    pub n_v: usize,
    /// Columns of a head. This layer has the instance's
    /// (`DeltaGeom::head_cols`).
    pub head: usize,
    /// Conv taps. This layer has the instance's (`DeltaGeom::conv_taps`).
    pub taps: usize,
    /// Key heads. `n_v` is a multiple of this, and it is not zero.
    pub n_k: usize,
    /// How a value head finds its key head.
    pub map: HeadMap,
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
    /// The L2 norm's epsilon.
    pub eps: f32,
}

fn refuse(detail: impl Into<String>) -> DeviceError {
    super::range(WHAT, detail)
}

/// The shape the kernel takes, refused by name before any window is read.
fn check_geometry(
    g: &DeltaGeom,
    n_v: usize,
    head: usize,
    taps: usize,
    n_k: usize,
    lanes: usize,
) -> Result<(), DeviceError> {
    if n_v != g.value_heads {
        return Err(refuse(format!(
            "n_v {n_v} is not this layer's n_v {}",
            g.value_heads
        )));
    }
    if head != g.head_cols {
        return Err(refuse(format!(
            "head {head} is not this layer's head {}",
            g.head_cols
        )));
    }
    if taps != g.conv_taps {
        return Err(refuse(format!(
            "conv taps {taps} are not this layer's taps {}",
            g.conv_taps
        )));
    }
    if n_k == 0 || !g.value_heads.is_multiple_of(n_k) {
        return Err(refuse(format!(
            "n_k {n_k} does not divide this layer's n_v {}",
            g.value_heads
        )));
    }
    if lanes == 0 {
        return Err(refuse("lanes of 0"));
    }
    Ok(())
}

/// The shape, refused by name before any window is read. A position window
/// that is not one position is refused here too.
fn check_shape(g: &DeltaGeom, p: &DeltaParams<'_>) -> Result<(), DeviceError> {
    check_geometry(g, p.n_v, p.head, p.taps, p.n_k, p.lanes)?;
    if p.pos.len() != 1 {
        return Err(refuse(format!(
            "the position window holds {}, the step takes one position",
            p.pos.len()
        )));
    }
    Ok(())
}

/// The lane window holds the words the read and the write index name.
fn check_lane_words(held: usize, lane_at: usize, lane_to: usize) -> Result<(), DeviceError> {
    for (what, at) in [("read", lane_at), ("write", lane_to)] {
        let need = at
            .checked_add(1)
            .ok_or_else(|| refuse(format!("the {what} lane index {at} does not fit")))?;
        if held < need {
            return Err(refuse(format!(
                "the lane window holds {held}, the {what} lane index needs {need}"
            )));
        }
    }
    Ok(())
}

fn check_windows(g: &DeltaGeom, p: &DeltaParams<'_>) -> Result<(), DeviceError> {
    let ch = g
        .channels(p.n_k)
        .ok_or_else(|| refuse(format!("n_k {} channels do not fit", p.n_k)))?;
    let conv = ch
        .checked_mul(g.conv_taps)
        .ok_or_else(|| refuse(format!("conv weights of {ch} channels do not fit")))?;
    let z_len = g
        .value_heads
        .checked_mul(g.head_cols)
        .ok_or_else(|| refuse("the gated numerator does not fit"))?;
    let state = g
        .state_elems(p.lanes)
        .ok_or_else(|| refuse(format!("the state of {} lanes does not fit", p.lanes)))?;
    let ring = g
        .ring_elems(p.n_k)
        .ok_or_else(|| refuse(format!("the ring of n_k {} does not fit", p.n_k)))?;
    check_window_len(WHAT, "x", p.x.len(), ch)?;
    check_window_len(WHAT, "conv weight", p.conv_w.len(), conv)?;
    check_window_len(WHAT, "b", p.b_raw.len(), g.value_heads)?;
    check_window_len(WHAT, "a", p.a_raw.len(), g.value_heads)?;
    check_window_len(WHAT, "dt bias", p.dt_bias.len(), g.value_heads)?;
    check_window_len(WHAT, "ssm a", p.ssm_a.len(), g.value_heads)?;
    check_window_len(WHAT, "z", p.z.len(), z_len)?;
    check_window_len(WHAT, "norm weight", p.norm_w.len(), g.head_cols)?;
    check_lane_words(p.lane.len().get(), p.lane_at, p.lane_to)?;
    word(WHAT, "the state length", state)?;
    word(WHAT, "the ring length", ring)?;
    check_window_len(WHAT, "state", p.state.len(), state)?;
    check_window_len(WHAT, "ring", p.ring.len(), ring)?;
    check_window_len(WHAT, "partials", p.partials.len(), g.groups)?;
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
        let g = &self.geom.delta;
        check_shape(g, &p)?;
        check_windows(g, &p)?;
        let n_k = word(WHAT, "n_k", p.n_k)?;
        let lanes = word(WHAT, "the lane count", p.lanes)?;
        let lane_at = word(WHAT, "the read lane index", p.lane_at)?;
        let lane_to = word(WHAT, "the write lane index", p.lane_to)?;
        let args = [
            n_k.to_ne_bytes(),
            p.map.word().to_ne_bytes(),
            lanes.to_ne_bytes(),
            lane_at.to_ne_bytes(),
            lane_to.to_ne_bytes(),
            p.layer.to_ne_bytes(),
            p.pos[0].to_ne_bytes(),
            p.eps.to_ne_bytes(),
        ];
        encoder.set_pipeline(&self.delta);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&p.conv_w)?)?;
        slot.set(Binding::of(&p.b_raw)?)?;
        slot.set(Binding::of(&p.a_raw)?)?;
        slot.set(Binding::of(&p.dt_bias)?)?;
        slot.set(Binding::of(&p.ssm_a)?)?;
        slot.set(Binding::of(&p.z)?)?;
        slot.set(Binding::of(&p.norm_w)?)?;
        slot.set(Binding::of(&p.lane)?)?;
        slot.set(Binding::of_mut(&p.state)?)?;
        slot.set(Binding::of_mut(&p.ring)?)?;
        slot.set(Binding::of_mut(&p.partials)?)?;
        slot.set(Binding::of_mut(&p.g)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(g.groups, 1, g.group_threads)
    }
}

/// The strides, in elements, between the rows of a chunk's buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaRowStrides {
    /// The conv input: at least `DeltaGeom::channels`.
    pub x: usize,
    /// The gate input: at least `value_heads · head_cols`.
    pub z: usize,
    /// β's input: at least `value_heads`.
    pub b: usize,
    /// The decay's input: at least `value_heads`.
    pub a: usize,
    /// The gated numerator: at least `value_heads · head_cols`.
    pub g: usize,
    /// The partial sums of `o²`: at least `DeltaGeom::partials_stride`
    /// (`value_heads` rows of the column blocks).
    pub partials: usize,
}

impl DeltaRowStrides {
    /// Rows laid end to end, each exactly as wide as the step's own window.
    #[must_use]
    pub fn of(g: &DeltaGeom, n_k: usize) -> Option<Self> {
        Some(Self {
            x: g.channels(n_k)?,
            z: g.value_heads * g.head_cols,
            b: g.value_heads,
            a: g.value_heads,
            g: g.value_heads * g.head_cols,
            partials: g.partials_stride,
        })
    }

    /// Rows laid end to end. A shim: the build's host instance's
    /// (`DeltaRowStrides::of`). Callers left: the probes.
    #[must_use]
    pub fn dense(n_k: usize) -> Option<Self> {
        Self::of(&HOST_GEOMETRY.delta, n_k)
    }
}

/// The copies a rows launch leaves after rows inside its chunk: copy `k` is
/// the state lane and the ring a chunk ending after row `after[k]` leaves,
/// bit for bit, written into the `k`-th lane and ring of these windows.
pub struct DeltaSnaps<'w> {
    /// The rows the copies follow, ascending, each below the chunk's rows;
    /// one to [`SNAPS`] of them.
    pub after: &'w [u32],
    /// One lane of the state a copy, laid out as a lane of the state:
    /// `after.len()` times `DeltaGeom::state_elems(1)` values.
    pub lane: WindowMut<'w, f32, MetalDevice>,
    /// One ring a copy, laid out as the ring: `after.len()` times
    /// `DeltaGeom::ring_elems` values.
    pub ring: WindowMut<'w, f32, MetalDevice>,
}

/// One gated delta step over a chunk of rows, rows in order in one launch.
/// The state is read from the lane the word at `lane_at` names before the
/// first row and written to the lane the word at `lane_to` names after the
/// last; the words may name one lane. The ring keeps what `rows` stepped
/// launches would leave.
pub struct DeltaRowsParams<'w> {
    /// The conv input: row `r` starts `strides.x` elements after row `r - 1`.
    pub x: Window<'w, f32, MetalDevice>,
    /// The conv weights, tap innermost: `4·channels`.
    pub conv_w: Window<'w, f32, MetalDevice>,
    /// `β`'s pre-sigmoid input, `value_heads` values a row.
    pub b_raw: Window<'w, f32, MetalDevice>,
    /// The decay's pre-softplus input, `value_heads` values a row.
    pub a_raw: Window<'w, f32, MetalDevice>,
    /// The decay bias, one value per value head.
    pub dt_bias: Window<'w, f32, MetalDevice>,
    /// The decay slope, one value per value head.
    pub ssm_a: Window<'w, f32, MetalDevice>,
    /// The gate activation's input, one value per value-head column a row.
    pub z: Window<'w, f32, MetalDevice>,
    /// The norm gain, one value per column. The norm scale is not applied.
    pub norm_w: Window<'w, f32, MetalDevice>,
    /// Lane words. The kernel reads the words at `lane_at` and `lane_to`.
    pub lane: Window<'w, u32, MetalDevice>,
    /// The recurrent state, every lane.
    pub state: WindowMut<'w, f32, MetalDevice>,
    /// The conv ring, every slot: the pre-chunk slots are read, then the last
    /// `min(rows, ring_rows)` positions' slots are written.
    pub ring: WindowMut<'w, f32, MetalDevice>,
    /// The partial sums of `o²`, `DeltaGeom::partials_stride` values a row.
    pub partials: WindowMut<'w, f32, MetalDevice>,
    /// The gated numerator, one value per value-head column a row. The
    /// recurrence leaves the column outputs here, and the launch's last
    /// dispatch replaces them with the numerators.
    pub g: WindowMut<'w, f32, MetalDevice>,
    /// The prepared rows' scratch, `DeltaGeom::prepared_elems` values a row, on a
    /// 16-byte boundary. The launch writes and reads it; the caller reads
    /// nothing from it.
    pub prep: WindowMut<'w, f32, MetalDevice>,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// Value heads. This layer has the instance's (`DeltaGeom::value_heads`).
    pub n_v: usize,
    /// Columns of a head. This layer has the instance's
    /// (`DeltaGeom::head_cols`).
    pub head: usize,
    /// Conv taps. This layer has the instance's (`DeltaGeom::conv_taps`).
    pub taps: usize,
    /// Key heads. `n_v` is a multiple of this, and it is not zero.
    pub n_k: usize,
    /// How a value head finds its key head.
    pub map: HeadMap,
    /// Lanes of the state. Zero is refused.
    pub lanes: usize,
    /// The word of the lane the chunk reads.
    pub lane_at: usize,
    /// The word of the lane the chunk writes.
    pub lane_to: usize,
    /// The layer a raise is recorded in.
    pub layer: u32,
    /// Rows in the chunk. Zero is refused.
    pub rows: usize,
    /// The position of row 0. Row `r` is position `pos + r`.
    pub pos: u32,
    /// The strides between rows of the windows above.
    pub strides: DeltaRowStrides,
    /// The L2 norm's epsilon.
    pub eps: f32,
    /// The copies after rows inside the chunk; none takes none.
    pub snaps: Option<DeltaSnaps<'w>>,
}

/// The pipelines of the rows launch: the state-free half, the recurrence, the
/// numerators and partials, and the ring commit; and the recurrence and the
/// ring's copy of a launch that leaves copies.
pub(crate) struct RowsPipelines {
    prep: Pipeline,
    rows: Pipeline,
    post: Pipeline,
    ring: Pipeline,
    rows_snap: Pipeline,
    ring_snap: Pipeline,
}

impl RowsPipelines {
    pub(crate) fn new(library: &Library) -> Result<Self, DeviceError> {
        Ok(Self {
            prep: library.pipeline("gdn_prep_rows")?,
            rows: library.pipeline("gdn_rows")?,
            post: library.pipeline("gdn_post_rows")?,
            ring: library.pipeline("gdn_ring_rows")?,
            rows_snap: library.pipeline("gdn_rows_snap")?,
            ring_snap: library.pipeline("gdn_ring_snap")?,
        })
    }
}

/// Threads in the ring commit's threadgroup.
const RING_THREADS: usize = 128;

fn check_rows(g: &DeltaGeom, p: &DeltaRowsParams<'_>) -> Result<(), DeviceError> {
    check_geometry(g, p.n_v, p.head, p.taps, p.n_k, p.lanes)?;
    if p.rows == 0 {
        return Err(refuse("rows of 0"));
    }
    let rows = word(WHAT, "rows", p.rows)?;
    if p.pos.checked_add(rows).is_none() {
        return Err(refuse(format!(
            "positions {}..+{rows} pass the 32-bit position word",
            p.pos
        )));
    }
    let ch = g
        .channels(p.n_k)
        .ok_or_else(|| refuse(format!("n_k {} channels do not fit", p.n_k)))?;
    let conv = ch
        .checked_mul(g.conv_taps)
        .ok_or_else(|| refuse(format!("conv weights of {ch} channels do not fit")))?;
    let z_len = g.value_heads * g.head_cols;
    let state = g
        .state_elems(p.lanes)
        .ok_or_else(|| refuse(format!("the state of {} lanes does not fit", p.lanes)))?;
    let ring = g
        .ring_elems(p.n_k)
        .ok_or_else(|| refuse(format!("the ring of n_k {} does not fit", p.n_k)))?;
    let st = p.strides;
    check_strided(WHAT, "x", p.x.len(), p.rows, st.x, ch)?;
    check_strided(WHAT, "z", p.z.len(), p.rows, st.z, z_len)?;
    check_strided(WHAT, "b", p.b_raw.len(), p.rows, st.b, g.value_heads)?;
    check_strided(WHAT, "a", p.a_raw.len(), p.rows, st.a, g.value_heads)?;
    check_strided(WHAT, "g", p.g.len(), p.rows, st.g, z_len)?;
    check_strided(
        WHAT,
        "partials",
        p.partials.len(),
        p.rows,
        st.partials,
        g.groups,
    )?;
    check_window_len(WHAT, "conv weight", p.conv_w.len(), conv)?;
    check_window_len(WHAT, "dt bias", p.dt_bias.len(), g.value_heads)?;
    check_window_len(WHAT, "ssm a", p.ssm_a.len(), g.value_heads)?;
    check_window_len(WHAT, "norm weight", p.norm_w.len(), g.head_cols)?;
    check_lane_words(p.lane.len().get(), p.lane_at, p.lane_to)?;
    word(WHAT, "the state length", state)?;
    word(WHAT, "the ring length", ring)?;
    check_window_len(WHAT, "state", p.state.len(), state)?;
    check_window_len(WHAT, "ring", p.ring.len(), ring)?;
    let layout = PrepLayout::of(g, p.n_k)
        .ok_or_else(|| refuse(format!("the prepared row of n_k {} does not fit", p.n_k)))?;
    check_strided(
        WHAT,
        "prepared rows",
        p.prep.len(),
        p.rows,
        layout.stride,
        layout.stride,
    )?;
    let range = p
        .prep
        .byte_range()
        .map_err(|e| refuse(format!("the prepared rows window: {e}")))?;
    if range.start.get() % 16 != 0 {
        return Err(refuse(format!(
            "the prepared rows start {} bytes in, not on the 16-byte word the recurrence reads",
            range.start.get()
        )));
    }
    if let Some(snaps) = &p.snaps {
        check_snaps(g, snaps, p.rows, ring)?;
    }
    Ok(())
}

/// The copies of a chunk of `rows` rows whose ring is `ring` values: one to
/// [`SNAPS`] rows, ascending, inside the chunk, and a lane and a ring a copy.
fn check_snaps(
    g: &DeltaGeom,
    s: &DeltaSnaps<'_>,
    rows: usize,
    ring: usize,
) -> Result<(), DeviceError> {
    let n = s.after.len();
    if n == 0 || n > SNAPS {
        return Err(refuse(format!(
            "copies after {n} rows; a launch takes 1 to {SNAPS}"
        )));
    }
    if !s.after.windows(2).all(|w| w[0] < w[1]) {
        return Err(refuse(format!(
            "copies after rows {:?}, not ascending",
            s.after
        )));
    }
    if let Some(&r) = s.after.iter().find(|&&r| r as usize >= rows) {
        return Err(refuse(format!(
            "a copy after row {r} of a chunk of {rows} rows"
        )));
    }
    let lane = g
        .state_elems(1)
        .ok_or_else(|| refuse("a lane of the state does not fit"))?;
    let lanes = n
        .checked_mul(lane)
        .ok_or_else(|| refuse(format!("{n} copies of a lane do not fit")))?;
    let rings = n
        .checked_mul(ring)
        .ok_or_else(|| refuse(format!("{n} copies of the ring do not fit")))?;
    word(WHAT, "a copy's lane length", lane)?;
    word(WHAT, "a copy's ring length", ring)?;
    check_window_len(WHAT, "the copies' lanes", s.lane.len(), lanes)?;
    check_window_len(WHAT, "the copies' rings", s.ring.len(), rings)
}

/// The copies' argument block: their count, the values of a copy's lane and
/// of its ring, and the row each follows (the words past the count are 0).
fn snap_args(
    g: &DeltaGeom,
    s: &DeltaSnaps<'_>,
    ring: usize,
) -> Result<[[u8; 4]; 3 + SNAPS], DeviceError> {
    let lane = g
        .state_elems(1)
        .ok_or_else(|| refuse("a lane of the state does not fit"))?;
    let mut words = [0u32; 3 + SNAPS];
    words[0] = word(WHAT, "the copy count", s.after.len())?;
    words[1] = word(WHAT, "a copy's lane length", lane)?;
    words[2] = word(WHAT, "a copy's ring length", ring)?;
    words[3..3 + s.after.len()].copy_from_slice(s.after);
    Ok(words.map(u32::to_ne_bytes))
}

impl Kernels {
    /// Encode the step over `p.rows` rows: the state-free half, a memory
    /// barrier, the ring commit with the recurrence, a barrier, and the
    /// numerators and partial sums. The shape and every window are checked
    /// before any dispatch, and a refused launch encodes nothing. The ring
    /// commit and the recurrence read and write disjoint windows, so they may
    /// overlap. With copies, the ring's copies are encoded beside the
    /// state-free half (both only read the ring, before the commit's barrier)
    /// and the recurrence writes the lanes' copies as it passes their rows.
    #[must_use = "a refused launch encoded nothing"]
    pub fn delta_rows(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: DeltaRowsParams<'_>,
    ) -> Result<(), DeviceError> {
        let g = &self.geom.delta;
        check_rows(g, &p)?;
        let ch = g.channels(p.n_k).expect("check_rows fitted the channels");
        let layout = PrepLayout::of(g, p.n_k).expect("check_rows fitted the prepared row");
        let st = p.strides;
        let step = [
            word(WHAT, "n_k", p.n_k)?,
            p.map.word(),
            word(WHAT, "the lane count", p.lanes)?,
            word(WHAT, "the read lane index", p.lane_at)?,
            word(WHAT, "the write lane index", p.lane_to)?,
            p.layer,
            p.pos,
            p.eps.to_bits(),
        ];
        let rows = [
            word(WHAT, "rows", p.rows)?,
            word(WHAT, "the x stride", st.x)?,
            word(WHAT, "the z stride", st.z)?,
            word(WHAT, "the b stride", st.b)?,
            word(WHAT, "the a stride", st.a)?,
            word(WHAT, "the g stride", st.g)?,
            word(WHAT, "the partials stride", st.partials)?,
        ];
        let args = step
            .into_iter()
            .chain(rows)
            .chain(layout.words()?)
            .map(u32::to_ne_bytes)
            .collect::<Vec<_>>();
        let args: [[u8; 4]; 22] = args.try_into().expect("twenty-two words");
        let ring_len = g.ring_elems(p.n_k).expect("check_rows fitted the ring");
        let snap = match &p.snaps {
            Some(s) => Some((s, snap_args(g, s, ring_len)?)),
            None => None,
        };
        let ring_args = [
            word(WHAT, "channels", ch)?,
            word(WHAT, "rows", p.rows)?,
            word(WHAT, "the x stride", st.x)?,
            p.pos,
        ]
        .map(u32::to_ne_bytes);

        encoder.set_pipeline(&self.delta_rows.prep);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&p.conv_w)?)?;
        slot.set(Binding::of(&p.b_raw)?)?;
        slot.set(Binding::of(&p.a_raw)?)?;
        slot.set(Binding::of(&p.dt_bias)?)?;
        slot.set(Binding::of(&p.ssm_a)?)?;
        slot.set(Binding::of(&p.z)?)?;
        slot.set(Binding::of_mut(&p.ring)?)?;
        slot.set(Binding::of_mut(&p.prep)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.dispatch_groups(p.n_k, p.rows, g.group_threads)?;

        if let Some((s, words)) = &snap {
            encoder.set_pipeline(&self.delta_rows.ring_snap);
            let mut slot = Bindings::new(encoder);
            slot.set(Binding::of(&p.x)?)?;
            slot.set(Binding::of_mut(&p.ring)?)?;
            slot.set(Binding::of_mut(&s.ring)?)?;
            slot.set(Binding::bytes(pack(&ring_args))?)?;
            slot.set(Binding::bytes(pack(words))?)?;
            slot.dispatch_groups(
                ch.div_ceil(RING_THREADS),
                s.after.len() * g.ring_rows,
                RING_THREADS,
            )?;
        }

        encoder.memory_barrier();
        encoder.set_pipeline(&self.delta_rows.ring);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of_mut(&p.ring)?)?;
        slot.set(Binding::bytes(pack(&ring_args))?)?;
        slot.dispatch_groups(
            ch.div_ceil(RING_THREADS),
            p.rows.min(g.ring_rows),
            RING_THREADS,
        )?;

        let rows = match &snap {
            Some(_) => &self.delta_rows.rows_snap,
            None => &self.delta_rows.rows,
        };
        encoder.set_pipeline(rows);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of_mut(&p.prep)?)?;
        slot.set(Binding::of(&p.lane)?)?;
        slot.set(Binding::of_mut(&p.state)?)?;
        slot.set(Binding::of_mut(&p.g)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        if let Some((s, words)) = &snap {
            slot.set(Binding::of_mut(&s.lane)?)?;
            slot.set(Binding::bytes(pack(words))?)?;
        }
        slot.dispatch_groups(g.groups, 1, g.group_threads)?;

        encoder.memory_barrier();
        encoder.set_pipeline(&self.delta_rows.post);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of_mut(&p.prep)?)?;
        slot.set(Binding::of(&p.norm_w)?)?;
        slot.set(Binding::of_mut(&p.partials)?)?;
        slot.set(Binding::of_mut(&p.g)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.dispatch_groups(g.value_heads, p.rows, g.group_threads)
    }
}
