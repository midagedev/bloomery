//! The expert stream: the one owner of a prompt unit's transient expert
//! copies, which any MoE family calls with its counts and its card map and
//! no stream code of its own.
//!
//! **The split.** Per layer of a prompt unit, after the residency machine's
//! pick has admitted its share into the pool ([`super::swap`]'s call mode),
//! the host experts the unit still routes split in two: the ones the rule
//! sends over the copy lane to the card for this unit alone (the stream),
//! and the ones the host union keeps. The rule is [`runtime::xsplit`]'s,
//! over the lane's rate the load prices from where the lane's sources sit
//! (`lane_rate`) and the costs the model carries for its expert shape
//! ([`Costs`]): an expert streams when its columns cost the union more than
//! its copy and the card's columns, ranked count descending, id ascending.
//! Two caps follow the rule, in that ranking: a half of the ring
//! ([`XStream::half_slots`]), and the balance — the layer's copies (the
//! pick's admits and the stream) with the card's columns stay under the
//! union the host keeps, since the two run side by side and the layer waits
//! for the longer. A unit narrower than the rule's least width
//! ([`runtime::xsplit::m_min`]) streams nothing.
//!
//! **The ring.** Card memory beside the stacks, sized once at the load from
//! what the card holds free past `keep_free` ([`XCfg::keep_free`]), in two
//! halves of [`XStream::half_slots`] slots; per part (gate, up and down, or
//! each one's three planes: an NVFP4 planes expert, [`MAX_PARTS`]) a slot
//! holds the widest layer's part. A layer's stream lands in one half, its
//! experts packed at their layer's part bytes from the half's first slot,
//! so a grouped GEMM reads the half as a stack of `half_slots` experts. The
//! halves alternate per streamed layer: a layer's copies wait on the copy
//! stream for the read event of the layer two streams back
//! ([`XStream::read`]), the last reader of that half, so a stream into a
//! half whose last stream is still open (read by fewer units than its group)
//! is refused by name: its copies would land under the card routes still to
//! read it.
//!
//! **The maps.** Per half two rows of the router's experts, in mapped host
//! memory the card reads in place: the ring row (a streamed expert's slot
//! in the half, [`HOST`] for every other) and the union row (the stage map's
//! card places, `n_card + i` for the `i`-th streamed expert, [`HOST`] for
//! the rest), which the card sum reads with `n_card + n` card places. A
//! half's rows are written when its layer streams, after the host has seen
//! the download of a layer past the half's last reader.
//!
//! **The group.** A walk whose units share one pick a layer holds one stream
//! for them all ([`XStream::share`]): the layer's exclusion set stands for
//! every unit's serve, and the stream closes, and its half is released, at the
//! last unit's read ([`XStream::read`]), so a call's end refuses a layer read
//! by fewer units than its group.
//!
//! **The lane.** Fill threads copy each streamed expert's source bytes into
//! a pinned staging ring ([`XCfg::staging_slots`] slots); the copy stream
//! waits for each slot's staged word, copies the parts into the ring and
//! raises the slot's drained word, which the fill thread that takes the slot
//! next waits for — the residency machine's word protocol: a fill thread
//! waits on nothing the card runs after the copy that waits for it, so one
//! hardware queue (WDDM, `CUDA_DEVICE_MAX_CONNECTIONS=1`) only serializes
//! the streams. A layer's copies record one landed event a batch
//! ([`LandBatch`], the batching this module owns), which the layer's card
//! route waits before the GEMM batch over that batch's slots — the copies
//! and the GEMMs of a layer's shadow pipeline instead of running one after
//! the other. When the pinned staging cannot be allocated the lane copies
//! from the source's pageable bytes on the copy stream, the engine thread
//! enqueueing each copy.
//!
//! **No wait without a bound.** Every host wait (a fill thread for a slot's
//! drained word, the engine thread for the staging backlog, the drop for the
//! copy stream) ends by [`XCfg::deadline`] with a named error. A staging
//! failure publishes its word all the same, so the copy stream never hangs
//! on it, and the next layer or the call's end refuses by name.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer, sys};
use runtime::sched::BatchOrder;
use runtime::xsplit::{self, Constants, Split};

use super::batch::BatchKey;
use super::slots::HOST;
use super::swap::{SwapSource, Transform};
use super::{Drain, poll_drained};
use crate::Gpu;
use crate::GpuError;
use crate::graph::{MappedHost, cu, mem_batch, op_wait_geq, op_write};
use crate::tensor::DeviceTensor;

/// The stream lever: `off`, `admit` (the residency machine's pick alone) or
/// `split` (the pick, then the stream). The values a binary takes are the
/// lever registry's, read at `main` (`bloomery_levers::Levers::xstream`).
pub use bloomery_levers::XSTREAM;

/// What [`XSTREAM`] asks of a prompt call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XMode {
    /// The call moves nothing: the decode steps' bits.
    Off,
    /// The residency machine's pick moves the pool toward the unit's
    /// hottest host experts; nothing streams.
    Admit,
    /// The pick, then the stream of the host experts the rule sends to the
    /// card.
    Split,
}

impl XMode {
    /// `v` as a value of [`XSTREAM`]: `off`, `admit` or `split`; anything
    /// else is refused by name.
    pub fn parse(v: &str) -> Result<XMode, GpuError> {
        match v {
            "off" => Ok(XMode::Off),
            "admit" => Ok(XMode::Admit),
            "split" => Ok(XMode::Split),
            _ => Err(GpuError::shape(
                "XMode::parse",
                format!("{XSTREAM}={v:?}: it takes off, admit or split"),
            )),
        }
    }

    /// The lever's word for this mode.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            XMode::Off => "off",
            XMode::Admit => "admit",
            XMode::Split => "split",
        }
    }
}

/// The rule's costs a model carries for its expert shape beside the lane's
/// priced rate (`lane_rate`): the host union's cost of one expert per column
/// and its floor, and the card's cost of one streamed expert, fixed and per
/// column, each in µs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Costs {
    pub host_us_per_col: f64,
    pub host_us_fixed: f64,
    pub card_us_fixed: f64,
    pub card_us_per_col: f64,
}

/// The shape an [`XStream`] is built for.
#[derive(Clone, Debug)]
pub struct XCfg {
    pub costs: Costs,
    /// The router's choice set and its picks per column.
    pub experts: usize,
    pub top_k: usize,
    /// The most host experts a layer holds: no half needs more slots.
    pub max_half: usize,
    /// The card bytes the ring leaves free when it sizes itself.
    pub keep_free: u64,
    /// The most card bytes past `keep_free` the ring may take: `None` when
    /// the card's free bytes alone bound it.
    pub room_cap: Option<u64>,
    /// The fewest slots a half may have; a card with less room is refused
    /// by name.
    pub min_half: usize,
    /// Fill threads and pinned staging slots, the most: the stream pins at
    /// most a half of them ([`XStream::new`]).
    pub fill_threads: usize,
    pub staging_slots: usize,
    /// The bound on every host wait.
    pub deadline: Duration,
    /// Host experts the load's lane probe streams, `(layer, id)` each.
    pub probe: Vec<(usize, u32)>,
}

impl XCfg {
    /// The cfg of one family's load: the rule's costs for its expert shape,
    /// the router's experts and picks per column, and the bound on every
    /// host wait — the ring's own sizing (`max_half`, `probe`) the tier sets
    /// from the map ([`super::HostTier::xstream_start`]), the constants this
    /// module owns beside them, and `room_cap` capping the card bytes past
    /// `keep_free` the ring may take (`None`: the free bytes alone bound
    /// it).
    pub const fn new(
        costs: Costs,
        experts: usize,
        top_k: usize,
        deadline: Duration,
        room_cap: Option<u64>,
    ) -> XCfg {
        XCfg {
            costs,
            experts,
            top_k,
            max_half: 0,
            keep_free: KEEP_FREE,
            room_cap,
            min_half: MIN_HALF,
            fill_threads: FILL_THREADS,
            staging_slots: STAGING_SLOTS,
            deadline,
            probe: Vec::new(),
        }
    }
}

/// A body whose prompt calls stream experts: the two calls the shared
/// `BLOOMERY_XSTREAM` rule drives any family's load through — set what the
/// next prompt calls move (starting the stream under `split`), and the load
/// line's `xstream=` word. The body's own glue (its placement's guards, the
/// costs row of its routed kind, its card's arming) stays in its
/// `set_xstream`; the rule and the word's shape are shared.
pub trait XStreamBody {
    /// What the next prompt calls move (`BLOOMERY_XSTREAM`): `off` until
    /// set here, so one load can run every arm. `Split` starts the expert
    /// stream at its first setting and arms the walk's card route for its
    /// ring; the stream then stays for the load's life, idle under `Admit`
    /// and `Off`. Refused by name: `Admit` or `Split` on a load that runs
    /// no residency machine, `Split` on a load the ring cannot serve, and
    /// any change while a call streams.
    fn set_xstream(&mut self, gpu: &Gpu, mode: XMode) -> Result<(), GpuError>;

    /// The load line's `xstream=` word: the mode, and under a started
    /// stream its ring's slots a half, its pinned staging slots (0:
    /// pageable), the lane's rate its probe measured and the rate the rule
    /// prices in GB/s, and the least unit width that streams.
    fn xstream_word(&self) -> String;
}

/// What one layer of a unit streamed ([`XStream::layer`]), for the
/// `xstream` record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct XLayer {
    pub layer: usize,
    /// The unit's columns.
    pub cols: usize,
    /// Host experts the unit routes at the layer after the pick.
    pub host: usize,
    /// Of them, the ones the rule sends to the card.
    pub tail: usize,
    /// Of those, the ones that stream: the rule's set cut at the ring's half
    /// and at the balance, in rank order.
    pub streamed: usize,
    /// Columns the streamed experts take off the host: the serve's excluded
    /// slots of the layer.
    pub streamed_columns: u64,
    /// Columns the host union keeps.
    pub host_columns: u64,
    /// Bytes the stream copies.
    pub bytes: u64,
    /// Host microseconds the split and the issue took, and of them waiting
    /// for the fill threads to take in the backlog.
    pub issue_us: u64,
    pub backlog_us: u64,
}

/// What a call streamed ([`XStream::end_call`]), for the `xstream end`
/// record.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct XReport {
    /// Layers that streamed, and their experts and bytes.
    pub layers: usize,
    pub streamed: usize,
    pub bytes: u64,
    pub issue_us: u64,
    pub backlog_us: u64,
    /// The ring's slots a half, the staging's slots (0: pageable) and the
    /// lane's rate its probe measured [B/µs], which the rule does not read.
    pub half_slots: usize,
    pub staging_slots: usize,
    pub lane_b_per_us: f64,
    /// Host microseconds the end took.
    pub end_us: u64,
}

/// The order a batch walk of several units over one shared stream runs in:
/// every unit's first front ahead of the shadows, so the first unit's wait
/// for its landings sits behind the other units' fronts and never in front
/// of them ([`BatchOrder::Fronts`]). A walk that shares the stream declares
/// it as its [`runtime::sched::LayerProgram::BATCH_ORDER`].
pub const GROUP_WALK_ORDER: BatchOrder = BatchOrder::Fronts;

/// The most landing batches a layer's copies split into ([`land_batch_size`]
/// keeps the count at or under it).
pub const LAND_BATCHES: usize = 3;

/// A layer's landing slots a card route's GEMM batch covers, and the event
/// that guards it: one of the batches a layer's copies on the copy stream
/// split into ([`land_batch_size`]), each batch's event recorded on the copy
/// stream after that batch's last copy, so the GEMM batch over the batch's
/// slots — reading the batch's map row, the layer's places with every expert
/// outside the batch [`HOST`] — waits for its own copies and no later one.
/// `event` is `None` for a batch over slots no copy lands in, which waits
/// nothing. Both landing paths fill them: the residency machine's pick (its
/// admits into the stage stacks) and this module's stream (its copies into a
/// ring half); the card route only reads them.
///
/// [`HOST`]: super::slots::HOST
#[derive(Clone, Debug)]
pub struct LandBatch {
    /// The batch's map row, host-mapped and device-visible: `experts` words.
    pub map: sys::CUdeviceptr,
    /// The copy stream's event after the batch's last copy; `None` when no
    /// copy lands in the batch's slots.
    pub event: Option<Arc<CudaEvent>>,
}

impl LandBatch {
    /// `f` over this batch's map row as a non-owning window of `experts`
    /// words, the row given back after: the window frees nothing, so the
    /// launches `f` enqueued read the row as they read a buffer of their
    /// own. `f`'s error is returned whole.
    pub fn with_map<T>(
        &self,
        ctx: &Arc<CudaContext>,
        experts: usize,
        f: impl FnOnce(&DeviceBuffer<u32>) -> Result<T, GpuError>,
    ) -> Result<T, GpuError> {
        // SAFETY: the batch's map row is `experts` words of the pool's
        // host-mapped rows ([`LandRows::row_dev`]), which stay in place for
        // the stream's life; the window owns nothing and frees nothing. Its
        // contents are the landing path's contract: filled before the
        // launches `f` enqueues were issued, rewritten only once they have
        // run.
        let bmap = unsafe { crate::tensor::window::<u32>(self.map, experts, ctx) };
        let ran = f(&bmap);
        drop(std::mem::ManuallyDrop::into_inner(bmap).into_raw_parts());
        ran
    }
}

/// The batch size rule: a layer's `n` landing slots split into batches of at
/// most this many, the first a third of them. The shape is Strata's measured
/// one (`prefill.cpp`'s `per = max(1, ring / 3)`, its comment: "the next
/// batch's blobs arrive while one computes"): with a third of the landing
/// set a GEMM batch, one third computes while the copy lane streams the
/// next and the issuer may run into the third after it, so the first batch
/// that waits is bound by a third of the layer's copy time and the GEMMs
/// finish under the copies whenever the copies are the layer's longer side.
/// Rounded up (Strata floors), so the batch count stays at or under
/// [`LAND_BATCHES`] for every `n`.
#[must_use]
pub fn land_batch_size(n: usize) -> usize {
    n.div_ceil(LAND_BATCHES).max(1)
}

/// Per layer of a map, the landing batches' map rows (host-mapped,
/// device-visible, one a batch past a leading batch-0 row) and events: the
/// pool both landing paths write their [`LandBatch`]s through. A layer's
/// rows are written by the host at its landing (before the launches that read
/// them are enqueued) and read by the card routes of that layer's units in
/// that walk alone: the layer's next landing is the next walk's pick, which
/// the host runs only after it has seen the next walk's first-unit download of
/// the layer complete, and the engine stream runs that download after every
/// card route of this walk.
pub struct LandRows {
    rows: MappedHost,
    /// Per layer, [`LAND_BATCHES`] events.
    events: Vec<Vec<Arc<CudaEvent>>>,
    experts: usize,
}

impl LandRows {
    /// `layers` layers of `experts` experts, their batch rows and events.
    /// Load-time only.
    pub fn new(
        ctx: &Arc<CudaContext>,
        layers: usize,
        experts: usize,
    ) -> Result<LandRows, GpuError> {
        let rows = MappedHost::new(
            ctx,
            layers * (1 + LAND_BATCHES) * experts * 4,
            "cuMemHostAlloc (landing batches)",
        )?;
        let events = (0..layers)
            .map(|_| {
                (0..LAND_BATCHES)
                    .map(|_| ctx.new_event(None).map(Arc::new))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(LandRows {
            rows,
            events,
            experts,
        })
    }

    /// Batch `batch`'s map row of `layer`, `experts` words for the host to
    /// fill. Batch 0 is each landing path's own leading row (the slots no
    /// copy lands in); the batches after it hold the landing batches.
    #[must_use]
    pub fn row_mut(&mut self, layer: usize, batch: usize) -> &mut [u32] {
        let at = (layer * (1 + LAND_BATCHES) + batch) * self.experts * 4;
        // SAFETY: the row is `experts` words inside the rows page (sized
        // `(1 + LAND_BATCHES)` rows a layer, page-aligned): inside it and
        // 4-aligned. Nothing reads the row on the card until the landing
        // path publishes this fill by enqueuing the launches that name it.
        unsafe { std::slice::from_raw_parts_mut(self.rows.host_at(at).cast::<u32>(), self.experts) }
    }

    /// Batch `batch`'s event of `layer` (`0..LAND_BATCHES`), recorded on the
    /// copy stream after that batch's last copy.
    #[must_use]
    pub fn event(&self, layer: usize, batch: usize) -> Arc<CudaEvent> {
        Arc::clone(&self.events[layer][batch])
    }

    /// The batch's map row's device address, for a [`LandBatch`].
    #[must_use]
    pub fn row_dev(&self, layer: usize, batch: usize) -> sys::CUdeviceptr {
        self.rows
            .dev_at((layer * (1 + LAND_BATCHES) + batch) * self.experts * 4)
    }
}

/// The card's view of a layer's stream, for its card route
/// ([`XStream::ring_layer`]).
#[derive(Clone, Debug)]
pub struct RingLayer {
    /// The experts the half holds, slots `0..n`.
    pub n: usize,
    /// The half's slots: the experts a route over it counts.
    pub half_slots: usize,
    /// The ring row and the union row ([`XStream`]'s module doc), each the
    /// router's experts of `u32`.
    pub ring_map: sys::CUdeviceptr,
    pub union_map: sys::CUdeviceptr,
    /// Per part the half's first slot, in the source's part order: gate, up
    /// and down, each one part or [`RingLayer::planes`] parts in a row; 0
    /// past the source's parts.
    pub parts: [sys::CUdeviceptr; MAX_PARTS],
    /// The parts a matrix takes: 1 (its bytes whole), or 3 for an NVFP4
    /// planes expert (its codes, block scales and tensor scale, each a stack
    /// of the half's slots).
    pub planes: usize,
    /// The landing batches the layer's copies split into, a GEMM batch each
    /// ([`LandBatch`]): the ring pass runs one per batch, waiting its event.
    pub batches: Vec<LandBatch>,
}

/// An expert's matrices: gate, up, down.
pub const PARTS: usize = 3;

/// The most parts the ring holds an expert in: each matrix whole ([`PARTS`]),
/// or in three planes, an NVFP4 planes expert's codes, block scales and
/// tensor scale (`super::swap::Transform`).
pub const MAX_PARTS: usize = 3 * PARTS;

/// A half of the expert stream's ring as a family's routed launches read a
/// layer's stacks ([`with_ring_windows`]): its gate, up and down, each a
/// stack of the half's slots at the layer's own types and row words.
pub struct RingStacks<'a> {
    pub gate: &'a DeviceTensor<u32>,
    pub up: &'a DeviceTensor<u32>,
    pub down: &'a DeviceTensor<u32>,
}

/// Layer `r`'s stream on the card for the length of `f`: its union map of
/// `n_expert` words, and its half's gate, up and down ([`RingStacks`]),
/// `half_slots` experts of `rows[p]` rows of `cols[p]` words in each part
/// `p`; the windows are given back after `f` on every path.
pub fn with_ring_windows<T>(
    r: &RingLayer,
    ctx: &Arc<CudaContext>,
    n_expert: usize,
    rows: [usize; PARTS],
    cols: [usize; PARTS],
    f: impl FnOnce(&DeviceBuffer<u32>, &RingStacks<'_>) -> Result<T, GpuError>,
) -> Result<T, GpuError> {
    // SAFETY: the stream's union row is `n_expert` words of its mapped host
    // rows, and its half holds `half_slots` experts of the layer's part bytes
    // in each part from the half's first slot — its gate and up rows of the
    // card stacks' words, its down rows likewise (`XStream::ring_layer`); all
    // stay allocated while the stream lives, past `f`'s launches, and the
    // windows are given back below.
    let (union_map, gate, up, down) = unsafe {
        (
            crate::tensor::window::<u32>(r.union_map, n_expert, ctx),
            DeviceTensor::<u32>::window(r.parts[0], r.half_slots * rows[0], cols[0], ctx),
            DeviceTensor::<u32>::window(r.parts[1], r.half_slots * rows[1], cols[1], ctx),
            DeviceTensor::<u32>::window(r.parts[2], r.half_slots * rows[2], cols[2], ctx),
        )
    };
    let ran = f(
        &union_map,
        &RingStacks {
            gate: &gate,
            up: &up,
            down: &down,
        },
    );
    drop(std::mem::ManuallyDrop::into_inner(union_map).into_raw_parts());
    for w in [gate, up, down] {
        DeviceTensor::release(w);
    }
    ran
}

/// The card bytes a ring leaves free when it sizes itself: half the margin
/// the placement's expert rule leaves (`placement::workstation::MARGIN`),
/// for what a load allocates past the plan after the ring.
pub const KEEP_FREE: u64 = 512 << 20;

/// The fewest slots a half may hold: fewer, and the ring's bytes would not
/// pay for its launches.
pub const MIN_HALF: usize = 8;

/// Fill threads and pinned staging slots: four threads keep the copy
/// engine fed from the source's pages, and a staging ring of at most
/// `workstation::STAGING_SLOTS` slots, and at most the ring's half
/// ([`XStream::new`]), holds a layer's stream ahead of the copy stream. A
/// plan reserves those slots on the host (`workstation::reserve_stream_staging`),
/// the most a load pins.
pub const FILL_THREADS: usize = 4;
pub use model::placement::workstation::STAGING_SLOTS;

/// The experts the load's lane probe streams.
pub const PROBE: usize = 32;

/// The lane's rate [B/µs] the rule prices a copy of an expert in host memory
/// at: the median of the load probe's quiet readings on the box the family's
/// [`Costs`] rows were fitted on (`tools/flow/q38-constants.tsv`,
/// `a6000.lane_gbs`).
const LANE_HELD_B_PER_US: f64 = 20_410.0;

/// The lane's rate [B/µs] the rule prices a copy of an expert the NVMe tier
/// pages at, read through the file's mapping from the drive: the load probe's
/// quiet readings on an RTX 5090 workstation over a paged Qwen3.8 plan.
const LANE_PAGED_B_PER_US: f64 = 3_600.0;

/// The lane's rate [B/µs] the rule prices every copy at, for a probe of
/// `(bytes, paged)` experts: their bytes over the time their copies take at
/// their sources' rates ([`LANE_PAGED_B_PER_US`] for an expert the NVMe tier
/// pages, [`LANE_HELD_B_PER_US`] for one in host memory). A function of the
/// plan alone, never of a timing: two loads of one plan price their copies
/// alike, so the rule's picks and streams are a function of the prompt. 0
/// for a probe of no bytes.
fn lane_rate(probe: impl IntoIterator<Item = (u64, bool)>) -> f64 {
    let (mut bytes, mut us) = (0.0, 0.0);
    for (b, paged) in probe {
        let rate = if paged {
            LANE_PAGED_B_PER_US
        } else {
            LANE_HELD_B_PER_US
        };
        bytes += b as f64;
        us += b as f64 / rate;
    }
    if us > 0.0 { bytes / us } else { 0.0 }
}

/// What the refusal of a ring the card has no room for names: a caller that
/// resolves an unset lever tells it from every other refusal by it.
pub const XSTREAM_ROOM: &str = "XStream ring room";

/// What the refusal of a stream on a map with no host expert on a layer with stage stacks names ([`XSTREAM_ROOM`]'s
/// reader's twin): nothing streams where the card holds every expert.
pub const XSTREAM_NO_HOST: &str = "XStream host experts";

/// What the refusal of a ring whose stacks the family's ring pass does not
/// read names ([`XSTREAM_ROOM`]'s reader's twin): the pass reads a matrix the
/// ring holds whole as words, so a load whose card route reads NVFP4 planes
/// streams nothing.
pub const XSTREAM_FORM: &str = "XStream ring form";

/// The ring's halves.
const HALVES: usize = 2;

/// Bytes between two words of the staging page: a cache line each.
const WORD_STRIDE: usize = 64;

/// The streams the ring holds at once: a layer's, and the one before it
/// that its card route may still read.
#[derive(Clone, Debug)]
struct Batch {
    half: usize,
    n: usize,
}

/// The units of a walk that share one layer's stream in an open call: the
/// unit the stream was made for ([`XStream::layer`]) and the ones that joined
/// it ([`XStream::share`]), and how many of their card routes have read it
/// ([`XStream::read`]). A group of one is the unit alone.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Group {
    units: Vec<BatchKey>,
    reads: usize,
}

impl Group {
    /// The group a stream made for `unit` starts.
    fn of(unit: BatchKey) -> Group {
        Group {
            units: vec![unit],
            reads: 0,
        }
    }

    /// Whether `unit` is in the group.
    fn holds(&self, unit: BatchKey) -> bool {
        self.units.contains(&unit)
    }

    /// `unit` joins the group. Refused, with the reason: no unit has a stream
    /// in this call, a card route has read the stream already (the group's
    /// size is closed at its first read, or the layer would close early), or
    /// `unit` is in the group.
    fn join(&mut self, unit: BatchKey) -> Result<(), &'static str> {
        if self.units.is_empty() {
            return Err("no unit streams it in this call");
        }
        if self.reads > 0 {
            return Err("a card route has read its stream already");
        }
        if self.holds(unit) {
            return Err("that unit is in its group already");
        }
        self.units.push(unit);
        Ok(())
    }

    /// One unit's card route has read the stream; whether that was the
    /// group's last.
    fn read(&mut self) -> bool {
        self.reads += 1;
        self.reads >= self.units.len()
    }
}

/// One expert's staging for a fill thread: staging slot `slot`, `ticket` its
/// use of that slot, from 1.
struct Job {
    layer: usize,
    id: u32,
    slot: usize,
    ticket: u32,
}

/// What the fill threads and the engine thread share.
struct LaneShared {
    source: Arc<dyn SwapSource>,
    /// `slots` slots of `slot_bytes`, pinned and device-mapped.
    staging: MappedHost,
    slot_bytes: usize,
    slots: usize,
    /// Per staging slot: `staged` (the ticket once the slot holds the
    /// expert, or once its failure is recorded) at `2k`, `drained` (the copy
    /// stream's ticket once it has read the slot) at `2k + 1`.
    words: MappedHost,
    deadline: Duration,
    /// Microseconds every fill thread waits before it stages a job: 0 but
    /// under a gate's seam ([`XStream::delay_lane`]).
    delay_us: AtomicU64,
    served: AtomicU64,
    stop: AtomicBool,
    /// The first staging failure, which the next layer or the call's end
    /// returns.
    failed: Mutex<Option<String>>,
}

impl LaneShared {
    fn word(&self, i: usize) -> &AtomicU32 {
        self.words
            .atomic_u32(i * WORD_STRIDE)
            .expect("the staging words hold two a staging slot")
    }

    fn staged(&self, k: usize) -> &AtomicU32 {
        self.word(2 * k)
    }

    fn drained(&self, k: usize) -> &AtomicU32 {
        self.word(2 * k + 1)
    }

    /// Let through every copy the copy stream holds behind a staging word,
    /// `issued` jobs issued: each slot's word gets the last ticket issued to
    /// it, which every wait on that slot is at or below. The wait is cyclic
    /// (`(i32)(word - ticket) >= 0`), so `u32::MAX` would not release a
    /// ticket below 2^31. For the drop only, once the fill threads are
    /// joined. Every issued ticket fits u32 (`copy_expert` refuses one past
    /// it).
    fn release_waits(&self, issued: u64) {
        let slots = self.slots as u64;
        for n in issued.saturating_sub(slots)..issued {
            if let Ok(ticket) = u32::try_from(n / slots + 1) {
                self.staged((n % slots) as usize)
                    .fetch_max(ticket, Ordering::AcqRel);
            }
        }
    }

    /// Wait until `ready`, within the deadline: `Ok(false)` when the lane
    /// stops first.
    fn wait_until(&self, ready: impl Fn() -> bool, what: &str) -> Result<bool, GpuError> {
        let t0 = Instant::now();
        let mut spins = 0u32;
        while !ready() {
            if self.stop.load(Ordering::Acquire) {
                return Ok(false);
            }
            if t0.elapsed() > self.deadline {
                return Err(GpuError::protocol(
                    "XStream lane",
                    format!("{what}: not in {:?}", self.deadline),
                ));
            }
            if spins < 256 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::sleep(Duration::from_micros(20));
            }
        }
        Ok(true)
    }

    /// Copy `job`'s expert from its source into its staging slot, part
    /// after part, once the slot's last copy has drained, between the
    /// source's open of the read and its release, which follows the copy
    /// whether it failed or not ([`SwapSource::open_read`]).
    fn serve(&self, job: &Job) -> Result<bool, GpuError> {
        const WHAT: &str = "XStream staging";
        let prev = job.ticket - 1;
        let drained = || self.drained(job.slot).load(Ordering::Acquire) >= prev;
        if !self.wait_until(drained, "a staging slot's last copy drained")? {
            return Ok(false);
        }
        let delay = self.delay_us.load(Ordering::Acquire);
        if delay > 0 {
            std::thread::sleep(Duration::from_micros(delay));
        }
        self.source.open_read(job.layer, job.id)?;
        let copied = (|| {
            let mut at = job.slot * self.slot_bytes;
            let end = at + self.slot_bytes;
            for (part, &want) in self.source.part_bytes(job.layer).iter().enumerate() {
                let piece = self.source.source(job.layer, job.id, part)?;
                if piece.target_len() != Some(want) || at + want > end {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "layer {} expert {} part {part}: {} source bytes ({:?}) for a part \
                             of {want} in a staging slot of {}",
                            job.layer,
                            job.id,
                            piece.bytes.len(),
                            piece.transform,
                            self.slot_bytes
                        ),
                    ));
                }
                // SAFETY: [at, at + want) lies inside the staging page (the
                // slot's span, checked above); no copy reads the slot until
                // its staged word publishes this use, and this thread holds
                // the slot's only job until then, so this is the span's one
                // reference while it lives. The word's Release store follows
                // these stores in program order.
                let dst = unsafe { std::slice::from_raw_parts_mut(self.staging.host_at(at), want) };
                piece.transform.apply(piece.bytes, dst)?;
                at += want;
            }
            Ok(())
        })();
        let released = self.source.release_read(job.layer, job.id);
        copied.and(released).map(|()| true)
    }

    fn fail(&self, job: &Job, e: &str) {
        let mut f = self
            .failed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if f.is_none() {
            *f = Some(format!(
                "staging layer {} expert {} failed: {e}",
                job.layer, job.id
            ));
        }
    }

    /// A fill thread: each job served and its ticket published, a failure
    /// recorded and published all the same.
    fn run(&self, jobs: &Mutex<mpsc::Receiver<Job>>) {
        loop {
            let job = {
                let rx = jobs
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                match rx.recv() {
                    Ok(j) => j,
                    Err(_) => return,
                }
            };
            match catch_unwind(AssertUnwindSafe(|| self.serve(&job))) {
                Ok(Ok(true)) => {}
                Ok(Ok(false)) => return,
                Ok(Err(e)) => self.fail(&job, &e.to_string()),
                Err(payload) => {
                    let why = payload
                        .downcast_ref::<&str>()
                        .map(ToString::to_string)
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "a payload that is not text".to_string());
                    self.fail(&job, &format!("the fill thread panicked: {why}"));
                }
            }
            self.served.fetch_add(1, Ordering::AcqRel);
            self.staged(job.slot)
                .fetch_max(job.ticket, Ordering::AcqRel);
        }
    }
}

/// The lane: pinned staging with its fill threads, or the source's pageable
/// bytes.
enum Lane {
    Pinned {
        shared: Arc<LaneShared>,
        tx: Option<mpsc::Sender<Job>>,
        threads: Vec<JoinHandle<()>>,
        jobs: u64,
    },
    Pageable {
        source: Arc<dyn SwapSource>,
    },
}

/// The ring's card memory: per part, two halves of `half` slots of the
/// widest layer's part.
struct DeviceRing {
    parts: Vec<DeviceBuffer<u32>>,
    widest: Vec<usize>,
    half: usize,
}

impl DeviceRing {
    /// The address of slot `i` of half `h` in part `p` at `bytes` a slot.
    fn at(&self, h: usize, p: usize, i: usize, bytes: usize) -> sys::CUdeviceptr {
        self.parts[p].cu_deviceptr() + ((h * self.half * self.widest[p]) + i * bytes) as u64
    }
}

/// The expert stream (module doc).
///
/// Field order is drop order after [`Drop`] has stopped the fill threads and
/// drained the copy stream.
pub struct XStream {
    cfg: XCfg,
    copy: Arc<CudaStream>,
    ring: DeviceRing,
    /// Per half the ring row and the union row, `experts` words each.
    rows: MappedHost,
    lane: Lane,
    /// Per half: the engine stream's event after that stream's last reader.
    used: Vec<CudaEvent>,
    /// Streams issued since the load: the next one's half.
    issued: u64,
    /// Per layer the stream of the unit's walk, until its read.
    cur: Vec<Option<Batch>>,
    /// Per layer the streamed ids of the layer's last stream in the open
    /// call, ascending, and its group of units: the host serve's exclusion
    /// set, for those units' serves alone. The serve of a unit's layer runs
    /// after its card route's read (`runtime::sched`'s batch orders), so the
    /// set stands past the read; a unit the walk takes no stream for (one
    /// narrower than the pick's floor) is in no group and reads none.
    excl: Vec<Vec<u32>>,
    groups: Vec<Group>,
    /// A call is open ([`XStream::begin_call`] to [`XStream::end_call`]).
    open: bool,
    /// The rule's scratch.
    split: Split,
    rank: Vec<u32>,
    /// The lane's rate the rule prices every copy at ([`lane_rate`] of the
    /// probe's experts), and the rate the probe measured, a witness the rule
    /// does not read.
    lane_b_per_us: f64,
    probed_b_per_us: f64,
    report: XReport,
    /// Per layer the rule's constants: `None` for a layer the source holds
    /// no stage stack of.
    consts: Vec<Option<Constants>>,
    /// The landing batches' rows and events ([`LandRows`]), and per layer
    /// how many batches its last stream took.
    land: LandRows,
    land_n: Vec<usize>,
}

impl XStream {
    /// The stream over `source`'s stage stacks for the map's `layers`
    /// layers: the ring sized from the card's free bytes past
    /// `cfg.keep_free` (at most `cfg.max_half` slots a half), the staging
    /// (pinned, else the source's pageable bytes), the fill threads, then the
    /// lane's probe — `cfg.probe`'s experts streamed into the first half and
    /// timed, a witness beside the rate the rule prices, [`lane_rate`] of the
    /// same experts by where their bytes sit. Refused by name: a ring of fewer
    /// than `cfg.min_half` slots a half, no layer with stage stacks, a part
    /// not a whole number of words, a part count other than [`PARTS`] or
    /// [`MAX_PARTS`] or other than another layer's, an empty probe.
    /// Load-time only.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        source: Arc<dyn SwapSource>,
        layers: usize,
        mut cfg: XCfg,
    ) -> Result<XStream, GpuError> {
        const WHAT: &str = "XStream::new";
        let mut widest: Vec<usize> = Vec::new();
        let mut slot_bytes = 0usize;
        for l in 0..layers {
            let parts = source.part_bytes(l);
            if parts.is_empty() {
                continue;
            }
            let count = !(parts.len() == PARTS || parts.len() == MAX_PARTS)
                || (!widest.is_empty() && widest.len() != parts.len());
            if count || parts.iter().any(|&b| b == 0 || b % 4 != 0) {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l}: expert parts of {parts:?} bytes; the stream holds {PARTS} or \
                         {MAX_PARTS} an expert, the same count every layer, each a nonzero \
                         multiple of 4"
                    ),
                ));
            }
            if widest.is_empty() {
                widest = vec![0; parts.len()];
            }
            for (w, &b) in widest.iter_mut().zip(parts) {
                *w = (*w).max(b);
            }
            slot_bytes = slot_bytes.max(parts.iter().sum());
        }
        if slot_bytes == 0 {
            return Err(GpuError::shape(
                WHAT,
                "no layer of the map holds stage stacks: nothing could stream",
            ));
        }
        if cfg.probe.is_empty() || cfg.min_half == 0 || cfg.experts == 0 {
            return Err(GpuError::shape(
                WHAT,
                "a probe of at least one expert, a half of at least one slot and the router's \
                 experts",
            ));
        }
        // A slot's bytes rounded to the GEMMs' 16-byte stack alignment.
        let widest: Vec<usize> = widest.iter().map(|b| b.next_multiple_of(16)).collect();
        let per_slot: usize = widest.iter().sum();
        let (free, _) = card_mem(ctx)?;
        let mut room = free.saturating_sub(cfg.keep_free);
        if let Some(cap) = cfg.room_cap {
            room = room.min(cap);
        }
        let fit = usize::try_from(room / (HALVES * per_slot) as u64).unwrap_or(usize::MAX);
        let half = fit.min(cfg.max_half);
        if half < cfg.min_half {
            return Err(GpuError::shape(
                XSTREAM_ROOM,
                format!(
                    "the card holds {free} B free, {room} B past the {} B the ring leaves free: \
                     {fit} slots of {per_slot} B fit a half, the map holds at most {} host \
                     experts a layer, so a half of {half}, fewer than the {} a stream needs",
                    cfg.keep_free, cfg.max_half, cfg.min_half
                ),
            ));
        }
        let parts = widest
            .iter()
            .map(|&w| DeviceBuffer::<u32>::zeroed(stream, HALVES * half * w / 4))
            .collect::<Result<Vec<_>, _>>()?;
        let ring = DeviceRing {
            parts,
            widest,
            half,
        };
        let rows = MappedHost::new(
            ctx,
            HALVES * 2 * cfg.experts * 4,
            "cuMemHostAlloc (xstream rows)",
        )?;
        let copy = crate::role_stream(ctx, crate::StreamRole::Background)?;
        let new_events = |n| {
            (0..n)
                .map(|_| ctx.new_event(None))
                .collect::<Result<Vec<_>, _>>()
        };
        let used = new_events(HALVES)?;
        let land = LandRows::new(ctx, layers, cfg.experts)?;
        // A layer's stream is at most a half of experts: the staging, pinned for the load's life,
        // holds one stream and no more.
        cfg.staging_slots = cfg.staging_slots.min(half);
        let lane = start_lane(ctx, source, slot_bytes, &cfg)?;
        let mut x = XStream {
            consts: Vec::new(),
            cfg,
            copy,
            ring,
            rows,
            lane,
            used,
            issued: 0,
            cur: vec![None; layers],
            excl: vec![Vec::new(); layers],
            groups: vec![Group::default(); layers],
            open: false,
            split: Split::default(),
            rank: Vec::new(),
            lane_b_per_us: 0.0,
            probed_b_per_us: 0.0,
            report: XReport::default(),
            land,
            land_n: vec![0; layers],
        };
        x.probed_b_per_us = x.probe()?;
        let source = Arc::clone(x.source());
        x.lane_b_per_us = lane_rate(x.probe_set().iter().map(|&(layer, id)| {
            let bytes: usize = source.part_bytes(layer).iter().sum();
            (bytes as u64, source.paged(layer, id))
        }));
        x.consts = (0..layers).map(|l| x.constants_of(l)).collect();
        x.report = x.fresh_report();
        Ok(x)
    }

    /// The lane's rate [B/µs]: the probe's experts streamed into the first
    /// half through the lane, timed from the first issue to the copy stream's
    /// drain. They land in half 0, ahead of the first stream's copies on the
    /// copy stream, and nothing reads them.
    fn probe(&mut self) -> Result<f64, GpuError> {
        const WHAT: &str = "XStream probe";
        let probe = self.probe_set().to_vec();
        let t0 = Instant::now();
        let mut bytes = 0u64;
        for (i, &(layer, id)) in probe.iter().enumerate() {
            self.wait_backlog()?;
            bytes += self.copy_expert(layer, id, 0, i)?;
        }
        drain_within(&self.copy, self.cfg.deadline, WHAT)?;
        self.refuse_failure(WHAT)?;
        let us = t0.elapsed().as_secs_f64() * 1e6;
        if us <= 0.0 || bytes == 0 {
            return Err(GpuError::protocol(WHAT, "a probe that copied nothing"));
        }
        Ok(bytes as f64 / us)
    }

    /// The probe's experts: `cfg.probe`'s, at most a half's slots of them.
    fn probe_set(&self) -> &[(usize, u32)] {
        let n = self.cfg.probe.len().min(self.ring.half);
        &self.cfg.probe[..n]
    }

    /// The rule's constants of layer `layer` at the lane's rate: the family
    /// row's costs with no unit's burst share and no model's card tail — a
    /// load knows neither — for the paths that want the row itself. The
    /// paths that know a unit's counts set its share through
    /// [`xsplit::union_burst_share`] before they run the rule.
    fn constants_of(&self, layer: usize) -> Option<Constants> {
        let parts = self.source().part_bytes(layer);
        if parts.is_empty() {
            return None;
        }
        let c = self.cfg.costs;
        Some(Constants {
            lane_b_per_us: self.lane_b_per_us,
            host_us_per_col: c.host_us_per_col,
            host_us_fixed: c.host_us_fixed,
            card_us_fixed: c.card_us_fixed,
            card_us_per_col: c.card_us_per_col,
            union_burst_share: 0.0,
            card_tail_us: 0.0,
            expert_b: parts.iter().map(|&b| b as u64).sum(),
            experts: self.cfg.experts,
            top_k: self.cfg.top_k,
        })
    }

    fn source(&self) -> &Arc<dyn SwapSource> {
        match &self.lane {
            Lane::Pinned { shared, .. } => &shared.source,
            Lane::Pageable { source } => source,
        }
    }

    fn fresh_report(&self) -> XReport {
        XReport {
            half_slots: self.ring.half,
            staging_slots: match &self.lane {
                Lane::Pinned { shared, .. } => shared.slots,
                Lane::Pageable { .. } => 0,
            },
            lane_b_per_us: self.probed_b_per_us,
            ..XReport::default()
        }
    }

    /// The ring's slots a half.
    #[must_use]
    pub fn half_slots(&self) -> usize {
        self.ring.half
    }

    /// The pinned staging's slots; 0 when the lane copies pageable bytes.
    #[must_use]
    pub fn staging_slots(&self) -> usize {
        self.fresh_report().staging_slots
    }

    /// The lane's rate the rule prices every copy at [B/µs] ([`lane_rate`]).
    #[must_use]
    pub fn lane_b_per_us(&self) -> f64 {
        self.lane_b_per_us
    }

    /// The lane's rate the load's probe measured [B/µs]: a witness of the
    /// sitting, which the rule does not read.
    #[must_use]
    pub fn probed_b_per_us(&self) -> f64 {
        self.probed_b_per_us
    }

    /// Layer `layer`'s constants of the rule; `None` for a layer with no
    /// stage stack.
    #[must_use]
    pub fn constants(&self, layer: usize) -> Option<Constants> {
        self.consts.get(layer).copied().flatten()
    }

    /// Set the rule's costs (the family's, or a calibration's): every layer's
    /// constants follow at the lane's priced rate. Refused by name while a
    /// call is open.
    pub fn set_costs(&mut self, costs: Costs) -> Result<(), GpuError> {
        if self.open {
            return Err(GpuError::state(
                "XStream::set_costs",
                "no call open (XStream::end_call)",
            ));
        }
        self.cfg.costs = costs;
        self.consts = (0..self.consts.len())
            .map(|l| self.constants_of(l))
            .collect();
        Ok(())
    }

    /// The least unit width any layer streams at ([`xsplit::m_min`]); `None`
    /// with no layer that streams.
    #[must_use]
    pub fn m_min(&self) -> Option<u64> {
        self.consts.iter().flatten().map(xsplit::m_min).min()
    }

    /// Open a call: its report starts over. Every unit's layers stream
    /// through [`XStream::layer`] and [`XStream::read`].
    pub fn begin_call(&mut self) {
        self.report = self.fresh_report();
        self.cur.fill(None);
        for e in &mut self.excl {
            e.clear();
        }
        self.groups.fill(Group::default());
        self.open = true;
    }

    /// Layer `unit.layer` of the unit `unit` (the batch walk's key of the
    /// layer's serve) of `cols` columns: `counts` one count per
    /// router expert (the unit's routed picks), `host` the layer's host
    /// experts after the pick (any order), `admitted` the pick's admits,
    /// `stage` the layer's stage map row (a card place, else [`HOST`]) and
    /// `n_card` the stage stacks' experts. The rule's set
    /// ([`stream_tail`]) cut at the half and at the balance streams: the
    /// half's rows written, the copies issued onto the copy stream behind
    /// the half's last reader, a batch event recorded after each batch's
    /// last copy ([`land_batch_size`]) — the engine stream waits for none of
    /// them here; the layer's card route waits each batch before the GEMM
    /// batch over its slots. A layer that streams nothing issues nothing.
    /// Refused by name: a layer with no stage stack, a layer streaming twice
    /// before its read, a stream into a half another layer's open stream
    /// still holds, a staging failure, and the rule's own refusals.
    #[allow(
        clippy::too_many_arguments,
        reason = "the unit, its width and counts, the host set, the pick, the stage row and the engine stream (rust-quality R8)"
    )]
    pub fn layer(
        &mut self,
        // The engine stream: no longer waited here — the layer's card route
        // waits each batch before the GEMM batch over its slots
        // (`XStream::wait_layer` for the wholesale form).
        _engine: &CudaStream,
        unit: BatchKey,
        cols: usize,
        counts: &[u32],
        (host, admitted): (&[u32], usize),
        (stage, n_card): (&[u32], usize),
    ) -> Result<XLayer, GpuError> {
        const WHAT: &str = "XStream::layer";
        let t0 = Instant::now();
        let layer = unit.layer;
        self.refuse_failure(WHAT)?;
        if !self.open {
            return Err(GpuError::protocol(
                WHAT,
                format!("layer {layer} streams outside a call (XStream::begin_call)"),
            ));
        }
        let mut k = self
            .constants(layer)
            .ok_or_else(|| GpuError::shape(WHAT, format!("layer {layer} holds no stage stack")))?;
        if self.cur.get(layer).is_none_or(Option::is_some) {
            return Err(GpuError::protocol(
                WHAT,
                format!("layer {layer} streams again before its stream's read"),
            ));
        }
        if stage.len() != self.cfg.experts {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a stage row of {} words for {} experts",
                    stage.len(),
                    self.cfg.experts
                ),
            ));
        }
        self.groups[layer] = Group::of(unit);
        self.excl[layer].clear();
        // The unit's own share over the host set the pick left: the counts
        // the union still serves, the experts it still lists. The rule's
        // every copy below prices at the beside-union rate this gives.
        let host_routed = host.iter().filter(|&&id| counts[id as usize] > 0).count();
        let host_cols: u64 = host.iter().map(|&id| u64::from(counts[id as usize])).sum();
        k.union_burst_share = xsplit::union_burst_share(host_cols, host_routed as u64, &k);
        stream_tail(counts, host, &k, &mut self.split)
            .map_err(|e| GpuError::shape(WHAT, format!("layer {layer}: {e}")))?;
        let tail = self.split.stream.len();
        let kept_us = k.host_us_per_col * self.split.host_columns as f64;
        let all: u64 = counts.iter().map(|&c| u64::from(c)).sum();
        let card_us = k.card_us_per_col * all.saturating_sub(host_cols) as f64;
        let side = Sides {
            admitted,
            kept_us,
            card_us,
        };
        let take = balance_cut(&self.split.stream, counts, &k, side).min(self.ring.half);
        self.rank.clear();
        self.rank.extend_from_slice(&self.split.stream[..take]);
        let host_columns = self.split.host_columns
            + self.split.stream[take..]
                .iter()
                .map(|&id| u64::from(counts[id as usize]))
                .sum::<u64>();
        let streamed_columns = self
            .rank
            .iter()
            .map(|&id| u64::from(counts[id as usize]))
            .sum();
        let mut out = XLayer {
            layer,
            cols,
            host: host_routed,
            tail,
            streamed: take,
            streamed_columns,
            host_columns,
            ..XLayer::default()
        };
        if take > 0 {
            let half = (self.issued % HALVES as u64) as usize;
            if let Some(held) = self
                .cur
                .iter()
                .position(|b| b.as_ref().is_some_and(|b| b.half == half))
            {
                let g = &self.groups[held];
                return Err(GpuError::protocol(
                    WHAT,
                    format!(
                        "layer {layer} streams into half {half}, which layer {held}'s stream \
                         still holds (read by {} of its {} units)",
                        g.reads,
                        g.units.len()
                    ),
                ));
            }
            self.write_rows(half, stage, n_card)?;
            self.write_land_rows(layer);
            let (bytes, backlog) = self.issue(layer, half)?;
            self.issued += 1;
            self.cur[layer] = Some(Batch { half, n: take });
            let ex = &mut self.excl[layer];
            ex.extend_from_slice(&self.rank);
            ex.sort_unstable();
            out.bytes = bytes;
            out.backlog_us = backlog;
        }
        out.issue_us = micros(t0);
        let r = &mut self.report;
        r.layers += usize::from(take > 0);
        r.streamed += take;
        r.bytes += out.bytes;
        r.issue_us += out.issue_us;
        r.backlog_us += out.backlog_us;
        Ok(out)
    }

    /// The half's rows for the ranked stream in `self.rank`: the ring row
    /// (slot `i` for the `i`-th, [`HOST`] for every other expert) and the
    /// union row (`stage`, with `n_card + i` for the `i`-th).
    fn write_rows(&mut self, half: usize, stage: &[u32], n_card: usize) -> Result<(), GpuError> {
        let e = self.cfg.experts;
        let base = u32::try_from(n_card)
            .ok()
            .filter(|_| n_card + self.rank.len() < HOST as usize)
            .ok_or_else(|| {
                GpuError::shape(
                    "XStream rows",
                    format!(
                        "{n_card} stage places and {} streamed past u32",
                        self.rank.len()
                    ),
                )
            })?;
        // SAFETY: the half's two rows are `e` words each at
        // `(2 · half + r) · e · 4` of the rows page (`HALVES · 2 · e · 4`
        // bytes, page-aligned): inside it and 4-aligned. The card's last
        // reads of this half were the card routes of the stream two back, the
        // last unit's of its group ([`XStream::read`]), which the engine
        // stream ran before the download of a later layer the host has
        // already waited for (the caller's count), so nothing reads these
        // words while they are written.
        let (ring, union) = unsafe {
            (
                std::slice::from_raw_parts_mut(
                    self.rows.host_at(2 * half * e * 4).cast::<u32>(),
                    e,
                ),
                std::slice::from_raw_parts_mut(
                    self.rows.host_at((2 * half + 1) * e * 4).cast::<u32>(),
                    e,
                ),
            )
        };
        ring.fill(HOST);
        union.copy_from_slice(stage);
        for (i, &id) in self.rank.iter().enumerate() {
            let i = u32::try_from(i).expect("a half's slots fit u32");
            ring[id as usize] = i;
            union[id as usize] = base + i;
        }
        Ok(())
    }

    /// The ring batch rows of the ranked stream in `self.rank` for layer
    /// `layer` ([`LandRows`]'s rows past the leading one): batch `b`'s row
    /// names only its own ranks' experts, at their ring slots (slot `i` the
    /// `i`-th ranked), every other expert [`HOST`].
    fn write_land_rows(&mut self, layer: usize) {
        let n = self.rank.len();
        let per = land_batch_size(n);
        self.land_n[layer] = n.div_ceil(per);
        for b in 0..self.land_n[layer] {
            let row = self.land.row_mut(layer, b + 1);
            row.fill(HOST);
            for i in (b * per)..n.min((b + 1) * per) {
                row[self.rank[i] as usize] = u32::try_from(i).expect("a half's slots fit u32");
            }
        }
    }

    /// Issue the ranked stream in `self.rank` of layer `layer` into half
    /// `half`: the copy stream waits for the half's last reader, each
    /// expert's copies follow, and the batch an expert's copy closes records
    /// its event after it ([`land_batch_size`]) — the engine stream's waits
    /// are the card route's, a batch a GEMM batch. The bytes copied and the
    /// host microseconds spent waiting for the staging backlog.
    fn issue(&mut self, layer: usize, half: usize) -> Result<(u64, u64), GpuError> {
        self.copy.wait(&self.used[half])?;
        let rank = std::mem::take(&mut self.rank);
        let mut bytes = 0u64;
        let mut backlog = 0u64;
        let per = land_batch_size(rank.len());
        let r = (|| {
            for (i, &id) in rank.iter().enumerate() {
                let t = Instant::now();
                self.wait_backlog()?;
                backlog += micros(t);
                bytes += self.copy_expert(layer, id, half, i)?;
                if (i + 1) % per == 0 || i + 1 == rank.len() {
                    self.land.event(layer, i / per).record(&self.copy)?;
                }
            }
            Ok::<(), GpuError>(())
        })();
        self.rank = rank;
        r?;
        Ok((bytes, backlog))
    }

    /// Wait, within the deadline, until the fill threads have taken in all
    /// but a staging ring's worth of the jobs issued: the copy stream never
    /// holds more copies behind unstaged jobs than the ring has slots.
    fn wait_backlog(&self) -> Result<(), GpuError> {
        let Lane::Pinned { shared, jobs, .. } = &self.lane else {
            return Ok(());
        };
        let (issued, bound) = (*jobs, shared.slots as u64);
        let fits = || issued - shared.served.load(Ordering::Acquire) < bound;
        match shared.wait_until(fits, "the fill threads taking in the staging backlog")? {
            true => Ok(()),
            false => Err(GpuError::protocol(
                "XStream lane",
                "the fill threads have stopped",
            )),
        }
    }

    /// Expert `id` of layer `layer` into slot `i` of half `half`, part by
    /// part, on the copy stream; the bytes.
    fn copy_expert(
        &mut self,
        layer: usize,
        id: u32,
        half: usize,
        i: usize,
    ) -> Result<u64, GpuError> {
        const WHAT: &str = "XStream copy";
        let source = Arc::clone(self.source());
        let parts = source.part_bytes(layer);
        if parts.len() != self.ring.widest.len() || i >= self.ring.half {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {layer}: {} parts into slot {i} of a half of {}",
                    parts.len(),
                    self.ring.half
                ),
            ));
        }
        let dst: Vec<sys::CUdeviceptr> = parts
            .iter()
            .enumerate()
            .map(|(p, &len)| self.ring.at(half, p, i, len))
            .collect();
        let copy = Arc::clone(&self.copy);
        match &mut self.lane {
            Lane::Pinned {
                shared, tx, jobs, ..
            } => {
                let n = *jobs;
                let slot = (n % shared.slots as u64) as usize;
                let ticket = u32::try_from(n / shared.slots as u64 + 1)
                    .map_err(|_| GpuError::protocol(WHAT, "the staging tickets passed u32"))?;
                tx.as_ref()
                    .and_then(|tx| {
                        tx.send(Job {
                            layer,
                            id,
                            slot,
                            ticket,
                        })
                        .ok()
                    })
                    .ok_or_else(|| GpuError::protocol(WHAT, "the fill threads have stopped"))?;
                *jobs += 1;
                let staged = shared.words.dev_at(2 * slot * WORD_STRIDE);
                // SAFETY: slot `slot`'s staged word lies in the words (sized
                // for `shared.slots` slots), which stay allocated until the
                // copy stream has drained (the drop drains it, or leaks them).
                unsafe {
                    mem_batch(
                        &copy,
                        &mut [op_wait_geq(staged, ticket)],
                        "xstream: wait for staging",
                    )?;
                }
                let mut at = slot * shared.slot_bytes;
                for (&len, &d) in parts.iter().zip(&dst) {
                    // SAFETY: the staging span [at, at + len) lies inside the
                    // slot (the parts sum to at most a slot's bytes, which
                    // `XStream::new` sized from every layer) and the page
                    // stays allocated until the copy stream has drained (the
                    // drop drains it, or leaks the page); `d` is a ring slot's
                    // part, `len` bytes inside the ring (`DeviceRing::at`).
                    let rc = unsafe {
                        sys::cuMemcpyHtoDAsync_v2(
                            d,
                            shared.staging.host_at(at).cast_const().cast(),
                            len,
                            copy.cu_stream(),
                        )
                    };
                    cu(rc, "cuMemcpyHtoDAsync_v2 (xstream slot)")?;
                    at += len;
                }
                let drained = shared.words.dev_at((2 * slot + 1) * WORD_STRIDE);
                // SAFETY: slot `slot`'s drained word lies in the words, which
                // stay allocated until the copy stream has drained, as above.
                unsafe {
                    mem_batch(
                        &copy,
                        &mut [op_write(drained, ticket)],
                        "xstream: staging drained",
                    )?;
                }
            }
            Lane::Pageable { .. } => {
                source.open_read(layer, id)?;
                let copied = (|| {
                    for (p, (&len, &d)) in parts.iter().zip(&dst).enumerate() {
                        let piece = source.source(layer, id, p)?;
                        if piece.bytes.len() != len || piece.transform != Transform::Identity {
                            return Err(GpuError::shape(
                                WHAT,
                                format!(
                                    "layer {layer} expert {id} part {p}: {} source bytes ({:?}) \
                                     for a part of {len}; the pageable lane copies a part as \
                                     its source holds it, a transformed part stages through the \
                                     pinned lane",
                                    piece.bytes.len(),
                                    piece.transform
                                ),
                            ));
                        }
                        // SAFETY: the open read keeps the source's bytes
                        // unchanged until the copy stream drains below;
                        // `d` holds this part's `len` bytes inside the ring.
                        let rc = unsafe {
                            sys::cuMemcpyHtoDAsync_v2(
                                d,
                                piece.bytes.as_ptr().cast(),
                                len,
                                copy.cu_stream(),
                            )
                        };
                        cu(rc, "cuMemcpyHtoDAsync_v2 (xstream slot, pageable)")?;
                    }
                    Ok(())
                })();
                // A failed part can follow copies already in flight.
                if let Err(e) = drain_within(&copy, self.cfg.deadline, WHAT) {
                    // The read stays open and its owner alive while DMA may
                    // still reach the source, including a reused arena slot.
                    std::mem::forget(source);
                    return Err(e);
                }
                let released = source.release_read(layer, id);
                copied.and(released)?;
            }
        }
        Ok(parts.iter().map(|&b| b as u64).sum())
    }

    /// Hold every fill thread `d` before it stages a job, so the lane runs
    /// behind the engine stream: a gate's seam (the landed clause of
    /// `gate_xstream`), never set on a serving path. Nothing on a pageable
    /// lane.
    #[doc(hidden)]
    pub fn delay_lane(&self, d: Duration) {
        if let Lane::Pinned { shared, .. } = &self.lane {
            let us = u64::try_from(d.as_micros()).unwrap_or(u64::MAX);
            shared.delay_us.store(us, Ordering::Release);
        }
    }

    /// Layer `layer`'s stream as its card route reads it; `None` when the
    /// layer streams nothing in the open unit.
    #[must_use]
    pub fn ring_layer(&self, layer: usize) -> Option<RingLayer> {
        let b = self.cur.get(layer)?.as_ref()?;
        let parts = self.source().part_bytes(layer);
        let e = self.cfg.experts;
        let mut at = [0; MAX_PARTS];
        for (p, a) in at.iter_mut().enumerate().take(self.ring.widest.len()) {
            *a = self
                .ring
                .at(b.half, p, 0, parts.get(p).copied().unwrap_or(0));
        }
        Some(RingLayer {
            n: b.n,
            half_slots: self.ring.half,
            ring_map: self.rows.dev_at(2 * b.half * e * 4),
            union_map: self.rows.dev_at((2 * b.half + 1) * e * 4),
            parts: at,
            planes: self.ring.widest.len() / PARTS,
            batches: (0..self.land_n.get(layer).copied().unwrap_or(0))
                .map(|b| LandBatch {
                    map: self.land.row_dev(layer, b + 1),
                    event: Some(self.land.event(layer, b)),
                })
                .collect(),
        })
    }

    /// The engine stream `engine` waits for every batch of layer `layer`'s
    /// stream — the contract the layer's card route honours batch by batch,
    /// for a caller that reads the ring wholesale. Nothing for a layer that
    /// streamed nothing.
    pub fn wait_layer(&self, engine: &CudaStream, layer: usize) -> Result<(), GpuError> {
        for b in 0..self.land_n.get(layer).copied().unwrap_or(0) {
            engine.wait(&self.land.event(layer, b))?;
        }
        Ok(())
    }

    /// Unit `unit` joins layer `layer`'s stream, made by the layer's
    /// [`XStream::layer`] for another unit of the same walk: its serve leaves
    /// out the same streamed experts ([`XStream::excluded`]), and the stream
    /// stays open until the unit's card route has read it too
    /// ([`XStream::read`]). A walk whose units share one pick a layer joins
    /// each later unit right after the layer's [`XStream::layer`], before any
    /// card route reads. Refused by name: outside a call, a unit of another
    /// layer than `layer`, a layer outside the stream's, a layer no unit
    /// streams in this call, a unit already in the group, and a layer whose
    /// stream a card route has read already.
    pub fn share(&mut self, layer: usize, unit: BatchKey) -> Result<(), GpuError> {
        const WHAT: &str = "XStream::share";
        if !self.open {
            return Err(GpuError::protocol(
                WHAT,
                format!("layer {layer} shared outside a call (XStream::begin_call)"),
            ));
        }
        if unit.layer != layer {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "unit {unit:?} of layer {} shared into layer {layer}",
                    unit.layer
                ),
            ));
        }
        let group = self
            .groups
            .get_mut(layer)
            .ok_or_else(|| GpuError::shape(WHAT, format!("layer {layer} outside the stream's")))?;
        group
            .join(unit)
            .map_err(|why| GpuError::protocol(WHAT, format!("layer {layer}: {why}")))
    }

    /// The streamed experts of `unit`'s layer, ascending: the host serve of
    /// that unit leaves them out. Empty for a unit outside the layer's last
    /// stream's group ([`XStream::layer`], [`XStream::share`]), when that
    /// unit streams nothing, and outside a call.
    #[must_use]
    pub fn excluded(&self, unit: BatchKey) -> &[u32] {
        if !self.open || !self.groups.get(unit.layer).is_some_and(|g| g.holds(unit)) {
            return &[];
        }
        self.excl.get(unit.layer).map_or(&[], Vec::as_slice)
    }

    /// One unit's card route of layer `layer` has run on the engine stream
    /// `engine`: the layer's stream has had its last read once every unit of
    /// its group has called this — one call a unit, in the order the units'
    /// card routes were enqueued. At the group's last unit the half's read
    /// event is recorded on `engine`, which the half's next stream waits for,
    /// and the layer's stream closes (its ring view is gone); before it the
    /// stream stays open for the units still to read, and the half stays
    /// held. The exclusion set stands for the units' serves either way.
    /// Nothing for a layer that streamed nothing.
    pub fn read(&mut self, layer: usize, engine: &CudaStream) -> Result<(), GpuError> {
        let Some(Some(b)) = self.cur.get(layer) else {
            return Ok(());
        };
        let half = b.half;
        if !self.groups[layer].read() {
            return Ok(());
        }
        self.cur[layer] = None;
        self.used[half].record(engine)?;
        Ok(())
    }

    /// End the call: a staging failure is refused by name, every layer's
    /// stream must have been read by every unit of its group; the call's
    /// report.
    pub fn end_call(&mut self) -> Result<XReport, GpuError> {
        const WHAT: &str = "XStream::end_call";
        let t0 = Instant::now();
        self.open = false;
        let unread = self.cur.iter().position(Option::is_some).map(|l| {
            let g = &self.groups[l];
            (l, g.reads, g.units.len())
        });
        for e in &mut self.excl {
            e.clear();
        }
        self.groups.fill(Group::default());
        self.refuse_failure(WHAT)?;
        if let Some((l, reads, units)) = unread {
            self.cur.iter_mut().for_each(|c| *c = None);
            let detail = match reads {
                0 => format!("layer {l}'s stream had no read before the call's end"),
                _ => format!(
                    "layer {l}'s stream was read by {reads} of its {units} units before the \
                     call's end"
                ),
            };
            return Err(GpuError::protocol(WHAT, detail));
        }
        let fresh = self.fresh_report();
        let mut r = std::mem::replace(&mut self.report, fresh);
        r.end_us = micros(t0);
        Ok(r)
    }

    /// A staging failure recorded by a fill thread, refused by name as
    /// `what`.
    fn refuse_failure(&self, what: &'static str) -> Result<(), GpuError> {
        let Lane::Pinned { shared, .. } = &self.lane else {
            return Ok(());
        };
        let f = shared
            .failed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match f.as_ref() {
            Some(e) => Err(GpuError::protocol(what, e.clone())),
            None => Ok(()),
        }
    }
}

impl Drop for XStream {
    fn drop(&mut self) {
        if let Lane::Pinned {
            shared,
            tx,
            threads,
            jobs,
        } = &mut self.lane
        {
            shared.stop.store(true, Ordering::Release);
            *tx = None;
            for t in threads.drain(..) {
                let _ = t.join();
            }
            // Every copy still waiting on a staging word goes through, the
            // ones whose job a stopped fill thread never published among
            // them: the bytes it copies land in a half no reader is left for.
            shared.release_waits(*jobs);
        }
        if let Err(e) = poll_drained(&self.copy, self.cfg.deadline) {
            let why = match e {
                Drain::Driver(e) => Some(e.to_string()),
                Drain::Exit(_) => None,
                Drain::Late(w) => Some(format!("still running after {w:?}")),
            };
            // The process's exit takes the ring and the staging with it.
            if let Some(why) = why {
                eprintln!(
                    "XStream drop: the copy stream did not drain ({why}); the ring and the \
                     staging are leaked"
                );
            }
            std::mem::forget(std::mem::take(&mut self.ring.parts));
            if let Lane::Pinned { shared, .. } = &self.lane {
                std::mem::forget(Arc::clone(shared));
            }
        }
    }
}

/// The rule's set over a layer's host experts with nothing admitted: the
/// experts the rule sends to the card, ranked, and the columns the union
/// keeps. The ring is lit by the unit's width alone ([`xsplit::m_min`]);
/// the pick ahead of it has already moved the pool. Refused by name: the
/// rule's refusals, and constants whose union floor passes one expert's lane
/// and card cost (the rule's fixed branch, which this cut does not read).
pub fn stream_tail(
    counts: &[u32],
    host: &[u32],
    k: &Constants,
    out: &mut Split,
) -> Result<(), xsplit::SplitError> {
    // `split` checks every input and ranks the host experts; its pool of one
    // lights the ring and takes the hottest into `admit`, which the rule's
    // set takes back when the ring is lit and that expert is past the floor.
    xsplit::split(counts, host, 1, k, out)?;
    if k.host_us_fixed > k.expert_b as f64 / xsplit::beside_b_per_us(k) + k.card_us_fixed {
        out.admit.clear();
        out.stream.clear();
        out.host_columns = 0;
        return Err(xsplit::SplitError::Param {
            name: "host_us_fixed",
            range: "at most one expert's lane and card fixed cost",
        });
    }
    let sum: u64 = counts.iter().map(|&c| u64::from(c)).sum();
    let unit = sum / k.top_k as u64;
    let Some(&top) = out.admit.first() else {
        return Ok(());
    };
    out.admit.clear();
    let c = counts[top as usize];
    if unit >= xsplit::m_min(k) && f64::from(c) > xsplit::m_star(k) {
        out.stream.insert(0, top);
    } else {
        out.host_columns += u64::from(c);
    }
    Ok(())
}

/// What each side of a layer holds before its stream ([`balance_cut`]): the
/// pick's admits, whose copies share the lane ahead of the stream; the
/// union's cost of the experts the rule keeps; the card's cost of the
/// columns its stacks already serve, which its route runs after the copies
/// land.
#[derive(Clone, Copy, Debug)]
struct Sides {
    admitted: usize,
    kept_us: f64,
    card_us: f64,
}

/// How many of the ranked `stream` the layer takes before the card's side —
/// the lane's copies beside the union, then its route over its own columns
/// and the streamed ones — passes the union the host keeps: the two run side
/// by side, so past that point a streamed expert lengthens the layer. The
/// copies price at the beside-union rate the unit's constants carry.
fn balance_cut(stream: &[u32], counts: &[u32], k: &Constants, side: Sides) -> usize {
    let copy_us = k.expert_b as f64 / xsplit::beside_b_per_us(k);
    let host_of = |c: u32| (k.host_us_per_col * f64::from(c)).max(k.host_us_fixed);
    let Sides {
        admitted,
        kept_us,
        card_us,
    } = side;
    let mut union: f64 = kept_us
        + stream
            .iter()
            .map(|&id| host_of(counts[id as usize]))
            .sum::<f64>();
    let mut lane = admitted as f64 * copy_us + card_us;
    for (i, &id) in stream.iter().enumerate() {
        let c = counts[id as usize];
        let card = copy_us + k.card_us_fixed + k.card_us_per_col * f64::from(c);
        if lane + card > union - host_of(c) {
            return i;
        }
        lane += card;
        union -= host_of(c);
    }
    stream.len()
}

/// Whole microseconds since `t0`, saturated.
fn micros(t0: Instant) -> u64 {
    u64::try_from(t0.elapsed().as_micros()).unwrap_or(u64::MAX)
}

/// The card's free and total bytes now.
fn card_mem(ctx: &Arc<CudaContext>) -> Result<(u64, u64), GpuError> {
    ctx.bind_to_thread()?;
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: the context is current on this thread (bound above) and both
    // outputs are live locals the call writes.
    let rc = unsafe { sys::cuMemGetInfo_v2(&mut free, &mut total) };
    cu(rc, "cuMemGetInfo (xstream ring)")?;
    Ok((free as u64, total as u64))
}

/// [`super::drain_within`] for the lane's waits.
fn drain_within(
    stream: &CudaStream,
    deadline: Duration,
    what: &'static str,
) -> Result<(), GpuError> {
    super::drain_within(stream, deadline, what)
}

/// The lane over `source`: pinned staging of `cfg.staging_slots` slots of
/// `slot_bytes` (halved down to four slots while the pin is refused) with
/// `cfg.fill_threads` fill threads on the pool's first workers' SMT
/// siblings, else the source's pageable bytes.
fn start_lane(
    ctx: &Arc<CudaContext>,
    source: Arc<dyn SwapSource>,
    slot_bytes: usize,
    cfg: &XCfg,
) -> Result<Lane, GpuError> {
    let mut slots = cfg.staging_slots.max(1);
    let staging = loop {
        match MappedHost::new(ctx, slots * slot_bytes, "cuMemHostAlloc (xstream staging)") {
            Ok(page) => break Some(page),
            Err(_) if slots > 4 => slots /= 2,
            Err(_) => break None,
        }
    };
    let Some(staging) = staging else {
        return Ok(Lane::Pageable { source });
    };
    let shared = Arc::new(LaneShared {
        source,
        staging,
        slot_bytes,
        slots,
        words: MappedHost::new(
            ctx,
            2 * slots * WORD_STRIDE,
            "cuMemHostAlloc (xstream words)",
        )?,
        deadline: cfg.deadline,
        delay_us: AtomicU64::new(0),
        served: AtomicU64::new(0),
        stop: AtomicBool::new(false),
        failed: Mutex::new(None),
    });
    let (tx, rx) = mpsc::channel::<Job>();
    let rx = Arc::new(Mutex::new(rx));
    let mut threads = Vec::with_capacity(cfg.fill_threads);
    for t in 0..cfg.fill_threads.max(1) {
        let place = match threads::built().and_then(|p| p.worker_cpu(t)) {
            Some(c) => threads::helper::Placement::Sibling(c),
            None => threads::helper::Placement::Float,
        };
        let (for_thread, jobs) = (Arc::clone(&shared), Arc::clone(&rx));
        let spawned = threads::helper::spawn_helper("xstream-fill", place, move || {
            for_thread.run(&jobs);
        });
        match spawned {
            Ok((h, _)) => threads.push(h),
            Err(e) => {
                shared.stop.store(true, Ordering::Release);
                drop(tx);
                for h in threads {
                    let _ = h.join();
                }
                return Err(GpuError::plan("XStream::new: a fill thread", e));
            }
        }
    }
    Ok(Lane::Pinned {
        shared,
        tx: Some(tx),
        threads,
        jobs: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ReusedSource {
        bytes: MappedHost,
        fail_part: std::sync::atomic::AtomicUsize,
    }

    impl SwapSource for ReusedSource {
        fn part_bytes(&self, _layer: usize) -> &[usize] {
            &[4; PARTS]
        }
        fn source(
            &self,
            _layer: usize,
            _id: u32,
            part: usize,
        ) -> Result<super::super::swap::Piece<'_>, GpuError> {
            if self.fail_part.load(Ordering::Acquire) == part {
                return Err(GpuError::state("test source", "part refused"));
            }
            // SAFETY: the pinned allocation holds four initialized bytes;
            // the source's read protects them until release_read.
            let bytes = unsafe { std::slice::from_raw_parts(self.bytes.host_at(0), 4) };
            Ok(super::super::swap::Piece {
                bytes,
                transform: Transform::Identity,
            })
        }
        fn release_read(&self, _layer: usize, _id: u32) -> Result<(), GpuError> {
            self.bytes
                .atomic_u32(0)
                .unwrap()
                .store(0x22222222, Ordering::Release);
            Ok(())
        }
        fn dest(
            &self,
            _layer: usize,
            _part: usize,
            _slot: u32,
        ) -> Result<sys::CUdeviceptr, GpuError> {
            Err(GpuError::state(
                "test source",
                "the ring owns its destination",
            ))
        }
        fn prepare_victim(&self, _layer: usize, _id: u32) -> Result<(), GpuError> {
            Ok(())
        }
        fn host_resident(&self, _layer: usize, _id: u32) -> Result<bool, GpuError> {
            Ok(true)
        }
        fn release_host(&self, _layer: usize, _id: u32) -> Result<u64, GpuError> {
            Ok(0)
        }
    }

    #[test]
    #[ignore = "needs a CUDA device; just gate-gpu-lib runs it"]
    fn hw_held_a_fallback_copy_keeps_its_source_until_it_lands() {
        use crate::checkpoint::hold::Hold;
        let (ctx, stream) = crate::capsync::fresh_stream(0).unwrap();
        let source = Arc::new(ReusedSource {
            bytes: MappedHost::new(&ctx, 4, "test source").unwrap(),
            fail_part: std::sync::atomic::AtomicUsize::new(usize::MAX),
        });
        let mut cfg = XCfg::new(
            Costs {
                host_us_per_col: 1.0,
                host_us_fixed: 1.0,
                card_us_fixed: 1.0,
                card_us_per_col: 1.0,
            },
            1,
            1,
            Duration::from_secs(3),
            None,
        );
        cfg.keep_free = 0;
        cfg.max_half = 1;
        cfg.min_half = 1;
        cfg.staging_slots = 1;
        cfg.fill_threads = 1;
        cfg.probe.push((0, 0));
        let mut x = XStream::new(&ctx, &stream, source.clone(), 1, cfg).unwrap();
        let lane = std::mem::replace(
            &mut x.lane,
            Lane::Pageable {
                source: source.clone(),
            },
        );
        if let Lane::Pinned { tx, threads, .. } = lane {
            drop(tx);
            for thread in threads {
                thread.join().unwrap();
            }
        }
        for fail_part in [usize::MAX, 1] {
            source.fail_part.store(fail_part, Ordering::Release);
            source
                .bytes
                .atomic_u32(0)
                .unwrap()
                .store(0x11111111, Ordering::Release);
            let hold = Hold::on(&x.copy).unwrap();
            let opener = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(100));
                drop(hold);
            });
            let result = x.copy_expert(0, 0, 0, 0);
            opener.join().unwrap();
            drain_within(&x.copy, x.cfg.deadline, "test copy").unwrap();
            assert_eq!(result.is_err(), fail_part != usize::MAX);
            let got = x.ring.parts[0].to_host_vec(&x.copy).unwrap();
            assert_eq!(
                got[0], 0x11111111,
                "the destination keeps the source's original bytes"
            );
            assert_eq!(
                source.bytes.atomic_u32(0).unwrap().load(Ordering::Acquire),
                0x22222222
            );
        }
    }

    /// The A6000 shape of the rule's own test, at the recalibrated family
    /// row and its calibration share: beside 13116.6, m* = 70.429,
    /// m_min 3606.
    fn a6000() -> Constants {
        Constants {
            lane_b_per_us: 21190.0,
            host_us_per_col: 3.67,
            host_us_fixed: 19.95,
            card_us_fixed: 13.0,
            card_us_per_col: 0.16,
            union_burst_share: 0.381,
            card_tail_us: 0.0,
            expert_b: 3072000,
            experts: 512,
            top_k: 10,
        }
    }

    /// With nothing admitted the tail is the rule's whole set: every host
    /// expert past the floor at a unit past m_min, ranked count descending,
    /// id ascending, the rest's columns kept (mutants: the hottest left in
    /// `admit`, the unit gate dropped, the floor dropped from the hottest's
    /// way back).
    ///
    /// PIN(2026-10-08): the streamed set (400..480) and the kept columns
    /// 32 x 20 = 640 held at m* 23.1; the rule now prices its copy beside the
    /// union, m* 70.429 (the constants' derivation), so the 40 experts at 60
    /// columns (440..480) sit under it and stay on the union: the streamed
    /// set is (400..440), the 40 at 80, and the kept columns are
    /// 40 x 60 + 32 x 20 = 3040.
    #[test]
    fn the_tail_is_the_rules_whole_set_with_nothing_admitted() {
        let k = a6000();
        let mut counts = [0u32; 512];
        // 4096 columns of ten picks, 40960: the host experts 400..512 get
        // 80, 60 and 20, the card's the rest.
        counts[..320].fill(87);
        counts[320..400].fill(86);
        counts[400..440].fill(80);
        counts[440..480].fill(60);
        counts[480..512].fill(20);
        assert_eq!(counts.iter().sum::<u32>(), 40960);
        let host: Vec<u32> = (400..512).rev().collect();
        let mut out = Split::default();
        stream_tail(&counts, &host, &k, &mut out).unwrap();
        assert!(out.admit.is_empty());
        let want: Vec<u32> = (400..440).collect();
        assert_eq!(out.stream, want);
        assert_eq!(out.host_columns, 40 * 60 + 32 * 20);
        // A 512-column unit is under m_min: nothing streams.
        let small = [10u32; 512];
        stream_tail(&small, &host, &k, &mut out).unwrap();
        assert!(out.stream.is_empty() && out.admit.is_empty());
        assert_eq!(out.host_columns, 112 * 10);
        // A unit past m_min whose hottest host expert is under the floor
        // streams nothing: the pool's one admit goes back to the union.
        let mut cold = [0u32; 512];
        cold[..320].fill(97);
        cold[320..400].fill(96);
        cold[400..512].fill(20);
        assert_eq!(cold.iter().sum::<u32>(), 40960);
        stream_tail(&cold, &host, &k, &mut out).unwrap();
        assert!(out.stream.is_empty() && out.admit.is_empty());
        assert_eq!(out.host_columns, 112 * 20);
    }

    /// The balance stops the stream where the layer's copies and card
    /// columns would pass the union it keeps, the pick's admits and the
    /// stacks' own columns on the card's side first and the rule's kept
    /// experts on the union — every copy at the beside-union rate the unit's
    /// constants carry (mutants: the admits left off the lane, the stacks'
    /// columns dropped, the kept union dropped, the cut one late, the copies
    /// at the lane's own rate).
    ///
    /// PIN(2026-10-08): the cuts 76, 56, 90, 69 and 7 held at the copy priced
    /// at the lane's own rate (145.07); the copy is now beside the union
    /// (234.2), so each cut re-derives from the comment's own arithmetic
    /// below: 53, 10, 71, 44 and 5.
    #[test]
    fn the_balance_stops_where_the_lane_passes_the_union() {
        let k = a6000();
        let counts: Vec<u32> = vec![80; 512];
        let stream: Vec<u32> = (0..100).collect();
        // A streamed expert costs the card's side its copy beside the union
        // (234.2), its fixed and its columns' cost — 260.0 with n streamed —
        // and saves the union max(19.95, 3.67 x 80) = 293.6 a one: against
        // 293.6 (100 - n), 53 stream and the 54th would pass.
        let side = |admitted, kept_us, card_us| Sides {
            admitted,
            kept_us,
            card_us,
        };
        assert_eq!(balance_cut(&stream, &counts, &k, side(0, 0.0, 0.0)), 53);
        // 100 admits ahead on the lane take 23,420.7 of it first: with n
        // streamed, 23,420.7 + 260.0 n against 293.6 (100 - n), so 10.
        assert_eq!(balance_cut(&stream, &counts, &k, side(100, 0.0, 0.0)), 10);
        // The union the rule keeps anyway counts on the host's side: 10,000
        // of it adds to it, 260.0 n against 10,000 + 293.6 (100 - n), so 71.
        assert_eq!(
            balance_cut(&stream, &counts, &k, side(0, 10_000.0, 0.0)),
            71
        );
        // The stacks' own columns count on the card's side: 5,000 of them,
        // 5,000 + 260.0 n against 293.6 (100 - n), so 44.
        assert_eq!(balance_cut(&stream, &counts, &k, side(0, 0.0, 5_000.0)), 44);
        // Ten candidates: 260.0 n against 293.6 (10 - n), so 5.
        assert_eq!(
            balance_cut(&stream[..10], &counts, &k, side(0, 0.0, 0.0)),
            5
        );
    }

    /// Unit `set` of layer 3 in a block of four-row units.
    fn unit_key(set: usize) -> BatchKey {
        BatchKey {
            layer: 3,
            set,
            at: 4 * set,
            u: 4 * set + 4,
        }
    }

    /// A group takes the units that join it and closes at the last unit's
    /// read: the first of two reads leaves the stream open, the second ends
    /// it, and a unit outside the group is not in it (mutants: a join that
    /// adds no unit, a read that closes at the first).
    #[test]
    fn a_group_closes_at_its_last_units_read() {
        let mut g = Group::of(unit_key(0));
        assert!(g.holds(unit_key(0)) && !g.holds(unit_key(1)));
        g.join(unit_key(1)).expect("a second unit joins");
        assert!(g.holds(unit_key(0)) && g.holds(unit_key(1)));
        assert!(!g.holds(unit_key(2)));
        assert!(!g.read(), "one of two units has read");
        assert!(g.read(), "the group's last unit's read closes the stream");
    }

    /// The lane's rate is the probe's bytes over the time their copies take
    /// at each expert's source rate: a held probe prices at the held rate, a
    /// paged one at the paged rate, a mixed one between them weighted by
    /// bytes, and no bytes at 0 (mutants: a paged expert priced at the held
    /// rate; the mix weighted by count, not bytes).
    #[test]
    fn the_lane_prices_each_copy_at_its_sources_rate() {
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b;
        let b = 3_072_000u64;
        assert!(close(lane_rate([(b, false); 3]), LANE_HELD_B_PER_US));
        assert!(close(lane_rate([(b, true); 3]), LANE_PAGED_B_PER_US));
        let us = 3.0 * b as f64 / LANE_PAGED_B_PER_US + b as f64 / LANE_HELD_B_PER_US;
        assert!(close(
            lane_rate([(3 * b, true), (b, false)]),
            4.0 * b as f64 / us
        ));
        assert_eq!(lane_rate([]).to_bits(), 0.0f64.to_bits());
    }

    /// A unit alone is today's stream: its one read closes it.
    #[test]
    fn a_group_of_one_closes_at_its_read() {
        let mut g = Group::of(unit_key(0));
        assert!(g.read());
    }

    /// A join is refused by name when no unit streams the layer, when a card
    /// route has read the stream already (the size of the group would grow
    /// under a read that already closed it), and for a unit already in it.
    #[test]
    fn a_join_is_refused_by_its_reason() {
        let mut none = Group::default();
        assert_eq!(
            none.join(unit_key(1)),
            Err("no unit streams it in this call")
        );
        let mut g = Group::of(unit_key(0));
        assert_eq!(
            g.join(unit_key(0)),
            Err("that unit is in its group already")
        );
        g.read();
        assert_eq!(
            g.join(unit_key(1)),
            Err("a card route has read its stream already")
        );
        assert!(!g.holds(unit_key(1)));
    }
}
