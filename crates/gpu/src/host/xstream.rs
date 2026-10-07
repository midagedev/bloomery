//! The expert stream: the one owner of a prompt unit's transient expert
//! copies, which any MoE family calls with its counts and its card map and
//! no stream code of its own.
//!
//! **The split.** Per layer of a prompt unit, after the residency machine's
//! pick has admitted its share into the pool ([`super::swap`]'s call mode),
//! the host experts the unit still routes split in two: the ones the rule
//! sends over the copy lane to the card for this unit alone (the stream),
//! and the ones the host union keeps. The rule is [`runtime::xsplit`]'s,
//! over constants the load measured (the lane's rate, [`XStream::new`]'s
//! probe) and the costs the model carries for its expert shape
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
//! halves of [`XStream::half_slots`] slots; per part (gate, up, down) a slot
//! holds the widest layer's part. A layer's stream lands in one half, its
//! experts packed at their layer's part bytes from the half's first slot,
//! so a grouped GEMM reads the half as a stack of `half_slots` experts. The
//! halves alternate per streamed layer: a layer's copies wait on the copy
//! stream for the read event of the layer two streams back
//! ([`XStream::read`]), the last reader of that half.
//!
//! **The maps.** Per half two rows of the router's experts, in mapped host
//! memory the card reads in place: the ring row (a streamed expert's slot
//! in the half, [`HOST`] for every other) and the union row (the stage map's
//! card places, `n_card + i` for the `i`-th streamed expert, [`HOST`] for
//! the rest), which the card sum reads with `n_card + n` card places. A
//! half's rows are written when its layer streams, after the host has seen
//! the download of a layer past the half's last reader.
//!
//! **The lane.** Fill threads copy each streamed expert's source bytes into
//! a pinned staging ring ([`XCfg::staging_slots`] slots); the copy stream
//! waits for each slot's staged word, copies the parts into the ring and
//! raises the slot's drained word, which the fill thread that takes the slot
//! next waits for — the residency machine's word protocol: a fill thread
//! waits on nothing the card runs after the copy that waits for it, so one
//! hardware queue (WDDM, `CUDA_DEVICE_MAX_CONNECTIONS=1`) only serializes
//! the streams. A layer's last copy records its landed event, which the
//! engine stream waits for before the layer's card route. When the pinned
//! staging cannot be allocated the lane copies from the source's pageable
//! bytes on the copy stream, the engine thread enqueueing each copy.
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
use runtime::xsplit::{self, Constants, Split};

use super::batch::BatchKey;
use super::slots::HOST;
use super::swap::{SwapSource, Transform};
use super::{Drain, poll_drained};
use crate::GpuError;
use crate::graph::{MappedHost, cu, mem_batch, op_wait_geq, op_write};

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
/// measured rate: the host union's cost of one expert per column and its
/// floor, and the card's cost of one streamed expert, fixed and per column,
/// each in µs.
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
    /// The fewest slots a half may have; a card with less room is refused
    /// by name.
    pub min_half: usize,
    /// Fill threads and pinned staging slots.
    pub fill_threads: usize,
    pub staging_slots: usize,
    /// The bound on every host wait.
    pub deadline: Duration,
    /// Host experts the load's lane probe streams, `(layer, id)` each.
    pub probe: Vec<(usize, u32)>,
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
    /// lane's measured rate [B/µs].
    pub half_slots: usize,
    pub staging_slots: usize,
    pub lane_b_per_us: f64,
    /// Host microseconds the end took.
    pub end_us: u64,
}

/// The card's view of a layer's stream, for its card route
/// ([`XStream::ring_layer`]).
#[derive(Clone, Copy, Debug)]
pub struct RingLayer {
    /// The experts the half holds, slots `0..n`.
    pub n: usize,
    /// The half's slots: the experts a route over it counts.
    pub half_slots: usize,
    /// The ring row and the union row ([`XStream`]'s module doc), each the
    /// router's experts of `u32`.
    pub ring_map: sys::CUdeviceptr,
    pub union_map: sys::CUdeviceptr,
    /// Per part (gate, up, down) the half's first slot.
    pub parts: [sys::CUdeviceptr; PARTS],
}

/// An expert's parts: gate, up, down.
pub const PARTS: usize = 3;

/// The card bytes a ring leaves free when it sizes itself: half the margin
/// the placement's expert rule leaves (`placement::workstation::MARGIN`),
/// for what a load allocates past the plan after the ring.
pub const KEEP_FREE: u64 = 512 << 20;

/// The fewest slots a half may hold: fewer, and the ring's bytes would not
/// pay for its launches.
pub const MIN_HALF: usize = 8;

/// Fill threads and pinned staging slots: four threads keep the copy
/// engine fed from the source's pages, and a staging ring of 64 slots holds
/// a layer's stream ahead of the copy stream.
pub const FILL_THREADS: usize = 4;
pub const STAGING_SLOTS: usize = 64;

/// The experts the load's lane probe streams.
pub const PROBE: usize = 32;

/// What the refusal of a ring the card has no room for names: a caller that
/// resolves an unset lever tells it from every other refusal by it.
pub const XSTREAM_ROOM: &str = "XStream ring room";

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
    /// after part, once the slot's last copy has drained.
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
        let mut at = job.slot * self.slot_bytes;
        let end = at + self.slot_bytes;
        for (part, &want) in self.source.part_bytes(job.layer).iter().enumerate() {
            let piece = self.source.source(job.layer, job.id, part)?;
            if piece.bytes.len() != want || at + want > end {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {} expert {} part {part}: {} source bytes for a part of {want} in \
                         a staging slot of {}",
                        job.layer,
                        job.id,
                        piece.bytes.len(),
                        self.slot_bytes
                    ),
                ));
            }
            match piece.transform {
                // SAFETY: [at, at + want) lies inside the staging page (the
                // slot's span, checked above); no copy reads the slot until
                // its staged word publishes this use, and this thread holds
                // the slot's only job until then. The word's Release store
                // follows these stores in program order.
                Transform::Identity => unsafe {
                    std::ptr::copy_nonoverlapping(
                        piece.bytes.as_ptr(),
                        self.staging.host_at(at),
                        want,
                    );
                },
            }
            at += want;
        }
        Ok(true)
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
    widest: [usize; PARTS],
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
    /// Per half: the copy stream's landed event of its last stream, and the
    /// engine stream's event after that stream's last reader.
    landed: Vec<CudaEvent>,
    used: Vec<CudaEvent>,
    /// Streams issued since the load: the next one's half.
    issued: u64,
    /// Per layer the stream of the unit's walk, until its read.
    cur: Vec<Option<Batch>>,
    /// Per layer the streamed ids of the layer's last unit in the open call,
    /// ascending, and that unit's key: the host serve's exclusion set, for
    /// that unit's serve alone. The serve of a unit's layer runs after its
    /// card route's read (`runtime::sched`'s batch order), so the set stands
    /// past the read; a unit the walk takes no stream for (one narrower than
    /// the pick's floor) has another key and reads none.
    excl: Vec<Vec<u32>>,
    excl_unit: Vec<Option<BatchKey>>,
    /// A call is open ([`XStream::begin_call`] to [`XStream::end_call`]).
    open: bool,
    /// The rule's scratch.
    split: Split,
    rank: Vec<u32>,
    lane_b_per_us: f64,
    report: XReport,
    /// Per layer the rule's constants: `None` for a layer the source holds
    /// no stage stack of.
    consts: Vec<Option<Constants>>,
}

impl XStream {
    /// The stream over `source`'s stage stacks for the map's `layers`
    /// layers: the ring sized from the card's free bytes past
    /// `cfg.keep_free` (at most `cfg.max_half` slots a half), the staging
    /// (pinned, else the source's pageable bytes), the fill threads, then the
    /// lane's probe — `cfg.probe`'s experts streamed into the first half and
    /// timed — whose rate is the rule's. Refused by name: a ring of fewer
    /// than `cfg.min_half` slots a half, no layer with stage stacks, a part
    /// not a whole number of words, more parts than [`PARTS`], an empty
    /// probe. Load-time only.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        source: Arc<dyn SwapSource>,
        layers: usize,
        cfg: XCfg,
    ) -> Result<XStream, GpuError> {
        const WHAT: &str = "XStream::new";
        let mut widest = [0usize; PARTS];
        let mut slot_bytes = 0usize;
        for l in 0..layers {
            let parts = source.part_bytes(l);
            if parts.is_empty() {
                continue;
            }
            if parts.len() != PARTS || parts.iter().any(|&b| b == 0 || b % 4 != 0) {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l}: expert parts of {parts:?} bytes; the stream holds {PARTS}, each \
                         a nonzero multiple of 4"
                    ),
                ));
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
        let widest = widest.map(|b| b.next_multiple_of(16));
        let per_slot: usize = widest.iter().sum();
        let (free, _) = card_mem(ctx)?;
        let room = free.saturating_sub(cfg.keep_free);
        let fit = usize::try_from(room / (HALVES * per_slot) as u64).unwrap_or(usize::MAX);
        let half = fit.min(cfg.max_half);
        if half < cfg.min_half {
            return Err(GpuError::shape(
                XSTREAM_ROOM,
                format!(
                    "the card holds {free} B free, {room} B past the {} B the ring leaves free: \
                     {fit} slots a half of {per_slot} B, fewer than the {} a stream needs",
                    cfg.keep_free, cfg.min_half
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
        let (landed, used) = (new_events(HALVES)?, new_events(HALVES)?);
        let lane = start_lane(ctx, source, slot_bytes, &cfg)?;
        let mut x = XStream {
            consts: Vec::new(),
            cfg,
            copy,
            ring,
            rows,
            lane,
            landed,
            used,
            issued: 0,
            cur: vec![None; layers],
            excl: vec![Vec::new(); layers],
            excl_unit: vec![None; layers],
            open: false,
            split: Split::default(),
            rank: Vec::new(),
            lane_b_per_us: 0.0,
            report: XReport::default(),
        };
        x.lane_b_per_us = x.probe()?;
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
        let probe = self.cfg.probe.clone();
        let n = probe.len().min(self.ring.half);
        let t0 = Instant::now();
        let mut bytes = 0u64;
        for (i, &(layer, id)) in probe[..n].iter().enumerate() {
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

    /// The rule's constants of layer `layer` at the lane's rate.
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
            lane_b_per_us: self.lane_b_per_us,
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

    /// The lane's rate the load measured [B/µs].
    #[must_use]
    pub fn lane_b_per_us(&self) -> f64 {
        self.lane_b_per_us
    }

    /// Layer `layer`'s constants of the rule; `None` for a layer with no
    /// stage stack.
    #[must_use]
    pub fn constants(&self, layer: usize) -> Option<Constants> {
        self.consts.get(layer).copied().flatten()
    }

    /// Set the rule's costs (the family's, or a calibration's): every layer's
    /// constants follow at the lane's measured rate. Refused by name while a
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
        self.excl_unit.fill(None);
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
    /// the half's last reader, the landed event recorded, and the engine
    /// stream `engine` made to wait for it. A layer that streams nothing
    /// issues nothing. Refused by name: a layer with no stage stack, a
    /// layer streaming twice before its read, a staging failure, and the
    /// rule's own refusals.
    #[allow(
        clippy::too_many_arguments,
        reason = "the unit, its width and counts, the host set, the pick, the stage row and the engine stream (rust-quality R8)"
    )]
    pub fn layer(
        &mut self,
        engine: &CudaStream,
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
        let k = self
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
        self.excl_unit[layer] = Some(unit);
        self.excl[layer].clear();
        stream_tail(counts, host, &k, &mut self.split)
            .map_err(|e| GpuError::shape(WHAT, format!("layer {layer}: {e}")))?;
        let tail = self.split.stream.len();
        let host_routed = host.iter().filter(|&&id| counts[id as usize] > 0).count();
        let kept_us = k.host_us_per_col * self.split.host_columns as f64;
        let all: u64 = counts.iter().map(|&c| u64::from(c)).sum();
        let host_cols: u64 = host.iter().map(|&id| u64::from(counts[id as usize])).sum();
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
            self.write_rows(half, stage, n_card)?;
            let (bytes, backlog) = self.issue(engine, layer, half)?;
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
        // reads of this half were the card route of the stream two back,
        // which the engine stream ran before the download of a later layer
        // the host has already waited for (the caller's count), so nothing
        // reads these words while they are written.
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

    /// Issue the ranked stream in `self.rank` of layer `layer` into half
    /// `half`: the copy stream waits for the half's last reader, each
    /// expert's copies follow, the half's landed event after them, and the
    /// engine stream waits for it. The bytes copied and the host
    /// microseconds spent waiting for the staging backlog.
    fn issue(
        &mut self,
        engine: &CudaStream,
        layer: usize,
        half: usize,
    ) -> Result<(u64, u64), GpuError> {
        self.copy.wait(&self.used[half])?;
        let rank = std::mem::take(&mut self.rank);
        let mut bytes = 0u64;
        let mut backlog = 0u64;
        let r = (|| {
            for (i, &id) in rank.iter().enumerate() {
                let t = Instant::now();
                self.wait_backlog()?;
                backlog += micros(t);
                bytes += self.copy_expert(layer, id, half, i)?;
            }
            Ok::<(), GpuError>(())
        })();
        self.rank = rank;
        r?;
        self.landed[half].record(&self.copy)?;
        engine.wait(&self.landed[half])?;
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
        if parts.len() != PARTS || i >= self.ring.half {
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
                mem_batch(
                    &copy,
                    &mut [op_wait_geq(staged, ticket)],
                    "xstream: wait for staging",
                )?;
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
                mem_batch(
                    &copy,
                    &mut [op_write(drained, ticket)],
                    "xstream: staging drained",
                )?;
            }
            Lane::Pageable { .. } => {
                for (p, (&len, &d)) in parts.iter().zip(&dst).enumerate() {
                    let piece = source.source(layer, id, p)?;
                    if piece.bytes.len() != len || piece.transform != Transform::Identity {
                        return Err(GpuError::shape(
                            WHAT,
                            format!(
                                "layer {layer} expert {id} part {p}: {} source bytes ({:?}) for a \
                                 part of {len}",
                                piece.bytes.len(),
                                piece.transform
                            ),
                        ));
                    }
                    // SAFETY: the source's bytes stay mapped for the
                    // source's life, which outlives the copy stream's drain
                    // (the drop drains it); `d` is a ring slot's part, `len`
                    // bytes inside the ring.
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
        let mut at = [0; PARTS];
        for (p, a) in at.iter_mut().enumerate() {
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
        })
    }

    /// The streamed experts of `unit`'s layer, ascending: the host serve of
    /// that unit leaves them out. Empty for a unit the layer's last
    /// [`XStream::layer`] did not take, when that unit streams nothing, and
    /// outside a call.
    #[must_use]
    pub fn excluded(&self, unit: BatchKey) -> &[u32] {
        if !self.open || self.excl_unit.get(unit.layer) != Some(&Some(unit)) {
            return &[];
        }
        self.excl.get(unit.layer).map_or(&[], Vec::as_slice)
    }

    /// Layer `layer`'s stream has had its last read on the engine stream
    /// `engine` (its card route): the half's read event is recorded there,
    /// which the half's next stream waits for, and the layer's stream
    /// closes; its exclusion set stands for the unit's serve. Nothing for a
    /// layer that streamed nothing.
    pub fn read(&mut self, layer: usize, engine: &CudaStream) -> Result<(), GpuError> {
        let Some(b) = self.cur.get_mut(layer).and_then(Option::take) else {
            return Ok(());
        };
        self.used[b.half].record(engine)?;
        Ok(())
    }

    /// End the call: a staging failure is refused by name, every layer's
    /// stream must have been read; the call's report.
    pub fn end_call(&mut self) -> Result<XReport, GpuError> {
        const WHAT: &str = "XStream::end_call";
        let t0 = Instant::now();
        self.open = false;
        for e in &mut self.excl {
            e.clear();
        }
        self.excl_unit.fill(None);
        self.refuse_failure(WHAT)?;
        if let Some(l) = self.cur.iter().position(Option::is_some) {
            self.cur.iter_mut().for_each(|c| *c = None);
            return Err(GpuError::protocol(
                WHAT,
                format!("layer {l}'s stream had no read before the call's end"),
            ));
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
            ..
        } = &mut self.lane
        {
            shared.stop.store(true, Ordering::Release);
            *tx = None;
            for t in threads.drain(..) {
                let _ = t.join();
            }
            // Every copy still waiting on a staging word goes through: the
            // bytes it copies land in a half no reader is left for.
            for k in 0..shared.slots {
                shared.staged(k).store(u32::MAX, Ordering::Release);
            }
        }
        if let Err(e) = poll_drained(&self.copy, self.cfg.deadline) {
            let why = match e {
                Drain::Driver(e) => e.to_string(),
                Drain::Late(w) => format!("still running after {w:?}"),
            };
            eprintln!(
                "XStream drop: the copy stream did not drain ({why}); the ring and the staging \
                 are leaked"
            );
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
    if k.host_us_fixed > k.expert_b as f64 / k.lane_b_per_us + k.card_us_fixed {
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
/// the lane's copies, then its route over its own columns and the streamed
/// ones — passes the union the host keeps: the two run side by side, so past
/// that point a streamed expert lengthens the layer.
fn balance_cut(stream: &[u32], counts: &[u32], k: &Constants, side: Sides) -> usize {
    let copy_us = k.expert_b as f64 / k.lane_b_per_us;
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

    /// The A6000 shape of the rule's own test: m* = 23.096, m_min 1183.
    fn a6000() -> Constants {
        Constants {
            lane_b_per_us: 21190.0,
            host_us_per_col: 7.0,
            host_us_fixed: 23.36,
            card_us_fixed: 13.0,
            card_us_per_col: 0.16,
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
        let want: Vec<u32> = (400..480).collect();
        assert_eq!(out.stream, want);
        assert_eq!(out.host_columns, 32 * 20);
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
    /// experts on the union (mutants: the admits left off the lane, the
    /// stacks' columns dropped, the kept union dropped, the cut one late).
    #[test]
    fn the_balance_stops_where_the_lane_passes_the_union() {
        let k = a6000();
        let counts: Vec<u32> = vec![80; 512];
        let stream: Vec<u32> = (0..100).collect();
        // A streamed expert costs the card's side its copy, its fixed and its
        // columns' cost, 170.77 n with n streamed, and saves the union 560 a
        // one: against 560 (100 - n), 76 stream and the 77th would pass.
        let side = |admitted, kept_us, card_us| Sides {
            admitted,
            kept_us,
            card_us,
        };
        assert_eq!(balance_cut(&stream, &counts, &k, side(0, 0.0, 0.0)), 76);
        // 100 admits ahead on the lane take 14,497 of it first: with n
        // streamed, 14,497 + 170.77 n against 560 (100 - n), so 56 stream.
        assert_eq!(balance_cut(&stream, &counts, &k, side(100, 0.0, 0.0)), 56);
        // The union the rule keeps anyway counts on the host's side: 10,000
        // of it, 170.77 n against 10,000 + 560 (100 - n), so 90.
        assert_eq!(
            balance_cut(&stream, &counts, &k, side(0, 10_000.0, 0.0)),
            90
        );
        // The stacks' own columns count on the card's side: 5,000 of them,
        // 5,000 + 170.77 n against 560 (100 - n), so 69.
        assert_eq!(balance_cut(&stream, &counts, &k, side(0, 0.0, 5_000.0)), 69);
        // Ten candidates: 170.77 n against 560 (10 - n), so 7.
        assert_eq!(
            balance_cut(&stream[..10], &counts, &k, side(0, 0.0, 0.0)),
            7
        );
    }
}
