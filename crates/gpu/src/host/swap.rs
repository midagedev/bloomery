//! Adaptive expert residency, the machine: which routed experts of each layer
//! sit in the stage card's stacks, moved between passes by the residency rule
//! ([`runtime::swaprule`]) without a recapture.
//!
//! A captured chain reads the stage card's copy of the slot map at replay
//! (the handoff's `map[row_off + id]`) and addresses an expert's weights by
//! its slot in one flat stack a layer, so moving an expert on or off the card
//! is two things: its bytes into a free slot of each stack, and two words of
//! the card's copy. The machine owns the ordering of both.
//!
//! **Slots.** A layer's card slots are its load's `n_l`, the map's capacity
//! ([`SlotMap::capacity`]). At [`SwapMachine::new`] the seed's last `spares`
//! slots are freed (their experts go to the host), so `n_l − spares` are live
//! and each layer has `spares` free slots; the first `pinned` seed experts
//! are never a victim. A slot is in one state of its layer's [`SlotLedger`]:
//! `Live(e)`, `Spare`, or `Filling` — being written for a flip made at an
//! earlier boundary. No map entry names a `Filling` slot, so no kernel reads
//! it and neither device view shows it.
//!
//! **Flips.** A flip made at boundary `b` admits an expert into a spare slot
//! while its victim stays live; both change at boundary `b + delay`. At `b`
//! the machine enqueues on its copy stream, behind the event of boundary `b`
//! (the slot's last reader has run), a wait for the staging thread's word,
//! the expert's parts from the staging ring into the slot's place in each
//! stack, a word back to the staging thread, and the flip's event; the
//! staging thread prepares the victim for the host
//! ([`SwapSource::prepare_victim`]), then copies the source bytes into the
//! ring, gated to the host leg's wait window ([`SwapMachine::window`]).
//!
//! **Boundaries.** [`SwapMachine::boundary`] runs before each pass's launch,
//! in this order: the engine stream waits for the copies of the flips live
//! at this boundary; the changed words of the card's copy are written on the
//! engine stream; the host [`SlotMap`] takes the same change; this
//! boundary's event is recorded; then the rule plans and the new flips'
//! copies are issued. A flip lands at `b + delay` whatever its copy's
//! progress: a late copy makes the engine stream wait, and the boundary never
//! moves, so the same history of passes gives the same map at every pass.
//! The boundary also waits, on the host, for the victim's preparation; a
//! flip whose victim the source then says the host cannot serve from
//! resident pages is refused by name there.
//!
//! **No wait without a bound.** A flip landing at a boundary makes its job
//! due: the staging thread stages a due job whatever the window says, so a
//! host that blocks on the engine stream (a synchronizing copy, a readback)
//! with the window closed still sees the stream drain. Every host wait the
//! machine makes — for a victim's preparation, for a ring slot's last copy,
//! for the copy stream at a reset — ends by [`MachineCfg::deadline`] with a
//! named error; only a job not yet due waits for the window as long as the
//! window stays closed.
//!
//! **Counting.** The ids a pass routes are noted per (layer, row) into a
//! [`Tally`] and folded into the rule at [`SwapMachine::end_pass`] for the
//! pass's kept rows only, so a rejected row leaves no trace.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer, sys};
use runtime::swaprule::{Flip, Shape, SwapParams, SwapRule};

use super::slots::{HOST, Slot, SlotMap};
use crate::GpuError;
use crate::graph::{MappedHost, cu, mem_batch, op_wait_geq, op_write};

/// The residency lever: `off`, or the rule's `mid` parameters with `P`
/// pinned seed experts a layer and `S` spare slots a layer (`mid-p<P>-s<S>`).
pub const RESIDENCY: &str = "BLOOMERY_RESIDENCY";

/// The values [`RESIDENCY`] takes.
pub const RESIDENCY_WORDS: [&str; 3] = ["off", "mid-p0-s1", "mid-p40-s1"];

/// What [`RESIDENCY`] asks a load for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    /// The load's slot map for the model's life: no machine.
    Off,
    /// The `mid` rule, `pinned` seed experts a layer never a victim, `spares`
    /// free slots a layer.
    Mid { pinned: usize, spares: usize },
}

impl Residency {
    /// `v` as a value of [`RESIDENCY`]; any other is refused by name.
    pub fn parse(v: &str) -> Result<Residency, GpuError> {
        match v {
            "off" => Ok(Residency::Off),
            "mid-p0-s1" => Ok(Residency::Mid {
                pinned: 0,
                spares: 1,
            }),
            "mid-p40-s1" => Ok(Residency::Mid {
                pinned: 40,
                spares: 1,
            }),
            _ => Err(GpuError::shape(
                "Residency::parse",
                format!(
                    "{RESIDENCY}={v:?}: it takes one of {}",
                    RESIDENCY_WORDS.join(", ")
                ),
            )),
        }
    }

    /// [`RESIDENCY`] from the process environment; unset is `off`, a value
    /// that is not UTF-8 is refused by name.
    pub fn from_env() -> Result<Residency, GpuError> {
        match std::env::var(RESIDENCY) {
            Ok(v) => Residency::parse(&v),
            Err(std::env::VarError::NotPresent) => Ok(Residency::Off),
            Err(std::env::VarError::NotUnicode(v)) => Err(GpuError::shape(
                "Residency::from_env",
                format!("{RESIDENCY}={v:?} is not UTF-8"),
            )),
        }
    }

    /// The rule's parameters at the model's live delay `delay` (passes from
    /// the boundary that makes a flip to the one it lands at), and the pinned
    /// count; `None` for `off`.
    #[must_use]
    pub fn params(self, delay: u64) -> Option<(SwapParams, usize)> {
        match self {
            Residency::Off => None,
            Residency::Mid { pinned, spares } => Some((
                SwapParams {
                    spares,
                    ..SwapParams::mid(delay)
                },
                pinned,
            )),
        }
    }
}

// ----------------------------------------------------------------- source

/// How the staging thread turns an expert's source bytes into its slot's
/// bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    /// The slot holds the source bytes as they are.
    Identity,
}

/// One part of an expert as its source holds it: `bytes`, turned into the
/// slot's by `transform`.
#[derive(Clone, Copy, Debug)]
pub struct Piece<'a> {
    pub bytes: &'a [u8],
    pub transform: Transform,
}

/// A model's side of the machine: where an expert's bytes come from, where
/// they go on the card, and whether the host can serve an expert the card
/// gives up. The staging thread calls [`SwapSource::source`] and
/// [`SwapSource::prepare_victim`]; every other call is on the thread that
/// drives the machine.
pub trait SwapSource: Send + Sync {
    /// Bytes one expert takes in each of a layer's stage stacks (its parts:
    /// gate, up, down, …), in stack order; every entry a multiple of 4.
    fn part_bytes(&self) -> &[usize];

    /// Part `part` of layer `layer`'s expert `id` as its source holds it:
    /// [`SwapSource::part_bytes`]`[part]` bytes once transformed.
    fn source(&self, layer: usize, id: u32, part: usize) -> Result<Piece<'_>, GpuError>;

    /// The device address slot `slot` of layer `layer`'s stack `part` starts
    /// at. The stacks stay allocated and in place for the machine's life.
    fn dest(&self, layer: usize, part: usize, slot: u32) -> Result<sys::CUdeviceptr, GpuError>;

    /// Bring layer `layer`'s expert `id` to where the host serves it from
    /// resident pages (read its pages in): called once a flip that evicts it
    /// is made, and at load for the experts the spare slots give up.
    fn prepare_victim(&self, layer: usize, id: u32) -> Result<(), GpuError>;

    /// Whether the host serves layer `layer`'s expert `id` from resident
    /// pages now: a flip whose victim it is not is refused by name.
    fn host_resident(&self, layer: usize, id: u32) -> Result<bool, GpuError>;

    /// Release the host pages of layer `layer`'s expert `id`, back on the
    /// card after a reset as at load; the bytes released.
    fn release_host(&self, layer: usize, id: u32) -> Result<u64, GpuError>;
}

// ------------------------------------------------------------------ tally

/// The ids a pass routes, per (row, layer), as the host sees them: filled
/// one slot at a time by [`Tally::note`] and folded into the rule at
/// [`SwapMachine::end_pass`]. `Tally::off()` notes nothing.
#[derive(Clone, Debug, Default)]
pub struct Tally {
    layers: Range<usize>,
    top_k: usize,
    max_rows: usize,
    /// `((row * layers + layer) * top_k + k)`.
    ids: Vec<u32>,
    /// Per (row, layer), the slots noted.
    filled: Vec<u32>,
}

impl Tally {
    /// A tally that notes nothing: the step port's without a machine.
    #[must_use]
    pub fn off() -> Tally {
        Tally::default()
    }

    fn new(layers: Range<usize>, top_k: usize, max_rows: usize) -> Tally {
        let cells = layers.len() * max_rows;
        Tally {
            layers,
            top_k,
            max_rows,
            ids: vec![0; cells * top_k],
            filled: vec![0; cells],
        }
    }

    /// Slot `k` of row `row`'s routing at layer `layer` is `id`. A tally that
    /// is off, a layer outside it, a row or a slot past its shape note
    /// nothing; [`SwapMachine::end_pass`] refuses a kept row whose layer
    /// holds another count than `top_k`.
    pub fn note(&mut self, layer: usize, row: usize, k: usize, id: u32) {
        let Some(l) = layer.checked_sub(self.layers.start) else {
            return;
        };
        if l >= self.layers.len() || row >= self.max_rows || k >= self.top_k {
            return;
        }
        let cell = row * self.layers.len() + l;
        self.ids[cell * self.top_k + k] = id;
        self.filled[cell] += 1;
    }

    /// Forget the noted ids: the next pass starts empty.
    pub fn clear(&mut self) {
        self.filled.fill(0);
    }
}

// ----------------------------------------------------------------- ledger

/// One card slot of one layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotState {
    /// Expert `e` is live in the slot: the map names it.
    Live(u32),
    /// Free: no map entry names it and nothing is being written into it.
    Spare,
    /// Being written for expert `e`, live from boundary `live_at` on; `event`
    /// is the machine's copy event for it. No map entry names it.
    Filling { e: u32, live_at: u64, event: usize },
}

/// Per layer, the state of each stage card slot below the layer's capacity:
/// the in-flight side of the slot map, which the map itself never shows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotLedger {
    layers: Range<usize>,
    /// Per layer, its slots.
    rows: Vec<Vec<SlotState>>,
}

impl SlotLedger {
    /// The ledger of `map`'s stage card: each live entry `Live`, every other
    /// slot below the row's capacity `Spare`.
    pub fn of_map(map: &SlotMap) -> Result<SlotLedger, GpuError> {
        let layers = map.layers();
        let mut rows = Vec::with_capacity(layers.len());
        for l in layers.clone() {
            let mut row = vec![SlotState::Spare; map.capacity(l)?];
            for id in 0..map.n_expert() as u32 {
                if let Some(Slot::Card(s)) = map.slot(l, id) {
                    row[s as usize] = SlotState::Live(id);
                }
            }
            rows.push(row);
        }
        Ok(SlotLedger { layers, rows })
    }

    /// Layer `layer`'s slots; empty for a layer outside the ledger.
    #[must_use]
    pub fn row(&self, layer: usize) -> &[SlotState] {
        layer
            .checked_sub(self.layers.start)
            .and_then(|i| self.rows.get(i))
            .map_or(&[], Vec::as_slice)
    }

    fn row_mut(&mut self, layer: usize) -> &mut Vec<SlotState> {
        &mut self.rows[layer - self.layers.start]
    }

    /// Layer `layer`'s lowest spare slot.
    fn spare(&self, layer: usize) -> Option<u32> {
        self.row(layer)
            .iter()
            .position(|s| *s == SlotState::Spare)
            .map(|s| s as u32)
    }
}

// ---------------------------------------------------------------- staging

/// Experts the staging ring holds at once.
pub const RING_SLOTS: usize = 4;

/// Bytes between two words of the staging page: a cache line each.
const WORD_STRIDE: usize = 64;

/// One expert's copy for the staging thread: first the victim to prepare,
/// then ring slot `ring`, `ticket` its use of that slot.
struct Job {
    /// The job's place in issue order.
    n: u64,
    layer: usize,
    id: u32,
    victim: Option<u32>,
    ring: usize,
    ticket: u32,
}

/// What the staging thread and the machine share.
struct Shared {
    source: Arc<dyn SwapSource>,
    /// `RING_SLOTS` experts of `slot_bytes`, pinned and device-mapped.
    ring: MappedHost,
    slot_bytes: usize,
    /// Per ring slot: `staged` (the thread's ticket once the slot holds the
    /// expert) at `2k`, `drained` (the copy stream's ticket once it has read
    /// the slot) at `2k + 1`.
    words: MappedHost,
    /// The host leg's wait window: staging runs while it is nonzero.
    window: Arc<AtomicU32>,
    /// Jobs whose victim is prepared, in job order.
    prepared: AtomicU64,
    /// Jobs below this are due: their flips land at a boundary the host has
    /// passed, so they stage whatever the window says.
    due: AtomicU64,
    /// The bound on every host wait ([`MachineCfg::deadline`]).
    deadline: Duration,
    /// Stage whatever the window says: a quiet boundary's relayout.
    flush: AtomicBool,
    stop: AtomicBool,
    /// The first staging failure, which the next boundary returns.
    failed: Mutex<Option<String>>,
}

impl Shared {
    fn word(&self, i: usize) -> &AtomicU32 {
        self.words
            .atomic_u32(i * WORD_STRIDE)
            .expect("the staging page holds 2 * RING_SLOTS words")
    }

    fn staged(&self, k: usize) -> &AtomicU32 {
        self.word(2 * k)
    }

    fn drained(&self, k: usize) -> &AtomicU32 {
        self.word(2 * k + 1)
    }

    /// Wait until `ready`: `Ok(true)` when it is, `Ok(false)` when the
    /// machine stops first, `Err(waited)` past `deadline` when one is given.
    fn wait_until(
        &self,
        ready: impl Fn() -> bool,
        deadline: Option<Duration>,
    ) -> Result<bool, Duration> {
        let t0 = Instant::now();
        let mut spins = 0u32;
        while !ready() {
            if self.stop.load(Ordering::Acquire) {
                return Ok(false);
            }
            if let Some(d) = deadline
                && t0.elapsed() > d
            {
                return Err(t0.elapsed());
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

    /// Copy `job`'s expert from its source into its ring slot, part after
    /// part.
    fn stage(&self, job: &Job) -> Result<(), GpuError> {
        let base = job.ring * self.slot_bytes;
        let mut at = 0usize;
        for (part, &want) in self.source.part_bytes().iter().enumerate() {
            let piece = self.source.source(job.layer, job.id, part)?;
            if piece.bytes.len() != want {
                return Err(GpuError::shape(
                    "SwapMachine staging",
                    format!(
                        "layer {} expert {} part {part}: {} source bytes for a slot of {want}",
                        job.layer,
                        job.id,
                        piece.bytes.len()
                    ),
                ));
            }
            match piece.transform {
                // SAFETY: the ring holds RING_SLOTS * slot_bytes bytes and
                // slot_bytes is the parts' sum, so [base + at, base + at +
                // want) is inside ring slot job.ring; the copy stream last
                // read that slot for the ticket before this one, which the
                // caller waited for (`drained`), and this thread is the
                // ring's only host writer.
                Transform::Identity => unsafe {
                    std::ptr::copy_nonoverlapping(
                        piece.bytes.as_ptr(),
                        self.ring.host_at(base + at),
                        want,
                    );
                },
            }
            at += want;
        }
        Ok(())
    }

    fn fail(&self, e: &GpuError) {
        let mut f = self
            .failed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if f.is_none() {
            *f = Some(e.to_string());
        }
    }

    /// The staging thread: each job prepares its victim and counts it, then
    /// waits for the window (or a flush, or the job due) and, within the
    /// deadline, for its ring slot's last copy; stages and publishes its
    /// ticket. A failure is kept for the next boundary and the ticket
    /// published anyway, so the copy stream never hangs on it: the bytes it
    /// copies land in a slot no entry names, and the boundary that would make
    /// them live refuses first.
    fn run(&self, jobs: &mpsc::Receiver<Job>) {
        while let Ok(job) = jobs.recv() {
            if let Some(v) = job.victim
                && let Err(e) = self.source.prepare_victim(job.layer, v)
            {
                self.fail(&e);
            }
            self.prepared.fetch_add(1, Ordering::AcqRel);
            let open = || {
                self.window.load(Ordering::Acquire) != 0
                    || self.flush.load(Ordering::Acquire)
                    || job.n < self.due.load(Ordering::Acquire)
            };
            if self.wait_until(open, None) != Ok(true) {
                return;
            }
            let prev = job.ticket - 1;
            let drained = || self.drained(job.ring).load(Ordering::Acquire) >= prev;
            let staged = match self.wait_until(drained, Some(self.deadline)) {
                Ok(true) => self.stage(&job),
                Ok(false) => return,
                Err(waited) => Err(GpuError::protocol(
                    "SwapMachine staging",
                    format!(
                        "ring slot {} was not drained of its last copy in {waited:?} (deadline \
                         {:?}): the copy stream is stuck",
                        job.ring, self.deadline
                    ),
                )),
            };
            if let Err(e) = staged {
                self.fail(&e);
            }
            self.staged(job.ring).store(job.ticket, Ordering::Release);
        }
    }
}

// ---------------------------------------------------------------- machine

/// What one boundary did ([`SwapMachine::boundary`]), for the `residency
/// pass` record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassReport {
    /// The boundary: passes since the load or the last reset.
    pub boundary: u64,
    /// The rows the pass before it kept (0 at boundary 0).
    pub kept: usize,
    /// Flips that went live here.
    pub landed: usize,
    /// Of those, the ones whose copy had not completed when the host reached
    /// the boundary, before it made them due: the engine stream waited for
    /// them.
    pub late: usize,
    /// Flips the rule made here.
    pub made: usize,
    /// Flips in flight after this boundary.
    pub in_flight: usize,
    /// Bytes this boundary's flips copy.
    pub bytes: u64,
}

/// What a reset did ([`SwapMachine::reset`]), for the `residency reset`
/// record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResetReport {
    /// Flips in flight the reset cancelled.
    pub cancelled: usize,
    /// Experts copied back onto the card.
    pub copies: usize,
    /// Entries of the live map that differ from the seed after the reset:
    /// experts live and not in the seed, plus seed experts not live.
    pub diff: usize,
    /// Host bytes released for the seed experts back on the card
    /// ([`SwapSource::release_host`]).
    pub dropped_bytes: u64,
}

/// The shape a machine is built for.
#[derive(Clone, Debug)]
pub struct MachineCfg {
    /// The rule's parameters; `spares` is each layer's free slots and
    /// `delay` must be 1 or more (a copy needs a pass to land in).
    pub params: SwapParams,
    /// Per layer of the map, the seed experts never a victim.
    pub pinned: Vec<usize>,
    /// Ids a row routes a layer.
    pub top_k: usize,
    /// The most rows one pass runs.
    pub max_rows: usize,
    /// The bound on every host wait the machine makes: a wait past it is
    /// a named error, never a hang.
    pub deadline: Duration,
}

/// The residency machine over one stage card's slot map.
///
/// Field order is drop order: the staging thread is stopped and joined in
/// `Drop` before the ring and the words it reads are freed.
pub struct SwapMachine {
    rule: SwapRule,
    ledger: SlotLedger,
    /// Per layer, the seed's live set at the load, slot order: what a reset
    /// returns to.
    seed: Vec<Vec<u32>>,
    pinned: Vec<usize>,
    layers: Range<usize>,
    n_expert: usize,
    rule_shape: Shape,
    /// The stage card's copy of the map: the words a boundary writes.
    view: sys::CUdeviceptr,
    copy: Arc<CudaStream>,
    /// Recorded on the engine stream at every boundary: a copy into a slot
    /// freed at or before it waits for it.
    boundary_event: CudaEvent,
    /// Per flip in flight, the copy stream's event and the flip's job;
    /// `free` lists the idle.
    events: Vec<CudaEvent>,
    event_job: Vec<u64>,
    free: Vec<usize>,
    /// Jobs issued: job `n` stages through ring slot `n % RING_SLOTS`.
    jobs_issued: u64,
    /// The boundary planned last; `None` until the first.
    planned: Option<u64>,
    kept: usize,
    ops: Vec<sys::CUstreamBatchMemOpParams>,
    changed: Vec<(usize, u32, u32)>,
    tx: Option<mpsc::Sender<Job>>,
    thread: Option<JoinHandle<()>>,
    shared: Arc<Shared>,
}

/// Operations one stream memory batch carries at most.
const BATCH_OPS: usize = 128;

impl SwapMachine {
    /// The machine over `slots`, the stage card's map, whose copy on the card
    /// is `view` (`layers × n_expert` words, row by row as
    /// [`SlotMap::stage_view`]), for the stacks `source` names; `stream` is
    /// the engine stream. Each layer's seed is its live experts in slot
    /// order, the map's capacity = live; its last `params.spares` go to the
    /// host now (their slots free) and its first `pinned[l]` are never a
    /// victim. A layer with no card slot takes no part. `view` and the
    /// stacks stay allocated and in place for the machine's life.
    ///
    /// Refused by name: a map with a tier entry (residency over the tier is
    /// not built), a layer whose capacity is not its live count, a layer with
    /// card slots fewer than `pinned + spares + 1`, a delay of 0, a pinned
    /// list of another length, a view of another length, a part not a whole
    /// number of words, and a freed expert the source cannot bring to the
    /// host.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        slots: &mut SlotMap,
        view: &DeviceBuffer<u32>,
        source: Arc<dyn SwapSource>,
        cfg: MachineCfg,
    ) -> Result<SwapMachine, GpuError> {
        const WHAT: &str = "SwapMachine::new";
        let layers = slots.layers();
        let n_expert = slots.n_expert();
        let MachineCfg {
            params,
            pinned,
            top_k,
            max_rows,
            deadline,
        } = cfg;
        if params.delay == 0 {
            return Err(GpuError::shape(
                WHAT,
                "a live delay of 0: a flip's copy needs a pass to land in",
            ));
        }
        if pinned.len() != layers.len() || view.len() != layers.len() * n_expert {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} pinned counts and a view of {} words for layers {layers:?} of {n_expert} \
                     experts",
                    pinned.len(),
                    view.len()
                ),
            ));
        }
        let parts = source.part_bytes();
        if parts.is_empty() || parts.iter().any(|&b| b == 0 || b % 4 != 0) {
            return Err(GpuError::shape(
                WHAT,
                format!("expert parts of {parts:?} bytes: each a nonzero multiple of 4"),
            ));
        }
        let slot_bytes: usize = parts.iter().sum();
        let s = params.spares;
        let mut seed = Vec::with_capacity(layers.len());
        let mut freed = Vec::new();
        let mut rule_pins = Vec::with_capacity(layers.len());
        for (i, l) in layers.clone().enumerate() {
            let (cap, live) = (slots.capacity(l)?, slots.on_card(l)?);
            if slots.on_tier(l)? > 0 || slots.tier_capacity(l)? > 0 {
                return Err(GpuError::shape(
                    WHAT,
                    format!("layer {l} has tier slots: residency over the tier is not built"),
                ));
            }
            if cap != live {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l}: {live} live experts in {cap} slots; the load's map is full"
                    ),
                ));
            }
            let mut order = vec![HOST; cap];
            for id in 0..n_expert as u32 {
                if let Some(Slot::Card(s)) = slots.slot(l, id) {
                    order[s as usize] = id;
                }
            }
            if cap == 0 {
                seed.push(Vec::new());
                rule_pins.push(0);
                continue;
            }
            if cap < pinned[i] + s + 1 {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l}: {cap} card slots for {} pinned, {s} spare and one that moves",
                        pinned[i]
                    ),
                ));
            }
            freed.extend(order[cap - s..].iter().map(|&id| (l, id)));
            seed.push(order);
            rule_pins.push(pinned[i]);
        }
        for &(l, id) in &freed {
            source.prepare_victim(l, id)?;
            if !source.host_resident(l, id)? {
                return Err(not_resident(WHAT, l, id, "a spare slot's expert"));
            }
        }
        let seed_refs: Vec<&[u32]> = seed.iter().map(Vec::as_slice).collect();
        let capacity: Vec<usize> = seed.iter().map(|o| o.len().saturating_sub(s)).collect();
        let shape = Shape {
            experts: n_expert,
            top_k,
            max_rows,
        };
        let rule = SwapRule::new_pinned(params, shape, &seed_refs, &capacity, &rule_pins).map_err(
            |e| GpuError::Plan {
                what: WHAT,
                source: Box::new(e),
            },
        )?;
        let ring = MappedHost::new(ctx, RING_SLOTS * slot_bytes, "cuMemHostAlloc (swap ring)")?;
        let words = MappedHost::new(
            ctx,
            2 * RING_SLOTS * WORD_STRIDE,
            "cuMemHostAlloc (swap words)",
        )?;
        let shared = Arc::new(Shared {
            source,
            ring,
            slot_bytes,
            words,
            window: Arc::new(AtomicU32::new(1)),
            prepared: AtomicU64::new(0),
            due: AtomicU64::new(0),
            deadline,
            flush: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            failed: Mutex::new(None),
        });
        let events = (0..layers.len() * s)
            .map(|_| ctx.new_event(None))
            .collect::<Result<Vec<_>, _>>()?;
        let (tx, rx) = mpsc::channel::<Job>();
        let for_thread = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("swap-staging".into())
            .spawn(move || for_thread.run(&rx))
            .map_err(|e| GpuError::shape(WHAT, format!("the staging thread: {e}")))?;
        let mut m = SwapMachine {
            rule,
            ledger: SlotLedger::of_map(slots)?,
            seed: seed
                .iter()
                .map(|o| o[..o.len().saturating_sub(s)].to_vec())
                .collect(),
            pinned: rule_pins,
            rule_shape: shape,
            layers: layers.clone(),
            n_expert,
            view: view.cu_deviceptr(),
            copy: ctx.new_stream()?,
            boundary_event: ctx.new_event(None)?,
            free: (0..events.len()).rev().collect(),
            event_job: vec![0; events.len()],
            events,
            jobs_issued: 0,
            planned: None,
            kept: 0,
            ops: Vec::with_capacity(BATCH_OPS),
            changed: Vec::new(),
            tx: Some(tx),
            thread: Some(thread),
            shared,
        };
        for &(l, id) in &freed {
            let s = slots.evict(l, id)?;
            if let Slot::Card(s) = s {
                m.ledger.row_mut(l)[s as usize] = SlotState::Spare;
            }
            m.changed.push((l, id, HOST));
        }
        m.write_changed(stream)?;
        stream.synchronize()?;
        Ok(m)
    }

    /// The host leg's wait window word: the staging thread copies while it
    /// is nonzero. The step port holds it open while its pool waits for a
    /// go, so the staging memcpy does not take DRAM from the host leg; a
    /// machine nobody gates leaves it at 1.
    #[must_use]
    pub fn window(&self) -> Arc<AtomicU32> {
        Arc::clone(&self.shared.window)
    }

    /// A tally of this machine's shape for a pass's routed ids
    /// ([`Tally::note`]): the step port's, which [`SwapMachine::end_pass`]
    /// folds.
    #[must_use]
    pub fn tally(&self) -> Tally {
        let r = self.rule_shape;
        Tally::new(self.layers.clone(), r.top_k, r.max_rows)
    }

    /// The slot ledger.
    #[must_use]
    pub fn ledger(&self) -> &SlotLedger {
        &self.ledger
    }

    /// The rule, for a reader of its card sets and flips in flight.
    #[must_use]
    pub fn rule(&self) -> &SwapRule {
        &self.rule
    }

    /// Layer `layer`'s seed: its live experts at the load, in slot order.
    #[must_use]
    pub fn seed(&self, layer: usize) -> &[u32] {
        layer
            .checked_sub(self.layers.start)
            .and_then(|i| self.seed.get(i))
            .map_or(&[], Vec::as_slice)
    }

    /// Layer `layer`'s pinned seed experts: its seed's first this many.
    #[must_use]
    pub fn pinned(&self, layer: usize) -> usize {
        layer
            .checked_sub(self.layers.start)
            .and_then(|i| self.pinned.get(i))
            .copied()
            .unwrap_or(0)
    }

    /// End the pass the last boundary opened: fold `tally`'s first `kept`
    /// rows into the rule and clear it. Refused by name: a pass no boundary
    /// opened, a tally of another shape ([`SwapMachine::tally`]), a kept row
    /// whose layer noted another count than `top_k`, and what the rule
    /// refuses.
    pub fn end_pass(&mut self, tally: &mut Tally, kept: usize) -> Result<(), GpuError> {
        const WHAT: &str = "SwapMachine::end_pass";
        if self.planned != Some(self.rule.passes()) {
            return Err(GpuError::state(WHAT, "a boundary before the pass"));
        }
        let r = self.rule_shape;
        if tally.layers != self.layers || tally.top_k != r.top_k || tally.max_rows != r.max_rows {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a tally of layers {:?}, top-{}, {} rows for a machine of layers {:?}, top-{}, \
                     {} rows",
                    tally.layers, tally.top_k, tally.max_rows, self.layers, r.top_k, r.max_rows
                ),
            ));
        }
        let t = &*tally;
        let n = self.layers.len();
        for row in 0..t.max_rows {
            for l in 0..n {
                let cell = row * n + l;
                let got = t.filled[cell] as usize;
                if got == 0 {
                    continue;
                }
                if got != t.top_k {
                    return Err(GpuError::shape(
                        WHAT,
                        format!(
                            "row {row} at layer {}: {got} routed ids noted, not {}",
                            self.layers.start + l,
                            t.top_k
                        ),
                    ));
                }
                let ids = &t.ids[cell * t.top_k..(cell + 1) * t.top_k];
                self.rule
                    .observe(l, row, ids)
                    .map_err(|e| rule_err(WHAT, e))?;
            }
        }
        self.rule.end_pass(kept).map_err(|e| rule_err(WHAT, e))?;
        tally.clear();
        self.kept = kept;
        Ok(())
    }

    /// The boundary before the next pass, on the engine stream `stream`, with
    /// the host map `slots`: the flips live here land (the stream waits for
    /// their copies, the card's copy and `slots` change), this boundary's
    /// event is recorded, then the rule plans and the new flips' copies are
    /// issued. Refused by name: a boundary twice with no pass between, a
    /// staging failure, a landing flip whose victim the host cannot serve
    /// from resident pages, and a host map that leaves the rule's card set.
    /// After a refusal the machine is not to be driven again.
    pub fn boundary(
        &mut self,
        stream: &CudaStream,
        slots: &mut SlotMap,
    ) -> Result<PassReport, GpuError> {
        const WHAT: &str = "SwapMachine::boundary";
        let b = self.rule.passes();
        if self.planned == Some(b) {
            return Err(GpuError::state(WHAT, "a pass since the last boundary"));
        }
        if let Some(e) = self
            .shared
            .failed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            return Err(GpuError::protocol(WHAT, format!("staging failed: {e}")));
        }
        let mut report = PassReport {
            boundary: b,
            kept: if b == 0 { 0 } else { self.kept },
            ..PassReport::default()
        };
        let landing = self.landing(b);
        for &(.., event, _) in &landing {
            if !self.events[event].query()? {
                report.late += 1;
            }
            let job = self.event_job[event];
            self.shared.due.fetch_max(job + 1, Ordering::AcqRel);
        }
        for &(l, slot, id, event, victim) in &landing {
            let job = self.event_job[event];
            let shared = &self.shared;
            let prepared = || shared.prepared.load(Ordering::Acquire) > job;
            match shared.wait_until(prepared, Some(shared.deadline)) {
                Ok(true) => {}
                Ok(false) => {
                    return Err(GpuError::protocol(WHAT, "the staging thread has stopped"));
                }
                Err(waited) => {
                    return Err(GpuError::protocol(
                        WHAT,
                        format!(
                            "layer {l}: the victim {victim} of expert {id} was not prepared for \
                             the host in {waited:?} (deadline {:?})",
                            shared.deadline
                        ),
                    ));
                }
            }
            if !self.shared.source.host_resident(l, victim)? {
                return Err(not_resident(
                    WHAT,
                    l,
                    victim,
                    &format!("the victim of expert {id} into slot {slot}"),
                ));
            }
        }
        for &(l, slot, id, event, victim) in &landing {
            stream.wait(&self.events[event])?;
            self.changed.push((l, id, slot));
            self.changed.push((l, victim, HOST));
        }
        self.write_changed(stream)?;
        for &(l, slot, id, event, victim) in &landing {
            let Slot::Card(freed) = slots.evict(l, victim)? else {
                return Err(GpuError::protocol(
                    WHAT,
                    format!("layer {l}: victim {victim} was not on the stage card"),
                ));
            };
            slots.admit(l, id, Slot::Card(slot))?;
            let row = self.ledger.row_mut(l);
            row[slot as usize] = SlotState::Live(id);
            row[freed as usize] = SlotState::Spare;
            self.free.push(event);
        }
        report.landed = landing.len();
        self.boundary_event.record(stream)?;
        if b > 0 {
            let made: Vec<Flip> = self.rule.plan(b).map_err(|e| rule_err(WHAT, e))?.to_vec();
            for f in &made {
                report.bytes += self.issue(slots, f)?;
            }
            report.made = made.len();
        }
        for &(l, ..) in &landing {
            self.agree(l, slots, WHAT)?;
        }
        report.in_flight = self.rule.in_flight().len();
        self.planned = Some(b);
        Ok(report)
    }

    /// The flips live at boundary `b`: (layer, slot, admitted, event,
    /// victim), in slot ledger order.
    fn landing(&self, b: u64) -> Vec<(usize, u32, u32, usize, u32)> {
        let mut out = Vec::new();
        for f in self.rule.in_flight().iter().filter(|f| f.live_at == b) {
            for (slot, st) in self
                .ledger
                .row(f.layer_of(self.layers.start))
                .iter()
                .enumerate()
            {
                if let SlotState::Filling { e, live_at, event } = *st
                    && e == f.admit
                    && live_at == b
                {
                    out.push((
                        f.layer_of(self.layers.start),
                        slot as u32,
                        e,
                        event,
                        f.evict,
                    ));
                }
            }
        }
        out
    }

    /// Issue flip `f`, made at this boundary, into its layer's lowest spare
    /// slot, with an event for its landing ([`SwapMachine::copy_into`]). The
    /// bytes it copies.
    fn issue(&mut self, slots: &SlotMap, f: &Flip) -> Result<u64, GpuError> {
        const WHAT: &str = "SwapMachine::issue";
        let l = f.layer_of(self.layers.start);
        let spare = self.ledger.spare(l).ok_or_else(|| {
            GpuError::protocol(WHAT, format!("layer {l}: a flip with no spare slot"))
        })?;
        let event = self
            .free
            .pop()
            .ok_or_else(|| GpuError::protocol(WHAT, "more flips in flight than spares"))?;
        self.event_job[event] = self.jobs_issued;
        let bytes = self.copy_into(slots, l, f.admit, spare, Some(f.evict))?;
        self.events[event].record(&self.copy)?;
        self.ledger.row_mut(l)[spare as usize] = SlotState::Filling {
            e: f.admit,
            live_at: f.live_at,
            event,
        };
        Ok(bytes)
    }

    /// Enqueue on the copy stream, behind this boundary's event and the
    /// staging word, layer `l`'s expert `id` into free slot `spare` of each
    /// stack, and hand the staging thread its job (preparing `victim` first).
    /// The bytes it copies.
    fn copy_into(
        &mut self,
        slots: &SlotMap,
        l: usize,
        id: u32,
        spare: u32,
        victim: Option<u32>,
    ) -> Result<u64, GpuError> {
        const WHAT: &str = "SwapMachine::copy_into";
        if slots.slot(l, id) != Some(Slot::Host) {
            return Err(GpuError::protocol(
                WHAT,
                format!(
                    "layer {l}: expert {id} to the card, which the map has on {:?}",
                    slots.slot(l, id)
                ),
            ));
        }
        let n = self.jobs_issued;
        let ring = (n % RING_SLOTS as u64) as usize;
        let ticket = u32::try_from(n / RING_SLOTS as u64 + 1)
            .map_err(|_| GpuError::protocol(WHAT, "the staging tickets passed u32"))?;
        let copy = Arc::clone(&self.copy);
        copy.wait(&self.boundary_event)?;
        let staged = self.shared.words.dev_at(2 * ring * WORD_STRIDE);
        let drained = self.shared.words.dev_at((2 * ring + 1) * WORD_STRIDE);
        mem_batch(
            &copy,
            &mut [op_wait_geq(staged, ticket)],
            "swap: wait for staging",
        )?;
        let mut at = ring * self.shared.slot_bytes;
        let mut bytes = 0u64;
        for (part, &len) in self.shared.source.part_bytes().iter().enumerate() {
            let dst = self.shared.source.dest(l, part, spare)?;
            // SAFETY: the ring holds RING_SLOTS * slot_bytes bytes and [at, at
            // + len) is ring slot `ring`'s part `part`; the staging thread
            // writes it only after `drained` reached this ticket's
            // predecessor and publishes it before `staged` reaches this
            // ticket, which the copy waits for above; the source vouches
            // `dst` holds `len` bytes of an allocation alive for the
            // machine's life, and no map entry names the slot until the
            // flip's event has completed on the engine stream.
            let rc = unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    dst,
                    self.shared.ring.host_at(at).cast_const().cast(),
                    len,
                    copy.cu_stream(),
                )
            };
            cu(rc, "cuMemcpyHtoDAsync_v2 (swap slot)")?;
            at += len;
            bytes += len as u64;
        }
        mem_batch(
            &copy,
            &mut [op_write(drained, ticket)],
            "swap: ring drained",
        )?;
        self.jobs_issued += 1;
        let job = Job {
            n,
            layer: l,
            id,
            victim,
            ring,
            ticket,
        };
        self.tx
            .as_ref()
            .and_then(|tx| tx.send(job).ok())
            .ok_or_else(|| GpuError::protocol(WHAT, "the staging thread has stopped"))?;
        Ok(bytes)
    }

    /// The host map's stage card set of layer `layer` is the rule's card
    /// set, else refused by name.
    fn agree(&self, layer: usize, slots: &SlotMap, what: &'static str) -> Result<(), GpuError> {
        let l = layer - self.layers.start;
        let rule: Vec<u32> = self.rule.live(l).map_err(|e| rule_err(what, e))?.collect();
        let map: Vec<u32> = (0..self.n_expert as u32)
            .filter(|&id| matches!(slots.slot(layer, id), Some(Slot::Card(_))))
            .collect();
        if rule != map {
            return Err(GpuError::protocol(
                what,
                format!(
                    "layer {layer}: the host map's card set {map:?} is not the rule's {rule:?}"
                ),
            ));
        }
        Ok(())
    }

    /// Write the queued changes of the card's copy on `stream`, in stream
    /// order: each (layer, id, entry) one word.
    fn write_changed(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        let (view, n) = (self.view, self.n_expert);
        let start = self.layers.start;
        for chunk in self.changed.chunks(BATCH_OPS) {
            self.ops.clear();
            self.ops.extend(chunk.iter().map(|&(l, id, entry)| {
                let word = (l - start) * n + id as usize;
                op_write(view + 4 * word as u64, entry)
            }));
            mem_batch(stream, &mut self.ops, "swap: the card's map words")?;
        }
        self.changed.clear();
        Ok(())
    }

    /// Back to the seed at a quiet boundary, on the engine stream `stream`
    /// with the host map `slots`: the staging drains, the flips in flight are
    /// cancelled, every live expert not in its layer's seed goes to the host
    /// and every seed expert not live is copied back onto the card, the
    /// engine stream waiting for the copies; the rule returns to its seed.
    /// The caller's tally is its own to clear ([`Tally::clear`]). Refused by
    /// name: an expert sent to the host the source cannot serve from
    /// resident pages, and a staging failure.
    pub fn reset(
        &mut self,
        stream: &CudaStream,
        slots: &mut SlotMap,
    ) -> Result<ResetReport, GpuError> {
        const WHAT: &str = "SwapMachine::reset";
        let mut report = ResetReport::default();
        self.drain_copies(WHAT)?;
        for l in self.layers.clone() {
            for st in self.ledger.row_mut(l).iter_mut() {
                if let SlotState::Filling { event, .. } = *st {
                    *st = SlotState::Spare;
                    self.free.push(event);
                    report.cancelled += 1;
                }
            }
        }
        let mut back = Vec::new();
        for (i, l) in self.layers.clone().enumerate() {
            let seed = &self.seed[i];
            let live: Vec<u32> = (0..self.n_expert as u32)
                .filter(|&id| matches!(slots.slot(l, id), Some(Slot::Card(_))))
                .collect();
            for &id in live.iter().filter(|id| !seed.contains(id)) {
                self.shared.source.prepare_victim(l, id)?;
                if !self.shared.source.host_resident(l, id)? {
                    return Err(not_resident(
                        WHAT,
                        l,
                        id,
                        "an admitted expert the reset sends back",
                    ));
                }
                if let Slot::Card(s) = slots.evict(l, id)? {
                    self.ledger.row_mut(l)[s as usize] = SlotState::Spare;
                }
                self.changed.push((l, id, HOST));
            }
            back.extend(
                seed.iter()
                    .filter(|id| !live.contains(id))
                    .map(|&id| (l, id)),
            );
        }
        self.boundary_event.record(stream)?;
        let mut placed = Vec::with_capacity(back.len());
        for &(l, id) in &back {
            let spare = self.ledger.spare(l).ok_or_else(|| {
                GpuError::protocol(
                    WHAT,
                    format!("layer {l}: no free slot for seed expert {id}"),
                )
            })?;
            self.copy_into(slots, l, id, spare, None)?;
            // Held until the copies land, so the next seed expert takes
            // another slot; the map names it only then.
            self.ledger.row_mut(l)[spare as usize] = SlotState::Live(id);
            placed.push((l, id, spare));
        }
        self.drain_copies(WHAT)?;
        if let Some(e) = self
            .shared
            .failed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            return Err(GpuError::protocol(WHAT, format!("staging failed: {e}")));
        }
        for &(l, id, spare) in &placed {
            slots.admit(l, id, Slot::Card(spare))?;
            self.changed.push((l, id, spare));
            report.dropped_bytes += self.shared.source.release_host(l, id)?;
        }
        report.copies = placed.len();
        self.write_changed(stream)?;
        self.rule.reset();
        self.planned = None;
        self.kept = 0;
        for (i, l) in self.layers.clone().enumerate() {
            let seed = &self.seed[i];
            let live: Vec<u32> = (0..self.n_expert as u32)
                .filter(|&id| matches!(slots.slot(l, id), Some(Slot::Card(_))))
                .collect();
            report.diff += live.iter().filter(|id| !seed.contains(id)).count();
            report.diff += seed.iter().filter(|id| !live.contains(id)).count();
        }
        Ok(report)
    }
}

impl SwapMachine {
    /// Stage every queued job whatever the window says and wait, within the
    /// deadline, for the copy stream to drain; past it the refusal of
    /// `what` names the wait.
    fn drain_copies(&self, what: &'static str) -> Result<(), GpuError> {
        self.shared.flush.store(true, Ordering::Release);
        let t0 = Instant::now();
        let r = loop {
            match self.copy.query() {
                Ok(true) => break Ok(()),
                Ok(false) if t0.elapsed() > self.shared.deadline => {
                    break Err(GpuError::protocol(
                        what,
                        format!(
                            "the copy stream did not drain in {:?} (deadline {:?})",
                            t0.elapsed(),
                            self.shared.deadline
                        ),
                    ));
                }
                Ok(false) => std::thread::sleep(Duration::from_micros(50)),
                Err(e) => break Err(e.into()),
            }
        };
        self.shared.flush.store(false, Ordering::Release);
        r
    }
}

impl Drop for SwapMachine {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.tx = None;
        if let Some(t) = self.thread.take() {
            // A panic on the staging thread has nowhere to go on the drop path.
            let _ = t.join();
        }
        // A copy still waiting for a ticket the thread never published is let
        // through, so the stream drains before the ring is freed.
        for k in 0..RING_SLOTS {
            self.shared.staged(k).store(u32::MAX, Ordering::Release);
        }
        // With every ticket published the copy stream waits on no host word,
        // only on boundary events of the engine stream, which waits on
        // nothing but these copies: it drains. The ring stays allocated until
        // it has. A failure on the drop path is unreportable and ignored.
        let _ = self.copy.synchronize();
    }
}

/// A flip's layer in the map's numbering, the rule's layer offset by `start`.
trait LayerOf {
    fn layer_of(&self, start: usize) -> usize;
}

impl LayerOf for Flip {
    fn layer_of(&self, start: usize) -> usize {
        start + self.layer
    }
}

fn rule_err(what: &'static str, e: runtime::swaprule::SwapRuleError) -> GpuError {
    GpuError::Plan {
        what,
        source: Box::new(e),
    }
}

fn not_resident(what: &'static str, layer: usize, id: u32, which: &str) -> GpuError {
    GpuError::protocol(
        what,
        format!(
            "layer {layer} expert {id}, {which}, is not host-resident: the host would serve it \
             from the file (a page fault a step), so the flip is refused"
        ),
    )
}
