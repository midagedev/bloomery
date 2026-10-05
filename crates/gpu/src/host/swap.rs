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
//! `Live(e)`, `Spare`, `Filling` — being written for a flip made at an
//! earlier boundary — or `Reserved(e)`, being written by a reset. No map
//! entry names a `Filling` or `Reserved` slot, so no kernel reads it and
//! neither device view shows it.
//!
//! **The tier.** A map may hold tier card entries (a two-card placement).
//! They are no part of the machine: never a victim, never admitted, never a
//! card slot. The rule sees the stage card's set alone and holds each tier
//! expert away ([`SwapRule::new_placed`]); the tier's stacks and its copy of
//! the map are never written.
//!
//! **Flips.** A flip made at boundary `b` admits an expert into a spare slot
//! while its victim stays live; both change at boundary `b + delay`. At `b`
//! the machine sends the staging thread the flip's job, then enqueues on its
//! copy stream, behind the event of boundary `b` (the slot's last reader has
//! run), a wait for the staging thread's word, per part the copy from the
//! staging ring into the slot's place in its stack and the source's convert
//! step ([`SwapSource::convert`]), a word back to the staging thread, and the
//! flip's event; the staging thread prepares the victim for the host
//! ([`SwapSource::prepare_victim`]), then copies the source bytes into the
//! ring, gated to the host leg's wait window ([`SwapMachine::window`]).
//!
//! **Boundaries.** [`SwapMachine::boundary`] runs before each pass's launch,
//! in this order: the jobs of the flips live here are made due, and the host
//! waits until the staging thread has published each (its victim prepared,
//! its bytes in the ring, or its failure recorded); a staging failure, or a
//! victim the host cannot serve from resident pages once its pages are read
//! in again, is refused by name there, before anything changes. Then the
//! engine stream waits for those
//! flips' copies; the changed words of the card's copy are written on the
//! engine stream; the host [`SlotMap`] takes the same change; this boundary's
//! event is recorded; then the rule plans and the new flips' copies are
//! issued. A flip lands at `b + delay` whatever its copy's progress: a late
//! copy makes the engine stream wait, and the boundary never moves, so the
//! same history of passes gives the same map at every pass. A boundary may be
//! made ahead of its pass ([`SwapMachine::boundary_ahead`]) — once the pass
//! before it is served and ended, before that pass's readback — and the next
//! launch takes it ([`SwapMachine::take_ahead`]): the same boundary, its
//! engine stream work in the same place between the two passes, its host
//! time under the pass before it.
//!
//! **No wait without a bound.** A flip landing at a boundary makes its job
//! due: the staging thread stages a due job whatever the window says. The
//! host's wait for it cannot close a cycle: it ends before this boundary
//! enqueues anything on the engine stream; staging a due job waits only on
//! its ring slot's previous copy, which waits on a boundary event recorded at
//! or before the boundary that issued the job, and the engine stream reaches
//! that event having waited only on copies that landed earlier — each of
//! which the host waited for at its own landing — and on the host words of
//! passes this thread served before this boundary: a boundary runs after the
//! last launched pass's host service has returned, after that pass's
//! readback or ahead of it. Every host wait the machine
//! makes — for a landing job, for a ring slot's last copy, for the copy
//! stream at a reset or a drop — ends by [`MachineCfg::deadline`] with a
//! named error; only a job not yet due waits for the window as long as the
//! window stays closed, and with it its copy and any context synchronize or
//! card free behind that copy: the machine's owner drops it before it frees
//! anything ([`crate::host::HostTier::stop_swap`]), and the drop releases
//! every such copy. A panic on the staging thread is a staging failure of its
//! job, published like any other.
//!
//! **Broken.** An error after a call's first change — a boundary's first
//! engine stream wait, `end_pass`'s first fold into the rule, a reset's first
//! cancel — leaves the device table, the host map, the ledger and the rule
//! out of step, and so does a staging failure: the machine then refuses every
//! later call by name. An error before the first change leaves it as it was.
//!
//! **Host residency.** Whether the host serves an expert the card gives up
//! from resident pages has one owner, `host_serves`, which every site that
//! sends an expert to the host asks: a boundary's landing victims, a call's
//! picked victims and the experts its end sends back, the spare slots'
//! experts at [`SwapMachine::new`] and the admitted experts a reset sends
//! back. A host set the load populated and did not lock can lose a page to
//! the page cache after the load or after the staging thread's prepare: the
//! owner then reads the expert's pages in again on the machine's thread
//! ([`SwapSource::prepare_victim`]), a page fault's cost paid once and
//! counted ([`PassReport::rereads`]), and refuses by name only an expert
//! still not resident after it — one the host set does not hold.
//!
//! **Counting.** The ids a pass routes are noted per (layer, row) into a
//! [`Tally`] and folded into the rule at [`SwapMachine::end_pass`] for the
//! pass's kept rows only, so a rejected row leaves no trace.
//!
//! **Calls.** A prompt call is one pass whose map moves inside it: between
//! [`SwapMachine::begin_call`] and [`SwapMachine::end_call`] the rule makes
//! no plan and every boundary, end of pass and reset is refused, and each
//! [`SwapMachine::call_pick`] moves one layer's pool — its live, unpinned
//! residents outside the rule's flips in flight — toward the hottest experts
//! of the counts it is given ([`SwapRule::call_pick`]), all at once: the
//! admitted experts' jobs go to the staging thread, which stages whatever the
//! window says for the call; the copy stream waits for the layer's reader
//! event (the last engine stream read of the layer's slots,
//! [`SwapMachine::call_reader`], or the call's start) and copies them in; the
//! layer's landed event follows; then the host map, the rule and the card's
//! copy of the layer's words change, the victims to the host (which serves
//! them from resident pages from this pick on) and each admitted expert to
//! its victim's slot, `Landing` in the ledger. No engine stream kernel may
//! read a landing slot before the engine stream waits for the layer's landed
//! event ([`SwapMachine::call_landed`]); the layer's next reader event makes
//! that wait itself and turns its landing slots `Live`, and a pick of a layer
//! whose last pick no reader has waited for is refused by name. The call
//! ends with every landing slot waited for and live, and either keeps its
//! placement for the passes after it or returns each layer to the set it
//! started with.

use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer, sys};
use runtime::swaprule::{Flip, KeptRows, Shape, SwapParams, SwapRule};

use super::slots::{HOST, Slot, SlotMap};
use super::{Drain, drain_within, poll_drained};
use crate::GpuError;
use crate::graph::{MappedHost, cu, mem_batch, op_write};

/// The residency lever: `off`, or the rule's `mid` parameters with `P`
/// pinned seed experts a layer and `S` spare slots a layer (`mid-p<P>-s<S>`).
/// The values a binary takes are the lever registry's, read at `main`
/// (`bloomery_levers::Levers::residency`); `bloomery_levers::residency_word`
/// is their grammar.
pub use bloomery_levers::RESIDENCY;

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
    /// `v` as a value of [`RESIDENCY`], in the lever's grammar
    /// ([`bloomery_levers::residency_word`]: `off`, or `mid-p<P>-s<S>`, `S` at
    /// least 1). Anything else is refused by name.
    pub fn parse(v: &str) -> Result<Residency, GpuError> {
        match bloomery_levers::residency_word(v) {
            Some(bloomery_levers::ResidencyWord::Off) => Ok(Residency::Off),
            Some(bloomery_levers::ResidencyWord::Mid { pinned, spares }) => {
                Ok(Residency::Mid { pinned, spares })
            }
            None => Err(GpuError::shape(
                "Residency::parse",
                format!("{RESIDENCY}={v:?}: it takes off or mid-p<P>-s<S> (S at least 1)"),
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
/// gives up. The staging thread calls [`SwapSource::source`], and
/// [`SwapSource::prepare_victim`] for a flip's victim; the thread that drives
/// the machine calls `prepare_victim` too, at [`SwapMachine::new`] for the
/// experts the spare slots give up and at [`SwapMachine::reset`] for the
/// admitted experts it sends back, and makes every other call.
pub trait SwapSource: Send + Sync {
    /// Bytes one expert takes in each of layer `layer`'s stage stacks (its
    /// parts: gate, up, down, …), in stack order; every entry a multiple of
    /// 4. Empty for a layer whose stacks the source holds none of: nothing
    /// of that layer ever stages, so the machine takes no part in it.
    fn part_bytes(&self, layer: usize) -> &[usize];

    /// Part `part` of layer `layer`'s expert `id` as its source holds it:
    /// [`SwapSource::part_bytes`]`(layer)[part]` bytes once transformed.
    fn source(&self, layer: usize, id: u32, part: usize) -> Result<Piece<'_>, GpuError>;

    /// The device address slot `slot` of layer `layer`'s stack `part` starts
    /// at. The stacks stay allocated and in place until the machine's copy
    /// stream has drained: its owner stops the machine before it frees them
    /// ([`crate::host::HostTier::stop_swap`]).
    fn dest(&self, layer: usize, part: usize, slot: u32) -> Result<sys::CUdeviceptr, GpuError>;

    /// Enqueue on `stream`, the machine's copy stream, whatever turns part
    /// `part`'s staged bytes, just copied to `dst`, into the slot's layout in
    /// place (a source whose card layout differs from its host bytes); the
    /// copies of later parts and the flip's event follow it on the stream.
    /// Nothing by default: the staged bytes are the slot's.
    fn convert(
        &self,
        layer: usize,
        part: usize,
        dst: sys::CUdeviceptr,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        let _ = (layer, part, dst, stream);
        Ok(())
    }

    /// Bring layer `layer`'s expert `id` to where the host serves it from
    /// resident pages (read its pages in). The staging thread calls it for a
    /// flip's victim ahead of the landing; the machine's thread calls it
    /// again for an expert it finds not host-resident when it sends it to
    /// the host, whose pages the page cache let go since.
    fn prepare_victim(&self, layer: usize, id: u32) -> Result<(), GpuError>;

    /// Whether the host serves layer `layer`'s expert `id` from resident
    /// pages now. The machine asks it through its one decision
    /// (`host_serves`): an expert that is not is prepared and asked again,
    /// and a flip whose victim still is not is refused by name.
    fn host_resident(&self, layer: usize, id: u32) -> Result<bool, GpuError>;

    /// Release the host pages of layer `layer`'s expert `id`, which stays on
    /// the card after a reset; the bytes released. The host pages of an
    /// admitted expert stay until a reset sends it back or releases them.
    fn release_host(&self, layer: usize, id: u32) -> Result<u64, GpuError>;
}

// ------------------------------------------------------------------ tally

/// The most ids a row routes a layer that a [`Tally`] holds: one bit each.
pub const TALLY_TOP_K: usize = 64;

/// The ids a pass routes, per (row, layer), as the host sees them: filled
/// one slot at a time by [`Tally::note`] and folded into the rule at
/// [`SwapMachine::end_pass`]. `Tally::off()` notes nothing.
#[derive(Clone, Debug, Default)]
pub struct Tally {
    on: bool,
    layers: Range<usize>,
    top_k: usize,
    max_rows: usize,
    /// `((row * layers + layer) * top_k + k)`.
    ids: Vec<u32>,
    /// Per (row, layer), a bit per slot noted this pass.
    filled: Vec<u64>,
    /// Per layer: no pass routes it, so no note of it is taken
    /// ([`MachineCfg::unrouted`]).
    unrouted: Vec<bool>,
}

impl Tally {
    /// A tally that notes nothing: the step port's without a machine.
    #[must_use]
    pub fn off() -> Tally {
        Tally::default()
    }

    fn new(layers: Range<usize>, top_k: usize, max_rows: usize, unrouted: Vec<bool>) -> Tally {
        let cells = layers.len() * max_rows;
        Tally {
            on: true,
            layers,
            top_k,
            max_rows,
            ids: vec![0; cells * top_k],
            filled: vec![0; cells],
            unrouted,
        }
    }

    /// Whether layer `layer` is one this tally counts: a caller that routes
    /// other layers too asks before it notes.
    #[must_use]
    pub fn covers(&self, layer: usize) -> bool {
        self.on && self.layers.contains(&layer)
    }

    /// Slot `k` of row `row`'s routing at layer `layer` is `id`. A tally that
    /// is off notes nothing. Refused by name: a layer it does not cover
    /// ([`Tally::covers`]), a row or a slot past its shape, a slot noted
    /// twice in one pass, and a layer no pass routes
    /// ([`MachineCfg::unrouted`]).
    pub fn note(&mut self, layer: usize, row: usize, k: usize, id: u32) -> Result<(), GpuError> {
        const WHAT: &str = "Tally::note";
        if !self.on {
            return Ok(());
        }
        if !self.layers.contains(&layer) || row >= self.max_rows || k >= self.top_k {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {layer} row {row} slot {k}: the tally holds layers {:?}, {} rows, \
                     top-{}",
                    self.layers, self.max_rows, self.top_k
                ),
            ));
        }
        if self.unrouted[layer - self.layers.start] {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "layer {layer} row {row}: a layer no pass routes (MachineCfg::unrouted); its \
                     routing is not the rule's"
                ),
            ));
        }
        let cell = row * self.layers.len() + (layer - self.layers.start);
        let bit = 1u64 << k;
        if self.filled[cell] & bit != 0 {
            return Err(GpuError::shape(
                WHAT,
                format!("layer {layer} row {row}: slot {k} noted twice in one pass"),
            ));
        }
        self.filled[cell] |= bit;
        self.ids[cell * self.top_k + k] = id;
        Ok(())
    }

    /// Forget the noted ids: the next pass starts empty.
    pub fn clear(&mut self) {
        self.filled.fill(0);
    }

    /// Every slot of a cell noted.
    fn full(&self) -> u64 {
        if self.top_k == TALLY_TOP_K {
            u64::MAX
        } else {
            (1u64 << self.top_k) - 1
        }
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
    /// Being written for seed expert `e` by a reset, live once the reset's
    /// copies have landed. No map entry names it.
    Reserved(u32),
    /// Named by the map for expert `e`, admitted by a call's pick whose
    /// copies no engine stream wait covers yet: `event` is the machine's
    /// landed event for it ([`SwapMachine::call_landed`]).
    Landing { e: u32, event: usize },
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

    /// Layer `layer`'s slots; `None` for a layer outside the ledger.
    #[must_use]
    pub fn row(&self, layer: usize) -> Option<&[SlotState]> {
        layer
            .checked_sub(self.layers.start)
            .and_then(|i| self.rows.get(i))
            .map(Vec::as_slice)
    }

    fn row_mut(&mut self, layer: usize) -> &mut Vec<SlotState> {
        &mut self.rows[layer - self.layers.start]
    }

    /// A call's pick sends slot `slot` of layer `layer` from victim `victim`
    /// to expert `e`, landing behind landed event `event`. Refused by name,
    /// the ledger unchanged: a layer or a slot outside it, and a slot that is
    /// not `Live(victim)`.
    pub fn to_landing(
        &mut self,
        layer: usize,
        slot: u32,
        victim: u32,
        e: u32,
        event: usize,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "SlotLedger::to_landing";
        let st = layer
            .checked_sub(self.layers.start)
            .and_then(|i| self.rows.get_mut(i))
            .and_then(|row| row.get_mut(slot as usize))
            .ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!("layer {layer} slot {slot} is outside the ledger"),
                )
            })?;
        if *st != SlotState::Live(victim) {
            return Err(GpuError::protocol(
                WHAT,
                format!("layer {layer} slot {slot} is {st:?}, not live with the victim {victim}"),
            ));
        }
        *st = SlotState::Landing { e, event };
        Ok(())
    }

    /// Layer `layer`'s landing slots turn `Live`: the engine stream waits for
    /// their landed event from here on. Whether it held any; a layer outside
    /// the ledger holds none.
    pub fn land(&mut self, layer: usize) -> bool {
        let Some(row) = layer
            .checked_sub(self.layers.start)
            .and_then(|i| self.rows.get_mut(i))
        else {
            return false;
        };
        let mut any = false;
        for st in row.iter_mut() {
            if let SlotState::Landing { e, .. } = *st {
                *st = SlotState::Live(e);
                any = true;
            }
        }
        any
    }

    /// Layer `layer`'s first landing slot, if any.
    #[must_use]
    pub fn landing(&self, layer: usize) -> Option<(u32, SlotState)> {
        self.row(layer)?
            .iter()
            .enumerate()
            .find(|(_, s)| matches!(s, SlotState::Landing { .. }))
            .map(|(i, &s)| (i as u32, s))
    }

    /// Layer `layer`'s lowest spare slot.
    fn spare(&self, layer: usize) -> Option<u32> {
        self.row(layer)?
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

/// Nanoseconds since `t0`, saturated ([`super::nanos`]).
fn nanos(t0: Instant) -> u64 {
    super::nanos(t0.elapsed())
}

/// Whole microseconds since `t0`, saturated.
fn micros(t0: Instant) -> u64 {
    u64::try_from(t0.elapsed().as_micros()).unwrap_or(u64::MAX)
}

/// Where the staging thread runs: the SMT sibling of the pool's first
/// worker. The copies are DRAM streams that sweep a ring slot's worth of L3,
/// so not beside the dispatcher (the critical path, whose sibling the engram
/// helper takes): the first worker's core sits on another CCD than the
/// dispatcher's. The copies run while the pool waits for its go, so that
/// worker only spins beside them. With no pinned pool, floating.
fn staging_placement() -> threads::helper::Placement {
    match threads::built().and_then(|p| p.worker_cpu(0)) {
        Some(c) => threads::helper::Placement::Sibling(c),
        None => threads::helper::Placement::Float,
    }
}

/// Job `n`'s ring slot and its ticket there (its use of that slot, from 1):
/// the one owner of the job → ring mapping.
fn ring_ticket(n: u64) -> Result<(usize, u32), GpuError> {
    let ring = (n % RING_SLOTS as u64) as usize;
    let ticket = u32::try_from(n / RING_SLOTS as u64 + 1)
        .map_err(|_| GpuError::protocol("SwapMachine staging", "the staging tickets passed u32"))?;
    Ok((ring, ticket))
}

/// The one way a wait on a staging word goes on the copy stream: a copy
/// stream wait whose word only the staging thread raises is enqueued only
/// for a job that thread already holds, so the thread enqueueing never waits
/// on a wait it has yet to release.
mod dispatch {
    use std::sync::mpsc;

    use cuda_core::CudaStream;

    use super::{Job, WORD_STRIDE};
    use crate::GpuError;
    use crate::graph::{MappedHost, mem_batch, op_wait_geq};

    /// A job the staging thread holds. Only [`send`] makes one.
    pub(super) struct Dispatched {
        ring: usize,
        ticket: u32,
    }

    impl Dispatched {
        pub(super) fn ring(&self) -> usize {
            self.ring
        }

        pub(super) fn ticket(&self) -> u32 {
            self.ticket
        }
    }

    /// `job` to the staging thread over `tx`; refused by name as `what` when
    /// the thread has stopped.
    pub(super) fn send(
        tx: Option<&mpsc::Sender<Job>>,
        job: Job,
        what: &'static str,
    ) -> Result<Dispatched, GpuError> {
        let (ring, ticket) = (job.ring, job.ticket);
        tx.and_then(|tx| tx.send(job).ok())
            .ok_or_else(|| GpuError::protocol(what, "the staging thread has stopped"))?;
        Ok(Dispatched { ring, ticket })
    }

    /// Enqueue on `copy` the wait for `d`'s staging: its ring slot's staged
    /// word in `words` at its ticket.
    pub(super) fn wait_staged(
        copy: &CudaStream,
        words: &MappedHost,
        d: &Dispatched,
    ) -> Result<(), GpuError> {
        let staged = words.dev_at(2 * d.ring * WORD_STRIDE);
        mem_batch(
            copy,
            &mut [op_wait_geq(staged, d.ticket)],
            "swap: wait for staging",
        )
    }
}

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

/// The staging ring: [`RING_SLOTS`] experts of `slot_bytes`, pinned and
/// device-mapped. Its bytes are written and read here only.
struct Ring {
    page: MappedHost,
    slot_bytes: usize,
}

impl Ring {
    fn new(ctx: &Arc<CudaContext>, slot_bytes: usize) -> Result<Ring, GpuError> {
        Ok(Ring {
            page: MappedHost::new(ctx, RING_SLOTS * slot_bytes, "cuMemHostAlloc (swap ring)")?,
            slot_bytes,
        })
    }

    /// The byte offset of `[at, at + len)` of ring slot `k`, or the refusal
    /// of `what` for a span past the slot.
    fn span(&self, k: usize, at: usize, len: usize, what: &'static str) -> Result<usize, GpuError> {
        if k >= RING_SLOTS || at.checked_add(len).is_none_or(|end| end > self.slot_bytes) {
            return Err(GpuError::shape(
                what,
                format!(
                    "bytes [{at}, +{len}) of ring slot {k}: {RING_SLOTS} slots of {}",
                    self.slot_bytes
                ),
            ));
        }
        Ok(k * self.slot_bytes + at)
    }

    /// `bytes` into ring slot `k` at byte `at`. The caller is the ring's one
    /// host writer and writes slot `k` only once the copy stream has read
    /// its previous use (`drained`).
    fn write(&self, k: usize, at: usize, bytes: &[u8]) -> Result<(), GpuError> {
        let off = self.span(k, at, bytes.len(), "SwapMachine ring write")?;
        // SAFETY: [off, off + len) lies inside the page (checked above); no
        // copy reads ring slot k until the staging word publishes this use,
        // and the staging thread is the only host writer. The word's Release
        // store follows these stores in program order, which x86 (TSO) keeps
        // in the coherent write-back pinned page, and the copy engine reads
        // the page over PCIe with no SM cache between: the copy the word
        // lets through reads these bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.page.host_at(off), bytes.len());
        }
        Ok(())
    }

    /// Enqueue on `stream` the copy of `len` bytes at byte `at` of ring slot
    /// `k` to device address `dst`.
    fn copy_to_device(
        &self,
        k: usize,
        at: usize,
        len: usize,
        dst: sys::CUdeviceptr,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        let off = self.span(k, at, len, "SwapMachine ring copy")?;
        // SAFETY: the source span lies inside the page (checked above), which
        // stays allocated until the copy stream has drained (the machine's
        // drop leaks it otherwise); the source vouches that `dst` holds `len`
        // bytes of an allocation alive until the copy stream has drained.
        let rc = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst,
                self.page.host_at(off).cast_const().cast(),
                len,
                stream.cu_stream(),
            )
        };
        cu(rc, "cuMemcpyHtoDAsync_v2 (swap slot)")
    }
}

/// The staging thread's first failure: its job and the thread's own error.
struct StagingFailure {
    job: u64,
    layer: usize,
    id: u32,
    error: GpuError,
}

/// A staging failure as the machine returns it, the source of a
/// [`GpuError::Plan`]: the job, its layer and expert, the boundary that
/// found it (`None` at a reset), and the staging thread's own error, which
/// is the chain's next link.
#[derive(Debug)]
pub struct StagingFailed {
    pub boundary: Option<u64>,
    pub job: u64,
    pub layer: usize,
    pub id: u32,
    pub error: GpuError,
}

impl std::fmt::Display for StagingFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let at = match self.boundary {
            Some(b) => format!("boundary {b}"),
            None => "a reset".to_string(),
        };
        write!(
            f,
            "staging failed for job {} (layer {} expert {}), found at {at}: {}",
            self.job, self.layer, self.id, self.error
        )
    }
}

impl std::error::Error for StagingFailed {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Experts found not host-resident and resident once their pages were read
/// in again (`host_serves`): how many, their bytes and the host nanoseconds
/// it took, since a boundary last took them ([`PassReport::rereads`]).
#[derive(Default)]
struct Rereads {
    n: AtomicU64,
    bytes: AtomicU64,
    ns: AtomicU64,
}

impl Rereads {
    fn note(&self, bytes: u64, ns: u64) {
        self.n.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        self.ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// The count, the bytes and the microseconds, each back to 0.
    fn take(&self) -> (usize, u64, u64) {
        let n = self.n.swap(0, Ordering::Relaxed);
        let bytes = self.bytes.swap(0, Ordering::Relaxed);
        let us = self.ns.swap(0, Ordering::Relaxed) / 1000;
        (n as usize, bytes, us)
    }
}

/// What the staging thread and the machine share.
struct Shared {
    source: Arc<dyn SwapSource>,
    ring: Ring,
    /// Per ring slot: `staged` (the thread's ticket once the slot holds the
    /// expert, or once its failure is recorded) at `2k`, `drained` (the copy
    /// stream's ticket once it has read the slot) at `2k + 1`.
    words: MappedHost,
    /// The host leg's wait window: staging runs while it is nonzero.
    window: Arc<AtomicU32>,
    /// Jobs below this are due: their flips land at a boundary the host has
    /// reached, so they stage whatever the window says.
    due: AtomicU64,
    /// The bound on every host wait ([`MachineCfg::deadline`]).
    deadline: Duration,
    /// Stage whatever the window says: a quiet boundary's relayout.
    flush: AtomicBool,
    /// Nanoseconds the staging thread spent copying into the ring and
    /// preparing victims, since a boundary last took them.
    stage_ns: AtomicU64,
    prepare_ns: AtomicU64,
    /// The experts the machine's thread read in again since a boundary last
    /// took them (`host_serves`).
    rereads: Rereads,
    /// Jobs the staging thread has finished (staged, or failed), counted
    /// before each one's ticket is published: `jobs_issued - served` copies
    /// wait for staging.
    served: AtomicU64,
    stop: AtomicBool,
    /// The first staging failure, which the next boundary or reset returns.
    failed: Mutex<Option<StagingFailure>>,
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
    /// part, at the offsets its layer's parts lay out.
    fn stage(&self, job: &Job) -> Result<(), GpuError> {
        let mut at = 0usize;
        for (part, &want) in self.source.part_bytes(job.layer).iter().enumerate() {
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
                Transform::Identity => self.ring.write(job.ring, at, piece.bytes)?,
            }
            at += want;
        }
        Ok(())
    }

    fn fail(&self, job: &Job, error: GpuError) {
        let mut f = self
            .failed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if f.is_none() {
            *f = Some(StagingFailure {
                job: job.n,
                layer: job.layer,
                id: job.id,
                error,
            });
        }
    }

    /// One job: prepare its victim, wait for the window (or a flush, or the
    /// job due) and, within the deadline, for its ring slot's last copy;
    /// stage. `Ok(false)` when the machine stops first.
    fn serve(&self, job: &Job) -> Result<bool, GpuError> {
        if let Some(v) = job.victim {
            let t0 = Instant::now();
            self.source.prepare_victim(job.layer, v)?;
            self.prepare_ns.fetch_add(nanos(t0), Ordering::Relaxed);
        }
        let open = || {
            self.window.load(Ordering::Acquire) != 0
                || self.flush.load(Ordering::Acquire)
                || job.n < self.due.load(Ordering::Acquire)
        };
        // No deadline: a job not yet due is idle while no pass runs (a
        // server between requests). What ends the wait: the step port's
        // window, a boundary landing the job (`due`), a reset's flush
        // (`drain_copies`, first thing in `reset`), and the drop's `stop`,
        // which `wait_until` reads on every turn.
        if self.wait_until(open, None) != Ok(true) {
            return Ok(false);
        }
        let prev = job.ticket - 1;
        let drained = || self.drained(job.ring).load(Ordering::Acquire) >= prev;
        match self.wait_until(drained, Some(self.deadline)) {
            Ok(true) => {}
            Ok(false) => return Ok(false),
            Err(waited) => {
                return Err(GpuError::protocol(
                    "SwapMachine staging",
                    format!(
                        "ring slot {} was not drained of its last copy in {waited:?} (deadline \
                         {:?}): the copy stream is stuck",
                        job.ring, self.deadline
                    ),
                ));
            }
        }
        let t0 = Instant::now();
        self.stage(job)?;
        self.stage_ns.fetch_add(nanos(t0), Ordering::Relaxed);
        Ok(true)
    }

    /// The staging thread: each job served ([`Shared::serve`]) and its
    /// ticket published. A failure — an error or a panic — is recorded for
    /// the next boundary, which waits for the ticket and refuses before its
    /// flip could go live, and the ticket is published anyway, so the copy
    /// stream never hangs on it: the bytes it copies land in a slot no entry
    /// names.
    fn run(&self, jobs: &mpsc::Receiver<Job>) {
        while let Ok(job) = jobs.recv() {
            match catch_unwind(AssertUnwindSafe(|| self.serve(&job))) {
                Ok(Ok(true)) => {}
                Ok(Ok(false)) => return,
                Ok(Err(e)) => self.fail(&job, e),
                Err(payload) => {
                    let why = payload
                        .downcast_ref::<&str>()
                        .map(ToString::to_string)
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "a payload that is not text".to_string());
                    let e = GpuError::protocol(
                        "SwapMachine staging",
                        format!("the staging thread panicked: {why}"),
                    );
                    self.fail(&job, e);
                }
            }
            self.served.fetch_add(1, Ordering::AcqRel);
            // A max, never a store: a word the machine released past this
            // ticket stays released.
            self.staged(job.ring)
                .fetch_max(job.ticket, Ordering::AcqRel);
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
    /// The kept rows as a mask, bit `r` = row `r` kept (0 at boundary 0):
    /// a prefix's mask for every pass but a drafted slots pass's, each
    /// slot's accepted rows.
    pub rows: u64,
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
    /// Host microseconds of the pass's end before this boundary (the fold of
    /// its kept rows into the rule).
    pub end_us: u64,
    /// Host microseconds of the whole boundary call: the wait, the landing,
    /// the plan and the issue.
    pub boundary_us: u64,
    /// Host microseconds this boundary waited for its landing jobs' staging.
    pub wait_us: u64,
    /// Host microseconds this boundary spent issuing its flips' copies (the
    /// copy stream's enqueues and the staging thread's jobs).
    pub issue_us: u64,
    /// Staging thread microseconds since the last boundary: copying experts'
    /// bytes into the ring, and preparing victims for the host.
    pub stage_us: u64,
    pub prepare_us: u64,
    /// Experts the machine's thread found not host-resident and read in
    /// again since the last boundary (the load's, a call's and a reset's
    /// with it): a page the page cache let go after the load or after the
    /// staging thread's prepare. Their bytes (whole experts, as the source
    /// holds them: the pages read are at most these), and the host
    /// microseconds the reads and their checks took on the machine's thread.
    pub rereads: usize,
    pub reread_bytes: u64,
    pub reread_us: u64,
    /// Made ahead of its pass ([`SwapMachine::boundary_ahead`]): after the
    /// pass before it was served and ended, before that pass's readback.
    pub ahead: bool,
    /// The engine's step and pass readbacks before this boundary, as its
    /// caller stamps it ([`BoundaryAt`]); 0 from a driver of the machine.
    pub reads: u64,
}

/// Where a residency boundary runs in the engine's pass order, with the
/// engine's readbacks before it ([`PassReport::reads`]): what a body's host
/// service is told at every boundary call
/// ([`crate::host::HostTier::swap_at`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryAt {
    /// Before a pass's launch: the boundary made ahead of it taken when the
    /// machine holds one, else the boundary made here.
    Launch { reads: u64 },
    /// After a pass's host service and its kept rows, before its readback:
    /// the next pass's boundary made ahead, under the rest of the pass.
    Ahead { reads: u64 },
}

impl BoundaryAt {
    /// The readbacks the caller stamps.
    #[must_use]
    pub fn reads(self) -> u64 {
        match self {
            BoundaryAt::Launch { reads } | BoundaryAt::Ahead { reads } => reads,
        }
    }
}

/// Why a dropped machine leaked its shared state ([`Leak`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeakReason {
    /// The staging thread was still inside the source past the deadline.
    Join,
    /// The copies waiting on staging words could not be let through.
    Release,
    /// The copy stream did not drain within the deadline.
    Drain,
    /// The copy stream's query returned a driver error: a sticky fault of
    /// the context, not a wait.
    Fault,
}

impl LeakReason {
    /// The reason as the `residency leak` record's word.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            LeakReason::Join => "join",
            LeakReason::Release => "release",
            LeakReason::Drain => "drain",
            LeakReason::Fault => "fault",
        }
    }
}

/// What a dropped machine leaked rather than free under a copy or a thread
/// that may still use it: the staging ring's and the staging words' pinned
/// bytes, and with them the source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Leak {
    pub reason: LeakReason,
    /// The driver error's `CUresult` on a fault ([`Leak::fault`]), kept for
    /// the `residency leak` record: the one fact of the fault the driver
    /// already reported. `None` on every other reason, which names no driver
    /// error.
    pub code: Option<u32>,
    pub ring_bytes: u64,
    pub words_bytes: u64,
}

impl Leak {
    /// The leak for the driver error the copy stream's query returned at a
    /// drop: the reason `fault` with the error's `CUresult` as `code`, so
    /// the record names which fault, not only that there was one. The
    /// driver's own text it asks the context for at print time, which a
    /// faulted context may not answer; the code it already gave. A drain's
    /// error is a driver error always (the poll builds it from the query),
    /// so a codeless fault is an error of another kind, which that path
    /// cannot make.
    #[must_use]
    pub fn fault(e: &GpuError, ring_bytes: u64, words_bytes: u64) -> Leak {
        Leak {
            reason: LeakReason::Fault,
            code: match e {
                GpuError::Driver { source, .. } => Some(source.0),
                _ => None,
            },
            ring_bytes,
            words_bytes,
        }
    }
}

/// Where a dropped machine reports a leak; set once a process.
static LEAK_SINK: OnceLock<fn(&Leak)> = OnceLock::new();

/// Report every leak of a dropped machine to `sink` (the binaries' `residency
/// leak` record). Unset, the drop names the leak on stderr. `false` when a
/// sink is set already.
pub fn set_leak_sink(sink: fn(&Leak)) -> bool {
    LEAK_SINK.set(sink).is_ok()
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
    /// Host bytes released ([`SwapSource::release_host`]): the seed experts
    /// back on the card, and the victims of cancelled flips, which stay on
    /// it.
    pub dropped_bytes: u64,
}

/// What a prompt call asks of the machine ([`SwapMachine::begin_call`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallCfg {
    /// The least count an admitted expert has in a pick's counts.
    pub floor: u32,
}

/// What one pick did ([`SwapMachine::call_pick`]), for the `call stream`
/// record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CallPick {
    pub layer: usize,
    /// Experts admitted, each in place of a pool resident sent to the host.
    pub admitted: usize,
    /// Pool residents the pick wanted and kept.
    pub kept: usize,
    /// Bytes the admitted experts' copies move.
    pub bytes: u64,
    /// The pick's input: FNV-1a 64 over its counts (each a little-endian
    /// `u32`, expert order). Two calls whose picks differ at a layer with
    /// equal digests differ in the card set they picked from, not the ids.
    pub counts: u64,
    /// Host microseconds the pick took: the choice, the jobs, the copy
    /// stream's enqueues and the words.
    pub pick_us: u64,
    /// Of those, host microseconds it waited for the staging thread to take
    /// in enough of the call's earlier jobs.
    pub backlog_us: u64,
}

/// What a call did ([`SwapMachine::end_call`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CallReport {
    /// Picks that admitted at least one expert.
    pub picks: usize,
    pub admitted: usize,
    pub bytes: u64,
    pub pick_us: u64,
    pub backlog_us: u64,
    /// Whether the call's placement stays for the passes after it.
    pub kept: bool,
    /// Experts copied back onto the card when it did not.
    pub restored: usize,
    /// Host microseconds the end took.
    pub end_us: u64,
}

/// An open call: what it asked, each layer's card set at its start (in the
/// rule's layer numbering, ascending ids), and its sums so far.
struct Call {
    cfg: CallCfg,
    start: Vec<Vec<u32>>,
    report: CallReport,
    /// The flush as the call found it.
    flush_was: bool,
}

/// Where a copy's wait for the readers of its slot comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gate {
    /// This boundary's event, enqueued once for every copy issued behind it.
    Boundary,
    /// Enqueued by the caller already (a call's pick: the layer's reader
    /// event).
    Waited,
}

/// The shape a machine is built for.
#[derive(Clone, Debug)]
pub struct MachineCfg {
    /// The rule's parameters; `spares` is each layer's free slots and
    /// `delay` must be 1 or more (a copy needs a pass to land in).
    pub params: SwapParams,
    /// Per layer of the map, the seed experts never a victim.
    pub pinned: Vec<usize>,
    /// Ids a row routes a layer, 1 to [`TALLY_TOP_K`].
    pub top_k: usize,
    /// The most rows one pass runs.
    pub max_rows: usize,
    /// Layers of the map no pass routes (a draft layer whose experts the
    /// host serves beside the chain): each has no card slot, a note of it is
    /// refused by name, and a kept row needs none of it
    /// ([`runtime::swaprule::SwapRule::with_unrouted`]). Empty for a load
    /// whose passes route every layer of its map.
    pub unrouted: Vec<usize>,
    /// The bound on every host wait the machine makes: a wait past it is a
    /// named error, never a hang. Each wait is counted from the call that
    /// makes it, and a landing's or a reset's wait includes the rest of the
    /// engine stream's pass before the boundary event its copies wait on, so
    /// the bound must be longer than the longest pass plus one expert's
    /// preparation and staging.
    pub deadline: Duration,
}

/// The machine's view of a load's map ([`SwapMachine::new`]): per layer, the
/// stage card's experts in slot order, the tier's experts, and the pinned
/// count; the experts the spare slots give up.
struct Layout {
    order: Vec<Vec<u32>>,
    tier: Vec<Vec<u32>>,
    pinned: Vec<usize>,
    freed: Vec<(usize, u32)>,
}

/// What [`SwapMachine::new`] starts for the staging: the shared state, the
/// job queue and the thread.
struct Staging {
    shared: Arc<Shared>,
    tx: mpsc::Sender<Job>,
    thread: JoinHandle<()>,
}

/// A flip live at a boundary as the ledger holds it.
#[derive(Clone, Copy, Debug)]
struct Landing {
    layer: usize,
    slot: u32,
    admit: u32,
    victim: u32,
    event: usize,
    job: u64,
}

/// The residency machine over one stage card's slot map.
///
/// Field order is drop order: the staging thread is stopped and joined in
/// `Drop` before the ring and the words it reads are freed.
pub struct SwapMachine {
    rule: SwapRule,
    ledger: SlotLedger,
    layers: Range<usize>,
    n_expert: usize,
    rule_shape: Shape,
    /// The stage card's copy of the map: the words a boundary writes.
    view: sys::CUdeviceptr,
    copy: Arc<CudaStream>,
    /// Recorded on the engine stream at every boundary: a copy into a slot
    /// freed at or before it waits for it.
    boundary_event: CudaEvent,
    /// The copy stream already waits for the last record of
    /// `boundary_event`: one wait serves every copy issued behind it.
    copy_waits_boundary: bool,
    /// Per flip in flight, the copy stream's event and the flip's job;
    /// `free` lists the idle.
    events: Vec<CudaEvent>,
    event_job: Vec<u64>,
    free: Vec<usize>,
    /// Jobs issued: job `n` stages through [`ring_ticket`]`(n)`.
    jobs_issued: u64,
    /// The boundary planned last; `None` until the first.
    planned: Option<u64>,
    /// A boundary made ahead of its pass ([`SwapMachine::boundary_ahead`])
    /// that no launch has taken yet ([`SwapMachine::take_ahead`]).
    ahead: Option<u64>,
    kept: KeptRows,
    /// Host microseconds the last pass's end took (the fold into the rule),
    /// for the next boundary's report.
    end_us: u64,
    ops: Vec<sys::CUstreamBatchMemOpParams>,
    changed: Vec<(usize, u32, u32)>,
    /// The error that broke the machine, named in every later refusal.
    broken: Option<String>,
    /// Per layer of the map: a call's landed event (on the copy stream,
    /// after a pick's copies) and reader event (on the engine stream, after
    /// the call's last read of the layer's slots); the open call.
    call_landed: Vec<CudaEvent>,
    call_read: Vec<CudaEvent>,
    call: Option<Call>,
    picks: Vec<Flip>,
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
    /// the engine stream. Each layer's seed is its stage card experts in slot
    /// order, the map's capacity = live; its last `params.spares` go to the
    /// host now (their slots free) and its first `pinned[l]` are never a
    /// victim; its tier experts stay where they are. A layer with no card
    /// slot takes no part. `view` and the stacks stay allocated and in place
    /// until the machine is dropped and its copy stream has drained.
    ///
    /// Refused by name: a layer whose stage card capacity is not its live
    /// count, a layer with card slots fewer than `pinned + spares + 1`, a
    /// delay of 0, a pinned list of another length, a view of another
    /// length, a `top_k` of 0 or past [`TALLY_TOP_K`], a part not a whole
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
        if cfg.params.delay == 0 {
            return Err(GpuError::shape(
                WHAT,
                "a live delay of 0: a flip's copy needs a pass to land in",
            ));
        }
        if cfg.top_k == 0 || cfg.top_k > TALLY_TOP_K {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "top-{}: a tally holds 1 to {TALLY_TOP_K} ids a row",
                    cfg.top_k
                ),
            ));
        }
        if cfg.pinned.len() != layers.len() || view.len() != layers.len() * n_expert {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} pinned counts and a view of {} words for layers {layers:?} of {n_expert} \
                     experts",
                    cfg.pinned.len(),
                    view.len()
                ),
            ));
        }
        // A ring slot holds any layer's expert, so it is sized by the widest:
        // the layers whose parts differ (a layer's stacks of another type)
        // stage at their own offsets and never fill the slot.
        let mut slot_bytes = 0usize;
        let mut parts_named = false;
        for l in layers.clone() {
            let parts = source.part_bytes(l);
            if parts.is_empty() {
                continue;
            }
            parts_named = true;
            if parts.iter().any(|&b| b == 0 || b % 4 != 0) {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l}: expert parts of {parts:?} bytes: each a nonzero multiple \
                             of 4"
                    ),
                ));
            }
            slot_bytes = slot_bytes.max(parts.iter().sum());
        }
        if !parts_named {
            return Err(GpuError::shape(
                WHAT,
                "no layer of the map names expert parts: the source holds no stacks the machine \
                 could move",
            ));
        }
        let layout = SwapMachine::layout(slots, &cfg.pinned, cfg.params.spares)?;
        let mut unrouted = Vec::with_capacity(cfg.unrouted.len());
        for &l in &cfg.unrouted {
            let cap = match layers.contains(&l) {
                true => slots.capacity(l)?,
                false => 0,
            };
            if !layers.contains(&l) || cap != 0 {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l} listed as routed by no pass: it must be a layer of the map \
                         {layers:?} with no card slot, and holds {cap}"
                    ),
                ));
            }
            unrouted.push(l - layers.start);
        }
        let rereads = Rereads::default();
        for &(l, id) in &layout.freed {
            source.prepare_victim(l, id)?;
            host_serves(&*source, &rereads, WHAT, l, id, || {
                "a spare slot's expert".to_string()
            })?;
        }
        let rule = SwapMachine::rule_of(&layout, &cfg, n_expert, &unrouted)?;
        let events = (0..layers.len() * cfg.params.spares)
            .map(|_| ctx.new_event(None))
            .collect::<Result<Vec<_>, _>>()?;
        let ledger = SlotLedger::of_map(slots)?;
        let copy = crate::role_stream(ctx, crate::StreamRole::Background)?;
        let boundary_event = ctx.new_event(None)?;
        let call_landed = (0..layers.len())
            .map(|_| ctx.new_event(None))
            .collect::<Result<Vec<_>, _>>()?;
        let call_read = (0..layers.len())
            .map(|_| ctx.new_event(None))
            .collect::<Result<Vec<_>, _>>()?;
        // The staging thread starts last: from here the machine's drop owns
        // it, and nothing can fail between.
        let Staging { shared, tx, thread } =
            SwapMachine::start_staging(ctx, source, slot_bytes, &cfg, rereads)?;
        let mut m = SwapMachine {
            rule,
            ledger,
            rule_shape: Shape {
                experts: n_expert,
                top_k: cfg.top_k,
                max_rows: cfg.max_rows,
            },
            layers,
            n_expert,
            view: view.cu_deviceptr(),
            copy,
            boundary_event,
            copy_waits_boundary: false,
            free: (0..events.len()).rev().collect(),
            event_job: vec![0; events.len()],
            events,
            jobs_issued: 0,
            planned: None,
            ahead: None,
            end_us: 0,
            kept: KeptRows::prefix(0),
            ops: Vec::with_capacity(BATCH_OPS),
            changed: Vec::new(),
            broken: None,
            call_landed,
            call_read,
            call: None,
            picks: Vec::new(),
            tx: Some(tx),
            thread: Some(thread),
            shared,
        };
        for &(l, id) in &layout.freed {
            let s = slots.evict(l, id)?;
            if let Slot::Card(s) = s {
                m.ledger.row_mut(l)[s as usize] = SlotState::Spare;
            }
            m.changed.push((l, id, HOST));
        }
        m.write_changed(stream)?;
        drain_within(
            stream,
            m.shared.deadline,
            "SwapMachine::new: the engine stream after the spare slots' words",
        )?;
        Ok(m)
    }

    /// The seed, tier and freed experts of each layer of `slots`, `pinned`
    /// and `spares` a layer; the load's map shapes [`SwapMachine::new`]
    /// refuses are refused here.
    fn layout(slots: &SlotMap, pinned: &[usize], spares: usize) -> Result<Layout, GpuError> {
        const WHAT: &str = "SwapMachine::new";
        let layers = slots.layers();
        let n_expert = slots.n_expert();
        let mut out = Layout {
            order: Vec::with_capacity(layers.len()),
            tier: Vec::with_capacity(layers.len()),
            pinned: Vec::with_capacity(layers.len()),
            freed: Vec::new(),
        };
        for (i, l) in layers.enumerate() {
            let (cap, live) = (slots.capacity(l)?, slots.on_card(l)?);
            if cap != live {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l}: {live} live experts in {cap} slots; the load's map is full"
                    ),
                ));
            }
            let mut order = vec![HOST; cap];
            let mut tier = Vec::new();
            for id in 0..n_expert as u32 {
                match slots.slot(l, id) {
                    Some(Slot::Card(s)) => order[s as usize] = id,
                    Some(Slot::Tier { .. }) => tier.push(id),
                    Some(Slot::Host) | None => {}
                }
            }
            out.tier.push(tier);
            if cap == 0 {
                out.order.push(Vec::new());
                out.pinned.push(0);
                continue;
            }
            if cap < pinned[i] + spares + 1 {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "layer {l}: {cap} card slots for {} pinned, {spares} spare and one that \
                         moves",
                        pinned[i]
                    ),
                ));
            }
            out.freed
                .extend(order[cap - spares..].iter().map(|&id| (l, id)));
            out.order.push(order);
            out.pinned.push(pinned[i]);
        }
        Ok(out)
    }

    /// The rule over `layout`: each layer's seed its slot order, its card set
    /// the seed less its spares, its tier experts away, the layers
    /// `unrouted` (the rule's numbering) routed by no pass.
    fn rule_of(
        layout: &Layout,
        cfg: &MachineCfg,
        n_expert: usize,
        unrouted: &[usize],
    ) -> Result<SwapRule, GpuError> {
        let seed: Vec<&[u32]> = layout.order.iter().map(Vec::as_slice).collect();
        let away: Vec<&[u32]> = layout.tier.iter().map(Vec::as_slice).collect();
        let capacity: Vec<usize> = layout
            .order
            .iter()
            .map(|o| o.len().saturating_sub(cfg.params.spares))
            .collect();
        let shape = Shape {
            experts: n_expert,
            top_k: cfg.top_k,
            max_rows: cfg.max_rows,
        };
        SwapRule::new_placed(cfg.params, shape, &seed, &capacity, &layout.pinned, &away)
            .and_then(|r| r.with_unrouted(unrouted))
            .map_err(|e| rule_err("SwapMachine::new", e))
    }

    /// The ring, the staging words and the staging thread over `source`;
    /// `rereads`, the load's, for the first boundary to take.
    fn start_staging(
        ctx: &Arc<CudaContext>,
        source: Arc<dyn SwapSource>,
        slot_bytes: usize,
        cfg: &MachineCfg,
        rereads: Rereads,
    ) -> Result<Staging, GpuError> {
        let shared = Arc::new(Shared {
            source,
            ring: Ring::new(ctx, slot_bytes)?,
            words: MappedHost::new(
                ctx,
                2 * RING_SLOTS * WORD_STRIDE,
                "cuMemHostAlloc (swap words)",
            )?,
            window: Arc::new(AtomicU32::new(1)),
            due: AtomicU64::new(0),
            deadline: cfg.deadline,
            flush: AtomicBool::new(false),
            served: AtomicU64::new(0),
            stage_ns: AtomicU64::new(0),
            prepare_ns: AtomicU64::new(0),
            rereads,
            stop: AtomicBool::new(false),
            failed: Mutex::new(None),
        });
        let (tx, rx) = mpsc::channel::<Job>();
        let (bound_tx, bound_rx) = mpsc::channel::<Result<(), GpuError>>();
        let for_thread = Arc::clone(&shared);
        let ctx = Arc::clone(ctx);
        // A helper, not a plain spawn: the step thread that builds the machine
        // may be pinned to one cpu, and the staging copies would take turns
        // with the step on it. The thread binds the context first: the
        // source's calls on it may reach the driver.
        let (thread, _) =
            threads::helper::spawn_helper("swap-staging", staging_placement(), move || {
                let bound = ctx.bind_to_thread().map_err(GpuError::from);
                let ok = bound.is_ok();
                if bound_tx.send(bound).is_ok() && ok {
                    for_thread.run(&rx);
                }
            })
            .map_err(|e| GpuError::plan("SwapMachine::new: the staging thread", e))?;
        match bound_rx.recv() {
            Ok(Ok(())) => Ok(Staging { shared, tx, thread }),
            // The thread is ending: joined here, so its reference to the
            // shared state goes before this one.
            Ok(Err(e)) => {
                let _ = thread.join();
                Err(e)
            }
            Err(_) => {
                let _ = thread.join();
                Err(GpuError::protocol(
                    "SwapMachine::new",
                    "the staging thread ended before it bound the context",
                ))
            }
        }
    }

    /// The host leg's wait window word: the staging thread copies while it
    /// is nonzero. The step port holds it open while its pool waits for a
    /// go, so the staging memcpy does not take DRAM from the host leg; a
    /// machine nobody gates leaves it at 1.
    #[must_use]
    pub fn window(&self) -> Arc<AtomicU32> {
        Arc::clone(&self.shared.window)
    }

    /// The machine's copy stream, for a gate that holds its copies to show
    /// the engine stream waits for them; nothing else enqueues on it.
    #[must_use]
    pub fn copy_stream(&self) -> &CudaStream {
        &self.copy
    }

    /// A tally of this machine's shape for a pass's routed ids
    /// ([`Tally::note`]): the step port's, which [`SwapMachine::end_pass`]
    /// folds.
    #[must_use]
    pub fn tally(&self) -> Tally {
        let r = self.rule_shape;
        Tally::new(
            self.layers.clone(),
            r.top_k,
            r.max_rows,
            self.rule.unrouted().to_vec(),
        )
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

    /// Whether a boundary has opened a pass that no [`SwapMachine::end_pass`]
    /// has ended yet: the next boundary needs that pass's kept rows.
    #[must_use]
    pub fn pass_open(&self) -> bool {
        self.planned == Some(self.rule.passes())
    }

    /// The boundary made ahead of its pass that no launch has taken yet
    /// ([`SwapMachine::boundary_ahead`]), when there is one.
    #[must_use]
    pub fn ahead(&self) -> Option<u64> {
        self.ahead
    }

    /// Refuse `what` while a boundary made ahead waits for its pass's
    /// launch: that pass has not run, so nothing may end it or open another
    /// boundary before it. One wording for every refusal of it.
    pub(crate) fn refuse_if_ahead(&self, what: &'static str) -> Result<(), GpuError> {
        match self.ahead {
            Some(b) => Err(GpuError::protocol(
                what,
                format!(
                    "boundary {b} was made ahead of its pass and no launch has taken it \
                     (SwapMachine::take_ahead): the pass it opened has not run"
                ),
            )),
            None => Ok(()),
        }
    }

    /// The error that broke the machine, when one has.
    #[must_use]
    pub fn broken(&self) -> Option<&str> {
        self.broken.as_deref()
    }

    /// Layer `layer`'s seed: its card set at the load less the spares, in
    /// slot order — the rule's ([`SwapRule::seed`]). A layer outside the map
    /// is refused by name.
    pub fn seed(&self, layer: usize) -> Result<Vec<u32>, GpuError> {
        const WHAT: &str = "SwapMachine::seed";
        let l = self.rule_layer(layer, WHAT)?;
        self.rule.seed(l).map_err(|e| rule_err(WHAT, e))
    }

    /// Layer `layer`'s pinned seed experts: its seed's first this many — the
    /// rule's ([`SwapRule::pinned`]). A layer outside the map is refused by
    /// name.
    pub fn pinned(&self, layer: usize) -> Result<usize, GpuError> {
        const WHAT: &str = "SwapMachine::pinned";
        let l = self.rule_layer(layer, WHAT)?;
        self.rule.pinned(l).map_err(|e| rule_err(WHAT, e))
    }

    /// Layer `layer`'s index in the rule, or the refusal of `what`.
    fn rule_layer(&self, layer: usize, what: &'static str) -> Result<usize, GpuError> {
        layer
            .checked_sub(self.layers.start)
            .filter(|&l| l < self.layers.len())
            .ok_or_else(|| {
                GpuError::shape(
                    what,
                    format!(
                        "layer {layer} is outside the machine's layers {:?}",
                        self.layers
                    ),
                )
            })
    }

    fn refuse_if_broken(&self, what: &'static str) -> Result<(), GpuError> {
        match &self.broken {
            Some(why) => Err(GpuError::protocol(
                what,
                format!("the machine is broken and refuses every call: {why}"),
            )),
            None => Ok(()),
        }
    }

    /// Refuse `what` while a prompt call is open: its map moves until
    /// [`SwapMachine::end_call`].
    fn refuse_in_call(&self, what: &'static str) -> Result<(), GpuError> {
        if self.call.is_some() {
            return Err(GpuError::state(what, "no prompt call open (end_call)"));
        }
        Ok(())
    }

    /// `r`, breaking the machine when it is an error: the call made a change
    /// before it failed. `at` names the call, made only on an error.
    fn after_change<T>(
        &mut self,
        r: Result<T, GpuError>,
        at: impl FnOnce() -> String,
    ) -> Result<T, GpuError> {
        if let Err(e) = &r
            && self.broken.is_none()
        {
            self.broken = Some(format!("{}: {e}", at()));
        }
        r
    }

    /// Let through every copy the copy stream holds behind a staging word:
    /// each ring slot's word gets the ticket after the last one issued to it.
    /// The wait is cyclic (`(i32)(word - ticket) >= 0`), so only a ticket just
    /// past the issued ones releases it; `u32::MAX` would not. For the drop
    /// only: the bytes those copies move land in slots no entry names.
    fn release_staging_waits(&self) -> Result<(), GpuError> {
        for n in self.jobs_issued..self.jobs_issued + RING_SLOTS as u64 {
            let (ring, ticket) = ring_ticket(n)?;
            self.shared.staged(ring).fetch_max(ticket, Ordering::AcqRel);
        }
        Ok(())
    }

    /// The staging thread's first failure as `what`'s error, which breaks the
    /// machine; `boundary` names where it was found.
    fn refuse_staging_failure(
        &mut self,
        what: &'static str,
        boundary: Option<u64>,
    ) -> Result<(), GpuError> {
        let failure = self
            .shared
            .failed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let Some(f) = failure else {
            return Ok(());
        };
        let e = StagingFailed {
            boundary,
            job: f.job,
            layer: f.layer,
            id: f.id,
            error: f.error,
        };
        self.broken = Some(e.to_string());
        Err(GpuError::plan(what, e))
    }

    /// End the pass the last boundary opened: fold `tally`'s `kept` rows
    /// into the rule and clear it. Refused by name, the machine
    /// unchanged: a pass no boundary opened, a tally of another shape or off
    /// ([`SwapMachine::tally`]), a kept row past the pass's rows, a kept row
    /// with a layer not every slot of which was noted, a row with some slots
    /// of a layer noted and not all, and a noted row routing an id twice or
    /// past the experts. A failure of the fold itself breaks the machine.
    pub fn end_pass(&mut self, tally: &mut Tally, kept: KeptRows) -> Result<(), GpuError> {
        const WHAT: &str = "SwapMachine::end_pass";
        let start = Instant::now();
        self.refuse_if_broken(WHAT)?;
        self.refuse_in_call(WHAT)?;
        self.refuse_if_ahead(WHAT)?;
        if self.planned != Some(self.rule.passes()) {
            return Err(GpuError::state(WHAT, "a boundary before the pass"));
        }
        let r = self.rule_shape;
        if !tally.on
            || tally.layers != self.layers
            || tally.top_k != r.top_k
            || tally.max_rows != r.max_rows
        {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a tally (on {}) of layers {:?}, top-{}, {} rows for a machine of layers \
                     {:?}, top-{}, {} rows",
                    tally.on,
                    tally.layers,
                    tally.top_k,
                    tally.max_rows,
                    self.layers,
                    r.top_k,
                    r.max_rows
                ),
            ));
        }
        if let Some(row) = kept.past(r.max_rows) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "row {row} of {} rows kept of a pass of at most {}",
                    kept.count(),
                    r.max_rows
                ),
            ));
        }
        self.check_tally(tally, kept, WHAT)?;
        let pass = self.rule.passes();
        let folded = self.fold(tally, kept, WHAT);
        self.after_change(folded, || format!("the end of pass {pass}"))?;
        tally.clear();
        self.kept = kept;
        self.end_us = micros(start);
        Ok(())
    }

    /// `tally` holds, for each kept row, every slot of every layer
    /// but the ones no pass routes, and for each other row every slot of a
    /// layer or none; every noted row names distinct ids below the experts.
    fn check_tally(&self, t: &Tally, kept: KeptRows, what: &'static str) -> Result<(), GpuError> {
        let n = self.layers.len();
        let full = t.full();
        for row in 0..t.max_rows {
            for (l, &unrouted) in self.rule.unrouted().iter().enumerate().take(n) {
                let cell = row * n + l;
                let mask = t.filled[cell];
                let layer = self.layers.start + l;
                if mask == 0 && (!kept.keeps(row) || unrouted) {
                    continue;
                }
                if mask != full {
                    let missing: Vec<usize> =
                        (0..t.top_k).filter(|&k| mask & (1 << k) == 0).collect();
                    return Err(GpuError::shape(
                        what,
                        format!(
                            "row {row} at layer {layer}: slots {missing:?} of top-{} not noted \
                             ({} rows kept)",
                            t.top_k,
                            kept.count()
                        ),
                    ));
                }
                let ids = &t.ids[cell * t.top_k..(cell + 1) * t.top_k];
                for (k, &id) in ids.iter().enumerate() {
                    if id as usize >= self.n_expert || ids[..k].contains(&id) {
                        return Err(GpuError::shape(
                            what,
                            format!(
                                "row {row} at layer {layer}: ids {ids:?}, one past the {} experts \
                                 or twice",
                                self.n_expert
                            ),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Every fully noted cell of `t` into the rule, then the pass ended at
    /// `kept`'s rows: the rule drops the rows outside the set.
    fn fold(&mut self, t: &Tally, kept: KeptRows, what: &'static str) -> Result<(), GpuError> {
        let n = self.layers.len();
        for row in 0..t.max_rows {
            for l in 0..n {
                let cell = row * n + l;
                if t.filled[cell] == 0 {
                    continue;
                }
                let ids = &t.ids[cell * t.top_k..(cell + 1) * t.top_k];
                self.rule
                    .observe(l, row, ids)
                    .map_err(|e| rule_err(what, e))?;
            }
        }
        self.rule.end_pass(kept).map_err(|e| rule_err(what, e))
    }

    /// The boundary before the next pass, on the engine stream `stream`, with
    /// the host map `slots`: the flips live here land (the stream waits for
    /// their copies, the card's copy and `slots` change), this boundary's
    /// event is recorded, then the rule plans and the new flips' copies are
    /// issued. Refused by name, the machine unchanged: a boundary twice with
    /// no pass between, one while a boundary made ahead waits for its launch
    /// ([`SwapMachine::boundary_ahead`]), a landing job not staged within the
    /// deadline, and a landing flip whose victim the host cannot serve from
    /// resident pages once its pages are read in again (a victim whose pages
    /// the page cache let go since the staging thread's prepare is read in
    /// on this thread, [`PassReport::rereads`]).
    /// Refused by name, the machine broken: a staging failure, a ledger that
    /// does not hold the flips the rule lands, and any error after the first
    /// change.
    pub fn boundary(
        &mut self,
        stream: &CudaStream,
        slots: &mut SlotMap,
    ) -> Result<PassReport, GpuError> {
        const WHAT: &str = "SwapMachine::boundary";
        let start = Instant::now();
        self.refuse_if_broken(WHAT)?;
        self.refuse_in_call(WHAT)?;
        self.refuse_if_ahead(WHAT)?;
        let b = self.rule.passes();
        if self.planned == Some(b) {
            return Err(GpuError::state(WHAT, "a pass since the last boundary"));
        }
        self.refuse_staging_failure(WHAT, Some(b))?;
        let landing = self.landing(b);
        let landing = self.after_change(landing, || format!("boundary {b}"))?;
        let mut report = PassReport {
            boundary: b,
            kept: if b == 0 { 0 } else { self.kept.count() },
            rows: if b == 0 { 0 } else { self.kept.mask() },
            end_us: std::mem::take(&mut self.end_us),
            ..PassReport::default()
        };
        for f in &landing {
            if !self.events[f.event].query()? {
                report.late += 1;
            }
            self.shared.due.fetch_max(f.job + 1, Ordering::AcqRel);
        }
        let t0 = Instant::now();
        for f in &landing {
            self.wait_staged(f, WHAT)?;
        }
        report.wait_us = micros(t0);
        self.refuse_staging_failure(WHAT, Some(b))?;
        let shared = &self.shared;
        for f in &landing {
            host_serves(
                &*shared.source,
                &shared.rereads,
                WHAT,
                f.layer,
                f.victim,
                || format!("the victim of expert {} into slot {}", f.admit, f.slot),
            )?;
        }
        let changed = self.land_and_plan(b, stream, slots, &landing, &mut report);
        self.after_change(changed, || format!("boundary {b}"))?;
        report.stage_us = self.shared.stage_ns.swap(0, Ordering::Relaxed) / 1000;
        report.prepare_us = self.shared.prepare_ns.swap(0, Ordering::Relaxed) / 1000;
        (report.rereads, report.reread_bytes, report.reread_us) = self.shared.rereads.take();
        self.planned = Some(b);
        report.boundary_us = micros(start);
        Ok(report)
    }

    /// [`SwapMachine::boundary`] made ahead of the pass it opens: after the
    /// pass before it was served and ended ([`SwapMachine::end_pass`]), while
    /// the engine stream may still run that pass's last kernels, so the
    /// boundary's host time runs under them. The engine stream's order is a
    /// boundary's at the launch: its waits, the card's words and its event
    /// follow the pass before it and precede the next launch. The pass it
    /// opens takes it exactly once ([`SwapMachine::take_ahead`]); until then
    /// every boundary, end of pass and prompt call is refused by name. Its
    /// refusals are [`SwapMachine::boundary`]'s.
    pub fn boundary_ahead(
        &mut self,
        stream: &CudaStream,
        slots: &mut SlotMap,
    ) -> Result<PassReport, GpuError> {
        let mut report = self.boundary(stream, slots)?;
        self.ahead = Some(report.boundary);
        report.ahead = true;
        Ok(report)
    }

    /// The launch of the pass a boundary made ahead opened takes it: the
    /// boundary, once. Refused by name, the machine unchanged: no boundary
    /// made ahead waiting (none was made, or a launch took it already — a
    /// second take), a broken machine and an open call.
    pub fn take_ahead(&mut self) -> Result<u64, GpuError> {
        const WHAT: &str = "SwapMachine::take_ahead";
        self.refuse_if_broken(WHAT)?;
        self.refuse_in_call(WHAT)?;
        let b = self.ahead.ok_or_else(|| {
            GpuError::state(
                WHAT,
                "a boundary made ahead of this pass (SwapMachine::boundary_ahead): none waits, \
                 so none was made or a launch took it already",
            )
        })?;
        if self.planned != Some(b) || self.rule.passes() != b {
            return Err(GpuError::protocol(
                WHAT,
                format!(
                    "boundary {b} was made ahead, and the machine stands at pass {} planned {:?}",
                    self.rule.passes(),
                    self.planned
                ),
            ));
        }
        self.ahead = None;
        Ok(b)
    }

    /// Wait, within the deadline, until landing flip `f`'s job is staged or
    /// its failure recorded.
    fn wait_staged(&self, f: &Landing, what: &'static str) -> Result<(), GpuError> {
        let (ring, ticket) = ring_ticket(f.job)?;
        let shared = &self.shared;
        let staged = || shared.staged(ring).load(Ordering::Acquire) >= ticket;
        match shared.wait_until(staged, Some(shared.deadline)) {
            Ok(true) => Ok(()),
            Ok(false) => Err(GpuError::protocol(what, "the staging thread has stopped")),
            Err(waited) => Err(GpuError::protocol(
                what,
                format!(
                    "layer {}: the victim {} of expert {} was not prepared for the host and its \
                     bytes staged in {waited:?} (deadline {:?})",
                    f.layer, f.victim, f.admit, shared.deadline
                ),
            )),
        }
    }

    /// The flips live at boundary `b` as the ledger holds them, in the rule's
    /// order. The ledger must hold each flip the rule lands there, once.
    fn landing(&self, b: u64) -> Result<Vec<Landing>, GpuError> {
        let mut out = Vec::new();
        let mut want = 0usize;
        for f in self.rule.in_flight().iter().filter(|f| f.live_at <= b) {
            want += 1;
            let layer = f.layer_of(self.layers.start);
            for (slot, st) in self.ledger.row(layer).unwrap_or(&[]).iter().enumerate() {
                if let SlotState::Filling { e, live_at, event } = *st
                    && e == f.admit
                    && live_at == f.live_at
                {
                    out.push(Landing {
                        layer,
                        slot: slot as u32,
                        admit: e,
                        victim: f.evict,
                        event,
                        job: self.event_job[event],
                    });
                }
            }
        }
        if out.len() != want {
            return Err(GpuError::protocol(
                "SwapMachine::boundary",
                format!(
                    "at boundary {b} the rule lands {want} flips and the slot ledger holds {} \
                     filling slots for them",
                    out.len()
                ),
            ));
        }
        Ok(out)
    }

    /// Boundary `b`'s changes, the first on: the engine stream's waits for
    /// the landing copies, the card's copy, the host map, the ledger, the
    /// boundary event, the rule's plan and the new flips' copies; then the
    /// host map, the ledger and the rule agree on every layer the rule
    /// landed.
    fn land_and_plan(
        &mut self,
        b: u64,
        stream: &CudaStream,
        slots: &mut SlotMap,
        landing: &[Landing],
        report: &mut PassReport,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "SwapMachine::boundary";
        for f in landing {
            stream.wait(&self.events[f.event])?;
            self.changed.push((f.layer, f.admit, f.slot));
            self.changed.push((f.layer, f.victim, HOST));
        }
        self.write_changed(stream)?;
        for f in landing {
            let Slot::Card(freed) = slots.evict(f.layer, f.victim)? else {
                return Err(GpuError::protocol(
                    WHAT,
                    format!(
                        "layer {}: victim {} was not on the stage card",
                        f.layer, f.victim
                    ),
                ));
            };
            slots.admit(f.layer, f.admit, Slot::Card(f.slot))?;
            let row = self.ledger.row_mut(f.layer);
            row[f.slot as usize] = SlotState::Live(f.admit);
            row[freed as usize] = SlotState::Spare;
            self.free.push(f.event);
        }
        report.landed = landing.len();
        self.boundary_event.record(stream)?;
        self.copy_waits_boundary = false;
        if b > 0 {
            let made: Vec<Flip> = self.rule.plan(b).map_err(|e| rule_err(WHAT, e))?.to_vec();
            let t0 = Instant::now();
            for f in &made {
                report.bytes += self.issue(slots, f)?;
            }
            report.issue_us = micros(t0);
            report.made = made.len();
        }
        for (i, f) in landing.iter().enumerate() {
            if landing[..i].iter().all(|g| g.layer != f.layer) {
                self.agree(f.layer, slots, WHAT)?;
            }
        }
        report.in_flight = self.rule.in_flight().len();
        Ok(())
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
        let bytes = self.copy_into(slots, l, f.admit, spare, Some(f.evict), Gate::Boundary)?;
        self.events[event].record(&self.copy)?;
        self.ledger.row_mut(l)[spare as usize] = SlotState::Filling {
            e: f.admit,
            live_at: f.live_at,
            event,
        };
        Ok(bytes)
    }

    /// Hand the staging thread layer `l`'s expert `id` as a job (preparing
    /// `victim` first), then enqueue on the copy stream, behind the wait for
    /// the slot's readers (`gate`) and the job's staging word, its copy into
    /// slot `spare` of each stack.
    /// Every stack's destination is resolved before anything is enqueued.
    /// The bytes it copies.
    fn copy_into(
        &mut self,
        slots: &SlotMap,
        l: usize,
        id: u32,
        spare: u32,
        victim: Option<u32>,
        gate: Gate,
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
        // Every copy waits on the copy stream for its staging. Behind a
        // closed window, with no flush, a job stages only once its flip is
        // due, which a later boundary of this host thread makes; so this
        // thread must never enqueue so much behind unstaged jobs that the
        // stream's queue fills and the enqueue blocks. A boundary issues at
        // most the flips the rule keeps in flight (one event each); a reset
        // stages whatever the window says. Past that the call is refused. A
        // prompt call's picks stage whatever the window says too, and wait,
        // within the deadline, until the staging thread has taken in all but
        // that many of them ([`SwapMachine::call_backlog`]).
        let waiting = self.jobs_issued - self.shared.served.load(Ordering::Acquire);
        if !self.shared.flush.load(Ordering::Acquire) && waiting >= self.events.len() as u64 {
            return Err(GpuError::protocol(
                WHAT,
                format!(
                    "layer {l} expert {id}: {waiting} copies already wait for staging with no \
                     flush, and a boundary issues at most the {} flips the rule keeps in flight",
                    self.events.len()
                ),
            ));
        }
        let source = Arc::clone(&self.shared.source);
        let parts = source.part_bytes(l);
        let dsts = (0..parts.len())
            .map(|part| source.dest(l, part, spare))
            .collect::<Result<Vec<_>, _>>()?;
        let n = self.jobs_issued;
        let (ring, ticket) = ring_ticket(n)?;
        let job = Job {
            n,
            layer: l,
            id,
            victim,
            ring,
            ticket,
        };
        let d = dispatch::send(self.tx.as_ref(), job, WHAT)?;
        self.jobs_issued += 1;
        let copy = Arc::clone(&self.copy);
        if gate == Gate::Boundary && !self.copy_waits_boundary {
            copy.wait(&self.boundary_event)?;
            self.copy_waits_boundary = true;
        }
        dispatch::wait_staged(&copy, &self.shared.words, &d)?;
        let drained = self.shared.words.dev_at((2 * d.ring() + 1) * WORD_STRIDE);
        let mut at = 0usize;
        for (part, (&len, &dst)) in parts.iter().zip(&dsts).enumerate() {
            self.shared
                .ring
                .copy_to_device(d.ring(), at, len, dst, &copy)?;
            source.convert(l, part, dst, &copy)?;
            at += len;
        }
        mem_batch(
            &copy,
            &mut [op_write(drained, d.ticket())],
            "swap: ring drained",
        )?;
        Ok(at as u64)
    }

    /// Layer `layer`'s host map, ledger and rule agree, else refused by name:
    /// the map's stage card set is the rule's card set, each `Live(e)` slot
    /// of the ledger is the map's slot of `e`, and the map names no other
    /// slot. Allocation-free.
    fn agree(&self, layer: usize, slots: &SlotMap, what: &'static str) -> Result<(), GpuError> {
        let l = self.rule_layer(layer, what)?;
        for id in 0..self.n_expert as u32 {
            let rule = self.rule.is_live(l, id).map_err(|e| rule_err(what, e))?;
            let map = matches!(slots.slot(layer, id), Some(Slot::Card(_)));
            if rule != map {
                return Err(GpuError::protocol(
                    what,
                    format!(
                        "layer {layer} expert {id}: on the stage card in the host map {map}, in \
                         the rule's card set {rule}"
                    ),
                ));
            }
        }
        let row = self.ledger.row(layer).unwrap_or(&[]);
        let mut live = 0usize;
        for (s, st) in row.iter().enumerate() {
            if let SlotState::Live(e) | SlotState::Landing { e, .. } = *st {
                live += 1;
                if slots.slot(layer, e) != Some(Slot::Card(s as u32)) {
                    return Err(GpuError::protocol(
                        what,
                        format!(
                            "layer {layer}: the ledger holds expert {e} live in slot {s}, the host \
                             map has it on {:?}",
                            slots.slot(layer, e)
                        ),
                    ));
                }
            }
        }
        let on_card = slots.on_card(layer)?;
        if live != on_card {
            return Err(GpuError::protocol(
                what,
                format!(
                    "layer {layer}: the host map names {on_card} stage card slots, the ledger \
                     holds {live} live"
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
    /// engine stream waiting for the copies; the host pages of the seed
    /// experts back on the card and of the cancelled flips' victims are
    /// released; the rule returns to its seed, and a boundary made ahead
    /// waiting for its launch goes with the pass it opened, which never ran.
    /// The caller's tally is its own to clear ([`Tally::clear`]). Refused by name, the machine unchanged:
    /// a copy stream that does not drain within the deadline. Refused by
    /// name, the machine broken: a staging failure, an expert sent to the
    /// host the source cannot serve from resident pages, and any error after
    /// the first change.
    pub fn reset(
        &mut self,
        stream: &CudaStream,
        slots: &mut SlotMap,
    ) -> Result<ResetReport, GpuError> {
        const WHAT: &str = "SwapMachine::reset";
        self.refuse_if_broken(WHAT)?;
        self.refuse_in_call(WHAT)?;
        self.drain_copies(WHAT)?;
        self.refuse_staging_failure(WHAT, None)?;
        let r = self.relayout(stream, slots);
        self.after_change(r, || "reset".to_string())
    }

    /// The reset's changes, the first on ([`SwapMachine::reset`]): the flips
    /// in flight cancelled, the admitted experts sent back to the host, the
    /// seed experts copied back onto the card, then the maps, the ledger and
    /// the rule at the seed.
    fn relayout(
        &mut self,
        stream: &CudaStream,
        slots: &mut SlotMap,
    ) -> Result<ResetReport, GpuError> {
        let mut report = ResetReport::default();
        let cancelled = self.cancel_in_flight(&mut report)?;
        let back = self.send_back(slots)?;
        let placed = self.copy_back(stream, slots, &back)?;
        self.commit(stream, slots, &placed, &cancelled, &mut report)?;
        Ok(report)
    }

    /// Every filling slot back to spare, its event free: the flips the rule
    /// has in flight, which the ledger must hold one a slot.
    fn cancel_in_flight(&mut self, report: &mut ResetReport) -> Result<Vec<Flip>, GpuError> {
        const WHAT: &str = "SwapMachine::reset";
        let cancelled: Vec<Flip> = self.rule.in_flight().to_vec();
        for l in self.layers.clone() {
            for st in self.ledger.row_mut(l).iter_mut() {
                if let SlotState::Filling { event, .. } = *st {
                    *st = SlotState::Spare;
                    self.free.push(event);
                    report.cancelled += 1;
                }
            }
        }
        if report.cancelled != cancelled.len() {
            return Err(GpuError::protocol(
                WHAT,
                format!(
                    "the rule has {} flips in flight and the slot ledger {} filling slots",
                    cancelled.len(),
                    report.cancelled
                ),
            ));
        }
        Ok(cancelled)
    }

    /// Every live expert not in its layer's seed to the host (prepared, and
    /// refused by name unless host-resident), its slot spare; the seed
    /// experts not live, per layer, which the reset copies back.
    fn send_back(&mut self, slots: &mut SlotMap) -> Result<Vec<(usize, u32)>, GpuError> {
        const WHAT: &str = "SwapMachine::reset";
        let mut back = Vec::new();
        for (i, l) in self.layers.clone().enumerate() {
            let seed = self.rule.seed(i).map_err(|e| rule_err(WHAT, e))?;
            let live: Vec<u32> = (0..self.n_expert as u32)
                .filter(|&id| matches!(slots.slot(l, id), Some(Slot::Card(_))))
                .collect();
            for &id in live.iter().filter(|id| !seed.contains(id)) {
                let shared = &self.shared;
                shared.source.prepare_victim(l, id)?;
                host_serves(&*shared.source, &shared.rereads, WHAT, l, id, || {
                    "an admitted expert the reset sends back".to_string()
                })?;
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
        Ok(back)
    }

    /// The copies of `back`'s seed experts into spare slots, reserved, behind
    /// this call's boundary event on `stream`, drained; the placements.
    fn copy_back(
        &mut self,
        stream: &CudaStream,
        slots: &SlotMap,
        back: &[(usize, u32)],
    ) -> Result<Vec<(usize, u32, u32)>, GpuError> {
        const WHAT: &str = "SwapMachine::reset";
        self.boundary_event.record(stream)?;
        self.copy_waits_boundary = false;
        // The seed's copies stage as they are issued, whatever the window
        // says, and go out a ring's worth at a time, each batch drained
        // within the deadline: hundreds of them behind unstaged copies would
        // fill the copy stream's queue while this thread still enqueues, a
        // block inside the driver with no bound. A copy past the ring waits
        // on a ring slot's earlier copy anyway, so a batch loses nothing. On
        // an error in between the flush stays on, which only stages sooner.
        self.shared.flush.store(true, Ordering::Release);
        let mut placed = Vec::with_capacity(back.len());
        for &(l, id) in back {
            if !placed.is_empty() && placed.len() % RING_SLOTS == 0 {
                self.drain_copies(WHAT)?;
            }
            let spare = self.ledger.spare(l).ok_or_else(|| {
                GpuError::protocol(
                    WHAT,
                    format!("layer {l}: no free slot for seed expert {id}"),
                )
            })?;
            self.copy_into(slots, l, id, spare, None, Gate::Boundary)?;
            self.ledger.row_mut(l)[spare as usize] = SlotState::Reserved(id);
            placed.push((l, id, spare));
        }
        self.drain_copies(WHAT)?;
        self.shared.flush.store(false, Ordering::Release);
        self.refuse_staging_failure(WHAT, None)?;
        Ok(placed)
    }

    /// The copied seed experts live in the host map, the ledger and the
    /// card's copy, the host pages of those and of the cancelled flips'
    /// victims still on the card released, the rule at its seed; then every
    /// layer's diff from the seed counted and its maps agreeing.
    fn commit(
        &mut self,
        stream: &CudaStream,
        slots: &mut SlotMap,
        placed: &[(usize, u32, u32)],
        cancelled: &[Flip],
        report: &mut ResetReport,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "SwapMachine::reset";
        for &(l, id, spare) in placed {
            slots.admit(l, id, Slot::Card(spare))?;
            self.ledger.row_mut(l)[spare as usize] = SlotState::Live(id);
            self.changed.push((l, id, spare));
            report.dropped_bytes += self.shared.source.release_host(l, id)?;
        }
        for f in cancelled {
            let l = f.layer_of(self.layers.start);
            if matches!(slots.slot(l, f.evict), Some(Slot::Card(_))) {
                report.dropped_bytes += self.shared.source.release_host(l, f.evict)?;
            }
        }
        report.copies = placed.len();
        self.write_changed(stream)?;
        self.rule.reset();
        self.planned = None;
        self.ahead = None;
        self.kept = KeptRows::prefix(0);
        self.end_us = 0;
        for (i, l) in self.layers.clone().enumerate() {
            let seed = self.rule.seed(i).map_err(|e| rule_err(WHAT, e))?;
            let live: Vec<u32> = (0..self.n_expert as u32)
                .filter(|&id| matches!(slots.slot(l, id), Some(Slot::Card(_))))
                .collect();
            report.diff += live.iter().filter(|id| !seed.contains(id)).count();
            report.diff += seed.iter().filter(|id| !live.contains(id)).count();
            self.agree(l, slots, WHAT)?;
        }
        Ok(())
    }

    /// Open a prompt call on the engine stream `stream`, inside the pass the
    /// last boundary opened: the rule makes no plan and every boundary, end
    /// of pass and reset is refused until [`SwapMachine::end_call`]; the
    /// staging thread stages every job whatever the window says; each
    /// layer's reader event is recorded here, after every pass before the
    /// call. Refused by name, the machine unchanged: a call open already and
    /// no pass open (a boundary first).
    pub fn begin_call(&mut self, stream: &CudaStream, cfg: CallCfg) -> Result<(), GpuError> {
        const WHAT: &str = "SwapMachine::begin_call";
        self.refuse_if_broken(WHAT)?;
        self.refuse_in_call(WHAT)?;
        self.refuse_if_ahead(WHAT)?;
        if !self.pass_open() {
            return Err(GpuError::state(WHAT, "a boundary before the call"));
        }
        let mut start = Vec::with_capacity(self.layers.len());
        for i in 0..self.layers.len() {
            let live = self.rule.live(i).map_err(|e| rule_err(WHAT, e))?;
            start.push(live.collect());
        }
        for ev in &self.call_read {
            ev.record(stream)?;
        }
        let flush_was = self.shared.flush.swap(true, Ordering::AcqRel);
        self.call = Some(Call {
            cfg,
            start,
            report: CallReport::default(),
            flush_was,
        });
        Ok(())
    }

    /// The layer's landed event of the open call: the engine stream waits for
    /// it before any read of the slots the layer's last pick admitted into.
    /// A layer outside the map is refused by name.
    pub fn call_landed(&self, layer: usize) -> Result<&CudaEvent, GpuError> {
        let i = self.rule_layer(layer, "SwapMachine::call_landed")?;
        Ok(&self.call_landed[i])
    }

    /// Layer `layer`'s slots have had their last read of the call so far on
    /// the engine stream `stream`: the stream waits for the layer's landed
    /// event, which turns its landing slots `Live`, and records the layer's
    /// reader event, which the layer's next pick's copies wait for. Refused
    /// by name: no call open, a layer outside the map.
    pub fn call_reader(&mut self, layer: usize, stream: &CudaStream) -> Result<(), GpuError> {
        const WHAT: &str = "SwapMachine::call_reader";
        self.refuse_if_broken(WHAT)?;
        if self.call.is_none() {
            return Err(GpuError::state(WHAT, "a prompt call open (begin_call)"));
        }
        let i = self.rule_layer(layer, WHAT)?;
        let r = self.land_layer(layer, stream);
        self.after_change(r, || format!("the call's reader of layer {layer}"))?;
        let r = self.call_read[i].record(stream).map_err(GpuError::from);
        self.after_change(r, || format!("the call's reader of layer {layer}"))
    }

    /// Layer `layer`'s landing slots, if any: the engine stream `stream`
    /// waits for the layer's landed event and they turn `Live`.
    fn land_layer(&mut self, layer: usize, stream: &CudaStream) -> Result<(), GpuError> {
        if self.ledger.landing(layer).is_none() {
            return Ok(());
        }
        stream.wait(&self.call_landed[layer - self.layers.start])?;
        self.ledger.land(layer);
        Ok(())
    }

    /// The most of a call's jobs the staging thread has not taken in when a
    /// pick issues another: the flips a boundary may keep in flight, and a
    /// ring's worth at least, so the copy stream never holds so many copies
    /// behind unstaged jobs that an enqueue blocks inside the driver.
    fn call_backlog(&self) -> u64 {
        self.events.len().max(RING_SLOTS) as u64
    }

    /// Wait, within the deadline, until the staging thread has taken in all
    /// but [`SwapMachine::call_backlog`] of the jobs issued.
    fn wait_backlog(&self, what: &'static str) -> Result<(), GpuError> {
        let (bound, issued) = (self.call_backlog(), self.jobs_issued);
        let shared = &self.shared;
        let fits = || issued - shared.served.load(Ordering::Acquire) < bound;
        match shared.wait_until(fits, Some(shared.deadline)) {
            Ok(true) => Ok(()),
            Ok(false) => Err(GpuError::protocol(what, "the staging thread has stopped")),
            Err(waited) => Err(GpuError::protocol(
                what,
                format!(
                    "the staging thread took in no job of the call's last {bound} in {waited:?} \
                     (deadline {:?})",
                    shared.deadline
                ),
            )),
        }
    }

    /// The open call's pick at layer `layer` from `counts`, the ids a batch
    /// of the call routes there (one count per expert of the layer), at most
    /// `cap` experts wanted ([`SwapRule::call_pick`], the call's floor): on
    /// the engine stream `stream` and the host map `slots`, each admitted
    /// expert's job goes to the staging thread and its copy onto the copy
    /// stream behind the layer's reader event, the layer's landed event
    /// after the last; then each victim goes to the host and each admitted
    /// expert to its victim's slot, `Landing`, in the host map, the rule and
    /// the card's copy of the layer's words (written on `stream`). Refused by
    /// name, the machine unchanged: no call open, a layer outside the map,
    /// counts of another length, a layer whose last pick no reader has
    /// waited for ([`SwapMachine::call_reader`]), and a victim the host
    /// cannot serve from resident pages once its pages are read in again.
    /// Any error after the first job is sent breaks the machine.
    pub fn call_pick(
        &mut self,
        stream: &CudaStream,
        slots: &mut SlotMap,
        layer: usize,
        counts: &[u32],
        cap: usize,
    ) -> Result<CallPick, GpuError> {
        const WHAT: &str = "SwapMachine::call_pick";
        let t0 = Instant::now();
        self.refuse_if_broken(WHAT)?;
        let floor = match &self.call {
            Some(c) => c.cfg.floor,
            None => return Err(GpuError::state(WHAT, "a prompt call open (begin_call)")),
        };
        let i = self.rule_layer(layer, WHAT)?;
        if let Some((s, st)) = self.ledger.landing(layer) {
            return Err(GpuError::protocol(
                WHAT,
                format!(
                    "layer {layer}: slot {s} is {st:?} from its last pick, which no reader has \
                     waited for (call_reader)"
                ),
            ));
        }
        let mut picks = std::mem::take(&mut self.picks);
        let kept = self
            .rule
            .call_pick(i, counts, cap, floor, &mut picks)
            .map_err(|e| rule_err(WHAT, e));
        let kept = match kept {
            Ok(k) => k,
            Err(e) => {
                self.picks = picks;
                return Err(e);
            }
        };
        let mut report = CallPick {
            layer,
            kept,
            counts: counts_digest(counts),
            ..CallPick::default()
        };
        let checked = self.check_victims(slots, layer, &picks, WHAT);
        if let Err(e) = checked {
            self.picks = picks;
            return Err(e);
        }
        let moved = self.move_pool(stream, slots, layer, &picks, &mut report);
        let moved = self.after_change(moved, || format!("the call's pick of layer {layer}"));
        report.admitted = picks.len();
        self.picks = picks;
        moved?;
        report.pick_us = micros(t0);
        if let Some(c) = self.call.as_mut() {
            let r = &mut c.report;
            r.picks += usize::from(report.admitted > 0);
            r.admitted += report.admitted;
            r.bytes += report.bytes;
            r.pick_us += report.pick_us;
            r.backlog_us += report.backlog_us;
        }
        Ok(report)
    }

    /// Every victim of `picks` is on the stage card in the host map and
    /// served by the host from resident pages — read in again here when the
    /// page cache let its pages go, since nothing prepared it before the
    /// pick — else refused by name as `what`.
    fn check_victims(
        &self,
        slots: &SlotMap,
        layer: usize,
        picks: &[Flip],
        what: &'static str,
    ) -> Result<(), GpuError> {
        for f in picks {
            if !matches!(slots.slot(layer, f.evict), Some(Slot::Card(_))) {
                return Err(GpuError::protocol(
                    what,
                    format!(
                        "layer {layer}: victim {} of expert {} is on {:?} in the host map, not \
                         the stage card",
                        f.evict,
                        f.admit,
                        slots.slot(layer, f.evict)
                    ),
                ));
            }
            let shared = &self.shared;
            host_serves(
                &*shared.source,
                &shared.rereads,
                what,
                layer,
                f.evict,
                || format!("the victim of expert {} in a call's pick", f.admit),
            )?;
        }
        Ok(())
    }

    /// A pick's changes, the first on ([`SwapMachine::call_pick`]).
    fn move_pool(
        &mut self,
        stream: &CudaStream,
        slots: &mut SlotMap,
        layer: usize,
        picks: &[Flip],
        report: &mut CallPick,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "SwapMachine::call_pick";
        if picks.is_empty() {
            return Ok(());
        }
        let i = layer - self.layers.start;
        let mut to = Vec::with_capacity(picks.len());
        for f in picks {
            let Some(Slot::Card(s)) = slots.slot(layer, f.evict) else {
                return Err(GpuError::protocol(
                    WHAT,
                    format!("layer {layer}: victim {} left the stage card", f.evict),
                ));
            };
            to.push(s);
        }
        self.copy.wait(&self.call_read[i])?;
        for (f, &s) in picks.iter().zip(&to) {
            let t = Instant::now();
            self.wait_backlog(WHAT)?;
            report.backlog_us += micros(t);
            report.bytes += self.copy_into(slots, layer, f.admit, s, None, Gate::Waited)?;
        }
        self.call_landed[i].record(&self.copy)?;
        let moves: Vec<Flip> = picks.iter().map(|f| Flip { layer: i, ..*f }).collect();
        self.rule.settle(&moves).map_err(|e| rule_err(WHAT, e))?;
        for (f, &s) in picks.iter().zip(&to) {
            self.ledger.to_landing(layer, s, f.evict, f.admit, i)?;
            slots.evict(layer, f.evict)?;
            slots.admit(layer, f.admit, Slot::Card(s))?;
            self.changed.push((layer, f.evict, HOST));
            self.changed.push((layer, f.admit, s));
        }
        self.write_changed(stream)?;
        self.agree(layer, slots, WHAT)
    }

    /// End the open call on the engine stream `stream` with the host map
    /// `slots`: every landing slot's landed event waited for and the slot
    /// `Live`; then, when `kept`, the placement stays for the passes after
    /// the call, else every layer returns to the set it started the call
    /// with, the experts the call sent to the host copied back behind a
    /// boundary event of `stream` into the slots of those it admitted, which
    /// go back to the host. The staging window gates the staging thread
    /// again. Refused by name, the machine unchanged: no call open. Any
    /// error after the first change breaks the machine.
    pub fn end_call(
        &mut self,
        stream: &CudaStream,
        slots: &mut SlotMap,
        kept: bool,
    ) -> Result<CallReport, GpuError> {
        const WHAT: &str = "SwapMachine::end_call";
        let t0 = Instant::now();
        self.refuse_if_broken(WHAT)?;
        let Some(call) = self.call.take() else {
            return Err(GpuError::state(WHAT, "a prompt call open (begin_call)"));
        };
        let r = self.close_call(stream, slots, &call, kept);
        let restored = self.after_change(r, || "the end of a call".to_string())?;
        self.shared.flush.store(call.flush_was, Ordering::Release);
        Ok(CallReport {
            kept,
            restored,
            end_us: micros(t0),
            ..call.report
        })
    }

    /// The end's changes ([`SwapMachine::end_call`]); the experts restored.
    fn close_call(
        &mut self,
        stream: &CudaStream,
        slots: &mut SlotMap,
        call: &Call,
        kept: bool,
    ) -> Result<usize, GpuError> {
        const WHAT: &str = "SwapMachine::end_call";
        for l in self.layers.clone() {
            self.land_layer(l, stream)?;
        }
        if kept {
            for l in self.layers.clone() {
                self.agree(l, slots, WHAT)?;
            }
            return Ok(0);
        }
        self.restore(stream, slots, &call.start)
    }

    /// Every layer back to `start`, its card set at the call's start (the
    /// rule's layer numbering, ascending ids): per layer the experts the call
    /// admitted and the ones it sent to the host pair up in ascending ids;
    /// each admitted one goes to the host, and its slot takes the other,
    /// copied behind a boundary event of `stream` a ring's worth at a time.
    /// The experts copied back.
    fn restore(
        &mut self,
        stream: &CudaStream,
        slots: &mut SlotMap,
        start: &[Vec<u32>],
    ) -> Result<usize, GpuError> {
        const WHAT: &str = "SwapMachine::end_call";
        let mut back = Vec::new();
        let mut undo = Vec::new();
        for (i, l) in self.layers.clone().enumerate() {
            let now: Vec<u32> = self.rule.live(i).map_err(|e| rule_err(WHAT, e))?.collect();
            let was = &start[i];
            let came: Vec<u32> = now.iter().copied().filter(|id| !was.contains(id)).collect();
            let went: Vec<u32> = was.iter().copied().filter(|id| !now.contains(id)).collect();
            if came.len() != went.len() {
                return Err(GpuError::protocol(
                    WHAT,
                    format!(
                        "layer {l}: the call admitted {came:?} and sent {went:?} to the host, \
                         not one for one"
                    ),
                ));
            }
            for (&a, &v) in came.iter().zip(&went) {
                let shared = &self.shared;
                host_serves(&*shared.source, &shared.rereads, WHAT, l, a, || {
                    "an expert a call admitted".to_string()
                })?;
                let Slot::Card(s) = slots.evict(l, a)? else {
                    return Err(GpuError::protocol(
                        WHAT,
                        format!("layer {l}: expert {a} the call admitted is off the stage card"),
                    ));
                };
                self.ledger.row_mut(l)[s as usize] = SlotState::Reserved(v);
                self.changed.push((l, a, HOST));
                back.push((l, v, s));
                undo.push(Flip {
                    layer: i,
                    admit: v,
                    evict: a,
                    live_at: self.rule.passes(),
                });
            }
        }
        self.write_changed(stream)?;
        self.boundary_event.record(stream)?;
        self.copy_waits_boundary = false;
        for (n, &(l, v, s)) in back.iter().enumerate() {
            if n > 0 && n % RING_SLOTS == 0 {
                self.drain_copies(WHAT)?;
            }
            self.copy_into(slots, l, v, s, None, Gate::Boundary)?;
        }
        self.drain_copies(WHAT)?;
        self.refuse_staging_failure(WHAT, None)?;
        for &(l, v, s) in &back {
            slots.admit(l, v, Slot::Card(s))?;
            self.ledger.row_mut(l)[s as usize] = SlotState::Live(v);
            self.changed.push((l, v, s));
        }
        self.write_changed(stream)?;
        self.rule.settle(&undo).map_err(|e| rule_err(WHAT, e))?;
        for l in self.layers.clone() {
            self.agree(l, slots, WHAT)?;
        }
        Ok(back.len())
    }

    /// The open call's floor from its next pick on: the least count an
    /// admitted expert has in a pick's counts ([`CallCfg::floor`]). Refused
    /// by name with no call open.
    pub fn set_call_floor(&mut self, floor: u32) -> Result<(), GpuError> {
        match self.call.as_mut() {
            Some(c) => {
                c.cfg.floor = floor;
                Ok(())
            }
            None => Err(GpuError::state(
                "SwapMachine::set_call_floor",
                "a prompt call open (begin_call)",
            )),
        }
    }

    /// Whether a prompt call is open.
    #[must_use]
    pub fn call_open(&self) -> bool {
        self.call.is_some()
    }

    /// Stage every queued job whatever the window says and wait, within the
    /// deadline, for the copy stream to drain; past it the refusal of
    /// `what` names the wait. The flush is as it was after.
    fn drain_copies(&self, what: &'static str) -> Result<(), GpuError> {
        let was = self.shared.flush.swap(true, Ordering::AcqRel);
        let r = poll_drained(&self.copy, self.shared.deadline).map_err(|e| match e {
            Drain::Driver(e) => e,
            Drain::Late(waited) => GpuError::protocol(
                what,
                format!(
                    "the copy stream did not drain in {waited:?} (deadline {:?})",
                    self.shared.deadline
                ),
            ),
        });
        self.shared.flush.store(was, Ordering::Release);
        r
    }

    /// The bound on every host wait the machine makes
    /// ([`MachineCfg::deadline`]).
    #[must_use]
    pub fn deadline(&self) -> Duration {
        self.shared.deadline
    }
}

/// FNV-1a 64 over `counts`, each a little-endian `u32`: a pick's input
/// ([`CallPick::counts`]).
fn counts_digest(counts: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &c in counts {
        for b in c.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

impl Drop for SwapMachine {
    fn drop(&mut self) {
        let deadline = self.shared.deadline;
        self.shared.stop.store(true, Ordering::Release);
        self.tx = None;
        let mut joined = true;
        if let Some(t) = self.thread.take() {
            let t0 = Instant::now();
            while !t.is_finished() && t0.elapsed() <= deadline {
                std::thread::sleep(Duration::from_micros(200));
            }
            if t.is_finished() {
                // A panic outside a job has nowhere to go on the drop path.
                let _ = t.join();
            } else {
                joined = false;
            }
        }
        // A copy still waiting for a ticket the thread never published is let
        // through, so the stream can drain before the ring is freed.
        let released = self.release_staging_waits().is_ok();
        // The copy stream waits on no host word now, only on boundary events
        // of the engine stream. Past the deadline (an engine stream held by
        // something else) the ring and the words are leaked, never freed
        // under a copy that may still read them. A thread still inside the
        // source past the deadline is left to finish on its own with its
        // reference, and the shared state is leaked too: its last owner would
        // free pinned pages and the source on that thread, inside the driver
        // beside whatever this process runs next.
        let ring_bytes = (RING_SLOTS * self.shared.ring.slot_bytes) as u64;
        let words_bytes = (2 * RING_SLOTS * WORD_STRIDE) as u64;
        let leak = if !joined {
            Some(Leak {
                reason: LeakReason::Join,
                code: None,
                ring_bytes,
                words_bytes,
            })
        } else if !released {
            Some(Leak {
                reason: LeakReason::Release,
                code: None,
                ring_bytes,
                words_bytes,
            })
        } else {
            match poll_drained(&self.copy, deadline) {
                Ok(()) => None,
                Err(Drain::Late(_)) => Some(Leak {
                    reason: LeakReason::Drain,
                    code: None,
                    ring_bytes,
                    words_bytes,
                }),
                Err(Drain::Driver(e)) => Some(Leak::fault(&e, ring_bytes, words_bytes)),
            }
        };
        if let Some(leak) = leak {
            std::mem::forget(Arc::clone(&self.shared));
            match LEAK_SINK.get() {
                Some(sink) => sink(&leak),
                None => eprintln!(
                    "SwapMachine drop: leaked the staging ring ({} B), the staging words ({} B) \
                     and the source: {}",
                    leak.ring_bytes,
                    leak.words_bytes,
                    leak.reason.word()
                ),
            }
        }
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

/// The machine's one decision that the host serves layer `layer`'s expert
/// `id` from resident pages, made on the machine's thread for every expert
/// it sends to the host: resident now, it is; else its pages are read in
/// again ([`SwapSource::prepare_victim`]) and it is asked once more — a page
/// the page cache let go since the load or the staging thread's prepare,
/// read back at a page fault's cost once and counted in `rereads`. Still not
/// resident (the host set does not hold it), the refusal of `what` names
/// `which` expert it is. A free function, not a method a source could
/// override: the decision and its count are the machine's.
fn host_serves(
    source: &dyn SwapSource,
    rereads: &Rereads,
    what: &'static str,
    layer: usize,
    id: u32,
    which: impl FnOnce() -> String,
) -> Result<(), GpuError> {
    if source.host_resident(layer, id)? {
        return Ok(());
    }
    let t0 = Instant::now();
    source.prepare_victim(layer, id)?;
    if !source.host_resident(layer, id)? {
        return Err(not_resident(what, layer, id, &which()));
    }
    let bytes: usize = source.part_bytes(layer).iter().sum();
    rereads.note(bytes as u64, nanos(t0));
    Ok(())
}

fn not_resident(what: &'static str, layer: usize, id: u32, which: &str) -> GpuError {
    GpuError::protocol(
        what,
        format!(
            "layer {layer} expert {id}, {which}, is not host-resident after its pages were read \
             in again: the host would serve it from the file (a page fault a step), so the flip \
             is refused"
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::super::slots::{HOST, SlotMap};
    use super::{Residency, SlotLedger, SlotState, Tally};

    /// The parse takes exactly what the lever's grammar takes
    /// (`bloomery_levers::residency_word`, the one owner), with the same
    /// numbers. Mutant: a parse of its own that drifts from the lever's.
    #[test]
    fn parse_is_the_levers_grammar() {
        for v in [
            "off",
            "mid-p0-s1",
            "mid-p33-s1",
            "mid-p40-s2",
            "",
            "on",
            "mid",
            "mid-p-s1",
            "mid-p4-s0",
            "mid-p04-s1",
            "mid-p4-s1x",
            "mid-p+4-s1",
        ] {
            assert_eq!(
                Residency::parse(v).is_ok(),
                bloomery_levers::residency_word(v).is_some(),
                "{v:?}"
            );
        }
        assert_eq!(
            Residency::parse("mid-p40-s2").ok(),
            Some(Residency::Mid {
                pinned: 40,
                spares: 2
            })
        );
    }

    /// An off tally notes nothing; a live one refuses a slot noted twice, a
    /// note outside its shape and a note of a layer no pass routes, by name.
    #[test]
    fn a_tally_refuses_a_double_and_an_out_of_shape_note() {
        let mut off = Tally::off();
        assert!(off.note(9, 9, 9, 1).is_ok() && !off.covers(0));
        let mut t = Tally::new(2..4, 3, 2, vec![false, true]);
        let unrouted = t.note(3, 0, 0, 1).expect_err("a layer no pass routes");
        assert!(
            unrouted.to_string().contains("no pass routes"),
            "{unrouted}"
        );
        t.note(2, 0, 1, 7).expect("a first note");
        let twice = t.note(2, 0, 1, 8).expect_err("a slot twice");
        assert!(twice.to_string().contains("noted twice"), "{twice}");
        for (layer, row, k) in [(1, 0, 0), (4, 0, 0), (2, 2, 0), (2, 0, 3)] {
            assert!(t.note(layer, row, k, 0).is_err(), "({layer}, {row}, {k})");
        }
        t.clear();
        t.note(2, 0, 1, 8)
            .expect("a cleared tally takes the slot again");
    }

    /// A call's pick takes a live slot from its victim to `Landing`, and the
    /// layer's reader makes it `Live`: the slot is named throughout, only a
    /// slot live with the named victim takes a landing, and landing touches
    /// its own layer alone.
    #[test]
    fn a_pick_lands_through_landing_to_live() {
        let h = HOST;
        let map = SlotMap::from_rows(3..5, 4, vec![0, 1, h, h, 1, h, 0, h]).expect("two rows");
        let mut ledger = SlotLedger::of_map(&map).expect("a ledger");
        assert_eq!(
            ledger.row(3),
            Some(&[SlotState::Live(0), SlotState::Live(1)][..])
        );
        ledger
            .to_landing(3, 1, 1, 3, 0)
            .expect("slot 1 holds victim 1");
        assert_eq!(
            ledger.landing(3),
            Some((1, SlotState::Landing { e: 3, event: 0 }))
        );
        assert_eq!(ledger.landing(4), None);
        let before = ledger.clone();
        for (layer, slot, victim) in [(3, 1, 1), (3, 0, 1), (4, 0, 0), (5, 0, 0), (3, 2, 0)] {
            assert!(
                ledger.to_landing(layer, slot, victim, 2, 0).is_err(),
                "layer {layer} slot {slot} victim {victim}"
            );
            assert_eq!(ledger, before);
        }
        assert!(!ledger.land(4), "layer 4 holds no landing");
        assert!(ledger.land(3));
        assert_eq!(
            ledger.row(3),
            Some(&[SlotState::Live(0), SlotState::Live(3)][..])
        );
        assert!(!ledger.land(3) && !ledger.land(9));
    }
}
