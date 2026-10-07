//! The host tier: one tier, two ports. The model's host computation
//! ([`HostExperts`]) is called through either port: the step port
//! ([`step::StepPort`]) serves a captured chain's handoffs per go, inside the
//! graph; the batch port ([`batch::BatchPort`]) serves an eager layer-batch
//! behind an event and a host sync. The tier ([`HostTier`]) owns what both
//! share: the host computation, the slot map that sends a layer's slots to
//! it, the fault word a refusal is named by, the health (a failed service
//! poisons the tier), the counters, which [`HostTier::stats`] shows as one
//! [`HybridStats`], and a placed load's host set ([`HostTier::residency`]).
//!
//! Per hybrid layer the captured chain carries, after the router:
//!
//! ```text
//! router → D2H(handoff) → go → [card experts, shared expert] → wait → add(hsum) → combine
//! ```
//!
//! The handoff is one device region the router's launches write — a
//! sequence word, the ids, the weights and the f32 normed activation — and
//! one copy node (or the chain's own launch) lands it in the host-mapped
//! page ([`page::PageLayout`]). `go` is one batch of stream memory
//! operations: a system-scope barrier (the image lands first), the layer's id
//! written into the page, a second barrier, then an atomic add of one to the
//! host-mapped generation word and to the device sequence word the next
//! handoff carries. `wait` waits until the host-mapped counter is at least
//! one and adds minus one; it sits after the card's experts and the shared
//! expert, so their work runs under the host's.
//!
//! No word is written by the card and the host at the same time: the
//! generation, sequence and layer words only by the card; the counter by the
//! host between a go and its wait, and by the card after the wait passes —
//! and the host touches it again only after the next go, which the stream
//! orders behind that add and a system barrier.
//!
//! A boundary of several columns a row ([`step::Boundary::with_cols`])
//! carries a chain of one row of `m` consecutive positions
//! ([`Chain::Cols`]): one handoff image of `m` columns, one go, one union
//! call over every column ([`HostExperts::experts_union_into`]) and one wait
//! a layer.
//!
//! A boundary of two rows carries a pass whose two tokens run one layer
//! apart, so a row's go can land while the other row's wait is pending: each
//! row has its own layer word, counter, image and sum, and the rule above
//! holds per row, since a row's next go sits behind its own wait. The
//! generation and sequence count every go in stream order, and the host
//! serves in that order.
//!
//! The step port's host side is the decode thread. After a graph launch it
//! serves the captured chain's hybrid layers in order
//! ([`HostTier::serve_captured`]); an eager chain is served layer by layer as
//! it is enqueued, so the stream never holds more than one layer of work
//! behind a wait. A service waits for its go with the whole pool spinning on
//! the generation word (a pool job: a worker inside a job is not parked, and
//! the expert dispatch that follows starts inside the spin window the job
//! leaves it in), checks the handoff's sequence number and layer, has the
//! host computation write the host experts' sum into the page and adds one to
//! the counter. The combine reads that sum in place through the mapping. The
//! protocol — the words, the sequence, the service loop — is this module's
//! alone; a model supplies only what one service computes.
//!
//! An eager batch of prompt tokens is served outside that protocol
//! ([`HostTier::serve_batch`]): the caller brings the batch's handoffs to the
//! host, and one service computes a layer's host experts for every token of
//! the batch in one union call ([`HostExperts::experts_union_into`]), reading
//! the activations where they landed. A service takes a set of host experts
//! to leave out, and its columns may span more than one batch. It moves no
//! flag word and no sequence number, so the next step's handoff finds the
//! host where the last step left it.
//!
//! A handoff the card should already have refused — a non-finite activation,
//! a routed id the slot map does not know — is never computed: its host sum
//! is NaN and the tier records it ([`HostTier::refusal`]). A step service
//! then fails with a [`GpuError::Protocol`] naming the layer, the row and
//! what the host saw, and releases the stream; it never reads the card's
//! fault word, because the card goes on to the next layer's wait, which this
//! thread serves. The step's caller reads the word once the stream has
//! drained and names the refusal with [`name_refusal`]: the card's fault when
//! the card raised one at or before the layer, the host's error when it did
//! not — never the host's own failure in place of the card's. A batch
//! service runs after the card's event, so with the word watched
//! ([`HostTier::watch_fault`]) it reads the word itself. A failed service
//! poisons the tier; the model's reset lifts a refusal's poison
//! ([`HostTier::reset`]), and any other stays until a reload.
//!
//! A tier may hang tier cards under it ([`tier::TierCard`],
//! [`HostTier::attach_tiers`]; one today, [`refuse_tier_count`]): per
//! hybrid layer each card computes the routed experts of its set in the host
//! leg's shadow, behind its own go and counter on the tiers' page
//! ([`tier::TierPage`]), and the stage card's wait waits for the host's
//! counter and for those of the tiers that hold experts of the layer. The
//! host launches each card's captured graph at the start of a replay's
//! service and, once it has served the replay's last layer, waits for each
//! card's progress under the go deadline and reads its fault copy (the module
//! comment of [`tier`]). A card that stops signalling poisons the tier as a
//! lost card, which no reset lifts.

pub mod batch;
pub mod handoff;
pub mod leg;
pub mod nvtier;
pub mod page;
pub mod residency;
pub mod route_trace;
pub mod run;
pub mod slots;
pub mod step;
pub mod swap;
pub mod swap_source;
pub mod tier;
pub mod xstream;

pub use leg::{BatchLeg, LegTimer, ServeNote, StepLeg};

use crate::GpuError;
use crate::fault::{Fault, LAYER_NONE};
use crate::tensor::DeviceTensor;
use batch::{BatchKey, BatchPort, BatchService, ServeTimes, Tier};
use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer};
use model::placement::Machine;
use model::{Tensor2, Tensor2View};
use page::{MAX_ROWS, Word};
use residency::HostResidency;
use route_trace::RouteTrace;
use runtime::swaprule::KeptRows;
use slots::{MAX_TIERS, Slot, SlotMap};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use step::{Boundary, Chain, GO_DEADLINE, HandoffTarget, SERVE, StepPort, stream_idle, word};
use tier::{Pass, TierCard, TierPage, TierTarget};

/// What a prompt group's start ([`HostTier::begin_group`]) is named as in its
/// refusals.
pub const BEGIN_GROUP: &str = "HostTier::begin_group";

/// The tier cards one host tier serves at most. The stage card's tier
/// entries hand one tier its image and join one tier's rows; N >= 2 lands
/// with V4.1's `_tiers` handoff.
pub const SERVED_TIERS: usize = 1;

/// Refuse, as `what`, a load of `n` expert tier cards past
/// [`SERVED_TIERS`], before anything of the load is uploaded: the one
/// owner of that rule, which [`HostTier::attach_tiers`] holds too.
pub fn refuse_tier_count(what: &'static str, n: usize) -> Result<(), GpuError> {
    if n <= SERVED_TIERS {
        return Ok(());
    }
    Err(GpuError::shape(
        what,
        format!(
            "{n} expert tier cards: the host tier serves one tier card; N >= 2 lands with V4.1's \
             `_tiers` handoff"
        ),
    ))
}

/// The tiers whose bits `mask` sets, in tier order, into `out`; their count.
fn tiers_of(mask: u32, out: &mut [usize; MAX_TIERS]) -> usize {
    let mut k = 0;
    for t in (0..MAX_TIERS).filter(|t| mask >> t & 1 == 1) {
        out[k] = t;
        k += 1;
    }
    k
}

const _: () = assert!(MAX_TIERS <= u32::BITS as usize);

/// Refuse, as `what`, a placement `machine` with an expert tier card, for a
/// load path that hangs no tier under its host tier: before any upload,
/// since the tier's experts would otherwise be served by nobody or by the
/// host unasked.
pub fn refuse_expert_tiers(what: &'static str, machine: &Machine) -> Result<(), GpuError> {
    match machine.tiers.as_slice() {
        [] => Ok(()),
        tiers => Err(GpuError::shape(
            what,
            format!(
                "the placement names {} expert tier card(s) ({}); this load hangs no tier under its \
                 host tier",
                tiers.len(),
                tiers
                    .iter()
                    .map(|t| t.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        )),
    }
}

/// What one host service computes, supplied by the architecture: the
/// weighted sum of the listed routed experts of layer `layer` for the
/// activation `x` (one column of the model width), written over `out` (that
/// width; an empty list writes zeros). The list is the handoff's host slots
/// in slot order, at most the boundary's routed width `n_used`, each id with
/// its routing weight; an architecture's scratch is made for that width at
/// load and refuses a longer list by name. The protocol around the call — waiting for the go, checking the
/// handoff, signalling the card, releasing it on a failure — is
/// [`HostTier`]'s alone.
pub trait HostExperts {
    fn experts_into(
        &mut self,
        layer: usize,
        x: &Tensor2,
        experts: &[(u32, f32)],
        out: &mut [f32],
    ) -> Result<(), GpuError>;

    /// The batch form: `lists[j]` for column `j` of `x`, its sum over
    /// `out[j · width ..][.. width]`, each column bit for bit what
    /// [`HostExperts::experts_into`] writes for that column and list alone —
    /// whatever the call's width, so a call over several batches' columns
    /// writes what their calls one batch at a time would. `x` is a view of
    /// the columns where they already are. A width past what the
    /// architecture's scratch holds is refused by name, and so is any call
    /// by an architecture without a batched host path.
    fn experts_union_into(
        &mut self,
        layer: usize,
        x: Tensor2View<'_>,
        lists: &[&[(u32, f32)]],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        let _ = (layer, x, lists, out);
        Err(GpuError::state(
            "HostExperts::experts_union_into",
            "a batched host path: this architecture's host tier serves one column a call",
        ))
    }
}

/// What the host side has done since load. Counted by the decode thread
/// alone; nothing here takes a lock.
#[derive(Clone, Copy, Debug, Default)]
pub struct HybridStats {
    /// Layers served.
    pub served: u64,
    /// Services whose go had already landed when the host began to wait for
    /// it — the card was waiting on the host.
    pub go_early: u64,
    /// Of those, the ones that opened a replay: the go of the chain's first
    /// hybrid layer landed before the launch returned to the host.
    pub go_early_first: u64,
    /// Host slots computed, summed over services.
    pub host_slots: u64,
    /// The host's share of each service's routed weight squared,
    /// `Σ_host w² / Σ_all w²`, summed over services.
    pub host_w2: f64,
    /// Host wall time from the go seen to the signal, summed (ns).
    pub leg_ns: u64,
    /// Pool workers that parked between the start of a service's wait and
    /// its signal, summed.
    pub parks_in_service: u64,
    /// Time from the decode thread seeing a go to the last pool thread
    /// leaving the wait — what a worker that was not spinning when the go
    /// landed (parked, or preempted) adds before the experts start. Summed
    /// and worst, over services whose go was not already there (ns).
    pub straggle_ns: u64,
    pub straggle_max_ns: u64,
    /// Of a two-row pass ([`Chain::Pair`]), the host slot ids of row 1's
    /// service of a layer that row 0's service of the same layer also
    /// listed, summed over row 1's services. The one-token step adds nothing.
    pub overlap_slots: u64,
    /// Host slots computed by row 1's services of a two-row pass, summed:
    /// the denominator of the row overlap `overlap_slots / pair_row1_slots`.
    /// The union of a layer's two host lists is `host_slots − overlap_slots`
    /// over any span of whole passes.
    pub pair_row1_slots: u64,
    /// Calls into the host experts by step services, one a service: a
    /// one-column service's `experts_into`, a `Cols` service's one
    /// `experts_union_into` over all its columns.
    pub host_calls: u64,
    /// Services of a one-row chain of several columns ([`Chain::Cols`]),
    /// each one go, one union call and one wait for every column, and the
    /// columns they carried, summed. Both count in `served` too.
    pub cols_served: u64,
    pub cols_cols: u64,
    /// Go waits written ([`HostTier::gap_summary`] of a span of them).
    pub gaps: u64,
    /// Batch services ([`HostTier::serve_batch`]): layers served, the columns
    /// they carried, the host slots those columns listed and the service
    /// computed, and the host wall time of the union calls (ns), summed. None
    /// of them counts above.
    pub batch_served: u64,
    pub batch_cols: u64,
    pub batch_host_slots: u64,
    pub batch_ns: u64,
    /// Of the host slots batch services' columns listed, those whose expert
    /// was in the service's exclusion set and so not computed, summed. A
    /// refused column counts in neither this nor `batch_host_slots`.
    pub batch_excluded_slots: u64,
    /// Services, step or batch, that met input the card should already have
    /// refused ([`HostTier::refusal`] names the first).
    pub refusals: u64,
    /// Resets that lifted a refusal's poison ([`HostTier::reset`],
    /// [`HostTier::lift_refusal`]).
    pub resets: u64,
    /// The last poison's kind, service and layer; [`HostTier::last_poison`]
    /// holds what it saw. Kept across a reset.
    pub last_poison: Option<PoisonMark>,
}

/// Input a host service met that the card should already have refused: a
/// routed id the slot map does not know, or a non-finite activation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    /// The service: `Hybrid::serve` or `Hybrid::serve_batch`.
    pub what: &'static str,
    pub layer: usize,
    /// Where and what: `row r: …` of a step, `column j of n: …` of a batch.
    pub detail: String,
}

/// The bound on a host wait for an expert tier's stream to drain at its
/// reset.
const DRAIN_DEADLINE: Duration = Duration::from_secs(30);

/// `d` in whole nanoseconds, saturated at `u64::MAX`.
#[must_use]
pub fn nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// Why [`poll_drained`] stopped short.
pub(crate) enum Drain {
    /// The query returned a driver error.
    Driver(GpuError),
    /// The stream was still running past the deadline, after this long.
    Late(Duration),
}

/// Poll `stream` until it has drained, at most `deadline`, sleeping
/// between polls from the first.
pub(crate) fn poll_drained(stream: &CudaStream, deadline: Duration) -> Result<(), Drain> {
    match await_within("poll_drained", deadline, Duration::ZERO, || stream.query()) {
        Ok(()) => Ok(()),
        Err(GpuError::Stalled { waited, .. }) => Err(Drain::Late(waited)),
        Err(e) => Err(Drain::Driver(e)),
    }
}

/// The bound on an engine-thread wait for the card on the step and prompt
/// path ([`await_done`]): past the longest such wait that is not a fault —
/// one ubatch of a prompt on the slowest card, behind a copy that takes the
/// residency machine's whole [`MachineCfg::deadline`] — so the machine's
/// own error comes first when it has one.
///
/// The bound sees only a stall the engine thread can observe from its own
/// polling loop: its own stream's work not finishing. A driver that runs
/// every stream on one hardware queue can instead wedge the thread inside a
/// driver call no bound here ends — a copy enqueued behind work that copy
/// itself feeds — and that wedge is named, not ended, by the engine
/// watchdog ([`crate::watchdog`]). The staging-window rule (`host::step`'s
/// `Closed`) is what keeps that cycle from closing.
///
/// [`MachineCfg::deadline`]: swap::MachineCfg::deadline
pub(crate) const ENGINE_BOUND: Duration = Duration::from_secs(60);

/// How long [`await_done`] spins on its query before it sleeps between
/// polls: longer than any decode step, so a step's readback never waits out
/// a sleep, and a wait past it shows its thread asleep.
const SPIN: Duration = Duration::from_secs(1);

/// The sleep between polls once the spin has passed.
const POLL: Duration = Duration::from_micros(50);

/// Wait, on the engine thread, until `done` — a stream's or an event's
/// query — says the work it covers has finished, within [`ENGINE_BOUND`]:
/// the query spun on for [`SPIN`], then polled every [`POLL`]. Past the
/// bound, [`GpuError::Stalled`] of `what`, with no note; a caller that holds
/// the host tier adds its note ([`HostTier::noted`]).
pub(crate) fn await_done(
    what: &'static str,
    done: impl FnMut() -> Result<bool, cuda_core::DriverError>,
) -> Result<(), GpuError> {
    await_within(what, ENGINE_BOUND, SPIN, done)
}

/// The one polling loop: `done` asked until it says done, at most `bound`,
/// spun on for `spin`, then polled every [`POLL`]; past the bound
/// [`GpuError::Stalled`] of `what`, with no note. A finished wait bumps the
/// engine watchdog's progress ([`crate::watchdog`]) when one is guarded.
fn await_within(
    what: &'static str,
    bound: Duration,
    spin: Duration,
    mut done: impl FnMut() -> Result<bool, cuda_core::DriverError>,
) -> Result<(), GpuError> {
    let t0 = Instant::now();
    loop {
        if done()? {
            crate::watchdog::wait_done();
            return Ok(());
        }
        let waited = t0.elapsed();
        if waited > bound {
            return Err(GpuError::Stalled {
                what,
                waited,
                bound,
                note: String::new(),
            });
        }
        if waited < spin {
            std::hint::spin_loop();
        } else {
            std::thread::sleep(POLL);
        }
    }
}

/// `e` with `swap`'s note ([`swap::SwapMachine::stall_note`]) when it is a
/// [`GpuError::Stalled`] and there is a machine; any other error as it is.
fn stall_noted(swap: Option<&swap::SwapMachine>, mut e: GpuError) -> GpuError {
    if let (GpuError::Stalled { note, .. }, Some(m)) = (&mut e, swap) {
        *note = m.stall_note();
    }
    e
}

/// Wait, polling, until `stream` has drained, at most `deadline`; past it
/// the error names the wait `what`.
pub(crate) fn drain_within(
    stream: &CudaStream,
    deadline: Duration,
    what: &'static str,
) -> Result<(), GpuError> {
    poll_drained(stream, deadline).map_err(|e| match e {
        Drain::Driver(e) => e,
        Drain::Late(waited) => GpuError::protocol(
            what,
            format!("the stream did not drain in {waited:?} (deadline {deadline:?})"),
        ),
    })
}

/// The kept rows the boundary ends the open pass with, from what its
/// caller gave ([`HostTier::keep_rows`]): `None` when no pass is `open`.
/// Refused by name: an open pass with no kept count, and a kept count with
/// no pass open, which no boundary would fold.
fn pass_kept(
    open: bool,
    kept: Option<(KeptRows, PassKind)>,
) -> Result<Option<(KeptRows, PassKind)>, GpuError> {
    const WHAT: &str = "HostTier::swap_boundary";
    match (open, kept) {
        (true, Some(k)) => Ok(Some(k)),
        (true, None) => Err(GpuError::state(
            WHAT,
            "the last pass's kept rows (HostTier::keep_rows)",
        )),
        (false, None) => Ok(None),
        (false, Some((rows, kind))) => Err(GpuError::protocol(WHAT, kept_with_no_pass(rows, kind))),
    }
}

/// Kept rows with no pass open, refused by name at the write
/// ([`HostTier::keep_rows`]) and at the read ([`pass_kept`], the next
/// boundary): one wording, one owner.
fn kept_with_no_pass(kept: KeptRows, kind: PassKind) -> String {
    format!(
        "{} rows kept as a {} pass with no pass open: no boundary opened the pass \
         they would end",
        kept.count(),
        kind.word()
    )
}

/// What a residency pass was, as its caller names it when it keeps its rows
/// ([`HostTier::keep_rows`]); the `residency pass` record's `pass` word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    /// No pass was open at the boundary: the first after the load or a reset.
    None,
    /// A decode step, one row kept.
    Step,
    /// A verify pass of two rows, its accepted rows kept.
    Pair,
    /// A pass of resident slots' rows: every row kept, rows of different
    /// sequences.
    Slots,
    /// A drafted pass of resident slots' verify rows: each slot's accepted
    /// rows kept, a rejected row leaves no trace.
    SlotsDrafted,
    /// A prompt call, one pass whose rows are not counted: 0 kept.
    Prompt,
    /// A pass its caller never kept (a failed pass, a verify a reset drops
    /// before its commit): 0 kept, its noted ids forgotten.
    Abandoned,
    /// A pass ended by a driver of the machine itself, not an engine's pass
    /// caller (`gate_swap`'s synthetic stacks).
    Driver,
}

impl PassKind {
    /// The record's word.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            PassKind::None => "none",
            PassKind::Step => "step",
            PassKind::Pair => "pair",
            PassKind::Slots => "slots",
            PassKind::SlotsDrafted => "slots_drafted",
            PassKind::Prompt => "prompt",
            PassKind::Abandoned => "abandoned",
            PassKind::Driver => "driver",
        }
    }
}

/// Why a service poisoned the tier ([`HostTier::last_poison`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Poison {
    /// Input the card should already have refused: the engine is sound, and
    /// [`HostTier::reset`] lifts it.
    Refused(Refusal),
    /// The service failed with an error of its own — the protocol's words,
    /// the handoff's shape, the host experts' answer: a broken engine, which
    /// only a reload clears.
    Failed {
        what: &'static str,
        layer: usize,
        error: String,
    },
    /// The host experts panicked: what they left behind is unknown, and only
    /// a reload clears it.
    Panicked {
        what: &'static str,
        layer: usize,
        message: String,
    },
    /// The expert tier's card stopped signalling ([`tier::TierCard`]): the
    /// card named `card`, what the host saw at layer `layer`. Only a reload
    /// clears it.
    CardLost {
        what: &'static str,
        card: String,
        layer: usize,
        detail: String,
    },
}

/// A [`Poison`]'s kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoisonKind {
    Refused,
    Failed,
    Panicked,
    CardLost,
}

/// A [`Poison`] without what it saw: what [`HybridStats`] carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoisonMark {
    pub kind: PoisonKind,
    /// The service: `Hybrid::serve` or `Hybrid::serve_batch`.
    pub what: &'static str,
    pub layer: usize,
}

impl Poison {
    #[must_use]
    pub fn mark(&self) -> PoisonMark {
        let (kind, what, layer) = match self {
            Poison::Refused(r) => (PoisonKind::Refused, r.what, r.layer),
            Poison::Failed { what, layer, .. } => (PoisonKind::Failed, *what, *layer),
            Poison::Panicked { what, layer, .. } => (PoisonKind::Panicked, *what, *layer),
            Poison::CardLost { what, layer, .. } => (PoisonKind::CardLost, *what, *layer),
        };
        PoisonMark { kind, what, layer }
    }
}

impl std::fmt::Display for Poison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Poison::Refused(r) => write!(
                f,
                "{} refused input at layer {}: {}",
                r.what, r.layer, r.detail
            ),
            Poison::Failed { what, layer, error } => {
                write!(f, "{what} failed at layer {layer}: {error}")
            }
            Poison::Panicked {
                what,
                layer,
                message,
            } => write!(f, "{what} panicked at layer {layer}: {message}"),
            Poison::CardLost {
                what,
                card,
                layer,
                detail,
            } => write!(
                f,
                "{what} lost the expert tier's card {card} at layer {layer}: {detail}"
            ),
        }
    }
}

/// The protocol's words as the host reads them ([`HostTier::words`]): the
/// card's go count, the host's service count, and each row's counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridWords {
    pub generation: u32,
    pub served: u32,
    /// Row `r`'s counter at `r`, for the boundary's rows; 0 past them.
    pub counters: [u32; MAX_ROWS],
    pub rows: usize,
}

impl HybridWords {
    /// What a drained stream leaves on a tier that serves every go and whose
    /// every signal a wait took back: the host has served every go the card
    /// made, and no counter holds a signal. A fresh tier holds all zeros.
    #[must_use]
    pub fn at_rest(&self) -> bool {
        self.generation == self.served && self.counters.iter().all(|&c| c == 0)
    }
}

impl std::fmt::Display for HybridWords {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "generation {} served {} counters {:?}",
            self.generation,
            self.served,
            &self.counters[..self.rows]
        )
    }
}

/// What the tier itself counts: refusals, the resets that lifted one's
/// poison, and the last poison. The fields are [`HybridStats`]'s of the same
/// names.
#[derive(Clone, Copy, Debug, Default)]
struct TierStats {
    refusals: u64,
    resets: u64,
    last_poison: Option<PoisonMark>,
}

/// The tier's health, which both ports share: whether a failed service
/// poisoned it and why, the refusal a service is failing on, the refusal
/// that failed one (else the first recorded), and the counters.
#[derive(Default)]
pub(crate) struct Health {
    /// A service failed and released the stream; nothing is served again
    /// until a reset lifts it ([`HostTier::reset`]), which only a refusal's
    /// poison allows.
    poisoned: bool,
    /// Why the last service that failed poisoned the tier; kept across a
    /// reset.
    poison: Option<Poison>,
    /// The refusal a service is failing on, from the moment it is recorded
    /// to the catch that poisons the tier.
    failing: Option<Refusal>,
    /// The refusal that failed a service, else the first one recorded.
    refusal: Option<Refusal>,
    stats: TierStats,
}

impl Health {
    /// Count `r` and keep it: a service that `fails` on it poisons the tier,
    /// so its refusal replaces any earlier one; one that returns keeps the
    /// first.
    pub(crate) fn record_refusal(&mut self, r: Refusal, fails: bool) {
        self.stats.refusals += 1;
        if fails {
            self.failing = Some(r.clone());
        }
        if fails || self.refusal.is_none() {
            self.refusal = Some(r);
        }
    }

    /// Poison the tier: a service of layer `layer` failed with `error`, or
    /// panicked with `panic`. A refusal the service recorded as it failed is
    /// the cause; anything else is the service's own failure.
    fn set_poison(
        &mut self,
        what: &'static str,
        layer: usize,
        error: Option<&GpuError>,
        panic: Option<&(dyn std::any::Any + Send)>,
    ) {
        let refused = self.failing.take();
        let cause = match (panic, refused, error) {
            (Some(p), _, _) => Poison::Panicked {
                what,
                layer,
                message: panic_message(p),
            },
            (None, Some(r), _) => Poison::Refused(r),
            (None, None, e) => Poison::Failed {
                what,
                layer,
                error: e.map_or_else(|| "no error recorded".to_string(), ToString::to_string),
            },
        };
        self.poisoned = true;
        self.stats.last_poison = Some(cause.mark());
        self.poison = Some(cause);
    }

    /// Poison the tier: the expert tier's card `card` was lost at layer
    /// `layer`, as `detail` says.
    fn set_lost(&mut self, what: &'static str, card: &str, layer: usize, detail: &str) {
        self.failing = None;
        let cause = Poison::CardLost {
            what,
            card: card.to_string(),
            layer,
            detail: detail.to_string(),
        };
        self.poisoned = true;
        self.stats.last_poison = Some(cause.mark());
        self.poison = Some(cause);
    }

    /// Refuse `what` on a poisoned tier, naming the poison.
    fn refuse_if_poisoned(&self, what: &'static str) -> Result<(), GpuError> {
        if !self.poisoned {
            return Ok(());
        }
        let poison = self
            .poison
            .clone()
            .expect("a poisoned tier holds its poison: set_poison and set_lost set both");
        Err(GpuError::HostPoisoned {
            what,
            poison: Box::new(poison),
        })
    }
}

/// The host tier: the model's host computation and the two ports it is
/// called through, with what they share.
///
/// Field order is drop order: the host set's lock (`residency`) spans pages
/// of the mappings `experts` keeps alive, so it is released first. The drop
/// stops the residency machine before any field goes
/// ([`HostTier::stop_swap`]).
pub struct HostTier<H> {
    /// What the placed load did to the plan's host set, when the tier holds
    /// it ([`HostTier::keep_residency`]).
    residency: Option<HostResidency>,
    experts: H,
    /// The card's fault word once [`HostTier::watch_fault`] has given it:
    /// what a batch service that meets refused input reads to name the
    /// card's fault.
    fault: Option<Arc<DeviceBuffer<u32>>>,
    /// A planted refusal the next host service returns — a step service's
    /// ([`HostTier::serve`]) or a batch service's ([`HostTier::serve_key`],
    /// [`HostTier::serve_batch`]) — for [`HostTier::plant_refusal`]: a test
    /// seam for the slots harness's poison clauses, never set on a serving
    /// path.
    planted: Option<String>,
    health: Health,
    /// Which experts each layer's card stack holds; the host serves the rest.
    slots: SlotMap,
    step: StepPort,
    /// The batch service's scratch and counters, and the port's sets once a
    /// caller made them ([`HostTier::prepare_batch`]).
    batch: BatchService,
    port: Option<BatchPort>,
    /// The expert tier cards in tier order, once attached
    /// ([`HostTier::attach_tiers`]): tier `t` serves the slot map's tier `t`.
    tiers: Vec<TierCard>,
    /// The tiers' page, which their graphs and windows address: after
    /// `tiers`, so it is freed after them.
    tier_page: Option<TierPage>,
    /// Per tier, the tier layers the host has served since load (wrapping):
    /// a tier whose progress is behind it holds the stage card at a wait.
    tier_goes: Vec<u32>,
    /// Per tier, a replay's tier layers in go order: the service's scratch.
    tier_lists: Vec<Vec<(usize, usize)>>,
    /// The residency machine over the stage card's slots, once started
    /// ([`HostTier::start_swap`]), and the rows the open pass keeps.
    swap: Option<swap::SwapMachine>,
    swap_kept: Option<(KeptRows, PassKind)>,
    /// The stage card's copy of the map the machine writes: after `swap`, so
    /// it outlives the machine that holds its address.
    swap_view: Option<Arc<DeviceTensor<u32>>>,
    /// The expert stream over the machine's source, once started
    /// ([`HostTier::xstream_start`]): its streamed experts leave the batch
    /// port's serve.
    xstream: Option<xstream::XStream>,
}

impl<H> HostTier<H> {
    /// Stop the residency machine, if one runs, and drop it: its staging
    /// thread stops, every copy waiting on a staging word is let through and
    /// the copy stream drains, each within the machine's deadline, or the
    /// drop leaks and names what it could not free ([`swap::SwapMachine`]'s
    /// `Drop`). A copy queued behind a job not yet due waits for as long as
    /// the staging window stays closed, and a context synchronize or a free
    /// on the card waits with it: every owner of a running machine calls
    /// this before it frees anything on the card. Idempotent.
    pub fn stop_swap(&mut self) {
        drop(self.xstream.take());
        drop(self.swap.take());
        self.swap_kept = None;
    }

    /// The step port's residency tally, the machine's shape once one runs:
    /// what a served pass's routed ids are noted into and the next
    /// [`HostTier::swap_boundary`] folds. A gate that drives a machine's
    /// passes without the handoff protocol notes into it through here;
    /// `None` without a machine. No engine caller: the step port's services
    /// note as they serve.
    #[doc(hidden)]
    #[must_use]
    pub fn swap_tally(&mut self) -> Option<&mut swap::Tally> {
        self.swap.is_some().then_some(&mut self.step.tally)
    }
}

impl<H> Drop for HostTier<H> {
    fn drop(&mut self) {
        self.stop_swap();
    }
}

impl<H: HostExperts> HostTier<H> {
    /// The tier over `boundary`: `experts` computes every expert `slots`
    /// sends to the host, and `layers` bounds the chain. Load-time only.
    pub fn new(
        boundary: Boundary,
        slots: SlotMap,
        experts: H,
        layers: usize,
    ) -> Result<HostTier<H>, GpuError> {
        Ok(HostTier {
            residency: None,
            experts,
            fault: None,
            planted: None,
            health: Health::default(),
            slots,
            step: StepPort::new(boundary, layers),
            batch: BatchService::default(),
            port: None,
            tiers: Vec::new(),
            tier_page: None,
            tier_goes: Vec::new(),
            tier_lists: Vec::new(),
            swap: None,
            swap_kept: None,
            swap_view: None,
            xstream: None,
        })
    }

    /// Run the residency machine over this tier's slot map
    /// ([`swap::SwapMachine::new`]): `view` is the stage card's copy of the
    /// map, which the chain reads and the machine writes, kept alive here for
    /// the machine's life; the step port notes its routed ids into the
    /// machine's tally and holds its staging window open while the pool waits
    /// for a go. The machine frees each layer's spare slots now, so every
    /// piece that sizes itself from the map's capacity is made first.
    /// Load-time only; a second machine is refused by name.
    pub fn start_swap(
        &mut self,
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        view: Arc<DeviceTensor<u32>>,
        source: Arc<dyn swap::SwapSource>,
        cfg: swap::MachineCfg,
    ) -> Result<(), GpuError> {
        if self.swap.is_some() {
            return Err(GpuError::state(
                "HostTier::start_swap",
                "a tier without a machine",
            ));
        }
        let machine =
            swap::SwapMachine::new(ctx, stream, &mut self.slots, view.buf(), source, cfg)?;
        self.step.tally = machine.tally();
        self.step.window = Some(machine.window());
        self.swap = Some(machine);
        self.swap_view = Some(view);
        self.swap_kept = None;
        Ok(())
    }

    /// The residency machine, once started.
    #[must_use]
    pub fn swap(&self) -> Option<&swap::SwapMachine> {
        self.swap.as_ref()
    }

    /// `e` with what the residency machine holds now when it is a
    /// [`GpuError::Stalled`] and the tier runs one ([`stall_noted`]); any
    /// other error as it is.
    #[must_use]
    pub fn noted(&self, e: GpuError) -> GpuError {
        stall_noted(self.swap.as_ref(), e)
    }

    /// The pass that just ran, a `kind`, keeps `kept` rows (a step its one
    /// row, a verify its accepted rows, a drafted slots pass each slot's
    /// accepted rows, a prompt call none): what the next boundary folds.
    /// Refused by name — the write side of the refusal the
    /// next boundary makes ([`pass_kept`]) — rows kept with no pass
    /// open, and ones kept while a boundary made ahead waits for its launch
    /// (the pass it opened has not run). Nothing without a machine.
    pub fn keep_rows(&mut self, kept: KeptRows, kind: PassKind) -> Result<(), GpuError> {
        const WHAT: &str = "HostTier::keep_rows";
        let Some(m) = self.swap.as_ref() else {
            return Ok(());
        };
        m.refuse_if_ahead(WHAT)?;
        if !m.pass_open() {
            return Err(GpuError::protocol(WHAT, kept_with_no_pass(kept, kind)));
        }
        self.swap_kept = Some((kept, kind));
        Ok(())
    }

    /// The residency boundary before a pass on `stream`
    /// ([`swap::SwapMachine::boundary`]): the open pass ends with its kept
    /// rows, then the machine lands and plans; its report, with the kind of
    /// the pass it ended ([`PassKind::None`] when none was open). `None`
    /// without a machine; a pass that ran with no kept count, and a kept
    /// count given with no pass open, are refused by name.
    pub fn swap_boundary(
        &mut self,
        stream: &CudaStream,
    ) -> Result<Option<(PassKind, swap::PassReport)>, GpuError> {
        self.boundary_of(stream, false)
    }

    /// The residency boundary an engine's pass order calls for at `at`
    /// ([`swap::BoundaryAt`]), stamped with its readbacks: before a launch,
    /// the boundary made ahead of it taken when the machine holds one
    /// ([`swap::SwapMachine::take_ahead`]; `None`, its report was given when
    /// it was made), else [`HostTier::swap_boundary`]; ahead, after a pass's
    /// service and kept rows and before its readback, the next pass's
    /// boundary made now ([`swap::SwapMachine::boundary_ahead`]). `None`
    /// without a machine. Refused by name: [`HostTier::swap_boundary`]'s
    /// refusals, and a boundary made ahead while another waits. Every
    /// boundary an engine call lands also moves the engine watchdog's
    /// progress and names its machine there ([`crate::watchdog`]).
    pub fn swap_at(
        &mut self,
        stream: &CudaStream,
        at: swap::BoundaryAt,
    ) -> Result<Option<(PassKind, swap::PassReport)>, GpuError> {
        let ahead = match (at, self.swap.as_mut()) {
            (_, None) => return Ok(None),
            (swap::BoundaryAt::Launch { .. }, Some(m)) if m.ahead().is_some() => {
                crate::watchdog::machine(m.shared());
                let b = m.take_ahead()?;
                crate::watchdog::boundary_landed(b);
                return Ok(None);
            }
            (swap::BoundaryAt::Launch { .. }, Some(m)) => {
                crate::watchdog::machine(m.shared());
                false
            }
            (swap::BoundaryAt::Ahead { .. }, Some(m)) => {
                crate::watchdog::machine(m.shared());
                true
            }
        };
        let mut made = self.boundary_of(stream, ahead)?;
        if let Some((_, r)) = made.as_mut() {
            r.reads = at.reads();
            crate::watchdog::boundary_landed(r.boundary);
        }
        Ok(made)
    }

    /// [`HostTier::swap_boundary`], made ahead of its pass when `ahead`.
    /// A boundary made ahead that waits for its launch is refused first, so
    /// the open pass's kept count stays where it was.
    fn boundary_of(
        &mut self,
        stream: &CudaStream,
        ahead: bool,
    ) -> Result<Option<(PassKind, swap::PassReport)>, GpuError> {
        let Some(m) = self.swap.as_mut() else {
            return Ok(None);
        };
        m.refuse_if_ahead("HostTier::swap_boundary")?;
        let mut kind = PassKind::None;
        if let Some((rows, of)) = pass_kept(m.pass_open(), self.swap_kept.take())? {
            m.end_pass(&mut self.step.tally, rows)?;
            kind = of;
        }
        let report = match ahead {
            true => m.boundary_ahead(stream, &mut self.slots)?,
            false => m.boundary(stream, &mut self.slots)?,
        };
        Ok(Some((kind, report)))
    }

    /// Open a prompt call on the residency machine
    /// ([`swap::SwapMachine::begin_call`]) inside the pass the last boundary
    /// opened. `false` without a machine: the map stays the load's.
    pub fn call_begin(
        &mut self,
        stream: &CudaStream,
        cfg: swap::CallCfg,
    ) -> Result<bool, GpuError> {
        match self.swap.as_mut() {
            Some(m) => m.begin_call(stream, cfg).map(|()| true),
            None => Ok(false),
        }
    }

    /// The open call's pick at layer `layer` from the batch counts `counts`
    /// ([`swap::SwapMachine::call_pick`]) over this tier's map. A pick the
    /// machine refuses for a victim the host does not serve from resident
    /// pages once its pages are read in again ([`GpuError::Unresident`],
    /// counted in the call's report) admits nothing: the layer's experts stay
    /// where they are, and the call goes on with the layer's host experts on
    /// the host. Refused by name without a machine, and the pick's other
    /// refusals.
    pub fn call_pick(
        &mut self,
        stream: &CudaStream,
        layer: usize,
        counts: &[u32],
        cap: usize,
    ) -> Result<swap::CallPick, GpuError> {
        let m = self.swap.as_mut().ok_or(GpuError::state(
            "HostTier::call_pick",
            "a residency machine (HostTier::start_swap)",
        ))?;
        match m.call_pick(stream, &mut self.slots, layer, counts, cap) {
            Err(GpuError::Unresident { .. }) => Ok(swap::CallPick::nothing(layer, counts)),
            r => r,
        }
    }

    /// [`HostTier::call_pick`] at `key`'s layer from the routed ids of
    /// `key`'s download ([`HostTier::routed_ids`], which waits for its
    /// copies), counted into `counts` — one count per expert of the slot
    /// map, the caller's buffer, reused. Refused by name before the pick: a
    /// routed id past the map's experts, and every refusal of the two it
    /// calls.
    pub fn call_pick_routed(
        &mut self,
        stream: &CudaStream,
        key: BatchKey,
        counts: &mut Vec<u32>,
        cap: usize,
    ) -> Result<swap::CallPick, GpuError> {
        let n_expert = self.slots.n_expert();
        counts.clear();
        counts.resize(n_expert, 0);
        for &id in self.routed_ids(key)? {
            let c = usize::try_from(id)
                .ok()
                .and_then(|i| counts.get_mut(i))
                .ok_or_else(|| {
                    GpuError::shape(
                        "HostTier::call_pick_routed",
                        format!(
                            "layer {}: a routed id {id} of {n_expert} experts",
                            key.layer
                        ),
                    )
                })?;
            *c += 1;
        }
        self.call_pick(stream, key.layer, counts, cap)
    }

    /// The open call's floor from its next pick on
    /// ([`swap::SwapMachine::set_call_floor`]). Refused by name without a
    /// machine.
    pub fn call_floor(&mut self, floor: u32) -> Result<(), GpuError> {
        self.swap
            .as_mut()
            .ok_or(GpuError::state(
                "HostTier::call_floor",
                "a residency machine (HostTier::start_swap)",
            ))?
            .set_call_floor(floor)
    }

    /// The open call's pick path's backlog bound from its next pick on
    /// ([`swap::SwapMachine::set_call_backlog`]). Refused by name without a
    /// machine.
    pub fn call_backlog(&mut self, backlog: u64) -> Result<(), GpuError> {
        self.swap
            .as_mut()
            .ok_or(GpuError::state(
                "HostTier::call_backlog",
                "a residency machine (HostTier::start_swap)",
            ))?
            .set_call_backlog(backlog)
    }

    /// Read the call's picks' `staged_us` back after its end
    /// ([`swap::SwapMachine::fill_staged_us`]): nothing without a machine.
    pub fn call_staged_us(&self, picks: &mut [(usize, swap::CallPick)]) {
        if let Some(m) = self.swap.as_ref() {
            m.fill_staged_us(picks);
        }
    }

    /// The open call's landed event of layer `layer`
    /// ([`swap::SwapMachine::call_landed`]); `None` without a machine.
    pub fn call_landed(&self, layer: usize) -> Result<Option<&CudaEvent>, GpuError> {
        self.swap.as_ref().map(|m| m.call_landed(layer)).transpose()
    }

    /// Layer `layer`'s last read of the call so far on `stream`
    /// ([`swap::SwapMachine::call_reader`]); nothing without a machine.
    pub fn call_reader(&mut self, layer: usize, stream: &CudaStream) -> Result<(), GpuError> {
        match self.swap.as_mut() {
            Some(m) => m.call_reader(layer, stream),
            None => Ok(()),
        }
    }

    /// End the open call ([`swap::SwapMachine::end_call`]), its placement
    /// `kept` for the passes after it or returned to the call's start; its
    /// report. `None` without a machine.
    pub fn call_end(
        &mut self,
        stream: &CudaStream,
        kept: bool,
    ) -> Result<Option<swap::CallReport>, GpuError> {
        match self.swap.as_mut() {
            Some(m) => m.end_call(stream, &mut self.slots, kept).map(Some),
            None => Ok(None),
        }
    }

    /// Start the expert stream ([`xstream::XStream::new`]) over the
    /// residency machine's source, for the map's layers: its ring sized from
    /// the card's free bytes, its probe streaming up to `probe` host experts
    /// of the first layer with stage stacks. `cfg`'s `max_half` and `probe`
    /// are set here from the map: the most host experts a layer holds, and
    /// that layer's. Refused by name: no machine, a second stream, and
    /// [`xstream::XStream::new`]'s refusals. Load-time only.
    pub fn xstream_start(
        &mut self,
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        mut cfg: xstream::XCfg,
        probe: usize,
    ) -> Result<&xstream::XStream, GpuError> {
        const WHAT: &str = "HostTier::xstream_start";
        if self.xstream.is_some() {
            return Err(GpuError::state(WHAT, "a tier without an expert stream"));
        }
        let source = self
            .swap
            .as_ref()
            .ok_or(GpuError::state(
                WHAT,
                "a residency machine (BLOOMERY_RESIDENCY=mid-p<P>-s<S>)",
            ))?
            .source();
        let n_expert = u32::try_from(self.slots.n_expert())
            .map_err(|_| GpuError::shape(WHAT, "a map of more experts than u32 counts"))?;
        let mut max_half = 0;
        let mut first = None;
        for l in self.slots.layers() {
            if source.part_bytes(l).is_empty() {
                continue;
            }
            let host: Vec<u32> = (0..n_expert)
                .filter(|&id| self.slots.slot(l, id) == Some(Slot::Host))
                .collect();
            max_half = max_half.max(host.len());
            if first.is_none() && !host.is_empty() {
                first = Some((l, host));
            }
        }
        let (l, host) = first.ok_or(GpuError::state(
            WHAT,
            "a layer with stage stacks and host experts",
        ))?;
        cfg.max_half = max_half;
        cfg.probe = host.iter().rev().take(probe).map(|&id| (l, id)).collect();
        let layers = self.slots.layers().end;
        let x = xstream::XStream::new(ctx, stream, source, layers, cfg)?;
        Ok(self.xstream.insert(x))
    }

    /// The expert stream, once started.
    #[must_use]
    pub fn xstream(&self) -> Option<&xstream::XStream> {
        self.xstream.as_ref()
    }

    /// The expert stream to change, once started.
    pub fn xstream_mut(&mut self) -> Option<&mut xstream::XStream> {
        self.xstream.as_mut()
    }

    /// Open a call on the expert stream; `false` without one.
    pub fn xstream_begin(&mut self) -> bool {
        match self.xstream.as_mut() {
            Some(x) => {
                x.begin_call();
                true
            }
            None => false,
        }
    }

    /// Layer `unit.layer` of the walk's unit `unit` (its serve's key), of
    /// `cols` columns, on the expert stream
    /// ([`xstream::XStream::layer`]) from the unit's `counts` (one a router
    /// expert, as [`HostTier::call_pick_routed`] counted them) after a pick
    /// that admitted `admitted`: the layer's host experts and its stage row
    /// read off this tier's map, the stage stacks' experts its capacity.
    /// The streamed experts leave the batch port's serve of the layer until
    /// [`HostTier::xstream_read`]. Refused by name without a stream.
    pub fn xstream_layer(
        &mut self,
        stream: &CudaStream,
        unit: BatchKey,
        cols: usize,
        counts: &[u32],
        admitted: usize,
    ) -> Result<xstream::XLayer, GpuError> {
        const WHAT: &str = "HostTier::xstream_layer";
        let layer = unit.layer;
        let x = self.xstream.as_mut().ok_or(GpuError::state(
            WHAT,
            "an expert stream (HostTier::xstream_start)",
        ))?;
        let row = self
            .slots
            .row(layer)
            .ok_or_else(|| GpuError::shape(WHAT, format!("layer {layer} outside the map")))?;
        let n_card = self.slots.capacity(layer)?;
        let mut host = Vec::with_capacity(row.len());
        let mut stage = Vec::with_capacity(row.len());
        for (id, &e) in (0u32..).zip(row) {
            match Slot::of(e) {
                Slot::Card(s) => stage.push(s),
                Slot::Host => {
                    host.push(id);
                    stage.push(slots::HOST);
                }
                Slot::Tier { .. } => stage.push(slots::HOST),
            }
        }
        x.layer(
            stream,
            unit,
            cols,
            counts,
            (&host, admitted),
            (&stage, n_card),
        )
    }

    /// Layer `layer`'s stream as its card route reads it; `None` without a
    /// stream or when the layer streams nothing.
    #[must_use]
    pub fn xstream_ring(&self, layer: usize) -> Option<xstream::RingLayer> {
        self.xstream.as_ref()?.ring_layer(layer)
    }

    /// Layer `layer`'s card route has run on `stream`
    /// ([`xstream::XStream::read`]); nothing without a stream.
    pub fn xstream_read(&mut self, layer: usize, stream: &CudaStream) -> Result<(), GpuError> {
        match self.xstream.as_mut() {
            Some(x) => x.read(layer, stream),
            None => Ok(()),
        }
    }

    /// End the call on the expert stream; `None` without one.
    pub fn xstream_end(&mut self) -> Result<Option<xstream::XReport>, GpuError> {
        self.xstream
            .as_mut()
            .map(xstream::XStream::end_call)
            .transpose()
    }

    /// The residency back to its seed at a quiet boundary on `stream`
    /// ([`swap::SwapMachine::reset`]), the tally cleared and no pass open;
    /// its report. `None` without a machine.
    pub fn swap_reset(
        &mut self,
        stream: &CudaStream,
    ) -> Result<Option<swap::ResetReport>, GpuError> {
        let Some(m) = self.swap.as_mut() else {
            return Ok(None);
        };
        let deadline = m.deadline();
        drain_within(
            stream,
            deadline,
            "HostTier::swap_reset: the engine stream before the reset",
        )?;
        let r = m.reset(stream, &mut self.slots)?;
        drain_within(
            stream,
            deadline,
            "HostTier::swap_reset: the engine stream after the reset",
        )?;
        self.step.tally.clear();
        self.swap_kept = None;
        Ok(Some(r))
    }

    /// Hang `tiers` under the host tier, tier `t` its `t`-th card, on
    /// `stage`, the card the chain launches on: the tiers' page made in
    /// `stage`'s context, each card bound to it. The slot map must name
    /// exactly these tiers, and each tier's set must be the map's rows of
    /// that tier ([`tier::TierSet::of_map`]); more tiers than one host tier
    /// serves ([`refuse_tier_count`]), no tier, a second attach, and cards
    /// cut for different pages are refused by name. Load-time only.
    pub fn attach_tiers(
        &mut self,
        mut tiers: Vec<TierCard>,
        stage: &crate::Gpu,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "HostTier::attach_tiers";
        refuse_tier_count(WHAT, tiers.len())?;
        if tiers.is_empty() || !self.tiers.is_empty() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} tier cards onto a host tier that holds {}: one attach of at least one",
                    tiers.len(),
                    self.tiers.len()
                ),
            ));
        }
        if self.slots.tiers() != tiers.len() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the slot map names {} tiers and the load hangs {} tier cards",
                    self.slots.tiers(),
                    tiers.len()
                ),
            ));
        }
        for (t, tier) in tiers.iter().enumerate() {
            let set = tier.set();
            if set.layers() != self.slots.layers() || set.n_expert() != self.slots.n_expert() {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "the tier's set covers layers {:?} of {} experts, the slot map {:?} of {}",
                        set.layers(),
                        set.n_expert(),
                        self.slots.layers(),
                        self.slots.n_expert()
                    ),
                ));
            }
            // An expert on two devices is the slot map's refusal (`SlotMap::from_rows`).
            if *set != tier::TierSet::of_map(&self.slots, t)? {
                return Err(GpuError::shape(
                    WHAT,
                    format!(
                        "the tier {} holds another set than the slot map's rows of tier {t}",
                        tier.name()
                    ),
                ));
            }
        }
        let page = TierPage::new(stage, tiers[0].page_layout(tiers.len())?)?;
        for (t, tier) in tiers.iter_mut().enumerate() {
            tier.bind(&page, t, stage)?;
        }
        self.tier_goes = tiers.iter().map(|t| t.progress(&page)).collect();
        self.tier_lists = tiers.iter().map(|_| Vec::new()).collect();
        self.tiers = tiers;
        self.tier_page = Some(page);
        Ok(())
    }

    /// Attach `trace`: from here on every one-row step's service records
    /// its routed ids into it ([`route_trace`]), and a prompt batch's
    /// service is refused by name. The trace must cover the slot map's
    /// layers and experts; a second trace is refused.
    pub fn attach_route_trace(&mut self, trace: RouteTrace) -> Result<(), GpuError> {
        if self.step.trace.is_some() {
            return Err(GpuError::shape(
                "HostTier::attach_route_trace",
                "a route trace is attached already",
            ));
        }
        trace.fits(&self.slots, self.step.boundary.layout.handoff().n_used)?;
        self.step.trace = Some(trace);
        Ok(())
    }

    /// Detach the route trace; what it wrote stays on disk, completed only
    /// by [`RouteTrace::finish`].
    pub fn take_route_trace(&mut self) -> Option<RouteTrace> {
        self.step.trace.take()
    }

    /// Whether a route trace is attached.
    #[must_use]
    pub fn route_traced(&self) -> bool {
        self.step.trace.is_some()
    }

    /// A prompt call of `n` ids from cache position `pos0` begins, fed one
    /// step an id: the route trace, when one is attached, marks its
    /// positions ([`RouteTrace::prompt`]); nothing without one.
    pub fn route_prompt(&mut self, pos0: u32, n: usize) -> Result<(), GpuError> {
        match self.step.trace.as_mut() {
            Some(t) => t.prompt(pos0, n),
            None => Ok(()),
        }
    }

    /// The refusal of a prompt batch's service while a route trace is
    /// attached: the trace records the steps' services alone.
    fn refuse_traced_batch(&self, what: &'static str) -> Result<(), GpuError> {
        if self.step.trace.is_some() {
            return Err(GpuError::shape(
                what,
                "a prompt batch while a route trace is attached: the trace records the step \
                 feed (BLOOMERY_PREFILL=steps)",
            ));
        }
        Ok(())
    }

    /// The expert tier cards, in tier order; empty without a tier.
    #[must_use]
    pub fn tiers(&self) -> &[TierCard] {
        &self.tiers
    }

    /// The expert tier cards, for a gate that drives their streams.
    pub fn tiers_mut(&mut self) -> &mut [TierCard] {
        &mut self.tiers
    }

    /// What each tier has done since load, in tier order.
    #[must_use]
    pub fn tier_stats(&self) -> Vec<tier::TierStats> {
        self.tiers.iter().map(TierCard::stats).collect()
    }

    /// The experts of layer `layer` the tier cards hold, summed over the
    /// tiers, as the slot map says; a layer the map has no row for is
    /// refused by name.
    pub fn on_tier(&self, layer: usize) -> Result<usize, GpuError> {
        self.slots.on_tier(layer)
    }

    /// The experts of layer `layer` tier `tier` holds, as the slot map
    /// says; a layer the map has no row for, and a tier it does not name, are
    /// refused by name.
    pub fn on_tier_of(&self, tier: usize, layer: usize) -> Result<usize, GpuError> {
        self.slots.on_tier_of(tier, layer)
    }

    /// Row `row`'s handoff target and its tier image as tier `tier` reads
    /// it, for a tier layer's handoff launch; refused without a tier.
    pub fn tier_handoff(
        &mut self,
        row: usize,
        tier: usize,
    ) -> Result<(HandoffTarget<'_>, TierTarget<'_>), GpuError> {
        let page = self.tier_page.as_mut().ok_or(GpuError::state(
            "HostTier::tier_handoff",
            "an expert tier (HostTier::attach_tiers)",
        ))?;
        let target = self.step.boundary.handoff_target_of(row)?;
        Ok((target, page.target_of(row, tier)?))
    }

    /// Row `row`'s routed rows the tiers write, for a tier layer's join;
    /// refused without a tier.
    pub fn tier_rows(&self, row: usize) -> Result<&DeviceBuffer<f32>, GpuError> {
        self.page_ref("HostTier::tier_rows")?.rows_of(row)
    }

    /// Enqueue a tier layer's go of layer `layer`, row `row`, on the stage
    /// card's `stream`: one go batch with a go for each tier that holds
    /// experts of the layer ([`TierPage::enqueue_go_of`]); refused without a
    /// tier, and for a layer no tier holds an expert of.
    pub fn enqueue_tier_go(
        &self,
        stream: &CudaStream,
        layer: usize,
        row: usize,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "HostTier::enqueue_tier_go";
        let mut tiers = [0; MAX_TIERS];
        let k = tiers_of(self.tier_mask(layer)?, &mut tiers);
        self.page_ref(WHAT)?
            .enqueue_go_of(stream, &self.step.boundary, layer, row, &tiers[..k])
    }

    /// Enqueue a tier layer's wait of layer `layer`, row `row`, on the stage
    /// card's `stream`, for the host's counter and the counter of each tier
    /// that holds experts of the layer — the settle rule of
    /// [`TierPage::enqueue_back_of`]; refused without a tier, and for a layer
    /// no tier holds an expert of.
    pub fn enqueue_tier_back(
        &self,
        stream: &CudaStream,
        layer: usize,
        row: usize,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "HostTier::enqueue_tier_back";
        let mut tiers = [0; MAX_TIERS];
        let k = tiers_of(self.tier_mask(layer)?, &mut tiers);
        self.page_ref(WHAT)?
            .enqueue_back_of(stream, &self.step.boundary, layer, row, &tiers[..k])
    }

    fn page_ref(&self, what: &'static str) -> Result<&TierPage, GpuError> {
        self.tier_page.as_ref().ok_or(GpuError::state(
            what,
            "an expert tier (HostTier::attach_tiers)",
        ))
    }

    /// The tiers that hold experts of layer `layer`, one bit a tier; 0
    /// without a tier. A layer the slot map has no row for is refused by
    /// name.
    fn tier_mask(&self, layer: usize) -> Result<u32, GpuError> {
        let mut mask = 0;
        for t in 0..self.tiers.len() {
            if self.slots.on_tier_of(t, layer)? > 0 {
                mask |= 1 << t;
            }
        }
        Ok(mask)
    }

    /// The tier cards' names, in tier order, for a refusal or a loss.
    fn tier_names(&self, mask: u32) -> String {
        let names: Vec<&str> = self
            .tiers
            .iter()
            .enumerate()
            .filter(|(t, _)| mask >> t & 1 == 1)
            .map(|(_, t)| t.name())
            .collect();
        names.join(", ")
    }

    /// The refusal of the direct batch service on a host tier with an
    /// expert tier: it has no tier path, so it would leave the tier's
    /// experts uncomputed. Prompt batches go through the port
    /// ([`HostTier::serve_key`]).
    fn refuse_tier_batch(&self, what: &'static str) -> Result<(), GpuError> {
        if self.tiers.is_empty() {
            return Ok(());
        }
        Err(GpuError::protocol(
            what,
            format!(
                "the direct batch service has no tier path (it would leave the experts on {} \
                 uncomputed): a model with an expert tier serves prompt batches through the port \
                 (HostTier::serve_key)",
                self.tier_names(u32::MAX)
            ),
        ))
    }

    /// The refusal of an untiered route of a layer whose expert tier holds
    /// experts: without its tier places the tier would not serve the block.
    fn refuse_untiered_route(&self, what: &'static str, layer: usize) -> Result<(), GpuError> {
        if !self.tiers.is_empty() && self.slots.on_tier(layer)? > 0 {
            return Err(GpuError::shape(
                what,
                format!(
                    "layer {layer}: a tiered layer's route without its tier places \
                     (HostTier::enqueue_download_tiered)"
                ),
            ));
        }
        Ok(())
    }

    /// Hold `residency`, the host set the placed load read in and locked,
    /// for the tier's lifetime. Load-time only.
    pub fn keep_residency(&mut self, residency: HostResidency) {
        self.residency = Some(residency);
    }

    /// What the placed load did to the plan's host set, when the tier holds
    /// it.
    #[must_use]
    pub fn residency(&self) -> Option<&HostResidency> {
        self.residency.as_ref()
    }

    /// Watch `word`, the fault word of the `Gpu` the chain launches on
    /// ([`crate::Gpu::fault_word`]); a word of another context is refused. A
    /// batch service ([`HostTier::serve_batch`]) that meets input the card
    /// should already have refused then reads the word and fails as
    /// [`name_refusal`] names it; unwatched, it returns with NaN in the
    /// refused columns' sums and the refusal recorded. Step services never
    /// read it. Load-time only.
    pub fn watch_fault(&mut self, word: &Arc<DeviceBuffer<u32>>) -> Result<(), GpuError> {
        let ctx = self.step.boundary.region.context();
        self.fault = Some(crate::module_fault_word(ctx, word, "Hybrid::watch_fault")?);
        Ok(())
    }

    /// The batch port's two sets, for blocks of up to `cap` tokens, made
    /// once, and with an expert tier its tier leg
    /// ([`HostTier::prepare_tier_batch`]): a decode that never prefills a
    /// batch never holds them, unless a body with a tier makes them at load.
    /// Load-time or first-prompt only; a port already made for another `cap`
    /// is refused by name.
    pub fn prepare_batch(&mut self, ctx: &Arc<CudaContext>, cap: usize) -> Result<(), GpuError> {
        match self.port.as_ref().map(BatchPort::cap) {
            None => {
                let h = self.step.boundary.layout.handoff();
                self.port = Some(BatchPort::new(ctx, h.hidden, h.n_used, cap)?);
            }
            Some(made) if made != cap => {
                return Err(GpuError::shape(
                    "HostTier::prepare_batch",
                    format!("the batch port holds blocks of {made} tokens, asked for {cap}"),
                ));
            }
            Some(_) => {}
        }
        self.prepare_tier_batch()
    }

    /// The batch port, once [`HostTier::prepare_batch`] has made it.
    fn port_mut(&mut self, what: &'static str) -> Result<&mut BatchPort, GpuError> {
        self.port.as_mut().ok_or(GpuError::State {
            what,
            missing: "the batch port's sets (HostTier::prepare_batch)",
        })
    }

    /// Both batch sets free, at a group's start ([`BatchPort::begin`]); a
    /// poisoned tier is refused first, as [`BEGIN_GROUP`], so a call after a
    /// failed service — a lost tier card among them — enqueues nothing.
    pub fn begin_group(&mut self) -> Result<(), GpuError> {
        self.health.refuse_if_poisoned(BEGIN_GROUP)?;
        self.port_mut(BEGIN_GROUP)?.begin();
        Ok(())
    }

    /// Enqueue the copies of `key`'s tokens' activations `x`, weights `w`
    /// and ids to the host into the next batch set ([`BatchPort::download`]).
    pub fn enqueue_download(
        &mut self,
        stream: &CudaStream,
        xw: [&DeviceBuffer<f32>; 2],
        ids: &DeviceBuffer<u32>,
        key: BatchKey,
    ) -> Result<(), GpuError> {
        self.refuse_untiered_route("HostTier::enqueue_download", key.layer)?;
        self.port_mut("HostTier::enqueue_download")?
            .download(stream, xw, ids, key)
    }

    /// [`HostTier::enqueue_download`] from routing buffers of `pitch`
    /// entries a token whose first `n_used` are the routed slots
    /// ([`BatchPort::download_pitched`]).
    pub fn enqueue_download_pitched(
        &mut self,
        stream: &CudaStream,
        xw: [&DeviceBuffer<f32>; 2],
        ids: &DeviceBuffer<u32>,
        pitch: usize,
        key: BatchKey,
    ) -> Result<(), GpuError> {
        self.refuse_untiered_route("HostTier::enqueue_download_pitched", key.layer)?;
        self.port_mut("HostTier::enqueue_download_pitched")?
            .download_pitched(stream, xw, ids, pitch, key)
    }

    /// Wait for the oldest download not served yet — `key`'s, else refused
    /// by name — and hand back its routed ids without serving it
    /// ([`BatchPort::routed_ids`]).
    pub fn routed_ids(&mut self, key: BatchKey) -> Result<&[u32], GpuError> {
        let swap = self.swap.as_ref();
        let port = self.port.as_mut().ok_or(GpuError::State {
            what: "HostTier::routed_ids",
            missing: "the batch port's sets (HostTier::prepare_batch)",
        })?;
        port.routed_ids(key).map_err(|e| stall_noted(swap, e))
    }

    /// Wait for the oldest download not served yet — `key`'s, else refused
    /// by name — and serve its layer's host experts for its tokens in one
    /// union call ([`HostTier::serve_batch`]'s service), the sums into the
    /// set the upload sends; a tiered set's tier service is enqueued between
    /// the wait and the union ([`HostTier::serve_port`]). A batch service's
    /// refusal reached no step port, so it is handed to the caller's naming
    /// ([`HostTier::take_step_refusal`]) here, as a step service's refusal
    /// is by its own record. A planted refusal
    /// ([`HostTier::plant_refusal`]) is returned here, as the service's own
    /// refusal of this layer's input. Returns the host time outside the
    /// union call.
    pub fn serve_key(&mut self, key: BatchKey) -> Result<ServeTimes, GpuError> {
        self.refuse_traced_batch("HostTier::serve_key")?;
        let r = match self.planted.take() {
            Some(detail) => Err(self.refuse_planted_batch(key.layer, detail)),
            None => self.serve_port(key),
        };
        self.hand_refusal_to_step();
        r
    }

    /// A batch service's refusal reached no step port: hand the poison's own
    /// refusal to the caller's naming ([`HostTier::take_step_refusal`]), as
    /// a step service's refusal hands itself — once, by the take.
    fn hand_refusal_to_step(&mut self) {
        if self.step.step_refusal.is_none()
            && self.health.poisoned
            && let Some(Poison::Refused(r)) = &self.health.poison
        {
            self.step.step_refusal = Some(r.clone());
        }
    }

    /// Enqueue the copy of the oldest served set's host sums — `key`'s, else
    /// refused by name — to `hsum` ([`BatchPort::upload`]).
    pub fn enqueue_upload(
        &mut self,
        stream: &CudaStream,
        hsum: &mut DeviceBuffer<f32>,
        key: BatchKey,
    ) -> Result<(), GpuError> {
        self.port_mut("HostTier::enqueue_upload")?
            .upload(stream, hsum, key)
    }

    /// The refusal that failed a service and so poisoned the tier, else the
    /// first one a returning (unwatched batch) service recorded — what the
    /// caller of a failed step names with [`name_refusal`] once the stream
    /// has drained.
    #[must_use]
    pub fn refusal(&self) -> Option<&Refusal> {
        self.health.refusal.as_ref()
    }

    /// The refusal the tier is poisoned by now, if it is [`Poison::Refused`]:
    /// what a model that parks another slot asks of a tier refusal it never
    /// recorded ([`crate::model::HostServed::refusal_poison`]), whose slot is
    /// unknown. A refusal a reset lifted is kept, not poisoned: this is
    /// none for it.
    #[must_use]
    pub fn refusal_poison(&self) -> Option<Refusal> {
        if !self.health.poisoned {
            return None;
        }
        match &self.health.poison {
            Some(Poison::Refused(r)) => Some(r.clone()),
            _ => None,
        }
    }

    /// The refusal that failed the last step service, once: what the step's
    /// caller names with [`name_refusal`] after the stream has drained. `None`
    /// when the step failed for another reason, or was already named.
    pub fn take_step_refusal(&mut self) -> Option<Refusal> {
        self.step.step_refusal.take()
    }

    /// Why the last service that failed poisoned the tier, kept across a
    /// reset; `None` if none has.
    #[must_use]
    pub fn last_poison(&self) -> Option<&Poison> {
        self.health.poison.as_ref()
    }

    /// Refuse `what` while a failed service has the tier poisoned
    /// ([`GpuError::HostPoisoned`], naming the poison): what a body checks
    /// before anything else a call of its own would refuse, since after a
    /// failed service the poison is the cause.
    pub fn refuse_if_poisoned(&self, what: &'static str) -> Result<(), GpuError> {
        self.health.refuse_if_poisoned(what)
    }

    /// Plant a refusal the next host service returns — a step service's
    /// ([`HostTier::serve`]) or a batch service's ([`HostTier::serve_key`],
    /// [`HostTier::serve_batch`]) — refusing its layer's input as `detail`
    /// says: a test seam for the slots harness's poison clauses
    /// (`crates/gpu-gates/src/slots_gate.rs` H5), the tier's input checks
    /// fire on state the card should already have refused, which no sound
    /// card produces, so a gate reaches a refusal through here. The service
    /// runs the whole of a refusing one's path: the refusal recorded for the
    /// caller's naming, the tier poisoned as [`Poison::Refused`], a step
    /// service's every wait released, the service's error returned. Gate
    /// use; never on a serving path.
    pub fn plant_refusal(&mut self, detail: &str) {
        self.planted = Some(detail.to_string());
    }

    /// The protocol's words as they stand now. On a drained stream a sound
    /// tier is [`HybridWords::at_rest`].
    #[must_use]
    pub fn words(&self) -> HybridWords {
        let page = &self.step.boundary.page;
        let rows = self.step.boundary.rows();
        let mut counters = [0u32; MAX_ROWS];
        for (row, c) in counters.iter_mut().enumerate().take(rows) {
            *c = word(page, Word::Cnt(row)).load(Ordering::Acquire);
        }
        HybridWords {
            generation: word(page, Word::Gen).load(Ordering::Acquire),
            served: self.step.served,
            counters,
            rows,
        }
    }

    /// Lift a refusal's poison, so that the next step is served as on a
    /// fresh tier; the model's reset calls it before it empties the caches.
    ///
    /// A tier poisoned by refused input is sound: the service refused before
    /// any host expert ran, and released the stream. Nothing that could still
    /// write the words runs: this call holds the tier mutably, so no service
    /// is running on any thread; every pool job a service dispatched ended
    /// before the service returned or unwound (`for_each_chunk` returns, and
    /// replays a chunk's panic, only once every worker has marked the job
    /// done), and the pool's threads only read the generation; a refusal
    /// means the go landed, so the replay's graph launch had been issued, and
    /// the replay read its launch thread's answer before it returned. The
    /// card is the last writer, and the synchronize below waits it out. Then
    /// the words take a fresh tier's relation: every counter at 0 — the
    /// release left `RELEASE` less the waits that drained through it, and a
    /// counter left there lets every later wait pass with no service — and
    /// `served` at the card's generation. A go adds one to the generation
    /// and to the sequence in one batch, and a handoff carries the sequence
    /// as it stood before its go, so the next go brings the generation to
    /// `served + 1` with the handoff's sequence at `served`: what a service
    /// waits for and checks, as on a fresh tier at 0. A captured chain keeps
    /// its nodes and addresses; its next replay meets these words. An
    /// attached route trace drops the position the refusal cut short
    /// (`RouteTrace::abandon_position`): its layers were never written.
    ///
    /// A tier poisoned any other way — a failure of its own, a panic in the
    /// host experts — stays poisoned and the call fails by name: only a
    /// reload clears it. On a tier that is not poisoned the call changes
    /// nothing, and fails by name when the stream has drained and the words
    /// are not at rest; a stream still running may hold a signal its wait has
    /// yet to take back.
    ///
    /// A refusal whose slots the model poisoned ([`PoisonWhy::Refused`])
    /// does not lift here: a body that serves several slots settles on every
    /// reset ([`HostTier::settle`]) and its model lifts the tier once every
    /// slot the refused call ran on has been reset
    /// ([`HostTier::lift_refusal`]).
    pub fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        const WHAT: &str = "Hybrid::reset";
        self.abandon_pass();
        if !self.health.poisoned {
            return self.settle_checks(stream, WHAT);
        }
        if let Some(e) = self.unlifted(WHAT) {
            return Err(e);
        }
        self.lift_words(stream, WHAT)
    }

    /// [`HostTier::reset`] without the refusal lift: the settling every
    /// reset of a body that serves several slots runs
    /// ([`crate::model::ChainBody::reset`]), the model naming when the
    /// tier's refusal poison lifts ([`HostTier::lift_refusal`], called
    /// through [`crate::model::HostServed`]). A sound tier is checked at
    /// rest as [`HostTier::reset`] checks it; a poisoned one is left as it
    /// stands — a refusal's poison waiting for the model's lift, any other
    /// poison failing by name as the reset's own arm does.
    pub fn settle(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        const WHAT: &str = "Hybrid::settle";
        self.abandon_pass();
        if !self.health.poisoned {
            return self.settle_checks(stream, WHAT);
        }
        match self.unlifted(WHAT) {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// [`HostTier::reset`]'s refusal lift alone, for
    /// [`crate::GpuModel::reset`] to call through
    /// [`crate::model::HostServed::lift_refusal`] once every slot a refused
    /// call ran on has been reset — or on any reset, for a tier refusal the
    /// model never recorded. Nothing on a sound tier; any poison but a
    /// refusal fails by name as [`HostTier::reset`]'s own arm does.
    pub fn lift_refusal(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        const WHAT: &str = "Hybrid::lift_refusal";
        if !self.health.poisoned {
            return Ok(());
        }
        if let Some(e) = self.unlifted(WHAT) {
            return Err(e);
        }
        self.lift_words(stream, WHAT)
    }

    /// A sound tier's reset check: once the stream has drained, every tier
    /// at rest and its faults cleared, the words at rest. A stream still
    /// running passes — it may hold a signal its wait has yet to take back.
    fn settle_checks(&mut self, stream: &CudaStream, what: &'static str) -> Result<(), GpuError> {
        if !stream_idle(stream)? {
            return Ok(());
        }
        let w = self.words();
        if let Some(page) = self.tier_page.as_ref() {
            for t in &mut self.tiers {
                drain_within(
                    t.gpu().stream(),
                    DRAIN_DEADLINE,
                    "Hybrid::reset: the expert tier's stream",
                )?;
                if !t.at_rest(page) {
                    return Err(GpuError::protocol(
                        what,
                        format!(
                            "the stream has drained and the expert tier is not at rest \
                             (progress {} of {} asked): a go the tier never took, or a signal \
                             no wait took back",
                            t.progress(page),
                            t.issued()
                        ),
                    ));
                }
                t.clear_fault(page)?;
            }
        }
        if !w.at_rest() {
            return Err(GpuError::protocol(
                what,
                format!(
                    "the stream has drained and the host tier is not at rest ({w}): a go the \
                     host never served, or a signal no wait took back"
                ),
            ));
        }
        Ok(())
    }

    /// A poison no reset lifts, as the error that says so; `None` for a
    /// refusal, the one a reset can lift.
    fn unlifted(&self, what: &'static str) -> Option<GpuError> {
        let cause = match &self.health.poison {
            Some(Poison::Refused(_)) => return None,
            Some(p) => p.to_string(),
            None => "a failure with no cause recorded".to_string(),
        };
        Some(GpuError::protocol(
            what,
            format!(
                "the host tier is poisoned ({cause}); a reset lifts only refused input — \
                 reload the model"
            ),
        ))
    }

    /// The refusal lift itself ([`HostTier::reset`]'s protocol): the stream
    /// synchronized, the card's generation and sequence checked equal, every
    /// counter back to 0, `served` at the generation, every tier reset, the
    /// route trace's cut-short position dropped and the poison cleared.
    fn lift_words(&mut self, stream: &CudaStream, what: &'static str) -> Result<(), GpuError> {
        stream.synchronize()?;
        let boundary = &self.step.boundary;
        let generation = word(&boundary.page, Word::Gen).load(Ordering::Acquire);
        let mut seq = [0u32; 1];
        boundary.seq_word().copy_to_host(stream, &mut seq)?;
        if seq[0] != generation {
            return Err(GpuError::protocol(
                what,
                format!(
                    "the card's generation is {generation} and its sequence {}: a go added to one \
                     and not the other — reload the model",
                    seq[0]
                ),
            ));
        }
        for row in 0..boundary.rows() {
            word(&boundary.page, Word::Cnt(row)).store(0, Ordering::Release);
        }
        self.step.rest_at(generation);
        if let Some(page) = self.tier_page.as_ref() {
            for (t, goes) in self.tiers.iter_mut().zip(&mut self.tier_goes) {
                t.reset(page)?;
                *goes = t.progress(page);
            }
        }
        if let Some(t) = self.step.trace.as_mut() {
            t.abandon_position();
        }
        let h = &mut self.health;
        h.poisoned = false;
        h.refusal = None;
        h.failing = None;
        h.stats.resets += 1;
        Ok(())
    }

    /// A pass its caller never kept (a failed pass, a verify the reset
    /// drops before its commit) keeps no row: its noted ids are forgotten and
    /// the next boundary ends it at 0 rows. A kept count already given
    /// stands.
    fn abandon_pass(&mut self) {
        if let Some(m) = &self.swap
            && m.pass_open()
            && self.swap_kept.is_none()
        {
            self.step.tally.clear();
            self.swap_kept = Some((KeptRows::prefix(0), PassKind::Abandoned));
        }
    }

    /// The boundary the chain enqueues its handoffs and waits on.
    #[must_use]
    pub fn boundary(&self) -> &Boundary {
        &self.step.boundary
    }

    /// The boundary, for a chain whose launches write its handoff.
    pub fn boundary_mut(&mut self) -> &mut Boundary {
        &mut self.step.boundary
    }

    /// The slot map the tier serves by: which experts each layer's card
    /// stack holds.
    #[must_use]
    pub fn slots(&self) -> &SlotMap {
        &self.slots
    }

    /// The boundary to enqueue a chain's handoffs on, and the slot map that
    /// says which of a layer's slots the card computes.
    pub fn boundary_and_slots(&mut self) -> (&mut Boundary, &SlotMap) {
        (&mut self.step.boundary, &self.slots)
    }

    /// The step port.
    #[must_use]
    pub fn step_port(&self) -> &StepPort {
        &self.step
    }

    /// The go waits of step services `from..to` (counts of
    /// [`HybridStats::gaps`]): the card's time between the host's signal and
    /// the next go, per service. No allocation.
    pub fn gap_summary(&self, from: u64, to: u64) -> Result<step::GapSummary, GpuError> {
        self.step.gap_summary(from, to)
    }

    /// What the host side has done since load: the step port's, the batch
    /// service's and the tier's own counters, as one view.
    #[must_use]
    pub fn stats(&self) -> HybridStats {
        let (s, b, t) = (&self.step.stats, &self.batch.stats, &self.health.stats);
        HybridStats {
            served: s.served,
            go_early: s.go_early,
            go_early_first: s.go_early_first,
            host_slots: s.host_slots,
            host_w2: s.host_w2,
            leg_ns: s.leg_ns,
            parks_in_service: s.parks_in_service,
            straggle_ns: s.straggle_ns,
            straggle_max_ns: s.straggle_max_ns,
            overlap_slots: s.overlap_slots,
            pair_row1_slots: s.pair_row1_slots,
            host_calls: s.host_calls,
            cols_served: s.cols_served,
            cols_cols: s.cols_cols,
            gaps: s.gaps,
            batch_served: b.batch_served,
            batch_cols: b.batch_cols,
            batch_host_slots: b.batch_host_slots,
            batch_ns: b.batch_ns,
            batch_excluded_slots: b.batch_excluded_slots,
            refusals: t.refusals,
            resets: t.resets,
            last_poison: t.last_poison,
        }
    }

    /// The host experts, for a caller that prepares their scratch ahead of a
    /// service.
    pub fn host_mut(&mut self) -> &mut H {
        &mut self.experts
    }

    /// Row 0's host sum as the page holds it now: the last layer served
    /// for that row.
    pub fn hsum_copy(&self) -> Result<Vec<f32>, GpuError> {
        self.step.boundary.hsum_copy("Hybrid::hsum_copy")
    }

    /// Open a one-token chain on `stream`: a capture records which layers
    /// its replays will ask for; an eager chain is served as it goes.
    pub fn begin_chain(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.begin_chain_of(stream, Chain::Step)
    }

    /// [`HostTier::begin_chain`] of `chain`; a pair needs a boundary of two
    /// rows.
    pub fn begin_chain_of(&mut self, stream: &CudaStream, chain: Chain) -> Result<(), GpuError> {
        self.health.refuse_if_poisoned("Hybrid::begin_chain")?;
        self.step.begin(stream, chain)?;
        self.note_capture();
        Ok(())
    }

    /// A chain being captured records a new go order: the tier's graph of
    /// that chain is captured again at its next replay.
    fn note_capture(&mut self) {
        if self.step.is_capturing() {
            let chain = self.step.chain();
            for t in &mut self.tiers {
                t.forget_order(chain);
            }
        }
    }

    /// Open the chain of a step walk of `units` rows of `cols` columns
    /// ([`StepPort`]'s point): one row of one column is [`Chain::Step`], two
    /// [`Chain::Pair`], one row of 2 up to the page's columns
    /// [`Chain::Cols`]; any other point is refused by name.
    pub fn open_step(
        &mut self,
        stream: &CudaStream,
        units: usize,
        cols: usize,
    ) -> Result<(), GpuError> {
        self.health.refuse_if_poisoned("Hybrid::begin_chain")?;
        self.step.open(stream, units, cols)?;
        self.note_capture();
        Ok(())
    }

    /// Layer `layer`'s hybrid work is enqueued: a capture notes it, an eager
    /// chain serves it now, before anything more joins the stream behind its
    /// wait. Row 0's.
    pub fn layer_enqueued(&mut self, layer: usize) -> Result<(), GpuError> {
        self.row_enqueued(layer, 0)
    }

    /// [`HostTier::layer_enqueued`] for row `row`. The chain enqueues its
    /// services' waits in go order, and so is served in it.
    pub fn row_enqueued(&mut self, layer: usize, row: usize) -> Result<(), GpuError> {
        if self.step.note(layer, row) {
            return Ok(());
        }
        let chain = self.step.chain();
        self.refuse_released(SERVE)?;
        let mask = self.tier_mask_in_service(layer)?;
        if mask != 0 {
            self.feed_tiers(layer, row, mask, chain)?;
        }
        self.serve(layer, row, false, chain)?;
        if mask != 0 {
            self.settle_tiers(layer, mask)
        } else {
            Ok(())
        }
    }

    /// Serve every hybrid layer a replay of the captured one-token chain
    /// submitted, in chain order.
    pub fn serve_captured(&mut self) -> Result<(), GpuError> {
        self.serve_captured_of(Chain::Step)
    }

    /// [`HostTier::serve_captured`] for a replay of `chain`'s capture.
    ///
    /// With expert tiers each tier's graph of `chain` is launched first when
    /// one is held for that tier's layers of the chain's go order; for a tier
    /// with none, the host enqueues each of its layers on its stream as it
    /// serves it, and captures its graph once the pass has settled
    /// ([`tier::Pass`]). Once the last layer is served the host waits for
    /// every tier with a layer in the pass to catch up and reads their fault
    /// copies ([`HostTier::settle_tiers`]).
    pub fn serve_captured_of(&mut self, chain: Chain) -> Result<(), GpuError> {
        let k = self.tiers.len();
        let mut pass = [Pass::Graph; MAX_TIERS];
        let (mut listed, mut last) = (0u32, None);
        if k > 0 {
            self.refuse_released(SERVE)?;
            for list in &mut self.tier_lists {
                list.clear();
            }
            let mut i = 0;
            while let Some((l, r)) = self.step.captured(chain, i) {
                let mask = self.tier_mask_in_service(l)?;
                for t in (0..k).filter(|t| mask >> t & 1 == 1) {
                    self.tier_lists[t].push((l, r));
                }
                if mask != 0 {
                    last = Some(l);
                }
                listed |= mask;
                i += 1;
            }
            for t in 0..k {
                match self.tiers[t].replay(chain, &self.tier_lists[t]) {
                    Ok(p) => pass[t] = p,
                    Err(e) => {
                        let at = self.tier_lists[t].first().map_or(0, |p| p.0);
                        return Err(self.fail_released(SERVE, at, e));
                    }
                }
            }
        }
        let fed = (0..k)
            .filter(|&t| pass[t] == Pass::Feed)
            .fold(0u32, |m, t| m | 1 << t);
        let mut i = 0;
        while let Some((layer, row)) = self.step.captured(chain, i) {
            if fed != 0 {
                let mask = self.tier_mask_in_service(layer)? & fed;
                if mask != 0 {
                    self.feed_tiers(layer, row, mask, chain)?;
                }
            }
            self.serve(layer, row, i == 0, chain)?;
            i += 1;
        }
        let Some(last) = last else {
            return Ok(());
        };
        self.settle_tiers(last, listed)?;
        for t in (0..k).filter(|t| fed >> t & 1 == 1) {
            let page = self
                .tier_page
                .as_ref()
                .expect("a host tier with tiers holds their page");
            if let Err(e) = self.tiers[t].capture(page, chain, &self.tier_lists[t]) {
                return Err(self.fail_released(SERVE, last, e));
            }
        }
        Ok(())
    }

    /// Enqueue tier layer `layer` of row `row` of `chain` on the streams of
    /// the tiers whose bits `mask` sets now, for an eager chain or a fed
    /// pass; a failure poisons the tier and releases every card's waits.
    fn feed_tiers(
        &mut self,
        layer: usize,
        row: usize,
        mask: u32,
        chain: Chain,
    ) -> Result<(), GpuError> {
        let r = match self.tier_page.as_ref() {
            Some(page) => self
                .tiers
                .iter_mut()
                .enumerate()
                .filter(|(t, _)| mask >> t & 1 == 1)
                .try_for_each(|(_, t)| t.enqueue_eager(page, chain, layer, row)),
            None => Ok(()),
        };
        if let Err(e) = r {
            return Err(self.fail_released(SERVE, layer, e));
        }
        Ok(())
    }

    /// The tiers that hold experts of layer `layer`, one bit a tier (0
    /// without a tier), asked while a step's service is under way: the stage
    /// card may wait at that layer already, so a layer the slot map has no
    /// row for fails the service — the tier poisoned, every card's waits
    /// released — instead of returning with the card left waiting.
    fn tier_mask_in_service(&mut self, layer: usize) -> Result<u32, GpuError> {
        if self.tiers.is_empty() {
            return Ok(0);
        }
        match self.tier_mask(layer) {
            Ok(mask) => Ok(mask),
            Err(e) => Err(self.fail_released(SERVE, layer, e)),
        }
    }

    /// A service `what` of layer `layer` failed with `e` after either card
    /// may have been given work: poison the tier, release every card's
    /// waits, and return `e`.
    fn fail_released(&mut self, what: &'static str, layer: usize, e: GpuError) -> GpuError {
        self.health.set_poison(what, layer, Some(&e), None);
        self.release_all();
        e
    }

    /// Wait under one go deadline for each tier whose bit `mask` sets to
    /// serve every layer asked of it — `layer` the last the host served —
    /// then read their fault copies: a raised copy is the step's fault,
    /// merged with the stage card's word once that card's stream has drained
    /// (the first layer wins). A tier that does not catch up in time is a
    /// lost card: the tier is poisoned, every card's waits released, and the
    /// error names each tier still behind.
    fn settle_tiers(&mut self, layer: usize, mask: u32) -> Result<(), GpuError> {
        let Some(page) = self.tier_page.as_ref() else {
            return Ok(());
        };
        let t0 = Instant::now();
        for (t, card) in self.tiers.iter_mut().enumerate() {
            if mask >> t & 1 == 1 && !card.wait_caught_up(page, t0 + GO_DEADLINE) {
                let behind = self.behind_mask(mask) | 1 << t;
                return Err(self.lose_tiers(SERVE, layer, behind, t0.elapsed()));
            }
        }
        match tier::merged_fault(page, &self.tiers, mask)? {
            Some(f) => Err(GpuError::fault(SERVE, f)),
            None => Ok(()),
        }
    }

    /// The tiers of `mask` that have served fewer layers than the host
    /// asked of them, one bit a tier.
    fn behind_mask(&self, mask: u32) -> u32 {
        let Some(page) = self.tier_page.as_ref() else {
            return 0;
        };
        self.tiers
            .iter()
            .enumerate()
            .filter(|(t, c)| mask >> t & 1 == 1 && c.behind(page))
            .fold(0, |m, (t, _)| m | 1 << t)
    }

    /// The expert tiers whose bits `behind` sets are lost to service `what`
    /// at layer `layer`, after the host `waited` for them: poison the tier as
    /// a lost card, release every card's waits, and name each card.
    fn lose_tiers(
        &mut self,
        what: &'static str,
        layer: usize,
        behind: u32,
        waited: Duration,
    ) -> GpuError {
        self.lose_tiers_as(what, layer, behind, |t, page| t.lost_detail(page, waited))
    }

    /// [`HostTier::lose_tiers`], each loss named by `detail` of its tier over
    /// the tiers' page, in tier order. With no tier of `behind` attached there
    /// is none to lose: the service fails by name instead, the tier poisoned
    /// and every card's waits released.
    fn lose_tiers_as(
        &mut self,
        what: &'static str,
        layer: usize,
        behind: u32,
        detail: impl Fn(&TierCard, &TierPage) -> String,
    ) -> GpuError {
        let details: Vec<String> = match self.tier_page.as_ref() {
            Some(page) => self
                .tiers
                .iter()
                .enumerate()
                .filter(|(t, _)| behind >> t & 1 == 1)
                .map(|(_, t)| detail(t, page))
                .collect(),
            None => Vec::new(),
        };
        if details.is_empty() {
            let e = GpuError::state(what, "an expert tier to lose (HostTier::attach_tiers)");
            return self.fail_released(what, layer, e);
        }
        let (name, detail) = (self.tier_names(behind), details.join("; "));
        self.health.set_lost(what, &name, layer, &detail);
        self.release_all();
        GpuError::protocol(what, format!("layer {layer}: {detail}"))
    }

    /// Release every wait still pending on every card.
    fn release_all(&self) {
        self.step.boundary.release();
        if let Some(page) = self.tier_page.as_ref() {
            for t in &self.tiers {
                t.release(page);
            }
        }
    }

    /// Refuse `what` on a poisoned tier, naming the poison, with every
    /// card's waits released first: a step service is asked after its
    /// chain's launch, so the card already waits on the host for it, and a
    /// refusal that left those waits standing would hang the next read of
    /// the stream instead of naming the poison.
    fn refuse_released(&self, what: &'static str) -> Result<(), GpuError> {
        let r = self.health.refuse_if_poisoned(what);
        if r.is_err() {
            self.release_all();
        }
        r
    }

    /// Serve layer `layer` for an eager batch of `x.ne1()` tokens in one
    /// union call, outside the go/wait protocol: the caller has brought the
    /// batch's handoffs to the host — `x`, one normed activation per column,
    /// read where they landed, and `ids` and `weights`, the routing, `n_used`
    /// per column in slot order — and reads the host sums back from `out`,
    /// `hidden` per column. Column `j`'s list is its slots the slot map sends
    /// to the host, in slot order, less those whose expert is in `exclude`:
    /// the list a step's service builds for that token when `exclude` is
    /// empty. `exclude` names experts of the layer's stack the slot map sends
    /// to the host, strictly ascending; an unsorted set, an id twice, an id
    /// past the stack or one of the card's is refused by name before anything
    /// runs. A column whose activation is not finite, or whose routing names
    /// an id the slot map does not know, is input the card should already
    /// have refused: its sum is NaN and no host expert runs for it, the other
    /// columns are served as always, the refusal is recorded, and the call
    /// fails as [`HostTier::watch_fault`] describes — unwatched it returns.
    /// The caller brought the handoffs down behind the card's event, so the
    /// kernels that wrote them have finished and a watched read of the fault
    /// word sees what they raised. Refused on a poisoned tier; a failure, or
    /// a panic inside the host experts, poisons it. A planted refusal
    /// ([`HostTier::plant_refusal`]) is returned here, as the service's own
    /// refusal of this layer's input.
    pub fn serve_batch(
        &mut self,
        layer: usize,
        x: Tensor2View<'_>,
        ids: &[u32],
        weights: &[f32],
        exclude: &[u32],
        out: &mut [f32],
    ) -> Result<(), GpuError> {
        self.refuse_tier_batch("Hybrid::serve_batch")?;
        self.refuse_traced_batch("Hybrid::serve_batch")?;
        if let Some(detail) = self.planted.take() {
            return Err(self.refuse_planted_batch(layer, detail));
        }
        let h = self.step.boundary.layout.handoff();
        let t = Tier {
            experts: &mut self.experts,
            health: &mut self.health,
            slots: &self.slots,
            fault: self.fault.as_ref(),
            hidden: h.hidden,
            n_used: h.n_used,
        };
        self.batch
            .serve_guarded(t, layer, x, ids, weights, exclude, out)
    }

    /// Serve layer `layer` of row `row` in `chain`. Any failure — an error
    /// or a panic inside the host experts, or a poisoned tier's refusal —
    /// releases every pending wait first, so the stream drains instead of
    /// hanging a later synchronize. A planted refusal
    /// ([`HostTier::plant_refusal`]) is returned here, as the service's own
    /// refusal of this layer's input.
    fn serve(
        &mut self,
        layer: usize,
        row: usize,
        opens_replay: bool,
        chain: Chain,
    ) -> Result<(), GpuError> {
        self.refuse_released(SERVE)?;
        if let Some(detail) = self.planted.take() {
            return Err(self.refuse_planted(layer, detail));
        }
        let (step, experts, health) = (&mut self.step, &mut self.experts, &mut self.health);
        let slots = &self.slots;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            step.serve_one(
                experts,
                &mut *health,
                slots,
                layer,
                row,
                opens_replay,
                chain,
            )
        }));
        match r {
            Ok(Ok(())) => {
                let mask = self.tier_mask_in_service(layer)?;
                for (t, card) in self.tiers.iter_mut().enumerate() {
                    if mask >> t & 1 == 1 {
                        self.tier_goes[t] = self.tier_goes[t].wrapping_add(1);
                        card.hit(layer, self.step.tier_slots[t]);
                    }
                }
                Ok(())
            }
            Ok(Err(e)) => {
                // The stage card waits at a tier layer the host has served
                // while a tier has not: past the grace, that tier is the one
                // behind.
                let t0 = Instant::now();
                let mut behind = 0u32;
                if let Some(page) = self.tier_page.as_ref() {
                    for (t, card) in self.tiers.iter().enumerate() {
                        if !card.wait_for(page, self.tier_goes[t], t0 + tier::TIER_GRACE) {
                            behind |= 1 << t;
                        }
                    }
                }
                if behind != 0 {
                    return Err(self.lose_tiers(SERVE, layer, behind, t0.elapsed()));
                }
                Err(self.fail_released(SERVE, layer, e))
            }
            Err(p) => {
                self.health.set_poison(SERVE, layer, None, Some(&*p));
                self.release_all();
                std::panic::resume_unwind(p)
            }
        }
    }

    /// A planted refusal of layer `layer`
    /// ([`HostTier::plant_refusal`]): the refusing service's whole path —
    /// the refusal recorded for the caller's naming
    /// ([`HostTier::take_step_refusal`], [`HostTier::refusal`]), the tier
    /// poisoned as [`Poison::Refused`] with the refusal its cause, every
    /// wait released so the stream drains ([`HostTier::serve`]'s failure
    /// arm) — and the error a refusing service returns.
    fn refuse_planted(&mut self, layer: usize, detail: String) -> GpuError {
        let r = Refusal {
            what: SERVE,
            layer,
            detail,
        };
        let e = GpuError::protocol(
            SERVE,
            format!(
                "host saw undefined input: layer {}, {}; the card's fault word is read once the \
                 step drains",
                r.layer, r.detail
            ),
        );
        self.step.step_refusal = Some(r.clone());
        self.health.record_refusal(r, true);
        self.health.set_poison(SERVE, layer, None, None);
        self.release_all();
        e
    }

    /// A planted refusal of layer `layer`, a batch service's
    /// ([`HostTier::plant_refusal`], [`HostTier::serve_key`],
    /// [`HostTier::serve_batch`]): the refusing batch service's path — the
    /// refusal recorded ([`HostTier::refusal`]), the tier poisoned as
    /// [`Poison::Refused`] with the refusal its cause — with the error a
    /// failing batch service returns. Like a batch service's own refusal it
    /// reaches no step port: [`HostTier::serve_key`] hands it to the
    /// caller's naming. No wait to release: a service that has not run left
    /// none.
    fn refuse_planted_batch(&mut self, layer: usize, detail: String) -> GpuError {
        let r = Refusal {
            what: batch::SERVE_BATCH,
            layer,
            detail,
        };
        let e = GpuError::protocol(
            batch::SERVE_BATCH,
            format!(
                "host saw undefined input: layer {}, column 0: {}",
                r.layer, r.detail
            ),
        );
        self.health.record_refusal(r, true);
        self.health
            .set_poison(batch::SERVE_BATCH, layer, None, None);
        e
    }
}

/// What a panic carried, as text.
fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with no message".to_string())
}

/// What the host saw in activation values that are not all finite: the first
/// such value and its index. `None` when all are finite.
fn non_finite(x: &[f32]) -> Option<String> {
    // A fold, not `all`: no early exit, so the test vectorizes.
    if x.iter().fold(true, |ok, v| ok & v.is_finite()) {
        return None;
    }
    let (i, v) = x.iter().enumerate().find(|(_, v)| !v.is_finite())?;
    Some(format!(
        "the activation holds {v} at value {i} of {}",
        x.len()
    ))
}

/// What the host saw in a routing whose slot `slot` names `id`, an expert
/// the slot map's `n_expert` do not include.
fn unknown_id(slot: usize, id: u32, n_expert: usize) -> String {
    format!("routed slot {slot} names expert {id}, past the slot map's {n_expert} experts")
}

/// The error the caller sees for refusal `r`, given `fault`, what the card's
/// fault word held once every launch before the refusal had finished: the
/// card's fault when it names `r`'s layer or an earlier one — or no layer
/// (an unlabelled launch), which cannot be ruled out — else the host's own
/// error, saying what the word holds. The one rule for both the step's
/// caller and a watched batch service.
#[must_use]
pub fn name_refusal(r: &Refusal, fault: Option<Fault>) -> GpuError {
    let word = match fault {
        Some(fault)
            if fault.layer == LAYER_NONE
                || usize::try_from(fault.layer).is_ok_and(|l| l <= r.layer) =>
        {
            return GpuError::fault(r.what, fault);
        }
        Some(f) => format!("first names {f:#} — later than this layer"),
        None => "is clean".to_owned(),
    };
    GpuError::protocol(
        r.what,
        format!(
            "host saw undefined input the card did not refuse: layer {}, {}; the card's fault \
             word {word}",
            r.layer, r.detail
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::{PassKind, SERVED_TIERS, await_within, pass_kept, refuse_tier_count};
    use crate::GpuError;
    use std::time::Duration;

    /// A bounded engine wait returns once its query says done, and past its
    /// bound is a named stall that carries the wait, its time and the bound.
    #[test]
    fn a_bounded_wait_ends_done_or_stalled_by_name() {
        let mut polls = 0;
        let done = await_within("t", Duration::from_secs(5), Duration::ZERO, || {
            polls += 1;
            Ok(polls == 3)
        });
        assert!(done.is_ok() && polls == 3);
        let bound = Duration::from_millis(20);
        let err = await_within("Head::tokens", bound, Duration::from_millis(5), || {
            Ok(false)
        })
        .expect_err("never done");
        match &err {
            GpuError::Stalled {
                what,
                waited,
                bound: b,
                note,
            } => assert!(*what == "Head::tokens" && waited > b && *b == bound && note.is_empty()),
            e => panic!("not a stall: {e}"),
        }
        assert!(
            err.to_string()
                .starts_with("Head::tokens: the card's work did not finish in "),
            "{err}"
        );
    }

    /// A load of one tier card, or none, passes; a second tier card is
    /// refused by name before anything uploads.
    #[test]
    fn a_second_tier_card_is_refused_by_name() {
        assert_eq!(SERVED_TIERS, 1);
        assert!(refuse_tier_count("t", 0).is_ok());
        assert!(refuse_tier_count("t", 1).is_ok());
        let err = refuse_tier_count("t", 2).expect_err("two tier cards");
        assert!(
            err.to_string()
                .contains("the host tier serves one tier card; N >= 2 lands"),
            "{err}"
        );
    }

    #[test]
    fn a_boundary_takes_kept_rows_only_for_an_open_pass() {
        let step = Some((super::KeptRows::prefix(1), PassKind::Step));
        assert_eq!(pass_kept(true, step).ok(), Some(step));
        assert_eq!(pass_kept(false, None).ok(), Some(None));
        let open_unkept = pass_kept(true, None).expect_err("an open pass with no kept count");
        assert!(
            open_unkept.to_string().contains("kept rows"),
            "{open_unkept}"
        );
        let kept_closed = pass_kept(false, step).expect_err("rows kept with no pass open");
        assert!(
            kept_closed.to_string().contains("no pass open"),
            "{kept_closed}"
        );
    }
}
