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
//! A tier may hang a second card under it ([`tier::TierCard`],
//! [`HostTier::attach_tier`]): per hybrid layer the card computes the
//! routed experts of its set in the host leg's shadow, behind its own go
//! and counter in its own page, and the stage card's wait waits for both
//! counters. The host launches the card's captured graph at the start of a
//! replay's service and, once it has served the replay's last layer, waits
//! for the card's progress under the go deadline and reads its fault copy
//! (the module comment of [`tier`]). A card that stops signalling poisons
//! the tier as a lost card, which no reset lifts.

pub mod batch;
pub mod handoff;
pub mod leg;
pub mod page;
pub mod residency;
pub mod run;
pub mod slots;
pub mod step;
pub mod tier;

pub use leg::{BatchLeg, StepLeg};

use crate::GpuError;
use crate::fault::{Fault, LAYER_NONE};
use batch::{BatchKey, BatchPort, BatchService, ServeTimes, Tier};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use model::{Tensor2, Tensor2View};
use page::{MAX_ROWS, Word};
use residency::HostResidency;
use slots::SlotMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;
use step::{Boundary, Chain, GO_DEADLINE, HandoffTarget, SERVE, StepPort, stream_idle, word};
use tier::{Pass, TierCard, TierTarget};

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
    /// Resets that lifted a refusal's poison ([`HostTier::reset`]).
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

    fn refuse_if_poisoned(&self, what: &'static str) -> Result<(), GpuError> {
        if self.poisoned {
            return Err(GpuError::state(
                what,
                "an earlier hybrid service failed and released the stream",
            ));
        }
        Ok(())
    }
}

/// The host tier: the model's host computation and the two ports it is
/// called through, with what they share.
///
/// Field order is drop order: the host set's lock (`residency`) spans pages
/// of the mappings `experts` keeps alive, so it is released first.
pub struct HostTier<H> {
    /// What the placed load did to the plan's host set, when the tier holds
    /// it ([`HostTier::keep_residency`]).
    residency: Option<HostResidency>,
    experts: H,
    /// The card's fault word once [`HostTier::watch_fault`] has given it:
    /// what a batch service that meets refused input reads to name the
    /// card's fault.
    fault: Option<Arc<DeviceBuffer<u32>>>,
    health: Health,
    /// Which experts each layer's card stack holds; the host serves the rest.
    slots: SlotMap,
    step: StepPort,
    /// The batch service's scratch and counters, and the port's sets once a
    /// caller made them ([`HostTier::prepare_batch`]).
    batch: BatchService,
    port: Option<BatchPort>,
    /// The expert tier's card, once attached ([`HostTier::attach_tier`]).
    tier: Option<TierCard>,
    /// Tier layers the host has served since load (wrapping): a tier whose
    /// progress is behind it holds the stage card at a wait.
    tier_goes: u32,
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
            health: Health::default(),
            slots,
            step: StepPort::new(boundary, layers),
            batch: BatchService::default(),
            port: None,
            tier: None,
            tier_goes: 0,
        })
    }

    /// Hang `tier` under the host tier, on `stage`, the card the chain
    /// launches on: the tier's set must be the slot map's tier rows
    /// ([`tier::TierSet::of_map`]), else it is refused by name. Load-time
    /// only.
    pub fn attach_tier(&mut self, mut tier: TierCard, stage: &crate::Gpu) -> Result<(), GpuError> {
        const WHAT: &str = "HostTier::attach_tier";
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
        if *set != tier::TierSet::of_map(&self.slots)? {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the tier {} holds another set than the slot map's tier rows",
                    tier.name()
                ),
            ));
        }
        tier.bind_stage(stage)?;
        self.tier_goes = tier.progress();
        self.tier = Some(tier);
        Ok(())
    }

    /// The expert tier's card, when one is attached.
    #[must_use]
    pub fn tier(&self) -> Option<&TierCard> {
        self.tier.as_ref()
    }

    /// The expert tier's card, for a gate that drives its stream.
    pub fn tier_mut(&mut self) -> Option<&mut TierCard> {
        self.tier.as_mut()
    }

    /// The experts of layer `layer` the tier card holds, as the slot map
    /// says; a layer the map has no row for is refused by name.
    pub fn on_tier(&self, layer: usize) -> Result<usize, GpuError> {
        self.slots.on_tier(layer)
    }

    /// Row `row`'s handoff target and its tier image, for a tier layer's
    /// handoff launch; refused without a tier.
    pub fn tier_handoff(
        &mut self,
        row: usize,
    ) -> Result<(HandoffTarget<'_>, TierTarget<'_>), GpuError> {
        let tier = self.tier.as_mut().ok_or(GpuError::state(
            "HostTier::tier_handoff",
            "an expert tier (HostTier::attach_tier)",
        ))?;
        let target = self.step.boundary.handoff_target_of(row)?;
        Ok((target, tier.target_of(row)?))
    }

    /// Row `row`'s routed rows the tier writes, for a tier layer's join;
    /// refused without a tier.
    pub fn tier_rows(&self, row: usize) -> Result<&DeviceBuffer<f32>, GpuError> {
        self.tier_ref("HostTier::tier_rows")?.rows_of(row)
    }

    /// Enqueue a tier layer's go of layer `layer`, row `row`, on the stage
    /// card's `stream` ([`TierCard`]'s go batch); refused without a tier.
    pub fn enqueue_tier_go(
        &self,
        stream: &CudaStream,
        layer: usize,
        row: usize,
    ) -> Result<(), GpuError> {
        self.tier_ref("HostTier::enqueue_tier_go")?.enqueue_go_of(
            stream,
            &self.step.boundary,
            layer,
            row,
        )
    }

    /// Enqueue a tier layer's wait of row `row` on the stage card's
    /// `stream`, for the host's counter and the tier's; refused without a
    /// tier.
    pub fn enqueue_tier_back(&self, stream: &CudaStream, row: usize) -> Result<(), GpuError> {
        self.tier_ref("HostTier::enqueue_tier_back")?
            .enqueue_back_of(stream, &self.step.boundary, row)
    }

    fn tier_ref(&self, what: &'static str) -> Result<&TierCard, GpuError> {
        self.tier.as_ref().ok_or(GpuError::state(
            what,
            "an expert tier (HostTier::attach_tier)",
        ))
    }

    /// The refusal of a prompt batch on a tier whose batch port is not
    /// built.
    fn refuse_tier_batch(&self, what: &'static str) -> Result<(), GpuError> {
        match &self.tier {
            Some(t) => Err(GpuError::protocol(
                what,
                format!(
                    "the tier's batch port is not built: tierbatch (a prompt batch would leave the experts on {} uncomputed)",
                    t.name()
                ),
            )),
            None => Ok(()),
        }
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
    /// once: a decode that never prefills a batch never holds them.
    /// Load-time or first-prompt only.
    pub fn prepare_batch(&mut self, ctx: &Arc<CudaContext>, cap: usize) -> Result<(), GpuError> {
        self.refuse_tier_batch("HostTier::prepare_batch")?;
        if self.port.is_none() {
            let h = self.step.boundary.layout.handoff();
            self.port = Some(BatchPort::new(ctx, h.hidden, h.n_used, cap)?);
        }
        Ok(())
    }

    /// The batch port, once [`HostTier::prepare_batch`] has made it.
    fn port_mut(&mut self, what: &'static str) -> Result<&mut BatchPort, GpuError> {
        self.port.as_mut().ok_or(GpuError::State {
            what,
            missing: "the batch port's sets (HostTier::prepare_batch)",
        })
    }

    /// Both batch sets free: at a group's start, once the stream holds
    /// nothing an earlier group enqueued ([`BatchPort::begin`]).
    pub fn begin_group(&mut self) -> Result<(), GpuError> {
        self.port_mut("HostTier::begin_group")?.begin();
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
        self.port_mut("HostTier::enqueue_download_pitched")?
            .download_pitched(stream, xw, ids, pitch, key)
    }

    /// Wait for the oldest download not served yet — `key`'s, else refused
    /// by name — and serve its layer's host experts for its tokens in one
    /// union call ([`HostTier::serve_batch`]'s service), the sums into the
    /// set the upload sends. Returns the host time outside the union call.
    pub fn serve_key(&mut self, key: BatchKey) -> Result<ServeTimes, GpuError> {
        self.refuse_tier_batch("HostTier::serve_key")?;
        let h = self.step.boundary.layout.handoff();
        let Some(port) = self.port.as_mut() else {
            return Err(GpuError::State {
                what: "HostTier::serve_key",
                missing: "the batch port's sets (HostTier::prepare_batch)",
            });
        };
        let (batch, experts, health) = (&mut self.batch, &mut self.experts, &mut self.health);
        let (slots, fault) = (&self.slots, self.fault.as_ref());
        port.serve(key, |layer, x, ids, w, out| {
            let t = Tier {
                experts,
                health,
                slots,
                fault,
                hidden: h.hidden,
                n_used: h.n_used,
            };
            batch.serve_guarded(t, layer, x, ids, w, &[], out)
        })
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
    /// its nodes and addresses; its next replay meets these words.
    ///
    /// A tier poisoned any other way — a failure of its own, a panic in the
    /// host experts — stays poisoned and the call fails by name: only a
    /// reload clears it. On a tier that is not poisoned the call changes
    /// nothing, and fails by name when the stream has drained and the words
    /// are not at rest; a stream still running may hold a signal its wait has
    /// yet to take back.
    pub fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        const WHAT: &str = "Hybrid::reset";
        if !self.health.poisoned {
            if !stream_idle(stream)? {
                return Ok(());
            }
            let w = self.words();
            if let Some(t) = self.tier.as_mut() {
                t.gpu().stream().synchronize()?;
                if !t.at_rest() {
                    return Err(GpuError::protocol(
                        WHAT,
                        format!(
                            "the stream has drained and the expert tier is not at rest (progress \
                             {} of {} asked): a go the tier never took, or a signal no wait took \
                             back",
                            t.progress(),
                            t.issued()
                        ),
                    ));
                }
                t.clear_fault()?;
            }
            if !w.at_rest() {
                return Err(GpuError::protocol(
                    WHAT,
                    format!(
                        "the stream has drained and the host tier is not at rest ({w}): a go the \
                         host never served, or a signal no wait took back"
                    ),
                ));
            }
            return Ok(());
        }
        let cause = match &self.health.poison {
            Some(Poison::Refused(_)) => None,
            Some(p) => Some(p.to_string()),
            None => Some("a failure with no cause recorded".to_string()),
        };
        if let Some(cause) = cause {
            return Err(GpuError::protocol(
                WHAT,
                format!(
                    "the host tier is poisoned ({cause}); a reset lifts only refused input — \
                     reload the model"
                ),
            ));
        }
        stream.synchronize()?;
        let boundary = &self.step.boundary;
        let generation = word(&boundary.page, Word::Gen).load(Ordering::Acquire);
        let mut seq = [0u32; 1];
        boundary.seq_word().copy_to_host(stream, &mut seq)?;
        if seq[0] != generation {
            return Err(GpuError::protocol(
                WHAT,
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
        if let Some(t) = self.tier.as_mut() {
            t.reset()?;
            self.tier_goes = t.progress();
        }
        let h = &mut self.health;
        h.poisoned = false;
        h.refusal = None;
        h.failing = None;
        h.stats.resets += 1;
        Ok(())
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
        if self.step.is_capturing()
            && let Some(t) = self.tier.as_mut()
        {
            t.forget_order(self.step.chain());
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
        let tiered = self.tier.is_some() && self.on_tier(layer)? > 0;
        if tiered {
            self.health.refuse_if_poisoned(SERVE)?;
            self.feed_tier(layer, row)?;
        }
        self.serve(layer, row, false, chain)?;
        if tiered {
            self.settle_tier(layer)
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
    /// With an expert tier the tier's graph of `chain` is launched first when
    /// one is held for the chain's go order; with none, the host enqueues
    /// each tier layer on the tier's stream as it serves it, and captures the
    /// graph once the pass has settled ([`tier::Pass`]). Once the last layer
    /// is served the host waits for the tier to catch up and reads its fault
    /// copy ([`HostTier::settle_tier`]).
    pub fn serve_captured_of(&mut self, chain: Chain) -> Result<(), GpuError> {
        let mut list = Vec::new();
        let mut pass = Pass::Graph;
        if self.tier.is_some() {
            self.health.refuse_if_poisoned(SERVE)?;
            for (l, r) in (0..).map_while(|i| self.step.captured(chain, i)) {
                if self.slots.on_tier(l)? > 0 {
                    list.push((l, r));
                }
            }
            let r = self
                .tier
                .as_mut()
                .map_or(Ok(Pass::Graph), |t| t.replay(chain, &list));
            pass = match r {
                Ok(p) => p,
                Err(e) => {
                    self.health
                        .set_poison(SERVE, list.first().map_or(0, |p| p.0), Some(&e), None);
                    self.release_all();
                    return Err(e);
                }
            };
        }
        let mut i = 0;
        while let Some((layer, row)) = self.step.captured(chain, i) {
            if pass == Pass::Feed && self.on_tier(layer)? > 0 {
                self.feed_tier(layer, row)?;
            }
            self.serve(layer, row, i == 0, chain)?;
            i += 1;
        }
        let Some(&(last, _)) = list.last() else {
            return Ok(());
        };
        self.settle_tier(last)?;
        if pass == Pass::Feed {
            let r = self
                .tier
                .as_mut()
                .map_or(Ok(()), |t| t.capture(chain, &list));
            if let Err(e) = r {
                self.health.set_poison(SERVE, last, Some(&e), None);
                self.release_all();
                return Err(e);
            }
        }
        Ok(())
    }

    /// Enqueue tier layer `layer` of row `row` on the tier's stream now, for
    /// an eager chain or a fed pass; a failure poisons the tier and releases
    /// both cards' waits.
    fn feed_tier(&mut self, layer: usize, row: usize) -> Result<(), GpuError> {
        let r = self
            .tier
            .as_mut()
            .map_or(Ok(()), |t| t.enqueue_eager(layer, row));
        if let Err(e) = r {
            self.health.set_poison(SERVE, layer, Some(&e), None);
            self.release_all();
            return Err(e);
        }
        Ok(())
    }

    /// Wait under the go deadline for the expert tier to serve every layer
    /// asked of it — `layer` the last the host served — then read its fault
    /// copy: a raised copy is the step's fault, merged with the stage card's
    /// word once that card's stream has drained (the first layer wins). A
    /// tier that does not catch up in time is a lost card: the tier is
    /// poisoned, both cards' waits released, and the error names the card.
    fn settle_tier(&mut self, layer: usize) -> Result<(), GpuError> {
        let Some(t) = self.tier.as_mut() else {
            return Ok(());
        };
        if !t.wait_caught_up(Instant::now() + GO_DEADLINE) {
            return Err(self.lose_tier(layer));
        }
        match t.merged_fault()? {
            Some(f) => Err(GpuError::fault(SERVE, f)),
            None => Ok(()),
        }
    }

    /// The expert tier is lost at layer `layer`: poison the tier as a lost
    /// card, release both cards' waits, and name the card.
    fn lose_tier(&mut self, layer: usize) -> GpuError {
        let (name, detail) = match self.tier.as_ref() {
            Some(t) => (t.name().to_string(), t.lost_detail(GO_DEADLINE)),
            None => (String::new(), "no tier".to_string()),
        };
        self.health.set_lost(SERVE, &name, layer, &detail);
        self.release_all();
        GpuError::protocol(SERVE, format!("layer {layer}: {detail}"))
    }

    /// Release every wait still pending on both cards.
    fn release_all(&self) {
        self.step.boundary.release();
        if let Some(t) = self.tier.as_ref() {
            t.release();
        }
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
    /// a panic inside the host experts, poisons it.
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
    /// or a panic inside the host experts — releases every pending wait
    /// first, so the stream drains instead of hanging a later synchronize.
    fn serve(
        &mut self,
        layer: usize,
        row: usize,
        opens_replay: bool,
        chain: Chain,
    ) -> Result<(), GpuError> {
        self.health.refuse_if_poisoned(SERVE)?;
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
                if self.tier.is_some() && self.slots.on_tier(layer)? > 0 {
                    self.tier_goes = self.tier_goes.wrapping_add(1);
                    let hits = self.step.tier_slots;
                    if let Some(t) = self.tier.as_mut() {
                        t.hit(layer, hits);
                    }
                }
                Ok(())
            }
            Ok(Err(e)) => {
                // The stage card waits at a tier layer the host has served
                // while the tier has not: past the grace, the tier is the
                // one behind.
                let want = self.tier_goes;
                if self
                    .tier
                    .as_ref()
                    .is_some_and(|t| !t.wait_for(want, Instant::now() + tier::TIER_GRACE))
                {
                    return Err(self.lose_tier(layer));
                }
                self.health.set_poison(SERVE, layer, Some(&e), None);
                self.release_all();
                Err(e)
            }
            Err(p) => {
                self.health.set_poison(SERVE, layer, None, Some(&*p));
                self.release_all();
                std::panic::resume_unwind(p)
            }
        }
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
