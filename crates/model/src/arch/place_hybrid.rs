//! A hybrid file planned onto a machine: the one path from a family's headers to a plan that keeps
//! its invariants, for every family whose body mixes with GQA layers (a plane, or a ring for a window
//! layer, and an index key row a position for a layer that selects blocks) and delta-rule layers (a
//! state and a conv ring), routes every layer's experts and puts the rest of them on the host. A
//! family reads its headers into a [`ModelSpec`] and its tensors' roles ([`PlanInputs::new`]); what
//! the layers hold is read off the description ([`KvLayout::of`]), the program off its architecture
//! ([`program_of`]), and the placement runs the expert rule on the layers whose stacks the card
//! experts read ([`placement::plan_routed`]). What stays in a family is what an architecture fact
//! forces: the keys it reads, the layers it refuses, the engine's name in the refusals that name it.
//!
//! The plan refuses by name a machine that hangs an expert tier card where the family's body serves
//! none ([`PlanInputs::tier_reserve`]), a context of no position and, unless the loader explicitly
//! serves the common NVMe tier ([`PlanInputs::with_nvme_tier`]), a host room that pages any expert
//! ([`PlaceError::NoNvmeTier`]).
//!
//! A body that makes buffers beside the plan's terms on its card (the prompt batch's rows) states
//! their bytes in [`PlanInputs::prompt_reserve`]: the expert rule spreads within the card's budget
//! less them ([`placement::plan_routed_reserving`]) and a card that cannot hold them beside its
//! other terms is refused by name ([`PlaceError::PromptOver`]). The family that makes the buffers
//! owns the formula of their bytes; the body holds its allocation to the same figure.

use std::num::NonZeroUsize;
use std::ops::Range;

use bloomery_placement::slots::{SeqTerms, Stores};
use models::{DeltaRule, Gqa, Mixer, ModelSpec, RopeMode, Selector};
use runtime::stores::{self, Rows};

use crate::arch::coverage;
use crate::placement::accepts::{program_of, routed_rule};
use crate::placement::ctx_ask::{PlanRefusal, Short, Side};
use crate::placement::workstation::{self, HostNeed, HostRead};
use crate::placement::{
    self, Card, CardTotals, KvBytes, Machine, ModelTensors, PlacementError, Plan, PlanLevers,
    Unimplemented, Violation, checked_beside, joined, kernels,
};

// The prompt batch a window ring keeps room for is the batch the card walks the prompt in.
const _: () = assert!(stores::PROMPT_BATCH == crate::moe::UNION_MAX_COLS as u64);

/// The columns a GEMM tile holds (the route's tile table) and the values a 32-value form's step
/// covers, as the prompt batches' wide forms and routes are sized by them; a body holds each equal
/// to the kernels' by assertion.
pub const GEMM_TILE_COLS: usize = 64;
/// See [`GEMM_TILE_COLS`].
pub const GEMM32_STEP: usize = 64;

/// Card bytes of the q8 blocks of 32 of `cols` columns of `k` values (Q8_0 and Q5_0 sites' input,
/// `GemmAct32`, [`gemm_act32_bytes`](crate::placement::kernels::gemm_act32_bytes)).
const fn act32_bytes(cols: usize, k: usize) -> usize {
    crate::placement::kernels::gemm_act32_bytes(cols as u64, k as u64) as usize
}

/// The most card bytes the forms a GEMM reads take at `k` values a column, `cols` columns: both,
/// where the width allows them. The load makes the ones its sites' types read.
#[must_use]
pub const fn wide_bytes(cols: usize, k: usize) -> usize {
    let q128 = if k.is_multiple_of(256) {
        kernels::q8_act_bytes(cols as u64, k as u64) as usize
    } else {
        0
    };
    let q32 = if k.is_multiple_of(32) {
        act32_bytes(cols, k)
    } else {
        0
    };
    q128 + q32
}

/// Card bytes of the GEMM route's table of `cols` slots over one expert: the slot list, the tiles
/// (a [`GEMM_TILE_COLS`] column tile each, and the expert's), the two counts and the all-zero ids.
#[must_use]
pub const fn route_bytes(cols: usize) -> usize {
    match kernels::ffn::dense_route_bytes(cols as u64) {
        Some(bytes) => bytes as usize,
        None => panic!("route_bytes: route table bytes overflow u64"),
    }
}

/// The forms a ring pass's route GEMMs read their rows in, by the stacks' file types: the gate's and
/// up's (one type, so one form), then the down's — `true` where the kernel table's GEMM entry for
/// the type reads q8 blocks of 32 (`kernels::gemm_reads_blocks32`, `GemmAct32`), `false` for the
/// grouped GEMM's q8_1 blocks of 128 (`GemmAct`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingForms {
    pub gate_up32: bool,
    pub down32: bool,
}

impl RingForms {
    /// Both reads in the 32-value form.
    pub const BLOCKS32: RingForms = RingForms {
        gate_up32: true,
        down32: true,
    };

    /// The forms of a gate and up of type `gate_up` and a down of type `down`; `None` where the
    /// kernel table names no route GEMM for one of them.
    #[must_use]
    pub const fn of(gate_up: gguf::GgmlType, down: gguf::GgmlType) -> Option<RingForms> {
        match (kernels::gemm(gate_up), kernels::gemm(down)) {
            (Some(g), Some(d)) => Some(RingForms {
                gate_up32: kernels::gemm_reads_blocks32(g),
                down32: kernels::gemm_reads_blocks32(d),
            }),
            _ => None,
        }
    }
}

/// The forms a plan prices a ring pass in over `model`'s routed layers (`stacks(l)`, the gate, up
/// and down of layer `l`): a role is the grouped GEMM's form where any layer's type reads it, the
/// larger of the two, else the 32-value form; a layer whose types no route GEMM reads, or that
/// holds no routed stacks, adds nothing.
#[must_use]
pub fn ring_forms_planned(
    model: &ModelTensors,
    stacks: impl Fn(usize) -> [String; 3],
) -> RingForms {
    let ty = |name: &String| model.tensors.iter().find(|t| &t.name == name).map(|t| t.ty);
    let mut forms = RingForms::BLOCKS32;
    for l in 0..model.layers {
        let [gate, _, down] = stacks(l);
        let Some(f) = ty(&gate)
            .zip(ty(&down))
            .and_then(|(g, d)| RingForms::of(g, d))
        else {
            continue;
        };
        forms.gate_up32 &= f.gate_up32;
        forms.down32 &= f.down32;
    }
    forms
}

/// Card bytes of `cols` columns of `k` values in one route GEMM's read form: the 32-value
/// form's where `blocks32`, else the grouped GEMM's q8_1 blocks of 128.
const fn ring_act_bytes(cols: usize, k: usize, blocks32: bool) -> usize {
    if blocks32 {
        act32_bytes(cols, k)
    } else {
        kernels::q8_act_bytes(cols as u64, k as u64) as usize
    }
}

/// Card bytes of a prompt batch's ring pass over a half of `half` slots of the expert stream's ring
/// with both reads in the 32-value form ([`ring_pass_bytes_in`] at [`RingForms::BLOCKS32`]).
#[must_use]
pub const fn ring_pass_bytes(
    rows: usize,
    n_used: usize,
    n: usize,
    ff: usize,
    half: usize,
) -> usize {
    ring_pass_bytes_in(rows, n_used, n, ff, half, RingForms::BLOCKS32)
}

/// Card bytes of a prompt batch's ring pass over a half of `half` slots of the expert stream's ring
/// (the ring GEMM buffers, made when the stream starts): the batch's `rows` normed rows of `n`
/// values in the gate's and up's form, every slot's gate and up rows (`rows · n_used` slots of
/// `ff` f32 values each), their SwiGLU in the down's form and the route table of those slots over
/// the half, the forms `forms`. The one formula of those bytes: the body holds its made bytes to
/// it.
#[must_use]
pub const fn ring_pass_bytes_in(
    rows: usize,
    n_used: usize,
    n: usize,
    ff: usize,
    half: usize,
    forms: RingForms,
) -> usize {
    let slots = rows * n_used;
    let routed = if half < slots { half } else { slots };
    let zeros = if half == 1 { slots } else { 0 };
    let route = 4 * (slots + 2 * (slots / GEMM_TILE_COLS + routed) + 2 + zeros);
    ring_act_bytes(rows, n, forms.gate_up32)
        + 2 * 4 * slots * ff
        + ring_act_bytes(slots, ff, forms.down32)
        + route
}

/// The card bytes a prompt call's expert stream takes beside the plan's terms
/// (`BLOOMERY_XSTREAM=split`), the rule every family's streaming plan keeps room by: the ring's two
/// halves of `half` slots of [`stream_slot_bytes`], and `beside`, what the body makes for the
/// ring's pass. Refused as [`stream_slot_bytes`] refuses.
pub fn stream_ring_bytes(
    model: &ModelTensors,
    stacks: impl Fn(usize) -> [String; 3],
    (n_expert, half): (u64, u64),
    beside: u64,
) -> Result<u64, PlacementError> {
    Ok(2 * half * stream_slot_bytes(model, stacks, n_expert)? + beside)
}

/// A slot of the expert stream: the widest layer's expert at its routed stacks' file bytes per
/// expert (`stacks(l)`, the gate, up and down of layer `l` of `model`'s `n_expert`-expert stacks; a
/// layer with none of the three streams nothing). The ring's card slots and the lane's pinned host
/// staging slots are of these bytes. Refused by name: a layer that holds some of its three stacks
/// and not all, and a stack that is not one of `n_expert` experts.
pub fn stream_slot_bytes(
    model: &ModelTensors,
    stacks: impl Fn(usize) -> [String; 3],
    n_expert: u64,
) -> Result<u64, PlacementError> {
    let mut slot = 0;
    for l in 0..model.layers {
        let names = stacks(l);
        let found: Vec<_> = names
            .iter()
            .map(|name| model.tensors.iter().find(|t| &t.name == name))
            .collect();
        if found.iter().all(Option::is_none) {
            continue;
        }
        let mut expert = 0;
        for (name, t) in names.iter().zip(found) {
            let t = t.ok_or_else(|| PlacementError::Tensor {
                name: name.clone(),
                detail: "is not in the file: the stream's ring holds a layer's three routed stacks"
                    .to_string(),
            })?;
            expert += placement::per_expert(t, n_expert)?.1;
        }
        slot = slot.max(expert);
    }
    Ok(slot)
}

/// The card bytes a tiered load makes beside the prompt batch's reserve: per routed slot of the
/// prompt batch's `rows` rows (`n_used` a row), the tier place, the copy of it the join reads and
/// the copy of the slot's route weight, three four-byte words.
#[must_use]
pub const fn tier_places_bytes(rows: usize, n_used: usize) -> u64 {
    match kernels::ffn::tier_places_bytes(rows as u64, n_used as u64) {
        Some(bytes) => bytes,
        None => panic!("tier_places_bytes: slot bytes overflow u64"),
    }
}

/// What a plan of a hybrid file is made from, read from its headers. `H` is the family's own
/// hyperparameters, carried for its body and never read here.
#[derive(Clone, Debug)]
pub struct PlanInputs<H> {
    /// The file's hyperparameters.
    pub hp: H,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// The typed model description the coverage check reads.
    pub spec: ModelSpec,
    /// Each trunk layer's stores.
    pub kv: KvLayout,
    /// The host's available bytes when the inputs were read, with the reading that decided
    /// ([`workstation::host_available_read`]): the routed experts the room cannot hold follow it to
    /// the NVMe tier ([`placement::expert_nvme_tier`]). A body without that loader contract refuses
    /// a plan that pages experts ([`PlaceError::NoNvmeTier`]). A gate that plans an arm by a room of
    /// its own sets it, read [`HostRead::Given`].
    ///
    /// The effective room is [`Plan::plan_at`]'s: the reading's plan room,
    /// capped by the machine's host room and unified pool.
    pub room: (u64, HostRead),
    /// The arena's bytes `BLOOMERY_NVTIER_BYTES` gave the split dial
    /// ([`workstation::nvtier_arena_given`]), `None` unset; the split takes it as its `given`. A
    /// gate that plans an arm by an arena of its own sets it.
    pub arena: Option<u64>,
    /// The card bytes the load makes beside the plan's terms, its prompt batch's buffers: every
    /// card keeps them out of the expert budget ([`placement::plan_routed_reserving`]), and the
    /// body refuses a load whose made bytes pass them. 0 for a family whose body makes none.
    pub prompt_reserve: u64,
    /// `None` for a family whose body hangs no expert tier card: a machine that names one is
    /// refused by name ([`PlaceError::Tier`]). `Some(bytes)` for a body that serves one: the card
    /// keeps `bytes` more than [`PlanInputs::prompt_reserve`] on a machine that names a tier card,
    /// the buffers only a tiered load makes.
    pub tier_reserve: Option<u64>,
    /// A slot of the expert stream's ring for a load whose prompt calls stream their host experts
    /// ([`PlanInputs::streaming`], [`stream_slot_bytes`]); 0 for a load that does not stream. The
    /// stream's lane pins its host staging in slots of these bytes for the load's life, which the
    /// seat reserves on the plan's host (`workstation::XSTREAM_STAGING_RESERVE`).
    pub stream_slot: u64,
    /// Whether the loader serves the common host-run NVMe arena. Opt-in only: a plan cannot
    /// promise paging merely because the placement can split the expert stacks.
    nvme_tier: bool,
    /// The engine the refusals that name one name.
    engine: &'static str,
}

/// Why a plan was refused.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// The placement could not be built.
    #[error("{0}")]
    Placement(#[from] PlacementError),
    /// A machine with an expert tier card: this program hangs no expert tier on its card experts.
    #[error(
        "the machine has the expert tier card {card} (tier {tier}); the program this plan is \
         for hangs no expert tier"
    )]
    Tier {
        /// The tier card's name.
        card: String,
        /// Its index among the machine's tiers.
        tier: usize,
    },
    /// The plan was built and breaks these invariants, every one of them.
    #[error("the plan breaks its invariants: {}", joined(.0))]
    Broken(Vec<Violation>),
    /// A context of no position: no plane, nothing to serve.
    #[error("ctx_max 0: a plan serves at least one position")]
    CtxZero,
    /// A card whose plan leaves less beside it than the prompt batch's buffers
    /// ([`PlanInputs::prompt_reserve`]) take: the dense granules, the cache, context, scratch,
    /// reserves and margin already pass what the card holds less them, so no expert placement
    /// makes room.
    #[error(
        "card {card}: the prompt batch's buffers take {reserve} B, and the plan leaves {left} B \
         beside its dense granules, cache, context, scratch, reserves and margin ({short} B short)"
    )]
    PromptOver {
        /// The card's name.
        card: String,
        /// The bytes the load makes beside the plan's terms.
        reserve: u64,
        /// The bytes the card has left beside the plan's terms.
        left: u64,
        /// `reserve - left`.
        short: u64,
    },
    /// The host room cannot hold the host experts, so the plan pages some to the NVMe expert tier.
    /// The body builds its host run without that tier (`HostRun::build`, not the common `host_run`
    /// that attaches it), so the paged experts would load into nothing that serves them.
    #[error(
        "the host need {need} B passes the room, so the plan pages {paged} B of routed experts \
         to the NVMe expert tier{room}, and the {engine} engine serves no NVMe expert tier; free \
         host memory (the room is the host's available bytes, or BLOOMERY_HOST_ROOM where set)"
    )]
    NoNvmeTier {
        /// The routed-expert bytes the plan leaves on the NVMe tier.
        paged: u64,
        /// The host need before any expert is paged ([`HostNeed::bytes`]).
        need: u64,
        /// The room the plan was made at, as the plan names it
        /// ([`placement::HostTotals::room_clause`]).
        room: String,
        /// The engine that serves no such tier.
        engine: &'static str,
    },
}

impl PlanRefusal for PlaceError {
    fn placement(&self) -> Option<&PlacementError> {
        match self {
            Self::Placement(e) => Some(e),
            _ => None,
        }
    }

    fn violations(&self) -> Option<&[Violation]> {
        match self {
            Self::Broken(v) => Some(v),
            _ => None,
        }
    }

    /// The card's reading, and the host's: a longer context leaves more routed experts to the
    /// host, and the body serves no NVMe tier past it ([`PlaceError::NoNvmeTier`]), so a smaller
    /// context is what that refusal asks for. The host held its need less the bytes it paged.
    fn short(&self) -> Option<Short> {
        match self {
            Self::NoNvmeTier { paged, need, .. } => Some(Short {
                on: Side::Host,
                need: *need,
                room: need.saturating_sub(*paged),
            }),
            Self::PromptOver { reserve, left, .. } => Some(Short {
                on: Side::Card,
                need: *reserve,
                room: *left,
            }),
            _ => self.card_short(),
        }
    }
}

impl<H> PlanInputs<H> {
    /// The inputs of a file whose `hp`, roles `model` and description `spec` a family read: what
    /// each trunk layer holds ([`KvLayout::of`]), then the host's room
    /// ([`workstation::host_available_read`]) and the arena's given bytes, a room that cannot be
    /// read refused by name. `engine` is the name the refusals that name one use.
    pub fn new(
        engine: &'static str,
        hp: H,
        model: ModelTensors,
        spec: ModelSpec,
    ) -> Result<PlanInputs<H>, PlacementError> {
        let kv = KvLayout::of(spec.layers.iter().map(|l| &l.mixer))?;
        let room = workstation::host_available_read().map_err(PlacementError::HostRoom)?;
        let arena = workstation::nvtier_arena_given().map_err(PlacementError::HostRoom)?;
        Ok(PlanInputs {
            hp,
            model,
            spec,
            kv,
            room,
            arena,
            prompt_reserve: 0,
            tier_reserve: None,
            stream_slot: 0,
            nvme_tier: false,
            engine,
        })
    }

    /// `self` with a load that makes `bytes` of card buffers beside the plan's terms
    /// ([`PlanInputs::prompt_reserve`]).
    #[must_use]
    pub fn with_prompt_reserve(self, bytes: u64) -> PlanInputs<H> {
        PlanInputs {
            prompt_reserve: bytes,
            ..self
        }
    }

    /// `self` for a load whose prompt calls stream their host experts (`BLOOMERY_XSTREAM=split`):
    /// the prompt reserve grows by the stream's `ring` bytes ([`stream_ring_bytes`]), and a slot of
    /// the ring is `slot` bytes ([`PlanInputs::stream_slot`]). A load that does not stream plans
    /// without them; its ring, if one is set later, takes what the card's margin leaves free past
    /// the stream's keep-free.
    #[must_use]
    pub fn streaming(self, ring: u64, slot: u64) -> PlanInputs<H> {
        let reserve = self.prompt_reserve.saturating_add(ring);
        PlanInputs {
            stream_slot: slot,
            ..self.with_prompt_reserve(reserve)
        }
    }

    /// `self` for a body that serves an expert tier card, which makes `bytes` of card buffers
    /// beside [`PlanInputs::prompt_reserve`] on a load with one ([`PlanInputs::tier_reserve`]).
    #[must_use]
    pub fn with_tier_reserve(self, bytes: u64) -> PlanInputs<H> {
        PlanInputs {
            tier_reserve: Some(bytes),
            ..self
        }
    }

    /// `self` for a loader that serves the common host-run NVMe arena, with the same residency
    /// identity and split as the plan. Existing loaders remain non-paging unless they opt in.
    #[must_use]
    pub fn with_nvme_tier(self) -> PlanInputs<H> {
        PlanInputs {
            nvme_tier: true,
            ..self
        }
    }

    /// `self` with each delta-rule layer's state `lanes` lanes ([`KvLayout::with_lanes`]): the
    /// stores of a load that verifies `lanes` rows a pass.
    #[must_use]
    pub fn with_lanes(self, lanes: NonZeroUsize) -> PlanInputs<H> {
        PlanInputs {
            kv: self.kv.with_lanes(lanes),
            ..self
        }
    }

    /// The card bytes the load of `machine` makes beside the plan's terms: the prompt batch's, and
    /// a tiered load's more. A machine that names an expert tier card where the body serves none
    /// is refused by name ([`PlaceError::Tier`]); a reserve-only member is no tier card.
    pub fn prompt_batch_bytes(&self, machine: &Machine) -> Result<u64, PlaceError> {
        match (machine.expert_tiers().next(), self.tier_reserve) {
            (Some(card), None) => Err(PlaceError::Tier {
                card: card.name.clone(),
                tier: 0,
            }),
            (Some(_), Some(more)) => Ok(self.prompt_reserve.saturating_add(more)),
            (None, _) => Ok(self.prompt_reserve),
        }
    }

    /// `self`, or — for a file with a feature the engine does not run —
    /// [`PlacementError::Unimplemented`] with every one listed.
    pub fn runnable(self) -> Result<PlanInputs<H>, PlacementError> {
        let missing = self.unimplemented();
        if missing.is_empty() {
            Ok(self)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// Every feature of the file the engine does not run, as [`coverage::check`] lists them. Empty
    /// for a file the program runs.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }

    /// The placement of the file on `machine` at `ctx_max` positions under the placement's
    /// `levers`: the expert rule on the layers whose routed stacks the program's rule carries
    /// ([`routed_rule`]), each such
    /// layer's id prefix on the card as far as its budget holds them, the next ids on the expert
    /// tier card of a machine that names one, the rest on the host; refused by name a context of no
    /// position, a machine that hangs an expert tier card where the body serves none
    /// ([`PlanInputs::tier_reserve`]), a context whose stores do not fit the card and a host room
    /// under the host experts' need
    /// ([`PlanInputs::plan_with_slots`]).
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_with_slots(machine, ctx_max, levers, NonZeroUsize::MIN)
    }

    /// The placement of a load that serves `slots` resident sequences of `ctx_max` positions each:
    /// every layer's stores counted `slots` times on the card ([`SeqTerms::slots_of`] over
    /// [`PlanInputs::seq_terms`]), refused as [`PlanInputs::plan`] refuses. A sequence holds nothing
    /// beside its stores, so the card's kv class is `slots` sequences' stores.
    ///
    /// The host experts are held to the room the inputs name ([`PlanInputs::room`]): a room under
    /// the host need runs the common NVMe tier split ([`placement::expert_nvme_tier`], which refuses
    /// a room under the tier's floor by name). A plan paging any routed expert is refused
    /// ([`PlaceError::NoNvmeTier`]) unless the loader serves it ([`PlanInputs::with_nvme_tier`]).
    pub fn plan_with_slots<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        slots: NonZeroUsize,
    ) -> Result<Plan<'a>, PlaceError> {
        if ctx_max == 0 {
            return Err(PlaceError::CtxZero);
        }
        let reserve = self.prompt_batch_bytes(machine)?;
        let program = program_of(self.spec.arch)
            .ok_or_else(|| PlacementError::Unimplemented(self.unimplemented()))?;
        let kv = self.seq_terms().slots_of(slots.get() as u64);
        let mut plan = placement::plan_routed_reserving(
            &self.model,
            machine,
            ctx_max,
            &kv,
            levers,
            routed_rule(program),
            reserve,
        )?;
        let need = HostNeed::of(&plan, 0).bytes();
        let room = plan.plan_at(&self.room);
        placement::expert_nvme_tier(&mut plan, room, self.arena)?;
        if plan.host.nvme_expert_bytes > 0 && !self.nvme_tier {
            return Err(PlaceError::NoNvmeTier {
                paged: plan.host.nvme_expert_bytes,
                need,
                room: plan.host.room_clause(),
                engine: self.engine,
            });
        }
        let plan = checked_beside(plan, reserve).map_err(PlaceError::Broken)?;
        for (card, totals) in machine.cards.iter().zip(&plan.cards) {
            let limit = plan.usable_bytes(card).saturating_sub(card.margin_bytes);
            refuse_reserve(
                card,
                card_terms(totals),
                limit,
                reserve,
                |card, reserve, left, short| PlaceError::PromptOver {
                    card,
                    reserve,
                    left,
                    short,
                },
            )?;
        }
        Ok(plan)
    }

    /// What one resident sequence of a load of the file holds on its card ([`SeqTerms`]).
    #[must_use]
    pub fn seq_terms(&self) -> SeqTerms<'_> {
        self.kv.seq_terms()
    }
}

/// The prompt batch's `reserve` beside `total` bytes of a plan's terms on `card`, `limit` its
/// usable bytes less its margin: refused by name when the terms fit alone and the reserve does not
/// fit beside them (terms past the limit are the plan's own violation). `over` is the family's
/// own refusal of those bytes: the rule is one, the variant each family names is its own (a
/// family's prompt batch's front and another's prompt batch's reserve are named apart).
pub(crate) fn refuse_reserve<E>(
    card: &Card,
    total: u64,
    limit: u64,
    reserve: u64,
    over: impl FnOnce(String, u64, u64, u64) -> E,
) -> Result<(), E> {
    match limit.checked_sub(total) {
        Some(left) if reserve > left => Err(over(card.name.clone(), reserve, left, reserve - left)),
        _ => Ok(()),
    }
}

/// A plan's terms on a card: its granules, cache, scratch, context and reserves.
pub(crate) fn card_terms(t: &CardTotals) -> u64 {
    t.dense_bytes
        + t.expert_bytes
        + t.rounding_bytes
        + t.kv_bytes
        + t.scratch_bytes
        + t.context_bytes
        + t.reserve_bytes
}

/// What one trunk layer holds per sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hold {
    /// A GQA layer: K and V of `row` bytes a position, over its `window` positions when it has one,
    /// and `index` bytes of index key a position over every position (0: the layer selects nothing).
    /// A latent layer: its cached row of `row` bytes a position, no index, no window.
    Attention {
        row: u64,
        index: u64,
        window: Option<u64>,
    },
    /// A delta-rule layer: its first state lane and conv ring (`fixed` bytes), and the lanes'
    /// stamps and the lanes past the first over `v_heads` heads of `d × d` f32, whatever the
    /// context.
    State {
        fixed: u64,
        v_heads: usize,
        d: usize,
    },
    /// A layer with no token mixer.
    Nothing,
}

/// One file's per-layer stores, read off its description. A full GQA layer holds a plane of every
/// position. A window layer holds that plane under [`Rows::Plane`], and
/// [`runtime::stores::ring_rows`] rows under [`Rows::Ring`]; each row is the layer's key heads' key
/// and value rows in f16. A GQA layer that selects blocks ([`Selector::BlockMax`]) attends every
/// position it holds (it has no window) and keeps, beside its planes, one f16 index key of the
/// selector's `d` values for every position. A delta-rule layer holds a state of `v_heads` heads of
/// `d × d` f32 in each of its stamped lanes, and the conv ring (`runtime::stores::recurrent_bytes`,
/// `delta_lane_bytes`): one lane, or the lanes of a load that verifies ([`KvLayout::with_lanes`]).
#[derive(Clone, Debug)]
pub struct KvLayout {
    /// Per trunk layer.
    holds: Vec<Hold>,
    /// Which row count a window layer uses.
    rows: Rows,
    /// The rope rows the body copies to its card, in bytes a position: one table of `dims` f32
    /// values a position for each distinct rope (base and dims) the rotating layers turn by.
    rope_row: u64,
    /// The block selector's sink row, in bytes a position: the index head kernel appends a value
    /// row beside each index key, to a plane of the widest selector's key nothing reads and every
    /// selecting layer shares; 0 for a trunk that selects nothing.
    sink_row: u64,
    /// The lanes of a delta-rule layer's state.
    lanes: NonZeroUsize,
}

impl KvLayout {
    /// The layout of the trunk layers `mixers` name, window layers on the ring; a mixer whose store
    /// this owner does not size (a Mamba-2 layer, a latent layer that compresses, selects or windows,
    /// a GQA layer with a selector other than the block selector or with a window beside it) is refused
    /// by name with its layer.
    pub fn of<'a>(mixers: impl IntoIterator<Item = &'a Mixer>) -> Result<KvLayout, PlacementError> {
        let mut holds = Vec::new();
        let mut ropes: Vec<(u32, u32)> = Vec::new();
        let mut sink_row = 0;
        let mut note = |rope: &models::Rope| {
            if rope.mode != RopeMode::None {
                let rope = (rope.base.to_bits(), rope.dims);
                if !ropes.contains(&rope) {
                    ropes.push(rope);
                }
            }
        };
        for (layer, mixer) in mixers.into_iter().enumerate() {
            holds.push(match mixer {
                Mixer::Gqa(g) => {
                    let index = match index_row(g) {
                        Ok(index) => index,
                        Err(what) => return Err(unsized_store(layer, what)),
                    };
                    note(&g.rope);
                    sink_row = sink_row.max(index);
                    if let Some(Selector::BlockMax { rope, .. }) = &g.select {
                        note(rope);
                    }
                    Hold::Attention {
                        row: u64::from(g.kv_heads)
                            * (u64::from(g.head_dim) + u64::from(g.value_dim))
                            * stores::F16_BYTES,
                        index,
                        window: g.window.map(u64::from),
                    }
                }
                // A latent layer that compresses, selects or windows nothing holds a plane of one
                // row a position, the latent and the key values beside it, in f16.
                Mixer::Latent(a)
                    if a.compress.is_none() && a.select.is_none() && a.window.is_none() =>
                {
                    Hold::Attention {
                        row: (u64::from(a.latent) + u64::from(a.pe)) * stores::F16_BYTES,
                        index: 0,
                        window: None,
                    }
                }
                Mixer::DeltaRule(d) => Hold::State {
                    fixed: state_bytes(d),
                    v_heads: d.v_heads as usize,
                    d: d.d as usize,
                },
                Mixer::None => Hold::Nothing,
                other => return Err(unsized_store(layer, other.what())),
            });
        }
        Ok(KvLayout {
            holds,
            rows: Rows::Ring,
            rope_row: ropes
                .iter()
                .map(|&(_, dims)| u64::from(dims) * stores::F32_BYTES)
                .sum(),
            sink_row,
            lanes: NonZeroUsize::MIN,
        })
    }

    /// The same layout with the window layers on `rows`.
    #[must_use]
    pub fn with_rows(self, rows: Rows) -> KvLayout {
        KvLayout { rows, ..self }
    }

    /// The same layout with each delta-rule layer's state `lanes` lanes: a load that verifies
    /// `lanes` rows a pass keeps one a row.
    #[must_use]
    pub fn with_lanes(self, lanes: NonZeroUsize) -> KvLayout {
        KvLayout { lanes, ..self }
    }

    /// The bytes of layers `layers` at `ctx_max` positions: what a card that runs them holds beside
    /// its weights ([`KvBytes::layer_bytes`] summed).
    #[must_use]
    pub fn bytes(&self, layers: Range<usize>, ctx_max: u64) -> u64 {
        layers.map(|l| self.layer_bytes(l, ctx_max)).sum()
    }

    /// The K and V rows layer `layer` holds at `ctx_max`. A layer that keeps no positions, or is
    /// past the trunk, holds none. The index keys of a layer that selects blocks are not counted
    /// here: it has no window, so they are one for each of the `ctx_max` positions
    /// ([`KvBytes::layer_bytes`] adds them).
    #[must_use]
    pub fn rows_at(&self, layer: usize, ctx_max: u64) -> u64 {
        match self.holds.get(layer) {
            Some(Hold::Attention { window, .. }) => self.rows.count(*window, ctx_max),
            _ => 0,
        }
    }

    /// One resident sequence's stores over the trunk, nothing beside them.
    #[must_use]
    pub fn seq_terms(&self) -> SeqTerms<'_> {
        SeqTerms {
            layers: Stores {
                kv: self,
                count: self.holds.len(),
            },
            draft: None,
            beside: 0,
        }
    }
}

/// The bytes of index key a position a GQA layer keeps: 0 for a layer that selects nothing, one f16
/// key of the block selector's `d` values for a layer that selects blocks. A selector of another
/// kind, or a block selector beside a window (the index keys span every position), is `Err` with
/// the words a refusal names the mixer by.
fn index_row(g: &Gqa) -> Result<u64, &'static str> {
    match (&g.select, g.window) {
        (None, _) => Ok(0),
        (Some(Selector::BlockMax { d, .. }), None) => Ok(u64::from(*d) * stores::F16_BYTES),
        (Some(Selector::BlockMax { .. }), Some(_)) => {
            Err("a GQA mixer with a block selector and a window")
        }
        (Some(_), _) => Err("a GQA mixer with a selector other than the block selector"),
    }
}

/// The refusal of layer `layer`, a mixer this owner does not size (`what` names it).
fn unsized_store(layer: usize, what: &str) -> PlacementError {
    PlacementError::Unimplemented(vec![Unimplemented {
        layer: Some(layer),
        feature: format!(
            "{what}: this plan sizes the stores of a GQA layer (plain, or selecting blocks), a plain \
             latent layer and a delta-rule layer"
        ),
    }])
}

/// A delta-rule layer's first state lane and conv ring in bytes, its stamps apart.
fn state_bytes(d: &DeltaRule) -> u64 {
    let (v_heads, k_heads) = (d.v_heads as usize, d.k_heads as usize);
    let (state, conv) = (d.d as usize, d.conv as usize);
    stores::recurrent_bytes(v_heads, k_heads, state, conv)
}

impl KvBytes for KvLayout {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        match self.holds.get(layer) {
            Some(Hold::Attention { row, index, .. }) => self
                .rows_at(layer, ctx_max)
                .saturating_mul(*row)
                .saturating_add(ctx_max.saturating_mul(*index)),
            Some(Hold::State { fixed, v_heads, d }) => {
                fixed + stores::delta_lane_bytes(*v_heads, *d, self.lanes.get())
            }
            Some(Hold::Nothing) | None => 0,
        }
    }

    fn stage_position_bytes(&self, ctx_max: u64) -> u64 {
        (self.rope_row + self.sink_row) * ctx_max
    }
}

#[cfg(test)]
mod tests {
    use gguf::GgmlType;
    use models::{DeltaKind, Gqa, KdaGate, Mamba2, OutGate, PoolRule, Rope};
    use runtime::stores::ring_rows;

    use super::*;
    use crate::placement::accepts::Program;
    use crate::placement::{CardFormat, ModelTensor, Role, RoutedRule};

    /// A GQA layer of `kv` key heads, key width `k` and value width `v`, a window of `window`
    /// positions or none, and a rope of `rope` (base, dims) or none.
    fn gqa(kv: u32, (k, v): (u32, u32), window: Option<u32>, rope: Option<(f32, u32)>) -> Mixer {
        Mixer::Gqa(Gqa {
            heads: kv * 4,
            kv_heads: kv,
            head_dim: k,
            value_dim: v,
            rope: rope.map_or(
                Rope {
                    mode: RopeMode::None,
                    dims: 0,
                    base: 0.0,
                    yarn: None,
                    factors: false,
                },
                |(base, dims)| Rope {
                    mode: RopeMode::Neox,
                    dims,
                    base,
                    yarn: None,
                    factors: false,
                },
            ),
            qk_norm: false,
            o_norm: false,
            out_gate: OutGate::None,
            select: None,
            window,
            sinks: false,
            value_scale: None,
        })
    }

    /// `mixer`, a GQA layer, selecting blocks by an index key of `d` values: its rope the
    /// selector's too, as the family's description gives it.
    fn selecting(mixer: Mixer, d: u32) -> Mixer {
        let Mixer::Gqa(g) = mixer else {
            panic!("{mixer:?} is not a GQA layer");
        };
        Mixer::Gqa(Gqa {
            select: Some(Selector::BlockMax {
                heads: g.kv_heads,
                d,
                top_k: 16,
                block: 128,
                local: 1,
                rope: g.rope,
            }),
            ..g
        })
    }

    /// A layer's rows: its own key-head count's key and value rows in f16 a position. Layers 1 and
    /// 2 are a window of 8; 0 and 3 are full. At 9 positions the ring still holds every position,
    /// because [`ring_rows`] pads the window plus one prompt batch past 9. Past that pad the ring
    /// stays at the pad and the plane grows with the context. A layer past the trunk holds nothing.
    /// The widths are key 16 and value 8 values, 2 key heads on a full layer and 1 on a window one.
    #[test]
    fn layer_bytes() {
        let pad = ring_rows(8, u64::MAX);
        assert!(pad > 9, "the ring pad is past the 9-position case");
        let mixers = [
            gqa(2, (16, 8), None, None),
            gqa(1, (16, 8), Some(8), None),
            gqa(1, (16, 8), Some(8), None),
            gqa(2, (16, 8), None, None),
        ];
        let ring = KvLayout::of(&mixers).expect("GQA layers are sized");
        let plane = ring.clone().with_rows(Rows::Plane);
        for kv in [&ring, &plane] {
            assert_eq!(kv.layer_bytes(0, 9), 9 * 2 * 48);
            assert_eq!(kv.layer_bytes(1, 9), 9 * 48);
            assert_eq!(kv.layer_bytes(4, 9), 0);
            assert_eq!(kv.bytes(0..4, 1), 6 * 48);
            assert_eq!(kv.bytes(0..4, 9), 9 * 6 * 48);
        }
        let past = pad + 32;
        assert_eq!(ring.rows_at(1, past), pad);
        assert_eq!(plane.rows_at(1, past), past);
        assert_eq!(ring.layer_bytes(1, past), pad * 48);
        assert_eq!(plane.layer_bytes(1, past), past * 48);
        assert_eq!(ring.layer_bytes(0, past), past * 2 * 48);
        assert_eq!(plane.layer_bytes(0, past), past * 2 * 48);
    }

    /// The published MiMo trunk: nine full layers of 4 key heads and thirty-nine window layers of
    /// 8, each value 320 f16 bytes a head. A position is 222,720 B on either term, because the ring
    /// is the context. At the gate context the plane is that figure times 4,160 and the ring is the
    /// full layers' plane plus 768 window rows.
    #[test]
    fn the_ring_bytes_at_the_gate_context() {
        let full = [0usize, 5, 11, 17, 23, 29, 35, 41, 47];
        let mixers: Vec<Mixer> = (0..48)
            .map(|l| {
                if full.contains(&l) {
                    gqa(4, (192, 128), None, None)
                } else {
                    gqa(8, (192, 128), Some(128), None)
                }
            })
            .collect();
        let ring = KvLayout::of(&mixers).expect("GQA layers are sized");
        let plane = ring.clone().with_rows(Rows::Plane);
        assert_eq!(plane.bytes(0..48, 1), 222_720);
        assert_eq!(ring.bytes(0..48, 1), 222_720);
        assert_eq!(plane.bytes(0..48, 4_160), 4_160 * 222_720);
        assert_eq!(ring.bytes(0..48, 4_160), 249_200_640);
        assert_eq!(ring_rows(128, 4_160), 768);
    }

    /// A delta-rule layer at Solar-Open2's published widths (64 heads of 128, conv 4): the state
    /// 64 × 128 × 128 f32 = 4,194,304 B, the conv ring (3 + 8 rows) of the 3 × 64 × 128 q, k and
    /// v channels in f32 = 1,081,344 B, and one lane's u32 stamp = 4 B [derived]: the same on every
    /// context. Mutant: the stamp or the pass rows left out of the ring.
    #[test]
    fn a_delta_rule_layer_holds_its_state_whatever_the_context() {
        let kda = Mixer::DeltaRule(DeltaRule {
            kind: DeltaKind::Kda {
                gate: KdaGate::Softplus,
                beta_scale: 2.0,
            },
            k_heads: 64,
            v_heads: 64,
            d: 128,
            conv: 4,
        });
        let kv = KvLayout::of([&kda, &gqa(8, (128, 128), None, None)]).expect("both are sized");
        assert_eq!(kv.layer_bytes(0, 1), 4_194_304 + 1_081_344 + 4);
        assert_eq!(kv.layer_bytes(0, 1 << 20), kv.layer_bytes(0, 1));
        assert_eq!(kv.rows_at(0, 4096), 0);
        assert_eq!(kv.layer_bytes(1, 4096), 4096 * 2 * 8 * 128 * 2);
    }

    /// One sequence's terms are the layout's stores over the trunk with nothing beside them, and
    /// `slots` sequences count each layer `slots` times — the card's kv class of a load that serves
    /// them.
    #[test]
    fn seq_terms_count_every_sequence() {
        let mixers = [
            gqa(2, (16, 8), None, None),
            gqa(1, (16, 8), Some(8), None),
            gqa(1, (16, 8), Some(8), None),
            gqa(2, (16, 8), None, None),
        ];
        let kv = KvLayout::of(&mixers).expect("GQA layers are sized");
        let terms = kv.seq_terms();
        assert_eq!(terms.bytes(9, 1), kv.bytes(0..4, 9));
        assert_eq!(terms.bytes(9, 3), 3 * kv.bytes(0..4, 9));
        assert_eq!(terms.plan_kv(9, 3), 3 * kv.bytes(0..4, 9));
        let three = terms.slots_of(3);
        for l in 0..4 {
            assert_eq!(three.layer_bytes(l, 9), 3 * kv.layer_bytes(l, 9));
        }
        assert_eq!(three.layer_bytes(4, 9), 0);
    }

    /// The stage's rope rows: one table a distinct rope the rotating layers turn by, `dims` f32
    /// values a position, as the body copies them to its card. Two bases are two tables, the same
    /// base on every layer is one, and a layer that turns nothing adds none.
    #[test]
    fn the_stage_rope_rows_are_a_table_a_rope() {
        let rope = |bases: [f32; 3]| {
            let mixers = bases.map(|b| gqa(1, (16, 8), None, (b > 0.0).then_some((b, 4))));
            KvLayout::of(&mixers)
                .expect("GQA layers are sized")
                .stage_position_bytes(10)
        };
        assert_eq!(rope([1e6, 1e7, 1e6]), 2 * 4 * 4 * 10);
        assert_eq!(rope([1e7, 1e7, 1e7]), 4 * 4 * 10);
        assert_eq!(rope([0.0, 0.0, 0.0]), 0);
        assert_eq!(rope([0.0, 1e7, 0.0]), 4 * 4 * 10);
    }

    /// A layer that selects blocks keeps one f16 index key of the selector's `d` values for every
    /// position beside its planes, on the plane and on the ring alike (it has no window to cut
    /// them to); a layer that selects nothing keeps none. The widths are 2 key heads, key 16 and
    /// value 8 values (96 B of K and V a position) and an index key of 12 values (24 B): at 9
    /// positions a plain layer is 9 × 96 = 864 B, a selecting one 9 × (96 + 24) = 1,080 B, and past
    /// the ring's pad (where a window layer stops growing) the selecting layer still grows with the
    /// context. Mutant: the index keys left out, counted per window row, or counted on the plain
    /// layer.
    #[test]
    fn a_block_selector_adds_an_index_key_row_a_position() {
        let mixers = [
            gqa(2, (16, 8), None, None),
            selecting(gqa(2, (16, 8), None, None), 12),
            gqa(2, (16, 8), Some(8), None),
        ];
        let ring = KvLayout::of(&mixers).expect("plain and selecting GQA layers are sized");
        let plane = ring.clone().with_rows(Rows::Plane);
        let past = ring_rows(8, u64::MAX) + 32;
        for kv in [&ring, &plane] {
            assert_eq!(kv.layer_bytes(0, 9), 9 * 96);
            assert_eq!(kv.layer_bytes(1, 9), 9 * (96 + 24));
            assert_eq!(kv.layer_bytes(1, past), past * (96 + 24));
            assert_eq!(kv.rows_at(1, past), past);
            assert_eq!(kv.layer_bytes(3, 9), 0);
        }
        assert_eq!(ring.bytes(0..3, 9), 9 * 96 + 9 * 120 + 9 * 96);
    }

    /// MiniMax-M3's cache at the released file's widths: 4 key heads of 128 are 2,048 B of f16 K and
    /// V a position on every one of 60 layers (122,880 B), and the 57 sparse layers (all but the
    /// first 3) keep an index key of 128 f16 values, 256 B, each (14,592 B). At 32,768 and 131,072
    /// positions the planes are 4,026,531,840 and 16,106,127,360 B and the index keys 478,150,656
    /// and 1,912,602,624 B [derived: 122,880 and 14,592 times the positions]. A dense layer keeps no
    /// index keys, and a layer past the model nothing. The stage keeps one rope table of the 64
    /// rotated values in f32 (256 B) a position, though the sparse layers' indexer rotates by the
    /// same rope, and one sink row of the index key's 128 f16 values (256 B) a position, which the
    /// 57 selecting layers share: 512 B. Mutant: the index keys on the dense layers, the index key
    /// at the plane's width, the indexer's rope a second table, or a sink row a selecting layer.
    #[test]
    fn the_m3_cache_is_the_released_files_bytes() {
        let rope = Some((5_000_000.0, 64));
        let mixers: Vec<Mixer> = (0..60)
            .map(|l| {
                let layer = gqa(4, (128, 128), None, rope);
                if l < 3 { layer } else { selecting(layer, 128) }
            })
            .collect();
        let kv = KvLayout::of(&mixers).expect("M3's layers are sized");
        let total = |ctx: u64| (0..64).map(|l| kv.layer_bytes(l, ctx)).sum::<u64>();
        for l in 0..3 {
            assert_eq!(kv.layer_bytes(l, 10), 10 * 2048, "layer {l}");
        }
        for l in [3, 30, 59] {
            assert_eq!(kv.layer_bytes(l, 10), 10 * (2048 + 256), "layer {l}");
        }
        assert_eq!(kv.layer_bytes(60, 10), 0);
        assert_eq!(total(1), 60 * 2048 + 57 * 256);
        assert_eq!(total(1), 122_880 + 14_592);
        assert_eq!(total(32_768), 4_026_531_840 + 478_150_656);
        assert_eq!(total(131_072), 16_106_127_360 + 1_912_602_624);
        assert_eq!(kv.stage_position_bytes(10), (64 * 4 + 128 * 2) * 10);
    }

    /// An index rotation of its own is one more table beside the layer's: the stage copies a table a
    /// distinct (base, dims) pair of the rotations the layers turn by, the selector's included.
    #[test]
    fn an_index_rotation_of_its_own_is_a_table_of_its_own() {
        let layer = selecting(gqa(1, (16, 8), None, Some((1e6, 4))), 12);
        let Mixer::Gqa(mut g) = layer else {
            panic!("a GQA layer");
        };
        let same = KvLayout::of([&Mixer::Gqa(g.clone())]).expect("sized");
        let Some(Selector::BlockMax { rope, .. }) = &mut g.select else {
            panic!("a block selector");
        };
        rope.base = 1e7;
        let own = KvLayout::of([&Mixer::Gqa(g)]).expect("sized");
        // Beside the tables, the selector's sink row: 12 f16 values a position.
        assert_eq!(same.stage_position_bytes(10), (4 * 4 + 12 * 2) * 10);
        assert_eq!(own.stage_position_bytes(10), (2 * 4 * 4 + 12 * 2) * 10);
    }

    /// The selector's sink: the head kernel appends a value row beside each index key to one
    /// plane every selecting layer shares, so the stage keeps the widest selector's key a position
    /// once, whatever the selecting layers' count, and a trunk that selects nothing keeps none.
    #[test]
    fn the_selectors_share_one_sink_row() {
        let rows = |mixers: &[Mixer]| {
            KvLayout::of(mixers)
                .expect("GQA layers are sized")
                .stage_position_bytes(10)
        };
        let plain = gqa(1, (16, 8), None, None);
        assert_eq!(rows(&[plain.clone(), plain.clone()]), 0);
        let one = [selecting(plain.clone(), 12), plain.clone()];
        assert_eq!(rows(&one), 12 * 2 * 10);
        let two = [selecting(plain.clone(), 12), selecting(plain, 16)];
        assert_eq!(rows(&two), 16 * 2 * 10);
    }

    /// A GQA layer whose selector this owner does not size (a pooled-token or a stream selector) or
    /// whose block selector sits beside a window is refused by name with its layer, never sized as
    /// a plain layer.
    #[test]
    fn a_selector_this_owner_does_not_size_is_refused_by_name() {
        let plain = gqa(1, (16, 8), None, None);
        let Mixer::Gqa(base) = gqa(1, (16, 8), None, None) else {
            panic!("a GQA layer");
        };
        let pooled = Mixer::Gqa(Gqa {
            select: Some(Selector::TokenPool {
                heads: 1,
                d: 12,
                top_k: 16,
                pool: 4,
                rule: PoolRule::Learned { key_eps: 1e-6 },
            }),
            ..base
        });
        let Mixer::Gqa(windowed) = selecting(gqa(1, (16, 8), None, None), 12) else {
            panic!("a GQA layer");
        };
        let windowed = Mixer::Gqa(Gqa {
            window: Some(8),
            ..windowed
        });
        for (mixer, words) in [
            (pooled, "a selector other than the block selector"),
            (windowed, "a block selector and a window"),
        ] {
            let err = KvLayout::of([&plain, &mixer]).expect_err("refused");
            let PlacementError::Unimplemented(items) = err else {
                panic!("{err:?}, not the unimplemented list");
            };
            assert_eq!(items.len(), 1, "{items:?}");
            assert_eq!(items[0].layer, Some(1));
            assert!(items[0].feature.contains(words), "{items:?}");
        }
    }

    /// A mixer whose store this owner does not size is refused by name with its layer, never read
    /// as a layer that holds nothing.
    #[test]
    fn a_mixer_with_no_sized_store_is_refused_by_name() {
        let mamba = Mixer::Mamba2(Mamba2 {
            d_inner: 8,
            d_state: 4,
            n_heads: 2,
            head_dim: 4,
            n_groups: 1,
            d_conv: 4,
            d_in_proj: 24,
        });
        let err = KvLayout::of([&gqa(1, (16, 8), None, None), &mamba]).expect_err("refused");
        let PlacementError::Unimplemented(items) = err else {
            panic!("{err:?}, not the unimplemented list");
        };
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].layer, Some(1));
        assert!(items[0].feature.contains("a Mamba-2 mixer"), "{items:?}");
    }

    /// A stack is card-routed when the common kernel table gives its type the entry this
    /// program's leg launches for its part, and not otherwise, for each body on the common leg
    /// ([`routed_rule`]): a Q6_K down is carded and a Q6_K gate or up is not; the 32-value downs
    /// other families run (Q8_0 and IQ4_NL) and the types with no row (Q2_K, the floats) keep
    /// their layer's experts on the host.
    // PIN(2026-10-11): card_runs now runs Q4_K/Q5_K through the common leg.
    // PIN(2026-10-11): the leg also runs IQ2_XS, IQ3_XXS and IQ4_XS (the raw-block entries of
    // `iq_sel.rs`), so those three are card-routed too.
    // PIN(2026-10-12): the leg also runs IQ3_S (`iq3_s_gate_up_sel`, `iq3_s_gemv_sel`), so it is
    // card-routed too.
    // PIN(2026-10-22): each part by its own arm. A Q6_K down (the leg's 128-value `_sel` launch)
    // is on the card beside a gate and up the gate·up arm runs; Q6_K has no gate·up arm. The
    // 32-value downs (Q8_0, IQ4_NL) stay off: these bodies' leg buffers hold the 128-value form
    // alone. Mutants: the down arm read through `card_runs` (Q6_K's down is `None`); the down arm
    // read through `card_down_runs` (Q8_0's down is carded).
    #[test]
    fn only_the_stacks_the_leg_runs_are_card_routed() {
        let rules = [Program::Mimo2Body, Program::Solar2Body].map(|p| {
            let RoutedRule::Leg { gate_up, down, .. } = routed_rule(p) else {
                panic!("{p:?}: a leg rule");
            };
            (p, gate_up, down)
        });
        // Each part's format, the same under every body's rule.
        let arms = |ty: GgmlType| {
            let parts: Vec<_> = rules
                .iter()
                .map(|(_, gate_up, down)| (gate_up(ty), down(ty)))
                .collect();
            assert!(parts.windows(2).all(|w| w[0] == w[1]), "{ty}: {rules:?}");
            parts[0]
        };
        let k = Some(CardFormat::KQuant);
        for ty in [
            GgmlType::MXFP4,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::IQ2_XS,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_S,
            GgmlType::IQ4_XS,
        ] {
            assert_eq!(arms(ty), (k, k), "{ty}");
        }
        assert_eq!(arms(GgmlType::Q6_K), (None, k), "Q6_K");
        for ty in [
            GgmlType::Q8_0,
            GgmlType::IQ4_NL,
            GgmlType::Q2_K,
            GgmlType::Q3_K,
            GgmlType::F32,
            GgmlType::BF16,
        ] {
            assert_eq!(arms(ty), (None, None), "{ty}");
        }
    }

    /// The minimax-m3 body carries its routed parts by glm5next's planes arms: its leg buffers
    /// hold the form each card layer's down reads, NVFP4 in its planes among them. Mutant: the
    /// body left on the common leg (a 32-value down never carded).
    #[test]
    fn the_minimax_body_routes_by_the_planes_arms() {
        let arms = |p| {
            let RoutedRule::Leg { gate_up, down, .. } = routed_rule(p) else {
                panic!("{p:?}: a leg rule");
            };
            (gate_up, down)
        };
        let (mm_gu, mm_down) = arms(Program::MiniMaxM3Body);
        let (glm_gu, glm_down) = arms(Program::Glm5nextBody);
        for id in 0..64u32 {
            let ty = GgmlType::from_u32(id);
            assert_eq!((mm_gu(ty), mm_down(ty)), (glm_gu(ty), glm_down(ty)), "{ty}");
        }
        for ty in [GgmlType::NVFP4, GgmlType::NVFP4_ROWS, GgmlType::Q8_0] {
            assert_eq!((mm_gu(ty), mm_down(ty)), (glm_gu(ty), glm_down(ty)), "{ty}");
        }
        assert!(
            mm_down(GgmlType::Q8_0).is_some(),
            "a 32-value down is carded"
        );
    }

    /// A host room under the host experts' need is refused by name: the need, the bytes the plan
    /// pages, the room the plan was made at, and the engine that serves no NVMe expert tier.
    #[test]
    fn a_room_under_the_host_need_is_refused_by_name() {
        let err = PlaceError::NoNvmeTier {
            paged: 1_073_741_824,
            need: 150_000_000_000,
            room: "; the plan was made at a reading of 149000000000 B (plan room 149000000000 B)"
                .to_owned(),
            engine: "MiMo",
        };
        assert_eq!(
            err.to_string(),
            "the host need 150000000000 B passes the room, so the plan pages 1073741824 B of \
             routed experts to the NVMe expert tier; the plan was made at a reading of \
             149000000000 B (plan room 149000000000 B), and the MiMo engine serves no NVMe \
             expert tier; free host memory (the room is the host's available bytes, or \
             BLOOMERY_HOST_ROOM where set)"
        );
    }

    /// The planned ring forms: a role is the grouped GEMM's form where any routed layer's type reads
    /// it (M3.1's i-quant GGUF), the 32-value form where every layer's does (NVFP4, Q8_0), and a
    /// layer no route GEMM reads (an F32 header) or with no stacks adds nothing. Mutants: the
    /// first layer's forms alone, a layer with no route GEMM read as the grouped form.
    #[test]
    fn a_ring_pass_is_planned_in_its_stacks_largest_forms() {
        let model = |types: &[(usize, GgmlType, GgmlType)]| ModelTensors {
            tensors: types
                .iter()
                .flat_map(|&(l, gu, d)| {
                    let [g, u, dn] = crate::arch::names::routed_stacks(l);
                    [(g, gu), (u, gu), (dn, d)]
                })
                .map(|(name, ty)| ModelTensor {
                    name,
                    shard: 0,
                    layer: None,
                    role: Role::RoutedExperts,
                    ty,
                    dims: Vec::new(),
                    file_bytes: 0,
                    gathered_rows: None,
                })
                .collect(),
            layers: 4,
            experts: 8,
            experts_used: 2,
            bf16_words: |_| false,
        };
        let planned = |types: &[(usize, GgmlType, GgmlType)]| {
            ring_forms_planned(&model(types), crate::arch::names::routed_stacks)
        };
        let grouped = RingForms {
            gate_up32: false,
            down32: false,
        };
        assert_eq!(
            planned(&[(1, GgmlType::IQ2_XS, GgmlType::IQ3_XXS)]),
            grouped
        );
        assert_eq!(
            planned(&[
                (0, GgmlType::Q8_0, GgmlType::Q8_0),
                (2, GgmlType::IQ2_XS, GgmlType::IQ4_XS)
            ]),
            grouped
        );
        assert_eq!(
            planned(&[
                (0, GgmlType::Q8_0, GgmlType::IQ3_XXS),
                (1, GgmlType::Q8_0, GgmlType::Q8_0)
            ]),
            RingForms {
                gate_up32: true,
                down32: false
            }
        );
        assert_eq!(
            planned(&[(0, GgmlType::Q8_0, GgmlType::Q8_0)]),
            RingForms::BLOCKS32
        );
        assert_eq!(
            planned(&[(0, GgmlType::F32, GgmlType::F32)]),
            RingForms::BLOCKS32
        );
        assert_eq!(planned(&[]), RingForms::BLOCKS32);
    }
}
