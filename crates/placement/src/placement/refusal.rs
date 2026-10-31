use std::fmt;

use gguf::GgmlType;

use super::churn;
use super::machine::{Device, ModelTensor, unimplemented_list};
use super::room::{ArenaFloor, least_term};
use super::tp;
use super::workstation;
use super::{Role, Unimplemented};

/// Why a placement could not be built; each variant names what it refused.
#[derive(Debug, thiserror::Error)]
pub enum PlacementError {
    /// Every name the classifier has no rule for, not only the first.
    #[error("{} tensors have no role: {}", names.len(), names.join(", "))]
    Unclassified { names: Vec<String> },
    #[error("metadata {key}: {detail}")]
    Metadata { key: String, detail: String },
    #[error("tensor {name}: {detail}")]
    Tensor { name: String, detail: String },
    /// A tensor its role puts on a card, of a type no card format loads.
    #[error(
        "tensor {name}: type {ty} has no device format, but its role ({role}) puts it on a card"
    )]
    NoCardFormat {
        name: String,
        ty: GgmlType,
        role: Role,
    },
    /// A routed stack with experts on the host, of a type or width no host kernel runs
    /// ([`crate::kernels::host_stack`]): the host would hold the bytes and compute nothing.
    #[error(
        "tensor {name}: experts {experts} on the host, where no host kernel runs {ty} rows of {k} \
         values ({why})"
    )]
    NoHostKernel {
        name: String,
        ty: GgmlType,
        k: usize,
        experts: String,
        why: qdot::QdotError,
    },
    /// The cards' layer ranges do not cover the model once, in order, with one head.
    #[error("stage map: {0}")]
    Stages(String),
    /// The host tier's lock, or its residency query, failed: the span and the
    /// kernel's answer.
    #[error("host lock: {0}")]
    Host(String),
    /// The host's available bytes could not be read
    /// ([`workstation::host_room`]'s reading): the file or value that failed.
    #[error("the host's room: {0}")]
    HostRoom(String),
    /// The draft layer's routed experts, held on the host beside the trunk's,
    /// pass the host room the plan was made at: the trunk has no room left to
    /// split at.
    #[error(
        "the draft layer's routed experts, {draft} B, pass the host room {room} B: free host \
         memory, or serve without the draft"
    )]
    DraftHostRoom { room: u64, draft: u64 },
    /// A host set, a walk or a page release met an r8 sidecar it cannot take
    /// beside its pair: the sidecar's own named refusal.
    #[error(transparent)]
    R8(#[from] crate::r8::R8Error),
    /// A tensor-parallel cut or rank of a named tensor refused.
    #[error(transparent)]
    Tp(#[from] tp::TpError),
    /// An expert list that is not one: a repeated id, an id past the stack.
    #[error("expert list: {0}")]
    Experts(String),
    /// A host room under the floor of the NVMe expert tier: the plan's host
    /// terms outside the routed experts, and the transient reads of two
    /// layers' NVMe-tier experts and one layer's host experts beside them —
    /// and, where the split dial would give the room an arena, the arena's
    /// least ([`ArenaFloor`]): a room between the two is the one the dial
    /// refuses rather than page through the file mapping.
    #[error(
        "the host room {room} B is under the NVMe expert tier's floor {floor} B (the host terms \
         beside the routed experts {base} B + 3 × the heaviest layer's host-served experts \
         {layer} B: two layers' transient reads and one layer's resident experts{least}), the \
         plan's need being {need} B: free host memory, or load by a placement whose cards hold \
         more of the model; a room read from the host is the reading less {drift} B, the host \
         bytes the load's own start may take",
        least = least_term(.arena.as_ref()),
        drift = workstation::HOST_DRIFT
    )]
    HostRoomFloor {
        room: u64,
        floor: u64,
        base: u64,
        layer: u64,
        need: u64,
        /// The arena's least, counted in `floor`; `None` where the floor
        /// holds none (a set `BLOOMERY_NVTIER_BYTES`, or a plan the dial
        /// gives no arena).
        arena: Option<ArenaFloor>,
    },
    /// An arena the lever `BLOOMERY_NVTIER_BYTES` names that the room cannot
    /// take: the host segments it leaves fall under the NVMe expert tier's
    /// floor.
    #[error(
        "the NVMe tier's arena {arena} B leaves the host room {room} B under its floor {floor} B \
         (the arena takes at most room − floor): lower the arena, or free host memory"
    )]
    NvTierArena { arena: u64, room: u64, floor: u64 },
    /// A unified machine's plan the NVMe expert tier's floor refused at the
    /// cards' full share, planned again under lowered card budgets
    /// ([`plan_in_pool`]) and refused at the last: the rounds, the last
    /// budget and the last refusal, and the host room and the floor of the
    /// first refusal.
    #[error(
        "on a unified pool the cards' share was lowered {rounds} times, to a card budget of \
         {budget} B, and the plan is still refused: {last}"
    )]
    PoolBudget {
        rounds: u32,
        budget: u64,
        last: Box<PlacementError>,
        room: u64,
        floor: u64,
    },
    /// An arena the lever `BLOOMERY_NVTIER_BYTES` names under the arena's
    /// floor ([`ArenaFloor`]): a call would name more distinct ids of a
    /// paged layer than it has slots.
    #[error(
        "the NVMe tier's arena {arena} B (BLOOMERY_NVTIER_BYTES) is under its floor, {least}: one \
         call's ids would evict each other; set at least {bytes} B, or 0 to page through the \
         file mapping",
        bytes = .least.bytes
    )]
    NvTierArenaUnder { arena: u64, least: ArenaFloor },
    /// A routed stack's expert on two cards at once, by their index in
    /// [`Machine::all_cards`]: an expert lives on one device.
    #[error("tensor {tensor}: expert {expert} is on card #{first} and on card #{second}")]
    ExpertOnTwoCards {
        tensor: String,
        expert: u32,
        first: usize,
        second: usize,
    },
    /// An expert tier card that says it runs a stage: layers, the head or
    /// the token embedding.
    #[error("tier card {card}: {detail}; a tier card runs no stage")]
    Tier { card: String, detail: String },
    /// Cards of one device ([`Card::same_device`]) whose budgets (each card's usable bytes, under a card budget the
    /// capped ones) sum past that device's usable bytes: one device's
    /// budget counted twice.
    #[error(
        "card {card}: {cards} plan cards are this one device, and their budgets sum to {budgets} \
         B, past its usable {usable} B: one device's budget counted twice (a card budget that \
         splits the device between them is planned)"
    )]
    DeviceTwice {
        card: String,
        cards: usize,
        budgets: u64,
        usable: u64,
    },
    /// An expert tier card left with no expert, because the cards before it
    /// already hold every routed expert a card can: a tier the plan does
    /// not need.
    #[error(
        "tier card {card} (tier {tier}) holds no expert: the cards before it hold every routed \
         expert of the layers whose stacks a card can hold; plan without it"
    )]
    IdleTier { card: String, tier: usize },
    /// A machine of several nodes other than one a tier card: this plan
    /// holds one node's pool and host and each tier's card on its own node,
    /// and [`nodes::plan_nodes`] plans the others.
    #[error(
        "a machine of {nodes} nodes for {tiers} tier cards: the card plan plans one node, or one \
         more a tier card; plan the others with placement::nodes"
    )]
    Nodes { nodes: usize, tiers: usize },
    /// A unified pool whose bytes and its cards' context self cost
    /// ([`PoolNeed::pool`]) pass u64.
    #[error("a unified pool of {bytes} B and its cards' context of {context} B pass u64 bytes")]
    PoolPastU64 { bytes: u64, context: u64 },
    /// A tier card on a plan that keeps every routed stack on the host
    /// ([`plan_host_routed`]): no card holds a routed expert, so the tier
    /// has nothing to hold.
    #[error(
        "tier card {card} (tier {tier}) holds no expert: this plan puts every \
         routed expert on the host; plan without it"
    )]
    HostRoutedTier { card: String, tier: usize },
    /// A named reserve ([`Machine::reserve`]) on a device the plan does not
    /// count: a card past the device list (`cards` counts the stage cards and
    /// the tier cards), the NVMe tier, or nowhere.
    #[error(
        "the {reserve} reserve is on {on:?}, outside the plan: a reserve sits on one of the \
         plan's {cards} cards, stage first, or on the host"
    )]
    ReserveOffPlan {
        reserve: String,
        on: Device,
        cards: usize,
    },
    /// A layer the plan's per-layer expert counts ([`Plan::n_l`],
    /// [`Plan::tier_n_l`]) do not reach: `layers` is how many they hold.
    #[error("layer {layer} is past the plan's per-layer expert counts ({layers} layers)")]
    LayerPastCounts { layer: usize, layers: usize },
    /// A layer whose stage card and tier cards hold more experts than the
    /// model has on it: the counts the plan holds, and the model's.
    #[error(
        "layer {layer}: the stage card holds {stage} experts and the tier cards {tiers}, past \
         the model's {experts}"
    )]
    LayerOverHeld {
        layer: usize,
        stage: u64,
        tiers: u64,
        experts: u64,
    },
    /// A tier card's prompt-batch staging and scratch summed past u64.
    #[error(
        "the tier card's prompt-batch staging of {staging} B and scratch of {scratch} B pass u64 bytes"
    )]
    TierBatchCard { staging: u64, scratch: u64 },
    /// The tiers' prompt-batch host rows, one set a tier, summed past u64.
    #[error("{tiers} tier cards' prompt-batch host rows of {bytes} B each pass u64 bytes")]
    TierBatchHost { tiers: usize, bytes: u64 },
    /// Features of the file the engine does not run yet — every one, each with
    /// the layer it is on, or none for a model-wide one.
    #[error("{} feature(s) of this file are not implemented: {}", .0.len(), unimplemented_list(.0))]
    Unimplemented(Vec<Unimplemented>),
    /// Adaptive residency's churn pool does not fit the host: the stage card
    /// experts past the pinned ones, which the host set holds too
    /// ([`churn::ChurnPool`]), take more than the plan's host headroom.
    #[error("{0}")]
    ResidencyOverHost(Box<churn::OverHost>),
    /// A card's reserves, set-aside, load floor or shared-pool total summed past u64 bytes.
    #[error("card {card}: required card or shared-pool bytes pass u64 bytes")]
    CardBytesPastU64 { card: String },
    /// The host's reserves, with the row-gathered tables' page-cache reserve,
    /// summed past u64 bytes: no host figure holds them.
    #[error("the host's reserves pass u64 bytes")]
    HostBytesPastU64,
    /// A card budget below what the card needs with no expert on it: the
    /// floor counts what the caller holds beside the plan
    /// ([`plan_routed_reserving`]) as the expert rule's budget does.
    #[error(
        "card {card}: budget {budget} B is below its floor {floor} B = dense {dense} B (the \
         allocator's granules) + KV {kv} B + context {context} B + scratch {scratch} B + \
         reserves {reserves} B + margin {margin} B + beside {beside} B (what the caller holds \
         beside the plan)"
    )]
    CardBudgetFloor {
        card: String,
        budget: u64,
        floor: u64,
        dense: u64,
        kv: u64,
        context: u64,
        scratch: u64,
        reserves: u64,
        margin: u64,
        beside: u64,
    },
    /// A per-card budget on a plan with a card no CUDA ordinal of this
    /// process names ([`card_budget::check`]).
    #[error("card {card}: the per-card budget {budget} names cards by CUDA ordinal, and {why}")]
    CardBudgetCard {
        card: String,
        budget: String,
        why: &'static str,
    },
    /// A per-card budget or margin (`what`) naming a device the census does
    /// not list ([`card_budget::check_visible`]).
    #[error(
        "the per-card {what} {budget} names CUDA{ordinal}, which this process's census does not \
         list (visible: {visible})"
    )]
    CardBudgetUnseen {
        what: &'static str,
        ordinal: u32,
        budget: String,
        visible: String,
    },
    /// A margin that leaves a card no usable byte ([`card_budget::keep_free`]).
    #[error(
        "card {card}: a margin of {margin} B kept free leaves none of its {usable} B a plan may \
         use"
    )]
    CardMargin {
        card: String,
        margin: u64,
        usable: u64,
    },
    /// A budget by shares the plan cannot turn into a budget per card
    /// ([`card_budget::split`]), with the terms that say why.
    #[error("card budget {shares}: {why}")]
    CardShares { shares: String, why: String },
    /// A card whose device another process holds: what it had free at plan
    /// time is below the card's floor — the dense tensors, KV, set-aside,
    /// margin and what the caller holds beside the plan, before any expert
    /// moved to the host. [`CardFreeFloor`]'s terms, boxed to keep the error
    /// small.
    #[error("{}", .0.as_ref())]
    CardFreeFloor(Box<CardFreeFloor>),
    /// A plan that holds its card but leaves `layers` of the stage's layers
    /// under `least` card experts each ([`super::ctx_ask::card_experts_floor`]):
    /// it holds `room` bytes of the `need` bytes `least` experts a layer take
    /// over every layer with routed experts. A context whose sequence state
    /// leaves the experts that little room is what this names, so a smaller
    /// context is what it asks for.
    #[error(
        "card {card}: the plan leaves {layers} layer(s) under {least} experts a layer, the least \
         the unset residency runs on: it holds {room} B of the {need} B that takes"
    )]
    CardExpertsFloor {
        card: String,
        least: u64,
        layers: usize,
        need: u64,
        room: u64,
    },
}

/// The terms of [`PlacementError::CardFreeFloor`]: a card whose device
/// another process holds, what it had free, and the trunk that alone passes
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardFreeFloor {
    pub card: String,
    /// The device shares the host's memory, so its free bytes are the host room.
    pub unified: bool,
    /// The device's free bytes at plan time.
    pub free: u64,
    /// The card's usable bytes.
    pub usable: u64,
    /// The dense trunk's floor: dense, KV, set-aside, margin and what the
    /// caller holds beside the plan summed.
    pub floor: u64,
    /// How far the floor passes the free bytes.
    pub over: u64,
    pub dense: u64,
    pub kv: u64,
    /// The card's context term less the part the free reading already
    /// carries.
    pub context: u64,
    pub scratch: u64,
    pub reserves: u64,
    pub margin: u64,
    /// What the caller holds beside the plan on the card
    /// ([`plan_routed_reserving`]).
    pub beside: u64,
    /// "" or " (held by …)", the census's holder list.
    pub held: String,
}

impl fmt::Display for CardFreeFloor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let CardFreeFloor {
            card,
            unified,
            free,
            usable,
            floor,
            over,
            dense,
            kv,
            context,
            scratch,
            reserves,
            margin,
            beside,
            held,
        } = self;
        write!(f, "card {card}: ")?;
        if *unified {
            write!(
                f,
                "the device's host room is {free} B ({} when set, otherwise the host's memory \
                 reading), of its usable {usable} B at plan time{held}",
                bloomery_levers::HOST_ROOM
            )?;
        } else {
            write!(
                f,
                "the device had {free} B free of its usable {usable} B at plan time{held}"
            )?;
        }
        write!(
            f,
            ", and the plan's dense trunk alone needs {floor} B = dense {dense} B (the \
             allocator's granules) + KV {kv} B + context {context} B + scratch {scratch} B + \
             reserves {reserves} B + margin {margin} B + beside {beside} B (what the caller holds \
             beside the plan), past the {free} B by {over} B: "
        )?;
        if *unified {
            write!(
                f,
                "raise or unset {} if set, or free host memory",
                bloomery_levers::HOST_ROOM
            )?;
        } else {
            f.write_str("free the card")?;
        }
        f.write_str(", or plan a placement on a card with room")
    }
}

impl std::error::Error for CardFreeFloor {}

impl PlacementError {
    /// The refusal that names a tensor: `host_lock` (in `bloomery-model`)
    /// builds these through the re-export.
    pub fn tensor(t: &ModelTensor, detail: impl Into<String>) -> PlacementError {
        PlacementError::Tensor {
            name: t.name.clone(),
            detail: detail.into(),
        }
    }
}

/// An invariant a plan breaks. [`Plan::violations`] returns every one.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Violation {
    /// A tensor with no row, or with more than one.
    PlacedTimes { tensor: String, times: usize },
    /// Segments its role does not allow: an expert stack that does not cover
    /// every expert exactly once, a whole tensor in pieces, or a card segment
    /// whose buffers cannot be derived or do not sum to its resident bytes.
    Segments { tensor: String, detail: String },
    /// A segment on a card whose stage does not run the tensor's layer or
    /// head, or on a tier card of a tensor that is not a routed stack.
    WrongCard { tensor: String, card: String },
    /// A card whose uploads' granules, KV, scratch, context and reserves pass
    /// usable − margin.
    CardOver {
        card: String,
        total: u64,
        limit: u64,
    },
    /// The plan's own host terms (the host's tensors, the cards' ring
    /// shadows and the machine's reserves, as a [`workstation::HostNeed`])
    /// pass the room the machine's figure leaves
    /// ([`workstation::HostNeed::machine_room`]): `total` is them with the
    /// OS reserve in it, beside the host's usable bytes.
    HostOver { total: u64, usable: u64 },
    /// On a unified machine, the `cards`' [`Violation::CardOver`] totals
    /// with their margins, the `host`'s need and the `beside` bytes the load
    /// holds past the plan's own terms ([`Plan::violations_beside`])
    /// together pass the pool: each side may fit its own figure while both
    /// read the same bytes. `pool` is the pool's bytes with each card's
    /// context self cost added back, as [`capped`] adds it to a card.
    PoolOver {
        cards: u64,
        host: u64,
        beside: u64,
        pool: u64,
    },
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Violation::PlacedTimes { tensor, times } => {
                write!(f, "tensor {tensor} is placed {times} times, not once")
            }
            Violation::Segments { tensor, detail } => write!(f, "tensor {tensor}: {detail}"),
            Violation::WrongCard { tensor, card } => {
                write!(
                    f,
                    "tensor {tensor} is on card {card}, whose stage does not use it"
                )
            }
            Violation::CardOver { card, total, limit } => write!(
                f,
                "card {card}: resident + rounding + KV + scratch + context + reserves = {total} B passes usable − margin = {limit} B by {} B",
                total - limit
            ),
            Violation::HostOver { total, usable } => write!(
                f,
                "host: tensors + ring shadows + reserves = {total} B pass usable {usable} B by {} B",
                total - usable
            ),
            Violation::PoolOver {
                cards,
                host,
                beside,
                pool,
            } => {
                let total = cards + host + beside;
                write!(
                    f,
                    "unified pool: the cards with their margins {cards} B + the host's need \
                     {host} B + {beside} B the load holds beside the plan = {total} B pass the \
                     pool {pool} B by {} B",
                    total - pool
                )
            }
        }
    }
}
