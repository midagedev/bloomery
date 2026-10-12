//! This workstation's figures for a plan (`docs/v41-placement.md` §4–§5): its
//! two cards, the host's RAM and reserves, the serving context, the two layer
//! maps of design §5, the expert-tier map (b′) and the gate placement, and
//! where the V4.1 file lies. The
//! placement gate and the GPU load gates build their plans from here, so they
//! check the same plans.

use std::fmt;
use std::num::NonZeroU64;
use std::ops::Range;

pub use super::devices::{
    ALIASES, CardNotInView, DeviceId, DeviceInfo, ORDINALS, Pick, PickError, PickWhy, WordError,
    census_usable, device_on_host, label, listed_index, resolve, spec_of_device, visible,
    word_picks, workstation_spec,
};
use super::{Card, Host, HostTotals, Machine, PlacementError, Plan, UnifiedPool};

pub const MIB: u64 = 1 << 20;

/// One card as `nvidia-smi` reports it with no process on it [measured], or
/// as the census gives a device ([`spec_of_device`]): its memory, the part
/// the driver keeps, and the device it is when it was resolved against this
/// process's devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CardSpec {
    /// The name a plan and its records give the card; a known card's driver
    /// name contains it.
    pub name: &'static str,
    pub total_bytes: u64,
    pub driver_reserve_bytes: u64,
    /// The device, when resolved ([`resolve`]); `None` for a card of this
    /// workstation by name, which the open finds by that name.
    pub device: Option<DeviceId>,
    /// The device's free bytes at census time, when the spec came from a
    /// census ([`spec_of_device`]); `None` for a card by name, whose plan
    /// knows only the card. The plan's card budget term is the usable bytes
    /// capped by this ([`super::plan_with`]).
    pub free_bytes: Option<u64>,
    /// The other processes holding the device at census time, one named
    /// list, when the tool that reads them answered; `None` names none.
    pub held_by: Option<&'static str>,
    /// The device allocates from host memory ([`DeviceInfo::integrated`]):
    /// its free reading is the host's room, and a machine of it plans
    /// against one pool ([`unified_of`]). False for a card by name.
    pub integrated: bool,
}

impl CardSpec {
    /// What the driver leaves for us.
    #[must_use]
    pub const fn usable_bytes(&self) -> u64 {
        self.total_bytes - self.driver_reserve_bytes
    }

    /// Whether `self` and `other` are one device: by device when both were
    /// resolved, else by name.
    #[must_use]
    pub fn same_device(&self, other: &CardSpec) -> bool {
        match (self.device, other.device) {
            (Some(a), Some(b)) => a.uuid == b.uuid,
            _ => self.name == other.name,
        }
    }
}

/// The RTX A6000 [measured, `nvidia-smi`].
pub const A6000: CardSpec = CardSpec {
    name: "A6000",
    total_bytes: 49_140 * MIB,
    driver_reserve_bytes: 548 * MIB,
    device: None,
    free_bytes: None,
    held_by: None,
    integrated: false,
};

/// The RTX 3090 [measured, `nvidia-smi`].
pub const RTX_3090: CardSpec = CardSpec {
    name: "3090",
    total_bytes: 24_576 * MIB,
    driver_reserve_bytes: 400 * MIB,
    device: None,
    free_bytes: None,
    held_by: None,
    integrated: false,
};

/// Every card a plan names by name, the most usable bytes first: the names
/// a `--place` word takes ([`word_picks`]), and this workstation's cards by
/// size ([`Pick::Rank`] on a census-free plan).
pub const CARDS: [CardSpec; 2] = [A6000, RTX_3090];

const _: () = assert!(A6000.usable_bytes() > RTX_3090.usable_bytes());

/// Per card: the CUDA context and the m = 1 scratch [assumed], and the margin
/// the expert rule leaves free. The context term covers the primary context's
/// own creation ([`CONTEXT_SELF`]) and what a load takes from the device past
/// a census reading that no other plan term counts: the bundles' module
/// images, the kernels' local-memory growth, a capture's graph.
pub const CONTEXT: u64 = 512 * MIB;
/// The primary context's own creation cost [measured: an idle A6000's census
/// reading sits this far below its usable bytes; an idle 3090's sits less far
/// below, a `Gpu` and its module included]. A census reads free bytes on that
/// context, so the reading is already net of it; a plan's cap adds it back
/// (placement's `capped`) and counts the context once, in [`CONTEXT`].
pub const CONTEXT_SELF: u64 = 271 * MIB;
pub const SCRATCH: u64 = 64 * MIB;
pub const MARGIN: u64 = 1 << 30;

/// Both cards' allocation granule [measured: what a load takes from
/// `cuMemGetInfo` is a whole number of them, and
/// `cuMemGetAllocationGranularity` reports the same size].
pub const GRANULE: NonZeroU64 = NonZeroU64::new(2 * MIB).expect("2 MiB is not zero");

/// The host's usable bytes and the OS and everything else [measured,
/// `free -b`].
pub const HOST_USABLE: u64 = 270_071_001_088;
pub const OS_OTHER: u64 = 6_694_629_376;

/// The plan room of this workstation: the host room a load has with nothing
/// loaded, [`HOST_USABLE`] less [`OS_OTHER`] [derived] — the room
/// [`HostNeed::check`] holds a plan of [`host`] to ([`HostNeed::machine_room`]),
/// and the one a plan that must not move with the host's reading is made at
/// (`tools/ref/plan-room.sh` reads this line, a literal, and nothing else).
pub const PLAN_ROOM: u64 = 263_376_371_712;

const _: () = assert!(PLAN_ROOM == HOST_USABLE - OS_OTHER);

/// The engram row cache: the hot rows a reader of row-gathered tables keeps
/// beside the drive, which a plan sets aside on the host while such a table
/// lies on the NVMe tier and its maker sized no reader of its own
/// ([`super::row_table_tier`] replaces it with [`super::row_room`]).
pub const ROW_CACHE: u64 = 4_294_967_296;

/// The page the row room counts in ([`super::row_room`]): the host's base
/// page, 4 KiB on x86_64, the unit a fault and an advise read.
pub const ROW_PAGE: u64 = 4096;

/// What the machine's [`OS_OTHER`] is called: what the OS and every other
/// process held when [`HOST_USABLE`] was measured.
pub const OS_RESERVE: &str = "OS and other";

/// The serving context the plans are made for.
pub const CTX_MAX: u64 = 32_768;

/// Where plan (b) cuts: a group boundary, so no compressed stream, index key
/// or top-k id crosses the cards.
pub const CUT: usize = 20;

/// The V4.1 first shard to open: [`gguf::v41::model`], the path's owner.
#[must_use]
pub fn model_v41() -> String {
    gguf::v41::model()
}

/// `spec` running `layers`, with this machine's context, scratch and margin.
/// The spec's free reading and its holders ride on the card for the plan's
/// budget term and its refusals ([`Card::free_bytes`], [`Card::held_by`]).
fn card(spec: CardSpec, layers: Range<usize>, head: bool) -> Card {
    Card {
        name: spec.name.to_string(),
        device: spec.device,
        usable_bytes: spec.usable_bytes(),
        context_bytes: CONTEXT,
        scratch_bytes: SCRATCH,
        margin_bytes: MARGIN,
        granule_bytes: GRANULE,
        layers,
        head,
        token_embedding: false,
        reserves: Vec::new(),
        free_bytes: spec.free_bytes,
        held_by: spec.held_by,
    }
}

/// `spec` as an expert tier card: no stage, this machine's context, scratch
/// and margin.
fn tier(spec: CardSpec) -> Card {
    card(spec, 0..0, false)
}

/// The host tier with its reserves: the OS's alone. What a plan's own
/// tables need beside them — the row reserve of its row-gathered tables on
/// the NVMe tier — is the plan's ([`super::HostTotals::row_reserve_bytes`]).
#[must_use]
pub fn host() -> Host {
    Host {
        usable_bytes: HOST_USABLE,
        reserves: vec![(OS_RESERVE.to_string(), OS_OTHER)],
    }
}

/// The pool a machine of `spec` plans against: the census's reading of an
/// integrated device ([`CardSpec::integrated`]), which is the host's room;
/// `None` for a card with memory of its own.
///
/// # Panics
///
/// On an integrated spec with no free reading: its pool was never read, and
/// planning it as a card of its own beside this workstation's host would
/// count the one memory twice. A census spec ([`spec_of_device`]) always
/// carries the reading.
#[must_use]
pub fn unified_of(spec: &CardSpec) -> Option<UnifiedPool> {
    if !spec.integrated {
        return None;
    }
    let Some(bytes) = spec.free_bytes else {
        panic!(
            "{}: an integrated card with no free reading has no pool to plan against; take its \
             spec from the census (spec_of_device)",
            spec.name
        );
    };
    Some(UnifiedPool { bytes })
}

/// The host a machine of `spec` plans beside: on a unified machine the
/// pool, with no OS reserve, because the reading already leaves out what the
/// OS and every other process hold; this workstation's [`host`] otherwise.
/// Panics as [`unified_of`] does.
#[must_use]
pub fn host_of(spec: &CardSpec) -> Host {
    match unified_of(spec) {
        Some(pool) => Host {
            usable_bytes: pool.bytes,
            reserves: Vec::new(),
        },
        None => host(),
    }
}

/// `spec` runs all `layers` and the head, beside the host tier alone.
#[must_use]
pub fn plan_on(spec: CardSpec, layers: usize) -> Machine {
    Machine {
        cards: vec![card(spec, 0..layers, true)],
        tiers: Vec::new(),
        host: host_of(&spec),
        unified: unified_of(&spec),
    }
}

/// The machine a placed load runs on: [`plan_on`] with the token embedding
/// on the card too and `scratch` bytes of scratch, the program's own figure
/// (its m = 1 scratch and what its load's arenas hold beyond it). No expert
/// tier.
#[must_use]
pub fn placed_on(spec: CardSpec, layers: usize, scratch: u64) -> Machine {
    let mut machine = plan_on(spec, layers);
    let card = &mut machine.cards[0];
    card.token_embedding = true;
    card.scratch_bytes = scratch;
    machine
}

/// Design §5 (a): the A6000 runs all `layers` and the head; the 3090 is idle.
#[must_use]
pub fn plan_a(layers: usize) -> Machine {
    plan_on(A6000, layers)
}

/// Design §5 (b): the A6000 runs the layers below [`CUT`], the 3090 the rest
/// and the head.
#[must_use]
pub fn plan_b(layers: usize) -> Machine {
    Machine {
        cards: vec![
            card(A6000, 0..CUT, false),
            card(RTX_3090, CUT..layers, true),
        ],
        tiers: Vec::new(),
        host: host(),
        unified: None,
    }
}

/// What the DSpark draft's reserve on a card is called.
pub const DRAFT_RESERVE: &str = "DSpark draft";

/// What the expert tier's prompt-batch service is called as a reserve: on
/// the tier card, its staging and its tile scratch; on the host, the rows it
/// hands back and the places it reads.
pub const TIER_BATCH_RESERVE: &str = "tier prompt batch";
pub const TIER_BATCH_HOST_RESERVE: &str = "tier prompt batch rows";

/// The expert tier's prompt-batch service in bytes ([`tier_batch_bytes`]):
/// on the tier card, the staging a block is copied into — the activations,
/// the tier places, their q8_1 form, the down outputs by slot — and the tile
/// path's scratch; on the host, per exchange set, the rows it hands back and
/// the places it reads. The GPU side checks its allocations against these at
/// load and refuses a difference by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierBatchBytes {
    pub staging: u64,
    pub scratch: u64,
    pub host: u64,
}

impl TierBatchBytes {
    /// The tier card's part.
    #[must_use]
    pub const fn card(&self) -> u64 {
        self.staging + self.scratch
    }
}

/// Experts the tile path's bucket table takes a layer at most, and the
/// columns a tile holds: the tile path's shapes, which size its scratch.
const TILE_BUCKET_EXPERTS: u64 = 1024;
const TILE_COLS: u64 = 8;

/// The q8_1 activation of `m` columns of `k` values as the card lays it out:
/// its Q3_K, Q4_K and Q6_K codes, its sums and its scales.
const fn q8_act_bytes(m: u64, k: u64) -> u64 {
    let n_sb = k / 256;
    8 * m * 64 * n_sb.div_ceil(2)
        + 4 * m * 256 * n_sb.div_ceil(4)
        + 4 * m * 128 * n_sb.div_ceil(2)
        + 4 * m * 8 * n_sb
        + 4 * m * 2 * n_sb
}

/// The tier card's staging of the tier's prompt-batch service for blocks of
/// up to `cols` tokens of rows of `n_embd`, `n_used` slots a token: the
/// activations, the tier places, their q8_1 form, the down outputs by slot.
#[must_use]
pub const fn tier_batch_staging_bytes(n_embd: u64, n_used: u64, cols: u64) -> u64 {
    let slots = cols * n_used;
    4 * cols * n_embd + 4 * slots + q8_act_bytes(cols, n_embd) + 4 * slots * n_embd
}

/// The tile path's scratch for blocks of up to `slots` slots of rows of
/// `n_embd` through experts of `ff`: the bucket table, the tiles, the q8_1
/// planes by entry, the SwiGLU outputs and their q8_1 form.
#[must_use]
pub const fn tier_block_scratch_bytes(n_embd: u64, ff: u64, slots: u64) -> u64 {
    let n_sb = n_embd / 256;
    let tile_cap = {
        let spread = (slots + (TILE_COLS - 1) * TILE_BUCKET_EXPERTS) / TILE_COLS;
        if slots < spread { slots } else { spread }
    };
    4 * slots
        + 4 * (TILE_BUCKET_EXPERTS + 1)
        + 4 * (tile_cap + 1)
        + 8 * slots * 64 * n_sb.div_ceil(2)
        + 4 * slots * 2 * n_sb
        + 4 * slots * ff
        + q8_act_bytes(slots, ff)
}

/// The Walk A tile path's scratch for prompt batches of up to `cols` tokens
/// of rows of `n_embd`, `n_used` slots a token, through experts of `ff`
/// (the K-quant grouped entries, `bloomery_gpu::kquant::tiles`): the rows'
/// q8_1 form a column a token, the bucket table, the tiles, the entries'
/// Walk A planes (q4 codes, 32-value sums, 128-value scales), the SwiGLU
/// outputs by entry and their q8_1 form, and the downs by slot.
#[must_use]
pub const fn walk_a_tile_scratch_bytes(n_embd: u64, ff: u64, n_used: u64, cols: u64) -> u64 {
    let slots = cols * n_used;
    let n_sb = n_embd / 256;
    let tile_cap = {
        let spread = (slots + (TILE_COLS - 1) * TILE_BUCKET_EXPERTS) / TILE_COLS;
        if slots < spread { slots } else { spread }
    };
    q8_act_bytes(cols, n_embd)
        + 4 * slots
        + 4 * (TILE_BUCKET_EXPERTS + 1)
        + 4 * (tile_cap + 1)
        + 4 * slots * (256 * n_sb.div_ceil(4) + 10 * n_sb)
        + 4 * slots * ff
        + q8_act_bytes(slots, ff)
        + 4 * slots * n_embd
}

/// The tier's chunk scratch for blocks of rows of `n_embd`, `n_used` slots a
/// token, through experts of `ff`, cut in chunks of `chunk` tokens: one run of
/// the card leg's buffers (`bloomery_gpu::card_leg::LegBufs`, the GLM-5.3-Flash
/// tier), in place of the tile path's scratch. Per `LegBufs` field: the q8_1
/// of a chunk's rows (`act_x`, a column a token), its gate·up rows (`h`, `ff`
/// values a slot), the q8_1 of its slots' columns (`act_h`, a column a slot
/// of `ff`) and its down rows (`down`, `n_embd` values a slot).
#[must_use]
pub const fn tier_chunk_scratch_bytes(n_embd: u64, ff: u64, n_used: u64, chunk: u64) -> u64 {
    let slots = chunk * n_used;
    q8_act_bytes(chunk, n_embd) + 4 * slots * ff + q8_act_bytes(slots, ff) + 4 * slots * n_embd
}

/// The host's part of the tier's prompt-batch service: per exchange set, the
/// rows the tier hands back and the places it reads.
#[must_use]
pub const fn tier_batch_host_bytes(n_embd: u64, n_used: u64, cols: u64) -> u64 {
    let slots = cols * n_used;
    2 * (4 * slots * n_embd + 4 * slots)
}

/// The expert tier's prompt-batch bytes for blocks of up to `cols` tokens of
/// rows of `n_embd`, routed to `n_used` experts of `ff` rows each.
#[must_use]
pub const fn tier_batch_bytes(n_embd: u64, ff: u64, n_used: u64, cols: u64) -> TierBatchBytes {
    TierBatchBytes {
        staging: tier_batch_staging_bytes(n_embd, n_used, cols),
        scratch: tier_block_scratch_bytes(n_embd, ff, cols * n_used),
        host: tier_batch_host_bytes(n_embd, n_used, cols),
    }
}

/// Plan (b′): the A6000 runs all `layers` and the head as in [`plan_a`];
/// the 3090 is an expert tier beside the host, holding each layer's next ids
/// after the stage's prefix, with `draft_bytes` — the DSpark draft's resident bytes
/// (`model::arch::dspark::card_bytes`), when the draft lives there — as a
/// named reserve, and the tier's prompt-batch service `batch`
/// ([`tier_batch_bytes`]) as named reserves on the tier and the host.
/// It is [`plan_tiers`] of the A6000 over the one tier card, the 3090.
#[must_use]
pub fn plan_bp(layers: usize, draft_bytes: Option<u64>, batch: TierBatchBytes) -> Machine {
    let draft = draft_bytes.map(|bytes| TierDraft { on: 0, bytes });
    tiered(layers, A6000, &[RTX_3090], draft, batch, batch.host)
}

/// A draft model's resident bytes on one of plan (b′)'s tier cards: the
/// tier's index in the plan's tier list, and the bytes its plan reserves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierDraft {
    pub on: usize,
    pub bytes: u64,
}

/// Plan (b′) over `tiers`: `stage` runs all `layers` and the head as in
/// [`plan_on`]; each tier card, in order, is an expert tier beside the host
/// holding each layer's next ids after the cards before it
/// ([`super::Machine::tiers`]), with the prompt-batch service `batch`
/// ([`tier_batch_bytes`]) reserved on it, and `draft` on the tier it names;
/// the host reserves the batch's rows once per tier. A card listed twice
/// ([`PlacementError::DeviceTwice`]) and a draft on a tier past the list
/// ([`PlacementError::DraftOffPlan`]) are refused ([`check_tiers`]). With no
/// tier it is [`plan_on`].
pub fn plan_tiers(
    layers: usize,
    stage: CardSpec,
    tiers: &[CardSpec],
    draft: Option<TierDraft>,
    batch: TierBatchBytes,
) -> Result<Machine, PlacementError> {
    let host_rows = check_tiers(stage, tiers, draft, batch)?;
    Ok(tiered(layers, stage, tiers, draft, batch, host_rows))
}

/// What [`plan_tiers`] refuses of its cards, before any layer count: a card
/// listed twice — one device ([`CardSpec::same_device`]), whose usable bytes
/// the plan would count once a listing — and a draft on a tier past the list. Returns the host's
/// prompt-batch rows, one set a tier.
pub fn check_tiers(
    stage: CardSpec,
    tiers: &[CardSpec],
    draft: Option<TierDraft>,
    batch: TierBatchBytes,
) -> Result<u64, PlacementError> {
    let all: Vec<CardSpec> = std::iter::once(stage)
        .chain(tiers.iter().copied())
        .collect();
    for (i, spec) in all.iter().enumerate() {
        let n = all.iter().filter(|s| s.same_device(spec)).count();
        if n > 1 && !all[..i].iter().any(|s| s.same_device(spec)) {
            let usable = spec.usable_bytes();
            return Err(PlacementError::DeviceTwice {
                card: spec.name.to_string(),
                cards: n,
                budgets: u64::try_from(n)
                    .ok()
                    .and_then(|n| usable.checked_mul(n))
                    .unwrap_or(u64::MAX),
                usable,
            });
        }
    }
    if let Some(d) = draft.filter(|d| d.on >= tiers.len()) {
        return Err(PlacementError::DraftOffPlan {
            on: d.on,
            tiers: tiers.len(),
        });
    }
    u64::try_from(tiers.len())
        .ok()
        .and_then(|k| batch.host.checked_mul(k))
        .ok_or(PlacementError::TierBatchHost {
            tiers: tiers.len(),
            bytes: batch.host,
        })
}

/// [`plan_tiers`]'s machine once [`check_tiers`] passed, `host_rows` its
/// sum.
fn tiered(
    layers: usize,
    stage: CardSpec,
    tiers: &[CardSpec],
    draft: Option<TierDraft>,
    batch: TierBatchBytes,
    host_rows: u64,
) -> Machine {
    let tiers: Vec<Card> = tiers
        .iter()
        .enumerate()
        .map(|(i, &spec)| {
            let mut t = tier(spec);
            t.reserves.extend(
                draft
                    .filter(|d| d.on == i)
                    .map(|d| (DRAFT_RESERVE.to_string(), d.bytes)),
            );
            t.reserves
                .push((TIER_BATCH_RESERVE.to_string(), batch.card()));
            t
        })
        .collect();
    let mut h = host_of(&stage);
    if !tiers.is_empty() {
        h.reserves
            .push((TIER_BATCH_HOST_RESERVE.to_string(), host_rows));
    }
    Machine {
        cards: vec![card(stage, 0..layers, true)],
        tiers,
        host: h,
        unified: unified_of(&stage),
    }
}

/// The gate placement: the 3090 runs all `layers` and the head, so the V4.1
/// body gates run on the gate card while the A6000 takes timing runs. Its
/// expert prefixes are the largest the card's budget allows — the expert
/// rule of [`super::plan`], on the same usable − KV − context − scratch −
/// margin arithmetic as [`plan_a`]'s card.
#[must_use]
pub fn plan_gate(layers: usize) -> Machine {
    plan_on(RTX_3090, layers)
}

/// The card spec a plan's card is named after.
#[must_use]
pub fn spec_of(card: &Card) -> Option<CardSpec> {
    CARDS.into_iter().find(|s| s.name == card.name)
}

/// What a placed load needs of the host's memory, which the host's
/// available bytes must cover ([`HostNeed::check`]) — `MemAvailable`, or
/// the smaller room under a cgroup v2 limit ([`host_room`]): the plan's
/// host experts and tables, the
/// cards' ring shadows, the host's reserves, `extra` (a residency churn
/// pool the host set also holds), the NVMe expert tier's RAM arena
/// ([`HostNeed::arena`]) and on a unified machine the cards' share of the
/// pool ([`HostNeed::cards`]), less the reserve named [`OS_RESERVE`] —
/// `MemAvailable` already leaves out what the OS and every other process
/// hold. The page cache the host set will reuse is inside `MemAvailable`
/// already, as reclaimable file pages, so no term adds it.
///
/// The one owner of a plan's host bound: the load holds the need to the
/// host's reading, and the plan its own terms to its machine's figure
/// ([`HostNeed::machine_room`]: `Plan::violations`' host check and a plan's
/// host headroom); the NVMe tier's arena and window, sized to the reading
/// the split took, are held to that reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostNeed {
    pub experts: u64,
    pub tables: u64,
    pub shadows: u64,
    pub reserves: u64,
    pub extra: u64,
    /// The plan's NVMe expert tier arena (`HostTotals::nvme_arena_bytes`):
    /// the anonymous bytes the tier's slots fill beside these terms, 0
    /// without one. [`HostNeed::bytes`] counts it;
    /// [`HostNeed::plan_bytes`] leaves it to the split dial that shapes it.
    pub arena: u64,
    /// On a unified machine, what the cards take from the pool the host's
    /// room is read from (`HostTotals::pool_card_bytes`); 0 where each card
    /// has memory of its own.
    pub cards: u64,
    pub os: u64,
    /// The plan was made under a card budget.
    pub card_budget: bool,
}

impl HostNeed {
    /// `plan`'s host terms, with `extra` bytes the host set holds beside
    /// them.
    #[must_use]
    pub fn of(plan: &Plan<'_>, extra: u64) -> HostNeed {
        HostNeed::of_totals(
            &plan.machine.host,
            &plan.host,
            plan.card_budget.is_some(),
            extra,
        )
    }

    /// The host terms `totals` hold on `host`, a plan made under a card
    /// budget or not, with `extra` beside them: [`HostNeed::of`]'s, before
    /// the plan that carries them is built.
    #[must_use]
    pub fn of_totals(host: &Host, totals: &HostTotals, card_budget: bool, extra: u64) -> HostNeed {
        HostNeed {
            experts: totals.expert_bytes,
            tables: totals.table_bytes,
            shadows: totals.shadow_bytes,
            reserves: totals.reserve_bytes,
            extra,
            arena: totals.nvme_arena_bytes,
            cards: totals.pool_card_bytes,
            os: os_reserve(host),
            card_budget,
        }
    }

    /// The room `host`'s own figure leaves a load: its usable bytes less
    /// the reserve named [`OS_RESERVE`], which a reading of the host leaves
    /// out already — the room a plan is held to before any reading
    /// ([`HostNeed::check`] in `Plan::violations`, and the plan's host
    /// headroom); [`PLAN_ROOM`] on this workstation's [`host`].
    #[must_use]
    pub fn machine_room(host: &Host) -> u64 {
        host.usable_bytes.saturating_sub(os_reserve(host))
    }

    /// The bytes `MemAvailable` must cover: the plan's host terms, `extra`,
    /// the arena and the cards' share of a unified pool — what a placed load
    /// of this plan holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        (self.experts
            + self.tables
            + self.shadows
            + self.reserves
            + self.extra
            + self.arena
            + self.cards)
            .saturating_sub(self.os)
    }

    /// [`HostNeed::bytes`] with the arena out: the plan's own host terms,
    /// `extra` and the cards' share of a unified pool — the sum the
    /// plan-time room arithmetic of `expert_nvme_tier` shapes, the arena
    /// being its own dial beside it (the room that arithmetic splits has
    /// already lost the arena).
    #[must_use]
    pub fn plan_bytes(&self) -> u64 {
        (self.experts + self.tables + self.shadows + self.reserves + self.extra + self.cards)
            .saturating_sub(self.os)
    }

    /// `available` bytes (the host's reading, [`host_room`]'s) cover the
    /// need, or the refusal that names both, each term and what lowers the
    /// need.
    pub fn check(self, available: u64) -> Result<(), HostShort> {
        if self.bytes() <= available {
            Ok(())
        } else {
            Err(HostShort {
                need: self,
                available,
                floor: None,
            })
        }
    }

    /// The load's check with an elastic arena: `available` bytes cover the
    /// need and the arena is granted as planned, or the arena shrinks by the
    /// shortfall to a grant of at least `least` (the arena's floor,
    /// `ArenaFloor::bytes`), or the refusal [`HostNeed::check`] would give,
    /// naming `least` when the plan has an arena that could not shrink past
    /// it. A plan with no arena is held exactly as `check` holds it. The one
    /// owner of the arena's shrink: the caller reads the grant, never
    /// recomputes it.
    pub fn grant(self, available: u64, least: u64) -> Result<ArenaGrant, HostShort> {
        let planned = self.arena;
        let Some(short) = self.bytes().checked_sub(available).filter(|&s| s > 0) else {
            return Ok(ArenaGrant::whole(planned));
        };
        match planned.checked_sub(short) {
            Some(granted) if planned > 0 && granted >= least => Ok(ArenaGrant { planned, granted }),
            _ => Err(HostShort {
                need: self,
                available,
                floor: (planned > 0).then_some(least),
            }),
        }
    }
}

/// The NVMe tier arena a load builds: the bytes its plan reserved and the
/// bytes the host's reading let the load take ([`HostNeed::grant`]). They
/// differ only when the reading fell short of the plan's need by less than
/// the arena's room above its floor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaGrant {
    pub planned: u64,
    pub granted: u64,
}

impl ArenaGrant {
    /// A load that takes the arena as its plan reserved it.
    #[must_use]
    pub fn whole(planned: u64) -> ArenaGrant {
        ArenaGrant {
            planned,
            granted: planned,
        }
    }

    /// Bytes the arena shrank by at load.
    #[must_use]
    pub fn shrunk(&self) -> u64 {
        self.planned - self.granted
    }
}

/// The bytes of `host`'s reserves named [`OS_RESERVE`].
fn os_reserve(host: &Host) -> u64 {
    host.reserves
        .iter()
        .filter(|(name, _)| name == OS_RESERVE)
        .map(|(_, b)| b)
        .sum()
}

/// A placed load whose host need passes the host's available bytes
/// ([`HostNeed::check`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostShort {
    pub need: HostNeed,
    pub available: u64,
    /// On a refusal of [`HostNeed::grant`] for a plan with an arena: the
    /// floor the arena could not shrink past. `None` from
    /// [`HostNeed::check`], which never shrinks it.
    pub floor: Option<u64>,
}

impl fmt::Display for HostShort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = &self.need;
        let need = n.bytes();
        write!(
            f,
            "the host has {} B available (MemAvailable, or the smaller room under a cgroup v2 \
             limit — host_room's reading), under the {need} B \
             this load needs by {} B: host experts {} B + tables {} B + ring shadows {} B + \
             reserves {} B",
            self.available,
            need.saturating_sub(self.available),
            n.experts,
            n.tables,
            n.shadows,
            n.reserves
        )?;
        if n.extra > 0 {
            write!(f, " + residency churn pool {} B", n.extra)?;
        }
        if n.arena > 0 {
            write!(f, " + NVMe tier arena {} B", n.arena)?;
            if let Some(floor) = self.floor {
                write!(
                    f,
                    " (the load shrinks the arena by the shortfall, and the shortfall leaves it \
                     under its floor {floor} B)"
                )?;
            }
        }
        if n.cards > 0 {
            write!(f, " + the cards' share of the unified pool {} B", n.cards)?;
        }
        write!(
            f,
            " − the OS reserve {} B, which MemAvailable leaves out already. Free host memory, or \
             load by a placement whose cards hold more of the model (V4.1: --place bp holds more \
             than a, and a more than gate; Qwen3.8: BLOOMERY_QWEN38_EXPERTS=card)",
            n.os
        )?;
        if n.extra > 0 {
            write!(f, "; BLOOMERY_RESIDENCY=off drops the churn pool")?;
        }
        if n.arena > 0 {
            write!(
                f,
                "; BLOOMERY_NVTIER_BYTES=0 plans the split with no arena (the mapping reads)"
            )?;
        }
        if n.card_budget {
            write!(
                f,
                "; BLOOMERY_CARD_BUDGET moves bytes from the cards to the host, so unset it"
            )?;
        }
        if n.cards > 0 {
            write!(
                f,
                "; the cards take from the same pool, so a smaller --ctx or quant lowers their share"
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for HostShort {}

/// `MemAvailable` of a `/proc/meminfo` text, in bytes. A text with no such
/// line (a kernel before 3.14), or one whose value is not a whole number of
/// kB, is refused by name.
pub fn mem_available(meminfo: &str) -> Result<u64, String> {
    let line = meminfo
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .ok_or("the meminfo text has no MemAvailable line")?;
    line.trim()
        .strip_suffix("kB")
        .map(str::trim_end)
        .filter(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|d| d.parse::<u64>().ok())
        .and_then(|kb| kb.checked_mul(1024))
        .ok_or_else(|| {
            format!("the meminfo text's MemAvailable is {line:?}, not a whole number of kB")
        })
}

/// Which reading of the host's available bytes decided ([`host_room`]):
/// `MemAvailable`, or the room under a cgroup v2 memory limit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostRead {
    /// `/proc/meminfo`'s `MemAvailable`.
    MemAvailable,
    /// The room under the cgroup v2 memory limit at `path`, as
    /// [`cgroup_v2_path`] names it under `/sys/fs/cgroup`.
    Cgroup { path: String },
    /// A room the caller gave instead of reading this machine (a gate that
    /// plans an arm by the room it names).
    Given,
}

impl fmt::Display for HostRead {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostRead::MemAvailable => write!(f, "MemAvailable in /proc/meminfo"),
            HostRead::Cgroup { path } => write!(
                f,
                "the room under the cgroup v2 memory limit at /sys/fs/cgroup{path} \
                 (memory.max − memory.current + active_file + inactive_file)"
            ),
            HostRead::Given => write!(f, "a room the caller gave, not read from this machine"),
        }
    }
}

impl HostRead {
    /// One word for a `key=value` line: `MemAvailable`, `cgroup:<path>`, or
    /// `given`.
    #[must_use]
    pub fn word(&self) -> String {
        match self {
            HostRead::MemAvailable => "MemAvailable".to_string(),
            HostRead::Cgroup { path } => format!("cgroup:{path}"),
            HostRead::Given => "given".to_string(),
        }
    }
}

/// The host bytes a load's own start may take between the reading its plan
/// is made at and the load's check ([`HostNeed::grant`]): a chosen bound,
/// not a derived one. A plan made at a read room keeps it ([`plan_room`]),
/// so a split that fills its room exactly still passes the check at the
/// load.
pub const HOST_DRIFT: u64 = 512 * MIB;

/// The room a plan is made at, for the room `room` that [`host_room`] (or a
/// caller) gave: a [`HostRead::Given`] room as given, a read room less
/// [`HOST_DRIFT`] (to 0 at most). The one place a read room becomes a
/// plan's room.
#[must_use]
pub fn plan_room(room: &(u64, HostRead)) -> u64 {
    match room.1 {
        HostRead::Given => room.0,
        HostRead::MemAvailable | HostRead::Cgroup { .. } => room.0.saturating_sub(HOST_DRIFT),
    }
}

/// The process's own cgroup v2 path of a `/proc/self/cgroup` text: the
/// `0::` line names it under `/sys/fs/cgroup`. `Ok(None)` when the text has
/// no `0::` line — the unified hierarchy holds this process nowhere, so no
/// `memory.max` of it binds (a cgroup v1 system among them; v1 limits are
/// not read). Two `0::` lines, or one whose path is not absolute or climbs
/// with a `..` component (the path joins `/sys/fs/cgroup`), are refused by
/// name.
pub fn cgroup_v2_path(text: &str) -> Result<Option<String>, String> {
    let mut own = None;
    for line in text.lines() {
        let Some(path) = line.strip_prefix("0::") else {
            continue;
        };
        if own.is_some() {
            return Err("the /proc/self/cgroup text has two unified-hierarchy lines".to_owned());
        }
        if !path.starts_with('/') {
            return Err(format!(
                "the /proc/self/cgroup text's unified-hierarchy path {path:?} is not absolute"
            ));
        }
        if path.split('/').any(|c| c == "..") {
            return Err(format!(
                "the /proc/self/cgroup text's unified-hierarchy path {path:?} climbs with .."
            ));
        }
        own = Some(path.to_owned());
    }
    Ok(own)
}

/// One cgroup v2 level above the process, as its files' texts: its `path`
/// under `/sys/fs/cgroup` ([`cgroup_v2_path`]'s), and that cgroup's
/// `memory.max`, `memory.current` and `memory.stat`.
#[derive(Clone, Copy, Debug)]
pub struct CgroupLevel<'a> {
    pub path: &'a str,
    pub max: &'a str,
    pub current: &'a str,
    pub stat: &'a str,
}

/// One cgroup v2 `memory.max` text: `Ok(None)` for `max`, the level sets no
/// memory limit; the limit's bytes otherwise. A value that is neither is
/// refused by name.
fn memory_max(text: &str) -> Result<Option<u64>, String> {
    let t = text.trim();
    if t == "max" {
        return Ok(None);
    }
    t.parse::<u64>().map(Some).map_err(|_| {
        format!("the cgroup's memory.max is {t:?}, neither `max` nor a whole number of bytes")
    })
}

/// One cgroup v2 counter text (`memory.current`): a whole number of bytes,
/// else refused by name.
fn memory_bytes(what: &str, text: &str) -> Result<u64, String> {
    text.trim().parse::<u64>().map_err(|_| {
        format!(
            "the cgroup's {what} is {:?}, not a whole number of bytes",
            text.trim()
        )
    })
}

/// The file pages a cgroup v2 `memory.stat` text charges: `active_file` +
/// `inactive_file`, in bytes. `memory.current` counts the page cache, and
/// the model's shards are page cache once populated, while `MemAvailable`
/// counts those LRU file pages as available — so a limit's room gives them
/// back, or the host set is subtracted twice (once in `memory.current`,
/// once as the plan's [`HostNeed`]) and a plan that fits is refused after
/// any populate. Locked and unevictable pages sit in neither line and stay
/// charged. A text missing either line, or holding a value that is not a
/// whole number of bytes, is refused by name.
fn memory_file_pages(stat: &str) -> Result<u64, String> {
    let mut active = None;
    let mut inactive = None;
    for line in stat.lines() {
        if let Some(v) = line.strip_prefix("active_file ") {
            active = Some(v.trim());
        } else if let Some(v) = line.strip_prefix("inactive_file ") {
            inactive = Some(v.trim());
        }
    }
    let page = |what: &str, v: Option<&str>| -> Result<u64, String> {
        v.ok_or_else(|| format!("the cgroup's memory.stat has no {what} line"))?
            .parse::<u64>()
            .map_err(|_| format!("the cgroup's memory.stat {what} is not a whole number of bytes"))
    };
    page("active_file", active)?
        .checked_add(page("inactive_file", inactive)?)
        .ok_or_else(|| "the cgroup's memory.stat file pages pass u64".to_owned())
}

/// The room under one cgroup v2 level's memory limit, from its files' texts
/// ([`CgroupLevel`]): `memory.max − memory.current + active_file +
/// inactive_file` ([`memory_file_pages`] says why the file pages come
/// back). `Ok(None)` when the level's `memory.max` is `max`: it binds
/// nothing. A usage over the limit, or a room that passes u64, is refused
/// by name.
pub fn cgroup_room(level: &CgroupLevel<'_>) -> Result<Option<u64>, String> {
    let Some(max) = memory_max(level.max)? else {
        return Ok(None);
    };
    let current = memory_bytes("memory.current", level.current)?;
    let pages = memory_file_pages(level.stat)?;
    max.checked_sub(current)
        .ok_or_else(|| {
            format!(
                "the cgroup at {} charges memory.current {current} B over its memory.max {max} B",
                level.path
            )
        })?
        .checked_add(pages)
        .ok_or_else(|| {
            format!(
                "the cgroup at {} sums its room past u64 (memory.max {max} B, file pages \
                 {pages} B)",
                level.path
            )
        })
        .map(Some)
}

/// The host's available bytes of a `/proc/meminfo` text and the cgroup v2
/// levels above the process (its own cgroup first, every ancestor after):
/// the smaller of `MemAvailable` and every level's room ([`cgroup_room`])
/// — inside a container (`docker run --memory`), a systemd scope
/// (`MemoryMax`) or a pod, `/proc/meminfo` still shows the whole machine,
/// so a limit's room is the reading that binds — with the reading that
/// decided ([`HostRead`]), for every refusal and line that prints the
/// bytes to name. A level that sets no limit does not bind; no levels is
/// `MemAvailable` alone; a tie leaves `MemAvailable` the named one.
pub fn host_room(meminfo: &str, levels: &[CgroupLevel<'_>]) -> Result<(u64, HostRead), String> {
    let mut best = (mem_available(meminfo)?, HostRead::MemAvailable);
    for level in levels {
        if let Some(room) = cgroup_room(level)?
            && room < best.0
        {
            best = (
                room,
                HostRead::Cgroup {
                    path: level.path.to_owned(),
                },
            );
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::{
        A6000, ArenaGrant, CONTEXT, Card, CardSpec, CgroupLevel, DRAFT_RESERVE, DeviceInfo,
        GRANULE, HOST_DRIFT, Host, HostNeed, HostRead, MARGIN, Machine, PLAN_ROOM, PlacementError,
        RTX_3090, SCRATCH, TIER_BATCH_HOST_RESERVE, TIER_BATCH_RESERVE, TierBatchBytes, TierDraft,
        UnifiedPool, card, cgroup_room, cgroup_v2_path, host, host_of, host_room, mem_available,
        placed_on, plan_a, plan_bp, plan_gate, plan_on, plan_room, plan_tiers, spec_of_device,
        tier, tier_batch_bytes, unified_of,
    };

    /// One plan room: `tools/ref/plan-room.sh --room`, the room
    /// `records-refresh` makes V4.1's plans at, prints [`PLAN_ROOM`], and
    /// the workstation's host leaves a load that room
    /// ([`HostNeed::machine_room`]), the room a plan of it is held to.
    #[test]
    fn the_plan_room_is_one_const() {
        let script = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/ref/plan-room.sh");
        let out = std::process::Command::new("bash")
            .args([script, "--room"])
            .output()
            .expect("bash runs plan-room.sh");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "plan-room.sh --room: {}: {stderr}",
            out.status
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout).trim(),
            PLAN_ROOM.to_string(),
            "plan-room.sh --room ({stderr})"
        );
        assert_eq!(HostNeed::machine_room(&host()), PLAN_ROOM);
    }

    /// Plan (a), the gate plan and plan (b′) as their own bodies built them
    /// before [`plan_tiers`]: the references the refactored plans must equal
    /// field for field.
    fn base_one(spec: CardSpec, layers: usize) -> Machine {
        Machine {
            cards: vec![card(spec, 0..layers, true)],
            tiers: Vec::new(),
            host: host(),
            unified: None,
        }
    }

    fn base_bp(layers: usize, draft_bytes: Option<u64>, batch: TierBatchBytes) -> Machine {
        let mut t = tier(RTX_3090);
        t.reserves
            .extend(draft_bytes.map(|b| (DRAFT_RESERVE.to_string(), b)));
        t.reserves
            .push((TIER_BATCH_RESERVE.to_string(), batch.card()));
        let mut h = host();
        h.reserves
            .push((TIER_BATCH_HOST_RESERVE.to_string(), batch.host));
        Machine {
            cards: vec![card(A6000, 0..layers, true)],
            tiers: vec![t],
            host: h,
            unified: None,
        }
    }

    /// Every layer count and batch the plans are built with (V4.1's 43
    /// layers and its tier batch, GLM-5.3-Flash's and Qwen3.8's shapes, and
    /// edge counts), with and without a draft: `plan_a`, `plan_gate` and
    /// `plan_bp` equal the bodies they had, and `plan_tiers` of the same
    /// cards equals each.
    #[test]
    fn plan_tiers_is_plan_bp_a_and_gate() {
        let batches = [
            tier_batch_bytes(5120, 2304, 6, 512),
            tier_batch_bytes(4096, 1536, 8, 512),
            tier_batch_bytes(2048, 768, 8, 4096),
            TierBatchBytes {
                staging: 1,
                scratch: 2,
                host: 3,
            },
        ];
        for layers in [0, 1, 20, 43, 47, 48, 61, 78] {
            assert_eq!(plan_a(layers), base_one(A6000, layers));
            assert_eq!(plan_gate(layers), base_one(RTX_3090, layers));
            for batch in batches {
                let tiers = plan_tiers(layers, A6000, &[], None, batch).expect("no tier");
                assert_eq!(tiers, plan_a(layers));
                let tiers = plan_tiers(layers, RTX_3090, &[], None, batch).expect("no tier");
                assert_eq!(tiers, plan_gate(layers));
                for draft in [None, Some(1), Some(1_843_200_000), Some(u64::MAX)] {
                    let base = base_bp(layers, draft, batch);
                    assert_eq!(plan_bp(layers, draft, batch), base, "{layers} {draft:?}");
                    let on = draft.map(|bytes| TierDraft { on: 0, bytes });
                    let tiers = plan_tiers(layers, A6000, &[RTX_3090], on, batch)
                        .unwrap_or_else(|e| panic!("{layers} {draft:?}: {e}"));
                    assert_eq!(tiers, base, "{layers} {draft:?}");
                }
            }
        }
    }

    /// `plan_tiers` refuses a card listed twice and a draft on a tier past
    /// the list by name; the 3090 as the stage with the A6000 its tier plans.
    #[test]
    fn plan_tiers_refuses_by_name() {
        let batch = tier_batch_bytes(5120, 2304, 6, 512);
        let draft = |on| Some(TierDraft { on, bytes: 7 });
        match plan_tiers(43, A6000, &[A6000], None, batch) {
            Err(PlacementError::DeviceTwice {
                card,
                cards,
                budgets,
                usable,
            }) => {
                assert_eq!((card.as_str(), cards), (A6000.name, 2));
                assert_eq!(
                    (budgets, usable),
                    (2 * A6000.usable_bytes(), A6000.usable_bytes())
                );
            }
            other => panic!("{other:?}, not DeviceTwice"),
        }
        assert!(matches!(
            plan_tiers(43, A6000, &[RTX_3090, RTX_3090], None, batch),
            Err(PlacementError::DeviceTwice { cards: 2, .. })
        ));
        match plan_tiers(43, A6000, &[RTX_3090], draft(1), batch) {
            Err(e @ PlacementError::DraftOffPlan { on: 1, tiers: 1 }) => assert_eq!(
                e.to_string(),
                "the draft's reserve is on tier 1, and the plan has 1 tier cards: the draft's \
                 card is outside the plan"
            ),
            other => panic!("{other:?}, not DraftOffPlan"),
        }
        assert!(matches!(
            plan_tiers(43, A6000, &[], draft(0), batch),
            Err(PlacementError::DraftOffPlan { on: 0, tiers: 0 })
        ));
        let swapped = plan_tiers(43, RTX_3090, &[A6000], draft(0), batch).expect("3090+a6000");
        assert_eq!(swapped.cards[0].name, RTX_3090.name);
        assert_eq!(swapped.tiers.len(), 1);
        assert_eq!(swapped.tiers[0].name, A6000.name);
        assert_eq!(swapped.tiers[0].reserve_bytes(), 7 + batch.card());
    }

    /// The host reserves the prompt batch's rows once per tier, in one row.
    #[test]
    fn plan_tiers_sums_the_host_rows_per_tier() {
        // Two tier specs of other names: the workstation has two cards, and
        // the rule is the sum over however many tiers a plan lists.
        let t1 = CardSpec {
            name: "t1",
            ..RTX_3090
        };
        let t2 = CardSpec {
            name: "t2",
            ..RTX_3090
        };
        let batch = TierBatchBytes {
            staging: 10,
            scratch: 5,
            host: 100,
        };
        let m = plan_tiers(4, A6000, &[t1, t2], None, batch).expect("two tiers");
        let rows: Vec<&(String, u64)> = m
            .host
            .reserves
            .iter()
            .filter(|(n, _)| n == TIER_BATCH_HOST_RESERVE)
            .collect();
        assert_eq!(rows, vec![&(TIER_BATCH_HOST_RESERVE.to_string(), 200)]);
        assert!(m.tiers.iter().all(|t| t.reserve_bytes() == 15));
        let huge = TierBatchBytes {
            host: u64::MAX / 2 + 1,
            ..batch
        };
        assert!(matches!(
            plan_tiers(4, A6000, &[t1, t2], None, huge),
            Err(PlacementError::TierBatchHost { tiers: 2, .. })
        ));
    }

    /// An integrated card's machine plans against one pool: the census's
    /// reading of it is the pool ([`super::unified_of`]) and the host's
    /// figure, with no OS reserve ([`super::host_of`]), and a whole load sizes
    /// its card with the host's reserves among the card's own
    /// ([`Machine::whole_card`]). The same device with memory of its own
    /// plans beside this workstation's host, its whole load on its card.
    #[test]
    fn an_integrated_card_plans_against_one_pool() {
        const ROOM: u64 = 120 << 30;
        let device = |integrated| DeviceInfo {
            ordinal: 0,
            name: "NVIDIA GB10".to_string(),
            total_bytes: 128 << 30,
            free_bytes: ROOM,
            integrated,
            uuid: [0x10; 16],
            pci_bus: "000f:01:00.0".to_string(),
            held_by: None,
        };
        let pinned = ("pinned".to_string(), 1 << 30);
        for integrated in [true, false] {
            let mut m = plan_on(spec_of_device(&device(integrated)), 4);
            let (pool, figure) = if integrated {
                let figure = Host {
                    usable_bytes: ROOM,
                    reserves: Vec::new(),
                };
                (Some(UnifiedPool { bytes: ROOM }), figure)
            } else {
                (None, host())
            };
            assert_eq!(
                (m.unified, &m.host),
                (pool, &figure),
                "integrated {integrated}"
            );
            m.host.reserves.push(pinned.clone());
            let mut whole = m.cards[0].clone();
            if integrated {
                whole.reserves.push(pinned.clone());
            }
            assert_eq!(
                m.whole_card(),
                Some(whole),
                "integrated {integrated}: the whole load's card"
            );
        }
    }

    /// A placed load's machine holds every layer, the head and the token
    /// embedding on its one card at the program's scratch, with this
    /// machine's context, margin and granule and the spec's host.
    #[test]
    fn a_placed_load_holds_the_whole_model_at_its_scratch() {
        for spec in [RTX_3090, A6000] {
            let m = placed_on(spec, 40, SCRATCH + 7);
            let want = Card {
                name: spec.name.to_string(),
                device: spec.device,
                usable_bytes: spec.usable_bytes(),
                context_bytes: CONTEXT,
                scratch_bytes: SCRATCH + 7,
                margin_bytes: MARGIN,
                granule_bytes: GRANULE,
                free_bytes: spec.free_bytes,
                held_by: spec.held_by,
                layers: 0..40,
                head: true,
                token_embedding: true,
                reserves: Vec::new(),
            };
            assert_eq!(m.cards, vec![want], "{}", spec.name);
            assert!(m.tiers.is_empty(), "{}", spec.name);
            assert_eq!(
                (&m.host, m.unified),
                (&host_of(&spec), unified_of(&spec)),
                "{}",
                spec.name
            );
        }
    }

    /// An integrated spec with no free reading is refused by name, not
    /// planned as a card of its own beside this workstation's host.
    #[test]
    #[should_panic(
        expected = "an integrated card with no free reading has no pool to plan against"
    )]
    fn an_integrated_card_with_no_reading_has_no_pool() {
        let _ = unified_of(&CardSpec {
            integrated: true,
            ..A6000
        });
    }

    /// V4.1's tier batch at the host union's 512 columns: the allocation
    /// sizes of the tier's staging and tile scratch, summed by hand.
    #[test]
    fn v41_tier_batch_bytes() {
        let b = tier_batch_bytes(5120, 2304, 6, 512);
        assert_eq!(b.staging, 10_485_760 + 12_288 + 8_273_920 + 62_914_560);
        assert_eq!(
            b.scratch,
            12_288 + 4_100 + 5_124 + 15_728_640 + 491_520 + 28_311_552 + 26_271_744
        );
        assert_eq!(b.card(), 152_511_496);
        assert_eq!(b.host, 2 * (62_914_560 + 12_288));
    }

    /// `MemAvailable` in bytes from a `/proc/meminfo` text; a text without
    /// the line, or with a value that is not a whole number of kB, is refused.
    #[test]
    fn mem_available_reads_the_line_in_kb() {
        let text = "MemTotal:       263741212 kB\nMemFree:         1234567 kB\n\
                    MemAvailable:   257300180 kB\nBuffers:            1234 kB\n";
        assert_eq!(mem_available(text), Ok(257_300_180 * 1024));
        assert!(mem_available("MemTotal: 1 kB\n").is_err());
        assert!(mem_available("MemAvailable:   12x kB\n").is_err());
        assert!(mem_available("MemAvailable:   12 MB\n").is_err());
        assert!(mem_available("MemAvailable:    kB\n").is_err());
    }

    /// One level's texts, its files as the kernel writes them: `pages` its
    /// `active_file` and `inactive_file` bytes, held beside the texts so
    /// [`CgroupLevel`] can borrow them.
    fn level(path: &str, max: &str, current: &str, pages: (u64, u64)) -> OwnedLevel {
        OwnedLevel {
            path: path.to_owned(),
            max: max.to_owned(),
            current: current.to_owned(),
            stat: format!(
                "anon 1048576\nactive_file {}\ninactive_file {}\nslab 2048\n",
                pages.0, pages.1
            ),
        }
    }

    /// A [`CgroupLevel`]'s own texts ([`level`]'s).
    struct OwnedLevel {
        path: String,
        max: String,
        current: String,
        stat: String,
    }

    impl OwnedLevel {
        fn as_level(&self) -> CgroupLevel<'_> {
            CgroupLevel {
                path: &self.path,
                max: &self.max,
                current: &self.current,
                stat: &self.stat,
            }
        }
    }

    /// Every level's [`CgroupLevel`], borrowing them ([`level`]'s).
    fn as_levels(levels: &[OwnedLevel]) -> Vec<CgroupLevel<'_>> {
        levels.iter().map(OwnedLevel::as_level).collect()
    }

    /// The unified-hierarchy line of a `/proc/self/cgroup` text: the path it
    /// names; no `0::` line (cgroup v1 only, or a process outside the
    /// unified hierarchy) is `None`; a second line, a path that is not
    /// absolute and one climbing with `..` are refused by name.
    #[test]
    fn cgroup_v2_path_reads_the_unified_line() {
        let v2 = "12:devices:/user.slice\n0::/user.slice/user-1000.slice/session-3.scope\n";
        assert_eq!(
            cgroup_v2_path(v2),
            Ok(Some(
                "/user.slice/user-1000.slice/session-3.scope".to_owned()
            ))
        );
        assert_eq!(cgroup_v2_path("0::/\n"), Ok(Some("/".to_owned())));
        let v1 = "11:blkio:/init.scope\n12:devices:/user.slice\n";
        assert_eq!(cgroup_v2_path(v1), Ok(None));
        assert_eq!(cgroup_v2_path(""), Ok(None));
        assert!(cgroup_v2_path("0::user.slice\n").is_err());
        assert!(cgroup_v2_path("0::/a/../b\n").is_err());
        assert!(cgroup_v2_path("0::/a\n0::/b\n").is_err());
    }

    /// One cgroup's room under its v2 memory limit: `max` binds nothing;
    /// the room gives the file pages `memory.current` charges back; a usage
    /// over the limit, and every malformed file, are refused by name.
    #[test]
    fn cgroup_room_reads_the_limit_and_gives_file_pages_back() {
        let gib = 1 << 30;
        let n = |b: u64| format!("{}\n", b);
        // A limit with usage: 61 − 20 + 12 (file) = 53 GiB.
        assert_eq!(
            cgroup_room(&level("/", &n(61 * gib), &n(20 * gib), (6 * gib, 6 * gib)).as_level()),
            Ok(Some(53 * gib))
        );
        // `max` sets no limit, whatever the usage says.
        assert_eq!(
            cgroup_room(&level("/", "max\n", "0\n", (0, 0)).as_level()),
            Ok(None)
        );
        // The file pages a populated host set holds do not eat the room: a
        // cgroup 20 GiB full of file pages under a 61 GiB limit still
        // reports 61 GiB, the room `MemAvailable` would give those pages.
        assert_eq!(
            cgroup_room(
                &level("/scope", &n(61 * gib), &n(20 * gib), (10 * gib, 10 * gib)).as_level()
            ),
            Ok(Some(61 * gib))
        );
        // Malformed files, each by name.
        assert!(
            cgroup_room(&level("/", "61G\n", "0\n", (0, 0)).as_level()).is_err(),
            "memory.max"
        );
        assert!(
            cgroup_room(&level("/", &n(8 * gib), "  \n", (0, 0)).as_level()).is_err(),
            "memory.current"
        );
        let owned = level("/", &n(8 * gib), "0\n", (0, 0));
        for stat in [
            "",
            "active_file 4096\n",
            "active_file 4k\ninactive_file 4096\n",
        ] {
            let bad = CgroupLevel {
                stat,
                ..owned.as_level()
            };
            assert!(cgroup_room(&bad).is_err(), "{stat:?}");
        }
        assert!(
            cgroup_room(&level("/", &n(8 * gib), &n(16 * gib), (0, 0)).as_level()).is_err(),
            "usage over the limit"
        );
    }

    /// The host's available bytes are the smaller of `MemAvailable` and
    /// every cgroup v2 level's room, the deciding reading named: no levels
    /// and a level without a limit leave `MemAvailable`; a nested limit
    /// binds through the smaller parent; a tie names `MemAvailable`.
    #[test]
    fn host_room_takes_the_smaller_reading() {
        let meminfo = "MemTotal:       263741212 kB\nMemAvailable:   257300180 kB\n";
        let mem = 257_300_180 * 1024;
        assert_eq!(host_room(meminfo, &[]), Ok((mem, HostRead::MemAvailable)));
        let unlimited = [level("/", "max\n", "0\n", (0, 0))];
        assert_eq!(
            host_room(meminfo, &as_levels(&unlimited)),
            Ok((mem, HostRead::MemAvailable))
        );
        // The scope's limit binds: the room is under MemAvailable.
        let scope = [level(
            "/user.slice/session-3.scope",
            "65539000000\n",
            "1073741824\n",
            (512 << 20, 512 << 20),
        )];
        assert_eq!(
            host_room(meminfo, &as_levels(&scope)),
            Ok((
                65_539_000_000,
                HostRead::Cgroup {
                    path: "/user.slice/session-3.scope".to_owned()
                }
            ))
        );
        // A nested limit: the smaller parent decides, and the deciding
        // reading's text names its path.
        let nested = [
            level(
                "/user.slice/session-3.scope",
                "107374182400\n",
                "0\n",
                (0, 0),
            ),
            level("/user.slice", "65539000000\n", "0\n", (0, 0)),
        ];
        let (bytes, read) = host_room(meminfo, &as_levels(&nested)).expect("the room");
        assert_eq!(
            (bytes, &read.to_string()),
            (
                65_539_000_000,
                &"the room under the cgroup v2 memory limit at /sys/fs/cgroup/user.slice \
                 (memory.max − memory.current + active_file + inactive_file)"
                    .to_owned()
            )
        );
        // A tie leaves MemAvailable the named reading.
        let tie = [level("/scope", &format!("{}\n", mem), "0\n", (0, 0))];
        assert_eq!(
            host_room(meminfo, &as_levels(&tie)),
            Ok((mem, HostRead::MemAvailable))
        );
    }

    /// The need is the plan's host terms less the OS reserve; exactly the
    /// available bytes pass, one byte more is refused, and the refusal names
    /// both figures and the levers that lower the need.
    #[test]
    fn host_need_is_refused_past_mem_available() {
        let need = HostNeed {
            experts: 1000,
            tables: 100,
            shadows: 10,
            reserves: 500,
            extra: 50,
            arena: 0,
            cards: 0,
            os: 400,
            card_budget: true,
        };
        assert_eq!(need.bytes(), 1260);
        assert_eq!(need.check(1260), Ok(()));
        let short = need.check(1259).expect_err("one byte short");
        let text = short.to_string();
        for part in [
            "1259 B available",
            "the 1260 B this load needs by 1 B",
            "residency churn pool 50 B",
            "the OS reserve 400 B",
            "BLOOMERY_RESIDENCY=off",
            "BLOOMERY_CARD_BUDGET",
        ] {
            assert!(text.contains(part), "{part:?} in {text}");
        }
    }

    /// The numbers of the paged Qwen3.8 load that a 32 GB host refused: the
    /// need's terms, its arena, and the lowered reading the load took.
    fn paged_need() -> HostNeed {
        HostNeed {
            experts: 2_035_302_400,
            tables: 0,
            shadows: 0,
            reserves: 11_955_785_728,
            extra: 0,
            arena: 20_335_071_232,
            cards: 0,
            os: 6_694_629_376,
            card_budget: false,
        }
    }

    const PAGED_AVAILABLE: u64 = 27_499_458_560;

    /// A reading under the need by less than the arena's room above its
    /// floor shrinks the arena by the shortfall, and the shrunk need passes
    /// the same reading.
    #[test]
    fn grant_shrinks_the_arena_by_the_shortfall() {
        let need = paged_need();
        assert_eq!(need.bytes(), 27_631_529_984);
        let least = 1 << 30;
        let grant = need
            .grant(PAGED_AVAILABLE, least)
            .expect("the arena shrinks");
        assert_eq!(
            grant,
            ArenaGrant {
                planned: 20_335_071_232,
                granted: 20_335_071_232 - 132_071_424,
            }
        );
        assert_eq!(grant.shrunk(), 132_071_424);
        let shrunk = HostNeed {
            arena: grant.granted,
            ..need
        };
        assert_eq!(shrunk.check(PAGED_AVAILABLE), Ok(()));
    }

    /// A shortfall past the arena's room above its floor is refused, naming
    /// the floor; the shortfall that leaves exactly the floor is granted; a
    /// plan with no arena is refused exactly as `check` refuses it.
    #[test]
    fn grant_refuses_a_shortfall_past_the_floor() {
        let need = paged_need();
        let least = 4 << 30;
        let room = need.arena - least;
        let at_floor = need.grant(need.bytes() - room, least).expect("the floor");
        assert_eq!(at_floor.granted, least);
        let past = need
            .grant(need.bytes() - room - 1, least)
            .expect_err("one byte past the floor");
        assert_eq!(past.floor, Some(least));
        let text = past.to_string();
        assert!(
            text.contains(&format!("under its floor {least} B")),
            "the floor in {text}"
        );
        let bare = HostNeed { arena: 0, ..need };
        let short = bare.bytes() - 1;
        let refused = bare.grant(short, least).expect_err("no arena to shrink");
        assert_eq!(Some(refused), bare.check(short).err());
        assert_eq!(refused.floor, None);
        assert_eq!(bare.grant(bare.bytes(), least), Ok(ArenaGrant::whole(0)));
    }

    /// A reading that covers the need grants the arena as planned, whatever
    /// the floor.
    #[test]
    fn grant_keeps_a_covered_arena_whole() {
        let need = paged_need();
        for available in [need.bytes(), need.bytes() + 1, u64::MAX] {
            for least in [0, 1 << 30, need.arena + 1] {
                assert_eq!(
                    need.grant(available, least),
                    Ok(ArenaGrant::whole(need.arena)),
                    "available {available} floor {least}"
                );
            }
        }
    }

    /// The numbers of the unpaged Qwen3.8 load that a 64 GB host refused:
    /// the need's terms, with no arena to shrink.
    fn unpaged_need() -> HostNeed {
        HostNeed {
            experts: 59_267_686_400,
            tables: 0,
            shadows: 0,
            reserves: 8_533_127_168,
            extra: 0,
            arena: 0,
            cards: 0,
            os: 6_694_629_376,
            card_budget: false,
        }
    }

    const UNPAGED_AVAILABLE: u64 = 60_943_515_648;
    const UNPAGED_SHORT: u64 = 162_668_544;

    /// A load with no arena is held to its need exactly: the reading under
    /// the need is refused, and `grant` has nothing to shrink.
    #[test]
    fn an_unpaged_load_short_of_its_need_is_refused() {
        let need = unpaged_need();
        assert_eq!(need.bytes(), 61_106_184_192);
        assert_eq!(need.bytes() - UNPAGED_AVAILABLE, UNPAGED_SHORT);
        let short = need
            .check(UNPAGED_AVAILABLE)
            .expect_err("the reading is short");
        assert_eq!(short.floor, None);
        assert_eq!(need.grant(UNPAGED_AVAILABLE, 0), Err(short));
        assert_eq!(need.check(need.bytes()), Ok(()));
    }

    /// A plan made at a read room holds its need a margin under the
    /// reading, so a load whose reading fell by the margin or less still
    /// covers it: the plan of the unpaged need above is made only at a
    /// reading of its need plus the margin, and the reading the load took
    /// there is the need or more.
    #[test]
    fn a_plan_room_covers_a_load_that_drifted_by_the_margin() {
        let need = unpaged_need().bytes();
        let at = |read: u64| plan_room(&(read, HostRead::MemAvailable));
        assert!(
            at(need) < need,
            "a reading of the need plans a room under it"
        );
        let planned_at = need + HOST_DRIFT;
        assert!(
            need <= at(planned_at),
            "the least reading that holds the need"
        );
        assert!(at(planned_at - 1) < need, "one byte under pages");
        assert_eq!(
            unpaged_need().check(planned_at - UNPAGED_SHORT),
            Ok(()),
            "the reporter's drift is under the margin"
        );
        assert!(unpaged_need().check(planned_at - HOST_DRIFT - 1).is_err());
    }
}
