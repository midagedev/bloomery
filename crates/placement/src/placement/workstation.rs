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
#[cfg(test)]
use super::nodes::{Fabric, Nodes};
use super::{
    Card, Device, Host, HostTotals, Machine, Node, PlacementError, Plan, Reserve, UnifiedPool,
};
use crate::kernels::{
    DownAct, DownForms, leg_bytes, q8_act_bytes, try_leg_bytes, try_q8_act_bytes,
};

pub const MIB: u64 = 1 << 20;

/// One card as the census gives a device ([`spec_of_device`]), or a card of
/// this workstation by name as `nvidia-smi` reports it with no process on it
/// [measured] ([`A6000`], [`RTX_3090`]): its memory, the part the driver
/// keeps, its allocation granule, and the device it is when it was resolved
/// against this process's devices.
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
    /// The bytes the card keeps free past its context and scratch
    /// (`BLOOMERY_CARD_MARGIN`, [`super::card_budget::keep_free`]); `None`
    /// keeps [`MARGIN`]. A plan's [`Card::margin_bytes`] takes it from
    /// [`CardSpec::margin`].
    pub margin_bytes: Option<u64>,
    /// The device allocator's granule ([`Card::granule_bytes`]): the
    /// census's reading of the device ([`DeviceInfo::granule_bytes`]), or a
    /// card by name's measured one ([`GRANULE`]).
    pub granule_bytes: NonZeroU64,
}

impl CardSpec {
    /// The bytes the card keeps free: its own, or [`MARGIN`].
    #[must_use]
    pub const fn margin(&self) -> u64 {
        match self.margin_bytes {
            Some(m) => m,
            None => MARGIN,
        }
    }

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

/// The maintainer workstation's RTX A6000 [measured, `nvidia-smi`]: a card
/// by name, the named plans' (plan (a), plan (b′)'s stage, `--place a6000`).
/// A census device of any name plans its own figures ([`spec_of_device`]).
pub const A6000: CardSpec = CardSpec {
    name: "A6000",
    total_bytes: 49_140 * MIB,
    driver_reserve_bytes: 548 * MIB,
    device: None,
    free_bytes: None,
    held_by: None,
    integrated: false,
    margin_bytes: None,
    granule_bytes: GRANULE,
};

/// The maintainer workstation's RTX 3090 [measured, `nvidia-smi`]: a card by
/// name, the gate plan's bytes ([`plan_gate`], `--place gate` on any card
/// with the room) and plan (b′)'s tier.
pub const RTX_3090: CardSpec = CardSpec {
    name: "3090",
    total_bytes: 24_576 * MIB,
    driver_reserve_bytes: 400 * MIB,
    device: None,
    free_bytes: None,
    held_by: None,
    integrated: false,
    margin_bytes: None,
    granule_bytes: GRANULE,
};

/// Every card a plan names by name, the most usable bytes first: the names
/// a `--place` word takes ([`word_picks`]), and the workstation's cards by
/// size an alias names with no census ([`workstation_spec`]), the gates'
/// reproduction of this workstation.
pub const CARDS: [CardSpec; 2] = [A6000, RTX_3090];

const _: () = assert!(A6000.usable_bytes() > RTX_3090.usable_bytes());

/// The CUDA context term every card sets aside, a card of the census or by
/// name [assumed]: the program's figure, not a reading of the card. It covers
/// the primary context's own creation ([`CONTEXT_SELF`]) and what a load takes
/// from the device past a census reading that no other plan term counts: the
/// bundles' module images, the kernels' local-memory growth, a capture's graph.
pub const CONTEXT: u64 = 512 * MIB;
/// The part of [`CONTEXT`] a census reading already carries, the primary
/// context's own creation cost [measured on the workstation: an idle A6000's
/// census reading sits this far below its usable bytes; an idle 3090's sits
/// less far below, a `Gpu` and its module included]. A census reads free bytes
/// on that context, so the reading is already net of it; a plan's cap adds it
/// back (placement's `capped`) and counts the context once, in [`CONTEXT`]. The
/// census prices no context of its own, so every card's cap adds this figure:
/// where a card's context costs more, the reading binds the cap; where it
/// costs less, the usable bytes do.
pub const CONTEXT_SELF: u64 = 271 * MIB;
/// The m = 1 scratch every card's stage sets aside [assumed]: the program's
/// figure, not the card's.
pub const SCRATCH: u64 = 64 * MIB;
/// What the expert rule leaves free on a card that `BLOOMERY_CARD_MARGIN`
/// gives no margin of its own ([`CardSpec::margin`]): a policy, not a reading.
pub const MARGIN: u64 = 1 << 30;

/// The workstation cards' allocation granule [measured: what a load takes
/// from `cuMemGetInfo` is a whole number of them, and
/// `cuMemGetAllocationGranularity` reports the same size]: a card by name's,
/// and a census device's on a backend that reads none. A census device's own
/// is its reading ([`DeviceInfo::granule_bytes`]).
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

/// This workstation's base page, the unit its row room counts in
/// ([`super::row_room`]). A host on another base page passes its own page
/// to the row room, and the row room counts the pages a row sits on.
pub const ROW_PAGE: u64 = 4096;

/// What the machine's [`OS_OTHER`] is called: what the OS and every other
/// process held when [`HOST_USABLE`] was measured.
pub const OS_RESERVE: &str = "OS and other";

/// What a host reserve set by [`reserve_part_reads`] is called: the least a
/// load from a part source reads before its host set, its parts one at a
/// time with the upload's staging ring. [`HostNeed`] counts it as the plan's
/// [`HostNeed::read_ahead`], not among the reserves.
pub const PART_READS: &str = "part read-ahead";

/// The context used to bound a part source's card reads before choosing the serving context:
/// the smallest nonzero slot share (`ctx_ask::choose`), which leaves the most parts on the card.
/// An explicit context may be below the default context search's floor.
pub const PART_READ_CONTEXT: u64 = 1;

/// Set `bytes` aside on `machine`'s host as a part source's least reads
/// ([`PART_READS`]), in place of any set before: a plan made on it counts
/// them in its host need ([`HostNeed::read_ahead`]), so the NVMe expert
/// tier's split ([`super::expert_nvme_tier`]) keeps them room and the load,
/// which counts its own read-ahead at the room it reads, holds the plan's
/// need. The caller reads `bytes` from the plan's card parts
/// (`bloomery_gpu::parts::least_reads`), which no host term moves.
pub fn reserve_part_reads(machine: &mut Machine, bytes: u64) {
    set_host_reserve(machine, PART_READS, bytes);
}

/// `machine`'s host reserve `name` at `bytes`, in place of any set before.
fn set_host_reserve(machine: &mut Machine, name: &'static str, bytes: u64) {
    machine.host_mut().reserves.retain(|(n, _)| n != name);
    machine
        .reserve(Reserve {
            on: Device::Host,
            name,
            bytes,
        })
        .expect("a machine has a host");
}

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

/// `spec` running `layers`, with the program's context and scratch, the
/// spec's margin and its allocation granule. The spec's free reading and its
/// holders ride on the card for the plan's budget term and its refusals
/// ([`Card::free_bytes`], [`Card::held_by`]).
fn card(spec: CardSpec, layers: Range<usize>, head: bool) -> Card {
    Card {
        name: spec.name.to_string(),
        tier_leg: None,
        device: spec.device,
        usable_bytes: spec.usable_bytes(),
        context_bytes: CONTEXT,
        scratch_bytes: SCRATCH,
        margin_bytes: spec.margin(),
        granule_bytes: spec.granule_bytes,
        layers,
        head,
        token_embedding: false,
        reserve_only: false,
        reserves: Vec::new(),
        free_bytes: spec.free_bytes,
        held_by: spec.held_by,
    }
}

/// `spec` as an expert tier card with no stage or head: the context,
/// scratch, margin and batch service. A reserve beside them enters through
/// [`Machine::reserve`].
#[must_use]
pub fn tier_card(spec: CardSpec, batch: TierBatchBytes) -> Card {
    let mut tier = card(spec, 0..0, false);
    tier.reserves.push((
        TIER_BATCH_RESERVE.to_string(),
        batch
            .card()
            .unwrap_or_else(|error| panic!("tier batch card reserve: {error}")),
    ));
    tier.tier_leg = batch.leg;
    tier
}

/// `spec` as a reserve-only member of a plan ([`Card::reserve_only`]): a card
/// with no stage, no head and no expert, only what is reserved on it
/// ([`Machine::reserve`]). It serves no tier, so it carries no batch service.
#[must_use]
pub fn holder_card(spec: CardSpec) -> Card {
    Card {
        reserve_only: true,
        ..card(spec, 0..0, false)
    }
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
        nodes: vec![node_of(&spec)],
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

/// [`placed_on`] with the m = 1 scratch ([`SCRATCH`]) plus `arena` bytes: what
/// a program's prompt arenas hold beyond it, which the program sizes and its
/// load refuses to pass ([`counted_arena_bytes`]).
#[must_use]
pub fn placed_with_arena(spec: CardSpec, layers: usize, arena: u64) -> Machine {
    placed_on(spec, layers, SCRATCH + arena)
}

/// The prompt-arena bytes a plan made by [`placed_with_arena`] counts on
/// `card`: its scratch past the m = 1 scratch. A load whose arenas hold more
/// is refused by name.
#[must_use]
pub const fn counted_arena_bytes(card: &Card) -> u64 {
    card.scratch_bytes.saturating_sub(SCRATCH)
}

/// Design §5 (a): the A6000 runs all `layers` and the head; the 3090 is idle.
#[must_use]
pub fn plan_a(layers: usize) -> Machine {
    plan_on(A6000, layers)
}

/// Plan (a) asked for the DSpark draft's resident bytes.
///
/// The plan is one stage card and no tier ([`plan_a`]); the draft sits on the
/// stage card, so its bytes leave the card's budget as [`DRAFT_RESERVE`]
/// before any expert. It is [`plan_tiers`] of that card over no tier. `None`
/// is [`plan_a`].
pub fn plan_a_draft(layers: usize, draft_bytes: Option<u64>) -> Result<Machine, PlacementError> {
    plan_tiers(
        layers,
        A6000,
        &[],
        draft_bytes.map(|bytes| draft_reserve(Device::Card(0), bytes)),
        NO_TIER_BATCH,
    )
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
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    }
}

/// What the DSpark draft's reserve on a card is called.
pub const DRAFT_RESERVE: &str = "DSpark draft";

/// The DSpark draft's resident bytes as a reserve on `on`, a card of the
/// plan's whole device list ([`Machine::all_cards`], the stage first).
#[must_use]
pub const fn draft_reserve(on: Device, bytes: u64) -> Reserve {
    Reserve {
        on,
        name: DRAFT_RESERVE,
        bytes,
    }
}

/// The prompt-batch bytes of a plan with no tier card: none is held, so the
/// plan reserves none.
pub const NO_TIER_BATCH: TierBatchBytes = TierBatchBytes {
    staging: 0,
    scratch: 0,
    host: 0,
    leg: None,
};

/// What the expert tier's prompt-batch service is called as a reserve: on
/// the tier card, its staging and its tile scratch; on the host, the rows it
/// hands back and the places it reads.
pub const TIER_BATCH_RESERVE: &str = "tier prompt batch";
pub const TIER_BATCH_HOST_RESERVE: &str = "tier prompt batch rows";

/// The host reserve that carries a load's checkpoint slots: the pinned host
/// buffers its sequences copy their recurrent state to.
pub const CHECKPOINTS_RESERVE: &str = "checkpoints";

/// The host reserve of a load with the MTP draft that carries its
/// checkpoints' hidden rows: each slot's points at most, one row each,
/// beside the checkpoints themselves.
pub const POINT_ROWS_RESERVE: &str = "mtp point rows";

/// The host reserve of a load whose prompt calls stream their host experts
/// (`BLOOMERY_XSTREAM=split`): the lane's pinned staging, its slots of a ring
/// slot's bytes each, held for the load's life beside every other term
/// ([`reserve_stream_staging`]).
pub const XSTREAM_STAGING_RESERVE: &str = "xstream staging";

/// The most pinned staging slots the expert stream's lane holds
/// (`bloomery_gpu::host::xstream`): a ring slot's bytes each, at most the
/// ring's half, which a plan does not know before the load sizes its ring.
pub const STAGING_SLOTS: usize = 64;

/// Set aside on `machine`'s host the staging a load whose prompt calls
/// stream pins for its life, [`STAGING_SLOTS`] slots of `slot` bytes (a ring
/// slot, `place_hybrid::stream_slot_bytes`), as [`XSTREAM_STAGING_RESERVE`]
/// in place of any set before: a plan made on it counts them pinned in its
/// host need and the NVMe expert tier's split leaves them room. A `slot` of 0,
/// a load that streams nothing, sets none. The caller asks before the plan
/// whether the load streams (the common `BLOOMERY_XSTREAM` rule).
pub fn reserve_stream_staging(machine: &mut Machine, slot: u64) {
    if slot == 0 {
        machine
            .host_mut()
            .reserves
            .retain(|(n, _)| n != XSTREAM_STAGING_RESERVE);
    } else {
        set_host_reserve(
            machine,
            XSTREAM_STAGING_RESERVE,
            slot.saturating_mul(STAGING_SLOTS as u64),
        );
    }
}

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
    /// A common leg's candidate-dependent scratch; `scratch` is its bound.
    pub leg: Option<TierLegScratch>,
}

impl TierBatchBytes {
    /// The tier card's part; refuses an invalid leg bound or a sum past u64.
    pub fn card(&self) -> Result<u64, PlacementError> {
        if let Some(leg) = self.leg
            && leg.bytes(leg.downs)? != self.scratch
        {
            return Err(PlacementError::Metadata {
                key: "tier leg batch".into(),
                detail: "scratch reserve differs from the descriptor's bound".into(),
            });
        }
        self.staging
            .checked_add(self.scratch)
            .ok_or(PlacementError::TierBatchCard {
                staging: self.staging,
                scratch: self.scratch,
            })
    }
}

/// The common leg's scratch geometry and the descriptor's admitted forms.
/// Placement prices its candidate's selected forms jointly with expert rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierLegScratch {
    pub n_embd: u64,
    pub ff: u64,
    pub n_used: u64,
    pub chunk: u64,
    pub downs: DownForms,
    /// One f32 input-restoration scale per slot of the leg's chunk.
    pub input_scales: bool,
}

impl TierLegScratch {
    /// Scratch for exactly these selected forms, refusing invalid geometry
    /// and arithmetic overflow before an allocation or placement candidate.
    pub fn bytes(self, downs: DownForms) -> Result<u64, PlacementError> {
        let invalid = || PlacementError::Metadata {
            key: "tier leg scratch".into(),
            detail: format!("invalid or overflowing geometry: {self:?}, selected {downs:?}"),
        };
        if self.chunk == 0
            || self.n_used == 0
            || self.ff == 0
            || self.n_embd == 0
            || [
                crate::kernels::DownAct::Q8Act,
                crate::kernels::DownAct::Q8Blocks32,
            ]
            .into_iter()
            .any(|act| downs.holds(act) && !self.downs.holds(act))
        {
            return Err(invalid());
        }
        let counted = || {
            let bytes = try_leg_bytes(self.chunk, self.n_used, self.n_embd, self.ff, downs)?;
            let scales = if self.input_scales {
                4u64.checked_mul(self.chunk)?.checked_mul(self.n_used)?
            } else {
                0
            };
            bytes.checked_add(scales)
        };
        counted().ok_or_else(invalid)
    }
}

/// Experts the tile path's bucket table takes a layer at most, and the
/// columns a tile holds: the tile path's shapes, which size its scratch.
const TILE_BUCKET_EXPERTS: u64 = 1024;
const TILE_COLS: u64 = 8;

/// The tier card's staging of the tier's prompt-batch service for blocks of
/// up to `cols` tokens of rows of `n_embd`, `n_used` slots a token: the
/// activations, the tier places, their q8_1 form, the down outputs by slot.
#[must_use]
pub const fn tier_batch_staging_bytes(n_embd: u64, n_used: u64, cols: u64) -> u64 {
    match try_tier_batch_staging_bytes(n_embd, n_used, cols) {
        Some(bytes) => bytes,
        None => panic!("tier_batch_staging_bytes: staging bytes overflow u64"),
    }
}

const fn try_tier_batch_staging_bytes(n_embd: u64, n_used: u64, cols: u64) -> Option<u64> {
    let Some(slots) = cols.checked_mul(n_used) else {
        return None;
    };
    let Some(places) = 4u64.checked_mul(slots) else {
        return None;
    };
    let Some(down) = places.checked_mul(n_embd) else {
        return None;
    };
    let Some(rows) = 4u64.checked_mul(cols) else {
        return None;
    };
    let Some(rows) = rows.checked_mul(n_embd) else {
        return None;
    };
    let Some(bytes) = rows.checked_add(places) else {
        return None;
    };
    let Some(q8) = try_q8_act_bytes(cols, n_embd) else {
        return None;
    };
    let Some(bytes) = bytes.checked_add(q8) else {
        return None;
    };
    bytes.checked_add(down)
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
/// the card leg's buffers (`bloomery_gpu::card_leg::LegBufs`) with its slots'
/// columns in the 128-value form ([`leg_bytes`]), in place of the tile path's
/// scratch. Selected-form callers use [`try_tier_leg_batch_bytes_of`].
#[must_use]
pub const fn tier_chunk_scratch_bytes(n_embd: u64, ff: u64, n_used: u64, chunk: u64) -> u64 {
    leg_bytes(chunk, n_used, n_embd, ff, DownForms::of(DownAct::Q8Act))
}

/// The host's part of the tier's prompt-batch service: per exchange set, the
/// rows the tier hands back and the places it reads.
#[must_use]
pub const fn tier_batch_host_bytes(n_embd: u64, n_used: u64, cols: u64) -> u64 {
    match try_tier_batch_host_bytes(n_embd, n_used, cols) {
        Some(bytes) => bytes,
        None => panic!("tier_batch_host_bytes: host bytes overflow u64"),
    }
}

const fn try_tier_batch_host_bytes(n_embd: u64, n_used: u64, cols: u64) -> Option<u64> {
    let Some(slots) = cols.checked_mul(n_used) else {
        return None;
    };
    let Some(places) = 4u64.checked_mul(slots) else {
        return None;
    };
    let Some(down) = places.checked_mul(n_embd) else {
        return None;
    };
    let Some(rows_and_places) = down.checked_add(places) else {
        return None;
    };
    2u64.checked_mul(rows_and_places)
}

/// The expert tier's prompt-batch bytes for blocks of up to `cols` tokens of
/// rows of `n_embd`, routed to `n_used` experts of `ff` rows each.
#[must_use]
pub const fn tier_batch_bytes(n_embd: u64, ff: u64, n_used: u64, cols: u64) -> TierBatchBytes {
    TierBatchBytes {
        staging: tier_batch_staging_bytes(n_embd, n_used, cols),
        scratch: tier_block_scratch_bytes(n_embd, ff, cols * n_used),
        host: tier_batch_host_bytes(n_embd, n_used, cols),
        leg: None,
    }
}

/// [`tier_batch_bytes`] for a candidate-selected common leg in the 128-value form.
///
/// Refuses invalid geometry or overflowing byte counts before placement.
pub fn tier_leg_batch_bytes(
    n_embd: u64,
    ff: u64,
    n_used: u64,
    chunk: u64,
    cols: u64,
) -> Result<TierBatchBytes, PlacementError> {
    try_tier_leg_batch_bytes_of(
        n_embd,
        ff,
        n_used,
        (chunk, cols),
        DownForms::of(DownAct::Q8Act),
        false,
    )
}

/// The common leg's staging, host rows and selected-form scratch, refusing
/// invalid geometry or overflowing byte counts before placement.
pub fn try_tier_leg_batch_bytes_of(
    n_embd: u64,
    ff: u64,
    n_used: u64,
    (chunk, cols): (u64, u64),
    downs: DownForms,
    input_scales: bool,
) -> Result<TierBatchBytes, PlacementError> {
    let leg = TierLegScratch {
        n_embd,
        ff,
        n_used,
        chunk,
        downs,
        input_scales,
    };
    let scratch = leg.bytes(downs)?;
    let counted = || -> Option<(u64, u64)> {
        if cols == 0 {
            return None;
        }
        Some((
            try_tier_batch_staging_bytes(n_embd, n_used, cols)?,
            try_tier_batch_host_bytes(n_embd, n_used, cols)?,
        ))
    };
    let (staging, host) = counted().ok_or_else(|| PlacementError::Metadata {
        key: "tier leg batch".into(),
        detail: format!("invalid or overflowing staging: {leg:?}, columns {cols}"),
    })?;
    Ok(TierBatchBytes {
        staging,
        scratch,
        host,
        leg: Some(leg),
    })
}

/// Plan (b′): the A6000 runs all `layers` and the head as in [`plan_a`];
/// the 3090 is an expert tier beside the host, holding each layer's next ids
/// after the stage's prefix, with `draft_bytes` — the DSpark draft's resident bytes
/// (`model::arch::dspark::card_bytes`), when the draft lives there — as a
/// named reserve on the tier (card 1), and the tier's prompt-batch service `batch`
/// ([`tier_batch_bytes`]) as named reserves on the tier and the host.
/// It is [`plan_tiers`] of the A6000 over the one tier card, the 3090.
#[must_use]
pub fn plan_bp(layers: usize, draft_bytes: Option<u64>, batch: TierBatchBytes) -> Machine {
    let draft = draft_bytes.map(|bytes| draft_reserve(Device::Card(1), bytes));
    tiered(layers, A6000, &[RTX_3090], &[], draft, batch, batch.host)
}

/// Plan (b′) over `tiers`: `stage` runs all `layers` and the head as in
/// [`plan_on`]; each tier card, in order, is an expert tier beside the host
/// holding each layer's next ids after the cards before it
/// ([`super::Machine::tiers`]), with the prompt-batch service `batch`
/// ([`tier_batch_bytes`]) reserved on it; `draft` is reserved on the card it
/// names, the stage or a tier, and the host reserves the batch's rows once
/// per tier. A card listed twice ([`PlacementError::DeviceTwice`]) and a
/// draft on a card past the device list ([`PlacementError::ReserveOffPlan`])
/// are refused ([`check_tiers`]). With no tier it is [`plan_on`], with the
/// draft's reserve on the stage card.
pub fn plan_tiers(
    layers: usize,
    stage: CardSpec,
    tiers: &[CardSpec],
    draft: Option<Reserve>,
    batch: TierBatchBytes,
) -> Result<Machine, PlacementError> {
    plan_holding(layers, stage, tiers, &[], draft, batch)
}

/// [`plan_tiers`] with `holders` joining the plan after the tier cards, each a
/// reserve-only member ([`holder_card`]): the card has no expert and carries
/// no batch service, only what is reserved on it. The device list is the
/// stage, the tiers, then the holders, in this order; a holder with nothing
/// reserved on it is the idle tier's refusal at the plan
/// ([`PlacementError::IdleTier`]). A holder is how a draft sits on a card the
/// plan's experts do not use without leaving the plan: the cards that hold
/// experts keep every expert they held.
pub fn plan_holding(
    layers: usize,
    stage: CardSpec,
    tiers: &[CardSpec],
    holders: &[CardSpec],
    draft: Option<Reserve>,
    batch: TierBatchBytes,
) -> Result<Machine, PlacementError> {
    let host_rows = check_holding(stage, tiers, holders, draft, batch)?;
    Ok(tiered(
        layers, stage, tiers, holders, draft, batch, host_rows,
    ))
}

/// The node a machine of `spec` alone plans on: its host and unified pool
/// ([`host_of`], [`unified_of`]). A tier card on a host of its own sits on
/// such a node, tier `t` on node `t + 1` ([`super::Machine::nodes`]).
#[must_use]
pub fn node_of(spec: &CardSpec) -> Node {
    Node {
        host: host_of(spec),
        unified: unified_of(spec),
    }
}

/// One card machine per node, node 0 first: `stage` runs the full model;
/// each remote spec runs [`super::nodes::routed_only`] with zero KV. Device
/// names and ordinals are local to each node, so they may repeat across
/// nodes. Each spec keeps its own free reading and unified pool.
///
/// Node 0's host holds the returned batch rows once per remote; each remote
/// card holds the batch staging and scratch. Each independent machine has one
/// head stage; a remote's routed-only model has no head tensors to place there.
/// A host-row sum past u64 is refused, as in [`check_tiers`]. With no remote
/// it is [`plan_on`].
pub fn plan_node_tiers(
    layers: usize,
    stage: CardSpec,
    remotes: &[CardSpec],
    batch: TierBatchBytes,
) -> Result<Vec<Machine>, PlacementError> {
    let host_rows = tier_service(remotes.len(), 0, None, batch)?;
    let mut machines = vec![tiered(layers, stage, &[], &[], None, batch, host_rows)];
    for &spec in remotes {
        let mut remote = plan_on(spec, layers);
        remote.cards[0]
            .reserves
            .extend(tier_card(spec, batch).reserves);
        remote.cards[0].tier_leg = batch.leg;
        machines.push(remote);
    }
    Ok(machines)
}

/// What [`plan_tiers`] refuses of its cards, before any layer count: a card
/// listed twice — one device ([`CardSpec::same_device`]), whose usable bytes
/// the plan would count once a listing — and a draft on a card past the
/// stage and tier list. Returns the host's prompt-batch rows, one set a tier.
pub fn check_tiers(
    stage: CardSpec,
    tiers: &[CardSpec],
    draft: Option<Reserve>,
    batch: TierBatchBytes,
) -> Result<u64, PlacementError> {
    check_holding(stage, tiers, &[], draft, batch)
}

/// [`check_tiers`] over the stage, the tiers and the `holders` after them
/// ([`plan_holding`]): the device list a draft's index runs over.
pub fn check_holding(
    stage: CardSpec,
    tiers: &[CardSpec],
    holders: &[CardSpec],
    draft: Option<Reserve>,
    batch: TierBatchBytes,
) -> Result<u64, PlacementError> {
    let all: Vec<CardSpec> = std::iter::once(stage)
        .chain(tiers.iter().copied())
        .chain(holders.iter().copied())
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
    tier_service(tiers.len(), holders.len(), draft, batch)
}

/// The draft's card and the host's batch-row sum for local or remote tiers.
fn tier_service(
    tiers: usize,
    holders: usize,
    draft: Option<Reserve>,
    batch: TierBatchBytes,
) -> Result<u64, PlacementError> {
    let cards = 1 + tiers + holders;
    if let Some(d) = draft.filter(|d| match d.on {
        Device::Card(i) => i >= cards,
        Device::Host => false,
        Device::Nvme | Device::Unused => true,
    }) {
        return Err(PlacementError::ReserveOffPlan {
            reserve: d.name.to_string(),
            on: d.on,
            cards,
        });
    }
    if tiers != 0 {
        batch.card()?;
    }
    u64::try_from(tiers)
        .ok()
        .and_then(|k| batch.host.checked_mul(k))
        .ok_or(PlacementError::TierBatchHost {
            tiers,
            bytes: batch.host,
        })
}

/// The stage machine, its local expert tiers, its reserve-only `holders` and
/// the returned batch rows from all local or remote tiers (`host_rows`), with
/// the draft's reserve on the card it names (checked by [`tier_service`]).
fn tiered(
    layers: usize,
    stage: CardSpec,
    tiers: &[CardSpec],
    holders: &[CardSpec],
    draft: Option<Reserve>,
    batch: TierBatchBytes,
    host_rows: u64,
) -> Machine {
    let expert_tiers = tiers.len();
    let tiers: Vec<Card> = tiers
        .iter()
        .map(|&spec| tier_card(spec, batch))
        .chain(holders.iter().map(|&spec| holder_card(spec)))
        .collect();
    let mut h = host_of(&stage);
    if expert_tiers != 0 || host_rows != 0 {
        h.reserves
            .push((TIER_BATCH_HOST_RESERVE.to_string(), host_rows));
    }
    let mut machine = Machine {
        cards: vec![card(stage, 0..layers, true)],
        tiers,
        nodes: vec![Node {
            host: h,
            unified: unified_of(&stage),
        }],
    };
    if let Some(d) = draft {
        machine
            .reserve(d)
            .expect("tier_service checked the draft's card against the device list");
    }
    machine
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

/// A DGX Spark's pool as the node plans read it: the usable pool for model,
/// cache and scratch [assumed, `docs/contrib/dgx-spark.md` §9.7 U], below the
/// idle reading the census takes.
#[cfg(test)]
pub(crate) const GB10_POOL: u64 = 115_000_000_000;

/// The rate our kernels read a GB10's pool at, bytes a second [measured,
/// `docs/contrib/dgx-spark.md` §9.7 BW].
#[cfg(test)]
pub(crate) const GB10_READ: u64 = 208_800_000_000;

/// A Spark's NIC a direction, bytes a second: the SoC side, whatever the port
/// count [derived, `docs/contrib/dgx-spark.md` §9.1].
#[cfg(test)]
pub(crate) const GB10_NIC: u64 = 23_700_000_000;

/// This workstation's ruler in permille: the 95 % interval on a difference
/// of two arm means at four rounds (AGENTS.md, "Know the ruler").
#[cfg(test)]
pub(crate) const RULER: u64 = 10;

/// A DGX Spark's GB10 as the census reads it with `free` bytes free
/// ([`spec_of_device`]): an integrated device of a synthetic identity.
#[cfg(test)]
pub(crate) fn gb10(free: u64) -> CardSpec {
    spec_of_device(&super::devices::DeviceInfo {
        ordinal: 0,
        name: "NVIDIA GB10".to_string(),
        total_bytes: 130_661_761_024,
        free_bytes: free,
        uuid: [0x10; 16],
        pci_bus: "000f:01:00.0".to_string(),
        held_by: None,
        compute_capability: Some((12, 1)),
        granule_bytes: None,
        integrated: true,
    })
}

/// `n` synthetic DGX Sparks on `fabric` ([`super::nodes`]): [`plan_on`] a
/// [`gb10`] with [`GB10_POOL`] free for all `layers` and the head, with
/// `n − 1` more nodes of that pool each; the read rate and NIC a GB10's,
/// the ruler this workstation's.
///
/// # Panics
///
/// At `n` 0: a machine has this process's node.
#[cfg(test)]
pub(crate) fn spark_nodes(n: usize, layers: usize, fabric: Fabric) -> Nodes {
    assert!(n > 0, "spark_nodes(0): a machine has this process's node");
    let spec = gb10(GB10_POOL);
    let batch = TierBatchBytes {
        staging: 0,
        scratch: 0,
        host: 0,
        leg: None,
    };
    let mut machines = plan_node_tiers(layers, spec, &vec![spec; n - 1], batch)
        .expect("synthetic nodes with no batch bytes");
    let mut machine = machines.remove(0);
    machine
        .nodes
        .extend(machines.into_iter().map(|m| m.nodes[0].clone()));
    Nodes {
        machine,
        fabric,
        read: GB10_READ,
        nic: GB10_NIC,
        ruler_permille: RULER,
    }
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
/// The need is the load's peak, not the sum of every term: a part source's
/// reads ([`HostNeed::read_ahead`]) end with the card upload, and a host set
/// that holds `extra` is read in after it (`GpuModel::load_placed`'s order),
/// so a load holds the larger of the two, never both. Every other term is
/// counted in both phases.
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
    /// Pinned ring buffers, including the part loader's upload staging at load time.
    pub shadows: u64,
    pub reserves: u64,
    pub extra: u64,
    /// Owned part results and their concurrent read/normalization scratch during loading, never
    /// held beside `extra`: a plan's [`PART_READS`] reserve, the least a load from a part source
    /// reads; a load's own read-ahead at the room it reads (`PartsLoad::host_need`).
    pub read_ahead: u64,
    /// The plan's NVMe expert tier arena (`HostTotals::nvme_arena_bytes`):
    /// the bytes the tier's slots fill beside these terms, anonymous or
    /// pinned, 0 without one. [`HostNeed::bytes`] counts it;
    /// [`HostNeed::plan_bytes`] leaves it to the split dial that shapes it.
    pub arena: u64,
    /// On a unified machine, what the cards take from the pool the host's
    /// room is read from (`HostTotals::pool_card_bytes`); 0 where each card
    /// has memory of its own.
    pub cards: u64,
    pub os: u64,
    /// The plan was made under a card budget the caller gave
    /// (`BLOOMERY_CARD_BUDGET`), not one [`crate::placement::plan_in_pool`]
    /// lowered to the NVMe expert tier's floor.
    pub card_budget: bool,
}

impl HostNeed {
    /// `plan`'s host terms, with `extra` bytes the host set holds beside
    /// them.
    #[must_use]
    pub fn of(plan: &Plan<'_>, extra: u64) -> HostNeed {
        HostNeed::of_totals(
            plan.machine.host(),
            &plan.host,
            plan.card_budget.is_set() && !plan.budget_from_pool,
            extra,
        )
    }

    /// The host terms `totals` hold on `host`, a plan made under a card
    /// budget or not, with `extra` beside them: [`HostNeed::of`]'s, before
    /// the plan that carries them is built.
    #[must_use]
    pub fn of_totals(host: &Host, totals: &HostTotals, card_budget: bool, extra: u64) -> HostNeed {
        let reads = named_reserve(host, PART_READS);
        HostNeed {
            experts: totals.expert_bytes,
            tables: totals.table_bytes,
            shadows: totals.shadow_bytes,
            reserves: totals.reserve_bytes.saturating_sub(reads),
            extra,
            read_ahead: reads,
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

    /// The terms [`HostNeed::bytes`] sums, less the OS reserve, exact: no
    /// sum of u64 terms wraps in u128. `arena` says whether the NVMe tier's
    /// arena is among them ([`HostNeed::plan_bytes`] leaves it out). The
    /// split of the NVMe expert tier does its room arithmetic on this sum.
    pub(super) fn exact(&self, arena: bool) -> u128 {
        [
            self.experts,
            self.tables,
            self.shadows,
            self.reserves,
            self.extra.max(self.read_ahead),
            if arena { self.arena } else { 0 },
            self.cards,
        ]
        .into_iter()
        .map(u128::from)
        .sum::<u128>()
        .saturating_sub(u128::from(self.os))
    }

    /// The bytes `MemAvailable` must cover: the plan's host terms, the larger
    /// of `extra` and the part reads, the arena and the cards' share of a
    /// unified pool — what a placed load of this plan holds at its peak. A
    /// need past u64 bytes reads `u64::MAX`, which no reading but `u64::MAX`
    /// covers ([`HostNeed::check`] holds the exact sum).
    #[must_use]
    pub fn bytes(&self) -> u64 {
        u64::try_from(self.exact(true)).unwrap_or(u64::MAX)
    }

    /// [`HostNeed::bytes`] with the arena out: the plan's own host terms,
    /// the larger of `extra` and the part reads, and the cards' share of a
    /// unified pool — the sum the plan-time room arithmetic of
    /// `expert_nvme_tier` shapes, the arena being its own dial beside it (the
    /// room that arithmetic splits has already lost the arena).
    #[must_use]
    pub fn plan_bytes(&self) -> u64 {
        u64::try_from(self.exact(false)).unwrap_or(u64::MAX)
    }

    /// What `room` leaves a part source's reads ([`HostNeed::read_ahead`]):
    /// the room less every term but `extra` and the reads, which the load
    /// never holds at once, so the reads may take the room the churn pool
    /// takes once they end.
    #[must_use]
    pub fn read_room(&self, room: u64) -> u64 {
        let beside = HostNeed {
            extra: 0,
            read_ahead: 0,
            ..*self
        };
        room.saturating_sub(beside.bytes())
    }

    /// What `room` leaves past [`HostNeed::plan_bytes`]: the plan's host
    /// headroom, the one budget a churn pool beside the plan must fit. The
    /// one definition — a plan at the machine's figure
    /// ([`HostNeed::machine_room`]), at a stated room ([`Plan::plan_at`]) and
    /// at the room a split read (`expert_nvme_tier`) all take it.
    #[must_use]
    pub fn headroom(&self, room: u64) -> i128 {
        i128::from(room) - i128::from(self.plan_bytes())
    }

    /// What the room a plan is made at on `reading` ([`plan_room`]) leaves
    /// past [`HostNeed::bytes`]: the budget an unset residency climb fits its
    /// churn pool to, the pool being the `extra` of the load's own check
    /// ([`HostNeed::check`]). A read room keeps [`HOST_DRIFT`] off, so a pool
    /// that fits still passes the check at a reading the load's own start has
    /// lowered by the margin or less; a given room is held as given.
    #[must_use]
    pub fn room_left(&self, reading: &(u64, HostRead)) -> i128 {
        i128::from(plan_room(reading)) - i128::from(self.bytes())
    }

    /// The host room `room` (its bytes and the reading that gave them,
    /// [`host_room`]'s or a given room) covers the need, or the refusal that
    /// names both, the reading, each term and what lowers the need.
    pub fn check(self, room: &(u64, HostRead)) -> Result<(), HostShort> {
        if self.exact(true) <= u128::from(room.0) {
            Ok(())
        } else {
            Err(HostShort {
                need: self,
                available: room.0,
                read: Box::new(room.1.clone()),
                floor: None,
            })
        }
    }

    /// [`HostNeed::check`] at the room `plan` was made at, with what decided
    /// it ([`HostTotals::room`], [`Plan::plan_at`]: the reading's plan room or
    /// the machine's caps), else at the machine's figure
    /// ([`HostNeed::machine_room`], [`HostRead::Machine`]) for a plan made
    /// without one.
    pub fn check_planned(self, plan: &Plan<'_>) -> Result<(), HostShort> {
        match &plan.host.room {
            Some((_, room)) => self.check(room),
            None => self.check(&(
                HostNeed::machine_room(plan.machine.host()),
                HostRead::Machine,
            )),
        }
    }

    /// The load's check with an elastic arena: the host room `room` covers the
    /// need and the arena is granted as planned, or the arena shrinks by the
    /// shortfall to a grant of at least `least` (the arena's floor,
    /// `ArenaFloor::bytes`), or the refusal [`HostNeed::check`] would give,
    /// naming `least` when the plan has an arena that could not shrink past
    /// it. A plan with no arena is held exactly as `check` holds it. The one
    /// owner of the arena's shrink: the caller reads the grant, never
    /// recomputes it.
    pub fn grant(self, room: &(u64, HostRead), least: u64) -> Result<ArenaGrant, HostShort> {
        let planned = self.arena;
        let Some(short) = self
            .exact(true)
            .checked_sub(u128::from(room.0))
            .filter(|&s| s > 0)
        else {
            return Ok(ArenaGrant::whole(planned));
        };
        match u64::try_from(short)
            .ok()
            .and_then(|short| planned.checked_sub(short))
        {
            Some(granted) if planned > 0 && granted >= least => Ok(ArenaGrant { planned, granted }),
            _ => Err(HostShort {
                need: self,
                available: room.0,
                read: Box::new(room.1.clone()),
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
    named_reserve(host, OS_RESERVE)
}

/// The bytes of `host`'s reserves called `name`.
fn named_reserve(host: &Host, name: &str) -> u64 {
    host.reserves
        .iter()
        .filter(|(n, _)| n == name)
        .map(|(_, b)| b)
        .sum()
}

/// A placed load whose host need passes the host room
/// ([`HostNeed::check`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostShort {
    pub need: HostNeed,
    /// The room's bytes.
    pub available: u64,
    /// The reading that gave the room: the refusal names a given room as
    /// given, so it does not read as this machine's shortage. Boxed to keep
    /// the refusal small on the `Ok` path of every check.
    pub read: Box<HostRead>,
    /// On a refusal of [`HostNeed::grant`] for a plan with an arena: the
    /// floor the arena could not shrink past. `None` from
    /// [`HostNeed::check`], which never shrinks it.
    pub floor: Option<u64>,
}

impl fmt::Display for HostShort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = &self.need;
        let need = n.bytes();
        let (available, read) = (self.available, &*self.read);
        write!(
            f,
            "the host room is {available} B ({read}), under the {need} B \
             this load needs by {} B: host experts {} B + tables {} B + ring shadows {} B + \
             reserves {} B",
            need.saturating_sub(available),
            n.experts,
            n.tables,
            n.shadows,
            n.reserves
        )?;
        match (n.extra, n.read_ahead) {
            (0, 0) => {}
            (pool, 0) => write!(f, " + residency churn pool {pool} B")?,
            (0, reads) => write!(f, " + part read-ahead {reads} B")?,
            (pool, reads) => write!(
                f,
                " + the larger of the residency churn pool {pool} B and the part read-ahead \
                 {reads} B, which the load never holds at once"
            )?,
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
            " − the OS reserve {} B, which the room leaves out already. {}, or load by a \
             placement whose cards hold more of the model (V4.1: --place bp holds more than a, and \
             a more than gate; Qwen3.8: BLOOMERY_QWEN38_EXPERTS=card)",
            n.os,
            match read {
                HostRead::Given => format!("Raise or unset {}", bloomery_levers::HOST_ROOM),
                HostRead::MemAvailable | HostRead::VmStats | HostRead::Cgroup { .. } => {
                    "Free host memory".to_string()
                }
                HostRead::Machine => "Ask for a shorter context".to_string(),
            }
        )?;
        if n.extra > n.read_ahead {
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

/// What can take a page of the host's need back: the class a broker reads a
/// load's need by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageClass {
    /// Pinned by the driver (`cuMemAllocHost`, `cuMemHostAlloc`): resident
    /// for the process's life.
    Pinned,
    /// Anonymous pages: the kernel can only swap them.
    Anon,
    /// File pages in the page cache: reclaimed under pressure and read back
    /// from the file.
    Cache,
    /// File pages `mlock`ed for the model's life (`BLOOMERY_HOST_LOCK`): no
    /// reader of the page cache evicts them.
    Locked,
    /// The cards' share of an integrated device's pool, which is the host's
    /// own memory ([`HostNeed::cards`]).
    Unified,
}

/// How a load holds its host set: the two choices that move a term of the
/// need between page classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hosting {
    /// `BLOOMERY_HOST_LOCK`: the host set is `mlock`ed after the populate.
    pub lock: bool,
    /// The NVMe tier's arena is pinned memory (`BLOOMERY_NVTIER_MEM` on the
    /// plan's device), not anonymous.
    pub arena_pinned: bool,
}

/// An addend of [`HostNeed::bytes`], the reserves split by what they set
/// aside, [`NeedTerm::Extra`] and [`NeedTerm::PartReads`] the one of the two
/// the need's peak holds. [`NeedTerm::class`] is the one table of which page
/// class holds each: a term added to the need is added here, and the match is
/// exhaustive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeedTerm {
    /// The routed experts the host keeps ([`HostNeed::experts`]): file pages
    /// of the shards, or of the r8 sidecar.
    Experts,
    /// The row-gathered tables the host holds ([`HostNeed::tables`]): the
    /// same mapping as the experts.
    Tables,
    /// The cards' pinned ring buffers and load-time upload staging ([`HostNeed::shadows`]).
    Shadows,
    /// The expert tier's prompt batch rows and places
    /// ([`TIER_BATCH_HOST_RESERVE`]), allocated with `cuMemAllocHost`.
    TierRows,
    /// The checkpoint slots a load's sequences keep
    /// ([`CHECKPOINTS_RESERVE`]), pinned host buffers.
    Checkpoints,
    /// The hidden rows of those checkpoints under the MTP draft
    /// ([`POINT_ROWS_RESERVE`]), pinned host buffers.
    PointRows,
    /// The expert stream's lane staging ([`XSTREAM_STAGING_RESERVE`]),
    /// allocated with `cuMemHostAlloc`.
    StreamStaging,
    /// The page cache the row-gathered tables on the NVMe tier are read
    /// through (`HostTotals::row_reserve_bytes`).
    RowCache,
    /// The page-cache window of the prompt run-ahead of the NVMe tier
    /// (what [`HostNeed::reserves`] holds past the named reserves and the
    /// row cache).
    RunAhead,
    /// The residency churn pool ([`HostNeed::extra`]): runs of the host set,
    /// read in and locked with it.
    Extra,
    /// Owned part results and read/normalization scratch ([`HostNeed::read_ahead`]).
    PartReads,
    /// The NVMe tier's RAM arena ([`HostNeed::arena`]).
    Arena,
    /// The cards' share of a unified pool ([`HostNeed::cards`]).
    Cards,
}

impl NeedTerm {
    /// Every term, in the order [`HostNeed::classes`] adds them.
    pub const ALL: [NeedTerm; 13] = [
        NeedTerm::Experts,
        NeedTerm::Tables,
        NeedTerm::Shadows,
        NeedTerm::TierRows,
        NeedTerm::Checkpoints,
        NeedTerm::PointRows,
        NeedTerm::StreamStaging,
        NeedTerm::RowCache,
        NeedTerm::RunAhead,
        NeedTerm::Extra,
        NeedTerm::PartReads,
        NeedTerm::Arena,
        NeedTerm::Cards,
    ];

    /// The page class of the term under `how`.
    #[must_use]
    pub const fn class(self, how: Hosting) -> PageClass {
        match self {
            NeedTerm::Experts | NeedTerm::Tables | NeedTerm::Extra => {
                if how.lock {
                    PageClass::Locked
                } else {
                    PageClass::Cache
                }
            }
            NeedTerm::Shadows
            | NeedTerm::TierRows
            | NeedTerm::Checkpoints
            | NeedTerm::PointRows
            | NeedTerm::StreamStaging => PageClass::Pinned,
            NeedTerm::RowCache | NeedTerm::RunAhead => PageClass::Cache,
            NeedTerm::PartReads => PageClass::Anon,
            NeedTerm::Arena => {
                if how.arena_pinned {
                    PageClass::Pinned
                } else {
                    PageClass::Anon
                }
            }
            NeedTerm::Cards => PageClass::Unified,
        }
    }
}

/// The term a host reserve named `name` is; `None` for [`OS_RESERVE`], which
/// the need leaves out. A name with no row is refused by [`HostNeed::classes`]:
/// a reserve nobody classed would be counted in no class.
fn reserve_term(name: &str) -> Result<Option<NeedTerm>, NeedClassError> {
    match name {
        OS_RESERVE => Ok(None),
        TIER_BATCH_HOST_RESERVE => Ok(Some(NeedTerm::TierRows)),
        CHECKPOINTS_RESERVE => Ok(Some(NeedTerm::Checkpoints)),
        POINT_ROWS_RESERVE => Ok(Some(NeedTerm::PointRows)),
        XSTREAM_STAGING_RESERVE => Ok(Some(NeedTerm::StreamStaging)),
        other => Err(NeedClassError::Reserve(other.to_string())),
    }
}

/// [`HostNeed::bytes`] by page class: the classes sum to it, the OS reserve
/// being out of both. A class past u64 bytes reads `u64::MAX`, as the sum
/// does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NeedClasses {
    pub pinned: u64,
    pub anon: u64,
    pub cache: u64,
    pub locked: u64,
    pub unified: u64,
}

impl NeedClasses {
    fn add(&mut self, class: PageClass, bytes: u64) {
        let slot = match class {
            PageClass::Pinned => &mut self.pinned,
            PageClass::Anon => &mut self.anon,
            PageClass::Cache => &mut self.cache,
            PageClass::Locked => &mut self.locked,
            PageClass::Unified => &mut self.unified,
        };
        *slot = slot.saturating_add(bytes);
    }

    /// The classes' sum, [`HostNeed::bytes`] of the need they were made from.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        let sum: u128 = [
            self.pinned,
            self.anon,
            self.cache,
            self.locked,
            self.unified,
        ]
        .into_iter()
        .map(u128::from)
        .sum();
        u64::try_from(sum).unwrap_or(u64::MAX)
    }
}

/// Why a need could not be put in classes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NeedClassError {
    /// A host reserve with no page class ([`reserve_term`]).
    Reserve(String),
    /// The need's reserves hold less than the machine's named reserves and
    /// the plan's row cache, or its OS reserve is not the host's: the need
    /// was not made from this plan.
    NotThePlans { why: String },
}

impl fmt::Display for NeedClassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeedClassError::Reserve(name) => write!(
                f,
                "the host reserve {name:?} has no page class: add it to workstation::reserve_term"
            ),
            NeedClassError::NotThePlans { why } => {
                write!(f, "the host need is not this plan's: {why}")
            }
        }
    }
}

impl std::error::Error for NeedClassError {}

impl HostNeed {
    /// The need by page class under `how`: each addend of [`HostNeed::bytes`]
    /// in the class [`NeedTerm::class`] gives it, the OS reserve out, and of
    /// the churn pool and the part reads the larger, the one the peak holds. `self`
    /// is [`HostNeed::of`] of `plan`; a need made of another plan, or edited
    /// since, is refused by name.
    pub fn classes(&self, plan: &Plan<'_>, how: Hosting) -> Result<NeedClasses, NeedClassError> {
        let host = plan.machine.host();
        let os = os_reserve(host);
        if self.os != os {
            return Err(NeedClassError::NotThePlans {
                why: format!("its OS reserve is {} B, the host's {os} B", self.os),
            });
        }
        let mut classes = NeedClasses::default();
        let mut add = |term: NeedTerm, bytes: u64| classes.add(term.class(how), bytes);
        let mut named: u128 = 0;
        // The part reads' reserve is the need's `read_ahead` ([`PART_READS`]), classed below.
        for (name, bytes) in host.reserves.iter().filter(|(name, _)| name != PART_READS) {
            named += u128::from(*bytes);
            if let Some(term) = reserve_term(name)? {
                add(term, *bytes);
            }
        }
        let rows = plan.host.row_reserve_bytes;
        let window = u128::from(self.reserves)
            .checked_sub(named + u128::from(rows))
            .and_then(|w| u64::try_from(w).ok())
            .ok_or_else(|| NeedClassError::NotThePlans {
                why: format!(
                    "its reserves hold {} B, under the host's named reserves {named} B and the \
                     row cache {rows} B",
                    self.reserves
                ),
            })?;
        add(NeedTerm::Experts, self.experts);
        add(NeedTerm::Tables, self.tables);
        add(NeedTerm::Shadows, self.shadows);
        add(NeedTerm::RowCache, rows);
        add(NeedTerm::RunAhead, window);
        if self.extra >= self.read_ahead {
            add(NeedTerm::Extra, self.extra);
        } else {
            add(NeedTerm::PartReads, self.read_ahead);
        }
        add(NeedTerm::Arena, self.arena);
        add(NeedTerm::Cards, self.cards);
        Ok(classes)
    }
}

/// Which reading of the host's available bytes decided ([`host_room`]):
/// the platform's own, or the room under a cgroup v2 memory limit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostRead {
    /// `/proc/meminfo`'s `MemAvailable`.
    MemAvailable,
    /// The host's free and inactive pages, as the platform's
    /// virtual-memory statistics report them (macOS's
    /// `host_statistics64(HOST_VM_INFO64)`).
    VmStats,
    /// The room under the cgroup v2 memory limit at `path`, as
    /// [`cgroup_v2_path`] names it under `/sys/fs/cgroup`.
    Cgroup { path: String },
    /// A room the caller gave instead of reading this machine (a gate that
    /// plans an arm by the room it names).
    Given,
    /// The machine's own figure: its host's usable bytes less the OS reserve
    /// ([`HostNeed::machine_room`]), or its unified pool at plan time, where
    /// that is under the reading's plan room ([`Plan::plan_at`]).
    Machine,
}

impl fmt::Display for HostRead {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostRead::MemAvailable => write!(f, "MemAvailable in /proc/meminfo"),
            HostRead::VmStats => write!(f, "the host's free and inactive pages (vm_statistics64)"),
            HostRead::Cgroup { path } => write!(
                f,
                "the room under the cgroup v2 memory limit at /sys/fs/cgroup{path} \
                 (memory.max − memory.current + active_file + inactive_file)"
            ),
            HostRead::Given => write!(
                f,
                "given by {} or the caller, not read from this machine",
                bloomery_levers::HOST_ROOM
            ),
            HostRead::Machine => write!(
                f,
                "the machine's figure, its host's usable bytes less the OS reserve or its unified \
                 pool, under the reading"
            ),
        }
    }
}

impl HostRead {
    /// One word for a `key=value` line: `MemAvailable`, `vmstats`,
    /// `cgroup:<path>`, `given`, or `machine`.
    #[must_use]
    pub fn word(&self) -> String {
        match self {
            HostRead::MemAvailable => "MemAvailable".to_string(),
            HostRead::VmStats => "vmstats".to_string(),
            HostRead::Cgroup { path } => format!("cgroup:{path}"),
            HostRead::Given => "given".to_string(),
            HostRead::Machine => "machine".to_string(),
        }
    }
}

/// The host bytes a load's own start may take between the reading its plan
/// is made at and the load's check ([`HostNeed::grant`]): a chosen bound,
/// not a derived one. A plan made at a read room keeps it ([`plan_room`]),
/// so a split that fills its room exactly still passes the check at the
/// load, and so does an unset residency climb's room
/// ([`HostNeed::room_left`]), so a churn pool that fills it exactly does.
pub const HOST_DRIFT: u64 = 512 * MIB;

/// The room a plan is made at, for the room `room` that [`host_room`] (or a
/// caller) gave: a [`HostRead::Given`] or [`HostRead::Machine`] room as given, a read room less
/// [`HOST_DRIFT`] (to 0 at most). The one place a read room becomes a
/// plan's room.
#[must_use]
pub fn plan_room(room: &(u64, HostRead)) -> u64 {
    match room.1 {
        HostRead::Given | HostRead::Machine => room.0,
        HostRead::MemAvailable | HostRead::VmStats | HostRead::Cgroup { .. } => {
            room.0.saturating_sub(HOST_DRIFT)
        }
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

/// The host's available bytes and the cgroup v2 levels above the process
/// (its own cgroup first, every ancestor after): the smaller of `available`
/// — the platform's own reading of the host's memory, named by `base`
/// ([`HostRead`]) — and every level's room ([`cgroup_room`]) — inside a
/// container (`docker run --memory`), a systemd scope (`MemoryMax`) or a
/// pod, the platform's reading still shows the whole machine, so a limit's
/// room is the reading that binds — with the reading that decided
/// ([`HostRead`]), for every refusal and line that prints the bytes to
/// name. A level that sets no limit does not bind; no levels is the
/// platform's reading alone; a tie leaves the platform's reading the named
/// one.
pub fn host_room(
    available: u64,
    base: HostRead,
    levels: &[CgroupLevel<'_>],
) -> Result<(u64, HostRead), String> {
    let mut best = (available, base);
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
        A6000, ArenaGrant, CONTEXT, Card, CardSpec, CgroupLevel, DRAFT_RESERVE, Device, DeviceId,
        DeviceInfo, GRANULE, HOST_DRIFT, Host, HostNeed, HostRead, MARGIN, Machine, NO_TIER_BATCH,
        Node, PLAN_ROOM, PlacementError, RTX_3090, SCRATCH, TIER_BATCH_HOST_RESERVE,
        TIER_BATCH_RESERVE, TierBatchBytes, UnifiedPool, card, cgroup_room, cgroup_v2_path,
        counted_arena_bytes, draft_reserve, gb10, holder_card, host, host_of, host_room,
        placed_with_arena, plan_a, plan_a_draft, plan_bp, plan_gate, plan_holding, plan_node_tiers,
        plan_on, plan_room, plan_tiers, spec_of_device, tier_batch_bytes, unified_of,
    };

    #[test]
    fn tier_input_scales_cost_one_f32_per_chunk_slot() {
        use crate::kernels::{DownAct, DownForms};

        let both = DownForms::of(DownAct::Q8Act).with(DownAct::Q8Blocks32);
        for chunk in [1, 8, 16] {
            for downs in [DownForms::NONE, DownForms::of(DownAct::Q8Act), both] {
                let direct =
                    super::try_tier_leg_batch_bytes_of(256, 256, 6, (chunk, 32), downs, false)
                        .expect("direct descriptor");
                let scaled =
                    super::try_tier_leg_batch_bytes_of(256, 256, 6, (chunk, 32), downs, true)
                        .expect("scaled descriptor");
                assert_eq!(scaled.staging, direct.staging);
                assert_eq!(scaled.host, direct.host);
                assert_eq!(scaled.scratch, direct.scratch + 4 * chunk * 6);
                assert_eq!(
                    scaled.card().expect("scaled card"),
                    direct.card().expect("direct card") + 4 * chunk * 6,
                );
                let leg = scaled.leg.expect("scaled leg");
                assert!(leg.input_scales);
                assert_eq!(
                    leg.bytes(DownForms::NONE).expect("empty selected union"),
                    direct
                        .leg
                        .expect("direct leg")
                        .bytes(DownForms::NONE)
                        .expect("empty union")
                        + 4 * chunk * 6,
                );
            }
        }
    }

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
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        }
    }

    fn base_bp(layers: usize, draft_bytes: Option<u64>, batch: TierBatchBytes) -> Machine {
        let mut t = card(RTX_3090, 0..0, false);
        t.reserves.push((
            TIER_BATCH_RESERVE.to_string(),
            batch.card().expect("card reserve"),
        ));
        t.reserves
            .extend(draft_bytes.map(|b| (DRAFT_RESERVE.to_string(), b)));
        let mut h = host();
        h.reserves
            .push((TIER_BATCH_HOST_RESERVE.to_string(), batch.host));
        Machine {
            cards: vec![card(A6000, 0..layers, true)],
            tiers: vec![t],
            nodes: vec![Node {
                host: h,
                unified: None,
            }],
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
                leg: None,
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
                    let on = draft.map(|bytes| draft_reserve(Device::Card(1), bytes));
                    let tiers = plan_tiers(layers, A6000, &[RTX_3090], on, batch)
                        .unwrap_or_else(|e| panic!("{layers} {draft:?}: {e}"));
                    assert_eq!(tiers, base, "{layers} {draft:?}");
                }
            }
        }
    }

    /// A one-card plan asked for the DSpark draft names [`DRAFT_RESERVE`] on
    /// its stage card and nowhere else: the plan is [`plan_a`] with that one
    /// reserve, and no other field moves.
    #[test]
    fn place_a_names_the_draft_or_refuses() {
        let bytes = 7;
        assert_eq!(plan_a_draft(43, None).expect("no draft"), plan_a(43));
        let m = match plan_a_draft(43, Some(bytes)) {
            Ok(m) => m,
            other => panic!("{other:?}, not a plan with the draft on its stage card"),
        };
        assert_eq!(m.cards[0].reserves, [(DRAFT_RESERVE.to_string(), bytes)]);
        let mut want = plan_a(43);
        want.cards[0].reserves = m.cards[0].reserves.clone();
        assert_eq!(m, want, "the draft's reserve is the only difference");
    }

    /// The draft's card is its place in the whole device list, the stage
    /// first: on the stage it leaves the tier cards' reserves as they were,
    /// on a tier it leaves the stage's, and the batch reserve stays on every
    /// tier either way.
    #[test]
    fn the_draft_is_reserved_on_the_card_its_index_names() {
        let batch = tier_batch_bytes(5120, 2304, 6, 512);
        let at = |on: usize| {
            let reserve = Some(draft_reserve(Device::Card(on), 7));
            plan_tiers(43, A6000, &[RTX_3090], reserve, batch).expect("a card of the list")
        };
        let named = [(DRAFT_RESERVE.to_string(), 7)];
        let tier_batch = [(
            TIER_BATCH_RESERVE.to_string(),
            batch.card().expect("card reserve"),
        )];
        let on_stage = at(0);
        assert_eq!(on_stage.cards[0].reserves, named);
        assert_eq!(on_stage.tiers[0].reserves, tier_batch);
        let on_tier = at(1);
        assert!(on_tier.cards[0].reserves.is_empty());
        assert_eq!(on_tier.tiers[0].reserves.len(), 2);
        assert_eq!(
            on_tier.tiers[0].reserve_bytes(),
            Some(7 + batch.card().expect("card reserve"))
        );
        assert_eq!(on_tier, plan_bp(43, Some(7), batch));
    }

    /// A holder joins the plan after the tiers as a reserve-only member: the
    /// machine is [`plan_a`]'s with that member listed and the draft reserved
    /// on it, no batch service on it or on the host, and the draft's index runs
    /// over the stage, the tiers and the holders. A holder that is the stage's
    /// device is listed twice, and an index past the holders is off the plan.
    #[test]
    fn a_holder_is_a_member_with_the_draft_and_nothing_else() {
        let held = |on: usize| {
            plan_holding(
                43,
                A6000,
                &[],
                &[RTX_3090],
                Some(draft_reserve(Device::Card(on), 7)),
                NO_TIER_BATCH,
            )
        };
        let machine = held(1).expect("the holder is card 1");
        let mut member = holder_card(RTX_3090);
        member.reserves.push((DRAFT_RESERVE.to_string(), 7));
        assert_eq!(machine.tiers, [member]);
        assert_eq!(machine.cards, plan_a(43).cards, "the stage is plan (a)'s");
        assert_eq!(
            machine.nodes,
            plan_a(43).nodes,
            "no tier service on the host"
        );
        assert_eq!(machine.expert_tiers().count(), 0);
        let on_stage = held(0).expect("the stage is card 0");
        assert_eq!(on_stage.cards[0].reserves, [(DRAFT_RESERVE.to_string(), 7)]);
        assert!(on_stage.tiers[0].reserves.is_empty());
        match held(2) {
            Err(PlacementError::ReserveOffPlan { cards: 2, .. }) => {}
            other => panic!("{other:?}, not ReserveOffPlan"),
        }
        match plan_holding(43, A6000, &[], &[A6000], None, NO_TIER_BATCH) {
            Err(PlacementError::DeviceTwice { cards: 2, .. }) => {}
            other => panic!("{other:?}, not DeviceTwice"),
        }
        assert_eq!(
            plan_holding(43, A6000, &[], &[], None, NO_TIER_BATCH).expect("no holder"),
            plan_a(43)
        );
    }

    /// `plan_tiers` refuses a card listed twice and a draft on a tier past
    /// the list by name; the 3090 as the stage with the A6000 its tier plans.
    #[test]
    fn plan_tiers_refuses_by_name() {
        let batch = tier_batch_bytes(5120, 2304, 6, 512);
        let on_card = |on| Some(draft_reserve(Device::Card(on), 7));
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
        match plan_tiers(43, A6000, &[RTX_3090], on_card(2), batch) {
            Err(e @ PlacementError::ReserveOffPlan { cards: 2, .. }) => assert_eq!(
                e.to_string(),
                "the DSpark draft reserve is on Card(2), outside the plan: a reserve sits on \
                 one of the plan's 2 cards, stage first, or on the host"
            ),
            other => panic!("{other:?}, not ReserveOffPlan"),
        }
        assert!(matches!(
            plan_tiers(43, A6000, &[], on_card(1), batch),
            Err(PlacementError::ReserveOffPlan { cards: 1, .. })
        ));
        assert!(plan_tiers(43, A6000, &[], on_card(0), batch).is_ok());
        let huge = TierBatchBytes {
            staging: u64::MAX,
            scratch: 1,
            host: 0,
            leg: None,
        };
        for result in [
            plan_tiers(4, A6000, &[RTX_3090], None, huge).map(|_| ()),
            plan_node_tiers(4, A6000, &[RTX_3090], huge).map(|_| ()),
        ] {
            let error = result.expect_err("the card reserve passes u64 bytes");
            assert_eq!(
                error.to_string(),
                format!(
                    "the tier card's prompt-batch staging of {} B and scratch of 1 B pass u64 bytes",
                    u64::MAX
                )
            );
        }
        assert_eq!(
            plan_tiers(4, A6000, &[], None, huge).expect("no tier"),
            plan_on(A6000, 4)
        );
        assert_eq!(
            plan_node_tiers(4, A6000, &[], huge).expect("no remote"),
            vec![plan_on(A6000, 4)]
        );
        let swapped = plan_tiers(43, RTX_3090, &[A6000], on_card(1), batch).expect("3090+a6000");
        assert_eq!(swapped.cards[0].name, RTX_3090.name);
        assert_eq!(swapped.tiers.len(), 1);
        assert_eq!(swapped.tiers[0].name, A6000.name);
        assert_eq!(
            swapped.tiers[0].reserve_bytes(),
            Some(7 + batch.card().expect("card reserve"))
        );
    }

    #[test]
    fn plan_node_tiers_keeps_each_census_and_service_on_its_node() {
        let mut specs = [
            gb10(100 << 30),
            gb10(80 << 30),
            gb10(60 << 30),
            gb10(40 << 30),
        ];
        for (i, spec) in specs.iter_mut().enumerate() {
            spec.device = Some(DeviceId {
                ordinal: 0,
                uuid: [u8::try_from(i).expect("four nodes"); 16],
            });
        }
        let batch = TierBatchBytes {
            staging: 10,
            scratch: 5,
            host: 100,
            leg: None,
        };
        let machines = plan_node_tiers(4, specs[0], &specs[1..], batch)
            .expect("four nodes of the same card name and ordinal");
        assert_eq!(machines.len(), 4);
        for (i, (m, spec)) in machines.iter().zip(specs).enumerate() {
            assert_eq!(m.nodes.len(), 1);
            assert!(m.tiers.is_empty());
            let [card] = m.cards.as_slice() else {
                panic!("one card per node: {m:?}");
            };
            assert_eq!(card.device, spec.device);
            assert_eq!(card.free_bytes, spec.free_bytes);
            assert_eq!(card.usable_bytes, spec.usable_bytes());
            assert_eq!(card.layers, 0..4);
            super::super::Stages::new(4, &m.cards).expect("each node has a complete card stage");
            let free = spec.free_bytes.expect("the census read the pool");
            assert_eq!(m.unified(), Some(UnifiedPool { bytes: free }));
            assert_eq!(m.host().usable_bytes, free);
            if i == 0 {
                assert!(card.reserves.is_empty());
                assert_eq!(
                    m.host().reserves,
                    vec![(TIER_BATCH_HOST_RESERVE.to_string(), 300)]
                );
            } else {
                assert!(m.host().reserves.is_empty());
                assert_eq!(card.reserves, vec![(TIER_BATCH_RESERVE.to_string(), 15)]);
            }
        }
        for stage in [A6000, specs[0]] {
            assert_eq!(
                plan_node_tiers(4, stage, &[], batch).expect("one node"),
                vec![plan_on(stage, 4)]
            );
        }
        assert!(plan_node_tiers(4, specs[0], &[specs[0]], batch).is_ok());
    }

    #[test]
    fn plan_node_tiers_refuses_service_off_plan() {
        let batch = TierBatchBytes {
            staging: 0,
            scratch: 0,
            host: u64::MAX / 2,
            leg: None,
        };
        let remotes = [gb10(80 << 30), gb10(60 << 30)];
        let machines =
            plan_node_tiers(4, gb10(100 << 30), &remotes, batch).expect("the host-row sum fits");
        assert_eq!(
            machines[0].host().reserves,
            vec![(TIER_BATCH_HOST_RESERVE.to_string(), u64::MAX - 1)]
        );
        let huge = TierBatchBytes {
            host: batch.host + 1,
            ..batch
        };
        assert!(matches!(
            plan_node_tiers(4, gb10(100 << 30), &remotes, huge),
            Err(PlacementError::TierBatchHost { tiers: 2, .. })
        ));
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
            leg: None,
        };
        let m = plan_tiers(4, A6000, &[t1, t2], None, batch).expect("two tiers");
        let rows: Vec<&(String, u64)> = m
            .host()
            .reserves
            .iter()
            .filter(|(n, _)| n == TIER_BATCH_HOST_RESERVE)
            .collect();
        assert_eq!(rows, vec![&(TIER_BATCH_HOST_RESERVE.to_string(), 200)]);
        assert!(m.tiers.iter().all(|t| t.reserve_bytes() == Some(15)));
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
            compute_capability: Some((12, 1)),
            granule_bytes: None,
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
                (m.unified(), m.host()),
                (pool, &figure),
                "integrated {integrated}"
            );
            m.host_mut().reserves.push(pinned.clone());
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
    /// embedding on its one card at the m = 1 scratch plus the program's
    /// arena, which the plan counts back ([`counted_arena_bytes`]), with this
    /// machine's context, margin and granule and the spec's host.
    #[test]
    fn a_placed_load_holds_the_whole_model_at_its_scratch() {
        for spec in [RTX_3090, A6000] {
            let m = placed_with_arena(spec, 40, 7);
            assert_eq!(counted_arena_bytes(&m.cards[0]), 7, "{}", spec.name);
            let want = Card {
                name: spec.name.to_string(),
                tier_leg: None,
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
                reserve_only: false,
                reserves: Vec::new(),
            };
            assert_eq!(m.cards, vec![want], "{}", spec.name);
            assert!(m.tiers.is_empty(), "{}", spec.name);
            assert_eq!(
                (m.host(), m.unified()),
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

    #[test]
    #[should_panic(expected = "tier batch card reserve: the tier card's prompt-batch staging")]
    fn tier_card_refuses_an_overflowing_batch() {
        let _ = super::tier_card(
            A6000,
            TierBatchBytes {
                staging: u64::MAX,
                scratch: 1,
                host: 0,
                leg: None,
            },
        );
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
        assert_eq!(b.card().expect("card reserve"), 152_511_496);
        let edge = TierBatchBytes {
            staging: u64::MAX - 1,
            scratch: 1,
            host: 0,
            leg: None,
        };
        assert_eq!(edge.card().expect("the sum fits u64"), u64::MAX);
        assert!(matches!(
            TierBatchBytes {
                staging: u64::MAX,
                ..edge
            }
            .card(),
            Err(PlacementError::TierBatchCard {
                staging: u64::MAX,
                scratch: 1
            })
        ));
        assert_eq!(b.host, 2 * (62_914_560 + 12_288));
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

    /// The host's available bytes are the smaller of the platform's reading
    /// and every cgroup v2 level's room, the deciding reading named: no
    /// levels and a level without a limit leave the platform's reading; a
    /// nested limit binds through the smaller parent; a tie names the
    /// platform's reading.
    #[test]
    fn host_room_takes_the_smaller_reading() {
        let mem = 257_300_180 * 1024;
        assert_eq!(
            host_room(mem, HostRead::MemAvailable, &[]),
            Ok((mem, HostRead::MemAvailable))
        );
        let unlimited = [level("/", "max\n", "0\n", (0, 0))];
        assert_eq!(
            host_room(mem, HostRead::MemAvailable, &as_levels(&unlimited)),
            Ok((mem, HostRead::MemAvailable))
        );
        // The scope's limit binds: the room is under the platform's reading.
        let scope = [level(
            "/user.slice/session-3.scope",
            "65539000000\n",
            "1073741824\n",
            (512 << 20, 512 << 20),
        )];
        assert_eq!(
            host_room(mem, HostRead::MemAvailable, &as_levels(&scope)),
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
        let (bytes, read) =
            host_room(mem, HostRead::MemAvailable, &as_levels(&nested)).expect("the room");
        assert_eq!(
            (bytes, &read.to_string()),
            (
                65_539_000_000,
                &"the room under the cgroup v2 memory limit at /sys/fs/cgroup/user.slice \
                 (memory.max − memory.current + active_file + inactive_file)"
                    .to_owned()
            )
        );
        // A tie leaves the platform's reading the named one.
        let tie = [level("/scope", &format!("{}\n", mem), "0\n", (0, 0))];
        assert_eq!(
            host_room(mem, HostRead::MemAvailable, &as_levels(&tie)),
            Ok((mem, HostRead::MemAvailable))
        );
    }

    /// A room this machine's `MemAvailable` gave.
    fn read(bytes: u64) -> (u64, HostRead) {
        (bytes, HostRead::MemAvailable)
    }

    /// The need is the plan's host terms less the OS reserve; exactly the
    /// available bytes pass, one byte more is refused, and the refusal names
    /// both figures, the reading that gave the room and the levers that lower
    /// the need. A room `BLOOMERY_HOST_ROOM` gave is named as given, with the
    /// lever to raise, not as this machine's shortage.
    #[test]
    fn host_need_is_refused_past_mem_available() {
        let need = HostNeed {
            experts: 1000,
            tables: 100,
            shadows: 10,
            reserves: 500,
            extra: 50,
            read_ahead: 0,
            arena: 0,
            cards: 0,
            os: 400,
            card_budget: true,
        };
        assert_eq!(need.bytes(), 1260);
        assert_eq!(need.check(&read(1260)), Ok(()));
        let short = need.check(&read(1259)).expect_err("one byte short");
        let text = short.to_string();
        for part in [
            "the host room is 1259 B (MemAvailable in /proc/meminfo)",
            "the 1260 B this load needs by 1 B",
            "residency churn pool 50 B",
            "the OS reserve 400 B",
            "Free host memory",
            "BLOOMERY_RESIDENCY=off",
            "BLOOMERY_CARD_BUDGET",
        ] {
            assert!(text.contains(part), "{part:?} in {text}");
        }
        assert!(!text.contains(bloomery_levers::HOST_ROOM), "{text}");
        let given = need
            .check(&(1259, HostRead::Given))
            .expect_err("one byte short of the given room")
            .to_string();
        for part in [
            "the host room is 1259 B (given by BLOOMERY_HOST_ROOM or the caller, not read from \
             this machine)",
            "the 1260 B this load needs by 1 B",
            "Raise or unset BLOOMERY_HOST_ROOM",
        ] {
            assert!(given.contains(part), "{part:?} in {given}");
        }
        assert!(!given.contains("Free host memory"), "{given}");
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
            read_ahead: 0,
            arena: 20_335_071_232,
            cards: 0,
            os: 6_694_629_376,
            card_budget: false,
        }
    }

    #[test]
    fn a_parts_load_reserves_its_read_ahead_before_allocating() {
        let base = unpaged_need();
        let need = HostNeed {
            read_ahead: 8 << 30,
            ..base
        };
        let available = base.bytes() + (1 << 30);
        assert!(base.check(&read(available)).is_ok());
        let short = need
            .grant(&read(available), 0)
            .expect_err("owned parts exceed the host limit");
        assert_eq!(short.need.read_ahead, 8 << 30);
        assert!(short.to_string().contains("part read-ahead 8589934592 B"));
        assert!(need.check(&read(base.bytes() + (8 << 30))).is_ok());
        assert!(need.check(&read(base.bytes() + (8 << 30) - 1)).is_err());
        let overflow = HostNeed {
            read_ahead: u64::MAX,
            ..base
        };
        assert!(overflow.check(&read(u64::MAX)).is_err());
    }

    /// A Qwen3.8 NVFP4 folder split on a GB10 under a 60 GB card budget with the residency
    /// unset: its host set reads the churn pool in after the card upload has ended the part
    /// reads, so the load holds the larger of the two, and the reads may take the room the pool
    /// takes once they end; the cards' share of the unified pool stays in both. A refusal at
    /// reads past the pool names no residency word, which would free nothing. Mutants: the two
    /// summed, which refuses this load; the residency hint on any pool.
    #[test]
    fn a_load_never_holds_its_part_reads_and_its_churn_pool_at_once() {
        let need = HostNeed {
            experts: 18_687_283_200,
            tables: 0,
            shadows: 100_663_296,
            reserves: 835_456_128,
            extra: 39_307_161_600,
            read_ahead: 10_171_207_688,
            arena: 0,
            cards: 64_139_335_376,
            os: 0,
            card_budget: true,
        };
        let room = read(123_551_232_000);
        let beside = 18_687_283_200 + 100_663_296 + 835_456_128 + 64_139_335_376;
        assert_eq!(need.bytes(), beside + need.extra);
        assert_eq!(need.check(&room), Ok(()));
        assert_eq!(need.read_room(room.0), room.0 - beside);
        let most = HostNeed {
            read_ahead: need.read_room(room.0),
            ..need
        };
        assert_eq!(most.bytes(), room.0);
        assert_eq!(most.check(&room), Ok(()));
        let past = HostNeed {
            read_ahead: most.read_ahead + 1,
            ..need
        };
        let text = past
            .check(&room)
            .expect_err("a byte past the room")
            .to_string();
        assert!(
            text.contains(
                "the larger of the residency churn pool 39307161600 B and the part read-ahead \
                 39788494001 B"
            ),
            "{text}"
        );
        assert!(text.contains("the cards' share of the unified pool 64139335376 B"));
        assert!(!text.contains("BLOOMERY_RESIDENCY=off"), "{text}");
        let pooled = HostNeed {
            extra: room.0 - beside + 1,
            ..need
        };
        let text = pooled
            .check(&room)
            .expect_err("a byte past the room")
            .to_string();
        assert!(
            text.contains("BLOOMERY_RESIDENCY=off drops the churn pool"),
            "{text}"
        );
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
            .grant(&read(PAGED_AVAILABLE), least)
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
        assert_eq!(shrunk.check(&read(PAGED_AVAILABLE)), Ok(()));
    }

    /// A shortfall past the arena's room above its floor is refused, naming
    /// the floor; the shortfall that leaves exactly the floor is granted; a
    /// plan with no arena is refused exactly as `check` refuses it.
    #[test]
    fn grant_refuses_a_shortfall_past_the_floor() {
        let need = paged_need();
        let least = 4 << 30;
        let room = need.arena - least;
        let at_floor = need
            .grant(&read(need.bytes() - room), least)
            .expect("the floor");
        assert_eq!(at_floor.granted, least);
        let past = need
            .grant(&read(need.bytes() - room - 1), least)
            .expect_err("one byte past the floor");
        assert_eq!(past.floor, Some(least));
        let text = past.to_string();
        assert!(
            text.contains(&format!("under its floor {least} B")),
            "the floor in {text}"
        );
        let bare = HostNeed { arena: 0, ..need };
        let short = bare.bytes() - 1;
        let refused = bare
            .grant(&read(short), least)
            .expect_err("no arena to shrink");
        assert_eq!(Some(&refused), bare.check(&read(short)).err().as_ref());
        assert_eq!(refused.floor, None);
        assert_eq!(
            bare.grant(&read(bare.bytes()), least),
            Ok(ArenaGrant::whole(0))
        );
    }

    /// A reading that covers the need grants the arena as planned, whatever
    /// the floor.
    #[test]
    fn grant_keeps_a_covered_arena_whole() {
        let need = paged_need();
        for available in [need.bytes(), need.bytes() + 1, u64::MAX] {
            for least in [0, 1 << 30, need.arena + 1] {
                assert_eq!(
                    need.grant(&read(available), least),
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
            read_ahead: 0,
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
            .check(&read(UNPAGED_AVAILABLE))
            .expect_err("the reading is short");
        assert_eq!(short.floor, None);
        assert_eq!(need.grant(&read(UNPAGED_AVAILABLE), 0), Err(short));
        assert_eq!(need.check(&read(need.bytes())), Ok(()));
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
            unpaged_need().check(&read(planned_at - UNPAGED_SHORT)),
            Ok(()),
            "the reporter's drift is under the margin"
        );
        assert!(
            unpaged_need()
                .check(&read(planned_at - HOST_DRIFT - 1))
                .is_err()
        );
    }

    /// An unset residency climb that fills its room exactly still passes the
    /// load after the reading falls by [`HOST_DRIFT`] or less. The climb
    /// budgets [`HostNeed::room_left`] at a read room ([`HostRead::MemAvailable`]);
    /// the load grants [`HostNeed::grant`] with that pool as `extra` and no
    /// arena to shrink, the call a placed load makes. One byte past the
    /// margin is refused. A given room is held as given, so the same fill
    /// is refused after one byte: a climb under a given room never sees the
    /// margin.
    #[test]
    fn an_unset_climb_that_fills_its_room_passes_the_load_after_drift() {
        let climb = unpaged_need();
        assert_eq!(climb.extra, 0);
        assert_eq!(climb.arena, 0);
        // One byte of pool past the margin, so the fill is exact either way.
        let reading_bytes = climb
            .bytes()
            .checked_add(HOST_DRIFT)
            .and_then(|b| b.checked_add(1))
            .expect("the reading stays in range");
        let reading = (reading_bytes, HostRead::MemAvailable);
        let mem_left = climb.room_left(&reading);
        let pool = u64::try_from(mem_left).expect("a read room leaves a pool");
        assert!(pool > 0, "the room leaves a pool to fill");
        let load = HostNeed {
            extra: pool,
            ..climb
        };
        let grants = |drop: u64| {
            let available = reading_bytes - drop;
            assert_eq!(
                load.check(&read(available)),
                load.grant(&read(available), 0).map(|_| ()),
                "with no arena, grant is check"
            );
            load.grant(&read(available), 0)
        };
        assert_eq!(
            grants(HOST_DRIFT),
            Ok(ArenaGrant::whole(0)),
            "a drop of the margin still grants"
        );
        assert_eq!(
            grants(HOST_DRIFT - 1),
            Ok(ArenaGrant::whole(0)),
            "a drop of one byte under the margin still grants"
        );
        assert_eq!(
            grants(1),
            Ok(ArenaGrant::whole(0)),
            "a drop of one byte still grants"
        );
        assert!(
            grants(HOST_DRIFT + 1).is_err(),
            "one byte past the margin is refused"
        );

        let given = (reading_bytes, HostRead::Given);
        let given_pool =
            u64::try_from(climb.room_left(&given)).expect("a given room leaves a pool");
        let given_load = HostNeed {
            extra: given_pool,
            ..climb
        };
        assert!(
            given_load
                .grant(&(reading_bytes - 1, HostRead::Given), 0)
                .is_err(),
            "a given room keeps no margin"
        );
    }
}
