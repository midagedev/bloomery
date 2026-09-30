//! This workstation's figures for a plan (`docs/v41-placement.md` §4–§5): its
//! two cards, the host's RAM and reserves, the serving context, the two layer
//! maps of design §5, the expert-tier map (b′) and the gate placement, and
//! where the V4.1 file lies. The
//! placement gate and the GPU load gates build their plans from here, so they
//! check the same plans.

use std::num::NonZeroU64;
use std::ops::Range;

use super::{Card, Host, Machine};

pub const MIB: u64 = 1 << 20;

/// One card as `nvidia-smi` reports it with no process on it [measured]: its
/// memory and the part the driver keeps.
#[derive(Clone, Copy, Debug)]
pub struct CardSpec {
    /// The name a plan gives the card; the CUDA device name contains it.
    pub name: &'static str,
    pub total_bytes: u64,
    pub driver_reserve_bytes: u64,
}

impl CardSpec {
    /// What the driver leaves for us.
    #[must_use]
    pub const fn usable_bytes(&self) -> u64 {
        self.total_bytes - self.driver_reserve_bytes
    }
}

/// The RTX A6000 [measured, `nvidia-smi`].
pub const A6000: CardSpec = CardSpec {
    name: "A6000",
    total_bytes: 49_140 * MIB,
    driver_reserve_bytes: 548 * MIB,
};

/// The RTX 3090 [measured, `nvidia-smi`].
pub const RTX_3090: CardSpec = CardSpec {
    name: "3090",
    total_bytes: 24_576 * MIB,
    driver_reserve_bytes: 400 * MIB,
};

/// Per card: the CUDA context and the m = 1 scratch [assumed], and the margin
/// the expert rule leaves free.
pub const CONTEXT: u64 = 512 * MIB;
pub const SCRATCH: u64 = 64 * MIB;
pub const MARGIN: u64 = 1 << 30;

/// Both cards' allocation granule [measured: what a load takes from
/// `cuMemGetInfo` is a whole number of them, and
/// `cuMemGetAllocationGranularity` reports the same size].
pub const GRANULE: NonZeroU64 = NonZeroU64::new(2 * MIB).expect("2 MiB is not zero");

/// The host's usable bytes, its engram row cache, and the OS and everything
/// else [measured, `free -b`].
pub const HOST_USABLE: u64 = 270_071_001_088;
pub const ROW_CACHE: u64 = 4_294_967_296;
pub const OS_OTHER: u64 = 6_694_629_376;

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
fn card(spec: CardSpec, layers: Range<usize>, head: bool) -> Card {
    Card {
        name: spec.name.to_string(),
        usable_bytes: spec.usable_bytes(),
        context_bytes: CONTEXT,
        scratch_bytes: SCRATCH,
        margin_bytes: MARGIN,
        granule_bytes: GRANULE,
        layers,
        head,
        token_embedding: false,
        reserves: Vec::new(),
    }
}

/// `spec` as an expert tier card: no stage, this machine's context, scratch
/// and margin.
fn tier(spec: CardSpec) -> Card {
    card(spec, 0..0, false)
}

/// The host tier with its reserves.
#[must_use]
pub fn host() -> Host {
    Host {
        usable_bytes: HOST_USABLE,
        reserves: vec![
            ("engram row cache".to_string(), ROW_CACHE),
            ("OS and other".to_string(), OS_OTHER),
        ],
    }
}

/// Design §5 (a): the A6000 runs all `layers` and the head; the 3090 is idle.
#[must_use]
pub fn plan_a(layers: usize) -> Machine {
    Machine {
        cards: vec![card(A6000, 0..layers, true)],
        tiers: Vec::new(),
        host: host(),
    }
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
#[must_use]
pub fn plan_bp(layers: usize, draft_bytes: Option<u64>, batch: TierBatchBytes) -> Machine {
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
    }
}

/// The gate placement: the 3090 runs all `layers` and the head, so the V4.1
/// body gates run on the gate card while the A6000 takes timing runs. Its
/// expert prefixes are the largest the card's budget allows — the expert
/// rule of [`super::plan`], on the same usable − KV − context − scratch −
/// margin arithmetic as [`plan_a`]'s card.
#[must_use]
pub fn plan_gate(layers: usize) -> Machine {
    Machine {
        cards: vec![card(RTX_3090, 0..layers, true)],
        tiers: Vec::new(),
        host: host(),
    }
}

/// The card spec a plan's card is named after.
#[must_use]
pub fn spec_of(card: &Card) -> Option<CardSpec> {
    [A6000, RTX_3090].into_iter().find(|s| s.name == card.name)
}

#[cfg(test)]
mod tests {
    use super::tier_batch_bytes;

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
}
