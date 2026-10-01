//! This workstation's figures for a plan (`docs/v41-placement.md` §4–§5): its
//! two cards, the host's RAM and reserves, the serving context, the two layer
//! maps of design §5, the expert-tier map (b′) and the gate placement, and
//! where the V4.1 file lies. The
//! placement gate and the GPU load gates build their plans from here, so they
//! check the same plans.

use std::fmt;
use std::num::NonZeroU64;
use std::ops::Range;

use super::{Card, Host, Machine, Plan};

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

/// What the host tier's two reserves are called: [`ROW_CACHE`], and
/// [`OS_OTHER`] — what the OS and every other process held when
/// [`HOST_USABLE`] was measured.
pub const ROW_CACHE_RESERVE: &str = "engram row cache";
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
            (ROW_CACHE_RESERVE.to_string(), ROW_CACHE),
            (OS_RESERVE.to_string(), OS_OTHER),
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

/// The tier's chunk scratch for blocks of rows of `n_embd`, `n_used` slots a
/// token, through experts of `ff`, cut in chunks of `chunk` tokens: the q8_1
/// of a chunk's rows, its gate·up rows a slot, and per chunk width `c` from
/// one to `chunk` the q8_1 of its `c · n_used` slots' columns — the scratch
/// of a tier that runs the stage card's chunked card path over a block (the
/// GLM-5.3-Flash tier), in place of the tile path's.
#[must_use]
pub const fn tier_chunk_scratch_bytes(n_embd: u64, ff: u64, n_used: u64, chunk: u64) -> u64 {
    let mut b = q8_act_bytes(chunk, n_embd) + 4 * chunk * n_used * ff;
    let mut c = 1;
    while c <= chunk {
        b += q8_act_bytes(c * n_used, ff);
        c += 1;
    }
    b
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

/// The one visible device a plan's card `card` runs on: the index in `seen`
/// (the visible devices' names, by ordinal) of the one whose name contains
/// the card's, as the GPU loader finds its card. None or several is
/// refused, naming what is in view and the placement that fits it.
pub fn card_on_host(card: &str, seen: &[String]) -> Result<usize, CardNotInView> {
    let named: Vec<usize> = (0..seen.len())
        .filter(|&i| seen[i].contains(card))
        .collect();
    match named.as_slice() {
        &[i] => Ok(i),
        _ => Err(CardNotInView {
            card: card.to_string(),
            seen: seen.to_vec(),
            named: named.len(),
        }),
    }
}

/// A plan's card that is not exactly one visible device ([`card_on_host`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardNotInView {
    pub card: String,
    /// The visible devices' names, by ordinal.
    pub seen: Vec<String>,
    /// How many of them are named like the card.
    pub named: usize,
}

impl fmt::Display for CardNotInView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let CardNotInView { card, seen, named } = self;
        if *named > 1 {
            return write!(
                f,
                "{named} visible devices are named like the plan's card {card} ({seen:?}): a plan \
                 finds each card by its name, so one device of a name may be in view; two 3090s \
                 are not a host this tree runs on (CUDA_VISIBLE_DEVICES keeps one in view)"
            );
        }
        write!(
            f,
            "no visible device is the plan's card {card} (visible: {seen:?})"
        )?;
        if card == A6000.name {
            write!(
                f,
                ". --place a, the default, runs the whole model on the A6000; this tree runs on \
                 an A6000, or an A6000 with a 3090, and not on two 3090s. On one 3090, --place \
                 gate plans the whole model on that card (its plan refuses by name when the card \
                 and the host cannot hold the model); an A6000 out of view \
                 (CUDA_VISIBLE_DEVICES) must be put in view"
            )
        } else if card == RTX_3090.name {
            write!(
                f,
                ". --place gate runs the whole model on a 3090 and --place bp puts one under the \
                 A6000's host tier; on an A6000 alone, --place a runs the model; a 3090 out of \
                 view (CUDA_VISIBLE_DEVICES) must be put in view"
            )
        } else {
            Ok(())
        }
    }
}

impl std::error::Error for CardNotInView {}

/// What a placed load needs of the host's memory, which `MemAvailable` must
/// cover ([`HostNeed::check`]): the plan's host experts and tables, the
/// cards' ring shadows, the host's reserves and `extra` (a residency churn
/// pool the host set also holds), less the reserve named [`OS_RESERVE`] —
/// `MemAvailable` already leaves out what the OS and every other process
/// hold. The page cache the host set will reuse is inside `MemAvailable`
/// already, as reclaimable file pages, so no term adds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostNeed {
    pub experts: u64,
    pub tables: u64,
    pub shadows: u64,
    pub reserves: u64,
    pub extra: u64,
    pub os: u64,
    /// The plan was made under a card budget.
    pub card_budget: bool,
}

impl HostNeed {
    /// `plan`'s host terms, with `extra` bytes the host set holds beside
    /// them.
    #[must_use]
    pub fn of(plan: &Plan<'_>, extra: u64) -> HostNeed {
        let os = plan
            .machine
            .host
            .reserves
            .iter()
            .filter(|(name, _)| name == OS_RESERVE)
            .map(|(_, b)| b)
            .sum();
        HostNeed {
            experts: plan.host.expert_bytes,
            tables: plan.host.table_bytes,
            shadows: plan.host.shadow_bytes,
            reserves: plan.host.reserve_bytes,
            extra,
            os,
            card_budget: plan.card_budget.is_some(),
        }
    }

    /// The bytes `MemAvailable` must cover.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        (self.experts + self.tables + self.shadows + self.reserves + self.extra)
            .saturating_sub(self.os)
    }

    /// `available` bytes (`MemAvailable`) cover the need, or the refusal
    /// that names both, each term and what lowers the need.
    pub fn check(self, available: u64) -> Result<(), HostShort> {
        if self.bytes() <= available {
            Ok(())
        } else {
            Err(HostShort {
                need: self,
                available,
            })
        }
    }
}

/// A placed load whose host need passes `MemAvailable` ([`HostNeed::check`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostShort {
    pub need: HostNeed,
    pub available: u64,
}

impl fmt::Display for HostShort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = &self.need;
        let need = n.bytes();
        write!(
            f,
            "the host has {} B available (MemAvailable in /proc/meminfo), under the {need} B \
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
        if n.card_budget {
            write!(
                f,
                "; BLOOMERY_CARD_BUDGET moves bytes from the cards to the host, so unset it"
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

#[cfg(test)]
mod tests {
    use super::{
        A6000, CardNotInView, HostNeed, RTX_3090, card_on_host, mem_available, tier_batch_bytes,
    };

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

    /// The card is the one visible device named like it; none or two is
    /// refused, and with no A6000 the refusal names `--place gate`.
    #[test]
    fn card_on_host_names_the_placement_that_fits() {
        let one = ["NVIDIA GeForce RTX 3090".to_string()];
        let both = [
            "NVIDIA GeForce RTX 3090".to_string(),
            "NVIDIA RTX A6000".to_string(),
        ];
        assert_eq!(card_on_host(A6000.name, &both), Ok(1));
        assert_eq!(card_on_host(RTX_3090.name, &both), Ok(0));
        let no_a6000 = card_on_host(A6000.name, &one).expect_err("no A6000 in view");
        assert_eq!(no_a6000.named, 0);
        assert!(no_a6000.to_string().contains("--place gate"), "{no_a6000}");
        let no_3090 = card_on_host(RTX_3090.name, &both[1..]).expect_err("no 3090 in view");
        assert!(no_3090.to_string().contains("--place a"), "{no_3090}");
        let two = [one[0].clone(), one[0].clone()];
        let e: CardNotInView = card_on_host(RTX_3090.name, &two).expect_err("two 3090s");
        assert_eq!(e.named, 2);
        assert!(e.to_string().contains("two 3090s"), "{e}");
    }
}
