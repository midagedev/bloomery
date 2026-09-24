//! This workstation's figures for a plan (`docs/v41-placement.md` §4–§5): its
//! two cards, the host's RAM and reserves, the serving context, the two layer
//! maps of design §5 and the gate placement, and where the V4.1 file lies. The
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
    }
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
        host: host(),
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
        host: host(),
    }
}

/// The card spec a plan's card is named after.
#[must_use]
pub fn spec_of(card: &Card) -> Option<CardSpec> {
    [A6000, RTX_3090].into_iter().find(|s| s.name == card.name)
}
