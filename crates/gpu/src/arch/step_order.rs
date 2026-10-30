//! The order a layer's step meets the fault sites in, per architecture —
//! the order a site mask prints in — and the map from an architecture to its
//! own. Each table names every site exactly once, at its *first* occurrence on
//! the decode path: a site a step meets twice (the norm and the q8_1
//! quantizer run before attention and again around the experts) stands where
//! it is first met, and the sites an architecture never raises follow the
//! ones it does, so no bit of a mask goes unprinted. The embedding's
//! [`FaultSite::TokenId`] comes first: it is raised before any layer (as an
//! unlabelled launch).

use bloomery_device::fault::FaultSite;
use bloomery_device::fault::FaultSite::{
    AttnCount, AttnSel, CachePos, CacheValue, CandMask, DeltaLane, DeltaStamp, ExpertId,
    F32Product, HcMix, HcQuant, KeyCount, KvQuant, LinearConv, LinearDelta, LinearGate, Logit,
    Mamba2Scan, NormQuant, Ple, PoolSelect, Q5Quant, QuantColumn, Router, TokenId,
};
use bloomery_device::fault::StepOrder;

/// DeepSeek-V2-Lite (`arch::deepseek2`): the fused norm and quantizer at
/// the layer's entry, the key path's cache append, the attention
/// output's q8_1 quantizer (in the flash launch or its own), the router,
/// the card's slot list and the routed `_sel` kernels, the down
/// projection's 32-value quantizer.
pub const DEEPSEEK2: &[FaultSite] = &[
    TokenId,
    NormQuant,
    CachePos,
    KvQuant,
    QuantColumn,
    Router,
    ExpertId,
    Q5Quant,
    AttnCount,
    AttnSel,
    KeyCount,
    HcQuant,
    LinearConv,
    LinearDelta,
    LinearGate,
    DeltaLane,
    CacheValue,
    PoolSelect,
    Ple,
    HcMix,
    F32Product,
    DeltaStamp,
    Logit,
    CandMask,
    Mamba2Scan,
];
/// DeepSeek-V4.1 (`gpu-deepseek41`): HC_PRE's in-register quantizer, the
/// attention norm, the projections' quantizer, the attention's visible
/// counts and selected rows, then the MoE sub-layer's router and the
/// routed experts (the handoff's id check among them).
pub const DEEPSEEK41: &[FaultSite] = &[
    TokenId,
    HcQuant,
    NormQuant,
    QuantColumn,
    CandMask,
    AttnCount,
    AttnSel,
    Router,
    ExpertId,
    KeyCount,
    Q5Quant,
    CachePos,
    KvQuant,
    LinearConv,
    LinearDelta,
    LinearGate,
    DeltaLane,
    CacheValue,
    PoolSelect,
    Ple,
    HcMix,
    F32Product,
    DeltaStamp,
    Logit,
    Mamba2Scan,
];
/// Qwen3-MoE (`arch::qwen3moe`): the fused norm and quantizer at the
/// layer's entry, the cache append's position, the flash's key count,
/// the attention output's quantizer, the router's fused norm, the routed
/// experts. On the ubatch path the grouped GEMM's route (`ExpertId`) is
/// enqueued before the router's norm; the table follows the decode path.
pub const QWEN3MOE: &[FaultSite] = &[
    TokenId,
    NormQuant,
    CachePos,
    KvQuant,
    KeyCount,
    QuantColumn,
    Router,
    ExpertId,
    AttnCount,
    AttnSel,
    Q5Quant,
    HcQuant,
    LinearConv,
    LinearDelta,
    LinearGate,
    DeltaLane,
    CacheValue,
    PoolSelect,
    Ple,
    HcMix,
    F32Product,
    DeltaStamp,
    Logit,
    CandMask,
    Mamba2Scan,
];

/// Qwen3.6-35B-A3B and Qwen3.8-Flash-Next (`arch::qwen3moe`'s `Body35`
/// and `Body38`; both are `Arch::Qwen35moe`, so one order serves both).
/// Qwen3.6's two layer kinds read in their launch order: the fused norm
/// and quantizer at the layer's entry; an attention layer's cache append
/// and flash key count, or a delta layer's conv, delta step (its lane
/// word right after it) and gated norm; the output projection's
/// quantizer; then the router's fused norm and the routed experts.
/// Qwen3.8's sites stand beside them: its PLE site before its layer's
/// mix, its gated-residual mix before the rest of the layer, its f32
/// products before the output projection's quantizer. That is not every
/// Qwen3.8 layer's launch order; among the differences, a delta layer's
/// first [`F32Product`] is the shared expert's sum, raised after
/// [`Router`] and [`ExpertId`], and a selecting layer raises
/// [`PoolSelect`] (its raw key append first) before its out gate's
/// [`F32Product`]. The mask names every site raised either way; only the
/// order it prints them in differs.
/// A q8_0 head's logits read after every layer's sites.
pub const QWEN35MOE: &[FaultSite] = &[
    TokenId,
    Ple,
    HcMix,
    NormQuant,
    CachePos,
    KvQuant,
    KeyCount,
    LinearConv,
    LinearDelta,
    DeltaLane,
    DeltaStamp,
    LinearGate,
    F32Product,
    QuantColumn,
    Router,
    ExpertId,
    Logit,
    AttnCount,
    AttnSel,
    Q5Quant,
    HcQuant,
    PoolSelect,
    CacheValue,
    CandMask,
    Mamba2Scan,
];

/// GLM-5.3-Flash (`gpu-glm5next`): a KDA layer's conv, delta step (its
/// lane word right after it) and gated norm; a latent layer's two cache
/// appends and its attention's visible count; then the router and the
/// handoff's id check; the q8_0 head's logits last. The token id is checked on the host before the
/// step (the embedding is a host row), and the hyper-connection and
/// projection launches read f32 rows with nothing to raise; those sites
/// follow the ones the step raises.
pub const GLM5NEXT: &[FaultSite] = &[
    TokenId,
    LinearConv,
    LinearDelta,
    DeltaLane,
    LinearGate,
    CachePos,
    CacheValue,
    KvQuant,
    AttnCount,
    Router,
    ExpertId,
    Logit,
    HcQuant,
    NormQuant,
    QuantColumn,
    AttnSel,
    KeyCount,
    Q5Quant,
    PoolSelect,
    Ple,
    HcMix,
    F32Product,
    DeltaStamp,
    CandMask,
    Mamba2Scan,
];

/// [`MIMO2`]'s list, under a name no architecture owns, so a family with no
/// program, or whose step raises the same sites in the same order, can
/// decode a fault word without another family's word on the line.
const SHARED_ORDER: &[FaultSite] = MIMO2;

/// No program runs `nemotron_h`. The order is [`MIMO2`]'s so a fault word
/// still decodes; it is not a step this architecture runs.
pub const NEMOTRON_H: &[FaultSite] = SHARED_ORDER;

/// The Gemma encoders have no step order of their own. The order is
/// [`MIMO2`]'s so a fault word still decodes; it is not the order the
/// `gemma-embedding2` encoder's pass raises its sites in.
pub const GEMMA_ENCODER: &[FaultSite] = SHARED_ORDER;

/// No program runs `nemotron_h_moe`. The order is [`MIMO2`]'s so a fault
/// word still decodes; it is not a step this architecture runs.
pub const NEMOTRON_H_MOE: &[FaultSite] = SHARED_ORDER;

/// Muse Spark (`gpu/src/arch/musespark`): the token id checked on the
/// host before the step; a layer's cache append (position) and its flash
/// (visible key count); the router and the `_sel` launches' id check;
/// the norms' scales; the head's logits last. The order is [`MIMO2`]'s
/// (a valid decode of those raises), the same list it has always been.
pub const MUSE_SPARK: &[FaultSite] = SHARED_ORDER;

/// MiMo-V2.6-Flash (`gpu-mimo2`): the token id checked on the host before
/// the step; a layer's cache append (position) and its flash (visible
/// key count); the router and the handoff's id check; the q8_0 head's
/// logits last. The projections and the norms read f32 rows with
/// nothing to raise; those sites follow the ones the step raises.
pub const MIMO2: &[FaultSite] = &[
    TokenId,
    CachePos,
    KeyCount,
    Router,
    ExpertId,
    Logit,
    QuantColumn,
    NormQuant,
    Q5Quant,
    HcQuant,
    AttnSel,
    AttnCount,
    LinearConv,
    LinearDelta,
    LinearGate,
    DeltaLane,
    CacheValue,
    PoolSelect,
    Ple,
    HcMix,
    F32Product,
    DeltaStamp,
    CandMask,
    KvQuant,
    Mamba2Scan,
];

/// MiniMax-M3 (`gpu-minimaxm3`): the token id checked on the host before the step; a layer's cache append
/// (position), the sparse layers' block pick (the selector's counts and the ids it keeps) and the flash's
/// visible key count; the router and the handoff's id check; the q8_0 head's logits last. The projections and
/// the norms read f32 rows with nothing to raise; the sites the step never raises follow in [`MIMO2`]'s order.
pub const MINIMAXM3: &[FaultSite] = &[
    TokenId,
    CachePos,
    PoolSelect,
    KeyCount,
    Router,
    ExpertId,
    Logit,
    QuantColumn,
    NormQuant,
    Q5Quant,
    HcQuant,
    AttnSel,
    AttnCount,
    LinearConv,
    LinearDelta,
    LinearGate,
    DeltaLane,
    CacheValue,
    Ple,
    HcMix,
    F32Product,
    DeltaStamp,
    CandMask,
    KvQuant,
    Mamba2Scan,
];

/// Solar-Open2 (`gpu-solar2`): the token id checked on the host before
/// the step; a layer's norm and the q8_1 quantizer of its normed row, the
/// Q5_0 halves' 32-value quantizer, a KDA layer's conv, delta step (its
/// lane word and stamp right after it) and gated norm, a GQA layer's
/// cache append (position), flash (visible key count) and out gate's
/// product (the norm's [`F32Product`] already stands first); then the
/// router and the handoff's id check, and the head's logits last. The
/// order of the sites the step raises is the order a KDA layer's launches
/// meet them, then the sites only a GQA layer meets; the sites the step
/// never raises follow. Raon-1 (`arch::raon1`) runs the same order: an RDA
/// layer launches a KDA layer's sites, and a latent layer's cache append
/// and visible count stand where a GQA layer's do.
pub const SOLAR2: &[FaultSite] = &[
    TokenId,
    F32Product,
    QuantColumn,
    Q5Quant,
    LinearConv,
    LinearDelta,
    DeltaLane,
    DeltaStamp,
    LinearGate,
    CachePos,
    KeyCount,
    Router,
    ExpertId,
    Logit,
    NormQuant,
    HcQuant,
    AttnCount,
    AttnSel,
    CacheValue,
    KvQuant,
    PoolSelect,
    Ple,
    HcMix,
    CandMask,
    Mamba2Scan,
];

/// The order of `arch`'s step, named as the architecture's `Debug` form.
#[must_use]
pub fn of(arch: model::arch::Arch) -> StepOrder {
    use model::arch::Arch;
    let (name, sites) = match arch {
        Arch::Deepseek2 => ("Deepseek2", DEEPSEEK2),
        Arch::Deepseek41 => ("Deepseek41", DEEPSEEK41),
        Arch::Qwen3moe => ("Qwen3moe", QWEN3MOE),
        Arch::Qwen35moe => ("Qwen35moe", QWEN35MOE),
        Arch::Glm5next => ("Glm5next", GLM5NEXT),
        Arch::Mimo2 => ("Mimo2", MIMO2),
        Arch::SolarOpen2 => ("SolarOpen2", SOLAR2),
        Arch::Raon1 => ("Raon1", SOLAR2),
        Arch::MiniMaxM3 => ("MiniMaxM3", MINIMAXM3),
        // No program runs nemotron_h. The listing is MiMo's so a fault
        // word still decodes; it is not a step this architecture runs.
        Arch::NemotronH => ("NemotronH", NEMOTRON_H),
        Arch::NemotronHMoe => ("NemotronHMoe", NEMOTRON_H_MOE),
        Arch::GemmaEmbedding => ("GemmaEmbedding", GEMMA_ENCODER),
        Arch::GemmaEmbedding2 => ("GemmaEmbedding2", GEMMA_ENCODER),
        Arch::MuseSpark => ("MuseSpark", MUSE_SPARK),
        // Reader only, each of them: a neutral decoding order, not a
        // runnable step.
        Arch::Flint => ("Flint", FaultSite::ALL),
        Arch::MistralLarge4 => ("MistralLarge4", FaultSite::ALL),
    };
    StepOrder { arch: name, sites }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model::arch::Arch;

    /// Every architecture; a new one is a compile error here until it is
    /// listed (and [`of`] gives it a table).
    fn every_arch() -> [Arch; 16] {
        let all = [
            Arch::Deepseek2,
            Arch::Deepseek41,
            Arch::Qwen3moe,
            Arch::Qwen35moe,
            Arch::Glm5next,
            Arch::Mimo2,
            Arch::SolarOpen2,
            Arch::MiniMaxM3,
            Arch::Raon1,
            Arch::NemotronH,
            Arch::NemotronHMoe,
            Arch::GemmaEmbedding,
            Arch::GemmaEmbedding2,
            Arch::Flint,
            Arch::MuseSpark,
            Arch::MistralLarge4,
        ];
        for a in all {
            match a {
                Arch::Deepseek2
                | Arch::Deepseek41
                | Arch::Qwen3moe
                | Arch::Qwen35moe
                | Arch::Glm5next
                | Arch::Mimo2
                | Arch::SolarOpen2
                | Arch::MiniMaxM3
                | Arch::Raon1
                | Arch::NemotronH
                | Arch::NemotronHMoe
                | Arch::GemmaEmbedding
                | Arch::GemmaEmbedding2
                | Arch::Flint
                | Arch::MuseSpark
                | Arch::MistralLarge4 => {}
            }
        }
        all
    }

    #[test]
    fn every_step_order_names_every_site_exactly_once() {
        for arch in every_arch() {
            let order = of(arch).sites;
            assert_eq!(
                order.len(),
                FaultSite::ALL.len(),
                "{arch:?}'s step order has {} sites, FaultSite has {}",
                order.len(),
                FaultSite::ALL.len()
            );
            for site in FaultSite::ALL {
                let n = order.iter().filter(|s| *s == site).count();
                assert_eq!(n, 1, "{arch:?}'s step order names {site:?} {n} times");
            }
        }
    }

    /// A fault's print names the architecture as its `Debug` form does.
    #[test]
    fn every_step_order_is_named_as_its_architecture() {
        for arch in every_arch() {
            assert_eq!(of(arch).arch, format!("{arch:?}"));
        }
    }

    /// MiMo-V2.6-Flash's order opens with the six sites its step raises, in
    /// the order its launches meet them: the host's token check, the append's
    /// position, the flash's key count, the router, the handoff, the head.
    #[test]
    fn the_mimo2_order_leads_with_the_sites_its_step_raises() {
        assert_eq!(
            of(Arch::Mimo2).sites[..6],
            [
                FaultSite::TokenId,
                FaultSite::CachePos,
                FaultSite::KeyCount,
                FaultSite::Router,
                FaultSite::ExpertId,
                FaultSite::Logit,
            ]
        );
    }

    /// Solar-Open2's order opens with the fourteen sites its step raises, in
    /// the order a layer's launches meet them: the host's token check, the
    /// norm, the quantizers, the KDA conv, delta step (lane word, then stamp)
    /// and gated norm, the GQA append and flash, the router, the handoff, the
    /// head. The out gate's product is the norm's site, standing first.
    #[test]
    fn the_solar2_order_leads_with_the_sites_its_step_raises() {
        assert_eq!(
            of(Arch::SolarOpen2).sites[..14],
            [
                FaultSite::TokenId,
                FaultSite::F32Product,
                FaultSite::QuantColumn,
                FaultSite::Q5Quant,
                FaultSite::LinearConv,
                FaultSite::LinearDelta,
                FaultSite::DeltaLane,
                FaultSite::DeltaStamp,
                FaultSite::LinearGate,
                FaultSite::CachePos,
                FaultSite::KeyCount,
                FaultSite::Router,
                FaultSite::ExpertId,
                FaultSite::Logit,
            ]
        );
    }
}
