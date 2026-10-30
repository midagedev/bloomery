//! What each layer program runs, by site and type: the one owner the plan's
//! rules and the loaders both read, so a plan its rule passes is a plan the
//! load takes. The card kernel table is [`kernels`]; a program's sets here
//! are the entries of it that its dispatch launches. The coverage check
//! (`model::arch::coverage`) names its items through these.

use gguf::GgmlType;
use models::Arch;

use crate::kernels::{
    self, DownAct, DownEntry, EmbedEntry, GateUpEntry, HeadEntry, SiteAct, SiteEntry,
};
use crate::placement::{CardFormat, LegPair, RoutedRule};

/// A layer program: the chain that runs a family's layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Program {
    /// The V4.1 chain (`crates/gpu-deepseek41`: `chain/`, `body.rs`).
    Deepseek41Chain,
    /// The whole-card qwen3moe body (`crates/gpu/src/arch/qwen3moe` `Body`).
    Qwen3moeBody,
    /// The whole-card qwen35moe, qwen35 and llama body
    /// (`crates/gpu/src/arch/qwen3moe` `Body35`): gated-delta-rule and GQA
    /// layers, its sites' types by `crate::site`.
    Qwen35Body,
    /// The glm5next body (`crates/gpu-glm5next`): the routed experts the plan
    /// puts on a card (the common pair) and the rest on the host tier.
    Glm5nextBody,
    /// The qwen4exp body (`crates/gpu/src/arch/qwen3moe` `Body38`): the
    /// routed experts the plan puts on a card (its own gate·up and down
    /// entries) and the rest on the host tier.
    Qwen38Body,
    /// The mimo2 body (`crates/gpu-mimo2`): the routed experts the plan puts
    /// on a card (MXFP4 stacks, the common pair) and the rest on the host
    /// tier.
    Mimo2Body,
    /// The solar2 body (`crates/gpu-solar2`): the routed experts the plan puts
    /// on a card (the common pair) and the rest on the host tier, beside a
    /// shared expert; its KDA and NoPE GQA layers run the common owners.
    Solar2Body,
    /// The minimax-m3 body (`crates/gpu-minimaxm3`): the routed experts the plan
    /// puts on a card (the common pair, or NVFP4 in its planes) and the rest on
    /// the host tier, beside a shared expert; its GQA layers select blocks by an
    /// index key (`attention.indexer.*`) and its experts run `swigluoai`.
    MiniMaxM3Body,
    /// The nemotron_h body (`crates/gpu/src/arch/nemotron_h`): whole-card,
    /// dense, its Mamba-2, NoPE GQA and squared-ReLU layers over the common
    /// kernels; no routed experts.
    NemotronHBody,
    /// The nemotron_h_moe body (`crates/gpu/src/arch/nemotron_h_moe`):
    /// whole-card, the Nano's Mamba-2 and NoPE GQA layers over the shared
    /// hybrid owner (`crates/gpu/src/arch/hybrid.rs`) and the latent MoE layer,
    /// whose routed stacks run [`latent_leg_runs`] and shared expert IQ2_XXS, Q8_0 or NVFP4.
    NemotronHMoeBody,
    /// The EmbeddingGemma encoders (`crates/gpu-gemma-embedding2`): one
    /// pass of a dense trunk on a card, no routed experts.
    GemmaEmbedding2Body,
    /// The raon1 body (`crates/gpu/src/arch/raon1` `Body`): whole-card, its
    /// RDA layers over the common KDA mixer, its gated latent layers, Raon_mHC's
    /// streams and SiTU blocks, and routed experts whose gate·up the common card
    /// leg's K-quant entries read and whose down it reads in the file's blocks
    /// or, for a Q5_0 stack, in the packed layout ([`resident_down_runs`]).
    Raon1Body,
    /// The muse-spark body (`crates/gpu/src/arch/musespark` `Body`):
    /// one stage card running every layer, each the one attention-over-latent-MoE
    /// block, its gated experts' id prefix on the card by the common i-quant
    /// gate·up and down `_sel` entries ([`MUSESPARK_LEG`]) and the rest on the
    /// host tier.
    MuseSparkBody,
    /// The head128 sink/SWA body with floor-normalized top-six experts.
    FlintBody,
}

/// The program that runs `arch`'s layers; `None` when none does.
#[must_use]
pub fn program_of(arch: Arch) -> Option<Program> {
    match arch {
        Arch::Deepseek41 | Arch::Deepseek4 => Some(Program::Deepseek41Chain),
        Arch::Qwen3Moe => Some(Program::Qwen3moeBody),
        Arch::Qwen35Moe | Arch::Qwen35 | Arch::Llama | Arch::Qwen3 => Some(Program::Qwen35Body),
        Arch::Glm5Next => Some(Program::Glm5nextBody),
        Arch::Qwen4Exp => Some(Program::Qwen38Body),
        Arch::MiMo2 => Some(Program::Mimo2Body),
        Arch::MiniMaxM3 => Some(Program::MiniMaxM3Body),
        Arch::Raon1 => Some(Program::Raon1Body),
        Arch::SolarOpen2 => Some(Program::Solar2Body),
        Arch::NemotronH => Some(Program::NemotronHBody),
        Arch::NemotronHMoe => Some(Program::NemotronHMoeBody),
        Arch::GemmaEmbedding | Arch::GemmaEmbedding2 => Some(Program::GemmaEmbedding2Body),
        Arch::Flint => Some(Program::FlintBody),
        Arch::MuseSpark => Some(Program::MuseSparkBody),
        Arch::Lfm3 | Arch::MistralLarge4 => None,
    }
}

impl Program {
    /// The family's word: what a line that speaks of the program names the
    /// family by (a gate's counts and check lines). One owner, so every line
    /// that names the family names it the same way.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Deepseek41Chain => "deepseek41",
            Self::Qwen3moeBody => "qwen3moe",
            Self::Qwen35Body => "qwen35moe",
            Self::Glm5nextBody => "glm5next",
            Self::Qwen38Body => "qwen38",
            Self::Mimo2Body => "mimo2",
            Self::Solar2Body => "solar2",
            Self::MiniMaxM3Body => "minimaxm3",
            Self::NemotronHBody => "nemotron_h",
            Self::NemotronHMoeBody => "nemotron_h_moe",
            Self::GemmaEmbedding2Body => "gemma-embedding2",
            Self::Raon1Body => "raon1",
            Self::MuseSparkBody => "musespark",
            Self::FlintBody => "flint",
        }
    }
}

/// The card gate·up entry the qwen3moe family's FFN dispatch
/// (`gpu/src/arch/qwen3moe/dispatch.rs`, and the placed side's dispatch arms
/// for the types it keeps) runs a routed stack of `ty` with: the card kernel
/// table's row ([`kernels::gate_up`], the decode `_sel`) when the table names
/// the type's prompt GEMM too ([`kernels::gemm`]), among the entries that
/// dispatch launches: the K-quant family's Q4_K and Q5_K. The table's Q8_0
/// row reads a stack in the file's block stream, its i-quant rows are
/// `iq_sel.rs`'s, its MXFP4 row is `mxfp4_sel.rs`'s and its NVFP4 row
/// `mxfp4_sel.rs`'s; the dispatch launches none of them, so a whole-card load
/// refuses such a stack by name. A placed card leg (`Body`'s and `Body35`'s)
/// runs those types through the common leg ([`qwen35_leg`]).
#[must_use]
pub const fn qwen35_gate_up(ty: GgmlType) -> Option<GateUpEntry> {
    let (Some(entry), Some(_)) = (kernels::gate_up(ty), kernels::gemm(ty)) else {
        return None;
    };
    match entry {
        GateUpEntry::Q4k | GateUpEntry::Q5k => Some(entry),
        GateUpEntry::Q8_0
        | GateUpEntry::Q6k
        | GateUpEntry::Iq2xxs
        | GateUpEntry::Iq2xs
        | GateUpEntry::Iq2s
        | GateUpEntry::Iq1s
        | GateUpEntry::Iq1m
        | GateUpEntry::Iq3xxs
        | GateUpEntry::Iq3s
        | GateUpEntry::Iq4xs
        | GateUpEntry::Mxfp4
        | GateUpEntry::Nvfp4 => None,
    }
}

/// The card down entry the qwen3moe family's FFN dispatch runs a routed stack
/// of `ty` with: the table's row ([`kernels::down`]) when it names the type's
/// prompt GEMM too ([`kernels::gemm`]), among the entries the FFN dispatch
/// launches: the downs over `Q8Act`, Q4_K, Q5_K and Q6_K. The downs over
/// `Q8Blocks32` (Q5_0, Q5_1, Q8_0, IQ4_NL, NVFP4) read an activation form the
/// dispatch does not quantize, so a whole-card load refuses such a stack by
/// name; a placed card leg runs the ones the common leg does
/// ([`qwen35_leg`]).
#[must_use]
pub const fn qwen35_down(ty: GgmlType) -> Option<DownEntry> {
    let (Some((entry, _)), Some(_)) = (kernels::down(ty), kernels::gemm(ty)) else {
        return None;
    };
    match entry {
        DownEntry::Q4k | DownEntry::Q5k | DownEntry::Q6k => Some(entry),
        DownEntry::Q5_0
        | DownEntry::Q5_1
        | DownEntry::Q8_0
        | DownEntry::Iq2xxs
        | DownEntry::Iq2xs
        | DownEntry::Iq2s
        | DownEntry::Iq1s
        | DownEntry::Iq1m
        | DownEntry::Iq3xxs
        | DownEntry::Iq3s
        | DownEntry::Iq4nl
        | DownEntry::Iq4xs
        | DownEntry::Mxfp4
        | DownEntry::Nvfp4 => None,
    }
}

/// Whether the placed card leg of the qwen3moe family (`Body` and `Body35`:
/// one arm pair, [`leg_arms`]) runs a routed gate or up stack of `ty` through
/// the common leg (`crate::card_leg`'s `CardLeg`): the common leg's
/// gate·up arms ([`kernels::card_gate_up_runs`], and the Q8_0 and Q6_K arms the
/// leg launches for the stacks no other family's plan reads) of the types whose
/// prompt GEMM the table names ([`kernels::gemm`]). The wide arm reads that
/// GEMM. The i-quants (IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M, IQ3_XXS, IQ3_S,
/// IQ4_XS) and Q6_K have one and the leg runs them. MXFP4, which has none,
/// stays on the host by name; NVFP4 is not admitted (no qwen35 file or fixture
/// proves it).
#[must_use]
pub const fn prompt_leg_gate_up(ty: GgmlType) -> bool {
    (kernels::card_gate_up_runs(ty)
        || matches!(
            kernels::gate_up(ty),
            Some(GateUpEntry::Q8_0 | GateUpEntry::Q6k)
        ))
        && kernels::gemm(ty).is_some()
}

/// Whether the qwen3moe family's placed card leg runs a routed down stack of
/// `ty` through the common leg: the common down set
/// ([`kernels::card_down_runs`]: the `Q8Act` downs, Q6_K included, and the
/// 32-value downs) of the types whose prompt GEMM the table names. The
/// 32-value downs (Q5_1, Q8_0, IQ4_NL) read the `Q8Blocks32` form the leg
/// quantizes for them.
#[must_use]
pub const fn prompt_leg_down(ty: GgmlType) -> bool {
    kernels::card_down_runs(ty) && kernels::gemm(ty).is_some()
}

/// How the qwen3moe family's placed card leg runs a layer's routed stacks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegSplit {
    /// The family's own dispatch launches ([`qwen35_gate_up`], [`qwen35_down`]):
    /// a Q4_K or Q5_K gate and up with a Q4_K, Q5_K or Q6_K down, the arms
    /// whose launches and graph are the load's reference.
    Dispatch,
    /// The common leg's launches ([`prompt_leg_gate_up`],
    /// [`prompt_leg_down`]): any other layer its arms run.
    Common,
}

/// The leg that runs a layer of routed stacks of types `[gate, up, down]` on
/// the placed card of `Body` or `Body35` (one arm pair), or `None` when none
/// does (the layer is the host tier's). The one classifier the plan's rule,
/// the load's check, the residency stack table and the placed side read.
#[must_use]
pub fn qwen35_leg([gate, up, down]: [GgmlType; 3]) -> Option<LegSplit> {
    if leg_runs(Program::Qwen35Body, [gate, up, down]) != Some(true) {
        return None;
    }
    Some(
        if qwen35_gate_up(gate).is_some() && qwen35_down(down).is_some() {
            LegSplit::Dispatch
        } else {
            LegSplit::Common
        },
    )
}

/// The dense site the qwen3moe body runs a projection of `ty` with: the
/// table's row ([`kernels::dense`]) when that site reads an activation the
/// body's arena carries ([`SiteAct::Q8x128`], [`SiteAct::Q8x32`],
/// [`SiteAct::F32`] — `Form::Q8x128`, `Form::Q8x32`, `Form::F32`). A Q5_0
/// site reads [`SiteAct::Q8Blocks32`], which the arena does not carry, and
/// is `None`. An NVFP4 site reads the q8_1 rows on its gemv and the q8 blocks
/// of 32 on its wide GEMM, and the body's dense dispatch and arena carry no
/// per-arm form choice, so it is `None`. A BF16 site
/// reads the words only a load under `ModelTensors::bf16_words` keeps, which
/// the qwen3moe files do not set, and is `None`. Its legacy shared-expert
/// dispatch does not feed IQ4_XS sites either. The body's load and the
/// coverage census both read this, so a site the arena cannot feed is
/// refused at load and listed together.
#[must_use]
pub const fn qwen3moe_dense(ty: GgmlType) -> Option<SiteEntry> {
    let Some(entry) = kernels::dense(ty) else {
        return None;
    };
    if let SiteEntry::Nvfp4 | SiteEntry::Bf16 | SiteEntry::Iq4xs = entry {
        return None;
    }
    match entry.reads() {
        SiteAct::Q8x128 | SiteAct::Q8x32 | SiteAct::F32 => Some(entry),
        SiteAct::Q8Blocks32 => None,
    }
}

/// The dense sites launched by name on both the gemv and wide GEMM arms:
/// the kernel table's entries that read f32 rows or q8 blocks of 32.
#[must_use]
pub const fn dense_by_name(ty: GgmlType) -> Option<SiteEntry> {
    match kernels::dense(ty) {
        Some(entry) if matches!(entry.reads(), SiteAct::F32 | SiteAct::Q8x32) => Some(entry),
        _ => None,
    }
}

/// The dense sites launched by name over up to eight columns on every family
/// the table's dense row names but Q5_0 (`site::ByName`): the f32-row entries
/// beside those that read q8_1 blocks of 128 (the K-quants, IQ4_XS, NVFP4), whose
/// activation the unit quantizes itself. Q5_0 reads 32-value blocks no
/// by-name unit holds.
#[must_use]
pub const fn dense_projected(ty: GgmlType) -> Option<SiteEntry> {
    match kernels::dense(ty) {
        Some(SiteEntry::Q5_0) => None,
        other => other,
    }
}

/// The head a body whose load widens BF16 to f32 (`ModelTensors::bf16_words`
/// naming none, as the qwen3moe body's files) runs an `output` of `ty` with: the
/// table's [`kernels::head`] row but BF16, whose words that load does not keep.
#[must_use]
pub const fn head_of_widened(ty: GgmlType) -> Option<HeadEntry> {
    match kernels::head(ty) {
        Some(HeadEntry::Bf16) | None => None,
        Some(e) => Some(e),
    }
}

/// The embedding rows a body whose load widens BF16 to f32 reads a
/// `token_embd` of `ty` with: the table's [`kernels::embed`] row but BF16, as
/// [`head_of_widened`].
#[must_use]
pub const fn embed_of_widened(ty: GgmlType) -> Option<EmbedEntry> {
    match kernels::embed(ty) {
        Some(EmbedEntry::Bf16) | None => None,
        Some(e) => Some(e),
    }
}

/// Whether a card leg whose buffers hold the form each layer's down reads
/// runs a routed gate or up stack of `ty`: the common leg's gate·up arms in
/// the file's blocks ([`kernels::card_gate_up_runs`]) or in NVFP4's planes
/// ([`kernels::card_planes_runs`]).
#[must_use]
pub const fn planes_leg_gate_up(ty: GgmlType) -> bool {
    kernels::card_gate_up_runs(ty) || kernels::card_planes_runs(ty)
}

/// Whether a card leg whose buffers hold the form each layer's down reads
/// runs a routed down stack of `ty`: the common leg's down arms in the file's
/// blocks over either form ([`kernels::card_down_runs`]: the 128-value
/// `Q8Act` or the 32-value `Q8Blocks32`) or in NVFP4's planes
/// ([`kernels::card_planes_runs`], over `Q8Blocks32`).
#[must_use]
pub const fn planes_leg_down(ty: GgmlType) -> bool {
    kernels::card_down_runs(ty) || kernels::card_planes_runs(ty)
}

/// The card gate·up entry `Body38`'s card leg (`crates/gpu/src/arch/qwen3moe/card38.rs`)
/// runs a routed gate or up stack of `ty` with: the table's [`kernels::gate_up`]
/// row of a type [`planes_leg_gate_up`] admits whose prompt GEMM the table names
/// ([`kernels::gemm`]), which its prompt walk reads the stack through
/// (IQ2_XXS's `iq2_xxs_gate_up_sel` among them).
#[must_use]
pub const fn qwen38_gate_up(ty: GgmlType) -> Option<GateUpEntry> {
    if planes_leg_gate_up(ty) && kernels::gemm(ty).is_some() {
        kernels::gate_up(ty)
    } else {
        None
    }
}

/// The card down entry `Body38`'s card leg runs a routed down stack of `ty`
/// with: the table's [`kernels::down`] row of a type [`planes_leg_down`] admits
/// over the one form its buffers hold, the 32-value [`DownAct::Q8Blocks32`],
/// whose prompt GEMM the table names.
#[must_use]
pub const fn qwen38_down(ty: GgmlType) -> Option<DownEntry> {
    match kernels::down(ty) {
        Some((e, DownAct::Q8Blocks32)) if planes_leg_down(ty) && kernels::gemm(ty).is_some() => {
            Some(e)
        }
        _ => None,
    }
}

/// Whether a whole-file body's card leg runs a routed down of type `ty`: the
/// common leg's entries ([`kernels::card_down_runs`]) but Q6_K, whose `_sel`
/// launch reads the 128-value form the raon1 body's leg buffers do not hold,
/// and the Q5_0 one, whose stack such a body holds in the `pack_q5_0` layout
/// (`CardFormat::Q5_0`) its `q5_0_gemv_sel` reads. The one owner of the plan's
/// rule and of the load's admission (`LegStacks::of_resident`).
#[must_use]
pub const fn resident_down_runs(ty: GgmlType) -> bool {
    (kernels::card_down_runs(ty) && !matches!(ty, GgmlType::Q6_K))
        || matches!(
            kernels::down(ty),
            Some((DownEntry::Q5_0, DownAct::Q8Blocks32))
        )
}

/// The card format every card leg reads a routed stack of `ty` in: NVFP4 in
/// its planes ([`kernels::card_planes_runs`]: codes, block scales and a tensor
/// scale an expert), Q5_0 in the packed layout its `q5_0_gemv_sel` reads
/// ([`CardFormat::Q5_0`]), every other type in the file's blocks. A leg's arms
/// say which types it runs; this says how each loads, for every leg alike.
#[must_use]
pub const fn leg_card_format(ty: GgmlType) -> CardFormat {
    if kernels::card_planes_runs(ty) {
        CardFormat::Nvfp4Planes
    } else if matches!(kernels::down(ty), Some((DownEntry::Q5_0, _))) {
        CardFormat::Q5_0
    } else {
        CardFormat::KQuant
    }
}

/// The V4.1 chain's fused card layer, in stack order: the Q3_K gate and up and
/// the Q4_K down its fused expert kernels read (`gpu-deepseek41`'s step and
/// batch tile launches). A layer of other types runs the common card leg
/// ([`ds41_gate_up`], [`ds41_down`], [`ds41_pair`]).
pub const DS41_LEG: [GgmlType; 3] = [GgmlType::Q3_K, GgmlType::Q3_K, GgmlType::Q4_K];

/// Whether V4.1's chain runs a routed gate or up stack of type `ty` on the
/// card: the fused layer's Q3_K or a type the common leg launches
/// ([`kernels::card_gate_up_runs`]).
#[must_use]
pub const fn ds41_gate_up(ty: GgmlType) -> bool {
    matches!(ty, GgmlType::Q3_K) || kernels::card_gate_up_runs(ty)
}

/// Whether V4.1's chain runs a routed down stack of type `ty` on the card: a
/// type the common leg launches over the 128-value form the chain's buffers
/// hold ([`kernels::card_q8act_down_runs`], the fused layer's Q4_K among them).
#[must_use]
pub const fn ds41_down(ty: GgmlType) -> bool {
    kernels::card_q8act_down_runs(ty)
}

/// Whether V4.1's chain runs a layer of this gate·up and down type: the fused
/// layer ([`DS41_LEG`]) or, on a stage card, the common leg's, never a mix of
/// the two — the fused gate·up reads its Q4_K down by the fused launches
/// alone, and the common leg has no Q3_K gate·up. An expert tier (`tier`)
/// runs the fused layer alone: its image carries the activation's Q3_K plane
/// and scales, which the common leg's gate·up entries do not read.
#[must_use]
pub const fn ds41_pair(gate_up: GgmlType, down: GgmlType, tier: bool) -> bool {
    if matches!(gate_up, GgmlType::Q3_K) {
        matches!(down, GgmlType::Q4_K)
    } else {
        !tier
    }
}

/// Program `p`'s layer rule over the gate·up and down types, where its leg
/// has one beyond its arms ([`RoutedRule::Leg`]'s `pair`).
#[must_use]
pub fn leg_pair(p: Program) -> Option<LegPair> {
    match p {
        Program::Deepseek41Chain => Some(ds41_pair),
        _ => None,
    }
}

/// The format a whole-card load of the latent MoE body uploads a routed stack
/// of `ty` in again, for the card leg's relu² form: the leg's card format
/// ([`leg_card_format`]) where the file's own layout ([`CardFormat::of`]) is
/// another (Q8_0's planes and Q5_1's packed codes, as the file's blocks);
/// `None` where the stack already is in it (Q5_0's packed words, NVFP4's planes) or the form
/// does not run the type.
#[must_use]
pub fn relu2_relayout(ty: GgmlType) -> Option<CardFormat> {
    let read = leg_card_format(ty);
    (kernels::card_relu2_runs(ty) && CardFormat::of(ty) != Some(read)).then_some(read)
}

/// The latent MoE body's routed stacks: an up and a down (no gate) of the
/// one type its `iq2_xxs_gemv_sel` launches read, in the file's blocks.
pub const BOLT_LEG: GgmlType = GgmlType::IQ2_XXS;

/// Whether the latent MoE body runs a routed up or down stack of type `ty` on
/// the card: [`BOLT_LEG`] by its `iq2_xxs_gemv_sel` launches, or a
/// 32-value-block type or NVFP4 in its planes by the card leg's relu² form
/// ([`kernels::card_relu2_runs`]: Q5_0, Q5_1, Q8_0, IQ4_NL, NVFP4, NVFP4_ROWS). A
/// layer runs one form for both stacks, so the load refuses by name a layer
/// whose up and down take different forms.
#[must_use]
pub const fn latent_leg_runs(ty: GgmlType) -> bool {
    matches!(ty, BOLT_LEG) || kernels::card_relu2_runs(ty)
}

/// The muse-spark body's routed stacks: a gate, an up and a down of the one
/// type its `iq2_xxs_gate_up_sel` and `iq2_xxs_gemv_sel` launches read, in
/// the file's blocks.
pub const MUSESPARK_LEG: GgmlType = GgmlType::IQ2_XXS;

/// A card leg's arms: the types its gate·up launches read a gate or an up
/// stack in, and the types its down launches read a down stack in.
#[derive(Clone, Copy, Debug)]
pub struct LegArms {
    pub gate_up: fn(GgmlType) -> bool,
    pub down: fn(GgmlType) -> bool,
}

/// Program `p`'s card leg arms, as its load checks each layer it gives card
/// experts; `None` for a program that runs no routed expert.
#[must_use]
pub fn leg_arms(p: Program) -> Option<LegArms> {
    match p {
        Program::Qwen3moeBody | Program::Qwen35Body | Program::FlintBody => Some(LegArms {
            gate_up: prompt_leg_gate_up,
            down: prompt_leg_down,
        }),
        Program::Qwen38Body => Some(LegArms {
            gate_up: |ty| qwen38_gate_up(ty).is_some(),
            down: |ty| qwen38_down(ty).is_some(),
        }),
        Program::Glm5nextBody | Program::MiniMaxM3Body => Some(LegArms {
            gate_up: planes_leg_gate_up,
            down: planes_leg_down,
        }),
        Program::Mimo2Body | Program::Solar2Body => Some(LegArms {
            gate_up: kernels::card_gate_up_runs,
            down: kernels::card_down_runs,
        }),
        Program::Deepseek41Chain => Some(LegArms {
            gate_up: ds41_gate_up,
            down: ds41_down,
        }),
        Program::NemotronHMoeBody => Some(LegArms {
            gate_up: latent_leg_runs,
            down: latent_leg_runs,
        }),
        Program::Raon1Body => Some(LegArms {
            gate_up: kernels::card_gate_up_runs,
            down: resident_down_runs,
        }),
        Program::MuseSparkBody => Some(LegArms {
            gate_up: |ty| ty == MUSESPARK_LEG,
            down: |ty| ty == MUSESPARK_LEG,
        }),
        Program::NemotronHBody | Program::GemmaEmbedding2Body => None,
    }
}

/// Whether program `p`'s card leg runs a layer of routed gate, up and down
/// stacks of these types ([`leg_arms`]): a gate and an up of one type its
/// gate·up launches read, and a down its down launches read. `None` for a
/// program with no card leg.
#[must_use]
pub fn leg_runs(p: Program, [gate, up, down]: [GgmlType; 3]) -> Option<bool> {
    leg_arms(p).map(|a| {
        gate == up
            && (a.gate_up)(gate)
            && (a.down)(down)
            && leg_pair(p).is_none_or(|pair| pair(gate, down, false))
    })
}

/// The union of down activation forms of the complete routed layers that
/// `p`'s placement rule admits, resolved from the file's descriptor at load.
#[must_use]
pub fn leg_down_forms(p: Program, model: &crate::placement::ModelTensors) -> kernels::DownForms {
    use crate::placement::{ExpertPart, Role};

    let RoutedRule::Leg {
        gate_up,
        down,
        pair,
    } = routed_rule(p)
    else {
        return kernels::DownForms::NONE;
    };
    let mut forms = kernels::DownForms::NONE;
    for l in 0..model.layers {
        let mut parts = [None; 3];
        for t in model
            .tensors
            .iter()
            .filter(|t| t.role == Role::RoutedExperts && t.layer == Some(l))
        {
            let part = match ExpertPart::of_name(&t.name) {
                Some(ExpertPart::Gate) => 0,
                Some(ExpertPart::Up) => 1,
                Some(ExpertPart::Down) => 2,
                None => continue,
            };
            parts[part] = Some(t.ty);
        }
        if let [Some(g), Some(u), Some(d)] = parts
            && g == u
            && gate_up(g).is_some()
            && down(d).is_some()
            && pair.is_none_or(|pair| pair(g, d, false))
            && let Some((_, act)) = kernels::down(d)
        {
            forms = forms.with(act);
        }
    }
    forms
}

/// Program `p`'s plan rule for its routed stacks ([`RoutedRule`]): on a card
/// leg, each part by the arm its load reads ([`leg_runs`]), so a layer the
/// plan gives card experts is one the load runs. The mimo2 and solar2
/// bodies' plans, over leg buffers of the 128-value form alone, carry each
/// part by its arm: a gate and up the gate·up launches read
/// ([`kernels::card_gate_up_runs`]) and a down read over that form
/// ([`kernels::card_q8act_down_runs`]), a Q6_K down among them. The glm5next
/// and minimax-m3 bodies', whose buffers hold the form each card layer's down
/// reads, carry each part by its arm ([`planes_leg_gate_up`],
/// [`planes_leg_down`]), NVFP4 in its planes. V4.1's chain runs [`DS41_LEG`]
/// or the common leg's types ([`ds41_pair`]). A program with no routed expert puts none on a card.
#[must_use]
pub fn routed_rule(p: Program) -> RoutedRule {
    match p {
        Program::Qwen3moeBody | Program::Qwen35Body | Program::FlintBody => RoutedRule::Leg {
            gate_up: |ty| prompt_leg_gate_up(ty).then(|| leg_card_format(ty)),
            down: |ty| prompt_leg_down(ty).then(|| leg_card_format(ty)),
            pair: leg_pair(p),
        },
        Program::Qwen38Body => RoutedRule::Leg {
            gate_up: |ty| qwen38_gate_up(ty).map(|_| leg_card_format(ty)),
            down: |ty| qwen38_down(ty).map(|_| leg_card_format(ty)),
            pair: leg_pair(p),
        },
        Program::Glm5nextBody | Program::MiniMaxM3Body => RoutedRule::Leg {
            gate_up: |ty| planes_leg_gate_up(ty).then(|| leg_card_format(ty)),
            down: |ty| planes_leg_down(ty).then(|| leg_card_format(ty)),
            pair: leg_pair(p),
        },
        Program::Mimo2Body | Program::Solar2Body => RoutedRule::Leg {
            gate_up: common_leg_gate_up,
            down: common_leg_down,
            pair: leg_pair(p),
        },
        Program::Deepseek41Chain => RoutedRule::Leg {
            gate_up: |ty| ds41_gate_up(ty).then(|| leg_card_format(ty)),
            down: |ty| ds41_down(ty).then(|| leg_card_format(ty)),
            pair: leg_pair(p),
        },
        Program::NemotronHMoeBody => RoutedRule::Leg {
            gate_up: bolt_leg_card,
            down: bolt_leg_card,
            pair: leg_pair(p),
        },
        Program::Raon1Body => RoutedRule::Leg {
            gate_up: |ty| kernels::card_gate_up_runs(ty).then(|| leg_card_format(ty)),
            down: resident_down_card,
            pair: leg_pair(p),
        },
        Program::MuseSparkBody => RoutedRule::Leg {
            gate_up: musespark_leg_card,
            down: musespark_leg_card,
            pair: leg_pair(p),
        },
        // No routed stack: every one is refused by name.
        Program::NemotronHBody | Program::GemmaEmbedding2Body => RoutedRule::PerType(|_| None),
    }
}

/// The latent MoE body's plan arm: [`latent_leg_runs`], in its card format.
fn bolt_leg_card(ty: GgmlType) -> Option<CardFormat> {
    latent_leg_runs(ty).then(|| leg_card_format(ty))
}

/// The muse-spark body's plan arm: [`MUSESPARK_LEG`], in its card format.
fn musespark_leg_card(ty: GgmlType) -> Option<CardFormat> {
    (ty == MUSESPARK_LEG).then(|| leg_card_format(ty))
}

/// The raon1 body's down arm ([`resident_down_runs`]), in the leg's card format
/// ([`leg_card_format`]: a Q5_0 stack in the packed layout its `q5_0_gemv_sel`
/// reads).
fn resident_down_card(ty: GgmlType) -> Option<CardFormat> {
    resident_down_runs(ty).then(|| leg_card_format(ty))
}

/// The common leg's gate·up arm in the mimo2 and solar2 bodies' plans
/// ([`routed_rule`]): a type [`kernels::card_gate_up_runs`] admits, in its
/// card format.
fn common_leg_gate_up(ty: GgmlType) -> Option<CardFormat> {
    kernels::card_gate_up_runs(ty).then(|| leg_card_format(ty))
}

/// The common leg's down arm in those plans: a type the leg reads over the
/// 128-value form its buffers hold ([`kernels::card_q8act_down_runs`]), Q6_K
/// included, in its card format.
fn common_leg_down(ty: GgmlType) -> Option<CardFormat> {
    kernels::card_q8act_down_runs(ty).then(|| leg_card_format(ty))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::{ModelTensor, ModelTensors, Role};

    const PROGRAMS: [Program; 13] = [
        Program::Deepseek41Chain,
        Program::Qwen3moeBody,
        Program::Qwen35Body,
        Program::Glm5nextBody,
        Program::Qwen38Body,
        Program::Mimo2Body,
        Program::Solar2Body,
        Program::MiniMaxM3Body,
        Program::NemotronHBody,
        Program::NemotronHMoeBody,
        Program::Raon1Body,
        Program::MuseSparkBody,
        Program::FlintBody,
    ];

    /// Every type the file format names.
    fn types() -> Vec<GgmlType> {
        (0..64)
            .map(GgmlType::from_u32)
            .filter(|t| !matches!(t, GgmlType::Unknown(_)))
            .collect()
    }

    /// One layer for each (gate, up, down) triple of `types`, its three
    /// routed stacks named by the tail a file gives them (the part is read
    /// from that tail, the layer from `layer`).
    fn triples(types: &[GgmlType]) -> (ModelTensors, Vec<[GgmlType; 3]>) {
        let mut tensors = Vec::new();
        let mut layers = Vec::new();
        for &g in types {
            for &u in types {
                for &d in types {
                    let l = layers.len();
                    for (part, ty) in [("gate", g), ("up", u), ("down", d)] {
                        tensors.push(ModelTensor {
                            name: format!("ffn_{part}_exps.weight"),
                            shard: 0,
                            layer: Some(l),
                            role: Role::RoutedExperts,
                            ty,
                            dims: vec![256, 4, 8],
                            file_bytes: 0,
                            gathered_rows: None,
                        });
                    }
                    layers.push([g, u, d]);
                }
            }
        }
        let model = ModelTensors {
            tensors,
            layers: layers.len(),
            experts: 8,
            experts_used: 2,
            bf16_words: |_| false,
        };
        (model, layers)
    }

    /// A layer each program's plan rule gives card experts is one its card
    /// leg's load runs, for every (gate, up, down) triple of the file
    /// format's types: the plan's rule and the load's check read one table.
    /// Mutants, each a defect as it shipped: Qwen3-30B's or Qwen3.8's rule
    /// read per type over the union of its parts (a Q6_K, a Q8_0 gate), the
    /// common leg's per type (a gate and an up of two types), V4.1's read per
    /// type in its own format (a Q6_K down) — each names its program and
    /// triples. A program with no card leg (the dense nemotron_h body) cards
    /// none.
    #[test]
    fn a_layer_the_plan_cards_is_one_its_load_runs() {
        let (model, layers) = triples(&types());
        let mut broken = Vec::new();
        for p in PROGRAMS {
            let rule = routed_rule(p);
            for (l, &tys) in layers.iter().enumerate() {
                let stacks = [3 * l, 3 * l + 1, 3 * l + 2];
                let cards = rule
                    .layer_runs(&stacks, &model, false)
                    .expect("named stacks");
                // PIN(2026-10-15): the dense nemotron_h body has no card leg, so
                // the assert that every program has one is gone; a program with
                // no card leg cards no layer.
                if cards && leg_runs(p, tys) != Some(true) {
                    broken.push(format!("{p:?} {tys:?}"));
                }
            }
        }
        assert!(
            broken.is_empty(),
            "{} layers the plan cards and the load refuses, first: {:?}",
            broken.len(),
            &broken[..broken.len().min(8)]
        );
    }

    /// V4.1's card layers over every (gate, up, down) triple of the file
    /// format's types: the fused layer (a Q3_K gate and up, a Q4_K down) or a
    /// gate and up of one type the common leg launches with a down it reads
    /// over the 128-value form (IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S,
    /// IQ3_XXS, IQ4_XS, Q4_K, Q5_K, MXFP4 gate·up), never a Q3_K gate and up
    /// beside another down. The plan's rule, its host-only list and the load's
    /// check ([`leg_runs`]) read the one pair. An expert tier runs the fused
    /// layer alone. Mutant: `leg_pair` returning `None` for V4.1 — a Q3_K gate
    /// and up beside an IQ2_XXS or Q5_K down is carded and its load refuses.
    #[test]
    fn ds41_cards_the_fused_layer_or_the_common_legs_and_never_a_mix() {
        let (model, layers) = triples(&types());
        let rule = routed_rule(Program::Deepseek41Chain);
        let host_only = rule.host_only(&model).expect("named stacks");
        let mut wrong = Vec::new();
        for (l, &[gate, up, down]) in layers.iter().enumerate() {
            let fused = [gate, up, down] == DS41_LEG;
            let common = gate == up
                && kernels::card_gate_up_runs(gate)
                && kernels::card_q8act_down_runs(down);
            let stacks = [3 * l, 3 * l + 1, 3 * l + 2];
            let cards = rule
                .layer_runs(&stacks, &model, false)
                .expect("named stacks");
            let on_tier = rule
                .layer_runs(&stacks, &model, true)
                .expect("named stacks");
            let held = host_only.iter().any(|h| h.layer == l);
            let runs = leg_runs(Program::Deepseek41Chain, [gate, up, down]) == Some(true);
            if cards != (fused || common) || held == cards || runs != cards || on_tier != fused {
                wrong.push(format!(
                    "{gate:?} {up:?} {down:?}: cards {cards}, host-only {held}"
                ));
            }
        }
        assert!(
            wrong.is_empty(),
            "{} layers off the pair rule, first: {:?}",
            wrong.len(),
            &wrong[..wrong.len().min(8)]
        );
    }

    /// `Body35`'s placed card leg: a Q4_K or Q5_K gate and up with a Q4_K, Q5_K
    /// or Q6_K down are the dispatch's own launches ([`LegSplit::Dispatch`], the
    /// layers whose graph the load's reference holds); a Q4_K, Q5_K, Q6_K,
    /// IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_XS or Q8_0 gate and up of one type with a
    /// Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_XS, Q5_1, Q8_0 or IQ4_NL
    /// down (IQ2_XS, IQ2_S, IQ1_S and IQ1_M also, on either side) are the rest of the common
    /// leg's ([`LegSplit::Common`]); every other triple, a gate and up of two
    /// types among them, is the host tier's. The sets are spelled out, so a
    /// type entering or leaving an arm goes red here. Mutant: the dispatch
    /// row widened to the common leg's gate·up arms — the IQ4_XS layer reads
    /// `Dispatch`.
    #[test]
    fn qwen35_leg_is_the_dispatch_for_its_types_and_the_common_leg_for_the_rest() {
        use GgmlType::{
            IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, Q4_K, Q5_1, Q5_K,
            Q6_K, Q8_0,
        };
        // PIN(2026-10-25): IQ2_XS, IQ2_S, IQ1_S and IQ1_M joined both arms with their prompt
        // GEMMs (IQ2_XS had a gate·up and a down and no GEMM).
        let gate_up = [
            Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, Q8_0,
        ];
        let down = [
            Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, Q5_1,
            Q8_0, IQ4_NL,
        ];
        let dispatch_gate_up = [Q4_K, Q5_K];
        let dispatch_down = [Q4_K, Q5_K, Q6_K];
        for g in types() {
            assert_eq!(prompt_leg_gate_up(g), gate_up.contains(&g), "{g} gate·up");
            assert_eq!(prompt_leg_down(g), down.contains(&g), "{g} down");
            for u in types() {
                for d in types() {
                    let want = if g == u && gate_up.contains(&g) && down.contains(&d) {
                        Some(
                            if dispatch_gate_up.contains(&g) && dispatch_down.contains(&d) {
                                LegSplit::Dispatch
                            } else {
                                LegSplit::Common
                            },
                        )
                    } else {
                        None
                    };
                    assert_eq!(qwen35_leg([g, u, d]), want, "{g} {u} {d}");
                }
            }
        }
    }

    /// The qwen3moe family's placed card leg has one arm pair for both bodies
    /// (`Body` for Qwen3-30B-A3B, `Body35` for Qwen3.6): the plan's rule and
    /// the load's check read the common leg's arms ([`prompt_leg_gate_up`],
    /// [`prompt_leg_down`]), and a layer the rule cards is one [`qwen35_leg`]
    /// runs, whichever body asks. Mutant: `Program::Qwen3moeBody` back on the
    /// FFN dispatch's arms ([`qwen35_gate_up`], [`qwen35_down`]) — the IQ4_XS
    /// layer is the host's again for `Body`.
    #[test]
    fn the_qwen3moe_bodies_read_the_placed_legs_arms_and_the_classifiers() {
        for program in [Program::Qwen3moeBody, Program::Qwen35Body] {
            let arms = leg_arms(program).expect("a card leg");
            let RoutedRule::Leg { gate_up, down, .. } = routed_rule(program) else {
                panic!("a leg rule");
            };
            for ty in types() {
                assert_eq!(
                    (arms.gate_up)(ty),
                    prompt_leg_gate_up(ty),
                    "{program:?} {ty} gate·up"
                );
                assert_eq!(
                    (arms.down)(ty),
                    prompt_leg_down(ty),
                    "{program:?} {ty} down"
                );
                assert_eq!(
                    gate_up(ty).is_some(),
                    prompt_leg_gate_up(ty),
                    "{program:?} {ty} gate·up rule"
                );
                assert_eq!(
                    down(ty).is_some(),
                    prompt_leg_down(ty),
                    "{program:?} {ty} down rule"
                );
            }
            for ty in [GgmlType::IQ4_XS, GgmlType::Q6_K, GgmlType::Q8_0] {
                assert_eq!(leg_runs(program, [ty; 3]), Some(true), "{program:?} {ty}");
                assert_eq!(qwen35_leg([ty; 3]), Some(LegSplit::Common), "{ty}");
            }
        }
    }

    /// Qwen3.8's arms are the card leg's launches: a Q4_K, Q5_K, IQ2_XXS,
    /// IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_XS, IQ1_M or NVFP4 gate·up and
    /// a Q5_1, Q8_0, IQ4_NL or NVFP4 down, no other type. Mutants: the prompt
    /// GEMM left out of the gate·up (MXFP4 joins it); the down's form left out
    /// (the `Q8Act` downs join it).
    // PIN(2026-10-25): the gate·up arm gains IQ2_XS, IQ2_S, IQ1_S and IQ1_M with
    // their prompt GEMMs; the down arm is unchanged (their downs read `Q8Act`).
    #[test]
    fn qwen38_arms_are_its_legs_launches() {
        for ty in types() {
            assert_eq!(
                qwen38_gate_up(ty).is_some(),
                matches!(
                    ty,
                    GgmlType::Q4_K
                        | GgmlType::Q5_K
                        | GgmlType::IQ2_XXS
                        | GgmlType::IQ2_XS
                        | GgmlType::IQ3_XXS
                        | GgmlType::IQ1_S
                        | GgmlType::IQ3_S
                        | GgmlType::IQ2_S
                        | GgmlType::IQ4_XS
                        | GgmlType::IQ1_M
                        | GgmlType::NVFP4
                ),
                "{ty} gate·up"
            );
            assert_eq!(
                qwen38_down(ty).is_some(),
                matches!(
                    ty,
                    GgmlType::Q5_1 | GgmlType::Q8_0 | GgmlType::IQ4_NL | GgmlType::NVFP4
                ),
                "{ty} down"
            );
        }
    }

    /// The glm5next body's arms are the common leg's in the file's blocks and
    /// NVFP4's in its planes, its downs over either form, and its plan rule
    /// reads them part by part (NVFP4 in its planes, the rest in the file's
    /// blocks): a Q4_K, Q5_K, MXFP4, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S,
    /// IQ2_S, IQ4_XS, IQ1_M or NVFP4 gate·up and those or a Q5_1, Q8_0 or IQ4_NL down; no Q6_K
    /// down, which
    /// the host tier keeps. The common leg's plan, over 128-value buffers alone,
    /// keeps NVFP4 and the 32-value downs on the host. Mutants: glm5next's rule
    /// left on the common leg's (an NVFP4 layer stays on the host); NVFP4 left
    /// out of the down arm.
    #[test]
    fn glm5next_arms_are_the_leg_with_planes_over_both_forms() {
        use GgmlType::{
            IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, MXFP4, NVFP4,
            Q4_K, Q5_1, Q5_K, Q6_K, Q8_0,
        };
        // PIN(2026-10-25): IQ2_S, IQ1_S and IQ1_M joined both arms with their gate·up and down
        // entries.
        let gate_up = [
            Q4_K, Q5_K, MXFP4, NVFP4, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_XS, IQ1_M,
        ];
        // PIN(2026-10-22): Q6_K joined the down arm (`card_down_runs` runs it over the 128-value
        // form, which this body's buffers hold for the layers whose down reads it).
        // PIN(2026-10-23): IQ2_XXS joined both arms with its gate·up and down entries.
        let down = [
            Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, MXFP4, NVFP4, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL,
            IQ3_S, IQ2_S, IQ4_XS, IQ1_M,
        ];
        let arms = leg_arms(Program::Glm5nextBody).expect("a card leg");
        let RoutedRule::Leg {
            gate_up: rule_gate_up,
            down: rule_down,
            ..
        } = routed_rule(Program::Glm5nextBody)
        else {
            panic!("a leg rule");
        };
        for ty in types() {
            assert_eq!((arms.gate_up)(ty), gate_up.contains(&ty), "{ty} gate·up");
            assert_eq!((arms.down)(ty), down.contains(&ty), "{ty} down");
            let format = if ty == NVFP4 {
                CardFormat::Nvfp4Planes
            } else {
                CardFormat::KQuant
            };
            let card = |admits: bool| admits.then_some(format);
            assert_eq!(
                rule_gate_up(ty),
                card((arms.gate_up)(ty)),
                "{ty} gate·up rule"
            );
            assert_eq!(rule_down(ty), card((arms.down)(ty)), "{ty} down rule");
        }
        assert_eq!(leg_runs(Program::Glm5nextBody, [NVFP4; 3]), Some(true));
        assert_eq!(common_leg_gate_up(NVFP4), None);
        assert_eq!(common_leg_down(NVFP4), None);
        assert_eq!(common_leg_down(Q8_0), None);
    }

    /// The minimax-m3 body's arms and plan rule are the glm5next body's (its leg
    /// buffers hold the form each card layer's down reads): NVFP4 and the
    /// folder's NVFP4_ROWS stacks in their planes, every other type the leg runs
    /// in the file's blocks. Mutant: the body back on the common leg's arms — an
    /// NVFP4_ROWS layer stays on the host.
    #[test]
    fn minimaxm3_arms_are_the_planes_leg() {
        use GgmlType::{NVFP4, NVFP4_ROWS};
        let p = Program::MiniMaxM3Body;
        let arms = leg_arms(p).expect("a card leg");
        let RoutedRule::Leg { gate_up, down, .. } = routed_rule(p) else {
            panic!("a leg rule");
        };
        for ty in types().into_iter().chain([NVFP4_ROWS]) {
            assert_eq!((arms.gate_up)(ty), planes_leg_gate_up(ty), "{ty} gate·up");
            assert_eq!((arms.down)(ty), planes_leg_down(ty), "{ty} down");
            let format = if matches!(ty, NVFP4 | NVFP4_ROWS) {
                CardFormat::Nvfp4Planes
            } else {
                CardFormat::KQuant
            };
            let card = |admits: bool| admits.then_some(format);
            assert_eq!(gate_up(ty), card((arms.gate_up)(ty)), "{ty} gate·up rule");
            assert_eq!(down(ty), card((arms.down)(ty)), "{ty} down rule");
        }
        assert_eq!(leg_runs(p, [NVFP4_ROWS; 3]), Some(true));
        assert_eq!(leg_runs(p, [NVFP4; 3]), Some(true));
    }

    /// The raon1 body's arms are the common leg's launches and the Q5_0 down: a gate·up of
    /// [`kernels::card_gate_up_runs`]' types, a down of [`kernels::card_down_runs`]' types or
    /// Q5_0, the Q5_0 one in the packed layout. Mutant: the Q5_0 arm dropped from the down.
    #[test]
    fn raon1_arms_are_the_legs_launches_and_the_packed_q5_0_down() {
        let arms = leg_arms(Program::Raon1Body).expect("a card leg");
        let RoutedRule::Leg { gate_up, down, .. } = routed_rule(Program::Raon1Body) else {
            panic!("a leg rule");
        };
        for ty in types() {
            assert_eq!(
                (arms.gate_up)(ty),
                kernels::card_gate_up_runs(ty),
                "{ty} gate·up"
            );
            assert_eq!(
                gate_up(ty).is_some(),
                (arms.gate_up)(ty),
                "{ty} gate·up rule"
            );
            // PIN(2026-10-22): Q6_K stays off this body's down arm though `card_down_runs`
            // admits it: its launch reads the 128-value form these leg buffers do not hold.
            assert_eq!(
                (arms.down)(ty),
                (kernels::card_down_runs(ty) && ty != GgmlType::Q6_K) || ty == GgmlType::Q5_0,
                "{ty} down"
            );
            assert_eq!(down(ty).is_some(), (arms.down)(ty), "{ty} down rule");
        }
        assert_eq!(down(GgmlType::Q5_0), Some(CardFormat::Q5_0));
        assert_eq!(down(GgmlType::Q8_0), Some(CardFormat::KQuant));
        assert_eq!(gate_up(GgmlType::Q4_K), Some(CardFormat::KQuant));
        assert_eq!(gate_up(GgmlType::Q5_0), None);
    }

    /// The common families card a Q6_K down beside a gate and up the leg runs,
    /// and keep a Q6_K gate and up on the host; the raon1 body, whose leg
    /// buffers hold the 32-value form alone, keeps a Q6_K down on the host.
    /// Mutant: the common rule's two arms back on `card_runs` —
    /// `[Q4_K, Q4_K, Q6_K]` is host-only.
    #[test]
    fn a_q6k_down_beside_a_card_gate_up_is_on_the_card() {
        use GgmlType::{Q4_K, Q5_K, Q6_K};
        let layers = [[Q4_K, Q4_K, Q6_K], [Q5_K, Q5_K, Q6_K], [Q6_K, Q6_K, Q6_K]];
        let mut tensors = Vec::new();
        for (l, &[g, u, d]) in layers.iter().enumerate() {
            for (part, ty) in [("gate", g), ("up", u), ("down", d)] {
                tensors.push(ModelTensor {
                    name: format!("ffn_{part}_exps.weight"),
                    shard: 0,
                    layer: Some(l),
                    role: Role::RoutedExperts,
                    ty,
                    dims: vec![256, 4, 8],
                    file_bytes: 0,
                    gathered_rows: None,
                });
            }
        }
        let model = ModelTensors {
            tensors,
            layers: layers.len(),
            experts: 8,
            experts_used: 2,
            bf16_words: |_| false,
        };
        let programs = [
            Program::Glm5nextBody,
            Program::Mimo2Body,
            Program::Solar2Body,
            Program::MiniMaxM3Body,
        ];
        assert_eq!(
            leg_runs(Program::Raon1Body, layers[0]),
            Some(false),
            "raon1"
        );
        for p in programs {
            let rule = routed_rule(p);
            assert_eq!(leg_runs(p, layers[0]), Some(true), "{p:?}");
            assert_eq!(leg_runs(p, layers[1]), Some(true), "{p:?}");
            assert_eq!(leg_runs(p, layers[2]), Some(false), "{p:?}");
            for (l, tys) in layers.iter().enumerate() {
                let stacks = [3 * l, 3 * l + 1, 3 * l + 2];
                let cards = rule
                    .layer_runs(&stacks, &model, false)
                    .expect("named stacks");
                assert_eq!(cards, l < 2, "{p:?} {tys:?}");
            }
        }
    }

    /// Every program's card leg loads a stack it runs by the one format rule
    /// ([`leg_card_format`]): NVFP4 in its planes, Q5_0 in the packed layout
    /// its `q5_0_gemv_sel` reads, every other type in the file's blocks.
    /// Mutants: the defect Qwen3.8's rule had (a leg that loads an NVFP4 stack
    /// it runs as the file's blocks) and a rule without its Q5_0 arm, which
    /// the raon1 and latent MoE bodies' Q5_0 stacks read.
    #[test]
    fn every_leg_loads_a_stack_by_the_shared_format() {
        for ty in types() {
            let want = match ty {
                GgmlType::NVFP4 => CardFormat::Nvfp4Planes,
                GgmlType::Q5_0 => CardFormat::Q5_0,
                _ => CardFormat::KQuant,
            };
            assert_eq!(leg_card_format(ty), want, "{ty}");
            for p in PROGRAMS {
                let RoutedRule::Leg { gate_up, down, .. } = routed_rule(p) else {
                    continue;
                };
                for f in [gate_up(ty), down(ty)].into_iter().flatten() {
                    assert_eq!(f, want, "{p:?} {ty}");
                }
            }
        }
    }

    /// A whole-card load of the latent MoE body re-uploads a stack for the relu² form only where
    /// the file's own layout is not the one its `_sel` reads: Q5_1 and Q8_0 as the file's blocks,
    /// not Q5_0, whose packed words are the leg's format already, not NVFP4 or NVFP4_ROWS, whose
    /// planes are, and no type the form does not run. Mutant: the re-upload as the leg's format
    /// for every type the form runs.
    #[test]
    fn a_relu2_stack_is_uploaded_again_only_off_the_legs_layout() {
        assert_eq!(relu2_relayout(GgmlType::Q5_0), None);
        assert_eq!(relu2_relayout(GgmlType::Q5_1), Some(CardFormat::KQuant));
        assert_eq!(relu2_relayout(GgmlType::Q8_0), Some(CardFormat::KQuant));
        assert_eq!(relu2_relayout(GgmlType::NVFP4), None);
        assert_eq!(relu2_relayout(GgmlType::NVFP4_ROWS), None);
        for ty in types() {
            if !kernels::card_relu2_runs(ty) {
                assert_eq!(relu2_relayout(ty), None, "{ty}");
            }
        }
    }

    /// The latent MoE body's card leg runs, beside IQ2_XXS and the four 32-value-block types the
    /// relu² form ran, NVFP4 and the checkpoint folder's NVFP4_ROWS, each loaded in its planes by
    /// the program's plan rule, and no other type. Mutant: `latent_leg_runs` reading
    /// [`kernels::card_sel32_runs`], which refuses both.
    #[test]
    fn the_latent_moe_body_runs_nvfp4_planes_in_the_relu2_form() {
        for ty in types() {
            let want = matches!(
                ty,
                GgmlType::IQ2_XXS
                    | GgmlType::Q5_0
                    | GgmlType::Q5_1
                    | GgmlType::Q8_0
                    | GgmlType::IQ4_NL
                    | GgmlType::NVFP4
            );
            assert_eq!(latent_leg_runs(ty), want, "{ty}");
        }
        let RoutedRule::Leg { gate_up, down, .. } = routed_rule(Program::NemotronHMoeBody) else {
            panic!("the latent MoE body's routed experts run on a card leg");
        };
        for ty in [GgmlType::NVFP4, GgmlType::NVFP4_ROWS] {
            assert!(latent_leg_runs(ty), "{ty}");
            assert_eq!(
                leg_runs(Program::NemotronHMoeBody, [ty; 3]),
                Some(true),
                "{ty}"
            );
            assert_eq!(gate_up(ty), Some(CardFormat::Nvfp4Planes), "{ty}");
            assert_eq!(down(ty), Some(CardFormat::Nvfp4Planes), "{ty}");
        }
    }

    /// The qwen3moe body's dense owner is the table row kept to the forms
    /// its arena carries: `None` for Q5_0 (it reads `Q8Blocks32`) and NVFP4
    /// (its gemv reads the q8_1 rows and its wide GEMM the q8 blocks of 32,
    /// two forms where a row of this owner holds one), `Some` for Q3_K,
    /// Q4_K, Q5_K, Q6_K, Q8_0 and F32. Nor its head and embedding rows BF16:
    /// its load keeps no BF16 words.
    #[test]
    fn the_qwen3moe_body_dense_owner_refuses_q5_0() {
        use crate::kernels::SiteEntry;
        assert_eq!(qwen3moe_dense(GgmlType::Q5_0), None);
        assert_eq!(qwen3moe_dense(GgmlType::NVFP4), None);
        assert_eq!(qwen3moe_dense(GgmlType::BF16), None);
        assert_eq!(qwen3moe_dense(GgmlType::IQ4_XS), None);
        assert_eq!(head_of_widened(GgmlType::BF16), None);
        assert_eq!(embed_of_widened(GgmlType::BF16), None);
        assert!(head_of_widened(GgmlType::Q8_0).is_some());
        assert!(embed_of_widened(GgmlType::Q8_0).is_some());
        assert_eq!(qwen3moe_dense(GgmlType::Q3_K), Some(SiteEntry::Q3k));
        assert_eq!(qwen3moe_dense(GgmlType::Q4_K), Some(SiteEntry::Q4k));
        assert_eq!(qwen3moe_dense(GgmlType::Q5_K), Some(SiteEntry::Q5k));
        assert_eq!(qwen3moe_dense(GgmlType::Q6_K), Some(SiteEntry::Q6k));
        assert_eq!(qwen3moe_dense(GgmlType::Q8_0), Some(SiteEntry::Q8_0));
        assert_eq!(qwen3moe_dense(GgmlType::F32), Some(SiteEntry::F32));
    }
}
