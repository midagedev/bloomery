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
use crate::placement::{CardFormat, RoutedRule};

/// A layer program: the chain that runs a family's layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Program {
    /// The V4.1 chain (`crates/gpu-deepseek41`: `chain/`, `body.rs`).
    Deepseek41Chain,
    /// The whole-card qwen3moe body (`crates/gpu/src/arch/qwen3moe` `Body`).
    Qwen3moeBody,
    /// The whole-card qwen35moe and qwen35 body (`crates/gpu/src/arch/qwen3moe`
    /// `Body35`): gated-delta-rule and gated GQA layers, its sites' types by
    /// `crate::site`.
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
    /// puts on a card (the common pair) and the rest on the host tier, beside a
    /// shared expert; its GQA layers select blocks by an index key
    /// (`attention.indexer.*`) and its experts run `swigluoai`.
    MiniMaxM3Body,
    /// The nemotron_h body (`crates/gpu/src/arch/nemotron_h`): whole-card,
    /// dense, its Mamba-2, NoPE GQA and squared-ReLU layers over the common
    /// kernels; no routed experts.
    NemotronHBody,
    /// The nemotron_h_moe body (`crates/gpu/src/arch/nemotron_h_moe`):
    /// whole-card, the Nano's Mamba-2 and NoPE GQA layers over the shared
    /// hybrid owner (`crates/gpu/src/arch/hybrid.rs`) and the latent MoE layer,
    /// whose routed up and down stacks and shared expert are IQ2_XXS read by
    /// `iq2_xxs_gemv_sel` ([`BOLT_LEG`]).
    NemotronHMoeBody,
    /// The EmbeddingGemma encoders (`crates/gpu-gemma-embedding2`): one
    /// pass of a dense trunk on a card, no routed experts.
    GemmaEmbedding2Body,
}

/// The program that runs `arch`'s layers; `None` when none does.
#[must_use]
pub fn program_of(arch: Arch) -> Option<Program> {
    match arch {
        Arch::Deepseek41 | Arch::Deepseek4 => Some(Program::Deepseek41Chain),
        Arch::Qwen3Moe => Some(Program::Qwen3moeBody),
        Arch::Qwen35Moe | Arch::Qwen35 => Some(Program::Qwen35Body),
        Arch::Glm5Next => Some(Program::Glm5nextBody),
        Arch::Qwen4Exp => Some(Program::Qwen38Body),
        Arch::MiMo2 => Some(Program::Mimo2Body),
        Arch::MiniMaxM3 => Some(Program::MiniMaxM3Body),
        Arch::SolarOpen2 => Some(Program::Solar2Body),
        Arch::NemotronH => Some(Program::NemotronHBody),
        Arch::NemotronHMoe => Some(Program::NemotronHMoeBody),
        Arch::GemmaEmbedding | Arch::GemmaEmbedding2 => Some(Program::GemmaEmbedding2Body),
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
        }
    }
}

/// The card gate·up entry `Body35` runs a routed stack of `ty` with: the
/// card kernel table's row ([`kernels::gate_up`], the decode `_sel`) when
/// the table names the type's prompt GEMM too ([`kernels::gemm`]), among
/// the entries the family's FFN dispatch launches
/// (`gpu/src/arch/qwen3moe/dispatch.rs`): the K-quant family's Q4_K and
/// Q5_K. The table's Q8_0 row reads a stack in the file's block stream, its
/// i-quant rows are `iq_sel.rs`'s, its MXFP4 row is `mxfp4_sel.rs`'s and its
/// NVFP4 row `mxfp4_sel.rs`'s; the dispatch launches none of them, so
/// such a stack is refused by name.
#[must_use]
pub const fn qwen35_gate_up(ty: GgmlType) -> Option<GateUpEntry> {
    let (Some(entry), Some(_)) = (kernels::gate_up(ty), kernels::gemm(ty)) else {
        return None;
    };
    match entry {
        GateUpEntry::Q4k | GateUpEntry::Q5k => Some(entry),
        GateUpEntry::Q8_0
        | GateUpEntry::Iq2xs
        | GateUpEntry::Iq3xxs
        | GateUpEntry::Iq3s
        | GateUpEntry::Iq4xs
        | GateUpEntry::Mxfp4
        | GateUpEntry::Nvfp4 => None,
    }
}

/// The card down entry `Body35` runs a routed stack of `ty` with: the
/// table's row ([`kernels::down`]) when it names the type's prompt GEMM too
/// ([`kernels::gemm`]), among the entries the FFN dispatch launches: the
/// downs over `Q8Act`, Q4_K, Q5_K and Q6_K. The downs over `Q8Blocks32`
/// (Q5_0, Q5_1, Q8_0, IQ4_NL, NVFP4) read an activation form the dispatch does not
/// quantize, so such a stack is refused by name.
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
        | DownEntry::Iq2xs
        | DownEntry::Iq3xxs
        | DownEntry::Iq3s
        | DownEntry::Iq4nl
        | DownEntry::Iq4xs
        | DownEntry::Mxfp4
        | DownEntry::Nvfp4 => None,
    }
}

/// The dense site the qwen3moe body runs a projection of `ty` with: the
/// table's row ([`kernels::dense`]) when that site reads an activation the
/// body's arena carries ([`SiteAct::Q8x128`], [`SiteAct::Q8x32`],
/// [`SiteAct::F32`] — `Form::Q8x128`, `Form::Q8x32`, `Form::F32`). A Q5_0
/// site reads [`SiteAct::Q8Blocks32`], which the arena does not carry, and
/// is `None`. An NVFP4 site has a gemv and no wide GEMM, which the body's
/// prompt units run every dense site through, and is `None`. A BF16 site
/// reads the words only a load under `ModelTensors::bf16_words` keeps, which
/// the qwen3moe files do not set, and is `None`. The body's load and the
/// coverage census both read this, so a site the arena cannot feed is
/// refused at load and listed together.
#[must_use]
pub const fn qwen3moe_dense(ty: GgmlType) -> Option<SiteEntry> {
    let Some(entry) = kernels::dense(ty) else {
        return None;
    };
    if let SiteEntry::Nvfp4 | SiteEntry::Bf16 = entry {
        return None;
    }
    match entry.reads() {
        SiteAct::Q8x128 | SiteAct::Q8x32 | SiteAct::F32 => Some(entry),
        SiteAct::Q8Blocks32 => None,
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

/// The card gate·up entry `Body38`'s card leg (`crates/gpu/src/arch/qwen3moe/card38.rs`)
/// runs a routed gate or up stack of `ty` with: the table's [`kernels::gate_up`]
/// row of the K-quants, i-quants and NVFP4 the leg launches (`kq_gate_up_act_q4k`,
/// `kq_gate_up_act_q5k`, `iq3_xxs_gate_up_sel`, `iq4_xs_gate_up_sel`,
/// `nvfp4_gate_up_sel`): a Q4_K, Q5_K, IQ3_XXS, IQ4_XS or NVFP4.
#[must_use]
pub const fn qwen38_gate_up(ty: GgmlType) -> Option<GateUpEntry> {
    match kernels::gate_up(ty) {
        Some(
            e @ (GateUpEntry::Q4k
            | GateUpEntry::Q5k
            | GateUpEntry::Iq3xxs
            | GateUpEntry::Iq4xs
            | GateUpEntry::Nvfp4),
        ) => Some(e),
        _ => None,
    }
}

/// The card down entry `Body38`'s card leg runs a routed down stack of `ty`
/// with: the table's [`kernels::down`] row over the 32-value activation the leg
/// quantizes ([`DownAct::Q8Blocks32`]: `q5_1_gemv_sel`, `q8_0_gemv_sel32`,
/// `iq4_nl_gemv_sel32`, `nvfp4_gemv_sel32`): a Q5_1, Q8_0, IQ4_NL or NVFP4.
#[must_use]
pub const fn qwen38_down(ty: GgmlType) -> Option<DownEntry> {
    match kernels::down(ty) {
        Some((
            e @ (DownEntry::Q5_1 | DownEntry::Q8_0 | DownEntry::Iq4nl | DownEntry::Nvfp4),
            DownAct::Q8Blocks32,
        )) => Some(e),
        _ => None,
    }
}

/// The card format every card leg reads a routed stack of `ty` in: NVFP4 in
/// its planes ([`kernels::card_planes_runs`]: codes, block scales and a tensor
/// scale an expert), every other type in the file's blocks. A leg's arms say
/// which types it runs; this says how each loads, for every leg alike.
#[must_use]
pub const fn leg_card_format(ty: GgmlType) -> CardFormat {
    if kernels::card_planes_runs(ty) {
        CardFormat::Nvfp4Planes
    } else {
        CardFormat::KQuant
    }
}

/// The V4.1 chain's card leg, in stack order: the Q3_K gate and up and the
/// Q4_K down its expert kernels read, the only types its card stacks load in
/// (`gpu-deepseek41`'s `CardStacks::of` and the swap's file stacks read this).
pub const DS41_LEG: [GgmlType; 3] = [GgmlType::Q3_K, GgmlType::Q3_K, GgmlType::Q4_K];

/// The latent MoE body's routed stacks: an up and a down (no gate) of the
/// one type its `iq2_xxs_gemv_sel` launches read, in the file's blocks.
pub const BOLT_LEG: GgmlType = GgmlType::IQ2_XXS;

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
        Program::Qwen3moeBody | Program::Qwen35Body => Some(LegArms {
            gate_up: |ty| qwen35_gate_up(ty).is_some(),
            down: |ty| qwen35_down(ty).is_some(),
        }),
        Program::Qwen38Body => Some(LegArms {
            gate_up: |ty| qwen38_gate_up(ty).is_some(),
            down: |ty| qwen38_down(ty).is_some(),
        }),
        Program::Glm5nextBody
        | Program::Mimo2Body
        | Program::Solar2Body
        | Program::MiniMaxM3Body => Some(LegArms {
            gate_up: kernels::card_gate_up_runs,
            down: kernels::card_down_runs,
        }),
        Program::Deepseek41Chain => Some(LegArms {
            gate_up: |ty| ty == DS41_LEG[0],
            down: |ty| ty == DS41_LEG[2],
        }),
        Program::NemotronHMoeBody => Some(LegArms {
            gate_up: |ty| ty == BOLT_LEG,
            down: |ty| ty == BOLT_LEG,
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
    leg_arms(p).map(|a| gate == up && (a.gate_up)(gate) && (a.down)(down))
}

/// Program `p`'s plan rule for its routed stacks ([`RoutedRule`]): on a card
/// leg, each part by the arm its load reads ([`leg_runs`]), so a layer the
/// plan gives card experts is one the load runs. The common leg's plan
/// carries a type on the card only where both its gate·up and its down
/// launches read it ([`kernels::card_runs`]), in every part. V4.1's chain
/// runs [`DS41_LEG`] alone. A program with no routed expert puts none on a
/// card.
#[must_use]
pub fn routed_rule(p: Program) -> RoutedRule {
    match p {
        Program::Qwen3moeBody | Program::Qwen35Body => RoutedRule::Leg {
            gate_up: |ty| qwen35_gate_up(ty).map(|_| leg_card_format(ty)),
            down: |ty| qwen35_down(ty).map(|_| leg_card_format(ty)),
        },
        Program::Qwen38Body => RoutedRule::Leg {
            gate_up: |ty| qwen38_gate_up(ty).map(|_| leg_card_format(ty)),
            down: |ty| qwen38_down(ty).map(|_| leg_card_format(ty)),
        },
        Program::Glm5nextBody
        | Program::Mimo2Body
        | Program::Solar2Body
        | Program::MiniMaxM3Body => RoutedRule::Leg {
            gate_up: common_leg_card,
            down: common_leg_card,
        },
        Program::Deepseek41Chain => RoutedRule::Leg {
            gate_up: |ty| (ty == DS41_LEG[0]).then(|| leg_card_format(ty)),
            down: |ty| (ty == DS41_LEG[2]).then(|| leg_card_format(ty)),
        },
        Program::NemotronHMoeBody => RoutedRule::Leg {
            gate_up: bolt_leg_card,
            down: bolt_leg_card,
        },
        // No routed stack: every one is refused by name.
        Program::NemotronHBody | Program::GemmaEmbedding2Body => RoutedRule::PerType(|_| None),
    }
}

/// The latent MoE body's plan arm: [`BOLT_LEG`], in its card format.
fn bolt_leg_card(ty: GgmlType) -> Option<CardFormat> {
    (ty == BOLT_LEG).then(|| leg_card_format(ty))
}

/// The common leg's plan arm: a type both its gate·up and down launches read.
fn common_leg_card(ty: GgmlType) -> Option<CardFormat> {
    kernels::card_runs(ty).then(|| leg_card_format(ty))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::{ModelTensor, ModelTensors, Role};

    const PROGRAMS: [Program; 10] = [
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
                let cards = rule.layer_runs(&stacks, &model).expect("named stacks");
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

    /// Qwen3.8's arms are the card leg's launches: a Q4_K, Q5_K, IQ3_XXS,
    /// IQ4_XS or NVFP4 gate·up and a Q5_1, Q8_0, IQ4_NL or NVFP4 down, no
    /// other type.
    #[test]
    fn qwen38_arms_are_its_legs_launches() {
        for ty in types() {
            assert_eq!(
                qwen38_gate_up(ty).is_some(),
                matches!(
                    ty,
                    GgmlType::Q4_K
                        | GgmlType::Q5_K
                        | GgmlType::IQ3_XXS
                        | GgmlType::IQ4_XS
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

    /// Every program's card leg loads a stack it runs by the one format rule
    /// ([`leg_card_format`]): NVFP4 in its planes, every other type in the
    /// file's blocks. Mutant, the defect Qwen3.8's rule had: a leg that loads
    /// an NVFP4 stack it runs as the file's blocks.
    #[test]
    fn every_leg_loads_a_stack_by_the_shared_format() {
        for ty in types() {
            let want = if ty == GgmlType::NVFP4 {
                CardFormat::Nvfp4Planes
            } else {
                CardFormat::KQuant
            };
            assert_eq!(leg_card_format(ty), want, "{ty}");
            for p in PROGRAMS {
                let RoutedRule::Leg { gate_up, down } = routed_rule(p) else {
                    continue;
                };
                for f in [gate_up(ty), down(ty)].into_iter().flatten() {
                    assert_eq!(f, want, "{p:?} {ty}");
                }
            }
        }
    }

    /// The qwen3moe body's dense owner is the table row kept to the forms
    /// its arena carries and to a wide GEMM: `None` for Q5_0 (it reads
    /// `Q8Blocks32`) and NVFP4 (no wide GEMM), `Some` for Q3_K, Q4_K, Q5_K,
    /// Q6_K, Q8_0 and F32. Nor its head and embedding rows BF16: its load
    /// keeps no BF16 words.
    #[test]
    fn the_qwen3moe_body_dense_owner_refuses_q5_0() {
        use crate::kernels::SiteEntry;
        assert_eq!(qwen3moe_dense(GgmlType::Q5_0), None);
        assert_eq!(qwen3moe_dense(GgmlType::NVFP4), None);
        assert_eq!(qwen3moe_dense(GgmlType::BF16), None);
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
