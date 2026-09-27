//! The coverage check: what of a file no program in this tree runs, every
//! item listed at once ([`check`]). It joins the model's [`needs`] with a
//! table of what exists ([`AVAILABLE`], one row per kernel instance or
//! program step, each citing the constant or the function that owns it; the
//! routers' and the GQA flashes' rows ask the shape table `models::shape`,
//! which the router crates are held to), the card formats of the tensors'
//! types ([`CardFormat`], the format owner), the program's per-tensor type
//! pins, the pre-tokenizers the tokenizer runs
//! (`tokenizer::runs_pre_tokenizer`, the tokenizer's own table) and the chat
//! parsers.
//!
//! The kernels' own file-against-constant checks stay behind it as a second
//! line: a file this check passes cannot trip them.
//!
//! An architecture no program runs yet ([`program_of`] is `None`) is listed
//! against the whole tree: a need any program's row covers, and a tensor type
//! any program's pin reads, is not an item; the program itself is one.

use gguf::GgmlType;
use models::shape::{RouterBody, gqa_row, select_router};
use models::{
    Arch, Collapse, DeltaKind, Extra, GdnGate, HcMix, KHeadMap, LatentUp, ModelSpec, Need,
    RopeMode, needs,
};

use crate::placement::{CardFormat, ModelTensors, Role, Unimplemented};

/// A layer program: the chain that runs a family's layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Program {
    /// The V4.1 chain (`crates/gpu-deepseek41`: `chain/`, `body.rs`).
    Deepseek41Chain,
    /// The whole-card qwen3moe body (`crates/gpu/src/arch/qwen3moe`).
    Qwen3moeBody,
    /// The glm5next body (`crates/gpu-glm5next`): every routed expert on the
    /// host tier.
    Glm5nextBody,
    /// The qwen4exp body (`crates/gpu/src/arch/qwen3moe` `Body38`): every
    /// routed expert on the host tier.
    Qwen38Body,
}

/// The program that runs `arch`'s layers; `None` when none does.
#[must_use]
pub fn program_of(arch: Arch) -> Option<Program> {
    match arch {
        Arch::Deepseek41 | Arch::Deepseek4 => Some(Program::Deepseek41Chain),
        Arch::Qwen3Moe | Arch::Qwen35Moe => Some(Program::Qwen3moeBody),
        Arch::Glm5Next => Some(Program::Glm5nextBody),
        Arch::Qwen4Exp => Some(Program::Qwen38Body),
    }
}

/// One thing a program runs: the need it covers, and where it lives.
#[derive(Clone, Copy)]
pub struct Available {
    pub program: Program,
    /// The constant or function that owns the instance.
    pub at: &'static str,
    /// Whether this row covers `need`.
    pub runs: fn(&Need) -> bool,
}

/// Everything the programs run that a [`Need`] names. A need no row covers
/// is an item of [`check`]; a row removed makes its needs items.
pub const AVAILABLE: &[Available] = &[
    Available {
        program: Program::Deepseek41Chain,
        at: "gpu-deepseek41/src/attn.rs LATENT; chain/attn.rs rope dims",
        runs: |n| {
            matches!(
                n,
                Need::Latent {
                    latent: 512,
                    rope: 64,
                    up: LatentUp::KeqV
                }
            )
        },
    },
    Available {
        program: Program::Deepseek41Chain,
        at: "gpu-deepseek41/src/compress.rs WIDTH",
        runs: |n| matches!(n, Need::Compress { width: 512 }),
    },
    Available {
        program: Program::Deepseek41Chain,
        at: "gpu-deepseek41/src/indexer.rs HEADS, HEAD_DIM; index_key.rs WIDTH",
        runs: |n| matches!(n, Need::StreamIndex { heads: 32, d: 128 }),
    },
    Available {
        program: Program::Deepseek41Chain,
        at: "models/src/shape.rs ROUTERS, the Ds41 body",
        runs: |n| routes(n, &[RouterBody::Ds41]),
    },
    Available {
        program: Program::Deepseek41Chain,
        at: "gpu-deepseek41/src/chain/ffn.rs (the shared expert)",
        runs: |n| {
            matches!(
                n,
                Need::Shared {
                    sigmoid_gate: false,
                    ..
                }
            )
        },
    },
    Available {
        program: Program::Deepseek41Chain,
        at: "gpu-deepseek41/src/hc.rs HC_STREAMS",
        runs: |n| matches!(n, Need::Hc { streams: 4 }),
    },
    Available {
        program: Program::Deepseek41Chain,
        at: "gpu-deepseek41/src/engram_gate.rs ROW",
        runs: |n| matches!(n, Need::Engram { row: 5120 }),
    },
    Available {
        program: Program::Qwen3moeBody,
        at: "models/src/shape.rs GQA",
        runs: |n| matches!(n, Need::Gqa { head, group } if gqa_row(*head, *group).is_some()),
    },
    Available {
        program: Program::Qwen3moeBody,
        at: "gpu/src/rope_neox.rs HEAD",
        runs: |n| {
            matches!(
                n,
                Need::QkRope {
                    qk_norm: true,
                    head: 128,
                    mode: RopeMode::Neox,
                    dims: 128
                }
            )
        },
    },
    Available {
        program: Program::Qwen3moeBody,
        at: "models/src/shape.rs ROUTERS, the Qwen3moe and Qwen35moe bodies",
        runs: |n| routes(n, &[RouterBody::Qwen3moe, RouterBody::Qwen35moe]),
    },
    Available {
        program: Program::Glm5nextBody,
        at: "gpu-glm5next/src/mla.rs (the absorbed heads over gpu-deepseek41 attn.rs LATENT)",
        runs: |n| {
            matches!(
                n,
                Need::Latent {
                    latent: 512,
                    rope: 0,
                    up: LatentUp::Absorbed { qk: 256, v: 256 }
                }
            )
        },
    },
    Available {
        program: Program::Glm5nextBody,
        at: "gpu-glm5next/src/mla.rs (the keys appended; every position attended up to \
             place::dense_positions, a later one refused)",
        runs: |n| {
            matches!(
                n,
                Need::TokenPool {
                    heads: 32,
                    d: 128,
                    pool: 4
                }
            )
        },
    },
    Available {
        program: Program::Glm5nextBody,
        at: "gpu/src/latent.rs INDEX_HEAD (index_key_ln_append)",
        runs: |n| matches!(n, Need::KeyLayerNorm),
    },
    Available {
        program: Program::Glm5nextBody,
        at: "gpu/src/linear/mod.rs HEAD, CONV_TAPS (kda_conv_prep, kda_delta)",
        runs: |n| {
            matches!(
                n,
                Need::DeltaRule {
                    kind: DeltaKind::Kda { .. },
                    d: 128,
                    conv: 4,
                    ..
                }
            )
        },
    },
    Available {
        program: Program::Glm5nextBody,
        at: "models/src/shape.rs ROUTERS, the Glm5next body",
        runs: |n| routes(n, &[RouterBody::Glm5next]),
    },
    Available {
        program: Program::Glm5nextBody,
        at: "gpu-glm5next/src/ffn.rs (the shared expert)",
        runs: |n| {
            matches!(
                n,
                Need::Shared {
                    sigmoid_gate: false,
                    ..
                }
            )
        },
    },
    Available {
        program: Program::Glm5nextBody,
        at: "gpu-glm5next/src/ffn.rs (the dense block)",
        runs: |n| matches!(n, Need::DenseFfn { .. }),
    },
    Available {
        program: Program::Glm5nextBody,
        at: "gpu-deepseek41/src/hc.rs HC_STREAMS (hc_pre_q8_0)",
        runs: |n| matches!(n, Need::Hc { streams: 4 }),
    },
    Available {
        program: Program::Glm5nextBody,
        at: "gpu-glm5next/src/body.rs (each sub-layer's own mix, the streams' mean)",
        runs: |n| {
            matches!(
                n,
                Need::HcWiring {
                    mix: HcMix::Own,
                    collapse: Collapse::Mean
                }
            )
        },
    },
    Available {
        program: Program::Glm5nextBody,
        at: "gpu-glm5next/src/kda.rs (the state and the conv ring, one lane)",
        runs: |n| matches!(n, Need::RecurrentState),
    },
    Available {
        program: Program::Glm5nextBody,
        at: "gpu-glm5next/src/program.rs (a mixer program per layer)",
        runs: |n| matches!(n, Need::MixedTrunk),
    },
    Available {
        program: Program::Qwen38Body,
        at: "gpu/src/arch/qwen3moe/plan38.rs GDN (norm_gate GATE_SIGMOID)",
        runs: |n| {
            matches!(
                n,
                Need::DeltaRule {
                    kind: DeltaKind::Gdn {
                        khead_map: KHeadMap::Tiled,
                        gate: GdnGate::Sigmoid
                    },
                    k_heads: 16,
                    v_heads: 48,
                    d: 128,
                    conv: 4,
                }
            )
        },
    },
    Available {
        program: Program::Qwen38Body,
        at: "models/src/shape.rs GQA, the `_256_p4` flash (Body38)",
        runs: |n| matches!(n, Need::Gqa { head, group } if gqa_row(*head, *group).is_some()),
    },
    Available {
        program: Program::Qwen38Body,
        at: "gpu/src/rope_neox.rs head_norm_neox_append_256 (Body38, 64 of 256 dims)",
        runs: |n| {
            matches!(
                n,
                Need::QkRope {
                    qk_norm: true,
                    head: 256,
                    mode: RopeMode::Imrope { .. },
                    dims: 64
                }
            )
        },
    },
    Available {
        program: Program::Qwen38Body,
        at: "gpu/src/q38.rs gqa_out_gate_f32",
        runs: |n| matches!(n, Need::OutGate),
    },
    Available {
        program: Program::Qwen38Body,
        at: "gpu/src/qsa.rs (pool, score, top-k; q38.rs qsa_key_append)",
        runs: |n| {
            matches!(
                n,
                Need::MeanPool {
                    heads: 4,
                    d: 128,
                    pool: 4,
                    rope: 64
                }
            )
        },
    },
    Available {
        program: Program::Qwen38Body,
        at: "models/src/shape.rs ROUTERS, the Qwen35moe body's `_512` instance (Body38)",
        runs: |n| routes(n, &[RouterBody::Qwen35moe]),
    },
    Available {
        program: Program::Qwen38Body,
        at: "gpu/src/arch/qwen3moe/program38.rs shared (q38.rs q38_shared_add)",
        runs: |n| {
            matches!(
                n,
                Need::Shared {
                    ff: 640,
                    sigmoid_gate: true
                }
            )
        },
    },
    Available {
        program: Program::Qwen38Body,
        at: "gpu/src/hc_gated.rs (mix and combine; Body38)",
        runs: |n| {
            matches!(
                n,
                Need::HcGated {
                    streams: 4,
                    rank: 320
                }
            )
        },
    },
    Available {
        program: Program::Qwen38Body,
        at: "gpu/src/arch/qwen3moe/program38.rs head_mix (the head's mix, no inject)",
        runs: |n| matches!(n, Need::HcGatedHead { rank: 320 }),
    },
    Available {
        program: Program::Qwen38Body,
        at: "gpu/src/ple.rs (gate, conv; Body38's host rows)",
        runs: |n| {
            matches!(
                n,
                Need::Ple {
                    row: 2560,
                    conv: 4,
                    dilation: 3
                }
            )
        },
    },
    Available {
        program: Program::Qwen38Body,
        at: "engram/src/hash.rs ple_rows_into (the image placeholder refused by name)",
        runs: |n| matches!(n, Need::ImageToken(_)),
    },
    Available {
        program: Program::Qwen38Body,
        at: "gpu/src/arch/qwen3moe/scratch38.rs Store38 (the state and the conv ring, one lane)",
        runs: |n| matches!(n, Need::RecurrentState),
    },
    Available {
        program: Program::Qwen38Body,
        at: "gpu/src/arch/qwen3moe/program38.rs (a mixer per layer's plan)",
        runs: |n| matches!(n, Need::MixedTrunk),
    },
];

/// Whether the router `n` names has an instance among `bodies`' rows: the
/// row [`select_router`] picks, so a count or a pick the launchers refuse
/// is an item here too.
fn routes(n: &Need, bodies: &[RouterBody]) -> bool {
    matches!(n, Need::Router(s) if select_router(*s).is_ok_and(|r| bodies.contains(&r.body)))
}

/// A tensor family whose type a program pins, and the types it reads.
struct TypePin {
    program: Program,
    role: Role,
    /// Only the quantized tensors of two or more dimensions (the matrices a
    /// gemv reads; an F32 parameter table is its layer's mixer's).
    matrices: bool,
    what: &'static str,
    reads: &'static [GgmlType],
}

/// The per-tensor type pins of the programs (the qwen3moe body's `kq_site`
/// calls in `Body::load`, the head's Q6_K gemv in `gpu/src/head.rs`, the
/// card embedding's `embed_rows_q4k`; the V4.1 chain's attention gemvs, whose
/// head-split sites in `chain/attn.rs` take q3_K or q8_0, the same head, and
/// the hyper-connection fn `hc.rs` reads as q3_K words, `hc_f32.rs` as f32;
/// the glm5next body's sites, each a q8_0 gemv or `hc_pre_q8_0`, the head's
/// q8_0 arm, and the embedding row the host dequantizes).
const TYPE_PINS: &[TypePin] = &[
    TypePin {
        program: Program::Qwen3moeBody,
        role: Role::Attention,
        matrices: true,
        what: "attention matrices (the body reads q4_K and q6_K)",
        reads: &[GgmlType::Q4_K, GgmlType::Q6_K],
    },
    TypePin {
        program: Program::Qwen3moeBody,
        role: Role::Head,
        matrices: true,
        what: "output head (the head reads q6_K)",
        reads: &[GgmlType::Q6_K],
    },
    TypePin {
        program: Program::Qwen3moeBody,
        role: Role::TokenEmbedding,
        matrices: true,
        what: "token embedding (the card reads q4_K rows)",
        reads: &[GgmlType::Q4_K],
    },
    TypePin {
        program: Program::Deepseek41Chain,
        role: Role::Attention,
        matrices: true,
        what: "attention matrices (the chain reads q3_K and q8_0)",
        reads: &[GgmlType::Q3_K, GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Deepseek41Chain,
        role: Role::Head,
        matrices: true,
        what: "output head (the head reads q6_K)",
        reads: &[GgmlType::Q6_K],
    },
    TypePin {
        program: Program::Deepseek41Chain,
        role: Role::HyperConnection,
        matrices: true,
        what: "hyper-connection fn (the chain reads q3_K and f32)",
        reads: &[GgmlType::Q3_K],
    },
    TypePin {
        program: Program::Glm5nextBody,
        role: Role::Attention,
        matrices: true,
        what: "attention matrices (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Glm5nextBody,
        role: Role::HyperConnection,
        matrices: true,
        what: "hyper-connection fn (hc_pre_q8_0 reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Glm5nextBody,
        role: Role::DenseFfn,
        matrices: true,
        what: "dense block (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Glm5nextBody,
        role: Role::SharedExpert,
        matrices: true,
        what: "shared expert (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Glm5nextBody,
        role: Role::Head,
        matrices: true,
        what: "output head (the head reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Glm5nextBody,
        role: Role::TokenEmbedding,
        matrices: true,
        what: "token embedding (the host reads q8_0 rows)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Qwen38Body,
        role: Role::Attention,
        matrices: true,
        what: "attention matrices (the body reads q8_0, and bf16 widened to f32)",
        reads: &[GgmlType::Q8_0, GgmlType::BF16],
    },
    TypePin {
        program: Program::Qwen38Body,
        role: Role::HyperConnection,
        matrices: true,
        what: "hyper-connection down and up (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Qwen38Body,
        role: Role::SharedExpert,
        matrices: true,
        what: "shared expert (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Qwen38Body,
        role: Role::Head,
        matrices: true,
        what: "output head (the head reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Qwen38Body,
        role: Role::TokenEmbedding,
        matrices: true,
        what: "token embedding (the card reads q8_0 rows)",
        reads: &[GgmlType::Q8_0],
    },
];

/// Every part of `spec` (with its tensors `model`) that no program runs:
/// layer by layer in layer order, then the tensor formats, then the
/// model-wide ones. Empty for a file a program runs whole.
#[must_use]
pub fn check(spec: &ModelSpec, model: &ModelTensors) -> Vec<Unimplemented> {
    check_with(spec, model, AVAILABLE)
}

/// [`check`] against the table `available` instead of [`AVAILABLE`].
#[must_use]
pub fn check_with(
    spec: &ModelSpec,
    model: &ModelTensors,
    available: &[Available],
) -> Vec<Unimplemented> {
    let program = program_of(spec.arch);
    let mut out = Vec::new();
    let mut at = |layer: Option<usize>, need: Need| {
        let f = Unimplemented {
            layer,
            feature: need.to_string(),
        };
        if !out.contains(&f) {
            out.push(f);
        }
    };
    // With no program, the row of any program covers a need.
    let runs = |need: &Need| {
        available
            .iter()
            .any(|a| program.is_none_or(|p| a.program == p) && (a.runs)(need))
    };
    let (layered, wide): (Vec<_>, Vec<_>) = needs(spec).into_iter().partition(|(_, l)| l.is_some());
    for (need, l) in layered {
        if !runs(&need) {
            at(l.map(|l| l as usize), need);
        }
    }
    for t in &model.tensors {
        let card = match program {
            // The V4.1 chain's rule: a stack of a type no card format loads.
            // The glm5next and qwen4exp bodies serve every stack on the host,
            // whose load refuses a type with no host kernel.
            Some(Program::Deepseek41Chain | Program::Glm5nextBody | Program::Qwen38Body) => {
                CardFormat::of(t.ty)
            }
            // A whole-card program, or the one still to be written, needs a
            // card expert kernel for the type.
            Some(Program::Qwen3moeBody) | None => CardFormat::of_routed(t.ty),
        };
        if t.role == Role::RoutedExperts && card.is_none() {
            at(t.layer, Need::RoutedFormat(t.ty));
        }
        let matrix = t.dims.len() >= 2 && t.ty != GgmlType::F32;
        let pins: Vec<&TypePin> = TYPE_PINS
            .iter()
            .filter(|pin| {
                program.is_none_or(|p| pin.program == p)
                    && pin.role == t.role
                    && (!pin.matrices || matrix)
            })
            .collect();
        let refused: Vec<&TypePin> = match program {
            Some(_) => pins
                .into_iter()
                .filter(|pin| !pin.reads.contains(&t.ty))
                .collect(),
            // No program: an item only when no program's pin reads the type,
            // named by the first pin.
            None if pins.iter().all(|pin| !pin.reads.contains(&t.ty)) => {
                pins.into_iter().take(1).collect()
            }
            None => Vec::new(),
        };
        for pin in refused {
            at(
                t.layer,
                Need::WeightFormat {
                    what: pin.what,
                    ty: t.ty,
                },
            );
        }
    }
    for (need, _) in wide {
        if !runs(&need) {
            at(None, need);
        }
    }
    match program {
        None => at(None, Need::Program(spec.arch.name())),
        Some(Program::Deepseek41Chain)
            if !spec
                .layers
                .iter()
                .any(|l| l.extras.contains(&Extra::Engram)) =>
        {
            at(None, Need::NoEngram);
        }
        Some(_) => {}
    }
    if !tokenizer::runs_pre_tokenizer(&spec.chat.pre) {
        at(None, Need::PreTokenizer(spec.chat.pre.clone()));
    }
    if spec.chat.template.is_some() && spec.chat.tools.is_none() {
        at(None, Need::ToolParser);
    }
    out
}
