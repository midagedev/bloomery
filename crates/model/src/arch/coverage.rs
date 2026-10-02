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

use crate::placement::{CardFormat, ModelTensor, ModelTensors, Role, Unimplemented};

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
        Arch::Qwen3Moe => Some(Program::Qwen3moeBody),
        Arch::Qwen35Moe | Arch::Qwen35 => Some(Program::Qwen35Body),
        Arch::Glm5Next => Some(Program::Glm5nextBody),
        Arch::Qwen4Exp => Some(Program::Qwen38Body),
    }
}

/// One thing a program runs: the need it covers, and where it lives.
#[derive(Clone, Copy)]
pub struct Available {
    /// The programs that run it.
    pub programs: &'static [Program],
    /// The constant or function that owns the instance.
    pub at: &'static str,
    /// Whether this row covers `need`.
    pub runs: fn(&Need) -> bool,
}

/// Everything the programs run that a [`Need`] names. A need no row covers
/// is an item of [`check`]; a row removed makes its needs items.
pub const AVAILABLE: &[Available] = &[
    Available {
        programs: &[Program::Deepseek41Chain],
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
        programs: &[Program::Deepseek41Chain],
        at: "gpu-deepseek41/src/compress.rs WIDTH",
        runs: |n| matches!(n, Need::Compress { width: 512 }),
    },
    Available {
        programs: &[Program::Deepseek41Chain],
        at: "gpu-deepseek41/src/indexer.rs HEADS, HEAD_DIM; index_key.rs WIDTH",
        runs: |n| matches!(n, Need::StreamIndex { heads: 32, d: 128 }),
    },
    Available {
        programs: &[Program::Deepseek41Chain],
        at: "models/src/shape.rs ROUTERS, the Ds41 body",
        runs: |n| routes(n, &[RouterBody::Ds41]),
    },
    Available {
        programs: &[Program::Deepseek41Chain],
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
        programs: &[Program::Deepseek41Chain],
        at: "gpu-deepseek41/src/hc.rs HC_STREAMS",
        runs: |n| matches!(n, Need::Hc { streams: 4 }),
    },
    Available {
        programs: &[Program::Deepseek41Chain],
        at: "gpu-deepseek41/src/engram_gate.rs ROW",
        runs: |n| matches!(n, Need::Engram { row: 5120 }),
    },
    Available {
        programs: &[Program::Qwen3moeBody, Program::Qwen35Body],
        at: "models/src/shape.rs GQA",
        runs: |n| matches!(n, Need::Gqa { head, group } if gqa_row(*head, *group).is_some()),
    },
    Available {
        programs: &[Program::Qwen3moeBody],
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
        programs: &[Program::Qwen3moeBody, Program::Qwen35Body],
        at: "models/src/shape.rs ROUTERS, the Qwen3moe and Qwen35moe bodies",
        runs: |n| routes(n, &[RouterBody::Qwen3moe, RouterBody::Qwen35moe]),
    },
    Available {
        programs: &[Program::Qwen35Body],
        at: "gpu/src/arch/qwen3moe/plan.rs delta_shape (Body35's delta layers)",
        runs: |n| {
            matches!(
                n,
                Need::DeltaRule {
                    kind: DeltaKind::Gdn {
                        khead_map: models::KHeadMap::Tiled,
                        gate: models::GdnGate::Silu
                    },
                    k_heads,
                    v_heads,
                    d: 128,
                    conv: 4,
                } if *k_heads > 0 && v_heads.is_multiple_of(*k_heads)
            )
        },
    },
    Available {
        programs: &[Program::Qwen35Body],
        at: "gpu/src/arch/qwen3moe/plan.rs moe_fits (Body35's shared expert, one more slot)",
        runs: |n| {
            matches!(
                n,
                Need::Shared {
                    ff: 512,
                    sigmoid_gate: true
                }
            )
        },
    },
    Available {
        programs: &[Program::Qwen35Body],
        at: "gpu/src/rope_neox.rs head_norm_neox_append_256 (Body35: 64 of 256 dims)",
        runs: |n| {
            matches!(
                n,
                Need::QkRope {
                    qk_norm: true,
                    head: 256,
                    mode: mode @ RopeMode::Imrope { .. },
                    dims: 64
                } if turns_as_neox(*mode, 64)
            )
        },
    },
    Available {
        programs: &[Program::Qwen35Body],
        at: "gpu/src/arch/qwen3moe/dispatch.rs gated_256 (Body35's output gate)",
        runs: |n| matches!(n, Need::OutGate),
    },
    Available {
        programs: &[Program::Qwen35Body],
        at: "gpu/src/arch/qwen3moe/plan.rs dense_fits (Body35's dense FFN: one expert, one slot)",
        runs: |n| matches!(n, Need::DenseFfn { ff } if ff.is_multiple_of(256)),
    },
    Available {
        programs: &[Program::Qwen35Body],
        at: "gpu/src/arch/qwen3moe/scratch.rs RecStore (Body35's delta stores)",
        runs: |n| matches!(n, Need::RecurrentState),
    },
    Available {
        programs: &[Program::Qwen35Body],
        at: "gpu/src/arch/qwen3moe/program.rs (Body35: a mixer per layer's plan)",
        runs: |n| matches!(n, Need::MixedTrunk),
    },
    Available {
        programs: &[Program::Glm5nextBody],
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
        programs: &[Program::Glm5nextBody],
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
        programs: &[Program::Glm5nextBody],
        at: "gpu/src/latent.rs INDEX_HEAD (index_key_ln_append)",
        runs: |n| matches!(n, Need::KeyLayerNorm),
    },
    Available {
        programs: &[Program::Glm5nextBody],
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
        programs: &[Program::Glm5nextBody],
        at: "models/src/shape.rs ROUTERS, the Glm5next body",
        runs: |n| routes(n, &[RouterBody::Glm5next]),
    },
    Available {
        programs: &[Program::Glm5nextBody],
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
        programs: &[Program::Glm5nextBody],
        at: "gpu-glm5next/src/ffn.rs (the dense block)",
        runs: |n| matches!(n, Need::DenseFfn { .. }),
    },
    Available {
        programs: &[Program::Glm5nextBody],
        at: "gpu-deepseek41/src/hc.rs HC_STREAMS (hc_pre_q8_0)",
        runs: |n| matches!(n, Need::Hc { streams: 4 }),
    },
    Available {
        programs: &[Program::Glm5nextBody],
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
        programs: &[Program::Glm5nextBody],
        at: "gpu-glm5next/src/kda.rs (the state and the conv ring, one lane)",
        runs: |n| matches!(n, Need::RecurrentState),
    },
    Available {
        programs: &[Program::Glm5nextBody],
        at: "gpu-glm5next/src/program.rs (a mixer program per layer)",
        runs: |n| matches!(n, Need::MixedTrunk),
    },
    Available {
        programs: &[Program::Qwen38Body],
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
        programs: &[Program::Qwen38Body],
        at: "models/src/shape.rs GQA, the `_256_p4` flash (Body38)",
        runs: |n| matches!(n, Need::Gqa { head: 256, group: 12 } if gqa_row(256, 12).is_some()),
    },
    Available {
        programs: &[Program::Qwen38Body],
        at: "gpu/src/rope_neox.rs head_norm_neox_append_256 (Body38, 64 of 256 dims)",
        runs: |n| {
            matches!(
                n,
                Need::QkRope {
                    qk_norm: true,
                    head: 256,
                    mode: mode @ RopeMode::Imrope { .. },
                    dims: 64
                } if turns_as_neox(*mode, 64)
            )
        },
    },
    Available {
        programs: &[Program::Qwen38Body],
        at: "gpu/src/q38.rs gqa_out_gate_f32",
        runs: |n| matches!(n, Need::OutGate),
    },
    Available {
        programs: &[Program::Qwen38Body],
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
        programs: &[Program::Qwen38Body],
        at: "models/src/shape.rs ROUTERS, the Qwen35moe body's `_512` instance (Body38)",
        runs: |n| routes(n, &[RouterBody::Qwen35moe]),
    },
    Available {
        programs: &[Program::Qwen38Body],
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
        programs: &[Program::Qwen38Body],
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
        programs: &[Program::Qwen38Body],
        at: "gpu/src/arch/qwen3moe/program38.rs head_mix (the head's mix, no inject)",
        runs: |n| matches!(n, Need::HcGatedHead { rank: 320 }),
    },
    Available {
        programs: &[Program::Qwen38Body],
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
        programs: &[Program::Qwen38Body],
        at: "engram/src/hash.rs ple_rows_into (the image placeholder refused by name)",
        runs: |n| matches!(n, Need::ImageToken(_)),
    },
    Available {
        programs: &[Program::Qwen38Body],
        at: "gpu/src/arch/qwen3moe/scratch38.rs Store38 (the state and the conv ring, one lane)",
        runs: |n| matches!(n, Need::RecurrentState),
    },
    Available {
        programs: &[Program::Qwen38Body],
        at: "gpu/src/arch/qwen3moe/program38.rs (a mixer per layer's plan)",
        runs: |n| matches!(n, Need::MixedTrunk),
    },
];

/// Whether a rope of `mode` over the first `dims` values of a head turns
/// them as NEOX does at a text-only position: NEOX itself, or IMROPE whose
/// sections cover the `dims / 2` pairs (a text-only position gives every
/// section the same position). The one test the head-256 kernels' plans and
/// their coverage rows share.
#[must_use]
pub fn turns_as_neox(mode: RopeMode, dims: u32) -> bool {
    match mode {
        RopeMode::Neox => true,
        RopeMode::Imrope { sections } => {
            sections.iter().map(|&s| u64::from(s)).sum::<u64>() == u64::from(dims / 2)
        }
        RopeMode::NormTail => false,
    }
}

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
    /// Only the tensors whose name this takes; `None`: every one of the role.
    names: Option<fn(&str) -> bool>,
    what: &'static str,
    reads: &'static [GgmlType],
}

/// The per-tensor type pins of the programs (the qwen3moe body's `kq_site`
/// calls in `Body::load`, the head's Q6_K gemv in `gpu/src/head.rs`, the
/// card embedding's `embed_rows_q4k`; `Body35`'s site types, its `PROJ`,
/// `ROUTED`, `ROUTED_DOWN`, `EMBED` and `HEAD_TY` in
/// `gpu/src/arch/qwen3moe/body35.rs`, each a launch of `gpu/src/site.rs`;
/// the V4.1 chain's attention gemvs, whose
/// head-split sites in `chain/attn.rs` take q3_K or q8_0, the same head, and
/// the hyper-connection fn `hc.rs` reads as q3_K words, `hc_f32.rs` as f32;
/// the glm5next body's sites, each a q8_0 gemv or `hc_pre_q8_0`, the head's
/// q8_0 arm, and the embedding row the host dequantizes).
const TYPE_PINS: &[TypePin] = &[
    TypePin {
        program: Program::Qwen3moeBody,
        role: Role::Attention,
        matrices: true,
        names: None,
        what: "attention matrices (the body reads q4_K and q6_K)",
        reads: &[GgmlType::Q4_K, GgmlType::Q6_K],
    },
    TypePin {
        program: Program::Qwen3moeBody,
        role: Role::DenseFfn,
        matrices: true,
        names: Some(|n| !n.ends_with(".ffn_down.weight")),
        what: "dense FFN gate and up (the body reads q4_K)",
        reads: &[GgmlType::Q4_K],
    },
    TypePin {
        program: Program::Qwen3moeBody,
        role: Role::DenseFfn,
        matrices: true,
        names: Some(|n| n.ends_with(".ffn_down.weight")),
        what: "dense FFN down (the body reads q4_K and q6_K)",
        reads: &[GgmlType::Q4_K, GgmlType::Q6_K],
    },
    TypePin {
        program: Program::Qwen3moeBody,
        role: Role::Head,
        matrices: true,
        names: None,
        what: "output head (the head reads q6_K)",
        reads: &[GgmlType::Q6_K],
    },
    TypePin {
        program: Program::Qwen3moeBody,
        role: Role::TokenEmbedding,
        matrices: true,
        names: None,
        what: "token embedding (the card reads q4_K rows)",
        reads: &[GgmlType::Q4_K],
    },
    TypePin {
        program: Program::Qwen35Body,
        role: Role::Attention,
        matrices: true,
        names: None,
        what: "attention and delta-rule matrices (the body reads q3_K, q4_K, q5_K, q6_K and q8_0)",
        reads: Q35_PROJ,
    },
    TypePin {
        program: Program::Qwen35Body,
        role: Role::DenseFfn,
        matrices: true,
        names: None,
        what: "dense FFN (the body reads q3_K, q4_K, q5_K, q6_K and q8_0)",
        reads: Q35_PROJ,
    },
    TypePin {
        program: Program::Qwen35Body,
        role: Role::SharedExpert,
        matrices: true,
        names: Some(|n| {
            n.ends_with(".ffn_gate_shexp.weight") || n.ends_with(".ffn_up_shexp.weight")
        }),
        what: "shared expert gate and up, joined into the routed stacks (the body reads q4_K)",
        reads: &[GgmlType::Q4_K],
    },
    TypePin {
        program: Program::Qwen35Body,
        role: Role::SharedExpert,
        matrices: true,
        names: Some(|n| n.ends_with(".ffn_down_shexp.weight")),
        what: "shared expert down, joined into the routed stack (the body reads q4_K and q6_K)",
        reads: &[GgmlType::Q4_K, GgmlType::Q6_K],
    },
    TypePin {
        program: Program::Qwen35Body,
        role: Role::Head,
        matrices: true,
        names: None,
        what: "output head (the head reads q6_K, q4_K and q8_0)",
        reads: &[GgmlType::Q6_K, GgmlType::Q4_K, GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Qwen35Body,
        role: Role::TokenEmbedding,
        matrices: true,
        names: None,
        what: "token embedding (the card reads q4_K, q5_K and q6_K rows and q8_0 planes)",
        reads: &[
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
        ],
    },
    TypePin {
        program: Program::Deepseek41Chain,
        role: Role::Attention,
        matrices: true,
        names: None,
        what: "attention matrices (the chain reads q3_K and q8_0)",
        reads: &[GgmlType::Q3_K, GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Deepseek41Chain,
        role: Role::Head,
        matrices: true,
        names: None,
        what: "output head (the head reads q6_K)",
        reads: &[GgmlType::Q6_K],
    },
    TypePin {
        program: Program::Deepseek41Chain,
        role: Role::HyperConnection,
        matrices: true,
        names: None,
        what: "hyper-connection fn (the chain reads q3_K and f32)",
        reads: &[GgmlType::Q3_K],
    },
    TypePin {
        program: Program::Glm5nextBody,
        role: Role::Attention,
        matrices: true,
        names: None,
        what: "attention matrices (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Glm5nextBody,
        role: Role::HyperConnection,
        matrices: true,
        names: None,
        what: "hyper-connection fn (hc_pre_q8_0 reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Glm5nextBody,
        role: Role::DenseFfn,
        matrices: true,
        names: None,
        what: "dense block (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Glm5nextBody,
        role: Role::SharedExpert,
        matrices: true,
        names: None,
        what: "shared expert (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Glm5nextBody,
        role: Role::Head,
        matrices: true,
        names: None,
        what: "output head (the head reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Glm5nextBody,
        role: Role::TokenEmbedding,
        matrices: true,
        names: None,
        what: "token embedding (the host reads q8_0 rows)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Qwen38Body,
        role: Role::Attention,
        matrices: true,
        names: Some(|n| !indexer_projection(n)),
        what: "attention matrices (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Qwen38Body,
        role: Role::Attention,
        matrices: true,
        names: Some(indexer_projection),
        what: "indexer projections (the body reads bf16 widened to f32)",
        reads: &[GgmlType::BF16],
    },
    TypePin {
        program: Program::Qwen38Body,
        role: Role::HyperConnection,
        matrices: true,
        names: None,
        what: "hyper-connection down and up (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Qwen38Body,
        role: Role::SharedExpert,
        matrices: true,
        names: None,
        what: "shared expert (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Qwen38Body,
        role: Role::Head,
        matrices: true,
        names: None,
        what: "output head (the head reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Qwen38Body,
        role: Role::TokenEmbedding,
        matrices: true,
        names: None,
        what: "token embedding (the card reads q8_0 rows)",
        reads: &[GgmlType::Q8_0],
    },
];

/// `Body35`'s projection types (`PROJ`): every attention, delta-rule and
/// dense FFN matrix.
const Q35_PROJ: &[GgmlType] = &[
    GgmlType::Q3_K,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q8_0,
];

/// A qwen4exp selector's key or query projection, which `Body38` reads as
/// bf16 widened to f32 (`plan38::plans`), where it reads every other
/// attention matrix as q8_0.
fn indexer_projection(name: &str) -> bool {
    name.ends_with(".indexer.k_proj.weight") || name.ends_with(".indexer.q_proj.weight")
}

/// The type pins of `program` (every program's when `None`) that tensor `t`
/// is of a type none of reads, as items: with a program, each pin of its role
/// (and name) whose types do not hold `t`'s; with none, the first such pin
/// when no program's pin reads the type.
fn weight_formats(program: Option<Program>, t: &ModelTensor) -> Vec<Need> {
    let matrix = t.dims.len() >= 2 && t.ty != GgmlType::F32;
    let pins: Vec<&TypePin> = TYPE_PINS
        .iter()
        .filter(|pin| {
            program.is_none_or(|p| pin.program == p)
                && pin.role == t.role
                && (!pin.matrices || matrix)
                && pin.names.is_none_or(|takes| takes(&t.name))
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
    refused
        .into_iter()
        .map(|pin| Need::WeightFormat {
            what: pin.what,
            ty: t.ty,
        })
        .collect()
}

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
            .any(|a| program.is_none_or(|p| a.programs.contains(&p)) && (a.runs)(need))
    };
    let (layered, wide): (Vec<_>, Vec<_>) = needs(spec).into_iter().partition(|(_, l)| l.is_some());
    for (need, l) in layered {
        if !runs(&need) {
            at(l.map(|l| l as usize), need);
        }
    }
    for t in &model.tensors {
        let card = match program {
            // The V4.1 chain's and the glm5next body's rule: a stack of a type
            // no card format loads. A layer whose stacks the program's card
            // experts do not read keeps them on the host, whose load refuses
            // a type with no host kernel; the qwen4exp body serves every
            // stack on the host.
            Some(Program::Deepseek41Chain | Program::Glm5nextBody | Program::Qwen38Body) => {
                CardFormat::of(t.ty)
            }
            // A whole-card program, or the one still to be written, needs a
            // card expert kernel for the type.
            Some(Program::Qwen3moeBody | Program::Qwen35Body) | None => CardFormat::of_routed(t.ty),
        };
        if t.role == Role::RoutedExperts && card.is_none() {
            at(t.layer, Need::RoutedFormat(t.ty));
        }
        for need in weight_formats(program, t) {
            at(t.layer, need);
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

#[cfg(test)]
mod tests {
    use gguf::GgmlType;
    use models::{Arch, Need};

    use super::{Program, program_of, weight_formats};
    use crate::placement::{ModelTensor, Role};

    /// A layer-0 matrix `name` of role `role` and type `ty`.
    fn matrix(name: &str, role: Role, ty: GgmlType) -> ModelTensor {
        ModelTensor {
            name: format!("blk.0.{name}"),
            shard: 0,
            layer: Some(0),
            role,
            ty,
            dims: vec![256, 256],
            file_bytes: 0,
            gathered_rows: None,
        }
    }

    /// The items `program` lists for `t`'s type, as their text.
    fn items(program: Program, t: &ModelTensor) -> Vec<String> {
        weight_formats(Some(program), t)
            .iter()
            .map(Need::to_string)
            .collect()
    }

    /// qwen35 and qwen35moe files are `Body35`'s, qwen3moe files `Body`'s.
    #[test]
    fn each_qwen_body_is_its_own_program() {
        assert_eq!(program_of(Arch::Qwen35), Some(Program::Qwen35Body));
        assert_eq!(program_of(Arch::Qwen35Moe), Some(Program::Qwen35Body));
        assert_eq!(program_of(Arch::Qwen3Moe), Some(Program::Qwen3moeBody));
    }

    /// A Q8_0 qwen35 file is no item: `Body35` launches Q8_0 at every
    /// projection, the head and the embedding; the qwen3moe body still
    /// refuses Q8_0 attention by name.
    #[test]
    fn a_q8_0_qwen35_file_is_no_item() {
        let q8 = GgmlType::Q8_0;
        for t in [
            matrix("attn_qkv.weight", Role::Attention, q8),
            matrix("attn_q.weight", Role::Attention, q8),
            matrix("ssm_out.weight", Role::Attention, q8),
            matrix("ffn_down.weight", Role::DenseFfn, q8),
            ModelTensor {
                name: "output.weight".into(),
                layer: None,
                ..matrix("", Role::Head, q8)
            },
            ModelTensor {
                name: "token_embd.weight".into(),
                layer: None,
                ..matrix("", Role::TokenEmbedding, q8)
            },
            matrix("ffn_down_shexp.weight", Role::SharedExpert, GgmlType::Q6_K),
        ] {
            assert_eq!(
                items(Program::Qwen35Body, &t),
                Vec::<String>::new(),
                "{}",
                t.name
            );
        }
        let attn = matrix("attn_q.weight", Role::Attention, q8);
        assert_eq!(
            items(Program::Qwen3moeBody, &attn),
            ["q8_0 attention matrices (the body reads q4_K and q6_K)"]
        );
    }

    /// A type `Body35` cannot launch is still an item by name: bf16
    /// attention and dense FFN, a q8_0 shared expert gate (joined into a
    /// q4_K stack).
    #[test]
    fn a_type_body35_cannot_launch_is_an_item() {
        let bf16 = GgmlType::BF16;
        assert_eq!(
            items(
                Program::Qwen35Body,
                &matrix("attn_qkv.weight", Role::Attention, bf16)
            ),
            [
                "bf16 attention and delta-rule matrices (the body reads q3_K, q4_K, q5_K, q6_K and q8_0)"
            ]
        );
        assert_eq!(
            items(
                Program::Qwen35Body,
                &matrix("ffn_up.weight", Role::DenseFfn, bf16)
            ),
            ["bf16 dense FFN (the body reads q3_K, q4_K, q5_K, q6_K and q8_0)"]
        );
        assert_eq!(
            items(
                Program::Qwen35Body,
                &matrix("ffn_gate_shexp.weight", Role::SharedExpert, GgmlType::Q8_0)
            ),
            ["q8_0 shared expert gate and up, joined into the routed stacks (the body reads q4_K)"]
        );
    }
}
