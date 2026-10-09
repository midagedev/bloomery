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
use models::shape::{RouterBody, gqa_row, gqa_row_v, select_router};
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
    /// The mimo2 body (`crates/gpu-mimo2`): every routed expert on the host
    /// tier, none on a card.
    Mimo2Body,
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
        runs: |n| {
            matches!(
                n,
                Need::Gqa {
                    head,
                    value,
                    group,
                    window: None,
                    sinks: false,
                    scaled: false,
                } if value == head && gqa_row(*head, *group).is_some()
            )
        },
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
        programs: &[Program::Glm5nextBody, Program::Mimo2Body],
        at: "models/src/shape.rs ROUTERS, the BiasedSigmoid body (gpu-deepseek41 router: glm5next at 288 experts, mimo2 at 256)",
        runs: |n| routes(n, &[RouterBody::BiasedSigmoid]),
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
        programs: &[Program::Glm5nextBody, Program::Mimo2Body],
        at: "gpu-deepseek41/src/experts.rs enqueue_shexp_gate_up (the dense block: glm5next ffn.rs, mimo2 ffn.rs)",
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
        runs: |n| {
            matches!(
                n,
                Need::Gqa {
                    head: 256,
                    value: 256,
                    group: 12,
                    window: None,
                    sinks: false,
                    scaled: false,
                } if gqa_row(256, 12).is_some()
            )
        },
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
    Available {
        programs: &[Program::Mimo2Body],
        at: "models/src/shape.rs GQA, the K192 row of a full layer (gpu/src/flash_gqa.rs \
             gqa_flash_seg_k192 and gqa_flash_merge)",
        runs: |n| {
            matches!(
                n,
                Need::Gqa {
                    head: 192,
                    value: 128,
                    group,
                    window: None,
                    sinks: false,
                    scaled: true,
                } if k192_row(*group)
            )
        },
    },
    Available {
        programs: &[Program::Mimo2Body],
        at: "models/src/shape.rs GQA, the K192 row of a window layer (gpu/src/flash_gqa.rs \
             gqa_flash_seg_k192 and gqa_flash_merge_sink)",
        runs: |n| {
            matches!(
                n,
                Need::Gqa {
                    head: 192,
                    value: 128,
                    group,
                    window: Some(_),
                    sinks: true,
                    scaled: true,
                } if k192_row(*group)
            )
        },
    },
    Available {
        programs: &[Program::Mimo2Body],
        at: "gpu/src/rope_neox.rs HEAD_K192, ROT_K192 (neox_append_k192: no QK norm, 64 of 192 dims)",
        runs: |n| {
            matches!(
                n,
                Need::QkRope {
                    qk_norm: false,
                    head: 192,
                    mode: RopeMode::Neox,
                    dims: 64
                }
            )
        },
    },
];

/// Whether the K192 flash row ([`gqa_row_v`] at head 192, value 128) takes
/// `group` query heads a key head, with the window, sinks and value scale
/// the MiMo rows above ask of it.
fn k192_row(group: u32) -> bool {
    gqa_row_v(192, 128, group).is_some_and(|r| r.window && r.sinks && r.value_scale)
}

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
/// the glm5next body's sites, each a q8_0 gemv or `hc_pre_q8_0`, the common
/// head's q8_0, q6_K and q4_K arms, and the embedding row the host
/// dequantizes).
const TYPE_PINS: &[TypePin] = &[
    TypePin {
        program: Program::Qwen3moeBody,
        role: Role::Attention,
        matrices: true,
        names: Some(|n| !n.ends_with(".attn_v.weight")),
        what: "attention q, k and output matrices (the body reads q4_K)",
        reads: &[GgmlType::Q4_K],
    },
    TypePin {
        program: Program::Qwen3moeBody,
        role: Role::Attention,
        matrices: true,
        names: Some(|n| n.ends_with(".attn_v.weight")),
        what: "attention value matrices (the body reads q4_K and q6_K)",
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
        program: Program::Qwen3moeBody,
        role: Role::RoutedExperts,
        matrices: true,
        names: Some(routed_gate_up),
        what: "routed experts gate and up (the body reads q4_K)",
        reads: &[GgmlType::Q4_K],
    },
    TypePin {
        program: Program::Qwen3moeBody,
        role: Role::RoutedExperts,
        matrices: true,
        names: Some(routed_down),
        what: "routed experts down (the body reads q4_K and q6_K)",
        reads: &[GgmlType::Q4_K, GgmlType::Q6_K],
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
        what: "token embedding (the card reads q3_K, q4_K, q5_K and q6_K rows and q8_0 planes)",
        reads: &[
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
        ],
    },
    TypePin {
        program: Program::Qwen35Body,
        role: Role::RoutedExperts,
        matrices: true,
        names: Some(routed_gate_up),
        what: "routed experts gate and up, each shared expert joined in (the body reads q4_K)",
        reads: &[GgmlType::Q4_K],
    },
    TypePin {
        program: Program::Qwen35Body,
        role: Role::RoutedExperts,
        matrices: true,
        names: Some(routed_down),
        what: "routed experts down, each shared expert joined in (the body reads q4_K and q6_K)",
        reads: &[GgmlType::Q4_K, GgmlType::Q6_K],
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
        what: "output head (the head reads q8_0 planes, q6_K or q4_K word planes)",
        // The common head's own set (`head_out_w` in `gpu/src/head.rs`).
        reads: &[GgmlType::Q8_0, GgmlType::Q6_K, GgmlType::Q4_K],
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
        what: "output head (the head reads q8_0 planes, q6_K or q4_K word planes)",
        reads: &[GgmlType::Q8_0, GgmlType::Q6_K, GgmlType::Q4_K],
    },
    TypePin {
        program: Program::Qwen38Body,
        role: Role::TokenEmbedding,
        matrices: true,
        names: None,
        what: "token embedding (the card reads q8_0 rows)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Mimo2Body,
        role: Role::Attention,
        matrices: true,
        names: None,
        what: "attention matrices (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Mimo2Body,
        role: Role::DenseFfn,
        matrices: true,
        names: None,
        what: "dense block (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Mimo2Body,
        role: Role::Head,
        matrices: true,
        names: None,
        what: "output head (the head reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Mimo2Body,
        role: Role::TokenEmbedding,
        matrices: true,
        names: None,
        what: "token embedding (the host reads q8_0 rows)",
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

/// A routed expert stack's gate or up rows (`ffn_gate_exps`/`ffn_up_exps`),
/// the part both whole-card bodies launch q4_K alone.
fn routed_gate_up(name: &str) -> bool {
    name.ends_with(".ffn_gate_exps.weight") || name.ends_with(".ffn_up_exps.weight")
}

/// A routed expert stack's down rows (`ffn_down_exps`), the part both
/// whole-card bodies launch q4_K or q6_K.
fn routed_down(name: &str) -> bool {
    name.ends_with(".ffn_down_exps.weight")
}

/// A routed stack no side of a plan runs: not one `card_rule` says a card
/// loads, and no fused qdot kernel serves it on the host at its row width —
/// `HostLayer::build`'s own admission ([`qdot::fuses`], the check the union
/// call makes of every stack it serves), asked here at the stack's first
/// dim, its `k`. A program with no card expert passes `|_| false`.
fn routed_unrun(t: &ModelTensor, card_rule: fn(GgmlType) -> bool) -> bool {
    let k = t.dims.first().copied().unwrap_or(0);
    !card_rule(t.ty) && !qdot::fuses(t.ty, usize::try_from(k).unwrap_or(usize::MAX))
}

/// The qwen4exp body's card rule: the types its card experts load
/// ([`card_routed`](crate::arch::qwen35moe::place::card_routed)).
fn qwen38_card_routed(ty: GgmlType) -> bool {
    crate::arch::qwen35moe::place::card_routed(ty).is_some()
}

/// The glm5next body's card rule: the types its card experts load
/// ([`card_routed`](crate::arch::glm5next::place::card_routed)).
fn glm5next_card_routed(ty: GgmlType) -> bool {
    crate::arch::glm5next::place::card_routed(ty).is_some()
}

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
        if t.role == Role::RoutedExperts {
            match program {
                // The V4.1 chain's rule: a stack of a type no card format
                // loads. A layer whose stacks the chain's card experts do not
                // read keeps them on the host, whose load refuses a type with
                // no host kernel.
                Some(Program::Deepseek41Chain) => {
                    if CardFormat::of(t.ty).is_none() {
                        at(t.layer, Need::RoutedFormat(t.ty));
                    }
                }
                // The glm5next body's rule: a stack passes when the program
                // can run it — on the card by its expert rule (`card_routed`)
                // or on the host by a qdot fused kernel at the stack's row
                // width ([`routed_unrun`], the one owner). A layer with a
                // stack the card experts do not read keeps all its experts on
                // the host.
                Some(Program::Glm5nextBody) => {
                    if routed_unrun(t, glm5next_card_routed) {
                        at(t.layer, Need::RoutedFormat(t.ty));
                    }
                }
                // The qwen4exp body's rule: a stack passes when the program
                // can run it — on the card by its expert rule (`card_routed`)
                // or on the host by a qdot fused kernel at the stack's row
                // width ([`routed_unrun`], the one owner).
                Some(Program::Qwen38Body) => {
                    if routed_unrun(t, qwen38_card_routed) {
                        at(t.layer, Need::RoutedFormat(t.ty));
                    }
                }
                // The mimo2 body's rule: the same, with no card expert — a
                // stack passes only on the host by a qdot fused kernel.
                Some(Program::Mimo2Body) => {
                    if routed_unrun(t, |_| false) {
                        at(t.layer, Need::RoutedFormat(t.ty));
                    }
                }
                // The one still to be written needs a card expert kernel for
                // the type.
                None => {
                    if CardFormat::of_routed(t.ty).is_none() {
                        at(t.layer, Need::RoutedFormat(t.ty));
                    }
                }
                // A whole-card program launches every stack itself, at its
                // pins' types: a part of another type is one of their items,
                // by name and part (the default expert rule's card formats
                // would pass q3_K and q6_K gate·up to the load's refusal).
                Some(Program::Qwen3moeBody | Program::Qwen35Body) => {}
            }
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
    use models::{Arch, ChatSpec, ModelSpec, Need};

    use super::{Program, program_of, weight_formats};
    use crate::placement::{ModelTensor, ModelTensors, Role};

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

    /// The Qwen bodies launch a flash with the value head the key head's
    /// width and no window, sinks or value scale: a need that differs in
    /// any of the four is no row's, so a MiMo layer is never read as covered.
    #[test]
    fn a_gqa_need_beyond_what_the_qwen_bodies_launch_is_uncovered() {
        use super::AVAILABLE;
        let plain = Need::Gqa {
            head: 128,
            value: 128,
            group: 8,
            window: None,
            sinks: false,
            scaled: false,
        };
        let runs = |n: &Need| {
            AVAILABLE
                .iter()
                .any(|a| a.programs.contains(&Program::Qwen3moeBody) && (a.runs)(n))
        };
        assert!(runs(&plain), "the plain head-128 flash is the body's");
        let Need::Gqa { head, group, .. } = plain else {
            unreachable!()
        };
        for (what, need) in [
            (
                "a narrower value head",
                Need::Gqa {
                    head,
                    value: 64,
                    group,
                    window: None,
                    sinks: false,
                    scaled: false,
                },
            ),
            (
                "a window",
                Need::Gqa {
                    head,
                    value: head,
                    group,
                    window: Some(128),
                    sinks: false,
                    scaled: false,
                },
            ),
            (
                "sinks",
                Need::Gqa {
                    head,
                    value: head,
                    group,
                    window: None,
                    sinks: true,
                    scaled: false,
                },
            ),
            (
                "a value scale",
                Need::Gqa {
                    head,
                    value: head,
                    group,
                    window: None,
                    sinks: false,
                    scaled: true,
                },
            ),
        ] {
            assert!(!runs(&need), "{what} read as covered: {need}");
        }
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
            ["q8_0 attention q, k and output matrices (the body reads q4_K)"]
        );
        let v = matrix("attn_v.weight", Role::Attention, q8);
        assert_eq!(
            items(Program::Qwen3moeBody, &v),
            ["q8_0 attention value matrices (the body reads q4_K and q6_K)"]
        );
    }

    /// The qwen3moe body's attention pins are its launches' by part: q, k and
    /// the output projection read q4_K alone, the values q4_K or q6_K, so a
    /// q6_K q or output is an item a looser one-list pin passes.
    #[test]
    fn the_qwen3moe_body_reads_attention_by_part() {
        for name in ["attn_q.weight", "attn_k.weight", "attn_output.weight"] {
            assert_eq!(
                items(
                    Program::Qwen3moeBody,
                    &matrix(name, Role::Attention, GgmlType::Q6_K)
                ),
                ["q6_K attention q, k and output matrices (the body reads q4_K)"],
                "{name}"
            );
        }
        let v = matrix("attn_v.weight", Role::Attention, GgmlType::Q6_K);
        assert_eq!(items(Program::Qwen3moeBody, &v), Vec::<String>::new());
    }

    /// The whole-card bodies' routed pins, by part: the gate and up stacks
    /// read q4_K alone, the down q4_K or q6_K — the parts the default expert
    /// rule's card formats pass (q3_K, q6_K, q8_0 gate·up; q3_K, q8_0 down)
    /// reach the loads' refusal otherwise.
    #[test]
    fn the_whole_card_bodies_read_routed_stacks_by_part() {
        for (program, join) in [
            (Program::Qwen3moeBody, ""),
            (Program::Qwen35Body, ", each shared expert joined in"),
        ] {
            let stack = |name: &str, part: &str, ty: GgmlType, reads: &str, item: bool| {
                let want = format!("{ty} routed experts {part}{join} (the body reads {reads})");
                assert_eq!(
                    items(program, &matrix(name, Role::RoutedExperts, ty)),
                    if item {
                        vec![want]
                    } else {
                        Vec::<String>::new()
                    },
                    "{name} {ty}"
                );
            };
            stack(
                "ffn_gate_exps.weight",
                "gate and up",
                GgmlType::Q4_K,
                "q4_K",
                false,
            );
            stack(
                "ffn_up_exps.weight",
                "gate and up",
                GgmlType::Q4_K,
                "q4_K",
                false,
            );
            stack(
                "ffn_down_exps.weight",
                "down",
                GgmlType::Q6_K,
                "q4_K and q6_K",
                false,
            );
            for ty in [
                GgmlType::Q3_K,
                GgmlType::Q6_K,
                GgmlType::Q8_0,
                GgmlType::Q5_K,
            ] {
                stack("ffn_gate_exps.weight", "gate and up", ty, "q4_K", true);
                stack("ffn_up_exps.weight", "gate and up", ty, "q4_K", true);
            }
            for ty in [GgmlType::Q3_K, GgmlType::Q8_0, GgmlType::Q5_K] {
                stack("ffn_down_exps.weight", "down", ty, "q4_K and q6_K", true);
            }
        }
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

    /// A routed stack of a qwen4exp file passes when the program can run it
    /// — on the card by the expert rule's types (the i-quants of the
    /// UD-Q3_K_XL file beside the K-quants) or on the host by a fused qdot
    /// kernel at its row width — and a stack no side runs is an item, named
    /// by its type.
    // PIN(2026-10-06): the Qwen38Body routed rule widened from
    // `CardFormat::of` to card-or-host so the UD-Q3_K_XL file's stacks
    /// (iq3_xxs, iq4_xs gate and up, iq4_nl down) are no items; the host
    /// admission is qdot's own (`routed_unrun`).
    #[test]
    fn a_qwen38_routed_stack_runs_on_a_card_or_the_host() {
        let routed = |ty: GgmlType, k: u64| ModelTensor {
            name: "blk.0.ffn_gate_exps.weight".to_string(),
            shard: 0,
            layer: Some(0),
            role: Role::RoutedExperts,
            ty,
            dims: vec![k, 640, 512],
            file_bytes: 0,
            gathered_rows: None,
        };
        // The card's: the K-quants and the i-quants at the file's widths.
        for (ty, k) in [
            (GgmlType::Q4_K, 2560),
            (GgmlType::Q5_K, 2560),
            (GgmlType::Q5_1, 640),
            (GgmlType::Q8_0, 640),
            (GgmlType::IQ3_XXS, 2560),
            (GgmlType::IQ4_XS, 2560),
            (GgmlType::IQ4_NL, 640),
        ] {
            assert!(
                !super::routed_unrun(&routed(ty, k), super::qwen38_card_routed),
                "{ty}"
            );
        }
        // The card's alone: a width no fused qdot kernel takes still runs on
        // the card, by type.
        assert!(
            !super::routed_unrun(&routed(GgmlType::Q4_K, 100), super::qwen38_card_routed),
            "q4_K at a width only the card takes"
        );
        // The host's alone: q6_K and q3_K stacks stay host-served.
        for ty in [GgmlType::Q6_K, GgmlType::Q3_K] {
            assert!(
                !super::routed_unrun(&routed(ty, 2560), super::qwen38_card_routed),
                "{ty}"
            );
        }
        // No side: a type no card expert kernel reads and no fused qdot
        // kernel serves.
        for ty in [GgmlType::IQ2_S, GgmlType::F32] {
            assert!(
                super::routed_unrun(&routed(ty, 2560), super::qwen38_card_routed),
                "{ty}"
            );
        }
    }

    /// The items [`super::check`] lists for a glm5next file holding
    /// `tensors` whose text holds `needle`: each its layer and text. A spec
    /// with no layers and no chat surface stands for the rest of the file,
    /// and `needle` leaves its items out. The check reads the CPU's features
    /// (`qdot::fuses`), so the host leg's answer is the x86-64-v3 host the
    /// engine itself requires.
    fn glm5next_items(tensors: Vec<ModelTensor>, needle: &str) -> Vec<(Option<usize>, String)> {
        let spec = ModelSpec {
            arch: Arch::Glm5Next,
            hidden: 4096,
            vocab: 154_880,
            ctx_train: 1_048_576,
            rms_eps: 1e-5,
            layers: Vec::new(),
            mtp: Vec::new(),
            hc: None,
            engram: None,
            chat: ChatSpec {
                pre: String::new(),
                template: None,
                tools: None,
                reasoning: None,
            },
        };
        let model = ModelTensors {
            tensors,
            layers: 5,
            experts: 288,
            experts_used: 8,
        };
        super::check(&spec, &model)
            .into_iter()
            .filter(|u| u.feature.contains(needle))
            .map(|u| (u.layer, u.feature))
            .collect()
    }

    /// The routed items for a glm5next file whose layers 3 and 4 hold
    /// routed stacks of `gate_up` (gate and up) and `down`, at
    /// GLM-5.3-Flash's widths (the gate and up map 4,096 to 2,048 values,
    /// the down 2,048 to 4,096, 288 experts).
    fn glm5next_routed_items(gate_up: GgmlType, down: GgmlType) -> Vec<(Option<usize>, String)> {
        let stack = |layer: usize, stem: &str, ty: GgmlType| {
            let (k, n) = if stem == "ffn_down_exps" {
                (2048, 4096)
            } else {
                (4096, 2048)
            };
            ModelTensor {
                name: format!("blk.{layer}.{stem}.weight"),
                shard: 0,
                layer: Some(layer),
                role: Role::RoutedExperts,
                ty,
                dims: vec![k, n, 288],
                file_bytes: 0,
                gathered_rows: None,
            }
        };
        let tensors = [3, 4]
            .into_iter()
            .flat_map(|l| {
                [
                    stack(l, "ffn_gate_exps", gate_up),
                    stack(l, "ffn_up_exps", gate_up),
                    stack(l, "ffn_down_exps", down),
                ]
            })
            .collect();
        glm5next_items(tensors, "routed experts on a card")
    }

    /// The items of the model-level tensor `name` of role `role` and type
    /// `ty` (GLM-5.3-Flash's 4,096 by 154,880 matrix) whose text holds
    /// `needle`.
    fn glm5next_whole_items(
        name: &str,
        role: Role,
        ty: GgmlType,
        needle: &str,
    ) -> Vec<(Option<usize>, String)> {
        let t = ModelTensor {
            name: name.to_string(),
            shard: 0,
            layer: None,
            role,
            ty,
            dims: vec![4096, 154_880],
            file_bytes: 0,
            gathered_rows: None,
        };
        glm5next_items(vec![t], needle)
    }

    /// A routed stack of a glm5next file passes when the program can run it
    /// — on the card by the card experts' types (q4_K, q5_K) or on the host
    /// by a fused qdot kernel at its row width, so a layer with a stack the
    /// card does not read keeps its experts on the host — and a type neither
    /// side runs is an item for each layer that holds it, named by its type,
    /// whether it is a gate, up or down.
    // PIN(2026-10-09): the glm5next routed rule widened from `CardFormat::of`
    // to card-or-host so a file's i-quant stacks (iq3_xxs, iq4_xs gate and
    // up or down) are no items; a bf16 stack, which `CardFormat::of` took but
    // no host kernel reads, is an item.
    #[test]
    fn a_glm5next_routed_stack_runs_on_a_card_or_the_host() {
        let none = Vec::<(Option<usize>, String)>::new();
        let item = |ty: GgmlType| -> Vec<(Option<usize>, String)> {
            [3, 4]
                .into_iter()
                .map(|l| (Some(l), Need::RoutedFormat(ty).to_string()))
                .collect()
        };
        let cases = [
            // The card's: q4_K gate and up with a q5_K down.
            (GgmlType::Q4_K, GgmlType::Q5_K, none.clone()),
            // The card's gate and up with a down only the host reads: the
            // layer's experts stay on the host.
            (GgmlType::Q4_K, GgmlType::Q6_K, none.clone()),
            // The host's alone: the i-quants of a layer no card expert reads.
            (GgmlType::IQ3_XXS, GgmlType::IQ4_XS, none.clone()),
            // The host's alone too, and no change: `CardFormat::of` takes
            // q3_K, so this passed before the card-or-host rule as well.
            (GgmlType::Q3_K, GgmlType::Q4_K, none.clone()),
            // No side: bf16 has no card expert and no fused qdot kernel, as
            // a gate and up or as a down.
            (GgmlType::BF16, GgmlType::Q4_K, item(GgmlType::BF16)),
            (GgmlType::Q4_K, GgmlType::BF16, item(GgmlType::BF16)),
        ];
        let wrong: Vec<_> = cases
            .iter()
            .map(|(gate_up, down, want)| {
                (gate_up, down, want, glm5next_routed_items(*gate_up, *down))
            })
            .filter(|(_, _, want, got)| want != &got)
            .collect();
        assert!(
            wrong.is_empty(),
            "(gate and up, down, want, got) of the cases off the rule: {wrong:#?}"
        );
    }

    /// The glm5next head's pin reads the set the head kernel reads
    /// (`gpu::head`'s `head_out_w`, which every head the body runs goes
    /// through): q8_0 planes, a q6_K or q4_K word plane; another type of
    /// `output` is an item, and the embedding's pin stays q8_0 alone.
    // PIN(2026-10-09): q6_K (and q4_K, which the same gemv reads) joined the
    // head's set so a file with a q6_K output loads.
    #[test]
    fn the_glm5next_head_pin_reads_the_head_kernels_set() {
        let head =
            |ty: GgmlType| glm5next_whole_items("output.weight", Role::Head, ty, "output head");
        for ty in [GgmlType::Q8_0, GgmlType::Q6_K, GgmlType::Q4_K] {
            assert_eq!(head(ty), Vec::<(Option<usize>, String)>::new(), "{ty}");
        }
        assert_eq!(
            head(GgmlType::Q5_K),
            [(
                None,
                "q5_K output head (the head reads q8_0 planes, q6_K or q4_K word planes)"
                    .to_string()
            )]
        );
        assert_eq!(
            glm5next_whole_items(
                "token_embd.weight",
                Role::TokenEmbedding,
                GgmlType::Q6_K,
                "token embedding"
            ),
            [(
                None,
                "q6_K token embedding (the host reads q8_0 rows)".to_string()
            )]
        );
    }

    /// The qwen4exp head's pin reads the set the head kernel reads
    /// (`gpu::head`'s `head_out_w`): q8_0 planes, a q6_K or q4_K word plane;
    /// another type of `output` is an item, and the embedding's pin stays
    /// q8_0 alone.
    // PIN(2026-10-06): q6_K (and q4_K, which the same gemv reads) joined the
    /// head's set so the UD-Q3_K_XL file's q6_K output loads.
    #[test]
    fn the_qwen38_head_pin_reads_the_head_kernels_set() {
        for ty in [GgmlType::Q8_0, GgmlType::Q6_K, GgmlType::Q4_K] {
            assert_eq!(
                items(
                    Program::Qwen38Body,
                    &ModelTensor {
                        name: "output.weight".into(),
                        layer: None,
                        ..matrix("", Role::Head, ty)
                    }
                ),
                Vec::<String>::new(),
                "{ty}"
            );
        }
        assert_eq!(
            items(
                Program::Qwen38Body,
                &ModelTensor {
                    name: "output.weight".into(),
                    layer: None,
                    ..matrix("", Role::Head, GgmlType::Q5_K)
                }
            ),
            ["q5_K output head (the head reads q8_0 planes, q6_K or q4_K word planes)"]
        );
        assert_eq!(
            items(
                Program::Qwen38Body,
                &ModelTensor {
                    name: "token_embd.weight".into(),
                    layer: None,
                    ..matrix("", Role::TokenEmbedding, GgmlType::Q6_K)
                }
            ),
            ["q6_K token embedding (the card reads q8_0 rows)"]
        );
    }

    /// MiMo-V2.6-Flash at the shapes `mimo2_meta` pins (48 layers, the nine
    /// full layers at 0, 5, 11, …, 47 and window layers between, key head 192
    /// over value head 128, 64 heads over 4 or 8 key heads, window 128 with
    /// sinks, value scale 0.707, a dense layer 0 of 16384 and routed layers
    /// of 256 experts, top 8, sigmoid with a selection bias), with the
    /// types the file carries: Q8_0 attention, dense block, head and
    /// embedding, F32 norms, sinks, routers and biases, MXFP4 stacks.
    fn mimo2_real() -> (models::ModelSpec, crate::placement::ModelTensors) {
        use crate::arch::mimo2::hparams::{Hparams, Kind};
        use crate::arch::mimo2::{names, spec};
        use crate::placement::ModelTensors;
        let full = [0usize, 5, 11, 17, 23, 29, 35, 41, 47];
        let kinds: Vec<Kind> = (0..48)
            .map(|l| {
                if full.contains(&l) {
                    Kind::Full
                } else {
                    Kind::Swa
                }
            })
            .collect();
        let hp = Hparams {
            n_layer: 48,
            n_trunk: 48,
            n_embd: 4096,
            n_head: 64,
            kv_heads: kinds
                .iter()
                .map(|k| if *k == Kind::Full { 4 } else { 8 })
                .collect(),
            head_k: 192,
            head_v: 128,
            window: 128,
            kinds: kinds.clone(),
            rope_dims: 64,
            rope_base: 1.0e7,
            rope_base_swa: 1.0e4,
            value_scale: 0.707,
            rms_eps: 1e-5,
            n_ctx_train: 1_048_576,
            n_vocab: 152_576,
            n_expert: 256,
            n_used: 8,
            expert_ff: 2048,
            dense_ff: Some(16384),
            weights_scale: 1.0,
            defaults: Vec::new(),
        };
        let mut tensors = Vec::new();
        let mut add =
            |name: String, layer: Option<usize>, role: Role, ty: GgmlType, dims: &[u64]| {
                tensors.push(ModelTensor {
                    name,
                    shard: 0,
                    layer,
                    role,
                    ty,
                    dims: dims.to_vec(),
                    file_bytes: 0,
                    gathered_rows: None,
                });
            };
        for (l, kind) in kinds.iter().enumerate() {
            let a = Some(l);
            let rows = match kind {
                Kind::Full => 64 * 192 + 4 * (192 + 128),
                Kind::Swa => 64 * 192 + 8 * (192 + 128),
            };
            add(
                names::attn_norm(l),
                a,
                Role::Attention,
                GgmlType::F32,
                &[4096],
            );
            add(
                names::attn_qkv(l),
                a,
                Role::Attention,
                GgmlType::Q8_0,
                &[4096, rows],
            );
            add(
                names::attn_output(l),
                a,
                Role::Attention,
                GgmlType::Q8_0,
                &[8192, 4096],
            );
            if *kind == Kind::Swa {
                add(
                    names::attn_sinks(l),
                    a,
                    Role::Attention,
                    GgmlType::F32,
                    &[64],
                );
            }
            add(names::ffn_norm(l), a, Role::FfnNorm, GgmlType::F32, &[4096]);
            if l == 0 {
                for n in [names::ffn_gate(l), names::ffn_up(l)] {
                    add(n, a, Role::DenseFfn, GgmlType::Q8_0, &[4096, 16384]);
                }
                add(
                    names::ffn_down(l),
                    a,
                    Role::DenseFfn,
                    GgmlType::Q8_0,
                    &[16384, 4096],
                );
            } else {
                add(
                    names::ffn_gate_inp(l),
                    a,
                    Role::Router,
                    GgmlType::F32,
                    &[4096, 256],
                );
                add(
                    names::exp_probs_b(l),
                    a,
                    Role::Router,
                    GgmlType::F32,
                    &[256],
                );
                for n in [names::ffn_gate_exps(l), names::ffn_up_exps(l)] {
                    add(
                        n,
                        a,
                        Role::RoutedExperts,
                        GgmlType::MXFP4,
                        &[4096, 2048, 256],
                    );
                }
                add(
                    names::ffn_down_exps(l),
                    a,
                    Role::RoutedExperts,
                    GgmlType::MXFP4,
                    &[2048, 4096, 256],
                );
            }
        }
        add(
            names::token_embd(),
            None,
            Role::TokenEmbedding,
            GgmlType::Q8_0,
            &[4096, 152_576],
        );
        add(
            names::output_norm(),
            None,
            Role::Head,
            GgmlType::F32,
            &[4096],
        );
        add(
            names::output(),
            None,
            Role::Head,
            GgmlType::Q8_0,
            &[4096, 152_576],
        );
        let model = ModelTensors {
            tensors,
            layers: 48,
            experts: 256,
            experts_used: 8,
        };
        let chat = models::ChatSpec {
            pre: "qwen2".to_string(),
            template: Some("{{ messages }}".to_string()),
            tools: spec::TOOLS,
            reasoning: Some(models::ReasoningFormat::ThinkSpan),
        };
        let spec = spec::spec_of(&hp, &model, chat).expect("the real shapes describe");
        (spec, model)
    }

    /// The real MiMo file is run whole by the mimo2 body: each need of its
    /// 48 layers has a row naming the program, each tensor type is one the
    /// body reads, and no stack is one the host cannot serve — so nothing is
    /// listed, and the plan's read does not refuse the file.
    // PIN(2026-10-08): the mimo2 coverage list is empty once the program runs
    // the file; before it, `mimo2_meta` pinned seven items (two flashes, the
    // rope, three mxfp4 rows and the program itself).
    #[test]
    fn the_real_mimo2_file_lists_nothing() {
        let (spec, model) = mimo2_real();
        let items = super::check(&spec, &model);
        assert!(items.is_empty(), "listed: {items:?}");
    }

    /// What the mimo2 body does not run is still an item, by name: a
    /// window layer with no sinks (the K192 window row folds them), a
    /// q4_K attention matrix (the body reads q8_0), a routed stack whose row
    /// width no fused qdot kernel serves, and a routed stack of a type with
    /// no host kernel at all — MiMo has no card expert, so no card rule
    /// passes a stack the host cannot.
    #[test]
    fn what_the_mimo2_body_does_not_run_is_an_item() {
        let (spec, model) = mimo2_real();
        let mut sinkless = spec.clone();
        let models::Mixer::Gqa(g) = &mut sinkless.layers[1].mixer else {
            panic!("a mimo2 layer that is not GQA");
        };
        g.sinks = false;
        let items = super::check(&sinkless, &model);
        assert_eq!(items.len(), 1, "{items:?}");
        assert!(items[0].feature.contains("window 128"), "{items:?}");

        let mut other = model.clone();
        for t in &mut other.tensors {
            if t.name == "blk.3.attn_qkv.weight" {
                t.ty = GgmlType::Q4_K;
            }
            if t.name == "blk.4.ffn_down_exps.weight" {
                t.dims[0] = 2000;
            }
            if t.name == "blk.5.ffn_up_exps.weight" {
                t.ty = GgmlType::IQ2_S;
            }
        }
        let items: Vec<String> = super::check(&spec, &other)
            .into_iter()
            .map(|u| u.feature)
            .collect();
        assert_eq!(
            items,
            [
                "q4_K attention matrices (the body reads q8_0)",
                "mxfp4 routed experts on a card",
                "iq2_s routed experts on a card",
            ],
            "{items:?}"
        );
    }
}
