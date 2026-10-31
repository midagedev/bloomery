//! The coverage check: what of a file no program in this tree runs, every
//! item listed at once ([`check`]). It joins the model's [`needs`] with a
//! table of what exists ([`rows`]: the common [`AVAILABLE`] and the rows a
//! family's own code keeps, which its registry row names; one row per
//! kernel instance or program step, each citing the constant or the
//! function that owns it; the routers' and the GQA flashes' rows ask the
//! shape table `models::shape`, which the router crates are held to), the
//! card formats of the tensors' types ([`CardFormat`], the format owner),
//! the program's per-tensor type pins, the pre-tokenizers the tokenizer runs
//! (`tokenizer::runs_pre_tokenizer`, the tokenizer's own table) and the chat
//! parsers.
//!
//! The kernels' own file-against-constant checks stay behind it as a second
//! line: a file this check passes cannot trip them.
//!
//! An architecture no program runs yet ([`program_of`] is `None`) is listed
//! against the whole tree: a need any program's row covers, and a tensor type
//! any program's pin reads, is not an item; the program itself is one.

use bloomery_placement::accepts::{
    BOLT_LEG, MUSESPARK_LEG, dense_by_name, dense_projected, embed_of_widened, head_of_widened,
    latent_leg_runs, leg_arms, qwen3moe_dense, resident_down_runs,
};
pub use bloomery_placement::accepts::{Program, program_of, qwen35_down, qwen35_gate_up};
use bloomery_placement::accepts::{prompt_leg_down, prompt_leg_gate_up};
pub use bloomery_placement::kernels::{DownEntry, GateUpEntry, SiteEntry};
use gguf::GgmlType;
use models::shape::{RouterBody, gqa_row, gqa_row_v, select_router};
use models::{
    Arch, Collapse, DeltaKind, Extra, GdnGate, HcMix, KHeadMap, KdaGate, LatentUp, ModelSpec, Need,
    RopeMode, needs,
};

use crate::placement::{CardFormat, ModelTensor, ModelTensors, Role, Unimplemented};

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

/// Everything the programs run that a [`Need`] names, beside the rows a
/// family's own code keeps ([`rows`]). A need no row covers is an item of
/// [`check`]; a row removed makes its needs items.
pub const AVAILABLE: &[Available] = &[
    Available {
        programs: &[Program::FlintBody],
        at: "gpu/src/attention_h128.rs Binding::enqueue and attention_h128_ring.rs",
        runs: |n| {
            matches!(
                n,
                Need::Gqa {
                    head: 128,
                    value: 128,
                    group: 8,
                    window: None | Some(513),
                    sinks: true,
                    scaled: false,
                }
            )
        },
    },
    Available {
        programs: &[Program::FlintBody],
        at: "gpu/src/attention_h128_ring.rs (head128 RMS with optional full NeoX)",
        runs: |n| {
            matches!(
                n,
                Need::QkRope {
                    qk_norm: true,
                    head: 128,
                    mode: RopeMode::None,
                    dims: 0,
                } | Need::QkRope {
                    qk_norm: true,
                    head: 128,
                    mode: RopeMode::Neox,
                    dims: 128,
                }
            )
        },
    },
    Available {
        programs: &[Program::FlintBody],
        at: "gpu/src/route_body.rs sigmoid_floor_256_6",
        runs: |n| routes(n, &[RouterBody::SigmoidFloor]),
    },
    Available {
        programs: &[Program::FlintBody],
        at: "gpu-deepseek41/src/shared_routed.rs (ungated shared expert)",
        runs: ungated_shared,
    },
    Available {
        programs: &[Program::FlintBody],
        at: "gpu-deepseek41/src/shared_routed.rs dense-leading gated blocks",
        runs: |n| matches!(n, Need::DenseFfn { ff } if ff.is_multiple_of(256)),
    },
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
        runs: ungated_shared,
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
        programs: &[Program::NemotronHBody, Program::NemotronHMoeBody],
        at: "gpu/src/mamba2.rs Mamba2Kernels::enqueue_step (STATE 128, CONV_TAPS 4)",
        runs: |n| {
            matches!(
                n,
                Need::Mamba2 {
                    d_state: 128,
                    d_conv: 4,
                    n_heads,
                    n_groups,
                    ..
                } if *n_groups > 0 && n_heads % n_groups == 0
            )
        },
    },
    Available {
        programs: &[Program::NemotronHBody],
        at: "gpu/src/elem.rs enqueue_relu_sqr",
        runs: |n| matches!(n, Need::ReluSqr { .. }),
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
                // The pack-5 and pack-16 rows (head 128, groups 5 and 16) are
                // launched by neither whole-card body, and the pack-7 row (group 7) by
                // Body35 alone; the head-128 pack-4 row is the head-128 attention
                // binding's. The programs that launch them name their own entries.
                } if value == head
                    && gqa_row(*head, *group).is_some_and(|r| {
                        r.pack != 5 && r.pack != 7 && r.pack != 16 && (r.head != 128 || r.pack != 4)
                    })
            )
        },
    },
    Available {
        programs: &[Program::NemotronHBody],
        at: "gpu/src/flash_gqa.rs gqa_flash_seg_g5",
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
                } if value == head && gqa_row(*head, *group).is_some_and(|r| r.pack == 5)
            )
        },
    },
    Available {
        programs: &[Program::Qwen35Body],
        at: "gpu/src/flash_gqa.rs GROUP_7 (gqa_flash_seg_mma_g7, gpu/src/flash_gqa_prefill.rs \
             gqa_prefill_flash_g7, and their q8 twins)",
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
                } if value == head && gqa_row(*head, *group).is_some_and(|r| r.pack == 7)
            )
        },
    },
    Available {
        programs: &[Program::NemotronHMoeBody],
        at: "gpu/src/flash_gqa.rs GROUP_16 (two group-8 passes a key head)",
        runs: gqa_group_16,
    },
    Available {
        programs: &[Program::NemotronHMoeBody],
        at: "models/src/shape.rs ROUTERS, the BiasedSigmoid body at 240, 304, 368, 432 and 496 \
             with the top 9 and at 128 with the top 6 (gpu/src/arch/nemotron_h_moe/router.rs)",
        runs: |n| {
            matches!(n, Need::Router(s) if NEMOTRON_ROUTERS.contains(&(s.experts, s.top_k)))
                && routes(n, &[RouterBody::BiasedSigmoid])
        },
    },
    Available {
        programs: &[Program::NemotronHMoeBody],
        at: "gpu/src/arch/nemotron_h_moe/moe.rs (ffn_latent_down, the experts' iq2_xxs_gemv_sel \
             at K = latent, ffn_latent_up)",
        runs: |n| matches!(n, Need::LatentExperts { latent } if latent.is_multiple_of(256)),
    },
    Available {
        programs: &[Program::NemotronHMoeBody],
        at: "gpu/src/arch/nemotron_h_moe/moe.rs (gpu/src/elem.rs enqueue_relu_sqr between the \
             up and down launches, routed and shared)",
        runs: |n| {
            matches!(
                n,
                Need::ExpertAct {
                    act: models::Act::ReluSqr,
                    ..
                }
            )
        },
    },
    Available {
        programs: &[Program::NemotronHMoeBody],
        at: "gpu/src/arch/nemotron_h_moe/moe.rs (the shared expert, one-expert iq2_xxs_gemv_sel \
             stacks or a q8_0 site)",
        runs: |n| {
            matches!(
                n,
                Need::Shared {
                    ff,
                    sigmoid_gate: false,
                } if ff.is_multiple_of(32)
            )
        },
    },
    Available {
        programs: &[Program::NemotronHMoeBody],
        at: "gpu/src/arch/nemotron_h_moe/moe.rs (each layer's router and stacks at its own \
             count and width)",
        runs: |n| matches!(n, Need::ExpertShapes { .. }),
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
        at: "gpu/src/rope_norm.rs rope_norm_append (NORM pairs over all 128 dims, no head norm)",
        runs: |n| {
            matches!(
                n,
                Need::QkRope {
                    qk_norm: false,
                    head: 128,
                    mode: RopeMode::NormTail,
                    dims: 128
                }
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
        at: "gpu/src/recur.rs RecStore (Body35's delta stores)",
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
                    kind: DeltaKind::Kda {
                        gate: KdaGate::Sigmoid { .. },
                        beta_scale,
                    },
                    d: 128,
                    conv: 4,
                    ..
                } if *beta_scale == 1.0
            )
        },
    },
    Available {
        programs: &[Program::Solar2Body],
        at: "gpu/src/linear/mixer_kda.rs (kda_conv_prep, kda_delta with the softplus decay and the β scale)",
        runs: |n| {
            matches!(
                n,
                Need::DeltaRule {
                    kind: DeltaKind::Kda {
                        gate: KdaGate::Softplus,
                        beta_scale,
                    },
                    d: 128,
                    conv: 4,
                    ..
                } if *beta_scale == 2.0
            )
        },
    },
    Available {
        programs: &[Program::Solar2Body],
        at: "gpu/src/arch/hybrid.rs gqa_layer (the NoPE layer over flash_gqa.rs HEAD, GROUP: head 128, group 8)",
        runs: |n| {
            matches!(
                n,
                Need::Gqa {
                    head: 128,
                    value: 128,
                    group: 8,
                    window: None,
                    sinks: false,
                    scaled: false,
                }
            )
        },
    },
    Available {
        programs: &[Program::Solar2Body],
        at: "gpu/src/arch/hybrid.rs GqaGate::Proj (the out gate of the layer's own projection)",
        runs: |n| matches!(n, Need::OutGateProj),
    },
    Available {
        programs: &[
            Program::Glm5nextBody,
            Program::Mimo2Body,
            Program::Solar2Body,
            Program::Raon1Body,
        ],
        at: "models/src/shape.rs ROUTERS, the BiasedSigmoid body at these bodies' own (experts, \
             kept) pairs (SIGMOID_ROUTERS; gpu-deepseek41 router: glm5next at 288 experts, mimo2 \
             at 256, solar2 at 320; gpu/src/route_body.rs sigmoid_128_6: raon1 and nemotron_h_moe \
             at 128)",
        runs: |n| {
            matches!(n, Need::Router(s) if SIGMOID_ROUTERS.contains(&(s.experts, s.top_k)))
                && routes(n, &[RouterBody::BiasedSigmoid])
        },
    },
    Available {
        programs: &[Program::Glm5nextBody, Program::Solar2Body],
        at: "gpu-glm5next/src/ffn.rs (the shared expert)",
        runs: ungated_shared,
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
        programs: &[Program::Glm5nextBody, Program::Solar2Body],
        at: "gpu-glm5next/src/kda.rs (the state and the conv ring, one lane)",
        runs: |n| matches!(n, Need::RecurrentState),
    },
    Available {
        programs: &[Program::Glm5nextBody, Program::Solar2Body],
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
    Available {
        programs: &[Program::GemmaEmbedding2Body],
        at: "gpu/src/flash_gqa_prefill.rs enqueue_256_p1_lower, enqueue_256_p2_lower (each row's first key from its reach)",
        runs: |n| matches!(n, Need::Bidirectional { head: 256, group, .. } if *group > 0),
    },
    Available {
        programs: &[Program::GemmaEmbedding2Body],
        at: "gpu/src/flash_gqa_prefill_512.rs enqueue_p4 (every key; four query heads a block)",
        runs: |n| {
            matches!(
                n,
                Need::Bidirectional {
                    head: 512,
                    group,
                    reach: None
                } if group.is_multiple_of(4)
            )
        },
    },
    Available {
        programs: &[Program::GemmaEmbedding2Body],
        at: "gpu/src/rope_neox.rs head_norm_neox_append_256_whole, head_norm_neox_append_512",
        runs: |n| {
            matches!(
                n,
                Need::QkRope {
                    qk_norm: true,
                    head: head @ (256 | 512),
                    mode: RopeMode::Neox,
                    dims,
                } if dims == head
            )
        },
    },
    Available {
        programs: &[Program::GemmaEmbedding2Body],
        at: "gpu/src/elem.rs enqueue_geglu between gpu/src/site.rs gemm launches",
        runs: |n| matches!(n, Need::GeGlu { ff: 1152 | 2048 }),
    },
    Available {
        programs: &[Program::GemmaEmbedding2Body],
        at: "gpu/src/elem.rs enqueue_rms_norm (a gain of ones)",
        runs: |n| matches!(n, Need::ValueNorm { .. }),
    },
    Available {
        programs: &[Program::GemmaEmbedding2Body],
        at: "gpu/src/elem.rs enqueue_rms_norm, enqueue_add (gemma-embedding2's pass)",
        runs: |n| matches!(n, Need::PostNorms),
    },
    Available {
        programs: &[Program::GemmaEmbedding2Body],
        at: "gpu/src/elem.rs enqueue_weighted_sum (the layer's scalar)",
        runs: |n| matches!(n, Need::LayerScale),
    },
    Available {
        programs: &[Program::GemmaEmbedding2Body],
        at: "gpu/src/gemm/f32tile.rs enqueue_f32_tile; gpu/src/elem.rs enqueue_rms_norm, \
             enqueue_geglu (gemma-embedding2's pass)",
        runs: |n| matches!(n, Need::PerLayerInput { width: 512 }),
    },
    Available {
        programs: &[Program::GemmaEmbedding2Body],
        at: "gpu-gemma-embedding2/src/pass.rs token_rows (the host's rows, scaled)",
        runs: |n| matches!(n, Need::EmbedScale),
    },
    Available {
        programs: &[Program::GemmaEmbedding2Body],
        at: "gpu/src/site.rs gemm (output); gpu-gemma-embedding2/src/pass.rs mean",
        runs: |n| matches!(n, Need::Pooled { proj } if matches!(proj.as_slice(), [768] | [3072, 768])),
    },
    Available {
        programs: &[Program::Raon1Body],
        at: "gpu/src/arch/raon1/rda.rs (the common KDA mixer, linear/mixer_kda.rs, with the write gate \
             linear/write_gate.rs: HEAD, CONV_TAPS, a bounded sigmoid decay)",
        runs: |n| {
            matches!(
                n,
                Need::DeltaRule {
                    kind: DeltaKind::Rda {
                        gate: KdaGate::Sigmoid { lower_bound },
                    },
                    k_heads,
                    v_heads,
                    d: 128,
                    conv: 4,
                } if k_heads == v_heads && *k_heads > 0 && *lower_bound < 0.0
            )
        },
    },
    Available {
        programs: &[Program::Raon1Body],
        at: "gpu/src/arch/raon1/mla.rs (the absorbed heads over gpu/src/latent.rs LATENT, \
             latent_extra_rms_append)",
        runs: |n| {
            matches!(
                n,
                Need::Latent {
                    latent: 512,
                    rope: 0,
                    up: LatentUp::Absorbed { qk: 192, v: 128 }
                }
            )
        },
    },
    Available {
        programs: &[Program::Raon1Body],
        at: "gpu/src/latent.rs latent_extra_rms_append (the extra values first, raw, at most 128)",
        runs: |n| matches!(n, Need::LatentPe { width } if (1..=128).contains(width)),
    },
    Available {
        programs: &[Program::Raon1Body],
        at: "gpu/src/arch/raon1/mla.rs (attn_gate, then gpu/src/linear/out_gate.rs)",
        runs: |n| matches!(n, Need::LatentOutGate),
    },
    Available {
        programs: &[Program::Raon1Body],
        at: "gpu/src/arch/raon1/ffn.rs Glu (the gate and up sites, gpu/src/elem.rs enqueue_situ, the \
             down site)",
        runs: |n| matches!(n, Need::Situ { ff } if ff.is_multiple_of(32)),
    },
    Available {
        programs: &[Program::Raon1Body],
        at: "gpu/src/arch/raon1/ffn.rs (the card leg's gate·up `_sel` under kquant::Act::Situ, \
             SITU_LINEAR_BETA)",
        runs: |n| {
            matches!(
                n,
                Need::ExpertAct {
                    shared: false,
                    act
                } if situ_runs(act)
            )
        },
    },
    Available {
        programs: &[Program::Raon1Body],
        at: "gpu/src/arch/raon1/ffn.rs Routed.shared (the shared expert's Glu block, added after \
             the routed sum)",
        runs: |n| {
            matches!(
                n,
                Need::Shared {
                    ff,
                    sigmoid_gate: false,
                } if ff.is_multiple_of(32)
            )
        },
    },
    Available {
        programs: &[Program::Raon1Body],
        at: "gpu/src/arch/raon1/ffn.rs Routed.shared (gpu/src/elem.rs enqueue_situ between the \
             shared expert's gate·up and down sites, SITU_LINEAR_BETA)",
        runs: |n| {
            matches!(
                n,
                Need::ExpertAct {
                    shared: true,
                    act
                } if situ_runs(act)
            )
        },
    },
    Available {
        programs: &[Program::Raon1Body],
        at: "gpu/src/arch/raon1/mhc.rs STREAMS (raon_hc_route, raon_hc_fold, raon_hc_post, raon_hc_mean)",
        runs: |n| {
            matches!(
                n,
                Need::HcChart {
                    streams: 4,
                    collapse: Collapse::Mean
                }
            )
        },
    },
    Available {
        programs: &[Program::Raon1Body],
        at: "gpu/src/arch/raon1/body.rs Mixer::Rda (the state, the stamp and the conv ring, one lane)",
        runs: |n| matches!(n, Need::RecurrentState),
    },
    Available {
        programs: &[Program::Raon1Body],
        at: "gpu/src/arch/raon1/body.rs Mixer (a delta or a latent mixer per layer's kind)",
        runs: |n| matches!(n, Need::MixedTrunk),
    },
    Available {
        programs: &[Program::MuseSparkBody],
        at: "models/src/shape.rs GQA, the head-64 row (gpu/src/flash_gqa.rs the _64 entries, \
             Muse Spark's: a layer's window of positions or none)",
        runs: |n| {
            matches!(
                n,
                Need::Gqa {
                    head: 64,
                    value: 64,
                    group: 8,
                    window: _,
                    sinks: false,
                    scaled: false,
                }
            ) && gqa_row(64, 8).is_some()
        },
    },
    Available {
        programs: &[Program::MuseSparkBody],
        at: "gpu/src/arch/musespark/body.rs (elem.rs rms_norm per head with the file's \
             [head] gains, the NORM-pair rope on the sliding layers over the sliding base, \
             none on the full ones)",
        runs: |n| {
            matches!(
                n,
                Need::QkRope {
                    qk_norm: true,
                    head: 64,
                    mode: RopeMode::NormTail | RopeMode::None,
                    dims: _,
                }
            )
        },
    },
    Available {
        programs: &[Program::MuseSparkBody],
        at: "gpu/src/arch/musespark/body.rs (σ of the gate's own projection through elem.rs \
             sigmoid_rows + mul_rows)",
        runs: |n| matches!(n, Need::OutGateProj),
    },
    Available {
        programs: &[Program::MuseSparkBody],
        at: "gpu/src/elem.rs enqueue_rms_norm (a gain of ones, k the head 64)",
        runs: |n| matches!(n, Need::OutNorm { head: 64 }),
    },
    Available {
        programs: &[Program::MuseSparkBody],
        at: "models/src/shape.rs ROUTERS, the BiasedSigmoid body at the muse-spark file's own \
             (experts, kept) pair (gpu/src/route_body.rs sigmoid_256_8; the fixture's 16/8)",
        runs: |n| {
            matches!(n, Need::Router(s) if MUSE_ROUTERS.contains(&(s.experts, s.top_k)))
                && routes(n, &[RouterBody::BiasedSigmoid])
        },
    },
    Available {
        programs: &[Program::MuseSparkBody],
        at: "gpu/src/arch/musespark/body.rs (ffn_latent_down, the experts' \
             iq2_xxs_gate_up_sel + iq2_xxs_gemv_sel at K = latent, ffn_latent_up)",
        runs: |n| matches!(n, Need::LatentExperts { latent } if latent.is_multiple_of(256)),
    },
    Available {
        programs: &[Program::MuseSparkBody],
        at: "gpu/src/elem.rs enqueue_rms_norm on the latent row (ffn_pre_exp_norm)",
        runs: |n| matches!(n, Need::PreExpNorm),
    },
    Available {
        programs: &[Program::MuseSparkBody],
        at: "gpu/src/elem.rs enqueue_gain_rms_norm on the experts' aggregate before the \
             router weights apply (ffn_post_exp_norm)",
        runs: |n| matches!(n, Need::PostExpNorm { .. }),
    },
    Available {
        programs: &[Program::MuseSparkBody],
        at: "gpu/src/elem.rs enqueue_mix_rows (α·stream + β·branch, each sublayer's output \
             normed first by its own row)",
        runs: |n| matches!(n, Need::AlphaBeta { .. }),
    },
    Available {
        programs: &[Program::MuseSparkBody],
        at: "gpu/src/head.rs enqueue_with_tail (elem.rs scale_rows + softcap_rows over the \
             Q6_K gemv's logits)",
        runs: |n| matches!(n, Need::LogitEpilogue { .. }),
    },
    Available {
        programs: &[Program::MuseSparkBody],
        at: "gpu/src/arch/musespark/body.rs (each layer's router and stacks at its own \
             count and width)",
        runs: |n| matches!(n, Need::ExpertShapes { .. }),
    },
];

/// The `value_beta` of the SiTU rule the device runs (`gpu/src/kquant/act.rs`
/// `SITU_LINEAR_BETA`): the one value the references' files carry.
const SITU_VALUE_BETA: f32 = 25.0;

/// Whether the device runs `act`: SiTU at a finite positive gate beta and the
/// one value beta ([`SITU_VALUE_BETA`]), the pair `Act::code` of the card's
/// gate·up kernels takes.
fn situ_runs(act: &models::Act) -> bool {
    matches!(
        act,
        models::Act::Situ { gate_beta, value_beta }
            if gate_beta.is_finite() && *gate_beta > 0.0 && value_beta.to_bits() == SITU_VALUE_BETA.to_bits()
    )
}

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
        RopeMode::NormTail | RopeMode::None => false,
    }
}

/// Whether the router `n` names has an instance among `bodies`' rows: the
/// row [`select_router`] picks, so a count or a pick the launchers refuse
/// is an item here too.
pub(crate) fn routes(n: &Need, bodies: &[RouterBody]) -> bool {
    matches!(n, Need::Router(s) if select_router(*s).is_ok_and(|r| bodies.contains(&r.body)))
}

/// Whether `n` is a shared expert without a sigmoid gate (`gpu-deepseek41/src/shared_routed.rs`).
pub(crate) fn ungated_shared(n: &Need) -> bool {
    matches!(
        n,
        Need::Shared {
            sigmoid_gate: false,
            ..
        }
    )
}

/// Whether `n` is a flash of the group-16 row (`gpu/src/flash_gqa.rs` `GROUP_16`, two group-8
/// passes a key head): the value head the key head's, no window, sinks or value scale.
pub(crate) fn gqa_group_16(n: &Need) -> bool {
    matches!(
        n,
        Need::Gqa {
            head,
            value,
            group,
            window: None,
            sinks: false,
            scaled: false,
        } if value == head && gqa_row(*head, *group).is_some_and(|r| r.pack == 16)
    )
}

/// A tensor family whose type a program pins, and the types it reads.
struct TypePin {
    /// The programs whose loads read the family at these types.
    programs: &'static [Program],
    role: Role,
    /// Only the quantized tensors of two or more dimensions (the matrices a
    /// gemv reads; an F32 parameter table is its layer's mixer's).
    matrices: bool,
    /// Only the tensors whose name this takes; `None`: every one of the role.
    names: Option<fn(&str) -> bool>,
    what: &'static str,
    reads: &'static [GgmlType],
}

/// The types a gated block's matrices read: the q8_0 pair, or K-quants
/// (`gpu-deepseek41/src/block.rs` `BLOCK_TYPES`, each a launch of
/// `gpu/src/site.rs` or the fused K-quant gate·up).
// PIN(2026-10-11): widened from `[Q8_0]` with the gated block's type dispatch,
// for the mimo2 dense block and the glm5next dense block and shared expert;
// derivation: each type is read by `site::kgemv` or the fused K-quant gate·up
// in `enqueue_block`, at the decode sites.
const GATED_BLOCK: &[GgmlType] = &[
    GgmlType::Q8_0,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
];

/// The types a nemotron_h body site reads (`gpu/src/arch/nemotron_h/step.rs`
/// `SITE_TYPES`, each a launch of `gpu/src/site.rs`).
const NEMOTRON_SITES: &[GgmlType] = &[
    GgmlType::Q5_0,
    GgmlType::Q8_0,
    GgmlType::Q4_K,
    GgmlType::Q6_K,
];

/// The per-tensor type pins of the programs, as lists (the qwen3moe body's
/// dense FFN; `Body35`'s
/// site types, its `PROJ` and `DENSE_FFN` in
/// the chain's reader, its `EMBED` and `HEAD_TY` in
/// `gpu/src/arch/qwen3moe/body35.rs`, each a launch of `gpu/src/site.rs`; the
/// pins read from the card kernel table's rows are [`ROW_PINS`]; the V4.1
/// chain's attention gemvs, whose head-split sites in `chain/attn.rs` take
/// q3_K, q5_K or q8_0, the head q6_K or q5_K, and the hyper-connection fn `hc.rs` reads as
/// q3_K words or f32; the glm5next body's mixer sites by name (`site::ByName`), its absorbed pair q8_0
/// and its hyper-connection fn `hc_pre_q8_0` or f32, the common
/// head's q8_0, q6_K and q4_K arms; the gemma-embedding2 encoder's GEMMs,
/// each `site::gemm` on the 32-value path).
const TYPE_PINS: &[TypePin] = &[
    TypePin {
        programs: &[Program::Qwen3moeBody],
        role: Role::DenseFfn,
        matrices: true,
        names: Some(|n| !n.ends_with(".ffn_down.weight")),
        what: "dense FFN gate and up (the body reads q4_K)",
        reads: &[GgmlType::Q4_K],
    },
    TypePin {
        programs: &[Program::Qwen3moeBody],
        role: Role::DenseFfn,
        matrices: true,
        names: Some(|n| n.ends_with(".ffn_down.weight")),
        what: "dense FFN down (the body reads q4_K and q6_K)",
        reads: &[GgmlType::Q4_K, GgmlType::Q6_K],
    },
    TypePin {
        programs: &[Program::Qwen35Body],
        role: Role::Attention,
        matrices: true,
        names: None,
        what: "attention and delta-rule matrices (the body reads q3_K, q4_K, q5_K, q6_K and q8_0)",
        reads: Q35_PROJ,
    },
    TypePin {
        programs: &[Program::Qwen35Body],
        role: Role::DenseFfn,
        matrices: true,
        names: None,
        what: "dense FFN (the body reads q3_K, q4_K, q5_K, q6_K, q8_0 and nvfp4)",
        reads: Q35_DENSE_FFN,
    },
    TypePin {
        programs: &[Program::Qwen35Body],
        role: Role::Head,
        matrices: true,
        names: None,
        what: "output head (the head reads q6_K, q4_K, q8_0 and nvfp4)",
        reads: &[
            GgmlType::Q6_K,
            GgmlType::Q4_K,
            GgmlType::Q8_0,
            GgmlType::NVFP4,
        ],
    },
    TypePin {
        programs: &[Program::Qwen35Body],
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
        programs: &[Program::Deepseek41Chain],
        role: Role::Attention,
        matrices: true,
        names: Some(|n| !super::deepseek41::roles::is_side_matrix(n)),
        what: "attention matrices (the chain reads q3_K, q5_K and q8_0)",
        reads: &[GgmlType::Q3_K, GgmlType::Q5_K, GgmlType::Q8_0],
    },
    TypePin {
        programs: &[Program::Deepseek41Chain],
        role: Role::Attention,
        matrices: true,
        names: Some(super::deepseek41::roles::is_side_matrix),
        what: "compressor and indexer matrices (the chain reads q3_K, q5_K, q8_0 and bf16)",
        reads: &[
            GgmlType::Q3_K,
            GgmlType::Q5_K,
            GgmlType::Q8_0,
            GgmlType::BF16,
        ],
    },
    TypePin {
        programs: &[Program::Deepseek41Chain],
        role: Role::Head,
        matrices: true,
        names: None,
        what: "output head (the head reads q6_K and q5_K)",
        reads: &[GgmlType::Q6_K, GgmlType::Q5_K],
    },
    TypePin {
        programs: &[Program::Deepseek41Chain],
        role: Role::HyperConnection,
        matrices: true,
        names: None,
        what: "hyper-connection fn (the chain reads q3_K and f32)",
        reads: &[GgmlType::Q3_K],
    },
    // The absorbed pair is the one mixer site with no common dense entry: a
    // per-head gemv over the weight's heads, which only q8_0 planes have.
    TypePin {
        programs: &[Program::Glm5nextBody],
        role: Role::Attention,
        matrices: true,
        names: Some(is_absorbed_pair),
        what: "absorbed key and value heads (the per-head gemv reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    // The selector's query and head weights run on its branch stream through
    // its own row-major gemv (`mla::select`), not `site::ByName`.
    TypePin {
        programs: &[Program::Glm5nextBody],
        role: Role::Attention,
        matrices: true,
        names: Some(is_selector_site),
        what: "selector query and head weights (its gemv reads q8_0, f32 or bf16)",
        reads: &[GgmlType::Q8_0, GgmlType::BF16],
    },
    // The selector's pool position table is read as f32 values: a BF16 one
    // loads widened.
    TypePin {
        programs: &[Program::Glm5nextBody],
        role: Role::Attention,
        matrices: true,
        names: Some(|n| n.ends_with(".indexer_compressor_ape.weight")),
        what: "selector pool positions (read as f32 values)",
        reads: &[GgmlType::BF16],
    },
    // PIN(2026-10-26): widened from `[Q8_0]` with the f32 hyper-connection fn
    // (`HcFn::F32`, `ds41_hc_pre_f32`): a BF16 fn loads widened to f32 values,
    // which that launch reads beside the q8_0 planes `hc_pre_q8_0` reads.
    TypePin {
        programs: &[Program::Glm5nextBody],
        role: Role::HyperConnection,
        matrices: true,
        names: None,
        what: "hyper-connection fn (hc_pre_q8_0 reads q8_0, ds41_hc_pre_f32 f32 values: bf16 widened)",
        reads: &[GgmlType::Q8_0, GgmlType::BF16],
    },
    TypePin {
        programs: &[Program::Glm5nextBody],
        role: Role::DenseFfn,
        matrices: true,
        names: None,
        what: "dense block (the body reads q8_0, q4_K, q5_K and q6_K)",
        reads: GATED_BLOCK,
    },
    TypePin {
        programs: &[Program::Glm5nextBody],
        role: Role::SharedExpert,
        matrices: true,
        names: None,
        what: "shared expert (the body reads q8_0, q4_K, q5_K and q6_K)",
        reads: GATED_BLOCK,
    },
    TypePin {
        programs: &[Program::Glm5nextBody],
        role: Role::Head,
        matrices: true,
        names: None,
        what: "output head (the head reads q8_0 planes, q6_K or q4_K word planes)",
        // The common head's own set (`head_out_w` in `gpu/src/head.rs`).
        reads: &[GgmlType::Q8_0, GgmlType::Q6_K, GgmlType::Q4_K],
    },
    TypePin {
        programs: &[Program::Qwen38Body],
        role: Role::HyperConnection,
        matrices: true,
        names: None,
        what: "hyper-connection down and up (the body reads q8_0 and bf16)",
        reads: &[GgmlType::Q8_0, GgmlType::BF16],
    },
    TypePin {
        programs: &[Program::Mimo2Body],
        role: Role::Attention,
        matrices: true,
        names: None,
        what: "attention matrices (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        programs: &[Program::Mimo2Body],
        role: Role::DenseFfn,
        matrices: true,
        names: None,
        what: "dense block (the body reads q8_0, q4_K, q5_K and q6_K)",
        reads: GATED_BLOCK,
    },
    TypePin {
        programs: &[Program::Mimo2Body],
        role: Role::Head,
        matrices: true,
        names: None,
        what: "output head (the head reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        programs: &[Program::NemotronHBody],
        role: Role::TokenEmbedding,
        matrices: false,
        names: None,
        what: "token embedding (the card reads q5_0 rows)",
        reads: &[GgmlType::Q5_0],
    },
    TypePin {
        programs: &[Program::NemotronHBody, Program::NemotronHMoeBody],
        role: Role::Head,
        matrices: false,
        names: Some(|n| n == "output.weight"),
        what: "output head (the head reads q6_K, q4_K, q8_0 and nvfp4)",
        reads: &[
            GgmlType::Q6_K,
            GgmlType::Q4_K,
            GgmlType::Q8_0,
            GgmlType::NVFP4,
        ],
    },
    TypePin {
        programs: &[Program::NemotronHBody],
        role: Role::Attention,
        matrices: false,
        names: Some(|n| {
            use super::nemotron_h::names as nm;
            [
                nm::SSM_IN,
                nm::SSM_OUT,
                nm::ATTN_Q,
                nm::ATTN_K,
                nm::ATTN_V,
                nm::ATTN_OUTPUT,
            ]
            .into_iter()
            .any(|stem| n.ends_with(stem))
        }),
        what: "Mamba-2 and attention projections (the body reads q5_0, q8_0, q4_K and q6_K)",
        reads: NEMOTRON_SITES,
    },
    TypePin {
        programs: &[Program::NemotronHBody],
        role: Role::DenseFfn,
        matrices: false,
        names: None,
        what: "dense FFN (the body reads q5_0, q8_0, q4_K and q6_K)",
        reads: NEMOTRON_SITES,
    },
    TypePin {
        programs: &[Program::NemotronHBody, Program::NemotronHMoeBody],
        role: Role::Attention,
        matrices: false,
        names: Some(|n| {
            use super::nemotron_h_moe::names as nm;
            [
                nm::ATTN_NORM,
                nm::SSM_A,
                nm::SSM_D,
                nm::SSM_DT_BIAS,
                nm::SSM_CONV,
                nm::SSM_CONV_BIAS,
                nm::SSM_NORM,
                nm::SSM_IN_SCALE,
                nm::SSM_OUT_SCALE,
            ]
            .into_iter()
            .any(|stem| n.ends_with(stem))
        }),
        what: "Mamba-2 parameters, projection scales and normalization gains (the body reads f32 \
               and bf16)",
        reads: &[GgmlType::F32, GgmlType::BF16],
    },
    TypePin {
        programs: &[Program::NemotronHBody, Program::NemotronHMoeBody],
        role: Role::FfnNorm,
        matrices: false,
        names: None,
        what: "normalization gains (the body reads f32 and bf16)",
        reads: &[GgmlType::F32, GgmlType::BF16],
    },
    TypePin {
        programs: &[Program::NemotronHBody, Program::NemotronHMoeBody],
        role: Role::Head,
        matrices: false,
        names: Some(|n| n == "output_norm.weight"),
        what: "output normalization gain (the head reads f32 and bf16)",
        reads: &[GgmlType::F32, GgmlType::BF16],
    },
    TypePin {
        programs: &[Program::NemotronHMoeBody],
        role: Role::TokenEmbedding,
        matrices: false,
        names: None,
        what: "token embedding (the card reads q8_0 rows)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        programs: &[Program::NemotronHMoeBody],
        role: Role::Attention,
        matrices: false,
        names: Some(|n| {
            use super::nemotron_h_moe::names as nm;
            [
                nm::SSM_IN,
                nm::SSM_OUT,
                nm::ATTN_Q,
                nm::ATTN_K,
                nm::ATTN_V,
                nm::ATTN_OUTPUT,
            ]
            .into_iter()
            .any(|stem| n.ends_with(stem))
        }),
        what: "Mamba-2 and attention projections (the body reads nvfp4 and q8_0)",
        reads: &[GgmlType::NVFP4, GgmlType::Q8_0],
    },
    TypePin {
        programs: &[Program::NemotronHMoeBody],
        role: Role::Router,
        matrices: false,
        names: Some(|n| {
            use super::nemotron_h_moe::names as nm;
            n.ends_with(nm::FFN_GATE_INP) || n.ends_with(nm::EXP_PROBS_B)
        }),
        what: "router and selection bias (the body reads f32 and bf16)",
        reads: &[GgmlType::F32, GgmlType::BF16],
    },
    TypePin {
        programs: &[Program::NemotronHMoeBody],
        role: Role::Router,
        matrices: false,
        names: Some(|n| {
            use super::nemotron_h_moe::names as nm;
            n.ends_with(nm::FFN_LATENT_DOWN) || n.ends_with(nm::FFN_LATENT_UP)
        }),
        what: "latent projections (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        programs: &[Program::GemmaEmbedding2Body],
        role: Role::Attention,
        matrices: true,
        names: None,
        what: "attention matrices (the encoder reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        programs: &[Program::GemmaEmbedding2Body],
        role: Role::DenseFfn,
        matrices: true,
        names: None,
        what: "feed-forward matrices (the encoder reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        programs: &[Program::GemmaEmbedding2Body],
        role: Role::EngramDense,
        matrices: true,
        names: Some(|n| !per_layer_model_proj(n)),
        what: "per-layer input gate and projection (the encoder reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        programs: &[Program::GemmaEmbedding2Body],
        role: Role::Head,
        matrices: true,
        names: None,
        what: "output projection (the encoder reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        programs: &[Program::Raon1Body],
        role: Role::Attention,
        matrices: true,
        names: None,
        what: "attention and delta-rule matrices (the body reads q3_K, q4_K, q5_K, q6_K and q8_0 \
               sites and the f16 rows it widens)",
        reads: RAON1_PROJ,
    },
    TypePin {
        programs: &[Program::Raon1Body],
        role: Role::DenseFfn,
        matrices: true,
        names: None,
        what: "dense FFN (the body reads q4_K, q5_K, q6_K, q8_0 and q5_0)",
        reads: RAON1_FFN,
    },
    TypePin {
        programs: &[Program::Raon1Body],
        role: Role::SharedExpert,
        matrices: true,
        names: None,
        what: "shared expert (the body reads q4_K, q5_K, q6_K, q8_0 and q5_0)",
        reads: RAON1_FFN,
    },
    TypePin {
        programs: &[Program::Raon1Body],
        role: Role::HyperConnection,
        matrices: true,
        names: None,
        what: "hyper-connection fn (the body reads the f16 rows it widens to f32)",
        reads: &[GgmlType::F16],
    },
    TypePin {
        programs: &[Program::MuseSparkBody],
        role: Role::Attention,
        matrices: true,
        names: None,
        what: "attention projections (the body reads q6_K and q8_0 sites)",
        reads: &[GgmlType::Q6_K, GgmlType::Q8_0],
    },
    TypePin {
        programs: &[Program::MuseSparkBody],
        role: Role::Router,
        matrices: false,
        names: Some(|n| {
            use super::musespark::names as nm;
            n.ends_with(nm::FFN_GATE_INP) || n.ends_with(nm::EXP_PROBS_B)
        }),
        what: "router and selection bias (the body reads f32 and bf16)",
        reads: &[GgmlType::F32, GgmlType::BF16],
    },
    TypePin {
        programs: &[Program::MuseSparkBody],
        role: Role::Router,
        matrices: false,
        names: Some(|n| {
            use super::musespark::names as nm;
            n.ends_with(nm::FFN_LATENT_DOWN) || n.ends_with(nm::FFN_LATENT_UP)
        }),
        what: "latent projections (the body reads q8_0)",
        reads: &[GgmlType::Q8_0],
    },
    TypePin {
        programs: &[Program::MuseSparkBody],
        role: Role::FfnNorm,
        matrices: false,
        names: None,
        what: "normalization gains and residual mixes (the body reads f32 and bf16)",
        reads: &[GgmlType::F32, GgmlType::BF16],
    },
    TypePin {
        programs: &[Program::MuseSparkBody],
        role: Role::Head,
        matrices: false,
        names: Some(|n| n == "output.weight"),
        what: "output head (the head reads q6_K)",
        reads: &[GgmlType::Q6_K],
    },
    TypePin {
        programs: &[Program::MuseSparkBody],
        role: Role::Head,
        matrices: false,
        names: Some(|n| n == "output_norm.weight"),
        what: "output normalization gain (the head reads f32 and bf16)",
        reads: &[GgmlType::F32, GgmlType::BF16],
    },
];

/// The types a raon1 attention or delta-rule matrix reads: the sites'
/// (`gpu/src/arch/raon1/{rda,mla}.rs` `DENSE`) and the f16 rows `Weights::load`
/// widens (the delta layers' `ssm_beta`, `ssm_rda`, `ssm_f_a` and `ssm_f_b`).
const RAON1_PROJ: &[GgmlType] = &[
    GgmlType::Q3_K,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q8_0,
    GgmlType::F16,
];

/// The types a raon1 feed-forward site reads (`gpu/src/arch/raon1/body.rs`
/// `FFN_TYPES`, each a launch of `gpu/src/site.rs`).
const RAON1_FFN: &[GgmlType] = &[
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q8_0,
    GgmlType::Q5_0,
];

/// A tensor family whose type a program pins by the card kernel table's
/// rows: the types it reads are those `row` takes, so a type the table gains
/// is admitted with no edit here.
struct RowPin {
    program: Program,
    role: Role,
    /// As [`TypePin::matrices`].
    matrices: bool,
    /// As [`TypePin::names`].
    names: Option<fn(&str) -> bool>,
    what: &'static str,
    row: fn(GgmlType) -> bool,
}

/// The per-tensor pins whose types are the card kernel table's: the qwen3moe
/// body's attention sites ([`qwen3moe_dense`]: the table's dense row kept to
/// the forms that body's arena carries, the fused Q4_K ones among them), head
/// ([`head_of_widened`]: `head.rs`) and token embedding ([`embed_of_widened`]:
/// `elem.rs`), each the table's row but BF16, the qwen4exp body's head and
/// token embedding (`kernels::head`, `kernels::embed`: the table's rows,
/// BF16's among them, which that load keeps in its words), and routed stacks
/// (the qwen3moe body's and `Body35`'s [`prompt_leg_gate_up`], [`prompt_leg_down`]: the
/// FFN dispatch and the common card leg), and `Body35`'s shared expert
/// ([`qwen35_shared`]),
/// and the glm5next and mimo2 bodies' token embedding (the host's rows,
/// `GgmlType::has_dequant`), and the gemma-embedding2 encoder's token table
/// and per-layer input projection (the host's rows), each `what` naming the
/// types its row takes or the table row it follows.
const ROW_PINS: &[RowPin] = &[
    RowPin {
        program: Program::FlintBody,
        role: Role::Attention,
        matrices: true,
        names: None,
        what: "head128 attention projections (common dense sites)",
        row: h128_projection,
    },
    RowPin {
        program: Program::FlintBody,
        role: Role::DenseFfn,
        matrices: true,
        names: None,
        what: "dense-leading gated blocks",
        row: gated_block_site,
    },
    RowPin {
        program: Program::FlintBody,
        role: Role::SharedExpert,
        matrices: true,
        names: None,
        what: "shared gated blocks",
        row: gated_block_site,
    },
    RowPin {
        program: Program::FlintBody,
        role: Role::TokenEmbedding,
        matrices: true,
        names: None,
        what: "common token embedding",
        row: |ty| embed_of_widened(ty).is_some(),
    },
    RowPin {
        program: Program::FlintBody,
        role: Role::Head,
        matrices: true,
        names: None,
        what: "common output head",
        row: |ty| head_of_widened(ty).is_some(),
    },
    // PIN(2026-10-22): Qwen38 dense pins admit MXFP8/FP8_B128 and Q8_0 indexers via BY_NAME; the old MXFP8 refusal fails the new coverage test (PR FAIL-first).
    // PIN(2026-10-26): the qwen38 body's dense sites (the delta rule's and the selecting attention's, the shared
    // expert's, the PLE site's) read the common dense entries by name (`site::ByName`) in every type but Q5_0, so the
    // K-quants and NVFP4 joined the f32-row types; each type is a launch of `site.rs`, which
    // `gate_qwen4exp_e2e` runs on the fixture variant `q38dense`. The shared expert's down has K = 640, not whole
    // super-blocks of 256, so no q8_1 family reads it: its own pin keeps the f32-row types.
    RowPin {
        program: Program::Qwen38Body,
        role: Role::Attention,
        matrices: true,
        names: None,
        what: "attention matrices (the types dense_projected names)",
        row: |ty| dense_projected(ty).is_some(),
    },
    RowPin {
        program: Program::Qwen38Body,
        role: Role::SharedExpert,
        matrices: true,
        names: None,
        what: "shared expert (the types dense_projected names)",
        row: |ty| dense_projected(ty).is_some(),
    },
    RowPin {
        program: Program::Qwen38Body,
        role: Role::SharedExpert,
        matrices: true,
        names: Some(|n| n.ends_with(".ffn_down_shexp.weight")),
        what: "shared expert down (K = 640 is not whole q8_1 super-blocks: the types dense_by_name names)",
        row: |ty| dense_by_name(ty).is_some(),
    },
    // PIN(2026-10-26): the latent body's mixer dense sites read the common dense
    // entries by name (`site::ByName`) in every type but Q5_0, where they read
    // q8_0 alone; each type is a launch of `site.rs`, which the family's e2e
    // gate runs on its mixer fixture variants.
    RowPin {
        program: Program::Glm5nextBody,
        role: Role::Attention,
        matrices: true,
        names: Some(|n| {
            !is_absorbed_pair(n)
                && !is_selector_site(n)
                && !n.ends_with(".indexer_compressor_ape.weight")
        }),
        what: "mixer matrices (the types dense_projected names)",
        row: |ty| dense_projected(ty).is_some(),
    },
    // The qwen4exp load keeps a bf16 head and embedding in their words
    // (`ModelTensors::bf16_words`), which the head and the embedding rows
    // read, so these two rows keep BF16 where the qwen3moe body's
    // (`head_of_widened`, `embed_of_widened`) do not.
    RowPin {
        program: Program::Qwen38Body,
        role: Role::Head,
        matrices: true,
        names: None,
        what: "output head (the types the card kernel table's head rows name)",
        row: |ty| crate::placement::kernels::head(ty).is_some(),
    },
    RowPin {
        program: Program::Qwen38Body,
        role: Role::TokenEmbedding,
        matrices: true,
        names: None,
        what: "token embedding (the types the card kernel table's embedding rows name)",
        row: |ty| crate::placement::kernels::embed(ty).is_some(),
    },
    RowPin {
        program: Program::Qwen3moeBody,
        role: Role::Attention,
        matrices: true,
        names: None,
        what: "attention matrices (the body's dense sites: the types qwen3moe_dense names)",
        row: |ty| qwen3moe_dense(ty).is_some(),
    },
    RowPin {
        program: Program::Qwen3moeBody,
        role: Role::Head,
        matrices: true,
        names: None,
        what: "output head (the types head_of_widened names)",
        row: |ty| head_of_widened(ty).is_some(),
    },
    RowPin {
        program: Program::Qwen3moeBody,
        role: Role::TokenEmbedding,
        matrices: true,
        names: None,
        what: "token embedding (the card's rows: the types embed_of_widened names)",
        row: |ty| embed_of_widened(ty).is_some(),
    },
    RowPin {
        program: Program::Glm5nextBody,
        role: Role::TokenEmbedding,
        matrices: true,
        names: None,
        what: "token embedding (the host's rows: the types dequant_row decodes)",
        row: GgmlType::has_dequant,
    },
    RowPin {
        program: Program::Mimo2Body,
        role: Role::TokenEmbedding,
        matrices: true,
        names: None,
        what: "token embedding (the host's rows: the types dequant_row decodes)",
        row: GgmlType::has_dequant,
    },
    RowPin {
        program: Program::Solar2Body,
        role: Role::TokenEmbedding,
        matrices: true,
        names: None,
        what: "token embedding (the host's rows: the types dequant_row decodes)",
        row: GgmlType::has_dequant,
    },
    RowPin {
        program: Program::GemmaEmbedding2Body,
        role: Role::TokenEmbedding,
        matrices: true,
        names: None,
        what: "token embedding (the host's rows: the types dequant_row decodes)",
        row: GgmlType::has_dequant,
    },
    RowPin {
        program: Program::GemmaEmbedding2Body,
        role: Role::EngramDense,
        matrices: true,
        names: Some(per_layer_model_proj),
        what: "per-layer input projection (the host's rows: the types dequant_row decodes)",
        row: GgmlType::has_dequant,
    },
    RowPin {
        program: Program::Qwen3moeBody,
        role: Role::RoutedExperts,
        matrices: true,
        names: Some(routed_gate_up),
        what: "routed experts gate and up (the body reads q4_K, q5_K, q6_K, iq2_xxs, iq2_xs, iq3_xxs, iq1_s, iq3_s, iq2_s, iq4_xs, iq1_m and q8_0)",
        row: prompt_leg_gate_up,
    },
    RowPin {
        program: Program::Qwen3moeBody,
        role: Role::RoutedExperts,
        matrices: true,
        names: Some(routed_down),
        what: "routed experts down (the body reads q4_K, q5_K, q6_K, iq2_xxs, iq2_xs, iq3_xxs, iq1_s, iq3_s, iq2_s, iq4_xs, iq1_m, q5_1, q8_0 and iq4_nl)",
        row: prompt_leg_down,
    },
    RowPin {
        program: Program::Qwen35Body,
        role: Role::SharedExpert,
        matrices: true,
        names: None,
        what: "shared expert (the body reads q3_K, q4_K, q5_K, q6_K and q8_0)",
        row: |ty| qwen35_shared(ty).is_some(),
    },
    RowPin {
        program: Program::Qwen35Body,
        role: Role::RoutedExperts,
        matrices: true,
        names: Some(routed_gate_up),
        what: "routed experts gate and up (the body reads q4_K, q5_K, q6_K, iq2_xxs, iq2_xs, iq3_xxs, iq1_s, iq3_s, iq2_s, iq4_xs, iq1_m and q8_0)",
        row: prompt_leg_gate_up,
    },
    RowPin {
        program: Program::Qwen35Body,
        role: Role::RoutedExperts,
        matrices: true,
        names: Some(routed_down),
        what: "routed experts down (the body reads q4_K, q5_K, q6_K, iq2_xxs, iq2_xs, iq3_xxs, iq1_s, iq3_s, iq2_s, iq4_xs, iq1_m, q5_1, q8_0 and iq4_nl)",
        row: prompt_leg_down,
    },
    RowPin {
        program: Program::Raon1Body,
        role: Role::Head,
        matrices: true,
        names: None,
        what: "output head (the types head_of_widened names)",
        row: |ty| head_of_widened(ty).is_some(),
    },
    RowPin {
        program: Program::Raon1Body,
        role: Role::TokenEmbedding,
        matrices: true,
        names: None,
        what: "token embedding (the card's rows: the types embed_of_widened names)",
        row: |ty| embed_of_widened(ty).is_some(),
    },
    RowPin {
        program: Program::Raon1Body,
        role: Role::RoutedExperts,
        matrices: true,
        names: Some(routed_gate_up),
        what: "routed experts gate and up (the card leg's types: kernels::card_gate_up_runs)",
        row: crate::placement::kernels::card_gate_up_runs,
    },
    RowPin {
        program: Program::Raon1Body,
        role: Role::RoutedExperts,
        matrices: true,
        names: Some(routed_down),
        what: "routed experts down (the card leg's types and q5_0: accepts::resident_down_runs)",
        row: resident_down_runs,
    },
    RowPin {
        program: Program::NemotronHMoeBody,
        role: Role::SharedExpert,
        matrices: false,
        names: None,
        what: "shared expert (the card's iq2_xxs_gemv_sel reads iq2_xxs, its sites read q8_0 \
               and nvfp4)",
        row: |ty| ty == BOLT_LEG || ty == GgmlType::Q8_0 || ty == GgmlType::NVFP4,
    },
    RowPin {
        program: Program::MuseSparkBody,
        role: Role::TokenEmbedding,
        matrices: true,
        names: None,
        what: "token embedding (the card's rows: the types embed_of_widened names)",
        row: |ty| embed_of_widened(ty).is_some(),
    },
];

fn h128_projection(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::IQ4_XS
            | GgmlType::Q5_0
            | GgmlType::Q8_0
            | GgmlType::F32
            | GgmlType::MXFP8
            | GgmlType::FP8_B128
            | GgmlType::BF16
    )
}

fn gated_block_site(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q8_0
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::IQ4_XS
            | GgmlType::MXFP8
    )
}

/// `Body35`'s projection types (`PROJ`): every attention and delta-rule matrix.
const Q35_PROJ: &[GgmlType] = &[
    GgmlType::Q3_K,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q8_0,
];

/// `Body35`'s dense FFN types (`Sites35::CARD.dense`, pinned to this list by the chain's test):
/// The projection types and NVFP4.
// PIN(2026-10-29): widened with NVFP4; derivation: a dense FFN's NVFP4 gate, up and down launch
// `site::kgemv`'s `nvfp4_gemv` on the gemv arm and `site::gated`'s `gemm_nvfp4p` on the wide arm
// (`dispatch::slots_down`, `wide::slots_gemm`); an NVFP4 attention or delta-rule matrix still
// reads no launch, so the mixers keep the projection list.
pub const Q35_DENSE_FFN: &[GgmlType] = &[
    GgmlType::Q3_K,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q8_0,
    GgmlType::NVFP4,
];

/// The dense site `Body35` runs a part of a shared expert of `ty` with when
/// the layer keeps it apart from its routed stacks (a part whose type is not
/// its stack's): the qwen3moe body's dense owner ([`qwen3moe_dense`]), the
/// table's dense row kept to the forms that body's arena carries. A shared
/// expert of its stacks' types is joined into them and runs at their entries.
#[must_use]
pub const fn qwen35_shared(ty: GgmlType) -> Option<SiteEntry> {
    qwen3moe_dense(ty)
}

/// The latent mixer's absorbed key and value heads (`attn_k_b`,
/// `attn_v_b`): the weights its per-head gemv reads.
fn is_absorbed_pair(name: &str) -> bool {
    name.ends_with(".attn_k_b.weight") || name.ends_with(".attn_v_b.weight")
}

/// The latent mixer's selector query and head weights: `mla::select` reads
/// them on its branch stream through its own row-major gemv.
fn is_selector_site(name: &str) -> bool {
    name.ends_with(".indexer.attn_q_b.weight") || name.ends_with(".indexer.proj.weight")
}

/// A routed expert stack's gate or up rows (`ffn_gate_exps`/`ffn_up_exps`):
/// the whole-card bodies read the gate·up part at its own types.
fn routed_gate_up(name: &str) -> bool {
    name.ends_with(".ffn_gate_exps.weight") || name.ends_with(".ffn_up_exps.weight")
}

/// A routed expert stack's down rows (`ffn_down_exps`): the whole-card
/// bodies read the down part at its own types.
fn routed_down(name: &str) -> bool {
    name.ends_with(".ffn_down_exps.weight")
}

/// A routed stack no side of a plan runs: not one `card_rule` says a card
/// loads, and no fused qdot kernel serves it on the host at its row width —
/// `HostLayer::build`'s own admission ([`qdot::fuses`], the check the union
/// call makes of every stack it serves), asked here at the stack's first
/// dim, its `k`. A program with no card expert passes `|_| false`.
fn routed_unrun(t: &ModelTensor, card_rule: impl Fn(GgmlType) -> bool) -> bool {
    let k = t.dims.first().copied().unwrap_or(0);
    !card_rule(t.ty) && !qdot::fuses(t.ty, usize::try_from(k).unwrap_or(usize::MAX))
}

/// The qwen4exp body's card rule: the types its card experts load
/// ([`card_routed`](crate::arch::qwen35moe::place::card_routed)).
fn qwen38_card_routed(ty: GgmlType) -> bool {
    crate::arch::qwen35moe::place::card_routed(ty).is_some()
}

/// The card rule of a body whose card leg holds the form each layer's down
/// reads (glm5next, minimax-m3): a type program `p`'s card leg reads in some
/// part ([`leg_arms`], whose arms its plan rule reads by part), the type union
/// coverage reads as it does qwen4exp's.
fn leg_card_routed(p: Program, ty: GgmlType) -> bool {
    leg_arms(p).is_some_and(|a| (a.gate_up)(ty) || (a.down)(ty))
}

/// gemma-embedding2's projection of the token rows to the per-layer inputs,
/// which the encoder decodes on the host where it reads the per-layer
/// input's other matrices as q8_0.
fn per_layer_model_proj(name: &str) -> bool {
    name == crate::arch::gemma_embedding2::names::PER_LAYER_MODEL_PROJ
}

/// The `nemotron_h_moe` router instances: the (experts, kept) pairs
/// `gpu/src/arch/nemotron_h_moe/router.rs` is built for.
const NEMOTRON_ROUTERS: &[(u32, u32)] =
    &[(240, 9), (304, 9), (368, 9), (432, 9), (496, 9), (128, 6)];

/// The glm5next, mimo2, solar2 and raon1 bodies' router instances: the
/// (experts, kept) pairs each family's code holds, every instance imported
/// by name — glm5next's `glm5next`, mimo2's `mimo2` and solar2's
/// `sigmoid_320_8` (`gpu-deepseek41/src/router.rs`), raon1's
/// `sigmoid_128_6` (`gpu/src/route_body.rs`). A shape-named instance no
/// family imports (`sigmoid_256_6`) is none of theirs, so its needs are
/// items for these programs.
const SIGMOID_ROUTERS: &[(u32, u32)] = &[(288, 8), (256, 8), (320, 8), (128, 6)];

/// The muse-spark body's router instances: the file's 256 experts with the
/// top 8 (`gpu/src/route_body.rs sigmoid_256_8`, the instance the family
/// shares with MiMo's own compiled row) and the fixture's 16 rivals at the
/// same pick (`sigmoid_16_8`).
const MUSE_ROUTERS: &[(u32, u32)] = &[(256, 8), (16, 8)];

/// The type pins of `program` (every program's when `None`), [`TYPE_PINS`]
/// then [`ROW_PINS`], that tensor `t` is of a type none of reads, as items:
/// with a program, each pin of its role (and name) whose types do not hold
/// `t`'s; with none, the first such pin when no program's pin reads the
/// type.
fn weight_formats(program: Option<Program>, t: &ModelTensor) -> Vec<Need> {
    let matrix = t.dims.len() >= 2 && t.ty != GgmlType::F32;
    let takes = |ps: &[Program], role: Role, matrices: bool, names: Option<fn(&str) -> bool>| {
        program.is_none_or(|q| ps.contains(&q))
            && role == t.role
            && (!matrices || matrix)
            && names.is_none_or(|takes| takes(&t.name))
    };
    // Each pin that takes `t`, with whether it reads `t`'s type.
    let pins: Vec<(&'static str, bool)> = TYPE_PINS
        .iter()
        .filter(|pin| takes(pin.programs, pin.role, pin.matrices, pin.names))
        .map(|pin| (pin.what, pin.reads.contains(&t.ty)))
        .chain(
            ROW_PINS
                .iter()
                .filter(|pin| {
                    takes(
                        std::slice::from_ref(&pin.program),
                        pin.role,
                        pin.matrices,
                        pin.names,
                    )
                })
                .map(|pin| (pin.what, (pin.row)(t.ty))),
        )
        .collect();
    let refused: Vec<&'static str> = match program {
        Some(_) => pins
            .iter()
            .filter(|(_, reads)| !reads)
            .map(|(what, _)| *what)
            .collect(),
        // No program: an item only when no program's pin reads the type,
        // named by the tensor's role (no program's pin names its reader).
        None if !pins.is_empty() && pins.iter().all(|(_, reads)| !reads) => {
            vec![unrun_role(t.role)]
        }
        None => Vec::new(),
    };
    refused
        .into_iter()
        .map(|what| Need::WeightFormat { what, ty: t.ty })
        .collect()
}

/// The name of a weight-format item for a file no program runs: what the
/// tensor is, by its role.
fn unrun_role(role: Role) -> &'static str {
    match role {
        Role::Attention => "mixer matrices (attention, Mamba-2 or delta rule)",
        Role::Router => "router",
        Role::RoutedExperts => "routed experts",
        Role::SharedExpert => "shared expert",
        Role::DenseFfn => "dense FFN",
        Role::TokenEmbedding => "token embedding",
        Role::Head => "output head",
        _ => "weights",
    }
}

/// Every part of `spec` (with its tensors `model`) that no program runs:
/// layer by layer in layer order, then the tensor formats, then the
/// model-wide ones. Empty for a file a program runs whole.
#[must_use]
pub fn check(spec: &ModelSpec, model: &ModelTensors) -> Vec<Unimplemented> {
    check_with(spec, model, &rows(spec.arch))
}

/// The rows a file of `arch` is checked against: [`AVAILABLE`] and the rows its family's own code
/// keeps, which its registry row names (`registry::ArchRow::coverage`).
#[must_use]
pub fn rows(arch: Arch) -> Vec<Available> {
    let own = super::registry::row_of(arch.name()).and_then(|row| row.coverage);
    AVAILABLE
        .iter()
        .chain(own.into_iter().flatten())
        .copied()
        .collect()
}

/// [`check`] against the table `available` instead of [`rows`].
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
                // The V4.1 chain's rule: a stack passes when some card format
                // loads its type or a qdot fused kernel serves it on the host at
                // its row width ([`routed_unrun`], the one owner). A layer whose
                // stacks the chain's card experts do not read keeps them on the
                // host.
                Some(Program::Deepseek41Chain) => {
                    if routed_unrun(t, |ty| CardFormat::of(ty).is_some()) {
                        at(t.layer, Need::RoutedFormat(t.ty));
                    }
                }
                // The glm5next and minimax-m3 bodies' rule: a stack passes
                // when the program can run it — on the card by its leg's arms
                // (`leg_arms`: the common leg's in the file's blocks and
                // NVFP4's planes) or on the host by a qdot fused kernel at the
                // stack's row width ([`routed_unrun`], the one owner). A layer
                // with a stack the card experts do not read keeps all its
                // experts on the host.
                Some(p @ (Program::Glm5nextBody | Program::MiniMaxM3Body | Program::FlintBody)) => {
                    if routed_unrun(t, |ty| leg_card_routed(p, ty)) {
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
                // The mimo2 and solar2 bodies' rule: the same — a stack
                // passes when the common card leg runs its type
                // (`kernels::card_runs`) or the host runs it by a qdot fused
                // kernel at the stack's row width.
                Some(Program::Mimo2Body | Program::Solar2Body) => {
                    if routed_unrun(t, crate::placement::kernels::card_runs) {
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
                Some(Program::Qwen3moeBody | Program::Qwen35Body | Program::Raon1Body) => {}
                // These bodies are dense: a routed stack is an item.
                Some(Program::NemotronHBody | Program::GemmaEmbedding2Body) => {
                    at(t.layer, Need::RoutedFormat(t.ty));
                }
                // The latent MoE body's rule: a stack passes when the
                // program can run it — on the card by its expert rule
                // ([`latent_leg_runs`], `routed_rule`'s arm) or on the host by a
                // qdot fused kernel at the stack's row width
                // ([`routed_unrun`], the one owner). A layer with a stack the
                // card experts do not read keeps all its experts on the
                // host.
                Some(Program::NemotronHMoeBody) => {
                    if routed_unrun(t, latent_leg_runs) {
                        at(t.layer, Need::RoutedFormat(t.ty));
                    }
                }
                // The muse-spark body's rule, the same cut over its gated leg
                // ([`MUSESPARK_LEG`]): a whole-card load that refuses a type
                // names it here first.
                Some(Program::MuseSparkBody) => {
                    if routed_unrun(t, |ty| ty == MUSESPARK_LEG) {
                        at(t.layer, Need::RoutedFormat(t.ty));
                    }
                }
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
    // An encoder writes no tokens, so it calls no tool.
    if spec.chat.template.is_some() && spec.chat.tools.is_none() && spec.encoder.is_none() {
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
                "group 5, a row no Qwen body launches",
                Need::Gqa {
                    head,
                    value: head,
                    group: 5,
                    window: None,
                    sinks: false,
                    scaled: false,
                },
            ),
            (
                "group 4, the head-128 attention binding's row",
                Need::Gqa {
                    head,
                    value: head,
                    group: 4,
                    window: None,
                    sinks: false,
                    scaled: false,
                },
            ),
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

    /// The head-128 group-of-7 flash and the head-128 NORM rope without a
    /// head norm are launched by `Body35` alone: the other whole-card body reads
    /// neither as covered.
    #[test]
    fn a_group_of_7_flash_and_a_norm_rope_are_body35s_alone() {
        use super::AVAILABLE;
        use models::RopeMode;
        let g7 = Need::Gqa {
            head: 128,
            value: 128,
            group: 7,
            window: None,
            sinks: false,
            scaled: false,
        };
        let norm = Need::QkRope {
            qk_norm: false,
            head: 128,
            mode: RopeMode::NormTail,
            dims: 128,
        };
        let runs = |program: Program, need: &Need| {
            AVAILABLE
                .iter()
                .any(|a| a.programs.contains(&program) && (a.runs)(need))
        };
        for (what, need) in [("the group-7 flash", &g7), ("the NORM rope", &norm)] {
            assert!(
                runs(Program::Qwen35Body, need),
                "{what}: Body35 launches it"
            );
            assert!(
                !runs(Program::Qwen3moeBody, need),
                "{what} read as the other whole-card body's"
            );
        }
    }

    /// The four sigmoid bodies launch one instance each, imported by name at
    /// its own (experts, kept) pair (`SIGMOID_ROUTERS`): a shape-named
    /// instance no family imports — the 256/top-6 row Flint's reader selects
    /// — is no row's for them, while each body's own pair stays covered.
    /// Mutant: the row's `runs` without its `SIGMOID_ROUTERS` check, which
    /// reads the 256/top-6 need as covered for all four.
    #[test]
    fn a_router_instance_no_sigmoid_body_launches_is_uncovered() {
        use models::shape::{MoeShape, rules};

        use super::AVAILABLE;
        let router = |experts: u32, top_k: u32| {
            Need::Router(MoeShape {
                rule: rules::BIASED_SIGMOID,
                experts,
                top_k,
            })
        };
        let runs = |program: Program, n: &Need| {
            AVAILABLE
                .iter()
                .any(|a| a.programs.contains(&program) && (a.runs)(n))
        };
        for (program, (experts, top_k)) in [
            (Program::Glm5nextBody, (288, 8)),
            (Program::Mimo2Body, (256, 8)),
            (Program::Solar2Body, (320, 8)),
            (Program::Raon1Body, (128, 6)),
        ] {
            assert!(
                runs(program, &router(experts, top_k)),
                "{program:?}'s own router pair read as uncovered"
            );
        }
        for program in [
            Program::Glm5nextBody,
            Program::Mimo2Body,
            Program::Solar2Body,
            Program::Raon1Body,
        ] {
            assert!(
                !runs(program, &router(256, 6)),
                "{program:?} read as launching the 256/top-6 sigmoid instance"
            );
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
    /// projection, the head and the embedding.
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
    }

    /// The file's global tensor `name` (no layer) of role `role` and type `ty`.
    fn global(name: &str, role: Role, ty: GgmlType) -> ModelTensor {
        ModelTensor {
            name: name.into(),
            layer: None,
            ..matrix("", role, ty)
        }
    }

    /// The qwen3moe body's attention, head and embedding read the card kernel
    /// table's rows ([`super::ROW_PINS`]): every attention matrix a dense
    /// site launches (q, k, v and output alike), the head types `head.rs`
    /// launches, the embedding rows `elem.rs` and `q38.rs` launch; a type
    /// with no row is an item that names the row.
    // PIN(2026-10-10): widened from q4_K attention (q, k and output; v q4_K
    // or q6_K), a q6_K head and q4_K embedding rows to the table's rows, so
    // the unsloth Qwen3-Coder Q4_K_S, UD-Q4_K_XL and Q5_K_M files' attention,
    // embedding and head are no items. Derivation: each type admitted below
    // has a launch the qwen3moe body runs on both paths (`gpu/src/site.rs`
    // `kgemv`/`gemv` and `gemm`; `Head::enqueue`; `enqueue_embed_rows_kquant`
    // and `embed_rows_q8_0`). Attention follows [`qwen3moe_dense`], the dense
    // row kept to the forms the body's arena carries, and `kernels`' own
    // tests pin each row's set. The routed stacks' pins are the FFN
    // dispatch's rows, below.
    #[test]
    fn the_qwen3moe_body_reads_the_kernel_tables_rows() {
        // Every mismatch is collected, so a base run lists each type it
        // refuses.
        let mut bad = Vec::<String>::new();
        let mut check = |what: String, t: ModelTensor, want: &[String]| {
            let got = items(Program::Qwen3moeBody, &t);
            if got != want {
                bad.push(format!("{what}: items {got:?}, want {want:?}"));
            }
        };
        let none = Vec::<String>::new();
        for name in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
        ] {
            for ty in [
                GgmlType::Q3_K,
                GgmlType::Q4_K,
                GgmlType::Q5_K,
                GgmlType::Q6_K,
                GgmlType::Q8_0,
            ] {
                check(
                    format!("{name} {ty}"),
                    matrix(name, Role::Attention, ty),
                    &none,
                );
            }
            for ty in [
                GgmlType::BF16,
                GgmlType::F16,
                GgmlType::IQ4_XS,
                GgmlType::Q5_0,
            ] {
                let item = format!(
                    "{ty} attention matrices (the body's dense sites: the types qwen3moe_dense \
                     names)"
                );
                check(
                    format!("{name} {ty}"),
                    matrix(name, Role::Attention, ty),
                    &[item],
                );
            }
        }
        // PIN(2026-10-25): the head gains Q5_K (`kernels::head`'s row: the `_sel` Q5_K
        // gemv over the matrix as one expert), so this body's `output` of Q5_K is read
        // by the common head instead of an item; every other type is unchanged.
        for ty in [
            GgmlType::Q6_K,
            GgmlType::Q5_K,
            GgmlType::Q4_K,
            GgmlType::Q8_0,
        ] {
            check(
                format!("head {ty}"),
                global("output.weight", Role::Head, ty),
                &none,
            );
        }
        {
            let ty = GgmlType::BF16;
            check(
                format!("head {ty}"),
                global("output.weight", Role::Head, ty),
                &[format!(
                    "{ty} output head (the types head_of_widened names)"
                )],
            );
        }
        for ty in [
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
        ] {
            check(
                format!("embedding {ty}"),
                global("token_embd.weight", Role::TokenEmbedding, ty),
                &none,
            );
        }
        for ty in [GgmlType::F16, GgmlType::BF16] {
            check(
                format!("embedding {ty}"),
                global("token_embd.weight", Role::TokenEmbedding, ty),
                &[format!(
                    "{ty} token embedding (the card's rows: the types embed_of_widened names)"
                )],
            );
        }
        // The routed stacks are the placed card leg's rows (`prompt_leg_gate_up`,
        // `prompt_leg_down`): a Q3_K gate·up is refused by name, a Q6_K gate·up
        // and a Q5_K down run.
        check(
            "routed gate q3_K".into(),
            matrix("ffn_gate_exps.weight", Role::RoutedExperts, GgmlType::Q3_K),
            &[
                "q3_K routed experts gate and up (the body reads q4_K, q5_K, q6_K, iq2_xxs, iq2_xs, iq3_xxs, iq1_s, iq3_s, iq2_s, iq4_xs, iq1_m and q8_0)"
                    .to_string(),
            ],
        );
        check(
            "routed gate q6_K".into(),
            matrix("ffn_gate_exps.weight", Role::RoutedExperts, GgmlType::Q6_K),
            &none,
        );
        check(
            "routed down q5_K".into(),
            matrix("ffn_down_exps.weight", Role::RoutedExperts, GgmlType::Q5_K),
            &none,
        );
        assert!(
            bad.is_empty(),
            "{} mismatches:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    /// The table-owned pins follow the table, not a list: each pin's verdict
    /// on every type equals whether the table's row names a launch for it.
    #[test]
    fn a_row_pin_follows_its_table_row() {
        use super::ROW_PINS;
        let types = [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::Q2_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
            GgmlType::IQ3_XXS,
            GgmlType::IQ4_NL,
            GgmlType::IQ4_XS,
        ];
        for pin in ROW_PINS {
            // A name the pin takes: the pins scoped by name are the routed
            // stacks' parts, the qwen38 shared expert's down and
            // gemma-embedding2's per-layer input projection.
            let name = [
                "blk.0.attn_q.weight",
                "blk.0.ffn_gate_exps.weight",
                "blk.0.ffn_down_exps.weight",
                "blk.0.ffn_down_shexp.weight",
                "per_layer_model_proj.weight",
            ]
            .into_iter()
            .find(|n| pin.names.is_none_or(|takes| takes(n)))
            .expect("a name the pin takes");
            for ty in types {
                let t = ModelTensor {
                    name: name.into(),
                    role: pin.role,
                    ..matrix("", pin.role, ty)
                };
                let item = weight_formats(Some(pin.program), &t)
                    .iter()
                    .any(|n| n.to_string() == format!("{ty} {}", pin.what));
                let is_matrix = t.dims.len() >= 2 && ty != GgmlType::F32;
                assert_eq!(item, !(pin.row)(ty) && (!pin.matrices || is_matrix), "{ty}");
            }
        }
    }

    /// The qwen3moe family's routed pins, by part: both bodies run the FFN
    /// dispatch's rows of the card kernel table and the common card leg's
    /// beside them (`prompt_leg_gate_up`, `prompt_leg_down`), one pair for the
    /// family's placed side.
    // PIN(2026-10-10): `Body35`'s routed pins moved from the joined stacks'
    // q4_K gate·up and q4_K or q6_K down to the table's rows its dispatch
    // runs (`qwen35_gate_up`, `qwen35_down`): q5_K gate·up and down left its
    // refused types; red on the tree before the move by those two. The
    // qwen3moe body's moved the same way (q4_K gate·up, q4_K or q6_K down
    // before): the dispatch is one code path, so one row decides both.
    // PIN(2026-10-22): `Body35`'s pins moved on to the rows of its placed card
    // leg (`prompt_leg_gate_up`, `prompt_leg_down`: the dispatch's types
    // and the common leg's): a q6_K, iq3_xxs, iq4_xs or q8_0 gate·up and an
    // iq3_xxs, iq4_xs, q5_1, q8_0 or iq4_nl down left its refused types, each a type the
    // common leg launches with a prompt GEMM; red on the tree before the move
    // by those. The qwen3moe body's moved with them.
    // PIN(2026-10-22): the qwen3moe body's pins moved to the same rows
    // (`prompt_leg_gate_up`, `prompt_leg_down`): its placed side runs the common
    // leg's layers (`placed_ffn`, one for both bodies); a q6_K, iq3_xxs, iq4_xs
    // or q8_0 gate·up and an iq3_xxs, iq4_xs, q5_1, q8_0 or iq4_nl down left its
    // refused types; red on the tree before the move by those.
    // PIN(2026-10-22): both bodies' pins gain iq3_s on both parts (its prompt
    // GEMM `gemm_iq3s` gives the table's gemm row `prompt_leg_gate_up` and
    // `prompt_leg_down` read); red on the tree before the move by those.
    // PIN(2026-10-25): and iq2_xs, iq2_s, iq1_s and iq1_m, whose prompt GEMMs
    // (`gemm_iq2xs`, `gemm_iq2s`, `gemm_iq1s`, `gemm_iq1m`) give the table's
    // gemm rows; an mxfp4 gate and up stand for the refused types.
    #[test]
    fn the_qwen3moe_bodies_read_routed_stacks_by_part() {
        use GgmlType::{
            IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, MXFP4, Q3_K, Q4_K, Q5_1,
            Q5_K, Q6_K, Q8_0,
        };
        type Parts = (&'static str, &'static [GgmlType], &'static [GgmlType]);
        let rows: (Parts, Parts) = (
            (
                "q4_K, q5_K, q6_K, iq2_xxs, iq2_xs, iq3_xxs, iq1_s, iq3_s, iq2_s, iq4_xs, iq1_m and q8_0",
                &[
                    Q4_K, Q5_K, Q6_K, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, Q8_0,
                ],
                &[Q3_K, MXFP4],
            ),
            (
                "q4_K, q5_K, q6_K, iq2_xxs, iq2_xs, iq3_xxs, iq1_s, iq3_s, iq2_s, iq4_xs, iq1_m, q5_1, q8_0 and iq4_nl",
                &[
                    Q4_K, Q5_K, Q6_K, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, Q5_1,
                    Q8_0, IQ4_NL,
                ],
                &[Q3_K, MXFP4],
            ),
        );
        let bodies = [Program::Qwen3moeBody, Program::Qwen35Body].map(|p| (p, rows.0, rows.1));
        for (program, gate_up, down) in bodies {
            for (names, part, (reads, runs, refused)) in [
                (
                    &["ffn_gate_exps.weight", "ffn_up_exps.weight"][..],
                    "gate and up",
                    gate_up,
                ),
                (&["ffn_down_exps.weight"][..], "down", down),
            ] {
                for name in names {
                    for (ty, item) in runs
                        .iter()
                        .map(|t| (t, false))
                        .chain(refused.iter().map(|t| (t, true)))
                    {
                        let want = format!("{ty} routed experts {part} (the body reads {reads})");
                        assert_eq!(
                            items(program, &matrix(name, Role::RoutedExperts, *ty)),
                            if item {
                                vec![want]
                            } else {
                                Vec::<String>::new()
                            },
                            "{program:?} {name} {ty}"
                        );
                    }
                }
            }
        }
    }

    /// The raon1 body reads routed stacks by part: a gate and an up of the common leg's gate·up
    /// types (q4_K, q5_K), a down of its down types or q5_0 (the packed layout); a q8_0 or q6_K gate
    /// and up and a q6_K down are items by name and part. Mutant: `resident_down_runs` without its
    /// q5_0 arm.
    #[test]
    fn the_raon1_body_reads_routed_stacks_by_part() {
        let item = |name: &str, ty: GgmlType| {
            items(Program::Raon1Body, &matrix(name, Role::RoutedExperts, ty))
        };
        for name in ["ffn_gate_exps.weight", "ffn_up_exps.weight"] {
            for ty in [GgmlType::Q4_K, GgmlType::Q5_K] {
                assert_eq!(item(name, ty), Vec::<String>::new(), "{name} {ty}");
            }
            for ty in [GgmlType::Q8_0, GgmlType::Q6_K, GgmlType::Q5_0] {
                assert_eq!(item(name, ty).len(), 1, "{name} {ty}: {:?}", item(name, ty));
            }
        }
        for ty in [
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q8_0,
            GgmlType::Q5_0,
        ] {
            assert_eq!(
                item("ffn_down_exps.weight", ty),
                Vec::<String>::new(),
                "down {ty}"
            );
        }
        assert_eq!(item("ffn_down_exps.weight", GgmlType::Q6_K).len(), 1);
    }

    /// The device runs SiTU at one value beta and a finite positive gate beta: the raon1 rows
    /// cover the file's pair and no other. Mutant: the value beta of the row at another value.
    #[test]
    fn the_raon1_situ_rows_run_the_one_value_beta() {
        use super::AVAILABLE;
        let covered = |need: &Need| {
            AVAILABLE
                .iter()
                .any(|a| a.programs.contains(&Program::Raon1Body) && (a.runs)(need))
        };
        let situ = |gate_beta: f32, value_beta: f32| models::Act::Situ {
            gate_beta,
            value_beta,
        };
        for shared in [false, true] {
            let need = |act| Need::ExpertAct { shared, act };
            assert!(covered(&need(situ(4.0, 25.0))), "shared {shared}");
            for (g, v) in [(4.0, 20.0), (0.0, 25.0), (-1.0, 25.0), (f32::NAN, 25.0)] {
                assert!(!covered(&need(situ(g, v))), "shared {shared} {g} {v}");
            }
        }
    }

    /// The FFN pins are the card kernel table's: over every type id, a
    /// routed part's matrix is an item exactly when its body's rows give its
    /// type no entry (both bodies': [`prompt_leg_gate_up`],
    /// [`prompt_leg_down`]), `Body35`'s shared expert's exactly when
    /// [`qwen35_shared`] does not, and each pin's text names the types its row
    /// takes (a pin reads no F32 matrix).
    #[test]
    fn the_ffn_pins_are_the_card_kernel_table() {
        use super::{prompt_leg_down, prompt_leg_gate_up, qwen35_shared};
        type Row = fn(GgmlType) -> bool;
        let parts: [(&str, Role, Program, Row); 5] = [
            (
                "ffn_gate_exps.weight",
                Role::RoutedExperts,
                Program::Qwen3moeBody,
                prompt_leg_gate_up,
            ),
            (
                "ffn_down_exps.weight",
                Role::RoutedExperts,
                Program::Qwen3moeBody,
                prompt_leg_down,
            ),
            (
                "ffn_gate_exps.weight",
                Role::RoutedExperts,
                Program::Qwen35Body,
                prompt_leg_gate_up,
            ),
            (
                "ffn_down_exps.weight",
                Role::RoutedExperts,
                Program::Qwen35Body,
                prompt_leg_down,
            ),
            (
                "ffn_up_shexp.weight",
                Role::SharedExpert,
                Program::Qwen35Body,
                |t| qwen35_shared(t).is_some(),
            ),
        ];
        let rows = |row: Row| -> Vec<GgmlType> {
            (0..64u32)
                .map(GgmlType::from_u32)
                .filter(|&t| t != GgmlType::F32 && row(t))
                .collect()
        };
        for (name, role, program, row) in parts {
            for id in 0..64u32 {
                let ty = GgmlType::from_u32(id);
                if ty == GgmlType::F32 {
                    continue;
                }
                assert_eq!(
                    items(program, &matrix(name, role, ty)).is_empty(),
                    row(ty),
                    "{program:?} {name} {ty:?}"
                );
            }
        }
        // Each pin's text names, in any order, exactly the types its row
        // takes: the list inside its parentheses, "a, b and c".
        for (name, role, program, row) in parts {
            let text = items(program, &matrix(name, role, GgmlType::BF16));
            let [text] = text.as_slice() else {
                panic!("{program:?} {name}: one item for bf16, got {text:?}");
            };
            let list = text
                .split_once("(the body reads ")
                .and_then(|(_, l)| l.strip_suffix(')'))
                .unwrap_or_else(|| panic!("{name}: no type list in {text:?}"));
            let mut named: Vec<String> = list
                .replace(" and ", ", ")
                .split(", ")
                .map(str::to_string)
                .collect();
            let mut want: Vec<String> = rows(row).iter().map(ToString::to_string).collect();
            named.sort();
            want.sort();
            assert_eq!(named, want, "{program:?} {name}: {text}");
        }
    }

    /// The FFN layer shapes of the four most downloaded Qwen3.6-35B-A3B
    /// files `-hf` picks (unsloth's `UD-Q4_K_M` with and without the
    /// next-token layer, bartowski's and ggml-org's `Q4_K_M`, their headers
    /// in the quantwide mirror) are no items: a q5_K routed down beside q4_K
    /// gate and up, a q8_0 shared expert beside q4_K, q5_K or q6_K routed
    /// stacks, a q6_K shared expert beside q4_K ones. A part of a type the
    /// table names no `Body35` launch for stays an item by name.
    // PIN(2026-10-10): red on the tree before the table's rows (the q5_K
    // down, the q8_0 and q6_K shared experts beside stacks of other types).
    // PIN(2026-10-22): the q8_0 routed gate·up and down it held as items run on
    // the common card leg now; the routed items it keeps are types that leg has
    // no prompt GEMM for (an iq2_xs gate·up, an mxfp4 down).
    #[test]
    fn the_hf_qwen36_layer_shapes_are_no_items() {
        use GgmlType::{Q4_K, Q5_K, Q6_K, Q8_0};
        let parts = |routed: [GgmlType; 3], shared: [GgmlType; 3]| {
            let mut out = Vec::new();
            for (part, (r, s)) in ["gate", "up", "down"]
                .iter()
                .zip(routed.into_iter().zip(shared))
            {
                out.push(matrix(
                    &format!("ffn_{part}_exps.weight"),
                    Role::RoutedExperts,
                    r,
                ));
                out.push(matrix(
                    &format!("ffn_{part}_shexp.weight"),
                    Role::SharedExpert,
                    s,
                ));
            }
            out
        };
        for (file, routed, shared) in [
            ("unsloth UD-Q4_K_M", [Q4_K, Q4_K, Q5_K], [Q8_0, Q8_0, Q8_0]),
            ("unsloth UD-Q4_K_M", [Q4_K, Q4_K, Q6_K], [Q8_0, Q8_0, Q8_0]),
            ("bartowski Q4_K_M", [Q4_K, Q4_K, Q6_K], [Q8_0, Q8_0, Q8_0]),
            ("bartowski Q4_K_M", [Q4_K, Q4_K, Q4_K], [Q6_K, Q6_K, Q6_K]),
            ("ggml-org Q4_K_M", [Q4_K, Q4_K, Q4_K], [Q8_0, Q8_0, Q8_0]),
        ] {
            for t in parts(routed, shared) {
                assert_eq!(
                    items(Program::Qwen35Body, &t),
                    Vec::<String>::new(),
                    "{file}: {} {}",
                    t.name,
                    t.ty
                );
            }
        }
        for (name, role, ty, want) in [
            (
                "ffn_gate_exps.weight",
                Role::RoutedExperts,
                GgmlType::MXFP4,
                "mxfp4 routed experts gate and up (the body reads q4_K, q5_K, q6_K, iq2_xxs, iq2_xs, iq3_xxs, iq1_s, iq3_s, iq2_s, iq4_xs, iq1_m and q8_0)",
            ),
            (
                "ffn_down_exps.weight",
                Role::RoutedExperts,
                GgmlType::MXFP4,
                "mxfp4 routed experts down (the body reads q4_K, q5_K, q6_K, iq2_xxs, iq2_xs, iq3_xxs, iq1_s, iq3_s, iq2_s, iq4_xs, iq1_m, q5_1, q8_0 and iq4_nl)",
            ),
            (
                "ffn_gate_shexp.weight",
                Role::SharedExpert,
                GgmlType::IQ4_XS,
                "iq4_xs shared expert (the body reads q3_K, q4_K, q5_K, q6_K and q8_0)",
            ),
        ] {
            assert_eq!(
                items(Program::Qwen35Body, &matrix(name, role, ty)),
                [want],
                "{name} {ty}"
            );
        }
    }

    /// A type `Body35` cannot launch is still an item by name: bf16
    /// attention and dense FFN (its FFN's pins:
    /// `the_ffn_pins_are_the_card_kernel_table`).
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
            ["bf16 dense FFN (the body reads q3_K, q4_K, q5_K, q6_K, q8_0 and nvfp4)"]
        );
    }

    /// `Body35` reads NVFP4 at its dense FFN and its head: those matrices are no items, while an
    /// NVFP4 attention, delta-rule, shared-expert or embedding matrix is one by name (their lists
    /// hold none).
    // FAIL-first: without NVFP4 in the dense FFN or head pin, that row of this test is an item.
    #[test]
    fn body35_reads_nvfp4_at_its_dense_ffn_and_head_only() {
        let body = |name, role| items(Program::Qwen35Body, &matrix(name, role, GgmlType::NVFP4));
        for (name, role) in [
            ("ffn_gate.weight", Role::DenseFfn),
            ("ffn_up.weight", Role::DenseFfn),
            ("ffn_down.weight", Role::DenseFfn),
            ("output.weight", Role::Head),
        ] {
            assert_eq!(body(name, role), Vec::<String>::new(), "{name}");
        }
        for (name, role, want) in [
            (
                "attn_qkv.weight",
                Role::Attention,
                "nvfp4 attention and delta-rule matrices (the body reads q3_K, q4_K, q5_K, q6_K and q8_0)",
            ),
            (
                "ffn_gate_shexp.weight",
                Role::SharedExpert,
                "nvfp4 shared expert (the body reads q3_K, q4_K, q5_K, q6_K and q8_0)",
            ),
            (
                "token_embd.weight",
                Role::TokenEmbedding,
                "nvfp4 token embedding (the card reads q3_K, q4_K, q5_K and q6_K rows and q8_0 planes)",
            ),
        ] {
            assert_eq!(body(name, role), [want], "{name}");
        }
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
        // The host's alone: q6_K, q3_K and q2_0 stacks are host-served.
        for ty in [GgmlType::Q6_K, GgmlType::Q3_K, GgmlType::Q2_0] {
            assert!(
                !super::routed_unrun(&routed(ty, 2560), super::qwen38_card_routed),
                "{ty}"
            );
        }
        // No side: a type no card expert kernel reads and no fused qdot
        // kernel serves.
        // PIN(2026-10-25): IQ2_S left this list when qdot's generic host dot began to serve it; q4_0 has neither side.
        for ty in [GgmlType::Q4_0, GgmlType::F32] {
            assert!(
                super::routed_unrun(&routed(ty, 2560), super::qwen38_card_routed),
                "{ty}"
            );
        }
    }

    /// The items [`super::check`] lists for a glm5next file holding
    /// `tensors` whose text holds `needle`: each its layer and text. A spec
    /// with no layers and no chat surface stands for the rest of the file,
    /// and `needle` leaves its items out. The check asks this machine's
    /// `qdot::fuses`: on x86_64 the x86-64-v3 host the engine itself requires,
    /// elsewhere the scalar mirrors, which fuse the same types.
    fn glm5next_items(tensors: Vec<ModelTensor>, needle: &str) -> Vec<(Option<usize>, String)> {
        arch_items(Arch::Glm5Next, tensors, needle)
    }

    /// The items whose text holds `needle` of a file of `arch` whose tensors are `tensors` over
    /// five layers of 288 experts, its description's layers empty.
    fn arch_items(
        arch: Arch,
        tensors: Vec<ModelTensor>,
        needle: &str,
    ) -> Vec<(Option<usize>, String)> {
        let spec = ModelSpec {
            arch,
            hidden: 4096,
            vocab: 154_880,
            ctx_train: 1_048_576,
            rms_eps: 1e-5,
            layers: Vec::new(),
            mtp: Vec::new(),
            hc: None,
            engram: None,
            encoder: None,
            logit_epilogue: None,
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
            bf16_words: |_| false,
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

    /// A routed stack of a minimax-m3 file passes when the program can run it — on the card by
    /// its leg's arms, the glm5next body's (the common leg's in the file's blocks, NVFP4's and
    /// NVFP4_ROWS' planes), or on the host by a fused qdot kernel at its row width — and a type
    /// neither side runs is an item for each layer that holds it. The widths here are off every
    /// fused kernel's K unit, so the card's arms alone decide. FAIL-first: under the mimo2 and
    /// solar2 bodies' rule (`kernels::card_runs`, both roles in the file's blocks), where the
    /// body sat before, the NVFP4_ROWS layer and the q6_K down are items.
    #[test]
    fn a_minimaxm3_routed_stack_runs_on_its_planes_leg_or_the_host() {
        let none = Vec::<(Option<usize>, String)>::new();
        let item = |ty: GgmlType| -> Vec<(Option<usize>, String)> {
            [3, 4]
                .into_iter()
                .map(|l| (Some(l), Need::RoutedFormat(ty).to_string()))
                .collect()
        };
        let routed = |gate_up: GgmlType, down: GgmlType| {
            let stack = |layer: usize, stem: &str, ty: GgmlType| ModelTensor {
                name: format!("blk.{layer}.{stem}.weight"),
                shard: 0,
                layer: Some(layer),
                role: Role::RoutedExperts,
                ty,
                dims: vec![2000, 2000, 288],
                file_bytes: 0,
                gathered_rows: None,
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
            arch_items(Arch::MiniMaxM3, tensors, "routed experts on a card")
        };
        let cases = [
            (GgmlType::NVFP4_ROWS, GgmlType::NVFP4_ROWS, none.clone()),
            (GgmlType::Q4_K, GgmlType::Q6_K, none),
            (GgmlType::BF16, GgmlType::Q4_K, item(GgmlType::BF16)),
        ];
        let wrong: Vec<_> = cases
            .iter()
            .map(|(gate_up, down, want)| (gate_up, down, want, routed(*gate_up, *down)))
            .filter(|(_, _, want, got)| want != &got)
            .collect();
        assert!(
            wrong.is_empty(),
            "(gate and up, down, want, got) of the cases off the rule: {wrong:#?}"
        );
    }

    /// A routed stack of a V4.1 file passes when some card format loads its type or a fused qdot
    /// kernel serves it on the host at its row width, so a layer whose stacks the chain's card
    /// leg does not read keeps its experts on the host; a type neither side runs is an item for
    /// each layer that holds it.
    // PIN(2026-10-25): the V4.1 routed rule widened from `CardFormat::of` to card-or-host so a
    // file's stacks of a type with a fused kernel and no card format (q2_K) are no items; a q4_0
    // stack, which no card format or host kernel reads, is still an item.
    // FAIL-first: under `CardFormat::of` alone the q2_K layers are items.
    #[test]
    fn a_deepseek41_routed_stack_runs_on_a_card_format_or_the_host() {
        let none = Vec::<(Option<usize>, String)>::new();
        let routed = |gate_up: GgmlType, down: GgmlType| {
            let stack = |layer: usize, stem: &str, ty: GgmlType| ModelTensor {
                name: format!("blk.{layer}.{stem}.weight"),
                shard: 0,
                layer: Some(layer),
                role: Role::RoutedExperts,
                ty,
                dims: vec![5120, 2304, 384],
                file_bytes: 0,
                gathered_rows: None,
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
            arch_items(Arch::Deepseek41, tensors, "routed experts on a card")
        };
        let item = |ty: GgmlType| -> Vec<(Option<usize>, String)> {
            [3, 4]
                .into_iter()
                .map(|l| (Some(l), Need::RoutedFormat(ty).to_string()))
                .collect()
        };
        let cases = [
            // The chain's card leg: q3_K gate and up with a q4_K down.
            (GgmlType::Q3_K, GgmlType::Q4_K, none.clone()),
            // A card format the chain's leg does not read: the layer stays on the host.
            (GgmlType::IQ2_XXS, GgmlType::IQ2_XXS, none.clone()),
            (GgmlType::IQ1_M, GgmlType::IQ1_M, none.clone()),
            // The host's alone: no card format, a fused kernel.
            (GgmlType::Q2_K, GgmlType::IQ2_XXS, none.clone()),
            (GgmlType::Q2_K, GgmlType::Q2_K, none),
            // No side: q4_0 has no card format and no fused kernel.
            (GgmlType::Q4_0, GgmlType::Q4_K, item(GgmlType::Q4_0)),
        ];
        let wrong: Vec<_> = cases
            .iter()
            .map(|(gate_up, down, want)| (gate_up, down, want, routed(*gate_up, *down)))
            .filter(|(_, _, want, got)| want != &got)
            .collect();
        assert!(
            wrong.is_empty(),
            "(gate and up, down, want, got) of the cases off the rule: {wrong:#?}"
        );
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
    /// — on the card by its leg's arms (q4_K, q5_K, nvfp4 in its planes) or on the host
    /// by a fused qdot kernel at its row width, so a layer with a stack the
    /// card does not read keeps its experts on the host — and a type neither
    /// side runs is an item for each layer that holds it, named by its type,
    /// whether it is a gate, up or down.
    // PIN(2026-10-09): the glm5next routed rule widened from `CardFormat::of`
    // to card-or-host so a file's i-quant stacks (iq3_xxs, iq4_xs gate and
    // up or down) are no items; a bf16 stack, which `CardFormat::of` took but
    // no host kernel reads, is an item.
    // PIN(2026-10-22): the glm5next leg holds NVFP4 stacks in their planes and
    // quantizes the 32-value form their down reads, so an NVFP4 layer is no
    // item (it was one).
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
            // The host's alone: UD-Q2_K_XL's trunk (iq2_xs gate and up, iq3_xxs
            // down) and its NextN layer (q2_K gate and up, q3_K down).
            (GgmlType::IQ2_XS, GgmlType::IQ3_XXS, none.clone()),
            (GgmlType::Q2_K, GgmlType::Q3_K, none.clone()),
            // The card's in its planes and the host's by a fused qdot kernel
            // at its 64-value block (placement's
            // `glm5next_arms_are_the_leg_with_planes_over_both_forms` holds
            // the rule-left-on-the-common-leg mutant).
            (GgmlType::NVFP4, GgmlType::NVFP4, none),
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
    /// `output` is an item.
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
    }

    /// A V4.1 file's attention matrices are items unless their type is one
    /// the chain's dense sites read (q3_K, q5_K, q8_0; the compressor's and
    /// the indexer's also bf16), and its head unless q6_K or q5_K.
    #[test]
    fn the_ds41_dense_pins_read_the_chains_types() {
        let items = |name: &str, role: Role, ty: GgmlType, needle: &str| {
            let spec = ModelSpec {
                arch: Arch::Deepseek41,
                hidden: 4096,
                vocab: 129_280,
                ctx_train: 65_536,
                rms_eps: 1e-6,
                layers: Vec::new(),
                mtp: Vec::new(),
                hc: None,
                engram: None,
                encoder: None,
                logit_epilogue: None,
                chat: ChatSpec {
                    pre: String::new(),
                    template: None,
                    tools: None,
                    reasoning: None,
                },
            };
            let model = ModelTensors {
                tensors: vec![ModelTensor {
                    name: name.to_string(),
                    shard: 0,
                    layer: None,
                    role,
                    ty,
                    dims: vec![4096, 1024],
                    file_bytes: 0,
                    gathered_rows: None,
                }],
                layers: 9,
                experts: 0,
                experts_used: 0,
                bf16_words: |_| false,
            };
            super::check(&spec, &model)
                .into_iter()
                .filter(|u| u.feature.contains(needle))
                .map(|u| u.feature)
                .collect::<Vec<String>>()
        };
        let attn = |name: &str, ty| items(name, Role::Attention, ty, "matrices");
        for ty in [GgmlType::Q3_K, GgmlType::Q5_K, GgmlType::Q8_0] {
            for name in [
                "blk.0.attn_q_b.weight",
                "blk.2.attn_compressor_kv.weight",
                "blk.7.indexer.proj.weight",
            ] {
                assert_eq!(attn(name, ty), Vec::<String>::new(), "{name} {ty}");
            }
        }
        for name in [
            "blk.2.attn_compressor_gate.weight",
            "blk.7.indexer.attn_q_b.weight",
            "blk.4.indexer.attn_k.weight",
        ] {
            assert_eq!(attn(name, GgmlType::BF16), Vec::<String>::new(), "{name}");
        }
        assert_eq!(
            attn("blk.0.attn_q_a.weight", GgmlType::BF16),
            ["bf16 attention matrices (the chain reads q3_K, q5_K and q8_0)".to_string()]
        );
        assert_eq!(
            attn("blk.2.attn_compressor_kv.weight", GgmlType::Q6_K),
            [
                "q6_K compressor and indexer matrices (the chain reads q3_K, q5_K, q8_0 and bf16)"
                    .to_string()
            ]
        );
        let head = |ty| items("output.weight", Role::Head, ty, "output head");
        for ty in [GgmlType::Q6_K, GgmlType::Q5_K] {
            assert_eq!(head(ty), Vec::<String>::new(), "{ty}");
        }
        assert_eq!(
            head(GgmlType::Q4_K),
            ["q4_K output head (the head reads q6_K and q5_K)".to_string()]
        );
    }

    /// A nemotron_h file's token table and head are items unless their type
    /// is one the body reads: q5_0 rows for the table, the common head's
    /// q8_0, q6_K or q4_K for `output`.
    #[test]
    fn the_nemotron_embedding_and_head_pins_read_the_bodys_types() {
        let whole = |name: &str, role: Role, ty: GgmlType, needle: &str| {
            let spec = ModelSpec {
                arch: Arch::NemotronH,
                hidden: 3136,
                vocab: 131_072,
                ctx_train: 262_144,
                rms_eps: 1e-5,
                layers: Vec::new(),
                mtp: Vec::new(),
                hc: None,
                engram: None,
                encoder: None,
                logit_epilogue: None,
                chat: ChatSpec {
                    pre: String::new(),
                    template: None,
                    tools: None,
                    reasoning: None,
                },
            };
            let model = ModelTensors {
                tensors: vec![ModelTensor {
                    name: name.to_string(),
                    shard: 0,
                    layer: None,
                    role,
                    ty,
                    dims: vec![3136, 131_072],
                    file_bytes: 0,
                    gathered_rows: None,
                }],
                layers: 42,
                experts: 0,
                experts_used: 0,
                bf16_words: |_| false,
            };
            super::check(&spec, &model)
                .into_iter()
                .filter(|u| u.feature.contains(needle))
                .map(|u| u.feature)
                .collect::<Vec<String>>()
        };
        let embed = |ty| {
            whole(
                "token_embd.weight",
                Role::TokenEmbedding,
                ty,
                "token embedding",
            )
        };
        assert_eq!(embed(GgmlType::Q5_0), Vec::<String>::new());
        assert_eq!(
            embed(GgmlType::Q8_0),
            ["q8_0 token embedding (the card reads q5_0 rows)".to_string()]
        );
        let head = |ty| whole("output.weight", Role::Head, ty, "output head");
        for ty in [
            GgmlType::Q8_0,
            GgmlType::Q6_K,
            GgmlType::Q4_K,
            GgmlType::NVFP4,
        ] {
            assert_eq!(head(ty), Vec::<String>::new(), "{ty}");
        }
        assert_eq!(
            head(GgmlType::Q5_K),
            ["q5_K output head (the head reads q6_K, q4_K, q8_0 and nvfp4)".to_string()]
        );
        assert_eq!(
            embed(GgmlType::F32),
            ["f32 token embedding (the card reads q5_0 rows)".to_string()]
        );
        assert_eq!(
            head(GgmlType::F32),
            ["f32 output head (the head reads q6_K, q4_K, q8_0 and nvfp4)".to_string()]
        );
        let proj = |name: &str, role: Role| whole(name, role, GgmlType::F32, "(the body reads");
        assert_eq!(
            proj("blk.0.ssm_in.weight", Role::Attention),
            [
                "f32 Mamba-2 and attention projections (the body reads q5_0, q8_0, q4_K and q6_K)"
                    .to_string()
            ]
        );
        assert_eq!(
            proj("blk.2.ffn_up.weight", Role::DenseFfn),
            ["f32 dense FFN (the body reads q5_0, q8_0, q4_K and q6_K)".to_string()]
        );
        let vector =
            |name: &str, role: Role, ty: GgmlType| whole(name, role, ty, "reads f32 and bf16");
        for (name, role) in [
            ("blk.0.attn_norm.weight", Role::Attention),
            ("blk.0.ssm_a", Role::Attention),
            ("blk.0.ssm_d", Role::Attention),
            ("blk.0.ssm_dt.bias", Role::Attention),
            ("blk.0.ssm_conv1d.weight", Role::Attention),
            ("blk.0.ssm_conv1d.bias", Role::Attention),
            ("blk.0.ssm_norm.weight", Role::Attention),
            ("blk.2.attn_norm.weight", Role::FfnNorm),
            ("output_norm.weight", Role::Head),
        ] {
            for ty in [GgmlType::F32, GgmlType::BF16] {
                assert_eq!(vector(name, role, ty), Vec::<String>::new(), "{name} {ty}");
            }
            assert_eq!(vector(name, role, GgmlType::Q8_0).len(), 1, "{name} q8_0");
        }
    }

    /// glm5next's mixer matrices read the common dense entries by name
    /// ([`dense_projected`]: every dense row but Q5_0), the absorbed pair reads
    /// q8_0 alone (its per-head gemv has no other plane), and the
    /// hyper-connection fn reads q8_0 planes or BF16 widened to f32.
    // PIN(2026-10-26): the mixer pin widened from `[Q8_0]`; derivation: each
    // type is a launch of `site::ByName::gemv`, whose arms `gate_glm5next_e2e`
    // runs on the fixture variants that retype the mixers.
    #[test]
    fn the_glm5next_mixer_pins_read_the_dense_entries_by_name() {
        let of = |name: &str, role: Role, ty: GgmlType| {
            items(Program::Glm5nextBody, &matrix(name, role, ty))
        };
        for ty in [
            GgmlType::Q8_0,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::BF16,
        ] {
            for name in ["attn_q.weight", "attn_output.weight", "ssm_f_a.weight"] {
                assert_eq!(
                    of(name, Role::Attention, ty),
                    Vec::<String>::new(),
                    "{name} {ty}"
                );
            }
        }
        for ty in [
            GgmlType::Q5_0,
            GgmlType::Q2_K,
            GgmlType::IQ3_S,
            GgmlType::IQ4_NL,
            GgmlType::Q4_0,
        ] {
            assert_eq!(
                of("attn_q.weight", Role::Attention, ty),
                [format!(
                    "{ty} mixer matrices (the types dense_projected names)"
                )],
                "{ty}"
            );
        }
        let pair = "absorbed key and value heads (the per-head gemv reads q8_0)";
        for name in ["attn_k_b.weight", "attn_v_b.weight"] {
            assert_eq!(
                of(name, Role::Attention, GgmlType::Q8_0),
                Vec::<String>::new()
            );
            assert_eq!(
                of(name, Role::Attention, GgmlType::Q6_K),
                [format!("{} {pair}", GgmlType::Q6_K)]
            );
        }
        let sel = "selector query and head weights (its gemv reads q8_0, f32 or bf16)";
        for name in ["indexer.attn_q_b.weight", "indexer.proj.weight"] {
            for ty in [GgmlType::Q8_0, GgmlType::BF16] {
                assert_eq!(
                    of(name, Role::Attention, ty),
                    Vec::<String>::new(),
                    "{name} {ty}"
                );
            }
            assert_eq!(
                of(name, Role::Attention, GgmlType::Q4_K),
                [format!("{} {sel}", GgmlType::Q4_K)]
            );
        }
        let ape = "indexer_compressor_ape.weight";
        assert_eq!(
            of(ape, Role::Attention, GgmlType::BF16),
            Vec::<String>::new()
        );
        assert_eq!(
            of(ape, Role::Attention, GgmlType::Q4_K),
            [format!(
                "{} selector pool positions (read as f32 values)",
                GgmlType::Q4_K
            )]
        );
        for ty in [GgmlType::Q8_0, GgmlType::BF16] {
            assert_eq!(
                of("hc_attn_fn.weight", Role::HyperConnection, ty),
                Vec::<String>::new()
            );
        }
        assert_eq!(
            of("hc_attn_fn.weight", Role::HyperConnection, GgmlType::Q6_K).len(),
            1
        );
    }

    /// The dense block's pin (mimo2, glm5next) and glm5next's shared-expert pin
    /// read the gated block's types (q8_0 and the K-quants `enqueue_block`
    /// dispatches); any other type is an item naming the type.
    // PIN(2026-10-11): q4_K, q5_K and q6_K joined the three pins' set (it was
    // q8_0 alone) with the gated block's type dispatch; derivation: see
    // `GATED_BLOCK`.
    #[test]
    fn the_gated_block_pins_read_the_gated_blocks_types() {
        for (program, role, text) in [
            (
                Program::Mimo2Body,
                Role::DenseFfn,
                "dense block (the body reads q8_0, q4_K, q5_K and q6_K)",
            ),
            (
                Program::Glm5nextBody,
                Role::DenseFfn,
                "dense block (the body reads q8_0, q4_K, q5_K and q6_K)",
            ),
            (
                Program::Glm5nextBody,
                Role::SharedExpert,
                "shared expert (the body reads q8_0, q4_K, q5_K and q6_K)",
            ),
        ] {
            let of = |ty: GgmlType| items(program, &matrix("ffn_gate.weight", role, ty));
            for ty in [
                GgmlType::Q8_0,
                GgmlType::Q4_K,
                GgmlType::Q5_K,
                GgmlType::Q6_K,
            ] {
                assert_eq!(of(ty), Vec::<String>::new(), "{program:?} {role:?} {ty}");
            }
            for ty in [GgmlType::Q3_K, GgmlType::IQ4_XS, GgmlType::F16] {
                assert_eq!(
                    of(ty),
                    [format!("{ty} {text}")],
                    "{program:?} {role:?} {ty}"
                );
            }
        }
    }

    /// The glm5next, mimo2 and gemma-embedding2 embedding pins read the set
    /// the host row reader decodes for the table (`GgmlType::has_dequant`, which
    /// `EmbedTable::new` checks too), pinned here by type name; a type
    /// outside it is an item naming the type.
    // PIN(2026-10-11): q4_K, q5_K, q6_K and the other types `dequant_row`
    // decodes joined the embedding's set (it was q8_0 alone) so a file whose
    // `token_embd` is any of them loads; derivation: the row is dequantized
    // on the host, one row a step, so no card launch reads the type.
    #[test]
    fn the_embedding_pins_read_the_host_rows_set() {
        let embed = |program: Program, ty: GgmlType| {
            items(
                program,
                &ModelTensor {
                    name: "token_embd.weight".into(),
                    layer: None,
                    ..matrix("", Role::TokenEmbedding, ty)
                },
            )
        };
        for program in [
            Program::Glm5nextBody,
            Program::Mimo2Body,
            Program::Solar2Body,
            Program::GemmaEmbedding2Body,
        ] {
            for ty in [
                GgmlType::Q8_0,
                GgmlType::Q3_K,
                GgmlType::Q4_K,
                GgmlType::Q5_K,
                GgmlType::Q6_K,
                GgmlType::F16,
                GgmlType::IQ4_XS,
                GgmlType::IQ2_XXS,
                GgmlType::IQ2_S,
                GgmlType::IQ1_S,
                GgmlType::IQ1_M,
            ] {
                assert_eq!(embed(program, ty), Vec::<String>::new(), "{program:?} {ty}");
            }
            // PIN(2026-10-16): IQ2_XXS joined the embedding's set when `dequant_row` began to decode it.
            // PIN(2026-10-25): IQ1_S left the refused list and IQ2_S, IQ1_S and IQ1_M joined the embedding's set for the
            // same reason; q4_0 is a type `dequant_row` still refuses.
            for ty in [GgmlType::Q4_0, GgmlType::I8] {
                assert_eq!(
                    embed(program, ty),
                    [format!(
                        "{ty} token embedding (the host's rows: the types dequant_row decodes)"
                    )],
                    "{program:?} {ty}"
                );
            }
        }
    }

    // PIN(2026-10-22): MXFP8/FP8_B128 dense and Q8_0 indexer refusals were wrong: both walks launch them.
    // PIN(2026-10-26): the K-quants (Q3_K to Q6_K) joined the dense sites' set: both walks project a site by name
    // (`site::ByName`), the q8_1 quantization and the K-quant gemv by chunks on the ubatch walk. The shared expert's
    // down keeps the f32-row types: K = 640 is not whole super-blocks.
    // PIN(2026-10-27): IQ4_XS moved from refused to q8_1; it reads q8_1 through the selected one-expert dispatch,
    // and the shared down still refuses it because its K = 640 is not whole super-blocks.
    #[test]
    fn the_qwen38_dense_pins_read_the_projected_sites() {
        let q8_1 = [
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::IQ4_XS,
        ];
        let f32_rows = [
            GgmlType::Q8_0,
            GgmlType::BF16,
            GgmlType::MXFP8,
            GgmlType::FP8_B128,
            GgmlType::F32,
        ];
        let refused = [
            GgmlType::Q5_0,
            GgmlType::Q2_K,
            GgmlType::IQ3_S,
            GgmlType::IQ4_NL,
            GgmlType::Q4_0,
        ];
        for (name, role) in [
            ("attn_q.weight", Role::Attention),
            ("ssm_out.weight", Role::Attention),
            ("indexer.q_proj.weight", Role::Attention),
            ("indexer.k_proj.weight", Role::Attention),
            ("ffn_gate_shexp.weight", Role::SharedExpert),
            ("ffn_up_shexp.weight", Role::SharedExpert),
        ] {
            for ty in q8_1.into_iter().chain(f32_rows) {
                let got = items(Program::Qwen38Body, &matrix(name, role, ty));
                assert!(got.is_empty(), "{name} {ty}: {got:?}");
            }
            for ty in refused {
                let got = items(Program::Qwen38Body, &matrix(name, role, ty));
                assert_eq!(got.len(), 1, "{name} {ty}: {got:?}");
            }
        }
        // The shared expert's down is K = 640: the q8_1 types' rows are whole
        // super-blocks of 256, so only the f32-row types read it, and a type
        // no dense entry reads is refused by both pins.
        let down = "ffn_down_shexp.weight";
        for ty in f32_rows {
            let got = items(Program::Qwen38Body, &matrix(down, Role::SharedExpert, ty));
            assert!(got.is_empty(), "{down} {ty}: {got:?}");
        }
        for ty in q8_1 {
            let got = items(Program::Qwen38Body, &matrix(down, Role::SharedExpert, ty));
            assert_eq!(got.len(), 1, "{down} {ty}: {got:?}");
        }
        for ty in refused {
            let got = items(Program::Qwen38Body, &matrix(down, Role::SharedExpert, ty));
            assert_eq!(got.len(), 2, "{down} {ty}: {got:?}");
        }
    }

    /// The qwen4exp head and embedding read the card kernel table's rows
    /// ([`super::ROW_PINS`], `kernels::head` and `kernels::embed`, BF16's
    /// among them: that load keeps BF16 in its words): `output` of Q8_0, Q6_K,
    /// Q5_K, Q4_K or BF16 and a `token_embd` of Q3_K, Q4_K, Q5_K, Q6_K, Q8_0
    /// or BF16 are no item; another type is one that names the row.
    // PIN(2026-10-06): q6_K (and q4_K, which the same gemv reads) joined the
    // head's set so the UD-Q3_K_XL file's q6_K output loads.
    // PIN(2026-10-20): bf16 joined the head's and the embedding's sets with
    // the bf16 head gemv and embedding rows (`prepare --keep-bf16`).
    // PIN(2026-10-26): the head's set is the table's head rows (q5_K joined:
    // the `_sel` Q5_K gemv over the matrix as one expert, `Head::with_gain`)
    // and the embedding's the table's embedding rows (q3_K, q4_K, q5_K and
    // q6_K joined: `Q38Kernels::enqueue_embed_lookup` runs
    // `ElemKernels::enqueue_embed_rows_kquant`), so the unsloth UD-IQ1_S to
    // UD-IQ3_XXS files' K-quant embedding and the ISTA files' Q5_K head are no
    // items. Derivation: each type admitted has a launch the qwen4exp walks
    // run on every path that embeds (the step, the pass and the ubatch front)
    // or projects (the step's head); the old test asserted the two refusals
    // this one inverts.
    #[test]
    fn the_qwen38_head_and_embedding_read_the_kernel_tables_rows() {
        let mut bad = Vec::<String>::new();
        let mut check = |what: String, t: ModelTensor, want: &[String]| {
            let got = items(Program::Qwen38Body, &t);
            if got != want {
                bad.push(format!("{what}: items {got:?}, want {want:?}"));
            }
        };
        let none = Vec::<String>::new();
        for ty in [
            GgmlType::Q8_0,
            GgmlType::Q6_K,
            GgmlType::Q5_K,
            GgmlType::Q4_K,
            GgmlType::BF16,
        ] {
            check(
                format!("head {ty}"),
                global("output.weight", Role::Head, ty),
                &none,
            );
        }
        for ty in [GgmlType::Q3_K, GgmlType::IQ4_XS, GgmlType::Q5_0] {
            check(
                format!("head {ty}"),
                global("output.weight", Role::Head, ty),
                &[format!(
                    "{ty} output head (the types the card kernel table's head rows name)"
                )],
            );
        }
        for ty in [
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
            GgmlType::BF16,
        ] {
            check(
                format!("embedding {ty}"),
                global("token_embd.weight", Role::TokenEmbedding, ty),
                &none,
            );
        }
        for ty in [GgmlType::Q2_K, GgmlType::IQ4_XS, GgmlType::Q5_0] {
            check(
                format!("embedding {ty}"),
                global("token_embd.weight", Role::TokenEmbedding, ty),
                &[format!(
                    "{ty} token embedding (the types the card kernel table's embedding rows \
                     name)"
                )],
            );
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
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
            bf16_words: |_| false,
        };
        let chat = models::ChatSpec {
            pre: "qwen2".to_string(),
            template: Some("{{ messages }}".to_string()),
            tools: spec::TOOLS,
            reasoning: Some(models::ReasoningFormat::THINK),
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
    /// width no fused qdot kernel serves and whose type its card experts do
    /// not read, and a routed stack of a type with neither. The card rule
    /// passes an MXFP4 stack whatever its width, as the other families' card
    /// rules pass theirs: the rule reads the type alone, and the width the
    /// host cannot fuse is not listed (the card holds a prefix of the experts
    /// and the host serves the rest).
    // PIN(2026-10-10): the MXFP4 stack of the same width (before that round an
    // item, as every stack the host could not fuse was) lists nothing, which
    // the last assertion holds.
    // PIN(2026-10-11): card_runs now runs Q4_K/Q5_K through the common leg, so
    // the narrow-down case is a q6_K stack, which no card entry runs.
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

        let mut other = model;
        for t in &mut other.tensors {
            if t.name == "blk.3.attn_qkv.weight" {
                t.ty = GgmlType::Q4_K;
            }
            if t.name == "blk.4.ffn_down_exps.weight" {
                t.dims[0] = 2000;
            }
            // PIN(2026-10-25): iq2_s stood here until it had a card route; tq2_0 has none.
            if t.name == "blk.5.ffn_up_exps.weight" {
                // PIN(2026-10-25): IQ2_S is served by qdot's generic host dot now; q4_0 has no kernel on either side.
                t.ty = GgmlType::Q4_0;
            }
            if t.name == "blk.6.ffn_down_exps.weight" {
                t.ty = GgmlType::Q6_K;
                t.dims[0] = 2000;
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
                "q4_0 routed experts on a card",
                "q6_K routed experts on a card",
            ],
            "{items:?}"
        );
        assert!(
            !items.iter().any(|i| i.contains("mxfp4")),
            "the MXFP4 down of 2000 values lists nothing: {items:?}"
        );
    }
}
