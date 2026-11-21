//! The Hy4-flash reader's [`ModelSpec`]. Every layer is a [`Mixer::Latent`] of `head_count[l]`
//! heads: the 1024-value query latent, the 512-value cache latent beside 64 rotated key values
//! (the pairwise rope at the layer's base), absorbed 192-wide queries and 256-wide values, a sink
//! a head and an output gate that is a projection of the layer input. A sparse-attention layer
//! also selects: a [`Selector::TokenPool`] of [`PoolRule::Gated`] pools of `kpool` positions over
//! 16 index heads of 128 values; a window layer attends the last `window` positions. The residual
//! is [`Residual::HcSandwich`] over four identity-comb streams that the head folds
//! ([`HcKind::Identity`]); the first layer is a dense SwiGLU block and the rest the sigmoid router
//! over gate-first-clamped experts in a latent beside an unclamped shared expert; the layers named
//! `ple.layers` run an [`Extra::Engram`] site ([`NgramRule::EngramMasked`]). The next-token block
//! is checked by the header reader and not described: no draft runs in the first serve. The
//! program is the hy4 body (`crates/gpu-hy4`, [`Program::Hy4Body`]); what it runs that the common
//! table does not list is [`AVAILABLE`].

use gguf::Split;
use models::shape::rules;
use models::{
    Act, Arch, ChatSpec, Collapse, EngramSpec, Extra, Ffn, HcKind, HcSpec, Latent, LatentOut,
    LatentUp, LayerSpec, Mixer, ModelSpec, Moe, Need, NgramRule, OutGate, PoolRule,
    ReasoningFormat, Residual, Rope, RopeMode, Router, RouterNorm, Selector, Shared, Source,
    ToolFormat,
};

use super::hparams::Hparams;
use super::roles;
use crate::arch::coverage::{self, Available};
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::PlacementError;
use crate::placement::accepts::Program;

/// `tokenizer.ggml.pre`: the deepseek-v3 set of regexes under the name the PR's converter writes
/// (`crates/tokenizer/src/pretok.rs`).
pub const PRE: &str = "hy_v4";

/// The tool-call markup the template teaches (`serve::glmxml::KvSpelling::Hy4`, read back by the
/// server from the call the template renders: no tag is spelled whole in its source).
pub const TOOLS: Option<ToolFormat> = Some(ToolFormat::Hy4);

/// The reasoning span, as the template spells it (`'<think{}>'.format(HYTK)`): the generation
/// prompt ends with the open tag (or the closed pair under `reasoning_effort` `no_think`), so the
/// model starts inside the span or past it.
pub const REASONING: ReasoningFormat = ReasoningFormat::ThinkSpan {
    open: "<think:6124c78e>",
    close: "</think:6124c78e>",
};

/// What the hy4 body runs that the common table ([`coverage::AVAILABLE`]) does not list for it: one
/// row per composed owner, citing the constant or function that owns it. The widths are the
/// architecture's facts the body is built for (`gpu-hy4/src/body.rs` `Shape::of` refuses any
/// other); the fixture keeps every one of them but the expert and dense widths, which the leg and
/// the sites take at any multiple of their blocks.
pub const AVAILABLE: &[Available] = &[
    Available {
        programs: &[Program::Hy4Body],
        at: "gpu/src/flash.rs enqueue_flash_latent_sel_sink and _window_sink (MMA_WIDTH 576, value 512, \
             a sink a head; gpu-hy4/src/body.rs ROPE, LATENT), the absorbed q·k_b at 192 and v_b at 256",
        runs: |n| match n {
            Need::Latent {
                latent: 512,
                rope: 64,
                up: LatentUp::Absorbed { qk, v: 256 },
            } => *qk + 64 == 256,
            Need::LatentPe { width } => *width == 64,
            _ => false,
        },
    },
    Available {
        programs: &[Program::Hy4Body],
        at: "gpu/src/linear/out_gate.rs (the sigmoid gate projection of the layer input at 256 a head)",
        runs: |n| matches!(n, Need::LatentOutGate),
    },
    Available {
        programs: &[Program::Hy4Body],
        at: "gpu/src/latent.rs index_key_rms_append, index_pool_norm_rope (INDEX_HEAD, POOL), \
             gpu/src/kpool.rs kpool_score (heads a launch fact), gpu/src/qsa.rs (the forced tail)",
        runs: |n| {
            matches!(
                n,
                Need::GatedPool {
                    heads: 16,
                    d: 128,
                    pool: 4
                }
            )
        },
    },
    Available {
        programs: &[Program::Hy4Body],
        at: "gpu/src/route_body.rs sigmoid_512_16 (the biased sigmoid, the selected sum floored)",
        runs: |n| {
            matches!(n, Need::Router(s)
                if (s.experts, s.top_k) == (512, 16) && s.rule == rules::BIASED_SIGMOID_FLOOR)
        },
    },
    Available {
        programs: &[Program::Hy4Body],
        at: "gpu-hy4/src/body.rs Ffn::Routed (the latent pair `ffn_routed_down`, `ffn_routed_up` \
             around the card leg and the host tier at the latent's width)",
        runs: |n| matches!(n, Need::LatentExperts { latent } if latent.is_multiple_of(256)),
    },
    Available {
        programs: &[Program::Hy4Body],
        at: "gpu/src/kquant/act.rs Act::SwigluGateClamp (the card leg), qdot swiglu_gclamp (the host \
             tier; gpu-hy4/src/tier.rs tier_act, host_layer)",
        runs: |n| {
            matches!(
                n,
                Need::ExpertAct {
                    shared: false,
                    act: Act::SwiGluGateFirst { .. }
                }
            )
        },
    },
    Available {
        programs: &[Program::Hy4Body],
        at: "gpu-hy4/src/step.rs glu (the SiLU shared expert and the dense lead block over the \
             common dense sites)",
        runs: |n| coverage::ungated_shared(n) || matches!(n, Need::DenseFfn { .. }),
    },
    Available {
        programs: &[Program::Hy4Body],
        at: "gpu/src/hc_ident.rs (HC: the pre mix, the post and the head fold of four streams), \
             the sandwich norms of gpu-hy4/src/step.rs",
        runs: |n| {
            matches!(
                n,
                Need::HcIdentity {
                    streams: 4,
                    collapse: Collapse::Head
                } | Need::HcSandwich { .. }
            )
        },
    },
    Available {
        programs: &[Program::Hy4Body],
        at: "gpu/src/engram_delta.rs (deltas 1-4 and the mask) and the engram gate at ROW 3072",
        runs: |n| matches!(n, Need::EngramMasked { row: 3072 }),
    },
];

/// `split`'s description and roles, and the defaults the key read took.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = chat_of(split, TOOLS, Some(REASONING))?;
    if chat.pre != PRE {
        return Err(PlacementError::Metadata {
            key: "tokenizer.ggml.pre".to_string(),
            detail: format!("is {:?}; a Hy4-flash file's is {PRE:?}", chat.pre),
        });
    }
    let spec = spec_of(&hp, chat)?;
    Ok(Read {
        spec,
        tensors,
        defaults: hp.defaults,
    })
}

/// The pairwise rope over the `rope.dimension_count` rotated key values at `base`.
fn rope_of(hp: &Hparams, base: f32) -> Result<Rope, PlacementError> {
    Ok(Rope {
        mode: RopeMode::NormTail,
        dims: spec_u32("rope.dimension_count", hp.mla.pe)?,
        base,
        yarn: None,
        factors: false,
    })
}

/// Trunk layer `l`'s description.
fn layer_of(hp: &Hparams, l: usize) -> Result<LayerSpec, PlacementError> {
    let rope = rope_of(hp, hp.rope_base_of(l))?;
    let select = hp
        .dsa(l)
        .then(|| -> Result<Selector, PlacementError> {
            Ok(Selector::TokenPool {
                heads: spec_u32("attention.indexer.head_count", hp.indexer.heads)?,
                d: spec_u32("attention.indexer.key_length", hp.indexer.d)?,
                top_k: spec_u32("attention.indexer.top_k", hp.indexer.top_k)?,
                pool: spec_u32("attention.indexer.kpool", hp.indexer.kpool)?,
                rule: PoolRule::Gated {
                    key_eps: hp.rms_eps,
                    rope,
                },
            })
        })
        .transpose()?;
    let mixer = Mixer::Latent(Latent {
        heads: spec_u32("attention.head_count", hp.heads(l))?,
        q_lora: spec_u32("attention.q_lora_rank", hp.mla.q_lora)?,
        latent: spec_u32("attention.kv_lora_rank", hp.mla.kv_lora)?,
        up: LatentUp::Absorbed {
            qk: spec_u32("attention.key_length_mla", hp.mla.nope())?,
            v: spec_u32("attention.value_length_mla", hp.mla.head_v)?,
        },
        rope: Some(rope),
        q_head_norm: false,
        ssmax: false,
        pe: spec_u32("rope.dimension_count", hp.mla.pe)?,
        cache: Source::Own(()),
        out_gate: OutGate::Proj,
        out: LatentOut::Plain,
        window: if hp.dsa(l) {
            None
        } else {
            Some(spec_u32("attention.sliding_window", hp.window)?)
        },
        sinks: true,
        compress: None,
        select,
    });
    let plain = Act::SwiGlu { limit: None };
    let ffn = if hp.dense(l) {
        Ffn::Dense {
            ff: spec_u32("feed_forward_length", hp.dense_ff)?,
            act: plain,
        }
    } else {
        let routed = Act::SwiGluGateFirst { limit: hp.clamp[l] };
        let moe = hp.experts.moe(routed)?;
        Ffn::Moe(Moe {
            latent: Some(spec_u32("expert_latent_length", hp.latent)?),
            shared: Some(Shared {
                act: plain,
                ..moe.shared.expect("noaux routers carry a shared expert")
            }),
            // The file sets no `expert_weights_norm_eps`, so llama.cpp's graph divides the kept
            // weights by their sum clamped at 2^-14 (src/llama-graph.cpp, build_moe_ffn).
            router: Router {
                norm: match moe.router.norm {
                    RouterNorm::SelectedSum => RouterNorm::SelectedSumFloor,
                    other => other,
                },
                ..moe.router
            },
            ..moe
        })
    };
    Ok(LayerSpec {
        mixer,
        ffn,
        residual: Residual::HcSandwich { eps: hp.rms_eps },
        extras: if hp.site(l) {
            vec![Extra::Engram]
        } else {
            Vec::new()
        },
    })
}

/// The description of the file `hp` was read from, with the chat surface `chat`.
pub fn spec_of(hp: &Hparams, chat: ChatSpec) -> Result<ModelSpec, PlacementError> {
    let engram = hp
        .engram
        .as_ref()
        .map(|g| -> Result<EngramSpec, PlacementError> {
            Ok(EngramSpec {
                heads: spec_u32("ple.heads_per_ngram", g.heads)?,
                max_ngram: spec_u32("ple.ngram_size", g.ngram)?,
                key_length: spec_u32("embedding_length_per_layer_input", g.key_length)?,
                rule: NgramRule::EngramMasked,
            })
        });
    Ok(ModelSpec {
        arch: Arch::Hy4,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx_train)?,
        rms_eps: hp.rms_eps,
        layers: (0..hp.n_layer)
            .map(|l| layer_of(hp, l))
            .collect::<Result<Vec<_>, _>>()?,
        mtp: Vec::new(),
        hc: Some(HcSpec {
            streams: spec_u32("hyper_connection.count", hp.hc.streams)?,
            kind: HcKind::Identity {
                eps: hp.hc.eps,
                magnitude: hp.hc.magnitude,
                collapse: Collapse::Head,
            },
        }),
        engram: engram.transpose()?,
        encoder: None,
        logit_epilogue: None,
        chat,
        loop_: None,
    })
}
