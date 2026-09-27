//! The glm5next reader's [`ModelSpec`]. A KDA layer is a
//! [`Mixer::DeltaRule`] of `attention.head_count` key and value heads (ik's
//! KDA layer passes the head count for both, `src/llama-kda.cpp`); a latent
//! layer caches `kv_lora_rank` values a position, absorbs the per-head up
//! projections (`attn_k_b`, `attn_v_b`), rotates nothing and picks its
//! positions by a token-pool indexer whose keys a biased LayerNorm normalizes
//! (`src/graphs/build_glm5next.cpp`). Every trunk block owns its
//! hyper-connection mix; the streams collapse to their mean before the head.
//! A dense layer and a shared expert take the shared SwiGLU limit
//! (`llm_build_ffn` in `src/llama-build-context.cpp`), the routed experts the
//! routed one. The tool-call syntax (`<tool_call>` with `<arg_key>` and
//! `<arg_value>`) has no parser here; the `<think>` span is read.

use gguf::Split;
use models::{
    Act, Arch, Collapse, DeltaKind, DeltaRule, Ffn, HcMix, HcSpec, Latent, LatentOut, LatentUp,
    LayerSpec, Mixer, ModelSpec, Moe, ReasoningFormat, Residual, Router, Score, Selector, Shared,
};

use super::hparams::{Hparams, Kind};
use super::roles;
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::{ModelTensors, PlacementError};

/// `split`'s description and roles, and the defaults the key read took.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = chat_of(split, None, Some(ReasoningFormat::ThinkSpan))?;
    let spec = spec_of(&hp, &tensors, chat)?;
    Ok(Read {
        spec,
        tensors,
        defaults: hp.defaults,
    })
}

/// The description of the file `hp` and `tensors` were read from.
pub fn spec_of(
    hp: &Hparams,
    tensors: &ModelTensors,
    chat: models::ChatSpec,
) -> Result<ModelSpec, PlacementError> {
    let bias = |l: usize| {
        let name = format!("blk.{l}.exp_probs_b.bias");
        tensors.tensors.iter().any(|t| t.name == name)
    };
    let layer = |l: usize| -> Result<LayerSpec, PlacementError> {
        let mixer = match hp.kinds[l] {
            Kind::Kda => Mixer::DeltaRule(DeltaRule {
                kind: DeltaKind::Kda {
                    gate_lower_bound: hp.gate_lower_bound,
                },
                k_heads: spec_u32("attention.head_count", hp.n_head)?,
                v_heads: spec_u32("attention.head_count", hp.n_head)?,
                d: spec_u32("kda.head_dim", hp.kda_head_dim)?,
                conv: spec_u32("ssm.conv_kernel", hp.conv)?,
            }),
            Kind::Latent => Mixer::Latent(Latent {
                heads: spec_u32("attention.head_count", hp.n_head)?,
                q_lora: spec_u32("attention.q_lora_rank", hp.q_lora)?,
                latent: spec_u32("attention.kv_lora_rank", hp.kv_lora)?,
                up: LatentUp::Absorbed {
                    qk: spec_u32("attention.key_length_mla", hp.head_k)?,
                    v: spec_u32("attention.value_length_mla", hp.head_v)?,
                },
                rope: None,
                q_head_norm: false,
                out: LatentOut::Plain,
                window: None,
                sinks: false,
                compress: None,
                select: Some(Selector::TokenPool {
                    heads: spec_u32("attention.indexer.head_count", hp.indexer.n_head)?,
                    d: spec_u32("attention.indexer.key_length", hp.indexer.head_dim)?,
                    top_k: spec_u32("attention.indexer.top_k", hp.indexer.top_k)?,
                    pool: spec_u32("attention.indexer.kpool", hp.indexer.kpool)?,
                    key_eps: hp.norm_eps,
                }),
            }),
        };
        let shared_act = Act::SwiGlu {
            limit: Some(hp.limit_shexp[l]),
        };
        let ffn = if l < hp.dense_lead {
            Ffn::Dense {
                ff: spec_u32("feed_forward_length", hp.dense_ff)?,
                act: shared_act,
            }
        } else {
            Ffn::Moe(Moe {
                experts: spec_u32("expert_count", hp.n_expert)?,
                top_k: spec_u32("expert_used_count", hp.n_used)?,
                expert_ff: spec_u32("expert_feed_forward_length", hp.expert_ff)?,
                act: Act::SwiGlu {
                    limit: Some(hp.limit_exp[l]),
                },
                router: Router {
                    score: Score::Sigmoid,
                    bias: bias(l),
                    norm: hp.weights_norm,
                    scale: hp.weights_scale,
                    hash: false,
                },
                shared: match hp.n_shared {
                    0 => None,
                    n => Some(Shared {
                        ff: spec_u32(
                            "expert_shared_count x expert_shared_feed_forward_length",
                            n.checked_mul(hp.shared_ff).ok_or_else(|| {
                                PlacementError::Metadata {
                                    key: "expert_shared_count".to_string(),
                                    detail: format!("{n} x {} overflows usize", hp.shared_ff),
                                }
                            })?,
                        )?,
                        act: shared_act,
                        sigmoid_gate: false,
                    }),
                },
            })
        };
        Ok(LayerSpec {
            mixer,
            ffn,
            residual: if l < hp.n_trunk {
                Residual::Hc
            } else {
                Residual::Plain
            },
            extras: Vec::new(),
        })
    };
    let layers = (0..hp.n_trunk).map(layer).collect::<Result<Vec<_>, _>>()?;
    let mtp = (hp.n_trunk..hp.n_layer)
        .map(layer)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ModelSpec {
        arch: Arch::Glm5Next,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx_train)?,
        rms_eps: hp.rms_eps,
        layers,
        mtp,
        hc: Some(HcSpec {
            streams: spec_u32("hyper_connection.count", hp.hc.streams)?,
            sinkhorn: spec_u32("hyper_connection.sinkhorn_iterations", hp.hc.sinkhorn)?,
            eps: hp.hc.eps,
            mix: HcMix::Own,
            collapse: Collapse::Mean,
        }),
        engram: None,
        chat,
    })
}
