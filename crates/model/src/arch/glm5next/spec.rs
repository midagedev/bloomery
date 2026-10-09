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
//! routed one. The tool calls are [`TOOLS`], the `<think>` span is read.

use gguf::Split;
use models::{
    Act, Arch, Collapse, DeltaKind, DeltaRule, Ffn, HcKind, HcMix, HcSpec, Latent, LatentOut,
    LatentUp, LayerSpec, Mixer, ModelSpec, Moe, PoolRule, ReasoningFormat, Residual, Router, Score,
    Selector, Shared, ToolFormat,
};

use super::hparams::{Hparams, Kind};
use super::roles;
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::{ModelTensors, PlacementError};

/// The tool-call markup the family's template teaches: `<tool_call>` with
/// `<arg_key>` and `<arg_value>`.
pub const TOOLS: Option<ToolFormat> = Some(ToolFormat::GlmXml);

/// `split`'s description and roles, and the defaults the key read took.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = chat_of(split, TOOLS, Some(ReasoningFormat::ThinkSpan))?;
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
        let name = format!("blk.{l}.{}", roles::SELECTION_BIAS);
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
                    rule: PoolRule::Learned {
                        key_eps: hp.norm_eps,
                    },
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
            kind: HcKind::Mhc {
                sinkhorn: spec_u32("hyper_connection.sinkhorn_iterations", hp.hc.sinkhorn)?,
                eps: hp.hc.eps,
                mix: HcMix::Own,
                collapse: Collapse::Mean,
            },
        }),
        engram: None,
        chat,
    })
}

#[cfg(test)]
mod tests {
    use models::Ffn;

    use super::super::hparams::tests::{keys, shaped, tensors};
    use super::super::roles::SELECTION_BIAS;
    use crate::arch::synthetic::{V, header_shaped};
    use crate::arch::{GLM5_NEXT_LLAMA_CPP, coverage};

    /// The small header of the `hparams` tests, with `tokenizer.ggml.pre`
    /// `glm4` and a chat template, and the tensors `keep` keeps; its read
    /// and the coverage check's items.
    fn read(tag: &str, keep: impl Fn(&str) -> bool) -> (super::Read, Vec<String>) {
        let tensors: Vec<String> = tensors().into_iter().filter(|t| keep(t)).collect();
        let global = [
            ("tokenizer.ggml.pre", V::Str("glm4")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let path = header_shaped(tag, "glm5next", &keys(), &global, &shaped(&tensors));
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = super::read(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        let read = read.expect("the header reads");
        let items = coverage::check(&read.spec, &read.tensors)
            .into_iter()
            .map(|u| u.feature)
            .collect();
        (read, items)
    }

    /// A GLM file's chat surface is one the tree runs: the tokenizer's
    /// `glm4` pre-tokenizer and the `<tool_call>` parser the family declares
    /// are not coverage items.
    #[test]
    fn the_chat_surface_is_covered() {
        let (read, items) = read("glm5next-chat", |_| true);
        assert_eq!(read.spec.chat.tools, super::TOOLS);
        for covered in ["pre-tokenizer glm4", "a tool-call parser for this template"] {
            assert!(
                !items.iter().any(|f| f == covered),
                "{covered:?} listed: {items:?}"
            );
        }
    }

    /// A file that spells the architecture as llama.cpp does, with the share
    /// key its converter writes, is the same model: the description and the
    /// defaults are the glm5next file's, and the reader lists no key unread.
    #[test]
    fn the_llama_cpp_spelling_reads_as_the_same_model() {
        let (base, _) = read("glm5next-spelling-base", |_| true);
        let names = tensors();
        let global = [
            ("tokenizer.ggml.pre", V::Str("glm4")),
            ("tokenizer.chat_template", V::Str("{{ messages }}")),
        ];
        let mut kv = keys();
        kv.push(("attention.indexer.index_share_mtp", V::Bool(true)));
        let path = header_shaped(
            "glm5-next-spelling",
            GLM5_NEXT_LLAMA_CPP,
            &kv,
            &global,
            &shaped(&names),
        );
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let read = crate::arch::spec(&split).map_err(|e| e.to_string());
        let unread = super::super::hparams::unread_keys(&split);
        let _ = std::fs::remove_file(&path);
        let read = read.expect("the header reads");
        assert_eq!(read.spec, base.spec);
        assert_eq!(read.defaults, base.defaults);
        assert!(unread.is_empty(), "{unread:?}");
    }

    /// The selection bias is optional, as ik loads it: a file without it
    /// reads, and its routers run with none.
    #[test]
    fn the_selection_bias_is_optional() {
        let bias = |r: &super::Read| -> Vec<bool> {
            r.spec
                .layers
                .iter()
                .chain(&r.spec.mtp)
                .filter_map(|l| match &l.ffn {
                    Ffn::Moe(m) => Some(m.router.bias),
                    _ => None,
                })
                .collect()
        };
        let (with, _) = read("glm5next-bias", |_| true);
        let (without, _) = read("glm5next-nobias", |t| !t.ends_with(SELECTION_BIAS));
        assert_eq!(bias(&with), [true; 4]);
        assert_eq!(bias(&without), [false; 4]);
    }
}
