//! The qwen35moe reader's [`ModelSpec`], for both variants. A GDN layer is a
//! [`Mixer::DeltaRule`] whose value head `j` reads key head `j mod k_heads`
//! (the converter tiles them, llama.cpp `convert_hf_to_gguf.py`); a GQA layer
//! rotates `rope.dimension_count` of its head values by IMROPE sections, and
//! its `attn_q` writes a per-head gate beside the query when it is twice the
//! query's width. The file folds the norms' `+1` offset into their gains, so
//! every norm is a plain RMS norm. No chat parser is bound for this family.
//!
//! What qwen4exp changes, by its architecture and not by a key (llama.cpp
//! `src/models/qwen4exp.cpp`): the GDN output gate is a sigmoid (:476-486);
//! every layer is wrapped in gated-residual hyper-connections; an attention
//! layer's positions come from a mean-pool top-k whose queries and pooled keys
//! take the layer's rope (:542-691); the PLE site's conv is dilated by the
//! n-gram size (:1246-1251).

use std::collections::HashMap;

use gguf::Split;
use models::{
    Act, Arch, DeltaKind, DeltaRule, EngramSpec, Extra, Ffn, GdnGate, Gqa, HcKind, HcSpec,
    KHeadMap, LayerSpec, Mixer, ModelSpec, Moe, NgramRule, PoolRule, Residual, Rope, RopeMode,
    Router, Score, Selector, Shared,
};

use super::hparams::{Hparams, Kind, Variant};
use super::roles;
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::{ModelTensors, PlacementError, Role};

/// `split`'s description and roles, and the defaults the key read took.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = chat_of(split, None, None)?;
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
    let by_name: HashMap<&str, &[u64]> = tensors
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t.dims.as_slice()))
        .collect();
    let has = |name: String| by_name.contains_key(name.as_str());
    let heads = spec_u32("attention.head_count", hp.n_head)?;
    let head_dim = spec_u32("attention.key_length", hp.head_dim)?;
    let arch = match hp.variant {
        Variant::Qwen35Moe => Arch::Qwen35Moe,
        Variant::Qwen4Exp => Arch::Qwen4Exp,
    };
    let exp = hp.exp.as_ref();
    let rope = Rope {
        mode: RopeMode::Imrope {
            sections: hp.rope_sections,
        },
        dims: spec_u32("rope.dimension_count", hp.rope_dims)?,
        base: hp.rope_base,
        yarn: None,
    };
    let layers = (0..hp.n_layer)
        .map(|l| {
            let mixer = match hp.kinds[l] {
                Kind::DeltaRule => Mixer::DeltaRule(DeltaRule {
                    kind: DeltaKind::Gdn {
                        khead_map: KHeadMap::Tiled,
                        gate: match hp.variant {
                            Variant::Qwen35Moe => GdnGate::Silu,
                            Variant::Qwen4Exp => GdnGate::Sigmoid,
                        },
                    },
                    k_heads: spec_u32("ssm.group_count", hp.k_heads)?,
                    v_heads: spec_u32("ssm.time_step_rank", hp.v_heads)?,
                    d: spec_u32("ssm.state_size", hp.state)?,
                    conv: spec_u32("ssm.conv_kernel", hp.conv)?,
                }),
                Kind::Attention => {
                    let q = format!("blk.{l}.attn_q.weight");
                    let rows = by_name.get(q.as_str()).and_then(|d| d.get(1)).copied();
                    let out_gate = out_gate(q, rows, heads, head_dim)?;
                    Mixer::Gqa(Gqa {
                        heads,
                        kv_heads: spec_u32("attention.head_count_kv", hp.n_head_kv)?,
                        head_dim,
                        rope,
                        qk_norm: has(format!("blk.{l}.attn_q_norm.weight"))
                            && has(format!("blk.{l}.attn_k_norm.weight")),
                        out_gate,
                        select: match exp {
                            None => None,
                            Some(e) => Some(Selector::TokenPool {
                                heads: spec_u32("attention.indexer.head_count", e.idx_heads)?,
                                d: spec_u32("attention.indexer.key_length", e.idx_dim)?,
                                top_k: spec_u32("attention.indexer.top_k", e.idx_top_k)?,
                                pool: spec_u32("attention.compress_ratios", e.ratios[l])?,
                                rule: PoolRule::Mean { rope },
                            }),
                        },
                    })
                }
            };
            let shexp = tensors
                .tensors
                .iter()
                .any(|t| t.layer == Some(l) && t.role == Role::SharedExpert);
            let shared = match (shexp, hp.shared_ff) {
                (false, _) => None,
                (true, Some(ff)) => Some(Shared {
                    ff: spec_u32("expert_shared_feed_forward_length", ff)?,
                    act: Act::SwiGlu { limit: None },
                    sigmoid_gate: has(format!("blk.{l}.ffn_gate_inp_shexp.weight")),
                }),
                (true, None) => {
                    return Err(PlacementError::Metadata {
                        key: format!("{}.expert_shared_feed_forward_length", arch.name()),
                        detail: format!("is absent, and layer {l} carries a shared expert"),
                    });
                }
            };
            Ok(LayerSpec {
                mixer,
                ffn: Ffn::Moe(Moe {
                    experts: spec_u32("expert_count", hp.n_expert)?,
                    top_k: spec_u32("expert_used_count", hp.n_used)?,
                    expert_ff: spec_u32("expert_feed_forward_length", hp.expert_ff)?,
                    act: Act::SwiGlu { limit: None },
                    router: Router {
                        score: Score::Softmax,
                        bias: false,
                        norm: true,
                        scale: 1.0,
                        hash: false,
                    },
                    shared,
                }),
                residual: if exp.is_some() {
                    Residual::Hc
                } else {
                    Residual::Plain
                },
                extras: if exp.and_then(|e| e.ple).is_some_and(|p| p.layer == l) {
                    vec![Extra::Ple]
                } else {
                    Vec::new()
                },
            })
        })
        .collect::<Result<Vec<_>, PlacementError>>()?;
    let hc = match exp {
        None => None,
        Some(e) => Some(HcSpec {
            streams: spec_u32("hyper_connection.count", e.hc_streams)?,
            kind: HcKind::Gated {
                rank: spec_u32("hyper_connection.low_rank", e.hc_rank)?,
            },
        }),
    };
    let engram = match exp.and_then(|e| e.ple) {
        None => None,
        Some(p) => Some(EngramSpec {
            heads: spec_u32("ple.heads_per_ngram", p.heads_per_ngram)?,
            max_ngram: spec_u32("ple.ngram_size", p.ngram)?,
            key_length: spec_u32("embedding_length_per_layer_input", p.row)?,
            rule: NgramRule::Ple {
                eos: p.eos,
                image: p.image,
                conv: spec_u32("ple.conv_kernel", p.conv)?,
                dilation: spec_u32("ple.ngram_size", p.ngram)?,
            },
        }),
    };
    Ok(ModelSpec {
        arch,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx_train)?,
        rms_eps: hp.rms_eps,
        layers,
        mtp: Vec::new(),
        hc,
        engram,
        chat,
    })
}

/// Whether an attention layer's `attn_q` (`q`, writing `rows`) writes a
/// per-head gate beside the query: twice the query's width with it, once
/// without, any other width refused.
pub(super) fn out_gate(
    q: String,
    rows: Option<u64>,
    heads: u32,
    head_dim: u32,
) -> Result<bool, PlacementError> {
    let width = u64::from(heads) * u64::from(head_dim);
    match rows {
        Some(r) if r == 2 * width => Ok(true),
        Some(r) if r == width => Ok(false),
        _ => Err(PlacementError::Tensor {
            name: q,
            detail: format!(
                "writes {rows:?} rows, not the query's {width} or twice it with the gate"
            ),
        }),
    }
}
