//! The qwen35moe reader's [`ModelSpec`]. A GDN layer is a
//! [`Mixer::DeltaRule`] whose value head `j` reads key head `j mod k_heads`
//! (the converter tiles them, llama.cpp `convert_hf_to_gguf.py`); a GQA layer
//! rotates `rope.dimension_count` of its head values by IMROPE sections, and
//! its `attn_q` writes a per-head gate beside the query when it is twice the
//! query's width. The file folds the norms' `+1` offset into their gains, so
//! every norm is a plain RMS norm. No chat parser is bound for this family.

use std::collections::HashMap;

use gguf::Split;
use models::{
    Act, Arch, DeltaKind, DeltaRule, Ffn, Gqa, KHeadMap, LayerSpec, Mixer, ModelSpec, Moe,
    Residual, Rope, RopeMode, Router, Score, Shared,
};

use super::hparams::{Hparams, Kind};
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
    let layers = (0..hp.n_layer)
        .map(|l| {
            let mixer = match hp.kinds[l] {
                Kind::DeltaRule => Mixer::DeltaRule(DeltaRule {
                    kind: DeltaKind::Gdn {
                        khead_map: KHeadMap::Tiled,
                    },
                    k_heads: spec_u32("ssm.group_count", hp.k_heads)?,
                    v_heads: spec_u32("ssm.time_step_rank", hp.v_heads)?,
                    d: spec_u32("ssm.state_size", hp.state)?,
                    conv: spec_u32("ssm.conv_kernel", hp.conv)?,
                }),
                Kind::Attention => {
                    let q = format!("blk.{l}.attn_q.weight");
                    let rows = by_name.get(q.as_str()).and_then(|d| d.get(1)).copied();
                    let width = u64::from(heads) * u64::from(head_dim);
                    let out_gate = match rows {
                        Some(r) if r == 2 * width => true,
                        Some(r) if r == width => false,
                        _ => {
                            return Err(PlacementError::Tensor {
                                name: q,
                                detail: format!(
                                    "writes {rows:?} rows, not the query's {width} or twice it with the gate"
                                ),
                            });
                        }
                    };
                    Mixer::Gqa(Gqa {
                        heads,
                        kv_heads: spec_u32("attention.head_count_kv", hp.n_head_kv)?,
                        head_dim,
                        rope: Rope {
                            mode: RopeMode::Imrope {
                                sections: hp.rope_sections,
                            },
                            dims: spec_u32("rope.dimension_count", hp.rope_dims)?,
                            base: hp.rope_base,
                            yarn: None,
                        },
                        qk_norm: has(format!("blk.{l}.attn_q_norm.weight"))
                            && has(format!("blk.{l}.attn_k_norm.weight")),
                        out_gate,
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
                        key: "qwen35moe.expert_shared_feed_forward_length".to_string(),
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
                residual: Residual::Plain,
                extras: Vec::new(),
            })
        })
        .collect::<Result<Vec<_>, PlacementError>>()?;
    Ok(ModelSpec {
        arch: Arch::Qwen35Moe,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx_train)?,
        rms_eps: hp.rms_eps,
        layers,
        mtp: Vec::new(),
        hc: None,
        engram: None,
        chat,
    })
}
