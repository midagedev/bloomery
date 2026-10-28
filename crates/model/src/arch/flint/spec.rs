//! Header-only description; a program must separately bind the normalization
//! floor, activated-value scaling, head norms, sinks and actual quantized sites.

use super::{hparams::Hparams, roles};
use crate::arch::classify::{Counts, classify_layers};
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::PlacementError;
use gguf::Split;
use models::{
    Act, Arch, Ffn, Gqa, LayerSpec, Mixer, ModelSpec, Moe, OutGate, ReasoningFormat, Residual,
    Rope, RopeMode, Router, Score, Shared, ToolFormat,
};

/// Describe Flint's layers and every tensor, refusing inconsistent metadata,
/// missing/misplaced tensors, wrong dimensions and unclaimed names.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = classify_layers(split, Counts::new(hp.layers, hp.experts, hp.used), |l| {
        roles::groups(l >= hp.dense_lead).iter().copied()
    })?;
    let chat = chat_of(
        split,
        Some(ToolFormat::GlmXml),
        Some(ReasoningFormat::THINK),
    )?;
    let act = Act::SwiGlu { limit: None };
    let layers = (0..hp.layers)
        .map(|l| {
            let mixer = Mixer::Gqa(Gqa {
                heads: spec_u32("attention.head_count", hp.heads)?,
                kv_heads: spec_u32("attention.head_count_kv", hp.kv[l])?,
                head_dim: spec_u32("attention.key_length", hp.head)?,
                value_dim: spec_u32("attention.value_length", hp.head)?,
                rope: if hp.rope[l] {
                    Rope {
                        mode: RopeMode::Neox,
                        dims: spec_u32("attention.key_length", hp.head)?,
                        base: hp.base,
                        yarn: None,
                        factors: false,
                    }
                } else {
                    Rope {
                        mode: RopeMode::None,
                        dims: 0,
                        base: 0.0,
                        yarn: None,
                        factors: false,
                    }
                },
                qk_norm: true,
                o_norm: false,
                out_gate: OutGate::None,
                select: None,
                window: hp.swa[l]
                    .then(|| spec_u32("attention.sliding_window", hp.window))
                    .transpose()?,
                sinks: true,
                value_scale: None,
            });
            let ffn = if l < hp.dense_lead {
                Ffn::Dense {
                    ff: spec_u32("feed_forward_length", hp.ff[l])?,
                    act,
                }
            } else {
                Ffn::Moe(Moe {
                    experts: spec_u32("expert_count", hp.experts)?,
                    top_k: spec_u32("expert_used_count", hp.used)?,
                    expert_ff: spec_u32("expert_feed_forward_length", hp.expert_ff[l])?,
                    latent: None,
                    pre_exp_norm: false,
                    post_exp_norm: None,
                    act,
                    router: Router {
                        score: Score::Sigmoid,
                        bias: split
                            .find(&format!("blk.{l}.{}", roles::SELECTION_BIAS))
                            .is_some(),
                        norm: if hp.norm {
                            models::RouterNorm::SelectedSumFloor
                        } else {
                            models::RouterNorm::None
                        },
                        scale: hp.scale,
                        hash: false,
                    },
                    shared: Some(Shared {
                        ff: spec_u32("expert_shared_feed_forward_length", hp.shared_ff)?,
                        act,
                        sigmoid_gate: false,
                    }),
                })
            };
            Ok(LayerSpec {
                mixer,
                ffn,
                residual: Residual::Plain,
                extras: Vec::new(),
            })
        })
        .collect::<Result<Vec<_>, PlacementError>>()?;
    Ok(Read {
        spec: ModelSpec {
            arch: Arch::Flint,
            hidden: spec_u32("embedding_length", hp.hidden)?,
            vocab: spec_u32("vocab", hp.vocab)?,
            ctx_train: spec_u32("context_length", hp.context)?,
            rms_eps: hp.eps,
            layers,
            mtp: Vec::new(),
            hc: None,
            engram: None,
            encoder: None,
            logit_epilogue: None,
            chat,
        },
        tensors,
        defaults: hp.defaults,
    })
}
