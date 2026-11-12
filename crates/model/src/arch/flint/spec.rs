//! Header-only description; a program must separately bind the normalization
//! policy, activated-value scaling, head norms, sinks and actual quantized sites.

use super::{hparams::Hparams, roles};
use crate::arch::rules::ChatParsers;
use crate::arch::{Read, chat_from_template, chat_of, spec_u32};
use crate::placement::PlacementError;
use gguf::Split;
use models::{
    Ffn, Gqa, LayerSpec, Mixer, ModelSpec, Moe, OutGate, Residual, Rope, Router, Score, Shared,
};

/// Describe the layers and every tensor of an architecture this reader serves
/// (`flint`, `exaone-moe`), refusing inconsistent metadata, missing/misplaced
/// tensors, wrong dimensions and unclaimed names. The facts no GGUF key states
/// come from the architecture's description row (`crate::arch::rules`), each
/// recorded in `defaults` with its source.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = match hp.rules.chat {
        ChatParsers::Fixed { tools, reasoning } => chat_of(split, tools, reasoning)?,
        ChatParsers::FromTemplate => chat_from_template(split)?,
    };
    let layer = |l: usize| -> Result<LayerSpec, PlacementError> {
        let mixer = Mixer::Gqa(Gqa {
            heads: spec_u32("attention.head_count", hp.heads)?,
            kv_heads: spec_u32("attention.head_count_kv", hp.kv[l])?,
            head_dim: spec_u32("attention.key_length", hp.head)?,
            value_dim: spec_u32("attention.value_length", hp.head)?,
            rope: if hp.rope[l] {
                Rope {
                    mode: hp.rules.pairs,
                    dims: spec_u32("attention.key_length", hp.head)?,
                    base: hp.base,
                    yarn: None,
                    factors: false,
                }
            } else {
                Rope {
                    mode: models::RopeMode::None,
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
            window: hp.window[l],
            sinks: hp.sinks,
            value_scale: None,
        });
        let ffn = if hp.dense(l) {
            Ffn::Dense {
                ff: spec_u32("feed_forward_length", hp.ff[l])?,
                act: hp.rules.act(hp.limit_dense[l]),
            }
        } else {
            Ffn::Moe(Moe {
                experts: spec_u32("expert_count", hp.experts)?,
                top_k: spec_u32("expert_used_count", hp.used)?,
                expert_ff: spec_u32("expert_feed_forward_length", hp.expert_ff[l])?,
                latent: None,
                pre_exp_norm: false,
                post_exp_norm: None,
                sum_norm: false,
                act: hp.rules.act(hp.limit_exp[l]),
                router: Router {
                    score: Score::Sigmoid,
                    bias: split
                        .find(&format!("blk.{l}.{}", roles::SELECTION_BIAS))
                        .is_some(),
                    norm: if hp.norm {
                        hp.rules.router_norm
                    } else {
                        models::RouterNorm::None
                    },
                    scale: hp.scale,
                    hash: false,
                },
                shared: Some(Shared {
                    ff: spec_u32("expert_shared_feed_forward_length", hp.shared_ff)?,
                    act: hp.rules.act(hp.limit_shexp[l]),
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
    };
    let layers = (0..hp.trunk).map(layer).collect::<Result<Vec<_>, _>>()?;
    let mtp = if hp.mtp == 0 {
        Vec::new()
    } else {
        (hp.trunk..hp.blocks)
            .map(layer)
            .collect::<Result<Vec<_>, _>>()?
    };
    Ok(Read {
        spec: ModelSpec {
            arch: hp.arch,
            hidden: spec_u32("embedding_length", hp.hidden)?,
            vocab: spec_u32("vocab", hp.vocab)?,
            ctx_train: spec_u32("context_length", hp.context)?,
            rms_eps: hp.eps,
            layers,
            mtp,
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
