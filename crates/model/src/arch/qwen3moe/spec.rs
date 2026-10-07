//! The qwen3moe reader's [`ModelSpec`]: the file's [`Hparams`] and tensor
//! roles as the typed model description. Every layer is a GQA layer with the
//! per-head q/k norm and a routed mixture without a shared expert, on a plain
//! residual; the router's score, renormalization and scale are the
//! architecture's constants [`Hparams::read`] already holds the file to.

use std::collections::HashSet;

use gguf::Split;
use models::{
    Act, Arch, ChatSpec, Ffn, Gqa, LayerSpec, Mixer, ModelSpec, Moe, ReasoningFormat, Residual,
    Rope, RopeMode, Router, Score, ToolFormat,
};

use super::hparams::{Hparams, RopeMode as HpRopeMode, Score as HpScore};
use super::{names, roles};
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::{ModelTensors, PlacementError};

/// `split`'s description and roles: [`Hparams::read`], then
/// [`roles::classify`], then [`spec_of`]. The chat surface is the one the
/// server applies to every model today (the DSML tool calls and the
/// `<think>` span).
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = chat_of(
        split,
        Some(ToolFormat::Dsml),
        Some(ReasoningFormat::ThinkSpan),
    )?;
    let spec = spec_of(&hp, &tensors, chat)?;
    Ok(Read {
        spec,
        tensors,
        defaults: Vec::new(),
    })
}

/// The description of the file `hp` and `tensors` were read from, with the
/// chat surface `chat`.
pub fn spec_of(
    hp: &Hparams,
    tensors: &ModelTensors,
    chat: ChatSpec,
) -> Result<ModelSpec, PlacementError> {
    let has: HashSet<&str> = tensors.tensors.iter().map(|t| t.name.as_str()).collect();
    let rope = Rope {
        mode: match hp.rope.mode {
            HpRopeMode::Neox => RopeMode::Neox,
        },
        dims: spec_u32("rope.dimension_count", hp.rope.dims)?,
        base: hp.rope.base,
        yarn: None,
    };
    let e = &hp.experts;
    let layers = (0..hp.n_layer)
        .map(|l| {
            Ok(LayerSpec {
                mixer: Mixer::Gqa(Gqa {
                    heads: spec_u32("attention.head_count", hp.n_head)?,
                    kv_heads: spec_u32("attention.head_count_kv", hp.n_head_kv)?,
                    head_dim: spec_u32("attention.key_length", hp.head_dim)?,
                    // The reader holds the file to one width for q, k and v.
                    value_dim: spec_u32("attention.key_length", hp.head_dim)?,
                    rope,
                    qk_norm: has.contains(names::attn_q_norm(l).as_str())
                        && has.contains(names::attn_k_norm(l).as_str()),
                    out_gate: false,
                    select: None,
                    window: None,
                    sinks: false,
                    value_scale: None,
                }),
                ffn: Ffn::Moe(Moe {
                    experts: spec_u32("expert_count", e.n_expert)?,
                    top_k: spec_u32("expert_used_count", e.n_used)?,
                    expert_ff: spec_u32("expert_feed_forward_length", e.ff)?,
                    act: Act::SwiGlu { limit: None },
                    router: Router {
                        score: match e.score {
                            HpScore::Softmax => Score::Softmax,
                        },
                        bias: false,
                        norm: e.weights_norm,
                        scale: 1.0,
                        hash: false,
                    },
                    shared: None,
                }),
                residual: Residual::Plain,
                extras: Vec::new(),
            })
        })
        .collect::<Result<Vec<_>, PlacementError>>()?;
    Ok(ModelSpec {
        arch: Arch::Qwen3Moe,
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
