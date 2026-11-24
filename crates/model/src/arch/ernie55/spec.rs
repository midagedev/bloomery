//! The ERNIE 5.5 reader's [`ModelSpec`]. An HCA layer is a [`Mixer::Latent`] of 64 heads over one
//! 512-value key-value row (the pairwise rope at the pooled rows' base, YaRN on), a sliding window
//! beside the pooled stream of one row every `ratio` positions, a sink a head, the grouped output
//! projection and an output gate that is a projection of the normed query latent
//! ([`OutGate::QLatent`], between the grouped projection and the final one). A latent layer is the
//! absorbed MLA row (the latent beside 64 rotated key values at the plain base) with the DSA
//! indexer's [`Selector::StreamTopK`] over keys from the layer input, a sink and the same gate on
//! the heads' outputs. The residual is [`Residual::Hc`] over four manifold-constrained streams
//! whose RMS factor is `1 / (sqrt(mean) + eps)` ([`HcKind::MhcRoot`]), each sublayer its own mix,
//! the head folding them with its trained mix. The first layer is a dense `situ` block and the
//! rest the sigmoid router over `situ` experts in a latent, the routed sum normed before its up
//! projection, beside a shared expert.
//!
//! What the file asks beyond [`ModelSpec`]'s fields is in [`super::ops`]; no program runs the
//! family, so [`AVAILABLE`] names none and every need of a description is an item of the coverage
//! check.

use gguf::Split;
use models::{
    Arch, ChatSpec, Collapse, Compress, Compressor, Ffn, HcKind, HcMix, HcSpec, IndexKeys, Latent,
    LatentOut, LayerSpec, Mixer, ModelSpec, OutGate, Residual, Rope, RopeMode, Selector, Source,
    Yarn,
};

use super::hparams::Hparams;
use super::roles;
use crate::arch::coverage::Available;
use crate::arch::{Read, chat_of, rules_of, spec_u32};
use crate::placement::PlacementError;

/// The indexer key's LayerNorm epsilon: not a header key, the family's code states it.
pub const INDEX_KEY_NORM_EPS: f32 = 1e-6;

/// What a program of this family runs that the common table does not list: nothing, because no
/// program runs the family yet.
pub const AVAILABLE: &[Available] = &[];

/// `split`'s description and roles, and the defaults the key read took and the description row's
/// facts.
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let chat = chat_of(split, None, None)?;
    let rules = rules_of(split)?;
    let spec = spec_of(&hp, chat)?;
    let mut defaults = hp.defaults;
    defaults.push(rules.router_norm_line());
    defaults.extend(rules.facts_lines());
    Ok(Read {
        spec,
        tensors,
        defaults,
    })
}

/// The description of the file `hp` was read from, with the chat surface `chat`.
pub fn spec_of(hp: &Hparams, chat: ChatSpec) -> Result<ModelSpec, PlacementError> {
    let act = hp.situ.act();
    let layer = |l: usize| -> Result<LayerSpec, PlacementError> {
        let mixer = Mixer::Latent(if hp.latent_layer(l) {
            latent_layer(hp)?
        } else {
            hca_layer(hp)?
        });
        let ffn = if hp.dense(l) {
            Ffn::Dense {
                ff: spec_u32("feed_forward_length", hp.dense_ff)?,
                act,
            }
        } else {
            let mut moe = hp.experts.moe(act)?;
            moe.latent = Some(spec_u32("expert_latent_length", hp.latent)?);
            moe.sum_norm = true;
            // The factor on the kept weights is the learned per-expert scale, an op of this
            // family ([`super::ops`]); the header's scale only initialises it.
            moe.router.scale = 1.0;
            Ffn::Moe(moe)
        };
        Ok(LayerSpec {
            mixer,
            ffn,
            residual: Residual::Hc,
            extras: Vec::new(),
        })
    };
    Ok(ModelSpec {
        arch: Arch::Ernie55,
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx_train)?,
        rms_eps: hp.rms_eps,
        layers: (0..hp.n_layer).map(layer).collect::<Result<Vec<_>, _>>()?,
        mtp: Vec::new(),
        hc: Some(HcSpec {
            streams: spec_u32("hyper_connection.count", hp.hc.streams)?,
            kind: HcKind::MhcRoot {
                sinkhorn: spec_u32("hyper_connection.sinkhorn_iterations", hp.hc.sinkhorn_iters)?,
                eps: hp.hc.eps,
                mix: HcMix::Own,
                collapse: Collapse::Head,
            },
        }),
        engram: None,
        encoder: None,
        logit_epilogue: None,
        chat,
        loop_: None,
    })
}

/// An HCA layer: the V4 compressed attention, gated and with a sink.
fn hca_layer(hp: &Hparams) -> Result<Latent, PlacementError> {
    let c = &hp.hca;
    Ok(Latent {
        heads: spec_u32("attention.head_count", hp.heads)?,
        q_lora: spec_u32("attention.q_lora_rank", hp.mla.q_lora)?,
        latent: spec_u32("attention.hca.head_length", c.head)?,
        up: models::LatentUp::KeqV,
        rope: Some(Rope {
            mode: RopeMode::NormTail,
            dims: spec_u32("rope.dimension_count", hp.mla.pe)?,
            base: hp.rope.compress_base,
            // The V4 attention's YaRN: no magnitude scaling, net 1.
            yarn: Some(Yarn {
                factor: hp.rope.factor,
                orig_ctx: spec_u32("rope.scaling.original_context_length", hp.rope.original_ctx)?,
                beta_fast: hp.rope.beta_fast,
                beta_slow: hp.rope.beta_slow,
                magnitude: 1.0,
            }),
            factors: false,
        }),
        q_head_norm: true,
        ssmax: false,
        pe: 0,
        cache: Source::Own(()),
        out_gate: OutGate::QLatent,
        out: LatentOut::Grouped {
            groups: spec_u32("attention.output_group_count", c.o_groups)?,
            rank: spec_u32("attention.output_lora_rank", c.o_lora)?,
        },
        window: Some(spec_u32("attention.sliding_window", c.window)?),
        sinks: true,
        compress: Some(Compress {
            ratio: spec_u32("attention.compress_ratios", c.ratio)?,
            rows: Source::Own(Compressor {
                gated: true,
                ape: true,
                overlap: false,
            }),
        }),
        select: None,
    })
}

/// A latent layer: the absorbed MLA row with the DSA indexer over keys from the layer input.
fn latent_layer(hp: &Hparams) -> Result<Latent, PlacementError> {
    let mut latent = hp.mla.latent(hp.heads)?;
    latent.pe = 0;
    latent.rope = Some(Rope {
        mode: RopeMode::NormTail,
        dims: spec_u32("rope.dimension_count", hp.mla.pe)?,
        base: hp.rope.base,
        yarn: None,
        factors: false,
    });
    latent.out_gate = OutGate::QLatent;
    latent.sinks = true;
    latent.select = Some(Selector::StreamTopK {
        heads: spec_u32("attention.indexer.head_count", hp.indexer.heads)?,
        d: spec_u32("attention.indexer.key_length", hp.indexer.d)?,
        k: spec_u32("attention.indexer.top_k", hp.indexer.top_k)?,
        keys: Source::Own(IndexKeys::FromInput {
            norm_eps: INDEX_KEY_NORM_EPS,
        }),
        list: Source::Own(()),
        candidates: None,
    });
    Ok(latent)
}
