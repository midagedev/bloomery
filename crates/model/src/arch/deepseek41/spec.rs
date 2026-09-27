//! The deepseek reader's [`ModelSpec`]: a V4.1 or V4 file's [`Hparams`] and
//! tensor roles as the typed model description ([`read`], [`spec_of`]), and
//! the DSpark draft's [`DraftSpec`] ([`draft_of`]). There is no second key
//! reader: every value comes from [`Hparams`], [`DraftHparams`] or the role
//! table, and the facts the file does not carry from the constants cited
//! where they are used.
//!
//! One refusal is this module's own: a window-only layer (ratio 0) that
//! carries index keys or an indexer's queries. [`Hparams::read`] lets it pass
//! (its walk records the tensors before it skips the layer) and the prompt
//! call's triangle turns itself off for it; a [`LayerSpec`] cannot hold an
//! index on a layer that reads no compressed stream.

use std::collections::HashSet;

use gguf::Split;
use models::{
    Act, Arch, BlockDraft, Candidates, Collapse, Compress, Compressor, DraftSpec, Extra, Ffn,
    HcKind, HcMix, HcSpec, IndexKeys, Latent, LatentOut, LatentUp, LayerSpec, Mixer, ModelSpec,
    Moe, NgramRule, ReasoningFormat, Residual, Rope, RopeMode, Router, Score, Selector, Shared,
    Source, ToolFormat, Yarn,
};

use super::hparams::{
    CANDIDATE_BLOCK_SIZE, CANDIDATE_SOURCE_LAYER, CANDIDATE_TOPK_BLOCKS, Collapse as HpCollapse,
    Hparams, LayerKind, Model,
};
use super::{names, roles};
use crate::arch::dspark::DraftHparams;
use crate::arch::{Read, chat_of, spec_u32};
use crate::placement::{ModelTensors, PlacementError};

/// `split`'s description and roles: [`Hparams::read`], then
/// [`roles::classify`], then [`spec_of`].
pub fn read(split: &Split) -> Result<Read, PlacementError> {
    let hp = Hparams::read(split)?;
    let tensors = roles::classify(split, &hp)?;
    let spec = spec_of(&hp, &tensors, chat(split)?)?;
    Ok(Read {
        spec,
        tensors,
        defaults: Vec::new(),
    })
}

/// `split`'s chat surface: V4.1's parsers, the DSML tool calls and the
/// `<think>` span, for any template.
pub fn chat(split: &Split) -> Result<models::ChatSpec, PlacementError> {
    chat_of(
        split,
        Some(ToolFormat::Dsml),
        Some(ReasoningFormat::ThinkSpan),
    )
}

/// The description of the file `hp` and `tensors` were read from, with the
/// chat surface `chat`.
pub fn spec_of(
    hp: &Hparams,
    tensors: &ModelTensors,
    chat: models::ChatSpec,
) -> Result<ModelSpec, PlacementError> {
    let has: HashSet<&str> = tensors.tensors.iter().map(|t| t.name.as_str()).collect();
    let layers = hp
        .layers
        .iter()
        .enumerate()
        .map(|(l, kind)| layer_of(hp, kind, l, &has))
        .collect::<Result<Vec<_>, _>>()?;
    let (mix, collapse) = match hp.collapse {
        HpCollapse::Lagged => (HcMix::Lagged, Collapse::LastMix),
        HpCollapse::Head => (HcMix::Own, Collapse::Head),
    };
    Ok(ModelSpec {
        arch: match hp.model {
            Model::Deepseek41 => Arch::Deepseek41,
            Model::Deepseek4 => Arch::Deepseek4,
        },
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        ctx_train: spec_u32("context_length", hp.n_ctx_train)?,
        rms_eps: hp.rms_eps,
        layers,
        mtp: Vec::new(),
        hc: Some(HcSpec {
            streams: spec_u32("hyper_connection.count", hp.hc.streams)?,
            kind: HcKind::Mhc {
                sinkhorn: spec_u32("hyper_connection.sinkhorn_iterations", hp.hc.sinkhorn_iters)?,
                eps: hp.hc.eps,
                mix,
                collapse,
            },
        }),
        engram: match &hp.engram {
            Some(e) => Some(models::EngramSpec {
                heads: spec_u32("engram.head_count", e.n_head)?,
                max_ngram: spec_u32("engram.max_ngram_size", e.max_ngram)?,
                key_length: spec_u32("engram.key_length", e.key_length)?,
                rule: NgramRule::Engram,
            }),
            None => None,
        },
        chat,
    })
}

/// Layer `l` of kind `kind`; `has` holds every tensor name of the file.
fn layer_of(
    hp: &Hparams,
    kind: &LayerKind,
    l: usize,
    has: &HashSet<&str>,
) -> Result<LayerSpec, PlacementError> {
    let compressor = || {
        kind.compressor
            .map(|c| Compressor {
                gated: c.gated,
                ape: c.ape,
                overlap: c.overlap,
            })
            .ok_or_else(|| PlacementError::Tensor {
                name: names::attn_compressor_kv(l),
                detail: format!("is not in the file, and layer {l} owns its stream"),
            })
    };
    no_index_on_window_layer(kind, l)?;
    let (compress, select) = match (kind.stream, kind.dense) {
        (Some(s), _) => {
            let rows = if s.kv_source == l {
                Source::Own(compressor()?)
            } else {
                from(s.kv_source)?
            };
            let keys = if s.index_key_source == l {
                Source::Own(match kind.index_compressor {
                    Some(c) => IndexKeys::Compressor(Compressor {
                        gated: c.gated,
                        ape: c.ape,
                        overlap: c.overlap,
                    }),
                    None => IndexKeys::FromRows,
                })
            } else {
                from(s.index_key_source)?
            };
            let list = if s.topk_source == l {
                Source::Own(())
            } else {
                from(s.topk_source)?
            };
            let select = Selector::StreamTopK {
                heads: spec_u32("attention.indexer.head_count", hp.indexer.n_head)?,
                d: spec_u32("attention.indexer.key_length", hp.indexer.head_dim)?,
                k: spec_u32("attention.indexer.top_k", hp.indexer.top_k)?,
                keys,
                list,
                candidates: candidates(hp.model, l)?,
            };
            (
                Some(Compress {
                    ratio: s.ratio,
                    rows,
                }),
                Some(select),
            )
        }
        (None, Some(d)) => (
            Some(Compress {
                ratio: d.ratio,
                rows: if d.kv_source == l {
                    Source::Own(compressor()?)
                } else {
                    from(d.kv_source)?
                },
            }),
            None,
        ),
        (None, None) => (None, None),
    };
    let rope = kind.rope;
    let mixer = Mixer::Latent(Latent {
        heads: spec_u32("attention.head_count", hp.n_head)?,
        q_lora: spec_u32("attention.q_lora_rank", hp.q_lora_rank)?,
        latent: spec_u32("attention.key_length", hp.head_dim)?,
        up: LatentUp::KeqV,
        rope: Some(Rope {
            mode: RopeMode::NormTail,
            dims: spec_u32("rope.dimension_count", hp.rope_dims)?,
            base: rope.base,
            // Hparams keeps YaRN's factor as ik passes it, `1 / factor`.
            yarn: (rope.ext_factor != 0.0).then(|| Yarn {
                factor: 1.0 / rope.freq_scale,
                orig_ctx: rope.n_ctx_orig,
                beta_fast: rope.beta_fast,
                beta_slow: rope.beta_slow,
            }),
        }),
        q_head_norm: hp.q_head_norm,
        out: LatentOut::Grouped {
            groups: spec_u32("attention.output_group_count", hp.o_groups)?,
            rank: spec_u32("attention.output_lora_rank", hp.o_lora_rank)?,
        },
        window: Some(spec_u32("attention.sliding_window", hp.window)?),
        sinks: has.contains(names::attn_sinks(l).as_str()),
        compress,
        select,
    });
    if !kind.routed {
        return Err(PlacementError::Tensor {
            name: names::ffn_gate_inp(l),
            detail: format!("is not in the file: layer {l} is dense, and this reader reads none"),
        });
    }
    let e = &hp.experts;
    let expert_ff = spec_u32("expert_feed_forward_length", e.ff)?;
    let ffn = Ffn::Moe(Moe {
        experts: spec_u32("expert_count", e.n_expert)?,
        top_k: spec_u32("expert_used_count", e.n_used)?,
        expert_ff,
        act: Act::SwiGlu {
            limit: Some(kind.swiglu_limit),
        },
        router: Router {
            score: Score::SqrtSoftplus,
            bias: has.contains(names::exp_probs_b(l).as_str()),
            norm: e.weights_norm,
            scale: e.routed_scale,
            hash: kind.hash_routed,
        },
        shared: match e.n_shared {
            0 => None,
            n => Some(Shared {
                ff: spec_u32("expert_shared_count", n)? * expert_ff,
                act: Act::SwiGlu {
                    limit: Some(kind.swiglu_limit_shared),
                },
                sigmoid_gate: false,
            }),
        },
    });
    Ok(LayerSpec {
        mixer,
        ffn,
        residual: Residual::Hc,
        extras: if kind.engram.is_some() {
            vec![Extra::Engram]
        } else {
            Vec::new()
        },
    })
}

/// A window-only layer (ratio 0) carries neither index keys nor an
/// indexer's queries: it reads no compressed stream to index.
fn no_index_on_window_layer(kind: &LayerKind, l: usize) -> Result<(), PlacementError> {
    if kind.compressed() {
        return Ok(());
    }
    let carried = [
        (kind.index_keys, names::indexer_attn_k(l)),
        (kind.indexer, names::indexer_attn_q_b(l)),
    ]
    .into_iter()
    .find(|(on, _)| *on);
    match carried {
        Some((_, name)) => Err(PlacementError::Tensor {
            name,
            detail: format!(
                "is in the file, and layer {l} compresses nothing: a window-only layer reads no index"
            ),
        }),
        None => Ok(()),
    }
}

/// What layer `src` writes, read by a later layer.
fn from<T>(src: usize) -> Result<Source<T>, PlacementError> {
    spec_u32("layer", src).map(Source::From)
}

/// Layer `l`'s candidate pool: V4.1's reference pools every layer after
/// [`CANDIDATE_SOURCE_LAYER`] among the blocks that layer's index scores rank
/// first (the HF `config.json` carries it, the GGUF does not); V4 has none.
fn candidates(model: Model, l: usize) -> Result<Option<Candidates>, PlacementError> {
    Ok(match model {
        Model::Deepseek41 if l > CANDIDATE_SOURCE_LAYER => Some(Candidates {
            source: spec_u32("candidate source layer", CANDIDATE_SOURCE_LAYER)?,
            blocks: spec_u32("candidate blocks", CANDIDATE_TOPK_BLOCKS)?,
            block: spec_u32("candidate block size", CANDIDATE_BLOCK_SIZE)?,
        }),
        Model::Deepseek41 | Model::Deepseek4 => None,
    })
}

/// The DSpark draft file's description: window-only V4.1 blocks with plain
/// rope, each routed with a shared expert, under the lagged hyper-connection
/// mix, and the block pass's width, taps, mask token and Markov rank.
pub fn draft_of(hp: &DraftHparams) -> Result<DraftSpec, PlacementError> {
    let e = &hp.experts;
    let expert_ff = spec_u32("expert_feed_forward_length", e.ff)?;
    let layers = (0..hp.n_layer)
        .map(|l| {
            Ok(LayerSpec {
                mixer: Mixer::Latent(Latent {
                    heads: spec_u32("attention.head_count", hp.n_head)?,
                    q_lora: spec_u32("attention.q_lora_rank", hp.q_lora_rank)?,
                    latent: spec_u32("attention.key_length", hp.head_dim)?,
                    up: LatentUp::KeqV,
                    rope: Some(Rope {
                        mode: RopeMode::NormTail,
                        dims: spec_u32("rope.dimension_count", hp.rope_dims)?,
                        base: hp.rope_base,
                        yarn: None,
                    }),
                    q_head_norm: false,
                    out: LatentOut::Grouped {
                        groups: spec_u32("attention.output_group_count", hp.o_groups)?,
                        rank: spec_u32("attention.output_lora_rank", hp.o_lora_rank)?,
                    },
                    window: Some(spec_u32("attention.sliding_window", hp.window)?),
                    sinks: true,
                    compress: None,
                    select: None,
                }),
                ffn: Ffn::Moe(Moe {
                    experts: spec_u32("expert_count", e.n_expert)?,
                    top_k: spec_u32("expert_used_count", e.n_used)?,
                    expert_ff,
                    act: Act::SwiGlu {
                        limit: Some(hp.swiglu_limit[l]),
                    },
                    router: Router {
                        score: Score::SqrtSoftplus,
                        bias: true,
                        norm: e.weights_norm,
                        scale: e.routed_scale,
                        hash: false,
                    },
                    shared: match e.n_shared {
                        0 => None,
                        n => Some(Shared {
                            ff: spec_u32("expert_shared_count", n)? * expert_ff,
                            act: Act::SwiGlu {
                                limit: Some(hp.swiglu_limit_shared[l]),
                            },
                            sigmoid_gate: false,
                        }),
                    },
                }),
                residual: Residual::Hc,
                extras: Vec::new(),
            })
        })
        .collect::<Result<Vec<_>, PlacementError>>()?;
    Ok(DraftSpec::Block(BlockDraft {
        hidden: spec_u32("embedding_length", hp.n_embd)?,
        vocab: spec_u32("vocab", hp.n_vocab)?,
        rms_eps: hp.rms_eps,
        hc: HcSpec {
            streams: spec_u32("hyper_connection.count", hp.hc.streams)?,
            kind: HcKind::Mhc {
                sinkhorn: spec_u32("hyper_connection.sinkhorn_iterations", hp.hc.sinkhorn_iters)?,
                eps: hp.hc.eps,
                mix: HcMix::Lagged,
                collapse: Collapse::LastMix,
            },
        },
        layers,
        width: spec_u32("block_size", hp.block_size)?,
        target_layers: hp
            .target_layers
            .iter()
            .map(|&l| spec_u32("target_layers", l))
            .collect::<Result<_, _>>()?,
        mask_token: hp.mask_token,
        markov_rank: spec_u32("markov rank", hp.markov_rank)?,
    }))
}

#[cfg(test)]
mod tests {
    use super::no_index_on_window_layer;
    use crate::arch::deepseek41::hparams::{LayerKind, Rope};
    use gguf::GgmlType;

    /// A window-only layer: no stream, no compressor, no index.
    fn window_only() -> LayerKind {
        LayerKind {
            stream: None,
            dense: None,
            compressor: None,
            index_keys: false,
            index_compressor: None,
            indexer: false,
            engram: None,
            routed: true,
            hash_routed: false,
            rope: Rope {
                base: 10_000.0,
                freq_scale: 1.0,
                ext_factor: 0.0,
                attn_factor: 1.0,
                beta_fast: 0.0,
                beta_slow: 0.0,
                n_ctx_orig: 0,
            },
            swiglu_limit: 10.0,
            swiglu_limit_shared: 10.0,
            shared_down: GgmlType::Q8_0,
        }
    }

    /// Index keys or an indexer on a window-only layer are refused by the
    /// tensor's name; the layer without them reads.
    #[test]
    fn a_window_only_layer_carries_no_index() {
        assert!(no_index_on_window_layer(&window_only(), 1).is_ok());
        for (keys, queries, name) in [
            (true, false, "blk.1.indexer.attn_k.weight"),
            (false, true, "blk.1.indexer.attn_q_b.weight"),
        ] {
            let k = LayerKind {
                index_keys: keys,
                indexer: queries,
                ..window_only()
            };
            let err = no_index_on_window_layer(&k, 1).expect_err(name).to_string();
            assert!(
                err.contains(name) && err.contains("compresses nothing"),
                "{err}"
            );
        }
    }
}
