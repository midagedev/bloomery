//! What a model needs a program to run: one [`Need`] per part of a layer or
//! of the model that a kernel instance or a program step has to exist for.
//! [`needs`] reads them off a [`ModelSpec`] alone; the formats of the file's
//! tensors, the tokenizer and the chat parsers join in the check that
//! compares them with what this tree runs.

use std::fmt;

use gguf::GgmlType;

use crate::{
    Collapse, DeltaKind, Extra, Ffn, HcMix, IndexKeys, LatentUp, LayerIdx, Mixer, ModelSpec,
    Residual, RopeMode, Score, Selector, Source,
};

/// A feature of a file the engine does not run yet: what it is, and the layer
/// it is on (`None` for the whole model).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unimplemented {
    pub layer: Option<usize>,
    pub feature: String,
}

/// One part of a model a program has to run. The fields are the part's
/// instance key: what a kernel is built for, not what it takes as an argument.
#[derive(Clone, Debug, PartialEq)]
pub enum Need {
    /// Grouped-query flash attention: `head` values per head, query heads
    /// packed `pack` to a key head (the largest power of two dividing the group).
    Gqa { head: u32, pack: u32 },
    /// The per-head q/k norm (when `qk_norm`) and rope of a GQA layer.
    QkRope {
        qk_norm: bool,
        head: u32,
        mode: RopeMode,
        dims: u32,
    },
    /// The GQA output multiplied by a per-head sigmoid gate `attn_q` writes.
    OutGate,
    /// Latent attention over `latent`-value rows with `rope` rotated values.
    Latent {
        latent: u32,
        rope: u32,
        up: LatentUp,
    },
    /// A per-head RMS norm on the latent query.
    QueryHeadNorm,
    /// A compressor writing `width`-value rows.
    Compress { width: u32 },
    /// A compressor's position table.
    CompressorApe,
    /// A compressor whose groups overlap.
    CompressorOverlap,
    /// A compressed stream attended whole.
    WholeStream,
    /// A top-k over index keys: `heads` heads of `d` values.
    StreamIndex { heads: u32, d: u32 },
    /// Index keys pooled by the indexer's own compressor.
    IndexKeyCompressor,
    /// A top-k over pooled keys of the raw positions.
    TokenPool { heads: u32, d: u32, pool: u32 },
    /// A delta-rule layer.
    DeltaRule {
        kind: DeltaKind,
        k_heads: u32,
        v_heads: u32,
        d: u32,
        conv: u32,
    },
    /// A router over `experts` keeping `top_k`.
    Router {
        score: Score,
        experts: u32,
        top_k: u32,
        bias: bool,
        norm: bool,
    },
    /// Experts from a token table.
    HashRouting,
    /// A shared expert of `ff` values.
    Shared { ff: u32, sigmoid_gate: bool },
    /// A dense feed-forward layer of `ff` values.
    DenseFfn { ff: u32 },
    /// Hyper-connections of `streams` streams.
    Hc { streams: u32 },
    /// The trained hyper-connection head.
    HcHead,
    /// A hyper-connection wiring other than the lagged mix with the last-mix collapse.
    HcWiring { mix: HcMix, collapse: Collapse },
    /// An engram site whose gate reads `row`-value rows.
    Engram { row: u32 },
    /// A model with no engram site, for a program that runs one.
    NoEngram,
    /// Delta-rule and attention layers in one trunk.
    MixedTrunk,
    /// A recurrent state per sequence: the delta-rule layers' state and conv inputs.
    RecurrentState,
    /// Routed expert stacks of `ty` on a card.
    RoutedFormat(GgmlType),
    /// A tensor family (`what`) of `ty` a program reads in another type.
    WeightFormat { what: &'static str, ty: GgmlType },
    /// The pre-tokenizer this name selects.
    PreTokenizer(String),
    /// A tool-call parser for the file's chat template.
    ToolParser,
    /// A layer program for this architecture.
    Program(&'static str),
}

impl fmt::Display for Need {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Need::Gqa { head, pack } => write!(f, "GQA flash, head {head}, pack {pack}"),
            Need::QkRope {
                qk_norm,
                head,
                mode,
                dims,
            } => {
                let norm = if *qk_norm {
                    "per-head QK norm plus rope"
                } else {
                    "rope without a QK norm"
                };
                write!(
                    f,
                    "{norm}: head {head}, {}, {dims} of {head} dims",
                    mode_name(*mode)
                )
            }
            Need::OutGate => f.write_str("attention output gate: sigmoid, interleaved with q"),
            Need::Latent { latent, rope, up } => match up {
                LatentUp::KeqV => write!(f, "latent attention {latent} x rope {rope}, K = V"),
                LatentUp::Absorbed { qk, v } => write!(
                    f,
                    "latent attention {latent} x rope {rope}, absorbed qk {qk} / v {v}"
                ),
            },
            Need::QueryHeadNorm => f.write_str("the per-head query RMS norm"),
            Need::Compress { width } => write!(f, "a compressor of {width}-value rows"),
            Need::CompressorApe => {
                f.write_str("the compressor's position table attn_compressor_ape")
            }
            Need::CompressorOverlap => f.write_str("overlapping compressor groups"),
            Need::WholeStream => f.write_str("a compressed stream attended whole, without a top-k"),
            Need::StreamIndex { heads, d } => write!(f, "stream indexer: {heads} heads x {d}"),
            Need::IndexKeyCompressor => f.write_str("index keys from the indexer's own compressor"),
            Need::TokenPool { heads, d, pool } => {
                write!(f, "token-pool indexer: {heads} heads x {d}, pool {pool}")
            }
            Need::DeltaRule {
                kind,
                k_heads,
                v_heads,
                d,
                conv,
            } => match kind {
                DeltaKind::Gdn { .. } => write!(
                    f,
                    "delta rule GDN: d {d}, {k_heads} k-heads and {v_heads} v-heads (tiled), conv {conv}"
                ),
                DeltaKind::Kda { .. } => {
                    write!(f, "delta rule KDA: d {d}, {v_heads} heads, conv {conv}")
                }
            },
            Need::Router {
                score,
                experts,
                top_k,
                bias,
                norm,
            } => {
                let score = match score {
                    Score::SqrtSoftplus => "sqrt-softplus",
                    Score::Softmax => "softmax",
                    Score::Sigmoid => "sigmoid",
                };
                write!(f, "router: {score}, {experts} experts, top {top_k}")?;
                if *bias {
                    f.write_str(", with a selection bias")?;
                }
                if !norm {
                    f.write_str(", kept weights not renormalized")?;
                }
                Ok(())
            }
            Need::HashRouting => f.write_str("hash routing by ffn_gate_tid2eid"),
            Need::Shared { ff, sigmoid_gate } => {
                write!(f, "shared expert, ff {ff}")?;
                if *sigmoid_gate {
                    f.write_str(", with a sigmoid gate")?;
                }
                Ok(())
            }
            Need::DenseFfn { ff } => write!(f, "dense SwiGLU layer: ff {ff}"),
            Need::Hc { streams } => write!(f, "hyper-connections of {streams} streams"),
            Need::HcHead => f.write_str("the hyper-connection head output_hc_*"),
            Need::HcWiring { mix, collapse } => {
                write!(f, "hyper-connection mix {mix:?}, collapse {collapse:?}")
            }
            Need::Engram { row } => write!(f, "an engram gate of {row}-value rows"),
            Need::NoEngram => f.write_str("a model without engram sites"),
            Need::MixedTrunk => {
                f.write_str("a program that runs delta-rule and attention layers in one trunk")
            }
            Need::RecurrentState => f.write_str(
                "a recurrent-state slot per sequence (delta-rule state and conv inputs)",
            ),
            Need::RoutedFormat(ty) => write!(f, "{ty} routed experts on a card"),
            Need::WeightFormat { what, ty } => write!(f, "{ty} {what}"),
            Need::PreTokenizer(name) => write!(f, "pre-tokenizer {name}"),
            Need::ToolParser => f.write_str("a tool-call parser for this template"),
            Need::Program(arch) => write!(f, "a layer program for {arch}"),
        }
    }
}

fn mode_name(mode: RopeMode) -> String {
    match mode {
        RopeMode::NormTail => "NORM tail".to_string(),
        RopeMode::Neox => "NeoX".to_string(),
        RopeMode::Imrope { sections } => format!("IMROPE {sections:?}"),
    }
}

/// Every part of `spec`'s trunk a program has to run, layer by layer in
/// layer order, then the model-wide ones (`None`). The next-token layers
/// (`spec.mtp`) are carried and not run, so they need nothing.
#[must_use]
pub fn needs(spec: &ModelSpec) -> Vec<(Need, Option<LayerIdx>)> {
    let mut out = Vec::new();
    for (l, layer) in (0..).zip(&spec.layers) {
        let mut at = |n: Need| out.push((n, Some(l)));
        match &layer.mixer {
            Mixer::Gqa(g) => {
                at(Need::Gqa {
                    head: g.head_dim,
                    pack: 1 << g.group().trailing_zeros(),
                });
                at(Need::QkRope {
                    qk_norm: g.qk_norm,
                    head: g.head_dim,
                    mode: g.rope.mode,
                    dims: g.rope.dims,
                });
                if g.out_gate {
                    at(Need::OutGate);
                }
            }
            Mixer::Latent(a) => {
                at(Need::Latent {
                    latent: a.latent,
                    rope: a.rope.map_or(0, |r| r.dims),
                    up: a.up,
                });
                if let Some(c) = &a.compress {
                    at(Need::Compress { width: a.latent });
                    if let Source::Own(own) = c.rows {
                        if own.ape {
                            at(Need::CompressorApe);
                        }
                        if own.overlap {
                            at(Need::CompressorOverlap);
                        }
                    }
                    if a.select.is_none() {
                        at(Need::WholeStream);
                    }
                }
                match &a.select {
                    Some(Selector::StreamTopK { heads, d, keys, .. }) => {
                        at(Need::StreamIndex {
                            heads: *heads,
                            d: *d,
                        });
                        if matches!(keys, Source::Own(IndexKeys::Compressor(_))) {
                            at(Need::IndexKeyCompressor);
                        }
                    }
                    Some(Selector::TokenPool { heads, d, pool, .. }) => at(Need::TokenPool {
                        heads: *heads,
                        d: *d,
                        pool: *pool,
                    }),
                    None => {}
                }
            }
            Mixer::DeltaRule(r) => at(Need::DeltaRule {
                kind: r.kind,
                k_heads: r.k_heads,
                v_heads: r.v_heads,
                d: r.d,
                conv: r.conv,
            }),
        }
        match &layer.ffn {
            Ffn::Moe(m) => {
                if m.router.hash {
                    at(Need::HashRouting);
                }
                at(Need::Router {
                    score: m.router.score,
                    experts: m.experts,
                    top_k: m.top_k,
                    bias: m.router.bias,
                    norm: m.router.norm,
                });
                if let Some(s) = m.shared {
                    at(Need::Shared {
                        ff: s.ff,
                        sigmoid_gate: s.sigmoid_gate,
                    });
                }
            }
            Ffn::Dense { ff, .. } => at(Need::DenseFfn { ff: *ff }),
        }
        if layer.residual == Residual::Hc
            && let Some(hc) = spec.hc
        {
            at(Need::Hc {
                streams: hc.streams,
            });
        }
        if layer.extras.contains(&Extra::Engram) {
            at(Need::Engram { row: spec.hidden });
        }
    }
    let mut model = |n: Need| out.push((n, None));
    if spec
        .layers
        .iter()
        .any(|l| l.latent().is_some_and(|a| a.q_head_norm))
    {
        model(Need::QueryHeadNorm);
    }
    if let Some(hc) = spec.hc {
        match (hc.mix, hc.collapse) {
            (_, Collapse::Head) => model(Need::HcHead),
            (HcMix::Lagged, Collapse::LastMix) => {}
            (mix, collapse) => model(Need::HcWiring { mix, collapse }),
        }
    }
    let delta = spec
        .layers
        .iter()
        .filter(|l| matches!(l.mixer, Mixer::DeltaRule(_)))
        .count();
    if delta > 0 {
        model(Need::RecurrentState);
        if delta < spec.layers.len() {
            model(Need::MixedTrunk);
        }
    }
    out
}
