//! What a model needs a program to run: one [`Need`] per part of a layer or
//! of the model that a kernel instance or a program step has to exist for.
//! [`needs`] reads them off a [`ModelSpec`] alone; the formats of the file's
//! tensors, the tokenizer and the chat parsers join in the check that
//! compares them with what this tree runs.

use std::fmt;

use gguf::GgmlType;

use crate::shape::MoeShape;
use crate::{
    Act, Collapse, DeltaKind, EngramSpec, Extra, Ffn, GdnGate, HcKind, HcMix, IndexKeys, KdaGate,
    LatentUp, LayerIdx, Mixer, ModelSpec, NgramRule, OutGate, PoolRule, Residual, RopeMode,
    RouterNorm, Score, Selector, Source,
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
    /// Grouped-query flash attention: `head` values per head, `group` query
    /// heads to a key head (the shape table picks the pack that serves it).
    ///
    /// The flash is also built for `value` values per value head, a `window`
    /// of positions (`None`: every position), per-head `sinks` and a
    /// multiplier on the value rows (`scaled`); a row serves a need only
    /// where they match what it runs.
    Gqa {
        head: u32,
        value: u32,
        group: u32,
        window: Option<u32>,
        sinks: bool,
        scaled: bool,
    },
    /// Grouped-query flash attention in both directions: `head` values per
    /// key and value head, `group` query heads to a key head, `reach` keys on
    /// each side of a row (`None`: every key).
    Bidirectional {
        head: u32,
        group: u32,
        reach: Option<u32>,
    },
    /// The per-head q/k norm (when `qk_norm`) and rope of a GQA layer. A
    /// layer with no rope and a norm carries `mode: None` and dims 0.
    QkRope {
        qk_norm: bool,
        head: u32,
        mode: RopeMode,
        dims: u32,
    },
    /// The GQA output multiplied by a per-head sigmoid gate `attn_q` writes.
    OutGate,
    /// The GQA output multiplied by a sigmoid gate a projection of the layer input of its own
    /// writes.
    OutGateProj,
    /// The GQA output multiplied by a sigmoid gate a projection of the layer input of its own
    /// writes, one gate per query head.
    OutGateProjHead,
    /// The rope's pair frequencies divided by the file's factors (`rope_freqs.weight`).
    RopeFactors,
    /// The heads' outputs RMS-normed per head, with no gain, before the output gate:
    /// `head` values per head.
    OutNorm { head: u32 },
    /// Latent attention over `latent`-value rows with `rope` rotated values.
    Latent {
        latent: u32,
        rope: u32,
        up: LatentUp,
    },
    /// A latent layer whose cached row carries `width` key values beside the latent that no rope
    /// turns, shared by every head.
    LatentPe { width: u32 },
    /// A latent layer's heads' outputs multiplied by a sigmoid gate a projection of the layer input
    /// of its own writes.
    LatentOutGate,
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
    /// Index keys projected from the layer input over the raw positions.
    IndexKeyFromInput,
    /// A latent layer that reads a cache another layer wrote: it stores
    /// nothing and attends the source's rows.
    SharedLatentCache,
    /// SSMax: the query scaled per head by `s·ln(pos + 1) + b`, the
    /// attention's and its indexer's.
    SsMax,
    /// A top-k over learned pools of the raw positions' keys.
    TokenPool { heads: u32, d: u32, pool: u32 },
    /// A LayerNorm with a gain and a bias on each index key.
    KeyLayerNorm,
    /// A top-k over mean pools of the raw positions' keys, RMS-normed and
    /// rotated (`rope` values) at the pool's first position, the heads summed.
    MeanPool {
        heads: u32,
        d: u32,
        pool: u32,
        rope: u32,
    },
    /// A top-k of position blocks scored by the largest index score in them,
    /// the query's own block kept inside the k: `heads` heads x `d`, blocks of
    /// `block` positions, `top_k` kept (`local` forced), keys RMS-normed and
    /// rotated (`rope` values).
    BlockMax {
        heads: u32,
        d: u32,
        top_k: u32,
        block: u32,
        local: u32,
        rope: u32,
    },
    /// A delta-rule layer.
    DeltaRule {
        kind: DeltaKind,
        k_heads: u32,
        v_heads: u32,
        d: u32,
        conv: u32,
    },
    /// A router: its rule, the experts it routes over and the experts each
    /// token keeps.
    Router(MoeShape),
    /// Experts from a token table.
    HashRouting,
    /// A shared expert of `ff` values.
    Shared { ff: u32, sigmoid_gate: bool },
    /// Routed experts that read and write a `latent`-value row: the layer's
    /// input projected down to it before them, their sum back up after.
    LatentExperts { latent: u32 },
    /// A weighted RMS norm on the routed experts' input (`ffn_pre_exp_norm`).
    PreExpNorm,
    /// The routed experts' aggregated output scaled by `ffn_post_exp_norm`,
    /// then normalized at `eps`, before the router's weights apply.
    PostExpNorm { eps: f32 },
    /// Experts (`shared`: the shared expert, else the routed ones) of an
    /// activation other than SwiGLU, the one every routed body runs.
    ExpertAct { shared: bool, act: Act },
    /// Routed stacks whose expert count or width differs from layer to layer:
    /// `shapes` distinct (count, width) pairs over the trunk.
    ExpertShapes { shapes: u32 },
    /// A dense feed-forward layer of `ff` values.
    DenseFfn { ff: u32 },
    /// A dense squared-ReLU layer of `ff` values, with no gate.
    ReluSqr { ff: u32 },
    /// A dense GELU GLU layer of `ff` values.
    GeGlu { ff: u32 },
    /// A dense layer of `ff` values of an activation with a clamped gate and
    /// an offset up path (`swigluoai`).
    SwigluOai { ff: u32 },
    /// A dense layer of `ff` values of the bounded gate and up activation `situ`.
    Situ { ff: u32 },
    /// An encoder's value rows RMS-normed per head of `head` values, with no
    /// gain.
    ValueNorm { head: u32 },
    /// An encoder layer's RMS norms on its attention's and its feed-forward
    /// block's outputs, before each residual add.
    PostNorms,
    /// An encoder layer's output multiplied by its trained scalar.
    LayerScale,
    /// An encoder layer's own input of `width` values, projected from the
    /// scaled token rows, gating a block of its own.
    PerLayerInput { width: u32 },
    /// An encoder's token rows scaled by the square root of the hidden width.
    EmbedScale,
    /// An encoder's output: the linear maps of `proj` widths after the output
    /// norm, then the mean of the token rows.
    Pooled { proj: Vec<u32> },
    /// A Mamba-2 mixer. The fields are the widths a scan kernel is built for.
    Mamba2 {
        d_inner: u32,
        d_state: u32,
        n_heads: u32,
        n_groups: u32,
        d_conv: u32,
    },
    /// A gated short-conv mixer of `taps` taps.
    ShortConv { taps: u32 },
    /// Hyper-connections of `streams` streams.
    Hc { streams: u32 },
    /// Chart-constrained hyper-connections of `streams` streams, every sublayer its own mix, the
    /// streams collapsed by `collapse`.
    HcChart { streams: u32, collapse: Collapse },
    /// Gated-residual hyper-connections of `streams` streams through a
    /// `rank`-value bottleneck.
    HcGated { streams: u32, rank: u32 },
    /// The trained hyper-connection head.
    HcHead,
    /// The gated-residual hyper-connection head.
    HcGatedHead { rank: u32 },
    /// A hyper-connection wiring other than the lagged mix with the last-mix collapse.
    HcWiring { mix: HcMix, collapse: Collapse },
    /// The residual joined per channel, α·stream + β·branch, each sublayer's
    /// output normalized at `eps` first.
    AlphaBeta { eps: f32 },
    /// Each sublayer's output weight-normed at `eps` before it joins the stream, scaled by
    /// `add` (mixer, feed-forward block) when it does.
    Sandwich { eps: f32, add: [f32; 2] },
    /// An engram site whose gate reads `row`-value rows.
    Engram { row: u32 },
    /// A model with no engram site, for a program that runs one.
    NoEngram,
    /// A per-layer n-gram embedding site whose gate reads `row`-value rows,
    /// with its conv of `conv` taps `dilation` apart.
    Ple { row: u32, conv: u32, dilation: u32 },
    /// A text prompt carrying the image token refused by name: the reference
    /// hashes an image's positions as that id.
    ImageToken(u32),
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
    /// The head's logit epilogue: the scale, then the tanh softcap.
    LogitEpilogue { scale: f32, softcap: f32 },
    /// A layer program for this architecture.
    Program(&'static str),
}

impl fmt::Display for Need {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Need::Gqa {
                head,
                value,
                group,
                window,
                sinks,
                scaled,
            } => {
                write!(f, "GQA flash, head {head}")?;
                if value != head {
                    write!(f, ", value {value}")?;
                }
                write!(f, ", group {group}")?;
                if let Some(w) = window {
                    write!(f, ", window {w}")?;
                }
                if *sinks {
                    f.write_str(", sinks")?;
                }
                if *scaled {
                    f.write_str(", value scale")?;
                }
                Ok(())
            }
            Need::Bidirectional { head, group, reach } => {
                write!(f, "bidirectional GQA flash, head {head}, group {group}")?;
                if let Some(r) = reach {
                    write!(f, ", {r} keys each side")?;
                }
                Ok(())
            }
            Need::QkRope {
                qk_norm,
                head,
                mode,
                dims,
            } => {
                let norm = match (*qk_norm, *mode) {
                    (true, RopeMode::None) => {
                        return write!(f, "per-head QK norm, no rope: head {head}");
                    }
                    (true, _) => "per-head QK norm plus rope",
                    (false, _) => "rope without a QK norm",
                };
                write!(
                    f,
                    "{norm}: head {head}, {}, {dims} of {head} dims",
                    mode_name(*mode)
                )
            }
            Need::OutGate => f.write_str("attention output gate: sigmoid, interleaved with q"),
            Need::OutGateProj => {
                f.write_str("attention output gate: sigmoid of its own projection")
            }
            Need::OutGateProjHead => f.write_str(
                "attention output gate: sigmoid of its own projection, one gate per head",
            ),
            Need::RopeFactors => f.write_str("rope frequency factors (rope_freqs.weight)"),
            Need::OutNorm { head } => {
                write!(
                    f,
                    "a gainless per-head RMS norm on the attention output: head {head}"
                )
            }
            Need::Latent { latent, rope, up } => match up {
                LatentUp::KeqV => write!(f, "latent attention {latent} x rope {rope}, K = V"),
                LatentUp::Absorbed { qk, v } => write!(
                    f,
                    "latent attention {latent} x rope {rope}, absorbed qk {qk} / v {v}"
                ),
            },
            Need::LatentPe { width } => write!(
                f,
                "latent attention with {width} unrotated key values beside the latent, shared by every head"
            ),
            Need::LatentOutGate => {
                f.write_str("latent attention output gate: sigmoid of its own projection")
            }
            Need::QueryHeadNorm => f.write_str("the per-head query RMS norm"),
            Need::Compress { width } => write!(f, "a compressor of {width}-value rows"),
            Need::CompressorApe => {
                f.write_str("the compressor's position table attn_compressor_ape")
            }
            Need::CompressorOverlap => f.write_str("overlapping compressor groups"),
            Need::WholeStream => f.write_str("a compressed stream attended whole, without a top-k"),
            Need::StreamIndex { heads, d } => write!(f, "stream indexer: {heads} heads x {d}"),
            Need::IndexKeyCompressor => f.write_str("index keys from the indexer's own compressor"),
            Need::IndexKeyFromInput => {
                f.write_str("index keys projected from the layer input, one per position")
            }
            Need::SharedLatentCache => {
                f.write_str("the latent cache another layer wrote, read in place of its own")
            }
            Need::SsMax => f.write_str("SSMax: the query scaled per head by s*ln(pos + 1) + b"),
            Need::TokenPool { heads, d, pool } => {
                write!(f, "token-pool indexer: {heads} heads x {d}, pool {pool}")
            }
            Need::KeyLayerNorm => f.write_str("a LayerNorm with a bias on the index keys"),
            Need::MeanPool {
                heads,
                d,
                pool,
                rope,
            } => write!(
                f,
                "mean-pool indexer: {heads} heads x {d}, pool {pool}, RMS-normed keys roped ({rope} dims) at the pool start, heads summed"
            ),
            Need::BlockMax {
                heads,
                d,
                top_k,
                block,
                local,
                rope,
            } => write!(
                f,
                "block-max indexer: {heads} heads x {d}, blocks of {block} keep {top_k} ({local} local), keys RMS-normed and roped ({rope} dims), each key head on its own"
            ),
            Need::DeltaRule {
                kind,
                k_heads,
                v_heads,
                d,
                conv,
            } => match kind {
                DeltaKind::Gdn { gate, .. } => {
                    write!(
                        f,
                        "delta rule GDN: d {d}, {k_heads} k-heads and {v_heads} v-heads (tiled), conv {conv}"
                    )?;
                    match gate {
                        GdnGate::Silu => Ok(()),
                        GdnGate::Sigmoid => f.write_str(", sigmoid output gate"),
                    }
                }
                DeltaKind::Kda { gate, beta_scale } => {
                    write!(f, "delta rule KDA: d {d}, {v_heads} heads, conv {conv}")?;
                    match gate {
                        KdaGate::Sigmoid { lower_bound } => {
                            write!(f, ", sigmoid gate bounded below by {lower_bound}")?;
                        }
                        KdaGate::Softplus => f.write_str(", softplus gate")?,
                    }
                    if *beta_scale != 1.0 {
                        write!(f, ", beta scaled by {beta_scale}")?;
                    }
                    Ok(())
                }
                DeltaKind::Rda { gate } => {
                    write!(
                        f,
                        "delta rule RDA: d {d}, {v_heads} heads, conv {conv}, erase and write gates apart, full-width output gate"
                    )?;
                    match gate {
                        KdaGate::Sigmoid { lower_bound } => {
                            write!(f, ", sigmoid gate bounded below by {lower_bound}")
                        }
                        KdaGate::Softplus => f.write_str(", softplus gate"),
                    }
                }
            },
            Need::Router(MoeShape {
                rule,
                experts,
                top_k,
            }) => {
                let score = match rule.score {
                    Score::SqrtSoftplus => "sqrt-softplus",
                    Score::Softmax => "softmax",
                    Score::Sigmoid => "sigmoid",
                };
                write!(f, "router: {score}, {experts} experts, top {top_k}")?;
                if rule.bias {
                    f.write_str(", with a selection bias")?;
                }
                match rule.norm {
                    RouterNorm::None => f.write_str(", kept weights not renormalized")?,
                    RouterNorm::SelectedSum => {}
                    RouterNorm::SelectedSumFloor => {
                        f.write_str(", kept weights divided by max(selected sum, 2^-14)")?;
                    }
                }
                if rule.gated {
                    f.write_str(", the shared expert's gate as one more row")?;
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
            Need::LatentExperts { latent } => write!(
                f,
                "routed experts in a {latent}-value latent (ffn_latent_down before them, \
                 ffn_latent_up after their sum)"
            ),
            Need::PreExpNorm => {
                f.write_str("a weighted RMS norm on the routed experts' input (ffn_pre_exp_norm)")
            }
            Need::PostExpNorm { eps } => write!(
                f,
                "the routed experts' output scaled by ffn_post_exp_norm, then normed at {eps}"
            ),
            Need::ExpertAct { shared, act } => {
                let who = if *shared {
                    "shared expert"
                } else {
                    "routed experts"
                };
                match act {
                    Act::ReluSqr => write!(f, "{who} of squared ReLU, no gate"),
                    other => write!(f, "{who} of {}", other.what()),
                }
            }
            Need::ExpertShapes { shapes } => write!(
                f,
                "routed stacks whose expert count and width differ by layer ({shapes} shapes)"
            ),
            Need::DenseFfn { ff } => write!(f, "dense SwiGLU layer: ff {ff}"),
            Need::ReluSqr { ff } => write!(f, "squared-ReLU dense layer: ff {ff}"),
            Need::GeGlu { ff } => write!(f, "dense GELU GLU layer: ff {ff}"),
            Need::SwigluOai { ff } => write!(f, "dense swigluoai layer: ff {ff}"),
            Need::Situ { ff } => write!(f, "dense situ layer: ff {ff}"),
            Need::ValueNorm { head } => {
                write!(f, "a gainless per-head RMS norm on the values: head {head}")
            }
            Need::PostNorms => f.write_str("RMS norms on the attention and feed-forward outputs"),
            Need::LayerScale => f.write_str("a trained scalar on the layer's output"),
            Need::PerLayerInput { width } => {
                write!(f, "a per-layer input of {width} values from the token rows")
            }
            Need::EmbedScale => f.write_str("token rows scaled by the root of the hidden width"),
            Need::Pooled { proj } => {
                write!(
                    f,
                    "the output projections {proj:?}, then the mean of the rows"
                )
            }
            Need::Mamba2 {
                d_inner,
                d_state,
                n_heads,
                n_groups,
                d_conv,
            } => write!(
                f,
                "Mamba-2 mixer: inner {d_inner}, state {d_state}, {n_heads} heads, {n_groups} groups, conv {d_conv}"
            ),
            Need::ShortConv { taps } => write!(f, "gated short-conv mixer: {taps} taps"),
            Need::Hc { streams } => write!(f, "hyper-connections of {streams} streams"),
            Need::HcChart { streams, collapse } => write!(
                f,
                "chart-constrained hyper-connections of {streams} streams, collapse {collapse:?}"
            ),
            Need::HcGated { streams, rank } => write!(
                f,
                "gated-residual hyper-connections of {streams} streams, rank {rank}"
            ),
            Need::HcHead => f.write_str("the hyper-connection head output_hc_*"),
            Need::HcGatedHead { rank } => write!(
                f,
                "the gated-residual hyper-connection head output_hc_*, rank {rank}"
            ),
            Need::HcWiring { mix, collapse } => {
                write!(f, "hyper-connection mix {mix:?}, collapse {collapse:?}")
            }
            Need::AlphaBeta { eps } => write!(
                f,
                "the alpha/beta residual mix, each sublayer's output normed at {eps} first"
            ),
            Need::Sandwich { eps, add } => write!(
                f,
                "sandwich norms: each sublayer's output weight-normed at {eps}, then added scaled by {add:?}"
            ),
            Need::Engram { row } => write!(f, "an engram gate of {row}-value rows"),
            Need::NoEngram => f.write_str("a model without engram sites"),
            Need::Ple {
                row,
                conv,
                dilation,
            } => write!(
                f,
                "a PLE site: a gate of {row}-value rows, a conv of {conv} taps {dilation} apart"
            ),
            Need::ImageToken(t) => write!(
                f,
                "a text prompt carrying the image token {t} refused by name"
            ),
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
            Need::LogitEpilogue { scale, softcap } => {
                if *softcap > 0.0 {
                    write!(
                        f,
                        "logit epilogue: scale {scale}, then the tanh softcap at {softcap}"
                    )
                } else {
                    write!(f, "logit epilogue: scale {scale}")
                }
            }
            Need::Program(arch) => write!(f, "a layer program for {arch}"),
        }
    }
}

fn mode_name(mode: RopeMode) -> String {
    match mode {
        RopeMode::NormTail => "NORM tail".to_string(),
        RopeMode::Neox => "NeoX".to_string(),
        RopeMode::Imrope { sections } => format!("IMROPE {sections:?}"),
        RopeMode::None => "no rope".to_string(),
    }
}

/// Every part of `spec`'s trunk a program has to run, layer by layer in
/// layer order, then the model-wide ones (`None`). The next-token layers
/// (`spec.mtp`) are carried and not run, so they need nothing. An encoder's
/// attention is [`Need::Bidirectional`], never a causal [`Need::Gqa`], and
/// what its [`crate::EncoderSpec`] carries past the trunk is a need too.
#[must_use]
pub fn needs(spec: &ModelSpec) -> Vec<(Need, Option<LayerIdx>)> {
    let mut out = Vec::new();
    for (l, layer) in (0..).zip(&spec.layers) {
        let mut at = |n: Need| out.push((n, Some(l)));
        match &layer.mixer {
            Mixer::Gqa(g) => {
                at(match &spec.encoder {
                    Some(e) => Need::Bidirectional {
                        head: g.head_dim,
                        group: g.group(),
                        reach: *e
                            .reach
                            .get(l as usize)
                            .expect("encoder readers build reach from one pattern entry per layer"),
                    },
                    None => Need::Gqa {
                        head: g.head_dim,
                        value: g.value_dim,
                        group: g.group(),
                        window: g.window,
                        sinks: g.sinks,
                        scaled: g.value_scale.is_some(),
                    },
                });
                // A layer with no rope still norms q and k per head: the norm
                // is not the rope's passenger.
                if g.qk_norm || g.rope.mode != RopeMode::None {
                    at(Need::QkRope {
                        qk_norm: g.qk_norm,
                        head: g.head_dim,
                        mode: g.rope.mode,
                        dims: g.rope.dims,
                    });
                }
                if g.rope.factors {
                    at(Need::RopeFactors);
                }
                match g.out_gate {
                    OutGate::None => {}
                    OutGate::Packed => at(Need::OutGate),
                    OutGate::Proj => at(Need::OutGateProj),
                    OutGate::ProjHead => at(Need::OutGateProjHead),
                }
                if g.o_norm {
                    at(Need::OutNorm { head: g.head_dim });
                }
                if let Some(e) = &spec.encoder
                    && e.value_norm
                {
                    at(Need::ValueNorm { head: g.head_dim });
                }
                if let Some(s) = &g.select {
                    select_needs(s, &mut at);
                }
            }
            Mixer::Latent(a) => {
                at(Need::Latent {
                    latent: a.latent,
                    rope: a.rope.map_or(0, |r| r.dims),
                    up: a.up,
                });
                if a.pe > 0 {
                    at(Need::LatentPe { width: a.pe });
                }
                if a.ssmax {
                    at(Need::SsMax);
                }
                if matches!(a.cache, Source::From(_)) {
                    at(Need::SharedLatentCache);
                }
                if a.out_gate == OutGate::Proj {
                    at(Need::LatentOutGate);
                }
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
                if let Some(s) = &a.select {
                    select_needs(s, &mut at);
                }
            }
            Mixer::DeltaRule(r) => at(Need::DeltaRule {
                kind: r.kind,
                k_heads: r.k_heads,
                v_heads: r.v_heads,
                d: r.d,
                conv: r.conv,
            }),
            Mixer::Mamba2(m) => at(Need::Mamba2 {
                d_inner: m.d_inner,
                d_state: m.d_state,
                n_heads: m.n_heads,
                n_groups: m.n_groups,
                d_conv: m.d_conv,
            }),
            Mixer::ShortConv(c) => at(Need::ShortConv { taps: c.taps }),
            Mixer::None => {}
        }
        match &layer.ffn {
            Ffn::Moe(m) => {
                if m.router.hash {
                    at(Need::HashRouting);
                }
                at(Need::Router(MoeShape::of(m)));
                if let Some(latent) = m.latent {
                    at(Need::LatentExperts { latent });
                }
                if m.pre_exp_norm {
                    at(Need::PreExpNorm);
                }
                if let Some(eps) = m.post_exp_norm {
                    at(Need::PostExpNorm { eps });
                }
                if !matches!(m.act, Act::SwiGlu { .. }) {
                    at(Need::ExpertAct {
                        shared: false,
                        act: m.act,
                    });
                }
                if let Some(s) = m.shared {
                    at(Need::Shared {
                        ff: s.ff,
                        sigmoid_gate: s.sigmoid_gate,
                    });
                    if !matches!(s.act, Act::SwiGlu { .. }) {
                        at(Need::ExpertAct {
                            shared: true,
                            act: s.act,
                        });
                    }
                }
            }
            Ffn::Dense { ff, act } => match act {
                Act::SwiGlu { .. } => at(Need::DenseFfn { ff: *ff }),
                // Width 0 is a layer with no feed-forward block. A squared
                // ReLU of a positive width is the block.
                Act::ReluSqr if *ff > 0 => at(Need::ReluSqr { ff: *ff }),
                Act::ReluSqr => {}
                Act::GeGlu => at(Need::GeGlu { ff: *ff }),
                Act::SwigluOai { .. } => at(Need::SwigluOai { ff: *ff }),
                Act::Situ { .. } => at(Need::Situ { ff: *ff }),
            },
        }
        match layer.residual {
            Residual::Plain => {}
            // `spec.hc` is None only where no layer is `Hc`.
            Residual::Hc => {
                if let Some(hc) = spec.hc {
                    at(match hc.kind {
                        HcKind::Mhc { .. } => Need::Hc {
                            streams: hc.streams,
                        },
                        HcKind::Chart { collapse, .. } => Need::HcChart {
                            streams: hc.streams,
                            collapse,
                        },
                        HcKind::Gated { rank } => Need::HcGated {
                            streams: hc.streams,
                            rank,
                        },
                    });
                }
            }
            Residual::AlphaBeta { eps } => {
                at(Need::AlphaBeta { eps });
            }
            Residual::Sandwich { eps, add } => at(Need::Sandwich { eps, add }),
        }
        if let Some(e) = &spec.encoder {
            if e.post_norms {
                at(Need::PostNorms);
            }
            if let Some(width) = e.per_layer {
                at(Need::PerLayerInput { width });
            }
            if e.layer_scale {
                at(Need::LayerScale);
            }
        }
        if layer.extras.contains(&Extra::Engram) {
            at(Need::Engram { row: spec.hidden });
        }
        if layer.extras.contains(&Extra::Ple)
            && let Some(EngramSpec {
                rule: NgramRule::Ple { conv, dilation, .. },
                ..
            }) = spec.engram
        {
            at(Need::Ple {
                row: spec.hidden,
                conv,
                dilation,
            });
        }
    }
    let mut model = |n: Need| out.push((n, None));
    let mut shapes: Vec<(u32, u32)> = Vec::new();
    for m in spec.layers.iter().filter_map(crate::LayerSpec::moe) {
        if !shapes.contains(&(m.experts, m.expert_ff)) {
            shapes.push((m.experts, m.expert_ff));
        }
    }
    if shapes.len() > 1 {
        model(Need::ExpertShapes {
            shapes: u32::try_from(shapes.len()).expect("a count of u32 layer shapes"),
        });
    }
    if spec
        .layers
        .iter()
        .any(|l| l.latent().is_some_and(|a| a.q_head_norm))
    {
        model(Need::QueryHeadNorm);
    }
    match spec.hc.map(|hc| hc.kind) {
        Some(HcKind::Mhc { mix, collapse, .. }) => match (mix, collapse) {
            (_, Collapse::Head) => model(Need::HcHead),
            (HcMix::Lagged, Collapse::LastMix) => {}
            (mix, collapse) => model(Need::HcWiring { mix, collapse }),
        },
        Some(HcKind::Chart { .. }) => {}
        Some(HcKind::Gated { rank }) => model(Need::HcGatedHead { rank }),
        None => {}
    }
    if let Some(e) = spec.logit_epilogue {
        model(Need::LogitEpilogue {
            scale: e.scale,
            softcap: e.softcap,
        });
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
    if let Some(EngramSpec {
        rule: NgramRule::Ple {
            image: Some(token), ..
        },
        ..
    }) = spec.engram
    {
        model(Need::ImageToken(token));
    }
    if let Some(e) = &spec.encoder {
        if e.embed_scale {
            model(Need::EmbedScale);
        }
        model(Need::Pooled {
            proj: e.proj.clone(),
        });
    }
    out
}

/// What a layer's position selector needs, pushed through `at`.
fn select_needs(select: &Selector, at: &mut impl FnMut(Need)) {
    match select {
        Selector::StreamTopK { heads, d, keys, .. } => {
            at(Need::StreamIndex {
                heads: *heads,
                d: *d,
            });
            match keys {
                Source::Own(IndexKeys::Compressor(_)) => at(Need::IndexKeyCompressor),
                Source::Own(IndexKeys::FromInput { .. }) => at(Need::IndexKeyFromInput),
                Source::Own(IndexKeys::FromRows) | Source::From(_) => {}
            }
        }
        Selector::TokenPool {
            heads,
            d,
            pool,
            rule: PoolRule::Learned { .. },
            ..
        } => {
            at(Need::TokenPool {
                heads: *heads,
                d: *d,
                pool: *pool,
            });
            at(Need::KeyLayerNorm);
        }
        Selector::TokenPool {
            heads,
            d,
            pool,
            rule: PoolRule::Mean { rope },
            ..
        } => at(Need::MeanPool {
            heads: *heads,
            d: *d,
            pool: *pool,
            rope: rope.dims,
        }),
        Selector::BlockMax {
            heads,
            d,
            top_k,
            block,
            local,
            rope,
        } => at(Need::BlockMax {
            heads: *heads,
            d: *d,
            top_k: *top_k,
            block: *block,
            local: *local,
            rope: rope.dims,
        }),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Act, Ffn, Gqa, LayerSpec, LogitEpilogue, Mixer, ModelSpec, Moe, Need, OutGate, Residual,
        Rope, RopeMode, Router, Score, ShortConv, needs,
    };

    /// One Muse-shaped layer: a Proj-gated GQA mixer whose output norm is
    /// `o_norm`'s to have or not, over latent experts carrying the norms
    /// `pre` and `post`, joined by `residual`, with the model-wide `logit`
    /// epilogue, beside nothing else.
    fn muse_spec(
        o_norm: bool,
        pre: bool,
        post: Option<f32>,
        residual: Residual,
        logit: Option<LogitEpilogue>,
    ) -> ModelSpec {
        ModelSpec {
            arch: crate::Arch::MuseSpark,
            hidden: 8,
            vocab: 2,
            ctx_train: 8,
            rms_eps: 1e-5,
            layers: vec![LayerSpec {
                mixer: Mixer::Gqa(Gqa {
                    heads: 2,
                    kv_heads: 1,
                    head_dim: 64,
                    value_dim: 64,
                    rope: Rope {
                        mode: RopeMode::NormTail,
                        dims: 64,
                        base: 0.0,
                        yarn: None,
                        factors: false,
                    },
                    qk_norm: true,
                    o_norm,
                    out_gate: OutGate::Proj,
                    select: None,
                    window: None,
                    sinks: false,
                    value_scale: None,
                }),
                ffn: Ffn::Moe(Moe {
                    experts: 6,
                    top_k: 2,
                    expert_ff: 24,
                    latent: Some(16),
                    pre_exp_norm: pre,
                    post_exp_norm: post,
                    act: Act::SwiGlu { limit: None },
                    router: Router {
                        score: Score::Sigmoid,
                        bias: true,
                        norm: crate::RouterNorm::SelectedSum,
                        scale: 1.0,
                        hash: false,
                    },
                    shared: None,
                }),
                residual,
                extras: Vec::new(),
            }],
            mtp: Vec::new(),
            hc: None,
            engram: None,
            encoder: None,
            logit_epilogue: logit,
            chat: crate::ChatSpec {
                pre: String::new(),
                template: None,
                tools: None,
                reasoning: None,
            },
        }
    }

    /// One GQA layer of head 64 with `rope`'s mode and `qk_norm`, beside
    /// nothing else.
    fn spec(layers: &[(RopeMode, bool)]) -> ModelSpec {
        ModelSpec {
            arch: crate::Arch::MuseSpark,
            hidden: 8,
            vocab: 2,
            ctx_train: 8,
            rms_eps: 1e-5,
            layers: layers
                .iter()
                .map(|&(mode, qk_norm)| LayerSpec {
                    mixer: Mixer::Gqa(Gqa {
                        heads: 2,
                        kv_heads: 1,
                        head_dim: 64,
                        value_dim: 64,
                        rope: Rope {
                            mode,
                            dims: if mode == RopeMode::None { 0 } else { 64 },
                            base: 0.0,
                            yarn: None,
                            factors: false,
                        },
                        qk_norm,
                        o_norm: false,
                        out_gate: OutGate::None,
                        select: None,
                        window: None,
                        sinks: false,
                        value_scale: None,
                    }),
                    ffn: Ffn::Dense {
                        ff: 0,
                        act: Act::SwiGlu { limit: None },
                    },
                    residual: Residual::Plain,
                    extras: Vec::new(),
                })
                .collect(),
            mtp: Vec::new(),
            hc: None,
            engram: None,
            encoder: None,
            logit_epilogue: None,
            chat: crate::ChatSpec {
                pre: String::new(),
                template: None,
                tools: None,
                reasoning: None,
            },
        }
    }

    /// A NoPE layer with a per-head q/k norm still needs it: the norm is not
    /// the rope's passenger, and a reader that turns rope off must not hide
    /// it. A NoPE layer with no norm needs neither. FAIL-first: with the need
    /// gated on `mode != None` alone, the second row of the first assertion
    /// fails; with the display naming the rope, the text assertion fails.
    #[test]
    fn a_nope_layer_with_a_qk_norm_still_needs_it() {
        let got = needs(&spec(&[
            (RopeMode::NormTail, true),
            (RopeMode::None, true),
            (RopeMode::None, false),
        ]));
        for (l, want) in [
            (
                Some(0),
                "per-head QK norm plus rope: head 64, NORM tail, 64 of 64 dims",
            ),
            (Some(1), "per-head QK norm, no rope: head 64"),
        ] {
            let text = got
                .iter()
                .find(|(n, layer)| *layer == l && matches!(n, Need::QkRope { .. }))
                .map(|(n, _)| n.to_string());
            assert_eq!(text.as_deref(), Some(want), "layer {l:?}");
        }
        assert!(
            !got.iter()
                .any(|(n, l)| *l == Some(2) && matches!(n, Need::QkRope { .. })),
            "a NoPE layer with no q/k norm needs no QkRope"
        );
    }

    /// Every new fact needs its row, spelled by its own display: the normed
    /// attention output beside the gate, the expert norms in the latent, the
    /// alpha/beta residual and the head's logit epilogue. FAIL-first: with
    /// any emission removed, its text row fails; with the o-norm's need
    /// riding the gate's instead of its own field, the second test fails.
    #[test]
    fn the_normed_gate_experts_residual_and_epilogue_need_their_rows() {
        let got = needs(&muse_spec(
            true,
            true,
            Some(1e-8),
            Residual::AlphaBeta { eps: 1e-8 },
            Some(LogitEpilogue {
                scale: 0.1767767,
                softcap: 20.0,
            }),
        ));
        for (l, want) in [
            (
                Some(0),
                "a gainless per-head RMS norm on the attention output: head 64",
            ),
            (
                Some(0),
                "the alpha/beta residual mix, each sublayer's output normed at 0.00000001 first",
            ),
            (
                Some(0),
                "a weighted RMS norm on the routed experts' input (ffn_pre_exp_norm)",
            ),
            (
                Some(0),
                "the routed experts' output scaled by ffn_post_exp_norm, then normed at 0.00000001",
            ),
            (
                None,
                "logit epilogue: scale 0.1767767, then the tanh softcap at 20",
            ),
        ] {
            assert!(
                got.iter()
                    .any(|(n, layer)| *layer == l && n.to_string() == want),
                "{want}: not in the needs"
            );
        }
    }

    /// None of it without the facts: a Proj-gated layer with no output norm,
    /// a latent mixture with no expert norms, the plain residual and no
    /// epilogue need none of the new rows. The gate without the norm is the
    /// cut that would hide one: the norm is not the gate's passenger, as the
    /// q/k norm is not the rope's.
    #[test]
    fn a_gated_layer_without_the_norms_needs_none_of_their_rows() {
        let got = needs(&muse_spec(false, false, None, Residual::Plain, None));
        assert!(
            !got.iter().any(|(n, _)| matches!(
                n,
                Need::OutNorm { .. }
                    | Need::PreExpNorm
                    | Need::PostExpNorm { .. }
                    | Need::AlphaBeta { .. }
                    | Need::LogitEpilogue { .. }
            )),
            "{got:?}"
        );
    }

    /// An LFM3-shaped pair of layers: a short-conv layer, then a GQA layer of head 128 whose rope
    /// carries `factors`, whose gate is `gate`, both joined as `residual`, over a dense SwiGLU
    /// block each.
    fn lfm3_spec(factors: bool, gate: OutGate, residual: Residual) -> ModelSpec {
        let dense = || Ffn::Dense {
            ff: 32,
            act: Act::SwiGlu { limit: None },
        };
        let mut s = spec(&[]);
        s.arch = crate::Arch::Lfm3;
        s.layers = vec![
            LayerSpec {
                mixer: Mixer::ShortConv(ShortConv { taps: 7 }),
                ffn: dense(),
                residual,
                extras: Vec::new(),
            },
            LayerSpec {
                mixer: Mixer::Gqa(Gqa {
                    heads: 4,
                    kv_heads: 2,
                    head_dim: 128,
                    value_dim: 128,
                    rope: Rope {
                        mode: RopeMode::Neox,
                        dims: 128,
                        base: 1e6,
                        yarn: None,
                        factors,
                    },
                    qk_norm: true,
                    o_norm: false,
                    out_gate: gate,
                    select: None,
                    window: None,
                    sinks: false,
                    value_scale: None,
                }),
                ffn: dense(),
                residual,
                extras: Vec::new(),
            },
        ];
        s
    }

    /// The short conv, the rope factors, the per-head gate and the sandwich residual each need a
    /// row of their own, on the layer that carries them. FAIL-first: with any emission removed,
    /// its row fails; with the factors riding the rope's row (a `QkRope` that names no factors),
    /// a program that runs the plain rope would cover the factored one and the second assertion
    /// fails.
    #[test]
    fn the_lfm3_parts_need_their_rows() {
        let got = needs(&lfm3_spec(
            true,
            OutGate::ProjHead,
            Residual::Sandwich {
                eps: 1e-5,
                add: [1.0, 1.0],
            },
        ));
        for (l, want) in [
            (0, "gated short-conv mixer: 7 taps"),
            (
                0,
                "sandwich norms: each sublayer's output weight-normed at 0.00001, then added scaled by [1.0, 1.0]",
            ),
            (1, "rope frequency factors (rope_freqs.weight)"),
            (
                1,
                "attention output gate: sigmoid of its own projection, one gate per head",
            ),
            (
                1,
                "sandwich norms: each sublayer's output weight-normed at 0.00001, then added scaled by [1.0, 1.0]",
            ),
        ] {
            assert!(
                got.iter()
                    .any(|(n, layer)| *layer == Some(l) && n.to_string() == want),
                "{want}: not in the needs of layer {l}: {got:?}"
            );
        }
        assert!(
            got.iter().any(|(n, l)| *l == Some(1)
                && matches!(
                    n,
                    Need::QkRope {
                        head: 128,
                        dims: 128,
                        mode: RopeMode::Neox,
                        ..
                    }
                )),
            "{got:?}"
        );
        assert!(
            !got.iter().any(|(n, _)| matches!(n, Need::OutGateProj)),
            "the per-head gate is not the per-channel one: {got:?}"
        );
    }

    /// None of it without the facts: no factors, the per-channel gate and the plain residual
    /// need none of the rows, and a short-conv layer needs no attention row.
    #[test]
    fn a_layer_without_the_lfm3_parts_needs_none_of_their_rows() {
        let got = needs(&lfm3_spec(false, OutGate::Proj, Residual::Plain));
        assert!(
            !got.iter().any(|(n, _)| matches!(
                n,
                Need::RopeFactors | Need::OutGateProjHead | Need::Sandwich { .. }
            )),
            "{got:?}"
        );
        assert!(got.iter().any(|(n, _)| matches!(n, Need::OutGateProj)));
        assert!(
            !got.iter()
                .any(|(n, l)| *l == Some(0) && matches!(n, Need::Gqa { .. } | Need::QkRope { .. })),
            "{got:?}"
        );
    }
}
