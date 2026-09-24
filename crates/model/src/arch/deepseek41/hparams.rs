//! Every hyperparameter the DeepSeek-V4.1 decode step reads and the kind of
//! every layer, resolved once at load from the first shard's metadata and the
//! tensors the file holds. This is the one reader of `<arch>.*` keys for this
//! architecture.
//!
//! Nothing here has a default: a key the file lacks is an error naming the key.
//! Three quantities have no key in the file and come from where ik takes them:
//! the vocabulary size from the token list, dense-ness from the router's
//! presence, and the layer kinds from the tensors each layer carries. A value
//! ik derives rather than reads is computed in one function whose comment
//! names ik's line.
//!
//! ik line numbers are those of the tree the V4.1 oracle sets were built from
//! (`tools/ref/models/deepseek41.sh`'s `IK`).

use gguf::Split;

use super::names;
use crate::arch::{
    meta_arr, meta_bool, meta_f32, meta_str, meta_u64, meta_usize, metadata, n_vocab,
};
use crate::placement::PlacementError;

/// ik's `LLM_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS` (llama-hparams.h:18).
const IK_SQRT_SOFTPLUS: u64 = 4;

/// The hyperparameters of one V4.1 file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`.
    pub n_layer: usize,
    /// `embedding_length` — the width of one hyper-connection stream.
    pub n_embd: usize,
    /// `attention.head_count`.
    pub n_head: usize,
    /// `attention.head_count_kv`.
    pub n_head_kv: usize,
    /// `attention.key_length`, which `attention.value_length` equals: one
    /// latent is every head's key and value.
    pub head_dim: usize,
    /// `attention.q_lora_rank` — the query latent's width.
    pub q_lora_rank: usize,
    /// `attention.output_group_count` — the blocks of `attn_output_a`'s diagonal.
    pub o_groups: usize,
    /// `attention.output_lora_rank` — one group's output width.
    pub o_lora_rank: usize,
    /// `rope.dimension_count` — the rotated tail of each head.
    pub rope_dims: usize,
    /// `attention.sliding_window` — the latest positions every layer attends.
    pub window: usize,
    /// `attention.layer_norm_rms_epsilon` — every RMS norm's, the
    /// hyper-connections' flattened input included.
    pub rms_eps: f32,
    /// The vocabulary size: `vocab_size` when the file carries it, else the
    /// length of `tokenizer.ggml.tokens` (ik's rule, llama-hparams.cpp:155).
    pub n_vocab: usize,
    /// `context_length` — the context the model was trained to, not the one
    /// this engine allocates.
    pub n_ctx_train: usize,
    /// The first ratio a layer compresses at, in layer order — ik's
    /// `dsv4_csa_ratio`, a segment and not a role.
    pub csa_ratio: u32,
    /// The next distinct ratio (the first again when there is one) — ik's
    /// `dsv4_hca_ratio`.
    pub hca_ratio: u32,
    /// The indexer's shape and selection width.
    pub indexer: Indexer,
    /// The hyper-connections' stream count and Sinkhorn constants.
    pub hc: HyperConnections,
    /// The router and the experts.
    pub experts: Experts,
    /// The engram dimensions.
    pub engram: Engram,
    /// One entry per layer, in layer order.
    pub layers: Vec<LayerKind>,
}

/// The lightning indexer that picks the compressed rows a layer attends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Indexer {
    /// `attention.indexer.head_count`.
    pub n_head: usize,
    /// `attention.indexer.key_length` — one index key's width.
    pub head_dim: usize,
    /// `attention.indexer.top_k` — the compressed rows one query keeps, unless
    /// [`Hparams::with_indexer_top_k`] replaced it.
    pub top_k: usize,
}

/// The hyper-connections every sublayer folds its input from and mixes its
/// output into.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HyperConnections {
    /// `hyper_connection.count` — the streams the residual is carried in.
    pub streams: usize,
    /// `hyper_connection.sinkhorn_iterations` — the rounds that make the
    /// combine mix doubly stochastic.
    pub sinkhorn_iters: usize,
    /// `hyper_connection.epsilon` — the floor on the mixes (the pre weights
    /// and the Sinkhorn normalization), not a norm's epsilon
    /// (build_deepseek4.cpp:654, 685).
    pub eps: f32,
}

/// The router's score function, `expert_gating_func`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Score {
    /// `sqrt(softplus(logit))` — the only one ik's V4 loader accepts
    /// (llama-hparams.cpp:2219-2221).
    SqrtSoftplus,
}

/// The mixture of experts every routed layer runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Experts {
    /// `expert_count` — routed experts per layer.
    pub n_expert: usize,
    /// `expert_used_count` — routed experts one token runs per layer.
    pub n_used: usize,
    /// `expert_shared_count`.
    pub n_shared: usize,
    /// `expert_feed_forward_length` — one expert's hidden width.
    pub ff: usize,
    /// `expert_weights_scale` — the factor on the chosen experts' weights.
    pub routed_scale: f32,
    /// `expert_weights_norm` — whether the chosen weights are renormalized to
    /// sum to one before the scale.
    pub weights_norm: bool,
    /// `expert_gating_func`.
    pub score: Score,
    /// The layers before the first routed one, which run a dense FFN: the
    /// router's presence decides, not a key.
    pub dense_lead: usize,
    /// `hash_layer_count` — the leading layers that route by a token table
    /// instead of the router's scores.
    pub hash_layers: usize,
}

/// The engram dimensions the step's chain needs; the hash constants are
/// `crates/engram`'s.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Engram {
    /// `engram.layer_ids` — the layers that carry a table, in site order.
    pub layer_ids: Vec<usize>,
    /// `engram.head_count` — hash heads per n-gram order.
    pub n_head: usize,
    /// `engram.max_ngram_size` — the longest n-gram hashed.
    pub max_ngram: usize,
    /// `engram.key_length` — values in one table row.
    pub key_length: usize,
}

impl Engram {
    /// Rows one site gathers per token: `(max_ngram − 1) · n_head`, ik's
    /// `n_cols` (build_deepseek4.cpp:996).
    pub fn rows_per_token(&self) -> usize {
        (self.max_ngram - 1) * self.n_head
    }

    /// The width of the gathered rows `engram_wkv` projects:
    /// `rows_per_token · key_length` (build_deepseek4.cpp:1005-1006).
    pub fn embedding_width(&self) -> usize {
        self.rows_per_token() * self.key_length
    }
}

/// One rope as ik's graph passes it to `ggml_rope_ext` for a layer
/// (build_deepseek4.cpp:1092-1100).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    /// The base of θ: `rope.freq_base` without YaRN,
    /// `attention.compress_rope_freq_base` under it.
    pub base: f32,
    /// `1 / rope.scaling.factor` under YaRN, 1 without it.
    pub freq_scale: f32,
    /// YaRN's extrapolation mix: 1 under YaRN, 0 (plain rope) without it.
    pub ext_factor: f32,
    /// The magnitude ik passes: `1/(1 + 0.1·ln(1/freq_scale))` under YaRN,
    /// which `rope_yarn` cancels; 1 without it.
    pub attn_factor: f32,
    /// `rope.scaling.yarn_beta_fast` under YaRN, 0 without it.
    pub beta_fast: f32,
    /// `rope.scaling.yarn_beta_slow` under YaRN, 0 without it.
    pub beta_slow: f32,
    /// YaRN's original context; 0 without YaRN.
    pub n_ctx_orig: u32,
}

/// A compressed stream a layer attends besides its window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stream {
    /// `attention.compress_ratios` at this layer: tokens pooled into one row.
    pub ratio: u32,
    /// The layer whose compressor writes the rows this layer reads; itself
    /// when it owns the compressor.
    pub kv_source: usize,
    /// The layer whose index keys the top-k scores; itself when it owns them.
    pub index_key_source: usize,
    /// The layer whose top-k selection this layer attends over; itself when
    /// it runs the indexer.
    pub topk_source: usize,
}

/// The compressor a source layer owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Compressor {
    /// Whether it pools with scores (`attn_compressor_gate`); a compressor
    /// that pools one token per row has none.
    pub gated: bool,
}

/// What one layer is, from the tensors it carries: the per-layer table the
/// step reads instead of layer numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerKind {
    /// The compressed stream it attends, or `None` on a window-only layer.
    pub stream: Option<Stream>,
    /// The compressor it owns (`attn_compressor_kv`).
    pub compressor: Option<Compressor>,
    /// It owns index keys (`indexer.attn_k`).
    pub index_keys: bool,
    /// It runs the indexer's top-k (`indexer.attn_q_b`).
    pub indexer: bool,
    /// Its engram site, an index into [`Engram::layer_ids`] (`engram_embd`).
    pub engram: Option<usize>,
    /// It routes to experts (`ffn_gate_inp`).
    pub routed: bool,
    /// Its rope: YaRN on a layer with a stream, plain on a window-only one.
    pub rope: Rope,
    /// `swiglu_clamp_exp` at this layer — the routed experts' SwiGLU clamp.
    pub swiglu_limit: f32,
    /// `swiglu_clamp_shexp` at this layer — the shared expert's.
    pub swiglu_limit_shared: f32,
}

impl LayerKind {
    /// `attention.compress_ratios` at this layer: 0 on a window-only layer.
    pub fn ratio(&self) -> u32 {
        self.stream.map_or(0, |s| s.ratio)
    }
}

impl Hparams {
    /// Everything from `split`'s headers; a missing or malformed key, or a
    /// layer table ik's loader would refuse, is an error naming the key or the
    /// tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let n_layer = meta_usize(split, "block_count")?;
        let head_dim = meta_usize(split, "attention.key_length")?;
        let value_length = meta_usize(split, "attention.value_length")?;
        if value_length != head_dim {
            return Err(metadata(
                split,
                "attention.value_length",
                format!("is {value_length}, the key length is {head_dim}: the latent is both"),
            ));
        }
        let rope_dims = meta_usize(split, "rope.dimension_count")?;
        if rope_dims > head_dim {
            return Err(metadata(
                split,
                "rope.dimension_count",
                format!("is {rope_dims}, more than a head's {head_dim}"),
            ));
        }
        let ropes = Ropes::read(split)?;
        let tables = LayerTables {
            ratios: compress_ratios(split, n_layer)?,
            swiglu: per_layer_f32(split, "swiglu_clamp_exp", n_layer)?,
            swiglu_shared: per_layer_f32(split, "swiglu_clamp_shexp", n_layer)?,
        };
        let engram = Engram::read(split, n_layer)?;
        let carries: Vec<Carries> = (0..n_layer).map(|l| Carries::read(split, l)).collect();
        let walk = walk_streams(
            &split.arch_key("attention.compress_ratios"),
            &carries,
            &tables.ratios,
        )?;
        let layers = layer_kinds(split, &carries, &walk.streams, &tables, &engram, &ropes)?;
        let dense_lead = dense_lead(split, &layers)?;
        Ok(Hparams {
            n_layer,
            n_embd: meta_usize(split, "embedding_length")?,
            n_head: meta_usize(split, "attention.head_count")?,
            n_head_kv: meta_usize(split, "attention.head_count_kv")?,
            head_dim,
            q_lora_rank: meta_usize(split, "attention.q_lora_rank")?,
            o_groups: meta_usize(split, "attention.output_group_count")?,
            o_lora_rank: meta_usize(split, "attention.output_lora_rank")?,
            rope_dims,
            window: meta_usize(split, "attention.sliding_window")?,
            rms_eps: meta_f32(split, "attention.layer_norm_rms_epsilon")?,
            n_vocab: n_vocab(split)?,
            n_ctx_train: meta_usize(split, "context_length")?,
            csa_ratio: walk.csa_ratio,
            hca_ratio: walk.hca_ratio,
            indexer: Indexer {
                n_head: meta_usize(split, "attention.indexer.head_count")?,
                head_dim: meta_usize(split, "attention.indexer.key_length")?,
                top_k: meta_usize(split, "attention.indexer.top_k")?,
            },
            hc: HyperConnections {
                streams: meta_usize(split, "hyper_connection.count")?,
                sinkhorn_iters: meta_usize(split, "hyper_connection.sinkhorn_iterations")?,
                eps: meta_f32(split, "hyper_connection.epsilon")?,
            },
            experts: Experts::read(split, dense_lead)?,
            engram,
            layers,
        })
    }

    /// The same model with the indexer keeping `top_k` rows: ik's
    /// `--override-kv <arch>.attention.indexer.top_k=int:<top_k>`, which the
    /// loader applies to that one key (llama-model-loader.cpp:824-829).
    #[must_use]
    pub fn with_indexer_top_k(mut self, top_k: usize) -> Hparams {
        self.indexer.top_k = top_k;
        self
    }
}

/// The two ropes a layer can have (build_deepseek4.cpp:1092-1100): a layer
/// without a stream rotates at `rope.freq_base` with no scaling, the others at
/// `attention.compress_rope_freq_base` under the file's YaRN.
struct Ropes {
    window: Rope,
    compressed: Rope,
}

impl Ropes {
    fn read(split: &Split) -> Result<Ropes, PlacementError> {
        let scaling = meta_str(split, "rope.scaling.type")?;
        if scaling != "yarn" {
            return Err(metadata(
                split,
                "rope.scaling.type",
                format!("is {scaling:?}; the compressed layers' rope is YaRN"),
            ));
        }
        let factor = meta_f32(split, "rope.scaling.factor")?;
        if factor.is_nan() || factor <= 0.0 {
            return Err(metadata(
                split,
                "rope.scaling.factor",
                format!("is {factor}, not a scaling factor"),
            ));
        }
        // ik's `rope_freq_scale_train` (llama-hparams.cpp:216); a yarn scaling
        // type turns the extrapolation mix on (llama.cpp:9059-9060).
        let freq_scale = 1.0 / factor;
        let ext_factor = 1.0;
        let original_ctx = meta_u64(split, "rope.scaling.original_context_length")?;
        let n_ctx_orig = u32::try_from(original_ctx).map_err(|_| {
            metadata(
                split,
                "rope.scaling.original_context_length",
                format!("{original_ctx} does not fit u32"),
            )
        })?;
        Ok(Ropes {
            window: Rope {
                base: meta_f32(split, "rope.freq_base")?,
                freq_scale: 1.0,
                ext_factor: 0.0,
                attn_factor: rope_attn_factor(1.0, 0.0),
                beta_fast: 0.0,
                beta_slow: 0.0,
                n_ctx_orig: 0,
            },
            compressed: Rope {
                base: meta_f32(split, "attention.compress_rope_freq_base")?,
                freq_scale,
                ext_factor,
                attn_factor: rope_attn_factor(freq_scale, ext_factor),
                beta_fast: meta_f32(split, "rope.scaling.yarn_beta_fast")?,
                beta_slow: meta_f32(split, "rope.scaling.yarn_beta_slow")?,
                n_ctx_orig,
            },
        })
    }
}

/// ik's `dsv4_rope_attn_factor` (build_deepseek4.cpp:23-29), in its f32 order:
/// under YaRN the inverse of the magnitude `rope_yarn` multiplies in
/// (ggml.c:20802), so the rotation keeps its length; 1 without YaRN.
fn rope_attn_factor(freq_scale: f32, ext_factor: f32) -> f32 {
    if ext_factor == 0.0 {
        return 1.0;
    }
    1.0 / (1.0 + 0.1 * (1.0 / freq_scale).ln())
}

impl Experts {
    fn read(split: &Split, dense_lead: usize) -> Result<Experts, PlacementError> {
        let gating = meta_u64(split, "expert_gating_func")?;
        if gating != IK_SQRT_SOFTPLUS {
            return Err(metadata(
                split,
                "expert_gating_func",
                format!("is {gating}; the V4.1 router scores with sqrt-softplus"),
            ));
        }
        Ok(Experts {
            n_expert: meta_usize(split, "expert_count")?,
            n_used: meta_usize(split, "expert_used_count")?,
            n_shared: meta_usize(split, "expert_shared_count")?,
            ff: meta_usize(split, "expert_feed_forward_length")?,
            routed_scale: meta_f32(split, "expert_weights_scale")?,
            weights_norm: meta_bool(split, "expert_weights_norm")?,
            score: Score::SqrtSoftplus,
            dense_lead,
            hash_layers: meta_usize(split, "hash_layer_count")?,
        })
    }
}

impl Engram {
    /// The keys ik's loader requires of a V4.1 file (llama-hparams.cpp:2075-2096),
    /// without the hash constants.
    fn read(split: &Split, n_layer: usize) -> Result<Engram, PlacementError> {
        let key = "engram.layer_ids";
        let mut layer_ids = Vec::new();
        for (i, v) in meta_arr(split, key)?.iter().enumerate() {
            let layer = v
                .as_unsigned()
                .and_then(|l| usize::try_from(l).ok())
                .filter(|&l| l < n_layer)
                .ok_or_else(|| {
                    metadata(
                        split,
                        key,
                        format!("entry {i} is not a layer below {n_layer}"),
                    )
                })?;
            if layer_ids.contains(&layer) {
                return Err(metadata(split, key, format!("lists layer {layer} twice")));
            }
            layer_ids.push(layer);
        }
        if layer_ids.is_empty() {
            return Err(metadata(split, key, "is empty"));
        }
        let n_head = meta_usize(split, "engram.head_count")?;
        if n_head == 0 {
            return Err(metadata(split, "engram.head_count", "is 0"));
        }
        let max_ngram = meta_usize(split, "engram.max_ngram_size")?;
        if max_ngram < 2 {
            return Err(metadata(
                split,
                "engram.max_ngram_size",
                format!("is {max_ngram}; the hash needs a 2-gram"),
            ));
        }
        Ok(Engram {
            layer_ids,
            n_head,
            max_ngram,
            key_length: meta_usize(split, "engram.key_length")?,
        })
    }
}

/// The per-layer metadata tables the layer walk reads.
struct LayerTables {
    ratios: Vec<u32>,
    swiglu: Vec<f32>,
    swiglu_shared: Vec<f32>,
}

/// The tensors of one layer that the stream walk reads.
#[derive(Clone, Copy, Debug, Default)]
struct Carries {
    /// `attn_compressor_kv`.
    compressor: bool,
    /// `attn_compressor_gate`.
    gate: bool,
    /// `indexer.attn_k`.
    index_keys: bool,
    /// `indexer.attn_q_b`.
    indexer: bool,
}

impl Carries {
    fn read(split: &Split, l: usize) -> Carries {
        let has = |name: String| split.find(&name).is_some();
        Carries {
            compressor: has(names::attn_compressor_kv(l)),
            gate: has(names::attn_compressor_gate(l)),
            index_keys: has(names::indexer_attn_k(l)),
            indexer: has(names::indexer_attn_q_b(l)),
        }
    }
}

/// What the stream walk resolves.
struct Walk {
    /// Per layer, the stream it attends.
    streams: Vec<Option<Stream>>,
    /// [`Hparams::csa_ratio`].
    csa_ratio: u32,
    /// [`Hparams::hca_ratio`].
    hca_ratio: u32,
}

/// ik's stream walk (llama-hparams.cpp:2147-2176) over what each layer carries
/// and its ratio in `ratios_key`. A layer whose ratio is not 0 reads the
/// stream of the last compressor at or before it, the index keys of the last
/// `indexer.attn_k` and the top-k of the last `indexer.attn_q_b`. ik's four
/// checks refuse the table at the layer that breaks one: the three sources
/// exist (:2159-2161), the stream's source compresses at the reader's ratio
/// (:2162-2165), a compressor that pools more than one token per row carries
/// its gate (:2166-2168), and the ratios above 0 take at most two values
/// (:2172-2175). Two refusals are ours: a compressor on a layer whose ratio is
/// 0 (ik lets it pass until a later layer reads it at a mismatched ratio), and
/// a table with no compressed layer (ik makes up a ratio of 1, :2179-2180).
fn walk_streams(
    ratios_key: &str,
    carries: &[Carries],
    ratios: &[u32],
) -> Result<Walk, PlacementError> {
    let refuse = |detail: String| PlacementError::Metadata {
        key: ratios_key.to_string(),
        detail,
    };
    let (mut last_kv, mut last_key, mut last_idx) = (None, None, None);
    let mut segments: Vec<u32> = Vec::with_capacity(2);
    let mut streams = Vec::with_capacity(ratios.len());
    for (l, (c, &ratio)) in carries.iter().zip(ratios).enumerate() {
        if c.compressor {
            last_kv = Some(l);
        }
        if c.index_keys {
            last_key = Some(l);
        }
        if c.indexer {
            last_idx = Some(l);
        }
        if ratio == 0 {
            if c.compressor {
                return Err(refuse(format!(
                    "is 0 at layer {l}, which carries a compressor"
                )));
            }
            streams.push(None);
            continue;
        }
        let no_source = |what: &str| {
            refuse(format!(
                "is {ratio} at layer {l}, and no layer up to it carries {what}"
            ))
        };
        let kv_source = last_kv.ok_or_else(|| no_source("attn_compressor_kv"))?;
        let index_key_source = last_key.ok_or_else(|| no_source("indexer.attn_k"))?;
        let topk_source = last_idx.ok_or_else(|| no_source("indexer.attn_q_b"))?;
        let source_ratio = ratios[kv_source];
        if source_ratio != ratio {
            return Err(refuse(format!(
                "is {ratio} at layer {l}, which reads layer {kv_source}'s stream, compressed at {source_ratio}"
            )));
        }
        if c.compressor && !c.gate && ratio != 1 {
            return Err(PlacementError::Tensor {
                name: names::attn_compressor_gate(l),
                detail: format!("is not in the file, and layer {l} pools {ratio} tokens per row"),
            });
        }
        if !segments.contains(&ratio) {
            if let [csa, hca] = segments[..] {
                return Err(refuse(format!(
                    "is {ratio} at layer {l}, a third ratio after {csa} and {hca}"
                )));
            }
            segments.push(ratio);
        }
        streams.push(Some(Stream {
            ratio,
            kv_source,
            index_key_source,
            topk_source,
        }));
    }
    let (csa_ratio, hca_ratio) = match segments[..] {
        [csa] => (csa, csa),
        [csa, hca] => (csa, hca),
        _ => return Err(refuse("is 0 on every layer".to_string())),
    };
    Ok(Walk {
        streams,
        csa_ratio,
        hca_ratio,
    })
}

/// Every layer's kind: its stream from the walk, what it carries, its engram
/// site (build_deepseek4.cpp:1543-1545) and whether it routes.
fn layer_kinds(
    split: &Split,
    carries: &[Carries],
    streams: &[Option<Stream>],
    tables: &LayerTables,
    engram: &Engram,
    ropes: &Ropes,
) -> Result<Vec<LayerKind>, PlacementError> {
    let mut layers = Vec::with_capacity(carries.len());
    for (l, (c, &stream)) in carries.iter().zip(streams).enumerate() {
        layers.push(LayerKind {
            stream,
            compressor: c.compressor.then_some(Compressor { gated: c.gate }),
            index_keys: c.index_keys,
            indexer: c.indexer,
            engram: engram_site(split, engram, l)?,
            routed: split.find(&names::ffn_gate_inp(l)).is_some(),
            rope: if stream.is_some() {
                ropes.compressed
            } else {
                ropes.window
            },
            swiglu_limit: tables.swiglu[l],
            swiglu_limit_shared: tables.swiglu_shared[l],
        });
    }
    Ok(layers)
}

/// Layer `l`'s engram site: it carries a table exactly when `engram.layer_ids`
/// lists it (ik asserts the one direction, build_deepseek4.cpp:1545).
fn engram_site(split: &Split, engram: &Engram, l: usize) -> Result<Option<usize>, PlacementError> {
    let table = names::engram_embd(l);
    let site = engram.layer_ids.iter().position(|&e| e == l);
    match (split.find(&table).is_some(), site) {
        (true, Some(s)) => Ok(Some(s)),
        (false, None) => Ok(None),
        (true, None) => Err(metadata(
            split,
            "engram.layer_ids",
            format!("does not list layer {l}, which carries {table}"),
        )),
        (false, Some(_)) => Err(PlacementError::Tensor {
            name: table,
            detail: format!("is not in the file, and engram.layer_ids lists layer {l}"),
        }),
    }
}

/// The layers before the first routed one: ik's `n_layer_dense_lead`, which
/// its graph runs as dense FFNs (build_deepseek4.cpp:1571). The file need not
/// carry `leading_dense_block_count` (ik reads it as optional,
/// llama-hparams.cpp:1971): the router's presence decides, the dense layers
/// must lead, and a key the file does carry must agree.
fn dense_lead(split: &Split, layers: &[LayerKind]) -> Result<usize, PlacementError> {
    let lead = layers.iter().take_while(|k| !k.routed).count();
    if let Some(l) = layers.iter().skip(lead).position(|k| !k.routed) {
        return Err(PlacementError::Tensor {
            name: names::ffn_gate_inp(lead + l),
            detail: "is not in the file, and an earlier layer is routed".to_string(),
        });
    }
    let key = "leading_dense_block_count";
    if split.value(&split.arch_key(key)).is_some() {
        let v = meta_usize(split, key)?;
        if v != lead {
            return Err(metadata(
                split,
                key,
                format!("is {v}, and {lead} layers lead without a router"),
            ));
        }
    }
    Ok(lead)
}

/// `attention.compress_ratios`, the first `n_layer` of them: the file also
/// lists layers it does not carry, and ik takes the first `block_count`
/// (llama-hparams.cpp:2115-2121).
fn compress_ratios(split: &Split, n_layer: usize) -> Result<Vec<u32>, PlacementError> {
    let key = "attention.compress_ratios";
    let items = meta_arr(split, key)?;
    if items.len() < n_layer {
        return Err(metadata(
            split,
            key,
            format!("has {} values for {n_layer} layers", items.len()),
        ));
    }
    items[..n_layer]
        .iter()
        .enumerate()
        .map(|(l, v)| {
            v.as_unsigned()
                .and_then(|r| u32::try_from(r).ok())
                .ok_or_else(|| metadata(split, key, format!("has no ratio for layer {l}")))
        })
        .collect()
}

/// `<architecture>.<suffix>` per layer the way ik's `get_key_or_arr` reads it
/// (llama-model-loader.cpp:845-882): an array of exactly one value per layer,
/// or one value for every layer.
fn per_layer_f32(split: &Split, suffix: &str, n_layer: usize) -> Result<Vec<f32>, PlacementError> {
    let Some(items) = split.arch_get_arr(suffix) else {
        return Ok(vec![meta_f32(split, suffix)?; n_layer]);
    };
    if items.len() != n_layer {
        return Err(metadata(
            split,
            suffix,
            format!("has {} values for {n_layer} layers", items.len()),
        ));
    }
    items
        .iter()
        .enumerate()
        .map(|(l, v)| {
            v.as_f32()
                .ok_or_else(|| metadata(split, suffix, format!("has no float for layer {l}")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATIOS_KEY: &str = "deepseek41.attention.compress_ratios";

    /// The served file's layer table as `just gate-ds41-meta` prints it: ratio
    /// 0 on layers 0-1, 2 on 2-19 and 1 on 20-39; compressors on 2, 8, 14
    /// (gated) and 20; index keys on the same four; indexer queries on 2, 8,
    /// 14, 20, 24, 28, 32 and 36.
    fn served() -> (Vec<Carries>, Vec<u32>) {
        let carries = (0..40)
            .map(|l| Carries {
                compressor: [2, 8, 14, 20].contains(&l),
                gate: [2, 8, 14].contains(&l),
                index_keys: [2, 8, 14, 20].contains(&l),
                indexer: [2, 8, 14, 20, 24, 28, 32, 36].contains(&l),
            })
            .collect();
        let ratios = [[0; 2].as_slice(), &[2; 18], &[1; 20]].concat();
        (carries, ratios)
    }

    /// The served table walks. Broken once per check, it is refused by that
    /// check, and the error names the layer that breaks it (the last case,
    /// no compressed layer, has none to name).
    #[test]
    fn walk_refuses_a_broken_table_at_its_layer() {
        let (carries, ratios) = served();
        walk_streams(RATIOS_KEY, &carries, &ratios).expect("the served table walks");
        type Break = fn(&mut [Carries], &mut [u32]);
        let cases: [(Break, &str); 7] = [
            (
                |_, r| r[1] = 2,
                "at layer 1, and no layer up to it carries attn_compressor_kv",
            ),
            (
                |c, _| c[2].indexer = false,
                "at layer 2, and no layer up to it carries indexer.attn_q_b",
            ),
            (
                |_, r| r[5] = 1,
                "at layer 5, which reads layer 2's stream, compressed at 2",
            ),
            (
                |c, _| c[8].gate = false,
                "and layer 8 pools 2 tokens per row",
            ),
            (
                |c, r| {
                    c[36].compressor = true;
                    c[36].gate = true;
                    r[36..].fill(4);
                },
                "at layer 36, a third ratio after 2 and 1",
            ),
            (
                |c, _| c[0].compressor = true,
                "is 0 at layer 0, which carries a compressor",
            ),
            (
                |c, r| {
                    c.fill(Carries::default());
                    r.fill(0);
                },
                "is 0 on every layer",
            ),
        ];
        for (i, (break_it, expected)) in cases.into_iter().enumerate() {
            let (mut c, mut r) = served();
            break_it(&mut c, &mut r);
            match walk_streams(RATIOS_KEY, &c, &r) {
                Ok(_) => panic!("case {i} walked; expected \"{expected}\""),
                Err(e) => assert!(e.to_string().contains(expected), "case {i}: {e}"),
            }
        }
    }

    /// A broken table that reaches the engine is `ModelError::Metadata` naming
    /// the ratios key, with the walk's own text.
    #[test]
    fn a_broken_table_is_a_model_metadata_error() {
        let (carries, mut ratios) = served();
        ratios[1] = 2;
        let Err(e) = walk_streams(RATIOS_KEY, &carries, &ratios) else {
            panic!("a ratio of 2 at layer 1 walked");
        };
        let text = e.to_string();
        let m = crate::ModelError::from(e);
        assert_eq!(m.to_string(), text, "the conversion keeps the walk's text");
        assert!(
            matches!(&m, crate::ModelError::Metadata { key, .. } if key == RATIOS_KEY),
            "expected ModelError::Metadata for the ratios key, got {m:?}"
        );
    }
}
