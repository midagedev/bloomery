//! Every hyperparameter the DeepSeek-V4.1 and V4 decode steps read and the kind
//! of every layer, resolved once at load from the first shard's metadata and
//! the tensors the file holds. This is the one reader of `<arch>.*` keys for
//! this module, whose two models ([`Model`]) differ in values and layer kinds,
//! not in types: what the chain reads is the table below, never the
//! architecture string.
//!
//! Nothing here has a default: a key the file lacks is an error naming the key.
//! Three quantities have no key in the file and come from where ik takes them:
//! the vocabulary size from the token list, dense-ness from the router's
//! presence, and the layer kinds from the tensors each layer carries. A value
//! ik derives rather than reads is computed in one function whose comment
//! names ik's line. Four values ik sets by architecture (`llama-hparams.h`:
//! `dsv4_shared_streams`, `dsv4_hc_lag`, `dsv4_q_head_norm`, the CSA overlap)
//! are the [`Model`]'s, and each is checked against the tensors that show it.
//!
//! ik line numbers are those of the tree the V4.1 oracle sets were built from
//! (`tools/ref/models/deepseek41.sh`'s `IK`).

use gguf::Split;
use gguf::quant::GgmlType;

use super::names;
use crate::ModelError;
use crate::arch::{
    meta_arr, meta_bool, meta_f32, meta_str, meta_u64, meta_usize, metadata, n_vocab,
};
use crate::placement::PlacementError;

/// ik's `LLM_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS` (llama-hparams.h:18).
const IK_SQRT_SOFTPLUS: u64 = 4;

/// Which model of this module a file holds, by its `general.architecture`
/// ([`crate::arch::deepseek41_model`] reads it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    /// `deepseek41` — DeepSeek-V4.1-Flash: compressed streams shared from
    /// source layers, lagged hyper-connection mixes, engram sites.
    Deepseek41,
    /// `deepseek4` — DeepSeek-V4-Flash: every compressed layer owns its
    /// compressor, the indexed layers their own index-key compressor, a
    /// trained hyper-connection head, per-head query norms, and hash routing on
    /// the leading layers.
    Deepseek4,
}

impl Model {
    /// The file's `general.architecture` string.
    pub fn name(self) -> &'static str {
        match self {
            Model::Deepseek41 => "deepseek41",
            Model::Deepseek4 => "deepseek4",
        }
    }
}

/// The hyperparameters of one V4.1 or V4 file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// The model the file holds.
    pub model: Model,
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
    /// Every query head is RMS-normed, without a gain, before its rope (ik's
    /// `dsv4_q_head_norm`, build_deepseek4.cpp:1107): V4 does, V4.1 does not.
    pub q_head_norm: bool,
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
    /// How the hyper-connection streams become one before the output norm.
    pub collapse: Collapse,
    /// The router and the experts.
    pub experts: Experts,
    /// The engram dimensions; `None` for a file without engram sites.
    pub engram: Option<Engram>,
    /// The types of the rows a step reads out of the file.
    pub rows: RowTypes,
    /// One entry per layer, in layer order.
    pub layers: Vec<LayerKind>,
}

/// The types of the rows a step's host half reads out of the file and hands
/// the card as the file stores them: a token's `token_embd` row and each
/// engram site's table rows. The step image is laid out by them and the card
/// decodes them, so they are read once here with the dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowTypes {
    /// `token_embd`'s type.
    pub token_embd: GgmlType,
    /// The engram tables' type, one for every site; `None` without sites.
    pub engram: Option<GgmlType>,
}

impl RowTypes {
    /// The engram tables' type, or the error that names the file's lack of
    /// engram sites.
    pub fn engram(&self) -> Result<GgmlType, ModelError> {
        self.engram.ok_or_else(|| no_engram_sites(Model::Deepseek4))
    }

    fn read(split: &Split, engram: Option<&Engram>) -> Result<RowTypes, PlacementError> {
        let ty = |name: String| {
            split
                .find(&name)
                .map(|(_, t)| t.ty)
                .ok_or_else(|| PlacementError::Tensor {
                    name,
                    detail: "is not in the file".to_string(),
                })
        };
        let token_embd = ty(names::token_embd())?;
        let Some(engram) = engram else {
            return Ok(RowTypes {
                token_embd,
                engram: None,
            });
        };
        let mut tables = engram
            .layer_ids
            .iter()
            .map(|&l| (l, ty(names::engram_embd(l))));
        let (first, table) = tables.next().ok_or_else(|| PlacementError::Tensor {
            name: names::engram_embd(0),
            detail: "no engram site names a table".to_string(),
        })?;
        let table = table?;
        for (l, t) in tables {
            let t = t?;
            if t != table {
                return Err(PlacementError::Tensor {
                    name: names::engram_embd(l),
                    detail: format!("is {t}, layer {first}'s table is {table}"),
                });
            }
        }
        Ok(RowTypes {
            token_embd,
            engram: Some(table),
        })
    }
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

/// How the hyper-connection streams collapse before the output norm
/// (build_deepseek4.cpp:1531-1540, 1686-1698).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Collapse {
    /// With the last FFN's mix, which no sublayer consumed: every sublayer
    /// computes the mix the next one folds with (ik's `dsv4_hc_lag`, V4.1).
    Lagged,
    /// With the trained head `output_hc_{fn,base,scale}`; every sublayer folds
    /// with its own mix (V4).
    Head,
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

/// A compressed stream a layer attends besides its window, through a top-k
/// selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stream {
    /// `attention.compress_ratios` at this layer: tokens pooled into one row.
    pub ratio: u32,
    /// The layer whose compressor writes the rows this layer reads; itself
    /// when it owns the compressor.
    pub kv_source: usize,
    /// The layer whose index keys the top-k scores; itself when it owns them
    /// (`indexer.attn_k`, or its own [`LayerKind::index_compressor`]).
    pub index_key_source: usize,
    /// The layer whose top-k selection this layer attends over; itself when
    /// it runs the indexer.
    pub topk_source: usize,
}

/// A compressed stream a layer attends whole: every row, no selection (V4's
/// HCA layers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseStream {
    /// `attention.compress_ratios` at this layer: tokens pooled into one row.
    pub ratio: u32,
    /// The layer whose compressor writes the rows; itself.
    pub kv_source: usize,
}

/// A compressor a layer owns: its latent's (`attn_compressor_*`) or its
/// indexer's (`indexer_compressor_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Compressor {
    /// Whether it pools with scores (`…_gate`); a compressor that pools one
    /// token per row has none.
    pub gated: bool,
    /// Whether it adds a position table to the scores (`…_ape`), one row per
    /// slot of a group.
    pub ape: bool,
    /// Whether its groups overlap: it projects to two rows' width, and a row
    /// pools its group with the previous one (ik's `dsv4_csa_overlap`). The
    /// projection's width says which: one row's, or two.
    pub overlap: bool,
}

/// What one layer is, from the tensors it carries: the per-layer table the
/// step reads instead of layer numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerKind {
    /// The compressed stream it attends through a top-k, or `None`.
    pub stream: Option<Stream>,
    /// The compressed stream it attends whole, or `None`. A layer attends at
    /// most one stream; a window-only layer has neither.
    pub dense: Option<DenseStream>,
    /// The compressor it owns (`attn_compressor_kv`).
    pub compressor: Option<Compressor>,
    /// It projects index keys from the pooled latent (`indexer.attn_k`).
    pub index_keys: bool,
    /// Its indexer's own compressor, which pools the index keys from the layer
    /// input (`indexer_compressor_kv`).
    pub index_compressor: Option<Compressor>,
    /// It runs the indexer's top-k (`indexer.attn_q_b`).
    pub indexer: bool,
    /// Its engram site, an index into [`Engram::layer_ids`] (`engram_embd`).
    pub engram: Option<usize>,
    /// It routes to experts (`ffn_gate_inp`).
    pub routed: bool,
    /// Its experts come from the token table `ffn_gate_tid2eid` and only
    /// their weights from the router (build_deepseek4.cpp:1589-1592).
    pub hash_routed: bool,
    /// Its rope: YaRN on a layer with a stream, plain on a window-only one.
    pub rope: Rope,
    /// `swiglu_clamp_exp` at this layer — the routed experts' SwiGLU clamp.
    pub swiglu_limit: f32,
    /// `swiglu_clamp_shexp` at this layer — the shared expert's.
    pub swiglu_limit_shared: f32,
    /// `ffn_down_shexp`'s type: whether the shared expert's down projection
    /// reads its input as is or in q8_1 decides a launch of the step.
    pub shared_down: GgmlType,
}

impl LayerKind {
    /// `attention.compress_ratios` at this layer: 0 on a window-only layer.
    pub fn ratio(&self) -> u32 {
        match (self.stream, self.dense) {
            (Some(s), _) => s.ratio,
            (None, Some(d)) => d.ratio,
            (None, None) => 0,
        }
    }

    /// It attends a compressed stream, selected or whole.
    pub fn compressed(&self) -> bool {
        self.stream.is_some() || self.dense.is_some()
    }
}

impl Hparams {
    /// Everything from `split`'s headers; a missing or malformed key, or a
    /// layer table ik's loader would refuse, is an error naming the key or the
    /// tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let model = crate::arch::deepseek41_model(split)?;
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
        let indexer = Indexer {
            n_head: meta_usize(split, "attention.indexer.head_count")?,
            head_dim: meta_usize(split, "attention.indexer.key_length")?,
            top_k: meta_usize(split, "attention.indexer.top_k")?,
        };
        let ropes = Ropes::read(split)?;
        let hash_layers = meta_usize(split, "hash_layer_count")?;
        let tables = LayerTables {
            ratios: compress_ratios(split, n_layer)?,
            swiglu: per_layer_f32(split, "swiglu_clamp_exp", n_layer)?,
            swiglu_shared: per_layer_f32(split, "swiglu_clamp_shexp", n_layer)?,
            hash_layers,
            widths: Widths {
                latent: head_dim as u64,
                index_key: indexer.head_dim as u64,
            },
        };
        let engram = match model {
            Model::Deepseek41 => Some(Engram::read(split, n_layer)?),
            Model::Deepseek4 => {
                no_engram(split, n_layer)?;
                None
            }
        };
        let rows = RowTypes::read(split, engram.as_ref())?;
        let carries: Vec<Carries> = (0..n_layer).map(|l| Carries::read(split, l)).collect();
        let ratios_key = split.arch_key("attention.compress_ratios");
        no_foreign_tensors(&carries, model)?;
        let walk = match model {
            Model::Deepseek41 => walk_streams(&ratios_key, &carries, &tables.ratios)?,
            Model::Deepseek4 => own_streams(&ratios_key, &carries, &tables.ratios)?,
        };
        let layers = layer_kinds(split, &carries, &walk, &tables, engram.as_ref(), &ropes)?;
        let dense_lead = dense_lead(split, &layers)?;
        Ok(Hparams {
            model,
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
            q_head_norm: model == Model::Deepseek4,
            n_vocab: n_vocab(split)?,
            n_ctx_train: meta_usize(split, "context_length")?,
            csa_ratio: walk.csa_ratio,
            hca_ratio: walk.hca_ratio,
            indexer,
            hc: HyperConnections {
                streams: meta_usize(split, "hyper_connection.count")?,
                sinkhorn_iters: meta_usize(split, "hyper_connection.sinkhorn_iterations")?,
                eps: meta_f32(split, "hyper_connection.epsilon")?,
            },
            collapse: collapse(split, model)?,
            experts: Experts::read(split, dense_lead, hash_layers)?,
            engram,
            rows,
            layers,
        })
    }

    /// The engram dimensions, or the error that names the file's lack of
    /// engram sites.
    pub fn engram(&self) -> Result<&Engram, ModelError> {
        self.engram
            .as_ref()
            .ok_or_else(|| no_engram_sites(self.model))
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

/// A file of `model` without engram sites, asked for them.
fn no_engram_sites(model: Model) -> ModelError {
    ModelError::Metadata {
        key: format!("{}.engram.layer_ids", model.name()),
        detail: "is absent: the file has no engram sites".to_string(),
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
    fn read(
        split: &Split,
        dense_lead: usize,
        hash_layers: usize,
    ) -> Result<Experts, PlacementError> {
        let gating = meta_u64(split, "expert_gating_func")?;
        if gating != IK_SQRT_SOFTPLUS {
            return Err(metadata(
                split,
                "expert_gating_func",
                format!("is {gating}; the V4 and V4.1 routers score with sqrt-softplus"),
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
            hash_layers,
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

/// The per-layer metadata tables the layer walk reads, and the widths its
/// compressors' projections are measured against.
struct LayerTables {
    ratios: Vec<u32>,
    swiglu: Vec<f32>,
    swiglu_shared: Vec<f32>,
    /// `hash_layer_count`.
    hash_layers: usize,
    widths: Widths,
}

/// The tensors of one layer that the stream walks read, and the widths of its
/// compressors' projections.
#[derive(Clone, Copy, Debug, Default)]
struct Carries {
    /// `attn_compressor_kv`, by its output width.
    compressor: Option<u64>,
    /// `attn_compressor_gate`.
    gate: bool,
    /// `attn_compressor_ape`.
    ape: bool,
    /// `indexer.attn_k`.
    index_keys: bool,
    /// `indexer_compressor_kv`, by its output width.
    index_compressor: Option<u64>,
    /// `indexer_compressor_gate`.
    index_gate: bool,
    /// `indexer_compressor_ape`.
    index_ape: bool,
    /// `indexer.attn_q_b`.
    indexer: bool,
    /// `ffn_gate_tid2eid`.
    hash_table: bool,
}

impl Carries {
    fn read(split: &Split, l: usize) -> Carries {
        let has = |name: String| split.find(&name).is_some();
        let width = |name: String| {
            split
                .find(&name)
                .map(|(_, t)| t.dims.get(1).copied().unwrap_or(0))
        };
        Carries {
            compressor: width(names::attn_compressor_kv(l)),
            gate: has(names::attn_compressor_gate(l)),
            ape: has(names::attn_compressor_ape(l)),
            index_keys: has(names::indexer_attn_k(l)),
            index_compressor: width(names::indexer_compressor_kv(l)),
            index_gate: has(names::indexer_compressor_gate(l)),
            index_ape: has(names::indexer_compressor_ape(l)),
            indexer: has(names::indexer_attn_q_b(l)),
            hash_table: has(names::ffn_gate_tid2eid(l)),
        }
    }
}

/// The row widths a compressor's projection is one or two of: the latent's
/// and an index key's.
#[derive(Clone, Copy, Debug)]
struct Widths {
    latent: u64,
    index_key: u64,
}

/// What a stream walk resolves.
struct Walk {
    /// Per layer, the stream it attends through a top-k.
    streams: Vec<Option<Stream>>,
    /// Per layer, the stream it attends whole.
    dense: Vec<Option<DenseStream>>,
    /// [`Hparams::csa_ratio`].
    csa_ratio: u32,
    /// [`Hparams::hca_ratio`].
    hca_ratio: u32,
}

/// The two ratios of a table: the first nonzero one in layer order, and the
/// next distinct one (the first again when there is none); a third is
/// refused at its layer, a table with none as a whole.
fn segments(
    refuse: impl Fn(String) -> PlacementError,
    ratios: &[u32],
) -> Result<(u32, u32), PlacementError> {
    let mut seen: Vec<u32> = Vec::with_capacity(2);
    for (l, &ratio) in ratios.iter().enumerate() {
        if ratio == 0 || seen.contains(&ratio) {
            continue;
        }
        if let [csa, hca] = seen[..] {
            return Err(refuse(format!(
                "is {ratio} at layer {l}, a third ratio after {csa} and {hca}"
            )));
        }
        seen.push(ratio);
    }
    match seen[..] {
        [csa] => Ok((csa, csa)),
        [csa, hca] => Ok((csa, hca)),
        _ => Err(refuse("is 0 on every layer".to_string())),
    }
}

/// A file of one model carries none of the other's layer tensors: V4.1 has no
/// position tables, index-key compressors or hash tables, V4 no
/// `indexer.attn_k`. A tensor its model's chain does not read would be
/// ignored, so it is refused by name.
fn no_foreign_tensors(carries: &[Carries], model: Model) -> Result<(), PlacementError> {
    for (l, c) in carries.iter().enumerate() {
        let foreign = match model {
            Model::Deepseek41 => [
                (c.ape, names::attn_compressor_ape(l)),
                (
                    c.index_compressor.is_some(),
                    names::indexer_compressor_kv(l),
                ),
                (c.hash_table, names::ffn_gate_tid2eid(l)),
            ]
            .into_iter()
            .find(|(has, _)| *has),
            Model::Deepseek4 => (c.index_keys).then(|| (true, names::indexer_attn_k(l))),
        };
        if let Some((_, name)) = foreign {
            return Err(PlacementError::Tensor {
                name,
                detail: format!("is not a {} tensor", model.name()),
            });
        }
    }
    Ok(())
}

/// A file without engram sites: no `engram.*` key and no `engram_embd`
/// table. A table or a key would be ignored, so each is refused by name.
fn no_engram(split: &Split, n_layer: usize) -> Result<(), PlacementError> {
    if let Some(l) = (0..n_layer).find(|&l| split.find(&names::engram_embd(l)).is_some()) {
        return Err(PlacementError::Tensor {
            name: names::engram_embd(l),
            detail: "is an engram table in a file whose model has no engram sites".to_string(),
        });
    }
    let key = "engram.layer_ids";
    if split.value(&split.arch_key(key)).is_some() {
        return Err(metadata(
            split,
            key,
            "is set in a file whose model has no engram sites",
        ));
    }
    Ok(())
}

/// A compressor of projection width `width` over rows of `row` values:
/// one row's width pools disjoint groups, two rows' overlapping ones
/// (build_deepseek4.cpp:945, the state of `2·ratio` rows at twice the width,
/// llama-dsv4.cpp:985-990); any other width is refused.
fn compressor_of(
    name: String,
    width: u64,
    row: u64,
    gated: bool,
    ape: bool,
) -> Result<Compressor, PlacementError> {
    let overlap = if width == row {
        false
    } else if width == 2 * row {
        true
    } else {
        return Err(PlacementError::Tensor {
            name,
            detail: format!("projects to {width} values, not one row of {row} or two"),
        });
    };
    Ok(Compressor {
        gated,
        ape,
        overlap,
    })
}

/// V4's streams (ik builds them without a walk: every compressed layer pools
/// its own rows, build_deepseek4.cpp:1170-1206). A layer whose ratio is not 0
/// carries its compressor, with its gate when it pools more than one token
/// per row. A layer that carries `indexer.attn_q_b` selects its rows through
/// its own index-key compressor (`indexer_compressor_kv`, with its gate); one
/// without attends every row. The ratios take at most two values, one a
/// selected stream's and one a dense stream's, since ik's graph tells the two
/// streams by ratio alone (`dsv4_csa_ratio`, `dsv4_hca_ratio`). A compressor
/// or an indexer on a layer whose ratio is 0 is refused.
fn own_streams(
    ratios_key: &str,
    carries: &[Carries],
    ratios: &[u32],
) -> Result<Walk, PlacementError> {
    let refuse = |detail: String| PlacementError::Metadata {
        key: ratios_key.to_string(),
        detail,
    };
    let missing = |name: String, detail: String| PlacementError::Tensor { name, detail };
    let (csa_ratio, hca_ratio) = segments(refuse, ratios)?;
    let (mut selected, mut whole): (Option<u32>, Option<u32>) = (None, None);
    let mut streams = Vec::with_capacity(ratios.len());
    let mut dense = Vec::with_capacity(ratios.len());
    for (l, (c, &ratio)) in carries.iter().zip(ratios).enumerate() {
        if ratio == 0 {
            let carried = [
                (c.compressor.is_some(), "a compressor"),
                (c.indexer, "an indexer"),
                (c.index_compressor.is_some(), "an index-key compressor"),
            ]
            .into_iter()
            .find(|(has, _)| *has);
            if let Some((_, what)) = carried {
                return Err(refuse(format!("is 0 at layer {l}, which carries {what}")));
            }
            streams.push(None);
            dense.push(None);
            continue;
        }
        if c.compressor.is_none() {
            return Err(missing(
                names::attn_compressor_kv(l),
                format!("is not in the file, and layer {l} compresses at {ratio}"),
            ));
        }
        if !c.gate && ratio != 1 {
            return Err(missing(
                names::attn_compressor_gate(l),
                format!("is not in the file, and layer {l} pools {ratio} tokens per row"),
            ));
        }
        let (kind, other) = if c.indexer {
            (&mut selected, whole)
        } else {
            (&mut whole, selected)
        };
        if other == Some(ratio) || kind.is_some_and(|r| r != ratio) {
            return Err(refuse(format!(
                "is {ratio} at layer {l}, and a {} stream takes one ratio",
                if c.indexer { "selected" } else { "dense" }
            )));
        }
        *kind = Some(ratio);
        if c.indexer {
            if c.index_compressor.is_none() {
                return Err(missing(
                    names::indexer_compressor_kv(l),
                    format!("is not in the file, and layer {l} runs the indexer"),
                ));
            }
            if !c.index_gate && ratio != 1 {
                return Err(missing(
                    names::indexer_compressor_gate(l),
                    format!("is not in the file, and layer {l} pools {ratio} tokens per index key"),
                ));
            }
            streams.push(Some(Stream {
                ratio,
                kv_source: l,
                index_key_source: l,
                topk_source: l,
            }));
            dense.push(None);
        } else {
            if c.index_compressor.is_some() {
                return Err(missing(
                    names::indexer_compressor_kv(l),
                    format!("is in the file, and layer {l} carries no indexer.attn_q_b"),
                ));
            }
            streams.push(None);
            dense.push(Some(DenseStream {
                ratio,
                kv_source: l,
            }));
        }
    }
    Ok(Walk {
        streams,
        dense,
        csa_ratio,
        hca_ratio,
    })
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
        if c.compressor.is_some() {
            last_kv = Some(l);
        }
        if c.index_keys {
            last_key = Some(l);
        }
        if c.indexer {
            last_idx = Some(l);
        }
        if ratio == 0 {
            if c.compressor.is_some() {
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
        if c.compressor.is_some() && !c.gate && ratio != 1 {
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
        dense: vec![None; streams.len()],
        streams,
        csa_ratio,
        hca_ratio,
    })
}

/// Every layer's kind: its stream from the walk, what it carries, its engram
/// site (build_deepseek4.cpp:1543-1545), whether it routes and whether by the
/// token table — which a layer carries exactly when it lies below
/// `hash_layer_count` (build_deepseek4.cpp:1589), and only on a routed layer.
fn layer_kinds(
    split: &Split,
    carries: &[Carries],
    walk: &Walk,
    tables: &LayerTables,
    engram: Option<&Engram>,
    ropes: &Ropes,
) -> Result<Vec<LayerKind>, PlacementError> {
    let (widths, hash_layers) = (tables.widths, tables.hash_layers);
    let mut layers = Vec::with_capacity(carries.len());
    for (l, c) in carries.iter().enumerate() {
        let (stream, dense) = (walk.streams[l], walk.dense[l]);
        let compressor = match c.compressor {
            Some(w) => Some(compressor_of(
                names::attn_compressor_kv(l),
                w,
                widths.latent,
                c.gate,
                c.ape,
            )?),
            None => None,
        };
        let index_compressor = match c.index_compressor {
            Some(w) => Some(compressor_of(
                names::indexer_compressor_kv(l),
                w,
                widths.index_key,
                c.index_gate,
                c.index_ape,
            )?),
            None => None,
        };
        let routed = split.find(&names::ffn_gate_inp(l)).is_some();
        let hash_routed = hash_site(split, c.hash_table, l, hash_layers, routed)?;
        layers.push(LayerKind {
            stream,
            dense,
            compressor,
            index_keys: c.index_keys,
            index_compressor,
            indexer: c.indexer,
            engram: engram_site(split, engram, l)?,
            routed,
            hash_routed,
            rope: if stream.is_some() || dense.is_some() {
                ropes.compressed
            } else {
                ropes.window
            },
            swiglu_limit: tables.swiglu[l],
            swiglu_limit_shared: tables.swiglu_shared[l],
            shared_down: split
                .find(&names::ffn_down_shexp(l))
                .map(|(_, t)| t.ty)
                .ok_or_else(|| PlacementError::Tensor {
                    name: names::ffn_down_shexp(l),
                    detail: "is not in the file".to_string(),
                })?,
        });
    }
    Ok(layers)
}

/// Whether layer `l` routes by the token table: it carries
/// `ffn_gate_tid2eid` exactly when it lies below `hash_layer_count`, and the
/// table picks among the experts of a routed layer.
fn hash_site(
    split: &Split,
    table: bool,
    l: usize,
    hash_layers: usize,
    routed: bool,
) -> Result<bool, PlacementError> {
    match (table, l < hash_layers) {
        (true, true) if !routed => Err(PlacementError::Tensor {
            name: names::ffn_gate_tid2eid(l),
            detail: format!("is in the file, and layer {l} carries no router"),
        }),
        (true, true) => Ok(true),
        (false, false) => Ok(false),
        (true, false) => Err(metadata(
            split,
            "hash_layer_count",
            format!(
                "is {hash_layers}, and layer {l} carries {}",
                names::ffn_gate_tid2eid(l)
            ),
        )),
        (false, true) => Err(PlacementError::Tensor {
            name: names::ffn_gate_tid2eid(l),
            detail: format!("is not in the file, and hash_layer_count is {hash_layers}"),
        }),
    }
}

/// How the streams collapse at the head: a V4.1 file with the lagged mix
/// and no head mix tensors, a V4 file with all three (ik takes the choice
/// from the architecture, llama-hparams.h `dsv4_hc_lag`; the tensors are what
/// show it here).
fn collapse(split: &Split, model: Model) -> Result<Collapse, PlacementError> {
    let head = [
        names::output_hc_fn(),
        names::output_hc_base(),
        names::output_hc_scale(),
    ];
    let want = model == Model::Deepseek4;
    if let Some(name) = head.into_iter().find(|n| split.find(n).is_some() != want) {
        let detail = if want {
            "is not in the file, and the model collapses its streams with the head mix"
        } else {
            "is in the file, and the model collapses its streams with the lagged mix"
        };
        return Err(PlacementError::Tensor {
            name,
            detail: detail.to_string(),
        });
    }
    Ok(if want {
        Collapse::Head
    } else {
        Collapse::Lagged
    })
}

/// Layer `l`'s engram site: it carries a table exactly when `engram.layer_ids`
/// lists it (ik asserts the one direction, build_deepseek4.cpp:1545).
fn engram_site(
    split: &Split,
    engram: Option<&Engram>,
    l: usize,
) -> Result<Option<usize>, PlacementError> {
    let table = names::engram_embd(l);
    let site = engram.and_then(|e| e.layer_ids.iter().position(|&x| x == l));
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
                compressor: [2, 8, 14, 20].contains(&l).then_some(512),
                gate: [2, 8, 14].contains(&l),
                index_keys: [2, 8, 14, 20].contains(&l),
                indexer: [2, 8, 14, 20, 24, 28, 32, 36].contains(&l),
                ..Carries::default()
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
                    c[36].compressor = Some(512);
                    c[36].gate = true;
                    r[36..].fill(4);
                },
                "at layer 36, a third ratio after 2 and 1",
            ),
            (
                |c, _| c[0].compressor = Some(512),
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

    const V4_RATIOS_KEY: &str = "deepseek4.attention.compress_ratios";

    /// The V4-Flash file's layer table as `just gate-deepseek4-meta` pins it:
    /// ratio 0 on layers 0-1, then 4 and 128 alternating from layer 2 to 42;
    /// every compressed layer owns its gated compressor (twice the latent wide
    /// at ratio 4), the ratio-4 layers their indexer and index-key compressor.
    fn v4() -> (Vec<Carries>, Vec<u32>) {
        let ratios: Vec<u32> = (0..43)
            .map(|l| match l {
                0 | 1 => 0,
                l if l % 2 == 0 => 4,
                _ => 128,
            })
            .collect();
        let carries = ratios
            .iter()
            .map(|&r| Carries {
                compressor: (r != 0).then_some(if r == 4 { 1024 } else { 512 }),
                gate: r != 0,
                ape: r != 0,
                index_compressor: (r == 4).then_some(256),
                index_gate: r == 4,
                index_ape: r == 4,
                indexer: r == 4,
                ..Carries::default()
            })
            .collect();
        (carries, ratios)
    }

    /// The V4 table walks into selected streams at 4 and dense ones at 128,
    /// every layer its own source.
    #[test]
    fn v4_walk_owns_every_stream() {
        let (c, r) = v4();
        let w = own_streams(V4_RATIOS_KEY, &c, &r).expect("the V4 table walks");
        assert_eq!((w.csa_ratio, w.hca_ratio), (4, 128));
        for l in 0..43 {
            let (s, d) = (w.streams[l], w.dense[l]);
            match r[l] {
                0 => assert!(s.is_none() && d.is_none(), "layer {l}"),
                4 => assert_eq!(
                    s,
                    Some(Stream {
                        ratio: 4,
                        kv_source: l,
                        index_key_source: l,
                        topk_source: l
                    }),
                    "layer {l}"
                ),
                _ => assert_eq!(
                    (s, d),
                    (
                        None,
                        Some(DenseStream {
                            ratio: 128,
                            kv_source: l
                        })
                    ),
                    "layer {l}"
                ),
            }
        }
    }

    /// The V4 table broken once per check is refused at the layer that
    /// breaks it.
    #[test]
    fn v4_walk_refuses_a_broken_table_at_its_layer() {
        type Break = fn(&mut [Carries], &mut [u32]);
        let cases: [(Break, &str); 7] = [
            (
                |c, _| c[3].compressor = None,
                "blk.3.attn_compressor_kv.weight: is not in the file, and layer 3 compresses at 128",
            ),
            (
                |c, _| c[5].gate = false,
                "blk.5.attn_compressor_gate.weight: is not in the file, and layer 5 pools 128",
            ),
            (
                |c, _| c[4].index_compressor = None,
                "blk.4.indexer_compressor_kv.weight: is not in the file, and layer 4 runs the indexer",
            ),
            (
                |c, _| c[7].index_compressor = Some(256),
                "blk.7.indexer_compressor_kv.weight: is in the file, and layer 7 carries no indexer",
            ),
            (
                |c, _| c[8].indexer = false,
                "is 4 at layer 8, and a dense stream takes one ratio",
            ),
            (
                |c, _| c[1].indexer = true,
                "is 0 at layer 1, which carries an indexer",
            ),
            (
                |_, r| r[42] = 2,
                "is 2 at layer 42, a third ratio after 4 and 128",
            ),
        ];
        for (i, (break_it, expected)) in cases.into_iter().enumerate() {
            let (mut c, mut r) = v4();
            break_it(&mut c, &mut r);
            match own_streams(V4_RATIOS_KEY, &c, &r) {
                Ok(_) => panic!("case {i} walked; expected \"{expected}\""),
                Err(e) => assert!(e.to_string().contains(expected), "case {i}: {e}"),
            }
        }
    }

    /// A compressor projects to one row's width or two; the second overlaps.
    #[test]
    fn compressor_width_decides_overlap() {
        let name = || "blk.2.attn_compressor_kv.weight".to_string();
        assert!(
            !compressor_of(name(), 512, 512, true, false)
                .unwrap()
                .overlap
        );
        assert!(
            compressor_of(name(), 1024, 512, true, true)
                .unwrap()
                .overlap
        );
        let e = compressor_of(name(), 768, 512, true, false).unwrap_err();
        assert!(
            e.to_string()
                .contains("projects to 768 values, not one row of 512 or two"),
            "{e}"
        );
    }

    /// A model's file carrying the other's layer tensor is refused by name.
    #[test]
    fn foreign_tensors_are_refused() {
        let (mut c, _) = v4();
        c[2].index_keys = true;
        let e = no_foreign_tensors(&c, Model::Deepseek4).unwrap_err();
        assert!(
            e.to_string()
                .contains("blk.2.indexer.attn_k.weight: is not a deepseek4 tensor"),
            "{e}"
        );
        let (mut c, _) = served();
        no_foreign_tensors(&c, Model::Deepseek41).expect("the served table is V4.1's");
        c[8].ape = true;
        let e = no_foreign_tensors(&c, Model::Deepseek41).unwrap_err();
        assert!(
            e.to_string()
                .contains("blk.8.attn_compressor_ape.weight: is not a deepseek41 tensor"),
            "{e}"
        );
    }
}
