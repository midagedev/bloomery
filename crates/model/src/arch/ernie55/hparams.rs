//! Every `ernie55.*` key the description reads, each layer's mixer and feed-forward kind, and the
//! tensors the file is held to: the names every layer's kind carries, and their dims. A key the
//! description needs is required here, and a value this reader does not run is refused by the
//! key's name. A layer is an HCA layer when `attention.compress_ratios` holds a ratio for it and a
//! latent layer when it holds 0; the layers before `leading_dense_block_count` carry a dense
//! feed-forward block. The checkpoint's own config is in [`super::config`].

use gguf::arch::ernie55::{ARCH, PRE};
use gguf::{Split, Value};

use super::names::{
    OUTPUT_HC_BASE, OUTPUT_HC_FN, OUTPUT_HC_SCALE, OUTPUT_MULTIMAX_RANGES, OUTPUT_MULTIMAX_TS,
};
use super::roles::{KINDS, Kind, stems_of};
use crate::arch::delta_mla::{Mla, Situ, dense_lead_of};
use crate::arch::hybrid::layers_agree;
use crate::arch::noaux::{self, Experts};
use crate::arch::{
    architecture_is, dims_agree, finite_positive, meta_f32, meta_str, meta_u64, metadata, n_vocab,
    per_layer_count, positive, unread_keys_of,
};
use crate::placement::PlacementError;

/// The values of the head's MultiMax epilogue: four range edges and four weights.
pub const MULTIMAX: usize = 4;

/// An HCA layer's attention: one key-value head of `head` values (the rotated ones included, key
/// and value the same row), `window` positions attended directly beside one pooled row of every
/// `ratio` positions, and the output projected in `o_groups` groups of `o_lora` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hca {
    /// `attention.hca.head_length`.
    pub head: usize,
    /// The ratio `attention.compress_ratios` gives every HCA layer.
    pub ratio: usize,
    /// `attention.sliding_window`.
    pub window: usize,
    /// `attention.output_group_count`.
    pub o_groups: usize,
    /// `attention.output_lora_rank`.
    pub o_lora: usize,
}

impl Hca {
    /// The values the grouped output projection's output holds: `o_groups` groups of `o_lora`.
    #[must_use]
    pub fn grouped(&self) -> usize {
        self.o_groups * self.o_lora
    }
}

/// The lightning indexer of a latent layer: `heads` index heads of `d` values, the `top_k`
/// positions a query attends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Indexer {
    /// `attention.indexer.head_count`.
    pub heads: usize,
    /// `attention.indexer.key_length`.
    pub d: usize,
    /// `attention.indexer.top_k`.
    pub top_k: usize,
}

/// The rope keys: the plain base of the window and the latent layers, and the base and YaRN of the
/// pooled rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    /// `rope.freq_base`.
    pub base: f32,
    /// `attention.compress_rope_freq_base`.
    pub compress_base: f32,
    /// `rope.scaling.factor`.
    pub factor: f32,
    /// `rope.scaling.original_context_length`.
    pub original_ctx: usize,
    /// `rope.scaling.yarn_beta_fast`.
    pub beta_fast: f32,
    /// `rope.scaling.yarn_beta_slow`.
    pub beta_slow: f32,
}

/// The hyper-connections: `streams` streams, the mixes' floor `eps` and the `sinkhorn_iters`
/// rounds that make the comb doubly stochastic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hc {
    /// `hyper_connection.count`.
    pub streams: usize,
    /// `hyper_connection.epsilon`.
    pub eps: f32,
    /// `hyper_connection.sinkhorn_iterations`.
    pub sinkhorn_iters: usize,
}

impl Hc {
    /// The values of a sublayer's mix: the pre and post weights of every stream and the comb.
    #[must_use]
    pub fn mix(&self) -> usize {
        (2 + self.streams) * self.streams
    }
}

/// The hyperparameters of one ERNIE 5.5 file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`: the trunk's layers.
    pub n_layer: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `vocab_size`, else the token list's length.
    pub n_vocab: usize,
    /// `context_length`.
    pub n_ctx_train: usize,
    /// `leading_dense_block_count`: the layers before the first routed one.
    pub dense_lead: usize,
    /// `feed_forward_length`: the dense layer's width.
    pub dense_ff: usize,
    /// `attention.layer_norm_rms_epsilon`: every norm's.
    pub rms_eps: f32,
    /// `attention.head_count`: every layer's query heads.
    pub heads: usize,
    /// The latent layers' widths.
    pub mla: Mla,
    /// The HCA layers' attention.
    pub hca: Hca,
    /// `attention.compress_ratios`: each layer's pooling ratio, 0 on a latent layer.
    pub ratios: Vec<usize>,
    pub indexer: Indexer,
    /// `attention.vha.rank`: the rank of the head mix's `U` and `V`.
    pub vha_rank: usize,
    pub rope: Rope,
    /// The routed layers' router and experts.
    pub experts: Experts,
    /// `expert_latent_length`: the width the routed experts run in.
    pub latent: usize,
    pub situ: Situ,
    pub hc: Hc,
    /// The keys this read took a default for: the key, the value, the line.
    pub defaults: Vec<String>,
}

/// Every `ernie55.` key [`Hparams::read`] reads but the routed block's ([`noaux::KEYS`]), by
/// suffix.
const READ: &[&str] = &[
    "block_count",
    "context_length",
    "embedding_length",
    "vocab_size",
    "feed_forward_length",
    "leading_dense_block_count",
    "attention.head_count",
    "attention.layer_norm_rms_epsilon",
    "attention.compress_ratios",
    "attention.sliding_window",
    "attention.key_length",
    "attention.key_length_mla",
    "attention.value_length_mla",
    "attention.q_lora_rank",
    "attention.kv_lora_rank",
    "attention.hca.head_length",
    "attention.output_group_count",
    "attention.output_lora_rank",
    "attention.indexer.head_count",
    "attention.indexer.key_length",
    "attention.indexer.top_k",
    "attention.vha.rank",
    "rope.dimension_count",
    "rope.freq_base",
    "attention.compress_rope_freq_base",
    "rope.scaling.type",
    "rope.scaling.factor",
    "rope.scaling.original_context_length",
    "rope.scaling.yarn_beta_fast",
    "rope.scaling.yarn_beta_slow",
    "expert_latent_length",
    "activation.situ_beta",
    "activation.situ_linear_beta",
    "hyper_connection.count",
    "hyper_connection.epsilon",
    "hyper_connection.sinkhorn_iterations",
];

/// The architecture keys and `general.sampling.*` keys of `split` that [`Hparams::read`] does not
/// read: listed, not refused.
#[must_use]
pub fn unread_keys(split: &Split) -> Vec<String> {
    unread_keys_of(split, &[READ, noaux::KEYS].concat())
}

/// The file's pre-tokenizer is ERNIE's ([`PRE`]): the one the vocabulary's ids are made by, so a
/// file that names another is another tokenizer's, refused by the key's name.
fn pre_is_ernie(split: &Split) -> Result<(), PlacementError> {
    match split.value(crate::arch::PRE).and_then(Value::as_str) {
        Some(PRE) => Ok(()),
        other => Err(PlacementError::Metadata {
            key: crate::arch::PRE.to_string(),
            detail: format!("is {other:?}; this reader reads the pre-tokenizer {PRE:?}"),
        }),
    }
}

impl Hparams {
    /// Every value from `split`'s headers; a missing required key, a value this reader does not
    /// run, a layer whose tensors disagree with its kind, or dims the keys do not give is an error
    /// naming the key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        if split.architecture() != Some(ARCH) {
            return Err(architecture_is(&format!(
                "{:?}; this reader reads {:?} files",
                split.architecture().unwrap_or("<missing>"),
                ARCH
            )));
        }
        pre_is_ernie(split)?;
        let mut defaults = Vec::new();
        let n_layer = positive(split, "block_count")?;
        let mla = Mla::read(split)?;
        let ratios = per_layer_count(split, "attention.compress_ratios", n_layer)?;
        let hca = Hca::read(split, &ratios)?;
        let hp = Hparams {
            n_layer,
            n_embd: positive(split, "embedding_length")?,
            n_vocab: n_vocab(split)?,
            n_ctx_train: positive(split, "context_length")?,
            dense_lead: dense_lead_of(split, n_layer)?,
            dense_ff: positive(split, "feed_forward_length")?,
            rms_eps: finite_positive(split, "attention.layer_norm_rms_epsilon")?,
            heads: positive(split, "attention.head_count")?,
            mla,
            hca,
            ratios,
            indexer: Indexer {
                heads: positive(split, "attention.indexer.head_count")?,
                d: positive(split, "attention.indexer.key_length")?,
                top_k: positive(split, "attention.indexer.top_k")?,
            },
            vha_rank: positive(split, "attention.vha.rank")?,
            rope: Rope::read(split)?,
            experts: Experts::read(split, "the ernie55 router", &mut defaults)?,
            latent: positive(split, "expert_latent_length")?,
            situ: Situ::read(split)?,
            hc: Hc {
                streams: positive(split, "hyper_connection.count")?,
                eps: finite_positive(split, "hyper_connection.epsilon")?,
                sinkhorn_iters: positive(split, "hyper_connection.sinkhorn_iterations")?,
            },
            defaults,
        };
        hp.keys_agree(split)?;
        hp.tensors_agree(split)?;
        dims_agree(
            split,
            hp.n_embd,
            hp.n_vocab,
            (0..hp.n_layer).map(|l| hp.layer_dims(l)),
        )?;
        hp.model_dims_agree(split)?;
        Ok(hp)
    }

    /// Whether layer `l` is a latent layer, not an HCA one.
    #[must_use]
    pub fn latent_layer(&self, l: usize) -> bool {
        self.ratios[l] == 0
    }

    /// Whether layer `l` is a dense feed-forward layer, not a routed one.
    #[must_use]
    pub fn dense(&self, l: usize) -> bool {
        l < self.dense_lead
    }

    /// Layer `l`'s kind for the tensor tables.
    pub(super) fn kind(&self, l: usize) -> Kind {
        Kind {
            latent: self.latent_layer(l),
            dense: self.dense(l),
        }
    }

    /// The keys that tie two widths: the heads' values divide into the output groups and the
    /// rotated values fit in an HCA head.
    fn keys_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let key = "attention.output_group_count";
        if !(self.heads * self.hca.head).is_multiple_of(self.hca.o_groups) {
            return Err(metadata(
                split,
                key,
                format!(
                    "is {}; {} heads of {} values do not divide into groups",
                    self.hca.o_groups, self.heads, self.hca.head
                ),
            ));
        }
        if self.mla.pe >= self.hca.head {
            return Err(metadata(
                split,
                "attention.hca.head_length",
                format!(
                    "is {}, not above rope.dimension_count {}",
                    self.hca.head, self.mla.pe
                ),
            ));
        }
        Ok(())
    }

    /// Every layer carries exactly the stems of its kind, and the model-level tensors are in the
    /// file: the first missing or misplaced name, with how many more there are.
    fn tensors_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let kinds: Vec<Kind> = (0..self.n_layer).map(|l| self.kind(l)).collect();
        layers_agree(split, &kinds, stems_of, &KINDS)?;
        let missing: Vec<&str> = [
            OUTPUT_HC_FN,
            OUTPUT_HC_BASE,
            OUTPUT_HC_SCALE,
            OUTPUT_MULTIMAX_RANGES,
            OUTPUT_MULTIMAX_TS,
        ]
        .into_iter()
        .filter(|n| split.find(n).is_none())
        .collect();
        match missing.first() {
            None => Ok(()),
            Some(name) => Err(PlacementError::Tensor {
                name: (*name).to_string(),
                detail: format!(
                    "is not in the file ({} model-level tensor(s) the head needs are missing: {})",
                    missing.len(),
                    missing.join(", ")
                ),
            }),
        }
    }

    /// The head's stream fold and MultiMax hold the dims the keys give them.
    fn model_dims_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let (e, s, m) = (self.n_embd as u64, self.hc.streams as u64, MULTIMAX as u64);
        for (name, dims) in [
            (OUTPUT_HC_FN, vec![s * e, s]),
            (OUTPUT_HC_BASE, vec![s]),
            (OUTPUT_HC_SCALE, vec![1]),
            (OUTPUT_MULTIMAX_RANGES, vec![m]),
            (OUTPUT_MULTIMAX_TS, vec![m]),
        ] {
            if let Some((_, t)) = split.find(name)
                && t.dims != dims
            {
                return Err(PlacementError::Tensor {
                    name: name.to_string(),
                    detail: format!("has dims {:?}; the keys give {dims:?}", t.dims),
                });
            }
        }
        Ok(())
    }

    /// Layer `l`'s tensors by stem and the dims the converter writes them with, ggml order. The
    /// routed stacks run in the latent; the routers and the shared expert read the hidden row; the
    /// output gate reads the query latent.
    #[must_use]
    pub fn layer_dims(&self, l: usize) -> Vec<(&'static str, Vec<u64>)> {
        let (e, h, s, q, r) = (
            self.n_embd as u64,
            self.heads as u64,
            self.hc.streams as u64,
            self.mla.q_lora as u64,
            self.vha_rank as u64,
        );
        let mut dims: Vec<(&'static str, Vec<u64>)> = vec![
            ("attn_norm.weight", vec![e]),
            ("attn_q_a.weight", vec![e, q]),
            ("attn_q_a_norm.weight", vec![q]),
            ("attn_sinks.weight", vec![h]),
            ("attn_vha_u.weight", vec![r, h]),
            ("attn_vha_v.weight", vec![r, h]),
        ];
        if self.latent_layer(l) {
            let m = &self.mla;
            let (kv, k, v) = (m.kv_lora as u64, m.head_k as u64, m.head_v as u64);
            let (ih, d) = (self.indexer.heads as u64, self.indexer.d as u64);
            dims.extend([
                ("attn_q_b.weight", vec![q, h * k]),
                ("attn_kv_a_mqa.weight", vec![e, m.row() as u64]),
                ("attn_kv_a_norm.weight", vec![kv]),
                ("attn_kv_b.weight", vec![kv, h * (m.nope() as u64 + v)]),
                ("attn_gate.weight", vec![q, h * v]),
                ("attn_output.weight", vec![h * v, e]),
                ("indexer.attn_q_b.weight", vec![q, ih * d]),
                ("indexer.attn_k.weight", vec![e, d]),
                ("indexer.proj.weight", vec![e, ih]),
                ("indexer.k_norm.weight", vec![d]),
                ("indexer.k_norm.bias", vec![d]),
            ]);
        } else {
            let c = &self.hca;
            let (head, groups, grouped) = (c.head as u64, c.o_groups as u64, c.grouped() as u64);
            dims.extend([
                ("attn_q_b.weight", vec![q, h * head]),
                ("attn_kv.weight", vec![e, head]),
                ("attn_kv_a_norm.weight", vec![head]),
                ("attn_output_a.weight", vec![h * head / groups, grouped]),
                ("attn_output_b.weight", vec![grouped, e]),
                ("attn_compressor_kv.weight", vec![e, head]),
                ("attn_compressor_gate.weight", vec![e, head]),
                ("attn_compressor_norm.weight", vec![head]),
                ("attn_compressor_ape.weight", vec![head, c.ratio as u64]),
                ("attn_gate.weight", vec![q, grouped]),
            ]);
        }
        let mix = self.hc.mix() as u64;
        dims.extend([
            ("hc_attn_fn.weight", vec![s * e, mix]),
            ("hc_attn_base.weight", vec![mix]),
            ("hc_attn_scale.weight", vec![3]),
            ("hc_ffn_fn.weight", vec![s * e, mix]),
            ("hc_ffn_base.weight", vec![mix]),
            ("hc_ffn_scale.weight", vec![3]),
        ]);
        if self.dense(l) {
            let ff = self.dense_ff as u64;
            dims.extend([
                ("ffn_norm.weight", vec![e]),
                ("ffn_gate.weight", vec![e, ff]),
                ("ffn_up.weight", vec![e, ff]),
                ("ffn_down.weight", vec![ff, e]),
            ]);
        } else {
            let latent = self.latent as u64;
            let n = self.experts.n_expert as u64;
            let narrow = self.experts.dims(latent);
            let mut block = self.experts.dims(e);
            for ((_, dims), (_, in_latent)) in block.iter_mut().zip(narrow) {
                if dims.len() == 3 {
                    *dims = in_latent;
                }
            }
            dims.push(("ffn_norm.weight", vec![e]));
            dims.extend(block);
            dims.extend([
                ("ffn_gate_inp_1.weight", vec![e, n]),
                ("ffn_exp_scale.weight", vec![n]),
                ("ffn_routed_down.weight", vec![e, latent]),
                ("ffn_routed_up.weight", vec![latent, e]),
                ("ffn_routed_norm.weight", vec![latent]),
            ]);
        }
        dims
    }
}

impl Hca {
    /// The HCA layers' keys; every HCA layer takes one ratio, the heads' values divide into the
    /// groups (checked with the head count, in the caller).
    fn read(split: &Split, ratios: &[usize]) -> Result<Hca, PlacementError> {
        let key = "attention.compress_ratios";
        let mut pooled = ratios.iter().copied().filter(|&r| r != 0);
        let ratio = pooled.next().unwrap_or(0);
        if let Some(other) = pooled.find(|&r| r != ratio) {
            return Err(metadata(
                split,
                key,
                format!("holds {ratio} and {other}; every HCA layer pools at one ratio"),
            ));
        }
        Ok(Hca {
            head: positive(split, "attention.hca.head_length")?,
            ratio,
            window: positive(split, "attention.sliding_window")?,
            o_groups: positive(split, "attention.output_group_count")?,
            o_lora: positive(split, "attention.output_lora_rank")?,
        })
    }
}

impl Rope {
    /// The rope keys; the pooled rows' scaling is YaRN, any other type or a factor that is not a
    /// scaling is refused by name.
    fn read(split: &Split) -> Result<Rope, PlacementError> {
        let scaling = meta_str(split, "rope.scaling.type")?;
        if scaling != "yarn" {
            return Err(metadata(
                split,
                "rope.scaling.type",
                format!("is {scaling:?}; the pooled rows' rope is YaRN"),
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
        let original = meta_u64(split, "rope.scaling.original_context_length")?;
        Ok(Rope {
            base: finite_positive(split, "rope.freq_base")?,
            compress_base: finite_positive(split, "attention.compress_rope_freq_base")?,
            factor,
            original_ctx: usize::try_from(original).map_err(|_| {
                metadata(
                    split,
                    "rope.scaling.original_context_length",
                    format!("{original} does not fit usize"),
                )
            })?,
            beta_fast: meta_f32(split, "rope.scaling.yarn_beta_fast")?,
            beta_slow: meta_f32(split, "rope.scaling.yarn_beta_slow")?,
        })
    }
}
