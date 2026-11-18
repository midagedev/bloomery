//! Every `hy_v4_flash.*` key the description reads, each block's mixer and feed-forward kind from
//! the per-block arrays, and the tensors the file is held to: the names every block's kind
//! carries, their dims and the quant types the engine reads them in. A key the loader requires is
//! required here, and a value this reader does not run is refused by the key's name. The file
//! holds `block_count` blocks: the trunk's layers, then the next-token block
//! (`nextn_predict_layers`), which the reader checks and never loads. A block is a
//! sparse-attention layer when `attention.indexer.types` says so and a window layer when
//! `attention.sliding_window_pattern` does, and the two arrays are each other's complement. The
//! dialect's authority is llama.cpp's `src/models/hy-v4-flash.cpp` (PR #32517, `load_arch_hparams`
//! and `load_arch_tensors`).

use gguf::Split;
use gguf::arch::hy_v4_flash as released;

use super::names::{ENGRAM_TABLE, OUTPUT_HC_BASE, OUTPUT_HC_FN, OUTPUT_HC_SCALE};
use super::roles::{KINDS, Kind, stems_of};
use crate::arch::classify;
use crate::arch::delta_mla::{Mla, dense_lead_of};
use crate::arch::hybrid::layers_agree;
use crate::arch::noaux::{self, Experts};
use crate::arch::{
    dims_agree, finite_positive, meta_arr, meta_bool, meta_usize, metadata, n_vocab,
    optional_usize, per_layer_bool, per_layer_count, per_layer_f32, plain_rope, positive,
    unread_keys_of,
};
use crate::placement::PlacementError;

/// The indexer's pooling and selection: `heads` index heads of `d` values, the `top_k` tokens a
/// layer attends, keys pooled `kpool` positions to one, the query's own pool forced in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Indexer {
    /// `attention.indexer.head_count`.
    pub heads: usize,
    /// `attention.indexer.key_length`.
    pub d: usize,
    /// `attention.indexer.top_k`: tokens kept: `top_k / kpool` pools, the current one forced.
    pub top_k: usize,
    /// `attention.indexer.kpool`: positions a pooled key covers.
    pub kpool: usize,
}

/// The hyper-connections: `streams` streams, the mixes' floor `eps` and the post mix's `magnitude`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hc {
    /// `hyper_connection.count`.
    pub streams: usize,
    /// `hyper_connection.epsilon`.
    pub eps: f32,
    /// `hyper_connection.magnitude`.
    pub magnitude: f32,
}

/// The n-gram sites: the `layers` (0-based, ascending) that gather rows of the per-layer table,
/// `heads` rows an n-gram order, orders 2 to `ngram`, rows of `key_length` values, and the
/// table's own row count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Engram {
    /// `ple.layers`.
    pub layers: Vec<usize>,
    /// `ple.ngram_size`.
    pub ngram: usize,
    /// `ple.heads_per_ngram`.
    pub heads: usize,
    /// `embedding_length_per_layer_input`: values a gathered row.
    pub key_length: usize,
    /// The table's rows: the last bucket's offset and size, `ple.head_offsets` and
    /// `ple.head_vocab_sizes`.
    pub rows: u64,
}

impl Engram {
    /// Columns a site gathers a token: `heads` for each order past the first.
    #[must_use]
    pub fn n_cols(&self) -> usize {
        (self.ngram - 1) * self.heads
    }

    /// Rows one token gathers over every site.
    #[must_use]
    pub fn rows_a_token(&self) -> usize {
        self.layers.len() * self.n_cols()
    }

    /// Whether layer `l` is a site.
    #[must_use]
    pub fn site(&self, l: usize) -> bool {
        self.layers.contains(&l)
    }
}

/// The hyperparameters of one Hy4-flash file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`: the trunk's layers and the next-token block.
    pub n_block: usize,
    /// The trunk's layers: `block_count` less `nextn_predict_layers`.
    pub n_layer: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `vocab_size`, else the token list's length.
    pub n_vocab: usize,
    /// `context_length`.
    pub n_ctx_train: usize,
    /// `leading_dense_block_count`: the layers before the first routed one.
    pub dense_lead: usize,
    /// `feed_forward_length`, the dense layers' width.
    pub dense_ff: usize,
    /// `attention.layer_norm_rms_epsilon`: every block norm's, the sandwich norms' and the
    /// hyper-connection's flat norm's among them.
    pub rms_eps: f32,
    pub mla: Mla,
    /// `attention.head_count`, each block's query heads.
    pub heads: Vec<usize>,
    /// `attention.indexer.types`: each trunk layer's sparse attention; the next-token block's is
    /// the window pattern's complement.
    pub dsa: Vec<bool>,
    /// `attention.sliding_window`: the positions a window layer attends, itself included.
    pub window: usize,
    /// `rope.freq_base`: a sparse-attention layer's rope base.
    pub rope_base: f32,
    /// `rope.freq_base_swa`: a window layer's rope base.
    pub rope_base_swa: f32,
    pub indexer: Indexer,
    /// The routed blocks' router and experts.
    pub experts: Experts,
    /// `expert_latent_length`: the width the routed experts run in.
    pub latent: usize,
    /// `swiglu_clamp_exp`: each block's routed experts' gate-first clamp.
    pub clamp: Vec<f32>,
    pub hc: Hc,
    pub engram: Option<Engram>,
    /// The keys this read took a default for: the key, the value, the line.
    pub defaults: Vec<String>,
}

/// Every `hy_v4_flash.` key [`Hparams::read`] reads but the routed block's ([`noaux::KEYS`]), by
/// suffix.
const READ: &[&str] = &[
    "block_count",
    "nextn_predict_layers",
    "context_length",
    "embedding_length",
    "vocab_size",
    "feed_forward_length",
    "leading_dense_block_count",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.key_length_mla",
    "attention.value_length_mla",
    "attention.q_lora_rank",
    "attention.kv_lora_rank",
    "attention.layer_norm_rms_epsilon",
    "attention.sliding_window",
    "attention.sliding_window_pattern",
    "attention.indexer.head_count",
    "attention.indexer.key_length",
    "attention.indexer.top_k",
    "attention.indexer.types",
    "attention.indexer.kpool",
    "attention.indexer.kpool_select_tail",
    "rope.dimension_count",
    "rope.freq_base",
    "rope.freq_base_swa",
    "expert_latent_length",
    "swiglu_clamp_exp",
    "hyper_connection.count",
    "hyper_connection.epsilon",
    "hyper_connection.magnitude",
    "ple.layers",
    "ple.ngram_size",
    "ple.heads_per_ngram",
    "ple.layer_multipliers",
    "ple.head_offsets",
    "ple.head_vocab_sizes",
    "embedding_length_per_layer_input",
    "engram.pad_token_id",
    "engram.token_map",
];

/// The architecture keys and `general.sampling.*` keys of `split` that [`Hparams::read`] does not
/// read: listed, not refused.
#[must_use]
pub fn unread_keys(split: &Split) -> Vec<String> {
    unread_keys_of(split, &[READ, noaux::KEYS].concat())
}

/// `<architecture>.<suffix>` as an array of unsigned values.
fn unsigned_arr(split: &Split, suffix: &str) -> Result<Vec<u64>, PlacementError> {
    meta_arr(split, suffix)?
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_unsigned()
                .ok_or_else(|| metadata(split, suffix, format!("has no count at index {i}")))
        })
        .collect()
}

impl Hparams {
    /// Every value from `split`'s headers; a missing required key, a value this reader does not
    /// run, a block whose tensors disagree with its kind, the keys or the released set's types is
    /// an error naming the key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        if split.architecture() != Some(released::ARCH) {
            return Err(crate::arch::architecture_is(&format!(
                "{:?}; this reader reads {:?} files",
                split.architecture().unwrap_or("<missing>"),
                released::ARCH
            )));
        }
        let mut defaults = Vec::new();
        let n_block = positive(split, "block_count")?;
        let nextn = optional_usize(split, "nextn_predict_layers")?.unwrap_or(0);
        if nextn > 1 || nextn >= n_block {
            return Err(metadata(
                split,
                "nextn_predict_layers",
                format!(
                    "is {nextn} for {n_block} blocks; this reader holds at most one next-token block"
                ),
            ));
        }
        let n_layer = n_block - nextn;
        let mla = Mla::read(split)?;
        let value = positive(split, "attention.value_length")?;
        if value != mla.kv_lora {
            return Err(metadata(
                split,
                "attention.value_length",
                format!(
                    "is {value}, not kv_lora_rank {}: the cached value is the latent",
                    mla.kv_lora
                ),
            ));
        }
        let heads = per_layer_count(split, "attention.head_count", n_block)?;
        if let Some(l) = heads.iter().position(|&h| h == 0) {
            return Err(metadata(
                split,
                "attention.head_count",
                format!("is 0 at block {l}"),
            ));
        }
        let kv = per_layer_count(split, "attention.head_count_kv", n_block)?;
        if let Some(l) = kv.iter().position(|&k| k != 1) {
            return Err(metadata(
                split,
                "attention.head_count_kv",
                format!(
                    "is {} at block {l}; every block is latent attention (1 key head)",
                    kv[l]
                ),
            ));
        }
        let swa = per_layer_bool(split, "attention.sliding_window_pattern", n_block)?;
        let types = meta_arr(split, "attention.indexer.types")?;
        if types.len() != n_layer && types.len() != n_block {
            return Err(metadata(
                split,
                "attention.indexer.types",
                format!("has {} values for {n_layer} trunk layers", types.len()),
            ));
        }
        // The next-token block's own entry is the window pattern's complement when the array
        // stops at the trunk.
        let mut dsa: Vec<bool> = swa.iter().map(|&w| !w).collect();
        for (l, v) in types.iter().enumerate() {
            dsa[l] = v.as_bool().ok_or_else(|| {
                metadata(
                    split,
                    "attention.indexer.types",
                    format!("has no bool at layer {l}"),
                )
            })?;
        }
        if let Some(l) = dsa.iter().zip(&swa).position(|(&d, &s)| d == s) {
            return Err(metadata(
                split,
                "attention.sliding_window_pattern",
                format!(
                    "is {} at block {l} where attention.indexer.types says {}: a block is sparse \
                     attention or a window, one of the two",
                    u8::from(swa[l]),
                    if dsa[l] {
                        "sparse attention"
                    } else {
                        "no indexer"
                    }
                ),
            ));
        }
        plain_rope(split)?;
        if !meta_bool(split, "attention.indexer.kpool_select_tail")? {
            return Err(metadata(
                split,
                "attention.indexer.kpool_select_tail",
                "is false; this reader keeps the query's own pool in the selection",
            ));
        }
        let indexer = Indexer {
            heads: positive(split, "attention.indexer.head_count")?,
            d: positive(split, "attention.indexer.key_length")?,
            top_k: positive(split, "attention.indexer.top_k")?,
            kpool: positive(split, "attention.indexer.kpool")?,
        };
        if !indexer.top_k.is_multiple_of(indexer.kpool) {
            return Err(metadata(
                split,
                "attention.indexer.top_k",
                format!(
                    "is {}, not a whole number of pools of attention.indexer.kpool {}",
                    indexer.top_k, indexer.kpool
                ),
            ));
        }
        let experts = Experts::read(split, "the hy_v4_flash router", &mut defaults)?;
        let clamp = per_layer_f32(split, "swiglu_clamp_exp", n_block)?
            .ok_or_else(|| metadata(split, "swiglu_clamp_exp", "is absent"))?;
        if let Some(l) = clamp.iter().position(|c| !(c.is_finite() && *c > 0.0)) {
            return Err(metadata(
                split,
                "swiglu_clamp_exp",
                format!(
                    "is {} at block {l}; a clamp is finite and above 0",
                    clamp[l]
                ),
            ));
        }
        let hc = Hc {
            streams: positive(split, "hyper_connection.count")?,
            eps: finite_positive(split, "hyper_connection.epsilon")?,
            magnitude: finite_positive(split, "hyper_connection.magnitude")?,
        };
        let n_vocab = n_vocab(split)?;
        let hp = Hparams {
            n_block,
            n_layer,
            n_embd: positive(split, "embedding_length")?,
            n_vocab,
            n_ctx_train: positive(split, "context_length")?,
            dense_lead: dense_lead_of(split, n_layer)?,
            dense_ff: positive(split, "feed_forward_length")?,
            rms_eps: finite_positive(split, "attention.layer_norm_rms_epsilon")?,
            mla,
            heads,
            dsa,
            window: positive(split, "attention.sliding_window")?,
            rope_base: finite_positive(split, "rope.freq_base")?,
            rope_base_swa: finite_positive(split, "rope.freq_base_swa")?,
            indexer,
            experts,
            latent: positive(split, "expert_latent_length")?,
            clamp,
            hc,
            engram: Some(Engram::read(split, n_layer, n_vocab)?),
            defaults,
        };
        hp.tensors_agree(split)?;
        dims_agree(
            split,
            hp.n_embd,
            hp.n_vocab,
            (0..hp.n_block).map(|l| hp.layer_dims(l)),
        )?;
        hp.model_dims_agree(split)?;
        types_agree(split, hp.n_block)?;
        Ok(hp)
    }

    /// Whether block `l` is the next-token block.
    #[must_use]
    pub fn is_nextn(&self, l: usize) -> bool {
        l >= self.n_layer
    }

    /// Whether block `l` is a dense feed-forward block, not a routed one.
    #[must_use]
    pub fn dense(&self, l: usize) -> bool {
        l < self.dense_lead
    }

    /// Whether block `l` runs sparse attention: the indexer and every position's latent row.
    #[must_use]
    pub fn dsa(&self, l: usize) -> bool {
        self.dsa[l]
    }

    /// Block `l`'s query heads.
    #[must_use]
    pub fn heads(&self, l: usize) -> usize {
        self.heads[l]
    }

    /// Block `l`'s rope base: the window layers' own.
    #[must_use]
    pub fn rope_base_of(&self, l: usize) -> f32 {
        if self.dsa[l] {
            self.rope_base
        } else {
            self.rope_base_swa
        }
    }

    /// Whether layer `l` runs an engram site before its mixer.
    #[must_use]
    pub fn site(&self, l: usize) -> bool {
        self.engram.as_ref().is_some_and(|e| e.site(l))
    }

    /// Block `l`'s kind for the tensor tables.
    pub(super) fn kind(&self, l: usize) -> Kind {
        Kind {
            dsa: self.dsa[l],
            dense: self.dense(l),
            engram: self.site(l),
            nextn: self.is_nextn(l),
        }
    }

    /// Every block carries exactly the stems of its kind, and the model-level tensors are in the
    /// file: the first missing or misplaced name, with how many more there are.
    fn tensors_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let kinds: Vec<Kind> = (0..self.n_block).map(|l| self.kind(l)).collect();
        layers_agree(split, &kinds, stems_of, &KINDS)?;
        let missing: Vec<&str> = [OUTPUT_HC_FN, OUTPUT_HC_BASE, OUTPUT_HC_SCALE, ENGRAM_TABLE]
            .into_iter()
            .filter(|n| split.find(n).is_none())
            .collect();
        match missing.first() {
            None => Ok(()),
            Some(name) => Err(PlacementError::Tensor {
                name: (*name).to_string(),
                detail: format!(
                    "is not in the file ({} model-level tensor(s) the head and the engram sites \
                     need are missing: {})",
                    missing.len(),
                    missing.join(", ")
                ),
            }),
        }
    }

    /// The head's stream fold and the per-layer table hold the dims the keys give them.
    fn model_dims_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let (e, s) = (self.n_embd as u64, self.hc.streams as u64);
        let mut want = vec![
            (OUTPUT_HC_FN, vec![s * e, s]),
            (OUTPUT_HC_BASE, vec![s]),
            (OUTPUT_HC_SCALE, vec![1]),
        ];
        if let Some(g) = &self.engram {
            want.push((ENGRAM_TABLE, vec![g.key_length as u64, g.rows]));
        }
        for (name, dims) in want {
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

    /// Block `l`'s tensors by stem and the dims llama.cpp creates them with, ggml order. The
    /// routed stacks run in the latent; the router and the shared expert read the hidden row.
    #[must_use]
    pub fn layer_dims(&self, l: usize) -> Vec<(&'static str, Vec<u64>)> {
        let (e, h, s) = (
            self.n_embd as u64,
            self.heads[l] as u64,
            self.hc.streams as u64,
        );
        let mut dims: Vec<(&'static str, Vec<u64>)> = self.mla.dims(e, h).to_vec();
        dims.extend([
            ("attn_sinks.weight", vec![h]),
            ("post_attention_norm.weight", vec![e]),
            ("post_ffw_norm.weight", vec![e]),
            ("ffn_norm.weight", vec![e]),
        ]);
        if self.is_nextn(l) {
            dims.extend([
                ("nextn.eh_proj.weight", vec![2 * e, e]),
                ("nextn.enorm.weight", vec![e]),
                ("nextn.hnorm.weight", vec![e]),
                ("nextn.shared_head_norm.weight", vec![e]),
            ]);
        } else {
            let mix = 2 * s;
            for sub in ["attn", "ffn"] {
                dims.extend([
                    (hc_stem(sub, "fn"), vec![s * e, mix]),
                    (hc_stem(sub, "base"), vec![mix]),
                    (hc_stem(sub, "scale"), vec![2]),
                ]);
            }
        }
        if self.dsa[l] {
            let (heads, d, q_lora) = (
                self.indexer.heads as u64,
                self.indexer.d as u64,
                self.mla.q_lora as u64,
            );
            dims.extend([
                ("indexer.attn_q_b.weight", vec![q_lora, heads * d]),
                ("indexer.attn_k.weight", vec![e, d]),
                ("indexer.proj.weight", vec![e, heads]),
                ("indexer.k_norm.weight", vec![d]),
                ("indexer_compressor_value.weight", vec![d, d]),
                ("indexer_compressor_gate.weight", vec![d, d]),
                (
                    "indexer_compressor_ape.weight",
                    vec![d, self.indexer.kpool as u64],
                ),
                ("indexer_compressor_norm.weight", vec![d]),
            ]);
        }
        if let Some(g) = self.engram.as_ref().filter(|g| g.site(l)) {
            let (key, cols) = (g.key_length as u64, g.n_cols() as u64);
            dims.extend([
                ("engram_kv.weight", vec![cols * key, (s + 1) * e]),
                ("engram_kv_b.weight", vec![e, s + 1]),
                ("engram_gate.weight", vec![e, s]),
                ("engram_hc_scale.weight", vec![e, s]),
                ("engram_lookup_norm.weight", vec![key, cols]),
                ("engram_pre_gamma.weight", vec![e]),
            ]);
        }
        if self.dense(l) {
            let ff = self.dense_ff as u64;
            dims.extend([
                ("ffn_gate.weight", vec![e, ff]),
                ("ffn_up.weight", vec![e, ff]),
                ("ffn_down.weight", vec![ff, e]),
            ]);
        } else {
            let latent = self.latent as u64;
            let narrow = self.experts.dims(latent);
            let mut block = self.experts.dims(e);
            for ((_, dims), (_, in_latent)) in block.iter_mut().zip(narrow) {
                if dims.len() == 3 {
                    *dims = in_latent;
                }
            }
            dims.extend(block);
            dims.extend([
                ("ffn_routed_down.weight", vec![e, latent]),
                ("ffn_routed_up.weight", vec![latent, e]),
            ]);
        }
        dims
    }
}

/// `hc_{sub}_{part}.weight` as a static stem: the three parts of the two sublayers' mixes.
fn hc_stem(sub: &str, part: &str) -> &'static str {
    match (sub, part) {
        ("attn", "fn") => "hc_attn_fn.weight",
        ("attn", "base") => "hc_attn_base.weight",
        ("attn", "scale") => "hc_attn_scale.weight",
        ("ffn", "fn") => "hc_ffn_fn.weight",
        ("ffn", "base") => "hc_ffn_base.weight",
        ("ffn", "scale") => "hc_ffn_scale.weight",
        _ => unreachable!("a hyper-connection stem of the two sublayers' three parts"),
    }
}

impl Engram {
    /// The sites' keys: the layers (ascending, inside the trunk), the n-gram size and heads, the
    /// row width, and the hash constants' counts (`ple.layer_multipliers` an order of every site,
    /// `ple.head_offsets` and `ple.head_vocab_sizes` a column of every site, each bucket at least
    /// two rows and its top inside an i32 row index), with the token map one value a token and the
    /// pad a token of it. The hash itself is `engram`'s, built from the same keys.
    fn read(split: &Split, n_layer: usize, n_vocab: usize) -> Result<Engram, PlacementError> {
        let layers: Vec<usize> = unsigned_arr(split, "ple.layers")?
            .into_iter()
            .map(|v| usize::try_from(v).unwrap_or(usize::MAX))
            .collect();
        if layers.is_empty() || layers.windows(2).any(|w| w[0] >= w[1]) {
            return Err(metadata(
                split,
                "ple.layers",
                "is empty or not ascending: the sites are distinct layers in order",
            ));
        }
        if let Some(&l) = layers.iter().find(|&&l| l >= n_layer) {
            return Err(metadata(
                split,
                "ple.layers",
                format!("names block {l}; the trunk has {n_layer} layers"),
            ));
        }
        let ngram = positive(split, "ple.ngram_size")?;
        if !(2..=8).contains(&ngram) {
            return Err(metadata(
                split,
                "ple.ngram_size",
                format!("is {ngram}; an n-gram size of 2 to 8"),
            ));
        }
        let heads = positive(split, "ple.heads_per_ngram")?;
        let g = Engram {
            layers,
            ngram,
            heads,
            key_length: positive(split, "embedding_length_per_layer_input")?,
            rows: 0,
        };
        let (sites, cols) = (g.layers.len(), g.n_cols());
        if sites * cols > 64 {
            return Err(metadata(
                split,
                "ple.layers",
                format!("holds {} buckets; the hash keeps at most 64", sites * cols),
            ));
        }
        let count = |suffix: &str, want: usize| -> Result<Vec<u64>, PlacementError> {
            let v = unsigned_arr(split, suffix)?;
            if v.len() != want {
                return Err(metadata(
                    split,
                    suffix,
                    format!(
                        "has {} values for {sites} sites of {want} entries in all",
                        v.len()
                    ),
                ));
            }
            Ok(v)
        };
        count("ple.layer_multipliers", sites * ngram)?;
        let prime = count("ple.head_vocab_sizes", sites * cols)?;
        let offset = count("ple.head_offsets", sites * cols)?;
        let mut top = 0u64;
        for (b, (&p, &o)) in prime.iter().zip(&offset).enumerate() {
            if p < 2 {
                return Err(metadata(
                    split,
                    "ple.head_vocab_sizes",
                    format!("is {p} at bucket {b}; a bucket holds at least 2 rows"),
                ));
            }
            if o != top {
                return Err(metadata(
                    split,
                    "ple.head_offsets",
                    format!("is {o} at bucket {b}; the buckets follow one another ({top})"),
                ));
            }
            top = o
                .checked_add(p)
                .filter(|&t| t <= i32::MAX as u64)
                .ok_or_else(|| {
                    metadata(
                        split,
                        "ple.head_offsets",
                        format!("bucket {b}'s top is outside an i32 row index"),
                    )
                })?;
        }
        let map = meta_arr(split, "engram.token_map")?;
        if map.len() != n_vocab {
            return Err(metadata(
                split,
                "engram.token_map",
                format!("has {} values for a vocabulary of {n_vocab}", map.len()),
            ));
        }
        let pad = meta_usize(split, "engram.pad_token_id")?;
        if pad >= map.len() {
            return Err(metadata(
                split,
                "engram.pad_token_id",
                format!("is {pad}, outside the {n_vocab}-token map"),
            ));
        }
        Ok(Engram { rows: top, ..g })
    }
}

/// Every tensor of `split` is held in a type the engine reads its stem in
/// ([`released::allowed_types`]), refused as the Hy4-flash reader ([`classify::types_agree`]).
fn types_agree(split: &Split, n_block: usize) -> Result<(), PlacementError> {
    classify::types_agree(split, n_block, "Hy4-flash", released::allowed_types)
}
