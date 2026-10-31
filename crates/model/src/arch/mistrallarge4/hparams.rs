//! Every `mistral-large4.` key the description reads. A layer is latent
//! attention with no query latent — the hidden row projects to every head
//! whole, and the latent's per-head up-projections absorb into the query —
//! over the YaRN rope on three of four layers (LeRoPE factors in the
//! model-level `rope_freqs.weight`) and a softmax-routed mixture of experts
//! with one shared expert, the first layer dense. The file's own per-layer
//! arrays decide everything else, layer by layer: `attention.rope_pattern`
//! which layers rope (a NoPE layer's head is all unrotated key and its cache
//! row the latent alone), `attention.indexer.types` which layers run the
//! lightning indexer (a shared layer carries none of its tensors and reuses
//! the previous full layer's top-k) and `attention.kv_source_map` who writes
//! each layer's cache (a layer that names another reads it and stores
//! nothing). The arrays are read as the file spells them, never reconstructed
//! from a rule, and their length is the block count.
//!
//! Three keys the header carries beside them are defaults in llama.cpp's
//! loader, taken with a recorded line when the file does not carry them
//! (`leading_dense_block_count`, `final_logit_softcapping`,
//! `rope.scaling.yarn_log_multiplier`).

use gguf::Split;

use super::names;
use crate::arch::classify::ROPE_FREQS;
use crate::arch::{
    expert_counts, finite_positive, meta_arr, meta_bool, meta_str, meta_u64, metadata, n_vocab,
    one_expert_group, optional_f32, optional_usize, per_layer_bool, positive, tensors_agree,
    unread_keys_of,
};
use crate::placement::PlacementError;

/// The router's softmax, ik's and llama.cpp-mainline's
/// `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` (src/llama-model.cpp:1076), one id.
const SOFTMAX: u64 = 1;

/// The index heads the score kernel runs: `HEADS`
/// (gpu-deepseek41/src/indexer.rs:68), a kernel constant, so a file that
/// names another is refused rather than planned for a launch that does not
/// exist.
const INDEX_HEADS: usize = 32;

/// The hyperparameters of one `mistral-large4` file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`, the trunk: the DSpark stages ship in a file of their
    /// own, so every block is a trunk layer.
    pub n_layer: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `context_length`.
    pub n_ctx: usize,
    /// `vocab_size`, else the token list's length.
    pub n_vocab: usize,
    /// `attention.layer_norm_rms_epsilon`: every block norm's.
    pub rms_eps: f32,
    /// `attention.layer_norm_epsilon`: the indexer's LayerNorms'.
    pub idx_norm_eps: f32,
    /// `attention.head_count`.
    pub n_head: usize,
    /// `attention.kv_lora_rank`: the latent's values.
    pub kv_lora: usize,
    /// `attention.key_length`: the roped layer's whole cached row.
    pub key_length: usize,
    /// `attention.key_length_mla`: values per head, unrotated and rotated
    /// together.
    pub head_k: usize,
    /// `attention.value_length_mla`: value values per head.
    pub head_v: usize,
    /// `rope.dimension_count`.
    pub rope_dims: usize,
    /// `rope.freq_base`.
    pub rope_base: f32,
    /// YaRN's `factor`, `original_context_length`, betas and log multiplier.
    pub yarn: YarnKeys,
    /// `attention.rope_pattern`: whether each layer ropes.
    pub rope_layer: Vec<bool>,
    /// `attention.indexer.types`: whether each layer runs its own indexer.
    pub indexer_full: Vec<bool>,
    /// `attention.kv_source_map`: the layer whose cache each layer reads.
    pub kv_source: Vec<usize>,
    /// `attention.indexer.head_count`.
    pub indexer_heads: usize,
    /// `attention.indexer.key_length`.
    pub indexer_d: usize,
    /// `attention.indexer.top_k`.
    pub indexer_top_k: usize,
    /// `leading_dense_block_count`.
    pub dense_lead: usize,
    /// `feed_forward_length`: the dense lead's width.
    pub dense_ff: usize,
    /// `expert_count`.
    pub n_expert: usize,
    /// `expert_used_count`.
    pub n_used: usize,
    /// `expert_feed_forward_length`: one expert's values.
    pub expert_ff: usize,
    /// `expert_shared_count`.
    pub n_shared: usize,
    /// `expert_weights_norm`: the kept weights renormalized to sum to one.
    pub weights_norm: bool,
    /// `expert_weights_scale`: the factor on the kept weights.
    pub weights_scale: f32,
    /// `final_logit_softcapping`: the tanh softcap's bound; 0: no cap.
    pub softcap: f32,
    /// The key defaults taken, each with its value and why.
    pub defaults: Vec<String>,
}

/// The YaRN keys, as the file spells them (`rope.scaling.*`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YarnKeys {
    /// `factor`.
    pub factor: f32,
    /// `original_context_length`, positions.
    pub orig_ctx: usize,
    /// `yarn_beta_fast`.
    pub beta_fast: f32,
    /// `yarn_beta_slow`.
    pub beta_slow: f32,
    /// `yarn_log_multiplier`: 0 when the file carries none (llama.cpp's
    /// default), 1 in this family's files.
    pub log_mul: f32,
}

impl YarnKeys {
    /// The net magnitude on the rotated values, llama.cpp's
    /// `yarn_attn_factor` times ggml's own scale (llama-context.cpp:277-303):
    /// with no log multiplier, or one of 1, the two cancel and the net is 1 —
    /// YaRN without magnitude scaling, the only form the converter writes
    /// (`assert not self.hparams["yarn"].get("apply_scale", True)`).
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        if self.log_mul == 0.0 {
            return 1.0;
        }
        let ln = self.factor.ln();
        (1.0 + 0.1 * ln) / (1.0 + 0.1 * self.log_mul * ln)
    }
}

/// Every `mistral-large4.` key [`Hparams::read`] reads, by suffix.
const READ: &[&str] = &[
    "block_count",
    "context_length",
    "embedding_length",
    "vocab_size",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.key_length_mla",
    "attention.kv_lora_rank",
    "attention.q_lora_rank",
    "attention.value_length",
    "attention.value_length_mla",
    "attention.layer_norm_rms_epsilon",
    "attention.layer_norm_epsilon",
    "attention.rope_pattern",
    "attention.indexer.head_count",
    "attention.indexer.key_length",
    "attention.indexer.top_k",
    "attention.indexer.types",
    "attention.kv_source_map",
    "rope.dimension_count",
    "rope.freq_base",
    "rope.scaling.type",
    "rope.scaling.factor",
    "rope.scaling.original_context_length",
    "rope.scaling.yarn_beta_fast",
    "rope.scaling.yarn_beta_slow",
    "rope.scaling.yarn_log_multiplier",
    "leading_dense_block_count",
    "feed_forward_length",
    "final_logit_softcapping",
    "expert_count",
    "expert_used_count",
    "expert_feed_forward_length",
    "expert_shared_count",
    "expert_gating_func",
    "expert_group_count",
    "expert_group_used_count",
    "expert_weights_norm",
    "expert_weights_scale",
];

/// The architecture keys and `general.sampling.*` keys of `split` that
/// [`Hparams::read`] does not read: listed, not refused.
#[must_use]
pub fn unread_keys(split: &Split) -> Vec<String> {
    unread_keys_of(split, READ)
}

impl Hparams {
    /// Every value from `split`'s headers. A missing key, a key of a value
    /// the model cannot hold (another gating func, another index head count,
    /// a rope wider than the head or the index head, a key length the rope
    /// and latent do not sum to, a query latent, a KV source above its layer,
    /// itself a reader or of another rope-ness, a shared first indexer), or a
    /// tensor missing or of other dims than the keys give is an error naming
    /// the key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let n_layer = positive(split, "block_count")?;
        let n_embd = positive(split, "embedding_length")?;
        let n_ctx = positive(split, "context_length")?;
        let rms_eps = finite_positive(split, "attention.layer_norm_rms_epsilon")?;
        // The indexer's LayerNorms run at their own epsilon (mistral-large4.cpp
        // reads `attention.layer_norm_eps` for them).
        let idx_norm_eps = finite_positive(split, "attention.layer_norm_epsilon")?;
        let n_head = positive(split, "attention.head_count")?;
        // MLA: one latent row a position; the file's own count is held to it.
        let n_head_kv = positive(split, "attention.head_count_kv")?;
        if n_head_kv != 1 {
            return Err(metadata(
                split,
                "attention.head_count_kv",
                format!("is {n_head_kv}; the latent cache is one row a position"),
            ));
        }
        let kv_lora = positive(split, "attention.kv_lora_rank")?;
        let key_length = positive(split, "attention.key_length")?;
        let head_k = positive(split, "attention.key_length_mla")?;
        let head_v = positive(split, "attention.value_length_mla")?;
        let value_length = positive(split, "attention.value_length")?;
        if value_length != kv_lora {
            return Err(metadata(
                split,
                "attention.value_length",
                format!("is {value_length}, the kv_lora_rank {kv_lora} it must equal"),
            ));
        }
        let rope_dims = positive(split, "rope.dimension_count")?;
        if rope_dims % 2 != 0 || rope_dims > head_k {
            return Err(metadata(
                split,
                "rope.dimension_count",
                format!("is {rope_dims}, for a head of {head_k}"),
            ));
        }
        if key_length != kv_lora + rope_dims {
            return Err(metadata(
                split,
                "attention.key_length",
                format!(
                    "is {key_length}, not kv_lora_rank {kv_lora} + rope.dimension_count \
                     {rope_dims} (the rope part sits beside the latent in the row)"
                ),
            ));
        }
        // The query is the hidden row's own projection: the converter writes no
        // q_lora (`assert self.hparams.get("q_lora_rank") is None`).
        if optional_usize(split, "attention.q_lora_rank")?.is_some_and(|q| q != 0) {
            return Err(metadata(
                split,
                "attention.q_lora_rank",
                "is carried nonzero; the query is the hidden row's own projection, and the \
                 converter writes no q_lora",
            ));
        }
        let rope_layer = per_layer_bool(split, "attention.rope_pattern", n_layer)?;
        let indexer_full = per_layer_bool(split, "attention.indexer.types", n_layer)?;
        if !indexer_full[0] {
            return Err(metadata(
                split,
                "attention.indexer.types",
                "is 0 (a shared indexer) at layer 0; the first layer must have a full indexer",
            ));
        }
        let kv_source = kv_source_map(split, n_layer)?;
        for l in 0..n_layer {
            let src = kv_source[l];
            let detail = if src > l {
                format!("is {src} at layer {l}, above it")
            } else if kv_source[src] != src {
                format!(
                    "is {src} at layer {l}, itself a reader of layer {}",
                    kv_source[src]
                )
            } else if rope_layer[src] != rope_layer[l] {
                format!(
                    "is {src} for layer {l}, and the pair's rope-ness differs (the source {} \
                     and the reader {})",
                    if rope_layer[src] {
                        "ropes"
                    } else {
                        "ropes not"
                    },
                    if rope_layer[l] { "ropes" } else { "ropes not" },
                )
            } else {
                continue;
            };
            return Err(metadata(split, "attention.kv_source_map", detail));
        }
        let indexer_heads = positive(split, "attention.indexer.head_count")?;
        if indexer_heads != INDEX_HEADS {
            return Err(metadata(
                split,
                "attention.indexer.head_count",
                format!("is {indexer_heads}; the score kernel runs {INDEX_HEADS} index heads"),
            ));
        }
        let indexer_d = positive(split, "attention.indexer.key_length")?;
        // The indexer head carries its rope part first (mistral-large4.cpp:
        // `GGML_ASSERT(n_rot <= n_embd_indexer_head)`).
        if rope_dims > indexer_d {
            return Err(metadata(
                split,
                "attention.indexer.key_length",
                format!("is {indexer_d}, under the rope width {rope_dims} its head carries first"),
            ));
        }
        let indexer_top_k = positive(split, "attention.indexer.top_k")?;
        let rope_base = finite_positive(split, "rope.freq_base")?;
        let scaling = "rope.scaling.type";
        let kind = meta_str(split, scaling)?;
        if kind != "yarn" {
            return Err(metadata(
                split,
                scaling,
                format!("is {kind:?}; this reader runs the model's YaRN"),
            ));
        }
        let factor = finite_positive(split, "rope.scaling.factor")?;
        let orig_ctx = positive(split, "rope.scaling.original_context_length")?;
        let beta_fast = finite_positive(split, "rope.scaling.yarn_beta_fast")?;
        let beta_slow = finite_positive(split, "rope.scaling.yarn_beta_slow")?;
        let mut defaults = Vec::new();
        let log_mul = match optional_f32(split, "rope.scaling.yarn_log_multiplier")? {
            Some(v) => v,
            None => {
                defaults.push(
                    "rope.scaling.yarn_log_multiplier = 0 (llama.cpp's default; the net \
                     magnitude is 1 either way)"
                        .to_string(),
                );
                0.0
            }
        };
        let (n_expert, n_used) = expert_counts(split)?;
        let expert_ff = positive(split, "expert_feed_forward_length")?;
        let n_shared = positive(split, "expert_shared_count")?;
        // The dialect builds the softmax router (mistral-large4.cpp passes the
        // key to build_moe_ffn's softmax arm), so a file that says another is
        // refused; the key is the file's own, no default.
        let gating = meta_u64(split, "expert_gating_func")?;
        if gating != SOFTMAX {
            return Err(metadata(
                split,
                "expert_gating_func",
                format!("is {gating}; this reader reads the softmax router ({SOFTMAX})"),
            ));
        }
        one_expert_group(split, "the mistral-large4 router")?;
        let weights_norm = meta_bool(split, "expert_weights_norm")?;
        let weights_scale = finite_positive(split, "expert_weights_scale")?;
        let dense_lead = match optional_usize(split, "leading_dense_block_count")? {
            Some(v) => v,
            None => {
                defaults.push("leading_dense_block_count = 0 (no dense layer)".to_string());
                0
            }
        };
        if dense_lead > n_layer {
            return Err(metadata(
                split,
                "leading_dense_block_count",
                format!("is {dense_lead} of block_count {n_layer}"),
            ));
        }
        let dense_ff = positive(split, "feed_forward_length")?;
        let softcap = match optional_f32(split, "final_logit_softcapping")? {
            Some(v) => v,
            None => {
                defaults
                    .push("final_logit_softcapping = 0 (no cap, llama.cpp's default)".to_string());
                0.0
            }
        };
        if !softcap.is_finite() || softcap < 0.0 {
            return Err(metadata(
                split,
                "final_logit_softcapping",
                format!("is {softcap}, not a finite nonnegative bound"),
            ));
        }
        let hp = Hparams {
            n_layer,
            n_embd,
            n_ctx,
            n_vocab: n_vocab(split)?,
            rms_eps,
            idx_norm_eps,
            n_head,
            kv_lora,
            key_length,
            head_k,
            head_v,
            rope_dims,
            rope_base,
            yarn: YarnKeys {
                factor,
                orig_ctx,
                beta_fast,
                beta_slow,
                log_mul,
            },
            rope_layer,
            indexer_full,
            kv_source,
            indexer_heads,
            indexer_d,
            indexer_top_k,
            dense_lead,
            dense_ff,
            n_expert,
            n_used,
            expert_ff,
            n_shared,
            weights_norm,
            weights_scale,
            softcap,
            defaults,
        };
        hp.tensors_agree(split)?;
        Ok(hp)
    }

    /// Every tensor the keys name, present at its dims: a missing one is
    /// refused by name with the count of the others, a present one of other
    /// dims by its want.
    fn tensors_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let mut want = vec![
            (
                names::token_embd(),
                vec![self.n_embd as u64, self.n_vocab as u64],
            ),
            (
                names::output(),
                vec![self.n_embd as u64, self.n_vocab as u64],
            ),
            (names::output_norm(), vec![self.n_embd as u64]),
            (ROPE_FREQS.to_string(), vec![self.rope_dims as u64 / 2]),
        ];
        for l in 0..self.n_layer {
            want.extend(
                self.want(l)
                    .into_iter()
                    .map(|(stem, dims)| (format!("blk.{l}.{stem}"), dims)),
            );
        }
        tensors_agree(split, want)
    }

    /// The dims the keys give layer `l`, ggml order: the layer's own kind
    /// decides which stems it carries.
    pub(crate) fn want(&self, l: usize) -> Vec<(&'static str, Vec<u64>)> {
        let e = self.n_embd as u64;
        let h = self.n_head as u64;
        // The unrotated part of a head grows by the rope width a NoPE layer
        // does not carry (mistral-large4.cpp: `n_embd_head_qk_nope`).
        let pe = self.pe(l) as u64;
        let qk = self.head_k as u64 - pe;
        let mut out = vec![
            (names::ATTN_NORM, vec![e]),
            (names::ATTN_Q, vec![e, h * self.head_k as u64]),
            (names::ATTN_K_B, vec![qk, self.kv_lora as u64, h]),
            (
                names::ATTN_V_B,
                vec![self.kv_lora as u64, self.head_v as u64, h],
            ),
            (names::ATTN_OUTPUT, vec![h * self.head_v as u64, e]),
            (names::ATTN_SSMAX_W, vec![h]),
            (names::ATTN_SSMAX_B, vec![h]),
        ];
        if self.owns_cache(l) {
            out.push((names::ATTN_KV_A_MQA, vec![e, self.kv_lora as u64 + pe]));
            out.push((names::ATTN_KV_A_NORM, vec![self.kv_lora as u64]));
        }
        if self.indexer_full[l] {
            let (ih, d) = (self.indexer_heads as u64, self.indexer_d as u64);
            out.extend([
                (names::INDEXER_Q_NORM_W, vec![e]),
                (names::INDEXER_Q_NORM_B, vec![e]),
                (names::INDEXER_ATTN_Q_B, vec![e, ih * d]),
                (names::INDEXER_ATTN_K, vec![e, d]),
                (names::INDEXER_K_NORM_W, vec![d]),
                (names::INDEXER_K_NORM_B, vec![d]),
                (names::INDEXER_PROJ, vec![e, ih]),
                (names::INDEXER_SSMAX_W, vec![ih]),
                (names::INDEXER_SSMAX_B, vec![ih]),
            ]);
        }
        out.push((names::FFN_NORM, vec![e]));
        if l < self.dense_lead {
            let ff = self.dense_ff as u64;
            out.extend([
                (names::FFN_GATE, vec![e, ff]),
                (names::FFN_UP, vec![e, ff]),
                (names::FFN_DOWN, vec![ff, e]),
            ]);
        } else {
            let (x, w) = (self.n_expert as u64, self.expert_ff as u64);
            let s = w * self.n_shared as u64;
            out.extend([
                (names::FFN_GATE_INP, vec![e, x]),
                (names::FFN_GATE_EXPS, vec![e, w, x]),
                (names::FFN_UP_EXPS, vec![e, w, x]),
                (names::FFN_DOWN_EXPS, vec![w, e, x]),
                (names::FFN_GATE_SHEXP, vec![e, s]),
                (names::FFN_UP_SHEXP, vec![e, s]),
                (names::FFN_DOWN_SHEXP, vec![s, e]),
            ]);
        }
        out
    }

    /// The rope values in layer `l`'s cached row: the rope width on a layer
    /// that ropes, 0 on a NoPE layer, whose row is the latent alone.
    #[must_use]
    pub fn pe(&self, l: usize) -> usize {
        if self.rope_layer[l] {
            self.rope_dims
        } else {
            0
        }
    }

    /// Whether layer `l` writes the cache it reads.
    #[must_use]
    pub fn owns_cache(&self, l: usize) -> bool {
        self.kv_source[l] == l
    }

    /// The last layer below `l` that runs its own indexer: whose top-k list
    /// and index keys a shared layer reads ("shared indexer layers reuse the
    /// top_k from the previous full indexer layer", mistral-large4.cpp). The
    /// first layer is full, so one is always there.
    #[must_use]
    pub fn prev_full_indexer(&self, l: usize) -> usize {
        (0..l)
            .rev()
            .find(|&i| self.indexer_full[i])
            .expect("the first layer has a full indexer, so a later layer has one below it")
    }
}

/// `attention.kv_source_map` as one layer index per layer: an array of
/// exactly `n_layer` indices, every one of them a layer index.
fn kv_source_map(split: &Split, n_layer: usize) -> Result<Vec<usize>, PlacementError> {
    let key = "attention.kv_source_map";
    let items = meta_arr(split, key)?;
    if items.len() != n_layer {
        return Err(metadata(
            split,
            key,
            format!("has {} values for {n_layer} layers", items.len()),
        ));
    }
    items
        .iter()
        .enumerate()
        .map(|(l, v)| {
            v.as_unsigned()
                .and_then(|x| usize::try_from(x).ok())
                .ok_or_else(|| metadata(split, key, format!("has no layer index at layer {l}")))
        })
        .collect()
}
