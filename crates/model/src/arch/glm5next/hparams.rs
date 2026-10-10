//! Every `glm5next.*` key the description reads, and each layer's kind from
//! the tensors it carries. A key ik's loader requires is required here; a key
//! it reads with a fallback of its own keeps that fallback, recorded in
//! [`Hparams::defaults`] with the line that sets it; a key it reads as
//! optional with no fallback (a zero it would run with) is required here.

use gguf::{Split, Value};

use super::roles::{DENSE, HC, KDA, LATENT, MOE, NEXTN, SHARED, required};
use crate::arch::{
    meta_arr, meta_bool, meta_f32, meta_u64, meta_usize, metadata, n_vocab, nextn_layers,
};
use crate::placement::PlacementError;

/// ik's `LLM_EXPERT_GATING_FUNC_SIGMOID` (`src/llama-hparams.h`).
const SIGMOID: u64 = 2;

/// What a layer mixes with: `attention.head_count_kv` 0 marks a KDA layer,
/// 1 a latent one (:2388-2391), and the tensors must agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `attn_q`, `ssm_*`: a KDA delta-rule layer.
    Kda,
    /// `attn_q_a`, `attn_kv_a_mqa`, the indexer: a latent attention layer.
    Latent,
}

/// The token-pool indexer's keys.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Indexer {
    /// `attention.indexer.head_count`.
    pub n_head: usize,
    /// `attention.indexer.key_length`.
    pub head_dim: usize,
    /// `attention.indexer.top_k`: tokens kept, a whole number of pools.
    pub top_k: usize,
    /// `attention.indexer.kpool`: tokens per pooled key.
    pub kpool: usize,
}

/// The hyper-connection keys.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hc {
    /// `hyper_connection.count`.
    pub streams: usize,
    /// `hyper_connection.sinkhorn_iterations`.
    pub sinkhorn: usize,
    /// `hyper_connection.epsilon`.
    pub eps: f32,
}

/// The hyperparameters of one glm5next file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`: the trunk and the next-token layers.
    pub n_layer: usize,
    /// The trunk's layers: `block_count` less `nextn_predict_layers`.
    pub n_trunk: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `attention.head_count`: latent query heads, and KDA key and value heads.
    pub n_head: usize,
    /// `vocab_size`, else the token list's length.
    pub n_vocab: usize,
    /// `context_length`.
    pub n_ctx_train: usize,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
    /// `attention.layer_norm_epsilon`: the index keys' LayerNorm.
    pub norm_eps: f32,
    /// `attention.q_lora_rank`.
    pub q_lora: usize,
    /// `attention.kv_lora_rank`: the cached latent row.
    pub kv_lora: usize,
    /// `attention.key_length_mla`: a query head's values before absorption.
    pub head_k: usize,
    /// `attention.value_length_mla`: a head's output values.
    pub head_v: usize,
    /// `ssm.conv_kernel`.
    pub conv: usize,
    /// `kda.head_dim`: values per KDA key and value head.
    pub kda_head_dim: usize,
    /// `kda.gate_lower_bound`: the decay's lower bound, below 0.
    pub gate_lower_bound: f32,
    pub indexer: Indexer,
    /// `attention.indexer.index_share_mtp`: whether a draft chain's later
    /// iterations reuse the first one's selection; `None`: the key is absent,
    /// which takes no default. A proposal is one id (the app's GLM
    /// `MtpBody::WIDTH`, held at 1 by a const assertion beside it; the NextN
    /// chain refuses a second walk), so no iteration follows the first and
    /// the value changes nothing; a wider proposal reads it first.
    pub index_share_mtp: Option<bool>,
    pub hc: Hc,
    /// `expert_count`.
    pub n_expert: usize,
    /// `expert_used_count`.
    pub n_used: usize,
    /// `expert_feed_forward_length`.
    pub expert_ff: usize,
    /// `expert_shared_count`.
    pub n_shared: usize,
    /// `expert_shared_feed_forward_length`.
    pub shared_ff: usize,
    /// `leading_dense_block_count`.
    pub dense_lead: usize,
    /// `feed_forward_length`: a dense layer's; 0 when none is dense.
    pub dense_ff: usize,
    /// `expert_weights_norm`.
    pub weights_norm: bool,
    /// `expert_weights_scale`.
    pub weights_scale: f32,
    /// `swiglu_clamp_exp`, per layer: the routed experts' limit.
    pub limit_exp: Vec<f32>,
    /// `swiglu_clamp_shexp`, per layer: the shared expert's and a dense layer's.
    pub limit_shexp: Vec<f32>,
    /// Every layer's kind, the next-token layers included.
    pub kinds: Vec<Kind>,
    /// The keys this read took a default for: the key, the value, the line.
    pub defaults: Vec<String>,
}

/// Every `glm5next.` key [`Hparams::read`] reads, by suffix.
const READ: &[&str] = &[
    "block_count",
    "nextn_predict_layers",
    "embedding_length",
    "context_length",
    "vocab_size",
    "feed_forward_length",
    "leading_dense_block_count",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.layer_norm_rms_epsilon",
    "attention.layer_norm_epsilon",
    "attention.q_lora_rank",
    "attention.kv_lora_rank",
    "attention.key_length",
    "attention.value_length",
    "attention.key_length_mla",
    "attention.value_length_mla",
    "rope.dimension_count",
    "ssm.conv_kernel",
    "ssm.state_size",
    "ssm.group_count",
    "kda.head_dim",
    "kda.gate_lower_bound",
    "attention.indexer.head_count",
    "attention.indexer.key_length",
    "attention.indexer.top_k",
    "attention.indexer.kpool",
    "attention.indexer.index_share_mtp",
    "hyper_connection.count",
    "hyper_connection.sinkhorn_iterations",
    "hyper_connection.epsilon",
    "expert_count",
    "expert_used_count",
    "expert_group_count",
    "expert_group_used_count",
    "expert_gating_func",
    "expert_weights_scale",
    "expert_weights_norm",
    "expert_feed_forward_length",
    "expert_shared_feed_forward_length",
    "expert_shared_count",
    "swiglu_clamp_exp",
    "swiglu_clamp_shexp",
];

/// The architecture keys and `general.sampling.*` keys of `split` that
/// [`Hparams::read`] does not read: listed, not refused.
#[must_use]
pub fn unread_keys(split: &Split) -> Vec<String> {
    let prefix = split.arch_key("");
    split
        .iter_kv()
        .map(|(k, _)| k)
        .filter(|k| match k.strip_prefix(prefix.as_str()) {
            Some(suffix) => !READ.contains(&suffix),
            None => k.starts_with("general.sampling."),
        })
        .map(str::to_string)
        .collect()
}

impl Hparams {
    /// Every value from `split`'s headers; a missing required key, a value the
    /// description cannot hold, or a layer whose tensors disagree with the
    /// keys is an error naming the key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let mut defaults = Vec::new();
        let n_layer = meta_usize(split, "block_count")?;
        let n_nextn = nextn(split, n_layer, &mut defaults)?;
        let n_trunk = n_layer - n_nextn;
        let n_embd = meta_usize(split, "embedding_length")?;
        let n_head = meta_usize(split, "attention.head_count")?;
        if n_head == 0 {
            return Err(metadata(split, "attention.head_count", "is 0"));
        }
        let rms_eps = meta_f32(split, "attention.layer_norm_rms_epsilon")?;
        // The index keys' LayerNorm epsilon: ik falls back to the RMS epsilon
        // (:2293-2297), a different norm's; this reader requires the key.
        let norm_eps = meta_f32(split, "attention.layer_norm_epsilon")?;
        let q_lora = positive(split, "attention.q_lora_rank")?;
        let kv_lora = positive(split, "attention.kv_lora_rank")?;
        let head_k = positive(split, "attention.key_length_mla")?;
        let head_v = positive(split, "attention.value_length_mla")?;
        let rope = meta_usize(split, "rope.dimension_count")?;
        if rope != 0 {
            return Err(metadata(
                split,
                "rope.dimension_count",
                format!("is {rope}; this reader reads rope-free latent attention"),
            ));
        }
        // The cached row is the latent and its rope values, which are none.
        for key in ["attention.key_length", "attention.value_length"] {
            if let Some(v) = optional_usize(split, key)?
                && v != kv_lora
            {
                return Err(metadata(
                    split,
                    key,
                    format!("is {v}, not kv_lora_rank {kv_lora} with no rope values"),
                ));
            }
        }
        let conv = positive(split, "ssm.conv_kernel")?;
        let kda_head_dim = match optional_usize(split, "kda.head_dim")? {
            Some(v) => v,
            None => {
                let v = meta_usize(split, "ssm.state_size")?;
                defaults.push(format!("kda.head_dim = {v} (ssm.state_size, :2370-2374)"));
                v
            }
        };
        if kda_head_dim == 0 {
            return Err(metadata(split, "kda.head_dim", "is 0"));
        }
        if let Some(g) = optional_usize(split, "ssm.group_count")?
            && g != n_head
        {
            return Err(metadata(
                split,
                "ssm.group_count",
                format!("is {g}; ik's KDA layer takes attention.head_count {n_head} key heads"),
            ));
        }
        let gate_lower_bound = or_default(
            "kda.gate_lower_bound",
            optional_f32(split, "kda.gate_lower_bound")?,
            -5.0,
            ":2375-2378",
            &mut defaults,
        );
        if !gate_lower_bound.is_finite() || gate_lower_bound >= 0.0 {
            return Err(metadata(
                split,
                "kda.gate_lower_bound",
                format!("is {gate_lower_bound}; the bounded decay needs a finite bound below 0"),
            ));
        }
        let indexer = indexer(split)?;
        // llama.cpp's draft chain reuses its first iteration's selection in the
        // later ones when this is true; ik reads no such key. No lever or flag
        // here runs a second iteration (the field's doc names the width), so
        // either value runs the same walk: the key is kept, not acted on.
        let index_share_mtp = optional_bool(split, "attention.indexer.index_share_mtp")?;
        let hc = Hc {
            streams: positive(split, "hyper_connection.count")?,
            sinkhorn: positive(split, "hyper_connection.sinkhorn_iterations")?,
            eps: or_default(
                "hyper_connection.epsilon",
                optional_f32(split, "hyper_connection.epsilon")?,
                rms_eps,
                "the RMS epsilon, :2358-2360",
                &mut defaults,
            ),
        };
        let n_expert = meta_usize(split, "expert_count")?;
        let n_used = meta_usize(split, "expert_used_count")?;
        if n_expert == 0 || n_used == 0 || n_used > n_expert {
            return Err(metadata(
                split,
                "expert_used_count",
                format!("is {n_used} of expert_count {n_expert}"),
            ));
        }
        for key in ["expert_group_count", "expert_group_used_count"] {
            if let Some(v) = optional_usize(split, key)?
                && v != 1
            {
                return Err(metadata(
                    split,
                    key,
                    format!("is {v}; ik's glm5next router takes one group"),
                ));
            }
        }
        let gating = match split.value(&split.arch_key("expert_gating_func")) {
            Some(_) => meta_u64(split, "expert_gating_func")?,
            None => {
                defaults.push("expert_gating_func = sigmoid (:2330-2333)".to_string());
                SIGMOID
            }
        };
        if gating != SIGMOID {
            return Err(metadata(
                split,
                "expert_gating_func",
                format!("is {gating}; this reader reads the sigmoid router ({SIGMOID})"),
            ));
        }
        // ik divides the picks' weights by their sum only `if (norm_w)`
        // (llm_build_context :1548); the body runs that rule alone.
        let weights_norm = meta_bool(split, "expert_weights_norm")?;
        if !weights_norm {
            return Err(metadata(
                split,
                "expert_weights_norm",
                "is false; the glm5next body runs only the router that renormalizes its picks",
            ));
        }
        let expert_ff = positive(split, "expert_feed_forward_length")?;
        let n_shared = or_default(
            "expert_shared_count",
            optional_usize(split, "expert_shared_count")?,
            0,
            "no shared expert, :2321",
            &mut defaults,
        );
        let shared_ff = or_default(
            "expert_shared_feed_forward_length",
            optional_usize(split, "expert_shared_feed_forward_length")?,
            expert_ff,
            "expert_feed_forward_length, :2323-2326",
            &mut defaults,
        );
        let dense_lead = or_default(
            "leading_dense_block_count",
            optional_usize(split, "leading_dense_block_count")?,
            0,
            ":2327",
            &mut defaults,
        );
        if dense_lead > n_trunk {
            return Err(metadata(
                split,
                "leading_dense_block_count",
                format!("is {dense_lead}, past the trunk's {n_trunk} layers"),
            ));
        }
        let dense_ff = if dense_lead > 0 {
            positive(split, "feed_forward_length")?
        } else {
            0
        };
        let limit_exp = per_layer_f32(split, "swiglu_clamp_exp", n_layer)?;
        let limit_shexp = if split.value(&split.arch_key("swiglu_clamp_shexp")).is_some() {
            per_layer_f32(split, "swiglu_clamp_shexp", n_layer)?
        } else {
            defaults.push("swiglu_clamp_shexp = swiglu_clamp_exp (:2336-2338)".to_string());
            limit_exp.clone()
        };
        let kinds = kinds(split, n_layer)?;
        let hp = Hparams {
            n_layer,
            n_trunk,
            n_embd,
            n_head,
            n_vocab: n_vocab(split)?,
            n_ctx_train: meta_usize(split, "context_length")?,
            rms_eps,
            norm_eps,
            q_lora,
            kv_lora,
            head_k,
            head_v,
            conv,
            kda_head_dim,
            gate_lower_bound,
            indexer,
            index_share_mtp,
            hc,
            n_expert,
            n_used,
            expert_ff,
            n_shared,
            shared_ff,
            dense_lead,
            dense_ff,
            weights_norm,
            weights_scale: meta_f32(split, "expert_weights_scale")?,
            limit_exp,
            limit_shexp,
            kinds,
            defaults,
        };
        hp.tensors_agree(split)?;
        hp.kda_dims_agree(split)?;
        Ok(hp)
    }

    /// Every layer carries its kind's required tensors (the stem table in
    /// `roles`), its feed-forward block's and its residual's, and nothing of
    /// the other kinds': the first missing or misplaced name, with how many
    /// more there are.
    fn tensors_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let has = |l: usize, stem: &str| split.find(&format!("blk.{l}.{stem}")).is_some();
        let mut missing = Vec::new();
        let mut misplaced = Vec::new();
        for (l, &kind) in self.kinds.iter().enumerate() {
            let trunk = l < self.n_trunk;
            let dense = l < self.dense_lead;
            let (mixer, other) = match kind {
                Kind::Kda => (KDA, LATENT),
                Kind::Latent => (LATENT, KDA),
            };
            let mut need: Vec<&str> = required(mixer).collect();
            need.extend(required(if dense { DENSE } else { MOE }));
            if !dense && self.n_shared > 0 {
                need.extend(required(SHARED));
            }
            need.extend(required(if trunk { HC } else { NEXTN }));
            missing.extend(
                need.iter()
                    .filter(|s| !has(l, s))
                    .map(|s| format!("blk.{l}.{s}")),
            );
            let mut banned: Vec<&str> = other
                .iter()
                .map(|s| s.name)
                .filter(|s| !need.contains(s))
                .collect();
            match (trunk, dense) {
                (true, true) => banned.push("ffn_gate_inp.weight"),
                (true, false) => banned.push("ffn_gate.weight"),
                (false, _) => banned.extend(HC.iter().map(|s| s.name)),
            }
            misplaced.extend(
                banned
                    .iter()
                    .filter(|s| has(l, s))
                    .map(|s| format!("blk.{l}.{s}")),
            );
        }
        if let Some(name) = missing.first() {
            return Err(PlacementError::Tensor {
                name: name.clone(),
                detail: format!(
                    "is not in the file ({} tensor(s) the layers' kinds need are missing: {})",
                    missing.len(),
                    missing.join(", ")
                ),
            });
        }
        if let Some(name) = misplaced.first() {
            return Err(PlacementError::Tensor {
                name: name.clone(),
                detail: format!(
                    "is in the file, and its layer's kind carries none ({} such: {})",
                    misplaced.len(),
                    misplaced.join(", ")
                ),
            });
        }
        Ok(())
    }

    /// A KDA layer's tensors by stem and the dims ik creates them with
    /// (`create_glm5next_tensors`), ggml order: `d_state` is `kda.head_dim`,
    /// `d_inner` that times the heads.
    pub(super) fn kda_dims(&self) -> [(&'static str, Vec<u64>); 16] {
        let (e, h, st) = (
            self.n_embd as u64,
            self.n_head as u64,
            self.kda_head_dim as u64,
        );
        let inner = st * h;
        let conv = self.conv as u64;
        [
            ("attn_norm.weight", vec![e]),
            ("attn_q.weight", vec![e, inner]),
            ("attn_k.weight", vec![e, inner]),
            ("attn_v.weight", vec![e, inner]),
            ("ssm_conv1d_q.weight", vec![conv, 1, inner]),
            ("ssm_conv1d_k.weight", vec![conv, 1, inner]),
            ("ssm_conv1d_v.weight", vec![conv, 1, inner]),
            ("ssm_f_a.weight", vec![e, st]),
            ("ssm_f_b.weight", vec![st, inner]),
            ("ssm_g_a.weight", vec![e, st]),
            ("ssm_g_b.weight", vec![st, inner]),
            ("ssm_beta.weight", vec![e, h]),
            ("ssm_a", vec![h]),
            ("ssm_dt.bias", vec![inner]),
            ("ssm_norm.weight", vec![st]),
            ("attn_output.weight", vec![inner, e]),
        ]
    }

    /// Every trunk KDA layer's tensors hold exactly [`Hparams::kda_dims`], as
    /// ik requires: the kernels check their buffers' lengths only from below,
    /// so a short tensor would leave the shared scratch's tail to the last
    /// layer and a long one would be read in part. The first tensor of other
    /// dims, by name, with how many more there are.
    fn kda_dims_agree(&self, split: &Split) -> Result<(), PlacementError> {
        let pad = |d: &[u64]| {
            let mut d = d.to_vec();
            while d.last() == Some(&1) {
                d.pop();
            }
            d
        };
        let want = self.kda_dims();
        let mut off = Vec::new();
        for (l, &kind) in self.kinds.iter().enumerate().take(self.n_trunk) {
            if kind != Kind::Kda {
                continue;
            }
            for (stem, dims) in &want {
                let name = format!("blk.{l}.{stem}");
                if let Some((_, info)) = split.find(&name)
                    && pad(&info.dims) != pad(dims)
                {
                    off.push((name, format!("{:?} (want {dims:?})", info.dims)));
                }
            }
        }
        match off.first() {
            None => Ok(()),
            Some((name, _)) => Err(PlacementError::Tensor {
                name: name.clone(),
                detail: format!(
                    "has other dims than ik creates the KDA tensor with ({} such: {})",
                    off.len(),
                    off.iter()
                        .map(|(n, d)| format!("{n} {d}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }),
        }
    }
}

/// `nextn_predict_layers`, optional (:2301); believed only when the first
/// next-token layer carries `nextn.eh_proj`, as ik does (:2302-2308).
fn nextn(
    split: &Split,
    n_layer: usize,
    defaults: &mut Vec<String>,
) -> Result<usize, PlacementError> {
    nextn_layers(split, n_layer, ":2301", ":2302-2308", defaults)
}

/// The token-pool indexer's keys, required: a latent layer without the
/// indexer would run dense attention in ik, and this reader refuses it.
fn indexer(split: &Split) -> Result<Indexer, PlacementError> {
    let ix = Indexer {
        n_head: positive(split, "attention.indexer.head_count")?,
        head_dim: positive(split, "attention.indexer.key_length")?,
        top_k: positive(split, "attention.indexer.top_k")?,
        kpool: positive(split, "attention.indexer.kpool")?,
    };
    if !ix.top_k.is_multiple_of(ix.kpool) {
        return Err(metadata(
            split,
            "attention.indexer.top_k",
            format!(
                "is {}, not a whole number of pools of {}",
                ix.top_k, ix.kpool
            ),
        ));
    }
    Ok(ix)
}

/// Every layer's kind from `attention.head_count_kv` (:2388-2391), each
/// layer's own marker tensor agreeing: `attn_q` for KDA, `attn_q_a` latent.
fn kinds(split: &Split, n_layer: usize) -> Result<Vec<Kind>, PlacementError> {
    let key = "attention.head_count_kv";
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
            let said = match v.as_unsigned() {
                Some(0) => Kind::Kda,
                Some(1) => Kind::Latent,
                other => {
                    return Err(metadata(
                        split,
                        key,
                        format!("is {other:?} at layer {l}; 0 marks KDA, 1 latent attention"),
                    ));
                }
            };
            let q = format!("blk.{l}.attn_q.weight");
            let q_a = format!("blk.{l}.attn_q_a.weight");
            let kind = match (split.find(&q).is_some(), split.find(&q_a).is_some()) {
                (true, false) => Kind::Kda,
                (false, true) => Kind::Latent,
                (both, _) => {
                    return Err(PlacementError::Tensor {
                        name: q_a,
                        detail: format!(
                            "is {} the file, and so is {q}: layer {l} needs exactly one",
                            if both { "in" } else { "not in" }
                        ),
                    });
                }
            };
            if kind != said {
                return Err(metadata(
                    split,
                    key,
                    format!("makes layer {l} {said:?}, and its tensors make it {kind:?}"),
                ));
            }
            Ok(kind)
        })
        .collect()
}

/// A per-layer f32 key: an array of `n_layer` values, or one value for every
/// layer (ik's `get_key_or_arr`); each finite and not negative.
fn per_layer_f32(split: &Split, suffix: &str, n_layer: usize) -> Result<Vec<f32>, PlacementError> {
    let values: Vec<Option<f32>> = match split.value(&split.arch_key(suffix)) {
        Some(Value::Array(items)) => {
            if items.len() != n_layer {
                return Err(metadata(
                    split,
                    suffix,
                    format!("has {} values for {n_layer} layers", items.len()),
                ));
            }
            items.iter().map(Value::as_f32).collect()
        }
        Some(v) => vec![v.as_f32(); n_layer],
        None => return Err(metadata(split, suffix, "is absent")),
    };
    values
        .into_iter()
        .enumerate()
        .map(|(l, v)| match v {
            Some(x) if x.is_finite() && x >= 0.0 => Ok(x),
            _ => Err(metadata(
                split,
                suffix,
                format!("has {v:?} at layer {l}, not a finite limit"),
            )),
        })
        .collect()
}

/// `<architecture>.<suffix>` as a count above 0.
fn positive(split: &Split, suffix: &str) -> Result<usize, PlacementError> {
    match meta_usize(split, suffix)? {
        0 => Err(metadata(split, suffix, "is 0")),
        v => Ok(v),
    }
}

/// `<architecture>.<suffix>` as a count, `None` when absent.
fn optional_usize(split: &Split, suffix: &str) -> Result<Option<usize>, PlacementError> {
    match split.value(&split.arch_key(suffix)) {
        None => Ok(None),
        Some(_) => meta_usize(split, suffix).map(Some),
    }
}

/// `<architecture>.<suffix>` as a float, `None` when absent.
fn optional_f32(split: &Split, suffix: &str) -> Result<Option<f32>, PlacementError> {
    match split.value(&split.arch_key(suffix)) {
        None => Ok(None),
        Some(_) => meta_f32(split, suffix).map(Some),
    }
}

/// `<architecture>.<suffix>` as a bool, `None` when absent.
fn optional_bool(split: &Split, suffix: &str) -> Result<Option<bool>, PlacementError> {
    match split.value(&split.arch_key(suffix)) {
        None => Ok(None),
        Some(_) => meta_bool(split, suffix).map(Some),
    }
}

/// `read`, else `default`, recorded in `defaults` with `why` (ik's line).
fn or_default<T: std::fmt::Display>(
    suffix: &str,
    read: Option<T>,
    default: T,
    why: &str,
    defaults: &mut Vec<String>,
) -> T {
    read.unwrap_or_else(|| {
        defaults.push(format!("{suffix} = {default} ({why})"));
        default
    })
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::roles::{DENSE, HC, KDA, LATENT, MOE, NEXTN, SHARED, Stem};
    use super::{Hparams, Kind};
    use crate::arch::synthetic::{V, header_shaped};

    /// Five layers: 0 KDA and dense, 1 and 3 latent, 2 KDA, then the
    /// next-token layer 4 (latent).
    const KINDS: [Kind; 5] = [
        Kind::Kda,
        Kind::Latent,
        Kind::Kda,
        Kind::Latent,
        Kind::Latent,
    ];

    pub(in crate::arch::glm5next) fn keys() -> Vec<(&'static str, V)> {
        vec![
            ("block_count", V::U32(5)),
            ("nextn_predict_layers", V::U32(1)),
            ("embedding_length", V::U32(64)),
            ("context_length", V::U32(1024)),
            ("feed_forward_length", V::U32(128)),
            ("leading_dense_block_count", V::U32(1)),
            ("attention.head_count", V::U32(4)),
            ("attention.head_count_kv", V::I32s(vec![0, 1, 0, 1, 1])),
            ("attention.layer_norm_rms_epsilon", V::F32(1e-5)),
            ("attention.layer_norm_epsilon", V::F32(1e-6)),
            ("attention.q_lora_rank", V::U32(16)),
            ("attention.kv_lora_rank", V::U32(8)),
            ("attention.key_length", V::U32(8)),
            ("attention.value_length", V::U32(8)),
            ("attention.key_length_mla", V::U32(4)),
            ("attention.value_length_mla", V::U32(4)),
            ("rope.dimension_count", V::U32(0)),
            ("ssm.conv_kernel", V::U32(4)),
            ("kda.head_dim", V::U32(8)),
            ("kda.gate_lower_bound", V::F32(-5.0)),
            ("attention.indexer.head_count", V::U32(2)),
            ("attention.indexer.key_length", V::U32(8)),
            ("attention.indexer.top_k", V::U32(16)),
            ("attention.indexer.kpool", V::U32(4)),
            ("hyper_connection.count", V::U32(4)),
            ("hyper_connection.sinkhorn_iterations", V::U32(20)),
            ("hyper_connection.epsilon", V::F32(1e-6)),
            ("expert_count", V::U32(8)),
            ("expert_used_count", V::U32(2)),
            ("expert_group_count", V::U32(1)),
            ("expert_group_used_count", V::U32(1)),
            ("expert_gating_func", V::U32(2)),
            ("expert_feed_forward_length", V::U32(16)),
            ("expert_shared_feed_forward_length", V::U32(16)),
            ("expert_shared_count", V::U32(1)),
            ("expert_weights_scale", V::F32(2.5)),
            ("expert_weights_norm", V::Bool(true)),
            ("swiglu_clamp_exp", V::F32s(vec![10.0; 5])),
            ("swiglu_clamp_shexp", V::F32s(vec![10.0; 5])),
        ]
    }

    pub(in crate::arch::glm5next) fn tensors() -> Vec<String> {
        let mut out: Vec<String> = ["token_embd.weight", "output_norm.weight", "output.weight"]
            .map(str::to_string)
            .to_vec();
        for (l, kind) in KINDS.iter().enumerate() {
            let mut stems: Vec<&Stem> = match kind {
                Kind::Kda => KDA.iter().collect(),
                Kind::Latent => LATENT.iter().collect(),
            };
            stems.extend(if l == 0 { DENSE } else { MOE });
            if l > 0 {
                stems.extend(SHARED);
            }
            stems.extend(if l < 4 { HC } else { NEXTN });
            out.extend(stems.iter().map(|s| format!("blk.{l}.{}", s.name)));
        }
        out
    }

    /// A KDA tensor's dims in the header of [`keys`] (64 wide, 4 heads of
    /// 8, conv 4: 32 inner values), ggml order; `None` for another stem.
    fn kda_dims(stem: &str) -> Option<Vec<u64>> {
        Some(match stem {
            "attn_norm.weight" => vec![64],
            "attn_q.weight" | "attn_k.weight" | "attn_v.weight" => vec![64, 32],
            "ssm_conv1d_q.weight" | "ssm_conv1d_k.weight" | "ssm_conv1d_v.weight" => {
                vec![4, 1, 32]
            }
            "ssm_f_a.weight" | "ssm_g_a.weight" => vec![64, 8],
            "ssm_f_b.weight" | "ssm_g_b.weight" => vec![8, 32],
            "ssm_beta.weight" => vec![64, 4],
            "ssm_a" => vec![4],
            "ssm_dt.bias" => vec![32],
            "ssm_norm.weight" => vec![8],
            "attn_output.weight" => vec![32, 64],
            _ => return None,
        })
    }

    /// `tensors` with their dims: a KDA layer's (layers 0 and 2) as ik
    /// creates them, every other one value.
    pub(in crate::arch::glm5next) fn shaped(tensors: &[String]) -> Vec<(String, Vec<u64>)> {
        tensors
            .iter()
            .map(|n| {
                let dims = ["blk.0.", "blk.2."]
                    .iter()
                    .find_map(|p| n.strip_prefix(p))
                    .and_then(kda_dims)
                    .unwrap_or_else(|| vec![1]);
                (n.clone(), dims)
            })
            .collect()
    }

    /// The read of a header of `kv` and `tensors`, or its error's text.
    fn read(tag: &str, kv: &[(&str, V)], tensors: &[String]) -> Result<Hparams, String> {
        read_shaped(tag, kv, &shaped(tensors))
    }

    /// [`read`] with each tensor's dims.
    fn read_shaped(
        tag: &str,
        kv: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> Result<Hparams, String> {
        let path = header_shaped(tag, "glm5next", kv, &[], tensors);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let hp = Hparams::read(&split).map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        hp
    }

    fn refused(tag: &str, kv: &[(&str, V)], tensors: &[String], want: &str) {
        let err = read(tag, kv, tensors).expect_err("the header is refused");
        assert!(err.contains(want), "want {want:?} in: {err}");
    }

    #[test]
    fn a_small_header_reads() {
        let hp = read("glm5next-ok", &keys(), &tensors()).expect("the header reads");
        assert_eq!(hp.kinds, KINDS);
        assert_eq!((hp.n_layer, hp.n_trunk, hp.dense_lead), (5, 4, 1));
        assert_eq!(hp.defaults, Vec::<String>::new());
    }

    /// A file without the share key reads as before: no value, and no
    /// default taken.
    #[test]
    fn an_absent_index_share_mtp_is_none_and_takes_no_default() {
        let hp = read("glm5next-noshare", &keys(), &tensors()).expect("the header reads");
        assert_eq!(hp.index_share_mtp, None);
        assert_eq!(hp.defaults, Vec::<String>::new());
    }

    /// The share key is read as the bool it is, whichever value: a proposal
    /// of one id runs the same walk under both, and the key is not listed
    /// among those the reader does not read.
    #[test]
    fn an_index_share_mtp_of_either_value_is_read() {
        for want in [true, false] {
            let mut kv = keys();
            kv.push(("attention.indexer.index_share_mtp", V::Bool(want)));
            let path = header_shaped("glm5next-share", "glm5next", &kv, &[], &shaped(&tensors()));
            let split = gguf::Split::open(&path).expect("the synthetic header opens");
            let hp = Hparams::read(&split).map_err(|e| e.to_string());
            let unread = super::unread_keys(&split);
            let _ = std::fs::remove_file(&path);
            let hp = hp.expect("the header reads");
            assert_eq!(hp.index_share_mtp, Some(want));
            assert_eq!(hp.defaults, Vec::<String>::new());
            assert!(
                !unread.iter().any(|k| k.ends_with("index_share_mtp")),
                "{unread:?}"
            );
        }
    }

    /// A share key of another type is refused by its name.
    #[test]
    fn a_non_bool_index_share_mtp_is_refused() {
        let mut kv = keys();
        kv.push(("attention.indexer.index_share_mtp", V::U32(1)));
        refused(
            "glm5next-share-u32",
            &kv,
            &tensors(),
            "glm5next.attention.indexer.index_share_mtp: is absent or not a bool",
        );
    }

    /// The recorded lines of the next-token count's two defaults, as ik's
    /// lines cite them.
    #[test]
    fn the_next_token_count_records_its_defaults_with_ik_lines() {
        let nextn = |tag: &str, kv: &[(&str, V)], t: &[String]| {
            let path = header_shaped(tag, "glm5next", kv, &[], &shaped(t));
            let split = gguf::Split::open(&path).expect("the synthetic header opens");
            let mut defaults = Vec::new();
            let n = super::nextn(&split, 5, &mut defaults);
            let _ = std::fs::remove_file(&path);
            (n.expect("a count below block_count"), defaults)
        };
        let (n, defaults) = nextn("glm5next-nx-ok", &keys(), &tensors());
        assert_eq!((n, defaults.len()), (1, 0));
        let absent: Vec<_> = keys()
            .into_iter()
            .filter(|(k, _)| *k != "nextn_predict_layers")
            .collect();
        let (n, defaults) = nextn("glm5next-nx-abs", &absent, &tensors());
        assert_eq!(
            (n, defaults),
            (0, ["nextn_predict_layers = 0 (:2301)".to_string()].to_vec())
        );
        let bare: Vec<_> = tensors()
            .into_iter()
            .filter(|n| !n.contains("nextn.eh_proj"))
            .collect();
        let (n, defaults) = nextn("glm5next-nx-probe", &keys(), &bare);
        assert_eq!(n, 0);
        assert_eq!(
            defaults,
            [
                "nextn_predict_layers = 0 (blk.4.nextn.eh_proj.weight is not in the file, :2302-2308)"
            ]
        );
    }

    #[test]
    fn a_missing_key_is_named() {
        for key in ["attention.q_lora_rank", "attention.layer_norm_epsilon"] {
            let kv: Vec<_> = keys().into_iter().filter(|(k, _)| *k != key).collect();
            refused("glm5next-key", &kv, &tensors(), &format!("glm5next.{key}"));
        }
    }

    #[test]
    fn a_short_layer_array_is_refused() {
        let kv: Vec<_> = keys()
            .into_iter()
            .map(|(k, v)| match k {
                "attention.head_count_kv" => (k, V::I32s(vec![0, 1, 0, 1])),
                _ => (k, v),
            })
            .collect();
        refused("glm5next-arr", &kv, &tensors(), "has 4 values for 5 layers");
    }

    #[test]
    fn a_latent_layer_without_kv_a_is_refused() {
        let t: Vec<String> = tensors()
            .into_iter()
            .filter(|n| n != "blk.1.attn_kv_a_mqa.weight")
            .collect();
        refused(
            "glm5next-kva",
            &keys(),
            &t,
            "tensor blk.1.attn_kv_a_mqa.weight: is not in",
        );
    }

    #[test]
    fn an_hc_tensor_on_the_next_token_layer_is_refused() {
        let mut t = tensors();
        t.push("blk.4.hc_attn_fn.weight".to_string());
        refused(
            "glm5next-hc",
            &keys(),
            &t,
            "tensor blk.4.hc_attn_fn.weight: is in",
        );
    }

    #[test]
    fn a_kda_layer_with_an_indexer_is_refused() {
        let mut t = tensors();
        t.push("blk.2.indexer.attn_k.weight".to_string());
        refused(
            "glm5next-idx",
            &keys(),
            &t,
            "tensor blk.2.indexer.attn_k.weight: is in",
        );
    }

    #[test]
    fn top_k_is_a_whole_number_of_pools() {
        let kv: Vec<_> = keys()
            .into_iter()
            .map(|(k, v)| match k {
                "attention.indexer.top_k" => (k, V::U32(18)),
                _ => (k, v),
            })
            .collect();
        refused(
            "glm5next-pool",
            &kv,
            &tensors(),
            "not a whole number of pools of 4",
        );
    }

    /// A bound of −∞ would make every decay 0 and erase the state each
    /// token: refused by name, as a bound at or above 0 is.
    #[test]
    fn a_decay_bound_not_finite_is_refused() {
        for lb in [f32::NEG_INFINITY, f32::NAN, 0.0] {
            let kv: Vec<_> = keys()
                .into_iter()
                .map(|(k, v)| match k {
                    "kda.gate_lower_bound" => (k, V::F32(lb)),
                    _ => (k, v),
                })
                .collect();
            refused(
                "glm5next-lb",
                &kv,
                &tensors(),
                "glm5next.kda.gate_lower_bound: is",
            );
        }
    }

    /// A router that does not renormalize its picks is one the body does
    /// not run: refused by name.
    #[test]
    fn unnormalized_expert_weights_are_refused() {
        let kv: Vec<_> = keys()
            .into_iter()
            .map(|(k, v)| match k {
                "expert_weights_norm" => (k, V::Bool(false)),
                _ => (k, v),
            })
            .collect();
        refused(
            "glm5next-norm",
            &kv,
            &tensors(),
            "glm5next.expert_weights_norm: is false",
        );
    }

    /// A KDA tensor of other dims than ik creates it with is refused by
    /// name, a short one and a long one alike, and only in a trunk layer.
    #[test]
    fn a_kda_tensor_of_other_dims_is_refused() {
        let with = |name: &str, dims: Vec<u64>| -> Vec<(String, Vec<u64>)> {
            shaped(&tensors())
                .into_iter()
                .map(|(n, d)| if n == name { (n, dims.clone()) } else { (n, d) })
                .collect()
        };
        for (name, dims) in [
            ("blk.2.ssm_f_a.weight", vec![64, 7]),
            ("blk.0.ssm_norm.weight", vec![32]),
            ("blk.0.ssm_conv1d_k.weight", vec![4, 32]),
        ] {
            let err = read_shaped("glm5next-dims", &keys(), &with(name, dims))
                .expect_err("the header is refused");
            assert!(
                err.contains(&format!("tensor {name}: has other dims")),
                "want {name} in: {err}"
            );
        }
    }
}
