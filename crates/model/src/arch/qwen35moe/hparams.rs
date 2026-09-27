//! Every `qwen35moe.*` key the description reads, and each layer's kind from
//! the tensors it carries. Required keys are llama.cpp's required set; a key
//! it reads as optional keeps its reading of the absence, and every default
//! taken is recorded ([`Hparams::defaults`]) with the line that sets it.

use gguf::Split;

use crate::arch::{meta_arr, meta_f32, meta_usize, metadata, n_vocab};
use crate::placement::PlacementError;

/// llama.cpp's `LLM_EXPERT_GATING_FUNC_TYPE_SOFTMAX`.
const SOFTMAX: u64 = 1;

/// What a trunk layer mixes with, from the tensors it carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `attn_qkv`: a gated delta-rule layer.
    DeltaRule,
    /// `attn_q`: a gated GQA layer.
    Attention,
}

/// The hyperparameters of one qwen35moe file.
#[derive(Clone, Debug, PartialEq)]
pub struct Hparams {
    /// `block_count`.
    pub n_layer: usize,
    /// `embedding_length`.
    pub n_embd: usize,
    /// `attention.head_count`.
    pub n_head: usize,
    /// `attention.head_count_kv`.
    pub n_head_kv: usize,
    /// `attention.key_length`, which the value length equals.
    pub head_dim: usize,
    /// `rope.dimension_count`: the rotated values of a head.
    pub rope_dims: usize,
    /// `rope.dimension_sections`: pairs per position axis.
    pub rope_sections: [u32; 4],
    /// `rope.freq_base`.
    pub rope_base: f32,
    /// `attention.layer_norm_rms_epsilon`.
    pub rms_eps: f32,
    /// `vocab_size`, else the token list's length.
    pub n_vocab: usize,
    /// `context_length`.
    pub n_ctx_train: usize,
    /// `expert_count`.
    pub n_expert: usize,
    /// `expert_used_count`.
    pub n_used: usize,
    /// `expert_feed_forward_length`.
    pub expert_ff: usize,
    /// `expert_shared_feed_forward_length`; `None` when the file has no shared expert.
    pub shared_ff: Option<usize>,
    /// `ssm.conv_kernel`.
    pub conv: usize,
    /// `ssm.state_size`: values per key and value head.
    pub state: usize,
    /// `ssm.time_step_rank`: value heads.
    pub v_heads: usize,
    /// `ssm.group_count`: key heads.
    pub k_heads: usize,
    /// `full_attention_interval`.
    pub interval: usize,
    /// Every layer's kind, in order.
    pub kinds: Vec<Kind>,
    /// The keys this read took a default for: the key, the value, the line.
    pub defaults: Vec<String>,
}

impl Hparams {
    /// Every value from `split`'s headers; a missing required key, a value the
    /// description cannot hold, or a layer whose tensors disagree with the
    /// interval is an error naming the key or the tensor.
    pub fn read(split: &Split) -> Result<Hparams, PlacementError> {
        let mut defaults = Vec::new();
        let n_layer = meta_usize(split, "block_count")?;
        if let Some(n) = optional_usize(split, "nextn_predict_layers")?
            && n > 0
        {
            return Err(metadata(
                split,
                "nextn_predict_layers",
                format!("is {n}; this reader reads no next-token layer"),
            ));
        }
        let n_embd = meta_usize(split, "embedding_length")?;
        let n_head = meta_usize(split, "attention.head_count")?;
        let n_head_kv = match optional_usize(split, "attention.head_count_kv")? {
            Some(v) => v,
            None => {
                defaults.push(format!(
                    "attention.head_count_kv = {n_head} (the query heads, llama-hparams.cpp)"
                ));
                n_head
            }
        };
        if n_head == 0 || n_head_kv == 0 || !n_head.is_multiple_of(n_head_kv) {
            return Err(metadata(
                split,
                "attention.head_count_kv",
                format!("is {n_head_kv} for {n_head} query heads"),
            ));
        }
        let head_dim = match optional_usize(split, "attention.key_length")? {
            Some(v) => v,
            None => {
                let v = n_embd / n_head;
                defaults.push(format!(
                    "attention.key_length = {v} (embedding_length / head_count, llama-hparams.cpp)"
                ));
                v
            }
        };
        if let Some(v) = optional_usize(split, "attention.value_length")?
            && v != head_dim
        {
            return Err(metadata(
                split,
                "attention.value_length",
                format!("is {v}, the key length {head_dim}; one width serves q, k and v"),
            ));
        }
        let rope_dims = match optional_usize(split, "rope.dimension_count")? {
            Some(v) => v,
            None => {
                defaults.push(format!(
                    "rope.dimension_count = {head_dim} (the head, llama-hparams.cpp)"
                ));
                head_dim
            }
        };
        if rope_dims > head_dim || rope_dims % 2 != 0 {
            return Err(metadata(
                split,
                "rope.dimension_count",
                format!("is {rope_dims}, for a head of {head_dim}"),
            ));
        }
        let rope_sections = sections(split)?;
        let conv = meta_usize(split, "ssm.conv_kernel")?;
        let state = meta_usize(split, "ssm.state_size")?;
        let v_heads = meta_usize(split, "ssm.time_step_rank")?;
        let k_heads = meta_usize(split, "ssm.group_count")?;
        let inner = meta_usize(split, "ssm.inner_size")?;
        if k_heads == 0 || v_heads == 0 || !v_heads.is_multiple_of(k_heads) {
            return Err(metadata(
                split,
                "ssm.group_count",
                format!("is {k_heads} for {v_heads} value heads"),
            ));
        }
        if inner != v_heads * state {
            return Err(metadata(
                split,
                "ssm.inner_size",
                format!("is {inner}, not ssm.time_step_rank {v_heads} x ssm.state_size {state}"),
            ));
        }
        let interval = match optional_usize(split, "full_attention_interval")? {
            Some(v) => v,
            None => {
                defaults.push("full_attention_interval = 4 (qwen35moe.cpp:21-25)".to_string());
                4
            }
        };
        if interval == 0 {
            return Err(metadata(split, "full_attention_interval", "is 0"));
        }
        let kinds = kinds(split, n_layer, interval)?;
        let n_expert = meta_usize(split, "expert_count")?;
        let n_used = meta_usize(split, "expert_used_count")?;
        if n_expert == 0 || n_used == 0 || n_used > n_expert {
            return Err(metadata(
                split,
                "expert_used_count",
                format!("is {n_used} of expert_count {n_expert}"),
            ));
        }
        let expert_ff = meta_usize(split, "expert_feed_forward_length")?;
        if expert_ff == 0 {
            return Err(metadata(split, "expert_feed_forward_length", "is 0"));
        }
        refuse_unless(split, "expert_gating_func", |v| v.as_u64() == Some(SOFTMAX))?;
        refuse_unless(split, "expert_weights_norm", |v| v.as_bool() == Some(true))?;
        refuse_unless(split, "expert_weights_scale", |v| v.as_f32() == Some(1.0))?;
        for (key, value) in [
            ("expert_gating_func", "softmax"),
            ("expert_weights_norm", "true"),
            ("expert_weights_scale", "1"),
        ] {
            if split.value(&split.arch_key(key)).is_none() {
                defaults.push(format!("{key} = {value} (qwen35moe.cpp:499-508)"));
            }
        }
        Ok(Hparams {
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            rope_dims,
            rope_sections,
            rope_base: meta_f32(split, "rope.freq_base")?,
            rms_eps: meta_f32(split, "attention.layer_norm_rms_epsilon")?,
            n_vocab: n_vocab(split)?,
            n_ctx_train: meta_usize(split, "context_length")?,
            n_expert,
            n_used,
            expert_ff,
            shared_ff: optional_usize(split, "expert_shared_feed_forward_length")?,
            conv,
            state,
            v_heads,
            k_heads,
            interval,
            kinds,
            defaults,
        })
    }
}

/// `rope.dimension_sections`, required (qwen35moe.cpp:8): four counts.
fn sections(split: &Split) -> Result<[u32; 4], PlacementError> {
    let key = "rope.dimension_sections";
    let items = meta_arr(split, key)?;
    if items.len() != 4 {
        return Err(metadata(
            split,
            key,
            format!("has {} values, not 4", items.len()),
        ));
    }
    let mut out = [0u32; 4];
    for (i, v) in items.iter().enumerate() {
        out[i] = v
            .as_unsigned()
            .and_then(|x| u32::try_from(x).ok())
            .ok_or_else(|| metadata(split, key, format!("has no count at {i}")))?;
    }
    Ok(out)
}

/// Every layer's kind from its tensors, cross-checked with the key that also
/// says it: `attention.recurrent_layers` when the file carries it, else every
/// `interval`-th layer attends (qwen35moe.cpp:18-26).
fn kinds(split: &Split, n_layer: usize, interval: usize) -> Result<Vec<Kind>, PlacementError> {
    let recurrent = match split.arch_get_arr("attention.recurrent_layers") {
        None => None,
        Some(items) => {
            if items.len() != n_layer {
                return Err(metadata(
                    split,
                    "attention.recurrent_layers",
                    format!("has {} values for {n_layer} layers", items.len()),
                ));
            }
            Some(
                items
                    .iter()
                    .enumerate()
                    .map(|(l, v)| {
                        v.as_bool()
                            .or_else(|| v.as_unsigned().map(|x| x != 0))
                            .ok_or_else(|| {
                                metadata(
                                    split,
                                    "attention.recurrent_layers",
                                    format!("has no flag for layer {l}"),
                                )
                            })
                    })
                    .collect::<Result<Vec<bool>, _>>()?,
            )
        }
    };
    (0..n_layer)
        .map(|l| {
            let qkv = format!("blk.{l}.attn_qkv.weight");
            let q = format!("blk.{l}.attn_q.weight");
            let kind = match (split.find(&qkv).is_some(), split.find(&q).is_some()) {
                (true, false) => Kind::DeltaRule,
                (false, true) => Kind::Attention,
                (both, _) => {
                    return Err(PlacementError::Tensor {
                        name: q,
                        detail: format!(
                            "is {} the file, and so is {qkv}: layer {l} needs exactly one",
                            if both { "in" } else { "not in" }
                        ),
                    });
                }
            };
            let (said, key) = match &recurrent {
                Some(r) => (r[l], "attention.recurrent_layers"),
                None => ((l + 1) % interval != 0, "full_attention_interval"),
            };
            if said != (kind == Kind::DeltaRule) {
                return Err(metadata(
                    split,
                    key,
                    format!(
                        "makes layer {l} {}, and its tensors make it {kind:?}",
                        if said { "recurrent" } else { "attention" }
                    ),
                ));
            }
            Ok(kind)
        })
        .collect()
}

/// `<architecture>.<suffix>` as a count, `None` when absent.
fn optional_usize(split: &Split, suffix: &str) -> Result<Option<usize>, PlacementError> {
    match split.value(&split.arch_key(suffix)) {
        None => Ok(None),
        Some(_) => meta_usize(split, suffix).map(Some),
    }
}

/// Err naming `suffix` when the file carries it with a value `ok` refuses:
/// llama.cpp's builder passes the router's softmax, renormalization and
/// scale 1 as constants (qwen35moe.cpp:499-508).
fn refuse_unless(
    split: &Split,
    suffix: &str,
    ok: impl Fn(&gguf::Value) -> bool,
) -> Result<(), PlacementError> {
    match split.value(&split.arch_key(suffix)) {
        Some(v) if !ok(v) => Err(metadata(
            split,
            suffix,
            format!("is {v:?}; the architecture's router takes softmax, renormalized, scale 1"),
        )),
        _ => Ok(()),
    }
}
