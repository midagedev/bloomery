//! Kimi K3's and K3.1's checkpoint folder (`kimi_k3`, `kimi_k31`; the original, compressed-tensors
//! `mxfp4-pack-quantized` on the routed experts and BF16 elsewhere) in the kimi-k3 names, layouts
//! and keys the program reads, as llama.cpp's converter (`KimiK3Model`) writes them into a GGUF.
//!
//! A tensor maps to one kimi-k3 tensor ([`model::arch::kimi_k3::names`]) with these moves:
//! - the cross-layer residual's `<x>_res_norm` and `<x>_res_proj` pair is one F32 score vector,
//!   their elementwise product ([`Step::ResProduct`]; the product commutes, so what the gate pins
//!   is each pair going to its own score tensor);
//! - `A_log` is `-exp` of its first `num_attention_heads` values ([`Step::NegExp`]), `dt_bias`
//!   the `ssm_dt` bias, and the KDA conv taps `[inner, 1, kernel]` are as stored;
//! - `g_proj` is the KDA layers' `ssm_g` and the latent layers' `attn_gate`, by the layer's kind;
//! - `kv_b_proj` splits per head into the key half, transposed, and the value half
//!   ([`Step::KeyHalf`], [`Step::ValueHalf`]);
//! - a layer's routed experts, one `w1` (gate), `w3` (up) or `w2` (down) tensor an expert, stack by
//!   expert id, in the OCP MXFP4 blocks the stored packs hold (the pack's even value is the low
//!   nibble, its scale `2^(e-127)`: [`hf::modelopt`]'s MXFP4 writer, whose order the gate pins);
//! - K3.1's indexer tensors are the names Deepseek-V4.1's and Mistral Large 4's descriptions give
//!   theirs, on the latent layers `dsa_config` lists: `attention.indexer.*` keys carry its
//!   geometry, and the K3 description's reader names them where it needs them.
//!
//! The vision tower and its projector are skipped with the reason. A layer holds the tensors of its
//! kind and no other, a K3 folder no indexer and a K3.1 folder one on each latent layer. The
//! header's keys are the config's ([`Checkpoint::parse`]); a field the map does not take is
//! refused by name.
//!
//! The header is written under the folder's own architecture word, `kimi-k3` or `kimi-k31`,
//! or under another of the two ([`FolderMap::with_arch`]): a K3.1 folder as
//! `kimi-k3` holds no indexer tensor and no indexer key, which a split of it written with
//! `--drop 'blk.*.indexer.*'` is, and a K3 folder as `kimi-k31` has no indexer to name and is
//! refused. The table the config derives of what the folder stores ([`FolderMap::stored`]) is
//! what `prepare --follow` plans from before the folder's shards are all there.

use std::collections::{BTreeMap, BTreeSet};

use gguf::arch::kimi_k3::ARCH as ARCHITECTURE;
use gguf::{GgmlType, Value};
use hf::modelopt::{Normal, Plain, Use};
use hf::quant_config::ModuleClass;
use hf::schema::{Fields, Rule::*, Type};
use model::arch::deepseek41::names as indexer;
use model::arch::kimi_k3::names;
use model::arch::mistrallarge4::names::INDEXER_K_NORM_B;
use safetensors::{Dtype, Folder};
use serde_json::Value as Json;

use super::section::Section;
use crate::folder::{Error, FolderMap, Mapped, Mapping, Origin, Stored, VocabSpec, refuse};
use crate::part;

#[cfg(test)]
mod tests;

/// `expert_gating_func`'s sigmoid router, llama.cpp's `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`.
const SIGMOID: u32 = 2;

/// A pack's scale covers this many values.
const MXFP4_BLOCK: usize = 32;

/// The checkpoint stores `A_log` padded to a multiple of this many values: 96 heads as 128.
const A_LOG_PAD: usize = 128;

/// The two checkpoints the map reads: K3 and K3.1, which adds the indexer to its latent layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    K3,
    K31,
}

/// K3.1's architecture word; the K3 description declares none for it, so R2 declares it.
const ARCHITECTURE_K31: &str = "kimi-k31";

impl Variant {
    /// The architecture word of the header a folder of this variant is written under.
    fn word(self) -> &'static str {
        match self {
            Variant::K3 => ARCHITECTURE,
            Variant::K31 => ARCHITECTURE_K31,
        }
    }

    /// The `architectures` entry of this variant's config.
    fn architecture(self) -> &'static str {
        match self {
            Variant::K3 => "KimiK3ForConditionalGeneration",
            Variant::K31 => "KimiK31ForConditionalGeneration",
        }
    }
}

const GENERATION: &str = "a transformers generation default; the map reads no sampling default";
const BOOKKEEPING: &str = "transformers' PretrainedConfig bookkeeping; no result depends on it";

/// `config.json`'s top-level keys.
const ROOT: &Fields = &[
    (
        "architectures",
        Need(r#"[["KimiK3ForConditionalGeneration"], ["KimiK31ForConditionalGeneration"]]"#),
    ),
    ("model_type", Need(r#"["kimi_k3", "kimi_k31"]"#)),
    (
        "auto_map",
        Ignore(Type::Object, "the remote modeling code HF would run"),
    ),
    ("text_config", Read),
    (
        "vision_config",
        Ignore(
            Type::Object,
            "the vision tower's geometry; its tensors are skipped",
        ),
    ),
    ("bos_token_id", Read),
    ("eos_token_id", Read),
    ("pad_token_id", Read),
    (
        "media_placeholder_token_id",
        Ignore(
            Type::Number,
            "the media placeholder; the vision tower is skipped",
        ),
    ),
    (
        "image_placeholder",
        Ignore(
            Type::String,
            "the image placeholder's text; the vision tower is skipped",
        ),
    ),
    (
        "ignore_index",
        Ignore(Type::Number, "a training loss's ignored label"),
    ),
    ("tie_word_embeddings", Only("[false]")),
    (
        "dtype",
        Ignore(
            Type::String,
            "the dtype HF loads in; each tensor keeps its stored dtype and format",
        ),
    ),
];

/// `text_config`'s keys the map takes.
const TEXT: &Fields = &[
    ("model_type", Need(r#"["kimi_linear"]"#)),
    ("architectures", Only(r#"[["KimiLinearForCausalLM"]]"#)),
    (
        "auto_map",
        Ignore(Type::Object, "the remote modeling code HF would run"),
    ),
    ("hidden_size", Read),
    ("intermediate_size", Read),
    ("num_hidden_layers", Read),
    ("num_attention_heads", Read),
    ("num_key_value_heads", Read),
    ("max_position_embeddings", Read),
    ("vocab_size", Read),
    ("bos_token_id", Read),
    ("eos_token_id", Read),
    ("pad_token_id", Read),
    ("sep_token_id", Read),
    ("rms_norm_eps", Read),
    ("hidden_act", Need(r#"["situ"]"#)),
    ("activation_situ_beta", Read),
    ("activation_situ_linear_beta", Read),
    ("attn_res_block_size", Read),
    ("q_lora_rank", Read),
    ("kv_lora_rank", Read),
    ("qk_nope_head_dim", Read),
    ("qk_rope_head_dim", Read),
    ("v_head_dim", Read),
    ("mla_use_nope", Need("[true]")),
    ("mla_use_output_gate", Need("[true]")),
    ("linear_attn_config", Read),
    ("dsa_config", Read),
    ("first_k_dense_replace", Read),
    ("moe_layer_freq", Need("[1]")),
    ("moe_intermediate_size", Read),
    ("num_experts", Read),
    ("num_experts_per_token", Read),
    ("num_shared_experts", Read),
    ("routed_expert_hidden_size", Read),
    ("latent_moe_use_norm", Need("[true]")),
    ("moe_renormalize", Read),
    ("moe_router_activation_func", Need(r#"["sigmoid"]"#)),
    ("routed_scaling_factor", Read),
    ("topk_method", Need(r#"["noaux_tc"]"#)),
    ("use_grouped_topk", Only("[true]")),
    ("num_expert_group", Need("[1]")),
    ("topk_group", Need("[1]")),
    ("num_nextn_predict_layers", Need("[0]")),
    ("tie_word_embeddings", Only("[false]")),
    ("quantization_config", Read),
    (
        "dtype",
        Ignore(
            Type::String,
            "the dtype HF loads in; each tensor keeps its stored dtype and format",
        ),
    ),
    (
        "transformers_version",
        Ignore(Type::String, "the version that wrote the file"),
    ),
    (
        "initializer_range",
        Ignore(Type::Number, "initializes weights the checkpoint holds"),
    ),
    (
        "use_cache",
        Ignore(
            Type::Bool,
            "whether HF keeps its cache; outputs are the same",
        ),
    ),
];

/// The generation and class bookkeeping every transformers config writes, which no tensor or
/// header key of the map depends on.
const HF_DEFAULTS: &Fields = &[
    ("_name_or_path", Ignore(Type::String, BOOKKEEPING)),
    ("add_cross_attention", Ignore(Type::Bool, BOOKKEEPING)),
    ("bad_words_ids", Only("[null]")),
    ("begin_suppress_tokens", Only("[null]")),
    ("chunk_size_feed_forward", Ignore(Type::Number, BOOKKEEPING)),
    ("cross_attention_hidden_size", Only("[null]")),
    ("decoder_start_token_id", Only("[null]")),
    ("diversity_penalty", Ignore(Type::Number, GENERATION)),
    ("do_sample", Ignore(Type::Bool, GENERATION)),
    ("early_stopping", Ignore(Type::Bool, GENERATION)),
    (
        "encoder_no_repeat_ngram_size",
        Ignore(Type::Number, GENERATION),
    ),
    ("exponential_decay_length_penalty", Only("[null]")),
    ("finetuning_task", Only("[null]")),
    ("forced_bos_token_id", Only("[null]")),
    ("forced_eos_token_id", Only("[null]")),
    ("id2label", Ignore(Type::Object, BOOKKEEPING)),
    ("is_decoder", Ignore(Type::Bool, BOOKKEEPING)),
    ("is_encoder_decoder", Ignore(Type::Bool, BOOKKEEPING)),
    ("label2id", Ignore(Type::Object, BOOKKEEPING)),
    ("length_penalty", Ignore(Type::Number, GENERATION)),
    ("max_length", Ignore(Type::Number, GENERATION)),
    ("min_length", Ignore(Type::Number, GENERATION)),
    ("no_repeat_ngram_size", Ignore(Type::Number, GENERATION)),
    ("num_beam_groups", Ignore(Type::Number, GENERATION)),
    ("num_beams", Ignore(Type::Number, GENERATION)),
    ("num_return_sequences", Ignore(Type::Number, GENERATION)),
    ("output_attentions", Ignore(Type::Bool, BOOKKEEPING)),
    ("output_hidden_states", Ignore(Type::Bool, BOOKKEEPING)),
    ("output_scores", Ignore(Type::Bool, BOOKKEEPING)),
    ("prefix", Only("[null]")),
    ("problem_type", Only("[null]")),
    ("pruned_heads", Ignore(Type::Object, BOOKKEEPING)),
    ("remove_invalid_values", Ignore(Type::Bool, GENERATION)),
    ("repetition_penalty", Ignore(Type::Number, GENERATION)),
    ("return_dict", Ignore(Type::Bool, BOOKKEEPING)),
    ("return_dict_in_generate", Ignore(Type::Bool, BOOKKEEPING)),
    ("suppress_tokens", Only("[null]")),
    ("task_specific_params", Only("[null]")),
    ("temperature", Ignore(Type::Number, GENERATION)),
    ("tf_legacy_loss", Ignore(Type::Bool, BOOKKEEPING)),
    ("tie_encoder_decoder", Ignore(Type::Bool, BOOKKEEPING)),
    ("tokenizer_class", Only("[null]")),
    ("top_k", Ignore(Type::Number, GENERATION)),
    ("top_p", Ignore(Type::Number, GENERATION)),
    ("torchscript", Ignore(Type::Bool, BOOKKEEPING)),
    ("typical_p", Ignore(Type::Number, GENERATION)),
    ("use_bfloat16", Ignore(Type::Bool, BOOKKEEPING)),
];

/// `text_config.linear_attn_config`'s keys: which layers are KDA and which latent (both lists
/// 1-based), and the KDA layer's geometry.
const LINEAR: &Fields = &[
    ("full_attn_layers", Read),
    ("kda_layers", Read),
    ("num_heads", Read),
    ("head_dim", Read),
    ("short_conv_kernel_size", Read),
    ("gate_lower_bound", Read),
    ("use_full_rank_gate", Need("[true]")),
];

/// `text_config.dsa_config`'s keys (K3.1): the layers that run the indexer (1-based) and its
/// geometry. A dsa layer reads every key position's score (`index_topk_freq` 1), with the
/// indexer's key a LayerNorm of the key projection, without a rope.
const DSA: &Fields = &[
    ("dsa_layers", Read),
    ("index_topk", Read),
    ("index_head_dim", Read),
    ("index_num_heads", Read),
    ("index_rope_dim", Need("[0]")),
    ("index_k_norm_type", Need(r#"["layernorm"]"#)),
    ("index_topk_freq", Need("[1]")),
    (
        "index_skip_topk_offset",
        Ignore(
            Type::Number,
            "a layer offset that acts only where index_topk_freq is not 1",
        ),
    ),
];

/// A move of a normalized tensor into the program's layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// `-exp(x)` of the first `heads` values of an F32 vector, computed in f64 and rounded once;
    /// the checkpoint pads `A_log` past the heads.
    NegExp { heads: usize },
    /// `kv_b_proj` `[heads · (nope + v), rank]`: each head's first `nope` rows, transposed to
    /// `[rank, nope]`.
    KeyHalf { heads: usize, nope: usize, v: usize },
    /// Each head's last `v` rows of `kv_b_proj`, `[v, rank]`.
    ValueHalf { heads: usize, nope: usize, v: usize },
    /// The two F32 vectors of a stacked pair, `[k, 1, 2]`, as their elementwise product `[k]`.
    ResProduct,
}

impl Step {
    /// Move `t`, the normalized source of the tensor `name`, into the layout: its bytes and dims
    /// become the tensor's. A source of other values than the move reads is refused with why,
    /// and `t` is then as it was.
    pub fn apply(&self, name: &str, t: &mut Normal) -> Result<(), String> {
        let why = |what: String| format!("{name}: {self:?} {what}");
        let f32 = GgmlType::F32.as_u32();
        let bf16 = GgmlType::BF16.as_u32();
        let (bytes, dims) = match *self {
            Step::NegExp { heads } => {
                if t.type_id != f32 || t.bytes.len() < heads.saturating_mul(4) {
                    return Err(why(format!(
                        "reads {heads} F32 values, not type {} of {} bytes",
                        t.type_id,
                        t.bytes.len()
                    )));
                }
                let mut out = Vec::with_capacity(heads * 4);
                for (i, b) in t.bytes.as_chunks::<4>().0[..heads].iter().enumerate() {
                    let y = super::neg_exp(f32::from_le_bytes(*b))
                        .map_err(|e| why(format!("at {i}: {e}")))?;
                    out.extend_from_slice(&y.to_le_bytes());
                }
                (out, vec![heads as u64])
            }
            Step::KeyHalf { heads, nope, v } | Step::ValueHalf { heads, nope, v } => {
                let [rank, rows] = t.dims[..] else {
                    return Err(why(format!("reads a matrix, not dims {:?}", t.dims)));
                };
                let (rank, rows) = (rank as usize, rows as usize);
                let size = 2;
                if t.type_id != bf16
                    || Some(rows) != (nope + v).checked_mul(heads)
                    || Some(t.bytes.len())
                        != rows.checked_mul(rank).and_then(|n| n.checked_mul(size))
                {
                    return Err(why(format!(
                        "reads {heads} heads of {nope} + {v} BF16 rows, not type {} dims {:?}",
                        t.type_id, t.dims
                    )));
                }
                let row = rank * size;
                let head = |h: usize| &t.bytes[h * (nope + v) * row..][..(nope + v) * row];
                if matches!(self, Step::KeyHalf { .. }) {
                    let mut out = Vec::with_capacity(heads * nope * row);
                    for h in 0..heads {
                        for c in 0..rank {
                            for r in 0..nope {
                                out.extend_from_slice(&head(h)[(r * rank + c) * size..][..size]);
                            }
                        }
                    }
                    (out, vec![nope as u64, rank as u64, heads as u64])
                } else {
                    let out = (0..heads)
                        .flat_map(|h| &head(h)[nope * row..])
                        .copied()
                        .collect();
                    (out, vec![rank as u64, v as u64, heads as u64])
                }
            }
            Step::ResProduct => {
                let [k, 1, 2] = t.dims[..] else {
                    return Err(why(format!("reads a stacked pair, not dims {:?}", t.dims)));
                };
                let k = k as usize;
                if t.type_id != f32 || Some(t.bytes.len()) != k.checked_mul(8) {
                    return Err(why(format!(
                        "reads two F32 vectors of {k}, not type {} of {} bytes",
                        t.type_id,
                        t.bytes.len()
                    )));
                }
                let (norm, proj) = t.bytes.split_at(k * 4);
                let mut out = Vec::with_capacity(k * 4);
                for (i, (a, b)) in norm
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(proj.as_chunks::<4>().0)
                    .enumerate()
                {
                    let y = f32::from_le_bytes(*a) * f32::from_le_bytes(*b);
                    if !y.is_finite() {
                        return Err(why(format!("at {i}: the product is {y}")));
                    }
                    out.extend_from_slice(&y.to_le_bytes());
                }
                (out, vec![k as u64])
            }
        };
        t.bytes = bytes;
        t.dims = dims;
        Ok(())
    }
}

/// The folder tensors no kimi-k3 tensor is made of, and why; `None` for one the map reads.
fn skip_reason(name: &str) -> Option<&'static str> {
    if name.starts_with("vision_tower.") || name.starts_with("mm_projector.") {
        Some("the vision tower and its projector, which the image path reads")
    } else {
        None
    }
}

/// The config values the map reads, checked, and the header keys they make.
#[derive(Debug)]
pub struct Checkpoint {
    /// The checkpoint's own variant, and the architecture word its header is written under
    /// (`prepare --arch`: a K3.1 folder may be written as K3's shape).
    variant: Variant,
    word: Variant,
    layers: usize,
    heads: usize,
    nope: usize,
    v_head: usize,
    kv_lora: usize,
    kda_head: usize,
    conv: usize,
    hidden: u64,
    vocab_size: u64,
    experts: usize,
    expert_ff: u64,
    latent: u64,
    /// The layers before the first routed one.
    dense_lead: usize,
    /// Whether each layer is a latent-attention layer, else a KDA one.
    latent_layer: Vec<bool>,
    /// Whether the latent layers carry the indexer (K3.1).
    indexer: bool,
    /// The query-latent width, the rotary width and the dense feed-forward width.
    q_lora: usize,
    pe: usize,
    ff: usize,
    /// The shared experts, as many as the expert width's multiples the shared projections hold.
    shared: usize,
    /// The indexer's heads and key width; none on a K3 folder.
    index: Option<(usize, usize)>,
    vocab: VocabSpec,
    /// The header's model keys by suffix, under the architecture's name.
    keys: Vec<(String, Value)>,
}

/// `key` of `s` as a set of 1-based layer numbers, each of `1..=layers` once.
fn layer_set(s: &Section<'_, Error>, key: &str, layers: usize) -> Result<BTreeSet<usize>, Error> {
    let list = s.array(key)?;
    let set: BTreeSet<usize> = list
        .iter()
        .map(|v| v.as_u64().and_then(|n| usize::try_from(n).ok()))
        .collect::<Option<_>>()
        .ok_or_else(|| s.bad(key, "is not a list of layer numbers"))?;
    if set.len() != list.len() || set.iter().any(|l| !(1..=layers).contains(l)) {
        return Err(s.bad(
            key,
            &format!("holds a layer twice or outside 1..={layers}: {list:?}"),
        ));
    }
    Ok(set)
}

/// A bool of `s`.
fn flag(s: &Section<'_, Error>, key: &str) -> Result<bool, Error> {
    s.field(key)?
        .as_bool()
        .ok_or_else(|| s.bad(key, "is not a bool"))
}

impl Checkpoint {
    fn read(config: &Json) -> Result<Checkpoint, Error> {
        let root = Section::<Error>::new(config, "config.json")?;
        root.check(&[ROOT])?;
        let variant = match root.field("model_type")?.as_str() {
            Some("kimi_k3") => Variant::K3,
            _ => Variant::K31,
        };
        let architectures = root.array("architectures")?;
        if architectures != [Json::from(variant.architecture())] {
            return Err(root.bad(
                "architectures",
                &format!(
                    "is {architectures:?}, where model_type {} names {}",
                    root.field("model_type")?,
                    variant.architecture()
                ),
            ));
        }
        let text = Section::<Error>::new(root.field("text_config")?, "text_config")?;
        text.check(&[TEXT, HF_DEFAULTS])?;
        let n = |key: &str| text.count(key);
        let f = |key: &str| text.number(key);
        let layers = n("num_hidden_layers")? as usize;
        let heads = n("num_attention_heads")? as usize;
        if n("num_key_value_heads")? as usize != heads {
            return Err(text.bad(
                "num_key_value_heads",
                "is not num_attention_heads: the latent layers' one key head is the program's",
            ));
        }

        let linear = Section::<Error>::new(
            text.field("linear_attn_config")?,
            "text_config.linear_attn_config",
        )?;
        linear.check(&[LINEAR])?;
        if linear.count("num_heads")? as usize != heads {
            return Err(linear.bad("num_heads", "is not text_config.num_attention_heads"));
        }
        let latent_set = layer_set(&linear, "full_attn_layers", layers)?;
        let kda_set = layer_set(&linear, "kda_layers", layers)?;
        if !latent_set.is_disjoint(&kda_set) || latent_set.len() + kda_set.len() != layers {
            return Err(linear.bad(
                "kda_layers",
                "and full_attn_layers do not split 1..=num_hidden_layers between them",
            ));
        }
        let latent_layer: Vec<bool> = (1..=layers).map(|l| latent_set.contains(&l)).collect();
        let lower_bound = linear
            .field("gate_lower_bound")?
            .as_f64()
            .filter(|b| b.is_finite() && *b < 0.0)
            .ok_or_else(|| linear.bad("gate_lower_bound", "must be a finite number below 0"))?;
        let kda_head = linear.count("head_dim")? as usize;
        let conv = linear.count("short_conv_kernel_size")? as usize;

        let dsa = text.value.get("dsa_config");
        let indexer = match (variant, dsa) {
            (Variant::K3, None) => None,
            (Variant::K31, Some(dsa)) => {
                let dsa = Section::<Error>::new(dsa, "text_config.dsa_config")?;
                dsa.check(&[DSA])?;
                if layer_set(&dsa, "dsa_layers", layers)? != latent_set {
                    return Err(dsa.bad(
                        "dsa_layers",
                        "are not linear_attn_config.full_attn_layers: the indexer is on every \
                         latent layer and no other",
                    ));
                }
                Some([
                    (
                        "attention.indexer.head_count",
                        dsa.count("index_num_heads")?,
                    ),
                    ("attention.indexer.key_length", dsa.count("index_head_dim")?),
                    ("attention.indexer.top_k", dsa.count("index_topk")?),
                ])
            }
            (Variant::K3, Some(_)) => {
                return Err(text.bad(
                    "dsa_config",
                    "is a key of K3.1, where model_type is kimi_k3",
                ));
            }
            (Variant::K31, None) => {
                return Err(text.bad("dsa_config", "is missing, where model_type is kimi_k31"));
            }
        };

        let nope = n("qk_nope_head_dim")?;
        let pe = n("qk_rope_head_dim")?;
        let kv_lora = n("kv_lora_rank")?;
        let v_head = n("v_head_dim")?;
        let dense_lead = text
            .field("first_k_dense_replace")?
            .as_u64()
            .and_then(|d| usize::try_from(d).ok())
            .filter(|&d| d < layers)
            .ok_or_else(|| {
                text.bad(
                    "first_k_dense_replace",
                    "must be a layer count below num_hidden_layers",
                )
            })?;
        let kv: Vec<Value> = latent_layer
            .iter()
            .map(|&latent| Value::I32(i32::from(latent)))
            .collect();
        let mut keys: Vec<(String, Value)> = vec![
            ("block_count", Value::U32(n("num_hidden_layers")?)),
            ("context_length", Value::U32(n("max_position_embeddings")?)),
            ("embedding_length", Value::U32(n("hidden_size")?)),
            ("vocab_size", Value::U32(n("vocab_size")?)),
            ("feed_forward_length", Value::U32(n("intermediate_size")?)),
            ("leading_dense_block_count", Value::U32(dense_lead as u32)),
            (
                "attention.head_count",
                Value::U32(n("num_attention_heads")?),
            ),
            ("attention.head_count_kv", Value::Array(kv)),
            ("attention.key_length", Value::U32(kv_lora + pe)),
            ("attention.value_length", Value::U32(kv_lora)),
            ("attention.key_length_mla", Value::U32(nope + pe)),
            ("attention.value_length_mla", Value::U32(v_head)),
            ("attention.q_lora_rank", Value::U32(n("q_lora_rank")?)),
            ("attention.kv_lora_rank", Value::U32(kv_lora)),
            (
                "attention.layer_norm_rms_epsilon",
                Value::F32(f("rms_norm_eps")?),
            ),
            ("rope.dimension_count", Value::U32(pe)),
            ("ssm.conv_kernel", Value::U32(conv as u32)),
            ("kda.head_dim", Value::U32(kda_head as u32)),
            ("kda.gate_lower_bound", Value::F32(lower_bound as f32)),
            ("expert_count", Value::U32(n("num_experts")?)),
            ("expert_used_count", Value::U32(n("num_experts_per_token")?)),
            (
                "expert_feed_forward_length",
                Value::U32(n("moe_intermediate_size")?),
            ),
            ("expert_shared_count", Value::U32(n("num_shared_experts")?)),
            ("expert_gating_func", Value::U32(SIGMOID)),
            (
                "expert_weights_scale",
                Value::F32(f("routed_scaling_factor")?),
            ),
            (
                "expert_weights_norm",
                Value::Bool(flag(&text, "moe_renormalize")?),
            ),
            (
                "expert_latent_length",
                Value::U32(n("routed_expert_hidden_size")?),
            ),
            ("attn_res.block_size", Value::U32(n("attn_res_block_size")?)),
            (
                "activation.situ_beta",
                Value::F32(f("activation_situ_beta")?),
            ),
            (
                "activation.situ_linear_beta",
                Value::F32(f("activation_situ_linear_beta")?),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let has_indexer = indexer.is_some();
        let index = indexer.map(|i| (i[0].1 as usize, i[1].1 as usize));
        keys.extend(
            indexer
                .into_iter()
                .flatten()
                .map(|(k, v)| (k.to_string(), Value::U32(v))),
        );
        Ok(Checkpoint {
            variant,
            word: variant,
            layers,
            heads,
            nope: nope as usize,
            v_head: v_head as usize,
            kv_lora: kv_lora as usize,
            kda_head,
            conv,
            hidden: u64::from(n("hidden_size")?),
            vocab_size: u64::from(n("vocab_size")?),
            experts: n("num_experts")? as usize,
            expert_ff: u64::from(n("moe_intermediate_size")?),
            latent: u64::from(n("routed_expert_hidden_size")?),
            dense_lead,
            latent_layer,
            indexer: has_indexer,
            q_lora: n("q_lora_rank")? as usize,
            pe: pe as usize,
            ff: n("intermediate_size")? as usize,
            shared: n("num_shared_experts")? as usize,
            index,
            vocab: hf::vocab::Config::parse(config)?.with_pre("kimi-k2")?,
            keys,
        })
    }

    /// The kimi-k3 tensors layer `l` holds: the mixer's by the layer's kind, the two mixes' scores
    /// and the feed-forward block.
    fn layer_names(&self, l: usize) -> Vec<String> {
        let mut v = vec![
            names::attn_norm(l),
            names::ffn_norm(l),
            names::attn_res_score(l),
            names::ffn_res_score(l),
            names::attn_output(l),
        ];
        if self.latent_layer[l] {
            v.extend([
                names::attn_q_a(l),
                names::attn_q_a_norm(l),
                names::attn_q_b(l),
                names::attn_kv_a_mqa(l),
                names::attn_kv_a_norm(l),
                names::attn_k_b(l),
                names::attn_v_b(l),
                names::attn_gate(l),
            ]);
            if self.indexer {
                v.extend(indexer_names(l));
            }
        } else {
            v.extend([
                names::attn_q(l),
                names::attn_k(l),
                names::attn_v(l),
                names::ssm_conv1d(l, 'q'),
                names::ssm_conv1d(l, 'k'),
                names::ssm_conv1d(l, 'v'),
                names::ssm_f_a(l),
                names::ssm_f_b(l),
                names::ssm_g(l),
                names::ssm_beta(l),
                names::ssm_a(l),
                names::ssm_dt_bias(l),
                names::ssm_norm(l),
            ]);
        }
        if l < self.dense_lead {
            v.extend([names::ffn_gate(l), names::ffn_up(l), names::ffn_down(l)]);
        } else {
            v.extend([
                names::ffn_gate_inp(l),
                names::exp_probs_b(l),
                names::ffn_gate_exps(l),
                names::ffn_up_exps(l),
                names::ffn_down_exps(l),
                names::ffn_gate_shexp(l),
                names::ffn_up_shexp(l),
                names::ffn_down_shexp(l),
                names::ffn_routed_down(l),
                names::ffn_routed_up(l),
                names::ffn_routed_norm(l),
            ]);
        }
        v
    }

    /// The language model's tensors a folder of this config stores, as the checkpoint names, types
    /// and shapes them, by name: BF16 matrices, F32 for the KDA's convolution taps, `A_log`, bias
    /// and output norm and for the router's bias, and each expert's U8 pack and scale. The vision
    /// tower and its projector are the folder's own and are not here.
    fn stored_tensors(&self) -> Vec<(String, Dtype, Vec<usize>)> {
        use Dtype::{Bf16, F32, U8};
        let (e, heads, kda) = (self.hidden as usize, self.heads, self.kda_head);
        let inner = heads * kda;
        let (eff, latent) = (self.expert_ff as usize, self.latent as usize);
        let mut t: Vec<(String, Dtype, Vec<usize>)> = vec![];
        let mut add = |name: String, dtype: Dtype, shape: &[usize]| {
            t.push((format!("language_model.{name}"), dtype, shape.to_vec()));
        };
        let vocab = self.vocab_size as usize;
        add("model.embed_tokens.weight".into(), Bf16, &[vocab, e]);
        add("lm_head.weight".into(), Bf16, &[vocab, e]);
        add("model.norm.weight".into(), Bf16, &[e]);
        add("model.output_attn_res_norm.weight".into(), Bf16, &[e]);
        add("model.output_attn_res_proj.weight".into(), Bf16, &[1, e]);
        for l in 0..self.layers {
            let mut layer = |rest: &str, dtype: Dtype, shape: &[usize]| {
                add(format!("model.layers.{l}.{rest}"), dtype, shape);
            };
            layer("input_layernorm.weight", Bf16, &[e]);
            layer("post_attention_layernorm.weight", Bf16, &[e]);
            layer("self_attention_res_norm.weight", Bf16, &[e]);
            layer("self_attention_res_proj.weight", Bf16, &[1, e]);
            layer("mlp_res_norm.weight", Bf16, &[e]);
            layer("mlp_res_proj.weight", Bf16, &[1, e]);
            if self.latent_layer[l] {
                let (rank, q_lora, pe) = (self.kv_lora, self.q_lora, self.pe);
                let (nope, v) = (self.nope, self.v_head);
                layer("self_attn.q_a_proj.weight", Bf16, &[q_lora, e]);
                layer("self_attn.q_a_layernorm.weight", Bf16, &[q_lora]);
                layer(
                    "self_attn.q_b_proj.weight",
                    Bf16,
                    &[heads * (nope + pe), q_lora],
                );
                layer("self_attn.kv_a_proj_with_mqa.weight", Bf16, &[rank + pe, e]);
                layer("self_attn.kv_a_layernorm.weight", Bf16, &[rank]);
                layer(
                    "self_attn.kv_b_proj.weight",
                    Bf16,
                    &[heads * (nope + v), rank],
                );
                layer("self_attn.g_proj.weight", Bf16, &[heads * v, e]);
                layer("self_attn.o_proj.weight", Bf16, &[e, heads * v]);
                if let Some((index_heads, index_dim)) = self.index {
                    layer(
                        "self_attn.indexer.wq_b.weight",
                        Bf16,
                        &[index_heads * index_dim, q_lora],
                    );
                    layer("self_attn.indexer.wk.weight", Bf16, &[index_dim, e]);
                    layer(
                        "self_attn.indexer.weights_proj.weight",
                        Bf16,
                        &[index_heads, e],
                    );
                    layer("self_attn.indexer.k_norm.weight", Bf16, &[index_dim]);
                    layer("self_attn.indexer.k_norm.bias", Bf16, &[index_dim]);
                }
            } else {
                for proj in ["q_proj", "k_proj", "v_proj", "g_proj"] {
                    layer(&format!("self_attn.{proj}.weight"), Bf16, &[inner, e]);
                }
                for conv in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                    layer(
                        &format!("self_attn.{conv}.weight"),
                        F32,
                        &[inner, 1, self.conv],
                    );
                }
                layer("self_attn.f_a_proj.weight", Bf16, &[kda, e]);
                layer("self_attn.f_b_proj.weight", Bf16, &[inner, kda]);
                layer("self_attn.b_proj.weight", Bf16, &[heads, e]);
                layer("self_attn.A_log", F32, &[heads.next_multiple_of(A_LOG_PAD)]);
                layer("self_attn.dt_bias", F32, &[inner]);
                layer("self_attn.o_norm.weight", F32, &[kda]);
                layer("self_attn.o_proj.weight", Bf16, &[e, inner]);
            }
            if l < self.dense_lead {
                layer("mlp.gate_proj.weight", Bf16, &[self.ff, e]);
                layer("mlp.up_proj.weight", Bf16, &[self.ff, e]);
                layer("mlp.down_proj.weight", Bf16, &[e, self.ff]);
                continue;
            }
            let (m, shared) = ("block_sparse_moe.", self.shared * eff);
            layer(&format!("{m}gate.weight"), Bf16, &[self.experts, e]);
            layer(
                &format!("{m}gate.e_score_correction_bias"),
                F32,
                &[self.experts],
            );
            layer(
                &format!("{m}routed_expert_down_proj.weight"),
                Bf16,
                &[latent, e],
            );
            layer(
                &format!("{m}routed_expert_up_proj.weight"),
                Bf16,
                &[e, latent],
            );
            layer(&format!("{m}routed_expert_norm.weight"), Bf16, &[latent]);
            layer(
                &format!("{m}shared_experts.gate_proj.weight"),
                Bf16,
                &[shared, e],
            );
            layer(
                &format!("{m}shared_experts.up_proj.weight"),
                Bf16,
                &[shared, e],
            );
            layer(
                &format!("{m}shared_experts.down_proj.weight"),
                Bf16,
                &[e, shared],
            );
            for id in 0..self.experts {
                for (w, rows, k) in [
                    ("w1", eff, latent),
                    ("w3", eff, latent),
                    ("w2", latent, eff),
                ] {
                    let x = format!("{m}experts.{id}.{w}.");
                    layer(&format!("{x}weight_packed"), U8, &[rows, k / 2]);
                    layer(&format!("{x}weight_scale"), U8, &[rows, k / MXFP4_BLOCK]);
                }
            }
        }
        t.sort_by(|a, b| a.0.cmp(&b.0));
        t
    }

    /// Every tensor the config says the folder holds is mapped, and a layer holds the tensors of
    /// its kind and no other.
    fn complete(&self, mapped: &BTreeMap<String, Mapped<Step>>) -> Result<(), Error> {
        for name in [
            names::token_embd(),
            names::output(),
            names::output_norm(),
            names::OUTPUT_RES_SCORE.to_string(),
        ] {
            if !mapped.contains_key(&name) {
                return Err(refuse(&name, "is missing from the folder"));
            }
        }
        let mut by_layer: BTreeMap<usize, BTreeSet<&str>> = BTreeMap::new();
        for name in mapped.keys() {
            if let Some((l, _)) = name
                .strip_prefix("blk.")
                .and_then(|r| r.split_once('.'))
                .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
            {
                by_layer.entry(l).or_default().insert(name);
            }
        }
        for l in 0..self.layers {
            let want = self.layer_names(l);
            let held = by_layer.remove(&l).unwrap_or_default();
            if let Some(name) = want.iter().find(|n| !held.contains(n.as_str())) {
                return Err(refuse(name, "is missing from the folder"));
            }
            let want: BTreeSet<&str> = want.iter().map(String::as_str).collect();
            if let Some(name) = held.iter().find(|n| !want.contains(**n)) {
                let kind = if self.latent_layer[l] {
                    "latent-attention"
                } else {
                    "KDA"
                };
                return Err(refuse(
                    &mapped[*name].sources[0],
                    format!(
                        "is a tensor a {kind} layer {l} of this config does not hold (its \
                         linear_attn_config and dsa_config say which layers are which)"
                    ),
                ));
            }
        }
        Ok(())
    }
}

/// K3.1's indexer tensors of layer `l`, in the names Deepseek-V4.1's and Mistral Large 4's
/// descriptions give theirs. The K3 description declares none of them: R2 declares it.
fn indexer_names(l: usize) -> [String; 5] {
    [
        indexer::indexer_attn_q_b(l),
        indexer::indexer_attn_k(l),
        indexer::indexer_proj(l),
        indexer::indexer_k_norm(l),
        model::arch::names::blk(l, INDEXER_K_NORM_B),
    ]
}

/// A cross-layer residual's pair: the layer's attention mix, its feed-forward mix, or the final mix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Mix {
    Attention(usize),
    FeedForward(usize),
    Output,
}

impl Mix {
    /// The score tensor the pair makes.
    fn score(self) -> String {
        match self {
            Mix::Attention(l) => names::attn_res_score(l),
            Mix::FeedForward(l) => names::ffn_res_score(l),
            Mix::Output => names::OUTPUT_RES_SCORE.to_string(),
        }
    }
}

/// One stack's routed experts by id: each stored name and its shape.
type Stack<'a> = BTreeMap<usize, (&'a str, &'a [usize])>;

/// A pair's norm and projection, each with its stored name and shape.
type Pair<'a> = [Option<(&'a str, &'a [usize])>; 2];

impl FolderMap for Checkpoint {
    type Step = Step;

    const MODEL_TYPES: &'static [&'static str] = &["kimi_k3", "kimi_k31"];

    fn parse(config: &Json) -> Result<Self, Error> {
        Checkpoint::read(config)
    }

    fn architecture(&self) -> &'static str {
        self.word.word()
    }

    fn with_arch(mut self, word: &str) -> Result<Self, Error> {
        let to = [Variant::K3, Variant::K31]
            .into_iter()
            .find(|v| v.word() == word)
            .ok_or_else(|| {
                Error::Refused(format!(
                    "--arch {word:?}: this map's words are {:?} and {:?}",
                    Variant::K3.word(),
                    Variant::K31.word()
                ))
            })?;
        if to == Variant::K31 && self.variant == Variant::K3 {
            return Err(Error::Refused(format!(
                "--arch {word:?}: a K3 folder holds no indexer tensors, which {word} has on its \
                 latent layers"
            )));
        }
        self.word = to;
        Ok(self)
    }

    fn module_class(&self, module: &str) -> Option<ModuleClass> {
        let (layer, rest) = module
            .strip_prefix("language_model.model.layers.")?
            .split_once('.')?;
        layer.parse::<usize>().ok()?;
        let (id, projection) = rest
            .strip_prefix("block_sparse_moe.experts.")?
            .split_once('.')?;
        id.parse::<usize>().ok()?;
        matches!(projection, "w1" | "w2" | "w3").then_some(ModuleClass::Linear)
    }

    fn map<'a>(&self, tensors: &[(&'a str, &'a [usize])]) -> Result<Mapping<Step>, Error> {
        let mut out = Mapping {
            tensors: BTreeMap::new(),
            skipped: BTreeMap::new(),
        };
        let mut stacks: BTreeMap<(usize, &str), Stack> = BTreeMap::new();
        let mut pairs: BTreeMap<Mix, Pair> = BTreeMap::new();
        let inner = self.heads * self.kda_head;
        for &(name, shape) in tensors {
            if let Some(why) = skip_reason(name) {
                out.skipped.insert(name.to_string(), why.to_string());
                continue;
            }
            if !super::trunk::sized(shape) {
                return Err(refuse(
                    name,
                    format!("shape {shape:?} is empty, zero, or overflows usize"),
                ));
            }
            let dims = || shape.iter().rev().map(|&d| d as u64).collect::<Vec<_>>();
            let made = |role: Use, steps: Vec<Step>, dims: Vec<u64>, origin: Origin| Mapped {
                sources: vec![name.to_string()],
                role,
                steps,
                dims,
                origin,
            };
            let one = |role: Use| made(role, vec![], dims(), Origin::Made);
            let matrix = || one(Use::Matrix(Plain::Q8_0));
            let f32s = || one(Use::Matrix(Plain::F32));
            let shaped = |want: &[usize]| -> Result<(), Error> {
                if shape == want {
                    Ok(())
                } else {
                    Err(refuse(name, format!("shape {shape:?}, not {want:?}")))
                }
            };
            let mut put = |to: String, m: Mapped<Step>| match out.tensors.insert(to, m) {
                Some(prev) => Err(refuse(
                    &prev.sources[0],
                    "maps to a kimi-k3 name another tensor has",
                )),
                None => Ok(()),
            };
            let short = name
                .strip_prefix("language_model.")
                .ok_or_else(|| refuse(name, "not a Kimi K3 tensor"))?;
            let hidden = self.hidden as usize;
            match short {
                "lm_head.weight" => {
                    shaped(&[self.vocab_size as usize, hidden])?;
                    put(names::output(), matrix())?;
                    continue;
                }
                "model.embed_tokens.weight" => {
                    shaped(&[self.vocab_size as usize, hidden])?;
                    put(
                        names::token_embd(),
                        made(
                            Use::Matrix(Plain::Bf16),
                            vec![],
                            dims(),
                            Origin::File(GgmlType::BF16),
                        ),
                    )?;
                    continue;
                }
                "model.norm.weight" => {
                    shaped(&[hidden])?;
                    put(names::output_norm(), f32s())?;
                    continue;
                }
                "model.output_attn_res_norm.weight" | "model.output_attn_res_proj.weight" => {
                    let slot = usize::from(short.ends_with("proj.weight"));
                    pairs.entry(Mix::Output).or_default()[slot] = Some((name, shape));
                    continue;
                }
                _ => {}
            }
            let (l, rest) = short
                .strip_prefix("model.layers.")
                .and_then(|r| r.split_once('.'))
                .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
                .ok_or_else(|| refuse(name, "not a Kimi K3 tensor"))?;
            if l >= self.layers {
                return Err(refuse(name, format!("layer {l} of {}", self.layers)));
            }
            if let Some((id, projection)) = rest
                .strip_prefix("block_sparse_moe.experts.")
                .and_then(|e| e.split_once('.'))
                .and_then(|(id, p)| Some((id.parse::<usize>().ok()?, p)))
            {
                if !matches!(
                    projection,
                    "w1.weight_packed" | "w2.weight_packed" | "w3.weight_packed"
                ) {
                    return Err(refuse(name, "not a Kimi K3 expert projection"));
                }
                if stacks
                    .entry((l, projection))
                    .or_default()
                    .insert(id, (name, shape))
                    .is_some()
                {
                    return Err(refuse(name, format!("duplicate expert id {id}")));
                }
                continue;
            }
            let latent = self.latent_layer[l];
            match rest {
                "input_layernorm.weight" => put(names::attn_norm(l), f32s())?,
                "post_attention_layernorm.weight" => put(names::ffn_norm(l), f32s())?,
                "self_attention_res_norm.weight" | "self_attention_res_proj.weight" => {
                    let slot = usize::from(rest.ends_with("proj.weight"));
                    pairs.entry(Mix::Attention(l)).or_default()[slot] = Some((name, shape));
                }
                "mlp_res_norm.weight" | "mlp_res_proj.weight" => {
                    let slot = usize::from(rest.ends_with("proj.weight"));
                    pairs.entry(Mix::FeedForward(l)).or_default()[slot] = Some((name, shape));
                }
                "self_attn.o_proj.weight" => put(names::attn_output(l), matrix())?,
                "self_attn.g_proj.weight" => put(
                    if latent {
                        names::attn_gate(l)
                    } else {
                        names::ssm_g(l)
                    },
                    matrix(),
                )?,
                "self_attn.q_proj.weight" => put(names::attn_q(l), matrix())?,
                "self_attn.k_proj.weight" => put(names::attn_k(l), matrix())?,
                "self_attn.v_proj.weight" => put(names::attn_v(l), matrix())?,
                "self_attn.q_conv1d.weight"
                | "self_attn.k_conv1d.weight"
                | "self_attn.v_conv1d.weight" => {
                    shaped(&[inner, 1, self.conv])?;
                    let part = rest.as_bytes()["self_attn.".len()] as char;
                    put(names::ssm_conv1d(l, part), f32s())?;
                }
                "self_attn.f_a_proj.weight" => put(names::ssm_f_a(l), matrix())?,
                "self_attn.f_b_proj.weight" => put(names::ssm_f_b(l), matrix())?,
                "self_attn.b_proj.weight" => put(names::ssm_beta(l), f32s())?,
                "self_attn.A_log" => {
                    let &[len] = shape else {
                        return Err(refuse(
                            name,
                            format!("shape {shape:?}, not one value a head"),
                        ));
                    };
                    if len < self.heads {
                        return Err(refuse(
                            name,
                            format!("{len} values, fewer than the {} heads", self.heads),
                        ));
                    }
                    put(
                        names::ssm_a(l),
                        made(
                            Use::Matrix(Plain::F32),
                            vec![Step::NegExp { heads: self.heads }],
                            vec![self.heads as u64],
                            Origin::Made,
                        ),
                    )?;
                }
                "self_attn.dt_bias" => put(names::ssm_dt_bias(l), f32s())?,
                "self_attn.o_norm.weight" => put(names::ssm_norm(l), f32s())?,
                "self_attn.q_a_proj.weight" => put(names::attn_q_a(l), matrix())?,
                "self_attn.q_a_layernorm.weight" => put(names::attn_q_a_norm(l), f32s())?,
                "self_attn.q_b_proj.weight" => put(names::attn_q_b(l), matrix())?,
                "self_attn.kv_a_proj_with_mqa.weight" => put(names::attn_kv_a_mqa(l), matrix())?,
                "self_attn.kv_a_layernorm.weight" => put(names::attn_kv_a_norm(l), f32s())?,
                "self_attn.kv_b_proj.weight" => {
                    let &[rows, rank] = shape else {
                        return Err(refuse(name, format!("shape {shape:?}, not a matrix")));
                    };
                    let (heads, nope, v) = (self.heads, self.nope, self.v_head);
                    if rows != heads * (nope + v) || rank != self.kv_lora {
                        return Err(refuse(
                            name,
                            format!(
                                "shape {shape:?}, not {heads} heads of {nope} + {v} rows by the \
                                 latent rank {}",
                                self.kv_lora
                            ),
                        ));
                    }
                    let (rank, heads64) = (rank as u64, heads as u64);
                    // A split of single values: the card reads each half as BF16.
                    let bf16 = Use::Matrix(Plain::Bf16);
                    put(
                        names::attn_k_b(l),
                        made(
                            bf16,
                            vec![Step::KeyHalf { heads, nope, v }],
                            vec![nope as u64, rank, heads64],
                            Origin::Made,
                        ),
                    )?;
                    put(
                        names::attn_v_b(l),
                        made(
                            bf16,
                            vec![Step::ValueHalf { heads, nope, v }],
                            vec![rank, v as u64, heads64],
                            Origin::Made,
                        ),
                    )?;
                }
                "self_attn.indexer.wq_b.weight" => put(indexer::indexer_attn_q_b(l), matrix())?,
                "self_attn.indexer.wk.weight" => put(indexer::indexer_attn_k(l), matrix())?,
                "self_attn.indexer.weights_proj.weight" => put(indexer::indexer_proj(l), matrix())?,
                "self_attn.indexer.k_norm.weight" => put(indexer::indexer_k_norm(l), f32s())?,
                "self_attn.indexer.k_norm.bias" => {
                    put(model::arch::names::blk(l, INDEXER_K_NORM_B), f32s())?;
                }
                "mlp.gate_proj.weight" => put(names::ffn_gate(l), matrix())?,
                "mlp.up_proj.weight" => put(names::ffn_up(l), matrix())?,
                "mlp.down_proj.weight" => put(names::ffn_down(l), matrix())?,
                "block_sparse_moe.gate.weight" => put(names::ffn_gate_inp(l), f32s())?,
                "block_sparse_moe.gate.e_score_correction_bias" => {
                    put(names::exp_probs_b(l), f32s())?;
                }
                "block_sparse_moe.routed_expert_down_proj.weight" => {
                    put(names::ffn_routed_down(l), matrix())?;
                }
                "block_sparse_moe.routed_expert_up_proj.weight" => {
                    put(names::ffn_routed_up(l), matrix())?;
                }
                "block_sparse_moe.routed_expert_norm.weight" => {
                    put(names::ffn_routed_norm(l), f32s())?;
                }
                "block_sparse_moe.shared_experts.gate_proj.weight" => {
                    put(names::ffn_gate_shexp(l), matrix())?;
                }
                "block_sparse_moe.shared_experts.up_proj.weight" => {
                    put(names::ffn_up_shexp(l), matrix())?;
                }
                "block_sparse_moe.shared_experts.down_proj.weight" => {
                    put(names::ffn_down_shexp(l), matrix())?;
                }
                _ => return Err(refuse(name, "not a Kimi K3 tensor")),
            }
        }
        for ((l, projection), by_id) in stacks {
            let (sources, dims) = safetensors::expert_stack(&by_id, self.experts)
                .map_err(|e| refuse(&format!("layer {l} {projection}"), e.to_string()))?;
            // w1 is the gate, w3 the up and w2 the down projection of an expert's SwiGLU; the
            // experts run in the latent, so w1 and w3 read it and w2 writes it.
            let (to, want) = match projection {
                "w1.weight_packed" => (names::ffn_gate_exps(l), [self.latent, self.expert_ff]),
                "w3.weight_packed" => (names::ffn_up_exps(l), [self.latent, self.expert_ff]),
                _ => (names::ffn_down_exps(l), [self.expert_ff, self.latent]),
            };
            let dims: Vec<u64> = dims.into_iter().map(|d| d as u64).collect();
            if dims[..2] != want {
                return Err(refuse(
                    &sources[0],
                    format!(
                        "dims {:?}, not the latent {} by the expert width {}",
                        &dims[..2],
                        self.latent,
                        self.expert_ff
                    ),
                ));
            }
            out.tensors.insert(
                to,
                Mapped {
                    sources,
                    role: Use::Matrix(Plain::Q8_0),
                    steps: vec![],
                    dims,
                    origin: Origin::Made,
                },
            );
        }
        for (mix, pair) in pairs {
            let [Some((norm, norm_shape)), Some((proj, proj_shape))] = pair else {
                let have = pair.iter().flatten().map(|p| p.0).collect::<Vec<_>>();
                return Err(refuse(
                    have.first().copied().unwrap_or("a residual pair"),
                    format!("{have:?} is half of a norm and projection pair"),
                ));
            };
            let hidden = self.hidden as usize;
            if norm_shape != [hidden] {
                return Err(refuse(
                    norm,
                    format!("shape {norm_shape:?}, not [{hidden}]"),
                ));
            }
            if proj_shape != [1, hidden] {
                return Err(refuse(
                    proj,
                    format!("shape {proj_shape:?}, not [1, {hidden}]"),
                ));
            }
            let to = mix.score();
            if out
                .tensors
                .insert(
                    to,
                    Mapped {
                        sources: vec![norm.to_string(), proj.to_string()],
                        role: Use::Matrix(Plain::F32),
                        steps: vec![Step::ResProduct],
                        dims: vec![self.hidden],
                        origin: Origin::Made,
                    },
                )
                .is_some()
            {
                return Err(refuse(norm, "maps to a kimi-k3 name another tensor has"));
            }
        }
        self.complete(&out.tensors)?;
        Ok(out)
    }

    fn mover(&self, name: &str, steps: &[Step]) -> Option<part::Step> {
        (!steps.is_empty()).then(|| {
            let (name, steps) = (name.to_string(), steps.to_vec());
            Box::new(move |n: &mut Normal| steps.iter().try_for_each(|s| s.apply(&name, n)))
                as part::Step
        })
    }

    fn header(
        &self,
        planned: &[(&str, &[u64])],
        _folder: &Folder,
    ) -> Result<Vec<(String, Value)>, Error> {
        if self.word == Variant::K3
            && let Some((name, _)) = planned.iter().find(|(n, _)| n.contains(".indexer."))
        {
            return Err(refuse(
                name,
                format!(
                    "is an indexer tensor, which {} has none of: --drop 'blk.*.indexer.*' \
                     leaves them out",
                    self.word.word()
                ),
            ));
        }
        let keys: Vec<_> = self
            .keys
            .iter()
            .filter(|(k, _)| self.word == Variant::K31 || !k.starts_with("attention.indexer."))
            .cloned()
            .collect();
        Ok(super::trunk::header(self.word.word(), &keys))
    }

    fn vocab(&self) -> &VocabSpec {
        &self.vocab
    }

    fn stored(&self) -> Option<Stored> {
        Some(Stored {
            tensors: self.stored_tensors(),
            skips: skip_reason,
        })
    }
}
