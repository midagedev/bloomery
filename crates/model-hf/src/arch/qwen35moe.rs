//! Qwen3.8-Flash-Next's Hugging Face checkpoint in the qwen4exp names, layouts and keys the
//! program reads, as llama.cpp's converter (`conversion/qwen4exp.py`, `Qwen4ExpTextModel`, and
//! the `qwen.py` classes it extends, at `098ce526`) writes them into a GGUF.
//!
//! A checkpoint tensor maps to one qwen4exp tensor ([`names`]), the way its role reads it
//! ([`Use`]) and the [`Step`]s that move it into the program's layout once its format is
//! normalized ([`hf::modelopt::normalize`]): the gated delta-rule V heads from grouped by K head
//! to tiled (`_LinearAttentionVReorderBase`), the indexer's joint q and k projection split by
//! rows, `+1` on the zero-centred gammas, and `A_log` as `-exp`. The routed experts of a
//! projection stack by expert id, and the PLE table's parts concatenate in part order. The
//! next-token layers (`mtp.*`) and the vision tower are left out; any other name is refused.

use std::collections::BTreeMap;
use std::ops::Range;

use gguf::Value;
use hf::modelopt::{self, Normal, Plain, Use};
use hf::quant_config::{Placed, Plan};
use hf::safetensors::{self, Raw};
use hf::schema::{self, Fields, Rule::*, Type};
use serde_json::Value as Json;

use model::arch::qwen35moe::hparams::{MAX_PLE_HEADS, MAX_PLE_NGRAM};
use model::arch::qwen35moe::names::{self, Sub};

/// Why the checkpoint was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("qwen4exp hf: config.json: {0}")]
    Config(String),
    #[error("qwen4exp hf: {name}: {what}")]
    Tensor { name: String, what: String },
}

/// The plan's place of the tensor `name`, or its refusal by name.
fn place<'p>(plan: &'p Plan, name: &str) -> Result<&'p Placed, Error> {
    plan.placed.get(name).ok_or_else(|| {
        let why = plan
            .refused
            .get(name)
            .map_or("not a weight of the plan", String::as_str);
        refuse(name, why)
    })
}

fn refuse(name: &str, what: impl Into<String>) -> Error {
    Error::Tensor {
        name: name.to_string(),
        what: what.into(),
    }
}

const ARCHITECTURE: &str = "Qwen4ExpForConditionalGeneration";

/// The checkpoint's prefix of the text model's tensors.
const TEXT: &str = "model.language_model.";

const VISION: &str = "the vision tower and its placeholders; this converter writes the text model";
const MTP: &str =
    "the next-token layers, which this converter leaves out and the trunk never reads";
const PLE_SIZES: &str = "sized the PLE hash tables in training; the checkpoint's ngram_heads_offsets \
    and ngram_heads_vocab_sizes hold the sizes, which are exported";

/// `config.json`. `hf::quant_config` reads `quantization_config`; the vocabulary keys are
/// `text_config`'s, which `hf::vocab::Config` reads.
const ROOT: &Fields = &[
    ("architectures", Read),
    ("model_type", Need(r#"["qwen4_exp"]"#)),
    ("text_config", Read),
    ("quantization_config", Read),
    // The map needs a separate lm_head.
    ("tie_word_embeddings", Only("[false]")),
    ("image_token_id", Read),
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
    ("vision_config", Ignore(Type::Object, VISION)),
    ("video_token_id", Ignore(Type::Number, VISION)),
    ("vision_start_token_id", Ignore(Type::Number, VISION)),
    ("vision_end_token_id", Ignore(Type::Number, VISION)),
    ("language_model_only", Ignore(Type::Bool, VISION)),
];

/// `config.json`'s `text_config`. A setting qwen4exp computes with one value and GGUF has no key
/// for is `Only` that value: another would be served as this one, and an absent one is it.
const TEXT_CONFIG: &Fields = &[
    ("num_hidden_layers", Read),
    ("max_position_embeddings", Read),
    ("hidden_size", Read),
    ("num_attention_heads", Read),
    ("num_key_value_heads", Read),
    ("num_experts", Read),
    ("num_experts_per_tok", Read),
    ("head_dim", Read),
    ("moe_intermediate_size", Read),
    ("shared_expert_intermediate_size", Read),
    ("linear_conv_kernel_dim", Read),
    ("linear_key_head_dim", Read),
    ("linear_num_key_heads", Read),
    ("linear_num_value_heads", Read),
    ("linear_value_head_dim", Read),
    ("full_attention_interval", Read),
    ("hc_count", Read),
    ("hc_lowrank", Read),
    ("indexer_n_heads", Read),
    ("indexer_head_dim", Read),
    ("indexer_budget", Read),
    ("indexer_compress_ratio", Read),
    ("rms_norm_eps", Read),
    ("layer_types", Read),
    ("rope_parameters", Read),
    ("partial_rotary_factor", Read),
    ("ple_layer_ids", Read),
    ("ngram_size", Read),
    ("heads_per_ngram", Read),
    ("ple_conv_kernel_size", Read),
    ("split_ngram_parts", Read),
    ("ple_embed_dim", Read),
    // hf::vocab::Config's.
    ("vocab_size", Read),
    ("bos_token_id", Read),
    ("eos_token_id", Read),
    ("pad_token_id", Read),
    ("unk_token_id", Read),
    ("sep_token_id", Read),
    ("model_type", Only(r#"["qwen4_exp_text"]"#)),
    // The routed and shared experts' SwiGLU.
    ("hidden_act", Only(r#"["silu"]"#)),
    // The gated delta-rule output gate.
    ("output_gate_type", Only(r#"["sigmoid"]"#)),
    // The router renormalizes its top k.
    ("norm_topk_prob", Only("[true]")),
    ("attention_bias", Only("[false]")),
    // The indexer's key projection is one head of indexer_head_dim rows.
    ("indexer_kv_heads", Only("[1]")),
    ("tie_word_embeddings", Only("[false]")),
    // Serving keeps the gated delta-rule state in F32.
    ("mamba_ssm_dtype", Only(r#"["float32"]"#)),
    (
        "dtype",
        Ignore(
            Type::String,
            "the dtype HF loads in; each tensor keeps its stored dtype and format",
        ),
    ),
    (
        "attention_dropout",
        Ignore(Type::Number, "dropout acts in training only"),
    ),
    (
        "initializer_range",
        Ignore(Type::Number, "initializes weights the checkpoint holds"),
    ),
    (
        "router_aux_loss_coef",
        Ignore(Type::Number, "weighs a training loss"),
    ),
    (
        "output_router_logits",
        Ignore(
            Type::Bool,
            "returns the router logits beside an unchanged output",
        ),
    ),
    (
        "use_cache",
        Ignore(
            Type::Bool,
            "whether HF keeps its cache; outputs are the same",
        ),
    ),
    ("seed", Ignore(Type::Number, "a training seed")),
    (
        "number_of_conv_states",
        Ignore(
            Type::Number,
            "the state slots HF's cache allocates a linear layer; serving keeps its own state",
        ),
    ),
    (
        "make_ngram_vocab_size_divisible_by",
        Ignore(Type::Number, PLE_SIZES),
    ),
    ("ngram_vocab_size_base", Ignore(Type::Number, PLE_SIZES)),
    (
        "ple_embedding_dtype",
        Ignore(
            Type::String,
            "the PLE table's storage format; the quantization config places each part by its tensors",
        ),
    ),
    ("mtp", Ignore(Type::Object, MTP)),
    ("mtp_num_hidden_layers", Ignore(Type::Number, MTP)),
    ("mtp_use_dedicated_embeddings", Ignore(Type::Bool, MTP)),
];

/// `text_config.rope_parameters`: interleaved M-RoPE (`RopeMode::Imrope`), unscaled.
const ROPE: &Fields = &[
    ("rope_type", Only(r#"["default"]"#)),
    ("mrope_interleaved", Need("[true]")),
    ("mrope_section", Read),
    ("rope_theta", Read),
    ("partial_rotary_factor", Read),
];

/// The config values the map reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Geometry {
    layers: usize,
    k_heads: usize,
    v_heads: usize,
    head_k: usize,
    head_v: usize,
    /// The indexer's query rows of `index_qk_proj`; its key rows follow.
    indexer_q: usize,
    experts: usize,
    /// The PLE table's parts (`split_ngram_parts`).
    ple_parts: usize,
}

/// The checkpoint values used by preparation, checked before mapping or writing.
#[derive(Debug)]
pub struct Checkpoint {
    pub geometry: Geometry,
    pub vocab: hf::vocab::Config,
    hidden: u32,
    indexer_k: u32,
    metadata: Vec<(String, Value)>,
    ple: Option<Ple>,
}

#[derive(Debug)]
struct Ple {
    layer: usize,
    ngram: u32,
    heads: u32,
}

struct Reader<'a> {
    value: &'a Json,
    path: &'a str,
}

impl<'a> Reader<'a> {
    fn new(value: &'a Json, path: &'a str) -> Result<Self, Error> {
        if !value.is_object() {
            return Err(Error::Config(format!("{path} is not an object")));
        }
        Ok(Self { value, path })
    }

    fn field(&self, key: &str) -> Result<&'a Json, Error> {
        self.value
            .get(key)
            .ok_or_else(|| self.bad(key, "is missing"))
    }

    fn bad(&self, key: &str, why: &str) -> Error {
        Error::Config(format!("{}.{key} {why}", self.path))
    }

    fn count(&self, key: &str) -> Result<u32, Error> {
        self.field(key)?
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| self.bad(key, "must be a positive u32"))
    }

    fn number(&self, key: &str) -> Result<f32, Error> {
        self.field(key)?
            .as_f64()
            .map(|n| n as f32)
            .filter(|n| n.is_finite() && *n > 0.0)
            .ok_or_else(|| self.bad(key, "must be a finite positive f32"))
    }

    fn array(&self, key: &str) -> Result<&'a [Json], Error> {
        self.field(key)?
            .as_array()
            .map(Vec::as_slice)
            .ok_or_else(|| self.bad(key, "is not an array"))
    }
}

impl Checkpoint {
    /// Read the fields this converter uses, and refuse by name a key [`ROOT`], [`TEXT_CONFIG`] or
    /// [`ROPE`] does not name or a value they do not allow.
    pub fn parse(config: &Json) -> Result<Self, Error> {
        let root = Reader::new(config, "config")?;
        let arch = root.array("architectures")?;
        if !matches!(arch, [Json::String(s)] if s == ARCHITECTURE) {
            return Err(root.bad(
                "architectures",
                &format!("{arch:?}: only {ARCHITECTURE} is supported"),
            ));
        }
        schema::check(config, "config", &[ROOT]).map_err(Error::Config)?;
        let c = Reader::new(root.field("text_config")?, "text_config")?;
        schema::check(c.value, "text_config", &[TEXT_CONFIG]).map_err(Error::Config)?;
        let vocab = hf::vocab::Config::parse(config).map_err(|e| Error::Config(e.to_string()))?;
        let mut counts = BTreeMap::new();
        for field in [
            "num_hidden_layers",
            "max_position_embeddings",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "num_experts",
            "num_experts_per_tok",
            "head_dim",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_value_head_dim",
            "full_attention_interval",
            "hc_count",
            "hc_lowrank",
            "indexer_n_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        ] {
            counts.insert(field, c.count(field)?);
        }
        let n = |key: &str| counts[key];
        let product = |a: &str, b: &str| {
            n(a).checked_mul(n(b))
                .ok_or_else(|| c.bad(a, &format!("times {b} overflows u32")))
        };
        let inner = product("linear_value_head_dim", "linear_num_value_heads")?;
        let indexer_q = product("indexer_n_heads", "indexer_head_dim")?;
        product("linear_num_key_heads", "linear_key_head_dim")?
            .checked_mul(2)
            .and_then(|qk| qk.checked_add(inner))
            .ok_or_else(|| {
                c.bad(
                    "linear_key_head_dim",
                    "makes the QKV row count overflow u32",
                )
            })?;
        if !n("linear_num_value_heads").is_multiple_of(n("linear_num_key_heads")) {
            return Err(c.bad(
                "linear_num_value_heads",
                "must group by linear_num_key_heads",
            ));
        }
        if !n("num_attention_heads").is_multiple_of(n("num_key_value_heads")) {
            return Err(c.bad("num_attention_heads", "must group by num_key_value_heads"));
        }
        if n("num_experts_per_tok") > n("num_experts") {
            return Err(c.bad("num_experts_per_tok", "exceeds num_experts"));
        }
        if n("hc_count") < 2 {
            return Err(c.bad("hc_count", "must be at least two"));
        }
        let ratio = i32::try_from(n("indexer_compress_ratio"))
            .map_err(|_| c.bad("indexer_compress_ratio", "does not fit i32"))?;
        if !n("indexer_budget").is_multiple_of(ratio as u32) {
            return Err(c.bad("indexer_compress_ratio", "must divide indexer_budget"));
        }
        let kinds = c.array("layer_types")?;
        if kinds.len() != n("num_hidden_layers") as usize {
            return Err(c.bad("layer_types", "length differs from num_hidden_layers"));
        }
        let mut ratios = vec![];
        for (i, kind) in kinds.iter().enumerate() {
            let full = match kind.as_str() {
                Some("linear_attention") => false,
                Some("full_attention") => true,
                _ => {
                    return Err(c.bad(
                        "layer_types",
                        &format!("unsupported layer type {kind} at {i}"),
                    ));
                }
            };
            if full != ((i + 1) % n("full_attention_interval") as usize == 0) {
                return Err(c.bad("layer_types", "disagrees with full_attention_interval"));
            }
            ratios.push(Value::I32(if full { ratio } else { 0 }));
        }
        let rope = Reader::new(c.field("rope_parameters")?, "rope_parameters")?;
        schema::check(rope.value, "rope_parameters", &[ROPE]).map_err(Error::Config)?;
        let [in_rope, in_text] = [&rope, &c].map(|r| {
            r.value
                .get("partial_rotary_factor")
                .map(|v| {
                    v.as_f64()
                        .ok_or_else(|| r.bad("partial_rotary_factor", "is not a number"))
                })
                .transpose()
        });
        let (in_rope, in_text) = (in_rope?, in_text?);
        if in_rope.zip(in_text).is_some_and(|(a, b)| a != b) {
            return Err(c.bad(
                "partial_rotary_factor",
                "disagrees with rope_parameters.partial_rotary_factor",
            ));
        }
        let partial = in_rope.or(in_text).unwrap_or(0.25);
        let rotary = f64::from(n("head_dim")) * partial;
        if !(partial > 0.0
            && partial <= 1.0
            && rotary >= 2.0
            && rotary.fract() == 0.0
            && (rotary as u32).is_multiple_of(2))
        {
            return Err(rope.bad(
                "partial_rotary_factor",
                "must be in (0, 1] and select a positive even whole head width",
            ));
        }
        let mut sections = vec![];
        let mut sum = 0u32;
        let raw_sections = rope.array("mrope_section")?;
        if raw_sections.is_empty() || raw_sections.len() > 4 {
            return Err(rope.bad("mrope_section", "must have one to four entries"));
        }
        for value in raw_sections {
            let count = value
                .as_u64()
                .and_then(|n| i32::try_from(n).ok())
                .ok_or_else(|| rope.bad("mrope_section", "must contain nonnegative i32 counts"))?;
            sum = sum
                .checked_add(count as u32)
                .ok_or_else(|| rope.bad("mrope_section", "sum overflows u32"))?;
            sections.push(Value::I32(count));
        }
        if sum.checked_mul(2) != Some(rotary as u32) {
            return Err(rope.bad("mrope_section", "must sum to half the rotary head width"));
        }
        sections.resize(4, Value::I32(0));
        let mut metadata: Vec<(String, Value)> = vec![
            ("block_count".into(), Value::U32(n("num_hidden_layers"))),
            (
                "context_length".into(),
                Value::U32(n("max_position_embeddings")),
            ),
            ("embedding_length".into(), Value::U32(n("hidden_size"))),
            (
                "attention.head_count".into(),
                Value::U32(n("num_attention_heads")),
            ),
            (
                "attention.head_count_kv".into(),
                Value::U32(n("num_key_value_heads")),
            ),
            ("rope.dimension_sections".into(), Value::Array(sections)),
            (
                "rope.freq_base".into(),
                Value::F32(rope.number("rope_theta")?),
            ),
            (
                "attention.layer_norm_rms_epsilon".into(),
                Value::F32(c.number("rms_norm_eps")?),
            ),
            ("expert_count".into(), Value::U32(n("num_experts"))),
            (
                "expert_used_count".into(),
                Value::U32(n("num_experts_per_tok")),
            ),
            ("attention.key_length".into(), Value::U32(n("head_dim"))),
            ("attention.value_length".into(), Value::U32(n("head_dim"))),
            (
                "expert_feed_forward_length".into(),
                Value::U32(n("moe_intermediate_size")),
            ),
            (
                "expert_shared_feed_forward_length".into(),
                Value::U32(n("shared_expert_intermediate_size")),
            ),
            (
                "ssm.conv_kernel".into(),
                Value::U32(n("linear_conv_kernel_dim")),
            ),
            (
                "ssm.state_size".into(),
                Value::U32(n("linear_key_head_dim")),
            ),
            (
                "ssm.group_count".into(),
                Value::U32(n("linear_num_key_heads")),
            ),
            (
                "ssm.time_step_rank".into(),
                Value::U32(n("linear_num_value_heads")),
            ),
            ("ssm.inner_size".into(), Value::U32(inner)),
            (
                "full_attention_interval".into(),
                Value::U32(n("full_attention_interval")),
            ),
            ("rope.dimension_count".into(), Value::U32(rotary as u32)),
            ("hyper_connection.count".into(), Value::U32(n("hc_count"))),
            (
                "hyper_connection.low_rank".into(),
                Value::U32(n("hc_lowrank")),
            ),
            (
                "attention.indexer.head_count".into(),
                Value::U32(n("indexer_n_heads")),
            ),
            (
                "attention.indexer.key_length".into(),
                Value::U32(n("indexer_head_dim")),
            ),
            (
                "attention.indexer.top_k".into(),
                Value::U32(n("indexer_budget")),
            ),
            ("attention.compress_ratios".into(), Value::Array(ratios)),
        ];
        if let Some(width) = c.value.get("ple_embed_dim")
            && width.as_u64() != Some(u64::from(n("hidden_size")))
        {
            // The PLE table's heads of `row` each are one hidden_size row (`keys`).
            return Err(c.bad("ple_embed_dim", "must equal hidden_size"));
        }
        let ple_layers = c
            .value
            .get("ple_layer_ids")
            .map(|_| c.array("ple_layer_ids"))
            .transpose()?
            .unwrap_or(&[]);
        let ple = if ple_layers.is_empty() {
            None
        } else {
            let [layer] = ple_layers else {
                return Err(c.bad("ple_layer_ids", "must name exactly one layer"));
            };
            let layer = layer
                .as_u64()
                .and_then(|i| i.checked_sub(1))
                .and_then(|i| usize::try_from(i).ok())
                .filter(|&i| kinds.get(i).and_then(Json::as_str) == Some("linear_attention"))
                .ok_or_else(|| {
                    c.bad(
                        "ple_layer_ids",
                        "must name a linear layer in 1..=num_hidden_layers",
                    )
                })?;
            let ngram = c.count("ngram_size")?;
            let per = c.count("heads_per_ngram")?;
            let heads = (ngram - 1)
                .checked_mul(per)
                .filter(|&h| h > 0 && h as usize <= MAX_PLE_HEADS)
                .filter(|_| ngram >= 2 && ngram as usize <= MAX_PLE_NGRAM)
                .ok_or_else(|| {
                    c.bad(
                        "ngram_size",
                        "or heads_per_ngram is outside the supported PLE geometry",
                    )
                })?;
            let eos = vocab
                .ple_eos()
                .ok_or_else(|| c.bad("eos_token_id", "is required for PLE"))?;
            metadata.extend([
                (
                    "ple.layers".into(),
                    Value::Array(vec![Value::I32(
                        i32::try_from(layer)
                            .map_err(|_| c.bad("ple_layer_ids", "does not fit i32"))?,
                    )]),
                ),
                ("ple.ngram_size".into(), Value::U32(ngram)),
                ("ple.heads_per_ngram".into(), Value::U32(per)),
                (
                    "ple.conv_kernel".into(),
                    Value::U32(c.count("ple_conv_kernel_size")?),
                ),
                ("ple.eos_token_id".into(), Value::U32(eos)),
            ]);
            if let Some(img) = config.get("image_token_id").filter(|v| !v.is_null()) {
                let id = img
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .filter(|&n| n < vocab.size())
                    .ok_or_else(|| {
                        root.bad("image_token_id", "must be a u32 id inside vocab_size")
                    })?;
                metadata.push(("ple.image_token_id".into(), Value::U32(id)));
            }
            Some(Ple {
                layer,
                ngram,
                heads,
            })
        };
        let ple_parts = if ple.is_some() {
            c.count("split_ngram_parts")? as usize
        } else {
            0
        };
        Ok(Self {
            geometry: Geometry {
                layers: n("num_hidden_layers") as usize,
                k_heads: n("linear_num_key_heads") as usize,
                v_heads: n("linear_num_value_heads") as usize,
                head_k: n("linear_key_head_dim") as usize,
                head_v: n("linear_value_head_dim") as usize,
                indexer_q: indexer_q as usize,
                experts: n("num_experts") as usize,
                ple_parts,
            },
            hidden: n("hidden_size"),
            indexer_k: n("indexer_head_dim"),
            vocab,
            metadata,
            ple,
        })
    }

    /// Map checked geometry and require token rows to match the checkpoint vocabulary.
    pub fn map<'a>(
        &self,
        tensors: impl IntoIterator<Item = (&'a str, &'a [usize])>,
    ) -> Result<BTreeMap<String, Mapped>, Error> {
        let mapped = map(&self.geometry, tensors)?;
        for m in mapped.values() {
            for name in &m.sources {
                if name.contains(".ple.")
                    && self.ple.as_ref().is_none_or(|ple| {
                        !name.starts_with(&format!("{TEXT}layers.{}.ple.", ple.layer))
                    })
                {
                    return Err(refuse(name, "PLE site disagrees with ple_layer_ids"));
                }
            }
        }
        for l in 0..self.geometry.layers {
            if let Some(m) = mapped.get(&names::indexer_k_proj(l))
                && m.dims[1] != u64::from(self.indexer_k)
            {
                return Err(refuse(
                    &m.sources[0],
                    format!(
                        "holds {} key rows, not one head of indexer_head_dim {}",
                        m.dims[1], self.indexer_k
                    ),
                ));
            }
        }
        for name in [names::token_embd(), names::output()] {
            let m = mapped
                .get(&name)
                .ok_or_else(|| refuse(&name, "is missing"))?;
            if m.dims != [u64::from(self.hidden), u64::from(self.vocab.size())] {
                return Err(refuse(
                    &m.sources[0],
                    format!(
                        "dims {:?} disagree with hidden_size {} and vocab_size {}",
                        m.dims,
                        self.hidden,
                        self.vocab.size()
                    ),
                ));
            }
        }
        Ok(mapped)
    }

    /// Metadata for the shared preparer, including the PLE table's shape and hash constants.
    pub fn header<'a>(
        &self,
        tensors: impl IntoIterator<Item = (&'a str, &'a [u64])>,
        folder: &safetensors::Folder,
    ) -> Result<Vec<(String, Value)>, Error> {
        let ple_shape = tensors
            .into_iter()
            .find(|(name, _)| *name == names::per_layer_token_embd())
            .map_or(&[][..], |(_, dims)| dims);
        let constant = |name: &str| {
            let raw = folder.raw(name).map_err(|e| Error::Config(e.to_string()))?;
            if raw.dtype != safetensors::Dtype::I64 || raw.shape.len() != 1 {
                return Err(Error::Config(format!(
                    "{name} is {} {:?}, not a one-dimensional I64 array",
                    raw.dtype.name(),
                    raw.shape
                )));
            }
            Ok(raw
                .bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b))
                .collect())
        };
        let mut keys = vec![(
            gguf::GENERAL_ARCHITECTURE.to_string(),
            Value::String("qwen4exp".into()),
        )];
        keys.extend(self.keys(ple_shape, constant)?);
        Ok(keys)
    }

    /// Checked metadata and PLE hash tensors, with the mapped PLE table's shape.
    fn keys(
        &self,
        ple_shape: &[u64],
        constant: impl Fn(&str) -> Result<Vec<i64>, Error>,
    ) -> Result<Vec<(String, Value)>, Error> {
        let mut out = self.metadata.clone();
        if let Some(Ple {
            layer,
            ngram,
            heads,
        }) = self.ple.as_ref()
        {
            let (ngram, heads) = (*ngram, *heads);
            let &[row, rows] = ple_shape else {
                return Err(Error::Config("PLE table must have two dimensions".into()));
            };
            let row = u32::try_from(row)
                .ok()
                .filter(|&r| r > 0 && r.checked_mul(heads) == Some(self.hidden))
                .ok_or_else(|| {
                    Error::Config(
                        "PLE table row width disagrees with hidden_size and heads_per_ngram".into(),
                    )
                })?;
            out.push(("embedding_length_per_layer_input".into(), Value::U32(row)));
            let mut arrays = vec![];
            for ((part, key), len) in PLE_CONSTANTS.into_iter().zip([ngram, heads, heads]) {
                let name = format!("{TEXT}layers.{layer}.ple.ple_embedding.{part}");
                let values = constant(&name)?;
                if values.len() != len as usize {
                    return Err(refuse(
                        part,
                        format!("has {} values, expected {len}", values.len()),
                    ));
                }
                arrays.push(values.iter().map(|&v| v as u64).collect::<Vec<_>>());
                out.push((
                    key.into(),
                    Value::Array(values.into_iter().map(|v| Value::U64(v as u64)).collect()),
                ));
            }
            for (&off, &size) in arrays[1].iter().zip(&arrays[2]) {
                if size == 0
                    || off
                        .checked_add(size)
                        .is_none_or(|end| end > i32::MAX as u64 || end > rows)
                {
                    return Err(refuse(
                        "ngram_heads_vocab_sizes",
                        "range is empty, outside i32, or past the PLE table",
                    ));
                }
            }
        } else if !ple_shape.is_empty() {
            return Err(Error::Config(
                "PLE table present without ple_layer_ids".into(),
            ));
        }
        Ok(out
            .into_iter()
            .map(|(k, v)| (format!("qwen4exp.{k}"), v))
            .collect())
    }
}

/// A move of a normalized tensor into the program's layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// Rows `from..` are V heads of `head` rows, grouped by K head; put them in tiled order
    /// (head `j` of group `i` to place `j · k_heads + i`).
    VRows { from: usize, head: usize },
    /// Each row's columns are V heads of `head` columns, grouped by K head; tile them.
    VCols { head: usize },
    /// Only these rows.
    Rows(Range<usize>),
    /// `x + 1`, on F32.
    PlusOne,
    /// `-exp(x)`, on F32, computed in f64 and rounded once.
    NegExp,
}

/// One qwen4exp tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mapped {
    /// The checkpoint tensors it is made from: one, a stack's experts in expert order, or the
    /// PLE table's parts in part order.
    pub sources: Vec<String>,
    pub role: Use,
    pub steps: Vec<Step>,
    /// ggml's dims, the fastest axis first.
    pub dims: Vec<u64>,
}

impl Mapped {
    /// Each source with its place in `plan`, in source order: what [`modelopt::normalize`]
    /// reads, and [`modelopt::format_of`] the one format of.
    pub fn placed<'p>(&'p self, plan: &'p Plan) -> Result<Vec<(&'p str, &'p Placed)>, Error> {
        self.sources
            .iter()
            .map(|s| Ok((s.as_str(), place(plan, s)?)))
            .collect()
    }

    /// A row table's parts, one at a time in order: each normalized by its place in `plan`, and
    /// refused when its type or tensor scale is not the first part's, as the table keeps one.
    pub fn parts<'s, 'a: 's>(
        &'s self,
        plan: &'s Plan,
        get: impl Fn(&str) -> Result<Raw<'a>, safetensors::Error> + Copy + 's,
    ) -> impl Iterator<Item = Result<Normal, Error>> + 's {
        let mut table: Option<(u32, Vec<f32>)> = None;
        self.sources.iter().map(move |src| {
            let p = place(plan, src)?;
            let n = modelopt::normalize(get, &[(src.as_str(), p)], self.role)
                .map_err(|e| refuse(src, e.to_string()))?;
            match &table {
                Some((ty, scale)) if (*ty, scale) != (n.type_id, &n.scale) => Err(refuse(
                    src,
                    format!(
                        "type {} scale {:?}, the table's type {ty} scale {scale:?}",
                        n.type_id, n.scale
                    ),
                )),
                _ => {
                    table.get_or_insert_with(|| (n.type_id, n.scale.clone()));
                    Ok(n)
                }
            }
        })
    }
}

/// Tensors stored in parts by index (a layer's experts, the PLE table's shards): name and shape.
type Parts<'a> = BTreeMap<usize, (&'a str, &'a [usize])>;

/// The PLE hash constants, by the checkpoint name's last part, and the keys they become.
const PLE_CONSTANTS: [(&str, &str); 3] = [
    ("layer_multipliers", "ple.layer_multipliers"),
    ("ngram_heads_offsets", "ple.head_offsets"),
    ("ngram_heads_vocab_sizes", "ple.head_vocab_sizes"),
];

/// The qwen4exp tensors, by name, of the checkpoint tensors `tensors` (name and row-major
/// logical shape: a packed weight's caller gives its unpacked shape) under `g`. The PLE hash
/// constants are keys ([`Checkpoint::keys`]), not tensors.
fn map<'a>(
    g: &Geometry,
    tensors: impl IntoIterator<Item = (&'a str, &'a [usize])>,
) -> Result<BTreeMap<String, Mapped>, Error> {
    let mut out = BTreeMap::new();
    let mut put = |name: String, m: Mapped| match out.insert(name, m) {
        Some(prev) => Err(refuse(
            &prev.sources[0],
            "maps to a qwen4exp name another tensor has",
        )),
        None => Ok(()),
    };
    // A layer's experts: (layer, projection) to each expert's name and shape.
    let mut experts: BTreeMap<(usize, &str), Parts> = BTreeMap::new();
    let mut ple = Parts::new();
    for (name, shape) in tensors {
        if !name.starts_with("mtp.")
            && !name.starts_with("model.visual.")
            && (shape.is_empty()
                || shape.contains(&0)
                || shape
                    .iter()
                    .try_fold(1usize, |n, &d| n.checked_mul(d))
                    .is_none())
        {
            return Err(refuse(
                name,
                format!("shape {shape:?} is empty, zero, or overflows usize"),
            ));
        }
        let dims = || shape.iter().rev().map(|&d| d as u64).collect::<Vec<_>>();
        let one = |role: Use, steps: Vec<Step>, dims: Vec<u64>| Mapped {
            sources: vec![name.to_string()],
            role,
            steps,
            dims,
        };
        if name.starts_with("mtp.") || name.starts_with("model.visual.") {
            continue;
        }
        if name == "lm_head.weight" {
            put(
                names::output(),
                one(Use::Matrix(Plain::Q8_0), vec![], dims()),
            )?;
            continue;
        }
        let short = name
            .strip_prefix(TEXT)
            .ok_or_else(|| refuse(name, "not a Qwen3.8 tensor"))?;
        let mixer = |rest: &str| match rest {
            "hc_norm.weight" => Some((Plain::F32, vec![Step::PlusOne])),
            "input_mix_weight_down.weight" | "input_mix_weight_up.weight" => {
                Some((Plain::Q8_0, vec![]))
            }
            _ => None,
        };
        if short == "embed_tokens.weight" {
            put(
                names::token_embd(),
                one(Use::Matrix(Plain::Q8_0), vec![], dims()),
            )?;
            continue;
        }
        if let Some(rest) = short.strip_prefix("hyper_connection_mixer.") {
            let (plain, steps) = mixer(rest).ok_or_else(|| refuse(name, "not a mixer tensor"))?;
            let to = match rest {
                "hc_norm.weight" => names::output_hc_norm(),
                "input_mix_weight_down.weight" => names::output_hc_down(),
                _ => names::output_hc_up(),
            };
            put(to, one(Use::Matrix(plain), steps, dims()))?;
            continue;
        }
        let (l, rest) = short
            .strip_prefix("layers.")
            .and_then(|r| r.split_once('.'))
            .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
            .ok_or_else(|| refuse(name, "not a Qwen3.8 tensor"))?;
        if l >= g.layers {
            return Err(refuse(name, format!("layer {l} of {}", g.layers)));
        }
        if let Some(e) = rest.strip_prefix("mlp.experts.") {
            let (id, proj) = e
                .split_once('.')
                .and_then(|(id, p)| Some((id.parse::<usize>().ok()?, p.strip_suffix(".weight")?)))
                .filter(|(_, p)| ["gate_proj", "up_proj", "down_proj"].contains(p))
                .ok_or_else(|| refuse(name, "not a routed expert projection"))?;
            if experts
                .entry((l, proj))
                .or_default()
                .insert(id, (name, shape))
                .is_some()
            {
                return Err(refuse(name, format!("duplicate expert id {id}")));
            }
            continue;
        }
        if let Some(p) = rest.strip_prefix("ple.ple_embedding.") {
            if let Some(part) = p
                .strip_prefix("ngram_embedding.shard_")
                .and_then(|p| p.strip_suffix(".weight"))
            {
                let i = part
                    .parse::<usize>()
                    .map_err(|_| refuse(name, "not a numbered PLE part"))?;
                if ple.insert(i, (name, shape)).is_some() {
                    return Err(refuse(name, format!("duplicate PLE part id {i}")));
                }
            } else if !PLE_CONSTANTS.iter().any(|(c, _)| *c == p) {
                return Err(refuse(name, "not a PLE tensor"));
            }
            continue;
        }
        for (to, m) in layer(g, l, rest, name, shape)? {
            put(to, m)?;
        }
    }
    for ((l, proj), by_id) in experts {
        let (sources, dims) = safetensors::expert_stack(&by_id, g.experts)
            .map_err(|e| refuse(&format!("layer {l} {proj}"), e.to_string()))?;
        let to = match proj {
            "gate_proj" => names::ffn_gate_exps(l),
            "up_proj" => names::ffn_up_exps(l),
            _ => names::ffn_down_exps(l),
        };
        put(
            to,
            Mapped {
                sources,
                role: Use::Matrix(Plain::Q8_0),
                steps: vec![],
                dims: dims.into_iter().map(|d| d as u64).collect(),
            },
        )?;
    }
    if let Some((&first, &(name, shape))) = ple.first_key_value() {
        if ple.len() != g.ple_parts
            || first != 0
            || ple.keys().next_back() != Some(&(g.ple_parts - 1))
        {
            return Err(refuse(
                name,
                format!("the PLE table has {} of {} parts", ple.len(), g.ple_parts),
            ));
        }
        let &[_, cols] = shape else {
            return Err(refuse(name, format!("{shape:?} is not [rows, cols]")));
        };
        let mut rows = 0usize;
        for &(n, s) in ple.values() {
            match *s {
                [r, c] if c == cols => {
                    rows = rows
                        .checked_add(r)
                        .ok_or_else(|| refuse(n, "PLE row count overflows usize"))?
                }
                _ => {
                    return Err(refuse(
                        n,
                        format!("{s:?}; the first part's rows are {cols} wide"),
                    ));
                }
            }
        }
        put(
            names::per_layer_token_embd(),
            Mapped {
                sources: ple.values().map(|(n, _)| n.to_string()).collect(),
                role: Use::Rows,
                steps: vec![],
                dims: vec![cols as u64, rows as u64],
            },
        )?;
    }
    Ok(out)
}

/// The qwen4exp tensors of `rest`, a tensor of layer `l` other than a routed expert or the PLE
/// table.
fn layer(
    g: &Geometry,
    l: usize,
    rest: &str,
    name: &str,
    shape: &[usize],
) -> Result<Vec<(String, Mapped)>, Error> {
    let dims: Vec<u64> = shape.iter().rev().map(|&d| d as u64).collect();
    let mapped = |plain: Plain, steps: Vec<Step>, dims: Vec<u64>| Mapped {
        sources: vec![name.to_string()],
        role: Use::Matrix(plain),
        steps,
        dims,
    };
    let one = |to: String, plain: Plain, steps: Vec<Step>| {
        Ok(vec![(to, mapped(plain, steps, dims.clone()))])
    };
    let qk = 2 * g.k_heads * g.head_k;
    // `[channels, 1, kernel]` taps as `[kernel, channels]`.
    let taps = || -> Result<Vec<u64>, Error> {
        match *shape {
            [c, 1, k] => Ok(vec![k as u64, c as u64]),
            _ => Err(refuse(
                name,
                format!("conv taps {shape:?} are not [channels, 1, kernel]"),
            )),
        }
    };
    let (sub, hc) = if let Some(r) = rest.strip_prefix("attn_hyper_connection.") {
        (Some(Sub::Attn), r)
    } else if let Some(r) = rest.strip_prefix("mlp_hyper_connection.") {
        (Some(Sub::Ffn), r)
    } else {
        (None, rest)
    };
    if let Some(sub) = sub {
        return match hc {
            "hc_norm.weight" => one(names::hc_norm(l, sub), Plain::F32, vec![Step::PlusOne]),
            "input_mix_weight_down.weight" => one(names::hc_down(l, sub), Plain::Q8_0, vec![]),
            "input_mix_weight_up.weight" => one(names::hc_up(l, sub), Plain::Q8_0, vec![]),
            "block_inject_weight.weight" => one(names::hc_inject(l, sub), Plain::F32, vec![]),
            _ => Err(refuse(name, "not a hyper-connection tensor")),
        };
    }
    let v = |from: usize, head: usize| vec![Step::VRows { from, head }];
    match rest {
        "linear_attn.in_proj_qkv.weight" => one(names::attn_qkv(l), Plain::Q8_0, v(qk, g.head_v)),
        "linear_attn.in_proj_z.weight" => one(names::attn_gate(l), Plain::Q8_0, v(0, g.head_v)),
        "linear_attn.in_proj_a.weight" => one(names::ssm_alpha(l), Plain::F32, v(0, 1)),
        "linear_attn.in_proj_b.weight" => one(names::ssm_beta(l), Plain::F32, v(0, 1)),
        "linear_attn.A_log" => one(
            names::ssm_a(l),
            Plain::F32,
            vec![Step::VCols { head: 1 }, Step::NegExp],
        ),
        "linear_attn.dt_bias" => one(
            names::ssm_dt_bias(l),
            Plain::F32,
            vec![Step::VCols { head: 1 }],
        ),
        "linear_attn.conv1d.weight" => Ok(vec![(
            names::ssm_conv1d(l),
            mapped(Plain::F32, v(qk, g.head_v), taps()?),
        )]),
        "linear_attn.norm.weight" => one(names::ssm_norm(l), Plain::F32, vec![]),
        "linear_attn.out_proj.weight" => one(
            names::ssm_out(l),
            Plain::Q8_0,
            vec![Step::VCols { head: g.head_v }],
        ),
        "self_attn.q_proj.weight" => one(names::attn_q(l), Plain::Q8_0, vec![]),
        "self_attn.k_proj.weight" => one(names::attn_k(l), Plain::Q8_0, vec![]),
        "self_attn.v_proj.weight" => one(names::attn_v(l), Plain::Q8_0, vec![]),
        "self_attn.o_proj.weight" => one(names::attn_output(l), Plain::Q8_0, vec![]),
        "self_attn.q_norm.weight" => one(names::attn_q_norm(l), Plain::F32, vec![Step::PlusOne]),
        "self_attn.k_norm.weight" => one(names::attn_k_norm(l), Plain::F32, vec![Step::PlusOne]),
        "self_attn.indexer.q_layernorm.weight" => {
            one(names::indexer_q_norm(l), Plain::F32, vec![Step::PlusOne])
        }
        "self_attn.indexer.k_layernorm.weight" => {
            one(names::indexer_k_norm(l), Plain::F32, vec![Step::PlusOne])
        }
        "self_attn.indexer.index_qk_proj.weight" => {
            let &[rows, cols] = shape else {
                return Err(refuse(name, format!("{shape:?} is not 2-D")));
            };
            if rows <= g.indexer_q {
                return Err(refuse(
                    name,
                    format!(
                        "{rows} rows hold no key rows after {} query rows",
                        g.indexer_q
                    ),
                ));
            }
            let part = |rows: Range<usize>| {
                let n = rows.len() as u64;
                mapped(Plain::Bf16, vec![Step::Rows(rows)], vec![cols as u64, n])
            };
            Ok(vec![
                (names::indexer_q_proj(l), part(0..g.indexer_q)),
                (names::indexer_k_proj(l), part(g.indexer_q..rows)),
            ])
        }
        "mlp.gate.weight" => one(names::ffn_gate_inp(l), Plain::F32, vec![]),
        "mlp.shared_expert_gate.weight" => match *shape {
            [1, n] => Ok(vec![(
                names::ffn_gate_inp_shexp(l),
                mapped(Plain::F32, vec![], vec![n as u64]),
            )]),
            _ => Err(refuse(name, format!("{shape:?} is not one gate row"))),
        },
        "mlp.shared_expert.gate_proj.weight" => one(names::ffn_gate_shexp(l), Plain::Q8_0, vec![]),
        "mlp.shared_expert.up_proj.weight" => one(names::ffn_up_shexp(l), Plain::Q8_0, vec![]),
        "mlp.shared_expert.down_proj.weight" => one(names::ffn_down_shexp(l), Plain::Q8_0, vec![]),
        "ple.key_proj.weight" => one(names::ple_key(l), Plain::Q8_0, vec![]),
        "ple.value_proj.weight" => one(names::ple_value(l), Plain::Q8_0, vec![]),
        "ple.norm_key.weight" => one(names::ple_norm_key(l), Plain::F32, vec![Step::PlusOne]),
        "ple.norm_query.weight" => one(names::ple_norm_query(l), Plain::F32, vec![Step::PlusOne]),
        "ple.norm_conv.weight" => one(names::ple_norm_conv(l), Plain::F32, vec![Step::PlusOne]),
        "ple.conv1d.weight" => Ok(vec![(
            names::ple_conv1d(l),
            mapped(Plain::F32, vec![], taps()?),
        )]),
        _ => Err(refuse(name, "not a Qwen3.8 tensor")),
    }
}

impl Step {
    /// `self` on `t`, a normalized tensor of `g`'s; `name` is the qwen4exp tensor's.
    pub fn apply(&self, g: &Geometry, name: &str, t: &mut Normal) -> Result<(), Error> {
        let width = t
            .dims
            .first()
            .copied()
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| refuse(name, "invalid tensor width"))?;
        let rows = t
            .dims
            .iter()
            .skip(1)
            .try_fold(1usize, |rows, &n| {
                usize::try_from(n)
                    .ok()
                    .filter(|&n| n > 0)
                    .and_then(|n| rows.checked_mul(n))
            })
            .ok_or_else(|| refuse(name, "tensor row count is zero or overflows usize"))?;
        if !t.bytes.len().is_multiple_of(rows) {
            return Err(refuse(name, "tensor bytes do not form whole rows"));
        }
        let row = t.bytes.len() / rows.max(1);
        let f32s = |t: &Normal| {
            if t.type_id == gguf::quant::GgmlType::F32.as_u32() {
                Ok(())
            } else {
                Err(refuse(
                    name,
                    format!("{self:?} needs F32, not type {}", t.type_id),
                ))
            }
        };
        // Tiled place of grouped V head `h` of `n` heads.
        let (k, per) = (g.k_heads, g.v_heads / g.k_heads);
        let tiled = |h: usize| (h % per) * k + h / per;
        let mut moved = vec![0u8; t.bytes.len()];
        match *self {
            Step::VRows { from, head } => {
                if rows < from || Some(rows - from) != g.v_heads.checked_mul(head) {
                    return Err(refuse(
                        name,
                        format!(
                            "{rows} rows: not {from} then {} V heads of {head}",
                            g.v_heads
                        ),
                    ));
                }
                moved[..from * row].copy_from_slice(&t.bytes[..from * row]);
                for h in 0..g.v_heads {
                    let span = head * row;
                    let src = &t.bytes[from * row + h * span..][..span];
                    moved[from * row + tiled(h) * span..][..span].copy_from_slice(src);
                }
            }
            Step::VCols { head } => {
                let block = gguf::quant::GgmlType::from_u32(t.type_id)
                    .blck_size()
                    .ok_or_else(|| refuse(name, format!("unsized tensor type {}", t.type_id)))?;
                if width == 0
                    || Some(width) != g.v_heads.checked_mul(head)
                    || !(head as u64).is_multiple_of(block)
                    || !row.is_multiple_of(g.v_heads)
                {
                    return Err(refuse(
                        name,
                        format!(
                            "{width} columns are not {} V heads of {head} whole blocks",
                            g.v_heads
                        ),
                    ));
                }
                let span = row / g.v_heads;
                for (src, dst) in t.bytes.chunks_exact(row).zip(moved.chunks_exact_mut(row)) {
                    for h in 0..g.v_heads {
                        dst[tiled(h) * span..][..span].copy_from_slice(&src[h * span..][..span]);
                    }
                }
            }
            Step::Rows(ref keep) => {
                if keep.end > rows {
                    return Err(refuse(name, format!("rows {keep:?} of {rows}")));
                }
                t.bytes = t.bytes[keep.start * row..keep.end * row].to_vec();
                t.dims = vec![width as u64, keep.len() as u64];
                return Ok(());
            }
            Step::PlusOne | Step::NegExp => {
                f32s(t)?;
                let neg = matches!(self, Step::NegExp);
                for (src, dst) in t
                    .bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(moved.as_chunks_mut::<4>().0)
                {
                    let x = f32::from_le_bytes(*src);
                    if !x.is_finite() {
                        return Err(refuse(name, format!("{self:?}: non-finite input {x}")));
                    }
                    let y = if neg {
                        (-f64::from(x).exp()) as f32
                    } else {
                        x + 1.0
                    };
                    if !y.is_finite() || (neg && y == 0.0) {
                        return Err(refuse(
                            name,
                            format!("{self:?}: result {y} is outside its F32 range"),
                        ));
                    }
                    *dst = y.to_le_bytes();
                }
            }
        }
        t.bytes = moved;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use hf::quant_config::{Config, plan};
    use hf::safetensors::Dtype;

    use super::*;

    fn fixture(name: &str) -> Vec<u8> {
        let path = format!("{}/../hf/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
        let raw = std::fs::read(&path).unwrap();
        if !name.ends_with(".gz") {
            return raw;
        }
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(raw.as_slice())
            .read_to_end(&mut out)
            .unwrap();
        out
    }

    fn json(name: &str) -> Json {
        serde_json::from_slice(&fixture(name)).unwrap()
    }

    fn config() -> Json {
        json("quant-configs/nvidia__Qwen3.8-Flash-Next-NVFP4__config.json.gz")
    }

    /// Layers 0, 1 (the PLE site) and 3 (attention) and every unlayered text tensor of
    /// nvidia/Qwen3.8-Flash-Next-NVFP4, placed by its quantization config, map to exactly the
    /// names and ggml dims unsloth's UD-Q4_K_XL holds for them, and every tensor that file keeps
    /// in Q8_0, F32 or BF16 gets that type from the table; the routed stacks are NVFP4 with one
    /// scale an expert and the PLE table F8 rows.
    #[test]
    fn layers_map_to_unsloth_s_names_dims_and_types() {
        let header =
            json("checkpoint-headers/qwen3.8-flash-next-nvfp4-layers-0-1-3.tensors.json.gz");
        let entries: Vec<(String, Dtype, Vec<usize>)> = header
            .as_object()
            .unwrap()
            .iter()
            .map(|(n, e)| {
                let shape = e["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap() as usize)
                    .collect();
                (
                    n.clone(),
                    Dtype::of(e["dtype"].as_str().unwrap()).unwrap(),
                    shape,
                )
            })
            .collect();
        let hq = json("quant-configs/nvidia__Qwen3.8-Flash-Next-NVFP4__hf_quant_config.json.gz");
        let p = plan(
            &Config::parse(&config(), Some(&hq)).unwrap(),
            entries
                .iter()
                .map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
        );
        let logical: Vec<(&str, Vec<usize>)> = entries
            .iter()
            .filter_map(|(n, _, s)| match p.placed.get(n) {
                Some(pl) => Some((n.as_str(), pl.format.logical(s))),
                None => p.refused.contains_key(n).then(|| (n.as_str(), s.clone())),
            })
            .collect();
        let g = Checkpoint::parse(&config()).unwrap().geometry;
        let mapped = map(&g, logical.iter().map(|(n, s)| (*n, s.as_slice()))).unwrap();
        let unsloth: Vec<(String, Vec<u64>, u32)> = serde_json::from_slice(&fixture(
            "checkpoint-headers/unsloth-qwen3.8-flash-next-ud-q4_k_xl-layers-0-1-3.tensors.json.gz",
        ))
        .unwrap();
        let mut names: Vec<&str> = mapped.keys().map(String::as_str).collect();
        names.sort_unstable();
        let want: Vec<&str> = unsloth.iter().map(|(n, _, _)| n.as_str()).collect();
        assert_eq!(names, want);
        for (name, dims, ty) in &unsloth {
            let m = &mapped[name];
            assert_eq!(&m.dims, dims, "{name}");
            let format = modelopt::format_of(&m.placed(&p).unwrap()).unwrap();
            let ours = modelopt::type_of(format, m.role).unwrap();
            let expect = match ty {
                0 | 8 | 30 => *ty,
                _ if name.ends_with("_exps.weight") => 40,
                _ => gguf::GgmlType::F8E4M3_ROWS.as_u32(),
            };
            assert_eq!(ours, expect, "{name}");
        }
        assert_eq!(mapped[&names::attn_q_norm(3)].steps, [Step::PlusOne]);
        assert_eq!(
            mapped[&names::ssm_a(0)].steps,
            [Step::VCols { head: 1 }, Step::NegExp]
        );
        assert_eq!(
            mapped[&names::attn_gate(0)].steps,
            [Step::VRows {
                from: 0,
                head: g.head_v
            }]
        );
        let stack = &mapped["blk.3.ffn_down_exps.weight"];
        assert_eq!(stack.sources.len(), 512);
        assert_eq!(
            stack.sources[7],
            "model.language_model.layers.3.mlp.experts.7.down_proj.weight"
        );
        assert_eq!(mapped["per_layer_token_embd.weight"].sources.len(), 128);
    }

    fn constant(part: &str) -> Result<Vec<i64>, Error> {
        let want =
            json("checkpoint-headers/unsloth-qwen3.8-flash-next-ud-q4_k_xl-qwen4exp-keys.json");
        let key = match part.rsplit('.').next().unwrap() {
            "layer_multipliers" => "qwen4exp.ple.layer_multipliers",
            "ngram_heads_offsets" => "qwen4exp.ple.head_offsets",
            "ngram_heads_vocab_sizes" => "qwen4exp.ple.head_vocab_sizes",
            _ => panic!("{part}"),
        };
        Ok(want[key][1][1]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect())
    }

    #[test]
    fn checkpoint_input_refusals() {
        use serde_json::json;
        let dir = std::env::temp_dir().join(format!("qwen-prepare-inputs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut valid = config();
        valid["text_config"]["vocab_size"] = json!(8);
        valid["text_config"]["bos_token_id"] = json!(2);
        valid["text_config"]["eos_token_id"] = json!(2);
        valid["image_token_id"] = json!(3);
        let tokenizer = json!({"normalizer":{"type":"NFC"},"pre_tokenizer":{"type":"Sequence","pretokenizers":[
            {"type":"Split","pattern":{"Regex":r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"},"behavior":"Isolated","invert":false},
            {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
            "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "model":{"type":"BPE","vocab":{"A":0,"a":1,"Aa":7},"merges":[["A","a"]]},
            "added_tokens":[{"id":2,"content":"</s>","special":true}]});
        let files = [
            ("config.json", valid.clone()),
            ("tokenizer.json", tokenizer),
            ("tokenizer_config.json", json!({})),
            ("generation_config.json", json!({})),
            (
                "header",
                json!({"embed":[8,2560],"output":[8,2560],"indexer":[640,2560],
                    "ple":[160,400000000],"ple_layer":1}),
            ),
            (
                "constants",
                json!({"layer_multipliers":constant("layer_multipliers").unwrap(),
                "ngram_heads_offsets":constant("ngram_heads_offsets").unwrap(),
                "ngram_heads_vocab_sizes":constant("ngram_heads_vocab_sizes").unwrap()}),
            ),
        ];
        let read = |files: &BTreeMap<&str, Json>| -> Result<(), Error> {
            for file in [
                "tokenizer.json",
                "tokenizer_config.json",
                "generation_config.json",
            ] {
                std::fs::write(dir.join(file), files[file].to_string()).unwrap();
            }
            let input = Checkpoint::parse(&files["config.json"])?;
            let shape = |key: &str| {
                files["header"][key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>()
            };
            let (embed, output) = (shape("embed"), shape("output"));
            let ple_name = format!(
                "model.language_model.layers.{}.ple.key_proj.weight",
                files["header"]["ple_layer"]
            );
            let indexer = shape("indexer");
            input.map([
                ("model.language_model.embed_tokens.weight", embed.as_slice()),
                ("lm_head.weight", output.as_slice()),
                (
                    "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
                    indexer.as_slice(),
                ),
                (ple_name.as_str(), &[2560, 2560]),
            ])?;
            input.keys(
                &shape("ple").iter().map(|&n| n as u64).collect::<Vec<_>>(),
                |part| {
                    let part = part.rsplit('.').next().unwrap();
                    Ok(files["constants"][part]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_i64().unwrap())
                        .collect())
                },
            )?;
            hf::vocab::keys(&dir, &input.vocab).map_err(|e| Error::Config(e.to_string()))?;
            Ok(())
        };
        let files: BTreeMap<&str, Json> = files.into_iter().collect();
        read(&files).unwrap();
        let mut cases: Vec<(&str, &str, Json, &str)> = serde_json::from_str(r#"[
            ["config.json","/architectures",["Qwen4ExpUnknown"],"architectures"],
            ["config.json","/architectures",["Qwen4ExpForConditionalGeneration",7],"architectures"],
            ["config.json","/text_config",7,"text_config"],
            ["config.json","/text_config/rope_parameters",7,"rope_parameters"],
            ["config.json","/text_config/rope_parameters/rope_theta",1e+300,"rope_theta"],
            ["config.json","/text_config/rope_parameters/partial_rotary_factor","0.25","partial_rotary_factor"],
            ["config.json","/text_config/rope_parameters/partial_rotary_factor",2,"partial_rotary_factor"],
            ["config.json","/text_config/rope_parameters/partial_rotary_factor",0.251,"partial_rotary_factor"],
            ["config.json","/text_config/rope_parameters/mrope_section",[11,11,10,0,1],"mrope_section"],
            ["config.json","/text_config/rope_parameters/mrope_section",[-1,11,10],"mrope_section"],
            ["config.json","/text_config/rope_parameters/mrope_section",[11,"11",10],"mrope_section"],
            ["config.json","/text_config/rope_parameters/mrope_section",[11,11,9],"mrope_section"],
            ["header","/indexer",[768,2560],"indexer_head_dim"],
            ["config.json","/text_config/ple_layer_ids","2","ple_layer_ids"],
            ["config.json","/text_config/ple_layer_ids",[2,"6"],"ple_layer_ids"],
            ["config.json","/text_config/ple_layer_ids",[0],"ple_layer_ids"],
            ["config.json","/text_config/ple_layer_ids",[49],"ple_layer_ids"],
            ["config.json","/text_config/ple_layer_ids",[2,2],"ple_layer_ids"],
            ["config.json","/text_config/ple_layer_ids",[4],"ple_layer_ids"],
            ["config.json","/text_config/eos_token_id",[2,"3"],"eos_token_id"],
            ["config.json","/text_config/eos_token_id",[],"eos_token_id"],
            ["config.json","/image_token_id",8,"image_token_id"],
            ["config.json","/image_token_id","3","image_token_id"],
            ["config.json","/text_config/linear_num_value_heads",47,"linear_num_value_heads"],
            ["config.json","/text_config/linear_num_value_heads",4294967295,"linear_value_head_dim"],
            ["config.json","/text_config/indexer_n_heads",4294967295,"indexer_n_heads"],
            ["config.json","/text_config/num_experts_per_tok",513,"num_experts_per_tok"],
            ["config.json","/text_config/num_attention_heads",23,"num_attention_heads"],
            ["config.json","/text_config/hc_count",1,"hc_count"],
            ["config.json","/text_config/indexer_compress_ratio",3,"indexer_compress_ratio"],
            ["config.json","/text_config/ngram_size",9,"ngram_size"],
            ["config.json","/text_config/heads_per_ngram",65,"heads_per_ngram"],
            ["header","/embed",[9,2560],"vocab_size"],
            ["header","/output",[8,2561],"hidden_size"],
            ["header","/ple",[4294967296,400000000],"row width"],
            ["header","/ple",[160,1],"ngram_heads_vocab_sizes"],
            ["constants","/layer_multipliers",[1],"layer_multipliers"],
            ["constants","/ngram_heads_offsets",[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],"ngram_heads_vocab_sizes"],
            ["constants","/ngram_heads_vocab_sizes",[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"ngram_heads_vocab_sizes"],
            ["tokenizer.json","/normalizer",{"type":"Lowercase"},"normalizer"],
            ["tokenizer.json","/normalizer",{"type":"Sequence","normalizers":[{"type":"Lowercase"}]},"normalizer"],
            ["tokenizer.json","/normalizer",7,"normalizer"],
            ["tokenizer.json","/pre_tokenizer",null,"pre_tokenizer"],
            ["tokenizer.json","/pre_tokenizer",{"type":"Whitespace"},"pre_tokenizer"],
            ["tokenizer.json","/pre_tokenizer/pretokenizers/0/type","Unknown","pre_tokenizer"],
            ["tokenizer.json","/pre_tokenizer/pretokenizers/0/behavior","MergedWithNext","pre_tokenizer"],
            ["tokenizer.json","/pre_tokenizer/pretokenizers/0/invert",true,"pre_tokenizer"],
            ["tokenizer.json","/pre_tokenizer/pretokenizers/0/pattern",{"Regex":"different"},"pre_tokenizer"],
            ["tokenizer.json","/pre_tokenizer/pretokenizers/1/use_regex",true,"pre_tokenizer"],
            ["tokenizer.json","/pre_tokenizer/pretokenizers/1/add_prefix_space",true,"pre_tokenizer"],
            ["tokenizer.json","/pre_tokenizer",{"type":"Sequence","pretokenizers":[{"type":"Unknown"}]},"pre_tokenizer"],
            ["tokenizer.json","/decoder",null,"decoder"],
            ["tokenizer.json","/decoder",7,"decoder"],
            ["tokenizer.json","/decoder/type","Unknown","decoder"],
            ["tokenizer.json","/decoder",{"type":"Sequence","decoders":[{"type":"Unknown"}]},"decoder"],
            ["tokenizer.json","/post_processor",7,"post_processor"],
            ["tokenizer.json","/post_processor/type","Unknown","post_processor"],
            ["tokenizer.json","/post_processor",{"type":"Sequence","processors":[{"type":"Unknown"}]},"post_processor"],
            ["tokenizer.json","/model/type","Unigram","model.type"],
            ["tokenizer.json","/model/vocab",{"a":0,"b":8},"id"],
            ["tokenizer.json","/model/vocab",{"a":0,"b":0},"id"],
            ["tokenizer.json","/model/vocab",{"a":-1},"id"],
            ["tokenizer.json","/model/vocab",{"a":4294967296},"id"],
            ["tokenizer.json","/model/vocab",{"A":0,"a":1,"[PAD3]":4},"[PAD3]"],
            ["tokenizer.json","/added_tokens",[{"id":8,"content":"z"}],"id"],
            ["tokenizer.json","/added_tokens",[{"id":0,"content":"z"}],"id"],
            ["tokenizer.json","/added_tokens",[{"id":2,"content":"z","normalized":false},{"id":2,"content":"z","normalized":false}],"id"],
            ["tokenizer.json","/added_tokens",[{"id":2,"content":"z","normalized":false},{"id":3,"content":"z","normalized":false}],"id"],
            ["tokenizer.json","/added_tokens",[{"id":2,"content":"z","special":false}],"normalized"],
            ["tokenizer.json","/added_tokens",[{"id":2,"content":"z"}],"normalized"],
            ["tokenizer.json","/added_tokens",{"id":2},"added_tokens"],
            ["tokenizer.json","/added_tokens",[{"id":2,"content":7}],"content"],
            ["tokenizer.json","/added_tokens",[{"id":2,"content":""}],"content"],
            ["tokenizer.json","/added_tokens",[{"id":2,"content":"z","special":"yes"}],"special"],
            ["tokenizer.json","/model/merges",[["a","b","c"]],"merge"],
            ["tokenizer_config.json","/eos_token",7,"eos_token"],
            ["tokenizer_config.json","/eos_token",{"id":2},"eos_token"],
            ["tokenizer_config.json","/eos_token","unknown","eos_token"],
            ["tokenizer_config.json","/add_bos_token","yes","add_bos_token"],
            ["tokenizer_config.json","/chat_template",[{"template":"t"}],"chat_template"],
            ["generation_config.json","/pad_token_id",[3,"4"],"pad_token_id"],
            ["generation_config.json","/pad_token_id",[3,8],"pad_token_id"],
            ["config.json","/text_config/eos_token_id",4294967296,"eos_token_id"],
            ["config.json","/text_config/eos_token_id",[2,8],"eos_token_id"],
            ["config.json","/image_token_id",4294967296,"image_token_id"],
            ["header","/ple_layer",2,"ple_layer_ids"],
            ["header","/embed",[0,2560],"shape"],
            ["header","/embed",[18446744073709551615,2560],"shape"],
            ["tokenizer.json","/added_tokens",[{"id":3,"content":"a"}],"id"]
        ]"#).unwrap();
        let mut missed = vec![];
        cases.extend([
            (
                "tokenizer_config.json",
                "/added_tokens_decoder",
                json!({"2":{"content":"</s>","special":false}}),
                "normalized",
            ),
            (
                "tokenizer_config.json",
                "/eos_token",
                json!({"content":"</s>","special":false}),
                "normalized",
            ),
            (
                "tokenizer_config.json",
                "/eos_token",
                json!({"content":"z","lstrip":true}),
                "lstrip",
            ),
            (
                "tokenizer_config.json",
                "/added_tokens_decoder",
                json!([]),
                "added_tokens_decoder",
            ),
            (
                "tokenizer_config.json",
                "/added_tokens_decoder",
                json!({"2":{"content":"different"}}),
                "added_tokens_decoder",
            ),
            (
                "config.json",
                "/text_config/layer_types",
                {
                    let mut a = valid["text_config"]["layer_types"]
                        .as_array()
                        .unwrap()
                        .clone();
                    a.insert(0, json!(7));
                    json!(a)
                },
                "layer_types",
            ),
            (
                "config.json",
                "/text_config/layer_types",
                {
                    let mut a = valid["text_config"]["layer_types"].clone();
                    a[0] = json!(7);
                    a
                },
                "layer_types",
            ),
            (
                "config.json",
                "/text_config/layer_types",
                {
                    let mut a = valid["text_config"]["layer_types"].clone();
                    a[0] = json!("unknown");
                    a
                },
                "layer_types",
            ),
            (
                "config.json",
                "/text_config/layer_types",
                {
                    let mut a = valid["text_config"]["layer_types"].clone();
                    a[0] = json!("full_attention");
                    a
                },
                "layer_types",
            ),
        ]);
        for key in [
            "num_hidden_layers",
            "max_position_embeddings",
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "num_experts",
            "num_experts_per_tok",
            "head_dim",
            "moe_intermediate_size",
            "shared_expert_intermediate_size",
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_value_head_dim",
            "full_attention_interval",
            "hc_count",
            "hc_lowrank",
            "indexer_n_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
            "split_ngram_parts",
            "ngram_size",
            "heads_per_ngram",
            "ple_conv_kernel_size",
            "vocab_size",
        ] {
            let path = format!("/text_config/{key}");
            let path = path.as_str();
            for value in [
                json!(0),
                json!(-1),
                json!(4294967296u64),
                json!("1"),
                Json::Null,
            ] {
                let mut input = files.clone();
                let c = input.get_mut("config.json").unwrap();
                if value.is_null() {
                    c["text_config"].as_object_mut().unwrap().remove(key);
                } else {
                    *c.pointer_mut(path).unwrap() = value.clone();
                }
                let error = read(&input).expect_err(&format!("{path}={value}"));
                assert!(error.to_string().contains(key), "{path}={value}: {error}");
            }
        }
        for (file, path, value, what) in cases.drain(..) {
            let mut input = files.clone();
            let (parent, key) = path.rsplit_once('/').unwrap();
            input.get_mut(file).unwrap().pointer_mut(parent).unwrap()[key] = value.clone();
            match read(&input) {
                Err(e) if e.to_string().contains(what) => {
                    println!("refused {file}{path} {value}: {e}")
                }
                other => missed.push(format!("{file}{path} {value}: {other:?}")),
            }
        }
        // A non-special added token keeps its spelling from the checkpoint into serving.
        let mut literal = files.clone();
        let t = literal.get_mut("tokenizer.json").unwrap();
        t["model"]["vocab"] = json!({"A":0,"a":1,"Aa":7,"\u{120}":5,"x":6});
        t["added_tokens"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":4,"content":"\u{2581}x","special":false,"normalized":false}));
        read(&literal).unwrap();
        let keys = hf::vocab::keys(
            &dir,
            &Checkpoint::parse(&literal["config.json"]).unwrap().vocab,
        )
        .unwrap()
        .keys;
        let path = dir.join("literal.gguf");
        let file = std::fs::File::create(&path).unwrap();
        let layout = gguf::write::Layout::new(&keys, Vec::new()).unwrap();
        gguf::write::Writer::new(file, layout)
            .and_then(gguf::write::Writer::finish)
            .unwrap();
        let served = tokenizer::Tokenizer::from_gguf(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let got = (
            served.text(4),
            served.encode("\u{2581}x", false, false),
            served.encode(" x", false, false),
            served.decode(&[4]),
        );
        if got != (Some("\u{2581}x"), vec![4], vec![5, 6], "\u{2581}x".into()) {
            missed.push(format!("added token \u{2581}x served as {got:?}"));
        }
        for file in [
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        ] {
            std::fs::remove_file(dir.join(file)).unwrap();
        }
        std::fs::remove_dir(dir).unwrap();
        assert!(
            missed.is_empty(),
            "accepted or unnamed refusals:\n{}",
            missed.join("\n")
        );
    }

    /// The config tables row by row: for every key, one value accepted and one refused by name
    /// (or by the reader that owns it), and a key no table names refused in each object. Both
    /// target checkpoints' `config.json` (NVFP4's is the base) pass as published.
    #[test]
    fn every_config_key_accepts_its_values_and_refuses_others() {
        use serde_json::json;
        let hq = json("quant-configs/nvidia__Qwen3.8-Flash-Next-NVFP4__hf_quant_config.json.gz");
        let run = |c: &Json| -> Result<(), String> {
            Checkpoint::parse(c).map_err(|e| e.to_string())?;
            Config::parse(c, Some(&hq))
                .map(drop)
                .map_err(|e| e.to_string())
        };
        let fp4_local = json("quant-configs/nvidia__Qwen3.8-Flash-Next-FP4-Local__config.json.gz");
        Checkpoint::parse(&fp4_local).unwrap();
        let base = config();
        let objects = [
            ("", ROOT),
            ("/text_config", TEXT_CONFIG),
            ("/text_config/rope_parameters", ROPE),
        ];
        // object, key, a value accepted ("$" keeps the checkpoint's), a value refused, and what
        // the refusal names.
        let mut rows: Vec<(String, String, Json, Json, String)> = serde_json::from_str(r#"[
            ["", "architectures", "$", ["Qwen4ExpUnknown"], "architectures"],
            ["", "model_type", "qwen4_exp", "qwen3_next", "model_type"],
            ["", "text_config", "$", 7, "text_config"],
            ["", "quantization_config", "$", {"quant_method":"quark"}, "Quark"],
            ["", "tie_word_embeddings", false, true, "tie_word_embeddings"],
            ["", "image_token_id", 248056, 248320, "image_token_id"],
            ["", "dtype", "bfloat16", 7, "dtype"],
            ["", "transformers_version", "5.8.0.dev0", 5, "transformers_version"],
            ["", "vision_config", {}, 7, "vision_config"],
            ["", "video_token_id", 248057, "x", "video_token_id"],
            ["", "vision_start_token_id", 248053, "x", "vision_start_token_id"],
            ["", "vision_end_token_id", 248054, "x", "vision_end_token_id"],
            ["", "language_model_only", true, "x", "language_model_only"],
            ["/text_config", "rms_norm_eps", 1e-5, "x", "rms_norm_eps"],
            ["/text_config", "layer_types", "$", "x", "layer_types"],
            ["/text_config", "rope_parameters", "$", 7, "rope_parameters"],
            ["/text_config", "partial_rotary_factor", 0.25, 0.5, "partial_rotary_factor"],
            ["/text_config", "ple_layer_ids", [2], [0], "ple_layer_ids"],
            ["/text_config", "ngram_size", 3, 9, "ngram_size"],
            ["/text_config", "heads_per_ngram", 8, 65, "heads_per_ngram"],
            ["/text_config", "ple_embed_dim", 2560, 2561, "ple_embed_dim"],
            ["/text_config", "bos_token_id", 248044, -1, "bos_token_id"],
            ["/text_config", "eos_token_id", [248046, 248044], [], "eos_token_id"],
            ["/text_config", "pad_token_id", null, "x", "pad_token_id"],
            ["/text_config", "unk_token_id", null, "x", "unk_token_id"],
            ["/text_config", "sep_token_id", 248044, 248320, "sep_token_id"],
            ["/text_config", "model_type", "qwen4_exp_text", "qwen3_next_text", "model_type"],
            ["/text_config", "hidden_act", "silu", "gelu", "hidden_act"],
            ["/text_config", "output_gate_type", "sigmoid", "silu", "output_gate_type"],
            ["/text_config", "norm_topk_prob", true, false, "norm_topk_prob"],
            ["/text_config", "attention_bias", false, true, "attention_bias"],
            ["/text_config", "indexer_kv_heads", 1, 2, "indexer_kv_heads"],
            ["/text_config", "tie_word_embeddings", false, true, "tie_word_embeddings"],
            ["/text_config", "mamba_ssm_dtype", "float32", "bfloat16", "mamba_ssm_dtype"],
            ["/text_config", "dtype", "bfloat16", 7, "dtype"],
            ["/text_config", "attention_dropout", 0.1, "x", "attention_dropout"],
            ["/text_config", "initializer_range", 0.02, "x", "initializer_range"],
            ["/text_config", "router_aux_loss_coef", 0.001, "x", "router_aux_loss_coef"],
            ["/text_config", "output_router_logits", true, "x", "output_router_logits"],
            ["/text_config", "use_cache", false, "x", "use_cache"],
            ["/text_config", "seed", 1234, "x", "seed"],
            ["/text_config", "number_of_conv_states", 3, "x", "number_of_conv_states"],
            ["/text_config", "make_ngram_vocab_size_divisible_by", 128, "x", "make_ngram_vocab_size_divisible_by"],
            ["/text_config", "ngram_vocab_size_base", 20000000, "x", "ngram_vocab_size_base"],
            ["/text_config", "ple_embedding_dtype", "nvfp4", 4, "ple_embedding_dtype"],
            ["/text_config", "mtp", "$", 1, "mtp"],
            ["/text_config", "mtp_num_hidden_layers", 1, "x", "mtp_num_hidden_layers"],
            ["/text_config", "mtp_use_dedicated_embeddings", false, "x", "mtp_use_dedicated_embeddings"],
            ["/text_config/rope_parameters", "rope_type", "default", "yarn", "rope_type"],
            ["/text_config/rope_parameters", "mrope_interleaved", true, false, "mrope_interleaved"],
            ["/text_config/rope_parameters", "mrope_section", [11, 11, 10], [11, 11, 9], "mrope_section"],
            ["/text_config/rope_parameters", "rope_theta", 1e6, 1e300, "rope_theta"],
            ["/text_config/rope_parameters", "partial_rotary_factor", 0.25, 2, "partial_rotary_factor"]
        ]"#)
        .unwrap();
        // The counts: their own value, and zero.
        for (key, rule) in TEXT_CONFIG {
            if matches!(rule, Read)
                && base["text_config"][key].is_u64()
                && !rows.iter().any(|r| r.1 == *key)
            {
                rows.push((
                    "/text_config".into(),
                    key.to_string(),
                    json!("$"),
                    json!(0),
                    key.to_string(),
                ));
            }
        }
        let mut missed = vec![];
        for (at, table) in objects {
            for (key, _) in table {
                if !rows
                    .iter()
                    .any(|r| (r.0.as_str(), r.1.as_str()) == (at, *key))
                {
                    missed.push(format!("{at} {key}: no row"));
                }
            }
            rows.push((at.into(), "zzz".into(), json!("$"), json!(1), "zzz".into()));
        }
        let with = |at: &str, key: &str, value: &Json| {
            let mut c = base.clone();
            if value != "$" {
                c.pointer_mut(at).unwrap()[key] = value.clone();
            }
            c
        };
        for (at, key, allowed, refused, what) in rows {
            if let Err(e) = run(&with(&at, &key, &allowed)) {
                missed.push(format!("{at} {key}={allowed}: refused: {e}"));
            }
            match run(&with(&at, &key, &refused)) {
                Err(e) if e.contains(&what) => {}
                other => missed.push(format!("{at} {key}={refused}: {other:?}")),
            }
        }
        let mut required = base.clone();
        required["text_config"]["rope_parameters"]
            .as_object_mut()
            .unwrap()
            .remove("mrope_interleaved");
        if !run(&required).is_err_and(|e| e.contains("mrope_interleaved is missing")) {
            missed.push("absent mrope_interleaved accepted".into());
        }
        assert!(missed.is_empty(), "{}", missed.join("\n"));
    }

    /// The keys from the checkpoint's `config.json` equal unsloth's `qwen4exp.*` keys, the PLE
    /// constants as stored.
    #[test]
    fn keys_equal_unsloth_s() {
        let want =
            json("checkpoint-headers/unsloth-qwen3.8-flash-next-ud-q4_k_xl-qwen4exp-keys.json");
        let keys = Checkpoint::parse(&config())
            .unwrap()
            .keys(&[160, 400000000], constant)
            .unwrap();
        let as_json = |v: &Value| -> Json {
            match v {
                Value::U32(x) => serde_json::json!(["U32", x]),
                Value::F32(x) => serde_json::json!(["F32", f64::from(*x)]),
                Value::Array(a) => {
                    let (tag, items): (&str, Vec<Json>) = match a.first() {
                        Some(Value::I32(_)) => (
                            "I32",
                            a.iter()
                                .map(|x| match x {
                                    Value::I32(i) => serde_json::json!(i),
                                    _ => unreachable!(),
                                })
                                .collect(),
                        ),
                        _ => (
                            "U64",
                            a.iter()
                                .map(|x| match x {
                                    Value::U64(i) => serde_json::json!(i),
                                    _ => unreachable!(),
                                })
                                .collect(),
                        ),
                    };
                    serde_json::json!(["ARR", [tag, items]])
                }
                other => panic!("{other:?}"),
            }
        };
        let got: serde_json::Map<String, Json> =
            keys.iter().map(|(k, v)| (k.clone(), as_json(v))).collect();
        assert_eq!(Json::Object(got), want);
    }

    /// Each step against its definition on a small geometry: two K heads of three V heads each,
    /// so grouped head `i · 3 + j` goes to tiled place `j · 2 + i`.
    #[test]
    fn steps_move_v_heads_rows_and_values() {
        let g = Geometry {
            layers: 1,
            k_heads: 2,
            v_heads: 6,
            head_k: 1,
            head_v: 2,
            indexer_q: 1,
            experts: 1,
            ple_parts: 1,
        };
        let f32s = |dims: Vec<u64>, v: &[f32]| Normal {
            type_id: 0,
            dims,
            bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            scale: vec![],
        };
        let vals = |n: &Normal| -> Vec<f32> {
            n.bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        };
        let tiled = [0, 3, 1, 4, 2, 5];
        // Rows: one plain row, then six V heads of two one-value rows.
        let mut t = f32s(vec![1, 13], &(0..13).map(|i| i as f32).collect::<Vec<_>>());
        Step::VRows { from: 1, head: 2 }
            .apply(&g, "t", &mut t)
            .unwrap();
        let mut want = vec![0.0];
        for &h in &tiled[..6] {
            want.extend([1.0 + 2.0 * h as f32, 2.0 + 2.0 * h as f32]);
        }
        assert_eq!(vals(&t), want);
        // Columns: one row of six heads of two values.
        let mut t = f32s(vec![12], &(0..12).map(|i| i as f32).collect::<Vec<_>>());
        Step::VCols { head: 2 }.apply(&g, "t", &mut t).unwrap();
        let want: Vec<f32> = tiled
            .iter()
            .flat_map(|&h| [2.0 * h as f32, 2.0 * h as f32 + 1.0])
            .collect();
        assert_eq!(vals(&t), want);
        let mut t = f32s(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Step::Rows(1..3).apply(&g, "t", &mut t).unwrap();
        assert_eq!(
            (t.dims.clone(), vals(&t)),
            (vec![2, 2], vec![3.0, 4.0, 5.0, 6.0])
        );
        Step::PlusOne.apply(&g, "t", &mut t).unwrap();
        assert_eq!(vals(&t), [4.0, 5.0, 6.0, 7.0]);
        let mut t = f32s(vec![2], &[0.0, 2.328125]);
        Step::NegExp.apply(&g, "t", &mut t).unwrap();
        assert_eq!(vals(&t), [-1.0, (-2.328125f64.exp()) as f32]);
        let mut missed = vec![];
        for (step, x) in [
            (Step::NegExp, f32::NEG_INFINITY),
            (Step::NegExp, f32::INFINITY),
            (Step::NegExp, f32::NAN),
            (Step::NegExp, 100.0),
            (Step::NegExp, -1000.0),
            (Step::PlusOne, f32::NEG_INFINITY),
            (Step::PlusOne, f32::INFINITY),
            (Step::PlusOne, f32::NAN),
        ] {
            let mut bad = f32s(vec![1], &[x]);
            match step.apply(&g, "bad.weight", &mut bad) {
                Err(e) if e.to_string().contains("bad.weight") => {}
                other => missed.push(format!("{step:?}({x}): {other:?}")),
            }
        }
        assert!(missed.is_empty(), "accepted: {missed:?}");
        let e = Step::VRows { from: 0, head: 2 }
            .apply(&g, "t", &mut t)
            .unwrap_err()
            .to_string();
        assert!(e.contains("1 rows: not 0 then 6 V heads of 2"), "{e}");
        let mut q = Normal {
            type_id: 8,
            dims: vec![32, 1],
            bytes: vec![0; 34],
            scale: vec![],
        };
        let e = Step::PlusOne
            .apply(&g, "q", &mut q)
            .unwrap_err()
            .to_string();
        assert!(e.contains("needs F32, not type 8"), "{e}");
        // Four V heads of 16 columns in one Q8_0 row of two blocks: each head is half a block.
        let g = Geometry {
            k_heads: 2,
            v_heads: 4,
            ..g
        };
        let mut q = Normal {
            type_id: 8,
            dims: vec![64, 1],
            bytes: vec![0; 68],
            scale: vec![],
        };
        let e = Step::VCols { head: 16 }
            .apply(&g, "q", &mut q)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("64 columns are not 4 V heads of 16 whole blocks"),
            "{e}"
        );
    }

    /// A table in two FP8 parts, each with its own scale: each part reads its own, so unequal
    /// scales are refused, and equal ones give each part's bytes under the one scale.
    #[test]
    fn a_table_s_parts_read_their_own_scales() {
        let (one, two) = (1.0f32.to_le_bytes(), 2.0f32.to_le_bytes());
        let rows = [0x38u8; 2 * 160];
        let file = |first: &[u8], second: &[u8]| {
            let list: [(&str, &str, &[usize], &[u8]); 4] = [
                ("e.shard_0.weight", "F8_E4M3", &[2, 160], &rows),
                ("e.shard_0.weight_scale", "F32", &[1], first),
                ("e.shard_1.weight", "F8_E4M3", &[2, 160], &[0x40u8; 320]),
                ("e.shard_1.weight_scale", "F32", &[1], second),
            ];
            hf::safetensors::Safetensors::parse(hf::safetensors::write(&list)).unwrap()
        };
        let hq = serde_json::json!({"quantization": {"quant_algo": "MIXED_PRECISION",
            "quantized_layers": {"e": {"quant_algo": "FP8"}}}});
        let m = Mapped {
            sources: vec!["e.shard_0.weight".into(), "e.shard_1.weight".into()],
            role: Use::Rows,
            steps: vec![],
            dims: vec![160, 4],
        };
        let (inf, nan) = (f32::INFINITY.to_le_bytes(), f32::NAN.to_le_bytes());
        let mut missed = vec![];
        for (first, second, want) in [
            (&one, &two, Some("e.shard_1.weight: type 2000 scale [2.0]")),
            (&one, &one, None),
            (&inf, &inf, Some("e.shard_0.weight_scale")),
            (&nan, &nan, Some("e.shard_0.weight_scale")),
        ] {
            let st = file(first, second);
            let entries: Vec<(String, Dtype, Vec<usize>)> = st
                .names()
                .map(|n| {
                    let r = st.raw(n).unwrap();
                    (n.to_string(), r.dtype, r.shape.to_vec())
                })
                .collect();
            let p = plan(
                &Config::parse(&serde_json::json!({}), Some(&hq)).unwrap(),
                entries
                    .iter()
                    .map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
            );
            let got: Vec<_> = m.parts(&p, |n| st.raw(n)).collect();
            if let Some(want) = want {
                if !got
                    .iter()
                    .any(|n| n.as_ref().is_err_and(|e| e.to_string().contains(want)))
                {
                    missed.push(format!("{want}: {got:?}"));
                }
            } else {
                assert!(
                    got.iter().all(|n| n.as_ref().unwrap().scale == [1.0]),
                    "{got:?}"
                );
                assert_eq!(
                    got.iter()
                        .map(|n| n.as_ref().unwrap().bytes.as_slice())
                        .collect::<Vec<_>>(),
                    vec![&[0x38u8; 320][..], &[0x40u8; 320][..]]
                );
            }
        }
        assert!(missed.is_empty(), "accepted: {missed:?}");
    }
}
