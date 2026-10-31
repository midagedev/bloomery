//! Qwen3.8-Flash-Next's Hugging Face checkpoint in the qwen4exp names, layouts and keys the
//! program reads, as llama.cpp's converter (`conversion/qwen4exp.py`, `Qwen4ExpTextModel`, and
//! the `qwen.py` classes it extends, at `098ce526`) writes them into a GGUF, Qwen3.6's MLX
//! folder in the qwen35moe ones, and the dense Qwen3.5-class MLX folder (a SwiGLU block in place
//! of the routed experts) in the qwen35 ones ([`Family`]), and ModelOpt's export of that dense
//! class (`nvidia/Qwen3.8-27B-NVFP4`: transformers' names, gammas and conv taps, FP8 attention,
//! NVFP4 feed-forward blocks and head) in the same ones.
//!
//! A checkpoint tensor maps to one tensor of the file ([`names`]), the way its role reads it
//! ([`Use`]) and the [`Step`]s that move it into the program's layout once its format is
//! normalized ([`hf::modelopt::normalize`]): the gated delta-rule V heads from grouped by K head
//! to tiled (`_LinearAttentionVReorderBase`; an MLX file keeps HF's grouped order), the indexer's
//! joint q and k projection split by rows, `+1` on the zero-centred gammas of an HF-native
//! checkpoint (an MLX file's carry it already), and `A_log` as `-exp`. The routed experts of a
//! projection stack by expert id (Qwen3.6's arrive stacked, one `switch_mlp` tensor), a load
//! reading a stack of plain NVFP4 experts where the folder stores it, as NVFP4 rows planes
//! ([`Origin::Stack`]), and the PLE table's parts concatenate in part order. Qwen3.8's target and next-token layers share the
//! layer map; Qwen3.6's next-token layers (`mtp.*`) and every family's vision tower are left out.
//! The dense families' next-token layer is left out by the name of each of its tensors
//! ([`dense_next_token`]) and refused by name as a draft when a folder holds one; any other name
//! is refused.

use std::collections::BTreeMap;
use std::ops::Range;

use gguf::{GgmlType, StackLayout, Value};
use hf::modelopt::{Normal, Plain, Use};
use hf::schema::{self, Fields, Rule, Rule::*, Type};
use serde_json::Value as Json;

use model::arch::names as common;
use model::arch::qwen35moe::hparams::{MAX_PLE_HEADS, MAX_PLE_NGRAM};
use model::arch::qwen35moe::names::{self, Sub};

use super::section::Section;
use super::trunk;
use crate::folder::{FolderMap, Mapping, Origin, PlusOne, Squeeze, VocabSpec};

pub mod folder;

/// Why the checkpoint was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("qwen35moe hf: config.json: {0}")]
    Config(String),
    #[error("qwen35moe hf: {name}: {what}")]
    Tensor { name: String, what: String },
}

impl From<String> for Error {
    fn from(why: String) -> Error {
        Error::Config(why)
    }
}

fn refuse(name: &str, what: impl Into<String>) -> Error {
    Error::Tensor {
        name: name.to_string(),
        what: what.into(),
    }
}

/// A checkpoint family the map reads: how its checkpoint spells names and config, how its tensors
/// lie, and the file it is written as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Qwen3.8-Flash-Next, written as a `qwen4exp` file.
    Qwen38,
    /// Qwen3.6 as MLX's convert stores it, written as a `qwen35moe` file.
    Qwen36,
    /// The dense Qwen3.5-class model (`Qwen3_5ForConditionalGeneration`: the 27B) as MLX's convert
    /// stores it, written as a `qwen35` file: Qwen3.6's layers with one SwiGLU block in place of
    /// the routed experts.
    Qwen35,
    /// [`Family::Qwen35`] in the text-only class (`Qwen3_5ForCausalLM`), whose `config.json` is
    /// the text model's own object.
    Qwen35Text,
    /// [`Family::Qwen35`]'s class as ModelOpt exports it (`nvidia/Qwen3.8-27B-NVFP4`): the same
    /// layers in transformers' own names, gammas and conv taps, written as a `qwen35` file. Its
    /// config names the class [`Family::Qwen35`] has, so [`Family::of`] tells them apart by the
    /// quantization config's producer.
    Qwen35Hf,
}

impl Family {
    /// The families a config's class name picks; [`Family::Qwen35Hf`] shares [`Family::Qwen35`]'s
    /// name and is picked by [`Family::of`] from it.
    const ALL: [Family; 4] = [
        Family::Qwen38,
        Family::Qwen36,
        Family::Qwen35,
        Family::Qwen35Text,
    ];

    /// The family whose checkpoint class is `architecture`, `modelopt` whether the config's
    /// `quantization_config.quant_method` is `modelopt`: the dense class is stored as MLX's convert
    /// writes it, unless ModelOpt exported it.
    fn of(architecture: &str, modelopt: bool) -> Option<Family> {
        let family = Family::ALL
            .into_iter()
            .find(|f| f.architecture() == architecture)?;
        Some(match family {
            Family::Qwen35 if modelopt => Family::Qwen35Hf,
            family => family,
        })
    }

    /// The checkpoint's class, `config.json`'s `architectures`.
    fn architecture(self) -> &'static str {
        match self {
            Family::Qwen38 => "Qwen4ExpForConditionalGeneration",
            Family::Qwen36 => "Qwen3_5MoeForConditionalGeneration",
            Family::Qwen35 | Family::Qwen35Hf => "Qwen3_5ForConditionalGeneration",
            Family::Qwen35Text => "Qwen3_5ForCausalLM",
        }
    }

    /// Whether the checkpoint is stored as transformers writes it (and ModelOpt exports it): its
    /// tensor names under `model.language_model.`, the norm gammas without the `+ 1` the program
    /// reads, and the conv taps as `[channels, 1, kernel]`. MLX's convert writes names under
    /// `language_model.model.`, the gammas with the `+ 1` folded in and taps `[channels, kernel,
    /// 1]`.
    fn hf(self) -> bool {
        matches!(self, Family::Qwen38 | Family::Qwen35Hf)
    }

    /// The role of an attention or delta-rule projection. The dense Qwen3.5 body reads these as
    /// Q8_0 whatever the checkpoint stores, so an HF dense family asks [`Use::Q8_0`] (an FP8
    /// projection is decoded exactly and quantized again at load); the others keep the stored
    /// format, and `Plain::Q8_0` writes an unquantized one as Q8_0.
    fn projection(self) -> Use {
        match self {
            Family::Qwen35Hf => Use::Q8_0,
            Family::Qwen38 | Family::Qwen36 | Family::Qwen35 | Family::Qwen35Text => {
                Use::Matrix(Plain::Q8_0)
            }
        }
    }

    /// The feed-forward block of every layer.
    fn ffn(self) -> trunk::Ffn {
        match self {
            Family::Qwen38 | Family::Qwen36 => trunk::Ffn::Routed,
            Family::Qwen35 | Family::Qwen35Text | Family::Qwen35Hf => trunk::Ffn::Dense,
        }
    }

    /// Whether `config.json` is the text model's own object, with no `text_config` inside it.
    fn text_only(self) -> bool {
        self == Family::Qwen35Text
    }

    /// Whether the map reads a next-token layer ([`FolderMap::draft`]).
    fn reads_draft(self) -> bool {
        self.ffn() == trunk::Ffn::Routed
    }

    /// The file's `general.architecture`, which prefixes its keys.
    fn gguf(self) -> &'static str {
        match self {
            Family::Qwen38 => "qwen4exp",
            Family::Qwen36 => "qwen35moe",
            Family::Qwen35 | Family::Qwen35Text | Family::Qwen35Hf => "qwen35",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Family::Qwen38 => "Qwen3.8",
            Family::Qwen36 => "Qwen3.6",
            Family::Qwen35 | Family::Qwen35Text | Family::Qwen35Hf => "Qwen3.5",
        }
    }

    /// The checkpoint's prefix of the text model's tensors.
    fn text(self) -> &'static str {
        if self.hf() {
            "model.language_model."
        } else {
            "language_model.model."
        }
    }

    /// The checkpoint's output head.
    fn lm_head(self) -> &'static str {
        if self.hf() {
            "lm_head.weight"
        } else {
            "language_model.lm_head.weight"
        }
    }

    /// The checkpoint's prefix of the vision tower's tensors.
    fn vision(self) -> &'static str {
        if self.hf() {
            "model.visual."
        } else {
            "vision_tower."
        }
    }

    /// Whether `name` is a tensor of the next-token layers: anything under `mtp.` for a routed
    /// family, whose draft map refuses what it does not know, and for a dense one the tensors
    /// [`dense_next_token`] names, so another `mtp.` name is refused by the map.
    fn next_token(self, name: &str) -> bool {
        match self.ffn() {
            trunk::Ffn::Routed => name.starts_with("mtp."),
            trunk::Ffn::Dense => dense_next_token(name),
        }
    }

    /// Whether `name` is a tensor the map leaves out: the vision tower by its prefix, and the
    /// next-token layers in the target's map or everything else in the draft's.
    fn leaves_out(self, name: &str, draft: bool) -> bool {
        name.starts_with(self.vision()) || self.next_token(name) != draft
    }

    /// The `config.json` keys the family has beside [`ROOT_SHARED`].
    fn root(self) -> &'static [&'static Fields] {
        match self {
            Family::Qwen38 => &[ROOT_38],
            Family::Qwen36 => &[ROOT_MLX, ROOT_36],
            Family::Qwen35 => &[ROOT_MLX, ROOT_35],
            Family::Qwen35Text => &[ROOT_MLX],
            Family::Qwen35Hf => &[ROOT_HF_DENSE],
        }
    }

    /// The `text_config` keys the family has: [`TEXT_SHARED`], the trunk's, its feed-forward
    /// block's, the shared expert's of a routed one, and its own.
    fn text_config(self) -> Vec<&'static Fields> {
        let mut tables = vec![TEXT_SHARED, trunk::KEYS, self.ffn().keys()];
        if self.ffn() == trunk::Ffn::Routed {
            tables.push(SHARED_EXPERT);
        }
        tables.push(match self {
            Family::Qwen38 => TEXT_38,
            Family::Qwen36 => TEXT_36,
            Family::Qwen35 | Family::Qwen35Text | Family::Qwen35Hf => TEXT_35,
        });
        tables
    }

    /// The `text_config` counts the family reads: [`COUNTS`], the trunk's and its block's, and
    /// its own.
    fn counts(self) -> impl Iterator<Item = &'static str> {
        let routed: &[&str] = match self.ffn() {
            trunk::Ffn::Routed => &["shared_expert_intermediate_size"],
            trunk::Ffn::Dense => &[],
        };
        let own: &[&str] = match self {
            Family::Qwen38 => COUNTS_38,
            Family::Qwen36 | Family::Qwen35 | Family::Qwen35Text | Family::Qwen35Hf => &[],
        };
        COUNTS
            .iter()
            .chain(trunk::COUNTS)
            .chain(self.ffn().counts())
            .chain(routed)
            .chain(own)
            .copied()
    }
}

const VISION: &str = "the vision tower and its placeholders; this converter writes the text model";
const MTP: &str =
    "the next-token layers, which this converter leaves out and the trunk never reads";
const SAMPLING: &str =
    "the checkpoint's sampling defaults; a request's own settings decide how the program samples";
const PLE_SIZES: &str = "sized the PLE hash tables in training; the checkpoint's ngram_heads_offsets \
    and ngram_heads_vocab_sizes hold the sizes, which are exported";

/// `config.json`, every family's keys. `hf::quant_config` reads `quantization_config`; the
/// vocabulary keys are `text_config`'s, which `hf::vocab::Config` reads.
const ROOT_SHARED: &Fields = &[
    ("architectures", Read),
    ("text_config", Read),
    ("quantization_config", Read),
    // The map needs a separate lm_head.
    trunk::HEAD_OWN,
    (
        "transformers_version",
        Ignore(Type::String, "the version that wrote the file"),
    ),
    ("vision_config", Ignore(Type::Object, VISION)),
    ("video_token_id", Ignore(Type::Number, VISION)),
    ("vision_start_token_id", Ignore(Type::Number, VISION)),
    ("vision_end_token_id", Ignore(Type::Number, VISION)),
];

/// `language_model_only`, a flag of the multimodal class that only the vision tower reads.
const LANGUAGE_MODEL_ONLY: (&str, Rule) = ("language_model_only", Ignore(Type::Bool, VISION));

/// `image_token_id` of a family whose program has no image token.
const IMAGE_TOKEN_UNREAD: (&str, Rule) = ("image_token_id", Ignore(Type::Number, VISION));

/// Qwen3.8's `config.json` keys beside [`ROOT_SHARED`].
const ROOT_38: &Fields = &[
    ("model_type", Need(r#"["qwen4_exp"]"#)),
    ("image_token_id", Read),
    trunk::DTYPE,
    LANGUAGE_MODEL_ONLY,
];

/// The HF-native dense family's `config.json` keys beside [`ROOT_SHARED`]: the keys of ModelOpt's
/// export, which has no MLX `quantization` block, no root `eos_token_id` (the text model's is
/// `hf::vocab::Config`'s) and no sampling defaults (they sit in `generation_config.json`).
const ROOT_HF_DENSE: &Fields = &[
    ("model_type", Need(r#"["qwen3_5"]"#)),
    IMAGE_TOKEN_UNREAD,
    trunk::DTYPE,
    LANGUAGE_MODEL_ONLY,
];

/// An MLX family's `config.json` keys beside [`ROOT_SHARED`]: `quantization` is MLX's block, which
/// `hf::quant_config` reads, and `eos_token_id` the ids `hf::vocab::Config` reads.
const ROOT_MLX: &Fields = &[
    ("quantization", Read),
    ("eos_token_id", Read),
    IMAGE_TOKEN_UNREAD,
];

/// Qwen3.6's `config.json` keys beside [`ROOT_SHARED`] and [`ROOT_MLX`].
const ROOT_36: &Fields = &[("model_type", Need(r#"["qwen3_5_moe"]"#))];

/// Qwen3.5's `config.json` keys beside [`ROOT_SHARED`] and [`ROOT_MLX`]: the checkpoint's
/// sampling defaults sit at the top level and in `generation_config`.
const ROOT_35: &Fields = &[
    ("model_type", Need(r#"["qwen3_5"]"#)),
    LANGUAGE_MODEL_ONLY,
    ("generation_config", Ignore(Type::Object, SAMPLING)),
    ("do_sample", Ignore(Type::Bool, SAMPLING)),
    ("temperature", Ignore(Type::Number, SAMPLING)),
    ("top_k", Ignore(Type::Number, SAMPLING)),
    ("top_p", Ignore(Type::Number, SAMPLING)),
];

/// `config.json`'s `text_config`, every family's keys beside [`trunk::KEYS`] and its block's. A
/// setting the program computes with one value and GGUF has no key for is `Only` that value:
/// another would be served as this one, and an absent one is it.
const TEXT_SHARED: &Fields = &[
    ("linear_conv_kernel_dim", Read),
    ("linear_key_head_dim", Read),
    ("linear_num_key_heads", Read),
    ("linear_num_value_heads", Read),
    ("linear_value_head_dim", Read),
    ("layer_types", Read),
    ("rope_parameters", Read),
    ("partial_rotary_factor", Read),
    // Serving keeps the gated delta-rule state in F32.
    ("mamba_ssm_dtype", Only(r#"["float32"]"#)),
    ("mtp_num_hidden_layers", Ignore(Type::Number, MTP)),
    ("mtp_use_dedicated_embeddings", Ignore(Type::Bool, MTP)),
];

/// The counts of [`TEXT_SHARED`] every family reads beside [`trunk::COUNTS`]: each a positive
/// u32.
const COUNTS: &[&str] = &[
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_value_head_dim",
];

/// The shared expert's width, which both routed families have.
const SHARED_EXPERT: &Fields = &[("shared_expert_intermediate_size", Read)];

/// Qwen3.8's counts beside [`COUNTS`] and [`trunk::COUNTS`].
const COUNTS_38: &[&str] = &[
    "full_attention_interval",
    "hc_count",
    "hc_lowrank",
    "indexer_n_heads",
    "indexer_head_dim",
    "indexer_budget",
    "indexer_compress_ratio",
];

/// Qwen3.8's `text_config` keys beside [`TEXT_SHARED`].
const TEXT_38: &Fields = &[
    ("full_attention_interval", Read),
    ("hc_count", Read),
    ("hc_lowrank", Read),
    ("indexer_n_heads", Read),
    ("indexer_head_dim", Read),
    ("indexer_budget", Read),
    ("indexer_compress_ratio", Read),
    ("ple_layer_ids", Read),
    ("ngram_size", Read),
    ("heads_per_ngram", Read),
    ("ple_conv_kernel_size", Read),
    ("split_ngram_parts", Read),
    ("ple_embed_dim", Read),
    ("model_type", Only(r#"["qwen4_exp_text"]"#)),
    // The gated delta-rule output gate.
    ("output_gate_type", Only(r#"["sigmoid"]"#)),
    // The router renormalizes its top k.
    ("norm_topk_prob", Only("[true]")),
    // The indexer's key projection is one head of indexer_head_dim rows.
    ("indexer_kv_heads", Only("[1]")),
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
];

/// Qwen3.6's `text_config` keys beside [`TEXT_SHARED`].
const TEXT_36: &Fields = &[
    (
        "full_attention_interval",
        Ignore(
            Type::Number,
            "layer_types lists every layer's kind, and the file's interval is made from that list",
        ),
    ),
    ("model_type", Only(r#"["qwen3_5_moe_text"]"#)),
    // The attention query projection holds each head's output gate beside its query.
    ("attn_output_gate", Only("[true]")),
];

/// Qwen3.5's `text_config` keys beside [`TEXT_SHARED`].
const TEXT_35: &Fields = &[
    (
        "full_attention_interval",
        Ignore(
            Type::Number,
            "layer_types lists every layer's kind, and the file's interval is made from that list",
        ),
    ),
    ("model_type", Only(r#"["qwen3_5_text"]"#)),
    // The attention query projection holds each head's output gate beside its query.
    ("attn_output_gate", Only("[true]")),
    // The gated delta rule's output gate is `silu(z)` (HF's `swish`, `GdnGate::Silu`).
    ("output_gate_type", Only(r#"["swish"]"#)),
    // The list names the layers a mixture-of-experts config makes dense; this family has no
    // routed layer, so the list is HF's empty default.
    ("mlp_only_layers", Only("[[]]")),
];

/// `text_config.rope_parameters`: interleaved M-RoPE (`RopeMode::Imrope`), unscaled.
const ROPE: &Fields = &[
    ("rope_type", Only(r#"["default"]"#)),
    // The older spelling of `rope_type`, which some exported folders write beside it or alone.
    ("type", Only(r#"["default"]"#)),
    ("mrope_interleaved", Need("[true]")),
    ("mrope_section", Read),
    ("rope_theta", Read),
    ("partial_rotary_factor", Read),
];

/// The config values the map reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Geometry {
    family: Family,
    layers: usize,
    /// Whether each layer attends, by `layer_types`; the others are gated delta-rule layers.
    full: Vec<bool>,
    k_heads: usize,
    v_heads: usize,
    head_k: usize,
    head_v: usize,
    /// The indexer's query rows of `index_qk_proj`, its key rows following; 0 for a family
    /// without the indexer.
    indexer_q: usize,
    /// The routed block's experts; 0 for a dense family.
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
    /// The dense block's width, `intermediate_size`; `None` for a routed family.
    dense_ff: Option<u32>,
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

/// What only a Qwen3.8 config has, checked: the indexer's query rows and the compress ratio.
struct Exp {
    indexer_q: u32,
    ratio: i32,
}

impl Exp {
    fn of(c: &Reader<'_>, counts: &BTreeMap<&str, u32>) -> Result<Exp, Error> {
        let n = |key: &str| counts[key];
        let indexer_q = n("indexer_n_heads")
            .checked_mul(n("indexer_head_dim"))
            .ok_or_else(|| c.bad("indexer_n_heads", "times indexer_head_dim overflows u32"))?;
        if n("hc_count") < 2 {
            return Err(c.bad("hc_count", "must be at least two"));
        }
        let ratio = i32::try_from(n("indexer_compress_ratio"))
            .map_err(|_| c.bad("indexer_compress_ratio", "does not fit i32"))?;
        if !n("indexer_budget").is_multiple_of(ratio as u32) {
            return Err(c.bad("indexer_compress_ratio", "must divide indexer_budget"));
        }
        Ok(Exp { indexer_q, ratio })
    }
}

/// The `N` for which exactly every `N`th layer of `full` attends, the interval the file's header
/// holds; `None` when no `N` makes the list.
fn attention_interval(full: &[bool]) -> Option<u32> {
    (1..=full.len() + 1)
        .find(|&n| {
            full.iter()
                .enumerate()
                .all(|(i, &f)| f == ((i + 1) % n == 0))
        })
        .and_then(|n| u32::try_from(n).ok())
}

/// The reads of an object of `config.json`, refusals in this map's error.
type Reader<'a> = Section<'a, Error>;

impl Checkpoint {
    /// Read the fields this converter uses, and refuse by name a key the family's tables
    /// ([`ROOT_SHARED`] and its own, [`TEXT_SHARED`] and its own, [`ROPE`]) do not name or a value
    /// they do not allow.
    pub fn parse(config: &Json) -> Result<Self, Error> {
        let root = Reader::new(config, "config")?;
        let arch = root.array("architectures")?;
        let modelopt = config
            .pointer("/quantization_config/quant_method")
            .and_then(Json::as_str)
            == Some("modelopt");
        let family = match arch {
            [Json::String(s)] => Family::of(s, modelopt),
            _ => None,
        }
        .ok_or_else(|| {
            let known: Vec<_> = Family::ALL.iter().map(|f| f.architecture()).collect();
            root.bad(
                "architectures",
                &format!("{arch:?}: only {known:?} are supported"),
            )
        })?;
        if family.text_only() && config.get("text_config").is_some() {
            return Err(root.bad(
                "text_config",
                "belongs to the multimodal class; a text-only config holds the text model's keys itself",
            ));
        }
        let mut root_tables = vec![ROOT_SHARED];
        root_tables.extend(family.root());
        let c = if family.text_only() {
            // A text-only checkpoint's config is the text model's own object: one object holds
            // both tables.
            root_tables.extend(family.text_config());
            schema::check(config, "config", &root_tables).map_err(Error::Config)?;
            Reader::new(config, "config")?
        } else {
            schema::check(config, "config", &root_tables).map_err(Error::Config)?;
            let c = Reader::new(root.field("text_config")?, "text_config")?;
            schema::check(c.value, "text_config", &family.text_config()).map_err(Error::Config)?;
            c
        };
        let vocab = hf::vocab::Config::parse(config).map_err(|e| Error::Config(e.to_string()))?;
        let mut counts = BTreeMap::new();
        for field in family.counts() {
            counts.insert(field, c.count(field)?);
        }
        let n = |key: &str| counts[key];
        let product = |a: &str, b: &str| {
            n(a).checked_mul(n(b))
                .ok_or_else(|| c.bad(a, &format!("times {b} overflows u32")))
        };
        let inner = product("linear_value_head_dim", "linear_num_value_heads")?;
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
        trunk::check_counts(&c, family.ffn(), n)?;
        let exp = match family {
            Family::Qwen38 => Some(Exp::of(&c, &counts)?),
            Family::Qwen36 | Family::Qwen35 | Family::Qwen35Text | Family::Qwen35Hf => None,
        };
        let kinds = c.array("layer_types")?;
        if kinds.len() != n("num_hidden_layers") as usize {
            return Err(c.bad("layer_types", "length differs from num_hidden_layers"));
        }
        let mut full = vec![];
        for (i, kind) in kinds.iter().enumerate() {
            full.push(match kind.as_str() {
                Some("linear_attention") => false,
                Some("full_attention") => true,
                _ => {
                    return Err(c.bad(
                        "layer_types",
                        &format!("unsupported layer type {kind} at {i}"),
                    ));
                }
            });
        }
        let interval = match &exp {
            Some(_) => {
                let interval = n("full_attention_interval");
                if full
                    .iter()
                    .enumerate()
                    .any(|(i, &f)| f != ((i + 1) % interval as usize == 0))
                {
                    return Err(c.bad("layer_types", "disagrees with full_attention_interval"));
                }
                interval
            }
            None => attention_interval(&full).ok_or_else(|| {
                c.bad(
                    "layer_types",
                    "is not one full-attention layer in every N, the file's one interval",
                )
            })?,
        };
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
        let mut metadata = trunk::metadata(
            n,
            family.ffn(),
            rope.number("rope_theta")?,
            c.number("rms_norm_eps")?,
        );
        // The sections follow the head counts, before the rope's base: the order the split's keys
        // have always been written in.
        metadata.insert(
            5,
            ("rope.dimension_sections".into(), Value::Array(sections)),
        );
        if family.ffn() == trunk::Ffn::Routed {
            metadata.push((
                "expert_shared_feed_forward_length".into(),
                Value::U32(n("shared_expert_intermediate_size")),
            ));
        }
        metadata.extend([
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
            ("full_attention_interval".into(), Value::U32(interval)),
            ("rope.dimension_count".into(), Value::U32(rotary as u32)),
        ]);
        if let Some(exp) = &exp {
            let ratios = full
                .iter()
                .map(|&f| Value::I32(if f { exp.ratio } else { 0 }));
            metadata.extend([
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
                (
                    "attention.compress_ratios".into(),
                    Value::Array(ratios.collect()),
                ),
            ]);
        }
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
                family,
                layers: n("num_hidden_layers") as usize,
                full,
                k_heads: n("linear_num_key_heads") as usize,
                v_heads: n("linear_num_value_heads") as usize,
                head_k: n("linear_key_head_dim") as usize,
                head_v: n("linear_value_head_dim") as usize,
                indexer_q: exp.as_ref().map_or(0, |e| e.indexer_q as usize),
                experts: counts.get("num_experts").map_or(0, |&e| e as usize),
                ple_parts,
            },
            hidden: n("hidden_size"),
            dense_ff: counts.get("intermediate_size").copied(),
            indexer_k: counts.get("indexer_head_dim").copied().unwrap_or(0),
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
        let mapped = map(&self.geometry, tensors, false)?;
        for m in mapped.values() {
            for name in &m.sources {
                if name.contains(".ple.")
                    && self.ple.as_ref().is_none_or(|ple| {
                        !name.starts_with(&format!(
                            "{}layers.{}.ple.",
                            self.geometry.family.text(),
                            ple.layer
                        ))
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
        if let Some(ff) = self.dense_ff {
            // Every layer's block: gate and up are `[hidden_size, intermediate_size]`, down the
            // transpose.
            let (hidden, ff) = (u64::from(self.hidden), u64::from(ff));
            for l in 0..self.geometry.layers {
                for (name, want) in [
                    (common::ffn_gate(l), [hidden, ff]),
                    (common::ffn_up(l), [hidden, ff]),
                    (common::ffn_down(l), [ff, hidden]),
                ] {
                    let m = mapped
                        .get(&name)
                        .ok_or_else(|| refuse(&name, "is missing"))?;
                    if m.dims != want {
                        return Err(refuse(
                            &m.sources[0],
                            format!(
                                "dims {:?} disagree with hidden_size {hidden} and \
                                 intermediate_size {ff}",
                                m.dims
                            ),
                        ));
                    }
                }
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
            Value::String(self.geometry.family.gguf().into()),
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
                let name = format!(
                    "{}layers.{layer}.ple.ple_embedding.{part}",
                    self.geometry.family.text()
                );
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
            .map(|(k, v)| (format!("{}.{k}", self.geometry.family.gguf()), v))
            .collect())
    }
}

impl From<Error> for crate::folder::Error {
    fn from(e: Error) -> crate::folder::Error {
        crate::folder::Error::Model(Box::new(e))
    }
}

/// Whether `name` is a tensor of the dense families' next-token layer: its fusion projection and
/// norms, and the layer's own, an attention layer's with a dense block (the names [`trunk`] gives
/// them). The dense map leaves these out until a draft reads them; any other `mtp.` name is not
/// the map's and is refused.
fn dense_next_token(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("mtp.") else {
        return false;
    };
    if [
        "fc.weight",
        "norm.weight",
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
    ]
    .contains(&rest)
    {
        return true;
    }
    rest.strip_prefix("layers.")
        .and_then(|r| r.split_once('.'))
        .is_some_and(|(layer, rest)| {
            layer.parse::<usize>().is_ok()
                && trunk::attention(0, rest)
                    .or_else(|| trunk::dense_ffn(0, rest))
                    .or_else(|| trunk::block_norms(0, rest, post_attention_norm))
                    .is_some()
        })
}

/// Why the checkpoint tensor `name` of `family` is left out of the header, when the map reads no
/// part of it: the next-token layers, the vision tower, and the PLE hash constants, which are
/// header keys ([`Checkpoint::header`]).
fn skip_reason(family: Family, name: &str) -> Option<&'static str> {
    if family.next_token(name) {
        return Some(MTP);
    }
    if name.starts_with(family.vision()) {
        return Some(VISION);
    }
    let (layer, rest) = name
        .strip_prefix(family.text())?
        .strip_prefix("layers.")?
        .split_once('.')?;
    layer.parse::<usize>().ok()?;
    let constant = rest.strip_prefix("ple.ple_embedding.")?;
    PLE_CONSTANTS
        .iter()
        .any(|(c, _)| *c == constant)
        .then_some("a PLE hash constant, read as a header key")
}

// Where a name below is `Checkpoint`'s own (`parse`, `map`, `header`), the inherent function is the
// one called: the trait's wrap it in the folder's error.
impl FolderMap for Checkpoint {
    type Step = Step;

    const MODEL_TYPES: &'static [&'static str] =
        &["qwen4_exp", "qwen3_5_moe", "qwen3_5", "qwen3_5_text"];

    fn parse(config: &Json) -> Result<Self, crate::folder::Error> {
        Ok(Checkpoint::parse(config)?)
    }

    fn module_class(&self, module: &str) -> Option<hf::quant_config::ModuleClass> {
        crate::arch::module_class(module)
    }

    fn map<'a>(
        &self,
        tensors: &[(&'a str, &'a [usize])],
    ) -> Result<Mapping<Step>, crate::folder::Error> {
        let tensors_of = tensors.iter().copied();
        let skipped = tensors
            .iter()
            .filter_map(|(n, _)| {
                Some((
                    n.to_string(),
                    skip_reason(self.geometry.family, n)?.to_string(),
                ))
            })
            .collect();
        Ok(Mapping {
            tensors: Checkpoint::map(self, tensors_of)?,
            skipped,
        })
    }

    fn mover(&self, name: &str, steps: &[Step]) -> Option<crate::part::Step> {
        (!steps.is_empty()).then(|| {
            let (g, name, steps) = (self.geometry.clone(), name.to_string(), steps.to_vec());
            Box::new(move |n: &mut Normal| {
                steps
                    .iter()
                    .try_for_each(|s| s.apply(&g, &name, n))
                    .map_err(|e| e.to_string())
            }) as crate::part::Step
        })
    }

    fn architecture(&self) -> &'static str {
        self.geometry.family.gguf()
    }

    fn header(
        &self,
        planned: &[(&str, &[u64])],
        folder: &safetensors::Folder,
    ) -> Result<Vec<(String, Value)>, crate::folder::Error> {
        Ok(Checkpoint::header(self, planned.iter().copied(), folder)?)
    }

    fn vocab(&self) -> &VocabSpec {
        &self.vocab
    }

    fn draft(
        &self,
        path: &std::path::Path,
        files: &safetensors::Folder,
        plan: &hf::quant_config::Plan,
        kvs: &[(String, Value)],
    ) -> Result<Option<(gguf::Split, Vec<folder::Planned>)>, crate::folder::Error> {
        if !self.geometry.family.reads_draft() {
            return match files.names().find(|n| self.geometry.family.next_token(n)) {
                Some(name) => Err(refuse(
                    name,
                    format!(
                        "a next-token layer, which the {} map does not read",
                        self.geometry.family.label()
                    ),
                )
                .into()),
                None => Ok(None),
            };
        }
        folder::mtp(path, self, (files, plan), kvs)
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
    /// Two stacked matrices joined along their columns, embedding before hidden.
    JoinColumns,
    /// Only these rows.
    Rows(Range<usize>),
    /// The one unit dim of a 3-D tensor dropped: Qwen3.6's conv taps are stored `[channels,
    /// kernel, 1]` (ggml dims `[1, kernel, channels]`), rows of `kernel` values once it is gone.
    Squeeze,
    /// `x + 1`, on F32: [`PlusOne`].
    PlusOne,
    /// `-exp(x)`, on F32, computed in f64 and rounded once.
    NegExp,
}

/// One qwen4exp tensor.
pub type Mapped = crate::folder::Mapped<Step>;

/// Tensors stored in parts by index (a layer's experts, the PLE table's shards): name and shape.
type Parts<'a> = BTreeMap<usize, (&'a str, &'a [usize])>;

/// The PLE hash constants, by the checkpoint name's last part, and the keys they become.
const PLE_CONSTANTS: [(&str, &str); 3] = [
    ("layer_multipliers", "ple.layer_multipliers"),
    ("ngram_heads_offsets", "ple.head_offsets"),
    ("ngram_heads_vocab_sizes", "ple.head_vocab_sizes"),
];

/// The tensors of `g`'s file, by name, of the checkpoint tensors `tensors` (name and row-major
/// logical shape: a packed weight's caller gives its unpacked shape) under `g`. The PLE hash
/// constants are keys ([`Checkpoint::keys`]), not tensors.
fn map<'a>(
    g: &Geometry,
    tensors: impl IntoIterator<Item = (&'a str, &'a [usize])>,
    draft: bool,
) -> Result<BTreeMap<String, Mapped>, Error> {
    let mut out = BTreeMap::new();
    let mut put = |name: String, m: Mapped| match out.insert(name, m) {
        Some(prev) => Err(refuse(
            &prev.sources[0],
            format!("maps to a {} name another tensor has", g.family.gguf()),
        )),
        None => Ok(()),
    };
    // A layer's experts: (layer, projection) to each expert's name and shape.
    let mut experts: BTreeMap<(usize, &str), Parts> = BTreeMap::new();
    let mut ple = Parts::new();
    let mut join = Parts::new();
    for (name, shape) in tensors {
        let left_out = g.family.leaves_out(name, draft);
        if !left_out && !trunk::sized(shape) {
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
            origin: Origin::Made,
        };
        if left_out {
            continue;
        }
        if draft {
            let stem = match name.strip_prefix("mtp.") {
                Some("fc_embedding.weight") => {
                    join.insert(0, (name, shape));
                    continue;
                }
                Some("fc_hidden.weight") => {
                    join.insert(1, (name, shape));
                    continue;
                }
                Some("pre_fc_norm_embedding.weight") => Some(("enorm.weight", true)),
                Some("pre_fc_norm_hidden.weight") => Some(("hnorm.weight", true)),
                Some("hyper_connection_mixer.hc_norm.weight") => {
                    Some(("hc_head_norm.weight", true))
                }
                Some("hyper_connection_mixer.input_mix_weight_down.weight") => {
                    Some(("hc_head_down.weight", false))
                }
                Some("hyper_connection_mixer.input_mix_weight_up.weight") => {
                    Some(("hc_head_up.weight", false))
                }
                _ => None,
            };
            if let Some((stem, plus_one)) = stem {
                let plain = if plus_one { Plain::F32 } else { Plain::Q8_0 };
                put(
                    format!("blk.{}.nextn.{stem}", g.layers),
                    one(
                        Use::Matrix(plain),
                        if plus_one {
                            vec![Step::PlusOne]
                        } else {
                            vec![]
                        },
                        dims(),
                    ),
                )?;
                continue;
            }
        }
        if name == g.family.lm_head() {
            put(
                names::output(),
                one(Use::Matrix(Plain::Q8_0), vec![], dims()),
            )?;
            continue;
        }
        let short = name
            .strip_prefix(if draft { "mtp." } else { g.family.text() })
            .ok_or_else(|| {
                refuse(
                    name,
                    if name.contains(".mtp.") {
                        "a next-token layer under the text prefix; this map knows `mtp.` names only"
                            .to_string()
                    } else if name.starts_with("mtp.") {
                        format!(
                            "not a tensor of the {} next-token layer this map names",
                            g.family.label()
                        )
                    } else {
                        format!("not a {} tensor", g.family.label())
                    },
                )
            })?;
        let mixer = |rest: &str| match rest {
            "hc_norm.weight" => Some((Plain::F32, vec![Step::PlusOne])),
            "input_mix_weight_down.weight" | "input_mix_weight_up.weight" => {
                Some((Plain::Q8_0, vec![]))
            }
            _ => None,
        };
        // Qwen3.8's output norm is its hyper-connection mix, below.
        if let Some((to, plain)) = trunk::global(short)
            .filter(|_| g.family != Family::Qwen38 || short == "embed_tokens.weight")
        {
            // The one F32 global is the output norm, whose gamma an HF checkpoint stores without
            // the `+ 1`.
            let steps = if plain == Plain::F32 && g.family.hf() {
                vec![Step::PlusOne]
            } else {
                vec![]
            };
            put(to, one(Use::Matrix(plain), steps, dims()))?;
            continue;
        }
        if let Some(rest) = short
            .strip_prefix("hyper_connection_mixer.")
            .filter(|_| g.family == Family::Qwen38)
        {
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
            .ok_or_else(|| refuse(name, format!("not a {} tensor", g.family.label())))?;
        let count = if draft { 1 } else { g.layers };
        if l >= count {
            return Err(refuse(name, format!("layer {l} of {count}")));
        }
        let l = if draft { g.layers } else { l };
        if let Some(e) = rest
            .strip_prefix("mlp.experts.")
            .filter(|_| g.family == Family::Qwen38)
        {
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
        if let Some(p) = rest
            .strip_prefix("ple.ple_embedding.")
            .filter(|_| g.family == Family::Qwen38)
        {
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
    if draft && !join.is_empty() {
        let (Some(&(embedding, e)), Some(&(hidden, h))) = (join.get(&0), join.get(&1)) else {
            return Err(refuse(
                "mtp.fc_embedding.weight",
                "the input join needs both fc_embedding and fc_hidden",
            ));
        };
        let &[rows, cols] = e else {
            return Err(refuse(embedding, "the input projection is not a matrix"));
        };
        if h != e || cols.checked_mul(2).is_none() {
            return Err(refuse(
                hidden,
                "the two input projections differ in shape or their joined width overflows",
            ));
        }
        put(
            format!("blk.{}.nextn.eh_proj.weight", g.layers),
            Mapped {
                sources: vec![embedding.to_string(), hidden.to_string()],
                role: Use::Matrix(Plain::Q8_0),
                steps: vec![Step::JoinColumns],
                dims: vec![(cols * 2) as u64, rows as u64],
                origin: Origin::Made,
            },
        )?;
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
                origin: Origin::Stack {
                    ty: GgmlType::NVFP4_ROWS,
                    layout: StackLayout::Planes,
                },
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
                origin: Origin::Made,
            },
        )?;
    }
    Ok(out)
}

/// The name of Qwen3.6's feed-forward block norm, which the file calls the post-attention one.
fn post_attention_norm(l: usize) -> String {
    common::blk(l, "post_attention_norm.weight")
}

/// The tensors of `g`'s file made of `rest`, a tensor of layer `l` other than a Qwen3.8 routed
/// expert or the PLE table.
fn layer(
    g: &Geometry,
    l: usize,
    rest: &str,
    name: &str,
    shape: &[usize],
) -> Result<Vec<(String, Mapped)>, Error> {
    let dims: Vec<u64> = shape.iter().rev().map(|&d| d as u64).collect();
    let made = |role: Use, steps: Vec<Step>, dims: Vec<u64>| Mapped {
        sources: vec![name.to_string()],
        role,
        steps,
        dims,
        origin: Origin::Made,
    };
    let mapped =
        |plain: Plain, steps: Vec<Step>, dims: Vec<u64>| made(Use::Matrix(plain), steps, dims);
    let one = |to: String, plain: Plain, steps: Vec<Step>| {
        Ok(vec![(to, mapped(plain, steps, dims.clone()))])
    };
    // An attention or delta-rule projection, in the role the family reads it
    // ([`Family::projection`]).
    let projection = |to: String, steps: Vec<Step>| {
        Ok(vec![(to, made(g.family.projection(), steps, dims.clone()))])
    };
    let qk = 2 * g.k_heads * g.head_k;
    let (exp, hf, routed) = (
        g.family == Family::Qwen38,
        g.family.hf(),
        g.family.ffn() == trunk::Ffn::Routed,
    );
    // The conv taps as `[kernel, channels]`.
    let taps = || -> Result<Vec<u64>, Error> {
        match (hf, shape) {
            (true, &[c, 1, k]) | (false, &[c, k, 1]) => Ok(vec![k as u64, c as u64]),
            _ => Err(refuse(
                name,
                format!(
                    "conv taps {shape:?} are not {}",
                    if hf {
                        "[channels, 1, kernel]"
                    } else {
                        "[channels, kernel, 1]"
                    }
                ),
            )),
        }
    };
    // The next-token layer (`l == g.layers`, Qwen3.8's draft) attends.
    let full = l == g.layers || g.full[l];
    match (rest.split_once('.'), full) {
        (Some(("linear_attn", _)), true) => {
            return Err(refuse(
                name,
                format!("layer {l} attends by layer_types and holds no linear_attn tensor"),
            ));
        }
        (Some(("self_attn", _)), false) => {
            return Err(refuse(
                name,
                format!(
                    "layer {l} is a delta-rule layer by layer_types and holds no self_attn tensor"
                ),
            ));
        }
        _ => {}
    }
    // Qwen3.6's experts arrive stacked: one `[experts, rows, columns]` tensor a projection.
    if g.family == Family::Qwen36
        && let Some(stack) = trunk::stacked_experts(l, rest, shape, g.experts)
    {
        let (to, plain) = stack.map_err(|why| refuse(name, why))?;
        return one(to, plain, vec![]);
    }
    let (sub, hc) = if let Some(r) = rest.strip_prefix("attn_hyper_connection.").filter(|_| exp) {
        (Some(Sub::Attn), r)
    } else if let Some(r) = rest.strip_prefix("mlp_hyper_connection.").filter(|_| exp) {
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
    // Every family stores the gated delta-rule's V heads grouped by K head (mlx-lm repeats a K
    // head over its V heads in place, `gated_delta.py`), which the program reads tiled.
    let v = |from: usize, head: usize| vec![Step::VRows { from, head }];
    // An HF checkpoint stores a norm gamma `x` and the program reads `x + 1`; an MLX file stores
    // `x + 1` already (mlx-lm's `sanitize` adds it only to a file whose conv taps are still HF's
    // `[channels, 1, kernel]`).
    let plus_one = || if hf { vec![Step::PlusOne] } else { vec![] };
    if let Some((to, plain)) = trunk::attention(l, rest) {
        // Only the per-head norms are F32 here, and they carry the `+ 1` an HF file lacks.
        return if plain == Plain::F32 {
            one(to, plain, plus_one())
        } else {
            projection(to, vec![])
        };
    }
    // The block: a routed layer's router (its experts and shared expert are below), or the dense
    // layer's three projections.
    let ffn = match g.family.ffn() {
        trunk::Ffn::Routed => trunk::router(l, rest),
        trunk::Ffn::Dense => trunk::dense_ffn(l, rest),
    };
    if let Some((to, plain)) = ffn {
        return one(to, plain, vec![]);
    }
    // Qwen3.8's block norms are its hyper-connection mixes, above.
    if !exp && let Some((to, plain)) = trunk::block_norms(l, rest, post_attention_norm) {
        return one(to, plain, plus_one());
    }
    match rest {
        "linear_attn.in_proj_qkv.weight" => projection(names::attn_qkv(l), v(qk, g.head_v)),
        "linear_attn.in_proj_z.weight" => projection(names::attn_gate(l), v(0, g.head_v)),
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
        "linear_attn.conv1d.weight" => {
            // MLX's taps arrive `[channels, kernel, 1]`, an HF file's `[channels, 1, kernel]`.
            let mut steps = if hf { vec![] } else { vec![Step::Squeeze] };
            steps.extend(v(qk, g.head_v));
            Ok(vec![(
                names::ssm_conv1d(l),
                mapped(Plain::F32, steps, taps()?),
            )])
        }
        "linear_attn.norm.weight" => one(names::ssm_norm(l), Plain::F32, vec![]),
        "linear_attn.out_proj.weight" => {
            projection(names::ssm_out(l), vec![Step::VCols { head: g.head_v }])
        }
        "self_attn.indexer.q_layernorm.weight" if exp => {
            one(names::indexer_q_norm(l), Plain::F32, vec![Step::PlusOne])
        }
        "self_attn.indexer.k_layernorm.weight" if exp => {
            one(names::indexer_k_norm(l), Plain::F32, vec![Step::PlusOne])
        }
        "self_attn.indexer.index_qk_proj.weight" if exp => {
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
        "mlp.shared_expert_gate.weight" if routed => match *shape {
            [1, n] => Ok(vec![(
                names::ffn_gate_inp_shexp(l),
                mapped(Plain::Bf16, vec![], vec![n as u64]),
            )]),
            _ => Err(refuse(name, format!("{shape:?} is not one gate row"))),
        },
        "mlp.shared_expert.gate_proj.weight" if routed => {
            one(names::ffn_gate_shexp(l), Plain::Q8_0, vec![])
        }
        "mlp.shared_expert.up_proj.weight" if routed => {
            one(names::ffn_up_shexp(l), Plain::Q8_0, vec![])
        }
        "mlp.shared_expert.down_proj.weight" if routed => {
            one(names::ffn_down_shexp(l), Plain::Q8_0, vec![])
        }
        "ple.key_proj.weight" if exp => one(names::ple_key(l), Plain::Q8_0, vec![]),
        "ple.value_proj.weight" if exp => one(names::ple_value(l), Plain::Q8_0, vec![]),
        "ple.norm_key.weight" if exp => {
            one(names::ple_norm_key(l), Plain::F32, vec![Step::PlusOne])
        }
        "ple.norm_query.weight" if exp => {
            one(names::ple_norm_query(l), Plain::F32, vec![Step::PlusOne])
        }
        "ple.norm_conv.weight" if exp => {
            one(names::ple_norm_conv(l), Plain::F32, vec![Step::PlusOne])
        }
        "ple.conv1d.weight" if exp => Ok(vec![(
            names::ple_conv1d(l),
            mapped(Plain::F32, vec![], taps()?),
        )]),
        _ => Err(refuse(name, format!("not a {} tensor", g.family.label()))),
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
            Step::JoinColumns => {
                let [_, matrix_rows, 2] = t.dims.as_slice() else {
                    return Err(refuse(name, "the input join needs two stacked matrices"));
                };
                let matrix_rows = usize::try_from(*matrix_rows)
                    .map_err(|_| refuse(name, "input join rows exceed usize"))?;
                let joined_width = t.dims[0]
                    .checked_mul(2)
                    .ok_or_else(|| refuse(name, "input join width overflows u64"))?;
                for r in 0..matrix_rows {
                    for part in 0..2 {
                        let src = (part * matrix_rows + r) * row;
                        let dst = (r * 2 + part) * row;
                        moved[dst..dst + row].copy_from_slice(&t.bytes[src..src + row]);
                    }
                }
                t.dims = vec![joined_width, matrix_rows as u64];
            }
            Step::Squeeze => {
                Squeeze.apply(t).map_err(|why| refuse(name, why))?;
                return Ok(());
            }
            Step::Rows(ref keep) => {
                if keep.end > rows {
                    return Err(refuse(name, format!("rows {keep:?} of {rows}")));
                }
                t.bytes = t.bytes[keep.start * row..keep.end * row].to_vec();
                t.dims = vec![width as u64, keep.len() as u64];
                return Ok(());
            }
            Step::PlusOne => {
                PlusOne.apply(t).map_err(|why| refuse(name, why))?;
                return Ok(());
            }
            Step::NegExp => {
                f32s(t)?;
                for (src, dst) in t
                    .bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(moved.as_chunks_mut::<4>().0)
                {
                    let y = super::neg_exp(f32::from_le_bytes(*src))
                        .map_err(|why| refuse(name, format!("{self:?}: {why}")))?;
                    *dst = y.to_le_bytes();
                }
            }
        }
        t.bytes = moved;
        Ok(())
    }
}

#[cfg(test)]
mod dense_tests;

#[cfg(test)]
mod hf_dense_tests;

#[cfg(test)]
mod tests {

    use gguf::GgmlType;
    use hf::modelopt;
    use hf::quant_config::{Config, plan};
    use safetensors::Dtype;

    use super::*;
    use crate::test_support::{
        Entries, Stored, bf16_tensor, entries, fixture, gguf_tensors, json, key_json,
        mlx_linear_pair, write_folder,
    };

    /// The next-token layers, the vision tower and the PLE hash constants are skipped with their
    /// reason; the PLE table's parts and every other tensor are the map's to read.
    #[test]
    fn the_tensors_the_map_leaves_out_are_named() {
        let text = Family::Qwen38.text();
        let ple = |p: &str| format!("{text}layers.1.ple.ple_embedding.{p}");
        for (family, name, why) in [
            (
                Family::Qwen38,
                "mtp.layers.0.mlp.gate.weight".to_string(),
                MTP,
            ),
            (
                Family::Qwen38,
                "model.visual.blocks.0.attn.qkv.weight".to_string(),
                VISION,
            ),
            (
                Family::Qwen36,
                "vision_tower.blocks.0.attn.qkv.weight".to_string(),
                VISION,
            ),
            (
                Family::Qwen38,
                ple("layer_multipliers"),
                "a PLE hash constant, read as a header key",
            ),
            (
                Family::Qwen38,
                ple("ngram_heads_offsets"),
                "a PLE hash constant, read as a header key",
            ),
            (
                Family::Qwen38,
                ple("ngram_heads_vocab_sizes"),
                "a PLE hash constant, read as a header key",
            ),
        ] {
            assert_eq!(skip_reason(family, &name), Some(why), "{name}");
        }
        for (family, name) in [
            (Family::Qwen38, ple("ngram_embedding.shard_0.weight")),
            (
                Family::Qwen38,
                format!("{text}layers.x.ple.ple_embedding.layer_multipliers"),
            ),
            (
                Family::Qwen38,
                format!("{text}layers.1.ple.key_proj.weight"),
            ),
            (Family::Qwen38, "lm_head.weight".to_string()),
            (
                Family::Qwen36,
                "model.visual.blocks.0.attn.qkv.weight".to_string(),
            ),
        ] {
            assert_eq!(skip_reason(family, &name), None, "{name}");
        }
    }

    pub(super) fn config() -> Json {
        json("quant-configs/nvidia__Qwen3.8-Flash-Next-NVFP4__config.json.gz")
    }

    /// [`config`] over [`tokenizer`]'s vocabulary: eight ids, `</s>` (id 2) the BOS and EOS,
    /// id 3 the image token.
    pub(super) fn small_config() -> Json {
        let mut c = config();
        c["text_config"]["vocab_size"] = 8.into();
        c["text_config"]["bos_token_id"] = 2.into();
        c["text_config"]["eos_token_id"] = 2.into();
        c["image_token_id"] = 3.into();
        c
    }

    /// The byte-level BPE tokenizer [`small_config`] reads, its one special token `</s>` id 2.
    pub(super) fn tokenizer() -> Json {
        crate::test_support::tokenizer("qwen35")
    }

    #[test]
    fn draft_mapping_joins_embedding_before_hidden_and_reuses_the_layer_map() {
        let c = Checkpoint::parse(&config()).unwrap();
        let entries = [
            ("mtp.fc_embedding.weight", vec![2, 2]),
            ("mtp.fc_hidden.weight", vec![2, 2]),
            ("mtp.layers.0.self_attn.q_norm.weight", vec![256]),
            ("mtp.pre_fc_norm_embedding.weight", vec![2560]),
            ("mtp.pre_fc_norm_hidden.weight", vec![10240]),
            (
                "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
                vec![4, 10240],
            ),
            ("mtp.hyper_connection_mixer.hc_norm.weight", vec![10240]),
            (
                "model.language_model.embed_tokens.weight",
                vec![248320, 2560],
            ),
        ];
        let mapped = map(
            &c.geometry,
            entries.iter().map(|(n, s)| (*n, s.as_slice())),
            true,
        )
        .unwrap();
        assert_eq!(mapped.len(), 6);
        for stem in ["enorm", "hnorm"] {
            assert_eq!(
                mapped[&format!("blk.{}.nextn.{stem}.weight", c.geometry.layers)].steps,
                [Step::PlusOne]
            );
        }
        let name = format!("blk.{}.nextn.eh_proj.weight", c.geometry.layers);
        let join = &mapped[&name];
        assert_eq!(
            join.sources,
            ["mtp.fc_embedding.weight", "mtp.fc_hidden.weight"]
        );
        let mut n = Normal {
            type_id: gguf::GgmlType::BF16.as_u32(),
            dims: vec![2, 2, 2],
            bytes: (0u16..8).flat_map(u16::to_le_bytes).collect(),
            scale: vec![],
        };
        join.steps[0].apply(&c.geometry, &name, &mut n).unwrap();
        assert_eq!(n.dims, join.dims);
        assert_eq!(
            n.bytes,
            [0u16, 1, 4, 5, 2, 3, 6, 7]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            mapped[&format!("blk.{}.attn_q_norm.weight", c.geometry.layers)].steps,
            [Step::PlusOne]
        );
        assert_eq!(
            mapped[&format!("blk.{}.hc_attn_inject.weight", c.geometry.layers)].role,
            Use::Matrix(Plain::F32)
        );
        assert!(
            map(
                &c.geometry,
                entries[..1].iter().map(|(n, s)| (*n, s.as_slice())),
                true
            )
            .unwrap_err()
            .to_string()
            .contains("both fc_embedding and fc_hidden")
        );
        assert!(
            map(
                &c.geometry,
                [("mtp.layers.1.self_attn.q_norm.weight", &[256][..])],
                true
            )
            .unwrap_err()
            .to_string()
            .contains("layer 1")
        );
        n.dims = vec![4, 2];
        assert!(
            Step::JoinColumns
                .apply(&c.geometry, &name, &mut n)
                .unwrap_err()
                .to_string()
                .contains("two stacked matrices")
        );
    }

    /// Layers 0, 1 (the PLE site) and 3 (attention) and every unlayered text tensor of
    /// nvidia/Qwen3.8-Flash-Next-NVFP4, placed by its quantization config, map to exactly the
    /// names and ggml dims unsloth's UD-Q4_K_XL holds for them, and every tensor that file keeps
    /// in Q8_0, F32 or BF16 gets that type from the table, but the router and the shared expert's
    /// gate, which keep the checkpoint's BF16 where that file widens them to F32; the routed
    /// stacks are NVFP4 with one scale an expert and the PLE table F8 rows.
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
            crate::arch::module_class,
        );
        let logical: Vec<(&str, Vec<usize>)> = entries
            .iter()
            .filter_map(|(n, _, s)| match p.placed.get(n) {
                Some(pl) => Some((n.as_str(), pl.format.logical(s))),
                None => p.refused.contains_key(n).then(|| (n.as_str(), s.clone())),
            })
            .collect();
        let g = Checkpoint::parse(&config()).unwrap().geometry;
        let mapped = map(&g, logical.iter().map(|(n, s)| (*n, s.as_slice())), false).unwrap();
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
            let router = name.ends_with(".ffn_gate_inp.weight")
                || name.ends_with(".ffn_gate_inp_shexp.weight");
            let expect = match ty {
                0 if router => 30,
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

    /// nvidia/Qwen3.8-Flash-Next-NVFP4's routed stacks (layers 0, 1 and 3) are kept for a load
    /// where the folder stores them, as NVFP4 rows planes with a tensor scale an expert; the
    /// split `prepare` writes makes them NVFP4, and so does a load of the stacks stored with
    /// `csf` block scales (FP4-Local's encoding), a form the host does not read in place.
    /// FAIL-first: the map makes every stack, so a load keeps none.
    #[test]
    fn a_load_keeps_plain_nvfp4_stacks_where_the_folder_stores_them() {
        use crate::folder::Make;
        let plain = entries(&json(
            "checkpoint-headers/qwen3.8-flash-next-nvfp4-layers-0-1-3.tensors.json.gz",
        ));
        let hq = json("quant-configs/nvidia__Qwen3.8-Flash-Next-NVFP4__hf_quant_config.json.gz");
        let mut csf_hq = hq.clone();
        for (module, entry) in csf_hq["quantization"]["quantized_layers"]
            .as_object_mut()
            .unwrap()
        {
            if module.starts_with("model.language_model.layers.") && module.ends_with(".experts") {
                entry["weight_scale_encoding"] = "csf".into();
            }
        }
        let csf: Entries = plain
            .iter()
            .flat_map(|(n, d, s)| match n.strip_suffix(".weight_scale") {
                Some(stem) if n.contains(".mlp.experts.") => {
                    let fixed = vec![s[0] / 16, 16 * (1 + s[1] / 2)];
                    vec![
                        (
                            format!("{stem}.weight_scale.nvfp4_csf_fixed"),
                            Dtype::U8,
                            fixed,
                        ),
                        (
                            format!("{stem}.weight_scale.nvfp4_csf_exceptions"),
                            Dtype::U32,
                            vec![1],
                        ),
                    ]
                }
                _ => vec![(n.clone(), *d, s.clone())],
            })
            .collect();
        let checkpoint = Checkpoint::parse(&config()).unwrap();
        let stacks = |hq: &Json, entries: &Entries, keep| {
            let quant = Config::parse(&config(), Some(hq)).unwrap();
            let (_, planned) = folder::plan_tensors(&checkpoint, &quant, entries, false, keep)
                .unwrap_or_else(|e| panic!("{e}"));
            planned
                .into_iter()
                .filter(|p| p.decl.name.ends_with("_exps.weight"))
                .map(|p| {
                    let Make::Tensor(m) = p.make else {
                        panic!("{}: a stack is a tensor", p.decl.name)
                    };
                    (p.decl.name, GgmlType::from_u32(p.decl.type_id), m.origin)
                })
                .collect::<Vec<_>>()
        };
        let kept = Origin::Stack {
            ty: GgmlType::NVFP4_ROWS,
            layout: StackLayout::Planes,
        };
        for (what, hq, entries, keep, want) in [
            (
                "load",
                &hq,
                &plain,
                folder::Keep::InPlace,
                (GgmlType::NVFP4_ROWS, kept),
            ),
            (
                "split",
                &hq,
                &plain,
                folder::Keep::Made,
                (GgmlType::NVFP4, Origin::Made),
            ),
            (
                "csf load",
                &csf_hq,
                &csf,
                folder::Keep::InPlace,
                (GgmlType::NVFP4, Origin::Made),
            ),
        ] {
            let got = stacks(hq, entries, keep);
            assert_eq!(got.len(), 9, "{what}: three layers' gate, up and down");
            for (name, ty, origin) in got {
                assert_eq!((ty, origin), want, "{what}: {name}");
            }
        }
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
        let dir = crate::test_support::temp_dir("qwen-prepare-inputs");
        let valid = small_config();
        let files = [
            ("config.json", valid.clone()),
            ("tokenizer.json", tokenizer()),
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
        let union = |a: &Fields, b: &Fields| -> Vec<(&'static str, schema::Rule)> {
            a.iter().chain(b).copied().collect()
        };
        let (root, text) = (
            union(ROOT_SHARED, ROOT_38),
            union(&union(TEXT_SHARED, SHARED_EXPERT), TEXT_38),
        );
        let objects = [
            ("", &root[..]),
            ("/text_config", &text[..]),
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
            ["/text_config/rope_parameters", "type", "default", "yarn", "type"],
            ["/text_config/rope_parameters", "mrope_interleaved", true, false, "mrope_interleaved"],
            ["/text_config/rope_parameters", "mrope_section", [11, 11, 10], [11, 11, 9], "mrope_section"],
            ["/text_config/rope_parameters", "rope_theta", 1e6, 1e300, "rope_theta"],
            ["/text_config/rope_parameters", "partial_rotary_factor", 0.25, 2, "partial_rotary_factor"]
        ]"#)
        .unwrap();
        // The counts: their own value, and zero.
        for (key, rule) in &text {
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
        let got: serde_json::Map<String, Json> =
            keys.iter().map(|(k, v)| (k.clone(), key_json(v))).collect();
        assert_eq!(Json::Object(got), want);
    }

    /// Each step against its definition on a small geometry: two K heads of three V heads each,
    /// so grouped head `i · 3 + j` goes to tiled place `j · 2 + i`.
    #[test]
    fn steps_move_v_heads_rows_and_values() {
        let g = Geometry {
            family: Family::Qwen38,
            layers: 1,
            full: vec![false],
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
        // The taps' unit dim: `[1, kernel, channels]` read as `[kernel, channels]`, bytes as they
        // are; a tensor with no unit dim, two of them or another rank is refused.
        let mut t = f32s(vec![1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Step::Squeeze.apply(&g, "t", &mut t).unwrap();
        assert_eq!(
            (t.dims.clone(), vals(&t)),
            (vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );
        for dims in [vec![2, 3], vec![1, 2, 3, 1], vec![1, 1, 6], vec![2, 2, 2]] {
            let mut bad = f32s(dims.clone(), &[0.0; 6]);
            let e = Step::Squeeze.apply(&g, "t", &mut bad).unwrap_err();
            assert!(e.to_string().contains("one unit dim"), "{dims:?}: {e}");
        }
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
        // Four V heads of one MXFP8 block each: the blocks move whole, each byte unchanged.
        let block = |h: u8| -> Vec<u8> { (0..33).map(|i| 33 * h + i).collect() };
        let mut x = Normal {
            type_id: gguf::quant::GgmlType::MXFP8.as_u32(),
            dims: vec![128, 1],
            bytes: (0..4).flat_map(block).collect(),
            scale: vec![],
        };
        Step::VCols { head: 32 }.apply(&g, "x", &mut x).unwrap();
        assert_eq!(
            x.bytes,
            [0, 2, 1, 3].into_iter().flat_map(block).collect::<Vec<_>>()
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
            safetensors::Safetensors::parse(safetensors::write(&list)).unwrap()
        };
        let hq = serde_json::json!({"quantization": {"quant_algo": "MIXED_PRECISION",
            "quantized_layers": {"e": {"quant_algo": "FP8"}}}});
        let m = Mapped {
            sources: vec!["e.shard_0.weight".into(), "e.shard_1.weight".into()],
            role: Use::Rows,
            steps: vec![],
            dims: vec![160, 4],
            origin: Origin::Made,
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
                crate::arch::module_class,
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

    /// The MLX type and role a GGUF fixture tensor of type `ty` becomes at the scale pair `pair`:
    /// the 4-bit linears are MLX_A4 (MLX_A4H at an F16 pair), read by Q8_0 roles but for the gated
    /// delta rule's alpha and beta, whose F32 role the map keeps; the router and the shared-expert
    /// gate are the 8-bit MLX_A8 (MLX_A8H at an F16 pair), read by the BF16 role the map gives a
    /// router; the BF16 vectors are F32.
    fn mlx_type_of(name: &str, ty: GgmlType, pair: Dtype) -> (GgmlType, Use) {
        let (a4, a8) = match pair {
            Dtype::Bf16 => (GgmlType::MLX_A4, GgmlType::MLX_A8),
            Dtype::F16 => (GgmlType::MLX_A4H, GgmlType::MLX_A8H),
            other => panic!("{name}: a pair the fixture is not written at: {other:?}"),
        };
        match ty {
            GgmlType::Q4_K if name.contains("ssm_alpha") || name.contains("ssm_beta") => {
                (a4, Use::Matrix(Plain::F32))
            }
            GgmlType::Q4_K | GgmlType::Q6_K => (a4, Use::Matrix(Plain::Q8_0)),
            GgmlType::F32 if name.contains("ffn_gate_inp") => (a8, Use::Matrix(Plain::Bf16)),
            GgmlType::F32 => (GgmlType::F32, Use::Matrix(Plain::F32)),
            other => panic!("{name}: {other}"),
        }
    }

    /// The fixture folder (the GGUF fixture's tensors as MLX's 4-bit checkpoint, its config and
    /// its tensors' headers committed) plans every tensor of the GGUF fixture: the same names and
    /// ggml dims, the router and the shared-expert gate 8-bit and every other linear 4-bit MLX
    /// blocks, the BF16 vectors F32; the bytes a planned tensor holds are the bytes its weight,
    /// scales and biases take in the folder, and nothing of the folder is left unplanned. Its
    /// header keys are the GGUF fixture's `qwen35moe.*` keys, the interval made from
    /// `layer_types`, which here (full attention at layers 1 and 3) disagrees with the config's
    /// own `full_attention_interval`.
    #[test]
    fn the_mlx_fixture_folder_plans_the_gguf_fixture_s_tensors() {
        let config = json("quant-configs/bloomery__qwen35moe-mlx-fixture__config.json.gz");
        let header = json("checkpoint-headers/qwen35moe-mlx-fixture.tensors.json.gz");
        let gguf = json("checkpoint-headers/qwen35moe-fixture-gguf.header.json.gz");
        let entries = entries(&header);
        let checkpoint = Checkpoint::parse(&config).unwrap();
        assert_eq!(checkpoint.geometry.family, Family::Qwen36);
        let (plan, planned) = folder::plan_tensors(
            &checkpoint,
            &Config::parse(&config, None).unwrap(),
            &entries,
            false,
            folder::Keep::InPlace,
        )
        .unwrap();
        assert!(plan.refused.is_empty(), "{:?}", plan.refused);
        let stored: BTreeMap<&str, usize> = entries
            .iter()
            .map(|(n, d, s)| (n.as_str(), s.iter().product::<usize>() * d.size()))
            .collect();
        let want = gguf_tensors(&gguf);
        let got: BTreeMap<&str, &folder::Planned> =
            planned.iter().map(|p| (p.decl.name.as_str(), p)).collect();
        assert_eq!(
            got.keys().collect::<Vec<_>>(),
            want.keys()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .iter()
                .collect::<Vec<_>>()
        );
        let mut held = 0;
        for (name, (dims, ty)) in &want {
            let planned = got[name.as_str()];
            let folder::Make::Tensor(m) = &planned.make else {
                panic!("{name}: a scale beside a type that reads none");
            };
            let (mlx, role) = mlx_type_of(name, *ty, Dtype::Bf16);
            assert_eq!(planned.decl.dims, *dims, "{name}");
            assert_eq!(GgmlType::from_u32(planned.decl.type_id), mlx, "{name}");
            assert_eq!(m.role, role, "{name}");
            assert!(m.steps.len() <= 2, "{name}: {:?}", m.steps);
            let bytes: usize = m
                .sources
                .iter()
                .flat_map(|s| {
                    let stem = s.strip_suffix(".weight").unwrap_or(s);
                    let own = [
                        s.clone(),
                        format!("{stem}.scales"),
                        format!("{stem}.biases"),
                    ];
                    own.into_iter()
                        .filter_map(|n| stored.get(n.as_str()).copied())
                        .collect::<Vec<_>>()
                })
                .sum();
            if matches!(mlx, GgmlType::MLX_A4 | GgmlType::MLX_A8) {
                assert_eq!(planned.decl.nbytes as usize, bytes, "{name}");
                held += bytes;
            }
        }
        // Every stored tensor but the BF16 vectors is a weight, scales or biases a planned tensor
        // holds.
        let vectors: usize = stored
            .iter()
            .filter(|(n, _)| !n.ends_with(".scales") && !n.ends_with(".biases"))
            .filter(|(n, _)| {
                entries
                    .iter()
                    .any(|(e, d, _)| e == **n && *d == Dtype::Bf16)
            })
            .map(|(_, bytes)| bytes)
            .sum();
        assert_eq!(held + vectors, stored.values().sum::<usize>());
        let keys = checkpoint.keys(&[], |_| unreachable!("no PLE")).unwrap();
        let got: serde_json::Map<String, Json> =
            keys.iter().map(|(k, v)| (k.clone(), key_json(v))).collect();
        let mut want = gguf["keys"].as_object().unwrap().clone();
        want.remove(gguf::GENERAL_ARCHITECTURE);
        assert_eq!(Json::Object(got), Json::Object(want));
    }

    /// The real checkpoint's names and shapes (layers 0 and 3, the unlayered tensors, the head
    /// and one vision pair of mlx-community's Qwen3.6-35B-A3B-4bit) map to the fixture's tensors
    /// at layers 0 (linear) and 3 (full): every text tensor, the vision pair left out; the file's
    /// interval is the config's 4.
    #[test]
    fn the_real_qwen36_headers_map_the_fixture_s_names() {
        let real = json("quant-configs/mlx-community__Qwen3.6-35B-A3B-4bit__config.json.gz");
        let header = json("checkpoint-headers/qwen3.6-mlx-4bit-layers-0-3.tensors.json.gz");
        let gguf = json("checkpoint-headers/qwen35moe-fixture-gguf.header.json.gz");
        let entries = entries(&header);
        let checkpoint = Checkpoint::parse(&real).unwrap();
        let (plan, planned) = folder::plan_tensors(
            &checkpoint,
            &Config::parse(&real, None).unwrap(),
            &entries,
            false,
            folder::Keep::InPlace,
        )
        .unwrap();
        assert!(plan.refused.is_empty(), "{:?}", plan.refused);
        // The fixture's layers 0 and 1 are the real file's 0 (linear) and 3 (full).
        let want: BTreeMap<String, (Vec<u64>, GgmlType)> = gguf_tensors(&gguf)
            .into_iter()
            .filter_map(|(n, t)| match n.strip_prefix("blk.") {
                None => Some((n, t)),
                Some(rest) if rest.starts_with("0.") => Some((n, t)),
                Some(rest) if rest.starts_with("1.") => Some((format!("blk.3.{}", &rest[2..]), t)),
                Some(_) => None,
            })
            .collect();
        let got: BTreeMap<&str, &folder::Planned> =
            planned.iter().map(|p| (p.decl.name.as_str(), p)).collect();
        assert_eq!(
            got.keys().copied().collect::<Vec<_>>(),
            want.keys().map(String::as_str).collect::<Vec<_>>()
        );
        for (name, (dims, ty)) in &want {
            assert_eq!(got[name.as_str()].decl.dims, *dims, "{name}");
            assert_eq!(
                GgmlType::from_u32(got[name.as_str()].decl.type_id),
                mlx_type_of(name, *ty, Dtype::Bf16).0,
                "{name}"
            );
        }
        // The real config's 16 K heads and 32 V heads, all of 128: the delta-rule matrices, taps
        // and vectors move their V heads (qkv and the taps after the 2 · 16 · 128 q and k rows), a
        // norm gamma is read as stored, `A_log` is `-exp`.
        let steps = |name: &str| -> Vec<Step> {
            let folder::Make::Tensor(m) = &got[name].make else {
                panic!("{name}: not a plain tensor");
            };
            m.steps.clone()
        };
        let rows = |from| vec![Step::VRows { from, head: 128 }];
        assert_eq!(steps("blk.0.attn_qkv.weight"), rows(4096));
        assert_eq!(
            steps("blk.0.ssm_conv1d.weight"),
            [
                Step::Squeeze,
                Step::VRows {
                    from: 4096,
                    head: 128
                }
            ]
        );
        assert_eq!(steps("blk.0.attn_gate.weight"), rows(0));
        assert_eq!(
            steps("blk.0.ssm_alpha.weight"),
            [Step::VRows { from: 0, head: 1 }]
        );
        assert_eq!(steps("blk.0.ssm_out.weight"), [Step::VCols { head: 128 }]);
        assert_eq!(
            steps("blk.0.ssm_a"),
            [Step::VCols { head: 1 }, Step::NegExp]
        );
        assert_eq!(steps("blk.0.ssm_dt.bias"), [Step::VCols { head: 1 }]);
        for name in [
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.post_attention_norm.weight",
            "blk.0.ssm_norm.weight",
            "blk.3.attn_q_norm.weight",
            "blk.3.attn_k_norm.weight",
        ] {
            assert_eq!(steps(name), [], "{name}");
        }
        let keys = checkpoint.keys(&[], |_| unreachable!("no PLE")).unwrap();
        let interval = keys
            .iter()
            .find(|(k, _)| k == "qwen35moe.full_attention_interval")
            .map(|(_, v)| v);
        assert_eq!(interval, Some(&Value::U32(4)));
        let (_, rope) = &keys
            .iter()
            .find(|(k, _)| k == "qwen35moe.rope.dimension_sections")
            .unwrap();
        assert_eq!(
            key_json(rope),
            serde_json::json!(["ARR", ["I32", [11, 11, 10, 0]]])
        );
    }

    /// What the Qwen3.6 map refuses, by name: a tensor no row names, a weight without its scales, a
    /// mixer the layer's `layer_types` does not give it, a stack of another expert count,
    /// conv taps in the Qwen3.8 order; a config of another family's key, another `model_type`
    /// and a `layer_types` no interval makes.
    #[test]
    fn the_qwen36_map_refuses_what_it_does_not_read_by_name() {
        let config = json("quant-configs/bloomery__qwen35moe-mlx-fixture__config.json.gz");
        let header = json("checkpoint-headers/qwen35moe-mlx-fixture.tensors.json.gz");
        let base = entries(&header);
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let quant = Config::parse(&config, None).unwrap();
        let refuse = |edit: &dyn Fn(&mut Entries)| {
            let mut e = base.clone();
            edit(&mut e);
            folder::plan_tensors(&checkpoint, &quant, &e, false, folder::Keep::InPlace)
                .err()
                .expect("refused")
                .to_string()
        };
        let l = "language_model.model.layers";
        let at = |e: &mut Entries, name: &str| e.iter().position(|(n, _, _)| n == name).unwrap();
        let e = refuse(&|e| {
            e.push((format!("{l}.0.mlp.zzz.weight"), Dtype::Bf16, vec![8]));
        });
        assert!(
            e.contains(&format!("{l}.0.mlp.zzz.weight")) && e.contains("not a Qwen3.6 tensor"),
            "{e}"
        );
        let e = refuse(&|e| {
            for suffix in ["scales", "biases"] {
                let i = at(e, &format!("{l}.0.mlp.gate.{suffix}"));
                e.remove(i);
            }
        });
        assert!(
            e.contains(&format!("{l}.0.mlp.gate.weight")) && e.contains("fits none of"),
            "{e}"
        );
        // A weight whose biases are gone is refused too, by the first tensor of the pair it
        // meets.
        let e = refuse(&|e| {
            let i = at(e, &format!("{l}.0.mlp.gate.scales"));
            e.remove(i);
        });
        assert!(e.contains(&format!("{l}.0.mlp.gate.")), "{e}");
        let e = refuse(&|e| {
            let from = format!("{l}.1.self_attn.k_proj");
            for (n, _, _) in e.iter_mut().filter(|(n, _, _)| n.starts_with(&from)) {
                *n = n.replace(".1.self_attn.k_proj", ".0.self_attn.k_proj");
            }
        });
        assert!(
            e.contains(&format!("{l}.0.self_attn.k_proj.weight"))
                && e.contains("layer 0 is a delta-rule layer by layer_types"),
            "{e}"
        );
        let e = refuse(&|e| {
            let i = at(e, &format!("{l}.0.linear_attn.A_log"));
            let (n, d, s) = e.remove(i);
            e.push((n.replace(".0.", ".1."), d, s));
        });
        assert!(
            e.contains(&format!("{l}.1.linear_attn.A_log"))
                && e.contains("layer 1 attends by layer_types"),
            "{e}"
        );
        let e = refuse(&|e| {
            let i = at(e, &format!("{l}.0.mlp.switch_mlp.gate_proj.weight"));
            e[i].2[0] = 255;
            for suffix in ["scales", "biases"] {
                let i = at(e, &format!("{l}.0.mlp.switch_mlp.gate_proj.{suffix}"));
                e[i].2[0] = 255;
            }
        });
        assert!(
            e.contains(&format!("{l}.0.mlp.switch_mlp.gate_proj.weight"))
                && e.contains("is not [256 experts, rows, columns]"),
            "{e}"
        );
        let e = refuse(&|e| {
            let i = at(e, &format!("{l}.0.linear_attn.conv1d.weight"));
            e[i].2 = vec![8192, 1, 4];
        });
        assert!(
            e.contains(&format!("{l}.0.linear_attn.conv1d.weight"))
                && e.contains("are not [channels, kernel, 1]"),
            "{e}"
        );
        let parse = |edit: &dyn Fn(&mut Json)| {
            let mut c = config.clone();
            edit(&mut c);
            Checkpoint::parse(&c).expect_err("refused").to_string()
        };
        let e = parse(&|c| c["text_config"]["hc_count"] = serde_json::json!(4));
        assert!(
            e.contains("text_config.hc_count is not a key this converter reads"),
            "{e}"
        );
        let e = parse(&|c| c["model_type"] = serde_json::json!("qwen4_exp"));
        assert!(e.contains("config.model_type"), "{e}");
        let e = parse(&|c| c["text_config"]["attn_output_gate"] = serde_json::json!(false));
        assert!(e.contains("attn_output_gate"), "{e}");
        let e = parse(&|c| {
            c["text_config"]["layer_types"] = serde_json::json!([
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention"
            ]);
        });
        assert!(
            e.contains("layer_types is not one full-attention layer in every N"),
            "{e}"
        );
        let e = parse(&|c| c["architectures"] = serde_json::json!(["Qwen3_5MoeForCausalLM"]));
        assert!(e.contains("architectures"), "{e}");
        let want = [
            (&[false, false, false, true][..], Some(4)),
            (&[false, true, false, true], Some(2)),
            (&[false, false], Some(3)),
            (&[true, true], Some(1)),
            (&[true, false, false], None),
            (&[false, true, true], None),
        ];
        for (full, interval) in want {
            assert_eq!(attention_interval(full), interval, "{full:?}");
        }
    }

    // A fixture tensor's groups pinned: `language_model.model.layers.1.self_attn.k_proj` of the
    // fixture folder (the GGUF fixture's `blk.1.attn_k.weight` as the maker quantized it),
    // rows 0 and 1, groups 0 and 1 of 32, the words, scales and biases as stored and the BF16
    // bits of `mx.dequantize(words, scales, biases, group_size=64, bits=4)` over them (pip mlx
    // 0.32.1).
    const FX_WORDS: [u32; 32] = [
        0x374c884e, 0xceb49b66, 0x37de22dd, 0x4836caac, 0xb4ffaa5f, 0x7f29a5b6, 0x71f7b63c,
        0x400a7a2f, 0x88888888, 0x88888888, 0x88888888, 0x88888888, 0x90f09065, 0xdf9b2eb0,
        0x713cf52d, 0x05f7c72a, 0xd3692638, 0x76888d4a, 0xdc2ae8b8, 0x2d88432b, 0xa5c5a910,
        0x6ea113f7, 0xa1a1daf1, 0x6ab571a3, 0xaad74334, 0xd43594b3, 0xc5a5b3dc, 0x98dd6465,
        0x0513fc5d, 0xf4a74fd9, 0xf944704b, 0x77f9a199,
    ];
    const FX_SCALES: [u16; 4] = [0xbbce, 0xb9c2, 0x3bad, 0xbb89];
    const FX_BIASES: [u16; 4] = [0x3d4e, 0x3b42, 0xbd2d, 0x3d09];
    const FX_MX_DEQ: [u16; 256] = [
        0xbd1a, 0x3cce, 0x0000, 0x0000, 0xbcce, 0x3cce, 0x3bce, 0x3d01, 0x3c4e, 0x3c4e, 0xbc9a,
        0xbbce, 0x3cce, 0xbc9a, 0xbd1a, 0xbcce, 0xbd01, 0xbd01, 0x3d1a, 0x3d1a, 0xbd1a, 0xbd01,
        0x3bce, 0x3d01, 0xbcce, 0xbc4e, 0xbc4e, 0xbcce, 0x3c4e, 0x3d01, 0x0000, 0x3cce, 0xbd34,
        0x3c9a, 0xbc4e, 0xbc4e, 0xbd34, 0xbd34, 0x3cce, 0xbc9a, 0x3c4e, 0xbc9a, 0x3c9a, 0xbc4e,
        0xbbce, 0x3d1a, 0xbd34, 0x3bce, 0xbcce, 0x3d01, 0x3c4e, 0xbc9a, 0x3bce, 0xbd34, 0x3d34,
        0x3bce, 0xbd34, 0x3d1a, 0xbc4e, 0x3bce, 0xbc4e, 0x3d4e, 0x3d4e, 0x3cce, 0x0000, 0x0000,
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
        0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x3a92, 0x3a42, 0x3b42,
        0xb9c2, 0x3b42, 0xbb2a, 0x3b42, 0xb9c2, 0x3b42, 0xba92, 0xbb12, 0x3b12, 0xba92, 0xb9c2,
        0xbb2a, 0xbaf2, 0xbaf2, 0x3b12, 0x3a92, 0xbb2a, 0xbac2, 0x3af2, 0x3b2a, 0x39c2, 0xba42,
        0x3b12, 0x39c2, 0xbac2, 0x39c2, 0xbb2a, 0x3a92, 0x3b42, 0x0000, 0xbcd8, 0xbc2d, 0xbd02,
        0x3bad, 0xbc2d, 0xbcd8, 0x3cd8, 0x3c2d, 0xbcad, 0x3cd8, 0x0000, 0x0000, 0x0000, 0xbc2d,
        0xbbad, 0x0000, 0x3c82, 0x0000, 0x3d02, 0x3c2d, 0xbd02, 0x3cad, 0x3cd8, 0x3c82, 0xbd02,
        0xbcd8, 0xbcad, 0x0000, 0x0000, 0x3cd8, 0xbd02, 0xbd2d, 0xbd17, 0x3bad, 0x3c2d, 0xbc82,
        0x3cad, 0xbc82, 0x3c2d, 0xbbad, 0x3d17, 0xbcd8, 0xbd17, 0xbd17, 0x3c2d, 0x3d02, 0xbc2d,
        0xbd17, 0x3d17, 0x3c2d, 0x3cd8, 0xbd17, 0x3c2d, 0xbd17, 0x3c2d, 0xbcd8, 0x3c2d, 0xbd17,
        0xbbad, 0xbc82, 0x3c82, 0x3c2d, 0xbc2d, 0x3c89, 0x3cab, 0x3cab, 0x3c89, 0x3b89, 0xbcab,
        0xbc09, 0xbc09, 0x3cab, 0xbc4e, 0x3c89, 0xbb89, 0x3c4e, 0x3cab, 0x3c89, 0xbcab, 0xbc89,
        0xbcab, 0x3cab, 0xbc4e, 0x3c4e, 0xbc09, 0x3c4e, 0xbc89, 0x3c4e, 0x3c09, 0x3c89, 0x3c09,
        0xbcab, 0xbcab, 0x0000, 0xbb89, 0xbcab, 0x3c4e, 0xbc89, 0xbcf0, 0x3cab, 0x3cf0, 0x3c4e,
        0x3d09, 0xbb89, 0xbcab, 0xbcf0, 0x3c89, 0x3b89, 0xbc09, 0x3c89, 0xbcf0, 0xbc4e, 0x3c89,
        0x3d09, 0x3b89, 0x3c89, 0x3c89, 0xbb89, 0xbcf0, 0xbb89, 0xbb89, 0x3cf0, 0xbc09, 0xbb89,
        0xbcf0, 0x3b89, 0x3b89,
    ];

    /// A tensor of the fixture folder, repacked by the map's own path (the plan places it, the map
    /// makes `blk.1.attn_k.weight` of it, `normalize` writes its blocks) and decoded by the GGUF
    /// type's decode, is the fixture maker's check rule bit for bit: each value is `s · q + b` in
    /// f32 from the stored words and pair, and rounded once to BF16 it is what `mx.dequantize`
    /// returned.
    #[test]
    fn a_fixture_tensor_repacks_to_the_makers_rule() {
        let config = json("quant-configs/bloomery__qwen35moe-mlx-fixture__config.json.gz");
        let stem = "language_model.model.layers.1.self_attn.k_proj";
        let le = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let words: Vec<u8> = FX_WORDS.iter().flat_map(|w| w.to_le_bytes()).collect();
        let st = safetensors::Safetensors::parse(safetensors::write(&[
            (&format!("{stem}.weight"), "U32", &[2, 16], &words),
            (&format!("{stem}.scales"), "BF16", &[2, 2], &le(&FX_SCALES)),
            (&format!("{stem}.biases"), "BF16", &[2, 2], &le(&FX_BIASES)),
        ]))
        .unwrap();
        let entries: Vec<(String, Dtype, Vec<usize>)> = st
            .names()
            .map(|n| {
                let r = st.raw(n).unwrap();
                (n.to_string(), r.dtype, r.shape.to_vec())
            })
            .collect();
        let p = plan(
            &Config::parse(&config, None).unwrap(),
            entries
                .iter()
                .map(|(n, d, s)| (n.as_str(), *d, s.as_slice())),
            crate::arch::module_class,
        );
        let name = format!("{stem}.weight");
        assert!(p.refused.is_empty(), "{:?}", p.refused);
        let g = Checkpoint::parse(&config).unwrap().geometry;
        let mapped = map(&g, [(name.as_str(), &[2usize, 128][..])], false).unwrap();
        let m = &mapped["blk.1.attn_k.weight"];
        let n = modelopt::normalize(|n| st.raw(n), &m.placed(&p).unwrap(), m.role).unwrap();
        assert_eq!(n.type_id, GgmlType::MLX_A4.as_u32());
        assert_eq!(n.dims, [128, 2]);
        let mut got = vec![0.0f32; 256];
        gguf::quant::dequant_row(GgmlType::MLX_A4, &n.bytes, &mut got).unwrap();
        for (i, v) in got.iter().enumerate() {
            let (group, at) = (i / 64, i % 64);
            let code = (FX_WORDS[group * 8 + at / 8] >> (4 * (at % 8))) & 0xf;
            let rule = gguf::quant::bf16_bits_to_f32(FX_SCALES[group]) * code as f32
                + gguf::quant::bf16_bits_to_f32(FX_BIASES[group]);
            assert_eq!(v.to_bits(), rule.to_bits(), "value {i}");
            assert_eq!(
                gguf::quant::fp32_to_bf16_bits(*v),
                FX_MX_DEQ[i],
                "value {i}: mx.dequantize's BF16"
            );
        }
    }

    /// A folder of two layers (gated delta rule, then attention) of a Qwen3.6 as small as the
    /// groups allow, written in MLX's names at both scale pairs the map reads (BF16's
    /// MLX_A4/MLX_A8 and F16's MLX_A4H/MLX_A8H), opens: its planned tensors and header keys read as
    /// a qwen35moe file through the engine's own header reader (kinds, interval), every tensor of
    /// the source reads whole in the planned bytes, and the steps ran on the stored values
    /// (`A_log` as `-exp`; the conv taps as `[kernel, channels]`; the delta-rule matrices, taps and
    /// vectors with their V heads from grouped by K head to tiled; the norm gammas as stored, they
    /// carry the one already).
    #[test]
    fn a_small_qwen36_folder_opens_as_a_qwen35moe_file() {
        // Both scale pairs the folder map reads: the BF16 pairs of the real
        // 4-bit folders and their F16 forms.
        for (pair, a4, a8, pair_bytes) in [
            (
                Dtype::Bf16,
                GgmlType::MLX_A4,
                GgmlType::MLX_A8,
                [0x00u8, 0x3f, 0x80, 0xbe],
            ),
            (
                Dtype::F16,
                GgmlType::MLX_A4H,
                GgmlType::MLX_A8H,
                [0x00u8, 0x38, 0x00, 0xb4],
            ),
        ] {
            let lin = |stem: &str, bits: usize, lead: &[usize], k: usize| {
                mlx_linear_pair(stem, bits, lead, k, pair)
            };

            use model::arch::qwen35moe::hparams::{Hparams, Kind};
            use model::part::{Source, Take};
            use serde_json::json;

            let p = "language_model.model";
            let mut gates = serde_json::Map::new();
            for l in 0..2 {
                for gate in ["mlp.gate", "mlp.shared_expert_gate"] {
                    gates.insert(
                        format!("{p}.layers.{l}.{gate}"),
                        json!({"group_size": 64, "bits": 8}),
                    );
                }
            }
            let mut quantization = json!({"group_size": 64, "bits": 4, "mode": "affine"});
            quantization.as_object_mut().unwrap().extend(gates);
            let config = json!({
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "model_type": "qwen3_5_moe",
                "eos_token_id": [2],
                "image_token_id": 3,
                "tie_word_embeddings": false,
                "quantization": quantization,
                "quantization_config": quantization,
                "text_config": {
                    "model_type": "qwen3_5_moe_text", "attn_output_gate": true,
                    "num_hidden_layers": 2, "max_position_embeddings": 4096, "hidden_size": 64,
                    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
                    "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 64,
                    "shared_expert_intermediate_size": 64, "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 64, "linear_num_key_heads": 2,
                    "linear_num_value_heads": 6, "linear_value_head_dim": 64,
                    "layer_types": ["linear_attention", "full_attention"],
                    "full_attention_interval": 4, "rms_norm_eps": 1e-6, "vocab_size": 8,
                    "bos_token_id": 2, "eos_token_id": 2, "pad_token_id": null,
                    "partial_rotary_factor": 0.5, "hidden_act": "silu",
                    "rope_parameters": {"mrope_interleaved": true, "mrope_section": [4, 2, 2],
                        "rope_theta": 1e6, "rope_type": "default", "partial_rotary_factor": 0.5},
                },
            });
            let dir = crate::test_support::temp_dir("qwen36-open");
            let mut t: Vec<Stored> = vec![];
            for stem in [
                format!("{p}.embed_tokens"),
                "language_model.lm_head".to_string(),
            ] {
                t.extend(lin(&stem, 4, &[8], 64));
            }
            t.push(bf16_tensor(&format!("{p}.norm.weight"), &[64], |_| 0x3fc0));
            for l in 0..2 {
                let m = format!("{p}.layers.{l}");
                t.push(bf16_tensor(
                    &format!("{m}.input_layernorm.weight"),
                    &[64],
                    |_| 0x3fc0,
                ));
                t.push(bf16_tensor(
                    &format!("{m}.post_attention_layernorm.weight"),
                    &[64],
                    |_| 0x3f80,
                ));
                t.extend(lin(&format!("{m}.mlp.gate"), 8, &[4], 64));
                t.extend(lin(&format!("{m}.mlp.shared_expert_gate"), 8, &[1], 64));
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    t.extend(lin(&format!("{m}.mlp.switch_mlp.{proj}"), 4, &[4, 64], 64));
                    t.extend(lin(&format!("{m}.mlp.shared_expert.{proj}"), 4, &[64], 64));
                }
            }
            let m = format!("{p}.layers.0.linear_attn");
            for (stem, rows, k) in [
                ("in_proj_qkv", 640, 64),
                ("in_proj_z", 384, 64),
                ("in_proj_a", 6, 64),
                ("in_proj_b", 6, 64),
                ("out_proj", 64, 384),
            ] {
                t.extend(lin(&format!("{m}.{stem}"), 4, &[rows], k));
            }
            t.push(bf16_tensor(
                &format!("{m}.conv1d.weight"),
                &[640, 4, 1],
                |i| 0x3f80 + (i % 7) as u16,
            ));
            t.push(bf16_tensor(&format!("{m}.A_log"), &[6], |i| {
                [0, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0][i]
            }));
            t.push(bf16_tensor(&format!("{m}.dt_bias"), &[6], |i| {
                0x3f80 + i as u16
            }));
            t.push(bf16_tensor(&format!("{m}.norm.weight"), &[64], |_| 0x3f80));
            let m = format!("{p}.layers.1.self_attn");
            for (stem, rows, k) in [
                ("q_proj", 128, 64),
                ("k_proj", 32, 64),
                ("v_proj", 32, 64),
                ("o_proj", 64, 64),
            ] {
                t.extend(lin(&format!("{m}.{stem}"), 4, &[rows], k));
            }
            t.push(bf16_tensor(&format!("{m}.q_norm.weight"), &[32], |_| {
                0x3e80
            }));
            t.push(bf16_tensor(&format!("{m}.k_norm.weight"), &[32], |_| {
                0x3e80
            }));
            t.push(bf16_tensor(
                "vision_tower.patch_embed.proj.weight",
                &[2, 3],
                |_| 0x3f80,
            ));
            write_folder(&dir, &t, &config, &tokenizer());
            let opened = crate::folder::open(&dir).unwrap();
            let hp = Hparams::read(&opened.header).unwrap();
            assert_eq!(hp.kinds, [Kind::DeltaRule, Kind::Attention]);
            assert_eq!(hp.interval, 2);
            assert!(hp.exp.is_none());
            let ty = |name: &str| opened.header.find(name).unwrap().1.ty;
            assert_eq!(ty("blk.0.attn_qkv.weight"), a4);
            assert_eq!(ty("blk.1.ffn_down_exps.weight"), a4);
            assert_eq!(ty("blk.1.ffn_gate_inp.weight"), a8);
            assert_eq!(ty("blk.0.ssm_a"), GgmlType::F32);
            assert_eq!(ty("output_norm.weight"), GgmlType::F32);
            assert!(
                opened
                    .header
                    .find("vision_tower.patch_embed.proj.weight")
                    .is_none()
            );
            let planned = opened
                .source
                .plan(|_, _, _| Ok::<_, String>(Take::Whole))
                .unwrap();
            assert_eq!(planned.len(), opened.header.iter_tensors().count());
            let read = |name: &str| -> Vec<u8> {
                let p = planned.iter().find(|p| p.info.name == name).unwrap();
                let part = opened.source.read(p).unwrap();
                assert_eq!(part.spans().concat().len() as u64, p.info.nbytes, "{name}");
                part.spans().concat()
            };
            let f32s = |name: &str| -> Vec<f32> {
                read(name)
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect()
            };
            for p in &planned {
                assert_eq!(read(&p.info.name).len() as u64, p.info.nbytes);
            }
            // 6 V heads of 2 K heads: grouped head `h` goes to place `(h % 3) · 2 + h / 3`, so the
            // places 0..6 hold the grouped heads 0, 3, 1, 4, 2, 5.
            let held = [0, 3, 1, 4, 2, 5];
            // `A_log` of 0..6 as `-exp`, and `dt_bias` of 1 + h/128 as stored, at those places.
            let neg_exp = |x: f64| (-x.exp()) as f32;
            assert_eq!(
                f32s("blk.0.ssm_a"),
                held.map(|h| neg_exp(h as f64)),
                "A_log"
            );
            assert_eq!(
                f32s("blk.0.ssm_dt.bias"),
                held.map(|h| 1.0 + h as f32 / 128.0),
                "dt_bias"
            );
            assert_eq!(f32s("blk.1.attn_q_norm.weight"), [0.25; 32]);
            assert_eq!(f32s("blk.1.attn_k_norm.weight"), [0.25; 32]);
            assert_eq!(f32s("blk.0.attn_norm.weight"), [1.5; 64]);
            assert_eq!(f32s("blk.0.post_attention_norm.weight"), [1.0; 64]);
            assert_eq!(f32s("output_norm.weight"), [1.5; 64]);
            // The stored row a planned row holds, for V heads of `head` rows from row `from`.
            let from_row = |row: usize, from: usize, head: usize| {
                if row < from {
                    row
                } else {
                    let at = row - from;
                    from + held[at / head] * head + at % head
                }
            };
            let taps = f32s("blk.0.ssm_conv1d.weight");
            assert_eq!(taps.len(), 4 * 640);
            for (row, tap) in (0..640).flat_map(|r| (0..4).map(move |t| (r, t))) {
                let stored = 4 * from_row(row, 256, 64) + tap;
                let want = f32::from_bits(u32::from(0x3f80 + (stored % 7) as u16) << 16);
                assert_eq!(taps[4 * row + tap], want, "tap {tap} of channel {row}");
            }
            // The matrices' rows are the stored rows moved whole: a row's first group is its stored
            // words, then its scale and its bias (0.5, -0.25).
            let words = |row: usize, words_a_row: usize, group: usize| -> Vec<u8> {
                let at = (row * words_a_row + group * 8) as u32;
                (at..at + 8)
                    .flat_map(|i| i.wrapping_mul(0x9e37_79b9).to_le_bytes())
                    .collect()
            };
            for (name, rows, from, head) in [
                ("attn_qkv", 640, 256, 64),
                ("attn_gate", 384, 0, 64),
                ("ssm_alpha", 6, 0, 1),
                ("ssm_beta", 6, 0, 1),
            ] {
                let m = read(&format!("blk.0.{name}.weight"));
                assert_eq!(m.len(), rows * 36, "{name}");
                for row in 0..rows {
                    let at = row * 36;
                    let stored = from_row(row, from, head);
                    assert_eq!(m[at..][..32], words(stored, 8, 0)[..], "{name} row {row}");
                    assert_eq!(m[at + 32..][..4], pair_bytes, "{name} row {row}");
                }
            }
            // The out projection moves columns: each row's 6 groups of 64 are the V heads.
            let out = read("blk.0.ssm_out.weight");
            assert_eq!(out.len(), 64 * 6 * 36);
            for row in [0, 1, 63] {
                for (group, &stored) in held.iter().enumerate() {
                    let at = (row * 6 + group) * 36;
                    assert_eq!(
                        out[at..][..32],
                        words(row, 48, stored)[..],
                        "out_proj row {row} group {group}"
                    );
                }
            }
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }
}
