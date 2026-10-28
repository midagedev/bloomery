//! Nemotron-H's ModelOpt checkpoint folder (`nemotron_h`: Nemotron-3.5-Lightning-30B-A3B-NVFP4)
//! in the `nemotron_h_moe` names, layouts and keys the program reads, as llama.cpp's converter
//! writes them into a GGUF, with the formats left as the folder stores them.
//!
//! A layer is Mamba-2, attention (no position) or a feed-forward block, as the config's layer
//! kinds say ([`Layers`]); a model with a routed layer is `nemotron_h_moe`, one with none is
//! `nemotron_h`, which this map refuses until its tensors are mapped. A Mamba-2 projection is FP8
//! with one tensor scale (FP8_B128), a layer's routed experts stay as one stack of NVFP4 planes a
//! projection ([`Origin::Stack`]), the shared experts and the head are NVFP4, the embedding and
//! the attention projections are BF16 their role reads as Q8_0, and the router and its bias stay
//! F32 in the folder's bytes. The norm gains, the convolution and the scan's constants are F32:
//! the convolution's taps and the grouped norm's gain change their dims and no byte, `D` too, and
//! `A` is stored as its log ([`Step::NegExp`]).
//!
//! The next-token block (`mtp.*`) and the FP8 KV-cache scales are left out with their reason. A
//! vision-language wrapper (Super's and Nano's `language_model.` tensors), a model of another
//! trunk (`nemotron_h_puzzle`), the dense layers' and the latent MoE's tensors and any other name
//! are refused by name; so is a config value the program does not run.

use std::collections::BTreeMap;

use gguf::{GgmlType, StackLayout, Value};
use hf::modelopt::{Normal, Plain, Use};
use hf::quant_config::ModuleClass;
use hf::schema::{Fields, Rule::*, Type};
use model::arch::names::{blk, output, output_norm, token_embd};
use model::arch::nemotron_h_moe::names::{
    ATTN_K, ATTN_NORM, ATTN_OUTPUT, ATTN_Q, ATTN_V, EXP_PROBS_B, FFN_DOWN_EXPS, FFN_DOWN_SHEXP,
    FFN_GATE_INP, FFN_UP_EXPS, FFN_UP_SHEXP, SSM_A, SSM_CONV, SSM_CONV_BIAS, SSM_D, SSM_DT_BIAS,
    SSM_IN, SSM_NORM, SSM_OUT,
};
use safetensors::Folder;
use serde_json::Value as Json;

use super::section::Section;
use super::trunk;
use crate::folder::{Error, FolderMap, Mapped, Mapping, Origin, VocabSpec, refuse};
use crate::part;

/// The `model_type`s of a vision-language wrapper around the text model, whose `llm_config` holds
/// it.
const WRAPPERS: &[&str] = &["nemotron_h_omni", "NemotronH_Nano_VL_V2"];

const NEXT_TOKEN: &str = "the next-token block, which the trunk never reads";
const KV_SCALE: &str = "the scale of an FP8 KV cache, which this program does not use";

/// The tensors of a layer past `layers.N.` that the dense layers (Nano) and the latent MoE (Super)
/// carry, whose tables are not mapped yet.
const LATER: &[&str] = &[
    "mixer.up_proj.weight",
    "mixer.down_proj.weight",
    "mixer.fc1_latent_proj.weight",
    "mixer.fc2_latent_proj.weight",
];

const DTYPE: &str = "the dtype HF loads in; each tensor keeps its stored dtype and format";
const TRAINING: &str = "dropout acts in training only";
const INIT: &str = "initializes weights the checkpoint holds";

/// `config.json` of the text model. A setting the program computes with one value, and GGUF has
/// no key for, is `Only` that value: another would be served as this one.
const ROOT: &Fields = &[
    ("architectures", Need(r#"[["NemotronHForCausalLM"]]"#)),
    ("model_type", Need(r#"["nemotron_h"]"#)),
    // `hf::quant_config` reads the quantization.
    ("quantization_config", Read),
    // The layer kinds, spelled by name or as HF's pattern string.
    ("layers_block_type", Read),
    ("hybrid_override_pattern", Read),
    ("num_hidden_layers", Read),
    ("max_position_embeddings", Read),
    ("hidden_size", Read),
    ("intermediate_size", Read),
    ("num_attention_heads", Read),
    ("num_key_value_heads", Read),
    ("head_dim", Read),
    ("partial_rotary_factor", Read),
    ("rope_theta", Read),
    ("norm_eps", Read),
    ("layer_norm_epsilon", Read),
    ("mamba_num_heads", Read),
    ("mamba_head_dim", Read),
    ("n_groups", Read),
    ("conv_kernel", Read),
    ("ssm_state_size", Read),
    ("n_routed_experts", Read),
    ("num_experts_per_tok", Read),
    ("moe_intermediate_size", Read),
    ("moe_shared_expert_intermediate_size", Read),
    ("moe_latent_size", Read),
    ("n_group", Read),
    ("topk_group", Read),
    ("norm_topk_prob", Read),
    ("routed_scaling_factor", Read),
    // The next-token block's description: its tensors are left out by name.
    ("num_nextn_predict_layers", Read),
    ("mtp_layers_block_type", Read),
    // hf::vocab::Config's.
    ("vocab_size", Read),
    ("bos_token_id", Read),
    ("eos_token_id", Read),
    ("pad_token_id", Read),
    ("unk_token_id", Read),
    ("sep_token_id", Read),
    // One shared expert on every routed layer.
    ("n_shared_experts", Need("[1]")),
    ("mlp_hidden_act", Only(r#"["relu2"]"#)),
    ("mamba_hidden_act", Only(r#"["silu"]"#)),
    ("mamba_ssm_cache_dtype", Only(r#"["float32"]"#)),
    ("residual_in_fp32", Only("[false]")),
    ("attention_bias", Only("[false]")),
    ("mlp_bias", Only("[false]")),
    ("mamba_proj_bias", Only("[false]")),
    ("use_bias", Only("[false]")),
    ("use_conv_bias", Only("[true]")),
    ("tie_word_embeddings", Only("[false]")),
    ("sliding_window", Only("[null]")),
    ("dtype", Ignore(Type::String, DTYPE)),
    (
        "transformers_version",
        Ignore(Type::String, "the version that wrote the file"),
    ),
    ("initializer_range", Ignore(Type::Number, INIT)),
    ("attention_dropout", Ignore(Type::Number, TRAINING)),
    ("hidden_dropout", Ignore(Type::Number, TRAINING)),
    ("rescale_prenorm_residual", Ignore(Type::Bool, INIT)),
    (
        "use_cache",
        Ignore(
            Type::Bool,
            "whether HF keeps its cache; outputs are the same",
        ),
    ),
    (
        "use_mamba_kernels",
        Ignore(Type::Bool, "picks HF's kernels, not a result"),
    ),
    (
        "num_logits_to_keep",
        Ignore(Type::Number, "how many logits HF's forward returns"),
    ),
    (
        "chunk_size",
        Ignore(
            Type::Number,
            "the scan's chunk length; the recurrence is the same at any chunking",
        ),
    ),
    (
        "expand",
        Ignore(
            Type::Number,
            "the inner width is mamba_num_heads times mamba_head_dim, which the tensors' dims are checked against",
        ),
    ),
    ("time_step_floor", Ignore(Type::Number, INIT)),
    ("time_step_min", Ignore(Type::Number, INIT)),
    ("time_step_max", Ignore(Type::Number, INIT)),
    (
        "moe_shared_expert_overlap",
        Ignore(
            Type::Bool,
            "overlaps the shared expert with the routed ones in time; the result is the same",
        ),
    ),
];

/// What a layer is, as the config names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Mamba,
    Attention,
    /// The dense feed-forward block.
    Mlp,
    /// The routed feed-forward block.
    Moe,
}

/// A layer of a `nemotron_h_moe` model: every feed-forward layer routes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layer {
    Mamba,
    Attention,
    Moe,
}

impl Layer {
    /// The layer as a refusal names it.
    fn what(self) -> &'static str {
        match self {
            Layer::Mamba => "a Mamba-2 layer",
            Layer::Attention => "an attention layer",
            Layer::Moe => "a MoE layer",
        }
    }
}

/// The program a model's layers make.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arch {
    /// No routed layer: `nemotron_h`.
    Dense,
    /// A routed layer: `nemotron_h_moe`.
    Moe,
}

impl Arch {
    /// The architecture's name in the header.
    fn name(self) -> &'static str {
        match self {
            Arch::Dense => "nemotron_h",
            Arch::Moe => "nemotron_h_moe",
        }
    }
}

/// The kinds of a config's layers and the key that spelled them.
#[derive(Debug)]
struct Layers {
    key: &'static str,
    kinds: Vec<Kind>,
}

impl Layers {
    /// The layer kinds of `root`: `layers_block_type`'s names (`mamba`, `attention`, `mlp`,
    /// `moe`) or `hybrid_override_pattern`'s characters (`M`, `*`, `-`, `E`), as many as
    /// `num_hidden_layers`. Either spelling alone; a name that is neither, a count that differs,
    /// both spellings and none are refused by key.
    fn read(root: &Section<'_, Error>) -> Result<Layers, Error> {
        let layers = root.count("num_hidden_layers")? as usize;
        let named = "layers_block_type";
        let pattern = "hybrid_override_pattern";
        let (key, kinds) = match (root.value.get(named), root.value.get(pattern)) {
            (Some(_), None) => (named, Self::named(root, named)?),
            (None, Some(_)) => (pattern, Self::pattern(root, pattern)?),
            (Some(_), Some(_)) => {
                return Err(root.bad(
                    pattern,
                    "is given beside layers_block_type: this map reads one spelling of the kinds",
                ));
            }
            (None, None) => {
                return Err(root.bad(
                    named,
                    "is missing, and so is hybrid_override_pattern: the config names no layer kinds",
                ));
            }
        };
        if kinds.len() != layers {
            return Err(root.bad(
                key,
                &format!(
                    "names {} layers, num_hidden_layers is {layers}",
                    kinds.len()
                ),
            ));
        }
        Ok(Layers { key, kinds })
    }

    fn named(root: &Section<'_, Error>, key: &str) -> Result<Vec<Kind>, Error> {
        root.array(key)?
            .iter()
            .enumerate()
            .map(|(l, v)| match v.as_str() {
                Some("mamba") => Ok(Kind::Mamba),
                Some("attention") => Ok(Kind::Attention),
                Some("mlp") => Ok(Kind::Mlp),
                Some("moe") => Ok(Kind::Moe),
                _ => Err(root.bad(
                    key,
                    &format!("names layer {l} {v}, not mamba, attention, mlp or moe"),
                )),
            })
            .collect()
    }

    fn pattern(root: &Section<'_, Error>, key: &str) -> Result<Vec<Kind>, Error> {
        root.field(key)?
            .as_str()
            .ok_or_else(|| root.bad(key, "is not a string"))?
            .chars()
            .enumerate()
            .map(|(l, c)| match c {
                'M' => Ok(Kind::Mamba),
                '*' => Ok(Kind::Attention),
                '-' => Ok(Kind::Mlp),
                'E' => Ok(Kind::Moe),
                _ => Err(root.bad(key, &format!("has {c:?} at layer {l}, not M, *, - or E"))),
            })
            .collect()
    }

    /// `nemotron_h_moe` when any layer routes, else `nemotron_h`.
    fn arch(&self) -> Arch {
        if self.kinds.contains(&Kind::Moe) {
            Arch::Moe
        } else {
            Arch::Dense
        }
    }

    /// The layers of a `nemotron_h_moe` model; `None` when a dense feed-forward layer stands
    /// among them, which that program has no tensors for.
    fn routed(&self) -> Option<Vec<Layer>> {
        self.kinds
            .iter()
            .map(|kind| match kind {
                Kind::Mamba => Some(Layer::Mamba),
                Kind::Attention => Some(Layer::Attention),
                Kind::Moe => Some(Layer::Moe),
                Kind::Mlp => None,
            })
            .collect()
    }
}

/// The Mamba-2 widths of the config; the tensors' dims follow from them.
#[derive(Clone, Copy, Debug)]
struct Ssm {
    conv_kernel: u64,
    /// `mamba_num_heads` times `mamba_head_dim`.
    inner: u64,
    groups: u64,
    heads: u64,
    /// The input projection's rows: z and x (`inner` each), B and C (`groups * state` each) and
    /// the time steps (`heads`).
    in_rows: u64,
    /// The convolution's channels: x, B and C.
    conv_channels: u64,
}

impl Ssm {
    fn read(root: &Section<'_, Error>) -> Result<Ssm, Error> {
        let wide = |key: &str| root.count(key).map(u64::from);
        let (heads, head_dim) = (wide("mamba_num_heads")?, wide("mamba_head_dim")?);
        let (groups, state) = (wide("n_groups")?, wide("ssm_state_size")?);
        let overflow = |key: &str| root.bad(key, "makes a width that overflows u64");
        let inner = heads
            .checked_mul(head_dim)
            .ok_or_else(|| overflow("mamba_head_dim"))?;
        if !inner.is_multiple_of(groups) {
            return Err(root.bad(
                "n_groups",
                &format!("{groups} does not divide the inner width {inner}"),
            ));
        }
        let bc = groups
            .checked_mul(state)
            .and_then(|x| x.checked_mul(2))
            .ok_or_else(|| overflow("ssm_state_size"))?;
        let in_rows = inner
            .checked_mul(2)
            .and_then(|x| x.checked_add(bc))
            .and_then(|x| x.checked_add(heads))
            .ok_or_else(|| overflow("mamba_head_dim"))?;
        let conv_channels = inner
            .checked_add(bc)
            .ok_or_else(|| overflow("ssm_state_size"))?;
        Ok(Ssm {
            conv_kernel: wide("conv_kernel")?,
            inner,
            groups,
            heads,
            in_rows,
            conv_channels,
        })
    }
}

/// The config values the map reads, checked, and the header keys they make.
#[derive(Debug)]
pub struct Checkpoint {
    layers: Vec<Layer>,
    hidden: u64,
    /// The attention's query rows and the key and value rows, `heads * head_dim` each.
    query_rows: u64,
    key_rows: u64,
    experts: usize,
    expert_ff: u64,
    shared_ff: u64,
    ssm: Ssm,
    vocab: VocabSpec,
    /// The header's model keys by suffix, under the architecture's name.
    keys: Vec<(String, Value)>,
}

/// A projection of a routed expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Projection {
    Up,
    Down,
}

/// A tensor of a layer, by its name past `layers.N.`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Part {
    /// `norm.weight`: the gain before whatever the layer mixes.
    BlockNorm,
    InProj,
    OutProj,
    ALog,
    D,
    DtBias,
    ConvWeight,
    ConvBias,
    GroupNorm,
    Query,
    Key,
    Value,
    Output,
    KScale,
    VScale,
    Router,
    RouterBias,
    SharedUp,
    SharedDown,
    Expert {
        id: usize,
        projection: Projection,
    },
}

impl Part {
    /// The part named `rest`, or `None` for a name this map does not know.
    fn of(rest: &str) -> Option<Part> {
        Some(match rest {
            "norm.weight" => Part::BlockNorm,
            "mixer.in_proj.weight" => Part::InProj,
            "mixer.out_proj.weight" => Part::OutProj,
            "mixer.A_log" => Part::ALog,
            "mixer.D" => Part::D,
            "mixer.dt_bias" => Part::DtBias,
            "mixer.conv1d.weight" => Part::ConvWeight,
            "mixer.conv1d.bias" => Part::ConvBias,
            "mixer.norm.weight" => Part::GroupNorm,
            "mixer.q_proj.weight" => Part::Query,
            "mixer.k_proj.weight" => Part::Key,
            "mixer.v_proj.weight" => Part::Value,
            "mixer.o_proj.weight" => Part::Output,
            "mixer.k_proj.k_scale" => Part::KScale,
            "mixer.v_proj.v_scale" => Part::VScale,
            "mixer.gate.weight" => Part::Router,
            "mixer.gate.e_score_correction_bias" => Part::RouterBias,
            "mixer.shared_experts.up_proj.weight" => Part::SharedUp,
            "mixer.shared_experts.down_proj.weight" => Part::SharedDown,
            _ => {
                let (id, projection) = rest.strip_prefix("mixer.experts.")?.split_once('.')?;
                let projection = match projection {
                    "up_proj.weight" => Projection::Up,
                    "down_proj.weight" => Projection::Down,
                    _ => return None,
                };
                Part::Expert {
                    id: id.parse().ok()?,
                    projection,
                }
            }
        })
    }

    /// The kind of layer that carries the part; `None` for the one every layer does.
    fn owner(self) -> Option<Layer> {
        match self {
            Part::BlockNorm => None,
            Part::InProj
            | Part::OutProj
            | Part::ALog
            | Part::D
            | Part::DtBias
            | Part::ConvWeight
            | Part::ConvBias
            | Part::GroupNorm => Some(Layer::Mamba),
            Part::Query | Part::Key | Part::Value | Part::Output | Part::KScale | Part::VScale => {
                Some(Layer::Attention)
            }
            Part::Router
            | Part::RouterBias
            | Part::SharedUp
            | Part::SharedDown
            | Part::Expert { .. } => Some(Layer::Moe),
        }
    }
}

/// What the map makes of one folder tensor.
enum Item {
    /// Left out, with why.
    Skip(&'static str),
    /// One tensor of the file.
    One(String, Mapped<Step>),
    /// One expert of a layer's stack, gathered with its layer's others.
    Expert {
        layer: usize,
        projection: Projection,
        id: usize,
    },
}

/// A move of a normalized tensor into the program's layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// `-exp(x)`, on F32, computed in f64 and rounded once: Mamba-2's `A`, stored as its log.
    NegExp,
}

impl Step {
    fn apply(&self, t: &mut Normal) -> Result<(), String> {
        match self {
            Step::NegExp => {
                if t.type_id != GgmlType::F32.as_u32() {
                    return Err(format!("-exp needs F32, not type {}", t.type_id));
                }
                for bytes in t.bytes.as_chunks_mut::<4>().0 {
                    *bytes = super::neg_exp(f32::from_le_bytes(*bytes))?.to_le_bytes();
                }
                Ok(())
            }
        }
    }
}

/// The config key `key` as a bool.
fn flag(root: &Section<'_, Error>, key: &str) -> Result<bool, Error> {
    root.field(key)?
        .as_bool()
        .ok_or_else(|| root.bad(key, "is not a bool"))
}

/// The object that holds the text model's config: the config itself for a `nemotron_h` model. A
/// vision-language wrapper, a model of another trunk and any other type are refused by name.
fn text_model(config: &Json) -> Result<Section<'_, Error>, Error> {
    let top = Section::<Error>::new(config, "config.json")?;
    let model_type = top
        .field("model_type")?
        .as_str()
        .ok_or_else(|| top.bad("model_type", "is not a string"))?;
    if model_type == "nemotron_h" {
        return Ok(top);
    }
    if !WRAPPERS.contains(&model_type) {
        return Err(top.bad(
            "model_type",
            &format!("is {model_type:?}, not a Nemotron-H type this map reads"),
        ));
    }
    let llm = Section::<Error>::new(top.field("llm_config")?, "config.json.llm_config")?;
    let inner = llm
        .field("model_type")?
        .as_str()
        .ok_or_else(|| llm.bad("model_type", "is not a string"))?;
    if inner != "nemotron_h" {
        return Err(llm.bad(
            "model_type",
            &format!(
                "is {inner:?}, not \"nemotron_h\": the map reads the one trunk every layer \
                 shares, not blocks that differ layer from layer"
            ),
        ));
    }
    Err(top.bad(
        "model_type",
        &format!(
            "{model_type:?} wraps the text model beside a vision tower, its tensors under a \
             `language_model.` prefix; the Super and Nano tables are not mapped yet"
        ),
    ))
}

impl Checkpoint {
    /// The model `root` describes, whose layers are `layers`.
    fn read(root: &Section<'_, Error>, layers: Vec<Layer>) -> Result<Self, Error> {
        if root
            .value
            .get("moe_latent_size")
            .is_some_and(|v| !v.is_null())
        {
            return Err(root.bad(
                "moe_latent_size",
                "is set: the latent pair around the experts (`fc1_latent_proj`, \
                 `fc2_latent_proj`) is not mapped yet",
            ));
        }
        let n = |key: &str| root.count(key);
        let wide = |key: &str| root.count(key).map(u64::from);
        let (heads, kv_heads, head_dim) = (
            n("num_attention_heads")?,
            n("num_key_value_heads")?,
            n("head_dim")?,
        );
        if !heads.is_multiple_of(kv_heads) {
            return Err(root.bad("num_attention_heads", "must group by num_key_value_heads"));
        }
        let (experts, used) = (n("n_routed_experts")?, n("num_experts_per_tok")?);
        if used > experts {
            return Err(root.bad("num_experts_per_tok", "exceeds n_routed_experts"));
        }
        let eps = root.number("norm_eps")?;
        if root.number("layer_norm_epsilon")? != eps {
            return Err(root.bad(
                "layer_norm_epsilon",
                "differs from norm_eps: this map reads one epsilon",
            ));
        }
        // Attention takes no position, so no reader rotates by the key; the key is the one a
        // rotary of the config's fraction of the head would take.
        let rotary = f64::from(head_dim) * f64::from(root.number("partial_rotary_factor")?);
        let rope_dims = u32::try_from(rotary as u64)
            .ok()
            .filter(|&d| {
                (2..=head_dim).contains(&d) && d.is_multiple_of(2) && f64::from(d) == rotary
            })
            .ok_or_else(|| {
                root.bad(
                    "partial_rotary_factor",
                    "does not make an even rotary width of the head",
                )
            })?;
        let ssm = Ssm::read(root)?;
        let inner = u32::try_from(ssm.inner)
            .map_err(|_| root.bad("mamba_head_dim", "makes an inner width past u32"))?;
        let per_layer = |key: &str, at: &dyn Fn(Layer) -> u32| -> Result<Value, Error> {
            layers
                .iter()
                .map(|&layer| {
                    i32::try_from(at(layer))
                        .map(Value::I32)
                        .map_err(|_| root.bad(key, "does not fit an i32"))
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Value::Array)
        };
        let dense_ff = n("intermediate_size")?;
        let count = |suffix: &str, v: u32| (suffix.to_string(), Value::U32(v));
        let float = |suffix: &str, v: f32| (suffix.to_string(), Value::F32(v));
        let vocab = VocabSpec::parse(root.value)?;
        let keys = vec![
            count("block_count", n("num_hidden_layers")?),
            count("context_length", n("max_position_embeddings")?),
            count("embedding_length", n("hidden_size")?),
            (
                "feed_forward_length".to_string(),
                per_layer("intermediate_size", &|l| {
                    if l == Layer::Moe { dense_ff } else { 0 }
                })?,
            ),
            count("attention.head_count", heads),
            (
                "attention.head_count_kv".to_string(),
                per_layer("num_key_value_heads", &|l| {
                    if l == Layer::Attention { kv_heads } else { 0 }
                })?,
            ),
            float("rope.freq_base", root.number("rope_theta")?),
            float("attention.layer_norm_rms_epsilon", eps),
            float("attention.layer_norm_epsilon", eps),
            count("expert_count", experts),
            count("expert_used_count", used),
            count("expert_group_count", n("n_group")?),
            count("expert_group_used_count", n("topk_group")?),
            count("vocab_size", vocab.size()),
            count("rope.dimension_count", rope_dims),
            count("ssm.conv_kernel", n("conv_kernel")?),
            count("ssm.state_size", n("ssm_state_size")?),
            count("ssm.group_count", n("n_groups")?),
            count("ssm.inner_size", inner),
            count("ssm.time_step_rank", n("mamba_num_heads")?),
            count("attention.key_length", head_dim),
            count("attention.value_length", head_dim),
            count("expert_feed_forward_length", n("moe_intermediate_size")?),
            count(
                "expert_shared_feed_forward_length",
                n("moe_shared_expert_intermediate_size")?,
            ),
            count("expert_shared_count", n("n_shared_experts")?),
            (
                "expert_weights_norm".to_string(),
                Value::Bool(flag(root, "norm_topk_prob")?),
            ),
            float(
                "expert_weights_scale",
                root.number("routed_scaling_factor")?,
            ),
            // The next-token block is skipped, so the file carries none.
            count("nextn_predict_layers", 0),
        ];
        Ok(Self {
            layers,
            hidden: wide("hidden_size")?,
            query_rows: u64::from(heads) * u64::from(head_dim),
            key_rows: u64::from(kv_heads) * u64::from(head_dim),
            experts: experts as usize,
            expert_ff: wide("moe_intermediate_size")?,
            shared_ff: wide("moe_shared_expert_intermediate_size")?,
            ssm,
            vocab,
            keys,
        })
    }

    /// The header's keys: the architecture and the model keys under its name.
    fn keys(&self) -> Vec<(String, Value)> {
        let arch = Arch::Moe.name();
        let mut keys = vec![(
            gguf::GENERAL_ARCHITECTURE.to_string(),
            Value::String(arch.into()),
        )];
        keys.extend(
            self.keys
                .iter()
                .map(|(suffix, v)| (format!("{arch}.{suffix}"), v.clone())),
        );
        keys
    }

    /// What the map makes of the folder tensor `name` of logical `shape`.
    fn item(&self, name: &str, shape: &[usize]) -> Result<Item, Error> {
        let ggml: Vec<u64> = shape.iter().rev().map(|&d| d as u64).collect();
        let one = |to: String, plain: Plain, steps: Vec<Step>, dims: Vec<u64>, origin: Origin| {
            Item::One(
                to,
                Mapped {
                    sources: vec![name.to_string()],
                    role: Use::Matrix(plain),
                    steps,
                    dims,
                    origin,
                },
            )
        };
        let made = |to: String, plain: Plain| one(to, plain, vec![], ggml.clone(), Origin::Made);
        match name {
            "backbone.embeddings.weight" => return Ok(made(token_embd(), Plain::Q8_0)),
            "backbone.norm_f.weight" => return Ok(made(output_norm(), Plain::F32)),
            "lm_head.weight" => return Ok(made(output(), Plain::Q8_0)),
            _ => {}
        }
        let (l, rest) = name
            .strip_prefix("backbone.layers.")
            .and_then(|r| r.split_once('.'))
            .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
            .ok_or_else(|| refuse(name, "not a Nemotron-H tensor"))?;
        let layer = self
            .layers
            .get(l)
            .copied()
            .ok_or_else(|| refuse(name, format!("layer {l} of {}", self.layers.len())))?;
        let Some(part) = Part::of(rest) else {
            let why = if LATER.contains(&rest) {
                "belongs to a dense or latent-MoE layer, whose table is not mapped yet"
            } else {
                "not a Nemotron-H tensor"
            };
            return Err(refuse(name, why));
        };
        if let Some(owner) = part.owner().filter(|&owner| owner != layer) {
            return Err(refuse(
                name,
                format!(
                    "belongs to {}, and layer {l} is {}",
                    owner.what(),
                    layer.what()
                ),
            ));
        }
        // The scan's per-head constants are stored `[heads]` and the program's are `[1, heads]`.
        let unit_first = || {
            std::iter::once(1)
                .chain(ggml.iter().copied())
                .collect::<Vec<u64>>()
        };
        let kept = |stem: &str| {
            one(
                blk(l, stem),
                Plain::F32,
                vec![],
                ggml.clone(),
                Origin::File(GgmlType::F32),
            )
        };
        Ok(match part {
            Part::BlockNorm => made(blk(l, ATTN_NORM), Plain::F32),
            Part::InProj => made(blk(l, SSM_IN), Plain::Q8_0),
            Part::OutProj => made(blk(l, SSM_OUT), Plain::Q8_0),
            Part::ALog => one(
                blk(l, SSM_A),
                Plain::F32,
                vec![Step::NegExp],
                unit_first(),
                Origin::Made,
            ),
            Part::D => one(
                blk(l, SSM_D),
                Plain::F32,
                vec![],
                unit_first(),
                Origin::Made,
            ),
            Part::DtBias => made(blk(l, SSM_DT_BIAS), Plain::F32),
            Part::ConvWeight => {
                // `[channels, 1, taps]` row-major is the program's `[taps, channels]` in ggml's
                // order, byte for byte.
                let &[channels, 1, taps] = shape else {
                    return Err(refuse(
                        name,
                        format!("shape {shape:?} is not [channels, 1, taps]"),
                    ));
                };
                one(
                    blk(l, SSM_CONV),
                    Plain::F32,
                    vec![],
                    vec![taps as u64, channels as u64],
                    Origin::Made,
                )
            }
            Part::ConvBias => made(blk(l, SSM_CONV_BIAS), Plain::F32),
            Part::GroupNorm => {
                // One gain a channel of the inner width, which the program holds a row a group.
                let &[width] = shape else {
                    return Err(refuse(
                        name,
                        format!("shape {shape:?} is not one gain a channel"),
                    ));
                };
                let groups = self.ssm.groups;
                if !(width as u64).is_multiple_of(groups) {
                    return Err(refuse(
                        name,
                        format!("{width} gains do not divide into {groups} groups"),
                    ));
                }
                one(
                    blk(l, SSM_NORM),
                    Plain::F32,
                    vec![],
                    vec![width as u64 / groups, groups],
                    Origin::Made,
                )
            }
            Part::Query => made(blk(l, ATTN_Q), Plain::Q8_0),
            Part::Key => made(blk(l, ATTN_K), Plain::Q8_0),
            Part::Value => made(blk(l, ATTN_V), Plain::Q8_0),
            Part::Output => made(blk(l, ATTN_OUTPUT), Plain::Q8_0),
            Part::KScale | Part::VScale => Item::Skip(KV_SCALE),
            Part::Router => kept(FFN_GATE_INP),
            Part::RouterBias => kept(EXP_PROBS_B),
            Part::SharedUp => made(blk(l, FFN_UP_SHEXP), Plain::Q8_0),
            Part::SharedDown => made(blk(l, FFN_DOWN_SHEXP), Plain::Q8_0),
            Part::Expert { id, projection } => Item::Expert {
                layer: l,
                projection,
                id,
            },
        })
    }

    /// The names and ggml dims of every tensor the config's geometry gives the file.
    fn expected(&self) -> Vec<(String, Vec<u64>)> {
        let (e, vocab) = (self.hidden, u64::from(self.vocab.size()));
        let (s, q, kv) = (&self.ssm, self.query_rows, self.key_rows);
        let (x, ff, shared) = (self.experts as u64, self.expert_ff, self.shared_ff);
        let mut want = vec![
            (token_embd(), vec![e, vocab]),
            (output_norm(), vec![e]),
            (output(), vec![e, vocab]),
        ];
        for (l, layer) in self.layers.iter().enumerate() {
            let at = |stem: &str, dims: Vec<u64>| (blk(l, stem), dims);
            want.push(at(ATTN_NORM, vec![e]));
            want.extend(match layer {
                Layer::Mamba => vec![
                    at(SSM_IN, vec![e, s.in_rows]),
                    at(SSM_OUT, vec![s.inner, e]),
                    at(SSM_A, vec![1, s.heads]),
                    at(SSM_D, vec![1, s.heads]),
                    at(SSM_DT_BIAS, vec![s.heads]),
                    at(SSM_CONV, vec![s.conv_kernel, s.conv_channels]),
                    at(SSM_CONV_BIAS, vec![s.conv_channels]),
                    at(SSM_NORM, vec![s.inner / s.groups, s.groups]),
                ],
                Layer::Attention => vec![
                    at(ATTN_Q, vec![e, q]),
                    at(ATTN_K, vec![e, kv]),
                    at(ATTN_V, vec![e, kv]),
                    at(ATTN_OUTPUT, vec![q, e]),
                ],
                Layer::Moe => vec![
                    at(FFN_GATE_INP, vec![e, x]),
                    at(EXP_PROBS_B, vec![x]),
                    at(FFN_UP_EXPS, vec![e, ff, x]),
                    at(FFN_DOWN_EXPS, vec![ff, e, x]),
                    at(FFN_UP_SHEXP, vec![e, shared]),
                    at(FFN_DOWN_SHEXP, vec![shared, e]),
                ],
            });
        }
        want
    }

    /// Every tensor the config's geometry gives the file is mapped, at those dims.
    fn complete(&self, mapped: &BTreeMap<String, Mapped<Step>>) -> Result<(), Error> {
        for (name, dims) in self.expected() {
            let m = mapped
                .get(&name)
                .ok_or_else(|| refuse(&name, "is missing from the folder"))?;
            if m.dims != dims {
                return Err(refuse(
                    &m.sources[0],
                    format!("holds dims {:?} where the config gives {dims:?}", m.dims),
                ));
            }
        }
        Ok(())
    }
}

impl FolderMap for Checkpoint {
    type Step = Step;

    const MODEL_TYPES: &'static [&'static str] =
        &["nemotron_h", "nemotron_h_omni", "NemotronH_Nano_VL_V2"];

    fn parse(config: &Json) -> Result<Self, Error> {
        let root = text_model(config)?;
        let layers = Layers::read(&root)?;
        if layers.arch() == Arch::Dense {
            return Err(root.bad(
                layers.key,
                &format!(
                    "names no routed layer, so the model is {}: its dense layers' tensors \
                     (`mixer.up_proj`, `mixer.down_proj`) are not mapped yet",
                    Arch::Dense.name()
                ),
            ));
        }
        let routed = layers.routed().ok_or_else(|| {
            root.bad(
                layers.key,
                &format!(
                    "names dense feed-forward layers beside routed ones; {} routes every \
                     feed-forward layer",
                    Arch::Moe.name()
                ),
            )
        })?;
        root.check(&[ROOT])?;
        Checkpoint::read(&root, routed)
    }

    fn module_class(&self, module: &str) -> Option<ModuleClass> {
        super::module_class(module)
    }

    fn map<'a>(&self, tensors: &[(&'a str, &'a [usize])]) -> Result<Mapping<Step>, Error> {
        let mut out: BTreeMap<String, Mapped<Step>> = BTreeMap::new();
        let mut skipped = BTreeMap::new();
        // A layer's experts: (layer, projection) to each expert's name and shape.
        let mut stacks: BTreeMap<(usize, Projection), BTreeMap<usize, (&str, &[usize])>> =
            BTreeMap::new();
        let mut put = |to: String, m: Mapped<Step>| match out.insert(to, m) {
            Some(prev) => Err(refuse(
                &prev.sources[0],
                "maps to a nemotron_h_moe name another tensor has",
            )),
            None => Ok(()),
        };
        for &(name, shape) in tensors {
            if name.starts_with("mtp.") {
                skipped.insert(name.to_string(), NEXT_TOKEN.to_string());
                continue;
            }
            if !trunk::sized(shape) {
                return Err(refuse(
                    name,
                    format!("shape {shape:?} is empty, zero, or overflows usize"),
                ));
            }
            match self.item(name, shape)? {
                Item::Skip(why) => {
                    skipped.insert(name.to_string(), why.to_string());
                }
                Item::One(to, m) => put(to, m)?,
                Item::Expert {
                    layer,
                    projection,
                    id,
                } => {
                    if stacks
                        .entry((layer, projection))
                        .or_default()
                        .insert(id, (name, shape))
                        .is_some()
                    {
                        return Err(refuse(name, format!("duplicate expert id {id}")));
                    }
                }
            }
        }
        for ((layer, projection), by_id) in stacks {
            let (sources, dims) = safetensors::expert_stack(&by_id, self.experts)
                .map_err(|e| refuse(&format!("layer {layer} {projection:?}"), e.to_string()))?;
            let stem = match projection {
                Projection::Up => FFN_UP_EXPS,
                Projection::Down => FFN_DOWN_EXPS,
            };
            put(
                blk(layer, stem),
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
        self.complete(&out)?;
        Ok(Mapping {
            tensors: out,
            skipped,
        })
    }

    fn mover(&self, name: &str, steps: &[Step]) -> Option<part::Step> {
        (!steps.is_empty()).then(|| {
            let (name, steps) = (name.to_string(), steps.to_vec());
            Box::new(move |n: &mut Normal| {
                steps
                    .iter()
                    .try_for_each(|s| s.apply(n).map_err(|why| format!("{name}: {why}")))
            }) as part::Step
        })
    }

    fn header(
        &self,
        _planned: &[(&str, &[u64])],
        _folder: &Folder,
    ) -> Result<Vec<(String, Value)>, Error> {
        Ok(self.keys())
    }

    fn vocab(&self) -> &VocabSpec {
        &self.vocab
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use hf::quant_config::Config;
    use safetensors::Dtype;
    use serde_json::json;

    use super::*;
    use crate::folder::{self, Make};
    use crate::test_support::{Entries, entries, gguf_tensors, json};

    const CONFIG: &str =
        "quant-configs/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4__config.json.gz";
    const HF_QUANT: &str = "quant-configs/nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4__hf_quant_config.json.gz";
    const TENSORS: &str =
        "checkpoint-headers/nemotron-3.5-lightning-nvfp4-layers-0-1-5.tensors.json.gz";
    const TWIN: &str =
        "checkpoint-headers/unsloth-nemotron-3.5-lightning-ud-q4_k_xl-gguf.header.json.gz";
    const BOLT: &str =
        "quant-configs/nvidia__NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB__config.json.gz";
    const NANO_4B: &str = "quant-configs/nvidia__NVIDIA-Nemotron-3-Nano-4B-FP8__config.json.gz";

    /// The tensors of the real folder, as its `model.safetensors.index.json` counts them.
    const REAL_TENSORS: usize = 18_487;

    /// The real folder's tensors: the committed layers 0 (Mamba-2), 1 (MoE) and 5 (attention)
    /// stand for every layer of their kind, laid out as `layers_block_type` orders the kinds.
    fn real_entries(config: &Json) -> Entries {
        let committed = entries(&json(TENSORS));
        let mut out: Entries = committed
            .iter()
            .filter(|(name, ..)| !name.starts_with("backbone.layers."))
            .cloned()
            .collect();
        for (l, kind) in config["layers_block_type"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let template = match kind.as_str().unwrap() {
                "mamba" => 0,
                "moe" => 1,
                "attention" => 5,
                other => panic!("{other}"),
            };
            let prefix = format!("backbone.layers.{template}.");
            out.extend(committed.iter().filter_map(|(name, dtype, shape)| {
                let rest = name.strip_prefix(&prefix)?;
                Some((format!("backbone.layers.{l}.{rest}"), *dtype, shape.clone()))
            }));
        }
        out
    }

    fn plan_of(
        config: &Json,
        entries: &Entries,
        q8: bool,
    ) -> Result<folder::Planning<Step>, folder::Error> {
        let checkpoint = Checkpoint::parse(config)?;
        let quant = Config::parse(config, Some(&json(HF_QUANT))).unwrap();
        folder::plan_tensors(&checkpoint, &quant, entries, q8)
    }

    /// The refusal of `entries` read under `config`.
    fn refusal(config: &Json, entries: &Entries) -> String {
        match plan_of(config, entries, true) {
            Err(why) => why.to_string(),
            Ok(_) => panic!("the folder planned"),
        }
    }

    /// The layer (none for a model-level tensor) and the stem of a GGUF tensor name.
    fn split(name: &str) -> (Option<usize>, &str) {
        name.strip_prefix("blk.")
            .and_then(|r| r.split_once('.'))
            .and_then(|(l, stem)| Some((Some(l.parse().ok()?), stem)))
            .unwrap_or((None, name))
    }

    /// The twin's types of the tensor `stem` and the type the folder holds it in, a BF16 matrix
    /// written as Q8_0: the Mamba-2 projections FP8 with one tensor scale, the shared experts and
    /// the head NVFP4, the routed stacks NVFP4 planes, and the F32 tensors F32.
    fn family(stem: &str) -> (&'static [GgmlType], GgmlType) {
        const Q8: &[GgmlType] = &[GgmlType::Q8_0];
        const ROUTED: &[GgmlType] = &[GgmlType::Q5_0, GgmlType::Q5_1, GgmlType::Q8_0];
        const F32: &[GgmlType] = &[GgmlType::F32];
        match stem {
            "ssm_in.weight" | "ssm_out.weight" => (Q8, GgmlType::FP8_B128),
            "token_embd.weight" | "attn_q.weight" | "attn_k.weight" | "attn_v.weight"
            | "attn_output.weight" => (Q8, GgmlType::Q8_0),
            "output.weight" | "ffn_up_shexp.weight" | "ffn_down_shexp.weight" => {
                (Q8, GgmlType::NVFP4)
            }
            "ffn_up_exps.weight" | "ffn_down_exps.weight" => (ROUTED, GgmlType::NVFP4_ROWS),
            _ => (F32, GgmlType::F32),
        }
    }

    /// The folder tensors the GGUF tensor `stem` of layer `layer` is made of, by llama.cpp's
    /// converter's names.
    fn sources_of(layer: Option<usize>, stem: &str) -> Vec<String> {
        let one = |name: String| vec![name];
        let Some(l) = layer else {
            return one(match stem {
                "token_embd.weight" => "backbone.embeddings.weight",
                "output_norm.weight" => "backbone.norm_f.weight",
                "output.weight" => "lm_head.weight",
                other => panic!("{other}"),
            }
            .to_string());
        };
        let mixer = |name: &str| one(format!("backbone.layers.{l}.mixer.{name}"));
        let experts = |projection: &str| -> Vec<String> {
            (0..128)
                .map(|e| format!("backbone.layers.{l}.mixer.experts.{e}.{projection}.weight"))
                .collect()
        };
        match stem {
            "attn_norm.weight" => one(format!("backbone.layers.{l}.norm.weight")),
            "ssm_in.weight" => mixer("in_proj.weight"),
            "ssm_out.weight" => mixer("out_proj.weight"),
            "ssm_a" => mixer("A_log"),
            "ssm_d" => mixer("D"),
            "ssm_dt.bias" => mixer("dt_bias"),
            "ssm_conv1d.weight" => mixer("conv1d.weight"),
            "ssm_conv1d.bias" => mixer("conv1d.bias"),
            "ssm_norm.weight" => mixer("norm.weight"),
            "attn_q.weight" => mixer("q_proj.weight"),
            "attn_k.weight" => mixer("k_proj.weight"),
            "attn_v.weight" => mixer("v_proj.weight"),
            "attn_output.weight" => mixer("o_proj.weight"),
            "ffn_gate_inp.weight" => mixer("gate.weight"),
            "exp_probs_b.bias" => mixer("gate.e_score_correction_bias"),
            "ffn_up_shexp.weight" => mixer("shared_experts.up_proj.weight"),
            "ffn_down_shexp.weight" => mixer("shared_experts.down_proj.weight"),
            "ffn_up_exps.weight" => experts("up_proj"),
            "ffn_down_exps.weight" => experts("down_proj"),
            other => panic!("{other}"),
        }
    }

    /// A header key as the committed key files write it.
    fn key_json(v: &Value) -> Json {
        match v {
            Value::U32(x) => json!(["U32", x]),
            Value::I32(x) => json!(["I32", x]),
            Value::F32(x) => json!(["F32", f64::from(*x)]),
            Value::Bool(x) => json!(["BOOL", x]),
            Value::String(s) => json!(["STR", s]),
            Value::Array(items) => {
                let items: Vec<Json> = items
                    .iter()
                    .map(|item| match item {
                        Value::I32(i) => json!(i),
                        other => panic!("{other:?}"),
                    })
                    .collect();
                json!(["ARR", ["I32", items]])
            }
            other => panic!("{other:?}"),
        }
    }

    /// The twin's key `key`, pinned at its value, replaced by the folder's.
    fn diverge(
        want: &mut serde_json::Map<String, Json>,
        key: &str,
        twin: Json,
        ours: Option<Json>,
    ) {
        let key = format!("nemotron_h_moe.{key}");
        assert_eq!(want.remove(&key), Some(twin), "{key}: the twin's value");
        if let Some(value) = ours {
            want.insert(key, value);
        }
    }

    /// The real folder's headers, read through the map, are the tensors and keys of the GGUF
    /// file of the same model: the same names at the same ggml dims, each in its type family, every
    /// tensor made of the folder tensors llama.cpp's converter makes it of, and the model keys
    /// but those that follow from the next-token block the folder leaves out.
    #[test]
    fn lightning_map_matches_the_twin_header() {
        let config = json(CONFIG);
        let entries = real_entries(&config);
        assert_eq!(
            entries.len(),
            REAL_TENSORS,
            "the committed layers repeat to the index"
        );
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let planning = plan_of(&config, &entries, true).unwrap();
        assert!(
            planning.plan.refused.is_empty(),
            "{:?}",
            planning.plan.refused
        );
        let twin = json(TWIN);
        // `blk.52` is the twin's next-token block, which the folder holds as `mtp.*`.
        let want: BTreeMap<String, (Vec<u64>, GgmlType)> = gguf_tensors(&twin)
            .into_iter()
            .filter(|(name, _)| !name.starts_with("blk.52."))
            .collect();
        let mut tensors = BTreeMap::new();
        let mut scales = Vec::new();
        for p in &planning.planned {
            match &p.make {
                Make::Tensor(m) => {
                    tensors.insert(p.decl.name.as_str(), (&p.decl, m));
                }
                Make::Scale { of } => scales.push((&p.decl, of.as_str())),
            }
        }
        assert_eq!(
            tensors.keys().copied().collect::<Vec<_>>(),
            want.keys().map(String::as_str).collect::<Vec<_>>(),
            "the folder's tensors are the twin's trunk"
        );
        assert_eq!(
            tensors.len(),
            401,
            "23 Mamba-2 x 9, 23 MoE x 7, 6 attention x 5 and 3"
        );
        for (name, (twin_dims, twin_ty)) in &want {
            let (decl, m) = tensors[name.as_str()];
            let (layer, stem) = split(name);
            let (twin_types, ty) = family(stem);
            assert_eq!(decl.dims, *twin_dims, "{name}: ggml dims");
            assert!(
                twin_types.contains(twin_ty),
                "{name}: the twin holds {twin_ty}"
            );
            assert_eq!(
                GgmlType::from_u32(decl.type_id),
                ty,
                "{name}: the folder's type"
            );
            assert_eq!(m.sources, sources_of(layer, stem), "{name}: its sources");
            let origin = match stem {
                "ffn_gate_inp.weight" | "exp_probs_b.bias" => Origin::File(GgmlType::F32),
                "ffn_up_exps.weight" | "ffn_down_exps.weight" => Origin::Stack {
                    ty: GgmlType::NVFP4_ROWS,
                    layout: StackLayout::Planes,
                },
                _ => Origin::Made,
            };
            assert_eq!(m.origin, origin, "{name}: where its bytes are read");
            let steps = if stem == "ssm_a" {
                vec![Step::NegExp]
            } else {
                vec![]
            };
            assert_eq!(m.steps, steps, "{name}: its moves");
        }
        assert_eq!(
            scales.len(),
            93,
            "46 stacks, 46 shared experts and the head read a scale"
        );
        for (decl, of) in &scales {
            let (owner, _) = tensors[*of];
            let n = if GgmlType::from_u32(owner.type_id) == GgmlType::NVFP4_ROWS {
                128
            } else {
                1
            };
            assert_eq!(decl.dims, vec![n], "{of}: its tensor scales");
            assert_eq!(
                decl.name,
                format!("{}.scale", of.strip_suffix(".weight").unwrap())
            );
        }
        // Opened as stored, a BF16 matrix stays BF16 where `prepare` writes Q8_0.
        let open = plan_of(&config, &entries, false).unwrap();
        for name in ["token_embd.weight", "blk.5.attn_q.weight"] {
            let planned = open.planned.iter().find(|p| p.decl.name == name).unwrap();
            assert_eq!(
                GgmlType::from_u32(planned.decl.type_id),
                GgmlType::BF16,
                "{name}"
            );
        }
        let mut model_keys: serde_json::Map<String, Json> = twin["keys"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(k, _)| k.starts_with("nemotron_h_moe.") || *k == gguf::GENERAL_ARCHITECTURE)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        diverge(
            &mut model_keys,
            "block_count",
            json!(["U32", 53]),
            Some(json!(["U32", 52])),
        );
        diverge(
            &mut model_keys,
            "nextn_predict_layers",
            json!(["U32", 1]),
            Some(json!(["U32", 0])),
        );
        // The twin's value is hidden_size / heads; the folder's, the rotary of the config's
        // fraction of the head, which no reader of an attention without positions uses.
        diverge(
            &mut model_keys,
            "rope.dimension_count",
            json!(["U32", 84]),
            Some(json!(["U32", 128])),
        );
        diverge(
            &mut model_keys,
            "rope.scaling.finetuned",
            json!(["BOOL", false]),
            None,
        );
        // The twin's per-layer arrays carry the next-token block's entry past the trunk.
        for key in [
            "nemotron_h_moe.feed_forward_length",
            "nemotron_h_moe.attention.head_count_kv",
        ] {
            let items = model_keys[key][1][1].as_array_mut().unwrap();
            assert_eq!(items.len(), 53, "{key}");
            items.truncate(52);
        }
        let got: serde_json::Map<String, Json> = checkpoint
            .keys()
            .iter()
            .map(|(k, v)| (k.clone(), key_json(v)))
            .collect();
        assert_eq!(Json::Object(got), Json::Object(model_keys));
    }

    /// The next-token block's tensors and the FP8 KV-cache scales are left out by name with their
    /// reason; no other tensor is, and the input scales the plan reads are no tensor of the file.
    #[test]
    fn lightning_skips_mtp_and_kv_scales_by_name() {
        let config = json(CONFIG);
        let entries = real_entries(&config);
        let planning = plan_of(&config, &entries, true).unwrap();
        let by_reason = |why: &str| {
            planning
                .skipped
                .iter()
                .filter(|(_, reason)| reason.as_str() == why)
                .map(|(name, _)| name.as_str())
                .collect::<Vec<&str>>()
        };
        let next_token = by_reason(NEXT_TOKEN);
        assert_eq!(next_token.len(), 270, "the block's 8 and 262 tensors");
        assert!(next_token.iter().all(|n| n.starts_with("mtp.")));
        let scales = by_reason(KV_SCALE);
        assert_eq!(
            scales.len(),
            12,
            "a K and a V scale on each of 6 attention layers"
        );
        assert!(
            scales
                .iter()
                .all(|n| n.ends_with(".k_proj.k_scale") || n.ends_with(".v_proj.v_scale"))
        );
        assert_eq!(planning.skipped.len(), 282);
        assert_eq!(
            planning.plan.input_scales.len(),
            46,
            "an in_proj's and an out_proj's"
        );
        let read: BTreeSet<&str> = planning
            .planned
            .iter()
            .filter_map(|p| match &p.make {
                Make::Tensor(m) => Some(m.sources.iter().map(String::as_str)),
                Make::Scale { .. } => None,
            })
            .flatten()
            .collect();
        assert!(
            planning.skipped.keys().all(|n| !read.contains(n.as_str())),
            "a tensor left out is read by none"
        );
    }

    /// A trunk of another kind, a wrapper, the dense and latent tensors, an unknown mixer, a
    /// mixer on the wrong layer, a layer the config does not have, a missing tensor, a stack of
    /// too few experts, a dim the config does not give and a setting the program does not run
    /// are each refused by name.
    #[test]
    fn refuses_puzzle_inner_and_unknown_mixer_by_name() {
        let config = json(CONFIG);
        let parse = |c: &Json| Checkpoint::parse(c).unwrap_err().to_string();
        let why = parse(&json(BOLT));
        assert!(
            why.contains("llm_config.model_type") && why.contains("nemotron_h_puzzle"),
            "{why}"
        );
        let wrapped = json!({"model_type": "nemotron_h_omni", "llm_config": config.clone()});
        let why = parse(&wrapped);
        assert!(
            why.contains("nemotron_h_omni") && why.contains("language_model."),
            "{why}"
        );
        let why = parse(&json!({"model_type": "llama"}));
        assert!(why.contains("model_type") && why.contains("llama"), "{why}");
        let why = parse(&json(NANO_4B));
        assert!(
            why.contains("hybrid_override_pattern") && why.contains("not mapped yet"),
            "{why}"
        );
        for (key, value, needle) in [
            ("mlp_hidden_act", json!("gelu"), "mlp_hidden_act"),
            (
                "time_step_limit",
                json!([0.0, 1.0e9]),
                "time_step_limit is not a key",
            ),
            ("moe_latent_size", json!(1024), "moe_latent_size"),
            ("layer_norm_epsilon", json!(1.0e-6), "differs from norm_eps"),
        ] {
            let mut with = config.clone();
            with[key] = value;
            let why = parse(&with);
            assert!(why.contains(needle), "{key}: {why}");
        }
        let entries = real_entries(&config);
        let extra = |name: &str| {
            let mut with = entries.clone();
            with.push((name.to_string(), Dtype::Bf16, vec![4, 4]));
            refusal(&config, &with)
        };
        for (name, needle) in [
            (
                "backbone.layers.1.mixer.foo_proj.weight",
                "not a Nemotron-H tensor",
            ),
            (
                "backbone.layers.1.mixer.in_proj.weight",
                "belongs to a Mamba-2 layer",
            ),
            (
                "backbone.layers.0.mixer.q_proj.weight",
                "belongs to an attention layer",
            ),
            (
                "backbone.layers.1.mixer.fc1_latent_proj.weight",
                "not mapped yet",
            ),
            ("backbone.layers.2.mixer.up_proj.weight", "not mapped yet"),
            ("backbone.layers.52.norm.weight", "layer 52 of 52"),
            (
                "language_model.backbone.embeddings.weight",
                "not a Nemotron-H tensor",
            ),
        ] {
            let why = extra(name);
            assert!(why.contains(name) && why.contains(needle), "{name}: {why}");
        }
        let mut missing = entries.clone();
        missing.retain(|(n, ..)| n != "backbone.layers.3.mixer.gate.e_score_correction_bias");
        let why = refusal(&config, &missing);
        assert!(
            why.contains("blk.3.exp_probs_b.bias") && why.contains("missing"),
            "{why}"
        );
        let mut short = entries.clone();
        short.retain(|(n, ..)| !n.starts_with("backbone.layers.1.mixer.experts.127.up_proj."));
        let why = refusal(&config, &short);
        assert!(why.contains("layer 1") && why.contains("0..128"), "{why}");
        let mut skewed = config.clone();
        skewed["mamba_head_dim"] = json!(32);
        let why = refusal(&skewed, &entries);
        assert!(
            why.contains("backbone.layers.0.mixer.in_proj.weight") && why.contains("config gives"),
            "{why}"
        );
    }

    /// Both spellings of the layer kinds are read, a model with a routed layer is
    /// `nemotron_h_moe` and one with none `nemotron_h`, and a spelling that names no kind, a count
    /// that differs, two spellings, none, and a dense layer among routed ones are refused.
    #[test]
    fn layer_kinds_pick_dense_or_moe_arch() {
        let read = |c: &Json| Layers::read(&Section::<Error>::new(c, "config.json").unwrap());
        let layers = read(&json(CONFIG)).unwrap();
        assert_eq!(layers.key, "layers_block_type");
        let count = |kind: Kind| layers.kinds.iter().filter(|&&k| k == kind).count();
        assert_eq!(
            [
                count(Kind::Mamba),
                count(Kind::Moe),
                count(Kind::Attention),
                count(Kind::Mlp)
            ],
            [23, 23, 6, 0]
        );
        assert_eq!(layers.arch(), Arch::Moe);
        assert_eq!(layers.arch().name(), "nemotron_h_moe");
        assert_eq!(layers.routed().map(|routed| routed.len()), Some(52));

        let nano = json(NANO_4B);
        let layers = read(&nano).unwrap();
        assert_eq!(layers.key, "hybrid_override_pattern");
        let pattern = nano["hybrid_override_pattern"].as_str().unwrap();
        for (c, kind) in [('M', Kind::Mamba), ('*', Kind::Attention), ('-', Kind::Mlp)] {
            let n = layers.kinds.iter().filter(|&&k| k == kind).count();
            assert_eq!(n, pattern.matches(c).count(), "{c}");
        }
        assert_eq!(layers.arch(), Arch::Dense);
        assert_eq!(layers.arch().name(), "nemotron_h");

        let spelled = |n: u32, pattern: &str| json!({"num_hidden_layers": n, "hybrid_override_pattern": pattern});
        let layers = read(&spelled(4, "ME*E")).unwrap();
        assert_eq!(
            layers.kinds,
            [Kind::Mamba, Kind::Moe, Kind::Attention, Kind::Moe]
        );
        assert_eq!(layers.arch(), Arch::Moe);
        let mixed = read(&spelled(3, "M-E")).unwrap();
        assert_eq!(mixed.arch(), Arch::Moe);
        assert!(
            mixed.routed().is_none(),
            "a dense layer has no tensor in nemotron_h_moe"
        );

        let refused = |c: Json| read(&c).unwrap_err().to_string();
        let why = refused(spelled(2, "MX"));
        assert!(why.contains("'X' at layer 1"), "{why}");
        let why = refused(spelled(3, "ME"));
        assert!(
            why.contains("names 2 layers, num_hidden_layers is 3"),
            "{why}"
        );
        let why = refused(json!({"num_hidden_layers": 1, "layers_block_type": ["mamba2"]}));
        assert!(why.contains("layer 0") && why.contains("mamba2"), "{why}");
        let both = json!({"num_hidden_layers": 1, "layers_block_type": ["mamba"],
            "hybrid_override_pattern": "M"});
        assert!(refused(both).contains("beside layers_block_type"));
        let why = refused(json!({"num_hidden_layers": 1}));
        assert!(why.contains("names no layer kinds"), "{why}");
        let why = Checkpoint::parse(&nano).unwrap_err().to_string();
        assert!(why.contains("names no routed layer"), "{why}");
    }

    /// `A` is stored as its log and the program holds `-exp(A)`, in f64 rounded once; a tensor
    /// that is not F32 and a value `-exp` leaves no finite nonzero F32 of are refused.
    #[test]
    fn neg_exp_step_makes_a_from_its_log() {
        let f32_tensor = |values: &[f32]| Normal {
            type_id: GgmlType::F32.as_u32(),
            dims: vec![1, values.len() as u64],
            bytes: values.iter().flat_map(|x| x.to_le_bytes()).collect(),
            scale: vec![],
        };
        let mut a = f32_tensor(&[0.0, 1.0]);
        Step::NegExp.apply(&mut a).unwrap();
        assert_eq!(a, f32_tensor(&[-1.0, (-1.0f64.exp()) as f32]));
        let mut bf16 = Normal {
            type_id: GgmlType::BF16.as_u32(),
            ..Normal::default()
        };
        assert!(
            Step::NegExp
                .apply(&mut bf16)
                .unwrap_err()
                .contains("needs F32")
        );
        let why = Step::NegExp
            .apply(&mut f32_tensor(&[f32::NAN]))
            .unwrap_err();
        assert!(why.contains("non-finite"), "{why}");
    }
}
