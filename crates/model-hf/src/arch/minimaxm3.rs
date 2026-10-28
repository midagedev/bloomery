//! MiniMax-M3.1-Flash's official checkpoint folder in the minimax-m3 names, layouts and keys the
//! program reads, as llama.cpp's converter (`conversion/minimax.py`, `MiniMaxM3Model`) writes them
//! into a GGUF, with the formats left as the folder stores them.
//!
//! A checkpoint tensor maps to one minimax-m3 tensor ([`names`]) the way its role reads it
//! ([`Use`]). The embedding, the head, the routers and their selection biases stay in the
//! folder's bytes ([`Origin::File`]), a layer's routed experts stay there as one stack of NVFP4
//! planes a projection ([`Origin::Stack`]; each expert's tensor scale is the reciprocal of its
//! `weight_global_scale`, the `input_global_scale` is read, checked and unused), every MXFP8
//! projection is made through [`hf::modelopt::normalize`], and every norm gain is made F32 with
//! `1` added (the Gemma norm: the folder stores the offset from 1). The vision tower, its
//! projector and the draft model are left out with their reason; any other name is refused.
//!
//! The header carries the keys the minimax-m3 reader asks of a file, from the config's text model;
//! a config value the program does not run, and a key this map does not know, is refused by name.

use std::collections::BTreeMap;

use gguf::{GgmlType, StackLayout, Value};
use hf::modelopt::{Normal, Plain, Use};
use hf::quant_config::ModuleClass;
use hf::schema::{self, Fields, Rule::*, Type};
use model::arch::minimaxm3::names;
use safetensors::Folder;
use serde_json::Value as Json;

use crate::folder::{Error, FolderMap, Mapping, Origin, PlusOne, VocabSpec, refuse};
use crate::part;

const ARCHITECTURE: &str = "MiniMaxM3SparseForConditionalGeneration";
const TEXT_ARCHITECTURE: &str = "MiniMaxM3SparseForCausalLM";

/// The checkpoint's prefix of the text model's layers, and of the unlayered tensors.
const MODEL: &str = "language_model.model.";
const LAYERS: &str = "language_model.model.layers.";
const HEAD: &str = "language_model.lm_head.weight";

const VISION: &str =
    "the vision tower, its projector and the patch-merge MLP; this map reads the text model";
const DRAFT: &str = "the draft model, which the trunk never reads";
const DTYPE: &str = "the dtype HF loads in; each tensor keeps its stored dtype and format";

/// `config.json`. `hf::quant_config` reads `quantization_config`; the vocabulary's size is
/// `text_config`'s, which `hf::vocab::Config` reads.
const ROOT: &Fields = &[
    ("architectures", Read),
    ("model_type", Need(r#"["minimax_m3_vl"]"#)),
    ("text_config", Read),
    ("quantization_config", Read),
    // The map needs a separate lm_head.
    super::trunk::HEAD_OWN,
    // A reward head would be tensors the map does not read.
    ("num_reward_heads", Only("[0]")),
    (
        "auto_map",
        Ignore(Type::Object, "names HF's classes for its own loader"),
    ),
    ("torch_dtype", Ignore(Type::String, DTYPE)),
    ("dtype", Ignore(Type::String, DTYPE)),
    (
        "transformers_version",
        Ignore(Type::String, "the version that wrote the file"),
    ),
    ("vision_config", Ignore(Type::Object, VISION)),
    ("img_token_compression_config", Ignore(Type::Object, VISION)),
    ("image_grid_pinpoints", Ignore(Type::String, VISION)),
    ("image_seq_length", Ignore(Type::Number, VISION)),
    ("image_token_index", Ignore(Type::Number, VISION)),
    ("video_token_index", Ignore(Type::Number, VISION)),
    ("multimodal_projector_bias", Ignore(Type::Bool, VISION)),
    ("process_image_mode", Ignore(Type::String, VISION)),
    ("projector_hidden_act", Ignore(Type::String, VISION)),
    ("projector_hidden_size", Ignore(Type::Number, VISION)),
    ("vision_feature_layer", Ignore(Type::Number, VISION)),
    (
        "vision_feature_select_strategy",
        Ignore(Type::String, VISION),
    ),
];

/// `config.json`'s `text_config`. A setting the program computes with one value, and GGUF has no
/// key for, is `Need` that value: another would be served as this one.
const TEXT_CONFIG: &Fields = &[
    ("hidden_size", Read),
    ("intermediate_size", Read),
    ("num_hidden_layers", Read),
    ("num_attention_heads", Read),
    ("num_key_value_heads", Read),
    ("head_dim", Read),
    ("vocab_size", Read),
    ("max_position_embeddings", Read),
    ("rms_norm_eps", Read),
    ("rope_theta", Read),
    ("rotary_dim", Read),
    ("partial_rotary_factor", Read),
    ("dense_intermediate_size", Read),
    ("shared_intermediate_size", Read),
    ("num_local_experts", Read),
    ("num_experts_per_tok", Read),
    ("n_shared_experts", Read),
    ("routed_scaling_factor", Read),
    ("moe_layer_freq", Read),
    ("sparse_attention_config", Read),
    ("architectures", Read),
    ("model_type", Only(r#"["minimax_m3_vl_text"]"#)),
    // The activation of every dense, shared and routed expert, with its two constants.
    ("hidden_act", Need(r#"["swigluoai"]"#)),
    ("swiglu_alpha", Need("[1.702]")),
    ("swiglu_limit", Need("[7.0, 7]")),
    // The norms' gain is 1 + the stored weight, per head for q and k, no output gate.
    ("use_gemma_norm", Need("[true]")),
    ("use_qk_norm", Need("[true]")),
    ("qk_norm_type", Need(r#"["per_head"]"#)),
    ("attention_output_gate", Need("[false]")),
    // The router scores by sigmoid and adds the stored selection bias to choose.
    ("scoring_func", Need(r#"["sigmoid"]"#)),
    ("use_routing_bias", Need("[true]")),
    super::trunk::HEAD_OWN,
    // The folder carries no next-token layers, and the program reads none.
    ("use_mtp", Only("[false]")),
    ("num_mtp_modules", Only("[0]")),
    ("mtp_transformer_layers", Only("[0]")),
    ("num_nextn_predict_layers", Only("[0]")),
    ("mtp_num_hidden_layers", Only("[0]")),
];

/// `text_config.sparse_attention_config`: MiniMax Sparse Attention's indexer.
const SPARSE: &Fields = &[
    ("use_sparse_attention", Need("[true]")),
    ("sparse_index_dim", Read),
    ("sparse_num_index_heads", Read),
    ("sparse_topk_blocks", Read),
    ("sparse_block_size", Read),
    ("sparse_local_block", Read),
    ("sparse_attention_freq", Read),
    // The block score is the max over its keys and heads, and no initial block is always kept.
    ("sparse_score_type", Need(r#"["max"]"#)),
    ("sparse_init_block", Need("[0]")),
    // The indexer is a selection branch alone: no value path on any layer.
    ("sparse_disable_index_value", Read),
];

/// The checkpoint values the map and the header read, checked.
#[derive(Debug)]
pub struct Checkpoint {
    layers: usize,
    experts: usize,
    /// The layers before the first routed one, which run a dense feed-forward block.
    dense_lead: usize,
    /// Where attention is sparse (carries the indexer), a layer each.
    sparse: Vec<bool>,
    hidden: u64,
    vocab: VocabSpec,
    /// The header's model keys by suffix, under the architecture's name.
    keys: Vec<(&'static str, Value)>,
}

/// One object of `config.json` and the path that names it in a refusal.
struct Section<'a> {
    value: &'a Json,
    path: &'a str,
}

impl<'a> Section<'a> {
    fn bad(&self, key: &str, why: &str) -> Error {
        Error::Refused(format!(
            "minimax-m3 hf: config.json: {}.{key} {why}",
            self.path
        ))
    }

    fn field(&self, key: &str) -> Result<&'a Json, Error> {
        self.value
            .get(key)
            .ok_or_else(|| self.bad(key, "is missing"))
    }

    fn object(&self, key: &str, path: &'a str) -> Result<Section<'a>, Error> {
        let value = self.field(key)?;
        if !value.is_object() {
            return Err(self.bad(key, "is not an object"));
        }
        Ok(Section { value, path })
    }

    /// `schema::check` of this object against `table`.
    fn check(&self, table: &Fields) -> Result<(), Error> {
        schema::check(self.value, self.path, &[table])
            .map_err(|e| Error::Refused(format!("minimax-m3 hf: config.json: {e}")))
    }

    fn count(&self, key: &str) -> Result<u32, Error> {
        self.field(key)?
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| self.bad(key, "must be a positive u32"))
    }

    fn number(&self, key: &str) -> Result<f64, Error> {
        self.field(key)?
            .as_f64()
            .filter(|n| n.is_finite() && *n > 0.0)
            .ok_or_else(|| self.bad(key, "must be a finite positive number"))
    }

    fn single(&self, key: &str) -> Result<f32, Error> {
        let n = self.number(key)?;
        Some(n as f32)
            .filter(|f| f.is_finite())
            .ok_or_else(|| self.bad(key, "does not fit f32"))
    }

    /// A list of one 0 or 1 a layer.
    fn flags(&self, key: &str, layers: usize) -> Result<Vec<bool>, Error> {
        let list = self
            .field(key)?
            .as_array()
            .ok_or_else(|| self.bad(key, "is not an array"))?;
        if list.len() != layers {
            return Err(self.bad(
                key,
                &format!("has {} entries for {layers} layers", list.len()),
            ));
        }
        list.iter()
            .enumerate()
            .map(|(l, v)| match v.as_u64() {
                Some(0) => Ok(false),
                Some(1) => Ok(true),
                _ => Err(self.bad(key, &format!("holds {v} at layer {l}, not 0 or 1"))),
            })
            .collect()
    }

    /// That `key` is the one string array `want`, which names the model class.
    fn architectures(&self, want: &str) -> Result<(), Error> {
        let got = self
            .field("architectures")?
            .as_array()
            .ok_or_else(|| self.bad("architectures", "is not an array"))?;
        match got.as_slice() {
            [Json::String(s)] if s == want => Ok(()),
            _ => Err(self.bad(
                "architectures",
                &format!("{got:?}: only {want} is supported"),
            )),
        }
    }
}

impl Checkpoint {
    /// Read the fields the map and the header use, and refuse by name a key [`ROOT`],
    /// [`TEXT_CONFIG`] or [`SPARSE`] does not name or a value they do not allow, and the layer
    /// patterns and widths the program has no way to run.
    fn read(config: &Json) -> Result<Self, Error> {
        let root = Section {
            value: config,
            path: "config",
        };
        root.check(ROOT)?;
        root.architectures(ARCHITECTURE)?;
        let text = root.object("text_config", "text_config")?;
        text.check(TEXT_CONFIG)?;
        text.architectures(TEXT_ARCHITECTURE)?;
        let sparse = text.object(
            "sparse_attention_config",
            "text_config.sparse_attention_config",
        )?;
        sparse.check(SPARSE)?;
        let vocab = VocabSpec::parse(config)?.with_pre("minimax-m2")?;

        let n = |key: &str| text.count(key);
        let layers = n("num_hidden_layers")? as usize;
        let (hidden, heads, kv, head_dim) = (
            n("hidden_size")?,
            n("num_attention_heads")?,
            n("num_key_value_heads")?,
            n("head_dim")?,
        );
        if !heads.is_multiple_of(kv) {
            return Err(text.bad("num_attention_heads", "must group by num_key_value_heads"));
        }
        let experts = n("num_local_experts")?;
        let used = n("num_experts_per_tok")?;
        if used > experts {
            return Err(text.bad("num_experts_per_tok", "exceeds num_local_experts"));
        }
        let (ff, shared_ff, shared) = (
            n("intermediate_size")?,
            n("shared_intermediate_size")?,
            n("n_shared_experts")?,
        );
        let partial = text.number("partial_rotary_factor")?;
        let rotary = n("rotary_dim")?;
        if partial > 1.0 || f64::from(head_dim) * partial != f64::from(rotary) {
            return Err(text.bad(
                "rotary_dim",
                "must be head_dim times partial_rotary_factor, within one head",
            ));
        }
        let moe = text.flags("moe_layer_freq", layers)?;
        let dense_lead = moe.iter().take_while(|&&routed| !routed).count();
        if moe[dense_lead..].iter().any(|&routed| !routed) {
            return Err(text.bad(
                "moe_layer_freq",
                "is not dense layers and then routed ones: the program's dense blocks are the leading layers",
            ));
        }
        let sparse_flags = sparse.flags("sparse_attention_freq", layers)?;
        if !sparse
            .flags("sparse_disable_index_value", layers)?
            .iter()
            .all(|&off| off)
        {
            return Err(sparse.bad(
                "sparse_disable_index_value",
                "is not 1 on every layer: the program's indexer has no value path",
            ));
        }
        let u32_key = |suffix, v: u32| (suffix, Value::U32(v));
        let keys = vec![
            u32_key("block_count", layers as u32),
            u32_key("context_length", n("max_position_embeddings")?),
            u32_key("embedding_length", hidden),
            u32_key("feed_forward_length", n("dense_intermediate_size")?),
            u32_key("attention.head_count", heads),
            u32_key("attention.head_count_kv", kv),
            u32_key("attention.key_length", head_dim),
            u32_key("attention.value_length", head_dim),
            (
                "attention.layer_norm_rms_epsilon",
                Value::F32(text.single("rms_norm_eps")?),
            ),
            ("rope.freq_base", Value::F32(text.single("rope_theta")?)),
            u32_key("rope.dimension_count", rotary),
            u32_key("expert_count", experts),
            u32_key("expert_used_count", used),
            u32_key("expert_feed_forward_length", ff),
            u32_key("expert_shared_count", shared),
            u32_key("expert_shared_feed_forward_length", shared_ff),
            // llama.cpp's LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID.
            u32_key("expert_gating_func", 2),
            (
                "expert_weights_scale",
                Value::F32(text.single("routed_scaling_factor")?),
            ),
            // The router's kept scores are renormalized to sum to one.
            ("expert_weights_norm", Value::Bool(true)),
            u32_key("leading_dense_block_count", dense_lead as u32),
            u32_key(
                "attention.indexer.head_count",
                sparse.count("sparse_num_index_heads")?,
            ),
            u32_key(
                "attention.indexer.key_length",
                sparse.count("sparse_index_dim")?,
            ),
            u32_key(
                "attention.indexer.top_k",
                sparse.count("sparse_topk_blocks")?,
            ),
            u32_key(
                "attention.indexer.block_size",
                sparse.count("sparse_block_size")?,
            ),
            u32_key(
                "attention.indexer.local_blocks",
                sparse.count("sparse_local_block")?,
            ),
            (
                "attention.sparse_layers",
                Value::Array(sparse_flags.iter().copied().map(Value::Bool).collect()),
            ),
        ];
        Ok(Checkpoint {
            layers,
            experts: experts as usize,
            dense_lead,
            sparse: sparse_flags,
            hidden: u64::from(hidden),
            vocab,
            keys,
        })
    }
}

/// One minimax-m3 tensor.
pub type Mapped = crate::folder::Mapped<PlusOne>;

/// Why the checkpoint tensor `name` is left out of the header: the vision tower, its projector and
/// the patch-merge MLP, and a draft model's layers.
fn skip_reason(name: &str) -> Option<&'static str> {
    if name.starts_with("vision_tower.")
        || name.starts_with("multi_modal_projector.")
        || name.starts_with("patch_merge_mlp.")
    {
        return Some(VISION);
    }
    name.strip_prefix(MODEL)?
        .starts_with("dspark.")
        .then_some(DRAFT)
}

/// The class of the weight module `module`, for a quantization config that targets classes.
fn module_class_of(module: &str) -> Option<ModuleClass> {
    match module {
        "language_model.model.embed_tokens" => return Some(ModuleClass::Embedding),
        "language_model.lm_head" => return Some(ModuleClass::Linear),
        _ => {}
    }
    let (layer, rest) = module.strip_prefix(LAYERS)?.split_once('.')?;
    layer.parse::<usize>().ok()?;
    if let Some(expert) = rest.strip_prefix("block_sparse_moe.experts.") {
        let (id, projection) = expert.split_once('.')?;
        id.parse::<usize>().ok()?;
        return matches!(projection, "w1" | "w2" | "w3").then_some(ModuleClass::Linear);
    }
    matches!(
        rest,
        "self_attn.q_proj"
            | "self_attn.k_proj"
            | "self_attn.v_proj"
            | "self_attn.o_proj"
            | "self_attn.index_q_proj"
            | "self_attn.index_k_proj"
            | "mlp.gate_proj"
            | "mlp.up_proj"
            | "mlp.down_proj"
            | "block_sparse_moe.gate"
            | "block_sparse_moe.shared_experts.gate_proj"
            | "block_sparse_moe.shared_experts.up_proj"
            | "block_sparse_moe.shared_experts.down_proj"
    )
    .then_some(ModuleClass::Linear)
}

/// A layer's routed projections, by the stack they make: an expert's id to its tensor name and
/// shape.
type Experts<'a> = BTreeMap<usize, (&'a str, &'a [usize])>;

impl Checkpoint {
    /// The minimax-m3 tensors, by name, of the checkpoint tensors `tensors` (name and row-major
    /// logical shape: a packed weight's unpacked shape), and each one `skipped` with its reason.
    fn mapped<'a>(&self, tensors: &[(&'a str, &'a [usize])]) -> Result<Mapping<PlusOne>, Error> {
        let mut out = Mapping {
            tensors: BTreeMap::new(),
            skipped: BTreeMap::new(),
        };
        let mut put = |to: String, m: Mapped| match out.tensors.insert(to, m) {
            Some(prev) => Err(refuse(
                &prev.sources[0],
                "maps to a minimax-m3 name another tensor has",
            )),
            None => Ok(()),
        };
        // A layer's experts: (layer, the projection's w1, w2 or w3) to each expert's tensor.
        let mut stacks: BTreeMap<(usize, &str), Experts> = BTreeMap::new();
        for &(name, shape) in tensors {
            if let Some(why) = skip_reason(name) {
                out.skipped.insert(name.to_string(), why.to_string());
                continue;
            }
            if shape.is_empty()
                || shape.contains(&0)
                || shape
                    .iter()
                    .try_fold(1usize, |n, &d| n.checked_mul(d))
                    .is_none()
            {
                return Err(refuse(
                    name,
                    format!("shape {shape:?} is empty, zero, or overflows usize"),
                ));
            }
            let dims = || shape.iter().rev().map(|&d| d as u64).collect::<Vec<_>>();
            let one = |role: Use, steps: Vec<PlusOne>, origin: Origin| Mapped {
                sources: vec![name.to_string()],
                role,
                steps,
                dims: dims(),
                origin,
            };
            // The gain of a Gemma norm, made F32 with 1 added.
            let norm = || one(Use::Matrix(Plain::F32), vec![PlusOne], Origin::Made);
            // A projection the card reads in the folder's quantized format.
            let matrix = || one(Use::Matrix(Plain::Q8_0), vec![], Origin::Made);
            // A tensor the host reads where the folder stores it.
            let kept =
                |plain: Plain, ty: GgmlType| one(Use::Matrix(plain), vec![], Origin::File(ty));
            match name {
                HEAD => {
                    put(names::output(), kept(Plain::Bf16, GgmlType::BF16))?;
                    continue;
                }
                "language_model.model.embed_tokens.weight" => {
                    put(names::token_embd(), kept(Plain::Bf16, GgmlType::BF16))?;
                    continue;
                }
                "language_model.model.norm.weight" => {
                    put(names::output_norm(), norm())?;
                    continue;
                }
                _ => {}
            }
            let (l, rest) = name
                .strip_prefix(LAYERS)
                .and_then(|r| r.split_once('.'))
                .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
                .ok_or_else(|| refuse(name, "not a MiniMax-M3 tensor"))?;
            if l >= self.layers {
                return Err(refuse(name, format!("layer {l} of {}", self.layers)));
            }
            let (dense, sparse) = (l < self.dense_lead, self.sparse[l]);
            let needs = |ok: bool, what: &str| {
                if ok {
                    Ok(())
                } else {
                    Err(refuse(name, format!("layer {l} has no {what}")))
                }
            };
            let (to, m) = match rest {
                "input_layernorm.weight" => (names::attn_norm(l), norm()),
                "post_attention_layernorm.weight" => (names::ffn_norm(l), norm()),
                "self_attn.q_norm.weight" => (names::attn_q_norm(l), norm()),
                "self_attn.k_norm.weight" => (names::attn_k_norm(l), norm()),
                "self_attn.q_proj.weight" => (names::attn_q(l), matrix()),
                "self_attn.k_proj.weight" => (names::attn_k(l), matrix()),
                "self_attn.v_proj.weight" => (names::attn_v(l), matrix()),
                "self_attn.o_proj.weight" => (names::attn_output(l), matrix()),
                "self_attn.index_q_proj.weight" => {
                    needs(sparse, "indexer (sparse_attention_freq is 0)")?;
                    (names::indexer_q_proj(l), matrix())
                }
                "self_attn.index_k_proj.weight" => {
                    needs(sparse, "indexer (sparse_attention_freq is 0)")?;
                    (names::indexer_k_proj(l), matrix())
                }
                "self_attn.index_q_norm.weight" => {
                    needs(sparse, "indexer (sparse_attention_freq is 0)")?;
                    (names::indexer_q_norm(l), norm())
                }
                "self_attn.index_k_norm.weight" => {
                    needs(sparse, "indexer (sparse_attention_freq is 0)")?;
                    (names::indexer_k_norm(l), norm())
                }
                "mlp.gate_proj.weight" => {
                    needs(dense, "dense block (moe_layer_freq is 1)")?;
                    (names::ffn_gate(l), matrix())
                }
                "mlp.up_proj.weight" => {
                    needs(dense, "dense block (moe_layer_freq is 1)")?;
                    (names::ffn_up(l), matrix())
                }
                "mlp.down_proj.weight" => {
                    needs(dense, "dense block (moe_layer_freq is 1)")?;
                    (names::ffn_down(l), matrix())
                }
                "block_sparse_moe.gate.weight" => {
                    needs(!dense, "routed block (moe_layer_freq is 0)")?;
                    (names::ffn_gate_inp(l), kept(Plain::F32, GgmlType::F32))
                }
                "block_sparse_moe.e_score_correction_bias" => {
                    needs(!dense, "routed block (moe_layer_freq is 0)")?;
                    (names::exp_probs_b(l), kept(Plain::F32, GgmlType::F32))
                }
                "block_sparse_moe.shared_experts.gate_proj.weight" => {
                    needs(!dense, "routed block (moe_layer_freq is 0)")?;
                    (names::ffn_gate_shexp(l), matrix())
                }
                "block_sparse_moe.shared_experts.up_proj.weight" => {
                    needs(!dense, "routed block (moe_layer_freq is 0)")?;
                    (names::ffn_up_shexp(l), matrix())
                }
                "block_sparse_moe.shared_experts.down_proj.weight" => {
                    needs(!dense, "routed block (moe_layer_freq is 0)")?;
                    (names::ffn_down_shexp(l), matrix())
                }
                _ => {
                    // `weight_packed`; a routed expert stored as a plain `weight` is the plan's
                    // to refuse, by its own reason.
                    let (id, projection) = rest
                        .strip_prefix("block_sparse_moe.experts.")
                        .and_then(|e| e.split_once('.'))
                        .and_then(|(id, p)| {
                            let projection = p
                                .strip_suffix(".weight_packed")
                                .or_else(|| p.strip_suffix(".weight"))?;
                            Some((id.parse::<usize>().ok()?, projection))
                        })
                        .filter(|(_, p)| ["w1", "w2", "w3"].contains(p))
                        .ok_or_else(|| refuse(name, "not a MiniMax-M3 tensor"))?;
                    needs(!dense, "routed block (moe_layer_freq is 0)")?;
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
            };
            put(to, m)?;
        }
        for ((l, projection), by_id) in stacks {
            let (sources, dims) = safetensors::expert_stack(&by_id, self.experts)
                .map_err(|e| refuse(&format!("layer {l} {projection}"), e.to_string()))?;
            // w1 is the gate, w3 the up and w2 the down projection of an expert's SwiGLU.
            let to = match projection {
                "w1" => names::ffn_gate_exps(l),
                "w3" => names::ffn_up_exps(l),
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
        for name in [names::token_embd(), names::output()] {
            let m = out
                .tensors
                .get(&name)
                .ok_or_else(|| refuse(&name, "is missing"))?;
            if m.dims != [self.hidden, u64::from(self.vocab.size())] {
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
        Ok(out)
    }
}

impl FolderMap for Checkpoint {
    type Step = PlusOne;

    const MODEL_TYPES: &'static [&'static str] = &["minimax_m3_vl"];

    fn parse(config: &Json) -> Result<Self, Error> {
        Checkpoint::read(config)
    }

    fn module_class(&self, module: &str) -> Option<ModuleClass> {
        module_class_of(module)
    }

    fn map<'a>(&self, tensors: &[(&'a str, &'a [usize])]) -> Result<Mapping<PlusOne>, Error> {
        self.mapped(tensors)
    }

    fn mover(&self, name: &str, steps: &[PlusOne]) -> Option<part::Step> {
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
        let mut keys = vec![(
            gguf::GENERAL_ARCHITECTURE.to_string(),
            Value::String("minimax-m3".into()),
        )];
        keys.extend(
            self.keys
                .iter()
                .map(|(suffix, v)| (format!("minimax-m3.{suffix}"), v.clone())),
        );
        Ok(keys)
    }

    fn vocab(&self) -> &VocabSpec {
        &self.vocab
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::Path;

    use gguf::Place;
    use hf::quant_config::Config;
    use model::arch::minimaxm3::hparams::{Attn, FfnKind, Hparams};
    use model::arch::minimaxm3::roles::classify;
    use model::part::{Source, Take};
    use safetensors::Dtype;
    use serde_json::json;

    use crate::folder::{self, plan_tensors};
    use crate::test_support::temp_dir;

    use super::*;

    /// One stored tensor, by what its header says: name, dtype and shape.
    type Entry = (String, &'static str, Vec<usize>);

    /// One stored tensor with its bytes.
    type Stored = (String, &'static str, Vec<usize>, Vec<u8>);

    /// The widths of a checkpoint, from which its config and its table of tensors are written.
    struct Dims {
        layers: usize,
        dense_lead: usize,
        sparse: Vec<bool>,
        hidden: usize,
        heads: usize,
        kv: usize,
        head: usize,
        vocab: usize,
        ff: usize,
        shared_ff: usize,
        dense_ff: usize,
        experts: usize,
        used: usize,
        idx_heads: usize,
        idx_dim: usize,
    }

    impl Dims {
        /// MiniMax-M3.1-Flash's.
        fn real() -> Dims {
            Dims {
                layers: 60,
                dense_lead: 3,
                sparse: vec![true; 60],
                hidden: 6144,
                heads: 64,
                kv: 4,
                head: 128,
                vocab: 200_064,
                ff: 3072,
                shared_ff: 3072,
                dense_ff: 12_288,
                experts: 128,
                used: 4,
                idx_heads: 4,
                idx_dim: 128,
            }
        }

        /// The fixture cut of the real folder (`tools/fixture-cuts/minimaxm31.json`): layers 0, 3
        /// and 59 as 0, 1 and 2, the routed width 512 and every other width the real one.
        fn cut() -> Dims {
            Dims {
                layers: 3,
                dense_lead: 1,
                sparse: vec![true; 3],
                ff: 512,
                ..Dims::real()
            }
        }

        /// One dense layer and one routed layer of four experts, every width a whole block of
        /// both the MXFP8 and the NVFP4 forms.
        fn small() -> Dims {
            Dims {
                layers: 2,
                dense_lead: 1,
                sparse: vec![true; 2],
                hidden: 64,
                heads: 4,
                kv: 2,
                head: 16,
                vocab: 8,
                ff: 64,
                shared_ff: 64,
                dense_ff: 96,
                experts: 4,
                used: 2,
                idx_heads: 2,
                idx_dim: 16,
            }
        }

        /// `config.json` as the checkpoint writes it, the vision tower's config included.
        fn config(&self) -> Json {
            let flags = |on: &dyn Fn(usize) -> bool| {
                (0..self.layers)
                    .map(|l| u8::from(on(l)))
                    .collect::<Vec<_>>()
            };
            let ignored: Vec<String> = (self.dense_lead..self.layers)
                .map(|l| format!("language_model.model.layers.{l}.block_sparse_moe.gate"))
                .chain(
                    [
                        "lm_head",
                        "model.embed_tokens",
                        "multi_modal_projector",
                        "patch_merge_mlp",
                        "vision_tower",
                    ]
                    .map(String::from),
                )
                .collect();
            let sparse = json!({
                "use_sparse_attention": true,
                "sparse_index_dim": self.idx_dim,
                "sparse_num_index_heads": self.idx_heads,
                "sparse_topk_blocks": 16,
                "sparse_block_size": 128,
                "sparse_score_type": "max",
                "sparse_init_block": 0,
                "sparse_local_block": 1,
                "sparse_attention_freq": flags(&|l| self.sparse[l]),
                "sparse_disable_index_value": flags(&|_| true)
            });
            let mut text = json!({
                "hidden_size": self.hidden,
                "intermediate_size": self.ff,
                "num_hidden_layers": self.layers,
                "num_attention_heads": self.heads,
                "num_key_value_heads": self.kv,
                "head_dim": self.head,
                "vocab_size": self.vocab,
                "max_position_embeddings": 1_048_576,
                "rms_norm_eps": 1e-06,
                "use_gemma_norm": true,
                "attention_output_gate": false,
                "rope_theta": 10_000_000,
                "rotary_dim": self.head / 2,
                "partial_rotary_factor": 0.5,
                "hidden_act": "swigluoai",
                "use_qk_norm": true,
                "tie_word_embeddings": false,
                "dense_intermediate_size": self.dense_ff,
                "shared_intermediate_size": self.shared_ff,
                "num_local_experts": self.experts,
                "num_experts_per_tok": self.used,
                "n_shared_experts": 1,
                "scoring_func": "sigmoid",
                "use_routing_bias": true,
                "moe_layer_freq": flags(&|l| l >= self.dense_lead),
                "qk_norm_type": "per_head",
                "swiglu_alpha": 1.702,
                "swiglu_limit": 7.0,
                "routed_scaling_factor": 2.0,
                "architectures": [TEXT_ARCHITECTURE]
            });
            text["sparse_attention_config"] = sparse;
            let quant = json!({
                "quant_method": "mxfp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [1, 32],
                "ignored_layers": ignored,
                "group_size": 16,
                "moe_quant_algo": "NVFP4",
                "moe_quant_format": "nvfp4-pack-quantized"
            });
            let mut config = json!({
                "architectures": [ARCHITECTURE],
                "auto_map": {"AutoConfig": "configuration_minimax_m3_vl.MiniMaxM3VLConfig"},
                "model_type": "minimax_m3_vl",
                "vision_config": {"hidden_size": 1280, "model_type": "clip_vision_model"},
                "img_token_compression_config": {"spatial_merge_size": 2},
                "image_grid_pinpoints": "[(336, 336)]",
                "image_seq_length": 576,
                "image_token_index": 200025,
                "video_token_index": 200026,
                "multimodal_projector_bias": true,
                "num_reward_heads": 0,
                "process_image_mode": "dynamic_res",
                "projector_hidden_act": "gelu",
                "vision_feature_layer": -1,
                "vision_feature_select_strategy": "full",
                "torch_dtype": "bfloat16",
                "transformers_version": "4.52.4",
                "projector_hidden_size": self.hidden
            });
            config["text_config"] = text;
            config["quantization_config"] = quant;
            config
        }

        /// Every tensor of the folder, as its header states it, written from the checkpoint's
        /// table and not from the map: the text model, then a sample of the vision tower and the
        /// draft that the map leaves out. A routed layer's experts come expert by expert.
        fn table(&self) -> Vec<Entry> {
            let mut out: Vec<Entry> = Vec::new();
            let mut add = |name: String, dtype, shape: &[usize]| {
                out.push((name, dtype, shape.to_vec()));
            };
            let (e, v) = (self.hidden, self.vocab);
            add(
                "language_model.model.embed_tokens.weight".into(),
                "BF16",
                &[v, e],
            );
            add("language_model.lm_head.weight".into(), "BF16", &[v, e]);
            add("language_model.model.norm.weight".into(), "BF16", &[e]);
            // An MXFP8 projection of `n` rows over `k` values: the values, then a scale a 32.
            let mxfp8 =
                |add: &mut dyn FnMut(String, &'static str, &[usize]), stem: String, n, k| {
                    add(format!("{stem}.weight"), "F8_E4M3", &[n, k]);
                    add(format!("{stem}.weight_scale_inv"), "U8", &[n, k / 32]);
                };
            let (q, kv) = (self.heads * self.head, self.kv * self.head);
            for l in 0..self.layers {
                let p = format!("language_model.model.layers.{l}");
                add(format!("{p}.input_layernorm.weight"), "BF16", &[e]);
                add(format!("{p}.post_attention_layernorm.weight"), "BF16", &[e]);
                add(format!("{p}.self_attn.q_norm.weight"), "BF16", &[self.head]);
                add(format!("{p}.self_attn.k_norm.weight"), "BF16", &[self.head]);
                mxfp8(&mut add, format!("{p}.self_attn.q_proj"), q, e);
                mxfp8(&mut add, format!("{p}.self_attn.k_proj"), kv, e);
                mxfp8(&mut add, format!("{p}.self_attn.v_proj"), kv, e);
                mxfp8(&mut add, format!("{p}.self_attn.o_proj"), e, q);
                if self.sparse[l] {
                    add(
                        format!("{p}.self_attn.index_q_norm.weight"),
                        "BF16",
                        &[self.idx_dim],
                    );
                    add(
                        format!("{p}.self_attn.index_k_norm.weight"),
                        "BF16",
                        &[self.idx_dim],
                    );
                    mxfp8(
                        &mut add,
                        format!("{p}.self_attn.index_q_proj"),
                        self.idx_heads * self.idx_dim,
                        e,
                    );
                    mxfp8(
                        &mut add,
                        format!("{p}.self_attn.index_k_proj"),
                        self.idx_dim,
                        e,
                    );
                }
                if l < self.dense_lead {
                    mxfp8(&mut add, format!("{p}.mlp.gate_proj"), self.dense_ff, e);
                    mxfp8(&mut add, format!("{p}.mlp.up_proj"), self.dense_ff, e);
                    mxfp8(&mut add, format!("{p}.mlp.down_proj"), e, self.dense_ff);
                    continue;
                }
                let m = format!("{p}.block_sparse_moe");
                add(format!("{m}.gate.weight"), "F32", &[self.experts, e]);
                add(
                    format!("{m}.e_score_correction_bias"),
                    "F32",
                    &[self.experts],
                );
                let sf = self.shared_ff;
                mxfp8(&mut add, format!("{m}.shared_experts.gate_proj"), sf, e);
                mxfp8(&mut add, format!("{m}.shared_experts.up_proj"), sf, e);
                mxfp8(&mut add, format!("{m}.shared_experts.down_proj"), e, sf);
                for x in 0..self.experts {
                    // w1 gate and w3 up are [ff, hidden], w2 down is [hidden, ff]: rows, then
                    // the values a row holds, two to a byte.
                    for (w, rows, k) in [("w1", self.ff, e), ("w2", e, self.ff), ("w3", self.ff, e)]
                    {
                        let x = format!("{m}.experts.{x}.{w}");
                        add(format!("{x}.weight_packed"), "U8", &[rows, k / 2]);
                        add(format!("{x}.weight_scale"), "F8_E4M3", &[rows, k / 16]);
                        add(format!("{x}.weight_global_scale"), "F32", &[]);
                        add(format!("{x}.input_global_scale"), "F32", &[]);
                    }
                }
            }
            add(
                "vision_tower.vision_model.pre_layrnorm.weight".into(),
                "BF16",
                &[1280],
            );
            add(
                "multi_modal_projector.linear_1.weight".into(),
                "BF16",
                &[e, 1280],
            );
            add(
                "patch_merge_mlp.linear_1.weight".into(),
                "BF16",
                &[e, 4 * e],
            );
            add(
                "language_model.model.dspark.fc.weight".into(),
                "BF16",
                &[e, 2 * e],
            );
            out
        }
    }

    /// The bytes of a stored tensor of `numel` values: a pattern of its name, but for the values
    /// a load reads at open (an expert's tensor scales) and the bytes MXFP8 and NVFP4 define.
    fn payload(name: &str, dtype: &str, numel: usize) -> Vec<u8> {
        let seed = name
            .bytes()
            .fold(7u32, |h, b| h.wrapping_mul(31) ^ u32::from(b));
        let tensor_scale = |base: f32| {
            let id = |s: &str| name.split(s).nth(1).and_then(|r| r.split('.').next());
            let x: f32 = id("experts.").and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let w = ["w1", "w2", "w3"]
                .iter()
                .position(|w| name.contains(&format!(".{w}.")));
            (base + 17.0 * x + 3.0 * w.unwrap_or(0) as f32)
                .to_le_bytes()
                .to_vec()
        };
        match (dtype, name) {
            (_, n) if n.ends_with(".weight_global_scale") => tensor_scale(8000.0),
            (_, n) if n.ends_with(".input_global_scale") => 1.0f32.to_le_bytes().to_vec(),
            ("F8_E4M3", _) => vec![0x38; numel],
            ("U8", n) if n.ends_with(".weight_scale_inv") => vec![127; numel],
            _ => {
                let size = Dtype::of(dtype).unwrap().size();
                (0..numel * size)
                    .map(|i| (seed as usize).wrapping_add(i * 7) as u8)
                    .collect()
            }
        }
    }

    /// How a routed layer's experts lie in the shard.
    #[derive(Clone, Copy, PartialEq)]
    enum Planes {
        /// Each expert's matrices one after another, codes then block scales, then the tensor
        /// scales: the experts out of order.
        Adjacent,
        /// Every block scale, then every code plane, as most of the real shards keep them: each
        /// plane in an order of its own.
        Apart,
    }

    /// The synthetic tokenizer every folder here carries: MiniMax's pre-tokenizer and decoder as
    /// its `tokenizer.json` spells them, over a vocabulary of three tokens and two added ones.
    fn write_tokenizer(dir: &Path, vocab: usize) {
        let regex = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        let tokenizer = json!({
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type":"Split","pattern":{"Regex":regex},"behavior":"Removed","invert":true},
                {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},
            "decoder": {"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
            "post_processor": null,
            "model": {"type":"BPE","vocab":{"A":0,"a":1,"Aa":vocab - 1},"merges":[["A","a"]]},
            "added_tokens": [
                {"id":2,"content":"<s>","special":true},
                {"id":3,"content":"</s>","special":true}]
        });
        for (file, value) in [
            ("tokenizer.json", tokenizer),
            (
                "tokenizer_config.json",
                json!({"bos_token":"<s>","eos_token":"</s>"}),
            ),
            ("generation_config.json", json!({})),
        ] {
            std::fs::write(dir.join(file), value.to_string()).unwrap();
        }
    }

    /// The folder of `dims` in `dir`: shard 1 holds the unlayered tensors and layer 0, shard 2 the
    /// other layers. `tweak` changes the config and the stored tensors before they are written.
    fn write_folder(
        dir: &Path,
        dims: &Dims,
        planes: Planes,
        tweak: impl FnOnce(&mut Json, &mut Vec<Stored>),
    ) {
        let mut config = dims.config();
        let mut stored: Vec<Stored> = dims
            .table()
            .into_iter()
            .map(|(n, d, s)| {
                let bytes = payload(&n, d, s.iter().product());
                (n, d, s, bytes)
            })
            .collect();
        tweak(&mut config, &mut stored);
        if planes == Planes::Apart {
            let expert = |n: &str| {
                let x: usize = n
                    .split("experts.")
                    .nth(1)?
                    .split('.')
                    .next()?
                    .parse()
                    .ok()?;
                Some((x, n.ends_with("weight_scale")))
            };
            let is_plane = |s: &Stored| expert(&s.0).is_some() && !s.0.ends_with("_global_scale");
            let (mut planes, rest): (Vec<Stored>, Vec<Stored>) =
                stored.into_iter().partition(is_plane);
            // Scales in one order of experts, codes in another, each plane's matrices w1 w2 w3.
            let order = |scale: bool| {
                move |s: &Stored| {
                    let (x, _) = expert(&s.0).unwrap();
                    let rank = if scale {
                        [1, 3, 0, 2][x % 4]
                    } else {
                        [2, 0, 3, 1][x % 4]
                    };
                    let layer =
                        s.0.split("layers.")
                            .nth(1)
                            .unwrap()
                            .split('.')
                            .next()
                            .unwrap();
                    (layer.parse::<usize>().unwrap(), x / 4, rank, s.0.clone())
                }
            };
            let mut scales: Vec<Stored> = planes
                .extract_if(.., |s| s.0.ends_with("weight_scale"))
                .collect();
            scales.sort_by_key(order(true));
            planes.sort_by_key(order(false));
            stored = rest;
            stored.extend(scales);
            stored.extend(planes);
        } else {
            // Expert by expert, in an order of their own.
            let rank = |s: &Stored| {
                let x =
                    s.0.split("experts.")
                        .nth(1)
                        .and_then(|r| r.split('.').next());
                x.and_then(|x| x.parse::<usize>().ok())
                    .map(|x| [2, 0, 3, 1][x % 4])
            };
            let (mut experts, rest): (Vec<Stored>, Vec<Stored>) =
                stored.into_iter().partition(|s| rank(s).is_some());
            experts.sort_by_key(|s| (rank(s), s.0.clone()));
            stored = rest;
            stored.extend(experts);
        }
        let shard = |name: &str| {
            let layer = name
                .strip_prefix(LAYERS)
                .and_then(|r| r.split('.').next())
                .and_then(|l| l.parse::<usize>().ok());
            if layer.is_some_and(|l| l >= 1) {
                "b"
            } else {
                "a"
            }
        };
        let mut map = BTreeMap::new();
        for file in ["a", "b"] {
            let view: Vec<(&str, &str, &[usize], &[u8])> = stored
                .iter()
                .filter(|s| shard(&s.0) == file)
                .map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), b.as_slice()))
                .collect();
            let name = format!(
                "model-0000{}-of-00002.safetensors",
                if file == "a" { 1 } else { 2 }
            );
            std::fs::write(dir.join(&name), safetensors::write(&view)).unwrap();
            map.extend(view.iter().map(|t| (t.0.to_string(), name.clone())));
        }
        std::fs::write(
            dir.join(Folder::INDEX),
            json!({"weight_map": map}).to_string(),
        )
        .unwrap();
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        write_tokenizer(dir, dims.vocab);
    }

    fn open_small(
        planes: Planes,
        tweak: impl FnOnce(&mut Json, &mut Vec<Stored>),
    ) -> Result<folder::Opened, String> {
        let dir = temp_dir("minimaxm3");
        write_folder(&dir, &Dims::small(), planes, tweak);
        let opened = folder::open(&dir).map_err(|e| e.to_string());
        std::fs::remove_dir_all(&dir).unwrap();
        opened
    }

    fn untouched(_: &mut Json, _: &mut Vec<Stored>) {}

    /// The weights of `text` that `plan_tensors` is given as entries, its dtype read.
    fn dtyped(entries: &[Entry]) -> Vec<(String, Dtype, Vec<usize>)> {
        entries
            .iter()
            .map(|(n, d, s)| (n.clone(), Dtype::of(d).unwrap(), s.clone()))
            .collect()
    }

    /// The real checkpoint's `config.json`, as the hf crate keeps it.
    fn real_config() -> Json {
        use std::io::Read;
        let path = format!(
            "{}/../hf/tests/fixtures/quant-configs/MiniMaxAI__MiniMax-M3.1-Flash__config.json.gz",
            env!("CARGO_MANIFEST_DIR")
        );
        let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut text = String::new();
        flate2::read::GzDecoder::new(file)
            .read_to_string(&mut text)
            .unwrap();
        serde_json::from_str(&text).unwrap()
    }

    /// The text model's bytes in the folder's own tensors that the map reads, from the table: a
    /// tensor's values and the scales stored beside them.
    fn folder_bytes(entries: &[Entry]) -> u64 {
        entries
            .iter()
            .filter(|(n, ..)| skip_reason(n).is_none())
            .map(|(_, d, s)| (s.iter().product::<usize>() * Dtype::of(d).unwrap().size()) as u64)
            .sum()
    }

    /// The real config's geometry and keys: sixty layers whose first three are dense blocks, 128
    /// experts of which four route, the indexer on every layer, and the keys the reader asks of
    /// an M3 file with `attention.sparse_layers` beside them.
    #[test]
    fn the_real_config_reads_as_the_m3_reader_asks() {
        let c = Checkpoint::read(&real_config()).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            (c.layers, c.experts, c.dense_lead, c.hidden),
            (60, 128, 3, 6144)
        );
        assert_eq!(c.sparse, vec![true; 60]);
        assert_eq!(c.vocab.size(), 200_064);
        let want: Vec<(&str, Value)> = vec![
            ("block_count", Value::U32(60)),
            ("context_length", Value::U32(1_048_576)),
            ("embedding_length", Value::U32(6144)),
            ("feed_forward_length", Value::U32(12_288)),
            ("attention.head_count", Value::U32(64)),
            ("attention.head_count_kv", Value::U32(4)),
            ("attention.key_length", Value::U32(128)),
            ("attention.value_length", Value::U32(128)),
            ("attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
            ("rope.freq_base", Value::F32(1e7)),
            ("rope.dimension_count", Value::U32(64)),
            ("expert_count", Value::U32(128)),
            ("expert_used_count", Value::U32(4)),
            ("expert_feed_forward_length", Value::U32(3072)),
            ("expert_shared_count", Value::U32(1)),
            ("expert_shared_feed_forward_length", Value::U32(3072)),
            ("expert_gating_func", Value::U32(2)),
            ("expert_weights_scale", Value::F32(2.0)),
            ("expert_weights_norm", Value::Bool(true)),
            ("leading_dense_block_count", Value::U32(3)),
            ("attention.indexer.head_count", Value::U32(4)),
            ("attention.indexer.key_length", Value::U32(128)),
            ("attention.indexer.top_k", Value::U32(16)),
            ("attention.indexer.block_size", Value::U32(128)),
            ("attention.indexer.local_blocks", Value::U32(1)),
            (
                "attention.sparse_layers",
                Value::Array(vec![Value::Bool(true); 60]),
            ),
        ];
        assert_eq!(c.keys, want);
    }

    /// Every tensor of the real checkpoint's text model, written from its table at the real
    /// widths, is mapped, and the vision and draft tensors are skipped with their reason. The
    /// folder holds 248,369,266,176 bytes of text (the census of its shard headers: every
    /// tensor but the vision tower's); the header declares the same tensors in 1,460,736 more:
    /// the norms' gains as F32 (1,548,288 more), less the 175,104 of the experts' tensor
    /// scales, which the header keeps as 171 `.scale` tensors of 512 (87,552).
    #[test]
    fn the_real_table_maps_whole_and_counts_as_its_shard_headers_do() {
        let config = real_config();
        let checkpoint = Checkpoint::read(&config).unwrap();
        let table = Dims::real().table();
        assert_eq!(folder_bytes(&table), 248_369_266_176);
        let quant = Config::parse(&config, None).unwrap();
        let plan = plan_tensors(&checkpoint, &quant, &dtyped(&table), false)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.skipped.len(), 4);
        assert!(
            plan.skipped
                .values()
                .all(|why| *why == VISION || *why == DRAFT),
            "{:?}",
            plan.skipped
        );
        // 3 model-level tensors, 12 a layer (8 and the indexer's 4), 3 dense blocks of 3, 57
        // routed blocks of 8 and their 3 `.scale` tensors.
        assert_eq!(plan.planned.len(), 3 + 60 * 12 + 3 * 3 + 57 * (8 + 3));
        let bytes = |pick: &dyn Fn(&str) -> bool| -> u64 {
            plan.planned
                .iter()
                .filter(|p| pick(&p.decl.name))
                .map(|p| p.decl.nbytes)
                .sum()
        };
        assert_eq!(bytes(&|_| true), 248_370_726_912);
        assert_eq!(
            bytes(&|n| n.contains("_exps.weight")),
            232_381_218_816,
            "the experts' codes and block scales: 57 layers of 128 experts of 31,850,496"
        );
        assert_eq!(bytes(&|n| n.ends_with("_exps.scale")), 87_552);
        assert_eq!(
            bytes(&|n| n.contains("norm.weight")),
            3_035_136 + 30_720 * 2
        );
        for p in &plan.planned {
            let ty = GgmlType::from_u32(p.decl.type_id);
            let want = match p.decl.name.as_str() {
                n if n.ends_with("_exps.weight") => GgmlType::NVFP4_ROWS,
                n if n.ends_with(".scale")
                    || n.contains("norm.weight")
                    || n.ends_with("ffn_gate_inp.weight")
                    || n.ends_with("exp_probs_b.bias") =>
                {
                    GgmlType::F32
                }
                "token_embd.weight" | "output.weight" => GgmlType::BF16,
                _ => GgmlType::MXFP8,
            };
            assert_eq!(ty, want, "{}", p.decl.name);
        }
    }

    /// The folder's tensors the map accounts for: each is a source of a header tensor, skipped
    /// with its reason, or a companion of a source (its scales).
    fn assert_accounted(opened: &folder::Opened, table: &[Entry]) {
        let sources: BTreeSet<&str> = opened
            .source
            .made()
            .values()
            .flat_map(|m| m.sources.iter().map(String::as_str))
            .collect();
        for (name, ..) in table {
            let companion = [
                "weight_scale_inv",
                "weight_scale",
                "weight_global_scale",
                "input_global_scale",
            ]
            .iter()
            .find_map(|part| name.strip_suffix(part)?.strip_suffix('.'))
            .is_some_and(|stem| {
                sources.contains(format!("{stem}.weight").as_str())
                    || sources.contains(format!("{stem}.weight_packed").as_str())
            });
            assert!(
                sources.contains(name.as_str()) || opened.skipped.contains_key(name) || companion,
                "{name} is neither read, skipped nor a companion"
            );
        }
    }

    /// A two-layer folder (a dense block, then four routed experts) opens through `folder::open`
    /// as a located split the M3 reader reads: its keys, every tensor's role, the tensors the
    /// host reads in place as the folder's bytes, each routed projection a stack of planes with
    /// each expert's scale the reciprocal of its `weight_global_scale`, and the vision tower and
    /// the draft skipped with their reason. The experts' planes lie each beside their block
    /// scales, or apart (every scale, then every code plane, in two orders of experts).
    #[test]
    fn a_two_layer_folder_opens_as_the_m3_reader_reads_it() {
        let table = Dims::small().table();
        for planes in [Planes::Adjacent, Planes::Apart] {
            let dir = temp_dir("minimaxm3-open");
            write_folder(&dir, &Dims::small(), planes, untouched);
            let opened = folder::open(&dir).unwrap_or_else(|e| panic!("{e}"));
            let split = &opened.header;
            assert_eq!(split.architecture(), Some("minimax-m3"));
            assert_accounted(&opened, &table);
            assert_eq!(
                opened
                    .skipped
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                [
                    "language_model.model.dspark.fc.weight",
                    "multi_modal_projector.linear_1.weight",
                    "patch_merge_mlp.linear_1.weight",
                    "vision_tower.vision_model.pre_layrnorm.weight",
                ]
            );
            assert_eq!(opened.skipped["patch_merge_mlp.linear_1.weight"], VISION);
            assert_eq!(
                opened.skipped["language_model.model.dspark.fc.weight"],
                DRAFT
            );

            let hp = Hparams::read(split).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!((hp.n_layer, hp.dense_lead, hp.n_vocab), (2, 1, 8));
            assert_eq!(hp.sparse_attn, [true, true]);
            assert_eq!(
                (hp.kind(0).attn, hp.kind(0).ffn),
                (Attn::Sparse, FfnKind::Dense)
            );
            assert_eq!(
                (hp.kind(1).attn, hp.kind(1).ffn),
                (Attn::Sparse, FfnKind::Routed)
            );
            assert_eq!(
                (hp.experts.n_expert, hp.experts.n_used, hp.experts.ff),
                (4, 2, 64)
            );
            assert_eq!(hp.dense_ff, Some(96));
            assert_eq!(hp.rope.dims, 8);
            assert_eq!((hp.msa.heads, hp.msa.d), (2, 16));
            let roles = classify(split, &hp).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(roles.tensors.len(), split.tensor_count());

            // Kept where the folder stores them.
            let stored = |name: &str, dtype: &str, shape: &[usize]| {
                payload(name, dtype, shape.iter().product())
            };
            for (to, from, dtype, shape) in [
                (
                    "token_embd.weight",
                    "language_model.model.embed_tokens.weight",
                    "BF16",
                    vec![8, 64],
                ),
                (
                    "output.weight",
                    "language_model.lm_head.weight",
                    "BF16",
                    vec![8, 64],
                ),
                (
                    "blk.1.ffn_gate_inp.weight",
                    "language_model.model.layers.1.block_sparse_moe.gate.weight",
                    "F32",
                    vec![4, 64],
                ),
                (
                    "blk.1.exp_probs_b.bias",
                    "language_model.model.layers.1.block_sparse_moe.e_score_correction_bias",
                    "F32",
                    vec![4],
                ),
            ] {
                assert!(matches!(split.place(to), Some(Place::File { .. })), "{to}");
                assert_eq!(split.data(to).unwrap(), stored(from, dtype, &shape), "{to}");
            }
            // The routed stacks: w1 is the gate, w3 the up and w2 the down.
            let moe = "language_model.model.layers.1.block_sparse_moe.experts";
            for (to, w, k, rows) in [
                ("blk.1.ffn_gate_exps.weight", "w1", 64, 64),
                ("blk.1.ffn_up_exps.weight", "w3", 64, 64),
                ("blk.1.ffn_down_exps.weight", "w2", 64, 64),
            ] {
                let Some(Place::Stack {
                    codes_per,
                    scales_per,
                    layout,
                    scales,
                    codes_at,
                    scales_at,
                    ..
                }) = split.place(to)
                else {
                    panic!("{to} is no stack");
                };
                assert_eq!(
                    (*layout, *codes_per, *scales_per),
                    (
                        StackLayout::Planes,
                        (rows * k / 2) as u64,
                        (rows * k / 16) as u64
                    ),
                    "{to}"
                );
                let (_, info) = split.find(to).unwrap_or_else(|| panic!("{to}"));
                assert_eq!(
                    (info.ty, info.dims.clone()),
                    (GgmlType::NVFP4_ROWS, vec![k as u64, rows as u64, 4]),
                    "{to}"
                );
                assert_eq!(scales.len(), 4, "{to}");
                for (e, scale) in scales.iter().enumerate() {
                    let part = |p: &str, dtype, shape: &[usize]| {
                        stored(&format!("{moe}.{e}.{w}.{p}"), dtype, shape)
                    };
                    let m = split.matrix(to, e).unwrap();
                    assert_eq!(
                        m.codes,
                        part("weight_packed", "U8", &[rows, k / 2]),
                        "{to} {e}"
                    );
                    assert_eq!(
                        m.block_scales,
                        part("weight_scale", "F8_E4M3", &[rows, k / 16]),
                        "{to} {e}"
                    );
                    let g = f32::from_le_bytes(
                        part("weight_global_scale", "F32", &[]).try_into().unwrap(),
                    );
                    assert_eq!(scale.to_bits(), (1.0 / g).to_bits(), "{to} {e}");
                }
                let beside = (0..4).all(|e| scales_at[e] == codes_at[e] + codes_per);
                assert_eq!(
                    beside,
                    planes == Planes::Adjacent,
                    "{to}: {codes_at:?} {scales_at:?}"
                );
                if planes == Planes::Apart {
                    let last_code = codes_at.iter().max().unwrap() + codes_per;
                    assert!(
                        scales_at
                            .iter()
                            .all(|&at| at + scales_per <= *codes_at.iter().min().unwrap())
                            || codes_at
                                .iter()
                                .all(|&at| at + codes_per <= *scales_at.iter().min().unwrap()),
                        "every scale plane lies before every code plane, or after: {codes_at:?} {scales_at:?} {last_code}"
                    );
                }
            }
            for name in [
                "blk.1.ffn_gate_exps.scale",
                "blk.0.attn_norm.weight",
                "blk.0.attn_q.weight",
            ] {
                assert_eq!(split.place(name), Some(&Place::Made), "{name}");
            }
            let (_, scale) = split.find("blk.1.ffn_up_exps.scale").unwrap();
            assert_eq!((scale.ty, scale.dims.clone()), (GgmlType::F32, vec![4]));
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }

    /// A folder of any map prints its plan through the common function, with no printing of the
    /// map's own: the two-layer folder reads four tensors in place (the embedding and the head, 8
    /// x 64 BF16, the router 4 x 64 F32 and its bias, 4 F32: 3,088 bytes), three stacks of four
    /// experts' planes (3 x 4 x (2,048 + 256) bytes) and makes the rest, and the vision tower, its
    /// projector, the patch-merge MLP and the draft are skipped by their reasons.
    #[test]
    fn a_two_layer_folder_prints_its_plan_by_the_common_function() {
        let dir = temp_dir("minimaxm3-plan");
        write_folder(&dir, &Dims::small(), Planes::Adjacent, untouched);
        let opened = folder::open(&dir).unwrap_or_else(|e| panic!("{e}"));
        std::fs::remove_dir_all(&dir).unwrap();
        let lines = folder::plan_lines(&opened.header, &opened.skipped);
        let tensors = opened.header.tensor_count();
        assert!(
            lines[0].starts_with(&format!("minimax-m3: {tensors} tensors, "))
                && lines[0].ends_with("; 4 folder tensors skipped"),
            "{lines:?}"
        );
        for want in [
            "4 tensors, 3088 bytes read in place from the folder's files".to_string(),
            "3 tensors, 27648 bytes stacks of matrices read in place".to_string(),
            format!("skipped 3 folder tensors: {VISION}"),
            format!("skipped 1 folder tensors: {DRAFT}"),
        ] {
            assert!(lines.contains(&want), "{want}: {lines:?}");
        }
        let made = format!("{} tensors, ", tensors - 7);
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with(&made)
                    && l.ends_with("made at load from the folder's tensors")),
            "{lines:?}"
        );
    }

    /// A shared expert of a width of its own (the fixture cut narrows the routed experts and keeps
    /// the shared one whole) is a key of its own: the map writes `expert_shared_feed_forward_length`
    /// from `shared_intermediate_size` and the shared tensors keep their stored dims.
    #[test]
    fn a_shared_expert_wider_than_the_routed_ones_keeps_its_width() {
        let dims = Dims {
            shared_ff: 96,
            ..Dims::small()
        };
        let dir = temp_dir("minimaxm3-shared");
        write_folder(&dir, &dims, Planes::Adjacent, untouched);
        let opened = folder::open(&dir).unwrap_or_else(|e| panic!("{e}"));
        std::fs::remove_dir_all(&dir).unwrap();
        let split = &opened.header;
        let key = |k: &str| split.value(&format!("minimax-m3.{k}")).cloned();
        assert_eq!(key("expert_feed_forward_length"), Some(Value::U32(64)));
        assert_eq!(
            key("expert_shared_feed_forward_length"),
            Some(Value::U32(96))
        );
        for (name, dims) in [
            ("blk.1.ffn_gate_shexp.weight", vec![64, 96]),
            ("blk.1.ffn_up_shexp.weight", vec![64, 96]),
            ("blk.1.ffn_down_shexp.weight", vec![96, 64]),
        ] {
            let (_, info) = split.find(name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(info.dims, dims, "{name}");
        }
    }

    /// The fixture cut's config and tensors (its table is the one the cut's own test derives from
    /// the real shard headers: 3,151 tensors, 6,975,526,912 bytes) map whole: every tensor read or
    /// skipped, three layers of which the first is dense, the routed width 512 beside a shared
    /// width of 3072.
    #[test]
    fn the_fixture_cut_maps_whole() {
        let mut config = Dims::cut().config();
        let checkpoint = Checkpoint::read(&config).unwrap_or_else(|e| panic!("{e}"));
        let table = Dims::cut().table();
        let text = table.iter().filter(|(n, ..)| skip_reason(n).is_none());
        assert_eq!(text.count(), 3151);
        assert_eq!(folder_bytes(&table), 6_975_526_912);
        let key = |k: &str| {
            checkpoint
                .keys
                .iter()
                .find(|(name, _)| *name == k)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(key("block_count"), Some(Value::U32(3)));
        assert_eq!(key("leading_dense_block_count"), Some(Value::U32(1)));
        assert_eq!(key("expert_feed_forward_length"), Some(Value::U32(512)));
        assert_eq!(
            key("expert_shared_feed_forward_length"),
            Some(Value::U32(3072))
        );
        // The cut keeps the quantization block as it is.
        config["text_config"]["num_hidden_layers"] = json!(3);
        let quant = Config::parse(&config, None).unwrap();
        let plan = plan_tensors(&checkpoint, &quant, &dtyped(&table), false)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(plan.skipped.len(), 4);
        // 3 model-level tensors, 12 a layer, a dense block of 3, two routed blocks of 8 and
        // their 3 `.scale` tensors.
        assert_eq!(plan.planned.len(), 3 + 3 * 12 + 3 + 2 * (8 + 3));
        // The folder's bytes, the norms' gains as F32 (89,088 more), less the experts' 6,144 of
        // tensor scalars, plus 6 `.scale` tensors of 512.
        let header: u64 = plan.planned.iter().map(|p| p.decl.nbytes).sum();
        assert_eq!(header, 6_975_526_912 + 89_088 - 6_144 + 3_072);
    }

    /// The real folder, or the fixture cut of it, at `$BLOOMERY_MM31_FOLDER` opened by its map
    /// (headers and tokenizer, no tensor bytes): every tensor the folder stores is read, skipped
    /// or a companion, the M3 reader reads the header and classifies every tensor, and the
    /// counts and bytes are those the tables above derive.
    #[test]
    #[ignore = "needs the MiniMax-M3.1-Flash folder, or its fixture cut, at $BLOOMERY_MM31_FOLDER"]
    fn hw_the_minimax_folder_opens_as_the_m3_reader_reads_it() {
        let dir = std::path::PathBuf::from(
            std::env::var_os("BLOOMERY_MM31_FOLDER")
                .expect("$BLOOMERY_MM31_FOLDER names the folder"),
        );
        let folder = Folder::open(&dir).unwrap_or_else(|e| panic!("{e}"));
        let table: Vec<Entry> = folder
            .names()
            .map(|n| {
                let raw = folder.raw(n).unwrap_or_else(|e| panic!("{n}: {e}"));
                (n.to_string(), raw.dtype.name(), raw.shape.to_vec())
            })
            .collect();
        let opened = folder::open(&dir).unwrap_or_else(|e| panic!("{e}"));
        let split = &opened.header;
        assert_accounted(&opened, &table);
        let hp = Hparams::read(split).unwrap_or_else(|e| panic!("{e}"));
        let roles = classify(split, &hp).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(roles.tensors.len(), split.tensor_count());
        let text = folder_bytes(&table);
        let header: u64 = split.iter_tensors().map(|(_, t)| t.nbytes).sum();
        println!(
            "minimax folder {}: layers={} dense_lead={} folder_tensors={} header_tensors={} skipped={} \
             folder_text_bytes={text} header_bytes={header}",
            dir.display(),
            hp.n_layer,
            hp.dense_lead,
            table.len(),
            split.tensor_count(),
            opened.skipped.len()
        );
        let (tensors, stored, declared) = match hp.n_layer {
            60 => (
                3 + 60 * 12 + 3 * 3 + 57 * 11,
                248_369_266_176,
                248_370_726_912,
            ),
            3 => (
                3 + 3 * 12 + 3 + 2 * 11,
                6_975_526_912,
                6_975_526_912 + 89_088 - 6_144 + 3_072,
            ),
            n => panic!("{n} layers: neither the real folder's nor the cut's"),
        };
        assert_eq!(split.tensor_count(), tensors);
        assert_eq!((text, header), (stored, declared));
    }

    /// `config` with the value at `pointer` (a JSON pointer whose parent exists) set to `value`.
    fn set(config: &mut Json, pointer: &str, value: Json) {
        let (parent, key) = pointer.rsplit_once('/').unwrap();
        let object = config.pointer_mut(parent).unwrap().as_object_mut().unwrap();
        object.insert(key.to_string(), value);
    }

    /// A config value the program does not run, or a field the map does not know, is refused by
    /// name when the config is read, before any tensor is: each of the values the program holds
    /// to one, the layer patterns it has no way to run, widths that disagree, and an MTP field.
    #[test]
    fn a_config_the_program_does_not_run_is_refused_by_name() {
        let cases: Vec<(String, Json, String)> = serde_json::from_str(
            r#"[
            ["/foo", 1, "config.foo is not a key this converter reads"],
            ["/model_type", "minimax_m3", "config.model_type \"minimax_m3\" is unsupported"],
            ["/architectures", ["MiniMaxM3SparseForCausalLM"], "config.architectures"],
            ["/tie_word_embeddings", true, "config.tie_word_embeddings true is unsupported"],
            ["/num_reward_heads", 1, "config.num_reward_heads 1 is unsupported"],
            ["/text_config/architectures", ["Other"], "text_config.architectures"],
            ["/text_config/rope_scaling", {"type": "yarn"}, "text_config.rope_scaling is not a key this converter reads"],
            ["/text_config/hidden_act", "silu", "text_config.hidden_act \"silu\" is unsupported; supported: [\"swigluoai\"]"],
            ["/text_config/swiglu_alpha", 1.5, "text_config.swiglu_alpha 1.5 is unsupported"],
            ["/text_config/swiglu_limit", 6.0, "text_config.swiglu_limit 6.0 is unsupported"],
            ["/text_config/use_gemma_norm", false, "text_config.use_gemma_norm false is unsupported; supported: [true]"],
            ["/text_config/use_qk_norm", false, "text_config.use_qk_norm false is unsupported"],
            ["/text_config/qk_norm_type", "per_layer", "text_config.qk_norm_type \"per_layer\" is unsupported"],
            ["/text_config/attention_output_gate", true, "text_config.attention_output_gate true is unsupported"],
            ["/text_config/use_routing_bias", false, "text_config.use_routing_bias false is unsupported"],
            ["/text_config/scoring_func", "softmax", "text_config.scoring_func \"softmax\" is unsupported"],
            ["/text_config/tie_word_embeddings", true, "text_config.tie_word_embeddings true is unsupported"],
            ["/text_config/use_mtp", true, "text_config.use_mtp true is unsupported; supported: [false]"],
            ["/text_config/num_mtp_modules", 3, "text_config.num_mtp_modules 3 is unsupported"],
            ["/text_config/mtp_transformer_layers", 1, "text_config.mtp_transformer_layers 1 is unsupported"],
            ["/text_config/num_nextn_predict_layers", 1, "text_config.num_nextn_predict_layers 1 is unsupported"],
            ["/text_config/mtp_num_hidden_layers", 1, "text_config.mtp_num_hidden_layers 1 is unsupported"],
            ["/text_config/sparse_attention_config/sparse_score_type", "mean", "sparse_score_type \"mean\" is unsupported"],
            ["/text_config/sparse_attention_config/sparse_init_block", 1, "sparse_init_block 1 is unsupported"],
            ["/text_config/sparse_attention_config/use_sparse_attention", false, "use_sparse_attention false is unsupported"],
            ["/text_config/sparse_attention_config/sparse_window", 4, "sparse_window is not a key this converter reads"],
            ["/text_config/sparse_attention_config/sparse_disable_index_value", [1, 0], "sparse_disable_index_value is not 1 on every layer"],
            ["/text_config/sparse_attention_config/sparse_disable_index_value", [1], "sparse_disable_index_value has 1 entries for 2 layers"],
            ["/text_config/sparse_attention_config/sparse_attention_freq", [1, 2], "sparse_attention_freq holds 2 at layer 1, not 0 or 1"],
            ["/text_config/moe_layer_freq", [1, 0], "moe_layer_freq is not dense layers and then routed ones"],
            ["/text_config/moe_layer_freq", [0, 1, 1], "moe_layer_freq has 3 entries for 2 layers"],
            ["/text_config/partial_rotary_factor", 0.25, "rotary_dim must be head_dim times partial_rotary_factor"],
            ["/text_config/num_experts_per_tok", 5, "num_experts_per_tok exceeds num_local_experts"],
            ["/text_config/num_key_value_heads", 3, "num_attention_heads must group by num_key_value_heads"],
            ["/text_config/hidden_size", 0, "hidden_size must be a positive u32"],
            ["/text_config/rms_norm_eps", -1.0, "rms_norm_eps must be a finite positive number"]
            ]"#,
        )
        .unwrap();
        for (pointer, value, want) in cases {
            let mut config = Dims::small().config();
            set(&mut config, &pointer, value);
            let e = Checkpoint::read(&config).expect_err(&pointer).to_string();
            assert!(e.contains(&want), "{pointer}: {e}");
        }
        let mut config = Dims::small().config();
        config["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("vocab_size");
        let e = Checkpoint::read(&config).unwrap_err().to_string();
        assert!(e.contains("vocab_size"), "{e}");
    }

    /// The same refusals arrive at `folder::open`, before the folder's tensors are read.
    #[test]
    fn a_folder_with_such_a_config_is_refused_at_open() {
        for (pointer, value, want) in [
            (
                "/text_config/rope_scaling",
                json!({"type": "yarn"}),
                "text_config.rope_scaling is not a key this converter reads",
            ),
            (
                "/text_config/use_gemma_norm",
                json!(false),
                "text_config.use_gemma_norm false is unsupported",
            ),
            (
                "/text_config/use_mtp",
                json!(true),
                "text_config.use_mtp true is unsupported",
            ),
        ] {
            let e = open_small(Planes::Adjacent, |c, _| set(c, pointer, value))
                .err()
                .expect(pointer);
            assert!(e.contains(want), "{pointer}: {e}");
        }
    }

    /// A routed expert's `input_global_scale` that is not a finite positive F32 is refused by name
    /// at open: the scale is read, checked and used by nothing.
    #[test]
    fn an_input_global_scale_that_is_not_finite_and_positive_is_refused_by_name() {
        let name = "language_model.model.layers.1.block_sparse_moe.experts.2.w3.input_global_scale";
        for (value, shown) in [
            (f32::NAN, "NaN"),
            (f32::INFINITY, "inf"),
            (0.0, "0"),
            (-1.0, "-1"),
        ] {
            let e = open_small(Planes::Adjacent, |_, stored| {
                let t = stored.iter_mut().find(|t| t.0 == name).unwrap();
                t.3 = value.to_le_bytes().to_vec();
            })
            .err()
            .expect(shown);
            assert!(
                e.contains(&format!(
                    "input scale {name} is {shown}, not a finite positive F32"
                )),
                "{shown}: {e}"
            );
        }
        // Any other positive value opens: it is no input of any arithmetic here.
        open_small(Planes::Adjacent, |_, stored| {
            let t = stored.iter_mut().find(|t| t.0 == name).unwrap();
            t.3 = 3.5f32.to_le_bytes().to_vec();
        })
        .unwrap();
    }

    /// A folder tensor no row of the map reads is refused by name, as is one the config says its
    /// layer has none of: an indexer on a layer of full attention, a dense block on a routed
    /// layer, a routed one on a dense layer, a layer past the model, and a routed projection whose
    /// experts are not every one of the config's.
    #[test]
    fn a_tensor_no_row_reads_is_refused_by_name() {
        let tensor = |name: &str| -> Stored { (name.to_string(), "F32", vec![2], vec![0; 8]) };
        let l = "language_model.model.layers";
        for (extra, want) in [
            ("stray.weight", "stray.weight: not a MiniMax-M3 tensor"),
            (
                "language_model.model.layers.0.self_attn.rotary_emb.inv_freq",
                "not a MiniMax-M3 tensor",
            ),
            (
                "language_model.model.layers.7.input_layernorm.weight",
                "layer 7 of 2",
            ),
            (
                "language_model.model.layers.1.mlp.gate_proj.weight",
                "layer 1 has no dense block (moe_layer_freq is 1)",
            ),
            (
                "language_model.model.layers.0.block_sparse_moe.gate.weight",
                "layer 0 has no routed block (moe_layer_freq is 0)",
            ),
            (
                "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_packed",
                "layer 0 has no routed block",
            ),
            (
                "language_model.model.layers.1.block_sparse_moe.experts.x.w1.weight_packed",
                "not a MiniMax-M3 tensor",
            ),
            (
                "language_model.model.layers.1.block_sparse_moe.experts.0.w4.weight_packed",
                "not a MiniMax-M3 tensor",
            ),
        ] {
            let e = open_small(Planes::Adjacent, |_, stored| stored.push(tensor(extra)))
                .err()
                .expect(extra);
            assert!(e.contains(want), "{extra}: {e}");
        }
        let e = open_small(Planes::Adjacent, |c, _| {
            set(
                c,
                "/text_config/sparse_attention_config/sparse_attention_freq",
                json!([0, 1]),
            );
        })
        .err()
        .expect("an indexer on a layer of full attention");
        assert!(
            e.contains(&format!("{l}.0.self_attn.index_"))
                && e.contains("layer 0 has no indexer (sparse_attention_freq is 0)"),
            "{e}"
        );
        let e = open_small(Planes::Adjacent, |_, stored| {
            stored.retain(|t| !t.0.contains("experts.3.w1."));
        })
        .err()
        .expect("a stack of three experts of four");
        assert!(
            e.contains("layer 1 w1") && e.contains("expert ids are not 0..4"),
            "{e}"
        );
    }

    /// Where the config says a layer attends every position, the folder holds no indexer for it,
    /// and the header's `attention.sparse_layers` says so: the M3 reader reads that layer as full
    /// attention over its dense block.
    #[test]
    fn a_layer_of_full_attention_has_no_indexer() {
        let mut dims = Dims::small();
        dims.sparse = vec![false, true];
        let dir = temp_dir("minimaxm3-full");
        write_folder(&dir, &dims, Planes::Adjacent, untouched);
        let opened = folder::open(&dir).unwrap_or_else(|e| panic!("{e}"));
        let hp = Hparams::read(&opened.header).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(hp.sparse_attn, [false, true]);
        assert_eq!(
            (hp.kind(0).attn, hp.kind(0).ffn),
            (Attn::Full, FfnKind::Dense)
        );
        assert!(opened.header.find("blk.0.indexer.q_proj.weight").is_none());
        assert!(opened.header.find("blk.1.indexer.q_proj.weight").is_some());
        classify(&opened.header, &hp).unwrap_or_else(|e| panic!("{e}"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The part source makes what the host cannot read in place: a norm's gain as F32 with 1
    /// added to the stored BF16 value, and an MXFP8 projection as blocks of the scale byte and 32
    /// values.
    #[test]
    fn the_part_source_makes_the_norm_gain_and_the_mxfp8_blocks() {
        let gain = [0.5f32, -0.25, 0.0, 2.0];
        let bf16: Vec<u8> = (0..64)
            .flat_map(|i| ((gain[i % 4].to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let norm = "language_model.model.layers.0.input_layernorm.weight";
        let dir = temp_dir("minimaxm3-make");
        write_folder(&dir, &Dims::small(), Planes::Adjacent, |_, stored| {
            stored.iter_mut().find(|t| t.0 == norm).unwrap().3 = bf16.clone();
        });
        let opened = folder::open(&dir).unwrap_or_else(|e| panic!("{e}"));
        let want = ["blk.0.attn_norm.weight", "blk.0.attn_q.weight"];
        let planned = opened
            .source
            .plan(|name, _, _| {
                Ok::<_, String>(if want.contains(&name) {
                    Take::Whole
                } else {
                    Take::Skip
                })
            })
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(planned.len(), 2);
        for p in &planned {
            let part = opened.source.read(p).unwrap_or_else(|e| panic!("{e}"));
            let bytes = part.spans().concat();
            match p.info.name.as_str() {
                "blk.0.attn_norm.weight" => {
                    assert_eq!((p.info.ty, p.info.dims.clone()), (GgmlType::F32, vec![64]));
                    let got: Vec<f32> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| f32::from_le_bytes(*b))
                        .collect();
                    let plus_one: Vec<f32> = (0..64).map(|i| gain[i % 4] + 1.0).collect();
                    assert_eq!(got, plus_one);
                }
                _ => {
                    assert_eq!(
                        (p.info.ty, p.info.dims.clone()),
                        (GgmlType::MXFP8, vec![64, 64])
                    );
                    let (blocks, _) = bytes.as_chunks::<33>();
                    assert_eq!(blocks.len(), 64 * 64 / 32);
                    assert!(
                        blocks
                            .iter()
                            .all(|b| b[0] == 127 && b[1..].iter().all(|&v| v == 0x38))
                    );
                }
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The map's mover adds 1 to a norm gain and refuses what is no F32 gain, or is not finite, by
    /// the tensor's name; a tensor with no step has no mover.
    #[test]
    fn the_mover_adds_one_to_a_gain_and_refuses_what_is_not_a_finite_f32_by_name() {
        let checkpoint =
            Checkpoint::read(&Dims::small().config()).unwrap_or_else(|e| panic!("{e}"));
        assert!(checkpoint.mover("g", &[]).is_none());
        let mover = checkpoint.mover("g", &[PlusOne]).unwrap();
        let f32s = |v: &[f32]| Normal {
            type_id: GgmlType::F32.as_u32(),
            dims: vec![v.len() as u64],
            bytes: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            scale: vec![],
        };
        let mut n = f32s(&[0.5, -1.0, 0.0]);
        mover(&mut n).unwrap();
        assert_eq!(n.bytes, f32s(&[1.5, 0.0, 1.0]).bytes);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let e = mover(&mut f32s(&[1.0, bad])).unwrap_err();
            assert!(e.contains("g: +1 of non-finite input"), "{bad}: {e}");
        }
        let mut n = f32s(&[1.0]);
        n.type_id = GgmlType::BF16.as_u32();
        let e = mover(&mut n).unwrap_err();
        assert!(e.contains("g: +1 needs F32"), "{e}");
    }

    /// The module names of the quantization config's class targets: a projection, a router gate, an
    /// expert's matrix and the head are linear, the embedding is an embedding, and a norm, a
    /// vision module or another layer name is none.
    #[test]
    fn module_classes_name_linear_projections_and_the_embedding() {
        use ModuleClass::{Embedding, Linear};
        for (module, class) in [
            ("language_model.model.embed_tokens", Some(Embedding)),
            ("language_model.lm_head", Some(Linear)),
            (
                "language_model.model.layers.3.self_attn.q_proj",
                Some(Linear),
            ),
            (
                "language_model.model.layers.3.self_attn.index_k_proj",
                Some(Linear),
            ),
            ("language_model.model.layers.0.mlp.down_proj", Some(Linear)),
            (
                "language_model.model.layers.3.block_sparse_moe.gate",
                Some(Linear),
            ),
            (
                "language_model.model.layers.3.block_sparse_moe.shared_experts.up_proj",
                Some(Linear),
            ),
            (
                "language_model.model.layers.3.block_sparse_moe.experts.127.w2",
                Some(Linear),
            ),
            (
                "language_model.model.layers.3.block_sparse_moe.experts.127.w4",
                None,
            ),
            (
                "language_model.model.layers.3.block_sparse_moe.experts.x.w1",
                None,
            ),
            ("language_model.model.layers.3.input_layernorm", None),
            ("language_model.model.layers.3.self_attn.q_norm", None),
            ("language_model.model.layers.x.self_attn.q_proj", None),
            (
                "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
                None,
            ),
            ("model.layers.3.self_attn.q_proj", None),
        ] {
            assert_eq!(module_class_of(module), class, "{module}");
        }
    }
}
