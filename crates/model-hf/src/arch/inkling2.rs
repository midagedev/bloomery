//! Inkling 2's MLX checkpoint folder (`inkling2_loop`: mlx-community's 4-bit convert of the
//! Mini) in the inkling2 names, layouts and keys the program reads — the same description the
//! GGUF twin of the model (unsloth's UD-Q4_K_XL) carries, so one reader takes both inputs: the
//! architecture word `inkling2` (the unsloth dialect: `block_count` the layer *visits*, the loop
//! read from `loop.core_*`, a core layer's controllers `loop_{attn,k,v,mlp}_{down,gain,forget}`
//! and the loop's message queries the model-level `loop_res_query` and `loop_kv_res_query`, two
//! rows a core layer).
//!
//! A checkpoint tensor maps to one header tensor the way its role reads it ([`Use`]), with the
//! moves the folder's layout asks for ([`Step`]): the conv taps arrive `[channels, kernel, 1]`
//! and are the file's rows once the unit dim is dropped; the windowless layers' *static* taps
//! arrive taps-major and move to the file's channel rows with each channel's taps reversed (the
//! file's order is oldest-first); the relative-bias bank arrives `[d_rel, extent]` and the file
//! holds it transposed, `extent` rows of `d_rel` values; the loop's message queries arrive as one
//! model-level table `[core layers, 4, hidden]` (a layer's rows its q, k, v and ffn queries) and
//! split into the two tensors the header carries — the attention-and-ffn rows and the k-and-v
//! rows, interleaved a core layer. The dense projections, the embedding and the head stay in
//! MLX's affine 4-bit blocks; the routed experts stay in the source's `nvfp4` (with a
//! `global_scale` an expert) and `mxfp8` modes as bloomery's `nvfp4_rows` and `mxfp8` rows, read
//! in place ([`Origin::Stack`]): no expert byte is repacked at a load. The vision tower's
//! tensors are not the text path's and are skipped by name.
//!
//! The header's keys are the config's: the visits, the per-visit head counts and window pattern
//! (a stored layer's repeats read its own), and the loop's bounds — the loop itself is config,
//! and the visits derive at load.

use std::collections::BTreeMap;

use gguf::{GgmlType, StackLayout, Value};
use hf::modelopt::{Normal, Plain, Use};
use hf::schema::{Fields, Rule::*, Type};
use model::arch::inkling2::names as ink;
use model::arch::names as common;
use serde_json::Value as Json;

use super::section::Section;
use super::trunk;
use crate::folder::{Error, FolderMap, Mapped, Mapping, Origin, Squeeze, VocabSpec, refuse};

const ARCHITECTURE: &str = "inkling2";

/// The checkpoint's prefix of the text model's tensors.
const MODEL: &str = "model.language_model.";

/// The output head, which a folder of an untied model holds.
const HEAD: &str = "lm_head.weight";

/// The loop's message-query table `model.language_model.message_input.query`: one row a (core
/// layer, query) pair, a layer's four rows its q, k, v and ffn queries, the order mlx-lm's
/// `sanitize` interleaves the checkpoint's two tables into.
const MESSAGE_INPUT: &str = "model.language_model.message_input.query";

/// `config.json`: the checkpoint's class and type, MLX's `quantization` block (which
/// `hf::quant_config` reads, with its verbatim copy), the text model's own object, and the vision
/// tower's config, whatever it holds (the tower's tensors are skipped by name). The draft's
/// config rides along unread: MLX's convert stores no draft tensors. `hf::vocab::Config` reads
/// `eos_token_id`.
const ROOT: &Fields = &[
    (
        "architectures",
        Need(r#"[["Inkling2LoopForConditionalGeneration"]]"#),
    ),
    ("model_type", Need(r#"["inkling2_loop"]"#)),
    ("eos_token_id", Read),
    (
        "mtp_config",
        Ignore(
            Type::Object,
            "the draft's config; the convert stores no draft tensors",
        ),
    ),
    ("quantization", Read),
    ("quantization_config", Read),
    ("text_config", Read),
    // The tower's config names nothing the text path reads.
    ("vision_config", Read),
];

/// `text_config`: the values the map reads into the header's keys and its tensors' geometry.
/// Every key the reader holds a fixed rule for is `Only` at the value that rule takes.
const TEXT: &Fields = &[
    // The context length, under the convert's own name.
    ("model_max_length", Read),
    (
        "torch_dtype",
        Ignore(
            Type::String,
            "the dtype HF loads in; each tensor keeps its stored dtype and format",
        ),
    ),
    ("hidden_size", Read),
    ("num_hidden_layers", Read),
    // hf::vocab::Config reads the vocabulary's size; the header carries it beside the unpadded
    // one.
    ("vocab_size", Read),
    // The windowless layers' head counts: the header carries one `attention.head_count` for
    // every visit, so the two kinds must agree.
    ("num_attention_heads", Read),
    ("swa_num_attention_heads", Read),
    ("num_key_value_heads", Read),
    ("swa_num_key_value_heads", Read),
    ("head_dim", Read),
    ("swa_head_dim", Read),
    ("sliding_window_size", Read),
    ("local_layer_ids", Read),
    ("d_rel", Read),
    ("rel_extent", Read),
    ("q_bias", Only("[false]")),
    ("o_bias", Only("[false]")),
    ("log_scaling_n_floor", Read),
    ("log_scaling_alpha", Read),
    ("rms_norm_eps", Read),
    // The value rows are normed gainlessly whatever the file's key says.
    ("use_value_norm_no_gamma", Only("[true]")),
    ("qk_l2_clip_threshold", Read),
    ("rel_l2_clip_threshold", Read),
    ("sconv_kernel_size", Read),
    ("unpadded_vocab_size", Read),
    ("final_logit_softcapping", Read),
    ("n_layers_per_attn_res_block", Read),
    ("attn_res_normalize_values", Only("[true]")),
    ("n_routed_experts", Read),
    ("num_experts_per_tok", Read),
    ("n_shared_experts", Only("[0]")),
    // The routed experts' width: the header's `expert_feed_forward_length`, and its
    // `feed_forward_length`, which no layer reads.
    ("intermediate_size", Read),
    // The `situ` activation's two betas.
    ("mlp_gate_softcap", Read),
    ("mlp_up_softcap", Read),
    ("route_scale", Read),
    // The router's selection bias, which every layer of this folder holds.
    ("use_gate_bias", Only("[true]")),
    ("loop_config", Read),
];

/// `loop_config`: the fixed loop's bounds.
const LOOP: &Fields = &[
    ("core_start", Read),
    ("core_end", Read),
    ("num_passes", Read),
    ("gate_rank", Read),
    // The research checkpoint's serve-time pass count over shared cache slots, which this
    // reader does not run.
    ("is_varloop", Only("[false]")),
];

/// The loop a looped folder carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Loop {
    core_start: usize,
    core_end: usize,
    passes: usize,
    gate_rank: usize,
}

impl Loop {
    /// The visits its core adds over the stored layers.
    fn extra(&self) -> usize {
        (self.passes - 1) * (self.core_end - self.core_start)
    }

    /// The stored layer of visit `v`: itself before the core's repeats, the core's layer inside
    /// them, itself less the repeats after (the reader's first-visits rule read forward).
    fn layer_of(&self, v: usize) -> usize {
        if v < self.core_end {
            v
        } else if v < self.core_end + self.extra() {
            self.core_start + (v - self.core_start) % (self.core_end - self.core_start)
        } else {
            v - self.extra()
        }
    }
}

/// A move the map makes of a normalized tensor, one of this architecture's own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// The conv taps' unit dim dropped: `[channels, kernel, 1]` stored, the file's
    /// `[kernel, channels]` rows, tap order unchanged ([`Squeeze`]).
    Squeeze,
    /// The relative-bias bank: `[d_rel, extent]` rows stored, the file's `extent` rows of
    /// `d_rel` values (a transpose).
    Bank,
    /// The windowless layers' static conv taps: the stored taps-major `[kernel, channels]` plane
    /// moved to the file's channel rows with each channel's taps reversed (the file's order is
    /// oldest-first, the stored one newest-first).
    StaticTaps,
    /// The model-level message-query table's rows, one a row of the made tensor: each entry the
    /// table's flat row `layer · 4 + query` the output row holds.
    Rows(Vec<u32>),
}

impl Step {
    /// The move `self` makes of the normalized tensor `t`, which the caller names; `t` is as it
    /// was when the move refuses.
    fn apply(&self, t: &mut Normal) -> Result<(), String> {
        match self {
            Step::Squeeze => Squeeze.apply(t),
            Step::Bank => transpose(t),
            Step::StaticTaps => {
                transpose(t)?;
                let Some(&taps) = t.dims.first() else {
                    return Err(format!("{:?} is not the static taps", t.dims));
                };
                let taps = taps as usize;
                let flat: Vec<[u8; 4]> = t.bytes.as_chunks::<4>().0.to_vec();
                for (c, channel) in flat.chunks(taps).enumerate() {
                    for (tap, bytes) in channel.iter().rev().enumerate() {
                        t.bytes[(c * taps + tap) * 4..][..4].copy_from_slice(bytes);
                    }
                }
                Ok(())
            }
            Step::Rows(rows) => {
                let Some((&width, &tables)) = t.dims.first().zip(t.dims.get(2)) else {
                    return Err(format!("{:?} is not the message-query table", t.dims));
                };
                let (row, held) = (width as usize * 4, tables as usize * 4);
                if rows.iter().any(|r| *r as usize >= held) {
                    return Err(format!("rows {rows:?} of a table of {held}"));
                }
                let bytes = rows
                    .iter()
                    .flat_map(|r| t.bytes[*r as usize * row..][..row].to_vec())
                    .collect();
                t.dims = vec![width, rows.len() as u64];
                t.bytes = bytes;
                Ok(())
            }
        }
    }
}

/// The 2-D tensor `t` transposed, its dims and bytes swapped.
fn transpose(t: &mut Normal) -> Result<(), String> {
    let Some((&cols, &rows)) = t.dims.first().zip(t.dims.get(1)) else {
        return Err(format!("{:?} is not 2-D", t.dims));
    };
    if t.dims.len() != 2 {
        return Err(format!("{:?} is not 2-D", t.dims));
    }
    let (cols, rows) = (cols as usize, rows as usize);
    let mut out = vec![0u8; t.bytes.len()];
    for (i, bytes) in t.bytes.as_chunks::<4>().0.iter().enumerate() {
        let at = ((i % cols) * rows + i / cols) * 4;
        out[at..at + 4].copy_from_slice(bytes);
    }
    t.dims = vec![rows as u64, cols as u64];
    t.bytes = out;
    Ok(())
}

/// The config values the map reads, checked, and the header keys they make.
#[derive(Debug)]
pub struct Checkpoint {
    /// The stored layers.
    layers: usize,
    /// Whether each stored layer is windowed, by `local_layer_ids`.
    windowed: Vec<bool>,
    /// The loop; `None`: one pass over every layer.
    loop_: Option<Loop>,
    hidden: u64,
    heads: u64,
    kv: u64,
    head_dim: u64,
    key_rows: u64,
    swa: u64,
    d_rel: u64,
    extent: u64,
    conv: u64,
    experts: u64,
    ff: u64,
    vocab: VocabSpec,
    /// The header's model keys by suffix, under the architecture's name.
    keys: Vec<(String, Value)>,
}

impl FolderMap for Checkpoint {
    type Step = Step;

    const MODEL_TYPES: &'static [&'static str] = &["inkling2_loop"];

    fn parse(config: &Json) -> Result<Self, Error> {
        let root = Section::<Error>::new(config, "config.json")?;
        root.check(&[ROOT])?;
        let text = Section::<Error>::new(root.field("text_config")?, "config.json.text_config")?;
        text.check(&[TEXT])?;
        const COUNTS: &[&str] = &[
            "model_max_length",
            "hidden_size",
            "num_hidden_layers",
            "vocab_size",
            "num_attention_heads",
            "swa_num_attention_heads",
            "num_key_value_heads",
            "swa_num_key_value_heads",
            "head_dim",
            "swa_head_dim",
            "sliding_window_size",
            "log_scaling_n_floor",
            "d_rel",
            "rel_extent",
            "sconv_kernel_size",
            "unpadded_vocab_size",
            "n_layers_per_attn_res_block",
            "n_routed_experts",
            "num_experts_per_tok",
            "intermediate_size",
        ];
        let mut counts = BTreeMap::new();
        for &field in COUNTS {
            counts.insert(field, text.count(field)?);
        }
        let n = |key: &str| counts[key] as usize;
        let wide = |key: &str| u64::from(counts[key]);
        for (key, swa_key) in [
            ("num_attention_heads", "swa_num_attention_heads"),
            ("num_key_value_heads", "swa_num_key_value_heads"),
            ("head_dim", "swa_head_dim"),
        ] {
            if counts[key] != counts[swa_key] {
                return Err(text.bad(
                    swa_key,
                    &format!(
                        "is {}, where {key} is {}: the header carries one head count",
                        counts[swa_key], counts[key]
                    ),
                ));
            }
        }
        if !n("num_attention_heads").is_multiple_of(n("num_key_value_heads")) {
            return Err(text.bad("num_attention_heads", "must group by num_key_value_heads"));
        }
        if n("unpadded_vocab_size") > n("vocab_size") {
            return Err(text.bad(
                "unpadded_vocab_size",
                &format!(
                    "is {} over the padded vocab_size {}",
                    n("unpadded_vocab_size"),
                    n("vocab_size")
                ),
            ));
        }
        if n("num_experts_per_tok") > n("n_routed_experts") {
            return Err(text.bad(
                "num_experts_per_tok",
                &format!("exceeds n_routed_experts {}", n("n_routed_experts")),
            ));
        }
        let layers = n("num_hidden_layers");
        let mut windowed = vec![false; layers];
        for (at, id) in text.array("local_layer_ids")?.iter().enumerate() {
            let id = id
                .as_u64()
                .and_then(|i| usize::try_from(i).ok())
                .filter(|&i| i < layers)
                .ok_or_else(|| {
                    text.bad(
                        "local_layer_ids",
                        &format!("holds {id} at {at}, not a layer of {layers}"),
                    )
                })?;
            if windowed[id] {
                return Err(text.bad("local_layer_ids", &format!("lists layer {id} twice")));
            }
            windowed[id] = true;
        }
        let loop_ = match text.value.get("loop_config") {
            Some(value) => {
                let section = Section::<Error>::new(value, "config.json.text_config.loop_config")?;
                section.check(&[LOOP])?;
                let loop_ = Loop {
                    core_start: section.count("core_start")? as usize,
                    core_end: section.count("core_end")? as usize,
                    passes: section.count("num_passes")? as usize,
                    gate_rank: section.count("gate_rank")? as usize,
                };
                if loop_.passes < 2 {
                    return Err(section.bad(
                        "num_passes",
                        &format!(
                            "is {}; the fixed loop runs at least two passes over its core",
                            loop_.passes
                        ),
                    ));
                }
                if loop_.core_start >= loop_.core_end || loop_.core_end > layers {
                    return Err(section.bad(
                        "core_end",
                        &format!(
                            "is {} with core_start {} of {} layers",
                            loop_.core_end, loop_.core_start, layers
                        ),
                    ));
                }
                let block = n("n_layers_per_attn_res_block");
                if !loop_.core_start.is_multiple_of(block)
                    || !(loop_.core_end - loop_.core_start).is_multiple_of(block)
                {
                    return Err(section.bad(
                        "core_start",
                        &format!(
                            "is {} with core_end {} for a block size {block}: the looped core \
                             must span whole blocks of the attention residual",
                            loop_.core_start, loop_.core_end
                        ),
                    ));
                }
                Some(loop_)
            }
            None => None,
        };
        let visits = layers + loop_.map_or(0, |loop_| loop_.extra());
        let pattern = |v: usize| match loop_ {
            Some(loop_) => windowed[loop_.layer_of(v)],
            None => windowed[v],
        };
        let u32_key = |key: &str, config: &str| (key.to_string(), Value::U32(counts[config]));
        let mut keys = vec![
            ("block_count".to_string(), Value::U32(visits as u32)),
            u32_key("context_length", "model_max_length"),
            u32_key("embedding_length", "hidden_size"),
            u32_key("vocab_size", "vocab_size"),
            u32_key("vocab_size_unpadded", "unpadded_vocab_size"),
            (
                "attention.head_count".to_string(),
                Value::Array(vec![
                    Value::I32(counts["num_attention_heads"] as i32);
                    visits
                ]),
            ),
            (
                "attention.head_count_kv".to_string(),
                Value::Array(vec![
                    Value::I32(counts["num_key_value_heads"] as i32);
                    visits
                ]),
            ),
            u32_key("attention.key_length", "head_dim"),
            u32_key("attention.value_length", "head_dim"),
            (
                "attention.layer_norm_rms_epsilon".to_string(),
                Value::F32(text.number("rms_norm_eps")?),
            ),
            u32_key("attention.sliding_window", "sliding_window_size"),
            (
                "attention.sliding_window_pattern".to_string(),
                Value::Array((0..visits).map(|v| Value::Bool(pattern(v))).collect()),
            ),
            u32_key("attention.relative_length", "d_rel"),
            u32_key("attention.relative_extent", "rel_extent"),
            (
                "attention.qk_clip".to_string(),
                Value::F32(text.number("qk_l2_clip_threshold")?),
            ),
            (
                "attention.relative_clip".to_string(),
                Value::F32(text.number("rel_l2_clip_threshold")?),
            ),
            u32_key("attention.log_scale_floor", "log_scaling_n_floor"),
            (
                "attention.log_scale_alpha".to_string(),
                Value::F32(text.number("log_scaling_alpha")?),
            ),
            u32_key("attn_res.block_size", "n_layers_per_attn_res_block"),
            u32_key("shortconv.l_cache", "sconv_kernel_size"),
            u32_key("expert_count", "n_routed_experts"),
            u32_key("expert_used_count", "num_experts_per_tok"),
            u32_key("expert_feed_forward_length", "intermediate_size"),
            // No layer is dense; the GGUF twin carries the width under this name, which no
            // reader takes.
            u32_key("feed_forward_length", "intermediate_size"),
            (
                "expert_weights_scale".to_string(),
                Value::F32(text.number("route_scale")?),
            ),
            (
                "final_logit_softcapping".to_string(),
                Value::F32(text.number("final_logit_softcapping")?),
            ),
            (
                "activation.situ_beta".to_string(),
                Value::F32(text.number("mlp_gate_softcap")?),
            ),
            (
                "activation.situ_linear_beta".to_string(),
                Value::F32(text.number("mlp_up_softcap")?),
            ),
        ];
        if let Some(loop_) = loop_ {
            keys.extend([
                (
                    "loop.core_start".to_string(),
                    Value::U32(loop_.core_start as u32),
                ),
                (
                    "loop.core_end".to_string(),
                    Value::U32(loop_.core_end as u32),
                ),
                ("num_loops".to_string(), Value::U32(loop_.passes as u32)),
                (
                    "loop.gate_rank".to_string(),
                    Value::U32(loop_.gate_rank as u32),
                ),
            ]);
        }
        Ok(Self {
            layers,
            windowed,
            loop_,
            hidden: wide("hidden_size"),
            heads: wide("num_attention_heads"),
            kv: wide("num_key_value_heads"),
            head_dim: wide("head_dim"),
            key_rows: wide("num_key_value_heads") * wide("head_dim"),
            swa: wide("sliding_window_size"),
            d_rel: wide("d_rel"),
            extent: wide("rel_extent"),
            conv: wide("sconv_kernel_size"),
            experts: wide("n_routed_experts"),
            ff: wide("intermediate_size"),
            vocab: hf::vocab::Config::parse(config)?.with_pre("gpt-4o")?,
            keys,
        })
    }

    fn module_class(&self, module: &str) -> Option<hf::quant_config::ModuleClass> {
        super::module_class(module)
    }

    fn map<'a>(&self, tensors: &[(&'a str, &'a [usize])]) -> Result<Mapping<Step>, Error> {
        let mut out: BTreeMap<String, Mapped<Step>> = BTreeMap::new();
        let mut skipped = BTreeMap::new();
        let put = |to: String,
                   m: Mapped<Step>,
                   out: &mut BTreeMap<String, Mapped<Step>>|
         -> Result<(), Error> {
            let source = m.sources[0].clone();
            if out.insert(to, m).is_some() {
                return Err(refuse(&source, "maps to a name another tensor has"));
            }
            Ok(())
        };
        for &(name, shape) in tensors {
            if !trunk::sized(shape) {
                return Err(refuse(
                    name,
                    format!("shape {shape:?} is empty, zero, or overflows usize"),
                ));
            }
            if name.starts_with("model.vision_tower.") {
                skipped.insert(
                    name.to_string(),
                    "the vision tower, which the text path does not read".to_string(),
                );
                continue;
            }
            let dims = || shape.iter().rev().map(|&d| d as u64).collect::<Vec<_>>();
            // One made tensor of this checkpoint tensor, its steps and the header's dims its own.
            let one = |steps: Vec<Step>, dims: Vec<u64>| Mapped {
                sources: vec![name.to_string()],
                role: Use::Matrix(Plain::F32),
                steps,
                dims,
                origin: Origin::Made,
            };
            let linear = |dims: Vec<u64>| Mapped {
                sources: vec![name.to_string()],
                role: Use::Matrix(Plain::Q8_0),
                steps: vec![],
                dims,
                origin: Origin::Made,
            };
            let f32_file = |dims: Vec<u64>| Mapped {
                sources: vec![name.to_string()],
                role: Use::Matrix(Plain::F32),
                steps: vec![],
                dims,
                origin: Origin::File(GgmlType::F32),
            };
            let stack = |dims: Vec<u64>| Mapped {
                sources: vec![name.to_string()],
                role: Use::Matrix(Plain::Q8_0),
                steps: vec![],
                dims,
                origin: Origin::Stack {
                    ty: GgmlType::NVFP4_ROWS,
                    layout: StackLayout::Planes,
                },
            };
            if name == MESSAGE_INPUT {
                continue;
            }
            if name == HEAD {
                put(common::output(), linear(dims()), &mut out)?;
                continue;
            }
            let short = name
                .strip_prefix(MODEL)
                .ok_or_else(|| refuse(name, "not an Inkling 2 tensor of the text model"))?;
            match short {
                "embed_tokens.weight" => {
                    put(common::token_embd(), linear(dims()), &mut out)?;
                    continue;
                }
                "embed_norm.weight" => {
                    put(
                        ink::TOKEN_EMBD_NORM.to_string(),
                        one(vec![], vec![self.hidden]),
                        &mut out,
                    )?;
                    continue;
                }
                "norm.weight" => {
                    put(common::output_norm(), one(vec![], dims()), &mut out)?;
                    continue;
                }
                "final_input.query" => {
                    put(
                        ink::OUTPUT_RES_QUERY.to_string(),
                        f32_file(vec![self.hidden]),
                        &mut out,
                    )?;
                    continue;
                }
                _ => {}
            }
            let Some((l, rest)) = short
                .strip_prefix("layers.")
                .and_then(|r| r.split_once('.'))
                .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
            else {
                if let Some(l) = short
                    .strip_prefix("layers.")
                    .and_then(|r| r.split_once('.'))
                    .map(|(l, _)| l)
                {
                    return Err(refuse(name, format!("layer {l:?} of {}", self.layers)));
                }
                return Err(refuse(name, "not an Inkling 2 tensor of the text model"));
            };
            if l >= self.layers {
                return Err(refuse(name, format!("layer {l} of {}", self.layers)));
            }
            let windowed = self.windowed[l];
            let stem = |to: &str| common::blk(l, to);
            let conv_taps = || one(vec![Step::Squeeze], vec![self.conv, self.key_rows]);
            match rest {
                "attention_input.query" => {
                    if l == 0 {
                        return Err(refuse(
                            name,
                            "layer 0 banks nothing to mix, and holds no attention mix query",
                        ));
                    }
                    put(stem("attn_res_query.weight"), f32_file(dims()), &mut out)?;
                }
                "mlp_input.query" => {
                    put(stem("ffn_res_query.weight"), f32_file(dims()), &mut out)?;
                }
                "post_attention_layernorm.weight" => {
                    put(stem("ffn_norm.weight"), one(vec![], dims()), &mut out)?;
                }
                "post_attention_convolution.conv1d.weight" => put(
                    stem("attn_post_conv.weight"),
                    one(vec![Step::Squeeze], vec![self.conv, self.hidden]),
                    &mut out,
                )?,
                "post_mlp_convolution.conv1d.weight" => put(
                    stem("ffn_post_conv.weight"),
                    one(vec![Step::Squeeze], vec![self.conv, self.hidden]),
                    &mut out,
                )?,
                "input_layernorm.weight" if windowed => {
                    put(stem("attn_norm.weight"), one(vec![], dims()), &mut out)?;
                }
                "q_input_layernorm.weight" if !windowed => {
                    let to = ink::inp(l, ink::Inp::NormQ);
                    put(to.clone(), one(vec![], dims()), &mut out)?;
                }
                "k_input_layernorm.weight" if !windowed => {
                    let to = ink::inp(l, ink::Inp::NormK);
                    put(to.clone(), one(vec![], dims()), &mut out)?;
                }
                "v_input_layernorm.weight" if !windowed => {
                    let to = ink::inp(l, ink::Inp::NormV);
                    put(to.clone(), one(vec![], dims()), &mut out)?;
                }
                "self_attn.q_proj.weight" => {
                    put(stem("attn_q.weight"), linear(dims()), &mut out)?;
                }
                "self_attn.k_proj.weight" => {
                    put(stem("attn_k.weight"), linear(dims()), &mut out)?;
                }
                "self_attn.v_proj.weight" => {
                    put(stem("attn_v.weight"), linear(dims()), &mut out)?;
                }
                "self_attn.o_proj.weight" => {
                    put(stem("attn_output.weight"), linear(dims()), &mut out)?;
                }
                "self_attn.gate_proj.weight" => {
                    put(stem("attn_gate.weight"), linear(dims()), &mut out)?;
                }
                "self_attn.sink_proj.weight" => {
                    put(stem("attn_sink_proj.weight"), linear(dims()), &mut out)?;
                }
                "self_attn.r_proj.weight" => {
                    put(stem("attn_rel.weight"), linear(dims()), &mut out)?;
                }
                "self_attn.rel_logits_proj.proj" => put(
                    stem("attn_rel_bias.weight"),
                    one(
                        vec![Step::Bank],
                        // A windowed layer's bias bank is its window wide, a windowless one's the
                        // config's extent, the widths the reader's keys give them.
                        vec![self.d_rel, if windowed { self.swa } else { self.extent }],
                    ),
                    &mut out,
                )?,
                "self_attn.q_l2_clip.scale" => {
                    put(stem("attn_q_clip.weight"), one(vec![], dims()), &mut out)?;
                }
                "self_attn.rel_l2_clip.scale" => {
                    put(stem("attn_rel_clip.weight"), one(vec![], dims()), &mut out)?;
                }
                "self_attn.k_sconv.conv1d.weight" if windowed => {
                    put(stem("attn_k_conv.weight"), conv_taps(), &mut out)?;
                }
                "self_attn.v_sconv.conv1d.weight" if windowed => {
                    put(stem("attn_v_conv.weight"), conv_taps(), &mut out)?;
                }
                "self_attn.k_sconv.static_weight" if !windowed => put(
                    stem("attn_k_conv.weight"),
                    one(vec![Step::StaticTaps], vec![self.conv, self.key_rows]),
                    &mut out,
                )?,
                "self_attn.v_sconv.static_weight" if !windowed => put(
                    stem("attn_v_conv.weight"),
                    one(vec![Step::StaticTaps], vec![self.conv, self.key_rows]),
                    &mut out,
                )?,
                "self_attn.k_sconv.weight_proj.weight" if !windowed => {
                    put(ink::inp(l, ink::Inp::ConvK), linear(dims()), &mut out)?;
                }
                "self_attn.v_sconv.weight_proj.weight" if !windowed => {
                    put(ink::inp(l, ink::Inp::ConvV), linear(dims()), &mut out)?;
                }
                "mlp.gate.weight" => {
                    put(stem("ffn_gate_inp.weight"), f32_file(dims()), &mut out)?;
                }
                "mlp.gate.e_score_correction_bias" => {
                    put(common::exp_probs_b(l), f32_file(dims()), &mut out)?;
                }
                "mlp.experts.gate_up_proj.weight" => {
                    put(stem("ffn_gate_up_exps.weight"), stack(dims()), &mut out)?;
                }
                "mlp.experts.down_proj.weight" => {
                    put(common::ffn_down_exps(l), stack(dims()), &mut out)?;
                }
                other if other.contains("_message_gate.") => {
                    if !self.in_core(l) {
                        return Err(refuse(
                            name,
                            format!(
                                "layer {l} is outside the loop's core, which holds no message \
                                 controller"
                            ),
                        ));
                    }
                    let to = self
                        .controller(l, other)
                        .ok_or_else(|| refuse(name, "not an Inkling 2 tensor"))?;
                    let dims = if other.ends_with("down_proj.weight") {
                        vec![2 * self.hidden, self.rank()]
                    } else {
                        vec![self.rank(), self.hidden]
                    };
                    if other.ends_with("down_proj.weight") {
                        put(to.clone(), one(vec![], dims), &mut out)?;
                    } else {
                        put(to.clone(), f32_file(dims), &mut out)?;
                    }
                }
                other => {
                    return Err(refuse(
                        name,
                        format!("{other:?} is not an Inkling 2 tensor"),
                    ));
                }
            }
        }
        let message = tensors
            .iter()
            .find(|&&(n, _)| n == MESSAGE_INPUT)
            .map(|&(n, _)| n);
        if let Some(loop_) = self.loop_
            && let Some(table) = message
        {
            let core = (loop_.core_end - loop_.core_start) as u32;
            // A core layer's rows follow one another, the attention-and-ffn table's a layer's q
            // and ffn rows, the k-and-v table's its k and v rows.
            let queries_of = |a: u32, b: u32| {
                (0..core)
                    .flat_map(move |i| [i * 4 + a, i * 4 + b])
                    .collect::<Vec<_>>()
            };
            for (to, queries) in [
                (ink::LOOP_RES_QUERY.to_string(), queries_of(0, 3)),
                (ink::LOOP_KV_RES_QUERY.to_string(), queries_of(1, 2)),
            ] {
                put(
                    to.clone(),
                    Mapped {
                        sources: vec![table.to_string()],
                        role: Use::Matrix(Plain::F32),
                        steps: vec![Step::Rows(queries)],
                        dims: vec![self.hidden, 2 * u64::from(core)],
                        origin: Origin::Made,
                    },
                    &mut out,
                )?;
            }
        }
        self.complete(&out, message)?;
        Ok(Mapping {
            tensors: out,
            skipped,
        })
    }

    fn mover(&self, name: &str, steps: &[Step]) -> Option<crate::part::Step> {
        (!steps.is_empty()).then(|| {
            let (name, steps) = (name.to_string(), steps.to_vec());
            Box::new(move |t: &mut Normal| {
                steps
                    .iter()
                    .try_for_each(|s| s.apply(t).map_err(|why| format!("{name}: {why}")))
            }) as crate::part::Step
        })
    }

    fn architecture(&self) -> &'static str {
        ARCHITECTURE
    }

    fn header(
        &self,
        _planned: &[(&str, &[u64])],
        _folder: &safetensors::Folder,
    ) -> Result<Vec<(String, Value)>, Error> {
        Ok(trunk::header(ARCHITECTURE, &self.keys))
    }

    fn vocab(&self) -> &VocabSpec {
        &self.vocab
    }
}

impl Checkpoint {
    /// Whether layer `l` is inside the loop's core.
    fn in_core(&self, l: usize) -> bool {
        self.loop_
            .is_some_and(|loop_| (loop_.core_start..loop_.core_end).contains(&l))
    }

    /// The loop's gate rank, zero where the file carries no loop.
    fn rank(&self) -> u64 {
        self.loop_.map_or(0, |loop_| loop_.gate_rank as u64)
    }

    /// The header tensor of one message-controller tensor `rest` (the name past `layers.N.`):
    /// the module's own stem for its projection, [`None`] for another name. `down` is the joined
    /// `[rms(cur); rms(msg)]` read-down, `message_gate` the gain and `forget_gate` the forget.
    fn controller(&self, l: usize, rest: &str) -> Option<String> {
        let (module, projection) = rest.split_once('.')?;
        let module = match module {
            "q_message_gate" => ink::Ctrl::Attn,
            "k_message_gate" => ink::Ctrl::K,
            "v_message_gate" => ink::Ctrl::V,
            "mlp_message_gate" => ink::Ctrl::Ffn,
            _ => return None,
        };
        let part = match projection {
            "down_proj.weight" => ink::CtrlPart::Down,
            "message_gate_proj.weight" => ink::CtrlPart::Gain,
            "forget_gate_proj.weight" => ink::CtrlPart::Forget,
            _ => return None,
        };
        Some(ink::loop_ctrl_pr(l, module, part))
    }

    /// Every tensor the header holds is mapped, at the dims the config gives: the model level,
    /// then each stored layer's tensors by its kind. `message` names the model-level
    /// message-query table, which a looped folder holds and a loopless one lacks.
    fn complete(
        &self,
        mapped: &BTreeMap<String, Mapped<Step>>,
        message: Option<&str>,
    ) -> Result<(), Error> {
        let (e, d) = (self.hidden, self.head_dim);
        let (heads, kv) = (self.heads, self.kv);
        let (rel, taps) = (self.d_rel, self.conv);
        let vocab = u64::from(self.vocab.size());
        let mut want: Vec<(String, Vec<u64>)> = vec![
            (common::token_embd(), vec![e, vocab]),
            (ink::TOKEN_EMBD_NORM.to_string(), vec![e]),
            (common::output(), vec![e, vocab]),
            (common::output_norm(), vec![e]),
            (ink::OUTPUT_RES_QUERY.to_string(), vec![e]),
        ];
        if let Some(loop_) = self.loop_ {
            let queries = vec![e, 2 * (loop_.core_end - loop_.core_start) as u64];
            want.push((ink::LOOP_RES_QUERY.to_string(), queries.clone()));
            want.push((ink::LOOP_KV_RES_QUERY.to_string(), queries));
        }
        for l in 0..self.layers {
            let windowed = self.windowed[l];
            let stem = |to: &str| common::blk(l, to);
            want.extend([
                (stem("attn_q.weight"), vec![e, d * heads]),
                (stem("attn_k.weight"), vec![e, d * kv]),
                (stem("attn_v.weight"), vec![e, d * kv]),
                (stem("attn_output.weight"), vec![d * heads, e]),
                (stem("attn_gate.weight"), vec![e, d * heads]),
                (stem("attn_q_clip.weight"), vec![d, heads]),
                (stem("attn_sink_proj.weight"), vec![e, heads]),
                (stem("attn_rel.weight"), vec![e, rel * heads]),
                (
                    stem("attn_rel_bias.weight"),
                    vec![rel, if windowed { self.swa } else { self.extent }],
                ),
                (stem("attn_rel_clip.weight"), vec![rel, heads]),
                (stem("attn_k_conv.weight"), vec![taps, d * kv]),
                (stem("attn_v_conv.weight"), vec![taps, d * kv]),
                (stem("attn_post_conv.weight"), vec![taps, e]),
                (stem("ffn_post_conv.weight"), vec![taps, e]),
            ]);
            if windowed {
                want.push((stem("attn_norm.weight"), vec![e]));
            } else {
                want.extend([
                    (ink::inp(l, ink::Inp::NormQ), vec![e]),
                    (ink::inp(l, ink::Inp::NormK), vec![e]),
                    (ink::inp(l, ink::Inp::NormV), vec![e]),
                    (ink::inp(l, ink::Inp::ConvK), vec![e, taps * kv]),
                    (ink::inp(l, ink::Inp::ConvV), vec![e, taps * kv]),
                ]);
            }
            if l > 0 {
                let queries = u64::from(!windowed) * 2 + 1;
                want.push((stem("attn_res_query.weight"), vec![e, queries]));
            }
            let (n, ff) = (self.experts, self.ff);
            want.extend([
                (stem("ffn_res_query.weight"), vec![e]),
                (stem("ffn_norm.weight"), vec![e]),
                (stem("ffn_gate_inp.weight"), vec![e, n]),
                (common::exp_probs_b(l), vec![n]),
                (stem("ffn_gate_up_exps.weight"), vec![e, 2 * ff, n]),
                (common::ffn_down_exps(l), vec![ff, e, n]),
            ]);
            if self.in_core(l) {
                for module in [ink::Ctrl::Attn, ink::Ctrl::K, ink::Ctrl::V, ink::Ctrl::Ffn] {
                    for part in [
                        ink::CtrlPart::Down,
                        ink::CtrlPart::Gain,
                        ink::CtrlPart::Forget,
                    ] {
                        let dims = if part == ink::CtrlPart::Down {
                            vec![2 * e, self.rank()]
                        } else {
                            vec![self.rank(), e]
                        };
                        want.push((ink::loop_ctrl_pr(l, module, part), dims));
                    }
                }
            }
        }
        if self.loop_.is_some() != message.is_some() {
            return Err(refuse(
                MESSAGE_INPUT,
                if self.loop_.is_some() {
                    "is missing from the folder, and the loop's message queries are its tensors"
                } else {
                    "is in the folder, and a loopless model carries no message query"
                },
            ));
        }
        let pad = |mut d: Vec<u64>| {
            while d.last() == Some(&1) {
                d.pop();
            }
            d
        };
        for (name, dims) in want {
            let m = mapped
                .get(&name)
                .ok_or_else(|| refuse(&name, "is missing from the folder"))?;
            if pad(m.dims.clone()) != pad(dims.clone()) {
                return Err(refuse(
                    &m.sources[0],
                    format!("holds dims {:?} where the config gives {dims:?}", m.dims),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use gguf::GgmlType;
    use hf::quant_config::Config;
    use safetensors::Dtype;
    use serde_json::json;

    use super::*;
    use crate::folder::{self, Origin};
    use crate::test_support::{Entries, entries, json, temp_dir, write_folder};

    /// The real folder's config, and its four shards' tensor headers, as the fixtures hold them.
    const CONFIG: &str = "quant-configs/mlx-community__Inkling-2-Mini-4bit__config.json.gz";

    fn shard(name: &str) -> String {
        format!("checkpoint-headers/inkling-2-mini-mlx4-model-{name}-of-00004.header.json.gz")
    }

    /// The real folder's 1,559 tensors, the four shards merged.
    fn real_entries() -> Entries {
        let mut out = Entries::new();
        for i in 1..=4 {
            let mut header = json(&shard(&format!("{i:05}")));
            header.as_object_mut().unwrap().remove("__metadata__");
            out.extend(entries(&header));
        }
        out
    }

    /// The plan of the real folder's tensors, in place as a load reads them.
    fn planning(config: &Json, entries: &Entries) -> folder::Planning<Step> {
        let checkpoint = Checkpoint::parse(config).unwrap();
        folder::plan_tensors(
            &checkpoint,
            &Config::parse(config, None).unwrap(),
            entries,
            false,
            folder::Keep::InPlace,
        )
        .unwrap()
    }

    /// The real folder's headers plan every text tensor and skip the vision tower: 734 tensors of
    /// the header (the GGUF twin's own count), the dense projections and the stacked experts read
    /// in place, the loop's two message-query tensors made of the one table, and every byte a
    /// planned tensor holds is one the folder stores (the vision tower's 636,512,256 bytes apart).
    /// The plan refuses nothing, and the map leaves no folder tensor unread.
    #[test]
    fn the_real_headers_plan_every_text_tensor() {
        let config = json(CONFIG);
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let entries = real_entries();
        assert_eq!(entries.len(), 1559);
        let planning = planning(&config, &entries);
        assert!(
            planning.plan.refused.is_empty(),
            "{:?}",
            planning.plan.refused
        );
        let skipped: Vec<&str> = planning.skipped.keys().map(String::as_str).collect();
        assert_eq!(skipped.len(), 422);
        assert!(skipped.iter().all(|n| n.starts_with("model.vision_tower.")));
        let stored: usize = entries
            .iter()
            .filter(|(n, _, _)| !n.starts_with("model.vision_tower."))
            .map(|(_, d, s)| s.iter().product::<usize>() * d.size())
            .sum();
        let vision: usize = entries
            .iter()
            .filter(|(n, _, _)| n.starts_with("model.vision_tower."))
            .map(|(_, d, s)| s.iter().product::<usize>() * d.size())
            .sum();
        assert_eq!(vision, 636_512_256);
        let held: usize = planning
            .planned
            .iter()
            .map(|p| p.decl.nbytes as usize)
            .sum();
        // [derived] Every stored text byte is a planned byte: the F32 rows are kept where they
        // lie, the affine blocks interleave at the same total, the experts' declared rows are the
        // bytes their code and scale planes hold, the message table is read over its two tensors,
        // and the BF16 vectors planned as F32 (the norm gains, the clips, the bias bank, the conv
        // taps and the controllers' down projections) widen by 30,788,768 B over the folder's
        // 16,671,610,880 stored text bytes, and the forty `.scale` rows beside the NVFP4 stacks
        // hold their own 20,480.
        assert_eq!(stored, 16_671_610_880);
        assert_eq!(held, 16_702_419_968);
        assert_eq!(
            planning
                .planned
                .iter()
                .map(|p| p.decl.name.as_str())
                .filter(|n| !n.ends_with(".scale"))
                .count(),
            694
        );
        assert_eq!(planning.planned.len(), 734);
        let by_name = |n: &str| {
            planning
                .planned
                .iter()
                .find(|p| p.decl.name == n)
                .unwrap_or_else(|| panic!("{n} is not planned"))
        };
        // The routed experts stay in their stored planes, the coda's MXFP8 and the stem's
        // NVFP4_ROWS alike, with a `.scale` an expert beside the NVFP4 ones.
        for (name, ty) in [
            ("blk.0.ffn_gate_up_exps.weight", GgmlType::NVFP4_ROWS),
            ("blk.0.ffn_down_exps.weight", GgmlType::NVFP4_ROWS),
            ("blk.19.ffn_down_exps.weight", GgmlType::NVFP4_ROWS),
            ("blk.20.ffn_gate_up_exps.weight", GgmlType::MXFP8),
            ("blk.23.ffn_down_exps.weight", GgmlType::MXFP8),
        ] {
            assert_eq!(by_name(name).decl.type_id, ty.as_u32(), "{name}");
        }
        assert_eq!(by_name("blk.19.ffn_down_exps.scale").decl.dims, [128]);
        assert!(
            planning
                .planned
                .iter()
                .all(|p| p.decl.name != "blk.20.ffn_gate_up_exps.scale"),
            "MXFP8 reads no scale"
        );
        // The moved and made tensors carry their steps; the plain F32 rows are kept where they
        // lie.
        let steps_of = |n: &str| match &by_name(n).make {
            folder::Make::Tensor(m) => m.steps.clone(),
            _ => panic!("{n} is not a tensor"),
        };
        assert_eq!(steps_of("blk.0.attn_k_conv.weight"), vec![Step::Squeeze]);
        assert_eq!(steps_of("blk.5.attn_k_conv.weight"), vec![Step::StaticTaps]);
        assert_eq!(steps_of("blk.0.attn_rel_bias.weight"), vec![Step::Bank]);
        assert_eq!(
            steps_of("loop_res_query.weight"),
            vec![Step::Rows(
                [
                    0, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28, 31, 32, 35, 36, 39, 40,
                    43, 44, 47
                ]
                .to_vec()
            )]
        );
        assert_eq!(
            steps_of("loop_kv_res_query.weight"),
            vec![Step::Rows(
                [
                    1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30, 33, 34, 37, 38, 41,
                    42, 45, 46
                ]
                .to_vec()
            )]
        );
        assert_eq!(steps_of("blk.0.ffn_gate_inp.weight"), vec![]);
        for (name, origin) in [
            ("blk.0.ffn_gate_inp.weight", Origin::File(GgmlType::F32)),
            ("blk.6.attn_res_query.weight", Origin::File(GgmlType::F32)),
            ("blk.6.loop_attn_gain.weight", Origin::File(GgmlType::F32)),
            ("blk.6.loop_attn_down.weight", Origin::Made),
            ("token_embd_norm.weight", Origin::Made),
            ("output.weight", Origin::Made),
        ] {
            let folder::Make::Tensor(m) = &by_name(name).make else {
                panic!("{name} is not a tensor");
            };
            assert_eq!(m.origin, origin, "{name}");
        }
        // The header's keys are the GGUF twin's: the visits, one head count and pattern value a
        // visit, and the loop's bounds.
        let key = |suffix: &str| {
            checkpoint
                .keys
                .iter()
                .find(|(k, _)| k == suffix)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| panic!("no key {suffix}"))
        };
        assert_eq!(key("block_count"), Value::U32(48));
        assert_eq!(key("loop.core_start"), Value::U32(6));
        assert_eq!(key("loop.core_end"), Value::U32(18));
        assert_eq!(key("num_loops"), Value::U32(3));
        assert_eq!(key("loop.gate_rank"), Value::U32(64));
        assert_eq!(
            key("attention.head_count"),
            Value::Array(vec![Value::I32(24); 48])
        );
        let Value::Array(pattern) = key("attention.sliding_window_pattern") else {
            panic!("the pattern is an array");
        };
        assert_eq!(pattern.len(), 48);
        let windowless: Vec<usize> = (0..48)
            .filter(|&v| pattern[v] == Value::Bool(false))
            .collect();
        assert_eq!(windowless, [5, 11, 17, 23, 29, 35, 41, 47]);
        assert_eq!(key("attention.relative_extent"), Value::U32(1024));
        assert_eq!(key("expert_weights_scale"), Value::F32(8.0));
        assert_eq!(key("activation.situ_beta"), Value::F32(4.0));
        assert_eq!(key("activation.situ_linear_beta"), Value::F32(25.0));
        assert_eq!(key("vocab_size_unpadded"), Value::U32(200_058));
    }

    /// The config refuses, by key, what it does not read: an unknown key at either level, a
    /// value a fixed rule does not take, the two head-count spellings apart, a value the header's
    /// one key cannot hold, a loop that is not whole blocks inside the trunk, a layer id out of
    /// range, and a vocabulary or routing count that does not fit.
    #[test]
    fn the_config_refuses_what_it_does_not_read_by_name() {
        let base = json(CONFIG);
        assert!(Checkpoint::parse(&base).is_ok());
        let refuse = |edit: &dyn Fn(&mut Json)| {
            let mut config = base.clone();
            edit(&mut config);
            Checkpoint::parse(&config).unwrap_err().to_string()
        };
        let text = |key: &'static str, value: Json| {
            move |c: &mut Json| c["text_config"][key] = value.clone()
        };
        for (key, value) in [
            ("zzz", json!(1)),
            ("tie_word_embeddings", json!(true)),
            ("rope_theta", json!(1e6)),
        ] {
            let e = refuse(&move |c: &mut Json| {
                c.as_object_mut()
                    .unwrap()
                    .insert(key.to_string(), value.clone());
            });
            assert!(e.contains(key), "{key}: {e}");
        }
        for (key, value) in [
            ("/text_config/q_bias", json!(true)),
            ("/text_config/n_shared_experts", json!(2)),
            ("/text_config/use_value_norm_no_gamma", json!(false)),
            ("/text_config/attn_res_normalize_values", json!(false)),
            ("/text_config/use_gate_bias", json!(false)),
        ] {
            let e = refuse(&move |c: &mut Json| {
                *c.pointer_mut(key).unwrap() = value.clone();
            });
            assert!(
                e.contains(key.split('/').next_back().unwrap()),
                "{key}: {e}"
            );
        }
        let e = refuse(&text("is_varloop", json!(true)));
        assert!(e.contains("is_varloop"), "{e}");
        let e = refuse(&text("swa_head_dim", json!(64)));
        assert!(
            e.contains("swa_head_dim is 64, where head_dim is 128"),
            "{e}"
        );
        let e = refuse(&text("swa_num_key_value_heads", json!(4)));
        assert!(
            e.contains("swa_num_key_value_heads is 4, where num_key_value_heads is 8"),
            "{e}"
        );
        let e = refuse(&|c: &mut Json| {
            c["text_config"]["num_key_value_heads"] = json!(48);
            c["text_config"]["swa_num_key_value_heads"] = json!(48);
        });
        assert!(e.contains("must group by num_key_value_heads"), "{e}");
        let e = refuse(&text("unpadded_vocab_size", json!(201_025)));
        assert!(e.contains("over the padded vocab_size"), "{e}");
        let e = refuse(&text("num_experts_per_tok", json!(129)));
        assert!(e.contains("exceeds n_routed_experts"), "{e}");
        let e = refuse(&text("local_layer_ids", json!([0, 24])));
        assert!(e.contains("not a layer of 24"), "{e}");
        let e = refuse(&text("local_layer_ids", json!([0, 0])));
        assert!(e.contains("lists layer 0 twice"), "{e}");
        let e = refuse(&text(
            "loop_config",
            json!({"core_start": 6, "core_end": 18, "num_passes": 1, "gate_rank": 64, "is_varloop": false}),
        ));
        assert!(e.contains("num_passes"), "{e}");
        let e = refuse(&text(
            "loop_config",
            json!({"core_start": 6, "core_end": 25, "num_passes": 3, "gate_rank": 64, "is_varloop": false}),
        ));
        assert!(e.contains("core_end"), "{e}");
        let e = refuse(&text(
            "loop_config",
            json!({"core_start": 7, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": false}),
        ));
        assert!(e.contains("whole blocks"), "{e}");
        // A loopless folder is the Small's shape, which this map reads too; only its message
        // table must then be gone (the map's own refusal).
        let mut loopless = base.clone();
        loopless["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("loop_config");
        Checkpoint::parse(&loopless).expect("a loopless config reads");
    }

    /// The map refuses, by name, a tensor no folder of MLX's holds, a message controller outside
    /// the loop's core, layer 0's attention mix query, a tensor of another model's prefix, a
    /// layer past the config's, a tensor the header needs that the folder lacks, and a tensor of
    /// other dims than the config gives.
    #[test]
    fn the_map_refuses_what_it_does_not_read_by_name() {
        let config = json(CONFIG);
        let base = real_entries();
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let quant = Config::parse(&config, None).unwrap();
        let refusal = |edit: &dyn Fn(&mut Entries)| {
            let mut e = base.clone();
            edit(&mut e);
            folder::plan_tensors(&checkpoint, &quant, &e, false, folder::Keep::InPlace)
                .err()
                .expect("refused")
                .to_string()
        };
        let vector = |name: &str| (name.to_string(), Dtype::Bf16, vec![2304]);
        let l = "model.language_model.layers";
        let e = refusal(&|e| e.push(vector(&format!("{l}.0.stray.weight"))));
        assert!(
            e.contains("stray") && e.contains("not an Inkling 2 tensor"),
            "{e}"
        );
        let e = refusal(&|e| e.push(vector(&format!("{l}.1.q_message_gate.down_proj.weight"))));
        assert!(e.contains("outside the loop's core"), "{e}");
        let e = refusal(&|e| e.push(vector(&format!("{l}.0.attention_input.query"))));
        assert!(e.contains("layer 0 banks nothing"), "{e}");
        let e = refusal(&|e| e.push(vector("model.llm.attn.wq_du.weight")));
        assert!(
            e.contains("not an Inkling 2 tensor of the text model"),
            "{e}"
        );
        let e = refusal(&|e| e.push(vector(&format!("{l}.24.mlp_input.query"))));
        assert!(e.contains("layer 24 of 24"), "{e}");
        let e = refusal(&|e| {
            let at = e
                .iter()
                .position(|(n, _, _)| n == &format!("{l}.2.self_attn.q_l2_clip.scale"))
                .unwrap();
            e.remove(at);
        });
        assert!(
            e.contains("blk.2.attn_q_clip.weight") && e.contains("is missing from the folder"),
            "{e}"
        );
        let e = refusal(&|e| {
            let at = e
                .iter()
                .position(|(n, _, _)| n == &format!("{l}.2.mlp.gate.weight"))
                .unwrap();
            e[at].2 = vec![128, 2303];
        });
        assert!(
            e.contains("holds dims [2303, 128] where the config gives [2304, 128]"),
            "{e}"
        );
        // The message-query table's absence refuses the loop's own tensors by its name.
        let e = refusal(&|e| {
            e.retain(|(n, _, _)| n != "model.language_model.message_input.query");
        });
        assert!(
            e.contains("message_input.query") && e.contains("the loop's message queries"),
            "{e}"
        );
    }

    /// One stored tensor of a small synthetic folder.
    type SmallTensor = (String, &'static str, Vec<usize>, Vec<u8>);

    /// A small Inkling 2 MLX folder's config at the reader's synthetic widths: six stored layers
    /// of ten visits, the core [2, 4) three times, layer 2 windowless, 4 experts of 32 at the
    /// top 2, four taps, a bias bank 4 x 32.
    fn small_config() -> Json {
        serde_json::from_str(
            r#"{
            "architectures": ["Inkling2LoopForConditionalGeneration"],
            "model_type": "inkling2_loop", "eos_token_id": 2,
            "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
            "text_config": {
                "model_max_length": 4096, "torch_dtype": "bfloat16",
                "hidden_size": 64, "num_hidden_layers": 6, "vocab_size": 8,
                "num_attention_heads": 4, "swa_num_attention_heads": 4,
                "num_key_value_heads": 2, "swa_num_key_value_heads": 2,
                "head_dim": 16, "swa_head_dim": 16,
                "sliding_window_size": 16, "local_layer_ids": [0, 1, 3, 4, 5],
                "d_rel": 4, "rel_extent": 32, "q_bias": false, "o_bias": false,
                "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
                "use_value_norm_no_gamma": true,
                "qk_l2_clip_threshold": 22.627417, "rel_l2_clip_threshold": 8.0,
                "sconv_kernel_size": 4, "unpadded_vocab_size": 8,
                "final_logit_softcapping": 30.0, "n_layers_per_attn_res_block": 2,
                "attn_res_normalize_values": true,
                "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 0,
                "intermediate_size": 32, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0,
                "route_scale": 8.0, "use_gate_bias": true,
                "loop_config": {"core_start": 2, "core_end": 4, "num_passes": 3,
                                 "gate_rank": 8, "is_varloop": false}
            }
        }"#,
        )
        .unwrap()
    }

    /// The small folder's tensors: the first four layers' experts NVFP4 and the last two's
    /// MXFP8, so the plan holds both forms.
    fn small_folder_tensors() -> Vec<SmallTensor> {
        use crate::test_support::mlx_linear;
        let bf16 = |x: f32| -> u16 {
            let bits = x.to_bits();
            ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        };
        let f32s = |values: &[f32]| {
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>()
        };
        let mut t: Vec<SmallTensor> = vec![];
        for stem in ["model.language_model.embed_tokens", "lm_head"] {
            let lin = mlx_linear(stem, 4, &[8], 64);
            t.extend(lin);
        }
        for (name, values) in [
            ("model.language_model.embed_norm.weight", vec![1.5f32; 64]),
            ("model.language_model.norm.weight", vec![1.0f32; 64]),
        ] {
            let bytes = values.iter().flat_map(|v| bf16(*v).to_le_bytes()).collect();
            t.push((name.into(), "BF16", vec![64], bytes));
        }
        t.push((
            "model.language_model.final_input.query".into(),
            "F32",
            vec![64],
            f32s(&(0..64).map(|i| i as f32).collect::<Vec<_>>()),
        ));
        // The message-query table: core layer i's rows 4i..4i+4, the row (i, q) holding 16 +
        // 16·i + q in every value (on the BF16 grid), so the gather's order is readable in the
        // made bytes.
        let table: Vec<f32> = (0..2)
            .flat_map(|i| (0..4).flat_map(move |q| (0..64).map(move |_| (16 + 16 * i + q) as f32)))
            .collect();
        t.push((
            "model.language_model.message_input.query".into(),
            "F32",
            vec![2, 4, 64],
            f32s(&table),
        ));
        for l in 0..6 {
            let m = format!("model.language_model.layers.{l}");
            let windowed = l != 2;
            let norm = |name: String, gain: f32| {
                let bytes: Vec<u8> = (0..64).flat_map(|_| bf16(gain).to_le_bytes()).collect();
                (name, "BF16", vec![64], bytes)
            };
            if windowed {
                t.push(norm(format!("{m}.input_layernorm.weight"), 1.5));
            } else {
                for part in ["q", "k", "v"] {
                    t.push(norm(format!("{m}.{part}_input_layernorm.weight"), 1.0));
                }
            }
            t.push(norm(format!("{m}.post_attention_layernorm.weight"), 1.25));
            // The conv taps: channel c's four taps c·10 + [0..4), reversed in the file.
            let conv = |name: String, rows: usize, kernel: bool| {
                let shape = if kernel {
                    vec![rows, 4, 1]
                } else {
                    vec![4, rows]
                };
                let bytes = if kernel {
                    // MLX's [channels, kernel, 1]: channel-major, tap (c, k) = c·4 + k.
                    let values: Vec<f32> = (0..rows)
                        .flat_map(|c| (0..4).map(move |k| (c * 4 + k) as f32))
                        .collect();
                    values.iter().flat_map(|v| bf16(*v).to_le_bytes()).collect()
                } else {
                    // MLX's static taps: taps-major, each tap a row of channels, newest-first.
                    let values: Vec<f32> = (0..4)
                        .flat_map(|k| (0..rows).map(move |c| (c * 4 + k) as f32))
                        .collect();
                    values.iter().flat_map(|v| bf16(*v).to_le_bytes()).collect()
                };
                (name, "BF16", shape, bytes)
            };
            if windowed {
                t.push(conv(
                    format!("{m}.self_attn.k_sconv.conv1d.weight"),
                    32,
                    true,
                ));
                t.push(conv(
                    format!("{m}.self_attn.v_sconv.conv1d.weight"),
                    32,
                    true,
                ));
            } else {
                t.push(conv(
                    format!("{m}.self_attn.k_sconv.static_weight"),
                    32,
                    false,
                ));
                t.push(conv(
                    format!("{m}.self_attn.v_sconv.static_weight"),
                    32,
                    false,
                ));
                t.extend(mlx_linear(
                    &format!("{m}.self_attn.k_sconv.weight_proj"),
                    4,
                    &[8],
                    64,
                ));
                t.extend(mlx_linear(
                    &format!("{m}.self_attn.v_sconv.weight_proj"),
                    4,
                    &[8],
                    64,
                ));
            }
            t.push(conv(
                format!("{m}.post_attention_convolution.conv1d.weight"),
                64,
                true,
            ));
            t.push(conv(
                format!("{m}.post_mlp_convolution.conv1d.weight"),
                64,
                true,
            ));
            for (stem, rows) in [
                ("q_proj", 64),
                ("k_proj", 32),
                ("v_proj", 32),
                ("o_proj", 64),
                ("gate_proj", 64),
                ("sink_proj", 4),
                ("r_proj", 16),
            ] {
                t.extend(mlx_linear(&format!("{m}.self_attn.{stem}"), 4, &[rows], 64));
            }
            // The clips: value (head, channel) = 100 + head, widened from BF16.
            for (name, heads, width) in [
                ("q_l2_clip.scale", 4usize, 16usize),
                ("rel_l2_clip.scale", 4, 4),
            ] {
                let values: Vec<f32> = (0..heads)
                    .flat_map(|h| (0..width).map(move |_| (100 + h) as f32))
                    .collect();
                let bytes: Vec<u8> = values.iter().flat_map(|v| bf16(*v).to_le_bytes()).collect();
                t.push((
                    format!("{m}.self_attn.{name}"),
                    "BF16",
                    vec![heads, width],
                    bytes,
                ));
            }
            // The bias bank: [d_rel, bank width] rows, value (r, x) = r·64 + x (every value on
            // the BF16 grid) — a windowed layer's bank its window wide, a windowless one's the
            // config's extent.
            let width = if windowed { 16 } else { 32 };
            let bank: Vec<f32> = (0..4)
                .flat_map(|r| (0..width).map(move |x| (r * 64 + x) as f32))
                .collect();
            t.push((
                format!("{m}.self_attn.rel_logits_proj.proj"),
                "BF16",
                vec![4, width],
                bank.iter().flat_map(|v| bf16(*v).to_le_bytes()).collect(),
            ));
            if l > 0 {
                let dims = if windowed { vec![64] } else { vec![3, 64] };
                let values: Vec<f32> = (0..dims.iter().product::<usize>())
                    .map(|i| i as f32)
                    .collect();
                t.push((
                    format!("{m}.attention_input.query"),
                    "F32",
                    dims,
                    f32s(&values),
                ));
            }
            t.push((
                format!("{m}.mlp_input.query"),
                "F32",
                vec![64],
                f32s(&(0..64).map(|i| i as f32).collect::<Vec<_>>()),
            ));
            t.push((
                format!("{m}.mlp.gate.weight"),
                "F32",
                vec![4, 64],
                f32s(&(0..256).map(|i| i as f32).collect::<Vec<_>>()),
            ));
            t.push((
                format!("{m}.mlp.gate.e_score_correction_bias"),
                "F32",
                vec![4],
                f32s(&[1.0, 2.0, 3.0, 4.0]),
            ));
            // The experts: one 3-D codes plane, one 3-D scale plane, NVFP4 with a global scale
            // (the first four layers) or MXFP8.
            for (rows, k, nvfp4) in [(64, 64, l < 4), (64, 32, l < 4)] {
                let (words, groups) = if nvfp4 {
                    (k / 8, k / 16)
                } else {
                    (k / 4, k / 32)
                };
                let codes: Vec<u8> = (0..4 * rows * words * 4).map(|i| i as u8).collect();
                let scales: Vec<u8> = (0..4 * rows * groups).map(|i| (i % 126) as u8).collect();
                let which = if k == 64 { "gate_up_proj" } else { "down_proj" };
                t.push((
                    format!("{m}.mlp.experts.{which}.weight"),
                    "U32",
                    vec![4, rows, words],
                    codes,
                ));
                t.push((
                    format!("{m}.mlp.experts.{which}.scales"),
                    "U8",
                    vec![4, rows, groups],
                    scales,
                ));
                if nvfp4 {
                    let globals: Vec<f32> = (1..=4).map(|e| (e as f32) * 2688.0).collect();
                    t.push((
                        format!("{m}.mlp.experts.{which}.global_scale"),
                        "F32",
                        vec![4],
                        f32s(&globals),
                    ));
                }
            }
            if (2..4).contains(&l) {
                for module in ["q", "k", "v", "mlp"] {
                    let down: Vec<f32> = (0..8 * 128).map(|i| i as f32).collect();
                    t.push((
                        format!("{m}.{module}_message_gate.down_proj.weight"),
                        "BF16",
                        vec![8, 128],
                        down.iter().flat_map(|v| bf16(*v).to_le_bytes()).collect(),
                    ));
                    for proj in ["message_gate_proj", "forget_gate_proj"] {
                        let values: Vec<f32> = (0..64 * 8).map(|i| i as f32).collect();
                        t.push((
                            format!("{m}.{module}_message_gate.{proj}.weight"),
                            "F32",
                            vec![64, 8],
                            f32s(&values),
                        ));
                    }
                }
            }
        }
        t
    }

    /// A small folder of this architecture opens as an inkling2 file: the header's keys and
    /// tensors read through the engine's own reader (six stored layers of ten visits, the loop of
    /// [2, 4) three times, layer 2 windowless), the experts' stacks are read in place at their
    /// planes with the per-expert multipliers of their global scales (and MXFP8 reads none), and
    /// the moves make the file's tensors of the folder's: the bias bank transposed, the static
    /// taps reversed into channel rows, the message queries gathered a core layer.
    #[test]
    fn a_small_folder_opens_as_an_inkling2_file() {
        use gguf::Place;
        use model::arch::inkling2::hparams::{Dialect, Hparams};

        let dir = temp_dir("inkling2-open");
        let mut stored = small_folder_tensors();
        let by_name: BTreeMap<String, SmallTensor> =
            stored.drain(..).map(|t| (t.0.clone(), t)).collect();
        let small: Vec<crate::test_support::Stored> = by_name
            .values()
            .map(|(n, d, s, b)| (n.clone(), *d, s.clone(), b.clone()))
            .collect();
        write_folder(
            &dir,
            &small,
            &small_config(),
            &crate::test_support::tokenizer("gpt-4o"),
        );
        let opened = crate::folder::open(&dir).unwrap();
        assert_eq!(opened.header.architecture(), Some("inkling2"));
        let hp = Hparams::read(&opened.header).unwrap();
        assert_eq!(hp.dialect, Dialect::Unsloth);
        assert_eq!((hp.n_blocks, hp.n_layer), (10, 6));
        assert_eq!(hp.windowed, [true, true, false, true, true, true]);
        assert_eq!(hp.n_expert, 4);
        assert_eq!(
            hp.loop_
                .map(|l| (l.core_start, l.core_end, l.passes, l.gate_rank)),
            Some((2, 4, 3, 8))
        );
        assert_eq!(
            opened.header.value("tokenizer.ggml.pre"),
            Some(&Value::String("gpt-4o".into()))
        );
        // The experts' stacks: NVFP4 on the first four layers, MXFP8 on the last two, read at
        // their planes with their multipliers.
        for (name, rows, k, nvfp4) in [
            ("blk.0.ffn_gate_up_exps.weight", 64usize, 64usize, true),
            ("blk.0.ffn_down_exps.weight", 64, 32, true),
            ("blk.4.ffn_gate_up_exps.weight", 64, 64, false),
            ("blk.5.ffn_down_exps.weight", 64, 32, false),
        ] {
            let (_, info) = opened.header.find(name).unwrap();
            let want = if nvfp4 {
                GgmlType::NVFP4_ROWS
            } else {
                GgmlType::MXFP8
            };
            assert_eq!(info.ty, want, "{name}");
            assert_eq!(info.dims, [k as u64, rows as u64, 4], "{name}");
            let Place::Stack { scales, .. } = opened.header.place(name).unwrap() else {
                panic!("{name} is not a stack");
            };
            if nvfp4 {
                assert_eq!(scales, &[1.0, 2.0, 3.0, 4.0], "{name}");
            } else {
                assert_eq!(scales, &[1.0; 4], "{name}");
            }
            let matrix = opened.header.matrix(name, 1).unwrap();
            let (words, groups) = if nvfp4 {
                (k / 8, k / 16)
            } else {
                (k / 4, k / 32)
            };
            let stem: usize = name
                .strip_prefix("blk.")
                .and_then(|r| r.split_once('.'))
                .and_then(|(l, _)| l.parse().ok())
                .unwrap();
            let which = if k == 64 { "gate_up_proj" } else { "down_proj" };
            let codes =
                &by_name[&format!("model.language_model.layers.{stem}.mlp.experts.{which}.weight")];
            let expert = rows * words * 4;
            assert_eq!(
                matrix.codes,
                &codes.3[expert..2 * expert],
                "{name}: expert 1's code plane"
            );
            let scale_plane =
                &by_name[&format!("model.language_model.layers.{stem}.mlp.experts.{which}.scales")];
            let plane = rows * groups;
            assert_eq!(
                matrix.block_scales,
                &scale_plane.3[plane..2 * plane],
                "{name}: expert 1's scale plane"
            );
        }
        // The `.scale` rows beside the NVFP4 stacks hold the multipliers; MXFP8 reads none.
        let (_, scale) = opened.header.find("blk.0.ffn_gate_up_exps.scale").unwrap();
        assert_eq!(
            (scale.ty, scale.dims.as_slice()),
            (GgmlType::F32, &[4u64][..])
        );
        assert!(opened.header.find("blk.4.ffn_gate_up_exps.scale").is_none());
        // The moved tensors read as the file's bytes: the bank transposed, the static taps
        // reversed into channel rows, the message queries interleaved a core layer.
        use model::part::{Source, Take};
        let planned = opened
            .source
            .plan(|_, _, _| Ok::<_, String>(Take::Whole))
            .unwrap();
        let read = |name: &str| -> Vec<f32> {
            let p = planned.iter().find(|p| p.info.name == name).unwrap();
            opened
                .source
                .read(p)
                .unwrap()
                .spans()
                .concat()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()
        };
        let bank = read("blk.0.attn_rel_bias.weight");
        assert_eq!(bank.len(), 4 * 16);
        for (x, row) in bank.chunks(4).enumerate() {
            for (r, value) in row.iter().enumerate() {
                assert_eq!(*value, (r * 64 + x) as f32, "bank ({r}, {x})");
            }
        }
        let static_taps = read("blk.2.attn_k_conv.weight");
        for (c, row) in static_taps.chunks(4).enumerate() {
            for (k, value) in row.iter().enumerate() {
                assert_eq!(*value, (c * 4 + (3 - k)) as f32, "static taps ({c}, {k})");
            }
        }
        let queries = read("loop_res_query.weight");
        for (i, value) in queries.chunks(64).enumerate() {
            let want = if i % 2 == 0 { 16.0 } else { 19.0 } + 16.0 * (i / 2) as f32;
            assert!(row_all_eq(value, want), "loop_res row {i}: {value:?}");
        }
        let kv = read("loop_kv_res_query.weight");
        for (i, value) in kv.chunks(64).enumerate() {
            let want = if i % 2 == 0 { 17.0 } else { 18.0 } + 16.0 * (i / 2) as f32;
            assert!(row_all_eq(value, want), "loop_kv row {i}: {value:?}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn row_all_eq(values: &[f32], want: f32) -> bool {
        !values.is_empty() && values.iter().all(|v| *v == want)
    }
}
