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
//! The official checkpoint (thinkingmachines' `Inkling-2-Mini`: the same `inkling2_loop` config,
//! tensors under `model.llm.`, `hf_quant_config.json` of `inkling2` parameter groups) maps to the
//! same header by its own names ([`Checkpoint::map_official`]). Its experts are one stack a
//! matrix: NVFP4 of layers 0 to 19 as GGUF's NVFP4 with a tensor scale an expert, MXFP8 of the
//! coda as its own format, each written at load ([`Origin::Made`], the fused gate and up rows
//! un-interleaved by [`Step::Deinterleave`]); the dense projections keep BF16 as stored, or are
//! written Q8_0 where a load asks ([`folder::read`]'s `q8`); the small tensors widen to F32.
//!
//! The header's keys are the config's: the visits, the per-visit head counts and window pattern
//! (a stored layer's repeats read its own), and the loop's bounds — the loop itself is config,
//! and the visits derive at load.

use std::collections::BTreeMap;
use std::path::Path;

use gguf::{GgmlType, StackLayout, Value};
use hf::modelopt::{Normal, Plain, Use};
use hf::schema::{Fields, Rule::*, Type};
use model::arch::inkling2::names as ink;
use model::arch::names as common;
use serde_json::Value as Json;

use super::section::Section;
use super::trunk;
use crate::folder::{self, Error, FolderMap, Mapped, Mapping, Origin, Squeeze, VocabSpec, refuse};

const ARCHITECTURE: &str = "inkling2";

/// The checkpoint's prefix of the text model's tensors.
const MODEL: &str = "model.language_model.";

/// The output head, which a folder of an untied model holds.
const HEAD: &str = "lm_head.weight";

/// The official checkpoint's prefix of the text model's tensors.
const OFFICIAL: &str = "model.llm.";

/// The official checkpoint's vision tower, which the text path does not read.
const VISUAL: &str = "model.visual.";

/// The official checkpoint's model-level table of the loop's attention-and-ffn message queries.
const OFFICIAL_QUERIES: &str = "model.llm.middle_cycle_cross_pass_queries";

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
        "dtype",
        Ignore(
            Type::String,
            "the dtype HF loads in; each tensor keeps its stored dtype and format",
        ),
    ),
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
    ("model_type", Only(r#"["inkling2_loop_text"]"#)),
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

/// `loop_config`: the loop's bounds. `is_varloop` marks the research checkpoint trained to run
/// its core at any pass count from one to five; its `num_passes` is the count it was
/// post-trained at, and [`open_passes`] serves another.
const LOOP: &Fields = &[
    ("core_start", Read),
    ("core_end", Read),
    ("num_passes", Read),
    ("gate_rank", Read),
    ("is_varloop", Read),
];

/// The most passes a variable loop runs: the research checkpoint was trained on one to five.
pub const MAX_PASSES: usize = bloomery_levers::INK2_PASSES_MAX as usize;

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
    /// The official stacks' fused gate·up rows, gate and up alternating (even rows the gate, odd
    /// the up), moved to the file's order: the `rows / 2` gate rows, then the up rows.
    Deinterleave,
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
            Step::Deinterleave => {
                let &[_, rows, experts] = t.dims.as_slice() else {
                    return Err(format!("{:?} is not a stack of matrices", t.dims));
                };
                let (rows, experts) = (rows as usize, experts as usize);
                if rows == 0 || !rows.is_multiple_of(2) || experts == 0 {
                    return Err(format!("{rows} rows of {experts} experts do not pair"));
                }
                t.bytes = deinterleave(&t.bytes, rows, experts).ok_or_else(|| {
                    format!(
                        "{} bytes are not {experts} experts of {rows} whole rows",
                        t.bytes.len()
                    )
                })?;
                Ok(())
            }
        }
    }
}

/// The rows of each of `experts` matrices of `rows` rows in `bytes` (row-major, every row of one
/// byte count), the even rows first and then the odd: output row `r` of the first half is input
/// row `2r`, of the second half input row `2(r - rows / 2) + 1`. `None` when `bytes` is not
/// whole rows of `rows` and `experts`.
fn deinterleave(bytes: &[u8], rows: usize, experts: usize) -> Option<Vec<u8>> {
    let count = rows.checked_mul(experts)?;
    if count == 0 || !bytes.len().is_multiple_of(count) {
        return None;
    }
    let row = bytes.len() / count;
    let half = rows / 2;
    let mut out = Vec::with_capacity(bytes.len());
    for matrix in bytes.chunks_exact(rows * row) {
        for r in 0..rows {
            let from = if r < half { 2 * r } else { 2 * (r - half) + 1 };
            out.extend_from_slice(&matrix[from * row..][..row]);
        }
    }
    Some(out)
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
    /// The core of a variable loop run at one pass: it runs once, so the header carries no loop,
    /// and the folder's message tensors, which no pass reads, are skipped by name.
    idle: Option<Loop>,
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

/// `config` with its loop run at `passes` passes. The message tensors are not indexed by pass, so
/// a fixed loop (`loop_config.is_varloop` false) reads the one set of weights at any count as a
/// variable loop does: the in-memory config is marked variable, so [`Checkpoint::parse`] takes
/// the count (one to [`MAX_PASSES`]) instead of the fixed loop's own two-or-more. The count is
/// held to that range by [`Checkpoint::parse`]; a folder with no loop is refused by name.
fn with_passes(config: &mut Json, passes: usize) -> Result<(), Error> {
    const AT: &str = "config.json.text_config.loop_config";
    let Some(loop_) = config
        .pointer_mut("/text_config/loop_config")
        .and_then(Json::as_object_mut)
    else {
        return Err(Error::Refused(format!(
            "{AT} is missing: a pass count needs a looped folder, and this one holds no loop"
        )));
    };
    loop_.insert("is_varloop".to_string(), Json::Bool(true));
    loop_.insert("num_passes".to_string(), Json::from(passes));
    Ok(())
}

/// The `num_passes` the folder at `dir` was trained at: its `config.json`'s own loop count.
pub fn trained_passes(dir: &Path) -> Result<usize, Error> {
    let config = folder::read_config(dir)?;
    config
        .pointer("/text_config/loop_config/num_passes")
        .and_then(Json::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| {
            Error::Refused(format!(
                "{}: config.json.text_config.loop_config.num_passes is missing: the folder holds no loop",
                dir.display()
            ))
        })
}

/// [`folder::open_with`] of an Inkling 2 folder with its loop run at `passes` passes where one is
/// asked (`None`: the folder's own `num_passes`, which is [`folder::open_with`]'s open). The
/// header names the visits and the loop at that count (one pass has no loop: its message tensors
/// are skipped), so a plan, a load and the step walk follow it. A folder of another model type
/// and a count outside one to [`MAX_PASSES`] are refused by name; a fixed loop (the Mini) takes
/// the count as an experiment, its output at another count than its own not guaranteed.
pub fn open_passes(dir: &Path, q8: bool, passes: Option<usize>) -> Result<folder::Opened, Error> {
    let mut config = folder::read_config(dir)?;
    let model_type = config
        .get("model_type")
        .and_then(Json::as_str)
        .unwrap_or_default();
    if !Checkpoint::MODEL_TYPES.contains(&model_type) {
        return Err(Error::Refused(format!(
            "{}: model_type {model_type:?} is no Inkling 2 folder",
            dir.display()
        )));
    }
    if let Some(passes) = passes {
        with_passes(&mut config, passes)?;
    }
    folder::open_as::<Checkpoint>(dir, &config, q8)
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
        let (loop_, idle) = match text.value.get("loop_config") {
            Some(value) => {
                let section = Section::<Error>::new(value, "config.json.text_config.loop_config")?;
                section.check(&[LOOP])?;
                let varloop = section
                    .field("is_varloop")?
                    .as_bool()
                    .ok_or_else(|| section.bad("is_varloop", "must be a boolean"))?;
                let loop_ = Loop {
                    core_start: section.count("core_start")? as usize,
                    core_end: section.count("core_end")? as usize,
                    passes: section.count("num_passes")? as usize,
                    gate_rank: section.count("gate_rank")? as usize,
                };
                if loop_.passes < 2 && !varloop {
                    return Err(section.bad(
                        "num_passes",
                        &format!(
                            "is {}; the fixed loop runs at least two passes over its core",
                            loop_.passes
                        ),
                    ));
                }
                if varloop && loop_.passes > MAX_PASSES {
                    return Err(section.bad(
                        "num_passes",
                        &format!(
                            "is {}; the variable loop was trained on one to {MAX_PASSES} passes",
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
                if loop_.passes == 1 {
                    (None, Some(loop_))
                } else {
                    (Some(loop_), None)
                }
            }
            None => (None, None),
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
            idle,
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
        // The official folder holds `model.llm.` names alone; a folder with both prefixes is no
        // official one, and the MLX map refuses the foreign names.
        if tensors.iter().any(|&(n, _)| n.starts_with(OFFICIAL))
            && !tensors.iter().any(|&(n, _)| n.starts_with(MODEL))
        {
            return self.map_official(tensors);
        }
        if self.idle.is_some() {
            return Err(refuse(
                MESSAGE_INPUT,
                "a variable loop at one pass is read from the official folder only: this map \
                 does not skip the message tensors",
            ));
        }
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
        self.complete(&out, message, MESSAGE_INPUT)?;
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

    /// Whether the official tensor `short` (the name past `model.llm.`) is a message tensor of
    /// the loop of a variable-loop folder run at one pass: the two query tables and a core
    /// layer's controllers.
    fn is_idle(&self, short: &str) -> bool {
        let Some(idle) = self.idle else {
            return false;
        };
        short == "middle_cycle_cross_pass_queries"
            || short == "middle_cycle_cross_pass_kv_queries"
            || short
                .strip_prefix("layers.")
                .and_then(|r| r.split_once('.'))
                .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
                .is_some_and(|(l, rest)| {
                    rest.starts_with("controllers.")
                        && (idle.core_start..idle.core_end).contains(&l)
                })
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

    /// The official checkpoint's tensors (`model.llm.`) mapped to the header's: each by its own
    /// name, the vision tower skipped by name, and every tensor the header holds mapped at the
    /// dims the config gives ([`Checkpoint::complete`]).
    fn map_official<'a>(&self, tensors: &[(&'a str, &'a [usize])]) -> Result<Mapping<Step>, Error> {
        let mut out: BTreeMap<String, Mapped<Step>> = BTreeMap::new();
        let mut skipped = BTreeMap::new();
        for &(name, shape) in tensors {
            if !trunk::sized(shape) {
                return Err(refuse(
                    name,
                    format!("shape {shape:?} is empty, zero, or overflows usize"),
                ));
            }
            if name.starts_with(VISUAL) {
                skipped.insert(
                    name.to_string(),
                    "the vision tower, which the text path does not read".to_string(),
                );
                continue;
            }
            let short = name
                .strip_prefix(OFFICIAL)
                .ok_or_else(|| refuse(name, "not an Inkling 2 tensor of the official folder"))?;
            if self.is_idle(short) {
                skipped.insert(
                    name.to_string(),
                    "a message tensor of the loop, which one pass never runs".to_string(),
                );
                continue;
            }
            let (to, m) = self.official_tensor(name, short, shape)?;
            if out.insert(to, m).is_some() {
                return Err(refuse(name, "maps to a name another tensor has"));
            }
        }
        let table = tensors
            .iter()
            .find(|&&(n, _)| n == OFFICIAL_QUERIES)
            .map(|&(n, _)| n)
            .filter(|_| self.idle.is_none());
        self.complete(&out, table, OFFICIAL_QUERIES)?;
        Ok(Mapping {
            tensors: out,
            skipped,
        })
    }

    /// The header tensor the official tensor `name` (`short` past `model.llm.`, stored as
    /// `shape`) becomes, and how: a dense projection as its role reads it (BF16 as stored, or
    /// Q8_0 where a load asks), the tensors the file keeps BF16 as stored, an F32 table in place,
    /// every other small tensor widened to F32.
    fn official_tensor(
        &self,
        name: &str,
        short: &str,
        shape: &[usize],
    ) -> Result<(String, Mapped<Step>), Error> {
        let dims = || shape.iter().rev().map(|&d| d as u64).collect::<Vec<_>>();
        let made = |role: Use, steps: Vec<Step>, dims: Vec<u64>, origin: Origin| Mapped {
            sources: vec![name.to_string()],
            role,
            steps,
            dims,
            origin,
        };
        let linear = |dims: Vec<u64>| made(Use::Matrix(Plain::Q8_0), vec![], dims, Origin::Made);
        let bf16 = |dims: Vec<u64>| made(Use::Matrix(Plain::Bf16), vec![], dims, Origin::Made);
        let widened = |steps: Vec<Step>, dims: Vec<u64>| {
            made(Use::Matrix(Plain::F32), steps, dims, Origin::Made)
        };
        let f32_file = |dims: Vec<u64>| {
            made(
                Use::Matrix(Plain::F32),
                vec![],
                dims,
                Origin::File(GgmlType::F32),
            )
        };
        // One matrix an expert, written at load: the stack's own type and `.scale`.
        let stack = |steps: Vec<Step>, dims: Vec<u64>| {
            made(Use::Matrix(Plain::Q8_0), steps, dims, Origin::Made)
        };
        match short {
            "embed.weight" => return Ok((common::token_embd(), linear(dims()))),
            "embed_norm.weight" => {
                return Ok((
                    ink::TOKEN_EMBD_NORM.to_string(),
                    widened(vec![], vec![self.hidden]),
                ));
            }
            "norm.weight" => return Ok((common::output_norm(), widened(vec![], dims()))),
            "unembed.weight" => return Ok((common::output(), linear(dims()))),
            "attn_res.output_query" => {
                return Ok((
                    ink::OUTPUT_RES_QUERY.to_string(),
                    f32_file(vec![self.hidden]),
                ));
            }
            "middle_cycle_cross_pass_queries" => {
                return Ok((ink::LOOP_RES_QUERY.to_string(), f32_file(dims())));
            }
            "middle_cycle_cross_pass_kv_queries" => {
                return Ok((ink::LOOP_KV_RES_QUERY.to_string(), f32_file(dims())));
            }
            _ => {}
        }
        // The per-layer query tables: a row a layer, the attention mix of layer `l` at `l - 1`.
        let table = |prefix: &str, shift: usize| -> Result<Option<usize>, Error> {
            let Some(i) = short.strip_prefix(prefix) else {
                return Ok(None);
            };
            i.parse::<usize>()
                .ok()
                .map(|i| i + shift)
                .filter(|&l| l < self.layers)
                .map(Some)
                .ok_or_else(|| refuse(name, format!("layer {i:?} of {}", self.layers)))
        };
        if let Some(l) = table("attn_res.attn_queries.", 1)? {
            return Ok((common::blk(l, "attn_res_query.weight"), f32_file(dims())));
        }
        if let Some(l) = table("attn_res.mlp_queries.", 0)? {
            return Ok((common::blk(l, "ffn_res_query.weight"), f32_file(dims())));
        }
        let Some((l, rest)) = short
            .strip_prefix("layers.")
            .and_then(|r| r.split_once('.'))
            .and_then(|(l, rest)| Some((l.parse::<usize>().ok()?, rest)))
        else {
            return Err(refuse(
                name,
                "not an Inkling 2 tensor of the official folder",
            ));
        };
        if l >= self.layers {
            return Err(refuse(name, format!("layer {l} of {}", self.layers)));
        }
        let windowed = self.windowed[l];
        let stem = |to: &str| common::blk(l, to);
        let conv_taps = || widened(vec![Step::Squeeze], vec![self.conv, self.key_rows]);
        let static_taps = || widened(vec![Step::StaticTaps], vec![self.conv, self.key_rows]);
        let (to, m) = match rest {
            "attn.wq_du.weight" => (stem("attn_q.weight"), linear(dims())),
            "attn.wk_dv.weight" => (stem("attn_k.weight"), linear(dims())),
            "attn.wv_dv.weight" => (stem("attn_v.weight"), linear(dims())),
            "attn.wo_ud.weight" => (stem("attn_output.weight"), linear(dims())),
            "attn.gate_attn.weight" => (stem("attn_gate.weight"), linear(dims())),
            "attn.wr_du.weight" => (stem("attn_rel.weight"), linear(dims())),
            "attn.attn_sink_proj.weight" => (stem("attn_sink_proj.weight"), bf16(dims())),
            "attn.rel_logits_proj.proj" => (
                stem("attn_rel_bias.weight"),
                widened(
                    vec![Step::Bank],
                    vec![self.d_rel, if windowed { self.swa } else { self.extent }],
                ),
            ),
            "attn.q_l2_clip.scale" => (stem("attn_q_clip.weight"), widened(vec![], dims())),
            "attn.rel_l2_clip.scale" => (stem("attn_rel_clip.weight"), widened(vec![], dims())),
            "attn.k_sconv.weight" if windowed => (stem("attn_k_conv.weight"), conv_taps()),
            "attn.v_sconv.weight" if windowed => (stem("attn_v_conv.weight"), conv_taps()),
            "attn.k_sconv.static_w" if !windowed => (stem("attn_k_conv.weight"), static_taps()),
            "attn.v_sconv.static_w" if !windowed => (stem("attn_v_conv.weight"), static_taps()),
            "attn.k_sconv.weight_proj" if !windowed => (ink::inp(l, ink::Inp::ConvK), bf16(dims())),
            "attn.v_sconv.weight_proj" if !windowed => (ink::inp(l, ink::Inp::ConvV), bf16(dims())),
            "attn_sconv.weight" => (
                stem("attn_post_conv.weight"),
                widened(vec![Step::Squeeze], vec![self.conv, self.hidden]),
            ),
            "mlp_sconv.weight" => (
                stem("ffn_post_conv.weight"),
                widened(vec![Step::Squeeze], vec![self.conv, self.hidden]),
            ),
            "attn_norm.weight" if windowed => (stem("attn_norm.weight"), widened(vec![], dims())),
            "q_input_norm.weight" if !windowed => {
                (ink::inp(l, ink::Inp::NormQ), widened(vec![], dims()))
            }
            "k_input_norm.weight" if !windowed => {
                (ink::inp(l, ink::Inp::NormK), widened(vec![], dims()))
            }
            "v_input_norm.weight" if !windowed => {
                (ink::inp(l, ink::Inp::NormV), widened(vec![], dims()))
            }
            "mlp_norm.weight" => (stem("ffn_norm.weight"), widened(vec![], dims())),
            "mlp.gate.weight" => (stem("ffn_gate_inp.weight"), widened(vec![], dims())),
            "mlp.gate.bias" => (common::exp_probs_b(l), f32_file(dims())),
            "mlp.experts.w13_weight" => (
                stem("ffn_gate_up_exps.weight"),
                stack(vec![Step::Deinterleave], dims()),
            ),
            "mlp.experts.w2_weight" => (common::ffn_down_exps(l), stack(vec![], dims())),
            other if other.starts_with("controllers.") => {
                if !self.in_core(l) {
                    return Err(refuse(
                        name,
                        format!(
                            "layer {l} is outside the loop's core, which holds no message \
                             controller"
                        ),
                    ));
                }
                let (module, part) = other
                    .strip_prefix("controllers.")
                    .and_then(|r| r.split_once('.'))
                    .ok_or_else(|| refuse(name, "not an Inkling 2 tensor"))?;
                let module = match module {
                    "attn" => ink::Ctrl::Attn,
                    "k" => ink::Ctrl::K,
                    "v" => ink::Ctrl::V,
                    "mlp" => ink::Ctrl::Ffn,
                    _ => return Err(refuse(name, "not an Inkling 2 controller")),
                };
                let part = match part {
                    "down_weight" => ink::CtrlPart::Down,
                    "gain_weight" => ink::CtrlPart::Gain,
                    "forget_weight" => ink::CtrlPart::Forget,
                    _ => return Err(refuse(name, "not an Inkling 2 controller tensor")),
                };
                let dims = if part == ink::CtrlPart::Down {
                    vec![2 * self.hidden, self.rank()]
                } else {
                    vec![self.rank(), self.hidden]
                };
                (ink::loop_ctrl_pr(l, module, part), widened(vec![], dims))
            }
            other => {
                return Err(refuse(
                    name,
                    format!("{other:?} is not an Inkling 2 tensor"),
                ));
            }
        };
        Ok((to, m))
    }

    /// Every tensor the header holds is mapped, at the dims the config gives: the model level,
    /// then each stored layer's tensors by its kind. `message` names the model-level
    /// message-query table, which a looped folder holds and a loopless one lacks.
    fn complete(
        &self,
        mapped: &BTreeMap<String, Mapped<Step>>,
        message: Option<&str>,
        table: &str,
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
                table,
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
        let e = refuse(&text(
            "loop_config",
            json!({"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": "yes"}),
        ));
        assert!(e.contains("is_varloop") && e.contains("boolean"), "{e}");
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

    const OFFICIAL_CONFIG: &str = "quant-configs/thinkingmachines__Inkling-2-Mini__config.json.gz";
    const OFFICIAL_QUANT: &str =
        "quant-configs/thinkingmachines__Inkling-2-Mini__hf_quant_config.json.gz";
    const OFFICIAL_HEADER: &str =
        "checkpoint-headers/inkling-2-mini-official-model-00001-of-00001.header.json.gz";

    /// The official folder's header planned as a load (`q8` false: BF16 as stored) or a Q8_0 load
    /// reads it.
    fn official_planning(q8: bool) -> (Entries, folder::Planning<Step>) {
        let config = json(OFFICIAL_CONFIG);
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let mut header = json(OFFICIAL_HEADER);
        header.as_object_mut().unwrap().remove("__metadata__");
        let entries = entries(&header);
        let planning = folder::plan_tensors(
            &checkpoint,
            &Config::parse(&config, Some(&json(OFFICIAL_QUANT))).unwrap(),
            &entries,
            q8,
            folder::Keep::InPlace,
        )
        .unwrap();
        (entries, planning)
    }

    /// The official folder's header plans every text tensor in the GGUF twin's types and skips the
    /// vision tower: the 694 tensors and 40 `.scale` rows the twin (unsloth's UD-Q4_K_XL) holds,
    /// the NVFP4 experts of layers 0 to 19 as GGUF's NVFP4 with a `.scale` an expert, the coda's
    /// as MXFP8, the dense projections BF16 as stored (Q8_0 where the load asks), the sink
    /// projection and the conv projections BF16 either way, and every small tensor F32. The plan
    /// refuses nothing and the map leaves no folder tensor unread.
    #[test]
    fn the_official_headers_plan_every_text_tensor_in_the_twin_s_types() {
        let (entries, bf16) = official_planning(false);
        let (_, q8) = official_planning(true);
        assert_eq!(entries.len(), 1244);
        for planning in [&bf16, &q8] {
            assert!(
                planning.plan.refused.is_empty(),
                "{:?}",
                planning.plan.refused
            );
            let visual = entries
                .iter()
                .filter(|(n, _, _)| n.starts_with("model.visual."))
                .count();
            assert_eq!(planning.skipped.len(), visual);
            assert_eq!(planning.planned.len(), 734);
            assert_eq!(
                planning
                    .planned
                    .iter()
                    .filter(|p| !p.decl.name.ends_with(".scale"))
                    .count(),
                694
            );
        }
        let ty = |p: &folder::Planning<Step>, n: &str| {
            p.planned
                .iter()
                .find(|p| p.decl.name == n)
                .unwrap_or_else(|| panic!("{n} is not planned"))
                .decl
                .type_id
        };
        let (bf16_ty, q8_ty) = (GgmlType::BF16.as_u32(), GgmlType::Q8_0.as_u32());
        for (name, nvfp4) in [
            ("blk.0.ffn_gate_up_exps.weight", true),
            ("blk.19.ffn_down_exps.weight", true),
            ("blk.20.ffn_gate_up_exps.weight", false),
            ("blk.23.ffn_down_exps.weight", false),
        ] {
            let want = if nvfp4 {
                GgmlType::NVFP4
            } else {
                GgmlType::MXFP8
            };
            for p in [&bf16, &q8] {
                assert_eq!(ty(p, name), want.as_u32(), "{name}");
            }
            let scale = name.replace(".weight", ".scale");
            assert_eq!(
                bf16.planned.iter().any(|p| p.decl.name == scale),
                nvfp4,
                "{scale}"
            );
        }
        assert_eq!(
            bf16.planned
                .iter()
                .find(|p| p.decl.name == "blk.19.ffn_down_exps.scale")
                .unwrap()
                .decl
                .dims,
            [128]
        );
        for name in [
            "token_embd.weight",
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.7.attn_k.weight",
            "blk.23.attn_output.weight",
            "blk.3.attn_gate.weight",
            "blk.5.attn_rel.weight",
        ] {
            assert_eq!(ty(&bf16, name), bf16_ty, "{name}");
            assert_eq!(ty(&q8, name), q8_ty, "{name}");
        }
        for name in [
            "blk.0.attn_sink_proj.weight",
            "blk.5.attn_k_conv_proj.weight",
            "blk.11.attn_v_conv_proj.weight",
        ] {
            assert_eq!(ty(&bf16, name), bf16_ty, "{name}");
            assert_eq!(ty(&q8, name), bf16_ty, "{name}");
        }
        for name in [
            "token_embd_norm.weight",
            "output_norm.weight",
            "output_res_query.weight",
            "loop_res_query.weight",
            "loop_kv_res_query.weight",
            "blk.0.attn_norm.weight",
            "blk.5.attn_q_inp_norm.weight",
            "blk.0.attn_k_conv.weight",
            "blk.5.attn_k_conv.weight",
            "blk.0.attn_post_conv.weight",
            "blk.0.attn_rel_bias.weight",
            "blk.0.attn_q_clip.weight",
            "blk.0.ffn_gate_inp.weight",
            "blk.0.exp_probs_b.bias",
            "blk.0.ffn_res_query.weight",
            "blk.1.attn_res_query.weight",
            "blk.6.loop_attn_down.weight",
            "blk.6.loop_mlp_gain.weight",
            "blk.17.loop_v_forget.weight",
        ] {
            assert_eq!(ty(&bf16, name), GgmlType::F32.as_u32(), "{name}");
            assert_eq!(ty(&q8, name), GgmlType::F32.as_u32(), "{name}");
        }
    }

    /// The fused gate·up rows of the official stacks alternate (even rows the gate, odd the up);
    /// the file holds the gate rows then the up rows. The step moves whole rows of any block
    /// type, expert by expert, and refuses a stack it cannot pair by name.
    ///
    /// FAIL-first: a step that leaves the rows as stored fails the row order.
    #[test]
    fn the_gate_up_rows_unweave_expert_by_expert() {
        let (rows, experts, row) = (6usize, 2usize, 3usize);
        let bytes: Vec<u8> = (0..rows * experts).flat_map(|r| [r as u8; 3]).collect();
        assert_eq!(bytes.len(), rows * experts * row);
        let mut t = Normal {
            type_id: GgmlType::NVFP4.as_u32(),
            dims: vec![64, rows as u64, experts as u64],
            bytes,
            scale: vec![1.0, 2.0],
        };
        Step::Deinterleave.apply(&mut t).unwrap();
        let order: Vec<u8> = t.bytes.chunks(row).map(|r| r[0]).collect();
        assert_eq!(order, [0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11]);
        assert!(t.bytes.chunks(row).all(|r| r.iter().all(|&b| b == r[0])));
        assert_eq!(t.dims, [64, 6, 2]);
        assert_eq!(t.scale, [1.0, 2.0]);
        let mut odd = Normal {
            dims: vec![64, 5, 2],
            bytes: vec![0; 30],
            ..Normal::default()
        };
        assert!(
            Step::Deinterleave
                .apply(&mut odd)
                .unwrap_err()
                .contains("do not pair")
        );
        let mut flat = Normal {
            dims: vec![64, 6],
            bytes: vec![0; 12],
            ..Normal::default()
        };
        assert!(
            Step::Deinterleave
                .apply(&mut flat)
                .unwrap_err()
                .contains("not a stack of matrices")
        );
        assert_eq!(flat.dims, [64, 6]);
    }

    /// The official Mini config as the variable-loop checkpoint's: the same fields with
    /// `is_varloop` true (`TML-d24-varloop-research`'s `config.json` differs by that one line), its
    /// loop run at `passes` where one is asked ([`with_passes`]).
    fn varloop_config(passes: Option<usize>) -> Result<Json, Error> {
        let mut config = json(OFFICIAL_CONFIG);
        config["text_config"]["loop_config"]["is_varloop"] = json!(true);
        if let Some(passes) = passes {
            with_passes(&mut config, passes)?;
        }
        Ok(config)
    }

    /// The official folder's header planned at `passes` of a variable loop: the map, its header
    /// keys and the plan of every text tensor (`q8`, the served load).
    fn varloop_planning(passes: Option<usize>) -> (Checkpoint, folder::Planning<Step>) {
        let config = varloop_config(passes).unwrap();
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let mut header = json(OFFICIAL_HEADER);
        header.as_object_mut().unwrap().remove("__metadata__");
        let planning = folder::plan_tensors(
            &checkpoint,
            &Config::parse(&config, Some(&json(OFFICIAL_QUANT))).unwrap(),
            &entries(&header),
            true,
            folder::Keep::InPlace,
        )
        .unwrap();
        (checkpoint, planning)
    }

    /// A variable loop's header follows the pass count: `block_count` the visits (the 24 stored
    /// layers and the core's 12 once a pass past the first), `num_loops` the count, the pattern
    /// and head counts a value a visit, and the cache slots' windowless visits one a pass of
    /// layers 11 and 17. One pass has no loop: the header carries no loop key, and the 146
    /// message tensors (the two query tables and the 12 core layers' 12 controller tensors each)
    /// are skipped by name, which no pass reads. The count the folder's own config gives (3) is
    /// the Mini's header key for key.
    ///
    /// FAIL-first: `with_passes` leaving `num_passes` as stored holds every count at 48 visits.
    #[test]
    fn a_variable_loop_s_header_follows_the_pass_count() {
        let key = |c: &Checkpoint, suffix: &str| {
            c.keys
                .iter()
                .find(|(k, _)| k == suffix)
                .map(|(_, v)| v.clone())
        };
        let mini = Checkpoint::parse(&json(OFFICIAL_CONFIG)).unwrap();
        for passes in 1..=MAX_PASSES {
            let (checkpoint, planning) = varloop_planning(Some(passes));
            let visits = 24 + (passes - 1) * 12;
            assert_eq!(
                key(&checkpoint, "block_count"),
                Some(Value::U32(visits as u32)),
                "{passes} passes"
            );
            let Some(Value::Array(pattern)) = key(&checkpoint, "attention.sliding_window_pattern")
            else {
                panic!("the pattern is an array");
            };
            let windowless: Vec<usize> = (0..visits)
                .filter(|&v| pattern[v] == Value::Bool(false))
                .collect();
            // Layers 5 and 23 once, 11 and 17 once a pass.
            let want: Vec<usize> = std::iter::once(5)
                .chain((0..passes).flat_map(|p| [11 + 12 * p, 17 + 12 * p]))
                .chain([23 + 12 * (passes - 1)])
                .collect();
            assert_eq!(windowless, want, "{passes} passes");
            let message = planning
                .skipped
                .keys()
                .filter(|n| !n.starts_with("model.visual."))
                .count();
            if passes == 1 {
                assert_eq!(key(&checkpoint, "num_loops"), None);
                assert_eq!(key(&checkpoint, "loop.core_end"), None);
                assert_eq!(message, 146, "the two tables and 12 layers' 12 controllers");
                assert_eq!(planning.planned.len(), 734 - 146);
            } else {
                assert_eq!(
                    key(&checkpoint, "num_loops"),
                    Some(Value::U32(passes as u32))
                );
                assert_eq!(message, 0, "{passes} passes");
                assert_eq!(planning.planned.len(), 734);
            }
            assert!(
                planning.plan.refused.is_empty(),
                "{:?}",
                planning.plan.refused
            );
        }
        let (own, _) = varloop_planning(None);
        let (three, _) = varloop_planning(Some(3));
        assert_eq!(
            own.keys, mini.keys,
            "the folder's own count is the Mini's header"
        );
        assert_eq!(three.keys, mini.keys, "an asked 3 is the folder's own");
    }

    /// A fixed loop takes an asked count as the variable loop does (the message tensors are not
    /// indexed by pass), the pass count is refused by name where the folder holds no loop or the
    /// count is outside one to five, and the folder's own fixed `num_passes` of 1 stays refused.
    #[test]
    fn a_pass_count_is_refused_off_a_loop_and_off_the_range() {
        for passes in 1..=MAX_PASSES {
            let mut fixed = json(OFFICIAL_CONFIG);
            with_passes(&mut fixed, passes).unwrap();
            let checkpoint = Checkpoint::parse(&fixed).unwrap();
            let mut variable = json(OFFICIAL_CONFIG);
            variable["text_config"]["loop_config"]["is_varloop"] = json!(true);
            with_passes(&mut variable, passes).unwrap();
            assert_eq!(
                checkpoint.keys,
                Checkpoint::parse(&variable).unwrap().keys,
                "{passes} passes"
            );
        }
        let mut over = json(OFFICIAL_CONFIG);
        with_passes(&mut over, MAX_PASSES + 1).unwrap();
        let e = Checkpoint::parse(&over).unwrap_err().to_string();
        assert!(e.contains("num_passes"), "{e}");
        let mut loopless = json(OFFICIAL_CONFIG);
        loopless["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("loop_config");
        let e = with_passes(&mut loopless, 2).unwrap_err().to_string();
        assert!(e.contains("loop_config"), "{e}");
        for passes in [0, MAX_PASSES + 1] {
            let e = Checkpoint::parse(&varloop_config(Some(passes)).unwrap())
                .unwrap_err()
                .to_string();
            assert!(e.contains("num_passes"), "{passes}: {e}");
        }
        // A fixed loop's one pass stays refused.
        let mut one = json(OFFICIAL_CONFIG);
        one["text_config"]["loop_config"]["num_passes"] = json!(1);
        let e = Checkpoint::parse(&one).unwrap_err().to_string();
        assert!(e.contains("at least two passes"), "{e}");
    }

    /// The MLX map does not skip the message tensors of a one-pass loop, so it refuses by name;
    /// the official map keeps a looped folder without its message tables refused.
    #[test]
    fn the_one_pass_loop_is_read_from_the_official_folder_only() {
        let checkpoint = Checkpoint::parse(&varloop_config(Some(1)).unwrap()).unwrap();
        let mlx: Vec<(&str, &[usize])> = vec![("model.language_model.norm.weight", &[2304])];
        let e = checkpoint.map(&mlx).err().unwrap().to_string();
        assert!(e.contains("official folder only"), "{e}");
    }

    /// A folder with tensors of both dialects (the MLX map's, which refuses the official names)
    /// and one with a tensor the official map does not name are refused by name.
    #[test]
    fn a_folder_of_neither_dialect_is_refused_by_name() {
        let config = json(OFFICIAL_CONFIG);
        let checkpoint = Checkpoint::parse(&config).unwrap();
        let both: Vec<(&str, &[usize])> = vec![
            ("model.llm.norm.weight", &[2304]),
            ("model.language_model.norm.weight", &[2304]),
        ];
        let e = checkpoint.map(&both).err().unwrap().to_string();
        assert!(
            e.contains("model.llm.norm.weight")
                && e.contains("not an Inkling 2 tensor of the text model"),
            "{e}"
        );
        let stray: Vec<(&str, &[usize])> = vec![
            ("model.llm.norm.weight", &[2304]),
            ("model.llm.layers.0.attn.unknown.weight", &[2, 2]),
        ];
        let e = checkpoint.map(&stray).err().unwrap().to_string();
        assert!(e.contains("model.llm.layers.0.attn.unknown.weight"), "{e}");
    }

    /// The official folder at `$BLOOMERY_INK2_OFFICIAL` read as the GGUF twin at
    /// `$BLOOMERY_INK2_GGUF` (unsloth's UD-Q4_K_XL file): the folder opens, tokenizer included, and
    /// every tensor the map makes of it is the twin's, byte for byte where the twin holds the same
    /// type — the NVFP4 stacks of layers 0 to 19 (codes, block scales and the `.scale` an expert
    /// as f32 bits, the fused gate and up rows in the file's order), every F32 tensor, the BF16
    /// ones the file keeps, and, in a read that asks for Q8_0, every dense projection and the
    /// embedding and head. A read that keeps BF16 holds each dense tensor as the folder stores it,
    /// two bytes a weight. The coda's MXFP8 stacks (layers 20 to 23) are the twin's dims in
    /// MXFP8's blocks, which the twin's Q8_0 does not share.
    ///
    /// FAIL-first: an identity un-interleave leaves the gate and up rows alternating, and the
    /// first NVFP4 stack differs from the twin's at its first up row.
    #[test]
    #[ignore = "needs the official folder at $BLOOMERY_INK2_OFFICIAL and the GGUF twin at $BLOOMERY_INK2_GGUF"]
    fn hw_the_official_folder_reads_as_the_served_gguf() {
        let dir = std::path::PathBuf::from(
            std::env::var_os("BLOOMERY_INK2_OFFICIAL")
                .expect("$BLOOMERY_INK2_OFFICIAL names the official folder"),
        );
        let twin = gguf::Gguf::open(
            std::env::var_os("BLOOMERY_INK2_GGUF").expect("$BLOOMERY_INK2_GGUF names the twin"),
        )
        .unwrap();
        let made = |read: &folder::Read<Checkpoint>, p: &folder::Planned<Step>| -> Normal {
            let folder::Make::Tensor(m) = &p.make else {
                panic!("{}: not a tensor of the folder", p.decl.name);
            };
            let mut n = hf::modelopt::normalize(
                |s| read.folder.raw(s),
                &m.placed(&read.plan).unwrap(),
                m.role,
            )
            .unwrap_or_else(|e| panic!("{}: {e}", p.decl.name));
            if let Some(step) = read.checkpoint.mover(&p.decl.name, &m.steps) {
                step(&mut n).unwrap();
            }
            assert_eq!(n.type_id, p.decl.type_id, "{}", p.decl.name);
            assert_eq!(n.bytes.len() as u64, p.decl.nbytes, "{}", p.decl.name);
            n
        };

        // Q8_0 where asked: the twin's own types, every tensor.
        let q8 = folder::read::<Checkpoint>(&dir, true).unwrap();
        assert_eq!(q8.tokenizer.pre(), "gpt-4o");
        let (mut equal, mut stacks, mut coda) = (0, 0, 0);
        for p in &q8.planned {
            let name = p.decl.name.as_str();
            if name.ends_with(".scale") {
                continue;
            }
            let info = twin
                .find(name)
                .unwrap_or_else(|| panic!("{name} is not in the twin"));
            assert_eq!(info.dims, p.decl.dims, "{name}");
            let layer = name
                .strip_prefix("blk.")
                .and_then(|r| r.split_once('.'))
                .and_then(|(l, _)| l.parse::<usize>().ok());
            if name.contains("_exps.") && layer.is_some_and(|l| l >= 20) {
                // MXFP8 against the twin's Q8_0: the same matrix, another block type.
                assert_eq!(p.decl.type_id, GgmlType::MXFP8.as_u32(), "{name}");
                assert_eq!(info.ty, GgmlType::Q8_0, "{name}");
                let n = made(&q8, p);
                assert_eq!(n.dims, p.decl.dims, "{name}");
                assert!(n.scale.is_empty(), "{name}");
                coda += 1;
                continue;
            }
            assert_eq!(info.ty.as_u32(), p.decl.type_id, "{name}");
            let n = made(&q8, p);
            assert!(
                n.bytes == twin.data(info).unwrap(),
                "{name}: {} bytes differ from the twin's",
                n.bytes.len()
            );
            if name.contains("_exps.") {
                let scale = twin
                    .find(&name.replace(".weight", ".scale"))
                    .unwrap_or_else(|| panic!("{name} has no .scale in the twin"));
                let want: Vec<u32> = twin
                    .data(scale)
                    .unwrap()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect();
                let got: Vec<u32> = n.scale.iter().map(|s| s.to_bits()).collect();
                assert_eq!(got, want, "{name}: the .scale an expert");
                stacks += 1;
            }
            equal += 1;
        }
        // 40 NVFP4 stacks, 8 MXFP8; every other tensor of the 694 the twin holds equal.
        assert_eq!((stacks, coda, equal), (40, 8, 686));

        // BF16 as stored where the read keeps it: each dense tensor two bytes a weight.
        let bf16 = folder::read::<Checkpoint>(&dir, false).unwrap();
        let mut dense = 0;
        for p in &bf16.planned {
            let info = twin.find(&p.decl.name);
            if p.decl.type_id != GgmlType::BF16.as_u32()
                || info.is_none_or(|i| i.ty != GgmlType::Q8_0)
            {
                continue;
            }
            let folder::Make::Tensor(m) = &p.make else {
                panic!("{}: not a tensor of the folder", p.decl.name);
            };
            let stored = bf16.folder.raw(&m.sources[0]).unwrap();
            assert_eq!(stored.dtype, Dtype::Bf16, "{}", p.decl.name);
            let n = made(&bf16, p);
            assert_eq!(n.bytes.len(), 2 * stored.shape.iter().product::<usize>());
            assert!(n.bytes == stored.bytes, "{}", p.decl.name);
            dense += 1;
        }
        assert_eq!(dense, 146);
    }
}
