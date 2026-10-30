//! Which architecture a file holds, and the only place in this crate that reads
//! `general.architecture`.
//!
//! Variant names are the GGUF strings themselves, not abbreviations: a single
//! `grep deepseek41` has to find the module, the metadata keys, the tool profile
//! and the gates at once.

use crate::ModelError;
use crate::placement::{ModelTensors, PlacementError};
use gguf::{Gguf, Split, Value};
use models::{ChatSpec, ModelSpec, ReasoningFormat, ToolFormat};

mod classify;
pub mod coverage;
pub mod deepseek2;
pub mod deepseek41;
pub mod dflash2;
pub mod draft;
pub mod dspark;
pub mod flint;
pub mod gemma_embedding;
pub mod gemma_embedding2;
mod gemma_encoder;
pub mod glm5next;
pub mod hybrid;
pub mod kda;
pub mod lfm3;
pub mod llama;
pub mod mimo2;
pub mod minimaxm3;
pub mod mistrallarge4;
pub mod musespark;
pub mod names;
pub mod nemotron_h;
pub mod nemotron_h_moe;
pub mod noaux;
pub mod place_hybrid;
pub mod qwen35moe;
pub mod qwen3moe;
pub mod raon1;
pub mod registry;
pub mod solar2;
mod stems;

/// The typed model description, its readers' output: re-exported so a
/// caller names one crate for the model and its description.
pub use models;

/// `split`'s [`ModelSpec`] by its family's reader ([`registry::ROWS`]), with every tensor's role:
/// what the coverage check ([`coverage::check`]) and the programs read. An
/// architecture no reader here reads is [`ModelError::UnknownArchitecture`]
/// with the string the file held. `deepseek2` has no reader.
pub fn spec(split: &Split) -> Result<Read, ModelError> {
    match reader_of(split) {
        Some(read) => read(split).map_err(ModelError::from),
        None => Err(ModelError::UnknownArchitecture(
            split.architecture().unwrap_or("<missing>").to_string(),
        )),
    }
}

/// [`spec`] for a caller whose refusals are the placement layer's
/// ([`PlacementError`]): the reader's own error as it stands, and an
/// architecture no reader here reads a metadata refusal naming
/// `general.architecture`.
pub fn spec_placed(split: &Split) -> Result<Read, PlacementError> {
    match reader_of(split) {
        Some(read) => read(split),
        None => Err(architecture_is(&format!(
            "{:?}; no reader here reads it",
            split.architecture().unwrap_or("<missing>")
        ))),
    }
}

/// The reader of `split`'s family ([`registry::ROWS`]); `None`: no reader.
fn reader_of(split: &Split) -> Option<registry::Reader> {
    split
        .architecture()
        .and_then(registry::row_of)
        .and_then(|row| row.read)
}

/// A reader's output: the description, the roles, and the keys it took a
/// default for (each with its value and the line that sets it).
#[derive(Debug)]
pub struct Read {
    pub spec: ModelSpec,
    pub tensors: ModelTensors,
    pub defaults: Vec<String>,
}

/// The architecture string of the DSpark draft file ([`dspark`]). It is not an
/// [`Arch`]: the draft is read beside a V4.1 model, never run as a model.
pub const DFLASH: &str = "dflash";

/// EmbeddingGemma's architecture string, read by [`gemma_embedding`].
pub const GEMMA_EMBEDDING: &str = "gemma-embedding";

/// EmbeddingGemma 2's architecture string, read by the [`gemma_embedding2`]
/// module: [`spec`], [`Arch::from_name`] and [`Arch::name`] take this one
/// spelling.
pub const GEMMA_EMBEDDING2: &str = "gemma-embedding2";

/// Whether `split` declares the draft's architecture.
pub fn is_dflash(split: &Split) -> bool {
    split.architecture() == Some(DFLASH)
}

/// The architecture string of a DeepSeek-V4-Flash file, which the
/// [`deepseek41`] module reads as its [`deepseek41::hparams::Model::Deepseek4`].
pub const DEEPSEEK4: &str = "deepseek4";

/// llama.cpp's spelling of GLM-5.3-Flash's architecture, which unsloth's
/// current shards carry: the same model and keys as ik's `glm5next`, read by
/// the [`glm5next`] module under either string.
pub const GLM5_NEXT_LLAMA_CPP: &str = "glm5-next";

/// Which of the [`deepseek41`] module's models `split` holds, by its
/// `general.architecture`; any other string is an error naming it.
pub fn deepseek41_model(split: &Split) -> Result<deepseek41::hparams::Model, PlacementError> {
    use deepseek41::hparams::Model;
    match split.architecture() {
        Some("deepseek41") => Ok(Model::Deepseek41),
        Some(DEEPSEEK4) => Ok(Model::Deepseek4),
        other => Err(PlacementError::Metadata {
            key: "general.architecture".to_string(),
            detail: format!(
                "is {:?}; the deepseek41 module reads deepseek41 and {DEEPSEEK4}",
                other.unwrap_or("<missing>")
            ),
        }),
    }
}

/// The `general.architecture` strings whose body is the Qwen3.5 dense trunk
/// ([`qwen35moe::hparams::Variant::Qwen35`]): `qwen35`, and `clef`, llama.cpp's
/// layout of the same trunk with a decision head's tensors beside it (the head
/// is read by `crates/decision`, never by the body). The metadata keys of the
/// body are read under the file's own architecture string.
pub const QWEN35_BODY: &[&str] = &["qwen35", "clef"];

/// Whether a file of architecture `arch` has the Qwen3.5 dense body
/// ([`QWEN35_BODY`]).
#[must_use]
pub fn is_qwen35_body(arch: &str) -> bool {
    QWEN35_BODY.contains(&arch)
}

/// Which of the [`qwen35moe`] module's variants `split` holds, by its
/// `general.architecture`; any other string is an error naming it.
pub fn qwen35moe_variant(split: &Split) -> Result<qwen35moe::hparams::Variant, PlacementError> {
    use qwen35moe::hparams::Variant;
    match split.architecture() {
        Some("qwen35moe") => Ok(Variant::Qwen35Moe),
        Some(a) if is_qwen35_body(a) => Ok(Variant::Qwen35),
        Some("qwen4exp") => Ok(Variant::Qwen4Exp),
        other => Err(PlacementError::Metadata {
            key: "general.architecture".to_string(),
            detail: format!(
                "is {:?}; the qwen35moe module reads qwen35moe, qwen4exp and the Qwen3.5 dense \
                 bodies {QWEN35_BODY:?}",
                other.unwrap_or("<missing>")
            ),
        }),
    }
}

/// The error naming `general.architecture`, which `is` completes: its value
/// and why a reader refuses it there.
fn architecture_is(is: &str) -> PlacementError {
    PlacementError::Metadata {
        key: "general.architecture".to_string(),
        detail: format!("is {is}"),
    }
}

/// An architecture a file's `general.architecture` names, when this crate has
/// a reader or a program for it. Flint has a reader and no executable program;
/// `muse-spark` and `mistral-large4` have a reader and no program.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arch {
    Deepseek2,
    Deepseek41,
    Qwen3moe,
    Qwen35moe,
    Glm5next,
    Mimo2,
    SolarOpen2,
    MiniMaxM3,
    Raon1,
    NemotronH,
    NemotronHMoe,
    GemmaEmbedding,
    GemmaEmbedding2,
    Flint,
    MuseSpark,
    /// Mistral Large 4: MLA with no query latent, SSMax, a lightning indexer
    /// over keys from the layer input, KV sharing and NoPE layers the file's
    /// arrays name, over softmax-routed experts with one shared. No
    /// executable program is registered.
    MistralLarge4,
}

impl Arch {
    /// The architecture a file declares. The error carries the string the file
    /// held, because that string is what the reader has to act on.
    pub fn detect(gguf: &Gguf) -> Result<Arch, ModelError> {
        let name = gguf
            .architecture()
            .ok_or_else(|| ModelError::UnknownArchitecture("<missing>".to_string()))?;
        Arch::from_name(name)
    }

    /// The string → variant step on its own, so it can be tested without a file.
    /// `deepseek4` (DeepSeek-V4-Flash) is read by the `deepseek41` module as
    /// one of its two models ([`deepseek41_model`]), and `glm5-next`
    /// ([`GLM5_NEXT_LLAMA_CPP`]) is `glm5next` under llama.cpp's spelling:
    /// the variants whose name is not the only string they stand for. The
    /// registry's rows hold the same strings (the registry's test).
    pub fn from_name(name: &str) -> Result<Arch, ModelError> {
        match name {
            "deepseek2" => Ok(Arch::Deepseek2),
            "deepseek41" | DEEPSEEK4 => Ok(Arch::Deepseek41),
            "qwen3moe" => Ok(Arch::Qwen3moe),
            "qwen35moe" => Ok(Arch::Qwen35moe),
            "glm5next" | GLM5_NEXT_LLAMA_CPP => Ok(Arch::Glm5next),
            "mimo2" => Ok(Arch::Mimo2),
            "solar-open2" => Ok(Arch::SolarOpen2),
            "minimax-m3" => Ok(Arch::MiniMaxM3),
            "raon1" => Ok(Arch::Raon1),
            "nemotron_h" => Ok(Arch::NemotronH),
            "nemotron_h_moe" => Ok(Arch::NemotronHMoe),
            GEMMA_EMBEDDING => Ok(Arch::GemmaEmbedding),
            GEMMA_EMBEDDING2 => Ok(Arch::GemmaEmbedding2),
            "flint" => Ok(Arch::Flint),
            "muse-spark" => Ok(Arch::MuseSpark),
            "mistral-large4" => Ok(Arch::MistralLarge4),
            other => Err(ModelError::UnknownArchitecture(other.to_string())),
        }
    }

    /// The module's name: the `general.architecture` string of its first
    /// model, which keys the reference-set families (`refset::arch::families`)
    /// and the oracle tables. A `deepseek4` file detects as
    /// [`Arch::Deepseek41`] and a `glm5-next` file as [`Arch::Glm5next`], so
    /// what names a file (a log line, an oracle directory) takes the file's
    /// own string (`Split::architecture`, or
    /// [`deepseek41::hparams::Model::name`]).
    pub fn name(&self) -> &'static str {
        match self {
            Arch::Deepseek2 => "deepseek2",
            Arch::Deepseek41 => "deepseek41",
            Arch::Qwen3moe => "qwen3moe",
            Arch::Qwen35moe => "qwen35moe",
            Arch::Glm5next => "glm5next",
            Arch::Mimo2 => "mimo2",
            Arch::SolarOpen2 => "solar-open2",
            Arch::MiniMaxM3 => "minimax-m3",
            Arch::Raon1 => "raon1",
            Arch::NemotronH => "nemotron_h",
            Arch::NemotronHMoe => "nemotron_h_moe",
            Arch::GemmaEmbedding => GEMMA_EMBEDDING,
            Arch::GemmaEmbedding2 => GEMMA_EMBEDDING2,
            Arch::Flint => "flint",
            Arch::MuseSpark => "muse-spark",
            Arch::MistralLarge4 => "mistral-large4",
        }
    }
}

// ------------------------------------------------------------ metadata keys
//
// The readers of `<architecture>.<suffix>` keys every architecture module
// shares: each returns the value or the error that names the key.

/// The token list ik counts when the file has no `vocab_size`.
const TOKENS: &str = "tokenizer.ggml.tokens";

/// The error that names `<architecture>.<suffix>`.
fn metadata(split: &Split, suffix: &str, detail: impl Into<String>) -> PlacementError {
    PlacementError::Metadata {
        key: split.arch_key(suffix),
        detail: detail.into(),
    }
}

/// `<architecture>.<suffix>` as an unsigned integer.
fn meta_u64(split: &Split, suffix: &str) -> Result<u64, PlacementError> {
    split
        .arch_get_u64(suffix)
        .ok_or_else(|| metadata(split, suffix, "is absent or not an unsigned integer"))
}

/// `meta_u64` for a count that indexes memory.
fn meta_usize(split: &Split, suffix: &str) -> Result<usize, PlacementError> {
    let v = meta_u64(split, suffix)?;
    usize::try_from(v).map_err(|_| metadata(split, suffix, format!("{v} does not fit usize")))
}

/// `<architecture>.<suffix>` as a count above 0.
pub(crate) fn positive(split: &Split, suffix: &str) -> Result<usize, PlacementError> {
    match meta_usize(split, suffix)? {
        0 => Err(metadata(split, suffix, "is 0")),
        v => Ok(v),
    }
}

/// `<architecture>.<suffix>` as a count, `None` when the file does not carry
/// the key; a key of another type is an error naming it.
pub(crate) fn optional_usize(split: &Split, suffix: &str) -> Result<Option<usize>, PlacementError> {
    match split.value(&split.arch_key(suffix)) {
        None => Ok(None),
        Some(_) => meta_usize(split, suffix).map(Some),
    }
}

/// `<architecture>.<suffix>` as a float.
fn meta_f32(split: &Split, suffix: &str) -> Result<f32, PlacementError> {
    split
        .arch_get_f32(suffix)
        .ok_or_else(|| metadata(split, suffix, "is absent or not a float"))
}

/// `<architecture>.<suffix>` as a float, `None` when the file does not carry the key; a key of
/// another type is an error naming it.
pub(crate) fn optional_f32(split: &Split, suffix: &str) -> Result<Option<f32>, PlacementError> {
    match split.value(&split.arch_key(suffix)) {
        None => Ok(None),
        Some(_) => meta_f32(split, suffix).map(Some),
    }
}

/// `<architecture>.<suffix>` as a finite positive float.
pub(crate) fn finite_positive(split: &Split, suffix: &str) -> Result<f32, PlacementError> {
    let v = meta_f32(split, suffix)?;
    if !v.is_finite() || v <= 0.0 {
        return Err(metadata(
            split,
            suffix,
            format!("is {v}, not a finite positive value"),
        ));
    }
    Ok(v)
}

/// Layer `l` has a positive key-head count that divides the query-head count.
pub(crate) fn check_kv(
    split: &Split,
    n_head: usize,
    l: usize,
    heads: usize,
) -> Result<(), PlacementError> {
    if heads == 0 || !n_head.is_multiple_of(heads) {
        return Err(metadata(
            split,
            "attention.head_count_kv",
            format!("is {heads} at layer {l} for {n_head} query heads"),
        ));
    }
    Ok(())
}

/// `<architecture>.<suffix>` as a string.
fn meta_str<'a>(split: &'a Split, suffix: &str) -> Result<&'a str, PlacementError> {
    split
        .arch_get_str(suffix)
        .ok_or_else(|| metadata(split, suffix, "is absent or not a string"))
}

/// `<architecture>.<suffix>` as a bool.
fn meta_bool(split: &Split, suffix: &str) -> Result<bool, PlacementError> {
    split
        .value(&split.arch_key(suffix))
        .and_then(Value::as_bool)
        .ok_or_else(|| metadata(split, suffix, "is absent or not a bool"))
}

/// `<architecture>.<suffix>` as an array's items.
fn meta_arr<'a>(split: &'a Split, suffix: &str) -> Result<&'a [Value], PlacementError> {
    split
        .arch_get_arr(suffix)
        .ok_or_else(|| metadata(split, suffix, "is absent or not an array"))
}

/// `<architecture>.<suffix>` as an array of exactly `n_layer` items.
/// Non-arrays and other lengths are refused by name; counts that also
/// accept a shared scalar use [`per_layer_count`].
fn per_layer_items<'a>(
    split: &'a Split,
    suffix: &str,
    n_layer: usize,
) -> Result<&'a [Value], PlacementError> {
    let items = meta_arr(split, suffix)?;
    if items.len() != n_layer {
        return Err(metadata(
            split,
            suffix,
            format!("has {} values for {n_layer} layers", items.len()),
        ));
    }
    Ok(items)
}

/// `<architecture>.<suffix>` as exactly one bool per layer.
/// Non-arrays, other lengths and non-bool items are refused by name.
fn per_layer_bool(
    split: &Split,
    suffix: &str,
    n_layer: usize,
) -> Result<Vec<bool>, PlacementError> {
    per_layer_items(split, suffix, n_layer)?
        .iter()
        .enumerate()
        .map(|(l, v)| {
            v.as_bool()
                .ok_or_else(|| metadata(split, suffix, format!("has no bool at layer {l}")))
        })
        .collect()
}

/// `<architecture>.<suffix>` as one shared count expanded to every trunk
/// layer or an array of the trunk's counts: an array of `n_trunk` counts, or
/// of the file's `n_layer` counts (a next-token layer's entries close it),
/// the trunk's taken. Other array lengths are refused by name. Zero is a
/// count; a family that refuses it checks that itself.
pub(crate) fn per_layer_count_trunk(
    split: &Split,
    suffix: &str,
    n_trunk: usize,
    n_layer: usize,
) -> Result<Vec<usize>, PlacementError> {
    if !matches!(split.value(&split.arch_key(suffix)), Some(Value::Array(_))) {
        return per_layer_count(split, suffix, n_trunk);
    }
    let items = meta_arr(split, suffix)?;
    if items.len() != n_trunk && items.len() != n_layer {
        return Err(metadata(
            split,
            suffix,
            format!("has {} values for {n_trunk} trunk layers", items.len()),
        ));
    }
    items
        .iter()
        .take(n_trunk)
        .enumerate()
        .map(|(l, v)| {
            v.as_unsigned()
                .and_then(|x| usize::try_from(x).ok())
                .ok_or_else(|| metadata(split, suffix, format!("has no count at layer {l}")))
        })
        .collect()
}

/// `<architecture>.<suffix>` as one shared count expanded to every layer
/// or an array of exactly `n_layer` counts. Other array lengths are refused
/// by name. Zero is a count; a family that refuses it checks that itself.
pub(crate) fn per_layer_count(
    split: &Split,
    suffix: &str,
    n_layer: usize,
) -> Result<Vec<usize>, PlacementError> {
    // PIN(2026-10-15): llama.cpp's LLAMA_MAX_LAYERS bounds per-layer count allocations.
    const MAX_LAYERS: usize = 512;
    if n_layer > MAX_LAYERS {
        return Err(metadata(
            split,
            "block_count",
            format!("is {n_layer}, above the layer limit {MAX_LAYERS}"),
        ));
    }
    if !matches!(split.value(&split.arch_key(suffix)), Some(Value::Array(_))) {
        return Ok(vec![meta_usize(split, suffix)?; n_layer]);
    }
    per_layer_items(split, suffix, n_layer)?
        .iter()
        .enumerate()
        .map(|(l, v)| {
            v.as_unsigned()
                .and_then(|x| usize::try_from(x).ok())
                .ok_or_else(|| metadata(split, suffix, format!("has no count at layer {l}")))
        })
        .collect()
}

/// The architecture keys of `split` whose suffix `read` does not list, and
/// its `general.sampling.*` keys: what a family's reader leaves unread,
/// listed for its meta gate to pin, not refused.
pub(crate) fn unread_keys_of(split: &Split, read: &[&str]) -> Vec<String> {
    let prefix = split.arch_key("");
    split
        .iter_kv()
        .map(|(k, _)| k)
        .filter(|k| match k.strip_prefix(prefix.as_str()) {
            Some(suffix) => !read.contains(&suffix),
            None => k.starts_with("general.sampling."),
        })
        .map(str::to_string)
        .collect()
}

/// `expert_gating_func`'s sigmoid router: ik's `LLM_EXPERT_GATING_FUNC_SIGMOID`
/// and llama.cpp-mainline's `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`
/// (src/llama-hparams.h:19), one id.
const SIGMOID: u64 = 2;

/// `expert_gating_func`, held to the sigmoid router, the one the sigmoid-router
/// families' bodies run: another value is refused by name. An absent key is
/// the sigmoid router where the family's loader defaults it — `absent` the
/// line that says where, recorded in `defaults` — and refused by name where
/// `absent` is `None`.
pub(crate) fn sigmoid_gating(
    split: &Split,
    absent: Option<&str>,
    defaults: &mut Vec<String>,
) -> Result<(), PlacementError> {
    let key = "expert_gating_func";
    let gating = match (split.value(&split.arch_key(key)), absent) {
        (None, Some(line)) => {
            defaults.push(format!("{key} = sigmoid ({line})"));
            SIGMOID
        }
        _ => meta_u64(split, key)?,
    };
    if gating != SIGMOID {
        return Err(metadata(
            split,
            key,
            format!("is {gating}; this reader reads the sigmoid router ({SIGMOID})"),
        ));
    }
    Ok(())
}

/// The rope scaling keys, held to the plain rope: `rope.scaling.type` absent or `none`, and
/// `rope.scaling.factor` absent, 0 or 1. Another value is refused by name.
pub(crate) fn plain_rope(split: &Split) -> Result<(), PlacementError> {
    let scaling = "rope.scaling.type";
    if split.value(&split.arch_key(scaling)).is_some() {
        let kind = meta_str(split, scaling)?;
        if kind != "none" {
            return Err(metadata(
                split,
                scaling,
                format!("is {kind:?}; this chain runs the plain rope, no scaling"),
            ));
        }
    }
    let factor = "rope.scaling.factor";
    if split.value(&split.arch_key(factor)).is_some() {
        let f = meta_f32(split, factor)?;
        if f != 0.0 && f != 1.0 {
            return Err(metadata(
                split,
                factor,
                format!("is {f}; this chain runs the rope at frequency scale 1"),
            ));
        }
    }
    Ok(())
}

/// `expert_count` and `expert_used_count`: both above 0, the used count within the routed count.
pub(crate) fn expert_counts(split: &Split) -> Result<(usize, usize), PlacementError> {
    let n_expert = meta_usize(split, "expert_count")?;
    let n_used = meta_usize(split, "expert_used_count")?;
    if n_expert == 0 || n_used == 0 || n_used > n_expert {
        return Err(metadata(
            split,
            "expert_used_count",
            format!("is {n_used} of expert_count {n_expert}"),
        ));
    }
    Ok((n_expert, n_used))
}

/// `expert_group_count` and `expert_group_used_count`: absent or 1. The group
/// stage runs only past one group (llama-graph.cpp `build_moe_ffn`'s
/// `n_expert_groups > 1`), which `router` — the router that refuses it, as
/// the refusal names it — does not run; any other count is refused by name.
pub(crate) fn one_expert_group(split: &Split, router: &str) -> Result<(), PlacementError> {
    for key in ["expert_group_count", "expert_group_used_count"] {
        if let Some(v) = optional_usize(split, key)?
            && v != 1
        {
            return Err(metadata(
                split,
                key,
                format!("is {v}; {router} takes one group"),
            ));
        }
    }
    Ok(())
}

/// Required tensors at their expected dimensions; missing names precede dimension errors.
fn tensors_agree(split: &Split, want: Vec<(String, Vec<u64>)>) -> Result<(), PlacementError> {
    let mut missing = Vec::new();
    let mut off = Vec::new();
    for (name, dims) in want {
        match split.find(&name) {
            None => missing.push(name),
            Some((_, info)) if info.dims != dims => {
                let detail = format!("{name} {:?} (want {dims:?})", info.dims);
                off.push((name, detail));
            }
            Some(_) => {}
        }
    }
    if let Some(name) = missing.first() {
        return Err(PlacementError::Tensor {
            name: name.clone(),
            detail: format!(
                "is not in the file ({} tensor(s) the model needs are missing: {})",
                missing.len(),
                missing.join(", ")
            ),
        });
    }
    if let Some((name, _)) = off.first() {
        return Err(PlacementError::Tensor {
            name: name.clone(),
            detail: format!(
                "has other dims than the keys give the tensor ({} such: {})",
                off.len(),
                off.iter()
                    .map(|(_, d)| d.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        });
    }
    Ok(())
}

/// The tensors whose dims the keys name hold exactly those dims: the three
/// model-level tensors and each layer's `layers` entries, ggml order. A
/// tensor the file does not carry is skipped: `layers_agree` already named a
/// required one. Every tensor of other dims, the first by name.
pub(crate) fn dims_agree(
    split: &Split,
    hidden: usize,
    vocab: usize,
    layers: impl Iterator<Item = Vec<(&'static str, Vec<u64>)>>,
) -> Result<(), PlacementError> {
    let (e, v) = (hidden as u64, vocab as u64);
    let mut want = vec![
        (names::token_embd(), vec![e, v]),
        (names::output(), vec![e, v]),
        (names::output_norm(), vec![e]),
    ];
    for (l, layer) in layers.enumerate() {
        want.extend(
            layer
                .into_iter()
                .map(|(stem, dims)| (format!("blk.{l}.{stem}"), dims)),
        );
    }
    want.retain(|(name, _)| split.find(name).is_some());
    tensors_agree(split, want)
}

/// The verdict of a tensor-set check: the first name the layers need and the file lacks, with how
/// many more there are; else the first name the file carries where no layer's block does. `need`
/// completes "the … tensor(s) … need are missing", `none_of` the sentence "is in the file, and …".
pub(crate) fn refuse_tensor_set(
    missing: &[String],
    misplaced: &[String],
    need: &str,
    none_of: &str,
) -> Result<(), PlacementError> {
    if let Some(name) = missing.first() {
        return Err(PlacementError::Tensor {
            name: name.clone(),
            detail: format!(
                "is not in the file ({} tensor(s) {need} are missing: {})",
                missing.len(),
                missing.join(", ")
            ),
        });
    }
    if let Some(name) = misplaced.first() {
        return Err(PlacementError::Tensor {
            name: name.clone(),
            detail: format!(
                "is in the file, and {none_of} ({} such: {})",
                misplaced.len(),
                misplaced.join(", ")
            ),
        });
    }
    Ok(())
}

/// `a * b`, or the error that names `<architecture>.<suffix>`.
fn mul(split: &Split, suffix: &str, a: usize, b: usize) -> Result<usize, PlacementError> {
    a.checked_mul(b)
        .ok_or_else(|| metadata(split, suffix, format!("{a} * {b} does not fit")))
}

/// What `nextn_predict_layers` settles about the blocks past the trunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NextnLayers {
    /// The blocks of `block_count` that are no trunk layer: the trunk is
    /// `block_count` less this. The declared count when the file carries the
    /// layers or stripped them (their blocks hold no tensor), else 0.
    past_trunk: usize,
    /// The next-token layers the file carries tensors for: the declared count
    /// only where `nextn.eh_proj` backs it, else 0. A reader of next-token
    /// tensors takes this, never `past_trunk`.
    carried: usize,
}

impl NextnLayers {
    /// No block past the trunk.
    const NONE: NextnLayers = NextnLayers {
        past_trunk: 0,
        carried: 0,
    };
}

/// One block's tensors, counted for [`nextn_layers`].
#[derive(Clone, Copy, Default)]
struct BlockTensors {
    /// Tensors named `blk.<this block>.*`.
    all: usize,
    /// Of them, the `nextn.*` ones.
    nextn: usize,
}

/// The tensors of each block `first..end`, in one pass over the file's names.
fn block_tensors(split: &Split, first: usize, end: usize) -> Vec<BlockTensors> {
    let mut blocks = vec![BlockTensors::default(); end - first];
    for (_, t) in split.iter_tensors() {
        let Some((block, stem)) = t
            .name
            .strip_prefix("blk.")
            .and_then(|rest| rest.split_once('.'))
        else {
            continue;
        };
        let Some(b) = block
            .parse::<usize>()
            .ok()
            .and_then(|l| l.checked_sub(first))
            .and_then(|i| blocks.get_mut(i))
        else {
            continue;
        };
        b.all += 1;
        b.nextn += usize::from(stem.starts_with("nextn."));
    }
    blocks
}

/// The blocks past the trunk and the next-token layers a file carries, from
/// `nextn_predict_layers` (n) and the tensors of the blocks `n_layer - n ..
/// n_layer`. Every default taken or shape read is recorded in `defaults`: an
/// absent key (`absent_note`, the line that defaults it; empty when the
/// family's dialect reads no next-token layer, where the absence is no
/// default taken). Where the first of the blocks carries `nextn.eh_proj`, the
/// count is believed. Where it does not, the blocks are read by their
/// tensors:
/// - none carries any tensor: a stripped next-token layer, the trunk is
///   `n_layer - n` and none is carried;
/// - each carries trunk tensors and none a `nextn.*` one: they are trunk
///   layers under a kept key, the count is 0 (`probe_note`, the line that
///   probes it, or empty);
/// - any other mix is refused by name, listing each block's tensors.
///
/// A count that leaves no trunk layer is refused by name.
fn nextn_layers(
    split: &Split,
    n_layer: usize,
    absent_note: &str,
    probe_note: &str,
    defaults: &mut Vec<String>,
) -> Result<NextnLayers, PlacementError> {
    let key = "nextn_predict_layers";
    let n = match split.value(&split.arch_key(key)) {
        None => {
            if !absent_note.is_empty() {
                defaults.push(format!("{key} = 0 ({absent_note})"));
            }
            return Ok(NextnLayers::NONE);
        }
        Some(_) => meta_usize(split, key)?,
    };
    if n >= n_layer {
        return Err(metadata(
            split,
            key,
            format!("is {n}, not below block_count {n_layer}"),
        ));
    }
    if n == 0 {
        return Ok(NextnLayers::NONE);
    }
    let first = n_layer - n;
    let probe = format!("blk.{first}.nextn.eh_proj.weight");
    if split.find(&probe).is_some() {
        return Ok(NextnLayers {
            past_trunk: n,
            carried: n,
        });
    }
    let blocks = block_tensors(split, first, n_layer);
    if blocks.iter().all(|b| b.all == 0) {
        let named = if n == 1 {
            format!("blk.{first} carries")
        } else {
            format!("blk.{first}..blk.{} carry", n_layer - 1)
        };
        defaults.push(format!(
            "{key} = {n} ({named} no tensor: a stripped next-token layer; the trunk is {first})"
        ));
        return Ok(NextnLayers {
            past_trunk: n,
            carried: 0,
        });
    }
    if blocks.iter().all(|b| b.all > 0 && b.nextn == 0) {
        let note = if probe_note.is_empty() {
            String::new()
        } else {
            format!(", {probe_note}")
        };
        defaults.push(format!("{key} = 0 ({probe} is not in the file{note})"));
        return Ok(NextnLayers::NONE);
    }
    let states: Vec<String> = (first..n_layer)
        .zip(&blocks)
        .map(|(l, b)| format!("blk.{l}: {} tensors, {} of them nextn.*", b.all, b.nextn))
        .collect();
    Err(metadata(
        split,
        key,
        format!(
            "is {n} and {probe} is not in the file, and its blocks are neither all empty (a \
             stripped next-token layer) nor all trunk layers without nextn.* tensors: {}",
            states.join("; ")
        ),
    ))
}

/// `tokenizer.ggml.pre`.
const PRE: &str = "tokenizer.ggml.pre";
/// `tokenizer.ggml.model`.
const TOKENIZER_MODEL: &str = "tokenizer.ggml.model";
/// `tokenizer.chat_template`.
const TEMPLATE: &str = "tokenizer.chat_template";
/// `tokenizer.ggml.model` of a SentencePiece file, and the pre-tokenizer rule
/// that tokenizer selects, whatever `tokenizer.ggml.pre` says (llama.cpp's
/// SentencePiece branch takes no BPE pre-tokenizer from it).
const SPM: &str = "llama";

/// Whether `split`'s tokenizer is SentencePiece.
fn is_spm(split: &Split) -> bool {
    split.value(TOKENIZER_MODEL).and_then(Value::as_str) == Some(SPM)
}

/// The file's chat surface: its pre-tokenizer and template, with the family's
/// parsers. A SentencePiece file's pre-tokenizer is [`SPM`] whatever
/// `tokenizer.ggml.pre` says ([`spm_pre_default`] names the rewrite); any other
/// file names its own, and is refused without one.
fn chat_of(
    split: &Split,
    tools: Option<ToolFormat>,
    reasoning: Option<ReasoningFormat>,
) -> Result<ChatSpec, PlacementError> {
    let pre = if is_spm(split) {
        if let Some(v) = split.value(PRE)
            && v.as_str().is_none()
        {
            return Err(PlacementError::Metadata {
                key: PRE.to_string(),
                detail: "is not a string".to_string(),
            });
        }
        SPM
    } else {
        split
            .value(PRE)
            .and_then(Value::as_str)
            .ok_or_else(|| PlacementError::Metadata {
                key: PRE.to_string(),
                detail: "is absent or not a string".to_string(),
            })?
    };
    Ok(ChatSpec {
        pre: pre.to_string(),
        template: template_of(split)?,
        tools,
        reasoning,
    })
}

/// [`chat_of`] for an architecture string several model families share: the
/// tool-call and reasoning parsers are the ones the file's own template
/// teaches ([`ToolFormat::of_template`], [`ReasoningFormat::of_template`]), and
/// `None` for a file with no template or one whose markup no parser here reads.
fn chat_from_template(split: &Split) -> Result<ChatSpec, PlacementError> {
    let mut chat = chat_of(split, None, None)?;
    if let Some(template) = &chat.template {
        chat.tools = ToolFormat::of_template(template);
        chat.reasoning = ReasoningFormat::of_template(template);
    }
    Ok(chat)
}

/// The line a reader records in its defaults when [`chat_of`] took a
/// SentencePiece file's pre-tokenizer from the tokenizer model: the file's own
/// `tokenizer.ggml.pre` is absent or another name. `None` for any other file,
/// and for one that already says [`SPM`].
fn spm_pre_default(split: &Split) -> Option<String> {
    if !is_spm(split) {
        return None;
    }
    let file = split.value(PRE).and_then(Value::as_str);
    (file != Some(SPM))
        .then(|| format!("{PRE} = {SPM} ({TOKENIZER_MODEL} {SPM}; the file says {file:?})"))
}

/// `tokenizer.chat_template`; `None` when the file has none.
fn template_of(split: &Split) -> Result<Option<String>, PlacementError> {
    split
        .value(TEMPLATE)
        .map(|v| {
            v.as_str()
                .map(str::to_string)
                .ok_or_else(|| PlacementError::Metadata {
                    key: TEMPLATE.to_string(),
                    detail: "is not a string".to_string(),
                })
        })
        .transpose()
}

/// `v`, a count a reader holds as `usize`, as the `u32` a [`ModelSpec`]
/// holds; `what` names it in the error.
fn spec_u32(what: &str, v: usize) -> Result<u32, PlacementError> {
    u32::try_from(v).map_err(|_| PlacementError::Metadata {
        key: what.to_string(),
        detail: format!("{v} does not fit u32"),
    })
}

/// The vocabulary size where ik takes it (llama-hparams.cpp:155):
/// `vocab_size` when the file carries it, else the token list's length.
fn n_vocab(split: &Split) -> Result<usize, PlacementError> {
    let key = "vocab_size";
    if split.value(&split.arch_key(key)).is_some() {
        return meta_usize(split, key);
    }
    match split.value(TOKENS) {
        Some(Value::Array(tokens)) => Ok(tokens.len()),
        _ => Err(PlacementError::Metadata {
            key: TOKENS.to_string(),
            detail: format!(
                "is absent or not an array, and so is {}",
                split.arch_key(key)
            ),
        }),
    }
}

/// Header-only GGUF files for the architecture modules' unit tests: the
/// refusals of a hyperparameter reader are tested against a file, through the
/// same `Split` the engine reads, not against a second table type.
#[cfg(test)]
pub(crate) mod synthetic {
    use std::path::PathBuf;

    /// A metadata value, as the GGUF type it is written as.
    pub(crate) enum V {
        U32(u32),
        /// A scalar U64. A count above `u32::MAX` is how an overflowing
        /// projection product is written without a tensor large enough to
        /// fail the file's own byte-size check first.
        U64(u64),
        F32(f32),
        Str(&'static str),
        Bool(bool),
        /// An array of I32, the type a converter writes a per-layer count in.
        I32s(Vec<i32>),
        /// An array of F32.
        F32s(Vec<f32>),
        /// An array of U64, the type a converter writes a hash constant in.
        U64s(Vec<u64>),
        /// An array of bools, the type a converter writes a per-layer flag in.
        Bools(Vec<bool>),
    }

    fn string(b: &mut Vec<u8>, x: &str) {
        b.extend_from_slice(&(x.len() as u64).to_le_bytes());
        b.extend_from_slice(x.as_bytes());
    }

    /// A file declaring architecture `arch`, with a two-token vocabulary, the
    /// keys `kv` under the `<arch>.` prefix, and one 1-value F32 tensor per
    /// name of `tensors`; written to a temp path named after `tag`.
    pub(crate) fn header(tag: &str, arch: &str, kv: &[(&str, V)], tensors: &[String]) -> PathBuf {
        header_with(tag, arch, kv, &[], tensors)
    }

    /// [`header`] with the keys `global` as well, written as they are named
    /// (`tokenizer.ggml.pre`, `tokenizer.chat_template`).
    pub(crate) fn header_with(
        tag: &str,
        arch: &str,
        kv: &[(&str, V)],
        global: &[(&str, V)],
        tensors: &[String],
    ) -> PathBuf {
        let shaped: Vec<(String, Vec<u64>)> =
            tensors.iter().map(|n| (n.clone(), vec![1])).collect();
        header_shaped(tag, arch, kv, global, &shaped)
    }

    /// [`header_with`] with each tensor's dims: zero-filled F32 data, each
    /// tensor's start aligned to 32 bytes.
    pub(crate) fn header_shaped(
        tag: &str,
        arch: &str,
        kv: &[(&str, V)],
        global: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
    ) -> PathBuf {
        let typed: Vec<(String, Vec<u64>, Ty)> = tensors
            .iter()
            .map(|(n, d)| (n.clone(), d.clone(), Ty::F32))
            .collect();
        header_typed(tag, arch, kv, global, &typed)
    }

    /// A tensor type the synthetic headers write.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Ty {
        F32,
        /// Two bytes a value.
        F16,
        /// Two bytes a value.
        Bf16,
        /// Blocks of 32 values in 34 bytes; the first dim a multiple of 32.
        Q8_0,
        /// Blocks of 256 values in 210 bytes; any first dim (a partial final
        /// block is a header-only fixture's).
        Q6K,
        /// Blocks of 256 values in 144 bytes; any first dim.
        Q4K,
        /// NVFP4: blocks of 64 values in 36 bytes; the first dim a multiple of 64.
        Nvfp4,
        /// IQ2_XXS: blocks of 256 values in 66 bytes; the first dim a multiple of
        /// 256.
        Iq2Xxs,
        /// bloomery's F8_E4M3 rows: one byte a value, scaled by `<stem>.scale`.
        F8Rows,
    }

    impl Ty {
        /// The ggml type id and the bytes of `n` values.
        fn id_and_bytes(self, n: u64) -> (u32, u64) {
            match self {
                Ty::F32 => (0, 4 * n),
                Ty::F16 => (1, 2 * n),
                Ty::Bf16 => (30, 2 * n),
                Ty::Q8_0 => {
                    assert!(n.is_multiple_of(32), "{n} values are not whole Q8_0 blocks");
                    (8, n / 32 * 34)
                }
                Ty::Q6K => (14, n.div_ceil(256) * 210),
                Ty::Q4K => (12, n.div_ceil(256) * 144),
                Ty::Nvfp4 => {
                    assert!(
                        n.is_multiple_of(64),
                        "{n} values are not whole NVFP4 blocks"
                    );
                    (40, n / 64 * 36)
                }
                Ty::Iq2Xxs => {
                    assert!(
                        n.is_multiple_of(256),
                        "{n} values are not whole IQ2_XXS blocks"
                    );
                    (16, n / 256 * 66)
                }
                Ty::F8Rows => (gguf::GgmlType::F8E4M3_ROWS.as_u32(), n),
            }
        }
    }

    /// [`header_shaped`] with each tensor's type.
    pub(crate) fn header_typed(
        tag: &str,
        arch: &str,
        kv: &[(&str, V)],
        global: &[(&str, V)],
        tensors: &[(String, Vec<u64>, Ty)],
    ) -> PathBuf {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&((kv.len() + global.len()) as u64 + 2).to_le_bytes());
        string(&mut b, "general.architecture");
        b.extend_from_slice(&8u32.to_le_bytes());
        string(&mut b, arch);
        string(&mut b, super::TOKENS);
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&2u64.to_le_bytes());
        string(&mut b, "a");
        string(&mut b, "b");
        let keys = kv
            .iter()
            .map(|(k, v)| (format!("{arch}.{k}"), v))
            .chain(global.iter().map(|(k, v)| ((*k).to_string(), v)));
        for (k, v) in keys {
            string(&mut b, &k);
            match v {
                V::U32(x) => {
                    b.extend_from_slice(&4u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                V::U64(x) => {
                    b.extend_from_slice(&10u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                V::F32(x) => {
                    b.extend_from_slice(&6u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                V::Str(x) => {
                    b.extend_from_slice(&8u32.to_le_bytes());
                    string(&mut b, x);
                }
                V::Bool(x) => {
                    b.extend_from_slice(&7u32.to_le_bytes());
                    b.push(u8::from(*x));
                }
                V::I32s(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&5u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for x in xs.iter() {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                }
                V::F32s(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&6u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for x in xs.iter() {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                }
                V::U64s(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&10u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for x in xs.iter() {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                }
                V::Bools(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&7u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for x in xs.iter() {
                        b.push(u8::from(*x));
                    }
                }
            }
        }
        let mut offset = 0u64;
        for (name, dims, ty) in tensors {
            let (id, bytes) = ty.id_and_bytes(dims.iter().product::<u64>());
            string(&mut b, name);
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims.iter() {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&id.to_le_bytes());
            b.extend_from_slice(&offset.to_le_bytes());
            offset += bytes.div_ceil(32) * 32;
        }
        b.resize(b.len().div_ceil(32) * 32 + offset as usize, 0);
        let path = std::env::temp_dir().join(format!("bloomery-{}-{tag}.gguf", std::process::id()));
        std::fs::write(&path, b).expect("write the synthetic header");
        path
    }

    /// What `read` makes of a shaped header of architecture `arch` with the keys `kv` and
    /// `global`: its value, or its error as text. The file is removed.
    pub(crate) fn verdict<T>(
        tag: &str,
        arch: &str,
        kv: &[(&str, V)],
        global: &[(&str, V)],
        tensors: &[(String, Vec<u64>)],
        read: impl FnOnce(&gguf::Split) -> Result<T, crate::placement::PlacementError>,
    ) -> Result<T, String> {
        let path = header_shaped(tag, arch, kv, global, tensors);
        let got = gguf::Split::open(&path)
            .map_err(|e| e.to_string())
            .and_then(|split| read(&split).map_err(|e| e.to_string()));
        let _ = std::fs::remove_file(&path);
        got
    }

    /// `model`'s routed stacks as rows a host kernel runs: Q8_0, each row widened to whole
    /// 32-value blocks, its bytes that type's. A shaped header's routed stacks are F32 rows of a
    /// few values, which no host kernel runs, so a plan that holds any expert on the host
    /// refuses them by name.
    pub(crate) fn host_run_experts(model: &mut crate::placement::ModelTensors) {
        for t in &mut model.tensors {
            if t.role == crate::placement::Role::RoutedExperts {
                t.ty = gguf::GgmlType::Q8_0;
                t.dims[0] = t.dims[0].next_multiple_of(32);
                t.file_bytes = t.dims.iter().product::<u64>() / 32 * 34;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::synthetic::{V, header_shaped};
    use super::{Arch, NextnLayers};

    #[test]
    fn scalar_layer_count_refuses_more_than_llama_max_layers() {
        for n_layer in [512_u64, 513, 1 << 61] {
            let path = header_shaped(
                "scalar-layer-limit",
                "gemma-embedding2",
                &[
                    ("block_count", V::U64(n_layer)),
                    ("attention.head_count_kv", V::U32(1)),
                ],
                &[],
                &[],
            );
            let split = gguf::Split::open(&path).unwrap();
            let n_layer = super::positive(&split, "block_count").unwrap();
            let result = super::per_layer_count(&split, "attention.head_count_kv", n_layer);
            drop(split);
            std::fs::remove_file(&path).expect("remove the synthetic header");
            if n_layer == 512 {
                assert_eq!(result.expect("the layer limit is inclusive"), vec![1; 512]);
            } else {
                let error =
                    result.expect_err("a scalar count cannot allocate more than 512 layers");
                assert!(
                    matches!(error, crate::placement::PlacementError::Metadata { key, detail }
                    if key == "gemma-embedding2.block_count" && detail.contains("512"))
                );
            }
        }
    }

    #[test]
    fn known_names_round_trip() {
        for a in [
            Arch::Deepseek2,
            Arch::Deepseek41,
            Arch::Qwen3moe,
            Arch::Qwen35moe,
            Arch::Glm5next,
            Arch::Mimo2,
            Arch::SolarOpen2,
            Arch::MiniMaxM3,
            Arch::Raon1,
            Arch::NemotronH,
            Arch::NemotronHMoe,
            Arch::GemmaEmbedding,
            Arch::GemmaEmbedding2,
            Arch::Flint,
            Arch::MuseSpark,
            Arch::MistralLarge4,
        ] {
            assert_eq!(Arch::from_name(a.name()).unwrap(), a);
        }
    }

    /// A V4-Flash file is read by the deepseek41 module.
    #[test]
    fn deepseek4_is_read_by_the_deepseek41_module() {
        assert_eq!(Arch::from_name(super::DEEPSEEK4).unwrap(), Arch::Deepseek41);
    }

    /// A file that spells GLM-5.3-Flash's architecture as llama.cpp does is
    /// read by the glm5next module, which stays the name of the module.
    #[test]
    fn glm5_next_is_the_glm5next_architecture() {
        assert_eq!(
            Arch::from_name(super::GLM5_NEXT_LLAMA_CPP).unwrap(),
            Arch::Glm5next
        );
        assert_eq!(Arch::Glm5next.name(), "glm5next");
    }

    /// An architecture no reader reads is refused by its name before any key
    /// is read.
    #[test]
    fn an_unknown_architecture_has_no_reader() {
        let path = super::synthetic::header("unknown-spec", "glm6", &[], &[]);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let err = super::spec(&split).expect_err("no glm6 reader").to_string();
        let _ = std::fs::remove_file(&path);
        assert!(err.contains("\"glm6\""), "{err}");
    }

    /// A `glm5next` header goes to its reader, which refuses a file without
    /// keys by the first key it reads.
    #[test]
    fn a_glm5next_header_goes_to_its_reader() {
        let path = super::synthetic::header("glm5next-spec", "glm5next", &[], &[]);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let err = super::spec(&split)
            .expect_err("a header without keys")
            .to_string();
        let _ = std::fs::remove_file(&path);
        assert!(err.contains("glm5next.block_count"), "{err}");
    }

    /// A `glm5-next` header goes to the glm5next reader, which names its keys
    /// under the file's own prefix.
    #[test]
    fn a_glm5_next_header_goes_to_the_glm5next_reader() {
        let path = super::synthetic::header("glm5-next-spec", super::GLM5_NEXT_LLAMA_CPP, &[], &[]);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let err = super::spec(&split)
            .expect_err("a header without keys")
            .to_string();
        let _ = std::fs::remove_file(&path);
        assert!(err.contains("glm5-next.block_count"), "{err}");
    }

    /// A `qwen4exp` header goes to the qwen35moe reader, which names its keys
    /// under the file's own prefix.
    #[test]
    fn a_qwen4exp_header_goes_to_the_qwen35moe_reader() {
        let path = super::synthetic::header("qwen4exp-spec", "qwen4exp", &[], &[]);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let err = super::spec(&split)
            .expect_err("a header without keys")
            .to_string();
        let _ = std::fs::remove_file(&path);
        assert!(err.contains("qwen4exp.block_count"), "{err}");
    }

    /// A `mimo2` header goes to its reader, which refuses a file without
    /// keys by the first key it reads.
    #[test]
    fn a_mimo2_header_goes_to_its_reader() {
        let path = super::synthetic::header("mimo2-spec", "mimo2", &[], &[]);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let err = super::spec(&split)
            .expect_err("a header without keys")
            .to_string();
        let _ = std::fs::remove_file(&path);
        assert!(err.contains("mimo2.block_count"), "{err}");
    }

    #[test]
    fn unknown_name_is_reported_verbatim() {
        let err = Arch::from_name("llama").unwrap_err();
        assert!(
            err.to_string().contains("llama"),
            "the error must name the string the file held, got {err}"
        );
    }

    /// `nextn_layers` on a header of `n_layer` blocks declaring `declared`
    /// next-token layers and holding `tensors` (one value each): the result
    /// or its error's text, and the defaults recorded.
    fn nextn_of(
        tag: &str,
        n_layer: u32,
        declared: u32,
        tensors: &[&str],
        probe_note: &str,
    ) -> (Result<NextnLayers, String>, Vec<String>) {
        let kv = [
            ("block_count", V::U32(n_layer)),
            ("nextn_predict_layers", V::U32(declared)),
        ];
        let t: Vec<(String, Vec<u64>)> = tensors
            .iter()
            .map(|n| ((*n).to_string(), vec![1]))
            .collect();
        let path = header_shaped(tag, "qwen35moe", &kv, &[], &t);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let mut defaults = Vec::new();
        let got = super::nextn_layers(
            &split,
            n_layer as usize,
            "absent",
            probe_note,
            &mut defaults,
        )
        .map_err(|e| e.to_string());
        let _ = std::fs::remove_file(&path);
        (got, defaults)
    }

    const TRUNK: [&str; 5] = [
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.1.attn_norm.weight",
        "blk.2.attn_norm.weight",
        "blk.3.attn_norm.weight",
    ];

    /// Shape A: `block_count` is the trunk and the declared layers, and those
    /// blocks hold no tensor (MiniCPM-V 4.7's file). The trunk is the blocks
    /// before them and no next-token layer is carried.
    #[test]
    fn a_declared_layer_stripped_of_every_tensor_is_no_trunk_layer() {
        let (got, defaults) = nextn_of("nx-strip-1", 5, 1, &TRUNK, "probe");
        assert_eq!(
            got,
            Ok(NextnLayers {
                past_trunk: 1,
                carried: 0
            })
        );
        assert_eq!(
            defaults,
            [
                "nextn_predict_layers = 1 (blk.4 carries no tensor: a stripped next-token layer; the trunk is 4)"
            ]
        );
        let (got, defaults) = nextn_of("nx-strip-2", 6, 2, &TRUNK, "probe");
        assert_eq!(
            got,
            Ok(NextnLayers {
                past_trunk: 2,
                carried: 0
            })
        );
        assert_eq!(
            defaults,
            [
                "nextn_predict_layers = 2 (blk.4..blk.5 carry no tensor: a stripped next-token layer; the trunk is 4)"
            ]
        );
    }

    /// Shape B: `block_count` is the trunk with the key kept, so the last
    /// blocks are trunk layers; the recorded line is ik's probe line. A layer
    /// backed by `nextn.eh_proj` is believed.
    #[test]
    fn a_kept_key_over_trunk_layers_records_the_probe_line() {
        let (got, defaults) = nextn_of("nx-kept", 4, 1, &TRUNK, "probe");
        assert_eq!(got, Ok(NextnLayers::NONE));
        assert_eq!(
            defaults,
            ["nextn_predict_layers = 0 (blk.3.nextn.eh_proj.weight is not in the file, probe)"]
        );
        let (_, defaults) = nextn_of("nx-kept-bare", 4, 1, &TRUNK, "");
        assert_eq!(
            defaults,
            ["nextn_predict_layers = 0 (blk.3.nextn.eh_proj.weight is not in the file)"]
        );
        let mut backed = TRUNK.to_vec();
        backed.push("blk.4.nextn.eh_proj.weight");
        let (got, defaults) = nextn_of("nx-backed", 5, 1, &backed, "probe");
        assert_eq!(
            got,
            Ok(NextnLayers {
                past_trunk: 1,
                carried: 1
            })
        );
        assert!(defaults.is_empty(), "{defaults:?}");
    }

    /// Blocks that are some empty and some not, or that carry `nextn.*`
    /// without the first one's `eh_proj`, are refused with each block's state.
    #[test]
    fn mixed_next_token_blocks_are_refused_by_name() {
        let with = |extra: &[&'static str]| {
            let mut t = TRUNK.to_vec();
            t.extend_from_slice(extra);
            t
        };
        let refused = |tag: &str, n_layer: u32, declared: u32, t: &[&str], want: &str| {
            let (got, _) = nextn_of(tag, n_layer, declared, t, "probe");
            let err = got.expect_err("the blocks are not one shape");
            assert!(err.contains(want), "want {want:?} in: {err}");
        };
        refused(
            "nx-mix-late",
            6,
            2,
            &with(&["blk.5.nextn.enorm.weight"]),
            "blk.4: 0 tensors, 0 of them nextn.*; blk.5: 1 tensors, 1 of them nextn.*",
        );
        refused(
            "nx-mix-empty-last",
            6,
            2,
            &with(&["blk.4.attn_norm.weight"]),
            "blk.4: 1 tensors, 0 of them nextn.*; blk.5: 0 tensors, 0 of them nextn.*",
        );
        refused(
            "nx-mix-no-eh",
            5,
            1,
            &with(&["blk.4.attn_norm.weight", "blk.4.nextn.enorm.weight"]),
            "blk.4: 2 tensors, 1 of them nextn.*",
        );
    }

    /// The routed and used expert counts read as one pair: each above 0 and the used count within the
    /// routed count, the refusal naming `expert_used_count` and both values. Mutant: `>` loosened
    /// to `>=`, or the zero checks dropped.
    #[test]
    fn expert_counts_are_a_used_within_routed_pair() {
        let read = |tag: &str, routed: u32, used: u32| {
            let path = header_shaped(
                tag,
                "qwen3moe",
                &[
                    ("expert_count", V::U32(routed)),
                    ("expert_used_count", V::U32(used)),
                ],
                &[],
                &[],
            );
            let split = gguf::Split::open(&path).expect("the synthetic header opens");
            let got = super::expert_counts(&split).map_err(|e| e.to_string());
            let _ = std::fs::remove_file(&path);
            got
        };
        assert_eq!(read("ec-ok", 128, 8), Ok((128, 8)));
        assert_eq!(read("ec-all", 8, 8), Ok((8, 8)));
        for (tag, routed, used) in [("ec-over", 8, 9), ("ec-zero-used", 8, 0), ("ec-zero", 0, 0)] {
            let want =
                format!("metadata qwen3moe.expert_used_count: is {used} of expert_count {routed}");
            assert_eq!(read(tag, routed, used), Err(want), "{tag}");
        }
    }

    /// A tensor set that disagrees with the file names the first missing tensor with the count of
    /// the others, before any misplaced one; with none missing it names the first misplaced one;
    /// an agreeing set passes. Mutant: the misplaced check ahead of the missing one, or the count
    /// dropped.
    #[test]
    fn a_tensor_set_refusal_names_the_first_missing_before_the_misplaced() {
        let names = |v: &[&str]| v.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
        let refuse = |missing: &[&str], misplaced: &[&str]| {
            super::refuse_tensor_set(
                &names(missing),
                &names(misplaced),
                "the layers' kinds need",
                "its layer's kind carries none",
            )
        };
        assert!(refuse(&[], &[]).is_ok());
        match refuse(&["blk.1.a", "blk.2.a"], &["blk.0.b"]) {
            Err(crate::placement::PlacementError::Tensor { name, detail }) => {
                assert_eq!(name, "blk.1.a");
                assert_eq!(
                    detail,
                    "is not in the file (2 tensor(s) the layers' kinds need are missing: \
                     blk.1.a, blk.2.a)"
                );
            }
            other => panic!("the first missing name: {other:?}"),
        }
        match refuse(&[], &["blk.0.b", "blk.3.b"]) {
            Err(crate::placement::PlacementError::Tensor { name, detail }) => {
                assert_eq!(name, "blk.0.b");
                assert_eq!(
                    detail,
                    "is in the file, and its layer's kind carries none (2 such: blk.0.b, blk.3.b)"
                );
            }
            other => panic!("the first misplaced name: {other:?}"),
        }
    }

    /// A family planner's error carries the placement refusal it wraps as its source, under the
    /// same text: the start reads a refusal's class from the first [`PlacementError`] of the
    /// chain, so a host room under the NVMe tier floor stays no fit whichever family refused it.
    #[test]
    fn a_family_placement_refusal_is_its_errors_source() {
        use super::{PlacementError, deepseek41, glm5next, qwen3moe, qwen35moe};
        use std::error::Error;
        let floor = || PlacementError::HostRoomFloor {
            room: 1,
            floor: 2,
            base: 0,
            layer: 0,
            need: 2,
            arena: None,
        };
        let families: [Box<dyn Error>; 5] = [
            Box::new(glm5next::place::PlaceError::from(floor())),
            Box::new(qwen3moe::place::PlaceError::from(floor())),
            Box::new(qwen35moe::place::PlaceError::from(floor())),
            Box::new(deepseek41::place::PlaceError::from(floor())),
            Box::new(deepseek41::plan::PlanError::from(floor())),
        ];
        for e in families {
            let source = e.source().and_then(|s| s.downcast_ref::<PlacementError>());
            assert!(
                matches!(source, Some(PlacementError::HostRoomFloor { .. })),
                "{e:?}"
            );
            assert_eq!(e.to_string(), floor().to_string(), "{e:?}");
        }
    }
}
