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
pub mod draft;
pub mod dspark;
pub mod glm5next;
pub mod mimo2;
pub mod names;
pub mod qwen35moe;
pub mod qwen3moe;
mod stems;

/// The typed model description, its readers' output: re-exported so a
/// caller names one crate for the model and its description.
pub use models;

/// `split`'s [`ModelSpec`] by its family's reader, with every tensor's role:
/// what the coverage check ([`coverage::check`]) and the programs read. An
/// architecture no reader here reads is [`ModelError::UnknownArchitecture`]
/// with the string the file held. `deepseek2` has no reader.
pub fn spec(split: &Split) -> Result<Read, ModelError> {
    Ok(match split.architecture() {
        Some("deepseek41" | DEEPSEEK4) => deepseek41::spec::read(split)?,
        Some("qwen3moe") => qwen3moe::spec::read(split)?,
        Some("qwen35moe" | "qwen4exp") => qwen35moe::spec::read(split)?,
        Some(a) if is_qwen35_body(a) => qwen35moe::spec::read(split)?,
        Some("glm5next" | GLM5_NEXT_LLAMA_CPP) => glm5next::spec::read(split)?,
        Some("mimo2") => mimo2::spec::read(split)?,
        other => {
            return Err(ModelError::UnknownArchitecture(
                other.unwrap_or("<missing>").to_string(),
            ));
        }
    })
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

/// The architectures this engine can run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arch {
    Deepseek2,
    Deepseek41,
    Qwen3moe,
    Qwen35moe,
    Glm5next,
    Mimo2,
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
    /// the variants whose name is not the only string they stand for.
    pub fn from_name(name: &str) -> Result<Arch, ModelError> {
        match name {
            "deepseek2" => Ok(Arch::Deepseek2),
            "deepseek41" | DEEPSEEK4 => Ok(Arch::Deepseek41),
            "qwen3moe" => Ok(Arch::Qwen3moe),
            "qwen35moe" => Ok(Arch::Qwen35moe),
            "glm5next" | GLM5_NEXT_LLAMA_CPP => Ok(Arch::Glm5next),
            "mimo2" => Ok(Arch::Mimo2),
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

/// `<architecture>.<suffix>` as a float.
fn meta_f32(split: &Split, suffix: &str) -> Result<f32, PlacementError> {
    split
        .arch_get_f32(suffix)
        .ok_or_else(|| metadata(split, suffix, "is absent or not a float"))
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
/// absent key (`absent_note`, the line that defaults it). Where the first of
/// the blocks carries `nextn.eh_proj`, the count is believed. Where it does
/// not, the blocks are read by their tensors:
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
            defaults.push(format!("{key} = 0 ({absent_note})"));
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
/// `tokenizer.chat_template`.
const TEMPLATE: &str = "tokenizer.chat_template";

/// The file's chat surface: its pre-tokenizer (required) and template, with
/// the family's parsers.
fn chat_of(
    split: &Split,
    tools: Option<ToolFormat>,
    reasoning: Option<ReasoningFormat>,
) -> Result<ChatSpec, PlacementError> {
    let pre = split
        .value(PRE)
        .and_then(Value::as_str)
        .ok_or_else(|| PlacementError::Metadata {
            key: PRE.to_string(),
            detail: "is absent or not a string".to_string(),
        })?;
    let template = match split.value(TEMPLATE) {
        None => None,
        Some(v) => Some(v.as_str().ok_or_else(|| PlacementError::Metadata {
            key: TEMPLATE.to_string(),
            detail: "is not a string".to_string(),
        })?),
    };
    Ok(ChatSpec {
        pre: pre.to_string(),
        template: template.map(str::to_string),
        tools,
        reasoning,
    })
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
        F32(f32),
        Str(&'static str),
        Bool(bool),
        /// An array of I32, the type a converter writes a per-layer count in.
        I32s(Vec<i32>),
        /// An array of F32.
        F32s(Vec<f32>),
        /// An array of U64, the type a converter writes a hash constant in.
        U64s(Vec<u64>),
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
        /// Blocks of 32 values in 34 bytes; the first dim a multiple of 32.
        Q8_0,
        /// Blocks of 256 values in 210 bytes; any first dim (a partial final
        /// block is a header-only fixture's).
        Q6K,
        /// Blocks of 256 values in 144 bytes; any first dim.
        Q4K,
    }

    impl Ty {
        /// The ggml type id and the bytes of `n` values.
        fn id_and_bytes(self, n: u64) -> (u32, u64) {
            match self {
                Ty::F32 => (0, 4 * n),
                Ty::Q8_0 => {
                    assert!(n.is_multiple_of(32), "{n} values are not whole Q8_0 blocks");
                    (8, n / 32 * 34)
                }
                Ty::Q6K => (14, n.div_ceil(256) * 210),
                Ty::Q4K => (12, n.div_ceil(256) * 144),
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
                    xs.iter()
                        .for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
                }
                V::F32s(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&6u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    xs.iter()
                        .for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
                }
                V::U64s(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&10u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    xs.iter()
                        .for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
                }
            }
        }
        let mut offset = 0u64;
        for (name, dims, ty) in tensors {
            let (id, bytes) = ty.id_and_bytes(dims.iter().product::<u64>());
            string(&mut b, name);
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            dims.iter()
                .for_each(|d| b.extend_from_slice(&d.to_le_bytes()));
            b.extend_from_slice(&id.to_le_bytes());
            b.extend_from_slice(&offset.to_le_bytes());
            offset += bytes.div_ceil(32) * 32;
        }
        b.resize(b.len().div_ceil(32) * 32 + offset as usize, 0);
        let path = std::env::temp_dir().join(format!("bloomery-{}-{tag}.gguf", std::process::id()));
        std::fs::write(&path, b).expect("write the synthetic header");
        path
    }
}

#[cfg(test)]
mod tests {
    use super::synthetic::{V, header_shaped};
    use super::{Arch, NextnLayers};

    #[test]
    fn known_names_round_trip() {
        for a in [
            Arch::Deepseek2,
            Arch::Deepseek41,
            Arch::Qwen3moe,
            Arch::Qwen35moe,
            Arch::Glm5next,
            Arch::Mimo2,
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
}
