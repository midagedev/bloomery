//! Which architecture a file holds, and the only place in this crate that reads
//! `general.architecture`.
//!
//! Variant names are the GGUF strings themselves, not abbreviations: a single
//! `grep deepseek41` has to find the module, the metadata keys, the tool profile
//! and the gates at once.

use crate::ModelError;
use crate::placement::PlacementError;
use gguf::{Gguf, Split, Value};

pub mod deepseek2;
pub mod deepseek41;
pub mod dspark;

/// The architecture string of the DSpark draft file ([`dspark`]). It is not an
/// [`Arch`]: the draft is read beside a V4.1 model, never run as a model.
pub const DFLASH: &str = "dflash";

/// Whether `split` declares the draft's architecture.
pub fn is_dflash(split: &Split) -> bool {
    split.architecture() == Some(DFLASH)
}

/// The architectures this engine can run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arch {
    Deepseek2,
    Deepseek41,
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
    pub fn from_name(name: &str) -> Result<Arch, ModelError> {
        match name {
            "deepseek2" => Ok(Arch::Deepseek2),
            "deepseek41" => Ok(Arch::Deepseek41),
            other => Err(ModelError::UnknownArchitecture(other.to_string())),
        }
    }

    /// The `general.architecture` string this variant stands for.
    pub fn name(&self) -> &'static str {
        match self {
            Arch::Deepseek2 => "deepseek2",
            Arch::Deepseek41 => "deepseek41",
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

#[cfg(test)]
mod tests {
    use super::Arch;

    #[test]
    fn known_names_round_trip() {
        for a in [Arch::Deepseek2, Arch::Deepseek41] {
            assert_eq!(Arch::from_name(a.name()).unwrap(), a);
        }
    }

    #[test]
    fn unknown_name_is_reported_verbatim() {
        let err = Arch::from_name("llama").unwrap_err();
        assert!(
            err.to_string().contains("llama"),
            "the error must name the string the file held, got {err}"
        );
    }
}
