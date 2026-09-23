//! `deepseek41` — DeepSeek-V4.1-Flash. The host code that knows this model: its
//! hyperparameters and per-layer kinds (`hparams`), the tensor names the step
//! reads (`names`), the role of every tensor (`roles`), the KV bytes each layer
//! holds (`kv`) and the integers each step's graph reads (`plan`).

pub mod hparams;
pub mod kv;
pub mod names;
pub mod plan;
pub mod roles;

use gguf::Split;

use crate::placement::PlacementError;

/// `<architecture>.<suffix>` as an unsigned integer, or the error that names the key.
fn meta_u64(split: &Split, suffix: &str) -> Result<u64, PlacementError> {
    split
        .arch_get_u64(suffix)
        .ok_or_else(|| PlacementError::Metadata {
            key: split.arch_key(suffix),
            detail: "is absent or not an unsigned integer".to_string(),
        })
}

/// `meta_u64` for a count that indexes memory.
fn meta_usize(split: &Split, suffix: &str) -> Result<usize, PlacementError> {
    let v = meta_u64(split, suffix)?;
    usize::try_from(v).map_err(|_| PlacementError::Metadata {
        key: split.arch_key(suffix),
        detail: format!("{v} does not fit usize"),
    })
}
