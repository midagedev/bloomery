//! The load of a Qwen3.5 dense file (`qwen35`, Clef's backbone) for its
//! prompt call, which `clef_hidden`, its gate and `bloomery-serve`'s decide
//! seat share, and the ids file reader of the first two.

use bloomery_gpu::Gpu;
use bloomery_gpu::arch::qwen3moe::{Open35, Qwen35moeModel};
use bloomery_gpu_gates::GateError;
use gguf::Split;
use model::arch::models::Arch;
use std::path::Path;

/// The first `n` ids of `path`, one decimal id a line; a line that is not an
/// id is refused by its line number, and a file of fewer than `n` ids by
/// name. `None` takes every id.
pub fn read_ids(path: &Path, n: Option<usize>) -> Result<Vec<u32>, GateError> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let lines = text.lines().take(n.unwrap_or(usize::MAX));
    let ids = lines
        .enumerate()
        .map(|(i, l)| {
            l.trim().parse::<u32>().map_err(|e| {
                format!(
                    "{} line {}: {l:?} is not an id ({e})",
                    path.display(),
                    i + 1
                )
            })
        })
        .collect::<Result<Vec<u32>, String>>()?;
    match n {
        Some(n) if ids.len() < n => Err(format!(
            "{} holds {} ids, and {n} were asked for",
            path.display(),
            ids.len()
        )
        .into()),
        _ if ids.is_empty() => Err(format!("{} holds no ids", path.display()).into()),
        _ => Ok(ids),
    }
}

/// The `qwen35` file at `path`, whole on one card: a cache of `ctx` rows,
/// prompt ubatches of `ubatch` ids, the scalar flash for the row passes
/// (`mma` false). The 27B file's group of 6 query heads a KV head runs the
/// pairs' pass, which has no tensor-core form and is refused with `mma` at
/// load; Clef-Flash's group of 4 would take it, but `gate_clef_hidden` holds
/// the scalar pass, so every file opens with it. A file of any other
/// architecture is refused by name.
pub fn open(path: &Path, ctx: usize, ubatch: usize) -> Result<(Qwen35moeModel, Split), GateError> {
    let split = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let want = Arch::Qwen35.name();
    if split.architecture() != Some(want) {
        return Err(format!(
            "{} is {:?}, and this reads {want} (Qwen3.5 dense)",
            path.display(),
            split.architecture()
        )
        .into());
    }
    let model = Qwen35moeModel::open(
        Gpu::new()?,
        Split::open(path)?,
        Open35 {
            ctx,
            mma: false,
            ubatch,
        },
    )?;
    Ok((model, split))
}
