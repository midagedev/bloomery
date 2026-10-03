//! The load of a Qwen3.5 dense file (`qwen35`, Clef's backbone) for its
//! prompt call, which `clef_hidden`, its gate and `bloomery-serve`'s decide
//! seat share, and the ids file reader of the first two.

use bloomery_gpu::Gpu;
use bloomery_gpu::arch::qwen3moe::{Open35, Qwen35moeModel};
use bloomery_gpu_gates::GateError;
use gguf::Split;
use model::arch::models::Arch;
use model::placement::workstation;
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
/// architecture is refused by name, and a card whose free bytes cannot hold
/// even the file's own bytes — a lower bound of the whole load's device
/// need, every tensor's upload — is refused by name before the load: a
/// dense backbone has no routed experts to move to the host tier.
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
    refuse_when_free_is_past_the_weights(path, &split)?;
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

/// Refuse the whole load of `split` when device 0 — the card it opens on —
/// had fewer free bytes at census time than the file's own bytes: every
/// tensor uploads, so the file's bytes are a lower bound of the load's
/// device need, and past them no placement of this dense file fits. The
/// refusal names the card, its free and usable bytes, the need and the
/// holders nvidia-smi named; a census of no device leaves the load to the
/// driver's own refusal.
fn refuse_when_free_is_past_the_weights(path: &Path, split: &Split) -> Result<(), GateError> {
    let census = bloomery_gpu_gates::gpu_census::census()?;
    let Some(d0) = census.iter().find(|d| d.ordinal == 0) else {
        return Ok(());
    };
    let weights: u64 = split.iter_tensors().map(|(_, t)| t.nbytes).sum();
    if d0.free_bytes >= weights {
        return Ok(());
    }
    let held = d0
        .held_by
        .as_ref()
        .map_or(String::new(), |h| format!(" (held by {h})"));
    Err(format!(
        "{} is a dense backbone of {weights} B, and {} (cuda0) had {} B free of its usable {} B \
         at plan time{held}: no routed experts to move to the host tier",
        path.display(),
        d0.name,
        d0.free_bytes,
        workstation::census_usable(d0.total_bytes),
    )
    .into())
}
