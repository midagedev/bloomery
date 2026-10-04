//! The load of a Qwen3.5 dense file (`qwen35`, Clef's backbone) for its
//! prompt call, which `clef_hidden`, its gate and `bloomery-serve`'s decide
//! seat share, and the ids file reader of the first two.

use bloomery_gpu::Gpu;
use bloomery_gpu::arch::qwen3moe::{KvQ8, Open35, Qwen35moeModel};
use bloomery_gpu_gates::GateError;
use gguf::Split;
use model::arch::models::Arch;
use model::arch::qwen3moe::place as q3;
use model::arch::qwen35moe::place as q35;
use model::placement::workstation::spec_of_device;
use model::placement::{KvBytes, whole_need};
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
/// architecture is refused by name, and so is a load the whole-fit verdict
/// does not take on device 0 — the card it opens on — before any upload,
/// with the verdict's terms ([`refuse_unless_whole_fits`]): a dense
/// backbone has no routed experts to move to the host tier.
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
    let o = Open35 {
        ctx,
        mma: false,
        ubatch,
        kv: KvQ8::F16,
    };
    refuse_unless_whole_fits(&split, o).map_err(|e| format!("{}: {e}", path.display()))?;
    let model = Qwen35moeModel::open(Gpu::new()?, Split::open(path)?, o)?;
    Ok((model, split))
}

/// Refuse the whole load of `split` under `o` unless the whole-fit verdict
/// (`placement::whole_need`, its one owner) takes it on device 0 at census
/// time: every tensor's granules, the cache of `o.ctx` rows, the card the
/// program's plan lays out (`q3::machine`: context and step arenas'
/// scratch), and the ubatch arena with the reserve the load's own fit check
/// keeps free past it (`GpuModel::whole_load`). The verdict's line goes to
/// stderr; a refusal names its terms, the card and its holders. A census
/// that reads no device 0 is refused by name.
fn refuse_unless_whole_fits(split: &Split, o: Open35) -> Result<(), GateError> {
    let census = bloomery_gpu_gates::gpu_census::census()?;
    let d0 = census
        .iter()
        .find(|d| d.ordinal == 0)
        .ok_or("the census reads no device 0, the card the whole load opens on")?;
    let inputs = q35::PlanInputs::describe(split)?;
    let layers = inputs.hp.n_layer;
    let ctx = u64::try_from(o.ctx)?;
    let kv = (0..layers).map(|l| inputs.kv.layer_bytes(l, ctx)).sum();
    let machine = q3::machine(spec_of_device(d0), layers, 0);
    let need = whole_need(
        &inputs.model,
        &machine.cards[0],
        kv,
        Qwen35moeModel::whole_load(split, o)?,
    )?;
    eprintln!("whole fit at ctx {}: {need}", o.ctx);
    if need.fits() {
        return Ok(());
    }
    Err(format!("{need}: a dense backbone has no routed experts to move to the host tier").into())
}
