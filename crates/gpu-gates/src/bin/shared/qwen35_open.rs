//! The load of a Qwen3.5 dense file (`qwen35`, or llama.cpp's Clef layout
//! `clef`, which is the same trunk with a head beside it) for its
//! prompt call, which `clef_hidden`, its gate and `bloomery-serve`'s decide
//! seat share, and the ids file reader of the first two.

use bloomery_gpu::Gpu;
use bloomery_gpu::arch::qwen3moe::{KvQ8, Open35, Qwen35moeModel};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::Place;
use gguf::Split;
use model::arch::qwen3moe::place as q3;
use model::arch::qwen35moe::place as q35;
use model::arch::{QWEN35_BODY, is_qwen35_body};
use model::placement::workstation::CardSpec;
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

/// The Qwen3.5 dense file ([`QWEN35_BODY`]) at `path`, whole on one card: a cache of `ctx` rows,
/// prompt ubatches of `ubatch` ids, the scalar flash for the row passes
/// (`mma` false). The 27B file's group of 6 query heads a KV head runs the
/// pairs' pass, which has no tensor-core form and is refused with `mma` at
/// load; Clef-Flash's group of 4 would take it, but `gate_clef_hidden` holds
/// the scalar pass, so every file opens with it. A file of any other
/// architecture is refused by name. A `clef` file's head tensors are never
/// loaded here. The card is `a`'s on one census
/// reading ([`Place::A`]: the largest visible card by usable bytes, ties to
/// the lower ordinal — the qwen3 seat's whole load takes the same pick,
/// `PlaceQ3::unplaced_on`), whatever the CUDA order puts at device 0; a
/// load the whole-fit verdict does not take there is refused before any
/// upload, with the verdict's terms ([`refuse_unless_whole_fits`]): a dense
/// backbone has no routed experts to move to the host tier.
pub fn open(path: &Path, ctx: usize, ubatch: usize) -> Result<(Qwen35moeModel, Split), GateError> {
    let split = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    if !split.architecture().is_some_and(is_qwen35_body) {
        return Err(format!(
            "{} is {:?}, and this reads {QWEN35_BODY:?} (Qwen3.5 dense)",
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
    let census = bloomery_gpu_gates::gpu_census::census()?;
    let card = Place::A.on(&census)?.card_specs()?[0];
    refuse_unless_whole_fits(&split, o, card).map_err(|e| format!("{}: {e}", path.display()))?;
    let gpu = Gpu::open_card(card.name, card.device)?;
    let model = Qwen35moeModel::open(gpu, Split::open(path)?, o)?;
    Ok((model, split))
}

/// Refuse the whole load of `split` under `o` unless the whole-fit verdict
/// (`placement::whole_need`, its one owner) takes it on `card`, the card it
/// opens on, as the census read it: every tensor's granules, the cache of
/// `o.ctx` rows, the card the program's plan lays out (`q3::machine`:
/// context and step arenas' scratch), and the ubatch arena with the reserve
/// the load's own fit check keeps free past it (`GpuModel::whole_load`). The
/// verdict's line goes to stderr; a refusal names its terms, the card and
/// its holders.
fn refuse_unless_whole_fits(split: &Split, o: Open35, card: CardSpec) -> Result<(), GateError> {
    let inputs = q35::PlanInputs::describe(split)?;
    let layers = inputs.hp.n_trunk;
    let ctx = u64::try_from(o.ctx)?;
    let kv = (0..layers).map(|l| inputs.kv.layer_bytes(l, ctx)).sum();
    let machine = q3::machine(card, layers, 0);
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
