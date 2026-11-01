//! The load of a dense file the Qwen3.5 body runs (`qwen35`, or llama.cpp's Clef layout
//! `clef`, which is the same trunk with a head beside it, or a dense `llama` trunk) for its
//! prompt call, which `clef_hidden`, its gate and `bloomery-serve`'s decide
//! seat share, and the ids file reader of the first two.

use bloomery_gpu::Gpu;
use bloomery_gpu::arch::qwen3moe::ubatch::UBATCH;
use bloomery_gpu::arch::qwen3moe::{Open35, Qwen35moeModel};
use bloomery_gpu::kv_format::KvQ8;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::beside;
use bloomery_gpu_gates::generate::{Place, SEAT_STEPS};
use bloomery_gpu_gates::ubatch_pick::{self, Rule, Step};
use gguf::Split;
use model::arch::models::Ffn;
use model::arch::qwen35moe::place as q35;
use model::placement::accepts::{Program, program_of};
use model::placement::ctx_ask::{Probe, Short, Side};
use model::placement::workstation::{self, CardSpec};
use model::placement::{Machine, ModelTensors, WholeNeed, whole_need};
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

/// The dense file at `path` the Qwen3.5 body runs (its description's program, every layer's FFN
/// dense: `qwen35`, Clef's `clef` and a dense `llama` trunk), whole on one card: a cache of `ctx`
/// rows, prompt ubatches of `ubatch` ids, the scalar flash for the row passes (`mma` false). The
/// 27B file's group of 6 query heads a KV head runs the pairs' pass, which has no tensor-core form
/// and is refused with `mma` at load; Clef-Flash's group of 4 would take it, but `gate_clef_hidden`
/// holds the scalar pass, so every file opens with it; a group of 7 has the tensor-core pass only
/// and is opened by [`open_options`]. A file of any other description is refused by name. A `clef`
/// file's head tensors are never loaded here. The card is `a`'s on one census reading
/// ([`Place::A`]: the largest visible card by usable bytes, ties to the lower ordinal — the qwen3
/// seat's whole load takes the same pick, `PlaceQ3::unplaced_on`), whatever the CUDA order puts at
/// device 0; a load the whole-fit verdict does not take there is refused before any upload, with
/// the verdict's terms ([`refuse_unless_whole_fits`]): a dense backbone has no routed experts to
/// move to the host tier.
pub fn open(path: &Path, ctx: usize, ubatch: usize) -> Result<(Qwen35moeModel, Split), GateError> {
    open_with(path, ctx, ubatch, false)
}

/// [`open`] with the sequences' rotation tables of a media load (`media`: [`Open35::media`]); the same
/// fit check, the same card.
pub fn open_with(
    path: &Path,
    ctx: usize,
    ubatch: usize,
    media: bool,
) -> Result<(Qwen35moeModel, Split), GateError> {
    open_seated(path, ctx, ubatch, media, &|_| {})
}

/// [`open_with`] whose whole-fit machine goes through `seat` before the verdict: a seat that holds
/// bytes of the card for something else (the image encoder on the stage card,
/// `beside::reserve_on`) puts them there, so the verdict counts them.
pub fn open_seated(
    path: &Path,
    ctx: usize,
    ubatch: usize,
    media: bool,
    seat: &dyn Fn(&mut Machine),
) -> Result<(Qwen35moeModel, Split), GateError> {
    let o = Open35 {
        ctx,
        mma: false,
        ubatch,
        kv: KvQ8::F16,
        media,
    };
    open_options(path, o, seat)
}

/// [`open_seated`] with the load's options `o` named whole: a file whose flash is the tensor-core
/// pass only (a group of 7 query heads a KV head) opens with `o.mma`.
pub fn open_options(
    path: &Path,
    o: Open35,
    seat: &dyn Fn(&mut Machine),
) -> Result<(Qwen35moeModel, Split), GateError> {
    let split = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    refuse_unless_dense_body(&split, path)?;
    let census = bloomery_gpu_gates::gpu_census::census()?;
    let card = Place::A.on(&census)?.card_specs()?[0];
    refuse_unless_whole_fits(&split, o, card, seat)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let gpu = Gpu::open_card(card.name, card.device)?;
    let model = Qwen35moeModel::open(gpu, Split::open(path)?, o)?;
    Ok((model, split))
}

/// Refused by name unless `split`'s description is a trunk the Qwen3.5 body runs with a dense FFN
/// in every layer: its architecture's program, from the common reader.
fn refuse_unless_dense_body(split: &Split, path: &Path) -> Result<(), GateError> {
    let spec = model::arch::spec(split)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .spec;
    let dense = spec
        .layers
        .iter()
        .all(|l| matches!(l.ffn, Ffn::Dense { .. }));
    if program_of(spec.arch) == Some(Program::Qwen35Body) && dense {
        return Ok(());
    }
    Err(format!(
        "{} is {:?}, and this opens a file the Qwen3.5 body runs with a dense FFN in every layer",
        path.display(),
        spec.arch,
    )
    .into())
}

/// The open the dense seat and its context probe share: scalar flash, f16
/// KV, prompt ubatches clipped to `ctx` at the family's ubatch cap, and media
/// tables when an encoder seat exists.
#[allow(
    dead_code,
    reason = "the hidden and media gates include this file and open at a context the caller names"
)]
pub fn open_shape(ctx: usize, media: bool) -> Open35 {
    Open35 {
        ctx,
        mma: false,
        ubatch: UBATCH.min(ctx),
        kv: KvQ8::F16,
        media,
    }
}

/// The whole-fit verdict of `o` on `card` (`placement::whole_need`): every
/// tensor's granules, the cache of `o.ctx` rows, the card the program's plan
/// lays out (`workstation::placed_with_arena`), the ubatch arena with the reserve the load's own
/// fit check keeps free past it (`GpuModel::whole_load`), and what `seat`
/// puts on the machine. The caller prints it.
pub fn whole_verdict(
    split: &Split,
    o: Open35,
    card: CardSpec,
    seat: &dyn Fn(&mut Machine),
) -> Result<WholeNeed, GateError> {
    let ctx = u64::try_from(o.ctx)?;
    let (model, layers, kv) = trunk_terms(split, ctx, o.kv)?;
    let mut machine = workstation::placed_with_arena(card, layers, 0);
    seat(&mut machine);
    let name = card.name;
    let view = machine
        .whole_card()
        .ok_or_else(|| format!("the whole load on {name}: a machine with no card"))?;
    whole_need(&model, &view, kv, Qwen35moeModel::whole_load(split, o)?).map_err(Into::into)
}

/// The decide seat's context probe at the shape `shape` names, whole on `card` with `seat`'s bytes
/// beside the backbone: the plan holds with the prompt ubatch `rule` settles ([`ubatch_pick::settle`],
/// the card's room unless the user asked a size; one `ubatch` line on
/// stderr; the caller prints the open's `whole fit` line), or it is short for the context choice to
/// search below. At the least context the choice tries, a size the user named that does not fit is
/// refused as that size's ([`ubatch_pick::asked_short`]): no smaller context helps.
#[allow(
    dead_code,
    reason = "the decide seat probes through it; the other bins open at the size they name"
)]
pub fn probe_whole(
    split: &Split,
    shape: Open35,
    card: CardSpec,
    rule: Rule,
    seat: &dyn Fn(&mut Machine),
) -> Result<Probe<usize>, GateError> {
    let (ubatch, step) = ubatch_pick::settle(rule, (shape.ubatch, shape.ctx, 1), |u| {
        let need = whole_verdict(split, Open35 { ubatch: u, ..shape }, card, seat)?;
        Ok::<_, GateError>(Step {
            terms: Some((need.need, need.budget)),
            fits: need.fits(),
            refused: false,
            kept: need,
        })
    })?;
    let need = step.kept;
    if need.fits() {
        return Ok(Probe::Holds(ubatch));
    }
    if let Some(ask) = rule.asked()
        && u64::try_from(shape.ctx)? <= SEAT_STEPS.floor
    {
        let largest = ubatch_pick::largest_fit(|u| {
            let at = whole_verdict(split, Open35 { ubatch: u, ..shape }, card, seat)?;
            Ok::<_, GateError>(at.fits())
        })?;
        return Err(
            ubatch_pick::asked_short(ask, (shape.ctx, 1), need.need, need.budget, largest).into(),
        );
    }
    Ok(Probe::Short(Short {
        on: Side::Card,
        need: need.need,
        room: need.budget,
    }))
}

/// The tensors with their roles, the trunk's layer count and the bytes of the cache at `ctx` rows
/// in the format `format` the whole-fit verdict counts: the description's layout summed over the
/// trunk ([`q35::KvLayout::trunk_bytes`]), whatever mixes its layers.
fn trunk_terms(
    split: &Split,
    ctx: u64,
    format: KvQ8,
) -> Result<(ModelTensors, usize, u64), GateError> {
    let read = model::arch::spec(split).map_err(|e| format!("the header: {e}"))?;
    let kv = q35::KvLayout::trunk_bytes(&read.spec, ctx, format == KvQ8::Q8)?;
    Ok((read.tensors, read.spec.layers.len(), kv))
}

/// Refuse the whole load of `split` under `o` unless [`whole_verdict`] takes
/// it on `card`, the card it opens on, as the census read it. The verdict's
/// line goes to stderr; a refusal names its terms, the card and its holders.
/// A dense backbone has no routed experts to move to the host tier.
fn refuse_unless_whole_fits(
    split: &Split,
    o: Open35,
    card: CardSpec,
    seat: &dyn Fn(&mut Machine),
) -> Result<(), GateError> {
    let need = whole_verdict(split, o, card, seat)?;
    let held = beside::note(&need.card);
    eprintln!("whole fit at ctx {}: {need}{held}", o.ctx);
    if need.fits() {
        return Ok(());
    }
    Err(
        format!("{need}{held}: a dense backbone has no routed experts to move to the host tier")
            .into(),
    )
}
