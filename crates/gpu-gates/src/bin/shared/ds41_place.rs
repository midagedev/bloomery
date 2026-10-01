//! What a V4.1 binary's placement decides beside its plan
//! (`bloomery_gpu_gates::generate::Place`): the expert tier's prompt-batch
//! bytes the plan reserves, and the `load` record's cards — the devices the
//! model runs on, and the tier card's experts and resident bytes when the
//! placement has one, and the `--place` word a V4.1 body serves.
//! `generate_ds41`, `bloomery-serve-ds41` and `bloomery-chat` read them here.

use bloomery_gpu_deepseek41::body::Deepseek41Model;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::record::Record;
use model::arch::deepseek41::hparams::Hparams;
use model::placement::workstation::TierBatchBytes;

/// The expert tier cards the V4.1 body serves: the host tier's count
/// (`Body::open_placed_tiered` refuses past it).
const SERVED_TIERS: usize = bloomery_gpu::host::SERVED_TIERS;

/// A V4.1 binary's `--place` word (`Place::parse`), refused by name before
/// any plan when it lists more tier cards than the V4.1 body serves.
pub fn parse(v: &str) -> Result<Place, GateError> {
    let place = Place::parse(v)?;
    place.serves("deepseek41", SERVED_TIERS)?;
    Ok(place)
}

/// The expert tier's prompt-batch bytes `place`'s plan reserves for the file
/// of hyperparameters `hp` (`model::arch::deepseek41::place::tier_batch`),
/// which `Place::machine` takes; `None` under a placement with no tier card.
pub fn tier_batch(place: Place, hp: &Hparams) -> Option<TierBatchBytes> {
    (!place.tier_cards().is_empty()).then(|| model::arch::deepseek41::place::tier_batch(hp))
}

/// `r` with the devices the model runs on (`cards`: the stage card, then
/// the tier cards), by the names their drivers report — each space written
/// `_`, since the field is one word — and, when the placement has an expert
/// tier card, the tier's experts and resident bytes. A device whose name does
/// not hold the placement's card name, a model whose tiers do not match the
/// placement's — another count, or a tier on another card — and more than
/// the one tier the record's fields hold are refused by name.
pub fn with_cards(
    m: &Deepseek41Model,
    place: Place,
    what: &'static str,
    r: Record,
) -> Result<Record, GateError> {
    let tiers = m.body(what)?.hybrid().tiers();
    let mut devices = vec![m.gpu().device_name()?];
    for t in tiers {
        devices.push(t.gpu().device_name()?);
    }
    let planned = place.cards();
    let named =
        devices.len() == planned.len() && devices.iter().zip(&planned).all(|(d, p)| d.contains(p));
    if !named {
        return Err(format!(
            "--place {}: the model runs on {devices:?}, the placement's cards are {planned:?}",
            place.name()
        )
        .into());
    }
    let r = r.csv("cards", devices.iter().map(|d| d.replace(' ', "_")));
    let want = place.tier_cards();
    let got: Vec<&str> = tiers.iter().map(|t| t.name()).collect();
    match (tiers, want.as_slice()) {
        ([], []) => Ok(r),
        ([t], [name]) if t.name() == *name => Ok(r
            .u("tier_experts", t.set().experts())
            .u("tier_bytes", t.weights().resident_bytes())),
        ([_, _, ..], _) if got == want => Err(format!(
            "--place {}: the load record holds one expert tier's fields, and the model loaded \
             {} tiers",
            place.name(),
            got.len()
        )
        .into()),
        _ => Err(format!(
            "--place {}: the loaded model's expert tiers are {got:?}, the placement's {want:?}",
            place.name()
        )
        .into()),
    }
}
