//! What a V4.1 binary's placement decides beside its plan
//! (`bloomery_gpu_gates::generate::Place`): the expert tier's prompt-batch
//! bytes the plan reserves, and the `load` record's cards — the devices the
//! model runs on, and the tier card's experts and resident bytes when the
//! placement has one. `generate_ds41`,
//! `bloomery-serve-ds41` and `bloomery-chat` read them here.

use bloomery_gpu_deepseek41::body::Deepseek41Model;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::record::Record;
use model::arch::deepseek41::hparams::Hparams;
use model::placement::workstation::TierBatchBytes;

/// The expert tier's prompt-batch bytes `place`'s plan reserves for the file
/// of hyperparameters `hp` (`model::arch::deepseek41::place::tier_batch`),
/// which `Place::machine` takes; `None` under a placement with no tier card.
pub fn tier_batch(place: Place, hp: &Hparams) -> Option<TierBatchBytes> {
    place
        .tier_card()
        .map(|_| model::arch::deepseek41::place::tier_batch(hp))
}

/// `r` with the devices the model runs on (`cards`: the stage card, then
/// the tier card), by the names their drivers report — each space written
/// `_`, since the field is one word — and, when the placement has an expert
/// tier card, the tier's experts and resident bytes. A device whose name does
/// not hold the placement's card name, and a model whose tier does not match
/// the placement — none where the placement names one, one where it names
/// none, or a tier on another card — are refused by name.
pub fn with_cards(
    m: &Deepseek41Model,
    place: Place,
    what: &'static str,
    r: Record,
) -> Result<Record, GateError> {
    let tier = m.body(what)?.hybrid().tier();
    let mut devices = vec![m.gpu().device_name()?];
    if let Some(t) = tier {
        devices.push(t.gpu().device_name()?);
    }
    let planned = place.cards();
    let named =
        devices.len() == planned.len() && devices.iter().zip(planned).all(|(d, p)| d.contains(p));
    if !named {
        return Err(format!(
            "--place {}: the model runs on {devices:?}, the placement's cards are {planned:?}",
            place.name()
        )
        .into());
    }
    let r = r.csv("cards", devices.iter().map(|d| d.replace(' ', "_")));
    match (tier, place.tier_card()) {
        (None, None) => Ok(r),
        (Some(t), Some(name)) if t.name() == name => Ok(r
            .u("tier_experts", t.set().experts())
            .u("tier_bytes", t.weights().resident_bytes())),
        (t, want) => Err(format!(
            "--place {}: the loaded model's expert tier is {:?}, the placement's {want:?}",
            place.name(),
            t.map(|t| t.name().to_owned())
        )
        .into()),
    }
}
