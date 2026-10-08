//! What a V4.1 binary's placement decides beside its plan
//! (`bloomery_gpu_gates::generate::Place`): the unset `--place` rule, the
//! expert tier's prompt-batch bytes the plan reserves, the `load` record's
//! cards (the shared rule, `generate::with_cards`), and the `--place` word a
//! V4.1 body serves. `generate_ds41`, `bloomery-serve-ds41` and
//! `bloomery-chat` read them here.

use bloomery_gpu_deepseek41::body::Deepseek41Model;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::{Place, PlaceWhy, TierRule};
use bloomery_gpu_gates::gpu_census;
use bloomery_gpu_gates::record::Record;
use model::arch::deepseek41::hparams::Hparams;
use model::placement::workstation::TierBatchBytes;

/// The expert tier cards the V4.1 body serves: the host tier's count
/// (`Body::open_placed_tiered` refuses past it).
const SERVED_TIERS: usize = bloomery_gpu::host::SERVED_TIERS;

/// The V4.1 family's input to the common unset rule (`Place::choose`): the
/// tier cards its body serves, and no break-even — no sitting has shown the
/// tier not slower, so two cards run `a`, the plan is never asked and no card
/// file decides anything.
pub const TIER_RULE: TierRule = TierRule {
    tiers: SERVED_TIERS,
    break_even: None,
    basis: "",
};

/// The placement a V4.1 binary runs by, and why, on this process's census
/// (`Place::choose` by [`TIER_RULE`]): `flag` as given when set, else `a`.
/// The caller prints [`PlaceWhy::record`].
pub fn choose(flag: Option<Place>) -> Result<PlaceWhy, GateError> {
    Place::choose(flag, &gpu_census::census()?, TIER_RULE, |offer| {
        Err(format!(
            "--place unset: V4.1 has no break-even yet, so the plan is not asked for {}",
            offer.name()
        )
        .into())
    })
}

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

/// `r` with the cards `m` runs on and its expert tier's fields
/// ([`bloomery_gpu_gates::generate::with_cards`], every body's rule).
pub fn with_cards(
    m: &Deepseek41Model,
    place: Place,
    what: &'static str,
    r: Record,
) -> Result<Record, GateError> {
    let tiers = m.body(what)?.hybrid().tiers();
    bloomery_gpu_gates::generate::with_cards(r, place, m.gpu(), tiers)
}
