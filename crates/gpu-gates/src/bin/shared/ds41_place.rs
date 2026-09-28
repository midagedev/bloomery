//! What a V4.1 binary's placement decides beside its plan
//! (`bloomery_gpu_gates::generate::Place`): the prompt feed a placement holds
//! every call to, and the `load` record's cards — the tier card's experts and
//! resident bytes when the placement has one. `generate_ds41`,
//! `bloomery-serve-ds41` and `bloomery-chat` read them here.

use bloomery_gpu_deepseek41::body::{Deepseek41Model, OpenCfg, PrefillMode};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::record::{self, Record};

/// `cfg` under `place`: where the placement decides the prompt feed
/// ([`Place::steps_only`]), the feed is one decode step per id, and the
/// `call feed` record says so and why; elsewhere `cfg` as parsed and no
/// record.
pub fn feed_under(place: Place, cfg: &mut OpenCfg) -> Option<Record> {
    let why = place.steps_only()?;
    cfg.body.prefill = PrefillMode::Steps;
    Some(
        Record::new(&record::CALL_FEED)
            .w("feed", PrefillMode::Steps.name())
            .w("why", why),
    )
}

/// `r` with the cards `place` loaded (`cards`) and, when it has an expert
/// tier card, the tier's experts and resident bytes. A model whose tier
/// does not match the placement — none where the placement names one, one
/// where it names none, or a tier on another card — is refused by name.
pub fn with_cards(
    m: &Deepseek41Model,
    place: Place,
    what: &'static str,
    r: Record,
) -> Result<Record, GateError> {
    let cards = place.cards();
    let tier = m.body(what)?.hybrid().tier();
    let r = r.csv("cards", cards.iter());
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
