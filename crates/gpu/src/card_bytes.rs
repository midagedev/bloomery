//! The card-byte invariant's reading: what a run made on the card a model
//! loaded on, from the device's free bytes when its `Gpu` opened
//! ([`Gpu::opened_free`]) to now, against what the model's plan counts on that
//! card ([`PlanCard`]). The comparison and its refusal's text are the
//! placement's ([`::model::placement::card_made`]); this module owns the
//! readings and the one refusal, [`refuse_card_past_plan`], which a run where
//! the card is held calls after the load, after the first prompt call and
//! after the first step. A run on a card it does not hold takes the reading
//! alone ([`card_reading`]) and prints it.
//!
//! Each reading is of this model's own card. A card of the same plan that
//! another `Gpu` holds (an expert tier card's) is read on that `Gpu`.

use ::model::placement::card_made::{
    CardMade, CardPhase, CardReading, CardUnmeasured, unmeasured_why,
};
use ::model::placement::{Card, CardTotals, Plan};

use crate::{Gpu, GpuError, host};

/// The plan's side of the card a model loaded on: its index among the plan's
/// cards, its machine card and its totals, kept from the load so every later
/// phase is checked against the plan it loaded by.
#[derive(Clone, Debug)]
pub struct PlanCard {
    pub index: usize,
    pub card: Card,
    pub totals: CardTotals,
}

impl PlanCard {
    /// Card `index` of `plan`; a plan with no such card is refused by name.
    pub fn of(plan: &Plan<'_>, index: usize) -> Result<PlanCard, GpuError> {
        const WHAT: &str = "PlanCard::of";
        let card = plan
            .machine
            .cards
            .get(index)
            .ok_or_else(|| GpuError::shape(WHAT, format!("the plan has no card {index}")))?;
        let totals = plan
            .cards
            .get(index)
            .ok_or_else(|| GpuError::shape(WHAT, format!("the plan totals no card {index}")))?;
        Ok(PlanCard {
            index,
            card: card.clone(),
            totals: totals.clone(),
        })
    }
}

/// Card `on`'s bytes at `phase`, read on `gpu` (the `Gpu` the model on that
/// card opened): the bytes made since the open against the plan's count, or,
/// on an integrated device or a card another process held at plan time, why
/// they are not measured. The engine stream is waited for first
/// ([`host::ENGINE_BOUND`]), so a stream-ordered allocation or free enqueued
/// before the call is in the reading.
pub fn card_reading(gpu: &Gpu, on: &PlanCard, phase: CardPhase) -> Result<CardReading, GpuError> {
    const WHAT: &str = "card_reading";
    let unmeasured = |why: String| {
        CardReading::Unmeasured(CardUnmeasured {
            card: on.index,
            name: on.card.name.clone(),
            phase,
            why,
        })
    };
    let Some(before) = gpu.opened_free() else {
        let why = unmeasured_why(&on.card, true)
            .ok_or_else(|| GpuError::shape(WHAT, "an integrated device with no reason"))?;
        return Ok(unmeasured(why));
    };
    if let Some(why) = unmeasured_why(&on.card, false) {
        return Ok(unmeasured(why));
    }
    host::await_done(WHAT, || gpu.stream().query())?;
    let now = u64::try_from(gpu.mem_info()?.0)
        .map_err(|_| GpuError::shape(WHAT, "free bytes past u64"))?;
    Ok(CardReading::Made(Box::new(CardMade::of(
        on.index, &on.card, &on.totals, phase, before, now,
    ))))
}

/// [`card_reading`], refused by name where the card is held: bytes made
/// past the plan's count (each term, the made bytes and the phase named), or
/// a card that cannot be measured. The reading within the plan is returned for
/// the caller's record.
pub fn refuse_card_past_plan(
    gpu: &Gpu,
    on: &PlanCard,
    phase: CardPhase,
) -> Result<CardReading, GpuError> {
    let reading = card_reading(gpu, on, phase)?;
    reading
        .check()
        .map_err(|e| GpuError::plan("refuse_card_past_plan", e))?;
    Ok(reading)
}
