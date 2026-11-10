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
    let now = free_now(gpu, WHAT)?;
    Ok(CardReading::Made(Box::new(CardMade::of(
        on.index, &on.card, &on.totals, phase, before, now,
    ))))
}

/// The device's free bytes on `gpu` once the engine stream is done
/// ([`host::ENGINE_BOUND`]): the one reading [`card_reading`] and
/// [`card_free`] take, so a stream-ordered allocation or free enqueued before
/// the call is in it.
fn free_now(gpu: &Gpu, what: &'static str) -> Result<u64, GpuError> {
    host::await_done(what, || gpu.stream().query())?;
    u64::try_from(gpu.mem_info()?.0).map_err(|_| GpuError::shape(what, "free bytes past u64"))
}

/// Card `on`'s free bytes now, read on `gpu` as [`card_reading`] reads them;
/// `None` where [`card_reading`] measures nothing (an integrated device, a
/// card another process held at plan time).
pub fn card_free(gpu: &Gpu, on: &PlanCard) -> Result<Option<u64>, GpuError> {
    if gpu.opened_free().is_none() || unmeasured_why(&on.card, false).is_some() {
        return Ok(None);
    }
    free_now(gpu, "card_free").map(Some)
}

/// The card bytes a load-time phase made: the free bytes before it less the
/// free bytes after it ([`card_free`] at each end), negative where the phase
/// freed more than it made; [`PhaseBytes::Unmeasured`] where either end was
/// not measured. Printed as the phase line's `bytes=` field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseBytes {
    Made(i128),
    Unmeasured,
}

impl PhaseBytes {
    /// The bytes made between a reading of `before` free bytes and one of
    /// `after`.
    #[must_use]
    pub fn between(before: Option<u64>, after: Option<u64>) -> PhaseBytes {
        match (before, after) {
            (Some(b), Some(a)) => PhaseBytes::Made(i128::from(b) - i128::from(a)),
            _ => PhaseBytes::Unmeasured,
        }
    }

    /// The sum of two phases' bytes: unmeasured when either is.
    #[must_use]
    pub fn plus(self, other: PhaseBytes) -> PhaseBytes {
        match (self, other) {
            (PhaseBytes::Made(a), PhaseBytes::Made(b)) => PhaseBytes::Made(a + b),
            _ => PhaseBytes::Unmeasured,
        }
    }
}

impl std::fmt::Display for PhaseBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhaseBytes::Made(b) => write!(f, "{b}"),
            PhaseBytes::Unmeasured => f.write_str("unmeasured"),
        }
    }
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
