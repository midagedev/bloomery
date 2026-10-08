//! The gate plan's card, one owner for both tiers: the real tier plans on the card the gate runner
//! put in view — `workstation::plan_gate`'s machine, the 3090's bytes, so the plan and every pin a
//! gate reads from it are the 3090's — on whichever one card is visible (tools/gpu-gate.sh's pick
//! under `BLOOMERY_GATE_CARD`); the fixture tier plans on `a`, the largest visible card as the
//! device reports itself (`--place a`), its free bytes included. The card spec keeps every byte of
//! `RTX_3090` in the real tier and takes the visible card's name, by which the open finds the
//! device, so on the 3090 the machine is `plan_gate`'s exactly and on the A6000 it plans the
//! 3090's usable bytes. A census of other than one visible card, or a card no name of `CARDS`
//! matches, is refused by name before any load.

use std::sync::OnceLock;

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::tier;
use model::placement::Machine;
use model::placement::workstation::{self, CARDS, CardSpec, RTX_3090};

static CARD: OnceLock<CardSpec> = OnceLock::new();

/// The census's card: the one card the runner put in view, planned by the 3090's bytes under its
/// own name.
///
/// # Errors
/// A census of other than one visible card, or a card none of `CARDS` matches.
fn census() -> Result<CardSpec, GateError> {
    let census = bloomery_gpu::census()?;
    let [device] = census.as_slice() else {
        return Err(format!(
            "the gate plan runs on one visible card and {} are visible ({}): run it through \
             tools/gpu-gate.sh, which puts the card it takes in view",
            census.len(),
            workstation::visible(&census)
        )
        .into());
    };
    let name = CARDS
        .iter()
        .find(|c| device.name.contains(c.name))
        .map(|c| c.name)
        .ok_or_else(|| {
            format!(
                "the visible card {} is none of this workstation's cards ({}): the gate plan \
                 opens its card by name",
                device.name,
                CARDS.map(|c| c.name).join(", ")
            )
        })?;
    Ok(CardSpec { name, ..RTX_3090 })
}

/// Reads the census (the real tier) or resolves `a` (the fixture tier) once and fixes the card
/// [`plan_gate`] plans on, printing its move proof (`tier::witness_card`). The first call decides
/// and proves; a later call returns the fixed card.
///
/// # Errors
/// The census's or `a`'s refusal, or a real tier's card spec that is not the 3090's bytes.
pub fn init() -> Result<CardSpec, GateError> {
    if let Some(card) = CARD.get() {
        return Ok(*card);
    }
    let card = tier::card(census)?;
    if !tier::witness_card(&card) {
        return Err("the gate card's bytes are not RTX_3090's in the real tier".into());
    }
    Ok(*CARD.get_or_init(|| card))
}

/// The card [`init`] fixed, reading the census on the first call.
#[allow(
    dead_code,
    reason = "iqleg plans on it and the Qwen3.8 e2e and mtp gates pass it as tier::card's real \
              arm; the gates that plan through plan_gate reach it there"
)]
pub fn card() -> Result<CardSpec, GateError> {
    match CARD.get() {
        Some(card) => Ok(*card),
        None => init(),
    }
}

/// The gate placement on the card [`init`] fixed: every layer and the head on it, beside the host.
///
/// # Panics
/// Before [`init`].
#[allow(
    dead_code,
    reason = "the callstream, twocard and Qwen3.8 gates plan on their own machines; the V4.1, \
              GLM, mimo2 and long gates plan through this"
)]
pub fn plan_gate(layers: usize) -> Machine {
    let card = CARD
        .get()
        .expect("gate_card::init runs before the gate plan is made");
    workstation::plan_on(*card, layers)
}
