//! The gate plan's card, one owner for both tiers: the real tier plans on the card the gate runner
//! put in view — `workstation::plan_gate`'s machine, the 3090's bytes, so the plan and every pin a
//! gate reads from it are the 3090's — on whichever one card is visible (tools/gpu-gate.sh's pick
//! under `BLOOMERY_GATE_CARD`); the fixture tier plans on `a`, the largest visible card as the
//! device reports itself (`--place a`), its free bytes included. The real tier resolves the card as
//! the gate placement does (`Pick::Room(&RTX_3090)`, a server's `--place gate`): the 3090's total
//! and driver reserve on the device in view, which keeps its own name, identity, allocation granule
//! and whether it shares the host's memory, held as `tier::real_card` holds a census card. On the
//! 3090 the machine is `plan_gate`'s exactly, on any larger card it plans the 3090's usable bytes,
//! and a unified card plans the pool its free reading is. A census of other than one visible card,
//! or a card with less room than the 3090's bytes, is refused by name before any load. A two-card
//! arm plans its own machine with both cards in view and never reads the gate card
//! (`lcpp_e2e::Arm::machine`'s `bp`).
//!
//! A gate that needs the card's own room (its whole load of a file the 3090's bytes cannot hold)
//! reads [`room`]: the same card in view with its own total and driver reserve, in either tier,
//! and names the room it needs. It never names a workstation card to get it.

use std::sync::OnceLock;

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::tier;
use model::placement::Machine;
use model::placement::workstation::{self, CardSpec, DeviceInfo, Pick, RTX_3090};

static CARD: OnceLock<CardSpec> = OnceLock::new();

/// The census's card: the one card the runner put in view, planned by the 3090's bytes.
///
/// # Errors
/// A census of other than one visible card, or a card with less room than the 3090's bytes.
fn census() -> Result<CardSpec, GateError> {
    let census = one_visible()?;
    // PIN(2026-10-31): was a refusal outside `CARDS` and the 5090 by name; `Pick::Room` now
    // requires the 3090's bytes on the card in view and refuses a card with less room.
    Ok(tier::real_card(
        workstation::resolve(&[Pick::Room(&RTX_3090)], &census)?[0],
    ))
}

/// The census of the one card the runner put in view.
///
/// # Errors
/// A census of other than one visible card.
fn one_visible() -> Result<Vec<DeviceInfo>, GateError> {
    let census = bloomery_gpu::census()?;
    if census.len() != 1 {
        return Err(format!(
            "the gate plan runs on one visible card and {} are visible ({}): run it through \
             tools/gpu-gate.sh, which puts the card it takes in view",
            census.len(),
            workstation::visible(&census)
        )
        .into());
    }
    Ok(census)
}

/// The card in view with its own room, as the real tier holds a census card
/// ([`tier::real_card`]): its total and driver reserve, none of its free reading.
///
/// # Errors
/// A census of other than one visible card.
fn census_room() -> Result<CardSpec, GateError> {
    let census = one_visible()?;
    Ok(tier::real_card(
        workstation::resolve(&[Pick::Rank(0)], &census)?[0],
    ))
}

/// The card in view with its own room, for a clause that needs more bytes than the gate plan's
/// ([`card`]): the real tier's census card with none of its free reading, the fixture tier's `a`
/// as the device reports itself ([`tier::card`]). On the 3090 it is the 3090's bytes, on a larger
/// card that card's: a gate compares what it needs with `usable_bytes()` and leaves the clause by
/// name ([`tier::room_clause`]) where the card has less.
///
/// # Errors
/// A census of other than one visible card, or `a` finding no card.
#[allow(
    dead_code,
    reason = "the gates that need the card's own room read it; the gates that plan the 3090's \
              bytes reach the card through init"
)]
pub fn room() -> Result<CardSpec, GateError> {
    tier::card(census_room)
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
