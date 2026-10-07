//! The gate plan on the card the gate runner put in view: `workstation::plan_gate`'s machine — the
//! 3090's bytes, so the plan and every pin a gate reads from it are the 3090's — on whichever one
//! card is visible (tools/gpu-gate.sh's pick under `BLOOMERY_GATE_CARD`). The card spec keeps every
//! byte of `RTX_3090` and takes the visible card's name, by which the open finds the device, so on
//! the 3090 the machine is `plan_gate`'s exactly and on the A6000 it plans the 3090's usable bytes.
//! A census of other than one visible card, or a card no name of `CARDS` matches, is refused by
//! name before any load.

use std::sync::OnceLock;

use bloomery_gpu_gates::GateError;
use model::placement::Machine;
use model::placement::workstation::{self, CARDS, CardSpec, RTX_3090};

static CARD: OnceLock<CardSpec> = OnceLock::new();

/// Reads the census once and fixes the card [`plan_gate`] plans on.
pub fn init() -> Result<CardSpec, GateError> {
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
    Ok(*CARD.get_or_init(|| CardSpec { name, ..RTX_3090 }))
}

/// The card [`init`] fixed, reading the census on the first call.
#[allow(
    dead_code,
    reason = "the Qwen3.8 gates plan through their own machine on this card; the V4.1 and GLM gates through plan_gate"
)]
pub fn card() -> Result<CardSpec, GateError> {
    match CARD.get() {
        Some(card) => Ok(*card),
        None => init(),
    }
}

/// `workstation::plan_gate` on the card [`init`] fixed.
#[allow(
    dead_code,
    reason = "the V4.1 and GLM gates plan through this; the Qwen3.8 gates through their own machine on card()"
)]
pub fn plan_gate(layers: usize) -> Machine {
    let card = CARD
        .get()
        .expect("gate_card::init runs before the gate plan is made");
    workstation::plan_on(*card, layers)
}
