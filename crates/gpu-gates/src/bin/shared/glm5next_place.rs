//! What a GLM-5.3-Flash binary's placement decides beside its plan
//! (`bloomery_gpu_gates::generate::Place`): the `--place` word the GLM body
//! serves, the expert tier's prompt-batch bytes the plan reserves, and the
//! experts a placement's tier cards hold in its plan — the unset rule's
//! question. `generate_glm5next` and the GLM seat of `bloomery-serve` read
//! them here.

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::{Place, TierRule};
use model::arch::glm5next::hparams::Hparams;
use model::arch::glm5next::place::{KdaLanes, NextnInputs, PlanInputs};
use model::placement::PlanLevers;
use model::placement::workstation::TierBatchBytes;

/// The expert tier cards the GLM body serves: the host tier's count
/// (`Body::open_placed_lanes` refuses past it before any upload).
const SERVED_TIERS: usize = bloomery_gpu::host::SERVED_TIERS;

/// The GLM family's input to the common unset rule (`Place::choose`): the
/// tier cards its body serves, and the tier kept only when the plan puts at
/// least this many experts on it — the tier size whose predicted decode gain
/// clears 0 (docs/cards/glmbp-ab.card).
pub const TIER_RULE: TierRule = TierRule {
    tiers: SERVED_TIERS,
    break_even: Some(526),
    basis: "docs/cards/glmbp-ab.card",
};

/// A GLM binary's `--place` word (`Place::parse`), refused by name before
/// any plan when it lists more tier cards than the GLM body serves.
pub fn parse(v: &str) -> Result<Place, GateError> {
    let place = Place::parse(v)?;
    place.serves("glm5next", SERVED_TIERS)?;
    Ok(place)
}

/// The expert tier's prompt-batch bytes `place`'s plan reserves for the file
/// of hyperparameters `hp` (`model::arch::glm5next::place::tier_batch`),
/// which `Place::machine` takes; `None` under a placement with no tier card.
pub fn tier_batch(place: Place, hp: &Hparams) -> Option<TierBatchBytes> {
    (!place.tier_cards().is_empty()).then(|| model::arch::glm5next::place::tier_batch(hp))
}

/// The experts `place`'s tier cards hold in its plan at `ctx` positions
/// under `levers` — the load `nextn` names planning the file's next-token
/// layer beside the target, else `lanes` KDA lanes serving `slots` resident
/// sequences — 0 with no tier card. `Place::choose`'s `tier_of`: the plan of
/// the offered placement, its tier counts summed, against the family's
/// break-even.
pub fn tier_experts(
    place: Place,
    inputs: &PlanInputs,
    ctx: u64,
    levers: &PlanLevers,
    nextn: Option<&NextnInputs>,
    lanes: KdaLanes,
    slots: usize,
) -> Result<u64, GateError> {
    let machine = place.machine(None, tier_batch(place, &inputs.hp))?(inputs.model.layers);
    let plan = match nextn {
        None => inputs.plan_slots(&machine, ctx, levers, lanes, slots)?,
        Some(n) => {
            inputs
                .plan_nextn_slots(&machine, ctx, levers, n, slots)?
                .plan
        }
    };
    Ok(plan.tier_n_l.iter().flatten().sum())
}
