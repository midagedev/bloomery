//! What a Qwen3.8 (qwen4exp) binary's placement decides beside its plan
//! (`bloomery_gpu_gates::generate::Place`): the family's input to the common
//! unset `--place` rule, and the experts a placement's tier card holds in its
//! plan — the unset rule's question. `generate_qwen3moe` and the Qwen3.8
//! serve seat read them here.

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::{Place, PlaceWhy, TierRule};
use bloomery_gpu_gates::residency38::{Seqs, reserve_checkpoints};
use model::arch::qwen35moe::place::{Experts, MtpInputs, PlanInputs, machine_bp_on, tier_batch};
use model::placement::PlanLevers;
use model::placement::workstation::{CardSpec, DeviceInfo};

/// The Qwen3.8 family's input to the common unset rule (`Place::choose`):
/// one expert tier card served, the tier count a measured sitting has shown
/// pays — the least whose predicted decode gain clears the ruler's band
/// (docs/cards/q38bpbug-ab.card holds the sitting and the derivation).
pub const Q38_RULE: TierRule = TierRule {
    tiers: 1,
    break_even: Some(1_152),
    basis: "docs/cards/q38bpbug-ab.card",
};

/// The placement a Qwen3.8 binary's `--place` names, and why, on `census`
/// (`Place::choose` by [`Q38_RULE`]): `flag` as given when set, else the
/// cards' offer kept by the rule — `tier_of` names the experts the offer's
/// plan holds on its tier card ([`tier_experts`]), a plan that refuses
/// running `a` with the refusal named on stderr. The caller prints
/// [`PlaceWhy::record`].
pub fn choose(
    flag: Option<Place>,
    census: &[DeviceInfo],
    tier_of: impl FnOnce(Place) -> Result<u64, GateError>,
) -> Result<PlaceWhy, GateError> {
    Place::choose(flag, census, Q38_RULE, tier_of)
}

/// The experts `cards`' plan (b′) holds on its tier card at `ctx` positions
/// and ubatches of `ub` under `experts` and `levers` — the load `mtp` names
/// planning the file's MTP draft beside the target, else the plain plan of
/// `seqs`' resident sequences, the machine's host reserving the checkpoints
/// the load declares (`residency38::reserve_checkpoints`, as the load's own
/// machine does) — the unset rule's `tier_of`. The plan is the model crate's
/// one owner of both (`machine_bp_on`,
/// `PlanInputs::plan_with_slots`/`plan_mtp_with_slots`); a plan that refuses
/// (a host expert rule on a tier machine, an idle tier card, a context past
/// the card's, a host room under the NVMe tier's floor) is the caller's
/// error, which `Place::choose` answers with `a`.
pub fn tier_experts(
    inputs: &PlanInputs,
    cards: (CardSpec, CardSpec),
    (ctx, ub): (u64, u64),
    experts: Experts,
    levers: &PlanLevers,
    mtp: Option<&MtpInputs>,
    seqs: Seqs,
) -> Result<u64, GateError> {
    let slots = seqs.slots;
    let mut machine = machine_bp_on(
        cards,
        inputs.spec.layers.len(),
        ub,
        mtp.map(|m| m.card_bytes_of(ctx, slots)).transpose()?,
        tier_batch(&inputs.hp, ub),
    );
    reserve_checkpoints(&mut machine, seqs);
    let plan = match mtp {
        None => inputs.plan_with_slots(&machine, ctx, levers, experts, slots)?,
        Some(mi) => {
            inputs
                .plan_mtp_with_slots(&machine, ctx, levers, mi, experts, slots)?
                .plan
        }
    };
    Ok(plan.tier_n_l.iter().flatten().sum())
}
