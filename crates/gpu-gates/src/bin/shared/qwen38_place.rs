//! What a Qwen3.8 (qwen4exp) binary's placement decides beside its plan
//! (`bloomery_gpu_gates::generate::Place`): the family's input to the common
//! unset `--place` rule, and the experts a placement's tier card holds in its
//! plan — the unset rule's question — and a serving slot's context choice.
//! `generate_qwen3moe`, the Qwen3.8 serve seat and the routed-leg gate read
//! them here.

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::beside::{self, Beside};
use bloomery_gpu_gates::generate::{Place, PlaceWhy, TierRule, choose_ctx_of, short_in};
use bloomery_gpu_gates::residency38::{Seqs, reserve_checkpoints};
use model::arch::qwen35moe::place::{
    Experts, KERNEL_POSITIONS, MtpInputs, PlaceError, PlanInputs, Stack, machine_bp_on, tier_batch,
};
use model::placement::ctx_ask::{Chosen, Probe};
use model::placement::workstation::{DeviceInfo, TierBatchBytes};
use model::placement::{Machine, PlanLevers};

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

/// The resident sequences and the named reserves held beside a tier plan.
pub(crate) struct Residents {
    pub(crate) seqs: Seqs,
    pub(crate) held: Beside,
    pub(crate) page: u64,
}

/// Plan (b′)'s machine over `place`'s stage and tier card for `layers`
/// layers at ubatches of `ub` positions (`machine_bp_on`, `draft` the MTP
/// draft's card bytes on the stage card, `batch` the tier's prompt-batch
/// bytes), the tier card on the node the placement plans it on
/// (`Place::machine_with`): the machine every qwen4exp load, seat and probe
/// with a tier card plans over. A placement of other than two cards is
/// refused by name.
pub fn machine_bp38(
    place: Place,
    layers: Stack,
    ub: u64,
    draft: Option<u64>,
    batch: TierBatchBytes,
) -> Result<Machine, GateError> {
    place.machine_with(|stage, tiers| match tiers {
        &[tier] => Ok(machine_bp_on((stage, tier), layers, ub, draft, batch)),
        _ => Err(format!(
            "--place {}: plan (b′) takes a stage card and one tier card",
            place.name()
        )
        .into()),
    })
}

/// The experts `place`'s plan (b′) holds on its tier card at `ctx` positions
/// and ubatches of `ub` under `experts` and `levers` — the load `mtp` names
/// planning the file's MTP draft beside the target, else the plain plan of
/// `seqs`' resident sequences, the machine's host reserving the checkpoints
/// the load declares (`residency38::reserve_checkpoints` at `ctx`, as the
/// load's own machine does) — the unset rule's `tier_of`. The machine and
/// the plan are the loads' own ([`machine_bp38`],
/// `PlanInputs::plan_target`); a plan that refuses
/// (a host expert rule on a tier machine, an idle tier card, a context past
/// the card's, a host room under the NVMe tier's floor) is the caller's
/// error, which `Place::choose` answers with `a`. The stage card takes `held`'s named
/// reserves before the tier count is chosen, just as the serving plan does.
pub fn tier_experts(
    inputs: &PlanInputs,
    place: Place,
    (ctx, ub): (u64, u64),
    experts: Experts,
    levers: &PlanLevers,
    mtp: Option<&MtpInputs>,
    Residents { seqs, held, page }: Residents,
) -> Result<u64, GateError> {
    let slots = seqs.slots;
    let mut machine = machine_bp38(
        place,
        inputs.stack(),
        ub,
        mtp.map(|m| m.card_bytes_of(ctx, slots)).transpose()?,
        tier_batch(&inputs.spec, ub),
    )?;
    inputs.reserve_room(
        &mut machine,
        (ctx, u64::try_from(slots)?, mtp.is_some()),
        page,
    )?;
    beside::reserve_on(&mut machine, held);
    reserve_checkpoints(&mut machine, seqs, ctx)?;
    let plan = inputs.plan_target(&machine, (ctx, ub), levers, experts, mtp, slots)?;
    Ok(plan.tier_n_l.iter().flatten().sum())
}

/// A serving slot's context by the seats' common choice
/// (`generate::choose_ctx_of`) under the file's limit: its trained context
/// within `place::KERNEL_POSITIONS`. `asked` is the set total (or none) and
/// the slots; `holds` plans every slot at a slot context, its refusal a
/// [`PlaceError`] the choice reads for the memory it was short of.
#[allow(
    dead_code,
    reason = "generate_qwen3moe includes this file and runs one sequence at its own --ctx"
)]
pub fn ctx_choice(
    inputs: &PlanInputs,
    asked: (Option<usize>, usize),
    mut holds: impl FnMut(u64) -> Result<(), GateError>,
) -> Result<Chosen<()>, GateError> {
    choose_ctx_of(
        u64::from(inputs.spec.ctx_train),
        asked,
        (Some((KERNEL_POSITIONS, "KERNEL_POSITIONS")), None),
        |c| probe(holds(c)),
    )
}

/// A plan's outcome (`held`) as the common choices read it: a refusal short of a memory is
/// [`Probe::Short`], any other refusal the error.
#[allow(
    dead_code,
    reason = "generate_qwen3moe includes this file and runs one sequence at its own --ctx"
)]
pub fn probe(held: Result<(), GateError>) -> Result<Probe<()>, GateError> {
    match held {
        Ok(()) => Ok(Probe::Holds(())),
        Err(e) => short_in::<PlaceError>(&e).map(Probe::Short).ok_or(e),
    }
}
