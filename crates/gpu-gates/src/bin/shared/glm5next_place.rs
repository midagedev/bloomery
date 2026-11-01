//! What a GLM-5.3-Flash binary's placement decides beside its plan
//! (`bloomery_gpu_gates::generate::Place`): the `--place` word the GLM body
//! serves, the expert tier's prompt-batch bytes the plan reserves, and the
//! experts a placement's tier cards hold in its plan — the unset rule's
//! question. `generate_glm5next` and the GLM seat of `bloomery-serve` read
//! them here.

use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::beside;
use bloomery_gpu_gates::bind::place_word;
use bloomery_gpu_gates::generate::{Place, TierRule, choose_ctx_of};
use bloomery_gpu_gates::residency38::glm_machine;
use bloomery_gpu_glm5next::checkpoint_price;
use model::arch::glm5next::place::{
    KdaLanes, NextnInputs, ORACLE_POSITIONS, PlaceError, PlanInputs,
};
use model::placement::ctx_ask::{Chosen, Probe, probe_of};
use model::placement::workstation::TierBatchBytes;
use model::placement::{Machine, Plan, PlanLevers};

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

/// A GLM binary's `--place` word (`bind::place_word`), refused by name before
/// any plan when it lists more tier cards than the GLM body serves.
pub fn parse(v: &str) -> Result<Place, GateError> {
    place_word(v, Some(("glm5next", SERVED_TIERS)))
}

/// The expert tier's prompt-batch bytes `place`'s plan reserves for the file
/// described by `inputs` (`model::arch::glm5next::place::tier_batch`),
/// which `Place::machine` takes; `None` under a placement with no tier card.
pub fn tier_batch(place: Place, inputs: &PlanInputs) -> Result<Option<TierBatchBytes>, GateError> {
    Ok((!place.tier_cards().is_empty())
        .then(|| model::arch::glm5next::place::tier_batch(inputs))
        .transpose()?)
}

/// The experts `place`'s tier cards hold in its plan at `ctx` positions
/// under `levers` — the load `nextn` names planning the file's next-token
/// layer beside the target, else `lanes` KDA lanes serving `slots` resident
/// sequences — 0 with no tier card. `Place::choose`'s `tier_of`: the plan of
/// the offered placement on the machine the load takes (`glm_machine`, its
/// slots' checkpoints at `ctx` declared), its tier counts summed, against
/// the family's break-even.
pub fn tier_experts(
    place: Place,
    inputs: &PlanInputs,
    ctx: u64,
    levers: &PlanLevers,
    nextn: Option<&NextnInputs>,
    lanes: KdaLanes,
    slots: usize,
) -> Result<u64, GateError> {
    let machine = glm_machine(
        place,
        tier_batch(place, inputs)?,
        slots,
        checkpoint_price(inputs)?,
        ctx,
    )?(inputs.model.layers);
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

/// The target's plan of `slots` resident sequences at `ctx` positions a slot
/// on the load `nextn` names: with the next-token layer
/// (`PlanInputs::plan_nextn_slots`, two KDA lanes) or without it
/// (`PlanInputs::plan_slots` at one lane) — the plans the load runs. The
/// refusal is the plan's own ([`named`] words it for a seat).
///
/// # Errors
/// The plan's refusal.
#[allow(
    dead_code,
    reason = "the serve seat and the routed legs plan with it; the CLI opens its own"
)]
pub fn plan_of<'a>(
    inputs: &'a PlanInputs,
    machine: &'a Machine,
    ctx: u64,
    levers: &PlanLevers,
    nextn: Option<&'a NextnInputs>,
    slots: usize,
) -> Result<Plan<'a>, PlaceError> {
    match nextn {
        None => inputs.plan_slots(machine, ctx, levers, KdaLanes::One, slots),
        Some(n) => inputs
            .plan_nextn_slots(machine, ctx, levers, n, slots)
            .map(|with| with.plan),
    }
}

/// A plan refusal on `machine` as the seat names it: the image encoder's
/// reserve named when the machine carries one ([`beside::refusal`]).
#[allow(
    dead_code,
    reason = "the serve seat names its refusals; the CLI and the routed legs read them bare"
)]
pub fn named(machine: &Machine, e: PlaceError) -> GateError {
    beside::refusal(beside::reserve_in(machine), e)
}

/// The serve seat's context, every number a slot's own and every plan
/// counting the `asked` slots ([`plan_of`]), by the seats' common choice
/// (`generate::choose_ctx_of`): the limit is the trained context, verified up
/// to `ORACLE_POSITIONS`; a set share past it stops by name, and one the
/// machine cannot hold stops with the largest it would hold; unset, a slot
/// asks the target and halves while the plan is short. Each probe plans the
/// machine laid at its context (`machine_at`, [`probe`]). The choice, whose
/// `ctx` line the caller prints.
///
/// # Errors
/// As `generate::choose_ctx_of`.
#[allow(
    dead_code,
    reason = "the serve seat and the routed legs choose with it; the CLI takes its own --ctx"
)]
pub fn serve_ctx(
    inputs: &PlanInputs,
    machine_at: &dyn Fn(u64) -> Result<Machine, GateError>,
    levers: &PlanLevers,
    nextn: Option<&NextnInputs>,
    asked: (Option<usize>, usize),
) -> Result<Chosen<()>, GateError> {
    choose_ctx_of(
        u64::try_from(inputs.hp.n_ctx_train)?,
        asked,
        (None, Some((ORACLE_POSITIONS, "ORACLE_POSITIONS"))),
        |c| probe(inputs, &machine_at(c)?, (c, asked.1), levers, nextn),
    )
}

/// The plan of `slots` sequences of `ctx` positions on `machine` ([`plan_of`]), as the
/// context choice and `--parallel auto`'s count read it: a refusal short of a memory is
/// [`Probe::Short`], any other the seat's, named by the machine's reserves ([`named`]).
#[allow(
    dead_code,
    reason = "the serve seat and the routed legs choose with it; the CLI takes its own --ctx"
)]
pub fn probe(
    inputs: &PlanInputs,
    machine: &Machine,
    (ctx, slots): (u64, usize),
    levers: &PlanLevers,
    nextn: Option<&NextnInputs>,
) -> Result<Probe<()>, GateError> {
    probe_of(plan_of(inputs, machine, ctx, levers, nextn, slots).map(|_| ()))
        .map_err(|e| named(machine, e))
}
