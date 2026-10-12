//! glm5next (GLM-5.3-Flash): the two `--place`-unset production defaults that
//! plan a glm5next file, as `generate_glm5next` and the GLM seat of
//! `bloomery-serve` call the planner.
//!
//! The CLI's is `Open::plan` (`PlanInputs::plan`, one KDA lane) at its 2048
//! positions for one sequence, on `glm_machine` of the placement its unset
//! rule picks (`Place::choose` by `glm_place::TIER_RULE`) with the slot's
//! checkpoints declared. The seat's is `Placed::of` for two slots: the
//! unset draft rule (`glm_unset`: the NextN draft on a file of exactly one
//! next-token layer), the placement of the same rule, the slot fit
//! (`default_slots_fit`: the default's second slot that does not plan falls
//! to one), the `--ctx` search (`default_slot_ctx` at the seat's base and
//! step over the plans' expert bytes) and the load's plan at the context it
//! answered (`plan_nextn_slots` with the draft, `plan_slots` without). The
//! draft's yield to a context under the base (the seat's second search) and
//! the image encoder's reserve are not mirrored: they move a context or a
//! reserve, not which layers the leg runs. The forced rooms go through the
//! same plan call at the arm's context and slots.

use std::sync::Arc;

use super::glmplace;
use super::{Adapter, Arm, Outcome, Refusal, census, outcome_of, probed, refused_of, view_of};
use app::mtp::MtpBody;
use bloomery_gpu::GpuError;
use bloomery_gpu_gates::bind::{SlotsFrom, default_slots_fit};
use bloomery_gpu_gates::generate::Place;
use bloomery_gpu_gates::residency38::glm_machine;
use bloomery_gpu_glm5next::{Body, checkpoint_price};
use bloomery_levers::{GlmAt, glm_unset};
use gguf::Split;
use model::arch::glm5next::place::{
    KdaLanes, NextnInputs, ORACLE_POSITIONS, PlaceError, PlanInputs,
};
use model::placement::ctx::{ExpertBytes, default_slot_ctx};
use model::placement::workstation::MARGIN;
use model::placement::{Machine, Plan, PlanLevers};

/// The CLI's default context (`generate_glm5next`'s `--ctx` unset).
const CLI_CTX: usize = 2048;

/// The seat's base context and the step its search moves in
/// (`serve_seats/glm.rs`'s own constants).
const SEAT_CTX: usize = 2048;
const SEAT_CTX_STEP: usize = 256;

/// The serve seat's resident sequences, `--parallel` unset.
const SERVE_PARALLEL: usize = 2;

/// The positions the seat's loop needs for one drafted window
/// (`serve_seats/glm.rs`'s `NEED`).
const NEED: usize = <Body as MtpBody>::VERIFY_ROWS + 1;

/// The family's adapter, one row of the gate's dispatch.
pub(super) struct ADAPTER;

impl Adapter for ADAPTER {
    fn name(&self) -> &'static str {
        "glm5next"
    }

    fn arms(&self, split: &Split) -> Vec<Result<Arm, Outcome>> {
        match read(split) {
            Ok(inputs) => {
                let inputs = Arc::new(inputs);
                vec![cli_arm(&inputs), serve_arm(&inputs)]
            }
            // The file is read once; a refusal of it is each arm's.
            Err(refusal) => vec![Err(refusal.clone()), Err(refusal)],
        }
    }
}

/// The CLI's default: one sequence at [`CLI_CTX`], no draft.
fn cli_arm(inputs: &Arc<PlanInputs>) -> Result<Arm, Outcome> {
    let ctx = u64::try_from(CLI_CTX).expect("the CLI's context is a u64");
    let place = choose(inputs, ctx, None, 1)?;
    let machine = machine(place, inputs, ctx, 1)?;
    let unbudgeted = planned(inputs, &machine, ctx, &PlanLevers::default(), None, 1);
    let inputs = Arc::clone(inputs);
    Ok(Arm {
        name: "cli",
        whole: false,
        unbudgeted: Some(unbudgeted),
        placed: Box::new(move |levers| planned(&inputs, &machine, ctx, levers, None, 1)),
    })
}

/// The serve seat's default: the draft the unset rule picks, the slot fit
/// from two, the context its search answers, the load's plan there.
fn serve_arm(inputs: &Arc<PlanInputs>) -> Result<Arm, Outcome> {
    let draft = glm_unset(GlmAt {
        serving_place: true,
        nextn_layers: inputs.hp.n_layer.saturating_sub(inputs.hp.n_trunk),
        need: NEED,
        ctx: SEAT_CTX,
        prefill_steps: false,
    })
    .draft;
    let nextn = (draft.word == "mtp")
        .then(|| NextnInputs::read(inputs))
        .transpose()
        .map_err(|e| refused_of(&e))?;
    let base = u64::try_from(SEAT_CTX).expect("the seat's base context is a u64");
    let place = choose(inputs, base, nextn.as_ref(), SERVE_PARALLEL)?;
    let (found, slots, _) = default_slots_fit(
        "gate_routed_legs",
        place.name(),
        SERVE_PARALLEL,
        SlotsFrom::Default,
        |slots| fit(place, inputs, nextn.as_ref(), slots),
    )?;
    let (machine, ctx) = found;
    let unbudgeted = planned(
        inputs,
        &machine,
        ctx,
        &PlanLevers::default(),
        nextn.as_ref(),
        slots,
    );
    let inputs = Arc::clone(inputs);
    Ok(Arm {
        name: "serve",
        whole: false,
        unbudgeted: Some(unbudgeted),
        placed: Box::new(move |levers| {
            planned(&inputs, &machine, ctx, levers, nextn.as_ref(), slots)
        }),
    })
}

/// The file read as the CLI and the seat read it (`PlanInputs::read`: the
/// coverage refusal lists every feature the engine does not run).
fn read(split: &Split) -> Result<PlanInputs, Outcome> {
    PlanInputs::read(split).map_err(|e| refused_of(&e))
}

/// The family's unset `--place` (`Place::choose` by `TIER_RULE`) on the
/// census the gate sees: the tier's plan is asked only of a two-card offer,
/// which the gate's one visible card never makes.
fn choose(
    inputs: &PlanInputs,
    ctx: u64,
    nextn: Option<&NextnInputs>,
    slots: usize,
) -> Result<Place, Outcome> {
    Place::choose(None, census()?, glmplace::TIER_RULE, |offer| {
        glmplace::tier_experts(
            offer,
            inputs,
            ctx,
            &PlanLevers::default(),
            nextn,
            KdaLanes::One,
            slots,
        )
    })
    .map(|chosen| chosen.place)
    .map_err(|e| outcome_of(e.as_ref()))
}

/// The machine a load of `slots` sequences of `ctx` positions plans on:
/// `glm_machine` of `place`, its slots' checkpoints declared.
fn machine(place: Place, inputs: &PlanInputs, ctx: u64, slots: usize) -> Result<Machine, Outcome> {
    let price = checkpoint_price(inputs).map_err(gpu_red)?;
    let tier_batch = glmplace::tier_batch(place, &inputs.hp);
    let make =
        glm_machine(place, tier_batch, slots, price, ctx).map_err(|e| outcome_of(e.as_ref()))?;
    Ok(make(inputs.model.layers))
}

/// The seat's `Placed::of` for `slots` sequences: the context its search
/// answers (`default_slot_ctx` over the plans' expert bytes, each probe on
/// the machine laid for its own context) and the machine at that context.
fn fit(
    place: Place,
    inputs: &PlanInputs,
    nextn: Option<&NextnInputs>,
    slots: usize,
) -> Result<(Machine, u64), Outcome> {
    let expert_bytes = |ctx: usize| -> Result<ExpertBytes, Outcome> {
        probed();
        let ctx = u64::try_from(ctx).map_err(|e| Outcome::Red(format!("a probed context: {e}")))?;
        let machine = machine(place, inputs, ctx, slots)?;
        let plan = plan_of(inputs, &machine, ctx, &PlanLevers::default(), nextn, slots)?;
        let card = plan.cards.first().ok_or_else(|| {
            Outcome::Red("a plan with no card, where the seat's search reads one".to_owned())
        })?;
        Ok(ExpertBytes {
            card: card.expert_bytes,
            nvme: plan.host.nvme_expert_bytes,
            arena: plan.host.nvme_arena_bytes,
        })
    };
    let oracle = usize::try_from(ORACLE_POSITIONS)
        .map_err(|e| Outcome::Red(format!("the oracle's positions: {e}")))?;
    let cap = inputs.hp.n_ctx_train.min(oracle);
    let rule = default_slot_ctx(cap, SEAT_CTX, SEAT_CTX_STEP, MARGIN, &expert_bytes)
        .map_err(|refused| refused.source)?;
    let ctx =
        u64::try_from(rule.ctx).map_err(|e| Outcome::Red(format!("the seat's context: {e}")))?;
    Ok((machine(place, inputs, ctx, slots)?, ctx))
}

/// The plan `levers` ask of `inputs` on `machine` for `slots` resident
/// sequences, as the gate reads it.
fn planned(
    inputs: &PlanInputs,
    machine: &Machine,
    ctx: u64,
    levers: &PlanLevers,
    nextn: Option<&NextnInputs>,
    slots: usize,
) -> Outcome {
    match plan_of(inputs, machine, ctx, levers, nextn, slots) {
        Ok(plan) => Outcome::Plan(view_of(&plan)),
        Err(outcome) => outcome,
    }
}

/// [`planned`]'s plan itself: the load's own call, with the NextN layer
/// (`plan_nextn_slots`, two KDA lanes) or without it (`plan_slots`, one).
fn plan_of<'a>(
    inputs: &'a PlanInputs,
    machine: &'a Machine,
    ctx: u64,
    levers: &PlanLevers,
    nextn: Option<&'a NextnInputs>,
    slots: usize,
) -> Result<Plan<'a>, Outcome> {
    match nextn {
        None => inputs.plan_slots(machine, ctx, levers, KdaLanes::One, slots),
        Some(n) => inputs
            .plan_nextn_slots(machine, ctx, levers, n, slots)
            .map(|with| with.plan),
    }
    .map_err(|e| place_of(&e))
}

/// A plan refusal as an outcome: the placement's own are named by
/// [`refused_of`], a context past the oracle and a front the card cannot
/// hold are named here, an invariant break or a machine of the wrong shape
/// is red.
fn place_of(p: &PlaceError) -> Outcome {
    let kind = match p {
        PlaceError::Placement(e) => return refused_of(e),
        PlaceError::PastOracle { .. } => "past-oracle",
        PlaceError::FrontOver { .. } => "front-over",
        PlaceError::Broken(_) | PlaceError::NextnCards { .. } | PlaceError::Slots { .. } => {
            return Outcome::Red(p.to_string());
        }
    };
    Outcome::Refused(Refusal {
        kind,
        floor: None,
        text: p.to_string(),
    })
}

/// A `GpuError` as a red outcome: the checkpoint price's read is not a
/// planner refusal.
fn gpu_red(e: GpuError) -> Outcome {
    Outcome::Red(e.to_string())
}
