//! qwen4exp (Qwen3.8): the two `--place`-unset production defaults that
//! plan a qwen4exp file, as `generate_qwen3moe` and the Qwen3.8 serve seat
//! call the planner. The unset placement is the family's common rule
//! (`q38place::choose` by `Q38_RULE`, the tier count asked of the offer's
//! plan — a question the gate's one-card census never reaches, the offer
//! being `a` alone). The plan is `PlanInputs::plan_with_slots` under
//! [`Experts::Card`] (`BLOOMERY_QWEN38_EXPERTS` unset) on the machine the
//! load lays out: the CLI's at the CLI's 4096 for one sequence with no
//! checkpoints (`machine38`'s one-card arm), the seat's at the context its
//! unset `--ctx` takes (`q38place::ctx_choice`, the seats' fit) for its
//! default slots, whose checkpoints are priced (`checkpoint_price38` of the
//! file's delta layers) and reserved.
//! Drafts are off in both (the draft levers unset).
//!
//! The loader's plan-time stack table (`swap38::stacks`, L6) is asked of every
//! plan with the plan in hand: it checks each layer the plan gives card
//! experts, on the stage card or a tier, against the body's own arms.

use std::sync::Arc;

use super::q38place;
use super::{Adapter, Arm, Outcome, census, floor_or, outcome_of, probed, view_of};
use bloomery_gpu::GpuError;
use bloomery_gpu::arch::qwen3moe::checkpoint_price38;
use bloomery_gpu::arch::qwen3moe::swap38;
use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::{Place, stage_page};
use bloomery_gpu_gates::residency38::{Seqs, reserve_checkpoints};
use gguf::Split;
use model::arch::models::Mixer;
use model::arch::qwen35moe::place::{Experts, PlanInputs, machine_for_experts, tier_batch};
use model::placement::ctx_ask::DEFAULT_SLOTS;
use model::placement::{Machine, Plan, PlanLevers};

/// The CLI's default context (`generate_qwen3moe`'s `--ctx` unset).
const CLI_CTX: usize = 4096;

/// The context the seat asks its placement at (`serve_seats/qwen38.rs`'s
/// `CTX`), before the context is chosen.
const SEAT_CTX: usize = 4096;

/// The family's adapter, one row of the gate's dispatch.
pub(super) struct Row;

impl Adapter for Row {
    fn name(&self) -> &'static str {
        "qwen38"
    }

    fn arms(&self, split: &Split) -> Vec<Result<Arm, Outcome>> {
        match describe(split) {
            Ok(inputs) => {
                let inputs = Arc::new(inputs);
                vec![cli_arm(&inputs), serve_arm(&inputs)]
            }
            // The file is read once; a refusal of it is each arm's.
            Err(refusal) => vec![Err(refusal.clone()), Err(refusal)],
        }
    }

    fn loader_table(&self) -> Option<&'static str> {
        Some("swap38::stacks")
    }
}

/// The CLI's default: the unset placement by the family's rule, the plan of
/// one sequence at the CLI's 4096.
fn cli_arm(inputs: &Arc<PlanInputs>) -> Result<Arm, Outcome> {
    let ctx = u64::try_from(CLI_CTX).expect("the CLI's context is a u64");
    let ub = ub_of(CLI_CTX)?;
    let place = choose(inputs, ctx, Seqs::none(1))?;
    let machine = machine(place, inputs, (ctx, ub), Seqs::none(1))?;
    let unbudgeted = planned(inputs, &machine, ctx, &PlanLevers::default(), 1);
    let inputs = Arc::clone(inputs);
    Ok(Arm {
        name: "cli",
        whole: false,
        unbudgeted: Some(unbudgeted),
        placed: Box::new(move |levers| planned(&inputs, &machine, ctx, levers, 1)),
    })
}

/// The serve seat's default: the unset placement by the family's rule, its
/// unset `--ctx`'s choice for the default checkpointing slots, then the
/// load's plan at the context it took.
fn serve_arm(inputs: &Arc<PlanInputs>) -> Result<Arm, Outcome> {
    let seqs = seqs38(inputs, DEFAULT_SLOTS);
    let ctx = u64::try_from(SEAT_CTX).expect("the seat's base context is a u64");
    let place = choose(inputs, ctx, seqs)?;
    let ctx = floor_or(
        q38place::ctx_choice(inputs, (None, DEFAULT_SLOTS), |c| {
            probe(inputs, place, seqs, c)
        })
        .map(|chosen| chosen.slot_ctx),
    )
    .map_err(|e| outcome_of(e.as_ref()))?;
    let ub =
        ub_of(usize::try_from(ctx).map_err(|e| Outcome::Red(format!("the seat's context: {e}")))?)?;
    let machine = machine(place, inputs, (ctx, ub), seqs)?;
    let unbudgeted = planned(inputs, &machine, ctx, &PlanLevers::default(), DEFAULT_SLOTS);
    let inputs = Arc::clone(inputs);
    Ok(Arm {
        name: "serve",
        whole: false,
        unbudgeted: Some(unbudgeted),
        placed: Box::new(move |levers| planned(&inputs, &machine, ctx, levers, DEFAULT_SLOTS)),
    })
}

/// One probe of the seat's context choice: whether the plan at a slot
/// context `c` holds, or why not.
fn probe(inputs: &PlanInputs, place: Place, seqs: Seqs, c: u64) -> Result<(), GateError> {
    probed();
    let ub = ub_of(usize::try_from(c)?).map_err(text_of)?;
    let machine = machine(place, inputs, (c, ub), seqs).map_err(text_of)?;
    inputs.plan_with_slots(
        &machine,
        (c, ub),
        &PlanLevers::default(),
        Experts::Card,
        seqs.slots,
    )?;
    Ok(())
}

/// The family's unset `--place` (`q38place::choose` by `Q38_RULE`): the
/// cards' offer kept by the rule, the tier count asked of the offer's plan
/// at the run's context — the CLI's plain plan (`Seqs::none` of its slots)
/// or the seat's checkpointing ones. Under the gate's one-card census the
/// offer is `a` alone and the question is never asked.
fn choose(inputs: &PlanInputs, ctx: u64, seqs: Seqs) -> Result<Place, Outcome> {
    let at = usize::try_from(ctx).expect("the run's context is a usize");
    let ub = ub_of(at)?;
    q38place::choose(None, census()?, |offer| {
        q38place::tier_experts(
            inputs,
            offer,
            (ctx, ub),
            Experts::Card,
            &PlanLevers::default(),
            // The default runs carry no MTP draft: the draft levers
            // unset.
            None,
            q38place::Residents {
                page: stage_page(offer.card_specs()?[0])?,
                seqs,
                held: bloomery_gpu_gates::beside::Beside::default(),
            },
        )
    })
    .map(|chosen| chosen.place)
    .map_err(|e| outcome_of(e.as_ref()))
}

/// The file described by the family's reader (`PlanInputs::describe`, the
/// host's room its own reading).
fn describe(split: &Split) -> Result<PlanInputs, Outcome> {
    PlanInputs::describe(split).map_err(|e| outcome_of(&e))
}

/// The ubatch a load of `ctx` positions runs (`ubatch_for`: the default
/// size clipped to the cache).
fn ub_of(ctx: usize) -> Result<u64, Outcome> {
    ubatch_for(ctx)
        .map_err(gpu_red)
        .and_then(|ub| u64::try_from(ub).map_err(|e| Outcome::Red(format!("the ubatch: {e}"))))
}

/// The machine a qwen4exp plan runs on over `place`'s cards
/// (`generate_qwen3moe`'s `machine38` and the seat's `Place38::machine`):
/// the one-card plan (a)'s `machine_for_experts` under card experts, plan
/// (b′)'s `q38place::machine_bp38`, the host reserving what `seqs` pin
/// (`reserve_checkpoints` at the context).
fn machine(
    place: Place,
    inputs: &PlanInputs,
    terms: (u64, u64),
    seqs: Seqs,
) -> Result<Machine, Outcome> {
    let (ctx, ub) = terms;
    let layers = inputs.stack();
    let cards = place.card_specs().map_err(|e| outcome_of(e.as_ref()))?;
    let mut machine = match cards.as_slice() {
        &[stage] => machine_for_experts(stage, layers, ub, Experts::Card),
        &[_, _] => q38place::machine_bp38(place, layers, ub, None, tier_batch(&inputs.spec, ub))
            .map_err(|e| outcome_of(e.as_ref()))?,
        cards => {
            return Err(Outcome::Red(format!(
                "the placement {} names {} cards; a qwen4exp plan takes a stage and at most \
                     one tier",
                place.name(),
                cards.len()
            )));
        }
    };
    let page = stage_page(cards[0]).map_err(|e| outcome_of(e.as_ref()))?;
    let slots = u64::try_from(seqs.slots).map_err(|e| Outcome::Red(e.to_string()))?;
    let drafts = false;
    inputs
        .reserve_room(&mut machine, (ctx, slots, drafts), page)
        .map_err(|e| outcome_of(&e))?;
    reserve_checkpoints(&mut machine, seqs, ctx).map_err(|e| outcome_of(e.as_ref()))?;
    Ok(machine)
}

/// The plan `levers` ask of `inputs` on `machine` for `slots` resident
/// sequences, as the gate reads it.
fn planned(
    inputs: &PlanInputs,
    machine: &Machine,
    ctx: u64,
    levers: &PlanLevers,
    slots: usize,
) -> Outcome {
    match plan_of(inputs, machine, ctx, levers, slots) {
        Ok(plan) => Outcome::Plan(
            view_of(&plan).with_loader(swap38::stacks(&inputs.model, &plan).map(drop)),
        ),
        Err(outcome) => outcome,
    }
}

/// [`planned`]'s plan itself, where a caller reads its bytes.
fn plan_of<'a>(
    inputs: &'a PlanInputs,
    machine: &'a Machine,
    ctx: u64,
    levers: &PlanLevers,
    slots: usize,
) -> Result<Plan<'a>, Outcome> {
    inputs
        .plan_with_slots(
            machine,
            (
                ctx,
                ub_of(usize::try_from(ctx).map_err(|e| Outcome::Red(e.to_string()))?)?,
            ),
            levers,
            Experts::Card,
            slots,
        )
        .map_err(|e| outcome_of(&e))
}

/// An outcome's text, where a search probe or the placement rule reports it
/// as an error the chain no longer types.
fn text_of(outcome: Outcome) -> GateError {
    match outcome {
        Outcome::Red(text) => text.into(),
        Outcome::Refused(r) => r.text.into(),
        Outcome::Plan(_) => "a plan, where an error was read".into(),
    }
}

/// The seat's resident sequences: every slot takes checkpoints, priced by
/// the body's own derivation from the file's delta and selecting layers
/// (`checkpoint_price38`).
fn seqs38(inputs: &PlanInputs, slots: usize) -> Seqs {
    let layers = &inputs.spec.layers;
    let gdn = layers
        .iter()
        .filter(|l| matches!(l.mixer, Mixer::DeltaRule(_)))
        .count();
    Seqs::priced(slots, checkpoint_price38(gdn, layers.len() - gdn))
}

/// A `GpuError` as a red outcome: the device census and the ubatch read
/// are not planner refusals.
fn gpu_red(e: GpuError) -> Outcome {
    Outcome::Red(e.to_string())
}
