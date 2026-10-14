//! deepseek41 (DeepSeek-V4.1-Flash, and V4 files, which run the same program
//! and planner): the two `--place`-unset production defaults that plan a V4.1
//! file, as `generate_ds41` and `bloomery-serve-ds41` call the planner. Both
//! take the placement the family's unset rule picks (`ds41_place::choose`:
//! `a`, the plan never asked of a tier) and its machine
//! (`Place::machine` over the file's layers), with no draft reserve (the
//! DSpark draft is off unset) and no image encoder. The CLI's is
//! `PlanInputs::plan` for one sequence at `workstation::CTX_MAX` positions
//! (`Open::plan`); the seat's is `PlanInputs::plan_with_slots` for its
//! default slots at the context its unset `--ctx` takes
//! (`ds41_place::ctx_choice`, the seats' ladder), on a machine whose
//! checkpoint reserve is none (V4.1's body makes no checkpoints). The forced
//! rooms go through the same call at the arm's context and slots.
//!
//! V4.1's card leg is exact ([`DS41_LEG`]: a Q3_K gate and up, a Q4_K down),
//! so the sweep holds the leg itself and the Q6_K down its per-type plan rule
//! once carded.

use std::num::NonZeroUsize;
use std::sync::Arc;

use super::ds41place;
use super::{Adapter, Arm, Outcome, census, floor_or, outcome_of, refused_of, view_of};
use bloomery_gpu_gates::generate::Place;
use gguf::{GgmlType, Split};
use model::arch::deepseek41::place::{PlaceError, PlanInputs};
use model::placement::accepts::DS41_LEG;
use model::placement::ctx_ask::DEFAULT_SLOTS;
use model::placement::workstation::CTX_MAX;
use model::placement::{Machine, Plan, PlanLevers};

/// The family's adapter, one row of the gate's dispatch.
pub(super) struct Row;

impl Adapter for Row {
    fn name(&self) -> &'static str {
        "deepseek41"
    }

    fn arms(&self, split: &Split) -> Vec<Result<Arm, Outcome>> {
        match PlanInputs::read(split).map_err(|e| refused_of(&e)) {
            Ok(inputs) => {
                let inputs = Arc::new(inputs);
                vec![
                    arm(&inputs, "cli", None),
                    arm(&inputs, "serve", Some(split)),
                ]
            }
            // The file is read once; a refusal of it is each arm's.
            Err(refusal) => vec![Err(refusal.clone()), Err(refusal)],
        }
    }

    fn extra_triples(&self) -> Vec<[GgmlType; 3]> {
        vec![DS41_LEG, [DS41_LEG[0], DS41_LEG[1], GgmlType::Q6_K]]
    }
}

/// One arm on the unset placement's machine: the CLI's one sequence of
/// `CTX_MAX` positions (`seat` none), or the seat's default slots at the
/// context its unset `--ctx` takes on `seat`, the file.
fn arm(inputs: &Arc<PlanInputs>, name: &'static str, seat: Option<&Split>) -> Result<Arm, Outcome> {
    // The family's unset rule (`ds41_place::choose`) on the cached census:
    // the plan is never asked of a tier.
    let place: Place = Place::choose(None, census()?, ds41place::TIER_RULE, |offer| {
        Err(format!(
            "--place unset: V4.1 has no break-even yet, so the plan is not asked for {}",
            offer.name()
        )
        .into())
    })
    .map_err(|e| outcome_of(e.as_ref()))?
    .place;
    let tier_batch = ds41place::tier_batch(place, &inputs.hp);
    let machine = place
        .machine(None, tier_batch)
        .map_err(|e| outcome_of(e.as_ref()))?(inputs.model.layers);
    let (n, ctx) = match seat {
        None => (NonZeroUsize::MIN, CTX_MAX),
        Some(split) => {
            let n = NonZeroUsize::new(DEFAULT_SLOTS).expect("the default serves a slot");
            let ctx = floor_or(
                ds41place::ctx_choice(
                    split,
                    inputs,
                    place,
                    (None, tier_batch, None),
                    (None, n),
                    &PlanLevers::default(),
                )
                .map(|chosen| chosen.slot_ctx),
            )
            .map_err(|e| outcome_of(e.as_ref()))?;
            (n, ctx)
        }
    };
    let unbudgeted = planned(inputs, &machine, &PlanLevers::default(), (n, ctx));
    let inputs = Arc::clone(inputs);
    Ok(Arm {
        name,
        whole: false,
        unbudgeted: Some(unbudgeted),
        placed: Box::new(move |levers| planned(&inputs, &machine, levers, (n, ctx))),
    })
}

/// The plan `levers` ask of `inputs` on `machine` for `slots` sequences of
/// `ctx` positions, as the gate reads it.
fn planned(
    inputs: &PlanInputs,
    machine: &Machine,
    levers: &PlanLevers,
    (slots, ctx): (NonZeroUsize, u64),
) -> Outcome {
    match plan_of(inputs, machine, levers, (slots, ctx)) {
        Ok(plan) => Outcome::Plan(view_of(&plan)),
        Err(outcome) => outcome,
    }
}

/// The load's own plan call, `PlanInputs::plan_with_slots` (`plan` is its
/// one-sequence case: the CLI's `Open::plan`).
fn plan_of<'a>(
    inputs: &'a PlanInputs,
    machine: &'a Machine,
    levers: &PlanLevers,
    (slots, ctx): (NonZeroUsize, u64),
) -> Result<Plan<'a>, Outcome> {
    inputs
        .plan_with_slots(machine, ctx, levers, slots)
        .map_err(|e| place_of(&e))
}

/// A plan refusal as an outcome: the placement's own are named by
/// [`refused_of`]; an invariant break is red.
fn place_of(p: &PlaceError) -> Outcome {
    match p {
        PlaceError::Placement(e) => refused_of(e),
        PlaceError::Broken(_) => Outcome::Red(p.to_string()),
    }
}
