//! deepseek41 (DeepSeek-V4.1-Flash, and V4 files, which run the same program
//! and planner): the two `--place`-unset production defaults that plan a V4.1
//! file, as `generate_ds41` and `bloomery-serve-ds41` call the planner. Both
//! take the placement the family's unset rule picks (`ds41_place::choose`:
//! `a`, the plan never asked of a tier) and its machine
//! (`Place::machine` over the file's layers), with no draft reserve (the
//! DSpark draft is off unset) and no image encoder. The CLI's is
//! `PlanInputs::plan` for one sequence at `workstation::CTX_MAX` positions
//! (`Open::plan`); the seat's is `PlanInputs::plan_with_slots` for its two
//! slots, each a full `CTX_MAX` (`slot_ctx`), on a machine whose checkpoint
//! reserve is none (V4.1's body makes no checkpoints). The forced rooms go
//! through the same call at the arm's context and slots.
//!
//! V4.1's card leg is exact ([`DS41_LEG`]: a Q3_K gate and up, a Q4_K down),
//! so the sweep holds the leg itself and the Q6_K down its per-type plan rule
//! once carded.

use std::num::NonZeroUsize;
use std::sync::Arc;

use super::ds41place;
use super::{Adapter, Arm, Outcome, census, outcome_of, refused_of, view_of};
use bloomery_gpu_gates::generate::Place;
use gguf::{GgmlType, Split};
use model::arch::deepseek41::place::{PlaceError, PlanInputs};
use model::placement::accepts::DS41_LEG;
use model::placement::workstation::CTX_MAX;
use model::placement::{Machine, Plan, PlanLevers};

/// The serve seat's resident sequences, `--parallel` unset
/// (`serve_seats/ds41.rs`'s `SLOTS`).
const SERVE_PARALLEL: usize = 2;

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
                    arm(&inputs, "cli", 1),
                    arm(&inputs, "serve", SERVE_PARALLEL),
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

/// One arm: `slots` sequences of `CTX_MAX` positions each on the unset
/// placement's machine.
fn arm(inputs: &Arc<PlanInputs>, name: &'static str, slots: usize) -> Result<Arm, Outcome> {
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
    let machine = place
        .machine(None, ds41place::tier_batch(place, &inputs.hp))
        .map_err(|e| outcome_of(e.as_ref()))?(inputs.model.layers);
    let n = NonZeroUsize::new(slots).expect("an arm serves at least one slot");
    let unbudgeted = planned(inputs, &machine, &PlanLevers::default(), n);
    let inputs = Arc::clone(inputs);
    Ok(Arm {
        name,
        whole: false,
        unbudgeted: Some(unbudgeted),
        placed: Box::new(move |levers| planned(&inputs, &machine, levers, n)),
    })
}

/// The plan `levers` ask of `inputs` on `machine` for `slots` sequences, as
/// the gate reads it.
fn planned(
    inputs: &PlanInputs,
    machine: &Machine,
    levers: &PlanLevers,
    slots: NonZeroUsize,
) -> Outcome {
    match plan_of(inputs, machine, levers, slots) {
        Ok(plan) => Outcome::Plan(view_of(&plan)),
        Err(outcome) => outcome,
    }
}

/// The load's own plan call: `PlanInputs::plan` for one sequence (the CLI's
/// `Open::plan`), `plan_with_slots` for several (the seat's, and the CLI's
/// `open_slots`).
fn plan_of<'a>(
    inputs: &'a PlanInputs,
    machine: &'a Machine,
    levers: &PlanLevers,
    slots: NonZeroUsize,
) -> Result<Plan<'a>, Outcome> {
    let plan = if slots.get() > 1 {
        inputs.plan_with_slots(machine, CTX_MAX, levers, slots)
    } else {
        inputs.plan(machine, CTX_MAX, levers)
    };
    plan.map_err(|e| place_of(&e))
}

/// A plan refusal as an outcome: the placement's own are named by
/// [`refused_of`]; an invariant break is red.
fn place_of(p: &PlaceError) -> Outcome {
    match p {
        PlaceError::Placement(e) => refused_of(e),
        PlaceError::Broken(_) => Outcome::Red(p.to_string()),
    }
}
