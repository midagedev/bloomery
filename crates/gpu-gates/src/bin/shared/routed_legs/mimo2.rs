//! mimo2 (MiMo-V2.6-Flash): the `--place`-unset production default that plans
//! a mimo2 file. MiMo ships one binary, the serve seat of `bloomery-serve`
//! (`serve_seats/mimo2.rs`): no `generate_*` binary opens it, so the one arm
//! is the seat's. It plans as the seat does: one default slot
//! (`ctx_ask::DEFAULT_SLOTS`) at the context `generate::choose_ctx` takes
//! (the seats' ladder; a host short is `PlaceError::NoNvmeTier`) and the plan
//! at that slot context (`PlanInputs::plan_with_slots`).
//!
//! The machine is the one card `a` names on a workstation whose largest card
//! is the A6000 (`workstation::plan_on` over `A6000`, which is what the
//! seat's `Place::A.machine` builds there; the body serves no tier card). The
//! 3090's usable bytes would plan no MiMo expert on the card: the trunk's
//! floor (25,338,839,040 B) leaves 11 MiB of them, less than any layer's
//! first expert, so every plan there places 0 of the routed layers and no
//! check could read a carded layer. The forced rooms go through the same
//! call at the arm's slot context and slots.
//!
//! The loader's plan-time stack table (`bloomery_gpu_mimo2::swap::stacks`,
//! L6) is asked of every plan: it reads the header alone.

use std::num::NonZeroUsize;
use std::sync::Arc;

use super::{
    Adapter, Arm, Outcome, Refusal, floor_or, outcome_of, probed, refused_of, view_of,
};
use bloomery_gpu_gates::GateError;
use bloomery_gpu_gates::generate::choose_ctx;
use bloomery_gpu_mimo2::swap;
use gguf::Split;
use model::arch::mimo2::place::{PlaceError, PlanInputs};
use model::placement::ctx_ask::{DEFAULT_SLOTS, probe_of};
use model::placement::workstation::{A6000, plan_on};
use model::placement::{Machine, Plan, PlanLevers};

/// The loader table L6 asks.
const LOADER: &str = "mimo2::swap::stacks";

/// The family's adapter, one row of the gate's dispatch.
pub(super) struct Row;

impl Adapter for Row {
    fn name(&self) -> &'static str {
        "mimo2"
    }

    fn arms(&self, split: &Split) -> Vec<Result<Arm, Outcome>> {
        match PlanInputs::read(split).map_err(|e| refused_of(&e)) {
            Ok(inputs) => vec![arm(&Arc::new(inputs), split)],
            Err(refusal) => vec![Err(refusal)],
        }
    }

    fn loader_table(&self) -> Option<&'static str> {
        Some(LOADER)
    }
}

/// The seat's arm: the default slot at the context its unset `--ctx` takes.
fn arm(inputs: &Arc<PlanInputs>, split: &Split) -> Result<Arm, Outcome> {
    let machine = plan_on(A6000, inputs.model.layers);
    let n = NonZeroUsize::new(DEFAULT_SLOTS).expect("the default serves a slot");
    let ctx = floor_or(
        choose_ctx(split, (None, DEFAULT_SLOTS), (None, None), |c| {
            probed();
            probe_of(inputs.plan_with_slots(&machine, c, &PlanLevers::default(), n))
        })
        .map(|chosen| chosen.slot_ctx),
    )
    .map_err(|e| place_gate(&e))?;
    let unbudgeted = planned(inputs, &machine, ctx, &PlanLevers::default(), n);
    let inputs = Arc::clone(inputs);
    Ok(Arm {
        name: "serve",
        whole: false,
        unbudgeted: Some(unbudgeted),
        placed: Box::new(move |levers| planned(&inputs, &machine, ctx, levers, n)),
    })
}

/// The plan `levers` ask of `inputs` on `machine` for `slots` sequences of
/// `ctx` positions, with the loader table's answer on its header.
fn planned(
    inputs: &PlanInputs,
    machine: &Machine,
    ctx: u64,
    levers: &PlanLevers,
    slots: NonZeroUsize,
) -> Outcome {
    match plan_of(inputs, machine, ctx, levers, slots) {
        Ok(plan) => Outcome::Plan(view_of(&plan).with_loader(swap::stacks(inputs).map(drop))),
        Err(outcome) => outcome,
    }
}

/// The load's own plan call (`PlanInputs::plan_with_slots`).
fn plan_of<'a>(
    inputs: &'a PlanInputs,
    machine: &'a Machine,
    ctx: u64,
    levers: &PlanLevers,
    slots: NonZeroUsize,
) -> Result<Plan<'a>, Outcome> {
    inputs
        .plan_with_slots(machine, ctx, levers, slots)
        .map_err(|e| place_of(&e))
}

/// A `choose_ctx` error as an outcome: the family's plan refusal keeps its
/// name ([`place_of`]); anything else is the gate's common reading.
fn place_gate(e: &GateError) -> Outcome {
    let mut cur: Option<&(dyn std::error::Error + 'static)> = Some(e.as_ref());
    while let Some(c) = cur {
        if let Some(p) = c.downcast_ref::<PlaceError>() {
            return place_of(p);
        }
        cur = c.source();
    }
    outcome_of(e.as_ref())
}

/// A plan refusal as an outcome: the placement's own are named by
/// [`refused_of`], a room that pages experts to the NVMe tier the body does
/// not serve is named here; a tier card, a context of no position or an
/// invariant break is red.
fn place_of(p: &PlaceError) -> Outcome {
    match p {
        PlaceError::Placement(e) => refused_of(e),
        PlaceError::NoNvmeTier { .. } => Outcome::Refused(Refusal {
            kind: "no-nvme-tier",
            floor: None,
            text: p.to_string(),
        }),
        PlaceError::Tier { .. } | PlaceError::Broken(_) | PlaceError::CtxZero => {
            Outcome::Red(p.to_string())
        }
    }
}
