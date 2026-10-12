//! mimo2 (MiMo-V2.6-Flash): the `--place`-unset production defaults that plan
//! a mimo2 file. MiMo ships one binary, the serve seat of `bloomery-serve`
//! (`serve_seats/mimo2.rs`): no `generate_*` binary opens it, so both arms
//! are the seat's. The first is its default: two slots (`--parallel` unset),
//! the second kept only while the plan takes it (`default_slots_fit`). The
//! second is the one-sequence server (`--parallel 1`), which is also the
//! slot count the first falls to. Each plans as the seat does: the context
//! its search answers (`placement::ctx::searched` over the trained context
//! at the seat's floor and grid, a slot holding ⌊total / slots⌋) and the plan
//! at a slot's share (`PlanInputs::plan_with_slots`).
//!
//! The machine is the one card `a` names on a workstation whose largest card
//! is the A6000 (`workstation::plan_on` over `A6000`, which is what the
//! seat's `Place::A.machine` builds there; the body serves no tier card). The
//! 3090's usable bytes would plan no MiMo expert on the card: the trunk's
//! floor (25,338,839,040 B, the same at every set and arm) leaves 11 MiB of
//! them, less than any layer's first expert, so every plan there places 0 of
//! the routed layers and no check could read a carded layer. The forced rooms
//! go through the same call at the arm's slot context and slots.
//!
//! The loader's plan-time stack table (`bloomery_gpu_mimo2::swap::stacks`,
//! L6) is asked of every plan: it reads the header alone.

use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use super::{Adapter, Arm, Outcome, probed, refused_of, view_of};
use bloomery_gpu_gates::bind::{SlotsFrom, default_slots_fit};
use bloomery_gpu_gates::generate::{CTX_GRAN, trained_ctx};
use bloomery_gpu_mimo2::swap;
use gguf::Split;
use model::arch::mimo2::place::{PlaceError, PlanInputs};
use model::placement::slots::split_ctx;
use model::placement::workstation::{A6000, plan_on};
use model::placement::{Machine, Plan, PlanLevers};

/// The floor of the seat's searched default context
/// (`serve_seats/mimo2.rs`'s `CTX`).
const CTX: usize = 4096;

/// The slots a seat with no `--parallel` and no `--ctx` serves
/// (`serve_seats/mimo2.rs`'s `SLOTS`).
const SLOTS: usize = 2;

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
            Ok(inputs) => {
                let inputs = Arc::new(inputs);
                let trained = trained_ctx(split);
                vec![
                    arm(&inputs, trained, ("serve", SLOTS), SlotsFrom::Default),
                    arm(&inputs, trained, ("serve1", 1), SlotsFrom::Flag),
                ]
            }
            // The file is read once; a refusal of it is each arm's.
            Err(refusal) => vec![Err(refusal.clone()), Err(refusal)],
        }
    }

    fn loader_table(&self) -> Option<&'static str> {
        Some(LOADER)
    }
}

/// One arm of the seat: `slots` sequences asked for as `from` says (a
/// default count falls to one when its plan is refused, a flag's never).
fn arm(
    inputs: &Arc<PlanInputs>,
    trained: Option<usize>,
    (name, slots): (&'static str, usize),
    from: SlotsFrom,
) -> Result<Arm, Outcome> {
    let machine = plan_on(A6000, inputs.model.layers);
    let (slot_ctx, slots, _) = default_slots_fit("gate_routed_legs", "a", slots, from, |slots| {
        let n = NonZeroUsize::new(slots)
            .ok_or_else(|| Outcome::Red("--parallel 0: the server serves no slot".to_owned()))?;
        let total = total_ctx(trained, inputs, &machine, n)?;
        let slot_ctx = slot_ctx_of(total, slots)?;
        // The seat plans here and keeps the plan; a refusal falls to one
        // slot or stands.
        plan_of(inputs, &machine, slot_ctx, &PlanLevers::default(), n).map(|_| slot_ctx)
    })?;
    let n = NonZeroUsize::new(slots).expect("the slot fit kept a count of at least one");
    let unbudgeted = planned(inputs, &machine, slot_ctx, &PlanLevers::default(), n);
    let inputs = Arc::clone(inputs);
    Ok(Arm {
        name,
        whole: false,
        unbudgeted: Some(unbudgeted),
        placed: Box::new(move |levers| planned(&inputs, &machine, slot_ctx, levers, n)),
    })
}

/// The seat's total context (`ctx_of` with `--ctx` unset): the file's
/// trained context capped to the largest on the [`CTX_GRAN`] grid whose
/// plan stands for every slot at its share, the floor [`CTX`] where the file
/// states none or nothing fits.
fn total_ctx(
    trained: Option<usize>,
    inputs: &PlanInputs,
    machine: &Machine,
    slots: NonZeroUsize,
) -> Result<usize, Outcome> {
    let fits = |c: usize| -> Result<bool, Outcome> {
        probed();
        let share = slot_ctx_of(c, slots.get()).ok();
        Ok(share
            .is_some_and(|s| plan_of(inputs, machine, s, &PlanLevers::default(), slots).is_ok()))
    };
    Ok(model::placement::ctx::searched(trained, CTX, CTX_GRAN, &fits)?.unwrap_or(CTX))
}

/// A slot's positions when `slots` slots split `total`: ⌊total / slots⌋
/// (`placement::slots::split_ctx`, llama-server's `-np N` without `-kvu`).
fn slot_ctx_of(total: usize, slots: usize) -> Result<usize, Outcome> {
    let share = split_ctx(total as u64, slots as u64, NonZeroU64::MIN)
        .map_err(|e| Outcome::Red(format!("--ctx {total} over {slots} slots: {e}")))?;
    usize::try_from(share).map_err(|e| Outcome::Red(format!("a slot's {share} positions: {e}")))
}

/// The plan `levers` ask of `inputs` on `machine` for `slots` sequences of
/// `slot_ctx` positions each, with the loader table's answer on its header.
fn planned(
    inputs: &PlanInputs,
    machine: &Machine,
    slot_ctx: usize,
    levers: &PlanLevers,
    slots: NonZeroUsize,
) -> Outcome {
    match plan_of(inputs, machine, slot_ctx, levers, slots) {
        Ok(plan) => Outcome::Plan(view_of(&plan).with_loader(swap::stacks(inputs).map(drop))),
        Err(outcome) => outcome,
    }
}

/// The load's own plan call (`PlanInputs::plan_with_slots`).
fn plan_of<'a>(
    inputs: &'a PlanInputs,
    machine: &'a Machine,
    slot_ctx: usize,
    levers: &PlanLevers,
    slots: NonZeroUsize,
) -> Result<Plan<'a>, Outcome> {
    let ctx = u64::try_from(slot_ctx)
        .map_err(|e| Outcome::Red(format!("a slot's context {slot_ctx}: {e}")))?;
    inputs
        .plan_with_slots(machine, ctx, levers, slots)
        .map_err(|e| place_of(&e))
}

/// A plan refusal as an outcome: the placement's own are named by
/// [`refused_of`]; a tier card, a context of no position or an invariant
/// break is red.
fn place_of(p: &PlaceError) -> Outcome {
    match p {
        PlaceError::Placement(e) => refused_of(e),
        PlaceError::Tier { .. } | PlaceError::Broken(_) | PlaceError::CtxZero => {
            Outcome::Red(p.to_string())
        }
    }
}
