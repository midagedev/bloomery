//! qwen35moe (Qwen3.6-35B-A3B, and any file of the same architecture
//! whatever its repo): the two `--place`-unset production defaults that plan
//! it, as `generate_qwen3moe` and the qwen3 serve seat call the planner. The
//! CLI's is `q3place::unplaced_qwen35` at the CLI's 4096 for one sequence
//! (the default `Open35`: the tensor-core flash, the default ubatch, the f16
//! cache, no media load); the seat's is its own choice of context and load
//! (`q3place::serve_choice` for the default slots, which take checkpoints,
//! `q3place::seqs_qwen35`), then its load at the total
//! (`q3place::unplaced_qwen35_slots`). The forced rooms go through
//! the `PlaceQ3::plan` the defaults hand their answer to, at the arm's
//! context and slots.

use super::q3place;
use super::{Adapter, Arm, Outcome, floor_or, outcome_of, view_of};
use bloomery_gpu::arch::qwen3moe::Open35;
use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_size;
use bloomery_gpu::kv_format::KvQ8;
use bloomery_gpu_gates::generate::Place;
use gguf::Split;
use model::placement::PlanLevers;
use model::placement::ctx_ask::DEFAULT_SLOTS;

/// The CLI's default context (`generate_qwen3moe`'s `--ctx` unset), the
/// precedent gate's own constant (`shared/qwen3moe_types.rs`'s `CLI_CTX`).
const CLI_CTX: usize = 4096;

/// The family's adapter, one row of the gate's dispatch.
pub(super) struct Row;

impl Adapter for Row {
    fn name(&self) -> &'static str {
        "qwen35moe"
    }

    fn arms(&self, split: &Split) -> Vec<Result<Arm, Outcome>> {
        vec![
            open35(split, CLI_CTX).and_then(|o| arm(split, "cli", &o, q3place::Seqs::ONE)),
            serve_arm(split),
        ]
    }
}

/// The seat's arm: its own choice of context first, then its load at the
/// total.
fn serve_arm(split: &Split) -> Result<Arm, Outcome> {
    let levers = PlanLevers::default();
    let opts = q3place::ServeOpts {
        place: None,
        levers: &levers,
        kv: KvQ8::F16,
        media: false,
        beside: bloomery_gpu_gates::beside::Beside::default(),
        ubatch: None,
    };
    let total = floor_or(
        q3place::serve_choice(split, (None, DEFAULT_SLOTS), opts).map(|(chosen, _)| chosen.total),
    )
    .map_err(|e| outcome_of(e.as_ref()))?;
    let total = usize::try_from(total).map_err(|e| outcome_of(&e))?;
    let seqs = q3place::seqs_qwen35(split, ubatch_size().map_err(gpu_red)?, DEFAULT_SLOTS)
        .map_err(|e| outcome_of(e.as_ref()))?;
    // The seat passes the total-context `Open35`; the default shares it
    // across the slots itself.
    arm(split, "serve", &open35(split, total)?, seqs)
}

/// The seat's `Open35` at `ctx` (`q3place::serve_open35`): the tensor-core
/// flash, the default ubatch, the f16 cache, no media load.
fn open35(split: &Split, ctx: usize) -> Result<Open35, Outcome> {
    q3place::serve_open35(split, ctx, KvQ8::F16, false).map_err(|e| outcome_of(e.as_ref()))
}

/// One arm's default at the context `o` names for its `seqs`: the
/// unbudgeted answer of `q3place::unplaced_qwen35_slots` (the CLI's own
/// `unplaced_qwen35` at one sequence), and the placed planner it hands its
/// answer to. Where the default answers the whole load, the placed planner
/// is the probe the default itself built on `a`'s card, on the same census
/// reading (`q3place::PlaceQ3::qwen35_on` rebuilds it).
fn arm(split: &Split, name: &'static str, o: &Open35, seqs: q3place::Seqs) -> Result<Arm, Outcome> {
    let census = q3place::census().map_err(|e| outcome_of(e.as_ref()))?;
    match q3place::unplaced_qwen35_slots(
        split,
        &census,
        o,
        seqs,
        bloomery_gpu_gates::beside::Beside::default(),
    )
    .map_err(|e| outcome_of(e.as_ref()))
    {
        Ok(q3place::Unplaced::Whole(_)) => {
            let placed = q3place::PlaceQ3::qwen35_on(
                split,
                Place::A,
                &census,
                q3place::share(o, o.ctx, seqs.slots),
                seqs,
                bloomery_gpu_gates::beside::Beside::default(),
            )
            .map_err(|e| outcome_of(e.as_ref()))?;
            let ctx = o.ctx;
            Ok(Arm {
                name,
                whole: true,
                unbudgeted: None,
                placed: Box::new(move |levers| planned(&placed, ctx, levers)),
            })
        }
        Ok(q3place::Unplaced::Placed(placed)) => {
            // The closure owns the context it plans at; the arm outlives the
            // `Open35` it was read from.
            let ctx = o.ctx;
            let unbudgeted = planned(&placed, ctx, &PlanLevers::default());
            Ok(Arm {
                name,
                whole: false,
                unbudgeted: Some(unbudgeted),
                placed: Box::new(move |levers| planned(&placed, ctx, levers)),
            })
        }
        // The default itself refused to read or plan the file: the arm's
        // own refusal, counted by kind.
        Err(outcome) => Err(outcome),
    }
}

/// The placed planner's answer under `levers`, as the gate reads it.
fn planned(placed: &q3place::PlaceQ3, ctx: usize, levers: &PlanLevers) -> Outcome {
    match placed.plan(ctx, levers) {
        Ok(plan) => Outcome::Plan(view_of(&plan)),
        Err(e) => outcome_of(e.as_ref()),
    }
}

/// A `GpuError` as a red outcome: the device census and the ubatch read
/// are not planner refusals.
fn gpu_red(e: bloomery_gpu::GpuError) -> Outcome {
    Outcome::Red(e.to_string())
}
