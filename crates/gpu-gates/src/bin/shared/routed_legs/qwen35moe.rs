//! qwen35moe (Qwen3.6-35B-A3B, and any file of the same architecture
//! whatever its repo): the two `--place`-unset production defaults that plan
//! it, as `generate_qwen3moe` and the qwen3 serve seat call the planner. The
//! CLI's is `q3place::unplaced_qwen35` at the CLI's 4096 for one sequence
//! (the default `Open35`: the tensor-core flash, the default ubatch, the f16
//! cache, no media load); the seat's is its `--ctx` search
//! (`q3place::default_ctx_qwen35` under `q3place::SERVE_RULE` for two slots
//! that take checkpoints, `q3place::seqs_qwen35`) and then its load at the
//! total (`q3place::unplaced_qwen35_slots`). The forced rooms go through
//! the `PlaceQ3::plan` the defaults hand their answer to, at the arm's
//! context and slots.

use super::q3place;
use super::{Adapter, Arm, Outcome, outcome_of, view_of};
use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_size;
use bloomery_gpu::arch::qwen3moe::{KvQ8, Open35};
use bloomery_gpu_gates::generate::Place;
use gguf::Split;
use model::placement::PlanLevers;

/// The CLI's default context (`generate_qwen3moe`'s `--ctx` unset), the
/// precedent gate's own constant (`shared/qwen3moe_types.rs`'s `CLI_CTX`).
const CLI_CTX: usize = 4096;

/// The serve seat's resident sequences, `--parallel` unset.
const SERVE_PARALLEL: usize = 2;

/// The family's adapter, one row of the gate's dispatch.
pub(super) struct Row;

impl Adapter for Row {
    fn name(&self) -> &'static str {
        "qwen35moe"
    }

    fn arms(&self, split: &Split) -> Vec<Result<Arm, Outcome>> {
        vec![
            open35(CLI_CTX).and_then(|o| arm(split, "cli", &o, q3place::Seqs::ONE)),
            serve_arm(split),
        ]
    }
}

/// The seat's arm: its `--ctx` search first — the rule for two checkpointing
/// slots, probed at the base context the seat itself names — then its load
/// at the total.
fn serve_arm(split: &Split) -> Result<Arm, Outcome> {
    let levers = PlanLevers::default();
    let seqs = q3place::seqs_qwen35(split, ubatch_size().map_err(gpu_red)?, SERVE_PARALLEL)
        .map_err(|e| outcome_of(e.as_ref()))?;
    let found = q3place::default_ctx_qwen35(
        split,
        &open35(q3place::SERVE_RULE.base)?,
        None,
        q3place::SERVE_RULE,
        (seqs, &levers, None),
    )
    .map_err(|e| outcome_of(e.as_ref()))?;
    let total = found.slot.ctx.saturating_mul(SERVE_PARALLEL);
    // The seat passes the total-context `Open35`; the default shares it
    // across the slots itself.
    arm(split, "serve", &open35(total)?, seqs)
}

/// The seat's `open35` at `ctx` (`serve_seats/qwen3.rs`): the tensor-core
/// flash, the default ubatch, the f16 cache, no media load.
fn open35(ctx: usize) -> Result<Open35, Outcome> {
    Ok(Open35 {
        ctx,
        mma: true,
        ubatch: ubatch_size().map_err(gpu_red)?,
        kv: KvQ8::F16,
        media: false,
    })
}

/// One arm's default at the context `o` names for its `seqs`: the
/// unbudgeted answer of `q3place::unplaced_qwen35_slots` (the CLI's own
/// `unplaced_qwen35` at one sequence), and the placed planner it hands its
/// answer to. Where the default answers the whole load, the placed planner
/// is the probe the default itself built on `a`'s card
/// (`q3place::PlaceQ3::qwen35_seqs` rebuilds it).
fn arm(split: &Split, name: &'static str, o: &Open35, seqs: q3place::Seqs) -> Result<Arm, Outcome> {
    match q3place::unplaced_qwen35_slots(split, o, seqs, None).map_err(|e| outcome_of(e.as_ref())) {
        Ok(q3place::Unplaced::Whole(_)) => {
            let placed = q3place::PlaceQ3::qwen35_seqs(
                split,
                Place::A,
                q3place::share(o, o.ctx, seqs.slots),
                seqs,
                None,
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
