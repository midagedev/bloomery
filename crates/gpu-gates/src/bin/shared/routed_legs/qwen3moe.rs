//! qwen3moe (Qwen3-30B-A3B): the two `--place`-unset production defaults
//! that plan a qwen3moe file, as `generate_qwen3moe` and the qwen3 serve
//! seat call the planner. The CLI's is `q3place::unplaced_qwen3` at the
//! CLI's 4096 for one sequence; the seat's is its `--ctx` search
//! (`q3place::default_ctx_qwen3` under `q3place::SERVE_RULE` for two
//! slots) and then its load at the total (`q3place::unplaced_qwen3_slots`).
//! The forced rooms go through the `PlaceQ3::plan` the defaults hand their
//! answer to, at the arm's context and slots. The cache planes are the
//! default f16 (`KvQ8::F16`).

use super::q3place;
use super::{Adapter, Arm, Outcome, outcome_of, view_of};
use bloomery_gpu::arch::qwen3moe::KvQ8;
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
        "qwen3moe"
    }

    fn arms(&self, split: &Split) -> Vec<Result<Arm, Outcome>> {
        let cli = arm(split, "cli", CLI_CTX, 1);
        // The seat's `--ctx` search first (its rule for two slots), then its
        // load at the total.
        let levers = PlanLevers::default();
        let serve = q3place::default_ctx_qwen3(
            split,
            None,
            q3place::SERVE_RULE,
            (SERVE_PARALLEL, &levers),
            KvQ8::F16,
        )
        .map_err(|e| outcome_of(e.as_ref()))
        .and_then(|found| {
            let total = found.slot.ctx.saturating_mul(SERVE_PARALLEL);
            arm(split, "serve", total, SERVE_PARALLEL)
        });
        vec![cli, serve]
    }
}

/// One arm's default at `ctx` positions for `slots` resident sequences: the
/// unbudgeted answer of `q3place::unplaced_qwen3_slots` (the CLI's own
/// `unplaced_qwen3` at one slot), and the placed planner it hands its
/// answer to. Where the default answers the whole load, the placed planner
/// is the probe the default itself built on `a`'s card, on the same census
/// reading (`q3place::PlaceQ3::qwen3_on` rebuilds it).
fn arm(split: &Split, name: &'static str, ctx: usize, slots: usize) -> Result<Arm, Outcome> {
    let kv = KvQ8::F16;
    let census = q3place::census().map_err(|e| outcome_of(e.as_ref()))?;
    match q3place::unplaced_qwen3_slots(split, &census, ctx, slots, kv)
        .map_err(|e| outcome_of(e.as_ref()))
    {
        Ok(q3place::Unplaced::Whole(_)) => {
            // The placed path at the same context: the probe the default
            // built on `a`'s card (`PlaceQ3::qwen3_on` lays out the same one
            // on the same reading; `Place::A` is the unresolved alias it
            // resolves there).
            let placed = q3place::PlaceQ3::qwen3_on(split, Place::A, &census, ctx, kv)
                .map_err(|e| outcome_of(e.as_ref()))?;
            Ok(Arm {
                name,
                whole: true,
                unbudgeted: None,
                placed: Box::new(move |levers| planned(&placed, ctx, levers)),
            })
        }
        Ok(q3place::Unplaced::Placed(placed)) => {
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
