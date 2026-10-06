//! The V4.1 side of `BLOOMERY_RESIDENCY` unset against the plan the load
//! runs ([`at_plan`]): the one call `generate_ds41` and the ds41 seat of
//! `bloomery-serve` make on the plan they are about to load by, its inputs
//! taken as the Qwen3.8 and GLM sides take theirs
//! (`residency38::residency38`, `serve_seats::glm`).

use crate::GateError;
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu_deepseek41::swap;
use bloomery_levers::{ResidencyPick, residency_at_plan, residency_word};
use model::placement::Plan;
use model::placement::workstation::{HostNeed, host_available};

/// `BLOOMERY_RESIDENCY` unset against `plan` ([`residency_at_plan`]): the
/// plan's card experts a layer, its host headroom, and what `MemAvailable`
/// leaves past the plan's own host need — the term the load's own check
/// (`HostNeed::check`, the churn pool as its extra bytes) uses, so a word
/// this keeps passes the load. The churn pool's bytes at a pinned count
/// come from the V4.1 plan's own [`swap::churn`], stage card 0.
pub fn at_plan(plan: &Plan<'_>, pick: ResidencyPick) -> Result<ResidencyPick, GateError> {
    let spares = match residency_word(pick.word) {
        Some(bloomery_levers::ResidencyWord::Mid { spares, .. }) => spares,
        _ => 0,
    };
    let mem_left = i128::from(host_available()?) - i128::from(HostNeed::of(plan, 0).bytes());
    Ok(residency_at_plan(
        pick,
        plan.n_l.iter().copied(),
        |pinned| {
            swap::churn(plan, 0, Residency::Mid { pinned, spares })
                .map(|pool| pool.expect("churn under mid holds a pool").bytes)
        },
        plan.host.headroom_bytes,
        mem_left,
    )?)
}
