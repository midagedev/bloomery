//! The residency a load runs and the records that say which: a set word on
//! any model's plan ([`residency_set`], the churn pool of the card it names,
//! and [`residency_room`], the word's room on the plan's card slots;
//! `generate_glm5next`, the GLM serve seat), and a qwen4exp load's lever, set or unset by the
//! Qwen3.8 rule (`bloomery_levers::residency38_unset` before the plan,
//! `residency38_at_plan` on it; `generate_qwen3moe`, the Qwen3.8 serve seat).

use crate::GateError;
use crate::record::{self, Record};
use bloomery_gpu::host::swap::Residency;
use bloomery_levers::{RESIDENCY38_SPARES, Residency38Pick, residency38_at_plan};
use model::placement::Plan;
use model::placement::churn::ChurnPool;
use model::placement::workstation::{HostNeed, host_available};

/// The plan's card a qwen4exp open loads: its one card.
pub const CARD38: usize = 0;

/// The plan's card GLM's residency machine runs over: plan (a)'s one card.
pub const GLM_CARD: usize = 0;

/// `BLOOMERY_RESIDENCY` as a qwen4exp load takes it: a set word and its
/// parse, or unset and what the rule decided before the plan — `None` when
/// plan (a) decides.
#[derive(Clone, Copy)]
pub enum Lever38<'a> {
    Set(Residency, &'a str),
    Unset(Option<Residency38Pick>),
}

/// The residency a load of `plan` runs under `lever`: a set word as given;
/// unset, the rule's before the plan or on plan (a) from it
/// (`residency38_at_plan`: P half the fewest card experts a layer, `off`
/// with why where the plan has no room, or the plan's host headroom or
/// `MemAvailable` none for the churn pool), its `residency unset` record
/// handed to `emit`. Under `mid`, the `residency host` record of `plan`
/// follows: the churn pool (card [`CARD38`]'s experts past the pinned ones)
/// the load's host set holds beside the plan's host segments, which the
/// load refuses by name for a set word when the plan's host headroom cannot
/// take it. `emit` is where the binary prints its records.
pub fn residency38(
    plan: &Plan<'_>,
    lever: Lever38<'_>,
    emit: fn(Record),
) -> Result<Residency, GateError> {
    let (residency, word) = match lever {
        Lever38::Set(r, word) => (r, word.to_owned()),
        Lever38::Unset(pre) => {
            let pick = match pre {
                Some(off) => off,
                None => {
                    // The load refuses a host set past `MemAvailable` before
                    // any upload; the default leaves the pool out instead.
                    let available = host_available()?;
                    let need = HostNeed::of(plan, 0).bytes();
                    residency38_at_plan(
                        plan.n_l.iter().copied(),
                        plan.host.experts,
                        |pinned| ChurnPool::of(plan, CARD38, pinned).map(|pool| pool.bytes),
                        plan.host.headroom_bytes,
                        i128::from(available) - i128::from(need),
                    )
                    .map_err(|e| format!("BLOOMERY_RESIDENCY unset: the churn pool: {e}"))?
                }
            };
            emit(record::residency_unset(&pick));
            let residency = match pick.pinned {
                None => Residency::Off,
                Some(pinned) => Residency::Mid {
                    pinned,
                    spares: RESIDENCY38_SPARES,
                },
            };
            (residency, pick.word())
        }
    };
    residency_set(plan, CARD38, residency, &word, 0, emit)
}

/// A load of `plan` under `residency`, the word `word` names, whatever the
/// model: under `mid`, the `residency host` record of `plan` handed to
/// `emit` — the churn pool (card `card`'s experts past the pinned ones) the
/// load's host set holds beside the plan's host segments, refused by name
/// when the plan cannot give it, and the headroom left after it and the
/// `beside` bytes the load hosts outside the plan (a draft layer's experts;
/// 0 for a plain load); `off` as it is, with no record.
pub fn residency_set(
    plan: &Plan<'_>,
    card: usize,
    residency: Residency,
    word: &str,
    beside: u64,
    emit: fn(Record),
) -> Result<Residency, GateError> {
    let Residency::Mid { pinned, .. } = residency else {
        return Ok(residency);
    };
    let pool = ChurnPool::of(plan, card, pinned)
        .map_err(|e| format!("BLOOMERY_RESIDENCY={word}: the churn pool: {e}"))?;
    emit(record::residency_host_beside(word, &pool, plan, beside));
    Ok(residency)
}

/// `word`'s pinned experts, its spares and one that moves fit every layer's
/// card experts in `plan`, else a named refusal before the load; `off`
/// fits any plan.
pub fn residency_room(plan: &Plan<'_>, r: Residency, word: &str) -> Result<(), GateError> {
    let Residency::Mid { pinned, spares } = r else {
        return Ok(());
    };
    let fewest = plan
        .n_l
        .iter()
        .copied()
        .filter(|&n| n > 0)
        .min()
        .ok_or_else(|| {
            format!("BLOOMERY_RESIDENCY={word}: the plan puts no routed expert on the card")
        })?;
    let need = pinned + spares + 1;
    if usize::try_from(fewest)? < need {
        return Err(format!(
            "BLOOMERY_RESIDENCY={word}: the plan's fewest card experts a layer is {fewest}, \
             fewer than {pinned} pinned, {spares} spare and one that moves"
        )
        .into());
    }
    Ok(())
}
