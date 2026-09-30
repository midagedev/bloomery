//! The residency a qwen4exp load runs, for every binary that loads one
//! (`generate_qwen3moe`, the Qwen3.8 serve seat): the lever's word as set,
//! or unset the Qwen3.8 rule's (`bloomery_levers::residency38_unset` before
//! the plan, `residency38_at_plan` on it), with the records that say which.

use crate::GateError;
use crate::record::{self, Record};
use bloomery_gpu::host::swap::Residency;
use bloomery_levers::{RESIDENCY38_SPARES, Residency38Pick, residency38_at_plan};
use model::placement::Plan;
use model::placement::churn::ChurnPool;
use model::placement::workstation::{HostNeed, host_available};

/// The plan's card a qwen4exp open loads: its one card.
pub const CARD38: usize = 0;

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
    let Residency::Mid { pinned, .. } = residency else {
        return Ok(residency);
    };
    let pool = ChurnPool::of(plan, CARD38, pinned)
        .map_err(|e| format!("BLOOMERY_RESIDENCY={word}: the churn pool: {e}"))?;
    emit(record::residency_host(&word, &pool, plan));
    Ok(residency)
}
