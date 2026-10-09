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
use runtime::seqstate::HOST_BUDGET;

/// The host bytes the checkpoints of `slots` sequences hold at most,
/// beside the plan's own need: one [`HOST_BUDGET`] a sequence — every
/// resident sequence's state carries its own `Checkpoints` with the full
/// budget (a `Slot38` of the Qwen3.8 body; a GLM sequence the same), its
/// pinned buffers made as its prompt calls take points — so the rooms a
/// load's host memory must leave ([`mem_left_38`], `CacheRam::of`'s
/// prompt-cache budget) count them. The bin-shared
/// `q3place::checkpoint_bytes` holds the same rule for the qwen3 and
/// qwen35moe seats, which reserve their checkpoints in a placed plan; this
/// is the lib's own, the bins' module being out of its reach.
#[must_use]
pub fn checkpoint_bytes(slots: usize) -> u64 {
    u64::try_from(slots).map_or(u64::MAX, |n| n.saturating_mul(HOST_BUDGET))
}

/// What the host's available bytes leave the unset residency rule's room:
/// past the plan's own host need and the checkpoints the load holds beside
/// it ([`checkpoint_bytes`]) — the load's own live sequence's, the one
/// every load holds: [`residency38`] reads the plan only, and a seat's
/// further sequences are its cache budget's term (`CacheRam::of`) and its
/// `checkpoints_fit` check, which know the seat's slot count.
#[must_use]
pub fn mem_left_38(available: u64, need: u64, slots: usize) -> i128 {
    i128::from(available) - i128::from(need) - i128::from(checkpoint_bytes(slots))
}

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
/// unset, the rule's before the plan or on plan (a) from it — `off` on a
/// plan that pages routed experts through the NVMe tier's RAM arena (the
/// arena the tier attaches on), whose promotions would copy those experts
/// through the file mapping, read cold from the drive after the tier's
/// drops; else `residency38_at_plan` (P half the fewest
/// card experts a layer, `off` with why where the plan has no room, or the
/// plan's host headroom or `MemAvailable` none for the churn pool) — its
/// `residency unset` record handed to `emit`. Under `mid`, the `residency
/// host` record of `plan` follows: the churn pool (card [`CARD38`]'s
/// experts past the pinned ones) the load's host set holds beside the
/// plan's host segments, which the load refuses by name for a set word
/// when the plan's host headroom cannot take it. `emit` is where the binary
/// prints its records.
pub fn residency38(
    plan: &Plan<'_>,
    lever: Lever38<'_>,
    emit: fn(Record),
) -> Result<Residency, GateError> {
    let (residency, word) = match lever {
        Lever38::Set(r, word) => (r, word.to_owned()),
        Lever38::Unset(None) if plan.host.nvme_arena_bytes > 0 => {
            emit(record::residency_unset_paged(
                plan.host.nvme_expert_bytes,
                plan.host.nvme_arena_bytes,
            ));
            (Residency::Off, "off".to_owned())
        }
        Lever38::Unset(pre) => {
            let pick = match pre {
                Some(off) => off,
                None => {
                    // The load refuses a host set past the host's available
                    // bytes before any upload; the default leaves the pool
                    // out instead, but counts the checkpoints the load holds
                    // beside the plan's need — its own live sequence's, as
                    // `CacheRam::of` counts the same term.
                    let available = host_available()?;
                    let need = HostNeed::of(plan, 0).bytes();
                    residency38_at_plan(
                        plan.n_l.iter().copied(),
                        plan.host.experts,
                        |pinned| ChurnPool::of(plan, CARD38, pinned).map(|pool| pool.bytes),
                        plan.host.headroom_bytes,
                        mem_left_38(available, need, 1),
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
