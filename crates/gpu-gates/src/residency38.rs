//! The residency a load runs and the records that say which: a set word on
//! any model's plan ([`residency_set`], the churn pool of the card it names,
//! and [`residency_room`], the word's room on the plan's card slots;
//! `generate_glm5next`, the GLM serve seat), and a qwen4exp load's lever, set or unset by the
//! Qwen3.8 rule (`bloomery_levers::residency38_unset` before the plan,
//! `residency38_at_plan` on it; `generate_qwen3moe`, the Qwen3.8 serve seat).
//! It also owns what a load's resident sequences pin on the host for their
//! checkpoints ([`reserve_checkpoints`]): every seat declares its load's
//! ([`Seqs`]) as a host reserve row of the machine it plans on, so the plan's
//! host need holds them and the rooms read from it (the unset rule's,
//! `CacheRam`'s) take them once.

use crate::GateError;
use crate::generate::Place;
use crate::record::{self, Record};
use bloomery_gpu::host::swap::Residency;
use bloomery_levers::{PlanTier, RESIDENCY38_SPARES, Residency38Pick, residency38_at_plan};
use model::placement::churn::ChurnPool;
use model::placement::workstation::{HostNeed, TierBatchBytes, host_available};
use model::placement::{Machine, Plan};
use runtime::seqstate::HOST_BUDGET;

/// The sequences a load holds and whether they take checkpoints: `slots`
/// resident sequences, each pinning host slots for the checkpoints its prompt
/// calls take when `checkpoints` — a load's declaration of what its body
/// does ([`reserve_checkpoints`]). Every GLM load takes them
/// ([`glm_seqs`]: its prompt call takes a point at each mark with no
/// switch); a Qwen3.6 or Qwen3.8 load takes them where its seat turns them
/// on (`set_checkpoints`), and a CLI's leaves them off; V4.1's body makes no
/// checkpoints, nor does a qwen3moe file's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seqs {
    pub slots: usize,
    pub checkpoints: bool,
}

impl Seqs {
    /// One sequence taking no checkpoints: the CLI's loads and a qwen3moe
    /// file's.
    pub const ONE: Seqs = Seqs {
        slots: 1,
        checkpoints: false,
    };
}

/// The host bytes the checkpoints of `slots` sequences pin at most: one
/// [`HOST_BUDGET`] a sequence, each sequence's state carrying its own
/// `Checkpoints` with the full budget, its pinned slots made as its prompt
/// calls take points; `u64::MAX` past u64, which no host holds.
#[must_use]
pub fn checkpoint_bytes(slots: usize) -> u64 {
    u64::try_from(slots).map_or(u64::MAX, |n| n.saturating_mul(HOST_BUDGET))
}

/// The host reserve row ([`Machine::host`]'s `reserves`) that carries a
/// load's checkpoints ([`reserve_checkpoints`]).
pub const CHECKPOINTS_RESERVE: &str = "checkpoints";

/// `machine`'s host with the checkpoints `seqs` pin reserved: a
/// [`CHECKPOINTS_RESERVE`] row of [`checkpoint_bytes`] of its slots when
/// they take checkpoints, nothing when they take none. The one owner of a
/// load's checkpoint bytes: the plan's host need ([`HostNeed::bytes`]
/// counts every reserve) holds them, so the load's host check, the NVMe
/// tier's floor and arena, the unset residency rule's room ([`mem_left_38`])
/// and the prompt cache's default budget (`CacheRam`) count them once, and a
/// load that declares none counts none. A machine reserves them once: a
/// second declaration is a panic by name.
pub fn reserve_checkpoints(machine: &mut Machine, seqs: Seqs) {
    assert!(
        !machine
            .host
            .reserves
            .iter()
            .any(|(name, _)| name == CHECKPOINTS_RESERVE),
        "the machine reserves its checkpoints already: a load declares them once \
         (residency38::reserve_checkpoints)"
    );
    if seqs.checkpoints {
        machine
            .host
            .reserves
            .push((CHECKPOINTS_RESERVE.to_owned(), checkpoint_bytes(seqs.slots)));
    }
}

/// The checkpoint bytes `machine`'s host reserves
/// ([`reserve_checkpoints`]), 0 for a load that declares none: inside the
/// plan's host need, read for the lines that print the term (`CacheRam`'s
/// `checkpoints`).
#[must_use]
pub fn checkpoints_reserved(machine: &Machine) -> u64 {
    machine
        .host
        .reserves
        .iter()
        .filter(|(name, _)| name == CHECKPOINTS_RESERVE)
        .map(|&(_, b)| b)
        .sum()
}

/// What the host's available bytes leave the unset residency rule's room
/// past the plan's host `need` ([`HostNeed::bytes`]), the load's
/// checkpoints inside it ([`reserve_checkpoints`]).
#[must_use]
pub fn mem_left_38(available: u64, need: u64) -> i128 {
    i128::from(available) - i128::from(need)
}

/// The plan's card a qwen4exp open loads: its one card.
pub const CARD38: usize = 0;

/// `plan`'s NVMe tier terms the unset residency rules read
/// ([`PlanTier`]): the routed-expert bytes it pages through the tier and the
/// RAM arena they page through, 0 and 0 on a plan with no tier.
#[must_use]
pub fn plan_tier(plan: &Plan<'_>) -> PlanTier {
    PlanTier {
        paged: plan.host.nvme_expert_bytes,
        arena: plan.host.nvme_arena_bytes,
    }
}

/// The plan's card GLM's residency machine runs over: plan (a)'s one card.
pub const GLM_CARD: usize = 0;

/// What a GLM load of `slots` resident sequences declares
/// ([`reserve_checkpoints`]): every slot takes checkpoints, the GLM body's
/// prompt call taking a point at each mark with no switch and every
/// sequence it makes carrying its own `Checkpoints` of `HOST_BUDGET`.
#[must_use]
pub fn glm_seqs(slots: usize) -> Seqs {
    Seqs {
        slots,
        checkpoints: true,
    }
}

/// `place`'s machine of a layer count (`Place::machine`, `batch` the expert
/// tier's prompt-batch bytes) for a GLM load of `slots` resident sequences,
/// every machine it makes declaring them ([`glm_seqs`]): the machine a GLM
/// load's offer, plans and open take, so each counts the checkpoints its
/// sequences pin.
pub fn glm_machine(
    place: Place,
    batch: Option<TierBatchBytes>,
    slots: usize,
) -> Result<impl Fn(usize) -> Machine + Copy, GateError> {
    let base = place.machine(None, batch)?;
    Ok(move |layers| {
        let mut machine = base(layers);
        reserve_checkpoints(&mut machine, glm_seqs(slots));
        machine
    })
}

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
/// (`residency38_at_plan`: `off` on a plan that pages routed experts through
/// the NVMe tier's RAM arena, whose promotions would copy those experts
/// through the file mapping, read cold from the drive after the tier's
/// drops; else P half the fewest card experts a layer, `off` with why where
/// the plan has no room, or the plan's host headroom or `MemAvailable` none
/// for the churn pool) — its `residency unset` record handed to `emit`.
/// Under `mid`, the `residency host` record of `plan` follows: the churn
/// pool (card [`CARD38`]'s experts past the pinned ones) the load's host set
/// holds beside the plan's host segments, which the load refuses by name for
/// a set word when the plan's host headroom cannot take it. `emit` is where
/// the binary prints its records.
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
                    // The load refuses a host set past the host's available
                    // bytes before any upload; the default leaves the pool
                    // out instead. The need holds the load's checkpoints
                    // ([`reserve_checkpoints`]), as `CacheRam`'s does.
                    let available = host_available()?;
                    let need = HostNeed::of(plan, 0).bytes();
                    residency38_at_plan(
                        plan.n_l.iter().copied(),
                        plan.host.experts,
                        plan_tier(plan),
                        |pinned| ChurnPool::of(plan, CARD38, pinned).map(|pool| pool.bytes),
                        plan.host.headroom_bytes,
                        mem_left_38(available, need),
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
