//! The residency a load runs and the records that say which: a set word on
//! any model's plan ([`residency_set`], the churn pool of the card it names,
//! and [`residency_room`], the word's room on the plan's card slots;
//! `generate_glm5next`, the GLM serve seat), and a qwen4exp load's lever, set or unset by the
//! Qwen3.8 rule (`bloomery_levers::residency38_unset` before the plan,
//! `residency38_at_plan` on it; `generate_qwen3moe`, the Qwen3.8 serve seat).
//! It also owns what a load's resident sequences pin on the host for their
//! checkpoints ([`reserve_checkpoints`]): every seat declares its load's
//! ([`Seqs`]) as a host reserve row of the machine it plans on, priced at the
//! context that machine is laid for, so the plan's host need holds them and
//! the rooms read from it (the unset rule's, `CacheRam`'s) take them once.

use crate::GateError;
use crate::generate::Place;
use crate::record::{self, Record};
use crate::residency::PlanRoom;
use bloomery_gpu::host::swap::Residency;
use bloomery_levers::{RESIDENCY38_SPARES, Residency38Pick, residency38_at_plan};
use model::placement::churn::ChurnPool;
use model::placement::workstation::TierBatchBytes;
use model::placement::{Machine, Plan};
use runtime::seqstate::{Price, pinned};

/// The sequences a load holds and what their checkpoints cost: `slots`
/// resident sequences, each pinning host slots for the checkpoints its prompt
/// calls take when `checkpoints` carries the family's price — a load's
/// declaration of what its body does ([`reserve_checkpoints`]). A Qwen3.6,
/// Qwen3.8 or GLM load takes them where its seat turns them on (the body's
/// `set_checkpoints`, GLM's on every load), and a CLI's leaves them off;
/// V4.1's body makes no checkpoints, nor does a qwen3moe file's. Each family
/// builds its price from the derivation its body makes its sequences'
/// `Checkpoints` with (`checkpoint_price`), so a sequence pins what the
/// machine reserved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seqs {
    pub slots: usize,
    pub checkpoints: Option<Price>,
}

impl Seqs {
    /// One sequence taking no checkpoints: the CLI's loads and a qwen3moe
    /// file's.
    pub const ONE: Seqs = Seqs::none(1);

    /// `slots` sequences taking no checkpoints.
    #[must_use]
    pub const fn none(slots: usize) -> Seqs {
        Seqs {
            slots,
            checkpoints: None,
        }
    }

    /// `slots` sequences taking checkpoints of `price`.
    #[must_use]
    pub const fn priced(slots: usize, price: Price) -> Seqs {
        Seqs {
            slots,
            checkpoints: Some(price),
        }
    }
}

/// The host reserve row ([`Machine::host`]'s `reserves`) that carries a
/// load's checkpoints ([`reserve_checkpoints`]).
pub const CHECKPOINTS_RESERVE: &str = "checkpoints";

/// The host bytes `seqs` pin at most when each sequence's load is of
/// `slot_ctx` positions: `slots` sequences of the price's budget at that
/// context ([`Price::budget`]); 0 for sequences that take no checkpoints.
/// Refused by name for a context the rule refuses.
pub fn checkpoints_pinned(seqs: Seqs, slot_ctx: u64) -> Result<u64, GateError> {
    let Some(price) = seqs.checkpoints else {
        return Ok(0);
    };
    let budget = price
        .budget(slot_ctx)
        .map_err(|e| format!("the checkpoint reserve at {slot_ctx} positions a slot: {e}"))?;
    Ok(pinned(seqs.slots, budget))
}

/// `machine`'s host with the checkpoints `seqs` pin reserved: a
/// [`CHECKPOINTS_RESERVE`] row of [`checkpoints_pinned`] at `slot_ctx`
/// positions a slot — the context each sequence's body is opened with, the
/// one the machine is laid for — when they take checkpoints, nothing when
/// they take none. The one owner of a load's checkpoint bytes: the plan's
/// host need (`HostNeed::bytes` counts every reserve) holds them, so the
/// load's host check, the NVMe tier's floor and arena, the unset residency
/// rule's room ([`mem_left_38`]) and the prompt cache's default budget
/// (`CacheRam`) count them once, and a load that declares none counts none.
/// Every machine builder prices its own context, so no probe of a context
/// search reserves a figure priced at another. A machine reserves them once:
/// a second declaration is a panic by name.
pub fn reserve_checkpoints(
    machine: &mut Machine,
    seqs: Seqs,
    slot_ctx: u64,
) -> Result<(), GateError> {
    let bytes = checkpoints_pinned(seqs, slot_ctx)?;
    declare_checkpoints(machine, seqs.checkpoints.map(|_| bytes));
    Ok(())
}

/// The one row a machine carries for its checkpoints: `bytes`, or none for
/// a load that takes none. A second declaration is a panic by name.
fn declare_checkpoints(machine: &mut Machine, bytes: Option<u64>) {
    assert!(
        !machine
            .host
            .reserves
            .iter()
            .any(|(name, _)| name == CHECKPOINTS_RESERVE),
        "the machine reserves its checkpoints already: a load declares them once \
         (residency38::reserve_checkpoints)"
    );
    if let Some(bytes) = bytes {
        machine
            .host
            .reserves
            .push((CHECKPOINTS_RESERVE.to_owned(), bytes));
    }
}

/// `machine`'s checkpoint reserve against what a load of `slots` sequences
/// of `budget` bytes pins (the body's `checkpoint_budget` at the context the
/// load is granted): refused by name when the reserve is short of it — the
/// plan was made for a smaller context than the body's sequences are made
/// for, and the host holds bytes the plan did not count. The one check of
/// every family's load.
pub fn check_reserve(machine: &Machine, slots: usize, budget: u64) -> Result<(), GateError> {
    let (reserved, pins) = (checkpoints_reserved(machine), pinned(slots, budget));
    if reserved < pins {
        return Err(format!(
            "the plan's machine reserves {reserved} B of checkpoints; the load's {slots} \
             sequences pin up to {pins} B ({budget} B each, the body's budget at the context \
             it is granted)"
        )
        .into());
    }
    Ok(())
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
/// past the plan's host `need` (`HostNeed::bytes`), the load's
/// checkpoints inside it ([`reserve_checkpoints`]).
#[must_use]
pub fn mem_left_38(available: u64, need: u64) -> i128 {
    i128::from(available) - i128::from(need)
}

/// The plan's card a qwen4exp open loads: its one card.
pub const CARD38: usize = 0;

/// The plan's card GLM's residency machine runs over: plan (a)'s one card.
pub const GLM_CARD: usize = 0;

/// `place`'s machine of a layer count (`Place::machine`, `batch` the expert
/// tier's prompt-batch bytes) for a GLM load of `slots` resident sequences
/// whose checkpoints cost `price` (`bloomery_gpu_glm5next::checkpoint_price`),
/// each of `slot_ctx` positions, every machine it makes declaring them
/// ([`reserve_checkpoints`]): the machine a GLM load's offer, plans and open
/// take, so each counts the checkpoints its sequences pin. The price is taken
/// as bytes here because this module is built without the GLM device crate;
/// a refusal of the rule is the call's, before any machine is made. Every GLM
/// load takes checkpoints: its prompt call takes a point at each mark with no
/// switch.
pub fn glm_machine(
    place: Place,
    batch: Option<TierBatchBytes>,
    slots: usize,
    price: Price,
    slot_ctx: u64,
) -> Result<impl Fn(usize) -> Machine + Copy, GateError> {
    let base = place.machine(None, batch)?;
    let bytes = checkpoints_pinned(Seqs::priced(slots, price), slot_ctx)?;
    Ok(move |layers| {
        let mut machine = base(layers);
        declare_checkpoints(&mut machine, Some(bytes));
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
/// (`residency38_at_plan`: P 0, or the first P the plan's card slots, host
/// headroom and `MemAvailable` leave a churn pool for, `off` with why where
/// they leave none) — its `residency unset` record handed to `emit`.
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
                    let room = PlanRoom::new(plan, CARD38, 0);
                    residency38_at_plan(
                        room.card_experts(),
                        plan.host.experts,
                        |pinned| room.pool_bytes(pinned),
                        room.headroom(),
                        room.mem_left()?,
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
