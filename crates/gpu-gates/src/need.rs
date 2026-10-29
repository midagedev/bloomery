//! The `need` record of a placed load, gathered where its plan is printed:
//! the plan's host need by page class, the serve's prompt cache, each card's
//! bytes and the worker pool's placement. [`print`] writes the line beside the
//! `plan` record and stages it for the need hook (`BLOOMERY_NEED_HOOK`), which
//! the placed load consults before it opens a card. The arithmetic ([`of_with`])
//! takes every input as a value; the readers of this process's levers and pool
//! ([`of`], [`staged`], [`print`]) come with the card.

#[cfg(feature = "gpu")]
use bloomery_levers::{need_levers, nvtier_levers};
use model::placement::Plan;
#[cfg(feature = "gpu")]
use model::placement::need_hook;
use model::placement::workstation::{HostNeed, Hosting};

use crate::GateError;
#[cfg(feature = "gpu")]
use crate::record::{self, Record};
use crate::record::{Need, PoolPlace};

/// The need of a load of `plan`, with `extra` bytes the host set holds beside
/// the plan's own (the churn pool and what the load keeps beside the plan),
/// `cache_ram` of prompt cache (0 for a binary with none), the way the load
/// holds its set (`how`) and the pool placed as `pool` says.
pub fn of_with(
    plan: &Plan<'_>,
    extra: u64,
    cache_ram: u64,
    how: Hosting,
    pool: PoolPlace,
) -> Result<Need, GateError> {
    let need = HostNeed::of(plan, extra);
    let classes = need.classes(plan, how)?;
    if classes.bytes() != need.bytes() {
        return Err(format!(
            "the host need's page classes sum to {} B and the need to {} B",
            classes.bytes(),
            need.bytes()
        )
        .into());
    }
    Ok(Need {
        bytes: need.bytes(),
        classes,
        os: need.os,
        cache_ram,
        lock: how.lock,
        cards: plan
            .machine
            .all_cards()
            .zip(&plan.cards)
            .map(|(card, totals)| (card.name.clone(), totals.bytes()))
            .collect(),
        pool,
    })
}

/// [`of_with`] as this process holds its set: `BLOOMERY_HOST_LOCK`, the
/// arena's memory by `BLOOMERY_NVTIER_MEM` on the plan's device, and the pool
/// [`threads::report`] places.
#[cfg(feature = "gpu")]
pub fn of(plan: &Plan<'_>, extra: u64, cache_ram: u64) -> Result<Need, GateError> {
    let levers = need_levers()?;
    let nvtier = nvtier_levers()?;
    let how = Hosting {
        lock: levers.host_lock,
        arena_pinned: nvtier.mem.pinned(plan.machine.unified().is_some()),
    };
    let pool = threads::report()?;
    let place = PoolPlace {
        threads: pool.threads,
        cpus: pool.workers,
        caller: pool.caller,
        allowed: pool.allowed,
    };
    of_with(plan, extra, cache_ram, how, place)
}

/// The `need` record of `plan`, staged for the load's hook: the caller prints
/// it where the plan's own record is printed. A caller that prints a plan's
/// records itself ([`print`] is the one that prints) builds this beside them.
#[cfg(feature = "gpu")]
pub fn staged(plan: &Plan<'_>, extra: u64, cache_ram: u64) -> Result<Record, GateError> {
    record::require(record::groups::need::PRINT, "need::staged");
    let need = of(plan, extra, cache_ram)?;
    crate::start::note_need(
        need.bytes,
        &need.cards,
        plan.host.expert_bytes + plan.host.nvme_expert_bytes,
        plan.host.nvme_expert_bytes,
    );
    need_hook::stage(&record::need(&need).line());
    Ok(record::need(&need))
}

/// Print the `need` record of `plan` (on stderr when `err`, else stdout) and
/// stage it for the load's hook. Called where the plan's own record is
/// printed, once the churn pool (`extra`) and the prompt cache (`cache_ram`)
/// are known.
#[cfg(feature = "gpu")]
pub fn print(plan: &Plan<'_>, extra: u64, cache_ram: u64, err: bool) -> Result<(), GateError> {
    staged(plan, extra, cache_ram)?.emit(err);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{self, BLOOMERY_SERVE_DS41, Log, NEED};
    use model::placement::{KvBytes, ModelTensors, plan_with, workstation};

    struct NoKv;
    impl KvBytes for NoKv {
        fn layer_bytes(&self, _: usize, _: u64) -> u64 {
            0
        }
    }

    /// A one-layer plan on the A6000 with the host terms a load of the
    /// V4.1 shape has: shadows, a row cache, an arena, tables.
    fn fixture() -> (ModelTensors, workstation::CardSpec) {
        (
            ModelTensors {
                tensors: Vec::new(),
                layers: 1,
                experts: 8,
                experts_used: 2,
                bf16_words: |_| false,
            },
            workstation::A6000,
        )
    }

    fn pool() -> PoolPlace {
        PoolPlace {
            threads: 4,
            cpus: vec![0, 8, 16],
            caller: 24,
            allowed: (0..32).collect(),
        }
    }

    /// The record's fields from a fixture plan: the host need by class, the
    /// OS reserve out, the prompt cache beside it, the card's bytes, the
    /// lock, and the pool's placement as given; the line reads back whole by
    /// its kind.
    #[test]
    fn the_need_record_states_the_plans_terms() {
        let (model, spec) = fixture();
        let mut machine = workstation::plan_on(spec, 1);
        machine
            .host_mut()
            .reserves
            .push((workstation::TIER_BATCH_HOST_RESERVE.to_string(), 40));
        let mut plan = plan_with(&model, &machine, 1, &NoKv, None).expect("plans");
        plan.host.shadow_bytes = 1000;
        plan.host.nvme_arena_bytes = 70;
        plan.host.table_bytes = 300;
        plan.host.expert_bytes = 5000;
        let extra = 6;
        let base = Hosting {
            lock: false,
            arena_pinned: false,
        };
        let n = of_with(&plan, extra, 123, base, pool()).expect("a need");
        let want = HostNeed::of(&plan, extra);
        assert_eq!(n.bytes, want.bytes());
        assert_eq!(n.os, workstation::OS_OTHER);
        assert_eq!(
            (n.classes.pinned, n.classes.anon, n.classes.locked),
            (1000 + 40, 70, 0)
        );
        assert_eq!(n.classes.cache, 5000 + 300 + extra);
        assert_eq!(n.cache_ram, 123);
        assert!(!n.lock);
        assert_eq!(n.cards.len(), 1);
        assert_eq!(n.cards[0], ("A6000".to_string(), plan.cards[0].bytes()));
        assert_eq!((n.pool.threads, n.pool.caller), (4, 24));
        assert_eq!(n.pool.cpus, vec![0, 8, 16]);
        // A lock moves the set's file pages to the locked class; a pinned
        // arena joins the pinned bytes.
        let locked = Hosting {
            lock: true,
            arena_pinned: true,
        };
        let l = of_with(&plan, extra, 0, locked, pool()).expect("a need");
        assert_eq!(l.bytes, n.bytes);
        assert_eq!(l.classes.locked, 5000 + 300 + extra);
        assert_eq!((l.classes.cache, l.classes.anon), (0, 0));
        assert_eq!(l.classes.pinned, 1000 + 40 + 70);
        assert!(l.lock);
        // The line carries each field, and reads back by the kind.
        let line = record::need(&n).line();
        assert!(line.starts_with("need bytes="), "{line}");
        let log = Log::of(&line, BLOOMERY_SERVE_DS41);
        let f = log.one(&NEED).expect("the record reads whole");
        assert_eq!(f.u64("bytes"), Ok(n.bytes));
        assert_eq!(f.u64("pinned"), Ok(1040));
        assert_eq!(f.u64("cache_ram"), Ok(123));
        assert_eq!(f.bool("lock"), Ok(false));
        assert_eq!(f.csv("cpus"), Ok(vec!["0", "8", "16"]));
        assert_eq!(f.u64("caller"), Ok(24));
        assert_eq!(f.csv("allowed").map(|a| a.len()), Ok(32));
        let cards = f.csv("cards").expect("cards");
        assert_eq!(cards, vec![format!("A6000:{}", plan.cards[0].bytes())]);
    }

    /// A need whose classes do not sum to it, or a reserve nobody classed,
    /// is refused by name and prints nothing.
    #[test]
    fn a_reserve_without_a_class_is_refused() {
        let (model, spec) = fixture();
        let mut machine = workstation::plan_on(spec, 1);
        machine
            .host_mut()
            .reserves
            .push(("an unclassed reserve".to_string(), 9));
        let plan = plan_with(&model, &machine, 1, &NoKv, None).expect("plans");
        let how = Hosting {
            lock: false,
            arena_pinned: false,
        };
        let err = of_with(&plan, 0, 0, how, pool()).expect_err("no class");
        assert!(err.to_string().contains("an unclassed reserve"), "{err}");
    }

    /// Every host reserve a placed load declares besides the OS's and the
    /// tier's rows has a class, pinned: the checkpoints
    /// (`residency38::reserve_checkpoints`) and, under the MTP draft, their
    /// hidden rows. The classes still sum to the need.
    #[test]
    fn the_reserves_a_load_declares_for_its_checkpoints_are_pinned() {
        let (model, spec) = fixture();
        let how = Hosting {
            lock: false,
            arena_pinned: false,
        };
        let need_of = |reserve: Option<&str>| {
            let mut machine = workstation::plan_on(spec, 1);
            if let Some(name) = reserve {
                machine.host_mut().reserves.push((name.to_string(), 77));
            }
            let plan = plan_with(&model, &machine, 1, &NoKv, None).expect("plans");
            of_with(&plan, 0, 0, how, pool()).expect("a need")
        };
        let without = need_of(None);
        for name in [
            workstation::CHECKPOINTS_RESERVE,
            workstation::POINT_ROWS_RESERVE,
        ] {
            let with = need_of(Some(name));
            assert_eq!(with.bytes, without.bytes + 77, "{name}");
            assert_eq!(with.classes.pinned, without.classes.pinned + 77, "{name}");
            assert_eq!(with.classes.cache, without.classes.cache, "{name}");
        }
    }
}
