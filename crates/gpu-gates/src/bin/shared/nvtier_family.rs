//! The NVMe expert tier's family gate: the five clauses every family that attaches the tier proves, over a
//! family-neutral handle ([`Family`]). A family's bin supplies the plan of its file at a room and an arena
//! ([`Family::plan`]), the load by that plan ([`Family::load`]) with its tier handle ([`Family::tier`]), the token
//! run ([`Family::run`]: one prompt call, then greedy steps, each position's argmax and logits digest) and, beside
//! them, the refusals only it has ([`Family::refusals`], a family without one returns none).
//!
//! The arms are planned from the file, never from a literal: the all-RAM plan at the room its need fits fixes the
//! arena's slot set, the paged arm's arena is [`SLOTS`] slot sets, and its room is the split's floor F (read off the
//! `HostRoomFloor` refusal at a [`PROBE_ROOM`] room under that arena) plus the arena. The two arms run one after the
//! other on the same ids; the clauses are judged on what each snapshotted:
//!
//! - `plan`: the all-RAM plan pages nothing; the paged plan pages routed experts through an arena of exactly the
//!   given bytes, [`SLOTS`] slots, which the tier carved too (`NvTierStats::slots_per_layer`). The room below the
//!   floor is the probe's refusal, so a plan that never splits fails here.
//! - `attach`: the load built a tier, it covers every paged layer, the carve is at least the arena's floor and under
//!   the ids a paged layer serves (a layer that serves no more than it holds slots for never evicts), and a run's
//!   misses, fills, evictions and prompt union claims are above 0 while no arena id was read through the file mapping.
//! - `bits`: every position's argmax and logits digest equal the all-RAM arm's.
//! - `refuse`: by name, a room under the floor and an arena one slot set under the arena's floor (both at plan time),
//!   and the family's own refusals.
//! - `residency`: the unset residency rule on the paged plan is `off`, with why `paged`.

use std::sync::Arc;

use bloomery_gpu::host::nvtier::{NvTier, NvTierStats};
use bloomery_gpu_gates::{GateError, checks_failed, record, tier, verdict};
use model::placement::workstation::HostNeed;
use model::placement::{
    ARENA_COLUMNS, ArenaFloor, Device, PlacementError, Plan, Role, arena_slots,
};

/// The slots a paged layer's arena holds, in the arm the gate pages: above the arena's floor, under the ids a layer
/// serves, so evictions happen.
pub const SLOTS: u64 = 64;

/// The room the floor is read at: under every file's floor.
const PROBE_ROOM: u64 = 1 << 30;

/// One arm's placement inputs: the host room the plan splits by and the arena's given bytes (`None`: the dial's).
#[derive(Clone, Copy, Debug)]
pub struct Arm {
    pub room: u64,
    pub arena: Option<u64>,
}

/// One position's output: its argmax and its logits' FNV-1a 64.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pos {
    pub argmax: u32,
    pub fnv: u64,
}

/// What a plan says of the tier, owned: the family reads it off its plan with [`PlanFacts::of`].
#[derive(Clone, Debug)]
pub struct PlanFacts {
    pub paged: u64,
    pub arena: u64,
    pub arena_slots: u64,
    pub need: u64,
    pub floor: ArenaFloor,
    /// The layers whose routed experts the plan pages (an NVMe segment), ascending.
    pub paged_layers: Vec<usize>,
    /// The expert ids a layer holds.
    pub experts: u32,
}

impl PlanFacts {
    /// # Errors
    /// The arena's slots or floor refused by the placement, or an expert count past `u32`.
    pub fn of(plan: &Plan<'_>) -> Result<PlanFacts, GateError> {
        let model = plan.model;
        let mut paged_layers: Vec<usize> = plan
            .rows
            .iter()
            .filter_map(|r| {
                let t = &model.tensors[r.tensor];
                (t.role == Role::RoutedExperts
                    && r.segments.iter().any(|s| s.device == Device::Nvme))
                .then_some(t.layer)
                .flatten()
            })
            .collect();
        paged_layers.sort_unstable();
        paged_layers.dedup();
        Ok(PlanFacts {
            paged: plan.host.nvme_expert_bytes,
            arena: plan.host.nvme_arena_bytes,
            arena_slots: arena_slots(plan)?,
            need: HostNeed::of(plan, 0).bytes(),
            floor: ArenaFloor::of(plan, ARENA_COLUMNS)?,
            paged_layers,
            experts: u32::try_from(model.experts)?,
        })
    }
}

/// What the paged arm's tier held, owned before its session drops: the tier's arena stays alive only while a
/// session holds it, so two loads never stand at once.
#[derive(Clone, Debug)]
pub struct TierFacts {
    /// The slots a paged layer holds, as the tier carved them.
    pub slots: u64,
    /// The paged layers the arena covers.
    pub covered: Vec<usize>,
    /// The ids a covered layer's arena serves, per covered layer.
    pub served: Vec<usize>,
    pub before: NvTierStats,
    pub after: NvTierStats,
}

impl TierFacts {
    fn read(t: &NvTier, plan: &PlanFacts, before: NvTierStats) -> TierFacts {
        let covered: Vec<usize> = plan
            .paged_layers
            .iter()
            .copied()
            .filter(|&l| t.covers(l))
            .collect();
        let served = covered
            .iter()
            .map(|&l| (0..plan.experts).filter(|&id| t.serves(l, id)).count())
            .collect();
        let after = t.stats();
        TierFacts {
            slots: after.slots_per_layer,
            covered,
            served,
            before,
            after,
        }
    }

    fn delta(&self, f: impl Fn(&NvTierStats) -> u64) -> u64 {
        f(&self.after).saturating_sub(f(&self.before))
    }
}

/// A refusal only one family has: the load `attempt` must fail with an error whose text holds `holds`; `Ok` is a
/// load that was not refused.
pub struct Refusal<'a> {
    pub what: &'static str,
    pub holds: &'static str,
    pub attempt: Box<dyn Fn() -> Result<(), GateError> + 'a>,
}

/// What a family supplies.
pub trait Family {
    /// The family's open session.
    type Session;

    /// The family's plan of its file at `arm`, as facts. The inner error is the planner's own refusal, the outer
    /// any other failure.
    ///
    /// # Errors
    /// A file or machine the plan cannot be asked of.
    fn plan(&self, arm: Arm) -> Result<Result<PlanFacts, PlacementError>, GateError>;

    /// The load by that plan, `r8` as the load's r8 lever.
    ///
    /// # Errors
    /// The load's refusal.
    fn load(&self, arm: Arm, r8: bool) -> Result<Self::Session, GateError>;

    /// The load's NVMe tier, `None` on a load with none.
    ///
    /// # Errors
    /// A session that holds no body.
    fn tier(&self, s: &Self::Session) -> Result<Option<Arc<NvTier>>, GateError>;

    /// One prompt call and the greedy steps after it, every position's output.
    ///
    /// # Errors
    /// The run's failure.
    fn run(&self, s: &mut Self::Session) -> Result<Vec<Pos>, GateError>;

    /// The session cleared and dropped.
    ///
    /// # Errors
    /// The teardown's failure.
    fn finish(&self, s: Self::Session) -> Result<(), GateError>;

    /// The unset residency word and why on the plan at `arm`: its word and the why's name.
    ///
    /// # Errors
    /// The rule's refusal.
    fn unset_residency(&self, arm: Arm) -> Result<(String, &'static str), GateError>;

    /// The family's own refusals on the paged arm `paged`.
    fn refusals(&self, _paged: Arm) -> Vec<Refusal<'_>> {
        Vec::new()
    }

    /// Whether the file keeps routed layers on the host in a number the tier can page (the family's
    /// count over the file's types, never the plan's placement): the clauses that read the paged
    /// experts are left by name ([`tier::host_clause`]) on a fixture where it is false. An
    /// approximation of the plan's own condition; a miss is a red clause, never a deferral.
    ///
    /// # Errors
    /// A file the count cannot be read from.
    fn host_layers_enough(&self) -> Result<bool, GateError> {
        Ok(true)
    }
}

/// What one arm left behind.
struct Ran {
    plan: PlanFacts,
    run: Vec<Pos>,
    tier: Option<TierFacts>,
}

/// The arm `arm` planned, loaded, run once and torn down.
fn arm_run<F: Family>(f: &F, name: &str, arm: Arm, plan: PlanFacts) -> Result<Ran, GateError> {
    let mut s = f.load(arm, false)?;
    let tier = f.tier(&s)?;
    let before = tier.as_ref().map(|t| t.stats());
    let run = f.run(&mut s)?;
    let facts = tier
        .as_ref()
        .zip(before)
        .map(|(t, b)| TierFacts::read(t, &plan, b));
    if let Some(r) = record::nvtier_of(tier.as_deref()) {
        r.print();
    }
    drop(tier);
    f.finish(s)?;
    println!(
        "{name}: arm room {} B arena {:?}: {} positions, {} paged layers, tier {}",
        arm.room,
        arm.arena,
        run.len(),
        plan.paged_layers.len(),
        if facts.is_some() { "built" } else { "none" }
    );
    Ok(Ran {
        plan,
        run,
        tier: facts,
    })
}

/// A clause's verdict line.
fn clause(name: &str, what: &str, ok: bool) -> bool {
    println!("{name}: {what}: {}", verdict(ok));
    ok
}

/// `plan`: see the module doc.
fn plan_clause(
    name: &str,
    ram: &PlanFacts,
    paged: &PlanFacts,
    a: u64,
    tier: Option<&TierFacts>,
) -> bool {
    let slot_set = ram.floor.slot_set;
    let carve = tier.is_none_or(|t| t.slots == paged.arena_slots);
    let ok = ram.paged == 0
        && ram.arena == 0
        && paged.paged > 0
        && paged.arena == a
        && paged.arena_slots == SLOTS
        && SLOTS >= ram.floor.slots
        && carve;
    clause(
        name,
        &format!(
            "all-RAM pages {} B; the paged plan pages {} B through an arena of {} B = {} slots of {slot_set} B \
             (given {a} B, {SLOTS} wanted, floor {}), the tier carved {:?}",
            ram.paged,
            paged.paged,
            paged.arena,
            paged.arena_slots,
            ram.floor.slots,
            tier.map(|t| t.slots)
        ),
        ok,
    )
}

/// `attach`: see the module doc.
fn attach_clause(name: &str, paged: &PlanFacts, tier: Option<&TierFacts>) -> bool {
    let Some(t) = tier else {
        return clause(name, "the paged plan built no arena", false);
    };
    let covers = t.covered == paged.paged_layers && !paged.paged_layers.is_empty();
    let least = t.served.iter().copied().min().unwrap_or(0);
    let carve = t.slots >= paged.floor.slots && (t.slots as usize) < least;
    let (misses, fills, evictions) = (
        t.delta(|s| s.misses),
        t.delta(|s| s.fills),
        t.delta(|s| s.evictions),
    );
    // `slot_reads` and `mapping_reads` are the residency lane's reads of an arena id, so a run with the residency
    // off has none of the first: the host leg's own traffic is the fills and the prompt's union claims.
    let (unions, slot_reads, mapping) = (
        t.delta(|s| s.union_small + s.union_batched),
        t.delta(|s| s.slot_reads),
        t.delta(|s| s.mapping_reads),
    );
    let counts = misses > 0 && fills > 0 && evictions > 0 && unions > 0 && mapping == 0;
    clause(
        name,
        &format!(
            "the tier covers {} of {} paged layers ({covers}); the carve {} slots lies in [{}, {least}) ids a \
             layer ({carve}); a run's misses {misses}, fills {fills}, evictions {evictions}, prompt unions {unions}, \
             mapping reads {mapping} (counts {counts}; slot reads {slot_reads}, the residency lane's)",
            t.covered.len(),
            paged.paged_layers.len(),
            t.slots,
            paged.floor.slots
        ),
        covers && carve && counts,
    )
}

/// `bits`: see the module doc.
fn bits_clause(name: &str, ram: &[Pos], paged: &[Pos]) -> bool {
    let first = ram.iter().zip(paged).position(|(a, b)| a != b);
    let ok = ram.len() == paged.len() && !ram.is_empty() && first.is_none();
    clause(
        name,
        &format!(
            "{} positions against {}, first difference {first:?}, last digest {:016x}",
            ram.len(),
            paged.len(),
            paged.last().map_or(0, |p| p.fnv)
        ),
        ok,
    )
}

/// One refusal line: the result of an attempt must be an error whose text holds `holds`.
fn refused(what: &str, holds: &str, got: Result<(), GateError>) -> bool {
    let (ok, text) = match got {
        Err(e) => (e.to_string().contains(holds), e.to_string()),
        Ok(()) => (false, "not refused".to_string()),
    };
    println!("refuse {what}: {text}");
    println!("refuse {what}: holds {holds:?}: {}", verdict(ok));
    ok
}

/// The gate: `f`'s file planned, its two arms run, the five clauses judged, none stopping the others after the
/// arms ran.
///
/// # Errors
/// A clause that is red ([`checks_failed`]), or a failure the clauses cannot name: a probe, a load, a run.
pub fn gate<F: Family>(name: &str, f: &F) -> Result<(), GateError> {
    let host = f.host_layers_enough()?;
    let plan_runs = tier::host_clause(
        "plan: the paged plan pages through an arena of exactly the given slots, all-RAM pages nothing",
        host,
    )?;
    let probe = |arm| f.plan(arm);
    // The base: a room no need passes, so no split runs and the plan's need and arena floor are the file's.
    let base = probe(Arm {
        room: u64::MAX,
        arena: None,
    })?
    .map_err(|e| format!("{name}: the all-RAM plan: {e}"))?;
    let a = SLOTS
        .checked_mul(base.floor.slot_set)
        .ok_or("the arena's bytes pass u64")?;
    let ram_arm = Arm {
        room: base.need,
        arena: None,
    };
    let ram_plan =
        probe(ram_arm)?.map_err(|e| format!("{name}: the all-RAM plan at its need: {e}"))?;
    // F: the floor a room under the split's floor is refused by name with, under the given arena.
    let low = Arm {
        room: PROBE_ROOM,
        arena: Some(a),
    };
    let floor = match probe(low)? {
        Err(PlacementError::HostRoomFloor { floor, .. }) => Some(floor),
        Err(e) => return Err(format!("{name}: the {PROBE_ROOM} B room's probe: {e}").into()),
        Ok(_) => None,
    };
    let Some(floor) = floor else {
        clause(
            "plan",
            &format!("a {PROBE_ROOM} B room is planned, not refused: no split ran on the plan"),
            false,
        );
        println!("{name}: {}", tier::tally_line());
        return Err(checks_failed());
    };
    let paged_arm = Arm {
        room: floor + a,
        arena: Some(a),
    };
    let paged_plan = match probe(paged_arm)? {
        Ok(p) => p,
        Err(e) => {
            clause(
                "plan",
                &format!("the paged plan at room {} B: {e}", paged_arm.room),
                false,
            );
            println!("{name}: {}", tier::tally_line());
            return Err(checks_failed());
        }
    };
    println!(
        "{name}: floor F {floor} B, arena A {a} B ({SLOTS} slots of {} B), paged room {} B, all-RAM room {} B",
        base.floor.slot_set, paged_arm.room, ram_arm.room
    );
    // One load at a time: the all-RAM arm's session is gone before the paged arm loads.
    let ram = arm_run(f, name, ram_arm, ram_plan)?;
    let paged = arm_run(f, name, paged_arm, paged_plan)?;
    let mut pass = true;
    if plan_runs {
        pass &= plan_clause("plan", &ram.plan, &paged.plan, a, paged.tier.as_ref());
    }
    if tier::host_clause(
        "attach: the tier is built, covers every paged layer and counts a run's traffic",
        host,
    )? {
        pass &= attach_clause("attach", &paged.plan, paged.tier.as_ref());
    }
    tier::sc("bits: every position equals the all-RAM arm's")?;
    pass &= bits_clause("bits", &ram.run, &paged.run);
    if tier::host_clause(
        "refuse: a room under the floor, an arena under the arena's floor and the family's own, by name",
        host,
    )? {
        let under = (base.floor.slots - 1) * base.floor.slot_set;
        pass &= refused(
            "room",
            "the NVMe expert tier's floor",
            probe(low)?.map(|_| ()).map_err(GateError::from),
        );
        pass &= refused(
            "arena",
            "is under its floor",
            probe(Arm {
                room: paged_arm.room,
                arena: Some(under),
            })?
            .map(|_| ())
            .map_err(GateError::from),
        );
        for r in f.refusals(paged_arm) {
            pass &= refused(r.what, r.holds, (r.attempt)());
        }
    }
    if tier::host_clause(
        "residency: the unset rule on the paged plan is off, why paged",
        host,
    )? {
        let (word, why) = f.unset_residency(paged_arm)?;
        pass &= clause(
            "residency",
            &format!("the unset word on the paged plan is {word:?}, why {why:?}"),
            word == "off" && why == "paged",
        );
    }
    println!("{name}: {}", tier::tally_line());
    if pass {
        println!("{name}: every clause passed");
        Ok(())
    } else {
        Err(checks_failed())
    }
}
