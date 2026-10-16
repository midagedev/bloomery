//! The NVMe expert tier's family gate: the six clauses every family that attaches the tier proves, over a
//! family-neutral handle ([`Family`]). A family's bin supplies the plan of its file at a room and an arena
//! ([`Family::plan`]), the load by that plan ([`Family::load`]) with its tier handle ([`Family::tier`]), the token
//! run ([`Family::run`]: one prompt call, then greedy steps, each position's argmax and logits digest), the switch to
//! eager steps ([`Family::eager`]), whether its load attaches the step port's card leg ([`Family::card_leg`]) and,
//! beside them, the refusals only it has ([`Family::refusals`], a family without one returns none).
//!
//! The arms are planned from the file, never from a literal: the all-RAM plan at the room its need fits fixes the
//! arena's slot set, the paged arm's room is the split's floor F (read off the `HostRoomFloor` refusal at a
//! [`PROBE_ROOM`] room under a given arena) plus the arena, and the arena is the slot sets [`paged_slots`] sizes from
//! the file: [`SLOTS`], or fewer where the need leaves less above F. The two arms run one after the other on the same
//! ids. The paged arm runs once more, in a child process of this binary ([`ARM_CHILD`]) given the other arena memory
//! (`BLOOMERY_NVTIER_MEM`, read where the tier is built): `anon` when the arena the process's levers built gave the
//! card leg its slots, so the host leg has an arm of its own, and `pinned` when it did not and the family attaches
//! the leg, so the card leg has one. The clauses are judged on what each snapshotted:
//!
//! - `plan`: the all-RAM plan pages nothing; the paged plan pages routed experts through an arena of exactly the
//!   given bytes and slots, which the tier carved too (`NvTierStats::slots_per_layer`). The room below the floor is
//!   the probe's refusal, so a plan that never splits fails here, as does a file whose need leaves no arena of the
//!   arena's floor under it.
//! - `attach`: the load built a tier, it covers every paged layer, the carve is at least the arena's floor and under
//!   the ids a paged layer serves (a layer that serves no more than it holds slots for never evicts), and a run's
//!   misses, fills, evictions and prompt union claims are above 0 while no arena id was read through the file mapping.
//! - `bits`: on the paged arm whose host leg computes the arena's ids (its tier's card leg is not `on`), every
//!   position's argmax and logits digest equal the all-RAM arm's.
//! - `card`: on a family that attaches the card leg, the paged arm whose card leg is `on` claimed on the step port
//!   only ids the card held, the card read some and the tier filled some; its steps run again eagerly from a cleared session give
//!   every position's logits bit for bit. On a family that attaches none, the tier's card leg is `unattached`.
//! - `refuse`: by name, a room under the floor and an arena one slot set under the arena's floor (both at plan time),
//!   and the family's own refusals.
//! - `residency`: the unset residency rule on the paged plan resolves as the plan's own numbers say (the card
//!   experts, the churn pool at the counts the rule probes, the host headroom and what the host's room leaves,
//!   [`RoomFacts`]), never with why `paged`: the family states the pair it expects ([`Family::expected_residency`]).

use std::process::Command;
use std::sync::Arc;

use bloomery_gpu::host::arena_leg::LegOff;
use bloomery_gpu::host::nvtier::{NvTier, NvTierStats};
use bloomery_gpu_gates::residency::PlanRoom;
use bloomery_gpu_gates::{GateError, checks_failed, record, tier, verdict};
use bloomery_levers::{NVTIER_MEM, RESIDENCY_SERVING, ResidencyWord, residency_word};
use model::placement::workstation::HostNeed;
use model::placement::{ArenaFloor, Device, PlacementError, Plan, Role, arena_slots};

/// The most slots a paged layer's arena holds in the arm the gate pages ([`paged_slots`]): above the arena's floor,
/// under the ids a layer serves, so evictions happen.
pub const SLOTS: u64 = 64;

/// The room the floor is read at: under every file's floor.
const PROBE_ROOM: u64 = 1 << 30;

/// The argument that makes this binary the paged arm's child ([`arm_child`]): the arm's room and arena follow it.
const ARM_CHILD: &str = "--nvtier-arm-child";

/// The arm child's arena memory where the host leg computes the arena's ids ([`arm_child`]).
const HOST_LEG_MEM: (&str, &str) = (NVTIER_MEM, "anon");

/// The arm child's arena memory the card leg reads ([`arm_child`]).
const CARD_LEG_MEM: (&str, &str) = (NVTIER_MEM, "pinned");

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
    /// What the unset residency rule reads of the plan.
    pub room: RoomFacts,
}

/// What the unset residency rule reads of a plan, taken through the rule's own reader ([`PlanRoom`]) so the gate
/// prints the numbers the rule decided by and judges them without running the rule.
#[derive(Clone, Copy, Debug)]
pub struct RoomFacts {
    /// The fewest card experts a layer that holds any; `None` where the plan holds none.
    pub fewest: Option<u64>,
    /// The serving word's free slots a layer.
    pub spares: usize,
    /// The churn pool's bytes at the first count the rule probes (the word's P, 0) and at the last
    /// (`fewest - spares - 1`, at least 0).
    pub pool_first: u64,
    pub pool_last: u64,
    /// The plan's host headroom, in bytes.
    pub headroom: i128,
    /// What the host's room leaves past the plan's host need, in bytes.
    pub mem_left: i128,
}

impl RoomFacts {
    /// The facts of `plan` for the churn pool of card `card`.
    ///
    /// # Errors
    /// A serving word whose P is not 0 (the table in [`Unset::serving`] covers P 0), a pool the placement
    /// refuses, or an unreadable `MemAvailable`.
    pub fn of(plan: &Plan<'_>, card: usize) -> Result<RoomFacts, GateError> {
        let Some(ResidencyWord::Mid {
            pinned: 0, spares, ..
        }) = residency_word(RESIDENCY_SERVING)
        else {
            return Err(format!(
                "the serving word {RESIDENCY_SERVING:?} is not a mid word at P 0: the table of \
                 `Unset::serving` covers P 0"
            )
            .into());
        };
        let room = PlanRoom::new(plan, card, 0);
        let fewest = room.card_experts().filter(|&n| n > 0).min();
        let last = fewest.map_or(0, |f| {
            usize::try_from(f).map_or(usize::MAX, |f| f.saturating_sub(spares + 1))
        });
        Ok(RoomFacts {
            fewest,
            spares,
            pool_first: room.pool_bytes(0)?,
            pool_last: room.pool_bytes(last)?,
            headroom: room.headroom(),
            mem_left: room.mem_left()?,
        })
    }
}

impl std::fmt::Display for RoomFacts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "fewest card experts a layer {:?}, churn pool {} B at P0 and {} B at the last count, host headroom {} B, \
             the host's room leaves {} B",
            self.fewest, self.pool_first, self.pool_last, self.headroom, self.mem_left
        )
    }
}

/// What the unset residency rule resolves on a plan, as the plan's numbers fix it ([`RoomFacts`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unset {
    /// The serving word as it is, why `place`.
    Kept,
    /// The serving word at a higher P the pool fits, why `pool_shrunk`: `mid-p<q>-s<spares>`, `1 <= q <= ceiling`;
    /// a load that hosts its pool.
    PoolShrunk { ceiling: usize, spares: usize },
    /// `off`, with this why's name.
    Off(&'static str),
}

impl Unset {
    /// The serving word's resolution on `room`, the table of the rule's clauses in `room_for`
    /// (`crates/levers/src/lib.rs:1279`), which both families' rules reach (`residency_at_plan`, `:1379`;
    /// `glm_residency_at_plan`, `:2134`, whose why names are the families' to map: `Serving` is `place`, `Moved`
    /// is `shrunk`):
    /// - no layer holds a card expert (`:1287`): `off`, `no_card_experts`;
    /// - the fewest layer cannot hold P 0 and the spares (`:1294-1306`, `fewest < spares + 1`, and half of it is
    ///   under the minimum P): `off`, `no_room`;
    /// - the pool at P 0 fits the headroom and what the host's room leaves: the word as it is;
    /// - else, on every body (an unset word never asks for its pool unhosted), the pool at the last count fits
    ///   (the pool only shrinks as the count grows): `pool_shrunk` to some `q` in `1..=fewest - spares - 1`;
    /// - else `off`, `pool_unhostable`.
    #[must_use]
    pub fn serving(room: &RoomFacts) -> Unset {
        let Some(fewest) = room.fewest else {
            return Unset::Off("no_card_experts");
        };
        let Some(ceiling) = usize::try_from(fewest)
            .ok()
            .and_then(|f| f.checked_sub(room.spares + 1))
        else {
            return Unset::Off("no_room");
        };
        let fits =
            |pool: u64| room.headroom >= i128::from(pool) && room.mem_left >= i128::from(pool);
        // PIN(2026-10-13): a pool that does not fit raises P to the first count whose pool fits
        // (`pool_shrunk`), or turns the word `off` (`pool_unhostable`) where none does, on every body. An
        // unset word that kept P 0 with the pool unhosted on a body that carries one decoded far slower than
        // the climb, so the unhosted pool is the word's suffix (`-unhosted`), never an unset word's
        // resolution.
        if fits(room.pool_first) {
            Unset::Kept
        } else if ceiling >= 1 && fits(room.pool_last) {
            Unset::PoolShrunk {
                ceiling,
                spares: room.spares,
            }
        } else {
            Unset::Off("pool_unhostable")
        }
    }

    /// Whether the rule's pair is this resolution. A why that names the removed `paged` clause is red whatever
    /// the table says.
    #[must_use]
    pub fn holds(&self, word: &str, why: &str) -> bool {
        why != "paged"
            && match *self {
                Unset::Kept => word == RESIDENCY_SERVING && why == "place",
                Unset::PoolShrunk { ceiling, spares } => {
                    why == "pool_shrunk"
                        && matches!(
                            residency_word(word),
                            Some(ResidencyWord::Mid { pinned, spares: s, .. })
                                if (1..=ceiling).contains(&pinned) && s == spares
                        )
                }
                Unset::Off(name) => word == "off" && why == name,
            }
    }
}

impl std::fmt::Display for Unset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unset::Kept => write!(f, "{RESIDENCY_SERVING:?}, why \"place\""),
            Unset::PoolShrunk { ceiling, spares } => {
                write!(f, "mid-p1..{ceiling}-s{spares}, why \"pool_shrunk\"")
            }
            Unset::Off(name) => write!(f, "\"off\", why {name:?}"),
        }
    }
}

impl PlanFacts {
    /// `plan`'s facts, its unset residency rule's inputs for the churn pool of card `card`.
    ///
    /// # Errors
    /// The arena's slots or floor refused by the placement, or an expert count past `u32`.
    pub fn of(plan: &Plan<'_>, card: usize) -> Result<PlanFacts, GateError> {
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
            floor: ArenaFloor::of(plan)?,
            paged_layers,
            experts: u32::try_from(model.experts)?,
            room: RoomFacts::of(plan, card)?,
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
    /// `None` when the step port's card leg reads the arena, else why not ([`NvTier::card_leg`]).
    pub leg: Option<LegOff>,
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
            leg: t.card_leg(),
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

    /// The session cleared to its first position, its steps eager from here on. The `card` clause calls it only on a
    /// family whose load attaches the card leg ([`Family::card_leg`]).
    ///
    /// # Errors
    /// The clear's failure; a family that attaches no card leg refuses it by name.
    fn eager(&self, _s: &mut Self::Session) -> Result<(), GateError> {
        Err("an eager rerun: this family attaches no card leg, so no arm runs one".into())
    }

    /// Whether the family's load attaches the step port's card leg
    /// (`bloomery_gpu::host::arena_leg`): it then reads a pinned arena's slots on the card.
    fn card_leg(&self) -> bool {
        false
    }

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

    /// The pair the family's unset rule must resolve on the paged plan `plan`, from the plan's numbers alone: the
    /// family names the rule it runs and its clauses (path:line), never runs it.
    ///
    /// PIN(2026-10-12): the unset word on a paged plan was `off`, why `paged`, until the `Paged` clause was
    /// removed; it now follows `room_for`'s next clause (`crates/levers/src/lib.rs:1287-1362`).
    fn expected_residency(&self, plan: &PlanFacts) -> Unset;

    /// The family's own refusals on the paged arm `paged`, whose arena holds `slots` slots a paged layer.
    fn refusals(&self, _paged: Arm, _slots: u64) -> Vec<Refusal<'_>> {
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

/// The paged arm's slots on a plan whose all-RAM need is `need` and whose split's floor is `floor`: [`SLOTS`], or
/// the most whose arena keeps the room `floor + arena` under the need where fewer fit, so the split runs (a plan
/// whose need the room holds is left as it is, `expert_nvme_tier`). At the split the host keeps the room less the
/// terms beside the routed experts, the two layers' transient reads and the arena — one heaviest layer's experts
/// (`base + 3 W + A − base − 2 W − A`), whatever the arena — so the arena's size moves which ids page, never whether.
/// `None` where the arena's floor (`least`) does not fit under the need.
fn paged_slots(need: u64, floor: u64, least: &ArenaFloor) -> Option<u64> {
    let above = need.checked_sub(floor)?.checked_sub(1)?;
    let slots = above.checked_div(least.slot_set)?.min(SLOTS);
    (slots >= least.slots).then_some(slots)
}

/// What one arm left behind.
struct Ran {
    plan: PlanFacts,
    run: Vec<Pos>,
    tier: Option<TierFacts>,
    /// The run again, eager from a cleared session, on an arm whose card leg is `on`.
    eager: Vec<Pos>,
}

/// An arm's card leg: on, off for a reason, or no tier to run it over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leg {
    NoTier,
    On,
    Off(LegOff),
}

impl Leg {
    /// The leg of an arm whose tier `tier` snapshotted.
    fn of(tier: Option<&TierFacts>) -> Leg {
        tier.map_or(Leg::NoTier, |t| t.leg.map_or(Leg::On, Leg::Off))
    }

    /// The word the gate prints, as the `nvtier` record's `card_leg` spells it.
    fn word(self) -> &'static str {
        match self {
            Leg::NoTier => "none",
            Leg::On => "on",
            Leg::Off(off) => off.name(),
        }
    }

    /// The leg a [`Leg::word`] names.
    fn parse(word: &str) -> Result<Leg, GateError> {
        match word {
            "none" => Ok(Leg::NoTier),
            "on" => Ok(Leg::On),
            w => LegOff::ALL
                .into_iter()
                .find(|off| off.name() == w)
                .map(Leg::Off)
                .ok_or_else(|| format!("the arm child's card leg {w:?}").into()),
        }
    }
}

impl Ran {
    /// The arm's card leg.
    fn leg(&self) -> Leg {
        Leg::of(self.tier.as_ref())
    }

    /// What the arm shows of the card leg.
    fn leg_facts(&self) -> LegFacts {
        let delta = |f: fn(&NvTierStats) -> u64| self.tier.as_ref().map_or(0, |t| t.delta(f));
        LegFacts {
            leg: self.leg(),
            run: self.run.clone(),
            eager: self.eager.clone(),
            claims: delta(|s| s.step_fills + s.step_hits),
            card_ids: delta(|s| s.card_ids),
            fills: delta(|s| s.fills),
        }
    }
}

/// The arm `arm` planned, loaded, run once (and, where its card leg is `on`, once more eagerly) and torn down.
fn arm_run<F: Family>(f: &F, name: &str, arm: Arm, plan: PlanFacts) -> Result<Ran, GateError> {
    record::require(
        record::groups::nvtier_family::ARM_RUN,
        "nvtier_family::arm_run",
    );
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
    let leg = Leg::of(facts.as_ref());
    let eager = if leg == Leg::On {
        f.eager(&mut s)?;
        f.run(&mut s)?
    } else {
        Vec::new()
    };
    f.finish(s)?;
    println!(
        "{name}: arm room {} B arena {:?}: {} positions, {} paged layers, tier {}, card leg {}",
        arm.room,
        arm.arena,
        run.len(),
        plan.paged_layers.len(),
        if facts.is_some() { "built" } else { "none" },
        leg.word()
    );
    Ok(Ran {
        plan,
        run,
        tier: facts,
        eager,
    })
}

/// What a paged arm shows of the step port's card leg: its state, the run and its eager rerun, and the run's tier
/// counts (the step port's claims, the ids the leg held and left to the host, the fills).
#[derive(Clone, Debug, PartialEq, Eq)]
struct LegFacts {
    leg: Leg,
    run: Vec<Pos>,
    eager: Vec<Pos>,
    claims: u64,
    card_ids: u64,
    fills: u64,
}

impl LegFacts {
    /// The facts as the child's reply record, [`LegFacts::parse`]'s input.
    fn print(&self) {
        record::require(
            record::groups::nvtier_family::ARM,
            "nvtier_family::LegFacts::print",
        );
        let mut r = record::Record::new(&record::NVTIER_ARM)
            .w("leg", self.leg.word())
            .u("claims", self.claims)
            .u("card_ids", self.card_ids)
            .u("fills", self.fills);
        for (key, ps) in [("run", &self.run), ("eager", &self.eager)] {
            r = r.csv(
                key,
                ps.iter().map(|p| format!("{}:{:016x}", p.argmax, p.fnv)),
            );
        }
        r.print();
    }

    /// The facts from a child's stdout.
    fn parse(out: &str) -> Result<LegFacts, GateError> {
        let f = record::Log::of(out, &[&record::NVTIER_ARM]).one(&record::NVTIER_ARM)?;
        let list = |key: &str| -> Result<Vec<Pos>, GateError> {
            f.csv(key)?
                .into_iter()
                .map(|w| {
                    let (argmax, fnv) = w
                        .split_once(':')
                        .ok_or_else(|| format!("the arm child's {key} entry {w:?}"))?;
                    Ok(Pos {
                        argmax: argmax.parse()?,
                        fnv: u64::from_str_radix(fnv, 16)?,
                    })
                })
                .collect()
        };
        Ok(LegFacts {
            leg: Leg::parse(f.word("leg")?)?,
            run: list("run")?,
            eager: list("eager")?,
            claims: f.u64("claims")?,
            card_ids: f.u64("card_ids")?,
            fills: f.u64("fills")?,
        })
    }
}

/// The paged arm `arm` (its room and arena) run by this binary as a child whose arena memory is `mem`
/// ([`HOST_LEG_MEM`] or [`CARD_LEG_MEM`]), after the parent's own loads are gone; its output forwarded, its facts
/// parsed.
fn arm_child(name: &str, arm: Arm, (lever, mem): (&str, &str)) -> Result<LegFacts, GateError> {
    let mut cmd = Command::new(std::env::current_exe()?);
    cmd.arg(ARM_CHILD)
        .arg(arm.room.to_string())
        .arg(arm.arena.map_or_else(|| "-".to_string(), |a| a.to_string()))
        .env(lever, mem);
    bloomery_gpu_gates::die_with_parent(&mut cmd);
    let out = cmd
        .output()
        .map_err(|e| format!("{name}: run the {mem} arm child: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    for line in stdout.lines() {
        println!("[{mem} arm] {line}");
    }
    for line in String::from_utf8_lossy(&out.stderr).lines() {
        eprintln!("[{mem} arm] {line}");
    }
    if !out.status.success() {
        return Err(format!("{name}: the {mem} arm child ended {}", out.status).into());
    }
    LegFacts::parse(&stdout)
}

/// The arm this process runs as the paged arm's child: `Some` of its room and arena after [`ARM_CHILD`], `None` for
/// the gate itself.
fn child_arm() -> Result<Option<Arm>, GateError> {
    let mut args = std::env::args().skip(1);
    if args.find(|a| a == ARM_CHILD).is_none() {
        return Ok(None);
    }
    let (Some(room), Some(arena)) = (args.next(), args.next()) else {
        return Err(format!("{ARM_CHILD} takes the arm's room and arena").into());
    };
    let arena = match arena.as_str() {
        "-" => None,
        a => Some(a.parse()?),
    };
    Ok(Some(Arm {
        room: room.parse()?,
        arena,
    }))
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
    (a, slots): (u64, u64),
    tier: Option<&TierFacts>,
) -> bool {
    let slot_set = ram.floor.slot_set;
    let carve = tier.is_none_or(|t| t.slots == paged.arena_slots);
    let ok = ram.paged == 0
        && ram.arena == 0
        && paged.paged > 0
        && paged.arena == a
        && paged.arena_slots == slots
        && slots >= ram.floor.slots
        && carve;
    clause(
        name,
        &format!(
            "all-RAM pages {} B; the paged plan pages {} B through an arena of {} B = {} slots of {slot_set} B \
             (given {a} B, {slots} wanted, floor {}), the tier carved {:?}",
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

/// `bits`: see the module doc. `leg` is the card leg state of the paged arm `paged` ran on.
///
/// PIN(2026-10-14): the clause holds the paged arm whose host leg computes the arena's ids; where the card leg reads
/// a pinned arena's slots it quantizes the activation as q8_1 and sums those ids on the card in its own order, and
/// the `card` clause holds that arm.
fn bits_clause(name: &str, ram: &[Pos], (leg, paged): (Leg, &[Pos])) -> bool {
    let first = ram.iter().zip(paged).position(|(a, b)| a != b);
    let ok = leg != Leg::On && ram.len() == paged.len() && !ram.is_empty() && first.is_none();
    clause(
        name,
        &format!(
            "the host leg's arm (card leg {}): {} positions against {}, first difference {first:?}, last digest \
             {:016x}",
            leg.word(),
            ram.len(),
            paged.len(),
            paged.last().map_or(0, |p| p.fnv)
        ),
        ok,
    )
}

/// `card`: see the module doc. `attaches` is the family's [`Family::card_leg`]; `leg` the paged arm's own card leg
/// state, `card` the facts of the arm whose card leg is `on` (`None` where no arm's tier gave it one); `host` the host
/// leg's arm, whose argmax per position is printed beside the card arm's and not judged.
fn card_clause(
    name: &str,
    attaches: bool,
    (leg, card): (Leg, Option<&LegFacts>),
    host: &[Pos],
) -> bool {
    if !attaches {
        return clause(
            name,
            &format!(
                "the family attaches no card leg; the paged arm's tier reads {}",
                leg.word()
            ),
            leg == Leg::Off(LegOff::Unattached),
        );
    }
    let Some(c) = card else {
        return clause(
            name,
            &format!(
                "no paged arm ran the card leg; the paged arm's tier reads {}",
                leg.word()
            ),
            false,
        );
    };
    let identity = c.claims == c.card_ids;
    let first = c.run.iter().zip(&c.eager).position(|(a, b)| a != b);
    let replay = !c.run.is_empty() && c.run.len() == c.eager.len() && first.is_none();
    let same = c
        .run
        .iter()
        .zip(host)
        .filter(|(a, b)| a.argmax == b.argmax)
        .count();
    clause(
        name,
        &format!(
            "card leg {}; the steps' claims {} = the leg's ids {} ({identity}); fills {}; graph \
             against eager: {} positions against {}, first difference {first:?}; argmax as the host leg's arm at \
             {same} of {} positions (printed)",
            c.leg.word(),
            c.claims,
            c.card_ids,
            c.fills,
            c.run.len(),
            c.eager.len(),
            c.run.len()
        ),
        c.leg == Leg::On && identity && c.card_ids > 0 && c.fills > 0 && replay,
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

/// The gate: `f`'s file planned, its arms run, the six clauses judged, none stopping the others after the
/// arms ran.
///
/// # Errors
/// A clause that is red ([`checks_failed`]), or a failure the clauses cannot name: a probe, a load, a run.
pub fn gate<F: Family>(name: &str, f: &F) -> Result<(), GateError> {
    if let Some(arm) = child_arm()? {
        let plan = f
            .plan(arm)?
            .map_err(|e| format!("{name}: the arm child's plan: {e}"))?;
        arm_run(f, name, arm, plan)?.leg_facts().print();
        return Ok(());
    }
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
    let ram_arm = Arm {
        room: base.need,
        arena: None,
    };
    let ram_plan =
        probe(ram_arm)?.map_err(|e| format!("{name}: the all-RAM plan at its need: {e}"))?;
    // F: the floor a room under the split's floor is refused by name with, under a given arena: the arena's floor,
    // since a given arena takes no part in F (`ArenaDial::room_floor`).
    let low = Arm {
        room: PROBE_ROOM,
        arena: Some(base.floor.bytes),
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
    // A file whose need leaves no arena of the floor's slots above F pages nothing at any arena: a red clause where
    // the paged clauses run, and [`SLOTS`] where they are left to the real tier (nothing pages, as on any arena).
    let slots = match paged_slots(base.need, floor, &base.floor) {
        Some(slots) => slots,
        None if !host => SLOTS,
        None => {
            clause(
                "plan",
                &format!(
                    "the all-RAM need {} B leaves no arena of the floor's {} slots of {} B above the floor F \
                     {floor} B: no split runs on this file",
                    base.need, base.floor.slots, base.floor.slot_set
                ),
                false,
            );
            println!("{name}: {}", tier::tally_line());
            return Err(checks_failed());
        }
    };
    let a = slots
        .checked_mul(base.floor.slot_set)
        .ok_or("the arena's bytes pass u64")?;
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
        "{name}: floor F {floor} B, arena A {a} B ({slots} slots of {} B, at most {SLOTS}), paged room {} B, \
         all-RAM room {} B",
        base.floor.slot_set, paged_arm.room, ram_arm.room
    );
    // One load at a time: the all-RAM arm's session is gone before the paged arm loads.
    let ram = arm_run(f, name, ram_arm, ram_plan)?;
    let paged = arm_run(f, name, paged_arm, paged_plan)?;
    // The paged arm again in a child, on the other arena memory: the host leg's arm where the card leg read this
    // one's slots, the card leg's where the family attaches it and this arena gave it none.
    let here = (paged.leg(), paged.run.clone());
    let (host_arm, card) = if paged.leg() == Leg::On {
        let child = arm_child(name, paged_arm, HOST_LEG_MEM)?;
        ((child.leg, child.run), Some(paged.leg_facts()))
    } else if paged.tier.is_some() && f.card_leg() {
        (here, Some(arm_child(name, paged_arm, CARD_LEG_MEM)?))
    } else {
        (here, None)
    };
    let mut pass = true;
    if plan_runs {
        pass &= plan_clause(
            "plan",
            &ram.plan,
            &paged.plan,
            (a, slots),
            paged.tier.as_ref(),
        );
    }
    if tier::host_clause(
        "attach: the tier is built, covers every paged layer and counts a run's traffic",
        host,
    )? {
        pass &= attach_clause("attach", &paged.plan, paged.tier.as_ref());
    }
    tier::sc("bits: every position of the host leg's paged arm equals the all-RAM arm's")?;
    pass &= bits_clause("bits", &ram.run, (host_arm.0, &host_arm.1));
    if tier::host_clause(
        "card: the card leg's arm claims only the leg's ids, and replays as it steps eagerly",
        host,
    )? {
        pass &= card_clause(
            "card",
            f.card_leg(),
            (paged.leg(), card.as_ref()),
            &host_arm.1,
        );
    }
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
        for r in f.refusals(paged_arm, slots) {
            pass &= refused(r.what, r.holds, (r.attempt)());
        }
    }
    if tier::host_clause(
        "residency: the unset rule on the paged plan resolves as the plan's numbers say, never why paged",
        host,
    )? {
        let (word, why) = f.unset_residency(paged_arm)?;
        let expected = f.expected_residency(&paged.plan);
        pass &= clause(
            "residency",
            &format!(
                "the unset word on the paged plan is {word:?}, why {why:?}; the plan's numbers give {expected} ({})",
                paged.plan.room
            ),
            expected.holds(&word, why),
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

#[cfg(test)]
mod tests {
    use super::{Leg, LegFacts, Pos};

    #[test]
    fn the_arm_child_reply_is_one_whole_record() {
        let line = "arm-child leg=on claims=2 card_ids=2 fills=2 \
                    run=[7:0123456789abcdef,8:fedcba9876543210] eager=[]";
        let facts = LegFacts::parse(&format!("load progress\n{line}\n")).unwrap();
        assert_eq!(
            facts,
            LegFacts {
                leg: Leg::On,
                run: vec![
                    Pos {
                        argmax: 7,
                        fnv: 0x0123456789abcdef
                    },
                    Pos {
                        argmax: 8,
                        fnv: 0xfedcba9876543210
                    },
                ],
                eager: vec![],
                claims: 2,
                card_ids: 2,
                fills: 2,
            }
        );
        for bad in [
            String::new(),
            format!("{line}\n{line}"),
            format!("{line} extra=1"),
            line.replace(" fills=2", ""),
            line.replace("claims=2", "claims=bad"),
            line.replace("leg=on", "leg=unknown"),
            line.replace("7:0123456789abcdef", "7"),
            line.replace("8:fedcba9876543210", "bad:fedcba9876543210"),
            line.replace("8:fedcba9876543210", "8:not-hex"),
        ] {
            assert!(LegFacts::parse(&bad).is_err(), "accepted {bad:?}");
        }
    }
}
