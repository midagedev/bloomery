//! A plan that reads its K/V at a floor charges the host K/V there and keeps on the host the
//! experts the static split pages at the cap that the room between the two holds: the need is
//! the cap's, the cap stays a refusal where the room cannot reach it, and the yields, done in
//! order, end at the static split ([`HostGrow`]).

use std::sync::{Arc, LazyLock};

use super::*;
use bloomery_levers::{KvHost, Room, RoomShort};
use runtime::stores::{Access, PosBuf, PosLayout};

/// A position's ring-shadow bytes a layer: host rooms that grow with the positions.
const SHADOW: u64 = 4;
/// The cap and the floor the plans below are made at.
const CAP: u64 = 80;
const FLOOR: u64 = 40;
/// Host pages past the floor a caller counted (`kv_host_planes`), none at the floor.
const PAGES: u64 = 1_000;
/// The host terms beside the routed experts.
const BESIDE: u64 = 1_000;

/// `layers` layers of one host-able room each.
fn rooms_of(layers: usize) -> PosLayout {
    PosLayout::new(
        (0..layers)
            .map(|_| {
                vec![PosBuf {
                    what: "selected",
                    row: 8,
                    per: 1,
                    runs: 1,
                    access: Access::Select { rows: 4 },
                }]
            })
            .collect(),
    )
}

static ROOMS: LazyLock<PosLayout> = LazyLock::new(|| rooms_of(4));
/// The rooms of the [`WIDE`]-layer model the pool tests plan.
static WIDE_ROOMS: LazyLock<PosLayout> = LazyLock::new(|| rooms_of(WIDE));

/// The layers of the model the pool tests plan, past [`layered`]'s four: the window the split
/// reserves is one layer's host experts, the churn pool takes every layer's.
const WIDE: usize = 16;

/// Layers of 100 B of state and 8 B a position, with a ring shadow of [`SHADOW`] B a position
/// on the host, over the rooms it is given.
struct Shadowed(&'static LazyLock<PosLayout>);

impl KvBytes for Shadowed {
    fn layer_bytes(&self, _layer: usize, ctx_max: u64) -> u64 {
        100 + 8 * ctx_max
    }

    fn shadow_bytes(&self, _layer: usize, ctx_max: u64) -> u64 {
        SHADOW * ctx_max
    }

    fn rooms(&self) -> Option<&PosLayout> {
        Some(&**self.0)
    }

    fn host_yields(&self) -> bool {
        true
    }
}

/// The four layers on a card that holds some of the experts, beside a host of `room`.
fn machine(room: u64) -> Machine {
    machine_of(4, 3, room)
}

/// `layers` layers on a card that holds `held` experts of each, beside a host of `room`.
fn machine_of(layers: usize, held: u64, room: u64) -> Machine {
    Machine {
        cards: vec![bytes_card(
            "stage",
            HEAD + layers as u64 * (100 + 8 * CAP + held * EXPERT),
            0..layers,
        )],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: Host {
                usable_bytes: room,
                reserves: vec![("beside".into(), BESIDE)],
            },
            unified: None,
        }],
    }
}

/// A plan's steps in order (the routed placement, the pages, the room, the split with no
/// arena) under `lever` at a floor of [`FLOOR`]: with its growth charged at the floor
/// (`arm`), or the static plan, every term at the cap.
fn grown<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    lever: KvHost,
    arm: bool,
) -> Result<Plan<'a>, PlacementError> {
    grown_under(model, machine, lever, arm, None)
}

/// [`grown`] with each routed layer's experts taken in `rank`'s order when there is one.
fn grown_under<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    lever: KvHost,
    arm: bool,
    rank: Option<Arc<ExpertRank>>,
) -> Result<Plan<'a>, PlacementError> {
    let levers = PlanLevers {
        kv_floor: Some(FLOOR),
        kv_host: lever,
        expert_rank: rank,
        ..PlanLevers::default()
    };
    let mut plan = plan(model, machine, CAP, &Shadowed(&ROOMS), &levers)?;
    if !arm {
        plan.host.grow = HostGrow::default();
    }
    // The tests of the yields' order and the books charge the growth alone.
    plan.host.grow.pool_room = false;
    if lever != KvHost::Off {
        plan.kv_host_planes(PAGES, 0)?;
    }
    let room = plan.plan_at(&(machine.host().usable_bytes, workstation::HostRead::Given));
    expert_nvme_tier(&mut plan, room, Some(0))?;
    Ok(plan)
}

/// A room the static plan pages in: the need of the plan with every host expert resident and
/// no pages, which the cap's pages and shadows pass.
fn tight_room(model: &ModelTensors) -> u64 {
    let ample = machine(1 << 40);
    let whole = grown(model, &ample, KvHost::Off, false).expect("a plan");
    u64::try_from(workstation::HostNeed::of(&whole, 0).exact(true)).expect("u64")
}

/// Each routed layer's host and NVMe ids, as the rows hold them.
fn split_of(plan: &Plan<'_>) -> Vec<(usize, Vec<u32>, Vec<u32>)> {
    let mut out = Vec::new();
    for r in &plan.rows {
        let t = &plan.model.tensors[r.tensor];
        let (Some(l), Role::RoutedExperts) = (t.layer, t.role) else {
            continue;
        };
        let ids = |d: Device| -> Vec<u32> {
            r.segments
                .iter()
                .filter(|s| s.device == d)
                .flat_map(|s| {
                    s.experts
                        .as_ref()
                        .map(|e| e.ids().to_vec())
                        .unwrap_or_default()
                })
                .collect()
        };
        out.push((l, ids(Device::Host), ids(Device::Nvme)));
    }
    out
}

/// The plan charges the ring shadows and the pages at the floor, the rest to the cap is the
/// growth, and the paged experts the growth's room holds stay on the host: the need, the
/// headroom and the host-served bytes are the static plan's, the paged bytes less by the
/// yields'. FAIL-first: the planner reading the shadows at the cap into the floor (the old
/// charge) leaves `floor_bytes` at the cap's 1,280 B; `HostNeed::shadows` without the yields
/// taken out passes the static plan's need by them.
#[test]
fn a_plan_at_the_floor_charges_the_floor() {
    let model = layered(4);
    let room = tight_room(&model);
    let tight = machine(room);
    let static_plan = grown(&model, &tight, KvHost::Spill, false).expect("the static plan");
    let plan = grown(&model, &tight, KvHost::Spill, true).expect("the plan at the floor");
    let g = &plan.host.grow;
    assert_eq!(g.floor, FLOOR);
    assert_eq!(
        g.floor_bytes,
        4 * SHADOW * FLOOR,
        "the shadows at the floor, no page"
    );
    assert_eq!(
        g.grow_bytes,
        4 * SHADOW * (CAP - FLOOR) + PAGES,
        "the shadows' growth and the pages past the floor"
    );
    assert_eq!(
        plan.host.shadow_bytes,
        4 * SHADOW * CAP,
        "the address range at the cap"
    );
    assert_eq!(g.hold, GrowHold::Yields);
    assert!(g.yield_bytes > 0 && g.yield_bytes <= g.grow_bytes, "{g}");
    assert_eq!(g.yield_bytes, g.yields.len() as u64 * EXPERT);
    assert_eq!(g.charged(), g.floor_bytes + g.grow_bytes - g.yield_bytes);
    let need = |p: &Plan<'_>| workstation::HostNeed::of(p, 0);
    assert_eq!(
        need(&plan).bytes(),
        need(&static_plan).bytes(),
        "the cap's need"
    );
    assert_eq!(plan.host.headroom_bytes, static_plan.host.headroom_bytes);
    assert_eq!(
        plan.host.nvme_expert_bytes + g.yield_bytes,
        static_plan.host.nvme_expert_bytes
    );
    assert_eq!(
        plan.host.expert_bytes,
        static_plan.host.expert_bytes + g.yield_bytes
    );
    assert!(plan.violations().is_empty(), "{:?}", plan.violations());
    assert_eq!(static_plan.host.grow, HostGrow::default());
}

/// The least room the static plan takes at the cap is the least the plan at the floor takes:
/// a byte under it is refused by name in both, the floor's charge does not pass it, and at it
/// both plan. FAIL-first: the host need taking the whole growth out of the K/V (`grow_bytes`
/// for `yield_bytes` in `HostNeed::of_totals`) plans the floor a byte under.
#[test]
fn a_cap_the_room_cannot_reach_is_refused_by_name() {
    let model = layered(4);
    // Whether the plan at `room` charges its K/V at the floor, or its refusal.
    let plans = |room: u64, arm: bool| {
        grown(&model, &machine(room), KvHost::Spill, arm).map(|p| p.host.grow.armed())
    };
    // The least room the static plan takes, between none and one that holds every expert.
    let (mut lo, mut hi) = (0u64, tight_room(&model));
    assert!(plans(hi, false).is_ok() && plans(lo, false).is_err());
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        match plans(mid, false) {
            Ok(_) => hi = mid,
            Err(_) => lo = mid,
        }
    }
    for arm in [false, true] {
        let err = plans(hi - 1, arm).expect_err("a byte under the cap's room");
        assert!(
            matches!(
                err,
                PlacementError::KvHostOverHost { .. } | PlacementError::HostRoomFloor { .. }
            ),
            "arm {arm}: {err:?}"
        );
        assert_eq!(plans(hi, arm).expect("the cap's room plans"), arm);
    }
}

/// Every yield done, in order, moves each id from its layer's host segment to its NVMe
/// segment and ends at the static split, id for id; each layer yields its id tail first.
/// FAIL-first: restores that take a layer's paged ids from the back (`yields_of`) name ids the
/// split keeps on the NVMe tier, so a yield's id is not among the host ids.
#[test]
fn the_yields_to_the_cap_end_at_the_static_split() {
    let model = layered(4);
    let tight = machine(tight_room(&model));
    let static_plan = grown(&model, &tight, KvHost::Spill, false).expect("the static plan");
    let plan = grown(&model, &tight, KvHost::Spill, true).expect("the plan at the floor");
    let mut split = split_of(&plan);
    let mut last: Vec<Option<u32>> = vec![None; 4];
    for y in &plan.host.grow.yields {
        assert!(
            last[y.layer].is_none_or(|prev| y.id < prev),
            "layer {} yields {} after {:?}: the id tail first",
            y.layer,
            y.id,
            last[y.layer]
        );
        last[y.layer] = Some(y.id);
        let (_, host, nvme) = split
            .iter_mut()
            .find(|(l, _, _)| *l == y.layer)
            .expect("a routed layer");
        let at = host.iter().position(|&id| id == y.id).expect("a host id");
        assert_eq!(at, host.len() - 1, "the layer's last host id yields");
        host.remove(at);
        nvme.push(y.id);
        nvme.sort_unstable();
    }
    assert_eq!(split, split_of(&static_plan));
}

/// Under a router rank the load keeps past the static split each layer's hottest paged ids,
/// every layer yields its coldest kept id first, and the yields done end at the static split
/// under the same rank, id for id. The rank orders every layer hottest first as 0, 1, 2, 4,
/// 7, 6, 5, 3, so its cold end runs against the id order. FAIL-first: the rank's paged ids
/// handed to the restores ascending (`cold_split` sorting them) keeps id 3, the coldest, past
/// the static split, colder than the 5 and 6 the split pages beside it.
#[test]
fn under_a_rank_the_coldest_yield_first() {
    let model = layered(4);
    let row = [0u32, 1, 2, 4, 7, 6, 5, 3];
    let text = (0..4).fold(
        "# router-rank\tn_expert=8\tn_used=2\tsets=prose\tpositions=100\n".to_owned(),
        |t, l| {
            let ids: Vec<String> = row.iter().map(u32::to_string).collect();
            format!("{t}{l}\t{}\n", ids.join(","))
        },
    );
    let rank = Arc::new(ExpertRank::parse("rank.tsv", &text).expect("a rank"));
    let ample = machine(1 << 40);
    let whole =
        grown_under(&model, &ample, KvHost::Off, false, Some(Arc::clone(&rank))).expect("a plan");
    let room = u64::try_from(workstation::HostNeed::of(&whole, 0).exact(true)).expect("u64");
    let tight = machine(room);
    let static_plan = grown_under(
        &model,
        &tight,
        KvHost::Spill,
        false,
        Some(Arc::clone(&rank)),
    )
    .expect("the static plan");
    let plan = grown_under(&model, &tight, KvHost::Spill, true, Some(rank))
        .expect("the plan at the floor");
    let g = &plan.host.grow;
    assert_eq!(g.hold, GrowHold::Yields, "{g}");
    assert!(!g.yields.is_empty(), "{g}");
    let heat = |id: u32| row.iter().position(|&r| r == id).expect("a ranked id");
    let static_split = split_of(&static_plan);
    let mut last: Vec<Option<u32>> = vec![None; 4];
    let mut split = split_of(&plan);
    for y in &g.yields {
        assert!(
            last[y.layer].is_none_or(|prev| heat(y.id) < heat(prev)),
            "layer {} yields {} after {:?}: the coldest first",
            y.layer,
            y.id,
            last[y.layer]
        );
        last[y.layer] = Some(y.id);
        let (_, _, paged) = &static_split[y.layer];
        for &other in paged {
            let kept = g.yields.iter().any(|z| z.layer == y.layer && z.id == other);
            assert!(
                kept || heat(y.id) < heat(other),
                "layer {} keeps {} past the split, colder than the {other} it pages",
                y.layer,
                y.id
            );
        }
        let (_, host, nvme) = split
            .iter_mut()
            .find(|(l, _, _)| *l == y.layer)
            .expect("a routed layer");
        let at = host.iter().position(|&id| id == y.id).expect("a host id");
        host.remove(at);
        nvme.push(y.id);
        nvme.sort_unstable();
    }
    assert_eq!(split, static_split);
}

/// A growth step past the charge yields the fewest of the plan's next yields that reach the
/// shortfall plus a page, first to last; the step to the cap with every yield done asks for
/// none, and a host past it is refused by name. Each fit below asks one page's 64 B on top of
/// what the host holds. FAIL-first: a `due` that reads the floor's bytes alone as the charge
/// asks for yields at the charge; one that leaves the page out asks one expert for a page of
/// an expert's bytes.
#[test]
fn a_growth_step_yields_the_shortfall_plus_a_page() {
    let model = layered(4);
    let tight = machine(tight_room(&model));
    let plan = grown(&model, &tight, KvHost::Spill, true).expect("the plan at the floor");
    let g = &plan.host.grow;
    assert!(g.yields.len() >= 2, "{g}");
    let start = YieldCursor::default();
    let charge = g.floor_bytes + g.grow_bytes - g.yield_bytes;
    // A fit that leaves the host's K/V at `backed`: its last 64 B are the fit's asks.
    let fit = |backed: u64| (backed - 64, 64);
    assert_eq!(
        g.due(fit(g.floor_bytes), start, 64),
        Ok(None),
        "the floor asks none"
    );
    assert_eq!(g.due(fit(charge), start, 64), Ok(None), "nor the charge");
    let ask = |backed: u64, at: YieldCursor, page: u64| {
        let a = g
            .due(fit(backed), at, page)
            .expect("room to yield")
            .expect("a yield");
        (a.from, a.to, a.bytes, a.shortfall)
    };
    assert_eq!(ask(charge + 1, start, 64), (0, 1, EXPERT, 1));
    assert_eq!(
        ask(charge + 1, start, EXPERT),
        (0, 2, 2 * EXPERT, 1),
        "a page past one"
    );
    let one = YieldCursor {
        done: 1,
        bytes: EXPERT,
    };
    assert_eq!(
        g.due(fit(charge + EXPERT), one, 64),
        Ok(None),
        "a done yield holds its bytes"
    );
    assert_eq!(ask(charge + EXPERT + 1, one, 64), (1, 2, EXPERT, 1));
    let all = YieldCursor {
        done: g.yields.len(),
        bytes: g.yield_bytes,
    };
    assert_eq!(
        g.due(fit(g.cap_bytes()), all, 64),
        Ok(None),
        "the cap with every yield"
    );
    let past = g
        .due((g.cap_bytes() + 1, 64), all, 64)
        .expect_err("a host past the cap");
    assert_eq!((past.held - past.allowed, past.left), (1, 0));
}

/// The experts a layer the card is sized to hold in the pool tests (it holds 3 or 4 of each):
/// the pool's pinned count runs 0 to 1, the ceiling `room_for` leaves the fewest layer less
/// the spare and the one that moves.
const POOL_HELD: u64 = 3;

/// The K/V growth the pool tests plan: 48 experts' bytes, host pages and the shadows' growth.
const POOL_GROWTH: u64 = 48 * EXPERT;

/// A [`WIDE`]-layer plan whose growth to the cap is [`POOL_GROWTH`], made as [`grown_under`]
/// makes one with its growth charged at the floor, with the churn pool's room budgeted when
/// `pool` is set.
fn pooled<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    pool: bool,
) -> Result<Plan<'a>, PlacementError> {
    pooled_by(model, machine, pool, |plan, room| {
        expert_nvme_tier(plan, room, Some(0))
    })
}

/// [`pooled`] with the split `split` (the plan and the room it is made at) in place of the one
/// with no arena.
fn pooled_by<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    pool: bool,
    split: impl FnOnce(&mut Plan<'a>, u64) -> Result<(), PlacementError>,
) -> Result<Plan<'a>, PlacementError> {
    let levers = PlanLevers {
        kv_floor: Some(FLOOR),
        kv_host: KvHost::Spill,
        ..PlanLevers::default()
    };
    let mut plan = plan(model, machine, CAP, &Shadowed(&WIDE_ROOMS), &levers)?;
    let shadows = plan.host.grow.grow_bytes;
    plan.kv_host_planes(POOL_GROWTH - shadows, 0)?;
    plan.host.grow.pool_room = pool;
    let room = plan.plan_at(&(machine.host().usable_bytes, workstation::HostRead::Given));
    split(&mut plan, room)?;
    Ok(plan)
}

/// The host need of the pool model's plan at the floor, every host expert resident and the
/// K/V at its floor: the room that leaves `slack` free is this plus it.
fn floor_need(model: &ModelTensors, held: u64) -> u64 {
    let ample = machine_of(WIDE, held, 1 << 40);
    let base = pooled(model, &ample, true).expect("the plan at an ample room");
    assert_eq!(
        base.host.grow.hold,
        GrowHold::Room,
        "an ample room splits nothing"
    );
    assert_eq!(base.host.grow.grow_bytes, POOL_GROWTH);
    let cap = u64::try_from(workstation::HostNeed::of(&base, 0).exact(true)).expect("u64");
    cap - POOL_GROWTH
}

/// What a plan's pool of `pinned` seed experts a layer takes on the card.
fn pool_at(plan: &Plan<'_>, pinned: usize) -> u64 {
    churn::ChurnPool::of(plan, 0, pinned)
        .expect("the pool")
        .bytes
}

/// The unset residency word's pick over `plan` made at `room`: what the climb of `room_for`
/// makes of the plan's headroom and of what the room leaves past its need.
fn pick_over(plan: &Plan<'_>, room: u64) -> Room {
    bloomery_levers::room_for::<std::convert::Infallible>(
        false,
        0,
        1,
        plan.n_l.iter().copied(),
        |pinned| Ok(pool_at(plan, pinned)),
        churn::ChurnPool::room(plan, 0),
        workstation::HostNeed::of(plan, 0).room_left(&(room, workstation::HostRead::Given)),
    )
    .expect("an infallible pool")
}

/// The window the split reserves on the pool model's plan: two of its heaviest layer's host
/// experts.
fn window_of(plan: &Plan<'_>) -> u64 {
    let heavy = plan
        .n_l
        .iter()
        .map(|&n| (8 - n) * EXPERT)
        .max()
        .expect("layers");
    2 * heavy
}

/// A plan whose room the growth passes by more than the window holds still budgets the churn
/// pool the floor's free room can take: the split rounds the headroom down to a sliver and the
/// unset word goes `off` (`pool_unhostable`) for want of room, though the host holds every host
/// expert and the K/V's floor with room to spare, and the window the split reserves is idle
/// until the first yield. That room is the pool's: the yields the plan lists are the growth's,
/// the window's and the pool's. Here the room is 44 experts, the pool of one pinned expert a
/// layer is 40 and of none 56: the word's pool at its first pinned count is refused as before,
/// and the climb ends at 1.
///
/// FAIL-first: a split that budgets no pool leaves the headroom at 0 and the pick ends
/// `PoolUnhostable { needs: 23040, headroom: 0, mem_left: 0 }`; a `HostNeed` that does not take
/// the yields past the K/V's out of the reserves leaves it at 22,144 B and the pick ends the
/// same way.
#[test]
fn a_churn_pool_the_floors_free_room_holds_is_budgeted_before_the_growth() {
    let model = layered(WIDE);
    let slack = 44 * EXPERT;
    let room = floor_need(&model, POOL_HELD) + slack;
    let tight = machine_of(WIDE, POOL_HELD, room);
    let plan = pooled(&model, &tight, true).expect("the plan at the floor");
    match pick_over(&plan, room) {
        Room::MovedForPool { from: 0, to: 1 } => {}
        other => panic!("the pool of the first count that fits: {other:?}"),
    }
    let g = &plan.host.grow;
    assert_eq!(g.hold, GrowHold::Yields, "{g}");
    assert_eq!(plan.nvme_bytes, 0, "every expert is on the host at load");
    assert!(
        (pool_at(&plan, 1)..pool_at(&plan, 0)).contains(&slack),
        "the free room holds the pool of one pinned expert a layer and not of none"
    );
    // The pool's room is the floor's free room, the yields the growth, the window and it.
    assert_eq!(g.pool_bytes, slack);
    let window = window_of(&plan);
    let owed = POOL_GROWTH + window;
    assert!(
        (owed..owed + EXPERT).contains(&g.yield_bytes),
        "{} B of yields for the growth's {POOL_GROWTH} B and the window's {window} B",
        g.yield_bytes
    );
    let headroom = u64::try_from(plan.host.headroom_bytes).expect("a headroom");
    assert!(
        (slack..slack + EXPERT).contains(&headroom),
        "the headroom is the room and the split's rounding: {headroom} B"
    );
    assert!(
        g.to_string()
            .contains("kept free at load for the churn pool"),
        "{g}"
    );
    assert!(plan.violations().is_empty(), "{:?}", plan.violations());
}

/// The books close in both states: the cap's (every yield done: the host experts left, the
/// K/V at the cap, the window and the host's other terms, and the plan's headroom hold the
/// room, so the churn pool fits the plan's own accounting) and the floor's (every host
/// expert, the K/V at its floor, the host's other terms and the pool's room hold the room to
/// the byte). The sums are taken from the plan's totals, not from [`workstation::HostNeed`].
///
/// FAIL-first: a `HostNeed` that takes the yields past the K/V's bytes out of the shadows alone
/// (saturating at 0) counts the window beside the yields that freed it, and the cap's sum and
/// the headroom fall 3,200 B short of the room (71,528 B against 74,728 B); a pool bound
/// without the floor's free room plans a cap's sum of 73,104 B.
#[test]
fn the_books_close_at_the_floor_and_at_the_cap_with_the_pool_budgeted() {
    let model = layered(WIDE);
    let slack = 44 * EXPERT;
    let room = floor_need(&model, POOL_HELD) + slack;
    let tight = machine_of(WIDE, POOL_HELD, room);
    let plan = pooled(&model, &tight, true).expect("the plan at the floor");
    assert_books(&plan, room);
}

/// The sums of [`the_books_close_at_the_floor_and_at_the_cap_with_the_pool_budgeted`], over a
/// plan of the pool model made at `room`.
fn assert_books(plan: &Plan<'_>, room: u64) {
    let (h, g) = (&plan.host, &plan.host.grow);
    let need = workstation::HostNeed::of(plan, 0);
    assert!(
        g.yield_bytes > h.shadow_bytes + h.stage_kv_bytes + h.kv_host_bytes,
        "the yields pass the K/V's bytes, the case the sums below are about"
    );
    let sum = |terms: &[u64]| terms.iter().map(|&t| i128::from(t)).sum::<i128>();
    let cap_state = sum(&[
        h.expert_bytes,
        h.table_bytes,
        h.shadow_bytes,
        h.stage_kv_bytes,
        h.kv_host_bytes,
        h.reserve_bytes,
        h.pool_card_bytes,
    ]) - i128::from(g.yield_bytes)
        - i128::from(need.os);
    assert_eq!(
        cap_state + h.headroom_bytes,
        i128::from(room),
        "the cap's state"
    );
    let floor_state = sum(&[
        h.expert_bytes,
        h.table_bytes,
        g.floor_bytes,
        h.reserve_bytes - window_of(plan),
        h.pool_card_bytes,
        g.pool_bytes,
    ]) - i128::from(need.os);
    assert_eq!(
        floor_state,
        i128::from(room),
        "the floor's state, the pool beside it"
    );
    assert_eq!(
        i128::from(need.plan_bytes()) + h.headroom_bytes,
        i128::from(room),
        "the need and the headroom are the room"
    );
}

/// A plan the room holds at the cap splits nothing, and the pool's room is not budgeted: it is
/// the plan it was, as is the plan of a caller that clears the budget on a room the growth
/// passes (the K/V charged at the floor with the growth alone yielded, which leaves the pool
/// nothing).
///
/// FAIL-first: a budget that ignores the clear gives the cleared plan the pool's 25,344 B and
/// yields past its growth.
#[test]
fn a_plan_the_room_holds_and_a_plan_with_the_budget_cleared_keep_their_books() {
    let model = layered(WIDE);
    let ample = machine_of(WIDE, POOL_HELD, 1 << 40);
    let with = pooled(&model, &ample, true).expect("the plan with the budget");
    let without = pooled(&model, &ample, false).expect("the plan without it");
    assert_eq!(with.host.grow.hold, GrowHold::Room);
    let split = |p: &Plan<'_>| (format!("{:?}", p.host.grow), p.host.headroom_bytes);
    assert_eq!(split(&with).1, split(&without).1);
    assert_eq!(split_of(&with), split_of(&without));
    // Pool_room itself is the difference between the two growths, and nothing else.
    let mut cleared = with.host.grow;
    cleared.pool_room = false;
    assert!(cleared == without.host.grow);

    let room = floor_need(&model, POOL_HELD) + 44 * EXPERT;
    let tight = machine_of(WIDE, POOL_HELD, room);
    let old = pooled(&model, &tight, false).expect("the plan with the budget cleared");
    assert_eq!(old.host.grow.pool_bytes, 0);
    assert!(
        old.host.grow.yield_bytes <= old.host.grow.grow_bytes,
        "the growth alone yields: {}",
        old.host.grow
    );
    assert!(
        matches!(
            pick_over(&old, room),
            Room::Short(RoomShort::PoolUnhostable(_))
        ),
        "the plan that charges the growth alone leaves the pool nothing"
    );
}

/// A host whose free room at the floor cannot hold the pool still says so: the pool's room is
/// what the floor leaves, a pool past it is `pool_unhostable` by the same climb, and a floor
/// that leaves nothing (the room the host experts and the K/V's floor take, or less) budgets
/// no pool and plans as the cleared budget does.
///
/// FAIL-first: a budget without the floor's free room as its bound plans a pool of 16,704 B on
/// a room with 11,520 B of it; one that budgets nothing plans none.
#[test]
fn a_host_that_cannot_hold_the_pool_at_the_floor_still_reports_it() {
    let model = layered(WIDE);
    let floor = floor_need(&model, POOL_HELD);
    for (name, room) in [
        ("a part of the pool", floor + 20 * EXPERT),
        ("no room past the floor", floor),
        ("a room under the floor", floor - 2 * EXPERT),
    ] {
        let tight = machine_of(WIDE, POOL_HELD, room);
        let plan = pooled(&model, &tight, true).expect(name);
        let Room::Short(RoomShort::PoolUnhostable(short)) = pick_over(&plan, room) else {
            panic!("{name}: the pool is unhostable");
        };
        assert_eq!(
            short.needs,
            pool_at(&plan, 1),
            "{name}: the pool at the card's last count"
        );
        assert!(
            short.headroom < i128::from(short.needs),
            "{name}: {short:?}"
        );
        assert!(
            plan.violations().is_empty(),
            "{name}: {:?}",
            plan.violations()
        );
        if room <= floor {
            let old = pooled(&model, &tight, false).expect(name);
            let mut grow = plan.host.grow.clone();
            grow.pool_room = false;
            assert!(
                grow == old.host.grow,
                "{name}: no pool, the growth the cleared budget makes"
            );
            assert_eq!(split_of(&plan), split_of(&old), "{name}: the same split");
            assert_eq!(plan.host.headroom_bytes, old.host.headroom_bytes, "{name}");
        } else {
            assert_eq!(
                plan.host.grow.pool_bytes,
                20 * EXPERT,
                "{name}: the room's bytes"
            );
        }
    }
}

/// The pool's room is no more than the pool the cards' routed experts can make: a card that
/// holds one or two experts a layer has 24 of them to churn, and the 40 experts the floor
/// leaves free are not reserved past them.
///
/// FAIL-first: a budget without that bound takes the floor's whole free room, 23,040 B for a
/// pool of 13,824 B, and the yields give up 16 experts the pool never uses.
#[test]
fn the_pool_room_is_no_more_than_the_cards_experts_can_make() {
    let model = layered(WIDE);
    let held = 1;
    let room = floor_need(&model, held) + 40 * EXPERT;
    let tight = machine_of(WIDE, held, room);
    let plan = pooled(&model, &tight, true).expect("the plan at the floor");
    let g = &plan.host.grow;
    assert_eq!(g.hold, GrowHold::Yields, "{g}");
    let cards = pool_at(&plan, 0);
    assert!(
        0 < cards && cards < 40 * EXPERT,
        "{cards} B of card experts"
    );
    assert_eq!(g.pool_bytes, cards, "{g}");
    let headroom = u64::try_from(plan.host.headroom_bytes).expect("a headroom");
    assert!((cards..cards + EXPERT).contains(&headroom), "{headroom} B");
}

/// The pool model's machine with the host's one reserve named as the OS's, so the need's page
/// classes can be read ([`workstation::HostNeed::classes`] refuses a reserve with no class).
fn named_machine_of(held: u64, room: u64) -> Machine {
    let mut machine = machine_of(WIDE, held, room);
    machine.host_mut().reserves = vec![(workstation::OS_RESERVE.to_string(), BESIDE)];
    machine
}

/// The pool model's host need at the cap on [`named_machine_of`], every host expert resident:
/// a room of this plus `h` holds the cap with `h` to spare.
fn cap_need(model: &ModelTensors, held: u64) -> u64 {
    let ample = named_machine_of(held, 1 << 40);
    let whole = pooled(model, &ample, true).expect("the plan at an ample room");
    u64::try_from(workstation::HostNeed::of(&whole, 0).exact(true)).expect("u64")
}

/// A room that holds the cap with less than the split's window to spare charges the K/V at its
/// floor all the same, and the pool takes the floor's free room: the growth's bytes more than
/// the headroom the cap leaves (a pool of the room mode's headroom, `h`, is the plan the room
/// made before). Here the model's window is 10 experts, the growth 48 and the cards' pool 56,
/// and the room holds the cap with 6 to spare [derived by hand]: the pool is 6 + 48 = 54
/// experts, the yields the growth's and the window's, 58, and the books close at the floor and
/// at the cap.
///
/// FAIL-first: a split that is not taken (the early return as it was) leaves `grow.hold` at
/// `Room` with no yield and the pool at nothing.
#[test]
fn a_room_that_holds_the_cap_without_the_window_beside_it_yields_for_a_pool() {
    let model = layered(WIDE);
    let room = cap_need(&model, POOL_HELD) + 6 * EXPERT;
    let tight = named_machine_of(POOL_HELD, room);
    let plan = pooled(&model, &tight, true).expect("the plan at the floor");
    let window = window_of(&plan);
    assert_eq!(
        window,
        10 * EXPERT,
        "two of the heaviest layer's host experts"
    );
    let g = &plan.host.grow;
    assert_eq!(g.hold, GrowHold::Yields, "{g}");
    assert_eq!(plan.nvme_bytes, 0, "every expert is on the host at load");
    assert_eq!(g.pool_bytes, 6 * EXPERT + POOL_GROWTH, "{g}");
    assert!(
        g.pool_bytes > 6 * EXPERT,
        "more pool than the headroom the cap leaves"
    );
    let owed = POOL_GROWTH + window;
    assert!(
        (owed..owed + EXPERT).contains(&g.yield_bytes),
        "{} B of yields for the growth's {POOL_GROWTH} B and the window's {window} B",
        g.yield_bytes
    );
    assert!(plan.violations().is_empty(), "{:?}", plan.violations());
    assert_books(&plan, room);
}

/// Whether a room that holds the cap yields is the window's: the room holds the cap with `h` to
/// spare, and a plan with `h` under the window yields while one with the window or more keeps the
/// plan an ample room makes, no yield and the K/V charged at the cap. Every plan the rule makes
/// is one its need can be classed by (`HostNeed::classes`) and holds the room. Run for cards that
/// hold 3 and 4 experts a layer.
///
/// FAIL-first: a split taken at any `h` (the first form of the rule: the yields branch with
/// `over` clamped at 0) yields at `h` = the window and more; a split taken only when `h` is
/// under the window less a byte keeps the plan at `h` = the window less a byte.
#[test]
fn a_room_that_holds_the_cap_and_the_window_beside_it_keeps_its_plan() {
    use workstation::{HostNeed, Hosting};
    let model = layered(WIDE);
    let how = Hosting {
        lock: false,
        arena_pinned: false,
    };
    for held in [POOL_HELD, 4] {
        let cap = cap_need(&model, held);
        let roomy = named_machine_of(held, 1 << 40);
        let ample = pooled(&model, &roomy, true).expect("the plan at an ample room");
        let window = window_of(&ample);
        for h in [
            0,
            EXPERT,
            window - EXPERT,
            window - 1,
            window,
            window + 1,
            window + 6 * EXPERT,
        ] {
            let room = cap + h;
            let tight = named_machine_of(held, room);
            let plan = pooled(&model, &tight, true).expect("the plan");
            let g = &plan.host.grow;
            let want = if h < window {
                GrowHold::Yields
            } else {
                GrowHold::Room
            };
            assert_eq!(g.hold, want, "held {held}, h = {h}: {g}");
            if want == GrowHold::Room {
                assert!(g.yields.is_empty() && g.pool_bytes == 0, "h = {h}: {g}");
                assert_eq!(split_of(&plan), split_of(&ample), "h = {h}: no split");
            }
            let need = HostNeed::of(&plan, 0);
            need.classes(&plan, how)
                .unwrap_or_else(|e| panic!("held {held}, h = {h}: {e}"));
            assert!(
                plan.violations().is_empty(),
                "held {held}, h = {h}: {:?}",
                plan.violations()
            );
            assert!(
                need.plan_bytes() <= room,
                "held {held}, h = {h}: the need {} B holds the room {room} B",
                need.plan_bytes()
            );
        }
    }
}

/// A room that holds the cap without the window keeps its plan where the yields would gain the
/// pool nothing: a caller that cleared the budget, and cards whose whole pool the headroom the
/// cap leaves already holds. The pool the split budgets must pass that headroom: at `h` = the
/// cards' pool (8 experts, the cards that hold the fewest) the room keeps its plan and one byte
/// under it yields.
///
/// FAIL-first: a split taken whatever the pool budgeted yields the window's bytes on the cleared
/// budget and at `h` = the cards' pool; a pool that must only reach the headroom (`>=`) does the
/// same at `h` = the cards' pool.
#[test]
fn a_room_that_holds_the_cap_keeps_its_plan_where_the_yields_gain_no_pool() {
    let model = layered(WIDE);
    let kept = |name: &str, plan: &Plan<'_>, hold: GrowHold| {
        let g = &plan.host.grow;
        assert_eq!(g.hold, hold, "{name}: {g}");
        assert_eq!(g.pool_bytes > 0, hold == GrowHold::Yields, "{name}: {g}");
    };
    let cleared = named_machine_of(POOL_HELD, cap_need(&model, POOL_HELD) + 6 * EXPERT);
    let plan = pooled(&model, &cleared, false).expect("the plan with the budget cleared");
    assert!(
        window_of(&plan) > 6 * EXPERT,
        "the window is past the headroom"
    );
    kept("a cleared budget", &plan, GrowHold::Room);
    let held = 0;
    let cap = cap_need(&model, held);
    let roomy = named_machine_of(held, 1 << 40);
    let ample = pooled(&model, &roomy, true).expect("the plan at an ample room");
    let cards = pool_at(&ample, 0);
    assert_eq!(
        cards,
        8 * EXPERT,
        "the pool of the cards that hold the fewest"
    );
    assert!(cards < window_of(&ample), "the window is past the headroom");
    for (name, room, hold) in [
        (
            "the cards' pool in the headroom",
            cap + cards,
            GrowHold::Room,
        ),
        ("a byte under it", cap + cards - 1, GrowHold::Yields),
    ] {
        let machine = named_machine_of(held, room);
        let plan = pooled(&model, &machine, true).expect(name);
        kept(name, &plan, hold);
    }
}

/// A room that holds the cap without the window keeps its plan when its split would take an
/// arena, and ignores an arena lever it never asked for: the plan is the one an ample room makes
/// under a given arena below the arena's floor (refused by name where the cap is passed) and
/// under the prompt passes' own grant.
///
/// FAIL-first: a split that returns the lever's refusal on a room that never needed it ends
/// `NvTierArenaUnder` (the arena of 1 B is under its floor of 7,274,496 B).
#[test]
fn a_room_that_holds_the_cap_ignores_an_arena_lever_it_never_asked_for() {
    let model = layered(WIDE);
    let room = cap_need(&model, POOL_HELD) + 6 * EXPERT;
    let tight = named_machine_of(POOL_HELD, room);
    let probe = pooled(&model, &tight, true).expect("a plan to read the arena's floor of");
    let least = ArenaFloor::of(&probe).expect("the floor").bytes;
    assert!(least > 1, "a given arena of 1 B is under the floor");
    let under = pooled_by(&model, &tight, true, |plan, room| {
        expert_nvme_tier(plan, room, Some(1))
    })
    .unwrap_or_else(|e| panic!("a room that holds the cap never reads the lever: {e}"));
    let passes = pooled_by(&model, &tight, true, expert_nvme_tier_for_passes)
        .expect("the passes' split of a room that holds the cap");
    for (name, plan) in [
        ("a given arena under the least", under),
        ("the passes' grant", passes),
    ] {
        assert_eq!(plan.host.grow.hold, GrowHold::Room, "{name}");
        assert_eq!(plan.host.nvme_arena_bytes, 0, "{name}");
    }
}

/// A room that holds the cap takes the split only where both hold: the window past what the room
/// leaves beside the cap (`over` > 0), and a pool past the headroom the cap leaves.
/// Each row of the table breaks one of them against a row that takes it.
///
/// FAIL-first: a predicate without the window term takes the split at `over` 0, and with `>=` at a
/// pool equal to the headroom.
#[test]
fn a_room_that_holds_the_cap_takes_the_split_only_where_it_pays() {
    // (over, pool, headroom, takes)
    let rows = [
        (1, 11, 10, true),
        (0, 11, 10, false),
        (1, 10, 10, false),
        (1, 0, 0, false),
    ];
    for (over, pool, headroom, takes) in rows {
        assert_eq!(
            room::split_pays(over, pool, headroom),
            takes,
            "over {over}, pool {pool}, headroom {headroom}"
        );
    }
}
