//! A plan that reads its K/V at a floor charges the host K/V there and keeps on the host the
//! experts the static split pages at the cap that the room between the two holds: the need is
//! the cap's, the cap stays a refusal where the room cannot reach it, and the yields, done in
//! order, end at the static split ([`HostGrow`]).

use std::sync::{Arc, LazyLock};

use super::*;
use bloomery_levers::KvHost;
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

static ROOMS: LazyLock<PosLayout> = LazyLock::new(|| {
    PosLayout::new(
        (0..4)
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
});

/// Four layers of 100 B of state and 8 B a position, with a ring shadow of [`SHADOW`] B a
/// position on the host.
struct Shadowed;

impl KvBytes for Shadowed {
    fn layer_bytes(&self, _layer: usize, ctx_max: u64) -> u64 {
        100 + 8 * ctx_max
    }

    fn shadow_bytes(&self, _layer: usize, ctx_max: u64) -> u64 {
        SHADOW * ctx_max
    }

    fn rooms(&self) -> Option<&PosLayout> {
        Some(&*ROOMS)
    }

    fn host_yields(&self) -> bool {
        true
    }
}

/// The four layers on a card that holds some of the experts, beside a host of `room`.
fn machine(room: u64) -> Machine {
    Machine {
        cards: vec![bytes_card(
            "stage",
            HEAD + 4 * (100 + 8 * CAP) + 12 * EXPERT,
            0..4,
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
    let mut plan = plan(model, machine, CAP, &Shadowed, &levers)?;
    if !arm {
        plan.host.grow = HostGrow::default();
    }
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
