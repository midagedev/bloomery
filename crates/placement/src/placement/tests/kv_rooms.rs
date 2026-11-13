//! What the plan reads of a cache by the rooms its [`KvBytes`] declares ([`KvBytes::rooms`]):
//! a load that declares none plans every byte at the cap under any floor, and a load that does
//! reads only the rooms' by-position buffers at the floor, the `Select` ones dropped under
//! `BLOOMERY_KV_HOST=all`.

use std::sync::LazyLock;

use super::*;
use bloomery_levers::KvHost;
use runtime::stores::{Access, PosBuf, PosLayout};

/// 100 B of fixed state a layer and, a position, 10 B scanned and 8 B selected.
const FIXED: u64 = 100;
const SCANNED: u64 = 10;
const SELECTED: u64 = 8;

static ROOMS: LazyLock<PosLayout> = LazyLock::new(|| {
    let layer = || {
        vec![
            PosBuf {
                what: "scanned",
                row: SCANNED,
                per: 1,
                runs: 1,
                access: Access::Scan,
            },
            PosBuf {
                what: "selected",
                row: SELECTED,
                per: 1,
                runs: 1,
                access: Access::Select { rows: 4 },
            },
        ]
    };
    PosLayout::new((0..4).map(|_| layer()).collect())
});

/// The four layers' bytes, declaring its rooms or not.
struct Layers {
    declares: bool,
}

impl KvBytes for Layers {
    fn layer_bytes(&self, _layer: usize, ctx_max: u64) -> u64 {
        FIXED + (SCANNED + SELECTED) * ctx_max
    }

    fn rooms(&self) -> Option<&PosLayout> {
        self.declares.then_some(&*ROOMS)
    }
}

fn kv_of(plan: &Plan<'_>) -> u64 {
    plan.cards[0].kv_bytes
}

fn machine() -> Machine {
    Machine {
        cards: vec![bytes_card("stage", HEAD + 4 * 10_000 + 10 * EXPERT, 0..4)],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    }
}

fn planned(declares: bool, floor: Option<u64>, kv_host: KvHost) -> u64 {
    let model = layered(4);
    let levers = PlanLevers {
        kv_floor: floor,
        kv_host,
        ..PlanLevers::default()
    };
    let machine = machine();
    let plan = plan(&model, &machine, 80, &Layers { declares }, &levers).expect("a plan");
    kv_of(&plan)
}

/// A load that declares no rooms plans every byte of its cache at the cap: no floor and no
/// backing lever is set (`off` is unset's value), so none has anything to read.
#[test]
fn a_load_with_no_rooms_plans_every_term_at_the_cap() {
    assert_eq!(
        planned(false, None, KvHost::Off),
        4 * (FIXED + (SCANNED + SELECTED) * 80)
    );
}

fn refused(declares: bool, floor: Option<u64>, kv_host: KvHost) -> Option<PlacementError> {
    let model = layered(4);
    let levers = PlanLevers {
        kv_floor: floor,
        kv_host,
        ..PlanLevers::default()
    };
    let machine = machine();
    plan(&model, &machine, 80, &Layers { declares }, &levers).err()
}

/// A lever that does nothing is refused by name: a floor count, or `spill`/`all`, on a load that
/// declares no rooms; the same levers on one that declares them plan.
///
/// FAIL-first: dropping the `refuse_unroomed` call in `plan_rule` turns the first three
/// assertions red (a plan at the cap instead of the refusal).
#[test]
fn a_set_kv_lever_on_a_load_with_no_rooms_is_refused_by_name() {
    for (floor, host, lever) in [
        (Some(8), KvHost::Off, "BLOOMERY_KV_FLOOR"),
        (None, KvHost::Spill, "BLOOMERY_KV_HOST"),
        (None, KvHost::All, "BLOOMERY_KV_HOST"),
    ] {
        match refused(false, floor, host) {
            Some(PlacementError::KvUnroomed { lever: got, .. }) => assert_eq!(got, lever),
            other => panic!("floor {floor:?} host {host:?}: {other:?}"),
        }
    }
    assert!(
        refused(false, None, KvHost::Off).is_none(),
        "unset and off pass"
    );
    assert!(
        refused(true, Some(8), KvHost::Spill).is_none(),
        "a rooms load takes both"
    );
}

/// A load that declares rooms reads their buffers at the floor and the rest at the cap; under
/// `all` the host-able buffers have no card bytes at all and the scanned ones keep theirs.
///
/// FAIL-first: `all` dropping every room (`rooms.bytes` for `rooms.host_able`) leaves 4 · 100.
#[test]
fn rooms_are_read_at_the_floor_and_the_selected_ones_leave_the_card_under_all() {
    let (scanned, selected) = (SCANNED, SELECTED);
    assert_eq!(
        planned(true, None, KvHost::Off),
        4 * (FIXED + (scanned + selected) * 80),
        "no floor: the cap's own cache"
    );
    for host in [KvHost::Off, KvHost::Spill] {
        assert_eq!(
            planned(true, Some(8), host),
            4 * (FIXED + (scanned + selected) * 8),
            "a floor of 8 under {host:?}"
        );
    }
    assert_eq!(
        planned(true, None, KvHost::All),
        4 * (FIXED + scanned * 80),
        "all: the selected buffers off the card, the scanned ones kept"
    );
}

/// Pages the plan counts on the host, a made-up page-rounded figure a caller computed.
const PAGES: u64 = 1_000;
/// The host terms beside the routed experts the machines below carry.
const BESIDE: u64 = 1_000;

/// The four layers on a card that holds some of the experts and a host whose figure is `room`.
fn host_machine(room: u64) -> Machine {
    Machine {
        cards: vec![bytes_card(
            "stage",
            HEAD + 4 * (FIXED + 90 * 10) + 12 * EXPERT,
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

/// The steps every plan that backs K/V pages on the host takes in this order: the routed
/// placement, the pages counted ([`Plan::kv_host_planes`], none for 0), the room the host
/// figure states, the NVMe tier's split with no arena. `Spill` and `Off` plan at a floor of
/// 40 of the 80 positions.
fn host_plan<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    lever: KvHost,
    pages: u64,
) -> Result<Plan<'a>, PlacementError> {
    let levers = PlanLevers {
        kv_floor: (lever != KvHost::All).then_some(40),
        kv_host: lever,
        ..PlanLevers::default()
    };
    let mut plan = plan(model, machine, 80, &Layers { declares: true }, &levers)?;
    if pages > 0 {
        plan.kv_host_planes(pages)?;
    }
    let room = plan.plan_at(&(machine.host().usable_bytes, workstation::HostRead::Given));
    expert_nvme_tier(&mut plan, room, Some(0))?;
    Ok(plan)
}

/// On a host whose figure holds its routed experts and no more, the pages counted beside them
/// displace experts: the NVMe tier takes those, and the host's need counts the pages. The
/// pages are no refusal while a split can make their room.
///
/// FAIL-first: the pages refused by the headroom the host's routed experts leave before the
/// split (`kv_host_planes` reading `headroom_bytes` alone) is `KvHostOverHost` here, with the
/// experts still on the host.
#[test]
fn pages_the_host_experts_displace_go_to_the_nvme_tier() {
    let model = layered(4);
    let ample = host_machine(1 << 40);
    let whole = host_plan(&model, &ample, KvHost::Off, 0).expect("a plan");
    assert!(
        whole.host.expert_bytes > 0,
        "the card leaves experts to the host"
    );
    let need = u64::try_from(workstation::HostNeed::of(&whole, 0).exact(true)).expect("u64");
    let tight = host_machine(need);
    let off = host_plan(&model, &tight, KvHost::Off, 0).expect("the room holds the need");
    assert_eq!(
        off.host.nvme_expert_bytes, 0,
        "without pages nothing is paged"
    );
    let spill = host_plan(&model, &tight, KvHost::Spill, PAGES)
        .expect("the pages are made room by the split, not refused");
    assert_eq!(
        spill.host.kv_host_bytes, PAGES,
        "the plan counts the pages the caller gave"
    );
    assert_eq!(
        workstation::HostNeed::of(&spill, 0).shadows,
        PAGES,
        "the need counts the pages"
    );
    // The split keeps `held − PAGES − 2 W` bytes on the host (W: the heaviest layer's host
    // bytes), so at least the pages and the window leave it.
    let heavy = (0..4)
        .map(|l| (8 - whole.n_l[l]) * EXPERT)
        .max()
        .expect("layers");
    assert!(
        spill.host.nvme_expert_bytes >= PAGES + 2 * heavy,
        "{} B paged for {PAGES} B of pages and a {heavy} B layer",
        spill.host.nvme_expert_bytes
    );
    assert_eq!(
        spill.host.expert_bytes + spill.host.nvme_expert_bytes,
        whole.host.expert_bytes,
        "every host expert is on the host or the NVMe tier"
    );
    assert!(
        spill.host.headroom_bytes >= 0,
        "the split leaves the room whole"
    );
    assert!(spill.violations().is_empty(), "{:?}", spill.violations());
}

/// Pages that pass the room even with no routed expert on the host are refused by name, with
/// the lever in force and what it holds: `spill` its pages past the floor, `all` the planes.
/// The headroom the refusal names is what the room leaves them past the other host terms, 0
/// where those pass it alone.
///
/// FAIL-first: the refusal reading the headroom with the routed experts in it names 0 B for
/// both, and one with no `lever` says `=all` under `spill`.
#[test]
fn pages_no_split_can_make_room_for_are_refused_naming_the_lever() {
    let model = layered(4);
    for (lever, held, instead) in [
        (
            KvHost::Spill,
            "the K/V planes' pages past the floor",
            "off)",
        ),
        (KvHost::All, "the K/V planes,", "off or spill)"),
    ] {
        let ample = host_machine(1 << 40);
        let whole = host_plan(&model, &ample, lever, 0).expect("a plan");
        // The host terms with no routed expert in them, and a room one byte short of them and
        // the pages.
        let others = u64::try_from(workstation::HostNeed::of(&whole, 0).exact(false)).expect("u64")
            - whole.host.expert_bytes;
        let machine = host_machine(others + PAGES - 1);
        let err = host_plan(&model, &machine, lever, PAGES).expect_err("the room cannot hold them");
        assert!(
            matches!(
                err,
                PlacementError::KvHostOverHost { lever: got, bytes: PAGES, headroom }
                    if got == lever && headroom == PAGES - 1
            ),
            "{lever:?}: {err:?}"
        );
        let text = err.to_string();
        assert!(
            text.contains(&format!("BLOOMERY_KV_HOST={}", lever.name()))
                && text.contains(held)
                && text.ends_with(&format!("(BLOOMERY_KV_HOST {instead}")),
            "{text}"
        );
        assert!(
            lever == KvHost::All || !text.contains("=all"),
            "spill's text names all: {text}"
        );
        // A room the other host terms pass alone leaves the pages no byte.
        let machine = host_machine(others - 1);
        let err = host_plan(&model, &machine, lever, PAGES).expect_err("the room is short");
        assert!(
            matches!(err, PlacementError::KvHostOverHost { headroom: 0, .. }),
            "{lever:?}: {err:?}"
        );
        // The room one byte more passes the pages' own check; the split's floor, which wants
        // three layers' host bytes more, speaks then.
        let machine = host_machine(others + PAGES);
        let err = host_plan(&model, &machine, lever, PAGES).expect_err("under the split's floor");
        assert!(
            matches!(err, PlacementError::HostRoomFloor { room, .. } if room == others + PAGES),
            "{lever:?}: {err:?}"
        );
    }
}
