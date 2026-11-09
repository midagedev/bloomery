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
