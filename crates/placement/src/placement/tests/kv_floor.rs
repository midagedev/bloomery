//! The plan's KV floor ([`PlanLevers::kv_floor`], `BLOOMERY_KV_FLOOR`) over
//! [`layered`]'s four-layer model: the expert spread runs at the floor's
//! positions while every other term stays at the cap, a floor above the cap
//! is refused by name, and unset the plan is the one made before the lever.

use super::*;
use bloomery_levers::KvFloor;
use runtime::stores::{Access, PosBuf, PosLayout};

/// 18 B of KV a position a layer (72 B a position over the four layers) and
/// 4 B of stage rows a position: ten positions of KV or twenty of rows take
/// one [`EXPERT`].
struct Kv;

/// The four layers' 18 B a position, one scanned buffer each: all of the layer, so a floor
/// reads all of it.
static KV_ROOMS: std::sync::LazyLock<PosLayout> = std::sync::LazyLock::new(|| {
    let layer = || {
        vec![PosBuf {
            what: "kv",
            row: 18,
            per: 1,
            runs: 1,
            access: Access::Scan,
        }]
    };
    PosLayout::new((0..4).map(|_| layer()).collect())
});

impl KvBytes for Kv {
    fn rooms(&self) -> Option<&PosLayout> {
        Some(&KV_ROOMS)
    }

    fn layer_bytes(&self, _layer: usize, ctx_max: u64) -> u64 {
        18 * ctx_max
    }

    fn stage_position_bytes(&self, ctx_max: u64) -> u64 {
        4 * ctx_max
    }
}

/// A card of the head, the cap's rows and KV and ten experts' room.
fn card_of_cap_kv_and_experts(cap: u64, experts: u64) -> Machine {
    Machine {
        cards: vec![bytes_card(
            "stage",
            HEAD + Kv.stage_position_bytes(cap) + 4 * 18 * cap + experts * EXPERT,
            0..4,
        )],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    }
}

/// The expert spread runs at the floor the levers name and every other term
/// at the cap: a plan at 80 positions with the floor 8 holds the KV of 8
/// positions — 5,184 B less than the cap's 5,760, room for nine more
/// experts — while its stage rows stay the cap's and its `ctx_max` is 80.
/// Unset, the spread runs at the cap itself: the plan's cache term is
/// exactly the cap's KV, the plan made before the lever.
///
/// FAIL-first: a floor read at the cap holds the cap's ten experts and no
/// more.
#[test]
fn the_spread_runs_at_the_floor_and_every_other_term_at_the_cap() {
    let model = layered(4);
    let machine = card_of_cap_kv_and_experts(80, 10);
    let unset = plan(&model, &machine, 80, &Kv, &PlanLevers::default()).expect("a plan");
    assert_eq!(unset.n_l, [3, 3, 2, 2], "ten experts at the cap's KV");
    assert_eq!(unset.kv_floor, None);
    assert_eq!(
        unset.cards[0].kv_bytes,
        (0..4).map(|l| Kv.layer_bytes(l, 80)).sum::<u64>(),
        "unset, the cache term is the cap's KV"
    );
    assert_eq!(unset.cards[0].stage_rows_bytes, 4 * 80);
    let floored = plan(
        &model,
        &machine,
        80,
        &Kv,
        &PlanLevers {
            kv_floor: Some(8),
            ..PlanLevers::default()
        },
    )
    .expect("a plan");
    assert_eq!(
        floored.n_l,
        [5, 5, 5, 4],
        "nineteen experts at the floor's KV"
    );
    assert_eq!(floored.kv_floor, Some(8));
    assert_eq!(floored.ctx_max, 80, "the cap stays the plan's context");
    assert_eq!(
        floored.cards[0].kv_bytes,
        (0..4).map(|l| Kv.layer_bytes(l, 8)).sum::<u64>(),
        "the cache term is the floor's KV"
    );
    assert_eq!(
        (floored.cards[0].stage_rows_bytes, floored.cards[0].experts),
        (4 * 80, 19),
        "the stage rows stay the cap's"
    );
}

/// A floor above the plan's context is refused by name, clamped by nothing:
/// the refusal names the lever, the floor and the cap. A floor at the cap is
/// the cap's own plan.
#[test]
fn a_floor_above_the_cap_is_refused_by_name() {
    let model = layered(4);
    let machine = card_of_cap_kv_and_experts(80, 10);
    let over = plan(
        &model,
        &machine,
        80,
        &Kv,
        &PlanLevers {
            kv_floor: Some(81),
            ..PlanLevers::default()
        },
    )
    .expect_err("a floor above the cap is refused");
    assert!(matches!(
        over,
        PlacementError::KvFloor { floor: 81, cap: 80 }
    ));
    let text = over.to_string();
    assert!(
        text.contains("BLOOMERY_KV_FLOOR") && text.contains("81") && text.contains("80"),
        "{text}"
    );
    let unset = plan(&model, &machine, 80, &Kv, &PlanLevers::default()).expect("a plan");
    let at = plan(
        &model,
        &machine,
        80,
        &Kv,
        &PlanLevers {
            kv_floor: Some(80),
            ..PlanLevers::default()
        },
    )
    .expect("a floor at the cap is the cap's own plan");
    assert_eq!(
        (at.cards[0].kv_bytes, at.n_l.as_slice()),
        (unset.cards[0].kv_bytes, unset.n_l.as_slice()),
        "a floor at the cap is the cap's own plan"
    );
}

/// A load's own default for the unset K/V levers, as a seat hands it: the server's.
const DEFAULT: KvDefault = KvDefault::SERVER;

/// The server's default for the K/V levers a seat leaves unset: the experts' spread at 8,192
/// positions, the selected rows' pages spilling to the host. Every seat of a body in rooms takes
/// it, so a change of either value is a change of every such seat's plan.
///
/// FAIL-first: a floor of another value or a word other than `spill` in the constant is red.
#[test]
fn the_server_default_is_a_floor_of_8192_with_a_spill() {
    assert_eq!(
        (KvDefault::SERVER.floor, KvDefault::SERVER.host),
        (8192, KvHost::Spill)
    );
}

/// [`plan`] at `cap` under a load's resolution of `floor` and `host`
/// ([`PlanLevers::resolve_kv`]) against [`DEFAULT`].
fn seat<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    cap: u64,
    (floor, host): (KvFloor, Option<KvHost>),
) -> Result<Plan<'a>, PlacementError> {
    let levers = PlanLevers::default().resolve_kv(floor, host, DEFAULT);
    plan(model, machine, cap, &Kv, &levers)
}

/// A load's default for the unset K/V levers ([`PlanLevers::resolve_kv`]): the
/// spread runs at its floor and the planes take its backing; a context no
/// more than the floor has none (no refusal), `off` is the whole context, a
/// set floor is its own value (and above the cap still refused), and a set
/// `all` takes no default floor.
///
/// FAIL-first: the unset levers resolved to the plan made before the
/// defaults, the whole context and the card's own memory.
#[test]
fn a_loads_unset_k_v_levers_take_its_default() {
    const FLOOR: u64 = DEFAULT.floor;
    let model = layered(4);
    let deep = 2 * FLOOR;
    let machine = card_of_cap_kv_and_experts(deep, 10);
    let whole = plan(&model, &machine, deep, &Kv, &PlanLevers::default()).expect("a plan");
    let unset = seat(&model, &machine, deep, (KvFloor::Unset, None)).expect("a plan");
    assert_eq!(unset.kv_floor, Some(FLOOR));
    assert_eq!(unset.kv_host, KvHost::Spill);
    assert_eq!(unset.ctx_max, deep);
    assert!(
        unset.cards[0].experts > whole.cards[0].experts,
        "the spread at the floor holds more experts than the whole context's"
    );
    assert_eq!(
        unset.cards[0].kv_bytes,
        (0..4).map(|l| Kv.layer_bytes(l, FLOOR)).sum::<u64>()
    );
    // At or under the floor the plan is the whole context's, floor-less.
    for cap in [FLOOR, FLOOR / 2] {
        let small_card = card_of_cap_kv_and_experts(cap, 10);
        let small = seat(&model, &small_card, cap, (KvFloor::Unset, None))
            .expect("no refusal at a small context");
        let base = plan(&model, &small_card, cap, &Kv, &PlanLevers::default()).expect("a plan");
        assert_eq!(small.kv_floor, None, "cap {cap}");
        assert_eq!(small.n_l, base.n_l, "cap {cap}");
    }
    // `off` is the whole context's spread; a set host word is its own.
    let off = seat(&model, &machine, deep, (KvFloor::Off, Some(KvHost::Off))).expect("a plan");
    assert_eq!((off.kv_floor, off.kv_host), (None, KvHost::Off));
    assert_eq!(off.n_l, whole.n_l);
    let set = seat(&model, &machine, deep, (KvFloor::At(4096), None)).expect("a plan");
    assert_eq!(set.kv_floor, Some(4096));
    let over =
        seat(&model, &machine, deep, (KvFloor::At(deep + 1), None)).expect_err("refused by name");
    assert!(matches!(over, PlacementError::KvFloor { .. }));
    // An unset floor beside `all` is no floor; a set one is still refused.
    let all = seat(&model, &machine, deep, (KvFloor::Unset, Some(KvHost::All)))
        .expect("all takes no default floor");
    assert_eq!((all.kv_floor, all.kv_host), (None, KvHost::All));
    let both = seat(
        &model,
        &machine,
        deep,
        (KvFloor::At(4096), Some(KvHost::All)),
    )
    .expect_err("refused by name");
    assert!(matches!(both, PlacementError::KvHostFloor { floor: 4096 }));
}

/// A seat's default is read off its stores' rooms ([`KvDefault::of`]): the server's where the cache
/// declares rooms, none where it declares none. A load of rooms that takes it spreads its experts
/// at the floor, which the same plan with the levers left alone does not; a load of plain buffers
/// takes none, and the server's default handed to it would be refused by name, so a seat that
/// handed it one would refuse every start of that family.
///
/// FAIL-first: `KvDefault::of` answering none for a cache of rooms (the unset levers plan the whole
/// context's spread), or the server's default for a plain one (the unset `spill` refused by name).
#[test]
fn a_seats_default_is_read_off_its_stores_rooms() {
    /// The same bytes as [`Kv`], held in plain buffers.
    struct Plain;
    impl KvBytes for Plain {
        fn layer_bytes(&self, _layer: usize, ctx_max: u64) -> u64 {
            18 * ctx_max
        }
    }
    assert_eq!(KvDefault::of(&Kv), Some(KvDefault::SERVER));
    assert_eq!(KvDefault::of(&Plain), None);

    let model = layered(4);
    let deep = 2 * KvDefault::SERVER.floor;
    let machine = card_of_cap_kv_and_experts(deep, 10);
    let default = KvDefault::of(&Kv).expect("a cache of rooms has a default");
    let levers = PlanLevers::default().resolve_kv(KvFloor::Unset, None, default);
    let seated = plan(&model, &machine, deep, &Kv, &levers).expect("a plan");
    let whole = plan(&model, &machine, deep, &Kv, &PlanLevers::default()).expect("a plan");
    assert_eq!(
        (seated.kv_floor, seated.kv_host),
        (Some(KvDefault::SERVER.floor), KvHost::Spill)
    );
    assert!(
        seated.cards[0].experts > whole.cards[0].experts,
        "the spread at the floor holds more experts than the whole context's"
    );

    let plain = plan(&model, &machine, deep, &Plain, &PlanLevers::default())
        .expect("a plain cache plans with its levers unset");
    assert_eq!((plain.kv_floor, plain.kv_host), (None, KvHost::Off));
    let handed = PlanLevers::default().resolve_kv(KvFloor::Unset, None, KvDefault::SERVER);
    assert!(matches!(
        plan(&model, &machine, deep, &Plain, &handed),
        Err(PlacementError::KvUnroomed { .. })
    ));
}

/// A body that loads one sequence's stores holds the plan's card term against the planner's own
/// rule ([`Plan::stores_counted`]), not against the stores' exact bytes at the cap: the term reads
/// the rooms at the plan's floor and carries the rounding the cache counts beside its rows
/// (`Padded`: 100 B a layer that does not move with the floor). So the figure a body compares is
/// the plan's own for every floor and backing the levers name, and is none for a card the plan
/// does not have. The layout at the cap, which a body's stores hold, is another number under a
/// floor or a rounding [derived: the four layers' 18 B a position over 80 positions is 5,760 B,
/// and 6,160 B with the 400 B of rounding; the floor of 8 holds 576 B of rows and the rounding's
/// 400 B, 976 B].
///
/// FAIL-first: `stores_counted` reading the layout at the cap (the floor and the rounding left to
/// the plan) is red at the first floor row: 5,760 B against the plan's 576 B.
#[test]
fn the_cards_stores_are_counted_by_the_planners_own_rule() {
    /// [`Kv`]'s rooms with a rounding of 100 B a layer beside the rows.
    struct Padded;
    impl KvBytes for Padded {
        fn rooms(&self) -> Option<&PosLayout> {
            Kv.rooms()
        }

        fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
            Kv.layer_bytes(layer, ctx_max) + 100
        }

        fn stage_position_bytes(&self, ctx_max: u64) -> u64 {
            Kv.stage_position_bytes(ctx_max)
        }
    }

    let model = layered(4);
    let machine = card_of_cap_kv_and_experts(80, 10);
    for (name, kv) in [("rows", &Kv as &dyn KvBytes), ("padded", &Padded)] {
        for (what, levers) in [
            ("unset", PlanLevers::default()),
            (
                "floor",
                PlanLevers {
                    kv_floor: Some(8),
                    ..PlanLevers::default()
                },
            ),
            (
                "floor and spill",
                PlanLevers {
                    kv_floor: Some(8),
                    kv_host: KvHost::Spill,
                    ..PlanLevers::default()
                },
            ),
            (
                "all",
                PlanLevers {
                    kv_host: KvHost::All,
                    ..PlanLevers::default()
                },
            ),
        ] {
            let planned = plan(&model, &machine, 80, kv, &levers).expect("a plan");
            assert_eq!(
                planned.stores_counted(0, kv),
                Some(planned.cards[0].kv_bytes),
                "{name}, {what}"
            );
            assert_eq!(
                planned.stores_counted(1, kv),
                None,
                "{name}, {what}: no card 1"
            );
        }
    }
    let floor = PlanLevers {
        kv_floor: Some(8),
        ..PlanLevers::default()
    };
    let padded = plan(&model, &machine, 80, &Padded, &floor).expect("a plan");
    let cap: u64 = (0..4).map(|l| Padded.layer_bytes(l, 80)).sum();
    assert_eq!(
        (padded.cards[0].kv_bytes, cap),
        (976, 6_160),
        "the card's term under the floor, and the layout at the cap"
    );
}
