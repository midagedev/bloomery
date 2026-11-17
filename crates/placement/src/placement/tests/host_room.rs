//! The room a plan is made at is the machine's own: a reading or a given
//! room past this workstation's figure plans as it is, and the figure is
//! the room of a plan made with none.

use super::*;

const GIB: u64 = 1 << 30;

/// `layers` layers of routed stacks of `stack` bytes each (eight experts) on
/// a card that holds none of them, beside this workstation's nominal host,
/// or beside a unified pool of `pool` bytes (host figure and pool the same
/// bytes, as `workstation::host_of` makes them).
fn leg(layers: usize, stack: u64, pool: Option<u64>) -> (ModelTensors, Machine) {
    let mut model = layered(layers);
    for t in model
        .tensors
        .iter_mut()
        .filter(|t| t.role == Role::RoutedExperts)
    {
        t.file_bytes = stack;
    }
    let node = match pool {
        None => Node {
            host: workstation::host(),
            unified: None,
        },
        Some(bytes) => Node {
            host: Host {
                usable_bytes: bytes,
                reserves: Vec::new(),
            },
            unified: Some(UnifiedPool { bytes }),
        },
    };
    let machine = Machine {
        cards: vec![bytes_card("stage", HEAD, 0..layers)],
        tiers: Vec::new(),
        nodes: vec![node],
    };
    (model, machine)
}

/// The host-bound violations a plan carries.
fn host_over(plan: &Plan<'_>) -> Vec<String> {
    plan.violations()
        .iter()
        .filter(|v| matches!(v, Violation::HostOver { .. }))
        .map(ToString::to_string)
        .collect()
}

/// A host leg past this workstation's room (320 GB against 263,376,371,712 B)
/// on a machine that reads 512 GiB: the plan's room is that reading less the
/// drift, the leg stays on the host whole, the plan holds no `HostOver`, and
/// its headroom is what the room leaves past the plan's terms. A plan made
/// with no room is held to the nominal figure, which this leg is past.
#[test]
fn a_reading_past_the_figure_is_the_plan_room() {
    let (model, machine) = leg(4, 80_000_000_000, None);
    let mut plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("the plan");
    assert_eq!(plan.host.expert_bytes, 320_000_000_000);
    assert!(plan.host.expert_bytes > workstation::PLAN_ROOM);
    assert_eq!(
        host_over(&plan).len(),
        1,
        "the figure holds this leg to 263 GB"
    );

    let reading = (512 * GIB, workstation::HostRead::MemAvailable);
    let planned = plan.plan_at(&reading);
    assert_eq!(planned, 512 * GIB - workstation::HOST_DRIFT);
    assert_eq!(
        plan.host.room,
        Some((reading.0, (planned, workstation::HostRead::MemAvailable)))
    );
    expert_nvme_tier(&mut plan, planned, None).expect("the split");
    assert_eq!(plan.host.nvme_expert_bytes, 0, "the reading holds the leg");
    assert_eq!(host_over(&plan), Vec::<String>::new());
    let terms = workstation::HostNeed::of(&plan, 0).plan_bytes();
    assert_eq!(
        plan.host.headroom_bytes,
        i128::from(planned) - i128::from(terms)
    );
    assert!(plan.host.headroom_bytes > 0);
}

/// A reading past the figure and short of the leg still pages it: 300 GB of
/// room against 320 GB moves whole experts to the NVMe tier, and the plan holds
/// no `HostOver`: the room it was made at is the bound.
#[test]
fn a_reading_past_the_figure_and_short_of_the_leg_still_pages() {
    let (model, machine) = leg(4, 80_000_000_000, None);
    let mut plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("the plan");
    let reading = (
        300_000_000_000 + workstation::HOST_DRIFT,
        workstation::HostRead::MemAvailable,
    );
    let planned = plan.plan_at(&reading);
    assert_eq!(planned, 300_000_000_000);
    expert_nvme_tier(&mut plan, planned, None).expect("the split");
    assert!(plan.host.nvme_expert_bytes > 0, "the room is under the leg");
    assert_eq!(host_over(&plan), Vec::<String>::new());
}

/// A room the caller gave past the figure is the plan room as given, up to
/// any size, and the split reads that room: the leg and two layers' reads
/// (480 GB) fit 500 GB whole, and 400 GB pages less than the figure's 263 GB.
#[test]
fn a_given_room_past_the_figure_is_the_plan_room() {
    let (model, machine) = leg(4, 80_000_000_000, None);
    let paged = |room: u64| {
        let mut plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("the plan");
        expert_nvme_tier(&mut plan, room, None).expect("the split");
        plan.host.nvme_expert_bytes
    };
    for given in [
        workstation::PLAN_ROOM + 1,
        400_000_000_000,
        500_000_000_000,
        u64::MAX,
    ] {
        let mut plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("the plan");
        let room = (given, workstation::HostRead::Given);
        assert_eq!(plan.plan_at(&room), given, "given {given}");
        assert_eq!(
            plan.host.room,
            Some((given, (given, workstation::HostRead::Given)))
        );
        let terms = workstation::HostNeed::of(&plan, 0).plan_bytes();
        assert_eq!(
            plan.host.headroom_bytes,
            i128::from(given) - i128::from(terms),
            "given {given}"
        );
        expert_nvme_tier(&mut plan, given, None).expect("the split");
        assert_eq!(host_over(&plan), Vec::<String>::new(), "given {given}");
    }
    assert!(paged(400_000_000_000) < paged(workstation::PLAN_ROOM));
    assert_eq!(paged(500_000_000_000), 0);
    assert_eq!(paged(u64::MAX), 0);
}

/// A room given past what this machine has (a plan of another host: a larger
/// workstation, a records refresh at a pinned room) plans as the same room
/// given within it does: the same plan room, headroom, paging and violations,
/// the only difference the reading that names it, which the plan keeps so its
/// lines say the room is above this machine's capacity; the refusal is the
/// load's ([`workstation::load_room`]). The machines are 200 GB, past the
/// leg's split floor (240 GB) in rooms past the nominal figure, and 100 GB
/// with rooms under it, where the headroom is the stated room's own only
/// because the room was stated.
#[test]
fn a_room_given_past_the_machine_plans_as_given_and_keeps_its_label() {
    let violations = |p: &Plan<'_>| {
        p.violations()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    };
    let cases: [(u64, u64, &[u64]); 2] = [
        (
            80_000_000_000,
            200_000_000_000,
            &[250_000_000_000, 400_000_000_000, 500_000_000_000, u64::MAX],
        ),
        (
            30_000_000_000,
            100_000_000_000,
            &[150_000_000_000, 200_000_000_000],
        ),
    ];
    for (stack, bytes, givens) in cases {
        let (model, machine) = leg(4, stack, None);
        let capacity = workstation::HostCapacity {
            bytes,
            from: "MemTotal in /proc/meminfo".to_owned(),
        };
        for &given in givens {
            let above = workstation::given_room(given, &capacity);
            assert!(matches!(above.1, workstation::HostRead::GivenAbove { .. }));
            let mut as_given = plan_with(&model, &machine, 4096, &NoKv, None).expect("the plan");
            let mut labelled = as_given.clone();
            let want = as_given.plan_at(&(given, workstation::HostRead::Given));
            assert_eq!(labelled.plan_at(&above), want, "given {given}");
            expert_nvme_tier(&mut as_given, want, None).expect("the split");
            expert_nvme_tier(&mut labelled, want, None).expect("the split");
            assert_eq!(
                labelled.host.nvme_expert_bytes, as_given.host.nvme_expert_bytes,
                "given {given}"
            );
            assert_eq!(
                labelled.host.headroom_bytes, as_given.host.headroom_bytes,
                "given {given}"
            );
            assert_eq!(
                violations(&labelled),
                violations(&as_given),
                "given {given}"
            );
            let (_, (planned, by)) = labelled.host.room.clone().expect("the room");
            assert_eq!(planned, want);
            assert_eq!(by, above.1, "the plan keeps what the room is");
            assert!(
                by.to_string()
                    .contains("given above this machine's capacity")
            );
        }
    }
}

/// A room at or under the figure plans as it always did: the room is the
/// reading's plan room or the given bytes, what decided it is the reading
/// itself, the headroom of a read room stays the figure's and a given room's
/// is its own, and the violations are those of a plan made with no room.
#[test]
fn a_room_at_or_under_the_figure_plans_as_before() {
    let figure = workstation::PLAN_ROOM;
    let drift = workstation::HOST_DRIFT;
    let (model, machine) = leg(4, 30_000_000_000, None);
    let base = plan_with(&model, &machine, 4096, &NoKv, None).expect("the plan");
    let terms = i128::from(workstation::HostNeed::of(&base, 0).plan_bytes());
    let at_figure = i128::from(figure) - terms;
    assert_eq!(base.host.headroom_bytes, at_figure);
    let violations = |p: &Plan<'_>| {
        p.violations()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    };
    let want = violations(&base);

    let reads = [1, 100_000_000_000, figure + drift - 1, figure + drift];
    for asked in reads {
        for read in [
            workstation::HostRead::MemAvailable,
            workstation::HostRead::VmStats,
            workstation::HostRead::Cgroup { path: "/x".into() },
        ] {
            let mut plan = base.clone();
            let planned = plan.plan_at(&(asked, read.clone()));
            assert_eq!(planned, asked.saturating_sub(drift), "{asked} {read:?}");
            assert_eq!(plan.host.room, Some((asked, (planned, read.clone()))));
            assert_eq!(plan.host.headroom_bytes, at_figure, "{asked} {read:?}");
            assert_eq!(violations(&plan), want, "{asked} {read:?}");
        }
    }
    for given in [0, 1, 100_000_000_000, figure] {
        let mut plan = base.clone();
        let planned = plan.plan_at(&(given, workstation::HostRead::Given));
        assert_eq!(planned, given);
        assert_eq!(
            plan.host.room,
            Some((given, (given, workstation::HostRead::Given)))
        );
        assert_eq!(plan.host.headroom_bytes, i128::from(given) - terms);
        assert_eq!(violations(&plan), want, "given {given}");
    }
}

/// A unified machine's room is still its pool: a room past the pool plans
/// at the pool, named by the machine's figure, and a room under it plans as
/// asked. The pool is the machine's own reading there, not a constant.
#[test]
fn a_unified_pool_still_caps_the_room() {
    const POOL: u64 = 100_000_000_000;
    let (model, machine) = leg(4, 1_000_000, Some(POOL));
    let base = plan_with(&model, &machine, 4096, &NoKv, None).expect("the plan");
    let want = base
        .violations()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let above = workstation::HostRead::GivenAbove {
        capacity: POOL / 2,
        from: "MemTotal in /proc/meminfo".to_owned(),
    };
    for (asked, read) in [
        (2 * POOL, workstation::HostRead::Given),
        (2 * POOL, above),
        (
            POOL + workstation::HOST_DRIFT + 1,
            workstation::HostRead::MemAvailable,
        ),
        (u64::MAX, workstation::HostRead::Given),
    ] {
        let mut plan = base.clone();
        assert_eq!(plan.plan_at(&(asked, read)), POOL);
        assert_eq!(
            plan.host.room,
            Some((asked, (POOL, workstation::HostRead::Machine)))
        );
        let now = plan
            .violations()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        assert_eq!(now, want);
    }
    let mut plan = base.clone();
    assert_eq!(
        plan.plan_at(&(POOL / 2, workstation::HostRead::Given)),
        POOL / 2
    );
    assert_eq!(
        plan.host.room,
        Some((POOL / 2, (POOL / 2, workstation::HostRead::Given)))
    );
}

/// A plan made with no room is checked at the nominal figure, and the
/// refusal says the figure is the fallback: nothing of this machine was read.
#[test]
fn the_fallback_figure_names_itself() {
    let (model, machine) = leg(4, 80_000_000_000, None);
    let plan = plan_with(&model, &machine, 4096, &NoKv, None).expect("the plan");
    assert_eq!(plan.host.room, None);
    let short = workstation::HostNeed::of(&plan, 0)
        .check_planned(&plan)
        .expect_err("320 GB passes the nominal figure");
    assert_eq!(*short.read, workstation::HostRead::Machine);
    assert_eq!(short.available, workstation::PLAN_ROOM);
    let text = short.to_string();
    assert!(
        text.contains("nominal") && text.contains("nothing was read"),
        "{text}"
    );
    assert_eq!(workstation::HostRead::Machine.word(), "machine");
}
