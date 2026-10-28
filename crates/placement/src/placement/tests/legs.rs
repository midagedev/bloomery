use super::*;

#[test]
fn a_pool_reserve_counts_local_cards_without_overflow() {
    let model = layered(4);
    for ranks in [1, 2, 4] {
        let machine = Machine {
            cards: (0..ranks)
                .map(|rank| Card {
                    head: rank + 1 == ranks,
                    token_embedding: rank == 0,
                    ..bytes_card(
                        &format!("pool-{rank}"),
                        1 << 30,
                        rank * 4 / ranks..(rank + 1) * 4 / ranks,
                    )
                })
                .collect(),
            tiers: Vec::new(),
            nodes: vec![Node {
                host: host(),
                unified: Some(UnifiedPool { bytes: 1 << 40 }),
            }],
        };
        let mut plan = plan_with(&model, &machine, 4096, &NoKv, None).unwrap();
        let before = plan.host.pool_card_bytes;
        let headroom = plan.host.headroom_bytes;
        let n = ranks as u64;
        let reserve = (u64::MAX - before) / n;
        plan.reserve_beside(reserve)
            .expect("the largest reserve fits u64");
        assert_eq!(plan.host.pool_card_bytes, before + reserve * n);
        assert_eq!(plan.host.headroom_bytes, headroom - i128::from(reserve * n));
        let before = (plan.host.pool_card_bytes, plan.host.headroom_bytes);
        for reserve in [1, u64::MAX, (u64::MAX / n).saturating_add(1)] {
            assert!(plan.reserve_beside(reserve).is_err());
            assert_eq!(
                (plan.host.pool_card_bytes, plan.host.headroom_bytes),
                before
            );
        }
    }
}

/// The census reads a card on its primary context, so an idle card's
/// reading is its usable bytes less the context's own creation cost,
/// which the card's context term counts again: the plan counts the
/// context once. An idle reading plans what the census-free card plans;
/// a reading another process holds 12 experts' bytes of plans what a
/// card budget that much below usable plans; and the free floor takes the
/// context less that cost — a trunk at the reading plans, one byte past it
/// is refused.
#[test]
fn a_census_reading_counts_the_context_once() {
    let model = layered(3);
    let context = workstation::CONTEXT;
    let own = workstation::CONTEXT_SELF;
    let usable = HEAD + 24 * EXPERT + context;
    let machine = |free: Option<u64>| Machine {
        cards: vec![Card {
            context_bytes: context,
            free_bytes: free,
            ..bytes_card("card", usable, 0..3)
        }],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let (none, idle, busy, budgeted) = (
        machine(None),
        machine(Some(usable - own)),
        machine(Some(usable - own - 12 * EXPERT)),
        machine(None),
    );
    let census_free = plan_with(&model, &none, 4096, &NoKv, None).expect("plans");
    assert_eq!(census_free.n_l.iter().sum::<u64>(), 24);
    let idle = plan_with(&model, &idle, 4096, &NoKv, None).expect("an idle card plans");
    assert_eq!(
        idle.n_l, census_free.n_l,
        "an idle reading plans as no reading"
    );
    assert_eq!(idle.usable_bytes(&idle.machine.cards[0]), usable);
    assert!(idle.violations().is_empty());
    let busy = plan_with(&model, &busy, 4096, &NoKv, None).expect("a held card plans");
    let budget = plan_with(&model, &budgeted, 4096, &NoKv, Some(usable - 12 * EXPERT))
        .expect("plans under a budget");
    assert_eq!(busy.n_l.iter().sum::<u64>(), 12);
    assert_eq!(busy.n_l, budget.n_l, "a held reading plans as its budget");
    assert!(busy.violations().is_empty());
    let at = HEAD + context - own;
    plan_with(&model, &machine(Some(at)), 4096, &NoKv, None).expect("a trunk at the reading");
    match plan_with(&model, &machine(Some(at - 1)), 4096, &NoKv, None) {
        Err(PlacementError::CardFreeFloor(terms)) => {
            assert_eq!(
                (terms.free, terms.floor, terms.context),
                (at - 1, at, context - own)
            );
        }
        Err(e) => panic!("{e}, not CardFreeFloor"),
        Ok(_) => panic!("a trunk past the reading was planned"),
    }
}

/// A card whose dense trunk alone passes what its device had free is
/// refused by name, every term in the message — the card, the free and
/// the usable bytes, the floor's terms and the holder the census named.
#[test]
fn a_trunk_past_the_free_bytes_is_refused_by_name() {
    let model = layered(3);
    let usable = HEAD + 24 * EXPERT;
    let machine = |free: u64| Machine {
        cards: vec![Card {
            free_bytes: Some(free),
            held_by: Some("pid 7 (977 MiB)"),
            ..bytes_card("held", usable, 0..3)
        }],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let held = machine(HEAD - 1);
    match plan_with(&model, &held, 4096, &NoKv, None) {
        Err(PlacementError::CardFreeFloor(terms)) => {
            assert_eq!((terms.free, terms.floor), (HEAD - 1, HEAD));
            let text = terms.to_string();
            for part in [
                "card held: the device had 4351 B free of its usable",
                "(held by pid 7 (977 MiB))",
                "needs 4352 B = dense 4352 B (the allocator's granules)",
                "+ KV 0 B + context 0 B + scratch 0 B + reserves 0 B + margin 0 B",
                "past the 4351 B by 1 B",
                "free the card, or plan a placement on a card with room",
            ] {
                assert!(text.contains(part), "{part:?} in {text}");
            }
        }
        Err(e) => panic!("{e}, not CardFreeFloor"),
        Ok(_) => panic!("a trunk past the free bytes was planned"),
    }
    // A floor within the free bytes never refuses; the holder list is
    // absent when the census named none.
    let mut quiet = machine(HEAD);
    quiet.cards[0].held_by = None;
    plan_with(&model, &quiet, 4096, &NoKv, None).expect("a trunk at the free bytes");
}

/// A reserve entered by [`Machine::reserve`] leaves the card before any
/// expert: the stage card of three layers holds all 24 experts with none,
/// 19 under five experts' bytes, none under the whole room. Under a card
/// budget, the reserve and the dense tensors are the floor: one byte under
/// it is the budget floor's refusal, at it the card holds no expert, and
/// one expert's bytes more hold one.
#[test]
fn a_reserve_leaves_the_card_fewer_experts() {
    let model = layered(3);
    let machine = |reserve: Option<u64>| {
        let mut machine = Machine {
            cards: vec![bytes_card("stage", HEAD + 24 * EXPERT, 0..3)],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        if let Some(bytes) = reserve {
            machine
                .reserve(Reserve {
                    on: Device::Card(0),
                    name: "draft",
                    bytes,
                })
                .expect("the stage card is card 0");
        }
        machine
    };
    let experts = |reserve: Option<u64>, budget: Option<u64>| {
        plan_with(&model, &machine(reserve), 4096, &NoKv, budget).map(|plan| plan.cards[0].experts)
    };
    assert_eq!(experts(None, None).expect("plans"), 24);
    assert_eq!(experts(Some(0), None).expect("plans"), 24);
    assert_eq!(experts(Some(5 * EXPERT), None).expect("plans"), 19);
    assert_eq!(experts(Some(24 * EXPERT), None).expect("plans"), 0);
    let reserve = 10 * EXPERT;
    let floor = HEAD + reserve;
    match experts(Some(reserve), Some(floor - 1)) {
        Err(PlacementError::CardBudgetFloor {
            floor: named,
            reserves,
            ..
        }) => assert_eq!((named, reserves), (floor, reserve)),
        other => panic!("{other:?}, not the budget floor's refusal"),
    }
    assert_eq!(experts(Some(reserve), Some(floor)).expect("plans"), 0);
    assert_eq!(
        experts(Some(reserve), Some(floor + EXPERT - 1)).expect("plans"),
        0
    );
    assert_eq!(
        experts(Some(reserve), Some(floor + EXPERT)).expect("plans"),
        1
    );
    assert_eq!(
        experts(None, Some(floor + EXPERT)).expect("plans"),
        11,
        "the same budget without the reserve holds what the reserve took"
    );
}

/// [`Machine::reserve`] takes a card by its place in
/// [`Machine::all_cards`]: the stage cards first, then the tiers; the host
/// by its own device. A place past the list, the NVMe tier and nowhere are
/// refused by name and set nothing aside.
#[test]
fn a_reserve_is_addressed_by_the_whole_device_list() {
    let mut machine = Machine {
        cards: vec![bytes_card("stage", 1_000, 0..3)],
        tiers: vec![card("tier", 0..0, false, false)],
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let reserve = |on, name, bytes| Reserve { on, name, bytes };
    for r in [
        reserve(Device::Card(0), "on the stage", 1),
        reserve(Device::Card(1), "on the tier", 2),
        reserve(Device::Host, "on the host", 3),
    ] {
        machine.reserve(r).expect("a device of the plan");
    }
    assert_eq!(machine.cards[0].reserves, [("on the stage".to_string(), 1)]);
    assert_eq!(machine.tiers[0].reserves, [("on the tier".to_string(), 2)]);
    assert!(
        machine
            .host()
            .reserves
            .contains(&("on the host".to_string(), 3))
    );
    let before = machine.clone();
    for (on, shown) in [
        (Device::Card(2), "Card(2)"),
        (Device::Nvme, "Nvme"),
        (Device::Unused, "Unused"),
    ] {
        match machine.reserve(reserve(on, "off the list", 4)) {
            Err(e @ PlacementError::ReserveOffPlan { cards: 2, .. }) => assert_eq!(
                e.to_string(),
                format!(
                    "the off the list reserve is on {shown}, outside the plan: a reserve \
                     sits on one of the plan's 2 cards, stage first, or on the host"
                )
            ),
            other => panic!("{other:?}, not ReserveOffPlan"),
        }
    }
    assert_eq!(machine, before, "a refused reserve sets nothing aside");
}

/// A reserve-only tier member ([`Card::reserve_only`]) holds its reserve and
/// no expert: the stage card's experts are what the plan without the member
/// holds, the member's counts are 0 on every layer, and it is the expert
/// tiers' count that excludes it. One with no reserve is the idle tier's
/// refusal; one whose reserve passes its room is the budget floor's, under
/// a card budget, as for any card.
#[test]
fn a_reserve_only_member_holds_a_reserve_and_no_expert() {
    let model = layered(3);
    let member = |reserve: Option<u64>, tiers: usize| {
        let mut machine = Machine {
            cards: vec![bytes_card("stage", HEAD + 5 * EXPERT, 0..3)],
            tiers: (0..tiers)
                .map(|i| Card {
                    reserve_only: true,
                    ..bytes_card(&format!("holder{i}"), 10 * EXPERT, 0..0)
                })
                .collect(),
            nodes: vec![Node {
                host: host(),
                unified: None,
            }],
        };
        if let Some(bytes) = reserve {
            machine
                .reserve(Reserve {
                    on: Device::Card(1),
                    name: "draft",
                    bytes,
                })
                .expect("the member is card 1");
        }
        machine
    };
    let stage = member(None, 0);
    let alone = plan_with(&model, &stage, 4096, &NoKv, None).expect("plans");
    let held = member(Some(4 * EXPERT), 1);
    let plan = plan_with(&model, &held, 4096, &NoKv, None).expect("plans");
    assert_eq!(plan.n_l, alone.n_l, "the stage card's prefix does not move");
    assert_eq!(plan.tier_n_l, vec![vec![0, 0, 0]]);
    assert_eq!(plan.cards[1].experts, 0);
    assert_eq!(plan.host.experts, alone.host.experts);
    assert_eq!(held.expert_tiers().count(), 0);
    assert_eq!(held.tiers.len(), 1);
    match plan_with(&model, &member(None, 1), 4096, &NoKv, None) {
        Err(PlacementError::IdleTier { card, tier: 0 }) => assert_eq!(card, "holder0"),
        other => panic!("{other:?}, not IdleTier"),
    }
    let over = member(Some(11 * EXPERT), 1);
    assert!(
        !plan_with(&model, &over, 4096, &NoKv, None)
            .expect("plans")
            .violations()
            .is_empty(),
        "a reserve past the member's room is a violation"
    );
    match plan_with(&model, &over, 4096, &NoKv, Some(10 * EXPERT)) {
        Err(PlacementError::CardBudgetFloor { card, .. }) => assert_eq!(card, "holder0"),
        other => panic!("{other:?}, not the budget floor's refusal"),
    }
    let host_routed = plan_host_routed(&model, &held, 4096, &NoKv, &PlanLevers::default());
    assert!(host_routed.is_ok(), "{host_routed:?}: no expert tier card");
    assert_eq!(
        host_routed.expect("plans").host.experts,
        24,
        "every routed expert on the host"
    );
}

/// A per-card budget caps a card by its own entry, found by the card's
/// device ordinal: the expert rule and [`Plan::usable_bytes`] read it, a
/// card the list does not name plans with its own bytes, an entry under
/// the card's floor is refused naming the card, and a census-free card
/// under a list is refused by name ([`card_budget::check`]).
#[test]
fn a_per_card_budget_caps_each_card_by_its_own_entry() {
    let model = layered(3);
    let full = HEAD + 24 * EXPERT + 100_000;
    let half = HEAD + 12 * EXPERT;
    let on = |device: Option<u32>| Machine {
        cards: vec![Card {
            device: device.map(|ordinal| devices::DeviceId {
                ordinal,
                uuid: [7; 16],
            }),
            ..bytes_card("stage", full, 0..3)
        }],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let list = |entries: &[(u32, u64)]| PlanLevers {
        card_budget_bytes: card_budget::CardBudget::Per(entries.to_vec()),
    };
    let named = on(Some(1));
    let p = plan(&model, &named, 4096, &NoKv, &list(&[(0, 1), (1, half)])).expect("plans");
    assert_eq!(
        (p.cards[0].experts, p.usable_bytes(&named.cards[0])),
        (12, half)
    );
    let other = on(Some(0));
    let p = plan(&model, &other, 4096, &NoKv, &list(&[(1, half)])).expect("plans");
    assert_eq!(
        (p.cards[0].experts, p.usable_bytes(&other.cards[0])),
        (24, full)
    );
    match plan(&model, &named, 4096, &NoKv, &list(&[(1, HEAD - 1)])) {
        Err(PlacementError::CardBudgetFloor { card, budget, .. }) => {
            assert_eq!((card.as_str(), budget), ("stage", HEAD - 1));
        }
        other => panic!("{:?}, not the stage's budget floor", other.map(|_| ())),
    }
    let census_free = on(None);
    match plan(&model, &census_free, 4096, &NoKv, &list(&[(0, half)])) {
        Err(PlacementError::CardBudgetCard { card, .. }) => assert_eq!(card, "stage"),
        other => panic!("{:?}, not a census-free card's refusal", other.map(|_| ())),
    }
}

/// A budget by shares becomes a budget per card from the plan's own terms: each named
/// card plans with its floor and its part of the routed experts, in the proportion of the
/// shares and as many in all as the tightest card's room for its share lets, the host tier
/// taking the rest, and the plan's record states the bytes. 3:1 over a stage and a tier of
/// 24 experts' room each holds all 24 as 18 and 6 (the default fill leaves the tier idle);
/// 1:1 with a tier of 2 experts' room holds 4 as 2 and 2 and leaves 20 to the host. A floor
/// past the card, a plan that places no routed expert on a card, shares that name no card
/// of the plan and shares of 0 are refused naming the cause.
#[test]
fn shares_become_a_budget_per_card_from_the_plans_terms() {
    let model = layered(3);
    let device = |ordinal: u32| {
        Some(devices::DeviceId {
            ordinal,
            uuid: [u8::try_from(ordinal).expect("small") + 7; 16],
        })
    };
    let on = |stage: u64, tier: u64| Machine {
        cards: vec![Card {
            device: device(0),
            ..bytes_card("stage", stage, 0..3)
        }],
        tiers: vec![Card {
            device: device(1),
            ..bytes_card("tier", tier, 0..0)
        }],
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let shares = |entries: &[(u32, u64)]| PlanLevers {
        card_budget_bytes: card_budget::CardBudget::Shares(entries.to_vec()),
    };
    let roomy = on(HEAD + 24 * EXPERT, 24 * EXPERT);
    let p = plan(&model, &roomy, 4096, &NoKv, &shares(&[(0, 3), (1, 1)])).expect("plans");
    assert_eq!(p.n_l, vec![6, 6, 6]);
    assert_eq!(p.tier_n_l, vec![vec![2, 2, 2]]);
    assert_eq!(p.host.experts, 0);
    assert_eq!(
        p.card_budget,
        card_budget::CardBudget::Per(vec![(0, HEAD + 18 * EXPERT), (1, 6 * EXPERT)])
    );
    assert!(matches!(
        plan_with(&model, &roomy, 4096, &NoKv, None),
        Err(PlacementError::IdleTier { .. })
    ));
    let tight = on(HEAD + 24 * EXPERT, 2 * EXPERT);
    let p = plan(&model, &tight, 4096, &NoKv, &shares(&[(0, 1), (1, 1)])).expect("plans");
    assert_eq!(p.n_l, vec![1, 1, 0]);
    assert_eq!(p.tier_n_l, vec![vec![1, 0, 1]]);
    assert_eq!(p.host.experts, 20);
    assert_eq!(
        p.card_budget,
        card_budget::CardBudget::Per(vec![(0, HEAD + 2 * EXPERT), (1, 2 * EXPERT)])
    );
    // Every term of a card's floor is in its part: context, scratch, reserves, margin and
    // what the caller holds beside the plan.
    let held = |c: &Card| Card {
        context_bytes: 1_000,
        scratch_bytes: 2_000,
        margin_bytes: 3_000,
        reserves: vec![("draft".to_string(), 4_000)],
        ..c.clone()
    };
    let slack = on(HEAD + 24 * EXPERT + 100_000, 24 * EXPERT + 100_000);
    let held_machine = Machine {
        cards: vec![held(&slack.cards[0])],
        tiers: vec![held(&slack.tiers[0])],
        ..slack.clone()
    };
    let beside = 5_000;
    let p = plan_routed_reserving(
        &model,
        &held_machine,
        4096,
        &NoKv,
        &shares(&[(0, 3), (1, 1)]),
        RoutedRule::PerType(CardFormat::of_routed),
        beside,
    )
    .expect("plans");
    assert_eq!(p.n_l, vec![6, 6, 6]);
    let past = 1_000 + 2_000 + 3_000 + 4_000 + beside;
    assert_eq!(
        p.card_budget,
        card_budget::CardBudget::Per(vec![(0, HEAD + past + 18 * EXPERT), (1, past + 6 * EXPERT)])
    );
    let refused =
        |machine: &Machine, levers: &PlanLevers| match plan(&model, machine, 4096, &NoKv, levers) {
            Err(PlacementError::CardShares { shares, why }) => (shares, why),
            other => panic!("{:?}, not a shares refusal", other.map(|_| ())),
        };
    let (named, why) = refused(&on(HEAD - 1, 24 * EXPERT), &shares(&[(0, 1), (1, 1)]));
    assert_eq!(named, "CUDA0=1x,CUDA1=1x");
    assert!(
        why.starts_with("card stage: its floor 4352 B = dense uploads 4352 B")
            && why.contains("passes the 4351 B it may plan with"),
        "{why}"
    );
    let (_, why) = refused(&roomy, &shares(&[(5, 1)]));
    assert!(
        why.contains("names no card of this plan that takes experts")
            && why.contains("stage, tier"),
        "{why}"
    );
    let (_, why) = refused(&roomy, &shares(&[(0, 0), (5, 1)]));
    assert!(why.contains("has a share above 0"), "{why}");
    let stage_only = Machine {
        tiers: Vec::new(),
        ..roomy.clone()
    };
    match plan_host_routed(&model, &stage_only, 4096, &NoKv, &shares(&[(0, 1)])) {
        Err(PlacementError::CardShares { why, .. }) => {
            assert!(why.contains("places no routed expert on a card"), "{why}");
        }
        other => panic!("{:?}, not a shares refusal", other.map(|_| ())),
    }
}

/// A plan whose caller keeps `BESIDE` bytes of the card beside it
/// ([`plan_routed_reserving`]) names a budget floor that counts them, as
/// the expert rule's budget does: the floor is the least budget that
/// plans, so the plan at the floor leaves the margin and the reserve both
/// unspent (the caller's bound on the sum), and one byte under it is
/// refused naming that floor, whatever the card's own reserves, context,
/// scratch and margin are.
#[test]
fn the_budget_floor_counts_what_the_caller_holds_beside_the_plan() {
    const CONTEXT: u64 = 1_000;
    const SCRATCH: u64 = 2_000;
    const MARGIN: u64 = 3_000;
    const CARD_RESERVE: u64 = 4_000;
    const BESIDE: u64 = 5_000;
    let model = layered(3);
    let machine = Machine {
        cards: vec![Card {
            context_bytes: CONTEXT,
            scratch_bytes: SCRATCH,
            margin_bytes: MARGIN,
            reserves: vec![("draft".to_string(), CARD_RESERVE)],
            ..bytes_card("stage", HEAD + 24 * EXPERT + 100_000, 0..3)
        }],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let plan_at = |budget: u64, beside: u64| {
        let levers = PlanLevers {
            card_budget_bytes: Some(budget).into(),
        };
        plan_routed_reserving(
            &model,
            &machine,
            4096,
            &NoKv,
            &levers,
            RoutedRule::PerType(CardFormat::of_routed),
            beside,
        )
    };
    let floor_of = |beside: u64| match plan_at(1, beside) {
        Err(PlacementError::CardBudgetFloor { floor, .. }) => floor,
        Err(e) => panic!("{e}, not CardBudgetFloor"),
        Ok(_) => panic!("a budget of 1 B planned"),
    };
    let trunk = HEAD + CONTEXT + SCRATCH + CARD_RESERVE + MARGIN;
    assert_eq!(floor_of(0), trunk, "a plan with nothing beside it");
    let floor = floor_of(BESIDE);
    let mut bad = Vec::new();
    if floor != trunk + BESIDE {
        bad.push(format!(
            "the floor beside {BESIDE} B is {floor} B, not the trunk's {trunk} B plus them"
        ));
    }
    match plan_at(floor, BESIDE) {
        Ok(plan) => {
            let card = &plan.cards[0];
            let left = card.headroom_bytes - i128::from(MARGIN);
            if left < i128::from(BESIDE) {
                bad.push(format!(
                    "at the floor {floor} B the plan leaves {left} B past the margin, under \
                     the {BESIDE} B beside it"
                ));
            }
            if card.experts != 0 || !plan.violations().is_empty() {
                bad.push(format!(
                    "at the floor {floor} B: {} experts, violations {:?}",
                    card.experts,
                    plan.violations()
                ));
            }
        }
        Err(e) => bad.push(format!("at the floor {floor} B: {e}")),
    }
    match plan_at(floor - 1, BESIDE) {
        Err(e @ PlacementError::CardBudgetFloor { .. }) => {
            let want = format!(
                "budget {} B is below its floor {floor} B = dense {HEAD} B (the allocator's \
                 granules) + KV 0 B + context {CONTEXT} B + scratch {SCRATCH} B + reserves \
                 {CARD_RESERVE} B + margin {MARGIN} B + beside {BESIDE} B (what the caller \
                 holds beside the plan)",
                floor - 1
            );
            if !e.to_string().contains(&want) {
                bad.push(format!("one byte under the floor: {e}, not {want:?}"));
            }
        }
        Err(e) => bad.push(format!("one byte under the floor {floor} B: {e}")),
        Ok(_) => bad.push(format!("one byte under the floor {floor} B planned")),
    }
    // A reserve that takes the sum past u64 bytes is refused with the floor
    // printed as u64::MAX, never wrapped.
    match plan_at(u64::MAX, u64::MAX) {
        Err(PlacementError::CardBudgetFloor { floor, beside, .. })
            if (floor, beside) == (u64::MAX, u64::MAX) => {}
        other => bad.push(format!("a reserve of u64::MAX: {:?}", other.map(|p| p.n_l))),
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// The free floor counts what the caller keeps beside the plan the same
/// way ([`the_budget_floor_counts_what_the_caller_holds_beside_the_plan`]):
/// a card whose device had its floor free plans with the margin and the
/// reserve both left, and one byte under it is refused naming that floor.
#[test]
fn the_free_floor_counts_what_the_caller_holds_beside_the_plan() {
    const MARGIN: u64 = 3_000;
    const BESIDE: u64 = 5_000;
    let model = layered(3);
    let context = workstation::CONTEXT;
    let own = workstation::CONTEXT_SELF;
    let usable = HEAD + 24 * EXPERT + context + 100_000;
    let machine = |free: u64| Machine {
        cards: vec![Card {
            context_bytes: context,
            margin_bytes: MARGIN,
            free_bytes: Some(free),
            ..bytes_card("held", usable, 0..3)
        }],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let plan_at = |free: u64, beside: u64| {
        plan_routed_reserving(
            &model,
            &machine(free),
            4096,
            &NoKv,
            &PlanLevers::default(),
            RoutedRule::PerType(CardFormat::of_routed),
            beside,
        )
        .map(|plan| (plan.cards[0].headroom_bytes, plan.cards[0].experts))
    };
    let floor_of = |beside: u64| match plan_at(1, beside) {
        Err(PlacementError::CardFreeFloor(terms)) => terms.floor,
        Err(e) => panic!("{e}, not CardFreeFloor"),
        Ok(_) => panic!("a reading of 1 B planned"),
    };
    let trunk = HEAD + context - own + MARGIN;
    assert_eq!(floor_of(0), trunk, "a plan with nothing beside it");
    let floor = floor_of(BESIDE);
    let mut bad = Vec::new();
    if floor != trunk + BESIDE {
        bad.push(format!(
            "the floor beside {BESIDE} B is {floor} B, not the trunk's {trunk} B plus them"
        ));
    }
    match plan_at(floor, BESIDE) {
        Ok((headroom, experts)) => {
            let left = headroom - i128::from(MARGIN);
            if left < i128::from(BESIDE) || experts != 0 {
                bad.push(format!(
                    "at the floor {floor} B the plan leaves {left} B past the margin, under \
                     the {BESIDE} B beside it, and keeps {experts} experts"
                ));
            }
        }
        Err(e) => bad.push(format!("at the floor {floor} B: {e}")),
    }
    match plan_at(floor - 1, BESIDE) {
        Err(PlacementError::CardFreeFloor(terms)) => {
            if (terms.free, terms.floor, terms.beside) != (floor - 1, floor, BESIDE) {
                bad.push(format!("one byte under the floor {floor} B: {terms:?}"));
            }
            let want = format!(
                "+ margin {MARGIN} B + beside {BESIDE} B (what the caller holds beside the plan)"
            );
            if !terms.to_string().contains(&want) {
                bad.push(format!("one byte under the floor: {terms}, not {want:?}"));
            }
        }
        Err(e) => bad.push(format!("one byte under the floor {floor} B: {e}")),
        Ok(_) => bad.push(format!("one byte under the floor {floor} B planned")),
    }
    assert!(bad.is_empty(), "{bad:#?}");
}

/// The whole fit counts what the whole load asks for: every tensor's
/// buffers through the card's allocator (the embedding's two q8_0
/// planes, each layer's stack, the head's planes: 22,528 B of buffers,
/// 36,864 B of 4 KiB granules — the planes' small halves share one), the
/// cache, the context, the scratch, and the program's arena with the
/// reserve its load keeps free past it. A card whose free reading holds
/// all of that but one byte of the arena does not fit; at the need it
/// fits. The budget is the reading as taken — the context's own creation
/// not added back, as the plan's cap adds it; a program with no checked
/// arena keeps the card's margin in its place; and a tensor no card
/// format loads is refused by name.
#[test]
fn the_whole_fit_counts_the_arena_and_the_granules() {
    const GRANULE: u64 = 4096;
    const KV: u64 = 3_000;
    const ARENA: u64 = 50_000;
    const RESERVE: u64 = 7_000;
    const MARGIN: u64 = 90_000;
    let model = layered(3);
    let context = workstation::CONTEXT;
    let scratch = workstation::SCRATCH;
    let load = WholeLoad::Checked {
        arena: ARENA,
        reserve: RESERVE,
    };
    let at = |free: u64| Card {
        context_bytes: context,
        scratch_bytes: scratch,
        margin_bytes: MARGIN,
        granule_bytes: NonZeroU64::new(GRANULE).expect("not zero"),
        free_bytes: Some(free),
        ..bytes_card("card", u64::MAX, 0..3)
    };
    let weights = 9 * GRANULE;
    let need = weights + KV + context + scratch + ARENA + RESERVE;
    let fit = whole_need(&model, &at(need), KV, load).expect("every tensor has a format");
    assert_eq!((fit.weights, fit.weights_unrounded), (weights, 22_528));
    assert_eq!((fit.kv, fit.need, fit.budget), (KV, need, need));
    assert!(fit.fits(), "a reading at the need fits");
    let short = whole_need(&model, &at(need - 1), KV, load).expect("sized");
    assert!(
        !short.fits(),
        "the weights, the cache and all but one byte of the arena are not a whole load"
    );
    assert_eq!(short.budget, need - 1, "the reading as taken");
    let text = short.to_string();
    for part in [
        "the whole load on card: does not fit: need",
        "weights 36864 B (the allocator's granules; 22528 B of buffers) + cache 3000 B",
        "arena 50000 B + reserve past it 7000 B",
    ] {
        assert!(text.contains(part), "{part:?} in {text}");
    }
    let unchecked = whole_need(&model, &at(need), KV, WholeLoad::Unchecked).expect("sized");
    assert_eq!(
        unchecked.need,
        weights + KV + context + scratch + MARGIN,
        "the margin in the arena's place"
    );
    // A routed stack of a type with no device format (q2_K has none) is refused by name.
    // PIN(2026-10-25): iq1_s stood here until it had a device format.
    let mut odd = layered(3);
    odd.tensors[1].ty = GgmlType::Q2_K;
    match whole_need(&odd, &at(need), KV, load) {
        Err(e @ PlacementError::NoCardFormat { .. }) => assert_eq!(
            e.to_string(),
            "tensor experts 0: type q2_K has no device format, but its role (routed) puts it \
             on a card"
        ),
        Err(e) => panic!("{e}, not NoCardFormat"),
        Ok(n) => panic!("a q2_K stack was sized as {n:?}"),
    }
}

/// Two plan cards of one name are one device: their budgets past its
/// usable bytes are refused by name, and a card budget that splits the
/// device between them plans.
#[test]
fn one_device_counted_twice_is_refused() {
    let model = layered(3);
    let usable = HEAD + 24 * EXPERT;
    let machine = Machine {
        cards: vec![bytes_card("dev", usable, 0..3)],
        tiers: vec![bytes_card("dev", usable, 0..0)],
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    match plan_with(&model, &machine, 4096, &NoKv, None) {
        Err(e @ PlacementError::DeviceTwice { .. }) => assert_eq!(
            e.to_string(),
            format!(
                "card dev: 2 plan cards are this one device, and their budgets sum to {} B, \
                 past its usable {usable} B: one device's budget counted twice (a card \
                 budget that splits the device between them is planned)",
                2 * usable
            )
        ),
        Err(e) => panic!("{e}, not DeviceTwice"),
        Ok(_) => panic!("one device was planned twice"),
    }
    let half = usable / 2;
    let plan = plan_with(&model, &machine, 4096, &NoKv, Some(half)).expect("split budgets");
    assert!(plan.tier_n_l[0].iter().sum::<u64>() > 0);
    assert!(matches!(
        plan_with(&model, &machine, 4096, &NoKv, Some(half + 1)),
        Err(PlacementError::DeviceTwice { cards: 2, .. })
    ));
}

/// Two plan cards of one name resolved to two devices (two 3090s) are
/// two devices: each keeps its whole budget. The same device under both
/// is still one device counted twice.
#[test]
fn two_devices_of_one_name_plan() {
    let model = layered(3);
    let usable = HEAD + 5 * EXPERT;
    let dev = |ordinal: u32| {
        Some(devices::DeviceId {
            ordinal,
            uuid: [u8::try_from(ordinal).expect("small") + 1; 16],
        })
    };
    let on = |c: Card, d| Card { device: d, ..c };
    let machine = |a, b| Machine {
        cards: vec![on(bytes_card("3090", usable, 0..3), a)],
        tiers: vec![on(bytes_card("3090", usable, 0..0), b)],
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let two = machine(dev(0), dev(1));
    let plan = plan_with(&model, &two, 4096, &NoKv, None).expect("two 3090s");
    assert!(plan.tier_n_l[0].iter().sum::<u64>() > 0);
    assert!(matches!(
        plan_with(&model, &machine(dev(0), dev(0)), 4096, &NoKv, None),
        Err(PlacementError::DeviceTwice { cards: 2, .. })
    ));
}

/// The list type itself: sorted on construction, duplicates and ids past
/// the stack refused, the complement and runs of a scattered list, and a
/// prefix recognised as one.
#[test]
fn expert_list_shapes() {
    let l = ExpertList::new(vec![6, 1, 2], 8).expect("a list");
    assert_eq!(l.ids(), &[1, 2, 6]);
    assert_eq!(l.runs(), vec![1..3, 6..7]);
    assert_eq!(l.to_string(), "1..3,6..7");
    assert_eq!(l.complement(8).expect("fits").ids(), &[0, 3, 4, 5, 7]);
    assert_eq!(l.slot_of(6), Some(2));
    assert_eq!(l.as_prefix(), None);
    assert!(ExpertList::new(vec![1, 1], 8).is_err());
    assert!(ExpertList::new(vec![8], 8).is_err());
    let p = ExpertList::prefix(3).expect("a prefix");
    assert_eq!(
        (p.as_prefix(), p.to_string()),
        (Some(3), "0..3".to_string())
    );
}

/// The whole-fit verdict a load of N resident slots opens by is the
/// total's, never one slot's share: the cache is linear in the context,
/// so on one card a total one position past the whole fit is short while
/// its half fits — a verdict asked at the share would open the whole load
/// and the slots added after it would hold a cache the verdict never
/// counted (the qwen3 seat makes its verdict once, at the total).
#[test]
fn a_total_past_the_whole_fit_fits_at_half() {
    const ROW: u64 = 2_048;
    const TOTAL: u64 = 4_096;
    let model = layered(3);
    let load = WholeLoad::Unchecked;
    let kv = |ctx: u64| ctx * ROW;
    let at = |free: u64| Card {
        granule_bytes: NonZeroU64::new(4096).expect("not zero"),
        free_bytes: Some(free),
        ..bytes_card("card", u64::MAX, 0..3)
    };
    let edge = whole_need(&model, &at(u64::MAX), kv(TOTAL - 1), load)
        .expect("every tensor has a format")
        .need;
    let card = at(edge);
    let total = whole_need(&model, &card, kv(TOTAL), load).expect("sized");
    let share = whole_need(&model, &card, kv(TOTAL / 2), load).expect("sized");
    assert!(
        whole_need(&model, &card, kv(TOTAL - 1), load)
            .expect("sized")
            .fits(),
        "the reading holds the whole load one position short of the total"
    );
    assert!(!total.fits(), "the total does not fit: {total}");
    assert!(share.fits(), "a slot's share of it does: {share}");
}

/// A card's KV class grown after the totals (`CardTotals::grow_kv`: the
/// bytes a model's resident sequences hold beside their stores, reserved
/// out of the expert budget) leaves its headroom the usable bytes less
/// every term, the KV it gained among them.
#[test]
fn grown_kv_leaves_the_headroom_usable_less_every_term() {
    let model = layered(3);
    let machine = Machine {
        cards: vec![Card {
            context_bytes: 1_000,
            scratch_bytes: 2_000,
            reserves: vec![("draft".to_string(), 3_000)],
            ..bytes_card("stage", HEAD + 12 * EXPERT + 6_000 + 5_000, 0..3)
        }],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let mut plan = plan_routed_reserving(
        &model,
        &machine,
        4096,
        &NoKv,
        &PlanLevers::default(),
        RoutedRule::PerType(CardFormat::of_routed),
        5_000,
    )
    .expect("the plan");
    let rest = |p: &Plan<'_>| {
        let t = &p.cards[0];
        i128::from(p.usable_bytes(&machine.cards[0]))
            - i128::from(
                t.dense_bytes
                    + t.expert_bytes
                    + t.rounding_bytes
                    + t.kv_bytes
                    + t.scratch_bytes
                    + t.context_bytes
                    + t.reserve_bytes,
            )
    };
    assert_eq!(plan.cards[0].headroom_bytes, rest(&plan));
    plan.cards[0].grow_kv(5_000);
    assert_eq!(plan.cards[0].kv_bytes, 5_000);
    assert_eq!(plan.cards[0].headroom_bytes, rest(&plan));
}

/// The plan's host bound is the load's own check at the room the
/// machine's figure leaves (`HostNeed::check` at
/// `HostNeed::machine_room`): a host whose usable bytes hold the plan's
/// host tensors and reserves to the byte passes and one byte under is a
/// `HostOver` naming both, the headroom the usable bytes less those terms
/// — whether the OS reserve carries the name a reading leaves out or
/// another — and the NVMe tier's arena and run-ahead window, sized to
/// the room the split read and held to it, stay out of the bound.
#[test]
fn the_host_bound_is_the_needs_check_at_the_machine_room() {
    let model = layered(3);
    for os in [workstation::OS_RESERVE, "other"] {
        let machine = |usable: u64| Machine {
            cards: vec![bytes_card("stage", HEAD + 12 * EXPERT, 0..3)],
            tiers: Vec::new(),
            nodes: vec![Node {
                host: Host {
                    usable_bytes: usable,
                    reserves: vec![(os.to_string(), 7_000), ("pool".to_string(), 1_000)],
                },
                unified: None,
            }],
        };
        let probe = machine(1 << 40);
        let plan = plan_with(&model, &probe, 4096, &NoKv, None).expect("the plan");
        assert!(plan.host.expert_bytes > 0, "{os}: the host holds experts");
        let terms = plan.host.expert_bytes + plan.host.table_bytes + plan.host.shadow_bytes + 8_000;
        let over = |p: &Plan<'_>| {
            p.violations()
                .iter()
                .filter(|v| matches!(v, Violation::HostOver { .. }))
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        };
        for usable in [terms, terms - 1] {
            let m = machine(usable);
            let plan = plan_with(&model, &m, 4096, &NoKv, None).expect("the plan");
            let want = if usable < terms {
                vec![format!(
                    "host: tensors + ring shadows + reserves = {terms} B pass usable {usable} \
                     B by 1 B"
                )]
            } else {
                Vec::new()
            };
            assert_eq!(over(&plan), want, "{os} at usable {usable}");
            assert_eq!(
                plan.host.headroom_bytes,
                i128::from(usable) - i128::from(terms),
                "{os} at usable {usable}"
            );
        }
        let m = machine(terms);
        let mut plan = plan_with(&model, &m, 4096, &NoKv, None).expect("the plan");
        plan.host.nvme_arena_bytes = 1;
        plan.host.reserve_bytes += 1;
        assert_eq!(
            over(&plan),
            Vec::<String>::new(),
            "{os}: the split's arena and window are held to its room"
        );
    }
}

/// The row room counts the pages a row sits on, in the host's base page:
/// a 90 B row can straddle a page boundary at 4, 16 or 64 KiB, so the
/// room of a call of 4,096 positions reading 16 rows a position is
/// 4,096 · 16 · 2 · page —
/// 536,870,912 B on this workstation's page and 2,147,483,648 B on the
/// 16 KiB one, and 8,589,934,592 B on the 64 KiB one.
#[test]
fn the_row_room_counts_the_hosts_base_page() {
    let model = ModelTensors {
        tensors: vec![ModelTensor {
            name: "ple table".into(),
            shard: 0,
            layer: Some(0),
            role: Role::EngramTable,
            ty: GgmlType::Q4_K,
            dims: vec![90, 16],
            file_bytes: 16 * 90,
            gathered_rows: Some(16),
        }],
        layers: 1,
        experts: 0,
        experts_used: 0,
        bf16_words: |_| false,
    };
    for source in [RowTableSource::Split, RowTableSource::Parts] {
        let room = |page| row_room(&model, 4096, page, source).expect("the row room");
        assert_eq!(room(4096), 536_870_912);
        assert_eq!(room(16384), 2_147_483_648);
        assert_eq!(room(65536), 8_589_934_592);
    }

    let mut model = model;
    let table = &mut model.tensors[0];
    table.ty = GgmlType::NVFP4_ROWS;
    table.dims = vec![160, 100];
    table.file_bytes = 90 * 100;
    table.gathered_rows = Some(1);
    let touched: u64 = [80, 10]
        .into_iter()
        .map(|width| {
            let start = 256 + 99 * width;
            (start + width - 1) / 4096 - start / 4096 + 1
        })
        .sum();
    assert_eq!(touched, 3);
    let reserve =
        row_room(&model, 1, 4096, RowTableSource::Parts).expect("the separate-plane row room");
    assert!(
        reserve >= touched * 4096,
        "{reserve} B reserved for {touched} pages"
    );
    model.tensors[0].gathered_rows = Some(16);
    for (source, want) in [
        (RowTableSource::Split, 536_870_912),
        (RowTableSource::Parts, 1_073_741_824),
    ] {
        for page in [4096, 16384, 65536] {
            assert_eq!(
                row_room(&model, 4096, page, source).unwrap(),
                want * (page / 4096)
            );
        }
    }
    model.tensors[0].file_bytes = 89 * 100;
    let e = row_room(&model, 4096, 4096, RowTableSource::Parts).unwrap_err();
    assert!(
        e.to_string()
            .contains("NVFP4 row bytes are not whole groups")
    );
}
