//! An elastic fill's static twin: the plan that holds the experts a fill
//! leaves on the card. The fill restores the same `k` slots on every layer
//! ([`crate::placement::elastic::fill_slots`] counts in quanta, one slot a
//! layer), and the expert rule hands one expert at a time to the layer with
//! the fewest, the first on a tie ([`super::super::rule::spread`]), so a
//! plan's per-layer prefixes are a function of its total alone.

use super::*;

/// A card of the head and room for `experts` of [`layered`]'s experts.
fn card_for(experts: u64) -> Machine {
    Machine {
        cards: vec![bytes_card("stage", HEAD + experts * EXPERT, 0..4)],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    }
}

fn n_l(model: &ModelTensors, experts: u64) -> Vec<u64> {
    plan_with(model, &card_for(experts), 1, &NoKv, None)
        .expect("a plan")
        .n_l
}

/// A fill of `k` slots a layer over a plan of `base` card experts holds the
/// prefixes of the plan of `base + k × layers`, whether the base ends on a
/// partial round (5 over 4 layers: `[2, 1, 1, 1]`) or a whole one (8:
/// `[2, 2, 2, 2]`), up to the stack's eight: the static twin of a fill is
/// the plan of the same total, and only its total can differ.
///
/// FAIL-first: a `spread` that takes the layer with the most (`max_by_key`,
/// the last on a tie) plans `[0, 0, 0, 5]` at 5 and the first assertion goes
/// red; past it, the fill makes `[1, 1, 1, 6]` where the plan of 9 is
/// `[0, 0, 1, 8]`, red at `base` 5, `k` 1.
#[test]
fn a_fill_of_k_a_layer_holds_the_prefixes_of_the_plan_k_rounds_wider() {
    let model = layered(4);
    assert_eq!(n_l(&model, 5), [2, 1, 1, 1], "a base on a partial round");
    assert_eq!(n_l(&model, 8), [2, 2, 2, 2], "a base on a whole round");
    for base in [5, 8] {
        let planned = n_l(&model, base);
        let deepest = *planned.iter().max().expect("four layers");
        for k in 1..=model.experts - deepest {
            let filled: Vec<u64> = planned.iter().map(|n| n + k).collect();
            let twin = n_l(&model, base + k * 4);
            assert_eq!(filled, twin, "base {base}, {k} a layer");
            assert_eq!(twin.iter().sum::<u64>(), base + k * 4);
        }
    }
}

/// [`plan`] of [`layered`]`(4)` on a card with room for `room` experts under
/// a cap of `cap` card experts.
fn capped_plan(model: &ModelTensors, room: u64, cap: u64) -> Result<Vec<u64>, PlacementError> {
    let machine = card_for(room);
    let levers = PlanLevers {
        card_experts_cap: Some(cap),
        ..PlanLevers::default()
    };
    let plan = plan(model, &machine, 1, &NoKv, &levers)?;
    assert_eq!(plan.card_experts_cap, Some(cap), "the plan names its cap");
    Ok(plan.n_l)
}

/// A cap of `n` card experts holds the same experts at every budget that
/// holds `n`: the plan of exactly `n`'s room, of a wider room, and of the
/// whole stack's are one plan. This is the static twin of an elastic load of
/// 5 filled one slot a layer: `[3, 2, 2, 2]`, whatever the twin's budget.
///
/// FAIL-first: a `spread` that drops `placed < most` plans `[3, 3, 3, 2]` at
/// a room of 11, red on the second budget.
#[test]
fn a_cap_holds_the_same_experts_at_every_budget_that_holds_it() {
    let model = layered(4);
    let filled: Vec<u64> = n_l(&model, 5).iter().map(|n| n + 1).collect();
    assert_eq!(filled, [3, 2, 2, 2]);
    for room in [9, 11, 32, 40] {
        assert_eq!(
            capped_plan(&model, room, 9).expect("a room that holds the cap"),
            filled,
            "room {room}"
        );
    }
}

/// The cap counts every card of the plan: a stage card that holds 5 and a
/// tier with room for 4 under a cap of 7 leave the tier the 2 the stage did
/// not take, not its whole room.
///
/// FAIL-first: a `spread` that counts only its own card's experts (`n_l`
/// without `held`) lets the tier take its four, `[1, 1, 2]`.
#[test]
fn a_cap_counts_the_experts_of_every_card() {
    let model = layered(3);
    let machine = Machine {
        cards: vec![bytes_card("stage", HEAD + 5 * EXPERT, 0..3)],
        tiers: vec![bytes_card("tier", 4 * EXPERT, 0..0)],
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    };
    let levers = PlanLevers {
        card_experts_cap: Some(7),
        ..PlanLevers::default()
    };
    let plan = plan(&model, &machine, 4096, &NoKv, &levers).expect("a capped tier plan");
    assert_eq!(plan.n_l, [2, 2, 1]);
    assert_eq!(plan.tier_n_l, vec![vec![1, 0, 1]]);
}

/// A cap past what the cards hold is refused by name with the count they
/// hold: the cap never widens a budget, and a planner that places no routed
/// expert on a card refuses any cap.
///
/// FAIL-first: a planner without the `held < cap` check plans the 5 the room
/// holds under a cap of 6 and the first `expect_err` goes red.
#[test]
fn a_cap_past_what_the_cards_hold_is_refused_with_their_count() {
    let model = layered(4);
    let e = capped_plan(&model, 5, 6).expect_err("a room of 5 under a cap of 6");
    assert!(
        matches!(e, PlacementError::CardExpertsCap { cap: 6, held: 5 }),
        "{e}"
    );
    assert!(
        e.to_string()
            .starts_with("BLOOMERY_CARD_EXPERTS_CAP=6: the plan's cards hold 5 routed experts"),
        "{e}"
    );
    let e = capped_plan(&model, 40, 33).expect_err("a cap past the model's 32 experts");
    assert!(
        matches!(e, PlacementError::CardExpertsCap { cap: 33, held: 32 }),
        "{e}"
    );
    let levers = PlanLevers {
        card_experts_cap: Some(1),
        ..PlanLevers::default()
    };
    let e = plan_host_routed(&model, &card_for(40), 1, &NoKv, &levers)
        .expect_err("a host-routed plan holds no card expert");
    assert!(
        matches!(e, PlacementError::CardExpertsCap { cap: 1, held: 0 }),
        "{e}"
    );
}
