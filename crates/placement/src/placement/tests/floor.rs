//! A default context's floor on the card's experts
//! ([`card_experts_floor`]) over [`layered`]'s four-layer model, one card
//! of its head and the experts its context leaves room for.

use super::*;
use crate::placement::ctx_ask::{
    Ask, Limit, Probe, Rule, Short, Side, Steps, card_experts_floor, choose, short_of,
};

/// 18 B of KV a position a layer: 72 B a position over the four layers, so
/// eight positions take one [`EXPERT`].
struct Kv;

impl KvBytes for Kv {
    fn layer_bytes(&self, _layer: usize, ctx_max: u64) -> u64 {
        18 * ctx_max
    }
}

const LIMIT: Limit = Limit {
    positions: 80,
    source: "test",
};

const STEPS: Steps = Steps { floor: 8, step: 8 };

/// A card of the head, the KV of the default's 80 positions and `spare`
/// experts: its plan at `c` positions, a multiple of 8, holds
/// `spare + 10 − c / 8` experts, as many as its room takes.
fn card_of(spare: u64) -> Machine {
    machine_of(HEAD + 80 * 72 + spare * EXPERT)
}

/// One card of `usable` bytes beside the host.
fn machine_of(usable: u64) -> Machine {
    Machine {
        cards: vec![bytes_card("stage", usable, 0..4)],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    }
}

/// A seat's probe at `c` positions: the plan's experts a layer when it
/// holds, the card's shortfall when it does not, and with `least` a plan
/// that leaves a layer fewer experts short too.
fn probe(
    model: &ModelTensors,
    machine: &Machine,
    c: u64,
    least: Option<u64>,
) -> Result<Probe<Vec<u64>>, PlacementError> {
    let plan = plan_with(model, machine, c, &Kv, None)?;
    if let Some(short) = short_of(&plan.violations()) {
        return Ok(Probe::Short(short));
    }
    match least.map(|least| card_experts_floor(&plan, 0, least)) {
        Some(Err(e)) => e.card_short().map(Probe::Short).ok_or(e),
        _ => Ok(Probe::Holds(plan.n_l)),
    }
}

/// A plan that leaves layers under the floor is short on the card by the
/// bytes of the experts it lacks, over every layer: five experts over four
/// layers leave three of them one, which two a layer asks one more each.
/// FAIL-first: a comparison of `<=` where the floor counts a layer short
/// (`held < want`) counts layer 0 too and turns the layer count red.
#[test]
fn a_plan_under_the_floor_is_short_by_the_experts_it_lacks() {
    let model = layered(4);
    let machine = card_of(5);
    let plan = plan_with(&model, &machine, 80, &Kv, None).expect("a plan");
    assert_eq!(
        plan.n_l,
        [2, 1, 1, 1],
        "five experts: the round robin's second round reaches layer 0 only"
    );
    let e = card_experts_floor(&plan, 0, 2).expect_err("three layers hold one expert");
    assert!(
        matches!(
            e,
            PlacementError::CardExpertsFloor { layers: 3, least: 2, need, room, .. }
                if (need, room) == (8 * EXPERT, 5 * EXPERT)
        ),
        "{e}"
    );
    assert_eq!(
        e.card_short(),
        Some(Short {
            on: Side::Card,
            need: 8 * EXPERT,
            room: 5 * EXPERT
        }),
        "the refusal reads as a short card by the experts the plan lacks"
    );
    card_experts_floor(&plan, 0, 1).expect("every layer holds one");
    card_experts_floor(&plan, 0, 0).expect("a floor of none asks nothing");
    assert!(
        matches!(
            card_experts_floor(&plan, 1, 2),
            Err(PlacementError::Experts(_))
        ),
        "a plan of one card names no card 1"
    );
}

/// A layer holds at most its stack, so a floor past it asks the whole stack.
/// FAIL-first: a `want` that is `least` without the stack's cap asks nine of
/// a layer of eight and turns this red.
#[test]
fn the_floor_asks_no_more_than_a_layers_stack_holds() {
    let model = layered(4);
    let machine = card_of(32);
    let plan = plan_with(&model, &machine, 80, &Kv, None).expect("a plan");
    assert_eq!(plan.n_l, [8, 8, 8, 8], "a card with room for every expert");
    card_experts_floor(&plan, 0, 9).expect("a layer holds its eight, all it has");
}

/// A stage with no routed stack has no experts to ask for.
/// FAIL-first: a loop that takes a layer with no routed stack for one of
/// zero-byte experts counts it short and turns this red.
#[test]
fn a_stage_with_no_routed_stack_has_no_floor() {
    let mut model = layered(4);
    model.tensors.retain(|t| t.role != Role::RoutedExperts);
    let machine = card_of(0);
    let plan = plan_with(&model, &machine, 80, &Kv, None).expect("a dense plan");
    assert_eq!(
        plan.n_l,
        [0, 0, 0, 0],
        "a dense plan holds no routed expert"
    );
    card_experts_floor(&plan, 0, 2).expect("no layer holds a routed expert to ask for");
}

/// A card the default's target fills with context and nothing else: the plain
/// fit takes the target and every layer holds no expert, where the fit that
/// reads the floor takes the largest context whose plan leaves two a layer —
/// 16 positions, eight experts — and names the plan one step above short by
/// the one expert it lacks there.
/// FAIL-first: a probe without the floor (`least` None) takes the target, 80
/// positions and no expert, and turns the floored assertions red; a floor that
/// skips the layers holding none, as the residency rule's `fewest` does, sees
/// no layer at the target and does the same.
#[test]
fn a_default_leaves_the_floor_where_the_plain_fit_spends_the_card_on_context() {
    let model = layered(4);
    let machine = card_of(0);
    let fit = |least: Option<u64>| {
        choose(Ask::Default, 1, LIMIT, STEPS, |c| {
            probe(&model, &machine, c, least)
        })
        .expect("a default")
    };
    let plain = fit(None);
    assert_eq!(
        (plain.slot_ctx, plain.rule),
        (80, Rule::Target),
        "the card holds the target with no expert beside it"
    );
    assert_eq!(plain.plan, [0, 0, 0, 0], "the whole card is context");
    let floored = fit(Some(2));
    assert_eq!(
        floored.slot_ctx, 16,
        "eight experts, two a layer, are what 16 positions leave"
    );
    assert_eq!(
        floored.rule,
        Rule::Fitted {
            from: 80,
            short: Short {
                on: Side::Card,
                need: 8 * EXPERT,
                room: 7 * EXPERT
            }
        },
        "the step above, 24 positions, leaves seven experts of the eight"
    );
    assert_eq!(floored.plan, [2, 2, 2, 2], "the floor's experts a layer");
}

/// A target that already leaves the floor, or more, stays the target: the
/// floor moves a default only where the plain fit spends the experts' room.
/// Twelve experts are three a layer, so a floor of three holds with nothing
/// to spare.
/// FAIL-first: a comparison of `<=` in the floor (`held < want`) counts a
/// layer that holds exactly the floor short, fits below the target for a
/// floor of three and turns this red.
#[test]
fn a_target_that_leaves_the_experts_stays_the_target() {
    let model = layered(4);
    let machine = card_of(12);
    for least in [None, Some(2), Some(3)] {
        let chosen = choose(Ask::Default, 1, LIMIT, STEPS, |c| {
            probe(&model, &machine, c, least)
        })
        .expect("a default");
        assert_eq!(
            (chosen.slot_ctx, chosen.rule),
            (80, Rule::Target),
            "{least:?}"
        );
        assert_eq!(
            chosen.plan,
            [3, 3, 3, 3],
            "{least:?}: twelve experts, three a layer"
        );
    }
}

/// A dense plan's fitted context is the same with the floor and without: its
/// stage holds no routed stack, so the floor asks nothing of it. The card
/// holds 40 of the 80 positions; both fits name the same plan one step above
/// short.
/// FAIL-first: a floor that reads a layer with no routed stack as one of
/// experts lacking (`stacks.peek()` dropped) sees every layer short, takes the
/// floor of 8 positions and turns this red.
#[test]
fn a_dense_plans_fitted_context_is_unchanged_by_the_floor() {
    let mut model = layered(4);
    model.tensors.retain(|t| t.role != Role::RoutedExperts);
    let machine = machine_of(HEAD + 40 * 72);
    let fit = |least: Option<u64>| {
        choose(Ask::Default, 1, LIMIT, STEPS, |c| {
            probe(&model, &machine, c, least)
        })
        .expect("a default")
    };
    let (plain, floored) = (fit(None), fit(Some(2)));
    assert_eq!(plain.slot_ctx, 40, "the card holds 40 positions of the 80");
    assert!(
        matches!(plain.rule, Rule::Fitted { from: 80, .. }),
        "{:?}",
        plain.rule
    );
    assert_eq!(
        (floored.slot_ctx, floored.rule),
        (plain.slot_ctx, plain.rule),
        "a stage with no routed stack asks nothing of the context"
    );
}
