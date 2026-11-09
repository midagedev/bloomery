//! The plan's KV floor ([`PlanLevers::kv_floor`], `BLOOMERY_KV_FLOOR`) over
//! [`layered`]'s four-layer model: the expert spread runs at the floor's
//! positions while every other term stays at the cap, a floor above the cap
//! is refused by name, and unset the plan is the one made before the lever.

use super::*;

/// 18 B of KV a position a layer (72 B a position over the four layers) and
/// 4 B of stage rows a position: ten positions of KV or twenty of rows take
/// one [`EXPERT`].
struct Kv;

impl KvBytes for Kv {
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
