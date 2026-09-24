//! Qwen3-MoE placement gate: the whole model on one 24 GB card. The file's
//! plan on the gate placement (`workstation::plan_gate`: the 3090 runs every
//! layer and the head) at the serving context keeps every routed expert of
//! every layer on the card, breaks no invariant, and its byte totals are the
//! sums of the tensors' device bytes recomputed from the header alone.
//!
//! The same plan under card budgets pins where "whole" stops: at exactly the
//! bytes the whole plan takes (its granules + KV + scratch + context +
//! margin) it is still whole; one byte less and a layer keeps an expert on
//! the host; below the card's no-expert floor the plan is refused.
//!
//! `hw_`: needs the file on the box (`just gate-qwen3moe-placement`, which
//! picks the qwen3moe tool profile). Headers only: no tensor bytes, no GPU.

use gguf::Split;
use model::arch::qwen3moe::place::{PlaceError, PlanInputs};
use model::placement::{CardFormat, Device, PlacementError, Role, workstation};

fn inputs() -> PlanInputs {
    let path = std::env::var("BLOOMERY_REF_MODEL").unwrap_or_else(|_| {
        panic!("BLOOMERY_REF_MODEL is unset: run through `just gate-qwen3moe-placement`")
    });
    let split = Split::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    assert_eq!(split.architecture(), Some("qwen3moe"), "{path}");
    PlanInputs::read(&split).unwrap_or_else(|e| panic!("plan inputs of {path}: {e}"))
}

#[test]
#[ignore = "needs the qwen3moe file on the box"]
fn hw_qwen3moe_whole_on_the_3090() {
    let inp = inputs();
    let (hp, model) = (&inp.hp, &inp.model);
    let machine = workstation::plan_gate(hp.n_layer);
    let ctx = workstation::CTX_MAX;
    let plan = inp
        .plan_with(&machine, ctx, None)
        .and_then(|p| inp.whole(p))
        .unwrap_or_else(|e| panic!("the whole-card plan at ctx {ctx}: {e}"));
    let card = &plan.cards[0];
    let spec = &machine.cards[0];

    // Every tensor's device bytes from the header alone: the card format's
    // arithmetic over its dims, and the token embedding's file bytes on the host.
    let (mut card_bytes, mut host_bytes) = (0u64, 0u64);
    for t in &model.tensors {
        if t.role == Role::TokenEmbedding {
            host_bytes += t.file_bytes;
            continue;
        }
        let format = CardFormat::of(t.ty).unwrap_or_else(|| panic!("{}: no card format", t.name));
        let rows: u64 = t.dims.iter().skip(1).product();
        card_bytes += format
            .resident_bytes(t.ty, t.dims[0], rows)
            .unwrap_or_else(|| panic!("{}: {format:?} has no size for {:?}", t.name, t.dims));
    }
    let on_card: u64 = plan
        .rows
        .iter()
        .flat_map(|r| &r.segments)
        .filter(|s| s.device == Device::Card(0))
        .map(|s| s.resident_bytes)
        .sum();
    let kv = hp.n_layer as u64 * ctx * 2 * (hp.n_head_kv * hp.head_dim) as u64 * 2;
    let whole_floor = card.dense_bytes
        + card.expert_bytes
        + card.rounding_bytes
        + card.kv_bytes
        + card.scratch_bytes
        + card.context_bytes
        + spec.margin_bytes;
    println!(
        "qwen3moe on {} at ctx {ctx}: dense {} + experts {} ({} experts) + rounding {} + KV {} + \
         scratch {} + context {} = {} B; margin {}; usable {}; headroom {} B; host tables {} B",
        spec.name,
        card.dense_bytes,
        card.expert_bytes,
        card.experts,
        card.rounding_bytes,
        card.kv_bytes,
        card.scratch_bytes,
        card.context_bytes,
        whole_floor - spec.margin_bytes,
        spec.margin_bytes,
        spec.usable_bytes,
        card.headroom_bytes,
        plan.host.table_bytes,
    );

    let mut bad = Vec::new();
    let mut check = |what: &str, got: u64, want: u64| {
        let mark = if got == want { "ok " } else { "BAD" };
        println!("  {mark} {what:<40} {got}  (want {want})");
        if got != want {
            bad.push(format!("{what}: {got}, want {want}"));
        }
    };
    check(
        "card resident = Σ card segments",
        card.dense_bytes + card.expert_bytes,
        on_card,
    );
    check("card resident = Σ header device bytes", on_card, card_bytes);
    check(
        "experts on the card",
        card.experts,
        model.experts * hp.n_layer as u64,
    );
    check("host experts", plan.host.experts, 0);
    check(
        "host tables = token_embd file bytes",
        plan.host.table_bytes,
        host_bytes,
    );
    check("KV = layers·ctx·2·kv_heads·head·f16", card.kv_bytes, kv);
    check("headroom ≥ 0", u64::from(card.headroom_bytes >= 0), 1);

    // Where "whole" stops.
    let at = |budget: u64| {
        inp.plan_with(&machine, ctx, Some(budget))
            .and_then(|p| inp.whole(p))
    };
    match at(whole_floor) {
        Ok(_) => println!("  ok  budget {whole_floor} B (the whole plan's bytes): whole"),
        Err(e) => bad.push(format!(
            "budget {whole_floor} B, the whole plan's own bytes: {e}"
        )),
    }
    match at(whole_floor - 1) {
        Err(PlaceError::NotWhole { short, .. }) => {
            println!(
                "  ok  budget {} B: not whole, {} layers short (first {:?}, last {:?})",
                whole_floor - 1,
                short.len(),
                short.first(),
                short.last()
            );
        }
        other => bad.push(format!(
            "budget {} B, one below the whole plan: {:?}, want NotWhole",
            whole_floor - 1,
            other.map(|p| p.n_l.clone())
        )),
    }
    let no_expert_floor = whole_floor - card.expert_bytes;
    match at(no_expert_floor / 2) {
        Err(PlaceError::Placement(PlacementError::CardBudgetFloor { floor, .. })) => {
            println!(
                "  ok  budget {} B: refused below the floor {floor} B",
                no_expert_floor / 2
            );
        }
        other => bad.push(format!(
            "budget {} B, below the no-expert floor: {:?}, want CardBudgetFloor",
            no_expert_floor / 2,
            other.map(|p| p.n_l.clone())
        )),
    }
    assert!(
        bad.is_empty(),
        "{} check(s) failed:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}
