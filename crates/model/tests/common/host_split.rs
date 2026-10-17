//! The host split of a plan, read from its rows: each routed layer's id
//! prefix `[0, n_l)` on the card and the rest on the host
//! (`Experts::Placed`). One owner of the clauses, for the fixture's unit gate
//! (`nemotron_h_moe::spec`) and the real file's (`nemotron_h_moe_meta`); both
//! name the placement types as `crate::placement`. Also the budget that
//! forces the split ([`half_expert_budget`]).

use std::collections::BTreeMap;

use crate::placement::{Device, Format, Plan, Role};

/// The one card's budget for its dense floor and half its expert bytes: the
/// card of `plan`, shrunk to it, holds some routed experts and not all.
///
/// # Panics
/// On a card whose floor passes u64 bytes.
pub fn half_expert_budget(plan: &Plan<'_>) -> u64 {
    let c = &plan.cards[0];
    let card = &plan.machine.cards[0];
    card.floor_bytes(
        c.dense_bytes + c.rounding_bytes + c.expert_bytes / 2,
        c.kv_bytes,
        c.stage_rows_bytes + card.margin_bytes,
    )
    .expect("bounded partial-plan budget")
}

/// A layer with routed stacks, and what its plan holds of them.
#[derive(Clone, Copy, Debug)]
pub struct MoeLayer {
    pub layer: usize,
    /// The layer's experts (`ModelTensors::layer_experts`).
    pub experts: u64,
    /// The experts the plan holds on the card (`Plan::n_l`).
    pub on_card: u64,
    /// The routed stacks' resident bytes on the card, and on the host.
    pub card_bytes: u64,
    pub host_bytes: u64,
}

/// Every layer with a routed stack and the clauses of a split that do not
/// depend on the card's size, each failure named: the card's headroom holds
/// the margin; no layer holds more than it has; each routed row is `[0, n_l)`
/// on card 0, then `[n_l, E_l)` on the host's file, every expert once; a
/// layer's stacks hold the same lists; the host segment's bytes are its
/// share of the file's; the host's expert count is the sum of the layers'
/// (`Plan::host_experts` agreeing).
pub fn prefix_split(plan: &Plan<'_>) -> (Vec<MoeLayer>, Vec<String>) {
    let model = plan.model;
    let mut bad = Vec::new();
    let margin = i128::from(plan.machine.cards[0].margin_bytes);
    if plan.cards[0].headroom_bytes < margin {
        bad.push(format!(
            "card headroom {} below the margin {margin}",
            plan.cards[0].headroom_bytes
        ));
    }
    let mut stacks: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, t) in model.tensors.iter().enumerate() {
        if t.role == Role::RoutedExperts {
            let layer = t.layer.expect("a routed stack belongs to a layer");
            stacks.entry(layer).or_default().push(i);
        }
    }
    let mut layers = Vec::new();
    let (mut sum_host, mut sum_host_of) = (0u64, 0u64);
    for (&layer, tensors) in &stacks {
        let experts = model.layer_experts(layer);
        let on_card = plan.n_l[layer];
        let mut at = MoeLayer {
            layer,
            experts,
            on_card,
            card_bytes: 0,
            host_bytes: 0,
        };
        if on_card > experts {
            bad.push(format!("layer {layer}: n_l {on_card} past its {experts}"));
            layers.push(at);
            continue;
        }
        sum_host += experts - on_card;
        sum_host_of += plan
            .host_experts(layer)
            .unwrap_or_else(|e| panic!("layer {layer}: {e}"));
        if tensors.len() != 2 {
            bad.push(format!(
                "layer {layer}: {} routed stacks, want up and down",
                tensors.len()
            ));
        }
        let ids = |lo: u64, hi: u64| -> Vec<u32> {
            (lo..hi)
                .map(|e| u32::try_from(e).expect("an expert id"))
                .collect()
        };
        let mut want: Vec<(Device, Vec<u32>)> = Vec::new();
        if on_card > 0 {
            want.push((Device::Card(0), ids(0, on_card)));
        }
        if on_card < experts {
            want.push((Device::Host, ids(on_card, experts)));
        }
        let mut lists: Vec<Vec<(Device, Vec<u32>)>> = Vec::new();
        for &i in tensors {
            let t = &model.tensors[i];
            let row = &plan.rows[i];
            assert_eq!(row.tensor, i, "rows are in the tensor order");
            let mut got: Vec<(Device, Vec<u32>)> = Vec::new();
            let mut seen: Vec<u32> = Vec::new();
            for s in &row.segments {
                let held = s
                    .experts
                    .as_ref()
                    .map_or_else(Vec::new, |list| list.ids().to_vec());
                seen.extend(&held);
                got.push((s.device, held));
                if s.device == Device::Host {
                    at.host_bytes += s.resident_bytes;
                    let share = (experts - on_card) * (t.file_bytes / experts);
                    if s.format != Format::HostFile || s.resident_bytes != share {
                        bad.push(format!(
                            "{}: host segment {} as {} bytes, want {share} as HostFile",
                            t.name, s.format, s.resident_bytes
                        ));
                    }
                } else {
                    at.card_bytes += s.resident_bytes;
                }
            }
            if got != want {
                bad.push(format!("{}: segments {got:?}, want {want:?}", t.name));
            }
            seen.sort_unstable();
            if seen.windows(2).any(|w| w[0] == w[1]) || seen.len() as u64 != experts {
                bad.push(format!(
                    "{}: {} ids for {experts} experts",
                    t.name,
                    seen.len()
                ));
            }
            lists.push(got);
        }
        if lists.windows(2).any(|w| w[0] != w[1]) {
            bad.push(format!("layer {layer}: its stacks hold different lists"));
        }
        layers.push(at);
    }
    if plan.host.experts != sum_host || sum_host != sum_host_of {
        bad.push(format!(
            "host.experts {}, sum of E_l - n_l {sum_host}, sum of host_experts {sum_host_of}",
            plan.host.experts
        ));
    }
    (layers, bad)
}
