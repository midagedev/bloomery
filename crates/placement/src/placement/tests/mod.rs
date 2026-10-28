use std::num::NonZeroU64;
use std::ops::Range;

use gguf::GgmlType;

use super::planner::capped;
use super::pool::POOL_ROUNDS;
use super::rule::{Footprint, Held, card_buffers, card_heaps, routed_row_on_as, rows_of};
use super::*;

mod bounds;
mod early;
mod legs;

/// A plan of no rows under `budget`, for a planner [`plan_in_pool`]
/// calls: only its budget is read.
fn budget_plan<'a>(model: &'a ModelTensors, machine: &'a Machine, budget: u64) -> Plan<'a> {
    Plan {
        model,
        machine,
        ctx_max: 1,
        rows: Vec::new(),
        cards: Vec::new(),
        host: HostTotals {
            expert_bytes: 0,
            experts: 0,
            nvme_expert_bytes: 0,
            nvme_arena_bytes: 0,
            arena_columns: 1,
            table_bytes: 0,
            shadow_bytes: 0,
            reserve_bytes: 0,
            row_reserve_bytes: 0,
            pool_card_bytes: 0,
            headroom_bytes: 0,
            room: None,
        },
        nvme_bytes: 0,
        n_l: Vec::new(),
        tier_n_l: Vec::new(),
        card_budget: card_budget::CardBudget::Every(budget),
        budget_from_pool: false,
    }
}
/// A card running `layers`, with the head and the token embedding as
/// given; its byte figures play no part in the stage map.
fn card(name: &str, layers: Range<usize>, head: bool, token_embedding: bool) -> Card {
    Card {
        name: name.to_string(),
        tier_leg: None,
        device: None,
        usable_bytes: 0,
        context_bytes: 0,
        scratch_bytes: 0,
        margin_bytes: 0,
        granule_bytes: NonZeroU64::MIN,
        free_bytes: None,
        held_by: None,
        layers,
        head,
        token_embedding,
        reserve_only: false,
        reserves: Vec::new(),
    }
}
/// A model of `layers` layers, each one routed q4_K stack of 8 experts of
/// four 256-value rows (576 B an expert), with a q8_0 token embedding and
/// head of 16 rows (4,352 B).
fn layered(layers: usize) -> ModelTensors {
    let tensor = |name: String, layer, role, ty: GgmlType, rows: &[u64]| {
        let mut dims = vec![256];
        dims.extend_from_slice(rows);
        let blocks = 256 / ty.blck_size().expect("a sized type");
        ModelTensor {
            name,
            shard: 0,
            layer,
            role,
            ty,
            dims,
            file_bytes: ty.type_size().expect("a sized type")
                * blocks
                * rows.iter().product::<u64>(),
            gathered_rows: (role == Role::TokenEmbedding).then_some(1),
        }
    };
    let mut tensors = vec![tensor(
        "embedding".into(),
        None,
        Role::TokenEmbedding,
        GgmlType::Q8_0,
        &[16],
    )];
    for l in 0..layers {
        tensors.push(tensor(
            format!("experts {l}"),
            Some(l),
            Role::RoutedExperts,
            GgmlType::Q4_K,
            &[4, 8],
        ));
    }
    tensors.push(tensor(
        "head".into(),
        None,
        Role::Head,
        GgmlType::Q8_0,
        &[16],
    ));
    ModelTensors {
        tensors,
        layers,
        experts: 8,
        experts_used: 2,
        bf16_words: |_| false,
    }
}
const HEAD: u64 = 4_352;
const EXPERT: u64 = 576;
/// A card of `usable` bytes and nothing else set aside.
fn bytes_card(name: &str, usable: u64, layers: Range<usize>) -> Card {
    Card {
        usable_bytes: usable,
        ..card(name, layers.clone(), !layers.is_empty(), false)
    }
}

fn host() -> Host {
    Host {
        usable_bytes: 1 << 40,
        reserves: Vec::new(),
    }
}
/// A host that holds eight layers of eight 1 TiB experts each, with a
/// reserve of `u64::MAX − 5 W` beside them (`W`, a layer's bytes: 8
/// experts): the host's need is `u64::MAX + 3 W` and a bit, past u64
/// bytes, while the split's floor `base + 3 W` is `u64::MAX − 2 W`, under
/// it.
fn host_need_past_u64() -> (ModelTensors, Machine, u64) {
    const BIG: u64 = 1 << 40;
    let w = 8 * BIG;
    let mut model = layered(8);
    for t in model
        .tensors
        .iter_mut()
        .filter(|t| t.role == Role::RoutedExperts)
    {
        t.file_bytes = 8 * BIG;
    }
    let machine = Machine {
        cards: vec![bytes_card("stage", HEAD, 0..8)],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: Host {
                usable_bytes: u64::MAX,
                reserves: vec![("beside".into(), u64::MAX - 5 * w)],
            },
            unified: None,
        }],
    };
    (model, machine, BIG)
}
/// [`layered`]'s embedding and head over one layer of routed stacks a
/// row of `types` (gate, up, down), each named by the tail a file gives it
/// (the part is read from that tail, the layer from `layer`).
fn legged(types: &[[GgmlType; 3]]) -> ModelTensors {
    let mut model = layered(types.len());
    model.tensors.retain(|t| t.role != Role::RoutedExperts);
    for (l, parts) in types.iter().enumerate() {
        for (part, &ty) in ["gate", "up", "down"].into_iter().zip(parts) {
            let blocks = 256 / ty.blck_size().expect("a sized type");
            model.tensors.push(ModelTensor {
                name: format!("ffn_{part}_exps.weight"),
                shard: 0,
                layer: Some(l),
                role: Role::RoutedExperts,
                ty,
                dims: vec![256, 4, 8],
                file_bytes: ty.type_size().expect("a sized type") * blocks * 32,
                gathered_rows: None,
            });
        }
    }
    model
}
