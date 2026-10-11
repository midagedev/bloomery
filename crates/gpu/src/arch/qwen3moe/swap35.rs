//! Qwen3.6's side of adaptive expert residency ([`crate::host::swap`]): the
//! model's stack table for the common file source
//! ([`crate::host::swap_source::FileSwap`]) — [`stacks`], its gate, up and
//! down.
//!
//! **Parts.** An expert is three parts in stack order: the gate and up
//! (`ff` rows of `hidden` values each) and the down (`hidden` rows of `ff`),
//! each of its layer's own type as the file holds it: the placed load uploads
//! them as file bytes in slot order (`model::arch::qwen3moe::place::card_routed`
//! keeps every card stack in `CardFormat::KQuant`). The shared expert is no
//! part of the routed stacks: a placed load keeps it on its own stacks
//! (`super::placed::apart_plans`).

use super::swap::{self, CardLeg};
use crate::GpuError;
use crate::host::swap_source::PlanStacks;
use gguf::quant::GgmlType;
use model::arch::coverage::{qwen35_down, qwen35_gate_up};
use model::placement::ModelTensors;

/// Qwen3.6's card leg ([`CardLeg`]): the card kernel table's entries the
/// family's FFN dispatch launches — [`qwen35_gate_up`] (a Q4_K or Q5_K gate
/// and up) and [`qwen35_down`] (a Q4_K, Q5_K or Q6_K down) — which are the
/// types its place rule loads on the card
/// (`model::arch::qwen3moe::place::card_routed`).
pub(super) struct Leg35;

impl CardLeg for Leg35 {
    const WHAT: &'static str = "swap35::stacks";

    fn gate_up(ty: GgmlType) -> bool {
        qwen35_gate_up(ty).is_some()
    }

    fn down(ty: GgmlType) -> bool {
        qwen35_down(ty).is_some()
    }
}

/// The stacks of `model`'s layers, each layer's types read from its tensors,
/// each layer `carded` names checked against [`Leg35`] ([`swap::stacks`]).
/// Refused by name: a layer without all three stacks, and a carded layer the
/// leg refuses (a Q6_K gate or up among them). Load-time only.
pub(super) fn stacks(
    model: &ModelTensors,
    carded: impl Fn(usize) -> Result<bool, GpuError>,
) -> Result<PlanStacks, GpuError> {
    swap::stacks::<Leg35>(model, carded)
}
