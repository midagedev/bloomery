//! Qwen3.6's side of adaptive expert residency ([`crate::host::swap`]): the
//! model's stack table for the common file source
//! ([`crate::host::swap_source::FileSwap`]) — [`stacks`], its gate, up and
//! down.
//!
//! **Parts.** An expert is three parts in stack order: the gate and up
//! (`ff` rows of `hidden` values each) and the down (`hidden` rows of `ff`),
//! each of its layer's own type as the file holds it: the placed load uploads
//! them as file bytes in slot order (`placement::accepts::routed_rule`
//! keeps every card stack in `CardFormat::KQuant`). The shared expert is no
//! part of the routed stacks: a placed load keeps it on its own stacks
//! (`super::placed::apart_plans`).

use super::swap::{self, CardLeg};
use crate::GpuError;
use crate::host::swap_source::PlanStacks;
use model::placement::ModelTensors;
use model::placement::accepts::Program;

/// Qwen3.6's card leg ([`CardLeg`]): the qwen35moe body's arms
/// (`placement::accepts::leg_arms`: the dispatch's types and the common leg's,
/// `qwen35_leg`), which its place rule reads too.
pub(super) struct Leg35;

impl CardLeg for Leg35 {
    const WHAT: &'static str = "swap35::stacks";
    const PROGRAM: Program = Program::Qwen35Body;
}

/// The stacks of `model`'s layers, each layer's types read from its tensors,
/// each layer `carded` names checked against [`Leg35`] ([`swap::stacks`]).
/// Refused by name: a layer without all three stacks, and a carded layer the
/// leg refuses (a gate or up of a type no card gate·up launch reads among
/// them). Load-time only.
pub(super) fn stacks(
    model: &ModelTensors,
    carded: impl Fn(usize) -> Result<bool, GpuError>,
) -> Result<PlanStacks, GpuError> {
    swap::stacks::<Leg35>(model, carded)
}
