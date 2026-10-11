//! Qwen3.8's side of adaptive expert residency ([`crate::host::swap`]): the
//! model's stack table for the common file source
//! ([`crate::host::swap_source::FileSwap`]) — [`stacks`], its gate, up and
//! down — with the machine's live delay and deadline
//! ([`crate::host::swap_source::LIVE_DELAY`]).
//!
//! **Parts.** An expert is three parts in stack order: the gate and up
//! (`geo::FF` rows of `geo::HIDDEN` values each) and the down (`geo::HIDDEN`
//! rows of `geo::FF`), each of its layer's own type as the file holds it —
//! most layers a Q4_K gate and up and a Q5_1 down, some a Q5_K gate and up or
//! a Q8_0 down, and the UD-Q3_K_XL file's an IQ3_XXS or IQ4_XS gate and up
//! with an IQ4_NL down
//! ([`stacks`] reads each layer's from the plan). On
//! the card each part of slot `s` is the file's bytes of that expert at byte
//! `s · part` of its layer's stack, which the placed load uploads as file
//! bytes in slot order ([`place::card_routed`] keeps all three stacks in
//! `CardFormat::KQuant`, and `super::card38` reads them exactly so). Nothing
//! converts: a staged part is its slot's bytes as they are, so the r8
//! sidecar, which exists for Q3_K stacks alone (`qdot::repack_q3k_r8`), is
//! undefined input here, refused by name.

use super::swap::{self, CardLeg};
use crate::GpuError;
use crate::host::swap_source::PlanStacks;
use gguf::quant::GgmlType;
use model::placement::ModelTensors;

pub use crate::host::swap_source::{DEADLINE, LIVE_DELAY};

/// Qwen3.8's card leg ([`CardLeg`]): the types it runs a routed stack in, by
/// its place: the gate and up Q4_K, Q5_K, IQ3_XXS or IQ4_XS, the down Q5_1,
/// Q8_0 or IQ4_NL ([`place::card_routed`]).
pub(super) struct Leg38;

impl CardLeg for Leg38 {
    const WHAT: &'static str = "swap38::stacks";

    fn gate_up(ty: GgmlType) -> bool {
        matches!(
            ty,
            GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::IQ3_XXS | GgmlType::IQ4_XS
        )
    }

    fn down(ty: GgmlType) -> bool {
        matches!(ty, GgmlType::Q5_1 | GgmlType::Q8_0 | GgmlType::IQ4_NL)
    }
}

/// Qwen3.8's stack table: the common one ([`PlanStacks`]) over this family's
/// facts ([`swap::stacks`]).
pub type Qwen38Stacks = PlanStacks;

/// The stacks of `model`'s layers, each layer's types read from its tensors
/// and checked against [`Leg38`]. Refused by name: a layer without all three
/// stacks, and a layer the leg refuses. Load-time only.
pub fn stacks(model: &ModelTensors) -> Result<Qwen38Stacks, GpuError> {
    swap::stacks::<Leg38>(model)
}
