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

use crate::GpuError;
use crate::host::swap_source::{PlanStacks, StackFacts};
use gguf::quant::GgmlType;
use model::placement::ModelTensors;

pub use crate::host::swap_source::{DEADLINE, LIVE_DELAY};

/// The types the card leg runs a routed stack in, by its place: the gate and
/// up Q4_K, Q5_K, IQ3_XXS or IQ4_XS, the down Q5_1, Q8_0 or IQ4_NL
/// ([`place::card_routed`]).
const GATE_UP: [GgmlType; 4] = [
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::IQ3_XXS,
    GgmlType::IQ4_XS,
];
const DOWN: [GgmlType; 3] = [GgmlType::Q5_1, GgmlType::Q8_0, GgmlType::IQ4_NL];

/// Layer `l`'s card-leg contract: a layer whose stacks are all card types
/// ([`place::card_routed`]) holds a gate and an up of one [`GATE_UP`] type and
/// a [`DOWN`] down; a layer with another type stays on the host.
fn card_types(l: usize, [gate, up, down]: [GgmlType; 3]) -> Result<(), GpuError> {
    let carded = [gate, up, down]
        .iter()
        .all(|&ty| model::arch::qwen35moe::place::card_routed(ty).is_some());
    if carded && (gate != up || !GATE_UP.contains(&gate) || !DOWN.contains(&down)) {
        return Err(GpuError::shape(
            "swap38::stacks",
            format!(
                "layer {l}'s gate {gate}, up {up} and down {down}: the card leg runs a \
                 gate and an up of one type, {GATE_UP:?}, and a down of {DOWN:?}"
            ),
        ));
    }
    Ok(())
}

/// Layer `l`'s routed gate, up and down names.
fn stack_names(l: usize) -> [String; 3] {
    [
        model::arch::qwen35moe::names::ffn_gate_exps(l),
        model::arch::qwen35moe::names::ffn_up_exps(l),
        model::arch::qwen35moe::names::ffn_down_exps(l),
    ]
}

/// The r8 sidecar's refusal ([`crate::host::swap_source::FileStacks::open`]):
/// undefined input for this family, not a conversion.
fn no_sidecar() -> GpuError {
    GpuError::shape(
        "swap38::stacks",
        "an r8 sidecar beside a qwen4exp file: the sidecar holds Q3_K \
         stacks (qdot::repack_q3k_r8), and this model's routed gate and up \
         are Q4_K, Q5_K, IQ3_XXS or IQ4_XS and its down Q5_1, Q8_0 or IQ4_NL \
         — undefined input, not a \
         conversion",
    )
}

/// Qwen3.8's stack table: the common one ([`PlanStacks`]) over this family's
/// facts — every layer holds all three stacks, each checked against the card
/// leg's types ([`card_types`]), the whole card range the map, every layer
/// routed.
pub type Qwen38Stacks = PlanStacks;

/// The stacks of `model`'s layers, each layer's types read from its tensors.
/// Refused by name: a layer without all three stacks, and a layer
/// [`card_types`] refuses. Load-time only.
pub fn stacks(model: &ModelTensors) -> Result<Qwen38Stacks, GpuError> {
    PlanStacks::of(
        model,
        StackFacts {
            what: "swap38::stacks",
            names: stack_names,
            check: Some(card_types),
            empty: false,
            map_layers: None,
            unrouted: Vec::new(),
            sidecar: no_sidecar,
        },
    )
}
