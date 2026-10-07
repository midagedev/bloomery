//! Qwen3.8's side of adaptive expert residency ([`crate::host::swap`]): the
//! model's [`FileStacks`] for the common file source
//! ([`crate::host::swap_source::FileSwap`]) — [`Qwen38Stacks`], its gate, up
//! and down — with the machine's live delay and deadline
//! ([`crate::host::swap_source::LIVE_DELAY`]).
//!
//! **Parts.** An expert is three parts in stack order: the gate and up
//! (`geo::FF` rows of `geo::HIDDEN` values each) and the down (`geo::HIDDEN`
//! rows of `geo::FF`), each of its layer's own type as the file holds it —
//! most layers a Q4_K gate and up and a Q5_1 down, some a Q5_K gate and up or
//! a Q8_0 down, and the UD-Q3_K_XL file's an IQ3_XXS or IQ4_XS gate and up
//! with an IQ4_NL down
//! ([`Qwen38Stacks::of`] reads each layer's from the plan). On
//! the card each part of slot `s` is the file's bytes of that expert at byte
//! `s · part` of its layer's stack, which the placed load uploads as file
//! bytes in slot order ([`place::card_routed`] keeps all three stacks in
//! `CardFormat::KQuant`, and `super::card38` reads them exactly so). Nothing
//! converts: a staged part is its slot's bytes as they are, so the r8
//! sidecar, which exists for Q3_K stacks alone (`qdot::repack_q3k_r8`), is
//! undefined input here, refused by name.

use std::sync::Arc;

use crate::GpuError;
use crate::host::swap_source::{Convert, FileStacks, PlanStacks, StackFacts};
use cuda_core::CudaContext;
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

/// Layer `l`'s card-leg contract on its stacks' types, the check
/// [`Qwen38Stacks::of`] runs layer by layer: a layer whose stacks are all
/// card types ([`place::card_routed`]), so that the plan puts its experts on
/// the card, must hold them as the leg runs them — a gate and an up of one
/// type, Q4_K, Q5_K, IQ3_XXS or IQ4_XS, and a down Q5_1, Q8_0 or IQ4_NL. A
/// layer with a stack of another type keeps its experts on the host and is
/// listed as the file holds it.
fn card_types(l: usize, [gate, up, down]: [GgmlType; 3]) -> Result<(), GpuError> {
    let carded = [gate, up, down]
        .iter()
        .all(|&ty| model::arch::qwen35moe::place::card_routed(ty).is_some());
    if carded && (gate != up || !GATE_UP.contains(&gate) || !DOWN.contains(&down)) {
        return Err(GpuError::shape(
            "Qwen38Stacks::of",
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

/// The r8 sidecar's refusal ([`FileStacks::open`]): the sidecar holds Q3_K
/// stacks (`qdot::repack_q3k_r8`), and this model's routed gate and up are
/// Q4_K, Q5_K, IQ3_XXS or IQ4_XS and its down Q5_1, Q8_0 or IQ4_NL —
/// undefined input, not a conversion.
fn no_sidecar() -> GpuError {
    GpuError::shape(
        "Qwen38Stacks::open",
        "an r8 sidecar beside a qwen4exp file: the sidecar holds Q3_K \
         stacks (qdot::repack_q3k_r8), and this model's routed gate and up \
         are Q4_K, Q5_K, IQ3_XXS or IQ4_XS and its down Q5_1, Q8_0 or IQ4_NL \
         — undefined input, not a \
         conversion",
    )
}

/// Qwen3.8's [`FileStacks`] for the common file source: the common three-stack
/// table ([`PlanStacks`]) with this family's facts — every layer holds all
/// three stacks, each complete layer checked against the card leg's types
/// ([`card_types`]), the whole card range the map, every layer routed, and
/// no convert step — the card holds the file's bytes, so a staged part is
/// its slot's. A model with no shape contract of its own on the stacks; the
/// load's own checks ([`crate::host::swap_source::FileSwap::new`],
/// `super::card38`) hold the parts and rows.
pub struct Qwen38Stacks(PlanStacks);

impl Qwen38Stacks {
    /// The stacks of the `model.layers` layers of `model`, each layer's
    /// types read from its tensors. Refused by name: a layer without all
    /// three stacks, and a layer whose stacks are all card types
    /// ([`place::card_routed`]), so that the plan puts its experts on the
    /// card, but not as the card leg runs them — a gate and an up of one
    /// type, Q4_K, Q5_K, IQ3_XXS or IQ4_XS, and a down Q5_1, Q8_0 or
    /// IQ4_NL. A layer with a stack of
    /// another type keeps its experts on the host and is listed as the file
    /// holds it. Load-time only.
    pub fn of(model: &ModelTensors) -> Result<Qwen38Stacks, GpuError> {
        PlanStacks::of(
            model,
            StackFacts {
                what: "Qwen38Stacks::of",
                names: stack_names,
                check: Some(card_types),
                empty: false,
                map_layers: None,
                unrouted: Vec::new(),
                sidecar: no_sidecar,
            },
        )
        .map(Qwen38Stacks)
    }

    /// Layer `l`'s gate·up type and down type, as [`FileStacks::types`] lists
    /// them; `None` past the model's layers.
    #[must_use]
    pub fn pair(&self, l: usize) -> Option<(GgmlType, GgmlType)> {
        self.0.pair(l)
    }
}

impl FileStacks for Qwen38Stacks {
    fn names(&self, layer: usize) -> Vec<String> {
        self.0.names(layer)
    }

    /// Layer `layer`'s own types as the plan holds them; none past the
    /// model's layers.
    fn types(&self, layer: usize) -> &[GgmlType] {
        self.0.types(layer)
    }

    fn open(
        &self,
        dims: &[u64],
        parts: &[usize],
        sidecar: bool,
        ctx: &Arc<CudaContext>,
    ) -> Result<Option<Arc<dyn Convert>>, GpuError> {
        self.0.open(dims, parts, sidecar, ctx)
    }
}
