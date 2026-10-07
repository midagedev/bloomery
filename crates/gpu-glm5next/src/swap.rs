//! GLM's side of adaptive expert residency ([`bloomery_gpu::host::swap`]):
//! the model's stack table for the common file source
//! (`bloomery_gpu::host::swap_source::FileSwap`) — [`stacks`], each
//! layer's gate, up and down — with the machine's live delay and deadline
//! ([`LIVE_DELAY`], [`DEADLINE`]).
//!
//! **Parts.** An expert is three parts in stack order: gate and up (`ff`
//! rows of `n_embd` values each) and down (`n_embd` rows of `ff`). The
//! parts are each layer's own as the file holds them, and the file's card
//! layers are all of one kind, gate and up Q4_K with a Q5_K down:
//! [`model::arch::glm5next::place::card_routed`] admits Q4_K and Q5_K alone
//! and a layer takes card slots only when every one of its stacks is
//! routable, so the layers whose down is Q6_K — among them the one layer
//! whose gate and up are Q5_K — keep their experts whole on the host. On the
//! card each part of slot `s` is the file's bytes of that expert at byte
//! `s · part` of its layer's stack, which the placed load uploads as file
//! bytes in slot order — no convert step, and no r8 sidecar (that format is
//! Q3_K's alone, and no GLM stack is one), so a staged part is its slot's
//! bytes as it stands.

use bloomery_gpu::GpuError;
use bloomery_gpu::host::swap_source::{PlanStacks, StackFacts};
use model::arch::glm5next::names;
use model::arch::glm5next::place::PlanInputs;

pub use bloomery_gpu::host::swap_source::{DEADLINE, LIVE_DELAY};

/// Layer `l`'s routed gate, up and down names.
fn stack_names(l: usize) -> [String; 3] {
    [
        names::ffn_gate_exps(l),
        names::ffn_up_exps(l),
        names::ffn_down_exps(l),
    ]
}

/// The r8 sidecar's refusal (`FileStacks::open`): the sidecar's format is
/// Q3_K's alone and no stack of the file is one.
fn no_sidecar() -> GpuError {
    GpuError::State {
        what: "glm5next::swap::stacks",
        missing: "no r8 sidecar for GLM's stacks: the sidecar's format is Q3_K's alone \
                  and no stack of the file is one (BLOOMERY_R8=off)",
    }
}

/// The stacks of the layers `inputs` describes, their types read from its
/// tensors, over GLM's facts: a layer may hold none of its stacks (a dense
/// block); the place rule owns which stacks take card slots (no type check
/// here); the slot map holds `map_layers` layers (every routed layer, and on
/// a NextN load the next-token layer after them), the last of them the
/// unrouted next-token layer `nextn` on a NextN load (the draft's walks serve
/// it through the batch port, which notes no id; it holds no card slot).
/// Refused by name: a layer that holds some of its three stacks but not all.
pub fn stacks(
    inputs: &PlanInputs,
    map_layers: usize,
    nextn: Option<usize>,
) -> Result<PlanStacks, GpuError> {
    PlanStacks::of(
        &inputs.model,
        StackFacts {
            what: "glm5next::swap::stacks",
            names: stack_names,
            check: None,
            empty: true,
            map_layers: Some(map_layers),
            unrouted: nextn.into_iter().collect(),
            sidecar: no_sidecar,
        },
    )
}
