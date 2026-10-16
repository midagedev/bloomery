//! MiMo's side of adaptive expert residency ([`bloomery_gpu::host::swap`]):
//! the model's stack table for the common file source
//! (`bloomery_gpu::host::swap_source::FileSwap`) — [`stacks`], each routed
//! layer's gate, up and down — with the machine's live delay and deadline
//! ([`LIVE_DELAY`], [`DEADLINE`]) and the rows a pass runs ([`LOAD_ROWS`]).
//!
//! **Facts.**
//! - The dense layer 0 holds none of the three stacks ([`StackFacts::empty`]);
//!   every other trunk layer holds all three.
//! - The slot map holds the routed run of the description
//!   ([`runtime::layer::hosted`]): [`StackFacts::map_layers`] is that run's
//!   length.
//! - No layer is unrouted: [`model::arch::mimo2::place::PlanInputs::read`]
//!   refuses a file whose next-token layers carry tensors, so the load has
//!   no draft layer.
//! - The card leg's types are the kernel table's
//!   ([`model::arch::mimo2::place::card_routed`], from
//!   `placement::kernels::card_runs`): a layer takes card slots only when
//!   every one of its stacks is a type the leg runs, and the leg reads its gate
//!   and up in one type ([`card_types`]).
//!
//! **Parts.** An expert is three parts in stack order: gate and up (`ff` rows
//! of `n_embd` values each) and down (`n_embd` rows of `ff`), each of its
//! layer's own type as the file holds it. On the card each part of slot `s` is
//! the file's bytes of that expert at byte `s · part` of its layer's stack,
//! which the placed load uploads as file bytes in slot order — no convert
//! step, and no r8 sidecar (that format is Q3_K's alone), so a staged part is
//! its slot's bytes as it stands.

use bloomery_gpu::GpuError;
use bloomery_gpu::host::swap_source::{PlanStacks, StackFacts};
use gguf::GgmlType;
use model::arch::mimo2::place::{PlanInputs, card_routed};
use model::arch::names::routed_stacks;
use runtime::layer::hosted;

use crate::body::slots::MAX_ROWS;

pub use bloomery_gpu::host::swap_source::{DEADLINE, LIVE_DELAY};

/// What the stack table's refusals name.
const WHAT: &str = "mimo2::swap::stacks";

/// The most rows one pass runs, which the machine's tally holds: the pass of
/// several slots' rows ([`MAX_ROWS`]).
pub const LOAD_ROWS: usize = MAX_ROWS;

/// Layer `l`'s card-leg contract: a layer whose stacks are all types the leg
/// runs ([`card_routed`]) holds its gate and up in one type (the leg refuses
/// a pair of two at load); a layer with another type stays on the host.
fn card_types(l: usize, [gate, up, down]: [GgmlType; 3]) -> Result<(), GpuError> {
    let carded = [gate, up, down].iter().all(|&ty| card_routed(ty).is_some());
    if carded && gate != up {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "layer {l}'s gate {gate} and up {up}: the card leg runs a gate and an up of one type"
            ),
        });
    }
    Ok(())
}

/// The stacks of the layers `inputs` describes, their types read from its
/// tensors, over MiMo's facts (module doc). Refused by name: a layer that
/// holds some of its three stacks but not all, a layer [`card_types`]
/// refuses, and a description whose routed layers are not one run.
/// Load-time only.
pub fn stacks(inputs: &PlanInputs) -> Result<PlanStacks, GpuError> {
    let run = hosted(&inputs.spec.layers).map_err(|e| GpuError::Shape {
        what: WHAT,
        detail: format!("the routed layers: {e}"),
    })?;
    PlanStacks::of(
        &inputs.model,
        StackFacts {
            what: WHAT,
            names: routed_stacks,
            check: Some(card_types),
            empty: true,
            map_layers: Some(run.len()),
            unrouted: Vec::new(),
        },
    )
}
