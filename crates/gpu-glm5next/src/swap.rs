//! GLM's side of adaptive expert residency ([`bloomery_gpu::host::swap`]):
//! the model's [`FileStacks`] for the common file source
//! (`bloomery_gpu::host::swap_source::FileSwap`) — [`Glm5Stacks`], each
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

use std::sync::Arc;

use bloomery_gpu::GpuError;
use bloomery_gpu::host::swap_source::{Convert, FileStacks, PlanStacks, StackFacts};
use cuda_core::CudaContext;
use gguf::quant::GgmlType;
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

/// The r8 sidecar's refusal ([`FileStacks::open`]): the sidecar's format is
/// Q3_K's alone and no stack of the file is one.
fn no_sidecar() -> GpuError {
    GpuError::State {
        what: "Glm5Stacks::open",
        missing: "no r8 sidecar for GLM's stacks: the sidecar's format is Q3_K's alone \
                  and no stack of the file is one (BLOOMERY_R8=off)",
    }
}

/// GLM's [`FileStacks`] for the common file source: the common three-stack
/// table ([`PlanStacks`]) with GLM's facts — a layer may hold none of its
/// stacks (a dense block), the place rule owns which stacks take card slots
/// (no type check of its own here), and the slot map's run of layers with
/// its unrouted next-token layer. The card holds the file's bytes; the file
/// is the one owner of the split.
pub struct Glm5Stacks(PlanStacks);

impl Glm5Stacks {
    /// The stacks of the layers `inputs` describes, their types read from
    /// its tensors, for a body whose slot map holds `map_layers` layers
    /// (the host tier's run: every routed layer, and on a NextN load the
    /// next-token layer after them — the machine's per-layer lists cover
    /// these, not the card's whole range the dense lead is part of), the
    /// last of them the next-token layer `nextn` on a NextN load (the map's
    /// last layer, which no pass routes — the draft's walks serve it through
    /// the batch port, which notes no id — and which holds no card slot).
    /// Refused by name: a layer that holds some of its three stacks but not
    /// all.
    pub fn of(
        inputs: &PlanInputs,
        map_layers: usize,
        nextn: Option<usize>,
    ) -> Result<Glm5Stacks, GpuError> {
        PlanStacks::of(
            &inputs.model,
            StackFacts {
                what: "Glm5Stacks::of",
                names: stack_names,
                check: None,
                empty: true,
                map_layers: Some(map_layers),
                unrouted: nextn.into_iter().collect(),
                sidecar: no_sidecar,
            },
        )
        .map(Glm5Stacks)
    }
}

impl FileStacks for Glm5Stacks {
    fn names(&self, layer: usize) -> Vec<String> {
        self.0.names(layer)
    }

    fn types(&self, layer: usize) -> &[GgmlType] {
        self.0.types(layer)
    }

    fn map_layers(&self) -> Option<usize> {
        self.0.map_layers()
    }

    fn unrouted(&self) -> Vec<usize> {
        self.0.unrouted()
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
