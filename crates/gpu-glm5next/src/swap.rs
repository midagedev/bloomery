//! GLM's side of adaptive expert residency ([`bloomery_gpu::host::swap`]):
//! the model's [`FileStacks`] for the common file source
//! (`bloomery_gpu::host::swap_source::FileSwap`) — [`Glm5Stacks`], each
//! layer's gate, up and down — with the machine's live delay and deadline.
//!
//! **Parts.** An expert is three parts in stack order: gate and up (`ff`
//! rows of `n_embd` values each) and down (`n_embd` rows of `ff`). The
//! parts are each layer's own as the file holds them: most layers' gate and
//! up are Q4_K with a Q5_K down, one layer's three stacks are all Q5_K, and
//! the layers whose down is Q6_K keep their experts whole on the host
//! ([`model::arch::glm5next::place::card_routed`] admits Q4_K and Q5_K
//! alone), so a card layer is one of the first two kinds. On the card each
//! part of slot `s` is the file's bytes of that expert at byte `s · part` of
//! its layer's stack, which the placed load uploads as file bytes in slot
//! order — no convert step, and no r8 sidecar (that format is Q3_K's alone,
//! and no GLM stack is one), so a staged part is its slot's bytes as it
//! stands.

use std::sync::Arc;
use std::time::Duration;

use bloomery_gpu::GpuError;
use bloomery_gpu::host::swap_source::{Convert, FileStacks};
use cuda_core::CudaContext;
use gguf::quant::GgmlType;
use model::arch::glm5next::names;
use model::arch::glm5next::place::PlanInputs;

/// Passes from the boundary that makes a flip to the one it lands at. The
/// victims are host-resident (the churn pool), so a flip waits on no NVMe
/// read, only on its staging and its copy: a planning pass makes at most the
/// rule's `cap` flips ([`runtime::swaprule::SwapParams::mid`]), each one
/// expert's memcpy into the pinned ring on one thread and one H2D copy over
/// the card's link. The widest expert a layer holds is the one layer whose
/// three stacks are all Q5_K; one thread's copy rate and the link's pinned
/// H2D rate (the machine facts doc's measured values) put a whole planning
/// pass of those at about 1.9 of GLM's measured decode steps' wall and about
/// 2.2 of the same line's prediction for the step [derived], so three steps'
/// wall holds the pass at either end and two hold it only at the measured
/// one. The staging runs in the host leg's wait window, and a late copy only
/// makes the engine stream wait at the landing.
pub const LIVE_DELAY: u64 = 3;

/// The bound on every host wait of the machine. Every boundary follows a
/// pass's readback, so the engine stream has drained and a wait is for the
/// staging thread and the copy stream alone: at most a planning pass's
/// `cap` experts ([`runtime::swaprule::SwapParams::mid`]), far inside this
/// bound.
pub const DEADLINE: Duration = Duration::from_secs(30);

/// GLM's [`FileStacks`]: each layer's gate, up and down by name, with the
/// types the file holds them in — read from the plan's tensors at load, the
/// file the one owner of the split. No convert step: the card's stacks are
/// the file's bytes, and the r8 sidecar is refused
/// ([`Glm5Stacks::open`]).
pub struct Glm5Stacks {
    /// Per trunk layer, its three stacks' types in stack order; `None` on a
    /// layer that holds none of them (a dense block).
    types: Vec<Option<[GgmlType; 3]>>,
    /// The layers the body's slot map holds (the host tier's run, every
    /// routed layer): the machine's per-layer lists cover these, not the
    /// card's whole range the dense lead is part of.
    map_layers: usize,
}

impl Glm5Stacks {
    /// The stacks of the layers `inputs` describes, their types read from
    /// its tensors, for a body whose slot map holds `map_layers` layers.
    /// Refused by name: a layer that holds some of its three stacks but not
    /// all.
    pub fn of(inputs: &PlanInputs, map_layers: usize) -> Result<Glm5Stacks, GpuError> {
        const WHAT: &str = "Glm5Stacks::of";
        let find = |name: &str| {
            inputs
                .model
                .tensors
                .iter()
                .find(|t| t.name == name)
                .map(|t| t.ty)
        };
        let mut types = Vec::with_capacity(inputs.model.layers);
        for l in 0..inputs.model.layers {
            let names = [
                names::ffn_gate_exps(l),
                names::ffn_up_exps(l),
                names::ffn_down_exps(l),
            ];
            let found: [Option<GgmlType>; 3] = [find(&names[0]), find(&names[1]), find(&names[2])];
            types.push(match found {
                [None, None, None] => None,
                [Some(gate), Some(up), Some(down)] => Some([gate, up, down]),
                _ => {
                    let (name, _) = names
                        .iter()
                        .zip(found)
                        .find(|(_, t)| t.is_none())
                        .expect("a layer with a missing stack names it");
                    return Err(GpuError::Tensor {
                        what: WHAT,
                        name: name.clone(),
                        need: "all three routed stacks: a layer holds either all of them or none",
                    });
                }
            });
        }
        Ok(Glm5Stacks { types, map_layers })
    }
}

impl FileStacks for Glm5Stacks {
    fn names(&self, layer: usize) -> Vec<String> {
        vec![
            names::ffn_gate_exps(layer),
            names::ffn_up_exps(layer),
            names::ffn_down_exps(layer),
        ]
    }

    fn types(&self, layer: usize) -> &[GgmlType] {
        self.types
            .get(layer)
            .and_then(Option::as_ref)
            .map_or(&[], |t| t.as_slice())
    }

    fn map_layers(&self) -> Option<usize> {
        Some(self.map_layers)
    }

    fn open(
        &self,
        _dims: &[u64],
        _parts: &[usize],
        sidecar: bool,
        _ctx: &Arc<CudaContext>,
    ) -> Result<Option<Arc<dyn Convert>>, GpuError> {
        if sidecar {
            return Err(GpuError::State {
                what: "Glm5Stacks::open",
                missing: "no r8 sidecar for GLM's stacks: the sidecar's format is Q3_K's alone \
                          and no stack of the file is one (BLOOMERY_R8=off)",
            });
        }
        Ok(None)
    }
}
