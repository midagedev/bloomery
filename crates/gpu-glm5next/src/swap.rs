//! GLM's side of adaptive expert residency ([`bloomery_gpu::host::swap`]):
//! the model's [`FileStacks`] for the common file source
//! (`bloomery_gpu::host::swap_source::FileSwap`) — [`Glm5Stacks`], each
//! layer's gate, up and down — with the machine's live delay and deadline.
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
/// the card's link, an expert's copy running under the next one's memcpy
/// (the ring holds [`bloomery_gpu::host::swap::RING_SLOTS`] experts). The
/// widest expert a card layer holds is gate and up Q4_K with a Q5_K down;
/// one thread's copy rate, taken as its share of the host's measured
/// all-core read rate, and the link's measured pinned H2D rate (the machine
/// facts doc) put a whole planning pass of those inside two of GLM's decode
/// steps both at their measured wall and at the prediction for the step
/// [derived], so two passes hold it. The staging runs in the host leg's wait
/// window, a share of the step this derivation does not price: a late
/// staging makes the host wait at the landing and a late copy the engine
/// stream, and neither moves the boundary a flip lands at.
pub const LIVE_DELAY: u64 = 2;

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
