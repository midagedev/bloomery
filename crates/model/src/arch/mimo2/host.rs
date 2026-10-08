//! The host tier's view of a mimo2 layer: the routed stacks, widths and
//! SwiGLU limit a [`HostLayer`] serves by, read from the hyperparameters and
//! the tensor names — the one place that wires them into a [`HostLayerSpec`].

use super::hparams::Hparams;
use super::names;
use crate::ModelError;
use crate::moe::{HostLayer, HostLayerSpec};
use crate::r8file::R8Source;

/// Layer `layer`'s routed experts as the host tier serves them, built from
/// `src` ([`HostLayer::build`]) at the plain combine: the routed experts
/// carry no clamp. A layer past the trunk runs a dense block and is refused
/// by name; which trunk layers route is the caller's run (the description's
/// routed layers), as the file states it by the tensors a layer carries.
/// Load-time only: the stacks are checked here.
pub fn layer(src: R8Source<'_>, hp: &Hparams, layer: usize) -> Result<HostLayer, ModelError> {
    if layer >= hp.n_trunk {
        return Err(ModelError::MissingTensor(format!(
            "layer {layer}'s routed stacks: a layer of the host run past the trunk"
        )));
    }
    let (gate, up, down) = (
        names::ffn_gate_exps(layer),
        names::ffn_up_exps(layer),
        names::ffn_down_exps(layer),
    );
    HostLayer::build(
        src,
        &HostLayerSpec {
            gate: &gate,
            up: &up,
            down: &down,
            n_expert: hp.n_expert,
            embd: hp.n_embd,
            ff: hp.expert_ff,
            swiglu_limit: 0.0,
        },
    )
}
