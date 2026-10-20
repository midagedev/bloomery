//! The host tier's view of a Solar-Open2 layer: the routed stacks, widths and SwiGLU limit a
//! [`HostLayer`] serves by, read from the hyperparameters and the tensor names — the one place that
//! wires them into a [`HostLayerSpec`].

use super::hparams::Hparams;
use crate::ModelError;
use crate::arch::names;
use crate::moe::{HostLayer, HostLayerSpec};
use crate::r8file::R8Source;

/// Layer `layer`'s routed experts as the host tier serves them, built from `src`
/// ([`HostLayer::build`]) at the SwiGLU clamp `swiglu_limit` the description gives that layer (0:
/// plain). A layer past the file's is refused by name; which layers route is the caller's run (the
/// description's routed layers). Load-time only: the stacks are checked here.
pub fn layer(
    src: R8Source<'_>,
    hp: &Hparams,
    layer: usize,
    swiglu_limit: f32,
) -> Result<HostLayer, ModelError> {
    if layer >= hp.n_layer {
        return Err(ModelError::MissingTensor(format!(
            "layer {layer}'s routed stacks: a layer of the host run past the file's {}",
            hp.n_layer
        )));
    }
    let [gate, up, down] = names::routed_stacks(layer);
    HostLayer::build(
        src,
        &HostLayerSpec {
            gate: &gate,
            up: &up,
            down: &down,
            n_expert: hp.experts.n_expert,
            embd: hp.n_embd,
            ff: hp.experts.ff,
            swiglu_limit,
        },
    )
}
