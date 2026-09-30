//! The host tier's view of a glm5next layer: the routed stacks, widths and
//! SwiGLU limit a [`HostLayer`] serves by, read from the hyperparameters and
//! the tensor names — the one place that wires them into a [`HostLayerSpec`].

use super::hparams::Hparams;
use super::names;
use crate::ModelError;
use crate::moe::{HostLayer, HostLayerSpec};
use crate::r8file::R8Source;

/// Layer `layer`'s routed experts as the host tier serves them — a trunk
/// layer's, or the next-token layer's when its load carries it — built from
/// `src` ([`HostLayer::build`]) with the routed limit (`swiglu_clamp_exp`);
/// `None` for a dense layer, an error for a layer past the file's. Load-time
/// only: the stacks are checked here.
pub fn layer(
    src: R8Source<'_>,
    hp: &Hparams,
    layer: usize,
) -> Result<Option<HostLayer>, ModelError> {
    if layer >= hp.n_layer {
        return Err(ModelError::Shape {
            what: "glm5next host layer: [layers, 1], a layer of the file",
            want_ne0: hp.n_layer,
            want_ne1: 1,
            got_ne0: layer,
            got_ne1: 1,
        });
    }
    if layer < hp.dense_lead {
        return Ok(None);
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
            swiglu_limit: hp.limit_exp[layer],
        },
    )
    .map(Some)
}
