//! The host tier's view of a V4.1 layer: the routed stacks, widths and SwiGLU
//! limit a [`HostLayer`] serves by, read from the hyperparameters and the
//! tensor names — the one place that wires them into a [`HostLayerSpec`].

use gguf::Split;

use super::hparams::Hparams;
use super::names;
use crate::ModelError;
use crate::moe::{HostLayer, HostLayerSpec};

/// Layer `layer`'s routed experts as the host tier serves them, built from
/// `split`; `None` for a layer that does not route, an error for a layer past
/// the model. Load-time only: [`HostLayer::build`] checks the stacks here.
pub fn layer(split: &Split, hp: &Hparams, layer: usize) -> Result<Option<HostLayer>, ModelError> {
    let kind = hp.layers.get(layer).ok_or(ModelError::Shape {
        what: "deepseek41 host layer: [layers, 1], a layer inside them",
        want_ne0: hp.layers.len(),
        want_ne1: 1,
        got_ne0: layer,
        got_ne1: 1,
    })?;
    if !kind.routed {
        return Ok(None);
    }
    let (gate, up, down) = (
        names::ffn_gate_exps(layer),
        names::ffn_up_exps(layer),
        names::ffn_down_exps(layer),
    );
    HostLayer::build(
        split,
        &HostLayerSpec {
            gate: &gate,
            up: &up,
            down: &down,
            n_expert: hp.experts.n_expert,
            embd: hp.n_embd,
            ff: hp.experts.ff,
            swiglu_limit: kind.swiglu_limit,
        },
    )
    .map(Some)
}
