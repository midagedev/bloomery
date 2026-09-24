//! The host tier's view of a Qwen3-MoE layer: the routed stacks and widths a
//! [`HostLayer`] serves by, read from the hyperparameters and the tensor
//! names — the one place that wires them into a [`HostLayerSpec`]. A plan
//! that holds the whole model on its cards never calls it; a card too small
//! for every expert (or a larger file of this architecture) does.

use gguf::Split;

use super::hparams::Hparams;
use super::names;
use crate::ModelError;
use crate::moe::{HostLayer, HostLayerSpec};

/// The SwiGLU limit of this architecture: none. `HostLayerSpec` reads a
/// limit at or below 1e-6 as the plain `up · silu(gate)` combine, which is
/// ik's `LLM_FFN_SILU` with no clamp (build_qwen3.cpp:147).
const NO_LIMIT: f32 = 0.0;

/// Layer `layer`'s routed experts as the host tier serves them, built from
/// `split`; an error for a layer past the model. Every layer routes.
/// Load-time only: [`HostLayer::build`] checks the stacks here.
pub fn layer(split: &Split, hp: &Hparams, layer: usize) -> Result<HostLayer, ModelError> {
    if layer >= hp.n_layer {
        return Err(ModelError::Shape {
            what: "qwen3moe host layer: [layers, 1], a layer inside them",
            want_ne0: hp.n_layer,
            want_ne1: 1,
            got_ne0: layer,
            got_ne1: 1,
        });
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
            swiglu_limit: NO_LIMIT,
        },
    )
}
