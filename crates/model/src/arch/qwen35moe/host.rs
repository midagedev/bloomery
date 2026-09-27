//! The host tier's view of a qwen4exp layer: the routed stacks and widths a
//! [`HostLayer`] serves by, read from the hyperparameters and the tensor
//! names — the one place that wires them into a [`HostLayerSpec`]. Every
//! layer routes, and the routed SwiGLU has no limit.
//!
//! Each stack's type is the file's, per layer: the host serves a layer only
//! when qdot fuses all three of its stacks at their row widths, and
//! [`HostLayer::build`] refuses any other by name (the stack, its type and
//! `k`) — never a slower path in its place.

use std::fmt;
use std::ops::Range;

use super::hparams::Hparams;
use super::names;
use crate::ModelError;
use crate::moe::{HostLayer, HostLayerSpec};
use crate::r8file::R8Source;

/// Layer `layer`'s routed experts as the host tier serves them, built from
/// `src` ([`HostLayer::build`]); an error for a layer past the model, and
/// [`HostLayer::build`]'s refusal for a stack no fused kernel serves.
/// Load-time only: the stacks are checked here.
pub fn layer(src: R8Source<'_>, hp: &Hparams, layer: usize) -> Result<HostLayer, ModelError> {
    if layer >= hp.n_layer {
        return Err(ModelError::Shape {
            what: "qwen4exp host layer: [layers, 1], a layer of the model",
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

/// Every layer of a host run whose stacks were refused, each with its
/// refusal: the whole list, not the first.
#[derive(Debug)]
pub struct RefusedLayers {
    pub layers: Vec<(usize, ModelError)>,
}

impl fmt::Display for RefusedLayers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list: Vec<String> = self
            .layers
            .iter()
            .map(|(l, why)| format!("layer {l}: {why}"))
            .collect();
        write!(
            f,
            "{} layer(s) of the qwen4exp host run cannot be served: {}",
            self.layers.len(),
            list.join("; ")
        )
    }
}

impl std::error::Error for RefusedLayers {}

// A host tier boxes the refusal as its load's error source.
const _: fn() = || {
    fn boxable<T: std::error::Error + Send + Sync + 'static>() {}
    boxable::<RefusedLayers>();
};

/// The host views of the layers `run` ([`layer`] of each), in order; every
/// layer is tried and the run is refused with every refused layer listed.
/// Load-time only.
pub fn layers(
    src: R8Source<'_>,
    hp: &Hparams,
    run: Range<usize>,
) -> Result<Vec<HostLayer>, RefusedLayers> {
    let mut views = Vec::with_capacity(run.len());
    let mut refused = Vec::new();
    for l in run {
        match layer(src, hp, l) {
            Ok(v) => views.push(v),
            Err(e) => refused.push((l, e)),
        }
    }
    if refused.is_empty() {
        Ok(views)
    } else {
        Err(RefusedLayers { layers: refused })
    }
}
