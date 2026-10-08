//! The host tier's view of a V4.1 layer: the routed stacks, widths and SwiGLU
//! limit a [`HostLayer`] serves by, read from the hyperparameters and the
//! tensor names — the one place that wires them into a [`HostLayerSpec`].

use std::ops::Range;

use super::hparams::Hparams;
use super::names;
use crate::ModelError;
use crate::moe::{HostLayer, HostLayerSpec};
use crate::r8file::R8Source;

/// Layer `layer`'s routed experts as the host tier serves them, built from
/// `src` — the gate and the up from its r8 sidecar when it reads one
/// ([`HostLayer::build`]); `None` for a layer that does not route, an error
/// for a layer past the model. Load-time only: the stacks are checked here.
pub fn layer(
    src: R8Source<'_>,
    hp: &Hparams,
    layer: usize,
) -> Result<Option<HostLayer>, ModelError> {
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
        src,
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

/// The host views of the layers `run` ([`layer`] of each), in order — a host
/// run's range of routed layers, from the parse's dense lead
/// (`hp.experts.dense_lead`) on. A layer inside the run that does not route
/// is refused by name; no file the parse accepts reaches it (its dense lead
/// refuses an unrouted layer after a routed one first). Load-time only.
pub fn layers(
    src: R8Source<'_>,
    hp: &Hparams,
    run: Range<usize>,
) -> Result<Vec<HostLayer>, ModelError> {
    run.map(|l| {
        layer(src, hp, l)?.ok_or_else(|| ModelError::Shape {
            what: "deepseek41 host layers: [dense_lead, n_layer), a run of routed layers",
            want_ne0: hp.experts.dense_lead,
            want_ne1: hp.layers.len(),
            got_ne0: l,
            got_ne1: 1,
        })
    })
    .collect()
}
