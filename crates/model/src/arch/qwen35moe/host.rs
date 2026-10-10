//! The host tier's view of a routed layer of the qwen3moe family (qwen4exp,
//! qwen35moe, qwen3moe): the routed stacks and widths a [`HostLayer`] serves
//! by, read from the hyperparameters ([`RoutedDims`]) and the tensor names —
//! the one place that wires them into a [`HostLayerSpec`]. Every layer
//! routes, and the routed SwiGLU has no limit.
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
    routed_layer(src, RoutedDims::of(hp), layer)
}

/// What a family layer's routed stacks are, as the host tier reads them:
/// the trunk's layers (a next-token layer's experts are never served), the
/// experts a layer, the model width and an expert's width.
/// The tensor names (`blk.L.ffn_{gate,up,down}_exps.weight`) are the same in
/// every file of the qwen3moe family (qwen3moe, qwen35moe, qwen4exp), and
/// the routed SwiGLU has no limit in any of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoutedDims {
    pub n_layer: usize,
    pub n_expert: usize,
    pub embd: usize,
    pub ff: usize,
}

impl RoutedDims {
    /// The dims `hp` describes.
    #[must_use]
    pub fn of(hp: &Hparams) -> RoutedDims {
        RoutedDims {
            n_layer: hp.n_trunk,
            n_expert: hp.n_expert,
            embd: hp.n_embd,
            ff: hp.expert_ff,
        }
    }
}

/// Layer `layer`'s routed experts of a file of the family whose stacks are
/// `dims` ([`layer`]'s rule).
pub fn routed_layer(
    src: R8Source<'_>,
    dims: RoutedDims,
    layer: usize,
) -> Result<HostLayer, ModelError> {
    if layer >= dims.n_layer {
        return Err(ModelError::Shape {
            what: "qwen3moe family host layer: [layers, 1], a layer of the model",
            want_ne0: dims.n_layer,
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
            n_expert: dims.n_expert,
            embd: dims.embd,
            ff: dims.ff,
            swiglu_limit: 0.0,
        },
    )
}

/// Every layer of a host run whose stacks were refused, each with its
/// refusal: the whole list, not the first.
#[derive(Debug)]
pub struct RefusedLayers {
    /// The architecture the run is of.
    pub arch: &'static str,
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
            "{} layer(s) of the {} host run cannot be served: {}",
            self.layers.len(),
            self.arch,
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
    routed_layers(src, RoutedDims::of(hp), run, "qwen4exp")
}

/// The host views of the layers `run` of a file of the family whose stacks
/// are `dims`, the refusal naming the run as `arch`'s ([`layers`]' rule).
pub fn routed_layers(
    src: R8Source<'_>,
    dims: RoutedDims,
    run: Range<usize>,
    arch: &'static str,
) -> Result<Vec<HostLayer>, RefusedLayers> {
    let mut views = Vec::with_capacity(run.len());
    let mut refused = Vec::new();
    for l in run {
        match routed_layer(src, dims, l) {
            Ok(v) => views.push(v),
            Err(e) => refused.push((l, e)),
        }
    }
    if refused.is_empty() {
        Ok(views)
    } else {
        Err(RefusedLayers {
            arch,
            layers: refused,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::RoutedDims;
    use crate::arch::qwen35moe::hparams::{Hparams, tests::keys, tests::tensors};
    use crate::arch::synthetic::{V, header_shaped};

    /// The host tier's layers are the trunk's: a file whose last block is a
    /// next-token layer has one layer fewer than its block count.
    #[test]
    fn the_routed_layers_are_the_trunk_of_a_file_with_a_next_token_layer() {
        let kv: Vec<(&str, V)> = keys()
            .into_iter()
            .map(|(k, v)| match k {
                "block_count" => (k, V::U32(5)),
                "attention.compress_ratios" => (k, V::I32s(vec![0, 0, 0, 4, 0])),
                _ => (k, v),
            })
            .chain([("nextn_predict_layers", V::U32(1))])
            .collect();
        let mut t = tensors();
        t.push(("blk.4.nextn.eh_proj.weight".to_string(), vec![1]));
        let path = header_shaped("q4x-host", "qwen4exp", &kv, &[], &t);
        let split = gguf::Split::open(&path).expect("the synthetic header opens");
        let hp = Hparams::read(&split).expect("the header reads");
        let _ = std::fs::remove_file(&path);
        assert_eq!((hp.n_layer, hp.n_trunk), (5, 4));
        assert_eq!(RoutedDims::of(&hp).n_layer, 4);
    }
}
