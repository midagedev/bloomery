//! The host tier's view of a Bolt layer's routed experts: relu² — the up and
//! down stacks, the layer's own expert count and width, read in the
//! `moe_latent_size` latent the experts run in. The one place that wires them
//! into a [`Relu2Spec`] (`HostLayer::build_relu2`); the gated form's spec
//! belongs to no layer of this architecture, whose files hold no gate stack.

use super::hparams::{Experts, Hparams, Kind};
use super::names;
use crate::ModelError;
use crate::moe::{HostLayer, Relu2Spec};
use crate::r8file::R8Source;

/// Layer `layer`'s routed experts as the host tier serves them: ungated
/// relu², two stacks at the layer's own expert count and width in the latent
/// width — the MoE layers' shapes differ from layer to layer, so a caller
/// serves each layer through the spec its own shapes make. `None` for a
/// layer that does not route (a Mamba-2 or attention layer), an error for a
/// layer past the model. Load-time only: the stacks are checked here.
pub fn layer(src: R8Source<'_>, hp: &Hparams, l: usize) -> Result<Option<HostLayer>, ModelError> {
    let kind = hp.kinds.get(l).ok_or(ModelError::Shape {
        what: "nemotron_h_moe host layer: [layers, 1], a layer inside them",
        want_ne0: hp.kinds.len(),
        want_ne1: 1,
        got_ne0: l,
        got_ne1: 1,
    })?;
    let Experts { n_expert, ff, .. } = hp
        .experts
        .get(l)
        .and_then(Option::as_ref)
        .ok_or(ModelError::Shape {
            what: "nemotron_h_moe host layer: a MoE layer's routed experts",
            want_ne0: l,
            want_ne1: 1,
            got_ne0: usize::from(*kind != Kind::Moe),
            got_ne1: 1,
        })?;
    HostLayer::build_relu2(
        src,
        &Relu2Spec {
            gate: &names::ffn_gate_exps(l),
            up: &names::ffn_up_exps(l),
            down: &names::ffn_down_exps(l),
            n_expert: *n_expert,
            embd: hp.latent,
            ff: *ff,
        },
    )
    .map(Some)
}
