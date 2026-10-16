//! The host tier's view of a Bolt layer's routed experts: relu² — the up and
//! down stacks, the layer's own expert count and width, read in the
//! `moe_latent_size` latent the experts run in. The one place that wires them
//! into a [`Relu2Spec`] (`HostLayer::build_relu2`); this architecture's files
//! hold no gate stack.

use super::hparams::{Experts, Hparams};
use super::names;
use crate::ModelError;
use crate::moe::{HostLayer, Relu2Spec};
use crate::r8file::R8Source;

/// Layer `l`'s routed experts as the host tier serves them: ungated relu²,
/// two stacks at the layer's own expert count and width in the latent width —
/// the MoE layers' shapes differ from layer to layer, so a caller serves each
/// layer through the layer its own shapes make. `None` for a layer that does
/// not route (a Mamba-2 or attention layer), an error for a layer past the
/// model. Load-time only: the stacks are checked here.
pub fn layer(src: R8Source<'_>, hp: &Hparams, l: usize) -> Result<Option<HostLayer>, ModelError> {
    let Some(experts) = hp.experts.get(l) else {
        return Err(ModelError::Shape {
            what: "nemotron_h_moe host layer: the highest layer index, a layer inside the model's",
            want_ne0: hp.experts.len().saturating_sub(1),
            want_ne1: 1,
            got_ne0: l,
            got_ne1: 1,
        });
    };
    let Some(Experts { n_expert, ff, .. }) = *experts else {
        return Ok(None);
    };
    HostLayer::build_relu2(
        src,
        &Relu2Spec {
            up: &names::ffn_up_exps(l),
            down: &names::ffn_down_exps(l),
            n_expert,
            embd: hp.latent,
            ff,
        },
    )
    .map(Some)
}
