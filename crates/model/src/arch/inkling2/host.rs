//! The host tier's view of an Inkling 2 layer: the fused gate·up stack, the down stack, the
//! widths and the SiTU betas a [`HostLayer`] serves by, read from the hyperparameters and the
//! tensor names — the one place that wires them into a [`FusedSituSpec`].

use super::hparams::Hparams;
use super::names;
use crate::ModelError;
use crate::moe::{FusedSituSpec, HostLayer};
use crate::r8file::R8Source;

/// Layer `layer`'s routed experts as the host tier serves them, built from `src`
/// ([`HostLayer::build_situ_fused`]): each expert's gate and up from the one fused stack the file
/// holds, combined by SiTU at the file's betas (the card leg's rule, with the host's `tanh` and
/// `exp` where the card's are libdevice's). A layer past the file's is refused by name; every
/// layer of the family routes. Load-time only: the stacks are checked here.
pub fn layer(src: R8Source<'_>, hp: &Hparams, layer: usize) -> Result<HostLayer, ModelError> {
    if layer >= hp.n_layer {
        return Err(ModelError::MissingTensor(format!(
            "layer {layer}'s routed stacks: a layer of the host run past the file's {}",
            hp.n_layer
        )));
    }
    let (gate_up, down) = (names::ffn_gate_up_exps(layer), names::ffn_down_exps(layer));
    HostLayer::build_situ_fused(
        src,
        &FusedSituSpec {
            gate_up: &gate_up,
            down: &down,
            n_expert: hp.n_expert,
            embd: hp.n_embd,
            ff: hp.expert_ff[layer],
            beta: hp.situ.gate_beta,
            linear_beta: hp.situ.value_beta,
        },
    )
}
