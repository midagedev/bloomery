//! What the Hy4-flash expert tier takes from the family: the activation a tier layer runs and the
//! host view of a routed layer's experts. The common tier ([`bloomery_gpu::host::leg_tier`]) runs
//! the card leg over the latent; the routed experts' rule is the gate-first clamp
//! `silu(min(g, limit)) · clamp(u, ±limit)` at the layer's `swiglu_clamp_exp`, which the common
//! [`LegFamily::tier_act`](bloomery_gpu::host::leg_tier::LegFamily::tier_act) default (the
//! clamped SwiGLU's `swiglu_limit`) does not read, so the family's override is [`tier_act`].

use bloomery_gpu::GpuError;
use bloomery_gpu::kquant::Act;
use model::ModelError;
use model::arch::hy4::hparams::Hparams;
use model::arch::hy4::names;
use model::arch::models::{Act as ModelAct, Moe};
use model::moe::{HostLayer, HostLayerSpec};
use model::r8file::R8Source;

/// What the tier's refusals name.
const WHAT: &str = "hy4 tier";

/// The activation a tier layer of `moe` runs: the card leg's gate-first clamp at the block's
/// limit. Any other activation, and a limit the card's rule does not run, is refused by name.
pub fn tier_act(moe: &Moe) -> Result<Act, GpuError> {
    let ModelAct::SwiGluGateFirst { limit } = moe.act else {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "a {} block: Hy4's routed experts run the gate-first clamped SwiGLU",
                moe.act.what()
            ),
        ));
    };
    let act = Act::SwigluGateClamp { limit };
    act.code(WHAT)?;
    Ok(act)
}

/// Layer `l`'s routed experts as a host tier serves them: the file's three stacks at the latent's
/// width (`expert_latent_length`), combined by the gate-first clamp at the layer's limit.
pub fn host_layer(src: R8Source<'_>, hp: &Hparams, l: usize) -> Result<HostLayer, ModelError> {
    let (gate, up, down) = (
        names::ffn_gate_exps(l),
        names::ffn_up_exps(l),
        names::ffn_down_exps(l),
    );
    let mut layer = HostLayer::build(
        src,
        &HostLayerSpec {
            gate: &gate,
            up: &up,
            down: &down,
            n_expert: hp.experts.n_expert,
            embd: hp.latent,
            ff: hp.experts.ff,
            swiglu_limit: 0.0,
        },
    )?;
    let limit = *hp.clamp.get(l).ok_or_else(|| ModelError::Metadata {
        key: "swiglu_clamp_exp".to_string(),
        detail: format!("no entry for layer {l} of {}", hp.clamp.len()),
    })?;
    layer.set_act(ModelAct::SwiGluGateFirst { limit })?;
    Ok(layer)
}
