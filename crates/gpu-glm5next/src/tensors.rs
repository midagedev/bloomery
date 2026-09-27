//! Each layer's tensor names, made once at load from its programs: the
//! step looks its weights up by these, and formats none.

use bloomery_gpu::GpuError;
use model::arch::glm5next::names::{self, Sub};
use runtime::layer::{FfnKind, Layer, MixerKind};

/// A sub-layer's hyper-connection tensors.
pub(crate) struct HcNames {
    pub fn_: String,
    pub scale: String,
    pub base: String,
}

impl HcNames {
    fn of(l: usize, sub: Sub) -> HcNames {
        HcNames {
            fn_: names::hc_fn(l, sub),
            scale: names::hc_scale(l, sub),
            base: names::hc_base(l, sub),
        }
    }
}

/// A KDA mixer's tensors, the joined q·k·v and conv taps among them.
pub(crate) struct KdaNames {
    pub norm: String,
    pub qkv: String,
    pub f_a: String,
    pub g_a: String,
    pub beta: String,
    pub f_b: String,
    pub g_b: String,
    pub conv: String,
    pub dt_bias: String,
    pub a: String,
    pub gate_norm: String,
    pub out: String,
}

/// A latent mixer's tensors, the joined projection of the normed input
/// among them.
pub(crate) struct LatentNames {
    pub norm: String,
    pub stack: String,
    pub q_a_norm: String,
    pub kv_a_norm: String,
    pub index_norm: String,
    pub index_norm_bias: String,
    pub q_b: String,
    pub k_b: String,
    pub v_b: String,
    pub out: String,
}

/// A layer's mixer's tensors, by its kind.
pub(crate) enum MixerNames {
    Kda(KdaNames),
    Latent(LatentNames),
}

/// A layer's feed-forward block's tensors, by its kind: the routed block's
/// router, its selection bias and its shared expert (the card experts'
/// stacks are the card's own, [`crate::ffn`]).
pub(crate) enum FfnNames {
    Dense {
        norm: String,
        gate: String,
        up: String,
        down: String,
    },
    Moe {
        norm: String,
        router: String,
        bias: String,
        sh_gate: String,
        sh_up: String,
        sh_down: String,
    },
}

/// One layer's tensor names.
pub(crate) struct LayerNames {
    pub hc_attn: HcNames,
    pub hc_ffn: HcNames,
    pub mixer: MixerNames,
    pub ffn: FfnNames,
}

impl LayerNames {
    /// Layer `l`'s names by its programs `kind`; a GQA mixer, which
    /// glm5next has none of, is refused by name.
    pub(crate) fn of(l: usize, kind: Layer) -> Result<LayerNames, GpuError> {
        let mixer = match kind.mixer {
            MixerKind::DeltaRule => MixerNames::Kda(KdaNames {
                norm: names::attn_norm(l),
                qkv: names::attn_qkv(l),
                f_a: names::ssm_f_a(l),
                g_a: names::ssm_g_a(l),
                beta: names::ssm_beta(l),
                f_b: names::ssm_f_b(l),
                g_b: names::ssm_g_b(l),
                conv: names::ssm_conv1d_qkv(l),
                dt_bias: names::ssm_dt_bias(l),
                a: names::ssm_a(l),
                gate_norm: names::ssm_norm(l),
                out: names::attn_output(l),
            }),
            MixerKind::Latent => MixerNames::Latent(LatentNames {
                norm: names::attn_norm(l),
                stack: names::attn_a_stack(l),
                q_a_norm: names::attn_q_a_norm(l),
                kv_a_norm: names::attn_kv_a_norm(l),
                index_norm: names::indexer_k_norm(l),
                index_norm_bias: names::indexer_k_norm_bias(l),
                q_b: names::attn_q_b(l),
                k_b: names::attn_k_b(l),
                v_b: names::attn_v_b(l),
                out: names::attn_output(l),
            }),
            MixerKind::Gqa => {
                return Err(GpuError::Shape {
                    what: "glm5next LayerNames",
                    detail: format!("layer {l}: a GQA mixer, which glm5next has none of"),
                });
            }
        };
        let ffn = match kind.ffn {
            FfnKind::Dense => FfnNames::Dense {
                norm: names::ffn_norm(l),
                gate: names::ffn_gate(l),
                up: names::ffn_up(l),
                down: names::ffn_down(l),
            },
            FfnKind::Moe => FfnNames::Moe {
                norm: names::ffn_norm(l),
                router: names::ffn_gate_inp(l),
                bias: names::exp_probs_b(l),
                sh_gate: names::ffn_gate_shexp(l),
                sh_up: names::ffn_up_shexp(l),
                sh_down: names::ffn_down_shexp(l),
            },
        };
        Ok(LayerNames {
            hc_attn: HcNames::of(l, Sub::Attn),
            hc_ffn: HcNames::of(l, Sub::Ffn),
            mixer,
            ffn,
        })
    }

    /// Sub-layer `sub`'s hyper-connection names.
    pub(crate) fn hc(&self, sub: Sub) -> &HcNames {
        match sub {
            Sub::Attn => &self.hc_attn,
            Sub::Ffn => &self.hc_ffn,
        }
    }
}

/// The error of a launch that finds another kind's names at its layer: the
/// names follow the programs, so this is a wiring fault, named.
pub(crate) fn other_kind(what: &'static str, l: usize) -> GpuError {
    GpuError::Shape {
        what,
        detail: format!("layer {l}: its names are another kind's than this launch runs"),
    }
}
