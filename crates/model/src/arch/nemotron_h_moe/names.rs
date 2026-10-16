//! Every tensor name a `nemotron_h_moe` file spells. The model-level names
//! are the common ones, a Mamba-2 or attention layer's the hybrid trunk's,
//! and a layer's names are `blk.{l}.{stem}`.

use crate::arch::names::blk;

pub use crate::arch::hybrid::{
    ATTN_K, ATTN_NORM, ATTN_OUTPUT, ATTN_Q, ATTN_V, SSM_A, SSM_CONV, SSM_CONV_BIAS, SSM_D,
    SSM_DT_BIAS, SSM_IN, SSM_NORM, SSM_OUT,
};
pub use crate::arch::names::{output, output_norm, token_embd};

/// The input projection's per-tensor scale: one f32 the graph multiplies
/// the projection's output by (llama-model.cpp:1623-1624). The loader marks
/// it not required.
pub const SSM_IN_SCALE: &str = "ssm_in.scale";
/// The output projection's per-tensor scale (llama-model.cpp:1626-1627).
pub const SSM_OUT_SCALE: &str = "ssm_out.scale";

/// The router: one score per expert of the layer, from the normed row.
pub const FFN_GATE_INP: &str = "ffn_gate_inp.weight";
/// The selection bias added to the scores for the pick only.
pub const EXP_PROBS_B: &str = "exp_probs_b.bias";
/// The projection of the normed row down to the latent the experts read.
pub const FFN_LATENT_DOWN: &str = "ffn_latent_down.weight";
/// The projection of the experts' weighted sum back up to the hidden row.
pub const FFN_LATENT_UP: &str = "ffn_latent_up.weight";
/// The routed experts' up stack. Its presence marks a MoE layer.
pub const FFN_UP_EXPS: &str = "ffn_up_exps.weight";
/// The routed experts' down stack.
pub const FFN_DOWN_EXPS: &str = "ffn_down_exps.weight";
/// The shared expert's up projection, with no gate.
pub const FFN_UP_SHEXP: &str = "ffn_up_shexp.weight";
/// The shared expert's down projection.
pub const FFN_DOWN_SHEXP: &str = "ffn_down_shexp.weight";

/// `blk.{l}.ffn_gate_inp.weight`.
#[must_use]
pub fn ffn_gate_inp(l: usize) -> String {
    blk(l, FFN_GATE_INP)
}
