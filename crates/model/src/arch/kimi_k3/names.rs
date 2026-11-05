//! Every tensor name a Kimi K3 step reads: the spellings every family takes from
//! [`crate::arch::names`], the delta rule's and the latent layer's among them, and the names of
//! this family's own: the two gates, the latent experts' pair and norm, and the cross-layer
//! residual's scores.

use crate::arch::names::blk;

pub use crate::arch::names::{
    attn_k, attn_k_b, attn_kv_a_mqa, attn_kv_a_norm, attn_norm, attn_output, attn_q, attn_q_a,
    attn_q_a_norm, attn_q_b, attn_v, attn_v_b, exp_probs_b, ffn_down, ffn_down_exps,
    ffn_down_shexp, ffn_gate, ffn_gate_exps, ffn_gate_inp, ffn_gate_shexp, ffn_norm, ffn_up,
    ffn_up_exps, ffn_up_shexp, output, output_norm, ssm_a, ssm_beta, ssm_conv1d, ssm_dt_bias,
    ssm_f_a, ssm_f_b, ssm_norm, token_embd,
};

/// `output_res_score.weight`: the score vector of the final mix of the banks and the running sum.
pub const OUTPUT_RES_SCORE: &str = "output_res_score.weight";

/// `blk.{l}.attn_gate.weight` — the latent layer's output gate projection, one value for each
/// value the heads write.
#[must_use]
pub fn attn_gate(l: usize) -> String {
    blk(l, "attn_gate.weight")
}

/// `blk.{l}.ssm_g.weight` — the delta layer's output gate projection, one value for each channel.
#[must_use]
pub fn ssm_g(l: usize) -> String {
    blk(l, "ssm_g.weight")
}

/// `blk.{l}.attn_res_score.weight` — the score vector of the mix the mixer reads.
#[must_use]
pub fn attn_res_score(l: usize) -> String {
    blk(l, "attn_res_score.weight")
}

/// `blk.{l}.ffn_res_score.weight` — the score vector of the mix the feed-forward block reads.
#[must_use]
pub fn ffn_res_score(l: usize) -> String {
    blk(l, "ffn_res_score.weight")
}

/// `blk.{l}.ffn_routed_down.weight` — the routed experts' input projection, hidden to latent.
#[must_use]
pub fn ffn_routed_down(l: usize) -> String {
    blk(l, "ffn_routed_down.weight")
}

/// `blk.{l}.ffn_routed_up.weight` — the routed experts' output projection, latent to hidden.
#[must_use]
pub fn ffn_routed_up(l: usize) -> String {
    blk(l, "ffn_routed_up.weight")
}

/// `blk.{l}.ffn_routed_norm.weight` — the gain of the norm on the routed experts' weighted sum.
#[must_use]
pub fn ffn_routed_norm(l: usize) -> String {
    blk(l, "ffn_routed_norm.weight")
}
