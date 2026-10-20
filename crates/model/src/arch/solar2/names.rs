//! Every tensor name a Solar-Open2 step reads: the spellings every family takes from
//! [`crate::arch::names`], the KDA mixer's among them, and the GQA layer's output gate.

use crate::arch::names::blk;

pub use crate::arch::names::{
    attn_k, attn_norm, attn_output, attn_q, attn_v, exp_probs_b, ffn_down_exps, ffn_down_shexp,
    ffn_gate_exps, ffn_gate_inp, ffn_gate_shexp, ffn_norm, ffn_up_exps, ffn_up_shexp, output,
    output_norm, ssm_a, ssm_beta, ssm_conv1d, ssm_dt_bias, ssm_f_a, ssm_f_b, ssm_g_a, ssm_g_b,
    ssm_norm, token_embd,
};

/// `blk.{l}.attn_gate.weight` — the GQA layer's output gate projection, one value for each value
/// the attention writes.
#[must_use]
pub fn attn_gate(l: usize) -> String {
    blk(l, "attn_gate.weight")
}
