//! Every tensor name a Raon-1 step reads: the spellings every family takes from
//! [`crate::arch::names`], the delta rule's, the latent layer's and the hyper-connections' among
//! them, the write gate of the mini dialect, and the fused stems and per-layer inputs of the
//! nano dialect.

use crate::arch::names::blk;

pub use crate::arch::names::{
    Sub, attn_gate, attn_k, attn_k_b, attn_kv_a_mqa, attn_kv_a_norm, attn_norm, attn_output,
    attn_q, attn_q_a, attn_q_a_norm, attn_q_b, attn_v, attn_v_b, exp_probs_b, ffn_down,
    ffn_down_exps, ffn_down_shexp, ffn_gate, ffn_gate_exps, ffn_gate_inp, ffn_gate_shexp, ffn_norm,
    ffn_up, ffn_up_exps, ffn_up_shexp, hc_base, hc_fn, hc_scale, output, output_norm, ssm_a,
    ssm_beta, ssm_conv1d, ssm_dt_bias, ssm_f_a, ssm_f_b, ssm_g, ssm_norm, token_embd,
};

/// `blk.{l}.ssm_rda.weight` — the delta layer's write gate offset, one value for each head.
#[must_use]
pub fn ssm_rda(l: usize) -> String {
    blk(l, "ssm_rda.weight")
}

/// `blk.{l}.attn_qkv.weight` — the nano delta layer's one projection, rows `q | k | v | g`.
pub const ATTN_QKV: &str = "attn_qkv.weight";

/// `blk.{l}.ssm_conv1d.weight` — the nano delta layer's one conv, over the fused `q | k | v`
/// channels (the mini dialect's three `ssm_conv1d_{q,k,v}` in one tensor).
pub const SSM_CONV1D: &str = "ssm_conv1d.weight";

/// `blk.{l}.inp_gate.weight` — the per-layer input's gate, from the layer's stream sum.
pub const INP_GATE: &str = "inp_gate.weight";

/// `blk.{l}.proj.weight` — the per-layer input's projection back to the residual width.
pub const PROJ: &str = "proj.weight";

/// `blk.{l}.post_norm.weight` — the RMS gain on that projection.
pub const POST_NORM: &str = "post_norm.weight";

/// `per_layer_token_embd.weight` — the per-layer input's table, one row of
/// `embedding_length_per_layer_input` values a layer a token.
pub const PER_LAYER_TOKEN_EMBD: &str = "per_layer_token_embd.weight";

/// `per_layer_model_proj.weight` — the projection of the token row to every layer's input.
pub const PER_LAYER_MODEL_PROJ: &str = "per_layer_model_proj.weight";

/// `per_layer_proj_norm.weight` — the RMS gain on each layer's slice of that projection.
pub const PER_LAYER_PROJ_NORM: &str = "per_layer_proj_norm.weight";
