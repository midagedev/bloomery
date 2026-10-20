//! Every tensor name an M3 step reads: the spellings every family takes from
//! [`crate::arch::names`], the indexer's four among them.

pub use crate::arch::names::{
    attn_k, attn_k_norm, attn_norm, attn_output, attn_q, attn_q_norm, attn_v, exp_probs_b,
    ffn_down, ffn_down_exps, ffn_down_shexp, ffn_gate, ffn_gate_exps, ffn_gate_inp, ffn_gate_shexp,
    ffn_norm, ffn_up, ffn_up_exps, ffn_up_shexp, indexer_k_norm, indexer_k_proj, indexer_q_norm,
    indexer_q_proj, output, output_norm, token_embd,
};
