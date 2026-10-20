//! Every tensor name an M3 step reads: the shared spellings every family takes from
//! [`crate::arch::names`], and the indexer's four, which only the sparse layers carry.

use crate::arch::names::blk;

pub use crate::arch::names::{
    attn_k, attn_k_norm, attn_norm, attn_output, attn_q, attn_q_norm, attn_v, exp_probs_b,
    ffn_down, ffn_down_exps, ffn_down_shexp, ffn_gate, ffn_gate_exps, ffn_gate_inp, ffn_gate_shexp,
    ffn_norm, ffn_up, ffn_up_exps, ffn_up_shexp, output, output_norm, token_embd,
};

/// The indexer's query projection, `[embedding, index heads · index width]`, f32.
pub fn indexer_q_proj(l: usize) -> String {
    blk(l, "indexer.q_proj.weight")
}

/// The indexer's key projection, `[embedding, index width]`, f32: one key head.
pub fn indexer_k_proj(l: usize) -> String {
    blk(l, "indexer.k_proj.weight")
}

/// The RMS gain every index query head is normed with, one index head wide.
pub fn indexer_q_norm(l: usize) -> String {
    blk(l, "indexer.q_norm.weight")
}

/// The RMS gain the index key is normed with, one index head wide.
pub fn indexer_k_norm(l: usize) -> String {
    blk(l, "indexer.k_norm.weight")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four indexer names are the GGUF spellings of the tensors the released files carry.
    /// Mutant: the query and key projections swapped.
    #[test]
    fn the_indexer_names_are_the_gguf_spellings() {
        assert_eq!(indexer_q_proj(3), "blk.3.indexer.q_proj.weight");
        assert_eq!(indexer_k_proj(3), "blk.3.indexer.k_proj.weight");
        assert_eq!(indexer_q_norm(3), "blk.3.indexer.q_norm.weight");
        assert_eq!(indexer_k_norm(3), "blk.3.indexer.k_norm.weight");
    }
}
