//! Every tensor name a mimo2 step reads, in one place: the model-level
//! names, then a layer's `blk.{N}.*` names by the sub-layer that reads
//! them. Which layers carry which is the layer's
//! [`LayerSpec`](models::LayerSpec), read by [`super::spec`]; the roles
//! table ([`super::roles`]) holds the same stems for the classifier.

use crate::arch::names::blk;

pub use crate::arch::names::{
    attn_k, attn_norm, attn_output, attn_q, attn_v, exp_probs_b, ffn_down, ffn_down_exps, ffn_gate,
    ffn_gate_exps, ffn_gate_inp, ffn_norm, ffn_up, ffn_up_exps, output, output_norm, token_embd,
};

/// The fused q, k and v projections, `[q heads · key + kv heads · (key +
/// value)]` rows in that order.
pub fn attn_qkv(l: usize) -> String {
    blk(l, "attn_qkv.weight")
}

/// A sliding-window layer's per-head softmax sinks; absent, the graph runs
/// without them.
pub fn attn_sinks(l: usize) -> String {
    blk(l, "attn_sinks.weight")
}

// ------------------------------------------------------- the next-token block

/// The next-token block's input projection of `[enorm(e); hnorm(h)]`.
pub fn nextn_eh_proj(l: usize) -> String {
    blk(l, "nextn.eh_proj.weight")
}

/// The next-token block's RMS gain of the token's embedding row.
pub fn nextn_enorm(l: usize) -> String {
    blk(l, "nextn.enorm.weight")
}

/// The next-token block's RMS gain of the target's hidden row.
pub fn nextn_hnorm(l: usize) -> String {
    blk(l, "nextn.hnorm.weight")
}

/// The head's own RMS gain a conversion writes instead of sharing the
/// model's `output_norm`.
pub fn layer_output_norm(l: usize) -> String {
    blk(l, "layer_output_norm.weight")
}

/// The block's own head norm, when the conversion writes one.
pub fn nextn_shared_head_norm(l: usize) -> String {
    blk(l, "nextn.shared_head_norm.weight")
}
