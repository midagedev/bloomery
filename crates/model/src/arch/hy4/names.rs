//! Every tensor name a Hy4-flash step reads: the spellings every family takes from
//! [`crate::arch::names`] (the latent layer's, the hyper-connection mixes', the indexer's among
//! them), and the names of this family's own: the sandwich norms, the indexer's pooling
//! compressor, the engram site's delta, the per-layer table, the head's stream fold and the
//! next-token block's.

use crate::arch::names::blk;

pub use crate::arch::names::{
    Sub, attn_gate, attn_k_b, attn_kv_a_mqa, attn_kv_a_norm, attn_norm, attn_output, attn_q_a,
    attn_q_a_norm, attn_q_b, attn_sinks, attn_v_b, exp_probs_b, ffn_down, ffn_down_exps,
    ffn_down_shexp, ffn_gate, ffn_gate_exps, ffn_gate_inp, ffn_gate_shexp, ffn_norm, ffn_up,
    ffn_up_exps, ffn_up_shexp, hc_base, hc_fn, hc_scale, indexer_attn_k, indexer_attn_q_b,
    indexer_k_norm, indexer_proj, output, output_norm, token_embd,
};

/// `per_layer_token_embd.weight`: the one fused table every engram site gathers its rows from.
pub const ENGRAM_TABLE: &str = "per_layer_token_embd.weight";

/// `output_hc_fn.weight`: the head's projection of the flattened streams to one mix a stream.
pub const OUTPUT_HC_FN: &str = "output_hc_fn.weight";

/// `output_hc_base.weight`: the head mix's bias.
pub const OUTPUT_HC_BASE: &str = "output_hc_base.weight";

/// `output_hc_scale.weight`: the head mix's scalar.
pub const OUTPUT_HC_SCALE: &str = "output_hc_scale.weight";

/// `blk.{l}.post_attention_norm.weight` — the gain of the norm on the mixer's output.
#[must_use]
pub fn post_attention_norm(l: usize) -> String {
    blk(l, "post_attention_norm.weight")
}

/// `blk.{l}.post_ffw_norm.weight` — the gain of the norm on the feed-forward block's output.
#[must_use]
pub fn post_ffw_norm(l: usize) -> String {
    blk(l, "post_ffw_norm.weight")
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

/// `blk.{l}.indexer_compressor_value.weight` — the pooling compressor's value projection.
#[must_use]
pub fn indexer_compressor_value(l: usize) -> String {
    blk(l, "indexer_compressor_value.weight")
}

/// `blk.{l}.indexer_compressor_gate.weight` — the pooling compressor's score projection.
#[must_use]
pub fn indexer_compressor_gate(l: usize) -> String {
    blk(l, "indexer_compressor_gate.weight")
}

/// `blk.{l}.indexer_compressor_ape.weight` — the pooling compressor's position table.
#[must_use]
pub fn indexer_compressor_ape(l: usize) -> String {
    blk(l, "indexer_compressor_ape.weight")
}

/// `blk.{l}.indexer_compressor_norm.weight` — the gain of the norm on a pooled key.
#[must_use]
pub fn indexer_compressor_norm(l: usize) -> String {
    blk(l, "indexer_compressor_norm.weight")
}

/// `blk.{l}.engram_kv.weight` — the engram site's fused key and value projection.
#[must_use]
pub fn engram_kv(l: usize) -> String {
    blk(l, "engram_kv.weight")
}

/// `blk.{l}.engram_kv_b.weight` — the bias of the fused key and value projection.
#[must_use]
pub fn engram_kv_b(l: usize) -> String {
    blk(l, "engram_kv_b.weight")
}

/// `blk.{l}.engram_gate.weight` — the gate's weights a stream.
#[must_use]
pub fn engram_gate(l: usize) -> String {
    blk(l, "engram_gate.weight")
}

/// `blk.{l}.engram_hc_scale.weight` — the delta's scale a stream and channel.
#[must_use]
pub fn engram_hc_scale(l: usize) -> String {
    blk(l, "engram_hc_scale.weight")
}

/// `blk.{l}.engram_lookup_norm.weight` — the gain of the norm on a gathered row.
#[must_use]
pub fn engram_lookup_norm(l: usize) -> String {
    blk(l, "engram_lookup_norm.weight")
}

/// `blk.{l}.engram_pre_gamma.weight` — the gain on the value before the gate adds it.
#[must_use]
pub fn engram_pre_gamma(l: usize) -> String {
    blk(l, "engram_pre_gamma.weight")
}

/// `blk.{l}.nextn.eh_proj.weight` — the next-token block's projection of the embedding and the
/// hidden row joined.
#[must_use]
pub fn nextn_eh_proj(l: usize) -> String {
    blk(l, "nextn.eh_proj.weight")
}
