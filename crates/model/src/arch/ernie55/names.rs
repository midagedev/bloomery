//! Every tensor name an ERNIE 5.5 step reads: the spellings every family takes from
//! [`crate::arch::names`] (the latent layer's, the hyper-connection mixes', the indexer's among
//! them), and the names of this family's own: the HCA layer's grouped output and compressor, the
//! output gate and the head mix every mixer carries, the second router and the learned expert
//! scale, the latent experts' pair and norm, and the head's stream fold and MultiMax vectors.

use crate::arch::names::blk;

pub use crate::arch::names::{
    Sub, attn_gate, attn_kv_a_mqa, attn_kv_a_norm, attn_norm, attn_output, attn_q_a, attn_q_a_norm,
    attn_q_b, attn_sinks, exp_probs_b, ffn_down, ffn_down_exps, ffn_down_shexp, ffn_gate,
    ffn_gate_exps, ffn_gate_inp, ffn_gate_shexp, ffn_norm, ffn_up, ffn_up_exps, ffn_up_shexp,
    hc_base, hc_fn, hc_scale, indexer_attn_k, indexer_attn_q_b, indexer_k_norm, indexer_proj,
    output, output_norm, token_embd,
};

/// `output_hc_fn.weight`: the head's projection of the flattened streams to one mix a stream.
pub const OUTPUT_HC_FN: &str = "output_hc_fn.weight";

/// `output_hc_base.weight`: the head mix's bias.
pub const OUTPUT_HC_BASE: &str = "output_hc_base.weight";

/// `output_hc_scale.weight`: the head mix's scalar.
pub const OUTPUT_HC_SCALE: &str = "output_hc_scale.weight";

/// `output_multimax_ranges.weight`: the four range edges of the head's MultiMax epilogue.
pub const OUTPUT_MULTIMAX_RANGES: &str = "output_multimax_ranges.weight";

/// `output_multimax_ts.weight`: the four weights of the head's MultiMax epilogue.
pub const OUTPUT_MULTIMAX_TS: &str = "output_multimax_ts.weight";

/// `blk.{l}.attn_kv_b.weight` — a latent layer's fused per-head up projection, the key and value
/// halves in one `[nope + v]`-row block a head, as the checkpoint stores it.
#[must_use]
pub fn attn_kv_b(l: usize) -> String {
    blk(l, "attn_kv_b.weight")
}

/// `blk.{l}.attn_kv.weight` — an HCA layer's projection to the one latent that is key and value.
#[must_use]
pub fn attn_kv(l: usize) -> String {
    blk(l, "attn_kv.weight")
}

/// `blk.{l}.attn_output_a.weight` — an HCA layer's grouped output projection, one block of the
/// diagonal per output group.
#[must_use]
pub fn attn_output_a(l: usize) -> String {
    blk(l, "attn_output_a.weight")
}

/// `blk.{l}.attn_output_b.weight` — an HCA layer's projection from the groups to the hidden row.
#[must_use]
pub fn attn_output_b(l: usize) -> String {
    blk(l, "attn_output_b.weight")
}

/// `blk.{l}.attn_compressor_kv.weight` — an HCA layer's pooling projection of the layer input.
#[must_use]
pub fn attn_compressor_kv(l: usize) -> String {
    blk(l, "attn_compressor_kv.weight")
}

/// `blk.{l}.attn_compressor_gate.weight` — an HCA layer's pooling scores.
#[must_use]
pub fn attn_compressor_gate(l: usize) -> String {
    blk(l, "attn_compressor_gate.weight")
}

/// `blk.{l}.attn_compressor_norm.weight` — the gain of the norm on a pooled row.
#[must_use]
pub fn attn_compressor_norm(l: usize) -> String {
    blk(l, "attn_compressor_norm.weight")
}

/// `blk.{l}.attn_compressor_ape.weight` — the pooling's position table: one score bias row a slot
/// of a pooled group.
#[must_use]
pub fn attn_compressor_ape(l: usize) -> String {
    blk(l, "attn_compressor_ape.weight")
}

/// `blk.{l}.attn_vha_u.weight` — the head mix's `U`, one rank-wide row a head.
#[must_use]
pub fn attn_vha_u(l: usize) -> String {
    blk(l, "attn_vha_u.weight")
}

/// `blk.{l}.attn_vha_v.weight` — the head mix's `V`, one rank-wide row a head.
#[must_use]
pub fn attn_vha_v(l: usize) -> String {
    blk(l, "attn_vha_v.weight")
}

/// `blk.{l}.indexer.k_norm.bias` — the bias of the indexer key's LayerNorm.
#[must_use]
pub fn indexer_k_norm_bias(l: usize) -> String {
    blk(l, "indexer.k_norm.bias")
}

/// `blk.{l}.ffn_gate_inp_1.weight` — the second router, whose sigmoid scores add to the first's.
#[must_use]
pub fn ffn_gate_inp_1(l: usize) -> String {
    blk(l, "ffn_gate_inp_1.weight")
}

/// `blk.{l}.ffn_exp_scale.weight` — the learned factor an expert, which the kept weights are
/// multiplied by after their renormalization.
#[must_use]
pub fn ffn_exp_scale(l: usize) -> String {
    blk(l, "ffn_exp_scale.weight")
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

/// `blk.{l}.ffn_routed_norm.weight` — the gain of the norm on the routed sum before the output
/// projection.
#[must_use]
pub fn ffn_routed_norm(l: usize) -> String {
    blk(l, "ffn_routed_norm.weight")
}
