//! Every tensor name a mimo2 step reads, in one place: the model-level
//! names, then a layer's `blk.{N}.*` names by the sub-layer that reads
//! them. Which layers carry which is the layer's
//! [`LayerSpec`](models::LayerSpec), read by [`super::spec`]; the roles
//! table ([`super::roles`]) holds the same stems for the classifier.

/// `blk.{l}.{stem}`.
fn blk(l: usize, stem: &str) -> String {
    format!("blk.{l}.{stem}")
}

pub fn token_embd() -> String {
    "token_embd.weight".to_string()
}

pub fn output_norm() -> String {
    "output_norm.weight".to_string()
}

pub fn output() -> String {
    "output.weight".to_string()
}

/// The mixer's RMS gain.
pub fn attn_norm(l: usize) -> String {
    blk(l, "attn_norm.weight")
}

/// The fused q, k and v projections, `[q heads · key + kv heads · (key +
/// value)]` rows in that order.
pub fn attn_qkv(l: usize) -> String {
    blk(l, "attn_qkv.weight")
}

/// The split query projection, the split form's (`attn_q.weight`).
pub fn attn_q(l: usize) -> String {
    blk(l, "attn_q.weight")
}

/// The split key projection (`attn_k.weight`).
pub fn attn_k(l: usize) -> String {
    blk(l, "attn_k.weight")
}

/// The split value projection (`attn_v.weight`), the narrow head's rows.
pub fn attn_v(l: usize) -> String {
    blk(l, "attn_v.weight")
}

/// The mixer's output projection, over the value head's width.
pub fn attn_output(l: usize) -> String {
    blk(l, "attn_output.weight")
}

/// A sliding-window layer's per-head softmax sinks; absent, the graph runs
/// without them.
pub fn attn_sinks(l: usize) -> String {
    blk(l, "attn_sinks.weight")
}

// -------------------------------------------------------- feed-forward

pub fn ffn_norm(l: usize) -> String {
    blk(l, "ffn_norm.weight")
}

pub fn ffn_gate(l: usize) -> String {
    blk(l, "ffn_gate.weight")
}

pub fn ffn_up(l: usize) -> String {
    blk(l, "ffn_up.weight")
}

pub fn ffn_down(l: usize) -> String {
    blk(l, "ffn_down.weight")
}

pub fn ffn_gate_inp(l: usize) -> String {
    blk(l, "ffn_gate_inp.weight")
}

/// The router's selection bias, which the graph adds for the choice only.
pub fn exp_probs_b(l: usize) -> String {
    blk(l, "exp_probs_b.bias")
}

pub fn ffn_gate_exps(l: usize) -> String {
    blk(l, "ffn_gate_exps.weight")
}

pub fn ffn_up_exps(l: usize) -> String {
    blk(l, "ffn_up_exps.weight")
}

pub fn ffn_down_exps(l: usize) -> String {
    blk(l, "ffn_down_exps.weight")
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

/// The block's own embedding table, when the conversion writes one.
pub fn nextn_embed_tokens(l: usize) -> String {
    blk(l, "nextn.embed_tokens.weight")
}

/// The block's own head, when the conversion writes one.
pub fn nextn_shared_head_head(l: usize) -> String {
    blk(l, "nextn.shared_head_head.weight")
}

/// The block's own head norm, when the conversion writes one.
pub fn nextn_shared_head_norm(l: usize) -> String {
    blk(l, "nextn.shared_head_norm.weight")
}
