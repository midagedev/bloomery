//! Every tensor name the Qwen3-MoE decode step reads, in one place: the three
//! model-level names, then the `blk.{N}.*` names every layer carries. There
//! are no per-layer kinds: every layer is attention with per-head QK norms and
//! a routed FFN. [`LAYER`] lists the per-layer stems, which the classifier
//! (`roles`) and the metadata gate walk.

/// `token_embd.weight` — the token embedding, one row per token.
pub fn token_embd() -> String {
    "token_embd.weight".to_string()
}

/// `output_norm.weight` — the RMS gain before the head.
pub fn output_norm() -> String {
    "output_norm.weight".to_string()
}

/// `output.weight` — the head, one row per vocabulary entry.
pub fn output() -> String {
    "output.weight".to_string()
}

/// `blk.{block}.attn_norm.weight` — the pre-attention RMS gain.
pub fn attn_norm(block: usize) -> String {
    format!("blk.{block}.attn_norm.weight")
}

/// `blk.{block}.attn_q.weight` — the query projection, every head.
pub fn attn_q(block: usize) -> String {
    format!("blk.{block}.attn_q.weight")
}

/// `blk.{block}.attn_k.weight` — the key projection, every key head.
pub fn attn_k(block: usize) -> String {
    format!("blk.{block}.attn_k.weight")
}

/// `blk.{block}.attn_v.weight` — the value projection, every value head.
pub fn attn_v(block: usize) -> String {
    format!("blk.{block}.attn_v.weight")
}

/// `blk.{block}.attn_q_norm.weight` — the RMS gain every query head is normed with, one head wide.
pub fn attn_q_norm(block: usize) -> String {
    format!("blk.{block}.attn_q_norm.weight")
}

/// `blk.{block}.attn_k_norm.weight` — the RMS gain every key head is normed with, one head wide.
pub fn attn_k_norm(block: usize) -> String {
    format!("blk.{block}.attn_k_norm.weight")
}

/// `blk.{block}.attn_output.weight` — the projection from the heads' outputs back to the residual.
pub fn attn_output(block: usize) -> String {
    format!("blk.{block}.attn_output.weight")
}

/// `blk.{block}.ffn_norm.weight` — the pre-FFN RMS gain.
pub fn ffn_norm(block: usize) -> String {
    format!("blk.{block}.ffn_norm.weight")
}

/// `blk.{block}.ffn_gate_inp.weight` — the router, one logit per expert. Every layer carries it: this
/// architecture has no dense FFN (`Hparams` refuses a layer without it).
pub fn ffn_gate_inp(block: usize) -> String {
    format!("blk.{block}.ffn_gate_inp.weight")
}

/// `blk.{block}.ffn_gate_exps.weight` — the routed experts' gate stack.
pub fn ffn_gate_exps(block: usize) -> String {
    format!("blk.{block}.ffn_gate_exps.weight")
}

/// `blk.{block}.ffn_up_exps.weight` — the routed experts' up stack.
pub fn ffn_up_exps(block: usize) -> String {
    format!("blk.{block}.ffn_up_exps.weight")
}

/// `blk.{block}.ffn_down_exps.weight` — the routed experts' down stack.
pub fn ffn_down_exps(block: usize) -> String {
    format!("blk.{block}.ffn_down_exps.weight")
}

/// The part after `blk.{N}.` of every per-layer name above, in the order
/// above. Each layer carries all of them and nothing else.
pub const LAYER: [&str; 12] = [
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "attn_output.weight",
    "ffn_norm.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_down_exps.weight",
];

#[cfg(test)]
mod tests {
    use super::*;

    /// [`LAYER`] is the functions' own stems, in their order.
    #[test]
    fn layer_stems_are_the_functions_names() {
        let fns: [fn(usize) -> String; 12] = [
            attn_norm,
            attn_q,
            attn_k,
            attn_v,
            attn_q_norm,
            attn_k_norm,
            attn_output,
            ffn_norm,
            ffn_gate_inp,
            ffn_gate_exps,
            ffn_up_exps,
            ffn_down_exps,
        ];
        for (stem, f) in LAYER.iter().zip(fns) {
            assert_eq!(f(7), format!("blk.7.{stem}"));
        }
    }
}
