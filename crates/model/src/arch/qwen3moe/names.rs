//! Every tensor name the Qwen3-MoE decode step reads, in one place: the three
//! model-level names, then the `blk.{N}.*` names every layer carries. There
//! are no per-layer kinds: every layer is attention with per-head QK norms and
//! a routed FFN. [`LAYER`] lists the per-layer stems, which the classifier
//! (`roles`) and the metadata gate walk.

pub use crate::arch::names::{
    attn_k, attn_k_norm, attn_norm, attn_output, attn_q, attn_q_norm, attn_v, ffn_down_exps,
    ffn_gate_exps, ffn_gate_inp, ffn_norm, ffn_up_exps, output, output_norm, token_embd,
};

/// The part after `blk.{N}.` of every per-layer name above, in the order
/// above. Each layer carries all of them and nothing else; the router
/// (`ffn_gate_inp`) is on every layer, for this architecture has no dense FFN
/// (`Hparams` refuses a layer without it).
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
