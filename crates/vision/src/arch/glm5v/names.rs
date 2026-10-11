//! Every tensor name of a `glm5v` encoder file: the patch embedding (two temporal halves), the
//! final RMS norm, the `v.blk.{N}.*` names every ViT block carries (two per-head norms among them)
//! and the merger's tensors. [`super::tensors`] checks a file against this list.
//!
//! There is no position table. Norm gains have no bias (RMS), except the merger's LayerNorm, and
//! the merger's linears have no bias, except the convolution. The names every projector type holds
//! are [`crate::arch::naming`]'s, re-exported here; the strings are built there.

use crate::arch::naming::{FFN_DOWN, FFN_UP, PATCH_EMBD, Part, blk, vit};
pub use crate::arch::naming::{
    attn_out_bias, attn_out_weight, attn_qkv_bias, attn_qkv_weight, ffn_down_weight, ffn_up_weight,
    ln1_weight, ln2_weight, patch_embd_bias, patch_embd_weight, post_ln_weight,
};

const ATTN_Q_NORM: &str = "attn_q_norm";
const ATTN_K_NORM: &str = "attn_k_norm";
const FFN_GATE: &str = "ffn_gate";

/// `v.patch_embd.weight.1` — the kernel of the second temporal frame, the same shape as
/// [`patch_embd_weight`]'s (the first frame's). A still image fills both frames with itself, so
/// the two halves act as one `[dim, 2·3·p²]` weight.
pub fn patch_embd_weight_second_frame() -> String {
    format!("{}.1", vit(PATCH_EMBD, Part::Weight))
}

/// `v.blk.{block}.attn_q_norm.weight` — the RMS gain of one query head (`[head]`, shared by every
/// head).
pub fn attn_q_norm_weight(block: usize) -> String {
    blk(block, ATTN_Q_NORM, Part::Weight)
}

/// `v.blk.{block}.attn_k_norm.weight`.
pub fn attn_k_norm_weight(block: usize) -> String {
    blk(block, ATTN_K_NORM, Part::Weight)
}

/// `v.blk.{block}.ffn_gate.weight`.
pub fn ffn_gate_weight(block: usize) -> String {
    blk(block, FFN_GATE, Part::Weight)
}

/// `v.blk.{block}.ffn_gate.bias`.
pub fn ffn_gate_bias(block: usize) -> String {
    blk(block, FFN_GATE, Part::Bias)
}

/// `v.blk.{block}.ffn_up.bias`.
pub fn ffn_up_bias(block: usize) -> String {
    blk(block, FFN_UP, Part::Bias)
}

/// `v.blk.{block}.ffn_down.bias`.
pub fn ffn_down_bias(block: usize) -> String {
    blk(block, FFN_DOWN, Part::Bias)
}

/// `mm.patch_merger.weight` — the 2×2 convolution that folds a merge group's four patch rows into
/// one, ggml dims `[merge, merge, dim, out]`.
pub fn patch_merger_weight() -> String {
    "mm.patch_merger.weight".into()
}

/// `mm.patch_merger.bias`.
pub fn patch_merger_bias() -> String {
    "mm.patch_merger.bias".into()
}

/// `mm.model.fc.weight` — the merger's linear, `out` to `out`, no bias.
pub fn fc_weight() -> String {
    "mm.model.fc.weight".into()
}

/// `mm.post_norm.weight` — the LayerNorm gain after the linear.
pub fn post_norm_weight() -> String {
    "mm.post_norm.weight".into()
}

/// `mm.post_norm.bias`.
pub fn post_norm_bias() -> String {
    "mm.post_norm.bias".into()
}

/// `mm.gate.weight` — the merger MLP's gate, `out` to the merger's hidden width.
pub fn mm_gate_weight() -> String {
    "mm.gate.weight".into()
}

/// `mm.up.weight`.
pub fn mm_up_weight() -> String {
    "mm.up.weight".into()
}

/// `mm.down.weight` — the merger MLP's last layer, back to `out`.
pub fn mm_down_weight() -> String {
    "mm.down.weight".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name of the file as llama.cpp's converter writes it (GLM-5.3-Flash's mmproj header).
    #[test]
    fn the_names_are_the_files() {
        assert_eq!(patch_embd_weight(), "v.patch_embd.weight");
        assert_eq!(patch_embd_weight_second_frame(), "v.patch_embd.weight.1");
        assert_eq!(patch_embd_bias(), "v.patch_embd.bias");
        assert_eq!(post_ln_weight(), "v.post_ln.weight");
        assert_eq!(ln1_weight(0), "v.blk.0.ln1.weight");
        assert_eq!(ln2_weight(5), "v.blk.5.ln2.weight");
        assert_eq!(attn_q_norm_weight(1), "v.blk.1.attn_q_norm.weight");
        assert_eq!(attn_k_norm_weight(2), "v.blk.2.attn_k_norm.weight");
        assert_eq!(attn_qkv_weight(23), "v.blk.23.attn_qkv.weight");
        assert_eq!(attn_qkv_bias(2), "v.blk.2.attn_qkv.bias");
        assert_eq!(attn_out_weight(3), "v.blk.3.attn_out.weight");
        assert_eq!(attn_out_bias(4), "v.blk.4.attn_out.bias");
        assert_eq!(ffn_gate_weight(6), "v.blk.6.ffn_gate.weight");
        assert_eq!(ffn_gate_bias(6), "v.blk.6.ffn_gate.bias");
        assert_eq!(ffn_up_weight(7), "v.blk.7.ffn_up.weight");
        assert_eq!(ffn_up_bias(8), "v.blk.8.ffn_up.bias");
        assert_eq!(ffn_down_weight(9), "v.blk.9.ffn_down.weight");
        assert_eq!(ffn_down_bias(10), "v.blk.10.ffn_down.bias");
        assert_eq!(patch_merger_weight(), "mm.patch_merger.weight");
        assert_eq!(patch_merger_bias(), "mm.patch_merger.bias");
        assert_eq!(fc_weight(), "mm.model.fc.weight");
        assert_eq!(post_norm_weight(), "mm.post_norm.weight");
        assert_eq!(post_norm_bias(), "mm.post_norm.bias");
        assert_eq!(mm_gate_weight(), "mm.gate.weight");
        assert_eq!(mm_up_weight(), "mm.up.weight");
        assert_eq!(mm_down_weight(), "mm.down.weight");
    }
}
