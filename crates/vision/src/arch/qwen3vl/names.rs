//! Every tensor name of a `qwen3vl_merger` encoder file: the patch embedding (two temporal
//! halves), the learned positions, the `v.blk.{N}.*` names every ViT block carries, the final
//! norm and the merger's two layers. [`super::tensors`] checks a file against this list.
//!
//! Every norm and every linear layer has a bias here. The names every projector type holds are
//! [`crate::arch::naming`]'s, re-exported here; the strings are built there.

use crate::arch::naming::{FFN_DOWN, FFN_UP, LN1, LN2, PATCH_EMBD, POST_LN, Part, blk, mm, vit};
pub use crate::arch::naming::{
    attn_out_bias, attn_out_weight, attn_qkv_bias, attn_qkv_weight, ffn_down_weight, ffn_up_weight,
    ln1_weight, ln2_weight, mm2_bias, mm2_weight, patch_embd_bias, patch_embd_weight,
    post_ln_weight,
};

/// The learned position table's stem.
const POSITION_EMBD: &str = "position_embd";

/// `v.patch_embd.weight.1` — the kernel of the second temporal frame, the same shape as
/// [`patch_embd_weight`]'s (the first frame's). A still image fills both frames with itself, so
/// the two halves act as one `[dim, 2·3·p·p]` weight.
pub fn patch_embd_weight_second_frame() -> String {
    format!("{}.1", vit(PATCH_EMBD, Part::Weight))
}

/// `v.position_embd.weight` — the learned position table, one row per position of a square grid.
pub fn position_embd() -> String {
    vit(POSITION_EMBD, Part::Weight)
}

/// `v.blk.{block}.ln1.bias`.
pub fn ln1_bias(block: usize) -> String {
    blk(block, LN1, Part::Bias)
}

/// `v.blk.{block}.ln2.bias`.
pub fn ln2_bias(block: usize) -> String {
    blk(block, LN2, Part::Bias)
}

/// `v.blk.{block}.ffn_up.bias`.
pub fn ffn_up_bias(block: usize) -> String {
    blk(block, FFN_UP, Part::Bias)
}

/// `v.blk.{block}.ffn_down.bias`.
pub fn ffn_down_bias(block: usize) -> String {
    blk(block, FFN_DOWN, Part::Bias)
}

/// `v.post_ln.bias`.
pub fn post_ln_bias() -> String {
    vit(POST_LN, Part::Bias)
}

/// `mm.0.weight` — the merger's first layer, from the four merged rows to the same width.
pub fn mm0_weight() -> String {
    mm(0, Part::Weight)
}

/// `mm.0.bias`.
pub fn mm0_bias() -> String {
    mm(0, Part::Bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name of the file as llama.cpp's converter writes it.
    #[test]
    fn the_names_are_the_files() {
        assert_eq!(patch_embd_weight(), "v.patch_embd.weight");
        assert_eq!(patch_embd_weight_second_frame(), "v.patch_embd.weight.1");
        assert_eq!(patch_embd_bias(), "v.patch_embd.bias");
        assert_eq!(position_embd(), "v.position_embd.weight");
        assert_eq!(ln1_weight(0), "v.blk.0.ln1.weight");
        assert_eq!(ln1_bias(1), "v.blk.1.ln1.bias");
        assert_eq!(attn_qkv_weight(26), "v.blk.26.attn_qkv.weight");
        assert_eq!(attn_qkv_bias(2), "v.blk.2.attn_qkv.bias");
        assert_eq!(attn_out_weight(3), "v.blk.3.attn_out.weight");
        assert_eq!(attn_out_bias(4), "v.blk.4.attn_out.bias");
        assert_eq!(ln2_weight(5), "v.blk.5.ln2.weight");
        assert_eq!(ln2_bias(6), "v.blk.6.ln2.bias");
        assert_eq!(ffn_up_weight(7), "v.blk.7.ffn_up.weight");
        assert_eq!(ffn_up_bias(8), "v.blk.8.ffn_up.bias");
        assert_eq!(ffn_down_weight(9), "v.blk.9.ffn_down.weight");
        assert_eq!(ffn_down_bias(10), "v.blk.10.ffn_down.bias");
        assert_eq!(post_ln_weight(), "v.post_ln.weight");
        assert_eq!(post_ln_bias(), "v.post_ln.bias");
        assert_eq!(mm0_weight(), "mm.0.weight");
        assert_eq!(mm0_bias(), "mm.0.bias");
        assert_eq!(mm2_weight(), "mm.2.weight");
        assert_eq!(mm2_bias(), "mm.2.bias");
    }
}
