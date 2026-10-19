//! Every tensor name of a `nemotron_omni` encoder file: the patch embedding (an image kernel and
//! the video kernel of six channels), the prefix rows and the position table, the
//! `v.blk.{N}.*` names every ViT block carries (query, key and value apart, each with a bias) and
//! the projector's three tensors. [`super::tensors`] checks a file against this list.
//!
//! The file has no patch bias, no final norm and no per-block fused projection; the projector's
//! names are `mm.model.mlp.{0,1,3}.weight`.

use crate::arch::naming::{Part, blk, vit};

const PATCH_EMBD: &str = "patch_embd";
const POSITION_EMBD: &str = "position_embd";
const LN1: &str = "ln1";
const LN2: &str = "ln2";
const ATTN_Q: &str = "attn_q";
const ATTN_K: &str = "attn_k";
const ATTN_V: &str = "attn_v";
const ATTN_OUT: &str = "attn_out";
const FFN_UP: &str = "ffn_up";
const FFN_DOWN: &str = "ffn_down";

/// `v.patch_embd.weight` — the image's patch kernel, ggml dims `[p, p, 3, dim]`: the same bytes
/// as a `[dim, 3·p·p]` linear weight, the 3·p·p axis in (channel, row, column) order.
pub fn patch_embd_weight() -> String {
    vit(PATCH_EMBD, Part::Weight)
}

/// `v.patch_embd.weight.1` — the video kernel over two frames of three channels, `[p, p, 6,
/// dim]`. A still image does not use it.
pub fn patch_embd_weight_video() -> String {
    format!("{}.1", patch_embd_weight())
}

/// `v.class_embd` — the prefix rows, ggml dims `[dim, PREFIX_ROWS]`; the name has no leaf.
pub fn class_embd() -> String {
    "v.class_embd".to_string()
}

/// `v.position_embd.weight` — the learned position table, one row per position of a square grid.
pub fn position_embd() -> String {
    vit(POSITION_EMBD, Part::Weight)
}

macro_rules! block_names {
    ($($weight:ident, $bias:ident, $stem:ident, $doc:literal;)*) => {$(
        #[doc = concat!("`v.blk.{block}.", $doc, ".weight`.")]
        pub fn $weight(block: usize) -> String {
            blk(block, $stem, Part::Weight)
        }
        #[doc = concat!("`v.blk.{block}.", $doc, ".bias`.")]
        pub fn $bias(block: usize) -> String {
            blk(block, $stem, Part::Bias)
        }
    )*};
}

block_names! {
    ln1_weight, ln1_bias, LN1, "ln1";
    ln2_weight, ln2_bias, LN2, "ln2";
    attn_q_weight, attn_q_bias, ATTN_Q, "attn_q";
    attn_k_weight, attn_k_bias, ATTN_K, "attn_k";
    attn_v_weight, attn_v_bias, ATTN_V, "attn_v";
    attn_out_weight, attn_out_bias, ATTN_OUT, "attn_out";
    ffn_up_weight, ffn_up_bias, FFN_UP, "ffn_up";
    ffn_down_weight, ffn_down_bias, FFN_DOWN, "ffn_down";
}

/// `mm.model.mlp.0.weight` — the projector's RMS norm gain over the merged rows.
pub fn mlp_norm() -> String {
    "mm.model.mlp.0.weight".to_string()
}

/// `mm.model.mlp.1.weight` — from the merged rows to the projector's hidden width.
pub fn mlp_up() -> String {
    "mm.model.mlp.1.weight".to_string()
}

/// `mm.model.mlp.3.weight` — from the hidden width to the text model's width.
pub fn mlp_down() -> String {
    "mm.model.mlp.3.weight".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name of the file as llama.cpp's converter writes it.
    #[test]
    fn the_names_are_the_files() {
        assert_eq!(patch_embd_weight(), "v.patch_embd.weight");
        assert_eq!(patch_embd_weight_video(), "v.patch_embd.weight.1");
        assert_eq!(class_embd(), "v.class_embd");
        assert_eq!(position_embd(), "v.position_embd.weight");
        assert_eq!(ln1_weight(0), "v.blk.0.ln1.weight");
        assert_eq!(ln1_bias(1), "v.blk.1.ln1.bias");
        assert_eq!(ln2_weight(5), "v.blk.5.ln2.weight");
        assert_eq!(ln2_bias(6), "v.blk.6.ln2.bias");
        assert_eq!(attn_q_weight(31), "v.blk.31.attn_q.weight");
        assert_eq!(attn_q_bias(2), "v.blk.2.attn_q.bias");
        assert_eq!(attn_k_weight(3), "v.blk.3.attn_k.weight");
        assert_eq!(attn_k_bias(4), "v.blk.4.attn_k.bias");
        assert_eq!(attn_v_weight(7), "v.blk.7.attn_v.weight");
        assert_eq!(attn_v_bias(8), "v.blk.8.attn_v.bias");
        assert_eq!(attn_out_weight(9), "v.blk.9.attn_out.weight");
        assert_eq!(attn_out_bias(10), "v.blk.10.attn_out.bias");
        assert_eq!(ffn_up_weight(11), "v.blk.11.ffn_up.weight");
        assert_eq!(ffn_up_bias(12), "v.blk.12.ffn_up.bias");
        assert_eq!(ffn_down_weight(13), "v.blk.13.ffn_down.weight");
        assert_eq!(ffn_down_bias(14), "v.blk.14.ffn_down.bias");
        assert_eq!(mlp_norm(), "mm.model.mlp.0.weight");
        assert_eq!(mlp_up(), "mm.model.mlp.1.weight");
        assert_eq!(mlp_down(), "mm.model.mlp.3.weight");
    }
}
