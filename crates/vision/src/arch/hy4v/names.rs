//! Every tensor name of a `hy_v4_flash` encoder file: the patch embedding with its bias, the
//! position table, the `v.blk.{N}.*` names every ViT block carries (query, key and value apart,
//! each with a bias), the projector's norm gain and two convolutions (`mm.0`, `mm.2`), and the
//! newline row. [`super::tensors`] checks a file against this list.
//!
//! The file has no pre or post norm of the ViT and no fused projection.

use crate::arch::naming::{Part, blk, mm, vit};

pub use crate::arch::naming::{patch_embd_bias, patch_embd_weight};

const POSITION_EMBD: &str = "position_embd";
const LN1: &str = "ln1";
const LN2: &str = "ln2";
const ATTN_Q: &str = "attn_q";
const ATTN_K: &str = "attn_k";
const ATTN_V: &str = "attn_v";
const ATTN_OUT: &str = "attn_out";
const FFN_UP: &str = "ffn_up";
const FFN_DOWN: &str = "ffn_down";

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

/// `mm.pre_norm.weight` — the projector's RMS norm gain over each patch's ViT row.
pub fn pre_norm_weight() -> String {
    "mm.pre_norm.weight".to_string()
}

/// `mm.0.weight` — the 2×2 strided convolution, ggml dims `[2, 2, dim, 4·dim]` (`perceive.proj.0`).
pub fn conv_weight() -> String {
    mm(0, Part::Weight)
}

/// `mm.0.bias`.
pub fn conv_bias() -> String {
    mm(0, Part::Bias)
}

/// `mm.2.weight` — the 1×1 convolution to the text model's width, ggml dims `[1, 1, 4·dim, out]`
/// (`perceive.proj.2`).
pub fn proj_weight() -> String {
    mm(2, Part::Weight)
}

/// `mm.2.bias`.
pub fn proj_bias() -> String {
    mm(2, Part::Bias)
}

/// `v.image_newline` — the learned row after every row of merged tokens; the name has no leaf.
pub fn image_newline() -> String {
    "v.image_newline".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name of the file as llama.cpp's converter writes it.
    #[test]
    fn the_names_are_the_files() {
        assert_eq!(patch_embd_weight(), "v.patch_embd.weight");
        assert_eq!(patch_embd_bias(), "v.patch_embd.bias");
        assert_eq!(position_embd(), "v.position_embd.weight");
        assert_eq!(ln1_weight(0), "v.blk.0.ln1.weight");
        assert_eq!(ln1_bias(1), "v.blk.1.ln1.bias");
        assert_eq!(ln2_weight(5), "v.blk.5.ln2.weight");
        assert_eq!(ln2_bias(6), "v.blk.6.ln2.bias");
        assert_eq!(attn_q_weight(26), "v.blk.26.attn_q.weight");
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
        assert_eq!(pre_norm_weight(), "mm.pre_norm.weight");
        assert_eq!(conv_weight(), "mm.0.weight");
        assert_eq!(conv_bias(), "mm.0.bias");
        assert_eq!(proj_weight(), "mm.2.weight");
        assert_eq!(proj_bias(), "mm.2.bias");
        assert_eq!(image_newline(), "v.image_newline");
    }
}
