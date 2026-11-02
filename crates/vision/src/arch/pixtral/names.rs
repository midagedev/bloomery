//! Every tensor name of a `pixtral` encoder file: the patch embedding, the norm before the first
//! block, the `v.blk.{N}.*` names every ViT block carries (query, key and value apart, a gated MLP,
//! no bias anywhere), the projector's input norm and patch merger, its two linears, and the text
//! model's `[IMG_BREAK]` row. [`super::tensors`] checks a file against this list.
//!
//! The file has no patch bias, no final norm and no bias in any block or in the projector.

use crate::arch::naming::{Part, blk, mm, vit};

pub use crate::arch::naming::{
    attn_out_weight, ffn_down_weight, ffn_up_weight, ln1_weight, ln2_weight, patch_embd_weight,
};

const PRE_LN: &str = "pre_ln";
const ATTN_Q: &str = "attn_q";
const ATTN_K: &str = "attn_k";
const ATTN_V: &str = "attn_v";
const FFN_GATE: &str = "ffn_gate";

/// `v.pre_ln.weight` — the RMS norm gain over the patch embeddings before block 0.
pub fn pre_ln_weight() -> String {
    vit(PRE_LN, Part::Weight)
}

macro_rules! block_names {
    ($($weight:ident, $stem:ident, $doc:literal;)*) => {$(
        #[doc = concat!("`v.blk.{block}.", $doc, ".weight`.")]
        pub fn $weight(block: usize) -> String {
            blk(block, $stem, Part::Weight)
        }
    )*};
}

block_names! {
    attn_q_weight, ATTN_Q, "attn_q";
    attn_k_weight, ATTN_K, "attn_k";
    attn_v_weight, ATTN_V, "attn_v";
    ffn_gate_weight, FFN_GATE, "ffn_gate";
}

/// `mm.input_norm.weight` — the projector's RMS norm gain over the ViT's output rows.
pub fn input_norm_weight() -> String {
    "mm.input_norm.weight".to_string()
}

/// `mm.patch_merger.weight` — the 2×2 merger, ggml dims `[merge²·dim, dim]`, each row's K in
/// `im2col`'s order ([`crate::arch::merger::conv_k_in_file`]).
pub fn patch_merger_weight() -> String {
    "mm.patch_merger.weight".to_string()
}

/// `mm.1.weight` — from the merged rows to the text model's width.
pub fn mm1_weight() -> String {
    mm(1, Part::Weight)
}

/// `mm.2.weight` — the text model's width to itself, after the GELU.
pub fn mm2_weight() -> String {
    mm(2, Part::Weight)
}

/// `v.token_embd.img_break` — the text model's `[IMG_BREAK]` embedding row, `out_dim` values; the
/// name has no leaf.
pub fn img_break() -> String {
    "v.token_embd.img_break".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name of the file as llama.cpp's converter writes it.
    #[test]
    fn the_names_are_the_files() {
        assert_eq!(patch_embd_weight(), "v.patch_embd.weight");
        assert_eq!(pre_ln_weight(), "v.pre_ln.weight");
        assert_eq!(ln1_weight(0), "v.blk.0.ln1.weight");
        assert_eq!(ln2_weight(5), "v.blk.5.ln2.weight");
        assert_eq!(attn_q_weight(29), "v.blk.29.attn_q.weight");
        assert_eq!(attn_k_weight(3), "v.blk.3.attn_k.weight");
        assert_eq!(attn_v_weight(7), "v.blk.7.attn_v.weight");
        assert_eq!(attn_out_weight(9), "v.blk.9.attn_out.weight");
        assert_eq!(ffn_gate_weight(10), "v.blk.10.ffn_gate.weight");
        assert_eq!(ffn_up_weight(11), "v.blk.11.ffn_up.weight");
        assert_eq!(ffn_down_weight(13), "v.blk.13.ffn_down.weight");
        assert_eq!(input_norm_weight(), "mm.input_norm.weight");
        assert_eq!(patch_merger_weight(), "mm.patch_merger.weight");
        assert_eq!(mm1_weight(), "mm.1.weight");
        assert_eq!(mm2_weight(), "mm.2.weight");
        assert_eq!(img_break(), "v.token_embd.img_break");
    }
}
