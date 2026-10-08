//! Every tensor name of a `deepseek41v` encoder file, in one place: the patch embedding, the
//! `v.blk.{N}.*` names every ViT block carries, the final norm, the aligner (`mm.1`, `mm.2`) and
//! the delimiters. [`super::tensors`] checks a file against this table.
//!
//! The names every projector type holds are [`crate::arch::naming`]'s, re-exported here (the norm
//! gains `ln1`, `ln2` and `post_ln`, and the MLP's `ffn_up` and `ffn_down`, under this module's
//! names for them); the strings are built there. The gate half of the MLP, `ffn_gate`, is V4.1's
//! alone (its MLP is SiLU-gated) and is built here.
//!
//! The reference module (vision.py, model.py) each is read from: `patch_embd` is
//! `PatchEmbed.proj`, `ln1` and `ln2` are `Block.norm1` and `Block.norm2` (an RMS gain), `attn_qkv`
//! is `Attention.wqkv` (the q, k and v rows in that order), `attn_out` is `Attention.wo`, `ffn_up`
//! and `ffn_down` are `MLP.w1`'s up half and `MLP.w2`, `post_ln` is `ViT.norm`, `mm.2` is
//! `Aligner.w2`.

use crate::arch::naming::{Part, blk, mm};
pub use crate::arch::naming::{
    attn_out_bias, attn_out_weight, attn_qkv_bias, attn_qkv_weight, ffn_down_weight as ffn_down,
    ffn_up_weight as ffn_up, ln1_weight as ln1, ln2_weight as ln2, mm2_bias, mm2_weight,
    patch_embd_bias, patch_embd_weight, post_ln_weight as post_ln,
};

/// The stem of the gate half of the MLP, which only a SiLU-gated MLP has.
const FFN_GATE: &str = "ffn_gate";

/// `v.blk.{block}.ffn_gate.weight` — the gate half of `MLP.w1` (its first `ff` rows).
pub fn ffn_gate(block: usize) -> String {
    blk(block, FFN_GATE, Part::Weight)
}

/// `mm.1.weight` — `Aligner.w1.weight`, from the 3×3-unfolded ViT rows to the text width.
pub fn mm1_weight() -> String {
    mm(1, Part::Weight)
}

/// `mm.1.bias` — `Aligner.w1.bias`.
pub fn mm1_bias() -> String {
    mm(1, Part::Bias)
}

/// `v.token_embd.img_start` — `Transformer.image_start`, the row of an image span's first position.
pub fn img_start() -> String {
    "v.token_embd.img_start".to_string()
}

/// `v.token_embd.img_end` — `Transformer.image_end`, the row of its last position.
pub fn img_end() -> String {
    "v.token_embd.img_end".to_string()
}

/// `v.image_newline` — `Transformer.image_newline`, the row after every token row.
pub fn image_newline() -> String {
    "v.image_newline".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name of the file as the reference's GGUF writes it.
    #[test]
    fn the_names_are_the_files() {
        assert_eq!(patch_embd_weight(), "v.patch_embd.weight");
        assert_eq!(patch_embd_bias(), "v.patch_embd.bias");
        assert_eq!(ln1(3), "v.blk.3.ln1.weight");
        assert_eq!(attn_qkv_weight(31), "v.blk.31.attn_qkv.weight");
        assert_eq!(attn_qkv_bias(0), "v.blk.0.attn_qkv.bias");
        assert_eq!(attn_out_weight(1), "v.blk.1.attn_out.weight");
        assert_eq!(attn_out_bias(2), "v.blk.2.attn_out.bias");
        assert_eq!(ln2(4), "v.blk.4.ln2.weight");
        assert_eq!(ffn_gate(5), "v.blk.5.ffn_gate.weight");
        assert_eq!(ffn_up(6), "v.blk.6.ffn_up.weight");
        assert_eq!(ffn_down(7), "v.blk.7.ffn_down.weight");
        assert_eq!(post_ln(), "v.post_ln.weight");
        assert_eq!(mm1_weight(), "mm.1.weight");
        assert_eq!(mm1_bias(), "mm.1.bias");
        assert_eq!(mm2_weight(), "mm.2.weight");
        assert_eq!(mm2_bias(), "mm.2.bias");
        assert_eq!(img_start(), "v.token_embd.img_start");
        assert_eq!(img_end(), "v.token_embd.img_end");
        assert_eq!(image_newline(), "v.image_newline");
    }
}
