//! Every tensor name of a `minicpmv4_7` encoder file: the patch embedding, the learned positions,
//! the separator rows, the `v.blk.{N}.*` names every ViT block carries (q, k and v are separate
//! tensors here), the final norm, the ViT merger after block 6 and the final merger.
//! [`super::tensors`] checks a file against this list.
//!
//! The names every projector type holds are [`crate::arch::naming`]'s, re-exported here.

use crate::arch::naming::{LN1, LN2, Part, blk, vit};
pub use crate::arch::naming::{
    attn_out_bias, attn_out_weight, ffn_down_weight, ffn_up_weight, ln1_weight, ln2_weight,
    patch_embd_bias, patch_embd_weight, post_ln_weight,
};

const ATTN_Q: &str = "attn_q";
const ATTN_K: &str = "attn_k";
const ATTN_V: &str = "attn_v";
const FFN_UP: &str = "ffn_up";
const FFN_DOWN: &str = "ffn_down";

/// `v.tok_embd_sep` — the four separator rows `[text width, 4]` (no `.weight` leaf): `</image>`,
/// `<slice>`, `</slice>`, `\n`, the rows of the text model's embedding table for those tokens.
pub fn tok_embd_sep() -> String {
    "v.tok_embd_sep".to_string()
}

/// `v.position_embd.weight` — the learned position table, one row per cell of a 70×70 grid.
pub fn position_embd() -> String {
    vit("position_embd", Part::Weight)
}

/// `v.post_ln.bias`.
pub fn post_ln_bias() -> String {
    vit("post_ln", Part::Bias)
}

/// `v.blk.{block}.ln1.bias`.
pub fn ln1_bias(block: usize) -> String {
    blk(block, LN1, Part::Bias)
}

/// `v.blk.{block}.ln2.bias`.
pub fn ln2_bias(block: usize) -> String {
    blk(block, LN2, Part::Bias)
}

/// `v.blk.{block}.attn_q.weight`.
pub fn attn_q_weight(block: usize) -> String {
    blk(block, ATTN_Q, Part::Weight)
}

/// `v.blk.{block}.attn_q.bias`.
pub fn attn_q_bias(block: usize) -> String {
    blk(block, ATTN_Q, Part::Bias)
}

/// `v.blk.{block}.attn_k.weight`.
pub fn attn_k_weight(block: usize) -> String {
    blk(block, ATTN_K, Part::Weight)
}

/// `v.blk.{block}.attn_k.bias`.
pub fn attn_k_bias(block: usize) -> String {
    blk(block, ATTN_K, Part::Bias)
}

/// `v.blk.{block}.attn_v.weight`.
pub fn attn_v_weight(block: usize) -> String {
    blk(block, ATTN_V, Part::Weight)
}

/// `v.blk.{block}.attn_v.bias`.
pub fn attn_v_bias(block: usize) -> String {
    blk(block, ATTN_V, Part::Bias)
}

/// `v.blk.{block}.ffn_up.bias`.
pub fn ffn_up_bias(block: usize) -> String {
    blk(block, FFN_UP, Part::Bias)
}

/// `v.blk.{block}.ffn_down.bias`.
pub fn ffn_down_bias(block: usize) -> String {
    blk(block, FFN_DOWN, Part::Bias)
}

/// `v.vit_merger.{stem}.weight`: the 2×2 merger after block 6 — `ln1`, `attn_q`, `attn_k`,
/// `attn_v`, `attn_out` (a window self-attention over the merge window), then `ds_ln`,
/// `ds_ffn_up`, `ds_ffn_down` (the MLP that folds the window's four rows).
pub fn vit_merger_weight(stem: &str) -> String {
    vit(&format!("vit_merger.{stem}"), Part::Weight)
}

/// `v.vit_merger.{stem}.bias`.
pub fn vit_merger_bias(stem: &str) -> String {
    vit(&format!("vit_merger.{stem}"), Part::Bias)
}

/// `mm.{stem}.weight`: the final merger — `input_norm`, `up`, `down`.
pub fn mm_weight(stem: &str) -> String {
    format!("mm.{stem}.weight")
}

/// `mm.{stem}.bias`.
pub fn mm_bias(stem: &str) -> String {
    format!("mm.{stem}.bias")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Names as the file's header dump spells them.
    #[test]
    fn the_names_are_the_files() {
        assert_eq!(tok_embd_sep(), "v.tok_embd_sep");
        assert_eq!(patch_embd_weight(), "v.patch_embd.weight");
        assert_eq!(patch_embd_bias(), "v.patch_embd.bias");
        assert_eq!(position_embd(), "v.position_embd.weight");
        assert_eq!(ln1_weight(0), "v.blk.0.ln1.weight");
        assert_eq!(ln2_bias(26), "v.blk.26.ln2.bias");
        assert_eq!(attn_q_weight(10), "v.blk.10.attn_q.weight");
        assert_eq!(attn_k_bias(1), "v.blk.1.attn_k.bias");
        assert_eq!(attn_v_weight(2), "v.blk.2.attn_v.weight");
        assert_eq!(attn_out_weight(3), "v.blk.3.attn_out.weight");
        assert_eq!(ffn_up_weight(7), "v.blk.7.ffn_up.weight");
        assert_eq!(ffn_down_bias(9), "v.blk.9.ffn_down.bias");
        assert_eq!(post_ln_weight(), "v.post_ln.weight");
        assert_eq!(vit_merger_weight("ln1"), "v.vit_merger.ln1.weight");
        assert_eq!(vit_merger_bias("ds_ffn_up"), "v.vit_merger.ds_ffn_up.bias");
        assert_eq!(
            vit_merger_weight("attn_out"),
            "v.vit_merger.attn_out.weight"
        );
        assert_eq!(mm_weight("up"), "mm.up.weight");
        assert_eq!(mm_bias("down"), "mm.down.bias");
        assert_eq!(mm_weight("input_norm"), "mm.input_norm.weight");
    }
}
