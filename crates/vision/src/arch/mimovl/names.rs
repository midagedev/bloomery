//! Every tensor name of a `mimovl` encoder file's vision half: the patch embedding (two temporal
//! halves), the final norm, the `v.blk.{N}.*` names every ViT block carries (a softmax sink among
//! them on the windowed blocks) and the merger's two layers. [`super::tensors`] checks a file
//! against this list.
//!
//! There is no position table and no patch bias. Norm gains have no bias (RMS), except the
//! merger's LayerNorm, which has a gain only, and the merger's linears have no bias. The names
//! every projector type holds are [`crate::arch::naming`]'s, re-exported here; the strings are
//! built there.

use crate::arch::naming::{Part, mm};
pub use crate::arch::naming::{
    attn_out_bias, attn_out_weight, attn_qkv_bias, attn_qkv_weight, ffn_down_bias, ffn_down_weight,
    ffn_gate_bias, ffn_gate_weight, ffn_up_bias, ffn_up_weight, ln1_weight, ln2_weight, mm2_weight,
    patch_embd_weight, patch_embd_weight_second_frame, post_ln_weight,
};

/// `v.blk.{block}.attn_sinks` — one softmax sink per query head (`[n_head]`, f32), on the windowed
/// blocks only.
pub fn attn_sinks(block: usize) -> String {
    format!("v.blk.{block}.attn_sinks")
}

/// `mm.0.weight` — the merger's first layer, from the four merged rows to the same width.
pub fn mm0_weight() -> String {
    mm(0, Part::Weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name of the file as llama.cpp's converter writes it (the MiMo-V2.6-Flash mmproj header).
    #[test]
    fn the_names_are_the_files() {
        assert_eq!(patch_embd_weight(), "v.patch_embd.weight");
        assert_eq!(patch_embd_weight_second_frame(), "v.patch_embd.weight.1");
        assert_eq!(ln1_weight(0), "v.blk.0.ln1.weight");
        assert_eq!(ln2_weight(5), "v.blk.5.ln2.weight");
        assert_eq!(attn_qkv_weight(27), "v.blk.27.attn_qkv.weight");
        assert_eq!(attn_qkv_bias(2), "v.blk.2.attn_qkv.bias");
        assert_eq!(attn_out_weight(3), "v.blk.3.attn_out.weight");
        assert_eq!(attn_out_bias(4), "v.blk.4.attn_out.bias");
        assert_eq!(attn_sinks(14), "v.blk.14.attn_sinks");
        assert_eq!(ffn_gate_weight(6), "v.blk.6.ffn_gate.weight");
        assert_eq!(ffn_gate_bias(6), "v.blk.6.ffn_gate.bias");
        assert_eq!(ffn_up_weight(7), "v.blk.7.ffn_up.weight");
        assert_eq!(ffn_up_bias(8), "v.blk.8.ffn_up.bias");
        assert_eq!(ffn_down_weight(9), "v.blk.9.ffn_down.weight");
        assert_eq!(ffn_down_bias(10), "v.blk.10.ffn_down.bias");
        assert_eq!(post_ln_weight(), "v.post_ln.weight");
        assert_eq!(mm0_weight(), "mm.0.weight");
        assert_eq!(mm2_weight(), "mm.2.weight");
    }
}
