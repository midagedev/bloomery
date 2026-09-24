//! Every tensor name of a `deepseek41v` encoder file, in one place: the patch embedding, the
//! `v.blk.{N}.*` names every ViT block carries, the final norm, the aligner (`mm.1`, `mm.2`) and
//! the delimiters. [`super::tensors`] checks a file against this table.
//!
//! The reference module each name is read from (vision.py, model.py) is named per entry.

/// `v.patch_embd.weight` — `PatchEmbed.proj.weight`, ggml dims `[p, p, 3, dim]`: the same bytes as
/// the `[dim, 3·p·p]` linear weight, the 3·p·p axis in (channel, row, column) order.
pub fn patch_embd_weight() -> String {
    "v.patch_embd.weight".to_string()
}

/// `v.patch_embd.bias` — `PatchEmbed.proj.bias`.
pub fn patch_embd_bias() -> String {
    "v.patch_embd.bias".to_string()
}

/// `v.blk.{block}.ln1.weight` — `Block.norm1`, the RMS gain before attention.
pub fn ln1(block: usize) -> String {
    format!("v.blk.{block}.ln1.weight")
}

/// `v.blk.{block}.attn_qkv.weight` — `Attention.wqkv.weight`, query, key and value rows in that order.
pub fn attn_qkv_weight(block: usize) -> String {
    format!("v.blk.{block}.attn_qkv.weight")
}

/// `v.blk.{block}.attn_qkv.bias` — `Attention.wqkv.bias`.
pub fn attn_qkv_bias(block: usize) -> String {
    format!("v.blk.{block}.attn_qkv.bias")
}

/// `v.blk.{block}.attn_out.weight` — `Attention.wo.weight`.
pub fn attn_out_weight(block: usize) -> String {
    format!("v.blk.{block}.attn_out.weight")
}

/// `v.blk.{block}.attn_out.bias` — `Attention.wo.bias`.
pub fn attn_out_bias(block: usize) -> String {
    format!("v.blk.{block}.attn_out.bias")
}

/// `v.blk.{block}.ln2.weight` — `Block.norm2`, the RMS gain before the MLP.
pub fn ln2(block: usize) -> String {
    format!("v.blk.{block}.ln2.weight")
}

/// `v.blk.{block}.ffn_gate.weight` — the gate half of `MLP.w1` (its first `ff` rows).
pub fn ffn_gate(block: usize) -> String {
    format!("v.blk.{block}.ffn_gate.weight")
}

/// `v.blk.{block}.ffn_up.weight` — the up half of `MLP.w1` (its last `ff` rows).
pub fn ffn_up(block: usize) -> String {
    format!("v.blk.{block}.ffn_up.weight")
}

/// `v.blk.{block}.ffn_down.weight` — `MLP.w2`.
pub fn ffn_down(block: usize) -> String {
    format!("v.blk.{block}.ffn_down.weight")
}

/// `v.post_ln.weight` — `ViT.norm`, the RMS gain after the last block.
pub fn post_ln() -> String {
    "v.post_ln.weight".to_string()
}

/// `mm.1.weight` — `Aligner.w1.weight`, from the 3×3-unfolded ViT rows to the text width.
pub fn mm1_weight() -> String {
    "mm.1.weight".to_string()
}

/// `mm.1.bias` — `Aligner.w1.bias`.
pub fn mm1_bias() -> String {
    "mm.1.bias".to_string()
}

/// `mm.2.weight` — `Aligner.w2.weight`, after the GELU.
pub fn mm2_weight() -> String {
    "mm.2.weight".to_string()
}

/// `mm.2.bias` — `Aligner.w2.bias`.
pub fn mm2_bias() -> String {
    "mm.2.bias".to_string()
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
