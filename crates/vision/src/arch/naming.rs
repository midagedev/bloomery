//! The tensor-name scheme of an encoder file, written once: every projector type's names are
//! `v.blk.{block}.{stem}.{part}` inside a block, `v.{stem}.{part}` outside the blocks and
//! `mm.{index}.{part}` for the projector's layers. The names two projector types both hold are
//! functions here, re-exported by each module's `names`; a module's own `names` adds the stems
//! only it holds, built with [`blk`], [`vit`] and [`mm`].

/// The two leaves a tensor name ends in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Part {
    Weight,
    Bias,
}

impl Part {
    fn leaf(self) -> &'static str {
        match self {
            Part::Weight => "weight",
            Part::Bias => "bias",
        }
    }
}

pub(crate) const PATCH_EMBD: &str = "patch_embd";
pub(crate) const POST_LN: &str = "post_ln";
pub(crate) const LN1: &str = "ln1";
pub(crate) const LN2: &str = "ln2";
pub(crate) const ATTN_QKV: &str = "attn_qkv";
pub(crate) const ATTN_OUT: &str = "attn_out";
pub(crate) const FFN_UP: &str = "ffn_up";
pub(crate) const FFN_DOWN: &str = "ffn_down";

/// `v.blk.{block}.{stem}.{part}`.
pub(crate) fn blk(block: usize, stem: &str, part: Part) -> String {
    format!("v.blk.{block}.{stem}.{}", part.leaf())
}

/// `v.{stem}.{part}`: a tensor outside the blocks.
pub(crate) fn vit(stem: &str, part: Part) -> String {
    format!("v.{stem}.{}", part.leaf())
}

/// `mm.{index}.{part}`: layer `index` of the projector.
pub(crate) fn mm(index: usize, part: Part) -> String {
    format!("mm.{index}.{}", part.leaf())
}

/// `v.patch_embd.weight` — the patch embedding, ggml dims `[p, p, 3, dim]`: the same bytes as a
/// `[dim, 3·p·p]` linear weight, the 3·p·p axis in (channel, row, column) order.
pub fn patch_embd_weight() -> String {
    vit(PATCH_EMBD, Part::Weight)
}

/// `v.patch_embd.bias`.
pub fn patch_embd_bias() -> String {
    vit(PATCH_EMBD, Part::Bias)
}

/// `v.blk.{block}.ln1.weight` — the norm gain before attention.
pub fn ln1_weight(block: usize) -> String {
    blk(block, LN1, Part::Weight)
}

/// `v.blk.{block}.attn_qkv.weight` — query, key and value rows in that order.
pub fn attn_qkv_weight(block: usize) -> String {
    blk(block, ATTN_QKV, Part::Weight)
}

/// `v.blk.{block}.attn_qkv.bias`.
pub fn attn_qkv_bias(block: usize) -> String {
    blk(block, ATTN_QKV, Part::Bias)
}

/// `v.blk.{block}.attn_out.weight` — the attention output projection.
pub fn attn_out_weight(block: usize) -> String {
    blk(block, ATTN_OUT, Part::Weight)
}

/// `v.blk.{block}.attn_out.bias`.
pub fn attn_out_bias(block: usize) -> String {
    blk(block, ATTN_OUT, Part::Bias)
}

/// `v.blk.{block}.ln2.weight` — the norm gain before the MLP.
pub fn ln2_weight(block: usize) -> String {
    blk(block, LN2, Part::Weight)
}

/// `v.blk.{block}.ffn_up.weight`.
pub fn ffn_up_weight(block: usize) -> String {
    blk(block, FFN_UP, Part::Weight)
}

/// `v.blk.{block}.ffn_down.weight`.
pub fn ffn_down_weight(block: usize) -> String {
    blk(block, FFN_DOWN, Part::Weight)
}

/// `v.post_ln.weight` — the norm gain after the last block.
pub fn post_ln_weight() -> String {
    vit(POST_LN, Part::Weight)
}

/// `mm.2.weight` — the projector's last layer, to the text model's width.
pub fn mm2_weight() -> String {
    mm(2, Part::Weight)
}

/// `mm.2.bias`.
pub fn mm2_bias() -> String {
    mm(2, Part::Bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_three_shapes() {
        assert_eq!(blk(26, ATTN_QKV, Part::Bias), "v.blk.26.attn_qkv.bias");
        assert_eq!(vit(PATCH_EMBD, Part::Weight), "v.patch_embd.weight");
        assert_eq!(mm(0, Part::Bias), "mm.0.bias");
    }

    /// The names both projector types hold, as the files write them.
    #[test]
    fn the_shared_names_are_the_files() {
        assert_eq!(patch_embd_weight(), "v.patch_embd.weight");
        assert_eq!(patch_embd_bias(), "v.patch_embd.bias");
        assert_eq!(ln1_weight(3), "v.blk.3.ln1.weight");
        assert_eq!(attn_qkv_weight(31), "v.blk.31.attn_qkv.weight");
        assert_eq!(attn_qkv_bias(0), "v.blk.0.attn_qkv.bias");
        assert_eq!(attn_out_weight(1), "v.blk.1.attn_out.weight");
        assert_eq!(attn_out_bias(2), "v.blk.2.attn_out.bias");
        assert_eq!(ln2_weight(4), "v.blk.4.ln2.weight");
        assert_eq!(ffn_up_weight(6), "v.blk.6.ffn_up.weight");
        assert_eq!(ffn_down_weight(7), "v.blk.7.ffn_down.weight");
        assert_eq!(post_ln_weight(), "v.post_ln.weight");
        assert_eq!(mm2_weight(), "mm.2.weight");
        assert_eq!(mm2_bias(), "mm.2.bias");
    }
}
