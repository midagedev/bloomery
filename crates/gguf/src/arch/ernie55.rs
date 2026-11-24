//! ERNIE 5.5's header facts: the architecture word and the quant types each tensor stem may carry.
//!
//! The table is cut from what the checkpoint folder stores and the map writes of it
//! (`bloomery_model_hf::arch::ernie55`), one tensor of the program a stored tensor, no move:
//! - the dense projections are F8_E4M3 beside an f32 scale a 128 by 128 block, held as
//!   [`GgmlType::FP8_B128`] (132 bytes a 128 values, the scale of the checkpoint's block a row
//!   carries); the stored BF16 matrices (the compressor's, the indexer's) stay BF16 or are written
//!   as Q8_0;
//! - the routed stacks are OCP MXFP4, an E8M0 scale a 32 values ([`GgmlType::MXFP4`]), one matrix
//!   an expert;
//! - the embedding, the head and the two routers are BF16, every norm, mixer, bias and vector F32.
//!
//! Two layouts differ from the folder's, by the loader's own moves: a hyper-connection mixer is
//! stored `[in, out]` and written `[out, in]` (a transpose), and a mixer's three scalars
//! (`alpha_pre`, `alpha_post`, `alpha_res`) are written as the one `*_scale` vector of three. The
//! head mix is the pair `attn_vha_u` and `attn_vha_v` as stored (the mix is `I + V Uᵀ`), the two
//! routers are `ffn_gate_inp` and `ffn_gate_inp_1`, and a latent-attention layer's `attn_kv_b` is
//! the fused per-head `[nope + v, rank]` rows as stored, which the per-head key and value halves
//! the card reads are cut from by a later move.
//!
//! A type outside a stem's list is a file this table was not written for, and the reader refuses it
//! by name; a type a later quantization of this family writes is added here with the kernel that
//! reads it.

use crate::GgmlType;

/// `general.architecture` of an ERNIE 5.5 file.
pub const ARCH: &str = "ernie5_5";

/// `tokenizer.ggml.pre` of an ERNIE 5.5 file: the pre-tokenizer `bloomery_tokenizer` runs under
/// that name (the reference's digit rule, or Hugging Face's reading of ERNIE's own
/// `tokenizer.json` where the vocabulary carries `tokenizer.bloomery.eog_token_ids`) and the row
/// of `hf::vocab::PRE` the folder's pipeline is.
pub const PRE: &str = "ernie5";

const F32: &[GgmlType] = &[GgmlType::F32];
/// A projection the dense gemv sites run: the checkpoint's FP8 128-blocks, or Q8_0 where a lever
/// rewrites it.
const DENSE: &[GgmlType] = &[GgmlType::FP8_B128, GgmlType::Q8_0];
/// A matrix the checkpoint stores BF16: kept, or written as Q8_0.
const BF16_OR_Q8: &[GgmlType] = &[GgmlType::BF16, GgmlType::Q8_0];
/// The embedding, the head and the router stack: BF16 as stored.
const BF16: &[GgmlType] = &[GgmlType::BF16];
/// The routed gate, up and down stacks: MXFP4, one matrix an expert.
const ROUTED: &[GgmlType] = &[GgmlType::MXFP4];

/// Every tensor stem of an ERNIE 5.5 file (the name past `blk.N.` for a layer's tensor, the whole
/// name for a model-level one) with the types it may carry.
pub const STEM_TYPES: &[(&str, &[GgmlType])] = &[
    ("token_embd.weight", BF16),
    ("output.weight", BF16),
    ("output_norm.weight", F32),
    // The head's MultiMax: the range edges and the temperatures, four values each.
    ("output_multimax_ranges.weight", F32),
    ("output_multimax_ts.weight", F32),
    // The model-level hyper-connection head: the contraction's mixer, its bias and scale.
    ("output_hc_fn.weight", F32),
    ("output_hc_base.weight", F32),
    ("output_hc_scale.weight", F32),
    // Every layer: the two norms, the output gate, the head mix and the two hyper-connection mixers.
    ("attn_norm.weight", F32),
    ("ffn_norm.weight", F32),
    ("attn_gate.weight", DENSE),
    ("attn_vha_u.weight", F32),
    ("attn_vha_v.weight", F32),
    ("hc_attn_fn.weight", F32),
    ("hc_attn_base.weight", F32),
    ("hc_attn_scale.weight", F32),
    ("hc_ffn_fn.weight", F32),
    ("hc_ffn_base.weight", F32),
    ("hc_ffn_scale.weight", F32),
    // Both attention kinds' query path and sinks.
    ("attn_q_a.weight", DENSE),
    ("attn_q_a_norm.weight", F32),
    ("attn_q_b.weight", DENSE),
    ("attn_sinks.weight", F32),
    // A compressed-attention layer: the one key-value head and its latent norm, the grouped output
    // projection pair, and the compressor (its value and gate projections, position table and norm).
    ("attn_kv.weight", DENSE),
    ("attn_kv_a_norm.weight", F32),
    ("attn_output_a.weight", DENSE),
    ("attn_output_b.weight", DENSE),
    ("attn_compressor_kv.weight", BF16_OR_Q8),
    ("attn_compressor_gate.weight", BF16_OR_Q8),
    ("attn_compressor_ape.weight", F32),
    ("attn_compressor_norm.weight", F32),
    // A latent-attention layer: the key-value latent projection with the rope key, its norm, the
    // fused per-head up projection and the output projection.
    ("attn_kv_a_mqa.weight", DENSE),
    ("attn_kv_b.weight", DENSE),
    ("attn_output.weight", DENSE),
    // That layer's indexer: the query projection from the query latent, the key projection with its
    // LayerNorm (gain and bias) and the head weights.
    ("indexer.attn_q_b.weight", DENSE),
    ("indexer.attn_k.weight", BF16_OR_Q8),
    ("indexer.proj.weight", BF16_OR_Q8),
    ("indexer.k_norm.weight", F32),
    ("indexer.k_norm.bias", F32),
    // The dense layer's feed-forward block.
    ("ffn_gate.weight", DENSE),
    ("ffn_up.weight", DENSE),
    ("ffn_down.weight", DENSE),
    // A routed layer's block: the two routers, the selection bias, the learned per-expert scale, the
    // routed stacks, the latent pair with its norm and the shared expert.
    ("ffn_gate_inp.weight", BF16),
    ("ffn_gate_inp_1.weight", BF16),
    ("exp_probs_b.bias", F32),
    ("ffn_exp_scale.weight", F32),
    ("ffn_gate_exps.weight", ROUTED),
    ("ffn_up_exps.weight", ROUTED),
    ("ffn_down_exps.weight", ROUTED),
    ("ffn_routed_down.weight", DENSE),
    ("ffn_routed_up.weight", DENSE),
    ("ffn_routed_norm.weight", F32),
    ("ffn_gate_shexp.weight", DENSE),
    ("ffn_up_shexp.weight", DENSE),
    ("ffn_down_shexp.weight", DENSE),
];

/// The types a tensor of stem `stem` may carry; `None` for a stem no ERNIE 5.5 file holds.
#[must_use]
pub fn allowed_types(stem: &str) -> Option<&'static [GgmlType]> {
    STEM_TYPES
        .iter()
        .find(|(name, _)| *name == stem)
        .map(|&(_, types)| types)
}

#[cfg(test)]
mod tests {
    use super::{ARCH, STEM_TYPES, allowed_types};
    use crate::GgmlType::{BF16, F32, FP8_B128, MXFP4, Q8_0};

    /// The table holds every stem once, in the types the checkpoint's formats give them: the dense
    /// projections the 128-block FP8 type (or Q8_0), the routed stacks MXFP4 alone, both routers
    /// and the head BF16, the norms and mixers F32. Mutant: a stem listed twice, or the routed
    /// stacks' list widened past MXFP4.
    #[test]
    fn the_table_holds_the_checkpoints_types() {
        assert_eq!(ARCH, "ernie5_5");
        assert_eq!(
            allowed_types("attn_q_a.weight"),
            Some(&[FP8_B128, Q8_0][..])
        );
        for stem in [
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ] {
            assert_eq!(allowed_types(stem), Some(&[MXFP4][..]), "{stem}");
        }
        for stem in [
            "ffn_gate_inp.weight",
            "ffn_gate_inp_1.weight",
            "output.weight",
        ] {
            assert_eq!(allowed_types(stem), Some(&[BF16][..]), "{stem}");
        }
        assert_eq!(allowed_types("hc_attn_fn.weight"), Some(&[F32][..]));
        assert_eq!(
            allowed_types("indexer.k_norm.bias"),
            Some(&[F32][..]),
            "the indexer's LayerNorm keeps its bias"
        );
        for (i, (stem, _)) in STEM_TYPES.iter().enumerate() {
            assert!(
                STEM_TYPES[..i].iter().all(|(s, _)| s != stem),
                "{stem} is listed twice"
            );
        }
    }

    /// A stem no ERNIE 5.5 file holds is no stem: the stored spellings of a mixer's parts, the cut
    /// halves of the fused up projection, another family's projection, a layer-numbered name and
    /// the empty name have no list.
    #[test]
    fn a_stem_no_file_holds_has_no_list() {
        for stem in [
            "hc_attn_alpha_pre.weight",
            "hc_attn_mapping.weight",
            "attn_k_b.weight",
            "attn_qkv.weight",
            "blk.0.attn_gate.weight",
            "engram_kv.weight",
            "",
        ] {
            assert_eq!(allowed_types(stem), None, "{stem:?}");
        }
    }
}
