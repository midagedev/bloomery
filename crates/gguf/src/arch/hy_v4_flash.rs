//! Hy4-flash's header facts: the architecture word and the quant types each tensor stem may carry.
//!
//! The table is cut from Tencent's Q4_K_M file (`Hy4-flash-Q4_K_M`, two shards, read from the tensor
//! table of shard 1) and from the BF16 form its next-token block is stored in:
//! - the trunk's projections are Q8_0, the routed gate and up stacks Q4_K, the down stack Q6_K on 19
//!   layers and Q4_K on 22, the head BF16, the token embedding Q8_0;
//! - the per-layer n-gram table (`per_layer_token_embd`) is ggml's F8_E4M3 under id 43, one byte a
//!   value and no tensor scale ([`GgmlType::F8E4M3`]: every row was scaled to the format's amax when
//!   the file was written and the scale dropped, so the first RMS norm of the engram delta undoes it);
//! - the next-token block (`blk.42`) keeps its projections and experts in BF16.
//!
//! A type outside a stem's list is a file this table was not written for, and the reader refuses it by
//! name; a type a later quantization of this family writes is added here with the kernel that reads it.

use crate::GgmlType;

/// `general.architecture` of a Hy4-flash file.
pub const ARCH: &str = "hy_v4_flash";

/// The per-layer n-gram table's tensor name.
pub const ENGRAM_TABLE: &str = "per_layer_token_embd.weight";

const F32: &[GgmlType] = &[GgmlType::F32];
/// A projection the dense gemv sites run: the file's Q8_0 and the next-token block's BF16.
const DENSE: &[GgmlType] = &[GgmlType::Q8_0, GgmlType::BF16];
/// The routed gate and up stacks: Q4_K in the trunk, BF16 in the next-token block.
const GATE_UP: &[GgmlType] = &[GgmlType::Q4_K, GgmlType::BF16];
/// The routed down stacks: Q6_K on 19 layers and Q4_K on 22 of the trunk, BF16 in the next-token block.
const DOWN: &[GgmlType] = &[GgmlType::Q6_K, GgmlType::Q4_K, GgmlType::BF16];
/// The token embedding: the card's row lookups read Q8_0.
const EMBED: &[GgmlType] = &[GgmlType::Q8_0];
/// The head: BF16.
const HEAD: &[GgmlType] = &[GgmlType::BF16];
/// The per-layer n-gram table: F8_E4M3 rows with no tensor scale.
const TABLE: &[GgmlType] = &[GgmlType::F8E4M3];

/// Every tensor stem of a Hy4-flash file (the name past `blk.N.` for a layer's tensor, the whole name
/// for a model-level one) with the types it may carry.
pub const STEM_TYPES: &[(&str, &[GgmlType])] = &[
    ("token_embd.weight", EMBED),
    ("output.weight", HEAD),
    ("output_norm.weight", F32),
    ("output_hc_fn.weight", F32),
    ("output_hc_base.weight", F32),
    ("output_hc_scale.weight", F32),
    (ENGRAM_TABLE, TABLE),
    // The attention block of every layer.
    ("attn_norm.weight", F32),
    ("attn_sinks.weight", F32),
    ("attn_q_a.weight", DENSE),
    ("attn_q_a_norm.weight", F32),
    ("attn_q_b.weight", DENSE),
    ("attn_kv_a_mqa.weight", DENSE),
    ("attn_kv_a_norm.weight", F32),
    ("attn_k_b.weight", DENSE),
    ("attn_v_b.weight", DENSE),
    ("attn_gate.weight", DENSE),
    ("attn_output.weight", DENSE),
    ("post_attention_norm.weight", F32),
    // The hyper-connection mixes of every trunk layer.
    ("hc_attn_fn.weight", F32),
    ("hc_attn_base.weight", F32),
    ("hc_attn_scale.weight", F32),
    ("hc_ffn_fn.weight", F32),
    ("hc_ffn_base.weight", F32),
    ("hc_ffn_scale.weight", F32),
    // The indexer of a sparse-attention layer: the query and key projections, the head weights, the key's
    // gain, and the pooling compressor's value and gate projections, position table and gain.
    ("indexer.attn_q_b.weight", DENSE),
    ("indexer.attn_k.weight", DENSE),
    ("indexer.proj.weight", &[GgmlType::F32, GgmlType::BF16]),
    ("indexer.k_norm.weight", F32),
    ("indexer_compressor_value.weight", DENSE),
    ("indexer_compressor_gate.weight", DENSE),
    ("indexer_compressor_ape.weight", F32),
    ("indexer_compressor_norm.weight", F32),
    // The engram site's delta: the fused key and value projection, its bias, the gate, the stream
    // scales, the lookup norm and the pre-gain.
    ("engram_kv.weight", DENSE),
    ("engram_kv_b.weight", F32),
    ("engram_gate.weight", F32),
    ("engram_hc_scale.weight", F32),
    ("engram_lookup_norm.weight", F32),
    ("engram_pre_gamma.weight", F32),
    // The feed-forward block: the dense layer's, the router, the routed stacks, the latent pair and
    // the shared expert.
    ("ffn_norm.weight", F32),
    ("post_ffw_norm.weight", F32),
    ("ffn_gate.weight", DENSE),
    ("ffn_up.weight", DENSE),
    ("ffn_down.weight", DENSE),
    ("ffn_gate_inp.weight", F32),
    ("exp_probs_b.bias", F32),
    ("ffn_gate_exps.weight", GATE_UP),
    ("ffn_up_exps.weight", GATE_UP),
    ("ffn_down_exps.weight", DOWN),
    ("ffn_routed_down.weight", DENSE),
    ("ffn_routed_up.weight", DENSE),
    ("ffn_gate_shexp.weight", DENSE),
    ("ffn_up_shexp.weight", DENSE),
    ("ffn_down_shexp.weight", DENSE),
    // The next-token block's own tensors.
    ("nextn.eh_proj.weight", &[GgmlType::BF16]),
    ("nextn.enorm.weight", F32),
    ("nextn.hnorm.weight", F32),
    ("nextn.shared_head_norm.weight", F32),
];

/// The types a tensor of stem `stem` may carry; `None` for a stem no Hy4-flash file holds.
#[must_use]
pub fn allowed_types(stem: &str) -> Option<&'static [GgmlType]> {
    STEM_TYPES
        .iter()
        .find(|(name, _)| *name == stem)
        .map(|&(_, types)| types)
}

#[cfg(test)]
mod tests {
    use super::{ARCH, ENGRAM_TABLE, STEM_TYPES, allowed_types};
    use crate::GgmlType::{BF16, F8E4M3, F8E4M3_ROWS, F32, Q4_K, Q6_K, Q8_0};

    /// The table holds every stem of the Q4_K_M file once, in the types that file carries them in:
    /// the engram table is the scale-less F8 type (not bloomery's scaled rows, whose `.scale` a file
    /// must carry), the routed down stack is both K-quants, the head BF16. Mutant: the table's type
    /// changed to `F8E4M3_ROWS`, or a stem listed twice.
    #[test]
    fn the_table_holds_the_q4_k_m_files_types() {
        assert_eq!(ARCH, "hy_v4_flash");
        assert_eq!(allowed_types(ENGRAM_TABLE), Some(&[F8E4M3][..]));
        assert!(!allowed_types(ENGRAM_TABLE).unwrap().contains(&F8E4M3_ROWS));
        assert_eq!(allowed_types("token_embd.weight"), Some(&[Q8_0][..]));
        assert_eq!(allowed_types("output.weight"), Some(&[BF16][..]));
        let down = allowed_types("ffn_down_exps.weight").unwrap();
        assert!(down.contains(&Q6_K) && down.contains(&Q4_K) && down.contains(&BF16));
        assert!(
            !allowed_types("ffn_gate_exps.weight")
                .unwrap()
                .contains(&Q6_K)
        );
        assert_eq!(allowed_types("ffn_gate_inp.weight"), Some(&[F32][..]));
        for (i, (stem, _)) in STEM_TYPES.iter().enumerate() {
            assert!(
                STEM_TYPES[..i].iter().all(|(s, _)| s != stem),
                "{stem} is listed twice"
            );
        }
    }

    /// A stem no Hy4-flash file holds is no stem: another family's spelling, a fused projection and a
    /// layer-numbered name have no list.
    #[test]
    fn a_stem_no_file_holds_has_no_list() {
        for stem in [
            "attn_q.weight",
            "attn_qkv.weight",
            "ffn_routed_norm.weight",
            "blk.0.attn_gate.weight",
            "engram_embd.weight",
            "",
        ] {
            assert_eq!(allowed_types(stem), None, "{stem:?}");
        }
    }
}
