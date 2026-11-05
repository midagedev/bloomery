//! Kimi K3's header facts: the architecture string and the quant type each tensor stem may carry.
//!
//! The types are the union over every shard of the released UD-IQ1_S set (unsloth/Kimi-K3-GGUF),
//! read from each shard's own tensor table: shard 1 holds no tensor, the 2,573 tensors are in
//! shards 2 to 14. The routed stacks are the one stem family that varies by layer (IQ1_S on every
//! layer, IQ2_XXS on the gate and up stacks of ten layers and the down stack of two, by the
//! quant's own recipe); every other stem has one type on every layer. A type outside a stem's
//! list is a file this set's reader was not written for, and its reader refuses it by name.

use crate::GgmlType;

/// `general.architecture` of a Kimi K3 file.
pub const ARCH: &str = "kimi-k3";

/// The types of a stem that one type holds on every layer.
const F32: &[GgmlType] = &[GgmlType::F32];
const Q6_K: &[GgmlType] = &[GgmlType::Q6_K];
const Q8_0: &[GgmlType] = &[GgmlType::Q8_0];
/// The routed stacks' two types.
const ROUTED: &[GgmlType] = &[GgmlType::IQ1_S, GgmlType::IQ2_XXS];

/// Every tensor stem of a Kimi K3 file (the name past `blk.N.` for a layer's tensor, the whole name
/// for a model-level one) with the types it may carry.
pub const STEM_TYPES: &[(&str, &[GgmlType])] = &[
    ("token_embd.weight", Q6_K),
    ("output.weight", Q6_K),
    ("output_norm.weight", F32),
    ("output_res_score.weight", F32),
    ("attn_norm.weight", F32),
    ("attn_res_score.weight", F32),
    ("ffn_norm.weight", F32),
    ("ffn_res_score.weight", F32),
    // A delta layer's mixer.
    ("attn_q.weight", Q6_K),
    ("attn_k.weight", Q6_K),
    ("attn_v.weight", Q6_K),
    ("ssm_conv1d_q.weight", F32),
    ("ssm_conv1d_k.weight", F32),
    ("ssm_conv1d_v.weight", F32),
    ("ssm_f_a.weight", Q6_K),
    ("ssm_f_b.weight", Q8_0),
    ("ssm_g.weight", Q6_K),
    ("ssm_beta.weight", F32),
    ("ssm_a", F32),
    ("ssm_dt.bias", F32),
    ("ssm_norm.weight", F32),
    // A latent layer's mixer; `attn_output` is both kinds'.
    ("attn_q_a.weight", Q6_K),
    ("attn_q_a_norm.weight", F32),
    ("attn_q_b.weight", Q6_K),
    ("attn_kv_a_mqa.weight", Q6_K),
    ("attn_kv_a_norm.weight", F32),
    ("attn_k_b.weight", Q8_0),
    ("attn_v_b.weight", Q6_K),
    ("attn_gate.weight", Q6_K),
    ("attn_output.weight", Q6_K),
    // The dense layer's feed-forward block.
    ("ffn_gate.weight", Q6_K),
    ("ffn_up.weight", Q6_K),
    ("ffn_down.weight", Q6_K),
    // A routed layer's block: the router, the routed stacks, the latent pair and its norm, the
    // shared expert.
    ("ffn_gate_inp.weight", F32),
    ("exp_probs_b.bias", F32),
    ("ffn_gate_exps.weight", ROUTED),
    ("ffn_up_exps.weight", ROUTED),
    ("ffn_down_exps.weight", ROUTED),
    ("ffn_routed_down.weight", Q6_K),
    ("ffn_routed_up.weight", Q6_K),
    ("ffn_routed_norm.weight", F32),
    ("ffn_gate_shexp.weight", Q6_K),
    ("ffn_up_shexp.weight", Q6_K),
    ("ffn_down_shexp.weight", Q6_K),
];

/// The types a tensor of stem `stem` may carry; `None` for a stem no Kimi K3 file holds.
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
    use crate::GgmlType;

    /// The union of the 14 shards: every stem has the types the released set's tensor tables
    /// hold, the routed stacks two, the model-level three and the expert-less stems one, and a
    /// stem the file does not hold has none. Mutant: a stem's list widened to another type, or
    /// the routed stacks cut to IQ1_S alone.
    #[test]
    fn the_table_is_the_released_sets_union() {
        assert_eq!(ARCH, "kimi-k3");
        let routed = [GgmlType::IQ1_S, GgmlType::IQ2_XXS];
        for stem in [
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ] {
            assert_eq!(allowed_types(stem), Some(&routed[..]), "{stem}");
        }
        let q6k = [
            "token_embd.weight",
            "output.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_gate.weight",
            "attn_q_a.weight",
            "attn_q_b.weight",
            "attn_kv_a_mqa.weight",
            "attn_v_b.weight",
            "ssm_f_a.weight",
            "ssm_g.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
            "ffn_routed_down.weight",
            "ffn_routed_up.weight",
        ];
        for stem in q6k {
            assert_eq!(allowed_types(stem), Some(&[GgmlType::Q6_K][..]), "{stem}");
        }
        for stem in ["ssm_f_b.weight", "attn_k_b.weight"] {
            assert_eq!(allowed_types(stem), Some(&[GgmlType::Q8_0][..]), "{stem}");
        }
        let f32s = [
            "output_norm.weight",
            "output_res_score.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
            "attn_res_score.weight",
            "ffn_res_score.weight",
            "attn_q_a_norm.weight",
            "attn_kv_a_norm.weight",
            "ssm_conv1d_q.weight",
            "ssm_conv1d_k.weight",
            "ssm_conv1d_v.weight",
            "ssm_beta.weight",
            "ssm_a",
            "ssm_dt.bias",
            "ssm_norm.weight",
            "ffn_gate_inp.weight",
            "exp_probs_b.bias",
            "ffn_routed_norm.weight",
        ];
        for stem in f32s {
            assert_eq!(allowed_types(stem), Some(&[GgmlType::F32][..]), "{stem}");
        }
        assert_eq!(STEM_TYPES.len(), 3 + q6k.len() + 2 + f32s.len());
        for stem in [
            "ssm_g_a.weight",
            "attn_qkv.weight",
            "blk.0.attn_q.weight",
            "",
        ] {
            assert_eq!(allowed_types(stem), None, "{stem}");
        }
    }
}
