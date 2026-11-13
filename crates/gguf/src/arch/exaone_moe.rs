//! K-EXAONE-2.0's header facts: the architecture word and the quant types each tensor stem may
//! carry.
//!
//! One table holds the four files the engine opens of this model:
//! - the engine's own quantization of the checkpoint (the bootstrap set of the quant path: no
//!   importance matrix, K-quants only): the routed stacks Q4_K, the trunk — the attention, the
//!   dense lead's and the MTP block's feed-forward projections, the shared experts — Q8_0, the
//!   head Q8_0, F32 norms, router and correction bias;
//! - the box's second quant (what the bootstrap file exists to make, over this engine's own
//!   importance matrix): the routed stacks' gate and up IQ2_XXS, their downs Q4_K;
//! - the one-Spark file: every routed stack IQ1_S, the smallest set the Spark's budget reads;
//! - the checkpoint as `prepare` writes it: the routed stacks in the checkpoint's FP8_B128
//!   blocks, the trunk Q8_0 (an FP8 projection decoded exactly and quantized again, which is how
//!   the map reads it), BF16 for the head, the embedding and the MTP block as stored, F32 norms,
//!   router and correction bias.
//!
//! The routed stacks' FP8_B128 is the checkpoint's own form: the card loads it as planes, and
//! the common card gate·up and down entries do not run it — a program that runs it names it in
//! its own dispatch, as qwen3moe's does. A type outside a stem's list is a file this table was
//! not written for, and the reader refuses it by name.

use crate::GgmlType;

/// `general.architecture` of a K-EXAONE-2.0 file.
pub const ARCH: &str = "exaone-moe";

/// The F32 stems: the norms, the router and its correction bias.
const F32: &[GgmlType] = &[GgmlType::F32];
/// A trunk projection: the engine's Q8_0 (an FP8 one decoded exactly and quantized again) and
/// the checkpoint's BF16 where `prepare` keeps the stored form — the MTP block's and the head's.
const TRUNK: &[GgmlType] = &[GgmlType::Q8_0, GgmlType::BF16];
/// The shared experts sit on the routed blocks alone: the map decodes their FP8 weights and
/// quantizes them again, so no file holds them in another type.
const TRUNK_Q8: &[GgmlType] = &[GgmlType::Q8_0];
/// The token embedding: the card's row lookups in the two files' types.
const EMBED: &[GgmlType] = &[GgmlType::Q8_0, GgmlType::BF16];
/// The head: the bootstrap set's Q8_0 and the checkpoint's BF16.
const HEAD: &[GgmlType] = &[GgmlType::Q8_0, GgmlType::BF16];
/// The routed stacks' gate and up: the bootstrap set's Q4_K, the second quant's IQ2_XXS, the
/// Spark file's IQ1_S and the checkpoint's FP8_B128.
const ROUTED_IN: &[GgmlType] = &[
    GgmlType::Q4_K,
    GgmlType::IQ2_XXS,
    GgmlType::IQ1_S,
    GgmlType::FP8_B128,
];
/// The routed stacks' downs: the same list less the second quant's IQ2_XXS, whose down is Q4_K.
const ROUTED_DOWN: &[GgmlType] = &[GgmlType::Q4_K, GgmlType::IQ1_S, GgmlType::FP8_B128];

/// Every tensor stem of a K-EXAONE-2.0 file (the name past `blk.N.` for a block's tensor, the
/// whole name for a model-level one) with the types it may carry.
pub const STEM_TYPES: &[(&str, &[GgmlType])] = &[
    ("token_embd.weight", EMBED),
    ("output.weight", HEAD),
    ("output_norm.weight", F32),
    ("attn_norm.weight", F32),
    ("ffn_norm.weight", F32),
    ("attn_q_norm.weight", F32),
    ("attn_k_norm.weight", F32),
    ("attn_q.weight", TRUNK),
    ("attn_k.weight", TRUNK),
    ("attn_v.weight", TRUNK),
    ("attn_output.weight", TRUNK),
    // The dense lead's and the MTP block's feed-forward block.
    ("ffn_gate.weight", TRUNK),
    ("ffn_up.weight", TRUNK),
    ("ffn_down.weight", TRUNK),
    // A routed block's: the router, its bias, the routed stacks and the shared expert.
    ("ffn_gate_inp.weight", F32),
    ("exp_probs_b.bias", F32),
    ("ffn_gate_exps.weight", ROUTED_IN),
    ("ffn_up_exps.weight", ROUTED_IN),
    ("ffn_down_exps.weight", ROUTED_DOWN),
    ("ffn_gate_shexp.weight", TRUNK_Q8),
    ("ffn_up_shexp.weight", TRUNK_Q8),
    ("ffn_down_shexp.weight", TRUNK_Q8),
    // The MTP block's input projection and its three norms.
    ("nextn.eh_proj.weight", TRUNK),
    ("nextn.enorm.weight", F32),
    ("nextn.hnorm.weight", F32),
    ("nextn.shared_head_norm.weight", F32),
];

/// The types a tensor of stem `stem` may carry; `None` for a stem no K-EXAONE-2.0 file holds.
#[must_use]
pub fn allowed_types(stem: &str) -> Option<&'static [GgmlType]> {
    STEM_TYPES
        .iter()
        .find(|(name, _)| *name == stem)
        .map(|&(_, types)| types)
}

/// Whether `word` is the architecture word of a K-EXAONE-2.0 file.
#[must_use]
pub fn is_arch(word: &str) -> bool {
    word == ARCH
}

#[cfg(test)]
mod tests {
    use super::{ARCH, STEM_TYPES, allowed_types, is_arch};
    use crate::GgmlType::{self, BF16, F32, FP8_B128, IQ1_S, IQ2_XXS, Q4_K, Q8_0};

    /// A stem's type in each of the four files the table accepts, `None` where the file has no
    /// such tensor: the bootstrap set, the box's second quant, the one-Spark file and the
    /// checkpoint as `prepare` writes it.
    type Profiles = (&'static str, [Option<GgmlType>; 4]);

    const PROFILES: &[Profiles] = &[
        (
            "token_embd.weight",
            [Some(Q8_0), Some(Q8_0), Some(Q8_0), Some(BF16)],
        ),
        (
            "output.weight",
            [Some(Q8_0), Some(Q8_0), Some(Q8_0), Some(BF16)],
        ),
        (
            "attn_q.weight",
            [Some(Q8_0), Some(Q8_0), Some(Q8_0), Some(Q8_0)],
        ),
        (
            "attn_k.weight",
            [Some(Q8_0), Some(Q8_0), Some(Q8_0), Some(Q8_0)],
        ),
        (
            "attn_v.weight",
            [Some(Q8_0), Some(Q8_0), Some(Q8_0), Some(BF16)],
        ),
        (
            "attn_output.weight",
            [Some(Q8_0), Some(Q8_0), Some(Q8_0), Some(Q8_0)],
        ),
        (
            "ffn_gate_shexp.weight",
            [Some(Q8_0), Some(Q8_0), Some(Q8_0), Some(Q8_0)],
        ),
        (
            "ffn_up_shexp.weight",
            [Some(Q8_0), Some(Q8_0), Some(Q8_0), Some(Q8_0)],
        ),
        (
            "ffn_down_shexp.weight",
            [Some(Q8_0), Some(Q8_0), Some(Q8_0), Some(Q8_0)],
        ),
        (
            "ffn_gate.weight",
            [Some(Q8_0), Some(Q8_0), Some(Q8_0), Some(BF16)],
        ),
        (
            "ffn_up.weight",
            [Some(Q8_0), Some(Q8_0), Some(Q8_0), Some(BF16)],
        ),
        (
            "ffn_down.weight",
            [Some(Q8_0), Some(Q8_0), Some(Q8_0), Some(BF16)],
        ),
        (
            "nextn.eh_proj.weight",
            [Some(Q8_0), Some(Q8_0), Some(Q8_0), Some(BF16)],
        ),
        (
            "ffn_gate_exps.weight",
            [Some(Q4_K), Some(IQ2_XXS), Some(IQ1_S), Some(FP8_B128)],
        ),
        (
            "ffn_up_exps.weight",
            [Some(Q4_K), Some(IQ2_XXS), Some(IQ1_S), Some(FP8_B128)],
        ),
        (
            "ffn_down_exps.weight",
            [Some(Q4_K), Some(Q4_K), Some(IQ1_S), Some(FP8_B128)],
        ),
    ];

    /// The stems every file holds in F32.
    const F32_STEMS: &[&str] = &[
        "output_norm.weight",
        "attn_norm.weight",
        "ffn_norm.weight",
        "attn_q_norm.weight",
        "attn_k_norm.weight",
        "ffn_gate_inp.weight",
        "exp_probs_b.bias",
        "nextn.enorm.weight",
        "nextn.hnorm.weight",
        "nextn.shared_head_norm.weight",
    ];

    /// One table accepts the four files: every stem of each profile is listed in the type that
    /// profile holds it in, and a stem no profile holds is no stem. Mutant: a profile's type cut
    /// from its stem's list (the Q8_0 the bootstrap holds, the IQ2_XXS the second quant does).
    #[test]
    fn one_table_accepts_the_four_files() {
        assert_eq!(ARCH, "exaone-moe");
        assert!(is_arch(ARCH) && !is_arch("exaone"));
        for (stem, profile) in PROFILES {
            let allowed = allowed_types(stem).unwrap_or_else(|| panic!("{stem} is no stem"));
            for ty in profile.iter().flatten() {
                assert!(allowed.contains(ty), "{stem}: {ty} is not in {allowed:?}");
            }
        }
        for stem in F32_STEMS {
            assert_eq!(allowed_types(stem), Some(&[F32][..]), "{stem}");
        }
        let listed = |stem: &str| {
            PROFILES.iter().any(|(s, _)| *s == stem) as usize + F32_STEMS.contains(&stem) as usize
        };
        for (stem, _) in STEM_TYPES {
            assert_eq!(
                listed(stem),
                1,
                "{stem} is in exactly one of the test's lists"
            );
        }
        assert_eq!(STEM_TYPES.len(), PROFILES.len() + F32_STEMS.len());
        for stem in [
            "attn_qkv.weight",
            "ffn_routed_up.weight",
            "blk.0.attn_q.weight",
            "nextn.embed_tokens.weight",
            "indexer.attn_k.weight",
            "",
        ] {
            assert_eq!(allowed_types(stem), None, "{stem}");
        }
    }

    /// Every listed type has the kernel of its stem's site in the card table (`GgmlType::card`):
    /// the dense gemv sites for a projection, the row lookup for the token embedding, the head's
    /// launch for `output.weight`, the common card gate·up and down entries for a Q4_K stack and
    /// its planes load for an FP8_B128 one, which those entries do not run. Mutant: a type no
    /// table entry runs added to a stem (FP8_B128 on an attention projection).
    #[test]
    fn every_listed_type_has_its_sites_kernel() {
        for (stem, types) in STEM_TYPES {
            if types == &[F32] {
                continue;
            }
            for ty in *types {
                let card = ty.card();
                let (what, runs) = match *stem {
                    "token_embd.weight" => ("embed", card.embed.is_some()),
                    "output.weight" => ("head", card.head.is_some()),
                    "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight"
                        if *ty == FP8_B128 =>
                    {
                        ("routed planes", card.routed.is_some())
                    }
                    "ffn_gate_exps.weight" | "ffn_up_exps.weight" => {
                        ("routed gate-up", card.gate_up.is_some())
                    }
                    "ffn_down_exps.weight" => ("routed down", card.down.is_some()),
                    _ => ("dense", card.dense.is_some()),
                };
                assert!(runs, "{stem}: {ty} has no {what} entry in the card table");
            }
        }
    }

    /// A type no file holds at a stem is not listed: the checkpoint's FP8_B128 on a trunk
    /// projection (the map quantizes it), a K-quant beside it on the norms and the router, BF16
    /// on the shared experts, an I-quant beside the four sets' own on the routed stacks, and the
    /// second quant's gate-and-up type on the downs, whose own is Q4_K. Mutant: one of these
    /// types added to a list.
    #[test]
    fn a_type_no_kernel_reads_is_not_listed() {
        let has = |stem: &str, ty: GgmlType| allowed_types(stem).unwrap().contains(&ty);
        for stem in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate_shexp.weight",
        ] {
            assert!(!has(stem, FP8_B128), "{stem}");
        }
        for stem in [
            "attn_norm.weight",
            "ffn_norm.weight",
            "ffn_gate_inp.weight",
            "exp_probs_b.bias",
            "nextn.enorm.weight",
        ] {
            assert!(!has(stem, BF16) && !has(stem, Q8_0), "{stem}");
        }
        for stem in [
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ] {
            assert!(!has(stem, GgmlType::IQ3_XXS) && !has(stem, Q8_0), "{stem}");
        }
        assert!(
            !has("ffn_down_exps.weight", IQ2_XXS),
            "the second quant's downs are Q4_K"
        );
        assert!(!has("token_embd.weight", Q4_K) && !has("ffn_up_shexp.weight", BF16));
    }
}
