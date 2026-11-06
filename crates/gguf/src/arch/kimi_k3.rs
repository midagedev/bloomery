//! Kimi K3's and K3.1's header facts: the two architecture words and the quant types each tensor
//! stem may carry.
//!
//! One table holds the three files the engine opens, so a stem's set is the union of what each
//! holds it in, cut to the types a kernel of the engine reads for that stem's site:
//! - the released UD-IQ1_S set (unsloth/Kimi-K3-GGUF, read from each shard's own tensor table:
//!   shard 1 holds no tensor, the 2,573 tensors are in shards 2 to 14): Q6_K trunk and
//!   embeddings, Q8_0 for the two stems whose width is no whole super-block, routed stacks IQ1_S
//!   on every layer and IQ2_XXS on the gate and up stacks of ten layers and the down stack of two;
//! - the engine's own quantization of the checkpoint: Q4_K trunk and token embedding, Q6_K head,
//!   the routed stacks in those two types, the K3.1 indexer's tensors left in BF16;
//! - the checkpoint as `prepare` writes it: BF16 projections, MXFP4 routed stacks, F32 norms,
//!   scores and vectors.
//!
//! Each list is cut against `placement::kernels` and the card site tables by a test of the model
//! crate (`arch::kimi_k3::hparams`); a type with no kernel for its stem is not listed, and a stem
//! the three files hold in a type only the checkpoint's own form has (the per-head absorb's
//! BF16 halves, which no per-head kernel reads) says so beside its list. A type outside a stem's
//! list is a file this table was not written for, and the reader refuses it by name.

use crate::GgmlType;

/// `general.architecture` of a Kimi K3 file.
pub const ARCH: &str = "kimi-k3";

/// `general.architecture` of a Kimi K3.1 file: K3's keys and tensors, and on every latent layer
/// the lightning indexer of DeepSeek's sparse attention.
pub const ARCH_K31: &str = "kimi-k31";

/// The F32 stems: norms, scores, the router, the delta layers' vectors and conv taps.
const F32: &[GgmlType] = &[GgmlType::F32];
/// A projection the dense gemv sites run: the released set's Q6_K, the quantization's Q4_K (with
/// Q5_K and Q8_0 for a tensor it keeps wider) and the checkpoint's BF16.
const TRUNK: &[GgmlType] = &[
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q8_0,
    GgmlType::BF16,
];
/// A projection whose first dim is the delta head's 128 (`ssm_f_b`), no whole super-block of a
/// K-quant: Q8_0 and the checkpoint's BF16.
const NARROW: &[GgmlType] = &[GgmlType::Q8_0, GgmlType::BF16];
/// `attn_v_b`, the per-head value projection: the kernels read it in Q6_K or Q8_0
/// (`gated_mla::ValueHeads`); BF16 is the checkpoint's form, which no per-head kernel reads.
const VALUE_HEADS: &[GgmlType] = &[GgmlType::Q6_K, GgmlType::Q8_0, GgmlType::BF16];
/// `attn_k_b`, the per-head key absorb: the kernel reads Q8_0 planes alone
/// (`latent_gated::enqueue_q8_0_gemv_heads`); BF16 is the checkpoint's form, which no per-head
/// kernel reads.
const KEY_HEADS: &[GgmlType] = &[GgmlType::Q8_0, GgmlType::BF16];
/// The token embedding: the card's row lookups (`Card::embed`) in the types the three files hold.
const EMBED: &[GgmlType] = &[GgmlType::Q4_K, GgmlType::Q6_K, GgmlType::BF16];
/// The head (`Card::head`) in the types the three files hold.
const HEAD: &[GgmlType] = &[GgmlType::Q6_K, GgmlType::Q8_0, GgmlType::BF16];
/// The routed stacks: the released set's two I-quants and the checkpoint's MXFP4 (OCP blocks of
/// 32 with an E8M0 scale).
const ROUTED: &[GgmlType] = &[GgmlType::IQ1_S, GgmlType::IQ2_XXS, GgmlType::MXFP4];
/// The K3.1 indexer's matrices: BF16 as the checkpoint stores them and the quantization leaves
/// them, Q8_0 as `prepare` writes them under `BLOOMERY_PREPARE_Q8`.
const INDEXER: &[GgmlType] = &[GgmlType::BF16, GgmlType::Q8_0];

/// Every tensor stem of a Kimi K3 or K3.1 file (the name past `blk.N.` for a layer's tensor, the
/// whole name for a model-level one) with the types it may carry.
pub const STEM_TYPES: &[(&str, &[GgmlType])] = &[
    ("token_embd.weight", EMBED),
    ("output.weight", HEAD),
    ("output_norm.weight", F32),
    ("output_res_score.weight", F32),
    ("attn_norm.weight", F32),
    ("attn_res_score.weight", F32),
    ("ffn_norm.weight", F32),
    ("ffn_res_score.weight", F32),
    // A delta layer's mixer.
    ("attn_q.weight", TRUNK),
    ("attn_k.weight", TRUNK),
    ("attn_v.weight", TRUNK),
    ("ssm_conv1d_q.weight", F32),
    ("ssm_conv1d_k.weight", F32),
    ("ssm_conv1d_v.weight", F32),
    ("ssm_f_a.weight", TRUNK),
    ("ssm_f_b.weight", NARROW),
    ("ssm_g.weight", TRUNK),
    ("ssm_beta.weight", F32),
    ("ssm_a", F32),
    ("ssm_dt.bias", F32),
    ("ssm_norm.weight", F32),
    // A latent layer's mixer; `attn_output` is both kinds'.
    ("attn_q_a.weight", TRUNK),
    ("attn_q_a_norm.weight", F32),
    ("attn_q_b.weight", TRUNK),
    ("attn_kv_a_mqa.weight", TRUNK),
    ("attn_kv_a_norm.weight", F32),
    ("attn_k_b.weight", KEY_HEADS),
    ("attn_v_b.weight", VALUE_HEADS),
    ("attn_gate.weight", TRUNK),
    ("attn_output.weight", TRUNK),
    // A K3.1 latent layer's lightning indexer: the query projection from the query latent, the key
    // projection from the layer input with its LayerNorm (gain and bias), and the head weights.
    ("indexer.attn_q_b.weight", INDEXER),
    ("indexer.attn_k.weight", INDEXER),
    ("indexer.proj.weight", INDEXER),
    ("indexer.k_norm.weight", F32),
    ("indexer.k_norm.bias", F32),
    // The dense layer's feed-forward block.
    ("ffn_gate.weight", TRUNK),
    ("ffn_up.weight", TRUNK),
    ("ffn_down.weight", TRUNK),
    // A routed layer's block: the router, the routed stacks, the latent pair and its norm, the
    // shared expert.
    ("ffn_gate_inp.weight", F32),
    ("exp_probs_b.bias", F32),
    ("ffn_gate_exps.weight", ROUTED),
    ("ffn_up_exps.weight", ROUTED),
    ("ffn_down_exps.weight", ROUTED),
    ("ffn_routed_down.weight", TRUNK),
    ("ffn_routed_up.weight", TRUNK),
    ("ffn_routed_norm.weight", F32),
    ("ffn_gate_shexp.weight", TRUNK),
    ("ffn_up_shexp.weight", TRUNK),
    ("ffn_down_shexp.weight", TRUNK),
];

/// The types a tensor of stem `stem` may carry; `None` for a stem no Kimi K3 file holds.
#[must_use]
pub fn allowed_types(stem: &str) -> Option<&'static [GgmlType]> {
    STEM_TYPES
        .iter()
        .find(|(name, _)| *name == stem)
        .map(|&(_, types)| types)
}

/// Whether `word` is the architecture word of a Kimi K3 file or a Kimi K3.1 one.
#[must_use]
pub fn is_arch(word: &str) -> bool {
    word == ARCH || word == ARCH_K31
}

#[cfg(test)]
mod tests {
    use super::{ARCH, ARCH_K31, STEM_TYPES, allowed_types, is_arch};
    use crate::GgmlType::{self, BF16, F32, IQ1_S, IQ2_XXS, MXFP4, Q4_K, Q5_K, Q6_K, Q8_0};

    /// A stem's type in each of the three files the table accepts, `None` where the file has no
    /// such tensor: the released UD-IQ1_S set (read from its shards), the engine's quantization
    /// (the recipe of the K3.1 quantization round) and the checkpoint as `prepare` writes it.
    /// The routed stacks are IQ1_S in the released set and the quantization (IQ2_XXS on the layers
    /// its recipe promotes, which the union holds beside it), MXFP4 in the checkpoint.
    type Profiles = (&'static str, [Option<GgmlType>; 3]);

    const PROFILES: &[Profiles] = &[
        ("token_embd.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        ("output.weight", [Some(Q6_K), Some(Q6_K), Some(BF16)]),
        ("attn_q.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        ("attn_k.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        ("attn_v.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        ("attn_output.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        ("attn_gate.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        ("attn_q_a.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        ("attn_q_b.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        ("attn_kv_a_mqa.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        ("attn_k_b.weight", [Some(Q8_0), Some(Q8_0), Some(BF16)]),
        ("attn_v_b.weight", [Some(Q6_K), Some(Q6_K), Some(BF16)]),
        ("ssm_f_a.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        ("ssm_f_b.weight", [Some(Q8_0), Some(Q8_0), Some(BF16)]),
        ("ssm_g.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        ("ffn_gate.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        ("ffn_up.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        ("ffn_down.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        (
            "ffn_gate_shexp.weight",
            [Some(Q6_K), Some(Q4_K), Some(BF16)],
        ),
        ("ffn_up_shexp.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        (
            "ffn_down_shexp.weight",
            [Some(Q6_K), Some(Q4_K), Some(BF16)],
        ),
        (
            "ffn_routed_down.weight",
            [Some(Q6_K), Some(Q4_K), Some(BF16)],
        ),
        ("ffn_routed_up.weight", [Some(Q6_K), Some(Q4_K), Some(BF16)]),
        (
            "ffn_gate_exps.weight",
            [Some(IQ1_S), Some(IQ1_S), Some(MXFP4)],
        ),
        (
            "ffn_up_exps.weight",
            [Some(IQ1_S), Some(IQ1_S), Some(MXFP4)],
        ),
        (
            "ffn_down_exps.weight",
            [Some(IQ1_S), Some(IQ1_S), Some(MXFP4)],
        ),
        ("indexer.attn_q_b.weight", [None, Some(BF16), Some(BF16)]),
        ("indexer.attn_k.weight", [None, Some(BF16), Some(BF16)]),
        ("indexer.proj.weight", [None, Some(BF16), Some(BF16)]),
    ];

    /// The stems every file holds in F32.
    const F32_STEMS: &[&str] = &[
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
        "indexer.k_norm.weight",
        "indexer.k_norm.bias",
    ];

    /// One table accepts the three files: every stem of each profile is listed in the type that
    /// profile holds it in, the IQ2_XXS promotion of the released set and the quantization's
    /// routed stacks included, and a stem no profile holds is no stem. Mutant: a profile's type
    /// cut from its stem's list (the BF16 the checkpoint holds, the Q4_K the quantization does,
    /// the MXFP4 stacks).
    #[test]
    fn one_table_accepts_the_three_files() {
        assert_eq!((ARCH, ARCH_K31), ("kimi-k3", "kimi-k31"));
        assert!(is_arch(ARCH) && is_arch(ARCH_K31) && !is_arch("kimi-linear"));
        for (stem, profile) in PROFILES {
            let allowed = allowed_types(stem).unwrap_or_else(|| panic!("{stem} is no stem"));
            for ty in profile.iter().flatten() {
                assert!(allowed.contains(ty), "{stem}: {ty} is not in {allowed:?}");
            }
        }
        for stem in F32_STEMS {
            assert_eq!(allowed_types(stem), Some(&[F32][..]), "{stem}");
        }
        for stem in [
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ] {
            assert!(allowed_types(stem).unwrap().contains(&IQ2_XXS), "{stem}");
        }
        let listed = |stem: &str| {
            usize::from(PROFILES.iter().any(|(s, _)| *s == stem))
                + usize::from(F32_STEMS.contains(&stem))
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
            "ssm_g_a.weight",
            "attn_qkv.weight",
            "blk.0.attn_q.weight",
            "indexer.q_norm.weight",
            "",
        ] {
            assert_eq!(allowed_types(stem), None, "{stem}");
        }
    }

    /// Every listed type has the kernel of its stem's site in the card table (`GgmlType::card`):
    /// the dense gemv sites for a projection, the row lookup for the token embedding, the head's
    /// launch for `output.weight` and the card leg's gate-up and down launches for a routed stack.
    /// The two per-head absorbs are no `Card` entry: their Q8_0 and Q6_K kernels are
    /// `latent_gated`'s, and the BF16 their lists hold is the checkpoint's own form (the card's
    /// per-head load refuses it by name). Mutant: a type no table entry runs added to a stem
    /// (F16 on the embedding, IQ1_S on the head).
    #[test]
    fn every_listed_type_has_its_sites_kernel() {
        let per_head = ["attn_k_b.weight", "attn_v_b.weight"];
        for (stem, types) in STEM_TYPES {
            if types == &[F32] {
                continue;
            }
            for ty in *types {
                let card = ty.card();
                let (what, runs) = match *stem {
                    "token_embd.weight" => ("embed", card.embed.is_some()),
                    "output.weight" => ("head", card.head.is_some()),
                    "ffn_gate_exps.weight" | "ffn_up_exps.weight" => {
                        ("routed gate-up", card.gate_up.is_some())
                    }
                    "ffn_down_exps.weight" => ("routed down", card.down.is_some()),
                    s if per_head.contains(&s) => {
                        assert!(
                            matches!(*ty, Q6_K | Q8_0 | BF16),
                            "{stem}: {ty} has no per-head kernel"
                        );
                        continue;
                    }
                    _ => ("dense", card.dense.is_some()),
                };
                assert!(runs, "{stem}: {ty} has no {what} entry in the card table");
            }
        }
    }

    /// A type no kernel reads for its stem is not listed: a K-quant on a stem whose first dim is
    /// no whole super-block (`ssm_f_b`, `attn_k_b`, 128 values), the per-head absorbs in anything
    /// but Q8_0 (key) and Q6_K or Q8_0 (value) beside the checkpoint's BF16, F32 and F16 on a
    /// projection, a K-quant beside MXFP4 on the routed stacks, and any non-F32 norm. Mutant: one
    /// of these types added to a list.
    #[test]
    fn a_type_no_kernel_reads_is_not_listed() {
        let has = |stem: &str, ty: GgmlType| allowed_types(stem).unwrap().contains(&ty);
        for ty in [Q4_K, Q5_K, Q6_K] {
            assert!(
                !has("ssm_f_b.weight", ty) && !has("attn_k_b.weight", ty),
                "{ty}"
            );
        }
        assert!(!has("attn_v_b.weight", Q4_K) && !has("attn_v_b.weight", Q5_K));
        for stem in [
            "attn_q.weight",
            "ffn_routed_up.weight",
            "indexer.proj.weight",
        ] {
            assert!(!has(stem, F32) && !has(stem, GgmlType::F16), "{stem}");
        }
        for stem in ["ffn_gate_exps.weight", "ffn_down_exps.weight"] {
            assert!(
                !has(stem, Q4_K) && !has(stem, Q6_K) && !has(stem, Q8_0),
                "{stem}"
            );
        }
        assert!(!has("token_embd.weight", Q5_K) && !has("output.weight", Q4_K));
        assert!(!has("attn_norm.weight", BF16) && !has("indexer.k_norm.bias", BF16));
    }
}
