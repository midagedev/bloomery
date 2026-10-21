//! The role of every tensor in a mimo2 file, by exact name: the three
//! model-level names, then the per-layer stems — one table, by the group a
//! layer carries ([`MIXER`] … [`NEXTN`]), which the header reader's tensor
//! check reads too. The value projection writes a narrower head than the key
//! (`attn_qkv`'s row count and `attn_output`'s width say which), and a
//! sliding-window layer may carry attention sinks a full layer never does.
//! Every tensor of a next-token layer is [`Role::Unused`]: carried, never
//! loaded. A name the table does not hold fails the file with every such
//! name listed.

use gguf::Split;

use super::hparams::Hparams;
use crate::arch::classify::{Counts, classify_with, role_with_nextn};
use crate::arch::stems::{opt, req};
use crate::placement::{ModelTensors, PlacementError, Role};

pub(super) use crate::arch::stems::{Stem, required};

/// Every layer's mixer block beside the projections: the norms. The fused
/// QKV matrix (`attn_qkv`) and the split one (`attn_q` + `attn_k` +
/// `attn_v`) are alternatives, exactly one of which a layer carries
/// (`create_tensor_qkv`, src/models/mimo2.cpp:53); the sinks are the
/// sliding-window layers' alone (mimo2.cpp:57, `TENSOR_NOT_REQUIRED`).
pub(super) const MIXER: &[Stem] = &[
    req("attn_norm.weight", Role::Attention),
    req("attn_output.weight", Role::Attention),
    opt("attn_sinks.weight", Role::Attention),
];

/// The fused qkv matrix, `[embedding, q·key + kv·(key + value)]` rows.
pub(super) const QKV_FUSED: Stem = req("attn_qkv.weight", Role::Attention);

/// The split projections, in the q, k, v order llama.cpp reads them.
pub(super) const QKV_SPLIT: &[Stem] = &[
    req("attn_q.weight", Role::Attention),
    req("attn_k.weight", Role::Attention),
    req("attn_v.weight", Role::Attention),
];

/// A dense layer's feed-forward block and a routed layer's, the common tables'. The router's
/// selection bias is carried per layer (mimo2.cpp:72) and the graph keeps it out of the weights.
pub(super) use crate::arch::stems::{DENSE, MOE, SELECTION_BIAS};

/// A next-token layer's own stems beside the trunk's: the input projection of
/// the two norms, the two norms, and the head's own norm and matrices some
/// conversions write instead of sharing the model's (mimo2.cpp:74-81, every
/// one of these four `TENSOR_NOT_REQUIRED` but the first three).
pub(super) const NEXTN: &[Stem] = &[
    req("nextn.eh_proj.weight", Role::Unused),
    req("nextn.enorm.weight", Role::Unused),
    req("nextn.hnorm.weight", Role::Unused),
    opt("nextn.embed_tokens.weight", Role::Unused),
    opt("nextn.shared_head_head.weight", Role::Unused),
    opt("nextn.shared_head_norm.weight", Role::Unused),
    opt("layer_output_norm.weight", Role::Unused),
];

/// Every stem a trunk layer may carry, by group.
const TRUNK: [&[Stem]; 5] = [MIXER, DENSE, MOE, QKV_SPLIT, &[QKV_FUSED]];

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    classify_with(
        split,
        |name| role_with_nextn(name, hp.n_layer, hp.n_trunk, &TRUNK, NEXTN),
        Counts::new(hp.n_trunk, hp.n_expert, hp.n_used),
    )
}

#[cfg(test)]
mod tests {
    use super::{NEXTN, TRUNK};
    use crate::arch::classify::{assert_one_role_per_stem, role_in};

    /// A stem in two groups names one role, and no group lists a stem twice; the fused qkv matrix
    /// and each split projection are the only stems their counterpart's check refuses by name, and
    /// a next-token layer's own stem is no trunk stem.
    #[test]
    fn a_stem_has_one_role() {
        assert_one_role_per_stem(&TRUNK);
        for (i, s) in NEXTN.iter().enumerate() {
            assert!(
                !NEXTN[..i].iter().any(|t| t.name == s.name),
                "{} twice",
                s.name
            );
            assert_eq!(
                role_in(&TRUNK, &format!("blk.0.{}", s.name), 1),
                None,
                "{}",
                s.name
            );
        }
    }
}
