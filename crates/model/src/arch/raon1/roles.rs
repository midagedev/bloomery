//! The role of every tensor in a Raon-1 file, by exact name: the three model-level names, then
//! the per-layer stems by the group a layer carries ([`RDA`] or [`MLA`], [`DENSE`] or
//! [`ROUTED`], and [`HC`]), which the header reader's tensor check reads too. A name the table
//! does not hold — a fused `attn_qkv`, a block past the model, a next-token tensor — fails the
//! file with every such name listed.

use gguf::Split;

use super::hparams::Hparams;
use crate::arch::classify::{Counts, classify_groups};
use crate::arch::kda;
use crate::arch::stems::{Stem, join, req};
use crate::placement::{ModelTensors, PlacementError, Role};

/// The two gates an RDA layer adds to the delta rule's common mixer: the output gate's
/// projection and the write gate's offset.
const GATES: &[Stem] = &[
    req("ssm_g.weight", Role::Attention),
    req("ssm_rda.weight", Role::Attention),
];

/// An RDA layer's mixer: the common [`kda`] head and tail around the two gates.
pub(super) const RDA: &[Stem] = &join::<16>(&[kda::HEAD, GATES, kda::TAIL]);

/// A latent layer's mixer: the query and key-value latents with their norms, the absorbed
/// per-head up projections, the output gate and the output projection.
pub(super) const MLA: &[Stem] = &[
    req("attn_norm.weight", Role::Attention),
    req("attn_q_a.weight", Role::Attention),
    req("attn_q_a_norm.weight", Role::Attention),
    req("attn_q_b.weight", Role::Attention),
    req("attn_kv_a_mqa.weight", Role::Attention),
    req("attn_kv_a_norm.weight", Role::Attention),
    req("attn_k_b.weight", Role::Attention),
    req("attn_v_b.weight", Role::Attention),
    req("attn_gate.weight", Role::Attention),
    req("attn_output.weight", Role::Attention),
];

/// A dense layer's feed-forward block and every layer's hyper-connection tensors, the common
/// table's.
pub(super) use crate::arch::stems::{DENSE, HC};

/// A routed layer's feed-forward block, the common noaux owner's.
pub(super) use crate::arch::noaux::BLOCK as ROUTED;

/// Every stem a layer may carry, by group.
const GROUPS: [&[Stem]; 5] = [RDA, MLA, DENSE, ROUTED, HC];

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    let e = &hp.experts;
    classify_groups(
        split,
        &GROUPS,
        Counts::new(hp.n_layer, e.n_expert, e.n_used),
    )
}

#[cfg(test)]
mod tests {
    use super::{GROUPS, MLA, RDA};
    use crate::arch::classify::{assert_one_role_per_stem, role_in};
    use crate::placement::Role;

    /// A tensor's role and layer by its name over the table's groups.
    fn role(name: &str, n_layer: usize) -> Option<(Role, Option<usize>)> {
        role_in(&GROUPS, name, n_layer)
    }

    /// A stem in two groups names one role (the two mixers both hold `attn_norm` and
    /// `attn_output`, the two feed-forward blocks `ffn_norm`), and no group lists a stem twice.
    /// Mutant: `attn_norm` given the router's role in one group.
    #[test]
    fn a_stem_has_one_role() {
        assert_one_role_per_stem(&GROUPS);
    }

    /// The table is exact: a stem of each group has its role on its layer, and a name the table
    /// does not hold — a fused projection, a layer past the model, a next-token tensor, a bias, a
    /// KDA low-rank gate, an indexer — has none, so `classify` lists it.
    #[test]
    fn the_table_is_exact() {
        assert_eq!(
            role("token_embd.weight", 25),
            Some((Role::TokenEmbedding, None))
        );
        assert_eq!(role("output.weight", 25), Some((Role::Head, None)));
        assert_eq!(
            role("blk.24.attn_gate.weight", 25),
            Some((Role::Attention, Some(24)))
        );
        assert_eq!(
            role("blk.1.ssm_rda.weight", 25),
            Some((Role::Attention, Some(1)))
        );
        assert_eq!(
            role("blk.2.hc_ffn_scale.weight", 25),
            Some((Role::HyperConnection, Some(2)))
        );
        assert_eq!(
            role("blk.24.exp_probs_b.bias", 25),
            Some((Role::Router, Some(24)))
        );
        assert_eq!((RDA.len(), MLA.len()), (16, 10));
        for name in [
            "blk.25.attn_q.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ssm_g_a.weight",
            "blk.3.indexer.proj.weight",
            "blk.3.nextn.eh_proj.weight",
            "blk.3.attn_q.bias",
            "rope_freqs.weight",
        ] {
            assert_eq!(role(name, 25), None, "{name}");
        }
    }
}
