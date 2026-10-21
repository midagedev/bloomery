//! The role of every tensor in a Raon-1 file, by exact name: the three model-level names, then
//! the per-layer stems by the groups a layer's kind carries ([`stems_of`]: [`RDA`] or [`MLA`],
//! [`DENSE`] or [`ROUTED`], and [`HC`]), which the header reader's tensor check reads too. A name
//! the layer's kind does not hold — a fused `attn_qkv`, a block past the model, a next-token
//! tensor, the other kind's stem — fails the file with every such name listed.

use gguf::Split;

use super::hparams::{Hparams, Kind};
use crate::arch::classify::{Counts, classify_layers};
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
const RDA: &[Stem] = &join::<16>(&[kda::HEAD, GATES, kda::TAIL]);

/// A latent layer's mixer: the query and key-value latents with their norms, the absorbed
/// per-head up projections, the output gate and the output projection.
const MLA: &[Stem] = &[
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
use crate::arch::stems::{DENSE, HC};

/// A routed layer's feed-forward block, the common noaux owner's.
use crate::arch::noaux::BLOCK as ROUTED;

/// Every kind of layer a file may hold: a mixer, and a dense (`true`) or routed feed-forward block.
pub(super) const KINDS: [(Kind, bool); 4] = [
    (Kind::Rda, true),
    (Kind::Rda, false),
    (Kind::Mla, true),
    (Kind::Mla, false),
];

/// The stem groups a layer of `kind` carries.
pub(super) fn stems_of((mixer, dense): (Kind, bool)) -> Vec<&'static [Stem]> {
    let mixer = match mixer {
        Kind::Rda => RDA,
        Kind::Mla => MLA,
    };
    vec![mixer, if dense { DENSE } else { ROUTED }, HC]
}

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    let e = &hp.experts;
    classify_layers(split, Counts::new(hp.n_layer, e.n_expert, e.n_used), |l| {
        stems_of((hp.kind(l), hp.dense(l)))
    })
}

#[cfg(test)]
mod tests {
    use super::{KINDS, MLA, RDA, stems_of};
    use crate::arch::classify::{assert_one_role_per_stem, role_by_layer};
    use crate::arch::raon1::hparams::Kind;
    use crate::placement::Role;

    /// A tensor's role and layer by its name in a 25-layer file whose layers 0 and 1 are dense RDA
    /// layers, 2 to 23 routed RDA layers and 24 a routed latent one.
    fn role(name: &str) -> Option<(Role, Option<usize>)> {
        role_by_layer(name, 25, |l| {
            stems_of((if l == 24 { Kind::Mla } else { Kind::Rda }, l < 2))
        })
    }

    /// A stem in two groups names one role (the two mixers both hold `attn_norm` and
    /// `attn_output`, the two feed-forward blocks `ffn_norm`), and no group lists a stem twice.
    /// Mutant: `attn_norm` given the router's role in one group.
    #[test]
    fn a_stem_has_one_role() {
        let groups: Vec<_> = KINDS.into_iter().flat_map(stems_of).collect();
        assert_one_role_per_stem(&groups);
    }

    /// The table is exact: a stem of each group has its role on a layer of its kind, and a name
    /// the layer's kind does not hold — the other kind's stem, a fused projection, a layer past
    /// the model, a next-token tensor, a bias, a KDA low-rank gate, an indexer — has none, so
    /// `classify` lists it. Mutant: every layer given the stems of every kind.
    #[test]
    fn the_table_is_exact() {
        assert_eq!(
            role("token_embd.weight"),
            Some((Role::TokenEmbedding, None))
        );
        assert_eq!(role("output.weight"), Some((Role::Head, None)));
        assert_eq!(
            role("blk.24.attn_gate.weight"),
            Some((Role::Attention, Some(24)))
        );
        assert_eq!(
            role("blk.1.ssm_rda.weight"),
            Some((Role::Attention, Some(1)))
        );
        assert_eq!(
            role("blk.2.hc_ffn_scale.weight"),
            Some((Role::HyperConnection, Some(2)))
        );
        assert_eq!(
            role("blk.24.exp_probs_b.bias"),
            Some((Role::Router, Some(24)))
        );
        assert_eq!(
            role("blk.1.ffn_gate.weight"),
            Some((Role::DenseFfn, Some(1)))
        );
        assert_eq!((RDA.len(), MLA.len()), (16, 10));
        for name in [
            "blk.24.ssm_rda.weight",
            "blk.3.attn_gate.weight",
            "blk.1.exp_probs_b.bias",
            "blk.2.ffn_gate.weight",
            "blk.25.attn_q.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ssm_g_a.weight",
            "blk.3.indexer.proj.weight",
            "blk.3.nextn.eh_proj.weight",
            "blk.3.attn_q.bias",
            "rope_freqs.weight",
        ] {
            assert_eq!(role(name), None, "{name}");
        }
    }
}
