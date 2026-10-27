//! The role of every tensor in an M3 file, by exact name: the three model-level names, then the
//! per-layer stems by the groups a layer's pair carries ([`stems_of`]: [`ATTENTION`], the dense or
//! routed feed-forward block, and [`INDEXER`] where attention is sparse), which the header
//! reader's tensor check reads too. The indexer is attention's, on the layer's card, whichever
//! feed-forward block the layer runs. A name the pair does not hold — a fused `attn_qkv`, a
//! next-token block, an `index_*` pre-merge name, the other block's stem — fails the file with
//! every such name listed.

use gguf::Split;

use super::hparams::{Attn, FfnKind, Hparams, Kind};
use crate::arch::classify::{Counts, classify_layers};
use crate::arch::stems::{Stem, req};
use crate::placement::{ModelTensors, PlacementError, Role};

/// Every layer's attention block, the common table's.
use crate::arch::stems::ATTENTION;

/// A dense layer's feed-forward block, the common table's.
use crate::arch::stems::DENSE;

/// A routed layer's feed-forward block, the common noaux owner's.
use crate::arch::noaux::BLOCK as ROUTED;

/// A sparse-attention layer's indexer.
pub(super) const INDEXER: &[Stem] = &[
    req("indexer.q_proj.weight", Role::Attention),
    req("indexer.k_proj.weight", Role::Attention),
    req("indexer.q_norm.weight", Role::Attention),
    req("indexer.k_norm.weight", Role::Attention),
];

/// The stem groups a layer of `kind` carries: attention, its feed-forward block, and the indexer
/// where attention is sparse.
pub(super) fn stems_of(kind: Kind) -> Vec<&'static [Stem]> {
    let mut groups = vec![ATTENTION];
    match kind.ffn {
        FfnKind::Dense => groups.push(DENSE),
        FfnKind::Routed => groups.push(ROUTED),
    }
    if kind.attn == Attn::Sparse {
        groups.push(INDEXER);
    }
    groups
}

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    let e = &hp.experts;
    classify_layers(split, Counts::new(hp.n_layer, e.n_expert, e.n_used), |l| {
        stems_of(hp.kind(l))
    })
}

#[cfg(test)]
mod tests {
    use super::stems_of;
    use crate::arch::classify::{assert_one_role_per_stem, role_by_layer};
    use crate::arch::minimaxm3::hparams::{Attn, FfnKind, Kind};
    use crate::placement::Role;

    fn pair(attn: Attn, ffn: FfnKind) -> Kind {
        Kind { attn, ffn }
    }

    /// A tensor's role and layer by its name in a 60-layer file whose first three layers are full
    /// attention over a dense block and the rest sparse attention over routed experts.
    fn role(name: &str) -> Option<(Role, Option<usize>)> {
        role_by_layer(name, 60, |l| {
            stems_of(if l < 3 {
                pair(Attn::Full, FfnKind::Dense)
            } else {
                pair(Attn::Sparse, FfnKind::Routed)
            })
        })
    }

    /// A stem in two groups names one role, and no group lists a stem twice. Mutant: `ffn_norm`
    /// given the dense role in one group.
    #[test]
    fn a_stem_has_one_role() {
        let groups: Vec<_> = [
            pair(Attn::Full, FfnKind::Dense),
            pair(Attn::Full, FfnKind::Routed),
            pair(Attn::Sparse, FfnKind::Dense),
            pair(Attn::Sparse, FfnKind::Routed),
        ]
        .into_iter()
        .flat_map(stems_of)
        .collect();
        assert_one_role_per_stem(&groups);
    }

    /// The table is exact: every stem has its role on a layer of its kind, and a name the layer's
    /// kind does not hold — the other kind's stem, a fused projection, a layer past the model, a
    /// pre-merge indexer name, a next-token tensor — has none, so `classify` lists it. Mutant:
    /// every layer given the stems of both kinds.
    #[test]
    fn the_table_is_exact() {
        assert_eq!(
            role("token_embd.weight"),
            Some((Role::TokenEmbedding, None))
        );
        assert_eq!(role("output.weight"), Some((Role::Head, None)));
        assert_eq!(role("output_norm.weight"), Some((Role::Head, None)));
        for (layer, kind) in [
            (2, pair(Attn::Full, FfnKind::Dense)),
            (59, pair(Attn::Sparse, FfnKind::Routed)),
        ] {
            for group in stems_of(kind) {
                for s in group {
                    let name = format!("blk.{layer}.{}", s.name);
                    assert_eq!(role(&name), Some((s.role, Some(layer))), "{name}");
                }
            }
        }
        for name in [
            "blk.2.indexer.q_proj.weight",
            "blk.2.exp_probs_b.bias",
            "blk.3.ffn_gate.weight",
            "blk.60.attn_q.weight",
            "blk.0.attn_qkv.weight",
            "blk.3.index_q_proj.weight",
            "blk.3.indexer.v_proj.weight",
            "blk.3.nextn.eh_proj.weight",
            "blk.3.attn_q.bias",
            "rope_freqs.weight",
            "output.bias",
        ] {
            assert_eq!(role(name), None, "{name}");
        }
    }

    /// A sparse-attention dense layer carries the indexer and the dense block, and a full-attention
    /// routed layer the routed block and no indexer. Mutant: the indexer tied to the routed block.
    #[test]
    fn a_decoupled_pair_carries_its_own_stems() {
        let sparse_dense =
            |name: &str| role_by_layer(name, 2, |_| stems_of(pair(Attn::Sparse, FfnKind::Dense)));
        let full_routed =
            |name: &str| role_by_layer(name, 2, |_| stems_of(pair(Attn::Full, FfnKind::Routed)));
        assert_eq!(
            sparse_dense("blk.0.indexer.k_proj.weight"),
            Some((Role::Attention, Some(0)))
        );
        assert_eq!(
            sparse_dense("blk.0.ffn_gate.weight"),
            Some((Role::DenseFfn, Some(0)))
        );
        assert_eq!(sparse_dense("blk.0.ffn_gate_exps.weight"), None);
        assert_eq!(
            full_routed("blk.1.ffn_down_exps.weight"),
            Some((Role::RoutedExperts, Some(1)))
        );
        assert_eq!(full_routed("blk.1.indexer.q_proj.weight"), None);
        assert_eq!(full_routed("blk.1.ffn_down.weight"), None);
    }
}
