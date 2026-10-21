//! The role of every tensor in an M3 file, by exact name: the three model-level names, then the
//! per-layer stems by the groups a layer's kind carries ([`stems_of`]: [`ATTENTION`] …
//! [`INDEXER`]), which the header reader's tensor check reads too. The indexer is attention's, on
//! the layer's card. A name the layer's kind does not hold — a fused `attn_qkv`, a next-token
//! block, an `index_*` pre-merge name, the other kind's stem — fails the file with every such
//! name listed.

use gguf::Split;

use super::hparams::{Hparams, Kind};
use crate::arch::classify::{Counts, classify_layers};
use crate::arch::stems::{Stem, req};
use crate::placement::{ModelTensors, PlacementError, Role};

/// Every layer's attention block, the common table's.
use crate::arch::stems::ATTENTION;

/// A dense layer's feed-forward block, the common table's.
use crate::arch::stems::DENSE;

/// A routed layer's feed-forward block, the common noaux owner's.
use crate::arch::noaux::BLOCK as ROUTED;

/// A sparse layer's indexer.
const INDEXER: &[Stem] = &[
    req("indexer.q_proj.weight", Role::Attention),
    req("indexer.k_proj.weight", Role::Attention),
    req("indexer.q_norm.weight", Role::Attention),
    req("indexer.k_norm.weight", Role::Attention),
];

/// The stem groups a layer of `kind` carries: a dense layer's feed-forward block, a sparse
/// layer's routed one and its indexer.
pub(super) fn stems_of(kind: Kind) -> Vec<&'static [Stem]> {
    match kind {
        Kind::Dense => vec![ATTENTION, DENSE],
        Kind::Sparse => vec![ATTENTION, ROUTED, INDEXER],
    }
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
    use crate::arch::minimaxm3::hparams::Kind;
    use crate::placement::Role;

    /// A tensor's role and layer by its name in a 60-layer file whose first three layers are dense.
    fn role(name: &str) -> Option<(Role, Option<usize>)> {
        role_by_layer(name, 60, |l| {
            stems_of(if l < 3 { Kind::Dense } else { Kind::Sparse })
        })
    }

    /// A stem in two groups names one role, and no group lists a stem twice. Mutant: `ffn_norm`
    /// given the dense role in one group.
    #[test]
    fn a_stem_has_one_role() {
        let groups: Vec<_> = [Kind::Dense, Kind::Sparse]
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
        for (layer, kind) in [(2, Kind::Dense), (59, Kind::Sparse)] {
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
}
