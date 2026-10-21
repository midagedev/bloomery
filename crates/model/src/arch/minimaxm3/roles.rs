//! The role of every tensor in an M3 file, by exact name: the three model-level names, then the
//! per-layer stems by the group a layer carries ([`ATTENTION`] … [`INDEXER`]), which the header
//! reader's tensor check reads too. The indexer is attention's, on the layer's card. A name the
//! table does not hold — a fused `attn_qkv`, a next-token block, an `index_*` pre-merge name —
//! fails the file with every such name listed.

use gguf::Split;

use super::hparams::Hparams;
use crate::arch::classify::{Counts, classify_groups};
use crate::arch::stems::{Stem, req};
use crate::placement::{ModelTensors, PlacementError, Role};

/// Every layer's attention block: the norms, the split projections and the output projection.
pub(super) const ATTENTION: &[Stem] = &[
    req("attn_norm.weight", Role::Attention),
    req("attn_q.weight", Role::Attention),
    req("attn_k.weight", Role::Attention),
    req("attn_v.weight", Role::Attention),
    req("attn_q_norm.weight", Role::Attention),
    req("attn_k_norm.weight", Role::Attention),
    req("attn_output.weight", Role::Attention),
];

/// A dense layer's feed-forward block, the common table's.
pub(super) use crate::arch::stems::DENSE;

/// A routed layer's feed-forward block, the common noaux owner's.
pub(super) use crate::arch::noaux::BLOCK as ROUTED;

/// A sparse layer's indexer.
pub(super) const INDEXER: &[Stem] = &[
    req("indexer.q_proj.weight", Role::Attention),
    req("indexer.k_proj.weight", Role::Attention),
    req("indexer.q_norm.weight", Role::Attention),
    req("indexer.k_norm.weight", Role::Attention),
];

/// Every stem a layer may carry, by group.
const GROUPS: [&[Stem]; 4] = [ATTENTION, DENSE, ROUTED, INDEXER];

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
    use super::GROUPS;
    use crate::arch::classify::{assert_one_role_per_stem, role_in};
    use crate::placement::Role;

    /// A tensor's role and layer by its name over the table's groups.
    fn role(name: &str, n_layer: usize) -> Option<(Role, Option<usize>)> {
        role_in(&GROUPS, name, n_layer)
    }

    /// A stem in two groups names one role, and no group lists a stem twice. Mutant: `ffn_norm`
    /// given the dense role in one group.
    #[test]
    fn a_stem_has_one_role() {
        assert_one_role_per_stem(&GROUPS);
    }

    /// The table is exact: every stem has its role on its layer, and a name the table does not
    /// hold — a fused projection, a layer past the model, a pre-merge indexer name, a next-token
    /// tensor — has none, so `classify` lists it.
    #[test]
    fn the_table_is_exact() {
        assert_eq!(
            role("token_embd.weight", 60),
            Some((Role::TokenEmbedding, None))
        );
        assert_eq!(role("output.weight", 60), Some((Role::Head, None)));
        assert_eq!(role("output_norm.weight", 60), Some((Role::Head, None)));
        for group in GROUPS {
            for s in group {
                let name = format!("blk.59.{}", s.name);
                assert_eq!(role(&name, 60), Some((s.role, Some(59))), "{name}");
            }
        }
        for name in [
            "blk.60.attn_q.weight",
            "blk.0.attn_qkv.weight",
            "blk.3.index_q_proj.weight",
            "blk.3.indexer.v_proj.weight",
            "blk.3.nextn.eh_proj.weight",
            "blk.3.attn_q.bias",
            "rope_freqs.weight",
            "output.bias",
        ] {
            assert_eq!(role(name, 60), None, "{name}");
        }
    }
}
