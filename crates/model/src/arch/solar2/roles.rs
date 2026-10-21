//! The role of every tensor in a Solar-Open2 file, by exact name: the three model-level names,
//! then the per-layer stems by the group a layer carries ([`GQA`] or the common KDA mixer's, and
//! the common routed block), which the header reader's tensor check reads too. A name the table
//! does not hold — a fused `attn_qkv`, a block past the model, a next-token tensor — fails the
//! file with every such name listed.

use gguf::Split;

use super::hparams::Hparams;
use crate::arch::classify::{Counts, classify_groups};
use crate::arch::stems::{Stem, req};
use crate::arch::{hybrid, kda, noaux};
use crate::placement::{ModelTensors, PlacementError, Role};

/// A GQA layer's mixer: the common no-rope attention's stems and the output gate projection.
pub(super) const GQA: &[&[Stem]] = &[hybrid::ATTENTION, GATE];

/// The GQA layer's output gate projection.
pub(super) const GATE: &[Stem] = &[req("attn_gate.weight", Role::Attention)];

/// A KDA layer's mixer: the common [`kda`] module's table.
pub(super) const KDA: &[Stem] = kda::STEMS;

/// A routed layer's feed-forward block: the common noaux owner's.
pub(super) const ROUTED: &[Stem] = noaux::BLOCK;

/// Every stem a layer may carry, by group.
const GROUPS: [&[Stem]; 4] = [hybrid::ATTENTION, GATE, kda::STEMS, noaux::BLOCK];

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

    /// A stem in two groups names one role (the KDA mixer and the attention both hold `attn_q`),
    /// and no group lists a stem twice. Mutant: `attn_norm` given the router's role in one group.
    #[test]
    fn a_stem_has_one_role() {
        assert_one_role_per_stem(&GROUPS);
    }

    /// The table is exact: the gate projection and a KDA stem have their roles on their layer,
    /// and a name the table does not hold — a fused projection, a layer past the model, a
    /// next-token tensor, a bias — has none, so `classify` lists it.
    #[test]
    fn the_table_is_exact() {
        assert_eq!(
            role("token_embd.weight", 48),
            Some((Role::TokenEmbedding, None))
        );
        assert_eq!(role("output.weight", 48), Some((Role::Head, None)));
        assert_eq!(
            role("blk.44.attn_gate.weight", 48),
            Some((Role::Attention, Some(44)))
        );
        assert_eq!(
            role("blk.1.ssm_f_b.weight", 48),
            Some((Role::Attention, Some(1)))
        );
        assert_eq!(
            role("blk.47.exp_probs_b.bias", 48),
            Some((Role::Router, Some(47)))
        );
        for name in [
            "blk.48.attn_q.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.attn_q_norm.weight",
            "blk.3.nextn.eh_proj.weight",
            "blk.3.attn_q.bias",
            "blk.3.ffn_gate.weight",
            "rope_freqs.weight",
        ] {
            assert_eq!(role(name, 48), None, "{name}");
        }
    }
}
