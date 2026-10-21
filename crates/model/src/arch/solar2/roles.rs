//! The role of every tensor in a Solar-Open2 file, by exact name: the three model-level names,
//! then the per-layer stems by the groups a layer's kind carries ([`stems_of`]: the GQA mixer or
//! the common KDA mixer's, and the common routed block), which the header reader's tensor check
//! reads too. A name the layer's kind does not hold — a fused `attn_qkv`, a block past the model,
//! a next-token tensor, the other kind's stem — fails the file with every such name listed.

use gguf::Split;

use super::hparams::{Hparams, Kind};
use crate::arch::classify::{Counts, classify_layers};
use crate::arch::stems::{Stem, req};
use crate::arch::{hybrid, kda, noaux};
use crate::placement::{ModelTensors, PlacementError, Role};

/// The GQA layer's output gate projection.
const GATE: &[Stem] = &[req("attn_gate.weight", Role::Attention)];

/// The stem groups a layer of `kind` carries: a KDA layer's mixer is the common [`kda`] module's
/// table, a GQA layer's the common no-rope attention's stems and the output gate projection, and
/// the feed-forward block is the common noaux owner's.
pub(super) fn stems_of(kind: Kind) -> Vec<&'static [Stem]> {
    match kind {
        Kind::Kda => vec![kda::STEMS, noaux::BLOCK],
        Kind::Gqa => vec![hybrid::ATTENTION, GATE, noaux::BLOCK],
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
    use crate::arch::solar2::hparams::Kind;
    use crate::placement::Role;

    /// A tensor's role and layer by its name in a 48-layer file whose every fourth layer, from the
    /// first, is a GQA layer.
    fn role(name: &str) -> Option<(Role, Option<usize>)> {
        role_by_layer(name, 48, |l| {
            stems_of(if l % 4 == 0 { Kind::Gqa } else { Kind::Kda })
        })
    }

    /// A stem in two groups names one role (the KDA mixer and the attention both hold `attn_q`),
    /// and no group lists a stem twice. Mutant: `attn_norm` given the router's role in one group.
    #[test]
    fn a_stem_has_one_role() {
        let groups: Vec<_> = [Kind::Kda, Kind::Gqa]
            .into_iter()
            .flat_map(stems_of)
            .collect();
        assert_one_role_per_stem(&groups);
    }

    /// The table is exact: the gate projection and a KDA stem have their roles on a layer of
    /// their kind, and a name the layer's kind does not hold — the other kind's stem, a fused
    /// projection, a layer past the model, a next-token tensor, a bias — has none, so `classify`
    /// lists it. Mutant: every layer given the stems of both kinds.
    #[test]
    fn the_table_is_exact() {
        assert_eq!(
            role("token_embd.weight"),
            Some((Role::TokenEmbedding, None))
        );
        assert_eq!(role("output.weight"), Some((Role::Head, None)));
        assert_eq!(
            role("blk.44.attn_gate.weight"),
            Some((Role::Attention, Some(44)))
        );
        assert_eq!(
            role("blk.1.ssm_f_b.weight"),
            Some((Role::Attention, Some(1)))
        );
        assert_eq!(
            role("blk.47.exp_probs_b.bias"),
            Some((Role::Router, Some(47)))
        );
        for name in [
            "blk.1.attn_gate.weight",
            "blk.44.ssm_f_b.weight",
            "blk.48.attn_q.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.attn_q_norm.weight",
            "blk.3.nextn.eh_proj.weight",
            "blk.3.attn_q.bias",
            "blk.3.ffn_gate.weight",
            "rope_freqs.weight",
        ] {
            assert_eq!(role(name), None, "{name}");
        }
    }
}
