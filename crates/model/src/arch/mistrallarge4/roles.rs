//! The role of every tensor in a `mistral-large4` file, by exact name. A
//! layer's stems follow its kind, as the per-layer arrays spell it: every
//! layer the attention block with its SSMax pairs, up-projections and output
//! projection, and the pre-FFN norm; a layer that writes its own cache the
//! latent's down-projection and norm besides; a full indexer layer the
//! indexer's nine stems of its own; the dense lead its gate, up and down; a
//! routed layer the router, the three routed stacks and the shared expert's
//! three. A name the table does not hold fails the file with every such name
//! listed — a vision projector or a DSpark stage asked of this reader is
//! refused here by its own tensors' names.

use gguf::Split;

use super::hparams::Hparams;
use super::names;
use crate::arch::classify::{Counts, classify_with, role_by_layer, rope_freqs_role};
use crate::arch::stems::{Stem, req};
use crate::placement::{ModelTensors, PlacementError, Role};

/// The attention block every layer carries.
const ATTN: &[Stem] = &[
    req(names::ATTN_NORM, Role::Attention),
    req(names::ATTN_Q, Role::Attention),
    req(names::ATTN_K_B, Role::Attention),
    req(names::ATTN_V_B, Role::Attention),
    req(names::ATTN_OUTPUT, Role::Attention),
    req(names::ATTN_SSMAX_W, Role::Attention),
    req(names::ATTN_SSMAX_B, Role::Attention),
];

/// The latent's down-projection and norm, a layer that writes its own cache.
const OWN_CACHE: &[Stem] = &[
    req(names::ATTN_KV_A_MQA, Role::Attention),
    req(names::ATTN_KV_A_NORM, Role::Attention),
];

/// The indexer's stems, a full indexer layer's own.
const INDEXER: &[Stem] = &[
    req(names::INDEXER_Q_NORM_W, Role::Attention),
    req(names::INDEXER_Q_NORM_B, Role::Attention),
    req(names::INDEXER_ATTN_Q_B, Role::Attention),
    req(names::INDEXER_ATTN_K, Role::Attention),
    req(names::INDEXER_K_NORM_W, Role::Attention),
    req(names::INDEXER_K_NORM_B, Role::Attention),
    req(names::INDEXER_PROJ, Role::Attention),
    req(names::INDEXER_SSMAX_W, Role::Attention),
    req(names::INDEXER_SSMAX_B, Role::Attention),
];

/// The dense lead's feed-forward block: the pre-FFN norm, the gate, up and
/// down projections.
const DENSE: &[Stem] = &[
    req(names::FFN_NORM, Role::FfnNorm),
    req(names::FFN_GATE, Role::DenseFfn),
    req(names::FFN_UP, Role::DenseFfn),
    req(names::FFN_DOWN, Role::DenseFfn),
];

/// A routed layer's feed-forward block: the pre-FFN norm, the router, the
/// three routed stacks and the shared expert's three.
const MOE: &[Stem] = &[
    req(names::FFN_NORM, Role::FfnNorm),
    req(names::FFN_GATE_INP, Role::Router),
    req(names::FFN_GATE_EXPS, Role::RoutedExperts),
    req(names::FFN_UP_EXPS, Role::RoutedExperts),
    req(names::FFN_DOWN_EXPS, Role::RoutedExperts),
    req(names::FFN_GATE_SHEXP, Role::SharedExpert),
    req(names::FFN_UP_SHEXP, Role::SharedExpert),
    req(names::FFN_DOWN_SHEXP, Role::SharedExpert),
];

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    classify_with(
        split,
        |name| {
            role_by_layer(name, hp.n_layer, |l| {
                let mut groups: Vec<&[Stem]> = vec![ATTN];
                if hp.owns_cache(l) {
                    groups.push(OWN_CACHE);
                }
                if hp.indexer_full[l] {
                    groups.push(INDEXER);
                }
                if l < hp.dense_lead {
                    groups.push(DENSE);
                } else {
                    groups.push(MOE);
                }
                groups
            })
            .or_else(|| rope_freqs_role(name))
        },
        Counts::new(hp.n_layer, hp.n_expert, hp.n_used),
    )
}

#[cfg(test)]
mod tests {
    use super::{ATTN, DENSE, INDEXER, MOE, OWN_CACHE};

    /// No stem is listed twice within a layer kind's groups. Mutant: a stem
    /// repeated in a table.
    #[test]
    fn the_tables_list_each_stem_once() {
        for (what, groups) in [
            ("attention", vec![ATTN, OWN_CACHE, INDEXER]),
            ("dense lead", vec![DENSE]),
            ("routed", vec![MOE]),
        ] {
            let mut seen: Vec<&str> = Vec::new();
            for group in groups {
                for s in group.iter() {
                    assert!(!seen.contains(&s.name), "{what}: {} twice", s.name);
                    seen.push(s.name);
                }
            }
        }
    }
}
