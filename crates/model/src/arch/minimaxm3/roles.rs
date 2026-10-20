//! The role of every tensor in an M3 file, by exact name: the three model-level names, then the
//! per-layer stems by the group a layer carries ([`ATTENTION`] … [`INDEXER`]), which the header
//! reader's tensor check reads too. The indexer is attention's, on the layer's card. A name the
//! table does not hold — a fused `attn_qkv`, a next-token block, an `index_*` pre-merge name —
//! fails the file with every such name listed.

use gguf::Split;

use super::hparams::Hparams;
use crate::arch::classify::{Counts, block_stem, classify_with, model_level_role};
use crate::arch::stems::{Stem, req};
use crate::placement::{ModelTensors, PlacementError, Role};

pub(super) use crate::arch::stems::required;

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

/// A dense layer's feed-forward block.
pub(super) const DENSE: &[Stem] = &[
    req("ffn_norm.weight", Role::FfnNorm),
    req("ffn_gate.weight", Role::DenseFfn),
    req("ffn_up.weight", Role::DenseFfn),
    req("ffn_down.weight", Role::DenseFfn),
];

/// A routed layer's feed-forward block: the router with its selection bias, the routed stacks and
/// the shared expert.
pub(super) const ROUTED: &[Stem] = &[
    req("ffn_norm.weight", Role::FfnNorm),
    req("ffn_gate_inp.weight", Role::Router),
    req("exp_probs_b.bias", Role::Router),
    req("ffn_gate_exps.weight", Role::RoutedExperts),
    req("ffn_up_exps.weight", Role::RoutedExperts),
    req("ffn_down_exps.weight", Role::RoutedExperts),
    req("ffn_gate_shexp.weight", Role::SharedExpert),
    req("ffn_up_shexp.weight", Role::SharedExpert),
    req("ffn_down_shexp.weight", Role::SharedExpert),
];

/// A sparse layer's indexer.
pub(super) const INDEXER: &[Stem] = &[
    req("indexer.q_proj.weight", Role::Attention),
    req("indexer.k_proj.weight", Role::Attention),
    req("indexer.q_norm.weight", Role::Attention),
    req("indexer.k_norm.weight", Role::Attention),
];

/// Every stem a layer may carry, by group.
const GROUPS: [&[Stem]; 4] = [ATTENTION, DENSE, ROUTED, INDEXER];

/// The role of a layer's stem.
fn layer_role(stem: &str) -> Option<Role> {
    GROUPS
        .iter()
        .flat_map(|g| g.iter())
        .find(|s| s.name == stem)
        .map(|s| s.role)
}

/// A tensor's role and layer by its name; `None` when the table does not hold it or its layer is
/// past the file's.
fn role(name: &str, n_layer: usize) -> Option<(Role, Option<usize>)> {
    if let Some(model_level) = model_level_role(name) {
        return Some(model_level);
    }
    let (layer, stem) = block_stem(name, n_layer)?;
    layer_role(stem).map(|r| (r, Some(layer)))
}

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    classify_with(
        split,
        |name| role(name, hp.n_layer),
        Counts {
            layers: hp.n_layer,
            experts: hp.experts.n_expert as u64,
            experts_used: hp.experts.n_used as u64,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::{GROUPS, layer_role, role};
    use crate::placement::Role;

    /// A stem in two groups names one role, and no group lists a stem twice. Mutant: `ffn_norm`
    /// given the dense role in one group.
    #[test]
    fn a_stem_has_one_role() {
        for (g, group) in GROUPS.iter().enumerate() {
            for (i, s) in group.iter().enumerate() {
                assert_eq!(layer_role(s.name), Some(s.role), "{}", s.name);
                assert!(
                    !group[..i].iter().any(|t| t.name == s.name),
                    "{} twice",
                    s.name
                );
                for other in &GROUPS[g + 1..] {
                    for t in *other {
                        assert!(
                            t.name != s.name || t.role == s.role,
                            "{} in two roles",
                            s.name
                        );
                    }
                }
            }
        }
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
