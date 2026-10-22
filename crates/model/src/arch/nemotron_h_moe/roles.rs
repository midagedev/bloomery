//! The role of every tensor in a `nemotron_h_moe` file, by exact name. A
//! Mamba-2 layer is the hybrid trunk's table and its projections' two
//! scales, an attention layer is the hybrid trunk's table, and a MoE layer is
//! this module's.
//! `attn_norm` is the mixer's norm on a Mamba-2 or attention layer and the
//! pre-FFN norm on a MoE layer. The latent pair around the experts is
//! [`Role::Router`]: like the router, a matrix of the routed block every
//! token reads on the layer's card, serial with the experts. A file without
//! a `moe_latent_size` has no latent pair: its experts act on the hidden row
//! itself.
//!
//! A next-token block (the blocks past the trunk, `nextn_predict_layers`)
//! is carried, never loaded: its own stems and the trunk's are
//! [`Role::Unused`]. A name the tables do not hold fails the file with every
//! such name listed.

use gguf::Split;

use super::hparams::{Hparams, Kind};
use super::names::{
    ATTN_NORM, EXP_PROBS_B, FFN_DOWN_EXPS, FFN_DOWN_SHEXP, FFN_GATE_INP, FFN_LATENT_DOWN,
    FFN_LATENT_UP, FFN_UP_EXPS, FFN_UP_SHEXP, NEXTN, SSM_IN_SCALE, SSM_OUT_SCALE,
};
use crate::arch::classify::{Counts, block_stem, classify_with, model_level_role};
use crate::arch::hybrid::{ATTENTION, MAMBA};
use crate::arch::stems::{Stem, opt, req};
use crate::placement::{ModelTensors, PlacementError, Role};

/// What a Mamba-2 layer carries beyond the hybrid trunk's: the per-tensor scales
/// of its two projections, which the loader does not require.
pub(super) const MAMBA_SCALES: &[Stem] = &[
    opt(SSM_IN_SCALE, Role::Attention),
    opt(SSM_OUT_SCALE, Role::Attention),
];

/// A MoE layer: the router and its bias, the routed up and down stacks and
/// the shared expert's up and down. No gate stack: the experts are not gated.
pub(super) const MOE: &[Stem] = &[
    req(ATTN_NORM, Role::FfnNorm),
    req(FFN_GATE_INP, Role::Router),
    req(EXP_PROBS_B, Role::Router),
    req(FFN_UP_EXPS, Role::RoutedExperts),
    req(FFN_DOWN_EXPS, Role::RoutedExperts),
    req(FFN_UP_SHEXP, Role::SharedExpert),
    req(FFN_DOWN_SHEXP, Role::SharedExpert),
];

/// The latent pair around the experts, where the file has a latent: like the
/// router, a matrix every token reads on the layer's card, serial with the
/// experts.
pub(super) const MOE_LATENT: &[Stem] = &[
    req(FFN_LATENT_DOWN, Role::Router),
    req(FFN_LATENT_UP, Role::Router),
];

/// The stem groups of `kind`: a file with a latent (`moe_latent_size`) has
/// the latent pair on its MoE layers, one without has none.
pub(super) fn stems_of(kind: Kind, latent: bool) -> Vec<&'static [Stem]> {
    match kind {
        Kind::Mamba => vec![MAMBA, MAMBA_SCALES],
        Kind::Attention => vec![ATTENTION],
        Kind::Moe => vec![MOE, if latent { MOE_LATENT } else { &[] }],
    }
}

/// Every stem a trunk layer of any kind carries: the union of the kinds'
/// groups, the file's latent pair included where it has one.
fn trunk_stems(latent: bool) -> Vec<&'static [Stem]> {
    let mut groups = vec![MAMBA, MAMBA_SCALES, ATTENTION, MOE];
    if latent {
        groups.push(MOE_LATENT);
    }
    groups
}

/// A tensor's role and layer by its name: the model-level three, else a
/// `blk.{l}.{stem}` of a trunk layer whose kind's groups hold the stem, else
/// a next-token block's stem (the trunk's or its own) as [`Role::Unused`].
/// `None` for any other name.
fn role(name: &str, hp: &Hparams) -> Option<(Role, Option<usize>)> {
    if let Some(model_level) = model_level_role(name) {
        return Some(model_level);
    }
    let (l, stem) = block_stem(name, hp.n_layer)?;
    let latent = hp.latent.is_some();
    if l < hp.n_trunk {
        return stems_of(hp.kinds[l], latent)
            .iter()
            .flat_map(|g| g.iter())
            .find(|s| s.name == stem)
            .map(|s| (s.role, Some(l)));
    }
    let trunk = trunk_stems(latent)
        .iter()
        .flat_map(|g| g.iter())
        .any(|s| s.name == stem);
    (trunk || NEXTN.contains(&stem)).then_some((Role::Unused, Some(l)))
}

/// Every tensor of `split` with its role and header facts. The expert
/// counts are `expert_count`, the largest layer's, and the top-k the MoE
/// layers share: a layer's own count is its description's
/// (`models::Moe::experts`).
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    let counts = Counts::new(hp.n_trunk, hp.n_expert, hp.top_k_max());
    classify_with(split, |name| role(name, hp), counts)
}

#[cfg(test)]
mod tests {
    use super::{MAMBA_SCALES, MOE, MOE_LATENT, Stem};
    use crate::arch::hybrid::{ATTENTION, MAMBA};

    /// No group lists a stem twice, and no two groups of one kind share a
    /// stem. `attn_norm` is the one stem the kinds share, and the MoE group
    /// gives it the pre-FFN role.
    #[test]
    fn a_kind_lists_each_stem_once() {
        for kind in [
            vec![MAMBA, MAMBA_SCALES],
            vec![ATTENTION],
            vec![MOE, MOE_LATENT],
        ] {
            let all: Vec<&Stem> = kind.iter().flat_map(|g| g.iter()).collect();
            for (i, s) in all.iter().enumerate() {
                assert!(
                    !all[..i].iter().any(|t| t.name == s.name),
                    "{} twice",
                    s.name
                );
            }
        }
        let norm = MOE
            .iter()
            .find(|s| s.name == super::ATTN_NORM)
            .map(|s| s.role);
        assert_eq!(norm, Some(crate::placement::Role::FfnNorm));
    }
}
