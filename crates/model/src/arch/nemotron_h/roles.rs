//! The role of every tensor in a `nemotron_h` file, by exact name. The
//! per-layer stems are one table per layer kind. `attn_norm` is the mixer's
//! norm on a Mamba-2 or attention layer and the pre-FFN norm on an MLP
//! layer. A name the tables do not hold fails the file with every such name
//! listed.

use gguf::Split;

use super::hparams::{Hparams, Kind};
use super::names::{
    ATTN_K, ATTN_NORM, ATTN_OUTPUT, ATTN_Q, ATTN_V, FFN_DOWN, FFN_UP, SSM_A, SSM_CONV,
    SSM_CONV_BIAS, SSM_D, SSM_DT_BIAS, SSM_IN, SSM_NORM, SSM_OUT,
};
use crate::arch::classify::{Counts, classify_with};
use crate::arch::stems::{Stem, opt, req};
use crate::placement::{ModelTensors, PlacementError, Role};

/// A Mamba-2 layer. The conv bias is the one stem the loader does not require.
pub(super) const MAMBA: &[Stem] = &[
    req(ATTN_NORM, Role::Attention),
    req(SSM_A, Role::Attention),
    req(SSM_D, Role::Attention),
    req(SSM_DT_BIAS, Role::Attention),
    req(SSM_CONV, Role::Attention),
    opt(SSM_CONV_BIAS, Role::Attention),
    req(SSM_IN, Role::Attention),
    req(SSM_NORM, Role::Attention),
    req(SSM_OUT, Role::Attention),
];

/// An attention layer. No rope table is among the stems.
pub(super) const ATTENTION: &[Stem] = &[
    req(ATTN_NORM, Role::Attention),
    req(ATTN_Q, Role::Attention),
    req(ATTN_K, Role::Attention),
    req(ATTN_V, Role::Attention),
    req(ATTN_OUTPUT, Role::Attention),
];

/// A dense MLP layer. `attn_norm` is this block's pre-FFN norm.
pub(super) const MLP: &[Stem] = &[
    req(ATTN_NORM, Role::FfnNorm),
    req(FFN_UP, Role::DenseFfn),
    req(FFN_DOWN, Role::DenseFfn),
];

/// The stems of `kind`.
pub(super) fn stems_of(kind: Kind) -> &'static [Stem] {
    match kind {
        Kind::Mamba => MAMBA,
        Kind::Attention => ATTENTION,
        Kind::Mlp => MLP,
    }
}

/// A tensor's role and layer by its name; `None` when the table of its
/// layer's kind does not hold it.
fn role(name: &str, hp: &Hparams) -> Option<(Role, Option<usize>)> {
    match name {
        "token_embd.weight" => return Some((Role::TokenEmbedding, None)),
        "output_norm.weight" | "output.weight" => return Some((Role::Head, None)),
        _ => {}
    }
    let (layer, stem) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer: usize = layer.parse().ok().filter(|&l| l < hp.n_layer)?;
    stems_of(hp.kinds[layer])
        .iter()
        .find(|s| s.name == stem)
        .map(|s| (s.role, Some(layer)))
}

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    classify_with(
        split,
        |name| role(name, hp),
        Counts {
            layers: hp.n_layer,
            experts: 0,
            experts_used: 0,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::{ATTENTION, MAMBA, MLP, Stem};

    /// No group lists a stem twice. `attn_norm` is the one stem two groups
    /// share, and the MLP group gives it the pre-FFN role.
    #[test]
    fn a_group_lists_each_stem_once() {
        for g in [MAMBA, ATTENTION, MLP] {
            for (i, s) in g.iter().enumerate() {
                assert!(
                    !g[..i].iter().any(|t| t.name == s.name),
                    "{} twice",
                    s.name
                );
            }
        }
        let norm = |g: &[Stem]| g.iter().find(|s| s.name == super::ATTN_NORM).map(|s| s.role);
        assert_eq!(norm(MAMBA), Some(crate::placement::Role::Attention));
        assert_eq!(norm(ATTENTION), Some(crate::placement::Role::Attention));
        assert_eq!(norm(MLP), Some(crate::placement::Role::FfnNorm));
    }
}
