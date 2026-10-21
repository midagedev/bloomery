//! The role of every tensor in a `nemotron_h` file, by exact name. The
//! per-layer stems are one table per layer kind. `attn_norm` is the mixer's
//! norm on a Mamba-2 or attention layer and the pre-FFN norm on an MLP
//! layer. A name the tables do not hold fails the file with every such name
//! listed.

use gguf::Split;

use super::hparams::{Hparams, Kind};
use super::names::{ATTN_NORM, FFN_DOWN, FFN_UP};
use crate::arch::classify::{Counts, classify_layers};
use crate::arch::hybrid::{ATTENTION, MAMBA};
use crate::arch::stems::{Stem, req};
use crate::placement::{ModelTensors, PlacementError, Role};

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

/// Every tensor of `split` with its role and header facts.
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    classify_layers(split, Counts::new(hp.n_layer, 0, 0), |l| {
        [stems_of(hp.kinds[l])]
    })
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
                assert!(!g[..i].iter().any(|t| t.name == s.name), "{} twice", s.name);
            }
        }
        let norm = |g: &[Stem]| {
            g.iter()
                .find(|s| s.name == super::ATTN_NORM)
                .map(|s| s.role)
        };
        assert_eq!(norm(MAMBA), Some(crate::placement::Role::Attention));
        assert_eq!(norm(ATTENTION), Some(crate::placement::Role::Attention));
        assert_eq!(norm(MLP), Some(crate::placement::Role::FfnNorm));
    }
}
