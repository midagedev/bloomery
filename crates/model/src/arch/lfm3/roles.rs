//! The role of every tensor in an `lfm3` file, by exact name. A layer's mixer tensors are one
//! table per layer kind, and every layer then carries the sandwich norms around its two
//! sublayers and a dense SwiGLU block. `attn_norm` is the mixer's pre-norm on both kinds. The
//! head is the token embedding unless the file carries `output.weight`, and the rope's factors
//! are a model-level tensor the load reads once. A name the tables do not hold fails the file
//! with every such name listed.

use gguf::Split;

use super::hparams::{Hparams, Kind};
use super::names::{
    ATTN_GATE, ATTN_NORM, CONV_IN, CONV_OUT, CONV_TAPS, POST_ATTN_NORM, POST_FFN_NORM, blk,
    output_norm, token_embd,
};
use crate::arch::classify::{Counts, classify_with, role_by_layer, rope_freqs_role};
use crate::arch::refuse_tensor_set;
use crate::arch::stems::{self, Stem, join, req, required};
use crate::placement::{ModelTensors, PlacementError, Role};

/// A short-conv layer's mixer: the pre-norm, the input projection, the filters and the output
/// projection.
pub(super) const CONV: &[Stem] = &[
    req(ATTN_NORM, Role::Attention),
    req(CONV_IN, Role::Attention),
    req(CONV_TAPS, Role::Attention),
    req(CONV_OUT, Role::Attention),
];

/// The per-head output gate's projection.
const GATE: &[Stem] = &[req(ATTN_GATE, Role::Attention)];

/// An attention layer's mixer: the shared block with per-head query and key norms, and the gate.
pub(super) const ATTENTION: &[Stem] = &join::<8>(&[stems::ATTENTION, GATE]);

/// The norm on the mixer's output.
const POST_MIXER: &[Stem] = &[req(POST_ATTN_NORM, Role::Attention)];

/// The norm on the feed-forward block's output.
const POST_BLOCK: &[Stem] = &[req(POST_FFN_NORM, Role::FfnNorm)];

/// What every layer carries after its mixer: the mixer's post-norm, the feed-forward block with
/// its pre-norm, and the block's post-norm.
pub(super) const BLOCK: &[Stem] = &join::<6>(&[POST_MIXER, stems::DENSE, POST_BLOCK]);

/// The stem groups of a layer of `kind`.
pub(super) fn stems_of(kind: Kind) -> [&'static [Stem]; 2] {
    match kind {
        Kind::Conv => [CONV, BLOCK],
        Kind::Sliding | Kind::Full => [ATTENTION, BLOCK],
    }
}

/// Every layer carries the required stems of its kind, and the model-level norm and embedding
/// are in the file: the first missing name, with how many more there are. A tensor of another
/// kind's table on a layer is no tensor of this one's: [`classify`] refuses it as unclaimed.
pub(super) fn require(split: &Split, kinds: &[Kind]) -> Result<(), PlacementError> {
    let has = |name: &str| split.find(name).is_some();
    let mut missing: Vec<String> = [token_embd(), output_norm()]
        .into_iter()
        .filter(|name| !has(name))
        .collect();
    for (l, &kind) in kinds.iter().enumerate() {
        missing.extend(
            stems_of(kind)
                .iter()
                .flat_map(|group| required(group))
                .map(|stem| blk(l, stem))
                .filter(|name| !has(name)),
        );
    }
    refuse_tensor_set(&missing, &[], "an lfm3 trunk needs", "")
}

/// Every tensor of `split` with its role and header facts, the file's own names only: a name no
/// table claims fails the file. The rope's factors are read once by the load, no step's operand
/// ([`Role::Unread`]).
pub fn classify(split: &Split, hp: &Hparams) -> Result<ModelTensors, PlacementError> {
    classify_with(
        split,
        |name| {
            rope_freqs_role(name)
                .or_else(|| role_by_layer(name, hp.n_layer, |l| stems_of(hp.kinds[l])))
        },
        Counts::new(hp.n_layer, 0, 0),
    )
}

#[cfg(test)]
mod tests {
    use super::{ATTENTION, BLOCK, CONV, Kind, stems_of};
    use crate::arch::classify::assert_one_role_per_stem;
    use crate::placement::Role;

    /// Each kind's groups give every stem one role, no group lists a stem twice, and the groups'
    /// sizes are the file's: a short-conv layer's four mixer tensors and six more, an attention
    /// layer's eight and six more. Mutant: a stem listed in two groups, or `join`'s count off.
    #[test]
    fn each_kind_holds_its_own_stems() {
        for kind in [Kind::Conv, Kind::Sliding, Kind::Full] {
            assert_one_role_per_stem(&stems_of(kind));
        }
        assert_eq!((CONV.len(), ATTENTION.len(), BLOCK.len()), (4, 8, 6));
        assert_eq!(
            stems_of(Kind::Sliding)[0].len(),
            stems_of(Kind::Full)[0].len()
        );
    }

    /// The sandwich norms sit with the layer: the mixer's post-norm with the mixer's tensors, the
    /// block's with the feed-forward norm. A role of `Head` or `Unread` on either leaves the layer's
    /// card without a tensor the layer's step reads.
    #[test]
    fn the_post_norms_are_the_layers() {
        let role = |name: &str| BLOCK.iter().find(|s| s.name == name).map(|s| s.role);
        assert_eq!(role("post_attention_norm.weight"), Some(Role::Attention));
        assert_eq!(role("post_ffw_norm.weight"), Some(Role::FfnNorm));
        assert_eq!(role("ffn_norm.weight"), Some(Role::FfnNorm));
        assert_eq!(role("ffn_down.weight"), Some(Role::DenseFfn));
    }
}
