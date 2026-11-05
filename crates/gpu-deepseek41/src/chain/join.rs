//! The slots of a V4.1 rank's sums over the ranks (`bloomery_gpu::host::rank::Sums`): the
//! outputs a rank computes as its part of a sum over the ranks, which every rank adds in rank
//! order, so every rank holds the same bits.
//!
//! Three outputs of a step are such parts: the attention sub-layer's after its output projection
//! (the rank's heads and output groups), the MoE sub-layer's after its combine (the rank's part of
//! every expert's inner width) and an engram site's `engram_wkv` over the rank's columns of the
//! site's rows. The trunk's two share [`TRUNK`], each a post and a sum in turn. Each engram site
//! has a slot of its own ([`site`]): its part depends on the token alone, so it is posted early and
//! summed where the site's layer reads it, while the trunk's slot posts and sums between. One row a
//! step.

use bloomery_gpu::GpuError;
use bloomery_gpu::host::GatherShape;
use model::arch::deepseek41::hparams::Hparams;

use crate::hc::HC_STREAMS;

/// The trunk's slot: the attention's and the MoE sub-layer's sums.
pub(crate) const TRUNK: usize = 0;

/// Engram site `s`'s slot.
#[must_use]
pub(crate) const fn site(s: usize) -> usize {
    1 + s
}

/// The shapes of a rank's sums for the model `hp` describes, in slot order: the trunk's row of
/// `n_embd` values, then each engram site's `engram_wkv` output, a value row and a key row a
/// stream.
pub fn shapes(hp: &Hparams) -> Result<Vec<GatherShape>, GpuError> {
    let sites = hp.engram()?.layer_ids.len();
    let width = |width| GatherShape { rows: 1, width };
    Ok(std::iter::once(width(hp.n_embd))
        .chain(std::iter::repeat_n(
            width((HC_STREAMS + 1) * hp.n_embd),
            sites,
        ))
        .collect())
}
