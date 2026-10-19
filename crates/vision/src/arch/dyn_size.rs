//! The image preprocessing of mtmd's `mtmd_image_preprocessor_dyn_size`, written once for the
//! projectors that run it (`qwen3vl_merger`, `mimovl`): the size rule plans the pixel size, the
//! image is resized onto a `PAD_CEIL` canvas of the projector's pad colour, normalized under the
//! projector's channel statistics and cut in merge order with every patch repeated over the
//! temporal frames. What differs per projector is the three arguments.

use crate::preprocess::{ChannelStats, GridPlan, PatchLayout, Patches, patchify_with};
use crate::resample::pad_ceil;
use crate::size::SizeRule;
use crate::{Rgb8, VisionError};

/// An image to the encoder's input: planned to a multiple of `patch * merge` by `rule`, resized
/// onto a canvas of that size over `pad`, normalized under `stats` and cut in merge order with each
/// patch repeated over `frames` temporal frames. The plan carries the pixel size and the
/// merged-token grid.
pub(crate) fn preprocess(
    image: &Rgb8,
    rule: &SizeRule,
    pad: [u8; 3],
    stats: &ChannelStats,
    frames: usize,
) -> Result<Patches, VisionError> {
    let (w_bar, h_bar) = rule.aligned(image.width, image.height)?;
    let canvas = pad_ceil(image, w_bar, h_bar, pad)?;
    let (nx, ny) = rule.grid(w_bar, h_bar);
    let plan = GridPlan {
        n_llm_h: ny,
        n_llm_w: nx,
        best_h: h_bar,
        best_w: w_bar,
    };
    let layout = PatchLayout {
        patch: rule.patch(),
        merge: rule.merge(),
        frames,
    };
    patchify_with(&canvas, plan, layout, stats)
}
