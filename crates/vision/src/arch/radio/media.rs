//! Bolt's image input as the server's media layer sees it: the placeholder `<img><image></img>`
//! whose `<image>` is expanded to one id per merged token (the delimiters are ordinary ids of the
//! placeholder text), and the preprocessing of an image (size plan, a direct resize to it,
//! normalize with the file's statistics, merge-order patches) as the prepare. Its content parts
//! join by llama-server's rule ([`PartJoin::LlamaServer`]).

use std::borrow::Cow;

use crate::GridPlan;
use crate::media::{MediaModel, PartJoin, Prepared};
use crate::preprocess::{ChannelStats, PatchLayout, Patches, patchify_with};
use crate::resample::resize_bicubic;
use crate::size::SizeRule;
use crate::{Rgb8, VisionError};

/// The placeholder Bolt's prompt carries for one image: mtmd's `img_beg`, the image token and
/// `img_end`.
pub const IMAGE_PLACEHOLDER: &str = "<img><image></img>";

/// `<image>`, config.json's `img_context_token_id`: the id between the delimiters of
/// [`IMAGE_PLACEHOLDER`], which [`crate::media::expand_spans`] expands to one copy per merged
/// token ([`Prepared::span_len`]).
pub const IMAGE_TOKEN_ID: u32 = 18;

/// An image to the encoder's input: planned to a multiple of `patch * merge` by the size rule,
/// resized to that size directly (llama.cpp's `PAD_NONE`: the aspect follows the plan, nothing
/// is padded, and a size already the plan's is a copy), normalized with `stats`, and cut in merge
/// order one frame a patch. The plan carries the pixel size and the merged-token grid.
pub fn preprocess(
    image: &Rgb8,
    rule: &SizeRule,
    stats: &ChannelStats,
) -> Result<Patches, VisionError> {
    let (w_bar, h_bar) = rule.aligned(image.width, image.height)?;
    let canvas = if (image.width, image.height) == (w_bar, h_bar) {
        image.clone()
    } else {
        resize_bicubic(image, w_bar, h_bar)?
    };
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
        frames: 1,
    };
    patchify_with(&canvas, plan, layout, stats)
}

/// Bolt's [`MediaModel`]: the size rule and the channel statistics of its encoder file, the rest
/// constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Media {
    rule: SizeRule,
    stats: ChannelStats,
}

impl Media {
    /// The media description of an encoder file whose size rule is `rule` and whose pixels are
    /// normalized with `stats` ([`super::Hparams::size_rule`] and [`super::Hparams::stats`],
    /// after `Hparams::read` refused every value this crate does not run).
    #[must_use]
    pub fn new(rule: SizeRule, stats: ChannelStats) -> Media {
        Media { rule, stats }
    }
}

impl MediaModel for Media {
    fn image_placeholder(&self, _index: usize) -> Cow<'_, str> {
        Cow::Borrowed(IMAGE_PLACEHOLDER)
    }

    fn image_marker(&self) -> &str {
        IMAGE_PLACEHOLDER
    }

    fn image_token(&self) -> Option<u32> {
        Some(IMAGE_TOKEN_ID)
    }

    fn part_join(&self) -> PartJoin {
        PartJoin::LlamaServer
    }

    fn prepare(&self, image: &Rgb8) -> Result<Prepared, VisionError> {
        let patches = preprocess(image, &self.rule, &self.stats)?;
        Ok(Prepared::of_grid(
            patches.plan.n_llm_h * patches.plan.n_llm_w,
            patches,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::{IMAGE_PLACEHOLDER, Media, preprocess};
    use crate::Rgb8;
    use crate::media::{MediaModel, PartJoin};
    use crate::preprocess::{CLIP, normalize_channel};
    use crate::size::{SizeRule, TokenLimits};

    /// The file's own budget: 256 to 3,328 tokens of 32×32 pixels.
    fn rule() -> SizeRule {
        SizeRule::new(16, 2, TokenLimits::new(256, 3328).expect("limits")).expect("rule")
    }

    fn media() -> Media {
        Media::new(rule(), CLIP)
    }

    #[test]
    fn data_is_bolts() {
        let m = media();
        assert_eq!(m.image_placeholder(0), IMAGE_PLACEHOLDER);
        assert_eq!(m.image_placeholder(0), "<img><image></img>");
        assert!(matches!(
            m.image_placeholder(5),
            Cow::Borrowed(IMAGE_PLACEHOLDER)
        ));
        assert_eq!(m.image_marker(), IMAGE_PLACEHOLDER);
        assert_eq!(m.image_token(), Some(18));
        assert_eq!(m.part_join(), PartJoin::LlamaServer);
    }

    /// A 512×512 image is its own plan (16×16 merged tokens, 32×32 patches of 768 values: one
    /// frame), and a single-colour image gives three constant planes of the file's statistics.
    #[test]
    fn a_512_image_is_256_tokens() {
        let p = media()
            .prepare(&Rgb8::filled(512, 512, [10, 200, 30]))
            .expect("prepare");
        let patches = p.grid().expect("one grid");
        assert_eq!(p.span_len, 256);
        assert_eq!((patches.plan.best_w, patches.plan.best_h), (512, 512));
        assert_eq!((patches.plan.n_llm_w, patches.plan.n_llm_h), (16, 16));
        assert_eq!((patches.n_vit_h, patches.n_vit_w), (32, 32));
        assert_eq!(patches.patch_len, 768);
        assert_eq!(patches.bf16.len(), 1024 * 768);
        for (c, v) in [(0, 10), (1, 200), (2, 30)] {
            let plane = &patches.bf16[c * 256..(c + 1) * 256];
            let want = normalize_channel(v, CLIP.mean[c], CLIP.std[c]);
            assert!(plane.iter().all(|&x| x == want), "channel {c}");
        }
    }

    /// Below the budget's floor the image is scaled up to it: a 128×96 image is 12 tokens of
    /// pixels, under the 256-token floor, so the plan is `ceil(side * sqrt(floor / pixels))`
    /// aligned up; and the whole canvas is the resized image, with no padded band.
    #[test]
    fn a_small_image_is_stretched_not_padded() {
        let p = media()
            .prepare(&Rgb8::filled(128, 96, [255, 255, 255]))
            .expect("prepare");
        let patches = p.grid().expect("one grid");
        let (w, h) = (patches.plan.best_w, patches.plan.best_h);
        assert!(w.is_multiple_of(32) && h.is_multiple_of(32));
        assert!(
            (256..=3328).contains(&(w * h / 1024)),
            "{w}x{h} is in the budget"
        );
        assert_eq!(p.span_len, w * h / 1024);
        // A white image resized stays white everywhere, the corner patches included.
        let white = normalize_channel(255, CLIP.mean[0], CLIP.std[0]);
        assert_eq!(patches.bf16[0], white);
        let last = patches.bf16.len() - 768;
        assert_eq!(patches.bf16[last], white);
    }

    /// The plan follows the image's width: a landscape image has more tokens along a line than
    /// lines. The expected grids are the size rule's, written from the image's own size.
    #[test]
    fn a_spans_grid_runs_along_the_images_width() {
        for (w, h, across, down) in [
            // 1,280 x 640 is 800 tokens of 32x32: 40 across, 20 down, inside the budget.
            (1280, 640, 40, 20),
            (640, 1280, 20, 40),
        ] {
            let p = media()
                .prepare(&Rgb8::filled(w, h, [10, 200, 30]))
                .expect("prepare");
            let patches = p.grid().expect("one grid");
            assert_eq!(patches.plan.span_grid(), (across, down), "{w}x{h}");
            assert_eq!(p.span_len, across * down);
        }
    }

    /// The plan caps at the budget: a 4,096×4,096 image is planned to at most 3,328 tokens, and
    /// the preprocess of a plain image of that size gives exactly the patches the plan names.
    #[test]
    fn a_large_image_is_planned_into_the_budget() {
        let patches =
            preprocess(&Rgb8::filled(4096, 4096, [1, 2, 3]), &rule(), &CLIP).expect("preprocess");
        let tokens = patches.plan.n_llm_h * patches.plan.n_llm_w;
        assert!(tokens <= 3328, "{tokens} tokens");
        assert_eq!(patches.bf16.len(), tokens * 4 * 768);
    }
}
