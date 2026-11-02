//! Mistral Large 4's image input as the server's media layer sees it: the placeholder
//! `[IMG][IMG_END]` whose `[IMG]` is expanded to one id per row the tower hands the text model
//! (its merged tokens and the `[IMG_BREAK]` rows between their grid rows, [`super::rows_of`]),
//! `[IMG_END]` an ordinary id of the placeholder text — mtmd's layout for this projector: no text
//! before the image, `[IMG_END]` after it, the break rows inside the image's own rows — and the
//! preprocessing of an image (size plan, pad on black, normalize with the file's statistics,
//! merge-order patches) as the prepare. Its content parts join by llama-server's rule
//! ([`PartJoin::LlamaServer`]).
//!
//! The ids belong to the text model's tokenizer (Mistral's Tekken vocabulary), not to the encoder
//! file.

use std::borrow::Cow;

use super::rows_of;
use crate::GridPlan;
use crate::media::{MediaModel, PartJoin, Prepared};
use crate::preprocess::{ChannelStats, PatchLayout, Patches, patchify_with};
use crate::resample::pad_ceil;
use crate::size::SizeRule;
use crate::{Rgb8, VisionError};

/// The placeholder the prompt carries for one image: the image token and mtmd's `img_end`.
pub const IMAGE_PLACEHOLDER: &str = "[IMG][IMG_END]";

/// `[IMG]`: the id [`crate::media::expand_spans`] expands to one copy per row of the image
/// ([`Prepared::span_len`]).
pub const IMAGE_TOKEN_ID: u32 = 10;

/// `[IMG_BREAK]`: the id whose embedding the file carries as `v.token_embd.img_break`; the tower
/// writes that row itself, so the prompt never holds this id.
pub const IMAGE_BREAK_ID: u32 = 12;

/// `[IMG_END]`: the last id [`IMAGE_PLACEHOLDER`] tokenizes to.
pub const IMAGE_END_ID: u32 = 13;

/// The canvas colour of this projector's `PAD_CEIL` pad: llama.cpp's default `image_pad_color`.
pub const PAD_BLACK: [u8; 3] = [0, 0, 0];

/// An image to the encoder's input: planned to a multiple of `patch * merge` by the size rule,
/// resized onto a [`PAD_BLACK`] canvas of that size (a size already the plan's is a copy),
/// normalized with `stats`, and cut in merge order one frame a patch. The plan carries the pixel
/// size and the merged-token grid.
pub fn preprocess(
    image: &Rgb8,
    rule: &SizeRule,
    stats: &ChannelStats,
) -> Result<Patches, VisionError> {
    let (w_bar, h_bar) = rule.aligned(image.width, image.height)?;
    let canvas = pad_ceil(image, w_bar, h_bar, PAD_BLACK)?;
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

/// Mistral Large 4's [`MediaModel`]: the size rule and the channel statistics of its encoder
/// file, the rest constants.
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
        let (nx, ny) = (patches.plan.n_llm_w, patches.plan.n_llm_h);
        Ok(Prepared::of_grid(rows_of(nx, ny), patches))
    }
}

#[cfg(test)]
mod tests {
    use super::{IMAGE_PLACEHOLDER, Media, PAD_BLACK, preprocess};
    use crate::Rgb8;
    use crate::media::{MediaModel, PartJoin};
    use crate::preprocess::{CLIP, normalize_channel};
    use crate::size::{SizeRule, TokenLimits};

    /// mtmd's budget for the file: 8 to 1,024 tokens of 28×28 pixels.
    fn rule() -> SizeRule {
        SizeRule::new(14, 2, TokenLimits::new(8, 1024).expect("limits")).expect("rule")
    }

    fn media() -> Media {
        Media::new(rule(), CLIP)
    }

    #[test]
    fn data_is_the_seats() {
        let m = media();
        assert_eq!(m.image_placeholder(0), "[IMG][IMG_END]");
        assert_eq!(m.image_marker(), IMAGE_PLACEHOLDER);
        assert_eq!(m.image_token(), Some(10));
        assert_eq!((super::IMAGE_BREAK_ID, super::IMAGE_END_ID), (12, 13));
        assert_eq!(PAD_BLACK, [0, 0, 0]);
        assert_eq!(m.part_join(), PartJoin::LlamaServer);
    }

    /// A 448×448 image is its own plan (16×16 merged tokens, 32×32 patches of 588 values), and its
    /// span is the 256 tokens and the 15 break rows between its 16 grid rows.
    #[test]
    fn a_448_image_is_256_tokens_and_15_breaks() {
        let p = media()
            .prepare(&Rgb8::filled(448, 448, [10, 200, 30]))
            .expect("prepare");
        let patches = p.grid().expect("one grid");
        assert_eq!(p.span_len, 271);
        assert_eq!((patches.plan.best_w, patches.plan.best_h), (448, 448));
        assert_eq!((patches.plan.n_llm_w, patches.plan.n_llm_h), (16, 16));
        assert_eq!((patches.n_vit_h, patches.n_vit_w), (32, 32));
        assert_eq!(patches.patch_len, 588);
        let patch = &patches.bf16[..588];
        for (c, v) in [(0, 10u8), (1, 200), (2, 30)] {
            let want = normalize_channel(v, CLIP.mean[c], CLIP.std[c]);
            assert!(
                patch[c * 196..(c + 1) * 196].iter().all(|&x| x == want),
                "channel {c}"
            );
        }
    }

    /// The plans of the oracle set's images, each derived by hand from mtmd's
    /// `calc_size_preserved_ratio` at align 28, 6,272 to 802,816 pixels (f32 arithmetic,
    /// `std::round`): `(w, h) → (w̄, h̄)`, and the rows `nx·ny + ny − 1`.
    #[test]
    fn the_sets_images_plan_as_mtmd_does() {
        for (w, h, bw, bh, rows) in [
            // round(640 / 28 = 22.86) = 23, round(480 / 28 = 17.14) = 17; resized and padded
            (640, 480, 644, 476, 23 * 17 + 16),
            // 37 × 27 = 999 tokens, 783,216 pixels: in the budget
            (1024, 768, 1036, 756, 999 + 26),
            // 84 × 56 is under 6,272: β = sqrt(6272 / 6144), ceil(64β / 28) = 3, ceil(96β / 28) = 4
            (96, 64, 112, 84, 12 + 2),
            (200, 200, 196, 196, 49 + 6),
            (1000, 700, 1008, 700, 900 + 24),
            (777, 513, 784, 504, 504 + 17),
            // over 802,816: β = sqrt(1800 · 2600 / 802,816) = 2.4144, floor(745.5 / 28) = 26,
            // floor(1076.9 / 28) = 38
            (2600, 1800, 1064, 728, 988 + 25),
            // one grid row: no break
            (1000, 1, 1008, 28, 36),
        ] {
            let p = preprocess(&Rgb8::filled(w, h, [90, 120, 30]), &rule(), &CLIP).expect("plan");
            assert_eq!((p.plan.best_w, p.plan.best_h), (bw, bh), "{w}x{h}");
            let span = media()
                .prepare(&Rgb8::filled(w, h, [90, 120, 30]))
                .expect("prepare")
                .span_len;
            assert_eq!(span, rows, "{w}x{h}");
        }
    }
}
