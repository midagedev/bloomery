//! Hy4-flash's image input as the server's media layer sees it: the placeholder
//! `<｜hy_image▁start｜><｜hy_image▁pad｜><｜hy_image▁end｜>` (each with the tokenizer's revision tag)
//! whose pad id is expanded to one id per row the tower hands the text model (its merged tokens
//! and the newline row after each of their grid rows, [`super::rows_of`]) — mtmd's layout for this
//! projector: `img_beg` before the image, `img_end` after it, the newline rows inside the image's
//! own rows — and the preprocessing of an image (size plan, a direct Lanczos resize, normalize with
//! the file's statistics, merge-order patches) as the prepare. Its content parts join by
//! llama-server's rule ([`PartJoin::LlamaServer`]).
//!
//! The ids belong to the text model's tokenizer, not to the encoder file; mtmd finds the two
//! delimiters by their spelling's prefix, so the tag is the vocabulary's.

use std::borrow::Cow;

use super::rows_of;
use crate::GridPlan;
use crate::media::{MediaModel, PartJoin, Prepared};
use crate::preprocess::{ChannelStats, PatchLayout, Patches, patchify_with};
use crate::resample::resize_lanczos;
use crate::size::SizeRule;
use crate::{Rgb8, VisionError};

/// `<｜hy_image▁start｜>` as Hy4-flash's vocabulary spells it: mtmd's `img_beg`.
pub const IMAGE_START: &str = "<｜hy_image▁start:6124c78e｜>";

/// `<｜hy_image▁pad｜>`: the token the placeholder expands, one copy a row of the image.
pub const IMAGE_PAD: &str = "<｜hy_image▁pad:6124c78e｜>";

/// `<｜hy_image▁end｜>`: mtmd's `img_end`.
pub const IMAGE_END: &str = "<｜hy_image▁end:6124c78e｜>";

/// The placeholder the prompt carries for one image: the three tokens, start, pad and end.
pub const IMAGE_PLACEHOLDER: &str =
    "<｜hy_image▁start:6124c78e｜><｜hy_image▁pad:6124c78e｜><｜hy_image▁end:6124c78e｜>";

/// The id of [`IMAGE_START`].
pub const IMAGE_START_ID: u32 = 120_118;

/// The id of [`IMAGE_END`].
pub const IMAGE_END_ID: u32 = 120_119;

/// The id of [`IMAGE_PAD`]: the id [`crate::media::expand_spans`] expands to one copy per row of
/// the image ([`Prepared::span_len`]).
pub const IMAGE_PAD_ID: u32 = 120_120;

/// An image to the encoder's input: planned to a multiple of `patch * merge` by the size rule,
/// resized to that size with the Lanczos filter (a size already the plan's is a copy), normalized
/// with `stats`, and cut in merge order one frame a patch. The plan carries the pixel size and the
/// merged-token grid.
pub fn preprocess(
    image: &Rgb8,
    rule: &SizeRule,
    stats: &ChannelStats,
) -> Result<Patches, VisionError> {
    let (w_bar, h_bar) = rule.aligned(image.width, image.height)?;
    let resized = resize_lanczos(image, w_bar, h_bar)?;
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
    patchify_with(&resized, plan, layout, stats)
}

/// Hy4-flash's [`MediaModel`]: the size rule and the channel statistics of its encoder file, the
/// rest constants.
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
        Some(IMAGE_PAD_ID)
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
    use super::{IMAGE_END, IMAGE_PAD, IMAGE_PLACEHOLDER, IMAGE_START, Media, preprocess};
    use crate::Rgb8;
    use crate::media::{MediaModel, PartJoin};
    use crate::preprocess::{ChannelStats, normalize_channel};
    use crate::size::{SizeRule, TokenLimits};

    /// mtmd's budget for the file: 4 to 4,096 tokens of 32×32 pixels.
    fn rule() -> SizeRule {
        SizeRule::new(16, 2, TokenLimits::new(4, 4096).expect("limits")).expect("rule")
    }

    fn media() -> Media {
        Media::new(rule(), ChannelStats::HALF)
    }

    #[test]
    fn data_is_the_seats() {
        let m = media();
        assert_eq!(
            m.image_placeholder(0),
            format!("{IMAGE_START}{IMAGE_PAD}{IMAGE_END}")
        );
        assert_eq!(m.image_marker(), IMAGE_PLACEHOLDER);
        assert_eq!(m.image_token(), Some(120_120));
        assert_eq!(
            (super::IMAGE_START_ID, super::IMAGE_END_ID),
            (120_118, 120_119)
        );
        assert!(IMAGE_START.starts_with("<｜hy_image▁start"));
        assert!(IMAGE_END.starts_with("<｜hy_image▁end"));
        assert_eq!(m.part_join(), PartJoin::LlamaServer);
    }

    /// A 512×512 image is its own plan (16×16 merged tokens, 32×32 patches of 768 values), and its
    /// span is the 256 tokens and a newline row after each of its 16 grid rows.
    #[test]
    fn a_512_image_is_256_tokens_and_16_newlines() {
        let p = media()
            .prepare(&Rgb8::filled(512, 512, [10, 200, 30]))
            .expect("prepare");
        let patches = p.grid().expect("one grid");
        assert_eq!(p.span_len, 272);
        assert_eq!((patches.plan.best_w, patches.plan.best_h), (512, 512));
        assert_eq!((patches.plan.n_llm_w, patches.plan.n_llm_h), (16, 16));
        assert_eq!((patches.n_vit_h, patches.n_vit_w), (32, 32));
        assert_eq!(patches.patch_len, 768);
        let half = ChannelStats::HALF;
        let patch = &patches.bf16[..768];
        for (c, v) in [(0, 10u8), (1, 200), (2, 30)] {
            let want = normalize_channel(v, half.mean[c], half.std[c]);
            assert!(
                patch[c * 256..(c + 1) * 256].iter().all(|&x| x == want),
                "channel {c}"
            );
        }
    }

    /// The plans of the oracle set's images, each derived by hand from mtmd's
    /// `calc_size_preserved_ratio` at align 32, 4,096 to 4,194,304 pixels (f32 arithmetic,
    /// `std::round`): `(w, h) → (w̄, h̄)`, and the rows `(nx + 1)·ny`.
    #[test]
    fn the_sets_images_plan_as_mtmd_does() {
        for (w, h, bw, bh, rows) in [
            // round(640 / 32 = 20) = 20, round(480 / 32 = 15) = 15: no resize
            (640, 480, 640, 480, 21 * 15),
            // round(1000 / 32 = 31.25) = 31, round(700 / 32 = 21.875) = 22
            (1000, 700, 992, 704, 32 * 22),
            // round(777 / 32 = 24.28) = 24, round(513 / 32 = 16.03) = 16
            (777, 513, 768, 512, 25 * 16),
            // round(200 / 32 = 6.25) = 6
            (200, 200, 192, 192, 7 * 6),
            // 32 × 32 is under 4,096: β = sqrt(4096 / 1024) = 2, ceil(32·2 / 32) = 2
            (32, 32, 64, 64, 3 * 2),
            // over 4,194,304: β = sqrt(3000 · 2000 / 4,194,304) = 1.19604, floor(2508.3 / 32) = 78,
            // floor(1672.2 / 32) = 52
            (3000, 2000, 2496, 1664, 79 * 52),
            // one grid row: round(1 / 32) = 0 is lifted to one token side, 1 newline
            (1000, 1, 992, 32, 32),
            // the rest of tools/ref/clefvis/images-hy4.tsv: multiples of 32 in the budget
            (448, 448, 448, 448, 15 * 14),
            (1024, 768, 1024, 768, 33 * 24),
            (96, 64, 96, 64, 4 * 2),
            // over 4,194,304: β = sqrt(2600 · 1800 / 4,194,304) = 1.05631, floor(2461.4 / 32) = 76,
            // floor(1704.0 / 32) = 53
            (2600, 1800, 2432, 1696, 77 * 53),
            // β = sqrt(3000 · 2200 / 4,194,304) = 1.25442, floor(2391.6 / 32) = 74,
            // floor(1753.8 / 32) = 54
            (3000, 2200, 2368, 1728, 75 * 54),
        ] {
            let p = preprocess(
                &Rgb8::filled(w, h, [90, 120, 30]),
                &rule(),
                &ChannelStats::HALF,
            )
            .expect("plan");
            assert_eq!((p.plan.best_w, p.plan.best_h), (bw, bh), "{w}x{h}");
            let span = media()
                .prepare(&Rgb8::filled(w, h, [90, 120, 30]))
                .expect("prepare")
                .span_len;
            assert_eq!(span, rows, "{w}x{h}");
        }
    }
}
