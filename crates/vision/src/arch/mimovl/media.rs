//! MiMo-V2.6-Flash's image input as the server's media layer sees it: the placeholder
//! `<|vision_start|><|image_pad|><|vision_end|>` whose `<|image_pad|>` is expanded to one id per
//! merged token (the delimiters are ordinary ids of the placeholder text), and the preprocessing of
//! an image (size plan, pad on this projector's colour, ImageNet normalize, merge-order patches) as
//! the prepare. Its content parts join by llama-server's rule ([`PartJoin::LlamaServer`]).

use std::borrow::Cow;

use super::{IMAGENET, TEMPORAL_FRAMES};
use crate::arch::dyn_size;
use crate::media::{MediaModel, PartJoin, Prepared};
use crate::preprocess::Patches;
use crate::size::SizeRule;
use crate::{Rgb8, VisionError};

/// The placeholder MiMo's prompt carries for one image.
pub const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// `<|vision_start|>`, config.json's `vision_start_token_id`: the first id [`IMAGE_PLACEHOLDER`]
/// tokenizes to. The ids of this file belong to the text model's tokenizer, not to the encoder
/// file.
pub const VISION_START_ID: u32 = 151_652;

/// `<|vision_end|>`, config.json's `vision_end_token_id`: the last id [`IMAGE_PLACEHOLDER`]
/// tokenizes to.
pub const VISION_END_ID: u32 = 151_653;

/// `<|image_pad|>`, config.json's `image_token_id`: the id between the two, which
/// [`crate::media::expand_spans`] expands to one copy per merged token ([`Prepared::span_len`]).
pub const IMAGE_TOKEN_ID: u32 = 151_655;

/// The canvas colour of this projector's `PAD_CEIL` pad: llama.cpp's `image_pad_color` default,
/// which `mimovl` leaves unset, is black.
pub const PAD_BLACK: [u8; 3] = [0, 0, 0];

/// An image to the encoder's input: planned to a multiple of `patch * merge` by the size rule,
/// resized onto a [`PAD_BLACK`] canvas of that size, normalized with ImageNet's statistics, and
/// cut in merge order with each patch repeated over the temporal frames. The plan carries the
/// pixel size and the merged-token grid.
pub fn preprocess(image: &Rgb8, rule: &SizeRule) -> Result<Patches, VisionError> {
    dyn_size::preprocess(image, rule, PAD_BLACK, &IMAGENET, TEMPORAL_FRAMES)
}

/// MiMo's [`MediaModel`]: the size rule of its encoder file, the rest constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Media {
    rule: SizeRule,
}

impl Media {
    /// The media description of an encoder file whose size rule is `rule`
    /// ([`super::Hparams::size_rule`], after `Hparams::read` refused every value this crate does
    /// not run).
    #[must_use]
    pub fn new(rule: SizeRule) -> Media {
        Media { rule }
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
        let patches = preprocess(image, &self.rule)?;
        Ok(Prepared::of_grid(
            patches.plan.n_llm_h * patches.plan.n_llm_w,
            patches,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::{
        IMAGE_PLACEHOLDER, IMAGE_TOKEN_ID, Media, PAD_BLACK, VISION_END_ID, VISION_START_ID,
    };
    use crate::Rgb8;
    use crate::arch::mimovl::IMAGENET;
    use crate::media::{MediaModel, PartJoin};
    use crate::preprocess::{ChannelStats, normalize};
    use crate::size::{SizeRule, TokenLimits};

    /// The file's own budget: 8 to 8,192 tokens of 32² pixels.
    fn media() -> Media {
        Media::new(SizeRule::new(16, 2, TokenLimits::new(8, 8192).expect("limits")).expect("rule"))
    }

    #[test]
    fn data_is_mimos() {
        let m = media();
        assert_eq!(
            m.image_placeholder(0),
            "<|vision_start|><|image_pad|><|vision_end|>"
        );
        assert!(matches!(
            m.image_placeholder(5),
            Cow::Borrowed(IMAGE_PLACEHOLDER)
        ));
        assert_eq!(m.image_marker(), IMAGE_PLACEHOLDER);
        assert_eq!(m.image_token(), Some(IMAGE_TOKEN_ID));
        assert_eq!(
            (VISION_START_ID, IMAGE_TOKEN_ID, VISION_END_ID),
            (151_652, 151_655, 151_653)
        );
        assert_eq!(PAD_BLACK, [0, 0, 0]);
        assert_eq!(m.part_separator(), "\n");
        assert_eq!(m.part_join(), PartJoin::LlamaServer);
    }

    /// A 448×448 image is its own plan (14×14 merged tokens, 28×28 patches of 1,536 values), and
    /// a single-colour image gives three constant planes under ImageNet's statistics, repeated
    /// over the two frames. ImageNet's channels differ, so a plane's value is its channel's.
    #[test]
    fn a_448_image_is_196_tokens_in_imagenet_statistics() {
        let p = media()
            .prepare(&Rgb8::filled(448, 448, [10, 200, 30]))
            .expect("prepare");
        let patches = p.grid().expect("one grid");
        assert_eq!(p.span_len, 196);
        assert_eq!((patches.plan.best_w, patches.plan.best_h), (448, 448));
        assert_eq!((patches.plan.n_llm_w, patches.plan.n_llm_h), (14, 14));
        assert_eq!((patches.n_vit_h, patches.n_vit_w), (28, 28));
        assert_eq!(patches.patch_len, 1536);
        assert_eq!(patches.bf16.len(), 784 * 1536);
        let patch = &patches.bf16[..1536];
        for (frame, c, v) in [
            (0, 0, 10),
            (0, 1, 200),
            (0, 2, 30),
            (1, 0, 10),
            (1, 1, 200),
            (1, 2, 30),
        ] {
            let plane = &patch[frame * 768 + c * 256..frame * 768 + (c + 1) * 256];
            let want = IMAGENET.normalize(c, v);
            assert!(
                plane.iter().all(|&x| x == want),
                "frame {frame} channel {c}"
            );
        }
        assert_ne!(
            IMAGENET.normalize(0, 10),
            ChannelStats::HALF.normalize(0, 10)
        );
        assert_ne!(IMAGENET.normalize(0, 10), normalize(10));
    }

    /// 96×64 (6,144 pixels, under the file's 8,192) is planned to 128×96 (12 tokens) and resized
    /// to 128×86 at offset 5: of the top-left patch's 16 rows, 5 are black and 11 the white
    /// image; black is the pad colour (0) under ImageNet's statistics, not `-1.0`.
    #[test]
    fn a_small_image_is_scaled_up_and_padded_black() {
        let p = media()
            .prepare(&Rgb8::filled(96, 64, [255, 255, 255]))
            .expect("prepare");
        let patches = p.grid().expect("one grid");
        assert_eq!(p.span_len, 12);
        assert_eq!((patches.plan.best_w, patches.plan.best_h), (128, 96));
        assert_eq!((patches.n_vit_h, patches.n_vit_w), (6, 8));
        let plane = |patch: usize, c: usize| &patches.bf16[patch * 1536 + c * 256..][..256];
        for c in 0..3 {
            let (black, white) = (IMAGENET.normalize(c, 0), IMAGENET.normalize(c, 255));
            let first = plane(0, c);
            for py in 0..16 {
                let want = if py < 5 { black } else { white };
                assert!(
                    first[py * 16..(py + 1) * 16].iter().all(|&v| v == want),
                    "channel {c}, row {py}"
                );
            }
        }
    }

    /// The grid a span hands the prompt follows the image's width, planned at the file's rule
    /// (one token per 32×32 pixels); the expected grids are written out from the image's own size
    /// (`w → round(w / 32)` tokens), not read from the plan.
    #[test]
    fn a_spans_grid_runs_along_the_images_width() {
        for (w, h, across, down) in [
            (640, 320, 20, 10),
            (320, 640, 10, 20),
            // 1000 / 32 = 31.25 rounds to 31 and 700 / 32 = 21.875 to 22: 682 tokens, the
            // 992×704 canvas of the oracle's `pad-1000x700` image.
            (1000, 700, 31, 22),
            (700, 1000, 22, 31),
        ] {
            let p = media()
                .prepare(&Rgb8::filled(w, h, [10, 200, 30]))
                .expect("prepare");
            let patches = p.grid().expect("one grid");
            let (nx, ny) = patches.plan.span_grid();
            assert_eq!((nx, ny), (across, down), "{w}x{h}");
            assert_eq!(p.span_len, across * down, "{w}x{h}");
        }
    }

    /// The oracle's six test images plan to the sizes and token counts its tower dump recorded
    /// (`# image` lines of `ref_mimo2_vis_preproc`): `name w h best_w best_h nx ny`.
    #[test]
    fn the_oracles_images_plan_as_the_oracle_did() {
        for (w, h, best_w, best_h, nx, ny) in [
            (448, 448, 448, 448, 14, 14),
            (640, 480, 640, 480, 20, 15),
            (1024, 768, 1024, 768, 32, 24),
            (96, 64, 128, 96, 4, 3),
            (1000, 700, 992, 704, 31, 22),
            (777, 513, 768, 512, 24, 16),
        ] {
            let p = media()
                .prepare(&Rgb8::filled(w, h, [90, 120, 30]))
                .expect("prepare");
            let g = p.grid().expect("one grid");
            assert_eq!(
                (g.plan.best_w, g.plan.best_h, g.plan.n_llm_w, g.plan.n_llm_h),
                (best_w, best_h, nx, ny),
                "{w}x{h}"
            );
            assert_eq!(p.span_len, nx * ny, "{w}x{h}");
        }
    }
}
