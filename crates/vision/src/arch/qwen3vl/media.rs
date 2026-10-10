//! Clef's image input as the server's media layer sees it: the placeholder
//! `<|vision_start|><|image_pad|><|vision_end|>` whose `<|image_pad|>` is expanded to one id per
//! merged token (the delimiters are ordinary ids of the placeholder text), and the preprocessing
//! of an image (size plan, pad on this projector's colour, normalize, merge-order patches) as the
//! prepare. Its content parts join by llama-server's rule ([`PartJoin::LlamaServer`]).

use super::TEMPORAL_FRAMES;
use super::size::SizeRule;
use crate::GridPlan;
use crate::media::{MediaModel, PartJoin, Prepared};
use crate::preprocess::{PatchLayout, Patches, patchify};
use crate::resample::pad_ceil;
use crate::{Rgb8, VisionError};

/// The placeholder Clef's prompt carries for one image.
pub const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// `<|vision_start|>`, config.json's `vision_start_token_id`: the first id [`IMAGE_PLACEHOLDER`]
/// tokenizes to. The ids of this file belong to the text model's tokenizer, not to the encoder
/// file.
pub const VISION_START_ID: u32 = 248_053;

/// `<|vision_end|>`, config.json's `vision_end_token_id`: the last id [`IMAGE_PLACEHOLDER`]
/// tokenizes to.
pub const VISION_END_ID: u32 = 248_054;

/// `<|image_pad|>`, config.json's `image_token_id`: the id between the two, which
/// `serve::media::expand_spans` expands to one copy per merged token ([`Prepared::span_len`]).
pub const IMAGE_TOKEN_ID: u32 = 248_056;

/// The canvas colour of this projector's `PAD_CEIL` pad: llama.cpp's `image_pad_color` for
/// `qwen3vl_merger` is black.
pub const PAD_BLACK: [u8; 3] = [0, 0, 0];

/// An image to the encoder's input: planned to a multiple of `patch * merge` by the size rule,
/// resized onto a [`PAD_BLACK`] canvas of that size, normalized, and cut in merge order with each
/// patch repeated over the temporal frames. The plan carries the pixel size and the merged-token
/// grid.
pub fn preprocess(image: &Rgb8, rule: &SizeRule) -> Result<Patches, VisionError> {
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
        frames: TEMPORAL_FRAMES,
    };
    patchify(&canvas, plan, layout)
}

/// Clef's [`MediaModel`]: the size rule of its encoder file, the rest constants.
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
    fn image_placeholder(&self) -> &str {
        IMAGE_PLACEHOLDER
    }

    fn image_token(&self) -> u32 {
        IMAGE_TOKEN_ID
    }

    fn part_join(&self) -> PartJoin {
        PartJoin::LlamaServer
    }

    fn prepare(&self, image: &Rgb8) -> Result<Prepared, VisionError> {
        let patches = preprocess(image, &self.rule)?;
        Ok(Prepared {
            span_len: patches.plan.n_llm_h * patches.plan.n_llm_w,
            patches,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{IMAGE_PLACEHOLDER, Media, PAD_BLACK, VISION_END_ID, VISION_START_ID};
    use crate::Rgb8;
    use crate::arch::qwen3vl::size::{SizeRule, TokenLimits};
    use crate::media::{MediaModel, PartJoin};
    use crate::preprocess::{f32_to_bf16, normalize};

    fn media() -> Media {
        Media::new(SizeRule::new(16, 2, TokenLimits::DEFAULT).expect("rule"))
    }

    #[test]
    fn data_is_clefs() {
        let m = media();
        assert_eq!(m.image_placeholder(), IMAGE_PLACEHOLDER);
        assert_eq!(
            m.image_placeholder(),
            "<|vision_start|><|image_pad|><|vision_end|>"
        );
        assert_eq!(m.image_token(), 248_056);
        assert_eq!((VISION_START_ID, VISION_END_ID), (248_053, 248_054));
        assert_eq!(PAD_BLACK, [0, 0, 0]);
        assert_eq!(m.part_separator(), "\n");
        assert_eq!(m.part_join(), PartJoin::LlamaServer);
    }

    /// A 448×448 image is its own plan (14×14 merged tokens, 28×28 patches of 1,536 values), and
    /// a single-colour image gives three constant planes, repeated over the two frames.
    #[test]
    fn a_448_image_is_196_tokens() {
        let p = media()
            .prepare(&Rgb8::filled(448, 448, [10, 200, 30]))
            .expect("prepare");
        assert_eq!(p.span_len, 196);
        assert_eq!((p.patches.plan.best_w, p.patches.plan.best_h), (448, 448));
        assert_eq!((p.patches.plan.n_llm_w, p.patches.plan.n_llm_h), (14, 14));
        assert_eq!((p.patches.n_vit_h, p.patches.n_vit_w), (28, 28));
        assert_eq!(p.patches.patch_len, 1536);
        assert_eq!(p.patches.bf16.len(), 784 * 1536);
        let patch = &p.patches.bf16[..1536];
        for (frame, c, v) in [
            (0, 0, 10),
            (0, 1, 200),
            (0, 2, 30),
            (1, 0, 10),
            (1, 1, 200),
            (1, 2, 30),
        ] {
            let plane = &patch[frame * 768 + c * 256..frame * 768 + (c + 1) * 256];
            assert!(
                plane.iter().all(|&x| x == normalize(v)),
                "frame {frame} channel {c}"
            );
        }
    }

    /// 96×64 is planned to 128×96 (12 tokens) and resized to 128×86 at offset 5: of the top-left
    /// patch's 16 rows, 5 are black (`-1.0`), 11 are the white image (`1.0`); of the last patch of
    /// the last merge group (patch row 5, column 7: pixel rows 80–95) rows 80–90 are image and
    /// 91–95 black.
    #[test]
    fn a_small_image_is_padded_black() {
        let p = media()
            .prepare(&Rgb8::filled(96, 64, [255, 255, 255]))
            .expect("prepare");
        assert_eq!(p.span_len, 12);
        assert_eq!((p.patches.plan.best_w, p.patches.plan.best_h), (128, 96));
        assert_eq!((p.patches.n_vit_h, p.patches.n_vit_w), (6, 8));
        let (black, white) = (f32_to_bf16(-1.0), f32_to_bf16(1.0));
        let plane = |patch: usize, c: usize| &p.patches.bf16[patch * 1536 + c * 256..][..256];
        let first = plane(0, 0);
        for py in 0..16 {
            let want = if py < 5 { black } else { white };
            assert!(
                first[py * 16..(py + 1) * 16].iter().all(|&v| v == want),
                "first, row {py}"
            );
        }
        // 48 patches, 12 groups of 4: the last patch is index 47.
        let last = plane(47, 2);
        for py in 0..16 {
            let want = if py < 11 { white } else { black };
            assert!(
                last[py * 16..(py + 1) * 16].iter().all(|&v| v == want),
                "last, row {py}"
            );
        }
    }
}
