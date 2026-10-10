//! V4.1's image input as the server's media layer sees it: the placeholder `<｜deepseek_image｜>`
//! and its token, `"\n\n"` between flattened content parts, and the reference `load_image` (plan,
//! pad or resize, normalize, patch) as the prepare.

use super::IMAGE_TOKEN_ID;
use super::grid::{GridParams, num_image_tokens, plan_image_grid};
use crate::media::{MediaModel, PartJoin, Prepared};
use crate::preprocess::{PatchLayout, Patches, patchify};
use crate::resample::pad;
use crate::{GridPlan, Rgb8, VisionError};

/// The text of [`IMAGE_TOKEN_ID`], the placeholder V4.1's chat carries for one image.
pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

/// The separator between V4.1's flattened content parts.
pub const PART_SEPARATOR: &str = "\n\n";

/// The fill of the reference's pad (`color=(127, 127, 127)`).
pub const PAD_GREY: [u8; 3] = [127, 127, 127];

/// Pad an image to its plan's pixel size, grey-filled, as the reference does.
pub fn padded(image: &Rgb8, plan: &GridPlan) -> Result<Rgb8, VisionError> {
    pad(image, plan.best_w, plan.best_h, PAD_GREY)
}

/// Cut an image already at a multiple of `patch` into normalized patches, row-major.
pub fn to_patches(image: &Rgb8, plan: GridPlan, p: &GridParams) -> Result<Patches, VisionError> {
    patchify(image, plan, PatchLayout::row_major(p.patch))
}

/// The reference `load_image` from a decoded image: plan, pad, normalize, patch.
pub fn preprocess(image: &Rgb8, p: &GridParams) -> Result<Patches, VisionError> {
    let plan = plan_image_grid(image.width, image.height, p);
    to_patches(&padded(image, &plan)?, plan, p)
}

/// V4.1's [`MediaModel`]: the resize parameters of its encoder file, the rest constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Media {
    grid: GridParams,
}

impl Media {
    /// The media description of an encoder file whose resize parameters are `grid`
    /// ([`super::Hparams::grid`], read after `Hparams::read` refused every value the reference
    /// does not run).
    #[must_use]
    pub fn new(grid: GridParams) -> Media {
        Media { grid }
    }
}

impl MediaModel for Media {
    fn image_placeholder(&self) -> &str {
        IMAGE_PLACEHOLDER
    }

    fn image_token(&self) -> u32 {
        IMAGE_TOKEN_ID
    }

    fn part_separator(&self) -> &str {
        PART_SEPARATOR
    }

    fn part_join(&self) -> PartJoin {
        PartJoin::Uniform(PART_SEPARATOR)
    }

    fn prepare(&self, image: &Rgb8) -> Result<Prepared, VisionError> {
        let patches = preprocess(image, &self.grid)?;
        Ok(Prepared {
            span_len: num_image_tokens(patches.plan.n_llm_h, patches.plan.n_llm_w),
            patches,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{IMAGE_PLACEHOLDER, Media, PAD_GREY, padded, preprocess, to_patches};
    use crate::arch::deepseek41v::grid::{GridParams, plan_image_grid};
    use crate::arch::deepseek41v::span::image_span;
    use crate::media::{MediaModel, PartJoin};
    use crate::preprocess::{PatchLayout, patchify};
    use crate::{GridPlan, Rgb8};

    /// The reference config's resize parameters.
    const V41: GridParams = GridParams {
        patch: 14,
        downsample: 3,
        max_tokens: 1024,
        min_pixels: 295_936,
    };

    #[test]
    fn data_is_v41s() {
        let m = Media::new(V41);
        assert_eq!(m.image_placeholder(), IMAGE_PLACEHOLDER);
        assert_eq!(m.image_placeholder(), "<\u{FF5C}deepseek_image\u{FF5C}>");
        assert_eq!(m.image_token(), 129_264);
        assert_eq!(m.part_separator(), "\n\n");
        assert_eq!(m.part_join(), PartJoin::Uniform("\n\n"));
    }

    /// A 448×448 image is scaled up to the 546×546 grid: 13×13 aligner tokens, 184 positions with
    /// the delimiters (the research table's first row), and the span the expansion makes has the
    /// length `image_span` gives the plan.
    #[test]
    fn prepare_gives_the_plans_span() {
        let m = Media::new(V41);
        let p = m
            .prepare(&Rgb8::filled(448, 448, [10, 200, 30]))
            .expect("prepare");
        assert_eq!(p.span_len, 184);
        assert_eq!((p.patches.plan.best_w, p.patches.plan.best_h), (546, 546));
        assert_eq!((p.patches.n_vit_h, p.patches.n_vit_w), (39, 39));
        assert_eq!(p.patches.bf16.len(), 39 * 39 * 3 * 14 * 14);
        assert_eq!(
            image_span(&p.patches.plan, m.image_token()).ids.len(),
            p.span_len
        );
    }

    /// The reference's pad is grey: a 100×91 image on a 100×100 plan keeps its colour on the rows
    /// the paste covers (offset 4) and takes [`PAD_GREY`] above and below.
    #[test]
    fn padded_fills_grey() {
        let plan = GridPlan {
            n_llm_h: 1,
            n_llm_w: 1,
            best_h: 100,
            best_w: 100,
        };
        let out = padded(&Rgb8::filled(100, 91, [10, 20, 30]), &plan).expect("pad");
        assert_eq!((out.width, out.height), (100, 100));
        let px = |y: usize, x: usize| &out.data[(y * 100 + x) * 3..(y * 100 + x) * 3 + 3];
        // The reference's `color=(127, 127, 127)`, written out.
        assert_eq!(PAD_GREY, [127, 127, 127]);
        for y in 0..100 {
            let want = if (4..95).contains(&y) {
                [10, 20, 30]
            } else {
                [127, 127, 127]
            };
            assert_eq!(px(y, 0), want, "row {y}");
            assert_eq!(px(y, 99), want, "row {y}");
        }
    }

    /// V4.1's cut is the shared cut at one patch per group and one frame, on several shapes and
    /// patch sizes.
    #[test]
    fn to_patches_is_the_row_major_cut() {
        for (s, w, h) in [
            (14, 42, 28),
            (14, 14, 70),
            (2, 8, 6),
            (1, 5, 3),
            (16, 64, 32),
        ] {
            let mut img = Rgb8::filled(w, h, [0, 0, 0]);
            for (i, v) in img.data.iter_mut().enumerate() {
                *v = (i * 7 % 251) as u8;
            }
            let plan = GridPlan {
                n_llm_h: 1,
                n_llm_w: 1,
                best_h: h,
                best_w: w,
            };
            let p = GridParams {
                patch: s,
                downsample: 1,
                max_tokens: 1,
                min_pixels: 1,
            };
            let got = to_patches(&img, plan, &p).expect("cut");
            assert_eq!(
                got,
                patchify(&img, plan, PatchLayout::row_major(s)).expect("cut"),
                "{s}-pixel patches of {w}x{h}"
            );
            assert_eq!(
                (got.n_vit_h, got.n_vit_w, got.patch_len),
                (h / s, w / s, 3 * s * s)
            );
        }
    }

    /// `preprocess` is the plan, the grey pad and the row-major cut in that order.
    #[test]
    fn preprocess_is_plan_pad_cut() {
        let mut img = Rgb8::filled(300, 200, [0, 0, 0]);
        for (i, v) in img.data.iter_mut().enumerate() {
            *v = (i * 13 % 253) as u8;
        }
        let plan = plan_image_grid(img.width, img.height, &V41);
        let want = to_patches(&padded(&img, &plan).expect("pad"), plan, &V41).expect("cut");
        assert_eq!(preprocess(&img, &V41).expect("preprocess"), want);
        assert_eq!(want.plan, plan);
    }
}
