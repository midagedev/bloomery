//! GLM-5.3-Flash's image input: the placeholder `<|begin_of_image|><|image|><|end_of_image|>`
//! whose `<|image|>` is expanded to one id per merged token (the delimiters are ordinary ids of
//! the placeholder text), and the preprocessing of an image (canvas plan, resize onto the canvas's
//! top-left corner over black, CLIP normalize, merge-order patches), the rule of mtmd's
//! `mtmd_image_preprocessor_glm5v`.

use super::size::{Canvas, SizeRule};
use super::{CLIP, TEMPORAL_FRAMES};
use crate::media::{MediaModel, PartJoin, Prepared};
use crate::preprocess::{PatchLayout, Patches, patchify_with};
use crate::resample::resize_bicubic;
use crate::{GridPlan, Rgb8, VisionError};

/// The placeholder GLM's prompt carries for one image.
pub const IMAGE_PLACEHOLDER: &str = "<|begin_of_image|><|image|><|end_of_image|>";

/// `<|begin_of_image|>`: the first id [`IMAGE_PLACEHOLDER`] tokenizes to. The ids of this file
/// belong to the text model's tokenizer, not to the encoder file.
pub const BEGIN_OF_IMAGE_ID: u32 = 154_830;

/// `<|end_of_image|>`: the last id [`IMAGE_PLACEHOLDER`] tokenizes to.
pub const END_OF_IMAGE_ID: u32 = 154_831;

/// `<|image|>`: the id between the two, expanded to one copy per merged token (`nx * ny`, the
/// canvas's [`SizeRule::grid`]).
pub const IMAGE_TOKEN_ID: u32 = 154_854;

/// The canvas colour: mtmd fills the canvas with black before it composites the image.
pub const PAD_BLACK: [u8; 3] = [0, 0, 0];

/// The image on its canvas: resized to the canvas's content size (bicubic, Pillow's arithmetic as
/// mtmd's `resize_pillow` computes it) and placed at pixel (0, 0) over [`PAD_BLACK`].
fn canvas_image(image: &Rgb8, canvas: &Canvas) -> Rgb8 {
    let content = resize_bicubic(image, canvas.content_w, canvas.content_h);
    let mut out = Rgb8::filled(canvas.width, canvas.height, PAD_BLACK);
    for y in 0..canvas.content_h {
        let from = &content.data[y * canvas.content_w * 3..(y + 1) * canvas.content_w * 3];
        let at = y * canvas.width * 3;
        out.data[at..at + from.len()].copy_from_slice(from);
    }
    out
}

/// An image to the encoder's input: planned onto a canvas by the size rule, resized and placed at
/// its top-left corner over black, normalized with CLIP's statistics, and cut in merge order with
/// each patch repeated over the temporal frames. The plan carries the canvas's pixel size and the
/// merged-token grid.
pub fn preprocess(image: &Rgb8, rule: &SizeRule) -> Result<Patches, VisionError> {
    let canvas = rule.canvas(image.width, image.height)?;
    let (nx, ny) = rule.grid(&canvas);
    let plan = GridPlan {
        n_llm_h: ny,
        n_llm_w: nx,
        best_h: canvas.height,
        best_w: canvas.width,
    };
    let layout = PatchLayout {
        patch: rule.patch(),
        merge: rule.merge(),
        frames: TEMPORAL_FRAMES,
    };
    patchify_with(&canvas_image(image, &canvas), plan, layout, &CLIP)
}

/// GLM-5.3-Flash's [`MediaModel`]: the size rule of its encoder file at the seat's token limits,
/// the rest constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Media {
    rule: SizeRule,
}

impl Media {
    /// The media description of an encoder file whose size rule is `rule`
    /// ([`super::Hparams::size_rule_at`], after `Hparams::read` refused every value this crate does
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
    use super::{
        BEGIN_OF_IMAGE_ID, END_OF_IMAGE_ID, IMAGE_PLACEHOLDER, IMAGE_TOKEN_ID, Media, PAD_BLACK,
        canvas_image, preprocess,
    };
    use crate::Rgb8;
    use crate::arch::glm5v::{CLIP, SizeRule};
    use crate::media::{MediaModel, PartJoin};

    fn rule() -> SizeRule {
        SizeRule::new(14, 2, 12_544, 6_272_000).expect("rule")
    }

    /// An image whose pixel `(x, y)` channel `c` is `f(c, y, x)`.
    fn image(w: usize, h: usize, f: impl Fn(usize, usize, usize) -> u8) -> Rgb8 {
        let mut data = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                data.extend((0..3).map(|c| f(c, y, x)));
            }
        }
        Rgb8 {
            width: w,
            height: h,
            data,
        }
    }

    /// No pixel of this image is black, so black on the canvas is pad.
    fn bright(c: usize, y: usize, x: usize) -> u8 {
        1 + ((c * 37 + x * 3 + y * 5) % 250) as u8
    }

    fn pixel(img: &Rgb8, x: usize, y: usize) -> [u8; 3] {
        let at = (y * img.width + x) * 3;
        [img.data[at], img.data[at + 1], img.data[at + 2]]
    }

    /// The image sits at the canvas's top-left corner (`composite(canvas, content, 0, 0)`,
    /// mtmd-image.cpp:857) and the rest of the 1008×616 canvas is black: pixel (0, 0) and the
    /// content's far corner are the image's, the column and row past it and the canvas's far
    /// corner are black.
    #[test]
    fn the_image_sits_top_left_over_black() {
        let img = image(1000, 600, bright);
        let canvas = rule().canvas(1000, 600).unwrap();
        assert_eq!((canvas.width, canvas.height), (1008, 616));
        let out = canvas_image(&img, &canvas);
        assert_eq!((out.width, out.height), (1008, 616));
        assert_eq!(pixel(&out, 0, 0), pixel(&img, 0, 0));
        assert_eq!(pixel(&out, 999, 599), pixel(&img, 999, 599));
        for (x, y) in [(1000, 0), (0, 600), (1007, 615), (1000, 599), (999, 600)] {
            assert_eq!(pixel(&out, x, y), PAD_BLACK, "{x},{y}");
        }
    }

    /// A resized image keeps the same corner: a uniform 450×300 image past the budget is resized to
    /// the 378×252 content of its 392×252 canvas, so column 377 is the image and column 378 pad.
    /// (A uniform image stays uniform under the bicubic filter's normalized weights.) The budget
    /// is 100,000 pixels so that the test's image is small.
    #[test]
    fn a_resized_image_lands_at_its_content_size_in_the_corner() {
        let rule = SizeRule::new(14, 2, 12_544, 100_000).unwrap();
        let canvas = rule.canvas(450, 300).unwrap();
        assert_eq!(
            (
                canvas.width,
                canvas.height,
                canvas.content_w,
                canvas.content_h
            ),
            (392, 252, 378, 252)
        );
        let out = canvas_image(&Rgb8::filled(450, 300, [200, 100, 50]), &canvas);
        assert_eq!(pixel(&out, 0, 0), [200, 100, 50]);
        assert_eq!(pixel(&out, 377, 251), [200, 100, 50]);
        assert_eq!(pixel(&out, 378, 0), PAD_BLACK);
        assert_eq!(pixel(&out, 391, 251), PAD_BLACK);
    }

    /// The patches of a 900×476 image on a 924×476 canvas: 66×34 patches of 1176 values in merge
    /// order, the plan's grid 33×17 tokens. The first patch starts at pixel (0, 0), the second at
    /// (14, 0), the third at (0, 14) (raster inside a merge group), and the last patch — columns
    /// 910..924, past the image — is CLIP's black in each channel: (0 - mean) / std, three different
    /// bf16 values, the two temporal frames alike.
    #[test]
    fn the_pad_normalizes_to_clips_black_in_merge_order() {
        let img = image(900, 476, bright);
        let p = preprocess(&img, &rule()).unwrap();
        assert_eq!((p.plan.best_w, p.plan.best_h), (924, 476));
        assert_eq!((p.plan.n_llm_w, p.plan.n_llm_h), (33, 17));
        assert_eq!((p.n_vit_w, p.n_vit_h, p.patch_len), (66, 34, 1176));
        assert_eq!(p.bf16.len(), 66 * 34 * 1176);
        let first = |patch: usize, c: usize| p.bf16[patch * 1176 + c * 196];
        for (patch, (x, y)) in [(0, (0, 0)), (1, (14, 0)), (2, (0, 14))] {
            for c in 0..3 {
                assert_eq!(
                    first(patch, c),
                    CLIP.normalize(c, bright(c, y, x)),
                    "{patch} {c}"
                );
            }
        }
        let black = [0xBFE5u16, 0xBFE0, 0xBFBD];
        for (c, &b) in black.iter().enumerate() {
            assert_eq!(CLIP.normalize(c, 0), b);
        }
        let last = &p.bf16[(66 * 34 - 1) * 1176..];
        let want: Vec<u16> = (0..2)
            .flat_map(|_| black.iter().flat_map(|&v| [v; 196]))
            .collect();
        assert_eq!(last, want.as_slice());
    }

    /// The marker ids are the text model's (`<|begin_of_image|>`, `<|image|>`, `<|end_of_image|>`).
    #[test]
    fn the_markers_are_the_text_models_ids() {
        assert_eq!(
            IMAGE_PLACEHOLDER,
            "<|begin_of_image|><|image|><|end_of_image|>"
        );
        assert_eq!(
            (BEGIN_OF_IMAGE_ID, IMAGE_TOKEN_ID, END_OF_IMAGE_ID),
            (154_830, 154_854, 154_831)
        );
    }

    /// The model's data is GLM's: the placeholder, the one id the server expands and llama-server's
    /// join rule, whose separator is a newline.
    #[test]
    fn the_media_data_is_glms() {
        let m = Media::new(rule());
        assert_eq!(m.image_placeholder(), IMAGE_PLACEHOLDER);
        assert_eq!(m.image_token(), IMAGE_TOKEN_ID);
        assert_eq!(m.part_join(), PartJoin::LlamaServer);
        assert_eq!(m.part_separator(), "\n");
    }

    /// The span an image takes is its canvas's merged grid: a 900×476 image is on a 924×476 canvas,
    /// 33×17 = 561 tokens; a 1000×600 image is on a 1008×616 canvas, 36×22 = 792 tokens. The
    /// plan carries the same grid and the canvas's pixel size.
    #[test]
    fn the_span_is_the_canvas_grid() {
        let rule = rule();
        let m = Media::new(rule);
        for ((w, h), (canvas, grid, span)) in [
            ((900, 476), ((924, 476), (33, 17), 561)),
            ((1000, 600), ((1008, 616), (36, 22), 792)),
        ] {
            let p = m.prepare(&image(w, h, bright)).expect("prepare");
            let c = rule.canvas(w, h).expect("canvas");
            assert_eq!((c.width, c.height), canvas, "{w}x{h}");
            assert_eq!(rule.grid(&c), grid, "{w}x{h}");
            assert_eq!(p.span_len, span, "{w}x{h}");
            assert_eq!(p.span_len, rule.grid(&c).0 * rule.grid(&c).1, "{w}x{h}");
            assert_eq!(
                (p.patches.plan.n_llm_w, p.patches.plan.n_llm_h),
                grid,
                "{w}x{h}"
            );
            assert_eq!(
                (p.patches.plan.best_w, p.patches.plan.best_h),
                canvas,
                "{w}x{h}"
            );
        }
    }
}
