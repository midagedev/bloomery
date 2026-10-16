//! Gemma4's vision input: dynamic aspect-preserving canvases, raster patches,
//! learned x/y positions and three-by-three pooled rows.

pub mod card;
mod hparams;
mod media;
pub mod tensors;

pub use hparams::Hparams;
pub use media::{Media, conv_pixels, prepare_video, preprocess};

/// The projector type of the vision half of a Gemma4 multimodal file.
pub const PROJECTOR_TYPE: &str = "gemma4v";
/// Patches pooled along each axis.
pub const MERGE: usize = 3;
/// The tokenizer spelling before an image's rows.
pub const IMAGE_BEGIN: &str = "<|image>";
/// The tokenizer spelling after an image's rows.
pub const IMAGE_END: &str = "<image|>";
/// The checkpoint's pooled-token budget for each decoded video frame.
pub(crate) const VIDEO_TOKENS: usize = 140;
/// The checkpoint's maximum number of decoded frames in one input.
pub(crate) const VIDEO_FRAMES: usize = 32;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::size::SizeRule;
    use crate::{Rgb8, arch::TokenLimits};

    #[test]
    fn decoded_video_preserves_frames_and_bounds_each_canvas() {
        let frames = [
            Rgb8::filled(777, 513, [255, 0, 0]),
            Rgb8::filled(200, 200, [0, 0, 255]),
        ];
        let prepared = prepare_video(&frames, 16).unwrap();
        assert_eq!(prepared.len(), frames.len());
        for (frame, prepared) in frames.iter().zip(&prepared) {
            let p = prepared.grid().unwrap();
            let rule =
                SizeRule::new(16, MERGE, TokenLimits::new(70, VIDEO_TOKENS).unwrap()).unwrap();
            let expected = preprocess(frame, &rule).unwrap();
            assert_eq!(
                (p.plan.best_w, p.plan.best_h),
                rule.aligned(frame.width, frame.height).unwrap()
            );
            assert_eq!(p.bf16, expected.bf16);
            assert_eq!(prepared.span_len, p.plan.n_llm_w * p.plan.n_llm_h);
            assert!(prepared.span_len <= 140);
        }
        let bound = vec![Rgb8::filled(1, 1, [0; 3]); 32];
        assert_eq!(prepare_video(&bound, 16).unwrap().len(), 32);
        assert!(prepare_video(&vec![bound[0].clone(); 33], 16).is_err());
        assert!(prepare_video(&[], 16).is_err());
        assert!(prepare_video(&[Rgb8::filled(0, 1, [0; 3])], 16).is_err());
        let error = prepare_video(&[Rgb8::filled(1000, 1, [0; 3])], 16).unwrap_err();
        assert!(error.to_string().contains("exceeds the image token budget"));
    }

    #[test]
    fn gemma4v_canvas_and_patch_order() {
        let rule = SizeRule::new(16, MERGE, TokenLimits::new(70, 1120).unwrap()).unwrap();
        for ((w, h), expected) in [
            ((224, 224), (432, 432)),
            ((1000, 700), (1008, 720)),
            ((2600, 1800), (1920, 1296)),
            ((1, 1), (432, 432)),
        ] {
            assert_eq!(rule.aligned(w, h).unwrap(), expected);
        }
        let small = SizeRule::new(16, MERGE, TokenLimits::new(1, 4).unwrap()).unwrap();
        let p = preprocess(&Rgb8::filled(96, 48, [0, 128, 255]), &small).unwrap();
        assert_eq!((p.n_vit_w, p.n_vit_h, p.plan.span_grid()), (6, 3, (2, 1)));
        assert_eq!(p.patch_len, 768);
        assert_eq!(p.bf16[0], crate::preprocess::f32_to_bf16(-1.0));
        assert_eq!(
            p.bf16[256],
            crate::preprocess::f32_to_bf16(2.0 * (128.0f32 / 255.0) - 1.0)
        );
        assert_eq!(p.bf16[512], crate::preprocess::f32_to_bf16(1.0));
        assert!(preprocess(&Rgb8::filled(0, 1, [0; 3]), &small).is_err());
        let encoded: Vec<_> = (0..=255).map(crate::preprocess::normalize).collect();
        let pixels = conv_pixels(&encoded).unwrap();
        for (byte, value) in pixels.into_iter().enumerate() {
            assert_eq!(
                value.to_bits(),
                (2.0 * (byte as f32 / 255.0) - 1.0).to_bits()
            );
        }
        for invalid in [0, 0x7fc0, 0x7f80, 0xff80] {
            assert!(conv_pixels(&[invalid]).is_err());
        }
    }
}
