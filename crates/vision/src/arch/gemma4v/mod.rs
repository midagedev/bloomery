//! Gemma4's vision input: dynamic aspect-preserving canvases, raster patches,
//! learned x/y positions and three-by-three pooled rows.

pub mod card;
mod hparams;
mod media;
pub mod tensors;

pub use hparams::Hparams;
pub use media::{Media, conv_pixels, preprocess};

/// The projector type of the vision half of a Gemma4 multimodal file.
pub const PROJECTOR_TYPE: &str = "gemma4v";
/// Patches pooled along each axis.
pub const MERGE: usize = 3;
/// The tokenizer spelling before an image's rows.
pub const IMAGE_BEGIN: &str = "<|image>";
/// The tokenizer spelling after an image's rows.
pub const IMAGE_END: &str = "<image|>";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::size::SizeRule;
    use crate::{Rgb8, arch::TokenLimits};

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
