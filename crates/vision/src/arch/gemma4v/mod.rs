//! Gemma4's vision input: dynamic aspect-preserving canvases, raster patches,
//! learned x/y positions and three-by-three pooled rows.

pub mod card;
mod hparams;
mod media;
pub mod tensors;

pub use hparams::Hparams;
pub use media::{Media, VIDEO_RULE, conv_pixels, prepare_clip, prepare_video, preprocess};

/// The projector type of the vision half of a Gemma4 multimodal file.
pub const PROJECTOR_TYPE: &str = "gemma4v";
/// Patches pooled along each axis.
pub const MERGE: usize = 3;
/// The tokenizer spelling before an image's rows.
pub const IMAGE_BEGIN: &str = "<|image>";
/// The tokenizer spelling after an image's rows.
pub const IMAGE_END: &str = "<image|>";
/// The checkpoint's pooled-token budget for each decoded video frame.
pub const VIDEO_TOKENS: usize = 140;
/// The decoded-frame interface limit, independent of the reference decoder.
pub(crate) const VIDEO_FRAMES: usize = 32;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::size::SizeRule;
    use crate::{Rgb8, arch::TokenLimits};

    #[test]
    fn constant_rate_clips_sample_output_ticks_and_refuse_overflow() {
        for (seconds, fps, first) in [(1, 24, 11), (7, 30, 14), (16, 24, 11), (32, 30, 14)] {
            let frames = (0..seconds * fps)
                .map(|i| Rgb8::filled(48, 48, [i as u8, (i / 256) as u8, 0]))
                .collect::<Vec<_>>();
            let (indices, prepared) = prepare_clip(&frames, (fps as u32, 1), 16).unwrap();
            let expected = (0..seconds).map(|i| first + i * fps).collect::<Vec<_>>();
            assert_eq!(indices, expected);
            assert_eq!(prepared.len(), seconds);
            for (&index, p) in indices.iter().zip(&prepared) {
                let single = prepare_video(&[frames[index].clone()], 16).unwrap();
                assert_eq!(p.span_len, single[0].span_len);
                assert_eq!(p.grid().unwrap().bf16, single[0].grid().unwrap().bf16);
            }
        }
        let frame = Rgb8::filled(48, 48, [0; 3]);
        let long = vec![frame.clone(); 40 * 30];
        assert!(
            prepare_clip(&long, (30, 1), 16)
                .unwrap_err()
                .to_string()
                .contains("40 frames")
        );
        assert!(prepare_clip(&long[..33 * 30], (30, 1), 16).is_err());
        assert!(prepare_clip(&[], (24, 1), 16).is_err());
        assert!(prepare_clip(&long, (0, 1), 16).is_err());
        assert!(prepare_clip(&long, (24, 0), 16).is_err());
        assert!(prepare_clip(&long[..11], (24, 1), 16).is_err());
        assert_eq!(prepare_clip(&long[..12], (24, 1), 16).unwrap().0, [11]);
        assert_eq!(prepare_clip(&long[..13], (25, 1), 16).unwrap().0, [12]);
        assert_eq!(prepare_clip(&[frame], (1, 2), 16).unwrap().0, [0, 0]);
        let mut mixed = vec![Rgb8::filled(64, 64, [0; 3]); 10];
        mixed[0] = Rgb8::filled(1000, 1, [0; 3]);
        let (indices, prepared) = prepare_clip(&mixed, (10, 1), 16).unwrap();
        assert_eq!(indices, [4]);
        let rule = SizeRule::new(16, MERGE, TokenLimits::new(70, VIDEO_TOKENS).unwrap()).unwrap();
        let (w, h) = rule.aligned(64, 64).unwrap();
        let (nx, ny) = rule.grid(w, h);
        assert_eq!(((w, h), nx * ny), ((432, 432), 81));
        assert_eq!(prepared.len(), 1);
        assert_eq!(prepared[0].span_len, nx * ny);
    }

    /// A clip's frames are sized to the cap the planner gives, not only to the 140-token rule: a
    /// 768x768 frame is 528x528 at 121 tokens under the rule and 480x480 at 100 under a cap of 100
    /// [derived: align 48, sqrt(768² / (100 · 48²)) = 1.6, 768 / 1.6 = 480]. Five of them fill a
    /// 500-token budget as five groups, where a refusal at 121 would have dropped one. A cap under
    /// the least canvas is a refusal that names the cap.
    #[test]
    fn the_clip_sizer_resizes_to_the_cap() {
        use crate::video::{ClipMeta, Sizer, plan};
        let rule = SizeRule::new(16, MERGE, TokenLimits::new(70, VIDEO_TOKENS).unwrap()).unwrap();
        let media = Media::new(rule);
        assert_eq!(media.group_size(768, 768, 140).unwrap(), ((528, 528), 121));
        assert_eq!(media.group_size(768, 768, 100).unwrap(), ((480, 480), 100));
        let refused = media.group_size(768, 768, 50).unwrap_err().to_string();
        assert!(refused.contains("over the 50 a group may"), "{refused}");
        let meta = ClipMeta {
            frames: 5,
            fps: (1, 1),
            width: 768,
            height: 768,
        };
        let planned = plan(&VIDEO_RULE, &meta, 500, &media).unwrap();
        assert_eq!(planned.groups.len(), 5);
        assert!(
            planned
                .groups
                .iter()
                .all(|g| (g.size, g.tokens) == ((480, 480), 100))
        );
    }

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
        for bits in 0..=u16::MAX {
            assert_eq!(conv_pixels(&[bits]).is_ok(), encoded.contains(&bits));
        }
    }
}
