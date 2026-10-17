//! Raster patches of the mtmd canvas.

use super::{IMAGE_BEGIN, IMAGE_END};
use crate::size::SizeRule;
use crate::{GridPlan, MediaModel, Patches, Prepared, Rgb8, VisionError};
use std::borrow::Cow;

/// Resize onto the mtmd canvas and cut raster patches after the graph's `2*x - 1`.
pub fn preprocess(image: &Rgb8, rule: &SizeRule) -> Result<Patches, VisionError> {
    image.check()?;
    let (best_w, best_h) = rule.aligned(image.width, image.height)?;
    let (nx, ny) = rule.grid(best_w, best_h);
    if nx.checked_mul(ny).is_none_or(|n| n > rule.max_tokens()) {
        return Err(VisionError::Limits {
            detail: "gemma4v aligned canvas exceeds the image token budget".into(),
        });
    }
    let plan = GridPlan {
        n_llm_h: ny,
        n_llm_w: nx,
        best_h,
        best_w,
    };
    let resized = crate::resample::pad_ceil(image, best_w, best_h, [0; 3])?;
    let patches = crate::preprocess::patchify(
        &resized,
        plan,
        crate::preprocess::PatchLayout::row_major(rule.patch()),
    )?;
    if patches.n_vit_h > 10240 || patches.n_vit_w > 10240 {
        return Err(VisionError::Limits {
            detail: "gemma4v canvas exceeds the learned position table".into(),
        });
    }
    Ok(patches)
}

/// Restore the original normalized bytes for the F32 convolution: their bf16 encodings are distinct.
pub fn conv_pixels(encoded: &[u16]) -> Result<Vec<f32>, VisionError> {
    encoded
        .iter()
        .map(|&bits| {
            let value = f32::from_bits(u32::from(bits) << 16);
            let byte = ((value + 1.0) * 127.5).round();
            if !value.is_finite()
                || !(0.0..=255.0).contains(&byte)
                || crate::preprocess::normalize(byte as u8) != bits
            {
                return Err(VisionError::Format(
                    "gemma4v convolution input is not a normalized RGB byte".into(),
                ));
            }
            Ok((byte / 255.0) * 2.0 - 1.0)
        })
        .collect()
}

/// Sample a zero-based constant-rate RGB clip at one frame per second.
/// Source FPS is a numerator and denominator. EOF is one source interval after the last frame.
/// Timestamps round to the nearest output tick, with half ticks rounded up.
/// The last source frame in each output tick wins; low-rate frames may repeat.
/// Clips that yield zero or more than 32 output frames are refused, not truncated.
pub fn prepare_clip(
    frames: &[Rgb8],
    source_fps: (u32, u32),
    patch: usize,
) -> Result<(Vec<usize>, Vec<Prepared>), VisionError> {
    let (num, den) = source_fps;
    if frames.is_empty() || num == 0 || den == 0 {
        return Err(VisionError::Limits {
            detail: "gemma4v clip needs decoded frames and positive source FPS".into(),
        });
    }
    let num = u128::from(num);
    let den = u128::from(den);
    let count = (2 * frames.len() as u128 * den + num) / (2 * num);
    if count == 0 || count > super::VIDEO_FRAMES as u128 {
        return Err(VisionError::Limits {
            detail: format!(
                "gemma4v clip samples to {count} frames; supported range is 1 to {}",
                super::VIDEO_FRAMES
            ),
        });
    }
    let indices = (0..count)
        .map(|tick| {
            let next = ((2 * tick + 1) * num).div_ceil(2 * den);
            (next - 1).min(frames.len() as u128 - 1) as usize
        })
        .collect::<Vec<_>>();
    let prepared = prepare_frames(indices.iter().map(|&i| &frames[i]), patch)?;
    Ok((indices, prepared))
}

/// Prepare already sampled RGB frames in input order, without timestamps or truncation.
pub fn prepare_video(frames: &[Rgb8], patch: usize) -> Result<Vec<Prepared>, VisionError> {
    prepare_frames(frames.iter(), patch)
}

fn prepare_frames<'a>(
    frames: impl ExactSizeIterator<Item = &'a Rgb8>,
    patch: usize,
) -> Result<Vec<Prepared>, VisionError> {
    if frames.len() == 0 || frames.len() > super::VIDEO_FRAMES {
        return Err(VisionError::Limits {
            detail: format!(
                "gemma4v video takes 1 to {} decoded frames",
                super::VIDEO_FRAMES
            ),
        });
    }
    let media = Media::new(SizeRule::new(
        patch,
        super::MERGE,
        crate::arch::TokenLimits::new(70, super::VIDEO_TOKENS)?,
    )?);
    frames.map(|frame| media.prepare(frame)).collect()
}

/// The image preparation rule; the side seat inserts delimiters through its tokenizer.
#[derive(Clone, Copy, Debug)]
pub struct Media {
    rule: SizeRule,
}

impl Media {
    /// An image preparer at the file's patch size and the request's token budget.
    #[must_use]
    pub fn new(rule: SizeRule) -> Self {
        Self { rule }
    }
}

impl MediaModel for Media {
    fn embedding_delimiters(&self) -> Option<(&str, &str)> {
        Some((IMAGE_BEGIN, IMAGE_END))
    }
    fn embedding_video(&self) -> Option<(&str, usize)> {
        Some(("Video:", super::VIDEO_FRAMES))
    }
    fn prepare_video(&self, frames: &[Rgb8]) -> Result<Vec<Prepared>, VisionError> {
        let prepared = prepare_video(frames, self.rule.patch())?;
        if prepared
            .iter()
            .any(|frame| frame.span_len > self.rule.max_tokens())
        {
            return Err(VisionError::Limits {
                detail: "video frame exceeds the loaded projector token capacity".into(),
            });
        }
        Ok(prepared)
    }
    fn image_placeholder(&self, _: usize) -> Cow<'_, str> {
        Cow::Owned(format!("{IMAGE_BEGIN}{IMAGE_END}"))
    }
    fn image_marker(&self) -> &str {
        IMAGE_BEGIN
    }
    fn image_token(&self) -> Option<u32> {
        None
    }
    fn prepare(&self, image: &Rgb8) -> Result<Prepared, VisionError> {
        let patches = preprocess(image, &self.rule)?;
        Ok(Prepared::of_grid(
            patches.plan.n_llm_h * patches.plan.n_llm_w,
            patches,
        ))
    }
}
