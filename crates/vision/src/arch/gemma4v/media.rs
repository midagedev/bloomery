//! Raster patches of the mtmd canvas.

use super::{IMAGE_BEGIN, IMAGE_END, MERGE, VIDEO_FRAMES, VIDEO_TOKENS};
use crate::media::Piece;
use crate::size::SizeRule;
use crate::video::{ClipMeta, ClipPlan, Overflow, Sampler, Sizer, Stamp, Tail, VideoRule, sample};
use crate::{GridPlan, MediaModel, Patches, Prepared, Rgb8, VisionError};
use std::borrow::Cow;

/// The text before a clip's frames in an embedding input.
pub(super) const VIDEO_PREFIX: &str = "Video:";

/// Gemma4's video numbers: one frame a second, taken at the middle of each second, 1 to 32 of
/// them, each frame its own group of at most [`VIDEO_TOKENS`] tokens, no stamp in the text.
pub const VIDEO_RULE: VideoRule = VideoRule {
    fps: (1, 1),
    frames: (1, VIDEO_FRAMES),
    sampler: Sampler::Ticks { centred: true },
    overflow: Overflow::Refuse,
    temporal: 1,
    tail: Tail::RepeatLast,
    group_tokens: VIDEO_TOKENS,
    clip_tokens: VIDEO_TOKENS * VIDEO_FRAMES,
    overhead: 0,
    stamp: Stamp::None,
};

/// The fewest tokens a video frame's canvas takes.
const VIDEO_MIN_TOKENS: usize = 70;

/// The size rule of a video frame: 70 to [`VIDEO_TOKENS`] tokens at the file's patch size.
fn video_size_rule(patch: usize) -> Result<SizeRule, VisionError> {
    video_size_rule_within(patch, VIDEO_TOKENS)
}

/// [`video_size_rule`] with the canvas held to `most` tokens: the frame is resized to fit, not
/// refused for the size the full rule would give it. A `most` under the 70-token floor takes the
/// floor, and the canvas that comes out is refused by its tokens against the cap.
fn video_size_rule_within(patch: usize, most: usize) -> Result<SizeRule, VisionError> {
    SizeRule::new(
        patch,
        MERGE,
        crate::arch::TokenLimits::new(
            VIDEO_MIN_TOKENS,
            most.clamp(VIDEO_MIN_TOKENS, VIDEO_TOKENS),
        )?,
    )
}

/// Resize onto the mtmd canvas and cut raster patches after the graph's `2*x - 1`.
pub fn preprocess(image: &Rgb8, rule: &SizeRule) -> Result<Patches, VisionError> {
    image.check()?;
    let (best_w, best_h) = rule.aligned(image.width, image.height)?;
    cut(image, rule, best_w, best_h)
}

/// Resize onto the canvas `best_w`×`best_h` and cut its patches.
fn cut(
    image: &Rgb8,
    rule: &SizeRule,
    best_w: usize,
    best_h: usize,
) -> Result<Patches, VisionError> {
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
    // Every frame shares the same inverse of the fixed RGB byte normalization.
    static PIXELS: std::sync::LazyLock<Vec<f32>> = std::sync::LazyLock::new(|| {
        let mut values = vec![f32::NAN; usize::from(u16::MAX) + 1];
        for byte in 0..=255u8 {
            values[usize::from(crate::preprocess::normalize(byte))] =
                (f32::from(byte) / 255.0) * 2.0 - 1.0;
        }
        values
    });
    let pixels = &*PIXELS;
    encoded
        .iter()
        .map(|&bits| {
            let value = pixels[usize::from(bits)];
            if !value.is_finite() {
                return Err(VisionError::Format(
                    "gemma4v convolution input is not a normalized RGB byte".into(),
                ));
            }
            Ok(value)
        })
        .collect()
}

/// Sample a zero-based constant-rate RGB clip at one frame per second.
/// Source FPS is a numerator and denominator. EOF is one source interval after the last frame.
/// The frames are the planner's ([`VIDEO_RULE`]): each second's frame is the last source frame
/// before the second's midpoint, so low-rate clips may repeat a frame, and a clip that samples to
/// zero or more than 32 frames is refused, not truncated. Each sampled frame is sized by its own
/// shape.
pub fn prepare_clip(
    frames: &[Rgb8],
    source_fps: (u32, u32),
    patch: usize,
) -> Result<(Vec<usize>, Vec<Prepared>), VisionError> {
    let Some(first) = frames.first() else {
        return Err(VisionError::Limits {
            detail: "gemma4v clip needs decoded frames and positive source FPS".into(),
        });
    };
    let meta = ClipMeta {
        frames: frames.len(),
        fps: source_fps,
        width: first.width,
        height: first.height,
    };
    let sampled = sample(&VIDEO_RULE, &meta)?;
    let indices = sampled.pick(sampled.count);
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
    let media = Media::new(video_size_rule(patch)?);
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

/// The sizer of a clip's frames: the canvas of the file's patch size the video rule gives, resized
/// to the cap the planner allows a group and to the tokens the loaded tower takes; a cap under the
/// rule's least canvas is a refusal that names it.
impl Sizer for Media {
    fn group_size(
        &self,
        width: usize,
        height: usize,
        cap: usize,
    ) -> Result<((usize, usize), usize), VisionError> {
        let rule = video_size_rule_within(self.rule.patch(), cap.min(self.rule.max_tokens()))?;
        let (best_w, best_h) = rule.aligned(width, height)?;
        let (nx, ny) = rule.grid(best_w, best_h);
        let tokens = nx * ny;
        if tokens > VIDEO_TOKENS {
            return Err(VisionError::Limits {
                detail: "gemma4v aligned canvas exceeds the image token budget".into(),
            });
        }
        if tokens > self.rule.max_tokens() {
            return Err(VisionError::Limits {
                detail: "video frame exceeds the loaded projector token capacity".into(),
            });
        }
        if tokens > cap {
            return Err(VisionError::Video(format!(
                "a frame of {width}x{height} takes {tokens} tokens, over the {cap} a group may"
            )));
        }
        Ok(((best_w, best_h), tokens))
    }
}

impl MediaModel for Media {
    fn video_rule(&self) -> Option<&VideoRule> {
        Some(&VIDEO_RULE)
    }
    fn video_sizer(&self) -> &dyn Sizer {
        self
    }
    fn prepare_group(
        &self,
        frames: &[Rgb8],
        size: (usize, usize),
    ) -> Result<Prepared, VisionError> {
        let [frame] = frames else {
            return Err(VisionError::Video(format!(
                "a gemma4v group is one frame, not {}",
                frames.len()
            )));
        };
        frame.check()?;
        let rule = video_size_rule(self.rule.patch())?;
        let patches = cut(frame, &rule, size.0, size.1)?;
        Ok(Prepared::of_grid(
            patches.plan.n_llm_h * patches.plan.n_llm_w,
            patches,
        ))
    }
    fn video_text(&self, plan: &ClipPlan) -> Result<Vec<Piece>, VisionError> {
        Ok(std::iter::once(Piece::Text(VIDEO_PREFIX.into()))
            .chain((0..plan.groups.len()).map(Piece::Group))
            .collect())
    }
    fn embedding_delimiters(&self) -> Option<(&str, &str)> {
        Some((IMAGE_BEGIN, IMAGE_END))
    }
    fn embedding_video(&self) -> Option<(&str, usize)> {
        Some((VIDEO_PREFIX, VIDEO_FRAMES))
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
