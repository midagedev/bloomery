//! The one planner of a video clip: which frames of the source a model sees, in what groups, at what
//! pixel size and with what stamp times.
//!
//! A model declares a [`VideoRule`], numbers only, and its own [`Sizer`] (how it brings a frame to a
//! pixel size and counts the tokens that size takes). [`plan`] turns the rule, the clip's probe
//! ([`ClipMeta`]) and the token budget the request leaves into a [`ClipPlan`]: the frame indices to
//! decode, the groups they form (a temporal patch tower takes two frames a group), each group's
//! size and tokens, and the times the model's text may name. The model's own code holds its rule,
//! its sizer and its text layout, nothing else; the decoder and the server's media layer read the
//! plan and never re-derive a number from it.
//!
//! Arithmetic is exact over integers where the references' floats agree with it, and f64 where a
//! reference is numpy's `linspace(..).round()` (ties to even), so the indices equal the reference's.
//! A frame's time is its index over the clip's average rate.

use std::ops::Range;

use crate::VisionError;

/// How a rule picks the frames of a clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampler {
    /// `count` indices spread over the clip: `numpy.linspace(0, frames - 1, count).round()`, the
    /// count `floor(duration × fps)` held to the rule's frame bounds and to the clip's frames.
    Linspace,
    /// One frame a tick of the rule's rate, the count `round(duration × fps)` with halves up. A
    /// `centred` tick takes the last source frame before the tick's midpoint, one that is not the
    /// first frame at or after the tick's start.
    Ticks { centred: bool },
}

/// What a rule does with a clip that samples to more frames than it takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overflow {
    /// Spread the maximum count over the sampled frames, evenly.
    Subsample,
    /// Refuse the clip by name.
    Refuse,
}

/// What a rule does with the frames left over when a clip's count is not a whole number of groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tail {
    /// Fill the last group with its last frame again (and its time again).
    RepeatLast,
    /// Leave the extra frames out.
    Drop,
}

/// Which time the model's text names for a group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stamp {
    /// None: the text carries no time.
    None,
    /// The mean of the group's frame times.
    GroupMean,
    /// The group's first frame time.
    GroupFirst,
    /// Every frame its own time: the text reads [`ClipPlan::times`].
    Each,
}

/// A model's video numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoRule {
    /// Sampled frames per second of source time, as a ratio.
    pub fps: (u32, u32),
    /// The fewest and the most frames sampled.
    pub frames: (usize, usize),
    pub sampler: Sampler,
    pub overflow: Overflow,
    /// Frames in one group: 1, or 2 for a tower with a temporal patch.
    pub temporal: usize,
    pub tail: Tail,
    /// The most tokens one group takes.
    pub group_tokens: usize,
    /// The most tokens the clip takes, before the request's context bound.
    pub clip_tokens: usize,
    /// The positions a group's text costs beside its tokens: delimiters and stamp, an upper bound.
    pub overhead: usize,
    pub stamp: Stamp,
}

/// What a probe says about a clip. The frame count is known: a probe without one is refused where
/// it is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipMeta {
    pub frames: usize,
    /// The average frame rate as a ratio.
    pub fps: (u32, u32),
    pub width: usize,
    pub height: usize,
}

/// One group of a plan: a run of the plan's frames, handed to the encoder as one input.
#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    /// Into [`ClipPlan::indices`] and [`ClipPlan::times`].
    pub frames: Range<usize>,
    /// The time the rule's [`Stamp`] picks, `None` for `Stamp::None` and `Stamp::Each`.
    pub time: Option<f64>,
    /// The pixel size the group's frames are brought to.
    pub size: (usize, usize),
    /// The tokens the group takes.
    pub tokens: usize,
}

/// A clip's plan: what to decode and how the model sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipPlan {
    /// The source frame of each plan frame, non-decreasing; a frame may repeat (a short clip, a
    /// repeated tail).
    pub indices: Vec<usize>,
    /// The source time of each plan frame in seconds.
    pub times: Vec<f64>,
    pub groups: Vec<Group>,
}

impl ClipPlan {
    /// The tokens of every group.
    #[must_use]
    pub fn tokens(&self) -> usize {
        self.groups.iter().map(|g| g.tokens).sum()
    }

    /// The source frames to decode: the plan's indices, once each, ascending.
    #[must_use]
    pub fn decode_indices(&self) -> Vec<usize> {
        let mut at = self.indices.clone();
        at.dedup();
        at
    }
}

/// A model's own resize rule, numbers only.
pub trait Sizer {
    /// The pixel size `(width, height)` one group's frames are brought to and the tokens that size
    /// takes, for a clip of `width`×`height` frames cut into `groups` groups, each taking at most
    /// `cap` tokens. The refusal names why even the model's smallest size is over `cap`.
    fn group_size(
        &self,
        groups: usize,
        width: usize,
        height: usize,
        cap: usize,
    ) -> Result<((usize, usize), usize), VisionError>;
}

/// The sizer of a projector that takes no video: every size is refused by name.
pub(crate) struct NoVideo;

impl Sizer for NoVideo {
    fn group_size(
        &self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<((usize, usize), usize), VisionError> {
        Err(refuse("the projector takes no video"))
    }
}

fn refuse(detail: impl Into<String>) -> VisionError {
    VisionError::Video(detail.into())
}

/// A refusal's own words, for a refusal that wraps it.
fn words(e: &VisionError) -> String {
    match e {
        VisionError::Video(detail) => detail.clone(),
        other => other.to_string(),
    }
}

/// The indices `numpy.linspace(0, len - 1, count).round()` gives: the step in f64, each index
/// `k × step` rounded with ties to even, the last the end.
fn linspace(len: usize, count: usize) -> Vec<usize> {
    if count <= 1 {
        return vec![0; count];
    }
    let step = (len - 1) as f64 / (count - 1) as f64;
    let mut at: Vec<usize> = (0..count)
        .map(|k| (k as f64 * step).round_ties_even() as usize)
        .collect();
    at[count - 1] = len - 1;
    at
}

/// The sampled frames before the fit: the count the rule takes of the clip, and how to pick fewer.
struct Sampled {
    /// `None`: every frame of the clip is a candidate. `Some`: the ticks are the candidates.
    ticks: Option<Vec<usize>>,
    frames: usize,
    count: usize,
}

impl Sampled {
    /// `count` indices spread over the candidates.
    fn pick(&self, count: usize) -> Vec<usize> {
        match &self.ticks {
            None => linspace(self.frames, count),
            Some(ticks) => linspace(ticks.len(), count)
                .into_iter()
                .map(|k| ticks[k])
                .collect(),
        }
    }
}

fn check(rule: &VideoRule, meta: &ClipMeta) -> Result<(), VisionError> {
    let (num, den) = rule.fps;
    if num == 0 || den == 0 || rule.frames.0 == 0 || rule.frames.0 > rule.frames.1 {
        return Err(refuse(format!(
            "the rule's rate {num}/{den} and frame bounds {}..{} are not a rule",
            rule.frames.0, rule.frames.1
        )));
    }
    if rule.temporal == 0 || rule.group_tokens == 0 {
        return Err(refuse(
            "the rule takes groups of at least one frame and token",
        ));
    }
    let (num, den) = meta.fps;
    if meta.frames == 0 || num == 0 || den == 0 || meta.width == 0 || meta.height == 0 {
        return Err(refuse(format!(
            "the clip has {} frames at {num}/{den} and {}x{} pixels; each must be positive",
            meta.frames, meta.width, meta.height
        )));
    }
    Ok(())
}

fn sample(rule: &VideoRule, meta: &ClipMeta) -> Result<Sampled, VisionError> {
    let (min, max) = rule.frames;
    let (a, b) = (u128::from(rule.fps.0), u128::from(rule.fps.1));
    let (num, den) = (u128::from(meta.fps.0), u128::from(meta.fps.1));
    let frames = meta.frames as u128;
    let too_many = |count: u128| {
        refuse(format!(
            "the clip samples to {count} frames; the rule takes {min} to {max}"
        ))
    };
    match rule.sampler {
        Sampler::Linspace => {
            let count = frames * den * a / (num * b);
            if count > max as u128 && rule.overflow == Overflow::Refuse {
                return Err(too_many(count));
            }
            let count = (count.max(min as u128).min(max as u128)).min(frames) as usize;
            Ok(Sampled {
                ticks: None,
                frames: meta.frames,
                count,
            })
        }
        Sampler::Ticks { centred } => {
            let count = (2 * frames * den * a + num * b) / (2 * num * b);
            if count < min as u128 {
                return Err(refuse(format!(
                    "the clip samples to {count} frames; the rule takes at least {min}"
                )));
            }
            if count > max as u128 && rule.overflow == Overflow::Refuse {
                return Err(too_many(count));
            }
            let last = frames - 1;
            let ticks: Vec<usize> = (0..count)
                .map(|t| {
                    let at = if centred {
                        ((2 * t + 1) * num * b).div_ceil(2 * den * a) - 1
                    } else {
                        t * b * num / (a * den)
                    };
                    at.min(last) as usize
                })
                .collect();
            let count = (count as usize).min(max);
            Ok(Sampled {
                ticks: Some(ticks),
                frames: meta.frames,
                count,
            })
        }
    }
}

/// Plan a clip: the frames and groups the rule takes of it, each group's size under the smaller of
/// the rule's clip tokens and `budget` (the tokens the request has left, groups' text included).
/// The size shrinks first, to the least the model's sizer takes; then the frames, evenly spread,
/// down to the rule's minimum; past that the clip is refused by name.
pub fn plan(
    rule: &VideoRule,
    meta: &ClipMeta,
    budget: usize,
    sizer: &dyn Sizer,
) -> Result<ClipPlan, VisionError> {
    check(rule, meta)?;
    let total = rule.clip_tokens.min(budget);
    let sampled = sample(rule, meta)?;
    let temporal = rule.temporal;
    let full = match rule.tail {
        Tail::RepeatLast => sampled.count.div_ceil(temporal),
        Tail::Drop => sampled.count / temporal,
    };
    if full == 0 {
        return Err(refuse(format!(
            "the clip samples to {} frames, fewer than a group of {temporal}",
            sampled.count
        )));
    }
    let fits = |groups: usize| {
        let cap = (total / groups)
            .saturating_sub(rule.overhead)
            .min(rule.group_tokens);
        if cap == 0 {
            return Err(refuse(format!(
                "the budget of {total} tokens leaves no room for {groups} groups of text"
            )));
        }
        let (size, tokens) = sizer.group_size(groups, meta.width, meta.height, cap)?;
        if tokens == 0 || tokens > cap {
            return Err(refuse(format!(
                "the model sized a group at {tokens} tokens under a cap of {cap}"
            )));
        }
        Ok((size, tokens))
    };
    let least = rule.frames.0.div_ceil(temporal).clamp(1, full);
    let (groups, size, tokens) = match fits(full) {
        Ok((size, tokens)) => (full, size, tokens),
        Err(_) if least < full => {
            fits(least).map_err(|e| {
                refuse(format!(
                    "the clip does not fit {total} tokens at its least {least} groups: {}",
                    words(&e)
                ))
            })?;
            // The largest group count that fits: `lo` fits, `hi` does not.
            let (mut lo, mut hi) = (least, full);
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if fits(mid).is_ok() {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let (size, tokens) = fits(lo)?;
            (lo, size, tokens)
        }
        Err(e) => {
            return Err(refuse(format!(
                "the clip does not fit {total} tokens at its least {least} groups: {}",
                words(&e)
            )));
        }
    };
    let frames = if groups == full {
        sampled.count
    } else {
        groups * temporal
    };
    let mut indices = if groups == full {
        sampled.pick(sampled.count)
    } else {
        sampled.pick(frames)
    };
    indices.truncate(groups * temporal);
    while indices.len() < groups * temporal {
        indices.push(*indices.last().expect("a plan has a frame"));
    }
    let (num, den) = meta.fps;
    let times: Vec<f64> = indices
        .iter()
        .enumerate()
        .map(|(k, &i)| {
            // A repeated tail frame keeps the time of the frame it repeats.
            let real = if k < frames { i } else { indices[frames - 1] };
            real as f64 * f64::from(den) / f64::from(num)
        })
        .collect();
    let groups = (0..groups)
        .map(|g| {
            let run = g * temporal..(g + 1) * temporal;
            let at = &times[run.clone()];
            let time = match rule.stamp {
                Stamp::None | Stamp::Each => None,
                Stamp::GroupFirst => Some(at[0]),
                Stamp::GroupMean => Some(at.iter().sum::<f64>() / at.len() as f64),
            };
            Group {
                frames: run,
                time,
                size,
                tokens,
            }
        })
        .collect();
    Ok(ClipPlan {
        indices,
        times,
        groups,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model that takes `per_px` pixels a token and a floor of `floor` tokens: its size is the
    /// clip's own, squeezed to the cap by the aspect-preserving integer square root.
    struct Grid {
        per_px: usize,
        floor: usize,
    }

    impl Sizer for Grid {
        fn group_size(
            &self,
            _groups: usize,
            w: usize,
            h: usize,
            cap: usize,
        ) -> Result<((usize, usize), usize), VisionError> {
            let tokens = (w * h).div_ceil(self.per_px);
            if tokens <= cap {
                return Ok(((w, h), tokens));
            }
            if cap < self.floor {
                return Err(refuse(format!(
                    "cap {cap} is under the floor {}",
                    self.floor
                )));
            }
            let side = ((cap * self.per_px) as f64).sqrt() as usize;
            let (w, h) = (side * w / w.max(h), side * h / w.max(h));
            Ok(((w, h), (w * h).div_ceil(self.per_px)))
        }
    }

    const QWEN_LIKE: VideoRule = VideoRule {
        fps: (2, 1),
        frames: (4, 768),
        sampler: Sampler::Linspace,
        overflow: Overflow::Subsample,
        temporal: 2,
        tail: Tail::RepeatLast,
        group_tokens: 768,
        clip_tokens: 12_288,
        overhead: 9,
        stamp: Stamp::GroupMean,
    };

    /// One centred tick a second, refusing a clip of more than 32 ticks: the tick sampler's own
    /// rule, with the group numbers of [`QWEN_LIKE`]. The planner's tests hold it here, not by a
    /// model's rule, so a model's retuning moves its own tests and not these.
    const TICKS: VideoRule = VideoRule {
        fps: (1, 1),
        frames: (1, 32),
        sampler: Sampler::Ticks { centred: true },
        overflow: Overflow::Refuse,
        temporal: 1,
        ..QWEN_LIKE
    };

    fn clip(frames: usize, fps: (u32, u32)) -> ClipMeta {
        ClipMeta {
            frames,
            fps,
            width: 64,
            height: 48,
        }
    }

    /// The indices numpy's `linspace(0, len - 1, count).round()` gave (numpy 2.5.1 on the machine
    /// that wrote this test), among them the ties that round to even.
    #[test]
    fn linspace_is_numpys() {
        for (len, count, want) in [
            (100, 7, vec![0, 16, 33, 50, 66, 82, 99]),
            (60, 4, vec![0, 20, 39, 59]),
            (31, 12, vec![0, 3, 5, 8, 11, 14, 16, 19, 22, 25, 27, 30]),
            (6, 3, vec![0, 2, 5]),
            (8, 3, vec![0, 4, 7]),
            (11, 4, vec![0, 3, 7, 10]),
            (12, 5, vec![0, 3, 6, 8, 11]),
            (1000, 7, vec![0, 166, 333, 500, 666, 832, 999]),
            (3, 3, vec![0, 1, 2]),
            (2, 2, vec![0, 1]),
            (1, 1, vec![0]),
            (10, 1, vec![0]),
        ] {
            assert_eq!(linspace(len, count), want, "{count} of {len}");
        }
        assert_eq!(linspace(1800, 120)[..7], [0, 15, 30, 45, 60, 76, 91]);
        assert_eq!(linspace(1800, 120)[119], 1799);
    }

    /// A 60 s clip at 30 fps under a rate of 2: 120 frames spread over 1,800, in 60 pairs, each
    /// pair stamped with the mean of its two times.
    #[test]
    fn linspace_clip_counts_indices_groups_and_stamps() {
        let sizer = Grid {
            per_px: 16,
            floor: 1,
        };
        let plan = plan(&QWEN_LIKE, &clip(1800, (30, 1)), usize::MAX, &sizer).unwrap();
        assert_eq!(plan.indices.len(), 120);
        assert_eq!(plan.indices[..7], [0, 15, 30, 45, 60, 76, 91]);
        assert_eq!(plan.groups.len(), 60);
        assert_eq!(plan.groups[0].frames, 0..2);
        assert_eq!(plan.groups[59].frames, 118..120);
        // Frame 15 of a 30 fps clip is at 0.5 s: the first pair's mean is 0.25 s.
        assert_eq!(plan.groups[0].time, Some(0.25));
        assert_eq!(plan.times[1], 0.5);
        assert_eq!(plan.groups[1].time, Some((1.0 + 1.5) / 2.0));
        // 64x48 is 192 tokens a group at 16 pixels a token; every group the same.
        assert!(
            plan.groups
                .iter()
                .all(|g| g.size == (64, 48) && g.tokens == 192)
        );
        assert_eq!(plan.tokens(), 60 * 192);
        assert_eq!(plan.decode_indices().len(), 120);
    }

    /// The count is `floor(duration × fps)` held to `[min, max]` and to the clip's frames, as HF's
    /// `sample_frames`: a short clip takes its frames, an odd count repeats its last frame, and
    /// the repeated frame keeps the time of the frame it repeats.
    #[test]
    fn linspace_count_bounds_and_tail() {
        let sizer = Grid {
            per_px: 16,
            floor: 1,
        };
        // 3 frames at 24 fps: floor(3/24 × 2) = 0, raised to 4, held to the 3 frames.
        let p = plan(&QWEN_LIKE, &clip(3, (24, 1)), usize::MAX, &sizer).unwrap();
        assert_eq!(p.indices, [0, 1, 2, 2]);
        assert_eq!(p.groups.len(), 2);
        assert_eq!(p.times[3], p.times[2]);
        assert_eq!(p.decode_indices(), [0, 1, 2]);
        // 6 s at 1 fps: 12 frames over 6 source frames is held to the 6.
        let p = plan(&QWEN_LIKE, &clip(6, (1, 1)), usize::MAX, &sizer).unwrap();
        assert_eq!(p.indices, [0, 1, 2, 3, 4, 5]);
        // The tail dropped: 7 frames at rate 1 sample 7, a group of two takes 3 pairs.
        let drop = VideoRule {
            tail: Tail::Drop,
            fps: (1, 1),
            ..QWEN_LIKE
        };
        let p = plan(&drop, &clip(7, (1, 1)), usize::MAX, &sizer).unwrap();
        assert_eq!((p.indices.len(), p.groups.len()), (6, 3));
        // Fewer frames than one group is a named refusal.
        let one = VideoRule {
            frames: (1, 768),
            tail: Tail::Drop,
            fps: (1, 100),
            ..QWEN_LIKE
        };
        let e = plan(&one, &clip(10, (1, 1)), usize::MAX, &sizer).unwrap_err();
        assert!(e.to_string().contains("fewer than a group of 2"), "{e}");
    }

    /// More frames than the rule takes: subsampled to the maximum, or refused by name.
    #[test]
    fn overflow_subsamples_or_refuses() {
        let sizer = Grid {
            per_px: 16,
            floor: 1,
        };
        let small = VideoRule {
            frames: (2, 10),
            ..QWEN_LIKE
        };
        let p = plan(&small, &clip(1800, (30, 1)), usize::MAX, &sizer).unwrap();
        assert_eq!(p.indices.len(), 10);
        assert_eq!(p.indices, linspace(1800, 10));
        let refuse = VideoRule {
            overflow: Overflow::Refuse,
            ..small
        };
        let e = plan(&refuse, &clip(1800, (30, 1)), usize::MAX, &sizer).unwrap_err();
        assert_eq!(
            e.to_string(),
            "video: the clip samples to 120 frames; the rule takes 2 to 10"
        );
        // Ticks subsample from the tick list.
        let ticks = VideoRule {
            frames: (1, 4),
            overflow: Overflow::Subsample,
            ..TICKS
        };
        let p = plan(&ticks, &clip(10 * 24, (24, 1)), usize::MAX, &sizer).unwrap();
        let all: Vec<usize> = (0..10).map(|t| 24 * t + 11).collect();
        assert_eq!(
            p.indices,
            linspace(10, 4).iter().map(|&k| all[k]).collect::<Vec<_>>()
        );
    }

    /// Ticks: the centred tick of a 24 fps clip at one a second is the frame before the second's
    /// midpoint (11, 35, …); an uncentred one is the second's first frame (0, 24, …); an index
    /// never passes the last frame; a clip under half a tick, or past the rule's maximum under
    /// `Refuse`, is refused by name.
    #[test]
    fn ticks_centred_and_not() {
        let sizer = Grid {
            per_px: 16,
            floor: 1,
        };
        let p = plan(&TICKS, &clip(7 * 24, (24, 1)), usize::MAX, &sizer).unwrap();
        assert_eq!(p.indices, (0..7).map(|t| 24 * t + 11).collect::<Vec<_>>());
        let plain = VideoRule {
            sampler: Sampler::Ticks { centred: false },
            ..TICKS
        };
        let p = plan(&plain, &clip(7 * 24, (24, 1)), usize::MAX, &sizer).unwrap();
        assert_eq!(p.indices, (0..7).map(|t| 24 * t).collect::<Vec<_>>());
        // One frame at half a frame a second: two ticks, both the one frame.
        let p = plan(&TICKS, &clip(1, (1, 2)), usize::MAX, &sizer).unwrap();
        assert_eq!(p.indices, [0, 0]);
        assert_eq!(p.decode_indices(), [0]);
        let e = plan(&TICKS, &clip(11, (24, 1)), usize::MAX, &sizer).unwrap_err();
        assert_eq!(
            e.to_string(),
            "video: the clip samples to 0 frames; the rule takes at least 1"
        );
        let e = plan(&TICKS, &clip(40 * 30, (30, 1)), usize::MAX, &sizer).unwrap_err();
        assert!(e.to_string().contains("40 frames"), "{e}");
    }

    /// The stamps: none, the group's mean, its first frame, and every frame's own time.
    #[test]
    fn stamp_times() {
        let sizer = Grid {
            per_px: 16,
            floor: 1,
        };
        let meta = clip(8, (2, 1));
        for (stamp, want) in [
            (Stamp::None, None),
            (Stamp::GroupFirst, Some(0.0)),
            (Stamp::GroupMean, Some(0.25)),
            (Stamp::Each, None),
        ] {
            let rule = VideoRule {
                stamp,
                frames: (2, 8),
                ..QWEN_LIKE
            };
            let p = plan(&rule, &meta, usize::MAX, &sizer).unwrap();
            assert_eq!(p.groups[0].time, want, "{stamp:?}");
            assert_eq!(p.times[..2], [0.0, 0.5], "{stamp:?}");
        }
        // 30000/1001 is 29.97 fps: frame 30 is at 30 × 1001/30000 s.
        let rule = VideoRule {
            frames: (1, 8),
            temporal: 1,
            stamp: Stamp::GroupFirst,
            ..QWEN_LIKE
        };
        let p = plan(&rule, &clip(900, (30_000, 1001)), usize::MAX, &sizer).unwrap();
        let k = p.indices.iter().position(|&i| i > 0).unwrap();
        assert_eq!(
            p.groups[k].time,
            Some(p.indices[k] as f64 * 1001.0 / 30_000.0)
        );
    }

    /// The shrink order: the size first (the sizer is asked for less), the frames only when the
    /// size has reached the model's floor, and a clip that does not fit at its least frames is a
    /// named refusal carrying the sizer's reason.
    #[test]
    fn the_budget_shrinks_size_before_frames() {
        let sizer = Grid {
            per_px: 16,
            floor: 40,
        };
        let rule = VideoRule {
            overhead: 0,
            ..QWEN_LIKE
        };
        let meta = clip(1800, (30, 1));
        // 60 groups × 192 tokens = 11,520: under the 12,288 cap, whole.
        let p = plan(&rule, &meta, usize::MAX, &sizer).unwrap();
        assert_eq!((p.groups.len(), p.groups[0].size), (60, (64, 48)));
        // A budget of 6,000: 100 tokens a group, so the size shrinks and every group stays.
        let p = plan(&rule, &meta, 6_000, &sizer).unwrap();
        assert_eq!(p.groups.len(), 60);
        assert!(p.groups[0].tokens <= 100 && p.groups[0].size.0 < 64);
        assert!(p.tokens() <= 6_000);
        // A budget of 1,200: the floor is 40 tokens a group, so 30 groups fit and no more.
        let p = plan(&rule, &meta, 1_200, &sizer).unwrap();
        assert_eq!(p.groups.len(), 30);
        assert_eq!(p.indices.len(), 60);
        assert_eq!(p.indices, linspace(1800, 60));
        assert!(p.tokens() <= 1_200);
        // Under four frames' worth (two groups at the floor, 80 tokens): refused.
        let e = plan(&rule, &meta, 79, &sizer).unwrap_err().to_string();
        assert!(
            e.contains("does not fit 79 tokens at its least 2 groups"),
            "{e}"
        );
        assert!(e.contains("under the floor"), "{e}");
        // The group's text is in the budget: 9 positions a group leave less than the cap.
        let with = plan(&QWEN_LIKE, &meta, 6_000, &sizer).unwrap();
        assert!(with.tokens() + 9 * with.groups.len() <= 6_000);
    }

    /// A rule or a probe that is not one is refused before anything is sampled.
    #[test]
    fn nonsense_is_refused_by_name() {
        let sizer = Grid {
            per_px: 16,
            floor: 1,
        };
        for (rule, meta) in [
            (
                VideoRule {
                    fps: (0, 1),
                    ..QWEN_LIKE
                },
                clip(10, (24, 1)),
            ),
            (
                VideoRule {
                    frames: (0, 4),
                    ..QWEN_LIKE
                },
                clip(10, (24, 1)),
            ),
            (
                VideoRule {
                    frames: (5, 4),
                    ..QWEN_LIKE
                },
                clip(10, (24, 1)),
            ),
            (
                VideoRule {
                    temporal: 0,
                    ..QWEN_LIKE
                },
                clip(10, (24, 1)),
            ),
            (QWEN_LIKE, clip(0, (24, 1))),
            (QWEN_LIKE, clip(10, (24, 0))),
            (
                QWEN_LIKE,
                ClipMeta {
                    width: 0,
                    ..clip(10, (24, 1))
                },
            ),
        ] {
            let e = plan(&rule, &meta, usize::MAX, &sizer).unwrap_err();
            assert!(matches!(e, VisionError::Video(_)), "{rule:?} {meta:?}: {e}");
        }
        let e = plan(&QWEN_LIKE, &clip(10, (24, 1)), 0, &sizer).unwrap_err();
        assert!(e.to_string().contains("no room"), "{e}");
    }
}
