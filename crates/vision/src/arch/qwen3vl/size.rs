//! The pixel size an image is brought to before it is cut into patches: llama.cpp's mtmd rule
//! for this projector (`img_tool::calc_size_preserved_ratio`, the dyn-size preprocessor), ported
//! operation for operation.
//!
//! Each side is rounded to a multiple of `align = patch * merge`; a size over the pixel limit is
//! scaled down by `beta = sqrt(h * w / max)` and floored to the multiple, a size under the other
//! limit is scaled up by `beta = sqrt(min / (h * w))` and ceiled. The arithmetic is f32 where the
//! C++ is (a `float` cast of an `int` is `as f32`, `std::round` rounds half away from zero, which
//! is `f32::round`), and the integers are widened to i64: llama.cpp's 32-bit pixel product is
//! undefined past its range, and this one is exact. The limits are in tokens, one token being
//! `align²` pixels.

use crate::VisionError;

/// The default `--image-min-tokens` of mtmd for this projector.
const MIN_TOKENS: usize = 8;
/// The default `--image-max-tokens`.
const MAX_TOKENS: usize = 4096;
/// The largest side mtmd resizes to: it refuses a resize target past it (an image already at its
/// planned size is copied there). A plan past it is refused here for every image.
const MAX_SIDE: i64 = 65_536;

/// The bounds on how many tokens an image takes: at least `min`, at most `max`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenLimits {
    min: usize,
    max: usize,
}

impl TokenLimits {
    /// 8 and 4,096, mtmd's defaults for this projector.
    pub const DEFAULT: TokenLimits = TokenLimits {
        min: MIN_TOKENS,
        max: MAX_TOKENS,
    };

    /// The most tokens an image takes.
    #[must_use]
    pub const fn max_tokens(self) -> usize {
        self.max
    }

    /// `min` and `max` tokens; each at least 1 and `min` at most `max`.
    pub fn new(min: usize, max: usize) -> Result<TokenLimits, VisionError> {
        if min == 0 || max == 0 || min > max {
            return Err(VisionError::Limits {
                detail: format!("min {min} and max {max} tokens; need 1 <= min <= max"),
            });
        }
        Ok(TokenLimits { min, max })
    }
}

/// The parameters of the size rule: the patch and merge sides and the token limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeRule {
    patch: usize,
    merge: usize,
    limits: TokenLimits,
}

impl SizeRule {
    /// The rule for `patch`-pixel patches merged `merge`×`merge`, with every pixel count the
    /// limits make fitting llama.cpp's 32-bit counts.
    pub fn new(patch: usize, merge: usize, limits: TokenLimits) -> Result<SizeRule, VisionError> {
        let limit_err = |detail: String| VisionError::Limits { detail };
        if patch == 0 || merge == 0 {
            return Err(limit_err(format!(
                "patch {patch} and merge {merge}; each must be at least 1"
            )));
        }
        let Some(align) = patch.checked_mul(merge).map(|a| a as u64) else {
            return Err(limit_err(format!(
                "patch {patch} times merge {merge} overflows"
            )));
        };
        let pixels = u64::try_from(limits.max)
            .ok()
            .and_then(|t| t.checked_mul(align * align));
        if pixels.is_none_or(|p| p > i32::MAX as u64) {
            return Err(limit_err(format!(
                "{} tokens of {align}x{align} pixels do not fit a 32-bit pixel count",
                limits.max
            )));
        }
        Ok(SizeRule {
            patch,
            merge,
            limits,
        })
    }

    /// Pixels per patch side.
    #[must_use]
    pub fn patch(&self) -> usize {
        self.patch
    }

    /// Patches per merge group side.
    #[must_use]
    pub fn merge(&self) -> usize {
        self.merge
    }

    /// Pixels a token covers on a side: `patch * merge`.
    #[must_use]
    pub fn align(&self) -> usize {
        self.patch * self.merge
    }

    /// The pixel size `(w̄, h̄)` an image of `width`×`height` is brought to: multiples of
    /// [`SizeRule::align`], at least one, in the token limits up to the rule's one rounding pass.
    pub fn aligned(&self, width: usize, height: usize) -> Result<(usize, usize), VisionError> {
        let size_err = |detail: String| VisionError::Size {
            width,
            height,
            detail,
        };
        if width == 0 || height == 0 {
            return Err(size_err(
                "an image with no pixels has no size to plan".into(),
            ));
        }
        let (Ok(w), Ok(h)) = (i32::try_from(width), i32::try_from(height)) else {
            return Err(size_err(
                "a side past what llama.cpp's 32-bit sizes hold".into(),
            ));
        };
        let (w, h) = (i64::from(w), i64::from(h));
        let align = self.align() as i64;
        let f = align as f32;
        let min_pixels = (self.limits.min as i64) * align * align;
        let max_pixels = (self.limits.max as i64) * align * align;
        let round_by = |x: f32| (x / f).round() as i64 * align;
        let ceil_by = |x: f32| (x / f).ceil() as i64 * align;
        let floor_by = |x: f32| (x / f).floor() as i64 * align;

        let mut w_bar = align.max(round_by(w as f32));
        let mut h_bar = align.max(round_by(h as f32));
        if h_bar * w_bar > max_pixels {
            let beta = ((h as f32 * w as f32) / max_pixels as f32).sqrt();
            h_bar = align.max(floor_by(h as f32 / beta));
            w_bar = align.max(floor_by(w as f32 / beta));
        } else if h_bar * w_bar < min_pixels {
            let beta = (min_pixels as f32 / (h as f32 * w as f32)).sqrt();
            h_bar = ceil_by(h as f32 * beta);
            w_bar = ceil_by(w as f32 * beta);
        }
        if w_bar > MAX_SIDE || h_bar > MAX_SIDE {
            return Err(size_err(format!(
                "the planned size {w_bar}x{h_bar} has a side past {MAX_SIDE}, where mtmd refuses \
                 to resize"
            )));
        }
        // Both are at least `align`, and at most MAX_SIDE.
        Ok((w_bar as usize, h_bar as usize))
    }

    /// The merged-token grid `(nx, ny)` of an image planned at `w_bar`×`h_bar` pixels: one token
    /// per `align`×`align` block.
    #[must_use]
    pub fn grid(&self, w_bar: usize, h_bar: usize) -> (usize, usize) {
        (w_bar / self.align(), h_bar / self.align())
    }
}

#[cfg(test)]
mod tests {
    use super::{SizeRule, TokenLimits};
    use crate::VisionError;

    fn rule() -> SizeRule {
        SizeRule::new(16, 2, TokenLimits::DEFAULT).expect("rule")
    }

    /// The size plan on the worked examples, each computed by hand from the formula with
    /// `align = 32`, `min = 8 · 1024 = 8,192` and `max = 4096 · 1024 = 4,194,304` pixels
    /// (`(w, h) → (w̄, h̄)`, then the tokens `nx · ny`).
    #[test]
    fn worked_examples() {
        let r = rule();
        for ((w, h), want, tokens) in [
            // round(448/32) = 14 on both sides → 448x448 = 200,704, inside both limits.
            ((448, 448), (448, 448), 196),
            // round(96/32) = 3, round(64/32) = 2 → 96x64 = 6,144 < 8,192: β = sqrt(8192/6144) =
            // 1.1547; w̄ = ceil(96 · β / 32 = 3.46) = 4 → 128, h̄ = ceil(64 · β / 32 = 2.31) = 3 → 96.
            ((96, 64), (128, 96), 12),
            // round(1000/32 = 31.25) = 31 → 992, round(700/32 = 21.875) = 22 → 704; 698,368 is inside.
            ((1000, 700), (992, 704), 682),
            // round(777/32 = 24.28) = 24 → 768, round(513/32 = 16.03) = 16 → 512; 393,216 is inside.
            ((777, 513), (768, 512), 384),
            // round(2600/32 = 81.25) = 81 → 2592, round(1800/32 = 56.25) = 56 → 1792; 4,644,864 >
            // 4,194,304: β = sqrt(1800 · 2600 / 4,194,304) = 1.0563; w̄ = floor(2600 / β / 32 =
            // 76.92) = 76 → 2432, h̄ = floor(1800 / β / 32 = 53.25) = 53 → 1696.
            ((2600, 1800), (2432, 1696), 4028),
            // round(144/32 = 4.5) = 5, half away from zero (to even it would be 4) → 160x160 =
            // 25,600, inside.
            ((144, 144), (160, 160), 25),
            // round(1000/32) = 31 → 992; round(1/32 = 0.03) = 0, at least 32 → 32; 31,744 is inside.
            ((1000, 1), (992, 32), 31),
            // round(1/32) = 0, at least 32 → 32; round(8000/32) = 250 → 8000; 256,000 is inside.
            ((1, 8000), (32, 8000), 250),
            // round(2/32) = 0 and round(1/32) = 0, both at least 32 → 32x32 = 1,024 < 8,192:
            // β = sqrt(8192 / 2) = 64; w̄ = ceil(2 · 64 / 32 = 4) = 4 → 128, h̄ = ceil(1 · 64 / 32 = 2)
            // = 2 → 64.
            ((2, 1), (128, 64), 8),
            // round(700/32 = 21.875) = 22 → 704, round(1000/32 = 31.25) = 31 → 992; 698,368 is inside.
            ((700, 1000), (704, 992), 682),
        ] {
            let got = r.aligned(w, h).expect("plan");
            assert_eq!(got, want, "{w}x{h}");
            let (nx, ny) = r.grid(got.0, got.1);
            assert_eq!(nx * ny, tokens, "{w}x{h} tokens");
        }
    }

    /// 3377×2494 is where the arithmetic's width decides: w̄ = round(105.53) = 106 → 3392, h̄ =
    /// round(77.94) = 78 → 2496; 8,466,432 > 4,194,304, β = sqrt(2494 · 3377 / 4,194,304) =
    /// 1.4171, and 2494 / β / 32 is 55 to within one part in a million. In f32 the quotient rounds
    /// to 55.0 and h̄ = 55 · 32 = 1760; in f64 it is 54.99999 and h̄ would be 1728. w̄ =
    /// floor(3377 / β / 32 = 74.47) = 74 → 2368 either way.
    #[test]
    fn f32_arithmetic_decides_a_borderline() {
        assert_eq!(rule().aligned(3377, 2494).expect("plan"), (2368, 1760));
    }

    /// The limits are parameters: 448×448 under at least 1,024 tokens (min = 1,048,576 pixels) is
    /// β = sqrt(1,048,576 / 200,704) = 16/7 = 2.2857 and 448 · β / 32 = 32 → 1024x1024; under at
    /// most 100 tokens (max = 102,400) it is β = sqrt(200,704 / 102,400) = 1.4 and 448 / 1.4 / 32
    /// = 10 → 320x320.
    #[test]
    fn limits_move_the_plan() {
        let rule = |min, max| {
            SizeRule::new(16, 2, TokenLimits::new(min, max).expect("limits")).expect("rule")
        };
        assert_eq!(
            rule(1024, 4096).aligned(448, 448).expect("plan"),
            (1024, 1024)
        );
        assert_eq!(rule(8, 100).aligned(448, 448).expect("plan"), (320, 320));
    }

    /// Every plan is a positive multiple of the alignment, over a sweep of sizes.
    #[test]
    fn plans_are_multiples_of_the_alignment() {
        let r = rule();
        for w in (1..=5000).step_by(97) {
            for h in (1..=5000).step_by(89) {
                let (pw, ph) = r.aligned(w, h).expect("plan");
                assert!(
                    pw >= 32 && ph >= 32 && pw % 32 == 0 && ph % 32 == 0,
                    "{w}x{h}"
                );
            }
        }
    }

    /// An image with no pixels, a side past 32 bits, a plan past mtmd's resize limit, a limit
    /// pair out of order and a rule with a zero side are each refused by name.
    #[test]
    fn refusals() {
        let r = rule();
        for (w, h) in [(0, 5), (5, 0), (1usize << 31, 8)] {
            assert!(
                matches!(r.aligned(w, h), Err(VisionError::Size { .. })),
                "{w}x{h}"
            );
        }
        // 70,000 / 32 = 2187.5 rounds to 2188 → 70,016 > 65,536.
        let e = r
            .aligned(70_000, 1)
            .expect_err("past the resize limit")
            .to_string();
        assert!(
            e.starts_with("image size 70000x1: the planned size 70016x32 has a side past 65536"),
            "{e}"
        );
        for (min, max) in [(0, 4096), (8, 0), (100, 50)] {
            assert!(
                matches!(TokenLimits::new(min, max), Err(VisionError::Limits { .. })),
                "{min}, {max}"
            );
        }
        let big = TokenLimits::new(8, 3_000_000).expect("limits");
        let e = SizeRule::new(16, 2, big)
            .expect_err("pixel count")
            .to_string();
        assert!(
            e.starts_with("image size limits: 3000000 tokens of 32x32 pixels do not fit"),
            "{e}"
        );
        assert!(SizeRule::new(0, 2, TokenLimits::DEFAULT).is_err());
        assert!(SizeRule::new(16, 0, TokenLimits::DEFAULT).is_err());
        assert!(SizeRule::new(usize::MAX, 2, TokenLimits::DEFAULT).is_err());
    }
}
