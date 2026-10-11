//! The canvas an image is planned onto: llama.cpp's `mtmd_image_preprocessor_glm5v`
//! (`tools/mtmd/mtmd-image.cpp`), the rule of this projector.
//!
//! Each edge is aligned up to `patch * merge` (28) pixels. A canvas below the file's minimum pixel
//! count is scaled up to it; one above the maximum is found by a search over the content height,
//! the width following at the image's aspect. The image is scaled to the canvas and never
//! upscaled unless it is itself below the minimum, and it sits in the canvas's top-left corner.
//! The arithmetic is the reference's: `i64` for the counts and `f64` for the scale, each
//! operation in its order, because a canvas's content size is a `floor` of a product that rounds.

use crate::VisionError;

/// The size rule of one file: the patch and merge sides and the pixel budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeRule {
    patch: usize,
    merge: usize,
    factor: i64,
    min_pixels: i64,
    max_pixels: i64,
}

/// Where an image lands: the canvas, and the content size the image is resized to before it is
/// placed at pixel (0, 0). The sides of the canvas are multiples of the rule's factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub content_w: usize,
    pub content_h: usize,
}

fn limits(detail: String) -> VisionError {
    VisionError::Limits { detail }
}

impl SizeRule {
    /// The rule of a file with `patch`-pixel patches, `merge`×`merge` patches a token and a canvas
    /// of `min_pixels..=max_pixels` pixels. A canvas is at least one token, so a maximum below
    /// one token's pixels is refused, as is a minimum above the maximum.
    pub fn new(
        patch: usize,
        merge: usize,
        min_pixels: u64,
        max_pixels: u64,
    ) -> Result<SizeRule, VisionError> {
        let factor = patch
            .checked_mul(merge)
            .and_then(|f| i64::try_from(f).ok())
            .filter(|&f| f > 0)
            .ok_or_else(|| {
                limits(format!(
                    "patch {patch} and merge {merge} must be at least 1"
                ))
            })?;
        let (Ok(min_pixels), Ok(max_pixels)) =
            (i64::try_from(min_pixels), i64::try_from(max_pixels))
        else {
            return Err(limits(format!(
                "the pixel budget {min_pixels}..={max_pixels} does not fit i64"
            )));
        };
        if min_pixels < 1 || max_pixels < min_pixels || max_pixels < factor * factor {
            return Err(limits(format!(
                "the pixel budget {min_pixels}..={max_pixels} must be 1 or more, at least one \
                 token ({} pixels) wide and not empty",
                factor * factor
            )));
        }
        Ok(SizeRule {
            patch,
            merge,
            factor,
            min_pixels,
            max_pixels,
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

    /// Pixels per patch side times patches per merge side: the canvas's edge multiple.
    #[must_use]
    pub fn factor(&self) -> usize {
        self.patch * self.merge
    }

    /// The fewest merged tokens a canvas holds that the minimum pixel count asks for.
    #[must_use]
    pub fn min_tokens(&self) -> usize {
        (self.min_pixels / (self.factor * self.factor)) as usize
    }

    /// The most merged tokens a canvas holds: its area is a multiple of `factor²` within the
    /// maximum pixel count.
    #[must_use]
    pub fn max_tokens(&self) -> usize {
        (self.max_pixels / (self.factor * self.factor)) as usize
    }

    /// The canvas of an image of `width`×`height` pixels (`mtmd-image.cpp:801-857`).
    pub fn canvas(&self, width: usize, height: usize) -> Result<Canvas, VisionError> {
        let size_err = |detail: &str| VisionError::Size {
            width,
            height,
            detail: detail.to_string(),
        };
        if width == 0 || height == 0 {
            return Err(size_err("an empty image has no canvas"));
        }
        // mtmd's sides are C `int`; below this bound `height * width`, the aligned canvas's area
        // and the search's `width * ch` stay inside i64.
        let (Ok(w), Ok(h)) = (i64::try_from(width), i64::try_from(height)) else {
            return Err(size_err("a side does not fit i64"));
        };
        if w > i64::from(i32::MAX) || h > i64::from(i32::MAX) {
            return Err(size_err("a side is beyond 2^31 - 1 pixels"));
        }
        let (factor, min_px, max_px) = (self.factor, self.min_pixels, self.max_pixels);
        let align = |v: i64| (v + factor - 1) / factor * factor;

        // :813-814
        let (mut canvas_h, mut canvas_w) = (align(h), align(w));
        // :816-820 — below the minimum, scale the image's pixel count up to it.
        if canvas_h * canvas_w < min_px {
            let scale = (min_px as f64 / (h * w) as f64).sqrt();
            canvas_h = align(((h as f64 * scale).ceil() as i64).max(1));
            canvas_w = align(((w as f64 * scale).ceil() as i64).max(1));
        }
        // :822-841 — above the maximum, the largest content height whose aligned canvas fits.
        if canvas_h * canvas_w > max_px {
            let (mut lo, mut hi) = (1, h);
            let (mut best_h, mut best_w) = (factor, factor);
            while lo <= hi {
                let ch = (lo + hi) / 2;
                let cw = (w * ch / h).max(1);
                let (ah, aw) = (align(ch), align(cw));
                if ah * aw <= max_px {
                    (best_h, best_w) = (ah, aw);
                    lo = ch + 1;
                } else {
                    hi = ch - 1;
                }
            }
            (canvas_h, canvas_w) = (best_h, best_w);
        }
        // :844-849 — `std::min(a, b)` is `(b < a) ? b : a`; the cap at 1 applies from the minimum
        // up.
        let (sh, sw) = (canvas_h as f64 / h as f64, canvas_w as f64 / w as f64);
        let mut scale = if sw < sh { sw } else { sh };
        if h * w >= min_px && 1.0 < scale {
            scale = 1.0;
        }
        let content =
            |side: i64, canvas: i64| (canvas.min((side as f64 * scale).floor() as i64)).max(1);
        Ok(Canvas {
            width: canvas_w as usize,
            height: canvas_h as usize,
            content_w: content(w, canvas_w) as usize,
            content_h: content(h, canvas_h) as usize,
        })
    }

    /// The merged-token grid of a canvas, `(nx, ny)`: `nx` tokens a row, `ny` rows. The image takes
    /// `nx * ny` positions in the prompt.
    #[must_use]
    pub fn grid(&self, canvas: &Canvas) -> (usize, usize) {
        (canvas.width / self.factor(), canvas.height / self.factor())
    }
}

#[cfg(test)]
mod tests {
    use super::{Canvas, SizeRule};

    /// GLM-5.3-Flash's rule: 14-pixel patches, 2×2 merge, 12,544..=6,272,000 pixels.
    fn rule() -> SizeRule {
        SizeRule::new(14, 2, 12_544, 6_272_000).expect("rule")
    }

    /// The canvas of a table of image sizes `(width, height)`, as `(canvas w, canvas h, content w,
    /// content h)`. Each row is `mtmd-image.cpp:801-857` worked in f64 and i64; the first four are
    /// worked by hand here.
    ///
    /// * 100×100: align → 112×112 = 12,544, not `<` min (:816), not `>` max (:822). scale =
    ///   min(112/100, 112/100) = 1.12; 100·100 < min, so no cap at 1 (:845): content
    ///   floor(100·1.12) = 112.
    /// * 50×50: align → 56×56 = 3,136 < min. scale = sqrt(12544/2500) is 2.2399999999999998 in f64,
    ///   one ulp under 2.24; ceil(50·scale) = 112 (a ceil of the exact 2.24's product would be 113
    ///   and align to 140) → 112×112, content floor(50·(112/50)) = 112.
    /// * 20×100: align → 28×112 < min. scale = sqrt(12544/2000) = 2.504…; ceil(100·scale) = 251 →
    ///   252, ceil(20·scale) = 51 → 56: canvas 56×252. scale = min(2.52, 2.8) = 2.52, no cap
    ///   (2,000 < min): content_h = floor(252.0) = 252, content_w = floor(50.4) = 50 — a ceil
    ///   would give 51, inside the canvas of 56.
    /// * 4000×3000: align → 4004×3024 > max. The search over the content height `ch` ∈ [1, 3000]
    ///   keeps the largest whose aligned canvas fits: ch = 2156 gives cw = 4000·2156/3000 = 2874
    ///   (integer division), aligned 2156×2884 = 6,217,904 ≤ max, and any ch above 2156 aligns to
    ///   ≥ 2184 rows with cw ≥ 2876 → 2884 columns: 6,298,656 > max. scale = min(0.7186…, 0.721) =
    ///   0.7186…: content floor(3000·0.7186…) = 2156, floor(4000·0.7186…) = 2874.
    ///
    /// The others: 113×111 is one pixel short of the minimum (12,543), so it may upscale: scale =
    /// min(112/111, 140/113) and floor(111·scale) = floor(111.99999999999999) = 111 — one row
    /// short of the canvas. 2240×2800 is exactly max and takes no search.
    #[test]
    fn the_canvas_of_a_table_of_sizes() {
        let rows = [
            ((1, 1), (112, 112, 112, 112)),
            ((100, 100), (112, 112, 112, 112)),
            ((50, 50), (112, 112, 112, 112)),
            ((20, 100), (56, 252, 50, 252)),
            ((113, 111), (140, 112, 114, 111)),
            ((112, 112), (112, 112, 112, 112)),
            ((1000, 600), (1008, 616, 1000, 600)),
            ((1036, 756), (1036, 756, 1036, 756)),
            ((10, 4000), (28, 4004, 10, 4000)),
            ((2240, 2800), (2240, 2800, 2240, 2800)),
            ((4000, 3000), (2884, 2156, 2874, 2156)),
            ((3000, 4000), (2156, 2884, 2156, 2874)),
            ((8000, 2000), (4956, 1260, 4956, 1239)),
            ((2500, 2509), (2492, 2492, 2483, 2492)),
            ((1, 10_000_000), (28, 224_000, 1, 224_000)),
        ];
        let rule = rule();
        for ((w, h), (cw, ch, kw, kh)) in rows {
            let want = Canvas {
                width: cw,
                height: ch,
                content_w: kw,
                content_h: kh,
            };
            assert_eq!(rule.canvas(w, h).unwrap(), want, "{w}x{h}");
        }
    }

    /// The prompt's token grid is the canvas over the factor: 1036×756 is 37×27 = 999 tokens, the
    /// budget's floor is 16 tokens and its ceiling 8000.
    #[test]
    fn the_token_grid_and_budget() {
        let rule = rule();
        let c = rule.canvas(1036, 756).unwrap();
        assert_eq!(rule.grid(&c), (37, 27));
        assert_eq!((rule.min_tokens(), rule.max_tokens()), (16, 8000));
        assert_eq!(rule.canvas(1, 1).unwrap().width / rule.factor(), 4);
    }

    /// Over a sweep of sizes every canvas is a multiple of 28 on both sides, holds between 16 and
    /// 8000 tokens, and holds its content.
    #[test]
    fn every_canvas_is_aligned_inside_the_budget_and_holds_its_content() {
        let rule = rule();
        let sizes = [
            1, 2, 13, 27, 28, 29, 55, 100, 333, 640, 1024, 1920, 4097, 9000,
        ];
        for &w in &sizes {
            for &h in &sizes {
                let c = rule.canvas(w, h).unwrap();
                assert_eq!((c.width % 28, c.height % 28), (0, 0), "{w}x{h}");
                let tokens = c.width / 28 * (c.height / 28);
                assert!((16..=8000).contains(&tokens), "{w}x{h}: {tokens} tokens");
                assert!(c.content_w >= 1 && c.content_w <= c.width, "{w}x{h}");
                assert!(c.content_h >= 1 && c.content_h <= c.height, "{w}x{h}");
            }
        }
    }

    /// An image with no pixels or a side past `int` has no canvas; a rule with no room for one token,
    /// an empty budget or a zero side is refused by name.
    #[test]
    fn empty_images_and_empty_rules_are_refused() {
        let rule = rule();
        assert_eq!(
            rule.canvas(0, 10).unwrap_err().to_string(),
            "image size 0x10: an empty image has no canvas"
        );
        assert_eq!(
            rule.canvas(10, 0).unwrap_err().to_string(),
            "image size 10x0: an empty image has no canvas"
        );
        assert_eq!(
            rule.canvas(1 << 31, 10).unwrap_err().to_string(),
            "image size 2147483648x10: a side is beyond 2^31 - 1 pixels"
        );
        for (patch, merge, min, max) in [
            (0, 2, 12_544, 6_272_000),
            (14, 0, 12_544, 6_272_000),
            (14, 2, 0, 6_272_000),
            (14, 2, 12_544, 12_543),
            (14, 2, 700, 700),
        ] {
            let err = SizeRule::new(patch, merge, min, max)
                .unwrap_err()
                .to_string();
            assert!(err.starts_with("image size limits:"), "{err}");
        }
    }
}
