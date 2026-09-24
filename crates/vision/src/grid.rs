//! The resize plan of one image: the pixel size it is brought to and the token grid that gives.
//!
//! A port of `plan_image_grid`, `safe_resize`, `solve_resize_ratio`, `llm_grid` and
//! `num_image_tokens` in the reference `image_processor.py`, expression for expression. Where the
//! Python computes in floats this does too, in the same order, with the same rounding: `int()` is a
//! truncation, `math.floor`/`math.ceil` are `floor`/`ceil` of the same f64, and `** 0.5` is libm's
//! `pow` (not `sqrt` — the two may differ in the last bit, and the result is truncated right after).
//! Integer operations (`//` on ints) stay integer. The plan is a pure function of its arguments.

/// The reference's resize parameters (`vision_patch_size`, `vision_downsample_ratio`,
/// `vision_max_n_token`, `vision_min_pixels`). `vision_max_wh_ratio` is `None` in the reference
/// config and the only value this plan runs; [`crate::arch`] refuses any other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridParams {
    /// Pixels per patch side.
    pub patch: usize,
    /// Patches per aligner cell side (the 3×3 unfold).
    pub downsample: usize,
    /// The most tokens one image may take, delimiters included.
    pub max_tokens: usize,
    /// An image with fewer pixels than this is scaled up to it first.
    pub min_pixels: usize,
}

/// The plan for one image: its pixel size before patching (`best_*`, multiples of the patch) and
/// the token grid of the aligner (`n_llm_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridPlan {
    /// Aligner token rows.
    pub n_llm_h: usize,
    /// Aligner tokens per row.
    pub n_llm_w: usize,
    /// Pixel height the image is padded or resized to.
    pub best_h: usize,
    /// Pixel width the image is padded or resized to.
    pub best_w: usize,
}

impl GridPlan {
    /// Patch rows of the ViT.
    #[must_use]
    pub fn n_vit_h(&self, p: &GridParams) -> usize {
        self.best_h / p.patch
    }

    /// Patches per row of the ViT.
    #[must_use]
    pub fn n_vit_w(&self, p: &GridParams) -> usize {
        self.best_w / p.patch
    }

    /// The image's span length in the prompt, delimiters included.
    #[must_use]
    pub fn n_tokens(&self) -> usize {
        num_image_tokens(self.n_llm_h, self.n_llm_w)
    }
}

/// `num_image_tokens`: one start, one end, and a newline after every row.
#[must_use]
pub fn num_image_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
    n_llm_h * (n_llm_w + 1) + 2
}

/// `llm_grid`: the aligner's token grid over a patch grid of this pixel size,
/// `ceil((best // patch) / downsample)` per axis (an int divided by an int, then ceiled).
#[must_use]
pub fn llm_grid(best_h: usize, best_w: usize, p: &GridParams) -> (usize, usize) {
    let axis = |best: usize| ((best / p.patch) as f64 / p.downsample as f64).ceil() as usize;
    (axis(best_h), axis(best_w))
}

/// `solve_resize_ratio`: the largest aspect-preserving pixel size whose token grid fits the budget.
fn solve_resize_ratio(height: usize, width: usize, p: &GridParams) -> (usize, usize) {
    let r = height as f64 / width as f64;
    let max_w_float = ((p.max_tokens - 2) as f64 / r + 0.25).sqrt() - 0.5;
    let max_h_float = max_w_float * r;
    let cell = p.patch * p.downsample;
    if max_w_float < 1.0 {
        return ((p.max_tokens - 2) / 2 * cell, cell);
    }
    if max_h_float < 1.0 {
        return (cell, (p.max_tokens - 3) * cell);
    }
    let by_w = (max_w_float.floor() as usize * cell) as f64 / width as f64;
    let by_h = (max_h_float.floor() as usize * cell) as f64 / height as f64;
    // Python's `min(a, b)` keeps `a` unless `b < a`.
    let beta = if by_h < by_w { by_h } else { by_w };
    let snap = |n: usize| (n as f64 * beta / p.patch as f64).floor() as usize * p.patch;
    (snap(height), snap(width))
}

/// `safe_resize`: shrink the pixel size until the image costs at most `max_tokens`.
fn safe_resize(
    height: usize,
    width: usize,
    best_h: usize,
    best_w: usize,
    p: &GridParams,
) -> GridPlan {
    let (n_llm_h, n_llm_w) = llm_grid(best_h, best_w, p);
    if num_image_tokens(n_llm_h, n_llm_w) <= p.max_tokens {
        return GridPlan {
            n_llm_h,
            n_llm_w,
            best_h,
            best_w,
        };
    }
    let (best_h, best_w) = solve_resize_ratio(height, width, p);
    let (n_llm_h, n_llm_w) = llm_grid(best_h, best_w, p);
    GridPlan {
        n_llm_h,
        n_llm_w,
        best_h,
        best_w,
    }
}

/// `plan_image_grid`: the resize plan for an image of `width`×`height` pixels.
#[must_use]
pub fn plan_image_grid(width: usize, height: usize, p: &GridParams) -> GridPlan {
    let (mut width, mut height) = (width, height);
    let pixels = width * height;
    if 0 < pixels && pixels < p.min_pixels {
        let ratio = (p.min_pixels as f64 / pixels as f64).powf(0.5);
        width = (width as f64 * ratio) as usize;
        height = (height as f64 * ratio) as usize;
    }
    let best_w = (width as f64 / p.patch as f64).ceil() as usize * p.patch;
    let best_h = (height as f64 / p.patch as f64).ceil() as usize * p.patch;
    safe_resize(height, width, best_h, best_w, p)
}

#[cfg(test)]
mod tests {
    use super::{GridParams, plan_image_grid};

    const V41: GridParams = GridParams {
        patch: 14,
        downsample: 3,
        max_tokens: 1024,
        min_pixels: 295_936,
    };

    /// The four rows of the research table (visionres §1), taken from the reference code itself.
    #[test]
    fn research_table() {
        for (w, h, best_w, best_h, tokens) in [
            (448, 448, 546, 546, 184),
            (1024, 1024, 1036, 1036, 652),
            (1920, 1080, 1708, 966, 968),
            (4000, 3000, 1512, 1134, 1001),
        ] {
            let plan = plan_image_grid(w, h, &V41);
            assert_eq!(
                (plan.best_w, plan.best_h, plan.n_tokens()),
                (best_w, best_h, tokens),
                "{w}x{h}"
            );
        }
    }
}
