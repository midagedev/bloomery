//! The slice plan of one image: mtmd's `mtmd_image_preprocessor_minicpmv::get_slice_instructions`
//! over the llava-uhd base (`tools/mtmd/mtmd-image.cpp:496-549`, `:893`), as pure functions of the
//! image's size and the file's hyperparameters.
//!
//! An image of no more than `slice_size²` pixels is one overview, stretched to a multiple of 56
//! pixels (up to the slice size). A larger one is cut: the overview is the image shrunk to about
//! `slice_size²` pixels at its aspect ratio, and a grid of slices is chosen by the log aspect ratio
//! from the grids of about as many cells as the image has slice-sized areas, up to the cap. The
//! image is stretched to the grid's size and cut. Every size is rounded to a multiple of 56
//! (`round()` of Python, ties to even), and the arithmetic is the reference's: single-precision
//! ratios, and a double-precision error for the grid.

use super::Hparams;
use super::hparams::ALIGN;
use crate::VisionError;

/// One slice's rectangle in the refined image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

/// How an image is cut: the overview's size, and when sliced, the grid, the refined size the image
/// is stretched to and the slices (line by line).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// The overview's size `(w, h)`, a multiple of 56 on both sides.
    pub overview: (usize, usize),
    /// Slices per line and lines `(columns, lines)`; `(0, 0)` for an image with no slice.
    pub grid: (usize, usize),
    /// The size the image is stretched to before it is cut; `(0, 0)` with no slice.
    pub refined: (usize, usize),
    /// The slices, line by line, all of one size.
    pub slices: Vec<Rect>,
}

impl Hparams {
    /// The plan of an image of `width × height` pixels.
    pub fn plan(&self, width: usize, height: usize) -> Result<Plan, VisionError> {
        let size_err = |detail: String| VisionError::Size {
            width,
            height,
            detail,
        };
        if width == 0 || height == 0 {
            return Err(size_err("an image with no pixels has no plan".into()));
        }
        let ss = self.slice_size;
        // The reference multiplies in `int`; a size past 32 bits is refused rather than wrapped.
        let area = (width as u64)
            .checked_mul(height as u64)
            .filter(|&a| a <= i32::MAX as u64)
            .ok_or_else(|| size_err("the pixel count does not fit a 32-bit plan".into()))?;
        let ratio = width as f32 * height as f32 / (ss * ss) as f32;
        if ratio <= 1.0 {
            return Ok(Plan {
                overview: best_resize(width, height, ss, ALIGN, true),
                grid: (0, 0),
                refined: (0, 0),
                slices: Vec::new(),
            });
        }
        debug_assert!(area > (ss * ss) as u64);
        let overview = best_resize(width, height, ss, ALIGN, false);
        let multiple = (ratio.ceil() as usize).min(self.max_slices);
        let log_ratio = f64::from(width as f32 / height as f32).ln() as f32;
        let grid = best_grid(self.max_slices, multiple, log_ratio);
        if grid.0 * grid.1 > self.max_slices {
            return Err(size_err(format!(
                "the plan cuts {}x{} = {} slices; the file's cap is {}",
                grid.0,
                grid.1,
                grid.0 * grid.1,
                self.max_slices
            )));
        }
        let refined = refine_size(width, height, grid, ss);
        let (sw, sh) = (refined.0 / grid.0, refined.1 / grid.1);
        let mut slices = Vec::with_capacity(grid.0 * grid.1);
        for iy in 0..grid.1 {
            for ix in 0..grid.0 {
                slices.push(Rect {
                    x: ix * sw,
                    y: iy * sh,
                    w: sw,
                    h: sh,
                });
            }
        }
        Ok(Plan {
            overview,
            grid,
            refined,
            slices,
        })
    }
}

/// Python's `round()`: nearest, ties to even (`align_round` of the minicpmv preprocessor).
fn round_even(v: f64) -> usize {
    v.round_ties_even() as usize
}

/// `ensure_divide`: `length` to the nearest multiple of `unit`, at least `unit`.
fn ensure_divide(length: usize, unit: usize) -> usize {
    (round_even(length as f64 / unit as f64) * unit).max(unit)
}

/// `get_best_resize`: the size at about `scale²` pixels (when `allow_upscale` or the image is
/// larger than that), aspect kept, each side rounded to a multiple of `unit`. The ratio is a
/// single-precision float and the two sides truncate, as in the reference.
fn best_resize(
    width: usize,
    height: usize,
    scale: usize,
    unit: usize,
    allow_upscale: bool,
) -> (usize, usize) {
    let (mut w, mut h) = (width, height);
    if width * height > scale * scale || allow_upscale {
        let r = width as f32 / height as f32;
        h = (scale as f32 / r.sqrt()) as usize;
        w = (h as f32 * r) as usize;
    }
    (ensure_divide(w, unit), ensure_divide(h, unit))
}

/// `get_best_grid`: of the grids with `multiple - 1`, `multiple` and `multiple + 1` cells (not 1,
/// not above `max`), the one whose log aspect ratio is nearest the image's; the first of equals.
fn best_grid(max: usize, multiple: usize, log_ratio: f32) -> (usize, usize) {
    let mut best = (1, 1);
    let mut min_error = f32::INFINITY;
    for cells in [multiple.saturating_sub(1), multiple, multiple + 1] {
        if cells <= 1 || cells > max {
            continue;
        }
        for m in (1..=cells).filter(|m| cells % m == 0) {
            let grid = (m, cells / m);
            let error = (f64::from(log_ratio) - (grid.0 as f64 / grid.1 as f64).ln()).abs() as f32;
            if error < min_error {
                best = grid;
                min_error = error;
            }
        }
    }
    best
}

/// `get_refine_size` (upscaling allowed): the image's size divided by the grid, each cell planned
/// like an overview, times the grid.
fn refine_size(width: usize, height: usize, grid: (usize, usize), scale: usize) -> (usize, usize) {
    let (rw, rh) = (ensure_divide(width, grid.0), ensure_divide(height, grid.1));
    let (cw, ch) = best_resize(rw / grid.0, rh / grid.1, scale, ALIGN, true);
    (cw * grid.0, ch * grid.1)
}

#[cfg(test)]
mod tests {
    use super::{Plan, Rect};
    use crate::arch::minicpmv::hparams::tests::hp;

    fn rects(w: usize, h: usize, cols: usize, lines: usize) -> Vec<Rect> {
        (0..lines)
            .flat_map(|y| {
                (0..cols).map(move |x| Rect {
                    x: x * w,
                    y: y * h,
                    w,
                    h,
                })
            })
            .collect()
    }

    /// A 448×448 image is its own overview and a 200×200 one is stretched up to it (the overview
    /// of a small image upscales, `allow_upscale`): `ratio <= 1`, no slice. Hand computation
    /// (`ss = 448`, unit 56): 200×200 has `r = 1`, `h = int(448 / 1) = 448`, `w = 448`.
    #[test]
    fn a_small_image_is_one_overview() {
        for (w, h) in [(448, 448), (200, 200)] {
            assert_eq!(
                hp().plan(w, h).expect("plan"),
                Plan {
                    overview: (448, 448),
                    grid: (0, 0),
                    refined: (0, 0),
                    slices: vec![]
                },
                "{w}x{h}"
            );
        }
        // 300×100: ratio 0.149 <= 1; r = 3, h = int(448 / sqrt(3)) = 258, w = int(258 * 3) = 774
        // (774 > 448: the overview of a small, wide image is not capped); 774 / 56 = 13.82 -> 14
        // -> 784; 258 / 56 = 4.607 -> 5 -> 280.
        assert_eq!(hp().plan(300, 100).expect("plan").overview, (784, 280));
    }

    /// 1000×700: ratio 3.4877, ceil 4. Overview: r = 1.428571, h = int(448 / 1.195229) = 374,
    /// w = int(374 * 1.428571) = 534, rounded to 560×392. Grids of 3, 4 and 5 cells against
    /// ln(1000/700) = 0.3567: 2×2 has error 0.3567, every other is further. The refined size:
    /// cells of 500×350 planned like an overview are 560×392, so 1120×784.
    #[test]
    fn a_1000_by_700_image_is_a_2x2_grid_of_560_by_392() {
        let p = hp().plan(1000, 700).expect("plan");
        assert_eq!(p.overview, (560, 392));
        assert_eq!((p.grid, p.refined), ((2, 2), (1120, 784)));
        assert_eq!(p.slices, rects(560, 392, 2, 2));
    }

    /// 3000×400: ratio 5.979, ceil 6. Overview: r = 7.5, h = int(448 / 2.7386) = 163,
    /// w = int(163 * 7.5) = 1222, rounded to 1232×168. Grids of 5, 6 and 7 cells against
    /// ln(7.5) = 2.0149: 7×1 (error 0.069) beats 6×1 (0.223). The refined size: 3000 / 7 rounds to
    /// 429 -> 3003, the cell 429×400 is planned to 448×448, so 3136×448: seven square slices.
    #[test]
    fn a_3000_by_400_image_is_seven_448_slices() {
        let p = hp().plan(3000, 400).expect("plan");
        assert_eq!(p.overview, (1232, 168));
        assert_eq!((p.grid, p.refined), ((7, 1), (3136, 448)));
        assert_eq!(p.slices, rects(448, 448, 7, 1));
    }

    /// The cap: 6000×6000 asks for 179 cells and is held to the file's 9, a 3×3 grid (the grids of
    /// 8, 9 and 10 cells; 10 is past the cap). A cap of 4 gives 2×2.
    #[test]
    fn the_slice_cap_holds() {
        let p = hp().plan(6000, 6000).expect("plan");
        assert_eq!((p.grid, p.slices.len()), ((3, 3), 9));
        let mut four = hp();
        four.max_slices = 4;
        assert_eq!(four.plan(6000, 6000).expect("plan").grid, (2, 2));
        // Equal errors keep the first grid: 600×600 (ratio 1.79, ceil 2) has 1×2 and 2×1 at the
        // same error ln 2, and the grid of one column and two lines is found first.
        assert_eq!(hp().plan(600, 600).expect("plan").grid, (1, 2));
        assert_eq!(hp().plan(1000, 1000).expect("plan").grid, (2, 2));
    }

    /// `round()` breaks ties to even (Python's, `align_round`): 84 / 56 = 1.5 -> 2, 140 / 56 = 2.5 -> 2,
    /// 28 / 56 = 0.5 -> 0 held to one unit, 1005 / 2 = 502.5 -> 502.
    #[test]
    fn sizes_round_ties_to_even() {
        use super::ensure_divide;
        assert_eq!(
            [
                ensure_divide(84, 56),
                ensure_divide(140, 56),
                ensure_divide(28, 56)
            ],
            [112, 112, 56]
        );
        assert_eq!(ensure_divide(1005, 2), 1004);
        assert_eq!(ensure_divide(1007, 2), 1008);
    }

    #[test]
    fn an_image_with_no_pixels_is_refused_by_name() {
        for (w, h) in [(0, 10), (10, 0), (0, 0)] {
            let e = hp().plan(w, h).unwrap_err().to_string();
            assert!(e.starts_with(&format!("image size {w}x{h}:")), "{e}");
        }
        assert!(hp().plan(70_000, 70_000).is_err());
    }
}
