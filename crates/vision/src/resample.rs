//! Pillow's bicubic resize and `ImageOps.pad`, ported so that the bytes are Pillow's bytes.
//!
//! The reference pads every image with `ImageOps.pad(image, (best_w, best_h), color=(127, 127, 127))`,
//! which is `ImageOps.contain` (an aspect-preserving `Image.resize` with the default bicubic
//! filter) and a paste onto a grey canvas. Pillow's resize of an 8-bit image is integer arithmetic
//! on top of f64 filter weights (`libImaging/Resample.c`): the weights of each output pixel are
//! computed in f64, normalized to sum 1, rounded to fixed point with 22 fractional bits, and each
//! pass accumulates `u8 × i32` from a half-unit start, shifts and clips to u8. The horizontal pass
//! runs first and its u8 output is what the vertical pass reads. This module is that code in Rust,
//! operation for operation — same f64 expressions in the same order, the same truncating casts and
//! the same clip — so for the version the oracle ran it produces the same bytes, which is what the
//! preprocessing gate checks.
//!
//! The Python side's rounding is Python's: `round()` of a float is round-half-to-even.
//!
//! The other pad is llama.cpp's `PAD_CEIL` (`img_tool::resize`, mtmd): the same resize, a contain
//! size rounded up and capped at the canvas in f32, and offsets by integer division. The fill is
//! the projector's (llama.cpp sets `image_pad_color` per projector type), so both pads take it as
//! an argument and no fill lives here. [`pad`] and [`pad_ceil`] differ in their geometry only; one
//! function resizes and pastes for both.

use crate::VisionError;
use crate::image::Rgb8;

/// Fractional bits of the fixed-point weights: 32 bits less 8 for the sample and 2 of headroom for
/// weights below 0 or above 1 (`PRECISION_BITS` in Resample.c).
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Bicubic support radius in input pixels at scale 1 (`BICUBIC = {bicubic_filter, 2.0}`).
const BICUBIC_SUPPORT: f64 = 2.0;

/// `bicubic_filter` with `a = -0.5`, written in Resample.c's association.
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = if x < 0.0 { -x } else { x };
    if x < 1.0 {
        return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    }
    if x < 2.0 {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * A;
    }
    0.0
}

/// One axis of a resize: for each output index the first input index, the tap count, and the taps
/// as fixed-point weights, `ksize` slots per output index (`precompute_coeffs` followed by
/// `normalize_coeffs_8bpc`).
struct Axis {
    ksize: usize,
    /// `(xmin, taps)` per output index.
    bounds: Vec<(usize, usize)>,
    weights: Vec<i32>,
}

impl Axis {
    /// `precompute_coeffs(in_size, in0, in1, out_size, &BICUBIC)` and the 8-bit normalization; the
    /// box is always the whole input (`in0 = 0`, `in1 = in_size`, both C floats).
    fn bicubic(in_size: usize, out_size: usize) -> Axis {
        let (in0, in1) = (0.0f32, in_size as f32);
        let scale = f64::from(in1 - in0) / out_size as f64;
        let filterscale = if scale < 1.0 { 1.0 } else { scale };
        let support = BICUBIC_SUPPORT * filterscale;
        let ksize = support.ceil() as usize * 2 + 1;
        let mut bounds = Vec::with_capacity(out_size);
        let mut pre = vec![0.0f64; out_size * ksize];
        for xx in 0..out_size {
            let center = f64::from(in0) + (xx as f64 + 0.5) * scale;
            let ss = 1.0 / filterscale;
            // C's `(int)` truncates toward zero; a negative start is then clamped to 0.
            let xmin = ((center - support + 0.5) as i64).max(0);
            let xmax = ((center + support + 0.5) as i64).min(in_size as i64) - xmin;
            let (xmin, taps) = (xmin as usize, xmax.max(0) as usize);
            let k = &mut pre[xx * ksize..xx * ksize + taps];
            let mut ww = 0.0f64;
            for (x, slot) in k.iter_mut().enumerate() {
                let w = bicubic(((x + xmin) as f64 - center + 0.5) * ss);
                *slot = w;
                ww += w;
            }
            if ww != 0.0 {
                k.iter_mut().for_each(|w| *w /= ww);
            }
            bounds.push((xmin, taps));
        }
        let one = f64::from(1u32 << PRECISION_BITS);
        let weights = pre
            .iter()
            .map(|&w| {
                if w < 0.0 {
                    (-0.5 + w * one) as i32
                } else {
                    (0.5 + w * one) as i32
                }
            })
            .collect();
        Axis {
            ksize,
            bounds,
            weights,
        }
    }

    fn taps(&self, out: usize) -> (usize, &[i32]) {
        let (xmin, taps) = self.bounds[out];
        (
            xmin,
            &self.weights[out * self.ksize..out * self.ksize + taps],
        )
    }
}

/// `clip8`: the accumulator's integer part, clipped to a byte.
fn clip8(acc: i32) -> u8 {
    (acc >> PRECISION_BITS).clamp(0, 255) as u8
}

/// One tap of every channel: `ss += (UINT8)in * k` (C int arithmetic, which wraps).
fn accumulate(acc: &mut [i32; 3], px: &[u8], w: i32) {
    for (a, &v) in acc.iter_mut().zip(px) {
        *a = a.wrapping_add(i32::from(v).wrapping_mul(w));
    }
}

/// The three clipped channels of one output pixel.
fn store(out: &mut [u8], acc: [i32; 3]) {
    for (o, a) in out.iter_mut().zip(acc) {
        *o = clip8(a);
    }
}

/// `Image.resize((width, height), Image.BICUBIC)` of an RGB image, as Pillow computes it. A size
/// equal to the input's is a copy, as in Pillow.
#[must_use]
pub fn resize_bicubic(src: &Rgb8, width: usize, height: usize) -> Rgb8 {
    if (width, height) == (src.width, src.height) {
        return src.clone();
    }
    if width == 0 || height == 0 || src.width == 0 || src.height == 0 {
        return Rgb8 {
            width,
            height,
            data: Vec::new(),
        };
    }
    let horiz = Axis::bicubic(src.width, width);
    let mut vert = Axis::bicubic(src.height, height);
    let need_h = width != src.width;
    let need_v = height != src.height;
    // First and one-past-last input row the vertical pass reads.
    let first = vert.bounds[0].0;
    let last = vert.bounds[height - 1].0 + vert.bounds[height - 1].1;
    let mut horizontal = None;
    if need_h {
        for b in &mut vert.bounds {
            b.0 -= first;
        }
        let mut out = Rgb8 {
            width,
            height: last - first,
            data: vec![0; width * (last - first) * 3],
        };
        for yy in 0..last - first {
            let row = &src.data[(yy + first) * src.width * 3..(yy + first + 1) * src.width * 3];
            for xx in 0..width {
                let (xmin, k) = horiz.taps(xx);
                let mut acc = [1i32 << (PRECISION_BITS - 1); 3];
                for (x, &w) in k.iter().enumerate() {
                    let p = &row[(x + xmin) * 3..(x + xmin) * 3 + 3];
                    accumulate(&mut acc, p, w);
                }
                let o = (yy * width + xx) * 3;
                store(&mut out.data[o..o + 3], acc);
            }
        }
        horizontal = Some(out);
    }
    let img = horizontal.as_ref().unwrap_or(src);
    if need_v {
        let w = img.width;
        let mut out = Rgb8 {
            width: w,
            height,
            data: vec![0; w * height * 3],
        };
        for yy in 0..height {
            let (ymin, k) = vert.taps(yy);
            for xx in 0..w {
                let mut acc = [1i32 << (PRECISION_BITS - 1); 3];
                for (y, &wt) in k.iter().enumerate() {
                    let i = ((y + ymin) * w + xx) * 3;
                    accumulate(&mut acc, &img.data[i..i + 3], wt);
                }
                let o = (yy * w + xx) * 3;
                store(&mut out.data[o..o + 3], acc);
            }
        }
        return out;
    }
    horizontal.unwrap_or_else(|| src.clone())
}

/// Where a pad puts an image of `width`×`height` on a `best_w`×`best_h` canvas: the size it is
/// resized to, and the offset of the paste.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PadGeometry {
    pub resized_w: usize,
    pub resized_h: usize,
    pub off_x: usize,
    pub off_y: usize,
}

impl PadGeometry {
    /// `ImageOps.contain(image, (best_w, best_h))`'s size and `ImageOps.pad`'s centred offset
    /// (centering (0.5, 0.5)), both with Python's round-half-to-even.
    pub fn of(
        width: usize,
        height: usize,
        best_w: usize,
        best_h: usize,
    ) -> Result<PadGeometry, VisionError> {
        let size_err = |detail: &str| VisionError::Size {
            width,
            height,
            detail: detail.to_string(),
        };
        if width == 0 || height == 0 || best_w == 0 || best_h == 0 {
            return Err(size_err("an empty image or canvas has no pad"));
        }
        let im_ratio = width as f64 / height as f64;
        let dest_ratio = best_w as f64 / best_h as f64;
        let (mut rw, mut rh) = (best_w, best_h);
        if im_ratio != dest_ratio {
            if im_ratio > dest_ratio {
                let new_h =
                    (height as f64 / width as f64 * best_w as f64).round_ties_even() as usize;
                if new_h != best_h {
                    rh = new_h;
                }
            } else {
                let new_w =
                    (width as f64 / height as f64 * best_h as f64).round_ties_even() as usize;
                if new_w != best_w {
                    rw = new_w;
                }
            }
        }
        if rw == 0 || rh == 0 {
            return Err(size_err("contain() rounds one side to 0 pixels"));
        }
        let (mut off_x, mut off_y) = (0, 0);
        if (rw, rh) != (best_w, best_h) {
            if rw != best_w {
                off_x = ((best_w - rw) as f64 * 0.5).round_ties_even() as usize;
            } else {
                off_y = ((best_h - rh) as f64 * 0.5).round_ties_even() as usize;
            }
        }
        Ok(PadGeometry {
            resized_w: rw,
            resized_h: rh,
            off_x,
            off_y,
        })
    }

    /// llama.cpp's `PAD_CEIL` geometry for an image of `width`×`height` on a `target_w`×`target_h`
    /// canvas: `scale = min(target_w / width, target_h / height)` in f32, each side
    /// `min(ceil(side * scale), target)` in f32, the offset `(target - size) / 2` in integers.
    pub fn ceil_of(
        width: usize,
        height: usize,
        target_w: usize,
        target_h: usize,
    ) -> Result<PadGeometry, VisionError> {
        if width == 0 || height == 0 || target_w == 0 || target_h == 0 {
            return Err(VisionError::Size {
                width,
                height,
                detail: "an empty image or canvas has no pad".to_string(),
            });
        }
        let scale_w = target_w as f32 / width as f32;
        let scale_h = target_h as f32 / height as f32;
        // `std::min(a, b)` is `(b < a) ? b : a`.
        let scale = if scale_h < scale_w { scale_h } else { scale_w };
        let resized_w = ((width as f32 * scale).ceil() as usize).min(target_w);
        let resized_h = ((height as f32 * scale).ceil() as usize).min(target_h);
        Ok(PadGeometry {
            resized_w,
            resized_h,
            off_x: (target_w - resized_w) / 2,
            off_y: (target_h - resized_h) / 2,
        })
    }
}

/// Resize `src` to the geometry's size and paste it at its offset on a `canvas_w`×`canvas_h`
/// canvas of `fill`; a size equal to the canvas is the resized image itself.
fn paste(src: &Rgb8, g: PadGeometry, canvas_w: usize, canvas_h: usize, fill: [u8; 3]) -> Rgb8 {
    let resized = resize_bicubic(src, g.resized_w, g.resized_h);
    if (g.resized_w, g.resized_h) == (canvas_w, canvas_h) {
        return resized;
    }
    let mut out = Rgb8::filled(canvas_w, canvas_h, fill);
    for y in 0..g.resized_h {
        let from = &resized.data[y * g.resized_w * 3..(y + 1) * g.resized_w * 3];
        let at = ((y + g.off_y) * canvas_w + g.off_x) * 3;
        out.data[at..at + from.len()].copy_from_slice(from);
    }
    out
}

/// `ImageOps.pad(image, (best_w, best_h), color=fill)` with the default bicubic filter and
/// centering.
pub fn pad(src: &Rgb8, best_w: usize, best_h: usize, fill: [u8; 3]) -> Result<Rgb8, VisionError> {
    let g = PadGeometry::of(src.width, src.height, best_w, best_h)?;
    Ok(paste(src, g, best_w, best_h, fill))
}

/// llama.cpp's `PAD_CEIL` resize of an image to a `target_w`×`target_h` canvas: a size equal to
/// the canvas is a plain copy, anything else is resized to [`PadGeometry::ceil_of`]'s size with
/// the bicubic filter and pasted on a canvas of `fill`, the projector's pad colour.
pub fn pad_ceil(
    src: &Rgb8,
    target_w: usize,
    target_h: usize,
    fill: [u8; 3],
) -> Result<Rgb8, VisionError> {
    if (src.width, src.height) == (target_w, target_h) {
        return Ok(src.clone());
    }
    let g = PadGeometry::ceil_of(src.width, src.height, target_w, target_h)?;
    Ok(paste(src, g, target_w, target_h, fill))
}

#[cfg(test)]
mod tests {
    use super::{Axis, PRECISION_BITS, PadGeometry, bicubic, pad, pad_ceil};
    use crate::{Rgb8, VisionError};

    /// Two pad colours of the two projector types that call the pads; the pads themselves take
    /// whichever colour they are given.
    const GREY: [u8; 3] = [127, 127, 127];
    const BLACK: [u8; 3] = [0, 0, 0];

    /// The filter is 1 at 0, 0 at the other integers, and its taps at a half-pixel phase sum to 1.
    #[test]
    fn bicubic_shape() {
        assert_eq!(bicubic(0.0), 1.0);
        for x in [1.0, 2.0, -1.0, -2.0, 2.5] {
            assert_eq!(bicubic(x), 0.0, "{x}");
        }
        let sum: f64 = [-1.5, -0.5, 0.5, 1.5].iter().map(|&x| bicubic(x)).sum();
        assert_eq!(sum, 1.0);
    }

    /// Every output pixel's fixed-point weights sum to one unit within the rounding of its taps,
    /// up and down.
    #[test]
    fn weights_sum_to_one_unit() {
        for (i, o) in [(448, 546), (2400, 1692), (600, 630), (1, 5), (7, 1)] {
            let a = Axis::bicubic(i, o);
            for out in 0..o {
                let (_, k) = a.taps(out);
                let sum: i64 = k.iter().map(|&w| i64::from(w)).sum();
                let dev = (sum - (1i64 << PRECISION_BITS)).abs();
                assert!(dev <= k.len() as i64, "{i}->{o} pixel {out}: sum {sum}");
            }
        }
    }

    /// Python's `round()` is half-to-even: a 9-pixel gap pads 4 above (4.5 → 4), a 7-pixel gap 4
    /// (3.5 → 4).
    #[test]
    fn pad_offset_rounds_half_to_even() {
        let g = PadGeometry::of(100, 91, 100, 100).expect("geometry");
        assert_eq!((g.resized_w, g.resized_h, g.off_y), (100, 91, 4));
        let g = PadGeometry::of(100, 93, 100, 100).expect("geometry");
        assert_eq!((g.resized_h, g.off_y), (93, 4));
    }

    /// llama.cpp's `PAD_CEIL` geometry on the sizes of the size plan's worked examples, each
    /// computed by hand in f32 (`scale = min(w̄/w, h̄/h)`, `new = min(ceil(side · scale), target)`,
    /// `offset = (target - new) / 2` in integers):
    /// * 1000×700 → 992×704: scale = min(992/1000, 704/700) = 0.992; 1000 · 0.992 = 992.0 in f32
    ///   → 992; 700 · 0.992 = 694.4 → 695; offset (0, (704 − 695) / 2 = 4).
    /// * 777×513 → 768×512: scale = min(768/777, 512/513) = 0.98842; 777 · 0.98842 = 768.0 in f32
    ///   → 768; 513 · 0.98842 = 507.06 → 508; offset (0, 2).
    /// * 1000×1 → 992×32: scale = min(0.992, 32) = 0.992; 992 wide, 1 · 0.992 = 0.992 → 1; offset
    ///   (0, (32 − 1) / 2 = 15).
    /// * 700×1000 → 704×992: scale = min(704/700, 992/1000) = 0.992; 700 · 0.992 = 694.4 → 695,
    ///   1000 · 0.992 = 992 → 992; offset ((704 − 695) / 2 = 4, 0).
    #[test]
    fn ceil_geometry_on_four_sizes() {
        for ((w, h, tw, th), want) in [
            ((1000, 700, 992, 704), (992, 695, 0, 4)),
            ((777, 513, 768, 512), (768, 508, 0, 2)),
            ((1000, 1, 992, 32), (992, 1, 0, 15)),
            ((700, 1000, 704, 992), (695, 992, 4, 0)),
        ] {
            let g = PadGeometry::ceil_of(w, h, tw, th).expect("geometry");
            assert_eq!(
                (g.resized_w, g.resized_h, g.off_x, g.off_y),
                want,
                "{w}x{h} on {tw}x{th}"
            );
        }
    }

    /// Where the content lands is where the pixels are: every pixel outside the pasted rectangle
    /// is the fill the caller gave and every pixel inside is the image's colour (a constant image
    /// stays itself under the bicubic filter's normalized weights). The fill is an argument, so a
    /// pad that ignored it and painted one colour fails on the other.
    #[test]
    fn ceil_pad_pastes_on_the_given_fill() {
        let img = Rgb8::filled(1000, 700, [200, 100, 50]);
        for fill in [BLACK, GREY, [1, 2, 3]] {
            let out = pad_ceil(&img, 992, 704, fill).expect("pad");
            assert_eq!((out.width, out.height), (992, 704));
            let (y0, y1) = (4, 4 + 695);
            for y in 0..704 {
                for x in 0..992 {
                    let px = &out.data[(y * 992 + x) * 3..(y * 992 + x) * 3 + 3];
                    if (y0..y1).contains(&y) {
                        assert_eq!(px, [200, 100, 50], "{fill:?}: inside at ({x}, {y})");
                    } else {
                        assert_eq!(px, fill, "{fill:?}: outside at ({x}, {y})");
                    }
                }
            }
        }
    }

    /// The bands above and below a wide strip are black and the strip is the resized row.
    #[test]
    fn ceil_pad_of_a_strip() {
        let out = pad_ceil(&Rgb8::filled(1000, 1, [255, 255, 255]), 992, 32, BLACK).expect("pad");
        assert_eq!((out.width, out.height), (992, 32));
        let row = |y: usize| &out.data[y * 992 * 3..(y + 1) * 992 * 3];
        for y in (0..32).filter(|&y| y != 15) {
            assert!(row(y).iter().all(|&v| v == 0), "row {y}");
        }
        assert!(row(15).iter().all(|&v| v == 255));
    }

    /// An image of the canvas's size is a plain copy, and one whose scaled size fills the canvas
    /// exactly is its resize with no fill.
    #[test]
    fn ceil_pad_same_size_is_a_copy_and_exact_fit_has_no_fill() {
        let img = Rgb8 {
            width: 3,
            height: 2,
            data: (0..18).collect(),
        };
        assert_eq!(pad_ceil(&img, 3, 2, BLACK).expect("pad"), img);
        let big = pad_ceil(&Rgb8::filled(2, 1, [9, 9, 9]), 128, 64, BLACK).expect("pad");
        assert_eq!((big.width, big.height), (128, 64));
        assert!(big.data.iter().all(|&v| v == 9));
    }

    /// A pad with no pixels is refused by name, not planned.
    #[test]
    fn ceil_pad_refuses_empty_sizes() {
        for (w, h, tw, th) in [(0, 5, 32, 32), (5, 0, 32, 32), (5, 5, 0, 32), (5, 5, 32, 0)] {
            assert!(
                matches!(
                    PadGeometry::ceil_of(w, h, tw, th),
                    Err(VisionError::Size { .. })
                ),
                "{w}x{h} on {tw}x{th}"
            );
        }
    }

    /// Both pads resize and paste through one function: the grey pad of an image the canvas
    /// does not fit keeps its fill, the black pad its own.
    #[test]
    fn the_two_pads_differ_in_geometry_and_fill_only() {
        let img = Rgb8::filled(100, 91, [10, 20, 30]);
        let grey = pad(&img, 100, 100, GREY).expect("pad");
        let black = pad_ceil(&img, 100, 100, BLACK).expect("pad");
        // ImageOps: height 91, offset round(4.5) = 4; PAD_CEIL: scale 1.0, height 91, offset 9 / 2 = 4.
        for out in [&grey, &black] {
            assert_eq!(&out.data[(4 * 100) * 3..(4 * 100) * 3 + 3], [10, 20, 30]);
        }
        assert_eq!(&grey.data[0..3], GREY);
        assert_eq!(&black.data[0..3], BLACK);
    }
}
