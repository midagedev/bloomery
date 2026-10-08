//! From an image already at its plan's pixel size to the ViT's input: the normalize and the cut,
//! the same for every projector type; each projector's own module plans the size and pads or
//! resizes the image to it.
//!
//! Every byte `v` becomes `((v / 255) - 0.5) / 0.5` in f32 — the three operations of V4.1's
//! reference, each rounded to f32 as torch rounds them — and that f32 is rounded to bf16 to
//! nearest, ties to even (torch's `.to(torch.bfloat16)`). Inside a patch the order is channel,
//! then pixel row, then pixel column (the reference's `reshape(3, h, p, w, p).permute(1, 3, 0, 2,
//! 4)`), so one patch is 3·p² values.
//!
//! One cut serves every projector type ([`patchify`]): a [`PatchLayout`] says how many patches
//! make a merge group on a side (the patches of a group are consecutive; a group of one is
//! row-major, the order of V4.1) and how many temporal frames each patch is repeated over.

use crate::VisionError;
use crate::image::Rgb8;

/// The plan for one image: its pixel size before patching (`best_*`, multiples of the patch group
/// the layout cuts) and the token grid the projector emits (`n_llm_*`: V4.1's aligner cells,
/// Clef's merged tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridPlan {
    /// Token rows the projector emits.
    pub n_llm_h: usize,
    /// Tokens per row.
    pub n_llm_w: usize,
    /// Pixel height the image is padded or resized to.
    pub best_h: usize,
    /// Pixel width the image is padded or resized to.
    pub best_w: usize,
}

/// The ViT's input for one image: `n_vit_h * n_vit_w` patches of `patch_len` bf16 values each, as
/// raw bf16 bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Patches {
    /// The pixel size the image was cut at and the token grid it takes in the prompt.
    pub plan: GridPlan,
    pub n_vit_h: usize,
    pub n_vit_w: usize,
    /// Values per patch, `frames * 3 * patch * patch`.
    pub patch_len: usize,
    /// `n_vit_h * n_vit_w * patch_len` bf16 bit patterns, patch-major, patches in the layout's
    /// order.
    pub bf16: Vec<u16>,
}

/// How an image is cut into patches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchLayout {
    /// Pixels per patch side.
    pub patch: usize,
    /// Patches per merge group side: the `merge`² patches of a group are consecutive, in raster
    /// order inside it, and the groups follow in raster order. 1 is row-major over the patch grid.
    pub merge: usize,
    /// Copies of each patch's values, one after the other (a still image fills every temporal
    /// frame of a video patch embedding with itself).
    pub frames: usize,
}

impl PatchLayout {
    /// Row-major patches, one frame.
    #[must_use]
    pub const fn row_major(patch: usize) -> PatchLayout {
        PatchLayout {
            patch,
            merge: 1,
            frames: 1,
        }
    }

    /// Values per patch: `frames` copies of the channel, row, column order.
    #[must_use]
    pub const fn patch_len(&self) -> usize {
        self.frames * 3 * self.patch * self.patch
    }
}

/// f32 to bf16 bits, round to nearest, ties to even; a NaN becomes the canonical quiet NaN
/// (torch's `c10::BFloat16` conversion).
#[must_use]
pub fn f32_to_bf16(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7FC0;
    }
    let b = x.to_bits();
    let bias = 0x7FFF + ((b >> 16) & 1);
    ((b + bias) >> 16) as u16
}

/// The normalized value of one byte, as bf16 bits.
#[must_use]
pub fn normalize(v: u8) -> u16 {
    let x = f32::from(v) / 255.0;
    f32_to_bf16((x - 0.5) / 0.5)
}

/// Cut an image already at the plan's pixel size into normalized patches, in `layout`'s order. The
/// size must be a multiple of `patch * merge` on both sides.
pub fn patchify(image: &Rgb8, plan: GridPlan, layout: PatchLayout) -> Result<Patches, VisionError> {
    let PatchLayout {
        patch: s,
        merge: m,
        frames,
    } = layout;
    if s == 0 || m == 0 || frames == 0 {
        return Err(VisionError::Limits {
            detail: format!(
                "the layout has patch {s}, merge {m} and {frames} frames; each must be at least 1"
            ),
        });
    }
    let size_err = |detail: String| VisionError::Size {
        width: image.width,
        height: image.height,
        detail,
    };
    let group = s * m;
    if image.width != plan.best_w
        || image.height != plan.best_h
        || !plan.best_w.is_multiple_of(group)
        || !plan.best_h.is_multiple_of(group)
    {
        return Err(size_err(format!(
            "the plan is {}x{} in groups of {m}x{m} {s}-pixel patches",
            plan.best_w, plan.best_h
        )));
    }
    let table: Vec<u16> = (0..=255u8).map(normalize).collect();
    let (n_vit_h, n_vit_w) = (image.height / s, image.width / s);
    let one_frame = 3 * s * s;
    let mut bf16 = Vec::with_capacity(n_vit_h * n_vit_w * layout.patch_len());
    for gy in 0..n_vit_h / m {
        for gx in 0..n_vit_w / m {
            for dy in 0..m {
                for dx in 0..m {
                    let (ph, pw) = (gy * m + dy, gx * m + dx);
                    let at = bf16.len();
                    for c in 0..3 {
                        for py in 0..s {
                            let row = ((ph * s + py) * image.width + pw * s) * 3;
                            for px in 0..s {
                                bf16.push(table[usize::from(image.data[row + px * 3 + c])]);
                            }
                        }
                    }
                    for _ in 1..frames {
                        bf16.extend_from_within(at..at + one_frame);
                    }
                }
            }
        }
    }
    Ok(Patches {
        plan,
        n_vit_h,
        n_vit_w,
        patch_len: layout.patch_len(),
        bf16,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{GridPlan, PatchLayout, f32_to_bf16, normalize, patchify};
    use crate::{Rgb8, VisionError};

    /// A plan of `w`×`h` pixels; the token grid is not what these tests read.
    fn plan(w: usize, h: usize) -> GridPlan {
        GridPlan {
            n_llm_h: 1,
            n_llm_w: 1,
            best_h: h,
            best_w: w,
        }
    }

    /// An image whose pixel `(x, y)` channel `c` is `f(c, y, x)`.
    fn image(w: usize, h: usize, f: impl Fn(usize, usize, usize) -> u8) -> Rgb8 {
        let mut data = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                data.extend((0..3).map(|c| f(c, y, x)));
            }
        }
        Rgb8 {
            width: w,
            height: h,
            data,
        }
    }

    /// Every byte's normalized value back to the byte (the 256 values are distinct in bf16).
    fn unnormalize() -> HashMap<u16, usize> {
        let t: HashMap<u16, usize> = (0..=255u8)
            .map(|v| (normalize(v), usize::from(v)))
            .collect();
        assert_eq!(t.len(), 256);
        t
    }

    /// The cut V4.1 had before the layout: patches row-major, channel, row, column inside.
    fn row_major_reference(image: &Rgb8, s: usize) -> Vec<u16> {
        let table: Vec<u16> = (0..=255u8).map(normalize).collect();
        let mut out = Vec::new();
        for ph in 0..image.height / s {
            for pw in 0..image.width / s {
                for c in 0..3 {
                    for py in 0..s {
                        let row = ((ph * s + py) * image.width + pw * s) * 3;
                        for px in 0..s {
                            out.push(table[usize::from(image.data[row + px * 3 + c])]);
                        }
                    }
                }
            }
        }
        out
    }

    /// Row-major with one frame is the cut V4.1 had, on several shapes and patch sizes.
    #[test]
    fn row_major_layout_is_the_former_cut() {
        for (s, w, h) in [
            (14, 42, 28),
            (14, 14, 70),
            (2, 8, 6),
            (1, 5, 3),
            (16, 64, 32),
        ] {
            let img = image(w, h, |c, y, x| (c * 91 + y * 17 + x * 5 + y * x) as u8);
            let got = patchify(&img, plan(w, h), PatchLayout::row_major(s)).expect("cut");
            assert_eq!(
                got.bf16,
                row_major_reference(&img, s),
                "{s}-pixel patches of {w}x{h}"
            );
            assert_eq!(
                (got.n_vit_h, got.n_vit_w, got.patch_len),
                (h / s, w / s, 3 * s * s)
            );
        }
    }

    /// Merge groups: the `merge`² patches of a group are consecutive, raster inside, the groups
    /// raster. The patch order is written out by hand; every value is decoded back to
    /// `(c, y, x)` of its source pixel (pixel value `c << 6 | y << 3 | x`) and compared to the
    /// patch's position, frame, channel, row and column.
    #[test]
    fn merge_order_index_by_index() {
        let inv = unnormalize();
        let value = |c: usize, y: usize, x: usize| (c << 6 | y << 3 | x) as u8;
        // 2-pixel patches, 2x2 groups, two frames.
        let layout = PatchLayout {
            patch: 2,
            merge: 2,
            frames: 2,
        };
        // 8 wide, 4 tall: patch grid 4x2, one row of two groups.
        let wide = [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 2), (0, 3), (1, 2), (1, 3)],
        ];
        // 4 wide, 8 tall: patch grid 2x4, two rows of one group.
        let tall = [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(2, 0), (2, 1), (3, 0), (3, 1)],
        ];
        // 8 wide, 8 tall: patch grid 4x4, two rows of two groups.
        let square = [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 2), (0, 3), (1, 2), (1, 3)],
            [(2, 0), (2, 1), (3, 0), (3, 1)],
            [(2, 2), (2, 3), (3, 2), (3, 3)],
        ];
        for (w, h, groups) in [(8, 4, &wide[..]), (4, 8, &tall[..]), (8, 8, &square[..])] {
            let got = patchify(&image(w, h, value), plan(w, h), layout).expect("cut");
            assert_eq!(got.patch_len, 24);
            assert_eq!((got.n_vit_h, got.n_vit_w), (h / 2, w / 2));
            assert_eq!(got.bf16.len(), groups.len() * 4 * 24);
            for (i, &(ph, pw)) in groups.iter().flatten().enumerate() {
                for frame in 0..2 {
                    for c in 0..3 {
                        for py in 0..2 {
                            for px in 0..2 {
                                let at = i * 24 + frame * 12 + c * 4 + py * 2 + px;
                                let (y, x) = (ph * 2 + py, pw * 2 + px);
                                let want = usize::from(value(c, y, x));
                                assert_eq!(
                                    inv[&got.bf16[at]], want,
                                    "{w}x{h}: patch {i} frame {frame} c {c} py {py} px {px}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The Clef layout (16-pixel patches, 2x2 groups, two frames) on a 64×64 image, index by
    /// index: 16 patches in merge order (two rows of two groups), 1,536 values each, the second
    /// 768 a copy of the first.
    #[test]
    fn the_clef_layout_is_1536_values_a_patch() {
        let value = |c: usize, y: usize, x: usize| ((c * 83 + y * 7 + x * 3) % 251) as u8;
        let layout = PatchLayout {
            patch: 16,
            merge: 2,
            frames: 2,
        };
        let got = patchify(&image(64, 64, value), plan(64, 64), layout).expect("cut");
        assert_eq!((got.n_vit_h, got.n_vit_w, got.patch_len), (4, 4, 1536));
        assert_eq!(got.bf16.len(), 16 * 1536);
        let groups = [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 2), (0, 3), (1, 2), (1, 3)],
            [(2, 0), (2, 1), (3, 0), (3, 1)],
            [(2, 2), (2, 3), (3, 2), (3, 3)],
        ];
        for (i, &(ph, pw)) in groups.iter().flatten().enumerate() {
            let patch = &got.bf16[i * 1536..(i + 1) * 1536];
            assert_eq!(patch[..768], patch[768..], "patch {i}: the frames differ");
            for c in 0..3 {
                for py in 0..16 {
                    for px in 0..16 {
                        let want = normalize(value(c, ph * 16 + py, pw * 16 + px));
                        assert_eq!(
                            patch[c * 256 + py * 16 + px],
                            want,
                            "patch {i} c {c} {py},{px}"
                        );
                    }
                }
            }
        }
    }

    /// A size that is not the plan's or not whole groups is refused as a size, by name.
    #[test]
    fn a_cut_that_does_not_fit_is_refused() {
        let layout = PatchLayout {
            patch: 16,
            merge: 2,
            frames: 2,
        };
        let img = image(64, 32, |_, _, _| 0);
        for (plan, layout) in [
            (plan(64, 64), layout),
            (
                plan(64, 32),
                PatchLayout {
                    patch: 17,
                    ..layout
                },
            ),
            (plan(48, 32), layout),
        ] {
            assert!(
                matches!(patchify(&img, plan, layout), Err(VisionError::Size { .. })),
                "{plan:?} {layout:?}"
            );
        }
        // 48 pixels are three patches: not whole 2x2 groups.
        let narrow = image(48, 32, |_, _, _| 0);
        let e = patchify(&narrow, plan(48, 32), layout)
            .expect_err("groups")
            .to_string();
        assert_eq!(
            e,
            "image size 48x32: the plan is 48x32 in groups of 2x2 16-pixel patches"
        );
    }

    /// A layout with a zero side or frame count is a configuration error, not a size one: it is
    /// refused as the limits, by name, whatever the image is.
    #[test]
    fn a_zero_layout_is_a_limits_error() {
        let layout = PatchLayout {
            patch: 16,
            merge: 2,
            frames: 2,
        };
        let img = image(64, 32, |_, _, _| 0);
        for (bad, text) in [
            (
                PatchLayout { merge: 0, ..layout },
                "image size limits: the layout has patch 16, merge 0 and 2 frames; each must be at least 1",
            ),
            (
                PatchLayout {
                    frames: 0,
                    ..layout
                },
                "image size limits: the layout has patch 16, merge 2 and 0 frames; each must be at least 1",
            ),
            (
                PatchLayout { patch: 0, ..layout },
                "image size limits: the layout has patch 0, merge 2 and 2 frames; each must be at least 1",
            ),
        ] {
            let err = patchify(&img, plan(64, 32), bad).expect_err("zero layout");
            assert!(matches!(err, VisionError::Limits { .. }), "{bad:?}: {err}");
            assert_eq!(err.to_string(), text);
        }
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        // 1 + 2^-8 is halfway between two bf16 values; the even one is 1.0.
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8001)), 0x3F81);
        assert_eq!(f32_to_bf16(-2.0), 0xC000);
        assert_eq!(f32_to_bf16(f32::NAN), 0x7FC0);
    }

    /// 0 maps to -1, 255 to 1, and the grey fill 127 to just below 0.
    #[test]
    fn normalize_ends() {
        assert_eq!(normalize(0), f32_to_bf16(-1.0));
        assert_eq!(normalize(255), f32_to_bf16(1.0));
        assert!(f32::from_bits(u32::from(normalize(127)) << 16) < 0.0);
    }
}
