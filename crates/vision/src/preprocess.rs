//! From an RGB image to the ViT's input: the reference `load_image` after the decode.
//!
//! The image is padded (or resized) to the plan's pixel size with [`crate::resample::pad`], every
//! byte `v` becomes `((v / 255) - 0.5) / 0.5` in f32 — the three operations of the reference, each
//! rounded to f32 as torch rounds them — and that f32 is rounded to bf16 to nearest, ties to even
//! (torch's `.to(torch.bfloat16)`). Patches are cut row-major over the patch grid; inside a patch
//! the order is channel, then pixel row, then pixel column (the reference's
//! `reshape(3, h, p, w, p).permute(1, 3, 0, 2, 4)`), so one patch is 3·p² values.

use crate::VisionError;
use crate::grid::{GridParams, GridPlan, plan_image_grid};
use crate::image::Rgb8;
use crate::resample::{PAD_GREY, pad};

/// The ViT's input for one image: `n_vit_h * n_vit_w` patches of `3 * patch * patch` bf16 values
/// each, as raw bf16 bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Patches {
    pub plan: GridPlan,
    pub n_vit_h: usize,
    pub n_vit_w: usize,
    /// Values per patch, `3 * patch * patch`.
    pub patch_len: usize,
    /// `n_vit_h * n_vit_w * patch_len` bf16 bit patterns, patch-major.
    pub bf16: Vec<u16>,
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

/// Pad an image to its plan's pixel size, grey-filled, as the reference does.
pub fn padded(image: &Rgb8, plan: &GridPlan) -> Result<Rgb8, VisionError> {
    pad(image, plan.best_w, plan.best_h, PAD_GREY)
}

/// Cut an image already at a multiple of `patch` into normalized patches.
pub fn to_patches(image: &Rgb8, plan: GridPlan, p: &GridParams) -> Result<Patches, VisionError> {
    let s = p.patch;
    if image.width != plan.best_w
        || image.height != plan.best_h
        || !plan.best_w.is_multiple_of(s)
        || !plan.best_h.is_multiple_of(s)
    {
        return Err(VisionError::Size {
            width: image.width,
            height: image.height,
            detail: format!(
                "the plan is {}x{} in {s}-pixel patches",
                plan.best_w, plan.best_h
            ),
        });
    }
    let table: Vec<u16> = (0..=255u8).map(normalize).collect();
    let (n_vit_h, n_vit_w) = (plan.n_vit_h(p), plan.n_vit_w(p));
    let patch_len = 3 * s * s;
    let mut bf16 = Vec::with_capacity(n_vit_h * n_vit_w * patch_len);
    for ph in 0..n_vit_h {
        for pw in 0..n_vit_w {
            for c in 0..3 {
                for py in 0..s {
                    let row = ((ph * s + py) * image.width + pw * s) * 3;
                    for px in 0..s {
                        bf16.push(table[usize::from(image.data[row + px * 3 + c])]);
                    }
                }
            }
        }
    }
    Ok(Patches {
        plan,
        n_vit_h,
        n_vit_w,
        patch_len,
        bf16,
    })
}

/// The reference `load_image` from a decoded image: plan, pad, normalize, patch.
pub fn preprocess(image: &Rgb8, p: &GridParams) -> Result<Patches, VisionError> {
    let plan = plan_image_grid(image.width, image.height, p);
    to_patches(&padded(image, &plan)?, plan, p)
}

#[cfg(test)]
mod tests {
    use super::{f32_to_bf16, normalize};

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
