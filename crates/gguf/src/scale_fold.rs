//! A per-expert tensor scale folded into a stack's own blocks.
//!
//! Unsloth's dynamic files keep the NVFP4 converter's `<stem>.scale` (one f32 an expert) beside a
//! fused gate·up stack that holds i-quant or K-quant blocks: the stack's values are those blocks'
//! values times the expert's scale. Every such block type carries its magnitude in f16 scalars at
//! fixed byte offsets (`d`, and `dmin` or `m` where the type has a minimum), and its decode is
//! linear in them, so multiplying the scalars by the expert's scale gives the blocks of the scaled
//! stack: the card kernels, which read no scale for these types, run the folded bytes as they run
//! any file's.
//!
//! The product is rounded to f16 (to nearest, ties to even, the one rounding [`f32_to_f16_bits`]
//! makes), a relative error of at most 2^-11 a scalar. A product past f16's range is refused by
//! name; one below its smallest subnormal rounds to zero (a block that small contributes under
//! 2^-24 of the scale's own magnitude).

use crate::quant::{GgmlType, f32_to_f16_bits, half_to_f32};

/// The byte offsets of a block's f16 scalars, `None` for a type the fold cannot express: its
/// magnitude is not an f16 at a fixed offset (IQ1_M packs its f16 across the high nibbles of its
/// scales; MXFP4 and NVFP4 hold power-of-two or E4M3 scales; the f32 and bf16 forms hold no block
/// scale).
const fn scalar_offsets(ty: GgmlType) -> Option<&'static [usize]> {
    match ty {
        GgmlType::Q8_0
        | GgmlType::Q4_0
        | GgmlType::Q5_0
        | GgmlType::IQ4_NL
        | GgmlType::IQ4_XS
        | GgmlType::IQ1_S
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_S
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S => Some(&[0]),
        GgmlType::Q4_1 | GgmlType::Q5_1 | GgmlType::Q4_K | GgmlType::Q5_K => Some(&[0, 2]),
        GgmlType::Q6_K => Some(&[208]),
        GgmlType::Q3_K => Some(&[108]),
        GgmlType::Q2_K => Some(&[80, 82]),
        _ => None,
    }
}

impl GgmlType {
    /// Whether a `<stem>.scale` beside a tensor of this type can be folded into its blocks
    /// ([`fold_scales`]). A file that holds the scale beside another type is refused by name.
    #[must_use]
    pub const fn folds_scale(self) -> bool {
        scalar_offsets(self).is_some()
    }
}

/// Why a fold was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoldError(pub String);

impl std::fmt::Display for FoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FoldError {}

/// Multiply the blocks of `bytes` by the scales in place: `per_scale` bytes (whole blocks of `ty`)
/// for each of `scales`, in order — an expert's rows for a stack's per-expert scales, the whole
/// tensor for a lone scale. Refused by name, before any byte changes: a type with no f16 scalar to
/// scale ([`GgmlType::folds_scale`]), a length that is not `per_scale` bytes a scale, a
/// `per_scale` that is not whole blocks, a scale that is not finite, and a block whose scaled
/// scalar passes f16's range.
///
/// # Errors
///
/// As above.
pub fn fold_scales(
    ty: GgmlType,
    bytes: &mut [u8],
    per_scale: usize,
    scales: &[f32],
) -> Result<(), FoldError> {
    let refuse = |detail: String| FoldError(format!("a {ty} stack's scale fold: {detail}"));
    let Some(offsets) = scalar_offsets(ty) else {
        return Err(refuse(
            "the type holds no f16 scalar a scale multiplies (a fold takes the legacy \
             quants, the K-quants and the i-quants but IQ1_M)"
                .to_string(),
        ));
    };
    let block = ty
        .type_size()
        .and_then(|s| usize::try_from(s).ok())
        .filter(|&s| s > 0)
        .ok_or_else(|| refuse("the type has no block size".to_string()))?;
    if per_scale == 0 || !per_scale.is_multiple_of(block) {
        return Err(refuse(format!(
            "{per_scale} bytes a scale are not whole blocks of {block}"
        )));
    }
    if scales.len().checked_mul(per_scale) != Some(bytes.len()) {
        return Err(refuse(format!(
            "{} scales of {per_scale} bytes do not cover {} bytes",
            scales.len(),
            bytes.len()
        )));
    }
    if let Some((i, s)) = scales.iter().enumerate().find(|(_, s)| !s.is_finite()) {
        return Err(refuse(format!("scale {i} is {s}, not a finite value")));
    }
    // Every product first: a refusal leaves the bytes as they were.
    let scalar = |b: &[u8], at: usize| u16::from_le_bytes([b[at], b[at + 1]]);
    for (i, (run, &s)) in bytes.chunks_exact(per_scale).zip(scales).enumerate() {
        for (n, b) in run.chunks_exact(block).enumerate() {
            for &at in offsets {
                let v = half_to_f32(scalar(b, at)) * s;
                if f32_to_f16_bits(v) & 0x7c00 == 0x7c00 {
                    return Err(refuse(format!(
                        "scale {i} ({s}) takes block {n}'s scalar at byte {at} to {v}, past f16's \
                         range"
                    )));
                }
            }
        }
    }
    for (run, &s) in bytes.chunks_exact_mut(per_scale).zip(scales) {
        for b in run.chunks_exact_mut(block) {
            for &at in offsets {
                let v = half_to_f32(scalar(b, at)) * s;
                b[at..at + 2].copy_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{fold_scales, scalar_offsets};
    use crate::quant::{GgmlType, dequant_row, f32_to_f16_bits, half_to_f32};

    /// Every type the fold takes, with blocks of seeded bytes whose scalars are set to values
    /// of a normal magnitude: the folded blocks decode to the scale times the original's decode
    /// within the f16 rounding of the scalars (2^-10 of the largest value of the block), and
    /// every byte that is not a scalar is the original's.
    #[test]
    fn a_fold_decodes_to_the_scale_times_the_original() {
        let types: Vec<GgmlType> = (0..=255u32)
            .map(GgmlType::from_u32)
            .filter(|t| t.folds_scale())
            .collect();
        assert!(types.len() >= 15, "the fold's types: {types:?}");
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for ty in types {
            let offsets = scalar_offsets(ty).unwrap();
            let block = usize::try_from(ty.type_size().unwrap()).unwrap();
            let blck = usize::try_from(ty.blck_size().unwrap()).unwrap();
            let blocks_per_row = 2usize;
            let row_bytes = block * blocks_per_row;
            let experts = 3usize;
            let rows_each = 2usize;
            let per = row_bytes * rows_each;
            let mut bytes: Vec<u8> = (0..per * experts).map(|_| (next() >> 24) as u8).collect();
            for b in bytes.chunks_exact_mut(block) {
                for &at in offsets {
                    let d = (next() % 1000) as f32 / 1000.0 * 0.02 + 0.001;
                    b[at..at + 2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
                }
            }
            let scales = [0.37f32, 1.9, 0.0123];
            let mut folded = bytes.clone();
            fold_scales(ty, &mut folded, per, &scales).unwrap();
            for (i, (a, b)) in bytes
                .chunks_exact(block)
                .zip(folded.chunks_exact(block))
                .enumerate()
            {
                let s = scales[i * block / per];
                for (at, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                    let sc = offsets.iter().find(|&&o| at == o || at == o + 1);
                    match sc {
                        None => assert_eq!(x, y, "{ty}: byte {at} of block {i} is no scalar"),
                        Some(&o) => {
                            if at == o {
                                let want = f32_to_f16_bits(
                                    half_to_f32(u16::from_le_bytes([a[o], a[o + 1]])) * s,
                                );
                                assert_eq!(
                                    u16::from_le_bytes([b[o], b[o + 1]]),
                                    want,
                                    "{ty}: scalar at {o} of block {i}"
                                );
                            }
                        }
                    }
                }
            }
            if !ty.has_dequant() {
                continue;
            }
            let k = blck * blocks_per_row;
            for (e, &scale) in scales.iter().enumerate().take(experts) {
                for r in 0..rows_each {
                    let at = e * per + r * row_bytes;
                    let (mut want, mut got) = (vec![0f32; k], vec![0f32; k]);
                    dequant_row(ty, &bytes[at..at + row_bytes], &mut want).unwrap();
                    dequant_row(ty, &folded[at..at + row_bytes], &mut got).unwrap();
                    let peak = want.iter().fold(0f32, |m, v| m.max(v.abs())) * scale;
                    for (w, g) in want.iter().zip(&got) {
                        assert!(
                            (w * scale - g).abs() <= peak * 2f32.powi(-9) + 1e-30,
                            "{ty}: expert {e} row {r}: {} against {g}",
                            w * scale
                        );
                    }
                }
            }
        }
    }

    /// A refusal names its cause and leaves the bytes as they were: a type with no scalar
    /// (IQ1_M, MXFP4, NVFP4, f32), bytes that are not whole blocks or not covered by the scales,
    /// a scale that is not finite, and a product past f16's range, found in the second scale's run.
    #[test]
    fn a_fold_refuses_by_name_and_changes_nothing() {
        let block = 98usize; // IQ3_XXS
        let mut bytes = vec![0u8; block * 4];
        for b in bytes.chunks_exact_mut(block) {
            b[0..2].copy_from_slice(&f32_to_f16_bits(0.01).to_le_bytes());
        }
        // The third block's scalar is large: scale 10 takes it past 65504.
        bytes[2 * block..2 * block + 2].copy_from_slice(&f32_to_f16_bits(60000.0).to_le_bytes());
        let before = bytes.clone();
        let ty = GgmlType::IQ3_XXS;
        let err = |r: Result<(), super::FoldError>| r.unwrap_err().to_string();
        assert!(
            err(fold_scales(GgmlType::IQ1_M, &mut bytes, 56, &[1.0])).contains("no f16 scalar")
        );
        assert!(err(fold_scales(GgmlType::F32, &mut bytes, 4, &[1.0])).contains("no f16 scalar"));
        assert!(err(fold_scales(ty, &mut bytes, 100, &[1.0; 4])).contains("not whole blocks"));
        assert!(err(fold_scales(ty, &mut bytes, block * 2, &[1.0])).contains("do not cover"));
        assert!(
            err(fold_scales(
                ty,
                &mut bytes,
                block,
                &[1.0, f32::NAN, 1.0, 1.0]
            ))
            .contains("not a finite value")
        );
        let over = err(fold_scales(ty, &mut bytes, block * 2, &[1.0, 10.0]));
        assert!(
            over.contains("past f16's range") && over.contains("block 0"),
            "{over}"
        );
        assert_eq!(bytes, before, "a refused fold leaves the bytes");
    }
}
