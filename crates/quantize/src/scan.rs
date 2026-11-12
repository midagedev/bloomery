//! The finite check of every source row, before a row reaches a quantizer: an Inf or NaN value, an MXFP4
//! block whose E8M0 scale byte is 255 (Inf), or an FP8_B128 block whose e4m3fn code is a NaN one (0x7F
//! or 0xFF; e4m3fn has no Inf) or whose f32 scale is not finite or is negative, is refused by tensor,
//! expert and row. The card's own status word catches a NaN; an Inf is a number to it, and the bytes it
//! would write are no longer ggml's meaning.

use crate::{Error, Result};
use gguf::quant::GgmlType;

/// The bytes of one row of `ne0` values of `ty`, or `None` for a type no quantizer here reads.
#[must_use]
pub fn row_bytes(ty: GgmlType, ne0: usize) -> Option<usize> {
    let (blck, size) = (ty.blck_size()? as usize, ty.type_size()? as usize);
    (ne0.is_multiple_of(blck)).then(|| ne0 / blck * size)
}

/// The source types the card and the host rule read, and the oracle (`tools/ref/iq_quant_ref.cpp`) covers.
pub const SOURCE_TYPES: [GgmlType; 5] = [
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::BF16,
    GgmlType::MXFP4,
    GgmlType::FP8_B128,
];

/// The first non-finite value of one row: its index and what it is, the what naming the block a
/// block-scaled type's value lies in.
fn bad_value(ty: GgmlType, row: &[u8]) -> Option<(usize, String)> {
    match ty {
        GgmlType::F32 => row
            .as_chunks::<4>()
            .0
            .iter()
            .enumerate()
            .find_map(|(i, b)| {
                let bits = u32::from_le_bytes(*b);
                ((bits >> 23) & 0xff == 0xff).then_some({
                    (
                        i,
                        if bits & 0x007f_ffff == 0 {
                            "Inf"
                        } else {
                            "NaN"
                        },
                    )
                })
            })
            .map(|(i, what)| (i, what.to_string())),
        GgmlType::F16 => row
            .as_chunks::<2>()
            .0
            .iter()
            .enumerate()
            .find_map(|(i, b)| {
                let bits = u16::from_le_bytes(*b);
                ((bits >> 10) & 0x1f == 0x1f)
                    .then_some((i, if bits & 0x03ff == 0 { "Inf" } else { "NaN" }))
            })
            .map(|(i, what)| (i, what.to_string())),
        GgmlType::BF16 => row
            .as_chunks::<2>()
            .0
            .iter()
            .enumerate()
            .find_map(|(i, b)| {
                let bits = u16::from_le_bytes(*b);
                ((bits >> 7) & 0xff == 0xff)
                    .then_some((i, if bits & 0x007f == 0 { "Inf" } else { "NaN" }))
            })
            .map(|(i, what)| (i, what.to_string())),
        GgmlType::MXFP4 => row
            .as_chunks::<17>()
            .0
            .iter()
            .position(|blk| blk[0] == 255)
            .map(|b| (b * 32, "Inf (E8M0 scale byte 255)".to_string())),
        // One block is its f32 scale then its 128 e4m3fn codes: a NaN code (byte 0x7F or 0xFF;
        // e4m3fn has no Inf), a scale that is not finite or a negative one is not a value a row
        // of the checkpoint holds. A zero scale stays: an all-zero block may carry one.
        GgmlType::FP8_B128 => row
            .as_chunks::<132>()
            .0
            .iter()
            .enumerate()
            .find_map(|(b, blk)| {
                let scale = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
                if !scale.is_finite() {
                    return Some((128 * b, format!("{scale} (block {b}'s scale)")));
                }
                if scale < 0.0 {
                    return Some((128 * b, format!("negative (block {b}'s scale is {scale})")));
                }
                blk[4..]
                    .iter()
                    .position(|&c| c & 0x7F == 0x7F)
                    .map(|j| (128 * b + j, format!("NaN (e4m3fn code of block {b})")))
            }),
        _ => None,
    }
}

/// Every row of `src` (whole rows of `ne0` values of `ty`, the first of them row `row0` of the tensor, `rows_per_expert`
/// rows an expert) is finite, or the first that is not is refused by name.
pub fn scan_rows(
    tensor: &str,
    ty: GgmlType,
    src: &[u8],
    ne0: usize,
    row0: usize,
    rows_per_expert: usize,
) -> Result<()> {
    let rb = row_bytes(ty, ne0).ok_or_else(|| {
        Error::refused(format!(
            "{tensor}: {ty:?} rows of {ne0} values are not read"
        ))
    })?;
    for (r, row) in src.chunks_exact(rb).enumerate() {
        if let Some((i, what)) = bad_value(ty, row) {
            let g = row0 + r;
            return Err(Error::refused(format!(
                "{tensor}: expert {} row {} value {i} is {what} ({ty:?}): a source row must be finite",
                g / rows_per_expert,
                g % rows_per_expert
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_type_names_its_first_bad_value() {
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let e = scan_rows(
            "t",
            GgmlType::F32,
            &f32s(&[1.0, 2.0, 3.0, f32::INFINITY]),
            2,
            4,
            3,
        )
        .unwrap_err()
        .to_string();
        assert!(
            e.starts_with("t: expert 1 row 2 value 1 is Inf (F32)"),
            "{e}"
        );
        let e = scan_rows("t", GgmlType::F32, &f32s(&[1.0, f32::NAN]), 2, 0, 3)
            .unwrap_err()
            .to_string();
        assert!(e.contains("expert 0 row 0 value 1 is NaN"), "{e}");
        assert!(scan_rows("t", GgmlType::F32, &f32s(&[1.0, -2.0, 0.0, 1e30]), 2, 0, 3).is_ok());
        // F16 Inf 0x7c00, NaN 0x7e00; BF16 Inf 0x7f80
        let h = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        assert!(scan_rows("t", GgmlType::F16, &h(&[0x3c00, 0x7c00]), 2, 0, 1).is_err());
        assert!(scan_rows("t", GgmlType::F16, &h(&[0x3c00, 0x7bff]), 2, 0, 1).is_ok());
        assert!(scan_rows("t", GgmlType::BF16, &h(&[0x3f80, 0xff80]), 2, 0, 1).is_err());
        assert!(scan_rows("t", GgmlType::BF16, &h(&[0x3f80, 0x7f7f]), 2, 0, 1).is_ok());
        // MXFP4: two blocks of a row, the second's scale byte 255
        let mut mx = vec![0u8; 34];
        mx[0] = 126;
        mx[17] = 255;
        let e = scan_rows("t", GgmlType::MXFP4, &mx, 64, 0, 1)
            .unwrap_err()
            .to_string();
        assert!(e.contains("value 32 is Inf (E8M0 scale byte 255)"), "{e}");
        mx[17] = 254;
        assert!(scan_rows("t", GgmlType::MXFP4, &mx, 64, 0, 1).is_ok());
    }

    /// An FP8_B128 row holds each block as its f32 scale then its 128 e4m3fn codes: a NaN code
    /// (0x7F or 0xFF; e4m3fn has no Inf), a scale that is not finite and a negative one are
    /// refused by tensor, expert, row and block, and a zero scale stays (an all-zero block may
    /// carry one). Mutants: the code check dropped, the scale checks dropped, zero refused.
    #[test]
    fn an_fp8_row_names_its_first_bad_code_or_scale() {
        let block = |scale: f32, code: u8| {
            let mut b = scale.to_le_bytes().to_vec();
            b.extend(std::iter::repeat_n(code, 128));
            b
        };
        let row = |blocks: &[Vec<u8>]| blocks.concat();
        // Two whole blocks: 256 values, 264 bytes.
        let ok = row(&[block(1.0, 0x30), block(0.0, 0x00)]);
        assert_eq!(ok.len(), 264);
        assert!(scan_rows("t", GgmlType::FP8_B128, &ok, 256, 0, 1).is_ok());
        // A NaN code in the second block, positive and negative.
        let mut bad = row(&[block(1.0, 0x30), block(1.0, 0x30)]);
        bad[132 + 4 + 5] = 0x7F;
        let e = scan_rows("t", GgmlType::FP8_B128, &bad, 256, 0, 1)
            .unwrap_err()
            .to_string();
        assert!(
            e.starts_with("t: expert 0 row 0 value 133 is NaN (e4m3fn code of block 1)"),
            "{e}"
        );
        bad[132 + 4 + 5] = 0xFF;
        let e = scan_rows("t", GgmlType::FP8_B128, &bad, 256, 0, 1)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("value 133 is NaN (e4m3fn code of block 1)"),
            "{e}"
        );
        // 0x7E is finite: the largest e4m3fn magnitude the check lets through.
        let top = row(&[block(1.0, 0x7E)]);
        assert!(scan_rows("t", GgmlType::FP8_B128, &top, 128, 0, 1).is_ok());
        // A scale that is not finite names its block.
        let nan = row(&[block(f32::NAN, 0x30), block(1.0, 0x30)]);
        let e = scan_rows("t", GgmlType::FP8_B128, &nan, 256, 0, 1)
            .unwrap_err()
            .to_string();
        assert!(e.contains("value 0 is NaN (block 0's scale)"), "{e}");
        // A negative scale names its block and its value.
        let neg = row(&[block(1.0, 0x30), block(-3.5, 0x30)]);
        let e = scan_rows("t", GgmlType::FP8_B128, &neg, 256, 0, 1)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("expert 0 row 0 value 128 is negative (block 1's scale is -3.5)"),
            "{e}"
        );
        // A row a later expert holds names its expert and row.
        let mut shifted = row(&[block(1.0, 0x30)]);
        shifted[6] = 0xFF;
        let e = scan_rows("t", GgmlType::FP8_B128, &shifted, 128, 3, 2)
            .unwrap_err()
            .to_string();
        assert!(e.contains("expert 1 row 1 value 2 is NaN"), "{e}");
    }

    #[test]
    fn row_bytes_follow_the_block_tables() {
        assert_eq!(row_bytes(GgmlType::F32, 512), Some(2048));
        assert_eq!(row_bytes(GgmlType::BF16, 512), Some(1024));
        assert_eq!(row_bytes(GgmlType::MXFP4, 512), Some(272));
        assert_eq!(row_bytes(GgmlType::MXFP4, 48), None);
        assert_eq!(row_bytes(GgmlType::FP8_B128, 256), Some(264));
        assert_eq!(row_bytes(GgmlType::FP8_B128, 6144), Some(48 * 132));
        assert_eq!(row_bytes(GgmlType::FP8_B128, 100), None);
    }
}
