//! Narrowing a tensor of the file to the bf16 the card's GEMM reads.
//!
//! The tower computes from bf16 weights. A file's matrices already are bf16 (a bf16 export) or f16
//! (an f16 export), and the patch kernels of a bf16 export are f32, so some tensors must be
//! narrowed on upload. A narrowing that rounds a value is a different network, so it is done only
//! when every value comes back unchanged through bf16 and f32; a tensor with one that does not,
//! or a value that is not finite, is refused by its name and the index of the first such value.

use gguf::quant::half_to_f32;

use crate::VisionError;
use crate::preprocess::f32_to_bf16 as round_to_bf16;

/// The bf16 value's f32.
fn widen(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// One f32 narrowed to bf16 bits, `None` for a value that does not come back unchanged.
fn exact(v: f32) -> Option<u16> {
    if !v.is_finite() {
        return None;
    }
    let bits = round_to_bf16(v);
    (widen(bits).to_bits() == v.to_bits()).then_some(bits)
}

fn refusal(name: &str, index: usize, v: f32, from: &str) -> VisionError {
    let detail = if v.is_finite() {
        format!(
            "value {index} is {v:e} ({from}), which bf16 holds as {:e}; the upload narrows only a \
             tensor whose every value round-trips",
            widen(round_to_bf16(v))
        )
    } else {
        format!("value {index} is {v} ({from}); the upload narrows only finite values")
    };
    VisionError::Tensor {
        name: name.to_string(),
        detail,
    }
}

/// The bf16 bits of an f32 tensor `name`, every value exact.
pub fn f32_to_bf16(name: &str, values: &[f32]) -> Result<Vec<u16>, VisionError> {
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| exact(v).ok_or_else(|| refusal(name, i, v, "f32")))
        .collect()
}

/// The bf16 bits of an f16 tensor `name` (given as f16 bits), every value exact.
pub fn f16_to_bf16(name: &str, bits: &[u16]) -> Result<Vec<u16>, VisionError> {
    bits.iter()
        .enumerate()
        .map(|(i, &b)| {
            let v = half_to_f32(b);
            exact(v).ok_or_else(|| refusal(name, i, v, "f16"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{f16_to_bf16, f32_to_bf16};

    /// bf16 values narrow to their own bits, the sign of zero included.
    #[test]
    fn values_bf16_holds_narrow_to_their_bits() {
        let v = [
            0.0,
            -0.0,
            1.0,
            -2.5,
            0.5,
            f32::from_bits(0x7F6C_0000),
            f32::from_bits(0x0001_0000),
        ];
        let got = f32_to_bf16("v.patch_embd.weight", &v).expect("every value is a bf16");
        assert_eq!(
            got,
            [0x0000, 0x8000, 0x3F80, 0xC020, 0x3F00, 0x7F6C, 0x0001]
        );
    }

    /// One value that bf16 would round refuses the whole tensor, by name and first index.
    #[test]
    fn a_value_that_would_round_is_refused_by_name_and_index() {
        // 1 + 2^-10 needs ten significand bits; bf16 has seven.
        let mut v = vec![1.0_f32; 40];
        v[17] = 1.0 + 0.000_976_562_5;
        v[30] = 1.0 + 0.000_976_562_5;
        let err = f32_to_bf16("v.patch_embd.weight.1", &v).unwrap_err();
        assert_eq!(
            err.to_string(),
            "tensor v.patch_embd.weight.1: value 17 is 1.0009766e0 (f32), which bf16 holds as \
             1e0; the upload narrows only a tensor whose every value round-trips"
        );
    }

    /// A value that is not finite is refused, though bf16 can hold an infinity.
    #[test]
    fn a_value_that_is_not_finite_is_refused() {
        for (v, shown) in [(f32::NAN, "NaN"), (f32::INFINITY, "inf")] {
            let err = f32_to_bf16("t", &[1.0, v]).unwrap_err().to_string();
            assert_eq!(
                err,
                format!(
                    "tensor t: value 1 is {shown} (f32); the upload narrows only finite values"
                )
            );
        }
    }

    /// f16 holds eleven significand bits, bf16 eight: 1 + 2^-8 is an f16 (0x3C04) and not a bf16,
    /// while 1 + 2^-7 is both (f16 0x3C08, bf16 0x3F81).
    #[test]
    fn an_f16_tensor_narrows_only_the_values_bf16_holds() {
        assert_eq!(
            f16_to_bf16("w", &[0x3C08, 0x3C00, 0xBC00]).unwrap(),
            [0x3F81, 0x3F80, 0xBF80]
        );
        let err = f16_to_bf16("w", &[0x3C00, 0x3C04]).unwrap_err().to_string();
        assert_eq!(
            err,
            "tensor w: value 1 is 1.0039063e0 (f16), which bf16 holds as 1e0; the upload narrows \
             only a tensor whose every value round-trips"
        );
    }
}
