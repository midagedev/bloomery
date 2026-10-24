//! The host side of the probe's MLX reference arm (`probe_mlx_qmm.metal`):
//! the engine's A4 blocks laid out the way MLX's `quantized_matmul` takes
//! them, the activations in the operand type, and a CPU check of what the
//! reference kernel wrote, so a rate printed for the arm is the rate of a
//! kernel that computes the product.

use bloomery_metal::kernels::Operand;
use gguf::quant::{bf16_bits_to_f32, f32_to_f16_bits};

/// The reference kernel's source, compiled by the probe into a library of
/// its own.
pub const SOURCE: &str = include_str!("probe_mlx_qmm.metal");

/// The values a group holds: MLX's group size, A4's block.
const GROUP: usize = 64;

/// The bytes of a code-word row piece one A4 block holds.
const BLOCK_CODE_BYTES: usize = 32;

/// The bytes of one A4 block.
const BLOCK_BYTES: usize = 36;

/// The name of the reference kernel for `operand`.
pub fn kernel_name(operand: Operand) -> &'static str {
    match operand {
        Operand::F16 => "mlxref_qmm_t_f16",
        Operand::F32 => "mlxref_qmm_t_f32",
    }
}

/// The bytes of one value of `operand`.
pub fn elem_bytes(operand: Operand) -> usize {
    match operand {
        Operand::F16 => 2,
        Operand::F32 => 4,
    }
}

/// The three arrays MLX's `quantized_matmul` takes for a weight tensor:
/// the code words (u32, row-major), the scales and the biases (one per
/// group of a row, in the operand type).
pub struct Layout {
    pub codes: Vec<u8>,
    pub scales: Vec<u8>,
    pub biases: Vec<u8>,
}

/// The value as the operand type's bytes.
fn value_bytes(operand: Operand, v: f32) -> Vec<u8> {
    match operand {
        Operand::F16 => f32_to_f16_bits(v).to_le_bytes().to_vec(),
        Operand::F32 => v.to_le_bytes().to_vec(),
    }
}

/// MLX's layout of an A4 tensor of `rows` by `cols` values: a block's eight
/// code words are eight consecutive words of its row, its BF16 scale and
/// bias widen into the operand type.
pub fn layout(blocks: &[u8], rows: usize, cols: usize, operand: Operand) -> Layout {
    let groups = rows * cols / GROUP;
    assert_eq!(
        blocks.len(),
        groups * BLOCK_BYTES,
        "an A4 tensor of whole blocks"
    );
    let mut codes = Vec::with_capacity(groups * BLOCK_CODE_BYTES);
    let mut scales = Vec::with_capacity(groups * elem_bytes(operand));
    let mut biases = Vec::with_capacity(groups * elem_bytes(operand));
    for block in blocks.as_chunks::<BLOCK_BYTES>().0 {
        codes.extend_from_slice(&block[..BLOCK_CODE_BYTES]);
        let widen = |at: usize| bf16_bits_to_f32(u16::from_le_bytes([block[at], block[at + 1]]));
        scales.extend_from_slice(&value_bytes(operand, widen(BLOCK_CODE_BYTES)));
        biases.extend_from_slice(&value_bytes(operand, widen(BLOCK_CODE_BYTES + 2)));
    }
    Layout {
        codes,
        scales,
        biases,
    }
}

/// The activations in the operand type's bytes.
pub fn activations(x: &[f32], operand: Operand) -> Vec<u8> {
    x.iter().flat_map(|&v| value_bytes(operand, v)).collect()
}

/// The f32 of an IEEE half's bits.
fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1F);
    let frac = f32::from(bits & 0x03FF);
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        0x1F => {
            if frac == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// The value of `bytes`' `i`-th element of the operand type.
fn value_at(bytes: &[u8], i: usize, operand: Operand) -> f32 {
    match operand {
        Operand::F16 => half_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])),
        Operand::F32 => f32::from_le_bytes([
            bytes[4 * i],
            bytes[4 * i + 1],
            bytes[4 * i + 2],
            bytes[4 * i + 3],
        ]),
    }
}

/// The relative band the output may sit in, against the sum of the absolute
/// products: the half arm rounds its weights, activations and output to half.
fn band(operand: Operand) -> f64 {
    match operand {
        Operand::F16 => 4.0e-3,
        Operand::F32 => 2.0e-5,
    }
}

/// Check the reference kernel's output `y` (rows `acts`, columns `rows`, in
/// the operand type) against a CPU product at a spread of positions; the
/// first position out of its band is the error.
pub fn check(
    layout: &Layout,
    x: &[u8],
    y: &[u8],
    rows: usize,
    cols: usize,
    acts: usize,
    operand: Operand,
) -> Result<(), String> {
    let groups = cols / GROUP;
    let positions = [
        (0, 0),
        (acts - 1, rows - 1),
        (acts / 2, rows / 3),
        (acts / 3, rows - 1),
        (1 % acts, 7 % rows),
        (acts - 1, 0),
    ];
    for (r, n) in positions {
        let mut sum = 0.0f64;
        let mut abs = 0.0f64;
        for k in 0..cols {
            let at = n * groups + k / GROUP;
            let scale = f64::from(value_at(&layout.scales, at, operand));
            let bias = f64::from(value_at(&layout.biases, at, operand));
            let word = &layout.codes[(n * cols + k) / 8 * 4..][..4];
            let word = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            let code = f64::from((word >> (4 * (k % 8))) & 0xF);
            let w = scale * code + bias;
            let xv = f64::from(value_at(x, r * cols + k, operand));
            sum += w * xv;
            abs += (w * xv).abs();
        }
        let got = f64::from(value_at(y, r * rows + n, operand));
        let err = (got - sum).abs();
        if err.is_nan() || err > band(operand) * abs + 1e-6 {
            return Err(format!(
                "the reference kernel's y[{r}][{n}] is {got}, the CPU product {sum} \
                 (band {} of {abs})",
                band(operand)
            ));
        }
    }
    Ok(())
}
