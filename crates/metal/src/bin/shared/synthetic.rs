//! The synthetic quantized weights a probe or a clause builds when the
//! fixture's own tensors are not what it wants: random blocks with the
//! scales (and the mins, the biases) small and finite, so a kernel's
//! finiteness checks take them.

use bloomery_metal::kernels::Quant;

use super::oracle::Rng;

/// One matrix of random blocks: random bytes, with the scales (and the
/// mins, the biases) small and finite — an f16 pair in [0.0005, 0.004] for
/// Q4_K, Q6_K and Q8_0, a BF16 pair for A4's and A8's groups (an f16 pair for
/// A4H's and A8H's), and an f16 pair in
/// [0.0001, 0.0008] for Q5_K: under its 6-bit sub-scales and 5-bit codes the
/// wider range's weights reach 0.004 · 63 · 31, and their pair outputs pass
/// the half range a GEMM's f16 operand takes.
pub fn matrix(quant: Quant, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    // F32 rows are the values themselves: small finite floats, not random
    // bits (which would be NaNs and infinities).
    if quant == Quant::F32 {
        let mut rng = Rng::new(seed);
        return (0..rows * cols)
            .flat_map(|_| (0.1 * rng.signed()).to_le_bytes())
            .collect();
    }
    let block = quant.block_bytes();
    let blocks = rows * cols / quant.values_per_block();
    let mut rng = Rng::new(seed);
    let mut bytes = vec![0u8; blocks * block];
    for chunk in bytes.chunks_mut(8) {
        let word = rng.next().to_le_bytes();
        let n = chunk.len();
        chunk.copy_from_slice(&word[..n]);
    }
    let small_f16 = |rng: &mut Rng| {
        let unit = (rng.signed() + 1.0) / 2.0;
        gguf::quant::f32_to_f16_bits(0.0005 + 0.0035 * unit).to_le_bytes()
    };
    let k5_f16 = |rng: &mut Rng| {
        let unit = (rng.signed() + 1.0) / 2.0;
        gguf::quant::f32_to_f16_bits(0.0001 + 0.0007 * unit).to_le_bytes()
    };
    let small_bf16 = |rng: &mut Rng| {
        let unit = (rng.signed() + 1.0) / 2.0;
        gguf::quant::fp32_to_bf16_bits(0.0005 + 0.0035 * unit).to_le_bytes()
    };
    let zero_bias = gguf::quant::fp32_to_bf16_bits(0.0).to_le_bytes();
    let zero_bias_f16 = gguf::quant::f32_to_f16_bits(0.0).to_le_bytes();
    for b in bytes.chunks_exact_mut(block) {
        match quant {
            Quant::Q4K => {
                b[0..2].copy_from_slice(&small_f16(&mut rng));
                b[2..4].copy_from_slice(&small_f16(&mut rng));
            }
            Quant::Q5K => {
                b[0..2].copy_from_slice(&k5_f16(&mut rng));
                b[2..4].copy_from_slice(&k5_f16(&mut rng));
            }
            Quant::Q6K => b[208..210].copy_from_slice(&small_f16(&mut rng)),
            Quant::Q8_0 => b[0..2].copy_from_slice(&small_f16(&mut rng)),
            // A group: the scale and the bias at its end, both BF16, so the
            // reader's finiteness check takes the matrix. A8's codes span
            // the byte, so its values reach 0.004 · 255 where A4's reach
            // 0.004 · 15.
            Quant::A4 => {
                b[32..34].copy_from_slice(&small_bf16(&mut rng));
                b[34..36].copy_from_slice(&zero_bias);
            }
            Quant::A8 => {
                b[64..66].copy_from_slice(&small_bf16(&mut rng));
                b[66..68].copy_from_slice(&zero_bias);
            }
            // The same groups with the pair as F16 bits.
            Quant::A4H => {
                b[32..34].copy_from_slice(&small_f16(&mut rng));
                b[34..36].copy_from_slice(&zero_bias_f16);
            }
            Quant::A8H => {
                b[64..66].copy_from_slice(&small_f16(&mut rng));
                b[66..68].copy_from_slice(&zero_bias_f16);
            }
            Quant::F32 => unreachable!("F32 rows are built above"),
        }
    }
    bytes
}
