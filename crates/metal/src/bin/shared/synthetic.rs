//! The synthetic quantized weights a probe or a clause builds when the
//! fixture's own tensors are not what it wants: random blocks with the
//! scales (and the mins, the biases) small and finite, so a kernel's
//! finiteness checks take them. `metal_matvec_time` holds its own copy of
//! this builder; this module is the one later bins share.

use bloomery_metal::kernels::Quant;

use super::oracle::Rng;

/// One matrix of random blocks: random bytes, with the scales (and the
/// mins, the biases) small and finite — an f16 pair in [0.0005, 0.004] for
/// Q4_K and Q6_K, a BF16 pair for A4's groups.
pub fn matrix(quant: Quant, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
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
    let small_bf16 = |rng: &mut Rng| {
        let unit = (rng.signed() + 1.0) / 2.0;
        gguf::quant::fp32_to_bf16_bits(0.0005 + 0.0035 * unit).to_le_bytes()
    };
    let zero_bias = gguf::quant::fp32_to_bf16_bits(0.0).to_le_bytes();
    for b in bytes.chunks_exact_mut(block) {
        match quant {
            Quant::Q4K => {
                b[0..2].copy_from_slice(&small_f16(&mut rng));
                b[2..4].copy_from_slice(&small_f16(&mut rng));
            }
            Quant::Q6K => b[208..210].copy_from_slice(&small_f16(&mut rng)),
            // A group: the scale and the bias at its end, both BF16, so the
            // reader's finiteness check takes the matrix.
            Quant::A4 => {
                b[32..34].copy_from_slice(&small_bf16(&mut rng));
                b[34..36].copy_from_slice(&zero_bias);
            }
        }
    }
    bytes
}
