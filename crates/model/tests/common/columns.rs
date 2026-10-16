//! The activation columns and the bit comparison the host-tier gates share.

/// `n` seeded values in [-1, 1), every 61st scaled by 8 so a block's scale
/// is set by an outlier the way real activations set it.
pub fn seeded(n: usize, key: u64) -> Vec<f32> {
    let mut s = key.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|i| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = ((s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / 8_388_608.0 - 1.0;
            if i % 61 == 0 { u * 8.0 } else { u }
        })
        .collect()
}

/// Cells of `got` whose bits differ from `want`, a length mismatch counting
/// every cell.
pub fn diff_cells(got: &[f32], want: &[f32]) -> usize {
    if got.len() != want.len() {
        return got.len().max(want.len());
    }
    got.iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}
