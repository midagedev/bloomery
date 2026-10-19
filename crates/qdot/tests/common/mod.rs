//! Harness code the qdot gates share: the one reader of the encoder's `q8_2_x4` column.

/// The q8_2 encoders' bytes for one column, split per block: (codes, d bits,
/// isum), the x4 groups' layout (d at 2i, isum at 8 + 2i, codes at 16 + 32i)
/// then the 36-byte tail blocks'.
pub fn q82_blocks(col: &[u8], n_blocks: usize) -> Vec<([i8; 32], u16, i16)> {
    const STRIDE: usize = 144;
    const TAIL: usize = 36;
    let nb4 = 4 * (n_blocks / 4);
    let le16 = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
    let codes = |b: &[u8]| std::array::from_fn(|i| b[i] as i8);
    (0..n_blocks)
        .map(|b| {
            if b < nb4 {
                let (g, ir) = (&col[b / 4 * STRIDE..], b % 4);
                (
                    codes(&g[16 + 32 * ir..]),
                    le16(&g[2 * ir..]),
                    le16(&g[8 + 2 * ir..]) as i16,
                )
            } else {
                let tb = &col[nb4 / 4 * STRIDE + (b - nb4) * TAIL..];
                (codes(&tb[4..]), le16(tb), le16(&tb[2..]) as i16)
            }
        })
        .collect()
}

/// A q8_2_x4 column as the kernels read it (`qdot::col_bytes`), each value the f64 product of its block's scale and
/// its code, in column order ([`q82_blocks`] holds the layout).
pub fn restore_q82x4(acol: &[u8]) -> Vec<f64> {
    assert!(
        acol.len().is_multiple_of(36),
        "q8_2_x4 column of {} bytes ends in a partial block",
        acol.len()
    );
    q82_blocks(acol, acol.len() / 36)
        .into_iter()
        .flat_map(|(codes, bits, _)| {
            let d = f64::from(f32::from_bits(u32::from(bits) << 16));
            codes.map(|q| d * f64::from(q))
        })
        .collect()
}
