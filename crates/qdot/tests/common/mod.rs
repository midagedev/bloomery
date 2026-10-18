//! Harness code the qdot gates share: the one reader of the encoder's `q8_2_x4` column.

/// A q8_2_x4 column as the kernels read it (`qdot::col_bytes`): groups of four 32-value blocks (144 bytes: four bf16
/// scales at 0, four i16 code sums at 8, four runs of 32 codes at 16), then 36-byte `block_q8_2` tails (bf16 scale at 0,
/// i16 sum at 2, codes at 4). Each value is the f64 product of its block's scale and its code, in column order.
pub fn restore_q82x4(acol: &[u8]) -> Vec<f64> {
    let (groups, rest) = acol.as_chunks::<144>();
    let (tails, left) = rest.as_chunks::<36>();
    assert!(
        left.is_empty(),
        "q8_2_x4 column of {} bytes ends in a partial block",
        acol.len()
    );
    let scale = |b: &[u8]| {
        f64::from(f32::from_bits(
            u32::from(u16::from_le_bytes([b[0], b[1]])) << 16,
        ))
    };
    let mut out = Vec::with_capacity(groups.len() * 128 + tails.len() * 32);
    for g in groups {
        for ir in 0..4 {
            let d = scale(&g[2 * ir..]);
            out.extend(
                g[16 + 32 * ir..][..32]
                    .iter()
                    .map(|&q| d * f64::from(q as i8)),
            );
        }
    }
    for b in tails {
        let d = scale(b);
        out.extend(b[4..].iter().map(|&q| d * f64::from(q as i8)));
    }
    out
}
