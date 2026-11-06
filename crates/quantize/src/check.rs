//! `--check N`: N rows of every quantized tensor run through the host rule (`gguf::iq_quant`, ggml's bytes by
//! `gate-gpu-iq-quant`) as well, and the first byte that differs from the backend's stops the run by name.

use crate::scan::row_bytes;
use crate::{Error, Result};
use gguf::iq_quant::{GridTables, IqKind, quantize_row};
use gguf::quant::{GgmlType, dequant_row};

/// The rows `--check n` runs of a tensor of `total` rows: `n` of them spread evenly from the first to the last
/// (every row when `n` covers them all).
#[must_use]
pub fn check_rows(total: usize, n: usize) -> Vec<usize> {
    if n == 0 || total == 0 {
        return Vec::new();
    }
    if n >= total {
        return (0..total).collect();
    }
    if n == 1 {
        return vec![0];
    }
    let mut rows: Vec<usize> = (0..n).map(|i| i * (total - 1) / (n - 1)).collect();
    rows.dedup();
    rows
}

/// `got` against `want`, one row's bytes: the first byte that differs is refused by tensor, expert, row, super-block
/// and byte.
pub fn compare_row(
    tensor: &str,
    kind: IqKind,
    row: usize,
    rows_per_expert: usize,
    got: &[u8],
    want: &[u8],
) -> Result<()> {
    if got.len() != want.len() {
        return Err(Error::refused(format!(
            "--check: {tensor} expert {} row {}: {} bytes, the host rule's {}",
            row / rows_per_expert,
            row % rows_per_expert,
            got.len(),
            want.len()
        )));
    }
    match got.iter().zip(want).position(|(a, b)| a != b) {
        None => Ok(()),
        Some(p) => Err(Error::Mismatch {
            tensor: tensor.to_string(),
            expert: row / rows_per_expert,
            row: row % rows_per_expert,
            block: p / kind.block_bytes(),
            byte: p % kind.block_bytes(),
            got: got[p],
            want: want[p],
        }),
    }
}

/// One source row widened as ggml widens it and quantized by the host rule with its expert's importance row.
pub fn host_row(
    kind: IqKind,
    tables: &GridTables,
    ty: GgmlType,
    src_row: &[u8],
    qw_expert: &[f32],
) -> Result<Vec<u8>> {
    let ne0 = qw_expert.len();
    debug_assert_eq!(row_bytes(ty, ne0), Some(src_row.len()));
    let mut x = vec![0.0f32; ne0];
    dequant_row(ty, src_row, &mut x).map_err(|e| Error::refused(format!("{ty:?} row: {e}")))?;
    quantize_row(kind, &x, qw_expert, tables.view())
        .map_err(|e| Error::refused(format!("the host rule refuses the row: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_spread_from_first_to_last() {
        assert_eq!(check_rows(10, 0), Vec::<usize>::new());
        assert_eq!(check_rows(10, 1), vec![0]);
        assert_eq!(check_rows(10, 2), vec![0, 9]);
        assert_eq!(check_rows(10, 3), vec![0, 4, 9]);
        assert_eq!(check_rows(3, 8), vec![0, 1, 2]);
        assert!(check_rows(1000, 7).windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn one_flipped_byte_is_named() {
        let want = vec![7u8; 100];
        assert!(compare_row("t", IqKind::Iq1S, 5, 4, &want, &want).is_ok());
        let mut got = want.clone();
        got[57] ^= 0x10;
        let e =
            compare_row("blk.2.ffn_up_exps.weight", IqKind::Iq1S, 9, 4, &got, &want).unwrap_err();
        assert!(
            matches!(
                e,
                Error::Mismatch {
                    expert: 2,
                    row: 1,
                    block: 1,
                    byte: 7,
                    got: 0x17,
                    want: 0x07,
                    ..
                }
            ),
            "{e}"
        );
        assert!(
            e.to_string()
                .contains("blk.2.ffn_up_exps.weight expert 2 row 1")
        );
        assert!(compare_row("t", IqKind::Iq1S, 0, 4, &got[..99], &want).is_err());
    }
}
