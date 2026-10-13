//! The row order of a tile and its position rows, as pure functions of the grid: the layout the
//! tower's two mergers read as reshapes (see [`super`]) and the table row of every patch; and the
//! rows that follow a tile's merger rows in the prompt.

use super::{BLOCK_SIDE, GROUP, POS_SIDE};

/// A grid the tower runs: both sides a multiple of [`BLOCK_SIDE`] (the plan rounds every tile to
/// 56 pixels, 4 patches).
pub fn check_grid(n_h: usize, n_w: usize) -> Result<(), String> {
    if n_h == 0 || n_w == 0 || !n_h.is_multiple_of(BLOCK_SIDE) || !n_w.is_multiple_of(BLOCK_SIDE) {
        return Err(format!(
            "a {n_h}x{n_w} grid is not whole {BLOCK_SIDE}x{BLOCK_SIDE} blocks of patches"
        ));
    }
    Ok(())
}

/// The row of patch `(i, j)` of an `n_h × n_w` grid in merge order (see the module doc).
#[must_use]
pub fn merge_row(n_w: usize, i: usize, j: usize) -> usize {
    let block = (i / BLOCK_SIDE) * (n_w / BLOCK_SIDE) + j / BLOCK_SIDE;
    let q = 2 * ((i % BLOCK_SIDE) / 2) + (j % BLOCK_SIDE) / 2;
    let p = 2 * (i % 2) + j % 2;
    (block * GROUP + q) * GROUP + p
}

/// For each merge-order row, the raster index `i·n_w + j` of the patch it holds. The grid must
/// pass [`check_grid`].
#[must_use]
pub fn merge_rows(n_h: usize, n_w: usize) -> Vec<usize> {
    let mut rows = vec![0; n_h * n_w];
    for i in 0..n_h {
        for j in 0..n_w {
            rows[merge_row(n_w, i, j)] = i * n_w + j;
        }
    }
    rows
}

/// The nearest bucket of cell `i` of `n` along one side of the 70×70 table:
/// `floor(70·i / n)` (`clip.cpp:4875`, `std::floor(70.0*i/pos_h)`).
#[must_use]
pub fn bucket(i: usize, n: usize) -> usize {
    POS_SIDE * i / n
}

/// The table row of the patch at raster index `i·n_w + j`, `bucket_h·70 + bucket_w`
/// (`clip.cpp:4882`), for each patch in raster order.
#[must_use]
pub fn position_index(n_h: usize, n_w: usize) -> Vec<usize> {
    (0..n_h)
        .flat_map(|i| (0..n_w).map(move |j| bucket(i, n_h) * POS_SIDE + bucket(j, n_w)))
        .collect()
}

/// For each window row of the tower (four consecutive merge-order rows, a 2×2 window of patches),
/// the window's index in mainline's order, `wi·(n_w/2) + wj` (`clip.cpp:4890-4898`): the order of
/// the rows the down-sample leaves, which blocks 7.. and the final merger read. The grid must pass
/// [`check_grid`].
#[must_use]
pub fn window_rows(n_h: usize, n_w: usize) -> Vec<usize> {
    let rows = merge_rows(n_h, n_w);
    (0..n_h * n_w / GROUP)
        .map(|w| {
            let raster = rows[w * GROUP];
            (raster / n_w / 2) * (n_w / 2) + raster % n_w / 2
        })
        .collect()
}

/// The rows of the projector's separator table (`v.tok_embd_sep`, the converter's order:
/// `</image>`, `<slice>`, `</slice>`, `\n`).
pub const SEPARATORS: usize = 4;

/// A tile's embeddings as mainline's `embd` holds them: the tower's merger rows (`rows`, `width`
/// values each) followed by the separator rows `suffix` names, each a row of `separators` (the
/// table, [`SEPARATORS`] rows of `width`; `Encoder::separators`), in `clip_suffix_rows`' order
/// (`clip-impl.h:681-697`). A row count that is not whole, a table that is not [`SEPARATORS`]
/// rows and a suffix naming a row past the table are named errors.
pub fn tile_embd(
    rows: &[f32],
    separators: &[f32],
    width: usize,
    suffix: &[u8],
) -> Result<Vec<f32>, String> {
    if width == 0 || !rows.len().is_multiple_of(width) {
        return Err(format!(
            "{} values are not whole rows of {width}",
            rows.len()
        ));
    }
    if separators.len() != SEPARATORS * width {
        return Err(format!(
            "the separator table holds {} values, not {SEPARATORS} rows of {width}",
            separators.len()
        ));
    }
    let mut out = Vec::with_capacity(rows.len() + suffix.len() * width);
    out.extend_from_slice(rows);
    for &s in suffix {
        let s = usize::from(s);
        if s >= SEPARATORS {
            return Err(format!(
                "the suffix names separator row {s}; the table has {SEPARATORS}"
            ));
        }
        out.extend_from_slice(&separators[s * width..(s + 1) * width]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mainline's `window_idx` (`clip.cpp:4904-4913`) as written there, raster patch indices.
    fn mainline_window_idx(n_h: usize, n_w: usize) -> Vec<usize> {
        let mut idx = Vec::new();
        for wi in 0..n_h / 2 {
            for wj in 0..n_w / 2 {
                idx.push(2 * wi * n_w + 2 * wj);
                idx.push(2 * wi * n_w + 2 * wj + 1);
                idx.push((2 * wi + 1) * n_w + 2 * wj);
                idx.push((2 * wi + 1) * n_w + 2 * wj + 1);
            }
        }
        idx
    }

    /// Mainline's `make_ds_idx` (`clip.cpp:4890-4898`).
    fn make_ds_idx(
        off_r: usize,
        off_c: usize,
        ds_h: usize,
        ds_w: usize,
        stride: usize,
    ) -> Vec<usize> {
        let mut idx = Vec::new();
        for i in 0..ds_h {
            for j in 0..ds_w {
                idx.push((2 * i + off_r) * stride + 2 * j + off_c);
            }
        }
        idx
    }

    /// Both merge levels against mainline's index tensors: row `4w + p` is patch `p` of the
    /// window that the final merger's group `t` takes as its `q`-th, `w = 4t + q`.
    fn check_merge_order(n_h: usize, n_w: usize) {
        let (half_h, half_w) = (n_h / 2, n_w / 2);
        let rows = merge_rows(n_h, n_w);
        let window = mainline_window_idx(n_h, n_w);
        let ds: Vec<Vec<usize>> = [(0, 0), (0, 1), (1, 0), (1, 1)]
            .iter()
            .map(|&(r, c)| make_ds_idx(r, c, half_h / 2, half_w / 2, half_w))
            .collect();
        for t in 0..ds[0].len() {
            for (q, d) in ds.iter().enumerate() {
                let w = 4 * t + q;
                for p in 0..GROUP {
                    assert_eq!(
                        rows[w * GROUP + p],
                        window[d[t] * GROUP + p],
                        "t {t} q {q} p {p}"
                    );
                }
            }
        }
        let mut seen = rows.clone();
        seen.sort_unstable();
        assert!(
            seen.iter().enumerate().all(|(i, &v)| i == v),
            "not a permutation"
        );
    }

    /// The row map of a 32×32 slice and a 24×40 overview: the first and last group at each
    /// level by hand, and the whole map against mainline's index tensors.
    #[test]
    fn merge_order_is_mainlines() {
        // 32x32: window 0 is patches (0,0) (0,1) (1,0) (1,1); window 1 of block 0 is (0,2)...
        let rows = merge_rows(32, 32);
        assert_eq!(&rows[0..4], &[0, 1, 32, 33]);
        assert_eq!(&rows[4..8], &[2, 3, 34, 35]);
        assert_eq!(&rows[8..12], &[64, 65, 96, 97]);
        assert_eq!(&rows[12..16], &[66, 67, 98, 99]);
        // block (0,1) starts at row 16 with patch (0,4); the last block (7,7) ends the grid
        assert_eq!(rows[16], 4);
        assert_eq!(rows[1008], 28 * 32 + 28);
        assert_eq!(&rows[1020..1024], &[990, 991, 1022, 1023]);
        // 24x40: 6x10 blocks; block (5,9) is the last
        let rows = merge_rows(24, 40);
        assert_eq!(&rows[0..4], &[0, 1, 40, 41]);
        assert_eq!(rows[16], 4);
        assert_eq!(
            &rows[956..960],
            &[22 * 40 + 38, 22 * 40 + 39, 23 * 40 + 38, 23 * 40 + 39]
        );
        // vit_merger level: window 3 of the 32x32 grid is half-grid cell (1,1) = mainline's
        // window index 33; final level: 64 groups of 4 windows, group 0 is cells (0,0) (0,1) (1,0) (1,1)
        check_merge_order(32, 32);
        check_merge_order(24, 40);
        assert!(
            check_grid(24, 40).is_ok() && check_grid(30, 32).is_err() && check_grid(0, 4).is_err()
        );
    }

    /// The position index of the corner patches, and the bucket against the reference's
    /// double-precision `floor(70.0·i / n)`.
    #[test]
    fn position_rows_are_the_nearest_bucket() {
        let idx = position_index(32, 32);
        assert_eq!(idx[0], 0);
        assert_eq!(idx[31], 67); // (0, 31): floor(70·31/32) = 67
        assert_eq!(idx[31 * 32], 67 * 70);
        assert_eq!(idx[32 * 32 - 1], 67 * 70 + 67);
        let idx = position_index(24, 40);
        assert_eq!(idx[39], 68); // floor(70·39/40) = 68
        assert_eq!(idx[23 * 40], 67 * 70); // floor(70·23/24) = 67
        assert_eq!(idx[24 * 40 - 1], 67 * 70 + 68);
        for n in 1..=140usize {
            for i in 0..n {
                assert_eq!(
                    bucket(i, n),
                    (70.0 * i as f64 / n as f64).floor() as usize,
                    "{i}/{n}"
                );
            }
        }
    }

    /// The window rows against the final merger's index tensors: group `t` takes the windows
    /// `d_q[t]` as its four members, which sit at rows `4t + q`.
    #[test]
    fn window_rows_are_mainlines_down_sample_order() {
        for (n_h, n_w) in [(32usize, 32usize), (24, 40), (28, 40)] {
            let (half_h, half_w) = (n_h / 2, n_w / 2);
            let win = window_rows(n_h, n_w);
            let ds: Vec<Vec<usize>> = [(0, 0), (0, 1), (1, 0), (1, 1)]
                .iter()
                .map(|&(r, c)| make_ds_idx(r, c, half_h / 2, half_w / 2, half_w))
                .collect();
            for t in 0..ds[0].len() {
                for (q, d) in ds.iter().enumerate() {
                    assert_eq!(win[4 * t + q], d[t], "{n_h}x{n_w} t {t} q {q}");
                }
            }
            let mut seen = win.clone();
            seen.sort_unstable();
            assert!(seen.iter().enumerate().all(|(i, &v)| i == v));
        }
    }

    /// The splice: the merger rows, then the named separator rows in order, and the three named
    /// refusals.
    #[test]
    fn a_tile_is_its_rows_then_its_suffix() {
        let table: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let rows = [100.0, 101.0, 102.0, 110.0, 111.0, 112.0];
        let got = tile_embd(&rows, &table, 3, &[2, 3, 1]).unwrap();
        assert_eq!(
            got,
            [
                100.0, 101.0, 102.0, 110.0, 111.0, 112.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 3.0, 4.0,
                5.0
            ]
        );
        assert_eq!(tile_embd(&rows, &table, 3, &[]).unwrap(), rows);
        assert!(tile_embd(&rows[..5], &table, 3, &[0]).is_err());
        assert!(tile_embd(&rows, &table[..9], 3, &[0]).is_err());
        assert!(tile_embd(&rows, &table, 3, &[4]).is_err());
    }
}
