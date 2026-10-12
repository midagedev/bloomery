//! [`Canvas`]: the positions of MiniCPM-V 4.7's image group, the overview and the slices of one
//! image in one chunk.
//!
//! The reference is llama.cpp's `MTMD_POS_TYPE_CANVAS` (`tools/mtmd/mtmd.cpp`: `canvas_tile_grid`,
//! `mtmd_canvas_scale`, `mtmd_canvas_decoder_pos`, `mtmd_image_tokens_get_n_pos`). The chunk is
//! `[overview, slice (0,0), slice (0,1), …]`, each tile's tokens in raster order followed by its
//! separators (rows of `v.tok_embd_sep`), and sits right after the `<image>` token, whose position
//! is the canvas origin. The overview is stretched over the whole canvas, each slice fills its own
//! cell, and the time slot `t` runs one position a tile and one a separator. The slot `z` is the
//! origin.

use super::MropeError;
use super::order::SLOTS;

/// The rows of `v.tok_embd_sep` after a tile (`clip_suffix_rows`): 0 is `</image>`, 1 `<slice>`, 2
/// `</slice>`, 3 `\n`. The overview of an image with slices takes `</image><slice>`, an overview
/// alone `</image>`; a slice takes `</slice><slice>`, the last of a line `</slice>\n<slice>`, the
/// last of all `</slice>`.
const OVERVIEW_ALONE: &[u8] = &[0];
const OVERVIEW_WITH_SLICES: &[u8] = &[0, 1];
const SLICE_MID: &[u8] = &[2, 1];
const SLICE_LINE_END: &[u8] = &[2, 3, 1];
const SLICE_LAST: &[u8] = &[2];

/// One tile of a group: its token grid and the separator rows after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub nx: usize,
    pub ny: usize,
    /// Rows of `v.tok_embd_sep`, in order.
    pub seps: &'static [u8],
}

/// One image group: the token grids of its overview and slices, and the grid of slices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Canvas {
    overview: (usize, usize),
    grid: (usize, usize),
    slice: (usize, usize),
}

impl Canvas {
    /// A group whose overview is `overview = (nx, ny)` tokens, laid over `grid = (columns, lines)`
    /// slices of `slice = (nx, ny)` tokens each. No slice is `grid = (0, 0)` and `slice = (0, 0)`.
    pub fn new(
        overview: (usize, usize),
        grid: (usize, usize),
        slice: (usize, usize),
    ) -> Result<Canvas, MropeError> {
        let bad = |detail: &'static str| MropeError::Canvas { detail };
        if overview.0 == 0 || overview.1 == 0 {
            return Err(bad("the overview has at least one token along each axis"));
        }
        let none = grid == (0, 0);
        if !none && (grid.0 == 0 || grid.1 == 0) {
            return Err(bad(
                "the slice grid has a column and a line, or none of either",
            ));
        }
        if none != (slice == (0, 0)) || (!none && (slice.0 == 0 || slice.1 == 0)) {
            return Err(bad("a slice has tokens exactly when the grid has slices"));
        }
        let c = Canvas {
            overview,
            grid,
            slice,
        };
        c.try_rows()
            .ok_or_else(|| bad("the rows of the group overflow"))?;
        Ok(c)
    }

    fn sliced(&self) -> bool {
        self.grid != (0, 0)
    }

    /// The tiles: the overview, then the slices line by line.
    #[must_use]
    pub fn tiles(&self) -> usize {
        1 + self.grid.0 * self.grid.1
    }

    /// Tile `k` of [`Canvas::tiles`]: tile 0 is the overview, tile `1 + s` slice `s` (line by
    /// line).
    #[must_use]
    pub fn tile(&self, k: usize) -> Tile {
        assert!(k < self.tiles(), "tile {k} of a group of {}", self.tiles());
        if k == 0 {
            let seps = if self.sliced() {
                OVERVIEW_WITH_SLICES
            } else {
                OVERVIEW_ALONE
            };
            return Tile {
                nx: self.overview.0,
                ny: self.overview.1,
                seps,
            };
        }
        let s = k - 1;
        let seps = if k + 1 == self.tiles() {
            SLICE_LAST
        } else if s % self.grid.0 == self.grid.0 - 1 {
            SLICE_LINE_END
        } else {
            SLICE_MID
        };
        Tile {
            nx: self.slice.0,
            ny: self.slice.1,
            seps,
        }
    }

    /// The slice grid, `(columns, lines)`; `(0, 0)` when the image has no slice.
    #[must_use]
    pub fn grid(&self) -> (usize, usize) {
        self.grid
    }

    fn try_rows(&self) -> Option<usize> {
        (0..self.tiles()).try_fold(0usize, |n, k| {
            let t = self.tile(k);
            n.checked_add(t.nx.checked_mul(t.ny)?)?
                .checked_add(t.seps.len())
        })
    }

    /// The group's rows: every tile's tokens and separators (`mtmd_image_tokens::n_tokens`).
    #[must_use]
    pub fn rows(&self) -> usize {
        self.try_rows().expect("checked by new")
    }

    /// The positions the group advances the sequence by: one a tile and one a separator
    /// (`mtmd_image_tokens_get_n_pos`).
    #[must_use]
    pub fn advance(&self) -> usize {
        (0..self.tiles()).map(|k| 1 + self.tile(k).seps.len()).sum()
    }

    /// The `[t, h, w, z]` position of group row `i` when the group's first row sits at position
    /// `p0`, which is one past the `<image>` token that is the origin
    /// (`mtmd_canvas_decoder_pos`).
    pub(super) fn slots(&self, p0: usize, i: usize) -> [usize; SLOTS] {
        let base = p0 - 1;
        // The reference's last entry gives the grid the others scale from, and the canvas size.
        let (nx, ny) = if self.sliced() {
            self.slice
        } else {
            self.overview
        };
        let (canvas_w, canvas_h) = if self.sliced() {
            (self.grid.0 * nx, self.grid.1 * ny)
        } else {
            (nx, ny)
        };
        let (mut i, mut t) = (i, p0);
        for k in 0..self.tiles() {
            let Tile {
                nx: gw,
                ny: gh,
                seps,
            } = self.tile(k);
            let seps = seps.len();
            if i < gw * gh {
                let (row, col) = (i / gw, i % gw);
                let (h, w) = if k == 0 {
                    (scale(row, gh, canvas_h), scale(col, gw, canvas_w))
                } else {
                    let s = k - 1;
                    ((s / self.grid.0) * ny + row, (s % self.grid.0) * nx + col)
                };
                return [t, base + h, base + w, base];
            }
            i -= gw * gh;
            if i < seps {
                let p = t + 1 + i;
                return [p; SLOTS];
            }
            i -= seps;
            t += 1 + seps;
        }
        panic!("group row {i} is past the group's {} rows", self.rows());
    }
}

/// `mtmd_canvas_scale`: tile coordinate `coord` of `grid` onto a canvas of `canvas` tokens,
/// `round(linspace(0, canvas - 1, grid))`, ties to even.
fn scale(coord: usize, grid: usize, canvas: usize) -> usize {
    if grid <= 1 || canvas <= 1 {
        return 0;
    }
    let v = coord as f64 * (canvas - 1) as f64 / (grid - 1) as f64;
    (v.round_ties_even() as usize).min(canvas - 1)
}

#[cfg(test)]
mod tests {
    use super::{Canvas, scale};
    use crate::mrope::MropeError;

    /// `round()` breaks ties to even, as the reference's `nearbyint` does: 3 tokens over 4 columns
    /// land on 0, 1.5 -> 2, 3.
    #[test]
    fn the_scale_rounds_ties_to_even() {
        assert_eq!(
            (0..4).map(|c| scale(c, 4, 4)).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert_eq!(
            (0..3).map(|c| scale(c, 3, 4)).collect::<Vec<_>>(),
            [0, 2, 3]
        );
        // 0.5 -> 0, 1.5 -> 2, 2.5 -> 2: coord * 5 / 2 over a canvas of 6.
        assert_eq!(
            (0..3).map(|c| scale(c, 3, 6)).collect::<Vec<_>>(),
            [0, 2, 5]
        );
        assert_eq!((0..2).map(|c| scale(c, 2, 2)).collect::<Vec<_>>(), [0, 1]);
        // A grid or canvas of one token is the origin.
        assert_eq!((scale(0, 1, 9), scale(5, 4, 1)), (0, 0));
    }

    #[test]
    fn a_canvas_with_no_tokens_or_a_half_grid_is_refused() {
        for (ov, grid, slice) in [
            ((0, 2), (0, 0), (0, 0)),
            ((2, 0), (0, 0), (0, 0)),
            ((2, 2), (2, 0), (1, 1)),
            ((2, 2), (2, 1), (0, 1)),
            ((2, 2), (0, 0), (1, 1)),
            ((2, 2), (1, 1), (0, 0)),
            ((usize::MAX, 2), (0, 0), (0, 0)),
        ] {
            assert!(
                matches!(Canvas::new(ov, grid, slice), Err(MropeError::Canvas { .. })),
                "{ov:?} {grid:?} {slice:?}"
            );
        }
    }
}
