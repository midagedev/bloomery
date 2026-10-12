//! The M-RoPE positions of a sequence that holds images, and the rows of a rope table they select.
//!
//! A text row at sequence row `r` has the position `r − δ`, where `δ` is what the images before
//! it saved: an image of `nx × ny` merged tokens occupies `nx · ny` rows but advances the position
//! by `max(nx, ny)` only. An image row `i` (raster order, `nx` to a line) of an image whose first
//! row sits at position `p₀` has the position triple `(t, h, w) = (p₀, p₀ + i / nx, p₀ + i % nx)`:
//! mtmd's `mtmd_image_tokens_get_decoder_pos` and `mtmd_image_tokens_get_n_pos` for a `MROPE`
//! projector.
//!
//! The rotation of a row takes each pair of the rotated head from one axis' position. ggml's
//! `ggml_mrope_cache_init` in its interleaved mode (the one `qwen3vl`, `qwen35moe` and `qwen4exp`
//! run) gives pair `s`, with `sector = s mod Σ sections`:
//!
//! * `sector mod 3 = 1` and `sector < 3·sections[1]`: the `h` position;
//! * `sector mod 3 = 2` and `sector < 3·sections[2]`: the `w` position;
//! * `sector mod 3 = 0` and `sector < 3·sections[0]`: the `t` position;
//! * any other sector: the extra position, which is 0 on every row (a text row is
//!   `[p, p, p, 0]` and an image row's fourth component is unused).
//!
//! The sections are data (the text file's `rope.dimension_sections`); nothing here fixes them.
//! [`MropeSeq::rows_into`] reads a rope table of plain positions (`row q` is the table at
//! position `q`, as the text model's own table is) and writes, for each sequence row, the table
//! row whose pair `s` comes from the row of the position that pair's axis gives. A table built that
//! way is exactly what a kernel that reads `table[row]` needs, so no kernel learns about M-RoPE.

/// What [`MropeSeq`] refuses.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum MropeError {
    /// An image with no tokens along an axis, or whose token count overflows.
    #[error("image grid {nx}x{ny}: {detail}")]
    Grid {
        nx: usize,
        ny: usize,
        detail: &'static str,
    },
    /// An image that starts before the previous one ends.
    #[error("image at row {at}: starts before row {prev_end}, where the previous image ends")]
    Overlap { at: usize, prev_end: usize },
    /// A cut that lands strictly inside an image: its rows are one unit.
    #[error("cut at row {row}: inside the image at rows {at}..{end}")]
    CutInsideSpan { row: usize, at: usize, end: usize },
    /// Rope sections that cannot select a pair, or a table or output of the wrong length.
    #[error("rows_into: {0}")]
    Shape(String),
    /// A row whose position lies past the rope table.
    #[error("row {row}: position {pos} is past the {rows} rows of the rope table")]
    PastTable { row: usize, pos: usize, rows: usize },
    /// An image whose rows reach past the sequence's rotation table.
    #[error("image at rows {at}..{end}: past the {ctx} rows of the sequence")]
    PastSequence { at: usize, end: usize, ctx: usize },
}

mod rows;
mod seq;

pub use rows::{RowRewrite, SeqRows};
pub use seq::MropeSeq;

/// What the tests of both subjects build their tables from.
#[cfg(test)]
mod fixture {
    /// The text file's `rope.dimension_sections` and rotated width.
    pub(super) const SECTIONS: [u32; 4] = [11, 11, 10, 0];
    pub(super) const ROT: usize = 64;

    /// A table whose value says where it came from: row `q`, value `k` is `1000·q + k`.
    pub(super) fn table(rows: usize, rot: usize) -> Vec<f32> {
        (0..rows * rot)
            .map(|i| (1000 * (i / rot) + i % rot) as f32)
            .collect()
    }
}
