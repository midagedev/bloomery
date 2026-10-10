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
}

/// One image of the sequence: its rows `at..at + nx·ny`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    at: usize,
    nx: usize,
    ny: usize,
    /// What this image and every image before it saved: `Σ (nx·ny − max(nx, ny))`.
    saved: usize,
}

impl Span {
    fn end(&self) -> usize {
        self.at + self.nx * self.ny
    }

    /// What the images before this one saved.
    fn saved_before(&self) -> usize {
        self.saved - saves(self.nx, self.ny)
    }
}

/// What an `nx × ny` image saves: its rows less the positions it advances by, `max(nx, ny)`.
fn saves(nx: usize, ny: usize) -> usize {
    nx * ny - nx.max(ny)
}

/// The images of one sequence and the positions they give its rows.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MropeSeq {
    spans: Vec<Span>,
}

impl MropeSeq {
    /// A sequence with no image: every row's position is its row.
    #[must_use]
    pub fn new() -> MropeSeq {
        MropeSeq::default()
    }

    /// The images, each as `(first row, nx, ny)`, in row order.
    pub fn spans(&self) -> impl ExactSizeIterator<Item = (usize, usize, usize)> + '_ {
        self.spans.iter().map(|s| (s.at, s.nx, s.ny))
    }

    /// Whether the sequence holds no image.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Append an image of `nx × ny` merged tokens whose first row is `row_at`. Images come in
    /// ascending row order and do not overlap.
    pub fn push(&mut self, row_at: usize, nx: usize, ny: usize) -> Result<(), MropeError> {
        let grid = |detail| MropeError::Grid { nx, ny, detail };
        if nx == 0 || ny == 0 {
            return Err(grid("an image has at least one token along each axis"));
        }
        nx.checked_mul(ny)
            .filter(|&n| row_at.checked_add(n).is_some())
            .ok_or_else(|| grid("the rows of the image overflow"))?;
        let prev = self.spans.last();
        if let Some(prev) = prev
            && row_at < prev.end()
        {
            return Err(MropeError::Overlap {
                at: row_at,
                prev_end: prev.end(),
            });
        }
        let saved = prev.map_or(0, |p| p.saved) + saves(nx, ny);
        self.spans.push(Span {
            at: row_at,
            nx,
            ny,
            saved,
        });
        Ok(())
    }

    /// Keep the first `row` rows of the sequence: the images that start at or after `row` go. A
    /// cut inside an image is refused.
    pub fn cut(&mut self, row: usize) -> Result<(), MropeError> {
        if let Some(s) = self.spans.iter().find(|s| s.at < row && row < s.end()) {
            return Err(MropeError::CutInsideSpan {
                row,
                at: s.at,
                end: s.end(),
            });
        }
        self.spans.retain(|s| s.at < row);
        Ok(())
    }

    /// What the images that end at or before `row` saved: a text row's position is `row − δ`.
    #[must_use]
    pub fn delta(&self, row: usize) -> usize {
        let ended = self.spans.partition_point(|s| s.end() <= row);
        ended.checked_sub(1).map_or(0, |k| self.spans[k].saved)
    }

    /// The `(t, h, w)` position of `row`.
    #[must_use]
    pub fn pos(&self, row: usize) -> (usize, usize, usize) {
        let k = self.spans.partition_point(|s| s.end() <= row);
        if let Some(s) = self.spans.get(k)
            && s.at <= row
        {
            let i = row - s.at;
            let p0 = s.at - s.saved_before();
            return (p0, p0 + i / s.nx, p0 + i % s.nx);
        }
        let p = row - k.checked_sub(1).map_or(0, |k| self.spans[k].saved);
        (p, p, p)
    }

    /// For the sequence rows `from..to`, write each row's rotation row into `out` (`rot` values a
    /// row, `(to − from) · rot` in all). `base` holds the plain table, `rot` values per position
    /// (a cos and a sin per pair, `rot / 2` pairs); pair `s` of row `r` is copied from the `base`
    /// row of the position that `sections` give pair `s` on row `r`.
    ///
    /// Every position is at most its row, so a table of `to` rows suffices; a position past the
    /// table is refused by row, and `out` then holds the rows written before it.
    pub fn rows_into(
        &self,
        base: &[f32],
        rot: usize,
        sections: [u32; 4],
        rows: std::ops::Range<usize>,
        out: &mut [f32],
    ) -> Result<(), MropeError> {
        let shape = |detail: String| MropeError::Shape(detail);
        let pairs = rot / 2;
        let sect_dims = sections.iter().map(|&s| s as usize).sum::<usize>();
        if rot == 0 || !rot.is_multiple_of(2) {
            return Err(shape(format!("rot {rot} must be even and positive")));
        }
        if sect_dims == 0 || sect_dims > rot {
            return Err(shape(format!(
                "sections {sections:?} sum to {sect_dims}; need 1..={rot}"
            )));
        }
        if !base.len().is_multiple_of(rot) {
            return Err(shape(format!(
                "the table has {} values, not whole rows of {rot}",
                base.len()
            )));
        }
        let table_rows = base.len() / rot;
        if rows.end < rows.start || out.len() != (rows.end - rows.start) * rot {
            return Err(shape(format!(
                "rows {}..{} need {} output values, got {}",
                rows.start,
                rows.end,
                rows.end.saturating_sub(rows.start) * rot,
                out.len()
            )));
        }
        let [s0, s1, s2, _] = sections.map(|s| s as usize);
        for (row, dst) in rows.zip(out.chunks_exact_mut(rot)) {
            let (t, h, w) = self.pos(row);
            let at = |pos: usize| {
                if pos < table_rows {
                    Ok(&base[pos * rot..(pos + 1) * rot])
                } else {
                    Err(MropeError::PastTable {
                        row,
                        pos,
                        rows: table_rows,
                    })
                }
            };
            let (rt, rh, rw) = (at(t)?, at(h)?, at(w)?);
            // The extra axis' position is 0 on every row.
            let re = at(0)?;
            for s in 0..pairs {
                let sector = s % sect_dims;
                let src = if sector % 3 == 1 && sector < 3 * s1 {
                    rh
                } else if sector % 3 == 2 && sector < 3 * s2 {
                    rw
                } else if sector % 3 == 0 && sector < 3 * s0 {
                    rt
                } else {
                    re
                };
                dst[2 * s..2 * s + 2].copy_from_slice(&src[2 * s..2 * s + 2]);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{MropeError, MropeSeq};

    /// The text file's `rope.dimension_sections` and rotated width.
    const SECTIONS: [u32; 4] = [11, 11, 10, 0];
    const ROT: usize = 64;

    /// A table whose value says where it came from: row `q`, value `k` is `1000·q + k`.
    fn table(rows: usize, rot: usize) -> Vec<f32> {
        (0..rows * rot)
            .map(|i| (1000 * (i / rot) + i % rot) as f32)
            .collect()
    }

    /// One 14×14 image (196 rows, 14 positions) at row 3, then text: the text saves 196 − 14 = 182
    /// positions, so row 199 (the first text row after the image) is position 17.
    #[test]
    fn one_image_then_text_saves_the_difference() {
        let mut m = MropeSeq::new();
        m.push(3, 14, 14).expect("push");
        assert_eq!((m.delta(2), m.delta(3), m.delta(198)), (0, 0, 0));
        assert_eq!((m.delta(199), m.delta(500)), (182, 182));
        // Text before the image is its row.
        assert_eq!(m.pos(2), (2, 2, 2));
        // The first image row: p₀ = 3. Row 4 is the second token of the first line, row 17
        // (i = 14) the first token of the second.
        assert_eq!(m.pos(3), (3, 3, 3));
        assert_eq!(m.pos(4), (3, 3, 4));
        assert_eq!(m.pos(17), (3, 4, 3));
        // The last image row: i = 195, line 13, column 13.
        assert_eq!(m.pos(198), (3, 16, 16));
        // Text after it continues at p₀ + max(nx, ny) = 17.
        assert_eq!(m.pos(199), (17, 17, 17));
        assert_eq!(m.pos(200), (18, 18, 18));
    }

    /// A non-square image advances by its larger side: 2×3 (nx 2, ny 3) is 6 rows and 3 positions.
    #[test]
    fn a_non_square_image_advances_by_its_larger_side() {
        let mut m = MropeSeq::new();
        m.push(0, 2, 3).expect("push");
        assert_eq!(m.pos(5), (0, 2, 1));
        assert_eq!(m.pos(6), (3, 3, 3));
        let mut m = MropeSeq::new();
        m.push(0, 3, 2).expect("push");
        assert_eq!(m.pos(5), (0, 1, 2));
        assert_eq!(m.pos(6), (3, 3, 3));
    }

    /// Two images accumulate: A (2×3) at row 1 saves 6 − 3 = 3, B (4×2) at row 10 saves 8 − 4 = 4.
    /// B starts at position 10 − 3 = 7, and the text after B is at 18 − 7 = 11 (B used 7..11).
    #[test]
    fn two_images_accumulate() {
        let mut m = MropeSeq::new();
        m.push(1, 2, 3).expect("A");
        m.push(10, 4, 2).expect("B");
        assert_eq!(m.spans().collect::<Vec<_>>(), [(1, 2, 3), (10, 4, 2)]);
        // Text between the images: rows 7, 8, 9 are positions 4, 5, 6.
        assert_eq!((m.pos(7), m.pos(9)), ((4, 4, 4), (6, 6, 6)));
        // B's first row is position 7; its sixth (i = 5: line 1, column 1) is (7, 8, 8).
        assert_eq!((m.pos(10), m.pos(15)), ((7, 7, 7), (7, 8, 8)));
        assert_eq!((m.delta(10), m.delta(17), m.delta(18)), (3, 3, 7));
        assert_eq!(m.pos(18), (11, 11, 11));
    }

    /// Images must come in ascending row order and may touch but not overlap; an empty or an
    /// overflowing grid is refused.
    #[test]
    fn push_refuses_overlap_and_empty_grids() {
        let mut m = MropeSeq::new();
        m.push(5, 2, 2).expect("push");
        assert_eq!(
            m.push(8, 2, 2),
            Err(MropeError::Overlap { at: 8, prev_end: 9 })
        );
        assert_eq!(
            m.push(2, 1, 1),
            Err(MropeError::Overlap { at: 2, prev_end: 9 })
        );
        m.push(9, 1, 1).expect("touching is fine");
        assert!(matches!(m.push(20, 0, 3), Err(MropeError::Grid { .. })));
        assert!(matches!(
            m.push(20, usize::MAX, 2),
            Err(MropeError::Grid { .. })
        ));
        assert_eq!(m.spans().len(), 2, "a refused push leaves the sequence");
    }

    /// A cut keeps the first `row` rows: it drops the images from `row` on, keeps one that ends at
    /// `row`, and refuses a row strictly inside an image.
    #[test]
    fn a_cut_drops_whole_images_and_refuses_a_split() {
        let mut m = MropeSeq::new();
        m.push(1, 2, 3).expect("A");
        m.push(10, 4, 2).expect("B");
        assert_eq!(
            m.clone().cut(12),
            Err(MropeError::CutInsideSpan {
                row: 12,
                at: 10,
                end: 18
            })
        );
        assert!(matches!(
            m.clone().cut(4),
            Err(MropeError::CutInsideSpan { .. })
        ));
        let mut at_end = m.clone();
        at_end.cut(18).expect("cut at an image's end");
        assert_eq!(at_end, m);
        let mut at_start = m.clone();
        at_start.cut(10).expect("cut at an image's start");
        assert_eq!(at_start.spans().collect::<Vec<_>>(), [(1, 2, 3)]);
        assert_eq!(at_start.delta(40), 3);
        let mut all = m.clone();
        all.cut(1).expect("cut before the first image");
        assert!(all.is_empty());
        assert_eq!(all.pos(40), (40, 40, 40));
        // Images pushed after a cut continue from the kept ones.
        at_start.push(10, 1, 1).expect("push after a cut");
        assert_eq!(at_start.pos(11), (8, 8, 8));
    }

    /// `rows_into` against hand-read sectors. The image is 14×14 at row 3; row 74 is its line 5,
    /// column 1: position (3, 8, 4). With sections [11, 11, 10, 0] and 32 pairs, pair `s` is the
    /// `h` row for s mod 3 = 1, the `w` row for s mod 3 = 2 below 30, and the `t` row otherwise.
    #[test]
    fn rows_pick_each_pair_from_its_axis() {
        let base = table(300, ROT);
        let mut m = MropeSeq::new();
        m.push(3, 14, 14).expect("push");
        let mut out = vec![0.0; ROT];
        m.rows_into(&base, ROT, SECTIONS, 74..75, &mut out)
            .expect("rows");
        assert_eq!(m.pos(74), (3, 8, 4));
        for (s, pair) in out.as_chunks::<2>().0.iter().enumerate() {
            let q = match s % 3 {
                1 => 8,
                2 if s < 30 => 4,
                _ => 3,
            };
            assert_eq!(
                *pair,
                [(1000 * q + 2 * s) as f32, (1000 * q + 2 * s + 1) as f32],
                "pair {s}"
            );
        }
        // Pairs 29 (w), 30 (t) and 31 (h) are the section boundaries.
        assert_eq!(out[2 * 29], 4058.0);
        assert_eq!(out[2 * 30], 3060.0);
        assert_eq!(out[2 * 31], 8062.0);
    }

    /// A text row of a sequence with an image is the table row at `row − δ`, whole; a text row
    /// before any image is its own row. The run is cut at any boundary the same.
    #[test]
    fn a_text_row_is_the_shifted_table_row() {
        let base = table(300, ROT);
        let mut m = MropeSeq::new();
        m.push(3, 14, 14).expect("push");
        let (from, to) = (2, 202);
        let mut whole = vec![0.0; (to - from) * ROT];
        m.rows_into(&base, ROT, SECTIONS, from..to, &mut whole)
            .expect("rows");
        let row = |r: usize| &whole[(r - from) * ROT..(r - from + 1) * ROT];
        assert_eq!(row(2), &base[2 * ROT..3 * ROT]);
        assert_eq!(row(199), &base[17 * ROT..18 * ROT]);
        assert_eq!(row(201), &base[19 * ROT..20 * ROT]);
        // Rows 100..120 written alone equal the same rows of the whole run.
        let mut part = vec![0.0; 20 * ROT];
        m.rows_into(&base, ROT, SECTIONS, 100..120, &mut part)
            .expect("rows");
        assert_eq!(part, &whole[(100 - from) * ROT..(120 - from) * ROT]);
    }

    /// A section that no axis covers reads the extra position, which is 0 on every row:
    /// sections [2, 2, 2, 2] over 8 pairs leave sectors 6 and 7 to it.
    #[test]
    fn an_uncovered_sector_reads_position_zero() {
        let rot = 16;
        let base = table(40, rot);
        let mut m = MropeSeq::new();
        m.push(5, 3, 3).expect("push");
        let mut out = vec![0.0; rot];
        m.rows_into(&base, rot, [2, 2, 2, 2], 20..21, &mut out)
            .expect("rows");
        // The image saves 9 − 3 = 6 positions: row 20 is position 14.
        assert_eq!(m.pos(20), (14, 14, 14));
        for (s, pair) in out.as_chunks::<2>().0.iter().enumerate() {
            let q = if s >= 6 { 0 } else { 14 };
            assert_eq!(pair[0], (1000 * q + 2 * s) as f32, "pair {s}");
        }
    }

    /// A position past the table is refused by its row, though no position exceeds its row; a
    /// table or output of the wrong length, and sections that cannot select a pair, are refused
    /// by what they are.
    #[test]
    fn rows_into_refuses_a_short_table_and_bad_shapes() {
        let mut m = MropeSeq::new();
        m.push(3, 14, 14).expect("push");
        // Row 199 is position 17: a table of 17 rows ends one short.
        let base = table(17, ROT);
        let mut out = vec![0.0; ROT];
        assert_eq!(
            m.rows_into(&base, ROT, SECTIONS, 199..200, &mut out),
            Err(MropeError::PastTable {
                row: 199,
                pos: 17,
                rows: 17
            })
        );
        let base = table(20, ROT);
        assert!(matches!(
            m.rows_into(&base[1..], ROT, SECTIONS, 199..200, &mut out),
            Err(MropeError::Shape(_))
        ));
        assert!(matches!(
            m.rows_into(&base, ROT, SECTIONS, 199..201, &mut out),
            Err(MropeError::Shape(_))
        ));
        assert!(matches!(
            m.rows_into(&base, ROT, [0; 4], 199..200, &mut out),
            Err(MropeError::Shape(_))
        ));
        assert!(matches!(
            m.rows_into(&base, 63, SECTIONS, 199..200, &mut out),
            Err(MropeError::Shape(_))
        ));
        m.rows_into(&base, ROT, SECTIONS, 199..200, &mut out)
            .expect("a table of 20 rows reaches position 17");
    }
}
