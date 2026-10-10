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

/// The rows of a sequence's rotation table that a change of its images rewrites: rows
/// `from..from + values.len() / rot`, `rot` values a row, in row order.
#[derive(Clone, Debug, PartialEq)]
pub struct RowRewrite {
    pub from: usize,
    pub values: Vec<f32>,
}

/// The rotation table of one sequence, on the host side of its card copy: `ctx` rows of `rot`
/// values, where row `r` is the plain table's row of position `r` for a text sequence and the
/// [`MropeSeq::rows_into`] row once the sequence holds images. A change of the images decides
/// which rows it dirties and makes their values; the caller copies them to wherever the table
/// lives. Every change is all or nothing: a refused change leaves the sequence as it was.
///
/// Rows before a change's first image are what they were: a position depends on the images
/// before its row only. Rows after it all move, since every later text row's position shifts by
/// what the new image saves.
#[derive(Clone, Debug)]
pub struct SeqRows {
    base: std::sync::Arc<[f32]>,
    rot: usize,
    sections: [u32; 4],
    ctx: usize,
    seq: MropeSeq,
}

impl SeqRows {
    /// A sequence of `ctx` rows over the plain table `base` (`ctx` rows of `rot` values, the
    /// rope table of positions `0..ctx`); `sections` are the text file's
    /// `rope.dimension_sections`. A shape [`MropeSeq::rows_into`] would refuse is refused here.
    pub fn new(
        base: std::sync::Arc<[f32]>,
        sections: [u32; 4],
        ctx: usize,
    ) -> Result<SeqRows, MropeError> {
        if ctx == 0 || base.is_empty() || !base.len().is_multiple_of(ctx) {
            return Err(MropeError::Shape(format!(
                "a table of {} values for a sequence of {ctx} rows",
                base.len()
            )));
        }
        let rot = base.len() / ctx;
        MropeSeq::new().rows_into(&base, rot, sections, 0..0, &mut [])?;
        Ok(SeqRows {
            base,
            rot,
            sections,
            ctx,
            seq: MropeSeq::new(),
        })
    }

    /// The plain table: what a sequence with no image holds.
    #[must_use]
    pub fn base(&self) -> &[f32] {
        &self.base
    }

    /// Values a row.
    #[must_use]
    pub fn rot(&self) -> usize {
        self.rot
    }

    /// The rows of the sequence.
    #[must_use]
    pub fn ctx(&self) -> usize {
        self.ctx
    }

    /// The images the table now holds.
    #[must_use]
    pub fn seq(&self) -> &MropeSeq {
        &self.seq
    }

    /// Add images, each `(first row, nx, ny)` in ascending order after the ones held: the rows
    /// from the first of them to the end change. An image that reaches past the sequence's rows
    /// is refused; no images change nothing.
    pub fn push_spans(
        &mut self,
        spans: &[(usize, usize, usize)],
    ) -> Result<Option<RowRewrite>, MropeError> {
        let Some(&(from, _, _)) = spans.first() else {
            return Ok(None);
        };
        let mut next = self.seq.clone();
        for &(at, nx, ny) in spans {
            next.push(at, nx, ny)?;
            let end = at + nx * ny;
            if end > self.ctx {
                return Err(MropeError::PastSequence {
                    at,
                    end,
                    ctx: self.ctx,
                });
            }
        }
        let rewrite = self.rewrite(&next, from)?;
        self.seq = next;
        Ok(Some(rewrite))
    }

    /// Keep the first `row` rows ([`MropeSeq::cut`]): the images that start at or after `row`
    /// go, and the rows from `row` on take the positions the kept images leave. A cut that drops
    /// no image changes nothing: every row after the last kept image already holds its shifted
    /// row.
    pub fn cut(&mut self, row: usize) -> Result<Option<RowRewrite>, MropeError> {
        let mut next = self.seq.clone();
        next.cut(row)?;
        if next.spans().len() == self.seq.spans().len() {
            return Ok(None);
        }
        let rewrite = self.rewrite(&next, row)?;
        self.seq = next;
        Ok(Some(rewrite))
    }

    /// Take `seq` as the images of the sequence (a saved sequence coming back): the rows from
    /// the first image that differs from the held ones change; the same images change nothing.
    pub fn restore(&mut self, seq: MropeSeq) -> Result<Option<RowRewrite>, MropeError> {
        if let Some((at, nx, ny)) = seq.spans().find(|&(at, nx, ny)| at + nx * ny > self.ctx) {
            return Err(MropeError::PastSequence {
                at,
                end: at + nx * ny,
                ctx: self.ctx,
            });
        }
        let from = {
            let (mut old, mut new) = (self.seq.spans(), seq.spans());
            loop {
                match (old.next(), new.next()) {
                    (None, None) => return Ok(None),
                    (Some(a), Some(b)) if a == b => {}
                    (a, b) => {
                        break a
                            .map_or(usize::MAX, |s| s.0)
                            .min(b.map_or(usize::MAX, |s| s.0));
                    }
                }
            }
        };
        let rewrite = self.rewrite(&seq, from)?;
        self.seq = seq;
        Ok(Some(rewrite))
    }

    /// The rows `from..ctx` of the table `seq` gives.
    fn rewrite(&self, seq: &MropeSeq, from: usize) -> Result<RowRewrite, MropeError> {
        let mut values = vec![0.0; (self.ctx - from) * self.rot];
        seq.rows_into(
            &self.base,
            self.rot,
            self.sections,
            from..self.ctx,
            &mut values,
        )?;
        Ok(RowRewrite { from, values })
    }
}

#[cfg(test)]
mod tests {
    use super::{MropeError, MropeSeq, RowRewrite, SeqRows};
    use std::sync::Arc;

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
    const CTX: usize = 300;

    /// The rows of `seq` over the whole sequence, rebuilt from scratch.
    fn full(base: &[f32], seq: &MropeSeq) -> Vec<f32> {
        let mut out = vec![0.0; CTX * ROT];
        seq.rows_into(base, ROT, SECTIONS, 0..CTX, &mut out)
            .expect("the full rebuild");
        out
    }

    /// A card's copy of the table: the plain rows, then each rewrite spliced where it says.
    fn apply(copy: &mut [f32], rewrite: &Option<RowRewrite>) {
        if let Some(r) = rewrite {
            copy[r.from * ROT..r.from * ROT + r.values.len()].copy_from_slice(&r.values);
        }
    }

    fn images(rows: &SeqRows) -> Vec<(usize, usize, usize)> {
        rows.seq().spans().collect()
    }

    /// Every change gives rows that, spliced into the table held before it, make the table a
    /// rebuild from scratch makes, and starts at the first row the change moves: a push at its
    /// first image, a cut that drops images at the cut, a restore at the first image that
    /// differs. A change that moves no row gives none.
    #[test]
    fn a_change_rewrites_from_its_first_moved_row_to_the_end() {
        let base: Arc<[f32]> = table(CTX, ROT).into();
        let mut rows = SeqRows::new(base.clone(), SECTIONS, CTX).expect("rows");
        let mut copy = base.to_vec();
        let (a, b) = ((3, 14, 14), (210, 4, 2));
        let check = |rows: &SeqRows, copy: &[f32], what: &str| {
            assert!(
                copy == full(&base, rows.seq()),
                "{what}: the copy is the rebuild"
            );
        };
        check(&rows, &copy, "a text sequence");
        assert_eq!(rows.push_spans(&[]), Ok(None));

        let rw = rows.push_spans(&[a]).expect("image a");
        assert_eq!(
            rw.as_ref().map(|r| (r.from, r.values.len())),
            Some((3, 297 * ROT))
        );
        apply(&mut copy, &rw);
        check(&rows, &copy, "image a");
        let rw = rows.push_spans(&[b]).expect("image b");
        assert_eq!(rw.as_ref().map(|r| r.from), Some(210));
        apply(&mut copy, &rw);
        check(&rows, &copy, "image b");
        let both = rows.seq().clone();

        // A cut that drops image b rewrites from the cut; one that drops nothing writes nothing,
        // since every row after the last kept image already holds its shifted row.
        let rw = rows.cut(215).expect_err("a cut inside image b");
        assert_eq!(
            rw,
            MropeError::CutInsideSpan {
                row: 215,
                at: 210,
                end: 218
            }
        );
        assert_eq!(images(&rows), [a, b], "a refused cut changes nothing");
        let rw = rows.cut(210).expect("a cut at image b");
        assert_eq!(rw.as_ref().map(|r| r.from), Some(210));
        apply(&mut copy, &rw);
        check(&rows, &copy, "image b cut");
        assert_eq!(
            rows.cut(250),
            Ok(None),
            "a cut after the last image drops none"
        );
        assert_eq!(
            rows.cut(199),
            Ok(None),
            "a cut at the last image's end drops none"
        );
        check(&rows, &copy, "cuts that drop none");
        assert!(matches!(
            rows.cut(100),
            Err(MropeError::CutInsideSpan { .. })
        ));
        let rw = rows.cut(3).expect("a cut at image a");
        assert_eq!(rw.as_ref().map(|r| r.from), Some(3));
        apply(&mut copy, &rw);
        assert!(copy == *base, "no image: the plain table again");

        // A saved sequence comes back from its first image; the same one writes nothing; one that
        // differs from the second image on writes from the earlier of the two firsts.
        let rw = rows.restore(both.clone()).expect("restore");
        assert_eq!(rw.as_ref().map(|r| r.from), Some(3));
        apply(&mut copy, &rw);
        check(&rows, &copy, "restored");
        assert_eq!(rows.restore(both.clone()), Ok(None));
        let mut other = MropeSeq::new();
        other.push(a.0, a.1, a.2).expect("a");
        other.push(220, 2, 2).expect("other b");
        let rw = rows
            .restore(other)
            .expect("restore a different second image");
        assert_eq!(rw.as_ref().map(|r| r.from), Some(210));
        apply(&mut copy, &rw);
        check(&rows, &copy, "restored with another second image");

        // Two images in one push: from the first.
        let mut rows = SeqRows::new(base.clone(), SECTIONS, CTX).expect("rows");
        let mut copy = base.to_vec();
        let rw = rows.push_spans(&[a, b]).expect("both");
        assert_eq!(rw.as_ref().map(|r| r.from), Some(3));
        apply(&mut copy, &rw);
        check(&rows, &copy, "both in one push");
        assert_eq!(rows.seq(), &both);
    }

    /// A refused change leaves the images as they were, a push that fails on its second image
    /// included; an image past the sequence's rows, an overlap and a table of another shape are
    /// refused by what they are.
    #[test]
    fn a_refused_change_leaves_the_sequence_as_it_was() {
        let base: Arc<[f32]> = table(CTX, ROT).into();
        let mut rows = SeqRows::new(base.clone(), SECTIONS, CTX).expect("rows");
        rows.push_spans(&[(3, 14, 14)]).expect("image a");
        let held = rows.seq().clone();
        assert_eq!(
            rows.push_spans(&[(290, 4, 4)]),
            Err(MropeError::PastSequence {
                at: 290,
                end: 306,
                ctx: CTX
            })
        );
        assert_eq!(
            rows.push_spans(&[(230, 2, 2), (231, 1, 1)]),
            Err(MropeError::Overlap {
                at: 231,
                prev_end: 234
            })
        );
        assert!(matches!(
            rows.push_spans(&[(100, 2, 2)]),
            Err(MropeError::Overlap { .. })
        ));
        let mut past = MropeSeq::new();
        past.push(298, 2, 2).expect("push");
        assert!(matches!(
            rows.restore(past),
            Err(MropeError::PastSequence { .. })
        ));
        assert_eq!(rows.seq(), &held);
        assert!(matches!(
            SeqRows::new(base.clone(), SECTIONS, CTX + 1),
            Err(MropeError::Shape(_))
        ));
        assert!(matches!(
            SeqRows::new(base.clone(), [0; 4], CTX),
            Err(MropeError::Shape(_))
        ));
        assert!(matches!(
            SeqRows::new(base, SECTIONS, 0),
            Err(MropeError::Shape(_))
        ));
    }
}
