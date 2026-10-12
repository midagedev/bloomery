//! [`MropeSeq`]: the images of one sequence and the positions they give its rows.

use super::MropeError;
use super::canvas::Canvas;
use super::order::{SLOTS, SectionOrder};

/// How an image's rows are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    /// `nx × ny` merged tokens in raster order, advancing the position by `max(nx, ny)`.
    Raster { nx: usize, ny: usize },
    /// One group of tiles and separators, advancing by one position a tile and a separator.
    Canvas(Canvas),
}

/// One image of the sequence: its rows `at..at + rows`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    at: usize,
    layout: Layout,
    /// What this image and every image before it saved: `Σ (rows − advance)`.
    saved: usize,
}

impl Span {
    fn rows(&self) -> usize {
        match self.layout {
            Layout::Raster { nx, ny } => nx * ny,
            Layout::Canvas(c) => c.rows(),
        }
    }

    fn end(&self) -> usize {
        self.at + self.rows()
    }

    /// What this image alone saves.
    fn saves(&self) -> usize {
        match self.layout {
            Layout::Raster { nx, ny } => saves(nx, ny),
            Layout::Canvas(c) => c.rows() - c.advance(),
        }
    }

    /// What the images before this one saved.
    fn saved_before(&self) -> usize {
        self.saved - self.saves()
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

    /// The images, each as `(first row, nx, ny)`, in row order: its rows are `nx · ny`. A canvas
    /// group is one line of all its rows (`ny = 1`).
    pub fn spans(&self) -> impl ExactSizeIterator<Item = (usize, usize, usize)> + '_ {
        self.spans.iter().map(|s| match s.layout {
            Layout::Raster { nx, ny } => (s.at, nx, ny),
            Layout::Canvas(c) => (s.at, c.rows(), 1),
        })
    }

    /// The first row where this sequence's images differ from `other`'s, `None` when they are the
    /// same images: a canvas group differs from another of the same row count.
    #[must_use]
    pub fn first_difference(&self, other: &MropeSeq) -> Option<usize> {
        let (mut a, mut b) = (self.spans.iter(), other.spans.iter());
        loop {
            match (a.next(), b.next()) {
                (None, None) => return None,
                (Some(x), Some(y)) if x.at == y.at && x.layout == y.layout => {}
                (x, y) => {
                    return Some(
                        x.map_or(usize::MAX, |s| s.at)
                            .min(y.map_or(usize::MAX, |s| s.at)),
                    );
                }
            }
        }
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
        self.append(row_at, Layout::Raster { nx, ny })
    }

    /// Append an image group whose first row is `row_at`, the row after the `<image>` token that is
    /// the canvas origin. Same order rules as [`MropeSeq::push`]; a group with nothing before it
    /// for an origin is refused.
    pub fn push_canvas(&mut self, row_at: usize, canvas: Canvas) -> Result<(), MropeError> {
        if row_at.checked_add(canvas.rows()).is_none() {
            return Err(MropeError::Canvas {
                detail: "the rows of the group overflow",
            });
        }
        self.append(row_at, Layout::Canvas(canvas))
    }

    fn append(&mut self, row_at: usize, layout: Layout) -> Result<(), MropeError> {
        let prev = self.spans.last();
        let mut span = Span {
            at: row_at,
            layout,
            saved: 0,
        };
        if let Some(prev) = prev
            && row_at < prev.end()
        {
            return Err(MropeError::Overlap {
                at: row_at,
                prev_end: prev.end(),
            });
        }
        let before = prev.map_or(0, |p| p.saved);
        if matches!(layout, Layout::Canvas(_)) && row_at - before == 0 {
            return Err(MropeError::Canvas {
                detail: "the group has no <image> token before it to be its origin",
            });
        }
        span.saved = before + span.saves();
        self.spans.push(span);
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

    /// The `(t, h, w)` position of `row`: the first three of [`MropeSeq::slots`].
    #[must_use]
    pub fn pos(&self, row: usize) -> (usize, usize, usize) {
        let [t, h, w, _] = self.slots(row);
        (t, h, w)
    }

    /// Whether `row` is a row of an image (an embedding row), as the section order maps only those.
    fn is_image_row(&self, row: usize) -> bool {
        let k = self.spans.partition_point(|s| s.end() <= row);
        self.spans.get(k).is_some_and(|s| s.at <= row)
    }

    /// The four position slots `[t, h, w, z]` of `row`. A text row is `[p, p, p, 0]`; a raster
    /// image row `i` of an image at position `p₀` is `[p₀, p₀ + i / nx, p₀ + i % nx, 0]`; a canvas
    /// group row is [`Canvas`]'s.
    #[must_use]
    pub fn slots(&self, row: usize) -> [usize; SLOTS] {
        let k = self.spans.partition_point(|s| s.end() <= row);
        if let Some(s) = self.spans.get(k)
            && s.at <= row
        {
            let i = row - s.at;
            let p0 = s.at - s.saved_before();
            return match s.layout {
                Layout::Raster { nx, .. } => [p0, p0 + i / nx, p0 + i % nx, 0],
                Layout::Canvas(c) => c.slots(p0, i),
            };
        }
        let p = row - k.checked_sub(1).map_or(0, |k| self.spans[k].saved);
        [p, p, p, 0]
    }

    /// For the sequence rows `from..to`, write each row's rotation row into `out` (`rot` values a
    /// row, `(to − from) · rot` in all). `base` holds the plain table, `rot` values per position
    /// (a cos and a sin per pair, `rot / 2` pairs); pair `s` of row `r` is copied from the `base`
    /// row of the position that `sections` give pair `s` on row `r`: on an image row, section `i`
    /// reads the slot `order` names for it; a text row reads them in order.
    ///
    /// A text row's position is at most its row, and so is an image row's, except a canvas
    /// group's overview, which spreads over the whole canvas; every position is within the rows of
    /// the image it belongs to. A position past the table is refused by row, and `out` then holds
    /// the rows written before it.
    pub fn rows_into(
        &self,
        base: &[f32],
        rot: usize,
        sections: [u32; 4],
        order: SectionOrder,
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
            let slots = self.slots(row);
            // The order maps the slots of embedding rows only; a text row's sections read the
            // slots as they stand (`src/llama-graph.cpp:176-190`: `is_embd ? slot_of_section[s] : s`).
            let slot_of_section = if self.is_image_row(row) {
                order.slot_of_section()
            } else {
                SectionOrder::Tyxz.slot_of_section()
            };
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
            // The rows the sections read; section 3 is the extra one.
            let [r0, r1, r2, re] = [0, 1, 2, 3].map(|s| at(slots[slot_of_section[s]]));
            let (r0, r1, r2, re) = (r0?, r1?, r2?, re?);
            for s in 0..pairs {
                let sector = s % sect_dims;
                let src = if sector % 3 == 1 && sector < 3 * s1 {
                    r1
                } else if sector % 3 == 2 && sector < 3 * s2 {
                    r2
                } else if sector % 3 == 0 && sector < 3 * s0 {
                    r0
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
    use crate::mrope::fixture::{ROT, SECTIONS, table};
    use crate::mrope::{Canvas, SectionOrder};

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
        m.rows_into(&base, ROT, SECTIONS, SectionOrder::Tyxz, 74..75, &mut out)
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
        m.rows_into(
            &base,
            ROT,
            SECTIONS,
            SectionOrder::Tyxz,
            from..to,
            &mut whole,
        )
        .expect("rows");
        let row = |r: usize| &whole[(r - from) * ROT..(r - from + 1) * ROT];
        assert_eq!(row(2), &base[2 * ROT..3 * ROT]);
        assert_eq!(row(199), &base[17 * ROT..18 * ROT]);
        assert_eq!(row(201), &base[19 * ROT..20 * ROT]);
        // Rows 100..120 written alone equal the same rows of the whole run.
        let mut part = vec![0.0; 20 * ROT];
        m.rows_into(
            &base,
            ROT,
            SECTIONS,
            SectionOrder::Tyxz,
            100..120,
            &mut part,
        )
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
        m.rows_into(
            &base,
            rot,
            [2, 2, 2, 2],
            SectionOrder::Tyxz,
            20..21,
            &mut out,
        )
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
            m.rows_into(&base, ROT, SECTIONS, SectionOrder::Tyxz, 199..200, &mut out),
            Err(MropeError::PastTable {
                row: 199,
                pos: 17,
                rows: 17
            })
        );
        let base = table(20, ROT);
        assert!(matches!(
            m.rows_into(
                &base[1..],
                ROT,
                SECTIONS,
                SectionOrder::Tyxz,
                199..200,
                &mut out
            ),
            Err(MropeError::Shape(_))
        ));
        assert!(matches!(
            m.rows_into(&base, ROT, SECTIONS, SectionOrder::Tyxz, 199..201, &mut out),
            Err(MropeError::Shape(_))
        ));
        assert!(matches!(
            m.rows_into(&base, ROT, [0; 4], SectionOrder::Tyxz, 199..200, &mut out),
            Err(MropeError::Shape(_))
        ));
        assert!(matches!(
            m.rows_into(&base, 63, SECTIONS, SectionOrder::Tyxz, 199..200, &mut out),
            Err(MropeError::Shape(_))
        ));
        m.rows_into(&base, ROT, SECTIONS, SectionOrder::Tyxz, 199..200, &mut out)
            .expect("a table of 20 rows reaches position 17");
    }

    /// MiniCPM-V 4.7's group of 1000×700 (overview 10×7 tokens, 2×2 slices of 10×7) after
    /// `<image_id>0</image_id><image>` (rows 0..3): the group's first row is 4, so the canvas origin
    /// (the `<image>` token) is position 3 and the first tile's `t` is 4. Every value below is the
    /// reference rule worked by hand (`mtmd_canvas_decoder_pos`, canvas 20×14 tokens): the overview
    /// is stretched over the canvas (`round(i · 13 / 6)`, `round(j · 19 / 9)`, ties to even), a slice
    /// `s` fills the cell at line `s / 2`, column `s % 2`, `t` runs one a tile and one a separator,
    /// and `z` stays the origin.
    #[test]
    fn a_canvas_group_is_placed_by_the_reference_rule() {
        let canvas = Canvas::new((10, 7), (2, 2), (10, 7)).expect("canvas");
        assert_eq!((canvas.rows(), canvas.advance()), (360, 15));
        let mut m = MropeSeq::new();
        m.push_canvas(4, canvas).expect("push");
        let at = |i: usize| m.slots(4 + i);
        // Text before the group, and the <image> token, are their rows.
        assert_eq!((m.slots(2), m.slots(3)), ([2, 2, 2, 0], [3, 3, 3, 0]));
        // The overview: first token, (1, 1), (3, 5) where 3 · 13 / 6 = 6.5 ties to 6 and
        // 5 · 19 / 9 = 10.56 rounds to 11, and the last token (6, 9) at the canvas corner.
        assert_eq!(at(0), [4, 3, 3, 3]);
        assert_eq!(at(11), [4, 5, 5, 3]);
        assert_eq!(at(35), [4, 9, 14, 3]);
        assert_eq!(at(69), [4, 16, 22, 3]);
        // `</image><slice>` follow at positions 5 and 6, equal in every slot.
        assert_eq!((at(70), at(71)), ([5; 4], [6; 4]));
        // Slice 0 at t = 7, its two separators at 8 and 9.
        assert_eq!((at(72), at(142), at(143)), ([7, 3, 3, 3], [8; 4], [9; 4]));
        // Slice 1 (line 0, column 1) at t = 10: columns from 10; separators 11, 12 (the line
        // end's `\n`), 13.
        assert_eq!((at(144), at(213)), ([10, 3, 13, 3], [10, 9, 22, 3]));
        assert_eq!((at(214), at(215), at(216)), ([11; 4], [12; 4], [13; 4]));
        // Slice 2 (line 1, column 0) at t = 14: lines from 7; separators 15, 16.
        assert_eq!(
            (at(217), at(287), at(288)),
            ([14, 10, 3, 3], [15; 4], [16; 4])
        );
        // The last slice at t = 17, its `</slice>` at 18, and the text after at 19 = 4 + 15.
        assert_eq!((at(289), at(358)), ([17, 10, 13, 3], [17, 16, 22, 3]));
        assert_eq!(at(359), [18; 4]);
        assert_eq!(m.slots(364), [19, 19, 19, 0]);
        assert_eq!((m.delta(364), m.pos(364)), (345, (19, 19, 19)));
    }

    /// An overview alone sits on its own grid (`round(i · (n − 1) / (n − 1)) = i`), and a group with
    /// nothing before it for an origin is refused.
    #[test]
    fn an_unsliced_group_and_a_group_with_no_origin() {
        let canvas = Canvas::new((3, 2), (0, 0), (0, 0)).expect("canvas");
        let mut m = MropeSeq::new();
        m.push_canvas(1, canvas).expect("push");
        // Origin 0; rows 1..7 are the tokens, row 7 the `</image>` at position 2.
        assert_eq!(
            (m.slots(1), m.slots(2), m.slots(6)),
            ([1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 2, 0])
        );
        assert_eq!(m.slots(7), [2; 4]);
        assert_eq!(m.slots(8), [3, 3, 3, 0]);
        let mut none = MropeSeq::new();
        assert!(matches!(
            none.push_canvas(0, canvas),
            Err(MropeError::Canvas { .. })
        ));
        let mut early = MropeSeq::new();
        early.push(0, 1, 1).expect("push");
        assert!(matches!(
            early.push_canvas(0, canvas),
            Err(MropeError::Overlap { .. })
        ));
    }

    /// `zyxt` against `tyxz` on the overview row (3, 5) of that group (sequence row 39, slots
    /// `[t, h, w, z] = [4, 9, 14, 3]`) with sections `[11, 11, 10, 0]`: pairs 1 mod 3 read the line
    /// (9) and pairs 2 mod 3 the column (14) under both; the pairs 0 mod 3 (section 0) read `t` = 4
    /// under `tyxz` and `z` = 3 under `zyxt`. With sections `[2, 2, 2, 2]` the uncovered pairs 6 and
    /// 7 (section 3) read `z` = 3 under `tyxz` and `t` = 4 under `zyxt`. A text row is the same
    /// under both orders: only embedding rows are mapped.
    #[test]
    fn zyxt_reads_z_where_tyxz_reads_t() {
        let canvas = Canvas::new((10, 7), (2, 2), (10, 7)).expect("canvas");
        let mut m = MropeSeq::new();
        m.push_canvas(4, canvas).expect("push");
        let pos_of = |order: SectionOrder, sections: [u32; 4], rot: usize, row: usize| {
            let base = table(300, rot);
            let mut out = vec![0.0; rot];
            m.rows_into(&base, rot, sections, order, row..row + 1, &mut out)
                .expect("rows");
            out.as_chunks::<2>()
                .0
                .iter()
                .enumerate()
                .map(|(s, p)| (p[0] as usize - 2 * s) / 1000)
                .collect::<Vec<_>>()
        };
        let want = |t_or_z: usize| -> Vec<usize> {
            (0..32)
                .map(|s| match s % 3 {
                    1 => 9,
                    2 if s < 30 => 14,
                    _ => t_or_z,
                })
                .collect()
        };
        assert_eq!(pos_of(SectionOrder::Tyxz, SECTIONS, ROT, 39), want(4));
        assert_eq!(pos_of(SectionOrder::Zyxt, SECTIONS, ROT, 39), want(3));
        let extra = |order| pos_of(order, [2, 2, 2, 2], 16, 39);
        assert_eq!(extra(SectionOrder::Tyxz), [4, 9, 14, 4, 9, 14, 3, 3]);
        assert_eq!(extra(SectionOrder::Zyxt), [3, 9, 14, 3, 9, 14, 4, 4]);
        // A text row (row 2) and a separator row (row 74 = 4 + 70, [5; 4]) are the same.
        for row in [2, 74] {
            assert_eq!(
                pos_of(SectionOrder::Tyxz, SECTIONS, ROT, row),
                pos_of(SectionOrder::Zyxt, SECTIONS, ROT, row),
                "row {row}"
            );
        }
        assert_eq!(pos_of(SectionOrder::Zyxt, SECTIONS, ROT, 2), vec![2; 32]);
    }
}
