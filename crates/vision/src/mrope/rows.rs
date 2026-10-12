//! [`SeqRows`]: the rotation table of one sequence and the rows a change of its images rewrites.

use super::canvas::Canvas;
use super::order::SectionOrder;
use super::{MropeError, MropeSeq};

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
    order: SectionOrder,
    ctx: usize,
    seq: MropeSeq,
}

impl SeqRows {
    /// A sequence of `ctx` rows over the plain table `base` (`ctx` rows of `rot` values, the
    /// rope table of positions `0..ctx`); `sections` are the text file's
    /// `rope.dimension_sections`. The sections read the slots in [`SectionOrder::Tyxz`], the order
    /// of every model but MiniCPM-V 4.7 ([`SeqRows::with_order`]). A shape
    /// [`MropeSeq::rows_into`] would refuse is refused here.
    pub fn new(
        base: std::sync::Arc<[f32]>,
        sections: [u32; 4],
        ctx: usize,
    ) -> Result<SeqRows, MropeError> {
        SeqRows::with_order(base, sections, SectionOrder::Tyxz, ctx)
    }

    /// [`SeqRows::new`] for a text file whose `rope.section_order` is `order`.
    pub fn with_order(
        base: std::sync::Arc<[f32]>,
        sections: [u32; 4],
        order: SectionOrder,
        ctx: usize,
    ) -> Result<SeqRows, MropeError> {
        if ctx == 0 || base.is_empty() || !base.len().is_multiple_of(ctx) {
            return Err(MropeError::Shape(format!(
                "a table of {} values for a sequence of {ctx} rows",
                base.len()
            )));
        }
        let rot = base.len() / ctx;
        MropeSeq::new().rows_into(&base, rot, sections, order, 0..0, &mut [])?;
        Ok(SeqRows {
            base,
            rot,
            sections,
            order,
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
            self.fit(at, nx * ny)?;
        }
        self.install(next, from)
    }

    /// Add image groups, each `(first row, canvas)` in ascending order after the ones held, as
    /// [`SeqRows::push_spans`] does for raster images.
    pub fn push_canvases(
        &mut self,
        groups: &[(usize, Canvas)],
    ) -> Result<Option<RowRewrite>, MropeError> {
        let Some(&(from, _)) = groups.first() else {
            return Ok(None);
        };
        let mut next = self.seq.clone();
        for &(at, canvas) in groups {
            next.push_canvas(at, canvas)?;
            self.fit(at, canvas.rows())?;
        }
        self.install(next, from)
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
        self.install(next, row)
    }

    /// Take `seq` as the images of the sequence (a saved sequence coming back): the rows from
    /// the first image that differs from the held ones change; the same images change nothing.
    pub fn restore(&mut self, seq: MropeSeq) -> Result<Option<RowRewrite>, MropeError> {
        for (at, nx, ny) in seq.spans() {
            self.fit(at, nx * ny)?;
        }
        match self.seq.first_difference(&seq) {
            None => Ok(None),
            Some(from) => self.install(seq, from),
        }
    }

    /// Refuse an image whose `rows` rows from `at` reach past the sequence's rows.
    fn fit(&self, at: usize, rows: usize) -> Result<(), MropeError> {
        let end = at + rows;
        if end > self.ctx {
            return Err(MropeError::PastSequence {
                at,
                end,
                ctx: self.ctx,
            });
        }
        Ok(())
    }

    /// Take `next` as the images of the sequence and return the rows from `from` on that it
    /// gives. The rows are made before `next` is taken, so a refusal leaves the held images as
    /// they were.
    fn install(&mut self, next: MropeSeq, from: usize) -> Result<Option<RowRewrite>, MropeError> {
        let rewrite = self.rewrite(&next, from)?;
        self.seq = next;
        Ok(Some(rewrite))
    }

    /// The rows `from..ctx` of the table `seq` gives.
    fn rewrite(&self, seq: &MropeSeq, from: usize) -> Result<RowRewrite, MropeError> {
        let mut values = vec![0.0; (self.ctx - from) * self.rot];
        seq.rows_into(
            &self.base,
            self.rot,
            self.sections,
            self.order,
            from..self.ctx,
            &mut values,
        )?;
        Ok(RowRewrite { from, values })
    }
}

#[cfg(test)]
mod tests {
    use super::{MropeError, MropeSeq, RowRewrite, SeqRows};
    use crate::mrope::fixture::{ROT, SECTIONS, table};
    use crate::mrope::{Canvas, SectionOrder};
    use std::sync::Arc;

    const CTX: usize = 300;

    /// The rows of `seq` over the whole sequence, rebuilt from scratch.
    fn full(base: &[f32], seq: &MropeSeq) -> Vec<f32> {
        let mut out = vec![0.0; CTX * ROT];
        seq.rows_into(base, ROT, SECTIONS, SectionOrder::Tyxz, 0..CTX, &mut out)
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

    /// A canvas group pushed, cut and restored rewrites from its first row and leaves the table a
    /// rebuild from scratch makes, under `zyxt` as under `tyxz`; two groups of the same row count
    /// but another layout are different images to a restore.
    #[test]
    fn a_canvas_group_rewrites_like_an_image() {
        let base: Arc<[f32]> = table(CTX, ROT).into();
        for order in [SectionOrder::Tyxz, SectionOrder::Zyxt] {
            let mut rows = SeqRows::with_order(base.clone(), SECTIONS, order, CTX).expect("rows");
            let mut copy = base.to_vec();
            let full = |seq: &MropeSeq| {
                let mut out = vec![0.0; CTX * ROT];
                seq.rows_into(&base, ROT, SECTIONS, order, 0..CTX, &mut out)
                    .expect("rebuild");
                out
            };
            let group = Canvas::new((3, 2), (2, 1), (2, 2)).expect("canvas");
            let rw = rows.push_canvases(&[(5, group)]).expect("push");
            assert_eq!(rw.as_ref().map(|r| r.from), Some(5));
            apply(&mut copy, &rw);
            assert!(copy == full(rows.seq()), "{order:?}: pushed");
            let held = rows.seq().clone();
            // The group is (6 + 2) + (4 + 2) + (4 + 1) = 19 rows, rows 5..24.
            assert_eq!(rows.seq().spans().collect::<Vec<_>>(), [(5, 19, 1)]);
            assert!(matches!(
                rows.push_canvases(&[(23, group)]),
                Err(MropeError::Overlap { .. })
            ));
            assert!(matches!(
                rows.push_canvases(&[(290, group)]),
                Err(MropeError::PastSequence { .. })
            ));
            assert!(matches!(
                rows.cut(10),
                Err(MropeError::CutInsideSpan { .. })
            ));
            // Another layout with the same 19 rows is another image to a restore.
            let other = Canvas::new((3, 2), (2, 1), (1, 4)).expect("canvas");
            assert_eq!(other.rows(), group.rows());
            let mut seq = MropeSeq::new();
            seq.push_canvas(5, other).expect("other");
            let rw = rows.restore(seq).expect("restore");
            assert_eq!(rw.as_ref().map(|r| r.from), Some(5));
            apply(&mut copy, &rw);
            assert!(copy == full(rows.seq()), "{order:?}: restored");
            assert_eq!(rows.restore(rows.seq().clone()), Ok(None));
            let rw = rows.cut(5).expect("cut at the group");
            apply(&mut copy, &rw);
            assert!(copy == *base, "{order:?}: no image, the plain table");
            let rw = rows.restore(held).expect("restore the first");
            apply(&mut copy, &rw);
            assert!(copy == full(rows.seq()), "{order:?}: restored again");
        }
    }
}
