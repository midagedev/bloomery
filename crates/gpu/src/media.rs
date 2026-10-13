//! Media in a prompt call: the spans a caller splices into the ids, the per-sequence rotation table
//! an image's positions need, and the trait a body that takes media implements.
//!
//! A media span is a run of positions of the call whose rows the caller gives (the vision tower's
//! output, a learned delimiter) instead of a token embedding: [`MediaSpan`]. What a span means to a
//! body is the body's own, so a body names its span type ([`MediaBody::Span`]) and shares the
//! checks of [`MediaSpan`].
//!
//! The rotation table [`SeqRope`] is per sequence: every rope consumer reads `table[row]`, so a row
//! that holds a position triple's rotation (see [`vision::mrope`]) turns an image's heads without a
//! kernel knowing about M-RoPE. The table is allocated once, before any graph is captured, because
//! a captured graph bakes the buffer's address; every later change writes the same buffer.

use crate::model::ChainBody;
use crate::tensor::WindowMut;
use crate::{GpuError, GpuModel};
use cuda_core::{CudaStream, DeviceBuffer};
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use vision::mrope::{Canvas, MropeSeq, RowRewrite, SectionOrder, SeqRows};

/// One media span of a prompt call: its positions `at` in the call's ids and the bf16 row of each
/// of them (`n_embd` values).
#[derive(Clone)]
pub struct MediaSpan<'a> {
    /// The span's positions in the call's ids, non-empty.
    pub at: Range<usize>,
    /// The span's rows, `at.len()` rows of `n_embd` bf16 values.
    pub rows: &'a [u16],
}

impl std::fmt::Debug for MediaSpan<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MediaSpan")
            .field("at", &self.at)
            .field("rows", &self.rows.len())
            .finish()
    }
}

impl<'a> MediaSpan<'a> {
    /// The row of position `p` (`n_embd` bf16 values); `None` off the span.
    #[must_use]
    pub fn row(&self, p: usize, n_embd: usize) -> Option<&'a [u16]> {
        if !self.at.contains(&p) {
            return None;
        }
        let i = p - self.at.start;
        self.rows.get(i * n_embd..(i + 1) * n_embd)
    }
}

/// What [`CallRows::new`] refuses.
#[derive(Debug, PartialEq, Eq)]
pub enum RowsRefusal {
    /// A span's positions do not fit the stores' position type.
    Position { at: Range<usize>, first: u32 },
    /// A span's rows are not `at.len()` rows of `n_embd`.
    Rows {
        at: Range<usize>,
        values: usize,
        want: usize,
    },
    /// The spans, as absolute positions, are not ascending and disjoint.
    Order { spans: Vec<Range<u32>> },
}

impl fmt::Display for RowsRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RowsRefusal::Position { at, first } => write!(
                f,
                "a media span at {at:?} of a call from position {first}: its positions pass u32"
            ),
            RowsRefusal::Rows { at, values, want } => write!(
                f,
                "a media span at {at:?} with {values} row values, want {want}: one row of the \
                 text width a position"
            ),
            RowsRefusal::Order { spans } => write!(
                f,
                "the media spans at positions {spans:?}: ascending and disjoint"
            ),
        }
    }
}

/// The absolute positions of the span `at` of a call whose first position is `first`: one per row,
/// consecutive, so a span of `n` rows ends `n` positions after it starts and the id after the span
/// takes the position after its last row.
fn span_positions(first: u32, at: &Range<usize>) -> Result<Range<u32>, RowsRefusal> {
    let bound = |i: usize| {
        u32::try_from(i)
            .ok()
            .and_then(|i| first.checked_add(i))
            .ok_or_else(|| RowsRefusal::Position {
                at: at.clone(),
                first,
            })
    };
    Ok(bound(at.start)?..bound(at.end)?)
}

/// A call's media spans at their absolute positions: the rows a walk or a batch stages by the
/// position it writes, so a batch boundary that cuts a span reads the right rows. The one lookup
/// of a span row by absolute position (a body's media call and the draft that walks beside it
/// read the same rows through it) and the one widening of a bf16 row ([`widen_bf16`]).
pub struct CallRows<'a> {
    n_embd: usize,
    /// Ascending, disjoint; each span's rows are `positions.len()` rows of `n_embd`.
    spans: Vec<(Range<u32>, &'a [u16])>,
}

impl fmt::Debug for CallRows<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CallRows")
            .field("n_embd", &self.n_embd)
            .field(
                "spans",
                &self.spans.iter().map(|(at, _)| at).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl<'a> CallRows<'a> {
    /// The spans `(at, rows)` of a call whose first position is `first`: each one made whole
    /// ([`RowsRefusal::Rows`]) and absolute ([`span_positions`]), and the spans ascending and
    /// disjoint ([`RowsRefusal::Order`]), which [`CallRows::row`]'s search needs.
    pub fn new(
        first: u32,
        n_embd: usize,
        spans: impl IntoIterator<Item = (&'a Range<usize>, &'a [u16])>,
    ) -> Result<CallRows<'a>, RowsRefusal> {
        let spans = spans
            .into_iter()
            .map(|(at, rows)| {
                let want = at.len() * n_embd;
                if rows.len() != want {
                    return Err(RowsRefusal::Rows {
                        at: at.clone(),
                        values: rows.len(),
                        want,
                    });
                }
                span_positions(first, at).map(|p| (p, rows))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if spans.windows(2).any(|p| p[0].0.end > p[1].0.start) {
            return Err(RowsRefusal::Order {
                spans: spans.into_iter().map(|(at, _)| at).collect(),
            });
        }
        Ok(CallRows { n_embd, spans })
    }

    /// The width of a row, in values.
    #[must_use]
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// The row of position `p` (`n_embd` bf16 values); `None` off the spans.
    #[must_use]
    pub fn row(&self, p: u32) -> Option<&'a [u16]> {
        let k = self.spans.partition_point(|(at, _)| at.end <= p);
        let (at, rows) = self.spans.get(k).filter(|(at, _)| at.start <= p)?;
        let i = usize::try_from(p - at.start).ok()? * self.n_embd;
        rows.get(i..i + self.n_embd)
    }
}

/// A bf16 row widened exactly into `out`: each value's bits over 16 zero bits. The two are the
/// same width.
pub fn widen_bf16(row: &[u16], out: &mut [f32]) {
    assert_eq!(
        row.len(),
        out.len(),
        "a bf16 row widened into its own width"
    );
    for (v, &b) in out.iter_mut().zip(row) {
        *v = f32::from_bits(u32::from(b) << 16);
    }
}

/// The checks every body's media call makes of its spans, once at the entry: each span non-empty,
/// inside a call of `n_ids` ids, whole (`rows.len() = at.len() · n_embd`), and the spans ascending
/// and disjoint. A span the call's end cuts is refused: a span's rows come with its call or not at
/// all.
pub fn check_spans(
    what: &'static str,
    spans: &[MediaSpan<'_>],
    n_ids: usize,
    n_embd: usize,
) -> Result<(), GpuError> {
    for span in spans {
        let at = &span.at;
        if at.is_empty() || at.end > n_ids || span.rows.len() != at.len() * n_embd {
            return Err(GpuError::shape(
                what,
                format!(
                    "a media span of {} positions at {at:?} of a prompt of {n_ids} ids, {} rows \
                     of {n_embd}: the call holds a span whole or not at all",
                    at.len(),
                    span.rows.len() / n_embd.max(1),
                ),
            ));
        }
    }
    if spans.windows(2).any(|p| p[0].at.end > p[1].at.start) {
        return Err(GpuError::shape(
            what,
            format!(
                "the media spans {:?}: ascending, each inside the prompt, disjoint",
                spans.iter().map(|s| s.at.clone()).collect::<Vec<_>>()
            ),
        ));
    }
    Ok(())
}

/// How an image's rows lie on the sequence's positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageShape {
    /// `nx` tokens along a line, `ny` lines, in raster order: `nx · ny` positions.
    Grid(usize, usize),
    /// One image group of tiles, each followed by its separator rows: `canvas.rows()` positions,
    /// placed on a shared canvas whose origin is the token just before the span.
    Canvas(Canvas),
}

/// One image of a call: its rows and the shape its M-RoPE positions come from.
#[derive(Clone, Debug)]
pub struct ImageSpan<'a> {
    pub span: MediaSpan<'a>,
    pub shape: ImageShape,
}

impl ImageSpan<'_> {
    /// [`check_spans`] over `spans` and the shape of each: a grid or a canvas holds as many
    /// positions as the span does, and the call's images are all grids or all canvases.
    pub fn check(
        what: &'static str,
        spans: &[ImageSpan<'_>],
        n_ids: usize,
        n_embd: usize,
    ) -> Result<(), GpuError> {
        let plain: Vec<MediaSpan<'_>> = spans.iter().map(|s| s.span.clone()).collect();
        check_spans(what, &plain, n_ids, n_embd)?;
        for s in spans {
            let len = s.span.at.len();
            match s.shape {
                ImageShape::Grid(nx, ny) => {
                    if nx.checked_mul(ny) != Some(len) {
                        return Err(GpuError::shape(
                            what,
                            format!(
                                "an image span of {len} positions at {:?} on a {nx}x{ny} grid: a \
                                 grid holds nx·ny positions",
                                s.span.at
                            ),
                        ));
                    }
                }
                ImageShape::Canvas(canvas) => {
                    if canvas.rows() != len {
                        return Err(GpuError::shape(
                            what,
                            format!(
                                "an image span of {len} positions at {:?} on a canvas group of {} \
                                 rows: a group holds its tiles' tokens and their separators",
                                s.span.at,
                                canvas.rows()
                            ),
                        ));
                    }
                }
            }
        }
        ImageSpan::canvas_count(what, spans).map(|_| ())
    }

    /// The grid spans as the sequence's rotation table takes them, `(first row, nx, ny)`, for a
    /// call whose first position is `first`; a canvas span is not among them.
    #[must_use]
    pub fn rows_of(first: usize, spans: &[ImageSpan<'_>]) -> Vec<(usize, usize, usize)> {
        spans
            .iter()
            .filter_map(|s| match s.shape {
                ImageShape::Grid(nx, ny) => Some((first + s.span.at.start, nx, ny)),
                ImageShape::Canvas(_) => None,
            })
            .collect()
    }

    /// The canvas spans as the sequence's rotation table takes them, `(first row, canvas)`, for a
    /// call whose first position is `first`; a grid span is not among them.
    #[must_use]
    pub fn canvases_of(first: usize, spans: &[ImageSpan<'_>]) -> Vec<(usize, Canvas)> {
        spans
            .iter()
            .filter_map(|s| match s.shape {
                ImageShape::Canvas(c) => Some((first + s.span.at.start, c)),
                ImageShape::Grid(..) => None,
            })
            .collect()
    }

    /// How many of `spans` are canvases: none or all of them, a call that mixes the two shapes is
    /// refused by name (a text file's images are of one kind, and the table takes a call's images
    /// in one change).
    fn canvas_count(what: &'static str, spans: &[ImageSpan<'_>]) -> Result<usize, GpuError> {
        let n = spans
            .iter()
            .filter(|s| matches!(s.shape, ImageShape::Canvas(_)))
            .count();
        if n != 0 && n != spans.len() {
            return Err(GpuError::shape(
                what,
                format!(
                    "{n} canvas image spans among {}: a call's images are all grids or all \
                     canvas groups",
                    spans.len()
                ),
            ));
        }
        Ok(n)
    }
}

/// A body whose prompt call takes media: the spans' rows stand in for the token embeddings of
/// their positions.
pub trait MediaBody: ChainBody {
    /// One span of a call as this body takes it: [`MediaSpan`] with what the body's rows and
    /// engine rules need beside it.
    type Span<'a>;

    /// The common span of `s`: the positions and rows every body's span carries.
    fn media_span<'s>(s: &'s Self::Span<'_>) -> &'s MediaSpan<'s>;

    /// Why a media call of `spans` spans on `m` is refused before it enters the body (a schedule
    /// with no place for a span's rows), `None` when it is taken.
    fn media_refusal(m: &GpuModel<Self>, spans: usize) -> Option<String> {
        let _ = (m, spans);
        None
    }

    /// Feed `ids` from the model's position on with `spans` spliced in, and return the greedy
    /// token after the last id. A span the call does not hold whole, spans out of order or of
    /// another shape are refused by name.
    fn prefill_media(
        m: &mut GpuModel<Self>,
        ids: &[u32],
        spans: &[Self::Span<'_>],
    ) -> Result<u32, GpuError>;

    /// [`MediaBody::prefill_media`] for a call that reads no token and returns every position's
    /// final-norm hidden row instead, `ids.len()` rows of the model's width. A body with no such
    /// call refuses it by name.
    fn prefill_hidden_media(
        m: &mut GpuModel<Self>,
        ids: &[u32],
        spans: &[Self::Span<'_>],
    ) -> Result<Vec<f32>, GpuError> {
        let _ = (m, ids, spans);
        Err(GpuError::shape(
            "MediaBody::prefill_hidden_media",
            "this body has no hidden-tail media call",
        ))
    }
}

/// One sequence's rotation table on the card, `ctx` rows of `rot` values: row `r` is what the
/// rope consumers read for the sequence's row `r`. A text sequence's table is the plain table
/// bit for bit; images shift the rows ([`SeqRows`]). The buffer is made once and every change
/// writes it in place, never before capture of the graphs that read it is over.
///
/// A change updates the host's record of the images and then the card's rows. A copy that fails
/// is a driver failure the model is poisoned by, so the two do not part silently.
pub struct SeqRope {
    table: DeviceBuffer<f32>,
    rows: SeqRows,
}

const WHAT: &str = "SeqRope";

impl SeqRope {
    /// A text sequence of `ctx` rows over the plain table `base` (the rope table of positions
    /// `0..ctx`, `ctx` rows of equal width) with the text file's `rope.dimension_sections` read in
    /// its `rope.section_order`, copied to a card buffer of its own. `base` is shared by every
    /// sequence of a model.
    pub fn new(
        stream: &CudaStream,
        base: Arc<[f32]>,
        sections: [u32; 4],
        order: SectionOrder,
        ctx: usize,
    ) -> Result<SeqRope, GpuError> {
        let rows = SeqRows::with_order(base, sections, order, ctx).map_err(shape)?;
        let table = DeviceBuffer::from_host(stream, rows.base())?;
        Ok(SeqRope { table, rows })
    }

    /// Add the images of a call, each `(first row, nx, ny)` of the sequence, in ascending order
    /// after the held ones: the rows from the first to the end are rewritten, one copy. A refused
    /// call (an overlap, an image past `ctx`) changes nothing.
    pub fn push_spans(
        &mut self,
        stream: &CudaStream,
        spans: &[(usize, usize, usize)],
    ) -> Result<(), GpuError> {
        let rewrite = self.rows.push_spans(spans).map_err(shape)?;
        self.write(stream, rewrite)
    }

    /// [`SeqRope::push_spans`] for image groups, each `(first row, canvas)` where the first row
    /// is the one after the group's origin token.
    pub fn push_canvases(
        &mut self,
        stream: &CudaStream,
        groups: &[(usize, Canvas)],
    ) -> Result<(), GpuError> {
        let rewrite = self.rows.push_canvases(groups).map_err(shape)?;
        self.write(stream, rewrite)
    }

    /// Add the images of a call whose first position is `first`: the grids by
    /// [`SeqRope::push_spans`], the canvases by [`SeqRope::push_canvases`]. A call that mixes the
    /// two is refused by name and changes nothing.
    pub fn push_images(
        &mut self,
        stream: &CudaStream,
        first: usize,
        spans: &[ImageSpan<'_>],
    ) -> Result<(), GpuError> {
        if ImageSpan::canvas_count(WHAT, spans)? == 0 {
            self.push_spans(stream, &ImageSpan::rows_of(first, spans))
        } else {
            self.push_canvases(stream, &ImageSpan::canvases_of(first, spans))
        }
    }

    /// Keep the first `row` rows: the images from `row` on go and the rows after it are rewritten
    /// to the positions the kept images leave. A cut inside an image is refused by name.
    pub fn cut(&mut self, stream: &CudaStream, row: usize) -> Result<(), GpuError> {
        let rewrite = self.rows.cut(row).map_err(shape)?;
        self.write(stream, rewrite)
    }

    /// The images the table holds, to keep across a prompt cache's save and a slot file.
    #[must_use]
    pub fn saved(&self) -> MropeSeq {
        self.rows.seq().clone()
    }

    /// Take `seq` (from [`SeqRope::saved`]) as the table's images.
    pub fn restore(&mut self, stream: &CudaStream, seq: MropeSeq) -> Result<(), GpuError> {
        let rewrite = self.rows.restore(seq).map_err(shape)?;
        self.write(stream, rewrite)
    }

    /// The card table: the buffer the rope consumers read, at the same address for the table's
    /// life.
    #[must_use]
    pub fn table(&self) -> &DeviceBuffer<f32> {
        &self.table
    }

    /// The host's record of the images.
    #[must_use]
    pub fn rows(&self) -> &SeqRows {
        &self.rows
    }

    /// Copy the rewritten rows into the table.
    fn write(&mut self, stream: &CudaStream, rewrite: Option<RowRewrite>) -> Result<(), GpuError> {
        let Some(RowRewrite { from, values }) = rewrite else {
            return Ok(());
        };
        if values.is_empty() {
            return Ok(());
        }
        let byte = from * self.rows.rot() * size_of::<f32>();
        let mut window: WindowMut<'_, f32> =
            WindowMut::of_mut(&mut self.table, byte, values.len())?;
        window.copy_from_host(stream, &values)?;
        Ok(())
    }
}

fn shape(e: vision::mrope::MropeError) -> GpuError {
    GpuError::shape(WHAT, e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECTIONS: [u32; 4] = [11, 11, 10, 0];
    const ROT: usize = 64;
    const CTX: usize = 300;

    /// A table of `ctx` rows whose value says where it came from: row `q`, value `k` is
    /// `1000·q + k`.
    fn table(ctx: usize) -> Arc<[f32]> {
        (0..ctx * ROT)
            .map(|i| (1000 * (i / ROT) + i % ROT) as f32)
            .collect()
    }

    fn base() -> Arc<[f32]> {
        table(CTX)
    }

    /// The rotation row of sequence row `r` from the rule alone, written out here and not taken
    /// from [`MropeSeq`]: the images as `(first row, nx, ny)`, ggml's interleaved M-RoPE (pair `s`
    /// of sector `s mod 32` takes the h position at sectors 1 mod 3 below 33, the w position at
    /// 2 mod 3 below 30, the t position at 0 mod 3 below 33).
    fn oracle_row(base: &[f32], images: &[(usize, usize, usize)], r: usize) -> Vec<f32> {
        let mut saved = 0;
        let mut pos = (r, r, r);
        for &(at, nx, ny) in images {
            if r >= at + nx * ny {
                saved += nx * ny - nx.max(ny);
                pos = (r - saved, r - saved, r - saved);
            } else if r >= at {
                let i = r - at;
                let p0 = at - saved;
                pos = (p0, p0 + i / nx, p0 + i % nx);
            }
        }
        let (t, h, w) = pos;
        let mut row = vec![0.0; ROT];
        for s in 0..ROT / 2 {
            let sector = s % 32;
            let q = if sector % 3 == 1 && sector < 33 {
                h
            } else if sector % 3 == 2 && sector < 30 {
                w
            } else if sector % 3 == 0 && sector < 33 {
                t
            } else {
                0
            };
            row[2 * s..2 * s + 2].copy_from_slice(&base[q * ROT + 2 * s..q * ROT + 2 * s + 2]);
        }
        row
    }

    /// The card table of `rope` equals, bit for bit, both the full rebuild by
    /// [`MropeSeq::rows_into`] and the rows [`oracle_row`] writes out.
    fn expect(
        stream: &CudaStream,
        rope: &SeqRope,
        base: &[f32],
        images: &[(usize, usize, usize)],
        what: &str,
    ) {
        let got = rope.table().to_host_vec(stream).expect("the table");
        assert_eq!(got.len(), CTX * ROT, "{what}: table length");
        let mut seq = MropeSeq::new();
        for &(at, nx, ny) in images {
            seq.push(at, nx, ny).expect("push");
        }
        let mut full = vec![0.0; CTX * ROT];
        seq.rows_into(base, ROT, SECTIONS, SectionOrder::Tyxz, 0..CTX, &mut full)
            .expect("the full rebuild");
        let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        let bad =
            (0..CTX).find(|&r| !same(&got[r * ROT..(r + 1) * ROT], &full[r * ROT..(r + 1) * ROT]));
        assert_eq!(bad, None, "{what}: first row off the full rebuild");
        let bad =
            (0..CTX).find(|&r| !same(&got[r * ROT..(r + 1) * ROT], &oracle_row(base, images, r)));
        assert_eq!(bad, None, "{what}: first row off the written-out rule");
    }

    /// S1: the card table after `push_spans`, `cut`, `saved` and `restore` is the host's rows bit
    /// for bit, over a first image, a second, a cut that drops the second, a cut inside the
    /// first (refused, the table unchanged), a cut that drops the first and a restore.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_card_table_is_the_host_rows() {
        let (_ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let base = base();
        let mut rope = SeqRope::new(&stream, base.clone(), SECTIONS, SectionOrder::Tyxz, CTX)
            .expect("a text sequence");
        let address = rope.table().cu_deviceptr();
        expect(&stream, &rope, &base, &[], "text sequence");
        let a = (3, 14, 14);
        let b = (210, 4, 2);
        rope.push_spans(&stream, &[a]).expect("image a");
        expect(&stream, &rope, &base, &[a], "after image a");
        rope.push_spans(&stream, &[b]).expect("image b");
        expect(&stream, &rope, &base, &[a, b], "after image b");
        let both = rope.saved();
        rope.cut(&stream, 210).expect("a cut at image b");
        expect(&stream, &rope, &base, &[a], "after cutting image b");
        let inside = rope.cut(&stream, 100).expect_err("a cut inside image a");
        assert!(inside.to_string().contains("inside the image"), "{inside}");
        expect(&stream, &rope, &base, &[a], "after the refused cut");
        rope.cut(&stream, 3).expect("a cut at image a");
        expect(&stream, &rope, &base, &[], "after cutting image a");
        rope.restore(&stream, both).expect("a restore");
        expect(&stream, &rope, &base, &[a, b], "after the restore");
        let past = rope
            .push_spans(&stream, &[(290, 4, 4)])
            .expect_err("an image past the sequence");
        assert!(past.to_string().contains("past"), "{past}");
        expect(&stream, &rope, &base, &[a, b], "after the refused push");
        assert_eq!(
            rope.table().cu_deviceptr(),
            address,
            "the buffer never moves"
        );
    }

    /// The sections of a text file whose fourth section reads the slot the order puts last.
    const SECTIONS_Z: [u32; 4] = [10, 10, 10, 2];
    const CANVAS_CTX: usize = 400;

    /// The 1000x700 image's group: an overview and a 2x2 grid of slices, 10x7 tokens each, 360
    /// rows.
    fn group_1000x700() -> Canvas {
        let canvas = Canvas::new((10, 7), (2, 2), (10, 7)).expect("the 2x2 group");
        assert_eq!((canvas.rows(), canvas.tiles()), (360, 5));
        canvas
    }

    /// The table `order` gives a sequence holding `group` at `at`, from [`SeqRows`] alone: the
    /// plain table with the rewrite of `push_canvases` spliced in.
    fn table_of(base: &Arc<[f32]>, order: SectionOrder, at: usize, group: Canvas) -> Vec<f32> {
        let mut rows =
            SeqRows::with_order(base.clone(), SECTIONS_Z, order, CANVAS_CTX).expect("rows");
        let rewrite = rows
            .push_canvases(&[(at, group)])
            .expect("the group")
            .expect("a rewrite");
        let mut table = base.to_vec();
        table[rewrite.from * ROT..rewrite.from * ROT + rewrite.values.len()]
            .copy_from_slice(&rewrite.values);
        table
    }

    fn same(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// The card table of a sequence opened in `zyxt` that takes the 1000x700 group through
    /// `push_images` equals the rows `SeqRows::push_canvases` makes, bit for bit, and differs from
    /// the `tyxz` rows of the same group; the table keeps its address; a save, a cut of the group
    /// and a restore give the host's rows.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_canvas_groups_card_table_is_the_host_rows_in_zyxt() {
        let (_ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let base = table(CANVAS_CTX);
        let group = group_1000x700();
        // A call that starts at row 3 holds the group at its ids 2..362: sequence rows 5..365, the
        // origin token at row 4.
        let (first, at) = (3, 2..2 + group.rows());
        let want = table_of(&base, SectionOrder::Zyxt, first + at.start, group);
        let tyxz = table_of(&base, SectionOrder::Tyxz, first + at.start, group);
        assert!(
            !same(&want, &tyxz),
            "the two orders give the group other rows"
        );

        let mut rope = SeqRope::new(
            &stream,
            base.clone(),
            SECTIONS_Z,
            SectionOrder::Zyxt,
            CANVAS_CTX,
        )
        .expect("a text sequence");
        let address = rope.table().cu_deviceptr();
        let span = ImageSpan {
            span: MediaSpan { at, rows: &[] },
            shape: ImageShape::Canvas(group),
        };
        rope.push_images(&stream, first, std::slice::from_ref(&span))
            .expect("the group");
        let got = rope.table().to_host_vec(&stream).expect("the table");
        assert!(same(&got, &want), "the card table is the host rows in zyxt");

        // The host's own record of the same changes: its cut gives the rows the card's must.
        let mut host =
            SeqRows::with_order(base.clone(), SECTIONS_Z, SectionOrder::Zyxt, CANVAS_CTX)
                .expect("rows");
        host.push_canvases(&[(first + 2, group)])
            .expect("the group");
        let cut = host.cut(5).expect("a cut at the group").expect("a rewrite");
        let mut after_cut = want.clone();
        after_cut[cut.from * ROT..cut.from * ROT + cut.values.len()].copy_from_slice(&cut.values);
        let held = rope.saved();
        rope.cut(&stream, 5).expect("a cut at the group");
        let got = rope.table().to_host_vec(&stream).expect("the table");
        assert!(same(&got, &after_cut), "the group cut away: the host rows");
        rope.restore(&stream, held).expect("a restore");
        let got = rope.table().to_host_vec(&stream).expect("the table");
        assert!(same(&got, &want), "the restored group's rows");
        assert_eq!(
            rope.table().cu_deviceptr(),
            address,
            "the buffer never moves"
        );
    }
}

#[cfg(test)]
mod image_spans {
    use super::{ImageShape, ImageSpan, MediaSpan};
    use vision::mrope::Canvas;

    const N_EMBD: usize = 2;

    fn canvas() -> Canvas {
        Canvas::new((10, 7), (2, 2), (10, 7)).expect("the 1000x700 group")
    }

    /// A span of `len` positions from id `from` with its rows.
    fn span(rows: &[u16], from: usize, len: usize, shape: ImageShape) -> ImageSpan<'_> {
        ImageSpan {
            span: MediaSpan {
                at: from..from + len,
                rows: &rows[..len * N_EMBD],
            },
            shape,
        }
    }

    /// A canvas span holds as many positions as its group has rows (tokens and separators): 360
    /// for the 1000x700 group. Any other length is refused by name, a grid's rule is as it was.
    #[test]
    fn a_canvas_span_of_the_wrong_length_is_refused_by_name() {
        let rows = vec![0u16; 400 * N_EMBD];
        let check = |len: usize, shape| {
            let spans = [span(&rows, 1, len, shape)];
            ImageSpan::check("test", &spans, 1 + len, N_EMBD).map_err(|e| e.to_string())
        };
        let group = ImageShape::Canvas(canvas());
        assert_eq!(check(360, group), Ok(()));
        for len in [359, 361, 280] {
            let e = check(len, group).expect_err("a span off the group's rows");
            assert!(
                e.contains(&format!("an image span of {len} positions"))
                    && e.contains("canvas group of 360 rows"),
                "{e}"
            );
        }
        assert_eq!(check(6, ImageShape::Grid(3, 2)), Ok(()));
        let e = check(7, ImageShape::Grid(3, 2)).expect_err("a grid off nx·ny");
        assert!(
            e.contains("on a 3x2 grid: a grid holds nx·ny positions"),
            "{e}"
        );
    }

    /// A call's images are all grids or all canvases; the table takes each kind by its own list,
    /// at the call's first position.
    #[test]
    fn a_call_is_all_grids_or_all_canvases() {
        let rows = vec![0u16; 400 * N_EMBD];
        let grid = span(&rows, 1, 6, ImageShape::Grid(3, 2));
        let group = span(&rows, 7, 360, ImageShape::Canvas(canvas()));
        let mixed = [grid.clone(), group.clone()];
        let e = ImageSpan::check("test", &mixed, 367, N_EMBD).expect_err("grids with a canvas");
        assert!(
            e.to_string().contains("1 canvas image spans among 2"),
            "{e}"
        );
        let grids = [grid.clone(), span(&rows, 9, 6, ImageShape::Grid(2, 3))];
        assert_eq!(ImageSpan::rows_of(10, &grids), [(11, 3, 2), (19, 2, 3)]);
        assert!(ImageSpan::canvases_of(10, &grids).is_empty());
        let groups = [group];
        assert_eq!(ImageSpan::canvases_of(10, &groups), [(17, canvas())]);
        assert!(ImageSpan::rows_of(10, &groups).is_empty());
    }
}

#[cfg(test)]
mod call_rows {
    use super::{CallRows, RowsRefusal, span_positions, widen_bf16};

    const N: usize = 3;

    /// bf16 bits of a value whose low 16 f32 bits are zero.
    fn bf16(x: f32) -> u16 {
        (x.to_bits() >> 16) as u16
    }

    /// Two rows of `N` bf16 values, 1.5 2.5 3.5 then 4.5 5.5 6.5.
    fn rows() -> Vec<u16> {
        (0..2 * N).map(|i| bf16(1.5 + i as f32)).collect()
    }

    #[test]
    fn span_positions_are_consecutive_and_save_none() {
        // 3 rows from index 2 of a call at 10: positions 12, 13, 14; the id after takes 15.
        assert_eq!(span_positions(10, &(2..5)), Ok(12..15));
        let at = 2..5;
        assert_eq!(
            span_positions(10, &at).map(|p| p.end - p.start),
            Ok(u32::try_from(at.len()).expect("small"))
        );
        // The first position past the span is the call's first plus the span's end index.
        assert_eq!(span_positions(10, &at).map(|p| p.end), Ok(10 + 5));
        assert_eq!(
            span_positions(u32::MAX - 1, &(0..3)),
            Err(RowsRefusal::Position {
                at: 0..3,
                first: u32::MAX - 1
            })
        );
    }

    #[test]
    fn a_position_reads_the_row_of_its_span_by_absolute_position() {
        // A call from position 5 over five ids, the span at 2..4: positions 7 and 8.
        let (at, media) = (2..4, rows());
        let call = CallRows::new(5, N, [(&at, media.as_slice())]).expect("rows");
        assert_eq!(call.n_embd(), N);
        assert_eq!(call.row(7), Some(&media[..N]));
        assert_eq!(call.row(8), Some(&media[N..]));
        // Off the span, whichever side.
        for p in [0, 5, 6, 9, 10] {
            assert_eq!(call.row(p), None, "position {p}");
        }
    }

    #[test]
    fn several_spans_are_searched_in_order_and_a_gap_holds_no_row() {
        let (a, b) = (0..1, 3..5);
        let (ra, rb) = (rows()[..N].to_vec(), rows());
        let call =
            CallRows::new(0, N, [(&a, ra.as_slice()), (&b, rb.as_slice())]).expect("two spans");
        assert_eq!(call.row(0), Some(&ra[..]));
        assert_eq!(call.row(1), None);
        assert_eq!(call.row(2), None);
        assert_eq!(call.row(3), Some(&rb[..N]));
        assert_eq!(call.row(4), Some(&rb[N..]));
    }

    #[test]
    fn spans_that_do_not_fit_are_refused_by_name() {
        let media = rows();
        let short = CallRows::new(0, N, [(&(0..3), media.as_slice())]).err();
        assert_eq!(
            short,
            Some(RowsRefusal::Rows {
                at: 0..3,
                values: 2 * N,
                want: 3 * N
            })
        );
        let (a, b) = (3..5, 0..2);
        let out_of_order = CallRows::new(10, N, [(&a, media.as_slice()), (&b, media.as_slice())])
            .expect_err("spans out of order");
        assert_eq!(
            out_of_order,
            RowsRefusal::Order {
                spans: vec![13..15, 10..12]
            }
        );
        assert!(out_of_order.to_string().contains("ascending and disjoint"));
    }

    #[test]
    fn a_bf16_row_widens_to_the_f32_of_its_bits() {
        let media = rows();
        let mut out = vec![0.0; N];
        widen_bf16(&media[..N], &mut out);
        assert_eq!(out, [1.5, 2.5, 3.5]);
        assert!(
            out.iter()
                .zip(&media)
                .all(|(v, &b)| v.to_bits() == u32::from(b) << 16)
        );
    }

    #[test]
    #[should_panic(expected = "a bf16 row widened into its own width")]
    fn a_widening_into_another_width_is_refused() {
        widen_bf16(&rows(), &mut [0.0; N]);
    }
}
