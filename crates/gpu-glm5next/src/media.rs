//! Media rows in a prompt call: where the batch's embedding gather takes a position's row from.
//!
//! An image's rows stand in for the token embeddings of the span's positions. The text model reads
//! them as it reads any row, so the only facts GLM adds to the common span
//! ([`bloomery_gpu::media::MediaSpan`]) are these:
//! - the position rule: the type is normal, an image's rows take one position each, consecutive
//!   from the span's first, and the text after the image continues at the next one (an image saves
//!   no position, unlike an M-RoPE grid);
//! - the image token: every id of a span is the model's `<|image|>`, the begin and end markers
//!   around it being ordinary ids outside the span;
//! - the row's form: `n_embd` bf16 values, widened to f32 and repeated into the hyper-connection
//!   streams as a token's embedding row is.
//!
//! The module holds no card type, so the engine's gather and its tests share one code path.

use std::fmt;
use std::ops::Range;

/// What a media call's own checks refuse; the common span checks
/// ([`bloomery_gpu::media::check_spans`]) refuse the rest.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// A span's id is not the image token.
    NotImage {
        at: Range<usize>,
        index: usize,
        id: u32,
        image: u32,
    },
    /// A span's positions do not fit the stores' position type.
    Position { at: Range<usize>, first: u32 },
    /// A span reaches past the call's ids.
    Outside { at: Range<usize>, n_ids: usize },
    /// A span's rows are not `at.len()` rows of `n_embd`.
    Rows {
        at: Range<usize>,
        values: usize,
        want: usize,
    },
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::NotImage {
                at,
                index,
                id,
                image,
            } => write!(
                f,
                "a media span at {at:?} holds id {id} at index {index}, not the image token \
                 {image}: every position of a span carries it"
            ),
            Refusal::Position { at, first } => write!(
                f,
                "a media span at {at:?} of a call from position {first}: its positions pass u32"
            ),
            Refusal::Outside { at, n_ids } => write!(
                f,
                "a media span at {at:?} of a call of {n_ids} ids: a span lies inside the call"
            ),
            Refusal::Rows { at, values, want } => write!(
                f,
                "a media span at {at:?} with {values} row values, want {want}: one row of the \
                 text width a position"
            ),
        }
    }
}

/// The absolute positions of the span `at` of a call whose first position is `first`: one per row,
/// consecutive, so a span of `n` rows ends `n` positions after it starts and the id after the span
/// takes the position after its last row.
pub(crate) fn span_positions(first: u32, at: &Range<usize>) -> Result<Range<u32>, Refusal> {
    let bound = |i: usize| {
        u32::try_from(i)
            .ok()
            .and_then(|i| first.checked_add(i))
            .ok_or_else(|| Refusal::Position {
                at: at.clone(),
                first,
            })
    };
    Ok(bound(at.start)?..bound(at.end)?)
}

/// `ids[at]` all the image token, or the first id that is not.
pub(crate) fn check_image_ids(ids: &[u32], at: &Range<usize>, image: u32) -> Result<(), Refusal> {
    let span = ids.get(at.clone()).ok_or_else(|| Refusal::Outside {
        at: at.clone(),
        n_ids: ids.len(),
    })?;
    match span.iter().position(|&id| id != image) {
        Some(i) => Err(Refusal::NotImage {
            at: at.clone(),
            index: at.start + i,
            id: span[i],
            image,
        }),
        None => Ok(()),
    }
}

/// A call's checked media spans at their absolute positions: the rows a batch stages by the
/// position it writes, so a batch boundary that cuts a span reads the right rows.
pub(crate) struct CallRows<'a> {
    n_embd: usize,
    /// Ascending, disjoint; each span's rows are `positions.len()` rows of `n_embd`.
    spans: Vec<(Range<u32>, &'a [u16])>,
}

impl<'a> CallRows<'a> {
    /// The spans `(at, rows)` of a call whose first position is `first`, in order (the common span
    /// checks) and each one made whole ([`Refusal::Rows`]) and absolute ([`span_positions`]).
    pub(crate) fn new(
        first: u32,
        n_embd: usize,
        spans: impl IntoIterator<Item = (&'a Range<usize>, &'a [u16])>,
    ) -> Result<CallRows<'a>, Refusal> {
        let spans = spans
            .into_iter()
            .map(|(at, rows)| {
                let want = at.len() * n_embd;
                if rows.len() != want {
                    return Err(Refusal::Rows {
                        at: at.clone(),
                        values: rows.len(),
                        want,
                    });
                }
                span_positions(first, at).map(|p| (p, rows))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(CallRows { n_embd, spans })
    }

    /// The row of position `p` (`n_embd` bf16 values); `None` off the spans.
    fn row(&self, p: u32) -> Option<&'a [u16]> {
        let k = self.spans.partition_point(|(at, _)| at.end <= p);
        let (at, rows) = self.spans.get(k).filter(|(at, _)| at.start <= p)?;
        let i = usize::try_from(p - at.start).ok()? * self.n_embd;
        rows.get(i..i + self.n_embd)
    }
}

/// Stage position `p`'s embedding row into `out` (`out.len()` a multiple of `n_embd`, one copy of
/// the row each): the call's media row where a span holds `p` (every row of a span is there,
/// [`CallRows::new`]), otherwise what `text` fills for `token`.
pub(crate) fn stage_row<E>(
    media: Option<&CallRows<'_>>,
    p: u32,
    token: u32,
    out: &mut [f32],
    text: impl FnOnce(u32, &mut [f32]) -> Result<(), E>,
) -> Result<(), E> {
    let Some((media, row)) = media.and_then(|m| Some((m, m.row(p)?))) else {
        return text(token, out);
    };
    let n = media.n_embd;
    let (first, rest) = out.split_at_mut(n);
    for (v, &b) in first.iter_mut().zip(row) {
        *v = f32::from_bits(u32::from(b) << 16);
    }
    for copy in rest.chunks_exact_mut(n) {
        copy.copy_from_slice(first);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CallRows, Refusal, check_image_ids, span_positions, stage_row};
    use std::ops::Range;

    const IMG: u32 = 900;
    const N: usize = 3;
    const STREAMS: usize = 4;

    /// The text rows' fill: every value the token's id plus the value's index over 100.
    fn text(token: u32, out: &mut [f32]) -> Result<(), String> {
        let n = out.len() / STREAMS;
        for (i, v) in out.iter_mut().enumerate() {
            *v = token as f32 + (i % n) as f32 / 100.0;
        }
        Ok(())
    }

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
            Err(Refusal::Position {
                at: 0..3,
                first: u32::MAX - 1
            })
        );
    }

    #[test]
    fn a_span_id_that_is_not_the_image_token_is_refused_at_its_index() {
        let ids = [5, IMG, IMG, IMG, 6];
        assert_eq!(check_image_ids(&ids, &(1..4), IMG), Ok(()));
        for bad in 1..4 {
            let mut ids = ids;
            ids[bad] = 7;
            assert_eq!(
                check_image_ids(&ids, &(1..4), IMG),
                Err(Refusal::NotImage {
                    at: 1..4,
                    index: bad,
                    id: 7,
                    image: IMG
                }),
                "id {bad}"
            );
        }
        let e = check_image_ids(&[IMG, 8], &(0..2), IMG).expect_err("a marker inside the span");
        assert!(
            e.to_string()
                .contains("holds id 8 at index 1, not the image token 900"),
            "{e}"
        );
    }

    /// A call from position 5 over `[t, t, IMG, IMG, t]`, the span at 2..4.
    fn staged(first: u32, ids: &[u32], at: &Range<usize>, media: &[u16]) -> Vec<Vec<f32>> {
        let call = CallRows::new(first, N, [(at, media)]).expect("rows");
        ids.iter()
            .enumerate()
            .map(|(i, &token)| {
                let mut out = vec![0.0; STREAMS * N];
                let p = first + u32::try_from(i).expect("small");
                stage_row(Some(&call), p, token, &mut out, text).expect("stage");
                out
            })
            .collect()
    }

    #[test]
    fn image_rows_are_widened_into_every_stream_and_text_continues_after_them() {
        let (at, media) = (2..4, rows());
        let ids = [10, 11, IMG, IMG, 12];
        let out = staged(5, &ids, &at, &media);
        let want = |v: [f32; N]| -> Vec<f32> { (0..STREAMS).flat_map(|_| v).collect() };
        assert_eq!(out[2], want([1.5, 2.5, 3.5]));
        assert_eq!(out[3], want([4.5, 5.5, 6.5]));
        // Text rows are the text fill, before and after the span, at their own positions.
        for i in [0, 1, 4] {
            let mut t = vec![0.0; STREAMS * N];
            text(ids[i], &mut t).expect("text");
            assert_eq!(out[i], t, "text row {i}");
        }
        // Bit for bit: the widened bits are the bf16 bits over 16 zero bits.
        assert!(
            out[2][..N]
                .iter()
                .zip(&media[..N])
                .all(|(v, &b)| v.to_bits() == u32::from(b) << 16)
        );
    }

    #[test]
    fn a_batch_that_starts_inside_a_span_reads_the_rows_of_its_positions() {
        // The same call as above fed from index 3 (the span's second row): position 8 is row 1.
        let (at, media) = (2..4, rows());
        let call = CallRows::new(5, N, [(&at, media.as_slice())]).expect("rows");
        let mut out = vec![0.0; STREAMS * N];
        stage_row(Some(&call), 8, IMG, &mut out, text).expect("stage");
        assert_eq!(&out[..N], &[4.5, 5.5, 6.5]);
        // Off the span, whichever side: the text fill.
        for p in [6, 9] {
            stage_row(Some(&call), p, 3, &mut out, text).expect("stage");
            assert_eq!(out[0], 3.0, "position {p}");
        }
    }

    #[test]
    fn spans_that_do_not_fit_are_refused_by_name() {
        let media = rows();
        let short = CallRows::new(0, N, [(&(0..3), media.as_slice())]).err();
        assert_eq!(
            short,
            Some(Refusal::Rows {
                at: 0..3,
                values: 2 * N,
                want: 3 * N
            })
        );
        assert_eq!(
            check_image_ids(&[IMG, IMG], &(1..3), IMG),
            Err(Refusal::Outside { at: 1..3, n_ids: 2 })
        );
    }

    #[test]
    fn a_call_with_no_media_stages_the_text_fill() {
        let mut out = vec![0.0; STREAMS * N];
        stage_row(None, 5, 4, &mut out, text).expect("stage");
        assert_eq!(out[0], 4.0);
        assert_eq!(out[N], 4.0);
    }
}
