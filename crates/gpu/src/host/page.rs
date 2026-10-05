//! The host-mapped page's layout, fixed at load ([`PageLayout`]): the flag
//! words, then per row its handoff image and its host sum. Arithmetic only —
//! no allocation and no driver call — so the layout is checked where it is
//! made and every offset the step port and a handoff launch use is read from
//! one value.
//!
//! A page carries `rows` rows (1..=[`MAX_ROWS`]), each a unit of `cols`
//! columns at one layer. The flag words take the page's first
//! [`PAYLOAD_OFF`] bytes, one 64-byte line each: the generation, and per row
//! its counter and its layer word. Row `r`'s image — the sequence word, the
//! routing's ids and weights (`cols · n_used` each) and the activation
//! (`cols · hidden` f32) — follows at a 256-byte boundary, the images one
//! after another, then every row's sum (`cols · hidden` f32), each at a
//! 256-byte boundary.
//!
//! The routed width `n_used` is the model's, any count from one: the page
//! holds what it was built for, and a call that brings more slots than that
//! is refused where it meets the page — the batch service's routing count,
//! the host scratches' list width.

use model::ops::DEFER_MAX_COLS;

/// Rows a page carries at most: tokens whose handoffs can be in flight at
/// once, each with its own layer word, image and sum.
pub const MAX_ROWS: usize = 4;

/// A flag word of the page. Each has a 64-byte line of its own, so a thread
/// spinning on one never shares a line with another.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Word {
    /// The card adds one per go; the host reads it.
    Gen,
    /// The host adds one per layer of row `.0` served; the card waits for it
    /// and takes it back.
    Cnt(usize),
    /// The card writes the layer of each go of row `.0`; the host reads it.
    Lyr(usize),
}

impl Word {
    /// The word's byte offset in the page.
    pub(crate) const fn offset(self) -> usize {
        match self {
            Word::Gen => 0,
            Word::Cnt(row) => 64 + 128 * row,
            Word::Lyr(row) => 128 + 128 * row,
        }
    }
}

/// Byte offset of row 0's handoff image in the page: past every flag word
/// of [`MAX_ROWS`] rows, whatever the page's own rows.
pub(crate) const PAYLOAD_OFF: usize = (128 * MAX_ROWS + 64).next_multiple_of(256);

const _: () = assert!(
    Word::Cnt(MAX_ROWS - 1).offset() + 64 <= PAYLOAD_OFF
        && Word::Lyr(MAX_ROWS - 1).offset() + 64 <= PAYLOAD_OFF
);

/// Words of the smallest routing field: a unit whose `cols · n_used` fits it
/// keeps the ids at word 16, the weights at 32 and the activation at 64.
const FIELD_MIN: usize = 16;

/// The activation's alignment in an image, in words (256 bytes).
const X_ALIGN: usize = 64;

/// One unit's handoff in its image: the word offsets of the sequence, the
/// routing's ids and weights (`n_used` a column each) and the activation
/// (`hidden` f32 a column, the image's tail).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandoffLayout {
    pub seq: usize,
    pub ids: usize,
    pub weights: usize,
    pub x: usize,
    pub n_used: usize,
    pub hidden: usize,
}

/// Why a page layout was refused ([`PageLayout::new`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PageError {
    /// Rows outside `1..=MAX_ROWS`.
    Rows(usize),
    /// Columns outside `1..=DEFER_MAX_COLS`: the step's union call claims
    /// the quantizations of at most that many.
    Cols(usize),
    /// No model width.
    Hidden,
    /// No routed slot a token.
    Used,
    /// A size past `usize`.
    Overflow,
}

impl std::fmt::Display for PageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PageError::Rows(r) => write!(f, "{r} rows: a page carries 1..={MAX_ROWS}"),
            PageError::Cols(c) => write!(
                f,
                "{c} columns a row: a host service serves 1..={DEFER_MAX_COLS} in one union call"
            ),
            PageError::Hidden => write!(f, "a model width of 0"),
            PageError::Used => write!(f, "0 routed slots a token: a page carries at least one"),
            PageError::Overflow => write!(f, "the page's size passes usize"),
        }
    }
}

/// The page's layout: `rows` rows of `cols` columns of the model width
/// `hidden`, `n_used` routed slots a column. Made once at load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageLayout {
    rows: usize,
    cols: usize,
    handoff: HandoffLayout,
    image_stride: usize,
    hsum_stride: usize,
}

impl PageLayout {
    /// The layout of `rows` rows of `cols` columns of `hidden` values with
    /// `n_used` routed slots a column, or the bound it breaks.
    pub fn new(
        rows: usize,
        cols: usize,
        hidden: usize,
        n_used: usize,
    ) -> Result<PageLayout, PageError> {
        if !(1..=MAX_ROWS).contains(&rows) {
            return Err(PageError::Rows(rows));
        }
        if !(1..=DEFER_MAX_COLS).contains(&cols) {
            return Err(PageError::Cols(cols));
        }
        if hidden == 0 {
            return Err(PageError::Hidden);
        }
        if n_used == 0 {
            return Err(PageError::Used);
        }
        let o = PageError::Overflow;
        let field = cols
            .checked_mul(n_used)
            .and_then(|k| k.checked_next_multiple_of(FIELD_MIN))
            .ok_or(o.clone())?;
        let x = field
            .checked_mul(2)
            .and_then(|f| f.checked_add(FIELD_MIN))
            .and_then(|w| w.checked_next_multiple_of(X_ALIGN))
            .ok_or(o.clone())?;
        let (ids, weights) = (FIELD_MIN, FIELD_MIN + field);
        let x_words = cols.checked_mul(hidden).ok_or(o.clone())?;
        let words = x.checked_add(x_words).ok_or(o.clone())?;
        let image_stride = words
            .checked_mul(4)
            .and_then(|b| b.checked_next_multiple_of(256))
            .ok_or(o.clone())?;
        let hsum_stride = x_words
            .checked_mul(4)
            .and_then(|b| b.checked_next_multiple_of(256))
            .ok_or(o.clone())?;
        let layout = PageLayout {
            rows,
            cols,
            handoff: HandoffLayout {
                seq: 0,
                ids,
                weights,
                x,
                n_used,
                hidden,
            },
            image_stride,
            hsum_stride,
        };
        layout.bytes().ok_or(o)?;
        Ok(layout)
    }

    /// Rows the page carries.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns a row carries.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// One unit's handoff layout in its image.
    #[must_use]
    pub fn handoff(&self) -> HandoffLayout {
        self.handoff
    }

    /// Words of one unit's image and of the handoff region it copies: the
    /// fields, then the activation.
    #[must_use]
    pub fn image_words(&self) -> usize {
        self.handoff.x + self.cols * self.handoff.hidden
    }

    /// Values of one unit's host sum.
    #[must_use]
    pub fn hsum_len(&self) -> usize {
        self.cols * self.handoff.hidden
    }

    /// Byte offset of row `row`'s image; `None` past the rows.
    #[must_use]
    pub fn image_off(&self, row: usize) -> Option<usize> {
        (row < self.rows).then(|| PAYLOAD_OFF + row * self.image_stride)
    }

    /// Byte offset of row `row`'s host sum; `None` past the rows.
    #[must_use]
    pub fn hsum_off(&self, row: usize) -> Option<usize> {
        (row < self.rows)
            .then(|| PAYLOAD_OFF + self.rows * self.image_stride + row * self.hsum_stride)
    }

    /// Bytes the page holds: the flag words, every image and every sum;
    /// `None` past `usize`.
    #[must_use]
    pub fn bytes(&self) -> Option<usize> {
        self.rows
            .checked_mul(self.image_stride + self.hsum_stride)?
            .checked_add(PAYLOAD_OFF)
    }
}

#[cfg(test)]
mod tests {
    use super::{FIELD_MIN, HandoffLayout, MAX_ROWS, PAYLOAD_OFF, PageError, PageLayout, Word};

    /// V4.1's page — two rows of one column of 4096 values, six slots — is
    /// the fixed layout the handoff kernel and the step port read: the
    /// sequence at word 0, the ids at 16, the weights at 32, the activation
    /// at 64; row 0's image at byte 768, row 1's 16,640 after it, the sums
    /// from 34,048 16,384 apart, 66,816 bytes in all.
    // PIN(2026-10-05): MAX_ROWS 2→4 moves PAYLOAD_OFF (128·4+64=576, rounded
    // to 768); the images and sums sit past it, so every offset and the
    // page's bytes moved +256.
    #[test]
    fn v41_page_is_the_fixed_layout() {
        let p = PageLayout::new(2, 1, 4096, 6).expect("V4.1's page");
        assert_eq!(PAYLOAD_OFF, 768);
        assert_eq!(
            p.handoff(),
            HandoffLayout {
                seq: 0,
                ids: 16,
                weights: 32,
                x: 64,
                n_used: 6,
                hidden: 4096
            }
        );
        assert_eq!(p.image_words(), 64 + 4096);
        assert_eq!(
            [p.image_off(0), p.image_off(1)],
            [Some(768), Some(768 + 16_640)]
        );
        assert_eq!(
            [p.hsum_off(0), p.hsum_off(1)],
            [Some(34_048), Some(34_048 + 16_384)]
        );
        assert_eq!(p.bytes(), Some(66_816));
        assert_eq!(p.image_off(2), None);
        assert_eq!(p.hsum_off(2), None);
    }

    /// Every layout of one column of up to `FIELD_MIN` (16) slots keeps the
    /// fixed field offsets, one row or four — the bound is the smallest
    /// routing field, not a routed width; V2-Lite's page (one row of 2048,
    /// six slots), GLM-5.3-Flash's eight and Qwen3.8's ten are among them.
    #[test]
    fn one_column_keeps_the_fixed_fields() {
        for rows in 1..=MAX_ROWS {
            for n_used in 1..=FIELD_MIN {
                let p = PageLayout::new(rows, 1, 2048, n_used).expect("a one-column page");
                let h = p.handoff();
                assert_eq!(
                    (h.seq, h.ids, h.weights, h.x),
                    (0, 16, 32, 64),
                    "{rows} {n_used}"
                );
                // PIN(2026-10-05): row 0's image starts at PAYLOAD_OFF, which
                // MAX_ROWS 4 moved from 512 to 768.
                assert_eq!(p.image_off(0), Some(768));
            }
        }
    }

    /// Wider units grow the routing fields and keep them apart: the ids,
    /// the weights and the activation never overlap, the activation starts
    /// at a 256-byte boundary, and rows, images and sums lie one after
    /// another inside the page.
    #[test]
    fn fields_hold_their_columns_apart() {
        for rows in 1..=MAX_ROWS {
            for cols in 1..=model::ops::DEFER_MAX_COLS {
                for n_used in 1..=2 * FIELD_MIN {
                    let p = PageLayout::new(rows, cols, 256, n_used).expect("a page");
                    let h = p.handoff();
                    let k = cols * n_used;
                    assert!(h.seq < h.ids && h.ids + k <= h.weights && h.weights + k <= h.x);
                    assert_eq!(h.x % 64, 0);
                    let last_image = p.image_off(rows - 1).unwrap() + 4 * p.image_words();
                    assert!(last_image <= p.hsum_off(0).unwrap());
                    let last_sum = p.hsum_off(rows - 1).unwrap() + 4 * p.hsum_len();
                    assert!(last_sum <= p.bytes().unwrap());
                    assert_eq!(p.hsum_off(0).unwrap() % 256, 0);
                }
            }
        }
    }

    /// A layout past a bound is refused by the bound's name. The routed
    /// width has none past `usize`: GLM-5.3-Flash's eight slots and
    /// Qwen3.8's ten fit, and so do sixty-four; none is refused.
    #[test]
    fn bounds_are_refused_by_name() {
        for n_used in [8, 10, 64] {
            let p = PageLayout::new(1, 1, 4096, n_used).expect("a routed width");
            assert_eq!(p.handoff().n_used, n_used);
        }
        assert_eq!(PageLayout::new(1, 1, 2048, 0), Err(PageError::Used));
        assert_eq!(
            PageLayout::new(1, 8, 2048, usize::MAX / 4),
            Err(PageError::Overflow)
        );
        assert_eq!(PageLayout::new(0, 1, 2048, 6), Err(PageError::Rows(0)));
        // PIN(2026-10-05): 3 rows moved inside 1..=MAX_ROWS (4) — the refused
        // probe is the bound's new outside, 5.
        assert_eq!(PageLayout::new(5, 1, 2048, 6), Err(PageError::Rows(5)));
        assert_eq!(PageLayout::new(1, 0, 2048, 6), Err(PageError::Cols(0)));
        assert_eq!(PageLayout::new(1, 9, 2048, 6), Err(PageError::Cols(9)));
        assert_eq!(PageLayout::new(1, 1, 0, 6), Err(PageError::Hidden));
        assert_eq!(
            PageLayout::new(1, 1, usize::MAX / 2, 6),
            Err(PageError::Overflow)
        );
        let e = PageError::Used.to_string();
        assert!(e.contains("0 routed slots"), "{e}");
    }

    /// The flag words sit in the page's first `PAYLOAD_OFF` bytes, each on a
    /// line of its own.
    #[test]
    fn flag_words_have_lines_of_their_own() {
        let mut offs = vec![Word::Gen.offset()];
        for r in 0..MAX_ROWS {
            offs.extend([Word::Cnt(r).offset(), Word::Lyr(r).offset()]);
        }
        offs.sort_unstable();
        assert!(offs.windows(2).all(|w| w[1] - w[0] >= 64));
        assert!(offs.iter().all(|&o| o % 64 == 0 && o + 64 <= PAYLOAD_OFF));
    }
}
