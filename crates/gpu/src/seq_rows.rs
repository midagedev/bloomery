//! One launch over the rows of several sequences: a pass of resident slots
//! runs every busy slot's rows as one unit, each slot's rows over its own
//! stores. A launch bound to the sequences' stores finds them in one
//! per-sequence table ([`SeqRows`]), its grid-constant parameter: sequence
//! `s` holds rows `start[s] .. start[s + 1]` of the launch, and its words are
//! the device addresses of its own buffers, in the order the entry names
//! them. Every `_seqs` entry runs its one-sequence entry's body over each
//! sequence's rows and buffers alone, so a row's bits are those of the launch
//! of its sequence alone. A captured launch holds the table as it holds every
//! buffer address it was given, so a replay reads the buffers the capture
//! read. Each entry's launcher builds the table from per-sequence arguments
//! that borrow the buffers for the enqueue, so no two sequences name one
//! buffer they write. (`flash_gqa::RowPlanes` is the older per-row form, the
//! head-128 f16 cache's.)

use crate::GpuError;
use std::ops::Range;

/// Sequences one launch takes: a pass of resident slots holds at most this
/// many rows, each slot one row or more.
pub(crate) const SEQS: usize = crate::model::MAX_PASS_ROWS;
/// Words a sequence holds: the most buffers an entry binds to one sequence.
pub(crate) const SEQ_WORDS: usize = 5;

/// The per-sequence table of a launch over several sequences' rows, its
/// grid-constant parameter: `n` sequences, sequence `s`'s rows `start[s] ..
/// start[s + 1]` (ascending from 0, none empty) and its words. Built on the
/// host by [`SeqRows::new`] alone.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct SeqRows {
    n: u32,
    start: [u32; SEQS + 1],
    words: [[u64; SEQ_WORDS]; SEQS],
}

impl SeqRows {
    /// The launch's sequences.
    #[inline(always)]
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.n as usize
    }

    /// Sequence `s`'s first row of the launch and its row count.
    ///
    /// # Safety
    /// `s < self.len()`.
    #[inline(always)]
    #[must_use]
    pub(crate) unsafe fn span(&self, s: usize) -> (usize, usize) {
        // SAFETY: s < n <= SEQS, so s + 1 <= SEQS indexes the starts.
        let (lo, hi) = unsafe {
            (
                *self.start.get_unchecked(s),
                *self.start.get_unchecked(s + 1),
            )
        };
        (lo as usize, (hi - lo) as usize)
    }

    /// The sequence that holds row `t`: the last one whose first row is at
    /// or below `t`, so every row of the launch finds its own.
    #[inline(always)]
    #[must_use]
    pub(crate) fn seq_of(&self, t: usize) -> usize {
        let mut s = 0;
        // SAFETY: s + 1 < n <= SEQS indexes the starts.
        while s + 1 < self.n as usize && unsafe { *self.start.get_unchecked(s + 1) } as usize <= t {
            s += 1;
        }
        s
    }

    /// Word `k` of sequence `s`.
    ///
    /// # Safety
    /// `s < self.len()` and `k < SEQ_WORDS`.
    #[inline(always)]
    #[must_use]
    pub(crate) unsafe fn word(&self, s: usize, k: usize) -> u64 {
        // SAFETY: s < n <= SEQS and k < SEQ_WORDS (the caller's).
        unsafe { *self.words.get_unchecked(s).get_unchecked(k) }
    }

    /// The table of `seqs`, each a sequence's rows and words in the order its
    /// entry names them. Refused by name as `what`'s: no sequence or more
    /// than [`SEQS`], an empty range, or ranges that do not tile the launch's
    /// rows from row 0 in order.
    pub(crate) fn new(
        what: &'static str,
        seqs: &[(Range<usize>, [u64; SEQ_WORDS])],
    ) -> Result<SeqRows, GpuError> {
        if seqs.is_empty() || seqs.len() > SEQS {
            return Err(GpuError::shape(
                what,
                format!("{} sequences; a launch takes 1..={SEQS}", seqs.len()),
            ));
        }
        let mut table = SeqRows {
            n: 0,
            start: [0; SEQS + 1],
            words: [[0; SEQ_WORDS]; SEQS],
        };
        let mut at = 0usize;
        for (s, (rows, words)) in seqs.iter().enumerate() {
            if rows.start != at || rows.is_empty() {
                return Err(GpuError::shape(
                    what,
                    format!(
                        "sequence {s}'s rows {rows:?} after rows 0..{at}: each sequence's rows \
                         follow the last's and hold a row"
                    ),
                ));
            }
            at = rows.end;
            table.start[s + 1] = crate::launch_u32(what, "rows", at)?;
            table.words[s] = *words;
        }
        table.n = crate::launch_u32(what, "sequences", seqs.len())?;
        Ok(table)
    }

    /// The launch's rows: every sequence's.
    #[must_use]
    pub(crate) fn rows(&self) -> usize {
        self.start[self.len()] as usize
    }

    /// The rows of the longest sequence.
    #[must_use]
    pub(crate) fn most(&self) -> usize {
        self.start
            .windows(2)
            .take(self.len())
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::{SEQ_WORDS, SEQS, SeqRows};

    /// The table tiles the launch's rows in order and finds each row's
    /// sequence; any other tiling is refused by name. Mutant: a row of a
    /// sequence's last row read as the next sequence's, or a gap accepted.
    #[test]
    fn a_table_tiles_the_rows_and_finds_each_rows_sequence() {
        let w = |k: u64| [k; SEQ_WORDS];
        let t =
            SeqRows::new("test", &[(0..2, w(1)), (2..3, w(2)), (3..7, w(3))]).expect("a tiling");
        assert_eq!((t.len(), t.rows(), t.most()), (3, 7, 4));
        let seqs: Vec<usize> = (0..7).map(|r| t.seq_of(r)).collect();
        assert_eq!(seqs, [0, 0, 1, 2, 2, 2, 2]);
        // SAFETY: 2 < len and 4 < SEQ_WORDS.
        let (span, word) = unsafe { (t.span(2), t.word(2, SEQ_WORDS - 1)) };
        assert_eq!((span, word), ((3, 4), 3));
        let refused: [&[(std::ops::Range<usize>, [u64; SEQ_WORDS])]; 4] = [
            &[],
            &[(1..2, w(0))],
            &[(0..2, w(0)), (3..4, w(0))],
            &[(0..1, w(0)), (1..1, w(0))],
        ];
        for seqs in refused {
            assert!(SeqRows::new("test", seqs).is_err(), "{seqs:?}");
        }
        let many: Vec<_> = (0..=SEQS).map(|s| (s..s + 1, w(0))).collect();
        assert!(SeqRows::new("test", &many).is_err());
    }
}
