//! The draft's context ring: the slots its layers' KV rows stand in, and
//! which positions they hold.
//!
//! The context row of target position `p` sits in ring slot `p % window`
//! ([`Positions::slot`]) and turns at rope position `p`; a block after the
//! last committed position sees the ring's first `min(committed, window)`
//! rows. A draft with no window keeps every position and its slot is the
//! position. [`Positions`] holds the positions' rules, one owner for every
//! family's layers: the rows appended in order from the committed position,
//! the positions a reader skipped without their rows, and the refusal of a
//! block that would read a slot no row of its own has filled.

use std::ops::Range;

use crate::GpuError;

const WHAT: &str = "dspark::ring";

/// Which positions the draft's context holds.
#[derive(Debug)]
pub struct Positions {
    /// How far back a block reads its context: `None` is the whole context.
    window: Option<usize>,
    /// Positions appended or skipped so far: the next committed position.
    committed: u32,
    /// The latest positions [`Positions::skip`] passed without a row: a block
    /// that would read one of their slots is refused.
    skipped: Option<Range<u32>>,
}

impl Positions {
    /// No position committed. A window of 0 positions is refused by name.
    pub fn new(window: Option<usize>) -> Result<Positions, GpuError> {
        if window == Some(0) {
            return Err(GpuError::shape(WHAT, "a context window of 0 positions"));
        }
        Ok(Positions {
            window,
            committed: 0,
            skipped: None,
        })
    }

    /// Positions committed so far.
    #[must_use]
    pub fn committed(&self) -> u32 {
        self.committed
    }

    /// The window the ring keeps, `None` for the whole context.
    #[must_use]
    pub fn window(&self) -> Option<usize> {
        self.window
    }

    /// The ring slot of position `pos`: `pos % window`, or `pos` itself with
    /// no window.
    #[must_use]
    pub fn slot(&self, pos: u32) -> u32 {
        match self.window.and_then(|w| u32::try_from(w).ok()) {
            Some(w) => pos % w,
            None => pos,
        }
    }

    /// Context rows a block after the committed position reads:
    /// `min(committed, window)`.
    #[must_use]
    pub fn ring_rows(&self) -> usize {
        let committed = self.committed as usize;
        self.window.map_or(committed, |w| committed.min(w))
    }

    /// The first position whose row a block after the committed position
    /// reads.
    fn seen_from(&self) -> usize {
        self.committed as usize - self.ring_rows()
    }

    /// The committed position after `n` rows appended at `first`, with the
    /// append checked and nothing changed: `first` is the committed position
    /// (a row appended elsewhere leaves a gap or overwrites one), `n` at
    /// least 1, and the next position fits a `u32`.
    pub fn check_append(&self, first: u32, n: usize) -> Result<u32, GpuError> {
        if first != self.committed {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "an append at position {first} after {} committed positions",
                    self.committed
                ),
            ));
        }
        let next = u32::try_from(n)
            .ok()
            .filter(|&n| n > 0)
            .and_then(|n| first.checked_add(n));
        next.ok_or_else(|| {
            GpuError::shape(
                WHAT,
                format!("an append of {n} rows at position {first}: 1.. rows that fit a u32"),
            )
        })
    }

    /// Commit `n` rows appended at `first` ([`Positions::check_append`]).
    pub fn append(&mut self, first: u32, n: usize) -> Result<(), GpuError> {
        self.committed = self.check_append(first, n)?;
        Ok(())
    }

    /// Commit positions `committed .. to` without their rows: a reader that
    /// keeps only a window's rows skips the older ones. Their ring slots hold
    /// nothing of theirs, and a block that would read one before appended rows
    /// overwrite it is refused ([`Positions::check_block`]). Refused for a
    /// `to` below the committed position.
    pub fn skip(&mut self, to: u32) -> Result<(), GpuError> {
        if to < self.committed {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a skip to position {to} after {} committed positions",
                    self.committed
                ),
            ));
        }
        if to > self.committed {
            self.skipped = Some(self.committed..to);
            self.committed = to;
        }
        Ok(())
    }

    /// Whether a block at position `pos` may run: after at least one
    /// committed position, at the committed position, and reading no slot of
    /// a skipped position.
    pub fn check_block(&self, pos: u32) -> Result<(), GpuError> {
        if self.committed == 0 {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a committed position before the first block",
            });
        }
        if pos != self.committed {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a block at position {pos}, the ring holds {} committed positions",
                    self.committed
                ),
            ));
        }
        let seen = self.seen_from();
        if let Some(gap) = self.skipped.as_ref().filter(|g| g.end as usize > seen) {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a block after position {} reads the ring rows of positions {seen}.., and \
                     positions {gap:?} were skipped without their rows",
                    self.committed
                ),
            ));
        }
        Ok(())
    }

    /// No committed position and nothing skipped.
    pub fn reset(&mut self) {
        self.committed = 0;
        self.skipped = None;
    }
}

#[cfg(test)]
mod tests {
    use super::Positions;

    /// The ring's rules, with no card: rows append in order at the committed
    /// position; a skip leaves a gap that a block is refused while its slots
    /// are still read, and takes back a step backwards; a block needs a
    /// committed position and runs at it; a context without a window never
    /// forgets a gap.
    #[test]
    fn positions_keep_the_ring_rules() {
        assert!(Positions::new(Some(0)).is_err());

        let mut p = Positions::new(Some(4)).expect("a window of 4");
        assert!(p.check_block(0).is_err(), "no block before a committed row");
        assert_eq!((p.slot(5), p.ring_rows()), (1, 0));

        p.append(0, 3).expect("rows 0..3");
        assert_eq!((p.committed(), p.ring_rows()), (3, 3));
        assert!(p.append(2, 1).is_err(), "an overlap");
        assert!(p.append(4, 1).is_err(), "a gap");
        assert!(p.append(3, 0).is_err(), "no rows");
        assert!(
            p.check_block(2).is_err(),
            "a block elsewhere than committed"
        );
        p.check_block(3).expect("a block at the committed position");

        p.skip(10).expect("a skip forward");
        assert_eq!(p.committed(), 10);
        assert!(p.skip(9).is_err(), "a skip backwards");
        assert!(
            p.check_block(10).is_err(),
            "the window reads positions 6.. and the gap ends at 10"
        );
        p.append(10, 3).expect("three rows after the gap");
        assert!(
            p.check_block(13).is_err(),
            "the window reads from position 9, and 9 was skipped"
        );
        p.append(13, 1).expect("a fourth row");
        assert_eq!(p.ring_rows(), 4);
        p.check_block(14)
            .expect("the window holds only appended rows");

        p.reset();
        assert_eq!(p.committed(), 0);
        p.append(0, 1).expect("rows again from 0");
        p.check_block(1).expect("a block with no gap");

        let mut whole = Positions::new(None).expect("no window");
        whole.skip(1).expect("a skip");
        whole.append(1, 8).expect("rows");
        assert_eq!((whole.slot(7), whole.ring_rows()), (7, 9));
        assert!(
            whole.check_block(9).is_err(),
            "the whole context keeps the gap"
        );
    }
}
