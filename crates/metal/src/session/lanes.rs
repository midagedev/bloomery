//! The lanes of the delta state: which lane holds the state at a position,
//! and what an M-row verify and its commit do with them.
//!
//! The recurrent state of each delta layer has [`LANES`] lanes, addressed by
//! position: a token at `pos` reads lane [`lane_of`]`(pos)` and writes
//! `lane_of(pos + 1)`. A verify of `m` rows from `pos0` leaves the state after
//! row `j` in lane `lane_of(pos0 + j + 1)` (`Kernels::delta_rows_ring`), so
//! keeping the first `a` rows is standing at position `pos0 + a`: the next
//! token reads the lane that already holds the state after row `a - 1`. A
//! commit moves a position and launches nothing, and a rejected row's lane is
//! a lane no position reads.
//!
//! The conv ring needs nothing of the kind: its slots are addressed by
//! position too, and a verify of at most [`PASS_ROWS`] rows leaves the slots
//! of the positions before it untouched, the ones the next token reads.
//!
//! Every lane beyond the first is one more copy of every delta layer's state
//! in the arena, so [`VERIFY_ROWS`] is the one constant that sets the room a
//! verify takes: [`LANES`] is its value.

use core::fmt;

use crate::delta_geom::{PASS_ROWS, RingRefusal, ring_fits};
use crate::geometry::MROWS;

/// The most rows a verify takes: the rows the M-row mat-vecs run. A verify of
/// `m` rows ends `m` distinct lanes, so the delta state holds this many lanes.
pub const VERIFY_ROWS: usize = MROWS;

/// Lanes of the delta state: [`VERIFY_ROWS`], and at least the two a token
/// that ran and was not taken needs to leave the lane its replacement reads
/// as it was.
pub const LANES: usize = VERIFY_ROWS;

const _: () = assert!(
    VERIFY_ROWS >= 2 && VERIFY_ROWS <= PASS_ROWS,
    "a verify takes 2 to PASS_ROWS rows: the lane ring needs a lane each and the conv ring rolls back PASS_ROWS positions",
);

/// The lane words: word `i` names lane `i`.
pub const LANE_WORDS: [u32; LANES] = {
    let mut words = [0u32; LANES];
    let mut i = 0;
    while i < LANES {
        words[i] = i as u32;
        i += 1;
    }
    words
};

/// The delta lane the token at `pos` reads; it writes `lane_of(pos + 1)`.
#[must_use]
pub const fn lane_of(pos: u32) -> usize {
    pos as usize % LANES
}

/// A verify the lanes cannot serve, or a commit that names no row of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneError {
    /// The rows do not fit the lanes or the conv ring's roll-back.
    Ring(RingRefusal),
    /// The positions of the verify pass the 32-bit position word.
    Position { pos0: u32, rows: usize },
    /// A commit keeps 1 to `rows` rows: row 0 is the token the verify started
    /// from, and a commit of none would leave the position where it was.
    Accepted { accepted: usize, rows: usize },
}

impl fmt::Display for LaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Ring(why) => write!(f, "a verify: {why}"),
            Self::Position { pos0, rows } => write!(
                f,
                "a verify of {rows} rows from {pos0} passes the 32-bit position word"
            ),
            Self::Accepted { accepted, rows } => write!(
                f,
                "a commit keeping {accepted} of a verify's {rows} rows; it keeps 1 to {rows}"
            ),
        }
    }
}

impl std::error::Error for LaneError {}

/// An `m`-row verify from a position, over a delta state of `lanes` lanes:
/// the lanes its launch reads and writes, and its commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Verify {
    lanes: usize,
    pos0: u32,
    rows: usize,
}

impl Verify {
    /// A verify of `rows` rows from `pos0` over `lanes` lanes. Refused by
    /// name when the rows do not fit the lanes or the conv ring's roll-back
    /// ([`ring_fits`]), or the positions pass the 32-bit word.
    ///
    /// # Errors
    ///
    /// [`LaneError`] naming what the lanes cannot serve.
    pub fn new(lanes: usize, pos0: u32, rows: usize) -> Result<Self, LaneError> {
        ring_fits(rows, lanes).map_err(LaneError::Ring)?;
        let past = u32::try_from(rows)
            .ok()
            .and_then(|r| pos0.checked_add(r))
            .is_none();
        if past {
            return Err(LaneError::Position { pos0, rows });
        }
        Ok(Self { lanes, pos0, rows })
    }

    /// A verify over the arena's [`LANES`].
    ///
    /// # Errors
    ///
    /// As [`Verify::new`].
    pub fn at(pos0: u32, rows: usize) -> Result<Self, LaneError> {
        Self::new(LANES, pos0, rows)
    }

    /// The position of row 0.
    #[must_use]
    pub const fn pos0(&self) -> u32 {
        self.pos0
    }

    /// The rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// The lane of the position `pos`.
    const fn lane(&self, pos: u64) -> usize {
        (pos % self.lanes as u64) as usize
    }

    /// The lane the launch reads: the lane of `pos0`. The launch's `lane_at`
    /// word names it.
    #[must_use]
    pub const fn lane_at(&self) -> usize {
        self.lane(self.pos0 as u64)
    }

    /// The lane of the last row's end state: the lane of `pos0 + rows`. The
    /// launch's `lane_to` word names it.
    #[must_use]
    pub const fn lane_to(&self) -> usize {
        self.lane(self.pos0 as u64 + self.rows as u64)
    }

    /// The lane row `row`'s end state lands in: the lane of the position
    /// after the row, `pos0 + row + 1`. None for a row the verify has not.
    #[must_use]
    pub const fn lane_after(&self, row: usize) -> Option<usize> {
        if row >= self.rows {
            return None;
        }
        Some(self.lane(self.pos0 as u64 + row as u64 + 1))
    }

    /// Keep the first `accepted` rows: the session stands at `pos0 +
    /// accepted`, and the token there reads the lane that holds the state
    /// after row `accepted - 1`. Moves a word and launches nothing.
    ///
    /// # Errors
    ///
    /// [`LaneError::Accepted`] when `accepted` is 0 or past the rows.
    pub const fn commit(&self, accepted: usize) -> Result<Committed, LaneError> {
        if accepted == 0 || accepted > self.rows {
            return Err(LaneError::Accepted {
                accepted,
                rows: self.rows,
            });
        }
        // `new` checked `pos0 + rows` fits, and `accepted` is at most `rows`.
        let pos = self.pos0 + accepted as u32;
        Ok(Committed {
            pos,
            lane: self.lane(pos as u64),
            dropped: self.rows - accepted,
        })
    }
}

/// Where a commit leaves the session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Committed {
    /// The position the session stands at: the next token's.
    pub pos: u32,
    /// The lane the next token reads (its `lane_at` word); it writes the lane
    /// of `pos + 1`.
    pub lane: usize,
    /// The rows the commit dropped. Their lanes and conv ring slots hold what
    /// the verify wrote and no position reads them.
    pub dropped: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every lane count a verify can run at, with every start parity.
    fn cases() -> impl Iterator<Item = (usize, u32, usize)> {
        (2..=PASS_ROWS).flat_map(|lanes| {
            (0..2 * lanes as u32 + 3)
                .flat_map(move |pos0| (1..=lanes).map(move |rows| (lanes, pos0, rows)))
        })
    }

    #[test]
    fn the_arena_holds_the_lanes_a_verify_needs() {
        assert_eq!(LANES, VERIFY_ROWS);
        assert_eq!(LANE_WORDS.len(), LANES);
        for (i, w) in LANE_WORDS.iter().enumerate() {
            assert_eq!(*w as usize, i, "word {i} names lane {i}");
        }
        for pos in 0..3 * LANES as u32 {
            assert_eq!(lane_of(pos), pos as usize % LANES);
            assert_ne!(
                lane_of(pos),
                lane_of(pos + 1),
                "a token reads and writes two lanes"
            );
        }
    }

    #[test]
    fn row_j_ends_in_the_lane_of_the_position_after_it() {
        for (lanes, pos0, rows) in cases() {
            let v = Verify::new(lanes, pos0, rows).unwrap();
            assert_eq!(
                v.lane_at(),
                pos0 as usize % lanes,
                "{lanes} lanes, {rows} rows from {pos0}"
            );
            for j in 0..rows {
                assert_eq!(
                    v.lane_after(j),
                    Some((pos0 as usize + j + 1) % lanes),
                    "{lanes} lanes, {rows} rows from {pos0}, row {j}"
                );
            }
            assert_eq!(v.lane_after(rows), None);
            assert_eq!(
                v.lane_after(rows - 1),
                Some(v.lane_to()),
                "the last row's lane is lane_to"
            );
        }
    }

    #[test]
    fn the_rows_end_in_distinct_lanes_and_only_the_last_may_end_in_the_lane_it_read() {
        for (lanes, pos0, rows) in cases() {
            let v = Verify::new(lanes, pos0, rows).unwrap();
            let mut seen = vec![false; lanes];
            for j in 0..rows {
                let lane = v.lane_after(j).unwrap();
                assert!(
                    !seen[lane],
                    "{lanes} lanes, {rows} rows from {pos0}: row {j} repeats lane {lane}"
                );
                seen[lane] = true;
                if lane == v.lane_at() {
                    assert_eq!(
                        j,
                        rows - 1,
                        "{lanes} lanes, {rows} rows from {pos0}: a row before the last wrote the lane read"
                    );
                    assert_eq!(
                        rows, lanes,
                        "only a verify of as many rows as lanes reaches the lane it read"
                    );
                }
            }
        }
    }

    #[test]
    fn a_commit_stands_at_the_position_whose_lane_holds_the_kept_row() {
        for (lanes, pos0, rows) in cases() {
            let v = Verify::new(lanes, pos0, rows).unwrap();
            for accepted in 1..=rows {
                let c = v.commit(accepted).unwrap();
                let at = format!("{lanes} lanes, {rows} rows from {pos0}, {accepted} kept");
                assert_eq!(
                    c.pos,
                    pos0 + accepted as u32,
                    "{at}: the position moves by the rows kept"
                );
                assert_eq!(c.dropped, rows - accepted, "{at}");
                assert_eq!(
                    c.lane,
                    lane_in(lanes, c.pos),
                    "{at}: the next token reads the lane of its position"
                );
                assert_eq!(
                    Some(c.lane),
                    v.lane_after(accepted - 1),
                    "{at}: that is the lane of row {}",
                    accepted - 1
                );
                assert_ne!(
                    c.lane,
                    lane_in(lanes, c.pos + 1),
                    "{at}: it writes another lane"
                );
            }
        }
    }

    fn lane_in(lanes: usize, pos: u32) -> usize {
        pos as usize % lanes
    }

    #[test]
    fn a_commit_keeping_none_or_more_than_the_rows_is_refused_by_name() {
        let v = Verify::at(5, 2).unwrap();
        assert_eq!(
            v.commit(0),
            Err(LaneError::Accepted {
                accepted: 0,
                rows: 2
            })
        );
        assert_eq!(
            v.commit(3),
            Err(LaneError::Accepted {
                accepted: 3,
                rows: 2
            })
        );
        let named = v.commit(3).unwrap_err().to_string();
        assert!(named.contains("keeping 3 of a verify's 2 rows"), "{named}");
    }

    #[test]
    fn a_verify_the_lanes_cannot_serve_is_refused_by_name() {
        let ring = |why| Err(LaneError::Ring(why));
        assert_eq!(Verify::new(2, 0, 0), ring(RingRefusal::NoRows));
        assert_eq!(
            Verify::new(2, 0, 3),
            ring(RingRefusal::Lanes { rows: 3, lanes: 2 })
        );
        assert_eq!(
            Verify::new(4, 0, 5),
            ring(RingRefusal::Lanes { rows: 5, lanes: 4 })
        );
        assert_eq!(
            Verify::new(PASS_ROWS + 1, 0, PASS_ROWS + 1),
            ring(RingRefusal::RollBack {
                rows: PASS_ROWS + 1
            })
        );
        assert_eq!(
            Verify::new(2, u32::MAX, 2),
            Err(LaneError::Position {
                pos0: u32::MAX,
                rows: 2
            })
        );
        assert!(
            Verify::new(2, u32::MAX - 2, 2).is_ok(),
            "the last position fits"
        );
        let rows = Verify::new(2, 0, 3).unwrap_err().to_string();
        assert!(
            rows.contains("needs 3 lanes") && rows.contains("holds 2"),
            "{rows}"
        );
        let arena = Verify::at(0, VERIFY_ROWS + 1).unwrap_err().to_string();
        assert!(arena.contains(&format!("holds {LANES}")), "{arena}");
    }

    #[test]
    fn the_arena_serves_its_own_verify_rows() {
        for pos0 in 0..2 * LANES as u32 {
            for rows in 1..=VERIFY_ROWS {
                assert!(Verify::at(pos0, rows).is_ok(), "{rows} rows from {pos0}");
            }
        }
    }
}
