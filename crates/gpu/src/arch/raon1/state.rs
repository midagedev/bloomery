//! What a pass of several rows of Raon-1's body leaves for its commit
//! (`runtime::Verify::commit`), and how the commit takes the rest back.
//!
//! A verify keeps each row's state exactly, where the reference restores a
//! checkpoint and runs the kept ids again (`speculative-simple.cpp`):
//! - an RDA layer's recurrent state has [`PASS_ROWS`] lanes on a load that
//!   verifies, one on a load that does not ([`rda_store`]). A verify's row
//!   `j` writes the state after it into lane `(c + j) mod lanes`
//!   (`linear::delta`'s row mode, `c` the committed lane), so keeping its
//!   first `k` rows moves the lane word to
//!   [`kept_lane`](crate::linear::delta::kept_lane)`(c, k)` and copies
//!   nothing. Row 0 writes lane `c` itself: a verify always keeps its first
//!   row. A step and a prompt chunk run in place on the committed lane;
//! - an RDA layer's conv ring holds the inputs of the last
//!   [`RING_ROWS`](crate::linear::RING_ROWS) positions, each in the slot of
//!   its position, so a call from any kept position reads its predecessors
//!   where the pass left them;
//! - a latent layer's cache is indexed by position and read only below the
//!   live count: a taken-back row is written again by the next call at its
//!   position before any launch reads it. Its length is the body's position.
//!
//! [`Lanes`] is the host's side of the lane word: the committed lane and the
//! verify waiting for its commit. While a verify waits, every other pass is
//! refused by name, since its rows would read a lane the commit has not
//! named.

use crate::GpuError;
use crate::linear::LinearShape;
use crate::linear::delta::kept_lane;
use crate::linear::mixer_kda::KdaState;
use crate::recur::RecStore;
use cuda_core::{CudaStream, DeviceBuffer};

/// What the refusals of this file name.
const WHAT: &str = "raon1 state";

/// The most rows one pass runs: a verify of the current id and up to seven
/// drafted ones, and the most query tokens the latent attention takes.
pub(crate) const PASS_ROWS: usize = crate::linear::PASS_ROWS;

/// One RDA layer's recurrent store at `shape` over `lanes` lanes, zeroed,
/// lane 0 at position 0 and every other lane never written:
/// [`PASS_ROWS`] lanes on a load that verifies, 1 on one that does not.
/// Load-time only.
pub(crate) fn rda_store(
    stream: &CudaStream,
    shape: LinearShape,
    lanes: usize,
) -> Result<RecStore, GpuError> {
    if !(1..=PASS_ROWS).contains(&lanes) {
        return Err(GpuError::shape(
            WHAT,
            format!("an RDA store of {lanes} lanes: 1 to {PASS_ROWS}"),
        ));
    }
    RecStore::stamped(stream, shape.lens(lanes), lanes)
}

/// `store` back to a fresh one in place (a captured chain keeps its
/// addresses): every lane and the ring zero, lane 0 at position 0. Never
/// inside a capture.
pub(crate) fn clear_store(stream: &CudaStream, store: &mut RecStore) -> Result<(), GpuError> {
    store.clear(stream)?;
    store.restamp(stream)
}

/// How a pass's rows meet the RDA state lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pass {
    /// A step or a prompt chunk: the rows in place on the committed lane.
    InPlace,
    /// A verify: each row's state in a lane of its own, the pass waiting for
    /// [`Lanes::commit`].
    Verify,
}

/// A verify of `rows` rows from position `pos0` that waits for its commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Waiting {
    pos0: u32,
    rows: u32,
}

/// The host rule of the lane word, apart from the device word it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Book {
    count: u32,
    committed: u32,
    waiting: Option<Waiting>,
}

impl Book {
    /// A pass of `rows` rows from `pos0`: refused by name while a verify
    /// waits, and a verify of fewer than 2 rows or more than the lanes; a
    /// verify is left waiting. Nothing moves on a refusal.
    fn open(&mut self, pass: Pass, pos0: u32, rows: usize) -> Result<(), GpuError> {
        if let Some(w) = self.waiting {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a pass of {rows} rows at position {pos0} while the verify of {} rows at {} \
                     waits for its commit",
                    w.rows, w.pos0
                ),
            ));
        }
        match pass {
            Pass::InPlace if rows >= 1 => Ok(()),
            Pass::InPlace => Err(GpuError::shape(
                WHAT,
                format!("a pass of no rows at position {pos0}"),
            )),
            Pass::Verify => {
                let rows = u32::try_from(rows)
                    .ok()
                    .filter(|r| (2..=self.count).contains(r))
                    .ok_or_else(|| {
                        GpuError::shape(
                            WHAT,
                            format!(
                                "a verify of {rows} rows over {} lanes: 2 to the lanes (a load \
                                 that verifies holds {PASS_ROWS})",
                                self.count
                            ),
                        )
                    })?;
                self.waiting = Some(Waiting { pos0, rows });
                Ok(())
            }
        }
    }

    /// The lane and the position after the waiting verify keeps its first
    /// `keep` rows; refused by name with no verify waiting or a `keep`
    /// outside `1..=rows`. Moves nothing.
    fn kept(&self, keep: usize) -> Result<(u32, u32), GpuError> {
        let w = self.waiting.ok_or_else(|| {
            GpuError::shape(
                WHAT,
                format!("a commit of {keep} rows with no verify waiting"),
            )
        })?;
        let k = u32::try_from(keep)
            .ok()
            .filter(|k| (1..=w.rows).contains(k))
            .ok_or_else(|| {
                GpuError::shape(
                    WHAT,
                    format!(
                        "a commit of {keep} rows of the verify of {} rows at {}: 1 to {}",
                        w.rows, w.pos0, w.rows
                    ),
                )
            })?;
        Ok((kept_lane(self.committed, k, self.count), w.pos0 + k))
    }
}

/// The lane word every RDA layer's delta step reads, and the host's record
/// of it (module doc). One word serves every layer: all the stores of a load
/// hold the same lanes and move together.
pub(crate) struct Lanes {
    word: DeviceBuffer<u32>,
    book: Book,
}

impl Lanes {
    /// Lane 0 of `count` lanes (1 to [`PASS_ROWS`]), nothing waiting.
    /// Load-time only.
    pub(crate) fn new(stream: &CudaStream, count: usize) -> Result<Lanes, GpuError> {
        let n = u32::try_from(count)
            .ok()
            .filter(|&n| (1..=PASS_ROWS as u32).contains(&n))
            .ok_or_else(|| GpuError::shape(WHAT, format!("{count} lanes: 1 to {PASS_ROWS}")))?;
        Ok(Lanes {
            word: DeviceBuffer::from_host(stream, &[0u32])?,
            book: Book {
                count: n,
                committed: 0,
                waiting: None,
            },
        })
    }

    /// The device word: the committed lane.
    pub(crate) fn word(&self) -> &DeviceBuffer<u32> {
        &self.word
    }

    /// The lanes every store holds.
    pub(crate) fn count(&self) -> usize {
        self.book.count as usize
    }

    /// The lane the word holds: the one lane a position's state is in.
    pub(crate) fn committed(&self) -> u32 {
        self.book.committed
    }

    /// The verify waiting for its commit: its first position and its rows.
    pub(crate) fn waiting(&self) -> Option<(u32, usize)> {
        self.book.waiting.map(|w| (w.pos0, w.rows as usize))
    }

    /// Open a pass of `rows` rows from `pos0` before its first launch:
    /// refused by name while a verify waits, and a verify of fewer than 2
    /// rows or more than the lanes; a verify then waits for its commit.
    pub(crate) fn open(&mut self, pass: Pass, pos0: u32, rows: usize) -> Result<(), GpuError> {
        self.book.open(pass, pos0, rows)
    }

    /// The waiting verify dropped with the word where it was: only for a
    /// verify refused before any launch ran, since its row 0 writes the
    /// committed lane.
    pub(crate) fn abandon(&mut self) {
        self.book.waiting = None;
    }

    /// The store and lane an RDA layer's launches of the open pass read:
    /// row mode while a verify waits, else in place. A graph captured before
    /// its pass opens sets `each` from the pass it records instead. A store
    /// of other lanes than the word's, or with no stamps, is refused by name.
    pub(crate) fn state<'a>(&'a self, store: &'a mut RecStore) -> Result<KdaState<'a>, GpuError> {
        if store.lanes != self.count() {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a store of {} lanes under a lane word over {}",
                    store.lanes,
                    self.count()
                ),
            ));
        }
        let RecStore {
            state,
            ring,
            lanes,
            stamp,
            ..
        } = store;
        let stamp = stamp
            .as_mut()
            .ok_or(GpuError::state(WHAT, "the store's stamps"))?;
        Ok(KdaState {
            state,
            stamp,
            ring,
            lane: &self.word,
            lanes: *lanes,
            each: self.book.waiting.is_some(),
            row: 0,
        })
    }

    /// Keep the first `keep` rows of the verify waiting, 1 to its rows: the
    /// word to the lane row `keep − 1` wrote, its copy enqueued when it
    /// moves, nothing waiting after. Returns the position the model stands
    /// at, the verify's first plus `keep`. Refused by name, nothing moved,
    /// with no verify waiting or a `keep` outside its rows. Never inside a
    /// capture.
    pub(crate) fn commit(&mut self, stream: &CudaStream, keep: usize) -> Result<u32, GpuError> {
        let (lane, pos) = self.book.kept(keep)?;
        if lane != self.book.committed {
            self.word.copy_from_host(stream, &[lane])?;
        }
        self.book.committed = lane;
        self.book.waiting = None;
        Ok(pos)
    }

    /// Lane 0, nothing waiting, the word's copy enqueued: beside
    /// [`clear_store`] of every store. Never inside a capture.
    pub(crate) fn reset(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.word.copy_from_host(stream, &[0u32])?;
        self.book.committed = 0;
        self.book.waiting = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Book, GpuError, PASS_ROWS, Pass, RecStore, Waiting, clear_store};

    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_clear_store_resets_values_and_stamps() -> Result<(), GpuError> {
        let (_ctx, stream) = crate::capsync::fresh_stream(0)?;
        let mut store = RecStore::stamped(&stream, [68, 13], 4)?;
        store.restamp_at(&stream, 3, 29)?;
        for buf in [&mut store.state, &mut store.ring] {
            buf.copy_from_host(&stream, &vec![f32::NAN; buf.len()])?;
        }
        clear_store(&stream, &mut store)?;
        for buf in [&store.state, &store.ring] {
            assert!(buf.to_host_vec(&stream)?.iter().all(|v| v.to_bits() == 0));
        }
        assert_eq!(
            store
                .stamp
                .as_ref()
                .expect("stamped store")
                .to_host_vec(&stream)?,
            crate::recur::fresh_stamps(4)
        );
        Ok(())
    }

    fn book(committed: u32) -> Book {
        Book {
            count: PASS_ROWS as u32,
            committed,
            waiting: None,
        }
    }

    /// A verify keeps 1 to its rows: the lane the last kept row wrote
    /// (`(c + k − 1) mod lanes`) and the position after it; any other count,
    /// or a commit with nothing waiting, is refused by name and moves
    /// nothing.
    #[test]
    fn a_commit_lands_on_the_lane_the_last_kept_row_wrote() {
        let mut b = book(3);
        b.open(Pass::Verify, 10, 8).unwrap();
        let waiting = b;
        for (keep, lane) in [(1, 3), (2, 4), (5, 7), (6, 0), (8, 2)] {
            assert_eq!(
                b.kept(keep).unwrap(),
                (lane, 10 + keep as u32),
                "keep {keep}"
            );
        }
        for keep in [0, 9] {
            let e = b.kept(keep).unwrap_err().to_string();
            assert!(e.contains(&format!("a commit of {keep} rows")), "{e}");
        }
        assert_eq!(b, waiting, "a refused commit moves nothing");
        let e = book(0).kept(1).unwrap_err().to_string();
        assert!(e.contains("no verify waiting"), "{e}");
    }

    /// While a verify waits, a step, a prompt chunk and another verify are
    /// refused by name; a verify of fewer than 2 rows or more than the lanes
    /// is refused and leaves nothing waiting.
    #[test]
    fn a_waiting_verify_refuses_every_other_pass() {
        let mut b = book(0);
        for rows in [0, 1, PASS_ROWS + 1] {
            let e = b.open(Pass::Verify, 4, rows).unwrap_err().to_string();
            assert!(e.contains(&format!("a verify of {rows} rows")), "{e}");
            assert_eq!(b.waiting, None);
        }
        assert!(b.open(Pass::InPlace, 4, 0).is_err());
        b.open(Pass::InPlace, 4, 3).unwrap();
        assert_eq!(b.waiting, None, "a prompt chunk waits for nothing");
        b.open(Pass::Verify, 7, 2).unwrap();
        assert_eq!(b.waiting, Some(Waiting { pos0: 7, rows: 2 }));
        for pass in [Pass::InPlace, Pass::Verify] {
            let e = b.open(pass, 9, 2).unwrap_err().to_string();
            assert!(e.contains("waits for its commit"), "{e}");
        }
    }
}
