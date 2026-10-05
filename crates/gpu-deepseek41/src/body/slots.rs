//! The body's pass of several resident slots ([`SlotRows`]): the pass's
//! rows on the pair pass's walk, one layer apart on the one stream, each row
//! bound to its slot's sequence, so the host serves one row's layer while
//! the card runs the other's.
//!
//! The walk carries two rows in flight with one column each: a pass holds at
//! most [`PAIR_ROWS`] rows and runs one of two shapes ([`PassShape`]) — two
//! slots of one row each, or one slot of two, which is the verify pair of
//! that slot's sequence. Any other is refused by name: one row is its slot's
//! step. The homes are canonical: slot 0's sequence is the live one, and
//! every other busy slot's is its entry of the owner's parked list, in the
//! pass's order.
//!
//! The host half plans each row as the step plans it, on the row's own
//! sequence: its token after that sequence's history, that sequence's stale
//! ring slots restored from its own shadows by its own holds, its image into
//! the row's lane. The enqueue is the pair pass's walk ([`walk_pair`]) with
//! each row's caches and shadows its sequence's ([`RowSeqs`]): the pair's
//! launches in the pair's order, so the pass shares the verify pair's go
//! order and node count, and each row runs its slot's step.

use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::Chain;
use bloomery_gpu::model::{SlotRange, SlotRows};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::CudaStream;
use runtime::sched;

use super::seq::RowSeqs;
use super::{Body, PAIR_ROWS, Seq, StepInput, walk_pair};

impl SlotRows for Body {
    /// Two rows in flight on the step port, one column each.
    const MAX_ROWS: usize = PAIR_ROWS;
    /// No: a pass's rows are kept whole as they run (the default
    /// [`SlotRows::keep_slot`]), so
    /// [`GpuModel::verify_slots`](bloomery_gpu::GpuModel::verify_slots) is
    /// refused by name and no pass of several slots ever waits for its
    /// commit on this body. On a body that settles a partial keep a pass
    /// waits between its verify and its commit, and [`super::snapshot`],
    /// which reads the selected sequence's state, refuses while one waits.
    const SETTLES_PARTIAL_KEEP: bool = false;

    /// The walk's point for the pass's rows, each row a lane of one column
    /// (`runtime::sched::slot_lanes`): the step for one row, the pair for
    /// two. The owner refuses a pass past [`SlotRows::MAX_ROWS`] rows before
    /// it asks, so no other count reaches here; one that does is a named
    /// panic.
    fn chain_of(rows: &[SlotRange]) -> Chain {
        let n = rows.last().map_or(0, |r| r.rows.end);
        sched::slot_lanes(n)
            .ok()
            .and_then(|l| Chain::of(l.overlap.units, l.overlap.cols, 1))
            .unwrap_or_else(|| {
                panic!(
                    "deepseek41 SlotRows::chain_of: a pass of {n} rows; the owner holds a pass \
                     to {PAIR_ROWS} rows"
                )
            })
    }

    /// The pass's host half, row by row in `rows` order, each row as the
    /// step's host half plans it
    /// ([`ChainBody::decode_input`](bloomery_gpu::model::ChainBody::decode_input),
    /// then its refresh) on its slot's sequence. Refused by name before any
    /// row is planned: a poisoned host tier, a feature tap attached (it reads
    /// one sequence's positions a row), a route trace attached (it records
    /// one sequence's positions), a pass of another shape than the two the
    /// walk runs ([`PassShape`]), ids of another count than the rows, and a
    /// slot whose sequence an earlier step's failed rows condemned or that
    /// does not stand at the slot's position ([`admit_slot`]). A prompt call
    /// holds the body for the whole call, so no prompt group is in flight
    /// here.
    fn plan_slots(
        &mut self,
        stream: &CudaStream,
        parked: &mut [&mut Seq],
        rows: &[SlotRange],
        ids: &[u32],
    ) -> Result<(), GpuError> {
        const WHAT: &str = "deepseek41 Body::plan_slots";
        self.hybrid.refuse_if_poisoned(WHAT)?;
        if self.tap.is_some() {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a body with no feature tap (attach_features): the tap reads one \
                          sequence's positions a row",
            });
        }
        if self.hybrid.route_traced() {
            return Err(GpuError::State {
                what: WHAT,
                missing: "a host tier with no route trace attached: the trace records one \
                          sequence's positions",
            });
        }
        pass_shape(WHAT, rows)?;
        let total = rows.last().map_or(0, |r| r.rows.end);
        if ids.len() != total {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("{} ids for a pass of {total} rows", ids.len()),
            });
        }
        let positions = self.positions();
        let mut entries = parked.iter_mut();
        for r in rows {
            let seq = match home(WHAT, &mut entries, r)? {
                Some(seq) => &*seq,
                None => &self.seq,
            };
            admit_slot(WHAT, seq, r, positions)?;
        }
        let mut entries = parked.iter_mut();
        for r in rows {
            let mut seq = home(WHAT, &mut entries, r)?;
            let tokens = ids.get(r.rows.clone()).ok_or(GpuError::State {
                what: WHAT,
                missing: "the row's id",
            })?;
            for ((row, pos), &token) in r.rows.clone().zip(r.pos0..).zip(tokens) {
                self.decode_on(WHAT, seq.as_deref_mut(), token, pos)?;
                self.refresh_on(stream, seq.as_deref_mut(), &StepInput { pos }, row)?;
            }
        }
        Ok(())
    }

    /// The pass the last [`SlotRows::plan_slots`] planned on the pair pass's
    /// walk ([`walk_pair`]), row `r` into `heads[r]`, each row's caches and
    /// ring shadows its slot's sequence's ([`RowSeqs`]). A pass of another
    /// shape, a busy slot with no parked entry, or another count of heads is
    /// refused by name. Asynchronous as
    /// [`Body::enqueue_pair`](super::Body::enqueue_pair) is.
    fn enqueue_slots(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        heads: &mut [Head],
        parked: &mut [&mut Seq],
        rows: &[SlotRange],
    ) -> Result<(), GpuError> {
        const WHAT: &str = "deepseek41 Body::enqueue_slots";
        let shape = pass_shape(WHAT, rows)?;
        let [a, b] = heads else {
            return Err(GpuError::Shape {
                what: WHAT,
                detail: format!("{} heads; the pass runs {PAIR_ROWS}", heads.len()),
            });
        };
        self.arrive_eager(gpu.stream())?;
        let (parts, hybrid) = self.parts_with(|live| row_seqs(WHAT, live, parked, &shape))?;
        walk_pair(gpu, w, parts, hybrid, [a, b], WHAT)
    }
}

/// The two passes of several slots the pair walk runs.
enum PassShape<'r> {
    /// One slot's next two tokens: the verify pair on that slot's sequence.
    OneSlot(&'r SlotRange),
    /// Two slots' next token each, one layer apart.
    TwoSlots([&'r SlotRange; PAIR_ROWS]),
}

/// `rows`' shape, or the named refusal of any other pass.
fn pass_shape<'r>(what: &'static str, rows: &'r [SlotRange]) -> Result<PassShape<'r>, GpuError> {
    match rows {
        [r] if r.rows.len() == PAIR_ROWS => Ok(PassShape::OneSlot(r)),
        [a, b] if a.rows.len() == 1 && b.rows.len() == 1 => Ok(PassShape::TwoSlots([a, b])),
        _ => Err(GpuError::Shape {
            what,
            detail: format!(
                "a pass of rows {:?}: this body's pass of several slots is two slots of one row \
                 each or one slot of two rows; one row is its slot's step",
                rows.iter()
                    .map(|r| (r.slot, r.rows.len()))
                    .collect::<Vec<_>>()
            ),
        }),
    }
}

/// The sequence of range `r`'s slot with the homes canonical: `None` for
/// slot 0, whose sequence is the live one, else the next of `entries` — the
/// owner's parked list, every other busy slot's sequence in `rows` order.
fn home<'p>(
    what: &'static str,
    entries: &mut std::slice::IterMut<'p, &mut Seq>,
    r: &SlotRange,
) -> Result<Option<&'p mut Seq>, GpuError> {
    if r.slot == 0 {
        return Ok(None);
    }
    match entries.next() {
        Some(seq) => Ok(Some(&mut **seq)),
        None => Err(GpuError::Shape {
            what,
            detail: format!(
                "slot {}'s parked sequence, which the pass was not handed",
                r.slot
            ),
        }),
    }
}

/// The sequences of the pass `shape` ([`RowSeqs`]): the live one for slot 0,
/// the next of `parked` for every other busy slot ([`home`]).
fn row_seqs<'a>(
    what: &'static str,
    live: &'a mut Seq,
    parked: &'a mut [&mut Seq],
    shape: &PassShape<'_>,
) -> Result<RowSeqs<'a>, GpuError> {
    let mut live = Some(live);
    let mut entries = parked.iter_mut();
    let mut seq_of = |r: &SlotRange| -> Result<&'a mut Seq, GpuError> {
        match home(what, &mut entries, r)? {
            Some(seq) => Ok(seq),
            None => live.take().ok_or_else(|| GpuError::Shape {
                what,
                detail: "slot 0 twice in one pass".to_string(),
            }),
        }
    };
    Ok(match shape {
        PassShape::OneSlot(r) => RowSeqs::One(seq_of(r)?),
        PassShape::TwoSlots([a, b]) => RowSeqs::Two([seq_of(a)?, seq_of(b)?]),
    })
}

/// Range `r`'s slot admitted on its sequence `seq`, as the step admits the
/// live one ([`Body::admit`]'s check of failed rows), and standing where the
/// pass plans its rows: at the slot's position, with room in caches of
/// `positions` for its rows.
fn admit_slot(
    what: &'static str,
    seq: &Seq,
    r: &SlotRange,
    positions: usize,
) -> Result<(), GpuError> {
    if seq.rows_failed {
        return Err(GpuError::Shape {
            what,
            detail: format!(
                "slot {} needs a reset: an earlier step's engram rows failed after its launch, \
                 and the card ran that step on stale rows",
                r.slot
            ),
        });
    }
    let at = seq.history.len();
    if at != r.pos0 as usize || at + r.rows.len() > positions {
        return Err(GpuError::Shape {
            what,
            detail: format!(
                "slot {}'s {} rows from position {} after {at} tokens, in caches of {positions} \
                 positions",
                r.slot,
                r.rows.len(),
                r.pos0
            ),
        });
    }
    Ok(())
}
