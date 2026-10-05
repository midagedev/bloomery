//! The body's pass of several resident slots ([`SlotRows`]): the pass's
//! rows on the pair pass's walk, one layer apart on the one stream, each row
//! bound to its slot's sequence, so the host serves one row's layer while
//! the card runs the other's.
//!
//! The walk carries two rows in flight with one column each: a pass holds at
//! most [`PAIR_ROWS`] rows — two slots of one row, or one slot of two, which
//! is the verify pair of that slot's sequence. The pass itself is not built
//! yet: the host half and the enqueue are refused by name, so a pass of
//! several slots on this body runs nowhere.

use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::Chain;
use bloomery_gpu::model::{SlotRange, SlotRows};
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use cuda_core::CudaStream;
use runtime::sched;

use super::{Body, PAIR_ROWS, Seq};

impl SlotRows for Body {
    /// Two rows in flight on the step port, one column each.
    const MAX_ROWS: usize = PAIR_ROWS;

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

    /// Refused by name: the per-slot host half is not built.
    fn plan_slots(
        &mut self,
        _stream: &CudaStream,
        _parked: &mut [&mut Seq],
        _rows: &[SlotRange],
        _ids: &[u32],
    ) -> Result<(), GpuError> {
        Err(not_built("deepseek41 Body::plan_slots"))
    }

    /// Refused by name: the walk over several slots' sequences is not
    /// built.
    fn enqueue_slots(
        &mut self,
        _gpu: &Gpu,
        _w: &Weights,
        _heads: &mut [Head],
        _parked: &mut [&mut Seq],
        _rows: &[SlotRange],
    ) -> Result<(), GpuError> {
        Err(not_built("deepseek41 Body::enqueue_slots"))
    }
}

/// The refusal of a pass of several slots on this body.
fn not_built(what: &'static str) -> GpuError {
    GpuError::State {
        what,
        missing: "the V4.1 pass of several slots (round stgv41b builds it; serve the slots by a \
                  select and a step a slot)",
    }
}
