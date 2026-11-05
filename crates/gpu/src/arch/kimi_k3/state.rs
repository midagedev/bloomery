//! The state book of Kimi K3's step: what a decode leaves on the card for the next one.
//!
//! - A delta layer's recurrent store ([`kda_store`]): one lane of the `[heads][head_dim][head_dim]`
//!   f32 state matrices, the conv ring of the last [`RING_ROWS`](crate::linear::RING_ROWS)
//!   positions' q·k·v inputs, and the lane's stamp, which the delta step checks against the
//!   position it stands at (a store a reset missed, or a step out of order, raises the fault word
//!   instead of reading another position's state). Every step runs in place on lane 0; the lane
//!   word ([`Lane`]) is one device word that never moves.
//! - A latent layer's cache: `[ctx][row]` f16, indexed by position and read only below the live
//!   count, so a reset needs nothing of it: every row below the live count is written by its own
//!   step before any launch reads it.
//! - The position and the token ([`StepIn`]): the one refresh before each step. The embedding
//!   writes the row's position and live key count from them.
//!
//! [`clear_store`] puts a store back to a fresh one in place (a captured chain keeps its
//! addresses).

use crate::GpuError;
use crate::linear::LinearShape;
use crate::linear::mixer_kda::KdaState;
use crate::recur::RecStore;
use cuda_core::{CudaStream, DeviceBuffer};

/// What the refusals of this file name.
const WHAT: &str = "kimi_k3 state";

/// One delta layer's recurrent store at `shape`, zeroed, lane 0 stamped at position 0.
/// Load-time only.
pub(super) fn kda_store(stream: &CudaStream, shape: LinearShape) -> Result<RecStore, GpuError> {
    RecStore::stamped(stream, shape.lens(1), 1)
}

/// `store` back to a fresh one in place: the state and the ring zero, the stamp at position 0.
/// Never inside a capture.
pub(super) fn clear_store(stream: &CudaStream, store: &mut RecStore) -> Result<(), GpuError> {
    store.clear(stream)?;
    store.restamp(stream)
}

/// The lane word every delta layer's step reads: lane 0, the only lane. One word serves every
/// layer.
pub(super) struct Lane(DeviceBuffer<u32>);

impl Lane {
    /// The word at lane 0. Load-time only.
    pub(super) fn new(stream: &CudaStream) -> Result<Lane, GpuError> {
        Ok(Lane(DeviceBuffer::from_host(stream, &[0u32])?))
    }

    /// The store a delta layer's step runs on, in place on lane 0. A store of other lanes than
    /// one, or with no stamps, is refused by name.
    pub(super) fn state<'a>(&'a self, store: &'a mut RecStore) -> Result<KdaState<'a>, GpuError> {
        if store.lanes != 1 {
            return Err(GpuError::shape(
                WHAT,
                format!("a store of {} lanes under the step's one lane", store.lanes),
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
            lane: &self.0,
            lanes: *lanes,
            each: false,
            row: 0,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        self.0.num_bytes()
    }
}

/// The step's input, rewritten by every refresh: the token, the position, and the embedding's
/// outputs for the one row, its position and its live key count.
pub(super) struct StepIn {
    pub(super) ids: DeviceBuffer<u32>,
    pub(super) pos0: DeviceBuffer<u32>,
    pub(super) pos: DeviceBuffer<u32>,
    pub(super) n_keys: DeviceBuffer<u32>,
}

impl StepIn {
    pub(super) fn new(stream: &CudaStream) -> Result<StepIn, GpuError> {
        let one = || DeviceBuffer::<u32>::zeroed(stream, 1);
        Ok(StepIn {
            ids: one()?,
            pos0: one()?,
            pos: one()?,
            n_keys: one()?,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        [&self.ids, &self.pos0, &self.pos, &self.n_keys]
            .into_iter()
            .map(DeviceBuffer::num_bytes)
            .sum()
    }
}
