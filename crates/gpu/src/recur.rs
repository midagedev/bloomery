//! Recurrent state a layer keeps on the card: the state lanes and the conv
//! ring. A delta layer passes [`crate::linear::LinearShape::lens`]; a later
//! scan passes its own lengths. The length pair is the only architecture
//! fact this module takes.

use crate::GpuError;
use crate::linear::delta::NEVER;
use cuda_core::{CudaStream, DeviceBuffer};

/// One layer's recurrent store: `state_f32` f32s of state (every lane) and
/// `ring_f32` f32s of conv ring, and — when a load keeps a lane per verify
/// row — the lanes' stamps ([`fresh_stamps`]).
pub(crate) struct RecStore {
    pub(crate) state: DeviceBuffer<f32>,
    pub(crate) ring: DeviceBuffer<f32>,
    pub(crate) lanes: usize,
    /// `lanes` u32 stamps ([`fresh_stamps`]); `None` on a store whose
    /// launches do not read stamps.
    pub(crate) stamp: Option<DeviceBuffer<u32>>,
}

impl RecStore {
    /// A zeroed store of `lanes` lanes over `lens` (`[state_f32, ring_f32]`),
    /// with no stamps. Load-time only.
    pub(crate) fn new(
        stream: &CudaStream,
        lens: [usize; 2],
        lanes: usize,
    ) -> Result<RecStore, GpuError> {
        let [state, ring] = lens;
        Ok(RecStore {
            state: DeviceBuffer::zeroed(stream, state)?,
            ring: DeviceBuffer::zeroed(stream, ring)?,
            lanes,
            stamp: None,
        })
    }

    /// [`RecStore::new`] with its stamps fresh ([`fresh_stamps`]). Load-time
    /// only.
    pub(crate) fn stamped(
        stream: &CudaStream,
        lens: [usize; 2],
        lanes: usize,
    ) -> Result<RecStore, GpuError> {
        let mut r = RecStore::new(stream, lens, lanes)?;
        r.stamp = Some(DeviceBuffer::from_host(stream, &fresh_stamps(lanes))?);
        Ok(r)
    }

    /// The stamps back to a fresh store's ([`fresh_stamps`]); a store with
    /// none is refused by name. Never inside a capture.
    pub(crate) fn restamp(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        let fresh = fresh_stamps(self.lanes);
        self.stamp_with(stream, &fresh)
    }

    /// The stamps with lane `lane` at position `at` and every other lane
    /// never written ([`stamps_at`]). A store with no stamps is refused by
    /// name. Never inside a capture.
    pub(crate) fn restamp_at(
        &mut self,
        stream: &CudaStream,
        lane: u32,
        at: u32,
    ) -> Result<(), GpuError> {
        let st = stamps_at(self.lanes, lane, at)?;
        self.stamp_with(stream, &st)
    }

    fn stamp_with(&mut self, stream: &CudaStream, st: &[u32]) -> Result<(), GpuError> {
        match self.stamp.as_mut() {
            Some(stamp) => Ok(stamp.copy_from_host(stream, st)?),
            None => Err(GpuError::state(
                "recur::RecStore::restamp",
                "a store of stamped lanes",
            )),
        }
    }

    /// Every lane of the state and the whole ring back to zero, in stream
    /// order through `zeros` (host zeros at least as long as the longer of
    /// the two). Never inside a capture.
    pub(crate) fn clear(&mut self, stream: &CudaStream, zeros: &[f32]) -> Result<(), GpuError> {
        for buf in [&mut self.state, &mut self.ring] {
            let n = buf.len();
            let src = zeros.get(..n).ok_or_else(|| {
                GpuError::shape(
                    "recur::RecStore::clear",
                    format!("{} host zeros for a {n}-value buffer", zeros.len()),
                )
            })?;
            buf.copy_from_host(stream, src)?;
        }
        Ok(())
    }

    pub(crate) fn bytes(&self) -> usize {
        self.state.num_bytes()
            + self.ring.num_bytes()
            + self.stamp.as_ref().map_or(0, DeviceBuffer::num_bytes)
    }
}

/// A fresh store's stamps over `lanes` lanes: lane 0 holds position 0, every
/// other lane was never written ([`NEVER`]).
pub(crate) fn fresh_stamps(lanes: usize) -> Vec<u32> {
    let mut st = vec![NEVER; lanes];
    if let Some(first) = st.first_mut() {
        *first = 0;
    }
    st
}

/// The stamps of a store over `lanes` lanes whose lane `lane` stands at
/// position `at` and whose every other lane was never written. A lane at or
/// past `lanes` is refused by name.
pub(crate) fn stamps_at(lanes: usize, lane: u32, at: u32) -> Result<Vec<u32>, GpuError> {
    let mut st = vec![NEVER; lanes];
    *st.get_mut(lane as usize).ok_or_else(|| {
        GpuError::shape("recur::stamps_at", format!("lane {lane} of {lanes} lanes"))
    })? = at;
    Ok(st)
}

/// A layer's state store: `K` (the architecture's K/V planes) or a
/// [`RecStore`].
pub(crate) enum LayerStore<K> {
    Kv(K),
    Rec(RecStore),
}
