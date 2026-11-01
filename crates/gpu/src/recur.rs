//! Recurrent state a layer keeps on the card: the state lanes, the conv
//! ring and, for a gated-delta-rule layer ([`RecStore::gdn`]), the carry of
//! its chunked prompt rule and the record of its verify
//! (`linear::delta`'s module doc). A delta layer passes its
//! [`crate::linear::LinearShape`]; a later scan passes its own lengths. The
//! lengths are the only architecture facts this module takes.

use crate::GpuError;
use crate::linear::LinearShape;
use crate::linear::delta::{NEVER, recovery_record_len};
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
    /// The carry ([`RecStore::gdn`]); `None` on a store whose prompt does
    /// not run the chunked rule.
    pub(crate) carry: Option<DeviceBuffer<f32>>,
    /// The verify record ([`RecStore::gdn`]); `None` on a store whose
    /// verify keeps a lane a row.
    pub(crate) record: Option<DeviceBuffer<f32>>,
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
            carry: None,
            record: None,
        })
    }

    /// A gated-delta-rule layer's store of `shape`: one state lane stamped
    /// at position 0, a zeroed carry of the chunked prompt rule and, for
    /// `rows` past 0, a verify record of `rows` rows (`linear::delta`'s
    /// module doc), the layout every body's delta layer keeps. Load-time
    /// only.
    pub(crate) fn gdn(
        stream: &CudaStream,
        shape: LinearShape,
        rows: usize,
    ) -> Result<RecStore, GpuError> {
        let mut r = RecStore::stamped(stream, shape.lens(1), 1)?;
        r.carry = Some(DeviceBuffer::zeroed(stream, shape.carry_len())?);
        if rows > 0 {
            r.record = Some(DeviceBuffer::zeroed(
                stream,
                recovery_record_len(shape, rows),
            )?);
        }
        Ok(r)
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

    /// Every lane of the state, the whole ring and the carry back to zero on
    /// the device, ordered on `stream`. The stamps are unchanged.
    pub(crate) fn clear(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        self.state.zero_async(stream)?;
        self.ring.zero_async(stream)?;
        if let Some(carry) = self.carry.as_mut() {
            carry.zero_async(stream)?;
        }
        Ok(())
    }

    pub(crate) fn bytes(&self) -> usize {
        self.state.num_bytes()
            + self.ring.num_bytes()
            + self.stamp.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.carry.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.record.as_ref().map_or(0, DeviceBuffer::num_bytes)
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

/// A closing mark's point at `at` ([`runtime::prompt::fold_marks`]) copied
/// from stores that stand a row past it: `stores` start with each of the
/// `delta` layers' lane, ring and carry ([`RecStore::gdn`]'s checkpoint
/// order). The walk kept its last chunk out of the lanes (the prompt form's
/// `keep`), so each carry gives back the row past `at`: its count one less,
/// standing at `at`. The rings keep a pass past their reach and are read by
/// position, so they stand as copied. Refused by name when a carry does not
/// stand a row past `at` with a row in it.
pub(crate) fn close_point(
    stores: &mut [&mut [f32]],
    delta: usize,
    at: u32,
) -> Result<(), GpuError> {
    const WHAT: &str = "recur::close_point";
    for i in (0..delta).map(|j| 3 * j + 2) {
        let carry = stores
            .get_mut(i)
            .filter(|c| c.len() >= 2)
            .ok_or(GpuError::state(
                WHAT,
                "each delta layer's carry in the point",
            ))?;
        let (rows, past) = (carry[0].to_bits(), carry[1].to_bits());
        if rows == 0 || past != at + 1 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "the closing mark at {at}: store {i}'s carry holds {rows} rows standing at \
                     {past}, not a row standing at {}",
                    at + 1
                ),
            ));
        }
        carry[0] = f32::from_bits(rows - 1);
        carry[1] = f32::from_bits(at);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Closing keeps every lane and ring and returns the last carried row.
    /// Omitting either refusal or changing the mark/count fails this clause.
    #[test]
    fn closing_point_checks_each_carry() {
        let mut point = [
            vec![1.0],
            vec![2.0],
            vec![f32::from_bits(8), f32::from_bits(17), 3.0],
        ];
        close_point(
            &mut point.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(),
            1,
            16,
        )
        .unwrap();
        assert_eq!(point[0], [1.0]);
        assert_eq!(point[1], [2.0]);
        assert_eq!(
            point[2].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            [7, 16, 3.0f32.to_bits()]
        );
        for (delta, carry) in [
            (1, vec![]),
            (1, vec![0.0]),
            (1, vec![0.0, f32::from_bits(17)]),
            (1, vec![f32::from_bits(1), f32::from_bits(16)]),
            (2, vec![f32::from_bits(1), f32::from_bits(17)]),
        ] {
            point[2] = carry;
            let err = close_point(
                &mut point.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(),
                delta,
                16,
            )
            .unwrap_err();
            assert!(err.to_string().contains("recur::close_point"));
        }
    }

    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_clear_keeps_stamps_and_addresses() -> Result<(), GpuError> {
        let (_ctx, stream) = crate::capsync::fresh_stream(0)?;
        for (lanes, carry) in [(1, 0), (2, 7), (4, 9)] {
            let mut store = RecStore::stamped(&stream, [17 * lanes, 13], lanes)?;
            if carry > 0 {
                store.carry = Some(DeviceBuffer::zeroed(&stream, carry)?);
            }
            let lane = u32::try_from(lanes - 1).expect("test lane fits u32");
            store.restamp_at(&stream, lane, 29)?;
            let stamps = store
                .stamp
                .as_ref()
                .expect("stamped store")
                .to_host_vec(&stream)?;
            let addresses = (
                store.state.cu_deviceptr(),
                store.ring.cu_deviceptr(),
                store.carry.as_ref().map(DeviceBuffer::cu_deviceptr),
            );
            for buf in [&mut store.state, &mut store.ring]
                .into_iter()
                .chain(store.carry.iter_mut())
            {
                buf.copy_from_host(&stream, &vec![f32::NAN; buf.len()])?;
            }
            store.clear(&stream)?;
            for buf in [&store.state, &store.ring]
                .into_iter()
                .chain(store.carry.iter())
            {
                assert!(buf.to_host_vec(&stream)?.iter().all(|v| v.to_bits() == 0));
            }
            assert_eq!(
                addresses,
                (
                    store.state.cu_deviceptr(),
                    store.ring.cu_deviceptr(),
                    store.carry.as_ref().map(DeviceBuffer::cu_deviceptr),
                )
            );
            assert_eq!(
                store
                    .stamp
                    .as_ref()
                    .expect("stamped store")
                    .to_host_vec(&stream)?,
                stamps
            );
        }
        Ok(())
    }
}
