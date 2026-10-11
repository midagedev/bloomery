//! The host→card lane's rate, as a load measures it once before it plans.
//!
//! A placed load's prompt runs its host experts onto the card over this
//! lane ([`crate::host::xstream`]), and the plan that decides how many
//! experts the card holds depends on the rate it gets. The one measurement
//! the tree had ([`crate::host::xstream::XStream`]'s probe) runs on the
//! ring a placed load already made, so no load could learn the lane before
//! it planned by it. [`lane_rate`] is that measurement: a load calls it
//! once, before the room its plan reads, and prints what it returned beside
//! its `load` record.
//!
//! The probe's buffers — a pinned host extent and a card extent — are
//! dropped with the handle before [`probe_card`] returns, so the room a
//! plan reads after it holds none of the probe's bytes. A load that probes
//! after its load would instead plan against room the model already took,
//! and a card allocation can fail on a full card; probe first, plan after.

use std::time::Instant;

use cuda_core::{DeviceBuffer, PinnedHostBuffer, sys};

use crate::{Gpu, GpuError, graph};

/// What each of [`lane_rate`]'s copies streams: enough that the copy's
/// time dominates its own issue overhead, little enough that the pinning
/// stays off a load's critical path.
pub const PROBE_BYTES: usize = 64 << 20;

/// How many copies [`lane_rate`] times: a median needs one.
const REPS: usize = 3;

/// One measurement of the host→card lane: `bytes` streamed from pinned host
/// memory to the card [`REPS`] times, each timed from its issue to the copy
/// stream's drain, as [`crate::host::xstream::XStream`]'s probe times its
/// own copies.
#[derive(Clone, Copy)]
pub struct LaneRate {
    /// The bytes each copy streamed.
    pub bytes: u64,
    /// The copies' median rate, GB/s.
    pub gbs_median: f64,
    /// The copies' least rate, GB/s.
    pub gbs_min: f64,
    /// Whether the host extent was page-locked: a pageable copy measures a
    /// different lane, and the number says which.
    pub pinned: bool,
}

/// The host→card lane's rate over `gpu`'s context: `bytes` copied from a
/// pinned host extent to a card extent on the engine's stream [`REPS`]
/// times, each timed from its issue to the stream's drain. Both extents
/// are dropped with the stream's drain before the return, so a room read
/// after this call sees none of its bytes.
///
/// The copies run on the engine's own stream, not a background-role one:
/// the probe is load-time, no engine work exists to stay behind, and a
/// device whose priority range holds one value refuses a background stream
/// outright ([`crate::role_stream`]).
pub fn lane_rate(gpu: &Gpu, bytes: usize) -> Result<LaneRate, GpuError> {
    const WHAT: &str = "lane_rate";
    if bytes == 0 {
        return Err(GpuError::shape(WHAT, "a probe of no bytes"));
    }
    let ctx = gpu.context();
    let stream = gpu.stream();
    let host = PinnedHostBuffer::<u8>::zeroed(ctx, bytes)?;
    let dst = DeviceBuffer::<u8>::zeroed(stream, bytes)?;
    // The allocations' zeroing drains before the first timed copy, so no
    // copy's window carries another's memset.
    stream.synchronize()?;
    let mut rates = [0.0f64; REPS];
    for rate in &mut rates {
        let t0 = Instant::now();
        // SAFETY: `host` is `bytes` of pinned memory this scope owns and
        // nothing writes until the synchronize below; `dst` is a device
        // buffer of `bytes` on the same context as the stream.
        let rc = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst.cu_deviceptr(),
                host.as_ptr().cast(),
                bytes,
                stream.cu_stream(),
            )
        };
        graph::cu(rc, "cuMemcpyHtoDAsync_v2 (lane probe)")?;
        stream.synchronize()?;
        *rate = bytes as f64 / (t0.elapsed().as_secs_f64() * 1e6);
    }
    rates.sort_by(|a, b| a.partial_cmp(b).expect("finite rates, checked below"));
    let refuse = |what: &str| {
        GpuError::protocol(
            WHAT,
            format!("{what}: the lane's copies returned no usable rate"),
        )
    };
    for r in rates {
        if !r.is_finite() || r <= 0.0 {
            return Err(refuse("a copy of a zero or non-finite rate"));
        }
    }
    let (gbs_median, gbs_min) = (rates[REPS / 2] / 1e3, rates[0] / 1e3);
    if !gbs_median.is_finite() || !gbs_min.is_finite() || gbs_min <= 0.0 {
        return Err(refuse("a median or least of no rate"));
    }
    Ok(LaneRate {
        bytes: u64::try_from(bytes).map_err(|_| GpuError::shape(WHAT, "a probe past u64"))?,
        gbs_median,
        gbs_min,
        pinned: true,
    })
}

/// The lane's rate off the card a load's plan stages on, probed the way a
/// load settles its room before reading it ([`Gpu::hold_card`]): the card
/// opened as the plan resolves it, [`PROBE_BYTES`] streamed over the lane,
/// and every extent and the handle itself dropped before the return — a
/// room read after this call plans by room that holds none of the probe's
/// bytes and none of the handle's.
pub fn probe_card(
    name: &str,
    device: Option<::model::placement::workstation::DeviceId>,
    bytes: usize,
) -> Result<LaneRate, GpuError> {
    let gpu = Gpu::open_card(name, device)?;
    let rate = lane_rate(&gpu, bytes)?;
    drop(gpu);
    Ok(rate)
}

#[cfg(test)]
mod tests {
    use super::{PROBE_BYTES, lane_rate};
    use crate::Gpu;

    /// The lane's rate on a real card: a median inside the band a working
    /// host→card copy over PCIe lands in, above a rate that only times the
    /// issue (an asynchronous copy the probe never drained would return in
    /// microseconds, far past the band's top) and below the card's own
    /// memory bandwidth, with the least at most the median and the bytes
    /// the probe was asked for. A sanity band, not a measurement: it runs
    /// under no lease and names no number outside itself. It holds a card
    /// and a pinned extent, so it runs in the gate's serial `hw_held_`
    /// batch, beside the room-reading tests its pinning must not perturb.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_held_lane_rate_in_band() {
        let gpu = Gpu::with_device(0).expect("a Gpu on device 0");
        let rate = lane_rate(&gpu, PROBE_BYTES).expect("the lane's rate");
        assert_eq!(rate.bytes, PROBE_BYTES as u64, "the bytes asked for");
        assert!(rate.pinned, "the probe's host extent is page-locked");
        assert!(
            rate.gbs_median > 1.0 && rate.gbs_median < 64.0,
            "a median of {} GB/s is outside the sanity band (1, 64)",
            rate.gbs_median
        );
        assert!(
            rate.gbs_min <= rate.gbs_median,
            "a least of {} GB/s over the median {}",
            rate.gbs_min,
            rate.gbs_median
        );
    }
}
