//! The timed command buffer the development probes share: a buffer holds
//! many dispatches, its time is its own `GPUEndTime` minus `GPUStartTime`,
//! read after it completed, and the figure is the median over the buffers.
//! A figure it gives is this machine's, taken with no timing lease.

use std::time::Duration;

use bloomery_device::conformance::why;
use bloomery_device::{DeviceError, FaultWord, Stream};
use bloomery_metal::device::MetalDevice;
use bloomery_metal::encode::ComputeEncoder;
use bloomery_metal::{GpuInterval, MetalStream};

/// The warm-up buffers ahead of the timed ones.
pub const WARMUPS: usize = 2;

/// How long the host waits for one timed command buffer.
const BUFFER_WAIT: Duration = Duration::from_secs(60);

/// One arm's figure: the per-dispatch microseconds over the timed buffers.
pub struct Timed {
    pub us_median: f64,
    pub us_min: f64,
    pub us_p90: f64,
}

pub fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Nearest rank of `p`: the value at `ceil(p * len)`, counting from one.
pub fn percentile(xs: &[f64], p: f64) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let rank = (p * v.len() as f64).ceil() as usize;
    v[rank.saturating_sub(1)]
}

/// Encode one timed command buffer per repeat, submit them one at a time and
/// read each buffer's GPU interval once it completed, so no buffer's interval
/// holds a neighbour's work. `encode` launches dispatch `n` of the
/// buffer; a barrier separates the dispatches. The warm-up buffers are not
/// part of the figure, and a raised fault word fails the arm.
pub fn time_arm(
    device: &MetalDevice,
    stream: &MetalStream,
    dispatches: usize,
    trials: usize,
    mut encode: impl FnMut(&mut ComputeEncoder<'_>, usize) -> Result<(), DeviceError>,
) -> Result<Timed, String> {
    if dispatches == 0 || trials == 0 {
        return Err("a timed arm needs a dispatch and a repeat".to_string());
    }
    let buffers = WARMUPS + trials;
    let mut pending = Vec::with_capacity(buffers);
    for _ in 0..buffers {
        let mut recording = stream.begin().map_err(why)?;
        let stamps = recording.stamps();
        {
            let mut encoder = recording.compute_encoder().map_err(why)?;
            for n in 0..dispatches {
                if n > 0 {
                    encoder.memory_barrier();
                }
                encode(&mut encoder, n).map_err(why)?;
            }
            encoder.end();
        }
        pending.push((recording, stamps));
    }
    let mut gaps = Vec::with_capacity(trials);
    for (i, (recording, stamps)) in pending.into_iter().enumerate() {
        stream.submit(recording).map_err(why)?;
        let gpu = match stamps.interval(BUFFER_WAIT) {
            Ok(GpuInterval::Ran(gpu)) => gpu,
            Ok(GpuInterval::StillRunning) => {
                return Err(
                    "the command buffer was still running on the GPU when the wait ended"
                        .to_string(),
                );
            }
            Err(e) => {
                let _drained = stream.sync();
                return Err(why(e));
            }
        };
        if i >= WARMUPS {
            gaps.push(gpu / dispatches as f64 * 1e6);
        }
    }
    stream.sync().map_err(why)?;
    if let Some(fault) = device.fault_word().read(stream).map_err(why)? {
        return Err(format!("the launch raised {fault}"));
    }
    device.fault_word().clear(stream).map_err(why)?;
    Ok(Timed {
        us_median: median(&gaps),
        us_min: gaps.iter().copied().fold(f64::INFINITY, f64::min),
        us_p90: percentile(&gaps, 0.9),
    })
}
