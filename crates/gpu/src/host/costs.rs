//! The device's and the host's side of the stream's and the split walk's prices
//! ([`runtime::costs`]): the card's rates read from its driver attributes ([`card_rates`]), the
//! host's cost of one enqueue ([`enqueue_us`]), and the key a timing of this device and host is
//! kept under in the cost store ([`device_key`]).

use std::fmt::Write as _;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, sys};
pub use runtime::costs::{
    CardRates, ExpertShape, Source, Store, UnionCosts, UnionTimes, card_costs, check_range,
    fit_union,
};

use crate::GpuError;
use crate::graph::cu;

const WHAT: &str = "host::costs";

/// The cost store's refusal ([`Store::get_or`]) is a load's protocol error, by its text.
impl From<runtime::costs::CostError> for GpuError {
    fn from(e: runtime::costs::CostError) -> Self {
        Self::protocol(WHAT, e.to_string())
    }
}

/// The rates of `ctx`'s device ([`CardRates::from_attributes`]).
pub fn card_rates(ctx: &CudaContext) -> Result<CardRates, GpuError> {
    let dev = ctx.cu_device();
    let attr = |a, what| crate::raw_device_attribute(dev, a, what);
    let cc = (
        attr(
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MAJOR)",
        )?,
        attr(
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            "cuDeviceGetAttribute(COMPUTE_CAPABILITY_MINOR)",
        )?,
    );
    CardRates::from_attributes(
        attr(
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            "cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)",
        )?,
        attr(
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
            "cuDeviceGetAttribute(CLOCK_RATE)",
        )?,
        cc,
        (
            attr(
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                "cuDeviceGetAttribute(MEMORY_CLOCK_RATE)",
            )?,
            attr(
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
                "cuDeviceGetAttribute(GLOBAL_MEMORY_BUS_WIDTH)",
            )?,
        ),
    )
    .map_err(|e| GpuError::shape(WHAT, e.to_string()))
}

/// The host's cost of one asynchronous enqueue on `stream` [µs]: the least of three timings
/// of 64 one-byte memsets enqueued back to back, the stream drained between them.
pub fn enqueue_us(stream: &CudaStream) -> Result<f64, GpuError> {
    const ENQUEUES: usize = 64;
    let buf = DeviceBuffer::<u8>::zeroed(stream, 1)?;
    stream.synchronize()?;
    let mut least = f64::INFINITY;
    for _ in 0..3 {
        let t0 = Instant::now();
        for _ in 0..ENQUEUES {
            // SAFETY: `buf` is one live device byte on the stream's context, which outlives
            // the drain below.
            let rc = unsafe { sys::cuMemsetD8Async(buf.cu_deviceptr(), 0, 1, stream.cu_stream()) };
            cu(rc, "cuMemsetD8Async (enqueue probe)")?;
        }
        least = least.min(t0.elapsed().as_secs_f64() * 1e6 / ENQUEUES as f64);
        stream.synchronize()?;
    }
    Ok(least)
}

/// The key of `ctx`'s device and this host a measured value is kept under: the card's UUID,
/// the driver's version, the host CPU's model and logical CPUs, and the effective host kernels.
pub fn device_key(ctx: &CudaContext) -> Result<String, GpuError> {
    let uuid = crate::raw_device_uuid(ctx.cu_device())?;
    let mut driver = 0;
    cu(
        // SAFETY: `driver` is a live out-pointer for the call.
        unsafe { sys::cuDriverGetVersion(&mut driver) },
        "cuDriverGetVersion",
    )?;
    let mut key = String::from("card=");
    for b in uuid {
        let _ = write!(key, "{b:02x}");
    }
    // The host's online CPUs, not the calling thread's mask: a load runs pinned.
    // SAFETY: sysconf reads a constant of the running system.
    let cpus = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    let _ = write!(
        key,
        " driver={driver} {}",
        runtime::costs::cpu_key("/proc/cpuinfo".as_ref(), cpus, model::ops::host_kernels())?
    );
    Ok(key)
}
