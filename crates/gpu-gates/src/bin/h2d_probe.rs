//! Host-to-device probe: copies `--bytes` (1 GiB by default) host → device
//! by three paths on device 0 of `CUDA_VISIBLE_DEVICES` and prints one line
//! per path:
//!
//! ```text
//! h2d path=pinned bytes=<n> reps=<k> GB/s=<v>
//! h2d path=pageable bytes=<n> reps=<k> GB/s=<v>
//! h2d path=registered bytes=<n> reps=<k> GB/s=<v> register_ms=<v> unregister_ms=<v>
//! ```
//!
//! - `pinned`: a page-locked allocation (`cuMemAllocHost`) to the device,
//!   async copies timed with a pair of events each.
//! - `pageable`: the first `bytes` of a read-only `mmap` of the V4.1 first
//!   shard (`$BLOOMERY_V41_MODEL`), every page read once first so the range
//!   is warm, then plain copies (the driver stages them), each timed on the
//!   host clock from a quiet stream to its return.
//! - `registered`: the same mapped range registered with `cuMemHostRegister`
//!   and `CU_MEMHOSTREGISTER_READ_ONLY` (the call timed on the host clock),
//!   then async copies timed with events, then the unregister, timed.
//!
//! Each path makes one untimed copy first, then `--reps` timed ones (at
//! least five); `GB/s` is `bytes · reps` over their summed time, in 10⁹
//! bytes per second. A number printed here outside `just time-gpu-h2d` is a
//! runtime value of whatever else the box was doing.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("h2d_probe: built without the `gpu` feature; see `just time-gpu-h2d`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("h2d_probe", probe::run())
}

#[cfg(feature = "gpu")]
mod probe {
    use std::ffi::c_void;
    use std::fs::File;
    use std::sync::Arc;
    use std::time::Instant;

    use bloomery_gpu_gates::GateError;
    use cuda_core::{CudaContext, CudaStream, DeviceBuffer, IntoResult, PinnedHostBuffer, sys};
    use memmap2::Mmap;

    /// Bytes each path copies unless `--bytes` says otherwise: 1 GiB.
    const BYTES: usize = 1 << 30;
    /// Timed copies per path unless `--reps` says otherwise.
    const REPS: usize = 8;
    /// The fewest timed copies a path may take.
    const MIN_REPS: usize = 5;
    /// The page size: the registered range starts on a page (the map's
    /// start) and spans whole pages.
    const PAGE: usize = 4096;

    struct Args {
        bytes: usize,
        reps: usize,
    }

    fn args() -> Result<Args, GateError> {
        let mut a = Args {
            bytes: BYTES,
            reps: REPS,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut value = |name: &str| -> Result<usize, GateError> {
                let v = it.next().ok_or_else(|| format!("{name} takes a value"))?;
                v.parse().map_err(|e| format!("{name} {v}: {e}").into())
            };
            match flag.as_str() {
                "--bytes" => a.bytes = value("--bytes")?,
                "--reps" => a.reps = value("--reps")?,
                other => {
                    return Err(format!(
                        "unknown argument {other}; usage: h2d_probe [--bytes N] [--reps K]"
                    )
                    .into());
                }
            }
        }
        if a.bytes == 0 || !a.bytes.is_multiple_of(PAGE) || a.reps < MIN_REPS {
            return Err(format!(
                "--bytes {} and --reps {}: the bytes a positive multiple of {PAGE}, the reps at \
                 least {MIN_REPS}",
                a.bytes, a.reps
            )
            .into());
        }
        Ok(a)
    }

    /// One untimed `copy`, then `reps` of them between a pair of timing
    /// events each: the summed milliseconds.
    fn timed_async(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        reps: usize,
        mut copy: impl FnMut() -> Result<(), GateError>,
    ) -> Result<f64, GateError> {
        copy()?;
        stream.synchronize()?;
        let flags = Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
        let (start, end) = (ctx.new_event(flags)?, ctx.new_event(flags)?);
        let mut ms = 0.0f64;
        for _ in 0..reps {
            start.record(stream)?;
            copy()?;
            end.record(stream)?;
            stream.synchronize()?;
            ms += f64::from(start.elapsed_ms(&end)?);
        }
        Ok(ms)
    }

    /// Device attribute `attr` of `ctx`'s card.
    fn attribute(ctx: &CudaContext, attr: sys::CUdevice_attribute) -> Result<i32, GateError> {
        let mut v = 0;
        // SAFETY: the call writes the live local `v`; the device is the
        // context's own.
        let rc = unsafe { sys::cuDeviceGetAttribute(&mut v, attr, ctx.cu_device()) };
        rc.result()
            .map_err(|e| format!("cuDeviceGetAttribute({attr}): {e:?}"))?;
        Ok(v)
    }

    /// The line's `GB/s` for `reps` copies of `bytes` in `ms` milliseconds.
    fn gbps(bytes: usize, reps: usize, ms: f64) -> f64 {
        (bytes * reps) as f64 / (ms * 1e-3) / 1e9
    }

    pub fn run() -> Result<(), GateError> {
        let a = args()?;
        let path = std::env::var("BLOOMERY_V41_MODEL").map_err(|_| {
            "BLOOMERY_V41_MODEL is unset: the pageable and registered paths read the V4.1 first \
             shard (tools/box.sh exports it)"
        })?;
        let ctx = CudaContext::new(0)?;
        let stream = ctx.new_stream()?;
        let (host_register, read_only_register, pageable_access) = (
            attribute(
                &ctx,
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED,
            )?,
            attribute(
                &ctx,
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED,
            )?,
            attribute(
                &ctx,
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
            )?,
        );
        println!(
            "h2d card={} shard={path} bytes={} reps={} host_register={host_register} \
             read_only_register={read_only_register} pageable_access={pageable_access}",
            ctx.device_name()?,
            a.bytes,
            a.reps
        );
        let mut dev = DeviceBuffer::<u8>::zeroed(&stream, a.bytes)?;

        let pinned = PinnedHostBuffer::<u8>::zeroed(&ctx, a.bytes)?;
        let ms = timed_async(&ctx, &stream, a.reps, || {
            // SAFETY: `pinned` outlives every copy: `timed_async` synchronizes
            // the stream after each, and `pinned` drops after it returns.
            unsafe { dev.copy_from_pinned_host_async(&stream, &pinned)? };
            Ok(())
        })?;
        drop(pinned);
        println!(
            "h2d path=pinned bytes={} reps={} GB/s={:.2}",
            a.bytes,
            a.reps,
            gbps(a.bytes, a.reps, ms)
        );

        let file = File::open(&path).map_err(|e| format!("{path}: {e}"))?;
        // SAFETY: the shard is opened read-only and nothing writes the model
        // files while a probe or a gate runs; the map is only read, here and
        // by the driver's copies.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {path}: {e}"))?;
        let src = map.get(..a.bytes).ok_or_else(|| {
            format!(
                "{path} holds {} bytes, the probe copies {}",
                map.len(),
                a.bytes
            )
        })?;
        let warm = src
            .iter()
            .step_by(PAGE)
            .fold(0u64, |s, &b| s.wrapping_add(u64::from(b)));
        std::hint::black_box(warm);

        dev.copy_from_host(&stream, src)?;
        let mut ms = 0.0f64;
        for _ in 0..a.reps {
            let t = Instant::now();
            dev.copy_from_host(&stream, src)?;
            ms += t.elapsed().as_secs_f64() * 1e3;
        }
        println!(
            "h2d path=pageable bytes={} reps={} GB/s={:.2}",
            a.bytes,
            a.reps,
            gbps(a.bytes, a.reps, ms)
        );

        let at = src.as_ptr().cast_mut().cast::<c_void>();
        ctx.bind_to_thread()?;
        let t = Instant::now();
        // SAFETY: `at .. at + bytes` is the live read-only map's first
        // `bytes` bytes, page-aligned (a map starts on a page) and whole
        // pages; READ_ONLY registers it without asking for write access, and
        // the unregister below runs before the map drops.
        let rc =
            unsafe { sys::cuMemHostRegister_v2(at, a.bytes, sys::CU_MEMHOSTREGISTER_READ_ONLY) };
        let register_ms = t.elapsed().as_secs_f64() * 1e3;
        rc.result().map_err(|e| {
            format!(
                "cuMemHostRegister(READ_ONLY) of {path}'s map: {e:?} (the card reports \
                     host_register={host_register} read_only_register={read_only_register})"
            )
        })?;
        let copies = timed_async(&ctx, &stream, a.reps, || {
            // SAFETY: `src` is the registered map, which outlives every copy:
            // `timed_async` synchronizes the stream after each.
            unsafe { dev.copy_from_host_async_unchecked(&stream, src)? };
            Ok(())
        });
        stream.synchronize()?;
        let t = Instant::now();
        // SAFETY: `at` is the base the register call above took, and no copy
        // from it is in flight: the stream is synchronized.
        let rc = unsafe { sys::cuMemHostUnregister(at) };
        let unregister_ms = t.elapsed().as_secs_f64() * 1e3;
        rc.result()
            .map_err(|e| format!("cuMemHostUnregister of {path}'s map: {e:?}"))?;
        let ms = copies?;
        println!(
            "h2d path=registered bytes={} reps={} GB/s={:.2} register_ms={register_ms:.1} \
             unregister_ms={unregister_ms:.1}",
            a.bytes,
            a.reps,
            gbps(a.bytes, a.reps, ms)
        );
        Ok(())
    }
}
