//! Host-register probe: can the host tier's weights be page-locked in place?
//! Registers `--bytes` (1 GiB by default) of host memory with
//! `cuMemHostRegister` on device 0 of `CUDA_VISIBLE_DEVICES`, three ways, and
//! prints one line per way:
//!
//! ```text
//! register arm=<arm> bytes=<n> flags=<f> rc=<CUDA_*> register_s=<v> s_per_gib=<v>
//!     rss_anon_kb=<before>-><registered>-><unregistered> rss_file_kb=<...>
//!     h2d_reps=<k> h2d_best_GB/s=<v> h2d_median_GB/s=<v> unregister_rc=<CUDA_*>
//! ```
//!
//! - `anon`: an anonymous private map, flags 0.
//! - `file_private`: a `MAP_PRIVATE | PROT_READ | PROT_WRITE` window of the
//!   V4.1 first shard (`$BLOOMERY_V41_MODEL`) from `--offset` (0 by default),
//!   flags 0 — the form the host set would be registered in.
//! - `file_shared`: the same window mapped `MAP_SHARED | PROT_READ`, flags 0
//!   — the form `gguf::Split` maps the file in today.
//! - `file_private_ro`: the same window mapped `MAP_PRIVATE | PROT_READ`,
//!   `CU_MEMHOSTREGISTER_READ_ONLY` — the form a host set the card only
//!   reads could take.
//!
//! The first line names the card and its `HOST_REGISTER_SUPPORTED` and
//! `READ_ONLY_HOST_REGISTER_SUPPORTED` attributes.
//!
//! The question the `file_private` line answers is its `rss_anon_kb`: flags 0
//! asks the driver for pages the card may write, and a private file page
//! written is a copy — if the registered window moves into anonymous memory,
//! the host set cannot be registered where it is mapped. `RssAnon` and
//! `RssFile` are this process's (`/proc/self/status`), read before the
//! register, after it and after the unregister. The copies are `--reps`
//! (8 by default) `cuMemcpyHtoDAsync` of the whole registered range into one
//! device buffer, each between a pair of events after an untimed one; the
//! best and the median (the upper of the middle two) in 10⁹ bytes per second.
//! An arm whose register is refused prints its code and no copies. Each map is unregistered and unmapped
//! before the next arm. A number printed here outside `just
//! probe-host-register` is a runtime value of whatever else the box was
//! doing.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "probe_host_register: built without the `gpu` feature; see `just probe-host-register`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("probe_host_register", probe::run())
}

#[cfg(feature = "gpu")]
mod probe {
    use std::ffi::{CStr, c_char, c_void};
    use std::fs::File;
    use std::sync::Arc;
    use std::time::Instant;

    use bloomery_gpu_gates::GateError;
    use cuda_core::{CudaContext, CudaStream, DeviceBuffer, IntoResult, sys};
    use memmap2::{MmapMut, MmapOptions};

    /// Bytes each arm registers unless `--bytes` says otherwise: 1 GiB.
    const BYTES: usize = 1 << 30;
    /// Timed copies per arm unless `--reps` says otherwise.
    const REPS: usize = 8;
    /// The page size: a registered range starts on a page (a map's start)
    /// and spans whole pages.
    const PAGE: usize = 4096;
    const GIB: f64 = (1u64 << 30) as f64;

    struct Args {
        bytes: usize,
        reps: usize,
        offset: u64,
    }

    /// `N`, `NK`, `NM` or `NG` (binary units) as bytes.
    fn size(v: &str) -> Result<u64, GateError> {
        let (num, shift) = match v.as_bytes().last() {
            Some(b'K' | b'k') => (&v[..v.len() - 1], 10),
            Some(b'M' | b'm') => (&v[..v.len() - 1], 20),
            Some(b'G' | b'g') => (&v[..v.len() - 1], 30),
            _ => (v, 0),
        };
        let n: u64 = num.parse().map_err(|e| format!("size {v}: {e}"))?;
        n.checked_shl(shift)
            .filter(|b| b >> shift == n)
            .ok_or_else(|| format!("size {v} overflows u64").into())
    }

    fn args() -> Result<Args, GateError> {
        let mut a = Args {
            bytes: BYTES,
            reps: REPS,
            offset: 0,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut value = |name: &str| -> Result<String, GateError> {
                it.next()
                    .ok_or_else(|| format!("{name} takes a value").into())
            };
            match flag.as_str() {
                "--bytes" => a.bytes = usize::try_from(size(&value("--bytes")?)?)?,
                "--reps" => {
                    let v = value("--reps")?;
                    a.reps = v.parse().map_err(|e| format!("--reps {v}: {e}"))?;
                }
                "--offset" => a.offset = size(&value("--offset")?)?,
                other => {
                    return Err(format!(
                        "unknown argument {other}; usage: probe_host_register [--bytes N[K|M|G]] \
                         [--reps K] [--offset N[K|M|G]]"
                    )
                    .into());
                }
            }
        }
        if a.bytes == 0
            || !a.bytes.is_multiple_of(PAGE)
            || !a.offset.is_multiple_of(PAGE as u64)
            || a.reps == 0
        {
            return Err(format!(
                "--bytes {} --offset {} --reps {}: bytes and offset whole pages of {PAGE}, bytes \
                 and reps positive",
                a.bytes, a.offset, a.reps
            )
            .into());
        }
        Ok(a)
    }

    /// The driver's name for `rc`.
    fn rc_name(rc: sys::CUresult) -> String {
        let mut name: *const c_char = std::ptr::null();
        // SAFETY: the call writes a pointer to a static NUL-terminated string
        // into the live local `name`, or fails and leaves it null.
        let known = unsafe { sys::cuGetErrorName(rc, &mut name) };
        if known.result().is_err() || name.is_null() {
            return format!("CUresult({rc})");
        }
        // SAFETY: the driver returned a static NUL-terminated string.
        unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned()
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

    /// This process's `RssAnon` and `RssFile`, in kB.
    fn rss() -> Result<(u64, u64), GateError> {
        let text = std::fs::read_to_string("/proc/self/status")?;
        let field = |key: &str| -> Result<u64, GateError> {
            let line = text
                .lines()
                .find(|l| l.starts_with(key))
                .ok_or_else(|| format!("/proc/self/status has no {key}"))?;
            let kb = line[key.len()..].trim().trim_end_matches("kB").trim();
            kb.parse()
                .map_err(|e| format!("/proc/self/status {key} {kb:?}: {e}").into())
        };
        Ok((field("RssAnon:")?, field("RssFile:")?))
    }

    /// `reps` copies of the registered `bytes` at `src` into `dev`, each
    /// between a pair of events after one untimed copy: the milliseconds of
    /// each.
    fn copies(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        dev: &DeviceBuffer<u8>,
        src: *const c_void,
        bytes: usize,
        reps: usize,
    ) -> Result<Vec<f64>, GateError> {
        let copy = || -> Result<(), GateError> {
            // SAFETY: `src .. src + bytes` is registered host memory that
            // outlives this call (the caller unregisters after it returns);
            // `dev` holds `bytes` bytes; the stream is synchronized before
            // the caller touches either again.
            let rc = unsafe {
                sys::cuMemcpyHtoDAsync_v2(dev.cu_deviceptr(), src, bytes, stream.cu_stream())
            };
            rc.result()
                .map_err(|e| format!("cuMemcpyHtoDAsync: {e:?}").into())
        };
        copy()?;
        stream.synchronize()?;
        let flags = Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
        let (start, end) = (ctx.new_event(flags)?, ctx.new_event(flags)?);
        let mut ms = Vec::with_capacity(reps);
        for _ in 0..reps {
            start.record(stream)?;
            copy()?;
            end.record(stream)?;
            stream.synchronize()?;
            ms.push(f64::from(start.elapsed_ms(&end)?));
        }
        Ok(ms)
    }

    /// One arm: register `at .. at + bytes` with `flags`, copy from it,
    /// unregister; the arm's line. A refused register is an answer, printed
    /// by its code; a failed unregister or copy is an error.
    #[allow(
        clippy::too_many_arguments,
        reason = "one arm's context, range, flags and name, all distinct roles"
    )]
    fn arm(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        dev: &DeviceBuffer<u8>,
        name: &str,
        at: *mut c_void,
        bytes: usize,
        flags: u32,
        reps: usize,
    ) -> Result<bool, GateError> {
        ctx.bind_to_thread()?;
        let before = rss()?;
        let t = Instant::now();
        // SAFETY: `at .. at + bytes` is a live map of whole pages from a page
        // boundary, kept mapped until after the unregister below.
        let rc = unsafe { sys::cuMemHostRegister_v2(at, bytes, flags) };
        let register_s = t.elapsed().as_secs_f64();
        let registered = rss()?;
        let head = format!(
            "register arm={name} bytes={bytes} flags={flags} rc={} register_s={register_s:.3} \
             s_per_gib={:.3}",
            rc_name(rc),
            register_s * GIB / bytes as f64
        );
        if rc.result().is_err() {
            println!(
                "{head} rss_anon_kb={}->{} rss_file_kb={}->{} (not registered: no copies)",
                before.0, registered.0, before.1, registered.1
            );
            return Ok(true);
        }
        let ms = copies(ctx, stream, dev, at.cast_const(), bytes, reps);
        stream.synchronize()?;
        // SAFETY: `at` is the base the register call above took, and no copy
        // from it is in flight: the stream is synchronized.
        let un = unsafe { sys::cuMemHostUnregister(at) };
        let after = rss()?;
        let mut ms = ms?;
        ms.sort_by(f64::total_cmp);
        let gbps = |m: f64| bytes as f64 / (m * 1e-3) / 1e9;
        println!(
            "{head} rss_anon_kb={}->{}->{} rss_file_kb={}->{}->{} h2d_reps={reps} \
             h2d_best_GB/s={:.2} h2d_median_GB/s={:.2} unregister_rc={}",
            before.0,
            registered.0,
            after.0,
            before.1,
            registered.1,
            after.1,
            gbps(ms[0]),
            gbps(ms[ms.len() / 2]),
            rc_name(un)
        );
        Ok(un.result().is_ok())
    }

    pub fn run() -> Result<(), GateError> {
        let a = args()?;
        let path = std::env::var("BLOOMERY_V41_MODEL").map_err(|_| {
            "BLOOMERY_V41_MODEL is unset: the file arms map the V4.1 first shard (tools/box.sh \
             exports it)"
        })?;
        let ctx = CudaContext::new(0)?;
        let stream = ctx.new_stream()?;
        let (host_register, read_only_register) = (
            attribute(
                &ctx,
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED,
            )?,
            attribute(
                &ctx,
                sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED,
            )?,
        );
        println!(
            "register card={} host_register={host_register} read_only_register={read_only_register} \
             shard={path} bytes={} offset={} reps={}",
            ctx.device_name()?,
            a.bytes,
            a.offset,
            a.reps
        );
        let dev = DeviceBuffer::<u8>::zeroed(&stream, a.bytes)?;
        let mut ok = true;

        let mut anon = MmapMut::map_anon(a.bytes).map_err(|e| format!("anonymous mmap: {e}"))?;
        let at = anon.as_mut_ptr().cast::<c_void>();
        ok &= arm(&ctx, &stream, &dev, "anon", at, a.bytes, 0, a.reps)?;
        drop(anon);

        let file = File::open(&path).map_err(|e| format!("{path}: {e}"))?;
        let len = file.metadata()?.len();
        if a.offset + a.bytes as u64 > len {
            return Err(format!(
                "{path} holds {len} bytes; the window is {} bytes from {}",
                a.bytes, a.offset
            )
            .into());
        }
        let mut opts = MmapOptions::new();
        opts.offset(a.offset).len(a.bytes);
        // SAFETY: a private mapping: writes through it (by this process or
        // the card) never reach the file, and nothing else writes the model
        // files while a probe runs.
        let mut window = unsafe { opts.map_copy(&file) }
            .map_err(|e| format!("MAP_PRIVATE RW mmap of {path}: {e}"))?;
        let at = window.as_mut_ptr().cast::<c_void>();
        ok &= arm(&ctx, &stream, &dev, "file_private", at, a.bytes, 0, a.reps)?;
        drop(window);
        // SAFETY: a read-only shared mapping of a file nothing writes while a
        // probe runs.
        let window =
            unsafe { opts.map(&file) }.map_err(|e| format!("MAP_SHARED RO mmap of {path}: {e}"))?;
        let at = window.as_ptr().cast_mut().cast::<c_void>();
        ok &= arm(&ctx, &stream, &dev, "file_shared", at, a.bytes, 0, a.reps)?;
        drop(window);
        // SAFETY: as the private window above; this mapping is read-only.
        let window = unsafe { opts.map_copy_read_only(&file) }
            .map_err(|e| format!("MAP_PRIVATE RO mmap of {path}: {e}"))?;
        let at = window.as_ptr().cast_mut().cast::<c_void>();
        let ro = sys::CU_MEMHOSTREGISTER_READ_ONLY;
        ok &= arm(
            &ctx,
            &stream,
            &dev,
            "file_private_ro",
            at,
            a.bytes,
            ro,
            a.reps,
        )?;
        drop(window);
        if !ok {
            return Err(bloomery_gpu_gates::checks_failed());
        }
        Ok(())
    }
}
