//! Host-registration probe: how long `cuMemHostRegister` takes to pin host
//! memory for DMA on this box, for the two kinds a registered expert arena
//! could live in — anonymous memory and a `/dev/shm` file mapped shared (the
//! exllamav3 #341 arena) — at flags 0 and `PORTABLE`, and whether the
//! registration gives every card in view a device address for the range.
//!
//! Per arm: `--chunks` (8) chunks of `--chunk-mib` (1,024) MiB, each touched
//! page by page before anything is registered, so the time is the pinning
//! alone, not the zero fill; the chunks registered one by one, the calls
//! timed together on the host clock; then, in every context in view,
//! `cuMemHostGetDevicePointer` on each chunk (a portable registration maps in
//! all of them, a flags-0 one in the registering card's alone); then the
//! unregisters, timed. A driver refusal ends that arm with its code
//! (`CUDA_ERROR_NOT_SUPPORTED`, 801, is an allowed outcome); the arm's record
//! then has no times. The registering context is card 0's.
//!
//! `probe_hostreg --check`: one 64 MiB chunk of each kind registered
//! `PORTABLE`, copied to card 0 and back, and the bytes compared; red on a
//! mismatch or a refusal. `--time` (under `tools/ref/probe-twocard.sh`: the
//! lease, both cards in view): the check, then the four arms. Every line is a
//! record kind of `bloomery_gpu_gates::record` (`--records-schema`).

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("probe_hostreg: built without the `gpu` feature; see `just probe-gpu-hostreg`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("probe_hostreg", probe::run())
}

#[cfg(feature = "gpu")]
mod probe {
    use std::ffi::c_void;
    use std::sync::Arc;
    use std::time::Instant;

    use bloomery_gpu_gates::record::{self, Record};
    use bloomery_gpu_gates::{GateError, checks_failed};
    use cuda_core::{CudaContext, DeviceBuffer, sys};
    use memmap2::{MmapMut, MmapOptions};

    const MIB: usize = 1 << 20;
    const GIB: f64 = (1u64 << 30) as f64;
    /// The check's chunk.
    const CHECK_BYTES: usize = 64 * MIB;

    #[derive(Clone, Copy)]
    enum Mem {
        Anon,
        Shm,
    }

    impl Mem {
        fn name(self) -> &'static str {
            match self {
                Mem::Anon => "anon",
                Mem::Shm => "shm",
            }
        }
    }

    /// One chunk: its mapping, and for shm the file's path, removed on drop.
    struct Chunk {
        map: MmapMut,
        shm: Option<std::path::PathBuf>,
    }

    impl Drop for Chunk {
        fn drop(&mut self) {
            if let Some(p) = &self.shm {
                let _ = std::fs::remove_file(p);
            }
        }
    }

    impl Chunk {
        /// A chunk of `bytes` of kind `mem`, every page written once.
        fn new(mem: Mem, bytes: usize, tag: usize) -> Result<Chunk, GateError> {
            let (mut map, shm) = match mem {
                Mem::Anon => (MmapOptions::new().len(bytes).map_anon()?, None),
                Mem::Shm => {
                    let path = std::path::PathBuf::from(format!(
                        "/dev/shm/bloomery-probe-hostreg-{}-{tag}",
                        std::process::id()
                    ));
                    let f = std::fs::OpenOptions::new()
                        .read(true)
                        .write(true)
                        .create_new(true)
                        .open(&path)?;
                    f.set_len(bytes as u64)?;
                    // SAFETY: the file is this process's own, created above
                    // and removed when the chunk drops; nothing else maps it.
                    let map = unsafe { MmapMut::map_mut(&f)? };
                    (map, Some(path))
                }
            };
            for (i, page) in map.chunks_mut(4096).enumerate() {
                page[0] = (i ^ tag) as u8;
            }
            Ok(Chunk { map, shm })
        }

        fn at(&mut self) -> *mut c_void {
            self.map.as_mut_ptr().cast()
        }
    }

    fn ok(rc: sys::CUresult, what: &str) -> Result<(), GateError> {
        if rc == sys::cudaError_enum_CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("{what}: {:?}", cuda_core::DriverError(rc)).into())
        }
    }

    struct Args {
        chunks: usize,
        chunk: usize,
        time: bool,
    }

    fn args() -> Result<Args, GateError> {
        let mut a = Args {
            chunks: 8,
            chunk: 1024 * MIB,
            time: false,
        };
        let mut mode = None;
        let mut it = std::env::args().skip(1);
        while let Some(x) = it.next() {
            match x.as_str() {
                "--check" | "--time" => mode = Some(x == "--time"),
                "--chunks" | "--chunk-mib" => {
                    let v: usize = it
                        .next()
                        .ok_or_else(|| format!("{x} wants a value"))?
                        .parse()
                        .map_err(|e| format!("{x}: {e}"))?;
                    if v == 0 {
                        return Err(format!("{x} wants at least 1").into());
                    }
                    if x == "--chunks" {
                        a.chunks = v;
                    } else {
                        a.chunk = v * MIB;
                    }
                }
                other => return Err(format!("unknown argument '{other}'").into()),
            }
        }
        a.time = mode.ok_or("want one of --check, --time")?;
        Ok(a)
    }

    /// One arm: register, map in each context, unregister; its record.
    fn arm(
        ctxs: &[Arc<CudaContext>],
        mem: Mem,
        portable: bool,
        chunks: usize,
        bytes: usize,
    ) -> Result<(), GateError> {
        let mut held: Vec<Chunk> = (0..chunks)
            .map(|i| Chunk::new(mem, bytes, i))
            .collect::<Result<_, _>>()?;
        let flags = if portable {
            sys::CU_MEMHOSTREGISTER_PORTABLE
        } else {
            0
        };
        ctxs[0].bind_to_thread()?;
        let mut rec = Record::new(&record::PROBE_HOSTREG)
            .w("mem", mem.name())
            .w("flags", if portable { "portable" } else { "0" })
            .u("chunk", bytes)
            .u("chunks", chunks)
            .u("cards", ctxs.len());
        let t0 = Instant::now();
        let mut done = 0;
        let mut refused = 0;
        for c in &mut held {
            // SAFETY: `c`'s mapping is `bytes` bytes, page-aligned, live until
            // after the unregister below.
            let rc = unsafe { sys::cuMemHostRegister_v2(c.at(), bytes, flags) };
            if rc != sys::cudaError_enum_CUDA_SUCCESS {
                refused = rc;
                break;
            }
            done += 1;
        }
        let register_s = t0.elapsed().as_secs_f64();
        let mut mapped = 0;
        if refused == 0 {
            for ctx in ctxs {
                ctx.bind_to_thread()?;
                let mut all = true;
                for c in &mut held {
                    let mut dev: sys::CUdeviceptr = 0;
                    // SAFETY: `c`'s range is registered (above) and live; the
                    // flags must be 0.
                    let rc = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, c.at(), 0) };
                    all &= rc == sys::cudaError_enum_CUDA_SUCCESS;
                }
                mapped += usize::from(all);
            }
        }
        ctxs[0].bind_to_thread()?;
        let t1 = Instant::now();
        for c in held.iter_mut().take(done) {
            // SAFETY: the base the register call above took; no copy is in
            // flight from it.
            let rc = unsafe { sys::cuMemHostUnregister(c.at()) };
            ok(rc, "cuMemHostUnregister")?;
        }
        let unregister_s = t1.elapsed().as_secs_f64();
        rec = rec.u("rc", refused);
        if refused == 0 {
            rec = rec
                .f("register_s", register_s)
                .f("s_per_gib", register_s / (chunks * bytes) as f64 * GIB)
                .f("unregister_s", unregister_s)
                .u("mapped_cards", mapped);
        }
        rec.print();
        Ok(())
    }

    /// The check: one PORTABLE chunk of `mem`, copied to card 0 and back.
    fn check(ctx: &Arc<CudaContext>, mem: Mem) -> Result<bool, GateError> {
        let mut c = Chunk::new(mem, CHECK_BYTES, 7)?;
        ctx.bind_to_thread()?;
        // SAFETY: the chunk's mapping, live until after the unregister below.
        let rc = unsafe {
            sys::cuMemHostRegister_v2(c.at(), CHECK_BYTES, sys::CU_MEMHOSTREGISTER_PORTABLE)
        };
        let reg = rc;
        let mut same = false;
        if reg == 0 {
            let stream = ctx.new_stream()?;
            let dev = DeviceBuffer::<u8>::from_host(&stream, &c.map[..])?;
            let back = dev.to_host_vec(&stream)?;
            same = back[..] == c.map[..];
            // SAFETY: the base the register call took; the copies are done
            // (to_host_vec synchronizes).
            let rc = unsafe { sys::cuMemHostUnregister(c.at()) };
            ok(rc, "cuMemHostUnregister")?;
        }
        let pass = reg == 0 && same;
        Record::new(&record::PROBE_CHECK)
            .w("probe", "probe_hostreg")
            .w("case", mem.name())
            .w(
                "saw",
                format!("register_rc={reg} round_trip_equal={same} bytes={CHECK_BYTES}"),
            )
            .w("pass", pass)
            .print();
        Ok(pass)
    }

    pub fn run() -> Result<(), GateError> {
        record::at_main("probe_hostreg", record::PROBE_HOSTREG_BIN);
        let a = args()?;
        let n = usize::try_from(cuda_core::Device::device_count()?)
            .map_err(|_| "a negative device count")?;
        let ctxs: Vec<Arc<CudaContext>> = (0..n).map(CudaContext::new).collect::<Result<_, _>>()?;
        if ctxs.is_empty() {
            return Err("no CUDA device in view".into());
        }
        let mut pass = true;
        for mem in [Mem::Anon, Mem::Shm] {
            pass &= check(&ctxs[0], mem)?;
        }
        Record::new(&record::PROBE_CHECK)
            .w("probe", "probe_hostreg")
            .w("case", "all")
            .w("saw", format!("{n} cards in view"))
            .w("pass", pass)
            .print();
        if !pass {
            return Err(checks_failed());
        }
        if !a.time {
            return Ok(());
        }
        for mem in [Mem::Anon, Mem::Shm] {
            for portable in [false, true] {
                arm(&ctxs, mem, portable, a.chunks, a.chunk)?;
            }
        }
        Ok(())
    }
}
