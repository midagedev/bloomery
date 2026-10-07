//! Host-to-device probe: copies host memory to the cards in view by the
//! arms `--arm` names (comma-separated, run in order; default
//! `pinned,pageable,registered`). Every arm first prints the card it copies
//! to, `h2d card ordinal=<i> name="<name>" bdf=<bdf> sysfs=<path>
//! max_link=<speed> x<width>`, the ordinal being the device's place in
//! `CUDA_VISIBLE_DEVICES` and `sysfs` the PCI device's real path, which
//! names the root port above it; every result line carries `card=` and
//! `bdf=`.
//!
//! - `pinned`: `--bytes` (1 GiB) from a page-locked allocation
//!   (`cuMemAllocHost`) to device 0, one untimed copy and `--reps` (8, at
//!   least five) timed ones, each between a pair of events:
//!   `h2d path=pinned … GB/s=<v>`, the bytes over the summed time.
//! - `pageable`: the same from the first `--bytes` of a read-only `mmap` of
//!   the V4.1 first shard (`gguf::v41::model`), every page read once first
//!   so the range is warm; plain copies (the driver stages them), each timed
//!   on the host clock.
//! - `registered`: the same range registered with `cuMemHostRegister` and
//!   `CU_MEMHOSTREGISTER_READ_ONLY` (the call timed on the host clock), async
//!   copies timed with events, then the unregister, timed. A card that
//!   refuses the registration ends the run with the driver's error.
//! - `sustained`: `--reps` (48) copies of `--bytes` (1,600,000,000, one
//!   Qwen3.8 layer-batch's routed bytes) from one pinned allocation to device
//!   0, enqueued back to back, each between its own pair of events:
//!   `h2d arm=sustained … GB/s_min= GB/s_median= GB/s_max= GB/s_mean=`, per
//!   copy, and the mean as all bytes over the summed time.
//! - `both`: `sustained` on every card in view at once, one thread and one
//!   pinned allocation per card, the loops released together. Per card the
//!   `sustained` line, then `h2d arm=both overlap_s=<w> sum_GB/s=<v>
//!   [<bdf>=<v> …]`: each card's rate between its first and last copy that
//!   completed inside the window where both loops ran (host clock, one
//!   origin), and their sum. Needs two cards in view.
//! - `attach`: `nvidia-smi topo -m`, each line as `h2d topo | <line>`, then
//!   every card in view's card line with its link as it reads idle.
//! - `pageable-loop`, `staged-loop`: the copy side of
//!   `tools/ref/dma-dram-share.sh`. They run alone, until `--stop-file`
//!   exists or `--seconds` pass (required, the bound), over a warm window of
//!   `--bytes` (1,600,000,000) of the V4.1 first shard's map, `--chunk` (64
//!   MiB) at a time, cycling. `pageable-loop` copies each chunk from the map
//!   (the driver stages it); `staged-loop` is the streaming design: a ring of
//!   `--ring` (4) pinned chunks, each filled from the map by `--fill-threads`
//!   (4) threads with plain `copy_from_slice` and then copied to the card
//!   while the next one fills, a slot refilled only after its copy's event.
//!   Every `--interval-ms` (1000): `h2d loop arm=<a> t_ms=<unix epoch ms>
//!   interval_ms=<ms> bytes=<b> GB/s=<v>`, bytes counted when their copy is
//!   known complete; then one `h2d loop arm=<a> summary …` line.
//!
//! Every copy arm samples each of its cards'
//! `current_link_speed`/`current_link_width` from sysfs every `SAMPLE_EVERY` while
//! its copies are in flight (an idle link trains down) and prints `h2d link
//! arm=<a> card= bdf= samples=<n> [<speed> x<width>: <count>, …]`.
//!
//! `GB/s` is 10⁹ bytes per second. A number printed here outside the lease
//! runners (`just time-gpu-h2d`, `just time-dma-dram`) is a runtime value of
//! whatever else the box was doing.

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
    use std::collections::BTreeMap;
    use std::ffi::c_void;
    use std::fs::File;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    use bloomery_gpu_gates::GateError;
    use cuda_core::{
        CudaContext, CudaEvent, CudaStream, DeviceBuffer, IntoResult, PinnedHostBuffer, sys,
    };
    use memmap2::Mmap;

    /// Bytes of the three one-shot paths unless `--bytes` says otherwise: 1 GiB.
    const PATH_BYTES: usize = 1 << 30;
    /// Timed copies of a one-shot path unless `--reps` says otherwise.
    const PATH_REPS: usize = 8;
    /// Bytes of a sustained copy and of a loop's source window: one Qwen3.8
    /// layer-batch's routed experts (a whole number of pages).
    const LAYER_BYTES: usize = 1_600_000_000;
    /// Back-to-back copies of the sustained arms.
    const SUSTAINED_REPS: usize = 48;
    /// The fewest timed copies an arm may take.
    const MIN_REPS: usize = 5;
    /// The page size: a mapped range starts on a page and spans whole pages.
    const PAGE: usize = 4096;
    /// A loop's chunk, its ring depth and its fill threads unless flags say otherwise.
    const CHUNK: usize = 64 << 20;
    const RING: usize = 4;
    const FILL_THREADS: usize = 4;
    const INTERVAL_MS: u64 = 1000;
    /// How often the link sampler reads sysfs.
    const SAMPLE_EVERY: Duration = Duration::from_millis(20);

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Arm {
        Pinned,
        Pageable,
        Registered,
        Sustained,
        Both,
        Attach,
        PageableLoop,
        StagedLoop,
    }

    impl Arm {
        const ALL: [(&'static str, Arm); 8] = [
            ("pinned", Arm::Pinned),
            ("pageable", Arm::Pageable),
            ("registered", Arm::Registered),
            ("sustained", Arm::Sustained),
            ("both", Arm::Both),
            ("attach", Arm::Attach),
            ("pageable-loop", Arm::PageableLoop),
            ("staged-loop", Arm::StagedLoop),
        ];

        fn parse(name: &str) -> Result<Arm, GateError> {
            Self::ALL
                .iter()
                .find(|(n, _)| *n == name)
                .map(|&(_, a)| a)
                .ok_or_else(|| {
                    let names: Vec<&str> = Self::ALL.iter().map(|(n, _)| *n).collect();
                    format!("unknown arm {name}; the arms are {}", names.join(", ")).into()
                })
        }

        fn name(self) -> &'static str {
            Self::ALL
                .iter()
                .find(|(_, a)| *a == self)
                .map(|(n, _)| *n)
                .expect("every arm is a row of ALL")
        }

        fn is_loop(self) -> bool {
            matches!(self, Arm::PageableLoop | Arm::StagedLoop)
        }

        fn default_bytes(self) -> usize {
            match self {
                Arm::Pinned | Arm::Pageable | Arm::Registered => PATH_BYTES,
                _ => LAYER_BYTES,
            }
        }

        fn default_reps(self) -> usize {
            match self {
                Arm::Sustained | Arm::Both => SUSTAINED_REPS,
                _ => PATH_REPS,
            }
        }
    }

    struct Args {
        arms: Vec<Arm>,
        bytes: Option<usize>,
        reps: Option<usize>,
        seconds: Option<f64>,
        stop_file: Option<PathBuf>,
        chunk_bytes: usize,
        ring: usize,
        fill_threads: usize,
        interval_ms: u64,
    }

    const USAGE: &str = "usage: h2d_probe [--arm A[,B…]] [--bytes N] [--reps K] [--seconds S] \
        [--stop-file PATH] [--chunk N] [--ring R] [--fill-threads T] [--interval-ms M]";

    fn args() -> Result<Args, GateError> {
        let mut a = Args {
            arms: vec![Arm::Pinned, Arm::Pageable, Arm::Registered],
            bytes: None,
            reps: None,
            seconds: None,
            stop_file: None,
            chunk_bytes: CHUNK,
            ring: RING,
            fill_threads: FILL_THREADS,
            interval_ms: INTERVAL_MS,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut value = |name: &str| -> Result<String, GateError> {
                it.next()
                    .ok_or_else(|| format!("{name} takes a value; {USAGE}").into())
            };
            let int = |name: &str, v: String| -> Result<usize, GateError> {
                v.parse()
                    .map_err(|e| format!("{name} {v}: {e}; {USAGE}").into())
            };
            match flag.as_str() {
                "--arm" => {
                    a.arms = value("--arm")?
                        .split(',')
                        .map(Arm::parse)
                        .collect::<Result<_, _>>()?;
                }
                "--bytes" => a.bytes = Some(int("--bytes", value("--bytes")?)?),
                "--reps" => a.reps = Some(int("--reps", value("--reps")?)?),
                "--seconds" => {
                    let v = value("--seconds")?;
                    let s: f64 = v
                        .parse()
                        .map_err(|e| format!("--seconds {v}: {e}; {USAGE}"))?;
                    if !(s.is_finite() && s > 0.0) {
                        return Err(format!("--seconds {v}: a positive number").into());
                    }
                    a.seconds = Some(s);
                }
                "--stop-file" => a.stop_file = Some(PathBuf::from(value("--stop-file")?)),
                "--chunk" => a.chunk_bytes = int("--chunk", value("--chunk")?)?,
                "--ring" => a.ring = int("--ring", value("--ring")?)?,
                "--fill-threads" => {
                    a.fill_threads = int("--fill-threads", value("--fill-threads")?)?
                }
                "--interval-ms" => {
                    let v = value("--interval-ms")?;
                    a.interval_ms = v
                        .parse()
                        .map_err(|e| format!("--interval-ms {v}: {e}; {USAGE}"))?;
                }
                other => return Err(format!("unknown argument {other}; {USAGE}").into()),
            }
        }
        check_args(&a)?;
        Ok(a)
    }

    fn check_args(a: &Args) -> Result<(), GateError> {
        let loops = a.arms.iter().filter(|x| x.is_loop()).count();
        if loops > 0 && a.arms.len() > 1 {
            return Err(
                "a loop arm (pageable-loop, staged-loop) runs alone: it copies until its stop \
                        file or its seconds, beside another process"
                    .into(),
            );
        }
        if loops > 0 && a.seconds.is_none() {
            return Err(
                "a loop arm needs --seconds, the bound it stops at without its stop file".into(),
            );
        }
        if loops == 0 && (a.seconds.is_some() || a.stop_file.is_some()) {
            return Err("--seconds and --stop-file belong to the loop arms".into());
        }
        for &arm in &a.arms {
            let bytes = a.bytes.unwrap_or(arm.default_bytes());
            let reps = a.reps.unwrap_or(arm.default_reps());
            if bytes == 0 || !bytes.is_multiple_of(PAGE) {
                return Err(format!(
                    "arm {}: --bytes {bytes} is not a positive multiple of {PAGE}",
                    arm.name()
                )
                .into());
            }
            if !arm.is_loop() && arm != Arm::Attach && reps < MIN_REPS {
                return Err(
                    format!("arm {}: --reps {reps}, at least {MIN_REPS}", arm.name()).into(),
                );
            }
            if arm.is_loop()
                && (a.chunk_bytes == 0
                    || !a.chunk_bytes.is_multiple_of(PAGE)
                    || a.chunk_bytes > bytes
                    || a.ring < 2
                    || a.fill_threads == 0
                    || a.interval_ms == 0)
            {
                return Err(format!(
                    "arm {}: --chunk {} a positive multiple of {PAGE} within --bytes {bytes}, --ring {} \
                     at least 2, --fill-threads {} and --interval-ms {} at least 1",
                    arm.name(),
                    a.chunk_bytes,
                    a.ring,
                    a.fill_threads,
                    a.interval_ms
                )
                .into());
            }
        }
        Ok(())
    }

    /// Device attribute `attr` of device `dev`.
    fn attribute(dev: sys::CUdevice, attr: sys::CUdevice_attribute) -> Result<i32, GateError> {
        let mut v = 0;
        // SAFETY: the call writes the live local `v`; `dev` is a device the
        // driver handed out.
        let rc = unsafe { sys::cuDeviceGetAttribute(&mut v, attr, dev) };
        rc.result()
            .map_err(|e| format!("cuDeviceGetAttribute({attr}): {e:?}"))?;
        Ok(v)
    }

    /// Cards in view (`CUDA_VISIBLE_DEVICES`). The first driver call of a
    /// process that has made no context yet, so it initializes the driver.
    fn device_count() -> Result<usize, GateError> {
        // SAFETY: flags 0 is the only value the driver takes; initializing
        // twice is a no-op.
        unsafe { cuda_core::init(0) }.map_err(|e| format!("cuInit: {e:?}"))?;
        let mut n = 0;
        // SAFETY: the call writes the live local `n`; the driver is
        // initialized above.
        let rc = unsafe { sys::cuDeviceGetCount(&mut n) };
        rc.result()
            .map_err(|e| format!("cuDeviceGetCount: {e:?}"))?;
        Ok(usize::try_from(n)?)
    }

    /// A card as the result lines name it.
    struct Card {
        ordinal: usize,
        name: String,
        bdf: String,
        sysfs: PathBuf,
    }

    impl Card {
        fn of(ctx: &CudaContext) -> Result<Card, GateError> {
            let dev = ctx.cu_device();
            let pci =
                |attr| -> Result<u32, GateError> { Ok(u32::try_from(attribute(dev, attr)?)?) };
            let bdf = format!(
                "{:04x}:{:02x}:{:02x}.0",
                pci(sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)?,
                pci(sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)?,
                pci(sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)?
            );
            let dir = PathBuf::from(format!("/sys/bus/pci/devices/{bdf}"));
            let sysfs = std::fs::canonicalize(&dir)
                .map_err(|e| format!("{}: {e} (the card's PCI device in sysfs)", dir.display()))?;
            Ok(Card {
                ordinal: ctx.ordinal(),
                name: ctx.device_name()?,
                bdf,
                sysfs,
            })
        }

        fn read(&self, file: &str) -> Result<String, GateError> {
            let p = self.sysfs.join(file);
            Ok(std::fs::read_to_string(&p)
                .map_err(|e| format!("{}: {e}", p.display()))?
                .trim()
                .to_string())
        }

        /// The `card=… bdf=…` pair every result line carries.
        fn tag(&self) -> String {
            format!("card=\"{}\" bdf={}", self.name, self.bdf)
        }

        fn print(&self) -> Result<(), GateError> {
            println!(
                "h2d card ordinal={} name=\"{}\" bdf={} sysfs={} max_link={} x{} link_now={} x{}",
                self.ordinal,
                self.name,
                self.bdf,
                self.sysfs.display(),
                self.read("max_link_speed")?,
                self.read("max_link_width")?,
                self.read("current_link_speed")?,
                self.read("current_link_width")?
            );
            Ok(())
        }
    }

    /// Reads a card's current link speed and width every [`SAMPLE_EVERY`]
    /// until stopped: what the link runs at while copies are in flight.
    struct Sampler {
        stop: Arc<AtomicBool>,
        handle: JoinHandle<Result<BTreeMap<String, usize>, String>>,
    }

    impl Sampler {
        fn start(card: &Card) -> Sampler {
            let stop = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&stop);
            let (speed, width) = (
                card.sysfs.join("current_link_speed"),
                card.sysfs.join("current_link_width"),
            );
            let handle = std::thread::spawn(move || {
                let mut seen = BTreeMap::new();
                let read = |p: &Path| {
                    std::fs::read_to_string(p)
                        .map(|s| s.trim().to_string())
                        .map_err(|e| format!("{}: {e}", p.display()))
                };
                while !flag.load(Ordering::Relaxed) {
                    let key = format!("{} x{}", read(&speed)?, read(&width)?);
                    *seen.entry(key).or_insert(0) += 1;
                    std::thread::sleep(SAMPLE_EVERY);
                }
                Ok(seen)
            });
            Sampler { stop, handle }
        }

        fn finish(self, arm: Arm, card: &Card) -> Result<(), GateError> {
            self.stop.store(true, Ordering::Relaxed);
            let seen = self
                .handle
                .join()
                .map_err(|_| "the link sampler thread panicked")??;
            let n: usize = seen.values().sum();
            if n == 0 {
                return Err(format!(
                    "arm {}: no link sample of {} while its copies ran",
                    arm.name(),
                    card.bdf
                )
                .into());
            }
            let cells: Vec<String> = seen.iter().map(|(k, c)| format!("{k}: {c}")).collect();
            println!(
                "h2d link arm={} {} samples={n} [{}]",
                arm.name(),
                card.tag(),
                cells.join(", ")
            );
            Ok(())
        }
    }

    /// `bytes · reps` over `ms` milliseconds, in GB/s.
    fn gbps(bytes: usize, reps: usize, ms: f64) -> f64 {
        (bytes * reps) as f64 / (ms * 1e-3) / 1e9
    }

    fn timing_event(ctx: &Arc<CudaContext>) -> Result<CudaEvent, GateError> {
        Ok(ctx.new_event(Some(sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?)
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
        let (start, end) = (timing_event(ctx)?, timing_event(ctx)?);
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

    /// The first `bytes` of the V4.1 first shard's read-only map, every page
    /// read once so the range is resident.
    struct Source {
        path: String,
        map: Mmap,
        bytes: usize,
    }

    impl Source {
        fn open(bytes: usize) -> Result<Source, GateError> {
            let path = gguf::v41::model();
            let file = File::open(&path).map_err(|e| format!("{path}: {e}"))?;
            // SAFETY: the shard is opened read-only and nothing writes the
            // model files while a probe or a gate runs; the map is only read,
            // here and by the driver's copies.
            let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {path}: {e}"))?;
            if map.len() < bytes {
                return Err(
                    format!("{path} holds {} bytes, the probe copies {bytes}", map.len()).into(),
                );
            }
            let warm = map[..bytes]
                .iter()
                .step_by(PAGE)
                .fold(0u64, |s, &b| s.wrapping_add(u64::from(b)));
            std::hint::black_box(warm);
            Ok(Source { path, map, bytes })
        }

        fn window(&self) -> &[u8] {
            &self.map[..self.bytes]
        }
    }

    fn one_shot(
        ctx: &Arc<CudaContext>,
        first: &Arc<CudaStream>,
        card: &Card,
        arm: Arm,
        a: &Args,
    ) -> Result<(), GateError> {
        let stream = Arc::clone(first);
        let bytes = a.bytes.unwrap_or(arm.default_bytes());
        let reps = a.reps.unwrap_or(arm.default_reps());
        let mut dev = DeviceBuffer::<u8>::zeroed(&stream, bytes)?;
        let (line, sampler) = match arm {
            Arm::Pinned => {
                let pinned = PinnedHostBuffer::<u8>::zeroed(ctx, bytes)?;
                let sampler = Sampler::start(card);
                let ms = timed_async(ctx, &stream, reps, || {
                    // SAFETY: `pinned` outlives every copy: `timed_async`
                    // synchronizes the stream after each, and `pinned` drops
                    // after it returns.
                    unsafe { dev.copy_from_pinned_host_async(&stream, &pinned)? };
                    Ok(())
                })?;
                (format!("GB/s={:.2}", gbps(bytes, reps, ms)), sampler)
            }
            Arm::Pageable => {
                let src = Source::open(bytes)?;
                let sampler = Sampler::start(card);
                dev.copy_from_host(&stream, src.window())?;
                let mut ms = 0.0f64;
                for _ in 0..reps {
                    let t = Instant::now();
                    dev.copy_from_host(&stream, src.window())?;
                    ms += t.elapsed().as_secs_f64() * 1e3;
                }
                (
                    format!("GB/s={:.2} shard={}", gbps(bytes, reps, ms), src.path),
                    sampler,
                )
            }
            Arm::Registered => {
                let src = Source::open(bytes)?;
                let sampler = Sampler::start(card);
                (registered(ctx, &stream, &mut dev, &src, reps)?, sampler)
            }
            _ => unreachable!("one_shot runs the three paths only"),
        };
        sampler.finish(arm, card)?;
        println!(
            "h2d path={} {} bytes={bytes} reps={reps} {line}",
            arm.name(),
            card.tag()
        );
        Ok(())
    }

    fn registered(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        dev: &mut DeviceBuffer<u8>,
        src: &Source,
        reps: usize,
    ) -> Result<String, GateError> {
        let bytes = src.bytes;
        let dev_id = ctx.cu_device();
        let host_register = attribute(
            dev_id,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED,
        )?;
        let read_only = attribute(
            dev_id,
            sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED,
        )?;
        let at = src.window().as_ptr().cast_mut().cast::<c_void>();
        ctx.bind_to_thread()?;
        let t = Instant::now();
        // SAFETY: `at .. at + bytes` is the live read-only map's first
        // `bytes` bytes, page-aligned (a map starts on a page) and whole
        // pages; READ_ONLY registers it without asking for write access, and
        // the unregister below runs before the map drops.
        let rc = unsafe { sys::cuMemHostRegister_v2(at, bytes, sys::CU_MEMHOSTREGISTER_READ_ONLY) };
        let register_ms = t.elapsed().as_secs_f64() * 1e3;
        rc.result().map_err(|e| {
            format!(
                "cuMemHostRegister(READ_ONLY) of {}'s map: {e:?} (the card reports \
                 host_register={host_register} read_only_register={read_only})",
                src.path
            )
        })?;
        let copies = timed_async(ctx, stream, reps, || {
            // SAFETY: the window is the registered map, which outlives every
            // copy: `timed_async` synchronizes the stream after each.
            unsafe { dev.copy_from_host_async_unchecked(stream, src.window())? };
            Ok(())
        });
        stream.synchronize()?;
        let t = Instant::now();
        // SAFETY: `at` is the base the register call above took, and no copy
        // from it is in flight: the stream is synchronized.
        let rc = unsafe { sys::cuMemHostUnregister(at) };
        let unregister_ms = t.elapsed().as_secs_f64() * 1e3;
        rc.result()
            .map_err(|e| format!("cuMemHostUnregister of {}'s map: {e:?}", src.path))?;
        Ok(format!(
            "GB/s={:.2} register_ms={register_ms:.1} unregister_ms={unregister_ms:.1}",
            gbps(bytes, reps, copies?)
        ))
    }

    /// One card's sustained loop: per copy its device milliseconds and the
    /// host time its completion was seen, in seconds from `origin`.
    struct Loop {
        card: Card,
        ms: Vec<f64>,
        done_s: Vec<f64>,
    }

    fn sustained_loop(
        ordinal: usize,
        bytes: usize,
        reps: usize,
        start: &Barrier,
        origin: Instant,
    ) -> Result<Loop, GateError> {
        let (ctx, stream) = bloomery_gpu::capsync::fresh_stream(ordinal)?;
        let card = Card::of(&ctx)?;
        card.print()?;
        let pinned = PinnedHostBuffer::<u8>::zeroed(&ctx, bytes)?;
        let mut dev = DeviceBuffer::<u8>::zeroed(&stream, bytes)?;
        // SAFETY: `pinned` outlives the copy: the stream is synchronized on
        // the next line.
        unsafe { dev.copy_from_pinned_host_async(&stream, &pinned)? };
        stream.synchronize()?;
        let events: Vec<(CudaEvent, CudaEvent)> = (0..reps)
            .map(|_| Ok((timing_event(&ctx)?, timing_event(&ctx)?)))
            .collect::<Result<_, GateError>>()?;
        start.wait();
        let sampler = Sampler::start(&card);
        for (a, b) in &events {
            a.record(&stream)?;
            // SAFETY: `pinned` outlives every copy: each end event is
            // synchronized below before `pinned` drops.
            unsafe { dev.copy_from_pinned_host_async(&stream, &pinned)? };
            b.record(&stream)?;
        }
        let mut ms = Vec::with_capacity(reps);
        let mut done_s = Vec::with_capacity(reps);
        for (a, b) in &events {
            b.synchronize()?;
            done_s.push(origin.elapsed().as_secs_f64());
            ms.push(f64::from(a.elapsed_ms(b)?));
        }
        sampler.finish(Arm::Sustained, &card)?;
        Ok(Loop { card, ms, done_s })
    }

    fn print_sustained(arm: Arm, l: &Loop, bytes: usize) {
        let mut sorted = l.ms.clone();
        sorted.sort_by(f64::total_cmp);
        let per = |ms: f64| gbps(bytes, 1, ms);
        let n = sorted.len();
        println!(
            "h2d arm={} {} bytes={bytes} reps={n} GB/s_min={:.2} GB/s_median={:.2} GB/s_max={:.2} \
             GB/s_mean={:.2}",
            arm.name(),
            l.card.tag(),
            per(sorted[n - 1]),
            per(sorted[n / 2]),
            per(sorted[0]),
            gbps(bytes, n, l.ms.iter().sum())
        );
    }

    fn sustained(a: &Args, arm: Arm) -> Result<(), GateError> {
        let bytes = a.bytes.unwrap_or(arm.default_bytes());
        let reps = a.reps.unwrap_or(arm.default_reps());
        let cards = if arm == Arm::Both {
            let n = device_count()?;
            if n < 2 {
                return Err(format!(
                    "arm both needs two cards in view; CUDA_VISIBLE_DEVICES shows {n} \
                     (run it under BLOOMERY_CARD=both through tools/ref/h2d-pcie.sh)"
                )
                .into());
            }
            n
        } else {
            1
        };
        let start = Barrier::new(cards);
        let origin = Instant::now();
        let loops: Vec<Loop> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..cards)
                .map(|i| {
                    let start = &start;
                    // The thread hands back its error as text: `GateError` is not `Send`.
                    s.spawn(move || {
                        sustained_loop(i, bytes, reps, start, origin).map_err(|e| e.to_string())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| match h.join() {
                    Ok(r) => r.map_err(GateError::from),
                    Err(_) => Err(GateError::from("a sustained loop's thread panicked")),
                })
                .collect::<Result<_, GateError>>()
        })?;
        for l in &loops {
            print_sustained(arm, l, bytes);
        }
        if arm == Arm::Both {
            overlap(&loops, bytes)?;
        }
        Ok(())
    }

    /// Each card's rate between its first and last completion inside the
    /// window where every loop ran, and their sum.
    fn overlap(loops: &[Loop], bytes: usize) -> Result<(), GateError> {
        let w0 = loops.iter().map(|l| l.done_s[0]).fold(f64::MIN, f64::max);
        let w1 = loops
            .iter()
            .map(|l| l.done_s[l.done_s.len() - 1])
            .fold(f64::MAX, f64::min);
        let mut cells = Vec::new();
        let mut sum = 0.0;
        for l in loops {
            let inside: Vec<f64> = l
                .done_s
                .iter()
                .copied()
                .filter(|&t| t >= w0 && t <= w1)
                .collect();
            if inside.len() < 2 {
                return Err(format!(
                    "arm both: {} completed {} copies inside the window where every loop ran \
                     ({w0:.3}..{w1:.3} s): the loops did not overlap",
                    l.card.bdf,
                    inside.len()
                )
                .into());
            }
            let rate = gbps(
                bytes,
                inside.len() - 1,
                (inside[inside.len() - 1] - inside[0]) * 1e3,
            );
            sum += rate;
            cells.push(format!("{}={rate:.2}", l.card.bdf));
        }
        println!(
            "h2d arm=both overlap_s={:.3} sum_GB/s={sum:.2} [{}]",
            w1 - w0,
            cells.join(" ")
        );
        Ok(())
    }

    fn attach() -> Result<(), GateError> {
        let out = std::process::Command::new("nvidia-smi")
            .args(["topo", "-m"])
            .output()
            .map_err(|e| format!("nvidia-smi topo -m: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "nvidia-smi topo -m: {} {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            )
            .into());
        }
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            println!("h2d topo | {line}");
        }
        for i in 0..device_count()? {
            let ctx = bloomery_gpu::capsync::fresh_context(i)?;
            Card::of(&ctx)?.print()?;
        }
        Ok(())
    }

    fn epoch_ms() -> Result<u128, GateError> {
        Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())
    }

    /// The per-interval and summary lines of a loop arm.
    struct Meter {
        arm: Arm,
        started: Instant,
        mark: Instant,
        mark_bytes: usize,
        bytes: usize,
        interval: Duration,
    }

    impl Meter {
        fn new(arm: Arm, interval_ms: u64) -> Meter {
            let now = Instant::now();
            Meter {
                arm,
                started: now,
                mark: now,
                mark_bytes: 0,
                bytes: 0,
                interval: Duration::from_millis(interval_ms),
            }
        }

        fn done(&mut self, bytes: usize) -> Result<(), GateError> {
            self.bytes += bytes;
            if self.mark.elapsed() >= self.interval {
                self.emit()?;
            }
            Ok(())
        }

        /// The interval since the last line, however short: called once when the loop stops, so the
        /// intervals cover the copy to its end (dma-dram-share tags a span they leave open [copy-gap]).
        fn flush(&mut self) -> Result<(), GateError> {
            if self.bytes > self.mark_bytes {
                self.emit()?;
            }
            Ok(())
        }

        fn emit(&mut self) -> Result<(), GateError> {
            let dt = self.mark.elapsed();
            let b = self.bytes - self.mark_bytes;
            println!(
                "h2d loop arm={} t_ms={} interval_ms={:.1} bytes={b} GB/s={:.2}",
                self.arm.name(),
                epoch_ms()?,
                dt.as_secs_f64() * 1e3,
                b as f64 / dt.as_secs_f64() / 1e9
            );
            self.mark = Instant::now();
            self.mark_bytes = self.bytes;
            Ok(())
        }
    }

    fn stop(a: &Args, started: Instant, seconds: f64) -> Option<&'static str> {
        if a.stop_file.as_deref().is_some_and(Path::exists) {
            Some("stop-file")
        } else if started.elapsed().as_secs_f64() >= seconds {
            Some("seconds")
        } else {
            None
        }
    }

    fn copy_loop(
        ctx: &Arc<CudaContext>,
        first: &Arc<CudaStream>,
        card: &Card,
        arm: Arm,
        a: &Args,
    ) -> Result<(), GateError> {
        let seconds = a
            .seconds
            .expect("check_args requires --seconds for a loop arm");
        let bytes = a.bytes.unwrap_or(arm.default_bytes());
        let chunk = a.chunk_bytes;
        let src = Source::open(bytes)?;
        let offsets: Vec<usize> = (0..bytes / chunk).map(|i| i * chunk).collect();
        let stream = Arc::clone(first);
        let mut dev = DeviceBuffer::<u8>::zeroed(&stream, chunk)?;
        let window = src.window();
        let sampler = Sampler::start(card);
        let mut meter = Meter::new(arm, a.interval_ms);
        println!(
            "h2d loop arm={} {} start t_ms={} window={bytes} chunk={chunk} ring={} fill_threads={} \
             fill_bytes_per_thread={} shard={}",
            arm.name(),
            card.tag(),
            epoch_ms()?,
            a.ring,
            a.fill_threads,
            chunk.div_ceil(a.fill_threads),
            src.path
        );
        let why = if arm == Arm::PageableLoop {
            let mut i = 0;
            loop {
                if let Some(why) = stop(a, meter.started, seconds) {
                    break why;
                }
                let off = offsets[i % offsets.len()];
                dev.copy_from_host(&stream, &window[off..off + chunk])?;
                meter.done(chunk)?;
                i += 1;
            }
        } else {
            staged(ctx, &stream, &mut dev, window, &offsets, a, &mut meter)?
        };
        meter.flush()?;
        sampler.finish(arm, card)?;
        let s = meter.started.elapsed().as_secs_f64();
        println!(
            "h2d loop arm={} {} summary t_ms={} seconds={s:.2} bytes={} GB/s={:.2} stopped={why}",
            arm.name(),
            card.tag(),
            epoch_ms()?,
            meter.bytes,
            meter.bytes as f64 / s / 1e9
        );
        Ok(())
    }

    /// The streaming design's copy: a ring of pinned chunks, each filled by
    /// the fill threads and copied to the card while the next fills.
    fn staged(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        dev: &mut DeviceBuffer<u8>,
        window: &[u8],
        offsets: &[usize],
        a: &Args,
        meter: &mut Meter,
    ) -> Result<&'static str, GateError> {
        let seconds = a
            .seconds
            .expect("check_args requires --seconds for a loop arm");
        let chunk = a.chunk_bytes;
        let per_thread = chunk.div_ceil(a.fill_threads);
        let mut ring: Vec<PinnedHostBuffer<u8>> = (0..a.ring)
            .map(|_| PinnedHostBuffer::<u8>::zeroed(ctx, chunk))
            .collect::<Result<_, _>>()?;
        let events: Vec<CudaEvent> = (0..a.ring)
            .map(|_| ctx.new_event(None))
            .collect::<Result<_, _>>()?;
        let mut i = 0;
        let why = loop {
            if let Some(why) = stop(a, meter.started, seconds) {
                break why;
            }
            let slot = i % a.ring;
            if i >= a.ring {
                events[slot].synchronize()?;
                meter.done(chunk)?;
            }
            let off = offsets[i % offsets.len()];
            let from = &window[off..off + chunk];
            std::thread::scope(|s| {
                for (dst, src) in ring[slot]
                    .as_mut_slice()
                    .chunks_mut(per_thread)
                    .zip(from.chunks(per_thread))
                {
                    s.spawn(move || dst.copy_from_slice(src));
                }
            });
            // SAFETY: `ring[slot]` is not written again until `events[slot]`,
            // recorded after this copy, has been synchronized, and the ring
            // outlives the stream's last copy (synchronized below).
            unsafe { dev.copy_from_pinned_host_async(stream, &ring[slot])? };
            events[slot].record(stream)?;
            i += 1;
        };
        stream.synchronize()?;
        meter.done(chunk * i.min(a.ring))?;
        Ok(why)
    }

    pub fn run() -> Result<(), GateError> {
        let a = args()?;
        for &arm in &a.arms {
            match arm {
                Arm::Attach => attach()?,
                Arm::Sustained | Arm::Both => sustained(&a, arm)?,
                _ => {
                    // The handle and its first stream (whose creation runs
                    // the context-wide synchronize) come from the
                    // capture-safe owner; the arm below reuses that stream.
                    let (ctx, first) = bloomery_gpu::capsync::fresh_stream(0)?;
                    let card = Card::of(&ctx)?;
                    card.print()?;
                    if arm.is_loop() {
                        copy_loop(&ctx, &first, &card, arm, &a)?;
                    } else {
                        one_shot(&ctx, &first, &card, arm, &a)?;
                    }
                }
            }
        }
        Ok(())
    }
}
