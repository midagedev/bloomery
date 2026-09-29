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
//! - `backleg-host`, `backleg-dev`, `backleg-host+chunks`,
//!   `backleg-dev+chunks`: the decode step's back-leg as a per-round host
//!   clock round trip, `--iters` (10,000) rounds after `--warm` (200)
//!   untimed, one card (device 0). Per round i the host sends its sum — the
//!   20,480 bytes (5120 f32) a routed layer's handoff carries, every value
//!   `i as f32` — plus a go counter, the card waits for the go, checks every
//!   value on the card and answers one ack word the host spins on; the arm
//!   prints `h2d arm=<a> card=… bdf=… iters=N warm=W bytes=B
//!   rt_us_median= rt_us_p90= rt_us_p99= rt_us_max=` (the bump-to-ack
//!   interval), the dev arms also `submit_us_median= submit_us_p99=` (the
//!   copy call's own host time), the +chunks arms also `chunks_GB/s=`.
//!   `backleg-host` is today's protocol: the sum lives in a host-mapped
//!   page, the host `fetch_add`s a mapped counter and the engine stream's
//!   `wait_geq` releases on it, the kernel reading the sum through the
//!   mapping. `backleg-dev` is the device-resident candidate: the host
//!   writes the sum plus the counter as the last word into a pinned buffer
//!   and issues one `cuMemcpyHtoDAsync` of the 20,484 bytes into VRAM on a
//!   copy stream; the engine stream waits on the VRAM counter word and the
//!   kernel reads the sum from VRAM — whether that wait releases only once
//!   the whole copy has landed is what the arm's on-card check tests, and a
//!   stale read ends the arm with a named error (that failure is a finding,
//!   not a broken run). The `+chunks` arms run a third thread looping 2 MiB
//!   pinned→VRAM copies on their own stream for the arm's duration — the
//!   paced-copy load a streaming back-leg design puts on the link. All the
//!   stream work is enqueued by a second thread ahead of the driving one, so
//!   the driver never blocks on a full queue and a round's interval measures
//!   the crossing, not an enqueue.
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

    use bloomery_gpu::probe::{BacklegPage, enqueue_wait_geq, enqueue_write, launch_window};
    use bloomery_gpu_gates::GateError;
    use cuda_core::{
        CudaContext, CudaEvent, CudaStream, DeviceBuffer, IntoResult, LaunchConfig1D,
        PinnedHostBuffer, sys,
    };
    use cuda_device::{
        DisjointSlice, cooperative_launch, grid, kernel, launch_bounds, launch_contract, ptx_asm,
        thread,
    };
    use cuda_host::cuda_module;
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

    // --------------------------------------------------------- back-leg arms

    /// The sum a back-leg round crosses: one routed layer's handoff, 5120 f32.
    const BACKLEG_VALUES: usize = 5120;
    /// Back-leg rounds an arm times unless `--iters` says otherwise.
    const BACKLEG_ITERS: usize = 10_000;
    /// Untimed back-leg rounds ahead of the timed ones unless `--warm` says
    /// otherwise.
    const BACKLEG_WARM: usize = 200;
    /// Threads a back-leg round's check runs: 5120 values, 20 blocks of 256.
    const BACKLEG_THREADS: u32 = 256;
    const BACKLEG_GRID: u32 = (BACKLEG_VALUES / BACKLEG_THREADS as usize) as u32;
    /// The dev arms' crossing: the sum plus the counter as its last word.
    const BACKLEG_DEV_WORDS: usize = BACKLEG_VALUES + 1;
    /// The +chunks arms' paced-copy load: the chunk size the device
    /// residency design's paced host→card copies carry.
    const BACKLEG_CHUNK: usize = 2 << 20;
    /// How many of the chunk loop's copies may be in flight at once.
    const BACKLEG_CHUNK_RING: usize = 8;
    /// How long the driving thread waits for a round's ack before naming the
    /// failure: a wait that never arrives is a protocol failure, not a hang.
    const BACKLEG_ACK_DEADLINE: Duration = Duration::from_secs(10);

    /// The back-leg round's check, one thread a value, then one grid-wide
    /// join and one ack word: the host's spin from its go to that word is
    /// what the arms price. A thread whose value is not the round's reduces
    /// the round into `fault[0]` (`min`) and the bits it read into `fault[1]`
    /// (`max`) — both order-independent, so any number of mismatching
    /// threads leaves the first failing round and the freshest stale value.
    #[cuda_module]
    mod backleg_kernels {
        use super::*;

        #[cooperative_launch]
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(
            domain = 1,
            block = (256, 1, 1),
            requires = (vals.len() >= n, ack.len() >= 1)
        )]
        pub fn backleg_check(
            vals: &[f32],
            n: u32,
            want: f32,
            round: u32,
            fault: *mut u32,
            mut ack: DisjointSlice<u32>,
            ack_value: u32,
        ) {
            let i = thread::index_1d().get();
            if i < n as usize {
                // SAFETY: i < n <= vals.len() by the launch contract.
                let got = unsafe { *vals.get_unchecked(i) };
                if got.to_bits() != want.to_bits() {
                    // SAFETY: `fault` addresses two live device u32s of this
                    // context (the arm's fault word) for every launch of this
                    // module, and the generic-to-global conversion is exact —
                    // the fault sink's own raise makes the same argument.
                    unsafe {
                        ptx_asm!(
                            "{ .reg .b64 g;\n\t\
                               cvta.to.global.u64 g, %0;\n\t\
                               red.relaxed.gpu.global.min.u32 [g], %1;\n\t\
                               add.u64 g, g, 4;\n\t\
                               red.relaxed.gpu.global.max.u32 [g], %2; }",
                            in("l") fault,
                            in("r") round + 1,
                            in("r") got.to_bits(),
                            clobber("memory")
                        );
                    }
                }
            }
            // Every thread reaches the join: the guarded read above is the
            // only early shape, and it skips nothing past here.
            grid::sync();
            if i == 0 {
                // SAFETY: ack.len() >= 1 by the launch contract; the grid
                // joined above, so every read this ack answers for has
                // happened.
                unsafe { *ack.get_unchecked_mut(0) = ack_value };
            }
        }
    }

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
        BacklegHost,
        BacklegDev,
        BacklegHostChunks,
        BacklegDevChunks,
    }

    impl Arm {
        const ALL: [(&'static str, Arm); 12] = [
            ("pinned", Arm::Pinned),
            ("pageable", Arm::Pageable),
            ("registered", Arm::Registered),
            ("sustained", Arm::Sustained),
            ("both", Arm::Both),
            ("attach", Arm::Attach),
            ("pageable-loop", Arm::PageableLoop),
            ("staged-loop", Arm::StagedLoop),
            ("backleg-host", Arm::BacklegHost),
            ("backleg-dev", Arm::BacklegDev),
            ("backleg-host+chunks", Arm::BacklegHostChunks),
            ("backleg-dev+chunks", Arm::BacklegDevChunks),
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

        fn is_backleg(self) -> bool {
            matches!(
                self,
                Arm::BacklegHost | Arm::BacklegDev | Arm::BacklegHostChunks | Arm::BacklegDevChunks
            )
        }

        /// The back-leg arms whose sum crosses by one host→card copy the
        /// host itself issues each round.
        fn is_dev(self) -> bool {
            matches!(self, Arm::BacklegDev | Arm::BacklegDevChunks)
        }

        fn is_chunks(self) -> bool {
            matches!(self, Arm::BacklegHostChunks | Arm::BacklegDevChunks)
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
        iters: Option<usize>,
        warm: Option<usize>,
        seconds: Option<f64>,
        stop_file: Option<PathBuf>,
        chunk_bytes: usize,
        ring: usize,
        fill_threads: usize,
        interval_ms: u64,
    }

    const USAGE: &str = "usage: h2d_probe [--arm A[,B…]] [--bytes N] [--reps K] [--iters N] \
        [--warm W] [--seconds S] [--stop-file PATH] [--chunk N] [--ring R] [--fill-threads T] \
        [--interval-ms M]";

    fn args() -> Result<Args, GateError> {
        let mut a = Args {
            arms: vec![Arm::Pinned, Arm::Pageable, Arm::Registered],
            bytes: None,
            reps: None,
            iters: None,
            warm: None,
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
                "--iters" => a.iters = Some(int("--iters", value("--iters")?)?),
                "--warm" => a.warm = Some(int("--warm", value("--warm")?)?),
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
        let backlegs = a.arms.iter().filter(|x| x.is_backleg()).count();
        if backlegs > 0 && (a.bytes.is_some() || a.reps.is_some()) {
            return Err(
                "--bytes and --reps belong to the copy arms; a backleg arm crosses the handoff's \
                 own 20,484 bytes a round and takes --iters and --warm"
                    .into(),
            );
        }
        if backlegs < a.arms.len() && (a.iters.is_some() || a.warm.is_some()) {
            return Err("--iters and --warm belong to the backleg arms".into());
        }
        for &arm in &a.arms {
            if arm.is_backleg() {
                let iters = a.iters.unwrap_or(BACKLEG_ITERS);
                if iters < MIN_REPS {
                    return Err(format!(
                        "arm {}: --iters {iters}, at least {MIN_REPS} (a percentile line needs \
                         its ranks)",
                        arm.name()
                    )
                    .into());
                }
                continue;
            }
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

    fn one_shot(ctx: &Arc<CudaContext>, card: &Card, arm: Arm, a: &Args) -> Result<(), GateError> {
        let stream = ctx.new_stream()?;
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
        let ctx = CudaContext::new(ordinal)?;
        let card = Card::of(&ctx)?;
        card.print()?;
        let stream = ctx.new_stream()?;
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
            let ctx = CudaContext::new(i)?;
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

    fn copy_loop(ctx: &Arc<CudaContext>, card: &Card, arm: Arm, a: &Args) -> Result<(), GateError> {
        let seconds = a
            .seconds
            .expect("check_args requires --seconds for a loop arm");
        let bytes = a.bytes.unwrap_or(arm.default_bytes());
        let chunk = a.chunk_bytes;
        let src = Source::open(bytes)?;
        let offsets: Vec<usize> = (0..bytes / chunk).map(|i| i * chunk).collect();
        let stream = ctx.new_stream()?;
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

    // ------------------------------------------------------------ back-leg arms

    /// The `p`-quantile (0..=1) of `sorted`: the element at rank `ceil(p·n)`.
    fn pct(sorted: &[f64], p: f64) -> f64 {
        let n = sorted.len();
        let rank = ((p * n as f64).ceil() as usize).clamp(1, n);
        sorted[rank - 1]
    }

    /// Spin on the page's ack word until it reaches `v`, at most
    /// [`BACKLEG_ACK_DEADLINE`]: a round whose ack never arrives is a named
    /// failure, not a hang.
    fn wait_ack(page: &BacklegPage, arm: Arm, v: u32) -> Result<(), GateError> {
        let deadline = Instant::now() + BACKLEG_ACK_DEADLINE;
        while page.ack_value() < v {
            if Instant::now() > deadline {
                return Err(format!(
                    "arm {}: the ack word never reached {v} within {BACKLEG_ACK_DEADLINE:?}: the \
                     wait behind it, or the check kernel after that, stopped",
                    arm.name()
                )
                .into());
            }
            std::hint::spin_loop();
        }
        Ok(())
    }

    /// The dev arms' copy side: the pinned staging words (the sum plus the
    /// counter as its last word) and the copy stream the per-round
    /// `cuMemcpyHtoDAsync` runs on. One copy is in flight at a time — the
    /// round's ack proves the previous one landed before the host rewrites
    /// the staging, and one stream's copies serialize anyway.
    struct DevLeg {
        stream: Arc<CudaStream>,
        pin: PinnedHostBuffer<u32>,
        vram: DeviceBuffer<u32>,
    }

    impl DevLeg {
        fn new(ctx: &Arc<CudaContext>, engine: &CudaStream) -> Result<DevLeg, GateError> {
            let vram = DeviceBuffer::<u32>::zeroed(engine, BACKLEG_DEV_WORDS)?;
            engine.synchronize()?;
            Ok(DevLeg {
                stream: ctx.new_stream()?,
                pin: PinnedHostBuffer::<u32>::zeroed(ctx, BACKLEG_DEV_WORDS)?,
                vram,
            })
        }

        /// Round `i`'s crossing; the call's own host time is the submit the
        /// dev arms report beside the round trip.
        fn submit(&mut self, i: u32) -> Result<(), GateError> {
            // SAFETY: `pin` holds BACKLEG_DEV_WORDS live pinned words, whose
            // bytes go into `vram`'s BACKLEG_DEV_WORDS words of the same
            // context; the previous round's copy has landed (its ack was
            // seen) and this stream's copies serialize, so nothing reads the
            // staging while a copy runs, and `vram` outlives the stream's
            // work (the arm synchronizes before it drops).
            let rc = unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    self.vram.cu_deviceptr(),
                    self.pin.as_ptr().cast(),
                    4 * BACKLEG_DEV_WORDS,
                    self.stream.cu_stream(),
                )
            };
            rc.result()
                .map_err(|e| format!("cuMemcpyHtoDAsync_v2 (backleg round {i}): {e:?}"))?;
            Ok(())
        }
    }

    /// The paced-copy load a streaming back-leg design puts on the link: 2
    /// MiB pinned→VRAM copies back to back on their own stream until `stop`,
    /// each counted once its copy is known complete. Answers the bytes it
    /// moved and the seconds it ran.
    fn chunk_loop(ctx: &Arc<CudaContext>, stop: &AtomicBool) -> Result<(u64, f64), String> {
        let stream = ctx.new_stream().map_err(|e| e.to_string())?;
        let pinned =
            PinnedHostBuffer::<u8>::zeroed(ctx, BACKLEG_CHUNK).map_err(|e| e.to_string())?;
        let mut dst =
            DeviceBuffer::<u8>::zeroed(&stream, BACKLEG_CHUNK).map_err(|e| e.to_string())?;
        stream.synchronize().map_err(|e| e.to_string())?;
        let events: Vec<CudaEvent> = (0..BACKLEG_CHUNK_RING)
            .map(|_| ctx.new_event(None))
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        let t0 = Instant::now();
        let (mut bytes, mut i) = (0u64, 0usize);
        while !stop.load(Ordering::Relaxed) {
            let slot = i % BACKLEG_CHUNK_RING;
            if i >= BACKLEG_CHUNK_RING {
                // The event this slot still carries is the copy a ring's
                // worth of copies ago: waiting it counts that copy's bytes.
                events[slot].synchronize().map_err(|e| e.to_string())?;
                bytes += BACKLEG_CHUNK as u64;
            }
            // SAFETY: `pinned` outlives the stream's copies — every event is
            // synchronized below before `pinned` drops.
            unsafe { dst.copy_from_pinned_host_async(&stream, &pinned) }
                .map_err(|e| e.to_string())?;
            events[slot].record(&stream).map_err(|e| e.to_string())?;
            i += 1;
        }
        for k in i.saturating_sub(BACKLEG_CHUNK_RING)..i {
            events[k % BACKLEG_CHUNK_RING]
                .synchronize()
                .map_err(|e| e.to_string())?;
            bytes += BACKLEG_CHUNK as u64;
        }
        Ok((bytes, t0.elapsed().as_secs_f64()))
    }

    /// One back-leg arm: `--warm` untimed rounds, then `--iters` timed, each
    /// round a full crossing (the sum and its go out, the card checks every
    /// value, one ack word comes back) whose host-clock go-to-ack interval
    /// the arm reports. A second thread enqueues every round's stream work
    /// ahead of the driving one, so the queue's backpressure lands there,
    /// never between a round's go and its ack; a third (the `+chunks` arms)
    /// loads the link with paced copies for the arm's duration.
    fn backleg(ctx: &Arc<CudaContext>, card: &Card, arm: Arm, a: &Args) -> Result<(), GateError> {
        let iters = a.iters.unwrap_or(BACKLEG_ITERS);
        let warm = a.warm.unwrap_or(BACKLEG_WARM);
        let total = warm + iters;
        let dev = arm.is_dev();
        // SAFETY: this binary's own module, loaded once into the card's
        // context before any launch.
        let pm = unsafe { backleg_kernels::load(ctx)? };
        let engine = ctx.new_stream()?;
        let page = BacklegPage::new(ctx, BACKLEG_VALUES)?;
        let fault = DeviceBuffer::<u32>::zeroed(&engine, 2)?;
        engine.synchronize()?;
        // A raw pointer is not `Send`; the address crosses the threads as a
        // number and becomes a pointer again inside the enqueue thread.
        let fault_addr = fault.cu_deviceptr() as usize;
        let prep =
            pm.prepare_backleg_check(LaunchConfig1D::new(BACKLEG_GRID, BACKLEG_THREADS, 0))?;
        let mut ack_win = launch_window::<u32>(page.ack_dev(), 1, ctx);
        let mut dev_leg = dev.then(|| DevLeg::new(ctx, &engine)).transpose()?;
        let vals_win = match dev_leg.as_ref() {
            Some(d) => launch_window::<f32>(d.vram.cu_deviceptr(), BACKLEG_VALUES, ctx),
            None => launch_window::<f32>(page.values_dev(), BACKLEG_VALUES, ctx),
        };
        let cnt_word = match dev_leg.as_ref() {
            Some(d) => d.vram.cu_deviceptr() + 4 * BACKLEG_VALUES as u64,
            None => page.cnt_dev(),
        };

        let mut rt: Vec<f64> = Vec::with_capacity(iters);
        let mut submit: Vec<f64> = Vec::with_capacity(iters);
        let finished = AtomicBool::new(false);
        // What the arm's three threads hand back, as text: the arm's own
        // error type is not `Send`.
        type ThreadsRan = (
            Result<(), String>,
            Result<(), String>,
            Result<Option<(u64, f64)>, String>,
        );
        let (enq_r, drive_r, chunk_r): ThreadsRan = std::thread::scope(|s| {
            // The enqueue thread: every round's wait and check, ahead of the
            // driver. It drives no go, so it can only run ahead — a full
            // queue holds it until the stream drains, which the driver's
            // rounds do.
            let enq = s.spawn(|| {
                let mut r = Ok(());
                for i in 0..total as u32 {
                    if finished.load(Ordering::Relaxed) {
                        break;
                    }
                    r = enqueue_wait_geq(&engine, cnt_word, i + 1)
                        .map_err(|e| e.to_string())
                        .and_then(|()| {
                            pm.backleg_check(
                                &engine,
                                &prep,
                                &vals_win,
                                BACKLEG_VALUES as u32,
                                i as f32,
                                i,
                                fault_addr as *mut u32,
                                &mut ack_win,
                                i + 1,
                            )
                            .map_err(|e| e.to_string())
                        });
                    if r.is_err() {
                        break;
                    }
                }
                r
            });
            let chunks = arm
                .is_chunks()
                .then(|| s.spawn(|| chunk_loop(ctx, &finished)));
            // The driving thread — this one: what lands between a round's go
            // and its ack is the arm's number.
            let mut drive: Result<(), String> = Ok(());
            for i in 0..total as u32 {
                let t0 = if dev {
                    let d = dev_leg.as_mut().expect("the dev arms carry a copy leg");
                    let words = d.pin.as_mut_slice();
                    words[..BACKLEG_VALUES].fill((i as f32).to_bits());
                    words[BACKLEG_VALUES] = i + 1;
                    let ts = Instant::now();
                    let sent = d.submit(i);
                    let t0 = Instant::now();
                    if let Err(e) = sent {
                        drive = Err(e.to_string());
                        break;
                    }
                    if i as usize >= warm {
                        submit.push((t0 - ts).as_secs_f64() * 1e6);
                    }
                    t0
                } else {
                    page.fill_values(i as f32);
                    let t0 = Instant::now();
                    let v = page.cnt_bump();
                    if v != i + 1 {
                        drive = Err(format!(
                            "arm {}: the counter jumped to {v} at round {i}",
                            arm.name()
                        ));
                        break;
                    }
                    t0
                };
                if let Err(e) = wait_ack(&page, arm, i + 1) {
                    drive = Err(e.to_string());
                    break;
                }
                if i as usize >= warm {
                    rt.push(t0.elapsed().as_secs_f64() * 1e6);
                }
            }
            // However the driving loop ended, the stream must be able to
            // drain: the counter goes far past anything a round bumps it to,
            // so every wait still pending passes and the enqueue thread —
            // which may be sitting in a full queue — goes home; the chunk
            // thread is told to stop.
            finished.store(true, Ordering::Release);
            page.release();
            if let Some(d) = dev_leg.as_ref()
                && let Err(e) = enqueue_write(&d.stream, cnt_word, 1 << 30)
            {
                drive = Err(format!("arm {}: releasing the waits: {e}", arm.name()));
            }
            let enq = match enq.join() {
                Ok(r) => r,
                Err(_) => Err("the enqueue thread panicked".to_string()),
            };
            let chunk = match chunks {
                Some(h) => match h.join() {
                    Ok(r) => r.map(Some),
                    Err(_) => Err("the chunk thread panicked".to_string()),
                },
                None => Ok(None),
            };
            (enq, drive, chunk)
        });
        enq_r.map_err(GateError::from)?;
        drive_r.map_err(GateError::from)?;
        let chunk_gbps = match chunk_r.map_err(GateError::from)? {
            Some((bytes, secs)) if secs > 0.0 => Some(bytes as f64 / secs / 1e9),
            Some(_) => {
                return Err(format!(
                    "arm {}: the chunk loop ran no time, so no load it put on the link",
                    arm.name()
                )
                .into());
            }
            None => None,
        };
        engine.synchronize()?;
        let f = fault.to_host_vec(&engine)?;
        if f[0] != 0 {
            // The card caught a value that was not its round's: the arm's
            // whole point — the named error says which crossing let it by.
            let got = f32::from_bits(f[1]);
            return Err(if dev {
                format!(
                    "arm {}: the card read a stale value: round {} of {total} read {got} where \
                     the round's value was {} — the wait on the copy's own counter word released \
                     before the copy's values were visible, so a device-resident back-leg needs a \
                     signal other than a memory-op wait on a copied word (an event the values' \
                     reader waits on, or a flag the copy's completion raises)",
                    arm.name(),
                    f[0],
                    f[0] - 1,
                )
            } else {
                format!(
                    "arm {}: the card read a stale value: round {} of {total} read {got} where \
                     the round's value was {} — the kernel read the mapped page before the host's \
                     stores of the round were visible to it",
                    arm.name(),
                    f[0],
                    f[0] - 1,
                )
            }
            .into());
        }
        let mut sorted = rt.clone();
        sorted.sort_by(f64::total_cmp);
        let mut line = format!(
            "iters={iters} warm={warm} bytes={} rt_us_median={:.2} rt_us_p90={:.2} \
             rt_us_p99={:.2} rt_us_max={:.2}",
            4 * if dev {
                BACKLEG_DEV_WORDS
            } else {
                BACKLEG_VALUES
            },
            pct(&sorted, 0.5),
            pct(&sorted, 0.9),
            pct(&sorted, 0.99),
            sorted[sorted.len() - 1],
        );
        if dev {
            let mut submits = submit.clone();
            submits.sort_by(f64::total_cmp);
            line = format!(
                "{line} submit_us_median={:.2} submit_us_p99={:.2}",
                pct(&submits, 0.5),
                pct(&submits, 0.99),
            );
        }
        if let Some(g) = chunk_gbps {
            line = format!("{line} chunks_GB/s={g:.2}");
        }
        println!("h2d arm={} {} {line}", arm.name(), card.tag());
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        let a = args()?;
        for &arm in &a.arms {
            match arm {
                Arm::Attach => attach()?,
                Arm::Sustained | Arm::Both => sustained(&a, arm)?,
                _ => {
                    let ctx = CudaContext::new(0)?;
                    let card = Card::of(&ctx)?;
                    card.print()?;
                    if arm.is_loop() {
                        copy_loop(&ctx, &card, arm, &a)?;
                    } else if arm.is_backleg() {
                        backleg(&ctx, &card, arm, &a)?;
                    } else {
                        one_shot(&ctx, &card, arm, &a)?;
                    }
                }
            }
        }
        Ok(())
    }
}
