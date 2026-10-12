//! The protocol probe of `docs/contrib/dgx-spark.md` section 9.6: the host
//! tier's word protocol carried between nodes over RDMA, timed as one round
//! trip a generation.
//!
//! N nodes, one process each: rank 0 is the stage, ranks 1.. are peers.
//! For generation g = 1, 2, … the stage's kernel writes the image, 256
//! lines of 64 bytes with every word g; its stream raises the image's flag
//! to g behind a system-scope barrier and waits until every peer's rows flag
//! is at least g. The card cannot post to the NIC, so a pump thread on each
//! node posts the writes: the stage's pump sends the image to every peer
//! once its flag moves, and each peer's pump sends its rows back the same
//! way. A peer's stream waits until its image flag is at least g, then runs
//! one kernel that counts the image lines not reading g (stale lines: data
//! its flag overtook) and writes 1024 lines of rows, every word g, and
//! raises its rows flag. The stage's next kernel counts the stale rows
//! lines.
//!
//! A node reuses its send buffer only after the NIC is done with it: the
//! pump raises the node's sent word to g once every write of generation g
//! completed, and the batch in which the stream raises its own flag to g
//! ends with a wait until the sent word is at least g, so the next kernel
//! writes the region only after that.
//!
//! Each node's page is `cuMemHostAlloc` memory registered with `ibv_reg_mr`
//! without relaxed ordering: on GB10 the NIC cannot reach `cudaMalloc`
//! memory. `--flag write` (the default) sends a region and its flag line as
//! one RDMA write, flag last; `--flag imm` sends the region as a write with
//! immediate, and the receiving pump raises the flag once the completion
//! arrives. Every node runs the same `--flag`: setup refuses a mismatch.
//!
//! The round trip of generation g is read on the stage's card from
//! `%globaltimer`: from the end of the kernel that wrote image g to the start
//! of the kernel its wait released. Every wait is bounded: with no progress
//! for `--deadline-ms` the probe prints each flag word and the sent word as
//! the host reads them (a wait that never passes is told from a write that
//! never landed). A failure of either side, the card thread's or the
//! pump's, ends the run the same way: the pump stops, the stream's waits are
//! released, and the stream drains within one more deadline; a stream not
//! seen to drain ends the process without teardown.
//!
//! Both ends run this binary on a host with rdma-core (`libibverbs.so.1`):
//!
//! ```text
//! link_probe --nodes 2 --rank 0 --dev <ibdev> --gid <index> --listen <ip:port>
//! link_probe --nodes 2 --rank 1 --dev <ibdev> --gid <index> --stage <ip:port>
//! ```
//!
//! Optional: `--iters` (timed generations), `--warmup`, `--flag write|imm`,
//! `--deadline-ms`. The stage prints one `link` line (the round trip's
//! percentiles, the stale lines of both directions) and fails when a line
//! was stale.
//!
//! No lease runner runs this probe, so its numbers are contributor evidence
//! (the list in CONTRIBUTING.md), never numbers for our tables.

#[cfg(feature = "gpu")]
#[path = "shared/verbs.rs"]
mod verbs;

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("link_probe", probe::run())
}

#[cfg(feature = "gpu")]
mod probe {
    use super::verbs::{
        Endpoint, Gid, Nic, Qp, Region, Verbs, WC_RDMA_WRITE, WC_RECV_RDMA_WITH_IMM, Wc, WriteKind,
    };
    use bloomery_gpu::{Gpu, MappedHost, mem_batch, op_barrier_sys, op_wait_geq, op_write, window};
    use bloomery_gpu_gates::{GateError, verdict};
    use cuda_core::{CudaEvent, DeviceBuffer, LaunchConfig1D, sys};
    use cuda_device::{DisjointSlice, debug, kernel, launch_bounds, launch_contract, thread};
    use cuda_host::cuda_module;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::panic::AssertUnwindSafe;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::{Duration, Instant};

    /// The most generations a run takes. The stream's wait is cyclic
    /// (`(i32)(word - value) >= 0`), so a release, one past the last
    /// generation, must stay under 2^31 above every waited value.
    const MAX_GENERATIONS: usize = 0x7fff_fffe;

    /// u32 words in a 64-byte line.
    const WORDS: usize = 16;
    const LINE: usize = 64;
    /// The handoff image: 16 KB.
    const IMAGE_LINES: usize = 256;
    /// One peer's rows: 64 KB.
    const ROWS_LINES: usize = 1024;
    /// Threads of the one block of [`link_kernels::lines`].
    const THREADS: u32 = 256;
    /// Receives each queue pair keeps posted under `--flag imm`.
    const RECVS: usize = 16;
    /// Generations the stream runs ahead of the host's bounded wait.
    const AHEAD: usize = 32;

    #[cuda_module]
    mod link_kernels {
        use super::*;

        /// One block: thread 0 stamps `%globaltimer` into `stamps[2 slot]` at
        /// entry. Unless `want` is 0, every line `l < seen_lines` of `seen`
        /// with `l % stride != stride - 1` (the last line of each stride is a
        /// flag line) whose 16 words are not all `want` counts as stale into
        /// the thread's `stale` slot; a thread reads whole lines. Every word of
        /// the `put_lines` lines of `put` is set to `seq`, word by word across
        /// the block, so a warp's store covers two whole lines. Thread 0 stamps
        /// `stamps[2 slot + 1]` once the block has written.
        #[allow(
            clippy::too_many_arguments,
            reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
        )]
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(
            domain = 1,
            block = (256, 1, 1),
            requires = (
                seen.len() >= 16 * seen_lines,
                stride >= 1,
                put.len() >= 16 * put_lines,
                stale.len() >= 256,
                stamps.len() >= 2 * slot + 2
            )
        )]
        pub fn lines(
            seen: &[u32],
            seen_lines: u32,
            stride: u32,
            want: u32,
            mut put: DisjointSlice<u32>,
            put_lines: u32,
            seq: u32,
            mut stale: DisjointSlice<u64>,
            mut stamps: DisjointSlice<u64>,
            slot: u32,
        ) {
            let t = thread::index_1d().get();
            let at = 2 * slot as usize;
            if t == 0 {
                // SAFETY: only thread 0 writes the stamps; at + 1 < stamps.len()
                // by the launch contract.
                unsafe { *stamps.get_unchecked_mut(at) = debug::globaltimer() };
            }
            let mut bad = 0u64;
            let mut l = t;
            while want != 0 && l < seen_lines as usize {
                if l % stride as usize != stride as usize - 1 {
                    let mut off = 0;
                    let mut w = 0;
                    while w < 16 {
                        // SAFETY: 16 l + w < 16 seen_lines <= seen.len() by the
                        // launch contract.
                        off |= unsafe { *seen.get_unchecked(16 * l + w) } ^ want;
                        w += 1;
                    }
                    bad += u64::from(off != 0);
                }
                l += 256;
            }
            let mut i = t;
            while i < 16 * put_lines as usize {
                // SAFETY: word i belongs to this thread alone (i = t mod 256),
                // and i < 16 put_lines <= put.len() by the launch contract.
                unsafe { *put.get_unchecked_mut(i) = seq };
                i += 256;
            }
            if let Some(s) = stale.get_mut(thread::index_1d()) {
                *s += bad;
            }
            thread::sync_threads();
            if t == 0 {
                // SAFETY: only thread 0 writes the stamps; at + 1 < stamps.len()
                // by the launch contract.
                unsafe { *stamps.get_unchecked_mut(at + 1) = debug::globaltimer() };
            }
        }
    }

    // ------------------------------------------------------------ the run

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Flag {
        /// The flag line rides last in the region's own write.
        Write,
        /// The region goes as a write with immediate; the receiver's pump
        /// raises the flag after the completion.
        Imm,
    }

    impl Flag {
        fn name(self) -> &'static str {
            match self {
                Flag::Write => "write",
                Flag::Imm => "imm",
            }
        }

        /// The mode's word in the endpoint record; 0 names no mode.
        fn word(self) -> u32 {
            match self {
                Flag::Write => 1,
                Flag::Imm => 2,
            }
        }
    }

    struct Opts {
        nodes: usize,
        rank: usize,
        dev: String,
        gid: u8,
        /// The stage's control address: listened on by rank 0, dialled by
        /// the peers.
        control: String,
        iters: usize,
        warmup: usize,
        /// `warmup + iters`, at most [`MAX_GENERATIONS`].
        total: usize,
        flag: Flag,
        deadline: Duration,
    }

    fn opts() -> Result<Opts, GateError> {
        const USAGE: &str = "link_probe --nodes N --rank R --dev IBDEV --gid INDEX \
             (--listen IP:PORT | --stage IP:PORT) [--iters N] [--warmup N] [--flag write|imm] \
             [--deadline-ms MS]";
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut o = Opts {
            nodes: 0,
            rank: usize::MAX,
            dev: String::new(),
            gid: 0,
            control: String::new(),
            iters: 100_000,
            warmup: 1_000,
            total: 0,
            flag: Flag::Write,
            deadline: Duration::from_secs(10),
        };
        let (mut listen, mut stage, mut gid) = (None, None, None);
        let mut it = args.iter();
        while let Some(k) = it.next() {
            let v = it
                .next()
                .ok_or_else(|| format!("{k} takes a value; usage: {USAGE}"))?;
            let bad = |_| format!("{k} {v}: not a number");
            match k.as_str() {
                "--nodes" => o.nodes = v.parse().map_err(bad)?,
                "--rank" => o.rank = v.parse().map_err(bad)?,
                "--dev" => o.dev.clone_from(v),
                "--gid" => gid = Some(v.parse().map_err(bad)?),
                "--listen" => listen = Some(v.clone()),
                "--stage" => stage = Some(v.clone()),
                "--iters" => o.iters = v.parse().map_err(bad)?,
                "--warmup" => o.warmup = v.parse().map_err(bad)?,
                "--deadline-ms" => o.deadline = Duration::from_millis(v.parse().map_err(bad)?),
                "--flag" => {
                    o.flag = match v.as_str() {
                        "write" => Flag::Write,
                        "imm" => Flag::Imm,
                        _ => return Err(format!("--flag {v}: want write or imm").into()),
                    }
                }
                _ => return Err(format!("unknown argument {k}; usage: {USAGE}").into()),
            }
        }
        if o.nodes < 2 || o.rank >= o.nodes || o.dev.is_empty() || o.iters == 0 {
            return Err(format!(
                "want --nodes >= 2, --rank below it, --dev and --iters >= 1; usage: {USAGE}"
            )
            .into());
        }
        o.gid = gid.ok_or_else(|| format!("--gid is required; usage: {USAGE}"))?;
        o.control = match (o.rank, listen, stage) {
            (0, Some(a), None) => a,
            (r, None, Some(a)) if r > 0 => a,
            _ => return Err("rank 0 takes --listen, every other rank --stage".into()),
        };
        o.total = o
            .warmup
            .checked_add(o.iters)
            .filter(|&t| t <= MAX_GENERATIONS)
            .ok_or_else(|| {
                format!(
                    "--iters plus --warmup above {MAX_GENERATIONS}: the stream's wait compares \
                     generations cyclically, so a release must stay under 2^31 above every wait"
                )
            })?;
        Ok(o)
    }

    /// The page every node allocates, in lines: region 0 is the image,
    /// region j (1..nodes) peer j's rows. A region is its data lines and then
    /// one flag line, whose first word is the region's flag. The last line is
    /// the node's own, which no peer writes: its first word is the sent word.
    #[derive(Clone, Copy)]
    struct Layout {
        nodes: usize,
    }

    impl Layout {
        fn data_lines(self, r: usize) -> usize {
            if r == 0 { IMAGE_LINES } else { ROWS_LINES }
        }

        /// The first line of region `r`; `start(nodes)` is the page's end.
        fn start(self, r: usize) -> usize {
            if r == 0 {
                0
            } else {
                IMAGE_LINES + 1 + (r - 1) * (ROWS_LINES + 1)
            }
        }

        /// The byte offset of region `r`'s flag word.
        fn flag(self, r: usize) -> usize {
            (self.start(r) + self.data_lines(r)) * LINE
        }

        /// The byte offset of the sent word.
        fn sent(self) -> usize {
            self.start(self.nodes) * LINE
        }

        fn bytes(self) -> usize {
            self.sent() + LINE
        }
    }

    pub fn run() -> Result<(), GateError> {
        let o = opts()?;
        let layout = Layout { nodes: o.nodes };
        let total = o.total;
        let gpu = Gpu::new()?;
        let ctx = gpu.context();
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let kernels = unsafe { bloomery_gpu::shared_module!(link_kernels, ctx)? };
        let page = MappedHost::new(ctx, layout.bytes(), "cuMemHostAlloc (the probe's page)")?;
        let verbs = Verbs::load()?;
        let nic = Nic::open(&verbs, &o.dev, o.gid)?;
        // SAFETY: `page` is declared before `region`, so it drops after it:
        // the range stays allocated for the region's whole life.
        let region = unsafe { nic.register(page.host_at(0), layout.bytes())? };
        let peers = if o.rank == 0 { o.nodes - 1 } else { 1 };
        let qps = (0..peers)
            .map(|_| nic.queue_pair())
            .collect::<Result<Vec<_>, _>>()?;
        let mut control = Control::open(&o)?;
        let local: Vec<Endpoint> = qps
            .iter()
            .enumerate()
            .map(|(i, qp)| Endpoint {
                qpn: qp.qpn(),
                psn: psn(o.rank, i),
                gid: nic.gid(),
                addr: page.host_at(0) as u64,
                rkey: region.rkey(),
            })
            .collect();
        let remote = control.exchange(&local, o.rank, o.flag)?;
        for ((qp, l), r) in qps.iter().zip(&local).zip(&remote) {
            qp.connect(l.psn, r)?;
        }
        if o.flag == Flag::Imm {
            for qp in &qps {
                (0..RECVS).try_for_each(|_| qp.post_recv())?;
            }
        }
        println!(
            "link setup rank={} nodes={} dev={} gid_index={} page_bytes={} access=local_write,remote_write \
             flag={} iters={} warmup={}",
            o.rank,
            o.nodes,
            o.dev,
            o.gid,
            layout.bytes(),
            o.flag.name(),
            o.iters,
            o.warmup
        );
        let mut stamps = DeviceBuffer::<u64>::zeroed(gpu.stream(), 2 * (total + 1))?;
        let mut stale = DeviceBuffer::<u64>::zeroed(gpu.stream(), THREADS as usize)?;
        let word = |off| {
            page.atomic_u32(off)
                .expect("every flag word and the sent word lie in the layout's page")
        };
        let flags: Vec<&AtomicU32> = (0..o.nodes).map(|r| word(layout.flag(r))).collect();
        let sent = word(layout.sent());
        let card = Card {
            gpu: &gpu,
            kernels: &kernels,
            layout,
            rank: o.rank,
            dev: page.dev_at(0),
            total,
            deadline: o.deadline,
            flags: &flags,
            sent,
        };
        card.launch(&mut stale, &mut stamps, 0, 0, 0, 0)?;
        gpu.stream().synchronize()?;
        stale.copy_from_host(gpu.stream(), &[0; THREADS as usize])?;
        control.ready()?;

        let own = if o.rank == 0 { 0 } else { o.rank };
        let out = Out {
            mode: o.flag,
            flag: flags[own],
            sent,
            local: page.host_at(0) as u64 + (layout.start(own) * LINE) as u64,
            bytes: u32::try_from(match o.flag {
                Flag::Write => (layout.data_lines(own) + 1) * LINE,
                Flag::Imm => layout.data_lines(own) * LINE,
            })?,
            targets: qps
                .iter()
                .zip(&remote)
                .map(|(qp, r)| (qp, r.addr + (layout.start(own) * LINE) as u64, r.rkey))
                .collect(),
        };
        let ins: Vec<(u32, &Qp<'_>, &AtomicU32)> = match o.flag {
            Flag::Write => Vec::new(),
            Flag::Imm if o.rank == 0 => (0..peers)
                .map(|i| (qps[i].qpn(), &qps[i], flags[i + 1]))
                .collect(),
            Flag::Imm => vec![(qps[0].qpn(), &qps[0], flags[0])],
        };
        let stop = AtomicBool::new(false);
        let (driven, pumped) = std::thread::scope(|s| {
            let card = &card;
            let (stale, stamps, stop) = (&mut stale, &mut stamps, &stop);
            let driver = s.spawn(move || card.drive(stale, stamps, stop));
            // Once the card raised `stop`, its release moves the words the
            // pump reads: a pump failure after that is the card's to report.
            let pumped = pump(&nic, &region, &out, &ins, total, o.deadline, stop).or_else(|e| {
                if stop.load(Ordering::Acquire) {
                    Ok(())
                } else {
                    Err(e)
                }
            });
            if pumped.is_err() {
                stop.store(true, Ordering::Release);
            }
            let driven = driver.join().unwrap_or_else(|_| {
                Err(Failed {
                    why: "the card thread panicked in its release".into(),
                    drained: Err("not released".into()),
                })
            });
            (driven, pumped.map_err(|e| e.to_string()))
        });
        ended(driven, pumped)?;
        let stale_here: u64 = stale.to_host_vec(gpu.stream())?.iter().sum();
        let stamps = stamps.to_host_vec(gpu.stream())?;
        let kernel_ns: Vec<u64> = (1..=total)
            .map(|g| stamps[2 * (g - 1) + 1] - stamps[2 * (g - 1)])
            .collect();
        if o.rank != 0 {
            control.report(stale_here, pct(&mut kernel_ns[o.warmup..].to_vec(), 50.0))?;
            println!("link peer rank={} stale_image={stale_here}", o.rank);
            return Ok(());
        }
        let reports = control.collect()?;
        report(&o, &stamps, stale_here, &reports, &kernel_ns)
    }

    /// The card thread's failure: why, and whether its stream drained after
    /// the release (the error says why that is not known).
    struct Failed {
        why: String,
        drained: Result<(), String>,
    }

    /// The run's end from both sides: the pump's own failure first (the pump
    /// returns `Ok` when the card stopped it), then the card's. A stream not
    /// seen to drain may still hold work over the page and the device
    /// buffers, and their drops would wait on it without bound, so the
    /// process ends here and leaves them to the OS.
    fn ended(driven: Result<(), Failed>, pumped: Result<(), String>) -> Result<(), GateError> {
        if driven.is_ok() && pumped.is_ok() {
            return Ok(());
        }
        let mut why = Vec::new();
        if let Err(p) = &pumped {
            why.push(format!("pump: {p}"));
        }
        if let Err(f) = &driven {
            let drained = match &f.drained {
                Ok(()) => "true".to_string(),
                Err(e) => format!("false ({e})"),
            };
            why.push(format!(
                "card: {}; stream drained after release: {drained}",
                f.why
            ));
        }
        let why = why.join("; ");
        if driven.is_err_and(|f| f.drained.is_err()) {
            // `eprintln!` panics when stderr fails, and the unwind would drop
            // the buffers this exit leaves alone.
            let _ = writeln!(
                std::io::stderr(),
                "link_probe: {why}; exiting without teardown"
            );
            std::process::exit(1);
        }
        Err(why.into())
    }

    /// The first packet sequence number of this rank's `i`th queue pair.
    fn psn(rank: usize, i: usize) -> u32 {
        u32::try_from(((rank << 12) | i) & 0xff_ffff).expect("masked to 24 bits")
    }

    /// The stage's line: the round trip of every timed generation, both
    /// stale counts, each node's kernel time.
    fn report(
        o: &Opts,
        stamps: &[u64],
        stale_rows: u64,
        reports: &[(u64, u64)],
        kernel_ns: &[u64],
    ) -> Result<(), GateError> {
        let mut rtt: Vec<u64> = (o.warmup + 1..=o.total)
            .map(|g| stamps[2 * g] - stamps[2 * (g - 1) + 1])
            .collect();
        let tick = rtt.iter().chain(kernel_ns).fold(0, |a, &b| gcd(a, b));
        let us = |ns: u64| ns as f64 / 1e3;
        let mean = rtt.iter().sum::<u64>() as f64 / rtt.len() as f64 / 1e3;
        let [p50, p90, p99, p999] = [50.0, 90.0, 99.0, 99.9].map(|p| us(pct(&mut rtt, p)));
        let (min, max) = (us(rtt[0]), us(rtt[rtt.len() - 1]));
        let stale_image: u64 = reports.iter().map(|r| r.0).sum();
        let peer_kernel = reports.iter().map(|r| r.1).max().unwrap_or(0);
        let stage_kernel = pct(&mut kernel_ns[o.warmup..].to_vec(), 50.0);
        println!(
            "link nodes={} flag={} iters={} rtt_us_p50={:.2} rtt_us_p90={:.2} rtt_us_p99={:.2} \
             rtt_us_p999={:.2} rtt_us_min={:.2} rtt_us_max={:.2} rtt_us_mean={:.2} stale_image={stale_image} \
             stale_rows={stale_rows} stage_kernel_us_p50={:.2} peer_kernel_us_p50={:.2} timer_gcd_ns={tick}",
            o.nodes,
            o.flag.name(),
            o.iters,
            p50,
            p90,
            p99,
            p999,
            min,
            max,
            mean,
            us(stage_kernel),
            us(peer_kernel),
        );
        for (i, r) in reports.iter().enumerate() {
            println!("link peer rank={} stale_image={}", i + 1, r.0);
        }
        let pass = stale_image == 0 && stale_rows == 0;
        println!(
            "{}: link_probe stale lines after the flag {}",
            verdict(pass),
            stale_image + stale_rows
        );
        if pass {
            Ok(())
        } else {
            Err("stale lines after the flag".into())
        }
    }

    /// The `p`th percentile of `v` (nearest rank), sorting it in place.
    fn pct(v: &mut [u64], p: f64) -> u64 {
        v.sort_unstable();
        let at = ((p / 100.0) * v.len() as f64).ceil() as usize;
        v[at.clamp(1, v.len()) - 1]
    }

    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    // ------------------------------------------------------------ the card

    /// One node's card side: the launches and stream memory operations of
    /// every generation, enqueued at most [`AHEAD`] generations in front of
    /// the host's bounded wait, and the words its stream waits on.
    struct Card<'a> {
        gpu: &'a Gpu,
        kernels: &'a link_kernels::LoadedModule,
        layout: Layout,
        rank: usize,
        dev: sys::CUdeviceptr,
        total: usize,
        deadline: Duration,
        /// Every region's flag word on this node's page, by region.
        flags: &'a [&'a AtomicU32],
        sent: &'a AtomicU32,
    }

    impl Card<'_> {
        /// A `u32` window of `lines` lines from line `start` of the page.
        fn span(&self, start: usize, lines: usize) -> std::mem::ManuallyDrop<DeviceBuffer<u32>> {
            // SAFETY: the span lies inside the page (the layout's own lines),
            // which is mapped into this context and outlives every launch:
            // the run drains the stream before the page drops, or ends the
            // process without dropping it.
            unsafe {
                window::<u32>(
                    self.dev + (start * LINE) as u64,
                    lines * WORDS,
                    self.gpu.context(),
                )
            }
        }

        /// One launch of [`link_kernels::lines`] for this rank: the stage
        /// checks every peer's rows against `want` and writes image `seq`;
        /// a peer checks the image against `want` and writes its rows.
        fn launch(
            &self,
            stale: &mut DeviceBuffer<u64>,
            stamps: &mut DeviceBuffer<u64>,
            want: u32,
            seq: u32,
            put_lines: usize,
            slot: usize,
        ) -> Result<(), GateError> {
            let l = self.layout;
            let (seen_start, seen_lines, stride, put_start) = if self.rank == 0 {
                (
                    l.start(1),
                    (l.nodes - 1) * (ROWS_LINES + 1),
                    ROWS_LINES + 1,
                    0,
                )
            } else {
                (0, IMAGE_LINES + 1, IMAGE_LINES + 1, l.start(self.rank))
            };
            let seen = self.span(seen_start, seen_lines);
            let mut put = self.span(
                put_start,
                l.data_lines(if self.rank == 0 { 0 } else { self.rank }),
            );
            let prep = self
                .kernels
                .prepare_lines(LaunchConfig1D::new(1, THREADS, 0))?;
            self.kernels.lines(
                self.gpu.stream(),
                &prep,
                &seen,
                u32::try_from(seen_lines)?,
                u32::try_from(stride)?,
                want,
                &mut put,
                u32::try_from(put_lines)?,
                seq,
                stale,
                stamps,
                u32::try_from(slot)?,
            )?;
            Ok(())
        }

        /// The words this node's stream waits on: every rows flag on the
        /// stage, the image flag on a peer, and the sent word.
        fn waits(&self) -> Vec<&AtomicU32> {
            let mut w = if self.rank == 0 {
                self.flags[1..].to_vec()
            } else {
                vec![self.flags[0]]
            };
            w.push(self.sent);
            w
        }

        /// Every generation, then a drained stream. Any failure on the way
        /// (a stall, a driver error, `stop` raised by the pump, or a panic)
        /// ends through [`Card::release`].
        fn drive(
            &self,
            stale: &mut DeviceBuffer<u64>,
            stamps: &mut DeviceBuffer<u64>,
            stop: &AtomicBool,
        ) -> Result<(), Failed> {
            let mut last = 0;
            let run = std::panic::catch_unwind(AssertUnwindSafe(|| {
                self.generations(stale, stamps, stop, &mut last)
            }));
            let why = match run {
                Ok(Ok(())) => return Ok(()),
                Ok(Err(e)) => e.to_string(),
                Err(_) => "the card thread panicked".to_string(),
            };
            Err(Failed {
                why,
                drained: self.release(last, stop),
            })
        }

        /// Enqueue generations 1..=total, at most [`AHEAD`] ahead of the
        /// host, and wait for the stream to drain; `last` follows the last
        /// generation whose waits were enqueued.
        fn generations(
            &self,
            stale: &mut DeviceBuffer<u64>,
            stamps: &mut DeviceBuffer<u64>,
            stop: &AtomicBool,
            last: &mut u32,
        ) -> Result<(), GateError> {
            let ctx = self.gpu.context();
            ctx.bind_to_thread()?;
            let stream = self.gpu.stream();
            let l = self.layout;
            let sent = self.dev + l.sent() as u64;
            let events = (0..AHEAD)
                .map(|_| ctx.new_event(None))
                .collect::<Result<Vec<_>, _>>()?;
            let mut progress = Instant::now();
            for g in 1..=self.total {
                if g > AHEAD {
                    self.settle(&events[g % AHEAD], progress, g - AHEAD, stop)?;
                    progress = Instant::now();
                }
                let seq = u32::try_from(g)?;
                *last = seq;
                if self.rank == 0 {
                    self.launch(stale, stamps, seq - 1, seq, IMAGE_LINES, g - 1)?;
                    let mut ops =
                        vec![op_barrier_sys(), op_write(self.dev + l.flag(0) as u64, seq)];
                    ops.extend((1..l.nodes).map(|r| op_wait_geq(self.dev + l.flag(r) as u64, seq)));
                    ops.push(op_wait_geq(sent, seq));
                    // SAFETY: every flag word and the sent word lie in the
                    // page (the layout's own words), which outlives every
                    // batch as in `span`.
                    unsafe {
                        mem_batch(
                            stream,
                            &mut ops,
                            "link_probe: the image flag and the rows waits",
                        )?;
                    }
                } else {
                    // SAFETY: the stage's flag word lies in the page, which
                    // outlives every batch as in `span`.
                    unsafe {
                        mem_batch(
                            stream,
                            &mut [op_wait_geq(self.dev + l.flag(0) as u64, seq)],
                            "link_probe: the image wait",
                        )?;
                    }
                    self.launch(stale, stamps, seq, seq, ROWS_LINES, g - 1)?;
                    let flag = self.dev + l.flag(self.rank) as u64;
                    // SAFETY: this rank's flag word and the sent word lie in
                    // the page, which outlives every batch as in `span`.
                    unsafe {
                        mem_batch(
                            stream,
                            &mut [
                                op_barrier_sys(),
                                op_write(flag, seq),
                                op_wait_geq(sent, seq),
                            ],
                            "link_probe: the rows flag and the sent wait",
                        )?;
                    }
                }
                events[g % AHEAD].record(stream)?;
            }
            if self.rank == 0 {
                let last = u32::try_from(self.total)?;
                self.launch(stale, stamps, last, 0, 0, self.total)?;
            }
            let end = ctx.new_event(None)?;
            end.record(stream)?;
            self.settle(&end, Instant::now(), self.total, stop)
        }

        /// Wait for `event`, generation `g`'s, until `since` is a deadline
        /// old or the pump raised `stop`.
        fn settle(
            &self,
            event: &CudaEvent,
            since: Instant,
            g: usize,
            stop: &AtomicBool,
        ) -> Result<(), GateError> {
            while !event.query()? {
                if stop.load(Ordering::Acquire) {
                    return Err(format!("rank {}: stopped at generation {g}", self.rank).into());
                }
                if since.elapsed() > self.deadline {
                    return Err(self.stalled(g));
                }
                std::hint::spin_loop();
            }
            Ok(())
        }

        /// The stall's report: every flag word and the sent word as the host
        /// reads them, before anything is released.
        fn stalled(&self, g: usize) -> GateError {
            let words: Vec<String> = self
                .flags
                .iter()
                .enumerate()
                .map(|(r, f)| format!("flag[{r}]={}", f.load(Ordering::Acquire)))
                .collect();
            format!(
                "rank {}: no progress for {:?} at generation {g}: {} sent={} (host reads; the stream waits on {} \
                 and sent)",
                self.rank,
                self.deadline,
                words.join(" "),
                self.sent.load(Ordering::Acquire),
                if self.rank == 0 { "every rows flag" } else { "flag[0]" },
            )
            .into()
        }

        /// The end of a failed run: `Ok` once the stream drained, else why
        /// it is not known to have. `stop` goes up first, so the pump ends at
        /// its next turn. Then every word the stream waits on is raised to
        /// the generation past `last`, which every enqueued wait takes
        /// (`u32::MAX` would hold them all: the wait is cyclic), again on
        /// every poll, since a peer's write still in flight or the pump's
        /// last turn can lower it, until the stream drains or the deadline
        /// passes.
        fn release(&self, last: u32, stop: &AtomicBool) -> Result<(), String> {
            stop.store(true, Ordering::Release);
            let past = last + 1;
            let ctx = self.gpu.context();
            let end = ctx
                .new_event(None)
                .map_err(|e| format!("no end event: {e}"))?;
            end.record(self.gpu.stream())
                .map_err(|e| format!("end event not recorded: {e}"))?;
            let waits = self.waits();
            let start = Instant::now();
            loop {
                for w in &waits {
                    w.fetch_max(past, Ordering::AcqRel);
                }
                match end.query() {
                    Ok(true) => return Ok(()),
                    Ok(false) if start.elapsed() <= self.deadline => std::hint::spin_loop(),
                    Ok(false) => return Err(format!("not drained within {:?}", self.deadline)),
                    Err(e) => return Err(format!("end event query: {e}")),
                }
            }
        }
    }

    // ------------------------------------------------------------ the pump

    /// What a node's pump sends: its own region, from `local` (`bytes`
    /// long), to each target queue pair at its remote address and key, once
    /// `flag` moves, as `mode` writes it; `sent` is raised once a
    /// generation's writes completed.
    struct Out<'a> {
        mode: Flag,
        flag: &'a AtomicU32,
        sent: &'a AtomicU32,
        local: u64,
        bytes: u32,
        targets: Vec<(&'a Qp<'a>, u64, u32)>,
    }

    /// Post a write per target for each new generation of `out.flag`, reap
    /// the completions, raise `out.sent` to the generation once all its
    /// writes completed and, under `--flag imm`, raise each incoming flag
    /// (`ins`: queue pair number, queue pair, flag) from its completion and
    /// repost the receive. Ends once all `total` generations went out and
    /// came in, at once with `Ok` when the card raised `stop` (the card
    /// reports why), or with the state named when nothing moved for
    /// `deadline`.
    fn pump(
        nic: &Nic<'_>,
        region: &Region<'_>,
        out: &Out<'_>,
        ins: &[(u32, &Qp<'_>, &AtomicU32)],
        total: usize,
        deadline: Duration,
        stop: &AtomicBool,
    ) -> Result<(), GateError> {
        let total = u32::try_from(total)?;
        let (mut posted, mut pending, mut received) = (0u32, 0usize, 0usize);
        let want_in = ins.len() * total as usize;
        let mut wc = [Wc::default(); 16];
        let mut progress = Instant::now();
        let mut spins = 0u32;
        loop {
            if stop.load(Ordering::Acquire) {
                return Ok(());
            }
            let g = out.flag.load(Ordering::Acquire);
            if g > posted {
                // The stream raised flag g after the sent word reached g - 1,
                // so generation g - 1 is the last one posted, and complete.
                if g != posted + 1 || g > total || pending != 0 {
                    return Err(format!(
                        "own flag {g} after generation {posted} posted, {pending} writes unacknowledged"
                    )
                    .into());
                }
                let kind = match out.mode {
                    Flag::Write => WriteKind::Plain,
                    Flag::Imm => WriteKind::Imm(g),
                };
                for &(qp, addr, rkey) in &out.targets {
                    qp.post_write(region, out.local, out.bytes, (addr, rkey), kind)?;
                    pending += 1;
                }
                posted = g;
                progress = Instant::now();
            }
            let n = nic.poll(&mut wc)?;
            for w in &wc[..n] {
                if w.status != 0 {
                    return Err(format!(
                        "completion status {} (vendor {:#x}) opcode {} on qp {}",
                        w.status, w.vendor_err, w.opcode, w.qp_num
                    )
                    .into());
                }
                match w.opcode {
                    WC_RDMA_WRITE => {
                        pending = pending
                            .checked_sub(1)
                            .ok_or("a write completion with no write outstanding")?;
                        if pending == 0 {
                            out.sent.fetch_max(posted, Ordering::AcqRel);
                        }
                    }
                    WC_RECV_RDMA_WITH_IMM => {
                        let &(_, qp, flag) = ins
                            .iter()
                            .find(|i| i.0 == w.qp_num)
                            .ok_or_else(|| format!("a receive on unknown qp {}", w.qp_num))?;
                        flag.fetch_max(u32::from_be(w.imm_data), Ordering::AcqRel);
                        qp.post_recv()?;
                        received += 1;
                    }
                    op => return Err(format!("unexpected completion opcode {op}").into()),
                }
            }
            if n > 0 {
                progress = Instant::now();
            }
            if posted == total && pending == 0 && received == want_in {
                return Ok(());
            }
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(1024) && progress.elapsed() > deadline {
                return Err(format!(
                    "no progress for {deadline:?}: posted {posted} of {total}, {pending} writes \
                     unacknowledged, received {received} of {want_in}, own flag {g}"
                )
                .into());
            }
            std::hint::spin_loop();
        }
    }

    // ------------------------------------------------------ control channel

    /// The stage's TCP connections to its peers (one, to the stage, on a
    /// peer): the endpoints and flag modes, the start, and the peers' counts
    /// at the end.
    struct Control {
        socks: Vec<TcpStream>,
        deadline: Duration,
    }

    const ENDPOINT_BYTES: usize = 40;

    impl Control {
        fn open(o: &Opts) -> Result<Control, GateError> {
            let deadline = Duration::from_secs(60).max(o.deadline);
            let start = Instant::now();
            let mut socks: Vec<Option<TcpStream>> = Vec::new();
            if o.rank == 0 {
                let listener = TcpListener::bind(&o.control)?;
                listener.set_nonblocking(true)?;
                socks.resize_with(o.nodes - 1, || None);
                while socks.iter().any(Option::is_none) {
                    match listener.accept() {
                        Ok((mut s, _)) => {
                            s.set_nonblocking(false)?;
                            s.set_read_timeout(Some(deadline))?;
                            let mut rank = [0u8; 4];
                            s.read_exact(&mut rank)?;
                            let r = u32::from_le_bytes(rank) as usize;
                            if r == 0 || r >= o.nodes || socks[r - 1].is_some() {
                                return Err(
                                    format!("a peer named rank {r} twice or out of range").into()
                                );
                            }
                            socks[r - 1] = Some(s);
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            if start.elapsed() > deadline {
                                return Err(format!(
                                    "peers did not all connect within {deadline:?}"
                                )
                                .into());
                            }
                            std::thread::sleep(Duration::from_millis(10));
                        }
                        Err(e) => return Err(e.into()),
                    }
                }
            } else {
                let mut s = loop {
                    match TcpStream::connect(&o.control) {
                        Ok(s) => break s,
                        Err(e) if start.elapsed() > deadline => {
                            return Err(
                                format!("cannot reach the stage at {}: {e}", o.control).into()
                            );
                        }
                        Err(_) => std::thread::sleep(Duration::from_millis(100)),
                    }
                };
                s.set_read_timeout(Some(deadline))?;
                s.write_all(&u32::try_from(o.rank)?.to_le_bytes())?;
                socks.push(Some(s));
            }
            Ok(Control {
                socks: socks.into_iter().flatten().collect(),
                deadline,
            })
        }

        /// Send each connection its local endpoint and this node's flag mode,
        /// and read the remote ones. Both ends write before they read, so each
        /// end of a mismatch refuses it by name before any generation.
        fn exchange(
            &mut self,
            local: &[Endpoint],
            rank: usize,
            flag: Flag,
        ) -> Result<Vec<Endpoint>, GateError> {
            let mut remote = Vec::with_capacity(local.len());
            for (s, l) in self.socks.iter_mut().zip(local) {
                let mut b = [0u8; ENDPOINT_BYTES];
                b[0..4].copy_from_slice(&l.qpn.to_le_bytes());
                b[4..8].copy_from_slice(&l.psn.to_le_bytes());
                b[8..24].copy_from_slice(&l.gid.raw);
                b[24..32].copy_from_slice(&l.addr.to_le_bytes());
                b[32..36].copy_from_slice(&l.rkey.to_le_bytes());
                b[36..40].copy_from_slice(&flag.word().to_le_bytes());
                s.write_all(&b)?;
            }
            for (i, s) in self.socks.iter_mut().enumerate() {
                let other = if rank == 0 { i + 1 } else { 0 };
                let mut b = [0u8; ENDPOINT_BYTES];
                s.read_exact(&mut b)
                    .map_err(|e| format!("rank {other}'s endpoint: {e}"))?;
                let word =
                    |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"));
                let theirs = match word(36) {
                    m if m == Flag::Write.word() => Flag::Write,
                    m if m == Flag::Imm.word() => Flag::Imm,
                    m => return Err(format!("rank {other} sent flag mode {m}").into()),
                };
                if theirs != flag {
                    let (stage, peer, r) = if rank == 0 {
                        (flag, theirs, other)
                    } else {
                        (theirs, flag, rank)
                    };
                    return Err(format!(
                        "flag mode differs: rank 0 {}, rank {r} {}",
                        stage.name(),
                        peer.name()
                    )
                    .into());
                }
                let mut gid = Gid::default();
                gid.raw.copy_from_slice(&b[8..24]);
                remote.push(Endpoint {
                    qpn: word(0),
                    psn: word(4),
                    gid,
                    addr: u64::from_le_bytes(b[24..32].try_into().expect("8 bytes")),
                    rkey: word(32),
                });
            }
            Ok(remote)
        }

        /// Every node's queue pairs are ready before any node starts: a byte
        /// each way.
        fn ready(&mut self) -> Result<(), GateError> {
            for s in &mut self.socks {
                s.write_all(b"R")?;
            }
            for s in &mut self.socks {
                let mut b = [0u8; 1];
                s.read_exact(&mut b)?;
            }
            Ok(())
        }

        /// A peer's stale image lines and its kernel's median time, to the stage.
        fn report(&mut self, stale: u64, kernel_ns: u64) -> Result<(), GateError> {
            let mut b = [0u8; 16];
            b[0..8].copy_from_slice(&stale.to_le_bytes());
            b[8..16].copy_from_slice(&kernel_ns.to_le_bytes());
            self.socks[0].write_all(&b)?;
            Ok(())
        }

        /// Every peer's report, in rank order.
        fn collect(&mut self) -> Result<Vec<(u64, u64)>, GateError> {
            self.socks
                .iter_mut()
                .enumerate()
                .map(|(i, s)| {
                    s.set_read_timeout(Some(self.deadline))?;
                    let mut b = [0u8; 16];
                    s.read_exact(&mut b)
                        .map_err(|e| format!("rank {}'s report: {e}", i + 1))?;
                    Ok((
                        u64::from_le_bytes(b[0..8].try_into().expect("8 bytes")),
                        u64::from_le_bytes(b[8..16].try_into().expect("8 bytes")),
                    ))
                })
                .collect()
        }
    }
}
