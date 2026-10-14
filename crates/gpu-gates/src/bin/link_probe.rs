//! The protocol probe of `docs/contrib/dgx-spark.md` section 9.6: the host
//! tier's counters carried between nodes by the engine's RDMA link ends
//! (`bloomery_gpu::host::RdmaEnd`), timed as one round trip a generation.
//!
//! N nodes, one process each: rank 0 is the stage, ranks 1.. are peers.
//! Every node allocates the same page: the image, each peer's rows, a go and
//! a done counter for each peer, then the link's tail. Rank 0 holds one end
//! for each peer and a peer one end, each end with its own queue pair and
//! pump thread. The counters cross as the tier protocol's do: a peer's go
//! carries the image to it, its done carries its rows back.
//!
//! For generation g = 1, 2, … the stage's kernel writes the image, 256
//! lines of 64 bytes with every word g; its stream adds one to every peer's
//! go behind a system-scope barrier, then waits for at least one on every
//! peer's done and takes it back. A peer's stream waits for at least one on
//! its go and takes it back, runs one kernel that counts the image lines not
//! reading g (stale lines: data its counter overtook) and writes 1024 lines
//! of rows, every word g, and adds one to its done behind a system-scope
//! barrier. The stage's next kernel counts the stale rows lines. A card
//! rewrites its region only for its next generation, which waits for the
//! other side to take this one: that take follows the landing, not the
//! completion (`docs/contrib/dgx-spark.md` 9.5, item 2).
//!
//! The round trip of generation g is read on the stage's card from
//! `%globaltimer`: from the end of the kernel that wrote image g to the start
//! of the kernel its waits released. Every wait is bounded: with no progress
//! for `--deadline-ms`, or with an end broken, the probe prints each counter
//! as the host reads it and why the end broke, releases every counter its
//! stream waits on, and waits one more deadline for the stream to drain; a
//! stream not seen to drain ends the process without teardown.
//!
//! `--size BYTES` runs no kernel: a host thread on each node plays its card
//! with the image and each peer's rows `BYTES` long, and only each region's
//! last word carries the generation. The stage prints half the round trip,
//! the one-way time `ib_write_lat` prints as `t_typical`, carried by the
//! link's posts.
//!
//! Both ends run this binary on a host with rdma-core (`libibverbs.so.1`):
//!
//! ```text
//! link_probe --nodes 2 --rank 0 --dev <ibdev> --gid <index> --listen <ip:port>
//! link_probe --nodes 2 --rank 1 --dev <ibdev> --gid <index> --stage <ip:port>
//! ```
//!
//! Optional: `--iters` (timed generations), `--warmup`, `--size`,
//! `--deadline-ms`. Every node runs the same `--size`: setup refuses a
//! mismatch. The stage prints one `link` line (the round trip's percentiles
//! and the stale lines of both directions, or under `--size` the one-way
//! time) and fails when a line was stale.
//!
//! No lease runner runs this probe, so its numbers are contributor evidence
//! (the list in CONTRIBUTING.md), never numbers for our tables.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("link_probe", probe::run())
}

#[cfg(feature = "gpu")]
mod probe {
    use bloomery_gpu::host::step::RELEASE;
    use bloomery_gpu::host::{Crossing, RdmaEnd, Reader, link_tail};
    use bloomery_gpu::{Gpu, MappedHost, mem_batch, op_add, op_barrier_sys, op_wait_geq, window};
    use bloomery_gpu_gates::{GateError, verdict};
    use bloomery_verbs::{Endpoint, Gid};
    use cuda_core::{CudaEvent, DeviceBuffer, LaunchConfig1D, sys};
    use cuda_device::{DisjointSlice, debug, kernel, launch_bounds, launch_contract, thread};
    use cuda_host::cuda_module;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::panic::AssertUnwindSafe;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{Duration, Instant};

    const LINE: usize = 64;
    /// The handoff image: 16 KB.
    const IMAGE_BYTES: usize = 256 * LINE;
    /// One peer's rows: 64 KB.
    const ROWS_BYTES: usize = 1024 * LINE;
    /// Threads of the one block of [`link_kernels::lines`].
    const THREADS: u32 = 256;
    /// Generations the stream runs ahead of the host's bounded wait.
    const AHEAD: usize = 32;

    #[cuda_module]
    mod link_kernels {
        use super::*;

        /// One block: thread 0 stamps `%globaltimer` into `stamps[2 slot]` at
        /// entry. Unless `want` is 0, every line `l < seen_lines` of `seen`
        /// whose 16 words are not all `want` counts as stale into the
        /// thread's `stale` slot; a thread reads whole lines. Every word of
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
                put.len() >= 16 * put_lines,
                stale.len() >= 256,
                stamps.len() >= 2 * slot + 2
            )
        )]
        pub fn lines(
            seen: &[u32],
            seen_lines: u32,
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
                let mut off = 0;
                let mut w = 0;
                while w < 16 {
                    // SAFETY: 16 l + w < 16 seen_lines <= seen.len() by the
                    // launch contract.
                    off |= unsafe { *seen.get_unchecked(16 * l + w) } ^ want;
                    w += 1;
                }
                bad += u64::from(off != 0);
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
        /// `warmup + iters`, a generation that fits a word.
        total: u32,
        /// `--size`: the host plays the card over regions this long; 0 runs
        /// the card.
        size: usize,
        deadline: Duration,
    }

    fn opts() -> Result<Opts, GateError> {
        const USAGE: &str = "link_probe --nodes N --rank R --dev IBDEV --gid INDEX \
             (--listen IP:PORT | --stage IP:PORT) [--iters N] [--warmup N] [--size BYTES] \
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
            size: 0,
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
                "--size" => o.size = v.parse().map_err(bad)?,
                "--deadline-ms" => o.deadline = Duration::from_millis(v.parse().map_err(bad)?),
                _ => return Err(format!("unknown argument {k}; usage: {USAGE}").into()),
            }
        }
        if o.nodes < 2 || o.rank >= o.nodes || o.dev.is_empty() || o.iters == 0 {
            return Err(format!(
                "want --nodes >= 2, --rank below it, --dev and --iters >= 1; usage: {USAGE}"
            )
            .into());
        }
        if !o.size.is_multiple_of(LINE) {
            return Err(format!("--size {}: want a multiple of {LINE} bytes", o.size).into());
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
            .and_then(|t| u32::try_from(t).ok())
            .filter(|&t| t < u32::MAX)
            .ok_or("--iters plus --warmup must stay under 2^32 - 1: a generation is a word")?;
        Ok(o)
    }

    /// The page every node allocates: the image, then each peer's rows, then
    /// a go and a done counter for each peer, each on its own line, then the
    /// link's tail, two lines of it for each peer.
    #[derive(Clone, Copy)]
    struct Layout {
        nodes: usize,
        image: usize,
        rows: usize,
    }

    impl Layout {
        /// The first byte of peer `j`'s rows.
        fn rows_off(self, j: usize) -> usize {
            self.image + (j - 1) * self.rows
        }

        /// Peer `j`'s go counter.
        fn go(self, j: usize) -> usize {
            self.rows_off(self.nodes) + 2 * LINE * (j - 1)
        }

        /// Peer `j`'s done counter.
        fn done(self, j: usize) -> usize {
            self.go(j) + LINE
        }

        /// The bytes before the link's tail.
        fn words(self) -> usize {
            self.go(self.nodes)
        }

        /// The first byte of peer `j`'s tail lines: past those of every
        /// peer before it.
        fn tail(self, j: usize) -> usize {
            link_tail(self.words(), 2 * (j - 1)).1
        }

        fn bytes(self) -> usize {
            link_tail(self.words(), 2 * (self.nodes - 1)).1
        }

        /// What crosses between the stage and peer `j`: its go with the
        /// image, its done with its rows.
        fn crossings(self, j: usize) -> Vec<Crossing> {
            let (image, rows) = (
                0..self.image,
                self.rows_off(j)..self.rows_off(j) + self.rows,
            );
            vec![
                Crossing {
                    counter: self.go(j),
                    reader: Reader::Tier,
                    spans: vec![image],
                    prog: None,
                },
                Crossing {
                    counter: self.done(j),
                    reader: Reader::Stage,
                    spans: vec![rows],
                    prog: None,
                },
            ]
        }
    }

    pub fn run() -> Result<(), GateError> {
        let o = opts()?;
        let layout = if o.size == 0 {
            Layout {
                nodes: o.nodes,
                image: IMAGE_BYTES,
                rows: ROWS_BYTES,
            }
        } else {
            Layout {
                nodes: o.nodes,
                image: o.size,
                rows: o.size,
            }
        };
        let gpu = Gpu::new()?;
        let ctx = gpu.context();
        let page = Arc::new(MappedHost::new(
            ctx,
            layout.bytes(),
            "cuMemHostAlloc (the probe's page)",
        )?);
        let peers: Vec<usize> = if o.rank == 0 {
            (1..o.nodes).collect()
        } else {
            vec![o.rank]
        };
        let side = if o.rank == 0 {
            Reader::Stage
        } else {
            Reader::Tier
        };
        let mut ends = peers
            .iter()
            .map(|&j| {
                RdmaEnd::new(
                    Arc::clone(&page),
                    layout.tail(j),
                    layout.crossings(j),
                    side,
                    &o.dev,
                    o.gid,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut control = Control::open(&o)?;
        let local: Vec<Endpoint> = ends
            .iter()
            .enumerate()
            .map(|(i, e)| e.endpoint(psn(o.rank, i)))
            .collect();
        let remote = control.exchange(&local, o.rank, o.size)?;
        for ((e, l), r) in ends.iter_mut().zip(&local).zip(&remote) {
            e.connect(l.psn, r)?;
        }
        println!(
            "link setup rank={} nodes={} dev={} gid_index={} page_bytes={} \
             access=local_write,remote_write size={} iters={} warmup={}",
            o.rank,
            o.nodes,
            o.dev,
            o.gid,
            layout.bytes(),
            o.size,
            o.iters,
            o.warmup
        );
        let card = Card {
            layout,
            rank: o.rank,
            peers: &peers,
            page: &page,
            ends: &ends,
            total: o.total,
            deadline: o.deadline,
        };
        if o.size != 0 {
            return host_run(&o, &card, &mut control);
        }
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let kernels = unsafe { bloomery_gpu::shared_module!(link_kernels, ctx)? };
        let total = o.total as usize;
        let mut stamps = DeviceBuffer::<u64>::zeroed(gpu.stream(), 2 * (total + 1))?;
        let mut stale = DeviceBuffer::<u64>::zeroed(gpu.stream(), THREADS as usize)?;
        let stream = Stream {
            card: &card,
            gpu: &gpu,
            kernels: &kernels,
            dev: page.dev_at(0),
        };
        stream.launch(&mut stale, &mut stamps, 0, 0, 0, 0)?;
        gpu.stream().synchronize()?;
        stale.copy_from_host(gpu.stream(), &[0; THREADS as usize])?;
        control.ready()?;
        stream.drive(&mut stale, &mut stamps)?;
        let stale_here: u64 = stale.to_host_vec(gpu.stream())?.iter().sum();
        let stamps = stamps.to_host_vec(gpu.stream())?;
        let kernel_ns: Vec<u64> = (1..=total)
            .map(|g| stamps[2 * (g - 1) + 1] - stamps[2 * (g - 1)])
            .collect();
        if o.rank != 0 {
            control.report(stale_here, pct(&mut kernel_ns[o.warmup..].to_vec(), 50.0))?;
            control.ready()?;
            println!("link peer rank={} stale_image={stale_here}", o.rank);
            return Ok(());
        }
        let reports = control.collect()?;
        control.ready()?;
        report(&o, &stamps, stale_here, &reports, &kernel_ns)
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
        let mut rtt: Vec<u64> = (o.warmup + 1..=o.total as usize)
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
            "link nodes={} iters={} rtt_us_p50={:.2} rtt_us_p90={:.2} rtt_us_p99={:.2} \
             rtt_us_p999={:.2} rtt_us_min={:.2} rtt_us_max={:.2} rtt_us_mean={:.2} stale_image={stale_image} \
             stale_rows={stale_rows} stage_kernel_us_p50={:.2} peer_kernel_us_p50={:.2} timer_gcd_ns={tick}",
            o.nodes,
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
        stale_verdict(stale_image + stale_rows)
    }

    fn stale_verdict(stale: u64) -> Result<(), GateError> {
        let pass = stale == 0;
        println!(
            "{}: link_probe stale lines after the counter {stale}",
            verdict(pass)
        );
        if pass {
            Ok(())
        } else {
            Err("stale lines after the counter".into())
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

    /// One node's side of the protocol, whoever plays it: the page, the
    /// counters this rank adds to and waits on, and the ends that carry them.
    struct Card<'a> {
        layout: Layout,
        rank: usize,
        /// The peers this rank's ends reach: every peer on the stage, itself
        /// on a peer.
        peers: &'a [usize],
        page: &'a MappedHost,
        ends: &'a [RdmaEnd],
        total: u32,
        deadline: Duration,
    }

    impl Card<'_> {
        fn word(&self, off: usize) -> &AtomicU32 {
            self.page
                .atomic_u32(off)
                .expect("every counter and region word lies in the layout's page")
        }

        /// The first end that broke, by why.
        fn broken(&self) -> Option<String> {
            self.ends.iter().find_map(RdmaEnd::broken)
        }

        /// The stall's report at generation `g`: every counter of this
        /// rank's peers as the host reads it, and any end's failure.
        fn stalled(&self, g: usize, why: &str) -> GateError {
            let l = self.layout;
            let words: Vec<String> = self
                .peers
                .iter()
                .map(|&j| {
                    format!(
                        "go[{j}]={} done[{j}]={}",
                        self.word(l.go(j)).load(Ordering::Acquire),
                        self.word(l.done(j)).load(Ordering::Acquire)
                    )
                })
                .collect();
            format!(
                "rank {}: {why} at generation {g}: {} (host reads); link: {}",
                self.rank,
                words.join(" "),
                self.broken().unwrap_or_else(|| "carrying".into())
            )
            .into()
        }

        /// Release every counter this rank waits on: its ends stop carrying
        /// every counter, and once their posts are over raise the ones this
        /// rank reads to [`RELEASE`]; `Err` names an end whose posts are not
        /// known to be over, whose counters stay ([`RdmaEnd::release`]).
        fn release(&self) -> Result<(), String> {
            RdmaEnd::release(self.ends, RELEASE)
        }
    }

    /// The host as the card: `--size`'s run on this node.
    fn host_run(o: &Opts, card: &Card<'_>, control: &mut Control) -> Result<(), GateError> {
        control.ready()?;
        let l = card.layout;
        let mut one_way = Vec::with_capacity(o.total as usize);
        let mut stale = 0u64;
        let played = (1..=card.total).try_for_each(|g| {
            let start = Instant::now();
            if o.rank == 0 {
                card.word(l.image - 4).store(g, Ordering::Relaxed);
                for &j in card.peers {
                    card.word(l.go(j)).fetch_add(1, Ordering::Release);
                }
                for &j in card.peers {
                    host_wait(card, l.done(j), g)?;
                    let last = l.rows_off(j) + l.rows - 4;
                    stale += u64::from(card.word(last).load(Ordering::Relaxed) != g);
                    card.word(l.done(j)).fetch_sub(1, Ordering::AcqRel);
                }
                one_way.push(start.elapsed().as_nanos() as u64 / 2);
            } else {
                host_wait(card, l.go(o.rank), g)?;
                stale += u64::from(card.word(l.image - 4).load(Ordering::Relaxed) != g);
                card.word(l.go(o.rank)).fetch_sub(1, Ordering::AcqRel);
                let last = l.rows_off(o.rank) + l.rows - 4;
                card.word(last).store(g, Ordering::Relaxed);
                card.word(l.done(o.rank)).fetch_add(1, Ordering::Release);
            }
            Ok::<(), GateError>(())
        });
        if let Err(e) = played {
            return Err(match card.release() {
                Ok(()) => e,
                Err(held) => format!("{e}; {held}").into(),
            });
        }
        if let Some(b) = card.broken() {
            return Err(format!("the link broke: {b}").into());
        }
        if o.rank != 0 {
            control.report(stale, 0)?;
            control.ready()?;
            println!(
                "link peer rank={} size={} stale_words={stale}",
                o.rank, o.size
            );
            return Ok(());
        }
        let peer_stale: u64 = control.collect()?.iter().map(|r| r.0).sum();
        control.ready()?;
        let t = &mut one_way[o.warmup..];
        let us = |ns: u64| ns as f64 / 1e3;
        let mean = t.iter().sum::<u64>() as f64 / t.len() as f64 / 1e3;
        let [p50, p90, p99] = [50.0, 90.0, 99.0].map(|p| us(pct(t, p)));
        println!(
            "link size nodes={} bytes={} iters={} one_way_us_p50={p50:.2} one_way_us_p90={p90:.2} \
             one_way_us_p99={p99:.2} one_way_us_min={:.2} one_way_us_mean={mean:.2} \
             stale_rows={stale} stale_image={peer_stale}",
            o.nodes,
            o.size,
            o.iters,
            us(t[0]),
        );
        stale_verdict(stale + peer_stale)
    }

    /// The host's wait for at least one on the counter at `off`, bounded by
    /// the deadline and by the ends' failure.
    fn host_wait(card: &Card<'_>, off: usize, g: u32) -> Result<(), GateError> {
        let start = Instant::now();
        let mut spins = 0u32;
        while card.word(off).load(Ordering::Acquire) < 1 {
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(1024) {
                if card.broken().is_some() {
                    return Err(card.stalled(g as usize, "the link broke"));
                }
                if start.elapsed() > card.deadline {
                    return Err(card.stalled(g as usize, "no progress"));
                }
            }
            std::hint::spin_loop();
        }
        Ok(())
    }

    /// The card's stream: the launches and stream memory operations of every
    /// generation, enqueued at most [`AHEAD`] generations in front of the
    /// host's bounded wait.
    struct Stream<'a> {
        card: &'a Card<'a>,
        gpu: &'a Gpu,
        kernels: &'a link_kernels::LoadedModule,
        dev: sys::CUdeviceptr,
    }

    impl Stream<'_> {
        /// A `u32` window of `bytes` bytes from byte `off` of the page.
        fn span(&self, off: usize, bytes: usize) -> std::mem::ManuallyDrop<DeviceBuffer<u32>> {
            // SAFETY: the span lies inside the page (the layout's own
            // regions), which is mapped into this context and outlives every
            // launch: the run drains the stream before the page drops, or
            // ends the process without dropping it.
            unsafe { window::<u32>(self.dev + off as u64, bytes / 4, self.gpu.context()) }
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
            let l = self.card.layout;
            let (seen_off, seen_bytes, put_off, put_bytes) = if self.card.rank == 0 {
                (
                    l.rows_off(1),
                    l.rows_off(l.nodes) - l.rows_off(1),
                    0,
                    l.image,
                )
            } else {
                let r = self.card.rank;
                (0, l.image, l.rows_off(r), l.rows)
            };
            let seen = self.span(seen_off, seen_bytes);
            let mut put = self.span(put_off, put_bytes);
            let prep = self
                .kernels
                .prepare_lines(LaunchConfig1D::new(1, THREADS, 0))?;
            self.kernels.lines(
                self.gpu.stream(),
                &prep,
                &seen,
                u32::try_from(seen_bytes / LINE)?,
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

        /// Every generation, then a drained stream. A failure on the way (a
        /// stall, a broken end, a driver error or a panic) releases the
        /// counters the stream waits on; a stream not seen to drain after
        /// that ends the process here, leaving the page and the device
        /// buffers it may still reach to the OS.
        fn drive(
            &self,
            stale: &mut DeviceBuffer<u64>,
            stamps: &mut DeviceBuffer<u64>,
        ) -> Result<(), GateError> {
            let run =
                std::panic::catch_unwind(AssertUnwindSafe(|| self.generations(stale, stamps)));
            let why = match run {
                Ok(Ok(())) => return Ok(()),
                Ok(Err(e)) => e.to_string(),
                Err(_) => "the card thread panicked".to_string(),
            };
            // The release and the drain each wait on the link; why the run
            // failed is out first. `eprintln!` panics when stderr fails.
            let _ = writeln!(std::io::stderr(), "link_probe: {why}; releasing the link");
            match self.drain_released() {
                Ok(()) => Err(format!("{why}; stream drained after release: true").into()),
                Err(e) => {
                    // `eprintln!` panics when stderr fails, and the unwind
                    // would drop the buffers this exit leaves alone.
                    let _ = writeln!(
                        std::io::stderr(),
                        "link_probe: stream drained after release: false ({e}); exiting without \
                         teardown"
                    );
                    std::process::exit(1);
                }
            }
        }

        /// Enqueue generations 1..=total, at most [`AHEAD`] ahead of the
        /// host, and wait for the stream to drain.
        fn generations(
            &self,
            stale: &mut DeviceBuffer<u64>,
            stamps: &mut DeviceBuffer<u64>,
        ) -> Result<(), GateError> {
            let ctx = self.gpu.context();
            ctx.bind_to_thread()?;
            let stream = self.gpu.stream();
            let (l, card) = (self.card.layout, self.card);
            let at = |off: usize| self.dev + off as u64;
            let events = (0..AHEAD)
                .map(|_| ctx.new_event(None))
                .collect::<Result<Vec<_>, _>>()?;
            let mut progress = Instant::now();
            let total = card.total as usize;
            for g in 1..=total {
                if g > AHEAD {
                    self.settle(&events[g % AHEAD], progress, g - AHEAD)?;
                    progress = Instant::now();
                }
                let seq = u32::try_from(g)?;
                if card.rank == 0 {
                    self.launch(stale, stamps, seq - 1, seq, l.image / LINE, g - 1)?;
                    let mut ops = vec![op_barrier_sys()];
                    ops.extend(card.peers.iter().map(|&j| op_add(at(l.go(j)), 1)));
                    for &j in card.peers {
                        ops.push(op_wait_geq(at(l.done(j)), 1));
                        ops.push(op_add(at(l.done(j)), u32::MAX));
                    }
                    // SAFETY: every counter lies in the page (the layout's
                    // own words), which outlives every batch as in `span`.
                    unsafe { mem_batch(stream, &mut ops, "link_probe: the gos and the dones")? };
                } else {
                    let r = card.rank;
                    // SAFETY: the go counter lies in the page, which outlives
                    // every batch as in `span`.
                    unsafe {
                        mem_batch(
                            stream,
                            &mut [op_wait_geq(at(l.go(r)), 1), op_add(at(l.go(r)), u32::MAX)],
                            "link_probe: the go",
                        )?;
                    }
                    self.launch(stale, stamps, seq, seq, l.rows / LINE, g - 1)?;
                    // SAFETY: the done counter lies in the page, which
                    // outlives every batch as in `span`.
                    unsafe {
                        mem_batch(
                            stream,
                            &mut [op_barrier_sys(), op_add(at(l.done(r)), 1)],
                            "link_probe: the done",
                        )?;
                    }
                }
                events[g % AHEAD].record(stream)?;
            }
            if card.rank == 0 {
                self.launch(stale, stamps, card.total, 0, 0, total)?;
            }
            let end = ctx.new_event(None)?;
            end.record(stream)?;
            self.settle(&end, Instant::now(), total)
        }

        /// Wait for `event`, generation `g`'s, until `since` is a deadline
        /// old or an end broke.
        fn settle(&self, event: &CudaEvent, since: Instant, g: usize) -> Result<(), GateError> {
            let mut spins = 0u32;
            while !event.query()? {
                spins = spins.wrapping_add(1);
                if spins.is_multiple_of(1024) {
                    if self.card.broken().is_some() {
                        return Err(self.card.stalled(g, "the link broke"));
                    }
                    if since.elapsed() > self.card.deadline {
                        return Err(self.card.stalled(g, "no progress"));
                    }
                }
                std::hint::spin_loop();
            }
            Ok(())
        }

        /// The end of a failed run: release every counter the stream waits
        /// on, then `Ok` once the stream drained, else why it is not known
        /// to have.
        fn drain_released(&self) -> Result<(), String> {
            self.card.release()?;
            let end = self
                .gpu
                .context()
                .new_event(None)
                .map_err(|e| format!("no end event: {e}"))?;
            end.record(self.gpu.stream())
                .map_err(|e| format!("end event not recorded: {e}"))?;
            let start = Instant::now();
            loop {
                match end.query() {
                    Ok(true) => return Ok(()),
                    Ok(false) if start.elapsed() <= self.card.deadline => std::hint::spin_loop(),
                    Ok(false) => {
                        return Err(format!("not drained within {:?}", self.card.deadline));
                    }
                    Err(e) => return Err(format!("end event query: {e}")),
                }
            }
        }
    }

    // ------------------------------------------------------ control channel

    /// The stage's TCP connections to its peers (one, to the stage, on a
    /// peer): the endpoints and `--size`, the start, and the peers' counts
    /// at the end.
    struct Control {
        socks: Vec<TcpStream>,
        deadline: Duration,
    }

    const ENDPOINT_BYTES: usize = 44;

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

        /// Send each connection its local endpoint and this node's `--size`,
        /// and read the remote ones. Both ends write before they read, so each
        /// end of a mismatch refuses it by name before any generation.
        fn exchange(
            &mut self,
            local: &[Endpoint],
            rank: usize,
            size: usize,
        ) -> Result<Vec<Endpoint>, GateError> {
            let mut remote = Vec::with_capacity(local.len());
            for (s, l) in self.socks.iter_mut().zip(local) {
                let mut b = [0u8; ENDPOINT_BYTES];
                b[0..4].copy_from_slice(&l.qpn.to_le_bytes());
                b[4..8].copy_from_slice(&l.psn.to_le_bytes());
                b[8..24].copy_from_slice(&l.gid.raw);
                b[24..32].copy_from_slice(&l.addr.to_le_bytes());
                b[32..36].copy_from_slice(&l.rkey.to_le_bytes());
                b[36..44].copy_from_slice(&(size as u64).to_le_bytes());
                s.write_all(&b)?;
            }
            for (i, s) in self.socks.iter_mut().enumerate() {
                let other = if rank == 0 { i + 1 } else { 0 };
                let mut b = [0u8; ENDPOINT_BYTES];
                s.read_exact(&mut b)
                    .map_err(|e| format!("rank {other}'s endpoint: {e}"))?;
                let word =
                    |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"));
                let theirs = u64::from_le_bytes(b[36..44].try_into().expect("8 bytes"));
                if theirs != size as u64 {
                    let (stage, peer, r) = if rank == 0 {
                        (size as u64, theirs, other)
                    } else {
                        (theirs, size as u64, rank)
                    };
                    return Err(format!("--size differs: rank 0 {stage}, rank {r} {peer}").into());
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

        /// A byte each way: before the first generation, every node's queue
        /// pairs are ready; after the last, the stage has every peer's done,
        /// so no peer drops its end (its pump and queue pair) with a done
        /// still to post.
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

        /// A peer's stale lines and its kernel's median time, to the stage.
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
