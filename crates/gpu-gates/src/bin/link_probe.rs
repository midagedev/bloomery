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
//! other side to take this one: that take follows a delivery, which waits
//! for the completion of every write its end posted
//! (`docs/contrib/dgx-spark.md` 9.5, item 2).
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
//! link's posts. A failed run prints why and releases every counter; a
//! release that leaves an end standing ends the process without teardown.
//!
//! `--gather ROWS` runs the all-gather of `bloomery_gpu::host::Gathers`
//! among the N nodes instead, each node holding an end to every other (the
//! stage passes the peers' endpoints between them): one replay is a captured
//! chain of a token's 91 gathers, two a layer for 45 layers (`ROWS` rows of
//! 4096 f32, the layer gathers taking the slot's two sets in turn) and the
//! head's (`ROWS` rows of two words). For each gather the card writes this
//! rank's part, every value a function of the replay, the gather, the rank
//! and its index read from device memory, posts it, waits for every other
//! rank's part, counts their values that are not the ones their replay
//! wrote (stale words), adds all parts in rank order
//! (`Gathers::enqueue_sum`) and folds the sum into a per-value digest. Each
//! node prints the gather latency lambda_g, from a stamp after its part is
//! written to a stamp once its wait released (one kernel launch included),
//! per gather and summed over a replay's chain, and checks its last replay's
//! sums against the host's rank-order sum; the stage fails on a stale word,
//! a sum the host disagrees with or a digest that differs between nodes.
//!
//! Both ends run this binary on a host with rdma-core (`libibverbs.so.1`):
//!
//! ```text
//! link_probe --nodes 2 --rank 0 --dev <ibdev> --gid <index> --listen <ip:port>
//! link_probe --nodes 2 --rank 1 --dev <ibdev> --gid <index> --stage <ip:port>
//! ```
//!
//! Optional: `--iters` (timed generations, or replays under `--gather`),
//! `--warmup`, `--size`, `--gather`, `--deadline-ms`. Every node runs the
//! same `--size` and `--gather`: setup refuses a mismatch. The stage prints one `link` line (the round trip's percentiles
//! and the stale lines of both directions, or under `--size` the one-way
//! time) and fails when a line was stale.
//!
//! No lease runner runs this probe, so its numbers are contributor evidence
//! (the list in CONTRIBUTING.md), never numbers for our tables.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::record::at_main("link_probe", bloomery_gpu_gates::record::LINK_PROBE);
    bloomery_gpu_gates::exit_with("link_probe", probe::run())
}

#[cfg(feature = "gpu")]
mod probe {
    use bloomery_gpu::host::step::RELEASE;
    use bloomery_gpu::host::{
        Crossing, GatherSet, GatherShape, Gathers, PostCols, RdmaEnd, Reader, Span, link_tail,
    };
    use bloomery_gpu::{
        Gpu, MappedHost, WindowMut, mem_batch, op_add, op_barrier_sys, op_wait_geq, window,
    };
    use bloomery_gpu_gates::record::{LINK_GATHER, Record};
    use bloomery_gpu_gates::{Fnv1a64, GateError, verdict};
    use bloomery_verbs::{Endpoint, Gid};
    use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, LaunchConfig1D, sys};
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

    /// Gathers a replay runs: a token's two a layer for 45 layers, then the
    /// head's.
    const GATHERS: usize = 91;
    /// Values a row of a layer's gather holds (GLM-5.3-Flash's hidden
    /// width) and of the head's (an argmax's value and id).
    const BODY_WIDTH: usize = 4096;
    const HEAD_WIDTH: usize = 2;

    /// Value `i` of `rank`'s part of gather `gather` in replay `seq`: finite
    /// and normal, of either sign, of an exponent that differs by rank, so a
    /// sum of the parts in another order rounds differently, and a part of
    /// another replay or gather holds other bits.
    #[inline(always)]
    fn value(seq: u32, gather: u32, rank: u32, i: u32) -> f32 {
        let mut h = seq.wrapping_mul(0x9e37_79b1)
            ^ gather.wrapping_mul(0x85eb_ca6b)
            ^ rank.wrapping_mul(0xc2b2_ae35)
            ^ i.wrapping_mul(0x27d4_eb2f);
        h ^= h >> 15;
        h = h.wrapping_mul(0x2c1b_3c6d);
        h ^= h >> 12;
        let exp = 110 + 9 * (rank & 7) + (h >> 28);
        f32::from_bits((h & 0x8000_0000) | (exp << 23) | (h & 0x007f_ffff))
    }

    #[cuda_module]
    mod gather_kernels {
        use super::*;

        /// The one word of `seq`, the replay's number, goes up by one.
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(domain = 1, block = (256, 1, 1))]
        pub fn begin(mut seq: DisjointSlice<u32>) {
            if let Some(s) = seq.get_mut(thread::index_1d()) {
                *s = s.wrapping_add(1);
            }
        }

        /// Thread i below `n` writes value i of `rank`'s part of gather
        /// `gather` in replay `seq[0]` into `part`.
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(domain = 1, block = (256, 1, 1), requires = (seq.len() >= 1))]
        pub fn part(seq: &[u32], gather: u32, rank: u32, n: u32, mut part: DisjointSlice<f32>) {
            let i = thread::index_1d().get();
            if i >= n as usize {
                return;
            }
            // SAFETY: seq.len() >= 1 by the launch contract.
            let s = unsafe { *seq.get_unchecked(0) };
            if let Some(v) = part.get_mut(thread::index_1d()) {
                *v = value(s, gather, rank, i as u32);
            }
        }

        /// Thread 0 stamps `%globaltimer` into stamp `half` of gather `gather`
        /// in replay `seq[0]`, `stamps[2 (gathers (seq[0] - 1) + gather) +
        /// half]`, when that lies in `stamps`.
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(domain = 1, block = (256, 1, 1), requires = (seq.len() >= 1))]
        pub fn mark(
            seq: &[u32],
            gather: u32,
            gathers: u32,
            half: u32,
            mut stamps: DisjointSlice<u64>,
        ) {
            if thread::index_1d().get() != 0 {
                return;
            }
            // SAFETY: seq.len() >= 1 by the launch contract.
            let s = unsafe { *seq.get_unchecked(0) } as usize;
            let at = 2 * (gathers as usize * s.wrapping_sub(1) + gather as usize) + half as usize;
            if s >= 1 && at < stamps.len() {
                // SAFETY: only thread 0 writes, and at < stamps.len() (checked above).
                unsafe { *stamps.get_unchecked_mut(at) = debug::globaltimer() };
            }
        }

        /// Thread i below `n` adds to `stale[i]` the other ranks whose value
        /// i of gather `gather` in replay `seq[0]` their part does not hold;
        /// rank r's part starts at value r `stride` of `parts`.
        #[allow(
            clippy::too_many_arguments,
            reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
        )]
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(
            domain = 1,
            block = (256, 1, 1),
            requires = (seq.len() >= 1, ranks >= 1, parts.len() >= stride * (ranks - 1) + n)
        )]
        pub fn check(
            seq: &[u32],
            gather: u32,
            rank: u32,
            ranks: u32,
            parts: &[f32],
            stride: u32,
            n: u32,
            mut stale: DisjointSlice<u32>,
        ) {
            let i = thread::index_1d().get();
            if i >= n as usize {
                return;
            }
            // SAFETY: seq.len() >= 1 by the launch contract.
            let s = unsafe { *seq.get_unchecked(0) };
            let mut bad = 0u32;
            let mut p = 0u32;
            while p < ranks {
                if p != rank {
                    // SAFETY: p <= ranks - 1 and i < n, so p stride + i <
                    // stride (ranks - 1) + n <= parts.len() by the contract.
                    let got = unsafe { *parts.get_unchecked(p as usize * stride as usize + i) };
                    bad += u32::from(got.to_bits() != value(s, gather, p, i as u32).to_bits());
                }
                p += 1;
            }
            if let Some(c) = stale.get_mut(thread::index_1d()) {
                *c += bad;
            }
        }

        /// Thread i below `n` folds the bits of `y[i]` into `digest[i]`.
        #[kernel]
        #[launch_bounds(256)]
        #[launch_contract(domain = 1, block = (256, 1, 1), requires = (y.len() >= n))]
        pub fn digest(y: &[f32], n: u32, mut digest: DisjointSlice<u32>) {
            let i = thread::index_1d().get();
            if i >= n as usize {
                return;
            }
            // SAFETY: i < n <= y.len() by the launch contract.
            let bits = unsafe { *y.get_unchecked(i) }.to_bits();
            if let Some(d) = digest.get_mut(thread::index_1d()) {
                let m = (*d ^ bits).wrapping_mul(0x9e37_79b1);
                *d = m.rotate_left(13);
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
        /// `--gather`: rows of each gather; 0 runs no gather.
        gather: usize,
        deadline: Duration,
    }

    fn opts() -> Result<Opts, GateError> {
        const USAGE: &str = "link_probe --nodes N --rank R --dev IBDEV --gid INDEX \
             (--listen IP:PORT | --stage IP:PORT) [--iters N] [--warmup N] [--size BYTES] \
             [--gather ROWS] [--deadline-ms MS]";
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
            gather: 0,
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
                "--gather" => o.gather = v.parse().map_err(bad)?,
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
        if o.gather != 0 && o.size != 0 {
            return Err("--gather and --size are two runs: give one".into());
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
                    spans: vec![Span::whole(image)],
                    prog: None,
                },
                Crossing {
                    counter: self.done(j),
                    reader: Reader::Stage,
                    spans: vec![Span::whole(rows)],
                    prog: None,
                },
            ]
        }
    }

    pub fn run() -> Result<(), GateError> {
        let o = opts()?;
        let gpu = Gpu::new()?;
        let ctx = gpu.context();
        if o.gather != 0 {
            return gather_run(&o, &gpu);
        }
        let peers: Vec<usize> = if o.rank == 0 {
            (1..o.nodes).collect()
        } else {
            vec![o.rank]
        };
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
        let page = Arc::new(MappedHost::new(
            ctx,
            layout.bytes(),
            "cuMemHostAlloc (the probe's page)",
        )?);
        let side = if o.rank == 0 {
            Reader::Stage
        } else {
            Reader::Tier
        };
        let (ends, mut control) = open_ends(
            &o,
            &page,
            &peers,
            |j| (layout.tail(j), layout.crossings(j), side),
            &PostCols::default(),
        )?;
        let watch = peers
            .iter()
            .flat_map(|&j| {
                [
                    (format!("go[{j}]"), layout.go(j)),
                    (format!("done[{j}]"), layout.done(j)),
                ]
            })
            .collect();
        let card = Card {
            node: Node {
                rank: o.rank,
                page: &page,
                ends: &ends,
                watch,
                deadline: o.deadline,
            },
            layout,
            peers: &peers,
            total: o.total,
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

    /// This node's ends over `page`, one for each of `peers`, its tail line,
    /// crossings and side from `of`, each posting the rows `cols` holds,
    /// connected to the other nodes' through the control channel, which is
    /// returned with them. A peer equal to this node's rank is the layout's
    /// own block on a star's peer, whose end reaches the stage.
    fn open_ends(
        o: &Opts,
        page: &Arc<MappedHost>,
        peers: &[usize],
        of: impl Fn(usize) -> (usize, Vec<Crossing>, Reader),
        cols: &PostCols,
    ) -> Result<(Vec<RdmaEnd>, Control), GateError> {
        let mut ends = peers
            .iter()
            .map(|&j| {
                let (tail, crossings, side) = of(j);
                RdmaEnd::new(
                    Arc::clone(page),
                    tail,
                    crossings,
                    side,
                    &o.dev,
                    o.gid,
                    cols.clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut control = Control::open(o)?;
        let local: Vec<(usize, Endpoint)> = ends
            .iter()
            .zip(peers)
            .enumerate()
            .map(|(i, (e, &j))| {
                let to = if j == o.rank { 0 } else { j };
                (to, e.endpoint(psn(o.rank, i)))
            })
            .collect();
        let remote = control.exchange(&local, o.rank, [o.size, o.gather])?;
        for ((e, (_, l)), r) in ends.iter_mut().zip(&local).zip(&remote) {
            e.connect(l.psn, r)?;
        }
        println!(
            "link setup rank={} nodes={} dev={} gid_index={} page_bytes={} \
             access=local_write,remote_write size={} gather={} iters={} warmup={}",
            o.rank,
            o.nodes,
            o.dev,
            o.gid,
            page.bytes(),
            o.size,
            o.gather,
            o.iters,
            o.warmup
        );
        Ok((ends, control))
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

    /// What one node holds of the link: its page, its ends, the counters a
    /// stall report names and the deadline every wait is bounded by.
    struct Node<'a> {
        rank: usize,
        page: &'a MappedHost,
        ends: &'a [RdmaEnd],
        /// Each counter a stall report prints, by name and byte.
        watch: Vec<(String, usize)>,
        deadline: Duration,
    }

    impl Node<'_> {
        fn word(&self, off: usize) -> &AtomicU32 {
            self.page
                .atomic_u32(off)
                .expect("every counter and region word lies in the layout's page")
        }

        /// The first end that broke, by why.
        fn broken(&self) -> Option<String> {
            self.ends.iter().find_map(RdmaEnd::broken)
        }

        /// The stall's report at generation `g`: every watched counter as the
        /// host reads it, and any end's failure.
        fn stalled(&self, g: usize, why: &str) -> GateError {
            let words: Vec<String> = self
                .watch
                .iter()
                .map(|(name, off)| format!("{name}={}", self.word(*off).load(Ordering::Acquire)))
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
            RdmaEnd::release(self.ends, RELEASE).map_err(|e| e.to_string())
        }

        /// Wait for `event`, generation `g`'s, until `since` is a deadline
        /// old or an end broke.
        fn settle(&self, event: &CudaEvent, since: Instant, g: usize) -> Result<(), GateError> {
            let mut spins = 0u32;
            while !event.query()? {
                spins = spins.wrapping_add(1);
                if spins.is_multiple_of(1024) {
                    if self.broken().is_some() {
                        return Err(self.stalled(g, "the link broke"));
                    }
                    if since.elapsed() > self.deadline {
                        return Err(self.stalled(g, "no progress"));
                    }
                }
                std::hint::spin_loop();
            }
            Ok(())
        }

        /// Run `body`, which enqueues this node's card work on `gpu`'s stream
        /// and waits for it to drain. A failure on the way (a stall, a broken
        /// end, a driver error or a panic) releases the counters the stream
        /// waits on; a stream not seen to drain after that ends the process
        /// here, leaving the page and the device buffers it may still reach to
        /// the OS.
        fn drive(
            &self,
            gpu: &Gpu,
            body: impl FnOnce() -> Result<(), GateError>,
        ) -> Result<(), GateError> {
            let why = match std::panic::catch_unwind(AssertUnwindSafe(body)) {
                Ok(Ok(())) => return Ok(()),
                Ok(Err(e)) => e.to_string(),
                Err(_) => "the card thread panicked".to_string(),
            };
            // The release and the drain each wait on the link; why the run
            // failed is out first. `eprintln!` panics when stderr fails.
            let _ = writeln!(std::io::stderr(), "link_probe: {why}; releasing the link");
            match self.drain_released(gpu) {
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

        /// The end of a failed run: release every counter the stream waits
        /// on, then `Ok` once the stream drained, else why it is not known
        /// to have.
        fn drain_released(&self, gpu: &Gpu) -> Result<(), String> {
            self.release()?;
            let end = gpu
                .context()
                .new_event(None)
                .map_err(|e| format!("no end event: {e}"))?;
            end.record(gpu.stream())
                .map_err(|e| format!("end event not recorded: {e}"))?;
            let start = Instant::now();
            loop {
                match end.query() {
                    Ok(true) => return Ok(()),
                    Ok(false) if start.elapsed() <= self.deadline => std::hint::spin_loop(),
                    Ok(false) => {
                        return Err(format!("not drained within {:?}", self.deadline));
                    }
                    Err(e) => return Err(format!("end event query: {e}")),
                }
            }
        }
    }

    /// One node's side of the protocol, whoever plays it: its hold on the
    /// link, the layout and the peers its ends reach.
    struct Card<'a> {
        node: Node<'a>,
        layout: Layout,
        /// The peers this rank's ends reach: every peer on the stage, itself
        /// on a peer.
        peers: &'a [usize],
        total: u32,
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
                card.node.word(l.image - 4).store(g, Ordering::Relaxed);
                for &j in card.peers {
                    card.node.word(l.go(j)).fetch_add(1, Ordering::Release);
                }
                for &j in card.peers {
                    host_wait(card, l.done(j), g)?;
                    let last = l.rows_off(j) + l.rows - 4;
                    stale += u64::from(card.node.word(last).load(Ordering::Relaxed) != g);
                    card.node.word(l.done(j)).fetch_sub(1, Ordering::AcqRel);
                }
                one_way.push(start.elapsed().as_nanos() as u64 / 2);
            } else {
                host_wait(card, l.go(o.rank), g)?;
                stale += u64::from(card.node.word(l.image - 4).load(Ordering::Relaxed) != g);
                card.node.word(l.go(o.rank)).fetch_sub(1, Ordering::AcqRel);
                let last = l.rows_off(o.rank) + l.rows - 4;
                card.node.word(last).store(g, Ordering::Relaxed);
                card.node
                    .word(l.done(o.rank))
                    .fetch_add(1, Ordering::Release);
            }
            Ok::<(), GateError>(())
        });
        let played = played.and_then(|()| match card.node.broken() {
            Some(b) => Err(format!("the link broke: {b}").into()),
            None => Ok(()),
        });
        if let Err(e) = played {
            // The release waits on the link; why the run failed is out first.
            // `eprintln!` panics when stderr fails.
            let _ = writeln!(std::io::stderr(), "link_probe: {e}; releasing the link");
            if let Err(held) = card.node.release() {
                // The NIC may still read the page: exit before the ends and
                // the page drop.
                let _ = writeln!(
                    std::io::stderr(),
                    "link_probe: {held}; exiting without teardown"
                );
                std::process::exit(1);
            }
            return Err(e);
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
        while card.node.word(off).load(Ordering::Acquire) < 1 {
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(1024) {
                if card.node.broken().is_some() {
                    return Err(card.node.stalled(g as usize, "the link broke"));
                }
                if start.elapsed() > card.node.deadline {
                    return Err(card.node.stalled(g as usize, "no progress"));
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
            let (seen_off, seen_bytes, put_off, put_bytes) = if self.card.node.rank == 0 {
                (
                    l.rows_off(1),
                    l.rows_off(l.nodes) - l.rows_off(1),
                    0,
                    l.image,
                )
            } else {
                let r = self.card.node.rank;
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

        /// Every generation, then a drained stream ([`Node::drive`]).
        fn drive(
            &self,
            stale: &mut DeviceBuffer<u64>,
            stamps: &mut DeviceBuffer<u64>,
        ) -> Result<(), GateError> {
            self.card
                .node
                .drive(self.gpu, || self.generations(stale, stamps))
        }

        /// Enqueue generations 1..=total, at most [`AHEAD`] ahead of the
        /// host, and wait for the stream to drain.
        fn generations(
            &self,
            stale: &mut DeviceBuffer<u64>,
            stamps: &mut DeviceBuffer<u64>,
        ) -> Result<(), GateError> {
            let ctx = self.gpu.context();
            let stream = self.gpu.stream();
            let (l, card) = (self.card.layout, self.card);
            let at = |off: usize| self.dev + off as u64;
            let total = card.total as usize;
            replays(
                |g| {
                    let seq = u32::try_from(g)?;
                    if card.node.rank == 0 {
                        self.launch(stale, stamps, seq - 1, seq, l.image / LINE, g - 1)?;
                        let mut ops = vec![op_barrier_sys()];
                        ops.extend(card.peers.iter().map(|&j| op_add(at(l.go(j)), 1)));
                        for &j in card.peers {
                            ops.push(op_wait_geq(at(l.done(j)), 1));
                            ops.push(op_add(at(l.done(j)), u32::MAX));
                        }
                        // SAFETY: every counter lies in the page (the layout's
                        // own words), which outlives every batch as in `span`.
                        unsafe {
                            mem_batch(stream, &mut ops, "link_probe: the gos and the dones")?
                        };
                    } else {
                        let r = card.node.rank;
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
                    Ok(())
                },
                self.gpu,
                &card.node,
                total,
            )?;
            if card.node.rank == 0 {
                self.launch(stale, stamps, card.total, 0, 0, total)?;
            }
            let end = ctx.new_event(None)?;
            end.record(stream)?;
            card.node.settle(&end, Instant::now(), total)
        }
    }

    // ------------------------------------------------------------ the gather

    /// Gather `g`'s slot and set: a layer's gathers take the body slot's
    /// sets in turn, and the head's, between two replays' layer gathers,
    /// keeps the head slot's first set.
    fn slot_set(g: usize) -> (usize, usize) {
        if g + 1 < GATHERS {
            (0, g % Gathers::SETS)
        } else {
            (1, 0)
        }
    }

    /// `--gather`'s run on this node: the page of a body and a head slot of
    /// `rows` rows, an end to every other node, the chain captured, `warmup +
    /// iters` replays, then this node's line and, on the stage, the verdict
    /// over every node.
    fn gather_run(o: &Opts, gpu: &Gpu) -> Result<(), GateError> {
        let rows = o.gather;
        let shapes = [BODY_WIDTH, HEAD_WIDTH].map(|width| GatherShape { rows, width });
        let g = Gathers::new(0, o.nodes, o.rank, &shapes)?;
        let ctx = gpu.context();
        let page = Arc::new(MappedHost::new(
            ctx,
            g.page_bytes(),
            "cuMemHostAlloc (the probe's gather page)",
        )?);
        let cols = PostCols::default();
        cols.set(rows);
        let peers: Vec<usize> = (0..o.nodes).filter(|&j| j != o.rank).collect();
        let (ends, mut control) = open_ends(
            o,
            &page,
            &peers,
            |j| (g.tail(j), g.crossings(j), g.side(j)),
            &cols,
        )?;
        let mut watch = Vec::new();
        for slot in 0..shapes.len() {
            for set in 0..Gathers::SETS {
                for r in 0..o.nodes {
                    watch.push((format!("gather[{slot}.{set}.{r}]"), g.counter(slot, set, r)));
                }
            }
        }
        let node = Node {
            rank: o.rank,
            page: &page,
            ends: &ends,
            watch,
            deadline: o.deadline,
        };
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let kernels = unsafe { bloomery_gpu::shared_module!(gather_kernels, ctx)? };
        let stream = gpu.stream();
        let total = o.total as usize;
        let n = rows * BODY_WIDTH;
        let mut seq = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        let mut stamps = DeviceBuffer::<u64>::zeroed(stream, 2 * GATHERS * total)?;
        let mut stale = DeviceBuffer::<u32>::zeroed(stream, n)?;
        let mut digest = DeviceBuffer::<u32>::zeroed(stream, n)?;
        let mut sums = [
            DeviceBuffer::<f32>::zeroed(stream, rows * BODY_WIDTH)?,
            DeviceBuffer::<f32>::zeroed(stream, rows * HEAD_WIDTH)?,
        ];
        let dev = page.dev_at(0);
        let steps = (0..GATHERS)
            .map(|gi| step(&g, gi, rows, (o.rank, o.nodes), dev))
            .collect::<Result<Vec<_>, _>>()?;
        let (rank, nodes) = (u32::try_from(o.rank)?, u32::try_from(o.nodes)?);
        let gathers = u32::try_from(GATHERS)?;
        let graph = gpu.capture(|s| {
            let one = LaunchConfig1D::new(1, THREADS, 0);
            kernels.begin(s, &kernels.prepare_begin(one)?, &mut seq)?;
            let mark = kernels.prepare_mark(one)?;
            for st in &steps {
                let cfg = LaunchConfig1D::new(st.values.div_ceil(THREADS), THREADS, 0);
                // SAFETY: this rank's part of the gather lies in the page,
                // which is mapped into this context and outlives every replay:
                // the run drains the stream before the page drops, or ends the
                // process without dropping it.
                let mut part =
                    unsafe { WindowMut::<f32>::from_raw(st.own, st.values as usize, ctx) };
                let prep = kernels.prepare_part(cfg)?;
                kernels.part(s, &prep, &seq, st.gather, rank, st.values, &mut part)?;
                kernels.mark(s, &mark, &seq, st.gather, gathers, 0, &mut stamps)?;
                let at = GatherSet {
                    dev,
                    slot: st.slot,
                    set: st.set,
                };
                let mut ops = g.post(at).to_vec();
                ops.extend(g.wait(at));
                // SAFETY: every counter lies in the page, which outlives every
                // replay as the part does.
                unsafe { mem_batch(s, &mut ops, "link_probe: a gather's post and wait")? };
                kernels.mark(s, &mark, &seq, st.gather, gathers, 1, &mut stamps)?;
                // SAFETY: every rank's part of the set lies in the page, which
                // outlives every replay as the part does.
                let parts = unsafe { WindowMut::<f32>::from_raw(st.parts, st.span, ctx) };
                kernels.check(
                    s,
                    &kernels.prepare_check(cfg)?,
                    &seq,
                    st.gather,
                    rank,
                    nodes,
                    &parts,
                    st.stride,
                    st.values,
                    &mut stale,
                )?;
                let sum = &mut sums[st.slot];
                // SAFETY: `dev` is the page of this layout, mapped into this
                // context and outliving every replay as the part does.
                unsafe { g.enqueue_sum(gpu.elem(), s, at, rows, sum)? };
                let prep = kernels.prepare_digest(cfg)?;
                kernels.digest(s, &prep, &*sum, st.values, &mut digest)?;
            }
            Ok(())
        })?;
        control.ready()?;
        node.drive(gpu, || {
            replays(|_| Ok(graph.launch(gpu.stream())?), gpu, &node, total)
        })?;
        let stale_here: u64 = stale
            .to_host_vec(stream)?
            .iter()
            .map(|&c| u64::from(c))
            .sum();
        let digest_here = Fnv1a64::default()
            .u32s(&digest.to_host_vec(stream)?)
            .value();
        let wrong = host_wrong(&sums, stream, rows, o.nodes, o.total)?;
        let stamps = stamps.to_host_vec(stream)?;
        let lambda = |k: usize, gi: usize| {
            let at = 2 * (GATHERS * k + gi);
            stamps[at + 1] - stamps[at]
        };
        let mut per_gather: Vec<u64> = (o.warmup..total)
            .flat_map(|k| (0..GATHERS).map(move |gi| (k, gi)))
            .map(|(k, gi)| lambda(k, gi))
            .collect();
        let mut per_chain: Vec<u64> = (o.warmup..total)
            .map(|k| (0..GATHERS).map(|gi| lambda(k, gi)).sum())
            .collect();
        let us = |ns: u64| ns as f64 / 1e3;
        let [g50, g90, g99] = [50.0, 90.0, 99.0].map(|p| us(pct(&mut per_gather, p)));
        let [c50, c90, c99] = [50.0, 90.0, 99.0].map(|p| us(pct(&mut per_chain, p)));
        Record::new(&LINK_GATHER)
            .u("rank", o.rank)
            .u("nodes", o.nodes)
            .u("rows", rows)
            .u("gathers", GATHERS)
            .u("iters", o.iters)
            .f("lambda_us_p50", g50)
            .f("lambda_us_p90", g90)
            .f("lambda_us_p99", g99)
            .f("lambda_us_max", us(per_gather[per_gather.len() - 1]))
            .f("chain_us_p50", c50)
            .f("chain_us_p90", c90)
            .f("chain_us_p99", c99)
            .u("stale_words", stale_here)
            .u("host_wrong", wrong)
            .w("digest", format!("{digest_here:#018x}"))
            .print();
        if o.rank != 0 {
            control.report(stale_here + wrong, digest_here)?;
            control.ready()?;
            return Ok(());
        }
        let reports = control.collect()?;
        control.ready()?;
        let bad = stale_here + wrong + reports.iter().map(|r| r.0).sum::<u64>();
        let same = reports.iter().all(|r| r.1 == digest_here);
        let pass = bad == 0 && same;
        println!(
            "{}: link_probe gather stale words and host disagreements {bad}, digests equal on \
             every node {same}",
            verdict(pass)
        );
        if pass {
            Ok(())
        } else {
            Err("a stale word, a sum the host disagrees with, or digests that differ".into())
        }
    }

    /// One gather of the chain as this rank enqueues it.
    struct Step {
        gather: u32,
        slot: usize,
        set: usize,
        /// Values of a part the gather carries.
        values: u32,
        /// Device addresses of this rank's part and of rank 0's, the first
        /// of the set's parts, which span `span` values, rank r's from value
        /// r `stride`.
        own: sys::CUdeviceptr,
        parts: sys::CUdeviceptr,
        span: usize,
        stride: u32,
    }

    /// Gather `gi` of the chain, `rows` rows, as rank `rank` of `ranks`
    /// enqueues it over its page at `dev`.
    fn step(
        g: &Gathers,
        gi: usize,
        rows: usize,
        (rank, ranks): (usize, usize),
        dev: sys::CUdeviceptr,
    ) -> Result<Step, GateError> {
        let (slot, set) = slot_set(gi);
        let values = rows * [BODY_WIDTH, HEAD_WIDTH][slot];
        let first = g.part(slot, set, 0).start;
        let stride = (g.part(slot, set, 1).start - first) / 4;
        Ok(Step {
            gather: u32::try_from(gi)?,
            slot,
            set,
            values: u32::try_from(values)?,
            own: dev + g.part(slot, set, rank).start as u64,
            parts: dev + first as u64,
            span: stride * (ranks - 1) + values,
            stride: u32::try_from(stride)?,
        })
    }

    /// Enqueue 1..=total through `enqueue`, at most [`AHEAD`] ahead of the
    /// host, and wait for the stream to drain.
    fn replays(
        mut enqueue: impl FnMut(usize) -> Result<(), GateError>,
        gpu: &Gpu,
        node: &Node<'_>,
        total: usize,
    ) -> Result<(), GateError> {
        let ctx = gpu.context();
        ctx.bind_to_thread()?;
        let stream = gpu.stream();
        let events = (0..AHEAD)
            .map(|_| ctx.new_event(None))
            .collect::<Result<Vec<_>, _>>()?;
        let mut progress = Instant::now();
        for k in 1..=total {
            if k > AHEAD {
                node.settle(&events[k % AHEAD], progress, k - AHEAD)?;
                progress = Instant::now();
            }
            enqueue(k)?;
            events[k % AHEAD].record(stream)?;
        }
        let end = ctx.new_event(None)?;
        end.record(stream)?;
        node.settle(&end, Instant::now(), total)
    }

    /// The values of the last replay's sums, the layer gathers' last and the
    /// head's, that are not the host's rank-order sum of the parts.
    fn host_wrong(
        sums: &[DeviceBuffer<f32>; 2],
        stream: &CudaStream,
        rows: usize,
        ranks: usize,
        replay: u32,
    ) -> Result<u64, GateError> {
        let mut wrong = 0u64;
        for (gi, sum) in [GATHERS - 2, GATHERS - 1].into_iter().zip(sums) {
            let got = sum.to_host_vec(stream)?;
            let (gu, n) = (
                u32::try_from(gi)?,
                rows * [BODY_WIDTH, HEAD_WIDTH][slot_set(gi).0],
            );
            for (i, v) in got[..n].iter().enumerate() {
                let i = u32::try_from(i)?;
                let want = (1..ranks).fold(value(replay, gu, 0, i), |acc, r| {
                    acc + value(replay, gu, r as u32, i)
                });
                wrong += u64::from(v.to_bits() != want.to_bits());
            }
        }
        Ok(wrong)
    }

    // ------------------------------------------------------ control channel

    /// The stage's TCP connections to its peers (one, to the stage, on a
    /// peer): the endpoints, `--size` and `--gather`, the start, and the
    /// peers' counts at the end.
    struct Control {
        socks: Vec<TcpStream>,
        /// Whether this node is the stage, the one end of every connection.
        stage: bool,
        deadline: Duration,
    }

    /// One endpoint as the control channel carries it: the rank whose end it
    /// is, the rank whose end connects to it, the endpoint, and the sender's
    /// `--size` and `--gather`.
    const FRAME_BYTES: usize = 60;

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
                stage: o.rank == 0,
                deadline,
            })
        }

        /// Trade endpoints: `local` holds this node's end to each peer (its
        /// rank and endpoint), and the result holds, in the same order, each
        /// peer's end to this node. A peer sends the stage its count and its
        /// frames; the stage sends each peer every frame that names it, its
        /// own included, so two peers learn each other's ends through it.
        /// Every frame carries its sender's `--size` and `--gather`, and a
        /// node that reads another value refuses it by name before any
        /// generation.
        fn exchange(
            &mut self,
            local: &[(usize, Endpoint)],
            rank: usize,
            shape: [usize; 2],
        ) -> Result<Vec<Endpoint>, GateError> {
            let shape = shape.map(|v| v as u64);
            let own: Vec<Frame> = local
                .iter()
                .map(|&(to, e)| Frame {
                    from: rank,
                    to,
                    e,
                    shape,
                })
                .collect();
            let mut mine = Vec::new();
            if self.stage {
                let mut all = own;
                for s in &mut self.socks {
                    all.extend(read_frames(s, shape)?);
                }
                for (i, s) in self.socks.iter_mut().enumerate() {
                    write_frames(s, all.iter().filter(|f| f.to == i + 1))?;
                }
                mine.extend(all.into_iter().filter(|f| f.to == 0));
            } else {
                let s = &mut self.socks[0];
                write_frames(s, own.iter())?;
                mine = read_frames(s, shape)?;
            }
            local
                .iter()
                .map(|&(peer, _)| {
                    let mut of = mine.iter().filter(|f| f.from == peer && f.to == rank);
                    match (of.next(), of.next()) {
                        (Some(f), None) => Ok(f.e),
                        _ => Err(format!("rank {rank}: not one end of rank {peer} to it").into()),
                    }
                })
                .collect()
        }

        /// A barrier, a byte each way: a peer writes, then reads; the stage
        /// reads every peer's byte before it writes any. Before the first
        /// generation, every node's queue pairs are ready, those between two
        /// peers included; after the last, the stage has every peer's done,
        /// so no peer drops its end (its pump and queue pair) with a done
        /// still to post.
        fn ready(&mut self) -> Result<(), GateError> {
            let read = |s: &mut TcpStream| s.read_exact(&mut [0u8; 1]);
            if self.stage {
                self.socks.iter_mut().try_for_each(read)?;
                self.socks.iter_mut().try_for_each(|s| s.write_all(b"R"))?;
            } else {
                self.socks.iter_mut().try_for_each(|s| s.write_all(b"R"))?;
                self.socks.iter_mut().try_for_each(read)?;
            }
            Ok(())
        }

        /// A peer's stale count and, by mode, its kernel's median time or its
        /// sums' digest, to the stage.
        fn report(&mut self, stale: u64, second: u64) -> Result<(), GateError> {
            let mut b = [0u8; 16];
            b[0..8].copy_from_slice(&stale.to_le_bytes());
            b[8..16].copy_from_slice(&second.to_le_bytes());
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

    /// An endpoint on the control channel ([`FRAME_BYTES`]).
    struct Frame {
        from: usize,
        to: usize,
        e: Endpoint,
        shape: [u64; 2],
    }

    /// Write the count of `frames`, then each.
    fn write_frames<'a>(
        s: &mut TcpStream,
        frames: impl Iterator<Item = &'a Frame> + Clone,
    ) -> Result<(), GateError> {
        s.write_all(&u32::try_from(frames.clone().count())?.to_le_bytes())?;
        for f in frames {
            let mut b = [0u8; FRAME_BYTES];
            b[0..4].copy_from_slice(&u32::try_from(f.from)?.to_le_bytes());
            b[4..8].copy_from_slice(&u32::try_from(f.to)?.to_le_bytes());
            b[8..12].copy_from_slice(&f.e.qpn.to_le_bytes());
            b[12..16].copy_from_slice(&f.e.psn.to_le_bytes());
            b[16..32].copy_from_slice(&f.e.gid.raw);
            b[32..40].copy_from_slice(&f.e.addr.to_le_bytes());
            b[40..44].copy_from_slice(&f.e.rkey.to_le_bytes());
            b[44..52].copy_from_slice(&f.shape[0].to_le_bytes());
            b[52..60].copy_from_slice(&f.shape[1].to_le_bytes());
            s.write_all(&b)?;
        }
        Ok(())
    }

    /// Read a count of frames, then each; a frame whose `--size` and
    /// `--gather` are not `shape` is refused by name.
    fn read_frames(s: &mut TcpStream, shape: [u64; 2]) -> Result<Vec<Frame>, GateError> {
        let mut n = [0u8; 4];
        s.read_exact(&mut n)
            .map_err(|e| format!("the endpoint count: {e}"))?;
        (0..u32::from_le_bytes(n))
            .map(|_| {
                let mut b = [0u8; FRAME_BYTES];
                s.read_exact(&mut b)
                    .map_err(|e| format!("an endpoint: {e}"))?;
                let w32 =
                    |at: usize| u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"));
                let w64 =
                    |at: usize| u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"));
                let from = w32(0) as usize;
                let theirs = [w64(44), w64(52)];
                if theirs != shape {
                    return Err(format!(
                        "--size and --gather differ: here {shape:?}, rank {from} {theirs:?}"
                    )
                    .into());
                }
                let mut gid = Gid::default();
                gid.raw.copy_from_slice(&b[16..32]);
                Ok(Frame {
                    from,
                    to: w32(4) as usize,
                    e: Endpoint {
                        qpn: w32(8),
                        psn: w32(12),
                        gid,
                        addr: w64(32),
                        rkey: w32(40),
                    },
                    shape: theirs,
                })
            })
            .collect()
    }
}
