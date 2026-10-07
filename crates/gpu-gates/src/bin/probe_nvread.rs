//! `probe_nvread` — how fast the page cache reads cold routed experts of a
//! Qwen3.8 split set, beside an `O_DIRECT` read of the same bytes. Host only,
//! no card.
//!
//!     probe_nvread --range-from <id> [--ks 1,2,3] [--batches N] [--warmup W]
//!                  [--threads T] [--range-batches R] [--seed S] [--lease]
//!
//! The model is the file the gates open (`BLOOMERY_REF_MODEL`, which
//! `just time-nvread` exports under the qwen4exp profile). Each layer's three
//! routed stacks (gate, up, down) open as row tables whose rows are experts
//! ([`engram::rows::SplitTable`]); a *run* is one expert's three parts, one in
//! each stack of its layer.
//!
//! A *case* is a kind of batch. `k1`, `k2`, `k3` (`--ks`) are batches of that
//! many runs, each from a layer of its own (a seeded draw, so no batch's
//! readahead warms another's); `range` (`--range-from`, required) is a batch
//! of every expert from that id to the end of one layer, over its three
//! stacks, the prompt's case. A case runs `--warmup` batches that are not
//! counted and then `--batches` that are (`--range-batches` for `range`).
//!
//! A batch, in order:
//!   1. drop its runs (`MADV_DONTNEED`, then `posix_fadvise(DONTNEED)`, both
//!      through engram) and read `mincore` over them: any resident page and
//!      the batch is redrawn and counted (`redraws`); past a bound the run
//!      ends by name;
//!   2. the `cache` arm: from the first `WILLNEED` over every part to the last
//!      part populated (`POPULATE_READ`, the parts split across `--threads`
//!      workers); then `mincore` must report every page resident, or the run
//!      ends by name;
//!   3. drop the runs again, so the next arms read what a cold cache holds;
//!   4. the `direct1` arm: an `O_DIRECT` pread of each part, on one thread,
//!      the file range rounded out to the alignment; the `directN` arm: the
//!      same preads split across the same workers as the cache arm; in the
//!      range case, the `drive` arm: the same bytes as each stack's one
//!      contiguous run, read on one thread in requests large enough that the
//!      drive's queue stays full: the drive's own rate, which the part-sized
//!      preads of a decode batch cannot reach at one request in flight.
//!
//! The cache arm runs first on bytes nothing has read, so its rate is never
//! one read after an `O_DIRECT` read of the same bytes. Rates are bytes the
//! batch's parts hold over the arm's wall, in GB/s of 10^9 B; the row also
//! carries the page-rounded bytes and every byte the arm asked of the drive.
//! A row is `nvread arm=<arm> case=<case> …` ([`bloomery_gpu_gates::record`]);
//! `--lease` says the run holds the machine lease (`tools/ref/nvtier-read.sh`
//! passes it), and a run without it prints `lease=false`: functional output,
//! not a measurement.
//!
//! Before the first batch the probe proves its instrument: after an eviction
//! `mincore` finds nothing of an expert resident, after a populate everything.
//! An undefined input (a file of another architecture, a stack missing from a
//! layer, an `--range-from` past the experts) is a named error.

use std::collections::HashMap;
use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Instant;

use bloomery_gpu_gates::record::{self, NVREAD, PROBE_NVREAD, Record};
use bloomery_gpu_gates::{GateError, exit_with, ref_model_path};
use engram::rows::{PageAdvice, SplitTable};
use engram::{DIRECT_ALIGN, RowTable, open_direct};
use gguf::Split;

/// The `general.architecture` the probe reads.
const ARCH: &str = "qwen4exp";
/// A layer's routed stacks, in the order a run's parts are read.
const STACKS: [&str; 3] = ["gate", "up", "down"];

fn main() -> std::process::ExitCode {
    exit_with("probe_nvread", run())
}

fn run() -> Result<(), GateError> {
    bloomery_levers::at_main(&[])?;
    record::at_main("probe_nvread", PROBE_NVREAD);
    let a = Args::parse(std::env::args().skip(1))?;
    let mut rig = Rig::open(a.threads)?;
    eprintln!(
        "probe_nvread: {} layers x {} experts, {} threads, seed {}, lease {}",
        rig.layers, rig.experts, a.threads, a.seed, a.lease
    );
    if !a.lease {
        eprintln!(
            "probe_nvread: not under the lease (rows carry lease=false): functional output, not a measurement"
        );
    }
    if a.range_from >= rig.experts {
        return Err(format!(
            "--range-from {} is not below the {} experts of a layer",
            a.range_from, rig.experts
        )
        .into());
    }
    if a.range_batches + a.warmup > rig.layers {
        return Err(format!(
            "the range case draws each of its {} batches from a layer of its own, and the model has {} layers",
            a.range_batches + a.warmup,
            rig.layers
        )
        .into());
    }
    rig.prove_instrument()?;
    for &k in &a.ks {
        if k > rig.layers {
            return Err(format!(
                "--ks {k}: a batch draws from distinct layers, and the model has {}",
                rig.layers
            )
            .into());
        }
        let mut draw = Draw(a.seed ^ ((k as u64) << 48));
        rig.case(&a, Shape::Runs(k), &mut draw)?;
    }
    let mut draw = Draw(a.seed ^ 0xA5A5_0000_0000_0000);
    rig.case(&a, Shape::Range(a.range_from), &mut draw)
}

// ------------------------------------------------------------- arguments

struct Args {
    ks: Vec<usize>,
    batches: usize,
    warmup: usize,
    threads: usize,
    range_from: usize,
    range_batches: usize,
    seed: u64,
    lease: bool,
}

impl Args {
    fn parse(args: impl Iterator<Item = String>) -> Result<Args, GateError> {
        let mut a = Args {
            ks: vec![1, 2, 3],
            batches: 64,
            warmup: 2,
            threads: 8,
            range_from: usize::MAX,
            range_batches: 16,
            seed: 0x6E76_7265_6164,
            lease: false,
        };
        let mut it = args;
        while let Some(flag) = it.next() {
            if flag == "--lease" {
                a.lease = true;
                continue;
            }
            let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
            match flag.as_str() {
                "--ks" => {
                    a.ks = value
                        .split(',')
                        .map(|k| number(&flag, k))
                        .collect::<Result<_, _>>()?;
                }
                "--batches" => a.batches = number(&flag, &value)?,
                "--warmup" => a.warmup = number(&flag, &value)?,
                "--threads" => a.threads = number(&flag, &value)?,
                "--range-from" => a.range_from = number(&flag, &value)?,
                "--range-batches" => a.range_batches = number(&flag, &value)?,
                "--seed" => {
                    a.seed = match value.strip_prefix("0x") {
                        Some(hex) => u64::from_str_radix(hex, 16),
                        None => value.parse(),
                    }
                    .map_err(|e| format!("--seed {value}: {e}"))?;
                }
                other => return Err(format!("unknown flag {other}").into()),
            }
        }
        if a.range_from == usize::MAX {
            return Err(
                "--range-from <id> is required: the first expert id of the range case".into(),
            );
        }
        if a.ks.is_empty() || a.ks.contains(&0) {
            return Err("--ks names batch sizes of at least one run".into());
        }
        for (flag, n) in [
            ("--batches", a.batches),
            ("--threads", a.threads),
            ("--range-batches", a.range_batches),
        ] {
            if n == 0 {
                return Err(format!("{flag} is at least 1").into());
            }
        }
        Ok(a)
    }
}

fn number(flag: &str, text: &str) -> Result<usize, GateError> {
    text.trim()
        .parse()
        .map_err(|e| format!("{flag} {text}: {e}").into())
}

// ------------------------------------------------------------------ draws

/// splitmix64: the same draw on every run of a seed.
struct Draw(u64);

impl Draw {
    fn step(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n`.
    fn below(&mut self, n: usize) -> usize {
        (self.step() % n as u64) as usize
    }
}

/// `k` distinct layers of `layers`: the first `k` of a seeded shuffle.
fn distinct_layers(draw: &mut Draw, layers: usize, k: usize) -> Vec<usize> {
    let mut all: Vec<usize> = (0..layers).collect();
    for i in 0..k {
        let j = i + draw.below(layers - i);
        all.swap(i, j);
    }
    all.truncate(k);
    all
}

// ------------------------------------------------------------------ pool

/// A job a worker runs, with that worker's scratch buffer.
type Job = Box<dyn FnOnce(&mut Vec<u8>) -> Result<(), String> + Send>;

/// Workers that stay up for the run, one channel each, so a batch's wall holds
/// the dispatch and the joins and not a thread's start.
struct Pool {
    jobs: Vec<Sender<Job>>,
    done: Receiver<Result<(), String>>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    fn new(n: usize) -> Pool {
        let (done_tx, done) = channel();
        let mut jobs = Vec::with_capacity(n);
        let mut workers = Vec::with_capacity(n);
        for _ in 0..n {
            let (tx, rx): (Sender<Job>, Receiver<Job>) = channel();
            let done_tx = done_tx.clone();
            workers.push(std::thread::spawn(move || {
                let mut scratch = Vec::new();
                while let Ok(job) = rx.recv() {
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        job(&mut scratch)
                    }))
                    .unwrap_or_else(|_| Err("a pool worker's job panicked".to_string()));
                    if done_tx.send(r).is_err() {
                        return;
                    }
                }
            }));
            jobs.push(tx);
        }
        Pool {
            jobs,
            done,
            workers,
        }
    }

    fn threads(&self) -> usize {
        self.jobs.len()
    }

    /// Job `i` on worker `i`; returns when every job is done, with the first
    /// error any returned.
    fn run(&self, jobs: Vec<Job>) -> Result<(), GateError> {
        if jobs.len() > self.jobs.len() {
            return Err(format!("{} jobs for {} workers", jobs.len(), self.jobs.len()).into());
        }
        let n = jobs.len();
        for (tx, job) in self.jobs.iter().zip(jobs) {
            tx.send(job).map_err(|_| "a pool worker is gone")?;
        }
        let mut first = None;
        for _ in 0..n {
            match self.done.recv() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    first.get_or_insert(e);
                }
                Err(_) => return Err("the pool's workers ended".into()),
            }
        }
        first.map_or(Ok(()), |e| Err(e.into()))
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.jobs.clear();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// `len` items in `n` contiguous ranges, sizes within one of each other; as
/// many ranges as there are items when `n` is more.
fn partition(len: usize, n: usize) -> Vec<Range<usize>> {
    let n = n.min(len);
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    let (base, extra) = (len / n, len % n);
    let mut at = 0;
    for i in 0..n {
        let width = base + usize::from(i < extra);
        out.push(at..at + width);
        at += width;
    }
    out
}

// ------------------------------------------------------------ statistics

/// The median of an ascending sample.
fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// The nearest-rank percentile `p` (0..=1) of an ascending sample.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = (p * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

// ----------------------------------------------------------- direct reads

/// One `O_DIRECT` read: the aligned file range that holds a part, and how
/// many of its bytes the part needs.
#[derive(Clone, Copy)]
struct Read {
    table: usize,
    start: u64,
    span: usize,
    need: usize,
}

/// The aligned read of `len` bytes at `at`: the start rounded down to the
/// alignment, the span to the end rounded up, and the bytes from the start
/// through the part's last byte.
fn aligned_span(at: u64, len: u64) -> (u64, usize, usize) {
    let a = DIRECT_ALIGN as u64;
    let start = at / a * a;
    let end = (at + len).div_ceil(a) * a;
    (start, (end - start) as usize, (at + len - start) as usize)
}

/// `r` read into the aligned part of `scratch`, which grows to hold it.
fn read_direct(file: &File, r: &Read, scratch: &mut Vec<u8>) -> Result<(), String> {
    let want = r.span + DIRECT_ALIGN;
    if scratch.len() < want {
        scratch.resize(want, 0);
    }
    let off = scratch.as_ptr().align_offset(DIRECT_ALIGN);
    if off >= DIRECT_ALIGN {
        return Err(format!(
            "no {DIRECT_ALIGN}-aligned start in the scratch buffer (offset {off})"
        ));
    }
    let buf = &mut scratch[off..off + r.span];
    if !(buf.as_ptr() as usize).is_multiple_of(DIRECT_ALIGN)
        || !r.start.is_multiple_of(DIRECT_ALIGN as u64)
    {
        return Err(format!(
            "an O_DIRECT read needs the buffer and the file offset {} aligned to {DIRECT_ALIGN}",
            r.start
        ));
    }
    let n = file
        .read_at(buf, r.start)
        .map_err(|e| format!("pread {} B at {}: {e}", r.span, r.start))?;
    if n < r.need {
        return Err(format!(
            "short pread at {}: {n} B of the {} B the part needs",
            r.start, r.need
        ));
    }
    Ok(())
}

/// The request size of the `drive` arm: large enough that one thread keeps
/// the drive's queue full with a single request in flight.
const DRIVE_CHUNK: u64 = 4 << 20;

/// The aligned [`DRIVE_CHUNK`]-byte reads that cover `at .. end` of a table's
/// shard: the first starts at `at` rounded down, the last ends at `end`
/// rounded up and needs the bytes to `end`.
fn chunked(table: usize, at: u64, end: u64) -> Vec<Read> {
    let a = DIRECT_ALIGN as u64;
    let last = end.div_ceil(a) * a;
    let mut reads = Vec::new();
    let mut start = at / a * a;
    while start < last {
        let stop = (start + DRIVE_CHUNK).min(last);
        reads.push(Read {
            table,
            start,
            span: (stop - start) as usize,
            need: (end.min(stop) - start) as usize,
        });
        start = stop;
    }
    reads
}

// -------------------------------------------------------------------- rig

/// One part of a run: a table of the rig and an expert.
#[derive(Clone, Copy)]
struct Part {
    t: usize,
    id: u32,
}

/// A case's batches.
#[derive(Clone, Copy)]
enum Shape {
    /// This many runs, each from a layer of its own.
    Runs(usize),
    /// One layer's experts from this id on.
    Range(usize),
}

/// The arms, in the order a batch runs and a case prints them; the last runs
/// in the range case only.
const ARMS: [&str; 4] = ["cache", "direct1", "directN", "drive"];

/// One batch's measure.
struct Sample {
    /// The bytes the parts hold.
    bytes: u64,
    /// Each arm's wall in ns, and the bytes it asked of the drive (the cache
    /// arm's are the pages the parts span, as `mincore` counted them); 0 and
    /// 0 for an arm the batch did not run.
    ns: [u64; 4],
    asked: [u64; 4],
}

struct Rig {
    /// Layer `l`'s stacks are tables `3 l ..= 3 l + 2`, in [`STACKS`] order.
    tables: Arc<Vec<SplitTable>>,
    /// Each table's shard, opened `O_DIRECT`.
    files: Arc<Vec<Arc<File>>>,
    pool: Pool,
    /// The single-thread arm's buffer.
    scratch: Vec<u8>,
    layers: usize,
    experts: usize,
}

impl Rig {
    fn open(threads: usize) -> Result<Rig, GateError> {
        let path = ref_model_path()?;
        let split =
            Arc::new(Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?);
        if split.architecture() != Some(ARCH) {
            return Err(format!(
                "the model file is {:?}, want {ARCH} — run through `just time-nvread`, which picks the {ARCH} profile",
                split.architecture()
            )
            .into());
        }
        let layers = split
            .arch_get_u64("block_count")
            .ok_or_else(|| format!("{}: no {}", path.display(), split.arch_key("block_count")))?
            as usize;
        let experts = split
            .arch_get_u64("expert_count")
            .ok_or_else(|| format!("{}: no {}", path.display(), split.arch_key("expert_count")))?
            as usize;
        let mut tables = Vec::with_capacity(layers * STACKS.len());
        for layer in 0..layers {
            for stack in STACKS {
                let name = format!("blk.{layer}.ffn_{stack}_exps.weight");
                let t = SplitTable::open_with(Arc::clone(&split), &name, PageAdvice::Normal)
                    .map_err(|e| format!("{name}: {e}"))?;
                if t.rows() != experts as u64 {
                    return Err(format!(
                        "{name}: {} experts, the metadata says {experts}",
                        t.rows()
                    )
                    .into());
                }
                tables.push(t);
            }
        }
        let mut handles: HashMap<PathBuf, Arc<File>> = HashMap::new();
        let mut files = Vec::with_capacity(tables.len());
        for t in &tables {
            let f = match handles.get(t.path()) {
                Some(f) => Arc::clone(f),
                None => {
                    let f = Arc::new(open_direct(t.path())?);
                    handles.insert(t.path().to_path_buf(), Arc::clone(&f));
                    f
                }
            };
            files.push(f);
        }
        let pool = Pool::new(threads);
        // Every buffer holds the largest part's aligned span before the first
        // batch, so no timed read grows or faults one in.
        let widest = tables
            .iter()
            .map(|t| t.row_bytes())
            .max()
            .unwrap_or(0)
            .max(DRIVE_CHUNK) as usize;
        let buffer = widest + 3 * DIRECT_ALIGN;
        let prime: Vec<Job> = (0..threads)
            .map(|_| {
                Box::new(move |s: &mut Vec<u8>| -> Result<(), String> {
                    s.resize(buffer, 0);
                    Ok(())
                }) as Job
            })
            .collect();
        pool.run(prime)?;
        Ok(Rig {
            tables: Arc::new(tables),
            files: Arc::new(files),
            pool,
            scratch: vec![0; buffer],
            layers,
            experts,
        })
    }

    /// The parts of a run, in the order they are read.
    fn run_parts(&self, layer: usize, id: usize) -> [Part; 3] {
        [0, 1, 2].map(|s| Part {
            t: layer * STACKS.len() + s,
            id: id as u32,
        })
    }

    /// Drop the parts' pages from the process and the page cache.
    fn drop_parts(&self, parts: &[Part]) -> Result<(), GateError> {
        for p in parts {
            self.tables[p.t].evict_rows(&[p.id])?;
        }
        Ok(())
    }

    /// (resident pages, spanned pages) over the parts.
    fn pages(&self, parts: &[Part]) -> Result<(u64, u64), GateError> {
        let (mut resident, mut total) = (0, 0);
        for p in parts {
            let c = self.tables[p.t].resident_pages(&[p.id])?;
            resident += c.resident;
            total += c.total;
        }
        Ok((resident, total))
    }

    /// `mincore` sees an expert's pages leave on an eviction and arrive on a
    /// populate, or no row of this run means what it says.
    fn prove_instrument(&self) -> Result<(), GateError> {
        // The middle expert of the middle layer's gate stack: its edge pages
        // are shared with neighbours nothing has read, not with a header or
        // another tensor.
        let id = self.experts / 2;
        let t = &self.tables[self.layers / 2 * STACKS.len()];
        let part = [Part {
            t: self.layers / 2 * STACKS.len(),
            id: id as u32,
        }];
        self.drop_parts(&part)?;
        let (resident, total) = self.pages(&part)?;
        if resident != 0 {
            return Err(format!(
                "after an eviction mincore finds {resident} of {total} pages of {} expert {id} resident: \
                 another process holds or maps them, or the file's pages cannot be dropped here",
                t.path().display()
            )
            .into());
        }
        t.prefetch(&[id as u32])?;
        t.populate(&[id as u32])?;
        let (resident, total) = self.pages(&part)?;
        if resident != total {
            return Err(format!(
                "after a populate mincore finds {resident} of {total} pages resident: it reports a file's \
                 page cache only to a caller that owns or may write the file"
            )
            .into());
        }
        self.drop_parts(&part)
    }

    /// The drawn batch of `shape`, its picks as labels, its parts dropped and
    /// checked cold; `redraws` counts the draws refused because a page stayed
    /// resident.
    fn cold_batch(
        &self,
        shape: Shape,
        draw: &mut Draw,
        used: &mut [bool],
        redraws: &mut usize,
        cap: usize,
    ) -> Result<(Vec<Part>, Vec<String>), GateError> {
        loop {
            let (parts, labels) = match shape {
                Shape::Runs(k) => {
                    let mut parts = Vec::new();
                    let mut labels = Vec::new();
                    for layer in distinct_layers(draw, self.layers, k) {
                        let id = draw.below(self.experts);
                        parts.extend(self.run_parts(layer, id));
                        labels.push(format!("{layer}:{id}"));
                    }
                    (parts, labels)
                }
                Shape::Range(from) => {
                    let free: Vec<usize> = (0..self.layers).filter(|&l| !used[l]).collect();
                    if free.is_empty() {
                        return Err("every layer's range was drawn or refused".into());
                    }
                    let layer = free[draw.below(free.len())];
                    used[layer] = true;
                    let mut parts = Vec::new();
                    for s in 0..STACKS.len() {
                        for id in from..self.experts {
                            parts.push(Part {
                                t: layer * STACKS.len() + s,
                                id: id as u32,
                            });
                        }
                    }
                    (parts, vec![layer.to_string()])
                }
            };
            self.drop_parts(&parts)?;
            let (resident, _) = self.pages(&parts)?;
            if resident == 0 {
                return Ok((parts, labels));
            }
            *redraws += 1;
            if *redraws > cap {
                return Err(format!(
                    "{redraws} batches redrawn because dropped runs stayed resident (the last: {resident} pages): \
                     another process holds or maps this file's pages"
                )
                .into());
            }
        }
    }

    /// The `O_DIRECT` reads of the parts.
    fn reads(&self, parts: &[Part]) -> Result<Vec<Read>, GateError> {
        parts
            .iter()
            .map(|p| {
                let (at, len) = self.tables[p.t].file_range(p.id)?;
                let (start, span, need) = aligned_span(at, len);
                Ok(Read {
                    table: p.t,
                    start,
                    span,
                    need,
                })
            })
            .collect()
    }

    /// The reads of a range case's parts as the drive is read at its best on
    /// one thread: each stack's experts are one contiguous run of the file
    /// (the parts of a table, ascending and adjacent), read in
    /// [`DRIVE_CHUNK`]-byte requests.
    fn contiguous_reads(&self, parts: &[Part]) -> Result<Vec<Read>, GateError> {
        let mut reads = Vec::new();
        let mut i = 0;
        while i < parts.len() {
            let t = parts[i].t;
            let mut j = i;
            while j + 1 < parts.len() && parts[j + 1].t == t {
                if parts[j + 1].id != parts[j].id + 1 {
                    return Err("a range's parts of one stack are not adjacent experts".into());
                }
                j += 1;
            }
            let (at, _) = self.tables[t].file_range(parts[i].id)?;
            let (last, len) = self.tables[t].file_range(parts[j].id)?;
            reads.extend(chunked(t, at, last + len));
            i = j + 1;
        }
        Ok(reads)
    }

    /// The cache arm, in ns: `WILLNEED` over every part, then each worker
    /// populating its share.
    fn cache_arm(&self, parts: &[Part]) -> Result<u64, GateError> {
        let jobs: Vec<Job> = partition(parts.len(), self.pool.threads())
            .into_iter()
            .map(|r| {
                let tables = Arc::clone(&self.tables);
                let chunk = parts[r].to_vec();
                Box::new(move |_: &mut Vec<u8>| -> Result<(), String> {
                    for p in &chunk {
                        tables[p.t].populate(&[p.id]).map_err(|e| e.to_string())?;
                    }
                    Ok(())
                }) as Job
            })
            .collect();
        let t0 = Instant::now();
        for p in parts {
            self.tables[p.t].prefetch(&[p.id])?;
        }
        self.pool.run(jobs)?;
        Ok(t0.elapsed().as_nanos() as u64)
    }

    /// The `direct1` arm, in ns: every read on this thread.
    fn direct1_arm(&mut self, reads: &[Read]) -> Result<u64, GateError> {
        let t0 = Instant::now();
        for r in reads {
            read_direct(&self.files[r.table], r, &mut self.scratch)?;
        }
        Ok(t0.elapsed().as_nanos() as u64)
    }

    /// The `directN` arm, in ns: the reads split across the workers as the
    /// cache arm splits its parts.
    fn directn_arm(&self, reads: &[Read]) -> Result<u64, GateError> {
        let jobs: Vec<Job> = partition(reads.len(), self.pool.threads())
            .into_iter()
            .map(|r| {
                let files = Arc::clone(&self.files);
                let chunk = reads[r].to_vec();
                Box::new(move |scratch: &mut Vec<u8>| -> Result<(), String> {
                    for r in &chunk {
                        read_direct(&files[r.table], r, scratch)?;
                    }
                    Ok(())
                }) as Job
            })
            .collect();
        let t0 = Instant::now();
        self.pool.run(jobs)?;
        Ok(t0.elapsed().as_nanos() as u64)
    }

    /// One batch of cold parts through the arms.
    fn batch(&mut self, parts: &[Part], shape: Shape) -> Result<Sample, GateError> {
        let mut s = Sample {
            bytes: parts.iter().map(|p| self.tables[p.t].row_bytes()).sum(),
            ns: [0; 4],
            asked: [0; 4],
        };
        s.ns[0] = self.cache_arm(parts)?;
        let (resident, total) = self.pages(parts)?;
        if resident != total {
            return Err(format!(
                "after the cache arm {resident} of {total} spanned pages are resident: the read did not land whole"
            )
            .into());
        }
        s.asked[0] = total * self.tables[parts[0].t].page_bytes();
        self.drop_parts(parts)?;
        let reads = self.reads(parts)?;
        s.asked[1] = reads.iter().map(|r| r.span as u64).sum();
        s.asked[2] = s.asked[1];
        s.ns[1] = self.direct1_arm(&reads)?;
        s.ns[2] = self.directn_arm(&reads)?;
        if let Shape::Range(_) = shape {
            let big = self.contiguous_reads(parts)?;
            s.asked[3] = big.iter().map(|r| r.span as u64).sum();
            s.ns[3] = self.direct1_arm(&big)?;
        }
        Ok(s)
    }

    /// A case: its warm-up and counted batches, then a row an arm.
    fn case(&mut self, a: &Args, shape: Shape, draw: &mut Draw) -> Result<(), GateError> {
        let (name, experts, counted) = match shape {
            Shape::Runs(k) => (format!("k{k}"), k, a.batches),
            Shape::Range(from) => ("range".to_string(), self.experts - from, a.range_batches),
        };
        let total = counted + a.warmup;
        let cap = total + 16;
        let mut used = vec![false; self.layers];
        let mut redraws = 0;
        let mut samples = Vec::with_capacity(counted);
        let mut asked = [0u64; 4];
        let mut draws = Vec::new();
        for i in 0..total {
            let (parts, labels) = self.cold_batch(shape, draw, &mut used, &mut redraws, cap)?;
            let s = self.batch(&parts, shape)?;
            for (all, one) in asked.iter_mut().zip(s.asked) {
                *all += one;
            }
            draws.extend(labels);
            if i >= a.warmup {
                samples.push(s);
            }
        }
        let arms = if matches!(shape, Shape::Range(_)) {
            4
        } else {
            3
        };
        for (i, arm) in ARMS.into_iter().take(arms).enumerate() {
            let mut rates: Vec<f64> = samples
                .iter()
                .map(|s| s.bytes as f64 / s.ns[i] as f64)
                .collect();
            rates.sort_by(f64::total_cmp);
            let mut walls: Vec<f64> = samples.iter().map(|s| s.ns[i] as f64 / 1e6).collect();
            walls.sort_by(f64::total_cmp);
            let n = samples.len() as u64;
            let row = Record::new(&NVREAD)
                .w("arm", arm)
                .w("case", &name)
                .u("experts", experts)
                .u("threads", self.pool.threads())
                .u("batches", counted)
                .u("warmup", a.warmup)
                .u("redraws", redraws)
                .u("bytes", samples.iter().map(|s| s.bytes).sum::<u64>() / n)
                .u(
                    "span_bytes",
                    samples.iter().map(|s| s.asked[i]).sum::<u64>() / n,
                )
                .u("read_bytes", asked[i])
                .f("median", median(&rates))
                .f("p10", percentile(&rates, 0.10))
                .f("p90", percentile(&rates, 0.90))
                .f("ms", median(&walls))
                .u("seed", a.seed)
                .w("lease", a.lease);
            if i == 0 {
                row.csv("draws", &draws).print();
            } else {
                row.print();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_covers_every_item_once_in_even_ranges() {
        for (len, n) in [(9, 8), (3, 8), (768, 8), (5, 1), (0, 4), (8, 8)] {
            let parts = partition(len, n);
            assert_eq!(parts.len(), n.min(len), "{len} items over {n}");
            let mut at = 0;
            for r in &parts {
                assert_eq!(r.start, at, "{len} over {n}: ranges are contiguous");
                at = r.end;
            }
            assert_eq!(at, len, "{len} over {n}: every item is in a range");
            if let (Some(lo), Some(hi)) = (
                parts.iter().map(|r| r.len()).min(),
                parts.iter().map(|r| r.len()).max(),
            ) {
                assert!(hi - lo <= 1, "{len} over {n}: sizes {lo}..{hi}");
            }
        }
    }

    #[test]
    fn quantiles_of_a_sorted_sample() {
        let ten: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(median(&ten), 5.5);
        assert_eq!(percentile(&ten, 0.10), 1.0);
        assert_eq!(percentile(&ten, 0.90), 9.0);
        assert_eq!(median(&[3.0]), 3.0);
        assert_eq!(percentile(&[3.0], 0.90), 3.0);
        assert_eq!(median(&[1.0, 2.0, 9.0]), 2.0);
        // 0.9 x 7 is not a whole rank: the nearest rank rounds it up to the 7th.
        let seven: Vec<f64> = (1..=7).map(f64::from).collect();
        assert_eq!(percentile(&seven, 0.90), 7.0);
        assert_eq!(percentile(&seven, 0.10), 1.0);
    }

    #[test]
    fn a_read_covers_its_part_on_aligned_ends() {
        let a = DIRECT_ALIGN as u64;
        for (at, len) in [
            (0, 1),
            (a, a),
            (a + 7, a),
            (3 * a - 1, 2),
            (1_000_003, 921_600),
        ] {
            let (start, span, need) = aligned_span(at, len);
            assert_eq!(start % a, 0, "start of {at}+{len}");
            assert_eq!(span as u64 % a, 0, "span of {at}+{len}");
            assert!(start <= at, "{at}+{len}: the start is not past the part");
            assert!(
                at + len <= start + span as u64,
                "{at}+{len}: the span holds the part"
            );
            assert_eq!(
                need as u64,
                at + len - start,
                "{at}+{len}: the bytes the part needs"
            );
            assert!(
                span - need < DIRECT_ALIGN,
                "{at}+{len}: no whole extra block is read"
            );
        }
    }

    #[test]
    fn chunks_cover_a_run_in_aligned_requests() {
        let a = DIRECT_ALIGN as u64;
        for (at, end) in [
            (a + 5, 3 * a),
            (0, DRIVE_CHUNK),
            (7, 2 * DRIVE_CHUNK + 11),
            (100 * a, 100 * a + 1),
        ] {
            let reads = chunked(3, at, end);
            assert!(!reads.is_empty());
            let mut next = reads[0].start;
            assert!(
                next <= at && at - next < a,
                "{at}..{end}: the first read starts in the part's first block"
            );
            for r in &reads {
                assert_eq!(r.start, next, "{at}..{end}: reads are adjacent");
                assert_eq!(r.start % a, 0);
                assert_eq!(r.span as u64 % a, 0);
                assert!(
                    r.span as u64 <= DRIVE_CHUNK,
                    "{at}..{end}: a request is at most a chunk"
                );
                assert!(
                    r.need <= r.span && r.need > 0,
                    "{at}..{end}: a read needs some of its bytes"
                );
                next = r.start + r.span as u64;
            }
            let last = reads.last().unwrap();
            assert!(
                next >= end && next - end < a,
                "{at}..{end}: the last read ends in the run's last block"
            );
            assert_eq!(
                last.start + last.need as u64,
                end,
                "{at}..{end}: the last read needs the bytes to the end"
            );
        }
    }

    #[test]
    fn a_batch_draws_distinct_layers_and_the_same_ones_for_a_seed() {
        let mut a = Draw(7);
        let mut b = Draw(7);
        for _ in 0..200 {
            let x = distinct_layers(&mut a, 48, 3);
            assert_eq!(x, distinct_layers(&mut b, 48, 3), "the seed fixes the draw");
            assert!(x.iter().all(|&l| l < 48));
            assert!(x[0] != x[1] && x[0] != x[2] && x[1] != x[2], "{x:?}");
        }
        let mut all = distinct_layers(&mut Draw(1), 5, 5);
        all.sort_unstable();
        assert_eq!(all, [0, 1, 2, 3, 4], "k = layers draws every layer");
    }
}
