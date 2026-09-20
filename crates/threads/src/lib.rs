//! bloomery-threads — the resident worker pool that the model crate's row
//! parallelization rides on (stage "pool").
//!
//! `matmul_q` is called over a thousand times per token, so per-call thread
//! spawning (`std::thread::scope` with 32 spawns) costs more than the work.
//! This crate keeps the workers resident instead: they spin briefly for
//! back-to-back dispatches (`BLOOMERY_SPIN`, default 20000) and park on a
//! condvar when idle, so a quiet machine does not burn 32 cores spinning.
//!
//! The topology reader, the pin call and the barrier discipline are ported
//! from `crates/q3k-cpu/src/main.rs` (its `ccd_topology`, `pin` and the
//! sense-reversing barrier): that file is a bench binary with its own RESULTS
//! record and is deliberately not touched or moved — this crate rewrites the
//! ideas as a library. Migrating q3k-cpu onto it is a later round.
//!
//! Threading shape: with `threads()` = T there are T-1 resident workers,
//! pinned across CCDs, and the calling thread runs the last chunk itself —
//! one fewer wake per call, same as the bench binary's "main is the last
//! participant". The pool never pins the caller on its own: it is process-wide
//! and must not hijack an arbitrary caller's affinity (`pin_caller` opts in). `BLOOMERY_THREADS=1`
//! spawns nothing at all and the caller runs the whole range inline, so the
//! single-threaded path pays no barrier cost.

use std::any::Any;
use std::cell::UnsafeCell;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

/// Default spin iterations before a waiting worker parks.
const DEFAULT_SPIN: u64 = 20_000;

thread_local! {
    /// Set on any thread that is inside `for_each_chunk` — the dispatching
    /// caller for the whole call, and each resident worker while it runs its
    /// chunk. `for_each_chunk` serializes on one mutex and does not return
    /// until every chunk is done, so a nested call deadlocks: from the caller
    /// on the mutex it already holds, from a worker on the mutex the caller
    /// holds. Either way it is a hang with no message and no stack naming the
    /// cause. This flag makes it an assertion instead.
    ///
    /// Nested parallelism is a reasonable thing to want (a parallel expert loop
    /// over parallel rows); it needs a second pool or work stealing, and this
    /// says so at the moment someone tries it rather than at 3 a.m. The refusal
    /// is deliberately uniform across thread counts: a contract that holds at
    /// `BLOOMERY_THREADS=32` and quietly relaxes at 1 is worse than no contract,
    /// because the single-threaded run is the one people debug in.
    static IN_PARALLEL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Sets [`IN_PARALLEL`] and clears it on drop, so an unwinding chunk closure
/// does not leave the thread permanently marked.
struct ParallelGuard;

impl ParallelGuard {
    fn enter() -> Self {
        assert!(
            !IN_PARALLEL.with(std::cell::Cell::get),
            "for_each_chunk is not reentrant: a chunk closure called it again, which \
             deadlocks on the dispatch mutex the outer call holds"
        );
        IN_PARALLEL.with(|f| f.set(true));
        ParallelGuard
    }
}

impl Drop for ParallelGuard {
    fn drop(&mut self) {
        IN_PARALLEL.with(|f| f.set(false));
    }
}

/// The process-wide pool. Built once, on first use.
pub fn pool() -> &'static Pool {
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let p: &'static Pool = Box::leak(Box::new(Pool::build()));
        p.spawn_workers();
        p
    })
}

/// The fixed pool of resident workers.
pub struct Pool {
    nthreads: usize,
    spin: u64,
    /// One Vec per CCD (shared L3), logical cpu ids, primaries before
    /// siblings — as detected, see `detect_topology`.
    topo: Vec<Vec<u32>>,
    pin_failed: AtomicBool,

    // --- dispatch state -------------------------------------------------
    // `dispatch` serializes `for_each_chunk` calls; the job slot is handed to
    // the workers through the seq/remaining protocol described on `job`.
    dispatch: Mutex<()>,
    job: UnsafeCell<JobSlot>,
    /// Job generation counter. A worker runs job k when it observes `seq`
    /// move; the dispatcher bumps `seq` (release) only after the job slot and
    /// `remaining` are in place, so the bump publishes them.
    seq: AtomicUsize,
    /// Per-worker completion marks: worker `t` stores the `seq` of the job it
    /// just finished. `remaining` in this file's comments is the number of
    /// marks that differ from the current `seq`; the job is over when it is
    /// zero. One padded line per worker, not one shared counter: thirty-one
    /// decrements of a single line serialize through its ownership transfers,
    /// and that chain was most of an empty dispatch.
    done: Box<[DoneMark]>,
    lock: Mutex<()>,
    /// Workers currently inside the park path (from before their final `seq`
    /// recheck until they wake). The dispatcher notifies only when this is
    /// nonzero: a `notify_all` is a futex syscall even with nobody waiting, and
    /// at hundreds of dispatches per step that is the dispatch path's one
    /// syscall. Correctness is the store-load handshake — the worker raises
    /// `parked` THEN rechecks `seq`, the dispatcher bumps `seq` THEN reads
    /// `parked`, all SeqCst, so at least one side sees the other.
    parked: AtomicUsize,
    /// The dispatcher's twin of `parked`, for the last worker's `cv_done` wake.
    dispatcher_parked: AtomicBool,
    /// Workers park here waiting for `seq` to move.
    cv_work: Condvar,
    /// The dispatcher parks here waiting for `remaining` to hit zero.
    cv_done: Condvar,
    /// First panic caught in the current job, replayed to the caller once
    /// every participant has arrived.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
    /// Workers that have taken their starting `seen` snapshot. `pool()` does
    /// not return until this reaches the worker count, so no dispatch can
    /// race a starting worker: a worker whose first `seq` load happens after
    /// job 1 is published would initialize `seen` past it, skip the job
    /// entirely and strand the dispatcher at `remaining > 0` forever.
    ready: AtomicUsize,

    // --- observability ----------------------------------------------------
    // Relaxed counting of the protocol itself, so a round that attacks
    // dispatch overhead can tell "workers park between matmul_q calls" from
    // "workers stay hot and the time goes elsewhere". One fetch_add per
    // dispatch and one per actual condvar wait — never inside a row loop —
    // so the counters cannot reorder or delay anything the seq/remaining
    // protocol publishes.
    dispatches: AtomicUsize,
    /// Times the dispatcher actually parked on `cv_done` (spin exhausted
    /// first). Zero on every call means the workers finish within the spin
    /// budget.
    dispatcher_parks: AtomicUsize,
    /// Times any worker actually parked on `cv_work`. At ~850 pool calls per
    /// decode step this is the number that separates a futex-wake dispatch
    /// from a spin-only one.
    worker_parks: AtomicUsize,
}

/// Snapshot of the pool's protocol counters, cumulative since process start.
/// See the `dispatches` field group on [`Pool`] for what each one rules in
/// or out.
#[derive(Clone, Copy, Debug)]
pub struct PoolStats {
    pub dispatches: u64,
    pub dispatcher_parks: u64,
    pub worker_parks: u64,
}

/// What one dispatched job consists of: the closure, lifetime-erased as a
/// thin data pointer plus a monomorphized call shim (see the SAFETY notes in
/// `for_each_chunk` and `Pool::worker_main`), and the range length to split.
/// A `*const dyn Fn` fat pointer would demand `F: 'static` — the shim pair
/// keeps the borrow local to the call.
struct JobSlot {
    data: *const (),
    call: fn(*const (), Range<usize>),
    n: usize,
}

/// Placeholder for the idle job slot.
fn noop_call(_: *const (), _: Range<usize>) {}

impl Clone for JobSlot {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for JobSlot {}

// SAFETY: the only state shared mutably across threads is the job slot, and
// every access to it is ordered by the seq/remaining protocol: the
// dispatcher writes the slot before the release-increment of `seq`, and does
// not write it again until the previous job's `remaining` reached zero —
// which requires every worker to have finished calling the closure. A worker
// reads the slot only between its acquire-observation of a new `seq` and its
// `remaining` decrement. `dispatch` serializes dispatchers, `panic` is
// behind a mutex, `pin_failed` is atomic, and everything else is immutable
// after construction.
unsafe impl Sync for Pool {}

impl Pool {
    pub fn threads(&self) -> usize {
        self.nthreads
    }

    /// Topology as detected: one Vec per CCD, logical cpu ids, primaries
    /// before siblings.
    pub fn topology(&self) -> &[Vec<u32>] {
        &self.topo
    }

    /// True if `sched_setaffinity` failed for at least one worker (containers
    /// and cgroup-restricted processes legitimately cannot pin). The pool
    /// keeps running unpinned in that case.
    pub fn pin_failed(&self) -> bool {
        self.pin_failed.load(Ordering::Relaxed)
    }

    /// Opt-in: pin the calling thread to the cpu slot of the chunk it runs
    /// (the last one). The pool never does this on its own — a binary that
    /// owns its main thread calls it once. Unpinned, the dispatcher floats onto
    /// an SMT sibling of a spinning worker and every barrier waits for it.
    pub fn pin_caller(&self) -> bool {
        pin(self.cpu_for(self.nthreads - 1))
    }

    /// Protocol counters since process start. Diagnostic only — nothing in
    /// the dispatch path branches on them.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            dispatches: self.dispatches.load(Ordering::Relaxed) as u64,
            dispatcher_parks: self.dispatcher_parks.load(Ordering::Relaxed) as u64,
            worker_parks: self.worker_parks.load(Ordering::Relaxed) as u64,
        }
    }

    /// Split `n` into `threads()` contiguous chunks and run `f` on each, in
    /// parallel, returning when every chunk is done.
    ///
    /// If a chunk's `f` panics, the panic is caught in the worker so the
    /// completion barrier still fires, and the first caught panic is rethrown
    /// at the caller once every chunk has arrived — the pool stays usable.
    pub fn for_each_chunk<F>(&self, n: usize, f: F)
    where
        F: Fn(Range<usize>) + Sync,
    {
        let _parallel = ParallelGuard::enter();
        if self.nthreads == 1 {
            // No workers exist; the calling thread does all of it with no
            // barrier cost.
            f(0..n);
            return;
        }
        let _dispatch = self.dispatch.lock().unwrap_or_else(|e| e.into_inner());

        // SAFETY: we hold the dispatch lock and the previous job (if any)
        // completed — its `remaining` reached zero before that call returned,
        // so no worker touches the slot until `seq` moves below. The closure
        // stays alive for the whole call: we do not return until `remaining`
        // is zero, i.e. every worker is done calling it.
        // SAFETY inside the shim: `p` is `&f` of the current call, alive
        // until `remaining` reaches zero (the dispatcher does not return
        // before that, so no worker can call past the borrow).
        let call: fn(*const (), Range<usize>) = |p, r| unsafe { (*p.cast::<F>())(r) };
        // SAFETY: we hold the dispatch lock and the previous job (if any)
        // completed — its `remaining` reached zero before that call returned,
        // so no worker touches the slot until `seq` moves below.
        unsafe {
            *self.job.get() = JobSlot {
                data: std::ptr::from_ref(&f).cast::<()>(),
                call,
                n,
            };
        }
        self.dispatches.fetch_add(1, Ordering::Relaxed);
        let cur = self.seq.fetch_add(1, Ordering::SeqCst).wrapping_add(1);
        if self.parked.load(Ordering::SeqCst) != 0 {
            // A worker that raised `parked` holds `lock` until it is inside
            // `wait`, so taking the lock here means the notify cannot land
            // between its recheck and its wait.
            let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            self.cv_work.notify_all();
        }

        // The calling thread runs the last chunk itself, in parallel with the
        // resident workers.
        let (start, end) = chunk_bounds(n, self.nthreads, self.nthreads - 1);
        let own = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(start..end)));

        // Wait for the workers: spin first (they are usually already done or
        // mid-chunk), then park on cv_done.
        let mut spins = self.spin;
        loop {
            if self.all_done(cur) {
                break;
            }
            if spins > 0 {
                spins -= 1;
                std::hint::spin_loop();
                continue;
            }
            let guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            self.dispatcher_parked.store(true, Ordering::SeqCst);
            if !self.all_done(cur) {
                self.dispatcher_parks.fetch_add(1, Ordering::Relaxed);
                drop(self.cv_done.wait(guard).unwrap_or_else(|e| e.into_inner()));
            }
            self.dispatcher_parked.store(false, Ordering::SeqCst);
        }

        if let Err(p) = own {
            let mut cell = self.panic.lock().unwrap_or_else(|e| e.into_inner());
            if cell.is_none() {
                *cell = Some(p);
            }
        }
        // SAFETY of the resume: every worker has arrived (remaining == 0), so
        // none is still inside `f`; replaying the panic with the dispatch
        // lock released (a poisoned `dispatch` is recovered via into_inner)
        // leaves the pool in a consistent, reusable state.
        let first = self.panic.lock().unwrap_or_else(|e| e.into_inner()).take();
        drop(_dispatch);
        if let Some(p) = first {
            std::panic::resume_unwind(p);
        }
    }

    fn build() -> Pool {
        let groups = detect_topology();
        let mut topo = Vec::with_capacity(groups.len());
        let mut physical = 0;
        for (primaries, siblings) in &groups {
            physical += primaries.len();
            let mut full = primaries.clone();
            full.extend_from_slice(siblings);
            topo.push(full);
        }
        // BLOOMERY_THREADS overrides the count; a missing, unparsable or zero
        // value falls back to physical cores. SMT siblings are a deliberate
        // opt-in, not the default: this tier is bandwidth-bound.
        let nthreads = std::env::var("BLOOMERY_THREADS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&t| t > 0)
            .unwrap_or(physical);
        let spin = std::env::var("BLOOMERY_SPIN")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_SPIN);
        Pool {
            nthreads: nthreads.max(1),
            spin,
            topo,
            pin_failed: AtomicBool::new(false),
            dispatch: Mutex::new(()),
            job: UnsafeCell::new(JobSlot {
                data: std::ptr::null(),
                call: noop_call,
                n: 0,
            }),
            seq: AtomicUsize::new(0),
            done: (0..nthreads.saturating_sub(1)).map(|_| DoneMark(AtomicUsize::new(0))).collect(),
            parked: AtomicUsize::new(0),
            dispatcher_parked: AtomicBool::new(false),
            lock: Mutex::new(()),
            cv_work: Condvar::new(),
            cv_done: Condvar::new(),
            panic: Mutex::new(None),
            ready: AtomicUsize::new(0),
            dispatches: AtomicUsize::new(0),
            dispatcher_parks: AtomicUsize::new(0),
            worker_parks: AtomicUsize::new(0),
        }
    }

    fn spawn_workers(&'static self) {
        for t in 0..self.nthreads.saturating_sub(1) {
            std::thread::Builder::new()
                .name(format!("bloomery-pool-{t}"))
                .spawn(move || self.worker_main(t))
                .expect("spawn pool worker");
        }
        // Startup barrier: `spawn` returning only means the thread exists,
        // not that `worker_main` has snapshotted `seq`. No job may be
        // dispatched until every worker has, or a late starter skips job 1.
        // yield_now rather than spin: the workers need the cpu to get here.
        let nworkers = self.nthreads - 1;
        while self.ready.load(Ordering::Acquire) < nworkers {
            std::thread::yield_now();
        }
    }

    /// Worker `t` is spread across CCDs: CCD `t % nccd`, slot `t / nccd` of
    /// that CCD's cpu list (physical cores first). Beyond the detected cpu
    /// count the last slot is shared rather than panicking.
    fn cpu_for(&self, t: usize) -> u32 {
        let nccd = self.topo.len();
        let g = &self.topo[t % nccd];
        g[(t / nccd).min(g.len() - 1)]
    }

    /// Every worker's mark equals `cur`: `remaining == 0`.
    fn all_done(&self, cur: usize) -> bool {
        self.done.iter().all(|d| d.0.load(Ordering::SeqCst) == cur)
    }

    fn worker_main(&'static self, t: usize) {
        if !pin(self.cpu_for(t)) {
            // Not fatal: containers cannot pin. Recorded, pool continues.
            self.pin_failed.store(true, Ordering::Relaxed);
        }
        let mut seen = self.seq.load(Ordering::Acquire);
        // Publish readiness only after the snapshot above, so the ordering
        // "snapshot -> ready -> first dispatch" is guaranteed by pool().
        self.ready.fetch_add(1, Ordering::Release);
        loop {
            // Wait for the next job: spin (back-to-back dispatches stay in
            // cache), then park on cv_work.
            let mut spins = self.spin;
            loop {
                let cur = self.seq.load(Ordering::Acquire);
                if cur != seen {
                    seen = cur;
                    break;
                }
                if spins > 0 {
                    spins -= 1;
                    std::hint::spin_loop();
                    continue;
                }
                let guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
                self.parked.fetch_add(1, Ordering::SeqCst);
                if self.seq.load(Ordering::SeqCst) == seen {
                    self.worker_parks.fetch_add(1, Ordering::Relaxed);
                    drop(self.cv_work.wait(guard).unwrap_or_else(|e| e.into_inner()));
                }
                self.parked.fetch_sub(1, Ordering::SeqCst);
            }

            // SAFETY: `seen` just moved, which happens-after the dispatcher's
            // store of this slot (release bump of `seq` -> this acquire
            // load), and the slot is not written again until this job's
            // `remaining` reaches zero — not before this worker's own
            // decrement below.
            let slot = unsafe { *self.job.get() };
            let (start, end) = chunk_bounds(slot.n, self.nthreads, t);
            // SAFETY: same ordering argument as the slot read; the data
            // pointer is `&f` of the current call, which outlives this call
            // because the dispatcher cannot return from `for_each_chunk`
            // before every worker decrements `remaining` — which happens
            // after this call returns.
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Marked for the duration of the chunk, so a nested dispatch
                // from inside the closure asserts here instead of blocking on
                // the mutex the dispatcher holds. The guard clears on unwind.
                let _parallel = ParallelGuard::enter();
                (slot.call)(slot.data, start..end)
            }));
            if let Err(p) = r {
                // Caught, recorded, and the barrier still fires below — a
                // panicking chunk must not strand the dispatcher.
                let mut cell = self.panic.lock().unwrap_or_else(|e| e.into_inner());
                if cell.is_none() {
                    *cell = Some(p);
                }
            }
            self.done[t].0.store(seen, Ordering::SeqCst);
            if self.dispatcher_parked.load(Ordering::SeqCst) {
                // The dispatcher is in its park path: it holds `lock` until it
                // is inside `wait`, so this cannot land between its recheck
                // and its wait. Any worker may wake it; it rechecks every mark.
                let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
                self.cv_done.notify_all();
            }
        }
    }
}

/// One worker's completion mark on its own cache line.
#[repr(align(64))]
struct DoneMark(AtomicUsize);

/// The partition `for_each_chunk` uses, exposed so it can be tested without
/// threads. Deterministic: the same `(n, threads)` always yields the same
/// split. Contiguous chunks (not interleaved) so each worker streams a
/// contiguous weight range and the CCD L3s do not evict each other. If `n`
/// does not divide evenly, the first `n % threads` chunks get one extra
/// element; empty chunks (`n < threads`) are normal and their `f` calls
/// return immediately.
pub fn chunks(n: usize, threads: usize) -> Vec<Range<usize>> {
    assert!(threads > 0, "threads must be at least 1");
    (0..threads)
        .map(|t| {
            let (start, end) = chunk_bounds(n, threads, t);
            start..end
        })
        .collect()
}

/// Chunk `t` of `threads` over `0..n`: the first `n % threads` chunks are one
/// element longer than the rest.
pub fn chunk_bounds(n: usize, threads: usize, t: usize) -> (usize, usize) {
    assert!(t < threads);
    let base = n / threads;
    let rem = n % threads;
    let start = if t < rem {
        t * (base + 1)
    } else {
        rem * (base + 1) + (t - rem) * base
    };
    let len = base + usize::from(t < rem);
    (start, start + len)
}

// ---------------------------------------------------------------- topology

fn parse_cpu_list(s: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for part in s.trim().split(',') {
        if let Some((a, b)) = part.split_once('-') {
            for c in a.trim().parse::<u32>().unwrap()..=b.trim().parse::<u32>().unwrap() {
                out.push(c);
            }
        } else {
            out.push(part.trim().parse().unwrap());
        }
    }
    out
}

fn read_sys(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// Detect the L3 (CCD) topology: one `(primaries, siblings)` pair per shared
/// L3, groups sorted by primary core id. The primary hyperthread of each
/// core is its lowest sibling id. Ported from
/// `crates/q3k-cpu/src/main.rs:324` (`ccd_topology`), restructured so the
/// physical-core count is recoverable — the bench binary's return shape
/// (primaries then siblings, flat) cannot say where the split is.
fn detect_topology() -> Vec<(Vec<u32>, Vec<u32>)> {
    let mut groups: Vec<(String, Vec<u32>)> = Vec::new(); // (l3 key, primaries)
    let mut cpu = 0u32;
    while let Some(sib) = read_sys(&format!(
        "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    )) {
        let siblings = parse_cpu_list(&sib);
        let primary = *siblings.iter().min().unwrap();
        if primary == cpu {
            let l3 = read_sys(&format!(
                "/sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list"
            ))
            .unwrap_or_default();
            match groups.iter_mut().find(|(k, _)| *k == l3) {
                Some((_, v)) => v.push(cpu),
                None => groups.push((l3, vec![cpu])),
            }
        }
        cpu += 1;
    }
    if groups.is_empty() {
        // No sysfs topology (container?): one group of everything found, so
        // the pool still has a nonzero default thread count.
        let cpus: Vec<u32> = if cpu == 0 {
            vec![0]
        } else {
            (0..cpu).collect()
        };
        return vec![(cpus, Vec::new())];
    }
    groups.sort_by_key(|(_, v)| v[0]);
    groups
        .into_iter()
        .map(|(_, primaries)| {
            let mut siblings = Vec::new();
            for &p in &primaries {
                let sib = parse_cpu_list(
                    &read_sys(&format!(
                        "/sys/devices/system/cpu/cpu{p}/topology/thread_siblings_list"
                    ))
                    .unwrap_or_default(),
                );
                for s in sib {
                    if s != p {
                        siblings.push(s);
                    }
                }
            }
            (primaries, siblings)
        })
        .collect()
}

// --------------------------------------------------------------------- pin

/// Pin the calling thread to one logical cpu. Returns false instead of
/// panicking — containers and cgroup-restricted processes legitimately
/// cannot pin, and the pool must keep running unpinned.
fn pin(cpu: u32) -> bool {
    // SAFETY: `set` is a zeroed cpu_set_t bitmask only written through
    // CPU_SET, and sched_setaffinity reads it with the matching size.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu as usize, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
    }
}
