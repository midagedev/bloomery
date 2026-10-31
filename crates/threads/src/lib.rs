//! bloomery-threads — the resident worker pool that the model crate's row
//! parallelization rides on (stage "pool").
//!
//! `matmul_q` is called over a thousand times per token, so per-call thread
//! spawning (`std::thread::scope` with 32 spawns) costs more than the work.
//! This crate keeps the workers resident instead: they spin briefly for
//! back-to-back dispatches (`BLOOMERY_SPIN`) and park on a condvar when idle,
//! so a quiet machine does not burn 32 cores spinning.
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
//! participant". The cpus are those the process may run on: the sysfs
//! topology is cut to the calling thread's `sched_getaffinity` mask when the
//! pool is built, a mask that cuts it is said in a line, and [`plan_pool`] /
//! [`report`] state the placement (threads, pinned cpus, the mask) before any
//! thread exists, for the plan that prints it. The pool never pins the caller
//! on its own: it is process-wide and must not hijack an arbitrary caller's
//! affinity (`pin_caller` opts in). `BLOOMERY_THREADS=1` spawns nothing at all
//! and the caller runs the whole range inline, so the single-threaded path
//! pays no barrier cost.

pub mod helper;
mod topology;

pub use topology::{Cut, PoolPlan};
use topology::{Shape, Spread, ccd_slot, detect_topology};

use std::any::Any;
use std::cell::UnsafeCell;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock, TryLockError};
use std::time::Instant;

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

static POOL: OnceLock<&'static Pool> = OnceLock::new();

/// The pool when something has built it already; `None` builds nothing.
#[must_use]
pub fn built() -> Option<&'static Pool> {
    POOL.get().copied()
}

/// The process-wide pool. Built once, on first use.
pub fn pool() -> &'static Pool {
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
    /// Where the threads can sit: the topology as detected, cut to the
    /// process's cpu mask — see [`Shape::of`].
    shape: Shape,
    pin_failed: AtomicBool,
    /// Whether [`Pool::pin_caller`] pinned its thread.
    caller_pinned: AtomicBool,

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
    /// Dispatches that found the `dispatch` mutex held by another caller,
    /// and the nanoseconds they waited for it, summed: the time callers on
    /// two threads (a decode chain and a fill or a lane on another) spend
    /// queued behind each other's jobs. An uncontended dispatch takes the
    /// lock on its first try and reads no clock.
    dispatch_waits: AtomicUsize,
    dispatch_wait_ns: AtomicU64,
}

/// Snapshot of the pool's protocol counters, cumulative since process start.
/// See the `dispatches` field group on [`Pool`] for what each one rules in
/// or out.
#[derive(Clone, Copy, Debug)]
pub struct PoolStats {
    pub dispatches: u64,
    pub dispatcher_parks: u64,
    pub worker_parks: u64,
    /// Dispatches that waited for another caller's job to end, and how long
    /// they waited, summed (ns).
    pub dispatch_waits: u64,
    pub dispatch_wait_ns: u64,
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
// behind a mutex, `pin_failed` and `caller_pinned` are atomic, and everything
// else is immutable after construction.
unsafe impl Sync for Pool {}

impl Pool {
    #[must_use]
    pub fn threads(&self) -> usize {
        self.nthreads
    }

    /// Topology as detected: one Vec per CCD, logical cpu ids, primaries
    /// before siblings.
    pub fn topology(&self) -> &[Vec<u32>] {
        &self.shape.topo
    }

    /// Where the pool's threads sit ([`PoolPlan`]): what a plan prints
    /// before the pool is built ([`plan_pool`]), from the pool itself. A pool
    /// whose pins failed lists no pinned worker: it floats over the process's
    /// cpus.
    #[must_use]
    pub fn plan(&self) -> PoolPlan {
        let mut plan = self.shape.plan();
        if self.pin_failed() {
            plan.workers.clear();
        }
        plan
    }

    /// True if `sched_setaffinity` failed for at least one worker (containers
    /// and cgroup-restricted processes legitimately cannot pin). The pool
    /// keeps running unpinned in that case.
    pub fn pin_failed(&self) -> bool {
        self.pin_failed.load(Ordering::Relaxed)
    }

    /// The cpu worker `t` is pinned to; `None` for no such worker or a pool
    /// whose pins failed.
    #[must_use]
    pub fn worker_cpu(&self, t: usize) -> Option<usize> {
        (t + 1 < self.nthreads && !self.pin_failed()).then(|| self.cpu_for(t) as usize)
    }

    /// The cpu [`Pool::pin_caller`] pins the dispatcher to.
    #[must_use]
    pub fn caller_cpu(&self) -> usize {
        self.cpu_for(self.nthreads - 1) as usize
    }

    /// Opt-in: pin the calling thread to the cpu slot of the chunk it runs
    /// (the last one). The pool never does this on its own — a binary that
    /// owns its main thread calls it once. Unpinned, the dispatcher floats onto
    /// an SMT sibling of a spinning worker and every barrier waits for it.
    pub fn pin_caller(&self) -> bool {
        let pinned = pin(self.cpu_for(self.nthreads - 1)).is_ok();
        if pinned {
            self.caller_pinned.store(true, Ordering::Relaxed);
        }
        pinned
    }

    /// Whether a [`Pool::pin_caller`] call succeeded. The pool does not know
    /// which thread it was, only that the dispatcher's slot is taken.
    #[must_use]
    pub fn caller_pinned(&self) -> bool {
        self.caller_pinned.load(Ordering::Relaxed)
    }

    /// Where each participant sits, for the dispatches and warms that place
    /// work by CCD: `None` unless every participant is pinned — a worker's
    /// pin failed, or [`Pool::pin_caller`] has not succeeded — because an
    /// unpinned participant is on no CCD in particular. A host with one L3 is
    /// one CCD.
    #[must_use]
    pub fn ccd_map(&'static self) -> Option<CcdMap> {
        if self.pin_failed() || !self.caller_pinned() {
            return None;
        }
        Some(self.shape.ccd_map())
    }

    /// Protocol counters since process start. Diagnostic only — nothing in
    /// the dispatch path branches on them.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            dispatches: self.dispatches.load(Ordering::Relaxed) as u64,
            dispatcher_parks: self.dispatcher_parks.load(Ordering::Relaxed) as u64,
            worker_parks: self.worker_parks.load(Ordering::Relaxed) as u64,
            dispatch_waits: self.dispatch_waits.load(Ordering::Relaxed) as u64,
            dispatch_wait_ns: self.dispatch_wait_ns.load(Ordering::Relaxed),
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
        // The first try is the uncontended path's one CAS; a dispatch that
        // finds the lock held times its wait.
        let _dispatch = match self.dispatch.try_lock() {
            Ok(g) => g,
            Err(TryLockError::Poisoned(e)) => e.into_inner(),
            Err(TryLockError::WouldBlock) => {
                let t0 = Instant::now();
                let g = self.dispatch.lock().unwrap_or_else(|e| e.into_inner());
                self.dispatch_waits.fetch_add(1, Ordering::Relaxed);
                self.dispatch_wait_ns.fetch_add(
                    u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
                g
            }
        };

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
        // The pool is process-wide, so it reads its own two levers
        // (`bloomery_levers::pool_levers`: a binary that parses at `main`
        // refused a bad value there first). BLOOMERY_THREADS overrides the
        // count, unset is the physical core count the process's cpu mask
        // leaves; a value the lever does not take — zero, a word — or a
        // retired lever that is set panics by name. SMT siblings are a
        // deliberate opt-in, not the default: this tier is bandwidth-bound.
        // BLOOMERY_SPIN is the spin before a waiting thread parks.
        let levers =
            bloomery_levers::pool_levers().unwrap_or_else(|e| panic!("the worker pool: {e}"));
        let shape = shape_now(levers.threads)
            .unwrap_or_else(|e| panic!("the worker pool: sched_getaffinity: {e}"));
        // What the process is cut from is said once, where the cut is made.
        for line in shape.lines() {
            eprintln!("{line}");
        }
        let nthreads = shape.threads;
        Pool {
            nthreads,
            spin: levers.spin,
            shape,
            pin_failed: AtomicBool::new(false),
            caller_pinned: AtomicBool::new(false),
            dispatch: Mutex::new(()),
            job: UnsafeCell::new(JobSlot {
                data: std::ptr::null(),
                call: noop_call,
                n: 0,
            }),
            seq: AtomicUsize::new(0),
            done: (0..nthreads.saturating_sub(1))
                .map(|_| DoneMark(AtomicUsize::new(0)))
                .collect(),
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
            dispatch_waits: AtomicUsize::new(0),
            dispatch_wait_ns: AtomicU64::new(0),
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

    /// Worker `t` is spread across CCDs ([`ccd_slot`]), a cpu of its own
    /// while the allowed cpus last. Beyond the allowed cpu count the last slot
    /// of its CCD is shared rather than panicking.
    fn cpu_for(&self, t: usize) -> u32 {
        self.shape.cpu_for(t)
    }

    /// Record that worker `t` could not pin to `cpu`, and the line that says
    /// so for the first failure of the pool (`None` for the later ones). The
    /// mask was cut to before any pin, so a refusal is the kernel's (a seccomp
    /// filter, a cpu gone offline), not a cpuset the topology missed. The pool
    /// can still run, floating, with the same results, so this is a line and
    /// the pool goes on: [`Pool::pin_failed`] and a plan's empty worker list
    /// ([`Pool::plan`]) carry it to whoever reads the placement.
    fn note_pin_failure(&self, t: usize, cpu: u32, e: &std::io::Error) -> Option<String> {
        (!self.pin_failed.swap(true, Ordering::Relaxed)).then(|| {
            format!(
                "bloomery-threads: worker {t} could not pin to cpu {cpu}: {e}; the pool \
                 floats over the process's cpus"
            )
        })
    }

    /// Every worker's mark equals `cur`: `remaining == 0`.
    fn all_done(&self, cur: usize) -> bool {
        self.done.iter().all(|d| d.0.load(Ordering::SeqCst) == cur)
    }

    fn worker_main(&'static self, t: usize) {
        let cpu = self.cpu_for(t);
        if let Err(e) = pin(cpu)
            && let Some(line) = self.note_pin_failure(t, cpu, &e)
        {
            eprintln!("{line}");
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

/// How the pool's participants sit on the CCDs, for the work that places
/// itself by CCD — a row dispatch visiting a matrix CCD-major, a warm routine
/// filling a CCD's L3 with the rows that CCD will compute. Participant `t`
/// (`threads()` of them, the calling thread the last) runs on the CCD the
/// rule [`Pool`] pins by places it on: over CCDs of equal core and cpu
/// counts CCD `t % ccds()` as that CCD's `t / ccds()`-th participant, so a
/// CCD holds `threads / ccds` participants and the first `threads % ccds`
/// CCDs one more; over CCDs the process's mask left unequal, a CCD takes no
/// further participant once its cores are taken while another CCD has one
/// free, and no sibling while any allowed core is free.
///
/// A value only answers by arithmetic: it is `Copy` and a call allocates
/// nothing. The pool's own is [`Pool::ccd_map`]; [`CcdMap::new`] names a
/// grouping a test wants to run the same work under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CcdMap {
    threads: usize,
    spread: Spread<'static>,
    l3_bytes: u64,
}

impl CcdMap {
    /// `threads` participants over `ccds` CCDs whose L3s hold `l3_bytes` in
    /// all (0: not known).
    #[must_use]
    pub fn new(threads: usize, ccds: usize, l3_bytes: u64) -> CcdMap {
        assert!(
            threads > 0 && ccds > 0,
            "a CCD map needs a participant and a CCD: {threads} threads, {ccds} CCDs"
        );
        CcdMap {
            threads,
            spread: Spread::Even(ccds),
            l3_bytes,
        }
    }

    /// Participants, the pool's `threads()`.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// CCDs the participants are spread over (L3 groups of the topology).
    #[must_use]
    pub fn ccds(&self) -> usize {
        self.spread.len()
    }

    /// CCDs that hold a participant: `min(ccds, threads)`, the first ones.
    #[must_use]
    pub fn active(&self) -> usize {
        self.ccds().min(self.threads)
    }

    /// The L3 bytes of the active CCDs, summed; 0 when the topology does not
    /// say (`cache/index3/size`).
    #[must_use]
    pub fn l3_bytes(&self) -> u64 {
        self.l3_bytes
    }

    /// Participant `t`'s CCD.
    #[must_use]
    pub fn ccd_of(&self, t: usize) -> usize {
        assert!(t < self.threads, "participant {t} of {}", self.threads);
        ccd_slot(t, self.spread).0
    }

    /// Participant `t`'s rank among its CCD's participants.
    #[must_use]
    pub fn rank_of(&self, t: usize) -> usize {
        assert!(t < self.threads, "participant {t} of {}", self.threads);
        ccd_slot(t, self.spread).1
    }

    /// Participants on CCD `c`; 0 past the active CCDs.
    #[must_use]
    pub fn width(&self, c: usize) -> usize {
        self.first_lane(c + 1) - self.first_lane(c)
    }

    /// Participants on the CCDs before `c`: CCD `c`'s first lane in the
    /// CCD-major order of participants (CCD 0's ranks, then CCD 1's, ...).
    /// `first_lane(ccds())` is `threads()`.
    #[must_use]
    pub fn first_lane(&self, c: usize) -> usize {
        self.spread.first_lane(self.threads, c)
    }

    /// Participant `t`'s place in the CCD-major order.
    #[must_use]
    pub fn lane_of(&self, t: usize) -> usize {
        self.first_lane(self.ccd_of(t)) + self.rank_of(t)
    }

    /// The CCD of lane `lane` of the CCD-major order (the inverse of
    /// [`CcdMap::lane_of`]'s CCD part).
    #[must_use]
    pub fn lane_ccd(&self, lane: usize) -> usize {
        assert!(lane < self.threads, "lane {lane} of {}", self.threads);
        (0..self.active())
            .find(|&c| lane < self.first_lane(c + 1))
            .expect("a lane below `threads` is some active CCD's")
    }

    /// CCD `c`'s share of `n` units — the one rule that splits a matrix over
    /// CCDs: its edge sits `first_lane(c) / threads` of the way through the
    /// units, `n · first_lane(c) / threads` rounded down, so a CCD's share is
    /// as wide as its participants are many (8/8/7/7 of 30 threads on four
    /// CCDs, not four quarters) to within one unit, whatever `n` is — no CCD
    /// is handed the remainders of every matrix. The spans of CCDs `0..ccds()`
    /// partition `0..n`, in order; a CCD with no participant has an empty
    /// one.
    #[must_use]
    pub fn span(&self, n: usize, c: usize) -> Range<usize> {
        let edge = |c: usize| n * self.first_lane(c) / self.threads;
        edge(c)..edge(c + 1)
    }
}

/// The partition `for_each_chunk` uses, exposed so it can be tested without
/// threads. Deterministic: the same `(n, threads)` always yields the same
/// split. Contiguous chunks (not interleaved) so each worker streams a
/// contiguous weight range and the CCD L3s do not evict each other. If `n`
/// does not divide evenly, the first `n % threads` chunks get one extra
/// element; empty chunks (`n < threads`) are normal and their `f` calls
/// return immediately.
#[must_use]
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
#[must_use]
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

// --------------------------------------------------------------------- mask

/// The cpus the calling thread may run on, ascending: the kernel's mask for
/// the calling thread, read into a buffer that grows until the mask fits, so
/// a machine of more than 1024 cpus reads whole.
fn mask_cpus() -> std::io::Result<Vec<u32>> {
    os::mask_cpus().map_err(std::io::Error::from)
}

/// The pool's shape from this machine: its sysfs topology cut to the calling
/// thread's cpu mask, with `threads` participants or one a core.
fn shape_now(threads: Option<usize>) -> std::io::Result<Shape> {
    Ok(Shape::of(detect_topology(), &mask_cpus()?, threads))
}

/// Where the pool's threads would sit if it were built now — the sysfs
/// topology cut to the calling thread's cpu mask, the thread count the pool
/// levers give — read without spawning a thread: what a plan prints
/// ([`report`]) before a load builds the pool.
pub fn plan_pool() -> Result<PoolPlan, String> {
    let levers = bloomery_levers::pool_levers().map_err(|e| e.to_string())?;
    shape_now(levers.threads)
        .map(|s| s.plan())
        .map_err(|e| format!("sched_getaffinity: {e}"))
}

/// The pool's placement now: the built pool's own ([`Pool::plan`]), else
/// what building it would give ([`plan_pool`]).
pub fn report() -> Result<PoolPlan, String> {
    match built() {
        Some(p) => Ok(p.plan()),
        None => plan_pool(),
    }
}

// --------------------------------------------------------------------- pin

/// Pin the calling thread to one logical cpu — the pool's own pin call, a
/// `sched_setaffinity` of that cpu alone, so a pinned caller outside the pool
/// sits where a pool worker would. Returns the kernel's refusal instead of
/// panicking — a seccomp filter or a cpu outside the process's mask cannot be
/// pinned, and the pool must keep running, floating. A platform with no
/// per-thread mask refuses the pin by name.
pub fn pin(cpu: u32) -> std::io::Result<()> {
    os::pin(cpu).map_err(std::io::Error::from)
}

/// The cpus a caller that wants one thread a physical core pins its first `t`
/// threads to: the shape the pool would build now ([`plan_pool`]'s), each
/// thread's cpu the pool's own placement rule (`Shape::cpu_for`, the
/// CCD-spread primaries, one a core). Refused by name — a pinned measurement
/// must not run on a guess — when sysfs listed no topology the process's mask
/// shares cpus with (the mask's cpus are not known to be one a core) or the
/// host has fewer physical cores than `t`.
pub fn pinned_cpus(t: usize) -> Result<Vec<u32>, String> {
    let shape = shape_now(None).map_err(|e| format!("sched_getaffinity: {e}"))?;
    if !shape.topology_read || shape.cut.as_ref().is_some_and(|cut| cut.from_mask) {
        return Err(
            "sysfs lists no cpu topology the process's mask shares cpus with: the mask's \
             cpus are not known to be one a physical core"
                .to_string(),
        );
    }
    if t > shape.physical {
        return Err(format!(
            "{t} threads over {} physical cores: a pin needs one cpu a core",
            shape.physical
        ));
    }
    Ok((0..t).map(|i| shape.cpu_for(i)).collect())
}

#[cfg(test)]
mod tests {
    use super::Pool;
    use std::sync::atomic::Ordering;

    /// Run `f` on a thread pinned to `cpus`, so the kernel's mask for it is
    /// theirs: a test of the real mask read.
    fn on_cpus<T: Send + 'static>(cpus: Vec<u32>, f: impl FnOnce() -> T + Send + 'static) -> T {
        std::thread::spawn(move || {
            os::set_mask(&cpus).unwrap_or_else(|e| panic!("the test thread pins to {cpus:?}: {e}"));
            f()
        })
        .join()
        .expect("the test thread ends")
    }

    /// The pool reads the calling thread's mask and cuts its topology by
    /// it: on a thread of one cpu it plans a one-participant pool of that cpu
    /// and says the mask cut the topology (when the process had more), on a
    /// thread of two the pool takes two cores or one with its sibling, and a
    /// thread count the lever gave stands over it. The unpinned process sees
    /// the whole mask, the topology whole and no cut.
    #[cfg_attr(
        not(target_os = "linux"),
        ignore = "the Linux fact it reads: a thread's own affinity mask"
    )]
    #[test]
    fn the_pool_plans_inside_the_threads_mask() {
        let whole = super::mask_cpus().expect("the mask reads");
        assert!(!whole.is_empty());
        let levers = bloomery_levers::pool_levers().expect("the pool levers");
        let plan = super::plan_pool().expect("a plan");
        assert_eq!(plan.allowed, whole);
        let first = whole[0];
        let one = on_cpus(vec![first], || super::plan_pool().expect("a plan"));
        assert_eq!(one.allowed, vec![first]);
        assert_eq!(
            one.threads,
            levers.threads.unwrap_or(1),
            "a thread of one cpu is one core"
        );
        assert!(one.workers.iter().all(|&w| w == first), "{one:?}");
        assert_eq!(one.caller, first);
        assert_eq!(
            one.cut.is_some(),
            whole.len() > 1,
            "the mask cuts the topology unless it was one cpu already"
        );
        if whole.len() >= 2 {
            let pair = vec![whole[0], whole[1]];
            let two = on_cpus(pair.clone(), || super::plan_pool().expect("a plan"));
            assert_eq!(two.allowed, pair);
            assert!(
                two.workers
                    .iter()
                    .chain([&two.caller])
                    .all(|c| pair.contains(c))
                    || levers.threads.is_some_and(|n| n > 2),
                "{two:?}"
            );
            assert!(two.cut.is_some(), "{two:?}");
        }
    }

    /// The built pool and the plan before it are one placement, and a pool
    /// built on a one-cpu thread pins its workers inside that mask.
    #[cfg_attr(
        not(target_os = "linux"),
        ignore = "the Linux fact it reads: a thread's own affinity mask"
    )]
    #[test]
    fn a_built_pool_is_the_plan() {
        let planned = super::plan_pool().expect("a plan");
        let p = Pool::build();
        assert_eq!(p.plan(), planned);
        let first = super::mask_cpus().expect("the mask reads")[0];
        let (threads, plan) = on_cpus(vec![first], || {
            let p = Pool::build();
            (p.threads(), p.plan())
        });
        assert_eq!(plan.allowed, vec![first]);
        assert!(plan.workers.iter().all(|&w| w == first));
        assert_eq!(plan.threads, threads);
    }

    /// A pin the kernel refuses is a named line once, the pool records it,
    /// and its plan lists no pinned worker; a cpu the set cannot hold is the
    /// kernel's refusal.
    #[test]
    fn a_refused_pin_is_named_once() {
        let p = Pool::build();
        assert!(!p.pin_failed());
        let e = std::io::Error::from_raw_os_error(libc::EPERM);
        let line = p
            .note_pin_failure(3, 17, &e)
            .expect("the first failure is said");
        assert!(
            line.contains("worker 3") && line.contains("cpu 17") && line.contains("floats"),
            "{line}"
        );
        assert!(p.note_pin_failure(4, 18, &e).is_none(), "said once");
        assert!(p.pin_failed());
        assert!(p.plan().workers.is_empty());
        let refused = super::pin(1 << 20).expect_err("no cpu_set_t holds that cpu");
        // Linux: the cpu is past the mask's size, and the refusal says so.
        // A platform with no per-thread mask refuses the pin by name.
        #[cfg(target_os = "linux")]
        {
            assert_eq!(refused.kind(), std::io::ErrorKind::InvalidInput);
            assert!(refused.to_string().contains("1048576"), "{refused}");
        }
        #[cfg(target_os = "macos")]
        assert_eq!(refused.kind(), std::io::ErrorKind::Unsupported, "{refused}");
    }

    /// A pool has a CCD map only once every participant is pinned: the
    /// caller by `pin_caller`, the workers by their own pins.
    #[test]
    fn ccd_map_needs_every_participant_pinned() {
        let p: &'static Pool = Box::leak(Box::new(Pool::build()));
        assert!(p.ccd_map().is_none(), "the caller has not pinned");
        p.caller_pinned.store(true, Ordering::Relaxed);
        let map = p.ccd_map().expect("a pinned caller and no failed pin");
        assert_eq!(
            (map.threads(), map.ccds()),
            (p.threads(), p.topology().len())
        );
        p.pin_failed.store(true, Ordering::Relaxed);
        assert!(p.ccd_map().is_none(), "a worker failed to pin");
    }
}
