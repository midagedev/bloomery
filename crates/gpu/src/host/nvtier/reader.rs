//! The NVMe tier's fill reader: threads the tier owns, started at its build,
//! that run a fill's units (one part's read a unit) off the caller. Each
//! thread is blocked in one read at a time, so the threads are the queue
//! depth the drive is read at. A fill is three parts, so one miss is three
//! units and those reads are in flight together. They never take the pool's
//! dispatch lock, the books' lock or wait on the card: a job is its own data
//! (the mapping, the files, the spans), and its completion is a latch the
//! caller waits on.
//!
//! One queue, by rank: a job of a lower rank is taken before every job of a
//! higher one, first come first served within a rank. A job's rank is its
//! [`Kind`]'s: a decode step's fill goes ahead of every queued prompt batch's,
//! and a prompt batch's ahead of every queued victim fill.
//! A job's units are taken in order, so the queue holds one front job at a
//! time and the threads that find it taken go on to the next.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::GpuError;
use model::moe::Claim;

/// The work of one unit of a job: one part of one pick. Runs on a reader thread.
type Unit = Box<dyn Fn(usize) -> Result<(), GpuError> + Send + Sync>;

/// Why a job's unit failed: the first failure of the job, by the unit's index.
#[derive(Debug)]
pub(super) struct Failure {
    pub unit: usize,
    pub error: GpuError,
}

/// What a job fills for, which sets its place in the queue: the one mapping
/// of a kind to a rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// A decode step's misses, read while the step computes its hits and
    /// waited for right after: ahead of every queued prompt fill, so a
    /// concurrent prompt batch does not stall a step.
    Step,
    /// A prompt batch's misses, read ahead of the compute that waits on them.
    Prompt,
    /// A residency flip's victim id: read inside the flip's window, behind
    /// every queued prompt fill.
    Victim,
}

/// The kinds a job is: the length of every per-kind counter.
const KINDS: usize = 3;

impl Kind {
    /// The queue rank: lower is taken first, and the index of the kind's
    /// counters.
    fn rank(self) -> u8 {
        match self {
            Kind::Step => 0,
            Kind::Prompt => 1,
            Kind::Victim => 2,
        }
    }

    fn from_rank(rank: u8) -> Kind {
        match rank {
            0 => Kind::Step,
            1 => Kind::Prompt,
            _ => Kind::Victim,
        }
    }

    /// The kind of the fill of a batch claimed as `claim`: the one mapping of
    /// a claim to a rank.
    pub(super) fn of_claim(claim: Claim) -> Kind {
        match claim {
            Claim::Step => Kind::Step,
            Claim::Keep | Claim::Transient => Kind::Prompt,
        }
    }
}

/// A fill in flight: `units` units of work, each run once on some reader
/// thread, and the latch [`Reader::wait`] blocks on.
pub(super) struct Job {
    /// The kind's rank. Written by [`Job::rearm`] only while the job is not
    /// queued, and read by the reader once it is.
    kind: AtomicU8,
    /// Units this submission runs. Same rule as `kind`: a reader copies the
    /// value onto its queue entry at insert, so a later rearm does not race
    /// the threads that already hold the job.
    units: AtomicUsize,
    run: Unit,
    state: Mutex<JobState>,
    done: Condvar,
}

struct JobState {
    /// Units not yet finished (run, failed or cancelled).
    left: usize,
    /// The first failure; the units after it are skipped, not run.
    failed: Option<Failure>,
    /// The first unit's start and the last unit's end: the job's own wall.
    first: Option<Instant>,
    last: Option<Instant>,
    /// The job's wall is in its kind's `run_ns`: counted by the first wait.
    run_counted: bool,
}

impl Job {
    /// A `kind` job of `units` units, each `run(unit)` once.
    pub(super) fn new(
        kind: Kind,
        units: usize,
        run: impl Fn(usize) -> Result<(), GpuError> + Send + Sync + 'static,
    ) -> Job {
        Job {
            kind: AtomicU8::new(kind.rank()),
            units: AtomicUsize::new(units),
            run: Box::new(run),
            state: Mutex::new(JobState {
                left: units,
                failed: None,
                first: None,
                last: None,
                run_counted: false,
            }),
            done: Condvar::new(),
        }
    }

    fn kind(&self) -> Kind {
        Kind::from_rank(self.kind.load(Ordering::Relaxed))
    }

    fn units(&self) -> usize {
        self.units.load(Ordering::Relaxed)
    }

    /// Arm this job for another submission. The caller has waited it out, and
    /// has not queued it again: the closure stays the one [`Job::new`] boxed.
    pub(super) fn rearm(&self, kind: Kind, units: usize) {
        self.kind.store(kind.rank(), Ordering::Relaxed);
        self.units.store(units, Ordering::Relaxed);
        let mut s = self.lock();
        s.left = units;
        s.failed = None;
        s.first = None;
        s.last = None;
        s.run_counted = false;
    }

    fn lock(&self) -> MutexGuard<'_, JobState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// One unit finished as `result`: the first failure is kept, the job's
    /// last end moves, and the latch opens when no unit is left.
    fn finish_unit(&self, unit: usize, result: Result<(), GpuError>, started: Option<Instant>) {
        let mut s = self.lock();
        if let Err(error) = result {
            s.failed.get_or_insert(Failure { unit, error });
        }
        if let Some(t) = started {
            s.first.get_or_insert(t);
            s.last = Some(Instant::now());
        }
        s.left -= 1;
        if s.left == 0 {
            self.done.notify_all();
        }
    }

    /// Units `from ..` will never run, the tier having stopped: they finish
    /// as the failure `error` names, so a waiter is released.
    fn cancel(&self, from: usize, error: impl Fn() -> GpuError) {
        for unit in from..self.units() {
            self.finish_unit(unit, Err(error()), None);
        }
    }

    /// Whether a unit of the job still runs or waits: the job's latch is
    /// open.
    fn is_open(&self) -> bool {
        self.lock().left > 0
    }
}

/// What the readers count of one kind of job.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KindStats {
    /// Jobs submitted, and their units (one part of a pick) — the fills that
    /// ran off the caller. A fill is three units.
    pub jobs: u64,
    pub units: u64,
    /// Waits that found their job unfinished, and the wall they blocked
    /// (ns): the part of the fills the waiter's own work did not hide.
    pub waits: u64,
    pub wait_ns: u64,
    /// The wall of the jobs a wait saw end (ns, each job's first unit's start
    /// to its last unit's end): what the units ran, beside the wake and the
    /// wait around them.
    pub run_ns: u64,
}

/// What the readers count ([`Reader::stats`]): the decode steps' fills, the
/// prompt batches' and the residency flips' victim fills, each kind alone.
/// `step.wait_ns` is the wall a step's compute waited for fills after the
/// overlap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReaderStats {
    pub step: KindStats,
    pub prompt: KindStats,
    pub victim: KindStats,
}

impl ReaderStats {
    /// What moved since `base`, an earlier reading of the same reader.
    #[must_use]
    pub fn since(&self, base: &ReaderStats) -> ReaderStats {
        let kind = |now: KindStats, then: KindStats| KindStats {
            jobs: now.jobs - then.jobs,
            units: now.units - then.units,
            waits: now.waits - then.waits,
            wait_ns: now.wait_ns - then.wait_ns,
            run_ns: now.run_ns - then.run_ns,
        };
        ReaderStats {
            step: kind(self.step, base.step),
            prompt: kind(self.prompt, base.prompt),
            victim: kind(self.victim, base.victim),
        }
    }
}

#[derive(Default)]
struct KindCounters {
    jobs: AtomicU64,
    units: AtomicU64,
    waits: AtomicU64,
    wait_ns: AtomicU64,
    run_ns: AtomicU64,
}

#[derive(Default)]
struct Counters {
    /// A kind's counters, at its rank.
    kinds: [KindCounters; KINDS],
}

impl Counters {
    fn of(&self, kind: Kind) -> &KindCounters {
        &self.kinds[usize::from(kind.rank())]
    }
}

/// A job waiting for its units to be taken: `next` is the first unit no
/// thread took yet.
struct Entry {
    rank: u8,
    job: Arc<Job>,
    next: usize,
    /// Copied from the job at insert, so the queue does not read `job.units`
    /// while a later rearm stores a new count.
    units: usize,
}

struct Queue {
    /// Jobs with a unit still to take, the front one first: by rank, then
    /// by arrival.
    jobs: VecDeque<Entry>,
    stop: bool,
    /// Threads running their loop.
    alive: usize,
}

struct Shared {
    queue: Mutex<Queue>,
    /// Signalled when a job arrives or the reader stops.
    work: Condvar,
    /// Signalled when a thread leaves its loop.
    exited: Condvar,
    /// Jobs submitted and not finished: a stop that finds one open has a
    /// ticket that outlived the tier.
    open: Mutex<Vec<Arc<Job>>>,
    counters: Counters,
}

impl Shared {
    fn queue(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A reader thread's loop: take the front job's next unit, run it, until
    /// the reader stops. A unit that panics fails its job by name; the
    /// thread goes on.
    fn work(&self) {
        let mut q = self.queue();
        loop {
            if q.stop {
                break;
            }
            let Some(front) = q.jobs.front_mut() else {
                q = self.work.wait(q).unwrap_or_else(|e| e.into_inner());
                continue;
            };
            let unit = front.next;
            front.next += 1;
            let job = Arc::clone(&front.job);
            if front.next == front.units {
                q.jobs.pop_front();
            }
            drop(q);
            self.run_unit(&job, unit);
            q = self.queue();
        }
        q.alive -= 1;
        self.exited.notify_all();
    }

    fn run_unit(&self, job: &Job, unit: usize) {
        if job.lock().failed.is_some() {
            // A job that failed runs no more of its units.
            job.finish_unit(unit, Ok(()), None);
            return;
        }
        let started = Instant::now();
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (job.run)(unit)));
        let result = ran.unwrap_or_else(|p| {
            let what = p
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| p.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a non-string payload".to_string());
            Err(GpuError::protocol(
                "NvTier::fill",
                format!("a fill thread panicked in unit {unit}: {what}"),
            ))
        });
        job.finish_unit(unit, result, Some(started));
    }
}

/// Why [`Reader::stop`] could not end cleanly.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum StopError {
    /// Jobs were still open: a ticket outlived the tier. Their queued units
    /// were cancelled and the running ones ended before the threads joined.
    Open { jobs: usize },
    /// Threads still read past the bound: a read the drive did not return.
    Stuck { threads: usize, bound: Duration },
}

impl std::fmt::Display for StopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopError::Open { jobs } => write!(
                f,
                "the NVMe tier's fill reader stopped with {jobs} batch fill(s) still open: a \
                 ticket outlived the tier"
            ),
            StopError::Stuck { threads, bound } => write!(
                f,
                "the NVMe tier's fill reader still has {threads} thread(s) in a read after {} s",
                bound.as_secs()
            ),
        }
    }
}

/// The readers: `threads` of them, a queue by rank and the counters.
pub(super) struct Reader {
    shared: Arc<Shared>,
    handles: Mutex<Vec<JoinHandle<()>>>,
}

impl Reader {
    /// `threads` reader threads named `name`-`i`, started now. Refused by
    /// name: a thread the host would not start.
    pub(super) fn start(threads: usize, name: &str) -> Result<Reader, GpuError> {
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                jobs: VecDeque::new(),
                stop: false,
                alive: 0,
            }),
            work: Condvar::new(),
            exited: Condvar::new(),
            open: Mutex::new(Vec::new()),
            counters: Counters::default(),
        });
        let mut handles = Vec::with_capacity(threads);
        for i in 0..threads {
            shared.queue().alive += 1;
            // Floating: the tier may be built on a thread pinned to one cpu,
            // whose mask a plain spawn would hand all of the readers.
            let spawned = threads::helper::spawn_helper(
                &format!("{name}-{i}"),
                threads::helper::Placement::Float,
                {
                    let shared = Arc::clone(&shared);
                    move || shared.work()
                },
            );
            match spawned {
                Ok((h, _)) => handles.push(h),
                Err(e) => {
                    let reader = Reader {
                        shared,
                        handles: Mutex::new(handles),
                    };
                    reader.shared.queue().alive -= 1;
                    // Stops the threads already started before the refusal.
                    let _ = reader.stop(Duration::from_secs(10));
                    return Err(GpuError::shape(
                        "NvTier::of_paged",
                        format!("the fill reader's thread {i}: {e}"),
                    ));
                }
            }
        }
        Ok(Reader {
            shared,
            handles: Mutex::new(handles),
        })
    }

    /// Queue `job`: its units run on the reader threads, after the jobs of a
    /// lower or equal rank (its kind's) already queued. Refused by name once
    /// the reader has stopped.
    pub(super) fn submit(&self, job: Job) -> Result<Arc<Job>, GpuError> {
        let job = Arc::new(job);
        self.enqueue(&job)?;
        Ok(job)
    }

    /// Queue `job` again. [`Job::rearm`] has set its kind and its units, and
    /// the job is the one a previous wait finished: the open list drops that
    /// arc before this submission is pushed, so one job is not open twice.
    pub(super) fn resubmit(&self, job: &Arc<Job>) -> Result<(), GpuError> {
        self.enqueue(job)
    }

    fn enqueue(&self, job: &Arc<Job>) -> Result<(), GpuError> {
        let rank = job.kind().rank();
        let units = job.units();
        let mut q = self.shared.queue();
        if q.stop {
            return Err(GpuError::protocol(
                "NvTier::fill",
                "a fill queued after the tier's fill reader stopped".to_string(),
            ));
        }
        let c = self.shared.counters.of(job.kind());
        c.jobs.fetch_add(1, Ordering::Relaxed);
        c.units.fetch_add(units as u64, Ordering::Relaxed);
        {
            let mut open = self.shared.open.lock().unwrap_or_else(|e| e.into_inner());
            open.retain(|j| !Arc::ptr_eq(j, job) && j.is_open());
            open.push(Arc::clone(job));
        }
        if units > 0 {
            let at = q.jobs.partition_point(|e| e.rank <= rank);
            q.jobs.insert(
                at,
                Entry {
                    rank,
                    job: Arc::clone(job),
                    next: 0,
                    units,
                },
            );
            drop(q);
            self.shared.work.notify_all();
        }
        Ok(())
    }

    /// Block until every unit of `job` has finished: the job's wall (ns, its
    /// first unit's start to its last unit's end), or its first failure. The
    /// jobs' own failures reach the caller here, never as a panic on a
    /// reader.
    pub(super) fn wait(&self, job: &Job) -> Result<u64, Failure> {
        let c = self.shared.counters.of(job.kind());
        let mut s = job.lock();
        if s.left > 0 {
            let t0 = Instant::now();
            while s.left > 0 {
                s = job.done.wait(s).unwrap_or_else(|e| e.into_inner());
            }
            c.waits.fetch_add(1, Ordering::Relaxed);
            c.wait_ns
                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        let wall = match (s.first, s.last) {
            (Some(a), Some(b)) => b.duration_since(a).as_nanos() as u64,
            _ => 0,
        };
        if !std::mem::replace(&mut s.run_counted, true) {
            c.run_ns.fetch_add(wall, Ordering::Relaxed);
        }
        if let Some(f) = s.failed.take() {
            return Err(f);
        }
        Ok(wall)
    }

    /// The readers' counters.
    pub(super) fn stats(&self) -> ReaderStats {
        let of = |kind: Kind| {
            let c = self.shared.counters.of(kind);
            KindStats {
                jobs: c.jobs.load(Ordering::Relaxed),
                units: c.units.load(Ordering::Relaxed),
                waits: c.waits.load(Ordering::Relaxed),
                wait_ns: c.wait_ns.load(Ordering::Relaxed),
                run_ns: c.run_ns.load(Ordering::Relaxed),
            }
        };
        ReaderStats {
            step: of(Kind::Step),
            prompt: of(Kind::Prompt),
            victim: of(Kind::Victim),
        }
    }

    /// Stop the readers and join them within `bound`: queued units are
    /// cancelled, a unit already reading ends, then the threads leave.
    /// Refused by name: a job still open (a ticket outlived the tier), and
    /// threads still in a read past the bound — those are left to leave on
    /// their own, since what a job holds is its own, never freed under it.
    pub(super) fn stop(&self, bound: Duration) -> Result<(), StopError> {
        let open = {
            let mut open = self.shared.open.lock().unwrap_or_else(|e| e.into_inner());
            open.retain(|j| j.is_open());
            open.len()
        };
        let cancelled = {
            let mut q = self.shared.queue();
            q.stop = true;
            std::mem::take(&mut q.jobs)
        };
        self.shared.work.notify_all();
        for e in cancelled {
            e.job.cancel(e.next, || {
                GpuError::protocol(
                    "NvTier::fill",
                    "the tier stopped with this batch's fill queued".to_string(),
                )
            });
        }
        let deadline = Instant::now() + bound;
        let mut q = self.shared.queue();
        while q.alive > 0 {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                return Err(StopError::Stuck {
                    threads: q.alive,
                    bound,
                });
            };
            q = self
                .shared
                .exited
                .wait_timeout(q, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        drop(q);
        let handles = std::mem::take(&mut *self.handles.lock().unwrap_or_else(|e| e.into_inner()));
        for h in handles {
            // A thread that panicked outside a unit has nothing left to run.
            let _ = h.join();
        }
        if open > 0 {
            return Err(StopError::Open { jobs: open });
        }
        Ok(())
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        // A reader dropped on its own (a build that refused) leaves no
        // thread behind; the tier's drop has stopped its own, so this is a
        // second, clean stop.
        let _ = self.stop(Duration::from_secs(10));
    }
}

#[cfg(test)]
mod tests {
    use super::{Failure, Job, Kind, KindStats, Reader, StopError};
    use crate::GpuError;
    use model::moe::Claim;
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// A unit that reports its start on `started` and holds until `release`
    /// answers: the test's latch, in place of a sleep.
    fn held(
        started: mpsc::Sender<usize>,
        release: mpsc::Receiver<()>,
    ) -> impl Fn(usize) -> Result<(), GpuError> + Send + Sync + 'static {
        let release = Mutex::new(release);
        move |unit| {
            started.send(unit).expect("the test listens");
            release
                .lock()
                .unwrap()
                .recv()
                .expect("the test releases the unit");
            Ok(())
        }
    }

    /// A step's fill is taken before every queued prompt fill and a prompt
    /// fill before every queued victim fill, whenever each was submitted, and
    /// the units of a kind in arrival order; a claim's kind is the one
    /// [`Kind::of_claim`] maps it to. One thread, so the order the units start
    /// in is the order the queue pops them.
    #[test]
    fn the_reader_takes_step_fills_before_prompts_before_victims() {
        let reader = Reader::start(1, "test-fill").unwrap();
        let (started_tx, started) = mpsc::channel();
        let (release_tx, release) = mpsc::channel();
        let log = Arc::new(Mutex::new(Vec::new()));
        let tagged = |tag: &'static str, log: &Arc<Mutex<Vec<&'static str>>>| {
            let log = Arc::clone(log);
            move |_unit: usize| -> Result<(), GpuError> {
                log.lock().unwrap().push(tag);
                Ok(())
            }
        };
        // The thread is held by a first job, so the next four queue behind
        // it: a victim, a prompt batch behind that victim, another victim
        // and another prompt batch.
        let first = reader
            .submit(Job::new(Kind::Prompt, 1, held(started_tx, release)))
            .unwrap();
        assert_eq!(started.recv().unwrap(), 0);
        let jobs: Vec<_> = [
            (Kind::Victim, "victim"),
            (Kind::Prompt, "prompt"),
            (Kind::Victim, "victim-later"),
            (Kind::Prompt, "prompt-later"),
            (Kind::of_claim(Claim::Step), "step"),
            (Kind::of_claim(Claim::Keep), "keep-later"),
        ]
        .into_iter()
        .map(|(kind, tag)| reader.submit(Job::new(kind, 1, tagged(tag, &log))).unwrap())
        .collect();
        release_tx.send(()).unwrap();
        for j in std::iter::once(&first).chain(&jobs) {
            assert!(reader.wait(j).is_ok());
        }
        assert_eq!(
            *log.lock().unwrap(),
            [
                "step",
                "prompt",
                "prompt-later",
                "keep-later",
                "victim",
                "victim-later"
            ]
        );
        let stats = reader.stats();
        let counted = |k: KindStats| (k.jobs, k.units);
        assert_eq!(
            (
                counted(stats.step),
                counted(stats.prompt),
                counted(stats.victim)
            ),
            ((1, 1), (4, 4), (2, 2)),
            "each kind counted alone"
        );
        assert!(
            stats.prompt.run_ns > 0,
            "the held job's wall is counted by its wait: {stats:?}"
        );
        reader.stop(Duration::from_secs(10)).unwrap();
    }

    /// The first failing unit is the job's failure, by its index, and the
    /// units after it do not run; the reader thread goes on, a panic in a
    /// unit included, which reaches the wait as a named failure.
    #[test]
    fn a_jobs_first_failure_skips_its_later_units() {
        let reader = Reader::start(1, "test-fill").unwrap();
        let ran = Arc::new(Mutex::new(Vec::new()));
        let job = {
            let ran = Arc::clone(&ran);
            Job::new(Kind::Prompt, 4, move |u| {
                ran.lock().unwrap().push(u);
                if u == 1 {
                    return Err(GpuError::shape("test", "a short read".to_string()));
                }
                Ok(())
            })
        };
        let job = reader.submit(job).unwrap();
        let Failure { unit, error } = reader.wait(&job).unwrap_err();
        assert_eq!(unit, 1);
        assert!(error.to_string().contains("a short read"), "{error}");
        assert_eq!(*ran.lock().unwrap(), [0, 1], "units 2 and 3 were skipped");
        let boom = reader
            .submit(Job::new(Kind::Prompt, 1, |_| panic!("a bad slice")))
            .unwrap();
        let e = reader.wait(&boom).unwrap_err().error.to_string();
        assert!(e.contains("panicked") && e.contains("a bad slice"), "{e}");
        let after = reader
            .submit(Job::new(Kind::Prompt, 1, |_| Ok(())))
            .unwrap();
        assert!(reader.wait(&after).is_ok(), "the thread outlived the panic");
        reader.stop(Duration::from_secs(10)).unwrap();
    }

    /// A stop cancels what is queued and names a job still open; a stop with
    /// every job finished is clean; a fill queued after it is refused by
    /// name.
    #[test]
    fn a_stop_names_an_open_ticket_and_refuses_later_fills() {
        let clean = Reader::start(2, "test-fill").unwrap();
        let ok = clean.submit(Job::new(Kind::Prompt, 3, |_| Ok(()))).unwrap();
        assert!(clean.wait(&ok).is_ok());
        assert_eq!(clean.stop(Duration::from_secs(10)), Ok(()));
        let e = clean
            .submit(Job::new(Kind::Prompt, 1, |_| Ok(())))
            .err()
            .unwrap();
        assert!(e.to_string().contains("stopped"), "{e}");

        let reader = Reader::start(1, "test-fill").unwrap();
        let (started_tx, started) = mpsc::channel();
        let (release_tx, release) = mpsc::channel();
        let running = reader
            .submit(Job::new(Kind::Prompt, 1, held(started_tx, release)))
            .unwrap();
        assert_eq!(started.recv().unwrap(), 0);
        let queued = reader
            .submit(Job::new(Kind::Prompt, 2, |_| Ok(())))
            .unwrap();
        // Releases the running unit from another thread once the stop has
        // counted the open jobs: `stop` counts them before it sets the
        // queue's flag, so a release seen after the flag cannot let the two
        // jobs finish ahead of the count.
        let shared = Arc::clone(&reader.shared);
        let (releaser, _) = threads::helper::spawn_helper(
            "test-release",
            threads::helper::Placement::Float,
            move || {
                while !shared.queue().stop {
                    std::thread::sleep(Duration::from_millis(1));
                }
                release_tx.send(()).unwrap();
            },
        )
        .unwrap();
        let stopped = reader.stop(Duration::from_secs(10));
        releaser.join().unwrap();
        assert_eq!(stopped, Err(StopError::Open { jobs: 2 }));
        let cancelled = reader.wait(&queued).unwrap_err().error.to_string();
        assert!(cancelled.contains("queued"), "{cancelled}");
        assert!(reader.wait(&running).is_ok());
    }

    /// A thread held in a read past the bound is a named refusal, not a
    /// hang; the read, once it returns, finishes its job.
    #[test]
    fn a_stop_past_its_bound_names_the_stuck_threads() {
        let reader = Reader::start(1, "test-fill").unwrap();
        let (started_tx, started) = mpsc::channel();
        let (release_tx, release) = mpsc::channel();
        let job = reader
            .submit(Job::new(Kind::Prompt, 1, held(started_tx, release)))
            .unwrap();
        assert_eq!(started.recv().unwrap(), 0);
        let stopped = reader.stop(Duration::from_millis(50));
        assert_eq!(
            stopped,
            Err(StopError::Stuck {
                threads: 1,
                bound: Duration::from_millis(50)
            })
        );
        release_tx.send(()).unwrap();
        assert!(reader.wait(&job).is_ok(), "what a job holds is its own");
        reader.stop(Duration::from_secs(10)).unwrap();
    }
}
