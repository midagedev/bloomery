//! The residency machine's copy lane, which the machine owns
//! ([`super::swap::SwapMachine`]): the staging ring its copy stream copies
//! from, the ring's staged and drained words, the job queue and the
//! [`LANE_THREADS`] threads that fill the ring. The machine issues a job a
//! copy and enqueues the copy behind the job's staged word ([`dispatch`]).
//!
//! **The receive lock.** A lane thread takes the next job in issue order
//! under one lock and, still under it, prepares the job's victim, waits for
//! the job to open (the staging window open, a flush, or the job due) and,
//! within the deadline, for its ring slot's previous copy to drain. Then it
//! copies the expert's bytes into the slot and raises the word: inside the
//! lock for a job the window opened, so those stage one at a time beside
//! the steps the window gates; outside it for a job a flush or its landing
//! opened (a prompt call's picks, a reset's and a restore's copies, a late
//! flip), so up to [`LANE_THREADS`] of those stage at once. A victim's
//! preparation stays in issue order under the lock, one lane thread at a
//! time: a source's preparation (the NVMe tier's books and its reads) is
//! written for one lane caller. The lock closes no cycle: job `n` waits only
//! on job `n − RING_SLOTS`'s copy, a job an earlier holder of the lock took,
//! whose staging needs the lock no more. A panic or a poisoned lock stops no
//! other thread. Why no wait here can close a cycle through the card is the
//! machine's to argue (its module doc, **No wait without a bound** and **No
//! cycle through the card**).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cuda_core::{CudaContext, CudaStream, sys};

use super::swap::{SwapSource, Transform};
use super::xstream::FILL_THREADS;
use crate::GpuError;
use crate::graph::{MappedHost, cu};

/// Experts the staging ring holds at once.
pub const RING_SLOTS: usize = 4;

/// The lane's threads. Jobs the staging window opened stage on one of them
/// at a time; jobs a flush or their landing opened stage on all of them at
/// once.
pub const LANE_THREADS: usize = 4;

/// Bytes between two words of the staging page: a cache line each.
pub(super) const WORD_STRIDE: usize = 64;

/// Whole nanoseconds since the epoch `t0`, saturated: the staging stamps'
/// and the pick starts' common clock.
fn nanos_since(t0: Instant) -> u64 {
    u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Whole nanoseconds of `t` since the epoch `t0`, saturated: a stamp's own
/// clock read (an already-taken `Instant` against the epoch — no new one).
pub(super) fn since_epoch(t0: Instant, t: Instant) -> u64 {
    u64::try_from(t.duration_since(t0).as_nanos()).unwrap_or(u64::MAX)
}

/// How many jobs the stamp ring holds cells for: a job's cell is the next
/// job's but this many on. A flip's stamps are read at the boundary that
/// lands it, a boundary or two after its issue; a late flip whose cell a
/// later job took first (a prompt call's picks issue a copy a pick) is noted
/// with its identities alone ([`super::swap::LateAt::Gone`]).
const STAMP_SLOTS: usize = 1024;

/// One job's staging stamps, one cell a job modulo [`STAMP_SLOTS`]: the
/// machine writes the issue stamp ([`JobStamps::issue`]), the lane the serve
/// stamps ([`JobStamps::note_prepare_at`], [`JobStamps::note_prepare`],
/// [`JobStamps::note_staged`]), each field an atomic word — the late-flip
/// ring's reader ([`super::swap::SwapMachine::take_late_flips`]) reads a
/// field or a 0, never a torn word, and a cell whose job tag is not the job
/// it asks for ([`JobStamps::of`]) answers nothing. A cell a cache line, so
/// two lane threads stamping neighbouring jobs never share one.
#[derive(Default)]
#[repr(align(64))]
pub(super) struct StampCell {
    job: AtomicU64,
    queued: AtomicU64,
    ahead: AtomicU64,
    prepare_at: AtomicU64,
    prepare: AtomicU64,
    read: AtomicU64,
    staged: AtomicU64,
}

/// One job's stamps as the late-flip ring reads them, copied, each in
/// nanoseconds since the stamp epoch but `prepare` and `read`, which are
/// lengths: 0 in `prepare_at` names a job whose victim's preparation had not
/// begun, 0 in `prepare` one whose preparation had not ended, and 0 in
/// `staged` one that had not staged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Stamp {
    pub(super) queued: u64,
    pub(super) ahead: u64,
    pub(super) prepare_at: u64,
    pub(super) prepare: u64,
    pub(super) read: u64,
    pub(super) staged: u64,
}

/// The per-job staging stamps ([`StampCell`]), a cell a job modulo
/// [`STAMP_SLOTS`]: what the late-flip ring decomposes a late copy's
/// issue→boundary time with ([`super::swap::LateFlip`]) — the queue wait
/// before the job's preparation (`prepare_at` − `queued`), the victim's
/// preparation, the source read, and the staging word's publish.
pub(super) struct JobStamps {
    cells: Vec<StampCell>,
}

impl JobStamps {
    pub(super) fn new() -> JobStamps {
        JobStamps {
            cells: (0..STAMP_SLOTS).map(|_| StampCell::default()).collect(),
        }
    }

    fn cell(&self, job: u64) -> &StampCell {
        &self.cells[(job % STAMP_SLOTS as u64) as usize]
    }

    /// The machine at job `job`'s issue: `queued` the issue stamp, `ahead`
    /// the lane's backlog then — the copies already issued and not yet
    /// staged, each of which the copy stream waits for ahead of this one.
    /// The serve stamps are cleared here and the job tag written last.
    pub(super) fn issue(&self, job: u64, queued: u64, ahead: u64) {
        let c = self.cell(job);
        c.queued.store(queued, Ordering::Relaxed);
        c.ahead.store(ahead, Ordering::Relaxed);
        c.prepare_at.store(0, Ordering::Relaxed);
        c.prepare.store(0, Ordering::Relaxed);
        c.read.store(0, Ordering::Relaxed);
        c.staged.store(0, Ordering::Relaxed);
        c.job.store(job, Ordering::Release);
    }

    /// The lane as the job's preparation begins: the stamp the queue wait
    /// reads against.
    pub(super) fn note_prepare_at(&self, job: u64, at: u64) {
        self.cell(job).prepare_at.store(at, Ordering::Relaxed);
    }

    /// The lane once the job's victim is prepared: how long the preparation
    /// took, published over the preparation's start.
    pub(super) fn note_prepare(&self, job: u64, took: u64) {
        self.cell(job).prepare.store(took, Ordering::Release);
    }

    /// The lane as the job's staging word is published: the read's length
    /// and the stamp the issue→staged and staged→boundary terms read
    /// against, written last so a `staged` a reader sees covers the read.
    pub(super) fn note_staged(&self, job: u64, at: u64, took: u64) {
        let c = self.cell(job);
        c.read.store(took, Ordering::Relaxed);
        c.staged.store(at, Ordering::Release);
    }

    /// The job's stamps, or `None` when the cell it would use holds another
    /// job, before or after the read: its stamps are gone. Each publishing
    /// word is read before the words it covers: a `staged` seen carries its
    /// `read`, a `prepare` seen its `prepare_at`.
    pub(super) fn of(&self, job: u64) -> Option<Stamp> {
        let c = self.cell(job);
        if c.job.load(Ordering::Acquire) != job {
            return None;
        }
        let staged = c.staged.load(Ordering::Acquire);
        let read = c.read.load(Ordering::Relaxed);
        let prepare = c.prepare.load(Ordering::Acquire);
        let stamp = Stamp {
            queued: c.queued.load(Ordering::Relaxed),
            ahead: c.ahead.load(Ordering::Relaxed),
            prepare_at: c.prepare_at.load(Ordering::Relaxed),
            prepare,
            read,
            staged,
        };
        (c.job.load(Ordering::Acquire) == job).then_some(stamp)
    }
}

/// Where lane thread `t` runs: the SMT sibling of the pool's worker
/// [`FILL_THREADS`]` + t`, past the expert stream's fill threads on the
/// first workers' siblings. The copies are DRAM streams that sweep a ring
/// slot's worth of L3; the pool spreads its workers over the CCDs in turn,
/// so the lane's threads spread over them too, none on the dispatcher's core
/// (the critical path, whose sibling the engram helper takes). A prompt
/// call's copies run beside the union the pool's workers compute, so those
/// workers share their cores with the copies then. Floating, thread by
/// thread, where the pool holds or pins no such worker.
fn lane_placement(t: usize) -> threads::helper::Placement {
    match threads::built().and_then(|p| p.worker_cpu(FILL_THREADS + t)) {
        Some(c) => threads::helper::Placement::Sibling(c),
        None => threads::helper::Placement::Float,
    }
}

/// Job `n`'s ring slot and its ticket there (its use of that slot, from 1):
/// the one owner of the job → ring mapping.
pub(super) fn ring_ticket(n: u64) -> Result<(usize, u32), GpuError> {
    let ring = (n % RING_SLOTS as u64) as usize;
    let ticket = u32::try_from(n / RING_SLOTS as u64 + 1)
        .map_err(|_| GpuError::protocol("SwapMachine staging", "the staging tickets passed u32"))?;
    Ok((ring, ticket))
}

/// The one way a wait on a staging word goes on the copy stream: a copy
/// stream wait whose word only the lane raises is enqueued only for a job
/// the lane's queue already holds, so the thread enqueueing never waits on
/// a wait it has yet to release.
pub(super) mod dispatch {
    use std::sync::mpsc;

    use cuda_core::CudaStream;

    use super::{Job, WORD_STRIDE};
    use crate::GpuError;
    use crate::graph::{MappedHost, mem_batch, op_wait_geq};

    /// A job the lane holds. Only [`send`] makes one.
    pub(in crate::host) struct Dispatched {
        ring: usize,
        ticket: u32,
    }

    impl Dispatched {
        pub(in crate::host) fn ring(&self) -> usize {
            self.ring
        }

        pub(in crate::host) fn ticket(&self) -> u32 {
            self.ticket
        }
    }

    /// `job` to the lane over `tx`; refused by name as `what` when its
    /// threads have stopped.
    pub(super) fn send(
        tx: Option<&mpsc::Sender<Job>>,
        job: Job,
        what: &'static str,
    ) -> Result<Dispatched, GpuError> {
        let (ring, ticket) = (job.ring, job.ticket);
        tx.and_then(|tx| tx.send(job).ok())
            .ok_or_else(|| GpuError::protocol(what, "the lane has stopped"))?;
        Ok(Dispatched { ring, ticket })
    }

    /// Enqueue on `copy` the wait for `d`'s staging: its ring slot's staged
    /// word in `words` at its ticket.
    pub(in crate::host) fn wait_staged(
        copy: &CudaStream,
        words: &MappedHost,
        d: &Dispatched,
    ) -> Result<(), GpuError> {
        let staged = words.dev_at(2 * d.ring * WORD_STRIDE);
        mem_batch(
            copy,
            &mut [op_wait_geq(staged, d.ticket)],
            "swap: wait for staging",
        )
    }
}

/// One expert's copy for the lane: first the victim to prepare, then ring
/// slot `ring`, `ticket` its use of that slot.
pub(super) struct Job {
    /// The job's place in issue order.
    pub(super) n: u64,
    pub(super) layer: usize,
    pub(super) id: u32,
    pub(super) victim: Option<u32>,
    pub(super) ring: usize,
    pub(super) ticket: u32,
    /// The job is a prompt pick's copy, one of the layer's counter the lane
    /// takes from ([`Shared::pick_left`]): a boundary's, a reset's and a
    /// restore's are not.
    pub(super) stamp: bool,
}

/// The staging ring: [`RING_SLOTS`] experts of `slot_bytes`, pinned and
/// device-mapped. Its bytes are written and read here only.
pub(super) struct Ring {
    page: MappedHost,
    slot_bytes: usize,
}

impl Ring {
    fn new(ctx: &Arc<CudaContext>, slot_bytes: usize) -> Result<Ring, GpuError> {
        Ok(Ring {
            page: MappedHost::new(ctx, RING_SLOTS * slot_bytes, "cuMemHostAlloc (swap ring)")?,
            slot_bytes,
        })
    }

    /// The bytes of one ring slot.
    pub(super) fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// The byte offset of `[at, at + len)` of ring slot `k`, or the refusal
    /// of `what` for a span past the slot.
    fn span(&self, k: usize, at: usize, len: usize, what: &'static str) -> Result<usize, GpuError> {
        if k >= RING_SLOTS || at.checked_add(len).is_none_or(|end| end > self.slot_bytes) {
            return Err(GpuError::shape(
                what,
                format!(
                    "bytes [{at}, +{len}) of ring slot {k}: {RING_SLOTS} slots of {}",
                    self.slot_bytes
                ),
            ));
        }
        Ok(k * self.slot_bytes + at)
    }

    /// `bytes` into ring slot `k` at byte `at`. The caller is slot `k`'s one
    /// host writer, the lane thread that holds the slot's job, and writes it
    /// only once the copy stream has read its previous use (`drained`).
    fn write(&self, k: usize, at: usize, bytes: &[u8]) -> Result<(), GpuError> {
        let off = self.span(k, at, bytes.len(), "SwapMachine ring write")?;
        // SAFETY: [off, off + len) lies inside the page (checked above); no
        // copy reads ring slot k until the staging word publishes this use,
        // and the thread holding the slot's job is its only host writer (job
        // n + RING_SLOTS writes the slot only once job n's copy has drained
        // it). The word's Release store follows these stores in program
        // order, which x86 (TSO) keeps in the coherent write-back pinned
        // page, and the copy engine reads the page over PCIe with no SM cache
        // between: the copy the word lets through reads these bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.page.host_at(off), bytes.len());
        }
        Ok(())
    }

    /// Enqueue on `stream` the copy of `len` bytes at byte `at` of ring slot
    /// `k` to device address `dst`.
    pub(super) fn copy_to_device(
        &self,
        k: usize,
        at: usize,
        len: usize,
        dst: sys::CUdeviceptr,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        let off = self.span(k, at, len, "SwapMachine ring copy")?;
        // SAFETY: the source span lies inside the page (checked above), which
        // stays allocated until the copy stream has drained (the machine's
        // drop leaks it otherwise); the source vouches that `dst` holds `len`
        // bytes of an allocation alive until the copy stream has drained.
        let rc = unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                dst,
                self.page.host_at(off).cast_const().cast(),
                len,
                stream.cu_stream(),
            )
        };
        cu(rc, "cuMemcpyHtoDAsync_v2 (swap slot)")
    }
}

/// The lane's first failure: its job and the lane thread's own error.
pub(super) struct StagingFailure {
    pub(super) job: u64,
    pub(super) layer: usize,
    pub(super) id: u32,
    pub(super) error: GpuError,
}

/// What the lane threads and the machine share.
pub(crate) struct Shared {
    pub(super) source: Arc<dyn SwapSource>,
    pub(super) ring: Ring,
    /// Per ring slot: `staged` (the thread's ticket once the slot holds the
    /// expert, or once its failure is recorded) at `2k`, `drained` (the copy
    /// stream's ticket once it has read the slot) at `2k + 1`.
    pub(super) words: MappedHost,
    /// The staging window: staging runs while it is nonzero. A step service
    /// zeroes it from its go's landing to its signal (`host::step`'s
    /// `Closed`), so the staging copies stay out of a layer's host experts;
    /// at every other moment it is open.
    pub(super) window: Arc<AtomicU32>,
    /// Jobs below this are due: their flips land at a boundary the host has
    /// reached, so they stage whatever the window says.
    pub(super) due: AtomicU64,
    /// The bound on every host wait ([`super::swap::MachineCfg::deadline`]).
    pub(super) deadline: Duration,
    /// Stage whatever the window says: a quiet boundary's relayout.
    pub(super) flush: AtomicBool,
    /// Nanoseconds the lane threads spent copying into the ring and
    /// preparing victims, summed over the threads, since a boundary last
    /// took them.
    pub(super) stage_ns: AtomicU64,
    pub(super) prepare_ns: AtomicU64,
    /// Jobs the lane has finished (staged, or failed), counted before each
    /// one's ticket is published, in the order they finish: `jobs_issued -
    /// served` copies wait for staging.
    pub(super) served: AtomicU64,
    /// Per layer index below the map's last layer (a job names its layer by
    /// its index in the model, not its place in the map), the jobs of the
    /// layer's current pick the lane has not stamped yet: the pick stores its
    /// count before it issues its copies, a lane thread takes one a staged
    /// job.
    pub(super) pick_left: Vec<AtomicU64>,
    /// Per layer index below the map's last layer, the stamp epoch's
    /// nanoseconds at the moment the layer's last outstanding pick job staged
    /// (0: none yet): the pick record's `staged_us` reads against the pick's
    /// own start.
    pub(super) pick_stamp_ns: Vec<AtomicU64>,
    /// The stamp epoch, the machine's birth: every stamp and pick start is
    /// nanoseconds since it.
    pub(super) epoch: Instant,
    /// The per-job staging stamps ([`JobStamps`]), for the machine's
    /// late-flip ring.
    pub(super) stamps: JobStamps,
    stop: AtomicBool,
    /// The first staging failure, which the next boundary or reset returns.
    failed: Mutex<Option<StagingFailure>>,
    /// Per lane thread, the jobs it took ([`LaneStats::taken`]).
    taken: [AtomicU64; LANE_THREADS],
    /// The jobs staging now that the window opened, and those a flush or
    /// their landing opened; the most of each at once ([`LaneStats`]).
    in_window: AtomicU32,
    in_open: AtomicU32,
    peak_window: AtomicU32,
    peak_open: AtomicU32,
    /// The machine's own job count and the next boundary one of its flips
    /// lands at, mirrored for a reader that holds no machine (the stall
    /// note, [`Shared::stall_note`]; the engine watchdog): stored where the
    /// machine changes them. [`NO_FLIP_LANDING`] when no flip fills a slot.
    pub(super) jobs_issued: AtomicU64,
    pub(super) next_landing: AtomicU64,
}

/// The `next_landing` mirror while no flip is filling a slot.
pub(super) const NO_FLIP_LANDING: u64 = u64::MAX;

impl Shared {
    /// What the machine holds, for the error of an engine wait that ran out
    /// of its bound and for the engine watchdog that names a wedge: the
    /// copies still waiting for staging, each ring slot's staged and drained
    /// tickets, the window, the due line and flush, and the next boundary a
    /// flip lands at. A reader with no machine sees the mirrors, which
    /// trail it only inside the change that stores them.
    pub(crate) fn stall_note(&self) -> String {
        let issued = self.jobs_issued.load(Ordering::Relaxed);
        let waiting = issued.saturating_sub(self.served.load(Ordering::Acquire));
        let ring = (0..RING_SLOTS)
            .map(|k| {
                format!(
                    "{}/{}",
                    self.staged(k).load(Ordering::Acquire),
                    self.drained(k).load(Ordering::Acquire)
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let lands = self.next_landing.load(Ordering::Relaxed);
        format!(
            "the residency machine holds {waiting} of {issued} copies waiting for staging (ring \
             staged/drained {ring}), the staging window {}, due below job {}, flush {}, the next \
             flip landing at boundary {}",
            if self.window.load(Ordering::Acquire) != 0 {
                "open"
            } else {
                "closed"
            },
            self.due.load(Ordering::Acquire),
            self.flush.load(Ordering::Acquire),
            if lands == NO_FLIP_LANDING {
                "none".to_string()
            } else {
                lands.to_string()
            },
        )
    }

    fn word(&self, i: usize) -> &AtomicU32 {
        self.words
            .atomic_u32(i * WORD_STRIDE)
            .expect("the staging page holds 2 * RING_SLOTS words")
    }

    pub(super) fn staged(&self, k: usize) -> &AtomicU32 {
        self.word(2 * k)
    }

    fn drained(&self, k: usize) -> &AtomicU32 {
        self.word(2 * k + 1)
    }

    /// Wait until `ready`: `Ok(true)` when it is, `Ok(false)` when the
    /// machine stops first, `Err(waited)` past `deadline` when one is given.
    pub(super) fn wait_until(
        &self,
        ready: impl Fn() -> bool,
        deadline: Option<Duration>,
    ) -> Result<bool, Duration> {
        let t0 = Instant::now();
        let mut spins = 0u32;
        while !ready() {
            if self.stop.load(Ordering::Acquire) {
                return Ok(false);
            }
            if let Some(d) = deadline
                && t0.elapsed() > d
            {
                return Err(t0.elapsed());
            }
            if spins < 256 {
                std::hint::spin_loop();
                spins += 1;
            } else {
                std::thread::sleep(Duration::from_micros(20));
            }
        }
        Ok(true)
    }

    /// The lane's first failure, taken: the next boundary or reset refuses
    /// with it.
    pub(super) fn take_failure(&self) -> Option<StagingFailure> {
        self.failed
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
    }

    /// Let through every copy the copy stream holds behind a staging word,
    /// `issued` jobs issued: each ring slot's word gets the ticket after the
    /// last one issued to it. The wait is cyclic (`(i32)(word - ticket) >=
    /// 0`), so only a ticket just past the issued ones releases it;
    /// `u32::MAX` would not. For the drop only: the bytes those copies move
    /// land in slots no entry names.
    pub(super) fn release_waits(&self, issued: u64) -> Result<(), GpuError> {
        for n in issued..issued + RING_SLOTS as u64 {
            let (ring, ticket) = ring_ticket(n)?;
            self.staged(ring).fetch_max(ticket, Ordering::AcqRel);
        }
        Ok(())
    }

    /// Copy `job`'s expert from its source into its ring slot, part after
    /// part, at the offsets its layer's parts lay out, between the source's
    /// open of the read and its release, which follows the copy whether it
    /// failed or not ([`SwapSource::open_read`]).
    fn stage(&self, job: &Job) -> Result<(), GpuError> {
        self.source.open_read(job.layer, job.id)?;
        let copied = (|| {
            let mut at = 0usize;
            for (part, &want) in self.source.part_bytes(job.layer).iter().enumerate() {
                let piece = self.source.source(job.layer, job.id, part)?;
                if piece.bytes.len() != want {
                    return Err(GpuError::shape(
                        "SwapMachine staging",
                        format!(
                            "layer {} expert {} part {part}: {} source bytes for a slot of {want}",
                            job.layer,
                            job.id,
                            piece.bytes.len()
                        ),
                    ));
                }
                match piece.transform {
                    Transform::Identity => self.ring.write(job.ring, at, piece.bytes)?,
                }
                at += want;
            }
            Ok(())
        })();
        let released = self.source.release_read(job.layer, job.id);
        copied.and(released)
    }

    /// What the lane did since the load ([`LaneStats`]).
    pub(super) fn lane_stats(&self) -> LaneStats {
        LaneStats {
            taken: std::array::from_fn(|t| self.taken[t].load(Ordering::Relaxed)),
            peak_window: self.peak_window.load(Ordering::Relaxed),
            peak_open: self.peak_open.load(Ordering::Relaxed),
        }
    }

    fn fail(&self, job: &Job, error: GpuError) {
        let mut f = self.failed.lock().unwrap_or_else(PoisonError::into_inner);
        if f.is_none() {
            *f = Some(StagingFailure {
                job: job.n,
                layer: job.layer,
                id: job.id,
                error,
            });
        }
    }

    /// One job, taken under the receive lock `taken`: under it, its victim
    /// prepared, the wait for the window (or a flush, or the job due) and,
    /// within the deadline, for its ring slot's last copy; then the stage,
    /// under the lock for a job the window opened, past it for one a flush
    /// or its landing opened. `Ok(false)` when the machine stops first.
    fn serve(
        &self,
        job: &Job,
        taken: MutexGuard<'_, mpsc::Receiver<Job>>,
    ) -> Result<bool, GpuError> {
        if let Some(v) = job.victim {
            let t0 = Instant::now();
            self.stamps
                .note_prepare_at(job.n, since_epoch(self.epoch, t0));
            self.source.prepare_victim(job.layer, v)?;
            let took = super::nanos(t0.elapsed());
            self.prepare_ns.fetch_add(took, Ordering::Relaxed);
            self.stamps.note_prepare(job.n, took);
        }
        let forced =
            || self.flush.load(Ordering::Acquire) || job.n < self.due.load(Ordering::Acquire);
        let open = || self.window.load(Ordering::Acquire) != 0 || forced();
        // No deadline: the window is closed only while a step service
        // computes one layer. What ends the wait: the window opening, a
        // boundary landing the job (`due`), a reset's flush (`drain_copies`,
        // first thing in `reset`), and the drop's `stop`, which `wait_until`
        // reads on every turn.
        if self.wait_until(open, None) != Ok(true) {
            return Ok(false);
        }
        // What opened it, read once it is open: a flush or its landing
        // before the window.
        let by_window = !forced();
        let prev = job.ticket - 1;
        let drained = || self.drained(job.ring).load(Ordering::Acquire) >= prev;
        match self.wait_until(drained, Some(self.deadline)) {
            Ok(true) => {}
            Ok(false) => return Ok(false),
            Err(waited) => {
                return Err(GpuError::protocol(
                    "SwapMachine staging",
                    format!(
                        "ring slot {} was not drained of its last copy in {waited:?} (deadline \
                         {:?}): the copy stream is stuck",
                        job.ring, self.deadline
                    ),
                ));
            }
        }
        let _staging = if by_window {
            InFlight::enter(&self.in_window, &self.peak_window)
        } else {
            drop(taken);
            InFlight::enter(&self.in_open, &self.peak_open)
        };
        let t0 = Instant::now();
        self.stage(job)?;
        let took = super::nanos(t0.elapsed());
        self.stage_ns.fetch_add(took, Ordering::Relaxed);
        // The staging stamp is the stage's own clock reads against the
        // epoch — no new one.
        self.stamps.note_staged(
            job.n,
            since_epoch(self.epoch, t0).saturating_add(took),
            took,
        );
        Ok(true)
    }

    /// Lane thread `me`: each job taken in issue order under the receive
    /// lock `jobs`, served ([`Shared::serve`]) and its ticket published. A
    /// failure — an error or a panic — is recorded for the next boundary,
    /// which waits for the ticket and refuses before its flip could go live,
    /// and the ticket is published anyway, so the copy stream never hangs on
    /// it: the bytes it copies land in a slot no entry names.
    fn run(&self, me: usize, jobs: &Mutex<mpsc::Receiver<Job>>) {
        loop {
            let taken = jobs.lock().unwrap_or_else(PoisonError::into_inner);
            let Ok(job) = taken.recv() else {
                return;
            };
            self.taken[me].fetch_add(1, Ordering::Relaxed);
            match catch_unwind(AssertUnwindSafe(|| self.serve(&job, taken))) {
                Ok(Ok(true)) => {}
                Ok(Ok(false)) => return,
                Ok(Err(e)) => self.fail(&job, e),
                Err(payload) => {
                    let why = payload
                        .downcast_ref::<&str>()
                        .map(ToString::to_string)
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "a payload that is not text".to_string());
                    let e = GpuError::protocol(
                        "SwapMachine staging",
                        format!("a lane thread panicked: {why}"),
                    );
                    self.fail(&job, e);
                }
            }
            self.served.fetch_add(1, Ordering::AcqRel);
            if job.stamp {
                // The pick's per-layer remaining-jobs counter, one atomic a
                // job: the thread that finishes the layer's last one, the
                // last to finish, stamps the wall for the pick record's
                // `staged_us`, before it publishes the ticket, so a copy that
                // landed has its stamp posted.
                let left = self.pick_left[job.layer].fetch_sub(1, Ordering::AcqRel);
                if left == 1 {
                    self.pick_stamp_ns[job.layer].store(nanos_since(self.epoch), Ordering::Release);
                }
            }
            // A max, never a store: a word the machine released past this
            // ticket stays released.
            self.staged(job.ring)
                .fetch_max(job.ticket, Ordering::AcqRel);
        }
    }
}

/// One job staging, counted in the in-flight count of its kind and that
/// kind's peak raised to it; the count comes down when it drops, a panic's
/// unwind included.
struct InFlight<'a>(&'a AtomicU32);

impl<'a> InFlight<'a> {
    fn enter(now: &'a AtomicU32, peak: &AtomicU32) -> InFlight<'a> {
        let n = now.fetch_add(1, Ordering::AcqRel) + 1;
        peak.fetch_max(n, Ordering::AcqRel);
        InFlight(now)
    }
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// What the lane did since the load ([`super::swap::SwapMachine::lane_stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LaneStats {
    /// Per lane thread, the jobs it took.
    pub taken: [u64; LANE_THREADS],
    /// The most jobs the staging window opened that staged at once: one,
    /// by the receive lock.
    pub peak_window: u32,
    /// The most jobs a flush or their landing opened that staged at once.
    pub peak_open: u32,
}

/// The lane's threads and their job queue, which the machine holds beside
/// the shared state ([`start`]) and stops in its drop ([`Lane::stop`]).
pub(super) struct Lane {
    tx: Option<mpsc::Sender<Job>>,
    threads: Vec<JoinHandle<()>>,
}

impl Lane {
    /// `job` to the lane ([`dispatch::send`]); refused by name as `what`
    /// when its threads have stopped.
    pub(super) fn send(
        &self,
        job: Job,
        what: &'static str,
    ) -> Result<dispatch::Dispatched, GpuError> {
        dispatch::send(self.tx.as_ref(), job, what)
    }

    /// Stop the lane threads of `shared` and join them, all within one
    /// `deadline`: whether every one ended. A thread still running past it
    /// is left to finish on its own.
    pub(super) fn stop(&mut self, shared: &Shared, deadline: Duration) -> bool {
        shared.stop.store(true, Ordering::Release);
        self.tx = None;
        let t0 = Instant::now();
        while !self.threads.iter().all(JoinHandle::is_finished) && t0.elapsed() <= deadline {
            std::thread::sleep(Duration::from_micros(200));
        }
        let mut joined = true;
        for t in self.threads.drain(..) {
            if t.is_finished() {
                // A panic outside a job has nowhere to go on the drop path.
                let _ = t.join();
            } else {
                joined = false;
            }
        }
        joined
    }
}

/// The ring, the staging words and the lane's [`LANE_THREADS`] threads over
/// `source`, every host wait bounded by `deadline`, whose pick stamps cover
/// the layer indices below `layers_end`.
pub(super) fn start(
    ctx: &Arc<CudaContext>,
    source: Arc<dyn SwapSource>,
    slot_bytes: usize,
    deadline: Duration,
    layers_end: usize,
) -> Result<(Arc<Shared>, Lane), GpuError> {
    let shared = Arc::new(Shared {
        source,
        ring: Ring::new(ctx, slot_bytes)?,
        words: MappedHost::new(
            ctx,
            2 * RING_SLOTS * WORD_STRIDE,
            "cuMemHostAlloc (swap words)",
        )?,
        window: Arc::new(AtomicU32::new(1)),
        due: AtomicU64::new(0),
        deadline,
        flush: AtomicBool::new(false),
        served: AtomicU64::new(0),
        pick_left: (0..layers_end).map(|_| AtomicU64::new(0)).collect(),
        pick_stamp_ns: (0..layers_end).map(|_| AtomicU64::new(0)).collect(),
        epoch: Instant::now(),
        stamps: JobStamps::new(),
        stage_ns: AtomicU64::new(0),
        prepare_ns: AtomicU64::new(0),
        stop: AtomicBool::new(false),
        failed: Mutex::new(None),
        jobs_issued: AtomicU64::new(0),
        next_landing: AtomicU64::new(NO_FLIP_LANDING),
        taken: std::array::from_fn(|_| AtomicU64::new(0)),
        in_window: AtomicU32::new(0),
        in_open: AtomicU32::new(0),
        peak_window: AtomicU32::new(0),
        peak_open: AtomicU32::new(0),
    });
    let (tx, rx) = mpsc::channel::<Job>();
    let jobs = Arc::new(Mutex::new(rx));
    let mut lane = Lane {
        tx: Some(tx),
        threads: Vec::with_capacity(LANE_THREADS),
    };
    for t in 0..LANE_THREADS {
        match spawn(ctx, &shared, &jobs, t) {
            Ok(thread) => lane.threads.push(thread),
            // The threads started are ending: joined here, so their
            // references to the shared state go before this one.
            Err(e) => {
                shared.stop.store(true, Ordering::Release);
                lane.tx = None;
                for thread in lane.threads.drain(..) {
                    let _ = thread.join();
                }
                return Err(e);
            }
        }
    }
    Ok((shared, lane))
}

/// Lane thread `t` over `shared`, taking from `jobs`, once it has bound
/// `ctx`.
fn spawn(
    ctx: &Arc<CudaContext>,
    shared: &Arc<Shared>,
    jobs: &Arc<Mutex<mpsc::Receiver<Job>>>,
    t: usize,
) -> Result<JoinHandle<()>, GpuError> {
    let (bound_tx, bound_rx) = mpsc::channel::<Result<(), GpuError>>();
    let (for_thread, jobs) = (Arc::clone(shared), Arc::clone(jobs));
    let ctx = Arc::clone(ctx);
    // A helper, not a plain spawn: the step thread that builds the machine
    // may be pinned to one cpu, and the staging copies would take turns
    // with the step on it. The thread binds the context first: the
    // source's calls on it may reach the driver.
    let (thread, _) = threads::helper::spawn_helper("swap-staging", lane_placement(t), move || {
        let bound = ctx.bind_to_thread().map_err(GpuError::from);
        let ok = bound.is_ok();
        if bound_tx.send(bound).is_ok() && ok {
            for_thread.run(t, &jobs);
        }
    })
    .map_err(|e| GpuError::plan("SwapMachine::new: a lane thread", e))?;
    match bound_rx.recv() {
        Ok(Ok(())) => Ok(thread),
        // The thread is ending: joined here, so its reference to the
        // shared state goes before the caller's.
        Ok(Err(e)) => {
            let _ = thread.join();
            Err(e)
        }
        Err(_) => {
            let _ = thread.join();
            Err(GpuError::protocol(
                "SwapMachine::new",
                "a lane thread ended before it bound the context",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{JobStamps, STAMP_SLOTS};

    /// A job's stamps read back as they were written — the issue stamp and
    /// the copies ahead from the machine, the preparation and staging stamps
    /// from the lane — and a cell reused by a later job answers nothing for
    /// the earlier one, its serve stamps cleared for the later one; a job
    /// whose staging word is unpublished reads `staged` 0.
    #[test]
    fn a_jobs_stamps_read_back_until_its_cell_is_reused() {
        let stamps = JobStamps::new();
        stamps.issue(5, 1_000, 2);
        assert_eq!(
            stamps.of(5).map(|s| (s.queued, s.ahead, s.staged)),
            Some((1_000, 2, 0)),
            "issued, never served: staged 0"
        );
        stamps.note_prepare_at(5, 1_400);
        stamps.note_prepare(5, 700);
        stamps.note_staged(5, 2_900, 800);
        assert_eq!(
            stamps
                .of(5)
                .map(|s| (s.prepare_at, s.prepare, s.read, s.staged)),
            Some((1_400, 700, 800, 2_900)),
            "the serve stamps over the issue one"
        );
        assert_eq!(stamps.of(4), None, "a cell no job wrote answers nothing");
        stamps.issue(5 + STAMP_SLOTS as u64, 9_000, 0);
        assert_eq!(
            stamps.of(5),
            None,
            "the reused cell holds the later job, not the earlier one's stamps"
        );
        assert_eq!(
            stamps
                .of(5 + STAMP_SLOTS as u64)
                .map(|s| (s.queued, s.staged)),
            Some((9_000, 0)),
            "the reuse clears the serve stamps"
        );
    }
}
