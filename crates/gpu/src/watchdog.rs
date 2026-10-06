//! The engine watchdog: one thread per loaded model that names an engine
//! thread wedged inside a driver call. [`crate::host::await_done`] bounds
//! only a stall the engine thread sees from its own polling loop — its own
//! stream's work not finishing. A driver that runs every stream on one
//! hardware queue can instead wedge the thread inside a driver call that has
//! no bound of ours (`cuStreamSynchronize` behind a copy that never comes),
//! and a wedge the engine thread cannot see from inside is debuggable only
//! from another thread.
//!
//! The engine thread writes its progress here — relaxed stores only, no
//! lock, no allocation — and the watchdog reads it:
//!
//! - "busy since": set at the entry of each outermost [`GpuModel`] call that
//!   enqueues card work and waits on it ([`busy`]), cleared by the guard's
//!   drop on every path (a return, an error, an unwind);
//! - "progress": bumped whenever the thread finishes a wait on the card
//!   ([`wait_done`]), lands a residency boundary ([`boundary_landed`]) or
//!   ends a guarded call. A long prompt call keeps making progress; only a
//!   thread that sits in one place for the bound is wedged.
//!
//! While the thread is busy and its progress has not moved for
//! [`crate::host::ENGINE_BOUND`], the watchdog prints one line to stderr
//! naming the call, and re-arms only once progress moves again. It names
//! the wedge; it does not end it. The staging-window rule
//! (`crate::host::step`'s `Closed`) is what keeps the residency staging
//! copies' cycle from closing on one queue in the first place.
//!
//! The thread is the model's: started at [`GpuModel::new`], left in place by
//! a reset, stopped and joined before anything of the model is freed
//! (`Drop for GpuModel`).

use crate::host::swap;
use crate::host::{ENGINE_BOUND, nanos};
use std::cell::RefCell;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How often the watchdog polls the progress words.
const POLL: Duration = Duration::from_secs(1);

/// The boundary word before the engine thread landed any boundary.
const NO_BOUNDARY: u64 = u64::MAX;

/// The guarded calls' names, one row each: the word the watch stores is the
/// row's index ([`Call`]), so the wedge line prints a name no allocation
/// carries. A new guarded entry point adds its row here and its [`Call`]
/// beside it.
const CALLS: [&str; 11] = [
    "GpuModel::step",
    "GpuModel::run_rows",
    "GpuModel::pass_boundary",
    "GpuModel::reset",
    "GpuModel::residency_reset",
    "GpuModel::step_rows",
    "GpuModel::rollback",
    "GpuModel::seed_depth",
    "GpuModel::step_slots",
    "GpuModel::verify_slots",
    "GpuModel::commit_slots",
];

/// One guarded call's name row of [`CALLS`]: what a guarded entry point
/// passes [`busy`] and the wedge line prints.
#[derive(Clone, Copy)]
pub(crate) struct Call(usize);

pub(crate) const STEP: Call = Call(0);
pub(crate) const RUN_ROWS: Call = Call(1);
pub(crate) const PASS_BOUNDARY: Call = Call(2);
pub(crate) const RESET: Call = Call(3);
pub(crate) const RESIDENCY_RESET: Call = Call(4);
pub(crate) const STEP_ROWS: Call = Call(5);
pub(crate) const ROLLBACK: Call = Call(6);
pub(crate) const SEED_DEPTH: Call = Call(7);
pub(crate) const STEP_SLOTS: Call = Call(8);
pub(crate) const VERIFY_SLOTS: Call = Call(9);
pub(crate) const COMMIT_SLOTS: Call = Call(10);

impl Call {
    fn name(self) -> &'static str {
        CALLS.get(self.0).copied().unwrap_or("an engine call")
    }
}

/// The engine thread's progress words, written by that thread with relaxed
/// stores only and read by its watchdog.
pub(crate) struct Watch {
    /// The watch's own time base: every ns word below is measured from it.
    base: Instant,
    /// ns since `base` the thread entered a guarded call; 0 = idle.
    busy_since: AtomicU64,
    /// Bumped whenever the thread finishes a wait on the card, lands a
    /// residency boundary, or ends a guarded call.
    progress: AtomicU64,
    /// The guarded call's entry readback count (`GpuModel::reads`).
    reads: AtomicU64,
    /// The last residency boundary the thread landed; [`NO_BOUNDARY`].
    boundary: AtomicU64,
    /// The guarded call's name row of [`CALLS`].
    call: AtomicUsize,
    /// The residency machine's shared state, handed over at the first
    /// boundary: what the stall note renders from, with no machine in reach.
    shared: OnceLock<Arc<swap::Shared>>,
    /// The watchdog's firings. Nothing on the engine path reads it; a test
    /// counts the lines through it.
    fired: AtomicU64,
}

impl Watch {
    pub(crate) fn new() -> Watch {
        Watch {
            base: Instant::now(),
            busy_since: AtomicU64::new(0),
            progress: AtomicU64::new(0),
            reads: AtomicU64::new(0),
            boundary: AtomicU64::new(NO_BOUNDARY),
            call: AtomicUsize::new(0),
            shared: OnceLock::new(),
            fired: AtomicU64::new(0),
        }
    }

    /// The busy word of a call entering now: name and read count first, the
    /// busy word last, so a reader gated on busy most likely sees them.
    fn enter(&self, call: Call, reads: u64) {
        self.call.store(call.0, Ordering::Relaxed);
        self.reads.store(reads, Ordering::Relaxed);
        self.busy_since
            .store(nanos(self.base.elapsed()), Ordering::Relaxed);
    }

    /// The one line a wedge prints: how long without progress, in what call
    /// busy since when, at what readback count and boundary, and the stall
    /// note — the machine's when one runs ([`swap::Shared::stall_note`]),
    /// else plain words.
    fn wedge_line(&self, wedged: Duration, busy_since: u64) -> String {
        let call = Call(self.call.load(Ordering::Relaxed)).name();
        let busy = if busy_since == 0 {
            Duration::ZERO
        } else {
            Duration::from_nanos(nanos(self.base.elapsed()).saturating_sub(busy_since))
        };
        let boundary = self.boundary.load(Ordering::Relaxed);
        let boundary = if boundary == NO_BOUNDARY {
            "none".to_string()
        } else {
            boundary.to_string()
        };
        let note = match self.shared.get() {
            Some(s) => s.stall_note(),
            None => "no residency machine".to_string(),
        };
        format!(
            "engine wedged: {call} made no progress for {wedged:?} (busy {busy:?}, read {}, \
             boundary {boundary}): {note}",
            self.reads.load(Ordering::Relaxed),
        )
    }
}

thread_local! {
    /// The watch of the guarded call this thread is inside, set by [`busy`]
    /// and restored by its guard's drop; `None` outside one. The waits and
    /// boundaries inside a guarded call reach their watch through it: one
    /// thread-local read, no lock, no allocation — the `Arc` a guard holds
    /// is a reference count, cloned once per guarded call.
    static GUARDED: RefCell<Option<Arc<Watch>>> = const { RefCell::new(None) };
}

/// What the guarded call this thread is inside reads, if any.
fn guarded<R>(f: impl FnOnce(&Watch) -> R) -> Option<R> {
    GUARDED.with(|g| g.borrow().as_deref().map(f))
}

/// Mark the engine thread busy in `call` until the returned guard drops,
/// which every path — a return, an error, an unwind — runs: the drop clears
/// the busy word and bumps progress. `None` when this model's guard is
/// already held: a nested call keeps the outer's mark, so the outer's drop
/// is the one that clears it.
pub(crate) fn busy(watch: &Arc<Watch>, call: Call, reads: u64) -> Option<Busy> {
    let prev = GUARDED.with(|g| {
        let mut g = g.borrow_mut();
        match g.as_ref() {
            Some(w) if Arc::ptr_eq(w, watch) => None,
            _ => Some(g.replace(Arc::clone(watch))),
        }
    })?;
    if let Some(outer) = &prev {
        // Another model's call runs inside ours: the thread moved on under
        // it, which is progress that model's watchdog must see.
        outer.progress.fetch_add(1, Ordering::Relaxed);
    }
    watch.enter(call, reads);
    Some(Busy { prev })
}

/// The engine thread's busy mark: clears the busy word and bumps progress
/// when dropped, and puts the previously guarded watch (another model's,
/// when calls nest across models) back.
pub(crate) struct Busy {
    prev: Option<Arc<Watch>>,
}

impl Drop for Busy {
    fn drop(&mut self) {
        // The whole cell back to what `busy` found, and what it held — this
        // call's watch — returned for its clear and bump.
        let w = GUARDED.with(|g| g.replace(self.prev.take()));
        if let Some(w) = w {
            w.busy_since.store(0, Ordering::Relaxed);
            w.progress.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A wait on the card finished (`host::await_within` returning `Ok`): the
/// engine thread is moving. Nothing outside a guarded call.
pub(crate) fn wait_done() {
    guarded(|w| w.progress.fetch_add(1, Ordering::Relaxed));
}

/// A residency boundary landed at `b`: the thread is moving, and the line
/// names this boundary. Nothing outside a guarded call.
pub(crate) fn boundary_landed(b: u64) {
    guarded(|w| {
        w.progress.fetch_add(1, Ordering::Relaxed);
        w.boundary.store(b, Ordering::Relaxed);
    });
}

/// The residency machine behind the guarded call, for the stall note: the
/// first boundary hands it over and later ones change nothing. Nothing
/// outside a guarded call.
pub(crate) fn machine(shared: &Arc<swap::Shared>) {
    guarded(|w| {
        if w.shared.get().is_none() {
            let _ = w.shared.set(Arc::clone(shared));
        }
    });
}

/// One watchdog thread over a model's [`Watch`]: `stop` sends it away
/// without waiting out a poll and joins it.
pub(crate) struct Watchdog {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
    /// Set by the thread as its last act: proof a joined thread is gone.
    exited: Arc<AtomicBool>,
}

impl Watchdog {
    /// The watchdog of a loaded model: a wedge is named after
    /// [`ENGINE_BOUND`] without progress, polled every [`POLL`].
    pub(crate) fn start(watch: Arc<Watch>) -> Watchdog {
        over(watch, ENGINE_BOUND, POLL)
    }

    /// Stop the thread and join it. Idempotent; called by `Drop for
    /// GpuModel` before anything of the model is freed.
    pub(crate) fn stop(&mut self) {
        drop(self.stop.take());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
            debug_assert!(
                self.exited.load(Ordering::Acquire),
                "a joined watchdog set its exit flag"
            );
        }
    }

    /// The thread has ended: read only after [`Watchdog::stop`], whose join
    /// it proves. A test's hold.
    #[cfg(test)]
    pub(crate) fn exited(&self) -> bool {
        self.exited.load(Ordering::Acquire)
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.stop();
    }
}

/// [`Watchdog::start`] with the bound and poll a test picks; never a runtime
/// lever.
fn over(watch: Arc<Watch>, bound: Duration, poll: Duration) -> Watchdog {
    let (tx, rx) = mpsc::channel();
    let exited = Arc::new(AtomicBool::new(false));
    // A plain thread, no helper placement: it never touches the card or the
    // pools, only these words and stderr.
    let thread = std::thread::Builder::new()
        .name("engine-watchdog".to_string())
        .spawn({
            let w = Arc::clone(&watch);
            let exited = Arc::clone(&exited);
            move || {
                run(w, bound, poll, rx);
                exited.store(true, Ordering::Release);
            }
        })
        .unwrap_or_else(|e| panic!("engine watchdog: the thread: {e}"));
    Watchdog {
        stop: Some(tx),
        thread: Some(thread),
        exited,
    }
}

/// Poll until `stop` goes away. While the engine thread is busy and its
/// progress has not moved for `bound`, print the wedge line once; a firing
/// re-arms only when progress moves again (or the thread goes idle).
fn run(watch: Arc<Watch>, bound: Duration, poll: Duration, stop: mpsc::Receiver<()>) {
    let mut seen = watch.progress.load(Ordering::Relaxed);
    let mut moved = Instant::now();
    let mut armed = true;
    loop {
        match stop.recv_timeout(poll) {
            Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => return,
            Err(mpsc::RecvTimeoutError::Timeout) => {}
        }
        let busy = watch.busy_since.load(Ordering::Relaxed);
        let p = watch.progress.load(Ordering::Relaxed);
        if p != seen || busy == 0 {
            seen = p;
            moved = Instant::now();
            armed = true;
            continue;
        }
        let wedged = moved.elapsed();
        if armed && wedged >= bound {
            armed = false;
            watch.fired.fetch_add(1, Ordering::Relaxed);
            eprintln!("{}", watch.wedge_line(wedged, busy));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    const TEST: Call = STEP;

    #[test]
    fn the_wedge_line_names_the_call_and_position() {
        let w = Watch::new();
        w.enter(TEST, 7);
        w.boundary.store(3, Ordering::Relaxed);
        let line = w.wedge_line(Duration::from_secs(62), nanos(w.base.elapsed()));
        assert!(
            line.starts_with(&format!(
                "engine wedged: {} made no progress for ",
                TEST.name()
            )),
            "{line}"
        );
        assert!(line.contains("read 7"), "{line}");
        assert!(line.contains("boundary 3"), "{line}");
        assert!(line.contains("no residency machine"), "{line}");
    }

    #[test]
    fn the_wedge_line_names_no_boundary_before_one_lands() {
        let w = Watch::new();
        w.enter(TEST, 0);
        let line = w.wedge_line(Duration::from_secs(1), nanos(w.base.elapsed()));
        assert!(line.contains("boundary none"), "{line}");
    }

    #[test]
    fn the_busy_guard_marks_clears_and_bumps() {
        let w = Arc::new(Watch::new());
        {
            let _b = busy(&w, TEST, 3).expect("the outer guard");
            assert_ne!(w.busy_since.load(Ordering::Relaxed), 0);
            assert_eq!(w.reads.load(Ordering::Relaxed), 3);
            let p0 = w.progress.load(Ordering::Relaxed);
            wait_done();
            boundary_landed(5);
            assert_eq!(w.progress.load(Ordering::Relaxed), p0 + 2);
            assert_eq!(w.boundary.load(Ordering::Relaxed), 5);
            // A nested call keeps the outer's mark and words.
            assert!(busy(&w, TEST, 4).is_none());
            assert_eq!(w.reads.load(Ordering::Relaxed), 3);
        }
        assert_eq!(w.busy_since.load(Ordering::Relaxed), 0);
        let p1 = w.progress.load(Ordering::Relaxed);
        wait_done();
        assert_eq!(w.progress.load(Ordering::Relaxed), p1, "no watch guarded");
    }

    #[test]
    fn the_busy_guard_clears_on_an_unwind() {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let w = Arc::new(Watch::new());
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _b = busy(&w, TEST, 9).expect("the guard");
            assert_ne!(w.busy_since.load(Ordering::Relaxed), 0);
            panic!("the unwind under the guard");
        }));
        std::panic::set_hook(hook);
        assert!(r.is_err(), "the closure panicked");
        assert_eq!(w.busy_since.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_wedge_fires_once_and_rearms_only_on_progress() {
        let w = Arc::new(Watch::new());
        w.enter(TEST, 1);
        let mut d = over(
            Arc::clone(&w),
            Duration::from_millis(150),
            Duration::from_millis(20),
        );
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(w.fired.load(Ordering::Relaxed), 1, "one line for the wedge");
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(
            w.fired.load(Ordering::Relaxed),
            1,
            "no second line while still wedged"
        );
        w.progress.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(
            w.fired.load(Ordering::Relaxed),
            2,
            "re-armed after progress"
        );
        d.stop();
        assert!(d.exited(), "the thread is joined");
    }

    #[test]
    fn an_idle_engine_never_fires() {
        let w = Arc::new(Watch::new());
        let mut d = over(
            Arc::clone(&w),
            Duration::from_millis(120),
            Duration::from_millis(20),
        );
        std::thread::sleep(Duration::from_millis(500));
        assert_eq!(w.fired.load(Ordering::Relaxed), 0);
        d.stop();
        assert!(d.exited());
    }

    #[test]
    fn a_busy_call_that_moves_never_fires() {
        let w = Arc::new(Watch::new());
        w.enter(TEST, 1);
        let mut d = over(
            Arc::clone(&w),
            Duration::from_millis(120),
            Duration::from_millis(20),
        );
        for _ in 0..15 {
            std::thread::sleep(Duration::from_millis(60));
            w.progress.fetch_add(1, Ordering::Relaxed);
        }
        assert_eq!(
            w.fired.load(Ordering::Relaxed),
            0,
            "progress is not a wedge"
        );
        d.stop();
        assert!(d.exited());
    }
}
