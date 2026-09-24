//! Who issues a replay's `cuGraphLaunch`, and the one place a replay is
//! launched and then served ([`replay`]).
//!
//! By default the decode thread makes the call and serves the host's share of
//! the replay once it returns. `cuGraphLaunch` of a whole-model graph returns
//! only after the driver has submitted it, and the chain's first hybrid go can
//! land before that: the first service then starts late by the rest of the
//! call. `BLOOMERY_LAUNCH_THREAD=1` gives the call to a [`Launcher`] thread,
//! spawned once at open: the decode thread posts the graph and goes straight
//! into the first service's wait. The graph is the same and the stream order
//! is the same (the post happens after every enqueue of the step's refresh,
//! and nothing else is enqueued until the launch is acknowledged), so the lever
//! moves no bit.
//!
//! A failed launch on that thread raises a flag every pending service's wait
//! watches ([`ReplayWatch::failed`]): the service returns, poisons the tier and
//! releases the stream as any failed service does, and [`replay`] returns the
//! launch's own error — the one the decode thread gets from a failed call of
//! its own.
//!
//! The thread starts with the affinity of the thread that opened the model.
//! A caller pinned to exactly one cpu (as `generate_ds41` pins its decode
//! thread before the open) would hand it that cpu, where the launch could not
//! run while the decode thread spins for the first go; the thread therefore
//! moves itself to that core's SMT sibling, as the engram helper does, and
//! the spawn fails by name when the core has none or the move is refused. A
//! caller with a wider mask leaves it floating. [`Launcher::cpu`] reports
//! where it went.

use crate::GpuError;
use crate::graph::{Graph, cu, launch_exec};
use crate::hybrid::{ReplayWatch, serving_replay};
use cuda_core::{CudaStream, sys};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// `BLOOMERY_LAUNCH_THREAD`: unset or `0` launches on the decode thread, `1`
/// on a [`Launcher`]. Read once per process.
pub(crate) fn lever() -> Result<bool, GpuError> {
    static LEVER: OnceLock<Result<bool, String>> = OnceLock::new();
    LEVER
        .get_or_init(|| match std::env::var("BLOOMERY_LAUNCH_THREAD") {
            Err(std::env::VarError::NotPresent) => Ok(false),
            Ok(v) if v.trim() == "0" => Ok(false),
            Ok(v) if v.trim() == "1" => Ok(true),
            Ok(v) => Err(format!("BLOOMERY_LAUNCH_THREAD={v:?}: want 0 or 1")),
            Err(e) => Err(format!("BLOOMERY_LAUNCH_THREAD: {e}")),
        })
        .clone()
        .map_err(|detail| GpuError::shape("launcher::lever", detail))
}

/// The lever's launcher for `stream`, or `None` with the lever off.
/// Load-time only.
pub(crate) fn at_open(stream: &Arc<CudaStream>) -> Result<Option<Launcher>, GpuError> {
    if lever()? {
        Launcher::spawn(Arc::clone(stream)).map(Some)
    } else {
        Ok(None)
    }
}

/// What the replays of `GpuModel::step` and `GpuModel::step_pair` have cost
/// their launch since load. Counted by the decode thread.
#[derive(Clone, Copy, Debug, Default)]
pub struct LaunchStats {
    /// Replays launched.
    pub launches: u64,
    /// Wall time of the `cuGraphLaunch` call itself, summed (ns) — on the
    /// decode thread, or on the launcher's with the lever.
    pub launch_ns: u64,
    /// With the lever, the time from the post to the launcher entering the
    /// call, summed (ns); zero without it.
    pub wake_ns: u64,
}

/// The launcher's state word. The decode thread moves it to `ARMED` and
/// `POSTED` and back to `IDLE`; the launcher from `POSTED` to `DONE` or
/// `FAILED`.
const IDLE: u32 = 0;
const ARMED: u32 = 1;
const POSTED: u32 = 2;
const DONE: u32 = 3;
const FAILED: u32 = 4;
const EXIT: u32 = 5;

/// How long an armed launcher polls for its post before it parks. It polls
/// by yielding: the SMT sibling it sits on is the engram helper's too, and
/// the helper works in the same refresh the launcher is armed across.
const ARM_SPIN: Duration = Duration::from_millis(2);

/// How long the decode thread waits for a posted launch to be answered.
const ANSWER_DEADLINE: Duration = Duration::from_secs(10);

/// What the decode thread and the launcher share.
struct Slot {
    state: AtomicU32,
    exec: AtomicPtr<sys::CUgraphExec_st>,
    /// The post's time, as ns since `base`.
    posted_ns: AtomicU64,
    base: Instant,
    launch_ns: AtomicU64,
    wake_ns: AtomicU64,
    rc: AtomicU32,
    /// Raised with `FAILED`, for the services' waits; lowered at each post.
    failed: Arc<AtomicBool>,
}

/// A thread that issues the graph launches of one stream, spawned once at open
/// and joined on drop.
pub(crate) struct Launcher {
    slot: Arc<Slot>,
    stream: Arc<CudaStream>,
    /// The cpu the thread pinned itself to; `None` when it floats.
    cpu: Option<usize>,
    handle: Option<JoinHandle<()>>,
}

impl Launcher {
    /// Spawn the launcher of `stream`; it makes the stream's context current
    /// on itself before it answers. Load-time only.
    pub(crate) fn spawn(stream: Arc<CudaStream>) -> Result<Launcher, GpuError> {
        let slot = Arc::new(Slot {
            state: AtomicU32::new(IDLE),
            exec: AtomicPtr::new(std::ptr::null_mut()),
            posted_ns: AtomicU64::new(0),
            base: Instant::now(),
            launch_ns: AtomicU64::new(0),
            wake_ns: AtomicU64::new(0),
            rc: AtomicU32::new(sys::cudaError_enum_CUDA_SUCCESS),
            failed: Arc::new(AtomicBool::new(false)),
        });
        let (ready_tx, ready_rx) = mpsc::channel();
        let shared = Arc::clone(&slot);
        let own = Arc::clone(&stream);
        let handle = std::thread::Builder::new()
            .name("graph-launch".into())
            .spawn(move || {
                let placed = place_off_the_caller();
                let bound = own.context().bind_to_thread();
                let ok = bound.is_ok() && placed.is_ok();
                // The owner waits on this; a closed receiver means it gave up.
                let _ = ready_tx.send((bound, placed));
                if ok {
                    serve_launches(&shared, &own);
                }
            })
            .map_err(|_| GpuError::state("Launcher::spawn", "the launch thread did not start"))?;
        let (bound, placed) = ready_rx
            .recv()
            .map_err(|_| GpuError::state("Launcher::spawn", "the launch thread exited at start"))?;
        let mut launcher = Launcher {
            slot,
            stream,
            cpu: None,
            handle: Some(handle),
        };
        // On an error the drop joins the thread, which has already returned.
        launcher.cpu = placed?;
        bound.map_err(|source| GpuError::Driver {
            op: Some("cuCtxSetCurrent (launch thread)"),
            source,
        })?;
        Ok(launcher)
    }

    /// The cpu the launch thread pinned itself to, `None` when it floats.
    #[must_use]
    pub(crate) fn cpu(&self) -> Option<usize> {
        self.cpu
    }

    /// Wake the launcher so that it spins for the post to come: a step calls
    /// this before its refresh, whose time hides the wake.
    pub(crate) fn arm(&self) {
        if self.slot.state.load(Ordering::Acquire) == IDLE {
            self.slot.state.store(ARMED, Ordering::Release);
            self.unpark();
        }
    }

    fn unpark(&self) {
        if let Some(h) = &self.handle {
            h.thread().unpark();
        }
    }

    /// Hand `exec` to the launcher.
    ///
    /// # Safety
    ///
    /// `exec` is a live instantiated graph of this launcher's stream's
    /// context (or null, which the driver refuses), kept alive until
    /// [`Launcher::wait`] has returned.
    unsafe fn post(&self, exec: sys::CUgraphExec) {
        let s = &self.slot;
        s.failed.store(false, Ordering::Relaxed);
        s.exec.store(exec, Ordering::Relaxed);
        let now = u64::try_from(s.base.elapsed().as_nanos()).unwrap_or(u64::MAX);
        s.posted_ns.store(now, Ordering::Relaxed);
        s.state.store(POSTED, Ordering::Release);
        self.unpark();
    }

    /// Wait for the posted launch's answer: its call's ns and the wake's, or
    /// the call's error as [`launch_exec`] maps it.
    fn wait(&self) -> Result<(u64, u64), GpuError> {
        let s = &self.slot;
        let deadline = Instant::now() + ANSWER_DEADLINE;
        let mut spins = 0u32;
        let state = loop {
            let st = s.state.load(Ordering::Acquire);
            if st == DONE || st == FAILED {
                break st;
            }
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(1 << 12)
                && (Instant::now() > deadline
                    || self.handle.as_ref().is_none_or(|h| h.is_finished()))
            {
                return Err(GpuError::state(
                    "Launcher::wait",
                    "the launch thread did not answer a posted launch",
                ));
            }
            std::hint::spin_loop();
        };
        let launch_ns = s.launch_ns.load(Ordering::Relaxed);
        let wake_ns = s.wake_ns.load(Ordering::Relaxed);
        let rc = s.rc.load(Ordering::Relaxed);
        s.state.store(IDLE, Ordering::Release);
        if state == FAILED {
            cu(rc, "cuGraphLaunch")?;
        }
        Ok((launch_ns, wake_ns))
    }
}

impl Drop for Launcher {
    fn drop(&mut self) {
        self.slot.state.store(EXIT, Ordering::Release);
        if let Some(h) = self.handle.take() {
            h.thread().unpark();
            let _ = h.join();
        }
    }
}

/// Move the calling thread off the one cpu it inherited: to that core's SMT
/// sibling, returning it. A thread whose mask holds more than one cpu stays
/// as it is (`None`).
fn place_off_the_caller() -> Result<Option<usize>, GpuError> {
    const WHAT: &str = "Launcher::spawn";
    let set_bytes = std::mem::size_of::<libc::cpu_set_t>();
    // SAFETY: `set` is a zeroed cpu_set_t that `sched_getaffinity` fills with
    // the matching size.
    let set = unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, set_bytes, &mut set) != 0 {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "sched_getaffinity refused the launch thread: {}",
                    std::io::Error::last_os_error()
                ),
            ));
        }
        set
    };
    // SAFETY: `c` stays below the set's own bit count; `CPU_ISSET` only reads.
    let mut on = (0..8 * set_bytes).filter(|&c| unsafe { libc::CPU_ISSET(c, &set) });
    let (Some(cpu), None) = (on.next(), on.next()) else {
        return Ok(None);
    };
    let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
    let list = std::fs::read_to_string(&path)
        .map_err(|e| GpuError::shape(WHAT, format!("{path}: {e}")))?;
    let Some(sibling) = cpu_list(&list).find(|&c| c != cpu && c < 8 * set_bytes) else {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "the opening thread is pinned to cpu {cpu}, whose core has no SMT sibling \
                 ({path}: {:?}): the launch thread would share its cpu",
                list.trim()
            ),
        ));
    };
    // SAFETY: `one` is a zeroed cpu_set_t only written through `CPU_SET` with
    // an index inside it, and `sched_setaffinity` reads it with the matching
    // size.
    let pinned = unsafe {
        let mut one: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(sibling, &mut one);
        libc::sched_setaffinity(0, set_bytes, &one) == 0
    };
    if !pinned {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "sched_setaffinity refused cpu {sibling}, the sibling of the opening thread's \
                 cpu {cpu}: {}",
                std::io::Error::last_os_error()
            ),
        ));
    }
    Ok(Some(sibling))
}

/// The cpus of a sysfs cpu list (`0,32`, `0-1`, `0-3,8`); a malformed entry
/// ends the list.
fn cpu_list(text: &str) -> impl Iterator<Item = usize> + '_ {
    text.trim()
        .split(',')
        .map_while(|part| match part.split_once('-') {
            Some((a, b)) => Some(a.parse::<usize>().ok()?..=b.parse::<usize>().ok()?),
            None => part.parse::<usize>().ok().map(|c| c..=c),
        })
        .flatten()
}

/// The launcher's loop: park until armed or posted, yield while armed (at
/// most [`ARM_SPIN`]), launch what is posted, stop at `EXIT`.
fn serve_launches(slot: &Slot, stream: &CudaStream) {
    let hs = stream.cu_stream();
    let mut armed_until: Option<Instant> = None;
    loop {
        match slot.state.load(Ordering::Acquire) {
            POSTED => {
                armed_until = None;
                let t0 = Instant::now();
                let since =
                    u64::try_from(t0.duration_since(slot.base).as_nanos()).unwrap_or(u64::MAX);
                let exec = slot.exec.load(Ordering::Relaxed);
                // SAFETY: the decode thread posted `exec` under `post`'s
                // contract (alive until it has read this answer) for `hs`,
                // the stream this thread was spawned for, whose context
                // `spawn` made current here.
                let rc = unsafe { sys::cuGraphLaunch(exec, hs) };
                let ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
                slot.launch_ns.store(ns, Ordering::Relaxed);
                let posted = slot.posted_ns.load(Ordering::Relaxed);
                slot.wake_ns
                    .store(since.saturating_sub(posted), Ordering::Relaxed);
                slot.rc.store(rc, Ordering::Relaxed);
                let ok = rc == sys::cudaError_enum_CUDA_SUCCESS;
                if !ok {
                    slot.failed.store(true, Ordering::Release);
                }
                slot.state
                    .store(if ok { DONE } else { FAILED }, Ordering::Release);
            }
            EXIT => return,
            ARMED => {
                let until = *armed_until.get_or_insert_with(|| Instant::now() + ARM_SPIN);
                if Instant::now() > until {
                    std::thread::park();
                } else {
                    std::thread::yield_now();
                }
            }
            _ => {
                armed_until = None;
                std::thread::park();
            }
        }
    }
}

/// Launch `graph` on `stream` and serve the host's share of that replay with
/// `serve` — on the decode thread, or with `launcher` posting the launch to
/// its thread and serving while the call runs. Either way the call's own time
/// is added to `stats`, `serve` runs inside [`serving_replay`] with the
/// replay's watch, and a failed launch returns the launch's error.
pub(crate) fn replay(
    launcher: Option<&Launcher>,
    stats: &mut LaunchStats,
    graph: &Graph,
    stream: &CudaStream,
    serve: impl FnOnce() -> Result<(), GpuError>,
) -> Result<(), GpuError> {
    // SAFETY: `graph` is borrowed for the whole call, and `replay_exec`
    // returns only after the launch's answer.
    unsafe { replay_exec(launcher, stats, graph.exec(), stream, serve) }
}

/// [`replay`] of a raw `exec`.
///
/// # Safety
///
/// `exec` is a live instantiated graph of `stream`'s context (or null, which
/// the driver refuses) and stays alive until this returns.
unsafe fn replay_exec(
    launcher: Option<&Launcher>,
    stats: &mut LaunchStats,
    exec: sys::CUgraphExec,
    stream: &CudaStream,
    serve: impl FnOnce() -> Result<(), GpuError>,
) -> Result<(), GpuError> {
    let issued = Instant::now();
    stats.launches += 1;
    let Some(l) = launcher else {
        // SAFETY: the caller's contract; `stream` is the engine stream of the
        // context current on this thread.
        unsafe { launch_exec(exec, stream.cu_stream()) }?;
        stats.launch_ns += u64::try_from(issued.elapsed().as_nanos()).unwrap_or(u64::MAX);
        return serving_replay(
            ReplayWatch {
                issued,
                failed: None,
            },
            serve,
        );
    };
    if l.stream.cu_stream() != stream.cu_stream() {
        return Err(GpuError::state(
            "launcher::replay",
            "the launcher was spawned for another stream",
        ));
    }
    // SAFETY: the caller's contract; the answer is read below before return,
    // on every path, the unwinding one included.
    unsafe { l.post(exec) };
    let watch = ReplayWatch {
        issued,
        failed: Some(Arc::clone(&l.slot.failed)),
    };
    let served = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        serving_replay(watch, serve)
    }));
    let launched = l.wait();
    let served = served.unwrap_or_else(|p| std::panic::resume_unwind(p));
    let (launch_ns, wake_ns) = launched?;
    stats.launch_ns += launch_ns;
    stats.wake_ns += wake_ns;
    served
}

#[cfg(test)]
mod tests {
    use super::{LaunchStats, Launcher, cpu_list, replay, replay_exec};
    use crate::GpuError;
    use crate::graph::Graph;
    use crate::hybrid::{Boundary, BoundaryShape, HostExperts, Hybrid, SlotMap};
    use cuda_core::CudaContext;
    use model::Tensor2;
    use std::time::{Duration, Instant};

    /// Host experts that sum nothing: the replay's routing sends no slot to
    /// the host.
    struct Zero;

    impl HostExperts for Zero {
        fn experts_into(
            &mut self,
            _layer: usize,
            _x: &Tensor2,
            _experts: &[(u32, f32)],
            out: &mut [f32],
        ) -> Result<(), GpuError> {
            out.fill(0.0);
            Ok(())
        }
    }

    /// A one-layer hybrid chain (go, then wait) replayed through the launch
    /// thread is served and drains. A launch that fails on that thread while a
    /// service waits for its go comes back to the caller as the error a failed
    /// launch on the caller's own thread gives, well inside the service's
    /// deadline, and the tier is poisoned as after any failed service.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_failed_launch_on_the_launch_thread_reaches_the_decode_thread() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.new_stream().expect("a stream");
        let shape = BoundaryShape {
            hidden: 64,
            n_used: 6,
        };
        let slots = SlotMap::prefix(0..1, 8, 4).expect("a slot map");
        let boundary = Boundary::new(&ctx, &stream, shape, slots, true).expect("a boundary");
        let mut hybrid = Hybrid::new(boundary, Zero, 1).expect("the tier");
        stream.synchronize().expect("the allocations land");
        let graph = Graph::capture(&stream, |s| {
            hybrid.begin_chain(s)?;
            hybrid.boundary().enqueue_go(s, 0)?;
            hybrid.layer_enqueued(0)?;
            hybrid.boundary().enqueue_back(s)
        })
        .expect("the capture");
        let launcher = Launcher::spawn(stream.clone()).expect("the launch thread");
        let mut stats = LaunchStats::default();

        launcher.arm();
        replay(Some(&launcher), &mut stats, &graph, &stream, || {
            hybrid.serve_captured()
        })
        .expect("a replay through the launch thread");
        stream.synchronize().expect("the replay drains");
        let st = hybrid.stats();
        assert_eq!((st.served, stats.launches), (1, 1), "{st:?} {stats:?}");
        assert!(
            stats.launch_ns > 0 && st.first_serve_lag_ns > 0,
            "{st:?} {stats:?}"
        );

        // SAFETY: a null exec, which the driver refuses; no launch runs.
        let direct =
            unsafe { replay_exec(None, &mut stats, std::ptr::null_mut(), &stream, || Ok(())) }
                .expect_err("a null exec does not launch");
        assert!(
            matches!(
                direct,
                GpuError::Driver {
                    op: Some("cuGraphLaunch"),
                    ..
                }
            ),
            "{direct:?}"
        );

        launcher.arm();
        let t = Instant::now();
        // SAFETY: as above.
        let err = unsafe {
            replay_exec(
                Some(&launcher),
                &mut stats,
                std::ptr::null_mut(),
                &stream,
                || hybrid.serve_captured(),
            )
        }
        .expect_err("the failed launch reaches the caller");
        let waited = t.elapsed();
        assert_eq!(format!("{err:?}"), format!("{direct:?}"));
        assert!(
            waited < Duration::from_secs(2),
            "the service waited {waited:?} for a go whose launch had failed"
        );
        assert!(
            hybrid.serve_captured().is_err(),
            "the tier serves again after a failed replay"
        );
    }

    /// An opener pinned to one cpu puts the launch thread on that core's SMT
    /// sibling, never on the opener's own cpu, where it could not run while
    /// the opener spins for the first go; a floating opener leaves it floating.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_pinned_opener_puts_the_launch_thread_on_its_sibling() {
        let ctx = CudaContext::new(0).expect("CUDA device 0");
        let stream = ctx.new_stream().expect("a stream");
        let floating = Launcher::spawn(stream.clone()).expect("a floating launch thread");
        assert_eq!(floating.cpu(), None, "the test thread floats");
        drop(floating);

        let opener = std::thread::spawn(move || {
            let cpu = 0;
            // SAFETY: `one` is a zeroed cpu_set_t only written through
            // `CPU_SET` with an index inside it, and `sched_setaffinity` reads
            // it with the matching size.
            let pinned = unsafe {
                let mut one: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_SET(cpu, &mut one);
                libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &one) == 0
            };
            assert!(pinned, "the opener pins itself to cpu {cpu}");
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
            let list = std::fs::read_to_string(&path).expect("the core's sibling list");
            let want = cpu_list(&list).find(|&c| c != cpu).expect("an SMT box");
            let l = Launcher::spawn(stream).expect("a launch thread off the pinned opener");
            (l.cpu(), want)
        });
        let (got, want) = opener.join().expect("the opener thread");
        assert_eq!(
            got,
            Some(want),
            "the launch thread sits on the opener's sibling"
        );
    }
}
