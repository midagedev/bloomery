//! The one owner of every context-wide synchronize this process can run,
//! and of the fresh context handles whose first stream runs one.
//!
//! A `CudaContext` handle is a handle on its device's primary context, and
//! the first stream a fresh handle makes runs `cuCtxSynchronize` first
//! (cuda-core's `create_stream`, on its stream count's zero-to-one step).
//! A context-wide synchronize while any thread captures on that context is
//! refused and invalidates the capture, whatever mode the capture began
//! in: the sync waits for the capturing stream itself. So the two sides
//! meet here, per device — a capture holds the device's lock shared for
//! its whole begin..end; a synchronize, or a fresh handle with its first
//! stream, holds it exclusive — and they never overlap on one device.
//! Work that reaches no context-wide synchronize (a module load, an
//! allocation, one stream's or event's own synchronize) takes no lock, and
//! captures on other devices are other locks: per-device contexts cannot
//! break each other's captures.
//!
//! A synchronize of this module's, and the implicit one every free makes
//! (`cuMemFree`), wait for every stream of the device, a stream parked on a
//! wait that a host store opens included, and the driver's lock stays with
//! the waiter: shared for a synchronize, which blocks a thread's next driver
//! call once any writer queues, exclusive for a free, which blocks it at
//! once. A thread that must make a driver call before it opens such a wait
//! waits behind the waiter while the waiter waits for the wait: neither
//! returns, and no bound of ours ends it. The engine makes its synchronizes
//! on the thread that drives the steps, at load and teardown, and its helper
//! threads copy and store but neither free nor synchronize; a test that
//! parks a stream runs alone (`checkpoint::hold::require_serial_pass`).
//!
//! A dropped device buffer is freed on none of those threads: cuda-core
//! queues its `cuMemFree` on its one `cuda-resource-retirement` thread,
//! which runs it a moment later, whenever the engine stands. An engine call
//! that launches a pass whose streams wait on host stores — the stage card's
//! joins on the host leg, a tier's go on the stage card's — and then makes a
//! driver call before those stores (a tier graph's launch, a fed tier
//! layer) closes the cycle above with that thread when a free starts in
//! between. [`Frees`] keeps the thread from starting any cleanup while it is
//! held; the engine holds it from a call's first pass launch to the call's
//! end (`GpuModel`'s pass launches, `watchdog::hold_frees_for_call`).
//!
//! A capture holds one device's shared side for its whole body; a body
//! that also took a second device's exclusive side could meet another
//! thread doing the two in the opposite order. No capture body of the
//! engine opens a handle.

use crate::{GpuError, StreamRole, role_stream};
use cuda_core::{CudaContext, CudaStream, DriverError};
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, PoisonError, RwLock, mpsc};
use std::time::Duration;

/// One lock per device: exclusive while a context-wide synchronize or a
/// fresh handle's first stream runs, shared over a capture's begin..end.
/// Grown on demand, never shrunk; a poisoned lock is recovered, as the
/// process's context anchors are.
static DEVICES: Mutex<Vec<Arc<RwLock<()>>>> = Mutex::new(Vec::new());

thread_local! {
    /// The devices this thread is capturing on, one entry per open capture
    /// (a nested capture pushes its device again). The owner refuses a
    /// context-wide synchronize from the capturing thread itself with a
    /// named panic instead of the wait the lock would impose — the thread
    /// would be waiting for its own capture to end — and a nested capture
    /// does not take the shared lock again, which a waiting writer could
    /// turn into a deadlock.
    static CAPTURING: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// Whether this thread is capturing on `device`.
fn capturing(device: usize) -> bool {
    CAPTURING.with(|open| open.borrow().contains(&device))
}

/// Refuse a context-wide synchronize, or a fresh handle's first stream, on
/// a device this thread captures on: that is a caller bug the lock cannot
/// mediate, so it is named — the device and the call site — never waited
/// out.
fn refuse_capturing(device: usize, site: &'static str) {
    assert!(
        !capturing(device),
        "{site}: device {device} is capturing on this thread; a context-wide synchronize \
         would break the capture"
    );
}

/// `device`'s lock, grown into the table on first use.
fn device_lock(device: usize) -> Arc<RwLock<()>> {
    let mut held = DEVICES.lock().unwrap_or_else(PoisonError::into_inner);
    if held.len() <= device {
        held.resize_with(device + 1, || Arc::new(RwLock::new(())));
    }
    Arc::clone(&held[device])
}

/// A fresh handle on `device`'s primary context: the only place outside
/// cuda-core the constructor is called, so every fresh handle in the
/// process passes one door.
fn new_context(device: usize) -> Result<Arc<CudaContext>, GpuError> {
    #[allow(
        clippy::disallowed_methods,
        reason = "the one owner: every fresh handle in the workspace goes through this module"
    )]
    Ok(CudaContext::new(device)?)
}

/// Run `during` with `device`'s lock held exclusive, after refusing it on
/// a device this thread captures on.
fn exclusive<T>(
    device: usize,
    site: &'static str,
    during: impl FnOnce() -> Result<T, GpuError>,
) -> Result<T, GpuError> {
    refuse_capturing(device, site);
    let lock = device_lock(device);
    let _held = lock.write().unwrap_or_else(PoisonError::into_inner);
    during()
}

/// A fresh handle on `device`'s primary context with its first stream, at
/// `role`'s priority: the process retains its anchor on the device first,
/// then the handle and the stream are made under the device's exclusive
/// lock, because a fresh handle's first stream runs the context-wide
/// synchronize this module owns. Everything a caller makes after the first
/// stream — the fault-word allocation, the module loads, later streams —
/// runs outside the lock: none of it synchronizes the context.
pub(crate) fn fresh_handle(
    device: usize,
    role: StreamRole,
) -> Result<(Arc<CudaContext>, Arc<CudaStream>), GpuError> {
    crate::anchor(device)?;
    exclusive(device, "capsync::fresh_handle", || {
        let ctx = new_context(device)?;
        let stream = role_stream(&ctx, role)?;
        Ok((ctx, stream))
    })
}

/// A fresh handle with its first stream at no priority: the pair a test or
/// a probe binary makes. See [`fresh_handle`].
pub fn fresh_stream(device: usize) -> Result<(Arc<CudaContext>, Arc<CudaStream>), GpuError> {
    exclusive(device, "capsync::fresh_stream", || {
        let ctx = new_context(device)?;
        let stream = ctx.new_stream()?;
        Ok((ctx, stream))
    })
}

/// A fresh handle that will make no stream of its own: `CudaContext::new`
/// runs no context-wide synchronize (only a handle's first stream does), so
/// a handle alone cannot break a capture. The first stream on the returned
/// handle must come from this module — [`fresh_stream`] makes its own
/// handle — or its synchronize runs outside the owner.
pub fn fresh_context(device: usize) -> Result<Arc<CudaContext>, GpuError> {
    exclusive(device, "capsync::fresh_context", || new_context(device))
}

/// The context-wide synchronize of `ctx`'s context, under its device's
/// exclusive lock: it never runs while any thread captures on that
/// context. `site` names the caller in the refusal.
pub fn ctx_sync(ctx: &Arc<CudaContext>, site: &'static str) -> Result<(), DriverError> {
    refuse_capturing(ctx.ordinal(), site);
    let lock = device_lock(ctx.ordinal());
    let _held = lock.write().unwrap_or_else(PoisonError::into_inner);
    #[allow(
        clippy::disallowed_methods,
        reason = "the one owner of the context-wide synchronize"
    )]
    ctx.synchronize()
}

/// [`ctx_sync`] for a drop path, which must not panic: on a device this
/// thread is capturing on the synchronize is skipped and named on stderr,
/// the way a failed synchronize on a drop is named, and the caller's own
/// error handling sees success; anything else synchronizes as [`ctx_sync`]
/// does.
pub fn ctx_sync_in_drop(ctx: &Arc<CudaContext>, site: &'static str) -> Result<(), DriverError> {
    let device = ctx.ordinal();
    if capturing(device) {
        eprintln!(
            "{site}: device {device} is capturing on this thread; the drop skips its \
             context synchronize"
        );
        return Ok(());
    }
    let lock = device_lock(device);
    let _held = lock.write().unwrap_or_else(PoisonError::into_inner);
    #[allow(
        clippy::disallowed_methods,
        reason = "the one owner of the context-wide synchronize"
    )]
    ctx.synchronize()
}

/// This thread's note that it captures on `device`, one entry of the
/// thread's capture set, removed when the note drops by scope or unwind —
/// the capture's end. Drop only removes the entry; it cannot panic.
struct CapturingOn(usize);

impl CapturingOn {
    fn note(device: usize) -> CapturingOn {
        CAPTURING.with(|open| open.borrow_mut().push(device));
        CapturingOn(device)
    }
}

impl Drop for CapturingOn {
    fn drop(&mut self) {
        CAPTURING.with(|open| {
            let mut open = open.borrow_mut();
            if let Some(at) = open.iter().rposition(|&device| device == self.0) {
                open.swap_remove(at);
            }
        });
    }
}

/// Run `during` — a capture's begin, body and end — with `device`'s lock
/// held shared and the capture noted on this thread: the shared hold and
/// the note come up together before the begin and come down together after
/// the end, on scope or unwind. A nested capture (this thread already
/// capturing on the device) only notes itself and takes no lock.
pub(crate) fn capture_hold<R>(device: usize, during: impl FnOnce() -> R) -> R {
    let lock = device_lock(device);
    let _held = if capturing(device) {
        None
    } else {
        Some(lock.read().unwrap_or_else(PoisonError::into_inner))
    };
    let _note = CapturingOn::note(device);
    during()
}

/// How long [`hold_frees`] waits for cuda-core's retirement thread to reach
/// its gate: the cleanups queued ahead of it run first, each a free that
/// waits for the device's queued work at most.
const FREES_ENTRY: Duration = Duration::from_secs(10);

/// The process's frees holds: how many are up, and the sender whose drop
/// opens the gate the retirement thread sits in while any is.
struct FreesHeld {
    count: usize,
    open: Option<mpsc::Sender<()>>,
}

static FREES: Mutex<FreesHeld> = Mutex::new(FreesHeld {
    count: 0,
    open: None,
});

thread_local! {
    /// The frees holds this thread has up: a settle of the retirement
    /// thread from under one would wait for itself
    /// (`tensor::try_settle_retirements` refuses it by name).
    static HOLDING: Cell<usize> = const { Cell::new(0) };
}

/// cuda-core's retirement thread held: while any `Frees` of the process is
/// up, that thread runs no cleanup — no buffer's `cuMemFree`, so none of the
/// implicit synchronizes this module's doc names — and what was queued runs
/// once the last one drops. The holds count: a second one, on any thread,
/// only adds to the count. Not `Send`: the thread that took it drops it.
#[must_use = "the retirement thread runs again as soon as the hold drops"]
pub struct Frees(PhantomData<*const ()>);

/// Hold cuda-core's retirement thread ([`Frees`]). The first hold of the
/// process queues a gate behind what the thread holds now, as an owner of an
/// event recorded on `quiet` — a stream no work is ever enqueued on, so the
/// gate waits on no stream — and returns once the thread sits in it: every
/// cleanup it started before has ended, and it starts none until the last
/// hold drops. Refused by name when the thread does not reach the gate in
/// [`FREES_ENTRY`]; the gate then opens at once when it is reached.
pub(crate) fn hold_frees(quiet: &CudaStream) -> Result<Frees, GpuError> {
    struct Gate {
        entered: mpsc::Sender<()>,
        open: mpsc::Receiver<()>,
    }
    impl Drop for Gate {
        /// Runs on the retirement thread: say it is here, then stay until
        /// the last hold drops its sender.
        fn drop(&mut self) {
            let _ = self.entered.send(());
            let _ = self.open.recv();
        }
    }
    let mut held = FREES.lock().unwrap_or_else(PoisonError::into_inner);
    if held.count == 0 {
        let (entered, at_gate) = mpsc::channel();
        let (open, opened) = mpsc::channel();
        cuda_core::completion::retain_until_complete(
            Gate {
                entered,
                open: opened,
            },
            quiet,
        )?;
        at_gate.recv_timeout(FREES_ENTRY).map_err(|_| {
            GpuError::state(
                "capsync::hold_frees",
                "cuda-core's retirement thread did not reach the frees gate within 10 s",
            )
        })?;
        held.open = Some(open);
    }
    held.count += 1;
    HOLDING.with(|h| h.set(h.get() + 1));
    Ok(Frees(PhantomData))
}

impl Drop for Frees {
    fn drop(&mut self) {
        HOLDING.with(|h| h.set(h.get().saturating_sub(1)));
        let mut held = FREES.lock().unwrap_or_else(PoisonError::into_inner);
        held.count = held.count.saturating_sub(1);
        if held.count == 0 {
            // The gate's receive ends with its sender: the thread runs on.
            held.open = None;
        }
    }
}

/// Whether this thread has a [`Frees`] hold up.
pub(crate) fn holding_frees() -> bool {
    HOLDING.with(Cell::get) > 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuda_core::{DeviceBuffer, sys};
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Instant;

    /// Runs on the driver's thread in the stream's order, and returns once
    /// the flag is up: the stage card's wait on a host store.
    unsafe extern "C" fn wait_open(flag: *mut c_void) {
        // SAFETY: the `Arc` count the test handed this call.
        let flag = unsafe { Arc::from_raw(flag.cast::<AtomicBool>()) };
        while !flag.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_micros(100));
        }
    }

    /// A pass launch takes the call's frees hold before its stream waits on a
    /// host store, so a free queued before the launch has run when the
    /// thread's next driver call comes, and none starts under the wait. Here
    /// the retirement thread runs a slow marker, a buffer is dropped behind
    /// that work, and the guarded call holds the frees ([`crate::Gpu::hold_frees_for_call`],
    /// what `GpuModel`'s pass launches take), parks the stream on a host
    /// store and makes a driver call after the busy work ended, before the
    /// store. Without the hold the free begins under the wait, takes the
    /// driver's lock and waits for the parked stream, and that call returns
    /// only when the watcher opens the store 5 s later. Runs alone
    /// (`hw_held_`): it parks a stream and makes a driver call before it
    /// opens it. FAIL-first: a `hold_frees_for_call` that holds nothing
    /// leaves the call waiting for the watcher.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_held_a_pass_launch_keeps_a_queued_free_out_of_its_host_wait() {
        crate::checkpoint::hold::require_serial_pass("capsync::tests (a parked pass)");
        struct Busy(mpsc::Sender<()>, mpsc::Sender<()>);
        impl Drop for Busy {
            fn drop(&mut self) {
                let _ = self.0.send(());
                std::thread::sleep(Duration::from_millis(100));
                let _ = self.1.send(());
            }
        }

        let gpu = crate::Gpu::new().expect("a Gpu on device 0");
        let stream = gpu.stream();
        let target = DeviceBuffer::<u32>::zeroed(stream, 1024).expect("a target");
        let dropped = DeviceBuffer::<u8>::zeroed(stream, 4096).expect("a buffer");
        stream.synchronize().expect("the buffers are ready");
        let (started_tx, started_rx) = mpsc::channel();
        let (ended_tx, ended_rx) = mpsc::channel();
        cuda_core::completion::retain_until_complete(Busy(started_tx, ended_tx), stream)
            .expect("work for the retirement thread");
        started_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the retirement thread is busy");
        drop(dropped);

        let watch = Arc::new(crate::watchdog::Watch::new());
        let call = crate::watchdog::busy(&watch, crate::watchdog::STEP, 0).expect("the call");
        gpu.hold_frees_for_call().expect("the frees hold");
        let flag = Arc::new(AtomicBool::new(false));
        let data = Arc::into_raw(Arc::clone(&flag)).cast_mut().cast::<c_void>();
        // SAFETY: `data` is a live `Arc` count that `wait_open` takes.
        let rc = unsafe { sys::cuLaunchHostFunc(stream.cu_stream(), Some(wait_open), data) };
        assert_eq!(
            rc,
            sys::cudaError_enum_CUDA_SUCCESS,
            "the stream parks on its host store"
        );
        let returned = Arc::new(AtomicBool::new(false));
        let watcher = std::thread::spawn({
            let (flag, returned) = (Arc::clone(&flag), Arc::clone(&returned));
            move || {
                let end = Instant::now() + Duration::from_secs(5);
                while Instant::now() < end && !returned.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                let wedged = !returned.load(Ordering::Acquire);
                flag.store(true, Ordering::Release);
                wedged
            }
        });
        ended_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("the busy work ended");
        // An unheld free begins once the busy work ends.
        std::thread::sleep(Duration::from_millis(20));
        // SAFETY: `target` holds 1024 words and lives until the stream is
        // synchronized below.
        let set =
            unsafe { sys::cuMemsetD32Async(target.cu_deviceptr(), 0, 1024, stream.cu_stream()) };
        returned.store(true, Ordering::Release);
        let wedged = watcher.join().expect("the watcher");
        flag.store(true, Ordering::Release);
        assert!(
            !wedged,
            "the driver call after the pass launch returns while the pass waits on its host \
             store: a free that began under the wait held the driver's lock until the watcher \
             opened the store"
        );
        assert_eq!(
            set,
            sys::cudaError_enum_CUDA_SUCCESS,
            "the memset is queued"
        );
        drop(call);
        stream.synchronize().expect("the stream ran on");
        assert_eq!(
            target.to_host_vec(stream).expect("the target"),
            vec![0; 1024]
        );
    }
}
