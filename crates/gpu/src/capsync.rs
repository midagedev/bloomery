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
//! A capture holds one device's shared side for its whole body; a body
//! that also took a second device's exclusive side could meet another
//! thread doing the two in the opposite order. No capture body of the
//! engine opens a handle.

use crate::{GpuError, StreamRole, role_stream};
use cuda_core::{CudaContext, CudaStream, DriverError};
use std::cell::RefCell;
use std::sync::{Arc, Mutex, PoisonError, RwLock};

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
