//! The Metal stream: the order the device's work runs in.
//!
//! The work of a stream is its committed command buffers, in the order they
//! were submitted. A buffer is made with [`MetalStream::begin`], encoded into,
//! and committed with [`MetalStream::submit`] or
//! [`MetalStream::submit_signalled`], which also register it: the stream
//! waits for exactly the buffers it submitted, and no other path commits one.
//! `record` and `wait` are signal-only and wait-only command buffers of the
//! same queue. The queue orders a signal-only buffer after the work submitted
//! before it and a wait-only buffer before the work submitted after it;
//! `event_record_orders` and `event_wait_holds` are the clauses that fail
//! where that does not hold.
//!
//! The buffers' resources are untracked, so the queue's order alone does not
//! keep one buffer's dispatches from overlapping the next one's. Work that
//! must follow the work before it opens with a token encoder
//! ([`Recording::token_encoder`], or [`Recording::open_token_encoder`] for a
//! caller that holds the encoder across calls): its buffer waits for the
//! stream's token boundary, a device event every signalled buffer signals at
//! its end.
//!
//! [`MetalStream::settle`] waits for one signalled buffer, not for every
//! pending one: it wakes on the stream's event at the value that buffer
//! signals, and returns the proof ([`Settled`]) a host read of what the
//! buffer wrote takes. Every host wait, a settle or a sync, is counted
//! ([`MetalStream::waits`]), and so is every committed buffer
//! ([`MetalStream::submits`]).

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use core::time::Duration;
use std::io::Write;
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Instant;

use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLDevice, MTLEvent, MTLSharedEvent,
};

use bloomery_device::{DeviceError, Event, Stream};

use crate::device::MetalDevice;
use crate::encode::{ComputeEncoder, EncodeCount};
use crate::handle::{CommandBufferHandle, Handle};
use crate::residency::Residency;

/// How long `sync` and an event's `sync` wait before they give up by name.
pub(crate) const SYNC_TIMEOUT: Duration = Duration::from_secs(60);

/// How long a wait for a command buffer sleeps between looks at its status.
const POLL: Duration = Duration::from_micros(50);

/// How long a settle waits on the event before it looks at the buffers'
/// status: a buffer that failed may never reach its signal.
const SETTLE_LOOK_MS: u64 = 100;

/// The buffers a stream holds while they run: the token a step waits for,
/// the token committed ahead of it, and one settled whose completion the
/// driver has not yet reported.
const PENDING: usize = 3;

/// The stream the device's work is enqueued on.
pub struct MetalStream {
    queue: Handle<dyn MTLCommandQueue>,
    /// The one event the stream's `record` and its signalled buffers signal,
    /// at rising values: the host's waits.
    timeline: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// The token boundary: every signalled buffer signals it at its end, at
    /// its timeline value, and a token encoder's buffer waits for the last
    /// such value. It is the device's own event, not the shared one: the
    /// device hands one buffer to the next through it with no host path.
    boundary: Retained<ProtocolObject<dyn MTLEvent>>,
    // The contract's `record` and `wait` take `&self` from any thread, so
    // the counter and the list of unfinished buffers are behind one lock: a
    // value is taken and its buffer committed in one step, so the values rise
    // in commit order.
    committed: Mutex<Committed>,
    /// The host's waits: every settle and every sync.
    waits: AtomicU64,
    /// The residency set of the buffers this stream's kernels bind, once
    /// [`MetalStream::enable_residency`] made one.
    residency: Option<Residency>,
}

/// What a stream has committed.
struct Committed {
    /// The value the last signal takes.
    last: u64,
    /// The value the last signalled buffer signals the boundary at.
    boundary: u64,
    /// The committed command buffers not yet seen complete.
    pending: Vec<CommandBufferHandle>,
    /// The command buffers committed so far.
    submits: u64,
}

impl Committed {
    /// Commit `buffer` and register it; the buffers already complete are
    /// dropped from the list first, so it holds only unfinished work and
    /// errors not yet reported.
    fn commit(&mut self, buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>) {
        self.pending
            .retain(|b| b.0.status() != MTLCommandBufferStatus::Completed);
        #[cfg(feature = "encode-trace")]
        crate::trace::commit(&buffer);
        mark("commit", format_args!("submit {}", self.submits + 1));
        buffer.commit();
        self.submits += 1;
        self.pending.push(CommandBufferHandle(buffer));
    }

    /// The next value a signal takes.
    fn next_value(&mut self) -> u64 {
        self.last += 1;
        self.last
    }
}

/// A recording committed with a signal of its own: the stream's event reaches
/// its value once the recording's work is done. It is settled by
/// [`MetalStream::settle`], which is where its failure is read.
#[must_use = "a signalled recording is settled, which reads its failure"]
#[derive(Debug)]
pub struct Signalled {
    value: u64,
}

impl Signalled {
    /// The signal's value: the token's identity on its stream, the key a
    /// holder of its stamps (the step clock) knows it by.
    pub(crate) fn value(&self) -> u64 {
        self.value
    }
}

/// The proof a signalled recording's work is done and what it wrote is the
/// host's to read: only [`MetalStream::settle`] makes one.
#[derive(Debug)]
pub struct Settled(());

/// A command buffer being encoded for a stream: made by
/// [`MetalStream::begin`], given its encoders, and committed by
/// [`MetalStream::submit`].
pub struct Recording<'s> {
    buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    stream: &'s MetalStream,
    /// The boundary value the buffer's first command waits for, 0 for none.
    waits_for: u64,
}

impl<'s> Recording<'s> {
    /// A compute encoder on the recording's command buffer. The recording is
    /// borrowed exclusively while the encoder lives: Metal ends one encoder
    /// before it starts another, and a recording cannot be submitted mid-encode.
    pub fn compute_encoder(&mut self) -> Result<ComputeEncoder<'_>, DeviceError> {
        ComputeEncoder::new(&self.buffer)
    }

    /// A compute encoder whose dispatches may overlap, on the recording's
    /// command buffer. The recording is borrowed exclusively while the
    /// encoder lives, as for [`Recording::compute_encoder`], and it cannot
    /// be submitted until the encoder ends.
    pub fn compute_encoder_concurrent(&mut self) -> Result<ComputeEncoder<'_>, DeviceError> {
        ComputeEncoder::concurrent(&self.buffer)
    }

    /// A concurrent compute encoder at the stream's token boundary: the
    /// buffer first waits for the boundary's last value, so the encoder's
    /// dispatches start after the work of every signalled buffer committed
    /// before it opened. The resources are untracked, so the queue's order
    /// does not keep two buffers' dispatches apart: without the wait, a
    /// token committed ahead starts before the token it embeds the pick of
    /// is done, and the e2e gate's `b_chain` is red.
    pub fn token_encoder(&mut self) -> Result<ComputeEncoder<'_>, DeviceError> {
        self.wait_for_boundary();
        ComputeEncoder::concurrent(&self.buffer)
    }

    /// [`Recording::token_encoder`], with the encoder kept beside the
    /// recording instead of borrowing it, so a caller holds the pair across
    /// calls and commits it when it chooses.
    pub fn open_token_encoder(mut self) -> Result<TokenRecording<'s>, DeviceError> {
        self.wait_for_boundary();
        let encoder = ComputeEncoder::concurrent_detached(&self.buffer)?;
        Ok(TokenRecording {
            encoder,
            recording: self,
        })
    }

    /// The buffer's first command: wait for the boundary's last value, when
    /// a signalled buffer has set one.
    fn wait_for_boundary(&mut self) {
        let last = self.stream.committed().boundary;
        self.waits_for = last;
        if last > 0 {
            #[cfg(feature = "encode-trace")]
            crate::trace::wait_event(&self.buffer, "boundary", last);
            self.buffer
                .encodeWaitForEvent_value(&self.stream.boundary, last);
        }
    }
}

/// A recording and its token encoder, open together: made by
/// [`Recording::open_token_encoder`], encoded into through
/// [`TokenRecording::encoder`], and closed by [`TokenRecording::finish`],
/// which ends the encoder and hands back the recording to commit. A
/// recording cannot be committed with its encoder open, so this type is the
/// only holder of both and ends the encoder before it gives the recording up.
pub struct TokenRecording<'s> {
    // Declared first, so a dropped pair ends its encoder before the buffer
    // goes.
    encoder: ComputeEncoder<'static>,
    recording: Recording<'s>,
}

impl<'s> TokenRecording<'s> {
    /// `f` over the open encoder. The encoder reaches `f` at a lifetime of
    /// its own, so it cannot leave the call or trade places with another
    /// pair's: a pair's `finish` always ends its own encoder.
    pub fn encoder<R>(&mut self, f: impl for<'e> FnOnce(&mut ComputeEncoder<'e>) -> R) -> R {
        f(&mut self.encoder)
    }

    /// End the encoder: the recording, ready to commit, and what the encoder
    /// encoded.
    #[must_use]
    pub fn finish(self) -> (Recording<'s>, EncodeCount) {
        let Self { encoder, recording } = self;
        let count = encoder.count();
        encoder.end();
        (recording, count)
    }
}

impl Recording<'_> {
    /// A handle on this recording's GPU stamps, read once
    /// [`MetalStream::submit`] has committed it and the GPU has run it.
    #[must_use]
    pub fn stamps(&self) -> GpuStamps {
        GpuStamps(CommandBufferHandle(self.buffer.clone()))
    }
}

/// The GPU's own start and end stamps of one recording's command buffer.
/// The handle is [`Send`](CommandBufferHandle), so a lock may hold the stamps
/// across threads; it is not `Sync`, and never leaves one.
pub struct GpuStamps(CommandBufferHandle);

/// A token's recording, encoded and held off the queue: made by
/// [`MetalStream::park`], committed later through [`MetalStream::unpark`].
/// Its buffer waits for the boundary value the stream held when it was
/// encoded, so it may run only while that is still the stream's boundary: a
/// signalled buffer committed in between makes it stale, and `unpark` gives
/// nothing back. A parked recording dropped is never committed, so it never
/// runs. It does not borrow the stream: a session holds it beside its own.
pub struct Parked {
    buffer: CommandBufferHandle,
    /// The boundary event the buffer's wait names, held so that no other
    /// stream's event takes its address while the recording is parked.
    boundary: Retained<ProtocolObject<dyn MTLEvent>>,
    waits_for: u64,
}

/// The GPU's own stamps of one recording's buffer, in seconds on the clock
/// [`host_secs`] reads: Apple GPUs stamp the mach timebase the host shares.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuTimes {
    pub start: f64,
    pub end: f64,
}

/// What [`GpuStamps::interval`] saw within its wait.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuInterval {
    /// The buffer completed: the seconds the GPU ran it.
    Ran(f64),
    /// The buffer was still running when the wait ended.
    StillRunning,
}

impl GpuStamps {
    /// The buffer's `GPUStartTime` and `GPUEndTime` once it has completed,
    /// polled for at most `wait`. The stamps are the driver's record of when
    /// the GPU ran the buffer, so a host that reads them late reads the same
    /// times. A buffer that failed, one that did not complete within the
    /// wait, and one that completed with no stamps or a non-positive
    /// interval are refused by name.
    pub fn times(&self, wait: Duration) -> Result<GpuTimes, DeviceError> {
        let (start, end) = self.raw(wait)?.ok_or_else(|| {
            DeviceError::backend(
                "command buffer",
                format!(
                    "the command buffer did not complete within {} ms, so its GPU times cannot \
                     be read",
                    wait.as_millis()
                ),
            )
        })?;
        Ok(GpuTimes { start, end })
    }

    /// `GPUEndTime` minus `GPUStartTime` once the buffer has completed, polled
    /// for at most `wait`; a buffer still running at the wait's end says so.
    /// The refusals are [`GpuStamps::times`]'s.
    pub fn interval(&self, wait: Duration) -> Result<GpuInterval, DeviceError> {
        match self.raw(wait)? {
            Some((start, end)) => Ok(GpuInterval::Ran(end - start)),
            None => Ok(GpuInterval::StillRunning),
        }
    }

    /// The stamps of a completed buffer, `None` while it still runs, with the
    /// failure, the missing stamps and the non-positive interval refused by
    /// name: the one read both readers of the stamps share.
    fn raw(&self, wait: Duration) -> Result<Option<(f64, f64)>, DeviceError> {
        let buffer = &self.0.0;
        let refuse = |detail: String| DeviceError::backend("command buffer", detail);
        let started = Instant::now();
        let status = loop {
            let status = buffer.status();
            if status == MTLCommandBufferStatus::Completed
                || status == MTLCommandBufferStatus::Error
            {
                break status;
            }
            if started.elapsed() > wait {
                return Ok(None);
            }
            std::thread::sleep(POLL);
        };
        if status == MTLCommandBufferStatus::Error {
            let detail = buffer.error().map_or_else(
                || "the command buffer failed".to_string(),
                |err| err.localizedDescription().to_string(),
            );
            return Err(refuse(format!(
                "the command buffer failed on the GPU: {detail}"
            )));
        }
        // The typed getters sit behind objc2-metal's `objc2-core-foundation`
        // feature, which this crate does not enable; the selectors are the
        // command buffer's own.
        // SAFETY: `GPUStartTime` and `GPUEndTime` are MTLCommandBuffer's
        // getters, each returning a `CFTimeInterval` (`f64`), sent to a live
        // retained command buffer that has completed.
        let (start, end): (f64, f64) = unsafe {
            (
                msg_send![&**buffer, GPUStartTime],
                msg_send![&**buffer, GPUEndTime],
            )
        };
        if start == 0.0 || end == 0.0 {
            return Err(refuse(
                "the command buffer completed and reports no GPU start or end time".to_string(),
            ));
        }
        let gpu = end - start;
        if !(gpu.is_finite() && gpu > 0.0) {
            return Err(refuse(format!(
                "the GPU's own interval is {gpu} s, not a positive finite duration"
            )));
        }
        Ok(Some((start, end)))
    }
}

/// The mach tick in seconds, as `mach_timebase_info` states it: the clock's
/// ticks are the timebase's fraction of a nanosecond, which is a nanosecond
/// on some machines and not on others. Read once.
static TICK_S: OnceLock<f64> = OnceLock::new();

/// The host's seconds on the clock the GPU's stamps read: macOS's
/// `mach_absolute_time`, scaled by the timebase. A command buffer's
/// `GPUStartTime` and `GPUEndTime` are seconds on this clock, so a stamp and
/// a host read subtract directly.
pub(crate) fn host_secs() -> f64 {
    #[repr(C)]
    struct Timebase {
        numer: u32,
        denom: u32,
    }
    unsafe extern "C" {
        fn mach_absolute_time() -> u64;
        fn mach_timebase_info(info: *mut Timebase) -> i32;
    }
    let per_tick = *TICK_S.get_or_init(|| {
        let mut tb = Timebase { numer: 0, denom: 0 };
        // SAFETY: `mach_timebase_info` writes the timebase into `tb`, which
        // matches `mach_timebase_info_data_t`'s layout, and reads nothing.
        let r = unsafe { mach_timebase_info(&mut tb) };
        assert!(
            r == 0 && tb.denom != 0,
            "mach_timebase_info said {r} with a timebase of {}/{}",
            tb.numer,
            tb.denom
        );
        f64::from(tb.numer) / f64::from(tb.denom) / 1e9
    });
    // SAFETY: `mach_absolute_time` is libSystem's clock read, declared
    // above; it takes no arguments and writes nothing.
    (unsafe { mach_absolute_time() } as f64) * per_tick
}

/// The crash breadcrumb file, opened once from `BLOOMERY_METAL_BREADCRUMB`
/// (the one Metal-host debug lever), or `None` when the name is unset. Each
/// driver-facing call class appends one line through it and fsyncs before the
/// call, so the file's last line names the call in flight after a panic.
/// Unset, [`mark`] is one branch and no work, off the step path's cost.
static BREADCRUMB: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();

/// The breadcrumb file, opened on the first [`mark`]. Read in place: the one
/// Metal-host lever (`BLOOMERY_METAL_BREADCRUMB`,
/// `crates/levers/src/registry.rs`); a name that cannot be opened for append
/// turns the breadcrumb off with one named stderr line, since the breadcrumb
/// only observes and must not stop the run it watches.
fn breadcrumb() -> &'static Option<Mutex<std::fs::File>> {
    BREADCRUMB.get_or_init(|| {
        let path = std::env::var_os("BLOOMERY_METAL_BREADCRUMB")?;
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(file) => Some(Mutex::new(file)),
            Err(e) => {
                eprintln!(
                    "bloomery_metal: BLOOMERY_METAL_BREADCRUMB={}: cannot open for append ({e}); \
                     the breadcrumb is off",
                    path.to_string_lossy()
                );
                None
            }
        }
    })
}

/// One breadcrumb line before a driver-facing call, then fsync; nothing when
/// the breadcrumb is unset (one branch, no formatting — `detail` is lazy).
/// `class` names the call class (`queue`, `set`, `event`, `commit`, `wait`,
/// `teardown`) and `detail` the value in flight. A poisoned lock is taken as
/// it stands: the file is only appended to.
pub(crate) fn mark(class: &str, detail: std::fmt::Arguments<'_>) {
    let Some(file) = breadcrumb() else {
        return;
    };
    let mut file = file.lock().unwrap_or_else(PoisonError::into_inner);
    // fsync so the line survives the kernel panic the breadcrumb hunts.
    let written =
        writeln!(file, "{:.9} {class} {detail}", host_secs()).and_then(|()| file.sync_data());
    if let Err(e) = written {
        // A line that did not reach the disk leaves an older last line naming
        // the wrong call after a panic: said once, the breadcrumb then stale.
        static SAID: AtomicBool = AtomicBool::new(false);
        if !SAID.swap(true, Ordering::Relaxed) {
            eprintln!(
                "bloomery_metal: BLOOMERY_METAL_BREADCRUMB: a line before `{class}` did not reach \
                 the disk ({e}); the file's last line may name an earlier call"
            );
        }
    }
}

/// A breadcrumb line at the start of a gate clause, so the file after a panic
/// names the clause that was running. The one breadcrumb entry the crate's
/// binaries reach from outside the crate; unset, it is one branch.
pub fn breadcrumb_clause(name: &str) {
    mark("clause", format_args!("{name}"));
}

impl MetalStream {
    /// A stream on `queue`, with its own event.
    pub(crate) fn new(
        device: Handle<dyn MTLDevice>,
        queue: Handle<dyn MTLCommandQueue>,
    ) -> Result<Self, DeviceError> {
        let timeline = device.newSharedEvent().ok_or_else(|| {
            DeviceError::backend(
                "shared event",
                "the device made no shared event".to_string(),
            )
        })?;
        let boundary = device
            .newEvent()
            .ok_or_else(|| DeviceError::backend("event", "the device made no event".to_string()))?;
        Ok(Self {
            queue,
            timeline,
            boundary,
            committed: Mutex::new(Committed {
                last: 0,
                boundary: 0,
                // The list starts with capacity, so no commit from the first
                // step on grows it; a sync drops finished buffers in place.
                pending: Vec::with_capacity(PENDING),
                submits: 0,
            }),
            waits: AtomicU64::new(0),
            residency: None,
        })
    }

    /// Give the stream a residency set on `device`, empty: the buffers made
    /// through [`Alloc`](crate::residency::Alloc) join it, and every submit
    /// renews its request. Enabling twice is refused by name.
    pub fn enable_residency(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<(), DeviceError> {
        if self.residency.is_some() {
            return Err(DeviceError::backend(
                "residency set",
                "the stream has one already".to_string(),
            ));
        }
        self.residency = Some(Residency::new(device, &self.queue)?);
        Ok(())
    }

    /// The stream's residency set, once enabled.
    #[must_use]
    pub fn residency(&self) -> Option<&Residency> {
        self.residency.as_ref()
    }

    /// The host's waits on this stream so far: every settle and every sync.
    #[must_use]
    pub fn waits(&self) -> u64 {
        self.waits.load(Ordering::Relaxed)
    }

    /// The command buffers this stream has committed so far: every submit,
    /// `record` and `wait`.
    #[must_use]
    pub fn submits(&self) -> u64 {
        self.committed().submits
    }

    /// The committed state. A panic while it was held left it whole (every
    /// change to it is one push, one retain or one count), so a poisoned lock is taken
    /// as it stands.
    fn committed(&self) -> MutexGuard<'_, Committed> {
        self.committed
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// A command buffer of this stream's queue, not yet committed.
    fn command_buffer(
        &self,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, DeviceError> {
        let made = self.queue.commandBuffer().ok_or_else(|| {
            DeviceError::backend(
                "command buffer",
                "the queue made no command buffer".to_string(),
            )
        });
        #[cfg(feature = "encode-trace")]
        if let Ok(buffer) = &made {
            crate::trace::command_buffer_made(buffer);
        }
        made
    }

    /// A recording for the next piece of work.
    pub fn begin(&self) -> Result<Recording<'_>, DeviceError> {
        Ok(Recording {
            buffer: self.command_buffer()?,
            stream: self,
            waits_for: 0,
        })
    }

    /// Hold a recording whose encoders have ended off the queue, to commit
    /// it later ([`MetalStream::unpark`]). A recording made by another
    /// stream is refused by name.
    pub fn park(&self, recording: Recording<'_>) -> Result<Parked, DeviceError> {
        if !core::ptr::eq(recording.stream, self) {
            return Err(DeviceError::backend(
                "command buffer",
                "the recording was made by another stream".to_string(),
            ));
        }
        Ok(Parked {
            buffer: CommandBufferHandle(recording.buffer),
            boundary: self.boundary.clone(),
            waits_for: recording.waits_for,
        })
    }

    /// A parked recording back, to submit: `None` when it is stale — the
    /// stream's boundary has moved past the value its wait names, so another
    /// token's work would run between them — or when another stream parked
    /// it. A stale recording is dropped uncommitted.
    #[must_use]
    pub fn unpark(&self, parked: Parked) -> Option<Recording<'_>> {
        let ours = Retained::as_ptr(&parked.boundary) == Retained::as_ptr(&self.boundary);
        let fresh = ours && parked.waits_for == self.committed().boundary;
        fresh.then(|| Recording {
            buffer: parked.buffer.0,
            stream: self,
            waits_for: parked.waits_for,
        })
    }

    /// Commit a recording's command buffer and register it, so that `sync`
    /// and every read wait for it. A recording made by another stream is
    /// refused by name.
    pub fn submit(&self, recording: Recording<'_>) -> Result<(), DeviceError> {
        if !core::ptr::eq(recording.stream, self) {
            return Err(DeviceError::backend(
                "command buffer",
                "the recording was made by another stream".to_string(),
            ));
        }
        self.keep_resident();
        self.committed().commit(recording.buffer);
        Ok(())
    }

    /// The residency set renewed before work is committed.
    fn keep_resident(&self) {
        if let Some(residency) = &self.residency {
            residency.keep();
        }
    }

    /// Commit a recording's command buffer with a signal of its own and
    /// register it; [`MetalStream::settle`] waits for that signal, and the
    /// next token encoder for the buffer's end. A recording made by another
    /// stream is refused by name.
    pub fn submit_signalled(&self, recording: Recording<'_>) -> Result<Signalled, DeviceError> {
        if !core::ptr::eq(recording.stream, self) {
            return Err(DeviceError::backend(
                "command buffer",
                "the recording was made by another stream".to_string(),
            ));
        }
        self.keep_resident();
        let mut c = self.committed();
        let value = c.next_value();
        #[cfg(feature = "encode-trace")]
        {
            crate::trace::signal_event(&recording.buffer, "boundary", value);
            crate::trace::signal_event(&recording.buffer, "timeline", value);
        }
        mark("event", format_args!("signal boundary+timeline {value}"));
        recording
            .buffer
            .encodeSignalEvent_value(&self.boundary, value);
        recording
            .buffer
            .encodeSignalEvent_value(ProtocolObject::from_ref(&*self.timeline), value);
        c.boundary = value;
        c.commit(recording.buffer);
        Ok(Signalled { value })
    }

    /// Wait for one signalled recording, not for every pending one. The wait
    /// sleeps on the stream's event and looks at the buffers' status between
    /// [`SETTLE_LOOK_MS`] waits, since a buffer that failed may never reach
    /// its signal; past [`SYNC_TIMEOUT`] it is refused by name. A failure the
    /// driver has reported by the time the signal arrives is named here; one
    /// it reports later is named by the next settle or sync.
    pub fn settle(&self, signalled: Signalled) -> Result<Settled, DeviceError> {
        self.waits.fetch_add(1, Ordering::Relaxed);
        let value = signalled.value;
        #[cfg(feature = "encode-trace")]
        crate::trace::host(|| format!("settle {value}"));
        mark("wait", format_args!("settle {value}"));
        let deadline = Instant::now() + SYNC_TIMEOUT;
        while !self
            .timeline
            .waitUntilSignaledValue_timeoutMS(value, SETTLE_LOOK_MS)
        {
            self.reap()?;
            if Instant::now() >= deadline {
                return Err(DeviceError::backend(
                    "command buffer",
                    format!(
                        "signal {value} not reached within {} ms",
                        SYNC_TIMEOUT.as_millis()
                    ),
                ));
            }
        }
        self.reap()?;
        Ok(Settled(()))
    }

    /// Drop the buffers seen complete and name the first one seen failed; a
    /// buffer still running stays pending.
    fn reap(&self) -> Result<(), DeviceError> {
        let mut first = None;
        self.committed().pending.retain(|b| match b.0.status() {
            MTLCommandBufferStatus::Completed => false,
            MTLCommandBufferStatus::Error => {
                first.get_or_insert_with(|| command_buffer_error(b));
                false
            }
            _ => true,
        });
        first.map_or(Ok(()), Err)
    }

    /// Every submitted buffer is done, or `timeout` has passed. Every pending
    /// buffer is looked at even when one failed, the first failure is
    /// returned, and a buffer that did not finish stays pending. The list is
    /// held under the lock through the wait, so no thread sees the stream
    /// settled while work submitted before its call still runs; a thread
    /// queued behind another thread's wait gives up by name at its own
    /// deadline.
    pub fn sync_for(&self, timeout: Duration) -> Result<(), DeviceError> {
        use std::sync::TryLockError::{Poisoned, WouldBlock};
        self.waits.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "encode-trace")]
        crate::trace::host(|| "sync".to_string());
        mark("wait", format_args!("sync {} ms", timeout.as_millis()));
        let deadline = Instant::now() + timeout;
        let not_done = || {
            DeviceError::backend(
                "command buffer",
                format!("not done within {} ms", timeout.as_millis()),
            )
        };
        let mut committed = loop {
            match self.committed.try_lock() {
                Ok(state) => break state,
                // A poisoned lock is taken as it stands: `committed` holds
                // the reason.
                Err(Poisoned(poison)) => break poison.into_inner(),
                Err(WouldBlock) => {
                    if Instant::now() >= deadline {
                        return Err(not_done());
                    }
                    std::thread::sleep(POLL);
                }
            }
        };
        let mut first = None;
        // One status read decides each buffer: a finished one leaves the
        // list (an error is reported once, to the caller that saw it), an
        // unfinished one stays for the next wait.
        committed.pending.retain(|buffer| {
            while !is_over(buffer.0.status()) && Instant::now() < deadline {
                std::thread::sleep(POLL);
            }
            match buffer.0.status() {
                MTLCommandBufferStatus::Completed => false,
                MTLCommandBufferStatus::Error => {
                    first.get_or_insert_with(|| command_buffer_error(buffer));
                    false
                }
                _ => {
                    first.get_or_insert_with(not_done);
                    true
                }
            }
        });
        drop(committed);
        first.map_or(Ok(()), Err)
    }
}

impl Drop for MetalStream {
    /// Drain the queue before the residency set is torn down.
    ///
    /// The residency set, and the buffers it holds as allocations, are
    /// attached to every command buffer the shared queue commits, and Metal
    /// lets a resource leave a set only once the GPU no longer needs it. A
    /// kernel fault in the driver's resource-group update on this queue
    /// (`set_resource_groups` dereferencing a resource that is gone) is read
    /// as that rule broken at teardown. The invariant: the set outlives every
    /// command buffer on the queue that it backs. So a stream with a set waits (bounded by [`SYNC_TIMEOUT`]) for
    /// every committed buffer to finish before `residency` drops
    /// ([`Residency`]'s `endResidency`, `removeResidencySet`, the buffers
    /// released). A stream with no set (the device's own) has nothing to free
    /// under the GPU and returns at once.
    fn drop(&mut self) {
        if self.residency.is_none() {
            return;
        }
        let pending = self.committed().pending.len();
        mark("teardown", format_args!("drain {pending} pending"));
        // The drain takes each finished buffer off the list, so a buffer that
        // ended in error is named here or nowhere; the pending list then
        // decides whether the set may go.
        if let Err(e) = self.sync_for(SYNC_TIMEOUT) {
            eprintln!("bloomery_metal: at a stream's teardown: {e}");
        }
        if !self.committed().pending.is_empty() {
            // The queue did not drain inside the bound: a command buffer is
            // still in flight. Freeing the set now would fault the driver, so
            // the set (and the buffers it retains) is kept for the process's
            // remaining life rather than torn down under the GPU. Loud, not
            // silent: a leaked set at teardown is a wedged queue to report.
            eprintln!(
                "bloomery_metal: a stream's command buffers did not finish within {} ms at \
                 teardown; its residency set is leaked, not freed, so the driver never walks a \
                 freed resource",
                SYNC_TIMEOUT.as_millis()
            );
            if let Some(residency) = self.residency.take() {
                core::mem::forget(residency);
            }
        }
    }
}

/// Whether a command buffer has finished, by completing or by failing.
fn is_over(status: MTLCommandBufferStatus) -> bool {
    matches!(
        status,
        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
    )
}

/// The failure a command buffer ended with, in its own words.
fn command_buffer_error(buffer: &CommandBufferHandle) -> DeviceError {
    let detail = buffer.0.error().map_or_else(
        || "the command buffer failed".to_string(),
        |e| e.localizedDescription().to_string(),
    );
    DeviceError::backend("command buffer", detail)
}

impl Stream for MetalStream {
    type Event = MetalEvent;

    fn record(&self) -> Result<Self::Event, DeviceError> {
        let buffer = self.command_buffer()?;
        let mut committed = self.committed();
        let value = committed.next_value();
        #[cfg(feature = "encode-trace")]
        crate::trace::signal_event(&buffer, "timeline", value);
        mark("event", format_args!("record signal timeline {value}"));
        buffer.encodeSignalEvent_value(ProtocolObject::from_ref(&*self.timeline), value);
        committed.commit(buffer);
        Ok(MetalEvent {
            event: self.timeline.clone(),
            value,
        })
    }

    fn wait(&self, e: &Self::Event) -> Result<(), DeviceError> {
        let buffer = self.command_buffer()?;
        #[cfg(feature = "encode-trace")]
        crate::trace::wait_event(&buffer, "timeline", e.value);
        mark("event", format_args!("wait timeline {}", e.value));
        buffer.encodeWaitForEvent_value(ProtocolObject::from_ref(&*e.event), e.value);
        self.committed().commit(buffer);
        Ok(())
    }

    fn sync(&self) -> Result<(), DeviceError> {
        self.sync_for(SYNC_TIMEOUT)
    }
}

/// A point in a stream's order: a shared event and the value it fires at.
pub struct MetalEvent {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    value: u64,
}

impl MetalEvent {
    /// The value the event fires at; events of one stream rise in the order
    /// they were recorded.
    #[must_use]
    pub fn value(&self) -> u64 {
        self.value
    }

    /// The event has fired, or `timeout` has passed and the wait is refused
    /// by name.
    pub fn sync_for(&self, timeout: Duration) -> Result<(), DeviceError> {
        let ms = u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX);
        if self.event.waitUntilSignaledValue_timeoutMS(self.value, ms) {
            Ok(())
        } else {
            Err(DeviceError::backend(
                "event",
                format!("value {} was not signalled within {ms} ms", self.value),
            ))
        }
    }
}

impl Event for MetalEvent {
    fn sync(&self) -> Result<(), DeviceError> {
        self.sync_for(SYNC_TIMEOUT)
    }

    fn done(&self) -> Result<bool, DeviceError> {
        Ok(self.event.signaledValue() >= self.value)
    }
}

/// An event the host signals: a gate for work that waits on it, which holds
/// until [`HostEvent::signal`].
pub struct HostEvent {
    event: MetalEvent,
}

impl HostEvent {
    /// An event, not yet signalled.
    pub fn new(device: &MetalDevice) -> Result<Self, DeviceError> {
        let event = device.metal().newSharedEvent().ok_or_else(|| {
            DeviceError::backend(
                "shared event",
                "the device made no shared event".to_string(),
            )
        })?;
        Ok(Self {
            event: MetalEvent { event, value: 1 },
        })
    }

    /// The event as a stream waits for it.
    #[must_use]
    pub fn event(&self) -> &MetalEvent {
        &self.event
    }

    /// The host signals the event: the work that waits for it may run.
    pub fn signal(&self) {
        self.event.event.setSignaledValue(self.event.value);
    }
}
