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
//! ([`Recording::token_encoder`]): its buffer waits for the stream's token
//! boundary, a device event every signalled buffer signals at its end.
//!
//! [`MetalStream::settle`] waits for one signalled buffer, not for every
//! pending one: it wakes on the stream's event at the value that buffer
//! signals, and returns the proof ([`Settled`]) a host read of what the
//! buffer wrote takes. Every host wait, a settle or a sync, is counted
//! ([`MetalStream::waits`]).

use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLDevice, MTLEvent, MTLSharedEvent,
};

use bloomery_device::{DeviceError, Event, Stream};

use crate::device::MetalDevice;
use crate::encode::ComputeEncoder;
use crate::handle::{CommandBufferHandle, Handle};

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
    timeline: Handle<dyn MTLSharedEvent>,
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
}

/// What a stream has committed.
struct Committed {
    /// The value the last signal takes.
    last: u64,
    /// The value the last signalled buffer signals the boundary at.
    boundary: u64,
    /// The committed command buffers not yet seen complete.
    pending: Vec<CommandBufferHandle>,
    /// An empty list holding capacity for the one [`MetalStream::sync_for`]
    /// leaves in `pending`'s place while it waits, so the wait commits into
    /// capacity and the sync itself allocates nothing.
    spare: Vec<CommandBufferHandle>,
}

impl Committed {
    /// Commit `buffer` and register it; the buffers already complete are
    /// dropped from the list first, so it holds only unfinished work and
    /// errors not yet reported.
    fn commit(&mut self, buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>) {
        self.pending
            .retain(|b| b.0.status() != MTLCommandBufferStatus::Completed);
        buffer.commit();
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
}

impl Recording<'_> {
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
        let last = self.stream.committed().boundary;
        if last > 0 {
            self.buffer
                .encodeWaitForEvent_value(&self.stream.boundary, last);
        }
        ComputeEncoder::concurrent(&self.buffer)
    }
}

impl Recording<'_> {
    /// A handle on this recording's GPU stamps, read once
    /// [`MetalStream::submit`] has committed it and the GPU has run it.
    #[must_use]
    pub fn stamps(&self) -> GpuStamps {
        GpuStamps(self.buffer.clone())
    }
}

/// The GPU's own start and end stamps of one recording's command buffer.
pub struct GpuStamps(Retained<ProtocolObject<dyn MTLCommandBuffer>>);

/// What [`GpuStamps::interval`] saw within its wait.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuInterval {
    /// The buffer completed: the seconds the GPU ran it.
    Ran(f64),
    /// The buffer was still running when the wait ended.
    StillRunning,
}

impl GpuStamps {
    /// `GPUEndTime` minus `GPUStartTime` once the buffer has completed, polled
    /// for at most `wait`. The stamps are the driver's record of when the GPU
    /// ran the buffer, so a host that reads them late reads the same interval.
    /// A buffer that failed, or completed with no stamps or a non-positive
    /// interval, is refused by name.
    pub fn interval(&self, wait: Duration) -> Result<GpuInterval, DeviceError> {
        let refuse = |detail: String| DeviceError::backend("command buffer", detail);
        let started = Instant::now();
        let status = loop {
            let status = self.0.status();
            if status == MTLCommandBufferStatus::Completed
                || status == MTLCommandBufferStatus::Error
            {
                break status;
            }
            if started.elapsed() > wait {
                return Ok(GpuInterval::StillRunning);
            }
            std::thread::sleep(POLL);
        };
        if status == MTLCommandBufferStatus::Error {
            let detail = self.0.error().map_or_else(
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
                msg_send![&*self.0, GPUStartTime],
                msg_send![&*self.0, GPUEndTime],
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
        Ok(GpuInterval::Ran(gpu))
    }
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
            timeline: Handle(timeline),
            boundary,
            committed: Mutex::new(Committed {
                last: 0,
                boundary: 0,
                // Both lists start with capacity, so no commit or sync from
                // the first step on grows one.
                pending: Vec::with_capacity(PENDING),
                spare: Vec::with_capacity(PENDING),
            }),
            waits: AtomicU64::new(0),
        })
    }

    /// The host's waits on this stream so far: every settle and every sync.
    #[must_use]
    pub fn waits(&self) -> u64 {
        self.waits.load(Ordering::Relaxed)
    }

    /// The committed state. A panic while it was held left it whole (every
    /// change to it is one push or one retain), so a poisoned lock is taken
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
        self.queue.commandBuffer().ok_or_else(|| {
            DeviceError::backend(
                "command buffer",
                "the queue made no command buffer".to_string(),
            )
        })
    }

    /// A recording for the next piece of work.
    pub fn begin(&self) -> Result<Recording<'_>, DeviceError> {
        Ok(Recording {
            buffer: self.command_buffer()?,
            stream: self,
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
        self.committed().commit(recording.buffer);
        Ok(())
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
        let mut c = self.committed();
        let value = c.next_value();
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
    /// returned, and a buffer that did not finish stays pending.
    pub fn sync_for(&self, timeout: Duration) -> Result<(), DeviceError> {
        self.waits.fetch_add(1, Ordering::Relaxed);
        let deadline = Instant::now() + timeout;
        // The spare list stands in for `pending` while the wait runs, so the
        // lock is held by no poll and a submit on another thread pushes into
        // capacity; the lists' capacity is the stream's, so a sync that finds
        // every buffer done allocates nothing.
        let mut pending = {
            let mut c = self.committed();
            let spare = core::mem::take(&mut c.spare);
            core::mem::replace(&mut c.pending, spare)
        };
        let mut first = None;
        let mut unfinished = Vec::new();
        for buffer in pending.drain(..) {
            while !is_over(buffer.0.status()) && Instant::now() < deadline {
                std::thread::sleep(POLL);
            }
            match buffer.0.status() {
                MTLCommandBufferStatus::Completed => {}
                MTLCommandBufferStatus::Error => {
                    first.get_or_insert_with(|| command_buffer_error(&buffer));
                }
                _ => {
                    first.get_or_insert_with(|| {
                        DeviceError::backend(
                            "command buffer",
                            format!("not done within {} ms", timeout.as_millis()),
                        )
                    });
                    unfinished.push(buffer);
                }
            }
        }
        {
            let mut c = self.committed();
            c.pending.append(&mut unfinished);
            c.spare = pending;
        }
        first.map_or(Ok(()), Err)
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
        buffer.encodeSignalEvent_value(ProtocolObject::from_ref(&*self.timeline), value);
        committed.commit(buffer);
        Ok(MetalEvent {
            event: self.timeline.clone(),
            value,
        })
    }

    fn wait(&self, e: &Self::Event) -> Result<(), DeviceError> {
        let buffer = self.command_buffer()?;
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
    event: Handle<dyn MTLSharedEvent>,
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
            event: MetalEvent {
                event: Handle(event),
                value: 1,
            },
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
