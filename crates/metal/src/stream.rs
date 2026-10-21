//! The Metal stream: the order the device's work runs in.
//!
//! The work of a stream is its committed command buffers, in the order they
//! were submitted. A buffer is made with [`MetalStream::begin`], encoded into,
//! and committed with [`MetalStream::submit`], which also registers it: the
//! stream waits for exactly the buffers it submitted, and no other path
//! commits one. `record` and `wait` are signal-only and wait-only command
//! buffers of the same queue. Two command buffers of one queue did not overlap
//! on the machine the gate measured (its `queue_overlap` clause reports the
//! answer on every machine), so the queue orders a signal-only buffer after
//! the work submitted before it and a wait-only buffer before the work
//! submitted after it; `event_record_orders` and `event_wait_holds` are the
//! clauses that fail where that does not hold.

use core::time::Duration;
use std::cell::{Cell, RefCell};
use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLDevice, MTLSharedEvent,
};

use bloomery_device::{DeviceError, Event, Stream};

use crate::device::MetalDevice;
use crate::encode::ComputeEncoder;
use crate::handle::{CommandBufferHandle, Handle};

/// How long `sync` and an event's `sync` wait before they give up by name.
pub(crate) const SYNC_TIMEOUT: Duration = Duration::from_secs(60);

/// How long a wait for a command buffer sleeps between looks at its status.
const POLL: Duration = Duration::from_micros(50);

/// The stream the device's work is enqueued on.
pub struct MetalStream {
    queue: Handle<dyn MTLCommandQueue>,
    /// The one event the stream's `record` signals, at rising values.
    timeline: Handle<dyn MTLSharedEvent>,
    // The contract's `record` and `wait` take `&self`, so the counter and the
    // list of unfinished buffers are interior; the stream is `Send` and not
    // `Sync`, so one thread touches them at a time.
    /// The value the last `record` signals.
    last: Cell<u64>,
    /// The committed command buffers not yet seen complete.
    pending: RefCell<Vec<CommandBufferHandle>>,
}

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
        Ok(Self {
            queue,
            timeline: Handle(timeline),
            last: Cell::new(0),
            pending: RefCell::new(Vec::new()),
        })
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
        self.commit(recording.buffer);
        Ok(())
    }

    /// Commit `buffer` and register it; the buffers already complete are
    /// dropped from the list first, so it holds only unfinished work and
    /// errors not yet reported.
    fn commit(&self, buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>) {
        let mut pending = self.pending.borrow_mut();
        pending.retain(|b| b.0.status() != MTLCommandBufferStatus::Completed);
        buffer.commit();
        pending.push(CommandBufferHandle(buffer));
    }

    /// Every submitted buffer is done, or `timeout` has passed. Every pending
    /// buffer is looked at even when one failed, the first failure is
    /// returned, and a buffer that did not finish stays pending.
    pub fn sync_for(&self, timeout: Duration) -> Result<(), DeviceError> {
        let deadline = Instant::now() + timeout;
        let pending = std::mem::take(&mut *self.pending.borrow_mut());
        let mut first = None;
        let mut unfinished = Vec::new();
        for buffer in pending {
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
        self.pending.borrow_mut().extend(unfinished);
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
        let value = self.last.get() + 1;
        let buffer = self.command_buffer()?;
        buffer.encodeSignalEvent_value(ProtocolObject::from_ref(&*self.timeline), value);
        self.commit(buffer);
        self.last.set(value);
        Ok(MetalEvent {
            event: self.timeline.clone(),
            value,
        })
    }

    fn wait(&self, e: &Self::Event) -> Result<(), DeviceError> {
        let buffer = self.command_buffer()?;
        buffer.encodeWaitForEvent_value(ProtocolObject::from_ref(&*e.event), e.value);
        self.commit(buffer);
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
