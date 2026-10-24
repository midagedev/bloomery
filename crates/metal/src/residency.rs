//! The residency of a stream's buffers: one Metal residency set holding every
//! buffer a session's kernels bind, requested resident at open and again
//! whenever a submit finds the last request older than [`KEEP`].
//!
//! Without a set the driver wires a buffer's pages at the first command
//! buffer that binds it and lets the wiring lapse once no submitted command
//! buffer has named the buffer for a while: the first token of a process pays
//! the wiring of every weight, and so does the first token after a pass long
//! enough that its early layers' weights have lapsed, or after an idle gap.
//! A set requested at open wires everything once, where the caller waits for
//! the open, and a request renewed before work that follows a gap keeps it
//! from lapsing. The cost is that the set's bytes stay wired for the
//! stream's life.
//!
//! The set is the stream's ([`MetalStream::enable_residency`]) and its
//! buffers are registered where a session allocates them ([`Alloc`]): a
//! buffer a kernel binds that was made outside an [`Alloc`] is not in the
//! set, which `Session::residency` shows as bytes short of what the device
//! holds (the `r_resident` clause).

use core::time::Duration;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAllocation, MTLCommandQueue, MTLDevice, MTLResidencySet, MTLResidencySetDescriptor,
};

use bloomery_device::{DeviceError, Elem, Elems};

use crate::buffer::MetalBuffer;
use crate::handle::{Handle, ResidencySetHandle};

/// How long a set may go without a request before the next submit renews it:
/// well under the gap after which the driver lets wiring lapse.
pub const KEEP: Duration = Duration::from_millis(250);

/// What a stream's residency set holds and how often it was requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyStats {
    /// The buffers registered.
    pub buffers: usize,
    /// The bytes the set holds, as the driver counts them.
    pub bytes: u64,
    /// The requests made: the commits that asked for residency, and the
    /// renewals a submit made after [`KEEP`].
    pub requests: u64,
}

/// A stream's residency set.
pub struct Residency {
    set: Mutex<ResidencySetHandle>,
    queue: Handle<dyn MTLCommandQueue>,
    born: Instant,
    /// Nanoseconds from `born` to the last request.
    requested_ns: AtomicU64,
    requests: AtomicU64,
}

impl Residency {
    /// An empty set on `device`, attached to `queue`.
    pub(crate) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &Handle<dyn MTLCommandQueue>,
    ) -> Result<Self, DeviceError> {
        let descriptor = MTLResidencySetDescriptor::new();
        let set = device
            .newResidencySetWithDescriptor_error(&descriptor)
            .map_err(|e| {
                DeviceError::backend("residency set", e.localizedDescription().to_string())
            })?;
        queue.addResidencySet(&set);
        Ok(Self {
            set: Mutex::new(ResidencySetHandle(set)),
            queue: queue.clone(),
            born: Instant::now(),
            requested_ns: AtomicU64::new(0),
            requests: AtomicU64::new(0),
        })
    }

    fn set(&self) -> std::sync::MutexGuard<'_, ResidencySetHandle> {
        // Every change to the set is one driver call: a poisoned lock holds
        // a set as the driver last left it.
        self.set
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// `buffer` staged into the set; it is resident after the next
    /// [`Residency::commit`].
    fn add<T: Elem>(&self, buffer: &MetalBuffer<T>) {
        let allocation: &ProtocolObject<dyn MTLAllocation> =
            ProtocolObject::from_ref(buffer.metal());
        self.set().0.addAllocation(allocation);
    }

    /// The staged buffers made part of the set and the set requested
    /// resident. This returns once the driver has wired them, which is what
    /// an open waits for.
    pub fn commit(&self) {
        let set = self.set();
        set.0.commit();
        set.0.requestResidency();
        drop(set);
        self.note_request();
    }

    /// The set requested again when the last request is older than [`KEEP`];
    /// a submit calls it before it commits.
    pub(crate) fn keep(&self) {
        let now = self.nanos();
        let last = self.requested_ns.load(Ordering::Relaxed);
        if now.saturating_sub(last) < u64::try_from(KEEP.as_nanos()).unwrap_or(u64::MAX) {
            return;
        }
        self.set().0.requestResidency();
        self.note_request();
    }

    fn nanos(&self) -> u64 {
        u64::try_from(self.born.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    fn note_request(&self) {
        self.requested_ns.store(self.nanos(), Ordering::Relaxed);
        self.requests.fetch_add(1, Ordering::Relaxed);
    }

    /// What the set holds and how often it was requested.
    #[must_use]
    pub fn stats(&self) -> ResidencyStats {
        let set = self.set();
        ResidencyStats {
            buffers: set.0.allocationCount(),
            bytes: set.0.allocatedSize(),
            requests: self.requests.load(Ordering::Relaxed),
        }
    }
}

impl Drop for Residency {
    fn drop(&mut self) {
        let set = self.set();
        set.0.endResidency();
        self.queue.removeResidencySet(&set.0);
    }
}

/// Where a session allocates the buffers its kernels bind: the device, and
/// the stream's residency set each buffer is registered into.
#[derive(Clone, Copy)]
pub(crate) struct Alloc<'a> {
    device: &'a ProtocolObject<dyn MTLDevice>,
    residency: &'a Residency,
}

impl<'a> Alloc<'a> {
    pub(crate) fn new(device: &'a ProtocolObject<dyn MTLDevice>, residency: &'a Residency) -> Self {
        Self { device, residency }
    }

    /// A buffer of `n` elements, contents undefined, registered into the set
    /// (resident at the next [`Residency::commit`]).
    pub(crate) fn buffer<T: Elem>(&self, n: Elems) -> Result<MetalBuffer<T>, DeviceError> {
        let buffer = MetalBuffer::new(self.device, n)?;
        self.residency.add(&buffer);
        Ok(buffer)
    }
}
