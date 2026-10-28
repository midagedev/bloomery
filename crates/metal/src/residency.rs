//! The residency of a stream's buffers: one Metal residency set holding every
//! buffer a session's kernels bind, requested resident at open and again
//! whenever a submit finds the last request [`KEEP`] old or more.
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
//! holds (the `r_resident` clause). An [`Alloc`] stages its buffers
//! ([`Staged`]) and they join the set only when the open that made them
//! commits: an open that fails leaves the set as it was.

use core::cell::RefCell;
use core::time::Duration;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAllocation, MTLBuffer, MTLCommandQueue, MTLDevice, MTLResidencySet,
    MTLResidencySetDescriptor,
};

use bloomery_device::{DeviceError, Elem, Elems};

use crate::buffer::MetalBuffer;
use crate::handle::{Handle, ResidencySetHandle};
use crate::stream::mark;

/// How long a set may go without a request before the next submit renews it:
/// well under the gap after which the driver lets wiring lapse.
pub const KEEP: Duration = Duration::from_millis(250);

/// `renewed_gap_min_ns` before any submit has renewed.
const NEVER: u64 = u64::MAX;

/// What a stream's residency set holds, how often it was requested, and the
/// gaps its submits decided at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyStats {
    /// The buffers registered.
    pub buffers: usize,
    /// The bytes the set holds, as the driver counts them.
    pub bytes: u64,
    /// The requests made: the commits that asked for residency, and the
    /// renewals a submit made after [`KEEP`].
    pub requests: u64,
    /// The submits that found the last request younger than [`KEEP`] and
    /// left it.
    pub held: u64,
    /// The longest gap since the last request a submit left it at: the rule
    /// keeps it under [`KEEP`]. Zero before the first held submit.
    pub held_gap_max: Duration,
    /// The shortest gap since the last request a submit renewed it at: the
    /// rule keeps it at [`KEEP`] or more. `None` before the first renewal.
    pub renewed_gap_min: Option<Duration>,
}

/// A stream's residency set.
pub struct Residency {
    set: Mutex<ResidencySetHandle>,
    queue: Handle<dyn MTLCommandQueue>,
    born: Instant,
    /// Nanoseconds from `born` to the last request.
    requested_ns: AtomicU64,
    requests: AtomicU64,
    held: AtomicU64,
    /// Nanoseconds: the largest gap a submit held at.
    held_gap_max_ns: AtomicU64,
    /// Nanoseconds: the smallest gap a submit renewed at, [`NEVER`] before one.
    renewed_gap_min_ns: AtomicU64,
}

impl Residency {
    /// An empty set on `device`, attached to `queue`.
    pub(crate) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &Handle<dyn MTLCommandQueue>,
    ) -> Result<Self, DeviceError> {
        let descriptor = MTLResidencySetDescriptor::new();
        mark("set", format_args!("newResidencySet + addResidencySet"));
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
            held: AtomicU64::new(0),
            held_gap_max_ns: AtomicU64::new(0),
            renewed_gap_min_ns: AtomicU64::new(NEVER),
        })
    }

    fn set(&self) -> std::sync::MutexGuard<'_, ResidencySetHandle> {
        // Every change to the set is one driver call: a poisoned lock holds
        // a set as the driver last left it.
        self.set
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// An empty staging list for an open's buffers.
    pub(crate) fn stage(&self) -> Staged<'_> {
        Staged {
            residency: self,
            buffers: RefCell::new(Vec::new()),
        }
    }

    /// `buffers` made part of the set and the set requested resident. This
    /// returns once the driver has wired them, which is what an open waits
    /// for.
    fn commit(&self, buffers: &[Retained<ProtocolObject<dyn MTLBuffer>>]) {
        let set = self.set();
        mark("set", format_args!("commit {} allocations", buffers.len()));
        for buffer in buffers {
            let allocation: &ProtocolObject<dyn MTLAllocation> =
                ProtocolObject::from_ref(&**buffer);
            set.0.addAllocation(allocation);
        }
        set.0.commit();
        set.0.requestResidency();
        drop(set);
        self.note_request();
    }

    /// The set requested again when the last request is at least [`KEEP`]
    /// old; a submit calls it before it commits. Each decision is recorded
    /// against the gap it measured, on the clock the rule reads
    /// ([`ResidencyStats::held_gap_max`], [`ResidencyStats::renewed_gap_min`]).
    pub(crate) fn keep(&self) {
        let now = self.nanos();
        let last = self.requested_ns.load(Ordering::Relaxed);
        let gap = now.saturating_sub(last);
        if gap < u64::try_from(KEEP.as_nanos()).unwrap_or(u64::MAX) {
            self.held.fetch_add(1, Ordering::Relaxed);
            self.held_gap_max_ns.fetch_max(gap, Ordering::Relaxed);
            return;
        }
        self.renewed_gap_min_ns.fetch_min(gap, Ordering::Relaxed);
        mark("set", format_args!("renew request at gap {gap} ns"));
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

    /// What the set holds, how often it was requested, and the gaps its
    /// submits decided at.
    #[must_use]
    pub fn stats(&self) -> ResidencyStats {
        let set = self.set();
        ResidencyStats {
            buffers: set.0.allocationCount(),
            bytes: set.0.allocatedSize(),
            requests: self.requests.load(Ordering::Relaxed),
            held: self.held.load(Ordering::Relaxed),
            held_gap_max: Duration::from_nanos(self.held_gap_max_ns.load(Ordering::Relaxed)),
            renewed_gap_min: match self.renewed_gap_min_ns.load(Ordering::Relaxed) {
                NEVER => None,
                ns => Some(Duration::from_nanos(ns)),
            },
        }
    }
}

impl Drop for Residency {
    fn drop(&mut self) {
        let set = self.set();
        mark("set", format_args!("endResidency + removeResidencySet"));
        set.0.endResidency();
        self.queue.removeResidencySet(&set.0);
    }
}

/// The buffers an open has made, held back from the set until the open
/// succeeds: [`Staged::commit`] adds them and requests the set resident, and
/// a list dropped uncommitted adds nothing.
pub(crate) struct Staged<'a> {
    residency: &'a Residency,
    buffers: RefCell<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
}

impl Staged<'_> {
    /// The staged buffers made part of the set, and the set requested
    /// resident.
    pub(crate) fn commit(self) {
        self.residency.commit(&self.buffers.into_inner());
    }
}

/// Where a session allocates the buffers its kernels bind: the device, and
/// the staging list of the stream's residency set each buffer joins.
#[derive(Clone, Copy)]
pub(crate) struct Alloc<'a> {
    device: &'a ProtocolObject<dyn MTLDevice>,
    staged: &'a Staged<'a>,
}

impl<'a> Alloc<'a> {
    pub(crate) fn new(device: &'a ProtocolObject<dyn MTLDevice>, staged: &'a Staged<'a>) -> Self {
        Self { device, staged }
    }

    /// A buffer of `n` elements, contents undefined, staged for the set
    /// (resident once the open's [`Staged::commit`] runs).
    pub(crate) fn buffer<T: Elem>(&self, n: Elems) -> Result<MetalBuffer<T>, DeviceError> {
        let buffer = MetalBuffer::new(self.device, n)?;
        self.staged
            .buffers
            .borrow_mut()
            .push(buffer.metal().retain());
        Ok(buffer)
    }
}
