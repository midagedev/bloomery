//! Metal objects held across threads.
//!
//! The shared event crosses threads as it comes: the bindings declare its
//! `MTLEvent` protocol `Send + Sync`. The other objects used from several
//! threads at once are wrapped by [`Handle`], whose [`ThreadSafe`] list is
//! Apple's list of thread-safe Metal objects. A command buffer is not among
//! them: it may move to another thread between uses, never be used by two at
//! once, so it is wrapped by [`CommandBufferHandle`], which is `Send` and not
//! `Sync`, and kept behind a lock. An encoder is not wrapped at all.

use core::ops::Deref;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLLibrary,
};

/// A Metal protocol that Apple documents as safe to use from several threads
/// at once.
///
/// # Safety
/// Apple's thread-safety list for Metal names the protocol: the device, the
/// command queue, the resources (buffers) and the compiled-code objects
/// (libraries and pipeline states).
pub(crate) unsafe trait ThreadSafe {}

// SAFETY: the device is on Apple's list of objects usable from any thread.
unsafe impl ThreadSafe for dyn MTLDevice {}
// SAFETY: the command queue is on Apple's list.
unsafe impl ThreadSafe for dyn MTLCommandQueue {}
// SAFETY: a buffer is a resource, on Apple's list.
unsafe impl ThreadSafe for dyn MTLBuffer {}
// SAFETY: a pipeline state is compiled code, on Apple's list.
unsafe impl ThreadSafe for dyn MTLComputePipelineState {}
// SAFETY: a library is compiled code, on Apple's list.
unsafe impl ThreadSafe for dyn MTLLibrary {}

/// A Metal object of a thread-safe protocol, shared across threads.
pub(crate) struct Handle<T: ?Sized + ThreadSafe>(pub(crate) Retained<ProtocolObject<T>>);

// SAFETY: `T: ThreadSafe` states the protocol is documented thread-safe, so the
// object may be released and used from any thread.
unsafe impl<T: ?Sized + ThreadSafe> Send for Handle<T> {}
// SAFETY: `T: ThreadSafe` states the protocol's methods may be called from
// several threads at once, which is what sharing the handle by reference does.
unsafe impl<T: ?Sized + ThreadSafe> Sync for Handle<T> {}

impl<T: ?Sized + ThreadSafe> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: ?Sized + ThreadSafe> Deref for Handle<T> {
    type Target = ProtocolObject<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A committed command buffer, moved between threads with its stream.
pub(crate) struct CommandBufferHandle(pub(crate) Retained<ProtocolObject<dyn MTLCommandBuffer>>);

// SAFETY: a command buffer is not documented thread-safe, but it may be
// handed to another thread when the first stops using it: the stream holds
// it behind its lock, so one thread uses it at a time.
unsafe impl Send for CommandBufferHandle {}
