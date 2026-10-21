//! The Metal buffers: the device's own, and the one the host and the
//! device share.
//!
//! Every buffer of this backend is `StorageModeShared`: the Mac's memory is
//! one pool, and a buffer's pages are the host's pages. The `Shared` buffer
//! is the one the host reads and writes in place; the plain buffer is the
//! one a kernel binds.

use core::ptr::NonNull;

use objc2::ffi::NSUInteger;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

use bloomery_device::{
    Buffer, Bytes, DeviceError, Elem, Elems, Shared, Stream, Window, WindowMut, check_window,
};

use crate::device::{MetalDevice, buffer_options};
use crate::handle::Handle;
use crate::pages::HostPages;
use crate::stream::MetalStream;

/// A buffer as a Metal kernel binds it: the buffer and the byte offset of its
/// own start inside its backing store. The window that carries it keeps the
/// buffer borrowed.
#[derive(Clone, Copy)]
pub struct MetalRaw {
    buffer: NonNull<ProtocolObject<dyn MTLBuffer>>,
    base: Bytes,
}

// SAFETY: the pointer addresses an `MTLBuffer`, a resource Apple documents as
// usable from several threads; the window that holds the raw keeps the buffer
// borrowed, so the pointer stays valid wherever the window goes.
unsafe impl Send for MetalRaw {}
// SAFETY: as above.
unsafe impl Sync for MetalRaw {}

impl MetalRaw {
    /// The byte offset of the buffer's own start inside its backing store.
    pub(crate) fn base(&self) -> Bytes {
        self.base
    }

    /// The buffer, for as long as the window `self` was lent by is borrowed.
    ///
    /// # Safety
    /// `self` is borrowed from a window of a live buffer, so the buffer lives
    /// as long as `&self` is held.
    pub(crate) unsafe fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        // SAFETY: the caller's contract: the buffer outlives this borrow.
        unsafe { self.buffer.as_ref() }
    }
}

/// A buffer the device holds.
pub struct MetalBuffer<T: Elem> {
    buffer: Handle<dyn MTLBuffer>,
    pages: HostPages<T>,
}

impl<T: Elem> MetalBuffer<T> {
    /// A buffer of `n` elements, its contents undefined.
    pub(crate) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        n: Elems,
    ) -> Result<Self, DeviceError> {
        let bytes = n.bytes_of::<T>()?;
        if bytes == Bytes::ZERO {
            return Err(DeviceError::Range {
                what: "buffer",
                detail: "a buffer of no elements is refused".to_string(),
            });
        }
        let length = NSUInteger::try_from(bytes.get()).map_err(|_| DeviceError::Range {
            what: "buffer",
            detail: format!("{} bytes are more than a Metal buffer holds", bytes.get()),
        })?;
        let buffer = device
            .newBufferWithLength_options(length, buffer_options())
            .ok_or_else(|| {
                DeviceError::backend("buffer", "the device made no buffer".to_string())
            })?;
        let ptr = buffer.contents().cast::<T>();
        if !ptr.is_aligned() {
            return Err(DeviceError::backend(
                "buffer",
                format!(
                    "the buffer's pages are not aligned for {}-byte elements",
                    core::mem::size_of::<T>()
                ),
            ));
        }
        // SAFETY: a shared buffer's `contents()` addresses its `bytes` bytes
        // of pages, aligned (checked above), for the buffer's whole life; the
        // buffer is kept in `Self` beside the pages, and every host access to
        // them goes through this value.
        let pages = unsafe { HostPages::new(ptr, n) };
        Ok(Self {
            buffer: Handle(buffer),
            pages,
        })
    }

    /// The buffer's own Metal object.
    pub(crate) fn metal(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// The buffer's contents, zeroed.
    pub(crate) fn zero(&mut self) {
        self.pages.zero();
    }

    /// The buffer's contents, `host`'s elements; any other length is refused
    /// by name.
    pub(crate) fn write(&mut self, host: &[T]) -> Result<(), DeviceError> {
        self.pages.write(host)
    }

    /// The pages the host sees.
    pub(crate) fn pages(&self) -> &HostPages<T> {
        &self.pages
    }

    /// The window's raw: the buffer and its base offset.
    fn raw(&self) -> MetalRaw {
        MetalRaw {
            buffer: NonNull::from(&*self.buffer),
            base: Bytes::ZERO,
        }
    }
}

impl<T: Elem> Buffer<T> for MetalBuffer<T> {
    type Device = MetalDevice;

    fn len(&self) -> Elems {
        self.pages.len()
    }

    fn window(&self, at: Elems, len: Elems) -> Result<Window<'_, T, MetalDevice>, DeviceError> {
        check_window("buffer", self.len(), at, len)?;
        // SAFETY: the raw is the buffer's own pages, which hold `at + len`
        // elements of `T` (checked above), and the borrow of the buffer
        // outlives the window.
        Ok(unsafe { Window::from_raw(self.raw(), at, len) })
    }

    fn window_mut(
        &mut self,
        at: Elems,
        len: Elems,
    ) -> Result<WindowMut<'_, T, MetalDevice>, DeviceError> {
        check_window("buffer", self.len(), at, len)?;
        // SAFETY: as in `window`, and the exclusive borrow of the buffer
        // outlives the window.
        Ok(unsafe { WindowMut::from_raw(self.raw(), at, len) })
    }

    fn read(&self, s: &MetalStream, into: &mut [T]) -> Result<(), DeviceError> {
        s.sync()?;
        self.pages.read(into)
    }
}

/// A buffer the host and the device share: the same pages. On Metal every
/// buffer is shared, so this is a buffer that also lends its pages.
pub struct MetalShared<T: Elem>(MetalBuffer<T>);

impl<T: Elem> MetalShared<T> {
    /// A shared buffer of `n` elements, zeroed.
    pub(crate) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        n: Elems,
    ) -> Result<Self, DeviceError> {
        let mut buffer = MetalBuffer::new(device, n)?;
        buffer.zero();
        Ok(Self(buffer))
    }
}

impl<T: Elem> Buffer<T> for MetalShared<T> {
    type Device = MetalDevice;

    fn len(&self) -> Elems {
        self.0.len()
    }

    fn window(&self, at: Elems, len: Elems) -> Result<Window<'_, T, MetalDevice>, DeviceError> {
        self.0.window(at, len)
    }

    fn window_mut(
        &mut self,
        at: Elems,
        len: Elems,
    ) -> Result<WindowMut<'_, T, MetalDevice>, DeviceError> {
        self.0.window_mut(at, len)
    }

    fn read(&self, s: &MetalStream, into: &mut [T]) -> Result<(), DeviceError> {
        self.0.read(s, into)
    }
}

impl<T: Elem> Shared<T> for MetalShared<T> {
    fn load(&self, i: usize) -> Result<T, DeviceError> {
        self.0.pages.load(i)
    }

    fn store(&mut self, i: usize, v: T) -> Result<(), DeviceError> {
        self.0.pages.store(i, v)
    }

    unsafe fn host(&self) -> &[T] {
        // SAFETY: the caller promises no device work in flight writes the
        // buffer.
        unsafe { self.0.pages.as_slice() }
    }

    unsafe fn host_mut(&mut self) -> &mut [T] {
        // SAFETY: the caller promises no device work in flight reads or
        // writes the buffer.
        unsafe { self.0.pages.as_mut_slice() }
    }
}
