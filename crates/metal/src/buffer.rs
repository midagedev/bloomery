//! The Metal buffers: the device's own, and the one the host and the
//! device share.
//!
//! Every buffer of this backend is `StorageModeShared`: the Mac's memory is
//! one pool, and a buffer's pages are the host's pages. The `Shared` buffer
//! is the one the host reads and writes in place; the plain buffer is the
//! one a kernel binds. Either one's pages are reached on the host only through
//! the contract's [`HostPages`].

use core::ptr::NonNull;

use objc2::ffi::NSUInteger;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice};

use bloomery_device::{
    BUFFER_ALIGN, Buffer, Bytes, DeviceError, Elem, Elems, HostPages, Shared, Window, WindowMut,
    buffer_bytes, check_window,
};

use crate::device::{MetalDevice, buffer_options};
use crate::handle::Handle;

/// A buffer as a Metal kernel binds it: the buffer itself. A window's byte
/// offset is the offset into this buffer, since every window is cut from the
/// buffer it names; the window that carries the raw keeps the buffer borrowed.
#[derive(Clone, Copy)]
pub struct MetalRaw {
    buffer: NonNull<ProtocolObject<dyn MTLBuffer>>,
}

// SAFETY: the pointer addresses an `MTLBuffer`, a resource Apple documents as
// usable from several threads; the window that holds the raw keeps the buffer
// borrowed, so the pointer stays valid wherever the window goes.
unsafe impl Send for MetalRaw {}
// SAFETY: the same pointer, to the same documented-thread-safe `MTLBuffer`; a
// shared reference to it grants nothing a copy of it does not.
unsafe impl Sync for MetalRaw {}

impl MetalRaw {
    /// The buffer, for as long as the window `self` was lent by is borrowed.
    ///
    /// # Safety
    /// `self` is borrowed from a window of a live buffer, so the buffer lives
    /// as long as `&self` is held.
    pub(crate) unsafe fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        // SAFETY: the caller's contract: the buffer outlives this borrow.
        unsafe { self.buffer.as_ref() }
    }

    /// Whether `self` and `other` name the same Metal buffer.
    #[must_use]
    pub(crate) fn same_buffer(self, other: Self) -> bool {
        self.buffer == other.buffer
    }

    /// The host pages of `len` elements of `T` at byte `at` of the buffer:
    /// a window's own elements, for a copy to or from the host.
    ///
    /// # Safety
    /// `self` is lent by a window of a live buffer whose elements are those
    /// `len` at byte `at` (the window's range), borrowed for as long as the
    /// pages live, and the pages keep the rest of [`HostPages::new`]'s
    /// contract: a read window's pages are only read, and a write window's
    /// are the only host access to its elements.
    pub(crate) unsafe fn pages<T: Elem>(
        &self,
        at: Bytes,
        len: Elems,
    ) -> Result<HostPages<T>, DeviceError> {
        let offset = at.get();
        // SAFETY: the caller's contract: the buffer lives while `&self` is
        // held.
        let contents = unsafe { self.buffer() }.contents().cast::<u8>();
        // SAFETY: the window's elements lie inside the buffer's bytes (the
        // caller's contract), so the offset stays inside the allocation
        // `contents()` addresses.
        let ptr = unsafe { contents.add(offset) }.cast::<T>();
        if !ptr.is_aligned() {
            return Err(DeviceError::Range {
                what: "window",
                detail: format!(
                    "byte {offset} is not aligned for {}-byte elements",
                    size_of::<T>()
                ),
            });
        }
        // SAFETY: `len` aligned elements of the window lie at `ptr` and stay
        // mapped while the window's borrow holds; the caller keeps the rest
        // of `new`'s contract.
        Ok(unsafe { HostPages::new(ptr, len) })
    }
}

/// A buffer the device holds.
pub struct MetalBuffer<T: Elem> {
    buffer: Handle<dyn MTLBuffer>,
    pages: HostPages<T>,
}

impl<T: Elem> MetalBuffer<T> {
    /// A buffer of `n` elements, its contents undefined. A buffer of no
    /// elements is refused by name.
    pub(crate) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        n: Elems,
    ) -> Result<Self, DeviceError> {
        let bytes = buffer_bytes::<T>(n)?;
        let length = NSUInteger::try_from(bytes.get()).map_err(|_| DeviceError::Range {
            what: "buffer",
            detail: format!("{} bytes are more than a Metal buffer holds", bytes.get()),
        })?;
        let buffer = device
            .newBufferWithLength_options(length, buffer_options())
            .ok_or_else(|| {
                DeviceError::backend("buffer", "the device made no buffer".to_string())
            })?;
        #[cfg(feature = "encode-trace")]
        crate::trace::buffer_made(core::ptr::from_ref(&*buffer).addr(), bytes.get());
        let ptr = buffer.contents().cast::<T>();
        if !ptr.as_ptr().addr().is_multiple_of(BUFFER_ALIGN) {
            return Err(DeviceError::backend(
                "buffer",
                format!("the buffer's pages do not start at a multiple of {BUFFER_ALIGN} bytes"),
            ));
        }
        // SAFETY: a shared buffer's `contents()` addresses its `bytes` bytes
        // of pages, aligned for every element (checked above), for the
        // buffer's whole life; the buffer is kept in `Self` beside the pages,
        // and every host access to them goes through this value or a window's
        // pages, which borrow it.
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

    /// Elements `at ..` of the buffer, `from`'s; a range past the end is
    /// refused by name.
    pub(crate) fn write_at(&mut self, at: Elems, from: &[T]) -> Result<(), DeviceError> {
        self.pages.write_at(at, from)
    }

    /// The pages the host sees.
    pub(crate) fn pages(&self) -> &HostPages<T> {
        &self.pages
    }

    /// The window's raw: the buffer.
    fn raw(&self) -> MetalRaw {
        MetalRaw {
            buffer: NonNull::from(&*self.buffer),
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
        // SAFETY: the raw is the buffer's own pages, which hold `at + len`
        // elements of `T` (checked above), and the exclusive borrow of the
        // buffer outlives the window.
        Ok(unsafe { WindowMut::from_raw(self.raw(), at, len) })
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
}

impl<T: Elem> Shared<T> for MetalShared<T> {
    fn pages(&self) -> &HostPages<T> {
        self.0.pages()
    }

    unsafe fn host_mut(&mut self) -> &mut [T] {
        // SAFETY: the caller promises no device work in flight reads or
        // writes the buffer.
        unsafe { self.0.pages.as_mut_slice() }
    }
}
