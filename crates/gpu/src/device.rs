//! The card as the device contract's [`Device`] (`bloomery_device`): the
//! cuda-oxide calls the engine makes, behind the contract's types, so the
//! contract's clauses run on the card as they run on Metal
//! (`bloomery_device::conformance`, the `gate_gpu_device` bin). The engine
//! keeps calling `Gpu`'s own methods; nothing here is on its path.
//!
//! A buffer holds bytes (a `DeviceBuffer<u8>`) and is read as its element
//! type through the contract's windows. A window's raw is a device address
//! and the context it belongs to; a call refuses a window or a stream of
//! another context by name. Every wait is the engine's bounded one
//! ([`host::await_done`]) but the one a host-to-device copy owes the source it
//! borrows, which waits for the copy alone.

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut, Range};
use core::ptr::NonNull;
use std::sync::Arc;

use bloomery_device::{
    BUFFER_ALIGN, Buffer, Bytes, Census, Device, DeviceError, Elem, Elems, Event, Fault, FaultWord,
    HostPages, Shared, Stream, Window, WindowMut, buffer_bytes, check_len, check_window,
};
use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer, sys};
use model::placement::workstation::DeviceInfo;

use crate::graph::{self, MappedHost};
use crate::{FAULT_WORDS, Gpu, GpuError, fault, host};

/// The backend's name, as a record prints it.
const BACKEND: &str = "cuda";

/// A failure of this crate, as the contract's error.
fn cuda(e: impl Into<GpuError>) -> DeviceError {
    DeviceError::from(e.into())
}

/// A [`GpuError`] as the contract's error. A caller's geometry mistake is a
/// range refusal; a fault, raised or poisoning, is the contract's fault and
/// keeps the whole error as its source; anything else is a backend failure
/// whose source is the error. The text is the error's own in every case but
/// the backend's, which the backend's name leads.
impl From<GpuError> for DeviceError {
    fn from(e: GpuError) -> DeviceError {
        match e {
            GpuError::Shape { what, detail } => DeviceError::Range { what, detail },
            e @ (GpuError::Fault { fault, .. } | GpuError::Poisoned { fault, .. }) => {
                DeviceError::Fault {
                    fault,
                    source: Some(Box::new(e)),
                }
            }
            e => DeviceError::Backend {
                what: BACKEND,
                source: Box::new(e),
            },
        }
    }
}

/// The contract's error as a [`GpuError`]: one that came from this crate is
/// that error again, variant and all, so a fault that went through the
/// contract still poisons the model; a range refusal is a geometry mistake;
/// a fault with no source of this crate is a fault of `device`; any other is
/// a refused plan whose source keeps its text.
impl From<DeviceError> for GpuError {
    fn from(e: DeviceError) -> GpuError {
        match e {
            DeviceError::Range { what, detail } => GpuError::Shape { what, detail },
            DeviceError::Backend { what, source } => match source.downcast::<GpuError>() {
                Ok(e) => *e,
                Err(source) => GpuError::Plan { what, source },
            },
            DeviceError::Fault { fault, source } => {
                match source.map(|s| s.downcast::<GpuError>()) {
                    Some(Ok(e)) => *e,
                    None => GpuError::fault("device", fault),
                    Some(Err(source)) => GpuError::Fault {
                        what: "device",
                        fault,
                        behind: Some(Box::new(GpuError::Plan {
                            what: "device",
                            source,
                        })),
                    },
                }
            }
            DeviceError::Unsupported { what, detail } => GpuError::Plan {
                what,
                source: detail.into(),
            },
        }
    }
}

/// A window's buffer as the card addresses it: the buffer's device address
/// and the context it was allocated in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuRaw {
    ptr: sys::CUdeviceptr,
    ctx: usize,
}

/// A context's identity: its driver handle's address.
fn context_id(ctx: &CudaContext) -> usize {
    ctx.cu_ctx().addr()
}

/// A buffer the card holds: bytes, read as `T` through the contract's
/// windows.
pub struct GpuBuffer<T: Elem> {
    bytes: DeviceBuffer<u8>,
    len: Elems,
    ctx: usize,
    _t: PhantomData<T>,
}

impl<T: Elem> GpuBuffer<T> {
    fn raw(&self) -> GpuRaw {
        GpuRaw {
            ptr: self.bytes.cu_deviceptr(),
            ctx: self.ctx,
        }
    }
}

impl<T: Elem> Buffer<T> for GpuBuffer<T> {
    type Device = Gpu;

    fn len(&self) -> Elems {
        self.len
    }

    fn window(&self, at: Elems, len: Elems) -> Result<Window<'_, T, Gpu>, DeviceError> {
        check_window("buffer", self.len, at, len)?;
        // SAFETY: the raw is the buffer's own allocation of `self.len`
        // elements of `T`, which holds `at + len` of them (checked above), and
        // the borrow of the buffer outlives the window.
        Ok(unsafe { Window::from_raw(self.raw(), at, len) })
    }

    fn window_mut(&mut self, at: Elems, len: Elems) -> Result<WindowMut<'_, T, Gpu>, DeviceError> {
        check_window("buffer", self.len, at, len)?;
        // SAFETY: the raw is the buffer's own allocation of `self.len`
        // elements of `T`, which holds `at + len` of them (checked above),
        // and the exclusive borrow of the buffer outlives the window.
        Ok(unsafe { WindowMut::from_raw(self.raw(), at, len) })
    }
}

/// A buffer the host and the card share: a mapped pinned page the host
/// reaches through its [`HostPages`] and the card at its device address. Its
/// drop waits for every stream of its context first, so no work in flight
/// reaches a freed page.
pub struct GpuShared<T: Elem> {
    pages: HostPages<T>,
    page: MappedHost,
    ctx: Arc<CudaContext>,
}

impl<T: Elem> GpuShared<T> {
    fn raw(&self) -> GpuRaw {
        GpuRaw {
            ptr: self.page.dev_at(0),
            ctx: context_id(&self.ctx),
        }
    }
}

impl<T: Elem> Drop for GpuShared<T> {
    fn drop(&mut self) {
        // The page is freed after this body; an error stays on the context,
        // as cuda-core's own buffers keep theirs.
        self.ctx.record_err(crate::capsync::ctx_sync_in_drop(
            &self.ctx,
            "shared page drop",
        ));
    }
}

impl<T: Elem> Buffer<T> for GpuShared<T> {
    type Device = Gpu;

    fn len(&self) -> Elems {
        self.pages.len()
    }

    fn window(&self, at: Elems, len: Elems) -> Result<Window<'_, T, Gpu>, DeviceError> {
        check_window("buffer", self.len(), at, len)?;
        // SAFETY: the raw is the page's device address, where the page's
        // `self.len()` elements of `T` lie (`at + len` of them, checked
        // above), and the borrow of the buffer outlives the window.
        Ok(unsafe { Window::from_raw(self.raw(), at, len) })
    }

    fn window_mut(&mut self, at: Elems, len: Elems) -> Result<WindowMut<'_, T, Gpu>, DeviceError> {
        check_window("buffer", self.len(), at, len)?;
        // SAFETY: the raw is the page's device address, where the page's
        // `self.len()` elements of `T` lie (`at + len` of them, checked
        // above), and the exclusive borrow of the buffer outlives the
        // window.
        Ok(unsafe { WindowMut::from_raw(self.raw(), at, len) })
    }
}

impl<T: Elem> Shared<T> for GpuShared<T> {
    fn pages(&self) -> &HostPages<T> {
        &self.pages
    }

    unsafe fn host_mut(&mut self) -> &mut [T] {
        // SAFETY: the caller promises no device work in flight reads or
        // writes the page.
        unsafe { self.pages.as_mut_slice() }
    }
}

/// A stream of the card: the engine's, or another of its context
/// ([`Gpu::new_stream`]).
#[repr(transparent)]
pub struct GpuStream(Arc<CudaStream>);

impl GpuStream {
    /// The engine's stream as the contract's.
    fn of(stream: &Arc<CudaStream>) -> &GpuStream {
        // SAFETY: `GpuStream` is `repr(transparent)` over `Arc<CudaStream>`,
        // so the two references have one layout, and the borrow is the same.
        unsafe { &*core::ptr::from_ref(stream).cast::<GpuStream>() }
    }

    /// The stream a launch goes on.
    #[must_use]
    pub fn cuda(&self) -> &Arc<CudaStream> {
        &self.0
    }
}

impl Stream for GpuStream {
    type Event = GpuEvent;

    fn record(&self) -> Result<GpuEvent, DeviceError> {
        self.0.record_event(None).map(GpuEvent).map_err(cuda)
    }

    fn wait(&self, e: &GpuEvent) -> Result<(), DeviceError> {
        self.0.wait(&e.0).map_err(cuda)
    }

    fn sync(&self) -> Result<(), DeviceError> {
        Ok(host::await_done("Stream::sync", || self.0.query())?)
    }
}

/// A point of a stream's work.
pub struct GpuEvent(CudaEvent);

impl Event for GpuEvent {
    fn sync(&self) -> Result<(), DeviceError> {
        Ok(host::await_done("Event::sync", || self.0.query())?)
    }

    fn done(&self) -> Result<bool, DeviceError> {
        self.0.query().map_err(cuda)
    }
}

/// A read window as a launch binds it: a buffer handle over the window's
/// elements, for as long as the window's borrow, lent shared only.
pub struct Bound<'w, T>(crate::WindowMut<'w, T>);

impl<T> Deref for Bound<'_, T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &DeviceBuffer<T> {
        &self.0
    }
}

/// A write window as a launch binds it: [`Bound`], lent exclusively too.
pub struct BoundMut<'w, T>(crate::WindowMut<'w, T>);

impl<T> Deref for BoundMut<'_, T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &DeviceBuffer<T> {
        &self.0
    }
}

impl<T> DerefMut for BoundMut<'_, T> {
    fn deref_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.0
    }
}

impl Gpu {
    /// The device address and byte count of the window `raw` and `bytes`
    /// name; a window of another context's buffer is refused by name.
    fn address(
        &self,
        raw: GpuRaw,
        bytes: Range<Bytes>,
    ) -> Result<(sys::CUdeviceptr, usize), DeviceError> {
        if raw.ctx != context_id(&self.ctx) {
            return Err(DeviceError::Range {
                what: "window",
                detail: "a window of another context's buffer".to_string(),
            });
        }
        let off = u64::try_from(bytes.start.get()).map_err(|_| DeviceError::Range {
            what: "window",
            detail: format!("byte {} passes a device address", bytes.start.get()),
        })?;
        Ok((raw.ptr + off, bytes.end.get() - bytes.start.get()))
    }

    /// Another stream of this `Gpu`'s context, for work the contract orders
    /// against the engine's own; never made during a capture.
    pub fn new_stream(&self) -> Result<GpuStream, DeviceError> {
        self.ctx.new_stream().map(GpuStream).map_err(cuda)
    }

    /// A read window as a launch binds it. A window of another context's
    /// buffer is refused by name.
    pub fn bind<'w, T: Elem>(&self, w: Window<'w, T, Gpu>) -> Result<Bound<'w, T>, DeviceError> {
        let (ptr, _) = self.address(*w.raw(), w.byte_range()?)?;
        // SAFETY: the window's elements lie inside a live buffer of this
        // context (its cut checked them, and the context is checked above),
        // aligned for `T` (a buffer starts on `BUFFER_ALIGN` and a window on
        // a whole element), and the buffer stays borrowed, in place, for
        // `'w`, the handle's lifetime; the handle lends the elements shared
        // only, as the window does.
        Ok(Bound(unsafe {
            crate::WindowMut::from_raw(ptr, w.len().get(), &self.ctx)
        }))
    }

    /// A write window as a launch binds it. A window of another context's
    /// buffer is refused by name.
    pub fn bind_mut<'w, T: Elem>(
        &self,
        w: WindowMut<'w, T, Gpu>,
    ) -> Result<BoundMut<'w, T>, DeviceError> {
        let (ptr, _) = self.address(*w.raw(), w.byte_range()?)?;
        // SAFETY: the window's elements lie inside a live buffer of this
        // context (its cut checked them, and the context is checked above),
        // aligned for `T`, and the buffer stays borrowed, in place, for `'w`;
        // the window's exclusive borrow of it, which the handle keeps, makes
        // the handle the elements' only one.
        Ok(BoundMut(unsafe {
            crate::WindowMut::from_raw(ptr, w.len().get(), &self.ctx)
        }))
    }

    /// The fault allocation's first [`FAULT_WORDS`] words once `s`'s work so
    /// far is done: what a gate compares with the contract's host model,
    /// word for word.
    pub fn fault_words(&self, s: &GpuStream) -> Result<Vec<u32>, DeviceError> {
        self.own_stream(&s.0)?;
        host::await_done("Gpu::fault_words", || s.0.query())?;
        let mut words = vec![0u32; FAULT_WORDS];
        fault::read_words(&self.fault, 0, &mut words)?;
        Ok(words)
    }
}

impl Device for Gpu {
    type Buffer<T: Elem> = GpuBuffer<T>;
    type Shared<T: Elem> = GpuShared<T>;
    type Stream = GpuStream;
    type Event = GpuEvent;
    type Raw = GpuRaw;
    const BACKEND: &'static str = BACKEND;

    fn info(&self) -> Result<DeviceInfo, DeviceError> {
        Ok(crate::device_info(self.ctx.ordinal())?)
    }

    fn stream(&self) -> &GpuStream {
        GpuStream::of(&self.stream)
    }

    fn zeroed<T: Elem>(&self, s: &GpuStream, n: Elems) -> Result<GpuBuffer<T>, DeviceError> {
        self.own_stream(&s.0)?;
        let bytes = buffer_bytes::<T>(n)?;
        let buffer = DeviceBuffer::<u8>::zeroed(&s.0, bytes.get()).map_err(cuda)?;
        if !buffer.cu_deviceptr().is_multiple_of(BUFFER_ALIGN as u64) {
            return Err(DeviceError::backend(
                "buffer",
                format!("the allocation does not start at a multiple of {BUFFER_ALIGN} bytes"),
            ));
        }
        Ok(GpuBuffer {
            bytes: buffer,
            len: n,
            ctx: context_id(&self.ctx),
            _t: PhantomData,
        })
    }

    fn upload<T: Elem>(&self, s: &GpuStream, host: &[T]) -> Result<GpuBuffer<T>, DeviceError> {
        let mut buffer = self.zeroed::<T>(s, Elems(host.len()))?;
        self.write(s, buffer.whole_mut()?, host)?;
        Ok(buffer)
    }

    fn shared<T: Elem>(&self, n: Elems) -> Result<GpuShared<T>, DeviceError> {
        let bytes = buffer_bytes::<T>(n)?;
        let page = MappedHost::new(&self.ctx, bytes.get(), "Device::shared")?;
        let ptr = NonNull::new(page.host_at(0).cast::<T>()).ok_or_else(|| {
            DeviceError::backend("shared", "the driver mapped no host page".to_string())
        })?;
        if !ptr.as_ptr().addr().is_multiple_of(BUFFER_ALIGN) {
            return Err(DeviceError::backend(
                "shared",
                format!("the page does not start at a multiple of {BUFFER_ALIGN} bytes"),
            ));
        }
        // SAFETY: the page holds `bytes` (`n` elements of `T`) zeroed bytes
        // at `ptr`, aligned (checked above), mapped while the page lives; the
        // value keeps the page beside the pages and frees it after them, and
        // the host reaches the page only through these pages.
        let pages = unsafe { HostPages::new(ptr, n) };
        Ok(GpuShared {
            pages,
            page,
            ctx: Arc::clone(&self.ctx),
        })
    }

    fn read<T: Elem>(
        &self,
        s: &GpuStream,
        from: Window<'_, T, Gpu>,
        into: &mut [T],
    ) -> Result<(), DeviceError> {
        check_len("read", into.len(), from.len())?;
        self.own_stream(&s.0)?;
        let (ptr, bytes) = self.address(*from.raw(), from.byte_range()?)?;
        host::await_done("Device::read", || s.0.query())?;
        self.ctx.bind_to_thread().map_err(cuda)?;
        // SAFETY: the source is the window's `bytes` bytes of a live buffer
        // of this context (checked above), whose writers on `s` are done
        // (waited for above); the destination is `into`, exactly `bytes`
        // bytes (its length checked above), borrowed past the synchronous
        // copy; the context is current on this thread (bound above).
        let rc = unsafe { sys::cuMemcpyDtoH_v2(into.as_mut_ptr().cast(), ptr, bytes) };
        graph::cu(rc, "cuMemcpyDtoH_v2 (Device::read)")?;
        Ok(())
    }

    fn write<T: Elem>(
        &self,
        s: &GpuStream,
        to: WindowMut<'_, T, Gpu>,
        from: &[T],
    ) -> Result<(), DeviceError> {
        check_len("write", from.len(), to.len())?;
        self.own_stream(&s.0)?;
        let (ptr, bytes) = self.address(*to.raw(), to.byte_range()?)?;
        host::await_done("Device::write", || s.0.query())?;
        self.ctx.bind_to_thread().map_err(cuda)?;
        // SAFETY: the destination is the window's `bytes` bytes of a live
        // buffer of this context (checked above); the source is `from`,
        // exactly `bytes` bytes (its length checked above), borrowed until
        // the stream wait below, the call's only step after the enqueue; the
        // context is current on this thread (bound above).
        let rc =
            unsafe { sys::cuMemcpyHtoDAsync_v2(ptr, from.as_ptr().cast(), bytes, s.0.cu_stream()) };
        graph::cu(rc, "cuMemcpyHtoDAsync_v2 (Device::write)")?;
        // `s`'s work before the copy is done (waited for above), so this
        // wait is the copy's alone. It is the stream's own wait, with no
        // fallible setup between the enqueue and it: `from` is borrowed only
        // for the call, so no return may come before the copy has landed.
        s.0.synchronize().map_err(cuda)
    }

    fn copy<T: Elem>(
        &self,
        s: &GpuStream,
        from: Window<'_, T, Gpu>,
        to: WindowMut<'_, T, Gpu>,
    ) -> Result<(), DeviceError> {
        check_len("copy", from.len().get(), to.len())?;
        self.own_stream(&s.0)?;
        let (src, bytes) = self.address(*from.raw(), from.byte_range()?)?;
        let (dst, _) = self.address(*to.raw(), to.byte_range()?)?;
        self.ctx.bind_to_thread().map_err(cuda)?;
        // SAFETY: both ranges are `bytes` bytes of live buffers of this
        // context (checked above), apart (the windows' borrows keep them
        // so); a buffer's drop waits for its context's streams before it
        // frees, so the copy never meets a freed one; the context is current
        // on this thread (bound above).
        let rc = unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, s.0.cu_stream()) };
        graph::cu(rc, "cuMemcpyDtoDAsync_v2 (Device::copy)")?;
        Ok(())
    }

    fn fault(&self) -> &dyn FaultWord<GpuStream> {
        self
    }
}

/// The card's fault word through the contract: the read and the clear of
/// [`Gpu::fault`] and [`Gpu::clear_fault`], ordered on the stream given.
impl FaultWord<GpuStream> for Gpu {
    fn read(&self, s: &GpuStream) -> Result<Option<Fault>, DeviceError> {
        self.own_stream(&s.0)?;
        host::await_done("FaultWord::read", || s.0.query())?;
        Ok(fault::read(&self.fault)?)
    }

    fn clear(&self, s: &GpuStream) -> Result<(), DeviceError> {
        Ok(self.clear_fault_on(&s.0)?)
    }
}

/// The visible cards in the contract's terms: [`crate::census`].
pub struct GpuCensus;

impl Census for GpuCensus {
    fn census() -> Result<Vec<DeviceInfo>, DeviceError> {
        Ok(crate::census()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An error of this crate goes through the contract and comes back the
    /// same error: its text, and the variant the engine acts on.
    #[test]
    fn an_error_through_the_contract_comes_back_as_it_was() {
        let fault = Fault::at(4, bloomery_device::FaultSite::ExpertId);
        let errors = [
            GpuError::shape("cut", "byte 3 is not aligned"),
            GpuError::fault("step", fault),
            GpuError::Fault {
                what: "step",
                fault,
                behind: Some(Box::new(GpuError::shape("launch", "a later refusal"))),
            },
            GpuError::Poisoned {
                what: "step",
                fault,
            },
            GpuError::State {
                what: "head",
                missing: "no output head",
            },
        ];
        for e in errors {
            let text = e.to_string();
            let kind = std::mem::discriminant(&e);
            let device = DeviceError::from(e);
            match &device {
                DeviceError::Range { .. } | DeviceError::Fault { .. } => {
                    assert_eq!(device.to_string(), text, "{device:?}");
                }
                DeviceError::Backend { .. } => {
                    assert_eq!(device.to_string(), format!("cuda: {text}"), "{device:?}");
                }
                DeviceError::Unsupported { .. } => panic!("{device:?}"),
            }
            let back = GpuError::from(device);
            assert_eq!(back.to_string(), text);
            assert_eq!(std::mem::discriminant(&back), kind, "{back:?}");
        }
    }

    /// A fault the contract made on its own is a fault of the device, which
    /// poisons as one the card raised.
    #[test]
    fn a_contract_fault_is_a_gpu_fault() {
        let fault = Fault::at(2, bloomery_device::FaultSite::Logit);
        match GpuError::from(DeviceError::fault(fault)) {
            GpuError::Fault {
                what: "device",
                fault: f,
                behind: None,
            } => assert_eq!(f, fault),
            other => panic!("{other:?}"),
        }
        match GpuError::from(DeviceError::Range {
            what: "window",
            detail: "past the end".to_string(),
        }) {
            GpuError::Shape { what: "window", .. } => {}
            other => panic!("{other:?}"),
        }
    }
}
