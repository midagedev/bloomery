//! Device-resident buffers with the shapes the kernels' launch contracts
//! speak in, allocated once and reused across steps (docs/gpu-design.md
//! decision 4: nothing is allocated inside `step`, so a captured graph can
//! replay against fixed addresses).

use crate::GpuError;
use crate::graph::cu;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, sys};
use std::marker::PhantomData;
use std::mem::{ManuallyDrop, align_of, size_of};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// Check the base alignment of a buffer read or written as vectors of `V`.
pub(crate) fn check_alignment<V, T>(
    what: &'static str,
    name: &'static str,
    buf: &DeviceBuffer<T>,
) -> Result<(), GpuError> {
    let align = align_of::<V>() as u64;
    let at = buf.cu_deviceptr();
    if !at.is_multiple_of(align) {
        return Err(GpuError::shape(
            what,
            format!("{name} at {at:#x} is not {align}-byte aligned"),
        ));
    }
    Ok(())
}

/// A non-owning window of `len` `T` at device address `ptr` in `ctx`. The
/// launches read it exactly as they read a buffer of their own;
/// `ManuallyDrop` keeps it from ever freeing memory it does not own.
///
/// # Safety
///
/// - `ptr .. ptr + len * size_of::<T>()` must be memory the device reaches in
///   `ctx` — a `cuMemAlloc` allocation or a host-mapped one — aligned for `T`.
/// - That memory must outlive the window and stay in place: a captured graph
///   bakes the address in.
pub unsafe fn window<T>(
    ptr: sys::CUdeviceptr,
    len: usize,
    ctx: &Arc<CudaContext>,
) -> ManuallyDrop<DeviceBuffer<T>> {
    // SAFETY: the range is the caller's contract. `from_raw_parts` asks for a
    // `cuMemAlloc` pointer because its drop frees one; a window is never
    // dropped, so the only uses left are the address and the length.
    ManuallyDrop::new(unsafe { DeviceBuffer::from_raw_parts(ptr, len, ctx.clone()) })
}

/// A non-owning window of `len` `T` at `byte_off` of a live parent buffer,
/// borrowed for the window's lifetime: the launches read it exactly as they
/// read a buffer of its own, and dropping it frees nothing — the parent
/// frees the allocation after the window is gone.
///
/// [`Window::of`] checks the span and the offset, so a window cannot leave
/// its parent or start misaligned for `T`. This is the shared form, over a
/// parent borrowed shared: it derefs to `&DeviceBuffer<T>` only. A window to
/// write through is a [`WindowMut`], from [`WindowMut::of_mut`], which takes
/// the parent exclusively.
pub struct Window<'a, T> {
    buf: ManuallyDrop<DeviceBuffer<T>>,
    _parent: PhantomData<&'a ()>,
}

impl<T> Deref for Window<'_, T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &DeviceBuffer<T> {
        &self.buf
    }
}

impl<T> Drop for Window<'_, T> {
    fn drop(&mut self) {
        // SAFETY: `buf` is taken once, here, and never read again.
        let buf = unsafe { ManuallyDrop::take(&mut self.buf) };
        // The window owns no memory: its raw parts are dropped, the context
        // handle with them, and nothing is freed.
        drop(buf.into_raw_parts());
    }
}

impl<T> Window<'_, T> {
    /// `len` `T` of `parent` from byte `byte_off`, shared with the parent's
    /// other readers.
    pub fn of<P>(
        parent: &DeviceBuffer<P>,
        byte_off: usize,
        len: usize,
    ) -> Result<Window<'_, T>, GpuError> {
        Ok(Window {
            buf: cut(parent, byte_off, len, "Window::of")?,
            _parent: PhantomData,
        })
    }

    /// The window's buffer as a plain non-owning handle, past its borrow: for
    /// a parent that moves beside its windows rather than staying borrowed
    /// (an arena cut at load).
    ///
    /// # Safety
    ///
    /// - The parent must stay alive and in place — the same allocation, not
    ///   freed — for as long as the handle is used.
    /// - The handle's owner owes it, exactly once, the release this window's
    ///   `Drop` would have done: dropping the buffer's raw parts, which
    ///   releases the context handle and frees nothing.
    pub(crate) unsafe fn into_handle(mut self) -> ManuallyDrop<DeviceBuffer<T>> {
        // SAFETY: `buf` is taken once, here; `self` is forgotten right below,
        // so its drop cannot take it again.
        let buf = ManuallyDrop::new(unsafe { ManuallyDrop::take(&mut self.buf) });
        std::mem::forget(self);
        buf
    }
}

/// A [`Window`] over a parent held exclusively: the form a staging copy or a
/// launch writes through. Holding the parent's exclusive borrow for the
/// window's lifetime is what makes handing out `&mut DeviceBuffer<T>` sound;
/// the checks and the release on drop are [`Window::of`]'s.
pub struct WindowMut<'a, T> {
    window: Window<'a, T>,
}

impl<T> Deref for WindowMut<'_, T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &DeviceBuffer<T> {
        &self.window.buf
    }
}

impl<T> DerefMut for WindowMut<'_, T> {
    fn deref_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.window.buf
    }
}

impl<T> WindowMut<'_, T> {
    /// [`Window::of`] over a parent held exclusively: the form a staging copy
    /// writes through.
    pub fn of_mut<P>(
        parent: &mut DeviceBuffer<P>,
        byte_off: usize,
        len: usize,
    ) -> Result<WindowMut<'_, T>, GpuError> {
        Ok(WindowMut {
            window: Window {
                buf: cut(parent, byte_off, len, "WindowMut::of_mut")?,
                _parent: PhantomData,
            },
        })
    }

    /// A window over `len` `T` at device address `ptr` in `ctx`, memory the
    /// caller owns (a mapped page, a buffer it also writes through): the
    /// launches read and write it as [`window`]'s, and dropping it releases
    /// the context handle and frees nothing.
    ///
    /// # Safety
    ///
    /// [`window`]'s contract, and the caller orders every access to the
    /// memory through this window against its other handles to it.
    pub unsafe fn from_raw(ptr: sys::CUdeviceptr, len: usize, ctx: &Arc<CudaContext>) -> Self {
        WindowMut {
            window: Window {
                // SAFETY: the range is the caller's contract.
                buf: unsafe { window(ptr, len, ctx) },
                _parent: PhantomData,
            },
        }
    }
}

/// The checked cut the window constructors share: `len` `T` at `byte_off`,
/// inside `parent` and on `T`'s alignment, as a non-owning buffer of
/// `parent`'s context. Each constructor ties the result to `parent`'s
/// borrow, so the parent stays in place for the window's lifetime.
fn cut<T, P>(
    parent: &DeviceBuffer<P>,
    byte_off: usize,
    len: usize,
    what: &'static str,
) -> Result<ManuallyDrop<DeviceBuffer<T>>, GpuError> {
    let (size, align) = (size_of::<T>(), align_of::<T>());
    let refuse = |detail: String| GpuError::Shape { what, detail };
    if len == 0 {
        return Err(refuse(format!("a window of 0 {size}-byte values")));
    }
    if !byte_off.is_multiple_of(align) {
        return Err(refuse(format!(
            "byte {byte_off}, not a multiple of {align}, a {size}-byte value's alignment"
        )));
    }
    let end = len
        .checked_mul(size)
        .and_then(|bytes| bytes.checked_add(byte_off))
        .ok_or_else(|| refuse(format!("{len} × {size} B at byte {byte_off} overflows")))?;
    let parent_bytes = parent.num_bytes();
    if end > parent_bytes {
        return Err(refuse(format!(
            "{len} × {size} B at byte {byte_off} in a {parent_bytes}-byte parent"
        )));
    }
    let off = u64::try_from(byte_off).map_err(|_| refuse(format!("byte {byte_off} passes u64")))?;
    // SAFETY: the span `off .. off + len × size` lies inside `parent`'s
    // allocation and starts aligned for `T` (each refused above); the
    // constructors tie the window to `parent`'s borrow, so the parent stays
    // in place until the window is gone.
    Ok(unsafe { window::<T>(parent.cu_deviceptr() + off, len, parent.context()) })
}

/// The geometry of one 2-D copy between a device buffer and host memory:
/// `rows` runs of `width` bytes, the device's `dev_pitch` bytes apart and the
/// host's `host_pitch` bytes apart. A strided store (the planes of a KV
/// cache) moves as one call instead of one a plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pitched {
    pub width: usize,
    pub rows: usize,
    pub dev_pitch: usize,
    pub host_pitch: usize,
}

impl Pitched {
    /// The bytes a side spans, its first byte to its last: the rows' pitch
    /// times all but the last, and the last's width.
    fn span(&self, pitch: usize) -> Option<usize> {
        self.rows
            .checked_sub(1)?
            .checked_mul(pitch)?
            .checked_add(self.width)
    }

    /// Refused by name unless the copy moves something, the rows do not
    /// overlap, and both spans lie inside the `dev_len` bytes of the device
    /// buffer past `dev_off` and the `host_len` bytes of the host's.
    fn check(
        &self,
        what: &'static str,
        dev_off: usize,
        dev_len: usize,
        host_len: usize,
    ) -> Result<(), GpuError> {
        let refuse = |detail: String| GpuError::Shape { what, detail };
        if self.width == 0 || self.rows == 0 {
            return Err(refuse(format!(
                "{} rows of {} bytes",
                self.rows, self.width
            )));
        }
        if self.rows > 1 && (self.dev_pitch < self.width || self.host_pitch < self.width) {
            return Err(refuse(format!(
                "{} rows of {} bytes at pitches of {} (device) and {} (host) overlap",
                self.rows, self.width, self.dev_pitch, self.host_pitch
            )));
        }
        let dev_span = self.span(self.dev_pitch);
        let host_span = self.span(self.host_pitch);
        let dev_end = dev_span.and_then(|d| d.checked_add(dev_off));
        if dev_end.is_none_or(|e| e > dev_len) || host_span.is_none_or(|h| h > host_len) {
            return Err(refuse(format!(
                "{} rows of {} bytes (device pitch {}, host pitch {}) from byte {dev_off} of a \
                 {dev_len}-byte buffer into {host_len} host bytes",
                self.rows, self.width, self.dev_pitch, self.host_pitch
            )));
        }
        Ok(())
    }

    /// The driver's descriptor of the copy. Device to host when `to_host`.
    fn descriptor(
        &self,
        dev: sys::CUdeviceptr,
        host: *mut u8,
        to_host: bool,
    ) -> sys::CUDA_MEMCPY2D {
        let device = sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE;
        let on_host = sys::CUmemorytype_enum_CU_MEMORYTYPE_HOST;
        sys::CUDA_MEMCPY2D {
            srcXInBytes: 0,
            srcY: 0,
            srcMemoryType: if to_host { device } else { on_host },
            srcHost: if to_host {
                std::ptr::null()
            } else {
                host.cast_const().cast()
            },
            srcDevice: if to_host { dev } else { 0 },
            srcArray: std::ptr::null_mut(),
            srcPitch: if to_host {
                self.dev_pitch
            } else {
                self.host_pitch
            },
            dstXInBytes: 0,
            dstY: 0,
            dstMemoryType: if to_host { on_host } else { device },
            dstHost: if to_host {
                host.cast()
            } else {
                std::ptr::null_mut()
            },
            dstDevice: if to_host { 0 } else { dev },
            dstArray: std::ptr::null_mut(),
            dstPitch: if to_host {
                self.host_pitch
            } else {
                self.dev_pitch
            },
            WidthInBytes: self.width,
            Height: self.rows,
        }
    }
}

/// The value host memory holds before a device-to-host copy lands in it
/// ([`host_filled`]). Never zero: a zero fill comes back from the allocator
/// as pages nothing has written yet. A value no copy overwrote reads as this,
/// not as a plausible zero.
pub trait HostFill: DeviceCopy + Copy {
    const FILL: Self;
}

impl HostFill for f32 {
    const FILL: f32 = f32::NAN;
}

impl HostFill for u16 {
    const FILL: u16 = u16::MAX;
}

/// `n` [`HostFill::FILL`] values in host memory whose every page the host has
/// written: the pageable destination of a device-to-host copy. A card that
/// reaches pageable memory through the host's page tables (GB10's unified
/// memory) faults on each page a copy is the first to write, one page at a
/// time; a new allocation nothing has written is such pages. A large one is
/// advised onto huge pages first ([`os::advise_huge_within`]). `h2d_probe --arm readback`
/// (`bloomery-gpu-gates`) times a readback into each kind of destination.
#[must_use]
pub fn host_filled<T: HostFill>(n: usize) -> Vec<T> {
    let mut host = Vec::with_capacity(n);
    os::advise_huge_within(host.spare_capacity_mut());
    host.resize(n, T::FILL);
    host
}

/// `src` read back into a new [`host_filled`] `Vec`: a readback the caller
/// keeps. Blocking.
pub fn to_host<T: HostFill>(
    stream: &CudaStream,
    src: &DeviceBuffer<T>,
) -> Result<Vec<T>, GpuError> {
    let mut host = host_filled(src.len());
    src.copy_to_host(stream, &mut host)?;
    Ok(host)
}

/// Enqueue the 2-D copy `shape` from `dev` (from byte `dev_off`) to the host
/// memory at `host`, `host_len` bytes of it: one call for a strided store.
/// Refused by name, nothing enqueued, when either span leaves its buffer.
///
/// The destination that costs only the copy is page-locked memory of `dev`'s
/// context (a pinned buffer or an extent of one). A pageable one costs its
/// pages too, so it is [`host_filled`] first.
///
/// # Safety
///
/// - `host .. host + host_len` must be writable.
/// - Nothing may read, write or free it until the copy has completed: an
///   event recorded on `stream` after this call and waited for, or a
///   synchronize of `stream`.
pub unsafe fn dtoh_2d<D>(
    stream: &CudaStream,
    dev: &DeviceBuffer<D>,
    dev_off: usize,
    host: *mut u8,
    host_len: usize,
    shape: Pitched,
) -> Result<(), GpuError> {
    shape.check("dtoh_2d", dev_off, dev.num_bytes(), host_len)?;
    let at = dev.cu_deviceptr() + dev_off as u64;
    let copy = shape.descriptor(at, host, true);
    // SAFETY: both spans lie inside their buffers (checked above); the
    // descriptor is read at the call; the host side's contract is the
    // caller's.
    let rc = unsafe { sys::cuMemcpy2DAsync_v2(&raw const copy, stream.cu_stream()) };
    cu(rc, "cuMemcpy2DAsync_v2 (device to host)")
}

/// Enqueue the 2-D copy `shape` from the host memory at `host`, `host_len`
/// bytes of it, to `dev` (from byte `dev_off`): one call for a strided store.
/// Refused by name, nothing enqueued, when either span leaves its buffer.
///
/// # Safety
///
/// - `host .. host + host_len` must be readable.
/// - It must not be written or freed until the copy has completed: a
///   synchronize of `stream`, or an event recorded on it after this call and
///   waited for. (A pageable source is copied out by the driver before the
///   call returns, but the caller does not rely on it.)
pub unsafe fn htod_2d<D>(
    stream: &CudaStream,
    dev: &mut DeviceBuffer<D>,
    dev_off: usize,
    host: *const u8,
    host_len: usize,
    shape: Pitched,
) -> Result<(), GpuError> {
    shape.check("htod_2d", dev_off, dev.num_bytes(), host_len)?;
    let at = dev.cu_deviceptr() + dev_off as u64;
    let copy = shape.descriptor(at, host.cast_mut(), false);
    // SAFETY: both spans lie inside their buffers (checked above); the
    // descriptor is read at the call; the host side's contract is the
    // caller's.
    let rc = unsafe { sys::cuMemcpy2DAsync_v2(&raw const copy, stream.cu_stream()) };
    cu(rc, "cuMemcpy2DAsync_v2 (host to device)")
}

/// The check of a device-to-device 2-D copy ([`dtod_2d`]): `shape` read with
/// `dev_pitch` as the source's pitch and `host_pitch` as the destination's,
/// the source span from byte `src_off` of `src_len` bytes and the destination
/// span from byte `dst_off` of `dst_len`. Refused by name as [`Pitched`]'s own
/// check refuses, and when `dst_off` is past the destination's end.
fn dtod_check(
    shape: &Pitched,
    (src_off, src_len): (usize, usize),
    (dst_off, dst_len): (usize, usize),
) -> Result<(), GpuError> {
    const WHAT: &str = "dtod_2d";
    let dst_room = dst_len
        .checked_sub(dst_off)
        .ok_or_else(|| GpuError::Shape {
            what: WHAT,
            detail: format!(
                "a destination offset of byte {dst_off} past its {dst_len}-byte buffer (pitch {})",
                shape.host_pitch
            ),
        })?;
    shape.check(WHAT, src_off, src_len, dst_room)
}

/// Enqueue the 2-D copy `shape` from device buffer `src` (from byte
/// `src_off`) to device buffer `dst` (from byte `dst_off`): `shape.rows` runs
/// of `shape.width` bytes, `shape.dev_pitch` bytes apart in the source and
/// `shape.host_pitch` bytes apart in the destination. One call regroups a
/// strided layout (token-major rows into head-major runs). Refused by name,
/// nothing enqueued, when either span leaves its buffer ([`dtod_check`]).
/// Asynchronous on `stream`; the two buffers are distinct
/// borrows, so the spans never overlap.
pub fn dtod_2d<D>(
    stream: &CudaStream,
    (src, src_off): (&DeviceBuffer<D>, usize),
    (dst, dst_off): (&mut DeviceBuffer<D>, usize),
    shape: Pitched,
) -> Result<(), GpuError> {
    dtod_check(
        &shape,
        (src_off, src.num_bytes()),
        (dst_off, dst.num_bytes()),
    )?;
    let device = sys::CUmemorytype_enum_CU_MEMORYTYPE_DEVICE;
    let copy = sys::CUDA_MEMCPY2D {
        srcXInBytes: 0,
        srcY: 0,
        srcMemoryType: device,
        srcHost: std::ptr::null(),
        srcDevice: src.cu_deviceptr() + src_off as u64,
        srcArray: std::ptr::null_mut(),
        srcPitch: shape.dev_pitch,
        dstXInBytes: 0,
        dstY: 0,
        dstMemoryType: device,
        dstHost: std::ptr::null_mut(),
        dstDevice: dst.cu_deviceptr() + dst_off as u64,
        dstArray: std::ptr::null_mut(),
        dstPitch: shape.host_pitch,
        WidthInBytes: shape.width,
        Height: shape.rows,
    };
    // SAFETY: both spans lie inside their buffers (checked above), the
    // buffers are distinct borrows that outlive the enqueue on `stream`'s
    // order, and the descriptor is read at the call.
    let rc = unsafe { sys::cuMemcpy2DAsync_v2(&raw const copy, stream.cu_stream()) };
    cu(rc, "cuMemcpy2DAsync_v2 (device to device)")
}

/// `n` f32 of `src` copied into `dst`, one row: a tap's or a checkpoint's
/// copy of a value the step leaves. Asynchronous on `stream`; refused by
/// name, nothing enqueued, when either span leaves its buffer
/// ([`dtod_2d`]'s own check).
pub fn copy_row(
    stream: &CudaStream,
    src: &DeviceBuffer<f32>,
    dst: &mut DeviceBuffer<f32>,
    n: usize,
) -> Result<(), GpuError> {
    let width = n * 4;
    dtod_2d(
        stream,
        (src, 0),
        (dst, 0),
        Pitched {
            width,
            rows: 1,
            dev_pitch: width,
            host_pitch: width,
        },
    )
}

/// One device allocation cut into `N` consecutive parts, each of which the
/// launches read and write as a buffer of its own, while a launch handed the
/// whole writes every part at once — a gemv over row-concatenated weights
/// fills each projection's output in place. The parts are [`window`]s into
/// the whole, which this type owns: they live exactly as long as it does
/// and free nothing.
pub struct PartedBuffer<T, const N: usize> {
    parts: [ManuallyDrop<DeviceBuffer<T>>; N],
    whole: DeviceBuffer<T>,
}

impl<T: DeviceCopy, const N: usize> PartedBuffer<T, N> {
    /// A zero-filled allocation of `lens` summed, part `i` the `lens[i]`
    /// elements after the parts before it. Load-time only.
    pub fn zeroed(stream: &CudaStream, lens: [usize; N]) -> Result<Self, GpuError> {
        let refuse = || GpuError::shape("PartedBuffer::zeroed", format!("parts {lens:?}"));
        // Each part's byte offset, and the total in elements.
        let mut offs = [0u64; N];
        let mut total = 0usize;
        for (off, &len) in offs.iter_mut().zip(&lens) {
            let bytes = total.checked_mul(size_of::<T>()).ok_or_else(refuse)?;
            *off = u64::try_from(bytes).map_err(|_| refuse())?;
            total = total.checked_add(len).ok_or_else(refuse)?;
        }
        let whole = DeviceBuffer::<T>::zeroed(stream, total)?;
        let base = whole.cu_deviceptr();
        let mut i = 0;
        let parts = lens.map(|len| {
            let off = offs[i];
            i += 1;
            // SAFETY: the part spans `len` elements from byte `off` of `whole`,
            // inside its `total` (the sum of `lens`); `whole` is a `cuMemAlloc`
            // allocation aligned for `T`, owned by `self` and freed after the
            // windows, which free nothing.
            unsafe { window::<T>(base + off, len, whole.context()) }
        });
        Ok(PartedBuffer { parts, whole })
    }
}

impl<T, const N: usize> PartedBuffer<T, N> {
    /// The whole allocation, every part in order.
    pub fn whole(&self) -> &DeviceBuffer<T> {
        &self.whole
    }

    /// The whole allocation, for a launch that writes every part.
    pub fn whole_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.whole
    }

    /// Part `i` (`i < N`).
    pub fn part(&self, i: usize) -> &DeviceBuffer<T> {
        &self.parts[i]
    }

    /// Part `i` (`i < N`), for a launch that writes it alone.
    pub fn part_mut(&mut self, i: usize) -> &mut DeviceBuffer<T> {
        &mut self.parts[i]
    }
}

impl<T, const N: usize> Drop for PartedBuffer<T, N> {
    fn drop(&mut self) {
        for part in &mut self.parts {
            // SAFETY: each window is taken once, here, and never read again;
            // its raw parts are dropped (the context handle with them) and no
            // memory is freed — `whole` frees the allocation after this.
            let buf = unsafe { ManuallyDrop::take(part) };
            drop(buf.into_raw_parts());
        }
    }
}

/// A 2-D device tensor: `rows * cols` elements of `T`, row-major, uploaded
/// once. The element type is whatever the consuming kernel loads (`u32`
/// words for the K-quant gemvs, `f32` for activations, `u16` for F16 KV).
pub struct DeviceTensor<T> {
    buf: DeviceBuffer<T>,
    rows: usize,
    cols: usize,
}

impl<T: DeviceCopy> DeviceTensor<T> {
    /// Upload `data` (exactly `rows * cols` elements) on `stream`. This is a
    /// load-time operation: `from_host` synchronizes, so it must never run
    /// inside a graph capture.
    pub fn upload(
        stream: &CudaStream,
        data: &[T],
        rows: usize,
        cols: usize,
    ) -> Result<Self, GpuError> {
        if data.len() != rows * cols {
            return Err(GpuError::shape(
                "DeviceTensor::upload",
                format!("data.len() {} != rows*cols = {}*{}", data.len(), rows, cols),
            ));
        }
        let buf = DeviceBuffer::from_host(stream, data)?;
        Ok(DeviceTensor { buf, rows, cols })
    }

    /// Zero-filled tensor (scratch / KV). Load-time only, like `upload`.
    pub fn zeroed(stream: &CudaStream, rows: usize, cols: usize) -> Result<Self, GpuError> {
        let buf = DeviceBuffer::zeroed(stream, rows * cols)?;
        Ok(DeviceTensor { buf, rows, cols })
    }

    /// A tensor whose bytes are what the allocation held, for a load-time
    /// buffer a copy writes whole: the memset [`DeviceTensor::zeroed`]
    /// enqueues would only be written over. Load-time only, like `zeroed`;
    /// `stream`'s context must be current.
    ///
    /// # Safety
    ///
    /// Every one of the `rows * cols` elements must be written on `stream`
    /// before anything reads the tensor, and a tensor whose writing failed
    /// must be dropped unread.
    pub(crate) unsafe fn unfilled(
        stream: &CudaStream,
        rows: usize,
        cols: usize,
    ) -> Result<Self, GpuError> {
        let size = rows
            .checked_mul(cols)
            .and_then(|len| Some((len, len.checked_mul(size_of::<T>())?)));
        let Some((len, bytes)) = size else {
            let detail = format!("{rows} rows of {cols} elements pass usize bytes");
            return Err(GpuError::shape("DeviceTensor::unfilled", detail));
        };
        // No allocation is made for no bytes, which the driver refuses: a
        // null pointer the drop skips, as `zeroed` gives.
        let ptr = if bytes == 0 {
            0
        } else {
            // SAFETY: the stream's context is current and `bytes` is positive.
            unsafe { cuda_core::simt::memory::malloc_sync(bytes)? }
        };
        // SAFETY: `ptr` owns this context's allocation of `len` elements, or
        // is null for no bytes; the buffer takes it over and frees it on drop.
        let buf = unsafe { DeviceBuffer::from_raw_parts(ptr, len, stream.context().clone()) };
        Ok(DeviceTensor { buf, rows, cols })
    }

    /// A `rows × cols` tensor over memory it does not own, for a launcher
    /// that takes a tensor: [`window`] with a shape. Give it back with
    /// [`DeviceTensor::release`].
    ///
    /// # Panics
    ///
    /// Panics when `rows * cols` overflows `usize`: the window's length is
    /// that product, and a wrapped length must not reach the launches.
    ///
    /// # Safety
    ///
    /// [`window`]'s contract, for the `rows * cols` `T` at `ptr`.
    pub unsafe fn window(
        ptr: sys::CUdeviceptr,
        rows: usize,
        cols: usize,
        ctx: &Arc<CudaContext>,
    ) -> ManuallyDrop<DeviceTensor<T>> {
        let len = rows.checked_mul(cols).unwrap_or_else(|| {
            panic!("DeviceTensor::window: rows {rows} × cols {cols} overflows usize")
        });
        // SAFETY: the span is the caller's, under `window`'s contract.
        let buf = unsafe { window::<T>(ptr, len, ctx) };
        ManuallyDrop::new(DeviceTensor {
            buf: ManuallyDrop::into_inner(buf),
            rows,
            cols,
        })
    }

    /// Give back a [`DeviceTensor::window`]: its handle on the context is
    /// dropped and no memory is freed.
    pub fn release(t: ManuallyDrop<DeviceTensor<T>>) {
        drop(ManuallyDrop::into_inner(t).buf.into_raw_parts());
    }

    /// A `rows × cols` window of `parent` from byte `byte_off`, for a
    /// launcher that takes a tensor: [`Window::of`]'s checks, with the shape
    /// on top. The window borrows `parent`; dropping it releases the context
    /// handle, as a [`DeviceTensor::window`] caller's
    /// [`DeviceTensor::release`] does.
    pub fn window_of<P>(
        parent: &DeviceBuffer<P>,
        byte_off: usize,
        rows: usize,
        cols: usize,
    ) -> Result<TensorWindow<'_, T>, GpuError> {
        let what = "DeviceTensor::window_of";
        let len = rows
            .checked_mul(cols)
            .filter(|len| *len > 0)
            .ok_or_else(|| GpuError::shape(what, format!("{rows} × {cols} values")))?;
        Ok(TensorWindow {
            tensor: Some(DeviceTensor {
                buf: ManuallyDrop::into_inner(cut(parent, byte_off, len, what)?),
                rows,
                cols,
            }),
            _parent: PhantomData,
        })
    }

    /// Row count.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Elements per row.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The flat buffer the generated launchers take (`&DeviceBuffer<T>` for
    /// `&[T]` kernel parameters).
    pub fn buf(&self) -> &DeviceBuffer<T> {
        &self.buf
    }

    /// Writable view for `DisjointSlice<T>` / `&mut [T]` kernel parameters.
    pub fn buf_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.buf
    }
}

/// A `rows × cols` non-owning window of `parent` from byte `byte_off`,
/// borrowed for the window's lifetime, for a launcher that takes a tensor:
/// [`DeviceTensor::window_of`]'s checked cut with the shape on top. Dropping
/// it frees nothing — the parent frees the allocation after the window is
/// gone.
///
/// Like [`Window::of`], this is the shared form, over a parent borrowed
/// shared: it derefs to `&DeviceTensor<T>` only.
pub struct TensorWindow<'p, T> {
    tensor: Option<DeviceTensor<T>>,
    _parent: PhantomData<&'p ()>,
}

impl<T> Deref for TensorWindow<'_, T> {
    type Target = DeviceTensor<T>;

    fn deref(&self) -> &DeviceTensor<T> {
        // `Some` for the window's whole life; only `Drop` takes it.
        self.tensor.as_ref().expect("a live tensor window")
    }
}

impl<T> Drop for TensorWindow<'_, T> {
    fn drop(&mut self) {
        // The window owns no memory: the tensor's raw parts are dropped, the
        // context handle with them, and nothing is freed — the parent frees
        // the allocation after the window is gone.
        if let Some(t) = self.tensor.take() {
            drop(t.buf.into_raw_parts());
        }
    }
}

/// The largest K a [`Q8Act`] takes: the widest K-quant input of the models
/// this engine runs — a dense FFN down of 22,016 values; DeepSeek-V4.1's
/// `hc_{attn,ffn}_fn`, which read all four residual streams, take 4 × 5120.
pub(crate) const Q8ACT_MAX_K: usize = 22_016;

/// The most columns a per-slot [`Q8Act`] takes ([`Q8Act::with_slots`]):
/// every routed slot of a prompt batch — the host union's `UNION_MAX_COLS`
/// tokens at eight slots each, GLM-5.3-Flash's (V4.1 routes six) — the down
/// input of the batch's grouped experts.
pub(crate) const Q8ACT_MAX_SLOTS: usize = 4096;
const _: () = assert!(Q8ACT_MAX_SLOTS == model::moe::UNION_MAX_COLS * 8);

/// The most token columns a tier's staged block activation takes
/// ([`Q8Act::with_tier_cols`]): a block of the widest batch port a body
/// opens, a ubatch of 4,096 positions.
pub(crate) const TIER_ACT_MAX_COLS: usize = 4096;

/// q8_1 activation scratch for up to `m` columns of `k` values each. One
/// set per distinct input site: sites that read the same activation
/// (gate·up, q·kv_a) share one (decision 2, as the CPU engine does).
///
/// `k` is a multiple of 256 with 256 <= k <= [`Q8ACT_MAX_K`]. The cap is a
/// sanity bound, not a layout limit: no kernel stages a K-sized array, and
/// every index the layout computes stays far inside u32 at the cap. With
/// n_sb = k/256 super-blocks per row, buffer sizes in elements per column
/// are: q3 `64 * ceil(n_sb/2)` u64, q4 `256 * ceil(n_sb/4)` u32,
/// q6 `128 * ceil(n_sb/2)` u32, s8 `8 * n_sb` i32, d8 `2 * n_sb` f32.
/// The q3/q6 buffers are permuted per 2-super-block group and q4 per
/// 4-super-block group; a partial final group (n_sb odd, or n_sb not a
/// multiple of 4) leaves the group's tail slots allocated but never written
/// or read — hence the ceil. At k = 2048 (n_sb = 8) these are 256 u64 /
/// 512 / 512 / 64 / 16 per column.
pub struct Q8Act {
    pub(crate) q3: DeviceBuffer<u64>,
    pub(crate) q4: DeviceBuffer<u32>,
    pub(crate) q6: DeviceBuffer<u32>,
    pub(crate) s8: DeviceBuffer<i32>,
    pub(crate) d8: DeviceBuffer<f32>,
    m: usize,
    k: usize,
}

impl Q8Act {
    /// Allocate scratch for `m` (1..=8) columns of `k` values. Load-time
    /// only.
    pub fn with_k(stream: &CudaStream, m: usize, k: usize) -> Result<Self, GpuError> {
        if !(1..=8).contains(&m) {
            return Err(GpuError::shape(
                "Q8Act::with_k",
                format!("1 <= m <= 8, got {m}"),
            ));
        }
        Q8Act::alloc("Q8Act::with_k", stream, m, k)
    }

    /// Allocate scratch for `cols` (1..=[`Q8ACT_MAX_SLOTS`]) columns of `k`
    /// values, one per expert slot: the input of a `_sel` down over several
    /// tokens' slots, which reads column `s` for slot `s`, and of the
    /// quantizer that writes it. The multi-column gemvs refuse more than
    /// eight columns by their own check. Load-time only.
    pub fn with_slots(stream: &CudaStream, cols: usize, k: usize) -> Result<Self, GpuError> {
        if !(1..=Q8ACT_MAX_SLOTS).contains(&cols) {
            return Err(GpuError::shape(
                "Q8Act::with_slots",
                format!("1 <= cols <= {Q8ACT_MAX_SLOTS}, got {cols}"),
            ));
        }
        Q8Act::alloc("Q8Act::with_slots", stream, cols, k)
    }

    /// Allocate scratch for `cols` (1..=[`TIER_ACT_MAX_COLS`]) token columns
    /// of `k` values: an expert tier's staged block of a prompt batch, one
    /// column a token, which the batch port's tier leg quantizes on the tier
    /// card. Load-time only.
    pub fn with_tier_cols(stream: &CudaStream, cols: usize, k: usize) -> Result<Self, GpuError> {
        if !(1..=TIER_ACT_MAX_COLS).contains(&cols) {
            return Err(GpuError::shape(
                "Q8Act::with_tier_cols",
                format!("1 <= cols <= {TIER_ACT_MAX_COLS}, got {cols}"),
            ));
        }
        Q8Act::alloc("Q8Act::with_tier_cols", stream, cols, k)
    }

    /// The planes for `m` columns of `k` values, `k` checked here.
    fn alloc(
        what: &'static str,
        stream: &CudaStream,
        m: usize,
        k: usize,
    ) -> Result<Self, GpuError> {
        if !k.is_multiple_of(256) || !(256..=Q8ACT_MAX_K).contains(&k) {
            return Err(GpuError::shape(
                what,
                format!("k must be a multiple of 256 in 256..={Q8ACT_MAX_K}, got {k}"),
            ));
        }
        let n_sb = k / 256;
        Ok(Q8Act {
            q3: DeviceBuffer::zeroed(stream, m * 64 * n_sb.div_ceil(2))?,
            q4: DeviceBuffer::zeroed(stream, m * 256 * n_sb.div_ceil(4))?,
            q6: DeviceBuffer::zeroed(stream, m * 128 * n_sb.div_ceil(2))?,
            s8: DeviceBuffer::zeroed(stream, m * 8 * n_sb)?,
            d8: DeviceBuffer::zeroed(stream, m * 2 * n_sb)?,
            m,
            k,
        })
    }

    pub fn m(&self) -> usize {
        self.m
    }

    pub(crate) fn k(&self) -> usize {
        self.k
    }

    /// Device bytes of the five planes, as allocated.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.q3.num_bytes()
            + self.q4.num_bytes()
            + self.q6.num_bytes()
            + self.s8.num_bytes()
            + self.d8.num_bytes()
    }

    /// Super-blocks per row (k/256) — the row geometry every K-quant gemv
    /// of this engine launches with.
    #[must_use]
    pub fn n_sb(&self) -> usize {
        self.k / 256
    }

    /// The codes `cores::q3k_row_dot` reads: `64 * ceil(n_sb/2)` u64 per
    /// column in the quantizer's pair permutation, for a Q3_K kernel outside
    /// this crate.
    #[must_use]
    pub fn q3(&self) -> &DeviceBuffer<u64> {
        &self.q3
    }

    /// The block scales that go with [`Q8Act::q3`]: `2 * n_sb` f32 per
    /// column.
    #[must_use]
    pub fn d8(&self) -> &DeviceBuffer<f32> {
        &self.d8
    }
}

#[cfg(test)]
thread_local! {
    pub(crate) static FAIL_ALLOCATION: std::cell::Cell<Option<&'static str>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(test)]
pub(crate) fn allocation_failure(what: &'static str) -> Result<(), GpuError> {
    if FAIL_ALLOCATION.get() == Some(what) {
        FAIL_ALLOCATION.set(None);
        return Err(GpuError::state(what, "injected allocation failure"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `DeviceTensor::window` panics by name on a product that wraps, before
    /// the call reaches the driver — but the `ctx` it must be handed exists
    /// only through `CudaContext::new`, so the test runs on the box.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_window_panics_by_name_when_rows_times_cols_wraps() {
        let ctx = crate::capsync::fresh_context(0).expect("CUDA device 0");
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: the product overflows, so the call panics before `ptr`
            // or `ctx` is used.
            unsafe { DeviceTensor::<u32>::window(sys::CUdeviceptr::MAX, usize::MAX, 2, &ctx) };
        }));
        let Err(payload) = panicked else {
            panic!("the overflow must panic, not wrap")
        };
        let msg = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied());
        let msg = msg.unwrap_or_default();
        assert!(
            msg.contains("DeviceTensor::window")
                && msg.contains(&format!("rows {}", usize::MAX))
                && msg.contains("cols 2"),
            "the panic names the constructor and both values: {msg}"
        );
    }

    /// A window over raw memory releases the context handle it holds when it
    /// drops and frees nothing: the buffer under it still reads back.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_raw_window_releases_its_context_handle() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let buf = DeviceBuffer::from_host(&stream, &[7u32; 4]).expect("a buffer");
        let held = Arc::strong_count(&ctx);
        // SAFETY: the window covers `buf`'s four values in `ctx`, which
        // outlive it, and nothing is enqueued through it.
        let w = unsafe { WindowMut::<u32>::from_raw(buf.cu_deviceptr(), 4, &ctx) };
        assert_eq!(Arc::strong_count(&ctx), held + 1);
        drop(w);
        assert_eq!(
            Arc::strong_count(&ctx),
            held,
            "the window's context handle released"
        );
        assert_eq!(buf.to_host_vec(&stream).expect("the buffer"), [7; 4]);
    }

    /// Four planes of 64 `u16` (a stride of 128 bytes) with the first 20 of
    /// each carried: one 2-D copy moves the bytes four plane-by-plane copies
    /// move (planes × width, packed at the host's pitch), and its way back
    /// writes those bytes and not one byte of the gap past a plane's width.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_a_2d_copy_moves_the_bytes_of_the_planes_one_by_one() {
        const PLANES: usize = 4;
        const STRIDE: usize = 64;
        const LEN: usize = 20;
        const GAP: u16 = 0xDEAD;
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let on_card: Vec<u16> = (0..PLANES * STRIDE).map(|i| i as u16 + 1).collect();
        let dev = DeviceBuffer::from_host(&stream, &on_card).expect("the device store");
        let width = LEN * 2;
        let shape = Pitched {
            width,
            rows: PLANES,
            dev_pitch: STRIDE * 2,
            host_pitch: width,
        };
        let mut pin =
            cuda_core::PinnedHostBuffer::<u8>::zeroed(&ctx, PLANES * width).expect("pinned");
        // The plane-by-plane bytes, worked out from the values.
        let by_plane: Vec<u8> = (0..PLANES)
            .flat_map(|p| {
                on_card[p * STRIDE..p * STRIDE + LEN]
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
            })
            .collect();
        let mut pageable = vec![0xA5u8; PLANES * width];
        for host in [pageable.as_mut_slice(), pin.as_mut_slice()] {
            let len = host.len();
            // SAFETY: `host` holds `len` writable bytes; nothing touches or
            // frees them before the synchronize below.
            unsafe { dtoh_2d(&stream, &dev, 0, host.as_mut_ptr(), len, shape) }
                .expect("a 2-D read");
            stream.synchronize().expect("the read lands");
            assert_eq!(host, &by_plane[..], "the bytes of the planes, one by one");
            let mut back = DeviceBuffer::from_host(&stream, &vec![GAP; PLANES * STRIDE])
                .expect("a poisoned store");
            // SAFETY: `host` holds `len` readable bytes and is neither written
            // nor freed before the synchronize below.
            unsafe { htod_2d(&stream, &mut back, 0, host.as_ptr(), len, shape) }
                .expect("a 2-D write");
            stream.synchronize().expect("the write lands");
            let got = back.to_host_vec(&stream).expect("the store");
            for p in 0..PLANES {
                let plane = &got[p * STRIDE..(p + 1) * STRIDE];
                assert_eq!(
                    &plane[..LEN],
                    &on_card[p * STRIDE..p * STRIDE + LEN],
                    "plane {p}"
                );
                assert!(
                    plane[LEN..].iter().all(|&v| v == GAP),
                    "plane {p} written past its width"
                );
            }
        }
        // From a byte offset: the last two planes only.
        let mut tail = cuda_core::PinnedHostBuffer::<u8>::zeroed(&ctx, 2 * width).expect("pinned");
        let (tail_len, shape) = (tail.len(), Pitched { rows: 2, ..shape });
        // SAFETY: as the first read, into `tail`.
        unsafe {
            dtoh_2d(
                &stream,
                &dev,
                2 * STRIDE * 2,
                tail.as_mut_ptr(),
                tail_len,
                shape,
            )
        }
        .expect("a 2-D read from an offset");
        stream.synchronize().expect("the read lands");
        assert_eq!(tail.as_slice(), &by_plane[2 * width..], "planes 2 and 3");
    }

    /// Host memory a copy lands in holds the fill, not zero: a zero fill comes
    /// back from the allocator as pages nothing has written.
    #[test]
    fn host_filled_memory_holds_the_fill_not_zero() {
        assert!(host_filled::<f32>(5).iter().all(|v| v.is_nan()));
        assert_eq!(host_filled::<u16>(5), [u16::MAX; 5]);
        let advised = host_filled::<u16>(os::ADVISE_FROM / 2 + 3);
        assert_eq!(advised.len(), os::ADVISE_FROM / 2 + 3);
        assert!(advised.iter().all(|&v| v == u16::MAX), "the advised fill");
    }

    /// An owned readback returns every value of the device buffer.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_an_owned_readback_returns_each_read_whole() {
        let (_ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let values: Vec<f32> = (0..4096u16).map(f32::from).collect();
        let on_card = DeviceBuffer::from_host(&stream, &values).expect("a buffer");
        assert_eq!(to_host(&stream, &on_card).expect("an owned read"), values);
    }

    /// A 2-D copy leaves by name, with nothing enqueued, when a side's span
    /// leaves its buffer, when the rows overlap, when it moves nothing, or when
    /// the span's arithmetic wraps; it takes a single row at any pitch.
    #[test]
    fn a_2d_copy_refuses_by_name_what_leaves_its_buffers() {
        let shape = Pitched {
            width: 40,
            rows: 4,
            dev_pitch: 128,
            host_pitch: 40,
        };
        // 3 pitches and the last row's width: 424 device bytes, 160 host bytes.
        assert!(shape.check("t", 0, 424, 160).is_ok());
        for (dev_off, dev_len, host_len, why) in [
            (0, 423, 160, "a device span one byte short"),
            (1, 424, 160, "an offset past the device's end"),
            (0, 424, 159, "a host span one byte short"),
        ] {
            let err = shape.check("t", dev_off, dev_len, host_len).expect_err(why);
            assert!(
                err.to_string().starts_with("t: ") && err.to_string().contains("pitch"),
                "{why}: {err}"
            );
        }
        for (shape, why) in [
            (Pitched { rows: 0, ..shape }, "no rows"),
            (Pitched { width: 0, ..shape }, "no width"),
            (
                Pitched {
                    dev_pitch: 32,
                    ..shape
                },
                "device rows overlap",
            ),
            (
                Pitched {
                    host_pitch: 32,
                    ..shape
                },
                "host rows overlap",
            ),
            (
                Pitched {
                    rows: usize::MAX,
                    dev_pitch: 2,
                    host_pitch: 2,
                    ..shape
                },
                "a span that wraps",
            ),
        ] {
            assert!(
                shape.check("t", 0, usize::MAX, usize::MAX).is_err(),
                "{why}"
            );
        }
        let one = Pitched {
            rows: 1,
            dev_pitch: 0,
            host_pitch: 0,
            ..shape
        };
        assert!(
            one.check("t", 0, 40, 40).is_ok(),
            "a single row has no pitch to overlap"
        );
    }

    /// A device-to-device copy checks both spans: four rows of 40 bytes read
    /// 128 bytes apart (424 source bytes) and written 40 apart (160 bytes from
    /// the destination's offset); a span a byte short or an offset past the
    /// destination's end is refused by name.
    #[test]
    fn a_device_to_device_2d_copy_refuses_by_name_what_leaves_its_buffers() {
        let shape = Pitched {
            width: 40,
            rows: 4,
            dev_pitch: 128,
            host_pitch: 40,
        };
        assert!(dtod_check(&shape, (0, 424), (8, 168)).is_ok());
        for (src, dst, why) in [
            ((0, 423), (0, 160), "a source span one byte short"),
            ((1, 424), (0, 160), "a source offset past the span's room"),
            ((0, 424), (1, 160), "a destination span one byte short"),
            ((0, 424), (161, 160), "a destination offset past its end"),
        ] {
            let err = dtod_check(&shape, src, dst).expect_err(why);
            assert!(
                err.to_string().starts_with("dtod_2d: ") && err.to_string().contains("pitch"),
                "{why}: {err}"
            );
        }
    }
}
