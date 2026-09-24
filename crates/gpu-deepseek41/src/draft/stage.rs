//! How a draft pass gets its per-call inputs onto the card: one pinned host
//! image, one asynchronous copy into its device twin, and typed windows of
//! that twin for the launches ([`Inbox`]). A captured graph reads the twin
//! at fixed addresses; the copy runs on the engine stream right before the
//! launch or the replay, outside any graph.

use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};

use bloomery_gpu::{Gpu, GpuError, window};
use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer};

const WHAT: &str = "draft::stage";

/// A non-owning window of a device buffer, borrowed for its lifetime.
pub(super) struct View<'a, T> {
    buf: ManuallyDrop<DeviceBuffer<T>>,
    _parent: PhantomData<&'a ()>,
}

impl<T> Deref for View<'_, T> {
    type Target = DeviceBuffer<T>;
    fn deref(&self) -> &DeviceBuffer<T> {
        &self.buf
    }
}

impl<T> DerefMut for View<'_, T> {
    fn deref_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.buf
    }
}

/// `len` values of `T` from value `off` of `parent`, whatever its element
/// type. The launches read and write a view as a buffer of their own; a
/// mutable view needs `parent` exclusively ([`view_mut`]).
pub(super) fn view<T, P>(
    parent: &DeviceBuffer<P>,
    off: usize,
    len: usize,
) -> Result<View<'_, T>, GpuError> {
    let size = size_of::<T>();
    let end = off.checked_add(len).and_then(|end| end.checked_mul(size));
    if len == 0 || end.is_none_or(|end| end > parent.num_bytes()) {
        return Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "a view of {len} × {size} B at {off} in {} B",
                parent.num_bytes()
            ),
        });
    }
    let bytes = u64::try_from(off * size).map_err(|_| GpuError::Shape {
        what: WHAT,
        detail: format!("offset {off}"),
    })?;
    // SAFETY: `off .. off + len` values of `T` lie inside `parent`'s
    // allocation (checked above), at a multiple of `T`'s size from its
    // (256-aligned) start; `parent` is borrowed for the view's lifetime, so
    // the memory outlives it and stays in place.
    let buf = unsafe { window::<T>(parent.cu_deviceptr() + bytes, len, parent.context()) };
    Ok(View {
        buf,
        _parent: PhantomData,
    })
}

/// [`view`] of a buffer the caller holds exclusively.
pub(super) fn view_mut<T, P>(
    parent: &mut DeviceBuffer<P>,
    off: usize,
    len: usize,
) -> Result<View<'_, T>, GpuError> {
    view(parent, off, len)
}

/// A pass's per-call inputs: `words` u32 on the host (pinned) and on the
/// card. The host side is written between calls, one copy moves a span of
/// it, and the launches read windows of the device side ([`Inbox::f32s`],
/// [`Inbox::u32s`]). f32 inputs are stored as their bits.
pub(super) struct Inbox {
    host: PinnedHostBuffer<u32>,
    dev: DeviceBuffer<u32>,
    /// Recorded after each copy: the host side is not written again before
    /// the copy has read it.
    copied: CudaEvent,
}

impl Inbox {
    /// A zeroed image of `words` words on both sides. Load-time only.
    pub(super) fn new(gpu: &Gpu, words: usize) -> Result<Inbox, GpuError> {
        Ok(Inbox {
            host: PinnedHostBuffer::zeroed(gpu.context(), words)?,
            dev: DeviceBuffer::zeroed(gpu.stream(), words)?,
            copied: gpu.context().new_event(None)?,
        })
    }

    /// The host image, writable once the last copy has read it (blocks until
    /// then; an image never copied is writable at once).
    pub(super) fn host_mut(&mut self) -> Result<&mut [u32], GpuError> {
        self.copied.synchronize()?;
        Ok(self.host.as_mut_slice())
    }

    /// Enqueue the copy of host words `0 .. words` to the same device words
    /// on `stream`. Asynchronous.
    pub(super) fn upload(&mut self, stream: &CudaStream, words: usize) -> Result<(), GpuError> {
        let src = self.host.as_slice().get(..words).ok_or(GpuError::Shape {
            what: WHAT,
            detail: format!("an upload of {words} words of {}", self.host.len()),
        })?;
        let mut dst = view_mut::<u32, u32>(&mut self.dev, 0, words)?;
        // SAFETY: `src` is pinned memory this inbox owns; it is not written
        // before `copied` (recorded right below, after the copy) has passed
        // ([`Inbox::host_mut`]), nor freed before it ([`Inbox`]'s drop), so
        // the copy reads it whole.
        unsafe { dst.copy_from_host_async_unchecked(stream, src)? };
        self.copied.record(stream)?;
        Ok(())
    }

    /// `len` f32 of the device image from word `off`.
    pub(super) fn f32s(&self, off: usize, len: usize) -> Result<View<'_, f32>, GpuError> {
        view(&self.dev, off, len)
    }

    /// `len` u32 of the device image from word `off`.
    pub(super) fn u32s(&self, off: usize, len: usize) -> Result<View<'_, u32>, GpuError> {
        view(&self.dev, off, len)
    }
}

impl Drop for Inbox {
    fn drop(&mut self) {
        // The pinned image outlives any copy still reading it. A failure on
        // the drop path is unreportable and ignored.
        let _ = self.copied.synchronize();
    }
}

/// `v`'s bits into `dst`, then zeros to its end.
pub(super) fn put_f32(dst: &mut [u32], v: &[f32]) {
    let (head, tail) = dst.split_at_mut(v.len().min(dst.len()));
    for (d, x) in head.iter_mut().zip(v) {
        *d = x.to_bits();
    }
    tail.fill(0);
}
