//! Windows of a device buffer that the launches take as buffers of their
//! own: a prompt batch keeps a value per token in one allocation, and a
//! launch over a chunk of its tokens reads and writes the chunk's run of it.
//! A [`Span`] borrows its buffer, which therefore outlives it and stays in
//! place; dropping it frees nothing. A [`SpanMut`] borrows it mutably, so no
//! other window of the same buffer is written while it lives.

use std::marker::PhantomData;
use std::mem::{ManuallyDrop, size_of};
use std::ops::{Deref, DerefMut};

use bloomery_gpu::{GpuError, window};
use cuda_core::{DeviceBuffer, DeviceCopy};

/// `len` values of a buffer from value `off`, read-only.
pub struct Span<'a, T> {
    buf: ManuallyDrop<DeviceBuffer<T>>,
    _parent: PhantomData<&'a DeviceBuffer<T>>,
}

/// `len` values of a buffer from value `off`, written by the launch that
/// takes it.
pub struct SpanMut<'a, T> {
    buf: ManuallyDrop<DeviceBuffer<T>>,
    _parent: PhantomData<&'a mut DeviceBuffer<T>>,
}

impl<T> Deref for Span<'_, T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &DeviceBuffer<T> {
        &self.buf
    }
}

impl<T> Deref for SpanMut<'_, T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &DeviceBuffer<T> {
        &self.buf
    }
}

impl<T> DerefMut for SpanMut<'_, T> {
    fn deref_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.buf
    }
}

/// The window's raw parts dropped, the context handle with them; no memory
/// is freed.
fn release<T>(buf: &mut ManuallyDrop<DeviceBuffer<T>>) {
    // SAFETY: the caller takes `buf` once, from its drop, and never reads it
    // again.
    let buf = unsafe { ManuallyDrop::take(buf) };
    drop(buf.into_raw_parts());
}

impl<T> Drop for Span<'_, T> {
    fn drop(&mut self) {
        release(&mut self.buf);
    }
}

impl<T> Drop for SpanMut<'_, T> {
    fn drop(&mut self) {
        release(&mut self.buf);
    }
}

/// The byte offset of value `off`, refused (as `what`'s error) when `off ..
/// off + len` is empty or passes the buffer's `have` values.
fn offset<T>(what: &'static str, have: usize, off: usize, len: usize) -> Result<u64, GpuError> {
    let refuse = || GpuError::Shape {
        what,
        detail: format!("a window of {len} values at {off} in a buffer of {have}"),
    };
    if len == 0 || off.checked_add(len).is_none_or(|end| end > have) {
        return Err(refuse());
    }
    u64::try_from(off * size_of::<T>()).map_err(|_| refuse())
}

/// Values `off .. off + len` of `buf`; refused when the run is empty or
/// passes the buffer's end.
pub fn span<'a, T: DeviceCopy>(
    what: &'static str,
    buf: &'a DeviceBuffer<T>,
    off: usize,
    len: usize,
) -> Result<Span<'a, T>, GpuError> {
    let at = offset::<T>(what, buf.len(), off, len)?;
    // SAFETY: the values lie inside `buf`'s allocation (checked above) and
    // start on a value boundary; `buf` is borrowed for the window's lifetime,
    // so it outlives the window and stays in place.
    let w = unsafe { window::<T>(buf.cu_deviceptr() + at, len, buf.context()) };
    Ok(Span {
        buf: w,
        _parent: PhantomData,
    })
}

/// Values `off .. off + len` of `buf`, to write; refused as [`span`].
pub fn span_mut<'a, T: DeviceCopy>(
    what: &'static str,
    buf: &'a mut DeviceBuffer<T>,
    off: usize,
    len: usize,
) -> Result<SpanMut<'a, T>, GpuError> {
    let at = offset::<T>(what, buf.len(), off, len)?;
    // SAFETY: as in `span`; `buf` is borrowed mutably for the window's
    // lifetime, so the window is the only writer of these values while it
    // lives.
    let w = unsafe { window::<T>(buf.cu_deviceptr() + at, len, buf.context()) };
    Ok(SpanMut {
        buf: w,
        _parent: PhantomData,
    })
}
