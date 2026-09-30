//! Windows of a device buffer that the launches take as buffers of their
//! own: a prompt batch keeps a value per token in one allocation, and a
//! launch over a chunk of its tokens reads and writes the chunk's run of it.
//! A [`Span`] borrows its buffer, which therefore outlives it and stays in
//! place; dropping it frees nothing. A [`SpanMut`] borrows it mutably, so no
//! other window of the same buffer is written while it lives.

use std::mem::size_of;

use bloomery_gpu::{GpuError, Window, WindowMut};
use cuda_core::{DeviceBuffer, DeviceCopy};

/// `len` values of a buffer from value `off`, read-only: the gpu crate's
/// checked, borrowed [`Window`].
pub type Span<'a, T> = Window<'a, T>;

/// `len` values of a buffer from value `off`, written by the launch that
/// takes it: a [`WindowMut`], from the buffer held exclusively.
pub type SpanMut<'a, T> = WindowMut<'a, T>;

/// The byte offset of value `off`, refused (as `what`'s error) when `off ..
/// off + len` is empty or passes the buffer's `have` values.
fn offset<T>(what: &'static str, have: usize, off: usize, len: usize) -> Result<usize, GpuError> {
    let refuse = || GpuError::Shape {
        what,
        detail: format!("a window of {len} values at {off} in a buffer of {have}"),
    };
    if len == 0 || off.checked_add(len).is_none_or(|end| end > have) {
        return Err(refuse());
    }
    off.checked_mul(size_of::<T>()).ok_or_else(refuse)
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
    Window::of(buf, at, len)
}

/// Values `off .. off + len` of `buf`, to write; refused as [`span`].
pub fn span_mut<'a, T: DeviceCopy>(
    what: &'static str,
    buf: &'a mut DeviceBuffer<T>,
    off: usize,
    len: usize,
) -> Result<SpanMut<'a, T>, GpuError> {
    let at = offset::<T>(what, buf.len(), off, len)?;
    WindowMut::of_mut(buf, at, len)
}
