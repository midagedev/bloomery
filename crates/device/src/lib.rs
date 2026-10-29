//! The device contract: the traits an engine programs against, whatever the
//! backend underneath is. Nothing here names a backend, an operating system
//! or a vendor type. A backend implements the traits; the engine never asks
//! which one it got.
//!
//! The counts are typed apart ([`Elems`] and [`Bytes`]): an element count and
//! a byte count are different numbers, and a signature that takes both says
//! which is which. A window outside its buffer is refused by name, never
//! clamped: a clamped window is a defined output for an undefined input.
//!
//! A window borrows its buffer. An owner that keeps a buffer beside the
//! windows it cuts from it keeps their [`Span`]s instead and takes each window
//! again when it binds it; a window of another element type over the same
//! bytes is [`Window::cast`]. Copies between the host and a window, and
//! between two windows, are the device's ([`Device::read`], [`Device::write`],
//! [`Device::copy`]), ordered on a stream like the kernels.

pub mod conformance;
pub mod fault;
pub mod pages;

use core::marker::PhantomData;
use core::ops::Range;
use core::sync::atomic::{AtomicI32, AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};

pub use fault::{FAULT_NONE, Fault, FaultSite, FaultWord, LAYER_HEAD, LAYER_NONE, StepOrder};
pub use pages::HostPages;

use bloomery_placement::placement::devices::DeviceInfo;

/// The alignment every backend's buffer starts at, at least. Every element
/// type's alignment is at most this, so a window whose element offset is
/// whole in the element type it is read as is aligned for it.
pub const BUFFER_ALIGN: usize = 16;

/// An element's atomic twin: how one element of a shared buffer is loaded and
/// stored while the device or another host thread may touch it.
pub trait Atom<T>: Send + Sync {
    /// The element, loaded with `order`.
    fn load(&self, order: Ordering) -> T;

    /// The element becomes `v`, stored with `order`.
    fn store(&self, v: T, order: Ordering);
}

/// Plain data a buffer holds: every bit pattern a value, no padding.
///
/// # Safety
/// The implementor has no padding and no invalid bit pattern, its alignment
/// is at most [`BUFFER_ALIGN`], and `Atom` has its size and alignment and
/// loads and stores the same bits.
pub unsafe trait Elem: Copy + Send + Sync + PartialEq + core::fmt::Debug + 'static {
    /// The atomic type of the element's size.
    type Atom: Atom<Self>;
}

/// The atomic twin of a plain integer: its own atomic type.
macro_rules! atom_of_int {
    ($($t:ty => $a:ty),* $(,)?) => {$(
        impl Atom<$t> for $a {
            fn load(&self, order: Ordering) -> $t {
                <$a>::load(self, order)
            }

            fn store(&self, v: $t, order: Ordering) {
                <$a>::store(self, v, order);
            }
        }
    )*};
}

atom_of_int!(u8 => AtomicU8, u16 => AtomicU16, u32 => AtomicU32, i32 => AtomicI32, u64 => AtomicU64);

impl Atom<f32> for AtomicU32 {
    fn load(&self, order: Ordering) -> f32 {
        f32::from_bits(AtomicU32::load(self, order))
    }

    fn store(&self, v: f32, order: Ordering) {
        AtomicU32::store(self, v.to_bits(), order);
    }
}

impl Atom<f64> for AtomicU64 {
    fn load(&self, order: Ordering) -> f64 {
        f64::from_bits(AtomicU64::load(self, order))
    }

    fn store(&self, v: f64, order: Ordering) {
        AtomicU64::store(self, v.to_bits(), order);
    }
}

/// `Elem` for the primitive types, each with its atomic twin.
macro_rules! elem_of_primitive {
    ($($t:ty => $a:ty),* $(,)?) => {$(
        // SAFETY: a primitive type has no padding and every bit pattern of it is
        // a value of the type (some of a float's are NaN); `$a` has the type's
        // size and alignment (`twins`, asserted below) and loads and stores its
        // bits (a float's through `to_bits`/`from_bits`); an `f16` or `bf16`
        // travels as the `u16` of its bits.
        unsafe impl Elem for $t {
            type Atom = $a;
        }
    )*};
}

elem_of_primitive!(
    u8 => AtomicU8,
    u16 => AtomicU16,
    u32 => AtomicU32,
    i32 => AtomicI32,
    u64 => AtomicU64,
    f32 => AtomicU32,
    f64 => AtomicU64,
);

/// Whether `T`'s atomic twin has its size and alignment, and `T` fits a
/// buffer's alignment: the layout half of [`Elem`]'s contract.
const fn twins<T: Elem>() -> bool {
    size_of::<T>() == size_of::<T::Atom>()
        && align_of::<T>() == align_of::<T::Atom>()
        && align_of::<T>() <= BUFFER_ALIGN
}

const _: () = assert!(
    twins::<u8>()
        && twins::<u16>()
        && twins::<u32>()
        && twins::<i32>()
        && twins::<u64>()
        && twins::<f32>()
        && twins::<f64>(),
    "an element's atomic twin does not have its layout"
);

/// A count of elements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Elems(pub usize);

impl Elems {
    /// The count of nothing.
    pub const ZERO: Self = Self(0);

    /// The count as a plain number.
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }

    /// The bytes `self` elements of `T` take, refused by name when the
    /// product wraps.
    pub fn bytes_of<T: Elem>(self) -> Result<Bytes, DeviceError> {
        let size = size_of::<T>();
        self.0
            .checked_mul(size)
            .map(Bytes)
            .ok_or_else(|| DeviceError::Range {
                what: "elems",
                detail: format!("{} elements of {size} bytes wrap", self.0),
            })
    }

    /// The sum, or `None` when it would wrap.
    #[must_use]
    pub fn checked_add(self, other: Self) -> Option<Self> {
        self.0.checked_add(other.0).map(Self)
    }

    /// The sum, refused by name when it would wrap.
    pub fn try_add(self, other: Self) -> Result<Self, DeviceError> {
        self.checked_add(other).ok_or_else(|| DeviceError::Range {
            what: "elems",
            detail: format!("{} + {} element counts wrap", self.0, other.0),
        })
    }
}

/// A count of bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bytes(pub usize);

impl Bytes {
    /// The count of nothing.
    pub const ZERO: Self = Self(0);

    /// The count as a plain number.
    #[must_use]
    pub fn get(self) -> usize {
        self.0
    }

    /// The elements `self` bytes hold, or `None` when they do not divide into
    /// whole elements of `T`.
    #[must_use]
    pub fn elems_of<T: Elem>(self) -> Option<Elems> {
        let size = size_of::<T>();
        if size == 0 || !self.0.is_multiple_of(size) {
            return None;
        }
        Some(Elems(self.0 / size))
    }
}

/// A backend's own error, kept whole: its kind, its driver code, its chain.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// What a device call can fail with, split by what the caller does with the
/// failure: a backend that failed, a range that was refused, a machine that
/// cannot run this backend at all, and a fault a kernel raised.
#[derive(Debug)]
pub enum DeviceError {
    /// The backend refused or failed: a driver result, an Objective-C error,
    /// a command buffer's status. `source` is the backend's own error, so a
    /// backend that converts it back gets the error it gave.
    Backend {
        /// What was being done.
        what: &'static str,
        /// The backend's own error.
        source: BoxError,
    },
    /// A window outside a buffer, a length that does not match, or an offset
    /// the backend cannot align.
    Range {
        /// What was being taken.
        what: &'static str,
        /// The range and the buffer's size.
        detail: String,
    },
    /// No device, or an operating system below the floor this backend needs.
    Unsupported {
        /// What was being opened.
        what: &'static str,
        /// The condition that refused it.
        detail: String,
    },
    /// A kernel raised the fault word: the caller condemns the state the
    /// faulting work wrote.
    Fault {
        /// The fault the word held.
        fault: Fault,
        /// The backend's own error naming it, when the backend made one.
        source: Option<BoxError>,
    },
}

impl DeviceError {
    /// The error a backend failure names.
    pub fn backend(what: &'static str, source: impl Into<BoxError>) -> Self {
        Self::Backend {
            what,
            source: source.into(),
        }
    }

    /// The error an unsupported machine names.
    pub fn unsupported(what: &'static str, detail: String) -> Self {
        Self::Unsupported { what, detail }
    }

    /// The error of a fault the word held, with no backend error of its own.
    #[must_use]
    pub fn fault(fault: Fault) -> Self {
        Self::Fault {
            fault,
            source: None,
        }
    }
}

impl core::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Backend { what, source } => write!(f, "{what}: {source}"),
            Self::Range { what, detail } | Self::Unsupported { what, detail } => {
                write!(f, "{what}: {detail}")
            }
            Self::Fault {
                source: Some(source),
                ..
            } => write!(f, "{source}"),
            Self::Fault {
                fault,
                source: None,
            } => write!(f, "device fault at {fault:#}"),
        }
    }
}

impl std::error::Error for DeviceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend { source, .. }
            | Self::Fault {
                source: Some(source),
                ..
            } => Some(&**source),
            _ => None,
        }
    }
}

/// The refusal of a window outside what holds it: elements `at..at+len` of a
/// `total`-element `of` (a buffer or a window). The one owner of the range
/// rule: every backend's `Buffer::window` and [`Window::sub`] go through it,
/// so a window past the end, or one whose end wraps, is refused by name and
/// never clamped.
pub fn check_window(
    of: &'static str,
    total: Elems,
    at: Elems,
    len: Elems,
) -> Result<(), DeviceError> {
    let end = at.checked_add(len).ok_or_else(|| DeviceError::Range {
        what: "window",
        detail: format!("elements {} and a length of {} wrap", at.get(), len.get()),
    })?;
    if end > total {
        return Err(DeviceError::Range {
            what: "window",
            detail: format!(
                "elements {}..{} pass the {of}'s {} elements",
                at.get(),
                end.get(),
                total.get()
            ),
        });
    }
    Ok(())
}

/// The refusal of a copy whose two sides differ in length: `got` host or
/// window elements against the `want` of the window they are copied to or
/// from. The one owner of the length rule of [`Device::read`],
/// [`Device::write`], [`Device::copy`] and the shared pages.
pub fn check_len(what: &'static str, got: usize, want: Elems) -> Result<(), DeviceError> {
    if got != want.get() {
        return Err(DeviceError::Range {
            what,
            detail: format!(
                "{got} elements do not match the window's {} elements",
                want.get()
            ),
        });
    }
    Ok(())
}

/// The bytes a new buffer of `n` elements of `T` takes: a buffer of no
/// elements, or one whose byte count wraps, is refused by name. Every
/// backend's allocation goes through it, so the refusal is one.
pub fn buffer_bytes<T: Elem>(n: Elems) -> Result<Bytes, DeviceError> {
    let bytes = n.bytes_of::<T>()?;
    if bytes == Bytes::ZERO {
        return Err(DeviceError::Range {
            what: "buffer",
            detail: "a buffer of no elements is refused".to_string(),
        });
    }
    Ok(bytes)
}

/// A device an engine programs against.
pub trait Device: Sized {
    /// The device's own buffer, the one a kernel binds.
    type Buffer<T: Elem>: Buffer<T, Device = Self>;
    /// The device's own host-visible buffer: the same pages the device reads.
    type Shared<T: Elem>: Shared<T, Device = Self>;
    /// The stream the device's work is ordered on.
    type Stream: Stream<Event = Self::Event>;
    /// The event a stream's work is waited for by.
    type Event: Event;
    /// A buffer as a kernel binds it: a device address, or a buffer and its
    /// base offset; the same whatever element type a window reads it as. It
    /// carries no lifetime: a [`Window`] holds the borrow that keeps the
    /// buffer alive, and [`Window::raw`] lends the raw for no longer than the
    /// window is borrowed.
    type Raw: Copy + Send + Sync;
    /// The backend's name, as a record prints it.
    const BACKEND: &'static str;

    /// The device as the census names it now: its free bytes are read when
    /// asked.
    fn info(&self) -> Result<DeviceInfo, DeviceError>;

    /// The device's own stream: the one its engine's work is ordered on.
    fn stream(&self) -> &Self::Stream;

    /// A buffer of `n` elements, all zero once `s`'s work so far is done. A
    /// buffer of no elements is refused by name ([`buffer_bytes`]).
    fn zeroed<T: Elem>(&self, s: &Self::Stream, n: Elems) -> Result<Self::Buffer<T>, DeviceError>;

    /// A buffer holding `host`'s elements, waited for: `host` is not read
    /// after the call returns. An empty `host` is refused by name.
    fn upload<T: Elem>(&self, s: &Self::Stream, host: &[T])
    -> Result<Self::Buffer<T>, DeviceError>;

    /// A buffer the host and the device share, zeroed.
    fn shared<T: Elem>(&self, n: Elems) -> Result<Self::Shared<T>, DeviceError>;

    /// Device to host, waited for: `from`'s elements once `s`'s work so far
    /// is done. `into` holds exactly the window's length; any other length is
    /// refused by name ([`check_len`]).
    fn read<T: Elem>(
        &self,
        s: &Self::Stream,
        from: Window<'_, T, Self>,
        into: &mut [T],
    ) -> Result<(), DeviceError>;

    /// Host to device, waited for: `to` holds `from`'s elements for the work
    /// enqueued on `s` after the call, and `from` is not read after it
    /// returns. Any other length is refused by name.
    fn write<T: Elem>(
        &self,
        s: &Self::Stream,
        to: WindowMut<'_, T, Self>,
        from: &[T],
    ) -> Result<(), DeviceError>;

    /// Device to device, enqueued on `s`: `to` holds `from`'s elements for
    /// the work enqueued on `s` after the call. The two windows never overlap
    /// (the borrows keep them apart); any other length is refused by name.
    fn copy<T: Elem>(
        &self,
        s: &Self::Stream,
        from: Window<'_, T, Self>,
        to: WindowMut<'_, T, Self>,
    ) -> Result<(), DeviceError>;

    /// The device's fault word, read and cleared against a stream.
    fn fault(&self) -> &dyn FaultWord<Self::Stream>;
}

/// The stream of a device: the type its work is ordered on.
pub type DeviceStream<D> = <D as Device>::Stream;

/// A buffer the device holds.
pub trait Buffer<T: Elem>: Send + Sync {
    /// The device the buffer belongs to.
    type Device: Device;

    /// The buffer's length in elements.
    fn len(&self) -> Elems;

    /// Whether the buffer holds no elements.
    fn is_empty(&self) -> bool {
        self.len() == Elems::ZERO
    }

    /// Elements `at..at+len`; outside the buffer is refused by name
    /// ([`check_window`]).
    fn window(&self, at: Elems, len: Elems) -> Result<Window<'_, T, Self::Device>, DeviceError>;

    /// Elements `at..at+len`, held exclusively; outside the buffer is refused
    /// by name ([`check_window`]).
    fn window_mut(
        &mut self,
        at: Elems,
        len: Elems,
    ) -> Result<WindowMut<'_, T, Self::Device>, DeviceError>;

    /// The whole buffer as one window.
    fn whole(&self) -> Result<Window<'_, T, Self::Device>, DeviceError> {
        self.window(Elems::ZERO, self.len())
    }

    /// The whole buffer as one window, held exclusively.
    fn whole_mut(&mut self) -> Result<WindowMut<'_, T, Self::Device>, DeviceError> {
        let len = self.len();
        self.window_mut(Elems::ZERO, len)
    }
}

/// A window of a buffer: elements `at..at+len` of it.
pub struct Window<'a, T: Elem, D: Device> {
    raw: D::Raw,
    at: Elems,
    len: Elems,
    _t: PhantomData<&'a [T]>,
}

// By hand: a derive would ask `D` itself to be `Clone` and `Copy`, and a
// device is neither. A window holds only `D::Raw` (which is `Copy`) and counts.
impl<T: Elem, D: Device> Clone for Window<'_, T, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Elem, D: Device> Copy for Window<'_, T, D> {}

impl<'a, T: Elem, D: Device> Window<'a, T, D> {
    /// A window over `raw` starting at `at`, `len` elements long.
    ///
    /// # Safety
    /// `raw` covers `at + len` elements of `T` and outlives `'a`.
    pub unsafe fn from_raw(raw: D::Raw, at: Elems, len: Elems) -> Self {
        Self {
            raw,
            at,
            len,
            _t: PhantomData,
        }
    }

    /// The window's length in elements.
    #[must_use]
    pub fn len(self) -> Elems {
        self.len
    }

    /// Whether the window holds no elements.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.len == Elems::ZERO
    }

    /// The window's byte range inside the buffer it was taken from, refused
    /// by name when a byte count wraps.
    pub fn byte_range(self) -> Result<Range<Bytes>, DeviceError> {
        byte_range_of::<T>(self.at, self.len)
    }

    /// The elements `at..at+len` of this window; outside it is refused by
    /// name ([`check_window`]).
    pub fn sub(self, at: Elems, len: Elems) -> Result<Self, DeviceError> {
        check_window("window", self.len, at, len)?;
        Ok(Self {
            raw: self.raw,
            at: self.at.try_add(at)?,
            len,
            _t: PhantomData,
        })
    }

    /// The same bytes as elements of `P`: a byte offset or a byte length that
    /// is not a whole number of `P` is refused by name ([`cast_counts`]).
    pub fn cast<P: Elem>(self) -> Result<Window<'a, P, D>, DeviceError> {
        let (at, len) = cast_counts::<T, P>(self.at, self.len)?;
        Ok(Window {
            raw: self.raw,
            at,
            len,
            _t: PhantomData,
        })
    }

    /// The window's place in its buffer, without the borrow.
    #[must_use]
    pub fn span(self) -> Span<T> {
        Span::new(self.at, self.len)
    }

    /// The window as a kernel binds it, for as long as this window is
    /// borrowed.
    #[must_use]
    pub fn raw(&self) -> &D::Raw {
        &self.raw
    }
}

/// A window of a buffer held exclusively: the one a kernel or a copy writes.
pub struct WindowMut<'a, T: Elem, D: Device> {
    raw: D::Raw,
    at: Elems,
    len: Elems,
    _t: PhantomData<&'a mut [T]>,
}

impl<'a, T: Elem, D: Device> WindowMut<'a, T, D> {
    /// A window over `raw` starting at `at`, `len` elements long.
    ///
    /// # Safety
    /// `raw` covers `at + len` elements of `T` and outlives `'a`, and no
    /// other window or borrow of those elements is live for `'a`.
    pub unsafe fn from_raw(raw: D::Raw, at: Elems, len: Elems) -> Self {
        Self {
            raw,
            at,
            len,
            _t: PhantomData,
        }
    }

    /// The window's length in elements.
    #[must_use]
    pub fn len(&self) -> Elems {
        self.len
    }

    /// Whether the window holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == Elems::ZERO
    }

    /// The window's byte range inside the buffer it was taken from, refused
    /// by name when a byte count wraps.
    pub fn byte_range(&self) -> Result<Range<Bytes>, DeviceError> {
        byte_range_of::<T>(self.at, self.len)
    }

    /// Two disjoint windows at `mid`: the left `mid` elements and the rest.
    /// A `mid` past the window is refused by name.
    pub fn split_at(self, mid: Elems) -> Result<(Self, Self), DeviceError> {
        check_window("window", self.len, Elems::ZERO, mid)?;
        let left = Self {
            raw: self.raw,
            at: self.at,
            len: mid,
            _t: PhantomData,
        };
        let right = Self {
            raw: self.raw,
            at: self.at.try_add(mid)?,
            len: Elems(self.len.get() - mid.get()),
            _t: PhantomData,
        };
        Ok((left, right))
    }

    /// The same bytes as elements of `P`, held exclusively: refused by name
    /// as [`Window::cast`] is.
    pub fn cast<P: Elem>(self) -> Result<WindowMut<'a, P, D>, DeviceError> {
        let (at, len) = cast_counts::<T, P>(self.at, self.len)?;
        Ok(WindowMut {
            raw: self.raw,
            at,
            len,
            _t: PhantomData,
        })
    }

    /// The window's place in its buffer, without the borrow.
    #[must_use]
    pub fn span(&self) -> Span<T> {
        Span::new(self.at, self.len)
    }

    /// The same window, read-only, for as long as this window is borrowed:
    /// this window cannot be split or consumed while the read window lives.
    ///
    /// ```
    /// use bloomery_device::{Buffer, Device, DeviceError, Elems};
    ///
    /// fn read_then_split<D: Device>(buffer: &mut D::Buffer<f32>) -> Result<(), DeviceError> {
    ///     let window = buffer.window_mut(Elems(0), Elems(4))?;
    ///     let read = window.as_read();
    ///     let n = read.len();
    ///     let (left, right) = window.split_at(Elems(2))?;
    ///     let _ = (n, left.len(), right.len());
    ///     Ok(())
    /// }
    /// ```
    ///
    /// A read window that outlives a split is refused by the compiler:
    ///
    /// ```compile_fail,E0505
    /// use bloomery_device::{Buffer, Device, DeviceError, Elems};
    ///
    /// fn split_under_a_read<D: Device>(buffer: &mut D::Buffer<f32>) -> Result<(), DeviceError> {
    ///     let window = buffer.window_mut(Elems(0), Elems(4))?;
    ///     let read = window.as_read();
    ///     let (left, _right) = window.split_at(Elems(2))?;
    ///     let _ = (read.len(), left.len());
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    pub fn as_read(&self) -> Window<'_, T, D> {
        Window {
            raw: self.raw,
            at: self.at,
            len: self.len,
            _t: PhantomData,
        }
    }

    /// The window as a kernel binds it, for as long as this window is
    /// borrowed.
    #[must_use]
    pub fn raw(&self) -> &D::Raw {
        &self.raw
    }
}

/// The byte range of elements `at..at+len` of `T`.
fn byte_range_of<T: Elem>(at: Elems, len: Elems) -> Result<Range<Bytes>, DeviceError> {
    Ok(at.bytes_of::<T>()?..at.try_add(len)?.bytes_of::<T>()?)
}

/// Elements `at..at+len` of `T` counted as elements of `P`: the offset and
/// the length of the same bytes, each refused by name when it is not a whole
/// number of `P`. A buffer starts at [`BUFFER_ALIGN`] or more, so a whole
/// element offset is an aligned one.
fn cast_counts<T: Elem, P: Elem>(at: Elems, len: Elems) -> Result<(Elems, Elems), DeviceError> {
    let range = byte_range_of::<T>(at, len)?;
    let whole = |bytes: Bytes, of: &str| {
        bytes.elems_of::<P>().ok_or_else(|| DeviceError::Range {
            what: "cast",
            detail: format!(
                "{of} {} is not a whole number of {}-byte elements",
                bytes.get(),
                size_of::<P>()
            ),
        })
    };
    let at_p = whole(range.start, "the byte offset")?;
    let len_p = whole(
        Bytes(range.end.get() - range.start.get()),
        "the byte length",
    )?;
    Ok((at_p, len_p))
}

/// A window's place in its buffer, without the borrow: elements `at..at+len`.
/// An owner that keeps a buffer beside the windows it cuts from it (an arena,
/// a buffer in parts) keeps their spans, and takes each window again from the
/// buffer when it binds it; the window is checked again against that buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span<T: Elem> {
    at: Elems,
    len: Elems,
    _t: PhantomData<fn() -> T>,
}

impl<T: Elem> Span<T> {
    /// Elements `at..at+len` of a buffer still to be named.
    #[must_use]
    pub fn new(at: Elems, len: Elems) -> Self {
        Self {
            at,
            len,
            _t: PhantomData,
        }
    }

    /// The span's first element.
    #[must_use]
    pub fn at(self) -> Elems {
        self.at
    }

    /// The span's length in elements.
    #[must_use]
    pub fn len(self) -> Elems {
        self.len
    }

    /// Whether the span holds no elements.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.len == Elems::ZERO
    }

    /// The span's window of `buffer`; outside it is refused by name.
    pub fn window<B: Buffer<T>>(self, buffer: &B) -> Result<Window<'_, T, B::Device>, DeviceError> {
        buffer.window(self.at, self.len)
    }

    /// The span's window of `buffer`, held exclusively; outside it is refused
    /// by name.
    pub fn window_mut<B: Buffer<T>>(
        self,
        buffer: &mut B,
    ) -> Result<WindowMut<'_, T, B::Device>, DeviceError> {
        buffer.window_mut(self.at, self.len)
    }
}

/// Host memory the device uses in place: CUDA's pinned and mapped pages,
/// Metal's `StorageModeShared` buffer — the same pages. One element is loaded
/// or stored atomically with an ordering, so a host flag protocol (data
/// stored, then a flag released; a flag acquired, then data loaded) is the
/// contract's own; a bulk access borrows the pages while no device work
/// touches them.
pub trait Shared<T: Elem>: Buffer<T> {
    /// The pages as the host sees them.
    fn pages(&self) -> &HostPages<T>;

    /// The element at `i`, an atomic load with `order`: the device or another
    /// host thread may be writing it.
    fn load(&self, i: usize, order: Ordering) -> Result<T, DeviceError> {
        self.pages().load(i, order)
    }

    /// The element at `i` becomes `v`, an atomic store with `order`.
    fn store(&self, i: usize, v: T, order: Ordering) -> Result<(), DeviceError> {
        self.pages().store(i, v, order)
    }

    /// The shared pages as the host sees them.
    ///
    /// # Safety
    /// No device work in flight and no host [`Shared::store`] writes this
    /// buffer while the slice lives.
    unsafe fn host(&self) -> &[T] {
        // SAFETY: the caller's contract is `as_slice`'s.
        unsafe { self.pages().as_slice() }
    }

    /// The shared pages as the host may write them.
    ///
    /// # Safety
    /// No device work in flight reads or writes this buffer while the slice
    /// lives.
    unsafe fn host_mut(&mut self) -> &mut [T];
}

/// The order the device's work runs in. A stream is shared by the threads
/// that enqueue on it (a host tier's and the engine's), so it is `Sync`.
pub trait Stream: Send + Sync {
    /// The event this stream's work is waited for by.
    type Event: Event;

    /// An event that fires when the work recorded so far is done.
    fn record(&self) -> Result<Self::Event, DeviceError>;

    /// The work enqueued after this point waits for `e`.
    fn wait(&self, e: &Self::Event) -> Result<(), DeviceError>;

    /// Every work enqueued on this stream so far is done. Every pending piece
    /// of work is waited for even when one fails, and the first failure is
    /// returned.
    fn sync(&self) -> Result<(), DeviceError>;
}

/// A point in a stream's order.
pub trait Event: Send + Sync {
    /// The event has fired.
    fn sync(&self) -> Result<(), DeviceError>;

    /// Whether the event has fired.
    fn done(&self) -> Result<bool, DeviceError>;
}

/// Every device a backend opens, in the placement's terms.
pub trait Census {
    /// The devices this backend can see.
    fn census() -> Result<Vec<DeviceInfo>, DeviceError>;
}

#[cfg(test)]
mod tests {
    use super::fault::{self, FAULT_NONE, FAULT_WORDS, FaultSite};
    use super::*;

    /// A device that keeps its buffers on the host: the contract's own
    /// arithmetic, checked without a card. It moves no data: its copies check
    /// their lengths and their windows and stop there. It is neither `Clone`
    /// nor `Copy`, as no real device is.
    struct MockDevice {
        stream: MockStream,
    }

    impl MockDevice {
        fn new() -> Self {
            Self { stream: MockStream }
        }
    }

    /// The mock's raw: the buffer's byte count, which covers any window taken
    /// from it, whatever element type it is read as.
    type MockRaw = usize;

    impl Device for MockDevice {
        type Buffer<T: Elem> = MockBuffer<T>;
        type Shared<T: Elem> = MockShared<T>;
        type Stream = MockStream;
        type Event = MockEvent;
        type Raw = MockRaw;
        const BACKEND: &'static str = "mock";

        fn info(&self) -> Result<DeviceInfo, DeviceError> {
            Ok(DeviceInfo {
                ordinal: 0,
                name: "mock".to_string(),
                total_bytes: 1024,
                free_bytes: 1024,
                integrated: true,
                uuid: [0; 16],
                pci_bus: String::new(),
                held_by: None,
                compute_capability: None,
            })
        }

        fn stream(&self) -> &MockStream {
            &self.stream
        }

        fn zeroed<T: Elem>(
            &self,
            _s: &Self::Stream,
            n: Elems,
        ) -> Result<Self::Buffer<T>, DeviceError> {
            buffer_bytes::<T>(n)?;
            Ok(MockBuffer { data: zeros(n) })
        }

        fn upload<T: Elem>(
            &self,
            _s: &Self::Stream,
            host: &[T],
        ) -> Result<Self::Buffer<T>, DeviceError> {
            buffer_bytes::<T>(Elems(host.len()))?;
            Ok(MockBuffer {
                data: host.to_vec(),
            })
        }

        fn shared<T: Elem>(&self, n: Elems) -> Result<Self::Shared<T>, DeviceError> {
            buffer_bytes::<T>(n)?;
            Ok(MockShared::new(n))
        }

        fn read<T: Elem>(
            &self,
            _s: &MockStream,
            from: Window<'_, T, Self>,
            into: &mut [T],
        ) -> Result<(), DeviceError> {
            check_len("read", into.len(), from.len())?;
            within(*from.raw(), from.byte_range()?)
        }

        fn write<T: Elem>(
            &self,
            _s: &MockStream,
            to: WindowMut<'_, T, Self>,
            from: &[T],
        ) -> Result<(), DeviceError> {
            check_len("write", from.len(), to.len())?;
            within(*to.raw(), to.byte_range()?)
        }

        fn copy<T: Elem>(
            &self,
            _s: &MockStream,
            from: Window<'_, T, Self>,
            to: WindowMut<'_, T, Self>,
        ) -> Result<(), DeviceError> {
            check_len("copy", from.len().get(), to.len())?;
            within(*from.raw(), from.byte_range()?)?;
            within(*to.raw(), to.byte_range()?)
        }

        fn fault(&self) -> &dyn FaultWord<Self::Stream> {
            unimplemented!("the mock holds no fault word")
        }
    }

    /// A window's bytes lie inside the buffer its raw names: what a backend's
    /// copy would rely on, and what a cast or a sub-window must keep.
    fn within(raw: MockRaw, range: Range<Bytes>) -> Result<(), DeviceError> {
        if range.end.get() > raw {
            return Err(DeviceError::Range {
                what: "mock",
                detail: format!("bytes {range:?} pass the buffer's {raw}"),
            });
        }
        Ok(())
    }

    fn zeros<T: Elem>(n: Elems) -> Vec<T> {
        (0..n.get())
            .map(|_| {
                // SAFETY: `Elem`'s contract is that every bit pattern of the
                // type is a value of it, so the all-zero one is.
                unsafe { core::mem::zeroed::<T>() }
            })
            .collect()
    }

    /// The mock's `Buffer` body for a host vector: the window and the
    /// exclusive window, over the contract's one range check.
    fn window_of<T: Elem>(
        data: &[T],
        at: Elems,
        len: Elems,
    ) -> Result<Window<'_, T, MockDevice>, DeviceError> {
        check_window("buffer", Elems(data.len()), at, len)?;
        // SAFETY: the mock's raw is the buffer's byte count, which covers
        // `at + len` elements after the check, and the borrow of the buffer
        // outlives the window.
        Ok(unsafe { Window::from_raw(size_of_val(data), at, len) })
    }

    fn window_mut_of<T: Elem>(
        data: &mut [T],
        at: Elems,
        len: Elems,
    ) -> Result<WindowMut<'_, T, MockDevice>, DeviceError> {
        check_window("buffer", Elems(data.len()), at, len)?;
        // SAFETY: the mock's raw is the buffer's byte count, which covers
        // `at + len` elements after the check, and the exclusive borrow of
        // the buffer outlives the window.
        Ok(unsafe { WindowMut::from_raw(size_of_val(data), at, len) })
    }

    struct MockBuffer<T: Elem> {
        data: Vec<T>,
    }

    impl<T: Elem> Buffer<T> for MockBuffer<T> {
        type Device = MockDevice;

        fn len(&self) -> Elems {
            Elems(self.data.len())
        }

        fn window(&self, at: Elems, len: Elems) -> Result<Window<'_, T, MockDevice>, DeviceError> {
            window_of(&self.data, at, len)
        }

        fn window_mut(
            &mut self,
            at: Elems,
            len: Elems,
        ) -> Result<WindowMut<'_, T, MockDevice>, DeviceError> {
            window_mut_of(&mut self.data, at, len)
        }
    }

    /// A shared buffer over a vector it owns: its pages are the vector's.
    struct MockShared<T: Elem> {
        data: Vec<T>,
        pages: HostPages<T>,
    }

    impl<T: Elem> MockShared<T> {
        fn new(n: Elems) -> Self {
            let mut data = zeros::<T>(n);
            let ptr = core::ptr::NonNull::new(data.as_mut_ptr()).expect("a vector's pointer");
            // SAFETY: the vector's heap block holds `n` aligned elements and
            // moves with the vector, which lives beside the pages; the mock
            // reaches it only through them.
            let pages = unsafe { HostPages::new(ptr, n) };
            Self { data, pages }
        }
    }

    impl<T: Elem> Buffer<T> for MockShared<T> {
        type Device = MockDevice;

        fn len(&self) -> Elems {
            self.pages.len()
        }

        fn window(&self, at: Elems, len: Elems) -> Result<Window<'_, T, MockDevice>, DeviceError> {
            window_of(&self.data, at, len)
        }

        fn window_mut(
            &mut self,
            at: Elems,
            len: Elems,
        ) -> Result<WindowMut<'_, T, MockDevice>, DeviceError> {
            window_mut_of(&mut self.data, at, len)
        }
    }

    impl<T: Elem> Shared<T> for MockShared<T> {
        fn pages(&self) -> &HostPages<T> {
            &self.pages
        }

        unsafe fn host_mut(&mut self) -> &mut [T] {
            // SAFETY: the mock runs no device work.
            unsafe { self.pages.as_mut_slice() }
        }
    }

    struct MockStream;

    impl Stream for MockStream {
        type Event = MockEvent;

        fn record(&self) -> Result<Self::Event, DeviceError> {
            Ok(MockEvent)
        }

        fn wait(&self, _e: &Self::Event) -> Result<(), DeviceError> {
            Ok(())
        }

        fn sync(&self) -> Result<(), DeviceError> {
            Ok(())
        }
    }

    struct MockEvent;

    impl Event for MockEvent {
        fn sync(&self) -> Result<(), DeviceError> {
            Ok(())
        }

        fn done(&self) -> Result<bool, DeviceError> {
            Ok(true)
        }
    }

    fn refusal<T>(result: Result<T, DeviceError>, what: &str) -> String {
        match result {
            Err(DeviceError::Range { what: got, detail }) => {
                assert_eq!(got, what, "{detail}");
                detail
            }
            Err(other) => panic!("a {what} refusal, not {other:?}"),
            Ok(_) => panic!("the {what} was accepted"),
        }
    }

    fn range_refusal(result: Result<(), DeviceError>) -> String {
        refusal(result, "window")
    }

    #[test]
    fn the_range_check_holds_at_the_edges_and_one_past() {
        let total = Elems(4);
        // Inside, up to and including the end, and an empty window at it.
        for (at, len) in [(2, 2), (4, 0), (0, 4), (0, 0), (3, 1)] {
            check_window("buffer", total, Elems(at), Elems(len))
                .unwrap_or_else(|e| panic!("{at}+{len} of 4 is inside: {e}"));
        }
        // One element past the end, and the whole range one past.
        let detail = range_refusal(check_window("buffer", total, Elems(3), Elems(2)));
        assert!(detail.contains("3..5"), "{detail}");
        assert!(detail.contains("buffer's 4 elements"), "{detail}");
        range_refusal(check_window("buffer", total, Elems(0), Elems(5)));
        range_refusal(check_window("buffer", total, Elems(5), Elems(0)));
        // A sum that wraps is refused by name, not folded back into range.
        let detail = range_refusal(check_window("buffer", total, Elems(usize::MAX), Elems(2)));
        assert!(detail.contains("wrap"), "{detail}");
        let detail = range_refusal(check_window("window", total, Elems(3), Elems(usize::MAX)));
        assert!(detail.contains("wrap"), "{detail}");
    }

    #[test]
    fn a_buffers_window_goes_through_the_range_check() {
        let device = MockDevice::new();
        let mut buffer = device
            .zeroed::<f32>(device.stream(), Elems(4))
            .expect("the mock allocates");
        assert_eq!(buffer.len(), Elems(4));
        assert!(buffer.window(Elems(2), Elems(2)).is_ok());
        assert!(buffer.window_mut(Elems(2), Elems(2)).is_ok());
        assert_eq!(buffer.whole().expect("whole").len(), Elems(4));
        assert_eq!(buffer.whole_mut().expect("whole").len(), Elems(4));
        range_refusal(buffer.window(Elems(3), Elems(2)).map(|_| ()));
        range_refusal(buffer.window_mut(Elems(0), Elems(5)).map(|_| ()));
    }

    #[test]
    fn a_buffer_of_no_elements_is_refused_by_name() {
        let device = MockDevice::new();
        let s = device.stream();
        for detail in [
            refusal(device.zeroed::<f32>(s, Elems(0)), "buffer"),
            refusal(device.upload::<u8>(s, &[]), "buffer"),
            refusal(device.shared::<u64>(Elems(0)), "buffer"),
        ] {
            assert!(detail.contains("no elements"), "{detail}");
        }
        refusal(buffer_bytes::<f64>(Elems(usize::MAX)), "elems");
        assert_eq!(buffer_bytes::<u16>(Elems(3)).expect("fits"), Bytes(6));
    }

    #[test]
    fn a_sub_window_is_taken_inside_its_window_and_composes_its_position() {
        let device = MockDevice::new();
        let buffer = device
            .zeroed::<f32>(device.stream(), Elems(6))
            .expect("the mock allocates");
        let window = buffer.window(Elems(1), Elems(4)).expect("inside");
        let sub = window.sub(Elems(1), Elems(2)).expect("inside");
        // Elements 2..4 of the buffer.
        assert_eq!(sub.len(), Elems(2));
        assert_eq!(sub.byte_range().expect("fits"), Bytes(8)..Bytes(16));
        // The whole window and the empty window at its end are inside it.
        assert!(window.sub(Elems(0), Elems(4)).is_ok());
        assert!(window.sub(Elems(4), Elems(0)).is_ok());

        let detail = range_refusal(window.sub(Elems(1), Elems(4)).map(|_| ()));
        assert!(detail.contains("1..5"), "{detail}");
        assert!(detail.contains("window's 4 elements"), "{detail}");
    }

    #[test]
    fn a_window_is_copy_even_when_the_device_is_not() {
        let device = MockDevice::new();
        let buffer = device
            .zeroed::<f32>(device.stream(), Elems(4))
            .expect("the mock allocates");
        let window = buffer.window(Elems(1), Elems(2)).expect("inside");
        // One window, used twice: every by-value method copies it.
        let len = window.len();
        let range = window.byte_range().expect("fits");
        let sub = window.sub(Elems(0), Elems(1)).expect("inside");
        assert_eq!(len, Elems(2));
        assert_eq!(range, Bytes(4)..Bytes(12));
        assert_eq!(sub.len(), Elems(1));
        assert_eq!(*window.raw(), 16);
        assert!(!window.is_empty());
    }

    #[test]
    fn a_split_gives_two_disjoint_windows() {
        let device = MockDevice::new();
        let mut buffer = device
            .zeroed::<f32>(device.stream(), Elems(6))
            .expect("the mock allocates");
        let window = buffer.window_mut(Elems(1), Elems(4)).expect("inside");
        let (left, right) = window.split_at(Elems(2)).expect("inside");
        assert_eq!(left.len(), Elems(2));
        assert_eq!(right.len(), Elems(2));
        assert_eq!(left.byte_range().expect("fits"), Bytes(4)..Bytes(12));
        assert_eq!(right.byte_range().expect("fits"), Bytes(12)..Bytes(20));
        assert_eq!(
            left.as_read().byte_range().expect("fits"),
            Bytes(4)..Bytes(12)
        );

        // At the end: the right half is empty, and the split is inside.
        let window = buffer.window_mut(Elems(1), Elems(4)).expect("inside");
        let (all, none) = window.split_at(Elems(4)).expect("the end is inside");
        assert_eq!(all.len(), Elems(4));
        assert!(none.is_empty());

        let window = buffer.window_mut(Elems(1), Elems(4)).expect("inside");
        let detail = range_refusal(window.split_at(Elems(5)).map(|_| ()));
        assert!(detail.contains("elements 0..5 pass"), "{detail}");
    }

    #[test]
    fn a_cast_keeps_the_bytes_and_refuses_a_part_of_an_element() {
        let device = MockDevice::new();
        let mut arena = device
            .zeroed::<u8>(device.stream(), Elems(32))
            .expect("the mock allocates");
        // Bytes 8..24 as four f32s at element 2.
        let f = arena
            .window(Elems(8), Elems(16))
            .expect("inside")
            .cast::<f32>()
            .expect("whole f32s");
        assert_eq!(f.len(), Elems(4));
        assert_eq!(f.byte_range().expect("fits"), Bytes(8)..Bytes(24));
        assert_eq!(*f.raw(), 32);
        // And back: the same bytes as u8s, and as two u64s.
        let back = f.cast::<u8>().expect("bytes are whole");
        assert_eq!(back.byte_range().expect("fits"), Bytes(8)..Bytes(24));
        let wide = f.cast::<u64>().expect("whole u64s");
        assert_eq!(
            (wide.len(), wide.byte_range().expect("fits").start),
            (Elems(2), Bytes(8))
        );

        let at = refusal(
            arena
                .window(Elems(2), Elems(8))
                .expect("inside")
                .cast::<f32>(),
            "cast",
        );
        assert!(at.contains("byte offset 2"), "{at}");
        let len = refusal(
            arena
                .window(Elems(4), Elems(6))
                .expect("inside")
                .cast::<f32>(),
            "cast",
        );
        assert!(len.contains("byte length 6"), "{len}");
        refusal(
            f.sub(Elems(1), Elems(2)).expect("inside").cast::<u64>(),
            "cast",
        );

        let mutable = arena
            .window_mut(Elems(16), Elems(16))
            .expect("inside")
            .cast::<u32>()
            .expect("whole u32s");
        assert_eq!(mutable.byte_range().expect("fits"), Bytes(16)..Bytes(32));
        refusal(
            arena
                .window_mut(Elems(1), Elems(4))
                .expect("inside")
                .cast::<u16>(),
            "cast",
        );
    }

    #[test]
    fn a_span_is_taken_again_from_its_buffer_and_checked_there() {
        let device = MockDevice::new();
        let mut big = device
            .zeroed::<u32>(device.stream(), Elems(8))
            .expect("the mock allocates");
        let small = device
            .zeroed::<u32>(device.stream(), Elems(4))
            .expect("the mock allocates");
        let span = big.window(Elems(3), Elems(4)).expect("inside").span();
        assert_eq!((span.at(), span.len()), (Elems(3), Elems(4)));
        let again = span.window(&big).expect("inside its buffer");
        assert_eq!(again.byte_range().expect("fits"), Bytes(12)..Bytes(28));
        assert_eq!(
            span.window_mut(&mut big)
                .expect("inside")
                .byte_range()
                .expect("fits"),
            Bytes(12)..Bytes(28)
        );
        let detail = range_refusal(span.window(&small).map(|_| ()));
        assert!(detail.contains("3..7"), "{detail}");
        assert!(Span::<u32>::new(Elems(2), Elems(0)).is_empty());
    }

    #[test]
    fn a_copy_of_another_length_is_refused_by_name() {
        let device = MockDevice::new();
        let s = device.stream();
        let mut a = device
            .zeroed::<u64>(s, Elems(6))
            .expect("the mock allocates");
        let mut b = device
            .zeroed::<u64>(s, Elems(6))
            .expect("the mock allocates");
        let from = a.window(Elems(1), Elems(3)).expect("inside");
        device
            .read(s, from, &mut [0u64; 3])
            .expect("the lengths match");
        for wrong in [2usize, 4] {
            let detail = refusal(device.read(s, from, &mut vec![0u64; wrong]), "read");
            assert!(detail.contains("window's 3 elements"), "{detail}");
        }
        device
            .copy(s, from, b.window_mut(Elems(3), Elems(3)).expect("inside"))
            .expect("the lengths match");
        refusal(
            device.copy(s, from, b.window_mut(Elems(0), Elems(4)).expect("inside")),
            "copy",
        );
        device
            .write(
                s,
                a.window_mut(Elems(2), Elems(2)).expect("inside"),
                &[1, 2],
            )
            .expect("the lengths match");
        refusal(
            device.write(s, a.window_mut(Elems(2), Elems(2)).expect("inside"), &[1]),
            "write",
        );
    }

    #[test]
    fn element_and_byte_counts_are_different_types_and_the_product_is_checked() {
        assert_eq!(Elems(4).bytes_of::<f32>().expect("fits"), Bytes(16));
        assert_eq!(Bytes(16).elems_of::<f32>(), Some(Elems(4)));
        assert_eq!(Bytes(15).elems_of::<f32>(), None);
        assert_eq!(Bytes(16).elems_of::<u8>(), Some(Elems(16)));
        assert_eq!(Elems(0).bytes_of::<u16>().expect("fits"), Bytes(0));
        assert_eq!(Elems(3).bytes_of::<f64>().expect("fits"), Bytes(24));
        assert_eq!(Elems(3).checked_add(Elems(4)), Some(Elems(7)));

        // 2^62 + 1 f32s is 4 bytes after a wrapping multiply: refused by name.
        let wrapping = Elems((1usize << (usize::BITS - 2)) + 1);
        match wrapping.bytes_of::<f32>() {
            Err(DeviceError::Range { what, detail }) => {
                assert_eq!(what, "elems");
                assert!(detail.contains("wrap"), "{detail}");
            }
            other => panic!("a range refusal, not {other:?}"),
        }
        assert!(Elems(usize::MAX).try_add(Elems(1)).is_err());
    }

    #[test]
    fn a_shared_store_loads_back_and_an_index_past_the_end_is_refused() {
        let device = MockDevice::new();
        let shared = device.shared::<u32>(Elems(2)).expect("the mock allocates");
        shared.store(1, 7, Ordering::Release).expect("inside");
        assert_eq!(shared.load(1, Ordering::Acquire).expect("inside"), 7);
        // SAFETY: the mock runs no device work and nothing stores meanwhile.
        assert_eq!(unsafe { shared.host() }, &[0, 7]);
        refusal(shared.store(2, 1, Ordering::Relaxed), "store");
        refusal(shared.load(2, Ordering::Relaxed), "load");
    }

    #[test]
    fn an_element_loads_and_stores_through_its_atomic_twin() {
        let device = MockDevice::new();
        let floats = device.shared::<f64>(Elems(1)).expect("the mock allocates");
        floats.store(0, -0.0, Ordering::Relaxed).expect("inside");
        assert_eq!(
            floats.load(0, Ordering::Relaxed).expect("inside").to_bits(),
            (-0.0f64).to_bits()
        );
        let halves = device.shared::<u16>(Elems(3)).expect("the mock allocates");
        halves.store(2, 0xBEEF, Ordering::Relaxed).expect("inside");
        assert_eq!(halves.load(2, Ordering::Relaxed).expect("inside"), 0xBEEF);
        let signed = device.shared::<i32>(Elems(1)).expect("the mock allocates");
        signed.store(0, -5, Ordering::Relaxed).expect("inside");
        assert_eq!(signed.load(0, Ordering::Relaxed).expect("inside"), -5);
    }

    #[test]
    fn a_backend_error_keeps_its_source_and_a_fault_its_cause_once() {
        let backend = DeviceError::backend("buffer", "the device made no buffer".to_string());
        assert_eq!(backend.to_string(), "buffer: the device made no buffer");
        assert!(std::error::Error::source(&backend).is_some());

        let f = fault::Fault::at(7, FaultSite::ExpertId);
        let plain = DeviceError::fault(f).to_string();
        assert_eq!(
            plain.matches(FaultSite::ExpertId.cause()).count(),
            1,
            "{plain}"
        );
        let wrapped = DeviceError::Fault {
            fault: f,
            source: Some("step: device fault at layer 7".into()),
        };
        assert_eq!(wrapped.to_string(), "step: device fault at layer 7");
    }

    fn fresh_words() -> Vec<u32> {
        let mut words = vec![0; FAULT_WORDS];
        words[0] = FAULT_NONE;
        words
    }

    #[test]
    fn the_fault_word_keeps_the_lowest_site_and_the_mask() {
        let mut words = fresh_words();
        assert_eq!(fault::Fault::decode(&words).expect("long enough"), None);

        for site in [
            FaultSite::Router,
            FaultSite::TokenId,
            FaultSite::QuantColumn,
        ] {
            fault::raise(&mut words, 3, site).expect("long enough");
        }
        // A higher site raised after the lowest keeps the lowest.
        fault::raise(&mut words, 3, FaultSite::TokenId).expect("long enough");

        let fault = fault::Fault::decode(&words)
            .expect("long enough")
            .expect("a fault");
        assert_eq!(fault.layer, 3);
        assert_eq!(fault.code, FaultSite::QuantColumn as u32);
        assert_eq!(
            fault.sites,
            fault::site_mask(FaultSite::Router)
                | fault::site_mask(FaultSite::TokenId)
                | fault::site_mask(FaultSite::QuantColumn)
        );
        assert_eq!(fault.site(), Some(FaultSite::QuantColumn));
    }

    #[test]
    fn the_fault_word_keeps_the_earliest_layer() {
        let mut words = fresh_words();
        fault::raise(&mut words, 7, FaultSite::Logit).expect("long enough");
        fault::raise(&mut words, 2, FaultSite::Router).expect("long enough");
        // A later layer raised after the earliest keeps the earliest.
        fault::raise(&mut words, 7, FaultSite::Logit).expect("long enough");

        let fault = fault::Fault::decode(&words)
            .expect("long enough")
            .expect("a fault");
        assert_eq!(fault.layer, 2);
        assert_eq!(fault.code, FaultSite::Router as u32);
        assert_eq!(fault.word(), (2 << 8) | FaultSite::Router as u32);
        // Layer 7's own mask holds its own site.
        assert_eq!(
            words[fault::mask_index(7)],
            fault::site_mask(FaultSite::Logit)
        );
    }

    #[test]
    fn a_layer_past_the_last_names_the_fault_and_writes_no_mask() {
        let mut words = fresh_words();
        let past = LAYER_NONE + 1;
        fault::raise(&mut words, past, FaultSite::Router).expect("long enough");

        let fault = fault::Fault::decode(&words)
            .expect("long enough")
            .expect("the word still names a fault");
        assert_eq!(fault.layer, past);
        assert_eq!(fault.sites, 0);
        assert!(words[1..].iter().all(|m| *m == 0), "no mask was written");
    }

    #[test]
    fn a_fault_word_too_short_is_refused_by_name() {
        let mut words = vec![FAULT_NONE; FAULT_WORDS - 1];
        for result in [
            fault::Fault::decode(&words).map(|_| ()),
            fault::raise(&mut words, 0, FaultSite::Router),
        ] {
            match result {
                Err(DeviceError::Range { what, detail }) => {
                    assert_eq!(what, "fault word");
                    assert!(detail.contains(&FAULT_WORDS.to_string()), "{detail}");
                }
                other => panic!("a range refusal, not {other:?}"),
            }
        }
    }

    #[test]
    fn a_cleared_word_is_clean() {
        let mut words = fresh_words();
        fault::raise(&mut words, 1, FaultSite::Router).expect("long enough");
        assert!(fault::Fault::decode(&words).expect("long enough").is_some());

        words[0] = FAULT_NONE;
        words[fault::mask_index(1)] = 0;
        assert_eq!(fault::Fault::decode(&words).expect("long enough"), None);
    }
}
