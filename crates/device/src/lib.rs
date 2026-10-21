//! The device contract: the traits an engine programs against, whatever the
//! backend underneath is. Nothing here names a backend, an operating system
//! or a vendor type. A backend implements the traits; the engine never asks
//! which one it got.
//!
//! The counts are typed apart ([`Elems`] and [`Bytes`]): an element count and
//! a byte count are different numbers, and a signature that takes both says
//! which is which. A window outside its buffer is refused by name, never
//! clamped: a clamped window is a defined output for an undefined input.

pub mod fault;
pub mod step;

use core::marker::PhantomData;
use core::ops::Range;

pub use fault::{FAULT_NONE, Fault, FaultSite, FaultWord, LAYER_HEAD, LAYER_NONE};
pub use step::{ReplayShape, StepRecord, TokenReplay};

use bloomery_placement::placement::devices::DeviceInfo;

/// Plain data: every bit pattern a value, no padding.
///
/// # Safety
/// The implementor has no padding and no invalid bit pattern.
pub unsafe trait Elem: Copy + Send + Sync + 'static {}

// SAFETY: the primitive types have no padding, and every bit pattern of each
// is a value of the type.
unsafe impl Elem for u8 {}
// SAFETY: as above; a u16 holds f16's or bf16's bits.
unsafe impl Elem for u16 {}
// SAFETY: as above.
unsafe impl Elem for u32 {}
// SAFETY: as above.
unsafe impl Elem for i32 {}
// SAFETY: as above; every bit pattern of an f32 is a value (some are NaN).
unsafe impl Elem for f32 {}

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
        let size = core::mem::size_of::<T>();
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
        self.checked_add(other).ok_or(DeviceError::Range {
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
        let size = core::mem::size_of::<T>();
        if size == 0 || !self.0.is_multiple_of(size) {
            return None;
        }
        Some(Elems(self.0 / size))
    }
}

/// What a device call can fail with, split by what the caller does with the
/// failure: a backend that failed, a range that was refused, a machine that
/// cannot run this backend at all, and a fault a kernel raised.
#[derive(Debug)]
pub enum DeviceError {
    /// The backend refused or failed: a driver result, an Objective-C error,
    /// a command buffer's status.
    Backend {
        /// What was being done.
        what: &'static str,
        /// The backend's own words.
        detail: String,
    },
    /// A window outside a buffer, or an offset the backend cannot align.
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
    /// A kernel raised the fault word.
    Fault(Fault),
}

impl DeviceError {
    /// The error a refused range names, with the buffer's size in the text.
    pub fn range(what: &'static str, detail: String) -> Self {
        Self::Range { what, detail }
    }

    /// The error a backend failure names.
    pub fn backend(what: &'static str, detail: String) -> Self {
        Self::Backend { what, detail }
    }

    /// The error an unsupported machine names.
    pub fn unsupported(what: &'static str, detail: String) -> Self {
        Self::Unsupported { what, detail }
    }
}

impl core::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Backend { what, detail }
            | Self::Range { what, detail }
            | Self::Unsupported { what, detail } => write!(f, "{what}: {detail}"),
            Self::Fault(fault) => write!(f, "fault: {fault}"),
        }
    }
}

impl std::error::Error for DeviceError {}

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
    /// A buffer as a kernel binds it: a CUDA device address, a Metal buffer
    /// and its base offset. It carries no lifetime: a [`Window`] holds the
    /// borrow that keeps the buffer alive, and [`Window::raw`] lends the raw
    /// for no longer than the window is borrowed.
    type Raw: Copy + Send + Sync;
    /// The backend's name, as a record prints it.
    const BACKEND: &'static str;

    /// The device as the census names it.
    fn info(&self) -> &DeviceInfo;

    /// A buffer of `n` elements, all zero. A backend that cannot hold no
    /// bytes refuses `n == 0` by name.
    fn zeroed<T: Elem>(&self, s: &Self::Stream, n: Elems) -> Result<Self::Buffer<T>, DeviceError>;

    /// A buffer holding `host`'s elements. An empty `host` is refused by name
    /// where [`Device::zeroed`] refuses `n == 0`.
    fn upload<T: Elem>(&self, s: &Self::Stream, host: &[T])
    -> Result<Self::Buffer<T>, DeviceError>;

    /// A buffer the host and the device share.
    fn shared<T: Elem>(&self, n: Elems) -> Result<Self::Shared<T>, DeviceError>;

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

    /// Elements `at..at+len`, host-side; outside the buffer is refused by
    /// name ([`check_window`]).
    fn window_mut(
        &mut self,
        at: Elems,
        len: Elems,
    ) -> Result<WindowMut<'_, T, Self::Device>, DeviceError>;

    /// Device to host, waited for: the buffer's elements once `s`'s work so
    /// far is done. `into` holds exactly the buffer's length; any other
    /// length is refused by name. For the gates and the fault paths only.
    fn read(&self, s: &DeviceStream<Self::Device>, into: &mut [T]) -> Result<(), DeviceError>;
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

    /// The window as a kernel binds it, for as long as this window is
    /// borrowed.
    #[must_use]
    pub fn raw(&self) -> &D::Raw {
        &self.raw
    }
}

/// A window of a buffer the host may write.
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
        if mid > self.len {
            return Err(DeviceError::Range {
                what: "window",
                detail: format!(
                    "the split at element {} passes the window's {} elements",
                    mid.get(),
                    self.len.get()
                ),
            });
        }
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

/// Host memory the device uses in place: CUDA's pinned and mapped pages,
/// Metal's `StorageModeShared` buffer — the same pages.
pub trait Shared<T: Elem>: Buffer<T> {
    /// The element at `i`, read from the shared pages. Volatile: the device
    /// may be writing a word (a flag).
    fn load(&self, i: usize) -> Result<T, DeviceError>;

    /// The element at `i`, written to the shared pages. Exclusive on the
    /// host, so two host threads never write one element at once.
    fn store(&mut self, i: usize, v: T) -> Result<(), DeviceError>;

    /// The shared pages as the host sees them.
    ///
    /// # Safety
    /// No device work in flight writes this buffer.
    unsafe fn host(&self) -> &[T];

    /// The shared pages as the host may write them.
    ///
    /// # Safety
    /// No device work in flight reads or writes this buffer.
    unsafe fn host_mut(&mut self) -> &mut [T];
}

/// The order the device's work runs in.
pub trait Stream: Send {
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
pub trait Event: Send {
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
    /// arithmetic, checked without a card. It is neither `Clone` nor `Copy`,
    /// as no real device is.
    struct MockDevice {
        info: DeviceInfo,
    }

    impl MockDevice {
        fn new() -> Self {
            Self {
                info: DeviceInfo {
                    ordinal: 0,
                    name: "mock".to_string(),
                    total_bytes: 1024,
                    free_bytes: 1024,
                    integrated: true,
                    uuid: [0; 16],
                    pci_bus: String::new(),
                    held_by: None,
                },
            }
        }
    }

    /// The mock's raw: the buffer's element count, which covers any window
    /// taken from it.
    type MockRaw = usize;

    impl Device for MockDevice {
        type Buffer<T: Elem> = MockBuffer<T>;
        type Shared<T: Elem> = MockShared<T>;
        type Stream = MockStream;
        type Event = MockEvent;
        type Raw = MockRaw;
        const BACKEND: &'static str = "mock";

        fn info(&self) -> &DeviceInfo {
            &self.info
        }

        fn zeroed<T: Elem>(
            &self,
            _s: &Self::Stream,
            n: Elems,
        ) -> Result<Self::Buffer<T>, DeviceError> {
            Ok(MockBuffer { data: zeros(n) })
        }

        fn upload<T: Elem>(
            &self,
            _s: &Self::Stream,
            host: &[T],
        ) -> Result<Self::Buffer<T>, DeviceError> {
            Ok(MockBuffer {
                data: host.to_vec(),
            })
        }

        fn shared<T: Elem>(&self, n: Elems) -> Result<Self::Shared<T>, DeviceError> {
            Ok(MockShared { data: zeros(n) })
        }

        fn fault(&self) -> &dyn FaultWord<Self::Stream> {
            unimplemented!("the mock holds no fault word")
        }
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

    /// The mock's `Buffer` body for a host vector: the window, the mutable
    /// window and the read, over the contract's one range check.
    fn window_of<T: Elem>(
        data: &[T],
        at: Elems,
        len: Elems,
    ) -> Result<Window<'_, T, MockDevice>, DeviceError> {
        check_window("buffer", Elems(data.len()), at, len)?;
        // SAFETY: the mock's raw is the buffer's element count, which covers
        // `at + len` elements after the check, and the borrow of the buffer
        // outlives the window.
        Ok(unsafe { Window::from_raw(data.len(), at, len) })
    }

    fn window_mut_of<T: Elem>(
        data: &mut [T],
        at: Elems,
        len: Elems,
    ) -> Result<WindowMut<'_, T, MockDevice>, DeviceError> {
        check_window("buffer", Elems(data.len()), at, len)?;
        // SAFETY: as in `window_of`, and the exclusive borrow of the buffer
        // outlives the window.
        Ok(unsafe { WindowMut::from_raw(data.len(), at, len) })
    }

    fn read_of<T: Elem>(data: &[T], into: &mut [T]) -> Result<(), DeviceError> {
        if into.len() != data.len() {
            return Err(DeviceError::Range {
                what: "read",
                detail: format!(
                    "{} elements do not match the buffer's {} elements",
                    into.len(),
                    data.len()
                ),
            });
        }
        into.copy_from_slice(data);
        Ok(())
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

        fn read(&self, _s: &MockStream, into: &mut [T]) -> Result<(), DeviceError> {
            read_of(&self.data, into)
        }
    }

    struct MockShared<T: Elem> {
        data: Vec<T>,
    }

    impl<T: Elem> Buffer<T> for MockShared<T> {
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

        fn read(&self, _s: &MockStream, into: &mut [T]) -> Result<(), DeviceError> {
            read_of(&self.data, into)
        }
    }

    impl<T: Elem> Shared<T> for MockShared<T> {
        fn load(&self, i: usize) -> Result<T, DeviceError> {
            self.data.get(i).copied().ok_or(DeviceError::Range {
                what: "load",
                detail: format!(
                    "element {i} is past the buffer's {} elements",
                    self.data.len()
                ),
            })
        }

        fn store(&mut self, i: usize, v: T) -> Result<(), DeviceError> {
            let total = self.data.len();
            let slot = self.data.get_mut(i).ok_or(DeviceError::Range {
                what: "store",
                detail: format!("element {i} is past the buffer's {total} elements"),
            })?;
            *slot = v;
            Ok(())
        }

        unsafe fn host(&self) -> &[T] {
            self.data.as_slice()
        }

        unsafe fn host_mut(&mut self) -> &mut [T] {
            self.data.as_mut_slice()
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

    fn range_refusal(result: Result<(), DeviceError>) -> String {
        match result {
            Err(DeviceError::Range { what, detail }) => {
                assert_eq!(what, "window");
                detail
            }
            Err(other) => panic!("a range refusal, not {other:?}"),
            Ok(()) => panic!("the window was accepted"),
        }
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
        let stream = MockStream;
        let mut buffer = device
            .zeroed::<f32>(&stream, Elems(4))
            .expect("the mock allocates");
        assert_eq!(buffer.len(), Elems(4));
        assert!(buffer.window(Elems(2), Elems(2)).is_ok());
        assert!(buffer.window_mut(Elems(2), Elems(2)).is_ok());
        range_refusal(buffer.window(Elems(3), Elems(2)).map(|_| ()));
        range_refusal(buffer.window_mut(Elems(0), Elems(5)).map(|_| ()));
    }

    #[test]
    fn a_sub_window_is_taken_inside_its_window_and_composes_its_position() {
        let device = MockDevice::new();
        let stream = MockStream;
        let buffer = device
            .zeroed::<f32>(&stream, Elems(6))
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
        let stream = MockStream;
        let buffer = device
            .zeroed::<f32>(&stream, Elems(4))
            .expect("the mock allocates");
        let window = buffer.window(Elems(1), Elems(2)).expect("inside");
        // One window, used twice: every by-value method copies it.
        let len = window.len();
        let range = window.byte_range().expect("fits");
        let sub = window.sub(Elems(0), Elems(1)).expect("inside");
        assert_eq!(len, Elems(2));
        assert_eq!(range, Bytes(4)..Bytes(12));
        assert_eq!(sub.len(), Elems(1));
        assert_eq!(*window.raw(), 4);
        assert!(!window.is_empty());
    }

    #[test]
    fn a_split_gives_two_disjoint_windows() {
        let device = MockDevice::new();
        let stream = MockStream;
        let mut buffer = device
            .zeroed::<f32>(&stream, Elems(6))
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
        assert!(detail.contains("split at element 5"), "{detail}");
    }

    #[test]
    fn element_and_byte_counts_are_different_types_and_the_product_is_checked() {
        assert_eq!(Elems(4).bytes_of::<f32>().expect("fits"), Bytes(16));
        assert_eq!(Bytes(16).elems_of::<f32>(), Some(Elems(4)));
        assert_eq!(Bytes(15).elems_of::<f32>(), None);
        assert_eq!(Bytes(16).elems_of::<u8>(), Some(Elems(16)));
        assert_eq!(Elems(0).bytes_of::<u16>().expect("fits"), Bytes(0));
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
    fn a_buffer_reads_back_whole_and_a_shorter_read_is_refused() {
        let device = MockDevice::new();
        let stream = MockStream;
        let buffer = device
            .upload(&stream, &[1.0f32, 2.0, 3.0])
            .expect("the mock allocates");
        let mut whole = [0.0f32; 3];
        buffer.read(&stream, &mut whole).expect("the lengths match");
        assert_eq!(whole, [1.0, 2.0, 3.0]);
        for wrong in [2usize, 4] {
            let mut into = vec![0.0f32; wrong];
            match buffer.read(&stream, &mut into) {
                Err(DeviceError::Range { what, .. }) => assert_eq!(what, "read"),
                other => panic!("a range refusal, not {other:?}"),
            }
        }
    }

    #[test]
    fn a_shared_store_is_exclusive_and_loads_it_back() {
        let device = MockDevice::new();
        let mut shared = device.shared::<u32>(Elems(2)).expect("the mock allocates");
        shared.store(1, 7).expect("inside");
        assert_eq!(shared.load(1).expect("inside"), 7);
        assert!(shared.store(2, 1).is_err());
        assert!(shared.load(2).is_err());
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
