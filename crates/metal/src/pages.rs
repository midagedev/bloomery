//! The pages of a shared buffer: the one owner of the contents pointer's
//! invariant.
//!
//! A `StorageModeShared` buffer's `contents()` pointer is the host's view of
//! the buffer's pages, valid for the buffer's whole life. Every host access
//! to those pages goes through [`HostPages`], which checks the element count
//! and states the aliasing rule once; the buffers' own code holds no pointer
//! arithmetic.

use core::ptr::NonNull;
use core::sync::atomic::AtomicU32;

use bloomery_device::{DeviceError, Elem, Elems};

/// A count mismatch between a host slice and the pages, refused by name.
fn length_refusal(what: &'static str, got: usize, pages: Elems) -> DeviceError {
    DeviceError::Range {
        what,
        detail: format!(
            "{got} elements do not match the buffer's {} elements",
            pages.get()
        ),
    }
}

/// An index past the pages, refused by name.
fn index_refusal(what: &'static str, i: usize, pages: Elems) -> DeviceError {
    DeviceError::Range {
        what,
        detail: format!("element {i} is past the buffer's {} elements", pages.get()),
    }
}

/// `len` elements of `T` at `ptr`: the host's view of a buffer's pages.
pub(crate) struct HostPages<T: Elem> {
    ptr: NonNull<T>,
    len: Elems,
}

// SAFETY: the pages are plain memory (`T: Elem` is `Send + Sync`). Shared
// reads take `&self` and every write takes `&mut self`; the one case of a
// read racing a write is the device's, which `load` documents as volatile.
unsafe impl<T: Elem> Send for HostPages<T> {}
// SAFETY: as above.
unsafe impl<T: Elem> Sync for HostPages<T> {}

impl<T: Elem> HostPages<T> {
    /// The view of `len` elements at `ptr`.
    ///
    /// # Safety
    /// `ptr` is aligned for `T` and addresses `len` elements that stay mapped
    /// for as long as this value lives. Nothing else on the host reads or
    /// writes them except through this value; the device's access is ordered
    /// by the contract's streams.
    pub(crate) unsafe fn new(ptr: NonNull<T>, len: Elems) -> Self {
        Self { ptr, len }
    }

    /// The element count.
    pub(crate) fn len(&self) -> Elems {
        self.len
    }

    /// Every element zero.
    pub(crate) fn zero(&mut self) {
        // SAFETY: `new`'s contract: `len` elements are mapped at `ptr`, and
        // `&mut self` makes this the only host access.
        unsafe { core::ptr::write_bytes(self.ptr.as_ptr(), 0, self.len.get()) };
    }

    /// The pages become `host`'s elements; any other length is refused by
    /// name.
    pub(crate) fn write(&mut self, host: &[T]) -> Result<(), DeviceError> {
        if host.len() != self.len.get() {
            return Err(length_refusal("write", host.len(), self.len));
        }
        // SAFETY: the pages hold `host.len()` elements (checked above), the
        // two ranges do not overlap (`host` is a Rust borrow, the pages are
        // not), and `&mut self` makes this the only host access.
        unsafe { core::ptr::copy_nonoverlapping(host.as_ptr(), self.ptr.as_ptr(), host.len()) };
        Ok(())
    }

    /// The pages into `into`; any other length is refused by name.
    pub(crate) fn read(&self, into: &mut [T]) -> Result<(), DeviceError> {
        if into.len() != self.len.get() {
            return Err(length_refusal("read", into.len(), self.len));
        }
        // SAFETY: the pages hold `into.len()` elements (checked above) and do
        // not overlap `into`, a Rust borrow of other memory.
        unsafe { core::ptr::copy_nonoverlapping(self.ptr.as_ptr(), into.as_mut_ptr(), into.len()) };
        Ok(())
    }

    /// The element at `i`, a volatile read: the device may be writing it.
    pub(crate) fn load(&self, i: usize) -> Result<T, DeviceError> {
        if i >= self.len.get() {
            return Err(index_refusal("load", i, self.len));
        }
        // SAFETY: `i` is inside the pages (checked above); volatile because
        // the device may be writing the word, which the contract allows.
        Ok(unsafe { self.ptr.as_ptr().add(i).read_volatile() })
    }

    /// The element at `i`, a volatile write.
    pub(crate) fn store(&mut self, i: usize, v: T) -> Result<(), DeviceError> {
        if i >= self.len.get() {
            return Err(index_refusal("store", i, self.len));
        }
        // SAFETY: `i` is inside the pages (checked above), and `&mut self`
        // makes this the only host access; volatile because the device may
        // be reading the word, which the contract allows.
        unsafe { self.ptr.as_ptr().add(i).write_volatile(v) };
        Ok(())
    }

    /// The pages as a slice.
    ///
    /// # Safety
    /// No device work in flight writes the pages for as long as the slice
    /// lives.
    pub(crate) unsafe fn as_slice(&self) -> &[T] {
        // SAFETY: `len` elements are mapped at `ptr` for as long as `self`
        // lives; the caller promises the device writes none of them.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len.get()) }
    }

    /// The pages as a mutable slice.
    ///
    /// # Safety
    /// No device work in flight reads or writes the pages for as long as the
    /// slice lives.
    pub(crate) unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as in `as_slice`, and `&mut self` makes this the only host
        // access; the caller promises the device touches none of them.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len.get()) }
    }
}

impl HostPages<u32> {
    /// The pages as atomic words: the device reaches them with atomic
    /// operations, and the host's reads and writes of them are atomic too.
    pub(crate) fn atomics(&self) -> &[AtomicU32] {
        // SAFETY: `AtomicU32` has the size and alignment of `u32`, `len`
        // words are mapped at `ptr` for as long as `self` lives, and an
        // atomic access is defined next to the device's atomic accesses.
        unsafe {
            core::slice::from_raw_parts(self.ptr.as_ptr().cast::<AtomicU32>(), self.len.get())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pages over a vector the test owns, in place of a Metal buffer's.
    fn over(words: &mut [u32]) -> HostPages<u32> {
        let ptr = NonNull::new(words.as_mut_ptr()).expect("a vector's pointer is not null");
        // SAFETY: the vector holds `len` aligned words and outlives the pages
        // in each test; the test reads the vector only through the pages.
        unsafe { HostPages::new(ptr, Elems(words.len())) }
    }

    fn refused(result: Result<(), DeviceError>, what: &str) {
        match result {
            Err(DeviceError::Range { what: got, .. }) => assert_eq!(got, what),
            other => panic!("a {what} refusal, not {other:?}"),
        }
    }

    #[test]
    fn zero_clears_dirty_pages() {
        let mut backing = vec![0xFFFF_FFFFu32; 8];
        let mut pages = over(&mut backing);
        pages.zero();
        let mut got = [1u32; 8];
        pages.read(&mut got).expect("the lengths match");
        assert_eq!(got, [0; 8]);
    }

    #[test]
    fn a_write_of_another_length_is_refused_and_touches_nothing() {
        let mut backing = vec![7u32; 4];
        let mut pages = over(&mut backing);
        for wrong in [0usize, 3, 5, 100] {
            refused(pages.write(&vec![9u32; wrong]), "write");
        }
        let mut got = [0u32; 4];
        pages.read(&mut got).expect("the lengths match");
        assert_eq!(got, [7; 4]);

        pages.write(&[1, 2, 3, 4]).expect("the lengths match");
        pages.read(&mut got).expect("the lengths match");
        assert_eq!(got, [1, 2, 3, 4]);
    }

    #[test]
    fn a_read_of_another_length_is_refused() {
        let mut backing = vec![5u32; 4];
        let pages = over(&mut backing);
        for wrong in [0usize, 3, 5] {
            refused(pages.read(&mut vec![0u32; wrong]), "read");
        }
    }

    #[test]
    fn load_and_store_are_checked_against_the_count() {
        let mut backing = vec![0u32; 3];
        let mut pages = over(&mut backing);
        pages.store(2, 11).expect("inside");
        assert_eq!(pages.load(2).expect("inside"), 11);
        refused(pages.store(3, 1), "store");
        assert!(matches!(
            pages.load(3),
            Err(DeviceError::Range { what: "load", .. })
        ));
        // SAFETY: nothing else touches the vector or the device.
        assert_eq!(unsafe { pages.as_slice() }, &[0, 0, 11]);
        // SAFETY: as above.
        let slice = unsafe { pages.as_mut_slice() };
        slice[0] = 4;
        assert_eq!(pages.load(0).expect("inside"), 4);
    }

    #[test]
    fn atomics_see_the_same_words() {
        use core::sync::atomic::Ordering;

        let mut backing = vec![0u32; 2];
        let mut pages = over(&mut backing);
        pages.store(1, 6).expect("inside");
        assert_eq!(pages.atomics()[1].load(Ordering::Relaxed), 6);
        pages.atomics()[0].store(9, Ordering::Relaxed);
        assert_eq!(pages.load(0).expect("inside"), 9);
        assert_eq!(pages.len(), Elems(2));
    }
}
