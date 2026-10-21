//! The pages of a shared buffer as the host sees them: the one owner of the
//! host pointer's invariant, for every backend.
//!
//! A shared buffer's host pointer (Metal's `contents()`, CUDA's mapped pinned
//! memory) addresses the pages the device reads, valid for the buffer's whole
//! life. Every host access to them goes through [`HostPages`], which checks
//! the element count and states the aliasing rule once; the backends' own
//! code holds no pointer arithmetic over them.
//!
//! An element is reached two ways. One at a time, as its atomic twin
//! ([`Elem::Atom`]): a load or a store with an ordering, which may meet the
//! device's or another host thread's access to the same element. In bulk, as
//! a copy: a read copies element by element through the atomics (relaxed), and
//! a write, which takes the pages exclusively, copies plainly.

use core::ptr::NonNull;
use core::sync::atomic::Ordering;

use crate::{Atom, DeviceError, Elem, Elems};

/// An index past the pages, refused by name.
fn index_refusal(what: &'static str, i: usize, pages: Elems) -> DeviceError {
    DeviceError::Range {
        what,
        detail: format!("element {i} is past the buffer's {} elements", pages.get()),
    }
}

/// `len` elements of `T` at `ptr`: the host's view of a shared buffer's pages.
pub struct HostPages<T: Elem> {
    ptr: NonNull<T>,
    len: Elems,
}

// SAFETY: the pages are plain memory (`T: Elem` is `Send + Sync`). An access
// through `&self` is atomic, and every plain write takes `&mut self`.
unsafe impl<T: Elem> Send for HostPages<T> {}
// SAFETY: as above.
unsafe impl<T: Elem> Sync for HostPages<T> {}

impl<T: Elem> HostPages<T> {
    /// The view of `len` elements at `ptr`.
    ///
    /// # Safety
    /// `ptr` is aligned for `T` and addresses `len` elements that stay mapped
    /// for as long as this value lives. While it lives, the host reaches those
    /// elements only through values of this type, and a plain write (through
    /// `&mut`) by one of them never meets another's access to the same
    /// element; the device's access is ordered by the contract's streams.
    pub unsafe fn new(ptr: NonNull<T>, len: Elems) -> Self {
        Self { ptr, len }
    }

    /// The element count.
    #[must_use]
    pub fn len(&self) -> Elems {
        self.len
    }

    /// Whether the pages hold no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == Elems::ZERO
    }

    /// The pages as their atomic twins.
    fn atoms(&self) -> &[T::Atom] {
        // SAFETY: `Elem`'s contract gives `T::Atom` the size and alignment of
        // `T`; `new`'s contract maps `len` aligned elements at `ptr` for as
        // long as `self` lives; an atomic access is defined next to the
        // device's and another host thread's atomic accesses.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr().cast::<T::Atom>(), self.len.get()) }
    }

    /// Every element zero.
    pub fn zero(&mut self) {
        // SAFETY: `new`'s contract: `len` elements are mapped at `ptr`, and
        // `&mut self` makes this the only host access.
        unsafe { core::ptr::write_bytes(self.ptr.as_ptr(), 0, self.len.get()) };
    }

    /// Elements `at..at + from.len()` become `from`'s; a range past the pages
    /// is refused by name.
    pub fn write_at(&mut self, at: Elems, from: &[T]) -> Result<(), DeviceError> {
        crate::check_window("buffer", self.len, at, Elems(from.len()))?;
        // SAFETY: the range lies inside the pages (checked above), the two
        // ranges do not overlap (`from` is a Rust borrow, the pages are not),
        // and `&mut self` makes this the only host access.
        unsafe {
            core::ptr::copy_nonoverlapping(
                from.as_ptr(),
                self.ptr.as_ptr().add(at.get()),
                from.len(),
            );
        }
        Ok(())
    }

    /// The pages become `host`'s elements; any other length is refused by
    /// name.
    pub fn write(&mut self, host: &[T]) -> Result<(), DeviceError> {
        crate::check_len("write", host.len(), self.len)?;
        self.write_at(Elems::ZERO, host)
    }

    /// Elements `at..at + into.len()` into `into`, each a relaxed atomic load;
    /// a range past the pages is refused by name.
    pub fn read_at(&self, at: Elems, into: &mut [T]) -> Result<(), DeviceError> {
        crate::check_window("buffer", self.len, at, Elems(into.len()))?;
        for (slot, atom) in into.iter_mut().zip(&self.atoms()[at.get()..]) {
            *slot = atom.load(Ordering::Relaxed);
        }
        Ok(())
    }

    /// The pages into `into`; any other length is refused by name.
    pub fn read(&self, into: &mut [T]) -> Result<(), DeviceError> {
        crate::check_len("read", into.len(), self.len)?;
        self.read_at(Elems::ZERO, into)
    }

    /// Every element, each a relaxed atomic load.
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        self.atoms()
            .iter()
            .map(|atom| atom.load(Ordering::Relaxed))
            .collect()
    }

    /// The element at `i`, an atomic load with `order`.
    pub fn load(&self, i: usize, order: Ordering) -> Result<T, DeviceError> {
        let atom = self
            .atoms()
            .get(i)
            .ok_or_else(|| index_refusal("load", i, self.len))?;
        Ok(atom.load(order))
    }

    /// The element at `i` becomes `v`, an atomic store with `order`.
    pub fn store(&self, i: usize, v: T, order: Ordering) -> Result<(), DeviceError> {
        let atom = self
            .atoms()
            .get(i)
            .ok_or_else(|| index_refusal("store", i, self.len))?;
        atom.store(v, order);
        Ok(())
    }

    /// The pages as a slice.
    ///
    /// # Safety
    /// No device work in flight and no host [`HostPages::store`] writes the
    /// pages for as long as the slice lives.
    #[must_use]
    pub unsafe fn as_slice(&self) -> &[T] {
        // SAFETY: `len` elements are mapped at `ptr` for as long as `self`
        // lives; the caller promises nothing writes them meanwhile.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len.get()) }
    }

    /// The pages as a mutable slice.
    ///
    /// # Safety
    /// No device work in flight reads or writes the pages for as long as the
    /// slice lives.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as in `as_slice`, and `&mut self` makes this the only host
        // access; the caller promises the device touches none of them.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len.get()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pages over a vector the test owns, in place of a backend's mapping.
    fn over<T: Elem>(words: &mut [T]) -> HostPages<T> {
        let ptr = NonNull::new(words.as_mut_ptr()).expect("a vector's pointer is not null");
        // SAFETY: the vector holds `len` aligned elements and outlives the
        // pages in each test; the test reads the vector only through them.
        unsafe { HostPages::new(ptr, Elems(words.len())) }
    }

    fn refused<T: core::fmt::Debug>(result: Result<T, DeviceError>, what: &str) {
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
    fn a_ranged_copy_lands_inside_and_a_range_past_the_end_is_refused() {
        let mut backing = vec![0u64; 6];
        let mut pages = over(&mut backing);
        pages.write_at(Elems(2), &[5, 6, 7]).expect("inside");
        let mut got = [0u64; 2];
        pages.read_at(Elems(3), &mut got).expect("inside");
        assert_eq!(got, [6, 7]);
        refused(pages.write_at(Elems(4), &[1, 2, 3]), "window");
        refused(pages.read_at(Elems(5), &mut [0u64; 2]), "window");
        refused(pages.read_at(Elems(usize::MAX), &mut [0u64; 2]), "window");
        let mut whole = [9u64; 6];
        pages.read(&mut whole).expect("the lengths match");
        assert_eq!(whole, [0, 0, 5, 6, 7, 0]);
    }

    #[test]
    fn load_and_store_are_checked_against_the_count() {
        let mut backing = vec![0.0f32; 3];
        let mut pages = over(&mut backing);
        pages.store(2, 1.5, Ordering::Release).expect("inside");
        assert_eq!(pages.load(2, Ordering::Acquire).expect("inside"), 1.5);
        refused(pages.store(3, 1.0, Ordering::Relaxed), "store");
        refused(pages.load(3, Ordering::Relaxed), "load");
        // SAFETY: nothing else touches the vector or the device.
        assert_eq!(unsafe { pages.as_slice() }, &[0.0, 0.0, 1.5]);
        // SAFETY: as above.
        let slice = unsafe { pages.as_mut_slice() };
        slice[0] = 4.0;
        assert_eq!(pages.load(0, Ordering::Relaxed).expect("inside"), 4.0);
        assert_eq!(pages.len(), Elems(3));
    }

    /// A store from one host thread is what another loads: the flag a writer
    /// releases after its data, the data a reader sees after it acquires it.
    #[test]
    fn a_release_store_publishes_what_was_written_before_it() {
        let mut backing = vec![0u32; 2];
        let pages = over(&mut backing);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                pages.store(0, 41, Ordering::Relaxed).expect("inside");
                pages.store(1, 1, Ordering::Release).expect("inside");
            });
            scope.spawn(|| {
                while pages.load(1, Ordering::Acquire).expect("inside") == 0 {
                    std::hint::spin_loop();
                }
                assert_eq!(pages.load(0, Ordering::Relaxed).expect("inside"), 41);
            });
        });
    }
}
