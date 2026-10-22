//! The success paths of the contract's range rules allocate nothing: a
//! refusal's text is built when there is a refusal, not on every call. Every
//! window operation and launch binding goes through these rules.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use bloomery_device::{Elems, buffer_bytes, check_len, check_window};

thread_local! {
    /// The allocator calls this thread has made.
    static CALLS: Cell<usize> = const { Cell::new(0) };
}

/// The system allocator, counting the calls of the thread that makes them: a
/// test thread's count is its own, whatever the other tests do.
struct Counting;

// SAFETY: every call is the system allocator's, with the caller's layout and
// pointer unchanged; the count is a thread-local cell that allocates nothing.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        CALLS.with(|calls| calls.set(calls.get() + 1));
        // SAFETY: the caller's contract is `System`'s.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` with `layout`, which is `System`'s.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// The allocator calls `f` makes on this thread.
fn calls_of(f: impl FnOnce()) -> usize {
    let before = CALLS.with(Cell::get);
    f();
    CALLS.with(Cell::get) - before
}

#[test]
fn the_range_rules_allocate_nothing_when_they_accept() {
    let calls = calls_of(|| {
        assert_eq!(Elems(3).try_add(Elems(4)).unwrap(), Elems(7));
        check_window("buffer", Elems(8), Elems(2), Elems(3)).unwrap();
        check_len("read", 4, Elems(4)).unwrap();
        assert_eq!(Elems(5).bytes_of::<f32>().unwrap().get(), 20);
        assert_eq!(buffer_bytes::<f32>(Elems(4)).unwrap().get(), 16);
    });
    assert_eq!(calls, 0, "an accepted range allocated {calls} times");
}

#[test]
fn the_counter_sees_a_refusal_built() {
    let calls = calls_of(|| {
        assert!(check_window("buffer", Elems(8), Elems(6), Elems(3)).is_err());
    });
    assert!(
        calls > 0,
        "the counter did not see the refusal's text built"
    );
}
