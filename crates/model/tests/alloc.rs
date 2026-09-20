//! Gate for allocator traffic: how often a steady-state decode step goes to malloc.
//!
//! No gate guards speed, but this is the one speed cause that reduces to an integer.
//! The main thread's steady-state profile (2026-09-20) had a quarter of its samples in
//! `memset` and the malloc family — every op returned a fresh zeroed `Vec` — and the
//! workers idle while the main thread does that. The count is over every thread, so a
//! worker closure that allocates per chunk shows up here too.
//!
//! `hw_` prefix: needs the box and the model file, not the oracle.
#[path = "common/oracle.rs"]
mod oracle;

use model::forward::{new_cache, step};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

static CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
/// The calling thread's share: the part that is serial, with every worker waiting on it.
static MAIN_CALLS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    // const-initialized and destructor-free, so reading it inside the allocator cannot recurse.
    static IS_MAIN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct Counting;

fn count(size: usize) {
    CALLS.fetch_add(1, Relaxed);
    BYTES.fetch_add(size as u64, Relaxed);
    if IS_MAIN.with(|m| m.get()) {
        MAIN_CALLS.fetch_add(1, Relaxed);
    }
}

// SAFETY: every method forwards to `System` with the caller's arguments unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        count(l.size());
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        count(l.size());
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        unsafe { System.realloc(p, l, new_size) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Allocator calls one single-token step may make, all threads. A ratchet, not a
/// target: it only moves down.
///
/// PIN(2026-09-20): 14454 calls / 24.5 MB per step before the `Tensor2` free list,
/// 11463 / 16.3 MB after (8952 of them on the calling thread); a step that grows the
/// KV blocks adds 28. What is left is per-call `Vec`s in `matmul_q_multi`, the
/// `TensorInfo` views built per expert and per head, and metadata keys formatted per
/// block — token-independent work that belongs in `Derived`.
/// PIN(2026-09-20): 11463 -> 1291 / 0.32 MB on the load-time plan round — every
/// tensor-name find, metadata read, norm-gain decode and per-head/per-expert view
/// moved into `Derived`, the quantization and flash scratch into per-thread
/// bucketed pools, and the MoE trace behind an opt-in. What is left: the routing
/// table's vectors and the per-batch bookkeeping (`ws`/`xs`/`bytes`/`outs`/`pairs`
/// of a `matmul_q_batch` call), plus the KV row a growing cache owns.
/// PIN(2026-09-21): 1291 -> 765 / 0.18 MB on the group dispatch round — the batch
/// bookkeeping (`ws`/`xs`/`outs`/`pairs`) moved to fixed-capacity stack arrays. A
/// step that grows the KV blocks adds 28.
const LIMIT: u64 = 850;

#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_steady_step_allocations_bounded() {
    let g = gguf::Gguf::open(oracle::model_path()).unwrap();
    let mut cache = new_cache(&g).unwrap();
    let derived = model::derived::Derived::new(&g).unwrap();
    step(
        &g,
        &[100000, 549, 6077, 280, 7239, 317],
        &mut cache,
        &derived,
    )
    .unwrap();
    // Warm steps first: pools fill and the KV blocks grow on their own schedule.
    IS_MAIN.with(|m| m.set(true));
    let mut worst = 0u64;
    for i in 0..8 {
        let (c0, b0, m0) = (
            CALLS.load(Relaxed),
            BYTES.load(Relaxed),
            MAIN_CALLS.load(Relaxed),
        );
        step(&g, &[549], &mut cache, &derived).unwrap();
        let (c, b) = (CALLS.load(Relaxed) - c0, BYTES.load(Relaxed) - b0);
        let m = MAIN_CALLS.load(Relaxed) - m0;
        eprintln!(
            "step {i}: {c} allocator calls ({m} on the calling thread), {:.1} KB",
            b as f64 / 1024.0
        );
        if i >= 4 {
            worst = worst.max(c);
        }
    }
    assert!(
        worst <= LIMIT,
        "steady decode step made {worst} allocator calls, limit {LIMIT}"
    );
}
