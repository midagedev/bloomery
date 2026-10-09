//! Gate for allocator traffic: how often a steady-state decode step goes to malloc.
//!
//! No gate guards speed, but this is the one speed cause that reduces to an integer.
//! The main thread's steady-state profile had a quarter of its samples in
//! `memset` and the malloc family — every op returned a fresh zeroed `Vec` — and the
//! workers idle while the main thread does that. The count is over every thread, so a
//! worker closure that allocates per chunk shows up here too.
//!
//! `hw_` prefix: needs the box and the model file, not the oracle.
#[path = "common/model_path.rs"]
mod model_path;
#[allow(
    dead_code,
    reason = "this gate renders one layer; the sidecar and the salted writer serve the union gates"
)]
#[path = "common/r8layer.rs"]
mod r8layer;

use gguf::{GgmlType, Split};
use model::arch::deepseek2::derived::Derived;
use model::arch::deepseek2::forward::{new_cache, step};
use model::moe::{HostLayer, HostScratch, UnionScratch};
use model::ops::{self, Tensor2};
use model::r8file::R8Source;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// The counters are the process's: one gate counts at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

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
        // SAFETY: the caller meets `alloc`'s contract for `l` (non-zero size), and
        // `System` receives `l` unchanged.
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        count(l.size());
        // SAFETY: the caller meets `alloc_zeroed`'s contract for `l` (non-zero size),
        // and `System` receives `l` unchanged.
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        // SAFETY: `p` came from this allocator with layout `l` — so from `System`, as every
        // allocating method forwards there — and the caller meets `realloc`'s contract
        // for `new_size`; `System` receives all three unchanged.
        unsafe { System.realloc(p, l, new_size) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` came from this allocator with layout `l`, so from `System`, as every
        // allocating method forwards there.
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
/// PIN(2026-09-21): 765 -> 630 / 0.13 MB on the fused attention round — the
/// per-head `wv_b` gather (sixteen activation blocks and its batch bookkeeping)
/// left the step; worst warm step 658. A step that grows the KV blocks adds 28.
/// PIN(2026-09-21): 630 -> 603 / 0.13 MB on the contiguous-KV round — one flat
/// buffer per block replaces one `Vec` per cached row plus its outer `Vec`.
/// LIMIT 700 -> 660 on the same round: worst warm step measured 631 (the capacity-
/// doubling step, 864 KB); 660 keeps ~5 % headroom over that and the ratchet only
/// moves down.
const LIMIT: u64 = 660;

#[test]
#[ignore = "hw: needs the box and the model file"]
fn hw_steady_step_allocations_bounded() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let g = gguf::Gguf::open(model_path::model_path()).unwrap();
    let mut cache = new_cache(&g).unwrap();
    let derived = Derived::new(&g).unwrap();
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

/// Allocator calls of the host tier's legs over a synthetic routed layer —
/// the one-column leg and a three-column step union, each a pair of pool
/// dispatches — in a steady state, under flat lanes and under CCD-major lanes
/// ([`ops::set_host_lanes`]): the CCD-major dispatch keeps its tables on the
/// stack, so its calls make no more than the flat ones' (a table made per
/// dispatch is two allocations a call). Each arm's count is the least of three
/// repeats of eight calls after twenty that fill the pools, so a worker's
/// first touch of a scratch is not read as a table.
#[test]
#[ignore = "hw: needs the box; reads no model file"]
fn hw_a_ccd_dispatch_allocates_no_more_than_a_flat_one() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let layer = r8layer::Layer::write("alloc-ccd", 512, 256, 16, GgmlType::Q4_K);
    let (embd, ff) = (layer.embd, layer.ff);
    let split = Split::open(&layer.source).unwrap();
    let src = R8Source::rows(&split);
    let host = HostLayer::build(src, &layer.spec()).unwrap();
    let mut hs = HostScratch::new(embd, ff, 4).unwrap();
    let mut us = UnionScratch::new_routed(embd, ff, 3, 4).unwrap();
    let x = Tensor2::from_vec(
        embd,
        1,
        (0..embd).map(|i| (i % 17) as f32 / 9.0 - 0.9).collect(),
    );
    let x3 = Tensor2::from_vec(
        embd,
        3,
        (0..3 * embd)
            .map(|i| (i % 23) as f32 / 11.0 - 1.0)
            .collect(),
    );
    let list =
        |j: u32| -> Vec<(u32, f32)> { (0..4).map(|i| ((j * 5 + i * 3) % 16, 0.25)).collect() };
    let lists = [list(0), list(1), list(2)];
    let refs: Vec<&[(u32, f32)]> = lists.iter().map(Vec::as_slice).collect();
    let (mut out1, mut out3) = (vec![0.0f32; embd], vec![0.0f32; 3 * embd]);
    let mut calls = |n: usize| {
        for _ in 0..n {
            host.experts_into(src, &x, &lists[0], &mut out1, &mut hs)
                .unwrap();
            host.experts_step_union_into(src, &x3, &refs, &mut out3, &mut us)
                .unwrap();
        }
    };
    IS_MAIN.with(|m| m.set(true));
    let mut least = [u64::MAX; 2];
    for (arm, lanes) in [(0, Some(0)), (1, Some(4))] {
        ops::set_host_lanes(lanes);
        calls(20);
        for _ in 0..3 {
            let c0 = CALLS.load(Relaxed);
            calls(8);
            least[arm] = least[arm].min(CALLS.load(Relaxed) - c0);
        }
    }
    ops::set_host_lanes(None);
    eprintln!(
        "host legs, 8 calls of one-column + 3-column step union: {} allocator calls flat, {} CCD-major",
        least[0], least[1]
    );
    assert!(
        least[1] <= least[0],
        "CCD-major lanes made {} allocator calls over 8 calls, the flat lanes {}",
        least[1],
        least[0]
    );
}
