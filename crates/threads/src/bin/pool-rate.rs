//! Dispatch-tax bench for the resident pool (MUL-35, track B): what ONE
//! `for_each_chunk` call costs when the work inside it is nothing, and when
//! it is a 4 KB sum — because the N=96 engine issues 322 pool dispatches per
//! decode step (MUL-35 level-1 pool lines), so µs/dispatch × 322 is the
//! per-step dispatch-tax estimate this round needs next to the granularity
//! probe (`qdot-rate-mt --granularity`, the ceiling that tax sits under).
//!
//! * (a) trivial: `for_each_chunk(N, |_| {})` — the pure protocol price:
//!   the dispatch mutex, the job-slot publish, the seq bump + notify, the
//!   chunk arithmetic, the spin/park wait, the `remaining` countdown.
//! * (b) 4 KB: each thread sums its contiguous 512-u64 (4 KB) slice of a
//!   128 KB shared buffer — the tax plus an L1-resident morsel of work,
//!   the shape of "a dispatch that also does something tiny".
//!
//! The pool is process-wide and built on first use, reading
//! BLOOMERY_THREADS (default: physical cores) and BLOOMERY_SPIN; the
//! printed thread count is part of the record. The stats() deltas say
//! whether the measured dispatches stayed spin-hot (worker_parks = 0) or
//! paid the parking path — the difference between a back-to-back step and
//! an idle machine, and the first thing to check before quoting a number.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use threads::{Pool, PoolStats, pool};

/// Pool dispatches per decode step at N=96 — the MUL-35 level-1 pool line
/// this bench's product is judged against.
const DISPATCHES_PER_STEP: usize = 322;

/// Rows per trivial dispatch: the Q4_K matmul site's 2048 rows/call — the
/// engine's own n at its finest-grained site. The protocol cost does not
/// depend on n; this grounds the call in the real shape anyway.
const N_TRIVIAL: usize = 2048;

/// 4 KB of u64 per thread over 32 threads: n such that n / 32 × 8 B = 4 KB.
const N_4K: usize = 32 * 512;

/// Warm-up dispatches before any timing, and the probe size the iteration
/// count is calibrated from (target ~1.5 s of timed work per bench).
const WARMUP: usize = 2_000;
const PROBE: usize = 5_000;

/// Field-wise difference of two stat snapshots (u64 counters, monotone).
fn delta(before: PoolStats, after: PoolStats) -> (u64, u64, u64) {
    (
        after.dispatches - before.dispatches,
        after.dispatcher_parks - before.dispatcher_parks,
        after.worker_parks - before.worker_parks,
    )
}

/// Calibrated timed run of `body`, which must issue exactly one dispatch
/// per call: warm up, probe, scale to ~1.5 s, time, return
/// (µs/dispatch, dispatches, stats delta).
fn timed(pool: &Pool, mut body: impl FnMut()) -> (f64, usize, (u64, u64, u64)) {
    for _ in 0..WARMUP {
        body();
    }
    let t0 = Instant::now();
    for _ in 0..PROBE {
        body();
    }
    let per = t0.elapsed().as_secs_f64() / PROBE as f64;
    let iters = ((1.5 / per) as usize).clamp(20_000, 2_000_000);
    let s0 = pool.stats();
    let t0 = Instant::now();
    for _ in 0..iters {
        body();
    }
    let dt = t0.elapsed().as_secs_f64();
    let d = delta(s0, pool.stats());
    (dt / iters as f64 * 1e6, iters, d)
}

/// (a) The pure protocol price. The closure is empty on purpose — the
/// mutex, the slot publish, the seq/remaining protocol and the spin or
/// park wait are the entire cost being measured.
fn bench_trivial(pool: &Pool) {
    let (us, iters, (disp, dpark, wpark)) = timed(pool, || pool.for_each_chunk(N_TRIVIAL, |_| {}));
    println!(
        "trivial  n={N_TRIVIAL}: {us:>7.3} µs/dispatch over {iters} dispatches \
         (dispatches +{disp}, dispatcher parks +{dpark}, worker parks +{wpark})"
    );
    println!(
        "  -> per decode step: {DISPATCHES_PER_STEP} dispatches x {us:.3} µs = {:.3} ms",
        us * DISPATCHES_PER_STEP as f64 / 1e3
    );
}

/// (b) The tax plus a 4 KB sum: each participant sums its contiguous slice
/// of the shared 128 KB buffer into a relaxed atomic — real work the
/// optimizer cannot drop, small enough to stay L1-resident so the delta
/// against (a) is the per-dispatch work floor, not memory.
fn bench_4k(pool: &Pool) {
    let buf: Vec<u64> = (0..N_4K as u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();
    let total = AtomicU64::new(0);
    let (us, iters, (disp, dpark, wpark)) = timed(pool, || {
        pool.for_each_chunk(N_4K, |rows| {
            let mut s = 0u64;
            for &v in &buf[rows] {
                s = s.wrapping_add(v);
            }
            total.fetch_add(s, Ordering::Relaxed);
        })
    });
    let sum = total.load(Ordering::Relaxed);
    println!(
        "4KB sum  n={N_4K}: {us:>7.3} µs/dispatch over {iters} dispatches \
         (dispatches +{disp}, dispatcher parks +{dpark}, worker parks +{wpark}, sum {sum})"
    );
    println!(
        "  -> per decode step: {DISPATCHES_PER_STEP} dispatches x {us:.3} µs = {:.3} ms",
        us * DISPATCHES_PER_STEP as f64 / 1e3
    );
}

fn main() {
    let pool = pool();
    println!(
        "pool-rate: threads = {} (BLOOMERY_THREADS or physical cores), pin_failed = {}",
        pool.threads(),
        pool.pin_failed()
    );
    bench_trivial(pool);
    bench_4k(pool);
}
