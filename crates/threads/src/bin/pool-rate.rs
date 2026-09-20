//! Dispatch-tax bench for the resident pool: measures `for_each_chunk` overhead
//! for empty closures and a 4 KB per-thread sum.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use threads::{Pool, PoolStats, pool};

/// Pool dispatches per decode step at N=96.
const DISPATCHES_PER_STEP: usize = 322;

/// Rows per trivial dispatch: matches the engine's finest-grained matmul site (2048).
const N_TRIVIAL: usize = 2048;

/// 4 KB of u64 per thread over 32 threads: n / 32 * 8 B = 4 KB.
const N_4K: usize = 32 * 512;

/// Warm-up and calibration dispatches.
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

/// Calibrated timed run of `body` (one dispatch per call).
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

/// Pure protocol overhead with an empty closure.
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

/// Overhead plus L1-resident work: 4 KB sum per thread into a relaxed atomic.
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
