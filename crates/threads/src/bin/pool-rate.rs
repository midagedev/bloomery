//! Dispatch-tax bench for the resident pool: measures `for_each_chunk` overhead
//! for empty closures and a 4 KB per-thread sum, how long a worker spins before
//! it parks, and when each participant starts a dispatch that finds the workers
//! parked against one that finds them spinning.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

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

/// Dispatches each wake arm stamps.
const WAKE_ROUNDS: usize = 200;

/// The wall from a dispatch's return until every worker has parked: the spin
/// window a worker waits out before its futex wait, as the pool runs it. The
/// median over the rounds.
fn bench_spin_window(pool: &Pool) {
    let workers = (pool.threads() - 1) as u64;
    let mut walls = Vec::with_capacity(WAKE_ROUNDS);
    for _ in 0..WAKE_ROUNDS {
        let p0 = pool.stats().worker_parks;
        pool.for_each_chunk(pool.threads(), |_| {});
        let t0 = Instant::now();
        while pool.stats().worker_parks - p0 < workers {
            std::hint::spin_loop();
            if t0.elapsed() > Duration::from_secs(1) {
                println!("spin window: the workers did not all park within 1 s");
                return;
            }
        }
        walls.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    walls.sort_by(f64::total_cmp);
    println!(
        "spin window: every worker parked {:.1} us after the dispatch returned (median of {WAKE_ROUNDS}; \
         p10 {:.1}, p90 {:.1})",
        walls[walls.len() / 2],
        walls[walls.len() / 10],
        walls[walls.len() * 9 / 10]
    );
}

/// When each participant enters a dispatch, in µs after the dispatcher called
/// it: `cold` waits for every worker to park first, else back to back. Prints
/// the mean start of each participant (the caller last), the mean and the max
/// over the workers per dispatch, and the lane-µs the dispatch loses to late
/// starts (the workers' summed start over the participants).
fn bench_wake(pool: &Pool, cold: bool) {
    let n = pool.threads();
    let workers = (n - 1) as u64;
    let stamps: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
    let mut sum = vec![0f64; n];
    let (mut means, mut maxes, mut lost) = (Vec::new(), Vec::new(), 0f64);
    for _ in 0..WAKE_ROUNDS {
        if cold {
            let p0 = pool.stats().worker_parks;
            pool.for_each_chunk(n, |_| {});
            while pool.stats().worker_parks - p0 < workers {
                std::thread::sleep(Duration::from_micros(200));
            }
            // Past the park: the cores may sink into an idle state, as a
            // layer's fill wait leaves them.
            std::thread::sleep(Duration::from_millis(2));
        }
        let t0 = Instant::now();
        pool.for_each_chunk(n, |r| {
            let ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
            for t in r {
                stamps[t].store(ns, Ordering::Relaxed);
            }
        });
        let at: Vec<f64> = stamps
            .iter()
            .map(|s| s.load(Ordering::Relaxed) as f64 / 1e3)
            .collect();
        for (s, a) in sum.iter_mut().zip(&at) {
            *s += a;
        }
        let w = &at[..n - 1];
        let wsum: f64 = w.iter().sum();
        means.push(wsum / w.len() as f64);
        maxes.push(w.iter().copied().fold(0f64, f64::max));
        lost += wsum / n as f64;
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        (v[v.len() / 2], v[v.len() * 9 / 10])
    };
    let (mean_p50, mean_p90) = med(&mut means);
    let (max_p50, max_p90) = med(&mut maxes);
    let arm = if cold { "cold" } else { "warm" };
    println!(
        "wake {arm}: workers' start mean p50 {mean_p50:.1} us (p90 {mean_p90:.1}), last worker p50 \
         {max_p50:.1} us (p90 {max_p90:.1}), lane-us lost a dispatch {:.1} (over {n} lanes)",
        lost / WAKE_ROUNDS as f64
    );
    let row: Vec<String> = sum
        .iter()
        .map(|s| format!("{:.1}", s / WAKE_ROUNDS as f64))
        .collect();
    println!(
        "wake {arm}: mean start by participant (us): {}",
        row.join(" ")
    );
}

fn main() {
    let pool = pool();
    println!(
        "pool-rate: threads = {} (BLOOMERY_THREADS or physical cores), pin_failed = {}",
        pool.threads(),
        pool.pin_failed()
    );
    if pool.threads() > 1 {
        pool.pin_caller();
    }
    bench_trivial(pool);
    bench_4k(pool);
    if pool.threads() > 1 {
        bench_spin_window(pool);
        bench_wake(pool, false);
        bench_wake(pool, true);
    }
}
