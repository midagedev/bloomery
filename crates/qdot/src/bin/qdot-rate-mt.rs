//! MT streaming ceiling for `dot_row` (MUL-35, track B), plus the
//! dispatch-granularity probe that goes with it — the two numbers that
//! separate "the kernels are slow" from "the dispatch shape is slow".
//!
//! Why this exists: the N=96 decode stage table (MUL-35 level 1, per-lease
//! measurement) shows every site's achieved GB/s tracking rows/call — the
//! pool's redispatch unit — rather than kernel quality:
//!
//! | site | MB/step | ms/step | GB/s | calls/step | rows/call |
//! |---|---|---|---|---|---|
//! | batch Q3_K | 399 | 5.67 | 70 | 53 | ~9333 |
//! | matmul_q Q3_K | 235 | 4.58 | 51 | 108 | ~2470 |
//! | batch Q5_0 | 309 | 4.56 | 68 | 26 | ~12288 |
//! | matmul_q Q4_K | 148 | 3.12 | 47 | 53 | 2048 |
//! | matmul_q Q6_K | 172 | 1.44 | 119 | 1 | 102400 |
//! | matmul_q Q5_1 | 16.8 | 0.223 | 75 | 1 | 2048 |
//!
//! while the single-core kernel rates (`qdot-rate`, MUL-26..34: 10.8 GB/s
//! Q3_K, 14.4 Q4_K, 16.7 Q6_K, per core) sit far above what any site
//! reaches per thread at 32 threads even against the 147.7 GB/s STREAM
//! ceiling, and the engine issues 322 pool dispatches per step. The
//! suspicion this bin is built to cut: achieved bandwidth is set by
//! rows/call, not by the kernels. Two modes, zero engine orchestration in
//! both:
//!
//! * default (the MT ceiling): T in {1, 8, 16, 32} plain
//!   `std::thread::scope` threads over one shared ~2.5 GB weight buffer,
//!   each streaming its own contiguous row range of real `dot_row` calls —
//!   no job slots, no parking, no pinning. std-only on purpose: this is
//!   the OS-scheduled ceiling of OUR kernels with the engine contributing
//!   nothing.
//! * `--granularity`: T = 32 fixed, R rows/call swept over the stage
//!   table's live range. Each thread walks its contiguous share of R rows,
//!   then all 32 resynchronize on a `std::sync::Barrier` before the next
//!   call. CAVEAT, corrected against this round's own measurement on the
//!   box (2026-09-20): std's Barrier is a mutex+condvar broadcast, and it
//!   is NOT cheaper than the pool's job-slot/parking protocol — at
//!   R = 2048 (the Q4_K site's granularity) this probe measures 18.3 GB/s
//!   while the engine itself reaches 47 GB/s on the same shape, and
//!   `pool-rate` prices the pool's spin-hot dispatch at ~4.7 µs vs the
//!   ~100 µs this probe's per-call Barrier sync implies. So the R sweep
//!   characterizes the Barrier, not the pool: with one fixed sync
//!   primitive it shows the rows/call law cleanly (GB/s rising monotone
//!   with R until ~25k rows/call), and at small R it is a FLOOR — the
//!   engine's cheaper dispatch already beats it. The engine's actual
//!   granularity at the Q4_K site is 2048 rows/call over 32 threads =
//!   64 rows/thread/call.
//!
//! Shapes are the five stage-table sites at their real k and row sizes
//! (source: the MUL-35 level-1 table quoted above). Weights are xorshift64*
//! filler — any bytes are valid quantized codes and the kernels are
//! data-independent — and the activation column goes through the real
//! `quantize_col`, exactly the discipline `qdot-rate` (MUL-26)
//! established. Each thread's accumulator is returned from its scoped
//! closure and printed after the timed region, so the loops cannot be
//! optimized away and the timing contains no printing. Aggregate GB/s
//! counts WEIGHT bytes only (the activation column is a few KB, shared
//! read-only, cache-resident).

use std::sync::Barrier;
use std::time::Instant;

use gguf::GgmlType;

/// Timed-section target per cell: long enough for a stable aggregate,
/// short enough that the whole shape sweep finishes in minutes.
const TARGET_SECS: f64 = 3.0;
/// Pass/sweep clamps: at least two passes (a single pass is a warm-up, not
/// a number), at most enough to keep the fastest cell bounded.
const MIN_PASSES: u32 = 2;
const MAX_PASSES: u32 = 4000;
const MIN_SWEEPS: usize = 2;
const MAX_SWEEPS: usize = 200;

/// The granularity probe's rows/call values (T = 32 fixed): the stage
/// table's live range from the Q4_K site's 2048 (and sub-site scales below
/// it) up to the Q6_K site's 102400 — with 9333 and 12288, the batch Q3_K
/// and Q5_0 sites' real ragged rows/call (~9333 measured as 53 calls over
/// ~495k rows, 12288 as 26 over 319k).
const RS: [usize; 9] = [512, 1024, 2048, 4096, 9333, 12288, 25600, 51200, 102400];
const T_GRAN: usize = 32;

/// One stage-table site: type, its real k, the model's real row size in
/// bytes, and the row count of the shared streaming buffer.
struct Shape {
    ty: GgmlType,
    k: usize,
    row_bytes: usize,
    /// Multiple of 102400 so every T in {1, 8, 16, 32} splits it into whole
    /// rows and every R in [`RS`] lands on whole-row call boundaries; sized
    /// so the buffer sits at ~2.5 GB — far past the 128 MB of L3, so every
    /// pass is DRAM streaming, not cache reuse.
    rows: usize,
    site: &'static str,
}

fn shapes() -> [Shape; 5] {
    [
        Shape {
            ty: GgmlType::Q3_K,
            k: 2048,
            row_bytes: (2048 / 256) * 110, // 880
            rows: 27 * 102_400,            // 2.43 GB
            site: "batch Q3_K + matmul_q Q3_K (k=2048, 880 B/row)",
        },
        Shape {
            ty: GgmlType::Q4_K,
            k: 2048,
            row_bytes: (2048 / 256) * 144, // 1152
            rows: 21 * 102_400,            // 2.48 GB
            site: "matmul_q Q4_K (k=2048, 1152 B/row)",
        },
        Shape {
            ty: GgmlType::Q6_K,
            k: 2048,
            row_bytes: (2048 / 256) * 210, // 1680
            rows: 14 * 102_400,            // 2.41 GB
            site: "matmul_q Q6_K (k=2048, 1680 B/row)",
        },
        Shape {
            ty: GgmlType::Q5_0,
            k: 1408,
            row_bytes: (1408 / 32) * 22, // 968
            rows: 25 * 102_400,          // 2.48 GB
            site: "batch Q5_0 (k=1408, 968 B/row)",
        },
        Shape {
            ty: GgmlType::Q5_1,
            k: 10_944,
            row_bytes: (10_944 / 32) * 24, // 8208
            rows: 3 * 102_400,             // 2.52 GB
            site: "matmul_q Q5_1 (k=10944, 8208 B/row)",
        },
    ]
}

/// xorshift64* filler — `qdot-rate`'s discipline (MUL-26): any bytes are
/// valid quantized codes and the kernels' time is data-independent. The
/// fill also touches every page, so the timed passes never fault.
fn fill(rows: usize, row_bytes: usize) -> Vec<u8> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let mut w = vec![0u8; rows * row_bytes];
    let (chunks, _) = w.as_chunks_mut::<8>();
    for c in chunks {
        c.copy_from_slice(&next().to_le_bytes());
    }
    w
}

/// The real activation column for `k` values, through the real
/// `quantize_col` — same filler column as `qdot-rate`.
fn acol(ty: GgmlType, k: usize) -> Vec<u8> {
    let col: Vec<f32> = (0..k)
        .map(|i| (((i as i64 % 31) as f32) - 15.0) / 16.0)
        .collect();
    let mut a = vec![0u8; qdot::col_bytes(ty, k)];
    qdot::quantize_col(ty, &col, &mut a);
    a
}

/// The pool's partition arithmetic, copied from `threads::chunk_bounds`
/// (kept local so this bin stays std-only): chunk `t` of `threads` over
/// `n` rows, the first `n % threads` chunks one row longer — exactly how
/// `for_each_chunk` splits R rows across the pool, ragged R included
/// (9333 = 21 x 292 + 11 x 291 over 32 threads).
fn chunk_bounds(n: usize, threads: usize, t: usize) -> (usize, usize) {
    debug_assert!(t < threads);
    let base = n / threads;
    let rem = n % threads;
    let start = if t < rem {
        t * (base + 1)
    } else {
        rem * (base + 1) + (t - rem) * base
    };
    (start, start + base + usize::from(t < rem))
}

// ---------------------------------------------------------------- mode 1

/// One MT-ceiling cell: `t_threads` scoped threads each stream their
/// contiguous `rows / t_threads` share of the shared buffer `passes`
/// times. The extra Barrier participant is the main thread: its `wait()`
/// return IS the start gun (workers are released by the same barrier), so
/// `t0` is taken at the true start and spawn cost stays outside the timed
/// region. Returns (weight bytes moved, seconds, checksum).
fn run_mt(sh: &Shape, w: &[u8], acol: &[u8], t_threads: usize, passes: u32) -> (f64, f64, f32) {
    let share = sh.rows / t_threads;
    let row = sh.row_bytes;
    let ty = sh.ty;
    let k = sh.k;
    let start = Barrier::new(t_threads + 1);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..t_threads)
            .map(|t| {
                let slice = &w[t * share * row..(t + 1) * share * row];
                let start = &start;
                s.spawn(move || {
                    start.wait();
                    let mut acc = 0.0f32;
                    for _ in 0..passes {
                        for r in 0..share {
                            acc += qdot::dot_row(ty, &slice[r * row..], acol, k).unwrap();
                        }
                    }
                    acc
                })
            })
            .collect();
        start.wait();
        let t0 = Instant::now();
        let sums: Vec<f32> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let dt = t0.elapsed().as_secs_f64();
        (
            passes as f64 * sh.rows as f64 * row as f64,
            dt,
            sums.iter().sum(),
        )
    })
}

/// One (shape, T) measurement: a single timed warm-up pass calibrates the
/// pass count for [`TARGET_SECS`], then the timed passes report.
fn cell_mt(sh: &Shape, w: &[u8], acol: &[u8], t_threads: usize) {
    let (_, warm, _) = run_mt(sh, w, acol, t_threads, 1);
    let passes = ((TARGET_SECS / warm).ceil() as u32).clamp(MIN_PASSES, MAX_PASSES);
    let (bytes, dt, sum) = run_mt(sh, w, acol, t_threads, passes);
    println!(
        "{:?} k={:<5} T={:>2}: {:>8} rows x {:>4} B x {:>4} passes in {:>7.3} s = {:>6.1} GB/s (sum {sum:.3})",
        sh.ty,
        sh.k,
        t_threads,
        sh.rows,
        sh.row_bytes,
        passes,
        dt,
        bytes / dt / 1e9
    );
}

// ---------------------------------------------------------------- mode 2

/// One granularity cell: 32 threads march the buffer in `calls_per_sweep`
/// calls of `rows_per_call` rows — each thread does its `chunk_bounds`
/// share, then everyone meets on the per-call Barrier, the sync stand-in
/// for a redispatch (heavier than the pool's spin-hot protocol — see the
/// header caveat). One timed warm-up sweep calibrates the sweep count; the
/// reported GB/s is the sustained rate across the timed sweeps.
fn run_gran(
    sh: &Shape,
    w: &[u8],
    acol: &[u8],
    rows_per_call: usize,
    sweeps: usize,
) -> (f64, f64, f32) {
    let calls = sh.rows / rows_per_call; // remainder rows are simply unvisited
    let row = sh.row_bytes;
    let ty = sh.ty;
    let k = sh.k;
    let call_sync = Barrier::new(T_GRAN);
    let start = Barrier::new(T_GRAN + 1);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..T_GRAN)
            .map(|t| {
                let call_sync = &call_sync;
                let start = &start;
                s.spawn(move || {
                    start.wait();
                    let (c0, c1) = chunk_bounds(rows_per_call, T_GRAN, t);
                    let width = c1 - c0;
                    let mut acc = 0.0f32;
                    for _ in 0..sweeps {
                        for call in 0..calls {
                            let base = (call * rows_per_call + c0) * row;
                            for r in 0..width {
                                acc += qdot::dot_row(ty, &w[base + r * row..], acol, k).unwrap();
                            }
                            // The dispatch stand-in: no thread starts call
                            // c+1 before every thread finished call c.
                            call_sync.wait();
                        }
                    }
                    acc
                })
            })
            .collect();
        start.wait();
        let t0 = Instant::now();
        let sums: Vec<f32> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let dt = t0.elapsed().as_secs_f64();
        (
            sweeps as f64 * calls as f64 * rows_per_call as f64 * row as f64,
            dt,
            sums.iter().sum(),
        )
    })
}

fn cell_gran(sh: &Shape, w: &[u8], acol: &[u8], rows_per_call: usize) {
    let (_, warm, _) = run_gran(sh, w, acol, rows_per_call, 1);
    let sweeps = ((TARGET_SECS / warm).ceil() as usize).clamp(MIN_SWEEPS, MAX_SWEEPS);
    let (bytes, dt, sum) = run_gran(sh, w, acol, rows_per_call, sweeps);
    println!(
        "{:?} k={:<5} T=32 R={:>6} ({:>5.1} rows/thread/call): {:>3} calls x {:>4} B x {:>3} sweeps in {:>7.3} s = {:>6.1} GB/s (sum {sum:.3})",
        sh.ty,
        sh.k,
        rows_per_call,
        rows_per_call as f64 / T_GRAN as f64,
        sh.rows / rows_per_call,
        sh.row_bytes,
        sweeps,
        dt,
        bytes / dt / 1e9
    );
}

fn main() {
    let gran = std::env::args().any(|a| a == "--granularity");
    if gran {
        println!(
            "qdot-rate-mt --granularity: T=32, std::sync::Barrier resync per R rows/call (characterizes the Barrier, NOT the pool — see the header caveat)"
        );
    } else {
        println!(
            "qdot-rate-mt: MT streaming ceiling, zero orchestration (std::thread::scope, no pool, no pinning)"
        );
    }
    for sh in shapes() {
        println!(
            "# {} — {} rows, {:.2} GB",
            sh.site,
            sh.rows,
            sh.rows as f64 * sh.row_bytes as f64 / 1e9
        );
        let w = fill(sh.rows, sh.row_bytes);
        let a = acol(sh.ty, sh.k);
        if gran {
            for r in RS {
                cell_gran(&sh, &w, &a, r);
            }
        } else {
            for t in [1usize, 8, 16, 32] {
                cell_mt(&sh, &w, &a, t);
            }
        }
        drop(w); // the ~2.5 GB goes back before the next shape allocates
    }
}
