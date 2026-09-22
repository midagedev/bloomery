//! MT streaming ceiling for `dot_row` and dispatch-granularity probe.
//!
//! Measures sustained bandwidth across thread counts T in {1, 8, 16, 32} using
//! scoped threads over a shared ~2.5 GB buffer, and under `--granularity`
//! measures rows/call scaling at T=32 with barrier resynchronization.

use std::sync::Barrier;
use std::time::Instant;

use gguf::GgmlType;
use threads::chunk_bounds;

/// Timed-section target per cell.
const TARGET_SECS: f64 = 3.0;
/// Pass/sweep clamps for calibration.
const MIN_PASSES: u32 = 2;
const MAX_PASSES: u32 = 4000;
const MIN_SWEEPS: usize = 2;
const MAX_SWEEPS: usize = 200;

/// The granularity probe's rows/call values (T = 32 fixed).
const RS: [usize; 9] = [512, 1024, 2048, 4096, 9333, 12288, 25600, 51200, 102400];
const T_GRAN: usize = 32;

/// Site shape: type, k, row size in bytes, and buffer row count.
struct Shape {
    ty: GgmlType,
    k: usize,
    row_bytes: usize,
    /// Row count sized to ~2.5 GB to exceed L3 cache.
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

/// xorshift64* filler: any bytes are valid quantized codes, and the
/// fill touches every page to avoid faults in timed passes.
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

/// Activation column for `k` values, through `quantize_col`.
fn acol(ty: GgmlType, k: usize) -> Vec<u8> {
    let col: Vec<f32> = (0..k)
        .map(|i| (((i as i64 % 31) as f32) - 15.0) / 16.0)
        .collect();
    let mut a = vec![0u8; qdot::col_bytes(ty, k)];
    qdot::quantize_col(ty, &col, &mut a);
    a
}

// ---------------------------------------------------------------- mode 1

/// One MT-ceiling cell: `t_threads` scoped threads stream contiguous shares
/// of the shared buffer. Returns (weight bytes moved, seconds, checksum).
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

/// One (shape, T) measurement: warm-up calibrates pass count, then timed passes report.
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

/// One granularity cell: 32 threads step through the buffer in calls of `rows_per_call` rows.
fn run_gran(
    sh: &Shape,
    w: &[u8],
    acol: &[u8],
    rows_per_call: usize,
    sweeps: usize,
) -> (f64, f64, f32) {
    let calls = sh.rows / rows_per_call; // remainder rows are unvisited
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
                            // Resync threads between calls.
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
        drop(w); // free memory before the next shape allocates
    }
}
