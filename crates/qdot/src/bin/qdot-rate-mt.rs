//! MT streaming ceiling for `dot_row` and dispatch-granularity probe.
//!
//! Measures sustained bandwidth across thread counts T in {1, 8, 16, 32} using
//! scoped threads over a shared ~2.5 GB buffer, and under `--granularity`
//! measures rows/call scaling at T=32 with barrier resynchronization.
//!
//! The shapes are the [`SHAPES`] table, keyed by (type, k); `--only TYPE[,TYPE…]` runs the
//! named rows alone (a name no row takes is refused by name), and `--iq1s-generic` runs the
//! IQ1_S row through the generic decoder (`generic::dot_avx2::<Iq1s>`), the path IQ1_S rows
//! took before their hand kernel — the generic-vs-hand comparison arm.
//!
//! `--pin` runs thread `i` of T pinned to physical-core primary `i` — the pool's own
//! placement (`threads::pinned_cpus`), one cpu a core — refused by name when the host has
//! fewer physical cores than the run's largest T or a thread's pin fails.

use std::sync::{Barrier, Mutex};
use std::time::Instant;

use gguf::GgmlType;
use threads::{chunk_bounds, pin, pinned_cpus};

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

/// The shapes table, keyed by (type, k): each row is a dispatch site's type and width, its
/// block bytes as the row size, and the row count that holds the buffer near 2.5 GB.
const SHAPES: [Shape; 7] = [
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
    Shape {
        ty: GgmlType::IQ1_S,
        k: 6144,
        row_bytes: (6144 / 256) * 50, // 1200
        rows: 20 * 102_400,           // 2.46 GB
        site: "ML4 host experts IQ1_S (k=6144, 1200 B/row)",
    },
    Shape {
        ty: GgmlType::IQ2_XXS,
        k: 6144,
        row_bytes: (6144 / 256) * 66, // 1584
        rows: 15 * 102_400,           // 2.43 GB
        site: "ML4 host experts IQ2_XXS (k=6144, 1584 B/row)",
    },
];

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

/// The row dot a cell runs: `qdot::dot_row`, or — under `--iq1s-generic` — the generic IQ1_S
/// decode, the path IQ1_S rows took before their hand kernel.
type Dot = fn(GgmlType, &[u8], &[u8], usize) -> f32;

/// The fused kernel the dispatch picks.
fn dot_fused(ty: GgmlType, w: &[u8], a: &[u8], k: usize) -> f32 {
    qdot::dot_row(ty, w, a, k).unwrap()
}

/// The generic IQ1_S decode, the pre-hand-kernel path.
fn dot_iq1s_generic(ty: GgmlType, w: &[u8], a: &[u8], k: usize) -> f32 {
    assert!(
        ty == GgmlType::IQ1_S,
        "the generic arm runs IQ1_S only, got {ty:?}"
    );
    #[cfg(target_arch = "x86_64")]
    {
        assert!(
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma"),
            "the generic arm needs the vector decode's AVX2+FMA"
        );
        // SAFETY: asserted just above; the shape sized `w` and `a` for k/256 blocks.
        unsafe { qdot::generic::dot_avx2::<qdot::generic::Iq1s>(w, a, k / 256) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (w, a, k);
        unimplemented!("the generic arm's vector decode is x86_64 only")
    }
}

/// Thread `t`'s `--pin` step: pin to `pins[t]`, recording a refusal in `failed`
/// for the caller to refuse by name once the scope ends — a thread that panicked
/// here would leave the start barrier short and the run wedged.
fn pin_thread(pins: &[u32], t: usize, failed: &Mutex<Vec<String>>) {
    let cpu = pins[t];
    if let Err(e) = pin(cpu) {
        failed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(format!("thread {t} to cpu {cpu}: {e}"));
    }
}

/// Refuse a `--pin` run whose threads recorded pin refusals, by name.
fn refuse_pin_failures(failed: Mutex<Vec<String>>) {
    let failed = failed.into_inner().unwrap_or_else(|e| e.into_inner());
    assert!(failed.is_empty(), "--pin refused: {}", failed.join("; "));
}

/// One MT-ceiling cell: `t_threads` scoped threads stream contiguous shares
/// of the shared buffer, each pinning itself to `pin[t]` first under `--pin`.
/// Returns (weight bytes moved, seconds, checksum).
fn run_mt(
    sh: &Shape,
    w: &[u8],
    acol: &[u8],
    t_threads: usize,
    passes: u32,
    pin: Option<&[u32]>,
    dot: Dot,
) -> (f64, f64, f32) {
    let share = sh.rows / t_threads;
    let row = sh.row_bytes;
    let ty = sh.ty;
    let k = sh.k;
    let start = Barrier::new(t_threads + 1);
    let failed = Mutex::new(Vec::new());
    let out = std::thread::scope(|s| {
        let handles: Vec<_> = (0..t_threads)
            .map(|t| {
                let slice = &w[t * share * row..(t + 1) * share * row];
                let start = &start;
                let failed = &failed;
                s.spawn(move || {
                    if let Some(pins) = pin {
                        pin_thread(pins, t, failed);
                    }
                    start.wait();
                    let mut acc = 0.0f32;
                    for _ in 0..passes {
                        for r in 0..share {
                            acc += dot(ty, &slice[r * row..], acol, k);
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
    });
    refuse_pin_failures(failed);
    out
}

/// One (shape, T) measurement: warm-up calibrates pass count, then timed passes report.
fn cell_mt(sh: &Shape, w: &[u8], acol: &[u8], t_threads: usize, pin: Option<&[u32]>, dot: Dot) {
    let (_, warm, _) = run_mt(sh, w, acol, t_threads, 1, pin, dot);
    let passes = ((TARGET_SECS / warm).ceil() as u32).clamp(MIN_PASSES, MAX_PASSES);
    let (bytes, dt, sum) = run_mt(sh, w, acol, t_threads, passes, pin, dot);
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

/// One granularity cell: 32 threads step through the buffer in calls of `rows_per_call`
/// rows, each pinning itself to `pin[t]` first under `--pin`.
fn run_gran(
    sh: &Shape,
    w: &[u8],
    acol: &[u8],
    rows_per_call: usize,
    sweeps: usize,
    pin: Option<&[u32]>,
    dot: Dot,
) -> (f64, f64, f32) {
    let calls = sh.rows / rows_per_call; // remainder rows are unvisited
    let row = sh.row_bytes;
    let ty = sh.ty;
    let k = sh.k;
    let call_sync = Barrier::new(T_GRAN);
    let start = Barrier::new(T_GRAN + 1);
    let failed = Mutex::new(Vec::new());
    let out = std::thread::scope(|s| {
        let handles: Vec<_> = (0..T_GRAN)
            .map(|t| {
                let call_sync = &call_sync;
                let start = &start;
                let failed = &failed;
                s.spawn(move || {
                    if let Some(pins) = pin {
                        pin_thread(pins, t, failed);
                    }
                    start.wait();
                    let (c0, c1) = chunk_bounds(rows_per_call, T_GRAN, t);
                    let width = c1 - c0;
                    let mut acc = 0.0f32;
                    for _ in 0..sweeps {
                        for call in 0..calls {
                            let base = (call * rows_per_call + c0) * row;
                            for r in 0..width {
                                acc += dot(ty, &w[base + r * row..], acol, k);
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
    });
    refuse_pin_failures(failed);
    out
}

fn cell_gran(
    sh: &Shape,
    w: &[u8],
    acol: &[u8],
    rows_per_call: usize,
    pin: Option<&[u32]>,
    dot: Dot,
) {
    let (_, warm, _) = run_gran(sh, w, acol, rows_per_call, 1, pin, dot);
    let sweeps = ((TARGET_SECS / warm).ceil() as usize).clamp(MIN_SWEEPS, MAX_SWEEPS);
    let (bytes, dt, sum) = run_gran(sh, w, acol, rows_per_call, sweeps, pin, dot);
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

/// The rows `--only` names (its comma-separated types), or every row when absent: a name no
/// row takes is refused by name.
fn selected(only: Option<&str>) -> Vec<&'static Shape> {
    let names = only.map(|v| v.split(',').map(str::trim).collect::<Vec<_>>());
    let rows = SHAPES
        .iter()
        .filter(|sh| {
            names
                .as_ref()
                .is_none_or(|ns| ns.iter().any(|n| *n == format!("{:?}", sh.ty)))
        })
        .collect::<Vec<_>>();
    if let Some(ns) = &names {
        for n in ns {
            if !rows.iter().any(|sh| format!("{:?}", sh.ty) == *n) {
                let table = SHAPES
                    .iter()
                    .map(|sh| format!("{:?}", sh.ty))
                    .collect::<Vec<_>>()
                    .join(", ");
                panic!("--only {n:?}: no table row of that type — the table holds {table}");
            }
        }
    }
    rows
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let gran = args.iter().any(|a| a == "--granularity");
    let generic = args.iter().any(|a| a == "--iq1s-generic");
    let pin = args.iter().any(|a| a == "--pin");
    let only = args
        .iter()
        .position(|a| a == "--only")
        .map(|p| args.get(p + 1).expect("--only needs TYPE[,TYPE...]"))
        .map(String::as_str);
    // T_GRAN is the largest T either mode runs, so the list covers every cell;
    // thread i of a smaller T runs the first i cpus.
    let pins = pin.then(|| pinned_cpus(T_GRAN).unwrap_or_else(|e| panic!("--pin refused: {e}")));
    let pin_note = match &pins {
        Some(cpus) => format!(
            "--pin: thread i of T on cpu i, one a physical core (the pool's primaries), cpus {cpus:?}"
        ),
        None => "no pinning".to_string(),
    };
    let mut shapes = selected(only);
    if generic {
        shapes = shapes
            .into_iter()
            .filter(|sh| sh.ty == GgmlType::IQ1_S)
            .collect::<Vec<_>>();
        assert!(
            !shapes.is_empty(),
            "--iq1s-generic needs the IQ1_S row (check --only)"
        );
    }
    let dot: Dot = if generic { dot_iq1s_generic } else { dot_fused };
    if gran {
        println!(
            "qdot-rate-mt --granularity: T=32, std::sync::Barrier resync per R rows/call (characterizes the Barrier, NOT the pool — see the header caveat); {pin_note}"
        );
    } else {
        println!(
            "qdot-rate-mt: MT streaming ceiling, zero orchestration (std::thread::scope, no pool, {pin_note})"
        );
    }
    if generic {
        println!(
            "qdot-rate-mt --iq1s-generic: IQ1_S through the generic decode (the pre-hand-kernel path)"
        );
    }
    for sh in shapes {
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
                cell_gran(sh, &w, &a, r, pins.as_deref(), dot);
            }
        } else {
            for t in [1usize, 8, 16, 32] {
                cell_mt(sh, &w, &a, t, pins.as_deref(), dot);
            }
        }
        drop(w); // free memory before the next shape allocates
    }
}
