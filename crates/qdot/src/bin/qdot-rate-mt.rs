//! MT streaming ceiling for `dot_row` and dispatch-granularity probe.
//!
//! Measures sustained bandwidth across thread counts T in {1, 8, 16, 32} using
//! scoped threads over a shared ~2.5 GB buffer, and under `--granularity`
//! measures rows/call scaling at T=32 with barrier resynchronization.
//! `--site <substr>` runs only the shapes whose site names the substring.

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

fn shapes() -> [Shape; 8] {
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
        // Bolt's IQ2_XXS experts: the routed up over the 1024-value latent,
        // the routed down over the widest routed width, the shared up over
        // the hidden row.
        Shape {
            ty: GgmlType::IQ2_XXS,
            k: 1024,
            row_bytes: iq2xxs_row_bytes(1024), // 264
            rows: 92 * 102_400,                // 2.49 GB
            site: "Bolt routed up IQ2_XXS (k=1024, 264 B/row)",
        },
        Shape {
            ty: GgmlType::IQ2_XXS,
            k: 2560,
            row_bytes: iq2xxs_row_bytes(2560), // 660
            rows: 37 * 102_400,                // 2.50 GB
            site: "Bolt routed down IQ2_XXS (k=2560, 660 B/row)",
        },
        Shape {
            ty: GgmlType::IQ2_XXS,
            k: 4096,
            row_bytes: iq2xxs_row_bytes(4096), // 1056
            rows: 23 * 102_400,                // 2.49 GB
            site: "Bolt shared up IQ2_XXS (k=4096, 1056 B/row)",
        },
    ]
}

/// An IQ2_XXS weight row's bytes for `k` values: `k`'s whole blocks (the
/// type's 256-value granularity) times the block size `GgmlType` carries.
/// [`check_iq2xxs_row_bytes`] ties this to the row size `dot_row` itself
/// demands, so neither constant can drift alone.
fn iq2xxs_row_bytes(k: usize) -> usize {
    let ty = GgmlType::IQ2_XXS;
    let block = ty.type_size().expect("IQ2_XXS has a block size") as usize;
    (k / qdot::k_granularity(ty)) * block
}

/// Every IQ2_XXS shape's `row_bytes` is the size `dot_row` itself demands for
/// its k: the empty-row refusal's `need` comes from the lib's own block
/// constant, so a granularity or block size that drifted apart is named here
/// before any cell runs.
fn check_iq2xxs_row_bytes(all: &[Shape]) {
    for sh in all.iter().filter(|sh| sh.ty == GgmlType::IQ2_XXS) {
        let Err(qdot::QdotError::ShortWeightRow { need, .. }) =
            qdot::dot_row(sh.ty, &[], &[], sh.k)
        else {
            panic!(
                "qdot-rate-mt: {} (k={}): dot_row must refuse an empty weight row",
                sh.site, sh.k
            );
        };
        assert_eq!(
            sh.row_bytes, need,
            "{}: row_bytes must equal dot_row's row size at k={}",
            sh.site, sh.k
        );
    }
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

// ---------------------------------------------------------------- site runs

/// The command line: the mode flag and the site selection.
struct Args {
    gran: bool,
    site: Option<String>,
}

/// `--granularity` selects mode 2; `--site <substr>` runs only the shapes
/// whose site contains the substring. Other arguments stay ignored, as they
/// always were.
fn parse_args(args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut gran = false;
    let mut site = None;
    let mut args = args.peekable();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--granularity" => gran = true,
            "--site" => {
                site = Some(
                    args.next()
                        .ok_or_else(|| "--site wants a substring".to_owned())?,
                );
            }
            _ => {}
        }
    }
    Ok(Args { gran, site })
}

/// The shapes a run holds: every shape, or with `--site` those whose site
/// contains the substring, in table order. A substring naming nothing is the
/// error listing every site.
fn selected<'a>(all: &'a [Shape], site: Option<&str>) -> Result<Vec<&'a Shape>, String> {
    let Some(want) = site else {
        return Ok(all.iter().collect());
    };
    let hit: Vec<&Shape> = all.iter().filter(|sh| sh.site.contains(want)).collect();
    if hit.is_empty() {
        return Err(format!(
            "--site '{want}' names no shape; the sites are:\n  {}",
            all.iter()
                .map(|sh| sh.site)
                .collect::<Vec<_>>()
                .join("\n  ")
        ));
    }
    Ok(hit)
}

fn main() {
    let args = parse_args(std::env::args().skip(1)).unwrap_or_else(|e| {
        eprintln!("qdot-rate-mt: {e}");
        std::process::exit(2);
    });
    let all = shapes();
    let run = selected(&all, args.site.as_deref()).unwrap_or_else(|e| {
        eprintln!("qdot-rate-mt: {e}");
        std::process::exit(2);
    });
    check_iq2xxs_row_bytes(&all);
    if args.gran {
        println!(
            "qdot-rate-mt --granularity: T=32, std::sync::Barrier resync per R rows/call (characterizes the Barrier, NOT the pool — see the header caveat)"
        );
    } else {
        println!(
            "qdot-rate-mt: MT streaming ceiling, zero orchestration (std::thread::scope, no pool, no pinning)"
        );
    }
    for sh in run {
        println!(
            "# {} — {} rows, {:.2} GB",
            sh.site,
            sh.rows,
            sh.rows as f64 * sh.row_bytes as f64 / 1e9
        );
        let w = fill(sh.rows, sh.row_bytes);
        let a = acol(sh.ty, sh.k);
        if args.gran {
            for r in RS {
                cell_gran(sh, &w, &a, r);
            }
        } else {
            for t in [1usize, 8, 16, 32] {
                cell_mt(sh, &w, &a, t);
            }
        }
        drop(w); // free memory before the next shape allocates
    }
}

#[cfg(test)]
mod tests {
    use gguf::GgmlType;

    use super::{Shape, selected, shapes};

    /// The sites a selection holds, in order.
    fn sites<'a>(v: &[&'a Shape]) -> Vec<&'a str> {
        v.iter().map(|s| s.site).collect()
    }

    /// FAIL-first: every IQ2_XXS shape's `row_bytes` is the size `dot_row`
    /// itself demands for its k — the empty-row refusal's `need`, straight
    /// from the lib's own block constant — and its site names that size, so a
    /// drifted constant or a stale site string fails here by site. row bytes
    /// computed from a wrong block size or granularity fail this.
    #[test]
    fn iq2xxs_row_bytes_are_what_dot_row_demands() {
        for sh in shapes() {
            if sh.ty != GgmlType::IQ2_XXS {
                continue;
            }
            let Err(qdot::QdotError::ShortWeightRow { need, .. }) =
                qdot::dot_row(sh.ty, &[], &[], sh.k)
            else {
                panic!(
                    "{} (k={}): dot_row must refuse an empty weight row",
                    sh.site, sh.k
                );
            };
            assert_eq!(
                sh.row_bytes, need,
                "{}: dot_row's row size at k={}",
                sh.site, sh.k
            );
            assert!(
                sh.site.contains(&format!("{} B/row", sh.row_bytes)),
                "{}: the site must name its row size",
                sh.site
            );
        }
    }

    /// `--site` holds the shapes whose site contains the substring, in table
    /// order (the five Q rows first, the three IQ2_XXS rows last, their
    /// strings unchanged); no `--site` holds every shape; a substring naming
    /// nothing is the refusal listing every site.
    #[test]
    fn site_selection_runs_substring_matches_in_order() {
        let all = shapes();
        let every = selected(&all, None).expect("no --site runs every shape");
        assert_eq!(
            sites(&every),
            [
                "batch Q3_K + matmul_q Q3_K (k=2048, 880 B/row)",
                "matmul_q Q4_K (k=2048, 1152 B/row)",
                "matmul_q Q6_K (k=2048, 1680 B/row)",
                "batch Q5_0 (k=1408, 968 B/row)",
                "matmul_q Q5_1 (k=10944, 8208 B/row)",
                "Bolt routed up IQ2_XXS (k=1024, 264 B/row)",
                "Bolt routed down IQ2_XXS (k=2560, 660 B/row)",
                "Bolt shared up IQ2_XXS (k=4096, 1056 B/row)",
            ]
        );
        let shared = selected(&all, Some("Bolt shared")).expect("the shared site is named");
        assert_eq!(
            sites(&shared),
            ["Bolt shared up IQ2_XXS (k=4096, 1056 B/row)"]
        );
        let iq2 = selected(&all, Some("IQ2_XXS")).expect("the IQ2_XXS sites are named");
        assert_eq!(iq2.len(), 3);
        let Err(none) = selected(&all, Some("nomatch")).map(|hit| hit.len()) else {
            panic!("a substring naming nothing is refused");
        };
        assert!(none.contains("--site 'nomatch' names no shape"), "{none}");
        for site in sites(&every) {
            assert!(none.contains(site), "{none} must list {site}");
        }
    }
}
