//! engram-rate — what one token's 48 engram rows cost, arm by arm.
//!
//! Eight arms over the same rows, so the differences are the levers and not the
//! draw:
//!
//! | arm | what it asks |
//! |---|---|
//! | `cold-serial` | 48 rows nobody has touched, read one at a time — queue depth 1 |
//! | `cold-prefetch` | the same rows advised first, then read — queue depth 48 |
//! | `cold-pipelined` | advise token t+1 while reading token t, the lead the engine actually has |
//! | `warm` | the rows already resident — the floor |
//! | `warm-prefetch` | resident rows advised anyway — what the prefetch costs when it buys nothing |
//! | `helper-touch` | a helper thread advises and copies; the copy takes the faults |
//! | `helper-populate` | the same, with `MADV_POPULATE_READ` between the advise and the copy |
//! | `helper-pipelined` | the engine's shape: submit t+1 as soon as t is taken back |
//!
//! Three counters decide whether an arm's label is true, and they are printed
//! for every arm. Major faults alone cannot: `MADV_WILLNEED` puts the folio in
//! the page cache with its read in flight, so the later touch is a **minor**
//! fault. A prefetched arm that really did IO shows ~0 major, ~51 minor and the
//! full `read_bytes`; a warm arm shows ~0 of all three.
//!
//! The prefetch arms time submission and reading separately. Their sum cannot
//! say which term of the model was wrong, and the two have opposite fixes: a
//! large submit is a syscall-batching problem, a large read is the device.
//!
//! **The first five arms time one thread's wall clock**, which is the same
//! thing as the step's cost only because that thread does everything. The
//! helper arms do not: they print the step thread's window and the helper's
//! own split side by side, because what the engine pays is the step thread's
//! serial time and nothing else.
//!
//! Expected shape of the helper arms. The step thread's window should collapse
//! to a memcpy of one token's rows plus the channel handoff, two orders below
//! the single-threaded arms. The helper's total should be about what the
//! pipelined arm costs today for `helper-touch`, and a little less for
//! `helper-populate` — populating removes the user-mode trap per fault, not the
//! kernel's fault work. `read_bytes` per token should not move: the same rows
//! are read from the same drive either way. The minor faults should move from
//! the step thread to the helper thread, which is the whole claim.
//!
//! That collapse is only visible when the step thread has other work to do
//! while the helper runs. With none, `wait` blocks for the helper's whole
//! duration and the window reads back as the helper's total by construction.
//! `--step-gap-us` spins that stand-in work between the submit and the wait of
//! `helper-pipelined`; at its default of 0 the arm has the shape the spec names
//! and measures the handoff alone.
//!
//! **This binary does not take the machine lease.** `tools/ref/engram-rate.sh`
//! does, and sets `BLOOMERY_ENGRAM_LEASE=1`; without it every line is stamped
//! `[not under lease]` and is not admissible as a measurement.

use std::env;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use engram::prefetch::{FillMode, Prefetcher};
use engram::{Engram, Faults, SeededRows, Site, faults, faults_thread, read_bytes};

const DEFAULT_DIR: &str = "/models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8";
const ARMS: [&str; 8] = [
    "cold-serial",
    "cold-prefetch",
    "cold-pipelined",
    "warm",
    "warm-prefetch",
    "helper-touch",
    "helper-populate",
    "helper-pipelined",
];

struct Args {
    dir: String,
    rows_per_token: usize,
    tokens: usize,
    seed: u64,
    arms: Vec<String>,
    cold_reset: bool,
    step_gap_us: u64,
}

fn usage() -> String {
    format!(
        "engram-rate [--model-dir DIR] [--rows-per-token N] [--tokens N] [--seed N] [--cold-reset] [--step-gap-us N] [--arms {}]",
        ARMS.join(",")
    )
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        dir: DEFAULT_DIR.to_string(),
        rows_per_token: 48,
        tokens: 2000,
        seed: 7,
        arms: ARMS.iter().map(|s| s.to_string()).collect(),
        cold_reset: false,
        step_gap_us: 0,
    };
    let argv: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].as_str();
        if flag == "-h" || flag == "--help" {
            return Err(usage());
        }
        if flag == "--cold-reset" {
            a.cold_reset = true;
            i += 1;
            continue;
        }
        let value = argv
            .get(i + 1)
            .ok_or_else(|| format!("{flag} wants a value\n{}", usage()))?;
        match flag {
            "--model-dir" => a.dir = value.clone(),
            "--rows-per-token" => {
                a.rows_per_token = value
                    .parse()
                    .map_err(|e| format!("--rows-per-token: {e}"))?;
            }
            "--tokens" => a.tokens = value.parse().map_err(|e| format!("--tokens: {e}"))?,
            "--seed" => a.seed = value.parse().map_err(|e| format!("--seed: {e}"))?,
            "--step-gap-us" => {
                a.step_gap_us = value.parse().map_err(|e| format!("--step-gap-us: {e}"))?;
            }
            "--arms" => {
                a.arms = value.split(',').map(|s| s.trim().to_string()).collect();
                if let Some(bad) = a.arms.iter().find(|s| !ARMS.contains(&s.as_str())) {
                    return Err(format!("unknown arm {bad:?}\n{}", usage()));
                }
            }
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
        i += 2;
    }
    if a.rows_per_token == 0 || a.tokens == 0 {
        return Err("--rows-per-token and --tokens must be positive".into());
    }
    Ok(a)
}

/// Every token's row ids, drawn once so all arms see the same rows.
///
/// `ids[t][s]` is site `s`'s ids for token `t`. The real split is 24 rows per
/// site (`docs/research/v41-ops.md`); here it is `rows_per_token` shared out as
/// evenly as the site count allows.
fn plan(sites: &[Site], rows_per_token: usize, tokens: usize, seed: u64) -> Vec<Vec<Vec<u32>>> {
    let mut rng = SeededRows::new(seed);
    let mut scratch = Vec::new();
    (0..tokens)
        .map(|_| {
            (0..sites.len())
                .map(|s| {
                    let n = rows_per_token / sites.len()
                        + usize::from(s < rows_per_token % sites.len());
                    rng.next_into(sites[s].rows(), n, &mut scratch);
                    scratch.clone()
                })
                .collect()
        })
        .collect()
}

/// Read every row of one token, touching the first and last byte so a row that
/// straddles a page boundary faults both. The accumulator is printed, so the
/// reads cannot be elided.
fn read_token<'a>(sites: &'a [Site], token: &[Vec<u32>], out: &mut Vec<&'a [u8]>, acc: &mut u64) {
    for (site, ids) in sites.iter().zip(token) {
        // An id out of range is a bug in the plan, not a runtime condition.
        site.rows_into(ids, out).expect("planned id is in range");
        for row in out.iter() {
            *acc = acc
                .wrapping_add(u64::from(row[0]))
                .wrapping_mul(0x100_0001)
                .wrapping_add(u64::from(row[row.len() - 1]));
        }
    }
}

/// The same fold over a token already copied into one flat buffer.
fn fold_buffer(buf: &[u8], stride: usize, acc: &mut u64) {
    for row in buf.chunks_exact(stride) {
        *acc = acc
            .wrapping_add(u64::from(row[0]))
            .wrapping_mul(0x100_0001)
            .wrapping_add(u64::from(row[row.len() - 1]));
    }
}

fn prefetch_token(sites: &[Site], token: &[Vec<u32>]) {
    for (site, ids) in sites.iter().zip(token) {
        site.prefetch(ids).expect("planned id is in range");
    }
}

fn evict_plan(sites: &[Site], plan: &[Vec<Vec<u32>>]) {
    for token in plan {
        for (site, ids) in sites.iter().zip(token) {
            site.evict_rows(ids).expect("planned id is in range");
        }
    }
}

/// Burn `us` microseconds on this thread, standing in for the work the engine's
/// step does between handing the helper a token and asking for it back. A sleep
/// would give the core away and pay a wakeup on the way back, which is the very
/// cost the measurement is trying to see.
fn spin_us(us: u64) {
    if us == 0 {
        return;
    }
    let until = Instant::now() + Duration::from_micros(us);
    while Instant::now() < until {
        std::hint::spin_loop();
    }
}

fn pct(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

fn p50_us(sorted: &[u64]) -> String {
    if sorted.is_empty() {
        "n/a".to_string()
    } else {
        format!("{:.1}", pct(sorted, 0.5) as f64 / 1000.0)
    }
}

/// The helper thread's side of a helper arm.
struct Helper {
    submit: Vec<u64>,
    fill: Vec<u64>,
    copy: Vec<u64>,
    /// Cumulative on the helper thread over the whole arm.
    faults: Faults,
    /// Taken on the step thread over the whole arm.
    step_faults: Faults,
}

struct Arm {
    /// Per-token wall time of the window the caller pays for: the whole token
    /// on the single-threaded arms, the step thread's share on the helper arms.
    total: Vec<u64>,
    /// Per-token wall time of the advises alone; empty for arms with none.
    submit: Vec<u64>,
    faults: Faults,
    read_bytes: u64,
    setup_s: f64,
    helper: Option<Helper>,
}

/// Put the rows where this arm's label says they are.
fn setup(name: &str, sites: &[Site], plan: &[Vec<Vec<u32>>]) -> f64 {
    let t0 = Instant::now();
    if name.starts_with("cold") || name.starts_with("helper") {
        evict_plan(sites, plan);
    } else {
        let mut acc = 0u64;
        let mut borrowed: Vec<&[u8]> = Vec::with_capacity(64);
        for token in plan {
            read_token(sites, token, &mut borrowed, &mut acc);
        }
    }
    t0.elapsed().as_secs_f64()
}

fn run_arm(name: &str, sites: &[Site], plan: &[Vec<Vec<u32>>]) -> Result<Arm, String> {
    let mut acc = 0u64;
    let mut borrowed: Vec<&[u8]> = Vec::with_capacity(64);

    let setup_s = setup(name, sites, plan);

    let mut total = Vec::with_capacity(plan.len());
    let mut submit =
        Vec::with_capacity(if name.contains("prefetch") || name.contains("pipelined") {
            plan.len()
        } else {
            0
        });

    let rb0 = read_bytes().map_err(|e| format!("/proc/self/io: {e}"))?;
    let f0 = faults();

    if name == "cold-pipelined" {
        // The engine knows token t+1's ids the moment token t is sampled, so the
        // advise for the next token overlaps this token's read.
        prefetch_token(sites, &plan[0]);
        for (t, token) in plan.iter().enumerate() {
            let start = Instant::now();
            if let Some(next) = plan.get(t + 1) {
                prefetch_token(sites, next);
            }
            let submitted = start.elapsed().as_nanos() as u64;
            read_token(sites, token, &mut borrowed, &mut acc);
            submit.push(submitted);
            total.push(start.elapsed().as_nanos() as u64);
        }
    } else {
        let prefetching = name.contains("prefetch");
        for token in plan {
            let start = Instant::now();
            if prefetching {
                prefetch_token(sites, token);
                submit.push(start.elapsed().as_nanos() as u64);
            }
            read_token(sites, token, &mut borrowed, &mut acc);
            total.push(start.elapsed().as_nanos() as u64);
        }
    }

    let f1 = faults();
    let rb1 = read_bytes().map_err(|e| format!("/proc/self/io: {e}"))?;
    // Printed so nothing above is dead code.
    println!("# {name} checksum {acc:#018x}");

    Ok(Arm {
        total,
        submit,
        faults: Faults {
            major: f1.major.saturating_sub(f0.major),
            minor: f1.minor.saturating_sub(f0.minor),
        },
        read_bytes: rb1.saturating_sub(rb0),
        setup_s,
        helper: None,
    })
}

/// The three arms that put the read on another core.
///
/// The step thread's window is the headline: `submit` + `wait` + one memcpy out
/// of the helper's buffer, which stands in for the staging copy the engine will
/// do on its way to the device. Everything else the read costs is the helper's,
/// and is printed as the helper's own split.
fn run_helper_arm(
    name: &str,
    engram: &Arc<Engram>,
    plan: &[Vec<Vec<u32>>],
    step_gap_us: u64,
) -> Result<Arm, String> {
    let sites = engram.sites();
    let setup_s = setup(name, sites, plan);

    let rows_per_site: Vec<usize> = plan[0].iter().map(Vec::len).collect();
    let mode = if name == "helper-populate" {
        FillMode::Populate
    } else {
        FillMode::Touch
    };
    let mut pf = Prefetcher::new(Arc::clone(engram), &rows_per_site, mode)
        .map_err(|e| format!("prefetcher: {e}"))?;

    // The step thread's own landing buffer, written once so its pages are
    // faulted in before the counters start.
    let buf_len: usize = sites
        .iter()
        .zip(&rows_per_site)
        .map(|(s, n)| n * s.row_bytes() as usize)
        .sum();
    let mut out = vec![0xA5u8; buf_len];
    // The checksum walks the flat buffer in one row stride, which it can do
    // only because every site carries 256-value Q8_0 rows. A table whose sites
    // differed would need the fold to walk them site by site.
    let stride = sites[0].row_bytes() as usize;
    assert!(
        sites.iter().all(|s| s.row_bytes() as usize == stride),
        "the checksum fold assumes one row stride for every site"
    );
    let mut acc = 0u64;

    let mut total = Vec::with_capacity(plan.len());
    let mut submit = Vec::new();
    let mut h_submit = Vec::with_capacity(plan.len());
    let mut h_fill = Vec::with_capacity(plan.len());
    let mut h_copy = Vec::with_capacity(plan.len());

    let rb0 = read_bytes().map_err(|e| format!("/proc/self/io: {e}"))?;
    let f0 = faults();
    let sf0 = faults_thread();

    if name == "helper-pipelined" {
        pf.submit(&plan[0]).map_err(|e| format!("submit: {e}"))?;
        for t in 0..plan.len() {
            // The measured window: take the token back and copy it out. The
            // submit for t+1 is issued after, and is not waited on.
            let start = Instant::now();
            pf.wait().map_err(|e| format!("wait: {e}"))?;
            out.copy_from_slice(pf.filled());
            let window = start.elapsed().as_nanos() as u64;
            let times = pf.last();

            if let Some(next) = plan.get(t + 1) {
                pf.submit(next).map_err(|e| format!("submit: {e}"))?;
            }
            // The step's other work, so the helper has somewhere to hide.
            spin_us(step_gap_us);
            fold_buffer(&out, stride, &mut acc);

            // The first token has no prefetch behind it — nothing was in flight
            // when it was submitted — so it is not one of this arm's tokens.
            if t > 0 {
                total.push(window);
                h_submit.push(times.submit_ns);
                h_fill.push(times.fill_ns);
                h_copy.push(times.copy_ns);
            }
        }
    } else {
        for token in plan {
            let start = Instant::now();
            pf.submit(token).map_err(|e| format!("submit: {e}"))?;
            submit.push(start.elapsed().as_nanos() as u64);
            pf.wait().map_err(|e| format!("wait: {e}"))?;
            out.copy_from_slice(pf.filled());
            total.push(start.elapsed().as_nanos() as u64);

            let times = pf.last();
            h_submit.push(times.submit_ns);
            h_fill.push(times.fill_ns);
            h_copy.push(times.copy_ns);
            fold_buffer(&out, stride, &mut acc);
        }
    }

    let sf1 = faults_thread();
    let f1 = faults();
    let rb1 = read_bytes().map_err(|e| format!("/proc/self/io: {e}"))?;
    let helper_faults = Faults {
        major: pf.last().major_faults,
        minor: pf.last().minor_faults,
    };
    println!("# {name} checksum {acc:#018x}");

    Ok(Arm {
        total,
        submit,
        faults: Faults {
            major: f1.major.saturating_sub(f0.major),
            minor: f1.minor.saturating_sub(f0.minor),
        },
        read_bytes: rb1.saturating_sub(rb0),
        setup_s,
        helper: Some(Helper {
            submit: h_submit,
            fill: h_fill,
            copy: h_copy,
            faults: helper_faults,
            step_faults: Faults {
                major: sf1.major.saturating_sub(sf0.major),
                minor: sf1.minor.saturating_sub(sf0.minor),
            },
        }),
    })
}

fn report(name: &str, a: &mut Arm, args: &Args, used_bytes: u64, leased: bool) {
    a.total.sort_unstable();
    a.submit.sort_unstable();
    let n = a.total.len() as f64;
    let majflt = a.faults.major as f64 / n;
    let read_per_token = a.read_bytes as f64 / n;
    // The number that tests the assumed device latency, and ik's implied 384 us.
    let us_per_majflt = if a.faults.major > 0 {
        format!("{:.1}", pct(&a.total, 0.5) as f64 / 1000.0 / majflt)
    } else {
        "n/a".to_string()
    };
    let submit_p50 = p50_us(&a.submit);
    print!(
        "arm={name} tokens={} rows/token={} seed={} \
         p50_us={:.1} p99_us={:.1} submit_p50_us={submit_p50} \
         majflt/tok={majflt:.2} minflt/tok={:.2} \
         read_bytes/tok={read_per_token:.0} used_bytes/tok={used_bytes} amp={:.1}x \
         us_per_majflt={us_per_majflt} setup_s={:.2}",
        args.tokens,
        args.rows_per_token,
        args.seed,
        pct(&a.total, 0.5) as f64 / 1000.0,
        pct(&a.total, 0.99) as f64 / 1000.0,
        a.faults.minor as f64 / n,
        read_per_token / used_bytes as f64,
        a.setup_s,
    );
    if let Some(h) = a.helper.as_mut() {
        h.submit.sort_unstable();
        h.fill.sort_unstable();
        h.copy.sort_unstable();
        // p50_us above is the step thread's window on these arms; everything
        // here is the other thread's, and the two do not add up to a token.
        print!(
            " step_gap_us={} \
             helper_submit_p50_us={} helper_fill_p50_us={} helper_copy_p50_us={} \
             step_majflt/tok={:.2} step_minflt/tok={:.2} \
             helper_majflt/tok={:.2} helper_minflt/tok={:.2}",
            args.step_gap_us,
            p50_us(&h.submit),
            p50_us(&h.fill),
            p50_us(&h.copy),
            h.step_faults.major as f64 / n,
            h.step_faults.minor as f64 / n,
            h.faults.major as f64 / n,
            h.faults.minor as f64 / n,
        );
    }
    println!("{}", if leased { "" } else { "  [not under lease]" });
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(64);
        }
    };
    let engram = match Engram::open_dir(&args.dir) {
        Ok(e) => Arc::new(e),
        Err(e) => {
            eprintln!("engram-rate: {e}");
            return ExitCode::FAILURE;
        }
    };
    let sites = engram.sites();
    let leased = env::var("BLOOMERY_ENGRAM_LEASE").is_ok_and(|v| v == "1");

    println!("# dir {}", args.dir);
    for s in sites {
        println!(
            "# site {} rows {} row_bytes {} bytes {}",
            s.name(),
            s.rows(),
            s.row_bytes(),
            s.rows() * s.row_bytes()
        );
    }
    let used_bytes: u64 = {
        let per_site = args.rows_per_token / sites.len();
        let extra = args.rows_per_token % sites.len();
        sites
            .iter()
            .enumerate()
            .map(|(i, s)| (per_site as u64 + u64::from(i < extra)) * s.row_bytes())
            .sum()
    };

    // A from-scratch reset: drop the whole table from the page cache, not just
    // the rows this run plans to read. Off by default because it evicts the
    // table for every process on the box, and because the per-arm eviction is
    // what makes an arm's own label true.
    if args.cold_reset {
        let t0 = Instant::now();
        for s in sites {
            if let Err(e) = s.evict() {
                eprintln!("engram-rate: evicting {}: {e}", s.name());
                return ExitCode::FAILURE;
            }
        }
        println!(
            "# cold-reset whole table {:.2}s",
            t0.elapsed().as_secs_f64()
        );
    }

    let plan = plan(sites, args.rows_per_token, args.tokens, args.seed);
    for name in &args.arms {
        let arm = if name.starts_with("helper") {
            run_helper_arm(name, &engram, &plan, args.step_gap_us)
        } else {
            run_arm(name, sites, &plan)
        };
        match arm {
            Ok(mut arm) => report(name, &mut arm, &args, used_bytes, leased),
            Err(e) => {
                eprintln!("engram-rate: arm {name}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}
