//! engram-rate — what one token's 48 engram rows cost, arm by arm.
//!
//! Five arms over the same rows, so the differences are the levers and not the
//! draw:
//!
//! | arm | what it asks |
//! |---|---|
//! | `cold-serial` | 48 rows nobody has touched, read one at a time — queue depth 1 |
//! | `cold-prefetch` | the same rows advised first, then read — queue depth 48 |
//! | `cold-pipelined` | advise token t+1 while reading token t, the lead the engine actually has |
//! | `warm` | the rows already resident — the floor |
//! | `warm-prefetch` | resident rows advised anyway — what the prefetch costs when it buys nothing |
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
//! **This binary does not take the machine lease.** `tools/ref/engram-rate.sh`
//! does, and sets `BLOOMERY_ENGRAM_LEASE=1`; without it every line is stamped
//! `[not under lease]` and is not admissible as a measurement.

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use engram::{Engram, Faults, SeededRows, Site, faults, read_bytes};

const DEFAULT_DIR: &str = "/models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8";
const ARMS: [&str; 5] = [
    "cold-serial",
    "cold-prefetch",
    "cold-pipelined",
    "warm",
    "warm-prefetch",
];

struct Args {
    dir: String,
    rows_per_token: usize,
    tokens: usize,
    seed: u64,
    arms: Vec<String>,
    cold_reset: bool,
}

fn usage() -> String {
    format!(
        "engram-rate [--model-dir DIR] [--rows-per-token N] [--tokens N] [--seed N] [--cold-reset] [--arms {}]",
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

fn pct(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

struct Arm {
    /// Per-token wall time of the whole token (submit + read).
    total: Vec<u64>,
    /// Per-token wall time of the advises alone; empty for arms with none.
    submit: Vec<u64>,
    faults: Faults,
    read_bytes: u64,
    setup_s: f64,
}

fn run_arm(name: &str, sites: &[Site], plan: &[Vec<Vec<u32>>]) -> Result<Arm, String> {
    let mut acc = 0u64;
    let mut borrowed: Vec<&[u8]> = Vec::with_capacity(64);

    // Put the rows where this arm's label says they are.
    let t0 = Instant::now();
    if name.starts_with("cold") {
        evict_plan(sites, plan);
    } else {
        for token in plan {
            read_token(sites, token, &mut borrowed, &mut acc);
        }
    }
    let setup_s = t0.elapsed().as_secs_f64();

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
    let submit_p50 = if a.submit.is_empty() {
        "n/a".to_string()
    } else {
        format!("{:.1}", pct(&a.submit, 0.5) as f64 / 1000.0)
    };
    println!(
        "arm={name} tokens={} rows/token={} seed={} \
         p50_us={:.1} p99_us={:.1} submit_p50_us={submit_p50} \
         majflt/tok={majflt:.2} minflt/tok={:.2} \
         read_bytes/tok={read_per_token:.0} used_bytes/tok={used_bytes} amp={:.1}x \
         us_per_majflt={us_per_majflt} setup_s={:.2}{}",
        args.tokens,
        args.rows_per_token,
        args.seed,
        pct(&a.total, 0.5) as f64 / 1000.0,
        pct(&a.total, 0.99) as f64 / 1000.0,
        a.faults.minor as f64 / n,
        read_per_token / used_bytes as f64,
        a.setup_s,
        if leased { "" } else { "  [not under lease]" },
    );
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
        Ok(e) => e,
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
        match run_arm(name, sites, &plan) {
            Ok(mut arm) => report(name, &mut arm, &args, used_bytes, leased),
            Err(e) => {
                eprintln!("engram-rate: arm {name}: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}
