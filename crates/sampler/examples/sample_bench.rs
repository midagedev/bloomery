//! Prices `Sampler::sample` per call, and per pass, on synthetic rows.
//!
//! Run in release: `cargo run --release -p bloomery-sampler --example sample_bench`.
//! The rows are synthetic (no logits row is recorded in the tree): a peaked row has
//! one logit 8..12 above the maximum of an N(0, 2) body, a flat row is N(0, 1). Each
//! cell rotates over several rows of its kind, so no call repeats the one before. A
//! warm cell repeats calls back to back; a cold cell sweeps a buffer larger than the
//! caches between calls (outside the timer), as the host pool's expert work does
//! between two decode steps.
//!
//! The per-pass breakdown runs a frozen copy of the chain as it was before the top-k
//! scan (every row materialized as candidates, one selection) next to the real call;
//! the copy's survivors are compared with `Sampler::candidates()` on every row, so a
//! copy that drifts from the source stops the run. The same copy is the "before" arm of
//! the interleaved table: its call and the real call alternate in one process, so a
//! load that moves between runs moves both arms.
//!
//! The relative-weight arm uses byte-radix count/mass cutoffs and the benchmark
//! chain's draw with the production RNG sequence. Its table checks candidates and
//! tokens on every call;
//! mismatches are reported and disqualify a cell regardless of its timing.
//! `--check-only` prints semantic boundaries; `--threshold-only` measures only the new pair.

#[path = "sample_bench/threshold.rs"]
mod threshold;

use sampler::{Candidate, Sampler, SamplerParams};
use std::cmp::Ordering;
use std::hint::black_box;
use std::time::Instant;

const SIZES: [usize; 2] = [151_936, 154_880];
const ROWS_PER_KIND: usize = 4;
const WARMUP: usize = 100;
const WARM_CALLS: usize = 1000;
const COLD_CALLS: usize = 300;
/// 96 MiB of u64: more than the M1 Pro's 24 MB system cache and the box's L3.
const FLUSH_WORDS: usize = 12 << 20;

/// splitmix64: the bench's own input generator.
struct Gen(u64);

impl Gen {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }
    /// Uniform in `(0, 1)`.
    fn unit(&mut self) -> f64 {
        ((self.u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// Normal with standard deviation `sd` (Box-Muller).
    fn normal(&mut self, sd: f64) -> f32 {
        let r = (-2.0 * self.unit().ln()).sqrt();
        (sd * r * (std::f64::consts::TAU * self.unit()).cos()) as f32
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Peaked,
    Flat,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Peaked => "peaked",
            Kind::Flat => "flat",
        }
    }

    fn row(self, n: usize, seed: u64) -> Vec<f32> {
        let mut g = Gen(seed);
        match self {
            Kind::Peaked => {
                let mut row: Vec<f32> = (0..n).map(|_| g.normal(2.0)).collect();
                let body_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let at = (g.u64() % n as u64) as usize;
                row[at] = body_max + 8.0 + 4.0 * g.unit() as f32;
                row
            }
            Kind::Flat => (0..n).map(|_| g.normal(1.0)).collect(),
        }
    }
}

struct Set {
    name: &'static str,
    params: SamplerParams,
}

fn sets() -> [Set; 4] {
    let defaults = SamplerParams::default();
    [
        Set {
            name: "(i) defaults T0.8 k40 p0.95 m0.05",
            params: defaults,
        },
        Set {
            name: "(ii) top_k 40 only",
            params: SamplerParams {
                top_p: 1.0,
                min_p: 0.0,
                ..defaults
            },
        },
        Set {
            name: "(iii) top_p 0.95 + min_p 0.05, k 0",
            params: SamplerParams {
                top_k: 0,
                ..defaults
            },
        },
        Set {
            name: "(iv) min_p 0.05 only, k 0, p 1",
            params: SamplerParams {
                top_k: 0,
                top_p: 1.0,
                ..defaults
            },
        },
    ]
}

// ---------------------------------------------------------------- copies of the chain's passes

/// Verbatim from `crates/sampler/src/lib.rs`.
fn sanitize(v: f32) -> f32 {
    if v.is_nan() {
        f32::NEG_INFINITY
    } else {
        v + 0.0
    }
}

/// Verbatim from `crates/sampler/src/lib.rs`.
fn best_first(a: &Candidate, b: &Candidate) -> Ordering {
    b.logit.total_cmp(&a.logit).then(a.id.cmp(&b.id))
}

/// Verbatim from `crates/sampler/src/lib.rs`.
fn weight(logit: f32, max: f32, inv_t: f64) -> f64 {
    (f64::from(logit - max) * inv_t).exp()
}

/// Nanoseconds of each pass of one call.
#[derive(Default, Clone, Copy)]
struct Passes {
    build: u64,
    select: u64,
    max: u64,
    norm: u64,
    retain: u64,
    sort: u64,
    mass: u64,
    draw: u64,
}

const PASS_NAMES: [&str; 8] = [
    "build", "select", "max", "norm", "retain", "sort", "mass", "draw",
];

impl Passes {
    fn as_array(self) -> [u64; 8] {
        [
            self.build,
            self.select,
            self.max,
            self.norm,
            self.retain,
            self.sort,
            self.mass,
            self.draw,
        ]
    }
}

fn ns_since(t: Instant) -> u64 {
    u64::try_from(t.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// The chain for a penalty-free call, pass by pass, in `Sampler`'s own order.
#[derive(Default)]
struct Chain {
    cand: Vec<Candidate>,
    cum: Vec<f64>,
}

impl Chain {
    fn run(&mut self, p: &SamplerParams, logits: &[f32], u: f64) -> (u32, Passes) {
        let mut t = Passes::default();
        let s = Instant::now();
        self.cand.clear();
        self.cand.extend(
            (0..=u32::MAX)
                .zip(logits.iter().copied())
                .map(|(id, v)| Candidate {
                    id,
                    logit: sanitize(v),
                }),
        );
        t.build = ns_since(s);

        let s = Instant::now();
        let k = p.top_k as usize;
        if k > 0 && k < self.cand.len() {
            self.cand.select_nth_unstable_by(k - 1, best_first);
            self.cand.truncate(k);
        }
        t.select = ns_since(s);

        let s = Instant::now();
        let max = self
            .cand
            .iter()
            .fold(f32::NEG_INFINITY, |m, c| m.max(c.logit));
        t.max = ns_since(s);
        assert!(max.is_finite(), "the bench rows have a finite maximum");
        let top_p = p.top_p < 1.0;
        let s = Instant::now();
        let norm: f64 = if top_p {
            self.cand.iter().map(|c| weight(c.logit, max, 1.0)).sum()
        } else {
            0.0
        };
        t.norm = ns_since(s);
        let s = Instant::now();
        if p.min_p > 0.0 {
            let floor = max + p.min_p.ln();
            self.cand.retain(|c| c.logit >= floor);
        }
        t.retain = ns_since(s);
        let s = Instant::now();
        self.cand.sort_unstable_by(best_first);
        t.sort = ns_since(s);
        let s = Instant::now();
        if top_p {
            let target = f64::from(p.top_p);
            let mut mass = 0.0;
            let mut keep = self.cand.len();
            for (i, c) in self.cand.iter().enumerate() {
                mass += weight(c.logit, max, 1.0) / norm;
                if mass >= target {
                    keep = i + 1;
                    break;
                }
            }
            self.cand.truncate(keep);
        }
        t.mass = ns_since(s);

        let s = Instant::now();
        let id = self.draw(p.temperature, u);
        t.draw = ns_since(s);
        (id, t)
    }

    fn draw(&mut self, temperature: f32, u: f64) -> u32 {
        let [first, rest @ ..] = self.cand.as_slice() else {
            return 0;
        };
        if rest.is_empty() {
            return first.id;
        }
        let max = first.logit;
        let inv_t = 1.0 / f64::from(temperature);
        self.cum.clear();
        let mut total = 0.0;
        for c in &self.cand {
            total += weight(c.logit, max, inv_t);
            self.cum.push(total);
        }
        let u = u * total;
        let i = self
            .cum
            .partition_point(|&c| c <= u)
            .min(self.cand.len() - 1);
        self.cand[i].id
    }
}

// ---------------------------------------------------------------- measurement

fn stats(mut ns: Vec<u64>) -> (f64, f64, f64) {
    ns.sort_unstable();
    let at = |q: f64| ns[((ns.len() - 1) as f64 * q).round() as usize] as f64 / 1e3;
    (at(0.5), at(0.9), at(0.0))
}

/// Evicts the caches: writes `flush` end to end.
fn sweep(flush: &mut [u64]) {
    for w in flush.iter_mut() {
        *w = w.wrapping_add(1);
    }
    black_box(&flush);
}

fn main() {
    let (check_only, threshold_only) = match std::env::args().skip(1).collect::<Vec<_>>().as_slice()
    {
        [] => (false, false),
        [arg] if arg == "--check-only" => (true, false),
        [arg] if arg == "--threshold-only" => (false, true),
        _ => panic!("usage: sample_bench [--check-only | --threshold-only]"),
    };
    threshold_checks();
    if check_only {
        return;
    }
    let mut flush = vec![0u64; FLUSH_WORDS];
    println!(
        "sample_bench: {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!(
        "{:<8} {:<7} {:<36} {:<5} {:>7} {:>9} {:>9} {:>9} {:>7}",
        "V", "row", "params", "cache", "calls", "med_us", "p90_us", "min_us", "ns/elem"
    );
    let mut breakdown: Vec<String> = Vec::new();
    let mut interleaved: Vec<String> = Vec::new();
    let mut thresholds: Vec<String> = Vec::new();
    for &n in &SIZES {
        for kind in [Kind::Peaked, Kind::Flat] {
            let rows: Vec<Vec<f32>> = (0..ROWS_PER_KIND as u64)
                .map(|s| kind.row(n, 0xB100_0000 + s))
                .collect();
            for set in sets() {
                if threshold_only {
                    thresholds.push(threshold_pair(n, kind, &set, &rows));
                    continue;
                }
                for cold in [false, true] {
                    let mut s = Sampler::new(set.params).expect("the bench's parameters are valid");
                    let calls = if cold { COLD_CALLS } else { WARM_CALLS };
                    let warmup = if cold { 20 } else { WARMUP };
                    let mut ns = Vec::with_capacity(calls);
                    for i in 0..warmup + calls {
                        let row = &rows[i % rows.len()];
                        if cold {
                            sweep(&mut flush);
                        }
                        let t = Instant::now();
                        let id = s.sample(black_box(row), &[]);
                        let spent = ns_since(t);
                        black_box(id);
                        if i >= warmup {
                            ns.push(spent);
                        }
                    }
                    let (med, p90, min) = stats(ns);
                    println!(
                        "{:<8} {:<7} {:<36} {:<5} {:>7} {:>9.1} {:>9.1} {:>9.1} {:>7.2}",
                        n,
                        kind.name(),
                        set.name,
                        if cold { "cold" } else { "warm" },
                        calls,
                        med,
                        p90,
                        min,
                        med * 1e3 / n as f64
                    );
                }
                breakdown.push(passes(n, kind, &set, &rows));
                interleaved.push(interleave(n, kind, &set, &rows));
                thresholds.push(threshold_pair(n, kind, &set, &rows));
            }
        }
    }
    if !threshold_only {
        println!(
            "\nper-pass medians over {WARM_CALLS} warm calls, us (frozen pre-scan chain; survivors checked against Sampler::candidates)"
        );
        println!(
            "{:<8} {:<7} {:<36} {}",
            "V",
            "row",
            "params",
            PASS_NAMES.map(|p| format!("{p:>8}")).join(" ")
        );
        for line in breakdown {
            println!("{line}");
        }
        println!(
            "\nbefore/after, calls interleaved in one process (frozen pre-scan chain vs Sampler::sample), warm rows, median us"
        );
        println!(
            "{:<8} {:<7} {:<36} {:>9} {:>9} {:>7}",
            "V", "row", "params", "before", "after", "ratio"
        );
        for line in interleaved {
            println!("{line}");
        }
    }
    println!("\nrelative-weight cutoff vs Sampler::sample, alternating order, warm rows, us");
    println!(
        "V row params real_us threshold_us paired_saved_us p90_saved_us candidate_bad token_bad status"
    );
    for line in thresholds {
        println!("{line}");
    }
}

/// Known boundaries are printed as comparison rows, including counterexamples
/// which must remain visible instead of silently becoming a fallback.
fn threshold_checks() {
    println!("relative-weight contract rows: name candidate_equal token_equal");
    let defaults = SamplerParams::default();
    for (name, row, params) in [
        (
            "top-k-tie",
            vec![0.0; 4],
            SamplerParams {
                top_k: 2,
                top_p: 1.0,
                min_p: 0.0,
                ..defaults
            },
        ),
        (
            "top-p-tie",
            vec![0.0; 4],
            SamplerParams {
                top_k: 0,
                top_p: 0.5,
                min_p: 0.0,
                ..defaults
            },
        ),
        (
            "min-p-rounded-floor",
            vec![1.0, 1.0 + 0.05f32.ln()],
            SamplerParams {
                top_k: 0,
                top_p: 1.0,
                ..defaults
            },
        ),
        (
            "pre-temperature",
            vec![0.0, -1.0, -2.0],
            SamplerParams {
                top_k: 0,
                temperature: 0.1,
                ..defaults
            },
        ),
        (
            "pre-min-p-normalizer",
            vec![0.0, -0.1, -1.5],
            SamplerParams {
                top_k: 0,
                min_p: 0.5,
                top_p: 0.8,
                ..defaults
            },
        ),
        (
            "nonfinite",
            vec![f32::NAN, f32::INFINITY, f32::INFINITY],
            defaults,
        ),
        ("all-nan", vec![f32::NAN; 4], defaults),
        ("greedy", vec![0.0, 2.0, 2.0], SamplerParams::greedy()),
        ("empty", vec![], defaults),
    ] {
        let mut real = Sampler::new(params).expect("valid check parameters");
        let mut trial = threshold::Threshold::new(params);
        let (mut candidate_equal, mut token_equal) = (true, true);
        for _ in 0..64 {
            let expected = real.sample(&row, &[]);
            let actual = trial.sample(&row);
            candidate_equal &= trial.candidates() == real.candidates();
            token_equal &= actual == expected;
        }
        println!("{name} {candidate_equal} {token_equal}");
    }
}

fn threshold_pair(n: usize, kind: Kind, set: &Set, rows: &[Vec<f32>]) -> String {
    let mut real = Sampler::new(set.params).expect("valid benchmark parameters");
    let mut trial = threshold::Threshold::new(set.params);
    let (mut real_ns, mut trial_ns, mut saved) = (Vec::new(), Vec::new(), Vec::new());
    let (mut candidate_bad, mut token_bad) = (0, 0);
    for i in 0..WARMUP + WARM_CALLS {
        let row = black_box(&rows[i % rows.len()]);
        let mut baseline = || {
            let t = Instant::now();
            let id = black_box(real.sample(row, &[]));
            (id, ns_since(t))
        };
        let mut experiment = || {
            let t = Instant::now();
            let id = black_box(trial.sample(row));
            (id, ns_since(t))
        };
        let (b, e) = if i % 2 == 0 {
            let b = baseline();
            (b, experiment())
        } else {
            let e = experiment();
            (baseline(), e)
        };
        candidate_bad += usize::from(trial.candidates() != real.candidates());
        token_bad += usize::from(b.0 != e.0);
        if i >= WARMUP {
            real_ns.push(b.1);
            trial_ns.push(e.1);
            saved.push(b.1 as f64 / 1e3 - e.1 as f64 / 1e3);
        }
    }
    saved.sort_by(f64::total_cmp);
    let med = saved[saved.len() / 2];
    let p90 = saved[(saved.len() - 1) * 9 / 10];
    let status = if candidate_bad + token_bad > 0 {
        "INVALID"
    } else if med >= 300.0 {
        "LOCAL_GAIN_REQUIRES_CONTRACT_REVIEW"
    } else {
        "BELOW_300US"
    };
    format!(
        "{n} {} {} {:.1} {:.1} {med:.1} {p90:.1} {candidate_bad} {token_bad} {status}",
        kind.name(),
        set.name,
        stats(real_ns).0,
        stats(trial_ns).0
    )
}

/// The frozen chain and the real sampler on alternating calls over the same rows.
fn interleave(n: usize, kind: Kind, set: &Set, rows: &[Vec<f32>]) -> String {
    let mut chain = Chain::default();
    let mut real = Sampler::new(set.params).expect("the bench's parameters are valid");
    let (mut before, mut after) = (Vec::new(), Vec::new());
    for i in 0..WARMUP + WARM_CALLS {
        let row = black_box(&rows[i % rows.len()]);
        let mut old = || {
            let t = Instant::now();
            black_box(chain.run(&set.params, row, 0.5).0);
            ns_since(t)
        };
        let mut new = || {
            let t = Instant::now();
            black_box(real.sample(row, &[]));
            ns_since(t)
        };
        let (b, a) = if i % 2 == 0 {
            let b = old();
            (b, new())
        } else {
            let a = new();
            (old(), a)
        };
        if i >= WARMUP {
            before.push(b);
            after.push(a);
        }
    }
    let (b, a) = (stats(before).0, stats(after).0);
    format!(
        "{:<8} {:<7} {:<36} {:>9.1} {:>9.1} {:>6.1}x",
        n,
        kind.name(),
        set.name,
        b,
        a,
        b / a
    )
}

/// The per-pass medians of one cell, as a table line.
fn passes(n: usize, kind: Kind, set: &Set, rows: &[Vec<f32>]) -> String {
    let mut chain = Chain::default();
    let mut real = Sampler::new(set.params).expect("the bench's parameters are valid");
    let mut per_pass: Vec<Vec<u64>> = vec![Vec::new(); PASS_NAMES.len()];
    for i in 0..WARMUP + WARM_CALLS {
        let row = &rows[i % rows.len()];
        let u = 0.5;
        let (id, t) = chain.run(&set.params, black_box(row), u);
        black_box(id);
        if i < ROWS_PER_KIND {
            let _ = real.sample(row, &[]);
            assert_eq!(
                chain.cand,
                real.candidates(),
                "the copy's survivors drifted from the sampler's (row {i})"
            );
        }
        if i >= WARMUP {
            for (v, x) in per_pass.iter_mut().zip(t.as_array()) {
                v.push(x);
            }
        }
    }
    let cols: Vec<String> = per_pass
        .into_iter()
        .map(|v| format!("{:>8.1}", stats(v).0))
        .collect();
    format!(
        "{:<8} {:<7} {:<36} {}",
        n,
        kind.name(),
        set.name,
        cols.join(" ")
    )
}
