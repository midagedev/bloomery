//! engram-reuse — how often a real token stream asks for a row it already had.
//!
//! The question behind it: `crates/engram` says caching the table is not the
//! design, and that stands on a uniform draw over 384 M rows. The real hash
//! keys on n-grams, so a stream that repeats a bigram repeats the 8 rows that
//! bigram's order produces. This binary runs the real hash over a real token
//! stream and reports what a DRAM hot-row cache would have caught.
//!
//! **It reads no table bytes.** Only the shard headers, for the hash constants;
//! the 194 GiB of rows is never mapped.
//!
//! ## Input
//!
//! `--ids <file>`: one token id per line, decimal, blank lines ignored — what
//! `tools/ref/engram-corpus.sh` writes. Text rather than packed `u32` so the
//! file greps, diffs and truncates like everything else in `$BLOOMERY_DATA`.
//! The stream is one sequence: the context is reset once, at the start.
//!
//! ## Cache sizes
//!
//! A row is 272 B (256 Q8_0 values = 8 blocks × 34 B), so a budget in bytes is
//! a budget in rows: 1 GiB = 2^30 / 272 = 3,947,580 rows, 4 GiB = 15,790,320,
//! 16 GiB = 63,161,283. The table prints each capacity it used, so the
//! arithmetic is checkable from the output and not only from here. One cache is
//! shared by both sites and all 24 buckets, because the engine has one DRAM
//! budget and not six.
//!
//! ## How the hit rates are computed
//!
//! One pass, exact LRU for every capacity at once. For each access the **stack
//! distance** is counted — how many distinct rows were touched since this row
//! was last touched — with a Fenwick tree carrying a 1 at each row's most
//! recent position. An LRU of `C` rows serves the access iff that distance is
//! `< C`, so one distance per access answers every capacity consistently, and
//! a first touch (distance = none) is a compulsory miss at every size.
//!
//! ## What the tables say
//!
//! Per site and n-gram order: rows requested, distinct rows, the distinct
//! n-grams that produced them, the re-hit rate an unbounded cache would get,
//! and the share of requests the hottest 1 % of that slice's distinct rows
//! serve. The distinct n-gram column is the witness that the hash is doing what
//! it should and not only producing in-range ids: all `n_heads` buckets of one
//! order reduce the same rolling value, so distinct rows in a slice must be
//! `n_heads ×` distinct n-grams up to collisions.
//!
//! Then the whole-token view, which is the engine's number: rows per token that
//! still miss at each cache size, and the bytes that costs.

use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::process::ExitCode;

use engram::{Context, Hash};

/// 256 Q8_0 values: 8 blocks of 34 B.
const ROW_BYTES: u64 = 272;

/// The budgets the report prints, in bytes.
const BUDGETS_GIB: [u64; 3] = [1, 4, 16];

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("engram-reuse: {e}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    model_dir: String,
    ids: String,
    name: String,
    top_pct: f64,
}

fn parse_args() -> Result<Args, String> {
    let mut model_dir = env::var("BLOOMERY_V41_DIR").unwrap_or_else(|_| {
        "/models/DeepSeek-V4.1-Flash-Q3_K_M-engramQ8-tokembdBF16-attnQ8".into()
    });
    let mut ids = String::new();
    let mut name = String::new();
    let mut top_pct = 1.0;

    let mut it = env::args().skip(1);
    while let Some(a) = it.next() {
        let mut want = |what: &str| it.next().ok_or_else(|| format!("{what} wants a value"));
        match a.as_str() {
            "--model-dir" => model_dir = want("--model-dir")?,
            "--ids" => ids = want("--ids")?,
            "--name" => name = want("--name")?,
            "--top-pct" => {
                top_pct = want("--top-pct")?
                    .parse()
                    .map_err(|e| format!("--top-pct: {e}"))?;
            }
            "--help" | "-h" => {
                println!(
                    "engram-reuse --ids <token id file> [--name <label>] \
                     [--model-dir <dir>] [--top-pct <f>]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    if ids.is_empty() {
        return Err("--ids <file> is required (one token id per line)".into());
    }
    if name.is_empty() {
        name = ids
            .rsplit('/')
            .next()
            .unwrap_or(&ids)
            .trim_end_matches(".ids")
            .to_string();
    }
    Ok(Args {
        model_dir,
        ids,
        name,
        top_pct,
    })
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let hash = Hash::from_dir(&args.model_dir).map_err(|e| format!("{}: {e}", args.model_dir))?;
    let tokens = read_ids(&args.ids)?;
    if tokens.is_empty() {
        return Err(format!("{}: no token ids", args.ids));
    }

    let sites = hash.sites();
    let n_cols = hash.n_cols();
    let orders = hash.n_gram() - 1;
    let accesses = tokens.len() * sites * n_cols;

    let mut sim = Lru::new(accesses);
    // Distinct n-grams per order, the witness that the rows track the text.
    let mut ngrams: Vec<HashSet<u64>> = (0..orders).map(|_| HashSet::new()).collect();

    let mut ctx = Context::new(&hash);
    let mut ids = vec![0u32; n_cols];
    for &token in &tokens {
        ctx.push(token);
        let window = ctx.window();
        for (o, seen) in ngrams.iter_mut().enumerate() {
            seen.insert(fold(&window[..=o + 1]));
        }
        for site in 0..sites {
            hash.rows_into(site, window, &mut ids)
                .map_err(|e| e.to_string())?;
            for (bucket, &id) in ids.iter().enumerate() {
                sim.access(key_of(site, id), site, hash.order_of_bucket(bucket) - 2);
            }
        }
    }

    report(&args, &hash, &tokens, &sim, &ngrams);
    Ok(())
}

/// One site's row id as a single key. Sites are separate tables, so the same
/// row number in two of them is two different rows.
fn key_of(site: usize, id: u32) -> u64 {
    ((site as u64) << 32) | u64::from(id)
}

/// A window of mapped ids folded into one value, only to count distinct
/// windows. Not the model's hash — this never leaves this binary.
fn fold(window: &[u64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &v in window {
        h = (h ^ v).wrapping_mul(0x1000_0000_01b3);
    }
    h
}

fn read_ids(path: &str) -> Result<Vec<u32>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        out.push(
            line.parse::<u32>()
                .map_err(|e| format!("{path}:{}: '{line}': {e}", n + 1))?,
        );
    }
    Ok(out)
}

/// What one slice of the stream asked for.
#[derive(Clone, Default)]
struct Slice {
    requests: u64,
    /// Accesses whose stack distance was under each budget.
    hits: [u64; BUDGETS_GIB.len()],
    /// Accesses that had been seen before at all.
    rehits: u64,
}

/// Exact LRU for every capacity at once, by stack distance.
struct Lru {
    /// Per row: the position of its most recent access, and how often it was
    /// asked for. One map serves both the distance and the frequency tail.
    seen: HashMap<u64, Row>,
    /// A 1 at each row's most recent position; the prefix sum between two
    /// positions is the count of distinct rows touched in between.
    fenwick: Vec<u32>,
    pos: usize,
    /// Keyed by (site, n-gram order index), filled as the slices are met.
    slices: HashMap<(usize, usize), Slice>,
    caps: [u64; BUDGETS_GIB.len()],
    total: Slice,
}

#[derive(Clone, Copy)]
struct Row {
    last: usize,
    count: u32,
    site: usize,
    order: usize,
}

impl Lru {
    fn new(accesses: usize) -> Lru {
        let mut caps = [0u64; BUDGETS_GIB.len()];
        for (c, gib) in caps.iter_mut().zip(BUDGETS_GIB) {
            *c = (gib << 30) / ROW_BYTES;
        }
        Lru {
            seen: HashMap::new(),
            fenwick: vec![0; accesses + 1],
            pos: 0,
            slices: HashMap::new(),
            caps,
            total: Slice::default(),
        }
    }

    fn access(&mut self, key: u64, site: usize, order: usize) {
        self.pos += 1;
        let pos = self.pos;
        let prev = match self.seen.get_mut(&key) {
            Some(row) => {
                row.count += 1;
                let prev = row.last;
                row.last = pos;
                Some(prev)
            }
            None => {
                self.seen.insert(
                    key,
                    Row {
                        last: pos,
                        count: 1,
                        site,
                        order,
                    },
                );
                None
            }
        };

        // Distinct rows touched strictly between the two accesses: each carries
        // exactly one mark, at its own most recent position. The tree is read
        // before either mark moves.
        let distance = prev.map(|prev| {
            u64::from(
                self.sum(pos - 1)
                    .checked_sub(self.sum(prev))
                    .expect("a row's mark sits at its own latest position, inside the span"),
            )
        });

        let caps = self.caps;
        let slice = self.slices.entry((site, order)).or_default();
        slice.requests += 1;
        self.total.requests += 1;
        if let Some(distance) = distance {
            slice.rehits += 1;
            self.total.rehits += 1;
            for (i, &cap) in caps.iter().enumerate() {
                if distance < cap {
                    slice.hits[i] += 1;
                    self.total.hits[i] += 1;
                }
            }
        }

        if let Some(prev) = prev {
            self.add(prev, -1);
        }
        self.add(pos, 1);
    }

    fn add(&mut self, mut i: usize, delta: i32) {
        while i < self.fenwick.len() {
            self.fenwick[i] = self.fenwick[i].wrapping_add_signed(delta);
            i += i.isolate_lowest_one();
        }
    }

    fn sum(&self, mut i: usize) -> u32 {
        let mut s = 0u32;
        while i > 0 {
            s = s.wrapping_add(self.fenwick[i]);
            i -= i.isolate_lowest_one();
        }
        s
    }
}

fn report(args: &Args, hash: &Hash, tokens: &[u32], sim: &Lru, ngrams: &[HashSet<u64>]) {
    let sites = hash.sites();
    let orders = hash.n_gram() - 1;
    let n_tokens = tokens.len() as u64;

    // The frequency tail, per slice, out of the one map the pass filled.
    let mut counts: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
    for row in sim.seen.values() {
        counts
            .entry((row.site, row.order))
            .or_default()
            .push(row.count);
    }

    println!("# engram row reuse — corpus `{}`", args.name);
    println!();
    println!("ids: `{}` — {n_tokens} tokens", args.ids);
    println!("model: `{}`", args.model_dir);
    println!(
        "sites: {sites} (blocks {:?}), {} buckets a site = {} heads x {} orders, {} rows a token",
        hash.layer_ids(),
        hash.n_cols(),
        hash.n_heads(),
        orders,
        sites * hash.n_cols(),
    );
    println!();

    println!("## Per site and n-gram order");
    println!();
    println!(
        "| site | order | rows asked | distinct rows | distinct n-grams | rows / n-gram | re-hit (unbounded) | top {:.3} % share |",
        args.top_pct
    );
    println!("|---|---|---:|---:|---:|---:|---:|---:|");
    for site in 0..sites {
        for (order, seen) in ngrams.iter().enumerate() {
            let Some(slice) = sim.slices.get(&(site, order)) else {
                continue;
            };
            let empty = Vec::new();
            let c = counts.get(&(site, order)).unwrap_or(&empty);
            let distinct = c.len() as u64;
            let ng = seen.len() as u64;
            println!(
                "| {} | {}-gram | {} | {} | {} | {:.3} | {:.2} % | {:.2} % |",
                hash.layer_ids()[site],
                order + 2,
                slice.requests,
                distinct,
                ng,
                if ng == 0 {
                    0.0
                } else {
                    distinct as f64 / ng as f64
                },
                pct(slice.rehits, slice.requests),
                top_share(c, slice.requests, args.top_pct),
            );
        }
    }
    println!();

    println!("## Whole token — one shared cache, the engine's number");
    println!();
    println!("| cache | rows | hit rate | rows missed / token | bytes missed / token |");
    println!("|---|---:|---:|---:|---:|");
    let rows_per_token = sim.total.requests as f64 / n_tokens as f64;
    for (i, gib) in BUDGETS_GIB.iter().enumerate() {
        let hits = sim.total.hits[i];
        let missed = (sim.total.requests - hits) as f64 / n_tokens as f64;
        println!(
            "| {gib} GiB | {} | {:.2} % | {:.2} | {:.0} |",
            sim.caps[i],
            pct(hits, sim.total.requests),
            missed,
            missed * ROW_BYTES as f64,
        );
    }
    let missed = (sim.total.requests - sim.total.rehits) as f64 / n_tokens as f64;
    println!(
        "| unbounded | {} | {:.2} % | {:.2} | {:.0} |",
        sim.seen.len(),
        pct(sim.total.rehits, sim.total.requests),
        missed,
        missed * ROW_BYTES as f64,
    );
    println!();
    println!(
        "rows asked a token: {rows_per_token:.1}; distinct rows over the whole stream: {} ({:.2} GiB resident if all kept)",
        sim.seen.len(),
        sim.seen.len() as f64 * ROW_BYTES as f64 / (1u64 << 30) as f64,
    );
}

fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        100.0 * part as f64 / whole as f64
    }
}

/// The share of a slice's requests that its hottest `pct` % of distinct rows
/// serve. The percentage is of the distinct rows the slice actually touched,
/// not of the table.
fn top_share(counts: &[u32], requests: u64, pct: f64) -> f64 {
    if counts.is_empty() || requests == 0 {
        return 0.0;
    }
    let take = ((counts.len() as f64 * pct / 100.0).ceil() as usize).clamp(1, counts.len());
    let mut c = counts.to_vec();
    // Only the head has to be in order.
    c.select_nth_unstable_by(take - 1, |a, b| b.cmp(a));
    let hot: u64 = c[..take].iter().map(|&n| u64::from(n)).sum();
    100.0 * hot as f64 / requests as f64
}
