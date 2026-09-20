//! Where one decode step's nanoseconds go — the single owner of the stage-1 profiler.
//!
//! This module only *attributes time*; the fixes are separate rounds fed by its
//! table (`just measure-profile` owns the measurement itself).
//!
//! Design constraints, each load-bearing:
//!
//!   * **Off must be free-ish.** Every hooked hot loop calls [`level`] first; when it
//!     returns 0 the path is one atomic load and a branch — no `Instant::now()`, so an
//!     unprofiled run is the old code plus a compare per call, and `tests/forward.rs`
//!     keeps its numbers to the last digit.
//!   * **Level 1 is call-granular on purpose.** One `Instant::now()` pair per hooked
//!     call; the tax is negligible against a matmul that takes milliseconds.
//!   * **Level 2 splits stages** (activation quant / weight dequant / dot) and pays two
//!     `Instant::now()` calls *per row* for it. That tax is real, which is why
//!     [`report`] prints the warning at level 2: **the stage ratios are the finding,
//!     the absolute stage times are not.**
//!   * **The lock is taken once per call**, never per row — a mutex in the row loop
//!     would profile the mutex. Call sites accumulate into a stack [`CallAcc`] and
//!     merge under the lock once, in [`record`].
//!
//! Levels come from `BLOOMERY_PROFILE` (unset/`0` = off, `1` = calls, `2` = stages),
//! read exactly once per process into a [`OnceLock`] — flipping the variable after the
//! first hooked call does nothing, which is what the decode binary's `--profile` flag
//! relies on when it sets the variable before any model code runs.
//!
//! Sites that are not matmuls record too. Three rules for reading their rows:
//!
//!   * **Typeless sites record through [`record_time`]** under the sentinel type
//!     `GgmlType::Unknown(0)` (tag 0 is F32, so no tensor in any file can produce
//!     it) and print `-` in the `ty` column. Their `rows`/`k`/`weight MB` are 0
//!     because there is no contraction and no weight walk to count — that is the
//!     reason the row exists, not missing data. `rms_norm` (F32 gain bytes),
//!     `flash_attn_latent` (F16 KV bytes) and `f32_tensor` (the F32 byte walk
//!     itself) do read weight-shaped bytes and fill their columns honestly.
//!   * **They are level 1 only**: their stage columns print 0.00 because the
//!     quant/dequant/dot split is a matmul statement. Splitting one of them is a
//!     later round's tool, chosen once this table says which site is worth it.
//!   * **A site that wraps already-hooked calls records self time.** `wv_b_heads`
//!     times its whole call and subtracts what its sixteen `matmul_q` children
//!     recorded in the interval ([`site_ns_total`]); `gain` does the same around
//!     its `f32_tensor` child. The coverage numerator never counts the same
//!     nanoseconds twice — an outer timer that included its children would let
//!     coverage drift past 100 % and stop meaning anything.
//!
//! The gate for this module is `tests/profile.rs` (`just gate-profile`): profiling must
//! not change the logits by one bit, every hooked site must record, and the instrumented
//! sites must cover ≥ 98 % of one decode step's wall time — that last one is how an
//! unhooked hot loop gets caught.

use gguf::GgmlType;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

/// One accumulator row per (call site, weight type). `GgmlType` is part of the key
/// because the three hypotheses are per-type statements: a Q3_K `matmul_q` row and an
/// F32 router row in the same site are different dequant stories.
type Key = (&'static str, GgmlType);

#[derive(Default)]
struct Acc {
    calls: u64,
    rows: u64,
    k_total: u64,
    weight_bytes: u64,
    ns_total: u64,
    ns_quant_act: u64,
    ns_dequant_w: u64,
    ns_dot: u64,
    ns_gather: u64,
}

static LEVEL: OnceLock<u8> = OnceLock::new();

static ACCS: Mutex<BTreeMap<Key, Acc>> = Mutex::new(BTreeMap::new());

/// The profiling level: 0 off, 1 per call, 2 with the stage split. Read from
/// `BLOOMERY_PROFILE` once per process; an unparseable value means off, and values
/// above 2 clamp to 2 rather than growing a secret level table.
pub fn level() -> u8 {
    *LEVEL.get_or_init(|| match std::env::var("BLOOMERY_PROFILE") {
        Ok(v) => v.trim().parse().unwrap_or(0).min(2),
        Err(_) => 0,
    })
}

/// Whether any hook should time itself. One atomic load when off.
pub fn enabled() -> bool {
    level() > 0
}

/// Per-call stage accumulator. The hot loop fills this stack struct; [`record`] merges
/// it under the lock once, so the lock never sits inside the measured region.
#[derive(Default)]
pub struct CallAcc {
    ns_quant_act: u64,
    ns_dequant_w: u64,
    ns_dot: u64,
    /// Caller-side join aftermath: chunk sort, error scan, accumulator merge.
    /// Level 2 only — the site's wall minus quant/dequant/dot/gather is what
    /// the pool protocol and chunk-arrival skew cost.
    ns_gather: u64,
}

impl CallAcc {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_quant_act(&mut self, ns: u64) {
        self.ns_quant_act += ns;
    }

    pub fn add_dequant_w(&mut self, ns: u64) {
        self.ns_dequant_w += ns;
    }

    pub fn add_dot(&mut self, ns: u64) {
        self.ns_dot += ns;
    }

    pub fn add_gather(&mut self, ns: u64) {
        self.ns_gather += ns;
    }

    /// Fold a finished worker accumulator into this one. The row-parallel sites
    /// build one `CallAcc` per pool chunk and merge them after the join, so
    /// [`record`] still fires exactly once per call and the accumulator mutex
    /// stays out of the workers.
    pub fn add_acc(&mut self, other: &CallAcc) {
        self.ns_quant_act += other.ns_quant_act;
        self.ns_dequant_w += other.ns_dequant_w;
        self.ns_dot += other.ns_dot;
        self.ns_gather += other.ns_gather;
    }
}

/// Merge one finished call into the accumulators. Call this exactly once per hooked
/// call, after its `Instant` pair closed — never from inside the measured region.
///
/// `rows` is the weight-row count the call touched, `k` the contraction work
/// (`rows · k` multiply-accumulates is the honest shape statement, so callers pass the
/// per-row `k` times the row count), `weight_bytes` the quantized bytes read.
pub fn record(
    site: &'static str,
    ty: GgmlType,
    rows: u64,
    k: u64,
    weight_bytes: u64,
    total_ns: u64,
    acc: &CallAcc,
) {
    let mut map = ACCS.lock().expect("profiler accumulator mutex");
    let e = map.entry((site, ty)).or_default();
    e.calls += 1;
    e.rows += rows;
    e.k_total += k;
    e.weight_bytes += weight_bytes;
    e.ns_total += total_ns;
    e.ns_quant_act += acc.ns_quant_act;
    e.ns_dequant_w += acc.ns_dequant_w;
    e.ns_dot += acc.ns_dot;
    e.ns_gather += acc.ns_gather;
}

/// The typeless sibling of [`record`]: a site that reads no weight matrix of its
/// own — residual adds, rope, softmax, activation shuffling — still owes the
/// coverage table its wall time, but has no `rows`/`k`/`weight_bytes` worth
/// printing, so those land as 0 (see the module doc: that is the reason the row
/// exists, not missing data). The row keys under the sentinel
/// `GgmlType::Unknown(0)`, which `report` prints as `-`; tag 0 is F32, so no
/// tensor in any file can collide with it. Level-1 only by construction: there
/// are no matmul stages to split, and the stage columns stay 0.00.
pub fn record_time(site: &'static str, total_ns: u64) {
    let mut map = ACCS.lock().expect("profiler accumulator mutex");
    let e = map.entry((site, GgmlType::Unknown(0))).or_default();
    e.calls += 1;
    e.ns_total += total_ns;
}

/// Whole-call time already recorded under `site`, summed over weight types.
/// A site that wraps already-hooked children (`wv_b_heads` spins sixteen
/// `matmul_q` calls; `gain` wraps one `f32_tensor`) times itself whole-call and
/// subtracts what the children recorded in the interval, so the coverage
/// numerator never counts the same nanoseconds twice. Sound on one thread:
/// `record` fires on the calling thread, so a before/after diff around a call is
/// exactly that call's children and nothing else's.
#[doc(hidden)]
pub fn site_ns_total(site: &str) -> u64 {
    ACCS.lock()
        .expect("profiler accumulator mutex")
        .iter()
        .filter(|((s, _), _)| *s == site)
        .map(|(_, a)| a.ns_total)
        .sum()
}

/// Drop everything recorded so far. The decode binary calls this between the prefill
/// report and the decode loop so the two tables do not mix work.
pub fn reset() {
    ACCS.lock().expect("profiler accumulator mutex").clear();
}

/// Typed row for the gate. The gate asserts on calls and coverage; parsing the
/// formatted [`report`] for that would couple the assertions to column cosmetics.
#[doc(hidden)]
pub struct Entry {
    pub site: &'static str,
    pub ty: GgmlType,
    pub calls: u64,
    pub ns_total: u64,
}

/// The accumulators as typed rows, keyed order be damned (the gate sums per site).
#[doc(hidden)]
pub fn entries() -> Vec<Entry> {
    ACCS.lock()
        .expect("profiler accumulator mutex")
        .iter()
        .map(|((site, ty), a)| Entry {
            site,
            ty: *ty,
            calls: a.calls,
            ns_total: a.ns_total,
        })
        .collect()
}

/// Sum of every site's whole-call time — the numerator of coverage. Level-1
/// granularity on purpose: the stage columns carry the level-2 timer tax and would
/// overstate coverage.
#[doc(hidden)]
pub fn instrumented_ns() -> u64 {
    ACCS.lock()
        .expect("profiler accumulator mutex")
        .values()
        .map(|a| a.ns_total)
        .sum()
}

/// The table, worst first. Level 1 prints the call columns; level 2 adds the three
/// stage columns and the timer-tax warning. The last line is coverage: instrumented
/// time over `wall_ns`, the number that says whether a hot loop is still unhooked.
pub fn report(wall_ns: u64, label: &str) -> String {
    let lvl = level();
    let map = ACCS.lock().expect("profiler accumulator mutex");
    let mut rows: Vec<(&Key, &Acc)> = map.iter().collect();
    rows.sort_by_key(|a| std::cmp::Reverse(a.1.ns_total));

    let ms = |ns: u64| format!("{:>10.2}", ns as f64 / 1e6);
    let mut out = format!(
        "\nprofile {label}: level {lvl}, wall {:.2} ms\n",
        wall_ns as f64 / 1e6
    );
    if lvl >= 2 {
        out.push_str(&format!(
            "{:<20} {:<6} {:>8} {:>10} {:>10} {:>10} {:>7} {:>10} {:>10} {:>10} {:>10}\n",
            "site",
            "ty",
            "calls",
            "rows",
            "weight MB",
            "ms",
            "% wall",
            "quant ms",
            "dequant ms",
            "dot ms",
            "gather ms"
        ));
    } else {
        out.push_str(&format!(
            "{:<20} {:<6} {:>8} {:>10} {:>10} {:>10} {:>7}\n",
            "site", "ty", "calls", "rows", "weight MB", "ms", "% wall"
        ));
    }
    for ((site, ty), a) in rows {
        let pct = if wall_ns > 0 {
            a.ns_total as f64 / wall_ns as f64 * 100.0
        } else {
            0.0
        };
        // `Unknown(0)` is the typeless sentinel (see `record_time`), not a type
        // a file ever produced — the row says "-" instead of pretending one.
        let ty_str = if *ty == GgmlType::Unknown(0) {
            "-".to_string()
        } else {
            format!("{ty:?}")
        };
        let base = format!(
            "{:<20} {:<6} {:>8} {:>10} {:>10.2} {} {:>6.1}%",
            site,
            ty_str,
            a.calls,
            a.rows,
            a.weight_bytes as f64 / 1e6,
            ms(a.ns_total),
            pct
        );
        if lvl >= 2 {
            out.push_str(&format!(
                "{} {} {} {} {}\n",
                base,
                ms(a.ns_quant_act),
                ms(a.ns_dequant_w),
                ms(a.ns_dot),
                ms(a.ns_gather)
            ));
        } else {
            out.push_str(&base);
            out.push('\n');
        }
    }
    let instrumented: u64 = map.values().map(|a| a.ns_total).sum();
    let coverage = if wall_ns > 0 {
        instrumented as f64 / wall_ns as f64 * 100.0
    } else {
        0.0
    };
    out.push_str(&format!(
        "coverage: {} ms instrumented of {:.2} ms wall ({:.1}%)\n",
        ms(instrumented).trim(),
        wall_ns as f64 / 1e6,
        coverage
    ));
    if lvl >= 2 {
        out.push_str(
            "level 2: the stage split calls Instant::now() twice per row — timer tax \
             inflates absolute stage times. The RATIOS are the finding; the absolute \
             numbers are not. Sites that run their rows on the thread pool sum \
             per-worker timers, so their stage columns are CPU-time totals across \
             workers, not wall time.\n",
        );
    }
    out
}
