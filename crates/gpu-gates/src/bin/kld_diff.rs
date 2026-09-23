//! `kld_diff` — two of ik's KL-divergence base files compared position by
//! position (`bloomery_gpu_gates::kld`): KL(P‖Q) summed over the whole
//! vocabulary in f64 from both files' stored levels, argmax agreement, and
//! the NLL of the scored id under each.
//!
//!     kld_diff <P.kld> <Q.kld> [--ubatch N]... [--ik TAG]
//!
//! P is taken as the truth, Q is scored against it (ik's `--kl-divergence`
//! order, P its base file). The two must share `n_ctx`, `n_vocab` and every
//! id over the chunks both hold; a file with more chunks is read on the
//! other's. Printed: the mean KLD with its standard error, p50/p99/max, the
//! same KLD over ik's cut (the entries ik's own sum keeps), same top, ln(PPL_Q
//! / PPL_P); then for each `--ubatch N` the KLD by distance from the last
//! boundary of that split (boundaries at multiples of N within a chunk; bins
//! 0, 1, 2, 3, 4–7, 8–127, 128+ and 0–3 pooled); the KLD by position in the
//! chunk, in bins of 64; the 20 worst positions. A bin prints its mean with
//! its standard error (none at ten positions or fewer, ik's rule) and its
//! median: the KLD is heavy-tailed, and one position can carry a small bin's
//! mean.
//!
//! `--ik TAG` judges the file pair against the summary ik printed when it
//! scored Q's split against P itself: TAG is that `tools/ref/ik-ppl.sh --kld`
//! run, its log `<TAG>.log` beside P. The run must have read P as its base and
//! run the tree, model and ubatch of the run that wrote Q (Q's log, beside
//! Q). Each printed quantity is checked in its derived band
//! (`PosDiff::ik_kld_band`, `PosDiff::ik_nll_band`); a check outside it fails
//! the run.
//!
//! Host-only: no device code, no card, no lease, no gate lock; plain `cargo
//! build --release -p bloomery-gpu-gates --bin kld_diff` builds it.

use bloomery_gpu_gates::kld::{F32_SLACK, KldBase, Pair, PosDiff, ResultLine, compare};
use bloomery_gpu_gates::{GateError, checks_failed, exit_with, verdict};
use std::path::{Path, PathBuf};

/// The distance bins, in positions after the last boundary: `(low, high)`,
/// both inclusive.
const DISTANCE_BINS: [(usize, usize); 7] = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 7),
    (8, 127),
    (128, usize::MAX),
];

/// Positions per bin of the by-position profile.
const POSITION_BIN: usize = 64;

/// Positions listed as the worst.
const WORST: usize = 20;

/// ik prints its KLD, ln-ratio and PPL means with six decimals (`%10.6lf`).
const PRINT_HALF: f64 = 5e-7;

fn main() -> std::process::ExitCode {
    exit_with("kld_diff", run())
}

struct Args {
    p: PathBuf,
    q: PathBuf,
    ubatch: Vec<usize>,
    ik: Option<String>,
}

const USAGE: &str = "usage: kld_diff <P.kld> <Q.kld> [--ubatch N]... [--ik TAG]";

fn args() -> Result<Args, GateError> {
    let mut it = std::env::args().skip(1);
    let (Some(p), Some(q)) = (it.next(), it.next()) else {
        return Err(USAGE.into());
    };
    let (mut ubatch, mut ik) = (Vec::new(), None);
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .ok_or_else(|| format!("{flag} takes a value; {USAGE}"))?;
        match flag.as_str() {
            "--ubatch" => match value.parse::<usize>() {
                Ok(n) if n > 0 => ubatch.push(n),
                _ => return Err(format!("--ubatch {value}: a positive integer").into()),
            },
            "--ik" => ik = Some(value),
            _ => return Err(format!("{flag}: unknown; {USAGE}").into()),
        }
    }
    Ok(Args {
        p: PathBuf::from(p),
        q: PathBuf::from(q),
        ubatch,
        ik,
    })
}

fn run() -> Result<(), GateError> {
    let args = args()?;
    let p = KldBase::open_own_vocab(&args.p)?;
    let q = KldBase::open_own_vocab(&args.q)?;
    let pair = Pair::new(&p, &q)?;
    let d: Vec<PosDiff> = pair.records().map(|(a, b)| compare(&a, &b)).collect();
    if d.is_empty() {
        return Err("no scored position in the chunks both files hold".into());
    }
    println!(
        "kld_diff: P {} ({} chunk(s)), Q {} ({} chunk(s)) — n_ctx {}, n_vocab {}, {} chunk(s) \
         compared, {} positions",
        p.path().display(),
        p.n_chunk(),
        q.path().display(),
        q.n_chunk(),
        p.n_ctx(),
        p.n_vocab(),
        pair.n_chunk(),
        d.len()
    );
    let all = Stats::of(d.iter());
    summary(&d, &all);
    for &n in &args.ubatch {
        distance_profile(&d, n, all.kld);
    }
    position_profile(&d, all.kld);
    worst(&d, &args.ubatch);
    match &args.ik {
        Some(tag) => judge_ik(&pair, &d, &all, tag),
        None => Ok(()),
    }
}

/// Mean and spread over a set of positions.
struct Stats {
    n: usize,
    /// Mean KLD over the whole vocabulary, and its standard error.
    kld: f64,
    kld_se: f64,
    /// Median KLD: the distribution is heavy-tailed, and one position can
    /// carry a small bin's mean.
    kld_p50: f64,
    /// Mean KLD over ik's cut.
    kld_cut: f64,
    same_top: usize,
    /// Mean NLL under P and under Q, and the standard error of their paired
    /// difference.
    nll_p: f64,
    nll_q: f64,
    dnll_se: f64,
}

impl Stats {
    fn of<'a>(d: impl Iterator<Item = &'a PosDiff>) -> Stats {
        let (mut n, mut k, mut k2, mut cut, mut top) = (0usize, 0.0, 0.0, 0.0, 0usize);
        let (mut np, mut nq, mut dn2) = (0.0, 0.0, 0.0);
        let mut sorted = Vec::new();
        for x in d {
            n += 1;
            sorted.push(x.kld);
            k += x.kld;
            k2 += x.kld * x.kld;
            cut += x.kld_ik_cut;
            top += usize::from(x.top_p == x.top_q);
            np += x.nll_p;
            nq += x.nll_q;
            dn2 += (x.nll_q - x.nll_p) * (x.nll_q - x.nll_p);
        }
        let nf = n as f64;
        let (kld, dnll) = (k / nf, (nq - np) / nf);
        sorted.sort_by(f64::total_cmp);
        Stats {
            n,
            kld,
            kld_se: std_err(k2 / nf - kld * kld, n),
            kld_p50: if n > 0 { percentile(&sorted, 0.5) } else { 0.0 },
            kld_cut: cut / nf,
            same_top: top,
            nll_p: np / nf,
            nll_q: nq / nf,
            dnll_se: std_err(dn2 / nf - dnll * dnll, n),
        }
    }

    fn same_top_pct(&self) -> (f64, f64) {
        let f = self.same_top as f64 / self.n as f64;
        let se = if self.n > 1 {
            (f * (1.0 - f) / (self.n - 1) as f64).sqrt()
        } else {
            0.0
        };
        (100.0 * f, 100.0 * se)
    }
}

/// The standard error of a mean of `n` values with variance `var` — ik's
/// `mean_and_uncertainty`: 0 for ten values or fewer.
fn std_err(var: f64, n: usize) -> f64 {
    if var > 0.0 && n > 10 {
        (var / (n - 1) as f64).sqrt()
    } else {
        0.0
    }
}

/// ik's percentile: linear between the two sorted values around
/// `fraction · (n − 1)`.
fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let at = fraction * (sorted.len() - 1) as f64;
    let (i, frac) = (at.floor() as usize, at.fract());
    let j = (i + 1).min(sorted.len() - 1);
    (1.0 - frac) * sorted[i] + frac * sorted[j]
}

fn summary(d: &[PosDiff], all: &Stats) {
    let mut sorted: Vec<f64> = d.iter().map(|x| x.kld).collect();
    sorted.sort_by(f64::total_cmp);
    let (top, top_se) = all.same_top_pct();
    println!(
        "  mean KLD           {:.6} ± {:.6}   (whole vocabulary)",
        all.kld, all.kld_se
    );
    println!(
        "  over ik's cut      {:.6}              (entries P stores above -16 nats — the sum ik prints)",
        all.kld_cut
    );
    let zero = d.iter().filter(|x| x.kld == 0.0).count();
    println!(
        "  p50 / p99 / max    {:.6} / {:.6} / {:.6}   (exactly 0 at {zero} of {} positions)",
        percentile(&sorted, 0.5),
        percentile(&sorted, 0.99),
        sorted[sorted.len() - 1],
        d.len()
    );
    println!(
        "  same top           {top:.3} ± {top_se:.3} %   ({} of {})",
        all.same_top, all.n
    );
    println!(
        "  ln(PPL_Q/PPL_P)    {:+.6} ± {:.6}   (PPL_P {:.6}, PPL_Q {:.6})",
        all.nll_q - all.nll_p,
        all.dnll_se,
        all.nll_p.exp(),
        all.nll_q.exp()
    );
    let ties = d.iter().filter(|x| x.q_top_count > 1).count();
    let floor = d.iter().filter(|x| x.q_next_at_floor).count();
    let cut_floor: u64 = d.iter().map(|x| u64::from(x.cut_at_q_floor)).sum();
    let mass = d.iter().map(|x| (x.p_mass - 1.0).abs()).fold(0.0, f64::max);
    println!(
        "  positions with a tie at Q's top level {ties}; Q's next at its floor {floor}; entries in \
         ik's cut at Q's floor {cut_floor}; max |Σp − 1| over P {mass:.2e}"
    );
}

/// One bin of a profile: its count, mean KLD with its standard error, the
/// mean's ratio to the whole pair's, the median KLD, same top and
/// ln(PPL_Q/PPL_P).
fn bin_line(label: &str, s: &Stats, mean: f64) {
    if s.n == 0 {
        println!("  {label:<11} {:>6}", 0);
        return;
    }
    let (top, _) = s.same_top_pct();
    let ratio = if mean > 0.0 {
        format!("{:.2}", s.kld / mean)
    } else {
        "-".to_string()
    };
    // `std_err` is 0 for ten values or fewer; such a bin prints no error.
    let se = if s.n > 10 {
        format!("{:.6}", s.kld_se)
    } else {
        "-".to_string()
    };
    println!(
        "  {label:<11} {:>6}   {:.6} ± {se:<8}   {ratio:>6}   {:.6}   {:>7.3}   {:+.6}",
        s.n,
        s.kld,
        s.kld_p50,
        top,
        s.nll_q - s.nll_p
    );
}

fn distance_profile(d: &[PosDiff], n: usize, mean: f64) {
    println!(
        "KLD by distance from the last boundary of --ubatch {n} (boundaries at multiples of {n} \
         in a chunk):"
    );
    println!(
        "  distance     count   mean KLD ± SE          × mean   median     same top   ln(PPL_Q/PPL_P)"
    );
    for (lo, hi) in DISTANCE_BINS {
        let label = match (lo, hi) {
            (lo, usize::MAX) => format!("{lo}+"),
            (lo, hi) if lo == hi => format!("{lo}"),
            (lo, hi) => format!("{lo}-{hi}"),
        };
        let s = Stats::of(d.iter().filter(|x| (lo..=hi).contains(&(x.pos % n))));
        bin_line(&label, &s, mean);
    }
    let s = Stats::of(d.iter().filter(|x| x.pos % n <= 3));
    bin_line("0-3, pooled", &s, mean);
}

fn position_profile(d: &[PosDiff], mean: f64) {
    println!("KLD by position in the chunk (bins of {POSITION_BIN}):");
    println!(
        "  positions    count   mean KLD ± SE          × mean   median     same top   ln(PPL_Q/PPL_P)"
    );
    let first = d.iter().map(|x| x.pos / POSITION_BIN).min().unwrap_or(0);
    let last = d.iter().map(|x| x.pos / POSITION_BIN).max().unwrap_or(0);
    for b in first..=last {
        let s = Stats::of(d.iter().filter(|x| x.pos / POSITION_BIN == b));
        let label = format!("{}-{}", b * POSITION_BIN, (b + 1) * POSITION_BIN - 1);
        bin_line(&label, &s, mean);
    }
}

fn worst(d: &[PosDiff], ubatch: &[usize]) {
    let mut order: Vec<usize> = (0..d.len()).collect();
    order.sort_by(|&a, &b| d[b].kld.total_cmp(&d[a].kld).then(a.cmp(&b)));
    let dist: String = ubatch.iter().map(|n| format!("  d{n:<4}")).collect();
    println!("The {WORST} worst positions:");
    println!("  chunk    pos{dist}   KLD        top P → Q        nll P     nll Q     next");
    for &i in order.iter().take(WORST) {
        let x = &d[i];
        let dist: String = ubatch
            .iter()
            .map(|n| format!("  {:<5}", x.pos % n))
            .collect();
        println!(
            "  {:>5} {:>6}{dist}   {:.6}   {:>6} → {:<6}   {:>7.4}   {:>7.4}   {}",
            x.chunk, x.pos, x.kld, x.top_p, x.top_q, x.nll_p, x.nll_q, x.next
        );
    }
}

/// The tag of the run that wrote `file`: its stem.
fn tag_of(file: &Path) -> Result<String, GateError> {
    file.file_stem()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .ok_or_else(|| format!("{} has no file stem", file.display()).into())
}

/// The pair against ik's own summary of Q's split scored against P (see the
/// header). Prints one line per check; `Err` when one fails.
fn judge_ik(pair: &Pair<'_>, d: &[PosDiff], all: &Stats, tag: &str) -> Result<(), GateError> {
    let (p_tag, q_tag) = (tag_of(pair.p().path())?, tag_of(pair.q().path())?);
    let q_run = ResultLine::read(
        &pair.q().path().with_file_name(format!("{q_tag}.log")),
        &q_tag,
    )?;
    let ik = ResultLine::read(&pair.p().path().with_file_name(format!("{tag}.log")), tag)?;
    let base = ik.text("base")?;
    if base != p_tag {
        return Err(format!("{tag} scored against {base}, not P ({p_tag})").into());
    }
    let (ctx, chunks): (usize, usize) = (ik.parse("ctx")?, ik.parse("chunks")?);
    if (ctx, chunks) != (pair.p().n_ctx(), pair.n_chunk()) || pair.p().n_chunk() != chunks {
        return Err(format!(
            "{tag} ran ctx {ctx} over {chunks} chunk(s); the pair holds ctx {} over {} (P has {})",
            pair.p().n_ctx(),
            pair.n_chunk(),
            pair.p().n_chunk()
        )
        .into());
    }
    for key in ["tree", "head", "model", "ubatch"] {
        let (a, b) = (ik.text(key)?, q_run.text(key)?);
        if a != b {
            return Err(
                format!("{tag} ran {key}={a}, but Q ({q_tag}) was written with {key}={b}").into(),
            );
        }
    }
    let (k_ik, top_ik, lr_ik, ppl_base_ik): (f64, f64, f64, f64) = (
        ik.parse("kld_mean")?,
        ik.parse("same_top")?,
        ik.parse("ln_ratio")?,
        ik.parse("ppl_base")?,
    );
    println!(
        "Against ik's summary of {tag} (base {base}, ubatch {}, ctx {ctx}, {chunks} chunk(s)):",
        ik.text("ubatch")?
    );
    let nf = d.len() as f64;
    let mut pass = true;
    // `delta` is the file pair's value minus ik's. One-sided: only an excess
    // of the file pair's value fails.
    let mut check =
        |what: &str, values: String, delta: f64, one_sided: bool, band: f64, note: &str| {
            let off = if one_sided { delta } else { delta.abs() };
            let ok = off <= band;
            pass &= ok;
            println!(
                "  {what:<18} {values}: Δ {delta:+.2e}, band {band:.2e}{note} — {}",
                verdict(ok)
            );
        };

    // ik's printed means sit within each quantity's band of the file pair's;
    // see `PosDiff::ik_kld_band` and `PosDiff::ik_nll_band`.
    let cut_floor: u64 = d.iter().map(|x| u64::from(x.cut_at_q_floor)).sum();
    let k_band = d.iter().map(PosDiff::ik_kld_band).sum::<f64>() / nf + PRINT_HALF;
    let k_note = if cut_floor == 0 {
        " (Q's half step + f32, derived)".to_string()
    } else {
        format!(" (one-sided: {cut_floor} entries in ik's cut sit at Q's floor)")
    };
    let values = format!("{:.7} vs ik {k_ik:.6}", all.kld_cut);
    check(
        "mean KLD, ik's cut",
        values,
        all.kld_cut - k_ik,
        cut_floor > 0,
        k_band,
        &k_note,
    );

    // A same-top count moves only where Q's argmax is one of several entries
    // at its top level: ik ranked Q's fresh logits, the file cannot.
    let n_ik = (top_ik / 100.0 * nf).round();
    let ties = d.iter().filter(|x| x.q_top_count > 1).count();
    let values = format!("{} vs ik {n_ik} ({top_ik} %)", all.same_top);
    check(
        "same top, count",
        values,
        all.same_top as f64 - n_ik,
        false,
        ties as f64,
        " (ties at Q's top level)",
    );

    // Q's side is ik_nll_band; P's side is read the same way, f32 apart.
    let lr = all.nll_q - all.nll_p;
    let lr_band = d.iter().map(PosDiff::ik_nll_band).sum::<f64>() / nf + F32_SLACK + PRINT_HALF;
    let floor = d.iter().filter(|x| x.q_next_at_floor).count();
    let lr_note = if floor == 0 {
        " (Q's half step + f32, derived)".to_string()
    } else {
        format!(" (one-sided: Q's next at its floor at {floor} positions)")
    };
    check(
        "ln(PPL_Q/PPL_P)",
        format!("{lr:+.7} vs ik {lr_ik:+.6}"),
        lr - lr_ik,
        floor > 0,
        lr_band,
        &lr_note,
    );

    let base_band = F32_SLACK + PRINT_HALF / ppl_base_ik;
    check(
        "PPL_P",
        format!("{:.7} vs ik {ppl_base_ik:.6}", all.nll_p.exp()),
        all.nll_p - ppl_base_ik.ln(),
        false,
        base_band,
        " (in ln; P is read the same way, f32 + print)",
    );
    if pass { Ok(()) } else { Err(checks_failed()) }
}
