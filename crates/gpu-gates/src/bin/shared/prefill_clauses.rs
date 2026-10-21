//! The prefill gate's clauses, over a seat: the prompt batch against the steps, the step after the prompt against
//! the steps, the group of two against the group of one, and the fault word. A family's seat runs them; what the
//! family fixes is the prompt lengths, the band of a layer's output distance and the group's size.
//!
//! The clauses are self-consistency: the engine's two paths (the prompt batch and the one-token steps) are compared
//! to each other, so both tiers run them and no oracle is read.

use crate::e2e::{rel, top_ids, walk};
use crate::lcpp_e2e::Seat;
use bloomery_gpu_gates::prompts::{k_of, log_softmax, sigma_lp};
use bloomery_gpu_gates::tier::{self, Tag, Tier};
use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, verdict};

/// The ids of the steps' row the logits band reads, as the end-to-end gate reads the oracle's top table.
const TOP: usize = 40;

/// What the prefill clauses read of the family: the prompt lengths, the band of a layer's output distance, the
/// group's size and the model's widths.
pub struct Case {
    /// The prompt lengths the clauses run.
    pub lengths: Vec<usize>,
    /// The ceiling of a layer's output distance.
    pub tap_ceil: fn(Tier, usize) -> f64,
    /// The ceilings of [`Case::tap_ceil`] as a line prints them.
    pub tap_ceils: fn(Tier) -> String,
    /// The group the group clause runs at.
    pub group: usize,
    /// The bound of the logits' distance, or `None` when the family defers the comparison: a
    /// top-4 near tie can flip between the batch's GEMM sum order and the step's gemv, and a
    /// flip is a discrete exchange of experts, not a Gaussian, so no sigma derives from it.
    pub logit_bound: Option<f64>,
    /// The model's vocabulary.
    pub vocab: usize,
    /// The model's width.
    pub embd: usize,
    /// The tier the run is on.
    pub tier: Tier,
}

/// What a path left behind after a prompt: the last logits row, every layer's output and every store's digest.
struct Left {
    logits: Vec<f32>,
    taps: Vec<f32>,
    stores: Vec<(String, u64)>,
    pos: u32,
}

type Clause = fn(&mut dyn Seat, &Case) -> Result<bool, GateError>;

/// The clauses, in the order they run.
pub const CLAUSES: &[(&str, Tag, Clause)] = &[
    (
        "prefill_near_steps",
        Tag::SelfConsistency,
        prefill_near_steps,
    ),
    (
        "prefill_then_step_near_steps",
        Tag::SelfConsistency,
        prefill_then_step_near_steps,
    ),
    (
        "prefill_group_is_group_one",
        Tag::SelfConsistency,
        prefill_group_is_group_one,
    ),
    ("fault_word_clean", Tag::SelfConsistency, fault_word_clean),
];

fn left(seat: &mut dyn Seat, logits: Vec<f32>) -> Result<Left, GateError> {
    Ok(Left {
        logits,
        taps: seat.taps()?,
        stores: seat.stores()?,
        pos: seat.pos(),
    })
}

/// A reset, then `ids` one a step.
fn by_steps(seat: &mut dyn Seat, ids: &[u32]) -> Result<Left, GateError> {
    seat.reset()?;
    let mut row = Vec::new();
    for &id in ids {
        row = seat.step(id)?;
    }
    left(seat, row)
}

/// A reset, then `ids` as one prompt call.
fn by_prompt(seat: &mut dyn Seat, ids: &[u32]) -> Result<Left, GateError> {
    seat.reset()?;
    let row = seat.prompt(ids)?;
    left(seat, row)
}

/// The batch's row `b` against the steps' row `s` over the steps' top [`TOP`] ids: whether every id's
/// `|Δ logprob| / σ_i` ([`sigma_lp`] by the steps' probabilities) is within [`k_of`], and the line that says so.
// PIN(2026-10-20): was bit for bit; the batch keeps its GEMM sites and f16 flash (performance first), so the band is
// two implementations' ([`SIGMA_LOGIT`]: the batch and the steps are two once their seed has grown to the q8 step, module
// doc). FAIL-first: the K and V planes swapped in the batch's appends are past it on the next step.
fn logprobs_near(s: &[f32], b: &[f32], bound: Option<f64>) -> (bool, String) {
    let (ls, lb) = (log_softmax(s), log_softmax(b));
    let sq: f64 = ls.iter().map(|&v| (2.0 * v).exp()).sum();
    let top = top_ids(s, TOP);
    let k = bound.unwrap_or_else(|| k_of(top.len()));
    // A NaN distance is out of band and wins the fold.
    let (z, at) = top.iter().fold((0.0f64, 0usize), |w, &id| {
        let z = (lb[id] - ls[id]).abs() / sigma_lp(ls[id].exp(), sq).max(f64::MIN_POSITIVE);
        if z.is_nan() || z > w.0 { (z, id) } else { w }
    });
    let pass = (bound.is_none() || z <= k) && top.len() == TOP;
    let bound_line = match bound {
        Some(_) => format!("bound {k:.2}"),
        None => {
            "the comparison deferred (a flip of the experts is a discrete exchange)".to_string()
        }
    };
    let line = format!(
        "logits worst z {z:.2} at id {at} (|dlogprob| {:.4}, sigma {:.4}) over the steps' top {}, {bound_line} {}",
        (lb[at] - ls[at]).abs(),
        sigma_lp(ls[at].exp(), sq),
        top.len(),
        verdict(pass)
    );
    (pass, line)
}

/// The batch's layer outputs `b` against the steps' `s` (`embd` values a layer) by relative distance: whether every
/// layer is within the case's ceiling, and the line that says so.
fn layers_near(s: &[f32], b: &[f32], case: &Case) -> (bool, String) {
    let table: Vec<f64> = s
        .chunks(case.embd)
        .zip(b.chunks(case.embd))
        .map(|(x, y)| rel(y, x))
        .collect();
    let (worst, at) =
        table.iter().enumerate().fold(
            (0.0f64, 0usize),
            |w, (l, &d)| if d > w.0 { (d, l) } else { w },
        );
    // `rel` is infinite on a NaN, so a NaN layer is past its ceiling.
    let first = table
        .iter()
        .enumerate()
        .position(|(l, &d)| d > (case.tap_ceil)(case.tier, l));
    let pass = first.is_none() && s.len() == b.len() && !table.is_empty();
    let probe: Vec<String> = [0, 1, 2, 3, 4, table.len().saturating_sub(1)]
        .into_iter()
        .filter(|&l| l < table.len())
        .map(|l| format!("L{l} {:.2e}", table[l]))
        .collect();
    let line = format!(
        "layer outputs worst {worst:.2e} at layer {at} ({}), {}, first past it {first:?} {}",
        probe.join(", "),
        (case.tap_ceils)(case.tier),
        verdict(pass)
    );
    (pass, line)
}

/// Whether two store lists name the same stores in the same order, and how many of their digests differ.
fn store_lists(a: &[(String, u64)], b: &[(String, u64)]) -> (bool, usize) {
    let same = a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.0 == y.0);
    let differ = a.iter().zip(b).filter(|(x, y)| x.1 != y.1).count();
    (same, differ)
}

/// The stores whose digests differ, by name, or whose lists differ in kind.
fn store_diff(a: &[(String, u64)], b: &[(String, u64)]) -> Vec<String> {
    if !store_lists(a, b).0 {
        return vec![format!(
            "the store lists differ ({} against {} stores)",
            a.len(),
            b.len()
        )];
    }
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.1 != y.1)
        .map(|(x, _)| x.0.clone())
        .collect()
}

fn prefill_near_steps(seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
    let mut ok = true;
    for p in &case.lengths {
        let ids = walk(*p, case.vocab);
        let (s, b) = (by_steps(seat, &ids)?, by_prompt(seat, &ids)?);
        let (logits, l_line) = logprobs_near(&s.logits, &b.logits, case.logit_bound);
        let (layers, t_line) = layers_near(&s.taps, &b.taps, case);
        let (same, differ) = store_lists(&s.stores, &b.stores);
        let pass = logits && layers && same && s.pos == b.pos && b.pos == u32::try_from(*p)?;
        println!(
            "prefill_near_steps: P={p}: store list {} ({} stores, {differ} digests moved) position {} {}",
            verdict(same),
            b.stores.len(),
            b.pos,
            verdict(pass)
        );
        println!("  {l_line}");
        println!("  {t_line}");
        ok &= pass;
    }
    Ok(ok)
}

fn prefill_then_step_near_steps(seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
    let mut ok = true;
    for p in &case.lengths {
        let ids = walk(p + 1, case.vocab);
        let steps = by_steps(seat, &ids)?;
        seat.reset()?;
        seat.prompt(&ids[..*p])?;
        let row = seat.step(ids[*p])?;
        let taps = seat.taps()?;
        let (logits, l_line) = logprobs_near(&steps.logits, &row, case.logit_bound);
        let (layers, t_line) = layers_near(&steps.taps, &taps, case);
        let pass = logits && layers;
        println!(
            "prefill_then_step_near_steps: P={p}: the step after the prompt against {} steps {}",
            p + 1,
            verdict(pass)
        );
        println!("  {l_line}");
        println!("  {t_line}");
        ok &= pass;
    }
    Ok(ok)
}

fn prefill_group_is_group_one(seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
    let group = case.group;
    let p = *case.lengths.last().unwrap_or(&1);
    let ids = walk(p, case.vocab);
    let one = by_prompt(seat, &ids)?;
    seat.set_group(group)?;
    let b = by_prompt(seat, &ids);
    seat.set_group(1)?;
    let b = b?;
    let differ = store_diff(&one.stores, &b.stores);
    let first = one
        .taps
        .chunks(case.embd)
        .zip(b.taps.chunks(case.embd))
        .position(|(x, y)| !bits_equal(x, y));
    let pass = bits_equal(&one.logits, &b.logits)
        && bits_equal(&one.taps, &b.taps)
        && differ.is_empty()
        && b.pos == u32::try_from(p)?;
    println!(
        "prefill_group_is_group_one: P={p} group {group} against group 1: logits {} layer outputs {}{} stores {}{} {}",
        verdict(bits_equal(&one.logits, &b.logits)),
        verdict(bits_equal(&one.taps, &b.taps)),
        first.map_or(String::new(), |l| format!(" (first layer {l})")),
        verdict(differ.is_empty()),
        if differ.is_empty() {
            String::new()
        } else {
            format!(" (differ: {})", differ.join(", "))
        },
        verdict(pass)
    );
    Ok(pass)
}

fn fault_word_clean(seat: &mut dyn Seat, _case: &Case) -> Result<bool, GateError> {
    let fault = seat.fault();
    println!("fault_word_clean: {fault:?} {}", verdict(fault.is_ok()));
    Ok(fault.is_ok())
}

/// The clauses over `seat`, the tally and the gate's own line: what a prefill gate's run is.
pub fn run(name: &str, case: &Case, seat: &mut dyn Seat) -> Result<(), GateError> {
    let mut ok = true;
    for (clause, tag, f) in CLAUSES {
        if !tier::run_clause(clause, *tag)? {
            continue;
        }
        let pass = f(seat, case)?;
        if !pass {
            println!("clause {clause} FAIL");
        }
        ok &= pass;
    }
    println!("{}", tier::tally_line());
    if !ok {
        return Err(checks_failed());
    }
    println!(
        "PASSED: {name} the prompt batch at {:?} ids gives the steps' logits row and layer outputs within two \
         implementations' band, the step after it the steps' within the same, the group of 2 the group of 1's bits, and \
         the fault word stays clean",
        case.lengths
    );
    Ok(())
}
