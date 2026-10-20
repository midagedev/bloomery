//! The Solar-Open2 prefill gate: the prompt batch (`bloomery_gpu_solar2::prefill`: up to `T_MAX` rows a unit, every layer over
//! the whole unit, the KDA scan over the rows, the GQA flash over the unit, the routed block over its slots) is the one-token
//! step run per id to within the error model's band, in the logits row and in every layer's output, and the stores it leaves
//! are the ones the next step reads; the prompt group changes no bit.
//!
//! The batch keeps its own arithmetic (the GEMM sites and the f16 prefill flash), so it is the steps' within a band, not
//! bit for bit. The band is the error model's ([`SIGMA_LOGIT`](bloomery_gpu_gates::prompts::SIGMA_LOGIT), [`seat::tap_ceil`]): the two paths differ first by sum order
//! (`P = 1`: each GEMM site adds the same block dots as the gemv in another f32 order) or by the prefill flash's f16 query
//! and weights (`P ≥ 2`), and every site after quantizes its input to q8_1 blocks of 32 values. Two inputs `e` apart
//! (`e` well under the step `d = amax/127`) round to codes a whole step apart with probability `e/d` a value, so past one
//! site they are `√(e² + e·d)` apart: the map has no fixed point under `d`, and either seed grows toward two independent
//! q8 roundings within a few sites, and no further. The band is therefore two implementations' difference of one
//! quantized model, the one `gate_solar2_e2e` holds ours to PAI's by; the f16 rounding (2^-11 against q8's 1.29e-2 a
//! value) adds nothing to it. The fixture tier's four random-weight layers have no early stretch, so every layer takes the
//! wide ceiling there ([`seat::tap_ceil`]).
//!
//! The prompt lengths straddle the unit: `P = 1` (a unit of one row, the step's shape), `7` (rows short of any alignment),
//! `T_MAX` (one whole unit) and `T_MAX + 1` (a whole unit and a unit of one row: the KDA state, the conv ring and the K/V
//! planes cross the cut between two units). The ids are the fixed walk of `e2e::walk`, none of the oracle's: the clauses compare
//! the engine's two paths to each other and need no reference.
//!
//! The clauses (all self-consistency, so both tiers run them):
//! - `prefill_near_steps`: from a reset, `prompt(P ids)` and `P` steps give last logits rows within [`logprobs_near`] (the
//!   steps' top [`TOP`] ids) and layer outputs at the last position within [`seat::tap_ceil`], the same store list
//!   ([`Seat::stores`]: each KDA layer's state, stamp and conv ring, each GQA layer's K and V planes over the `P` written
//!   positions; the digests are printed, every one moves with the first rounding), and the position is `P`.
//! - `prefill_then_step_near_steps`: after `prompt(P ids)` one more step gives the logits row and layer outputs of `P + 1`
//!   steps within the same bands, so the stores a batch leaves are the ones the next step reads.
//! - `prefill_group_is_group_one`: at `P = T_MAX + 1` with the prompt group at 2 (the two units one walk, the second unit's
//!   front under the first's host serve) the prompt gives the group of 1's logits row, layer outputs and store digests bit
//!   for bit: the group orders the units' walks, not any sum. The group is put back at 1.
//! - `fault_word_clean`: the fault word is clean after every run.
//!
//! `--arm a|bp` picks the placement arm as `gate_solar2_e2e` does.
//!
//! FAIL-first (box): a batch that leaves each GQA layer's K rows in the V plane and its V rows in the K plane (its own flash
//! reading them the right way round) puts `prefill_then_step_near_steps` red at every `P` (the logits' worst z 5.2 at
//! `P = 1`, 26 to 35 at `P >= 7` with layer 0 past its ceiling) while `prefill_near_steps` and the group clause stay green.

#[cfg(feature = "solar2")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_solar2_prefill", gate::run())
}

#[cfg(feature = "solar2")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "solar2")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "solar2")]
#[allow(
    dead_code,
    reason = "the seat serves two gates; each takes the calls it reads"
)]
#[path = "shared/solar2_seat.rs"]
mod seat;

#[cfg(feature = "solar2")]
mod gate {
    use crate::e2e::{rel, top_ids, walk};
    use crate::seat::{self, Seat};
    use bloomery_gpu_gates::prompts::{k_of, log_softmax, sigma_lp};
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, verdict};
    use bloomery_gpu_solar2::body::T_MAX;

    const NAME: &str = "gate_solar2_prefill";

    /// The prompt lengths: one row, rows short of an alignment, one whole unit, a unit and a row.
    const LENGTHS: [usize; 4] = [1, 7, T_MAX, T_MAX + 1];

    /// Positions of the cache beyond the longest prompt and the step after it.
    const CTX_SPARE: usize = 8;

    /// The ids of the steps' row the logits band reads, as `gate_solar2_e2e` reads PAI's top table.
    const TOP: usize = 40;

    /// What a path left behind after a prompt: the last logits row, every layer's output and every store's digest.
    struct Left {
        logits: Vec<f32>,
        taps: Vec<f32>,
        stores: Vec<(String, u64)>,
        pos: u32,
    }

    struct Case {
        vocab: usize,
        embd: usize,
        tier: Tier,
    }

    type Clause = fn(&mut dyn Seat, &Case) -> Result<bool, GateError>;

    /// The clauses, in the order they run.
    const CLAUSES: &[(&str, Tag, Clause)] = &[
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
    fn logprobs_near(s: &[f32], b: &[f32]) -> (bool, String) {
        let (ls, lb) = (log_softmax(s), log_softmax(b));
        let sq: f64 = ls.iter().map(|&v| (2.0 * v).exp()).sum();
        let top = top_ids(s, TOP);
        let k = k_of(top.len());
        // A NaN distance is out of band and wins the fold.
        let (z, at) = top.iter().fold((0.0f64, 0usize), |w, &id| {
            let z = (lb[id] - ls[id]).abs() / sigma_lp(ls[id].exp(), sq).max(f64::MIN_POSITIVE);
            if z.is_nan() || z > w.0 { (z, id) } else { w }
        });
        let pass = z <= k && top.len() == TOP;
        let line = format!(
            "logits worst z {z:.2} at id {at} (|dlogprob| {:.4}, sigma {:.4}) over the steps' top {}, bound {k:.2} {}",
            (lb[at] - ls[at]).abs(),
            sigma_lp(ls[at].exp(), sq),
            top.len(),
            verdict(pass)
        );
        (pass, line)
    }

    /// The batch's layer outputs `b` against the steps' `s` (`embd` values a layer) by relative distance: whether every
    /// layer is within [`seat::tap_ceil`], and the line that says so.
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
            .position(|(l, &d)| d > seat::tap_ceil(case.tier, l));
        let pass = first.is_none() && s.len() == b.len() && !table.is_empty();
        let probe: Vec<String> = [0, 1, 2, 3, 4, table.len().saturating_sub(1)]
            .into_iter()
            .filter(|&l| l < table.len())
            .map(|l| format!("L{l} {:.2e}", table[l]))
            .collect();
        let line = format!(
            "layer outputs worst {worst:.2e} at layer {at} ({}), {}, first past it {first:?} {}",
            probe.join(", "),
            seat::tap_ceils(case.tier),
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
        for p in LENGTHS {
            let ids = walk(p, case.vocab);
            let (s, b) = (by_steps(seat, &ids)?, by_prompt(seat, &ids)?);
            let (logits, l_line) = logprobs_near(&s.logits, &b.logits);
            let (layers, t_line) = layers_near(&s.taps, &b.taps, case);
            let (same, differ) = store_lists(&s.stores, &b.stores);
            let pass = logits && layers && same && s.pos == b.pos && b.pos == u32::try_from(p)?;
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
        for p in LENGTHS {
            let ids = walk(p + 1, case.vocab);
            let steps = by_steps(seat, &ids)?;
            seat.reset()?;
            seat.prompt(&ids[..p])?;
            let row = seat.step(ids[p])?;
            let taps = seat.taps()?;
            let (logits, l_line) = logprobs_near(&steps.logits, &row);
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

    /// The prompt group of two over a call of two units (a whole unit and a unit of one row).
    const GROUP: usize = 2;

    fn prefill_group_is_group_one(seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
        let p = T_MAX + 1;
        let ids = walk(p, case.vocab);
        let one = by_prompt(seat, &ids)?;
        seat.set_group(GROUP)?;
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
            "prefill_group_is_group_one: P={p} group {GROUP} against group 1: logits {} layer outputs {}{} stores {}{} {}",
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

    pub fn run() -> Result<(), GateError> {
        let tier_now = Tier::from_env()?;
        let path = seat::model_path(tier_now, "gate-gpu-solar2-prefill")?;
        let ctx = T_MAX + 1 + CTX_SPARE;
        let t = std::time::Instant::now();
        let arm = seat::arm_arg()?;
        let mut seat = seat::open(&path, ctx, arm)?;
        let dims = seat.dims();
        println!(
            "load arm={} layers={} embd={} vocab={} ctx={ctx} t_max={T_MAX} in {:.1} s (runtime value)",
            arm.word(),
            dims.layers,
            dims.embd,
            dims.vocab,
            t.elapsed().as_secs_f64()
        );
        let case = Case {
            vocab: dims.vocab,
            embd: dims.embd,
            tier: tier_now,
        };
        let mut ok = true;
        for (name, tag, clause) in CLAUSES {
            if !tier::run_clause(name, *tag)? {
                continue;
            }
            let pass = clause(seat.as_mut(), &case)?;
            if !pass {
                println!("clause {name} FAIL");
            }
            ok &= pass;
        }
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the prompt batch at {LENGTHS:?} ids gives the steps' logits row and layer outputs within two \
             implementations' band, the step after it the steps' within the same, the group of 2 the group of 1's bits, and \
             the fault word stays clean"
        );
        Ok(())
    }
}
