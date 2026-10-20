//! The Solar-Open2 end-to-end gate: the whole decode chain — 36 KDA layers (low-rank forget and output gates, depthwise conv,
//! beta = 2 sigmoid, softplus decay) and 12 NoPE gated GQA layers (64 query heads over 8 key heads, the gate by its own
//! projection), every layer with a 320-expert routed block and one shared expert, the head and the argmax — on the gate card,
//! against the PAI tree's CPU dump of the same file (`refset::arch::solar2`, `tools/ref/solar2-dump.sh`: llama.cpp 6ea215d17 plus
//! prometheusAIR's solar_open2 patch, the only faithful implementation; ik has none, LRR's port omits beta = 2 sigmoid).
//!
//! The prompts are the dump's four (`tools/ref/data/solar2`): a an English paragraph (41 ids), b a Korean one (40), c a Rust
//! function (114), d the first 1,536 ids of wikitext-2; each is fed one id a step from a reset, and the oracle's reads are the
//! set's own: the last position's whole logits row (`result_output`), each layer's output (`l_out-L`, sets a, b and c), the
//! 40 largest logprobs (`topk.tsv`) and the 64 greedy tokens with their margins (`greedy.tsv`, sets a, b and c).
//!
//! The clauses are a list ([`CLAUSES`]); each prints lines ending in PASS or FAIL and carries its tier tag
//! (`tier::run_clause`):
//! - `finite_and_fault_clean` (self-consistency): every prompt's logits row and layer outputs are finite and the fault word is
//!   clean.
//! - `reset_rewinds` (self-consistency): after a reset prompt a gives the same logits and layer outputs bit for bit.
//! - `layer_outputs_near_pai` (oracle) and its fixture twin `layer_outputs_near_pai_fx` (fixture-oracle, the `fx_` sets
//!   dumped on the fixture file): each layer's output at the last position of sets a, b and c against PAI's `l_out-L`, by the
//!   relative distance `‖ours − pai‖ / ‖pai‖`, every layer within [`tap_ceil`] (tight over the first [`TAP_EARLY`] layers, wide
//!   after). The first layer past it is named: a beta that is not doubled, or a decay of another form, moves layer 1's output
//!   (the first KDA layer) by the order of its own contribution.
//! - `top_logprobs_in_band` (oracle): on each prompt, over the ids both our top 40 and PAI's top 40 hold, the log-softmax of
//!   the logits row within the error model's band of PAI's logprob ([`sigma_lp`] an id, the worst of the ids within
//!   [`k_of`]), and our top 10 sharing at least [`overlap_floor`] ids with PAI's. Both are two implementations' difference
//!   of one quantized model ([`SIGMA_LOGIT`]); the argmax is printed, not a clause (PAI's top-1 flips between its own cuts
//!   on two prompts).
//! - `forced_greedy_count_pin` (oracle): PAI's greedy tokens of a, b and c fed as input (teacher-forced), our argmax at each of
//!   the 192 positions; the disagreements at a PAI margin of at least `MARGIN_FLOOR` are held to the pin derived from the margins
//!   themselves ([`forced_pin`]).
//!
//! - `prefill_near_pai` (oracle): the same prompts fed by the prompt batch (`Seat::prompt`) from a reset, their layer outputs and
//!   logprobs held to the bands the steps are, so the batch is as near PAI as the one-token step is (the prefill gate holds the
//!   two paths to each other).
//!
//! The real tier runs every clause but the fixture's; the fixture tier runs the self-consistency clauses and
//! `layer_outputs_near_pai_fx` and leaves the others to the real tier by name (random weights give a near-flat head, so the
//! logprob, argmax and greedy clauses have no meaning there). A set absent from the box, dumped from another file or build, or
//! unfinished ends the gate by name before the card loads.
//!
//! `--arm a|bp` picks the placement arm (default `a`: the largest card the stage; `bp`: the next-largest card an expert tier under
//! the host tier, refused by name until the tier piece lands). The clauses are the arm's own: the same bands hold on both.
//!
//! FAIL-first (box, on the first run of the body): the engine's beta scale set to 1 puts `layer_outputs_near_pai` red at layer
//! 1 and `top_logprobs_in_band` red; a decay kernel of the GLM form (sigmoid, lower bound) the same; a prompt that skips the
//! shared expert's add, the layer outputs red at layer 0.

#[cfg(feature = "solar2")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_solar2_e2e", gate::run())
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
    use crate::e2e::{argmax, rel, tap, top_ids};
    use crate::seat::{self, Dims, Seat};
    use bloomery_gpu_gates::prompts::{
        GreedyRow, MARGIN_FLOOR, SIGMA_LOGIT, compare_forced, k_of, log_softmax, sigma_lp,
        std_normal_cdf,
    };
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{GateError, RefManifest, bits_equal, checks_failed, verdict};
    use refset::arch::solar2::read::{self, Greedy, Top};
    use refset::arch::solar2::{self, fixture};
    use refset::family::Family;

    const NAME: &str = "gate_solar2_e2e";

    /// The fixture-oracle clauses the gate declares.
    const FIXTURE_ORACLE: usize = 1;

    /// Positions of the cache beyond the longest prompt and its continuation.
    const CTX_SPARE: usize = 16;

    /// The standard deviation of the top1-top2 margin of ours against PAI's: two implementations ([`SIGMA_LOGIT`]; PAI's CPU
    /// dump is the second), each σ from the exact one.
    const SIGMA_PAIR: f64 = SIGMA_LOGIT * std::f64::consts::SQRT_2;

    /// The draws [`overlap_floor`] takes.
    const DRAWS: usize = 20_000;

    /// The top-10 overlap floor against the row `lp` (log-probabilities) over its ranked ids `top`, and the mean overlap the
    /// model expects.
    ///
    /// [derived]: two implementations' logits differ by an independent N(0, σ²) an id ([`SIGMA_LOGIT`]). The ten of `top`
    /// so moved are drawn [`DRAWS`] times (a fixed seed, so the floor is the row's alone; the ids past `top` trail its last by
    /// more), and the floor is the least overlap the draws fall below no more often than one Gaussian past three standard
    /// deviations, Φ(−3).
    fn overlap_floor(lp: &[f64], top: &[usize]) -> (f64, usize) {
        let ten = top.len().min(10);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        // splitmix64, and its top 53 bits as a uniform in (0, 1).
        let mut unit = move || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 + 0.5 / (1u64 << 53) as f64
        };
        let mut hist = [0usize; 11];
        let mut moved: Vec<(f64, usize)> = Vec::with_capacity(top.len());
        for _ in 0..DRAWS {
            moved.clear();
            for (rank, &id) in top.iter().enumerate() {
                // Box-Muller: one standard normal from two uniforms.
                let g = (-2.0 * unit().ln()).sqrt() * (std::f64::consts::TAU * unit()).cos();
                moved.push((lp[id] + SIGMA_LOGIT * g, rank));
            }
            moved.sort_by(|x, y| y.0.total_cmp(&x.0));
            hist[moved[..ten].iter().filter(|m| m.1 < ten).count()] += 1;
        }
        let mean = hist
            .iter()
            .enumerate()
            .map(|(c, &n)| (c * n) as f64)
            .sum::<f64>()
            / DRAWS as f64;
        let rate = std_normal_cdf(-3.0) * DRAWS as f64;
        let (mut below, mut floor) = (0usize, 0usize);
        for (c, &n) in hist.iter().enumerate() {
            if (below + n) as f64 > rate {
                floor = c;
                break;
            }
            below += n;
        }
        (mean, floor)
    }

    /// The layers before the stream carries its outlier channels: PAI's own CPU and GPU runs agree to a few percent there.
    const TAP_EARLY: usize = 4;

    /// The ceiling of a layer output's relative distance from PAI's: [`TAP_EARLY_CEIL`] over the first [`TAP_EARLY`] layers,
    /// [`TAP_CEIL`] after.
    ///
    /// [derived]: PAI's CPU dump against its hybrid GPU run of the same prompt (the oracle round's per-tensor last-row sums of
    /// squares and first 8 values) gives the norm distance at most 0.024 and, from the 8 values, an estimate of
    /// `‖Δ‖ / ‖ref‖` (the rms-normalised error, if the error is spread evenly) of 0.008-0.022 at layers 0-3 and 0.034-0.158
    /// from layer 4 on (median 0.106): the stream grows its outlier channels there and the same roundings weigh more. Ours
    /// against PAI is a third implementation of that kind, so each ceiling is about twice its stretch's worst. A beta that is
    /// not doubled, or a decay of another form, moves layer 1 by the order of its own contribution, past the early ceiling.
    // PIN(2026-10-20): provisional until the first body runs on the box; the first run prints every layer's distance and the
    // ceilings are re-pinned there with the full-row distances.
    const TAP_EARLY_CEIL: f64 = 0.05;
    const TAP_CEIL: f64 = 0.3;

    fn tap_ceil(layer: usize) -> f64 {
        if layer < TAP_EARLY {
            TAP_EARLY_CEIL
        } else {
            TAP_CEIL
        }
    }

    /// One prompt's oracle: its set, ids, PAI's last logits row, top table and greedy continuation.
    struct Oracle {
        name: &'static str,
        man: RefManifest,
        ids: Vec<u32>,
        pai_logits: Vec<f32>,
        top: Vec<Top>,
        greedy: Greedy,
        /// Whether the set holds every layer's output.
        layer_outputs: bool,
    }

    /// What the body did on a prompt, fed one id a step from a reset.
    struct Run {
        logits: Vec<f32>,
        taps: Vec<f32>,
    }

    struct Case {
        dims: Dims,
        oracles: Vec<Oracle>,
        runs: Vec<Run>,
    }

    type Clause = fn(&mut dyn Seat, &Case) -> Result<bool, GateError>;

    /// The clauses, in the order they run.
    const CLAUSES: &[(&str, Tag, Clause)] = &[
        (
            "finite_and_fault_clean",
            Tag::SelfConsistency,
            finite_and_fault,
        ),
        ("reset_rewinds", Tag::SelfConsistency, reset_rewinds),
        ("layer_outputs_near_pai", Tag::Oracle, layer_outputs),
        (
            "layer_outputs_near_pai_fx",
            Tag::FixtureOracle,
            layer_outputs,
        ),
        ("top_logprobs_in_band", Tag::Oracle, top_logprobs),
        ("forced_greedy_count_pin", Tag::Oracle, forced_greedy),
        ("prefill_near_pai", Tag::Oracle, prefill_near_pai),
    ];

    // ------------------------------------------------------------ the oracle

    /// The set `name` of `family` opened through its check and read: the tier decides the path (the real set under
    /// `Family::path`, the fixture's through `tier::fixture_set`).
    fn open_oracle(
        tier_now: Tier,
        name: &'static str,
        family: &'static Family,
    ) -> Result<Oracle, GateError> {
        let path = match tier_now {
            Tier::Real => family.path(name),
            Tier::Fixture => tier::fixture_set("the PAI clauses", family, name)?,
        };
        let man = RefManifest::open(&path, family)?;
        let ids = read::tokens(&man)?.to_vec();
        let vocab = usize::try_from(man.tensor("result_output", 0)?.ne[0])?;
        let pai_logits = crate::e2e::ik_last(&man, vocab)?;
        let top = read::topk(&man)?;
        let greedy = read::greedy(&man)?;
        // The set's own tables are the logits row's: a dump whose top-k table disagrees with its logits is no oracle.
        let lp = log_softmax(&pai_logits);
        let consistent = top.first().map(|t| t.id) == Some(argmax(&pai_logits))
            && top
                .iter()
                .all(|t| (lp[t.id as usize] - t.logprob).abs() < 1e-4);
        if !consistent {
            return Err(format!(
                "{}: topk.tsv is not the log-softmax of its result_output row",
                path.display()
            )
            .into());
        }
        let layer_outputs = man.tensor("l_out-0", 0).is_ok();
        Ok(Oracle {
            name,
            man,
            ids,
            pai_logits,
            top,
            greedy,
            layer_outputs,
        })
    }

    fn open_oracles(tier_now: Tier) -> Result<Vec<Oracle>, GateError> {
        let (family, names): (&'static Family, [&'static str; 4]) = match tier_now {
            Tier::Real => (&solar2::PAI, [solar2::A, solar2::B, solar2::C, solar2::D]),
            Tier::Fixture => (
                &fixture::PAI,
                [fixture::A, fixture::B, fixture::C, fixture::D],
            ),
        };
        names
            .into_iter()
            .map(|n| open_oracle(tier_now, n, family))
            .collect()
    }

    // ------------------------------------------------------------ the body

    /// A reset, then `ids` one a step; the last row.
    fn feed(seat: &mut dyn Seat, ids: &[u32]) -> Result<Vec<f32>, GateError> {
        seat.reset()?;
        let mut row = Vec::new();
        for &id in ids {
            row = seat.step(id)?;
        }
        Ok(row)
    }

    fn run_of(seat: &mut dyn Seat, ids: &[u32]) -> Result<Run, GateError> {
        let logits = feed(seat, ids)?;
        Ok(Run {
            logits,
            taps: seat.taps()?,
        })
    }

    /// A reset, then `ids` as one prompt call.
    fn run_by_prompt(seat: &mut dyn Seat, ids: &[u32]) -> Result<Run, GateError> {
        seat.reset()?;
        let logits = seat.prompt(ids)?;
        Ok(Run {
            logits,
            taps: seat.taps()?,
        })
    }

    // ------------------------------------------------------------ the clauses

    fn finite_and_fault(seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
        let mut ok = true;
        for (o, r) in case.oracles.iter().zip(&case.runs) {
            let finite = r.logits.iter().chain(&r.taps).all(|v| v.is_finite())
                && r.logits.len() == case.dims.vocab
                && r.taps.len() == case.dims.layers * case.dims.embd;
            println!(
                "finite_and_fault_clean: prompt {} {} {}",
                o.name,
                o.ids.len(),
                verdict(finite)
            );
            ok &= finite;
        }
        let fault = seat.fault();
        println!(
            "finite_and_fault_clean: fault word {fault:?} {}",
            verdict(fault.is_ok())
        );
        Ok(ok && fault.is_ok())
    }

    fn reset_rewinds(seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
        let (o, first) = (&case.oracles[0], &case.runs[0]);
        let again = run_of(seat, &o.ids)?;
        let pass = bits_equal(&again.logits, &first.logits) && bits_equal(&again.taps, &first.taps);
        println!(
            "reset_rewinds: prompt {} again from a reset, logits and layer outputs bit for bit {}",
            o.name,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Each layer's output at the last position against PAI's `l_out-L`: the last `embd` values of the node (the last layer's
    /// holds the last position alone, the others every position).
    fn layer_outputs(_seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
        layer_outputs_of("layer_outputs", case, &case.runs)
    }

    /// [`layer_outputs`] over `runs`, one a prompt of the case, its lines headed `label`.
    fn layer_outputs_of(label: &str, case: &Case, runs: &[Run]) -> Result<bool, GateError> {
        let Dims { layers, embd, .. } = case.dims;
        let mut ok = true;
        for (o, r) in case
            .oracles
            .iter()
            .zip(runs)
            .filter(|(o, _)| o.layer_outputs)
        {
            let mut table = Vec::with_capacity(layers);
            for l in 0..layers {
                let pai = tap(&o.man, &format!("l_out-{l}"))?;
                let row = pai
                    .len()
                    .checked_sub(embd)
                    .map(|at| &pai[at..])
                    .ok_or_else(|| {
                        format!(
                            "{}: l_out-{l} holds {} values, a layer output is {embd}",
                            o.name,
                            pai.len()
                        )
                    })?;
                table.push(rel(&r.taps[l * embd..(l + 1) * embd], row));
            }
            let (worst, at) =
                table.iter().enumerate().fold(
                    (0.0f64, 0usize),
                    |w, (l, &d)| if d > w.0 { (d, l) } else { w },
                );
            let first = table.iter().enumerate().position(|(l, &d)| d > tap_ceil(l));
            let pass = first.is_none();
            println!(
                "{label}: prompt {} ({} ids): worst {worst:.4} at layer {at}, ceilings {TAP_EARLY_CEIL} (layers < {TAP_EARLY}) and {TAP_CEIL}, first past \
                 it {first:?} {}",
                o.name,
                o.ids.len(),
                verdict(pass)
            );
            println!(
                "  by layer: {}",
                table
                    .iter()
                    .map(|d| format!("{d:.4}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            ok &= pass;
        }
        Ok(ok)
    }

    fn top_logprobs(_seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
        top_logprobs_of("top_logprobs", case, &case.runs)
    }

    /// [`top_logprobs`] over `runs`, one a prompt of the case, its lines headed `label`.
    // PIN(2026-10-20): the band was PAI's own cut-to-cut spread (a 0.424, b 0.438, c 0.683, d 0.853), one implementation's
    // instability, not ours against it; now two implementations' ([`sigma_lp`], [`k_of`], [`overlap_floor`]). FAIL-first:
    // s2step's M3 (worst 2.08 to 8.61) and M5 (d 2.28) are past the widest band an id can have, 3.99 · √2 · 0.35 = 1.97;
    // M1, M2 and M4 share at most one top-10 id, under every floor.
    fn top_logprobs_of(label: &str, case: &Case, runs: &[Run]) -> Result<bool, GateError> {
        let mut ok = true;
        for (o, r) in case.oracles.iter().zip(runs) {
            // PAI's side from its own logits row, which `open_oracle` held its top table to.
            let (lp, pai) = (log_softmax(&r.logits), log_softmax(&o.pai_logits));
            let sq: f64 = pai.iter().map(|&v| (2.0 * v).exp()).sum();
            let ours = top_ids(&r.logits, o.top.len());
            let pai_top: Vec<usize> = o.top.iter().map(|t| t.id as usize).collect();
            let shared: Vec<usize> = pai_top
                .iter()
                .copied()
                .filter(|id| ours.contains(id))
                .collect();
            let k = k_of(shared.len());
            // A NaN distance is out of band and wins the fold.
            let (z, at) = shared.iter().fold((0.0f64, 0usize), |w, &id| {
                let z =
                    (lp[id] - pai[id]).abs() / sigma_lp(pai[id].exp(), sq).max(f64::MIN_POSITIVE);
                if z.is_nan() || z > w.0 { (z, id) } else { w }
            });
            let (e, floor) = overlap_floor(&pai, &pai_top);
            let overlap = pai_top
                .iter()
                .take(10)
                .filter(|id| ours[..10].contains(id))
                .count();
            let pass = z <= k && overlap >= floor;
            println!(
                "{label}: prompt {} ({} ids): worst z {z:.2} at id {at} (|dlogprob| {:.4}, sigma {:.4}) over {} shared ids of {}, \
                 bound {k:.2}; top-10 overlap {overlap}, floor {floor} (mean {e:.2}); argmax ours {} pai {} {}",
                o.name,
                o.ids.len(),
                (lp[at] - pai[at]).abs(),
                sigma_lp(pai[at].exp(), sq),
                shared.len(),
                o.top.len(),
                ours[0],
                o.top[0].id,
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// The prompts fed by the prompt batch instead of steps: finite, then the layer outputs and the logprobs held to the steps'
    /// own bands.
    fn prefill_near_pai(seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
        let mut runs = Vec::with_capacity(case.oracles.len());
        for o in &case.oracles {
            runs.push(run_by_prompt(seat, &o.ids)?);
        }
        let finite = runs
            .iter()
            .all(|r| r.logits.iter().chain(&r.taps).all(|v| v.is_finite()));
        println!(
            "prefill_near_pai: every prompt's logits row and layer outputs finite {}",
            verdict(finite)
        );
        let layers = layer_outputs_of("prefill_near_pai layer_outputs", case, &runs)?;
        let logprobs = top_logprobs_of("prefill_near_pai top_logprobs", case, &runs)?;
        let fault = seat.fault();
        println!(
            "prefill_near_pai: fault word {fault:?} {}",
            verdict(fault.is_ok())
        );
        Ok(finite && layers && logprobs && fault.is_ok())
    }

    /// The disagreements the Gaussian error of two implementations predicts at the clear positions, and the pin that follows:
    /// the expected count plus three standard deviations of it (Poisson), rounded up.
    ///
    /// [derived]: at a PAI margin `m` the argmax flips with probability Φ(−m/σ_pair) ([`SIGMA_PAIR`]); the sum over the positions
    /// at `m >= MARGIN_FLOOR` is the expected count of disagreements the floor classifies as clear.
    // PIN(2026-10-20): provisional with `SIGMA_LOGIT`; the pin is computed from the set's margins, not a literal.
    fn forced_pin(rows: &[GreedyRow]) -> (f64, usize) {
        let expected: f64 = rows
            .iter()
            .flat_map(|r| &r.gen_margins)
            .filter(|&&m| m >= MARGIN_FLOOR)
            .map(|&m| std_normal_cdf(-f64::from(m) / SIGMA_PAIR))
            .sum();
        let pin = (expected + 3.0 * expected.sqrt()).ceil();
        (expected, pin as usize)
    }

    fn forced_greedy(seat: &mut dyn Seat, case: &Case) -> Result<bool, GateError> {
        let (mut rows, mut ours) = (Vec::new(), Vec::new());
        for o in case.oracles.iter().filter(|o| !o.greedy.ids.is_empty()) {
            let g = &o.greedy;
            let mut row = feed(seat, &o.ids)?;
            let mut table = vec![argmax(&row)];
            for &id in &g.ids[..g.ids.len() - 1] {
                row = seat.step(id)?;
                table.push(argmax(&row));
            }
            ours.push(table);
            rows.push(GreedyRow {
                id: rows.len(),
                n_tokens: o.ids.len(),
                argmax: o.top[0].id,
                top5: o.top.iter().take(5).map(|t| t.id).collect(),
                gen_ids: g.ids.clone(),
                gen_margins: g.margins.clone(),
            });
        }
        let report = compare_forced(&ours, &rows, MARGIN_FLOOR);
        let (expected, pin) = forced_pin(&rows);
        let pass = report.disagree_ge_floor <= pin;
        println!(
            "forced_greedy: positions={} disagree={} buckets={:?} ge_floor={} expected {expected:.2}, pin {pin}, max margin {:?} {}",
            report.positions,
            report.disagree.len(),
            report.buckets,
            report.disagree_ge_floor,
            report.max_margin,
            verdict(pass)
        );
        for p in &report.disagree {
            println!(
                "  prompt {} step {}: ours {} pai {} (pai margin {:.4})",
                rows[p.id].n_tokens, p.step, p.ours, p.theirs, p.ref_margin
            );
        }
        Ok(pass)
    }

    // ------------------------------------------------------------ run

    pub fn run() -> Result<(), GateError> {
        let tier_now = Tier::from_env()?;
        // The sets open before the card loads: one the tier runs, missing or stale, ends the gate here.
        let oracles = open_oracles(tier_now)?;
        let path = seat::model_path(tier_now, "gate-gpu-solar2-e2e")?;
        let longest = oracles
            .iter()
            .map(|o| o.ids.len() + o.greedy.ids.len())
            .max()
            .unwrap_or(0);
        let t = std::time::Instant::now();
        let arm = seat::arm_arg()?;
        let mut seat = seat::open(&path, longest + CTX_SPARE, arm)?;
        let dims = seat.dims();
        println!(
            "load arm={} layers={} embd={} vocab={} ctx={} in {:.1} s (runtime value)",
            arm.word(),
            dims.layers,
            dims.embd,
            dims.vocab,
            longest + CTX_SPARE,
            t.elapsed().as_secs_f64()
        );
        let mut runs = Vec::with_capacity(oracles.len());
        for o in &oracles {
            if o.pai_logits.len() != dims.vocab {
                return Err(format!(
                    "{}: PAI's logits row is {} wide, the body's vocabulary {}",
                    o.name,
                    o.pai_logits.len(),
                    dims.vocab
                )
                .into());
            }
            runs.push(run_of(seat.as_mut(), &o.ids)?);
        }
        let case = Case {
            dims,
            oracles,
            runs,
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
        tier::expect_fixture_oracle(NAME, FIXTURE_ORACLE)?;
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME} the logits and layer outputs are finite with the fault word clean; a reset rewinds them bit for bit; \
             every layer's output stays within {TAP_EARLY_CEIL} (first {TAP_EARLY} layers) or {TAP_CEIL} of PAI's; in the real tier the shared top logprobs of each prompt stay within \
             two implementations' band of PAI's with the top-10 overlap held, and the teacher-forced greedy disagreements at a clear margin stay within the pin"
        );
        Ok(())
    }
}
