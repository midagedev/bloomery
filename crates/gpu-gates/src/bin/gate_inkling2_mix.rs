//! Inkling 2 Mini with a share of every layer's routed experts on the host: the real tier's
//! UD-IQ3_XXS file (an i-quant mix the card leg runs, 11.7 GB) planned under a card budget too
//! small to hold it whole (`BLOOMERY_CARD_BUDGET`, 12G: the 3060's), so the plan keeps about four
//! fifths of each layer's experts on the card (the id prefix `[0, n_l)`, the same `n_l` on every
//! layer give or take one) and the host tier serves the rest. `--file <path>` names the file; the
//! recipe passes the raw file of `BLOOMERY_INK2_FOLD`.
//!
//! - `mix_placement_splits_the_experts`: the plan holds experts on the card and on the host
//!   (a card count above 0 and under every expert), the layers' `n_l` differ by at most one, and
//!   the loaded body hosts every layer with `n_card` equal to the plan's `n_l`.
//! - `mix_forced_{card,host,any}_picks_are_the_whole_card`: the same file on the card whole
//!   (`GpuModel::open`, nothing on the host) and mixed (`GpuModel::open_placed`), both fed the
//!   seeded prompt one id an eager step with every visit's picks forced to one seeded table
//!   (`Body::force_picks`: eight distinct ids and weights summing to 1 a visit, the same at every
//!   step), and each prompt position's logits row compared by relative L2. The card table's ids
//!   all sit below every layer's `n_l`: nothing is asked of the host, and the rows are the same
//!   bits ([`CARD_REL`]), which holds the placed walk (embedding, visits, pass edges, the card's
//!   sum and the host row added) to the whole card's. The host table's ids all sit at or above
//!   every `n_l` and the any table's anywhere: the rows are within [`HOST_REL`], the host dots'
//!   rounding. The router's own picks are not held to a row band: a perturbation of rounding size
//!   flips a near-tied pick somewhere in the 48 visits of a position, and the flip sets the row's
//!   distance (1e-2 to 8e-2 at every host share), so the natural run is held by its argmax below.
//! - `mix_is_the_whole_card`: the router's own picks. The whole card's greedy ids fed to the mixed
//!   load one a step (teacher-forced): the mixed load's argmax disagreements at the whole card's
//!   margin of [`MARGIN_FLOOR`] or more are at most the pin the Gaussian error model gives at the
//!   measured σ̂ of the two arms' logit rows, and σ̂ is under [`SIGMA_CEIL`]. The rows' relative
//!   L2 is printed.
//! - `mix_pass_attached`: the mixed load holds the prompt pass's buffers (`Body::pass_rows` is
//!   `place::pass_rows` at the context) and they take no more than the plan's reserve for them
//!   (`Inputs::pass_reserve`), so the prompts below run as passes, not steps.
//! - `mix_pass_forced_{card,host,any}_picks_are_the_whole_card_pass`: the seeded prompt as one call
//!   of the prompt pass (`GpuModel::prompt_pass_scored`, every position kept) on the whole card and
//!   on the mixed load, every visit's picks forced to the step clauses' tables (`Body::force_pass_picks`:
//!   a visit's picks repeated at every position), each position's logits row compared by relative L2
//!   (the card table in its step band, the host and any tables within [`PASS_HOST_REL`]); and the host tier's batch service computes exactly the table's host
//!   slots (none for the card table, every slot of every position of every visit for the host table,
//!   at most that for the any table).
//! - `mix_pass_forced_{card,host,any}_picks_are_the_placed_steps`: the mixed load's pass against its
//!   own stepped prompt under the same tables, each position's row within [`PASS_STEPS_REL`].
//!
//! The two arms hold the same weights and run the same card launches on the card's slots; the
//! mixed load's host share is dotted by the host tier's kernels, which quantize the activation on
//! their own grid and sum in their own order, so the arms differ by that share's rounding and
//! nothing else: a card sum left out, a shadow not run, a visit of the looped trunk whose
//! pass edge or host service is wrong moves the row by order 1. Each arm's development timing
//! (the steps' wall after [`DEV_WARM`] warm ones, and a prompt pass of [`DEV_P`] ids on the two
//! loads that run one) is printed: development numbers, no lease. A step that raises the fault word
//! ends the gate by name.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_inkling2_mix", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "gpu")]
mod gate {
    use crate::gate_card;
    use bloomery_gpu::arch::inkling2::Body;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuModel};
    use bloomery_gpu_gates::prompts::{MARGIN_FLOOR, std_normal_cdf};
    use bloomery_gpu_gates::rel::rel;
    use bloomery_gpu_gates::tier::{self, Tier};
    use bloomery_gpu_gates::{GateError, GateRng, checks_failed, expect_arch, verdict};
    use bloomery_levers::{CARD_BUDGET, CARD_DONTNEED, HOST_LOCK, HOST_POPULATE, R8};
    use gguf::Split;
    use model::arch::Arch;
    use model::arch::inkling2::place::{self, Inputs};
    use model::placement::{Plan, PlanLevers};
    use std::path::PathBuf;

    /// The gate's name: the refusals'.
    const NAME: &str = "gate_inkling2_mix";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-inkling2-mix";

    /// Positions of every store: the prompt and the steps fit.
    const CTX: usize = 1024;
    /// The seeded prompt's length in ids.
    const PROMPT: usize = 48;
    /// Greedy steps after the prompt.
    const STEPS: usize = 32;
    /// The steps the timing reads, after this many warm ones.
    const DEV_WARM: usize = 4;
    /// The seed of the prompt's ids.
    const SEED: u64 = 0x4D49_5845_2B17_A4C3;

    /// The band of the forced table whose slots all sit on the card: the placed walk adds the
    /// host's row (zero: no slot was asked of it) to the card's sum, so the arms are the same
    /// bits. PIN(2026-11-24): the first runs' 0 at every one of 48 positions (12G and 6G
    /// budgets), a millionth for a sum a launch orders differently.
    const CARD_REL: f64 = 1.0e-6;

    /// The band of the forced tables whose slots sit on the host or anywhere: the largest over the
    /// prompt's positions of the logits row's relative L2 between the arms. The host dots a
    /// slot's activation on its own grid, and a visit with any slot hosted moves the row to a
    /// floor that more hosted slots do not raise (9 % to 100 % of the slots, 4e-3 to 8e-3 at
    /// every position), of the order of `gate_inkling2_fold`'s pass-versus-steps row (every slot
    /// on another grid, 7.6e-3).
    /// PIN(2026-11-24): 1.5 times the first run's 8.1e-3 (the host-only table, the 12G budget).
    const HOST_REL: f64 = 1.2e-2;

    /// The ceiling of σ̂, the RMS of the two arms' logit rows' difference over the natural run's
    /// steps, in logit units: the router's own picks flip near-tied choices, so the rows differ at
    /// the same scale whatever the host share (0.23 to 0.35 at 1 % to 98 % of the slots on the
    /// host), and a visit whose sum or edge is wrong moves them by order 4.
    /// PIN(2026-11-24): 1.5 times the first run's 0.277 (the 12G budget).
    const SIGMA_CEIL: f64 = 0.42;

    /// The band of the mixed load's prompt pass against its own stepped prompt under one forced
    /// table, the largest over the prompt's positions of the logits row's relative L2: the pass
    /// reads its projections and routed rows on wider grids than the step's gemvs and leg
    /// (`gate_inkling2_fold`'s pass-versus-steps row, every slot on another grid, 7.6e-3), and the
    /// host's slots add the host dots' own grid, so the two stack to the order of 1e-2.
    /// PIN(2026-11-24): 1.5 times the first run's 1.31e-2 (the host-only table, the 12G budget;
    /// the card-only table 1.07e-2, the any table 9.7e-3).
    const PASS_STEPS_REL: f64 = 2.0e-2;

    /// The band of the mixed load's prompt pass against the whole card's pass under a forced table
    /// with slots on the host or anywhere: both arms read the pass's grids, so the host dots' own
    /// grid is the difference, as in [`HOST_REL`], at the pass's floor (the any table's rows sit
    /// at that floor with 8 % of the slots hosted, as the host-only table's with all of them).
    /// PIN(2026-11-24): 1.5 times the first run's 1.24e-2 (the any table, the 12G budget; the
    /// host-only table 8.9e-3).
    const PASS_HOST_REL: f64 = 1.9e-2;

    /// The prompt length of the passes' development timing.
    const DEV_P: usize = 512;

    /// Experts a visit's pick list holds (the router's top count).
    const N_USED: usize = 8;

    /// Where a forced table's picks fall.
    #[derive(Clone, Copy)]
    enum Kind {
        /// Ids below every layer's `n_l`: every slot on the card, nothing for the host to serve.
        Card,
        /// Ids at or above every layer's `n_l`: every slot on the host.
        Host,
        /// Any id.
        Any,
    }

    impl Kind {
        /// The table's name in the pass clauses.
        fn tag(self) -> &'static str {
            match self {
                Kind::Card => "card",
                Kind::Host => "host",
                Kind::Any => "any",
            }
        }
    }

    /// The forced tables the gate runs, by clause name.
    const TABLE_KINDS: [(&str, Kind, f64); 3] = [
        (
            "mix_forced_card_picks_are_the_whole_card",
            Kind::Card,
            CARD_REL,
        ),
        (
            "mix_forced_host_picks_are_the_whole_card",
            Kind::Host,
            HOST_REL,
        ),
        (
            "mix_forced_any_picks_are_the_whole_card",
            Kind::Any,
            HOST_REL,
        ),
    ];

    /// One arm's run: the argmax of each position (the prompt's last, then the steps') and the
    /// logits row after each, cut to the unpadded vocabulary, and the step wall.
    struct Trace {
        ids: Vec<u32>,
        rows: Vec<Vec<f32>>,
        step_ms: f64,
    }

    fn file_arg() -> Result<PathBuf, GateError> {
        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() {
            if a == "--file"
                && let Some(p) = it.next()
            {
                return Ok(PathBuf::from(p));
            }
        }
        Err(format!("{NAME}: --file <the Mini's UD-IQ3_XXS file> names the file").into())
    }

    /// The prompt fed one id a step on a model in graph mode, then `STEPS` steps: each fed its own
    /// argmax, or the `forced` ids (teacher-forced). Positions of the trace: the prompt's last id
    /// and each fed id.
    fn walk(
        m: &mut GpuModel<Body>,
        prompt: &[u32],
        forced: Option<&[u32]>,
        vocab: usize,
    ) -> Result<Trace, GateError> {
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let mut id = 0;
        for &p in prompt {
            id = m.step(&[p])?;
        }
        let mut ids = vec![id];
        let mut rows = vec![m.logits()?[..vocab].to_vec()];
        let mut wall = 0.0;
        for i in 0..STEPS {
            let feed = forced.map_or(id, |f| f[i]);
            let t = std::time::Instant::now();
            id = m.step(&[feed])?;
            if i >= DEV_WARM {
                wall += t.elapsed().as_secs_f64();
            }
            ids.push(id);
            rows.push(m.logits()?[..vocab].to_vec());
        }
        m.set_mode(StepMode::Eager);
        Ok(Trace {
            ids,
            rows,
            step_ms: wall * 1e3 / (STEPS - DEV_WARM) as f64,
        })
    }

    /// One seeded pick table for `visits` visits: per visit `N_USED` distinct expert ids from
    /// `ids_in` and positive weights summing to 1.
    fn forced_table(
        rng: &mut GateRng,
        visits: usize,
        ids_in: std::ops::Range<usize>,
    ) -> Result<(Vec<u32>, Vec<f32>), GateError> {
        if ids_in.len() < N_USED {
            return Err(format!(
                "{NAME}: the ids {ids_in:?} are fewer than the {N_USED} a visit picks: the plan's \
                 split leaves too few experts on one side for a forced table"
            )
            .into());
        }
        let (mut ids, mut weights) = (Vec::new(), Vec::new());
        for _ in 0..visits {
            let mut taken: Vec<u32> = Vec::with_capacity(N_USED);
            while taken.len() < N_USED {
                let id = (ids_in.start + rng.below(ids_in.len())) as u32;
                if !taken.contains(&id) {
                    taken.push(id);
                }
            }
            let raw: Vec<f32> = (0..N_USED).map(|_| rng.uni(0.5, 1.5)).collect();
            let sum: f32 = raw.iter().sum();
            ids.extend(taken);
            weights.extend(raw.iter().map(|w| w / sum));
        }
        Ok((ids, weights))
    }

    /// The prompt fed one id an eager step with every visit's picks forced to `table`: the logits
    /// row after each id, cut to the unpadded vocabulary.
    fn walk_forced(
        m: &mut GpuModel<Body>,
        prompt: &[u32],
        table: &(Vec<u32>, Vec<f32>),
        vocab: usize,
    ) -> Result<Vec<Vec<f32>>, GateError> {
        m.set_mode(StepMode::Eager);
        m.reset()?;
        {
            let (gpu, _, body) = m.body_parts(NAME)?;
            body.force_picks(gpu, (&table.0, &table.1))?;
        }
        let mut rows = Vec::with_capacity(prompt.len());
        for &p in prompt {
            m.step(&[p])?;
            rows.push(m.logits()?[..vocab].to_vec());
        }
        m.body_parts(NAME)?.2.unforce_picks();
        Ok(rows)
    }

    /// `per_visit`'s pick lists (`N_USED` entries a visit) each repeated at `rows` positions: the
    /// pass's forced picks, visit-major, then position-major.
    fn repeated<T: Copy>(per_visit: &[T], rows: usize) -> Vec<T> {
        per_visit
            .chunks(N_USED)
            .flat_map(|c| (0..rows).flat_map(move |_| c.iter().copied()))
            .collect()
    }

    /// The prompt as one call of the prompt pass with every visit's picks forced to `table` (a
    /// visit's picks at every position): the logits row after each position, cut to the unpadded
    /// vocabulary.
    fn pass_forced(
        m: &mut GpuModel<Body>,
        prompt: &[u32],
        table: &(Vec<u32>, Vec<f32>),
        vocab: usize,
    ) -> Result<Vec<Vec<f32>>, GateError> {
        m.reset()?;
        {
            let (gpu, _, body) = m.body_parts(NAME)?;
            let ids = repeated(&table.0, prompt.len());
            let weights = repeated(&table.1, prompt.len());
            body.force_pass_picks(gpu, (&ids, &weights), prompt.len())?;
        }
        let kept: Vec<usize> = (0..prompt.len()).collect();
        let called = m.prompt_pass_scored(prompt, &kept);
        m.body_parts(NAME)?.2.unforce_pass_picks();
        let (_, rows) = called?;
        rows.into_iter()
            .map(|r| {
                r.get(..vocab).map(<[f32]>::to_vec).ok_or_else(|| {
                    GateError::from(format!(
                        "{NAME}: a kept row of {} values under a vocabulary of {vocab}",
                        r.len()
                    ))
                })
            })
            .collect()
    }

    /// The wall in ms of the router's own prompt pass over `ids` after a warm call of an eighth of
    /// them.
    fn dev_pass_ms(m: &mut GpuModel<Body>, ids: &[u32]) -> Result<f64, GateError> {
        m.reset()?;
        m.prompt_pass(&ids[..ids.len() / 8])?;
        m.reset()?;
        let t = std::time::Instant::now();
        m.prompt_pass(ids)?;
        Ok(t.elapsed().as_secs_f64() * 1e3)
    }

    /// The host tier's batch service so far: layer-batches served and host slots computed.
    fn host_batch(m: &GpuModel<Body>) -> Result<(u64, u64), GateError> {
        let s = m
            .body(NAME)?
            .placed()
            .ok_or("the load has no placed side")?
            .hybrid()
            .stats();
        Ok((s.batch_served, s.batch_host_slots))
    }

    /// Each position's relative L2 between two prompts' logits rows.
    fn rows_rel(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| rel(x, y)).collect()
    }

    /// `pos` as the clauses print it: the largest, then each position's.
    fn summarize(pos: &[f64]) -> (f64, String) {
        let max = pos.iter().copied().fold(0.0f64, f64::max);
        let each = pos
            .iter()
            .map(|r| format!("{r:.1e}"))
            .collect::<Vec<_>>()
            .join(" ");
        (max, each)
    }

    /// The top-two gap of a logits row.
    fn margin(row: &[f32]) -> f32 {
        let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &v in row {
            if v > a {
                b = a;
                a = v;
            } else if v > b {
                b = v;
            }
        }
        a - b
    }

    /// The expected disagreements at a whole-card margin of [`MARGIN_FLOOR`] or more under the
    /// Gaussian model at `sigma` (one arm's logit deviation from the other's), and the pin: the
    /// expectation and three standard deviations.
    fn forced_pin(margins: &[f32], sigma: f64) -> (f64, usize) {
        let sigma_pair = sigma * std::f64::consts::SQRT_2;
        let expected: f64 = margins
            .iter()
            .filter(|&&m| m >= MARGIN_FLOOR)
            .map(|&m| std_normal_cdf(-f64::from(m) / sigma_pair))
            .sum();
        (expected, (expected + 3.0 * expected.sqrt()).ceil() as usize)
    }

    /// The RMS of two traces' rows' difference, in logit units.
    fn sigma_hat(a: &Trace, b: &Trace) -> f64 {
        let (mut sum, mut n) = (0.0f64, 0usize);
        for (ra, rb) in a.rows.iter().zip(&b.rows) {
            for (x, y) in ra.iter().zip(rb) {
                sum += (f64::from(*x) - f64::from(*y)).powi(2);
                n += 1;
            }
        }
        (sum / n.max(1) as f64).sqrt()
    }

    /// Clause `mix_placement_splits_the_experts` on the plan, and on the loaded body when there is
    /// one.
    fn placement(
        inputs: &Inputs,
        plan: &Plan<'_>,
        m: Option<&GpuModel<Body>>,
    ) -> Result<bool, GateError> {
        let (n_layer, n_expert) = (inputs.hp.n_layer, inputs.hp.n_expert as u64);
        let on_card: u64 = plan.n_l.iter().sum();
        let (lo, hi) = (
            plan.n_l.iter().copied().min().unwrap_or(0),
            plan.n_l.iter().copied().max().unwrap_or(0),
        );
        let split = on_card > 0 && on_card < n_layer as u64 * n_expert && hi < n_expert;
        let even = hi - lo <= 1;
        let loaded = match m {
            None => true,
            Some(m) => {
                let p = m
                    .body(NAME)?
                    .placed()
                    .ok_or("the load has no placed side")?;
                (0..n_layer).all(|l| p.hosted(l) && p.n_card(l) as u64 == plan.n_l[l])
            }
        };
        let pass = split && even && loaded;
        println!(
            "mix_placement_splits_the_experts layers={n_layer} n_l={lo}..{hi} of {n_expert} \
             card_experts={on_card} card_expert_bytes={} host_expert_bytes={} loaded={} {}",
            plan.cards[0].expert_bytes,
            plan.host.expert_bytes,
            m.is_some(),
            verdict(pass)
        );
        Ok(pass)
    }

    pub fn run() -> Result<(), GateError> {
        if Tier::from_env()? != Tier::Real {
            return Err(format!(
                "{NAME}: the dynamic mixes have no fixture; the gate runs in the real tier only"
            )
            .into());
        }
        let path = file_arg()?;
        let levers =
            bloomery_levers::at_main(&[CARD_BUDGET, HOST_POPULATE, HOST_LOCK, CARD_DONTNEED, R8])?;
        let plan_levers = PlanLevers::from_levers(&levers)?;
        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        expect_arch(&split, Arch::Inkling2, RECIPE)?;
        let inputs = Inputs::read(&split).map_err(|e| format!("{NAME}: the file: {e}"))?;
        let vocab = inputs.hp.n_vocab_unpadded;
        let mut rng = GateRng(SEED);
        let prompt: Vec<u32> = (0..PROMPT).map(|_| rng.below(vocab) as u32).collect();
        gate_card::init()?;
        let machine = gate_card::plan_gate(inputs.model.layers);
        let plan = inputs
            .plan(&machine, CTX as u64, &plan_levers)
            .map_err(|e| format!("{NAME}: the plan: {e}"))?;
        let mut ok = placement(&inputs, &plan, None)?;
        if !ok {
            return Err(checks_failed());
        }
        let (lo, hi) = (
            plan.n_l.iter().copied().min().unwrap_or(0),
            plan.n_l.iter().copied().max().unwrap_or(0),
        );
        // The whole card first: the file as `GpuModel::open` holds it.
        let (whole, whole_forced, whole_passes, tables, dev) = {
            let whole_split =
                Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
            let t = std::time::Instant::now();
            let mut m = GpuModel::<Body>::open(Gpu::new()?, whole_split, CTX)?;
            println!(
                "{NAME}: whole card opened in {:.1} s, resident_bytes={}",
                t.elapsed().as_secs_f64(),
                m.resident_bytes()
            );
            let visits = m.body(NAME)?.visit_table().len();
            let tables = TABLE_KINDS.map(|(_, kind, _)| {
                let ids_in = match kind {
                    Kind::Card => 0..lo as usize,
                    Kind::Host => hi as usize..inputs.hp.n_expert,
                    Kind::Any => 0..inputs.hp.n_expert,
                };
                forced_table(&mut rng, visits, ids_in)
            });
            let tables = tables.into_iter().collect::<Result<Vec<_>, _>>()?;
            let natural = walk(&mut m, &prompt, None, vocab)?;
            let forced = tables
                .iter()
                .map(|t| walk_forced(&mut m, &prompt, t, vocab))
                .collect::<Result<Vec<_>, _>>()?;
            let passes = tables
                .iter()
                .map(|t| pass_forced(&mut m, &prompt, t, vocab))
                .collect::<Result<Vec<_>, _>>()?;
            let dev_ids: Vec<u32> = (0..DEV_P).map(|_| rng.below(vocab) as u32).collect();
            let dev_ms = dev_pass_ms(&mut m, &dev_ids)?;
            (natural, forced, passes, tables, (dev_ids, dev_ms))
        };
        let t = std::time::Instant::now();
        let mut m = match GpuModel::<Body>::open_placed(split, &plan, &inputs, 0, levers.host()) {
            Ok(m) => m,
            Err(e) => {
                println!("opens {e} {}", verdict(false));
                return Err(e.into());
            }
        };
        println!(
            "{NAME}: mixed opened in {:.1} s, resident_bytes={} ctx={CTX}",
            t.elapsed().as_secs_f64(),
            m.resident_bytes()
        );
        ok &= placement(&inputs, &plan, Some(&m))?;
        let attached = {
            let body = m.body(NAME)?;
            let (rows, made) = (body.pass_rows(), body.pass_made_bytes() as u64);
            let want = place::pass_rows(CTX);
            let reserve = inputs
                .pass_reserve(CTX as u64)
                .map_err(|e| format!("{NAME}: the pass reserve: {e}"))?;
            let pass = rows == want && made <= reserve;
            println!(
                "mix_pass_attached rows={rows} want={want} pass_bytes={made} reserve={reserve} {}",
                verdict(pass)
            );
            pass
        };
        ok &= attached;
        let before = m
            .body(NAME)?
            .placed()
            .ok_or("no placed side")?
            .hybrid()
            .stats();
        let mixed = walk(&mut m, &prompt, Some(&whole.ids[..STEPS]), vocab)?;
        let after = m
            .body(NAME)?
            .placed()
            .ok_or("no placed side")?
            .hybrid()
            .stats();
        let steps = (PROMPT + STEPS) as u64;
        println!(
            "mix_is_the_whole_card split per step: served={} host_slots={} leg_ms={:.2} \
             go_early={}/{} straggle_ms={:.2}",
            (after.served - before.served) / steps,
            (after.host_slots - before.host_slots) / steps,
            (after.leg_ns - before.leg_ns) as f64 * 1e-6 / steps as f64,
            after.go_early - before.go_early,
            after.served - before.served,
            (after.straggle_ns - before.straggle_ns) as f64 * 1e-6 / steps as f64,
        );
        let mut step_rows = Vec::with_capacity(TABLE_KINDS.len());
        for ((name, _, band), (table, whole_rows)) in
            TABLE_KINDS.iter().zip(tables.iter().zip(&whole_forced))
        {
            let mixed_rows = walk_forced(&mut m, &prompt, table, vocab)?;
            let pos: Vec<f64> = whole_rows
                .iter()
                .zip(&mixed_rows)
                .map(|(a, b)| rel(a, b))
                .collect();
            let max = pos.iter().copied().fold(0.0f64, f64::max);
            let pass = pos.iter().all(|r| r.is_finite()) && max <= *band;
            println!(
                "{name} prompt={PROMPT} max_rel={max:.3e} band={band:.1e} per-position: {} {}",
                pos.iter()
                    .map(|r| format!("{r:.1e}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                verdict(pass)
            );
            ok &= pass;
            step_rows.push(mixed_rows);
        }
        let lr = rel(&whole.rows[0], &mixed.rows[0]);
        let sigma = sigma_hat(&whole, &mixed);
        let margins: Vec<f32> = whole.rows.iter().map(|r| margin(r)).collect();
        let positions = whole.ids.len();
        let disagree: Vec<usize> = (0..positions)
            .filter(|&p| whole.ids[p] != mixed.ids[p])
            .collect();
        let ge_floor = disagree
            .iter()
            .filter(|&&p| margins[p] >= MARGIN_FLOOR)
            .count();
        let (expected, pin) = forced_pin(&margins, sigma);
        let same = ge_floor <= pin && sigma <= SIGMA_CEIL;
        println!(
            "mix_is_the_whole_card prompt={PROMPT} steps={STEPS} logits_rel={lr:.3e} \
             sigma_hat={sigma:.4} ceil={SIGMA_CEIL} positions={positions} disagree={} \
             ge_floor={ge_floor} expected {expected:.2} pin {pin} {}",
            disagree.len(),
            verdict(same)
        );
        ok &= same;
        if attached {
            let visits = m.body(NAME)?.visit_table().len();
            let slots_all = (visits * PROMPT * N_USED) as u64;
            for (i, (_, kind, band)) in TABLE_KINDS.iter().enumerate() {
                let tag = kind.tag();
                let (served0, slots0) = host_batch(&m)?;
                let rows = pass_forced(&mut m, &prompt, &tables[i], vocab)?;
                let (served1, slots1) = host_batch(&m)?;
                let (served, slots) = (served1 - served0, slots1 - slots0);
                let slots_ok = match kind {
                    Kind::Card => slots == 0,
                    Kind::Host => slots == slots_all,
                    Kind::Any => slots <= slots_all,
                };
                let to_whole = rows_rel(&whole_passes[i], &rows);
                let (max, each) = summarize(&to_whole);
                let band = match kind {
                    Kind::Card => *band,
                    Kind::Host | Kind::Any => PASS_HOST_REL,
                };
                let pass = to_whole.iter().all(|r| r.is_finite()) && max <= band && slots_ok;
                println!(
                    "mix_pass_forced_{tag}_picks_are_the_whole_card_pass prompt={PROMPT} \
                     max_rel={max:.3e} band={band:.1e} host_slots={slots} of {slots_all} \
                     served={served} per-position: {each} {}",
                    verdict(pass)
                );
                ok &= pass;
                let to_steps = rows_rel(&step_rows[i], &rows);
                let (max, each) = summarize(&to_steps);
                let pass = to_steps.iter().all(|r| r.is_finite()) && max <= PASS_STEPS_REL;
                println!(
                    "mix_pass_forced_{tag}_picks_are_the_placed_steps prompt={PROMPT} \
                     max_rel={max:.3e} band={PASS_STEPS_REL:.1e} per-position: {each} {}",
                    verdict(pass)
                );
                ok &= pass;
            }
        }
        println!(
            "  dev whole card: {:.3} ms a step ({:.1} tok/s) after {DEV_WARM} warm ones, depth \
             ~{PROMPT}, ctx {CTX}; mixed: {:.3} ms a step ({:.1} tok/s) (development numbers: no \
             lease)",
            whole.step_ms,
            1e3 / whole.step_ms,
            mixed.step_ms,
            1e3 / mixed.step_ms
        );
        if attached {
            let (dev_ids, whole_ms) = dev;
            let placed_ms = dev_pass_ms(&mut m, &dev_ids)?;
            let tok_s = |ms: f64| DEV_P as f64 * 1e3 / ms;
            println!(
                "  dev prompt pass P={DEV_P} (the router's picks): mixed {:.1} tok/s ({placed_ms:.1} \
                 ms), whole card {:.1} tok/s ({whole_ms:.1} ms); the mixed load's stepped prompt \
                 runs at its step rate, {:.1} tok/s (development numbers: no lease)",
                tok_s(placed_ms),
                tok_s(whole_ms),
                1e3 / mixed.step_ms
            );
        }
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME}: the plan split every layer's experts between the card and the host, \
             the mixed load's prompt logits row and teacher-forced argmax held to the whole \
             card's, and its prompt pass to the whole card's pass and to its own steps"
        );
        Ok(())
    }
}
