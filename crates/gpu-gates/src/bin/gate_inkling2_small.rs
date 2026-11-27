//! The Inkling 2 Small's host-tier gate: the real file (UD-Q2_K_XL, six shards,
//! `refset::arch::inkling2::small::MODEL`) on one card with every routed expert served by the host
//! tier (`GpuModel::<Body>::open_placed`), against the patched llama.cpp PR #32695 tree's CPU node
//! dumps of that file (refset family `lcpp-inkling2s`, set `d`). Real tier only: the Small's quant
//! mix (IQ2_XS, IQ3_XXS, IQ4_XS experts, q6_K, q8_0 and BF16 attention) has no fixture, so the
//! gate names the file itself and does not read `BLOOMERY_MODEL`.
//!
//! - `small_placement_is_the_hosts`: the plan on the gate card holds no routed expert on the card
//!   (the card leg's down arm reads no 128-value type, `accepts::down32_leg`: the Small's IQ3_XXS and IQ4_XS downs), the host's
//!   expert bytes are the file's routed bytes, and the loaded body hosts every layer with none of
//!   its experts on the card.
//! - `small_decode_is_vendors`: set `d`'s prompt one id a step, the logits row at its last position
//!   within [`LOGITS_BAND`] of the vendor's `result_output` by relative L2 (the Mini's gate's
//!   measure), then the vendor's greedy ids fed back one a step (teacher-forced), our argmax at
//!   each of the greedy table's positions against its id: the disagreements at a vendor margin of
//!   [`MARGIN_FLOOR`] or more are at most the pin the Gaussian error model gives at [`SIGMA`].
//!
//! Before the card loads, a set absent from the box, dumped from another file or build, or
//! unfinished ends the gate by name. A step that raises the fault word ends the gate by name.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_inkling2_small", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "gpu")]
mod gate {
    use crate::gate_card;
    use bloomery_gpu::GpuModel;
    use bloomery_gpu::arch::inkling2::Body;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::prompts::{GreedyRow, MARGIN_FLOOR, compare_forced, std_normal_cdf};
    use bloomery_gpu_gates::rel::rel;
    use bloomery_gpu_gates::tier::{self, Tag};
    use bloomery_gpu_gates::{
        GateError, RefManifest, checks_failed, expect_arch, ref_tensor_logical_in, verdict,
    };
    use bloomery_levers::{CARD_BUDGET, CARD_DONTNEED, HOST_LOCK, HOST_POPULATE, R8};
    use gguf::Split;
    use model::arch::Arch;
    use model::arch::inkling2::place::Inputs;
    use model::placement::{PlanLevers, Role};
    use refset::arch::inkling2::small;
    use refset::arch::prompt_tables::{self, Greedy, Top};

    /// The gate's name: the recipe's and the refusals'.
    const NAME: &str = "gate_inkling2_small";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-inkling2-small-host";

    /// Positions of every store: set `d`'s prompt and its greedy ids fit.
    const CTX: usize = 2048;

    /// The logits row's relative L2 from the vendor's at the prompt's last position.
    /// PIN(2026-11-23): 1.5 times the first run's 3.06e-2 (set `d`, 5 prompt ids, the Q2_K_XL
    /// mix; the Mini's real band at its Q4_K_XL mix is 1.2e-1).
    const LOGITS_BAND: f64 = 4.6e-2;

    /// One implementation's logit deviation from the other's, in logit units: the σ of the
    /// teacher-forced argmax pin.
    /// PIN(2026-11-23): 1.5 times the first run's σ̂ of 0.190 (set `d`'s greedy table, 31 steps).
    const SIGMA: f64 = 0.285;

    /// One set's prompt, its tables and the vendor's logits at the prompt's last position.
    struct Set {
        prompt: Vec<u32>,
        top: Vec<Top>,
        greedy: Greedy,
        logits: Vec<f32>,
    }

    /// The last row of `n` values of node `name`'s last occurrence (a ubatch an occurrence).
    fn last_row(man: &RefManifest, name: &str, n: usize) -> Result<Vec<f32>, GateError> {
        let mut occ = 0u32;
        while man.tensor(name, occ + 1).is_ok() {
            occ += 1;
        }
        let row = man.tensor(name, occ)?;
        let all = ref_tensor_logical_in(&man.dir, row)?;
        if all.len() < n || !all.len().is_multiple_of(n) {
            return Err(format!("{name} occurrence {occ} is {:?}, not rows of {n}", row.ne).into());
        }
        Ok(all[all.len() - n..].to_vec())
    }

    fn open_set(name: &'static str) -> Result<Set, GateError> {
        let man = RefManifest::open(&small::LCPP.path(name), &small::LCPP)?;
        let prompt: Vec<u32> = prompt_tables::tokens(&man)?.to_vec();
        let vocab = man.tensor("result_output", 0)?.ne[0] as usize;
        let set = Set {
            top: prompt_tables::topk(&man)?,
            greedy: prompt_tables::greedy(&man)?,
            logits: last_row(&man, "result_output", vocab)?,
            prompt,
        };
        if set.greedy.ids.is_empty() || set.greedy.margins.len() != set.greedy.ids.len() {
            return Err(format!(
                "{name}: a greedy table of {} ids and {} margins",
                set.greedy.ids.len(),
                set.greedy.margins.len()
            )
            .into());
        }
        let need = set.prompt.len() + set.greedy.ids.len();
        if need > CTX {
            return Err(format!("{name}: {need} positions, the gate's stores hold {CTX}").into());
        }
        Ok(set)
    }

    /// The expected disagreements at a vendor margin of [`MARGIN_FLOOR`] or more under the
    /// Gaussian model at `sigma`, and the pin: the expectation and three standard deviations.
    fn forced_pin(margins: &[f32], sigma: f64) -> (f64, usize) {
        let sigma_pair = sigma * std::f64::consts::SQRT_2;
        let expected: f64 = margins
            .iter()
            .filter(|&&m| m >= MARGIN_FLOOR)
            .map(|&m| std_normal_cdf(-f64::from(m) / sigma_pair))
            .sum();
        (expected, (expected + 3.0 * expected.sqrt()).ceil() as usize)
    }

    /// The RMS of the two logits rows' difference, in logit units.
    fn sigma_hat(ours: &[f32], theirs: &[f32]) -> f64 {
        let n = theirs.len().max(1) as f64;
        (ours
            .iter()
            .zip(theirs)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum::<f64>()
            / n)
            .sqrt()
    }

    /// Clause `small_placement_is_the_hosts`: the plan's card counts and host bytes, and the
    /// loaded body's hosted layers.
    fn placement(
        inputs: &Inputs,
        plan: &model::placement::Plan<'_>,
        m: Option<&GpuModel<Body>>,
    ) -> Result<bool, GateError> {
        let n_l: u64 = plan.n_l.iter().sum();
        let routed: u64 = inputs
            .model
            .tensors
            .iter()
            .filter(|t| t.role == Role::RoutedExperts)
            .map(|t| t.file_bytes)
            .sum();
        let host_bytes = plan.host.expert_bytes;
        let (hosted, on_card) = match m {
            None => (0, 0),
            Some(m) => {
                let p = m
                    .body(NAME)?
                    .placed()
                    .ok_or("the load has no placed side")?;
                (
                    (0..inputs.hp.n_layer).filter(|&l| p.hosted(l)).count(),
                    (0..inputs.hp.n_layer).map(|l| p.n_card(l)).sum::<usize>(),
                )
            }
        };
        let planned = n_l == 0 && host_bytes == routed;
        let pass = planned && (m.is_none() || (hosted == inputs.hp.n_layer && on_card == 0));
        println!(
            "small_placement_is_the_hosts layers={} n_l_sum={n_l} host_expert_bytes={host_bytes} \
             routed_bytes={routed} loaded={} hosted_layers={hosted} card_experts={on_card} {}",
            inputs.hp.n_layer,
            m.is_some(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause `small_decode_is_vendors` on set `s`.
    fn decode(m: &mut GpuModel<Body>, s: &Set) -> Result<bool, GateError> {
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let t = std::time::Instant::now();
        let first = m.step(&s.prompt)?;
        let prompt_s = t.elapsed().as_secs_f64();
        let logits = m.logits()?;
        let ours = logits
            .get(..s.logits.len())
            .ok_or("the logits row is shorter than the vendor's")?;
        let lr = rel(ours, &s.logits);
        let sigma_hat = sigma_hat(ours, &s.logits);
        let logits_ok = lr <= LOGITS_BAND;
        println!(
            "small_decode_is_vendors ids={} logits_rel={lr:.3e} band={LOGITS_BAND:.1e} \
             sigma_hat={sigma_hat:.3} prompt_s={prompt_s:.1} {}",
            s.prompt.len(),
            verdict(logits_ok)
        );
        let mut table = vec![first];
        let before = m
            .body(NAME)?
            .placed()
            .ok_or("no placed side")?
            .hybrid()
            .stats();
        let t = std::time::Instant::now();
        for &id in &s.greedy.ids[..s.greedy.ids.len() - 1] {
            table.push(m.step(&[id])?);
        }
        let steps = s.greedy.ids.len() - 1;
        let step_s = t.elapsed().as_secs_f64();
        let after = m
            .body(NAME)?
            .placed()
            .ok_or("no placed side")?
            .hybrid()
            .stats();
        // The step's split: the host leg's wall (go seen to signal) against the step's.
        let (served, leg_s) = (
            after.served - before.served,
            (after.leg_ns - before.leg_ns) as f64 * 1e-9,
        );
        println!(
            "small_decode_is_vendors split step_ms={:.1} host_leg_ms={:.1} served_per_step={} \
             leg_ms_per_service={:.3} go_early={}/{} straggle_ms_per_step={:.2}",
            step_s * 1e3 / steps as f64,
            leg_s * 1e3 / steps as f64,
            served / steps as u64,
            leg_s * 1e3 / served.max(1) as f64,
            after.go_early - before.go_early,
            served,
            (after.straggle_ns - before.straggle_ns) as f64 * 1e-6 / steps as f64,
        );
        let vendor = GreedyRow {
            id: 0,
            n_tokens: s.prompt.len(),
            argmax: s.top.first().map_or(u32::MAX, |t| t.id),
            top5: s.top.iter().take(5).map(|t| t.id).collect(),
            gen_ids: s.greedy.ids.clone(),
            gen_margins: s.greedy.margins.clone(),
        };
        let report = compare_forced(&[table], &[vendor], MARGIN_FLOOR);
        let (expected, pin) = forced_pin(&s.greedy.margins, SIGMA);
        let forced_ok = report.disagree_ge_floor <= pin;
        println!(
            "small_decode_is_vendors forced positions={} disagree={} buckets={:?} ge_floor={} \
             expected {expected:.2} pin {pin} steps={steps} step_s={step_s:.2} {}",
            report.positions,
            report.disagree.len(),
            report.buckets,
            report.disagree_ge_floor,
            verdict(forced_ok)
        );
        m.set_mode(StepMode::Eager);
        Ok(logits_ok && forced_ok)
    }

    pub fn run() -> Result<(), GateError> {
        if !tier::run_clause("small_placement_is_the_hosts", Tag::FileBound)? {
            return Ok(());
        }
        let levers =
            bloomery_levers::at_main(&[CARD_BUDGET, HOST_POPULATE, HOST_LOCK, CARD_DONTNEED, R8])?;
        let split = Split::open(small::MODEL).map_err(|e| format!("open {}: {e}", small::MODEL))?;
        expect_arch(&split, Arch::Inkling2, RECIPE)?;
        let inputs = Inputs::read(&split).map_err(|e| format!("{NAME}: the file: {e}"))?;
        let set = open_set(small::D)?;
        gate_card::init()?;
        let machine = gate_card::plan_gate(inputs.model.layers);
        let plan = inputs
            .plan(
                &machine,
                CTX as u64,
                &PlanLevers::default(),
                bloomery_gpu::NATIVE_FP4,
            )
            .map_err(|e| format!("{NAME}: the plan: {e}"))?;
        let mut ok = placement(&inputs, &plan, None)?;
        if !ok {
            return Err(checks_failed());
        }
        let t = std::time::Instant::now();
        let mut m = match GpuModel::<Body>::open_placed(split, &plan, &inputs, 0, levers.host()) {
            Ok(m) => m,
            Err(e) => {
                println!("opens {e} {}", verdict(false));
                return Err(e.into());
            }
        };
        println!(
            "{NAME}: device {}, resident_bytes={} ctx={CTX}, opened in {:.1} s",
            m.body_parts(NAME)?.0.device_name()?,
            m.resident_bytes(),
            t.elapsed().as_secs_f64()
        );
        ok &= placement(&inputs, &plan, Some(&m))?;
        if tier::run_clause("small_decode_is_vendors", Tag::Oracle)? {
            ok &= decode(&mut m, &set)?;
        }
        println!("{}", tier::tally_line());
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME}: every routed expert on the host and none on the card; set d's logits \
             row within its band and the teacher-forced argmax disagreements within the pin"
        );
        Ok(())
    }
}
