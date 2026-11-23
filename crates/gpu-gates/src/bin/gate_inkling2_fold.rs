//! Inkling 2's per-expert scale beside i-quant and K-quant stacks, loaded as Unsloth ships it
//! against the same file with the scale folded into the blocks' `d`: real tier only (a fixture
//! cannot carry the dynamic mixes' types). `--raw <file>` names the file as downloaded (a fused
//! gate·up stack of an i-quant type with its `<stem>.scale` tensor beside it), `--folded <file>`
//! the copy whose stacks hold the blocks times the scale (`d` and `dmin` rounded to f16) and
//! whose scale tensors are gone. The recipe passes both from `BLOOMERY_INK2_FOLD` (`raw,folded`).
//!
//! - `raw_serves_as_its_folded_copy`: both files open on the card (one after the other) and run
//!   the same seeded prompt ([`PROMPT`] ids) through the prompt pass, then [`STEPS`] greedy steps:
//!   the two id tables are equal, and the pass's last logits rows are within [`LOGITS_REL`] of each
//!   other by relative L2. The loaded bytes are the same bytes (the load's fold is the copy's
//!   fold), so a difference is the fold's: a scale applied to the wrong expert, to one half of a
//!   fused stack, a scalar left out.
//! - `pass_is_steps`: on each file, the prompt pass's last logits row against the row the same
//!   ids make one id a step (the card leg's `_sel` launches, no route block): within
//!   [`PASS_BAND`] by relative L2. The i-quant stacks' prompt GEMMs read 128-value activation
//!   rows where the step reads its own quantization of the same values, so a pass that wires the
//!   wrong form to a stack reads garbage the step's rows do not. Both files fold to the same
//!   bytes, so equal garbage would pass the first clause: this one holds the pass to the steps.
//!
//! Each file's development timing at depth [`DEV_DEPTH`] (the prompt pass's wall, then [`DEV_STEPS`]
//! graph steps after [`DEV_WARM`] warm ones) is printed after its clauses: development numbers, no
//! lease. A step that raises the fault word ends the gate by name.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_inkling2_fold", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::arch::inkling2::Body;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuModel};
    use bloomery_gpu_gates::rel::rel;
    use bloomery_gpu_gates::tier::{self, Tier};
    use bloomery_gpu_gates::{GateError, GateRng, checks_failed, expect_arch, verdict};
    use gguf::Split;
    use model::arch::Arch;
    use model::arch::inkling2::hparams::Hparams;
    use std::path::PathBuf;

    /// The gate's name: the refusals'.
    const NAME: &str = "gate_inkling2_fold";
    /// The recipe that runs it.
    const RECIPE: &str = "gate-inkling2-fold";

    /// Positions of the store: the prompt and the steps fit.
    const CTX: usize = 1024;
    /// The seeded prompt's length in ids.
    const PROMPT: usize = 96;
    /// Greedy steps after the prompt.
    const STEPS: usize = 64;
    /// The timing's prompt length, and its graph steps after the warm ones.
    const DEV_DEPTH: usize = 500;
    const DEV_WARM: usize = 4;
    const DEV_STEPS: usize = 128;
    /// The seed of the prompt's ids.
    const SEED: u64 = 0x1F0D_5EED_2B17_A4C3;
    /// The prompt's last logits rows, relative L2: the loaded bytes are equal, so only an
    /// order of float sums that varies between runs shows.
    const LOGITS_REL: f64 = 1.0e-5;

    /// The pass's last logits row against the steps' (relative L2), per file: the pass quantizes
    /// the routed rows onto grids the step's launches do not use, and a router tie can fall
    /// either way. A routed form wired wrong moves the row by order 1.
    /// PIN(2026-11-24): 1.5 times the first run's 7.63e-3 (UD-IQ3_XXS and its folded copy, the
    /// seeded prompt of 96 ids).
    const PASS_BAND: f64 = 1.2e-2;

    /// One file's run: the greedy ids after the prompt pass, the pass's last logits row and the
    /// row the same ids make one id a step, each cut to the unpadded vocabulary (the head's
    /// padding slots hold -1e30, which would be the whole norm of the row).
    struct Run {
        ids: Vec<u32>,
        pass_logits: Vec<f32>,
        step_logits: Vec<f32>,
        resident: usize,
    }

    fn args() -> Result<(PathBuf, PathBuf), GateError> {
        let mut raw = None;
        let mut folded = None;
        let mut it = std::env::args().skip(1);
        while let Some(a) = it.next() {
            let slot = match a.as_str() {
                "--raw" => &mut raw,
                "--folded" => &mut folded,
                _ => continue,
            };
            *slot = it.next().map(PathBuf::from);
        }
        match (raw, folded) {
            (Some(r), Some(f)) => Ok((r, f)),
            _ => Err(format!("{NAME}: --raw <file> --folded <file> name the two files").into()),
        }
    }

    /// Open `path` on the card, run the seeded prompt and the greedy steps, and release the card.
    fn run_file(path: &PathBuf) -> Result<Run, GateError> {
        let split = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        expect_arch(&split, Arch::Inkling2, RECIPE)?;
        let hp = Hparams::read(&split).map_err(|e| format!("{NAME}: the file's header: {e}"))?;
        let vocab = hp.n_vocab_unpadded;
        let mut rng = GateRng(SEED);
        let prompt: Vec<u32> = (0..PROMPT).map(|_| rng.below(vocab) as u32).collect();
        let t = std::time::Instant::now();
        let mut m = GpuModel::<Body>::open(Gpu::new()?, split, CTX)?;
        println!(
            "{NAME}: {} opened in {:.1} s, resident_bytes={}",
            path.display(),
            t.elapsed().as_secs_f64(),
            m.resident_bytes()
        );
        m.reset()?;
        let mut id = m.prompt_pass(&prompt)?;
        let pass_logits = m.logits()?[..vocab].to_vec();
        let mut ids = vec![id];
        m.set_mode(StepMode::Graph);
        for _ in 0..STEPS {
            id = m.step(&[id])?;
            ids.push(id);
        }
        m.set_mode(StepMode::Eager);
        m.reset()?;
        for &p in &prompt {
            m.step(&[p])?;
        }
        let step_logits = m.logits()?[..vocab].to_vec();
        let dev: Vec<u32> = (0..DEV_DEPTH).map(|_| rng.below(vocab) as u32).collect();
        m.reset()?;
        let t = std::time::Instant::now();
        let mut id = m.prompt_pass(&dev)?;
        let pass_s = t.elapsed().as_secs_f64();
        m.set_mode(StepMode::Graph);
        for _ in 0..DEV_WARM {
            id = m.step(&[id])?;
        }
        let t = std::time::Instant::now();
        for _ in 0..DEV_STEPS {
            id = m.step(&[id])?;
        }
        let step_s = t.elapsed().as_secs_f64();
        m.set_mode(StepMode::Eager);
        println!(
            "  dev {}: pass of {DEV_DEPTH} ids {:.1} ms ({:.0} tok/s), {DEV_STEPS} graph steps \
             {:.3} ms a step ({:.1} tok/s) at depth {DEV_DEPTH}, ctx {CTX} (development numbers: no lease)",
            path.display(),
            pass_s * 1e3,
            DEV_DEPTH as f64 / pass_s,
            step_s * 1e3 / DEV_STEPS as f64,
            DEV_STEPS as f64 / step_s
        );
        Ok(Run {
            ids,
            pass_logits,
            step_logits,
            resident: m.resident_bytes(),
        })
    }

    pub fn run() -> Result<(), GateError> {
        if Tier::from_env()? != Tier::Real {
            return Err(format!(
                "{NAME}: the dynamic mixes have no fixture; the gate runs in the real tier only"
            )
            .into());
        }
        let (raw, folded) = args()?;
        let a = run_file(&raw)?;
        let b = run_file(&folded)?;
        let lr = rel(&a.pass_logits, &b.pass_logits);
        let first_diff = a.ids.iter().zip(&b.ids).position(|(x, y)| x != y);
        let same = a.ids == b.ids && lr <= LOGITS_REL;
        println!(
            "raw_serves_as_its_folded_copy prompt={PROMPT} steps={STEPS} ids_equal={} \
             first_diff={first_diff:?} logits_rel={lr:.3e} band={LOGITS_REL:.1e} \
             resident raw={} folded={} {}",
            a.ids == b.ids,
            a.resident,
            b.resident,
            verdict(same)
        );
        let (pa, pb) = (
            rel(&a.pass_logits, &a.step_logits),
            rel(&b.pass_logits, &b.step_logits),
        );
        let steps_ok = pa <= PASS_BAND && pb <= PASS_BAND;
        println!(
            "pass_is_steps raw={pa:.3e} folded={pb:.3e} band={PASS_BAND:.1e} {}",
            verdict(steps_ok)
        );
        println!("{}", tier::tally_line());
        if !(same && steps_ok) {
            return Err(checks_failed());
        }
        println!(
            "PASSED: {NAME}: the file as downloaded and its scale-folded copy ran the same \
             {} greedy ids and the same prompt logits row, and each one's pass held to its steps",
            a.ids.len()
        );
        Ok(())
    }
}
