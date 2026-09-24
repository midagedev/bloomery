//! GPU gate for long greedy runs of the V4.1 engine on the gate placement
//! (`workstation::plan_gate`, the 3090), the model opened through the
//! engine's own entry (`body::open`). Every position the run steps goes
//! first through the finite probe (`shared/ds41_finite.rs`): the position's
//! step eagerly, outside the graph, each sub-layer's streams read where it
//! wrote them; then the position is taken back and the token stepped through
//! the engine's captured graph, whose argmax is the run's token.
//!
//! Two arms on one load, each from a reset:
//! - `--free`: prompt row 0 of `$BLOOMERY_DATA/greedy-ds41` ("The capital
//!   of France is", the oracle's five ids), then `-n` greedy tokens (330).
//! - `--trigger`: the same prompt and [`TRIGGER`] fed, then 16 greedy tokens.
//!   [`TRIGGER`] is the 311 ids the free arm generated on a plan of 809 card
//!   experts before its streams went non-finite: at position 315, its last,
//!   every expert the router selects at layer 34 scores `sqrt_softplus` 0
//!   (their logits sit below −16.6, where `1 + e^x` rounds to 1), and ik's
//!   CPU build divides the same zeros by their zero sum. The free arm's path
//!   moves with the placement and the rounding; this one is fed.
//!
//! Red on any of: a position whose streams hold a NaN or an infinity at any
//! seam (the engine's step refuses it too, with the fault word's layer and
//! site; the probe names the seam and the buffer); a
//! run of [`COLLAPSE`] or more equal generated tokens; a position whose eager
//! argmax is not the engine's token; and, on the free arm, a first difference
//! with ik's greedy ids (`greedy-ik-cpu-64-p0.tsv`, which stops at ik's EOS)
//! where our margin is not below [`GREEDY_MARGIN`].

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_long: built without the `deepseek41` feature; see `just gate-gpu-ds41-long`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_long", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_finite.rs"]
mod finite;

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::head::Head;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_deepseek41::body::{self, Deepseek41Model};
    use bloomery_gpu_gates::prompts::{read_greedy, read_prompts};
    use bloomery_gpu_gates::{GateError, checks_failed, data_dir, verdict};
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::placement::workstation;

    use crate::finite;

    /// The serving context, the gate placement's.
    const CTX_MAX: u64 = workstation::CTX_MAX;
    /// Generated tokens of the free arm unless `-n` says otherwise.
    const FREE_N: usize = 330;
    /// Generated tokens after the trigger's fed ids.
    const TRIGGER_N: usize = 16;
    /// A run of this many equal generated tokens is a collapse.
    const COLLAPSE: usize = 8;
    /// The step gate's `--greedy` rule and value (`gate_deepseek41_step.rs`):
    /// at the first id that differs from ik's, our margin must be below it.
    const GREEDY_MARGIN: f32 = 1.5;

    /// The free arm's generated ids through position 315 on a gate placement
    /// of 809 card experts, the last of them fed at position 315.
    #[rustfmt::skip]
    const TRIGGER: [u32; 311] = [
        11111, 16, 1, 1, 201, 61, 19836, 14, 223, 19, 14, 223, 20, 14, 223, 21,
        14, 223, 22, 14, 223, 23, 14, 223, 24, 14, 223, 25, 14, 223, 26, 14,
        223, 27, 14, 223, 553, 14, 223, 779, 14, 223, 736, 14, 223, 907, 14, 223,
        929, 14, 223, 856, 14, 223, 926, 14, 223, 1002, 14, 223, 864, 14, 223, 511,
        14, 223, 397, 14, 223, 1602, 14, 223, 1302, 14, 223, 1349, 14, 223, 1173, 14,
        223, 1069, 14, 223, 1450, 14, 223, 1477, 14, 223, 1449, 14, 223, 1557, 14, 223,
        1059, 14, 223, 2181, 14, 223, 2111, 14, 223, 1671, 14, 223, 2012, 14, 223, 1810,
        14, 223, 1872, 14, 223, 1942, 14, 223, 2080, 14, 223, 2116, 14, 223, 1484, 14,
        223, 3286, 14, 223, 3180, 14, 223, 3354, 14, 223, 2240, 14, 223, 1883, 14, 223,
        2372, 14, 223, 2491, 14, 223, 2170, 14, 223, 2505, 14, 223, 1328, 14, 223, 4287,
        14, 223, 4157, 14, 223, 4414, 14, 223, 4364, 14, 223, 2315, 14, 223, 3661, 14,
        223, 3351, 14, 223, 3175, 14, 223, 3318, 14, 223, 1683, 14, 223, 4739, 14, 223,
        4858, 14, 223, 4774, 14, 223, 2892, 14, 223, 2738, 14, 223, 2574, 14, 223, 3186,
        14, 223, 2973, 14, 223, 3259, 14, 223, 2122, 14, 223, 5863, 14, 223, 4610, 14,
        223, 5817, 14, 223, 6048, 14, 223, 2402, 14, 223, 4307, 14, 223, 3045, 14, 223,
        2597, 14, 223, 3981, 14, 223, 1892, 14, 223, 5929, 14, 223, 6078, 14, 223, 6131,
        14, 223, 5844, 14, 223, 5361, 14, 223, 5926, 14, 223, 5198, 14, 223, 2851, 14,
        223, 4362, 14, 223, 2225, 14, 223, 6207, 14, 223, 6152, 14, 223, 6420, 14, 223,
        6338, 14, 223, 2875, 14, 223, 5936, 14, 223, 5106, 14, 223, 3565, 14, 223, 1977,
        14, 223, 1457, 2296, 1, 0, 5,
    ];

    struct Args {
        n_gen: usize,
        free: bool,
        trigger: bool,
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str = "usage: gate_deepseek41_long [--free] [--trigger] [-n N]";
        let mut a = Args {
            n_gen: FREE_N,
            free: false,
            trigger: false,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--free" => a.free = true,
                "--trigger" => a.trigger = true,
                "-n" => {
                    a.n_gen = it
                        .next()
                        .ok_or_else(|| format!("-n needs a value: {USAGE}"))?
                        .parse()?;
                }
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        if !(a.free || a.trigger) {
            (a.free, a.trigger) = (true, true);
        }
        if a.n_gen < 2 {
            return Err("-n wants at least two generated tokens".into());
        }
        Ok(a)
    }

    pub fn run() -> Result<(), GateError> {
        let args = parse_args()?;
        let path = workstation::model_v41();
        let file = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let hp = Hparams::read(&file)?;
        let mut m = body::open(file, workstation::plan_gate, usize::try_from(CTX_MAX)?)?;
        m.set_mode(StepMode::Graph);
        let mut head = {
            let (gpu, w, body) = m.body_parts("gate_deepseek41_long")?;
            let map = body.slot_map();
            let held: Vec<usize> = map
                .layers()
                .map(|l| map.on_card(l))
                .filter(|&n| n > 0)
                .collect();
            println!(
                "load: gate placement, {} card experts on {} layers",
                held.iter().sum::<usize>(),
                held.len()
            );
            Head::new(gpu, w, hp.rms_eps)?
        };
        println!("capture graph_nodes={}", m.capture_step()?);
        let dir = data_dir().join("greedy-ds41");
        let prompt = read_prompts(&dir.join("prompt0.tsv"))?
            .into_iter()
            .next()
            .ok_or("prompt0.tsv holds no row")?
            .tokens;
        let mut pass = true;
        if args.free {
            let ik = read_greedy(&dir.join("greedy-ik-cpu-64-p0.tsv"))?
                .into_iter()
                .next()
                .ok_or("greedy-ik-cpu-64-p0.tsv holds no row")?;
            let arm = run_arm(&mut m, &mut head, &prompt, args.n_gen)?;
            pass &= arm.report("free", prompt.len());
            pass &= vs_ik(&arm, &ik.gen_ids);
        }
        if args.trigger {
            let fed: Vec<u32> = prompt.iter().chain(&TRIGGER).copied().collect();
            let arm = run_arm(&mut m, &mut head, &fed, TRIGGER_N)?;
            pass &= arm.report("trigger", fed.len());
        }
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: gate_deepseek41_long");
        Ok(())
    }

    /// What one arm's positions read and generated.
    #[derive(Default)]
    struct Arm {
        /// The generated tokens, token 0 out of the last fed position, and
        /// each one's margin (top logit minus the runner-up).
        tokens: Vec<u32>,
        margins: Vec<f32>,
        positions: usize,
        /// Positions whose streams were not finite somewhere, and the first
        /// one's reading.
        nonfinite: usize,
        first_bad: Option<(u32, String)>,
        /// Positions whose eager argmax was not the engine's token, and the
        /// first one: its position, the eager argmax and the engine's.
        differs: usize,
        first_differs: Option<(u32, u32, u32)>,
    }

    impl Arm {
        /// The arm's three verdict lines and its tokens; whether all passed.
        fn report(&self, name: &str, fed: usize) -> bool {
            let generated = self.tokens.len();
            let finite = self.nonfinite == 0;
            match &self.first_bad {
                None => println!(
                    "{name}: {} positions ({fed} fed, {generated} generated), every seam finite: {}",
                    self.positions,
                    verdict(true)
                ),
                Some((pos, reading)) => println!(
                    "{name}: {} positions ({fed} fed, {generated} generated), {} with a non-finite \
                     stream, the first at position {pos}: {reading}: {}",
                    self.positions,
                    self.nonfinite,
                    verdict(false)
                ),
            }
            let (len, at) = longest_run(&self.tokens);
            let run_ok = len < COLLAPSE;
            println!(
                "{name}: longest run of one generated token {len} (token {} from generated step \
                 {at}), a collapse at {COLLAPSE}: {}",
                self.tokens.get(at).copied().unwrap_or(0),
                verdict(run_ok)
            );
            let eager_ok = self.differs == 0;
            match self.first_differs {
                None => println!(
                    "{name}: the eager argmax is the engine's token at every position: {}",
                    verdict(true)
                ),
                Some((pos, eager, engine)) => println!(
                    "{name}: {} positions whose eager argmax is not the engine's token, the first \
                     at position {pos} (eager {eager}, engine {engine}): {}",
                    self.differs,
                    verdict(false)
                ),
            }
            println!("{name}: tokens {:?}", self.tokens);
            finite && run_ok && eager_ok
        }
    }

    /// The longest run of one value in `tokens` and where it starts.
    fn longest_run(tokens: &[u32]) -> (usize, usize) {
        let (mut best, mut at, mut len, mut start) = (0, 0, 0, 0);
        for (i, &t) in tokens.iter().enumerate() {
            if i > 0 && tokens[i - 1] == t {
                len += 1;
            } else {
                (len, start) = (1, i);
            }
            if len > best {
                (best, at) = (len, start);
            }
        }
        (best, at)
    }

    /// Feed `fed` from a reset, then `n_gen` greedy tokens, every position
    /// through the probe and then the engine.
    fn run_arm(
        m: &mut Deepseek41Model,
        head: &mut Head,
        fed: &[u32],
        n_gen: usize,
    ) -> Result<Arm, GateError> {
        m.reset()?;
        let mut arm = Arm::default();
        let (&last, before) = fed.split_last().ok_or("an arm with no fed ids")?;
        for &t in before {
            checked(m, head, t, &mut arm)?;
        }
        let mut next = last;
        while arm.tokens.len() < n_gen {
            let (tok, margin) = checked(m, head, next, &mut arm)?;
            arm.tokens.push(tok);
            arm.margins.push(margin);
            next = tok;
        }
        Ok(arm)
    }

    /// One position: `tok`'s step observed, the position taken back, `tok`
    /// stepped through the engine; the engine's token and the observed
    /// logits' margin back, the reading kept in `arm`.
    fn checked(
        m: &mut Deepseek41Model,
        head: &mut Head,
        tok: u32,
        arm: &mut Arm,
    ) -> Result<(u32, f32), GateError> {
        let pos = m.pos();
        let o = finite::observed_step(m, head, tok, pos, &mut |_, _, _| Ok(()))?;
        m.rollback(pos)?;
        let next = m.step(&[tok])?;
        arm.positions += 1;
        if o.first_nonfinite().is_some() {
            arm.nonfinite += 1;
            arm.first_bad.get_or_insert_with(|| (pos, o.describe()));
        }
        if o.token() != next {
            arm.differs += 1;
            arm.first_differs.get_or_insert((pos, o.token(), next));
        }
        Ok((next, margin(&o.logits)))
    }

    /// The top logit minus the runner-up, from the logits' bits.
    fn margin(logits: &[u32]) -> f32 {
        let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &bits in logits {
            let x = f32::from_bits(bits);
            if x > a {
                (a, b) = (x, a);
            } else if x > b {
                b = x;
            }
        }
        a - b
    }

    /// The free arm's tokens against ik's greedy ids, as far as ik's go (its
    /// greedy stops at the model's EOS, which it records): equal, or at the
    /// first difference our margin below [`GREEDY_MARGIN`].
    fn vs_ik(arm: &Arm, ik: &[u32]) -> bool {
        let n = ik.len().min(arm.tokens.len());
        let first = (0..n).find(|&i| arm.tokens[i] != ik[i]);
        let ok = first.is_none_or(|i| arm.margins[i] < GREEDY_MARGIN);
        let how = match first {
            None => "equal".to_string(),
            Some(i) => format!(
                "first difference at generated step {i}, our margin {:.4} (pass rule < \
                 {GREEDY_MARGIN})",
                arm.margins[i]
            ),
        };
        println!(
            "free: ik's greedy ids {ik:?} (to its EOS), ours {:?}: {how}: {}",
            &arm.tokens[..n],
            verdict(ok)
        );
        ok
    }
}
