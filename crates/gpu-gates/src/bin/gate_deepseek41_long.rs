//! GPU gate for long greedy runs of the V4.1 engine on the gate placement
//! (`workstation::plan_gate`, the 3090), the model opened through the
//! engine's own entry (`body::open`). Every position the run steps goes
//! first through the finite probe (`shared/ds41_finite.rs`): the position's
//! step eagerly, outside the graph, each sub-layer's streams read where it
//! wrote them; then the position is taken back and the token stepped through
//! the engine's captured graph, whose argmax is the run's token.
//!
//! Two arms on one load, each from a reset:
//! - `--free`: prompt row 7 of `$BLOOMERY_DATA/greedy-ds41` (four ids; ik's
//!   greedy run on it reaches 64 ids without an EOS, so the long run is
//!   compared over all of them — prompt 0's stops at its third token), then
//!   greedy tokens until the file's EOS, at most `-n` (330). The EOS is where
//!   generation ends: nothing after it is generated or judged.
//! - `--trigger`: prompt row 0 ("The capital of France is", the oracle's five
//!   ids) and [`TRIGGER`] fed, then 16 greedy tokens.
//!   [`TRIGGER`] is the 311 ids the free arm generated on a plan of 809 card
//!   experts before its streams went non-finite: at position 315, its last,
//!   every expert the router selects at layer 34 scores `sqrt_softplus` 0
//!   (their logits sit below −16.6, where `1 + e^x` rounds to 1), and ik's
//!   CPU build divides the same zeros by their zero sum. The free arm's path
//!   moves with the placement and the rounding; this one is fed.
//!
//! - `--faults` (its own recipe, `just gate-gpu-ds41-faults`, alone in its
//!   process so no earlier arm has touched the host tier's pages): the same
//!   prompt from a reset, then [`FAULT_STEPS`] greedy tokens through the
//!   engine's captured graph alone, each step's page faults read around it
//!   (`getrusage(RUSAGE_SELF)`, less the engram helper thread's own, which
//!   reads the engram table the load does not populate). Red when a step
//!   from the second on takes a major fault, or more than [`FAULT_MINOR_PIN`]
//!   minor ones: with the host set populated and locked at load
//!   (`BLOOMERY_HOST_LOCK=1`, which the recipe sets) a step touches no host
//!   page the load did not map and keep. Populated but not locked, another
//!   process's reads reclaim pages between the load and the step.
//!
//! Red on any of: a position whose streams hold a NaN or an infinity at any
//! seam (the engine's step refuses it too, with the fault word's layer and
//! site; the probe names the seam and the buffer); a
//! run of [`COLLAPSE`] or more generated tokens that repeat with period 1 or
//! 2 (one token, or two alternating); a position whose eager
//! argmax is not the engine's token; and, on the free arm, a first difference
//! with ik's greedy ids (`greedy-ik-cpu-64-p7.tsv`, which would stop at ik's EOS)
//! where our margin is not below
//! [`GREEDY_MARGIN`](bloomery_gpu_gates::GREEDY_MARGIN), the step gate's rule,
//! or, where ours equals ik's through ik's EOS, a run that did not stop at
//! that EOS itself. A run that differs from ik's within the margin rule does
//! not have to reach ik's EOS; the case says whether and where it stopped.

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

/// The longest stretch of `tokens` that repeats with one of `periods`
/// (`tokens[i] == tokens[i - p]` all through it), as `(length, start,
/// period)`, the smallest period first on a tie; `(0, 0, 0)` for no tokens.
/// A stretch of period `p` counts its first `p` tokens: period 1 is the
/// longest run of one token, period 2 an alternation's full length.
#[cfg(feature = "deepseek41")]
fn longest_periodic_run(tokens: &[u32], periods: &[usize]) -> (usize, usize, usize) {
    let mut best = (0, 0, 0);
    for &p in periods {
        if p == 0 || tokens.is_empty() {
            continue;
        }
        let (mut len, mut start) = (p.min(tokens.len()), 0);
        if len > best.0 {
            best = (len, start, p);
        }
        for i in p..tokens.len() {
            if tokens[i] == tokens[i - p] {
                len += 1;
            } else {
                (len, start) = (p, i + 1 - p);
            }
            if len > best.0 {
                best = (len, start, p);
            }
        }
    }
    best
}

/// The collapse check against token lists whose answer is known: a
/// two-token alternation, one token repeated, and an enumeration of
/// changing values around a fixed pair — whether each reads as a collapse
/// of at least `collapse` tokens is the list's expected verdict. The gate
/// runs it before it loads the model.
#[cfg(feature = "deepseek41")]
fn collapse_self_check(periods: &[usize], collapse: usize) -> bool {
    let alternation: Vec<u32> = (0..12)
        .map(|i| if i % 2 == 0 { 2296 } else { 83358 })
        .collect();
    let repeated = vec![7u32; 9];
    let counting: Vec<u32> = (0..10).flat_map(|n| [14, 223, 19 + n]).collect();
    let mut short_alt = vec![1u32, 2, 3];
    short_alt.extend([5, 6, 5, 6, 5, 6, 5]);
    let cases: [(&str, &[u32], bool); 4] = [
        ("alternation 2296/83358 x12", &alternation, true),
        ("one token x9", &repeated, true),
        ("enumeration 14,223,N x10", &counting, false),
        ("alternation of 7", &short_alt, false),
    ];
    let mut ok = true;
    for (name, tokens, want) in cases {
        let (len, at, period) = longest_periodic_run(tokens, periods);
        let got = len >= collapse;
        println!(
            "collapse self-check {name}: longest periodic run {len} (period {period} from {at}), \
             collapse={got} want={want}: {}",
            bloomery_gpu_gates::verdict(got == want)
        );
        ok &= got == want;
    }
    ok
}

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::head::Head;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_deepseek41::body::{self, Deepseek41Model};
    use bloomery_gpu_gates::prompts::{read_greedy, read_prompts};
    use bloomery_gpu_gates::{GREEDY_MARGIN, GateError, checks_failed, data_dir, verdict};
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::placement::workstation;

    use crate::finite;

    /// The serving context, the gate placement's.
    const CTX_MAX: u64 = workstation::CTX_MAX;
    /// The most generated tokens of the free arm unless `-n` says otherwise;
    /// the arm ends earlier at the file's EOS.
    const FREE_N: usize = 330;
    /// Generated tokens after the trigger's fed ids.
    const TRIGGER_N: usize = 16;
    /// A run of this many generated tokens repeating with a short period
    /// ([`COLLAPSE_PERIODS`]) is a collapse.
    const COLLAPSE: usize = 8;
    /// The periods the collapse check reads: one token repeated, and two
    /// alternating. Not three or more: a greedy run past the reference's end
    /// of text settles into a phrase loop of the model's own, which is text,
    /// not a collapse; and at [`COLLAPSE`] tokens a longer period is under
    /// three repetitions, which an enumeration reaches (`14, 223, N` in
    /// [`TRIGGER`]).
    const COLLAPSE_PERIODS: [usize; 2] = [1, 2];
    /// Generated tokens of the faults arm.
    const FAULT_STEPS: usize = 32;
    /// The most minor faults a step from the second on may take outside the
    /// engram helper.
    /// PIN(2026-09-25): measured 2 at generated step 26 and 0 at every other
    /// step, with the host set populated and locked (and on a populated run
    /// no other process reclaimed from); populate and lock off read 3,955 to
    /// 53,198 per step.
    const FAULT_MINOR_PIN: u64 = 2;

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
        faults: bool,
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str = "usage: gate_deepseek41_long [--free] [--trigger] [--faults] [-n N]";
        let mut a = Args {
            n_gen: FREE_N,
            free: false,
            trigger: false,
            faults: false,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--free" => a.free = true,
                "--trigger" => a.trigger = true,
                "--faults" => a.faults = true,
                "-n" => {
                    a.n_gen = it
                        .next()
                        .ok_or_else(|| format!("-n needs a value: {USAGE}"))?
                        .parse()?;
                }
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        if !(a.free || a.trigger || a.faults) {
            (a.free, a.trigger) = (true, true);
        }
        if a.n_gen < 2 {
            return Err("-n wants at least two generated tokens".into());
        }
        Ok(a)
    }

    pub fn run() -> Result<(), GateError> {
        let args = parse_args()?;
        if !crate::collapse_self_check(&COLLAPSE_PERIODS, COLLAPSE) {
            return Err(checks_failed());
        }
        let path = workstation::model_v41();
        let file = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let hp = Hparams::read(&file)?;
        let eos = file
            .value("tokenizer.ggml.eos_token_id")
            .and_then(|v| match v {
                gguf::Value::U32(e) => Some(*e),
                gguf::Value::I32(e) => u32::try_from(*e).ok(),
                _ => None,
            })
            .ok_or("the file names no EOS token")?;
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
        if args.faults {
            pass &= faults_arm(&mut m, &prompt)?;
        }
        if args.free {
            // Prompt 7: ik's greedy ids for it run the full 64 without an
            // EOS, so the free arm's long run is compared and judged; prompt
            // 0's stop at its third token.
            let free_prompt = read_prompts(&dir.join("prompt7.tsv"))?
                .into_iter()
                .next()
                .ok_or("prompt7.tsv holds no row")?
                .tokens;
            let ik = read_greedy(&dir.join("greedy-ik-cpu-64-p7.tsv"))?
                .into_iter()
                .next()
                .ok_or("greedy-ik-cpu-64-p7.tsv holds no row")?;
            let arm = run_arm(&mut m, &mut head, &free_prompt, args.n_gen, Some(eos))?;
            pass &= arm.report("free", free_prompt.len());
            pass &= vs_ik(&arm, &ik.gen_ids, eos);
        }
        if args.trigger {
            let fed: Vec<u32> = prompt.iter().chain(&TRIGGER).copied().collect();
            let arm = run_arm(&mut m, &mut head, &fed, TRIGGER_N, None)?;
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
        /// The generated step whose token ended the run (the stop token).
        stopped_at: Option<usize>,
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
            let (len, at, period) = crate::longest_periodic_run(&self.tokens, &COLLAPSE_PERIODS);
            let run_ok = len < COLLAPSE;
            println!(
                "{name}: longest periodic run {len} (period {period}, tokens {:?} from generated \
                 step {at}), a collapse at {COLLAPSE} with periods {COLLAPSE_PERIODS:?}: {}",
                self.tokens
                    .get(at..(at + period).min(self.tokens.len()))
                    .unwrap_or(&[]),
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

    /// This process's page faults so far, major and minor.
    fn process_faults() -> Result<[u64; 2], GateError> {
        // SAFETY: `rusage` is integers only, so all-zero is a valid value,
        // and `getrusage` writes only through the pointer it is given.
        let (rc, ru) = unsafe {
            let mut ru: libc::rusage = std::mem::zeroed();
            let rc = libc::getrusage(libc::RUSAGE_SELF, &raw mut ru);
            (rc, ru)
        };
        if rc != 0 {
            return Err(format!("getrusage: {}", std::io::Error::last_os_error()).into());
        }
        Ok([u64::try_from(ru.ru_majflt)?, u64::try_from(ru.ru_minflt)?])
    }

    /// The engram helper thread's page faults so far, major and minor; zero
    /// with the helper off (`BLOOMERY_ENGRAM_HELPER=0`), whose rows the
    /// step thread reads and which then count.
    fn helper_faults(m: &Deepseek41Model) -> Result<[u64; 2], GateError> {
        Ok(m.body("faults")?
            .step_rows()
            .helper_faults()
            .map_or([0, 0], |f| [f.major, f.minor]))
    }

    /// The faults arm: `prompt` from a reset, then [`FAULT_STEPS`] greedy
    /// steps of the engine's graph, each step's faults outside the engram
    /// helper against the pins from the second step on.
    fn faults_arm(m: &mut Deepseek41Model, prompt: &[u32]) -> Result<bool, GateError> {
        let host = bloomery_gpu::hybrid::host_levers()?;
        m.reset()?;
        let mut next = m.step(prompt)?;
        let (mut tokens, mut rows) = (
            Vec::with_capacity(FAULT_STEPS),
            Vec::with_capacity(FAULT_STEPS),
        );
        for _ in 0..FAULT_STEPS {
            let (p0, h0) = (process_faults()?, helper_faults(m)?);
            let tok = m.step(&[next])?;
            let (p1, h1) = (process_faults()?, helper_faults(m)?);
            let own = |i: usize| (p1[i] - p0[i]).saturating_sub(h1[i] - h0[i]);
            rows.push([own(0), own(1), h1[0] - h0[0], h1[1] - h0[1]]);
            tokens.push(tok);
            next = tok;
        }
        let mut ok = true;
        for (i, [maj, min, hmaj, hmin]) in rows.iter().enumerate() {
            let step = i + 1;
            let judged = step >= 2;
            let good = !judged || (*maj == 0 && *min <= FAULT_MINOR_PIN);
            ok &= good;
            println!(
                "faults step={step} majflt={maj} minflt={min} (engram helper majflt={hmaj} \
                 minflt={hmin}){}",
                if judged {
                    format!(": {}", verdict(good))
                } else {
                    " (first step: not judged)".to_string()
                }
            );
        }
        let judged = &rows[1..];
        let max_min = judged.iter().map(|r| r[1]).max().unwrap_or(0);
        let sum_maj: u64 = judged.iter().map(|r| r[0]).sum();
        println!(
            "faults: populate={} lock={}, steps 2..={FAULT_STEPS} outside the engram helper: \
             majflt total {sum_maj} (pin 0), minflt max {max_min} per step (pin \
             {FAULT_MINOR_PIN}): {}",
            host.populate,
            host.lock,
            verdict(ok)
        );
        println!("faults: tokens {tokens:?}");
        Ok(ok)
    }

    /// Feed `fed` from a reset, then greedy tokens, every position through
    /// the probe and then the engine: `n_gen` of them, or fewer when `stop`
    /// names a token (the file's EOS) — the run ends right after emitting it.
    fn run_arm(
        m: &mut Deepseek41Model,
        head: &mut Head,
        fed: &[u32],
        n_gen: usize,
        stop: Option<u32>,
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
            if Some(tok) == stop {
                arm.stopped_at = Some(arm.tokens.len() - 1);
                break;
            }
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
    /// greedy stops at the model's EOS, which it records as its last id):
    /// equal, or at the first difference our margin below [`GREEDY_MARGIN`].
    /// Where ours equals ik's through ik's EOS, our run must have stopped at
    /// that same EOS; where it differs within the rule, where ours stopped
    /// (or that it did not) is printed and not judged — ik's EOS is then not
    /// ours to reach.
    fn vs_ik(arm: &Arm, ik: &[u32], eos: u32) -> bool {
        let n = ik.len().min(arm.tokens.len());
        let first = (0..n).find(|&i| arm.tokens[i] != ik[i]);
        let margin_ok = first.is_none_or(|i| arm.margins[i] < GREEDY_MARGIN);
        let how = match first {
            None => "equal".to_string(),
            Some(i) => format!(
                "first difference at generated step {i}, our margin {:.4} (pass rule < \
                 {GREEDY_MARGIN})",
                arm.margins[i]
            ),
        };
        let ik_eos = (ik.last() == Some(&eos)).then(|| ik.len() - 1);
        let ours = match arm.stopped_at {
            Some(j) => format!("ours stopped at EOS {eos} at generated step {j}"),
            None => format!(
                "ours did not emit EOS {eos} in {} generated tokens",
                arm.tokens.len()
            ),
        };
        let (eos_ok, eos_how) = match (ik_eos, first) {
            (None, _) => (
                true,
                format!(
                    "ik's {} ids end without EOS {eos}: no stop to hold; {ours}",
                    ik.len()
                ),
            ),
            (Some(e), None) => {
                let ok = arm.stopped_at == Some(e);
                (
                    ok,
                    format!(
                        "ik stopped at EOS at generated step {e}, ours equal through it: {ours}"
                    ),
                )
            }
            (Some(e), Some(i)) => (
                true,
                format!(
                    "ik stopped at EOS at generated step {e}; ours differs from step {i}, so that \
                     stop is not ours to reach: {ours} (printed, not judged)"
                ),
            ),
        };
        println!(
            "free: ik's greedy ids {ik:?} (to its EOS), ours {:?}: {how}: {}",
            &arm.tokens[..n],
            verdict(margin_ok)
        );
        println!("free: EOS {eos_how}: {}", verdict(eos_ok));
        margin_ok && eos_ok
    }
}
