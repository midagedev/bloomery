//! `generate` — the thin end-to-end decode CLI, and the round's A/B ruler.
//!
//! Greedy only, one token at a time, no prefill kernel, no streaming, no
//! chat template, no tokenizer: prompts are pre-tokenized in
//! `tools/ref/prompts.tsv` or given as ids on the command line. The whole
//! job is to drive `GpuModel::step` — every layer plus the head, one
//! position per call — and to print what came out.
//!
//!     generate [--prompt-id <id> | --tokens a,b,c | --seed-depth D]
//!              [-n N] [--ctx C] [--mode eager|graph] [--time [--warm W]]
//!              [--ab R [--ab-set SET]]
//!
//! Defaults: prompt 0, N 32, ctx 512, mode graph, W 0.
//!
//! `--seed-depth D` stands the model at depth `D` without decoding a prompt
//! there. `GpuModel::seed_depth` fills `D - 1` cache rows directly and one
//! literal token is then fed through the ordinary path, so the timed steps
//! run at exactly the positions a `--tokens <D ids>` prompt leaves — same
//! `pos`, same `n_keys`, same untimed step in front to absorb the first
//! graph launch. What it does not reproduce is the values: the seeded rows
//! are a deterministic pattern, so the tokens printed under this flag are
//! meaningless and only the timing footer is. It exists because a depth of
//! 4096 otherwise costs 4096 single-token bodies before the first timed
//! step. It is refused together with a prompt, and together with `--ab`:
//! every `--ab` arm rewinds with `reset()`, which would erase the seed and
//! silently measure depth 1.
//!
//! `--time` is a MEASUREMENT and belongs under the machine-wide lease
//! (`tools/ref/time-gate.sh generate … --time`), never at a bare prompt. It
//! times each generated step host-side around `step(&[tok])` — the argmax
//! readback synchronizes inside, so the wall time is the whole step. The
//! prompt is fed untimed: it is P steps of the same body and would drag the
//! per-step distribution toward whatever the prompt length happens to be.
//! Nothing is printed between two timed steps: every line of the loop is
//! held and written after it, because a `println!` is a write syscall on
//! the host and the steps it would separate are the measurement.
//!
//! `--warm W` drops the first W generated steps from the `--time`
//! statistics; they still run, and their `time step` lines still print,
//! marked `warm`, so the clock-ramp question stays readable in the log. The
//! default of 0 is the old behaviour. It is refused with `--ab`, which
//! already runs one untimed round of every arm before it counts anything.
//!
//! Which number is the record: the fusion rounds this runner serves compare
//! GRAPH REPLAY µs, so the graph-mode footer is the one that becomes the
//! record and the eager footer prices the host submit path. Two costs sit
//! inside every timed step whichever mode it is — the four
//! `refresh_params` host-to-device copies, each of which synchronizes the
//! stream today, and the argmax readback.
//!
//! `--ab R` is the same-binary arm comparison, and also a MEASUREMENT. It
//! runs every `StepProbe` launch-shape arm `R` times inside ONE process on
//! ONE loaded model, so what separates two arms is the launch shape and not
//! a rebuild's link layout, a different allocation or a different lease.
//! The arm order rotates by round: in a fixed order the first arm of a
//! round reads fast, which is enough to flip a claim of about a percent.
//! Each arm run rewinds to an empty cache and re-captures its graph, so an
//! arm never inherits the previous one's state.
//!
//! `--ab-set` picks which arms: `launch` (the default) is the launch-shape
//! levers, each restoring one merge to the pair it replaced; `keyaxis` is
//! `StepProbe::keyaxis_arms` — the shipped path and each flash
//! stage-doubling lever alone, where an arm's gap to `base` is the price of
//! the attention stage it does twice. Both sets are same-binary, and every
//! arm of either writes the tokens `base` writes.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("generate: built without the `gpu` feature; see `just generate`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
use bloomery_gpu::Deepseek2Model;
#[cfg(feature = "gpu")]
use bloomery_gpu::model::{StepMode, StepProbe};
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::prompts::read_prompts;
#[cfg(feature = "gpu")]
use bloomery_gpu_gates::{GateError, open_model};

/// ik's own decode on this model, tok/s at the depths the depth table
/// stands at. A decode step's attention term is linear in the cached keys,
/// so a footer that quotes one row at every depth quotes the wrong engine
/// number everywhere but there.
#[cfg(feature = "gpu")]
const IK_REFERENCE: [(usize, f64); 3] = [(0, 207.59), (1024, 196.40), (4096, 181.32)];

/// Where the rows above were read. They are 3090 numbers and timed runs
/// are now taken on the A6000, so the footer carries the card with them
/// and a reader does not put the two in one table.
#[cfg(feature = "gpu")]
const IK_REFERENCE_WHERE: &str = "A6000 at 300 W, 2026-09-22, 3 rounds";

/// The ik row a run at `depth` is read against: the deepest measured depth
/// that is not past it, printed with that depth so the line is never
/// mistaken for a reading at this run's own. Rows are not interpolated — a
/// number between two measured points is one nobody measured.
#[cfg(feature = "gpu")]
fn ik_reference(depth: usize) -> String {
    let (at, tok_s) = IK_REFERENCE
        .iter()
        .rev()
        .find(|(d, _)| *d <= depth)
        .copied()
        .unwrap_or(IK_REFERENCE[0]);
    format!(
        "ik {tok_s} tok/s at depth {at} (deepest measured row at or below this run's \
         depth {depth}; {IK_REFERENCE_WHERE})"
    )
}

/// `tools/ref/prompts.tsv`, relative to this crate — the same file both
/// engines read their token ids from.
#[cfg(feature = "gpu")]
fn prompts_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/prompts.tsv")
}

/// The argument after flag `name`: `None` when the flag is absent, an error
/// when the flag is the last argument.
#[cfg(feature = "gpu")]
fn flag_value(name: &str) -> Result<Option<String>, GateError> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .map(|i| {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| GateError::from(format!("generate: {name} needs a value")))
        })
        .transpose()
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("generate", run())
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), GateError> {
    let timed = std::env::args().any(|a| a == "--time");
    let mode = match flag_value("--mode")?.as_deref() {
        None | Some("graph") => StepMode::Graph,
        Some("eager") => StepMode::Eager,
        Some(other) => {
            return Err(format!("generate: --mode is eager or graph, not {other}").into());
        }
    };
    let n_gen: usize = flag_value("-n")?.map_or(Ok(32), |s| s.parse())?;
    let ctx: usize = flag_value("--ctx")?.map_or(Ok(512), |s| s.parse())?;
    let warm: usize = flag_value("--warm")?.map_or(Ok(0), |s| s.parse())?;

    let seed_depth: Option<usize> = match flag_value("--seed-depth")? {
        Some(s) => Some(s.parse()?),
        None => None,
    };
    if seed_depth.is_some()
        && (flag_value("--tokens")?.is_some() || flag_value("--prompt-id")?.is_some())
    {
        return Err(
            "generate: --seed-depth prepares the cache instead of decoding a \
                    prompt, so it takes neither --tokens nor --prompt-id. Pass one."
                .into(),
        );
    }

    let tokens: Vec<u32> = match (seed_depth, flag_value("--tokens")?) {
        // Under `--seed-depth` one token is fed at the last seeded position:
        // the ordinary untimed step in front of the timed ones. Which token
        // does not matter — the seeded rows already make the output
        // meaningless — so it is the prompt set's own first id rather than a
        // second place where this model's BOS is written down.
        (Some(_), _) => {
            let rows = read_prompts(&prompts_path())?;
            vec![
                *rows.first().and_then(|r| r.tokens.first()).ok_or_else(|| {
                    format!("generate: no tokens in {}", prompts_path().display())
                })?,
            ]
        }
        (None, Some(list)) => list
            .split(',')
            .map(|t| t.trim().parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?,
        (None, None) => {
            let id: usize = flag_value("--prompt-id")?.map_or(Ok(0), |s| s.parse())?;
            let rows = read_prompts(&prompts_path())?;
            rows.iter()
                .find(|r| r.id == id)
                .ok_or_else(|| format!("generate: no prompt {id} in {}", prompts_path().display()))?
                .tokens
                .clone()
        }
    };
    if tokens.is_empty() {
        return Err("generate: empty prompt".into());
    }
    // The depth this run occupies: the seeded rows plus their one decoded
    // token, or the prompt's own length.
    let depth = seed_depth.unwrap_or(tokens.len());
    if depth == 0 {
        return Err("generate: --seed-depth wants a depth of at least 1".into());
    }
    if depth + n_gen > ctx {
        return Err(format!(
            "generate: depth {depth} + {n_gen} generated tokens exceed --ctx {ctx}"
        )
        .into());
    }

    // The node-price probe. Either lever makes this run a timing instrument
    // and the tokens it prints meaningless — the footer says which arm it was
    // so a row can never be read as a value run.
    let probe = StepProbe {
        pad_per_layer: flag_value("--probe-pad")?.map_or(Ok(0), |s| s.parse())?,
        skip_quant: std::env::args().any(|a| a == "--probe-skip-quant"),
        split_heads: std::env::args().any(|a| a == "--probe-split-heads"),
        split_kqvc: std::env::args().any(|a| a == "--probe-split-kqvc"),
        split_flash_quant: std::env::args().any(|a| a == "--probe-split-flash-quant"),
        split_moe_quant: std::env::args().any(|a| a == "--probe-split-moe-quant"),
        // The flash stage-doubling levers are an arm set, not flags: they
        // are only worth reading interleaved against `base`.
        ..StepProbe::default()
    };

    // `split_kqvc` rolls the kqvc quantize PAIR back into two launches, and
    // since the fmerge round there is no pair unless `split_flash_quant`
    // first rolls the fold back into one. Alone it is a silent no-op, which
    // is the shape a later round mistakes for "the lever is broken" — so it
    // is refused rather than ignored.
    if probe.split_kqvc && !probe.split_flash_quant {
        return Err("--probe-split-kqvc has no effect without \
                    --probe-split-flash-quant: the kqvc quantize is folded \
                    into the flash kernel, so there is no separate pair \
                    launch to split. Pass both, or neither."
            .into());
    }

    // Refused rather than ignored: an arm set with no `--ab` to run it is
    // the shape a later round reads as "the set is broken".
    let ab_set = flag_value("--ab-set")?;
    if ab_set.is_some() && flag_value("--ab")?.is_none() {
        return Err(
            "generate: --ab-set picks the arms of --ab, which this run has not asked \
                    for. Pass both, or neither."
                .into(),
        );
    }

    // Refused rather than ignored, like the levers above. `--ab` runs one
    // untimed round of every arm before it counts anything, so a second
    // warming lever there would be two names for one thing; and outside
    // `--time` there are no statistics for `--warm` to trim.
    if flag_value("--warm")?.is_some() {
        if flag_value("--ab")?.is_some() {
            return Err(
                "generate: --warm warms --time, and --ab already runs one untimed \
                        round of every arm before it counts. Pass one, not both."
                    .into(),
            );
        }
        if !timed {
            return Err("generate: --warm drops the first W steps from --time's \
                        statistics, which this run has not asked for. Pass both, or neither."
                .into());
        }
    }
    if timed && warm > 0 && warm >= n_gen.saturating_sub(1) {
        return Err(format!(
            "generate: --warm {warm} leaves no timed step of the {} that -n {n_gen} \
             generates — generated token 0 comes out of the prompt's own step, so the \
             warm count must stay under n - 1.",
            n_gen.saturating_sub(1)
        )
        .into());
    }

    let gguf = open_model()?;
    let mut model = Deepseek2Model::load_full(&gguf, ctx)?;
    model.set_mode(mode);
    if let Some(rounds) = flag_value("--ab")? {
        if seed_depth.is_some() {
            return Err(
                "generate: --ab with --seed-depth would measure depth 1: every arm \
                        rewinds with reset(), which zeroes the seeded rows. Use --time."
                    .into(),
            );
        }
        let rounds: usize = rounds.parse()?;
        let arms: Vec<(&str, StepProbe)> = match ab_set.as_deref() {
            None | Some("launch") => launch_arms().to_vec(),
            Some("keyaxis") => StepProbe::keyaxis_arms().to_vec(),
            Some(other) => {
                return Err(format!("generate: --ab-set is launch or keyaxis, not {other}").into());
            }
        };
        return ab(&mut model, &tokens, n_gen, rounds, mode, &arms);
    }
    if probe != StepProbe::default() {
        model.set_probe(probe)?;
    }
    println!(
        "probe pad_per_layer={} skip_quant={} split_heads={} split_kqvc={} \
         split_flash_quant={} split_moe_quant={}",
        probe.pad_per_layer,
        probe.skip_quant,
        probe.split_heads,
        probe.split_kqvc,
        probe.split_flash_quant,
        probe.split_moe_quant
    );
    println!(
        "load resident_bytes={} ctx={ctx} layers={} mode={} flash_mma={} seg_keys={}",
        model.resident_bytes(),
        model.stages()[0].layers().len(),
        if mode == StepMode::Graph {
            "graph"
        } else {
            "eager"
        },
        bloomery_gpu::flash::flash_mma(),
        bloomery_gpu::flash::seg_keys(),
    );
    if mode == StepMode::Graph {
        // Capture before the prompt so the node count prints once and the
        // first timed step is a replay, not a capture.
        println!("capture graph_nodes={}", model.capture_step()?);
    }

    // Depth `D` is `D - 1` prepared rows plus the one token below, so the
    // step that follows lands in row `D - 1` exactly as the last token of a
    // `D`-token prompt does.
    // Depth 1 is that token alone and has no rows to prepare.
    if let Some(d) = seed_depth.filter(|d| *d > 1) {
        model.seed_depth(d - 1)?;
        println!("seed rows={} pos={}", d - 1, model.pos());
    }

    // The prompt, one token per position; the argmax of its last token is
    // generated token 0, so it costs a `step` of P tokens and is not timed.
    // The N - 1 feedback steps after it are the per-step distribution.
    let mut next = model.step(&tokens)?;
    println!("step 0 {} {next}", model.pos() - 1);
    // `(pos, token, ms)` per feedback step, written out after the loop:
    // holding them keeps every write syscall out of the gaps between two
    // timed steps. The Vec is sized up front so the loop allocates nothing.
    let mut rows: Vec<(u32, u32, f64)> = Vec::with_capacity(n_gen.saturating_sub(1));
    for _ in 1..n_gen {
        let t0 = std::time::Instant::now();
        next = model.step(&[next])?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        rows.push((model.pos() - 1, next, ms));
    }
    for (k, (pos, tok, ms)) in rows.iter().enumerate() {
        let i = k + 1;
        println!("step {i} {pos} {tok}");
        if timed {
            // The warm steps print too, marked: `--warm` answers whether
            // the early steps are slow, and a log that hid them could not.
            let tag = if i <= warm { " warm" } else { "" };
            println!("time step {i}{tag} ms={ms:.4}");
        }
    }
    let step_ms: Vec<f64> = rows.iter().map(|(_, _, ms)| *ms).collect();

    if timed {
        if step_ms.is_empty() {
            return Err("generate: --time with -n 1 has no generated step to time".into());
        }
        let counted = &step_ms[warm..];
        let mut sorted = counted.to_vec();
        sorted.sort_by(f64::total_cmp);
        let p50 = sorted[sorted.len() / 2];
        let mean = counted.iter().sum::<f64>() / counted.len() as f64;
        // `steps` is the number of COUNTED steps: generated token 0 comes
        // out of the prompt's own `step` call and is not one of them, and
        // the first `warm` feedback steps are run but not counted.
        println!(
            "SMOKE mode={} prompt_tokens={} depth={depth} seeded={} generated={n_gen} \
             warm={warm} steps={} p50_ms={p50:.4} \
             mean_ms={mean:.4} tok/s(p50)={:.2} probe_pad={pad} probe_skip_quant={skip} \
             probe_split_heads={split} probe_split_kqvc={split_kqvc} \
             probe_split_flash_quant={split_fq} probe_split_moe_quant={split_mq}",
            if mode == StepMode::Graph {
                "graph"
            } else {
                "eager"
            },
            tokens.len(),
            seed_depth.is_some(),
            counted.len(),
            1e3 / p50,
            pad = probe.pad_per_layer,
            skip = probe.skip_quant,
            split = probe.split_heads,
            split_kqvc = probe.split_kqvc,
            split_fq = probe.split_flash_quant,
            split_mq = probe.split_moe_quant,
        );
        println!("reference {}", ik_reference(depth));
    }
    Ok(())
}

/// The same-binary arm comparison. Every arm is one `StepProbe` launch
/// shape; a round runs each arm once, and the round's starting arm rotates
/// so no arm keeps the first position. Per arm the runner prints every
/// round's p50 and then the mean of those p50s with its sample standard
/// deviation, which is what a reader needs to know whether a sub-percent
/// gap is a gap — this card's same-binary scatter is of that size.
///
/// `base` (every lever off) is the shipped path; each other arm of the
/// `launch` set restores one merge to the launch shape it replaced, and
/// each arm of the `keyaxis` set does one attention stage twice. Their
/// outputs are all bit-identical by the gates, so a difference here is
/// launch cost or that stage's cost alone.
#[cfg(feature = "gpu")]
fn ab(
    model: &mut Deepseek2Model,
    tokens: &[u32],
    n_gen: usize,
    rounds: usize,
    mode: StepMode,
    arms: &[(&str, StepProbe)],
) -> Result<(), GateError> {
    if rounds == 0 || n_gen < 2 {
        return Err("generate: --ab wants R >= 1 and -n >= 2".into());
    }
    let mut p50s: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); arms.len()];
    // One untimed warm round: the first arm of the first round otherwise
    // carries the load's cold caches into its own number.
    for (_, probe) in arms {
        arm_p50(model, tokens, n_gen, *probe)?;
    }
    for r in 0..rounds {
        for k in 0..arms.len() {
            let a = (k + r) % arms.len();
            let (name, probe) = arms[a];
            let p50 = arm_p50(model, tokens, n_gen, probe)?;
            p50s[a].push(p50);
            println!("ab round={r} slot={k} arm={name} p50_ms={p50:.4}");
        }
    }
    println!(
        "ab mode={} rounds={rounds} steps_per_round={} arms={}",
        if mode == StepMode::Graph {
            "graph"
        } else {
            "eager"
        },
        n_gen - 1,
        arms.len()
    );
    let base_mean = mean(&p50s[0]);
    for (i, (name, _)) in arms.iter().enumerate() {
        let m = mean(&p50s[i]);
        let sd = sd(&p50s[i]);
        println!(
            "ab arm={name} mean_p50_ms={m:.4} sd_ms={sd:.4} sd_pct={:.2} tok/s={:.2} \
             vs_base_pct={:+.2} vs_base_us={:+.1}",
            100.0 * sd / m,
            1e3 / m,
            100.0 * (m - base_mean) / base_mean,
            1e3 * (m - base_mean)
        );
    }
    // `--ab` is refused with `--seed-depth`, so the prompt's length is the
    // depth every arm stands at.
    println!("reference {}", ik_reference(tokens.len()));
    Ok(())
}

/// The launch-shape arm set, `--ab-set launch`: the shipped path and the
/// rollback of each merge the fusion rounds made.
#[cfg(feature = "gpu")]
fn launch_arms() -> [(&'static str, StepProbe); 4] {
    [
        ("base", StepProbe::default()),
        (
            "split_flash_quant",
            StepProbe {
                split_flash_quant: true,
                ..StepProbe::default()
            },
        ),
        (
            "split_moe_quant",
            StepProbe {
                split_moe_quant: true,
                ..StepProbe::default()
            },
        ),
        (
            "split_both",
            StepProbe {
                split_flash_quant: true,
                split_moe_quant: true,
                ..StepProbe::default()
            },
        ),
    ]
}

/// One arm run: rewind to an empty cache, set the probe (which drops any
/// captured graph, so the arm captures its own), feed the prompt untimed,
/// then time the `n_gen - 1` feedback steps and return their p50.
#[cfg(feature = "gpu")]
fn arm_p50(
    model: &mut Deepseek2Model,
    tokens: &[u32],
    n_gen: usize,
    probe: StepProbe,
) -> Result<f64, GateError> {
    model.reset()?;
    model.set_probe(probe)?;
    let mut next = model.step(tokens)?;
    let mut step_ms: Vec<f64> = Vec::with_capacity(n_gen - 1);
    for _ in 1..n_gen {
        let t0 = std::time::Instant::now();
        next = model.step(&[next])?;
        step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    step_ms.sort_by(f64::total_cmp);
    Ok(step_ms[step_ms.len() / 2])
}

#[cfg(feature = "gpu")]
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Sample standard deviation; 0.0 for a single round, which is honest — one
/// round has no spread to report.
#[cfg(feature = "gpu")]
fn sd(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}
