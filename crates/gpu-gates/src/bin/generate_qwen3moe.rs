//! `generate_qwen3moe` — the decode CLI of the qwen3moe engine: the whole
//! model on one card, greedy, one token per step, and the timing runner's
//! ruler.
//!
//!     generate_qwen3moe (--prompt <text> | --tokens a,b,c | --seed-depth D)
//!                       [-n N] [--ctx C] [--mode eager|graph] [--prefill auto|pass|gemm]
//!                       [--time [--warm W]] [--logits]
//!     generate_qwen3moe --arm a,b,c[/N] [--arm ...] [--arm-sync] [-n N] [--ctx C] ...
//!
//! Defaults: N 32, C 4096, mode graph, prefill auto, W 0. A flag given twice
//! takes its last value. `--prompt` tokenizes the text with the file's own vocabulary
//! (`tokenizer`, no BOS: the file sets `add_bos_token` false; no
//! chat template) and prints the generated text after the ids.
//!
//! The prompt is prefilled (`Qwen3moeModel::prefill_with` by the `--prefill`
//! path: `auto` takes one pass for a prompt of up to eight ids and GEMM
//! ubatches of up to the load's ubatch size for a longer one, a tail of up
//! to eight after them one pass; `pass` passes of up to eight positions, the
//! same cache rows and answer as one step per token, in graph mode each a
//! replay of the pass of its size; `gemm` ubatches only); the argmax after
//! its last token is generated token 0, and `N − 1` feedback steps follow.
//! The ubatch size is `BLOOMERY_QWEN3_UBATCH` (1..=4096, default 4096), read
//! at load.
//! Lines: `prompt_ids`, `load` (with the decode flash pass: `flash_mma=`; the
//! ubatches' attention, `ubatch_attn=gqa_prefill_flash`; and their size,
//! `ubatch=`, on the `load` line because the `time prompt` row's shape is
//! parsed to its end; and `rope_table_us=`, the host time that computed the
//! rope table every path reads at load, one `RopeTable::push` for each of
//! the `ctx` positions — a runtime value), in graph
//! mode `capture graph_nodes=` and `capture prefill_graphs=<n> nodes=<m=1>,…
//! ms= vram_bytes=` (every pass size captured before the prompt: its wall
//! and the card's free bytes it took, runtime values), `step 0 pos tok`
//! (with the units the prompt took: `prefill_steps=`, and their shapes:
//! `plan=ubatch:512x2 pass:1`, a run of equal sizes as `<size>x<k>`), then,
//! all written after the loop,
//! `time prompt n=<P> ms= tok/s= passes=<K> kind=<gemm|prefill>` (`K` the
//! ubatches and passes; `gemm` when a ubatch ran)
//! (the wall of `prefill` through its token's readback, on every run: a
//! runtime value like the `load` line, a measurement only under the lease;
//! the prefill arena is allocated at load and the passes captured before
//! it, so that wall carries neither),
//! `stat prompt ubatch_tokens=<n> image_bytes= fill_us= copy_us=` (the host
//! prologue inside that wall before the first ubatch launch: the prompt
//! image's fill and the enqueue of its copy to the card, which the launches
//! behind it wait for, not the host; runtime values read from three clock
//! reads the engine takes on every prompt; `ubatch_tokens=0` when no ubatch
//! ran),
//! per feedback step `step i pos tok` (and `time step i ms=` under
//! `--time`); then
//! `tokens [..]`, `text` for a `--prompt` run, and under `--time` the
//! `SMOKE` footer with `generate`'s keys (`p50_ms=`, `mean_ms=`, `warm=`,
//! `tok/s(p50)=`).
//!
//! `--seed-depth D` stands the model at depth D: `seed_depth(D − 1)` fills
//! the caches with a pattern, and one literal token (id 0) is prefilled
//! after them, so the timed steps run at the positions a D-token prompt
//! leaves. The tokens it prints are meaningless; only the timing is. Its
//! `time prompt` row is that one token's prefill, `n=1`.
//!
//! `--arm a,b,c[/N]` runs several arms after one load (`app::Session::arms`),
//! in the order given: each prefills its ids and generates its `N` (`-n`
//! when it names none), and each after the first starts from the session's
//! clear, so it prints what it prints in a fresh process. Each arm opens
//! with `arm i=<i> arms=<k> ids=<P> n=<N>` and its `prompt_ids` line, then
//! the lines a one-prompt run prints from `step 0` on; the load and capture
//! lines print once, before arm 0. `--arm` does not mix with `--prompt`,
//! `--tokens` or `--seed-depth`. `--arm-sync` makes each arm wait, after its
//! `arm` line, for one line on stdin: the timing runner takes its witness
//! blocks there. A failed arm ends the process, naming the arm.
//!
//! `--logits` prints `logits n= argmax= fnv64=` after the `tokens` line: the
//! head's last logits row, read back once, by its f32 bits (FNV-1a 64).
//!
//! `--time` is a MEASUREMENT and belongs under the machine-wide lease
//! (`tools/ref/time-gate.sh`), never at a bare prompt. The cache's
//! height is `--ctx`: the flash's segment grid is fixed by it, so a timed
//! run names its ctx.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("generate_qwen3moe: built without the `gpu` feature; see `just gen-qwen3moe`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("generate_qwen3moe", cli::run())
}

#[cfg(feature = "gpu")]
mod cli {
    use app::Session;
    use bloomery_gpu::arch::qwen3moe::PrefillPath;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, Qwen3moeModel};
    use bloomery_gpu_gates::{GateError, open_split, ref_model_path};
    use model::arch::Arch;
    use std::time::Instant;
    use tokenizer::Tokenizer;

    /// The last value of flag `name`, if given.
    fn flag(name: &str) -> Result<Option<String>, GateError> {
        Ok(flags(name)?.pop())
    }

    /// Every value of flag `name`, in order.
    fn flags(name: &str) -> Result<Vec<String>, GateError> {
        let args: Vec<String> = std::env::args().collect();
        let mut out = Vec::new();
        for (i, a) in args.iter().enumerate() {
            if a == name {
                out.push(
                    args.get(i + 1)
                        .ok_or_else(|| format!("{name} needs a value"))?
                        .clone(),
                );
            }
        }
        Ok(out)
    }

    /// Comma-separated ids.
    fn ids_of(s: &str) -> Result<Vec<u32>, GateError> {
        Ok(s.split(',')
            .map(|v| v.trim().parse::<u32>())
            .collect::<Result<_, _>>()?)
    }

    /// What every arm of the run shares.
    struct Run {
        timed: bool,
        warm: usize,
        mode: StepMode,
        path: PrefillPath,
        seed_depth: Option<usize>,
        logits: bool,
        tok: Option<Tokenizer>,
        /// `--ctx`, as the `SMOKE` line prints it.
        ctx: usize,
    }

    /// One arm: its ids and its generated count.
    struct Arm {
        ids: Vec<u32>,
        n_gen: usize,
    }

    pub fn run() -> Result<(), GateError> {
        // A lever set to a value it does not take, or a retired name that is
        // set, is refused by name before anything loads.
        bloomery_levers::at_main(&[])?;
        let timed = std::env::args().any(|a| a == "--time");
        let sync = std::env::args().any(|a| a == "--arm-sync");
        let logits = std::env::args().any(|a| a == "--logits");
        let n_gen: usize = flag("-n")?.map_or(Ok(32), |s| s.parse())?;
        let ctx: usize = flag("--ctx")?.map_or(Ok(4096), |s| s.parse())?;
        let warm: usize = flag("--warm")?.map_or(Ok(0), |s| s.parse())?;
        let mode = match flag("--mode")?.as_deref() {
            None | Some("graph") => StepMode::Graph,
            Some("eager") => StepMode::Eager,
            Some(o) => return Err(format!("--mode is eager or graph, not {o}").into()),
        };
        let path = match flag("--prefill")?.as_deref() {
            None | Some("auto") => PrefillPath::Auto,
            Some("pass") => PrefillPath::Pass,
            Some("gemm") => PrefillPath::Gemm,
            Some(o) => return Err(format!("--prefill is auto, pass or gemm, not {o}").into()),
        };
        let seed_depth: Option<usize> = flag("--seed-depth")?.map(|s| s.parse()).transpose()?;
        let text = flag("--prompt")?;
        let tokens = flag("--tokens")?;
        let arm_specs = flags("--arm")?;
        let sources = usize::from(text.is_some())
            + usize::from(tokens.is_some())
            + usize::from(seed_depth.is_some())
            + usize::from(!arm_specs.is_empty());
        if sources != 1 {
            return Err("give exactly one of --prompt, --tokens, --seed-depth, --arm".into());
        }
        if sync && arm_specs.is_empty() {
            return Err("--arm-sync paces the arms of an --arm list, and none is given".into());
        }
        let tok = match &text {
            Some(_) => Some(Tokenizer::from_gguf(ref_model_path()?)?),
            None => None,
        };
        let arms: Vec<Arm> = if arm_specs.is_empty() {
            let ids: Vec<u32> = match (&text, &tokens, seed_depth) {
                (Some(t), _, _) => tok.as_ref().ok_or("no tokenizer")?.encode(t, true, false),
                (_, Some(s), _) => ids_of(s)?,
                (_, _, Some(_)) => vec![0],
                _ => unreachable!("one source by the check above"),
            };
            vec![Arm { ids, n_gen }]
        } else {
            arm_specs
                .iter()
                .map(|spec| {
                    let (ids, n) = match spec.split_once('/') {
                        Some((ids, n)) => (ids, n.parse::<usize>()?),
                        None => (spec.as_str(), n_gen),
                    };
                    Ok(Arm {
                        ids: ids_of(ids)?,
                        n_gen: n,
                    })
                })
                .collect::<Result<_, GateError>>()?
        };
        for arm in &arms {
            if arm.ids.is_empty() {
                return Err("the prompt has no ids".into());
            }
            if arm.n_gen == 0 || (timed && arm.n_gen <= warm + 1) {
                return Err(
                    format!("-n {} leaves no counted step (warm {warm})", arm.n_gen).into(),
                );
            }
            let depth = seed_depth.unwrap_or(arm.ids.len());
            if depth + arm.n_gen > ctx {
                return Err(
                    format!("depth {depth} + {} tokens pass --ctx {ctx}", arm.n_gen).into(),
                );
            }
        }
        let listed = !arm_specs.is_empty();
        if !listed {
            println!("prompt_ids {:?}", arms[0].ids);
        }
        let t = Instant::now();
        let file = open_split(Arch::Qwen3moe, "gen-qwen3moe")?;
        let mut m = Qwen3moeModel::open(Gpu::new()?, file, Qwen3moeModel::lever_opts(ctx)?)?;
        m.set_mode(mode);
        println!(
            "load resident_bytes={} ctx={ctx} layers={} mode={} flash_mma={} \
             ubatch_attn=gqa_prefill_flash ubatch={} rope_table_us={:.1} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            mode_name(mode),
            m.body("generate_qwen3moe")?.flash_mma(),
            m.ubatch()?,
            m.ubatch_prologue()?.table_build.as_secs_f64() * 1e6,
            t.elapsed().as_secs_f64()
        );
        if mode == StepMode::Graph {
            println!("capture graph_nodes={}", m.capture_step()?);
            let gpu = m.gpu();
            let (free0, _) = gpu.mem_info()?;
            let t = Instant::now();
            let nodes = m.capture_prefill()?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let (free1, _) = m.gpu().mem_info()?;
            let list: Vec<String> = nodes.iter().map(usize::to_string).collect();
            println!(
                "capture prefill_graphs={} nodes={} ms={ms:.1} vram_bytes={} (runtime values)",
                nodes.len(),
                list.join(","),
                free0.saturating_sub(free1)
            );
        }
        let run = Run {
            timed,
            warm,
            mode,
            path,
            seed_depth,
            logits,
            tok,
            ctx,
        };
        let mut s = Session::from_model(m, u32::try_from(ctx)?);
        let count = arms.len();
        s.arms(&arms, |s, i, arm| {
            if listed {
                println!(
                    "arm i={i} arms={count} ids={} n={}",
                    arm.ids.len(),
                    arm.n_gen
                );
                if sync {
                    let mut line = String::new();
                    if std::io::stdin().read_line(&mut line)? == 0 {
                        return Err(
                            format!("--arm-sync: stdin closed before arm {i} of {count}").into(),
                        );
                    }
                }
                println!("prompt_ids {:?}", arm.ids);
            }
            run_arm(s.model_mut(), &run, arm)
        })
        .map_err(|f| Box::new(f) as GateError)
    }

    fn mode_name(mode: StepMode) -> &'static str {
        if mode == StepMode::Graph {
            "graph"
        } else {
            "eager"
        }
    }

    /// One arm on the loaded model: its prefill, its steps, and every line
    /// from `step 0` on.
    fn run_arm(m: &mut Qwen3moeModel, run: &Run, arm: &Arm) -> Result<(), GateError> {
        let (ids, n_gen, warm) = (&arm.ids, arm.n_gen, run.warm);
        if let Some(d) = run.seed_depth.filter(|&d| d > 1) {
            m.seed_depth(d - 1)?;
            println!("seed rows={} pos={}", d - 1, m.pos());
        }
        let depth = run.seed_depth.unwrap_or(ids.len());
        let plan = m.prefill_plan(ids.len(), run.path)?;
        let t = Instant::now();
        let mut next = m.prefill_with(ids, run.path)?;
        let prefill_wall = t.elapsed();
        let image = m.ubatch_prologue()?.last;
        println!(
            "step 0 {} {next} (the {} prompt ids in prefill_steps={} units, plan={plan}, {:.2} s, \
             runtime value)",
            m.pos() - 1,
            ids.len(),
            plan.steps.len(),
            prefill_wall.as_secs_f64()
        );
        let mut tokens_out = Vec::with_capacity(n_gen);
        tokens_out.push(next);
        // Every line of the loop is held and written after it: a write is a
        // syscall, and the steps it would separate are the measurement.
        let mut rows: Vec<(u32, u32, f64)> = Vec::with_capacity(n_gen - 1);
        for _ in 1..n_gen {
            let t0 = Instant::now();
            next = m.step(&[next])?;
            rows.push((m.pos() - 1, next, t0.elapsed().as_secs_f64() * 1e3));
        }
        let prefill_ms = prefill_wall.as_secs_f64() * 1e3;
        println!(
            "time prompt n={} ms={prefill_ms:.4} tok/s={:.2} passes={} kind={}",
            ids.len(),
            ids.len() as f64 * 1e3 / prefill_ms,
            plan.steps.len(),
            plan.kind()
        );
        match image {
            Some(w) => println!(
                "stat prompt ubatch_tokens={} image_bytes={} fill_us={:.1} copy_us={:.1} \
                 (runtime values)",
                w.tokens,
                w.bytes,
                w.fill.as_secs_f64() * 1e6,
                w.copy.as_secs_f64() * 1e6
            ),
            None => println!("stat prompt ubatch_tokens=0 (no ubatch ran)"),
        }
        for (k, &(pos, tok, ms)) in rows.iter().enumerate() {
            let i = k + 1;
            tokens_out.push(tok);
            println!("step {i} {pos} {tok}");
            if run.timed {
                let tag = if i <= warm { " warm" } else { "" };
                println!("time step {i}{tag} ms={ms:.4}");
            }
        }
        println!("tokens {tokens_out:?}");
        if run.logits {
            let row = m.logits()?;
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(y.1).then(y.0.cmp(&x.0)))
                .map_or(0, |(i, _)| i);
            let fnv = row.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, v| {
                v.to_bits().to_le_bytes().iter().fold(h, |h, &b| {
                    (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
                })
            });
            println!("logits n={} argmax={argmax} fnv64={fnv:016x}", row.len());
        }
        if let Some(t) = &run.tok {
            println!("text {:?}", t.decode(&tokens_out));
        }
        if run.timed {
            let counted: Vec<f64> = rows[warm..].iter().map(|r| r.2).collect();
            let mut sorted = counted.clone();
            sorted.sort_by(f64::total_cmp);
            let p50 = sorted[sorted.len() / 2];
            let mean = counted.iter().sum::<f64>() / counted.len() as f64;
            println!(
                "SMOKE mode={} prompt_tokens={} depth={depth} seeded={} generated={n_gen} \
                 warm={warm} steps={} p50_ms={p50:.4} mean_ms={mean:.4} tok/s(p50)={:.2} ctx={}",
                mode_name(run.mode),
                ids.len(),
                run.seed_depth.is_some(),
                counted.len(),
                1e3 / p50,
                run.ctx
            );
        }
        Ok(())
    }
}
