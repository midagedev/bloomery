//! `generate_qwen3moe` — the decode CLI of the qwen3moe family's engines: a
//! qwen3moe (Qwen3-30B-A3B) or a qwen35moe (Qwen3.6-35B-A3B) file, the
//! architecture read from the file's header (any other is refused by name),
//! the whole model on one card, greedy, one token per step, and the timing
//! runner's ruler.
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
//!
//! A qwen35moe file runs `auto` and `gemm` through `Body35`'s prompt call
//! (`Qwen35moeModel::prefill_with`: the same plan, every unit a walk of the
//! layer program over the ubatch arena, eager in either mode — a ubatch of
//! more than eight ids through each op's wide arm, a pass through its gemv
//! arm), its ubatch size `BLOOMERY_QWEN3_UBATCH` too, clipped to the cache;
//! `pass` runs passes of up to eight positions, each a `step_rows` of its
//! size (in graph mode a replay) — bit for bit its one-token steps — and a
//! pass of one id a step. Its `load` line names `ubatch=`; its `stat prompt`
//! line's `ubatch_tokens=` counts the ubatches' ids, and its `image_bytes=`
//! is the whole prompt's image, which the passes after them read too.
//! `--prompt` is refused by the tokenizer (its pre-tokenizer `qwen35` is not
//! one the crate runs).
//!
//! Lines: `prompt_ids`, `load` (`arch=` the file's architecture; with the
//! decode flash pass: `flash_mma=`; for qwen3moe the ubatches' attention,
//! `ubatch_attn=gqa_prefill_flash`, and their size,
//! `ubatch=`, on the `load` line because the `time prompt` row's shape is
//! parsed to its end; and `rope_table_us=`, the host time that computed the
//! rope table every path reads at load, one `RopeTable::push` for each of
//! the `ctx` positions — a runtime value; for qwen35moe `store_bytes=`, the
//! attention layers' K/V planes and the delta layers' states), in graph
//! mode `capture graph_nodes=` and `capture prefill_graphs=<n> nodes=<m=1>,…
//! ms= vram_bytes=` (every pass size captured before the prompt: its wall
//! and the card's free bytes it took, runtime values; for qwen35moe the
//! `m = 1` entry is the decode step's graph, which a pass of one id
//! replays), `step 0 pos tok`
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
//! the caches with a pattern (for qwen35moe the attention layers' K/V
//! planes; the delta layers' states stay as they stand), and one literal
//! token (id 0) is prefilled after them, so the timed steps run at the
//! positions a D-token prompt leaves. The tokens it prints are
//! meaningless; only the timing is. Its `time prompt` row is that one
//! token's prefill, `n=1`.
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
    use bloomery_gpu::arch::qwen3moe::router::MAX_TOKENS;
    use bloomery_gpu::arch::qwen3moe::ubatch::{ImageWrite, ubatch_size};
    use bloomery_gpu::arch::qwen3moe::{
        Body, Body35, Open35, PrefillPath, PrefillPlan, PrefillStep, Qwen35moeModel,
    };
    use bloomery_gpu::model::{Instrumented, MAX_PASS_ROWS, StepMode};
    use bloomery_gpu::{Gpu, GpuModel, Qwen3moeModel};
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use gguf::Split;
    use model::arch::Arch;
    use std::num::NonZeroUsize;
    use std::time::Instant;
    use tokenizer::Tokenizer;

    // A Qwen3.6 prompt is cut by the qwen3moe pass plan (`PrefillPlan`),
    // whose passes are `MAX_TOKENS` long; each must be a pass `step_rows`
    // takes.
    const _: () = assert!(MAX_TOKENS == MAX_PASS_ROWS);

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

    /// The model file ([`ref_model_path`]) and its architecture, read from
    /// its first shard's header (`Arch::detect`): qwen3moe or qwen35moe; any
    /// other file is refused by name.
    fn open_file() -> Result<(Split, Arch), GateError> {
        let path = ref_model_path()?;
        let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let first = file
            .shard(0)
            .ok_or_else(|| format!("{} opened with no shard", path.display()))?;
        match Arch::detect(first)? {
            arch @ (Arch::Qwen3moe | Arch::Qwen35moe) => Ok((file, arch)),
            other => Err(format!(
                "{} is a {} file; generate_qwen3moe runs qwen3moe and qwen35moe files",
                path.display(),
                file.architecture().unwrap_or(other.name())
            )
            .into()),
        }
    }

    /// A body's prompt schedule as this CLI drives it, beside the
    /// skeleton's own step.
    trait Prompted: Instrumented {
        /// The path the body runs for `--prefill <path>`; a path it has not
        /// is refused by name, before the load.
        fn path(path: PrefillPath) -> Result<PrefillPath, GateError>;
        /// The units a prompt of `n` ids runs as by `path` (a
        /// [`Prompted::path`] answer).
        fn plan(m: &GpuModel<Self>, n: usize, path: PrefillPath) -> Result<PrefillPlan, GateError>;
        /// Run `ids` by that plan from the model's position; the argmax
        /// after the last.
        fn prefill(
            m: &mut GpuModel<Self>,
            ids: &[u32],
            path: PrefillPath,
        ) -> Result<u32, GateError>;
        /// The last prompt image a ubatch of `plan` wrote; `None` when none
        /// did.
        fn image(m: &GpuModel<Self>, plan: &PrefillPlan) -> Result<Option<ImageWrite>, GateError>;
    }

    impl Prompted for Body {
        fn path(path: PrefillPath) -> Result<PrefillPath, GateError> {
            Ok(path)
        }

        fn plan(m: &Qwen3moeModel, n: usize, path: PrefillPath) -> Result<PrefillPlan, GateError> {
            Ok(m.prefill_plan(n, path)?)
        }

        fn prefill(
            m: &mut Qwen3moeModel,
            ids: &[u32],
            path: PrefillPath,
        ) -> Result<u32, GateError> {
            Ok(m.prefill_with(ids, path)?)
        }

        fn image(m: &Qwen3moeModel, _: &PrefillPlan) -> Result<Option<ImageWrite>, GateError> {
            Ok(m.ubatch_prologue()?.last)
        }
    }

    impl Prompted for Body35 {
        /// Every path: `pass` the captured passes (`step_rows`), `auto` and
        /// `gemm` the prompt call.
        fn path(path: PrefillPath) -> Result<PrefillPath, GateError> {
            Ok(path)
        }

        fn plan(m: &Qwen35moeModel, n: usize, path: PrefillPath) -> Result<PrefillPlan, GateError> {
            match path {
                PrefillPath::Pass => Ok(PrefillPlan::new(n, path, NonZeroUsize::MIN)),
                PrefillPath::Auto | PrefillPath::Gemm => Ok(m.prefill_plan(n, path)?),
            }
        }

        /// `pass`: each pass of the plan as one `step_rows` of its size, a
        /// pass of one id as a step, bit for bit one step per token; `auto`
        /// and `gemm`: the prompt call.
        fn prefill(
            m: &mut Qwen35moeModel,
            ids: &[u32],
            path: PrefillPath,
        ) -> Result<u32, GateError> {
            if path != PrefillPath::Pass {
                return Ok(m.prefill_with(ids, path)?);
            }
            let plan = Self::plan(m, ids.len(), path)?;
            let (mut at, mut next) = (0usize, None);
            for step in &plan.steps {
                let PrefillStep::Pass(k) = *step else {
                    return Err(format!("a qwen35moe pass plan holds a ubatch ({plan})").into());
                };
                let rows = ids
                    .get(at..at + k)
                    .ok_or_else(|| format!("the plan {plan} runs past {} ids", ids.len()))?;
                at += k;
                next = Some(pass35(m, rows)?);
            }
            next.ok_or_else(|| "the prompt has no ids".into())
        }

        /// The prompt call's image when a ubatch of `plan` ran, its
        /// `tokens` the ubatches' ids.
        fn image(m: &Qwen35moeModel, plan: &PrefillPlan) -> Result<Option<ImageWrite>, GateError> {
            let ub = plan.ubatch_tokens();
            Ok(m.prompt_image()?
                .filter(|_| ub > 0)
                .map(|w| ImageWrite { tokens: ub, ..w }))
        }
    }

    /// One pass of `rows` ids on the Qwen3.6 model: the argmax after the last.
    fn pass35(m: &mut Qwen35moeModel, rows: &[u32]) -> Result<u32, GateError> {
        fn last<const M: usize>(m: &mut Qwen35moeModel, rows: &[u32]) -> Result<u32, GateError> {
            let rows: [u32; M] = rows.try_into()?;
            Ok(m.step_rows::<M>(rows)?[M - 1])
        }
        match rows.len() {
            1 => Ok(m.step(rows)?),
            2 => last::<2>(m, rows),
            3 => last::<3>(m, rows),
            4 => last::<4>(m, rows),
            5 => last::<5>(m, rows),
            6 => last::<6>(m, rows),
            7 => last::<7>(m, rows),
            8 => last::<8>(m, rows),
            k => Err(format!("a pass of {k} ids; a Qwen3.6 pass takes 1..={MAX_PASS_ROWS}").into()),
        }
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
        let t = Instant::now();
        let (file, arch) = open_file()?;
        let path = match arch {
            Arch::Qwen35moe => Body35::path(path)?,
            _ => Body::path(path)?,
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
        if arch == Arch::Qwen35moe
            && path == PrefillPath::Pass
            && logits
            && arms.iter().any(|a| a.n_gen == 1)
        {
            return Err(
                "--logits with -n 1 and --prefill pass on a qwen35moe file: the prompt's last \
                 pass leaves its logits in its last row's head, and --logits reads row 0's"
                    .into(),
            );
        }
        let listed = !arm_specs.is_empty();
        if !listed {
            println!("prompt_ids {:?}", arms[0].ids);
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
        match arch {
            Arch::Qwen35moe => drive(open_qwen35(file, ctx, mode, t)?, &run, &arms, listed, sync),
            _ => drive(open_qwen3(file, ctx, mode, t)?, &run, &arms, listed, sync),
        }
    }

    /// The Qwen3-30B-A3B model of `file`, its `load` line, and in graph mode
    /// the step and every prefill pass captured and their `capture` lines.
    fn open_qwen3(
        file: Split,
        ctx: usize,
        mode: StepMode,
        t: Instant,
    ) -> Result<Qwen3moeModel, GateError> {
        let mut m = Qwen3moeModel::open(Gpu::new()?, file, Qwen3moeModel::lever_opts(ctx)?)?;
        m.set_mode(mode);
        println!(
            "load arch=qwen3moe resident_bytes={} ctx={ctx} layers={} mode={} flash_mma={} \
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
            let (free0, _) = m.gpu().mem_info()?;
            let t = Instant::now();
            let nodes = m.capture_prefill()?;
            capture_line(&nodes, t, free0, m.gpu())?;
        }
        Ok(m)
    }

    /// The Qwen3.6-35B-A3B model of `file` with the tensor-core decode
    /// flash (the engine's), its `load` line, and in graph mode the step and
    /// the passes of 2 to [`MAX_PASS_ROWS`] rows captured and their
    /// `capture` lines.
    fn open_qwen35(
        file: Split,
        ctx: usize,
        mode: StepMode,
        t: Instant,
    ) -> Result<Qwen35moeModel, GateError> {
        let o = Open35 {
            ctx,
            mma: true,
            ubatch: ubatch_size()?,
        };
        let mut m = Qwen35moeModel::open(Gpu::new()?, file, o)?;
        m.set_mode(mode);
        let body = m.body("generate_qwen3moe")?;
        println!(
            "load arch=qwen35moe resident_bytes={} ctx={ctx} layers={} mode={} flash_mma={} \
             store_bytes={} ubatch_attn=gqa_prefill_flash_256 ubatch={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            mode_name(mode),
            body.flash_mma(),
            body.store_bytes(),
            m.ubatch()?,
            t.elapsed().as_secs_f64()
        );
        if mode == StepMode::Graph {
            let step = m.capture_step()?;
            println!("capture graph_nodes={step}");
            let (free0, _) = m.gpu().mem_info()?;
            let t = Instant::now();
            let nodes = vec![
                step,
                m.capture_rows::<2>()?,
                m.capture_rows::<3>()?,
                m.capture_rows::<4>()?,
                m.capture_rows::<5>()?,
                m.capture_rows::<6>()?,
                m.capture_rows::<7>()?,
                m.capture_rows::<8>()?,
            ];
            capture_line(&nodes, t, free0, m.gpu())?;
        }
        Ok(m)
    }

    /// The `capture prefill_graphs=` line: each pass size's node count, the
    /// captures' wall since `t` and the card bytes they took since `free0`.
    fn capture_line(nodes: &[usize], t: Instant, free0: usize, gpu: &Gpu) -> Result<(), GateError> {
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let (free1, _) = gpu.mem_info()?;
        let list: Vec<String> = nodes.iter().map(usize::to_string).collect();
        println!(
            "capture prefill_graphs={} nodes={} ms={ms:.1} vram_bytes={} (runtime values)",
            nodes.len(),
            list.join(","),
            free0.saturating_sub(free1)
        );
        Ok(())
    }

    /// Every arm of the run on the loaded model `m`, in a session over it;
    /// `listed` for an `--arm` list, whose arms open with their `arm` lines.
    fn drive<B: Prompted>(
        m: GpuModel<B>,
        run: &Run,
        arms: &[Arm],
        listed: bool,
        sync: bool,
    ) -> Result<(), GateError> {
        let mut s = Session::from_model(m, u32::try_from(run.ctx)?);
        let count = arms.len();
        s.arms(arms, |s, i, arm| {
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
            run_arm(s.model_mut(), run, arm)
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
    fn run_arm<B: Prompted>(m: &mut GpuModel<B>, run: &Run, arm: &Arm) -> Result<(), GateError> {
        let (ids, n_gen, warm) = (&arm.ids, arm.n_gen, run.warm);
        if let Some(d) = run.seed_depth.filter(|&d| d > 1) {
            m.seed_depth(d - 1)?;
            println!("seed rows={} pos={}", d - 1, m.pos());
        }
        let depth = run.seed_depth.unwrap_or(ids.len());
        let plan = B::plan(m, ids.len(), run.path)?;
        let t = Instant::now();
        let mut next = B::prefill(m, ids, run.path)?;
        let prefill_wall = t.elapsed();
        let image = B::image(m, &plan)?;
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
