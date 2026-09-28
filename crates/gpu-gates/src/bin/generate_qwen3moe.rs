//! `generate_qwen3moe` — the decode CLI of the qwen3moe family's engines: a
//! qwen3moe (Qwen3-30B-A3B), a qwen35moe (Qwen3.6-35B-A3B) or a qwen4exp
//! (Qwen3.8-Flash-Next) file, the architecture read from the file's header
//! (any other is refused by name), the whole model on one card, greedy, one
//! token per step, and the timing runner's ruler.
//!
//!     generate_qwen3moe (--prompt <text> | --tokens a,b,c | --seed-depth D)
//!                       [-n N] [--ctx C] [--mode eager|graph]
//!                       [--prefill auto|pass|gemm|step] [--place a|gate]
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
//! A qwen4exp file runs through `Body38`, placed by its own plan
//! (`model::arch::qwen35moe::place`: every layer, the head and the embedding
//! on one card, every routed expert on the host tier) on the card `--place`
//! names — `a` the A6000 (the default), `gate` the 3090 — and printed as a
//! `plan` line. Its `--prefill` is `auto` (the default: `gemm` for a prompt
//! of nine positions or more, `pass` below), `gemm` (ubatches of up to the
//! load's size through the host tier's batch port at their width, the Q8_0
//! projections on q8 activations), `pass` (eager passes of up to eight
//! positions through the same port) or `step` (one captured step a
//! position); `--seed-depth` is refused by name (no synthetic depth), and so
//! is `--prompt` (the tokenizer, as for qwen35moe). Its `load` line names
//! `store_bytes=`, `prefill=`, `ubatch=` and `place=`; in graph mode its
//! `capture` line counts the step graph's nodes by kind against the
//! program's count, and a mismatch ends the run by name. Its plan prints as
//! `step:1x<P>`, `pass:<sizes>` or `ubatch:<sizes>`, and `time prompt`'s
//! `kind=` is the path the prompt ran — `step`, `pass` or `gemm`, never
//! `auto`. `--place` on any other file is refused by name.
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
        Body, Body35, Body38, Open35, PrefillPath, PrefillPlan, PrefillStep, Prompt38,
        Qwen35moeModel, Qwen38Model,
    };
    use bloomery_gpu::model::{ChainBody, MAX_PASS_ROWS, StepMode};
    use bloomery_gpu::{Gpu, GpuModel, Qwen3moeModel};
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use bloomery_levers::Levers;
    use cuda_core::sys;
    use gguf::Split;
    use model::arch::Arch;
    use model::arch::qwen35moe::place::{PlanInputs, machine};
    use model::placement::PlanLevers;
    use model::placement::workstation::{A6000, RTX_3090};
    use std::num::NonZeroUsize;
    use std::time::Instant;
    use tokenizer::Tokenizer;

    // A Qwen3.6 prompt is cut by the qwen3moe pass plan (`PrefillPlan`),
    // whose passes are `MAX_TOKENS` long; each must be a pass `step_rows`
    // takes.
    const _: () = assert!(MAX_TOKENS == MAX_PASS_ROWS);

    // A Qwen3.8 pass plan is printed by the same cut; its passes are
    // `Prompt38::PASS_ROWS` long.
    const _: () = assert!(Prompt38::PASS_ROWS == MAX_TOKENS);

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

    /// The engines this CLI drives, one a file architecture.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Family {
        Qwen3,
        Qwen35,
        /// `qwen4exp`, which the qwen35moe reader reads and `Arch` does not
        /// name.
        Qwen38,
    }

    /// The model file ([`ref_model_path`]) and its engine, read from its
    /// first shard's header: `qwen4exp` by its name, else `Arch::detect`'s
    /// qwen3moe or qwen35moe; any other file is refused by name.
    fn open_file() -> Result<(Split, Family), GateError> {
        let path = ref_model_path()?;
        let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        if file.architecture() == Some("qwen4exp") {
            return Ok((file, Family::Qwen38));
        }
        let first = file
            .shard(0)
            .ok_or_else(|| format!("{} opened with no shard", path.display()))?;
        match Arch::detect(first)? {
            Arch::Qwen3moe => Ok((file, Family::Qwen3)),
            Arch::Qwen35moe => Ok((file, Family::Qwen35)),
            other => Err(format!(
                "{} is a {} file; generate_qwen3moe runs qwen3moe, qwen35moe and qwen4exp files",
                path.display(),
                file.architecture().unwrap_or(other.name())
            )
            .into()),
        }
    }

    /// `--prefill` for the qwen3moe and qwen35moe bodies: `auto` (the
    /// default), `pass` or `gemm`.
    fn prefill_path(arg: Option<&str>) -> Result<PrefillPath, GateError> {
        match arg {
            None | Some("auto") => Ok(PrefillPath::Auto),
            Some("pass") => Ok(PrefillPath::Pass),
            Some("gemm") => Ok(PrefillPath::Gemm),
            Some(o) => Err(format!("--prefill is auto, pass or gemm, not {o}").into()),
        }
    }

    /// The units a prompt runs as, as the `step 0` and `time prompt` lines
    /// print them.
    struct Units {
        /// `plan=`.
        text: String,
        /// `prefill_steps=` and `passes=`.
        count: usize,
        /// `kind=`.
        kind: &'static str,
        /// The ubatches' ids.
        ubatch_tokens: usize,
    }

    impl From<&PrefillPlan> for Units {
        fn from(plan: &PrefillPlan) -> Units {
            Units {
                text: plan.to_string(),
                count: plan.steps.len(),
                kind: plan.kind(),
                ubatch_tokens: plan.ubatch_tokens(),
            }
        }
    }

    /// A body's prompt schedule as this CLI drives it, beside the
    /// skeleton's own step.
    trait Prompted: ChainBody {
        /// What `--prefill` selects.
        type Path: Copy;
        /// The path the body runs for `--prefill <arg>` (`None` when not
        /// given); a path it has not is refused by name, before the load.
        fn path(arg: Option<&str>) -> Result<Self::Path, GateError>;
        /// The units a prompt of `n` ids runs as by `path`.
        fn plan(m: &GpuModel<Self>, n: usize, path: Self::Path) -> Result<Units, GateError>;
        /// Run `ids` by that plan from the model's position; the argmax
        /// after the last.
        fn prefill(m: &mut GpuModel<Self>, ids: &[u32], path: Self::Path)
        -> Result<u32, GateError>;
        /// The last prompt image a ubatch of `units` wrote; `None` when none
        /// did.
        fn image(m: &GpuModel<Self>, units: &Units) -> Result<Option<ImageWrite>, GateError>;
        /// Fill the first `rows` cache rows with the body's synthetic
        /// pattern (`--seed-depth`).
        fn seed(m: &mut GpuModel<Self>, rows: usize) -> Result<(), GateError>;
    }

    impl Prompted for Body {
        type Path = PrefillPath;

        fn path(arg: Option<&str>) -> Result<PrefillPath, GateError> {
            prefill_path(arg)
        }

        fn plan(m: &Qwen3moeModel, n: usize, path: PrefillPath) -> Result<Units, GateError> {
            Ok(Units::from(&m.prefill_plan(n, path)?))
        }

        fn prefill(
            m: &mut Qwen3moeModel,
            ids: &[u32],
            path: PrefillPath,
        ) -> Result<u32, GateError> {
            Ok(m.prefill_with(ids, path)?)
        }

        fn image(m: &Qwen3moeModel, _: &Units) -> Result<Option<ImageWrite>, GateError> {
            Ok(m.ubatch_prologue()?.last)
        }

        fn seed(m: &mut Qwen3moeModel, rows: usize) -> Result<(), GateError> {
            Ok(m.seed_depth(rows)?)
        }
    }

    /// A Qwen3.6 prompt's plan by `path`: `pass` the captured passes,
    /// `auto` and `gemm` the prompt call's.
    fn plan35(m: &Qwen35moeModel, n: usize, path: PrefillPath) -> Result<PrefillPlan, GateError> {
        match path {
            PrefillPath::Pass => Ok(PrefillPlan::new(n, path, NonZeroUsize::MIN)),
            PrefillPath::Auto | PrefillPath::Gemm => Ok(m.prefill_plan(n, path)?),
        }
    }

    impl Prompted for Body35 {
        type Path = PrefillPath;

        /// Every path: `pass` the captured passes (`step_rows`), `auto` and
        /// `gemm` the prompt call.
        fn path(arg: Option<&str>) -> Result<PrefillPath, GateError> {
            prefill_path(arg)
        }

        fn plan(m: &Qwen35moeModel, n: usize, path: PrefillPath) -> Result<Units, GateError> {
            Ok(Units::from(&plan35(m, n, path)?))
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
            let plan = plan35(m, ids.len(), path)?;
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

        /// The prompt call's image when a ubatch of `units` ran, its
        /// `tokens` the ubatches' ids.
        fn image(m: &Qwen35moeModel, units: &Units) -> Result<Option<ImageWrite>, GateError> {
            let ub = units.ubatch_tokens;
            Ok(m.prompt_image()?
                .filter(|_| ub > 0)
                .map(|w| ImageWrite { tokens: ub, ..w }))
        }

        fn seed(m: &mut Qwen35moeModel, rows: usize) -> Result<(), GateError> {
            Ok(m.seed_depth(rows)?)
        }
    }

    impl Prompted for Body38 {
        type Path = Prompt38;

        /// `auto` (the default), `gemm`, `pass` or `step`
        /// (`Prompt38::parse`).
        fn path(arg: Option<&str>) -> Result<Prompt38, GateError> {
            Ok(arg.map_or(Ok(Prompt38::Auto), Prompt38::parse)?)
        }

        /// `step:1x<n>` — one captured step a position — the pass cut
        /// `pass:<sizes>`, or the ubatch cut `ubatch:<sizes>`; the kind is
        /// the path `auto` resolves to.
        fn plan(m: &Qwen38Model, n: usize, path: Prompt38) -> Result<Units, GateError> {
            let path = path.resolve(n);
            Ok(match path {
                Prompt38::Gemm | Prompt38::Auto => {
                    let rows = m.body("plan")?.ubatch_rows();
                    let plan = PrefillPlan {
                        steps: (0..n.div_ceil(rows))
                            .map(|i| PrefillStep::Ubatch(rows.min(n - i * rows)))
                            .collect(),
                    };
                    Units {
                        kind: path.name(),
                        ubatch_tokens: 0,
                        ..Units::from(&plan)
                    }
                }
                Prompt38::Step => Units {
                    text: if n == 1 {
                        "step:1".to_string()
                    } else {
                        format!("step:1x{n}")
                    },
                    count: n,
                    kind: path.name(),
                    ubatch_tokens: 0,
                },
                Prompt38::Pass => {
                    let plan = PrefillPlan::new(n, PrefillPath::Pass, NonZeroUsize::MIN);
                    Units {
                        kind: path.name(),
                        ..Units::from(&plan)
                    }
                }
            })
        }

        /// The prompt by `path`, then a `stat prompt host` line: the host
        /// tier's batch services the prompt took, their columns and host
        /// slots, and the union calls' wall — the host term of a pass's or a
        /// ubatch's layer.
        fn prefill(m: &mut Qwen38Model, ids: &[u32], path: Prompt38) -> Result<u32, GateError> {
            let before = m.body("prefill")?.hybrid().stats();
            let next = m.prompt38(ids, path)?;
            let after = m.body("prefill")?.hybrid().stats();
            let served = after.batch_served - before.batch_served;
            let ns = after.batch_ns - before.batch_ns;
            println!(
                "stat prompt host services={served} cols={} host_slots={} union_ms={:.3} \
                 per_service_ms={:.4}",
                after.batch_cols - before.batch_cols,
                after.batch_host_slots - before.batch_host_slots,
                ns as f64 / 1e6,
                if served == 0 {
                    0.0
                } else {
                    ns as f64 / 1e6 / served as f64
                }
            );
            Ok(next)
        }

        /// No ubatch runs: no image.
        fn image(_: &Qwen38Model, _: &Units) -> Result<Option<ImageWrite>, GateError> {
            Ok(None)
        }

        fn seed(_: &mut Qwen38Model, rows: usize) -> Result<(), GateError> {
            Err(no_seed38(rows + 1))
        }
    }

    /// The refusal of `--seed-depth D` on a qwen4exp file.
    fn no_seed38(depth: usize) -> GateError {
        format!(
            "--seed-depth {depth}: qwen4exp has no synthetic depth (its delta states and selection \
             pools are not a pattern a seed can stand for)"
        )
        .into()
    }

    /// Where `--place` puts a qwen4exp plan's one card.
    #[derive(Clone, Copy)]
    enum Place38 {
        /// The A6000, the timing card (the default).
        A,
        /// The 3090, the gate card.
        Gate,
    }

    impl Place38 {
        fn parse(arg: Option<&str>) -> Result<Place38, GateError> {
            match arg {
                None | Some("a") => Ok(Place38::A),
                Some("gate") => Ok(Place38::Gate),
                Some(o) => Err(format!("--place is a or gate, not {o}").into()),
            }
        }

        fn name(self) -> &'static str {
            match self {
                Place38::A => "a",
                Place38::Gate => "gate",
            }
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
        let levers = bloomery_levers::at_main(&[])?;
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
        let prefill = flag("--prefill")?;
        let place = flag("--place")?;
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
        let (file, family) = open_file()?;
        let prefill = prefill.as_deref();
        let chosen = match family {
            Family::Qwen3 => Chosen::Qwen3(Body::path(prefill)?),
            Family::Qwen35 => Chosen::Qwen35(Body35::path(prefill)?),
            Family::Qwen38 => {
                if let Some(d) = seed_depth {
                    return Err(no_seed38(d));
                }
                Chosen::Qwen38(Body38::path(prefill)?, Place38::parse(place.as_deref())?)
            }
        };
        if family != Family::Qwen38 && place.is_some() {
            return Err(
                "--place picks a qwen4exp plan's card; a qwen3moe or qwen35moe file runs on the \
                 card box.sh puts in view"
                    .into(),
            );
        }
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
        if matches!(chosen, Chosen::Qwen35(PrefillPath::Pass))
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
            seed_depth,
            logits,
            tok,
            ctx,
        };
        match chosen {
            Chosen::Qwen3(path) => drive(
                open_qwen3(file, ctx, mode, t)?,
                &run,
                path,
                &arms,
                listed,
                sync,
            ),
            Chosen::Qwen35(path) => drive(
                open_qwen35(file, ctx, mode, t)?,
                &run,
                path,
                &arms,
                listed,
                sync,
            ),
            Chosen::Qwen38(path, place) => {
                let m = open_qwen38(file, &levers, (ctx, mode), (path, place), t)?;
                drive(m, &run, path, &arms, listed, sync)
            }
        }
    }

    /// The engine and its `--prefill` path (and, for qwen4exp, its card),
    /// chosen before the load.
    #[derive(Clone, Copy)]
    enum Chosen {
        Qwen3(PrefillPath),
        Qwen35(PrefillPath),
        Qwen38(Prompt38, Place38),
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

    /// The Qwen3.8-Flash-Next model of `file`, placed by its plan on the
    /// card `place` names: the `plan` and `load` lines, and in graph mode the
    /// step captured and its `capture` line, its node kinds held to the
    /// program's count.
    fn open_qwen38(
        file: Split,
        levers: &Levers,
        (ctx, mode): (usize, StepMode),
        (path, place): (Prompt38, Place38),
        t: Instant,
    ) -> Result<Qwen38Model, GateError> {
        let inputs = PlanInputs::describe(&file)?;
        let card = match place {
            Place38::A => A6000,
            Place38::Gate => RTX_3090,
        };
        let machine = machine(card, inputs.spec.layers.len());
        let plan = inputs.plan(
            &machine,
            u64::try_from(ctx)?,
            &PlanLevers::from_levers(levers)?,
        )?;
        println!(
            "plan place={} card={} ctx_max={} host_experts={} card_experts={}",
            place.name(),
            card.name,
            plan.ctx_max,
            plan.host.experts,
            plan.cards[0].experts
        );
        let mut m = Body38::open_placed(file, &plan, &inputs, 0, levers.host())?;
        m.set_mode(mode);
        let body = m.body("generate_qwen3moe")?;
        println!(
            "load arch=qwen4exp resident_bytes={} ctx={ctx} layers={} mode={} store_bytes={} \
             prefill={} ubatch={} place={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            mode_name(mode),
            body.store_bytes(),
            path.name(),
            body.ubatch_rows(),
            place.name(),
            t.elapsed().as_secs_f64()
        );
        if mode == StepMode::Graph {
            let (launches, memops) = m.body("generate_qwen3moe")?.step_launches();
            let nodes = m.capture_step()?;
            let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
            let memop = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
            let ([k, b], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memop]);
            println!(
                "capture graph_nodes={nodes} kernel={k} batch_mem_op={b} other={other} (the \
                 program counts {launches}, {memops} of them batch_mem_op)"
            );
            if nodes != launches || b != memops || k + b != nodes || other != 0 {
                return Err(format!(
                    "the captured step is not the program's: {nodes} nodes ({k} kernel, {b} \
                     batch_mem_op, {other} other) against {launches} launches, {memops} of them \
                     batch_mem_op"
                )
                .into());
            }
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
        path: B::Path,
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
            run_arm(s.model_mut(), run, path, arm)
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
    fn run_arm<B: Prompted>(
        m: &mut GpuModel<B>,
        run: &Run,
        path: B::Path,
        arm: &Arm,
    ) -> Result<(), GateError> {
        let (ids, n_gen, warm) = (&arm.ids, arm.n_gen, run.warm);
        if let Some(d) = run.seed_depth.filter(|&d| d > 1) {
            B::seed(m, d - 1)?;
            println!("seed rows={} pos={}", d - 1, m.pos());
        }
        let depth = run.seed_depth.unwrap_or(ids.len());
        let plan = B::plan(m, ids.len(), path)?;
        let t = Instant::now();
        let mut next = B::prefill(m, ids, path)?;
        let prefill_wall = t.elapsed();
        let image = B::image(m, &plan)?;
        println!(
            "step 0 {} {next} (the {} prompt ids in prefill_steps={} units, plan={}, {:.2} s, \
             runtime value)",
            m.pos() - 1,
            ids.len(),
            plan.count,
            plan.text,
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
            plan.count,
            plan.kind
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
