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
//!     generate_qwen3moe --dump-taps DIR --tokens-file F [--tokens-file F ...]
//!                       --seqs S --prompt-len P [-n N] [--ctx C]
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
//! The ubatch size is `BLOOMERY_QWEN3_UBATCH` (1..=4096, default 4096),
//! clipped to the cache, read once: the plan counts that ubatch's arena and
//! the load runs it.
//!
//! A qwen4exp file's plan puts each layer's id prefix on the card as its
//! budget holds (`place::Experts::Card`, `BLOOMERY_QWEN38_EXPERTS` unset or
//! `card`), which the step's, the verify's and the pass's card leg and the
//! ubatch walk's card route run, or with `BLOOMERY_QWEN38_EXPERTS=host` every
//! routed expert on the host tier; a set `card` is refused on the other
//! families' files; the `plan` line prints `experts=` and the `load` line `card_layers=`.
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
//! `--prompt` is tokenized with the file's pre-tokenizer (`qwen35`).
//!
//! A qwen4exp file runs through `Body38`, placed by its own plan
//! (`model::arch::qwen35moe::place`: every layer, the head and the embedding
//! on one card, its routed experts as above) on the card `--place`
//! names — `a` the A6000 (the default), `gate` the 3090 — and printed as a
//! `plan` line. Its `--prefill` is `auto` (the default: `gemm` for a prompt
//! of nine positions or more, `pass` below), `gemm` (ubatches of up to the
//! load's size through the host tier's batch port at their width, the Q8_0
//! projections on q8 activations), `pass` (eager passes of up to eight
//! positions through the same port) or `step` (one captured step a
//! position); `--seed-depth` is refused by name (no synthetic depth). Its
//! `load` line names `store_bytes=`, `prefill=`, `ubatch=` and `place=`; in
//! graph mode its
//! `capture` line counts the step graph's nodes by kind against the
//! program's count, and a mismatch ends the run by name. Its plan prints as
//! `step:1x<P>`, `pass:<sizes>` or `ubatch:<sizes>`, and `time prompt`'s
//! `kind=` is the path the prompt ran — `step`, `pass` or `gemm`, never
//! `auto`. `--place` on any other file is refused by name.
//!
//! `BLOOMERY_ROUTE_TRACE=<dir>` (a qwen4exp file only, refused by name on
//! the others) writes the engine's route trace of the run into `dir`, a new
//! directory made before the load (`crates/gpu/src/host/route_trace.rs`):
//! every position's routed ids per layer and the slot each ran in, as a
//! router set. The prompt's ids run one step each and are recorded as the
//! call's positions — `--prefill step` is required, `--prefill pass` and
//! `gemm` (and the unset `auto`) refused by name, and so is `--time` (the
//! trace rewrites its manifest after every position, so a timed run's
//! numbers are not a measurement). A `--arm` list writes one `call` row an
//! arm under one set, and the set's `chunk` header line names the positions
//! every arm writes when they all write the same count. The run ends with
//! `route trace <dir> positions=<n> complete` once the set is sealed.
//!
//! `BLOOMERY_RESIDENCY` resolves first thing, unset to `off`, and prints as a
//! `residency lever` record on every run. Set to `mid-p<P>-s<S>` on a
//! qwen4exp file (plain or drafted, either `--place`), the load runs the
//! common residency machine over the card's routed stacks
//! (`Body38::open_placed_residency`): a `residency host` record follows the
//! `plan` line, each arm prints its boundaries' `residency pass` records
//! after its other lines — after its error, when it failed — and each arm
//! after the first opens with the `residency reset` record of the clear
//! before it. `mid-…` is
//! refused by name on a qwen3moe or qwen35moe file, under `--dump-taps` and
//! beside `BLOOMERY_ROUTE_TRACE`, and by the body at a step-fed prompt.
//!
//! Under `BLOOMERY_DRAFT=mtp` (a qwen4exp file only; every other family and
//! word is refused by name) the decode runs through the runtime's
//! speculative loop with the file's MTP draft (`app::arch::qwen4exp`'s
//! `MtpDraft` over the shared draft file beside the target or
//! `BLOOMERY_MTP_DRAFT`'s, its head reduced under `BLOOMERY_MTP_HEAD_ROWS`):
//! windows of four rows — the target's
//! verify of the draft's three ids, the kept rows committed, the draft's
//! next chain one readback — and the greedy ids are the plain run's. A
//! `load draft=mtp` line follows the `load` line (the draft's resident
//! bytes, its program's arena and its head), the `capture` line the verify
//! passes' widths; each pass prints its `step` lines one kept token a line
//! and, under `--time`, a `time pass` row (its wall, positions and kept
//! rows) and a `time step` row a kept position (its pass's wall over its
//! positions, the row a plain run's step wall compares with), and an
//! `mtp summary` record closes the arm: the windows' kept lengths, the
//! positions and their rate. A sampling request is a server matter; this
//! CLI is greedy.
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
//! reads the engine takes on every prompt; a body that writes no image, or a
//! prompt no ubatch ran, prints `ubatch_tokens=<n> (no prompt image)`),
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
//! `BLOOMERY_STEP_STATS=1` on a qwen4exp file reads, before the first
//! generated step and after each one, the host tier's counters
//! (`HybridStats`), the process's page faults (`/proc/self/stat`) and the
//! card's free device bytes, and prints, after the arm's other lines, one
//! `stat step` record a step (`record::STAT_STEP_HOST`: `served` layers,
//! their summed `leg_us` and `host_slots`) and a `stat summary` over the
//! steps past `--warm`. It also arms the ubatch walk's timing: after a
//! prompt's `stat prompt host` line, one `stat prompt38 lb` record a
//! layer-batch its ubatches served and a `stat prompt38 split` over them
//! (the serve's wait and union beside the card time of every part). Unset,
//! or on another file, nothing is read and nothing recorded.
//!
//! `--dump-taps DIR` (a qwen3moe file only) writes the layer-tap dump of
//! `shared/qwen3moe_taps.rs` into DIR, an empty or new directory: from each
//! `--tokens-file` (one id a line) `S` prompts of `P` ids, windows spread
//! over the file, each run from a reset one eager step per id with the layer
//! taps on, then `N` greedy steps. The steps are the decode path the pass
//! prefill equals bit for bit, so a sequence's ids are what `--tokens <its
//! prompt> -n N --prefill pass --ctx C` prints; the GEMM ubatches, which `auto`
//! takes for a prompt of more than eight ids, have no taps. It prints the
//! `load` line, then a `taps seq` record per sequence and a `taps dump`
//! record (`record::GENERATE_QWEN3MOE`), and mixes with no flag but `-n` and
//! `--ctx`.
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
#[path = "shared/qwen3moe_taps.rs"]
mod taps;

#[cfg(feature = "gpu")]
mod cli {
    use super::taps;
    use app::Session;
    use app::arch::qwen3moe::Q38Cfg;
    use app::mtp::MtpDraft;
    use bloomery_gpu::arch::qwen3moe::router::MAX_TOKENS;
    use bloomery_gpu::arch::qwen3moe::ubatch::{ImageWrite, ubatch_for, ubatch_size};
    use bloomery_gpu::arch::qwen3moe::{
        Body, Body35, Body38, Open35, PrefillPath, PrefillPlan, PrefillStep, Prompt38,
        Qwen35moeModel, Qwen38Model,
    };
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::route_trace::{RouteTrace, TraceHeader};
    use bloomery_gpu::host::swap::{PassReport, Residency};
    use bloomery_gpu::hybrid::HybridStats;
    use bloomery_gpu::model::{ChainBody, MAX_PASS_ROWS, StepMode};
    use bloomery_gpu::{Gpu, GpuModel, Qwen3moeModel};
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::record::{self, Record};
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use bloomery_levers::{Levers, ResidencyAt, ResidencyPick};
    use cuda_core::sys;
    use gguf::Split;
    use model::arch::Arch;
    use model::arch::models::HeadRows;
    use model::arch::qwen35moe::place::{
        Experts, MtpInputs, PlanInputs, machine_for_experts, read_head_rows,
    };
    use model::placement::churn::ChurnPool;
    use model::placement::workstation::{A6000, RTX_3090};
    use model::placement::{Plan, PlanLevers};
    use refset::arch::qwen4exp::mtp::draft_file;
    use runtime::{Advance, Committed, PassSink, Speculative, Stop, Target};
    use std::num::NonZeroUsize;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};
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
        /// The host tier's counters since load, for `BLOOMERY_STEP_STATS`;
        /// `None` for a body with no host tier.
        fn host_stats(_: &GpuModel<Self>) -> Result<Option<HybridStats>, GateError> {
            Ok(None)
        }

        /// Mark a prompt call of `n` ids from cache position `pos` on the
        /// host tier's route trace (`Hybrid::route_prompt`), before the ids
        /// run; nothing for a body with no host tier.
        fn mark_prompt(_m: &mut GpuModel<Self>, _pos: u32, _n: usize) -> Result<(), GateError> {
            Ok(())
        }

        /// Take the route trace the body's host tier holds, if one, finish
        /// it and return its directory and the positions it wrote; `None`
        /// with no trace.
        fn finish_trace(_m: &mut GpuModel<Self>) -> Result<Option<(PathBuf, u64)>, GateError> {
            Ok(None)
        }

        /// The residency boundaries' reports since the last take, each with
        /// the kind of the pass it ended; none for a body with no residency
        /// machine.
        fn residency_passes(
            _m: &mut GpuModel<Self>,
        ) -> Result<Vec<(PassKind, PassReport)>, GateError> {
            Ok(Vec::new())
        }
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

    /// The qwen4exp prompt call's own lines (the `Prompted` impl's and the
    /// drafted path's): the host tier's batch services over the call, and
    /// under `BLOOMERY_STEP_STATS` the ubatch walk's records.
    fn after38_prompt(m: &mut Qwen38Model, before: HybridStats) -> Result<(), GateError> {
        let after = m.body("prefill")?.hybrid().stats();
        let served = after.batch_served - before.batch_served;
        let ns = after.batch_ns - before.batch_ns;
        // No service, no time a service: `-`, never a plausible 0.
        let per_service = if served == 0 {
            "-".to_string()
        } else {
            format!("{:.4}", ns as f64 / 1e6 / served as f64)
        };
        println!(
            "stat prompt host services={served} cols={} host_slots={} union_ms={:.3} \
         per_service_ms={per_service}",
            after.batch_cols - before.batch_cols,
            after.batch_host_slots - before.batch_host_slots,
            ns as f64 / 1e6,
        );
        if let Some(s) = m.take_prompt38_stats()? {
            let ms = |ns: u64| ns as f64 / 1e6;
            let lbs = s.rows.len() as u64;
            let (mut union, mut wait, mut serve, mut enqueue, mut slots) =
                (0u64, 0u64, 0u64, 0u64, 0u64);
            let (mut experts, mut m_hot, mut cols_hot, mut m_sq) = (0u64, 0u64, 0u64, 0u64);
            let mut m_max = 0usize;
            let (mut front, mut down, mut shadow, mut up, mut back) = (0.0_f64, 0.0, 0.0, 0.0, 0.0);
            for r in &s.rows {
                union += r.union_ns;
                wait += r.wait_ns;
                serve += r.serve_ns;
                enqueue += r.enqueue_ns;
                slots += r.slots;
                experts += r.experts as u64;
                m_max = m_max.max(r.m_max);
                m_hot += r.m_hot as u64;
                cols_hot += r.cols_hot as u64;
                m_sq += r.m_sq;
                front += r.front_ms;
                down += r.down_ms;
                shadow += r.shadow_ms;
                up += r.upload_ms;
                back += r.back_ms;
                Record::new(&record::STAT_PROMPT38_LB)
                    .u("b", r.b)
                    .u("layer", r.layer as u64)
                    .u("cols", r.cols as u64)
                    .u("slots", r.slots)
                    .u("experts", r.experts as u64)
                    .u("m_max", r.m_max as u64)
                    .u("m_hot", r.m_hot as u64)
                    .u("cols_hot", r.cols_hot as u64)
                    .u("m_sq", r.m_sq)
                    .f("wait_ms", ms(r.wait_ns))
                    .f("union_ms", ms(r.union_ns))
                    .f("serve_ms", ms(r.serve_ns))
                    .f("enqueue_ms", ms(r.enqueue_ns))
                    .f("card_front_ms", r.front_ms)
                    .f("card_down_ms", r.down_ms)
                    .f("card_shadow_ms", r.shadow_ms)
                    .f("card_upload_ms", r.upload_ms)
                    .f("card_back_ms", r.back_ms)
                    .print();
            }
            let per = |v: f64| if lbs == 0 { 0.0 } else { v / lbs as f64 };
            Record::new(&record::STAT_PROMPT38_SPLIT)
                .u("ubatches", s.ubatches)
                .u("layer_batches", lbs)
                .f("prologue_ms", ms(s.prologue_ns))
                .f("walk_ms", ms(s.walk_ns))
                .f("union_ms", ms(union))
                .f("wait_ms", ms(wait))
                .f("serve_ms", ms(serve))
                .f("enqueue_ms", ms(s.walk_ns.saturating_sub(serve)))
                .u("host_slots", slots)
                .f("union_lb", per(ms(union)))
                .f("wait_lb", per(ms(wait)))
                .f("serve_lb", per(ms(serve)))
                .f("enqueue_lb", per(ms(enqueue)))
                .f("slots_lb", per(slots as f64))
                .f("experts_lb", per(experts as f64))
                .u("m_max", m_max as u64)
                .f("m_hot_lb", per(m_hot as f64))
                .f("cols_hot_lb", per(cols_hot as f64))
                .f("m_sq_lb", per(m_sq as f64))
                .f("card_front_ms", front)
                .f("card_down_ms", down)
                .f("card_shadow_ms", shadow)
                .f("card_upload_ms", up)
                .f("card_back_ms", back)
                .f("card_front_lb", per(front))
                .f("card_down_lb", per(down))
                .f("card_shadow_lb", per(shadow))
                .f("card_upload_lb", per(up))
                .f("card_back_lb", per(back))
                .print();
        }
        Ok(())
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
                Prompt38::Gemm => {
                    let plan = PrefillPlan {
                        steps: m
                            .body("plan")?
                            .ubatch_cut(n)
                            .map(PrefillStep::Ubatch)
                            .collect(),
                    };
                    Units {
                        kind: path.name(),
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
                Prompt38::Auto => {
                    return Err(format!(
                        "prompt path auto at {n} positions: `Prompt38::resolve` returned `auto`"
                    )
                    .into());
                }
            })
        }

        /// The prompt by `path`, then a `stat prompt host` line: the host
        /// tier's batch services the prompt took, their columns and host
        /// slots, and the union calls' wall — the host term of a pass's or a
        /// ubatch's layer. With the ubatch walk's timing armed
        /// (`BLOOMERY_STEP_STATS`), the prompt's `stat prompt38` records
        /// follow: one `stat prompt38 lb` a layer-batch the ubatches served,
        /// then the `stat prompt38 split` over them.
        fn prefill(m: &mut Qwen38Model, ids: &[u32], path: Prompt38) -> Result<u32, GateError> {
            let before = m.body("prefill")?.hybrid().stats();
            let next = m.prompt38(ids, path)?;
            after38_prompt(m, before)?;
            Ok(next)
        }

        /// The Qwen3.8 walks write no prompt image.
        fn image(_: &Qwen38Model, _: &Units) -> Result<Option<ImageWrite>, GateError> {
            Ok(None)
        }

        fn seed(_: &mut Qwen38Model, rows: usize) -> Result<(), GateError> {
            Err(no_seed38(rows + 1))
        }

        fn host_stats(m: &Qwen38Model) -> Result<Option<HybridStats>, GateError> {
            Ok(Some(m.body("generate_qwen3moe")?.hybrid().stats()))
        }

        /// The trace's `call` row: the arm's prompt ids one step each from
        /// `pos` ([`Hybrid::route_prompt`]).
        fn mark_prompt(m: &mut Qwen38Model, pos: u32, n: usize) -> Result<(), GateError> {
            Ok(m.body_parts("generate_qwen3moe")?
                .2
                .hybrid_mut()
                .route_prompt(pos, n)?)
        }

        /// [`Hybrid::take_route_trace`], the set sealed by
        /// [`RouteTrace::finish`].
        fn finish_trace(m: &mut Qwen38Model) -> Result<Option<(PathBuf, u64)>, GateError> {
            let Some(t) = m
                .body_parts("generate_qwen3moe")?
                .2
                .hybrid_mut()
                .take_route_trace()
            else {
                return Ok(None);
            };
            let dir = t.dir().to_path_buf();
            Ok(Some((dir, t.finish()?)))
        }

        /// [`Body38::take_residency_passes`]: empty unless the load runs the
        /// machine and [`log38`] asked for the log.
        fn residency_passes(m: &mut Qwen38Model) -> Result<Vec<(PassKind, PassReport)>, GateError> {
            Ok(m.body_parts("generate_qwen3moe")?.2.take_residency_passes())
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
        /// `BLOOMERY_STEP_STATS`, as the levers hold it.
        stats: bool,
    }

    /// One arm: its ids and its generated count.
    struct Arm {
        ids: Vec<u32>,
        n_gen: usize,
    }

    pub fn run() -> Result<(), GateError> {
        // A lever set to a value it does not take, or a retired name that is
        // set, is refused by name before anything loads.
        let levers = bloomery_levers::at_main(&[
            bloomery_levers::STEP_STATS,
            bloomery_levers::QWEN38_EXPERTS,
            bloomery_levers::ROUTE_TRACE,
            bloomery_levers::DRAFT,
            bloomery_levers::MTP_HEAD_ROWS,
            bloomery_levers::MTP_DRAFT,
            bloomery_levers::RESIDENCY,
        ])?;
        record::at_main("generate_qwen3moe", record::GENERATE_QWEN3MOE);
        // Unset is `off` here: the machine runs over a qwen4exp load only when
        // the lever asks for it.
        let pick = levers.residency_at(ResidencyAt::FIXED);
        record::residency_lever(pick).print();
        let residency = Residency::parse(pick.word)?;
        let experts = experts38(&levers)?;
        // Unset, the lever is a qwen4exp plan's card experts and nothing on
        // another family's file; only a set `card` is refused there.
        let card_set = levers.qwen38_experts_set() == Some("card");
        if let Some(dir) = flag("--dump-taps")? {
            if card_set {
                return Err(
                    "BLOOMERY_QWEN38_EXPERTS=card places a qwen4exp plan's routed experts; \
                     --dump-taps runs a qwen3moe file"
                        .into(),
                );
            }
            if levers.route_trace().is_some() {
                return Err(
                    "BLOOMERY_ROUTE_TRACE records a qwen4exp host tier's steps; --dump-taps runs \
                     a qwen3moe file"
                        .into(),
                );
            }
            if levers.mtp_draft().is_some() {
                return Err(
                    "BLOOMERY_MTP_DRAFT names a qwen4exp file's MTP draft; --dump-taps runs a \
                     qwen3moe file"
                        .into(),
                );
            }
            if residency != Residency::Off {
                return Err(format!(
                    "BLOOMERY_RESIDENCY={} moves a qwen4exp plan's card experts; --dump-taps runs \
                     a qwen3moe file",
                    pick.word
                )
                .into());
            }
            return dump_taps(&dir);
        }
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
        let draft = match family {
            Family::Qwen38 => draft38(&levers)?,
            other => {
                draft_refused_on_other(&levers, other)?;
                Draft38::Off
            }
        };
        let chosen = match family {
            Family::Qwen3 => Chosen::Qwen3(Body::path(prefill)?),
            Family::Qwen35 => Chosen::Qwen35(Body35::path(prefill)?),
            Family::Qwen38 => {
                if let Some(d) = seed_depth {
                    return Err(no_seed38(d));
                }
                Chosen::Qwen38(
                    Body38::path(prefill)?,
                    Place38::parse(place.as_deref())?,
                    draft,
                )
            }
        };
        if family != Family::Qwen38 && place.is_some() {
            return Err(
                "--place picks a qwen4exp plan's card; a qwen3moe or qwen35moe file runs on the \
                 card box.sh puts in view"
                    .into(),
            );
        }
        if family != Family::Qwen38 && card_set {
            return Err(
                "BLOOMERY_QWEN38_EXPERTS=card places a qwen4exp plan's routed experts; a \
                 qwen3moe or qwen35moe file has no host tier"
                    .into(),
            );
        }
        if family != Family::Qwen38 && levers.route_trace().is_some() {
            return Err(
                "BLOOMERY_ROUTE_TRACE records a qwen4exp host tier's routing; a qwen3moe or \
                 qwen35moe plan holds every one on the card"
                    .into(),
            );
        }
        if family != Family::Qwen38 && residency != Residency::Off {
            return Err(format!(
                "BLOOMERY_RESIDENCY={} moves a qwen4exp plan's card experts; a qwen3moe or \
                 qwen35moe plan holds every one on the card",
                pick.word
            )
            .into());
        }
        // The trace is the input the residency model replays under a fixed
        // seed; under the machine its slot files would record the machine's
        // own moves.
        if residency != Residency::Off && levers.route_trace().is_some() {
            return Err(format!(
                "BLOOMERY_ROUTE_TRACE records a fixed placement's routing; \
                 BLOOMERY_RESIDENCY={} moves the slot map under it",
                pick.word
            )
            .into());
        }
        if family == Family::Qwen38 && draft == Draft38::Off && levers.mtp_head_rows().is_some() {
            return Err(
                "BLOOMERY_MTP_HEAD_ROWS reduces the MTP draft's head; it needs BLOOMERY_DRAFT=mtp \
                 on a qwen4exp file"
                    .into(),
            );
        }
        if draft == Draft38::Off && levers.mtp_draft().is_some() {
            return Err(
                "BLOOMERY_MTP_DRAFT names the MTP draft file; it needs BLOOMERY_DRAFT=mtp on a \
                 qwen4exp file"
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
        if matches!(chosen, Chosen::Qwen38(.., Draft38::Mtp)) && logits {
            return Err(
                "--logits with BLOOMERY_DRAFT=mtp: the drafted run's last call is a verify of \
                 several rows into one head, and --logits reads the step head's row"
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
            stats: levers.step_stats(),
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
            Chosen::Qwen38(path, place, draft) => match draft {
                Draft38::Off => {
                    let trace = trace38(
                        &levers,
                        &file,
                        (path, place, experts),
                        arms_chunk(&arms),
                        timed,
                    )?;
                    let mut m = open_qwen38(
                        file,
                        &levers,
                        (ctx, mode),
                        (path, place, experts),
                        (residency, pick),
                        t,
                    )?;
                    log38(&mut m, residency, &arms)?;
                    if let Some(t) = trace {
                        m.body_parts("generate_qwen3moe")?
                            .2
                            .hybrid_mut()
                            .attach_route_trace(t)?;
                    }
                    m.set_prompt38_stats(run.stats)?;
                    drive(m, &run, path, &arms, listed, sync)
                }
                Draft38::Mtp => {
                    if levers.route_trace().is_some() {
                        return Err(
                            "BLOOMERY_ROUTE_TRACE records the plain run's routing, one step a \
                                    position; it is refused beside BLOOMERY_DRAFT=mtp"
                                .into(),
                        );
                    }
                    let (mut m, cfg) = open_qwen38_mtp(
                        file,
                        &levers,
                        (ctx, mode),
                        (path, place, experts),
                        (residency, pick),
                        t,
                    )?;
                    log38(&mut m, residency, &arms)?;
                    m.set_prompt38_stats(run.stats)?;
                    drive38_mtp(m, cfg, &run, &arms, listed, sync)
                }
            },
        }
    }

    /// The flags `--dump-taps` refuses by name: every other mode's.
    const NOT_WITH_DUMP: [&str; 11] = [
        "--prompt",
        "--tokens",
        "--seed-depth",
        "--arm",
        "--arm-sync",
        "--time",
        "--warm",
        "--logits",
        "--prefill",
        "--mode",
        "--place",
    ];

    /// Whether `BLOOMERY_DRAFT` drafts this run: `mtp` on a qwen4exp file;
    /// every other family and word is refused by name.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Draft38 {
        Off,
        Mtp,
    }

    /// `BLOOMERY_DRAFT` on a qwen4exp file: unset the plain path, `mtp` the
    /// MTP draft; the V4.1 words and any other are refused by name.
    fn draft38(levers: &Levers) -> Result<Draft38, GateError> {
        match levers.draft() {
            None => Ok(Draft38::Off),
            Some("mtp") => Ok(Draft38::Mtp),
            Some(other) => Err(format!(
                "BLOOMERY_DRAFT={other}: on a qwen4exp file mtp drafts the window; lookup and \
                 dspark are the V4.1 binaries'"
            )
            .into()),
        }
    }

    /// `BLOOMERY_DRAFT` on the other families: no word of it drafts them.
    fn draft_refused_on_other(levers: &Levers, family: Family) -> Result<(), GateError> {
        match levers.draft() {
            None => Ok(()),
            Some(word) => Err(format!(
                "BLOOMERY_DRAFT={word}: only a qwen4exp file runs a draft (mtp); this is a {} \
                 file",
                match family {
                    Family::Qwen3 => "qwen3moe",
                    Family::Qwen35 => "qwen35moe",
                    Family::Qwen38 => "qwen4exp",
                }
            )
            .into()),
        }
    }

    /// `--dump-taps DIR`: the tap dump of every `--tokens-file`'s windows on
    /// a qwen3moe file, eager, then the manifest read back.
    fn dump_taps(dir: &str) -> Result<(), GateError> {
        let args: Vec<String> = std::env::args().collect();
        if let Some(f) = NOT_WITH_DUMP.iter().find(|f| args.iter().any(|a| a == *f)) {
            return Err(format!(
                "--dump-taps runs each prompt one eager step per id; {f} does not apply to it"
            )
            .into());
        }
        let files = flags("--tokens-file")?;
        if files.is_empty() {
            return Err("--dump-taps needs at least one --tokens-file".into());
        }
        let seqs: usize = flag("--seqs")?
            .ok_or("--dump-taps needs --seqs (prompts per --tokens-file)")?
            .parse()?;
        let len: usize = flag("--prompt-len")?
            .ok_or("--dump-taps needs --prompt-len")?
            .parse()?;
        let n_gen: usize = flag("-n")?.map_or(Ok(32), |s| s.parse())?;
        let ctx: usize = flag("--ctx")?.map_or(Ok(4096), |s| s.parse())?;
        if len + n_gen > ctx {
            return Err(format!("--prompt-len {len} + -n {n_gen} pass --ctx {ctx}").into());
        }
        let mut all = Vec::new();
        for f in &files {
            all.extend(taps::windows(Path::new(f), seqs, len)?);
        }
        let t = Instant::now();
        let (file, family) = open_file()?;
        if family != Family::Qwen3 {
            return Err(format!(
                "--dump-taps runs a qwen3moe file (the taps are its chain's), not a {}",
                file.architecture().unwrap_or("?")
            )
            .into());
        }
        let mut m = open_qwen3(file, ctx, StepMode::Eager, t)?;
        let hidden = m.body("generate_qwen3moe")?.hparams().n_embd;
        let out = Path::new(dir);
        let mut dump = taps::Dump::create(out, &ref_model_path()?, hidden)?;
        let t = Instant::now();
        for s in &all {
            let t_seq = Instant::now();
            let row = dump.seq(&mut m, s, n_gen)?;
            let (ids_b, taps_b) = taps::sizes(row, hidden);
            println!(
                "{}",
                Record::new(&record::TAPS_SEQ)
                    .u("k", row.seq)
                    .w("source", s.source.display())
                    .u("offset", s.offset)
                    .u("n_prompt", row.n_prompt)
                    .u("n_total", row.n_total)
                    .u("bytes", ids_b + taps_b)
                    .f("wall_s", t_seq.elapsed().as_secs_f64())
                    .line()
            );
        }
        let rows = taps::read_manifest(out, hidden)?;
        if rows.len() != all.len() {
            return Err(format!(
                "{}: the manifest reads back {} rows of {} sequences",
                out.display(),
                rows.len(),
                all.len()
            )
            .into());
        }
        let bytes: u64 = rows
            .iter()
            .map(|r| {
                let (a, b) = taps::sizes(r, hidden);
                a + b
            })
            .sum();
        println!(
            "{}",
            Record::new(&record::TAPS_DUMP)
                .w("dir", out.display())
                .u("seqs", rows.len())
                .u("positions", rows.iter().map(|r| r.n_total).sum::<usize>())
                .u("bytes", bytes)
                .csv("layers", taps::TAPS)
                .w("prefill", "step")
                .f("wall_s", t.elapsed().as_secs_f64())
                .line()
        );
        Ok(())
    }

    /// The engine and its `--prefill` path (and, for qwen4exp, its card and
    /// draft), chosen before the load; a qwen4exp plan's expert rule is
    /// `BLOOMERY_QWEN38_EXPERTS`'s.
    #[derive(Clone, Copy)]
    enum Chosen {
        Qwen3(PrefillPath),
        Qwen35(PrefillPath),
        Qwen38(Prompt38, Place38, Draft38),
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

    /// `BLOOMERY_QWEN38_EXPERTS` as a qwen4exp plan's expert rule: as set,
    /// else the card experts.
    fn experts38(levers: &Levers) -> Result<Experts, GateError> {
        match levers.qwen38_experts() {
            "host" => Ok(Experts::Host),
            "card" => Ok(Experts::Card),
            other => Err(format!("BLOOMERY_QWEN38_EXPERTS={other}: host or card").into()),
        }
    }

    /// The name the `plan` line prints for `experts`.
    fn experts_name(experts: Experts) -> &'static str {
        match experts {
            Experts::Host => "host",
            Experts::Card => "card",
        }
    }

    /// The positions every arm of the run writes — its prompt's ids one
    /// step each, then its generated tokens less the first, which the
    /// prompt's last step already answered — when they all write the same
    /// count, for the trace's `chunk` header line; `None` when they differ
    /// (the line names one context shape, so a run of unequal arms writes
    /// none and the replay tool refuses the set's unknown contexts by name).
    fn arms_chunk(arms: &[Arm]) -> Option<usize> {
        let of = |a: &Arm| a.ids.len() + a.n_gen - 1;
        let first = of(arms.first()?);
        arms.iter().all(|a| of(a) == first).then_some(first)
    }

    /// The route trace `BLOOMERY_ROUTE_TRACE` asks for on a qwen4exp file,
    /// its directory made here, before the load: every position's routed
    /// ids per layer and the slot each ran in
    /// (`crates/gpu/src/host/route_trace.rs`), the prompt's ids one step
    /// each recorded as the call's positions. Refused by name under a
    /// prompt path but `step` — a pass runs a multi-row service and a
    /// ubatch a prompt batch, and the trace records one-row steps — and
    /// beside `--time`: the trace rewrites its manifest after every
    /// position, so a timed run's numbers would not be a measurement.
    fn trace38(
        levers: &Levers,
        file: &Split,
        (path, place, experts): (Prompt38, Place38, Experts),
        chunk: Option<usize>,
        timed: bool,
    ) -> Result<Option<RouteTrace>, GateError> {
        let Some(dir) = levers.route_trace() else {
            return Ok(None);
        };
        if path != Prompt38::Step {
            return Err(
                "BLOOMERY_ROUTE_TRACE records one-row steps: pass --prefill step (a pass runs a \
                 multi-row service and a ubatch a prompt batch, which the trace does not record)"
                    .into(),
            );
        }
        if timed {
            return Err(
                "BLOOMERY_ROUTE_TRACE rewrites its manifest after every position, so --time \
                 beside it is not a measurement: run the trace without --time"
                    .into(),
            );
        }
        let inputs = PlanInputs::describe(file)?;
        let hp = &inputs.hp;
        let mut extra = vec![
            ("place".to_owned(), place.name().to_owned()),
            ("experts".to_owned(), experts_name(experts).to_owned()),
            ("prefill".to_owned(), path.name().to_owned()),
        ];
        if let Some(n) = chunk {
            extra.push(("chunk".to_owned(), n.to_string()));
        }
        let header = TraceHeader {
            model: ref_model_path()?,
            arch: "qwen4exp".to_owned(),
            build: "generate_qwen3moe".to_owned(),
            n_expert: hp.n_expert,
            n_used: hp.n_used,
            first_layer: 0,
            n_layer: inputs.spec.layers.len(),
            extra,
        };
        Ok(Some(RouteTrace::create(dir, header)?))
    }

    /// The Qwen3.8-Flash-Next model of `file`, placed by its plan on the
    /// card `place` names, its routed experts where `experts` says, under
    /// `residency` (the lever's word `pick`): the `plan` line, under `mid`
    /// the `residency host` line, the `load` line, and in graph mode the step
    /// captured and its `capture` line, its node kinds held to the program's
    /// count.
    fn open_qwen38(
        file: Split,
        levers: &Levers,
        (ctx, mode): (usize, StepMode),
        (path, place, experts): (Prompt38, Place38, Experts),
        (residency, pick): (Residency, ResidencyPick),
        t: Instant,
    ) -> Result<Qwen38Model, GateError> {
        let inputs = PlanInputs::describe(&file)?;
        let card = match place {
            Place38::A => A6000,
            Place38::Gate => RTX_3090,
        };
        let ub = ubatch_for(ctx)?;
        let machine =
            machine_for_experts(card, inputs.spec.layers.len(), u64::try_from(ub)?, experts);
        let plan = inputs.plan_with(
            &machine,
            u64::try_from(ctx)?,
            &PlanLevers::from_levers(levers)?,
            experts,
        )?;
        println!(
            "{}",
            Record::new(&record::PLAN38)
                .w("place", place.name())
                .w("card", card.name)
                .w("experts", experts_name(experts))
                .u("ctx_max", plan.ctx_max)
                .u("host_experts", plan.host.experts)
                .u("card_experts", plan.cards[0].experts)
                .line()
        );
        residency_host38(&plan, residency, pick)?;
        let mut m = Body38::open_placed_residency(
            file,
            &plan,
            &inputs,
            CARD38,
            levers.host(),
            ub,
            residency,
        )?;
        m.set_mode(mode);
        let body = m.body("generate_qwen3moe")?;
        println!(
            "load arch=qwen4exp resident_bytes={} ctx={ctx} layers={} mode={} store_bytes={} \
             prefill={} ubatch={} place={} card_layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            mode_name(mode),
            body.store_bytes(),
            path.name(),
            body.ubatch_rows(),
            place.name(),
            body.card_layers(),
            t.elapsed().as_secs_f64()
        );
        capture38_check(&mut m)?;
        Ok(m)
    }

    /// The Qwen3.8-Flash-Next model of `file` with its MTP draft loaded
    /// beside it (`Body38::open_placed_mtp_residency`, the draft file `draft_file`
    /// picks, its head reduced under `BLOOMERY_MTP_HEAD_ROWS`): the `plan`,
    /// `residency host`, `load`, `load draft=mtp` and `capture` lines of the
    /// plain open, the draft's own
    /// resident bytes, its program's arena and its head named on the draft's
    /// line. In graph mode the verify passes of 2 to 4 rows are captured by
    /// the session's `with_draft`, whose log prints each width's nodes.
    fn open_qwen38_mtp(
        file: Split,
        levers: &Levers,
        (ctx, mode): (usize, StepMode),
        (path, place, experts): (Prompt38, Place38, Experts),
        (residency, pick): (Residency, ResidencyPick),
        t: Instant,
    ) -> Result<(Qwen38Model, Q38Cfg), GateError> {
        let inputs = PlanInputs::describe(&file)?;
        let card = match place {
            Place38::A => A6000,
            Place38::Gate => RTX_3090,
        };
        let ub = ubatch_for(ctx)?;
        let machine =
            machine_for_experts(card, inputs.spec.layers.len(), u64::try_from(ub)?, experts);
        let rows = match levers.mtp_head_rows() {
            Some(p) => read_head_rows(p, &file, inputs.spec.vocab)?,
            None => HeadRows::Full,
        };
        let (draft_path, from) = draft_file(levers.mtp_draft(), &ref_model_path()?);
        let draft_split = Split::open(&draft_path).map_err(|e| {
            format!(
                "open the MTP draft {} ({}): {e}",
                draft_path.display(),
                from.describe()
            )
        })?;
        let mtp = MtpInputs::read(&draft_split, &file, &inputs, rows)?;
        let plan = inputs.plan_mtp_with(
            &machine,
            u64::try_from(ctx)?,
            &PlanLevers::from_levers(levers)?,
            &mtp,
            experts,
        )?;
        println!(
            "{}",
            Record::new(&record::PLAN38)
                .w("place", place.name())
                .w("card", card.name)
                .w("experts", experts_name(experts))
                .u("ctx_max", plan.plan.ctx_max)
                .u("host_experts", plan.plan.host.experts)
                .u("card_experts", plan.plan.cards[0].experts)
                .line()
        );
        residency_host38(&plan.plan, residency, pick)?;
        let mut m = Body38::open_placed_mtp_residency(
            file,
            &plan,
            &inputs,
            CARD38,
            levers.host(),
            ub,
            &draft_split,
            &mtp,
            residency,
        )?;
        m.set_mode(mode);
        let body = m.body("generate_qwen3moe")?;
        println!(
            "load arch=qwen4exp resident_bytes={} ctx={ctx} layers={} mode={} store_bytes={} \
             prefill={} ubatch={} place={} card_layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            mode_name(mode),
            body.store_bytes(),
            path.name(),
            body.ubatch_rows(),
            place.name(),
            body.card_layers(),
            t.elapsed().as_secs_f64()
        );
        let draft = body.mtp().ok_or("the load opened no MTP draft")?;
        let head = match draft.head_map() {
            Some((_, n)) => format!("rows={n}"),
            None => "full".to_string(),
        };
        println!(
            "load draft=mtp resident={} arena={} head={head} card_bytes={} in {:.1} s (runtime \
             value)",
            draft.resident_bytes(),
            draft.arena_bytes(),
            plan.draft_card_bytes() + plan.arena_bytes,
            t.elapsed().as_secs_f64()
        );
        capture38_check(&mut m)?;
        Ok((
            m,
            Q38Cfg {
                prompt: path,
                draft: mode,
            },
        ))
    }

    /// The plan's card a qwen4exp open loads: its one card.
    const CARD38: usize = 0;

    /// Under `mid`, the `residency host` record of `plan`: the churn pool
    /// (card [`CARD38`]'s experts past the pinned ones) the load's host set
    /// holds beside the plan's host segments, which the load refuses by name
    /// when the plan's host headroom cannot take it; nothing under `off`.
    fn residency_host38(
        plan: &Plan<'_>,
        residency: Residency,
        pick: ResidencyPick,
    ) -> Result<(), GateError> {
        let Residency::Mid { pinned, .. } = residency else {
            return Ok(());
        };
        let pool = ChurnPool::of(plan, CARD38, pinned)
            .map_err(|e| format!("BLOOMERY_RESIDENCY={}: the churn pool: {e}", pick.word))?;
        record::residency_host(pick.word, &pool, plan).print();
        Ok(())
    }

    /// Under `mid`, keep the residency boundaries' reports for the `residency
    /// pass` records each arm prints after its lines; nothing under `off`.
    fn log38(m: &mut Qwen38Model, residency: Residency, arms: &[Arm]) -> Result<(), GateError> {
        if residency == Residency::Off {
            return Ok(());
        }
        // An arm's boundaries: its prompt call's, then at most one a
        // generated token (a step, or a verify keeping at least one).
        let passes = arms.iter().map(|a| a.n_gen).max().unwrap_or(0) + 1;
        m.body_parts("generate_qwen3moe")?.2.log_residency(passes);
        Ok(())
    }

    /// In graph mode: capture the decode step and hold its node kinds to the
    /// program's count, a mismatch ending the run by name.
    fn capture38_check(m: &mut Qwen38Model) -> Result<(), GateError> {
        if m.mode() != StepMode::Graph {
            return Ok(());
        }
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
        Ok(())
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
    /// A run whose model holds a route trace ends by sealing the set
    /// ([`Prompted::finish_trace`]) once every arm ran; a failed arm leaves
    /// the set without its `complete` line, as a killed run's is.
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
        let ran = s.arms(arms, |s, i, arm| {
            if let Some(c) = s.take_cleared() {
                record::residency_reset(&c).print();
            }
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
            let ran = run_arm(s.model_mut(), run, path, arm);
            after_passes(s.model_mut(), ran)
        });
        ran.map_err(|f| Box::new(f) as GateError)?;
        if let Some((dir, rows)) = B::finish_trace(s.model_mut())? {
            println!("route trace {} positions={rows} complete", dir.display());
        }
        Ok(())
    }

    /// The verify pass's capture log: one line a width.
    struct VerifyCaptures;

    impl app::RowsLog for VerifyCaptures {
        fn capture_rows(&mut self, rows: usize, nodes: usize) -> Result<(), app::SessionError> {
            println!("capture verify rows={rows} nodes={nodes}");
            Ok(())
        }
    }

    /// Every arm on the loaded model through the MTP draft: the session over
    /// it, the draft opened beside it (the verify passes of 2 to 4 rows
    /// captured in graph mode, each width's nodes printed), each arm its
    /// prompt — the draft walked over its units — and its windows; each
    /// after the first from the session's clear, the draft started over.
    fn drive38_mtp(
        m: Qwen38Model,
        cfg: Q38Cfg,
        run: &Run,
        arms: &[Arm],
        listed: bool,
        sync: bool,
    ) -> Result<(), GateError> {
        let mut s = Session::from_model(m, u32::try_from(run.ctx)?);
        let path = cfg.prompt;
        let draft = MtpDraft::open(s.model(), path, cfg.draft)?;
        let mut spec = s.with_draft::<MtpDraft<Body38>, 4>(draft, &mut VerifyCaptures)?;
        let count = arms.len();
        for (i, arm) in arms.iter().enumerate() {
            if i > 0 {
                s.clear()?;
                spec.draft_mut().restart();
                if let Some(c) = s.take_cleared() {
                    record::residency_reset(&c).print();
                }
            }
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
            let ran = run_arm38_mtp(&mut s, &mut spec, path, run, arm);
            after_passes(s.model_mut(), ran)?;
        }
        Ok(())
    }

    /// An arm's result `ran`, with the `residency pass` records of its
    /// boundaries printed after it ([`print_passes`]): a failed print is the
    /// arm's error, and when the arm failed too the error names both.
    fn after_passes<B: Prompted, T>(
        m: &mut GpuModel<B>,
        ran: Result<T, GateError>,
    ) -> Result<T, GateError> {
        match (ran, print_passes(m)) {
            (ran, Ok(())) => ran,
            (Ok(_), Err(p)) => Err(p),
            (Err(r), Err(p)) => {
                Err(format!("{r}; then printing the residency passes after it: {p}").into())
            }
        }
    }

    /// The `residency pass` records of the boundaries since the last print,
    /// after the arm's lines: nothing prints between two timed steps.
    fn print_passes<B: Prompted>(m: &mut GpuModel<B>) -> Result<(), GateError> {
        for (kind, r) in B::residency_passes(m)? {
            record::residency_pass_of(kind, &r).print();
        }
        Ok(())
    }

    /// What the drafted generation's sink keeps: every pass's outcome and
    /// wall, every kept token at its position, and the stats probes.
    struct Windows {
        stats: bool,
        emitted: Vec<(u32, u32)>,
        passes: Vec<(bool, usize, f64, Duration)>,
        probes: Vec<Probe>,
        n_gen: usize,
    }

    impl PassSink<Session<Body38>> for Windows {
        type Error = GateError;

        fn begin(&mut self, t: &Session<Body38>) -> Result<(), GateError> {
            if self.stats {
                self.probes.reserve_exact(self.n_gen);
                self.probes.push(probe38(t.model())?);
            }
            Ok(())
        }

        fn pass(
            &mut self,
            t: &Session<Body38>,
            c: &Committed,
            tokens: &[u32],
            wall: Duration,
        ) -> Result<(), GateError> {
            for (r, &tok) in (0u32..).zip(tokens) {
                self.emitted.push((c.pos + r, tok));
            }
            self.passes.push((
                c.proposed,
                c.kept,
                wall.as_secs_f64() * 1e3 / c.kept as f64,
                wall,
            ));
            if self.stats {
                self.probes.push(probe38(t.model())?);
            }
            Ok(())
        }
    }

    /// One arm through the draft: the prompt — the draft walked over its
    /// units, its own lines as the plain run's — then the windows until `-n`
    /// tokens are out, every kept token the target's own argmax.
    fn run_arm38_mtp(
        s: &mut Session<Body38>,
        spec: &mut Speculative<MtpDraft<Body38>, 4>,
        path: Prompt38,
        run: &Run,
        arm: &Arm,
    ) -> Result<(), GateError> {
        let (ids, n_gen, warm) = (&arm.ids, arm.n_gen, run.warm);
        let plan = <Body38 as Prompted>::plan(s.model(), ids.len(), path)?;
        let t = Instant::now();
        let before = s.model().body("prefill")?.hybrid().stats();
        let next = spec.prompt(s, ids)?;
        after38_prompt(s.model_mut(), before)?;
        let prefill_wall = t.elapsed();
        println!(
            "step 0 {} {next} (the {} prompt ids in prefill_steps={} units, plan={}, {:.2} s, \
             runtime value)",
            s.pos() - 1,
            ids.len(),
            plan.count,
            plan.text,
            prefill_wall.as_secs_f64()
        );
        let mut sink = Windows {
            stats: run.stats,
            emitted: Vec::with_capacity(n_gen),
            passes: Vec::with_capacity(n_gen),
            probes: Vec::new(),
            n_gen,
        };
        let stop = Stop::new(n_gen, s.ctx())?;
        let out = runtime::generate(s, spec, ids, next, &stop, &mut sink)?;
        if out.tokens.len() < n_gen {
            return Err(format!(
                "generate_qwen3moe: the generation stopped at {} after {} tokens, before -n",
                out.stop.name(),
                out.tokens.len()
            )
            .into());
        }
        let prefill_ms = prefill_wall.as_secs_f64() * 1e3;
        println!(
            "time prompt n={} ms={prefill_ms:.4} tok/s={:.2} passes={} kind={}",
            ids.len(),
            ids.len() as f64 * 1e3 / prefill_ms,
            plan.count,
            plan.kind
        );
        println!(
            "stat prompt ubatch_tokens={} (no prompt image)",
            plan.ubatch_tokens
        );
        let kept = &sink.emitted[..n_gen - 1];
        for (k, &(pos, tok)) in kept.iter().enumerate() {
            println!("step {} {pos} {tok}", k + 1);
        }
        if run.timed {
            // A pass's wall over its positions is the row a plain run's step
            // wall compares with: one a kept position.
            let mut at = 0usize;
            for (i, &(proposed, rows, per, wall)) in sink.passes.iter().enumerate() {
                let tag = if i < warm { " warm" } else { "" };
                println!(
                    "time pass {}{tag} ms={:.4} positions={rows} kind={}",
                    i + 1,
                    wall.as_secs_f64() * 1e3,
                    if proposed { "mtp" } else { "plain" }
                );
                for _ in 0..rows {
                    if at >= n_gen - 1 {
                        break;
                    }
                    at += 1;
                    let tag = if at <= warm { " warm" } else { "" };
                    println!("time step {at}{tag} ms={per:.4}");
                }
            }
        }
        let tokens: Vec<u32> = std::iter::once(next)
            .chain(kept.iter().map(|&(_, t)| t))
            .collect();
        println!("tokens {tokens:?}");
        if let Some(t) = &run.tok {
            println!("text {:?}", t.decode(&tokens));
        }
        mtp_summary(&sink.passes, warm);
        if run.timed {
            let counted = &sink.passes[warm..];
            let positions: usize = counted.iter().map(|&(_, k, ..)| k).sum();
            let ms: f64 = counted
                .iter()
                .map(|&(_, _, _, w)| w.as_secs_f64() * 1e3)
                .sum();
            let mut per: Vec<f64> = counted.iter().map(|&(_, _, p, _)| p).collect();
            per.sort_by(f64::total_cmp);
            let p50 = per[per.len() / 2];
            let mean = ms / positions as f64;
            println!(
                "SMOKE mode={} prompt_tokens={} depth={} generated={n_gen} warm={warm} \
                 steps={positions} passes={} p50_ms={p50:.4} mean_ms={mean:.4} \
                 tok/s(p50)={:.2} tok/s(mean)={:.2} ctx={}",
                mode_name(run.mode),
                ids.len(),
                ids.len(),
                counted.len(),
                1e3 / p50,
                1e3 / mean,
                run.ctx
            );
        }
        print_stats(&sink.probes, warm);
        Ok(())
    }

    /// The `mtp summary` record: the windows' proposals, the kept lengths'
    /// histogram, the positions and their rate over the counted passes.
    fn mtp_summary(passes: &[(bool, usize, f64, Duration)], warm: usize) {
        let mut kept = [0u64; 4];
        let mut positions = 0usize;
        let mut proposals = 0usize;
        for &(p, k, ..) in passes {
            if p {
                proposals += 1;
            }
            kept[k - 1] += 1;
            positions += k;
        }
        let counted = &passes[warm..];
        let counted_positions: usize = counted.iter().map(|&(_, k, ..)| k).sum();
        let ms: f64 = counted
            .iter()
            .map(|&(_, _, _, w)| w.as_secs_f64() * 1e3)
            .sum();
        Record::new(&record::MTP_SUMMARY)
            .u("proposals", proposals)
            .list("kept", &kept)
            .u("positions", positions)
            .u("passes", passes.len())
            .f("tok/s(positions)", counted_positions as f64 * 1e3 / ms)
            .print();
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
        B::mark_prompt(m, m.pos(), ids.len())?;
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
        let mut probes: Vec<Probe> = Vec::with_capacity(if run.stats { n_gen } else { 0 });
        if run.stats {
            probes.extend(Probe::read(m)?);
        }
        for _ in 1..n_gen {
            let t0 = Instant::now();
            next = m.step(&[next])?;
            rows.push((m.pos() - 1, next, t0.elapsed().as_secs_f64() * 1e3));
            if run.stats {
                probes.extend(Probe::read(m)?);
            }
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
            None => println!(
                "stat prompt ubatch_tokens={} (no prompt image)",
                plan.ubatch_tokens
            ),
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
        print_stats(&probes, warm);
        Ok(())
    }

    /// The counters one `BLOOMERY_STEP_STATS` read takes: the host tier's
    /// since load, the process's page faults since start, and the card's
    /// free device bytes now.
    struct Probe {
        hybrid: HybridStats,
        majflt: u64,
        minflt: u64,
        vram_free: u64,
    }

    impl Probe {
        /// One read of `m`; `None` for a body with no host tier.
        fn read<B: Prompted>(m: &GpuModel<B>) -> Result<Option<Probe>, GateError> {
            let Some(hybrid) = B::host_stats(m)? else {
                return Ok(None);
            };
            let (minflt, majflt) = faults()?;
            Ok(Some(Probe {
                hybrid,
                majflt,
                minflt,
                vram_free: u64::try_from(m.gpu().mem_info()?.0)?,
            }))
        }
    }

    /// [`Probe::read`] of the qwen4exp body, which always has a host tier.
    fn probe38(m: &Qwen38Model) -> Result<Probe, GateError> {
        let (minflt, majflt) = faults()?;
        Ok(Probe {
            hybrid: m.body("generate_qwen3moe")?.hybrid().stats(),
            majflt,
            minflt,
            vram_free: u64::try_from(m.gpu().mem_info()?.0)?,
        })
    }

    /// The process's minor and major page faults since start: fields 10 and
    /// 12 of `/proc/self/stat`, counted past the command's closing
    /// parenthesis.
    fn faults() -> Result<(u64, u64), GateError> {
        let stat = std::fs::read_to_string("/proc/self/stat")?;
        let rest = stat
            .rsplit_once(')')
            .map(|(_, r)| r)
            .ok_or("/proc/self/stat: no command field")?;
        let fields: Vec<&str> = rest.split_whitespace().collect();
        let at = |i: usize| -> Result<u64, GateError> {
            Ok(fields
                .get(i)
                .ok_or_else(|| {
                    format!("/proc/self/stat: {} fields past the command", fields.len())
                })?
                .parse::<u64>()?)
        };
        Ok((at(7)?, at(9)?))
    }

    /// One `stat step` record per generated step from the deltas of `probes`
    /// (one read before the first generated step, one after each), then the
    /// `stat summary` over the steps past `warm`; nothing without two reads.
    /// `leg_us` and `host_slots` are summed over the step's `served` layers;
    /// `vram_free` is the read after the step, not a delta.
    fn print_stats(probes: &[Probe], warm: usize) {
        let mut legs: Vec<f64> = Vec::with_capacity(probes.len());
        let (mut straggle_max, mut slots, mut majflt, mut minflt) = (0.0_f64, 0_u64, 0_u64, 0_u64);
        let mut vram_free_min = u64::MAX;
        for (k, w) in probes.windows(2).enumerate() {
            let i = k + 1;
            let (p, q) = (&w[0].hybrid, &w[1].hybrid);
            let served = q.served - p.served;
            let leg_us = (q.leg_ns - p.leg_ns) as f64 / 1e3;
            let straggle_us = (q.straggle_ns - p.straggle_ns) as f64 / 1e3;
            let host_slots = q.host_slots - p.host_slots;
            let host_w2 = if served == 0 {
                0.0
            } else {
                (q.host_w2 - p.host_w2) / served as f64
            };
            let (dmaj, dmin) = (w[1].majflt - w[0].majflt, w[1].minflt - w[0].minflt);
            Record::new(&record::STAT_STEP_HOST)
                .u("i", i)
                .flag("warm", i <= warm)
                .u("served", served)
                .f("leg_us", leg_us)
                .f("straggle_us", straggle_us)
                .f("straggle_max_us", q.straggle_max_ns as f64 / 1e3)
                .u("host_slots", host_slots)
                .f("host_w2", host_w2)
                .u("go_early", q.go_early - p.go_early)
                .u("parks", q.parks_in_service - p.parks_in_service)
                .u("majflt", dmaj)
                .u("minflt", dmin)
                .u("vram_free", w[1].vram_free)
                .print();
            if i > warm {
                legs.push(leg_us);
                straggle_max = straggle_max.max(straggle_us);
                slots += host_slots;
                majflt += dmaj;
                minflt += dmin;
                vram_free_min = vram_free_min.min(w[1].vram_free);
            }
        }
        let n = legs.len();
        if n == 0 {
            return;
        }
        let mean = legs.iter().sum::<f64>() / n as f64;
        legs.sort_by(f64::total_cmp);
        Record::new(&record::STAT_SUMMARY_HOST)
            .u("steps", n)
            .f("leg_us_mean", mean)
            .f("leg_us_p50", legs[n / 2])
            .f("straggle_us_max", straggle_max)
            .f("host_slots_mean", slots as f64 / n as f64)
            .u("majflt", majflt)
            .u("minflt", minflt)
            .u("vram_free_load", probes[0].vram_free)
            .u("vram_free_min", vram_free_min)
            .print();
    }
}
