//! `generate_ds41` — the thin end-to-end decode CLI of the V4.1 engine, and
//! its timing ruler: `generate`'s shape over `bloomery_gpu_deepseek41::body`.
//!
//!     generate_ds41 [--prompt-id P | --tokens a,b,c] [--depth D] [-n N]
//!                   [--ctx C] [--place a|gate] [--mode eager|graph]
//!                   [--time [--warm W]]
//!
//! Defaults: prompt 0 (none when `--depth` is given), N 32, C the serving
//! context (`workstation::CTX_MAX`), place `a`, mode graph, W 0. A flag given
//! twice takes its last value, so a recipe's default can be overridden by
//! the arguments after it.
//!
//! Greedy only, one token per generated step. The fed ids go in batches of
//! up to `body::T_MAX` positions (`body::prefill`, bit for bit the steps'
//! state); `BLOOMERY_PREFILL=steps` feeds them one real step per id instead,
//! the same binary's timing arm, and the `load` line prints which one ran
//! (`prefill=`). Prompt ids are
//! the V4.1 file's own: row P of `tools/ref/prompts.tsv` as
//! `tools/ref/ik-greedy.sh` tokenized it into
//! `$BLOOMERY_DATA/greedy-ds41/prompt<P>.tsv` — the ids the long gate's
//! `--free`, the dsloop gate and ik's continuation read.
//!
//! `--place` is the placement the engine loads by: `a` is the serving plan
//! (`workstation::plan_a`, every layer and the head on the A6000, each routed
//! layer's expert prefix the budget allows on the card, the rest on the host
//! tier) and the one the timing runners use; `gate` is the step gate's
//! (`workstation::plan_gate`, the same on the 3090). The card is found by
//! name, so the box's card pin decides which placements can load. The ring
//! shadows are page-locked host memory: the `plan` line prints the plan's
//! figure (`host_shadow=`), the `load` line the allocation
//! (`shadow=host <bytes>`) and the card's unified addressing the load
//! checked; `resident_bytes` counts device bytes only.
//!
//! `--depth D` stands the run at depth D before its first generated token:
//! the prompt, then the depth tables' sequence (`lcg_prompt` in
//! `tools/ref/lease.sh`) from index `P` on, every id one real step. With no
//! prompt the fed ids are exactly `lcg_prompt D`, the ids the V2-Lite depth
//! runner feeds. There is no synthetic depth: `seed_depth` refuses on this
//! body.
//!
//! Any depth up to `--ctx` runs: every indexer layer selects its stream's
//! list at every position (the identity while the visible rows fit in
//! `top_k`, the indexer's top-k after). The run selects with the file's
//! `top_k`, the model's own; the load line prints it, and a body that
//! loaded with another is refused before the first step.
//!
//! `--time` is a MEASUREMENT and belongs under the machine-wide lease
//! (`tools/ref/time-gate.sh generate_ds41 … --time`,
//! `tools/ref/depth-ds41.sh`). It times each generated step host-side around
//! `step(&[tok])`; the argmax readback synchronizes inside, and the host
//! tier's share runs inside it, so the wall time is the whole step. The fed
//! ids (the prompt, then the depth ids) are timed as one wall, not step by
//! step: the `time prompt n=<P> ms= tok/s= passes=<K> kind=` row runs from
//! before the first fed step to after the readback of generated token 0.
//! `passes` is the passes the feed took: the batches under `kind=batch`
//! (with the DSpark draft, each batch's feature rows are read and appended
//! to the draft inside the wall), one step per id under `kind=steps` — the
//! step rate by construction, at or above it, since that feed runs one body
//! per id and reads back once at the end — or `dspark` (the step feed with
//! the draft: one readback per id, its feature reads and draft appends) or
//! `checked` (the finite probe's eager steps run inside the wall; the probe
//! always feeds step by step). The row prints on every run,
//! after the loop, as a runtime value like the `load` line; it is a
//! measurement only under the lease, as `time step` is. Nothing is printed
//! between two timed steps. `--warm W` drops the first W generated steps from the
//! statistics and still prints them, marked. The `SMOKE` footer carries the
//! keys `generate`'s does (`p50_ms=`, `mean_ms=`, `warm=`), so the runners
//! that read one read the other.
//!
//! A batched feed prints the batch's device bytes (`prefill batch_bytes=`,
//! of them the batch-wide attention projections' `proj_bytes=` and what the
//! batches past a group's first hold for themselves, `group_bytes=`, for
//! groups of `group=` batches) before the prompt and a `stat prefill` line after its `step 0` line: the
//! host tier's batch services since load (`union_layers`, `union_cols`,
//! `union_host_slots`) and the union calls' wall (`union_ms`), the part of
//! the feed the card waits on the host; a runtime value, as `time prompt` is.
//! A `stat prefill split` line follows (`body::PrefillStats`): the group
//! lever (`group=`), the prologue, the enqueue with its union calls, waits
//! on the route's copies and activation copies, and the enqueue time left,
//! summed and per layer-batch — the waits also per layer-batch of a group's
//! first batch (`wait_first_lb=`: in a group of two or more, the route the
//! previous layer's last batch enqueued ahead); the queue entries — launches, event records, stream waits —
//! the route and the shadow put in a layer-batch (`entries_route=`,
//! `entries_shadow=`, the launch-queue model's N_r and N_s) and the host
//! tier's batch-excluded slots a layer-batch (`excluded_lb=`); with
//! `BLOOMERY_STEP_STATS=1` also each layer's card time by event pairs
//! (`card_out`: its first launch to its route's copies; `card_in`: its
//! shadow; `card_proj`: the batch-wide attention projections, inside
//! `card_out`), whose reads add a wait per batch to the feed.
//! The `load` line's `group=` is `BLOOMERY_PREFILL_GROUP` (default 2; 1 runs
//! each batch alone), the batches whose layers a batched feed runs in turn.
//! A second `stat prefill ced=` line names the triangle's state (the `load`
//! line's `ced=`: `on`, or `off (reason)`) and the last call's needs: its
//! positions, the first whose features were kept, the blocks and latent
//! parts it ran over every layer, and each layer's block and latent starts.
//!
//! The binary owns its main thread, so it pins it to the dispatcher's cpu
//! slot (`threads::pool().pin_caller()`), as `bloomery-decode` and
//! `bench_v41_host` do; `BLOOMERY_PIN_MAIN=0` leaves it floating, for the
//! A/B. The `load` line prints both the ask and the outcome, and
//! `launch_thread=` whether `BLOOMERY_LAUNCH_THREAD=1` gave the replays'
//! launches their own thread, and `launch_cpu=` where it runs: the SMT
//! sibling of the pinned main thread's cpu, or `float` beside
//! `BLOOMERY_PIN_MAIN=0`.
//!
//! `BLOOMERY_STEP_STATS=1` reads, after every generated step, the host
//! tier's counters (`HybridStats`), the replays' launch costs
//! (`LaunchStats`), the process's page faults (`getrusage`)
//! and the card's free device bytes (`cuMemGetInfo`), and prints one
//! `stat step` line per step and a `stat summary` over the steps `--warm`
//! keeps, after the loop, as the `time` lines are. The summary's
//! `vram_free_load` is the read before the first generated step (after the
//! load, the capture and the fed ids); a replayed step allocates nothing,
//! so `vram_free` staying there is the expected line. Unset, the stat path
//! does not run: no read, no call.
//!
//! `BLOOMERY_DRAFT=lookup` serves an n-gram lookup draft
//! (`bloomery_gpu_gates::draft::Lookup`, fed the fed ids and every emitted
//! token) through the skewed two-row pass. A pass with a proposal `d` runs
//! `step_pair(next, d)`: row A's argmax equal to `d` accepts both rows' tokens
//! (two positions), otherwise the second position is taken back and row A's
//! token alone is emitted (one position). A pass with no proposal is one
//! `step`. Greedy either way, so the `tokens` line equals the plain run's;
//! the last pass may overshoot `-n` by one, and the lines print the first
//! `-n` tokens. The run needs `--ctx` to hold that one extra position. Its
//! lines differ from the plain path's only where the lever is: `time pass`
//! rows (`positions=`, `kind=plain|pair-accept|pair-reject`) instead of
//! `time step`, a `draft summary`, and the `SMOKE` line's trailing
//! `positions=` and `tok/s(positions)=`, the positions the kept passes
//! advanced over their summed wall time. The pair pass's head and capture
//! are made before the prompt by one pair at position 0 and a `reset`, as
//! the step's capture is. `BLOOMERY_STEP_STATS` reads once per pass.
//!
//! `BLOOMERY_DRAFT=dspark` serves the DSpark draft at width 1
//! (`shared/ds41_dspark.rs`): the draft file is `$BLOOMERY_DSPARK_MODEL`, its
//! card `BLOOMERY_DSPARK_CARD` (the 3090 when unset), and the target carries
//! the feature tap of the draft's `target_layers`, built before the capture.
//! The fed ids go one step each, every position's features into the draft;
//! then every pass proposes (`kind=` on the `draft summary` line names the
//! draft), runs the pair over `[next, proposal]` and appends the features of
//! the positions it keeps. The rows and lines are the lookup's; a
//! `load draft=dspark` line follows the `load` line.
//!
//! `BLOOMERY_CHECK_FINITE=1` runs every position the run steps — the fed ids
//! and each generated token — first through the finite probe
//! (`shared/ds41_finite.rs`): the position's step eagerly, outside the graph,
//! each sub-layer's streams read where it wrote them; then the position is
//! taken back and the token stepped through the engine as without the lever,
//! so the `tokens` line is the plain run's. After the loop, a `stat finite
//! step` line per generated step (`ok`, or the non-finite seams, the first
//! one's `(layer, site)` and, at a MoE seam, its routing and buffers), a line
//! per fed position that is not `ok`, and a `stat finite summary`. A position
//! whose eager argmax differs from the engine's token says so. Refused with
//! `--time`, `BLOOMERY_DRAFT` and `BLOOMERY_STEP_STATS` (its host-tier
//! counters would count the probe's step too). Unset, the probe does not
//! run.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!("generate_ds41: built without the `deepseek41` feature; see `just gen-ds41`.");
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("generate_ds41", drive::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_finite.rs"]
mod finite;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_dspark.rs"]
mod dspark;

#[cfg(feature = "deepseek41")]
mod drive {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use bloomery_gpu::GpuError;
    use bloomery_gpu::head::Head;
    use bloomery_gpu::hybrid::HybridStats;
    use bloomery_gpu::model::{LaunchStats, StepMode};
    use bloomery_gpu_deepseek41::body::{self, Deepseek41Model};
    use bloomery_gpu_gates::draft::Lookup;
    use bloomery_gpu_gates::{GateError, data_dir, ref_model_path};
    use gguf::Split;
    use model::arch::deepseek41::place::PlanInputs;
    use model::placement::{HotList, Machine, workstation};

    use crate::dspark::{self, Dspark, Verdict};
    use crate::finite;

    const USAGE: &str = "usage: generate_ds41 [--prompt-id P | --tokens a,b,c] [--depth D] \
                         [-n N] [--ctx C] [--place a|gate] [--mode eager|graph] \
                         [--time [--warm W]]";

    /// Which placement the engine loads by.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Place {
        /// The serving plan, on the A6000.
        A,
        /// The step gate's plan, on the 3090.
        Gate,
    }

    impl Place {
        fn machine(self) -> fn(usize) -> Machine {
            match self {
                Place::A => workstation::plan_a,
                Place::Gate => workstation::plan_gate,
            }
        }

        fn name(self) -> &'static str {
            match self {
                Place::A => "a",
                Place::Gate => "gate",
            }
        }
    }

    /// Where the prompt comes from.
    enum Prompt {
        /// Row 0, or nothing under `--depth`.
        Default,
        Row(u32),
        Ids(Vec<u32>),
    }

    struct Args {
        prompt: Prompt,
        depth: Option<usize>,
        n_gen: usize,
        ctx: usize,
        place: Place,
        mode: StepMode,
        timed: bool,
        warm: Option<usize>,
    }

    fn parse_args() -> Result<Args, GateError> {
        let mut a = Args {
            prompt: Prompt::Default,
            depth: None,
            n_gen: 32,
            ctx: usize::try_from(workstation::CTX_MAX)?,
            place: Place::A,
            mode: StepMode::Graph,
            timed: false,
            warm: None,
        };
        let (mut row, mut ids) = (None, None);
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            if flag == "--time" {
                a.timed = true;
                continue;
            }
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value, or is unknown: {USAGE}"))?;
            match flag.as_str() {
                "--prompt-id" => row = Some(v.parse()?),
                "--tokens" => {
                    ids = Some(
                        v.split(',')
                            .map(|t| t.trim().parse::<u32>())
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                }
                "--depth" => a.depth = Some(v.parse()?),
                "-n" => a.n_gen = v.parse()?,
                "--ctx" => a.ctx = v.parse()?,
                "--warm" => a.warm = Some(v.parse()?),
                "--place" => {
                    a.place = match v.as_str() {
                        "a" => Place::A,
                        "gate" => Place::Gate,
                        other => return Err(format!("--place is a or gate, not {other}").into()),
                    };
                }
                "--mode" => {
                    a.mode = match v.as_str() {
                        "graph" => StepMode::Graph,
                        "eager" => StepMode::Eager,
                        other => {
                            return Err(format!("--mode is eager or graph, not {other}").into());
                        }
                    };
                }
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        a.prompt = match (row, ids) {
            (Some(_), Some(_)) => {
                return Err("--prompt-id and --tokens both name the prompt. Pass one.".into());
            }
            (Some(p), None) => Prompt::Row(p),
            (None, Some(v)) => Prompt::Ids(v),
            (None, None) => Prompt::Default,
        };
        check_counts(&a)?;
        Ok(a)
    }

    /// Refused rather than ignored, as `generate` refuses them: a `--warm`
    /// with nothing to trim, and one that trims every timed step.
    fn check_counts(a: &Args) -> Result<(), GateError> {
        if a.n_gen == 0 {
            return Err("-n wants at least one generated token".into());
        }
        if a.timed && a.n_gen < 2 {
            return Err(
                "--time with -n 1 has no generated step to time: token 0 comes out of the \
                 prompt's own step"
                    .into(),
            );
        }
        match a.warm {
            Some(_) if !a.timed => Err("--warm drops the first W steps from --time's \
                                        statistics, which this run has not asked for. Pass \
                                        both, or neither."
                .into()),
            Some(w) if w >= a.n_gen - 1 => Err(format!(
                "--warm {w} leaves no timed step of the {} that -n {} generates",
                a.n_gen - 1,
                a.n_gen
            )
            .into()),
            _ => Ok(()),
        }
    }

    /// The prompt's ids from `$BLOOMERY_DATA/greedy-ds41/prompt<P>.tsv`
    /// (`tools/ref/ik-greedy.sh`): the first row's third column.
    fn prompt_row(p: u32) -> Result<Vec<u32>, GateError> {
        let path = data_dir()
            .join("greedy-ds41")
            .join(format!("prompt{p}.tsv"));
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("{}: {e} — run just ik-greedy-ds41 {p}", path.display()))?;
        let row = text
            .lines()
            .find(|l| !l.starts_with('#') && !l.is_empty())
            .ok_or_else(|| format!("{}: no prompt row", path.display()))?;
        let ids = row
            .split('\t')
            .nth(2)
            .ok_or_else(|| format!("{}: the row has no ids", path.display()))?;
        Ok(ids
            .split(',')
            .map(|t| t.trim().parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?)
    }

    /// The depth tables' sequence, `lcg_prompt` of `tools/ref/lease.sh`: 100000,
    /// then an LCG walk over [1000, 91000). awk computes it in doubles, so this
    /// does too — past the second id the exact 64-bit LCG is another sequence.
    fn depth_ids(n: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(n);
        let mut s: f64 = 12345.0;
        for i in 0..n {
            if i == 0 {
                out.push(100_000);
                continue;
            }
            s = (s * 1_103_515_245.0 + 12345.0) % 2_147_483_648.0;
            // An integer-valued double in [0, 90000): the cast is exact.
            out.push(1000 + (s % 90_000.0) as u32);
        }
        out
    }

    /// The ids the run feeds before its first generated token, and how many
    /// of them are the prompt's.
    fn fed_ids(a: &Args) -> Result<(Vec<u32>, usize), GateError> {
        let mut ids = match &a.prompt {
            Prompt::Ids(v) => v.clone(),
            Prompt::Row(p) => prompt_row(*p)?,
            Prompt::Default if a.depth.is_some() => Vec::new(),
            Prompt::Default => prompt_row(0)?,
        };
        let prompt_len = ids.len();
        if let Some(d) = a.depth {
            if d < prompt_len {
                return Err(
                    format!("--depth {d} is shorter than the prompt's {prompt_len} ids").into(),
                );
            }
            ids.extend_from_slice(&depth_ids(d)[prompt_len..]);
        }
        if ids.is_empty() {
            return Err("an empty prompt and no --depth: nothing to feed".into());
        }
        Ok((ids, prompt_len))
    }

    pub fn run() -> Result<(), GateError> {
        let a = parse_args()?;
        let draft = draft_lever()?;
        let check_finite = finite_lever(&a, draft)?;
        let prefill_mode = body::PrefillMode::from_env()?;
        let pin_main = !std::env::var("BLOOMERY_PIN_MAIN").is_ok_and(|v| v == "0");
        let pinned = pin_main && threads::pool().pin_caller();
        let (ids, prompt_len) = fed_ids(&a)?;
        let depth = ids.len();
        // Positions 0 .. depth − 1 are the fed ids; the N − 1 feedback steps
        // take depth .. depth + N − 2.
        let fed = depth + a.n_gen - 1;
        if draft != Draft::Off && a.n_gen < 2 {
            return Err(format!(
                "BLOOMERY_DRAFT={} with -n 1 has no pass to draft: token 0 comes out of the \
                 prompt's own step",
                draft.name()
            )
            .into());
        }
        let draft_file = match draft {
            Draft::Dspark => Some(dspark::draft_hparams()?),
            _ => None,
        };

        let path = ref_model_path()?;
        let t = Instant::now();
        let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::read(&file)?;
        let ctx_max = print_plan(&inputs, a.place, a.ctx)?;
        // The caches hold the plan's ctx_max positions, the value the model
        // is loaded with; --ctx only asks for it.
        if fed > ctx_max {
            return Err(format!(
                "depth {depth} + {} fed tokens exceed the plan's ctx_max {ctx_max} \
                 (--ctx {})",
                a.n_gen - 1,
                a.ctx
            )
            .into());
        }
        if draft != Draft::Off && fed + 1 > ctx_max {
            return Err(format!(
                "BLOOMERY_DRAFT={}: depth {depth} + {} fed tokens and the last pair's \
                 overshoot exceed the plan's ctx_max {ctx_max} (--ctx {})",
                draft.name(),
                a.n_gen - 1,
                a.ctx
            )
            .into());
        }

        let mut m = body::open(file, a.place.machine(), a.ctx)?;
        m.set_mode(a.mode);
        let top_k = m.body("generate_ds41")?.indexer_top_k();
        let shadow = m.body("generate_ds41")?.shadow_host();
        if top_k != inputs.hp.indexer.top_k {
            return Err(format!(
                "the body selects {top_k} rows per stream, the file's top_k is {}: a step \
                 past that many visible rows would not be the model's",
                inputs.hp.indexer.top_k
            )
            .into());
        }
        println!(
            "load resident_bytes={} shadow=host {} unified_addressing={} ctx={} layers={} \
             top_k={top_k} mode={} place={} pin_main={} pinned={pinned} launch_thread={} \
             prefill={} ced={} group={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            shadow.bytes,
            shadow.unified_addressing,
            a.ctx,
            inputs.hp.n_layer,
            mode_name(a.mode),
            a.place.name(),
            if pin_main { "on" } else { "off" },
            m.launch_thread()
                .map_or("off".to_string(), |cpu| match cpu {
                    Some(c) => format!("on launch_cpu={c}"),
                    None => "on launch_cpu=float".to_string(),
                }),
            if check_finite {
                "steps"
            } else {
                prefill_mode.name()
            },
            m.body("generate_ds41")?.ced(),
            body::Body::prefill_group_lever()?,
            t.elapsed().as_secs_f64()
        );
        if let Some(h) = m.host_residency() {
            match h.populated() {
                Some(w) => println!(
                    "host_populate={} in {:.1} s (runtime value)",
                    w.bytes(),
                    w.wall().as_secs_f64()
                ),
                None => println!("host_populate=off"),
            }
            if let Some(l) = h.lock() {
                println!("host_lock={} B", l.bytes());
            }
        }
        // The draft builds the target's feature tap, so it loads before the
        // capture: the captured step then carries the tap.
        let mut spark = match &draft_file {
            Some((draft_split, hp)) => {
                let t = Instant::now();
                let card = dspark::draft_card()?;
                let target = Arc::new(
                    Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?,
                );
                let mut d = Dspark::open(&mut m, draft_split, hp, target, card)?;
                let (free, total) = d.mem_info()?;
                println!(
                    "load draft=dspark card={} width={} target_layers={:?} feature_width={} \
                     draft_card_free={free} of {total} in {:.1} s (runtime value)",
                    d.card(),
                    dspark::WIDTH,
                    m.body("generate_ds41")?
                        .feature_layers()
                        .unwrap_or_default(),
                    m.body("generate_ds41")?.feature_width(),
                    t.elapsed().as_secs_f64()
                );
                Some(d)
            }
            None => None,
        };
        if a.mode == StepMode::Graph {
            // Captured before the prompt, so the first timed step is a replay.
            println!("capture graph_nodes={}", m.capture_step()?);
        }
        // The batch's buffers made before the prompt, so the timed feed
        // allocates nothing.
        let feed_mode = if check_finite {
            body::PrefillMode::Steps
        } else {
            prefill_mode
        };
        if feed_mode == body::PrefillMode::Batch {
            body::prepare_prefill(&mut m)?;
            if std::env::var("BLOOMERY_STEP_STATS").is_ok_and(|v| v == "1") {
                let (gpu, _, b) = m.body_parts("generate_ds41")?;
                b.set_prefill_card_timing(gpu, true)?;
            }
            let body = m.body("generate_ds41")?;
            let (group, group_bytes) = body.prefill_group().unwrap_or_default();
            println!(
                "prefill batch_bytes={} proj_bytes={} group={group} group_bytes={group_bytes}",
                body.batch_bytes(),
                body.batch_proj_bytes()
            );
        }
        let mut check = if check_finite {
            let (gpu, w, _) = m.body_parts("generate_ds41")?;
            let head = Head::new(gpu, w, inputs.hp.rms_eps)?;
            println!(
                "check_finite=on: every position observed eagerly, taken back, then stepped \
                 through the engine"
            );
            Some(FiniteCheck {
                head,
                rows: Vec::with_capacity(fed + 1),
            })
        } else {
            None
        };
        match draft {
            Draft::Off => decode(&mut m, &a, &ids, prompt_len, feed_mode, check.as_mut()),
            Draft::Lookup => decode_draft(&mut m, &a, &ids, prompt_len, feed_mode, None),
            Draft::Dspark => decode_draft(&mut m, &a, &ids, prompt_len, feed_mode, spark.as_mut()),
        }
    }

    /// `BLOOMERY_CHECK_FINITE`: unset or `0` is off, `1` on; any other value
    /// is refused, and so is the lever beside `--time`, the draft and the
    /// step stats.
    fn finite_lever(a: &Args, draft: Draft) -> Result<bool, GateError> {
        let on = match std::env::var("BLOOMERY_CHECK_FINITE") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(v) if v == "0" => false,
            Ok(v) if v == "1" => true,
            Ok(v) => {
                return Err(format!("BLOOMERY_CHECK_FINITE is 1, 0 or unset, not {v:?}").into());
            }
            Err(e) => return Err(format!("BLOOMERY_CHECK_FINITE: {e}").into()),
        };
        let beside = [
            (
                a.timed,
                "--time: the probe's eager step would sit between two timed steps",
            ),
            (
                draft != Draft::Off,
                "BLOOMERY_DRAFT: the probe reads one-row steps",
            ),
            (
                std::env::var("BLOOMERY_STEP_STATS").is_ok_and(|v| v == "1"),
                "BLOOMERY_STEP_STATS=1: the host tier's counters would count the probe's step too",
            ),
        ];
        match beside.iter().find(|(set, _)| on && *set) {
            Some((_, why)) => Err(format!("BLOOMERY_CHECK_FINITE=1 is refused with {why}").into()),
            None => Ok(on),
        }
    }

    /// The finite probe's own head and what each checked position read.
    struct FiniteCheck {
        head: Head,
        rows: Vec<FiniteRow>,
    }

    /// One checked position: the probe's reading of it and the engine's
    /// token after it.
    struct FiniteRow {
        pos: u32,
        observed: finite::Observed,
        stepped: u32,
    }

    impl FiniteRow {
        /// The probe's reading, and the eager argmax where it is not the
        /// engine's token.
        fn describe(&self) -> String {
            let o = &self.observed;
            let mut line = o.describe();
            if o.token() != self.stepped {
                line.push_str(&format!(
                    " eager_differs: observed {} engine {}",
                    o.token(),
                    self.stepped
                ));
            }
            line
        }
    }

    /// One position through the probe, then the engine: `tok`'s step at the
    /// model's position observed eagerly, the position taken back, and `tok`
    /// stepped through the engine, whose token comes back.
    fn checked_step(
        m: &mut Deepseek41Model,
        c: &mut FiniteCheck,
        tok: u32,
    ) -> Result<u32, GateError> {
        let pos = m.pos();
        let observed = finite::observed_step(m, &mut c.head, tok, pos, &mut |_, _, _| Ok(()))?;
        m.rollback(pos)?;
        let stepped = m.step(&[tok])?;
        c.rows.push(FiniteRow {
            pos,
            observed,
            stepped,
        });
        Ok(stepped)
    }

    /// The `stat finite` lines: one per generated step (the rows past the
    /// `fed` fed positions), one per fed position that is not `ok`, and the
    /// summary.
    fn print_finite(c: &FiniteCheck, fed: usize) {
        for (k, r) in c.rows.iter().enumerate() {
            if k < fed {
                if r.observed.first_nonfinite().is_some() || r.observed.token() != r.stepped {
                    println!("stat finite fed pos {} {}", r.pos, r.describe());
                }
            } else {
                println!(
                    "stat finite step {} pos {} {}",
                    k + 1 - fed,
                    r.pos,
                    r.describe()
                );
            }
        }
        let bad: Vec<&FiniteRow> = c
            .rows
            .iter()
            .filter(|r| r.observed.first_nonfinite().is_some())
            .collect();
        let differs = c
            .rows
            .iter()
            .filter(|r| r.observed.token() != r.stepped)
            .count();
        let first = bad.first().map_or_else(
            || "none".to_string(),
            |r| {
                format!(
                    "pos {} {}",
                    r.pos,
                    r.observed
                        .first_nonfinite()
                        .map_or_else(String::new, finite::site_name)
                )
            },
        );
        println!(
            "stat finite summary positions={} nonfinite_positions={} first={first} \
             eager_differs={differs}",
            c.rows.len(),
            bad.len()
        );
    }

    /// Which draft `BLOOMERY_DRAFT` serves.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Draft {
        /// Unset: the plain path.
        Off,
        /// The n-gram lookup.
        Lookup,
        /// The DSpark draft (`shared/ds41_dspark.rs`).
        Dspark,
    }

    impl Draft {
        fn name(self) -> &'static str {
            match self {
                Draft::Off => "unset",
                Draft::Lookup => "lookup",
                Draft::Dspark => "dspark",
            }
        }
    }

    /// `BLOOMERY_DRAFT`: unset is the plain path, `lookup` and `dspark` the
    /// served drafts; any other value is refused.
    fn draft_lever() -> Result<Draft, GateError> {
        let draft = match std::env::var("BLOOMERY_DRAFT") {
            Err(std::env::VarError::NotPresent) => Draft::Off,
            Ok(v) if v == "lookup" => Draft::Lookup,
            Ok(v) if v == "dspark" => Draft::Dspark,
            Ok(v) => {
                return Err(format!("BLOOMERY_DRAFT is lookup, dspark or unset, not {v:?}").into());
            }
            Err(e) => return Err(format!("BLOOMERY_DRAFT: {e}").into()),
        };
        Ok(draft)
    }

    /// The plan the engine is about to load: where the experts sit. Returns
    /// the plan's `ctx_max`, the positions the caches are sized for.
    fn print_plan(inputs: &PlanInputs, place: Place, ctx: usize) -> Result<usize, GateError> {
        let machine = place.machine()(inputs.model.layers);
        let plan = inputs.plan(&machine, u64::try_from(ctx)?)?;
        let hot_list = HotList::from_env()?.map_or("none", HotList::path);
        let held: Vec<u64> = plan.n_l.iter().copied().filter(|&n| n > 0).collect();
        let card = &plan.cards[0];
        println!(
            "plan place={} card={} ctx_max={} card_experts={} ({} B) host_experts={} ({} B) \
             host_shadow={} B n_l={}..{} on {} layers card_budget={} hot_list={hot_list}",
            place.name(),
            machine.cards[0].name,
            plan.ctx_max,
            card.experts,
            card.expert_bytes,
            plan.host.experts,
            plan.host.expert_bytes,
            plan.host.shadow_bytes,
            held.iter().min().copied().unwrap_or(0),
            held.iter().max().copied().unwrap_or(0),
            held.len(),
            plan.card_budget
                .map_or_else(|| "none".to_string(), |b| b.to_string())
        );
        Ok(usize::try_from(plan.ctx_max)?)
    }

    fn mode_name(mode: StepMode) -> &'static str {
        if mode == StepMode::Graph {
            "graph"
        } else {
            "eager"
        }
    }

    /// The host tier's batch services since load, one line: the layers
    /// served, the columns and host slots they carried, and the union calls'
    /// wall — the part of a batched feed the card waits on the host.
    fn print_union(m: &mut Deepseek41Model) -> Result<(), GateError> {
        let split = m.body_parts("generate_ds41")?.2.take_prefill_stats();
        let b = m.body("generate_ds41")?;
        let s = b.hybrid().stats();
        println!(
            "stat prefill union_layers={} union_cols={} union_host_slots={} union_ms={:.1}",
            s.batch_served,
            s.batch_cols,
            s.batch_host_slots,
            s.batch_ns as f64 / 1e6
        );
        if let Some(need) = b.prefill_need() {
            println!("stat prefill ced={} {}", b.ced(), need.describe());
        }
        println!("stat prefill split {}", split.describe());
        Ok(())
    }

    /// The prompt feed's wall and shape, the `time prompt` row.
    struct FeedTime {
        /// The fed ids.
        n: usize,
        /// The steps the feed took.
        passes: usize,
        /// Before the first fed step to after the readback of generated
        /// token 0.
        wall: Duration,
        /// `steps`, `dspark`, or `checked` under the finite probe.
        kind: &'static str,
    }

    impl FeedTime {
        /// `time prompt n= ms= tok/s= passes= kind=`, written after the loop.
        fn print(&self) {
            let ms = self.wall.as_secs_f64() * 1e3;
            println!(
                "time prompt n={} ms={ms:.4} tok/s={:.2} passes={} kind={}",
                self.n,
                self.n as f64 * 1e3 / ms,
                self.passes,
                self.kind
            );
        }
    }

    /// Feed `ids` — in batches under `mode` `Batch`, else one real step per
    /// id, each through the finite probe first under `check` — print the
    /// `fed` and `step 0` lines, and return the first generated token and the
    /// feed's wall.
    fn feed(
        m: &mut Deepseek41Model,
        ids: &[u32],
        prompt_len: usize,
        mode: body::PrefillMode,
        check: Option<&mut FiniteCheck>,
    ) -> Result<(u32, FeedTime), GateError> {
        let depth = ids.len();
        let head: Vec<u32> = ids.iter().copied().take(4).collect();
        let tail: Vec<u32> = ids.iter().copied().skip(depth.saturating_sub(4)).collect();
        println!("fed ids={depth} first={head:?} last={tail:?} depth_sequence_from={prompt_len}");
        let batch = check.is_none() && mode == body::PrefillMode::Batch;
        let kind = match (&check, batch) {
            (Some(_), _) => "checked",
            (None, true) => "batch",
            (None, false) => "steps",
        };
        let t = Instant::now();
        let next = match check {
            None if batch => body::prefill(m, ids)?,
            None => m.step(ids)?,
            Some(c) => {
                let mut next = 0;
                for &id in ids {
                    next = checked_step(m, c, id)?;
                }
                next
            }
        };
        let wall = t.elapsed();
        println!(
            "step 0 {} {next} (the {depth} fed steps in {:.1} s, runtime value)",
            m.pos() - 1,
            wall.as_secs_f64()
        );
        if batch {
            print_union(m)?;
        }
        let time = FeedTime {
            n: depth,
            passes: if batch {
                body::batch_count(depth)
            } else {
                depth
            },
            wall,
            kind,
        };
        Ok((next, time))
    }

    /// Feed `ids`, then the N − 1 feedback steps, timed when asked; every
    /// line after the loop.
    fn decode(
        m: &mut Deepseek41Model,
        a: &Args,
        ids: &[u32],
        prompt_len: usize,
        mode: body::PrefillMode,
        mut check: Option<&mut FiniteCheck>,
    ) -> Result<(), GateError> {
        let depth = ids.len();
        let (mut next, feed_time) = feed(m, ids, prompt_len, mode, check.as_deref_mut())?;
        let mut rows: Vec<(u32, u32, f64)> = Vec::with_capacity(a.n_gen - 1);
        let mut tokens: Vec<u32> = Vec::with_capacity(a.n_gen);
        tokens.push(next);
        let stats_on = std::env::var("BLOOMERY_STEP_STATS").is_ok_and(|v| v == "1");
        let mut probes: Vec<Probe> = Vec::new();
        if stats_on {
            probes.reserve_exact(a.n_gen);
            probes.push(Probe::read(m)?);
        }
        for _ in 1..a.n_gen {
            let t0 = Instant::now();
            if let Some(c) = check.as_deref_mut() {
                next = checked_step(m, c, next)?;
            } else {
                next = m.step(&[next])?;
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            rows.push((m.pos() - 1, next, ms));
            if stats_on {
                probes.push(Probe::read(m)?);
            }
        }
        let warm = a.warm.unwrap_or(0);
        feed_time.print();
        for (k, (pos, tok, ms)) in rows.iter().enumerate() {
            let i = k + 1;
            tokens.push(*tok);
            println!("step {i} {pos} {tok}");
            if a.timed {
                let tag = if i <= warm { " warm" } else { "" };
                println!("time step {i}{tag} ms={ms:.4}");
            }
        }
        println!("tokens {tokens:?}");
        if stats_on {
            print_stats(&probes, warm);
        }
        if let Some(c) = check {
            print_finite(c, depth);
        }
        if a.timed {
            let counted: Vec<f64> = rows[warm..].iter().map(|r| r.2).collect();
            let mut sorted = counted.clone();
            sorted.sort_by(f64::total_cmp);
            let p50 = sorted[sorted.len() / 2];
            let mean = counted.iter().sum::<f64>() / counted.len() as f64;
            println!(
                "SMOKE mode={} place={} prompt_tokens={prompt_len} depth={depth} generated={} \
                 warm={warm} steps={} p50_ms={p50:.4} mean_ms={mean:.4} tok/s(p50)={:.2}",
                mode_name(a.mode),
                a.place.name(),
                a.n_gen,
                counted.len(),
                1e3 / p50
            );
        }
        Ok(())
    }

    /// What one draft pass ran, and so how many positions it advanced.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum PassKind {
        /// No proposal: one `step`.
        Plain,
        /// Row A's argmax was the draft: both rows' tokens kept.
        Accept,
        /// Row A's argmax was not the draft: the second position taken back.
        Reject,
    }

    impl PassKind {
        fn name(self) -> &'static str {
            match self {
                PassKind::Plain => "plain",
                PassKind::Accept => "pair-accept",
                PassKind::Reject => "pair-reject",
            }
        }

        fn positions(self) -> usize {
            if self == PassKind::Accept { 2 } else { 1 }
        }
    }

    /// `decode` under `BLOOMERY_DRAFT`: passes until `-n` tokens are out, each
    /// timed around the proposal, the pass and the draft's update — the
    /// lookup's push, or with `spark` the DSpark draft's features
    /// (`dspark::pass`). Under `spark` the draft gets every fed position's
    /// features, from the batches' feature rows or one step per id.
    fn decode_draft(
        m: &mut Deepseek41Model,
        a: &Args,
        ids: &[u32],
        prompt_len: usize,
        mode: body::PrefillMode,
        mut spark: Option<&mut Dspark>,
    ) -> Result<(), GateError> {
        // The pair pass's second head and, in graph mode, its capture are
        // made by the first pair: one here, then back to the fresh context,
        // so no timed pass pays for them.
        m.step_pair(ids[0], ids[0])?;
        m.reset()?;
        if a.mode == StepMode::Graph {
            println!("capture pair_graph_nodes={}", m.pair_graph_nodes()?.len());
        }
        let depth = ids.len();
        let (first, feed_time) = match spark.as_deref_mut() {
            None => feed(m, ids, prompt_len, mode, None)?,
            Some(d) => feed_dspark(m, d, ids, prompt_len, mode)?,
        };
        let mut next = first;
        let mut look = Lookup::new();
        for &t in ids {
            look.push(t);
        }
        look.push(next);
        // (position the token is the argmax after, token), past token 0.
        let mut emitted: Vec<(u32, u32)> = Vec::with_capacity(a.n_gen);
        let mut passes: Vec<(PassKind, f64)> = Vec::with_capacity(a.n_gen - 1);
        let stats_on = std::env::var("BLOOMERY_STEP_STATS").is_ok_and(|v| v == "1");
        let mut probes: Vec<Probe> = Vec::new();
        if stats_on {
            probes.reserve_exact(a.n_gen);
            probes.push(Probe::read(m)?);
        }
        while emitted.len() + 1 < a.n_gen {
            let t0 = Instant::now();
            let pos = m.pos();
            let verdict = match spark.as_deref_mut() {
                Some(d) => Some(dspark::pass(m, d, next)?),
                None => look
                    .propose()
                    .map(|d| lookup_pass(m, next, d))
                    .transpose()?,
            };
            let kind = match verdict {
                Some(Verdict::Accept([ta, tb])) => {
                    emitted.extend_from_slice(&[(pos, ta), (pos + 1, tb)]);
                    next = tb;
                    PassKind::Accept
                }
                Some(Verdict::Reject(ta)) => {
                    emitted.push((pos, ta));
                    next = ta;
                    PassKind::Reject
                }
                None => {
                    next = m.step(&[next])?;
                    emitted.push((pos, next));
                    PassKind::Plain
                }
            };
            if spark.is_none() {
                for &(_, t) in &emitted[emitted.len() - kind.positions()..] {
                    look.push(t);
                }
            }
            passes.push((kind, t0.elapsed().as_secs_f64() * 1e3));
            if stats_on {
                probes.push(Probe::read(m)?);
            }
        }
        if let Some(d) = spark.as_deref_mut() {
            d.check_fault()?;
        }
        let warm = a.warm.unwrap_or(0);
        if warm >= passes.len() {
            return Err(format!(
                "--warm {warm} leaves no timed pass of the {} this run took",
                passes.len()
            )
            .into());
        }
        feed_time.print();
        print_draft_rows(a, first, &emitted, &passes);
        if stats_on {
            print_stats(&probes, warm);
        }
        let kind = if spark.is_some() { "dspark" } else { "lookup" };
        print_draft_summary(a, &passes, prompt_len, depth, kind);
        Ok(())
    }

    /// The lookup's pass: the pair over `[next, d]`, the second position
    /// taken back unless row A's argmax is `d`.
    fn lookup_pass(m: &mut Deepseek41Model, next: u32, d: u32) -> Result<Verdict, GateError> {
        let [ta, tb] = m.step_pair(next, d)?;
        if ta == d {
            Ok(Verdict::Accept([ta, tb]))
        } else {
            m.rollback(m.pos() - 1)?;
            Ok(Verdict::Reject(ta))
        }
    }

    /// [`feed`] for the DSpark draft: the fed ids one step each, every
    /// position's features into the draft (`dspark::feed`), the same `fed`
    /// and `step 0` lines, and the wall as `kind=dspark`.
    fn feed_dspark(
        m: &mut Deepseek41Model,
        d: &mut Dspark,
        ids: &[u32],
        prompt_len: usize,
        mode: body::PrefillMode,
    ) -> Result<(u32, FeedTime), GateError> {
        let depth = ids.len();
        let head: Vec<u32> = ids.iter().copied().take(4).collect();
        let tail: Vec<u32> = ids.iter().copied().skip(depth.saturating_sub(4)).collect();
        println!("fed ids={depth} first={head:?} last={tail:?} depth_sequence_from={prompt_len}");
        let batch = mode == body::PrefillMode::Batch;
        let width = m.body("generate_ds41")?.feature_width();
        let t = Instant::now();
        let next = if batch {
            // Each position's features into the draft as the step feed hands
            // them over, one row at a time, whole groups through its graph.
            d.reset()?;
            let window = d.window();
            let mut append = |first: u32, rows: &[f32]| -> Result<(), GpuError> {
                d.skip_to(first)?;
                rows.chunks_exact(width).try_for_each(|row| d.feed(row))
            };
            let rows = body::FeatureRows {
                window,
                sink: &mut append,
            };
            let next = body::prefill_with(m, ids, Some(rows))?;
            d.flush()?;
            next
        } else {
            dspark::feed(m, d, ids)?
        };
        let wall = t.elapsed();
        println!(
            "step 0 {} {next} (the {depth} fed steps in {:.1} s, runtime value)",
            m.pos() - 1,
            wall.as_secs_f64()
        );
        if batch {
            print_union(m)?;
        }
        let time = FeedTime {
            n: depth,
            passes: if batch {
                body::batch_count(depth)
            } else {
                depth
            },
            wall,
            kind: if batch { "batch" } else { "dspark" },
        };
        Ok((next, time))
    }

    /// The `step` lines of the first `-n` tokens, the `time pass` rows when
    /// timed, and the `tokens` line, capped at `-n`.
    fn print_draft_rows(a: &Args, first: u32, emitted: &[(u32, u32)], passes: &[(PassKind, f64)]) {
        let kept = &emitted[..a.n_gen - 1];
        for (k, (pos, tok)) in kept.iter().enumerate() {
            println!("step {} {pos} {tok}", k + 1);
        }
        if a.timed {
            let warm = a.warm.unwrap_or(0);
            for (k, (kind, ms)) in passes.iter().enumerate() {
                let i = k + 1;
                let tag = if i <= warm { " warm" } else { "" };
                println!(
                    "time pass {i}{tag} ms={ms:.4} positions={} kind={}",
                    kind.positions(),
                    kind.name()
                );
            }
        }
        let tokens: Vec<u32> = std::iter::once(first)
            .chain(kept.iter().map(|&(_, t)| t))
            .collect();
        println!("tokens {tokens:?}");
    }

    /// The `draft summary` over every pass (the rate over the passes past
    /// `--warm`), then the `SMOKE` footer when timed: `generate`'s keys over
    /// the kept passes, then `positions=` and `tok/s(positions)=`.
    fn print_draft_summary(
        a: &Args,
        passes: &[(PassKind, f64)],
        prompt_len: usize,
        depth: usize,
        kind: &str,
    ) {
        let warm = a.warm.unwrap_or(0);
        let proposals = passes.iter().filter(|p| p.0 != PassKind::Plain).count();
        let accepts = passes.iter().filter(|p| p.0 == PassKind::Accept).count();
        let positions: usize = passes.iter().map(|p| p.0.positions()).sum();
        let kept = &passes[warm..];
        let kept_positions: usize = kept.iter().map(|p| p.0.positions()).sum();
        let kept_ms: f64 = kept.iter().map(|p| p.1).sum();
        let rate = kept_positions as f64 * 1e3 / kept_ms;
        println!(
            "draft summary proposals={proposals} accepts={accepts} positions={positions} \
             passes={} tok/s(positions)={rate:.2} kind={kind}",
            passes.len()
        );
        if a.timed {
            let mut sorted: Vec<f64> = kept.iter().map(|p| p.1).collect();
            sorted.sort_by(f64::total_cmp);
            let p50 = sorted[sorted.len() / 2];
            let mean = kept_ms / kept.len() as f64;
            println!(
                "SMOKE mode={} place={} prompt_tokens={prompt_len} depth={depth} generated={} \
                 warm={warm} steps={} p50_ms={p50:.4} mean_ms={mean:.4} tok/s(p50)={:.2} \
                 positions={kept_positions} tok/s(positions)={rate:.2}",
                mode_name(a.mode),
                a.place.name(),
                a.n_gen,
                kept.len(),
                1e3 / p50
            );
        }
    }

    /// The counters one `BLOOMERY_STEP_STATS` read takes: the host tier's
    /// since load, the engram rows' since load, the process's page faults
    /// since start, and the card's free device bytes now; and where the
    /// engram rows' helper runs (`None` off, `Some(None)` floating).
    #[derive(Clone, Copy)]
    struct Probe {
        hybrid: HybridStats,
        launch: LaunchStats,
        eng: bloomery_gpu_deepseek41::chain::glue::EngramStats,
        eng_helper: Option<Option<usize>>,
        majflt: u64,
        minflt: u64,
        vram_free: u64,
    }

    impl Probe {
        fn read(m: &Deepseek41Model) -> Result<Probe, GateError> {
            let body = m.body("generate_ds41")?;
            let hybrid = body.hybrid().stats();
            let launch = m.launch_stats();
            let eng = body.step_rows().engram_stats();
            let eng_helper = body.step_rows().helper_cpu();
            let stage = m
                .stages()
                .first()
                .ok_or("generate_ds41: the model has no stage")?;
            let vram_free = u64::try_from(stage.gpu().mem_info()?.0)?;
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
            Ok(Probe {
                hybrid,
                launch,
                eng,
                eng_helper,
                majflt: u64::try_from(ru.ru_majflt)?,
                minflt: u64::try_from(ru.ru_minflt)?,
                vram_free,
            })
        }
    }

    /// One line per generated step from the deltas of `probes` (one read
    /// before the first generated step, one after each), then the summary
    /// over the steps past `warm`. `straggle_max_us` is the worst single
    /// service since load (a maximum has no delta); `host_w2` is the step's
    /// mean host share of routed weight squared per service. `vram_free` is
    /// the read after the step, not a delta. `eng_warm`/`eng_cold` are the
    /// step's engram rows found in the page cache or not before the read,
    /// `eng_direct` the ones the step thread read itself (the helper off),
    /// `eng_wait_us` the step thread's time in the engram read (the wait on
    /// the helper, or the direct copy), `eng_helper_us` the helper's own and
    /// `eng_classify_us` the time the warm/cold count itself took (inside
    /// `eng_wait_us` with the helper on, outside it off). `overlap` is the
    /// step's host slot ids of a two-row pass's row 1 that row 0 also sent
    /// to the host at the same layer, `union` the distinct host slots of the
    /// step's rows per layer, summed (`host_slots − overlap`: a one-row step
    /// prints `overlap=0 union=<host_slots>`). The summary's `phi_mean` is
    /// the pooled row overlap over the kept steps, `Σ overlap / Σ` row 1's
    /// host slots, 0 when no kept step ran two rows. `go_early_first` is the
    /// step's first service finding its go already landed, `launch_us` the
    /// step's `cuGraphLaunch` call (on the launch thread under
    /// `BLOOMERY_LAUNCH_THREAD=1`), `first_serve_lag_us` the time from just
    /// before the launch was issued to the first service's entry, and
    /// `launch_wake_us` the launch thread's time from the post to the call
    /// (0 without it); the summary's `_mean`s are over the kept steps.
    fn print_stats(probes: &[Probe], warm: usize) {
        let mut legs: Vec<f64> = Vec::with_capacity(probes.len());
        let mut waits: Vec<f64> = Vec::with_capacity(probes.len());
        let (mut eng_warm, mut eng_cold, mut eng_direct) = (0_u64, 0_u64, 0_u64);
        let (mut eng_helper_ns, mut eng_classify_ns) = (0_u64, 0_u64);
        let mut vram_free_min = u64::MAX;
        let (mut straggle_max, mut slots, mut majflt, mut minflt) = (0.0_f64, 0_u64, 0_u64, 0_u64);
        let (mut overlap_sum, mut row1_sum) = (0_u64, 0_u64);
        let (mut early_first, mut launch_ns, mut lag_ns, mut wake_ns) =
            (0_u64, 0_u64, 0_u64, 0_u64);
        for (k, w) in probes.windows(2).enumerate() {
            let i = k + 1;
            let (p, q) = (&w[0].hybrid, &w[1].hybrid);
            let served = q.served - p.served;
            let leg_us = (q.leg_ns - p.leg_ns) as f64 / 1e3;
            let straggle_us = (q.straggle_ns - p.straggle_ns) as f64 / 1e3;
            let host_slots = q.host_slots - p.host_slots;
            let overlap = q.overlap_slots - p.overlap_slots;
            let row1 = q.pair_row1_slots - p.pair_row1_slots;
            let union = host_slots - overlap;
            let host_w2 = if served == 0 {
                0.0
            } else {
                (q.host_w2 - p.host_w2) / served as f64
            };
            let dmaj = w[1].majflt - w[0].majflt;
            let dmin = w[1].minflt - w[0].minflt;
            let (e, f) = (&w[0].eng, &w[1].eng);
            let (ew, ec, ed) = (f.warm - e.warm, f.cold - e.cold, f.direct - e.direct);
            let wait_us = (f.wait_ns - e.wait_ns) as f64 / 1e3;
            let helper_ns = f.helper_ns - e.helper_ns;
            let classify_ns = f.classify_ns - e.classify_ns;
            let early_first_d = q.go_early_first - p.go_early_first;
            let lag_d = q.first_serve_lag_ns - p.first_serve_lag_ns;
            let (lp, lq) = (&w[0].launch, &w[1].launch);
            let (launch_d, wake_d) = (lq.launch_ns - lp.launch_ns, lq.wake_ns - lp.wake_ns);
            let tag = if i <= warm { " warm" } else { "" };
            println!(
                "stat step {i}{tag} served={served} leg_us={leg_us:.1} straggle_us={straggle_us:.1} \
                 straggle_max_us={:.1} host_slots={host_slots} host_w2={host_w2:.4} \
                 overlap={overlap} union={union} go_early={} parks={} majflt={dmaj} minflt={dmin} \
                 vram_free={} eng_warm={ew} eng_cold={ec} \
                 eng_direct={ed} eng_wait_us={wait_us:.1} eng_helper_us={:.1} eng_classify_us={:.1} \
                 go_early_first={early_first_d} launch_us={:.1} first_serve_lag_us={:.1} launch_wake_us={:.1}",
                q.straggle_max_ns as f64 / 1e3,
                q.go_early - p.go_early,
                q.parks_in_service - p.parks_in_service,
                w[1].vram_free,
                helper_ns as f64 / 1e3,
                classify_ns as f64 / 1e3,
                launch_d as f64 / 1e3,
                lag_d as f64 / 1e3,
                wake_d as f64 / 1e3
            );
            if i > warm {
                early_first += early_first_d;
                launch_ns += launch_d;
                lag_ns += lag_d;
                wake_ns += wake_d;
                waits.push(wait_us);
                eng_warm += ew;
                eng_cold += ec;
                eng_direct += ed;
                eng_helper_ns += helper_ns;
                eng_classify_ns += classify_ns;
                vram_free_min = vram_free_min.min(w[1].vram_free);
                legs.push(leg_us);
                straggle_max = straggle_max.max(straggle_us);
                slots += host_slots;
                overlap_sum += overlap;
                row1_sum += row1;
                majflt += dmaj;
                minflt += dmin;
            }
        }
        let n = legs.len();
        if n == 0 {
            return;
        }
        let mean = legs.iter().sum::<f64>() / n as f64;
        legs.sort_by(f64::total_cmp);
        let wait_mean = waits.iter().sum::<f64>() / n as f64;
        waits.sort_by(f64::total_cmp);
        let phi_mean = if row1_sum == 0 {
            0.0
        } else {
            overlap_sum as f64 / row1_sum as f64
        };
        let helper = match probes[0].eng_helper {
            None => "off".to_string(),
            Some(None) => "floating".to_string(),
            Some(Some(cpu)) => format!("cpu{cpu}"),
        };
        println!(
            "stat summary steps={n} leg_us_mean={mean:.1} leg_us_p50={:.1} straggle_us_max={straggle_max:.1} \
             host_slots_mean={:.1} phi_mean={phi_mean:.4} majflt={majflt} minflt={minflt} vram_free_load={} \
             vram_free_min={vram_free_min} eng_helper={helper} eng_warm={eng_warm} \
             eng_cold={eng_cold} eng_direct={eng_direct} eng_wait_us_mean={wait_mean:.1} \
             eng_wait_us_p50={:.1} eng_wait_us_max={:.1} eng_helper_us_mean={:.1} \
             eng_classify_us_mean={:.1} go_early_first_mean={:.3} launch_us_mean={:.1} \
             first_serve_lag_us_mean={:.1} launch_wake_us_mean={:.1}",
            legs[n / 2],
            slots as f64 / n as f64,
            probes[0].vram_free,
            waits[n / 2],
            waits[n - 1],
            eng_helper_ns as f64 / 1e3 / n as f64,
            eng_classify_ns as f64 / 1e3 / n as f64,
            early_first as f64 / n as f64,
            launch_ns as f64 / 1e3 / n as f64,
            lag_ns as f64 / 1e3 / n as f64,
            wake_ns as f64 / 1e3 / n as f64
        );
    }
}
