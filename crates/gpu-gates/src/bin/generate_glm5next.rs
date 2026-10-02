//! Greedy generation with the GLM-5.3-Flash program: the file planned onto
//! one card, the routed experts the card experts read by the expert rule
//! (each layer's id prefix) and the rest on the
//! host tier (`model::arch::glm5next::place`), loaded through the session
//! (`app::Loaded`), the prompt fed as `--prefill` says, then `-n` greedy
//! steps.
//!
//! `generate_glm5next --tokens a,b,c [-n N] [--ctx C] [--place a|gate|bp|<stage>[+<tier>…]]
//! [--mode graph|eager] [--prefill batch|steps] [--last-step]
//! [--time [--warm W]] [--pair] [--logits] [--plan]`
//!
//! - `--tokens`: the prompt's ids (the file's own vocabulary, no BOS added).
//! - `--ctx`: the positions the caches hold; the plan refuses more than the
//!   deepest context a reference set checks the selector at
//!   (`place::ORACLE_POSITIONS`). Default 2048.
//! - `--place`: the shared placement word (`generate::Place`): `a` is the
//!   serving plan (`workstation::plan_a`, the A6000), `gate` the gate card's
//!   (`workstation::plan_gate`, the 3090), `bp` plan (b′)
//!   (`workstation::plan_bp`: plan (a) on the A6000, the 3090 an expert tier
//!   under the host tier, its prompt-batch bytes `place::tier_batch`'s), also
//!   spelled `a6000+3090`. A list of more tier cards than the GLM body serves
//!   and a stage other than the A6000 are refused by name before the plan.
//!   Default `gate`. The NextN draft and residency run beside `bp` as beside
//!   `a`: the tier serves the target's experts it holds, the residency
//!   machine moves the stage card's alone, and the draft's walk stays on the
//!   stage card and the host.
//! - `--prefill`: `batch` feeds the prompt in batches
//!   (`bloomery_gpu_glm5next::prefill`), `steps` one decode step a position,
//!   the same bits at any `--ctx`, past the positions the latent layers
//!   attend whole too, as long as no batch is longer than a chunk (one past
//!   it runs its mixers' projections on the GEMM); the same-binary arm.
//!   Default `batch`.
//! - `--last-step`: the prompt less its last id fed as `--prefill` says,
//!   then the last id as a decode step — the cut the server's seat feeds a
//!   prompt by. Refused by name beside the draft, `--time` and `--pair`.
//! - The model file is `$BLOOMERY_REF_MODEL` (`ref_model_path`), which
//!   `tools/box.sh` exports from the `glm5next` profile, as in every other bin.
//! - `--plan` prints the plan and exits before the load.
//! - `--logits` prints the head's last logits row by its bits after the
//!   tokens.
//! - `--time` times the feed (`time prompt`, `kind` the feed's mode and
//!   `passes` its batches or steps, through the readback of generated token
//!   0) and each generated step
//!   after token 0 (`time step`, the step through its token's readback), and
//!   ends with the `SMOKE` footer over the kept steps. `--warm W` drops the
//!   first W of them from the footer. A measurement: it belongs under the
//!   machine-wide lease (`tools/ref/depth-glm5next.sh`). The `step` and `time
//!   step` records print after the last step, so no write sits between two
//!   timed steps.
//! - `--pair` then runs the same tokens again as verifies of two rows
//!   (`runtime::Verify` on the session, `bloomery_gpu_glm5next`'s verify) on
//!   the verify's load (`app::arch::glm5next::open_pair`: each KDA layer's
//!   state two lanes, the plan's card experts fewer by the second lane's
//!   bytes; the plain run's load holds one): the model cut back to the prompt's end (its checkpoint), in graph mode
//!   the verify captured first (`capture` with `pair_graph_nodes`), then
//!   pass `k` feeds the plain run's fed tokens `2k` and `2k + 1` at their
//!   positions and keeps both rows — the draft is the target's own greedy
//!   token, so every row is accepted — for `(N − 1) / 2` passes. Each pass's
//!   two tokens must be the plain run's next two, else the run ends in a
//!   named error: the verify is not two steps. With `--time` each pass
//!   prints `time pass` (`kind=pair`, `positions=2`), its wall from the
//!   verify's plan through its commit, both rows' readbacks included; `--warm
//!   W` marks the passes over the first W steps' positions. The pass wall
//!   over the step's (`smoke`'s `p50_ms`) is the verify's cost `V2/S1`, both
//!   from one process on one load. Refused beside the route trace (the trace
//!   records one-row steps) and with `-n` under 3 (no pass).
//!
//! The `load` line's `top_k` is the file's indexer top-k: a position whose
//! whole pools hold at most `top_k` positions attends every position; past
//! that, the ones its selector lists. The levers it acts on are parsed once,
//! at `main`; every line is a record of a kind `bloomery_gpu_gates::record`
//! declares (`--records-schema` prints them).
//!
//! Under `BLOOMERY_DRAFT=mtp` the load carries the file's next-token layer
//! beside the target (`app::arch::glm5next::open_nextn`) and the generated
//! tokens run through the shared MTP window (`app::mtp::MtpDraft<Body>`,
//! `runtime::Speculative` of two rows): each window verifies the target's
//! next token and the draft's one proposal, the target's step mode as
//! `--mode` says, the draft's walks eager; the greedy ids are the plain
//! run's. In graph mode the verify of two rows is captured before the
//! prompt (a `capture` line of its nodes). Each kept token prints its `step`
//! line and, under `--time`, a `time step` line (its window's wall over the
//! window's kept rows, the row a plain step's wall compares with; `--warm W`
//! marks the first W), and each window a `time pass` line (its wall from the
//! proposal through the commit, the rows it kept, `kind=mtp`, or `plain` for
//! a window with no proposal). The feed's `time prompt` wall holds the
//! draft's store walks over the prompt's units. The `mtp summary` line counts the proposals
//! and the windows by rows kept; the `SMOKE` footer is over the kept
//! positions past the warm ones. `lookup` and `dspark` are refused by name,
//! as are `--pair`, `--logits` and the route trace beside the draft. Unset or
//! `off`, the load, the plan and every step are the plain run's.
//!
//! `BLOOMERY_RESIDENCY` runs adaptive expert residency (`host::swap`) over
//! the card's routed stacks: unset or `off`, the load's slot map for the
//! model's life; set, the word prints as a `residency lever` record before
//! the load, and `mid-p<P>-s<S>` opens the load under it
//! (`app::arch::glm5next::open_resident`): after the `plan` line, the
//! `residency host` record of the churn pool the host set also holds, the
//! word refused by name when the plan's fewest card experts a layer leave
//! no room for P pinned, S spares and one that moves; after the run's
//! lines, the `residency pass` record of every boundary (the prompt call's,
//! then each step's), none between two timed steps. Beside the MTP draft the
//! NextN load runs under it (`open_nextn` with the word): each window's
//! boundary ends a `pair` pass keeping the rows its commit accepted, or a
//! `step` pass for a window with no proposal, the draft's walks between a
//! commit and the next boundary noting no id; the next-token layer stays on
//! the host. Refused by name beside `--pair`, the route trace and
//! `--prefill steps`.
//!
//! `BLOOMERY_STEP_STATS=1` reads the host tier before the first generated
//! step and after each (`host_stats::Probe`) and prints, after `SMOKE`, a
//! `stat step` record a step past `--warm` and one `stat summary` (kinds
//! `stat_step_host`, `stat_summary_host`: each step's host leg, served slots, straggle and go-wait gaps). It also arms the batch walks' timing
//! (`bloomery_gpu_glm5next::set_prompt_stats`): after a batched prompt's
//! `step 0` line, one `stat prompt lb` a routed layer-batch and the `stat
//! prompt split` over them (`record::prompt_stats`; its `fault_ms` and
//! `ckpt_ms` the call's fault reads and checkpoint waits). Refused
//! by name beside the MTP draft.
//!
//! `BLOOMERY_PREFILL_GROUP` sets the batches a prompt group runs layer by
//! layer (`bloomery_gpu_glm5next::set_prefill_group`), the registry's 2
//! unset, whose units the plan reserves (`place::prompt_reserve_bytes`). The
//! `load` line prints it as `group=`.
//!
//! `BLOOMERY_ROUTE_TRACE=<dir>` writes the engine's route trace of the run
//! into `dir`, a new directory made before the load
//! (`crates/gpu/src/host/route_trace.rs`): every position's routed ids per
//! layer and the slot each ran in, as a router set. The prompt's ids run one
//! step each and are recorded as the call's positions — `--prefill steps` is
//! required (the batched call is a prompt batch, which the trace does not
//! record) and `--time` is refused, each by name. The trace holds the host
//! tier's routed run, past the dense lead, under the file's own layer
//! numbers, as a reference engine's set of the same file. The run ends with
//! `route trace <dir> positions=<n> complete` once the set is sealed.

#[cfg(not(feature = "glm5next"))]
fn main() {
    eprintln!("generate_glm5next: built without the `glm5next` feature; see `just gen-glm5next`.");
    std::process::exit(2);
}

#[cfg(feature = "glm5next")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("generate_glm5next", cli::run())
}

// The placement word the GLM body serves and its tier batch bytes.
#[cfg(feature = "glm5next")]
#[path = "shared/glm5next_place.rs"]
mod glm_place;

#[cfg(feature = "glm5next")]
mod cli {
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use crate::glm_place;
    use app::arch::glm5next::{GlmCfg, open_nextn, open_pair, open_resident};
    use app::mtp::MtpDraft;
    use app::{Loaded, OpenArgs, OpenLog, RowsLog, Session, SessionError};
    use bloomery_gpu::host::route_trace::{RouteTrace, TraceHeader};
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::generate::{Place, with_cards};
    use bloomery_gpu_gates::host_stats::{Probe, print_stats};
    use bloomery_gpu_gates::record::{self, Record};
    use bloomery_gpu_gates::residency38::{GLM_CARD, residency_room, residency_set};
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use bloomery_gpu_glm5next::{Body, Glm5nextModel, PrefillMode, prompt_bytes};
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, DRAFT, HOST_LOCK, HOST_POPULATE, LANE_PREFETCH, PREFILL_GROUP,
        R8, RESIDENCY, ROUTE_TRACE, ResidencyPick, ResidencyWhy, STEP_STATS,
    };
    use gguf::Split;
    use model::arch::glm5next::hparams::Hparams;
    use model::arch::glm5next::place::PlanInputs;
    use model::placement::{Machine, Plan, PlanLevers};
    use runtime::layer::hosted;
    use runtime::{Advance, Committed, PassSink, Stop, Target, Verify, Want};

    const ACTS_ON: &[&str] = &[
        CARD_BUDGET,
        HOST_POPULATE,
        HOST_LOCK,
        CARD_DONTNEED,
        R8,
        ROUTE_TRACE,
        DRAFT,
        RESIDENCY,
        STEP_STATS,
        PREFILL_GROUP,
        LANE_PREFETCH,
    ];

    /// The drafted window's verify: the target's next token and the draft's
    /// one proposal.
    const VERIFY_ROWS: usize = 2;

    /// The last value of flag `name`, if given.
    fn flag(name: &str) -> Result<Option<String>, GateError> {
        let args: Vec<String> = std::env::args().collect();
        let mut out = None;
        for (i, a) in args.iter().enumerate() {
            if a == name {
                out = Some(
                    args.get(i + 1)
                        .ok_or_else(|| format!("{name} needs a value"))?
                        .clone(),
                );
            }
        }
        Ok(out)
    }

    fn has(name: &str) -> bool {
        std::env::args().any(|a| a == name)
    }

    /// Comma-separated ids.
    fn ids_of(s: &str) -> Result<Vec<u32>, GateError> {
        Ok(s.split(',')
            .map(|v| v.trim().parse::<u32>())
            .collect::<Result<_, _>>()?)
    }

    fn on_off(on: bool) -> &'static str {
        if on { "on" } else { "off" }
    }

    fn mode_name(m: StepMode) -> &'static str {
        match m {
            StepMode::Eager => "eager",
            StepMode::Graph => "graph",
        }
    }

    /// What the open prints: the plan, the load, the capture.
    struct Log {
        /// The file's indexer top-k, which the plan read.
        top_k: usize,
        place: &'static str,
        /// The placement, whose cards the `load` record names.
        placement: Place,
        mode: StepMode,
        prefill: PrefillMode,
        /// Batches a prompt group runs.
        group: usize,
        ctx: usize,
        stop_at_plan: bool,
        t: Instant,
        /// `BLOOMERY_RESIDENCY` set to `mid-…`: its parse and its word.
        residency: Option<(Residency, &'static str)>,
        /// The host bytes the load holds beside the plan's: the NextN
        /// layer's host experts on a drafted load, else 0.
        beside: u64,
    }

    impl OpenLog<Body> for Log {
        fn beside(&mut self, bytes: u64) {
            self.beside = bytes;
        }

        fn plan(
            &mut self,
            place: &'static str,
            inputs: &PlanInputs,
            machine: &Machine,
            plan: &Plan<'_>,
        ) -> Result<bool, SessionError> {
            record::plan(place, machine, plan).print();
            self.top_k = inputs.hp.indexer.top_k;
            if let Some((r, word)) = self.residency {
                residency_room(plan, r, word).map_err(SessionError::Caller)?;
                residency_set(plan, GLM_CARD, r, word, self.beside, Record::print)
                    .map_err(SessionError::Caller)?;
            }
            Ok(!self.stop_at_plan)
        }

        fn load(&mut self, m: &Glm5nextModel) -> Result<(), SessionError> {
            let b = m.body("generate_glm5next")?;
            let r = Record::new(&record::LOAD_GENERATOR)
                .u("resident_bytes", m.resident_bytes())
                .u("ctx", self.ctx)
                .u("layers", b.kinds().len())
                .u("top_k", self.top_k)
                .w("shadow", "none")
                .u("shadow_bytes", 0)
                .u("unified_addressing", 0);
            with_cards(r, self.placement, m.gpu(), b.hybrid().tiers())
                .map_err(SessionError::Caller)?
                .w("prefill", self.prefill.name())
                .u("group", self.group)
                .w("lane_prefetch", on_off(model::ops::lane_prefetch()))
                .w("mode", mode_name(self.mode))
                .w("place", self.place)
                .w("pin_main", "off")
                .w("pinned", false)
                .f("load_s", self.t.elapsed().as_secs_f64())
                .print();
            if let Some(h) = b.hybrid().residency() {
                for r in record::host_residency(h) {
                    r.print();
                }
            }
            Ok(())
        }

        fn capture(&mut self, nodes: usize) -> Result<(), SessionError> {
            Record::new(&record::CAPTURE)
                .u("graph_nodes", nodes)
                .print();
            Ok(())
        }

        fn prompt_buffers(&mut self, m: &Glm5nextModel) -> Result<(), SessionError> {
            if let Some(b) = prompt_bytes(m)? {
                Record::new(&record::PROMPT_UNITS)
                    .u("shared_bytes", b.shared)
                    .u("unit_bytes", b.unit)
                    .u("units", b.units)
                    .u("hsum_bytes", b.hsum)
                    .u("free_bytes", b.free)
                    .print();
            }
            Ok(())
        }
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(ACTS_ON)?;
        model::ops::set_lane_prefetch(levers.lane_prefetch());
        record::at_main("generate_glm5next", record::GENERATE_GLM5NEXT);
        let ids = ids_of(&flag("--tokens")?.ok_or("--tokens a,b,c is required")?)?;
        if ids.is_empty() {
            return Err("--tokens holds no id".into());
        }
        let n_gen: usize = flag("-n")?.map_or(Ok(16), |s| s.parse())?;
        let timed = has("--time");
        let warm: usize = match flag("--warm")? {
            None => 0,
            Some(_) if !timed => {
                return Err("--warm drops steps from --time's footer; it needs --time".into());
            }
            Some(w) => w.parse()?,
        };
        if timed && n_gen < 2 {
            return Err(
                "--time needs -n 2 or more: token 0 comes out of the feed, and the steps after it are timed"
                    .into(),
            );
        }
        if timed && warm >= n_gen - 1 {
            return Err(format!(
                "--warm {warm} leaves no timed step of the {} that -n {n_gen} generates",
                n_gen - 1
            )
            .into());
        }
        let pair = has("--pair");
        let last_step = has("--last-step");
        if last_step && (timed || pair || matches!(levers.draft(), Some(d) if d != "off")) {
            return Err(
                "--last-step is refused beside --time, --pair and BLOOMERY_DRAFT: it feeds the \
                 plain run's prompt only"
                    .into(),
            );
        }
        if pair && n_gen < 3 {
            return Err(
                "--pair needs -n 3 or more: a verify runs two of the fed tokens and checks the next"
                    .into(),
            );
        }
        let drafted = match levers.draft() {
            None | Some("off") => false,
            Some("mtp") => true,
            Some(other) => {
                return Err(format!(
                    "BLOOMERY_DRAFT={other}: on a glm5next file mtp drafts the window (the \
                     file's NextN layer); lookup and dspark are the V4.1 binaries'"
                )
                .into());
            }
        };
        if drafted {
            if n_gen < 2 {
                return Err(
                    "BLOOMERY_DRAFT=mtp needs -n 2 or more: token 0 comes out of the feed, and the \
                     windows run after it"
                        .into(),
                );
            }
            let beside = if pair {
                Some("--pair (the plain verify probe runs without the draft)")
            } else if has("--logits") {
                Some("--logits (the drafted run's last call is a verify of two rows)")
            } else if levers.route_trace().is_some() {
                Some("BLOOMERY_ROUTE_TRACE (the trace records one-row steps)")
            } else if levers.step_stats() {
                Some("BLOOMERY_STEP_STATS (the probes read the plain run's steps)")
            } else {
                None
            };
            if let Some(what) = beside {
                return Err(format!("BLOOMERY_DRAFT=mtp is refused beside {what}").into());
            }
        }
        let ctx: usize = flag("--ctx")?.map_or(Ok(2048), |s| s.parse())?;
        let placement = match flag("--place")? {
            None => Place::Gate,
            Some(v) => glm_place::parse(&v)?,
        }
        .on_host()?;
        let place = placement.name();
        let mode = match flag("--mode")?.as_deref() {
            None | Some("graph") => StepMode::Graph,
            Some("eager") => StepMode::Eager,
            Some(o) => return Err(format!("--mode is eager or graph, not {o}").into()),
        };
        let prefill = match flag("--prefill")? {
            None => PrefillMode::Batch,
            Some(p) => PrefillMode::from_name(&p)
                .ok_or_else(|| format!("--prefill is batch or steps, not {p}"))?,
        };
        let residency = residency_of(&levers, pair, prefill)?;
        if has("--model") {
            return Err(
                "--model is not a flag: the file is $BLOOMERY_REF_MODEL, which \
                 tools/box.sh exports from the glm5next profile"
                    .into(),
            );
        }
        let path = ref_model_path()?
            .into_os_string()
            .into_string()
            .map_err(|p| format!("BLOOMERY_REF_MODEL is not UTF-8: {p:?}"))?;
        // The last generated token is read out, not fed: the run takes the
        // prompt's positions and one a step after the first token. A drafted
        // window's verify runs one row past the token it keeps last.
        let takes = ids.len() + n_gen.saturating_sub(1) + usize::from(drafted);
        if takes > ctx {
            return Err(format!(
                "{} prompt ids and {n_gen} generated take {takes} positions, past --ctx {ctx}",
                ids.len()
            )
            .into());
        }
        let t = Instant::now();
        let file = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let tier_batch = glm_place::tier_batch(placement, &Hparams::read(&file)?);
        let machine = placement.machine(None, tier_batch)?;
        let trace = trace_of(
            &levers,
            &file,
            prefill,
            place,
            &path,
            ids.len() + n_gen.saturating_sub(1),
            timed,
        )?;
        if trace.is_some() && pair {
            return Err(
                "BLOOMERY_ROUTE_TRACE records one-row steps: --pair's verifies are not; run the \
                 trace without --pair"
                    .into(),
            );
        }
        let group = levers.prefill_group();
        let cfg = GlmCfg {
            place: PlanLevers::from_levers(&levers)?,
            host: levers.host(),
            prefill,
            group,
        };
        let mut log = Log {
            top_k: 0,
            place,
            placement,
            mode,
            prefill,
            group,
            ctx,
            stop_at_plan: has("--plan"),
            t,
            residency,
            beside: 0,
        };
        let args = OpenArgs {
            place,
            machine,
            ctx,
            mode,
            cfg,
        };
        let mut s = if drafted {
            let lever = residency.map_or(Residency::Off, |(r, _)| r);
            let Some(s) = open_nextn(file, args, lever, &mut log)? else {
                return Ok(());
            };
            s
        } else if pair {
            let Some(s) = open_pair(file, args, &mut log)? else {
                return Ok(());
            };
            s
        } else if let Some((r, _)) = residency {
            let Some(s) = open_resident(file, args, r, &mut log)? else {
                return Ok(());
            };
            s
        } else {
            let Some(loaded) = Loaded::<Body>::open(file, args, &mut log)? else {
                return Ok(());
            };
            loaded.ready(&mut log)?
        };
        if let Some(t) = trace {
            s.model_mut()
                .body_parts("generate_glm5next")?
                .2
                .hybrid_mut()
                .attach_route_trace(t)?;
        }
        if residency.is_some() {
            // The prompt call's boundary, then one a generated step.
            s.model_mut()
                .body_parts("generate_glm5next")?
                .2
                .log_residency(n_gen + 1);
        }
        let passes = match prefill {
            PrefillMode::Batch => bloomery_gpu_glm5next::batches_of(s.model(), ids.len())?.len(),
            PrefillMode::Steps => ids.len(),
        };
        if drafted {
            let arm = Arm {
                ids: &ids,
                n_gen,
                timed,
                warm,
                mode,
                place,
                prefill,
                passes,
            };
            return drafted_run(&mut s, &arm);
        }
        if levers.step_stats() {
            bloomery_gpu_glm5next::set_prompt_stats(s.model_mut(), true)?;
        }
        fed(&ids);
        let t_feed = Instant::now();
        if prefill == PrefillMode::Steps {
            let pos = s.pos();
            s.model_mut()
                .body_parts("generate_glm5next")?
                .2
                .hybrid_mut()
                .route_prompt(pos, ids.len())?;
        }
        let mut next = match ids.split_last() {
            Some((&last, head)) if last_step => {
                if !head.is_empty() {
                    s.prompt(head, Want::Argmax)?;
                }
                s.step(last, Want::Argmax)?.argmax()
            }
            _ => s.prompt(&ids, Want::Argmax)?.argmax(),
        };
        let feed = t_feed.elapsed();
        Record::new(&record::STEP0)
            .u("pos", s.pos() - 1)
            .u("token", next)
            .u("fed", ids.len())
            .f("feed_s", feed.as_secs_f64())
            .print();
        if let Some(st) = bloomery_gpu_glm5next::take_prompt_stats(s.model_mut())? {
            for r in record::prompt_stats(&st) {
                r.print();
            }
        }
        let fed_end = s.pos();
        let mut tokens = vec![next];
        // (i, pos, token, ms): printed after the last step.
        let mut rows: Vec<(usize, u32, u32, f64)> = Vec::with_capacity(n_gen);
        let stats = levers.step_stats();
        let mut probes: Vec<Probe> = Vec::with_capacity(if stats { n_gen } else { 0 });
        if stats {
            probes.push(probe5(s.model(), None)?);
        }
        for i in 1..n_gen {
            let pos = s.pos();
            let t = Instant::now();
            next = s.step(next, Want::Argmax)?.argmax();
            rows.push((i, pos, next, t.elapsed().as_secs_f64() * 1e3));
            tokens.push(next);
            if stats {
                probes.push(probe5(s.model(), probes.last())?);
            }
        }
        let passes_ms = if pair {
            Some(pairs(&mut s, mode, fed_end, &tokens)?)
        } else {
            None
        };
        if timed {
            let ms = feed.as_secs_f64() * 1e3;
            Record::new(&record::TIME_PROMPT)
                .u("n", ids.len())
                .f("ms", ms)
                .f("tok/s", ids.len() as f64 * 1e3 / ms)
                .u("passes", passes)
                .w("kind", prefill.name())
                .print();
        }
        for &(i, pos, token, ms) in &rows {
            Record::new(&record::STEP)
                .u("i", i)
                .u("pos", pos)
                .u("token", token)
                .print();
            if timed {
                Record::new(&record::TIME_STEP)
                    .u("i", i)
                    .flag("warm", i <= warm)
                    .f("ms", ms)
                    .print();
            }
        }
        if let (true, Some(ms)) = (timed, &passes_ms) {
            for (k, &ms) in ms.iter().enumerate() {
                let i = k + 1;
                Record::new(&record::TIME_PASS)
                    .u("i", i)
                    .flag("warm", 2 * i - 1 <= warm)
                    .f("ms", ms)
                    .u("positions", 2)
                    .w("kind", "pair")
                    .print();
            }
        }
        Record::new(&record::TOKENS).list("tokens", &tokens).print();
        if has("--logits") {
            let row = s.model().logits()?;
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
            Record::new(&record::LOGITS)
                .u("n", row.len())
                .u("argmax", argmax)
                .w("fnv64", format!("{fnv:016x}"))
                .print();
        }
        if timed {
            let kept: Vec<f64> = rows[warm..].iter().map(|r| r.3).collect();
            let mut sorted = kept.clone();
            sorted.sort_by(f64::total_cmp);
            let p50 = sorted[sorted.len() / 2];
            let mean = kept.iter().sum::<f64>() / kept.len() as f64;
            Record::new(&record::SMOKE)
                .w("mode", mode_name(mode))
                .w("place", place)
                .u("prompt_tokens", ids.len())
                .u("depth", ids.len())
                .u("generated", n_gen)
                .u("warm", warm)
                .u("steps", kept.len())
                .f("p50_ms", p50)
                .f("mean_ms", mean)
                .f("tok/s(p50)", 1e3 / p50)
                .print();
        }
        print_stats(&probes, warm);
        for (kind, r) in s
            .model_mut()
            .body_parts("generate_glm5next")?
            .2
            .take_residency_passes()
        {
            record::residency_pass_of(kind, &r).print();
        }
        if let Some(t) = s
            .model_mut()
            .body_parts("generate_glm5next")?
            .2
            .hybrid_mut()
            .take_route_trace()
        {
            let dir = t.dir().to_path_buf();
            println!(
                "route trace {} positions={} complete",
                dir.display(),
                t.finish()?
            );
        }
        Ok(())
    }

    /// The `fed` record, printed just before the feed's timer starts, after
    /// every capture: the runner counts the timed window's faults from it.
    /// A [`Probe`] of the body's host tier, which a glm5next load always has.
    fn probe5(m: &Glm5nextModel, prev: Option<&Probe>) -> Result<Probe, GateError> {
        Probe::read(m.body("generate_glm5next")?.hybrid(), m.gpu(), prev)
    }

    fn fed(ids: &[u32]) {
        let head: Vec<u32> = ids.iter().copied().take(4).collect();
        let tail: Vec<u32> = ids
            .iter()
            .copied()
            .skip(ids.len().saturating_sub(4))
            .collect();
        Record::new(&record::FED)
            .u("ids", ids.len())
            .list("first", &head)
            .list("last", &tail)
            .u("depth_sequence_from", ids.len())
            .print();
    }

    /// What a drafted run is asked for.
    struct Arm<'a> {
        ids: &'a [u32],
        n_gen: usize,
        timed: bool,
        warm: usize,
        mode: StepMode,
        place: &'static str,
        prefill: PrefillMode,
        /// The feed's batches or steps, as the plain run counts them.
        passes: usize,
    }

    /// The verify pass's capture: its nodes, one line.
    struct PairCapture;

    impl RowsLog for PairCapture {
        fn capture_rows(&mut self, _rows: usize, nodes: usize) -> Result<(), SessionError> {
            Record::new(&record::CAPTURE_PAIR)
                .u("pair_graph_nodes", nodes)
                .print();
            Ok(())
        }
    }

    /// What the drafted generation's windows kept: every kept token at its
    /// position with its window's wall over the window's kept rows, and every
    /// window's proposal, kept rows and wall in ms.
    struct Windows {
        kept: Vec<(u32, u32, f64)>,
        passes: Vec<(bool, usize, f64)>,
    }

    impl PassSink<Session<Body>> for Windows {
        type Error = GateError;

        fn begin(&mut self, _: &Session<Body>) -> Result<(), GateError> {
            Ok(())
        }

        fn pass(
            &mut self,
            _: &Session<Body>,
            c: &Committed,
            tokens: &[u32],
            wall: Duration,
        ) -> Result<(), GateError> {
            let ms = wall.as_secs_f64() * 1e3;
            let per = ms / c.kept as f64;
            for (r, &token) in (0u32..).zip(tokens) {
                self.kept.push((c.pos + r, token, per));
            }
            self.passes.push((c.proposed, c.kept, ms));
            Ok(())
        }
    }

    /// `BLOOMERY_DRAFT=mtp`: the draft opened over the NextN load, in graph
    /// mode the verify of two rows captured before the `fed` record, the
    /// prompt fed with the draft's
    /// store walked over its units, then windows until `-n` tokens are out,
    /// every kept token the target's own argmax; the lines the module doc
    /// names, the `residency pass` records last.
    fn drafted_run(s: &mut Session<Body>, a: &Arm<'_>) -> Result<(), GateError> {
        let draft = MtpDraft::open(s.model(), a.prefill, StepMode::Eager)?;
        let mut spec = s.with_draft::<MtpDraft<Body>, VERIFY_ROWS>(draft, &mut PairCapture)?;
        fed(a.ids);
        let t_feed = Instant::now();
        let first = spec.prompt(s, a.ids)?;
        let feed = t_feed.elapsed();
        Record::new(&record::STEP0)
            .u("pos", s.pos() - 1)
            .u("token", first)
            .u("fed", a.ids.len())
            .f("feed_s", feed.as_secs_f64())
            .print();
        let mut sink = Windows {
            kept: Vec::with_capacity(a.n_gen + VERIFY_ROWS),
            passes: Vec::with_capacity(a.n_gen),
        };
        let stop = Stop::new(a.n_gen, s.ctx())?;
        let out = runtime::generate(s, &mut spec, a.ids, first, &stop, &mut sink)?;
        if out.tokens.len() < a.n_gen {
            return Err(format!(
                "generate_glm5next: the drafted run stopped at {} after {} tokens, before -n {}",
                out.stop.name(),
                out.tokens.len(),
                a.n_gen
            )
            .into());
        }
        // Generated token i >= 1 is kept token i - 1; the last window may
        // keep one past -n.
        let rows = &sink.kept[..a.n_gen - 1];
        if a.timed {
            let ms = feed.as_secs_f64() * 1e3;
            Record::new(&record::TIME_PROMPT)
                .u("n", a.ids.len())
                .f("ms", ms)
                .f("tok/s", a.ids.len() as f64 * 1e3 / ms)
                .u("passes", a.passes)
                .w("kind", a.prefill.name())
                .print();
        }
        for (i, &(pos, token, per)) in (1usize..).zip(rows) {
            Record::new(&record::STEP)
                .u("i", i)
                .u("pos", pos)
                .u("token", token)
                .print();
            if a.timed {
                Record::new(&record::TIME_STEP)
                    .u("i", i)
                    .flag("warm", i <= a.warm)
                    .f("ms", per)
                    .print();
            }
        }
        if a.timed {
            // Pass k's first kept position is generated token `at`.
            let mut at = 1;
            for (k, &(proposed, kept, ms)) in (1usize..).zip(&sink.passes) {
                Record::new(&record::TIME_PASS)
                    .u("i", k)
                    .flag("warm", at <= a.warm)
                    .f("ms", ms)
                    .u("positions", kept)
                    .w("kind", if proposed { "mtp" } else { "plain" })
                    .print();
                at += kept;
            }
        }
        Record::new(&record::TOKENS)
            .list("tokens", &out.tokens[..a.n_gen])
            .print();
        let counted: Vec<f64> = rows[a.warm.min(rows.len())..].iter().map(|r| r.2).collect();
        let counted_ms: f64 = counted.iter().sum();
        let rate = counted.len() as f64 * 1e3 / counted_ms;
        let mut kept = [0u64; VERIFY_ROWS];
        for &(_, k, _) in &sink.passes {
            kept[k - 1] += 1;
        }
        Record::new(&record::MTP_SUMMARY)
            .u("proposals", sink.passes.iter().filter(|p| p.0).count())
            .list("kept", &kept)
            .u("positions", sink.passes.iter().map(|p| p.1).sum::<usize>())
            .u("passes", sink.passes.len())
            .f("tok/s(positions)", rate)
            .print();
        if a.timed {
            let mut sorted = counted.clone();
            sorted.sort_by(f64::total_cmp);
            let p50 = sorted[sorted.len() / 2];
            let mean = counted_ms / counted.len() as f64;
            Record::new(&record::SMOKE)
                .w("mode", mode_name(a.mode))
                .w("place", a.place)
                .u("prompt_tokens", a.ids.len())
                .u("depth", a.ids.len())
                .u("generated", a.n_gen)
                .u("warm", a.warm)
                .u("steps", counted.len())
                .f("p50_ms", p50)
                .f("mean_ms", mean)
                .f("tok/s(p50)", 1e3 / p50)
                .u("positions", counted.len())
                .f("tok/s(positions)", rate)
                .print();
        }
        for (kind, r) in s
            .model_mut()
            .body_parts("generate_glm5next")?
            .2
            .take_residency_passes()
        {
            record::residency_pass_of(kind, &r).print();
        }
        Ok(())
    }

    /// `--pair`: the model cut back to `from`, the prompt's end, in graph
    /// mode the verify captured, then `(tokens.len() − 1) / 2` verifies of
    /// the plain run's fed tokens `tokens[2k]`, `tokens[2k + 1]`, each
    /// committing both rows and each giving the plain run's next two tokens
    /// or ending the run in a named error. Returns each pass's wall in ms,
    /// from its plan through its commit.
    fn pairs(
        s: &mut Session<Body>,
        mode: StepMode,
        from: u32,
        tokens: &[u32],
    ) -> Result<Vec<f64>, GateError> {
        s.cut(from)?;
        if mode == StepMode::Graph {
            let m = s.model_mut();
            m.capture_rows::<2>()?;
            Record::new(&record::CAPTURE_PAIR)
                .u("pair_graph_nodes", m.rows_graph_nodes::<2>()?.len())
                .print();
        }
        let n = (tokens.len() - 1) / 2;
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let pos = s.pos();
            let fed = [tokens[2 * k], tokens[2 * k + 1]];
            let t = Instant::now();
            let got = s.verify(fed)?;
            s.commit(2)?;
            out.push(t.elapsed().as_secs_f64() * 1e3);
            let want = [tokens[2 * k + 1], tokens[2 * k + 2]];
            if got != want {
                return Err(format!(
                    "pass {} at position {pos}: the verify of {fed:?} gave {got:?} where the steps \
                     gave {want:?}: the verify is not two steps",
                    k + 1
                )
                .into());
            }
        }
        Ok(out)
    }

    /// `BLOOMERY_RESIDENCY` as this run takes it: unset or `off`, `None` (the
    /// plain load); set to `mid-…`, its parse and word, its `residency lever`
    /// record printed. Refused by name beside `--pair` (its verifies run
    /// after a cut back to the prompt's end), the route trace (a fixed
    /// placement's routing) and the steps feed (each prompt id would end a
    /// pass the rule counts; the body refuses it at the feed, this before the
    /// load). Beside the MTP draft the NextN load takes it.
    fn residency_of(
        levers: &bloomery_levers::Levers,
        pair: bool,
        prefill: PrefillMode,
    ) -> Result<Option<(Residency, &'static str)>, GateError> {
        let Some(word) = levers.residency() else {
            return Ok(None);
        };
        let r = Residency::parse(word)?;
        record::residency_lever(ResidencyPick {
            word,
            why: ResidencyWhy::Set,
        })
        .print();
        if r == Residency::Off {
            return Ok(None);
        }
        let beside = if pair {
            Some("--pair (its verifies run after a cut back to the prompt's end)")
        } else if levers.route_trace().is_some() {
            Some("BLOOMERY_ROUTE_TRACE (the trace records a fixed placement's routing)")
        } else if prefill == PrefillMode::Steps {
            Some("--prefill steps (each prompt id would end a pass the rule counts)")
        } else {
            None
        };
        if let Some(what) = beside {
            return Err(format!("BLOOMERY_RESIDENCY={word} is refused beside {what}").into());
        }
        Ok(Some((r, word)))
    }

    /// The route trace `BLOOMERY_ROUTE_TRACE` asks for, its directory made
    /// here, before the load: every position's routed ids per layer and the
    /// slot each ran in (`crates/gpu/src/host/route_trace.rs`), the prompt's
    /// ids one step each recorded as the call's positions. The trace covers
    /// the routed layers the host tier serves, as many as its slot map holds
    /// ([`hosted`]). Refused by name under the batched feed — a batch is a
    /// prompt batch, which the trace does not record — and beside `--time`:
    /// the trace rewrites its manifest after every position, so a timed
    /// run's numbers would not be a measurement.
    fn trace_of(
        levers: &bloomery_levers::Levers,
        file: &Split,
        prefill: PrefillMode,
        place: &'static str,
        path: &str,
        chunk: usize,
        timed: bool,
    ) -> Result<Option<RouteTrace>, GateError> {
        let Some(dir) = levers.route_trace() else {
            return Ok(None);
        };
        if prefill != PrefillMode::Steps {
            return Err(
                "BLOOMERY_ROUTE_TRACE records one-row steps: pass --prefill steps (the batched \
                 call is a prompt batch, which the trace does not record)"
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
        let run = hosted(&inputs.spec.layers)
            .map_err(|e| format!("the file's layers hold no host run: {e}"))?;
        let header = TraceHeader {
            model: PathBuf::from(path),
            arch: "glm5next".to_owned(),
            build: "generate_glm5next".to_owned(),
            n_expert: inputs.hp.n_expert,
            n_used: inputs.hp.n_used,
            first_layer: run.start,
            n_layer: run.len(),
            extra: vec![
                ("place".to_owned(), place.to_owned()),
                (
                    "card_budget".to_owned(),
                    levers
                        .card_budget_bytes()
                        .map_or_else(|| "each card's own".to_owned(), |b| b.to_string()),
                ),
                ("prefill".to_owned(), prefill.name().to_owned()),
                ("chunk".to_owned(), chunk.to_string()),
            ],
        };
        Ok(Some(RouteTrace::create(dir, header)?))
    }
}
