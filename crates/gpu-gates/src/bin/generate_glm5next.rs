//! Greedy generation with the GLM-5.3-Flash program: the file planned onto
//! one card, the routed experts the card experts read by the expert rule
//! (each layer's id prefix) and the rest on the
//! host tier (`model::arch::glm5next::place`), loaded through the session
//! (`app::Loaded`), the prompt fed as `--prefill` says, then `-n` greedy
//! steps.
//!
//! `generate_glm5next --tokens a,b,c [-n N] [--ctx C] [--place a|gate]
//! [--mode graph|eager] [--prefill batch|steps] [--time [--warm W]]
//! [--pair] [--logits] [--plan]`
//!
//! - `--tokens`: the prompt's ids (the file's own vocabulary, no BOS added).
//! - `--ctx`: the positions the caches hold; the plan refuses more than the
//!   deepest context a reference set checks the selector at
//!   (`place::ORACLE_POSITIONS`). Default 2048.
//! - `--place`: `a` is the serving plan (`workstation::plan_a`, the A6000),
//!   `gate` the gate card's (`workstation::plan_gate`, the 3090). Default
//!   `gate`.
//! - `--prefill`: `batch` feeds the prompt in batches
//!   (`bloomery_gpu_glm5next::prefill`), `steps` one decode step a position,
//!   the same bits at any `--ctx`, past the positions the latent layers
//!   attend whole too; the same-binary arm. Default `batch`.
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
//!   (`runtime::Verify` on the session, `bloomery_gpu_glm5next`'s verify):
//!   the model cut back to the prompt's end (its checkpoint), in graph mode
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

#[cfg(feature = "glm5next")]
mod cli {
    use std::path::PathBuf;
    use std::time::Instant;

    use app::arch::glm5next::GlmCfg;
    use app::{Loaded, OpenArgs, OpenLog, Session, SessionError};
    use bloomery_gpu::host::route_trace::{RouteTrace, TraceHeader};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_gates::record::{self, Record};
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use bloomery_gpu_glm5next::{Body, Glm5nextModel, PrefillMode};
    use bloomery_levers::{CARD_BUDGET, CARD_DONTNEED, HOST_LOCK, HOST_POPULATE, R8, ROUTE_TRACE};
    use gguf::Split;
    use model::arch::glm5next::place::PlanInputs;
    use model::placement::{Machine, Plan, PlanLevers, workstation};
    use runtime::layer::hosted;
    use runtime::{Target, Verify, Want};

    const ACTS_ON: &[&str] = &[
        CARD_BUDGET,
        HOST_POPULATE,
        HOST_LOCK,
        CARD_DONTNEED,
        R8,
        ROUTE_TRACE,
    ];

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
        mode: StepMode,
        prefill: PrefillMode,
        ctx: usize,
        stop_at_plan: bool,
        t: Instant,
    }

    impl OpenLog<Body> for Log {
        fn plan(
            &mut self,
            place: &'static str,
            inputs: &PlanInputs,
            machine: &Machine,
            plan: &Plan<'_>,
        ) -> Result<bool, SessionError> {
            record::plan(place, machine, plan).print();
            self.top_k = inputs.hp.indexer.top_k;
            Ok(!self.stop_at_plan)
        }

        fn load(&mut self, m: &Glm5nextModel) -> Result<(), SessionError> {
            let b = m.body("generate_glm5next")?;
            Record::new(&record::LOAD_GENERATOR)
                .u("resident_bytes", m.resident_bytes())
                .u("ctx", self.ctx)
                .u("layers", b.kinds().len())
                .u("top_k", self.top_k)
                .w("shadow", "none")
                .u("shadow_bytes", 0)
                .u("unified_addressing", 0)
                .w("prefill", self.prefill.name())
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

        fn prompt_buffers(&mut self, _m: &Glm5nextModel) -> Result<(), SessionError> {
            Ok(())
        }
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(ACTS_ON)?;
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
        if pair && n_gen < 3 {
            return Err(
                "--pair needs -n 3 or more: a verify runs two of the fed tokens and checks the next"
                    .into(),
            );
        }
        let ctx: usize = flag("--ctx")?.map_or(Ok(2048), |s| s.parse())?;
        let (place, machine): (&'static str, fn(usize) -> Machine) =
            match flag("--place")?.as_deref() {
                None | Some("gate") => ("gate", workstation::plan_gate),
                Some("a") => ("a", workstation::plan_a),
                Some(o) => return Err(format!("--place is a or gate, not {o}").into()),
            };
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
        // prompt's positions and one a step after the first token.
        let takes = ids.len() + n_gen.saturating_sub(1);
        if takes > ctx {
            return Err(format!(
                "{} prompt ids and {n_gen} generated take {takes} positions, past --ctx {ctx}",
                ids.len()
            )
            .into());
        }
        let t = Instant::now();
        let file = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
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
        let cfg = GlmCfg {
            place: PlanLevers::from_levers(&levers)?,
            host: levers.host(),
            prefill,
        };
        let mut log = Log {
            top_k: 0,
            place,
            mode,
            prefill,
            ctx,
            stop_at_plan: has("--plan"),
            t,
        };
        let args = OpenArgs {
            place,
            machine,
            ctx,
            mode,
            cfg,
        };
        let Some(loaded) = Loaded::<Body>::open(file, args, &mut log)? else {
            return Ok(());
        };
        let mut s = loaded.ready(&mut log)?;
        if let Some(t) = trace {
            s.model_mut()
                .body_parts("generate_glm5next")?
                .2
                .hybrid_mut()
                .attach_route_trace(t)?;
        }
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
        let passes = match prefill {
            PrefillMode::Batch => bloomery_gpu_glm5next::batches_of(s.model(), ids.len())?.len(),
            PrefillMode::Steps => ids.len(),
        };
        let t_feed = Instant::now();
        if prefill == PrefillMode::Steps {
            let pos = s.pos();
            s.model_mut()
                .body_parts("generate_glm5next")?
                .2
                .hybrid_mut()
                .route_prompt(pos, ids.len())?;
        }
        let mut next = s.prompt(&ids, Want::Argmax)?.argmax();
        let feed = t_feed.elapsed();
        Record::new(&record::STEP0)
            .u("pos", s.pos() - 1)
            .u("token", next)
            .u("fed", ids.len())
            .f("feed_s", feed.as_secs_f64())
            .print();
        let fed_end = s.pos();
        let mut tokens = vec![next];
        // (i, pos, token, ms): printed after the last step.
        let mut rows: Vec<(usize, u32, u32, f64)> = Vec::with_capacity(n_gen);
        for i in 1..n_gen {
            let pos = s.pos();
            let t = Instant::now();
            next = s.step(next, Want::Argmax)?.argmax();
            rows.push((i, pos, next, t.elapsed().as_secs_f64() * 1e3));
            tokens.push(next);
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
