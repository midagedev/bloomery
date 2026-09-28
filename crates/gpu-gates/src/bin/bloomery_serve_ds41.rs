//! `bloomery-serve-ds41` — the llama-server-compatible HTTP API on the V4.1 engine.
//!
//!     bloomery-serve-ds41 [--host 127.0.0.1] [--port 8080] [--place a|gate|bp]
//!                         [--ctx C] [--alias NAME] [--cache-ram MIB]
//!
//! The model is `$BLOOMERY_REF_MODEL`; its first shard gives the vocabulary,
//! the chat template (`tokenizer.chat_template`) and the default alias
//! (`general.name`). The plan line, then the `load`, host and `capture`
//! lines of [`Generator::open`] go to stderr as `generate_ds41` prints them,
//! then `listening on http://<addr>` once the model is loaded and the port is
//! bound (`--port 0` binds a free one). Sampling is the sampler crate's chain
//! with no repetition penalty; `temperature <= 0` is the engine's argmax, the
//! ids `generate_ds41 --tokens <the prompt's ids>` prints.
//!
//! `/props`' `engine` object carries the file's header facts and the printed
//! plan's resident bytes per device and class, each card named by its
//! nvidia-smi index — under `--place bp` (plan (b′): plan (a) on the A6000,
//! the 3090 an expert tier; see `generate_ds41`) the tier card's row too,
//! with no `layers`; when an index cannot be found the placement is left
//! out, with the reason on stderr. Under `bp` a prompt call is fed as
//! `BLOOMERY_PREFILL` says, the tier serving its experts' slots of each
//! batch, and a lost tier card is an engine error like any other: the
//! request's 500 and `/health`'s 503 carry its message, which names the card.
//!
//! The server serves the lesser of `--ctx` and the positions V4.1 is computed
//! at (`Hparams::candidate_free_positions`): `/props`' `n_ctx` is that number,
//! a prompt that long is a 400 before it reaches the engine, and generation
//! stops there with `truncated`.
//!
//! The prompt cache (llama-server's `--cache-ram`, in MiB; 0 turns it off)
//! holds the body's saved sequence states in host RAM. Its default is the
//! lesser of [`CACHE_RAM_CAP`] and half the host headroom the printed plan
//! leaves (host RAM less the host expert set, tables, shadows and reserves);
//! the `cache` record prints it, and the token a prompt call is cut at so a
//! later request keeps the start of a user message ([`USER_START`], with
//! whether the chat template writes it). Every cache event and every prefix
//! the body keeps less of than a request shares is a record line.
//!
//! An engine error ends the process: the request gets a 500, `/health` a 503
//! for a moment, then the crash block (card, position, error) goes to stderr
//! and the exit code is 70.
//!
//! `BLOOMERY_ROUTE_TRACE=<dir>` writes a route trace
//! (`bloomery_gpu::host::route_trace`) into `<dir>`, which `main` creates as
//! a new directory before the load: every position the engine runs, each
//! layer's routed ids and the slot each ran in, a `call` row per prompt call.
//! It needs the step feed (`BLOOMERY_PREFILL=steps`) and no draft, and is
//! refused by name otherwise. Its `# build` names this binary's version; the
//! commit is `/props`' `engine.version`.
//!
//! The levers it acts on (`serve_levers::ACTS_ON`) are parsed once, at
//! `main` (`bloomery_levers::at_main`), which refuses by name a lever set
//! outside them and a `BLOOMERY_*` name no registry row names; `--levers`
//! prints them with this process's values and exits. A prompt is fed the way
//! the engine holds `BLOOMERY_PREFILL`. The stderr lines named
//! above are records of the kinds `bloomery_gpu_gates::record` declares;
//! `--records-schema` prints those kinds and exits.
//!
//! The model opens as `generate_ds41`'s does (`app::Loaded`, then
//! `Loaded::ready`), and `BLOOMERY_DRAFT=lookup|dspark` serves its draft the
//! same way (`shared/ds41_draft.rs`): the DSpark draft's file is
//! `$BLOOMERY_DSPARK_MODEL` and its card `BLOOMERY_DSPARK_CARD` (the 3090
//! when unset; under `bp` the tier card, whose plan reserves the draft's
//! bytes, and the lever may name no other), loaded once, its `load draft=dspark` line after the `load`
//! line, the pair pass captured after the step. Every greedy token after a
//! request's first then comes out of a pass (`serve::Engine::advance`), and
//! a request that needs the logits row — `temperature` above 0 (llama-server's
//! default 0.8 included) or `ignore_eos` — is a 400 naming the field.
//! `timings` carry `draft_n` and `draft_n_accepted`, `/metrics` their sums,
//! and `/props`' `engine.draft` names the draft (its kind, file and device),
//! whose card's placement row holds its resident bytes as the class `draft`.
//!
//! The prompt cache under the DSpark draft: the draft keeps no state of its
//! own worth saving — its window ring is a function of the last
//! `attention.sliding_window` positions' features, which the target's
//! prompt call hands over again. A request that continues exactly where the
//! draft stands keeps every position the target keeps. Any other — a cut, a
//! saved state put back — starts the draft over, so the target keeps no
//! position past the request's shared prefix less a window: the prompt call
//! then carries the window's features to the draft, and the ids are the full
//! prompt's. The lookup draft rebuilds its n-gram tables from the target's
//! token history at a request's first token.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "bloomery-serve-ds41: built without the `deepseek41` feature; see `just gate-gpu-ds41-serve`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    match drive::run() {
        // EX_SOFTWARE: the engine, not the listener or the load, ended the run.
        Ok(serve::ServeError::Engine(f)) => {
            eprintln!("bloomery-serve-ds41: {f}");
            std::process::ExitCode::from(70)
        }
        Ok(e) => bloomery_gpu_gates::exit_with("bloomery-serve-ds41", Err(e.into())),
        Err(e) => bloomery_gpu_gates::exit_with("bloomery-serve-ds41", Err(e)),
    }
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_serve_levers.rs"]
mod serve_levers;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_dspark.rs"]
mod dspark;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_draft.rs"]
mod draft;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_place.rs"]
mod place;

#[cfg(feature = "deepseek41")]
mod drive {
    use std::any::Any;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Instant;

    use app::arch::deepseek41::{CardDraft, Ds41Cfg};
    use app::{Loaded, OpenLog, RowsLog, Session, SessionError};
    use bloomery_gpu::host::route_trace::{RouteTrace, TraceHeader};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_deepseek41::body::{self, Body, Deepseek41Model, PAIR_ROWS, SeqSnapshot};
    use bloomery_gpu_deepseek41::draft::DraftBody;
    use bloomery_gpu_gates::bind::{
        Ds41Engine, Seat, Vocab, model_props, nvidia_smi_index, placement_props, sampler_factory,
    };
    use bloomery_gpu_gates::generate::{Place, mode_name};
    use bloomery_gpu_gates::record::{self, Record};
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use gguf::Split;
    use model::arch::deepseek41::place::PlanInputs;
    use model::arch::dspark::DraftHparams;
    use model::placement::workstation::{self, TierBatchBytes};
    use model::placement::{HotList, Machine, Plan, PlanLevers};
    use runtime::{Committed, Lookup, Speculative, Target, Want};
    use serve::{
        CacheNote, DeviceProps, DraftProps, Drafted, EngineProps, FATAL_LINGER, PlacementProps,
        Saved, ServeError, Server, ServerConfig,
    };
    use tokenizer::Tokenizer;

    use crate::draft::{Draft, open_dspark};
    use crate::{dspark, place};

    const USAGE: &str = "usage: bloomery-serve-ds41 [--host H] [--port P] [--place a|gate|bp] \
                         [--ctx C] [--alias NAME] [--cache-ram MIB]";

    /// The token V4.1's chat template opens every user and tool message with.
    pub const USER_START: &str = "<｜User｜>";

    /// The most the prompt cache takes by default: llama-server's
    /// `--cache-ram` default.
    pub const CACHE_RAM_CAP: u64 = 8192 << 20;

    /// A prompt call is cut at a message start only this far past the call's
    /// start: below it the cut's second call costs more than the prefix a
    /// later request keeps saves.
    const SPLIT_MIN: usize = 64;

    /// The class `/props` files the draft's resident bytes under.
    const DRAFT_CLASS: &str = "draft";

    struct Args {
        host: String,
        port: u16,
        place: Place,
        ctx: usize,
        alias: Option<String>,
        /// `--cache-ram` in bytes; `None` takes the default.
        cache_ram: Option<u64>,
    }

    fn parse_args() -> Result<Args, GateError> {
        let mut a = Args {
            host: "127.0.0.1".to_owned(),
            port: 8080,
            place: Place::A,
            ctx: usize::try_from(workstation::CTX_MAX)?,
            alias: None,
            cache_ram: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            if flag == "--help" || flag == "-h" {
                return Err(USAGE.into());
            }
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value, or is unknown: {USAGE}"))?;
            match flag.as_str() {
                "--host" => a.host = v,
                "--port" => a.port = v.parse()?,
                "--place" => a.place = Place::parse(&v)?,
                "--ctx" => a.ctx = v.parse()?,
                "--alias" => a.alias = Some(v),
                "--cache-ram" => {
                    let mib: u64 = v.parse()?;
                    a.cache_ram = Some(
                        mib.checked_mul(1 << 20)
                            .ok_or_else(|| format!("--cache-ram {mib} MiB passes u64 bytes"))?,
                    );
                }
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        Ok(a)
    }

    /// Loads the model and serves until the listener or the engine fails;
    /// `Ok` carries why the server ended.
    pub fn run() -> Result<ServeError, GateError> {
        let levers = bloomery_levers::at_main(crate::serve_levers::ACTS_ON)?;
        record::at_main("bloomery-serve-ds41", record::BLOOMERY_SERVE_DS41);
        let a = parse_args()?;
        let cfg = body::OpenCfg::from_levers(&levers)?;
        let draft = Draft::from_levers(&levers)?;
        // The draft's file is read before the target's load, which takes a minute.
        let draft_file = match draft {
            Draft::Dspark => Some(dspark::draft_hparams()?),
            Draft::Off | Draft::Lookup => None,
        };
        let path = ref_model_path()?;
        let reserve = match &draft_file {
            Some((d, _)) => dspark::draft_reserve(a.place, d, &path)?,
            None => None,
        };
        let vocab =
            Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?.with_user_start(USER_START)?);
        let inv = gguf::inventory_of(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let template = inv
            .value("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{}: no tokenizer.chat_template", path.display()))?
            .to_owned();
        let name = inv
            .value("general.name")
            .and_then(|v| v.as_str())
            .unwrap_or("deepseek-v4.1")
            .to_owned();
        drop(inv);

        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::read(&split)?;
        let model = model_props(&split, &inputs.model);
        let tier_batch = place::tier_batch(a.place, &inputs.hp);
        drop(split);
        let trace = route_trace(&levers, &cfg, draft, a.place, &path, &inputs)?;
        let (card, placement, headroom) =
            print_plan(&inputs, a.place, reserve, tier_batch, a.ctx, &cfg.place)?;
        let cache_ram = a.cache_ram.unwrap_or_else(|| {
            u64::try_from(headroom / 2).map_or(0, |half| half.min(CACHE_RAM_CAP))
        });
        Record::new(&record::CACHE_CONFIG)
            .u("ram", cache_ram)
            .u("headroom", headroom)
            .w("user_start", USER_START)
            .w("in_template", template.contains(USER_START))
            .eprint();
        let props = EngineProps {
            model: Some(model),
            placement,
            ..EngineProps::default()
        };
        let open = SeatArgs {
            place: a.place,
            ctx: a.ctx,
            cfg,
            pin_main: levers.pin_main(),
            want_top_k: inputs.hp.indexer.top_k,
            n_layer: inputs.hp.n_layer,
            path: path.clone(),
            draft,
            draft_file,
            reserve,
            tier_batch,
            trace,
        };
        let engine = Ds41Engine::spawn(
            move || V41::open(open),
            inputs.hp.candidate_free_positions(),
            vocab,
            card,
            props,
            cache_ram,
        )?;

        let config = ServerConfig {
            model_alias: a.alias.unwrap_or(name),
            model_path: path.display().to_string(),
            chat_template: template,
            sampler: Some(sampler_factory()),
            fatal_linger: FATAL_LINGER,
            slot_save_path: None,
        };
        let server = Server::bind((a.host.as_str(), a.port), Box::new(engine), config)?;
        Record::new(&record::LISTENING)
            .w("place", a.place.name())
            .u("ctx", a.ctx)
            .w("addr", server.local_addr()?)
            .eprint();
        Ok(server.run())
    }

    /// The route trace `BLOOMERY_ROUTE_TRACE` asks for, its directory made
    /// now, before the load; refused by name under the batched feed or a
    /// draft, which it does not record.
    fn route_trace(
        levers: &bloomery_levers::Levers,
        cfg: &body::OpenCfg,
        draft: Draft,
        place: Place,
        path: &Path,
        inputs: &PlanInputs,
    ) -> Result<Option<RouteTrace>, GateError> {
        let Some(dir) = levers.route_trace() else {
            return Ok(None);
        };
        if cfg.body.prefill != body::PrefillMode::Steps {
            return Err(
                "BLOOMERY_ROUTE_TRACE records the step feed: set BLOOMERY_PREFILL=steps \
                        (the batched prompt call routes a CED layer only at the positions a later \
                        reader needs)"
                    .into(),
            );
        }
        if draft != Draft::Off {
            return Err(format!(
                "BLOOMERY_ROUTE_TRACE records one-row steps; BLOOMERY_DRAFT={} runs two-row passes",
                draft.name()
            )
            .into());
        }
        let hp = &inputs.hp;
        let yes = |b: bool| if b { "on" } else { "off" }.to_owned();
        let header = TraceHeader {
            model: path.to_path_buf(),
            arch: "deepseek41".to_owned(),
            build: format!(
                "bloomery-serve-ds41 {} (the commit is /props engine.version)",
                env!("CARGO_PKG_VERSION")
            ),
            n_expert: hp.experts.n_expert,
            n_used: hp.experts.n_used,
            n_layer: hp.n_layer,
            extra: vec![
                ("placement".to_owned(), place.name().to_owned()),
                (
                    "hot_list".to_owned(),
                    levers
                        .hot_list()
                        .map_or_else(|| "none".to_owned(), |p| p.display().to_string()),
                ),
                (
                    "card_budget".to_owned(),
                    levers
                        .card_budget_bytes()
                        .map_or_else(|| "each card's own".to_owned(), |b| b.to_string()),
                ),
                ("r8".to_owned(), yes(cfg.body.host.r8)),
                ("prefill".to_owned(), cfg.body.prefill.name().to_owned()),
            ],
        };
        Ok(Some(RouteTrace::create(dir, header)?))
    }

    /// The plan the engine is about to load under the placement's `levers`
    /// (with `reserve`, the DSpark draft's, and `tier_batch`, the tier's
    /// prompt-batch bytes, on its tier card), on stderr;
    /// returns its cards' names, the plan's placement for `/props` (`None`,
    /// and a line saying why, when a card's nvidia-smi index cannot be found)
    /// and the plan's host headroom in bytes.
    fn print_plan(
        inputs: &PlanInputs,
        place: Place,
        reserve: Option<u64>,
        tier_batch: Option<TierBatchBytes>,
        ctx: usize,
        levers: &PlanLevers,
    ) -> Result<(String, Option<PlacementProps>, i64), GateError> {
        let machine = place.machine(reserve, tier_batch)?(inputs.model.layers);
        let plan = inputs.plan(&machine, u64::try_from(ctx)?, levers)?;
        let hot_list = levers.hot.as_ref().map_or("none", HotList::path);
        record::plan(place.name(), &machine, &plan, hot_list).eprint();
        let gpus: Result<Vec<String>, String> = machine
            .all_cards()
            .map(|c| nvidia_smi_index(&c.name).map(|i| format!("GPU{i}")))
            .collect();
        let placement = gpus.and_then(|g| placement_props(&plan, &g));
        if let Err(e) = &placement {
            eprintln!("bloomery-serve-ds41: /props leaves the placement out: {e}");
        }
        let headroom = i64::try_from(plan.host.headroom_bytes).map_err(|_| {
            format!(
                "the plan's host headroom {} B passes i64",
                plan.host.headroom_bytes
            )
        })?;
        let cards: Vec<&str> = machine.all_cards().map(|c| c.name.as_str()).collect();
        Ok((cards.join("+"), placement.ok(), headroom))
    }

    const WHAT: &str = "bloomery-serve-ds41";

    /// What the engine thread opens the seat with.
    struct SeatArgs {
        place: Place,
        ctx: usize,
        cfg: body::OpenCfg,
        pin_main: bool,
        /// The file's `top_k` and layer count, from its headers.
        want_top_k: usize,
        n_layer: usize,
        path: PathBuf,
        draft: Draft,
        draft_file: Option<(Split, DraftHparams)>,
        /// The draft's reserve on the placement's tier card.
        reserve: Option<u64>,
        /// The tier's prompt-batch bytes on the placement's tier card.
        tier_batch: Option<TierBatchBytes>,
        /// The route trace, attached once the session is ready.
        trace: Option<RouteTrace>,
    }

    /// The draft the seat serves, verified by the pair pass.
    enum Served {
        Off,
        /// The lookup, and whether its context is the target's token history
        /// and the token the target stands before.
        Lookup(Speculative<Lookup, PAIR_ROWS>, bool),
        /// Boxed: the draft body is the size of its graphs and buffers.
        Dspark(Box<Speculative<CardDraft<DraftBody>, PAIR_ROWS>>),
    }

    /// The V4.1 session on the engine thread, and its draft.
    struct V41 {
        s: Session<Body>,
        /// The positions served (`--ctx`).
        ctx: usize,
        draft: Served,
    }

    /// A body's saved sequence state, as the server's prompt cache holds it,
    /// and the positions a request after its resume leaves the DSpark draft
    /// to feed (0 without it).
    struct Ds41Saved {
        state: SeqSnapshot,
        window: usize,
    }

    impl Saved for Ds41Saved {
        fn n_tokens(&self) -> usize {
            self.state.positions()
        }

        fn n_bytes(&self) -> u64 {
            self.state.bytes() as u64
        }

        /// A resume starts the DSpark draft over: a window less.
        fn keepable(&self, n: usize) -> usize {
            self.state.keep_point(n.saturating_sub(self.window))
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl V41 {
        /// The session by `a.place` (the `load`, host set and `capture`
        /// lines), the draft `a.draft` names on it (its `load draft=dspark`
        /// line before the capture, the pair pass's capture after it), on
        /// the calling thread, pinned to the dispatcher's cpu slot when asked.
        fn open(mut a: SeatArgs) -> Result<V41, GateError> {
            let trace = a.trace.take();
            let pinned = a.pin_main && threads::pool().pin_caller();
            let t = Instant::now();
            let file =
                Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
            let args = app::OpenArgs {
                place: a.place.name(),
                machine: a.place.machine(a.reserve, a.tier_batch)?,
                ctx: a.ctx,
                mode: StepMode::Graph,
                cfg: Ds41Cfg {
                    feed: a.cfg.body.prefill,
                    open: a.cfg.clone(),
                    card_timing: false,
                },
            };
            let mut log = Log { a: &a, pinned, t };
            let mut loaded = Loaded::<Body>::open(file, args, &mut log)?
                .ok_or("bloomery-serve-ds41: the open planned nothing")?;
            let spark = match &a.draft_file {
                Some(file) => {
                    let (d, load) =
                        open_dspark(&mut loaded, file, &a.path, WHAT, a.place, a.reserve)?;
                    load.eprint();
                    Some(d)
                }
                None => None,
            };
            let mut s = loaded.ready(&mut log)?;
            if let Some(t) = trace {
                s.model_mut()
                    .body_parts(WHAT)?
                    .2
                    .hybrid_mut()
                    .attach_route_trace(t)?;
            }
            let draft = match (a.draft, spark) {
                (Draft::Off, _) => Served::Off,
                (Draft::Lookup, _) => Served::Lookup(
                    s.with_draft::<_, PAIR_ROWS>(Lookup::new(), &mut log)?,
                    false,
                ),
                (Draft::Dspark, Some(d)) => {
                    Served::Dspark(Box::new(s.with_draft::<_, PAIR_ROWS>(d, &mut log)?))
                }
                (Draft::Dspark, None) => {
                    return Err("bloomery-serve-ds41: the DSpark draft did not load".into());
                }
            };
            Ok(V41 {
                s,
                ctx: a.ctx,
                draft,
            })
        }

        fn model(&self) -> &Deepseek41Model {
            self.s.model()
        }

        /// The DSpark draft's window: the positions a request that starts
        /// the draft over feeds it before its first proposal.
        fn window(&self) -> usize {
            match &self.draft {
                Served::Dspark(d) => d.draft().window(),
                Served::Off | Served::Lookup(..) => 0,
            }
        }

        /// The target stands elsewhere than its drafts were fed.
        fn moved(&mut self) {
            match &mut self.draft {
                Served::Off => {}
                Served::Lookup(_, follows) => *follows = false,
                Served::Dspark(d) => d.draft_mut().forget(),
            }
        }

        /// The lookup's context made the target's token history and `next`,
        /// the token the target stands before, unless it is that already.
        fn lookup_follows(&mut self, next: u32) -> Result<(), GateError> {
            let V41 { s, draft, .. } = self;
            if let Served::Lookup(spec, follows) = draft
                && !*follows
            {
                let l = spec.draft_mut();
                l.reset();
                for &id in s.model().body(WHAT)?.history() {
                    l.push(id);
                }
                l.push(next);
                *follows = true;
            }
            Ok(())
        }
    }

    /// A pass's draft counts: the proposal's ids, the ones kept.
    fn drafted(c: Committed) -> Drafted {
        if c.proposed {
            Drafted {
                proposed: c.rows - 1,
                accepted: c.kept - 1,
            }
        } else {
            Drafted::default()
        }
    }

    impl Seat for V41 {
        fn pos(&self) -> usize {
            self.s.pos() as usize
        }

        fn ctx_max(&self) -> usize {
            self.ctx
        }

        fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
            match &mut self.draft {
                Served::Dspark(d) => Ok(d.draft_mut().feed_call(&mut self.s, ids)?),
                Served::Lookup(_, follows) => {
                    *follows = false;
                    Ok(self.s.prompt(ids, Want::Argmax)?.argmax())
                }
                Served::Off => Ok(self.s.prompt(ids, Want::Argmax)?.argmax()),
            }
        }

        fn step(&mut self, last: u32) -> Result<u32, GateError> {
            let next = self.s.step(last, Want::Argmax)?.argmax();
            match &mut self.draft {
                Served::Lookup(spec, true) => spec.draft_mut().push(next),
                Served::Dspark(d) => {
                    runtime::Draft::stepped(d.draft_mut(), &mut self.s, last, next)?;
                }
                Served::Lookup(_, false) | Served::Off => {}
            }
            self.lookup_follows(next)?;
            Ok(next)
        }

        fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
            Ok(self.model().logits_into(row)?)
        }

        fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
            self.lookup_follows(last)?;
            let c = match &mut self.draft {
                Served::Lookup(spec, _) => runtime::Advance::pass(spec, &mut self.s, last, out)?,
                Served::Dspark(spec) => {
                    runtime::Advance::pass(&mut **spec, &mut self.s, last, out)?
                }
                Served::Off => {
                    out.push(self.step(last)?);
                    return Ok(Drafted::default());
                }
            };
            Ok(drafted(c))
        }

        fn pass_rows(&self) -> usize {
            match self.draft {
                Served::Off => 1,
                Served::Lookup(..) | Served::Dspark(_) => PAIR_ROWS,
            }
        }

        fn reset(&mut self) -> Result<(), GateError> {
            self.s.reset()?;
            match &mut self.draft {
                Served::Off => {}
                Served::Lookup(_, follows) => *follows = false,
                Served::Dspark(d) => d.draft_mut().restart()?,
            }
            Ok(())
        }

        fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
            self.moved();
            Ok(self.s.model_mut().rollback(pos)?)
        }

        /// The body's rule; under the DSpark draft, a cut anywhere but where
        /// the draft stands also leaves the request's prompt call a window.
        fn keep(&self, n: usize) -> (usize, Option<String>) {
            let b = match self.model().body(WHAT) {
                Ok(b) => b,
                Err(e) => return (0, Some(e.to_string())),
            };
            let continues = match &self.draft {
                Served::Dspark(d) => {
                    u32::try_from(n).is_ok_and(|n| n == self.s.pos() && d.draft().follows(n))
                }
                Served::Off | Served::Lookup(..) => true,
            };
            if continues {
                let (k, why) = b.keep_why(n);
                return (k, why.map(|w| w.to_string()));
            }
            let window = self.window();
            let (k, why) = b.keep_why(n.saturating_sub(window));
            let why = format!(
                "the DSpark draft starts over, so the prompt call carries its window of \
                 {window} positions: cut to at most {}{}",
                n.saturating_sub(window),
                why.map(|w| format!(", then {w}")).unwrap_or_default()
            );
            (k, Some(why))
        }

        /// The body's cuts under the batched feed; the step feed leaves no
        /// hole and needs none.
        fn splits(&self, first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
            match self.model().body(WHAT) {
                Ok(b) if b.prefill_mode() == body::PrefillMode::Batch => {
                    b.prefill_splits(first, end, marks, SPLIT_MIN)
                }
                _ => Vec::new(),
            }
        }

        fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
            let window = self.window();
            Ok(Arc::new(Ds41Saved {
                state: body::snapshot(self.s.model_mut())?,
                window,
            }))
        }

        fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError> {
            let saved = state
                .as_any()
                .downcast_ref::<Ds41Saved>()
                .ok_or("a saved state that is not a V4.1 body's")?;
            self.moved();
            Ok(body::resume(self.s.model_mut(), &saved.state)?)
        }

        /// `engine.draft`, and the draft's resident bytes in its card's
        /// placement row, the class `draft` (a row of its own when the target
        /// has no layer there). The card's index missing leaves the device out.
        fn props(&self, mut p: EngineProps) -> EngineProps {
            match &self.draft {
                Served::Off => {}
                Served::Lookup(..) => {
                    p.draft = Some(DraftProps {
                        model: Draft::Lookup.name().to_owned(),
                        n_max: Some(1),
                        kind: Some(Draft::Lookup.name().to_owned()),
                        ..DraftProps::default()
                    });
                }
                Served::Dspark(spec) => {
                    let d = spec.draft();
                    let device = nvidia_smi_index(d.card()).ok().map(|i| format!("GPU{i}"));
                    let path = dspark::draft_path().ok();
                    let file = path
                        .as_deref()
                        .and_then(Path::file_name)
                        .map(|f| f.to_string_lossy().into_owned());
                    if let (Some(dev), Some(pl)) = (&device, p.placement.as_mut()) {
                        add_draft_bytes(pl, dev, d.resident_bytes() as u64);
                    }
                    p.draft = Some(DraftProps {
                        model: file.unwrap_or_else(|| Draft::Dspark.name().to_owned()),
                        n_max: Some(1),
                        kind: Some(Draft::Dspark.name().to_owned()),
                        path: path.map(|p| p.display().to_string()),
                        device,
                    });
                }
            }
            p
        }

        fn note(note: &CacheNote) {
            let r = match note {
                CacheNote::Reuse {
                    common,
                    ask,
                    kept,
                    held,
                    reason,
                } => Record::new(&record::CACHE_REUSE)
                    .u("common", common)
                    .u("ask", ask)
                    .u("kept", kept)
                    .u("held", held)
                    .w("reason", reason.as_deref().unwrap_or("unstated")),
                CacheNote::Save {
                    positions,
                    bytes,
                    ms,
                    entries,
                    cache_bytes,
                } => Record::new(&record::CACHE_SAVE)
                    .u("positions", positions)
                    .u("bytes", bytes)
                    .f("ms", *ms)
                    .u("entries", entries)
                    .u("cache_bytes", cache_bytes),
                CacheNote::Load {
                    positions,
                    common,
                    kept,
                    slot_kept,
                    bytes,
                    ms,
                } => Record::new(&record::CACHE_LOAD)
                    .u("positions", positions)
                    .u("common", common)
                    .u("kept", kept)
                    .u("slot_kept", slot_kept)
                    .u("bytes", bytes)
                    .f("ms", *ms),
                CacheNote::Evict {
                    positions,
                    bytes,
                    why,
                } => Record::new(&record::CACHE_EVICT)
                    .u("positions", positions)
                    .u("bytes", bytes)
                    .w("why", why),
                CacheNote::Skip { positions, why } => Record::new(&record::CACHE_SKIP)
                    .u("positions", positions)
                    .w("why", why),
                CacheNote::Split { first, end, at } => Record::new(&record::PREFILL_SPLIT)
                    .u("first", first)
                    .u("end", end)
                    .csv("at", at),
            };
            r.eprint();
        }
    }

    /// `bytes` of the draft on `device`: into that card's row, or a row of its
    /// own before the host's.
    fn add_draft_bytes(p: &mut PlacementProps, device: &str, bytes: u64) {
        if let Some(d) = p.devices.iter_mut().find(|d| d.device == device) {
            *d.class_bytes.entry(DRAFT_CLASS.to_owned()).or_default() += bytes;
            return;
        }
        let at = p
            .devices
            .iter()
            .position(|d| d.device == "CPU")
            .unwrap_or(p.devices.len());
        p.devices.insert(
            at,
            DeviceProps {
                device: device.to_owned(),
                class_bytes: [(DRAFT_CLASS.to_owned(), bytes)].into(),
                layers: None,
            },
        );
    }

    /// What the open prints and checks: the body's selection against the
    /// file's, the `load` record and the host set's, the captures.
    struct Log<'a> {
        a: &'a SeatArgs,
        pinned: bool,
        t: Instant,
    }

    impl OpenLog<Body> for Log<'_> {
        /// The plan was printed before the engine thread started.
        fn plan(
            &mut self,
            _place: &'static str,
            _inputs: &PlanInputs,
            _machine: &Machine,
            _plan: &Plan<'_>,
        ) -> Result<bool, SessionError> {
            Ok(true)
        }

        fn load(&mut self, m: &Deepseek41Model) -> Result<(), SessionError> {
            let a = self.a;
            let b = m.body(WHAT)?;
            let top_k = b.indexer_top_k();
            let shadow = b.shadow_host();
            if top_k != a.want_top_k {
                return Err(SessionError::Refused(format!(
                    "the body selects {top_k} rows per stream, the file's top_k is {}: a step \
                     past that many visible rows would not be the model's",
                    a.want_top_k
                )));
            }
            let load = Record::new(&record::LOAD_GENERATOR)
                .u("resident_bytes", m.resident_bytes())
                .u("ctx", a.ctx)
                .u("layers", a.n_layer)
                .u("top_k", top_k)
                .w("shadow", "host")
                .u("shadow_bytes", shadow.bytes)
                .u("unified_addressing", shadow.unified_addressing);
            place::with_cards(m, a.place, WHAT, load)
                .map_err(|e| SessionError::Refused(e.to_string()))?
                .w("prefill", b.prefill_mode().name())
                .w("mode", mode_name(StepMode::Graph))
                .w("place", a.place.name())
                .w("pin_main", if a.pin_main { "on" } else { "off" })
                .w("pinned", self.pinned)
                .f("load_s", self.t.elapsed().as_secs_f64())
                .eprint();
            if let Some(h) = b.hybrid().residency() {
                for r in record::host_residency(h) {
                    r.eprint();
                }
            }
            Ok(())
        }

        fn capture(&mut self, nodes: usize) -> Result<(), SessionError> {
            Record::new(&record::CAPTURE)
                .u("graph_nodes", nodes)
                .eprint();
            Ok(())
        }

        fn prompt_buffers(&mut self, _m: &Deepseek41Model) -> Result<(), SessionError> {
            Ok(())
        }
    }

    impl RowsLog for Log<'_> {
        fn capture_rows(&mut self, rows: usize, nodes: usize) -> Result<(), SessionError> {
            if rows != PAIR_ROWS {
                return Err(SessionError::Refused(format!(
                    "a verify capture of {rows} rows; the pair pass runs {PAIR_ROWS}"
                )));
            }
            Record::new(&record::CAPTURE_PAIR)
                .u("pair_graph_nodes", nodes)
                .eprint();
            Ok(())
        }
    }
}
