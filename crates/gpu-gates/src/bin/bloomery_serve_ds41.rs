//! `bloomery-serve-ds41` — the llama-server-compatible HTTP API on the V4.1 engine.
//!
//!     bloomery-serve-ds41 [--host 127.0.0.1] [--port 8080] [--place a|gate]
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
//! plan's resident bytes per device and class, the card named by its
//! nvidia-smi index; when that index cannot be found the placement is left
//! out, with the reason on stderr.
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
//! The levers it acts on (`serve_levers::ACTS_ON`) are parsed once, at
//! `main` (`bloomery_levers::at_main`), which refuses by name a lever set
//! outside them and a `BLOOMERY_*` name no registry row names; `--levers`
//! prints them with this process's values and exits. A prompt is fed the way
//! the engine holds `BLOOMERY_PREFILL`. The stderr lines named
//! above are records of the kinds `bloomery_gpu_gates::record` declares;
//! `--records-schema` prints those kinds and exits.

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
mod drive {
    use std::any::Any;
    use std::sync::Arc;

    use bloomery_gpu::GpuError;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_deepseek41::body::{self, Body, Deepseek41Model, SeqSnapshot};
    use bloomery_gpu_gates::bind::{
        BodyOps, Ds41Engine, Vocab, model_props, nvidia_smi_index, placement_props, sampler_factory,
    };
    use bloomery_gpu_gates::generate::{Generator, OpenArgs, Place};
    use bloomery_gpu_gates::record::{self, Record};
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use gguf::Split;
    use model::arch::deepseek41::place::PlanInputs;
    use model::placement::{HotList, PlanLevers, workstation};
    use serve::{
        CacheNote, EngineProps, FATAL_LINGER, PlacementProps, Saved, ServeError, Server,
        ServerConfig,
    };
    use tokenizer::Tokenizer;

    const USAGE: &str = "usage: bloomery-serve-ds41 [--host H] [--port P] [--place a|gate] \
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
        let path = ref_model_path()?;
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
        drop(split);
        let (card, placement, headroom) = print_plan(&inputs, a.place, a.ctx, &cfg.place)?;
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
        let want_top_k = inputs.hp.indexer.top_k;
        let defined = inputs.hp.candidate_free_positions();
        let n_layer = inputs.hp.n_layer;
        let open = OpenArgs {
            place: a.place,
            ctx: a.ctx,
            mode: StepMode::Graph,
            pin_main: levers.pin_main(),
        };
        let engine = Ds41Engine::spawn(
            move || {
                let mut g = Generator::open(
                    open,
                    |file, machine, ctx| body::open(file, machine, ctx, &cfg),
                    |m: &Deepseek41Model, load: Record| {
                        let body = m.body("bloomery-serve-ds41")?;
                        let top_k = body.indexer_top_k();
                        let shadow = body.shadow_host();
                        if top_k != want_top_k {
                            return Err(format!(
                                "the body selects {top_k} rows per stream, the file's top_k is \
                                 {want_top_k}: a step past that many visible rows would not be \
                                 the model's"
                            )
                            .into());
                        }
                        Ok(load
                            .u("layers", n_layer)
                            .u("top_k", top_k)
                            .w("shadow", "host")
                            .u("shadow_bytes", shadow.bytes)
                            .u("unified_addressing", shadow.unified_addressing)
                            .w("prefill", body.prefill_mode().name()))
                    },
                    &mut std::io::stderr(),
                )?;
                let mode = g.model_mut().body("bloomery-serve-ds41")?.prefill_mode();
                if mode == body::PrefillMode::Batch {
                    body::prepare_prefill(g.model_mut())?;
                }
                Ok(g)
            },
            V41,
            defined,
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

    /// The plan the engine is about to load under the placement's `levers`,
    /// on stderr; returns the card's name, the plan's placement for `/props`
    /// (`None`, and a line saying why, when a card's nvidia-smi index cannot
    /// be found) and the plan's host headroom in bytes.
    fn print_plan(
        inputs: &PlanInputs,
        place: Place,
        ctx: usize,
        levers: &PlanLevers,
    ) -> Result<(String, Option<PlacementProps>, i64), GateError> {
        let machine = place.machine()(inputs.model.layers);
        let plan = inputs.plan(&machine, u64::try_from(ctx)?, levers)?;
        let hot_list = levers.hot.as_ref().map_or("none", HotList::path);
        record::plan(place.name(), &machine, &plan, hot_list).eprint();
        let gpus: Result<Vec<String>, String> = machine
            .cards
            .iter()
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
        Ok((machine.cards[0].name.to_string(), placement.ok(), headroom))
    }

    const WHAT: &str = "bloomery-serve-ds41";

    /// The V4.1 body's side of the engine thread.
    struct V41;

    /// A body's saved sequence state, as the server's prompt cache holds it.
    struct Ds41Saved(SeqSnapshot);

    impl Saved for Ds41Saved {
        fn n_tokens(&self) -> usize {
            self.0.positions()
        }

        fn n_bytes(&self) -> u64 {
            self.0.bytes() as u64
        }

        fn keepable(&self, n: usize) -> usize {
            self.0.keep_point(n)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl BodyOps<Body> for V41 {
        fn keep(&self, m: &Deepseek41Model, n: usize) -> (usize, Option<String>) {
            match m.body(WHAT) {
                Ok(b) => {
                    let (k, why) = b.keep_why(n);
                    (k, why.map(|w| w.to_string()))
                }
                Err(e) => (0, Some(e.to_string())),
            }
        }

        fn prefill(&self, m: &mut Deepseek41Model, ids: &[u32]) -> Result<u32, GpuError> {
            match m.body(WHAT)?.prefill_mode() {
                body::PrefillMode::Batch => body::prefill(m, ids),
                body::PrefillMode::Steps => m.step(ids),
            }
        }

        /// The body's cuts under the batched feed; the step feed leaves no
        /// hole and needs none.
        fn splits(
            &self,
            m: &Deepseek41Model,
            first: usize,
            end: usize,
            marks: &[usize],
        ) -> Vec<usize> {
            match m.body(WHAT) {
                Ok(b) if b.prefill_mode() == body::PrefillMode::Batch => {
                    b.prefill_splits(first, end, marks, SPLIT_MIN)
                }
                _ => Vec::new(),
            }
        }

        fn snapshot(&self, m: &mut Deepseek41Model) -> Result<Arc<dyn Saved>, GpuError> {
            Ok(Arc::new(Ds41Saved(body::snapshot(m)?)))
        }

        fn resume(&self, m: &mut Deepseek41Model, state: &dyn Saved) -> Result<(), GpuError> {
            let s = state
                .as_any()
                .downcast_ref::<Ds41Saved>()
                .ok_or(GpuError::State {
                    what: WHAT,
                    missing: "a V4.1 body's saved state",
                })?;
            body::resume(m, &s.0)
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
}
