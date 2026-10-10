//! `bloomery-serve --model mimo2` — the llama-server-compatible HTTP API on
//! the MiMo-V2.6-Flash engine: a mimo2 file, its layers on one card with
//! every routed expert on the host tier (`model::placement::plan_host_routed`:
//! the program runs no card expert), opened as `gate_mimo2_e2e` opens it
//! (`app::Loaded`, in graph mode with the step captured).
//!
//!     bloomery-serve --model mimo2 [-m PATH | --hf <repo>[:<quant>]]
//!                    [--host 127.0.0.1] [--port 8080] [--ctx C] [--alias NAME]
//!                    [--chat-template-file PATH] [--place W] [--plan]
//!                    [--prefill batch|steps] [--parallel 1] [--queue-depth Q]
//!                    [--api-key KEY] [--api-key-file FNAME]
//!
//! The server takes `-m`/`--hf` out before this seat parses (its module
//! doc); `--ctx-size` and `-c` are `--ctx` under llama-server's spellings. The
//! model's first shard gives the vocabulary, `tokenizer.chat_template` the chat
//! template (`--chat-template-file` replaces it) and `general.name` the alias
//! (`--alias` replaces it; `mimo-v2.6-flash` for a file that names none).
//! Sampling is the sampler crate's chain with no repetition penalty,
//! `temperature <= 0` the engine's argmax. Any other flag is refused as an
//! unknown argument.
//!
//! Three facts of the architecture set this seat's shape, and nothing else
//! does: the body holds one sequence (no `Slots`), the program runs no verify
//! pass of several rows, and no draft is read. So the server feeds the prompt
//! less its last id as one prompt call, then steps the last, as
//! `bloomery-serve --model qwen3` does; `--parallel` beyond one slot and a
//! prompt cache are refused by name, and a tier card has no expert to hold.
//!
//! The prompt call is fed as `--prefill` says (`app::Prompt for Body`): in
//! batches of up to a union's columns (`batch`, the default), or one decode
//! step an id (`steps`, the same-binary arm); the two leave the same bits. A
//! word other than those is refused naming the flag and the word. The batch
//! feed's buffers come out of the card's free bytes at the load, refused by
//! name when they do not fit, and the group lever is not read: a group runs
//! one batch.
//!
//! `--place W` is the shared placement word (`generate::Place`): `a` (the
//! largest visible card), `gate`, or one card of the census by name or
//! ordinal. Unset, the common rule every serving seat takes decides
//! (`Place::choose_untiered`: the body serves no tier card), on one census
//! reading: a `place unset` record names the word `a` and why. A placement
//! that hangs an expert tier card (`bp`, a two-card list) is refused before
//! any load by the plan's own refusal, which names the card and the all-host
//! plan; the tier serves no prompt batch here, so its batch reserve is zero
//! bytes.
//!
//! `--ctx` is one request's context, the one slot's whole (`--ctx 0` is
//! refused). Unset, it is the file's trained context (`<arch>.context_length`)
//! capped to the largest whose plan stands, searched on a grid of
//! `generate::CTX_GRAN` positions from the floor [`CTX`]; a file that states no
//! trained context takes the floor, and one stderr line names a cap that
//! binds. Every routed expert sits on the host, so no card-expert bytes trade
//! against the context and the expert-margin guard (`placement::ctx::
//! within_margin`) has nothing to bound; the plan's own fit is the cap.
//!
//! The records on stderr, in order: `place unset` (word and why), one `ctx`
//! line (the rule — `set`, `trained`, `card` or `floor` —, the context, the
//! slot, the total and the trained context), the `plan` record, the `parallel`
//! line (`rule=slots slots=1 slot_ctx=C total=C from=<word>`); then the `load`
//! record (architecture, resident bytes, context, slots, layers, the
//! routed experts the plan keeps on the host and on the card, the step graph's
//! nodes and the load's wall) and, once the port is bound, the `listening`
//! record (`record::BLOOMERY_SERVE_MIMO2`). `--plan` ends the process after
//! the `parallel` line, nothing loaded. An engine error ends the process with
//! the crash block and exit code 70, as every seat's.
//!
//! A request keeps the longest prefix it shares with what the slot holds
//! (`app::Keep for Body`): the caches are per-position and the body takes its
//! rollback as given, so every held position is keepable and a resend that
//! diverges at `j` re-feeds from `j` and nothing before it. No prompt call is
//! cut (`Seat::splits` marks nowhere). The seat runs no prompt cache: a save
//! and a resume are refused by name.
//!
//! The levers this seat acts on ([`ACTS_ON`]) are parsed once, at `main`
//! (`bloomery_levers::at_main`), which refuses by name a lever set outside
//! them: the plan's card budget, the host set's load settings, and
//! `BLOOMERY_PIN_MAIN` with `BLOOMERY_HOST_LANES` (the engine thread pinned to
//! the dispatcher's cpu slot, which gives the host legs the pool's CCD map). The seat
//! sits behind the `deepseek41` feature beside `mimo2`: the server surface it
//! binds through (`bind`, the `serve` and `sampler` crates) is scoped to
//! `deepseek41`; it runs no V4.1 code.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use app::arch::mimo2::Mimo2Cfg;
use app::{Loaded, OpenArgs, OpenLog, Session, SessionError};
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::bind::{
    ChatSurface, Listen, Seat, SeatEngine, Vocab, bind_server, model_props, note_line,
    parallel_line, placement_of, slots_given,
};
use bloomery_gpu_gates::generate::{CTX_GRAN, Place, trained_ctx};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::{GateError, gpu_census, ref_model_path};
use bloomery_gpu_mimo2::{Body, Mimo2Model, PrefillMode};
use bloomery_levers::HostCfg;
use gguf::Split;
use model::arch::Arch;
use model::arch::mimo2::place::PlanInputs;
use model::placement::workstation::TierBatchBytes;
use model::placement::{Machine, Plan, PlanLevers};
use runtime::Target as _;
use serve::flag::number;
use serve::{CacheNote, EngineProps, Saved, ServeError};
use tokenizer::Tokenizer;

const NAME: &str = "bloomery-serve-mimo2";

const USAGE: &str = "usage: bloomery-serve --model mimo2 [-m PATH | --hf <repo>[:<quant>]] \
                     [--host H] [--port P] [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--place W] [--plan] [--prefill batch|steps] [--parallel 1] \
                     [--queue-depth Q] [--api-key KEY] [--api-key-file FNAME]";

/// The levers this seat acts on: the plan's card budget and the host set's
/// load settings, which the open reads (`PlanLevers::from_levers`,
/// `Levers::host`), the engine thread's pin and the host legs' lane rule.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_LANES,
];

/// The floor of the searched default context.
const CTX: usize = 4096;

/// Why the seat takes no sequence state: it runs no prompt cache.
const NO_CACHE: &str = "the mimo2 seat runs no prompt cache";

/// MiMo fact: the program runs no card expert, so a tier card holds nothing
/// and serves no prompt batch. Planning a placement that names one with zero
/// batch bytes reaches the plan's own refusal.
const NO_BATCH: TierBatchBytes = TierBatchBytes {
    staging: 0,
    scratch: 0,
    host: 0,
};

struct Args {
    host: String,
    port: u16,
    ctx: Option<usize>,
    place: Option<Place>,
    alias: Option<String>,
    /// `--chat-template-file`, replacing the file's own template.
    template_file: Option<PathBuf>,
    /// `--plan`: the records before the load, then exit.
    plan_only: bool,
    /// `--prefill`: how a prompt call is fed.
    prefill: PrefillMode,
    /// `--parallel`: only 1 is served.
    parallel: Option<usize>,
    queue_depth: Option<usize>,
    /// `--api-key`/`--api-key-file`: the keys every request is checked
    /// against ([`serve::flag::ApiKeys`]).
    api_keys: serve::flag::ApiKeys,
}

fn parse_args(args: &[String]) -> Result<Args, GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        ctx: None,
        place: None,
        alias: None,
        template_file: None,
        plan_only: false,
        prefill: PrefillMode::Batch,
        parallel: None,
        queue_depth: None,
        api_keys: serve::flag::ApiKeys::default(),
    };
    let mut it = args.iter().map(String::as_str);
    while let Some(flag) = it.next() {
        if flag == "--help" || flag == "-h" {
            return Err(USAGE.into());
        }
        if flag == "--plan" {
            a.plan_only = true;
            continue;
        }
        let v = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value, or is unknown: {USAGE}"))?;
        match flag {
            "--host" => a.host = v.to_owned(),
            "--port" => a.port = number(flag, v)?,
            f if serve::flag::CTX.contains(&f) => a.ctx = Some(number(flag, v)?),
            "--place" => a.place = Some(Place::parse(v)?),
            "--alias" => a.alias = Some(v.to_owned()),
            "--chat-template-file" => a.template_file = Some(PathBuf::from(v)),
            "--prefill" => {
                a.prefill = PrefillMode::from_name(v)
                    .ok_or_else(|| format!("--prefill is batch or steps, not {v}"))?;
            }
            "--parallel" | "-np" => a.parallel = Some(number(flag, v)?),
            "--queue-depth" => a.queue_depth = Some(number(flag, v)?),
            f if serve::flag::KEYS.contains(&f) => a.api_keys.add(f, v)?,
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    if a.ctx == Some(0) {
        return Err("--ctx 0: the stores hold no position".into());
    }
    match a.parallel {
        Some(0) => return Err("--parallel 0: the server serves no slot".into()),
        Some(n) if n > 1 => {
            // MiMo fact: the body holds one sequence, and a second would
            // need its own caches; the prompt cache is not served either.
            return Err(format!(
                "--parallel {n}: the mimo2 seat serves one slot (the body holds one sequence) \
                 and runs no prompt cache either, so no request waits on another's state; \
                 give --parallel 1 or none"
            )
            .into());
        }
        _ => {}
    }
    Ok(a)
}

/// What decided the seat's context, for its `ctx` line.
struct CtxRule {
    ctx: usize,
    /// `set` (`--ctx`), `trained` (the file's own context stands), `card`
    /// (the plan stands to a context under it) or `floor` ([`CTX`]: a file
    /// that states no trained context, or a plan that stands nowhere at the
    /// floor, which the load's own plan call then refuses by name).
    rule: &'static str,
    trained: Option<usize>,
}

/// The seat's context: `set` as given, else the file's trained context capped
/// to the largest on the [`CTX_GRAN`] grid whose plan stands
/// (`placement::ctx::searched`). The plan counts one sequence of the whole
/// context.
fn ctx_of(
    set: Option<usize>,
    split: &Split,
    inputs: &PlanInputs,
    machine: &Machine,
    levers: &PlanLevers,
) -> Result<CtxRule, GateError> {
    let trained = trained_ctx(split);
    if let Some(ctx) = set {
        return Ok(CtxRule {
            ctx,
            rule: "set",
            trained,
        });
    }
    let fits = |c: usize| -> Result<bool, GateError> {
        Ok(u64::try_from(c).is_ok_and(|c| inputs.plan(machine, c, levers).is_ok()))
    };
    let (ctx, rule) = match model::placement::ctx::searched(trained, CTX, CTX_GRAN, &fits)? {
        Some(c) if trained.is_some_and(|t| c >= t) => (c, "trained"),
        Some(c) => (c, "card"),
        None => (CTX, "floor"),
    };
    Ok(CtxRule { ctx, rule, trained })
}

/// The MiMo session on the engine thread and the positions its stores were
/// sized for.
struct Mimo {
    s: Session<Body>,
    ctx: usize,
}

/// What the engine thread opens the seat with.
struct OpenSeat {
    place: Place,
    ctx: usize,
    path: PathBuf,
    plan: PlanLevers,
    host: HostCfg,
    pin_main: bool,
    prefill: PrefillMode,
}

/// The open's records: the plan's expert counts and the model's size are
/// kept until the capture, where the step graph's nodes complete the `load`
/// record.
struct Log {
    t: Instant,
    ctx: usize,
    host_experts: u64,
    card_experts: u64,
    resident_bytes: usize,
    layers: usize,
    /// `BLOOMERY_PIN_MAIN`'s ask and whether the pin took.
    pin_main: bool,
    pinned: bool,
}

impl OpenLog<Body> for Log {
    fn plan(
        &mut self,
        _place: &'static str,
        _inputs: &PlanInputs,
        _machine: &Machine,
        plan: &Plan<'_>,
    ) -> Result<bool, SessionError> {
        self.host_experts = plan.host.experts;
        self.card_experts = plan.cards[0].experts;
        Ok(true)
    }

    fn load(&mut self, m: &Mimo2Model) -> Result<(), SessionError> {
        self.resident_bytes = m.resident_bytes();
        self.layers = m.layers().len();
        Ok(())
    }

    fn capture(&mut self, nodes: usize) -> Result<(), SessionError> {
        Record::new(&record::LOAD_MIMO2)
            .w("arch", "mimo2")
            .u("resident_bytes", self.resident_bytes)
            .u("ctx", self.ctx)
            .u("slots", 1)
            .u("layers", self.layers)
            .u("host_experts", self.host_experts)
            .u("card_experts", self.card_experts)
            .u("graph_nodes", nodes)
            .w("pin_main", if self.pin_main { "on" } else { "off" })
            .w("pinned", self.pinned)
            .f("load_s", self.t.elapsed().as_secs_f64())
            .eprint();
        Ok(())
    }

    fn prompt_buffers(&mut self, _m: &Mimo2Model) -> Result<(), SessionError> {
        Ok(())
    }
}

impl Mimo {
    /// The session of the file at `a.path` on `a.place`, the step captured,
    /// the `load` record on stderr, on the calling thread, pinned to the
    /// dispatcher's cpu slot when asked.
    fn open(a: OpenSeat) -> Result<Mimo, GateError> {
        let pinned = a.pin_main && threads::pool().pin_caller();
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        let tier = (!a.place.tier_cards().is_empty()).then_some(NO_BATCH);
        let mut log = Log {
            t: Instant::now(),
            ctx: a.ctx,
            pin_main: a.pin_main,
            pinned,
            host_experts: 0,
            card_experts: 0,
            resident_bytes: 0,
            layers: 0,
        };
        let args = OpenArgs {
            place: a.place.name(),
            machine: a.place.machine(None, tier)?,
            ctx: a.ctx,
            mode: StepMode::Graph,
            cfg: Mimo2Cfg {
                place: a.plan,
                host: a.host,
                prefill: a.prefill,
                group: 1,
            },
        };
        let s = Loaded::<Body>::open(file, args, &mut log)?
            .ok_or("the open stopped at its plan")?
            .ready(&mut log)?;
        Ok(Mimo { s, ctx: a.ctx })
    }
}

impl Seat for Mimo {
    fn pos(&self) -> usize {
        self.s.model().pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The prompt as `--prefill` says (`app::Prompt for Body`), the argmax
    /// after the last.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        Ok(<Body as app::Prompt>::prompt(self.s.model_mut(), ids)?)
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        Ok(self.s.model_mut().step(&[last])?)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    /// The model's reset ([`GpuModel::reset`]): the body never holds a
    /// verify's rows, so the session's own reset adds nothing to it.
    fn reset(&mut self) -> Result<(), GateError> {
        Ok(self.s.model_mut().reset()?)
    }

    /// The session's cut (`app::Keep`), which takes only a position
    /// [`Seat::keep`] granted; any other is refused by name.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        Ok(self.s.cut(pos)?)
    }

    /// Every held position (`app::Keep for Body`), the session's answer.
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        self.s.keep_query(n)
    }

    /// Nowhere: the caches are per-position, so a cut never needs the calls
    /// on either side of it to be whole prompt calls of their own.
    fn splits(&self, _first: usize, _end: usize, _marks: &[usize]) -> Vec<usize> {
        Vec::new()
    }

    /// Never asked: the seat's prompt cache budget is 0.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        Err(format!("a snapshot: {NO_CACHE}").into())
    }

    fn resume(&mut self, _state: &dyn Saved) -> Result<(), GateError> {
        Err(format!("a resume: {NO_CACHE}").into())
    }

    fn note(note: &CacheNote) {
        note_line(NAME, note);
    }
}

/// Loads the model and serves until the listener or the engine fails;
/// `Ok` carries why the server ended.
pub fn run(args: &[String]) -> Result<ServeError, GateError> {
    let levers = bloomery_levers::at_main(ACTS_ON)?;
    model::ops::set_host_lanes(levers.host_lanes());
    record::at_main(NAME, record::BLOOMERY_SERVE_MIMO2);
    let a = parse_args(args)?;
    let path = ref_model_path()?;
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let first = split
        .shard(0)
        .ok_or_else(|| format!("{} opened with no shard", path.display()))?;
    let arch = Arch::detect(first).map_err(|e| format!("{}: {e}", path.display()))?;
    if !matches!(arch, Arch::Mimo2) {
        return Err(format!(
            "{} is a {} file; the mimo2 seat serves mimo2 files",
            path.display(),
            arch.name()
        )
        .into());
    }
    let ChatSurface { template, name } = ChatSurface::read(
        &path,
        split.value("tokenizer.chat_template"),
        split.value("general.name"),
        a.template_file.as_deref(),
        "mimo-v2.6-flash",
    )?;
    let inputs = PlanInputs::read(&split)?;
    let plan_levers = PlanLevers::from_levers(&levers)?;
    // The common unset rule, once, on one census reading.
    let census = gpu_census::census()?;
    let chosen = Place::choose_untiered(a.place, &census, "mimo2")?;
    chosen.record().eprint();
    let place = chosen.place;
    let tier = (!place.tier_cards().is_empty()).then_some(NO_BATCH);
    let machine = place.machine(None, tier)?(inputs.model.layers);
    let rule = ctx_of(a.ctx, &split, &inputs, &machine, &plan_levers)?;
    let ctx = rule.ctx;
    // The plan the load runs by: a tier card, a context the card cannot hold
    // and a budget that binds are refused here by name, before any load.
    let plan = inputs
        .plan(&machine, u64::try_from(ctx)?, &plan_levers)
        .map_err(|e| format!("--place {}: {e}", place.name()))?;
    if let (Some(trained), "card") = (rule.trained, rule.rule) {
        eprintln!(
            "{NAME}: --ctx defaults to {ctx} of the file's {trained} trained positions (the plan \
             stands to there; pass --ctx to choose)"
        );
    }
    eprintln!(
        "ctx rule={} ctx={ctx} slots=1 total={ctx} trained={}",
        rule.rule,
        rule.trained
            .map_or_else(|| "none".to_owned(), |t| t.to_string())
    );
    record::plan(place.name(), &machine, &plan).eprint();
    let (slots, from) = slots_given(a.parallel, a.ctx, 1)?;
    parallel_line(slots, ctx, from, None);
    if a.plan_only {
        // The records before the load are out; nothing was opened on a card.
        std::process::exit(0);
    }
    let props = EngineProps {
        model: Some(model_props(&split, &inputs.model)),
        placement: placement_of(NAME, &machine, &plan),
        ..EngineProps::default()
    };
    let card = machine.cards[0].name.clone();
    drop(plan);
    drop(split);
    let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?);
    let open = OpenSeat {
        place,
        ctx,
        path: path.clone(),
        plan: plan_levers,
        host: levers.host(),
        pin_main: levers.pin_main(),
        prefill: a.prefill,
    };
    let engine = SeatEngine::spawn(move || Mimo::open(open), ctx, vocab, card, props, 0)?;
    let server = bind_server(
        engine,
        Listen {
            host: &a.host,
            port: a.port,
            alias: a.alias.unwrap_or(name),
            path: &path,
            template,
            slot_save_path: None,
            api_keys: a.api_keys,
            parallel: slots,
            queue_depth: a.queue_depth,
        },
    )?;
    Record::new(&record::LISTENING_MIMO2)
        .w("place", place.name())
        .u("ctx", ctx)
        .u("slots", slots)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}
