//! `bloomery-serve-ds41` — the llama-server-compatible HTTP API on the V4.1 engine.
//!
//! This module is the seat: `bloomery-serve-ds41` and `bloomery-serve --model
//! ds41` are each one call of [`run`], which takes the process's arguments
//! (`--model` already taken out by the one-binary server).
//!
//!     bloomery-serve-ds41 [--host 127.0.0.1] [--port 8080] [--place PLACE]
//!                         [--ctx C] [--alias NAME] [--cache-ram MIB]
//!                         [--parallel N] [--queue-depth Q] [--slot-save-path DIR]
//!
//! `PLACE` is `a`, `gate`, `bp` or a card list `<stage>[+<tier>…]`
//! (`generate::Place`, as `generate_ds41` takes it); a list whose stage card
//! is not the A6000 and a list of more tier cards than the V4.1 body serves
//! are refused by name before any plan.
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
//! Each slot serves `--ctx` positions (below): that is `/props`' `n_ctx`, a
//! prompt that long is a 400 before it reaches the engine, and generation
//! stops there with `truncated`.
//! `/props`' `engine.ctx_verified` is the
//! deepest context the reference sets hold our numbers to ik's at
//! (`refset::arch::deepseek41::VERIFIED_POSITIONS`), which bounds nothing.
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
//! `--parallel N` (`-np N`, default [`SLOTS`]) serves N resident sequences
//! inside the one model (`Session::add_slots` over the body's `Slots`): the
//! server steps the busy slots together, a select and a step a slot each
//! round, the sequences switched by pointer exchange, so each request's
//! tokens are its solo run's; nothing parks and no slot waits for another.
//! `--parallel 1` is exactly the one-sequence server, and `--parallel 0` is
//! refused by name. Every slot holds a whole context: each serves the full
//! `--ctx` positions, refused by name before the load below the pair pass's
//! rows ([`slot_ctx`]). The plan counts every sequence — each one's caches on
//! the card and ring shadows on the host (`PlanInputs::plan_with_slots`) — so
//! slots the machine cannot hold are refused before the load by the plan's
//! own refusal, as one sequence's are (the card's floor or its usable bytes
//! past its dense trunk, KV, context, scratch, reserves and margin; the
//! host's usable bytes past its tensors, ring shadows and reserves); and
//! `--park-ram` is refused by name: resident slots hold their state in the
//! plan. A `parallel` line on stderr names the rule (`slots`), the slots, a
//! slot's context, the total and what set the count (`flag`, `default`, or
//! one slot under the DSpark draft and under a route trace); a `slots` line
//! after the load names each sequence's card and page-locked host bytes.
//! `--queue-depth Q` bounds the requests that wait for a slot. Several slots
//! are refused by name under the DSpark draft (its window is one sequence's,
//! on its own card: a select would start it over) and under
//! `BLOOMERY_ROUTE_TRACE` (the trace records one sequence's steps after its
//! call row); the lookup draft rebuilds its tables from the selected slot's
//! history at that slot's next pass, so its passes are the slot's alone.
//!
//! `--slot-save-path DIR` turns on the slot actions as llama-server's: an
//! erase empties that slot alone; a save or a restore to a file is refused
//! by name (the engine answers 501: the body's sequence state goes to the
//! prompt cache in host RAM, not to a file).
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
//! refused by name otherwise. Its `# build` names the server's version with
//! its commit, `/props`' `engine.version` before the engine's note.
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
//! default 0.8 included) or `ignore_eos` — takes plain steps, each step's row
//! the target's (the draft writes only its own card's buffers or host tables).
//! A greedy request's `timings` carry `draft_n` and `draft_n_accepted`,
//! `/metrics` their sums,
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

use std::any::Any;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use app::arch::deepseek41::{CardDraft, Ds41Cfg};
use app::{Loaded, OpenLog, RowsLog, Session, SessionError};
use bloomery_gpu::host::route_trace::{RouteTrace, TraceHeader};
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::{Slots, StepMode};
use bloomery_gpu_deepseek41::body::{self, Body, Deepseek41Model, PAIR_ROWS, SeqSnapshot};
use bloomery_gpu_deepseek41::draft::DraftBody;
use bloomery_gpu_deepseek41::swap;
use bloomery_gpu_gates::bind::{
    Seat, SeatEngine, Vocab, model_props, nvidia_smi_index, placement_props, sampler_factory,
};
use bloomery_gpu_gates::generate::{Place, mode_name};
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::{GateError, ref_model_path};
use bloomery_levers::{ResidencyAt, ResidencyPick, ResidencyWhy};
use gguf::Split;
use model::arch::deepseek41::place::PlanInputs;
use model::arch::dspark::DraftHparams;
use model::placement::workstation::{self, TierBatchBytes};
use model::placement::{Machine, Plan, PlanLevers};
use refset::arch::deepseek41::VERIFIED_POSITIONS;
use runtime::{Committed, Lookup, Speculative, Target, Want};
use serve::{
    CacheNote, DeviceProps, DraftProps, Drafted, EngineProps, FATAL_LINGER, PlacementProps,
    ResidencyReset, Saved, ServeError, Server, ServerConfig, SlotConfig,
};
use tokenizer::Tokenizer;

use crate::draft::{Draft, open_dspark};
use crate::{dspark, place};

const USAGE: &str = "usage: bloomery-serve-ds41 [--host H] [--port P] \
                     [--place a|gate|bp|<stage>[+<tier>…]] [--ctx C] [--alias NAME] \
                     [--cache-ram MIB] [--parallel N] [--queue-depth Q] \
                     [--slot-save-path DIR]";

/// The resident sequences the seat serves when `--parallel` is not given.
pub const SLOTS: usize = 2;

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
    /// `--parallel`: the resident sequences the seat serves; `None` takes
    /// [`slot_count`]'s default.
    parallel: Option<usize>,
    queue_depth: Option<usize>,
    /// `--slot-save-path`: the directory the slot actions answer from;
    /// `None` refuses every one, as llama-server does.
    slot_save_path: Option<PathBuf>,
}

fn parse_args(args: &[String]) -> Result<Args, GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        place: Place::A,
        ctx: usize::try_from(workstation::CTX_MAX)?,
        alias: None,
        cache_ram: None,
        parallel: None,
        queue_depth: None,
        slot_save_path: None,
    };
    let mut park_ram = false;
    let mut it = args.iter().map(|s| s.as_str());
    while let Some(flag) = it.next() {
        if flag == "--help" || flag == "-h" {
            return Err(USAGE.into());
        }
        let v = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value, or is unknown: {USAGE}"))?;
        match flag {
            "--host" => a.host = v.to_owned(),
            "--port" => a.port = v.parse()?,
            "--place" => a.place = place::parse(v)?,
            "--ctx" => a.ctx = v.parse()?,
            "--alias" => a.alias = Some(v.to_owned()),
            "--cache-ram" => a.cache_ram = Some(mib_bytes(flag, v)?),
            "--parallel" | "-np" => a.parallel = Some(v.parse()?),
            "--queue-depth" => a.queue_depth = Some(v.parse()?),
            "--slot-save-path" => a.slot_save_path = Some(PathBuf::from(v)),
            "--park-ram" => {
                mib_bytes(flag, v)?;
                park_ram = true;
            }
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    if a.parallel == Some(0) {
        return Err("--parallel 0: the server serves no slot".into());
    }
    if park_ram {
        return Err(
            "--park-ram holds the states of slots that take the model in turns; this seat's \
             slots are resident sequences, which park nothing — the plan counts their state"
                .into(),
        );
    }
    a.place = a.place.on_host()?;
    Ok(a)
}

/// A flag's value in MiB, as bytes.
fn mib_bytes(flag: &str, v: &str) -> Result<u64, GateError> {
    let mib: u64 = v.parse()?;
    Ok(mib
        .checked_mul(1 << 20)
        .ok_or_else(|| format!("{flag} {mib} MiB passes u64 bytes"))?)
}

/// The resident sequences the seat serves and what set the count:
/// `parallel` (`--parallel`) as given, else [`SLOTS`] — one under the DSpark
/// draft, whose window is one sequence's on its own card, and one under a
/// route trace (`traced`), which records one sequence's steps. Several under
/// the DSpark draft are refused by name here; under a route trace,
/// [`route_trace`] refuses them.
fn slot_count(
    parallel: Option<usize>,
    draft: Draft,
    traced: bool,
) -> Result<(NonZeroUsize, &'static str), GateError> {
    let (n, from) = match parallel {
        Some(n) if n > 1 && draft == Draft::Dspark => {
            return Err(format!(
                "--parallel {n} under BLOOMERY_DRAFT=dspark: the draft's window is one \
                 sequence's, on its own card, and a select would start it over; serve one \
                 slot, or the lookup draft"
            )
            .into());
        }
        Some(n) => (n, "flag"),
        None if draft == Draft::Dspark => (1, "dspark"),
        None if traced => (1, "route-trace"),
        None => (SLOTS, "default"),
    };
    let n = NonZeroUsize::new(n).ok_or("--parallel 0: the server serves no slot")?;
    Ok((n, from))
}

/// The positions each of `slots` slots serves for `--ctx` `ctx`: the whole
/// of it, every slot a full context, so the plan counts `slots` times one
/// sequence's bytes. The one place the rule lives: a split (⌊ctx / slots⌋,
/// llama-server's `-np` without `-kvu`) is this function's body. Refused by
/// name below [`PAIR_ROWS`], the pair pass's rows: the most one call of a
/// slot runs on a capture.
fn slot_ctx(ctx: usize, slots: NonZeroUsize) -> Result<usize, GateError> {
    if ctx < PAIR_ROWS {
        return Err(format!(
            "--ctx {ctx}: each of the {slots} slots serves {ctx} positions, below the \
             {PAIR_ROWS} a slot serves at least (the pair pass's rows)"
        )
        .into());
    }
    Ok(ctx)
}

/// Loads the model and serves until the listener or the engine fails;
/// `Ok` carries why the server ended.
pub fn run(args: &[String]) -> Result<ServeError, GateError> {
    let levers = bloomery_levers::at_main(crate::serve_levers::ACTS_ON)?;
    record::at_main("bloomery-serve-ds41", record::BLOOMERY_SERVE_DS41);
    let a = parse_args(args)?;
    let at = ResidencyAt {
        serving_place: a.place != Place::Gate,
        check_finite: false,
        route_trace: levers.route_trace().is_some(),
        prefill_steps: body::PrefillMode::from_name(levers.prefill())
            == Some(body::PrefillMode::Steps),
    };
    let residency = levers.residency_at(at);
    record::residency_lever(residency).eprint();
    let cfg = body::OpenCfg::from_levers_at(&levers, at)?;
    if cfg.body.residency != Residency::Off && a.place == Place::Gate {
        return Err(format!(
            "BLOOMERY_RESIDENCY={} under --place gate: the residency machine runs under \
             --place a and bp only",
            residency.word
        )
        .into());
    }
    let draft = Draft::from_levers(&levers)?;
    let (slots, from) = slot_count(a.parallel, draft, levers.route_trace().is_some())?;
    // Before anything is read or planned: a context no slot can serve.
    let ctx = slot_ctx(a.ctx, slots)?;
    eprintln!(
        "parallel rule=slots slots={slots} slot_ctx={ctx} total={} from={from}",
        slots.get() * ctx
    );
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
    let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?.with_user_start(USER_START)?);
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
    let trace = route_trace(
        &levers,
        &cfg,
        residency,
        (draft, slots),
        a.place,
        &path,
        &inputs,
    )?;
    let (card, placement, headroom) = print_plan(
        &inputs,
        a.place,
        (reserve, tier_batch),
        (ctx, slots),
        &cfg.place,
        (cfg.body.residency, residency),
    )?;
    let cache_ram = a
        .cache_ram
        .unwrap_or_else(|| u64::try_from(headroom / 2).map_or(0, |half| half.min(CACHE_RAM_CAP)));
    Record::new(&record::CACHE_CONFIG)
        .u("ram", cache_ram)
        .u("headroom", headroom)
        .w("user_start", USER_START)
        .w("in_template", template.contains(USER_START))
        .eprint();
    let props = EngineProps {
        model: Some(model),
        placement,
        ctx_verified: Some(VERIFIED_POSITIONS),
        ..EngineProps::default()
    };
    let open = SeatArgs {
        place: a.place,
        ctx,
        slots,
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
    let engine = SeatEngine::spawn(move || V41::open(open), ctx, vocab, card, props, cache_ram)?;
    let config = ServerConfig {
        model_alias: a.alias.unwrap_or(name),
        model_path: path.display().to_string(),
        chat_template: template,
        sampler: Some(sampler_factory()),
        fatal_linger: FATAL_LINGER,
        slot_save_path: a.slot_save_path,
    };
    // The seat's resident slots are the server's, one sequence each: the
    // server selects and steps them together, no turns and no park.
    let config_slots = SlotConfig {
        parallel: slots.get(),
        queue_depth: a.queue_depth,
        ..SlotConfig::default()
    };
    let server = Server::bind_with(
        (a.host.as_str(), a.port),
        Box::new(engine),
        config,
        config_slots,
    )?;
    Record::new(&record::LISTENING)
        .w("place", a.place.name())
        .u("ctx", ctx)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}

/// The route trace `BLOOMERY_ROUTE_TRACE` asks for, its directory made
/// now, before the load; refused by name under the batched feed, a draft or
/// several resident slots (`slots`), which it does not record.
fn route_trace(
    levers: &bloomery_levers::Levers,
    cfg: &body::OpenCfg,
    residency: ResidencyPick,
    (draft, slots): (Draft, NonZeroUsize),
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
    if slots.get() > 1 {
        return Err(format!(
            "BLOOMERY_ROUTE_TRACE records one sequence's steps after its call row; \
             --parallel {slots} puts another sequence's steps between them"
        )
        .into());
    }
    // The trace is the input the residency model replays under a fixed
    // seed; under the machine its slot files would record the machine's
    // own moves, its header would not say so, and the step feed would
    // count each prompt id as a kept row where the batched call keeps 0.
    if cfg.body.residency != Residency::Off {
        return Err(format!(
            "BLOOMERY_ROUTE_TRACE records a fixed placement's routing; \
             BLOOMERY_RESIDENCY={} moves the slot map under it",
            residency.word
        )
        .into());
    }
    let hp = &inputs.hp;
    let yes = |b: bool| if b { "on" } else { "off" }.to_owned();
    let header = TraceHeader {
        model: path.to_path_buf(),
        arch: "deepseek41".to_owned(),
        build: format!("bloomery-serve-ds41 {}", serve::VERSION),
        n_expert: hp.experts.n_expert,
        n_used: hp.experts.n_used,
        first_layer: 0,
        n_layer: hp.n_layer,
        extra: vec![
            ("placement".to_owned(), place.name().to_owned()),
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
/// prompt-batch bytes, on its tier card), `slots` resident sequences of
/// `ctx` positions each (`PlanInputs::plan_with_slots`), on stderr;
/// returns its cards' names, the plan's placement for `/props` (`None`,
/// and a line saying why, when a card's nvidia-smi index cannot be found)
/// and the plan's host headroom in bytes. Under `residency` (the rule and
/// the lever as resolved) the churn pool's record follows the plan's, and
/// the headroom is what the pool leaves; a pool that does not fit is
/// refused by name, naming the lever when it is the placement's default.
fn print_plan(
    inputs: &PlanInputs,
    place: Place,
    (reserve, tier_batch): (Option<u64>, Option<TierBatchBytes>),
    (ctx, slots): (usize, NonZeroUsize),
    levers: &PlanLevers,
    residency: (Residency, ResidencyPick),
) -> Result<(String, Option<PlacementProps>, i64), GateError> {
    let machine = place.machine(reserve, tier_batch)?(inputs.model.layers);
    let plan = inputs.plan_with_slots(&machine, u64::try_from(ctx)?, levers, slots)?;
    record::plan(place.name(), &machine, &plan).eprint();
    let mut host_headroom = plan.host.headroom_bytes;
    if let Some(pool) = swap::churn(&plan, 0, residency.0)? {
        let pick = residency.1;
        record::residency_host(pick.word, &pool, &plan).eprint();
        host_headroom = match pool.check(&plan) {
            Err(e) if pick.why == ResidencyWhy::Place => {
                return Err(format!(
                    "BLOOMERY_RESIDENCY unset is {} under --place {}: {e}; \
                     BLOOMERY_RESIDENCY=off loads the fixed placement",
                    pick.word,
                    place.name()
                )
                .into());
            }
            r => r?,
        };
    }
    let gpus: Result<Vec<String>, String> = machine
        .all_cards()
        .map(|c| nvidia_smi_index(&c.name, c.device).map(|i| format!("GPU{i}")))
        .collect();
    let placement = gpus.and_then(|g| placement_props(&plan, &g));
    if let Err(e) = &placement {
        eprintln!("bloomery-serve-ds41: /props leaves the placement out: {e}");
    }
    let headroom = i64::try_from(host_headroom)
        .map_err(|_| format!("the plan's host headroom {host_headroom} B passes i64"))?;
    let cards: Vec<&str> = machine.all_cards().map(|c| c.name.as_str()).collect();
    Ok((cards.join("+"), placement.ok(), headroom))
}

const WHAT: &str = "bloomery-serve-ds41";

/// What the engine thread opens the seat with.
struct SeatArgs {
    place: Place,
    /// A slot's context: the positions each resident sequence serves.
    ctx: usize,
    /// The resident sequences the plan counts and the load makes.
    slots: NonZeroUsize,
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
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        // The plan counts every resident sequence, which the session's own
        // open (one sequence) does not: the model is opened here by that
        // plan, then handed to the session.
        let machine = a.place.machine(a.reserve, a.tier_batch)?;
        let mut m = body::open_slots(file, machine, a.ctx, &a.cfg, a.slots)?;
        m.set_mode(StepMode::Graph);
        let mut log = Log { a: &a, pinned, t };
        log.load(&m)?;
        let cfg = Ds41Cfg {
            feed: a.cfg.body.prefill,
            open: a.cfg.clone(),
            card_timing: false,
        };
        let mut loaded = Loaded::<Body>::from_model(m, cfg, u32::try_from(a.ctx)?);
        let spark = match &a.draft_file {
            Some(file) => {
                let (d, load) = open_dspark(&mut loaded, file, &a.path, WHAT, a.place, a.reserve)?;
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
        if a.cfg.body.residency != Residency::Off {
            // A request's passes are not known at load, and no request
            // is a timed window: the log grows as it must.
            s.model_mut().body_parts(WHAT)?.2.log_residency(0);
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
        // The resident sequences, made after the captures: the live slot's
        // chains stay with it, each other slot capturing on its first use.
        s.add_slots(a.slots.get())?;
        let b = s.model().body(WHAT)?;
        eprintln!(
            "slots made={} planned={} seq_bytes={} seq_shadow_bytes={} resident_bytes={}",
            s.slots(),
            b.slots_planned(),
            b.seq_bytes(),
            b.seq_shadow_bytes(),
            s.model().resident_bytes()
        );
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

    /// The `residency pass` records of the boundaries the last call made,
    /// on stderr.
    fn print_passes(&mut self) -> Result<(), GateError> {
        for (kind, r) in self
            .s
            .model_mut()
            .body_parts(WHAT)?
            .2
            .take_residency_passes()
        {
            record::residency_pass_of(kind, &r).eprint();
        }
        Ok(())
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
        let next = match &mut self.draft {
            Served::Dspark(d) => d.draft_mut().feed_call(&mut self.s, ids)?,
            Served::Lookup(_, follows) => {
                *follows = false;
                self.s.prompt(ids, Want::Argmax)?.argmax()
            }
            Served::Off => self.s.prompt(ids, Want::Argmax)?.argmax(),
        };
        self.print_passes()?;
        Ok(next)
    }

    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        let next = self.s.step(last, Want::Argmax)?.argmax();
        self.print_passes()?;
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
            Served::Dspark(spec) => runtime::Advance::pass(&mut **spec, &mut self.s, last, out)?,
            Served::Off => {
                out.push(self.step(last)?);
                return Ok(Drafted::default());
            }
        };
        self.print_passes()?;
        Ok(drafted(c))
    }

    fn pass_rows(&self) -> usize {
        match self.draft {
            Served::Off => 1,
            Served::Lookup(..) | Served::Dspark(_) => PAIR_ROWS,
        }
    }

    /// The resident sequences the load made ([`Session::slots`]): one a
    /// `--parallel 1` load (and under the DSpark draft), the `--parallel`
    /// the seat served past it.
    fn slots(&self) -> usize {
        self.s.slots()
    }

    /// The session's slot ([`Session::select_slot`]): the target stands
    /// on another sequence, so the draft no longer follows it
    /// ([`V41::moved`]) — the lookup rebuilds its tables from that slot's
    /// history at its next pass. Selecting the live slot moves nothing.
    fn select(&mut self, slot: usize) -> Result<(), GateError> {
        if slot == self.s.selected() {
            return Ok(());
        }
        self.s.select_slot(slot)?;
        self.moved();
        Ok(())
    }

    /// The lookup's state is a function of the selected slot's history,
    /// rebuilt at each select ([`V41::select`]): a slot's drafted passes are
    /// the passes it would run alone. The DSpark draft serves one slot.
    fn slot_drafts(&self) -> bool {
        matches!(self.draft, Served::Lookup(..))
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

    /// [`Session::residency_reset`], its `residency reset` record on
    /// stderr.
    fn residency_reset(&mut self) -> Result<Option<ResidencyReset>, GateError> {
        let Some(r) = self.s.residency_reset()? else {
            return Ok(None);
        };
        record::residency_reset(&r).eprint();
        let n = |v: usize| v as u64;
        Ok(Some(ResidencyReset {
            cancelled: n(r.cancelled),
            copies: n(r.copies),
            diff: n(r.diff),
            dropped_bytes: r.dropped_bytes,
        }))
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
                let device = d
                    .device()
                    .ok()
                    .and_then(|dev| nvidia_smi_index(d.card(), Some(dev)).ok())
                    .map(|i| format!("GPU{i}"));
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
                copied: _,
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
        for h in threads::helper::helpers() {
            record::helper(&h.name, h.asked, h.pinned, h.cpus).eprint();
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
