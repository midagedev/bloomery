//! The GLM-5.3-Flash seat — the llama-server-compatible HTTP API on the
//! glm5next engine, opened and stepped as `generate_glm5next` opens and steps
//! the model. `bloomery-serve --model glm` is one call of [`run`], which
//! takes the process's arguments (`--model` already taken out by the
//! one-binary server).
//!
//!     [--host 127.0.0.1] [--port 8080] [--place a|gate] [--ctx C]
//!     [--alias NAME] [--chat-template-file PATH] [--prefill batch|steps] [--plan]
//!
//! The model is `$BLOOMERY_REF_MODEL`; its first shard gives the vocabulary,
//! `tokenizer.chat_template` the chat template (`--chat-template-file`
//! replaces it, as the qwen38 seat takes it) and `general.name` the default
//! alias. Tool calls are the template's: the server reads their markup out of
//! the generation (`serve::glmxml`, which this file's template teaches), so
//! the seat adds nothing there. The `plan` record, then the `load` and
//! `capture` lines of the open, go to stderr as `generate_glm5next` prints
//! them, then the `listening on http://<addr>` line once the model is loaded
//! and the port is bound (`--port 0` binds a free one). Sampling is the
//! sampler crate's chain with no repetition penalty; `temperature <= 0` is
//! the engine's argmax, the
//! ids `generate_glm5next --tokens <the prompt's ids>` prints.
//!
//! `--place` is `a` (the serving plan, `workstation::plan_a`, on the A6000 —
//! the default, as both other serving seats) or `gate` (the gate card's). The
//! positions the server serves are the stores the load sized (`--ctx`,
//! default 2048 as the CLI's, refused past `place::ORACLE_POSITIONS`):
//! `/props`' `n_ctx` is that number, a prompt that long is a 400 before it
//! reaches the engine, and generation stops there with `truncated`.
//!
//! The keep rule: each KDA layer holds one recurrent state, and its history
//! only in the checkpoints a prompt call takes (its start, every 512th
//! position of the whole context, its end — `CHECKPOINT_EVERY`), so a cut
//! keeps every fed position, the empty model, or the checkpoint at or below
//! it — never a position between two checkpoints. A request that shares less
//! than that with the slot is a `cache reuse` record with the rule. The seat
//! marks no user-start token, so the server asks nowhere to cut a prompt
//! call: the checkpoint spacing owns where a cut can land.
//!
//! The server's host prompt cache is off for this seat (its budget 0): a
//! checkpoint lives in the model's own host slots, not a value a cache could
//! hold, and the latent layers' cache rows have no copy either, so
//! `snapshot` and `resume` refuse by name and slot save/restore answers the
//! server's own 501.
//!
//! `BLOOMERY_DRAFT=mtp` loads the file's next-token layer beside the target
//! (`app::arch::glm5next::open_nextn`, the plan `PlanInputs::plan_nextn`
//! makes: the layer's card bytes and arena reserved, each KDA layer's state
//! two lanes) and drives the session through the runtime's speculative loop
//! with the shared window `app::mtp::MtpDraft`, as `generate_glm5next` does:
//! windows of two rows, the target's next token and the draft's one
//! proposal, the draft's walks eager, `pass_rows` 2; the greedy ids are the
//! plain run's of the same load. A sampling or id-banning request takes
//! plain steps (the server's loop asks a pass only of a greedy request with
//! no banned id), each step's row the target's, read before the step is told
//! to the draft. A `load draft=mtp` record follows the `load` line, and
//! `/props`' `engine.draft` names the draft (the target file, whose NextN
//! layer it is) with the plan's bytes for it as the card's `draft` class.
//! `BLOOMERY_DRAFT=off` is the plain path, one step a pass, with a `load
//! draft=off (<why>)` record. Unset follows the placement
//! (`bloomery_levers::glm_unset`), its word and why a `draft unset` record
//! before the plan: `mtp` under `--place a` on a file of one next-token
//! layer; the plain path under `--place gate`, on a file of other than one,
//! and with stores too short for one window, never a refusal. Set, refused by
//! name: `mtp` with stores too short for one window (`--ctx` under 3: the
//! prompt's last id, its first token and the window's second row), and every
//! other word.
//!
//! `BLOOMERY_RESIDENCY` set prints as a `residency lever` record first
//! thing. Unset follows the placement and the plan
//! (`bloomery_levers::glm_unset`, `glm_residency_at_plan`), its word and why a
//! `residency unset` record after the `plan` line: under `--place a`
//! `mid-p0-s1`; `off` under `--place gate`, with stores too short for one
//! window, beside `--prefill steps`, and when the plan holds no card expert,
//! its fewest leave no room for the word, or the churn pool does not fit the
//! plan's host headroom less the NextN layer's host experts or what
//! `MemAvailable` leaves past the load's host need — never a refusal.
//! `--plan` prints these records and the `plan` line and exits 0 before any
//! card is opened. Running `mid-p<P>-s<S>` (plain or
//! drafted), the load runs the common residency machine over the card's
//! routed stacks (`open_resident`, or `open_nextn` with the word, the
//! next-token layer's experts on the host): a `residency host` record
//! follows the `plan` line, the word refused by name when the plan's fewest
//! card experts a layer leave no room for P pinned, S spares and one that
//! moves or its host headroom cannot take the churn pool, and each call (a
//! prompt, a step, a pass) prints its boundaries' `residency pass` records
//! after it. A request's reset keeps the residency where use has taken it;
//! `POST /residency/reset` moves it back to its seed and prints a `residency
//! reset` record (without the residency, the server's 501). `mid-…` set is
//! refused by name beside `--prefill steps`, as the CLI refuses it.
//!
//! An engine error ends the process: the request gets a 500, `/health` a 503
//! for a moment, then the crash block (card, position, error) goes to stderr
//! and the exit code is 70. The levers this seat acts on ([`ACTS_ON`]) are
//! parsed once, at `main` (`bloomery_levers::at_main`), which refuses by name
//! a lever set outside them; `--levers` prints them with this process's
//! values and exits. The stderr lines named above are records of the kinds
//! `bloomery_gpu_gates::record` declares; `--records-schema` prints those
//! kinds and exits.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use app::arch::glm5next::{GlmCfg, open_nextn, open_resident};
use app::mtp::{MtpBody, MtpDraft};
use app::{Loaded, OpenLog, RowsLog, Session, SessionError};
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::bind::{
    Seat, SeatEngine, Vocab, model_props, nvidia_smi_index, placement_props, sampler_factory,
};
use bloomery_gpu_gates::generate::mode_name;
use bloomery_gpu_gates::record::{self, Kind, Record};
use bloomery_gpu_gates::residency38::{GLM_CARD, residency_room, residency_set};
use bloomery_gpu_gates::{GateError, ref_model_path};
use bloomery_gpu_glm5next::{Body, Glm5nextModel, PrefillMode};
use bloomery_levers::{
    GlmAt, GlmPick, ResidencyPick, ResidencyWhy, glm_residency_at_plan, glm_unset,
};
use gguf::Split;
use model::arch::glm5next::place::{NextnInputs, PlanInputs};
use model::placement::churn::ChurnPool;
use model::placement::workstation::{self, HostNeed, host_available};
use model::placement::{Machine, Plan, PlanLevers};
use runtime::Target;
use runtime::seqstate::Why;
use serve::{
    CacheNote, DraftProps, Drafted, EngineProps, FATAL_LINGER, ResidencyReset, Saved, ServeError,
    Server, ServerConfig,
};
use tokenizer::Tokenizer;

use super::drafted::DraftedSeat;

/// The seat's name, as its records and errors print it.
const WHAT: &str = "bloomery-serve-glm";

const USAGE: &str = "usage: bloomery-serve [--model glm] [--host H] [--port P] [--place a|gate] \
                     [--ctx C] [--alias NAME] [--chat-template-file PATH] \
                     [--prefill batch|steps] [--plan]";

/// The positions the stores are sized for when `--ctx` names none:
/// `generate_glm5next`'s default.
const CTX: usize = 2048;

/// Why no state is saved or put back: the recurrent state lives only in the
/// model's own checkpoint slots and the latent cache rows have no copy.
const SNAPSHOT_WHY: &str = "the KDA recurrent state lives only in the model's own checkpoint \
                            slots and the latent layers' cache rows have no copy: no sequence \
                            state is a value the cache could hold";

/// What this seat prints, all on stderr (its `--records-schema`): the
/// residency lever's word, the `plan`, `load` and `capture` lines
/// `generate_glm5next` prints, the host set's records of a placed load, the
/// draft's load line and its verify's capture, the listening line, the reuse
/// records the checkpoint rule answers, the draft's joins, and the residency
/// machine's records.
static KINDS: &[&Kind] = &[
    &record::RESIDENCY_LEVER,
    &record::DRAFT_UNSET_GLM,
    &record::PLAN,
    &record::RESIDENCY_UNSET_GLM,
    &record::RESIDENCY_HOST,
    &record::LOAD_GENERATOR,
    &record::HOST_POPULATE,
    &record::HOST_POPULATE_OFF,
    &record::HOST_LOCK,
    &record::LOAD_DRAFT_GLM,
    &record::LOAD_DRAFT_OFF_GLM,
    &record::CAPTURE,
    &record::CAPTURE_PAIR,
    &record::LISTENING_GLM,
    &record::CACHE_REUSE,
    &record::MTP_PROMPT,
    &record::RESIDENCY_PASS,
    &record::RESIDENCY_RESET,
    &record::RESIDENCY_LEAK,
];

/// The levers this seat acts on: those `generate_glm5next` reads for its load
/// and its draft (`BLOOMERY_ROUTE_TRACE` and `BLOOMERY_STEP_STATS` left out —
/// each is one run's instrument, which a server that serves many prompts
/// does not wire, and a lever set outside this list is refused by name at
/// `main`), with `BLOOMERY_PIN_MAIN` for the engine thread's cpu slot, as
/// every serving seat reads it; `gate_glm5next_serve` holds the same list.
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::DRAFT,
    bloomery_levers::RESIDENCY,
];

/// The drafted window's verify: the target's next token and the draft's one
/// proposal.
const VERIFY_ROWS: usize = <Body as MtpBody>::VERIFY_ROWS;

/// The positions the server's loop needs for one window from an empty
/// model: it takes one only while its rows fit, the first at a one-id
/// prompt's first generated token, 1 + 1 + rows − 1.
const NEED: usize = VERIFY_ROWS + 1;

/// Where `--place` puts the plan's one card, as `generate_glm5next` takes it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GlmPlace {
    /// The serving plan (`workstation::plan_a`), on the A6000. The default:
    /// a server belongs on the serving card.
    A,
    /// The gate card's plan (`workstation::plan_gate`), on the 3090.
    Gate,
}

impl GlmPlace {
    fn parse(v: &str) -> Result<GlmPlace, GateError> {
        match v {
            "a" => Ok(GlmPlace::A),
            "gate" => Ok(GlmPlace::Gate),
            other => Err(format!("--place is a or gate, not {other}").into()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            GlmPlace::A => "a",
            GlmPlace::Gate => "gate",
        }
    }

    fn machine(self) -> fn(usize) -> Machine {
        match self {
            GlmPlace::A => workstation::plan_a,
            GlmPlace::Gate => workstation::plan_gate,
        }
    }
}

/// `BLOOMERY_DRAFT` on stores of `ctx` positions, `unset` the seat's rule
/// for it: `None` drafts with the NextN layer; `Some` runs the plain path,
/// with why (the `load draft=off` record's). Unset, the rule's word and why
/// print as a `draft unset` record. Refused by name: `mtp` set with no room
/// for one window, and every word but `mtp` and `off`, as `generate_glm5next`
/// refuses it.
fn draft_of(
    levers: &bloomery_levers::Levers,
    ctx: usize,
    unset: GlmPick,
) -> Result<Option<String>, GateError> {
    match levers.draft() {
        Some("mtp") if ctx < NEED => Err(format!(
            "BLOOMERY_DRAFT=mtp needs --ctx {NEED} or more: token 0 comes out of the feed, and \
             a window's {VERIFY_ROWS} rows run after it (--ctx {ctx})"
        )
        .into()),
        Some("mtp") => Ok(None),
        Some("off") => Ok(Some("BLOOMERY_DRAFT=off".to_owned())),
        Some(other) => Err(format!(
            "BLOOMERY_DRAFT={other}: on a glm5next file mtp drafts the window (the file's NextN \
             layer); lookup and dspark are the V4.1 binaries'"
        )
        .into()),
        None => {
            Record::new(&record::DRAFT_UNSET_GLM)
                .w("draft", unset.word)
                .w("why", unset.why)
                .eprint();
            match unset.word {
                "mtp" => Ok(None),
                _ => Ok(Some(unset.why.to_string())),
            }
        }
    }
}

/// `BLOOMERY_RESIDENCY` as this seat takes it before the plan.
enum ResidencyLever {
    /// Set: its parse and word.
    Set(Residency, &'static str),
    /// Unset: the seat's rule before the plan, which the plan then decides
    /// ([`bloomery_levers::glm_residency_at_plan`]).
    Unset(GlmPick),
}

/// `BLOOMERY_RESIDENCY` before the plan: set, as given, its `residency
/// lever` record printed, `mid-…` refused by name beside the steps feed (each
/// prompt id would end a pass the rule counts; the body refuses it at the
/// feed, this before the load); unset, `unset`.
fn residency_of(
    levers: &bloomery_levers::Levers,
    prefill: PrefillMode,
    unset: GlmPick,
) -> Result<ResidencyLever, GateError> {
    let Some(word) = levers.residency() else {
        return Ok(ResidencyLever::Unset(unset));
    };
    let r = Residency::parse(word)?;
    record::residency_lever(ResidencyPick {
        word,
        why: ResidencyWhy::Set,
    })
    .eprint();
    if r != Residency::Off && prefill == PrefillMode::Steps {
        return Err(format!(
            "BLOOMERY_RESIDENCY={word} is refused beside --prefill steps (each prompt id would \
             end a pass the rule counts)"
        )
        .into());
    }
    Ok(ResidencyLever::Set(r, word))
}

/// The residency the load of `plan` runs, `beside` the bytes the load hosts
/// outside the plan (the NextN layer's experts): a set word as given, refused
/// by name when the plan's card slots leave it no room; unset, the rule's
/// word where the plan has room and the host the churn pool, else `off`
/// with why, its `residency unset` record printed. Then, under `mid`, the
/// `residency host` record ([`residency_set`]).
fn residency_at(
    plan: &Plan<'_>,
    lever: ResidencyLever,
    beside: u64,
) -> Result<Residency, GateError> {
    let (r, word) = match lever {
        ResidencyLever::Set(r, word) => {
            residency_room(plan, r, word)?;
            (r, word)
        }
        ResidencyLever::Unset(pick) => {
            // The load refuses a host set past `MemAvailable` before any
            // upload; the default leaves the pool out instead.
            let mem_left = match pick.word {
                "off" => 0,
                _ => i128::from(host_available()?) - i128::from(HostNeed::of(plan, beside).bytes()),
            };
            let pick = glm_residency_at_plan(
                pick,
                plan.n_l.iter().copied(),
                |pinned| ChurnPool::of(plan, GLM_CARD, pinned).map(|pool| pool.bytes),
                plan.host.headroom_bytes - i128::from(beside),
                mem_left,
            )
            .map_err(|e| format!("BLOOMERY_RESIDENCY unset: the churn pool: {e}"))?;
            Record::new(&record::RESIDENCY_UNSET_GLM)
                .w("residency", pick.word)
                .w("why", pick.why)
                .eprint();
            (Residency::parse(pick.word)?, pick.word)
        }
    };
    residency_set(plan, GLM_CARD, r, word, beside, Record::eprint)
}

struct Args {
    host: String,
    port: u16,
    place: GlmPlace,
    ctx: usize,
    alias: Option<String>,
    /// `--chat-template-file`, replacing the file's own template.
    template_file: Option<PathBuf>,
    prefill: PrefillMode,
    /// `--plan`: the records before the load, then exit.
    plan_only: bool,
}

fn parse_args(args: &[String]) -> Result<Args, GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        place: GlmPlace::A,
        ctx: CTX,
        alias: None,
        template_file: None,
        prefill: PrefillMode::Batch,
        plan_only: false,
    };
    let mut it = args.iter().map(|s| s.as_str());
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
            "--port" => a.port = v.parse()?,
            "--place" => a.place = GlmPlace::parse(v)?,
            "--ctx" => a.ctx = v.parse()?,
            "--alias" => a.alias = Some(v.to_owned()),
            "--chat-template-file" => a.template_file = Some(PathBuf::from(v)),
            "--prefill" => {
                a.prefill = PrefillMode::from_name(v)
                    .ok_or_else(|| format!("--prefill is batch or steps, not {v}"))?;
            }
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    if a.ctx == 0 {
        return Err("--ctx 0: the stores hold no position".into());
    }
    Ok(a)
}

/// Loads the model and serves until the listener or the engine fails; `Ok`
/// carries why the server ended.
pub fn run(args: &[String]) -> Result<ServeError, GateError> {
    let levers = bloomery_levers::at_main(ACTS_ON)?;
    record::at_main(WHAT, KINDS);
    let a = parse_args(args)?;
    let plan_levers = PlanLevers::from_levers(&levers)?;
    let path = ref_model_path()?;
    let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?);
    let inv = gguf::inventory_of(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let template = match &a.template_file {
        Some(file) => std::fs::read_to_string(file)
            .map_err(|e| format!("--chat-template-file {}: {e}", file.display()))?,
        None => inv
            .value("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{}: no tokenizer.chat_template", path.display()))?
            .to_owned(),
    };
    let name = inv
        .value("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or("glm-5.3-flash")
        .to_owned();
    drop(inv);

    // The plan record, and `/props` from the same plan the load runs by:
    // drafting, the target's plan beside the NextN layer, whose card bytes
    // and arena are the card's `draft` class and whose host experts the
    // load's host set holds beside the plan's.
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let inputs = PlanInputs::read(&split)?;
    let unset = glm_unset(GlmAt {
        place_a: a.place == GlmPlace::A,
        nextn_layers: inputs.hp.n_layer.saturating_sub(inputs.hp.n_trunk),
        need: NEED,
        ctx: a.ctx,
        prefill_steps: a.prefill == PrefillMode::Steps,
    });
    let lever = residency_of(&levers, a.prefill, unset.residency)?;
    let draft_off = draft_of(&levers, a.ctx, unset.draft)?;
    let model = model_props(&split, &inputs.model);
    let machine = (a.place.machine())(inputs.model.layers);
    let ctx = u64::try_from(a.ctx)?;
    let nextn = match draft_off {
        None => Some(NextnInputs::read(&inputs)?),
        Some(_) => None,
    };
    let (plan, beside, draft_bytes) = match &nextn {
        None => (inputs.plan(&machine, ctx, &plan_levers)?, 0, 0),
        Some(n) => {
            let with = inputs.plan_nextn(&machine, ctx, &plan_levers, n)?;
            let beside = with.host_runs()?.1;
            let bytes = with.nextn_card_bytes() + with.arena_bytes;
            (with.plan, beside, bytes)
        }
    };
    record::plan(a.place.name(), &machine, &plan).eprint();
    let residency = residency_at(&plan, lever, beside)?;
    if a.plan_only {
        // The records before the load are out; nothing was opened on a card.
        std::process::exit(0);
    }
    let gpu = nvidia_smi_index(&machine.cards[0].name)
        .map(|i| format!("GPU{i}"))
        .and_then(|g| placement_props(&plan, &[g]));
    if let Err(e) = &gpu {
        eprintln!("{WHAT}: /props leaves the placement out: {e}");
    }
    let props = EngineProps {
        model: Some(model),
        placement: gpu.ok(),
        ..EngineProps::default()
    };

    let open = SeatArgs {
        place: a.place,
        ctx: a.ctx,
        cfg: GlmCfg {
            place: plan_levers,
            host: levers.host(),
            prefill: a.prefill,
        },
        pin_main: levers.pin_main(),
        path: path.clone(),
        draft_off,
        draft_bytes,
        residency,
    };
    // The prompt cache is off (its budget 0): the seat's snapshot refuses, and
    // slot save/restore is the server's own 501.
    let engine = SeatEngine::spawn(
        move || Glm::open(open),
        a.ctx,
        vocab,
        machine.cards[0].name.clone(),
        props,
        0,
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
    Record::new(&record::LISTENING_GLM)
        .w("place", a.place.name())
        .u("ctx", a.ctx)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}

/// What the engine thread opens the seat with.
struct SeatArgs {
    place: GlmPlace,
    ctx: usize,
    cfg: GlmCfg,
    pin_main: bool,
    path: PathBuf,
    /// Drafting nothing, why; `None` drafts with the NextN layer.
    draft_off: Option<String>,
    /// The plan's bytes for the NextN layer (its card terms and arena), the
    /// card's `draft` class in `/props`.
    draft_bytes: u64,
    /// The residency the load runs.
    residency: Residency,
}

/// The GLM session on the engine thread: the session over the model, the
/// draft it drives when one runs (its windows of two rows through the
/// runtime's speculative loop), and the positions its stores were sized
/// for.
struct Glm {
    s: Session<Body>,
    drafted: DraftedSeat<Body, VERIFY_ROWS>,
    ctx: usize,
    /// The plan's bytes for the NextN layer, for `/props`' `draft` class.
    draft_bytes: u64,
    /// The target file, whose NextN layer `/props`' `draft` names.
    path: PathBuf,
    /// The load runs the residency machine: each call prints its
    /// boundaries' `residency pass` records.
    residency: bool,
}

impl Glm {
    /// The session by `a.place` (the `plan` was printed before the engine
    /// thread started; the `load` and `capture` lines go to stderr as
    /// `generate_glm5next` prints them, the draft's `load draft=…` line after
    /// the `load` line), on the calling thread, pinned to the dispatcher's
    /// cpu slot when asked: drafting, the NextN load under the residency
    /// (`open_nextn`) and the window over it, its verify captured (a
    /// `capture` line of its nodes); else the residency's load
    /// (`open_resident`) or the plain one.
    fn open(a: SeatArgs) -> Result<Glm, GateError> {
        let pinned = a.pin_main && threads::pool().pin_caller();
        let t = Instant::now();
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        let mut log = Log {
            top_k: 0,
            place: a.place.name(),
            prefill: a.cfg.prefill,
            ctx: a.ctx,
            pin_main: a.pin_main,
            pinned,
            t,
            draft_off: a.draft_off.clone(),
            draft_bytes: a.draft_bytes,
        };
        let prefill = a.cfg.prefill;
        let args = app::OpenArgs {
            place: a.place.name(),
            machine: a.place.machine(),
            ctx: a.ctx,
            mode: StepMode::Graph,
            cfg: a.cfg,
        };
        let planned = || format!("{WHAT}: the open planned nothing");
        let mut s = match (&a.draft_off, a.residency) {
            (None, r) => open_nextn(file, args, r, &mut log)?.ok_or_else(planned)?,
            (Some(_), Residency::Off) => Loaded::<Body>::open(file, args, &mut log)?
                .ok_or_else(planned)?
                .ready(&mut log)?,
            (Some(_), r) => open_resident(file, args, r, &mut log)?.ok_or_else(planned)?,
        };
        let residency = a.residency != Residency::Off;
        if residency {
            // A request's passes are not known at load: the log grows as it
            // must, and every call takes it.
            s.model_mut().body_parts(WHAT)?.2.log_residency(0);
        }
        let drafted = DraftedSeat::new(match a.draft_off {
            Some(_) => None,
            None => {
                let draft = MtpDraft::open(s.model(), prefill, StepMode::Eager)?;
                Some(s.with_draft::<MtpDraft<Body>, VERIFY_ROWS>(draft, &mut PairCapture)?)
            }
        });
        Ok(Glm {
            s,
            drafted,
            ctx: a.ctx,
            draft_bytes: a.draft_bytes,
            path: a.path,
            residency,
        })
    }

    /// The `residency pass` records of the boundaries the last call made, on
    /// stderr; nothing without the residency.
    fn print_passes(&mut self) -> Result<(), GateError> {
        if !self.residency {
            return Ok(());
        }
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
}

/// The verify pass's capture: its nodes, one line.
struct PairCapture;

impl RowsLog for PairCapture {
    fn capture_rows(&mut self, _rows: usize, nodes: usize) -> Result<(), SessionError> {
        Record::new(&record::CAPTURE_PAIR)
            .u("pair_graph_nodes", nodes)
            .eprint();
        Ok(())
    }
}

/// What the open prints: the plan's top_k, the load, the draft's load, the
/// capture.
struct Log {
    /// The file's indexer top-k, which the plan read.
    top_k: usize,
    place: &'static str,
    prefill: PrefillMode,
    ctx: usize,
    pin_main: bool,
    pinned: bool,
    t: Instant,
    /// Drafting nothing, why.
    draft_off: Option<String>,
    /// The plan's bytes for the NextN layer.
    draft_bytes: u64,
}

impl OpenLog<Body> for Log {
    /// The plan was printed before the engine thread started.
    fn plan(
        &mut self,
        _place: &'static str,
        inputs: &PlanInputs,
        _machine: &Machine,
        _plan: &model::placement::Plan<'_>,
    ) -> Result<bool, app::SessionError> {
        self.top_k = inputs.hp.indexer.top_k;
        Ok(true)
    }

    fn load(&mut self, m: &Glm5nextModel) -> Result<(), app::SessionError> {
        let b = m.body(WHAT)?;
        Record::new(&record::LOAD_GENERATOR)
            .u("resident_bytes", m.resident_bytes())
            .u("ctx", self.ctx)
            .u("layers", b.kinds().len())
            .u("top_k", self.top_k)
            .w("shadow", "none")
            .u("shadow_bytes", 0)
            .u("unified_addressing", 0)
            .w("prefill", self.prefill.name())
            .w("mode", mode_name(StepMode::Graph))
            .w("place", self.place)
            .w("pin_main", if self.pin_main { "on" } else { "off" })
            .w("pinned", self.pinned)
            .f("load_s", self.t.elapsed().as_secs_f64())
            .eprint();
        if let Some(h) = b.hybrid().residency() {
            for r in record::host_residency(h) {
                r.eprint();
            }
        }
        match &self.draft_off {
            Some(why) => Record::new(&record::LOAD_DRAFT_OFF_GLM)
                .w("why", why)
                .eprint(),
            None => {
                let d = b.nextn().ok_or_else(|| {
                    SessionError::Refused(format!("{WHAT}: the NextN load holds no NextN layer"))
                })?;
                Record::new(&record::LOAD_DRAFT_GLM)
                    .u("layer", d.index())
                    .u("resident", d.resident_bytes())
                    .u("arena", d.arena_bytes())
                    .w("head", if d.head_rows() { "rows" } else { "full" })
                    .u("plan_bytes", self.draft_bytes)
                    .f("load_s", self.t.elapsed().as_secs_f64())
                    .eprint();
            }
        }
        Ok(())
    }

    fn capture(&mut self, nodes: usize) -> Result<(), app::SessionError> {
        Record::new(&record::CAPTURE)
            .u("graph_nodes", nodes)
            .eprint();
        Ok(())
    }

    fn prompt_buffers(&mut self, _m: &Glm5nextModel) -> Result<(), app::SessionError> {
        Ok(())
    }
}

impl Seat for Glm {
    fn pos(&self) -> usize {
        self.s.pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The body's feed ([`bloomery_gpu_glm5next::feed`]): the batched prompt
    /// call or one step a position, as `--prefill` says, each call taking the
    /// checkpoints its marks name; under the draft the draft's own prompt
    /// call, its store walked over the prompt's units.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        let next = self.drafted.prefill(&mut self.s, ids)?;
        self.print_passes()?;
        Ok(next)
    }

    /// One step; under the draft the rows it left waiting walked first
    /// (`MtpDraft::before_step`: a request that continues the held sequence
    /// joins it here when its prompt call is empty).
    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        let next = self.drafted.step(&mut self.s, last)?;
        self.print_passes()?;
        Ok(next)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    /// One step and the target's row of it, read before the step is told to
    /// the draft ([`DraftedSeat::step_with_row`]).
    fn step_row(&mut self, last: u32, row: &mut [f32]) -> Result<u32, GateError> {
        let next = self.drafted.step_with_row(&mut self.s, last, row)?;
        self.print_passes()?;
        Ok(next)
    }

    /// One pass from `last`: under the draft the window of two rows, its
    /// kept tokens and counts; without it one step.
    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        let d = self.drafted.pass(&mut self.s, last, out)?;
        self.print_passes()?;
        Ok(d)
    }

    /// The most positions one pass runs: the window's two rows, or one step
    /// without the draft.
    fn pass_rows(&self) -> usize {
        self.drafted.pass_rows()
    }

    /// The session's reset, then the draft started over: the residency stays
    /// where use has taken it (only [`Seat::residency_reset`] moves it back).
    fn reset(&mut self) -> Result<(), GateError> {
        self.drafted.reset(&mut self.s)
    }

    /// [`Session::residency_reset`], its `residency reset` record on stderr;
    /// `None` without the residency (the server's 501).
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

    /// `/props`' `engine.draft` under the MTP draft: its kind, the target
    /// file whose NextN layer it is, its width, and the plan's bytes for it
    /// as the card's `draft` class.
    fn props(&self, mut p: EngineProps) -> EngineProps {
        if !self.drafted.drafts() {
            return p;
        }
        if let Some(place) = p.placement.as_mut() {
            for d in place.devices.iter_mut() {
                if d.device.starts_with("GPU") {
                    d.class_bytes.insert("draft".to_owned(), self.draft_bytes);
                }
            }
        }
        p.draft = Some(DraftProps {
            model: self.path.file_name().map_or_else(
                || self.path.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            ),
            n_max: Some(<Body as MtpBody>::WIDTH as u64),
            kind: Some("mtp".to_owned()),
            path: Some(self.path.display().to_string()),
            device: None,
        });
        p
    }

    /// The body's rule ([`Body::kept`]): every fed position, the empty model,
    /// or the checkpoint at or below the cut, with the rule that kept less.
    /// A query past u32 positions is the query for every position.
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        let asked = u32::try_from(n).unwrap_or(u32::MAX);
        let k = self.s.kept(asked);
        let why = match k.why {
            Why::Current | Why::Empty => None,
            _ => Some(k.to_string()),
        };
        (k.at as usize, why)
    }

    /// The checkpoint the cut keeps, taken back by the body's own rollback;
    /// any other position is refused by name, the body's own message.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        Ok(self.s.model_mut().rollback(pos)?)
    }

    /// Nowhere: this seat marks no user start, and a cut keeps the
    /// checkpoints the body's own rule takes — a call cut at a mark would add
    /// one the checkpoint spacing did not choose.
    fn splits(&self, _: usize, _: usize, _: &[usize]) -> Vec<usize> {
        Vec::new()
    }

    /// A GLM sequence state is not a value a cache could hold
    /// ([`SNAPSHOT_WHY`]). The prompt cache is off, so this is never asked;
    /// a caller that asks is refused by name.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        Err(format!("a snapshot at position {}: {SNAPSHOT_WHY}", self.s.pos()).into())
    }

    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError> {
        Err(format!(
            "a resume of a state of {} positions: {SNAPSHOT_WHY}",
            state.n_tokens()
        )
        .into())
    }

    /// Only `Reuse` arrives (the cache is off and no call is cut); the other
    /// notes belong to a caching engine and print as they come.
    fn note(note: &CacheNote) {
        if let CacheNote::Reuse {
            common,
            ask,
            kept,
            held,
            reason,
        } = note
        {
            Record::new(&record::CACHE_REUSE)
                .u("common", common)
                .u("ask", ask)
                .u("kept", kept)
                .u("held", held)
                .w("reason", reason.as_deref().unwrap_or("unstated"))
                .eprint();
            return;
        }
        eprintln!("{WHAT}: {note}");
    }
}
