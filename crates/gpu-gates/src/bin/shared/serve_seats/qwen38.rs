//! `bloomery-serve-qwen38` — the llama-server-compatible HTTP API on the
//! Qwen3.8-Flash-Next (qwen4exp) engine, for a client to attach to.
//!
//! This module is the seat: `bloomery-serve-qwen38` and `bloomery-serve --model
//! qwen38` are each one call of [`run`], which takes the process's arguments
//! (`--model` already taken out by the one-binary server).
//!
//!     bloomery-serve-qwen38 [--host 127.0.0.1] [--port 8080] [--place a|gate]
//!                           [--ctx-size C] [--alias NAME]
//!                           [--chat-template-file PATH]
//!
//! `--ctx` is `--ctx-size` under its family's other spelling. The model is
//! `$BLOOMERY_REF_MODEL` (a `--model` flag is not taken: the family's
//! binaries read the one the profile pins); its first shard gives the
//! vocabulary, `tokenizer.chat_template` the chat template
//! (`--chat-template-file` replaces it) and `general.name` the default alias.
//! The `plan` record, then the `load` and `capture` lines of the qwen4exp
//! open, go to stderr as `generate_qwen3moe` prints them, then `listening on
//! http://<addr>` once the model is loaded and the port is bound (`--port 0`
//! binds a free one). Sampling is the sampler crate's chain with no
//! repetition penalty; `temperature <= 0` is the engine's argmax, the ids
//! `generate_qwen3moe --tokens <the prompt's ids>` prints for a prompt of at
//! most eight ids (past that the ubatch walk the server and the CLI both take
//! leaves a position's bits a function of its own inputs, so a longer
//! prompt's greedy ids are the walk's, not a step-fed run's).
//!
//! The positions the server serves are the stores the load sized
//! (`--ctx-size`, default 4096): `/props`' `n_ctx` is that number, a prompt
//! that long is a 400 before it reaches the engine, and generation stops
//! there with `truncated`.
//!
//! A request keeps no prefix of another's: the recurrent layers (the GDN
//! state lanes, the PLE hash history) keep no state for an earlier position,
//! so the seat's keep rule grants nothing and every request prefills from a
//! reset — the same ids as a fresh one, never a silently wrong state. The
//! server's host prompt cache is therefore off for this engine (its budget is
//! 0) and slot save/restore answers the server's own 501; every prefix a
//! request shares with what the slot holds is a `cache reuse` record with the
//! rule. Where the body does take positions back — a verify's commit — no
//! request path reaches.
//!
//! `BLOOMERY_DRAFT` unset follows the placement as `generate_qwen3moe`'s
//! does (`bloomery_levers::draft38_unset`): under `--place a` the MTP draft
//! runs when a regular file is where it would be opened
//! (`BLOOMERY_MTP_DRAFT`, else the shared draft file beside the target); the
//! plain path runs under `--place gate`, with no file there, and with stores
//! too short for one window (`--ctx-size` under 5), each printed as a `load
//! draft=off (<why>)` record after the `load` line (`no file at <path>` for
//! the missing file), never a refusal. `BLOOMERY_DRAFT=off` is the plain
//! path with the same record; `mtp` drafts wherever the draft loads. The
//! CLI's `--logits` and route-trace conditions have no seat equivalent: a
//! request that reads the logits row steps plainly, and the seat does not
//! take the route trace.
//!
//! Drafting, the seat drives the session through the runtime's speculative
//! loop with the shared window `app::mtp::MtpDraft` (the shared draft file
//! beside the target or `BLOOMERY_MTP_DRAFT`'s, its head reduced under
//! `BLOOMERY_MTP_HEAD_ROWS`): windows of four rows, the greedy ids the plain
//! server's, `pass_rows` 4. A sampling or id-banning request takes plain
//! steps (the server's loop asks a pass only of a greedy request with no
//! banned id), each step's row the target's, read before the step is told
//! to the draft; its ids are the plain server's. A `load draft=mtp` line
//! follows the `load` line, and `/props`' `engine.draft` names the draft
//! with its resident bytes as the card's `draft` class. Every other word of
//! the lever is refused by name, and so are `BLOOMERY_MTP_HEAD_ROWS` and
//! `BLOOMERY_MTP_DRAFT` set on a server that drafts nothing, with why.
//!
//! `BLOOMERY_RESIDENCY` set prints as a `residency lever` record first
//! thing. Unset, the Qwen3.8 rule picks the word as in `generate_qwen3moe`
//! (`bloomery_levers::residency38_unset`, `residency38_at_plan`) and a
//! `residency unset` record after the `plan` line prints it with why: under
//! `--place a` `mid-p<P>-s1`, P half the fewest card experts a layer of the
//! plan the load runs (the plain or the MTP plan, at `--ctx-size`); `off`
//! under `--place gate`, when the plan holds no card expert or its fewest
//! leave no room, and when the churn pool does not fit the plan's host
//! headroom or what `MemAvailable` leaves past the plan's host need — never
//! a refusal. Running `mid-p<P>-s<S>` (plain or drafted, either `--place`),
//! the load runs the common residency machine over the card's routed stacks
//! (`Body38::open_placed_residency`, `open_placed_mtp_residency`): a
//! `residency host` record follows the `plan` line, and each call (a prompt,
//! a step, a pass) prints its boundaries' `residency pass` records after it.
//! A request's reset keeps the residency where use has taken it; `POST
//! /residency/reset` moves it back to its seed on a free slot and prints a
//! `residency reset` record (without the residency, the server's 501). The
//! seat's prompt path is `auto` — passes below nine ids, ubatches from nine
//! on, never one step an id — so the body's refusal of a step-fed prompt
//! beside the machine is never reached.
//!
//! An engine error ends the process: the request gets a 500, `/health` a 503
//! for a moment, then the crash block (card, position, error) goes to stderr
//! and the exit code is 70.
//!
//! The binary builds under the `deepseek41` feature, whose name is the
//! server surface's scoping (gpu-gates' `bind` and `serve_client`, the
//! `serve` and `sampler` crates), not a model: it runs no V4.1 code.
//!
//! The levers it acts on ([`ACTS_ON`]) are parsed once, at `main`
//! (`bloomery_levers::at_main`), which refuses by name a lever set outside
//! them and a `BLOOMERY_*` name no registry row names; `--levers` prints them
//! with this process's values and exits. The Qwen3.8 levers are
//! `BLOOMERY_QWEN38_EXPERTS` (the plan's expert rule) with
//! `BLOOMERY_CARD_BUDGET` bounding its card plan, the host
//! tier's load settings, `BLOOMERY_PIN_MAIN`, the draft's levers and
//! `BLOOMERY_RESIDENCY`; the ubatch size
//! (`BLOOMERY_QWEN3_UBATCH`) is read where the load sizes its arena. The
//! stderr lines named above are records of the kinds
//! `bloomery_gpu_gates::record` declares; `--records-schema` prints those
//! kinds and exits.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use app::mtp::MtpBody;
use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
use bloomery_gpu::arch::qwen3moe::{Body38, Prompt38};
use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::bind::{
    Seat, SeatEngine, Vocab, model_props, nvidia_smi_index, placement_props, sampler_factory,
};
use bloomery_gpu_gates::nodes::count_kinds;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::residency38::{CARD38, Lever38, residency38};
use bloomery_gpu_gates::{GateError, ref_model_path};
use bloomery_levers::{
    Draft38At, Draft38Off, Residency38At, ResidencyPick, ResidencyWhy, draft38_unset,
    residency38_unset,
};
use cuda_core::sys;
use gguf::Split;
use model::arch::models::HeadRows;
use model::arch::qwen35moe::place::{
    Experts, MtpInputs, PlanInputs, machine_for_experts, read_head_rows,
};
use model::placement::workstation::{A6000, CardSpec, RTX_3090};
use model::placement::{Plan, PlanLevers};
use refset::arch::qwen4exp::mtp::{DraftFrom, draft_file};
use runtime::Target as _;
use serve::{
    CacheNote, DraftProps, Drafted, EngineProps, FATAL_LINGER, ResidencyReset, Saved, ServeError,
    Server, ServerConfig,
};
use tokenizer::Tokenizer;

use super::drafted::DraftedSeat;

/// The levers `bloomery-serve-qwen38` acts on, for its own `main` and for
/// `gate_qwen38_serve`'s (the gate starts the server with its own
/// environment, so a lever the server would refuse is refused by the gate
/// first; the two lists must stay one).
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::QWEN38_EXPERTS,
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
    bloomery_levers::DRAFT,
    bloomery_levers::MTP_HEAD_ROWS,
    bloomery_levers::MTP_DRAFT,
    bloomery_levers::RESIDENCY,
];

const USAGE: &str = "usage: bloomery-serve-qwen38 [--host H] [--port P] [--place a|gate] \
                     [--ctx-size C] [--alias NAME] [--chat-template-file PATH]";

/// The positions the stores are sized for when `--ctx-size` names none:
/// `generate_qwen3moe`'s default.
const CTX: usize = 4096;

/// Why no prefix is kept: the recurrent layers hold no state for an
/// earlier position.
const KEEP_WHY: &str = "the recurrent state (the GDN lanes, the PLE hash history) keeps no \
                        prefix: every request prefills from a reset";

/// Where `--place` puts the plan's one card, as `generate_qwen3moe`
/// takes it.
#[derive(Clone, Copy)]
enum Place38 {
    /// The A6000, the timing card (the default).
    A,
    /// The 3090, the gate card.
    Gate,
}

impl Place38 {
    fn parse(v: &str) -> Result<Place38, GateError> {
        match v {
            "a" => Ok(Place38::A),
            "gate" => Ok(Place38::Gate),
            other => Err(format!("--place is a or gate, not {other}").into()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Place38::A => "a",
            Place38::Gate => "gate",
        }
    }

    fn spec(self) -> CardSpec {
        match self {
            Place38::A => A6000,
            Place38::Gate => RTX_3090,
        }
    }
}

/// The MTP draft file at `path`, which `from` picked; an error names both.
fn open_draft(path: &Path, from: DraftFrom) -> Result<Split, GateError> {
    Split::open(path).map_err(|e| {
        format!(
            "open the MTP draft {} ({}): {e}",
            path.display(),
            from.describe()
        )
        .into()
    })
}

/// `BLOOMERY_QWEN38_EXPERTS` as the plan's expert rule.
fn experts38(levers: &bloomery_levers::Levers) -> Result<Experts, GateError> {
    match levers.qwen38_experts() {
        "host" => Ok(Experts::Host),
        "card" => Ok(Experts::Card),
        other => Err(format!("BLOOMERY_QWEN38_EXPERTS={other}: host or card").into()),
    }
}

/// `BLOOMERY_DRAFT` on the seat at `place` with stores of `ctx` positions,
/// `file` the MTP draft file the load would open: `mtp` the draft, `off` the
/// plain path; unset, `generate_qwen3moe`'s rule
/// (`bloomery_levers::draft38_unset`); and, drafting nothing, why (the `load
/// draft=off` record's). The V4.1 words and any other are refused by name.
///
/// The rule's run conditions as the seat meets them: no `--logits` (a
/// request that reads the logits row steps plainly, the row the target's),
/// no route trace (the seat does not act on `BLOOMERY_ROUTE_TRACE`, so
/// `at_main` refuses it set), and the positions the server's loop needs for
/// one window — it takes one only while its rows fit, the first at a one-id
/// prompt's first generated token: 1 + 1 + rows − 1.
fn draft38(
    levers: &bloomery_levers::Levers,
    place: Place38,
    ctx: usize,
    file: &Path,
) -> Result<(bool, Option<Draft38Off>), GateError> {
    match levers.draft() {
        Some("mtp") => Ok((true, None)),
        Some("off") => Ok((false, Some(Draft38Off::Set))),
        Some(other) => Err(format!(
            "BLOOMERY_DRAFT={other}: on a qwen4exp file mtp drafts the window and off runs the \
             plain path; lookup and dspark are the V4.1 binaries'"
        )
        .into()),
        None => {
            let at = Draft38At {
                place_a: matches!(place, Place38::A),
                logits: false,
                route_trace: false,
                file,
                file_is_there: file.is_file(),
                need: <Body38 as MtpBody>::VERIFY_ROWS + 1,
                ctx,
            };
            Ok(match draft38_unset(&at) {
                None => (true, None),
                Some(off) => (false, Some(off)),
            })
        }
    }
}

/// The residency the load of `plan` at `place` runs: `set` (the word and
/// its parse) as given; unset, the Qwen3.8 rule's
/// ([`residency38`]: before the plan `off` under `--place gate`, else on plan
/// (a) from it), its records on stderr.
fn residency38_at(
    plan: &Plan<'_>,
    place: Place38,
    set: Option<(Residency, &str)>,
) -> Result<Residency, GateError> {
    let lever = match set {
        Some((r, word)) => Lever38::Set(r, word),
        None => {
            // The seat feeds no prompt by steps (its path is `auto`) and
            // takes no route trace.
            let at = Residency38At {
                qwen38_file: true,
                dump_taps: false,
                place_a: matches!(place, Place38::A),
                route_trace: false,
                prefill_step: false,
            };
            Lever38::Unset(residency38_unset(at))
        }
    };
    residency38(plan, lever, Record::eprint)
}

struct Args {
    host: String,
    port: u16,
    place: Place38,
    ctx: usize,
    alias: Option<String>,
    /// `--chat-template-file`, replacing the file's own template.
    template_file: Option<PathBuf>,
}

fn parse_args(args: &[String]) -> Result<Args, GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        place: Place38::A,
        ctx: CTX,
        alias: None,
        template_file: None,
    };
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
            "--place" => a.place = Place38::parse(v)?,
            "--ctx-size" | "--ctx" => a.ctx = v.parse()?,
            "--alias" => a.alias = Some(v.to_owned()),
            "--chat-template-file" => a.template_file = Some(PathBuf::from(v)),
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    if a.ctx == 0 {
        return Err("--ctx-size 0: the stores hold no position".into());
    }
    Ok(a)
}

/// Loads the model and serves until the listener or the engine fails;
/// `Ok` carries why the server ended.
pub fn run(args: &[String]) -> Result<ServeError, GateError> {
    let levers = bloomery_levers::at_main(ACTS_ON)?;
    record::at_main("bloomery-serve-qwen38", record::BLOOMERY_SERVE_QWEN38);
    // Set, the word runs as given; unset, the Qwen3.8 rule picks it once the
    // placement and the plan are known (`residency unset`).
    if let Some(word) = levers.residency() {
        record::residency_lever(ResidencyPick {
            word,
            why: ResidencyWhy::Set,
        })
        .eprint();
    }
    let set = match levers.residency() {
        Some(word) => Some((Residency::parse(word)?, word)),
        None => None,
    };
    let a = parse_args(args)?;
    let path = ref_model_path()?;
    let (draft_path, draft_from) = draft_file(levers.mtp_draft(), &path);
    let (mtp, draft_off) = draft38(&levers, a.place, a.ctx, &draft_path)?;
    let head_rows = levers.mtp_head_rows().map(PathBuf::from);
    // A draft lever set on a server that drafts nothing is refused, with why.
    if let Some(why) = &draft_off {
        if head_rows.is_some() {
            return Err(format!(
                "BLOOMERY_MTP_HEAD_ROWS reduces the MTP draft's head; the server drafts nothing \
                 ({why})"
            )
            .into());
        }
        if levers.mtp_draft().is_some() {
            return Err(format!(
                "BLOOMERY_MTP_DRAFT names the MTP draft file; the server drafts nothing ({why})"
            )
            .into());
        }
    }
    let experts = experts38(&levers)?;
    let plan_levers = PlanLevers::from_levers(&levers)?;
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
        .unwrap_or("qwen3.8")
        .to_owned();
    drop(inv);

    // The plan record, and `/props` from the same plan the load runs by:
    // under the draft `plan_mtp_with`'s, its draft's card bytes (granules,
    // store, row map) and its program's arena the card's `draft` class.
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let inputs = PlanInputs::describe(&split)?;
    let model = model_props(&split, &inputs.model);
    let ub = ubatch_for(a.ctx)?;
    let machine = machine_for_experts(
        a.place.spec(),
        inputs.spec.layers.len(),
        u64::try_from(ub)?,
        experts,
    );
    let mtp_inputs = match mtp {
        false => None,
        true => {
            let rows = match head_rows.as_deref() {
                Some(p) => read_head_rows(p, &split, inputs.spec.vocab)?,
                None => HeadRows::Full,
            };
            let draft = open_draft(&draft_path, draft_from)?;
            Some(MtpInputs::read(&draft, &split, &inputs, rows)?)
        }
    };
    drop(split);
    let (plan, draft_bytes) = match &mtp_inputs {
        None => (
            inputs.plan_with(&machine, u64::try_from(a.ctx)?, &plan_levers, experts)?,
            0,
        ),
        Some(mi) => {
            let with =
                inputs.plan_mtp_with(&machine, u64::try_from(a.ctx)?, &plan_levers, mi, experts)?;
            let bytes = with.draft_card_bytes() + with.arena_bytes;
            (with.plan, bytes)
        }
    };
    Record::new(&record::PLAN38)
        .w("place", a.place.name())
        .w("card", machine.cards[0].name.as_str())
        .w(
            "experts",
            if experts == Experts::Card {
                "card"
            } else {
                "host"
            },
        )
        .u("ctx_max", plan.ctx_max)
        .u("host_experts", plan.host.experts)
        .u("card_experts", plan.cards[0].experts)
        .eprint();
    let residency = residency38_at(&plan, a.place, set)?;
    let gpu = nvidia_smi_index(&machine.cards[0].name)
        .map(|i| format!("GPU{i}"))
        .and_then(|g| placement_props(&plan, &[g]));
    if let Err(e) = &gpu {
        eprintln!("bloomery-serve-qwen38: /props leaves the placement out: {e}");
    }
    let props = EngineProps {
        model: Some(model),
        placement: gpu.ok(),
        ..EngineProps::default()
    };

    let open = SeatArgs {
        place: a.place,
        ctx: a.ctx,
        experts,
        plan_levers,
        host: levers.host(),
        pin_main: levers.pin_main(),
        path: path.clone(),
        mtp,
        head_rows,
        draft_path,
        draft_from,
        draft_bytes,
        draft_off,
        residency,
    };
    // The prompt cache is off (its budget 0): the seat keeps no prefix
    // worth saving, and slot save/restore is the server's own 501.
    let engine = SeatEngine::spawn(
        move || Q38::open(open),
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
    Record::new(&record::LISTENING38)
        .w("place", a.place.name())
        .u("ctx", a.ctx)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}

/// What the engine thread opens the seat with.
struct SeatArgs {
    place: Place38,
    ctx: usize,
    experts: Experts,
    plan_levers: PlanLevers,
    host: bloomery_levers::HostCfg,
    pin_main: bool,
    path: PathBuf,
    /// `BLOOMERY_DRAFT=mtp`, the MTP draft; unset the plain path.
    mtp: bool,
    /// `BLOOMERY_MTP_HEAD_ROWS`, the draft head's row list.
    head_rows: Option<std::path::PathBuf>,
    /// The MTP draft file under `mtp`, and what picked it.
    draft_path: PathBuf,
    draft_from: DraftFrom,
    /// The plan's `draft` class bytes, which `/props` files.
    draft_bytes: u64,
    /// Drafting nothing, why: the `load draft=off` record's.
    draft_off: Option<Draft38Off>,
    /// The residency the load runs ([`residency38`]).
    residency: Residency,
}

/// The Qwen3.8 session on the engine thread: the session over the model,
/// the draft it drives when one runs (its windows of four rows through
/// the runtime's speculative loop), and the positions its stores were
/// sized for.
struct Q38 {
    s: app::Session<Body38>,
    drafted: DraftedSeat<Body38, { <Body38 as MtpBody>::VERIFY_ROWS }>,
    ctx: usize,
    /// The plan's draft card bytes and arena, for `/props`' `draft` class.
    draft_bytes: u64,
    /// The MTP draft file, which `/props`' `draft` names.
    draft_path: PathBuf,
    /// The load runs the residency machine: each call prints its
    /// boundaries' `residency pass` records.
    residency: bool,
}

impl Q38 {
    /// The session by `a.place` (the `load` and `capture` lines, as
    /// `generate_qwen3moe` prints them; under `mtp` the draft loaded
    /// beside the target, its own `load draft=mtp` line and the verify
    /// passes' capture lines after them), on the calling thread, pinned
    /// to the dispatcher's cpu slot when asked.
    fn open(a: SeatArgs) -> Result<Q38, GateError> {
        const WHAT: &str = "bloomery-serve-qwen38";
        if a.pin_main {
            // The engine thread runs every step; give it the pool's
            // dispatcher slot.
            let _ = threads::pool().pin_caller();
        }
        let t = Instant::now();
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        let inputs = PlanInputs::describe(&file)?;
        let ub = ubatch_for(a.ctx)?;
        let machine = machine_for_experts(
            a.place.spec(),
            inputs.spec.layers.len(),
            u64::try_from(ub)?,
            a.experts,
        );
        let mut m = match a.mtp {
            false => {
                let plan =
                    inputs.plan_with(&machine, u64::try_from(a.ctx)?, &a.plan_levers, a.experts)?;
                Body38::open_placed_residency(
                    file,
                    &plan,
                    &inputs,
                    CARD38,
                    a.host,
                    ub,
                    a.residency,
                )?
            }
            true => {
                let rows = match a.head_rows.as_deref() {
                    Some(p) => read_head_rows(p, &file, inputs.spec.vocab)?,
                    None => HeadRows::Full,
                };
                let draft = open_draft(&a.draft_path, a.draft_from)?;
                let mtp = MtpInputs::read(&draft, &file, &inputs, rows)?;
                let plan = inputs.plan_mtp_with(
                    &machine,
                    u64::try_from(a.ctx)?,
                    &a.plan_levers,
                    &mtp,
                    a.experts,
                )?;
                Body38::open_placed_mtp_residency(
                    file,
                    &plan,
                    &inputs,
                    CARD38,
                    a.host,
                    ub,
                    &draft,
                    &mtp,
                    a.residency,
                )?
            }
        };
        m.set_mode(StepMode::Graph);
        eprintln!(
            "load arch=qwen4exp resident_bytes={} ctx={} layers={} mode=graph store_bytes={} \
             prefill=auto ubatch={} place={} card_layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            a.ctx,
            m.layers().len(),
            m.body(WHAT)?.store_bytes(),
            m.body(WHAT)?.ubatch_rows(),
            a.place.name(),
            m.body(WHAT)?.card_layers(),
            t.elapsed().as_secs_f64()
        );
        if let Some(why) = &a.draft_off {
            Record::new(&record::LOAD_DRAFT_OFF38)
                .w("why", why)
                .eprint();
        }
        if a.mtp {
            let d = m.body(WHAT)?.mtp().ok_or("the load opened no MTP draft")?;
            let bytes = (d.resident_bytes() + d.arena_bytes()) as u64;
            let head = match d.head_map() {
                Some((_, n)) => format!("rows={n}"),
                None => "full".to_string(),
            };
            eprintln!(
                "load draft=mtp resident={} arena={} head={head} card_bytes={bytes} (the \
                 plan's {}) in {:.1} s (runtime value)",
                d.resident_bytes(),
                d.arena_bytes(),
                a.draft_bytes,
                t.elapsed().as_secs_f64(),
            );
        }
        let (launches, memops) = m.body(WHAT)?.step_launches();
        let nodes = m.capture_step()?;
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memop = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
        let ([k, b], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memop]);
        eprintln!(
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
        let mut s = app::Session::from_model(m, u32::try_from(a.ctx)?);
        let residency = a.residency != Residency::Off;
        if residency {
            // A request's passes are not known at load: the log grows as it
            // must, and every call takes it.
            s.model_mut().body_parts(WHAT)?.2.log_residency(0);
        }
        let drafted = DraftedSeat::new(match a.mtp {
            false => None,
            true => {
                let draft = app::mtp::MtpDraft::open(s.model(), Prompt38::Auto, StepMode::Graph)?;
                struct Captures;
                impl app::RowsLog for Captures {
                    fn capture_rows(
                        &mut self,
                        rows: usize,
                        nodes: usize,
                    ) -> Result<(), app::SessionError> {
                        eprintln!("capture verify rows={rows} nodes={nodes}");
                        Ok(())
                    }
                }
                Some(s.with_draft(draft, &mut Captures)?)
            }
        });
        Ok(Q38 {
            s,
            drafted,
            ctx: a.ctx,
            draft_bytes: a.draft_bytes,
            draft_path: a.draft_path,
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
            .body_parts("bloomery-serve-qwen38")?
            .2
            .take_residency_passes()
        {
            record::residency_pass_of(kind, &r).eprint();
        }
        Ok(())
    }
}

impl Seat for Q38 {
    fn pos(&self) -> usize {
        self.s.pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The prompt through the ubatch walk `--prefill auto` takes: `gemm`
    /// from nine positions on, `pass` below — never one step per id;
    /// under the draft the draft's own prompt call, its store walked over
    /// the prompt's units.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        let next = self.drafted.prefill(&mut self.s, ids)?;
        self.print_passes()?;
        Ok(next)
    }

    /// One step; under the draft the rows it left waiting walked first
    /// (`MtpDraft::before_step`: a request that continues the held
    /// sequence joins it here when its prompt call is empty).
    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        let next = self.drafted.step(&mut self.s, last)?;
        self.print_passes()?;
        Ok(next)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    /// One step and the target's row of it, read before the step is told
    /// to the draft ([`DraftedSeat::step_with_row`]).
    fn step_row(&mut self, last: u32, row: &mut [f32]) -> Result<u32, GateError> {
        let next = self.drafted.step_with_row(&mut self.s, last, row)?;
        self.print_passes()?;
        Ok(next)
    }

    /// The session's reset: the residency stays where use has taken it
    /// (only [`Seat::residency_reset`] moves it back).
    fn reset(&mut self) -> Result<(), GateError> {
        self.drafted.reset(&mut self.s)
    }

    /// [`app::Session::residency_reset`], its `residency reset` record on
    /// stderr; `None` without the residency (the server's 501).
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

    /// One pass from `last`: under the draft the window of four rows,
    /// its kept tokens and counts; without it one step.
    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        let d = self.drafted.pass(&mut self.s, last, out)?;
        self.print_passes()?;
        Ok(d)
    }

    /// The most positions one pass runs: the draft's four rows, or one
    /// step without it.
    fn pass_rows(&self) -> usize {
        self.drafted.pass_rows()
    }

    /// The commit's rule, `Body38::kept`: with no verify waiting only
    /// the position the stores stand at is kept — nothing to take back —
    /// and a request's shorter prefix grants nothing. [`KEEP_WHY`] is
    /// the rule string the server's reuse records print.
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        let pos = self.s.pos() as usize;
        if n >= pos {
            return (pos, None);
        }
        (0, Some(KEEP_WHY.to_owned()))
    }

    /// The body's commit, which only a waiting verify serves; a request
    /// path never reaches it (the server resets instead of cutting), so
    /// any other position is refused by name, the body's own message.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        Ok(self.s.model_mut().rollback(pos)?)
    }

    /// Nowhere: the engine cuts a prompt call only where a later request
    /// can keep the cut, and this engine keeps no cut.
    fn splits(&self, _: usize, _: usize, _: &[usize]) -> Vec<usize> {
        Vec::new()
    }

    /// `/props`' `engine.draft` under the MTP draft: its kind, the draft
    /// file, its width and its resident bytes as the card's `draft`
    /// class.
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
            model: self.draft_path.file_name().map_or_else(
                || self.draft_path.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            ),
            n_max: Some(<Body38 as MtpBody>::WIDTH as u64),
            kind: Some("mtp".to_owned()),
            path: Some(self.draft_path.display().to_string()),
            device: None,
        });
        p
    }

    /// A Qwen3.8 sequence state is not a value a cache could hold: the
    /// recurrent stores and the PLE history have no copy to put back.
    /// The server's prompt cache is off, so this is never asked; a caller
    /// that asks is refused by name.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        Err(format!("a snapshot at position {}: {KEEP_WHY}", self.s.pos()).into())
    }

    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError> {
        Err(format!(
            "a resume of a state of {} positions: {KEEP_WHY}",
            state.n_tokens()
        )
        .into())
    }

    /// Only `Reuse` arrives (the cache is off and no call is cut); the
    /// other notes belong to a caching engine and print as they come.
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
        eprintln!("bloomery-serve-qwen38: {note}");
    }
}
